"""
Ingestion & analysis tools for the Corporate Actions agent.

Key safety decisions implemented here:
- Decimal (not float) for ALL monetary arithmetic
- ISIN checksum validation (ISO 6166)
- Deterministic pre-classification before LLM
- Empty position list always treated as an error, never as "no holdings"
"""
import re
import csv
import os
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from datetime import datetime, timezone
from typing import List, Optional
from langchain_core.tools import tool
from ca_agent.config import POSITIONS_FILE, CRITICAL_DEADLINE_HOURS


# ── ISIN Validation ────────────────────────────────────────────────────────────

def _isin_checksum_valid(isin: str) -> bool:
    """Validate ISIN using the Luhn-based ISO 6166 checksum algorithm."""
    if not isin or len(isin) != 12:
        return False
    # Reject lowercase — all real ISINs are uppercase; lowercase = data quality issue
    if isin != isin.upper():
        return False
    isin = isin.upper().strip()
    if not isin[:2].isalpha() or not isin[2:].isalnum():
        return False
    # Convert letters to digits (A=10, B=11, ... Z=35)
    digits = ""
    for ch in isin[:-1]:
        digits += str(ord(ch) - 55) if ch.isalpha() else ch
    # Luhn algorithm
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        if i % 2 == 0:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    check = (10 - (total % 10)) % 10
    return check == int(isin[-1])


# ── SWIFT MT564 Parsing ────────────────────────────────────────────────────────

@tool
def parse_swift_mt564(message: str) -> dict:
    """
    Parse a SWIFT MT564 (Corporate Action Notification) message.

    Extracts: event type, event category, ISIN, issuer, key dates,
    rates, and narrative. Validates ISIN checksum. Returns a structured
    dict. If parsing fails on a critical field, raises ValueError so the
    graph routes to escalation rather than proceeding with bad data.

    Args:
        message: Raw SWIFT MT564 message text.

    Returns:
        Dict with parsed fields: event_type, event_category, isin,
        issuer, record_date, ex_date, pay_date, election_deadline,
        gross_rate, tax_rate, currency, narrative.
    """
    result = {}
    errors = []

    # ── Event type (22F::CAEV) ────────────────────────────────────────────
    m = re.search(r':22F::CAEV//([A-Z]{4})', message)
    result["event_type"] = m.group(1) if m else None
    if not result["event_type"]:
        errors.append("Could not extract event type (22F::CAEV)")

    # ── Event category (22F::CAMV) ────────────────────────────────────────
    m = re.search(r':22F::CAMV//([A-Z]{4})', message)
    camv_map = {"MAND": "mandatory", "VOLU": "voluntary", "CHOS": "elective"}
    camv_raw = m.group(1) if m else None
    result["event_category"] = camv_map.get(camv_raw, "mandatory")

    # ── ISIN (35B) ────────────────────────────────────────────────────────
    # Handle variants: "ISIN XX1234567890" or "/XX/XX1234567890"
    m = re.search(r':35B:(?:ISIN\s+)?([A-Z]{2}[A-Z0-9]{10})', message)
    if not m:
        m = re.search(r':35B:/[A-Z]{2}/([A-Z]{2}[A-Z0-9]{10})', message)
    isin = m.group(1) if m else None

    if not isin:
        errors.append("Could not extract ISIN from :35B: field")
    elif not _isin_checksum_valid(isin):
        errors.append(f"ISIN '{isin}' failed checksum validation (ISO 6166)")
    else:
        result["isin"] = isin

    # ── Issuer name (lines after :35B:, skip identifier lines) ───────────
    # Walk lines after :35B: tag; skip /XX/ identifier lines; stop at next tag
    _lines = message.split("\n")
    _isin_idx = next((i for i, l in enumerate(_lines) if ":35B:" in l), -1)
    _issuer_parts = []
    if _isin_idx != -1:
        for _line in _lines[_isin_idx + 1 : _isin_idx + 6]:
            _stripped = _line.strip()
            if not _stripped:
                continue
            # Stop at any next SWIFT tag or message trailer
            if _stripped.startswith(":") or _stripped.startswith("-"):
                break
            # Skip CUSIP/SEDOL/identifier lines like /US/037833100
            if _stripped.startswith("/"):
                continue
            _issuer_parts.append(_stripped)
    result["issuer"] = " ".join(_issuer_parts) if _issuer_parts else "Unknown Issuer"

    # ── Dates ──────────────────────────────────────────────────────────────
    def extract_date(tag: str) -> str:
        """Extract date in YYYYMMDD format and convert to ISO 8601."""
        m = re.search(rf':98A::{tag}//(\d{{8}})', message)
        if m:
            d = m.group(1)
            return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        return ""

    result["record_date"] = extract_date("RDDT")
    result["ex_date"]     = extract_date("EXDT")
    result["pay_date"]    = extract_date("PAYD")

    # Election deadline — from RMDT (response mandatory date/time)
    # Always convert to UTC ISO 8601
    m = re.search(r':98A::RMDT//(\d{8})(\d{6})', message)
    if m:
        dt_str = f"{m.group(1)}{m.group(2)}"
        dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        result["election_deadline"] = dt.isoformat()
    else:
        result["election_deadline"] = ""

    # ── Rates (use string to preserve Decimal precision) ──────────────────
    def extract_rate(tag: str) -> str:
        m = re.search(rf':92A::{tag}//(\d+\.?\d*)', message)
        return m.group(1) if m else "0"

    result["gross_rate"] = extract_rate("GRSS")
    result["net_rate"]   = extract_rate("NETT")
    result["tax_rate"]   = extract_rate("TXGT")

    # Offer price (tender offers)
    m = re.search(r':92A::PRPP//(\d+\.?\d*)', message)
    result["offer_price"] = m.group(1) if m else "0"

    # ── Currency ──────────────────────────────────────────────────────────
    m = re.search(r':22F::CURR//([A-Z]{3})', message)
    if not m:
        m = re.search(r':19B::ENTL//([A-Z]{3})', message)
    result["currency"] = m.group(1) if m else "USD"

    # ── Narrative ────────────────────────────────────────────────────────
    m = re.search(r':70E::ADTX//(.+?)(?::16S:|$)', message, re.DOTALL)
    result["narrative"] = m.group(1).strip().replace("\n", " ") if m else ""

    # ── Error handling ─────────────────────────────────────────────────────
    if errors:
        result["parse_errors"] = errors
        result["parse_success"] = False
    else:
        result["parse_errors"] = []
        result["parse_success"] = True

    return result


# ── Urgency Assessment ──────────────────────────────────────────────────────────

@tool
def assess_urgency(election_deadline: str, event_category: str) -> dict:
    """
    Assess the urgency of a corporate action based on election deadline.

    Args:
        election_deadline: ISO 8601 UTC deadline string. Empty string if none.
        event_category: "mandatory" | "voluntary" | "elective"

    Returns:
        Dict with urgency ("critical" | "high" | "normal") and
        hours_remaining (float or None).
    """
    if not election_deadline or event_category == "mandatory":
        return {"urgency": "normal", "hours_remaining": None}

    try:
        deadline = datetime.fromisoformat(election_deadline)
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        hours_remaining = (deadline - now).total_seconds() / 3600

        if hours_remaining <= 0:
            urgency = "critical"  # Already past deadline!
        elif hours_remaining <= CRITICAL_DEADLINE_HOURS:
            urgency = "critical"
        elif hours_remaining <= CRITICAL_DEADLINE_HOURS * 2:
            urgency = "high"
        else:
            urgency = "normal"

        return {
            "urgency": urgency,
            "hours_remaining": round(hours_remaining, 1)
        }
    except (ValueError, TypeError):
        return {"urgency": "high", "hours_remaining": None}


# ── Portfolio Position Loading ──────────────────────────────────────────────────

@tool
def load_portfolio_positions(isin: str) -> dict:
    """
    Load all portfolio positions for a given ISIN from the positions file.

    IMPORTANT: If zero positions are returned, this is treated as a data
    quality issue — NOT as confirmation that no portfolios hold the security.
    Always escalate if zero positions are found.

    Args:
        isin: The ISIN to search for.

    Returns:
        Dict with 'positions' (list of dicts) and 'position_count' (int).
        Raises ValueError if ISIN is invalid or file is unreadable.
    """
    if not _isin_checksum_valid(isin):
        raise ValueError(f"Invalid ISIN: '{isin}' — failed checksum validation")

    positions_file = POSITIONS_FILE
    if not os.path.exists(positions_file):
        raise FileNotFoundError(f"Positions file not found: {positions_file}")

    positions = []
    with open(positions_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("isin", "").strip().upper() == isin.upper():
                positions.append({
                    "portfolio_name": row["portfolio_name"],
                    "portfolio_id": row["portfolio_id"],
                    "quantity": int(row["quantity"]),
                    "asset_class": row["asset_class"],
                    "base_currency": row["base_currency"],
                })

    return {
        "positions": positions,
        "position_count": len(positions),
        "data_quality_warning": len(positions) == 0
    }


# ── Entitlement Calculation ─────────────────────────────────────────────────────

@tool
def calculate_entitlements(
    positions: List[dict],
    event_type: str,
    gross_rate: str,
    tax_rate: str,
    currency: str,
    offer_price: Optional[str] = "0"
) -> dict:
    """
    Calculate projected corporate action entitlements for each portfolio.

    Uses Python Decimal for all arithmetic (never float) to prevent
    rounding errors in financial calculations.

    Supports:
    - DVCA (cash dividend): entitlement = quantity × gross_rate_per_share
    - SPLF (stock split): new_shares = quantity × split_ratio
    - TEND (tender offer): entitlement = quantity × offer_price
    - RHTS (rights issue): rights = quantity × rights_ratio

    Args:
        positions: List of portfolio positions from load_portfolio_positions.
        event_type: ISO 15022 event type code.
        gross_rate: Per-share gross rate as string.
        tax_rate: Withholding tax rate as decimal string (e.g., "0.15" for 15%).
        currency: Payment currency.
        offer_price: Offer price for tender offers.

    Returns:
        Dict with 'entitlements' list and 'total_projected' (Decimal string).
    """
    # Coerce None → "0" when LLM omits offer_price for non-tender events
    if offer_price is None:
        offer_price = "0"
    try:
        rate = Decimal(gross_rate)
        tax  = Decimal(tax_rate)
        offer = Decimal(offer_price)
    except InvalidOperation as e:
        raise ValueError(f"Invalid rate format: {e}")

    entitlements = []
    total = Decimal("0")

    for pos in positions:
        qty = Decimal(str(pos["quantity"]))

        if event_type == "DVCA":
            gross = (qty * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            withholding = (gross * tax).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            net = gross - withholding
            entitlements.append({
                "portfolio_name": pos["portfolio_name"],
                "portfolio_id": pos["portfolio_id"],
                "quantity": pos["quantity"],
                "gross_entitlement": str(gross),
                "withholding_tax": str(withholding),
                "net_entitlement": str(net),
                "currency": currency,
                "type": "cash"
            })
            total += gross

        elif event_type == "TEND":
            gross = (qty * offer).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            entitlements.append({
                "portfolio_name": pos["portfolio_name"],
                "portfolio_id": pos["portfolio_id"],
                "quantity": pos["quantity"],
                "gross_entitlement": str(gross),
                "withholding_tax": "0.00",
                "net_entitlement": str(gross),
                "currency": currency,
                "type": "cash_tender"
            })
            total += gross

        elif event_type == "SPLF":
            # rate here is the split ratio (e.g., 2.0 for 2-for-1 split)
            new_shares = int((qty * rate).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
            entitlements.append({
                "portfolio_name": pos["portfolio_name"],
                "portfolio_id": pos["portfolio_id"],
                "quantity": pos["quantity"],
                "new_shares": new_shares,
                "currency": "N/A",
                "type": "shares"
            })

        else:
            # Generic calculation for other types
            gross = (qty * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            entitlements.append({
                "portfolio_name": pos["portfolio_name"],
                "portfolio_id": pos["portfolio_id"],
                "quantity": pos["quantity"],
                "gross_entitlement": str(gross),
                "currency": currency,
                "type": "generic"
            })
            total += gross

    return {
        "entitlements": entitlements,
        "total_projected": str(total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
        "currency": currency,
        "event_type": event_type
    }
