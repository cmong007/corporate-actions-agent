"""
Reconciliation tools — the core control and risk mitigation layer.

Compares projected entitlements vs actual confirmations (MT566).
Detects and classifies breaks. Generates structured exception reports.
"""
import re
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from datetime import datetime, timezone
from typing import List
from langchain_core.tools import tool
from ca_agent.config import BREAK_THRESHOLD


# ── MT566 Parsing ───────────────────────────────────────────────────────────────

@tool
def parse_swift_mt566(message: str) -> dict:
    """
    Parse a SWIFT MT566 (Corporate Action Confirmation) message.

    Extracts the confirmed entitlement amounts received from the custodian.
    These are compared against projected amounts to detect breaks.

    Args:
        message: Raw SWIFT MT566 message text.

    Returns:
        Dict with confirmed amounts: entl_amount, gross_amount,
        net_amount, tax_amount, currency, settlement_date, event_ref.
    """
    result = {}

    # Event reference
    m = re.search(r':20C::CORP//(.+)', message)
    result["event_ref"] = m.group(1).strip() if m else "UNKNOWN"

    # Settlement date
    m = re.search(r':98C::SETT//(\d{8})', message)
    if m:
        d = m.group(1)
        result["settlement_date"] = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    else:
        result["settlement_date"] = ""

    # Financial amounts — extract as strings for Decimal safety
    def extract_amount(tag: str) -> str:
        m = re.search(rf':19B::{tag}//[A-Z]{{3}}(\d+\.?\d*)', message)
        return m.group(1) if m else "0"

    def extract_currency(message: str) -> str:
        m = re.search(r':19B::ENTL//([A-Z]{3})', message)
        return m.group(1) if m else "USD"

    result["entl_amount"] = extract_amount("ENTL")
    result["gross_amount"] = extract_amount("GRSS")
    result["net_amount"]   = extract_amount("NETT")
    result["tax_amount"]   = extract_amount("TXGT")
    result["currency"]     = extract_currency(message)

    # Narrative
    m = re.search(r':70E::ADTX//(.+?)(?::16S:|$)', message, re.DOTALL)
    result["narrative"] = m.group(1).strip().replace("\n", " ") if m else ""

    result["parse_success"] = True
    return result


# ── Break Detection ─────────────────────────────────────────────────────────────

@tool
def compare_entitlements(
    projected_entitlements: List[dict],
    actual_gross: str,
    actual_net: str,
    actual_tax: str,
    currency: str
) -> dict:
    """
    Compare projected entitlements against actual confirmed amounts.

    Uses Decimal arithmetic throughout. Classifies the likely cause
    of any break found. Flags breaks exceeding the configured threshold.

    Args:
        projected_entitlements: List from calculate_entitlements tool.
        actual_gross: Confirmed gross amount from MT566 (string).
        actual_net: Confirmed net amount from MT566 (string).
        actual_tax: Confirmed tax amount from MT566 (string).
        currency: Payment currency.

    Returns:
        Dict with breaks list, recon_status, max_break_amount,
        and break_classification.
    """
    try:
        act_gross = Decimal(actual_gross)
        act_net   = Decimal(actual_net)
        act_tax   = Decimal(actual_tax)
    except InvalidOperation as e:
        raise ValueError(f"Invalid amount in MT566: {e}")

    # Sum projected amounts
    total_proj_gross = Decimal("0")
    total_proj_tax   = Decimal("0")
    total_proj_net   = Decimal("0")

    for e in projected_entitlements:
        if "gross_entitlement" in e:
            total_proj_gross += Decimal(e["gross_entitlement"])
        if "withholding_tax" in e:
            total_proj_tax += Decimal(e["withholding_tax"])
        if "net_entitlement" in e:
            total_proj_net += Decimal(e["net_entitlement"])

    # Calculate breaks
    gross_break = (act_gross - total_proj_gross).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    tax_break = (act_tax - total_proj_tax).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    net_break = (act_net - total_proj_net).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )

    breaks = []
    max_break = Decimal("0")

    for tag, amount, proj in [
        ("GROSS", gross_break, total_proj_gross),
        ("TAX",   tax_break,   total_proj_tax),
        ("NET",   net_break,   total_proj_net),
    ]:
        abs_break = abs(amount)
        if abs_break > Decimal("0"):
            pct = (abs_break / proj * 100).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ) if proj != Decimal("0") else Decimal("100")

            # Classify likely cause
            cause = _classify_break_cause(tag, amount, proj)

            breaks.append({
                "break_type": tag,
                "projected": str(proj),
                "actual": str(proj + amount),
                "break_amount": str(amount),
                "break_amount_abs": str(abs_break),
                "break_pct": str(pct),
                "currency": currency,
                "likely_cause": cause,
                "requires_escalation": float(abs_break) > BREAK_THRESHOLD
            })
            if abs_break > max_break:
                max_break = abs_break

    return {
        "breaks": breaks,
        "recon_status": "breaks_found" if breaks else "clean",
        "max_break_amount": float(max_break),
        "summary": {
            "projected_gross": str(total_proj_gross),
            "actual_gross": str(act_gross),
            "projected_net": str(total_proj_net),
            "actual_net": str(act_net),
        }
    }


def _classify_break_cause(break_type: str, amount: Decimal, projected: Decimal) -> str:
    """Heuristic classification of break cause."""
    abs_amount = abs(amount)

    if break_type == "TAX":
        return (
            "Withholding tax rate mismatch — custodian applied different rate. "
            "Check account tax treaty classification."
        )
    if break_type == "GROSS" and abs_amount < Decimal("10"):
        return "Minor rounding difference — likely FX rounding or lot size rounding."
    if break_type == "GROSS" and amount < Decimal("0"):
        return (
            "Actual gross below projected — possible position discrepancy. "
            "Verify custodian position matches internal record date holdings."
        )
    if break_type == "NET":
        pct = abs_amount / projected * 100 if projected else Decimal("100")
        if pct > Decimal("5"):
            return "Significant net break — may indicate incorrect tax rate or position error."
        return "Net break driven by tax difference — investigate tax classification."

    return "Unknown — manual investigation required."


# ── Exception Report Generation ─────────────────────────────────────────────────

@tool
def generate_recon_report(
    breaks: List[dict],
    event_type: str,
    isin: str,
    issuer: str,
    pay_date: str,
    affected_portfolios: List[str]
) -> str:
    """
    Generate a structured reconciliation exception report.

    Produces output in the same format a senior operations analyst would
    write — suitable for distribution to Senior Management or custodians.

    Args:
        breaks: List of break dicts from compare_entitlements.
        event_type: ISO event type code.
        isin: Security ISIN.
        issuer: Issuer name.
        pay_date: Payment/settlement date.
        affected_portfolios: Portfolio names affected.

    Returns:
        Formatted string report.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    portfolio_names = ", ".join(
        p.get("portfolio_name", p) if isinstance(p, dict) else str(p)
        for p in affected_portfolios
    )

    if not breaks:
        return (
            f"RECONCILIATION REPORT — {timestamp}\n"
            f"{'='*60}\n"
            f"Event Type : {event_type}\n"
            f"ISIN       : {isin}\n"
            f"Issuer     : {issuer}\n"
            f"Pay Date   : {pay_date}\n"
            f"Portfolios : {portfolio_names}\n"
            f"{'='*60}\n"
            f"STATUS: ✅ CLEAN — No breaks detected.\n"
            f"All entitlements reconcile to custodian confirmation.\n"
        )

    lines = [
        f"RECONCILIATION EXCEPTION REPORT — {timestamp}",
        "=" * 60,
        f"Event Type : {event_type}",
        f"ISIN       : {isin}",
        f"Issuer     : {issuer}",
        f"Pay Date   : {pay_date}",
        f"Portfolios : {portfolio_names}",
        "=" * 60,
        f"STATUS: ⚠️  BREAKS FOUND — {len(breaks)} exception(s)",
        "",
        "BREAK DETAIL:",
        "-" * 40,
    ]

    for b in breaks:
        escalation_flag = " ← ESCALATION REQUIRED" if b.get("requires_escalation") else ""
        lines += [
            f"Break Type    : {b['break_type']}",
            f"Projected     : {b['currency']} {b['projected']}",
            f"Actual        : {b['currency']} {b['actual']}",
            f"Difference    : {b['currency']} {b['break_amount']} ({b['break_pct']}%){escalation_flag}",
            f"Likely Cause  : {b['likely_cause']}",
            f"Action        : Contact custodian for written confirmation of amount applied.",
            "-" * 40,
        ]

    lines += [
        "",
        "NEXT STEPS:",
        "1. Contact custodian operations for written clarification.",
        "2. Obtain amended MT566 if custodian confirms error.",
        "3. Adjust accounting system entries upon written confirmation.",
        "4. Document resolution and close exception within 2 business days.",
    ]

    return "\n".join(lines)
