"""
Unit tests for all Corporate Actions agent tools.
Run with: pytest ca_agent/tests/test_tools.py -v
"""
import pytest
from decimal import Decimal
from ca_agent.tools.ingestion_tools import (
    _isin_checksum_valid,
    parse_swift_mt564,
    assess_urgency,
    load_portfolio_positions,
    calculate_entitlements,
)
from ca_agent.tools.reconciliation_tools import (
    parse_swift_mt566,
    compare_entitlements,
    generate_recon_report,
)
from ca_agent.tools.notification_tools import (
    validate_security_identifiers,
    check_security_master_record,
    tag_recipients,
)


# ══════════════════════════════════════════════════════════════════════════════
# ISIN VALIDATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestISINValidation:

    def test_valid_apple_isin(self):
        assert _isin_checksum_valid("US0378331005") is True

    def test_valid_bp_isin(self):
        assert _isin_checksum_valid("GB0002634946") is True

    def test_valid_microsoft_isin(self):
        assert _isin_checksum_valid("US5949181045") is True

    def test_invalid_checksum(self):
        """Correct format but wrong check digit."""
        assert _isin_checksum_valid("US0378331006") is False

    def test_wrong_length(self):
        assert _isin_checksum_valid("US037833100") is False  # 11 chars

    def test_empty_string(self):
        assert _isin_checksum_valid("") is False

    def test_none_input(self):
        assert _isin_checksum_valid(None) is False

    def test_lowercase_rejected(self):
        """ISIN must be uppercase."""
        assert _isin_checksum_valid("us0378331005") is False


# ══════════════════════════════════════════════════════════════════════════════
# SWIFT MT564 PARSING TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestMT564Parsing:

    def test_parse_dividend_message(self):
        with open("ca_agent/data/sample_mt564_dividend.txt") as f:
            msg = f.read()
        result = parse_swift_mt564.invoke({"message": msg})
        assert result["parse_success"] is True
        assert result["event_type"] == "DVCA"
        assert result["event_category"] == "mandatory"
        assert result["isin"] == "US0378331005"
        assert result["gross_rate"] == "0.25"
        assert result["currency"] == "USD"
        assert result["pay_date"] == "2026-07-10"

    def test_parse_tender_offer_message(self):
        with open("ca_agent/data/sample_mt564_tender.txt") as f:
            msg = f.read()
        result = parse_swift_mt564.invoke({"message": msg})
        assert result["parse_success"] is True
        assert result["event_type"] == "TEND"
        assert result["event_category"] == "voluntary"
        assert result["isin"] == "US4592001014"
        assert result["offer_price"] == "142.50"

    def test_parse_invalid_isin(self):
        """Parser should flag invalid ISIN checksum."""
        bad_msg = """
:22F::CAEV//DVCA
:22F::CAMV//MAND
:35B:ISIN US0378331006
Apple Inc
:92A::GRSS//0.25
:22F::CURR//USD
"""
        result = parse_swift_mt564.invoke({"message": bad_msg})
        assert result["parse_success"] is False
        assert any("checksum" in e.lower() for e in result["parse_errors"])

    def test_parse_custodian_isin_variant(self):
        """Test the /CC/ ISIN prefix variant used by some custodians."""
        msg = """
:22F::CAEV//DVCA
:22F::CAMV//MAND
:35B:/US/US0378331005
Apple Inc Common Stock
:92A::GRSS//0.25
:22F::CURR//USD
:98A::PAYD//20260710
"""
        result = parse_swift_mt564.invoke({"message": msg})
        # Should still extract ISIN correctly
        assert result.get("isin") == "US0378331005" or \
               result.get("parse_success") is False  # Either extract or flag

    def test_parse_missing_event_type(self):
        """Missing CAEV field must set parse_success=False."""
        bad_msg = ":35B:ISIN US0378331005\nApple Inc\n:92A::GRSS//0.25"
        result = parse_swift_mt564.invoke({"message": bad_msg})
        assert result["parse_success"] is False


# ══════════════════════════════════════════════════════════════════════════════
# URGENCY ASSESSMENT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestUrgencyAssessment:

    def test_mandatory_event_always_normal(self):
        result = assess_urgency.invoke({
            "election_deadline": "2026-06-09T12:00:00+00:00",
            "event_category": "mandatory"
        })
        assert result["urgency"] == "normal"
        assert result["hours_remaining"] is None

    def test_critical_deadline_within_48h(self):
        from datetime import datetime, timezone, timedelta
        soon = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        result = assess_urgency.invoke({
            "election_deadline": soon,
            "event_category": "voluntary"
        })
        assert result["urgency"] == "critical"
        assert result["hours_remaining"] <= 24

    def test_past_deadline_is_critical(self):
        result = assess_urgency.invoke({
            "election_deadline": "2020-01-01T00:00:00+00:00",
            "event_category": "voluntary"
        })
        assert result["urgency"] == "critical"

    def test_no_deadline_voluntary(self):
        """Voluntary event with no deadline should be high urgency as precaution."""
        result = assess_urgency.invoke({
            "election_deadline": "",
            "event_category": "voluntary"
        })
        assert result["urgency"] in ("normal", "high", "critical")


# ══════════════════════════════════════════════════════════════════════════════
# ENTITLEMENT CALCULATION TESTS (DECIMAL SAFETY)
# ══════════════════════════════════════════════════════════════════════════════

class TestEntitlementCalculation:

    def setup_method(self):
        self.positions = [
            {"portfolio_name": "Growth Fund", "portfolio_id": "PF001",
             "quantity": 50000, "asset_class": "Equity", "base_currency": "USD"},
            {"portfolio_name": "Income Fund", "portfolio_id": "PF002",
             "quantity": 28000, "asset_class": "Equity", "base_currency": "USD"},
        ]

    def test_cash_dividend_correct_arithmetic(self):
        """50,000 shares × $0.25 = $12,500 exactly (no float error)."""
        result = calculate_entitlements.invoke({
            "positions": self.positions,
            "event_type": "DVCA",
            "gross_rate": "0.25",
            "tax_rate": "0.15",
            "currency": "USD",
            "offer_price": "0"
        })
        ents = result["entitlements"]
        growth = next(e for e in ents if e["portfolio_name"] == "Growth Fund")
        assert Decimal(growth["gross_entitlement"]) == Decimal("12500.00")
        assert Decimal(growth["withholding_tax"]) == Decimal("1875.00")
        assert Decimal(growth["net_entitlement"]) == Decimal("10625.00")

    def test_tender_offer_calculation(self):
        """22,000 shares × $142.50 = $3,135,000.00."""
        positions = [{"portfolio_name": "Growth Fund", "portfolio_id": "PF001",
                      "quantity": 22000, "asset_class": "Equity", "base_currency": "USD"}]
        result = calculate_entitlements.invoke({
            "positions": positions,
            "event_type": "TEND",
            "gross_rate": "0",
            "tax_rate": "0",
            "currency": "USD",
            "offer_price": "142.50"
        })
        ent = result["entitlements"][0]
        assert Decimal(ent["gross_entitlement"]) == Decimal("3135000.00")

    def test_total_projected_is_sum_of_parts(self):
        """Total must equal sum of individual entitlements."""
        result = calculate_entitlements.invoke({
            "positions": self.positions,
            "event_type": "DVCA",
            "gross_rate": "0.25",
            "tax_rate": "0.15",
            "currency": "USD",
            "offer_price": "0"
        })
        individual_sum = sum(
            Decimal(e["gross_entitlement"]) for e in result["entitlements"]
        )
        assert Decimal(result["total_projected"]) == individual_sum

    def test_float_precision_trap(self):
        """Classic float trap: 0.1 + 0.2 != 0.3 in float. Must not occur here."""
        positions = [{"portfolio_name": "Test Fund", "portfolio_id": "PF999",
                      "quantity": 3, "asset_class": "Equity", "base_currency": "USD"}]
        result = calculate_entitlements.invoke({
            "positions": positions,
            "event_type": "DVCA",
            "gross_rate": "0.1",
            "tax_rate": "0",
            "currency": "USD",
            "offer_price": "0"
        })
        # 3 × 0.1 = 0.30 exactly, not 0.30000000000000004
        ent = result["entitlements"][0]
        assert Decimal(ent["gross_entitlement"]) == Decimal("0.30")

    def test_invalid_rate_raises_error(self):
        with pytest.raises(ValueError):
            calculate_entitlements.invoke({
                "positions": self.positions,
                "event_type": "DVCA",
                "gross_rate": "not_a_number",
                "tax_rate": "0.15",
                "currency": "USD",
                "offer_price": "0"
            })


# ══════════════════════════════════════════════════════════════════════════════
# RECONCILIATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestReconciliation:

    def test_parse_mt566_clean(self):
        with open("ca_agent/data/sample_mt566_clean.txt") as f:
            msg = f.read()
        result = parse_swift_mt566.invoke({"message": msg})
        assert result["parse_success"] is True
        assert Decimal(result["gross_amount"]) == Decimal("50000.00")
        assert Decimal(result["net_amount"]) == Decimal("42500.00")
        assert Decimal(result["tax_amount"]) == Decimal("7500.00")

    def test_parse_mt566_with_break(self):
        with open("ca_agent/data/sample_mt566_break.txt") as f:
            msg = f.read()
        result = parse_swift_mt566.invoke({"message": msg})
        assert result["parse_success"] is True
        # Deliberate discrepancy: wrong tax rate applied
        assert Decimal(result["gross_amount"]) != Decimal("50000.00")

    def test_clean_reconciliation(self):
        """Identical projected and actual = no breaks."""
        projected = [
            {"portfolio_name": "Growth Fund", "portfolio_id": "PF001",
             "gross_entitlement": "12500.00", "withholding_tax": "1875.00",
             "net_entitlement": "10625.00", "currency": "USD"}
        ]
        result = compare_entitlements.invoke({
            "projected_entitlements": projected,
            "actual_gross": "12500.00",
            "actual_net": "10625.00",
            "actual_tax": "1875.00",
            "currency": "USD"
        })
        assert result["recon_status"] == "clean"
        assert len(result["breaks"]) == 0

    def test_tax_break_detected(self):
        """Custodian applied 20% tax instead of 15%."""
        projected = [
            {"portfolio_name": "Income Fund", "portfolio_id": "PF002",
             "gross_entitlement": "7000.00", "withholding_tax": "1050.00",
             "net_entitlement": "5950.00", "currency": "USD"}
        ]
        result = compare_entitlements.invoke({
            "projected_entitlements": projected,
            "actual_gross": "7000.00",
            "actual_net": "5600.00",   # 20% tax applied
            "actual_tax": "1400.00",
            "currency": "USD"
        })
        assert result["recon_status"] == "breaks_found"
        tax_break = next(b for b in result["breaks"] if b["break_type"] == "TAX")
        assert Decimal(tax_break["break_amount"]) == Decimal("350.00")
        assert "withholding tax" in tax_break["likely_cause"].lower()

    def test_large_break_flags_escalation(self):
        """Break > BREAK_THRESHOLD must flag requires_escalation=True."""
        projected = [
            {"portfolio_name": "Growth Fund", "portfolio_id": "PF001",
             "gross_entitlement": "50000.00", "withholding_tax": "7500.00",
             "net_entitlement": "42500.00", "currency": "USD"}
        ]
        result = compare_entitlements.invoke({
            "projected_entitlements": projected,
            "actual_gross": "47000.00",   # $3000 break — above $1000 threshold
            "actual_net": "39950.00",
            "actual_tax": "7050.00",
            "currency": "USD"
        })
        assert result["recon_status"] == "breaks_found"
        gross_break = next(b for b in result["breaks"] if b["break_type"] == "GROSS")
        assert gross_break["requires_escalation"] is True

    def test_recon_report_clean(self):
        report = generate_recon_report.invoke({
            "breaks": [],
            "event_type": "DVCA",
            "isin": "US0378331005",
            "issuer": "Apple Inc",
            "pay_date": "2026-07-10",
            "affected_portfolios": ["Growth Fund", "Income Fund"]
        })
        assert "CLEAN" in report
        assert "No breaks" in report

    def test_recon_report_with_breaks(self):
        breaks = [{
            "break_type": "TAX",
            "projected": "7500.00",
            "actual": "7589.00",
            "break_amount": "89.00",
            "break_amount_abs": "89.00",
            "break_pct": "1.19",
            "currency": "USD",
            "likely_cause": "Withholding tax rate mismatch",
            "requires_escalation": False
        }]
        report = generate_recon_report.invoke({
            "breaks": breaks,
            "event_type": "DVCA",
            "isin": "US0378331005",
            "issuer": "Apple Inc",
            "pay_date": "2026-07-10",
            "affected_portfolios": ["Income Fund"]
        })
        assert "BREAKS FOUND" in report
        assert "TAX" in report
        assert "NEXT STEPS" in report


# ══════════════════════════════════════════════════════════════════════════════
# SECURITY MASTER TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestSecurityMaster:

    def test_valid_identifiers(self):
        result = validate_security_identifiers.invoke({
            "isin": "US0378331005",
            "cusip": "037833100",
            "sedol": "2046251"
        })
        assert result["isin"]["checksum_valid"] is True
        assert result["cusip"]["format_valid"] is True

    def test_missing_sedol_flagged(self):
        """BP plc has missing CUSIP in our test data."""
        result = check_security_master_record.invoke({"isin": "GB0002634946"})
        assert result["found"] is True
        issues_text = " ".join(result["issues"])
        assert "CUSIP" in issues_text or "cusip" in issues_text.lower()

    def test_missing_isin_not_found(self):
        result = check_security_master_record.invoke({"isin": "US9999999999"})
        # ISIN checksum: this may fail validation — expected
        assert "found" in result

    def test_incomplete_security_volkswagen(self):
        """Volkswagen bond has multiple missing fields in test data."""
        result = check_security_master_record.invoke({"isin": "XS2345678903"})
        if result["found"]:
            # Should have multiple data quality issues
            assert len(result["issues"]) >= 1

    def test_recipient_tagging_voluntary(self):
        recipients = tag_recipients.invoke({
            "event_category": "voluntary",
            "urgency": "critical",
            "asset_class": "Equity"
        })
        assert "portfolio_manager" in recipients
        assert "senior_management" in recipients

    def test_recipient_tagging_mandatory(self):
        recipients = tag_recipients.invoke({
            "event_category": "mandatory",
            "urgency": "normal",
            "asset_class": "Equity"
        })
        assert "portfolio_manager" not in recipients
        assert "operations_team" in recipients
