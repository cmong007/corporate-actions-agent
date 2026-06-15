"""
Unit tests for the Corporate Actions Processing Agent feedback loops.
Run with: python -m pytest ca_agent/tests/test_loops.py -v
"""
import pytest
from ca_agent.graph.state import AgentState
from ca_agent.graph.edges import (
    route_after_checking,
    route_after_notification,
    route_after_reconciliation,
    route_after_escalation,
)
from ca_agent.graph.nodes import checking_node, reconciliation_node


class TestParserFeedbackLoop:

    def test_checking_node_first_retry(self):
        """First time checking_node fails, it should set feedback and increment retry count."""
        state: AgentState = {
            "iteration_count": 0,
            "issuer": ":35B::Contaminated Issuer",  # Raw tag
            "parser_retry_count": 0,
            "completed_nodes": [],
        }
        res = checking_node(state)
        assert res.get("parser_retry_count") == 1
        assert "Contaminated Issuer" in res.get("parser_feedback")
        assert res.get("data_quality_warning") is True
        # Verify it has not written raw tag error to escalation reasons yet
        assert not res.get("escalation_reason", "").startswith("Checking Agent (Max Retries)")

    def test_checking_node_max_retries_exceeded(self):
        """After 2 retries, checking_node should stop retrying and escalate."""
        state: AgentState = {
            "iteration_count": 0,
            "issuer": ":35B::Contaminated Issuer",
            "parser_retry_count": 2,
            "completed_nodes": [],
            "parse_errors": ["Previous error"],
        }
        res = checking_node(state)
        # Should not set parser_feedback or increment parser_retry_count further
        assert "parser_feedback" not in res
        assert "parser_retry_count" not in res
        assert res.get("data_quality_warning") is True
        assert "Checking Agent (Max Retries)" in res.get("escalation_reason", "")

    def test_checking_node_bypass_when_approved(self):
        """If already approved, checking_node should bypass and allow normal routing."""
        state: AgentState = {
            "iteration_count": 0,
            "issuer": ":35B::Contaminated Issuer",
            "approval_status": "approved",
            "completed_nodes": [],
        }
        res = checking_node(state)
        assert "parser_feedback" not in res
        assert "parse_errors" not in res
        assert "data_quality_warning" not in res

    def test_routing_after_checking(self):
        """Verify routing choices depending on errors and feedback availability."""
        # 1. Successful parse
        state: AgentState = {
            "iteration_count": 1,
            "parse_errors": [],
            "data_quality_warning": False,
        }
        assert route_after_checking(state) == "notification_node"

        # 2. Parse errors, retries remaining (feedback present)
        state: AgentState = {
            "iteration_count": 1,
            "parse_errors": ["Malformed tag"],
            "data_quality_warning": True,
            "parser_feedback": "Malformed tag details",
            "parser_retry_count": 1,
        }
        assert route_after_checking(state) == "planner_node"

        # 3. Parse errors, max retries reached (feedback cleared)
        state: AgentState = {
            "iteration_count": 1,
            "parse_errors": ["Malformed tag"],
            "data_quality_warning": True,
            "parser_feedback": "",
            "parser_retry_count": 2,
        }
        assert route_after_checking(state) == "escalation_gate"

        # 4. Approved state bypasses errors
        state: AgentState = {
            "iteration_count": 1,
            "parse_errors": ["Malformed tag"],
            "data_quality_warning": True,
            "approval_status": "approved",
        }
        assert route_after_checking(state) == "notification_node"


class TestReconciliationFeedbackLoop:

    def test_routing_after_reconciliation(self):
        """Verify reconciliation routing choices for clean, retry, or escalation states."""
        # 1. Clean reconciliation
        state: AgentState = {
            "iteration_count": 1,
            "recon_status": "clean",
            "max_break_amount": 0.0,
        }
        assert route_after_reconciliation(state) == "security_master_node"

        # 2. Break found, retry remaining (feedback present)
        state: AgentState = {
            "iteration_count": 1,
            "recon_status": "breaks_found",
            "max_break_amount": 1200.0,  # > 1000 threshold
            "recon_feedback": "Break detected in Growth Fund",
            "recon_retry_count": 1,
        }
        assert route_after_reconciliation(state) == "planner_node"

        # 3. Break found, max retries reached (feedback cleared)
        state: AgentState = {
            "iteration_count": 1,
            "recon_status": "breaks_found",
            "max_break_amount": 1200.0,
            "recon_feedback": "",
            "recon_retry_count": 1,
        }
        assert route_after_reconciliation(state) == "escalation_gate"

        # 4. Break found but approved by human
        state: AgentState = {
            "iteration_count": 1,
            "recon_status": "breaks_found",
            "max_break_amount": 1200.0,
            "approval_status": "approved",
        }
        assert route_after_reconciliation(state) == "security_master_node"


class TestPostEscalationLoop:

    def test_routing_after_notification_bypass(self):
        """Voluntary event should bypass escalation if already approved."""
        state: AgentState = {
            "iteration_count": 1,
            "event_category": "voluntary",
        }
        assert route_after_notification(state) == "escalation_gate"

        state["approval_status"] = "approved"
        assert route_after_notification(state) == "reconciliation_node"

    def test_routing_after_escalation_reroute(self):
        """Approved escalation should reroute to checking_node instead of action_executor."""
        state: AgentState = {
            "iteration_count": 1,
            "approval_status": "approved",
        }
        assert route_after_escalation(state) == "checking_node"

        state["approval_status"] = "rejected"
        assert route_after_escalation(state) == "error_node"
