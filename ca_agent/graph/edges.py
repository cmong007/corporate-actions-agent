"""
Conditional edge routing functions for the Corporate Actions agent.

Each function inspects the current state and returns a string
telling LangGraph which node to run next.
"""
from langgraph.graph import END
from ca_agent.graph.state import AgentState
from ca_agent.config import MAX_ITERATIONS, BREAK_THRESHOLD


def route_after_planner(state: AgentState) -> str:
    """After parsing and classification — always route to checking_node first."""

    # Hard stop
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "error_node"

    # Execution error — skip checking and go straight to error
    if state.get("error"):
        return "error_node"

    # All valid or invalid inputs pass through the checking agent next
    return "checking_node"


def route_after_checking(state: AgentState) -> str:
    """After Checking Agent validates parsed fields — escalate or continue."""

    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "error_node"

    if state.get("error"):
        return "error_node"

    # Bypass checks if approved
    if state.get("approval_status") == "approved":
        return "notification_node"

    parse_errors = state.get("parse_errors", [])
    data_quality_warning = state.get("data_quality_warning")

    if parse_errors or data_quality_warning:
        # If parser_feedback is set, route back to planner_node for self-correction.
        # It is cleared in the planner_node once processed.
        if state.get("parser_feedback"):
            return "planner_node"
        return "escalation_gate"

    # CRITICAL voluntary/elective with imminent deadline — skip to escalation
    event_category = state.get("event_category", "mandatory")
    urgency = state.get("urgency", "normal")
    if event_category in ("voluntary", "elective") and urgency == "critical":
        return "escalation_gate"

    # Standard path: notification first
    return "notification_node"


def route_after_notification(state: AgentState) -> str:
    """After drafting notification — proceed to reconciliation or escalate."""

    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "error_node"

    if state.get("error"):
        return "error_node"

    # Voluntary events still need PM confirmation via HITL
    event_category = state.get("event_category", "mandatory")
    if event_category in ("voluntary", "elective"):
        # Bypass escalation if approved
        if state.get("approval_status") == "approved":
            return "reconciliation_node"
        return "escalation_gate"

    return "reconciliation_node"


def route_after_reconciliation(state: AgentState) -> str:
    """After break detection — escalate large breaks or proceed to SecMaster."""

    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "error_node"

    if state.get("error"):
        return "error_node"

    # Bypass break check if approved
    if state.get("approval_status") == "approved":
        return "security_master_node"

    recon_status = state.get("recon_status", "pending")
    max_break = state.get("max_break_amount", 0.0)

    if recon_status == "breaks_found" and float(max_break) > BREAK_THRESHOLD:
        # If recon_feedback is set, route back to planner_node for self-correction.
        # It is cleared in the planner_node once processed.
        if state.get("recon_feedback"):
            return "planner_node"
        return "escalation_gate"

    return "security_master_node"


def route_after_escalation(state: AgentState) -> str:
    """After human responds to escalation — resume or end."""

    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "error_node"

    approval_status = state.get("approval_status", "pending")

    if approval_status == "approved":
        # Post-escalation reconciliation gate: route back to checking_node
        return "checking_node"

    if approval_status == "rejected":
        # Agent logs rejection and ends — manual handling required
        return "error_node"

    # Still pending — graph should not reach here, but handle gracefully
    return END


def route_after_security_master(state: AgentState) -> str:
    """After Security Master check — escalate critical issues or proceed to final execution."""

    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "error_node"

    issues = state.get("security_master_issues", [])
    approval_status = state.get("approval_status", "")

    # Critical issues that should block booking: missing asset class, inactive security,
    # or missing exchange. Surface to HITL for analyst to confirm before booking.
    if issues and approval_status != "approved":
        critical_keywords = ["missing", "inactive", "not found", "invalid", "no exchange", "no asset class"]
        has_critical = any(
            any(kw in str(issue).lower() for kw in critical_keywords)
            for issue in issues
        )
        if has_critical:
            return "escalation_gate_node"

    return "action_executor_node"

