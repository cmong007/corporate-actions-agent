"""
Assembled LangGraph StateGraph for the Corporate Actions agent.

Key architectural decisions:
- SqliteSaver: persists checkpoints to disk (survives server restarts)
- interrupt_before=["escalation_gate_node"]: graph pauses BEFORE escalation,
  emitting the alert. Resumes only after human calls /approve or /reject.
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from ca_agent.graph.state import AgentState
from ca_agent.graph.nodes import (
    planner_node,
    checking_node,
    notification_node,
    reconciliation_node,
    security_master_node,
    escalation_gate_node,
    action_executor_node,
    error_node,
)
from ca_agent.graph.edges import (
    route_after_planner,
    route_after_checking,
    route_after_notification,
    route_after_reconciliation,
    route_after_escalation,
    route_after_security_master,
)
from ca_agent.config import CHECKPOINT_DB_PATH

import sqlite3


def build_graph():
    """Build and compile the Corporate Actions LangGraph state machine."""

    builder = StateGraph(AgentState)

    # ── Register all nodes ──────────────────────────────────────────────────
    builder.add_node("planner_node",        planner_node)
    builder.add_node("checking_node",       checking_node)
    builder.add_node("notification_node",   notification_node)
    builder.add_node("reconciliation_node", reconciliation_node)
    builder.add_node("security_master_node",security_master_node)
    builder.add_node("escalation_gate_node",escalation_gate_node)
    builder.add_node("action_executor_node",action_executor_node)
    builder.add_node("error_node",          error_node)

    # ── Entry point ─────────────────────────────────────────────────────────
    builder.add_edge(START, "planner_node")

    # ── Conditional edges (routing decisions) ───────────────────────────────
    builder.add_conditional_edges(
        "planner_node",
        route_after_planner,
        {
            "checking_node":       "checking_node",
            "error_node":          "error_node",
        }
    )

    builder.add_conditional_edges(
        "checking_node",
        route_after_checking,
        {
            "notification_node":   "notification_node",
            "escalation_gate":     "escalation_gate_node",
            "error_node":          "error_node",
            "planner_node":        "planner_node",
        }
    )

    builder.add_conditional_edges(
        "notification_node",
        route_after_notification,
        {
            "reconciliation_node": "reconciliation_node",
            "escalation_gate":     "escalation_gate_node",
            "error_node":          "error_node",
        }
    )

    builder.add_conditional_edges(
        "reconciliation_node",
        route_after_reconciliation,
        {
            "security_master_node": "security_master_node",
            "escalation_gate":      "escalation_gate_node",
            "error_node":           "error_node",
            "planner_node":         "planner_node",
        }
    )

    builder.add_conditional_edges(
        "escalation_gate_node",
        route_after_escalation,
        {
            "action_executor_node": "action_executor_node",
            "error_node":           "error_node",
            "checking_node":        "checking_node",
            END:                    END,
        }
    )

    builder.add_conditional_edges(
        "security_master_node",
        route_after_security_master,
        {
            "action_executor_node": "action_executor_node",
            "escalation_gate_node": "escalation_gate_node",
            "error_node":           "error_node",
        }
    )

    # ── Terminal edges ───────────────────────────────────────────────────────
    builder.add_edge("action_executor_node", END)
    builder.add_edge("error_node", END)

    # ── Persistence: SQLite checkpointer (survives server restarts) ──────────
    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    # ── Compile with HITL interrupt ──────────────────────────────────────────
    # The graph will PAUSE before escalation_gate_node runs.
    # Execution resumes only after a human calls /approve or /reject.
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["escalation_gate_node"],
    )

    return graph


# Singleton graph instance
graph = build_graph()
