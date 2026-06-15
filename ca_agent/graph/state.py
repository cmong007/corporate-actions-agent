"""Shared AgentState TypedDict — the agent's working memory.

Every node reads from and writes to this object.
No node communicates directly with another node — only through state.
"""
from typing import TypedDict, Any, Optional, Annotated
from operator import add


class AgentState(TypedDict, total=False):
    audit_log: Annotated[list, add]

    # ── INPUT ───────────────────────────────────────────────────────────────
    task_id: str
    raw_input: str          # Raw SWIFT text, CSV row, or plain English
    input_source: str       # "MT564" | "MT566" | "CSV" | "manual"
    parse_success: bool
    parse_errors: list
    data_quality_warning: bool

    # ── CLASSIFICATION (set by Planner Node) ────────────────────────────────
    event_type: str         # ISO 15022: DVCA, SPLF, MRGR, TEND, RHTS, BONU
    event_category: str     # "mandatory" | "voluntary" | "elective"
    event_id: str           # Unique event reference from custodian
    isin: str
    issuer: str
    record_date: str
    ex_date: str
    pay_date: str
    election_deadline: str  # ISO 8601 UTC — always stored in UTC
    urgency: str            # "critical" | "high" | "normal"
    gross_rate: str         # Per-share rate as string (Decimal-safe)
    net_rate: str
    tax_rate: str
    currency: str
    offer_price: str        # For tender offers

    # ── ANALYSIS (set by Analysis Node) ─────────────────────────────────────
    affected_portfolios: list          # Portfolios holding the ISIN
    projected_entitlements: list       # Calculated entitlements per portfolio
    total_projected: str               # Total across all funds (Decimal string)
    election_required: bool

    # ── NOTIFICATION (set by Notification Node) ──────────────────────────────
    notification_draft: str
    recipients: list

    # ── RECONCILIATION (set by Reconciliation Node) ──────────────────────────
    actual_entitlements: list          # From MT566 confirmation
    breaks: list                       # Discrepancies found
    recon_status: str                  # "clean" | "breaks_found" | "pending"
    max_break_amount: float
    generate_recon_report_result: str  # Reconciliation report text

    # ── SECURITY MASTER (set by SecMaster Node) ──────────────────────────────
    security_master_issues: list
    suggested_fixes: list

    # ── CONTROL ──────────────────────────────────────────────────────────────
    pending_escalation: dict
    escalation_reason: str
    approval_status: str               # "pending" | "approved" | "rejected"
    approved_by: str                   # Who approved (audit trail)
    approved_at: str                   # Timestamp (audit trail)
    final_report: str
    iteration_count: int               # Safety counter
    completed_nodes: list              # Which nodes have run (audit trail)
    error: str
    error_node: str                    # Which node failed
    
    # Feedback loop control
    parser_retry_count: int
    parser_feedback: str
    recon_retry_count: int
    recon_feedback: str

