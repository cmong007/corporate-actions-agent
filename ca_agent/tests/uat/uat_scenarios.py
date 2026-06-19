"""
UAT Scenarios for the Corporate Actions Processing Agent.

These are the three end-to-end demo scenarios for the Neuberger interview.
Each scenario maps directly to a real operational workflow.

Run individual scenarios:
    python -m ca_agent.tests.uat.run_uat --scenario 1
    python -m ca_agent.tests.uat.run_uat --scenario 2
    python -m ca_agent.tests.uat.run_uat --scenario 3

Run all scenarios:
    python -m ca_agent.tests.uat.run_uat --all
"""

SCENARIOS = {

    # ══════════════════════════════════════════════════════════════════════
    # SCENARIO 1: Mandatory Cash Dividend — Full Straight-Through Processing
    # ══════════════════════════════════════════════════════════════════════
    1: {
        "name": "Mandatory Cash Dividend — Full STP",
        "description": (
            "Apple Inc. declares a cash dividend of USD 0.25 per share. "
            "This is a mandatory event (no PM election required). "
            "The agent should process end-to-end without any human intervention. "
            "MT566 confirmation matches projected amounts exactly."
        ),
        "jd_mapping": [
            "Analysis and organization of corporate action announcement data",
            "Creation of internal CA event notifications",
            "Processing of CA entitlements",
        ],
        "expected_route": [
            "planner_node",
            "notification_node",
            "reconciliation_node",
            "security_master_node",
            "action_executor_node",
        ],
        "expected_outcomes": {
            "event_type": "DVCA",
            "event_category": "mandatory",
            "isin": "US0378331005",
            "escalation_required": False,
            "recon_status": "clean",
            "portfolios_affected": 3,
        },
        "input": {
            "input_source": "MT564",
            "raw_input": open("ca_agent/data/sample_mt564_dividend.txt").read()
                         + "\n\n--- MT566 CONFIRMATION ---\n\n"
                         + open("ca_agent/data/sample_mt566_clean.txt").read()
        },
        "pass_criteria": [
            "event_type == 'DVCA'",
            "event_category == 'mandatory'",
            "isin == 'US0378331005'",
            "No escalation triggered",
            "recon_status == 'clean'",
            "final_report contains 'CLEAN'",
            "notification_draft is not empty",
            "recipients includes 'operations_team'",
        ],
        "fail_criteria": [
            "Agent triggers escalation gate for mandatory event",
            "Entitlement calculation uses float (precision error)",
            "ISIN checksum validation fails for valid ISIN",
        ]
    },

    # ══════════════════════════════════════════════════════════════════════
    # SCENARIO 2: Voluntary Tender Offer — HITL Election Required
    # ══════════════════════════════════════════════════════════════════════
    2: {
        "name": "Voluntary Tender Offer — HITL PM Election",
        "description": (
            "A tender offer is announced for IBM at USD 142.50/share. "
            "This is a VOLUNTARY event — the Portfolio Manager MUST decide "
            "whether to tender. The agent must pause at the escalation gate "
            "and wait for approval before proceeding. "
            "Election deadline: 36 hours (CRITICAL urgency)."
        ),
        "jd_mapping": [
            "Operating to the highest standards of risk mitigation",
            "Point of contact and escalation for queries",
            "Timely escalation of issues to Senior Management",
        ],
        "expected_route": [
            "planner_node",
            "notification_node",
            "escalation_gate_node",  # ← PAUSES HERE
            # After /approve:
            "action_executor_node",
        ],
        "expected_outcomes": {
            "event_type": "TEND",
            "event_category": "voluntary",
            "isin": "US4592001014",
            "escalation_required": True,
            "urgency": "critical",
            "portfolios_affected": 1,
        },
        "input": {
            "input_source": "MT564",
            "raw_input": open("ca_agent/data/sample_mt564_tender.txt").read()
        },
        "pass_criteria": [
            "event_type == 'TEND'",
            "event_category == 'voluntary'",
            "Agent pauses at escalation gate",
            "API returns status='paused_for_approval'",
            "Escalation alert printed to console",
            "After /approve: graph resumes successfully",
            "final_report contains approved_by",
            "election_required == True",
        ],
        "fail_criteria": [
            "Agent processes voluntary event without pausing",
            "Agent submits election without human approval",
            "Graph fails to resume after /approve",
            "Task state lost on hypothetical server restart (SQLite persistence check)",
        ]
    },

    # ══════════════════════════════════════════════════════════════════════
    # SCENARIO 3: Reconciliation Break Detection — Tax Withholding Mismatch
    # ══════════════════════════════════════════════════════════════════════
    3: {
        "name": "Reconciliation Break — Tax Withholding Mismatch",
        "description": (
            "Apple Inc. cash dividend processed (same as Scenario 1). "
            "However, the MT566 confirmation shows withholding tax applied "
            "at 20% instead of the expected 15%. This creates a cash break. "
            "The agent must detect the break, classify it, generate an "
            "exception report, and escalate if above threshold."
        ),
        "jd_mapping": [
            "Processing and reconciliation of CA entitlements",
            "Operating to the highest standards of risk mitigation",
            "Identify and manage resolution of data quality issues",
        ],
        "expected_route": [
            "planner_node",
            "notification_node",
            "reconciliation_node",
            "escalation_gate_node",  # ← break exceeds threshold
        ],
        "expected_outcomes": {
            "event_type": "DVCA",
            "recon_status": "breaks_found",
            "break_type": "TAX",
            "likely_cause_contains": "withholding tax",
            "escalation_required": True,
        },
        "input": {
            "input_source": "MT566",
            "raw_input": open("ca_agent/data/sample_mt564_dividend.txt").read()
                         + "\n\n--- MT566 CONFIRMATION (WITH BREAK) ---\n\n"
                         + open("ca_agent/data/sample_mt566_break.txt").read()
        },
        "pass_criteria": [
            "recon_status == 'breaks_found'",
            "breaks list is non-empty",
            "TAX break detected",
            "break classified as withholding tax issue",
            "Exception report generated with NEXT STEPS",
            "Escalation triggered if max_break > BREAK_THRESHOLD",
        ],
        "fail_criteria": [
            "recon_status == 'clean' despite mismatched MT566",
            "Break classified incorrectly",
            "No exception report generated",
            "Float precision error in break calculation",
        ]
    },
}
