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

    # ══════════════════════════════════════════════════════════════════════
    # SCENARIO 4: Data Quality — Invalid ISIN Checksum
    # ══════════════════════════════════════════════════════════════════════
    4: {
        "name": "Data Quality Failure — Invalid ISIN",
        "description": (
            "An incoming MT564 message contains an ISIN with an invalid "
            "checksum digit (a common data quality issue from some vendors). "
            "The agent must detect this immediately and route to escalation "
            "rather than proceeding with invalid data."
        ),
        "jd_mapping": [
            "Identify and manage the resolution of data quality issues",
            "Timely escalation of issues to Senior Management",
        ],
        "expected_route": [
            "planner_node",
            "escalation_gate_node",  # ← invalid data must escalate
        ],
        "expected_outcomes": {
            "parse_success": False,
            "escalation_required": True,
        },
        "input": {
            "input_source": "MT564",
            "raw_input": """{1:F01CITIUS33AXXX0000000000}{4:
:22F::CAEV//DVCA
:22F::CAMV//MAND
:35B:ISIN US0378331006
Apple Inc (INVALID CHECKSUM)
:92A::GRSS//0.25
:22F::CURR//USD
:98A::PAYD//20260710
-}"""
        },
        "pass_criteria": [
            "parse_success == False",
            "parse_errors contains checksum message",
            "Agent routes to escalation, NOT notification",
            "No entitlement calculation performed on invalid data",
        ],
        "fail_criteria": [
            "Agent proceeds with invalid ISIN",
            "Entitlements calculated for invalid ISIN",
            "No error raised for checksum failure",
        ]
    },

    # ══════════════════════════════════════════════════════════════════════
    # SCENARIO 5: Security Master Data Quality Check
    # ══════════════════════════════════════════════════════════════════════
    5: {
        "name": "Security Master — Missing Identifier Fields",
        "description": (
            "A Volkswagen bond event arrives. The security exists in our "
            "positions file but has missing fields in the Security Master "
            "(no CUSIP, no SEDOL, no exchange). "
            "The agent should process the event AND flag the Security Master "
            "issues with corrective action recommendations."
        ),
        "jd_mapping": [
            "Set up, maintenance and review of securities in trading systems",
            "Identify and manage the resolution of data quality issues",
            "Recommending appropriate corrective action",
        ],
        "expected_outcomes": {
            "security_master_issues_count_gte": 1,
            "suggested_fixes_count_gte": 1,
        },
        "input": {
            "input_source": "manual",
            "raw_input": """:22F::CAEV//DVCA
:22F::CAMV//MAND
:35B:ISIN XS2345678903
Volkswagen 1.875% 2028
:92A::GRSS//18.75
:22F::CURR//EUR
:98A::PAYD//20260715
:98A::RDDT//20260710
:98A::EXDT//20260709"""
        },
        "pass_criteria": [
            "security_master_issues is non-empty",
            "suggested_fixes is non-empty",
            "Issues mention missing fields",
            "Agent completes processing despite SM issues",
        ],
        "fail_criteria": [
            "Agent crashes on incomplete SM record",
            "No data quality issues flagged",
        ]
    },
}
