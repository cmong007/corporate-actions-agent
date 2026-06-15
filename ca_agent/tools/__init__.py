"""Tool registry — maps each node to its designated tool group."""
from ca_agent.tools.ingestion_tools import (
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
    draft_internal_notification,
    tag_recipients,
    validate_security_identifiers,
    check_security_master_record,
)

# Each node only sees its own tool group.
# This is both a token-saving strategy and a safety control.
TOOL_REGISTRY = {
    "ingestion": [
        parse_swift_mt564,
        assess_urgency,
        load_portfolio_positions,
        calculate_entitlements,
    ],
    "notification": [
        draft_internal_notification,
        tag_recipients,
    ],
    "reconciliation": [
        parse_swift_mt566,
        compare_entitlements,
        generate_recon_report,
    ],
    "security_master": [
        validate_security_identifiers,
        check_security_master_record,
    ],
}
