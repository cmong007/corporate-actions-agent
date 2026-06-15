"""
Audit Trail Reader for Corporate Actions Processing Agent.

Queries the LangGraph SqliteSaver checkpoint database for a given Task ID
and prints the complete step-by-step audit log of agent thoughts and actions.

Usage:
    python read_audit_trail.py <task_id>
"""
import sys
import os

# Ensure the root directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ca_agent.graph.graph import graph

# Terminal colors
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
MAGENTA = "\033[95m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def main():
    if len(sys.argv) < 2:
        print(f"{RED}{BOLD}Error: Missing Task ID.{RESET}")
        print("Usage: python read_audit_trail.py <task_id>")
        sys.exit(1)

    task_id = sys.argv[1].strip()
    config = {"configurable": {"thread_id": task_id}}

    try:
        # Fetch graph state
        state = graph.get_state(config)
    except Exception as e:
        print(f"{RED}{BOLD}Error retrieving state: {e}{RESET}")
        sys.exit(1)

    if not state.values:
        print(f"{RED}{BOLD}No state found for Task ID: '{task_id}'{RESET}")
        print("Please verify the Task ID is correct and exists in the checkpoint database.")
        sys.exit(1)

    values = state.values
    audit_log = values.get("audit_log", [])

    print(f"\n{BOLD}{BLUE}{'='*75}")
    print(f"🕵️‍♂️  CORPORATE ACTIONS AUDIT TRAIL")
    print(f"   Task ID: {task_id}")
    print(f"{'='*75}{RESET}")

    # General Metadata
    print(f"{BOLD}Metadata:{RESET}")
    print(f"  • Event Type    : {values.get('event_type', 'N/A')} ({values.get('event_category', 'N/A')})")
    print(f"  • Security (ISIN): {values.get('issuer', 'N/A')} ({values.get('isin', 'N/A')})")
    print(f"  • Recon Status  : {values.get('recon_status', 'N/A')}")
    print(f"  • Approval      : {values.get('approval_status', 'N/A')}")
    if values.get("approval_status") == "approved":
        print(f"    - Approved by : {values.get('approved_by', 'N/A')} at {values.get('approved_at', 'N/A')}")
    print(f"  • Execution Path: {' → '.join(values.get('completed_nodes', []))}")
    print("-" * 75)

    if not audit_log:
        print(f"{YELLOW}⚠️  No audit log entries found in state.{RESET}")
    else:
        print(f"{BOLD}Audit Log ({len(audit_log)} steps):{RESET}\n")
        for i, entry in enumerate(audit_log, 1):
            timestamp = entry.get("timestamp", "N/A")
            node = entry.get("node", "N/A")
            thought = entry.get("thought", "No reasoning provided.")
            actions = entry.get("actions", [])

            print(f"{BOLD}{CYAN}Step {i}: {node}{RESET}   [{timestamp}]")
            
            # Print thoughts
            print(f"  {BOLD}Thought Process:{RESET}")
            # Wrap/indent thought lines for readability
            thought_lines = thought.split("\n")
            for t_line in thought_lines:
                if t_line.strip():
                    print(f"    {t_line.strip()}")
            
            # Print actions
            if actions:
                print(f"  {BOLD}Actions Taken / Tools Called:{RESET}")
                for action in actions:
                    print(f"    - {GREEN}{action}{RESET}")
            else:
                print(f"  {BOLD}Actions Taken / Tools Called:{RESET} None")
            
            print("-" * 75)

    # Final Summary Report
    final_report = values.get("final_report", "")
    if final_report:
        print(f"\n{BOLD}{MAGENTA}Final Execution Report Summary:{RESET}")
        print(final_report)
        print("="*75)


if __name__ == "__main__":
    main()
