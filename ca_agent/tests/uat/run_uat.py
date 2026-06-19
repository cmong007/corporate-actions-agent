"""
UAT Test Runner for the Corporate Actions Processing Agent.

Usage:
    # Run all scenarios
    python -m ca_agent.tests.uat.run_uat --all

    # Run a specific scenario
    python -m ca_agent.tests.uat.run_uat --scenario 1

    # Run with verbose output
    python -m ca_agent.tests.uat.run_uat --all --verbose
"""
import argparse
import sys
import time
import uuid
import os
from datetime import datetime, timezone

from ca_agent.graph.graph import graph
from ca_agent.tests.uat.uat_scenarios import SCENARIOS


# ── Colours for terminal output ─────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def print_banner(scenario_num: int, scenario: dict):
    print(f"\n{'='*65}")
    print(f"{BOLD}{BLUE}UAT SCENARIO {scenario_num}: {scenario['name']}{RESET}")
    print(f"{'='*65}")
    print(f"{scenario['description']}")
    print(f"\n{BOLD}JD Mapping:{RESET}")
    for item in scenario.get("jd_mapping", []):
        print(f"  * {item}")
    print(f"\n{BOLD}Expected Route:{RESET} {' -> '.join(scenario.get('expected_route', []))}")
    print("-" * 65)


def run_scenario(scenario_num: int, scenario: dict, verbose: bool = False) -> dict:
    """
    Execute a single UAT scenario against the live agent graph.
    Returns a results dict with pass/fail status.
    """
    print_banner(scenario_num, scenario)

    task_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": task_id}}

    inp = scenario["input"]
    initial_state = {
        "task_id": task_id,
        "raw_input": inp["raw_input"],
        "input_source": inp.get("input_source", "manual"),
        "iteration_count": 0,
        "completed_nodes": [],
    }

    start_time = time.time()
    results = {
        "scenario": scenario_num,
        "name": scenario["name"],
        "task_id": task_id,
        "passed": [],
        "failed": [],
        "warnings": [],
        "state": {},
        "duration_s": 0,
    }

    print(f"[RUN] Running agent... (task_id: {task_id[:8]}...)")

    try:
        # ── Run the graph ─────────────────────────────────────────────────────
        final_state = graph.invoke(initial_state, config=config)
        graph_state = graph.get_state(config)
        is_paused = bool(graph_state.next)
        results["state"] = final_state
        results["is_paused"] = is_paused

        if verbose:
            print(f"\n{BOLD}Agent State Snapshot:{RESET}")
            for key in ["event_type", "event_category", "isin", "urgency",
                        "recon_status", "breaks", "security_master_issues",
                        "completed_nodes", "error"]:
                val = final_state.get(key)
                if val is not None and val != [] and val != "":
                    print(f"  {key}: {val}")

        # ── Scenario 2: Simulate HITL approval ───────────────────────────────
        if scenario_num == 2 and is_paused:
            print(f"\n{YELLOW}[PAUSE] Agent paused for HITL approval (as expected){RESET}")
            print(f"   Simulating: POST /task/{task_id[:8]}.../approve")
            time.sleep(0.5)  # Simulate human review time

            graph.update_state(config, {
                "approval_status": "approved",
                "approved_by": "UAT_Test_PM",
                "approved_at": datetime.now(timezone.utc).isoformat(),
            })
            final_state = graph.invoke(None, config=config)
            results["state"] = final_state
            results["is_paused"] = False
            print(f"{GREEN}[OK] Approval simulated - graph resumed{RESET}")

        # ── Evaluate pass criteria ────────────────────────────────────────────
        _evaluate_criteria(scenario, final_state, is_paused, results)

    except Exception as e:
        results["failed"].append(f"AGENT EXCEPTION: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()

    results["duration_s"] = round(time.time() - start_time, 2)

    # -- Print results ---------------------------------------------------------
    print(f"\n{BOLD}Results:{RESET}")
    for p in results["passed"]:
        print(f"  {GREEN}[OK] PASS{RESET}: {p}")
    for f in results["failed"]:
        print(f"  {RED}[FAIL] FAIL{RESET}: {f}")
    for w in results["warnings"]:
        print(f"  {YELLOW}[WARN] WARN{RESET}: {w}")

    status_colour = GREEN if not results["failed"] else RED
    status_label  = "PASSED" if not results["failed"] else "FAILED"
    print(f"\n{BOLD}{status_colour}SCENARIO {scenario_num}: {status_label}{RESET} "
          f"({results['duration_s']}s) — "
          f"{len(results['passed'])} passed, {len(results['failed'])} failed")

    return results


def _evaluate_criteria(scenario, final_state, is_paused, results):
    """Check pass/fail criteria against actual agent output."""
    expected = scenario.get("expected_outcomes", {})
    state = final_state or {}

    # ── Check expected outcomes ───────────────────────────────────────────────
    if "event_type" in expected:
        actual = state.get("event_type")
        if actual == expected["event_type"]:
            results["passed"].append(f"event_type == '{expected['event_type']}'")
        else:
            results["failed"].append(
                f"event_type: expected '{expected['event_type']}', got '{actual}'"
            )

    if "event_category" in expected:
        actual = state.get("event_category")
        if actual == expected["event_category"]:
            results["passed"].append(f"event_category == '{expected['event_category']}'")
        else:
            results["failed"].append(
                f"event_category: expected '{expected['event_category']}', got '{actual}'"
            )

    if "isin" in expected:
        actual = state.get("isin")
        if actual == expected["isin"]:
            results["passed"].append(f"isin == '{expected['isin']}'")
        else:
            results["failed"].append(f"isin: expected '{expected['isin']}', got '{actual}'")

    if "escalation_required" in expected:
        if expected["escalation_required"]:
            if is_paused or state.get("pending_escalation") or \
               "escalation_gate_node" in state.get("completed_nodes", []):
                results["passed"].append("Escalation triggered correctly")
            else:
                results["failed"].append("Expected escalation but agent did not pause")
        else:
            if not is_paused:
                results["passed"].append("No escalation triggered (correct for mandatory event)")
            else:
                results["failed"].append("Agent unexpectedly paused for escalation")

    if "recon_status" in expected:
        actual = state.get("recon_status")
        if actual == expected["recon_status"]:
            results["passed"].append(f"recon_status == '{expected['recon_status']}'")
        elif actual is None:
            results["warnings"].append("recon_status not set (MT566 may not have been provided)")
        else:
            results["failed"].append(
                f"recon_status: expected '{expected['recon_status']}', got '{actual}'"
            )

    if "parse_success" in expected:
        actual = state.get("parse_success")
        if actual == expected["parse_success"]:
            results["passed"].append(f"parse_success == {expected['parse_success']}")
        else:
            results["failed"].append(
                f"parse_success: expected {expected['parse_success']}, got {actual}"
            )

    if expected.get("security_master_issues_count_gte", 0) > 0:
        issues = state.get("security_master_issues", [])
        if len(issues) >= expected["security_master_issues_count_gte"]:
            results["passed"].append(
                f"Security master issues found: {len(issues)} >= "
                f"{expected['security_master_issues_count_gte']}"
            )
        else:
            results["failed"].append(
                f"Expected >= {expected['security_master_issues_count_gte']} "
                f"security master issues, got {len(issues)}"
            )

    # ── Check for no agent error ──────────────────────────────────────────────
    error = state.get("error", "")
    if error and "escalation" not in scenario["name"].lower():
        if state.get("error_node") == "error_node":
            results["failed"].append(f"Agent hit error_node: {error[:100]}")
        else:
            results["warnings"].append(f"Error in state: {error[:100]}")

    # ── Check final report generated ─────────────────────────────────────────
    report = state.get("final_report", "")
    if report:
        results["passed"].append("Final report generated")
    else:
        results["warnings"].append("No final report in state")

    # ── Check notification drafted ────────────────────────────────────────────
    notification = state.get("notification_draft", "")
    if notification:
        results["passed"].append("Internal notification drafted")


def main():
    parser = argparse.ArgumentParser(description="CA Agent UAT Test Runner")
    parser.add_argument("--scenario", type=int, help="Run specific scenario number")
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if not args.scenario and not args.all:
        parser.print_help()
        sys.exit(1)

    scenarios_to_run = list(SCENARIOS.keys()) if args.all else [args.scenario]
    all_results = []

    print(f"\n{BOLD}{'='*65}")
    print(f"  CORPORATE ACTIONS AGENT - USER ACCEPTANCE TESTING")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*65}{RESET}")

    for num in scenarios_to_run:
        if num not in SCENARIOS:
            print(f"{RED}Scenario {num} not found{RESET}")
            continue
        result = run_scenario(num, SCENARIOS[num], verbose=args.verbose)
        all_results.append(result)

    # ── Final summary ─────────────────────────────────────────────────────────
    if len(all_results) > 1:
        total = len(all_results)
        passed = sum(1 for r in all_results if not r["failed"])
        failed = total - passed
        total_duration = sum(r["duration_s"] for r in all_results)

        print(f"\n{'='*65}")
        print(f"{BOLD}UAT SUMMARY{RESET}")
        print(f"{'='*65}")
        for r in all_results:
            status = f"{GREEN}PASS{RESET}" if not r["failed"] else f"{RED}FAIL{RESET}"
            print(f"  Scenario {r['scenario']}: {status} - {r['name']}")

        status_colour = GREEN if failed == 0 else RED
        print(f"\n{BOLD}{status_colour}"
              f"TOTAL: {passed}/{total} scenarios passed "
              f"({total_duration:.1f}s total){RESET}")
        print("=" * 65)

    sys.exit(0 if all(not r["failed"] for r in all_results) else 1)


if __name__ == "__main__":
    main()
