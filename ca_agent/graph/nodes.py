"""
All LangGraph node functions for the Corporate Actions agent.

Safety rules applied in EVERY node:
1. Increment iteration_count immediately
2. Check MAX_ITERATIONS before doing any work
3. Track completed_nodes for audit trail
4. All tool calls wrapped in try/except — failures route to error_node
"""
import json
import os
from datetime import datetime, timezone
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from ca_agent.graph.state import AgentState
from ca_agent.llm.provider import get_llm
from ca_agent.tools import TOOL_REGISTRY
from ca_agent.config import MAX_ITERATIONS, ESCALATION_WEBHOOK_URL, BREAK_THRESHOLD, active_reasoning


# ── Shared Helpers ─────────────────────────────────────────────────────────────

def _increment_and_check(state: AgentState, node_name: str) -> dict | None:
    """Increment iteration counter. Return error dict if limit exceeded."""
    count = state.get("iteration_count", 0) + 1
    completed = list(state.get("completed_nodes", []))
    completed.append(node_name)

    if count > MAX_ITERATIONS:
        return {
            "iteration_count": count,
            "completed_nodes": completed,
            "error": f"MAX_ITERATIONS ({MAX_ITERATIONS}) exceeded at {node_name}.",
            "error_node": node_name,
        }
    return {"iteration_count": count, "completed_nodes": completed}


def _execute_tool_calls(response, tools: list[BaseTool]) -> dict:
    """
    Execute tool calls requested by the LLM and collect results.
    Returns a merged dict of all tool outputs.
    """
    tool_map = {t.name: t for t in tools}
    results = {}

    if not hasattr(response, "tool_calls") or not response.tool_calls:
        return results

    for tc in response.tool_calls:
        tool_name = tc["name"]
        tool_args = tc.get("args", {})

        if tool_name not in tool_map:
            continue

        try:
            output = tool_map[tool_name].invoke(tool_args)
            if isinstance(output, dict):
                results.update(output)
            else:
                # Store non-dict outputs under the tool name
                results[tool_name + "_result"] = output
        except Exception as e:
            results["tool_error"] = f"{tool_name} failed: {str(e)}"
            results["error"] = str(e)

    return results


def _extract_text_content(content) -> str:
    """Extract string text content from a LangChain message content (which could be a list)."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif "text" in item:
                    parts.append(item["text"])
            elif hasattr(item, "text"):
                parts.append(item.text)
            elif hasattr(item, "get") and item.get("text"):
                parts.append(item.get("text"))
        return " ".join(parts)
    return str(content)


def _format_thoughts(thoughts: list[str]) -> str:
    """Format list of thoughts into a clean bulleted list with newlines."""
    formatted = []
    for t in thoughts:
        t_clean = t.strip()
        if not t_clean:
            continue
        # Split by newlines if the thought content contains them
        for line in t_clean.split("\n"):
            line_clean = line.strip()
            if not line_clean:
                continue
            if line_clean.startswith(("•", "-", "*", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
                formatted.append(line_clean)
            else:
                formatted.append(f"• {line_clean}")
    return "\n".join(formatted)


def _run_node_llm_loop(llm, tools: list[BaseTool], messages: list, node_name: str, task_id: str = None) -> dict:
    """
    Runs a multi-turn LLM agent loop inside a single node, executing tool calls
    and feeding the results back to the LLM until it stops calling tools.
    Returns a merged dict of all tool results and state updates.
    """
    tool_map = {t.name: t for t in tools}
    accumulated_results = {}
    thoughts = []
    actions = []

    try:
        for _ in range(4):
            response = llm.invoke(messages)
            messages.append(response)

            if response.content:
                text_content = _extract_text_content(response.content)
                if text_content.strip():
                    thoughts.append(text_content.strip())

            # Update active reasoning for the live UI audit trail
            if task_id:
                thought_str = _format_thoughts(thoughts)
                current_actions = []
                if hasattr(response, "tool_calls") and response.tool_calls:
                    for tc in response.tool_calls:
                        current_actions.append(tc["name"])
                all_actions = actions + current_actions
                active_reasoning[task_id] = {
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                    "node": node_name,
                    "thought": thought_str or "• Analyzing...",
                    "actions": list(dict.fromkeys(all_actions))
                }

            if not hasattr(response, "tool_calls") or not response.tool_calls:
                # LLM is done and has responded with text
                break

            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc.get("args", {})
                tool_id = tc.get("id")

                actions.append(tool_name)

                if tool_name not in tool_map:
                    err_msg = f"Error: Tool '{tool_name}' not found."
                    messages.append(ToolMessage(content=err_msg, tool_call_id=tool_id))
                    continue

                try:
                    output = tool_map[tool_name].invoke(tool_args)

                    # Store in accumulated results
                    if isinstance(output, dict):
                        accumulated_results.update(output)
                        output_str = json.dumps(output)
                    else:
                        accumulated_results[tool_name + "_result"] = output
                        output_str = str(output)

                    messages.append(ToolMessage(content=output_str, tool_call_id=tool_id))
                except Exception as e:
                    err_str = f"Error executing tool {tool_name}: {str(e)}"
                    accumulated_results["tool_error"] = err_str
                    accumulated_results["error"] = str(e)
                    messages.append(ToolMessage(content=err_str, tool_call_id=tool_id))

            if "error" in accumulated_results:
                break

        # Build audit log entry
        unique_actions = list(dict.fromkeys(actions))
        thought_str = _format_thoughts(thoughts)
        if not thought_str:
            thought_str = "• No reasoning content provided by LLM."

        log_entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "node": node_name,
            "thought": thought_str,
            "actions": unique_actions
        }
        accumulated_results["audit_log"] = [log_entry]

    finally:
        # Clean active reasoning storage when done
        if task_id in active_reasoning:
            del active_reasoning[task_id]

    return accumulated_results


# ── Node 1: Planner / Ingestion ─────────────────────────────────────────────────

def planner_node(state: AgentState) -> dict:
    """
    Parse and classify the incoming corporate action.

    Uses cheap model — this is routing and extraction, not heavy reasoning.
    Implements deterministic pre-checks before the LLM runs.
    """
    guard = _increment_and_check(state, "planner_node")
    if "error" in guard:
        return guard

    tools = TOOL_REGISTRY["ingestion"]
    llm = get_llm("planner").bind_tools(tools)

    raw_input = state.get("raw_input", "")

    messages = [
        SystemMessage(content="""You are a Senior Corporate Actions Operations specialist.

Your job is to process incoming corporate action announcements and extract all key data.

COGNITIVE REASONING REQUIREMENT:
You MUST output a brief, single-sentence explanation in your text response before calling each tool. Do not leave the response content empty when calling tools.
CRITICAL: If you are retrying because of a validation warning (parser feedback or reconciliation feedback in the conversation history), you MUST explain the issue you are correcting and why (e.g. 'I am correcting the previous parsing error due to raw tags by...' or 'I am correcting the entitlement calculation because of the withholding tax break by...').
Otherwise, explain that you are parsing the SWIFT MT564 message, that you are assessing the urgency of the event, that you are loading portfolio positions, or that you are calculating entitlements. Your explanations will be displayed in real-time on the operator console.

INSTRUCTIONS:
1. Call parse_swift_mt564() if the input looks like a SWIFT MT564 message (contains ':22F::CAEV').
2. After parsing, call assess_urgency() with the election_deadline and event_category.
3. Call load_portfolio_positions() with the extracted ISIN.
4. Call calculate_entitlements() with the positions and rate data.

CRITICAL RULES:
- If parse_swift_mt564 returns parse_success=False or has parse_errors, DO NOT proceed.
  Return your findings and flag for escalation.
- If load_portfolio_positions returns data_quality_warning=True (zero positions),
  DO NOT calculate entitlements. Flag as data quality issue.
- For event_category, use the CAMV field: MAND=mandatory, VOLU=voluntary, CHOS=elective.
- Always pass gross_rate as a string, not a float.
"""),
        HumanMessage(content=f"Process this corporate action:\n\n{raw_input}")
    ]

    resets = {}
    parser_feedback = state.get("parser_feedback")
    if parser_feedback:
        messages.append(
            HumanMessage(
                content=(
                    f"ATTENTION: Your previous parse attempt had validation issues: "
                    f"{parser_feedback}. Please re-run the parse_swift_mt564 tool and "
                    f"correct these errors. Avoid capturing raw tags (like :92A::GRSS//18.75 "
                    f"or /US/037833100) as values. Extract only clean human-readable values."
                )
            )
        )
        resets["parser_feedback"] = ""
        resets["parse_errors"] = []
        resets["data_quality_warning"] = False

    recon_feedback = state.get("recon_feedback")
    if recon_feedback:
        messages.append(
            HumanMessage(
                content=(
                    f"ATTENTION: A reconciliation break was detected between our projected calculations "
                    f"and the custodian's actual confirmation. Details:\n{recon_feedback}\n"
                    f"Please review the raw input / narrative, double check if you parsed the rates "
                    f"or options correctly, re-calculate the entitlements using calculate_entitlements "
                    f"with the correct rates, and output the updated values. Adjust your extraction accordingly."
                )
            )
        )
        resets["recon_feedback"] = ""
        resets["breaks"] = []
        resets["recon_status"] = ""
        resets["max_break_amount"] = 0.0
        resets["generate_recon_report_result"] = ""

    tool_results = _run_node_llm_loop(llm, tools, messages, "planner_node", state.get("task_id"))

    # Handle tool errors
    if "tool_error" in tool_results or "error" in tool_results:
        return {
            **guard,
            **resets,
            "error": tool_results.get("error", "Tool execution failed in planner_node"),
            "error_node": "planner_node",
        }

    # Map tool return keys to AgentState keys
    if "entitlements" in tool_results:
        tool_results["projected_entitlements"] = tool_results["entitlements"]
    if "positions" in tool_results:
        tool_results["affected_portfolios"] = tool_results["positions"]

    return {**guard, **resets, **tool_results}


# ── Node 2: Notification ────────────────────────────────────────────────────────

def notification_node(state: AgentState) -> dict:
    """
    Draft the internal CA notification and tag recipients.
    Uses cheap model — this is structured drafting, not complex reasoning.
    """
    guard = _increment_and_check(state, "notification_node")
    if "error" in guard:
        return guard

    tools = TOOL_REGISTRY["notification"]
    llm = get_llm("planner").bind_tools(tools)

    entitlements = state.get("projected_entitlements", [])
    if not entitlements:
        entitlements = state.get("entitlements", [])

    messages = [
        SystemMessage(content="""You are a Corporate Actions Operations analyst.

COGNITIVE REASONING REQUIREMENT:
You MUST output a brief, single-sentence explanation in your text response before calling each tool. Do not leave the response content empty when calling tools. For example, explain that you are drafting the internal corporate action notification, or that you are tagging the relevant recipients based on the notification content. Your explanations will be displayed in real-time on the operator console.

Draft an internal notification for the corporate action event using the
draft_internal_notification tool. Then call tag_recipients to determine
who should receive the notification.

Use all the event details provided. Be precise with dates and amounts.
"""),
        HumanMessage(content=f"""
Event Type       : {state.get('event_type', 'UNKNOWN')}
Event Category   : {state.get('event_category', 'mandatory')}
ISIN             : {state.get('isin', '')}
Issuer           : {state.get('issuer', '')}
Record Date      : {state.get('record_date', '')}
Ex-Date          : {state.get('ex_date', '')}
Pay Date         : {state.get('pay_date', '')}
Election Deadline: {state.get('election_deadline', '')}
Gross Rate       : {state.get('gross_rate', '0')}
Currency         : {state.get('currency', 'USD')}
Total Projected  : {state.get('total_projected', '0')}
Urgency          : {state.get('urgency', 'normal')}
Entitlements     : {json.dumps(entitlements[:5])}
Narrative        : {state.get('narrative', '')[:200]}

Draft the notification and identify recipients.
""")
    ]

    tool_results = _run_node_llm_loop(llm, tools, messages, "notification_node", state.get("task_id"))

    # Map tool return keys to AgentState keys
    if "draft_internal_notification_result" in tool_results:
        tool_results["notification_draft"] = tool_results["draft_internal_notification_result"]
    if "tag_recipients_result" in tool_results:
        tool_results["recipients"] = tool_results["tag_recipients_result"]

    return {**guard, **tool_results}


# ── Node 3: Reconciliation ──────────────────────────────────────────────────────

def reconciliation_node(state: AgentState) -> dict:
    """
    Parse MT566 confirmation and detect entitlement breaks.
    Routes to escalation if breaks exceed threshold.
    """
    guard = _increment_and_check(state, "reconciliation_node")
    if "error" in guard:
        return guard

    # If no MT566 provided, mark as pending
    raw_input = state.get("raw_input", "")
    if "566" not in raw_input and "MT566" not in raw_input.upper():
        return {
            **guard,
            "recon_status": "pending",
            "breaks": [],
            "max_break_amount": 0.0,
        }

    tools = TOOL_REGISTRY["reconciliation"]
    llm = get_llm("planner").bind_tools(tools)

    entitlements = state.get("projected_entitlements", [])
    if not entitlements:
        entitlements = state.get("entitlements", [])

    messages = [
        SystemMessage(content="""You are a Corporate Actions reconciliation specialist.

COGNITIVE REASONING REQUIREMENT:
You MUST output a brief, single-sentence explanation in your text response before calling each tool. Do not leave the response content empty when calling tools. For example, explain that you are parsing the incoming MT566 payment confirmation message, that you are comparing projected entitlements with actual values to detect breaks, or that you are generating the reconciliation exception report. Your explanations will be displayed in real-time on the operator console.

1. Call parse_swift_mt566() on the MT566 message to extract confirmed amounts.
2. Call compare_entitlements() to detect breaks between projected and actual.
3. Call generate_recon_report() to produce the exception report.

Use Decimal-safe string values for all amounts.
"""),
        HumanMessage(content=f"""
MT566 Message (in raw_input):
{raw_input}

Projected Entitlements: {json.dumps(entitlements[:10])}
ISIN    : {state.get('isin', '')}
Issuer  : {state.get('issuer', '')}
Pay Date: {state.get('pay_date', '')}
Portfolios: {json.dumps([p.get('portfolio_name', '') for p in state.get('affected_portfolios', [])])}

Run reconciliation now.
""")
    ]

    tool_results = _run_node_llm_loop(llm, tools, messages, "reconciliation_node", state.get("task_id"))

    recon_status = tool_results.get("recon_status", "pending")
    max_break = tool_results.get("max_break_amount", 0.0)
    breaks = tool_results.get("breaks", [])

    if recon_status == "breaks_found" and float(max_break) > BREAK_THRESHOLD:
        if state.get("approval_status") == "approved":
            pass
        else:
            retry_count = state.get("recon_retry_count", 0)
            if retry_count < 1:
                recon_report = tool_results.get("generate_recon_report_result", "No details available.")
                feedback = (
                    f"Reconciliation break found. Max Break: {max_break}. "
                    f"Break details:\n{recon_report}"
                )
                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                log_entry = {
                    "timestamp": timestamp,
                    "node": "reconciliation_node",
                    "thought": f"• Reconciliation detected breaks exceeding threshold ({max_break} > {BREAK_THRESHOLD}). Initiating self-correction retry #1.",
                    "actions": ["compare_entitlements", "trigger_recon_retry"]
                }
                existing_log = list(tool_results.get("audit_log", []))
                existing_log.append(log_entry)
                tool_results["audit_log"] = existing_log

                tool_results["recon_retry_count"] = retry_count + 1
                tool_results["recon_feedback"] = feedback

    return {**guard, **tool_results}


# ── Node 4: Security Master ─────────────────────────────────────────────────────

def security_master_node(state: AgentState) -> dict:
    """
    Validate ISIN identifiers and check Security Master data quality.
    """
    guard = _increment_and_check(state, "security_master_node")
    if "error" in guard:
        return guard

    tools = TOOL_REGISTRY["security_master"]
    llm = get_llm("planner").bind_tools(tools)

    messages = [
        SystemMessage(content="""You are a Security Master data quality specialist.

COGNITIVE REASONING REQUIREMENT:
You MUST output a brief, single-sentence explanation in your text response before calling each tool. Do not leave the response content empty when calling tools. For example, explain that you are validating the security identifiers (ISIN/CUSIP/SEDOL) to ensure checksum integrity, or that you are checking the completeness of the security master record. Your explanations will be displayed in real-time on the operator console.

1. Call validate_security_identifiers() for the ISIN (and CUSIP/SEDOL if available).
2. Call check_security_master_record() to audit the security's data completeness.

Report all issues found and suggest corrective actions.
"""),
        HumanMessage(content=f"""
ISIN   : {state.get('isin', '')}
Issuer : {state.get('issuer', '')}

Run Security Master validation now.
""")
    ]

    tool_results = _run_node_llm_loop(llm, tools, messages, "security_master_node", state.get("task_id"))

    # Consolidate issues from both tools
    issues = []
    fixes = []
    if "issues" in tool_results:
        if isinstance(tool_results["issues"], list):
            issues.extend(tool_results["issues"])
    if "suggested_fixes" in tool_results:
        if isinstance(tool_results["suggested_fixes"], list):
            fixes.extend(tool_results["suggested_fixes"])

    return {
        **guard,
        "security_master_issues": issues,
        "suggested_fixes": fixes,
        **{k: v for k, v in tool_results.items()
           if k not in ("issues", "suggested_fixes", "isin", "cusip", "sedol")}
    }


# ── Node 5: Checking Agent ──────────────────────────────────────────────────────

_SWIFT_TAG_RE = __import__("re").compile(r"^:\w+::")

def checking_node(state: AgentState) -> dict:
    """
    Automated quality gate — runs after planner_node, before notification_node.

    Inspects every key parsed field for raw SWIFT tags (e.g. ':92A::GRSS//18.75')
    that indicate the parser captured a code line instead of a human-readable value.
    Also validates that the issuer name is not a SWIFT tag or empty.

    This node is purely deterministic — no LLM call needed.
    Any failure is logged to the audit trail and escalated.
    """
    guard = _increment_and_check(state, "checking_node")
    if "error" in guard:
        return guard

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    issues_found = []

    # Fields to check for raw SWIFT tag contamination
    fields_to_check = {
        "issuer": state.get("issuer", ""),
        "event_type": state.get("event_type", ""),
        "currency": state.get("currency", ""),
        "pay_date": state.get("pay_date", ""),
        "record_date": state.get("record_date", ""),
        "ex_date": state.get("ex_date", ""),
    }

    for field_name, field_value in fields_to_check.items():
        if not field_value:
            continue
        value_str = str(field_value).strip()
        # Detect raw SWIFT tag pattern like ":92A::" or ":35B:"
        if _SWIFT_TAG_RE.match(value_str):
            issues_found.append(
                f"Field '{field_name}' contains a raw SWIFT tag: '{value_str[:40]}'"
            )
        # Detect issuer that's a numeric/identifier-only line (e.g. "/US/037833100")
        if field_name == "issuer" and value_str.startswith("/"):
            issues_found.append(
                f"Field 'issuer' contains an identifier code line: '{value_str[:40]}'"
            )

    # Compose thought for audit trail
    if issues_found:
        if state.get("approval_status") == "approved":
            thought = (
                "• Checking Agent detected issues in parsed fields post-approval, but bypassing escalation: "
                + "; ".join(issues_found)
            )
            log_entry = {
                "timestamp": timestamp,
                "node": "checking_node",
                "thought": thought,
                "actions": ["field_quality_check", "bypass_escalation_approved"]
            }
            return {
                **guard,
                "audit_log": [log_entry]
            }

        retry_count = state.get("parser_retry_count", 0)
        if retry_count < 2:
            thought = (
                f"• Checking Agent detected issues. Initiating self-correction retry "
                f"#{retry_count + 1}. Issues: " + "; ".join(issues_found)
            )
            log_entry = {
                "timestamp": timestamp,
                "node": "checking_node",
                "thought": thought,
                "actions": ["field_quality_check", "trigger_parser_retry"]
            }
            return {
                **guard,
                "parser_retry_count": retry_count + 1,
                "parser_feedback": "; ".join(issues_found),
                "parse_errors": issues_found,
                "data_quality_warning": True,
                "audit_log": [log_entry]
            }
        else:
            thought = (
                "• Checking Agent detected data quality issues after maximum self-correction retries. "
                "Routing to escalation. Issues: " + "; ".join(issues_found)
            )
            existing_errors = list(state.get("parse_errors") or [])
            log_entry = {
                "timestamp": timestamp,
                "node": "checking_node",
                "thought": thought,
                "actions": ["field_quality_check", "route_to_escalation"]
            }
            return {
                **guard,
                "data_quality_warning": True,
                "parse_errors": existing_errors + issues_found,
                "escalation_reason": f"Checking Agent (Max Retries): {issues_found[0]}",
                "audit_log": [log_entry]
            }

    # All checks passed
    thought = (
        f"• Checking Agent validated all parsed fields successfully. "
        f"Issuer: '{state.get('issuer', 'N/A')}', ISIN: '{state.get('isin', 'N/A')}', "
        f"Event: {state.get('event_type', 'N/A')} ({state.get('event_category', 'N/A')}). "
        f"No raw SWIFT tags detected. Proceeding to notification."
    )
    log_entry = {
        "timestamp": timestamp,
        "node": "checking_node",
        "thought": thought,
        "actions": ["field_quality_check"]
    }
    return {
        **guard,
        "audit_log": [log_entry]
    }


# ── Node 6: Escalation Gate (HITL) ─────────────────────────────────────────────

def escalation_gate_node(state: AgentState) -> dict:
    """
    Human-in-the-Loop checkpoint. Emits a structured escalation alert.

    This node runs AFTER the graph has been interrupted by LangGraph's
    interrupt_before mechanism. By the time this function executes,
    a human has already approved via the API.

    The escalation alert is emitted during the PAUSE (before this node runs).
    This node handles POST-APPROVAL logic.
    """
    guard = _increment_and_check(state, "escalation_gate_node")
    if "error" in guard:
        return guard

    # Emit escalation alert (to console or webhook)
    _emit_escalation_alert(state)

    log_entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "node": "escalation_gate_node",
        "thought": f"• Escalation resolved with status: {state.get('approval_status', 'pending')}. Approved by {state.get('approved_by', 'N/A')}.",
        "actions": ["emit_escalation_alert"]
    }

    return {
        **guard,
        "approval_status": state.get("approval_status", "pending"),
        "audit_log": [log_entry]
    }


def _emit_escalation_alert(state: AgentState) -> None:
    """Print or send escalation alert."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    task_id = state.get("task_id", "UNKNOWN")
    reason = state.get("escalation_reason", "Manual review required")

    alert = f"""
╔══════════════════════════════════════════════════════════════╗
║              ⚠️  CORPORATE ACTIONS ESCALATION ALERT           ║
╠══════════════════════════════════════════════════════════════╣
║ Task ID  : {task_id:<50} ║
║ Time     : {timestamp:<50} ║
║ ISIN     : {state.get('isin', 'N/A'):<50} ║
║ Event    : {state.get('event_type', 'N/A')} - {state.get('issuer', 'N/A'):<44} ║
║ Urgency  : {state.get('urgency', 'N/A').upper():<50} ║
╠══════════════════════════════════════════════════════════════╣
║ REASON: {reason[:54]:<54} ║
╠══════════════════════════════════════════════════════════════╣
║ TO APPROVE: POST /task/{task_id[:36]}/approve              ║
║ TO REJECT : POST /task/{task_id[:36]}/reject               ║
╚══════════════════════════════════════════════════════════════╝
"""
    try:
        print(alert)
    except UnicodeEncodeError:
        print(alert.encode('ascii', errors='replace').decode('ascii'))

    # Optionally send to webhook (Slack, Teams, etc.)
    if ESCALATION_WEBHOOK_URL:
        try:
            import httpx
            httpx.post(ESCALATION_WEBHOOK_URL, json={
                "task_id": task_id,
                "event_type": state.get("event_type"),
                "isin": state.get("isin"),
                "issuer": state.get("issuer"),
                "urgency": state.get("urgency"),
                "reason": reason,
                "approve_url": f"/task/{task_id}/approve",
                "reject_url": f"/task/{task_id}/reject",
            }, timeout=5)
        except Exception:
            pass  # Webhook failure must never block the agent


# ── Node 6: Action Executor ─────────────────────────────────────────────────────

def action_executor_node(state: AgentState) -> dict:
    """
    Final step — generates the processing summary report.
    In production: would write to Aladdin, send election instructions, etc.
    """
    guard = _increment_and_check(state, "action_executor_node")
    if "error" in guard:
        return guard

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    breaks = state.get("breaks", [])
    sm_issues = state.get("security_master_issues", [])
    recon_report = state.get("generate_recon_report_result", "")

    report_lines = [
        f"CORPORATE ACTIONS PROCESSING SUMMARY",
        f"Generated: {timestamp}",
        f"Task ID  : {state.get('task_id', 'N/A')}",
        "=" * 60,
        f"Event    : {state.get('event_type', 'N/A')} — {state.get('issuer', 'N/A')}",
        f"ISIN     : {state.get('isin', 'N/A')}",
        f"Category : {state.get('event_category', 'N/A')}",
        f"Pay Date : {state.get('pay_date', 'N/A')}",
        f"Currency : {state.get('currency', 'N/A')}",
        "",
        "ENTITLEMENTS",
        "-" * 40,
        f"Total Projected: {state.get('currency', '')} {state.get('total_projected', '0')}",
        f"Portfolios     : {len(state.get('affected_portfolios', []))} affected",
        "",
        "RECONCILIATION",
        "-" * 40,
        f"Status: {state.get('recon_status', 'pending').upper()}",
    ]

    if breaks:
        report_lines.append(f"Breaks : {len(breaks)} exception(s) found")
        for b in breaks:
            report_lines.append(
                f"  ⚠️  {b['break_type']}: {b['currency']} {b['break_amount']} "
                f"({b['break_pct']}%) — {b['likely_cause']}"
            )
    else:
        report_lines.append("Breaks : ✅ None — fully reconciled")

    report_lines += ["", "SECURITY MASTER", "-" * 40]
    if sm_issues:
        report_lines.append(f"Issues : {len(sm_issues)} data quality issue(s)")
        for issue in sm_issues:
            report_lines.append(f"  ⚠️  {issue}")
    else:
        report_lines.append("Issues : ✅ No data quality issues")

    if state.get("approval_status") == "approved":
        report_lines += [
            "", "APPROVAL", "-" * 40,
            f"✅ Approved by: {state.get('approved_by', 'Operations Team')}",
            f"   At        : {state.get('approved_at', timestamp)}",
        ]

    report_lines += [
        "", "NODES EXECUTED", "-" * 40,
        " → ".join(state.get("completed_nodes", [])),
        "=" * 60,
        "Processing complete.",
    ]

    log_entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "node": "action_executor_node",
        "thought": "• Generating final processing summary report.",
        "actions": ["generate_final_report"]
    }

    return {
        **guard,
        "final_report": "\n".join(report_lines),
        "audit_log": [log_entry]
    }


# ── Node 7: Error Handler ───────────────────────────────────────────────────────

def error_node(state: AgentState) -> dict:
    """
    Handles failures gracefully — logs error context and produces
    a structured error report so operations can manually complete the task.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    approval_status = state.get("approval_status")
    if approval_status == "rejected":
        report = (
            f"❌ TASK REJECTED BY OPERATOR — {timestamp}\n"
            f"Task ID    : {state.get('task_id', 'N/A')}\n"
            f"ISIN       : {state.get('isin', 'N/A')}\n"
            f"Event      : {state.get('event_type', 'N/A')} — {state.get('issuer', 'N/A')}\n\n"
            f"STATUS     : Aborted / Rejected\n"
            f"Reason     : Action rejected by operator during manual review."
        )
        thought = "• Task rejected by human operator. Aborting execution."
    else:
        report = (
            f"⛔ PROCESSING ERROR — {timestamp}\n"
            f"Task ID    : {state.get('task_id', 'N/A')}\n"
            f"ISIN       : {state.get('isin', 'N/A')}\n"
            f"Failed at  : {state.get('error_node', 'unknown')}\n"
            f"Error      : {state.get('error', 'Unknown error')}\n"
            f"Iterations : {state.get('iteration_count', 0)}\n"
            f"Completed  : {' → '.join(state.get('completed_nodes', []))}\n\n"
            f"ACTION REQUIRED: Manual processing needed. "
            f"Contact CA Operations team with Task ID above."
        )
        thought = f"• Handling error from {state.get('error_node', 'unknown')}: {state.get('error', 'Unknown error')}"

    try:
        print(f"\n{report}\n")
    except UnicodeEncodeError:
        print(f"\n{report.encode('ascii', errors='replace').decode('ascii')}\n")

    log_entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "node": "error_node",
        "thought": thought,
        "actions": ["generate_error_report"]
    }

    return {
        "final_report": report,
        "audit_log": [log_entry]
    }
