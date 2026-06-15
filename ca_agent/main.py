"""FastAPI application — REST interface for the CA agent."""
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ca_agent.graph.graph import graph

# Global set of running task IDs to distinguish active execution from HITL pauses.
running_tasks = set()


def run_graph_in_background(task_id: str, state_input: dict | None, config: dict):
    """Executes the graph in the background and removes the task ID when done."""
    try:
        graph.invoke(state_input, config=config)
    except Exception as e:
        import traceback
        print(f"Error executing graph in background for task {task_id}: {e}")
        traceback.print_exc()
    finally:
        running_tasks.discard(task_id)


app = FastAPI(
    title="Corporate Actions Processing Agent",
    description="Agentic CA processing with Human-in-the-Loop escalation",
    version="1.0.0",
)


class TaskRequest(BaseModel):
    raw_input: str
    input_source: str = "MT564"  # "MT564" | "MT566" | "CSV" | "manual"


class ApprovalRequest(BaseModel):
    approved_by: str = "Operations Team"


# ── Process a new corporate action event ────────────────────────────────────────

@app.post("/process-event")
def process_event(request: TaskRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())

    initial_state = {
        "task_id": task_id,
        "raw_input": request.raw_input,
        "input_source": request.input_source,
        "iteration_count": 0,
        "completed_nodes": [],
    }

    config = {"configurable": {"thread_id": task_id}}

    running_tasks.add(task_id)
    background_tasks.add_task(run_graph_in_background, task_id, initial_state, config=config)

    return {
        "task_id": task_id,
        "status": "processing",
        "paused_at": [],
        "escalation_reason": "",
        "isin": "",
        "event_type": "",
        "urgency": "",
        "final_report": "",
    }


# ── Poll task status ─────────────────────────────────────────────────────────────

@app.get("/task/{task_id}")
def get_task(task_id: str):
    config = {"configurable": {"thread_id": task_id}}
    try:
        state = graph.get_state(config)
    except Exception:
        if task_id in running_tasks:
            return {
                "task_id": task_id,
                "status": "processing",
                "paused_at": [],
                "approval_status": "",
                "recon_status": "",
                "isin": "",
                "event_type": "",
                "completed_nodes": [],
                "final_report": "",
                "error": "",
                "audit_log": [],
                "escalation_reason": "",
            }
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if not state or not state.values:
        if task_id in running_tasks:
            return {
                "task_id": task_id,
                "status": "processing",
                "paused_at": [],
                "approval_status": "",
                "recon_status": "",
                "isin": "",
                "event_type": "",
                "completed_nodes": [],
                "final_report": "",
                "error": "",
                "audit_log": [],
                "escalation_reason": "",
            }
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    is_running = task_id in running_tasks
    completed_nodes = state.values.get("completed_nodes", [])
    
    # We are paused if it's not currently running and the next node is escalation_gate_node
    is_paused = not is_running and bool(state.next) and "escalation_gate_node" in state.next

    if is_running:
        status = "processing"
    elif is_paused:
        status = "paused_for_approval"
    elif state.values.get("approval_status") == "rejected":
        status = "rejected"
    elif "action_executor_node" in completed_nodes or "error_node" in completed_nodes:
        status = "complete"
    else:
        status = "processing"

    audit_log = list(state.values.get("audit_log", []))
    from ca_agent.config import active_reasoning
    if task_id in active_reasoning:
        active_entry = active_reasoning[task_id]
        if not audit_log or audit_log[-1].get("node") != active_entry.get("node"):
            audit_log.append(active_entry)

    return {
        "task_id": task_id,
        "status": status,
        "paused_at": list(state.next) if is_paused else [],
        "approval_status": state.values.get("approval_status", ""),
        "recon_status": state.values.get("recon_status", ""),
        "isin": state.values.get("isin", ""),
        "event_type": state.values.get("event_type", ""),
        "completed_nodes": completed_nodes,
        "final_report": state.values.get("final_report", ""),
        "error": state.values.get("error", ""),
        "audit_log": audit_log,
        "escalation_reason": state.values.get("escalation_reason", ""),
    }


# ── Approve a paused escalation ──────────────────────────────────────────────────

@app.post("/task/{task_id}/approve")
def approve_task(task_id: str, background_tasks: BackgroundTasks, body: ApprovalRequest = ApprovalRequest()):
    config = {"configurable": {"thread_id": task_id}}

    try:
        current = graph.get_state(config)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    # Guard against active execution
    if task_id in running_tasks:
        raise HTTPException(status_code=400, detail="Task is currently running")

    # Idempotency check
    current_approval = current.values.get("approval_status", "")
    if current_approval == "approved":
        return {"task_id": task_id, "status": "already_approved", "message": "Task was already approved"}

    if not current.next or "escalation_gate_node" not in current.next:
        return {"task_id": task_id, "status": "not_paused", "message": "Task is not waiting for approval"}

    # Inject approval into state
    timestamp = datetime.now(timezone.utc).isoformat()
    graph.update_state(
        config,
        {
            "approval_status": "approved",
            "approved_by": body.approved_by,
            "approved_at": timestamp,
        },
    )

    # Resume graph from checkpoint in background task
    running_tasks.add(task_id)
    background_tasks.add_task(run_graph_in_background, task_id, None, config=config)

    return {
        "task_id": task_id,
        "status": "processing",
        "approved_by": body.approved_by,
        "approved_at": timestamp,
        "final_report": "",
    }


# ── Reject a paused escalation ───────────────────────────────────────────────────

@app.post("/task/{task_id}/reject")
def reject_task(task_id: str, background_tasks: BackgroundTasks, body: ApprovalRequest = ApprovalRequest()):
    config = {"configurable": {"thread_id": task_id}}

    try:
        current = graph.get_state(config)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    # Guard against active execution
    if task_id in running_tasks:
        raise HTTPException(status_code=400, detail="Task is currently running")

    if current.values.get("approval_status") == "approved":
        raise HTTPException(status_code=409, detail="Task already approved — cannot reject")

    if not current.next or "escalation_gate_node" not in current.next:
        raise HTTPException(status_code=400, detail="Task is not waiting for rejection")

    timestamp = datetime.now(timezone.utc).isoformat()
    graph.update_state(
        config,
        {
            "approval_status": "rejected",
            "approved_by": body.approved_by,
            "approved_at": timestamp,
            "escalation_reason": "Action REJECTED by " + body.approved_by,
        },
    )

    running_tasks.add(task_id)
    background_tasks.add_task(run_graph_in_background, task_id, None, config=config)

    return {
        "task_id": task_id,
        "status": "processing",
        "rejected_by": body.approved_by,
        "message": "Task rejected. Manual processing required.",
        "final_report": "",
    }


# ── Health check ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/", response_class=HTMLResponse)
def get_index():
    import os
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/templates")
def get_templates():
    import os
    templates = {}
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    try:
        s1 = open(os.path.join(data_dir, "sample_mt564_dividend.txt")).read() + "\n\n--- MT566 CONFIRMATION ---\n\n" + open(os.path.join(data_dir, "sample_mt566_clean.txt")).read()
    except Exception:
        s1 = ""
    try:
        s2 = open(os.path.join(data_dir, "sample_mt564_tender.txt")).read()
    except Exception:
        s2 = ""
    try:
        s3 = open(os.path.join(data_dir, "sample_mt564_dividend.txt")).read() + "\n\n--- MT566 CONFIRMATION (WITH BREAK) ---\n\n" + open(os.path.join(data_dir, "sample_mt566_break.txt")).read()
    except Exception:
        s3 = ""
        
    s4 = """{1:F01CITIUS33AXXX0000000000}{4:
:22F::CAEV//DVCA
:22F::CAMV//MAND
:35B:ISIN US0378331006
Apple Inc (INVALID CHECKSUM)
:92A::GRSS//0.25
:22F::CURR//USD
:98A::PAYD//20260710
-}"""

    s5 = """:22F::CAEV//DVCA
:22F::CAMV//MAND
:35B:ISIN XS2345678903
Volkswagen 1.875% 2028
:92A::GRSS//18.75
:22F::CURR//EUR
:98A::PAYD//20260715
:98A::RDDT//20260710
:98A::EXDT//20260709"""

    return {
        "scenario1": s1,
        "scenario2": s2,
        "scenario3": s3,
        "scenario4": s4,
        "scenario5": s5,
    }
