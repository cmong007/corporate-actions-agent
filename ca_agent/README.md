# Corporate Actions Processing Agent

A domain-specific, production-quality AI agent that automates the core responsibilities of a **Corporate Actions Operations Analyst** at an asset management firm.

Built in Python using LangGraph with a model-agnostic LLM layer, SQLite-backed checkpoint persistence, and Human-in-the-Loop (HITL) approval gates for all high-risk actions.

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and set your LLM_PROVIDER and API key
```

### 3. Run the API server
```bash
uvicorn ca_agent.main:app --reload
```

---

## Demo Scenarios

### Run all UAT scenarios
```bash
python -m ca_agent.tests.uat.run_uat --all --verbose
```

### Run a specific scenario
```bash
python -m ca_agent.tests.uat.run_uat --scenario 1  # Mandatory cash dividend
python -m ca_agent.tests.uat.run_uat --scenario 2  # Voluntary tender offer (HITL)
python -m ca_agent.tests.uat.run_uat --scenario 3  # Reconciliation break detection
```

---

## Unit Tests
```bash
pytest ca_agent/tests/test_tools.py -v
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/process-event` | POST | Submit a CA event for processing |
| `/task/{id}` | GET | Poll for task status |
| `/task/{id}/approve` | POST | Approve a pending escalation (HITL) |
| `/task/{id}/reject` | POST | Reject a pending escalation |
| `/health` | GET | Liveness check |

---

## Architecture

```
POST /process-event
        │
  planner_node          ← Parses SWIFT MT564, classifies event, calculates entitlements
        │
  notification_node     ← Drafts internal notification, tags recipients
        │
  ┌─────┴──────┐
  │            │
recon_node   escalation_gate ⏸  ← HITL: pauses for human approval
  │            │
security_    action_executor
master_node
  │
action_executor_node    ← Generates final processing report
```

---

## Key Design Decisions

- **Decimal arithmetic** — all monetary calculations use `decimal.Decimal`, never `float`
- **ISIN checksum validation** — ISO 6166 Luhn algorithm validated on every ISIN
- **SQLite checkpointing** — graph state persists across server restarts
- **Dynamic tool loading** — each node only loads its designated tools (token efficiency)
- **Model cascading** — cheap model for routing/drafting, specialist model for complex reasoning
- **MAX_ITERATIONS guard** — every node checks the counter to prevent infinite loops
- **Idempotent approve endpoint** — double-approval returns 200 without re-executing
