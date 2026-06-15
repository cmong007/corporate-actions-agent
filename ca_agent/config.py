"""Central configuration — reads from environment variables."""
import os
from dotenv import load_dotenv

load_dotenv()

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "gpt-4o-mini")
SPECIALIST_MODEL = os.getenv("SPECIALIST_MODEL", "gpt-4o")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "hermes3")

# Agent control
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "20"))
BREAK_THRESHOLD = float(os.getenv("BREAK_THRESHOLD", "1000"))
CRITICAL_DEADLINE_HOURS = int(os.getenv("CRITICAL_DEADLINE_HOURS", "48"))

# Persistence
CHECKPOINT_DB_PATH = os.getenv("CHECKPOINT_DB_PATH", "ca_agent_checkpoints.db")

# Escalation
ESCALATION_WEBHOOK_URL = os.getenv("ESCALATION_WEBHOOK_URL", "")

# Data paths
POSITIONS_FILE = os.getenv("POSITIONS_FILE", "ca_agent/data/sample_positions.csv")
SECURITY_MASTER_FILE = os.getenv("SECURITY_MASTER_FILE", "ca_agent/data/security_master.csv")

# Active reasoning storage for live UI trail updates (task_id -> active_log_entry)
active_reasoning = {}

