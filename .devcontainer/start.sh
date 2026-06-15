#!/bin/bash
# Runs automatically every time the Codespace starts or resumes.

VENV=".venv"
ENV_FILE="ca_agent/.env"

# Warn if the API key hasn't been set yet
if grep -q "your-api-key-here" "$ENV_FILE" 2>/dev/null; then
  echo ""
  echo "========================================================"
  echo " WARNING: API key not set in $ENV_FILE"
  echo " Open the file and replace 'your-api-key-here' with your"
  echo " actual OpenAI (or other provider) API key, then run:"
  echo "   uvicorn ca_agent.main:app --reload"
  echo "========================================================"
  echo ""
  exit 0
fi

# Activate venv and start the server
echo "==> Activating virtual environment..."
source "$VENV/bin/activate"

echo "==> Starting Corporate Actions Agent on http://localhost:8000"
exec uvicorn ca_agent.main:app --host 0.0.0.0 --port 8000 --reload
