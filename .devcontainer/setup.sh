#!/bin/bash
set -e

echo "==> Creating virtual environment..."
python -m venv .venv

echo "==> Installing dependencies into .venv..."
.venv/bin/pip install --upgrade pip --quiet
.venv/bin/pip install -r requirements.txt --quiet

# Create the ca_agent/.env configuration template if it doesn't exist
if [ ! -f ca_agent/.env ]; then
  echo "==> Generating .env template..."
  cat > ca_agent/.env << 'EOF'
LLM_PROVIDER=openai
OPENAI_API_KEY=your-api-key-here
EOF
  echo "ACTION REQUIRED: Open ca_agent/.env and set your API key before running."
fi

echo "==> Setup complete. Server will auto-start on Codespace launch."
