#!/bin/bash
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create the ca_agent/.env configuration template if it doesn't exist
if [ ! -f ca_agent/.env ]; then
  echo "Generating environment configuration template..."
  echo "LLM_PROVIDER=openai" > ca_agent/.env
  echo "OPENAI_API_KEY=your-api-key-here" >> ca_agent/.env
fi

echo "Codespace ready. Run: uvicorn ca_agent.main:app --reload"
