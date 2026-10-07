# LangGraph First Agent

A minimal LangGraph agent starter project.

## Setup

1. Create a virtual environment:
   python3 -m venv .venv
   source .venv/bin/activate
2. Install dependencies:
   pip install -r requirements.txt
3. Copy the example environment file and add your OpenAI API key:
   cp .env.example .env

## Run

python app.py "Tell me a short joke."

If `OPENAI_API_KEY` is not set, the app falls back to a local response so the app can still run.
