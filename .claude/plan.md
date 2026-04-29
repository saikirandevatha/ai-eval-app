# Plan: Contract NER Chatbot with AI Evals

## Context
User wants to understand what AI evals are in practice — specifically how evals change
when you vary the prompt, the model, or other conditions, and how you measure accuracy/
recall/precision on LLM outputs. We're building a Flask web app that extracts named
entities from uploaded contracts (parties, dates, amounts, obligations, governing law),
then running a structured eval harness that makes these measurement concepts tangible.

The central learning goal: **see evals as a feedback loop** — change a prompt or swap a
model, rerun evals, observe how scores move.

## Stack
- Flask + Vanilla JS
- OpenAI SDK for both OpenAI and OpenRouter (same API shape)
- pypdf for PDF parsing
- openpyxl + python-docx for exports

## Key files
- `app.py` — Flask routes
- `llm_client.py` — unified chat() for both providers
- `extractor.py` — NER logic + 3 named prompt variants
- `eval_runner.py` — eval harness, logs to eval_logs.jsonl
- `eval_dataset.json` — 5 ground-truth-labeled contracts
- `templates/index.html` — upload + extract UI
- `templates/evals.html` — eval dashboard with run history

## Setup
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Add keys to .claude/.env
python app.py  # runs on port 5001
```

## .claude/.env format
```
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...
FLASK_SECRET=your-secret-key
```
