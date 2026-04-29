# AI Eval App — Contract NER with Evals

A Flask web app that extracts named entities from contracts (parties, dates, amounts, obligations, governing law) using LLMs, and runs a structured eval harness to measure precision, recall, and F1 scores across different models and prompts.

**The goal:** understand AI evals hands-on — change a prompt or swap a model, rerun evals, and watch the scores change.

![Python](https://img.shields.io/badge/python-3.10+-blue) ![Flask](https://img.shields.io/badge/flask-3.0-green)

---

## What it does

| Page | URL | Purpose |
|---|---|---|
| Extractor | `/` | Upload a contract PDF/TXT, extract entities as JSON, download as CSV / Excel / Word |
| Eval Dashboard | `/evals` | Run the eval suite against 5 ground-truth contracts, see precision/recall/F1 scores, compare runs |

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/saikirandevatha/ai-eval-app.git
cd ai-eval-app
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Add your API keys

```bash
mkdir -p .claude
cp .env.example .claude/.env
# Now edit .claude/.env and fill in your keys
```

You need at least one of:
- **OpenAI API key** — from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **OpenRouter API key** — from [openrouter.ai/keys](https://openrouter.ai/keys) (has a free tier with Mistral, Llama 3, Gemma)

### 3. Run

```bash
python app.py
# Open http://127.0.0.1:5001
```

---

## How evals work

The eval harness (`eval_runner.py`) runs 5 built-in contracts with known ground truth answers through the LLM and scores the output:

| Metric | What it measures |
|---|---|
| **Precision** | Of entities the model found, what % were correct? (hallucination check) |
| **Recall** | Of entities that exist, what % did the model find? (miss check) |
| **F1** | Harmonic mean of precision + recall — the overall accuracy score |
| **Schema compliance** | Did the model return valid JSON every time? |

### The eval loop

1. Go to `/evals`
2. Pick `v1_basic` prompt + any model → Run Evals → note F1 scores
3. Switch to `v3_strict` prompt → Run Evals again → compare scores
4. Try a different model → compare again
5. All runs are logged to `eval_logs.jsonl` — click **Raw Output** to see exactly what the model returned

---

## Prompt variants

Three prompts in `extractor.py` to experiment with:

| Variant | Description |
|---|---|
| `v1_basic` | Minimal instruction — just asks for JSON with the right keys |
| `v2_detailed` | Detailed field descriptions and formatting rules |
| `v3_strict` | Same as v2 + explicit rules against hallucination and invention |

---

## Models supported

**OpenAI (direct):** gpt-4.1-nano, gpt-4.1-mini, gpt-4.1, gpt-4o-mini, gpt-4o, gpt-5, o1-mini, o3-mini

**OpenRouter (free tier):** Mistral 7B, Llama 3 8B, Gemma 3 12B

**OpenRouter (paid):** Claude 3.5 Haiku, Claude Sonnet 4.5, Gemini 2.0 Flash, GPT-4o-mini

---

## Project structure

```
ai-eval-app/
├── app.py               # Flask routes
├── llm_client.py        # Unified chat() for OpenAI + OpenRouter
├── extractor.py         # NER logic + 3 prompt variants
├── eval_runner.py       # Eval harness (web + CLI)
├── eval_dataset.json    # 5 ground-truth-labeled test contracts
├── requirements.txt
├── .env.example         # Copy to .claude/.env and add your keys
└── templates/
    ├── index.html       # Extractor UI
    └── evals.html       # Eval dashboard
```

---

## CLI usage

Run evals from the terminal without starting the server:

```bash
# python eval_runner.py <provider> <model> <prompt_variant>
python eval_runner.py openai gpt-4o-mini v2_detailed
python eval_runner.py openrouter mistralai/mistral-7b-instruct v3_strict
```

Results are printed as a table and saved to `eval_results.json`.
