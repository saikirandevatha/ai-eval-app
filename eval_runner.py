import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from extractor import extract_entities  # used by CLI only

LOG_PATH = Path("eval_logs.jsonl")


def _append_log(run: dict):
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(run) + "\n")


def load_logs() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    logs = []
    for line in LOG_PATH.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return list(reversed(logs))  # newest first

PASS_F1_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[$,]", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def _tokens(s: str) -> set:
    return set(_norm(s).split())


def _fuzzy_match(pred: str, truth: str, threshold: float = 0.6) -> bool:
    pt, tt = _tokens(pred), _tokens(truth)
    if not tt:
        return False
    overlap = len(pt & tt) / len(tt)
    return overlap >= threshold


def _extract_value(item) -> str:
    if isinstance(item, dict):
        return str(item.get("value", ""))
    return str(item)


# ---------------------------------------------------------------------------
# Per-field precision / recall / F1
# ---------------------------------------------------------------------------

def _score_list_field(predicted: list, truth: list) -> dict:
    if not truth and not predicted:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not truth:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    if not predicted:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0}

    pred_vals = [_extract_value(p) for p in predicted]
    truth_vals = [_extract_value(t) for t in truth]

    tp_pred = sum(1 for p in pred_vals if any(_fuzzy_match(p, t) for t in truth_vals))
    tp_truth = sum(1 for t in truth_vals if any(_fuzzy_match(p, t) for p in pred_vals))

    precision = tp_pred / len(pred_vals)
    recall = tp_truth / len(truth_vals)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def _score_scalar_field(predicted, truth) -> dict:
    if truth is None and (predicted is None or predicted == ""):
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if truth is None:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    if predicted is None or predicted == "":
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0}
    match = _fuzzy_match(str(predicted), str(truth))
    score = 1.0 if match else 0.0
    return {"precision": score, "recall": score, "f1": score}


def _score_case(predicted: dict, ground_truth: dict) -> dict:
    list_fields = ["parties", "dates", "amounts", "obligations"]
    scalar_fields = ["governing_law"]
    fields = {}

    for field in list_fields:
        pred = predicted.get(field, []) if isinstance(predicted.get(field), list) else []
        truth = ground_truth.get(field, [])
        fields[field] = _score_list_field(pred, truth)

    for field in scalar_fields:
        fields[field] = _score_scalar_field(predicted.get(field), ground_truth.get(field))

    all_f1 = [v["f1"] for v in fields.values()]
    overall_f1 = round(sum(all_f1) / len(all_f1), 3)
    return {"fields": fields, "overall_f1": overall_f1}


# ---------------------------------------------------------------------------
# Main eval function
# ---------------------------------------------------------------------------

def run_evals(
    dataset_path: str = "eval_dataset.json",
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    prompt_variant: str = "v2_detailed",
) -> dict:
    dataset = json.loads(Path(dataset_path).read_text())
    results = []

    from llm_client import chat
    from extractor import PROMPT_VARIANTS, _strip_fences

    for case in dataset["cases"]:
        system_prompt = PROMPT_VARIANTS.get(prompt_variant, PROMPT_VARIANTS["v2_detailed"])
        raw_output = chat(provider, model, system_prompt, case["document"], max_tokens=1500)

        # Parse from raw output directly — avoids a second LLM call
        try:
            predicted = json.loads(_strip_fences(raw_output))
        except (json.JSONDecodeError, Exception) as e:
            predicted = {"error": f"JSON parse error: {e}", "raw": raw_output}
        schema_valid = (
            "error" not in predicted
            and all(k in predicted for k in ["parties", "dates", "amounts", "obligations", "governing_law"])
        )

        if not schema_valid:
            result = {
                "id": case["id"],
                "name": case["name"],
                "schema_valid": False,
                "raw_output": raw_output,
                "error": predicted.get("error", "missing keys"),
                "fields": {f: {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                           for f in ["parties", "dates", "amounts", "obligations", "governing_law"]},
                "overall_f1": 0.0,
                "pass": False,
            }
        else:
            scores = _score_case(predicted, case["ground_truth"])
            result = {
                "id": case["id"],
                "name": case["name"],
                "schema_valid": True,
                "raw_output": raw_output,
                "predicted": predicted,
                "fields": scores["fields"],
                "overall_f1": scores["overall_f1"],
                "pass": scores["overall_f1"] >= PASS_F1_THRESHOLD,
            }
        results.append(result)

    valid = [r for r in results if r["schema_valid"]]
    field_names = ["parties", "dates", "amounts", "obligations", "governing_law"]

    def avg(metric):
        vals = [r["fields"][f][metric] for r in valid for f in field_names]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    passed = sum(1 for r in results if r["pass"])
    summary = {
        "avg_precision": avg("precision"),
        "avg_recall": avg("recall"),
        "avg_f1": avg("f1"),
        "schema_compliance": f"{len(valid)}/{len(results)}",
        "pass_rate": f"{passed}/{len(results)}",
    }

    run = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {"provider": provider, "model": model, "prompt_variant": prompt_variant},
        "results": results,
        "summary": summary,
    }
    _append_log(run)
    return run


# ---------------------------------------------------------------------------
# CLI pretty-print
# ---------------------------------------------------------------------------

def _print_results(data: dict):
    cfg = data["config"]
    print(f"\n=== EVAL RESULTS | {cfg['provider']} / {cfg['model']} / prompt:{cfg['prompt_variant']} ===\n")

    header = f"{'Contract':<35} {'Schema':^7} {'Parties':^8} {'Dates':^8} {'Amounts':^8} {'Obligations':^12} {'GovLaw':^8} {'F1':^6} {'Pass':^5}"
    print(header)
    print("-" * len(header))

    for r in data["results"]:
        f = r["fields"]
        row = (
            f"{r['name'][:34]:<35}"
            f" {'✓' if r['schema_valid'] else '✗':^7}"
            f" {f['parties']['f1']:^8.2f}"
            f" {f['dates']['f1']:^8.2f}"
            f" {f['amounts']['f1']:^8.2f}"
            f" {f['obligations']['f1']:^12.2f}"
            f" {f['governing_law']['f1']:^8.2f}"
            f" {r['overall_f1']:^6.2f}"
            f" {'✓' if r['pass'] else '✗':^5}"
        )
        print(row)

    s = data["summary"]
    print(f"\nSummary: Precision={s['avg_precision']}  Recall={s['avg_recall']}  "
          f"F1={s['avg_f1']}  Schema={s['schema_compliance']}  Pass={s['pass_rate']}\n")


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=".claude/.env")

    provider = sys.argv[1] if len(sys.argv) > 1 else "openai"
    model = sys.argv[2] if len(sys.argv) > 2 else "gpt-4o-mini"
    variant = sys.argv[3] if len(sys.argv) > 3 else "v2_detailed"

    data = run_evals(provider=provider, model=model, prompt_variant=variant)
    _print_results(data)

    out_path = "eval_results.json"
    Path(out_path).write_text(json.dumps(data, indent=2))
    print(f"Results saved to {out_path}")
