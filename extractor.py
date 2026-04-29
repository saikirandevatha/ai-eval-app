import json
import re
from llm_client import chat

PROMPT_VARIANTS = {
    "v1_basic": (
        "Extract entities from this contract. "
        "Return JSON with exactly these keys: parties, dates, amounts, obligations, governing_law. "
        "Return only the JSON object, no extra text."
    ),
    "v2_detailed": (
        "You are a contract analyst. Extract ALL named entities from the contract below.\n\n"
        "Return a JSON object with these keys:\n"
        "- parties: list of strings (all companies, individuals, and organizations)\n"
        "- dates: list of objects with 'label' and 'value' (ISO format where possible)\n"
        "- amounts: list of objects with 'label' and 'value' (keep original currency symbols)\n"
        "- obligations: list of strings (each a distinct duty or requirement)\n"
        "- governing_law: string (jurisdiction/state, or null if not specified)\n\n"
        "Return only the JSON object, no markdown fences, no extra text."
    ),
    "v3_strict": (
        "You are a contract analyst. Extract ALL named entities from the contract below.\n\n"
        "Return a JSON object with these keys:\n"
        "- parties: list of strings (all companies, individuals, and organizations)\n"
        "- dates: list of objects with 'label' and 'value' (ISO format where possible)\n"
        "- amounts: list of objects with 'label' and 'value' (keep original currency symbols)\n"
        "- obligations: list of strings (each a distinct duty or requirement)\n"
        "- governing_law: string (jurisdiction/state, or null if not specified)\n\n"
        "STRICT RULES:\n"
        "- If a field has no data in the contract, return an empty list [] or null.\n"
        "- Never invent, infer, or hallucinate values not explicitly stated in the contract.\n"
        "- Do not include information from outside the contract text.\n"
        "Return only the JSON object, no markdown fences, no extra text."
    ),
}


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def extract_entities(text: str, provider: str, model: str, prompt_variant: str = "v2_detailed") -> dict:
    system_prompt = PROMPT_VARIANTS.get(prompt_variant, PROMPT_VARIANTS["v2_detailed"])
    raw = chat(provider, model, system_prompt, text, max_tokens=1500)

    if raw.startswith("[LLM ERROR]"):
        return {"error": raw, "raw": raw}

    try:
        return json.loads(_strip_fences(raw))
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}", "raw": raw}
