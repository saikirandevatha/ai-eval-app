import os
from openai import OpenAI


def chat(provider: str, model: str, system_prompt: str, user_message: str, max_tokens: int = 1024) -> str:
    try:
        if provider == "openai":
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        elif provider == "openrouter":
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        else:
            return f"[LLM ERROR] Unknown provider: {provider}"

        system_role = "developer" if model.startswith(("o1", "o3")) else "system"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": system_role, "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"[LLM ERROR] {e}"
