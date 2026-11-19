"""Àlies curts per als models (qualsevol altre id de Hugging Face també val tal qual)."""

MODEL_ALIASES = {
    "instruct": "Qwen/Qwen3-4B-Instruct-2507",  # sense reasoning
    "thinking": "Qwen/Qwen3-4B-Thinking-2507",  # amb reasoning
}


def resolve_model(name: str) -> str:
    return MODEL_ALIASES.get(name, name)
