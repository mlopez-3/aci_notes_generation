"""Prompts per generar la nota clínica. Un "system" (zeroshot, fewshot...) = una funció que construeix els missatges."""
from __future__ import annotations

SYSTEM = (
    "You are a clinical documentation assistant. You write a structured clinical note "
    "from a doctor-patient conversation."
)

INSTRUCTIONS = """Write a clinical note from the conversation below.

Use exactly these section headers, in this order, each on its own line in capital letters:
CHIEF COMPLAINT
HISTORY OF PRESENT ILLNESS
REVIEW OF SYSTEMS
PHYSICAL EXAM
RESULTS
ASSESSMENT AND PLAN

Rules:
- Use ONLY information stated in the conversation. Do not add facts, numbers, medications or findings that are not there.
- If the conversation gives no information for a section, omit that section entirely.
- Write in third person, in standard clinical style. Keep negations exactly as stated.
- Include every relevant symptom, finding, medication, dose, test result and plan item mentioned.

CONVERSATION:
{src}

CLINICAL NOTE:"""


def build_zeroshot(src: str) -> list[dict]:
    """Zero-shot: només instruccions + conversa, sense cap exemple."""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": INSTRUCTIONS.format(src=src)},
    ]


# Per afegir una estratègia nova (p. ex. fewshot), escriu la funció i registra-la aquí:
# després ja es pot triar amb --system.
PROMPT_BUILDERS = {
    "zeroshot": build_zeroshot,
}
