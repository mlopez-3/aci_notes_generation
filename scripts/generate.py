"""Genera notes clíniques amb un LLM i les desa a predictions/ amb el nom <split>__<system>__<model>__<variant>__<stamp>.jsonl.

Exemples:
  python scripts/generate.py --n 2                                              # prova ràpida: Qwen3-4B-Instruct-2507 (sense reasoning)
  python scripts/generate.py --model instruct                                   # el mateix, amb àlies
  python scripts/generate.py --model thinking --split valid                     # Qwen3-4B-Thinking-2507 (amb reasoning)
  python scripts/generate.py --model Qwen/Qwen3-32B                             # híbrid: genera sense i amb reasoning
  python scripts/generate.py --adapter <id-adaptador> --split valid             # amb l'adaptador LoRA (millor època)
  python scripts/generate.py --stamp 20261117T0900Z --resume                    # reprèn una execució tallada
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from acinotes.data import SPLIT_FILES  # noqa: E402
from acinotes.generate import generate_predictions, load_model, thinking_modes  # noqa: E402
from acinotes.models import resolve_model  # noqa: E402
from acinotes.naming import resolve_adapter, utc_stamp  # noqa: E402
from acinotes.prompts import PROMPT_BUILDERS  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"  # només sense reasoning; el reasoning és Qwen/Qwen3-4B-Thinking-2507


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="àlies (instruct, thinking) o id de Hugging Face del model")
    ap.add_argument("--split", default="valid", choices=sorted(SPLIT_FILES))
    ap.add_argument("--system", default="zeroshot", choices=sorted(PROMPT_BUILDERS), help="estratègia de prompt")
    ap.add_argument("--thinking", default=None, choices=["off", "on", "both"],
                    help="reasoning: off, on o both (una predicció per mode). Per defecte, els modes que admet el model")
    ap.add_argument("--n", type=int, default=None, help="limita el nombre d'exemples (per provar)")
    ap.add_argument("--quant", default="4bit", choices=["none", "8bit", "4bit"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16"])
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="per defecte 1300 sense reasoning i 8192 amb reasoning")
    ap.add_argument("--temperature", type=float, default=None,
                    help="per defecte 0 (greedy) sense reasoning i 0.6 amb reasoning")
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--repetition-penalty", type=float, default=None,
                    help="p. ex. 1.05-1.1; penalitza repetir tokens (contra els bucles). Per defecte, sense penalització")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adapter", default=None,
                    help="adaptador LoRA: id (adapters/<id>, usa 'best'), <id>/epoch_2 o ruta. La predicció es diu 'lora'")
    ap.add_argument("--adapters-dir", default=str(ROOT / "adapters"))
    ap.add_argument("--stamp", default=None, help="força el stamp del nom (necessari per --resume)")
    ap.add_argument("--resume", action="store_true", help="continua una predicció existent amb el mateix --stamp")
    ap.add_argument("--predictions-dir", default=str(ROOT / "predictions"))
    a = ap.parse_args()
    a.model = resolve_model(a.model)
    adapter = resolve_adapter(a.adapter, a.adapters_dir) if a.adapter else None
    if a.resume and not a.stamp:
        ap.error("--resume necessita --stamp")

    stamp = a.stamp or utc_stamp()  # el mateix stamp per als dos modes d'una execució
    try:
        modes = thinking_modes(a.model, a.thinking)  # falla abans de carregar el model si no és compatible
    except ValueError as e:
        ap.error(str(e))

    tok, model, dtype_name = load_model(a.model, a.quant, a.dtype, adapter.weights if adapter else None)  # es carrega una sola vegada
    for thinking in modes:
        generate_predictions(
            tok, model, model_name=a.model, split=a.split, thinking=thinking, quant=a.quant,
            dtype_name=dtype_name, system=a.system, n=a.n,
            max_new_tokens=a.max_new_tokens or (8192 if thinking else 1300),
            temperature=a.temperature, top_p=a.top_p, top_k=a.top_k, seed=a.seed,
            repetition_penalty=a.repetition_penalty,
            stamp=stamp, resume=a.resume, adapter=adapter,
            predictions_dir=a.predictions_dir, repo_dir=ROOT)


if __name__ == "__main__":
    main()
