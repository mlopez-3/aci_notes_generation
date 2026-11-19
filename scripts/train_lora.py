"""Entrena un adaptador LoRA (QLoRA) sobre el model SENSE reasoning i el desa a adapters/<id>/.

Exemples:
  python scripts/train_lora.py --smoke                  # PROVA DE FUM: 4 exemples, 1 època. Mesura memòria i temps
  python scripts/train_lora.py                          # entrenament complet (3 èpoques, r=16, atenció)
  python scripts/train_lora.py --r 8 --epochs 2         # altres hiperparàmetres (queden al nom de l'adaptador)
  python scripts/train_lora.py --resume <id-adaptador>  # continua des de l'última època desada

Després, per generar amb la millor època:
  python scripts/generate.py --adapter <id-adaptador> --split valid
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from acinotes.models import resolve_model  # noqa: E402
from acinotes.train_lora import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="instruct", help="àlies (instruct) o id de Hugging Face; no el model Thinking")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=4, help="exemples per pas de l'optimitzador")
    ap.add_argument("--max-len", type=int, default=4096, help="tokens màxims (prompt + nota); els més llargs s'exclouen")
    ap.add_argument("--r", type=int, default=16, help="rang LoRA")
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--targets", choices=["attn", "all"], default="attn",
                    help="attn = q,k,v,o (pla acordat); all = també el MLP (més paràmetres)")
    ap.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="auto",
                    help="auto: bf16 si la GPU el suporta (L4), si no fp16 (T4)")
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-eval-before", action="store_true", help="no calculis la pèrdua de valid abans d'entrenar")
    ap.add_argument("--resume", default=None, metavar="ID", help="id (o ruta) d'un adaptador existent per continuar")
    ap.add_argument("--smoke", action="store_true", help="prova de fum: 4 exemples de train, 2 de valid, 1 època")
    ap.add_argument("--stamp", default=None, help="força el stamp de l'id (per defecte, ara)")
    ap.add_argument("--adapters-dir", default=str(ROOT / "adapters"))
    a = ap.parse_args()
    a.model = resolve_model(a.model)
    run(a)


if __name__ == "__main__":
    main()
