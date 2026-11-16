"""Copia los JSON de ACI-Bench al repositorio (data/raw/aci-bench/).

Uso:
  python scripts/import_data.py --src "C:/Users/46743574F/mlops/aci-bench/data/challenge_data_json"
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from acinotes.data import SPLIT_FILES, load_raw  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="carpeta challenge_data_json de aci-bench")
    ap.add_argument("--dst", default=str(ROOT / "data" / "raw" / "aci-bench"))
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    lines = []
    for split, fname in SPLIT_FILES.items():
        s = src / fname
        if not s.exists():
            sys.exit(f"Falta {s}")
        shutil.copy2(s, dst / fname)
        n = len(load_raw(dst / fname))
        lines.append(f"| {split} | {fname} | {n} | {sha256(dst / fname)[:16]} |")
        print(f"{split:6s} {n:4d} conversaciones  <- {fname}")
    (dst / "CHECKSUMS.md").write_text(
        "| split | fichero | nº de ejemplos | sha256 (16 primeros) |\n|---|---|---|---|\n"
        + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
