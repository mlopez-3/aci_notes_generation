"""Trazabilidad: guardar con cada predicción o métrica cómo se obtuvo (configuración y versión del código)."""
from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path


def git_commit(repo_dir: str | Path = ".") -> dict:
    """Commit actual y si hay cambios sin guardar. {} si no es un repositorio git."""
    def run(*args):
        return subprocess.run(["git", "-C", str(repo_dir), *args], capture_output=True, text=True, timeout=10)
    try:
        h = run("rev-parse", "HEAD")
        if h.returncode != 0:
            return {}
        dirty = run("status", "--porcelain").stdout.strip() != ""
        return {"commit": h.stdout.strip(), "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {}


def write_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_meta(path: str | Path, config: dict, repo_dir: str | Path = ".") -> Path:
    """Escribe un .meta.json con la configuración recibida + fecha + commit del código."""
    payload = {"created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
               "code_version": git_commit(repo_dir), "config": config}
    return write_json(path, payload)
