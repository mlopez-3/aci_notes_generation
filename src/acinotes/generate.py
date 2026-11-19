"""Generació de notes clíniques amb un LLM de Hugging Face (zero-shot, amb o sense reasoning).

Les funcions pures (split_reasoning, sampling_kwargs...) no necessiten torch, així es poden provar a part.
torch i transformers només s'importen quan es carrega el model.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import naming
from .data import load_split
from .prompts import PROMPT_BUILDERS
from .provenance import write_meta

THINK_END = "</think>"
THINK_START = "<think>"


def split_reasoning(text: str, thinking: bool) -> tuple[str, str, bool]:
    """Separa el text generat en (nota, reasoning, truncat).

    truncat = True si el mode thinking s'ha quedat sense tokens abans de tancar </think>:
    en aquest cas no hi ha nota i es deixa buida (el reasoning no és la nota).
    """
    if THINK_END in text:
        reasoning, note = text.split(THINK_END, 1)
        return note.strip(), reasoning.replace(THINK_START, "").strip(), False
    if thinking:
        return "", text.replace(THINK_START, "").strip(), True
    return text.strip(), "", False


def thinking_modes(model_name: str, requested: str | None = None) -> list[bool]:
    """Modes de reasoning (False = sense, True = amb) que admet el model.

    - Qwen3-*-Instruct-2507: només sense reasoning (no té mode thinking).
    - Qwen3-*-Thinking-2507: sempre raona (la plantilla ja obre <think>), només "amb".
    - Resta (p. ex. Qwen3-32B, híbrid): els dos modes.
    requested: None (el per defecte del model), "off", "on" o "both". Si no el suporta, error.
    """
    name = model_name.lower()
    if "instruct-2507" in name:
        allowed = [False]
    elif "thinking-2507" in name:
        allowed = [True]
    else:
        allowed = [False, True]
    wanted = {None: allowed, "both": [False, True], "off": [False], "on": [True]}[requested]
    if requested is not None and not set(wanted) <= set(allowed):
        other = "Qwen/Qwen3-4B-Thinking-2507" if allowed == [False] else "Qwen/Qwen3-4B-Instruct-2507"
        raise ValueError(
            f"{model_name} no admet thinking={'on' if True in wanted and True not in allowed else 'off'}. "
            f"Modes possibles: {['off' if not m else 'on' for m in allowed]}. "
            f"Per al mode contrari fes servir l'altre model (p. ex. {other}).")
    return [m for m in wanted if m in allowed] if requested == "both" else wanted


def sampling_kwargs(thinking: bool, temperature=None, top_p=None, top_k=None) -> dict:
    """Paràmetres de decoding.

    Sense reasoning: greedy (determinista). Amb reasoning: sampling amb els valors que recomana Qwen3
    (temperature 0.6, top_p 0.95, top_k 20), perquè el greedy en mode thinking tendeix a entrar en bucles.
    """
    if temperature is None:
        temperature = 0.6 if thinking else 0.0
    if temperature <= 0:
        return {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
    return {
        "do_sample": True,
        "temperature": temperature,
        "top_p": 0.95 if top_p is None else top_p,
        "top_k": 20 if top_k is None else top_k,
    }


def adapter_variant(adapter: naming.AdapterRef, base_variant: str) -> str:
    """Variant del nom d'una predicció LoRA: hiperparàmetres-stamp-versió-mode-quantització-dtype."""
    return naming.slug(f"{adapter.aid.variant}-{adapter.aid.stamp}-{adapter.which}-{base_variant}")


def model_short_name(model_name: str) -> str:
    """'Qwen/Qwen3-32B' -> 'qwen3-32b' (el que va al nom del fitxer)."""
    return naming.slug(model_name.rstrip("/").split("/")[-1])


def load_model(model_name: str, quant: str = "4bit", dtype_choice: str = "auto", adapter_path=None):
    """Carrega tokenizer i model. quant: none | 8bit | 4bit. Retorna (tok, model, nom_dtype).

    adapter_path: carpeta d'un adaptador LoRA (adapters/<id>/best); s'aplica sobre el model base.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if dtype_choice == "fp16":
        dtype = torch.float16
    elif dtype_choice == "bf16":
        dtype = torch.bfloat16
    else:  # auto: bf16 si la GPU el suporta, si no fp16 (la T4 no té bf16 natiu)
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    dtype_name = "bf16" if dtype == torch.bfloat16 else "fp16"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CAP (CPU)"
    print(f"GPU: {gpu} | dtype: {dtype_name} | quantització: {quant}")

    kw = {"device_map": "auto"}
    if quant == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    elif quant == "8bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kw["torch_dtype"] = dtype

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
    if adapter_path is not None:
        from peft import PeftModel
        print(f"Aplicant l'adaptador LoRA {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return tok, model, dtype_name


def generate_predictions(tok, model, *, model_name: str, split: str, thinking: bool,
                         quant: str, dtype_name: str, system: str = "zeroshot",
                         n: int | None = None, max_new_tokens: int = 1300,
                         temperature=None, top_p=None, top_k=None, seed: int = 0,
                         stamp: str | None = None, resume: bool = False,
                         adapter: naming.AdapterRef | None = None,
                         predictions_dir="predictions", repo_dir=".") -> Path:
    """Genera les notes d'un split i les desa a predictions/ amb el nom que marca naming.py.

    adapter: si es passa, la predicció es diu "lora" i la variant identifica l'adaptador (hiperparàmetres,
    stamp i versió: best/last/epoch_N), perquè mai es confonguin dues proves.
    """
    import torch

    if system not in PROMPT_BUILDERS:
        raise ValueError(f"system desconegut {system!r}; opcions: {sorted(PROMPT_BUILDERS)}")
    build = PROMPT_BUILDERS[system]
    sampling = sampling_kwargs(thinking, temperature, top_p, top_k)

    variant = f"{'think' if thinking else 'nothink'}-{quant}-{dtype_name}"
    system_name = system
    if adapter is not None:
        if adapter.aid.model != model_short_name(model_name):
            raise ValueError(f"L'adaptador és de {adapter.aid.model} i el model base és {model_short_name(model_name)}")
        system_name = "lora"
        variant = adapter_variant(adapter, variant)
    pid = naming.PredictionId.new(split, system_name, model_short_name(model_name), variant, stamp)
    out_path = naming.prediction_path(pid, predictions_dir)
    meta = naming.meta_path(pid, predictions_dir)

    done = set()
    if resume and out_path.exists():
        done = {json.loads(l)["file"] for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()}
        print(f"Reprenent: {len(done)} notes ja generades a {out_path.name}")
    else:
        naming.ensure_new(out_path)  # no sobreescriu una predicció existent
    if not meta.exists():
        import transformers
        write_meta(meta, {
            "model": model_name, "system": system_name, "prompt": system, "split": split, "thinking": thinking,
            "adapter": None if adapter is None else {"id": adapter.aid.stem, "version": adapter.which},
            "quantization": quant, "dtype": dtype_name, "max_new_tokens": max_new_tokens,
            "sampling": sampling, "seed": seed, "n": n,
            "torch": torch.__version__, "transformers": transformers.__version__,
        }, repo_dir)

    rows = load_split(split)
    if n:
        rows = rows[:n]
    device = next(model.parameters()).device

    with out_path.open("a", encoding="utf-8") as f:
        for i, r in enumerate(rows, 1):
            if r["file"] in done:
                continue
            # enable_thinking només el fan servir les plantilles de Qwen3 híbrides; les altres l'ignoren
            text = tok.apply_chat_template(build(r["src"]), tokenize=False,
                                           add_generation_prompt=True, enable_thinking=thinking)
            inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
            n_prompt = inputs["input_ids"].shape[1]

            torch.manual_seed(seed)  # mateixa seed per nota: el sampling és reproduïble
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, **sampling,
                    pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
            seconds = time.time() - t0

            gen_ids = out[0, n_prompt:]
            raw = tok.decode(gen_ids, skip_special_tokens=True)
            note, reasoning, truncated = split_reasoning(raw, thinking)
            n_out = int(gen_ids.shape[0])

            row = {"file": r["file"], "pred": note, "ref": r["tgt"],
                   "prompt_tokens": n_prompt, "output_tokens": n_out,
                   "hit_limit": n_out >= max_new_tokens, "reasoning_truncated": truncated,
                   "seconds": round(seconds, 1)}
            if thinking:
                row["reasoning"] = reasoning
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(rows)}] {r['file']}: {n_out} tokens, {seconds:.0f}s"
                  + ("  (TRUNCAT dins del reasoning!)" if truncated else ""))
    print(f"Desat: {out_path}")
    return out_path
