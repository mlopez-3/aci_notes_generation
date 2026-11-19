"""QLoRA per generar notes clíniques amb ACI-Bench (model base en 4 bits, adaptador LoRA en fp32).

Disseny (pla acordat):
  * Mateix model, mateixa càrrega en 4 bits i MATEIX prompt que a la generació (acinotes.prompts):
    així la comparació prompting vs LoRA és justa.
  * La pèrdua es calcula NOMÉS sobre la nota (el prompt i la transcripció s'emmascaren).
  * Batch 1 + acumulació de gradient + gradient checkpointing; adaptador LoRA en fp32.
  * Només es calculen els logits dels tokens de la nota (logits_to_keep): amb 150 000 tokens de vocabulari,
    els logits d'una seqüència de 4000 tokens ocuparien diversos GB.
  * Els exemples més llargs que --max-len s'EXCLOUEN (no es trunquen: truncar la nota ensenyaria a tallar-la).
  * La pèrdua de valid s'avalua abans d'entrenar i al final de cada època; es desa la millor època.
  * Dades: es llegeixen directament dels JSON originals (train per entrenar, valid per triar l'època).

On queda el resultat:  adapters/<id>/
    best/    adaptador de la millor època segons valid   <- el que s'utilitza per generar
    last/    adaptador de l'última època (per reprendre)
    epoch_N/ adaptador de cada època
        adapter_model.safetensors   pesos de les matrius LoRA A i B (la part entrenada, desenes de MB)
        adapter_config.json         configuració (rang, alpha, mòduls...)
    config.json, log.jsonl (pèrdua per pas i per època), state.pt (estat de l'optimitzador, per --resume)
  <id> = lora__<model>__<variant>__<stamp>   p. ex. lora__qwen3-4b-instruct-2507__r16-attn-ep3-lr0.0002-bf16__20261118T2130Z
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

from . import naming
from .data import load_split
from .generate import model_short_name
from .prompts import PROMPT_BUILDERS
from .provenance import write_meta

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
ALL_LINEAR = ATTN + ["gate_proj", "up_proj", "down_proj"]


# ---------- Dades (sense torch: es poden provar a part) ----------
def completion_end_token(tok):
    vocab = tok.get_vocab()
    return "<|im_end|>" if "<|im_end|>" in vocab else tok.eos_token


def build_example(tok, src, tgt, max_len, prompt="zeroshot"):
    """Retorna {'ids', 'prompt_len', 'comp_len'} o None si no cap a max_len. La pèrdua només cobreix la nota."""
    msgs = PROMPT_BUILDERS[prompt](src)  # el MATEIX prompt que s'usa en generar
    prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    note = tgt.strip()
    full_text = tok.apply_chat_template(msgs + [{"role": "assistant", "content": note}],
                                        tokenize=False, enable_thinking=False)
    if full_text.startswith(prompt_text):
        comp_text = full_text[len(prompt_text):].rstrip("\n")
    else:  # plantilles amb blocs extra (p. ex. <think> buit): s'escriu la resposta a mà
        comp_text = note + completion_end_token(tok)
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    comp_ids = tok(comp_text, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(comp_ids) > max_len:
        return None
    return {"ids": prompt_ids + comp_ids, "prompt_len": len(prompt_ids), "comp_len": len(comp_ids)}


def load_examples(tok, rows, max_len, limit=None):
    """rows: [{file, src, tgt}] (de load_split). Retorna (exemples, fitxers exclosos per massa llargs)."""
    if limit:
        rows = rows[:limit]
    exs, skipped = [], []
    for r in rows:
        e = build_example(tok, r["src"], r["tgt"], max_len)
        if e is None:
            skipped.append(r["file"])
        else:
            e["file"] = r["file"]
            exs.append(e)
    return exs, skipped


def make_groups(n, accum, seed, epoch):
    """Ordre aleatori (diferent a cada època) dividit en grups de `accum` exemples (un pas de l'optimitzador)."""
    order = list(range(n))
    random.Random(seed + epoch).shuffle(order)
    return [order[i:i + accum] for i in range(0, n, accum)]


def lr_factor(step, total, warm):
    """Escalfament lineal i després cosinus fins al 10 % del lr."""
    if step < warm:
        return (step + 1) / warm
    progress = (step - warm) / max(1, total - warm)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def adapter_variant_name(args, dtype_name: str) -> str:
    """Variant de l'id de l'adaptador: identifica els hiperparàmetres (smoke es distingeix a part)."""
    v = f"r{args.r}-{args.targets}-ep{args.epochs}-lr{args.lr:g}-{dtype_name}"
    return f"smoke-{v}" if args.smoke else v


# ---------- Entrenament ----------
def run(args) -> Path:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    try:
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    except ImportError as e:
        raise SystemExit(f"No es pot importar peft ({e}). Instal·la'l amb: pip install -e '.[ml]'")

    if "thinking" in args.model.lower():
        raise SystemExit("Aquest entrenament és per al model SENSE reasoning (Instruct). "
                         "La plantilla de Thinking és diferent i no està preparada.")
    if not torch.cuda.is_available():
        raise SystemExit("No hi ha GPU: l'entrenament necessita CUDA.")
    device = torch.device("cuda:0")

    # dtype de càlcul: bf16 si la GPU el suporta (L4, A100...), si no fp16 (T4)
    if args.dtype == "fp16":
        cdt = torch.float16
    elif args.dtype == "bf16":
        cdt = torch.bfloat16
    else:
        cdt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    dtype_name = "bf16" if cdt == torch.bfloat16 else "fp16"
    use_scaler = cdt == torch.float16  # el GradScaler només fa falta amb fp16

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.smoke:
        args.epochs, args.grad_accum = 1, 2

    if args.resume:
        ref = naming.resolve_adapter(args.resume, args.adapters_dir)
        aid, out_dir = ref.aid, ref.root
    else:
        aid = naming.AdapterId.new(model_short_name(args.model), adapter_variant_name(args, dtype_name), args.stamp)
        out_dir = naming.adapter_dir(aid, args.adapters_dir)
        naming.ensure_new(out_dir / "config.json")  # no sobreescriu un adaptador existent
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Adaptador: {aid.stem}\nCarpeta   : {out_dir}")

    tok = AutoTokenizer.from_pretrained(args.model)
    train, skipped_t = load_examples(tok, load_split("train"), args.max_len, 4 if args.smoke else None)
    valid, skipped_v = load_examples(tok, load_split("valid"), args.max_len, 2 if args.smoke else None)
    print(f"train: {len(train)} exemples usats, {len(skipped_t)} exclosos per superar {args.max_len} tokens {skipped_t}")
    print(f"valid: {len(valid)} exemples usats, {len(skipped_v)} exclosos {skipped_v}")
    if not train:
        raise SystemExit("No hi ha exemples d'entrenament (max-len massa baix?).")
    lens = [e["prompt_len"] + e["comp_len"] for e in train]
    print(f"longitud (tokens) de train: mitjana={sum(lens) / len(lens):.0f} màx={max(lens)}; "
          f"nota: mitjana={sum(e['comp_len'] for e in train) / len(train):.0f}")

    # Model base en 4 bits (la mateixa càrrega que en generar) + LoRA
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=cdt, bnb_4bit_use_double_quant=True)
    kw = dict(quantization_config=bnb, device_map={"": 0})
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=cdt, **kw)
    except TypeError:  # versions antigues de transformers
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=cdt, **kw)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    last_dir, state_path = out_dir / "last", out_dir / "state.pt"
    resuming = bool(args.resume) and last_dir.exists() and state_path.exists()
    if resuming:
        model = PeftModel.from_pretrained(model, str(last_dir), is_trainable=True)
        print(f"Reprenent des de {last_dir}")
    else:
        cfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
                         target_modules=ATTN if args.targets == "attn" else ALL_LINEAR, task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    bad = [p.dtype for p in params if p.dtype != torch.float32]
    if bad:
        print(f"ATENCIÓ: {len(bad)} paràmetres entrenables no són fp32 ({set(bad)}): pot donar NaN.")

    steps_per_epoch = math.ceil(len(train) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warm = max(1, int(args.warmup_ratio * total_steps))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total_steps, warm))
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    start_epoch, best_val, best_epoch, history = 0, float("inf"), None, []
    if resuming:
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        start_epoch, best_val, best_epoch, history = st["epoch"], st["best_val"], st["best_epoch"], st["history"]
        print(f"Última època completada: {start_epoch}; millor pèrdua de valid fins ara: {best_val:.4f}")
    print(f"passos de l'optimitzador: {steps_per_epoch} per època, {total_steps} en total (escalfament {warm}); "
          f"dtype de càlcul {dtype_name}")

    def example_loss(ex):
        ids = torch.tensor([ex["ids"]], device=device)
        k = ex["comp_len"] + 1  # logits que prediuen cada token de la nota
        with torch.autocast("cuda", dtype=cdt):
            try:
                logits = model(input_ids=ids, logits_to_keep=k).logits
            except TypeError:  # sense logits_to_keep: es calculen tots (més memòria)
                logits = model(input_ids=ids).logits
        logits = logits[:, -k:-1, :].float()
        labels = ids[:, -ex["comp_len"]:]
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

    @torch.no_grad()
    def eval_loss():
        model.eval()
        tot, n = 0.0, 0
        for ex in valid:
            tot += example_loss(ex).item() * ex["comp_len"]
            n += ex["comp_len"]
        model.train()
        return tot / max(n, 1)

    def log(rec):
        with open(out_dir / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    if not resuming:
        write_meta(out_dir / "config.json",
                   {**{k: v for k, v in vars(args).items()}, "adapter_id": aid.stem, "dtype_used": dtype_name,
                    "n_train": len(train), "n_valid": len(valid), "skipped_train": skipped_t,
                    "skipped_valid": skipped_v, "torch": torch.__version__}, ".")
    torch.cuda.reset_peak_memory_stats()
    if valid and not args.no_eval_before and not resuming:
        v0 = eval_loss()
        print(f"Pèrdua de valid ABANS d'entrenar (model sense tocar): {v0:.4f}")
        log({"event": "eval_before", "val_loss": v0})

    model.train()
    nan_count, step = 0, start_epoch * steps_per_epoch
    for epoch in range(start_epoch, args.epochs):
        t_epoch = time.time()
        ep_losses = []
        for group in make_groups(len(train), args.grad_accum, args.seed, epoch):
            t0, loss_sum, used = time.time(), 0.0, 0
            for idx in group:
                loss = example_loss(train[idx])
                if not torch.isfinite(loss):
                    nan_count += 1
                    print(f"ATENCIÓ: pèrdua no finita a {train[idx]['file']} ({nan_count})")
                    if nan_count >= 5:
                        raise RuntimeError("Massa pèrdues NaN/inf: l'entrenament és inestable. Prova --lr més baix.")
                    continue
                scaler.scale(loss / len(group)).backward()
                loss_sum += loss.item()
                used += 1
            scaler.unscale_(opt)
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm))
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            avg = loss_sum / max(used, 1)
            ep_losses.append(avg)
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"època {epoch + 1}/{args.epochs} pas {step}/{total_steps}  pèrdua={avg:.4f}  "
                  f"grad={gnorm:.2f}  lr={sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s  memòria pic={peak:.1f} GB")
            log({"event": "step", "epoch": epoch + 1, "step": step, "loss": avg, "grad_norm": gnorm})

        val = eval_loss() if valid else float("nan")
        mean_train = sum(ep_losses) / max(len(ep_losses), 1)
        print(f"== Època {epoch + 1}: pèrdua train={mean_train:.4f}  valid={val:.4f}  "
              f"({(time.time() - t_epoch) / 60:.1f} min)")
        history.append({"epoch": epoch + 1, "train_loss": mean_train, "val_loss": val})
        log({"event": "epoch", "epoch": epoch + 1, "train_loss": mean_train, "val_loss": val})

        model.save_pretrained(str(out_dir / f"epoch_{epoch + 1}"))
        if valid and val < best_val:
            best_val, best_epoch = val, epoch + 1
            model.save_pretrained(str(out_dir / "best"))
            print(f"   millor època fins ara (valid={val:.4f}) -> {out_dir / 'best'}")
        model.save_pretrained(str(last_dir))
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch + 1, "best_val": best_val, "best_epoch": best_epoch, "history": history}, state_path)

    if not valid:  # sense valid no hi ha "millor": s'utilitza l'última
        model.save_pretrained(str(out_dir / "best"))
    print(f"\nAcabat. Millor època: {best_epoch} (valid={best_val:.4f}). Adaptadors a {out_dir}")
    print(f"Pesos de la millor època: {out_dir / 'best' / 'adapter_model.safetensors'}")
    print(f"Memòria pic: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    print("Per generar amb l'adaptador:")
    print(f"  python scripts/generate.py --adapter {aid.stem} --split valid")
    return out_dir
