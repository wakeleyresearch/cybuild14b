"""QLoRA fine-tuning on one GPU, written for a 16 GB card.

    python scripts/train_lora.py --model models/<name> --out runs/<name>              # full run
    python scripts/train_lora.py --model models/<name> --smoke 12 --max-len 2048      # fit and speed test only

Trains a LoRA adapter on data/train.jsonl (chat format) with the base model loaded in 4-bit. The loss is computed
on the assistant's reply only. Progress goes to <out>/log.jsonl; the adapter is saved every --save-steps optimizer
steps and at the end, and --resume continues from the newest checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def keep_awake() -> None:
    """Ask Windows not to sleep while this process runs (released automatically when it exits)."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


def os_gpu_memory() -> tuple[float, float]:
    """(GB on the card, GB spilled into system RAM) for this process, as Windows counts it.

    PyTorch's own figures miss the spill: on Windows, GPU memory beyond the card's free space is silently placed in
    shared (system) RAM instead of raising an out-of-memory error.
    """
    if os.name != "nt":
        return 0.0, 0.0
    import subprocess
    pid = os.getpid()
    cmd = (f"(Get-Counter '\\GPU Process Memory(pid_{pid}*)\\Dedicated Usage','\\GPU Process Memory(pid_{pid}*)\\Shared Usage').CounterSamples | "
           "ForEach-Object { $_.Path.Split('\\')[-1] + '=' + $_.CookedValue }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=60).stdout
    except Exception:  # noqa: BLE001
        return 0.0, 0.0
    ded = sum(float(ln.split("=")[1]) for ln in out.splitlines() if ln.startswith("dedicated")) / 1e9
    shared = sum(float(ln.split("=")[1]) for ln in out.splitlines() if ln.startswith("shared")) / 1e9
    return ded, shared


def load_examples(path: str, tok, max_len: int, limit: int | None, seed: int):
    """-> list of (input_ids, labels) with labels masked (-100) everywhere except the assistant reply."""
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    random.Random(seed).shuffle(rows)
    out, clipped = [], 0
    for r in rows[:limit] if limit else rows:
        msgs = r["messages"]
        prefix = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        a_ids = tok(msgs[-1]["content"] + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
        if len(p_ids) > max_len - 64:        # an over-long question: keep its end, where the actual ask usually is
            p_ids = p_ids[-(max_len - 64):]
        room = max_len - len(p_ids)
        if len(a_ids) > room:
            a_ids, clipped = a_ids[:room], clipped + 1
        out.append((p_ids + a_ids, [-100] * len(p_ids) + a_ids))
    return out, clipped


def reply_loss(model, batch: list[tuple[list[int], list[int]]], device, pad_id: int = 0) -> torch.Tensor:
    """Mean loss per reply token over a batch of (input_ids, labels); shorter examples are padded on the right."""
    width = max(len(ids) for ids, _ in batch)
    x = torch.full((len(batch), width), pad_id, dtype=torch.long)
    y = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for r, (ids, labels) in enumerate(batch):
        x[r, :len(ids)] = torch.tensor(ids)
        y[r, :len(ids)] = torch.tensor(labels)
        mask[r, :len(ids)] = 1
    x, y, mask = x.to(device), y.to(device), mask.to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=x, attention_mask=mask).logits
    keep = y[:, 1:] != -100                  # predict token t+1 from position t, reply positions only
    sel = logits[:, :-1][keep]
    tgt = y[:, 1:][keep]
    total = sel.new_zeros((), dtype=torch.float32)
    for i in range(0, sel.shape[0], 256):    # in slices, so the float32 copy of the vocabulary scores stays small
        total = total + torch.nn.functional.cross_entropy(sel[i:i + 256].float(), tgt[i:i + 256], reduction="sum")
    return total / max(1, tgt.numel())


def make_batches(data, budget: int, max_batch: int, seed: int) -> list[list[int]]:
    """Group examples of similar length so each batch holds at most `budget` tokens after padding.

    The 4-bit model pays a fixed cost per forward pass, so several short examples in one pass train much faster
    than one at a time. Batches are shuffled, so the model still sees topics in mixed order.
    """
    rng = random.Random(seed)
    idx = list(range(len(data)))
    rng.shuffle(idx)
    batches = []
    for s in range(0, len(idx), 2048):
        cur, widest = [], 0
        for i in sorted(idx[s:s + 2048], key=lambda j: len(data[j][0])):
            n = len(data[i][0])
            if cur and (max(widest, n) * (len(cur) + 1) > budget or len(cur) >= max_batch):
                batches.append(cur)
                cur, widest = [], 0
            cur.append(i)
            widest = max(widest, n)
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", default=os.path.join(ROOT, "data", "train.jsonl"))
    ap.add_argument("--eval", default=os.path.join(ROOT, "data", "eval.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "run"))
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--accum", type=int, default=16, help="examples per optimizer step")
    ap.add_argument("--tok-budget", type=int, default=2048, help="most tokens in one forward pass after padding (sets GPU memory use)")
    ap.add_argument("--max-batch", type=int, default=8, help="most examples in one forward pass")
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--eval-examples", type=int, default=120)
    ap.add_argument("--smoke", type=int, default=0, help="run this many worst-case-length steps, report memory and speed, then exit")
    ap.add_argument("--smoke-real", type=int, default=0, metavar="N", help="run N real training batches (mixed shapes), report what Windows says the "
                    "process holds on the card and in system RAM, then exit")
    ap.add_argument("--empty-cache-every", type=int, default=0, metavar="STEPS", help="release PyTorch's cached GPU blocks every this many optimizer steps (0 = never)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", metavar="OUT.json", help="score the held-out set per topic (with --adapter if given), write the result, and exit")
    ap.add_argument("--adapter", help="a saved adapter folder to load for --eval-only")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    keep_awake()

    tok = AutoTokenizer.from_pretrained(args.model)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant, device_map={"": 0}, dtype=torch.bfloat16)
    print(f"loaded {args.model} in {time.time() - t0:.0f}s; GPU memory in use {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    for p in model.parameters():
        p.requires_grad_(False)

    if args.eval_only:
        if args.adapter:
            model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
        rows = [json.loads(line) for line in open(args.eval, encoding="utf-8")]
        by_topic: dict[str, list[float]] = {}
        by_kind: dict[str, list[float]] = {}
        t1 = time.time()
        with torch.no_grad():
            for r in rows:
                msgs = r["messages"]
                p_ids = tok(tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"][-(args.max_len - 64):]
                a_ids = tok(msgs[-1]["content"] + (tok.eos_token or ""), add_special_tokens=False)["input_ids"][:args.max_len - len(p_ids)]
                loss = reply_loss(model, [(p_ids + a_ids, [-100] * len(p_ids) + a_ids)], dev).item()
                by_topic.setdefault(r["meta"]["topic"], []).append(loss)
                by_kind.setdefault(r["meta"]["kind"], []).append(loss)
        table = lambda d: {t: {"n": len(v), "loss": round(sum(v) / len(v), 4)} for t, v in sorted(d.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))}  # noqa: E731
        res = {"model": args.model, "adapter": args.adapter, "examples": len(rows), "seconds": round(time.time() - t1),
               "loss": round(sum(sum(v) for v in by_topic.values()) / len(rows), 4), "by_topic": table(by_topic), "by_kind": table(by_kind)}
        os.makedirs(os.path.dirname(os.path.abspath(args.eval_only)), exist_ok=True)
        json.dump(res, open(args.eval_only, "w", encoding="utf-8"), indent=1)
        print(json.dumps(res, indent=1), flush=True)
        return

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()   # lets gradients flow through checkpointed blocks although the base model is frozen
    model.config.use_cache = False

    ckpts = sorted((d for d in os.listdir(args.out) if d.startswith("step-")), key=lambda d: int(d[5:])) if os.path.isdir(args.out) else []
    start_step = 0
    if args.resume and ckpts:
        model = PeftModel.from_pretrained(model, os.path.join(args.out, ckpts[-1]), is_trainable=True)
        start_step = int(ckpts[-1][5:])
        print(f"resumed adapter from {ckpts[-1]}", flush=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, target_modules=TARGETS, task_type="CAUSAL_LM"))
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable adapter parameters: {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))
    opt_path = os.path.join(args.out, "optimizer_latest.pt")
    if args.resume and ckpts and os.path.isfile(opt_path):
        saved = torch.load(opt_path, map_location="cpu")
        if saved.get("step") == start_step:      # only reuse optimizer state that belongs to this exact checkpoint
            opt.load_state_dict(saved["optimizer"])
            print("restored optimizer state", flush=True)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if args.smoke:
        model.train()
        total = torch.cuda.mem_get_info()[1]
        for rows, width in ((1, args.tok_budget), (4, args.tok_budget // 4), (8, args.tok_budget // 8)):   # one long example, then typical short ones batched
            ids = list(range(1000, 1000 + width))
            batch = [(ids, [-100] * (width // 8) + ids[width // 8:])] * rows
            torch.cuda.reset_peak_memory_stats()
            t1 = time.time()
            for i in range(args.smoke):
                reply_loss(model, batch, dev, pad_id).backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                if i == 0:
                    torch.cuda.synchronize()
                    t1 = time.time()         # the first pass includes one-off warm-up
            torch.cuda.synchronize()
            per = (time.time() - t1) / max(1, args.smoke - 1)
            print(f"SMOKE {rows} x {width} tokens: peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} of {total / 2**30:.1f} GiB; "
                  f"{per:.2f} s/pass; {rows * width / per:.0f} tokens/s", flush=True)
        return

    if args.smoke_real:        # the real batch mix: shapes vary from pass to pass, which is what makes the allocator's cache grow
        data, _ = load_examples(args.train, tok, args.max_len, 6000, args.seed)
        batches = make_batches(data, args.tok_budget, args.max_batch, args.seed)
        model.train()
        t1, toks = time.time(), 0
        for i, rows in enumerate(batches[:args.smoke_real], 1):
            batch = [data[j] for j in rows]
            reply_loss(model, batch, dev, pad_id).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            toks += sum(len(ids) for ids, _ in batch)
            if args.empty_cache_every and i % (args.empty_cache_every * 5) == 0:   # about 5 batches make one optimizer step
                torch.cuda.empty_cache()
            if i % 25 == 0 or i == args.smoke_real:
                torch.cuda.synchronize()
                ded, shared = os_gpu_memory()
                print(f"SMOKE-REAL batch {i}: {toks / (time.time() - t1):.0f} tokens/s; torch allocated peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
                      f"reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB; Windows: {ded:.2f} GB on the card, {shared:.2f} GB in system RAM", flush=True)
        return

    os.makedirs(args.out, exist_ok=True)
    data, clipped = load_examples(args.train, tok, args.max_len, args.max_examples, args.seed)
    evals, _ = load_examples(args.eval, tok, args.max_len, args.eval_examples, args.seed) if os.path.isfile(args.eval) else ([], 0)
    n_tok = sum(len(i) for i, _ in data)
    total_steps = math.ceil(len(data) * args.epochs / args.accum)
    batches = make_batches(data, args.tok_budget, args.max_batch, args.seed)
    print(f"{len(data)} training examples, {n_tok / 1e6:.1f}M tokens ({clipped} clipped to {args.max_len}); {len(batches)} batches "
          f"(average {len(data) / len(batches):.1f} examples each); {total_steps} optimizer steps", flush=True)
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)
    warm = max(10, int(total_steps * 0.03))
    lr_at = lambda s: args.lr * (s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))  # noqa: E731
    log = open(os.path.join(args.out, "log.jsonl"), "a", encoding="utf-8")

    def evaluate() -> float:
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for ex in evals:
                tot += reply_loss(model, [ex], dev, pad_id).item()
        model.train()
        return tot / max(1, len(evals))

    model.train()
    step, seen_tok, run_loss, run_n, skipped = start_step, 0, 0.0, 0, 0
    t_start = time.time()
    pos = 0
    for _ in range(start_step):                      # resuming: replay the batch bookkeeping of the finished steps exactly.
        taken = 0                                    # A step takes whole batches until it has at least --accum examples, so it
        while taken < args.accum:                    # usually takes a few more than --accum; counting "accum per step" would
            taken += len(batches[pos % len(batches)])  # land too early and train on data already seen.
            pos += 1
    while step < total_steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        taken = 0
        while taken < args.accum:                     # one optimizer step covers about --accum examples
            rows = batches[pos % len(batches)]
            pos += 1
            taken += len(rows)
            batch = [data[i] for i in rows]
            try:
                loss = reply_loss(model, batch, dev, pad_id)
                (loss * len(rows) / args.accum).backward()
                run_loss += loss.item() * len(rows)
                run_n += len(rows)
                seen_tok += sum(len(ids) for ids, _ in batch)
            except torch.OutOfMemoryError:
                skipped += len(rows)
                torch.cuda.empty_cache()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if args.empty_cache_every and step % args.empty_cache_every == 0:
            torch.cuda.empty_cache()             # hand cached blocks back to the driver so they cannot pile up beyond the card
        if step % 5 == 0 or step == total_steps:
            el = time.time() - t_start
            rec = {"step": step, "of": total_steps, "loss": round(run_loss / max(1, run_n), 4), "lr": round(lr_at(step), 7), "tokens_per_s": round(seen_tok / el),
                   "hours_elapsed": round(el / 3600, 2), "hours_left": round(el / max(1, step - start_step) * (total_steps - step) / 3600, 2),
                   "gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2), "gpu_reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2),
                   "skipped_oom": skipped}
            run_loss, run_n = 0.0, 0
            if evals and (step % args.eval_steps == 0 or step == total_steps):
                rec["eval_loss"] = round(evaluate(), 4)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
        if step % args.save_steps == 0 or step == total_steps:
            model.save_pretrained(os.path.join(args.out, f"step-{step}"))
            torch.save({"step": step, "optimizer": opt.state_dict()}, opt_path)   # one file, overwritten: enough to resume here or on another machine
    model.save_pretrained(os.path.join(args.out, "final"))
    tok.save_pretrained(os.path.join(args.out, "final"))
    print(f"done: adapter saved to {os.path.join(args.out, 'final')}", flush=True)


if __name__ == "__main__":
    main()
