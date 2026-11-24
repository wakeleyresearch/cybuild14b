"""Generate answers from the base model and a trained adapter, with and without lookup context, for side-by-side reading.

    python scripts/answer.py --model models/<name> --adapter runs/<run>/final --questions data/probe_questions.jsonl --out runs/<run>_answers
    python scripts/answer.py --model models/<name> --adapter runs/<run>/final --ask "How do I ...?"

Variants (choose with --variants; default is all that are possible):
  base            the untuned model, question only
  tuned           the model with the adapter, question only
  tuned+lookup    the model with the adapter, question plus passages from the lookup library (scripts/retrieve.py)
  base+lookup     the untuned model with the same passages, to separate what tuning adds from what lookup adds

Writes <out>.jsonl (everything) and <out>.md (for reading). If a question lists `expect` patterns, the share of them
each answer mentions is reported. That is a keyword check, not a judgement of correctness: read the answers.
Needs the GPU, so run it only when no training job is using the card.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# the same system prompt the training examples carry (rank_dataset.py); a different one would test a prompt the adapter never saw
SYSTEM = ("You are a cybersecurity and systems operations assistant. Give accurate, practical answers with the commands, queries, "
          "configuration or code that solve the problem. State assumptions, and say so when an answer depends on context you were not given.")
WITH_CONTEXT = ("{question}\n\nReference material retrieved for this question follows. It may be incomplete or beside the point. Use what applies, "
                "say which numbered blocks you relied on, and say so plainly if it does not answer the question.\n\n{context}")


def get_context(question: str, max_chars: int) -> dict:
    """Passages from the lookup library, or an empty context with the reason if it is not available."""
    try:
        import retrieve
        return retrieve.build_context(question, max_chars=max_chars)
    except Exception as e:  # noqa: BLE001 - the library is optional; an answer without it is still an answer
        return {"context": "", "sources": [], "library": {"available": False, "error": f"{type(e).__name__}: {e}"}}


def generate(model, tok, question: str, context: str, max_new: int) -> tuple[str, float]:
    user = WITH_CONTEXT.format(question=question, context=context) if context else question
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new, do_sample=False, repetition_penalty=1.05,
                             pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    new = out[0, ids.shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip(), len(new) / max(1e-6, time.time() - t0)


def coverage(answer: str, expect: list[str]) -> float | None:
    if not expect:
        return None
    return round(sum(bool(re.search(p, answer, re.I)) for p in expect) / len(expect), 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", help="a saved adapter folder; without it only the base variants can run")
    ap.add_argument("--questions", help="JSON lines with `question` and optionally `id`, `topic`, `expect` (list of regular expressions)")
    ap.add_argument("--ask", help="a single question instead of a file")
    ap.add_argument("--variants", nargs="+", choices=["base", "tuned", "tuned+lookup", "base+lookup"])
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "answers"))
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--context-chars", type=int, default=6000)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    if args.ask:
        rows = [{"id": "ask", "topic": "", "question": args.ask}]
    elif args.questions:
        rows = [json.loads(line) for line in open(args.questions, encoding="utf-8") if line.strip()][:args.limit]
    else:
        ap.error("give --questions or --ask")

    probe = get_context(rows[0]["question"], args.context_chars)
    lookup_ok = bool(probe.get("library", {}).get("available"))
    variants = args.variants or (["base"] + (["tuned"] if args.adapter else []) + ((["tuned+lookup"] if args.adapter else ["base+lookup"]) if lookup_ok else []))
    if not lookup_ok and any("lookup" in v for v in variants):
        print(f"lookup library not available ({probe.get('library', {}).get('error')}); dropping the lookup variants", flush=True)
        variants = [v for v in variants if "lookup" not in v]
    if not args.adapter:
        variants = [v for v in variants if not v.startswith("tuned")]
    print(f"{len(rows)} questions; variants: {', '.join(variants)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant, device_map={"": 0}, dtype=torch.bfloat16)
    model.eval()

    contexts = {}
    if any("lookup" in v for v in variants):
        for r in rows:
            contexts[r.get("id") or r["question"]] = get_context(r["question"], args.context_chars)

    results = [{**r, "answers": {}, "coverage": {}, "tokens_per_s": {}} for r in rows]
    # the base variants run first: loading the adapter wraps the model in place
    for group in (["base", "base+lookup"], ["tuned", "tuned+lookup"]):
        todo = [v for v in group if v in variants]
        if not todo:
            continue
        if group[0] == "tuned":
            model = PeftModel.from_pretrained(model, args.adapter)
            model.eval()
        for v in todo:
            for i, res in enumerate(results, 1):
                ctx = contexts.get(res.get("id") or res["question"], {}) if "lookup" in v else {}
                text, rate = generate(model, tok, res["question"], ctx.get("context", ""), args.max_new_tokens)
                res["answers"][v] = text
                res["tokens_per_s"][v] = round(rate, 1)
                res["coverage"][v] = coverage(text, res.get("expect") or [])
                if "lookup" in v:
                    res["lookup_sources"] = [{k: s.get(k) for k in ("n", "kind", "title", "source", "fetched")} for s in ctx.get("sources", [])]
                print(f"  {v:13s} {i}/{len(results)}  {rate:5.1f} tok/s  coverage {res['coverage'][v]}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for res in results:
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
    with open(args.out + ".md", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# Answers: {', '.join(variants)}\n\nModel `{args.model}`" + (f", adapter `{args.adapter}`" if args.adapter else "") + ".\n"
                 "Coverage is the share of expected keywords an answer mentions. It is a rough signal, not a correctness score.\n")
        by_topic: dict[str, dict[str, list[float]]] = {}
        for res in results:
            fh.write(f"\n---\n\n## {res.get('id', '')} ({res.get('topic', '')})\n\n**Question:** {res['question']}\n")
            if res.get("expect"):
                fh.write(f"\n*Expected to mention:* {', '.join('`' + e + '`' for e in res['expect'])}\n")
            for v in variants:
                cov = res["coverage"].get(v)
                fh.write(f"\n### {v}" + (f" (coverage {cov})" if cov is not None else "") + f"\n\n{res['answers'].get(v, '')}\n")
                if cov is not None:
                    by_topic.setdefault(res.get("topic", ""), {}).setdefault(v, []).append(cov)
            if res.get("lookup_sources"):
                fh.write("\n*Lookup passages supplied:* " + "; ".join(f"[{s['n']}] {s.get('title') or s.get('source')}" for s in res["lookup_sources"]) + "\n")
        if by_topic:
            fh.write("\n---\n\n## Keyword coverage by topic\n\n| Topic | " + " | ".join(variants) + " |\n|---|" + "---|" * len(variants) + "\n")
            for t, d in sorted(by_topic.items()):
                fh.write(f"| {t} | " + " | ".join(f"{sum(d[v]) / len(d[v]):.2f}" if d.get(v) else "" for v in variants) + " |\n")
            fh.write("| **all** | " + " | ".join(f"{sum(x for d in by_topic.values() for x in d.get(v, [])) / max(1, sum(len(d.get(v, [])) for d in by_topic.values())):.2f}"
                                             for v in variants) + " |\n")
    print(f"wrote {args.out}.jsonl and {args.out}.md", flush=True)


if __name__ == "__main__":
    main()
