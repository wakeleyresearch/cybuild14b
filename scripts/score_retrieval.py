"""Strict re-score of the lookup library's search against data/eval_keys.jsonl. CPU only.

    python -I scripts/score_retrieval.py                          # all search modes the library offers, top 5 and top 10
    python -I scripts/score_retrieval.py --modes combined --limit 100

For each test question answerable from sources/, the best tier reached in the top 5 and top 10 results:
  section     the passage is the very section the question quotes (only for the questions that quote a heading)
  exact       a result comes from the answer's file; for the 29 files holding many entries the passage must also name the
              entry; for the 74 record questions a result must carry the record id (CVE, ATT&CK, CWE, CAPEC)
  entity      an equivalent document for the same thing: the same cmdlet in another PowerShell version, the same tool's
              man page, tldr page or Kali page, the same Windows event page
  collection  a result comes from the right collection (sources/<area>/<collection>/), nothing closer
Rates are reported overall, per topic, and by whether the question names its answer (quoted heading, backticked name,
an ID) or not, because questions that name their answer are easy for keyword search.

Writes data/retrieval_scores.jsonl (one line per question and mode, with the top results) and data/retrieval_summary.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retrieve  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
TIERS = ("section", "exact", "entity", "collection", "miss")
RECORD_PLACES = {"CVE": ("cve/", "prioritization/cisa-vulnrichment"), "T": ("references/mitre-attack",), "CWE": ("remediation/mitre",),
                 "CAPEC": ("remediation/mitre",)}
MAN_DIRS = r"(?:kali/man-pages/text|man-pages/debian/text|man-pages/linux-man-pages/text)"


def rel(p: str | None) -> str:
    p = (p or "").replace("\\", "/")
    return p.split("sources/", 1)[1] if "sources/" in p else p.lstrip("./")


def collection(path: str) -> str:
    parts = path.split("/")
    return "/".join(parts[:2]) if len(parts) > 2 else parts[0]


def norm_text(s: str | None) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", (s or "").casefold()).split())


def cue_type(q: str) -> str:
    if re.search(r'"[^"\n]{3,120}"', q):
        return "quotes a heading"
    if re.search(r"`[^`\n]+`", q):
        return "names a tool or setting"
    if re.search(r"\b(CVE-\d{4}-\d+|T\d{4}(\.\d{3})?|CWE-\d+|CAPEC-\d+|event (ID )?\d{3,5}|\d{4})\b", q):
        return "contains an ID"
    return "names nothing"


def record_id(row: dict) -> str | None:
    k = row.get("entity_key") or ""
    return k if re.fullmatch(r"(CVE-\d{4}-\d+|T\d{4}(\.\d{3})?|CWE-\d+|CAPEC-\d+)", k) else None


def entry_name(row: dict) -> str | None:
    """For answers that are one entry inside a file holding many: the entry's name, which the passage must contain."""
    if "one of many" in row["resolved_by"] or "section of a single" in row["resolved_by"]:
        return row["entity_key"].partition(":")[2] or None
    return None


def tier_of(row: dict, r: dict, quoted: list[str]) -> str:
    src, text = rel(r.get("source")), " ".join(str(r.get(f) or "") for f in ("id", "title", "heading", "text"))
    rid = record_id(row)
    if rid:
        if re.search(rf"(?<![\w-]){re.escape(rid)}(?![\w.-])", text, re.I):
            return "exact"
        prefix = rid.split("-")[0] if not rid.startswith("T") else "T"
        return "collection" if src.startswith(RECORD_PLACES.get(prefix, ("\0",))) else "miss"
    f = row.get("file") or ""
    if src == f:
        entry = entry_name(row)
        if entry and norm_text(entry)[:60] not in norm_text(text):
            return "entity"                                   # right file, but a different entry in it
        if quoted and norm_text(r.get("heading") or r.get("title")) in {norm_text(h) for h in quoted}:
            return "section"
        return "exact"
    pre, _, val = (row.get("entity_key") or "").partition(":")
    if pre == "ps" and src.startswith("powershell/") and src.endswith(f"/{val}.md"):
        return "entity"
    if pre == "tool":
        name = re.escape(val.lower())
        if (re.fullmatch(rf"{MAN_DIRS}/{name}\.[^/]+\.txt", src) or re.fullmatch(rf"man-pages/tldr/pages/[^/]+/{name.replace(' ', '-')}\.md", src)
                or src == f"kali/kali-tools/{val}/index.md"):
            return "entity"
    if pre == "event" and src.endswith(f"event-{val}.txt"):
        return "entity"
    if f and collection(src) == collection(f):
        return "collection"
    return "miss"


def best(row: dict, results: list[dict], q: str, k: int) -> str:
    quoted = re.findall(r'"([^"\n]{3,120})"', q)
    tiers = [tier_of(row, r, quoted) for r in results[:k]]
    return min(tiers, key=TIERS.index) if tiers else "miss"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", default="keyword,meaning,combined")
    ap.add_argument("--limit", type=int, help="score only the first N answerable questions (a quick check)")
    args = ap.parse_args()
    lib = retrieve.library()          # raises LibraryUnavailable with the reason
    rows = [json.loads(line) for line in open(os.path.join(DATA, "eval_keys.jsonl"), encoding="utf-8")]
    rows = [r for r in rows if r["answerable_from_sources"]][:args.limit]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    summary: dict = {"questions": len(rows), "modes": {}}
    with open(os.path.join(DATA, "retrieval_scores.jsonl"), "w", encoding="utf-8", newline="\n") as out:
        for mode in modes:
            t0, times = time.time(), []
            tally = {k: collections.Counter() for k in (5, 10)}
            by_topic, by_cue = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
            try:
                for row in rows:
                    q = row["question"]
                    t1 = time.time()
                    res = lib.search(q, k=10, mode=mode)
                    times.append(time.time() - t1)
                    b5, b10 = best(row, res, q, 5), best(row, res, q, 10)
                    tally[5][b5] += 1
                    tally[10][b10] += 1
                    good = b10 in ("section", "exact")
                    by_topic[row["topic"]]["n"] += 1
                    by_topic[row["topic"]]["hit"] += good
                    cue = cue_type(q)
                    by_cue[cue]["n"] += 1
                    by_cue[cue]["hit"] += good
                    out.write(json.dumps({"mode": mode, "index": row["index"], "topic": row["topic"], "cue": cue, "best5": b5, "best10": b10,
                                          "want": row.get("file") or row.get("entity_key"), "question": q,
                                          "top": [{"source": rel(r.get("source")), "heading": r.get("heading"), "score": r.get("score")} for r in res[:5]]},
                                         ensure_ascii=False) + "\n")
            except NotImplementedError:
                summary["modes"][mode] = "not offered by this library"
                continue
            n = len(rows)
            cum = lambda c: {t: round(sum(c[x] for x in TIERS[:TIERS.index(t) + 1]) / n, 3) for t in TIERS[:-1]}  # noqa: E731
            times.sort()
            summary["modes"][mode] = {
                "at_5": cum(tally[5]), "at_10": cum(tally[10]),
                "exact_or_better_at_10_by_topic": {t: f"{c['hit']}/{c['n']}" for t, c in sorted(by_topic.items())},
                "exact_or_better_at_10_by_cue": {t: f"{c['hit']}/{c['n']}" for t, c in sorted(by_cue.items())},
                "seconds_per_query_median": round(times[len(times) // 2], 3), "seconds_total": round(time.time() - t0),
            }
            print(f"{mode}: exact-or-better {summary['modes'][mode]['at_10']['exact']:.1%} at 10, "
                  f"{summary['modes'][mode]['at_5']['exact']:.1%} at 5; collection {summary['modes'][mode]['at_10']['collection']:.1%} at 10", flush=True)
    summary["free_ram_gb_after"] = round(retrieve.free_ram_gb(), 2)
    json.dump(summary, open(os.path.join(DATA, "retrieval_summary.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
