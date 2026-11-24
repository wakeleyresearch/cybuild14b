"""Mechanical quality audit of a training file: counts leftover markup and other defects per kind of example.

    python -I scripts/audit_set.py data/r2/train_new.jsonl            # summary table
    python -I scripts/audit_set.py data/r2/train_new.jsonl --show 2   # plus two offending snippets per check

It finds debris that survived conversion (tags, template braces, entities, flattened code, replacement characters).
It cannot tell whether an answer is correct.
"""
from __future__ import annotations

import collections
import json
import re
import sys

CHECKS = {
    "html or xml tag in prose": re.compile(r"</?(?:p|div|span|br|a|b|i|em|strong|code|pre|ul|ol|li|td|tr|table|para|literal|xref|link|term|varname|filename|command|emphasis|tt|sup|sub)\b[^<>\n]{0,80}>", re.I),
    "template braces": re.compile(r"\{\{|\}\}|\{%|%\}"),
    "html entity": re.compile(r"&(?:nbsp|lt|gt|amp|quot|#\d{2,5});"),
    "rst role or directive": re.compile(r":(?:ref|doc|term|class|func|meth|option|envvar|c:func|c:macro|file|kbd|program):`|^\.\. (?:code-block|note|warning|toctree|include|_[\w-]+:)", re.M),
    "wiki markup": re.compile(r"\[\[[^\]\n]{1,80}\]\]|'''"),
    "replacement character": re.compile("�"),
    "placeholder": re.compile(r"\b(?:TBD|TODO|FIXME|lorem ipsum)\b"),
    "empty reference": re.compile(r"\((?:See|see)\s+for\b|\bThe (?:parameter|section|option|chapter)\s{2,}\w|\bsee\s{2,}[.,)]"),
    "line over 600 chars": re.compile(r"^[^\n]{600,}$", re.M),
    "broken table row": re.compile(r"^\|[^\n|]*$", re.M),
}


def prose(text: str) -> str:
    """The answer without fenced code blocks and inline code, where markup is legitimate."""
    return re.sub(r"`[^`\n]*`", "", re.sub(r"```.*?```", "", text, flags=re.S))


def main() -> None:
    path = sys.argv[1]
    show = int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 0
    total = collections.Counter()
    hits = collections.defaultdict(collections.Counter)
    samples = collections.defaultdict(list)
    extra = collections.Counter()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            kind = r["meta"]["kind"]
            q, a = r["messages"][-2]["content"], r["messages"][-1]["content"]
            total[kind] += 1
            body = prose(a)
            no_fences = re.sub(r"```.*?```", "", a, flags=re.S)       # inline code kept: removing it would fake an "empty reference"
            for name, rx in CHECKS.items():
                target = a if name in ("line over 600 chars", "replacement character") else no_fences if name in ("empty reference", "placeholder") else body
                m = rx.search(target)
                if m:
                    hits[name][kind] += 1
                    if len(samples[(name, kind)]) < show:
                        s = max(0, m.start() - 70)
                        samples[(name, kind)].append(target[s:m.end() + 70].replace("\n", " / "))
            if len(a) < 60:
                extra["answer under 60 chars"] += 1
            if a.count("```") % 2:
                extra["unbalanced code fence"] += 1
            if not a.rstrip().endswith((".", "!", "?", "`", ")", "]", "}", "|", ":", "*", ">", '"', "'")) and not a.rstrip()[-1:].isalnum():
                extra["ends on an odd character"] += 1
            if len(q) < 25:
                extra["question under 25 chars"] += 1
    n = sum(total.values())
    print(f"{path}: {n} examples in {len(total)} kinds\n")
    print(f"{'check':28s} {'examples':>8s} {'share':>6s}   kinds most affected (count / share of that kind)")
    for name in CHECKS:
        c = hits[name]
        worst = sorted(c.items(), key=lambda kv: -kv[1])[:5]
        print(f"{name:28s} {sum(c.values()):8d} {sum(c.values()) / n:6.1%}   " + ", ".join(f"{k} {v} ({v / total[k]:.0%})" for k, v in worst))
    for k, v in extra.items():
        print(f"{k:28s} {v:8d} {v / n:6.1%}")
    if show:
        print()
        for (name, kind), items in sorted(samples.items(), key=lambda kv: -hits[kv[0][0]][kv[0][1]])[:40]:
            for s in items:
                print(f"[{name} | {kind}] ...{s.encode('ascii', 'replace').decode()[:260]}...")


if __name__ == "__main__":
    main()
