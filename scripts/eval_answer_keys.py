"""Write strict answer keys for the fixed test set, so a retrieval check can require the right document, not just the right collection.

    python -I scripts/eval_answer_keys.py            # writes data/eval_keys.jsonl and prints a summary

For each of the 741 questions in data/eval.jsonl: whether its answer comes from sources/ at all, the entity it is about
(from the `key` recorded in data/ranked.jsonl, joined on exact question text), and the file under sources/ that holds the
answer when that can be resolved. Many converters recorded a collection label ("mdn", "sigma", "tldr-pages") as the
source, so a source-only match would only prove the right collection was searched.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = os.path.join(ROOT, "sources")
DATA = os.path.join(ROOT, "data")


def rel(p: str) -> str:
    return os.path.relpath(p, S).replace("\\", "/")


def first(pattern: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(S, pattern), recursive=True), key=len)
    return rel(hits[0]) if hits else None


_INDEX: dict[tuple[str, str], dict[str, str]] = {}


def by_id(pattern: str, rx: str, ident: str, prefix: bool = False) -> str | None:
    """The file under `pattern` whose text declares this identifier (rule ids and entry names live inside the files).
    `rx` has one capture group for the identifier; each file is read once and indexed."""
    idx = _INDEX.get((pattern, rx))
    if idx is None:
        idx, needle = {}, re.compile(rx, re.M)
        for p in glob.glob(os.path.join(S, pattern), recursive=True):
            try:
                with open(p, encoding="utf-8", errors="replace") as fh:
                    text = fh.read(4_000_000)
            except OSError:
                continue
            for m in needle.finditer(text):
                idx.setdefault(m.group(1).strip(), rel(p))
        _INDEX[(pattern, rx)] = idx
    if ident in idx:
        return idx[ident]
    if prefix:                                    # keys that were truncated when the example was built
        return next((f for k, f in idx.items() if k.startswith(ident)), None)
    return None


UUID = r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"


def man_file(base: str) -> str | None:
    for d in ("kali/man-pages/text", "man-pages/debian/text", "man-pages/linux-man-pages/text"):
        p = os.path.join(S, d, base + ".txt")
        if os.path.isfile(p):
            return rel(p)
    return None


def resolve(source: str, key: str, kind: str) -> tuple[str | None, str]:
    """(file under sources/ or None, how it was resolved)."""
    if source and os.path.isfile(os.path.join(S, source)):
        return source, "source is a file"
    if source.startswith("pdf:"):
        txt = source[4:]
        pdf = re.sub(r"\.txt$", ".pdf", txt)
        return (pdf if os.path.isfile(os.path.join(S, pdf)) else None), "pdf (text extracted to data/pdf_text)"
    if source.startswith("man:"):
        return man_file(source[4:].split(" + ")[0]), "man page"
    if source.startswith("rfc") and source[3:].isdigit():
        return first(f"references/rfc/{source}.txt"), "rfc"
    pre, _, val = key.partition(":")
    if pre == "event":
        return first(f"logging/windows-security-auditing/event-{val}.txt"), "windows event"
    if pre == "audit":
        return first(f"logging/windows-security-auditing/{val}"), "audit policy"
    if pre == "ps":
        return (first(f"powershell/powershell-docs/reference/7.4/**/{val}.md") or first(f"powershell/powershell-docs/reference/5.1/**/{val}.md")
                or first(f"powershell/windows-powershell-docs/**/{val}.md")), "cmdlet"
    if pre == "mdn":
        return first(f"web/mdn-content/files/en-us/{val.lower()}/index.md"), "mdn page"
    if pre == "wincmd":
        return first(f"services/windows-server-docs/WindowsServerDocs/administration/windows-commands/{val.replace(' ', '-')}.md"), "windows command"
    if pre == "kb":
        return first(f"services/windows-support-articles/**/{val}"), "support article"
    if pre == "nse":
        return first(f"tool-docs/nmap/scripts/{val}.nse"), "nse script"
    if pre == "lolbin":
        return first(f"tool-docs/lolbas/yml/**/{val.replace('.exe', '').replace('.dll', '')}.yml"), "lolbas entry"
    if pre == "alert":
        return first(f"indicators/prometheus-operator-runbooks/content/**/{val}.md"), "runbook"
    if pre == "osquery":
        return first(f"indicators/osquery/specs/**/{val}.table"), "osquery table"
    if pre == "cac":
        return first(f"remediation/compliance-as-code/**/{val}/rule.yml"), "hardening rule"
    if pre == "legal":
        return first(f"legal/**/{val}.txt"), "legal text"
    if pre == "sigma":
        return by_id("tool-docs/sigma/rules/**/*.yml", r"^id:\s*" + UUID + r"\s*$", val), "sigma rule id"
    if pre == "splunk":
        return by_id("remediation/splunk-security-content/detections/**/*.yml", r"^id:\s*" + UUID + r"\s*$", val), "splunk detection id"
    if pre == "elastic":
        return by_id("remediation/elastic-detection-rules/rules/**/*.toml", r"rule_id\s*=\s*\"" + UUID + "\"", val), "elastic rule id"
    if pre == "sentinel":
        return (by_id("logging/azure-sentinel/Detections/**/*.yaml", r"^id:\s*" + UUID + r"\s*$", val)
                or by_id("logging/azure-sentinel/Hunting Queries/**/*.yaml", r"^id:\s*" + UUID + r"\s*$", val)), "sentinel id"
    if pre == "tool" and source.startswith("tldr"):
        return first(f"man-pages/tldr/pages/*/{val.lower().replace(' ', '-')}.md"), "tldr page"
    if pre == "tool" and source.startswith("kali-tools"):
        return (first(f"kali/kali-tools/{val}/index.md") or by_id("kali/kali-tools/*/index.md", r"^ ##### (\S+)\s*$", val)), "kali tool page"
    if pre == "artifact":
        return by_id("logging/forensic-artifacts/artifacts/data/*.yaml", r"^name:\s*(\S+)\s*$", val), "forensic artifact (one of many in the file)"
    if pre == "sca":
        return by_id("logging/wazuh-ruleset/ruleset/sca/**/*.yml", r"title:\s*\"([^\"\n]+)\"", val, prefix=True), "wazuh check (one of many in the file)"
    if pre == "service":
        for d, f in (("awesome-prometheus-alerts", "indicators/awesome-prometheus-alerts/_data/rules.yml"),):
            if source.startswith(d):
                return f, "service section of a single rules file"
        if source.startswith("netdata"):
            name = val.replace(" ", "_")
            return first(f"indicators/netdata-health/src/health/health.d/{name}.conf"), "netdata health file"
        return None, "service group without a single file"
    if pre == "codeql":                           # key is codeql:<language>:<query file name split at capitals>
        lang, _, name = val.partition(":")
        return first(f"testing/codeql/{lang}/**/{name.replace(' ', '')}.qhelp"), "codeql query help"
    for d in ("lua/manual", "prioritization/first"):   # labels that are file stems
        stem = source if source.endswith(".txt") else source + ".txt"
        if os.path.isfile(os.path.join(S, d, stem)):
            return f"{d}/{stem}", "file stem"
    if re.fullmatch(r"(CVE-\d{4}-\d+|T\d{4}(\.\d{3})?|CWE-\d+|CAPEC-\d+)", key):
        return None, "record id (no single file; match the record)"
    return None, "collection label only"


def main() -> None:
    evals = [json.loads(line) for line in open(os.path.join(DATA, "eval.jsonl"), encoding="utf-8")]
    want = {r["messages"][-2]["content"]: i for i, r in enumerate(evals)}
    keys: dict[int, dict] = {}
    with open(os.path.join(DATA, "ranked.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if '"selected": true' not in line:
                continue
            c = json.loads(line)
            i = want.get(c["q"])
            if i is not None and i not in keys:
                keys[i] = {"key": c.get("key", ""), "source": c.get("source", ""), "kind": c.get("kind", "")}
    out, tally, by_how = [], collections.Counter(), collections.Counter()
    for i, r in enumerate(evals):
        m = r["meta"]
        k = keys.get(i, {})
        source, key = m.get("source", ""), k.get("key", "")
        in_sources = not (source.startswith("cyber-security/") or source.startswith("dolly"))
        file, how = resolve(source, key, m["kind"]) if in_sources else (None, "not answerable from sources/")
        if in_sources:
            tally["answerable"] += 1
            tally["resolved to one file" if file else "entity key only" if key and how != "collection label only" else "collection only"] += 1
            by_how[how + (" -> file" if file else "")] += 1
        out.append({"index": i, "question": r["messages"][-2]["content"], "topic": m["topic"], "kind": m["kind"], "source": source,
                    "answerable_from_sources": in_sources, "entity_key": key, "file": file, "resolved_by": how})
    with open(os.path.join(DATA, "eval_keys.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"questions: {len(evals)}; joined to a ranked entry: {len(keys)}")
    for k in ("answerable", "resolved to one file", "entity key only", "collection only"):
        print(f"  {k:24s} {tally[k]}")
    print("by how the answer location was determined:")
    for k, v in by_how.most_common():
        print(f"  {v:4d}  {k}")
    print("wrote data/eval_keys.jsonl")


if __name__ == "__main__":
    main()
