"""Turn a question into reference context for the model, using the lookup library. CPU only; never loads the language model.

    from retrieve import build_context
    ctx = build_context("What does event ID 4625 mean?", max_chars=6000)
    ctx["context"]   # numbered reference blocks to put in front of the question
    ctx["sources"]   # one dict per block: n, kind, id, title, heading, source, url, licence, fetched, score, chars

    python -I scripts/retrieve.py --context "What does event ID 4625 mean?"
    python -I scripts/retrieve.py --detect "What does event ID 4625 mean, and is TCP port 445 involved?"
    python -I scripts/retrieve.py --detect-file data/eval.jsonl        # run over a test set and show what was found

detect_ids() finds identifiers in a question (CVE, ATT&CK technique, CWE, CAPEC, Windows event ID, port, tool or cmdlet
name) so they are answered from exact records rather than search results. The adapter over the laptop-built library
(_Adapter) is written once lookup/ is on the desktop; until then build_context() returns an empty context and says why.

Numeric identifiers need a cue word next to them ("event 4625", "port 445", "445/tcp"): bare numbers in a question are
far more often RFC numbers, versions, years or document numbers (NIST SP 800-53), and a wrong exact record would be
presented to the model as authoritative.
"""
from __future__ import annotations

import functools
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = os.path.join(ROOT, "sources")

CVE = re.compile(r"\bCVE-(\d{4})-(\d{4,7})\b", re.I)
TECHNIQUE = re.compile(r"(?<![\w-])(T\d{4}(?:\.\d{3})?)(?![\w-])")
CWE = re.compile(r"\bCWE-(\d{1,4})\b", re.I)
CAPEC = re.compile(r"\bCAPEC-(\d{1,4})\b", re.I)
EVENT = re.compile(r"\b(?:windows\s+|security\s+|sysmon\s+)?event(?:\s*id|\s+code)?\s*[:#]?\s*(\d{1,5})\b|\beventid\s*[:=]?\s*(\d{1,5})\b", re.I)
PORT = re.compile(r"\bports?\s*(?:number\s*)?(\d{1,5})\b|\b(\d{1,5})/(?:tcp|udp)\b|\b(?:tcp|udp)/(\d{1,5})\b", re.I)
BACKTICK = re.compile(r"`([^`\n]{1,80})`")
TOOL_CUE = re.compile(r"(?:^|[\s(\"'])([A-Za-z][\w.+-]{1,40})\s+(?:command|tool|utility|man ?page|manual)\b", re.I)
# Tool names that are also everyday English words: accepted only inside backticks, never from cue words in prose
COMMON_WORDS = frozenset("""in at on to of is it be as an or if do go no so up we us me my by all and the for not can may use run set get put add see new
top cut find date time file test join sort yes true false look more less watch which who write read last kill wait sleep make link touch split
paste head tail tee man help info type open print echo exit select while until case home host free last users groups id env locale clear
history jobs fold expand column pr nl od fmt tr uniq comm diff patch bind trap alias source eval exec export shift times umask unset log
mail list show load save stop start restart status apply check build install update upgrade search query copy move rename send
""".split())
MAN_FOR = re.compile(r"\bman(?:ual)?\s+(?:page\s+)?(?:for|of)\s+`?([A-Za-z][\w.+-]{1,40})`?", re.I)
TOOL_FLAG = re.compile(r"(?:^|[\s(`\"'])([a-z][\w.+-]{1,40})\s+-{1,2}[A-Za-z]")
CMDLET_VERBS = ("Add Approve Assert Backup Block Build Checkpoint Clear Close Compare Complete Compress Confirm Connect Convert ConvertFrom "
                "ConvertTo Copy Debug Deny Deploy Disable Disconnect Dismount Edit Enable Enter Exit Expand Export Find Format Get Grant Group "
                "Hide Import Initialize Install Invoke Join Limit Lock Measure Merge Mount Move New Open Optimize Out Ping Pop Protect Publish "
                "Push Read Receive Redo Register Remove Rename Repair Request Reset Resize Resolve Restart Restore Resume Revoke Save Search "
                "Select Send Set Show Skip Split Start Step Stop Submit Suspend Switch Sync Test Trace Unblock Undo Uninstall Unlock "
                "Unprotect Unpublish Unregister Update Use Wait Watch Where Write ForEach Sort Tee").split()
CMDLET = re.compile(r"\b((?:" + "|".join(CMDLET_VERBS) + r")-[A-Z][A-Za-z0-9]{2,})\b")


@functools.lru_cache(maxsize=1)
def tool_names() -> frozenset[str]:
    """Names that have a man page, tldr page or Kali tool page in sources/ (lower case)."""
    names: set[str] = set()
    for d in ("kali/man-pages/text", "man-pages/debian/text", "man-pages/linux-man-pages/text"):
        p = os.path.join(SOURCES, d)
        if os.path.isdir(p):
            names.update(f.rsplit(".", 2)[0].lower() for f in os.listdir(p) if f.endswith(".txt"))
    tl = os.path.join(SOURCES, "man-pages", "tldr", "pages")
    if os.path.isdir(tl):
        for plat in ("common", "linux", "windows", "osx"):
            p = os.path.join(tl, plat)
            if os.path.isdir(p):
                names.update(f[:-3].lower() for f in os.listdir(p) if f.endswith(".md"))
    kt = os.path.join(SOURCES, "kali", "kali-tools")
    if os.path.isdir(kt):
        names.update(d.lower() for d in os.listdir(kt) if os.path.isdir(os.path.join(kt, d)) and not d.startswith("."))
    return frozenset(n for n in names if len(n) >= 2)


def _add(found: dict[str, list[str]], kind: str, value: str) -> None:
    if value not in found.setdefault(kind, []):
        found[kind].append(value)


def detect_ids(question: str) -> dict[str, list[str]]:
    """Identifiers in a question, by kind, in order of appearance. Keys present only when something was found:
    cve, technique, cwe, capec, event, port, cmdlet, tool."""
    found: dict[str, list[str]] = {}
    q = question
    for m in CVE.finditer(q):
        _add(found, "cve", f"CVE-{m.group(1)}-{m.group(2)}")
    for m in TECHNIQUE.finditer(q):
        _add(found, "technique", m.group(1))
    for m in CWE.finditer(q):
        _add(found, "cwe", f"CWE-{int(m.group(1))}")
    for m in CAPEC.finditer(q):
        _add(found, "capec", f"CAPEC-{int(m.group(1))}")
    for m in EVENT.finditer(q):
        v = m.group(1) or m.group(2)
        if v and 100 <= int(v) <= 65535:                 # Windows event IDs; excludes "event 1" style ordinals
            _add(found, "event", str(int(v)))
    for m in PORT.finditer(q):
        v = m.group(1) or m.group(2) or m.group(3)
        if v and 1 <= int(v) <= 65535:
            _add(found, "port", str(int(v)))
    for m in CMDLET.finditer(q):
        _add(found, "cmdlet", m.group(1))
    tools = tool_names()
    quoted = [(b.strip().split()[0], True) for b in BACKTICK.findall(q) if b.strip() and not b.strip().startswith("-")]
    cued = [(m.group(1), False) for rx in (TOOL_CUE, MAN_FOR, TOOL_FLAG) for m in rx.finditer(re.sub(r"`[^`]*`", " ", q))]
    for c, in_backticks in quoted + cued:
        name = re.sub(r"\(\d\w*\)$", "", c.strip("`'\"()[],.:;").lower())     # nmap(1) -> nmap
        if name in tools and not CMDLET.fullmatch(c) and (in_backticks or name not in COMMON_WORDS):
            _add(found, "tool", name)
    return found


# ---------------------------------------------------------------------------------------------------- the library
LOOKUP_DIR = os.environ.get("NEURALPROJ_LOOKUP", os.path.join(ROOT, "lookup"))
EXACT_ORDER = ("cve", "technique", "cwe", "capec", "event", "port", "cmdlet", "tool")


class LibraryUnavailable(RuntimeError):
    """The lookup library is missing, unreadable, or would not fit in the memory available."""


class _Adapter:
    """Thin wrapper over the laptop-built library. To be written against lookup/README.md once the folder is here.

    search(question, k, mode) -> list of passages, best first, each a dict with
        text, title, heading, source (path relative to sources/), url, licence, fetched (YYYY-MM-DD), score
        mode: "combined" (default), "keyword" or "meaning"
    exact(kind, ident) -> record dict (id, kind, text, source, url, licence, fetched) or None when not found
        kind: cve, technique, cwe, capec, event, port, tool (cmdlets are tools)
    """

    def __init__(self, path: str) -> None:
        raise LibraryUnavailable(f"found {path}, but the adapter for it has not been written yet (read its README.md first)")

    def search(self, question: str, k: int = 8, mode: str = "combined") -> list[dict]:
        raise NotImplementedError

    def exact(self, kind: str, ident: str) -> dict | None:
        raise NotImplementedError


_LIB: _Adapter | None = None


def library() -> _Adapter:
    """The lookup library, loaded on first use. Raises LibraryUnavailable with the reason."""
    global _LIB
    if _LIB is None:
        if not os.path.isdir(LOOKUP_DIR):
            raise LibraryUnavailable(f"no lookup library at {LOOKUP_DIR}")
        _LIB = _Adapter(LOOKUP_DIR)
    return _LIB


def free_ram_gb() -> float:
    """Physical memory available right now. Checked before loading anything large: this desktop has 16 GB and a training
    job that already holds part of it."""
    try:
        import psutil
        return psutil.virtual_memory().available / 2**30
    except Exception:  # noqa: BLE001
        return float("inf")


# ---------------------------------------------------------------------------------------------------- context building
def strip_code(text: str) -> str:
    return re.sub(r"```.*?```", " ", text, flags=re.S)


def trim(text: str, limit: int) -> str:
    """Shorten to `limit` characters at a paragraph or sentence boundary, never mid-line, and never inside an open code block."""
    text = text.strip()
    if len(text) <= limit:
        return text
    head = text[:limit]
    if head.count("```") % 2:
        head = head[:head.rfind("```")]
    for sep in ("\n\n", ". ", "\n"):
        cut = head.rfind(sep)
        if cut > limit * 0.5:
            return head[:cut + (1 if sep == ". " else 0)].rstrip() + " [...]"
    return head.rstrip() + " [...]"


def _label(kind: str, item: dict) -> str:
    if kind == "record":
        bits = [item.get("kind") or "record"] + ([f"fetched {item['fetched']}"] if item.get("fetched") else [])
        return f"{item.get('id', '')} ({', '.join(bits)})"
    title, heading = item.get("title") or "", item.get("heading") or ""
    name = title + (f" > {heading}" if heading and heading.lower() != title.lower() else "")
    return f"{name or 'Untitled'} ({item.get('source', '')}" + (f", fetched {item['fetched']}" if item.get("fetched") else "") + ")"


def build_context(question: str, max_chars: int = 6000, k: int = 8, per_block: int = 1800, max_records: int = 4) -> dict:
    """Reference material for one question, within `max_chars` characters of context text.

    Returns {"context": str, "sources": [...], "ids": {...}, "truncated": bool, "library": {"available": bool, "error": str|None}}.
    `context` is numbered blocks only ("[1] label" then text), records for identifiers in the question first, then search
    passages, best first. No instructions are included; the caller words those. `sources[i]` describes block i+1:
    n, kind ("record" or "passage"), id, title, heading, source, url, licence, fetched, score, chars.
    When the library is missing or fails, context is "" and library.available is False, so answering can go on without it.
    """
    ids = detect_ids(question)
    out = {"context": "", "sources": [], "ids": ids, "truncated": False, "library": {"available": True, "error": None}}
    try:
        lib = library()
        outside = detect_ids(strip_code(question))                     # identifiers in prose outrank ones inside pasted code
        wanted = [(t, v) for t in EXACT_ORDER for v in outside.get(t, [])]
        wanted += [(t, v) for t in EXACT_ORDER for v in ids.get(t, []) if (t, v) not in wanted]
        items: list[tuple[str, dict]] = []
        for t, v in wanted[:max_records]:
            rec = lib.exact("tool" if t == "cmdlet" else t, v)
            if rec:
                items.append(("record", rec))
        items += [("passage", p) for p in lib.search(question, k=k)]
    except Exception as e:  # noqa: BLE001 - the model must still be able to answer without the library
        out["library"] = {"available": False, "error": f"{type(e).__name__}: {e}"}
        return out
    parts, seen, used = [], set(), 0
    for kind, item in items:
        sig = (item.get("source"), item.get("heading"), (item.get("text") or "")[:200])
        if sig in seen or not (item.get("text") or "").strip():
            continue
        seen.add(sig)
        n = len(parts) + 1
        head = f"[{n}] {_label(kind, item)}"
        room = max_chars - used - len(head) - 3
        if room < 300:
            out["truncated"] = True
            break
        body = trim(item["text"], min(per_block, room))
        parts.append(f"{head}\n{body}")
        used += len(parts[-1]) + 2
        out["sources"].append({"n": n, "kind": kind, "id": item.get("id"), "title": item.get("title"), "heading": item.get("heading"),
                               "source": item.get("source"), "url": item.get("url"), "licence": item.get("licence"),
                               "fetched": item.get("fetched"), "score": item.get("score"), "chars": len(body)})
    out["context"] = "\n\n".join(parts)
    return out


def _cli() -> None:
    args = sys.argv[1:]
    if args[:1] == ["--context"] and len(args) > 1:
        print(json.dumps(build_context(" ".join(args[1:])), indent=1, ensure_ascii=False))
        return
    if args[:1] == ["--detect"] and len(args) > 1:
        print(json.dumps(detect_ids(" ".join(args[1:])), indent=1))
        return
    if args[:1] == ["--detect-file"] and len(args) > 1:
        import collections
        counts, examples = collections.Counter(), collections.defaultdict(list)
        n = 0
        for line in open(args[1], encoding="utf-8"):
            r = json.loads(line)
            q = r["messages"][-2]["content"] if "messages" in r else r.get("question", "")
            n += 1
            for kind, vals in detect_ids(q).items():
                counts[kind] += 1
                if len(examples[kind]) < int(args[2]) if len(args) > 2 else 12:
                    examples[kind].append((vals, " ".join(q.split())[:150]))
        print(f"{args[1]}: {n} questions")
        for kind, c in counts.most_common():
            print(f"\n{kind}: found in {c} questions")
            for vals, q in examples[kind]:
                print(f"   {vals}  <-  {q.encode('ascii', 'replace').decode()}")
        return
    print(__doc__)


if __name__ == "__main__":
    _cli()
