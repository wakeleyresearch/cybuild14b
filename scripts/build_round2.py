"""Round two of the candidate build: more and better-combined examples for the topics the first run measured as weakest.

    python -I scripts/build_round2.py          # writes data/candidates_r2.jsonl and data/candidates_r2_stats.json

Adds to (does not replace) data/candidates.jsonl from build_candidates.py; rank_dataset.py reads both.
Focus: logging (parsing, locations, what to alert on), Lua (language reference and Lua as embedded in security tools),
tools (task cards, examples), prioritization (decision tables, CVSS breakdowns, multi-item triage), then services,
networking, kernel hardening, testing and browser tooling. PDFs must first be extracted with extract_pdfs.py.

Nothing from the sources is executed. YAML is read with safe_load only; Python and Lua files are read as text.
"""
from __future__ import annotations

import collections
import csv
import gzip
import html
import json
import os
import re
import sys
import unicodedata

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_candidates as bc  # noqa: E402
from build_candidates import G, S, ROOT, clean, cut, docs, emit, front_matter, guard, pick, read, sections  # noqa: E402

rng = bc.rng
PDF = os.path.join(ROOT, "data", "pdf_text")
SafeLoader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def yload(path: str, limit: int = 4_000_000):
    try:
        return yaml.load(read(path, limit), Loader=SafeLoader)
    except Exception:  # noqa: BLE001
        return None


def first_sentence(text: str, limit: int = 220) -> str:
    t = " ".join(str(text or "").split())
    m = re.search(r"(.{20,%d}?[.!?])(\s|$)" % limit, t)
    return (m.group(1) if m else t[:limit]).strip()


def lower1(s: str) -> str:
    return s[:1].lower() + s[1:] if s[:2] != s[:2].upper() else s


# ------------------------------------------------------------------ markup helpers (DocBook XML/SGML, HTML)
CODE_TAGS = r"programlisting|screen|literallayout|example|pre|synopsis|highlight"
INLINE_TAGS = (r"literal|code|command|filename|option|varname|constant|function|tt|var|directive|parameter|envar|systemitem|replaceable|value|"
               r"structfield|type|application|keycap|userinput|refentrytitle")


def markup_text(t: str) -> str:
    """Tagged documentation -> plain text with fenced code blocks and backticked identifiers."""
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"<indexterm\b.*?</indexterm>", "", t, flags=re.S)                # index entries are not part of the text
    # a cross-reference renders as its target's name in the published manual; here the target id is the best stand-in
    t = re.sub(r'<xref\s+linkend="([^"]+)"\s*/?>', lambda m: "`" + m.group(1)[4:].replace("-", "_") + "`" if m.group(1).startswith("guc-") else m.group(1).replace("-", " "), t)
    t = re.sub(r"<!\[CDATA\[(.*?)\]\]>", lambda m: html.escape(m.group(1)), t, flags=re.S)
    t = re.sub(rf"<({CODE_TAGS})\b[^>]*>(.*?)</\1>", lambda m: "\n```\n" + html.unescape(re.sub(r"<[^>]+>", "", m.group(2))).strip("\n") + "\n```\n", t, flags=re.S)
    t = re.sub(rf"<({INLINE_TAGS})\b[^>]*>(.*?)</\1>", lambda m: "`" + " ".join(re.sub(r"<[^>]+>", "", m.group(2)).split()) + "`", t, flags=re.S)
    t = re.sub(r"<(listitem|li)\b[^>]*>\s*(<(para|p)\b[^>]*>)?\s*", "\n- ", t)
    t = re.sub(r"</(para|p|listitem|li|title|term|row|entry|member|h\d|div|tr|simpara|refpurpose|note|warning|caution)>", "\n", t)
    t = html.unescape(re.sub(r"<[^>]+>", "", t))
    out, code = [], False
    for ln in t.split("\n"):                      # prose in these formats is indented and hard-wrapped; code is left alone
        if ln.strip() == "```":
            code = not code
            out.append("```")
        else:
            out.append(ln.rstrip() if code else ln.strip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


SECT_RE = re.compile(r"<(sect[1-5]|section|chapter|refsect[1-3]|appendix|simplesect)\b[^>]*>\s*(?:<\?[^>]*\?>\s*)?(?:<\w+info>.*?</\w+info>\s*)?<title[^>]*>(.*?)</title>", re.S)


def docbook_sections(raw: str) -> list[tuple[str, str]]:
    raw = re.sub(r"<indexterm\b.*?</indexterm>", "", raw, flags=re.S)
    ms = list(SECT_RE.finditer(raw))
    return [(html.unescape(" ".join(re.sub(r"<[^>]+>", "", m.group(2)).split())), raw[m.end():ms[i + 1].start() if i + 1 < len(ms) else len(raw)]) for i, m in enumerate(ms)]


def docbook(kind: str, topic: str, tier: str, path: str, ctx: str, *, cap: int = 60, min_body: int = 350, licence: str = "") -> None:
    n = 0
    for title, body in docbook_sections(read(path, 6_000_000)):
        text = markup_text(body)
        if not title or len(text) < min_body or bc.SKIP_HEAD.match(title) or n >= cap:
            continue
        emit(kind, topic, tier, pick(bc.DOC_Q, h=title, ctx=ctx), text, source=os.path.relpath(path, S).replace("\\", "/"), key=f"{kind}:{os.path.basename(path)}", licence=licence)
        n += 1


def varlist(raw: str) -> list[tuple[str, str]]:
    """[(term, description)] for DocBook variable lists, which is how option references are written."""
    out = []
    raw = re.sub(r"<indexterm\b.*?</indexterm>", "", raw, flags=re.S)
    for block in re.findall(r"<varlistentry\b[^>]*>(.*?)</varlistentry>", raw, re.S):
        terms = [" ".join(re.sub(r"<[^>]+>", "", t).split()) for t in re.findall(r"<term\b[^>]*>(.*?)</term>", block, re.S)]
        item = re.search(r"<listitem\b[^>]*>(.*)</listitem>", block, re.S)
        if terms and item:
            out.append((html.unescape(", ".join(t for t in terms if t)), markup_text(item.group(1))))
    return out


def html_to_md(raw: str) -> str:
    t = re.sub(r"<(script|style|head)\b.*?</\1>", "", raw, flags=re.S | re.I)
    t = re.sub(r'<div id="(nav|head|foot)">.*?</div>', "", t, flags=re.S)
    t = re.sub(r"<pre[^>]*>(.*?)</pre>", lambda m: "\n```\n" + html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip("\n") + "\n```\n", t, flags=re.S | re.I)
    t = re.sub(r"<h([1-4])[^>]*>(.*?)</h\1>", lambda m: "\n\n" + "#" * int(m.group(1)) + " " + " ".join(re.sub(r"<[^>]+>", "", m.group(2)).split()) + "\n\n", t, flags=re.S | re.I)
    t = re.sub(r"<(code|tt)\b[^>]*>(.*?)</\1>", lambda m: "`" + re.sub(r"<[^>]+>", "", m.group(2)) + "`", t, flags=re.S | re.I)
    t = re.sub(r"<li[^>]*>", "\n- ", t, flags=re.I)
    t = re.sub(r"</t[dh]>\s*<t[dh][^>]*>", " | ", t, flags=re.I)      # keep table cells apart
    t = re.sub(r"</(p|div|li|ul|ol|tr|table|dd|dt)>", "\n", t, flags=re.I)
    return html.unescape(re.sub(r"<[^>]+>", "", t))


RST_CODE = re.compile(r"^(\s*)\.\. (?:code-block|code|sourcecode|parsed-literal)::\s*(\S*)\s*$")
RST_NOTE = re.compile(r"^(\s*)\.\. (note|warning|tip|important|caution|attention|seealso|versionadded|versionchanged|deprecated)::\s*(.*)$")
RST_DROP = re.compile(r"^\s*\.\. (?:_[^:]*:|\[\w+\]|toctree::|contents::|include::|literalinclude::|index::|highlight::|default-domain::|only::|raw::|image::|figure::|kernel-doc::|SPDX)")


def indent_of(s: str) -> int:
    return len(s) - len(s.lstrip())


def rst_to_md(t: str) -> str:
    """reStructuredText -> the plain Markdown the rest of the set uses: fenced code, backticked names, no roles or directives.
    Section headings are left alone so sections() can still find them."""
    lines, out, i = t.split("\n"), [], 0
    while i < len(lines):
        ln = lines[i]
        m = RST_CODE.match(ln)
        literal = not m and ln.rstrip().endswith("::") and not ln.lstrip().startswith("..")
        if m or literal:
            base = indent_of(ln)
            if literal and ln.strip() != "::":
                out.append(ln.rstrip()[:-1])                                      # "like this::" introduces a literal block
            j = i + 1
            while m and j < len(lines) and re.match(r"^\s+:[\w-]+:", lines[j]):   # the directive's options
                j += 1
            block = []
            while j < len(lines) and (not lines[j].strip() or indent_of(lines[j]) > base):
                block.append(lines[j])
                j += 1
            while block and not block[-1].strip():
                block.pop()
            while block and not block[0].strip():
                block.pop(0)
            if block:
                ind = min(indent_of(b) for b in block if b.strip())
                out += ["", "```" + (m.group(2) if m else "")] + [b[ind:] for b in block] + ["```", ""]
            i = j
            continue
        n = RST_NOTE.match(ln)
        if n:
            out.append(f"{n.group(1)}{n.group(2).capitalize()}: {n.group(3)}".rstrip())
            i += 1
            continue
        if RST_DROP.match(ln):
            base, i = indent_of(ln), i + 1
            while i < len(lines) and lines[i].strip() and indent_of(lines[i]) > base:   # its option lines
                i += 1
            continue
        out.append(ln)
        i += 1
    t = "\n".join(out)
    t = re.sub(r":[a-zA-Z:+-]+:`([^`<]+?)\s*<[^`>]+>`", r"\1", t)                  # :ref:`text <target>`
    t = re.sub(r":[a-zA-Z:+-]+:`~?([^`]+)`", r"`\1`", t)                           # :func:`name`
    t = re.sub(r"``([^`\n]+)``", r"`\1`", t)
    return re.sub(r"`([^`<\n]+?)\s*<(https?://[^`>\s]+)>`__?", r"\1 (\2)", t)      # `text <url>`_


def rst_docs(kind: str, topic: str, tier: str, files: list[str], ctx, *, cap_per_file: int = 12, min_body: int = 350, licence: str = "") -> None:
    """build_candidates.docs() for .rst files, with the markup converted first."""
    for path in files:
        if bc.JUNK_FILE.match(os.path.basename(path)):
            continue
        text = rst_to_md(read(path))
        title = next((h for h, _ in sections(clean(text)) if h), "")
        if bc.SKIP_HEAD.match(title):
            continue
        sectioned(kind, topic, tier, text, ctx(path) if callable(ctx) else ctx, source=os.path.relpath(path, S).replace("\\", "/"),
                  key=f"{kind}:{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}", cap=cap_per_file, min_body=min_body, licence=licence, title=title)


def sectioned(kind: str, topic: str, tier: str, text: str, ctx: str, *, source: str, key: str, cap: int = 12, min_body: int = 350, licence: str = "", title: str = "") -> None:
    """Like build_candidates.docs(), for text that is already in memory."""
    n = 0
    for head, body in sections(clean(text)):
        body = body.strip()
        label = head or title
        if not label or len(body) < min_body or bc.SKIP_HEAD.match(label) or n >= cap:
            continue
        emit(kind, topic, tier, pick(bc.DOC_Q, h=label, ctx=f"{ctx} ({title})" if title and label != title else ctx), body, source=source, key=key, licence=licence)
        n += 1


# ------------------------------------------------------------------ PDFs (text extracted by extract_pdfs.py)
PDF_HEAD = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2}){0,3}\.?|[A-Z]\.\d{1,2}(?:\.\d{1,2})*\.?)\s+([A-Z][^\n]{3,90})$")
PDF_Q = ["According to {d}, what should I know about \"{h}\"?", "What does {d} say about \"{h}\"?", "Explain \"{h}\" as covered in {d}."]
DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
TRAILING = {"or", "and", "the", "of", "to", "a", "an", "in", "for", "with", "that", "is", "are", "by", "on", "as"}


def pdf_doc(kind: str, topic: str, tier: str, rel: str, label: str, *, cap: int = 40, min_body: int = 500, licence: str = "US government work") -> None:
    path = os.path.join(PDF, rel)
    if not os.path.isfile(path):
        bc.errors[f"pdf text missing: {rel}"] += 1
        return
    secs, head, buf = [], "", []
    text = unicodedata.normalize("NFKC", read(path, 12_000_000))                # ligatures such as "fi" come out of PDFs as single characters
    text = re.sub(r"/(zero|one|two|three|four|five|six|seven|eight|nine)\.pnum", lambda m: str(DIGITS.index(m.group(1))), text)   # digits some fonts emit as glyph names
    for ln in text.replace("\f", "\n").split("\n"):
        s = ln.strip()
        if re.search(r"(?:\.\s?){5,}\s*\d+\s*$", s) or re.fullmatch(r"\d{1,4}", s):   # contents-page entries (dot leaders, spaced or not) and bare page numbers
            continue
        m = PDF_HEAD.match(s)
        # a heading is short, is not a sentence, does not trail off into the next line, and does not end in a page number
        if m and not s.endswith((".", ",", ";", ":")) and len(s.split()) <= 12 and ". " not in m.group(1) and s.split()[-1].lower() not in TRAILING \
                and not re.search(r"\s\d{1,4}$", m.group(1)):
            if buf:
                secs.append((head, "\n".join(buf)))
            head, buf = m.group(1).strip(), []
        else:
            buf.append(s)
    if buf:
        secs.append((head, "\n".join(buf)))
    n = 0
    for head, body in secs:
        body = clean(re.sub(r"(?<![.:;!?])\n(?=[a-z(\[])", " ", body))            # re-flow lines broken mid-sentence
        if not head or len(body) < min_body or bc.SKIP_HEAD.match(head) or n >= cap:
            continue
        lines = [ln for ln in body.split("\n") if ln.strip()]
        if sum(len(ln) < 35 for ln in lines) > 0.45 * len(lines):                 # mostly short fragments: a table that did not survive extraction
            bc.stats[f"dropped:{kind}:table-like"] += 1
            continue
        if "�" in body or "�" in head:                                   # characters the PDF's fonts could not be mapped to text
            bc.stats[f"dropped:{kind}:unreadable characters"] += 1
            continue
        emit(kind, topic, tier, pick(PDF_Q, h=head, d=label), body, source="pdf:" + rel.replace("\\", "/"), key=f"pdf:{rel}", licence=licence)
        n += 1


# ================================================================== LOGGING
def src_name(path: str) -> str:
    return re.sub(r"_(decoders|rules)$", "", re.sub(r"^\d+-", "", os.path.basename(path)[:-4])).replace("_", " ").replace("-", " ")


def tagval(block: str, tag: str) -> str:
    m = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", block, re.S)
    return html.unescape(" ".join(m.group(1).split())) if m else ""


def wazuh_logs() -> None:
    """Per log source: how it is parsed (decoders) joined with what to alert on (rules that consume those decoders)."""
    lic = "GPLv2 (Wazuh ruleset)"
    dec_owner, dec_cards = {}, {}
    for path in G("logging/wazuh-ruleset/ruleset/decoders/*.xml"):
        text, name = read(path), src_name(path)
        decs = re.findall(r'<decoder name="([^"]+)"[^>]*>(.*?)</decoder>', text, re.S)
        if not decs:
            continue
        lines, shown = [], set()
        for dname, body in decs:
            dec_owner.setdefault(dname, path)
            bits = [f"program name `{tagval(body, 'program_name')}`" if tagval(body, "program_name") else "", f"prematch `{tagval(body, 'prematch')}`" if tagval(body, "prematch") else "",
                    f"regex `{tagval(body, 'regex')}`" if tagval(body, "regex") else "", f"extracts {tagval(body, 'order')}" if tagval(body, "order") else ""]
            line = f"- **{dname}**" + (f" (child of `{tagval(body, 'parent')}`)" if tagval(body, "parent") else "") + ": " + "; ".join(b for b in bits if b)
            if line not in shown and len(lines) < 12 and any(bits):
                shown.add(line)
                lines.append(line)
        sample = "\n\n".join(m.group(0) for m in list(re.finditer(r'<decoder name="[^"]+"[^>]*>.*?</decoder>', text, re.S))[:3])
        dec_cards[path] = (name, "\n".join(lines), cut(sample, 1400))
    linked = set()
    for path in G("logging/wazuh-ruleset/ruleset/rules/*.xml"):
        text, name = read(path), src_name(path)
        rules = re.findall(r'(<rule id="(\d+)" level="(\d+)"[^>]*>(.*?)</rule>)', text, re.S)
        if not rules:
            continue
        ranked = sorted(((int(lvl), rid, tagval(body, "description"), tagval(body, "match") or tagval(body, "regex"), re.findall(r"<id>(T[\d.]+)</id>", body))
                         for _, rid, lvl, body in rules if tagval(body, "description")), key=lambda r: -r[0])
        alerts = "\n".join(f"- level {lvl} (rule {rid}): {d}" + (f" Matches `{pat}`." if pat else "") + (f" ATT&CK {', '.join(mit)}." if mit else "") for lvl, rid, d, pat, mit in ranked[:16] if lvl >= 3)
        owners = {dec_owner[d] for d in re.findall(r"<decoded_as>([^<]+)</decoded_as>", text) if d in dec_owner}
        if alerts and owners:
            linked |= owners
            dn, dl, ds = dec_cards[sorted(owners)[0]]
            emit("log_source_card", "logging", "B", pick(["For {n} logs: how are they parsed into fields, and what should I alert on?", "Walk me through handling {n} logs: field extraction first, then the conditions worth alerting on."], n=name),
                 f"**Parsing.** Wazuh decodes these in two stages: a parent decoder recognizes the log by program name or a prematch pattern, then child decoders pull fields out with a regex. "
                 f"The names after \"extracts\" are the fields rules can test.\n\n{dl}\n\n**Alerting.** Conditions in the stock ruleset, highest severity first (Wazuh's 0-15 scale):\n\n{alerts}",
                 source="wazuh-ruleset (decoders + rules)", key=f"logsrc:{name}", joined=True, licence=lic)
        elif alerts:
            emit("log_alert_conditions", "logging", "B", pick(["What should I alert on in {n} logs? Rank the conditions by severity.", "Which {n} log events matter for security monitoring?"], n=name),
                 f"Conditions worth alerting on in {name} logs, highest severity first (Wazuh's 0-15 scale):\n\n{alerts}", source="wazuh-ruleset (rules)", key=f"logsrc:{name}", licence=lic)
        n = 0
        for raw, rid, lvl, body in rules:
            d = tagval(body, "description")
            if int(lvl) >= 8 and d and (tagval(body, "match") or tagval(body, "regex")) and n < 5:
                emit("wazuh_rule", "logging", "B", f"Write a Wazuh rule for {name} logs that fires on: {d}",
                     "```xml\n" + "\n".join(ln[2:] if ln.startswith("  ") else ln for ln in raw.split("\n")) + "\n```\n\n"
                     + ("`if_matched_sid` with `frequency` and `timeframe` makes it fire only when the referenced rule has matched that many times inside the window. " if "<if_matched_sid>" in raw else
                        f"`if_sid` chains it to the rule that first recognized the log, so the pattern is only tested against {name} events. " if "<if_sid>" in raw else "")
                     + f"Level {lvl} sets its severity.",
                     source="wazuh-ruleset (rules)", key=f"wazuhrule:{rid}", min_a=120, licence=lic)
                n += 1
    for path, (name, dl, ds) in dec_cards.items():
        if path not in linked and dl:
            emit("log_decoder", "logging", "B", pick(["How do I parse {n} logs into fields? Show working decoders.", "Show me how {n} log lines are decoded field by field."], n=name),
                 f"Wazuh's decoders for {name} logs. A parent decoder recognizes the log; child decoders extract fields with a regex, named in order:\n\n{dl}\n\n```xml\n{ds}\n```",
                 source="wazuh-ruleset (decoders)", key=f"logsrc:{name}", licence=lic)


GROK_NAMES = {"grok-patterns": "the base grok pattern set (numbers, IP addresses, paths, timestamps, syslog pieces)", "httpd": "Apache httpd access and error logs", "linux-syslog": "Linux syslog",
              "firewalls": "firewall logs (Cisco ASA, iptables, Shorewall, SFW2)", "bind": "BIND DNS query logs", "haproxy": "HAProxy logs", "aws": "AWS S3, ELB and CloudFront access logs",
              "zeek": "Zeek logs", "bro": "Bro (legacy Zeek) logs", "squid": "Squid proxy access logs", "postgresql": "PostgreSQL logs", "mongodb": "MongoDB logs", "redis": "Redis logs",
              "exim": "Exim mail logs", "nagios": "Nagios logs", "junos": "Juniper JunOS firewall logs", "java": "Java stack traces and Tomcat/Catalina logs", "rails": "Ruby on Rails logs"}


def log_parsing() -> None:
    for path in G("logging/logstash-grok-patterns/patterns/ecs-v1/*"):
        name = GROK_NAMES.get(os.path.basename(path), os.path.basename(path) + " logs")
        chunk, size, part = [], 0, 0
        for block in re.split(r"\n\s*\n", read(path)) + [""]:
            if block and size + len(block) < 2400:
                chunk.append(block)
                size += len(block)
                continue
            if chunk and size > 150:
                lead = next((ln.lstrip("# ").strip() for ln in "\n".join(chunk).split("\n") if ln.startswith("#") and len(ln) > 4), "")
                q = f"Give me grok patterns for parsing {name}." if part == 0 else f"Grok patterns for {name}: show the ones covering {lower1(lead) if lead else 'the remaining formats'}."
                emit("grok_patterns", "logging", "A", q, f"Logstash's bundled patterns, using ECS field names. Reference one from a grok filter as `%{{NAME}}`, or `%{{NAME:field}}` to capture into a field:\n\n```\n" + "\n\n".join(chunk) + "\n```",
                     source="logstash-patterns-core", key=f"grok:{os.path.basename(path)}", min_a=200, licence="Apache-2.0")
                part += 1
            chunk, size = ([block], len(block)) if block else ([], 0)
    docs("log_parsing_docs", "logging", "A", G("logging/logstash-filter-*/docs/*.asciidoc"), lambda p: "the Logstash " + os.path.basename(os.path.dirname(os.path.dirname(p))).split("-")[-1] + " filter",
         cap_per_file=40, min_body=200, licence="Apache-2.0")
    docs("log_parsing_docs", "logging", "A", G("logging/loki-docs/docs/**/*.md"), "Grafana Loki and LogQL", cap_per_file=14, min_body=250, licence="AGPL-3.0 (Loki repository)")
    for path in G("logging/azure-sentinel/Parsers/**/*.yaml"):
        d = yload(path)
        q = d.get("ParserQuery") if isinstance(d, dict) else None
        prod = ((d or {}).get("Product") or {}).get("Name") if isinstance(d, dict) else None
        schema = ((d or {}).get("Normalization") or {}).get("Schema") if isinstance(d, dict) else None
        if not (isinstance(q, str) and prod and schema and 300 < len(q) < 4300):
            continue
        emit("asim_parser", "logging", "B", pick(["Write a KQL parser that normalizes {p} logs to the ASIM {s} schema in Microsoft Sentinel.", "How do I map {p} events onto Sentinel's ASIM {s} schema? Show the parser query."], p=prod, s=schema),
             f"{' '.join(str(d.get('Description', '')).split())}\n\n```kql\n{q.strip()}\n```", source="azure-sentinel/Parsers", key=f"asim:{d.get('ParserName', os.path.basename(path))}", licence="MIT")


def log_locations() -> None:
    for path in G("logging/kape-files/Targets/**/*.tkape"):
        d = yload(path)
        targets = [t for t in (d.get("Targets") or []) if isinstance(t, dict) and t.get("Path")] if isinstance(d, dict) else []
        if not targets or not d.get("Description") or all(str(t["Path"]).lower().endswith(".tkape") for t in targets):
            continue
        rows = []
        for t in targets[:16]:
            if str(t["Path"]).lower().endswith(".tkape"):
                continue
            rows.append(f"- **{t.get('Name', '')}**" + (f" ({t['Category']})" if t.get("Category") else "") + f": `{t['Path']}`" + (f", files matching `{t['FileMask']}`" if t.get("FileMask") else "")
                        + (", including subfolders" if t.get("Recursive") else "") + (f". {' '.join(str(t['Comment']).split())}" if t.get("Comment") else ""))
        stem = os.path.basename(path)[:-6]
        emit("artifact_collection", "logging", "B", pick(["Which files should I collect from a Windows system for \"{d}\"?", "Where on disk does Windows keep the evidence for \"{d}\"?"], d=str(d["Description"]).strip().rstrip(".")),
             f"These are the locations KAPE's `{stem}` target collects (`%user%` stands for each profile):\n\n" + "\n".join(rows), source="kape-files", key=f"kape:{stem}", min_a=110, licence="MIT")
    for path in G("logging/velociraptor-artifacts/artifacts/definitions/**/*.yaml"):
        if os.path.relpath(path, os.path.join(S, "logging/velociraptor-artifacts/artifacts/definitions")).split(os.sep)[0] not in ("Windows", "Linux", "MacOS", "Generic", "Network"):
            continue
        d = yload(path)
        if not isinstance(d, dict) or not d.get("name") or not d.get("description"):
            continue
        desc = str(d["description"]).strip()
        params = "\n".join(f"- `{p.get('name')}`" + (f" (default `{str(p.get('default'))[:80]}`)" if p.get("default") not in (None, "") and len(str(p.get("default"))) < 80 else "")
                           + (f": {' '.join(str(p.get('description')).split())}" if p.get("description") else "") for p in (d.get("parameters") or [])[:10] if isinstance(p, dict))
        vql = "\n\n".join(str(s.get("query")).strip() for s in (d.get("sources") or []) if isinstance(s, dict) and s.get("query"))
        a = f"Use the `{d['name']}` artifact.\n\n{desc}" + (f"\n\n**Parameters:**\n{params}" if params else "") + (f"\n\n```vql\n{vql}\n```" if 0 < len(vql) < 2800 else "")
        emit("velociraptor_artifact", "logging", "B", pick(["Which Velociraptor artifact covers this, and how does it work: {s}", "In Velociraptor, how do I collect this: {s}"], s=first_sentence(desc)),
             a, source="velociraptor artifacts", key=f"vr:{d['name']}", licence="AGPL-3.0")
    text = read(os.path.join(S, "indicators/auditd-rules/audit.rules")) if os.path.isfile(os.path.join(S, "indicators/auditd-rules/audit.rules")) else ""
    for m in re.finditer(r"^# ([^\n#-][^\n]*?) -{5,}\n(.*?)(?=^# [^\n#-][^\n]*? -{5,}\n|\Z)", text, re.M | re.S):
        chunk, size = [], 0
        for block in re.split(r"\n\s*\n", m.group(2)) + [""]:
            if block and size + len(block) < 1900:
                chunk.append(block)
                size += len(block)
                continue
            body = "\n\n".join(chunk)
            if re.search(r"^-[aw] ", body, re.M):
                sub = next((ln.lstrip("# ").strip() for ln in body.split("\n") if ln.startswith("## ")), "")
                emit("auditd_rules", "logging", "B", f"Write Linux auditd rules for: {m.group(1).strip()}" + (f" ({lower1(sub)})" if sub else "") + ".",
                     f"```\n{body}\n```\n\n`-w path -p perms -k key` watches a file or directory; `-a always,exit ... -S syscall` audits system calls; the `-k` key is what you search for later with `ausearch -k`.",
                     source="Neo23x0/auditd", key=f"auditd:{m.group(1)[:30]}", min_a=150, licence="Apache-2.0")
            chunk, size = ([block], len(block)) if block else ([], 0)


def windows_logging() -> None:
    for path in G("logging/windows-security-auditing/event-*.txt"):
        text = read(path)
        ev, title = re.search(r"event-(\d+)", path).group(1), text.split("\n", 1)[0].strip()
        m = re.search(r"\nField Descriptions:?\s*\n(.*?)(?=\nSecurity Monitoring Recommendations|\Z)", text, re.S)
        if m and len(m.group(1).strip()) > 300:
            emit("windows_event_fields", "logging", "A", pick(["In Windows Security event {e}, what do the fields mean?", "Explain the fields of Windows event ID {e} ({t}).", "How do I read the fields in a Windows {e} event?"], e=ev, t=title.rstrip(".")),
                 f"**{title}**\n\n{cut(m.group(1).strip(), 5200)}", source="microsoft-security-auditing", key=f"eventfields:{ev}", licence="CC BY 4.0")
    for path in G("logging/sysmon-modular/*/*.xml"):
        folder, stem = os.path.basename(os.path.dirname(path)), os.path.basename(path)[:-4]
        m = re.match(r"((?:\d+_)+)(.+)", folder)
        text = read(path).strip()
        if not m or folder == "templates" or not 250 < len(text) < 4500:
            continue
        ids = m.group(1).strip("_").replace("_", "/")
        mode = "exclude" if stem.startswith("exclude") else "include"
        subject = re.sub(r"^(include|exclude)_?", "", stem).replace("_", " ").strip() or m.group(2).replace("_", " ")
        techs = sorted(set(re.findall(r"technique_id=(T[\d.]+)", text)))
        emit("sysmon_rule", "logging", "B", f"Write a Sysmon configuration fragment for event ID {ids} ({m.group(2).replace('_', ' ')}) to {mode} {subject}.",
             f"```xml\n{text}\n```\n\n`onmatch=\"{mode}\"` means matching events are {'logged' if mode == 'include' else 'dropped'}."
             + (f" The `name` attribute is written to the event's RuleName field, here tagging matches with ATT&CK {', '.join(techs[:6])}." if techs else ""),
             source="sysmon-modular", key=f"sysmon:{folder}/{stem}", min_a=200, licence="MIT")
    for path in G("indicators/hayabusa-rules/hayabusa/**/*.yml"):
        r = yload(path)
        if not isinstance(r, dict) or not r.get("title") or not r.get("detection"):
            continue
        core = {k: r[k] for k in ("title", "description", "level", "logsource", "detection", "falsepositives") if r.get(k)}
        emit("event_log_rule", "logging", "B", pick(["Which Windows event log entries show this: {t}? Write the rule.", "Write a Sigma-style rule over Windows event logs for: {t}."], t=str(r["title"]).strip()),
             "```yaml\n" + yaml.safe_dump(core, sort_keys=False, allow_unicode=True, width=120) + "```", source="hayabusa-rules", key=f"hayabusa:{r.get('id', r['title'])}", min_a=150,
             licence="Detection Rule License 1.1")


def log_schemas() -> None:
    for path in G("logging/elastic-common-schema/schemas/*.yml"):
        d = yload(path)
        fs = d[0] if isinstance(d, list) and d and isinstance(d[0], dict) else None
        fields = [f for f in (fs.get("fields") or []) if isinstance(f, dict) and f.get("name")] if fs else []
        for i in range(0, len(fields), 18):
            part = fields[i:i + 18]
            rows = "\n".join(f"- `{fs['name']}.{f['name']}` ({f.get('type', '')}, {f.get('level', '')}): {first_sentence(f.get('short') or f.get('description'), 260)}"
                             + (f" Example: `{f['example']}`." if f.get("example") not in (None, "") and len(str(f["example"])) < 90 else "") for f in part)
            q = f"Which Elastic Common Schema fields describe {lower1(str(fs.get('title', fs['name'])))}?" if i == 0 else \
                f"In the Elastic Common Schema, what are `{fs['name']}.{part[0]['name']}` through `{fs['name']}.{part[-1]['name']}` for?"
            emit("ecs_fields", "logging", "A", q, (f"{' '.join(str(fs.get('description', '')).split())}\n\n" if i == 0 else "") + rows, source="elastic-common-schema", key=f"ecs:{fs['name']}", licence="Apache-2.0")
    for path in G("logging/ocsf-schema/events/**/*.json"):
        try:
            d = json.loads(read(path))
        except Exception:  # noqa: BLE001
            continue
        attrs = {k: v for k, v in (d.get("attributes") or {}).items() if isinstance(v, dict) and not k.startswith("$")}
        if "@deprecated" in d or not d.get("caption") or not d.get("description") or not attrs:
            continue
        rows = "\n".join(f"- `{k}` ({v.get('requirement', 'optional')})" + (f": {re.sub(r'<[^>]+>', '', str(v['description']))}" if v.get("description") else "") for k, v in list(attrs.items())[:24])
        emit("ocsf_class", "logging", "A", f"What does the OCSF event class \"{d['caption']}\" represent, and which attributes does it define?",
             f"{re.sub(r'<[^>]+>', '', d['description'])}\n\nCategory: {os.path.basename(os.path.dirname(path)).replace('_', ' ')}. It extends `{d.get('extends', 'base_event')}` and adds:\n\n{rows}",
             source="ocsf-schema", key=f"ocsf:{d.get('name')}", min_a=160, licence="Apache-2.0")
    dd = os.path.join(S, "logging/ossem-data-dictionaries")
    files = [p for p in G("logging/ossem-data-dictionaries/**/*.yml") if os.path.relpath(p, dd).split(os.sep)[0] not in ("windows", "osquery")] + \
        [p for name in ("Microsoft-Windows-Sysmon", "Microsoft-Windows-Security-Auditing", "Microsoft-Windows-PowerShell", "Microsoft-Windows-WMI-Activity", "Microsoft-Windows-TaskScheduler")
         for p in G(f"logging/ossem-data-dictionaries/windows/**/{name}/events/*.yml")]
    for path in files:
        d = yload(path)
        fl = [f for f in (d.get("event_fields") or []) if isinstance(f, dict) and f.get("name")] if isinstance(d, dict) else []
        described = [f for f in fl if f.get("description") and str(f["description"]).strip() not in ("TBD", "None")]
        if len(fl) < 4 or len(described) < 0.6 * len(fl):
            continue
        told = lambda f: "" if str(f.get("description") or "").strip() in ("", "TBD", "None") else ": " + " ".join(str(f["description"]).split())  # noqa: E731
        rows = "\n".join(f"- `{f['name']}`" + (f" ({f['type']})" if f.get("type") else "") + told(f) for f in fl[:30])
        src = str(d.get("log_source") or os.path.relpath(path, dd).split(os.sep)[0])
        emit("log_field_dictionary", "logging", "B", f"What fields does the {src} \"{str(d.get('name', '')).strip()}\" log record contain?",
             (f"{' '.join(str(d['description']).split())}\n\n" if d.get("description") else "") + rows, source="OSSEM-DD", key=f"ossem:{src}:{d.get('name')}", licence="MIT")


LOG_PDFS = [("logging/guidance/NIST.SP.800-92r1.ipd.txt", "NIST SP 800-92r1 (draft Cybersecurity Log Management Planning Guide)"), ("logging/guidance/OMB-M-21-31-logging-maturity.txt", "OMB memorandum M-21-31 on federal logging maturity"),
            ("logging/guidance/JPCERT-detecting-lateral-movement-through-event-logs.txt", "JPCERT/CC's report on detecting lateral movement through event logs"),
            ("logging/guidance/best-practices-for-event-logging-and-threat-detection.txt", "the ASD/CISA Best Practices for Event Logging and Threat Detection"),
            ("references/nist/NIST.SP.800-92.txt", "NIST SP 800-92 (Guide to Computer Security Log Management)")]


def logging_pdfs() -> None:
    for rel, label in LOG_PDFS:
        pdf_doc("logging_guidance", "logging", "A", rel.replace("/", os.sep), label, cap=60, licence="government publication")


# ================================================================== LUA
LIBS = "coroutine|debug|io|math|os|package|string|table|utf8|file|bit32"
BASE_FUNCS = "assert|collectgarbage|dofile|error|getmetatable|ipairs|load|loadfile|loadstring|next|pairs|pcall|print|rawequal|rawget|rawlen|rawset|require|select|setmetatable|tonumber|tostring|type|unpack|xpcall|getfenv|setfenv|module"
LUA_ENTRY = re.compile(rf"^((?:(?:{LIBS})[.:]\w+|{BASE_FUNCS}) \([^\n]*\)|luaL?_[A-Za-z_]+)[ \t]*$", re.M)


def lua_reference() -> None:
    for ver in ("5.4", "5.1", "5.3"):
        path = os.path.join(S, "lua", "manual", f"lua-{ver}-manual.txt")
        if not os.path.isfile(path):
            continue
        t = read(path, 4_000_000)
        ms = list(LUA_ENTRY.finditer(t))
        sec = re.compile(r"^\d+(?:\.\d+)* – ", re.M)
        for i, m in enumerate(ms):
            end = ms[i + 1].start() if i + 1 < len(ms) else len(t)
            s = sec.search(t, m.end())
            body = t[m.end():min(end, s.start()) if s else end].strip()
            sig = m.group(1)
            if len(body) < 80:
                continue
            if sig.startswith("lua"):
                emit("lua_c_api", "lua", "A", pick(["Lua {v} C API: what does `{n}` do?", "In the Lua {v} C API, how is `{n}` used?"], v=ver, n=sig), cut(body, 3500),
                     source=f"lua-{ver}-manual", key=f"luafn:{sig}", min_a=100, licence="MIT")
            else:
                name = sig.split(" (")[0]
                emit("lua_function", "lua", "A", pick(["Lua {v}: what does `{n}` do, and what are its arguments?", "How does `{n}` work in Lua {v}?", "Explain `{n}` in Lua {v}."], v=ver, n=name),
                     f"`{sig}`\n\n{cut(body, 3500)}", source=f"lua-{ver}-manual", key=f"luafn:{name}", min_a=100, licence="MIT")
    for path in G("lua/luajit/doc/*.html"):
        sectioned("lua_embedding", "lua", "A", html_to_md(read(path)), "LuaJIT", source="luajit/doc/" + os.path.basename(path), key=f"luajit:{os.path.basename(path)}", cap=30, min_body=250, licence="MIT")

    def ctx(p: str) -> str:
        p = p.replace("\\", "/")
        return ("Lua scripting in Suricata" if "/suricata/" in p else "Envoy's Lua HTTP filter" if "/envoy-docs/" in p else "Lua scripting in Redis" if "/redis-docs/" in p else
                "Lua in HAProxy" if "/haproxy-lua/" in p else "Lua in Wireshark" if "/wireshark" in p else "OpenResty's lua-nginx-module")
    files = G("tool-docs/suricata/doc/userguide/lua/**/*.rst") + G("services/envoy-docs/docs/root/**/*lua*.rst") + G("services/redis-docs/content/develop/**/*lua*.md") + \
        G("services/redis-docs/content/develop/**/eval-intro.md") + G("services/redis-docs/content/develop/**/functions-intro.md") + G("lua/haproxy-lua/doc/**/*.rst") + \
        G("tool-docs/wireshark-guides/doc/wsdg_src/*lua*.adoc") + G("lua/lua-nginx-module/*.markdown")
    docs("lua_embedding", "lua", "A", [f for f in files if not f.endswith(".rst")], ctx, cap_per_file=400, min_body=220, licence="see each project's licence in the manifest")
    rst_docs("lua_embedding", "lua", "A", [f for f in files if f.endswith(".rst")], ctx, cap_per_file=400, min_body=220, licence="see each project's licence in the manifest")
    for path in G("tool-docs/snort3/lua/*.lua"):
        text = read(path)
        parts = re.split(r"^-{20,}\n-- (.+)\n-{20,}\n", text, flags=re.M)
        for i in range(1, len(parts) - 1, 2):
            body = parts[i + 1].strip()
            if 200 < len(body) < 4200 and not parts[i].lower().startswith("snort++ configuration"):
                emit("lua_config", "lua", "A", f"Snort 3 is configured in Lua. Show the `{os.path.basename(path)}` section for \"{re.sub(r'^[0-9]+[.] ', '', parts[i]).strip()}\".",
                     f"```lua\n{body}\n```", source="snort3/lua", key=f"snortlua:{os.path.basename(path)}", min_a=200, licence="GPL-2.0")


def nsedoc(lines: list[str]) -> str:
    out, params, rets, mode = [], [], [], None
    for ln in lines:
        s = re.sub(r"</?(?:b|i|em|strong|ul|ol|li|pre|br ?/?)>", "", re.sub(r"</?code>", "`", re.sub(r"^-{2,} ?", "", ln).rstrip()))
        if s.strip() == "`":                      # a <code> tag alone on its line opens or closes a multi-line example
            s = "```"
        m = re.match(r"@(\w+)\s*(.*)", s.strip())
        if m:
            tag, rest = m.group(1), m.group(2)
            if tag == "param":
                pm = rest.split(None, 1)
                if pm:
                    params.append(f"- `{pm[0]}`: {pm[1] if len(pm) > 1 else ''}")
                mode = "param"
            elif tag in ("return", "returns"):
                rets.append(rest)
                mode = "ret"
            elif tag == "usage":
                out += ["", "Usage:", rest] if rest else ["", "Usage:"]
                mode = None
            else:
                mode = "skip"
        elif mode == "param" and s.strip() and params:
            params[-1] += " " + s.strip()
        elif mode == "ret" and s.strip() and rets:
            rets[-1] += " " + s.strip()
        elif mode == "skip":
            mode = None if not s.strip() else "skip"
        else:
            mode = None
            out.append(s)
    text = "\n".join(out).strip()
    if params:
        text += "\n\n**Parameters:**\n" + "\n".join(params)
    if any(rets):
        text += "\n\n**Returns:** " + "; ".join(r for r in rets if r)
    return text


SAFE_NSE = {"safe", "discovery", "default", "version"}


def nse() -> None:
    lic = "Nmap Public Source License"
    for path in G("tool-docs/nmap/nselib/*.lua"):
        lib, text = os.path.basename(path)[:-4], read(path, 3_000_000)
        hdr = []
        for ln in text.split("\n"):
            if ln.startswith("--"):
                hdr.append(ln)
            elif hdr:
                break
        desc = nsedoc(hdr)
        emit("nse_library", "lua", "A", pick(["What does the Nmap NSE library `{l}` provide, and how do I load it in a script?", "I'm writing an NSE script. What is the `{l}` library for?"], l=lib),
             f"{desc}\n\nLoad it at the top of a script with `local {lib} = require \"{lib}\"`.", source=f"nmap/nselib/{lib}.lua", key=f"nselib:{lib}", min_a=220, licence=lic)
        n = 0
        for m in re.finditer(r"((?:^---.*\n)(?:^--.*\n)*)^(?:function ([\w.:]+) *\(([^)]*)\)|([\w.]+) *= *function *\(([^)]*)\))", text, re.M):
            fname, args = (m.group(2) or m.group(4)), (m.group(3) if m.group(2) else m.group(5)) or ""
            if fname.startswith("_") or "._" in fname or n >= 14:
                continue
            qual = fname if ("." in fname or ":" in fname) else f"{lib}.{fname}"
            body = nsedoc(m.group(1).rstrip("\n").split("\n"))
            emit("nse_function", "lua", "A", pick(["In an Nmap NSE script, how do I use `{q}`?", "NSE: what does `{q}` do and what does it return?"], q=qual),
                 f"`{qual}({' '.join(args.split())})` from the `{lib}` library.\n\n{body}", source=f"nmap/nselib/{lib}.lua", key=f"nselib:{lib}", min_a=160, licence=lic)
            n += 1
    for path in G("tool-docs/nmap/scripts/*.nse"):                                   # full sources only for discovery-type scripts
        text = read(path).strip()
        cats = set(re.findall(r'"(\w+)"', (re.search(r"categories\s*=\s*\{([^}]*)\}", text) or [None, ""])[1]))
        d = re.search(r"description\s*=\s*\[\[(.*?)\]\]", text, re.S)
        if not d or not cats or not cats <= SAFE_NSE or not 600 < len(text) < 5600:
            continue
        emit("nse_script_source", "lua", "A", pick(["Write an Nmap NSE script in Lua that does this: {d}", "Show me a complete NSE script (Lua) for the following: {d}"], d=first_sentence(d.group(1), 320)),
             f"```lua\n{text}\n```", source="nmap/scripts/" + os.path.basename(path), key=f"nse-src:{os.path.basename(path)}", min_a=400, licence=lic)
    p = os.path.join(S, "tool-docs/nmap/docs/scripting.xml")
    if os.path.isfile(p):
        docbook("nse_guide", "lua", "A", p, "the Nmap Scripting Engine", cap=200, min_body=300, licence=lic)


# ================================================================== TOOLS
def tool_cards() -> None:
    for path in G("man-pages/tldr/pages/**/*.md"):
        platform = os.path.basename(os.path.dirname(path))
        if platform not in ("common", "linux", "windows", "osx"):
            continue
        text = read(path)
        name = (re.search(r"^# (.+)$", text, re.M) or [None, os.path.basename(path)[:-3]])[1].strip()
        desc = re.sub(r"`", "", " ".join(d.strip() for d in re.findall(r"^> (.+)$", text, re.M) if not d.startswith(("More information", "See also"))))
        ex = [(re.sub(r"\[(\w)\]", r"\1", d).strip().rstrip(":"), re.sub(r"\{\{(.+?)\}\}", r"<\1>", re.sub(r"\{\{\[(-[^|\]]+)\|[^\]]*\]\}\}", r"\1", c)))
              for d, c in re.findall(r"^- (.+?):?\n\n`(.+?)`$", text, re.M)]
        if len(ex) < 3 or not desc:
            continue
        plat = "" if platform == "common" else f" ({'macOS' if platform == 'osx' else platform.capitalize()})"
        emit("tool_tasks", "tools", "B", pick(["What is `{n}`{p} used for? Show the invocations I'm most likely to need.", "Give me a quick reference for `{n}`{p}.", "`{n}`{p}: what does it do and how do I run the common tasks?"], n=name, p=plat),
             f"{desc}\n\n" + "\n".join(f"- {d}:\n  `{c}`" for d, c in ex) + "\n\nValues in `<angle brackets>` are placeholders.", source="tldr-pages", key=f"tool:{name}", min_a=200, licence="CC BY 4.0")
        for d, c in rng.sample(ex, min(2, len(ex))):
            emit("command_explain", "tools", "B", pick(["What does this command do?\n\n`{c}`", "Explain this command:\n\n`{c}`"], c=c),
                 f"{d}.\n\n`{name}` is: {lower1(desc)}", source="tldr-pages", key=f"tool:{name}", min_a=50, licence="CC BY 4.0")
    done: set[str] = set()
    extra = ("FILES", "ENVIRONMENT", "EXIT STATUS", "SECURITY", "SECURITY CONSIDERATIONS", "NOTES", "CAVEATS", "DIAGNOSTICS", "CONFIGURATION", "SIGNALS")
    for folder in ("kali/man-pages/text", "man-pages/debian/text", "man-pages/linux-man-pages/text"):
        for path in G(folder + "/*.txt"):
            base = os.path.basename(path)[:-4]
            name, _, sec = base.rpartition(".")
            if base in done or not name:
                continue
            done.add(base)
            p = bc.man_page(read(path))
            ex = p.get("EXAMPLES") or p.get("EXAMPLE") or ""
            if len(ex) > 200 and sec[:1] in ("1", "5", "8") and max(len(ln) for ln in ex.split("\n")) < 400:   # skip pages whose code lost its line breaks
                emit("man_examples", "tools", "A", pick(["Show me worked examples of using `{n}`.", "Give me practical `{n}` examples from its manual.", "How is `{n}` used in practice? Examples, please."], n=name),
                     cut(ex, 4500), source=f"man:{base}", key=f"tool:{name}", licence="see the page's source package")
            n = 0
            for sname in extra:
                body = p.get(sname, "")
                if len(body) > 250 and n < 2:
                    emit("man_section", "kernel-memory" if sec[:1] in ("2", "4") else "tools", "A", f"`{name}({sec})`: what does its manual say under {sname.title()}?", cut(body, 3500),
                         source=f"man:{base}", key=f"tool:{name}", licence="see the page's source package")
                    n += 1
    p = os.path.join(S, "tool-docs/nmap/docs/refguide.xml")
    if os.path.isfile(p):
        raw = read(p, 4_000_000)
        for term, body in varlist(raw):
            if len(body) > 200 and term.startswith("-"):
                emit("nmap_option", "tools", "A", pick(["Nmap: what does `{t}` do, and when would I use it?", "Explain the Nmap option `{t}`."], t=term), cut(body, 4500),
                     source="nmap/docs/refguide.xml", key=f"nmapopt:{term[:24]}", licence="Nmap Public Source License")
        docbook("nmap_reference", "tools", "A", p, "Nmap's reference guide", cap=60, min_body=400, licence="Nmap Public Source License")


# ================================================================== PRIORITIZATION
TABLES = {"deployer_patch_application_priority_1_0_0": ("the SSVC Deployer decision table (patch application priority)", "ssvc"),
          "supplier_patch_development_priority_1_0_0": ("the SSVC Supplier decision table (patch development priority)", "ssvc"),
          "coordinator_triage_1_0_0": ("the SSVC Coordinator triage decision table", "ssvc"), "coordinator_publish_decision_table_1_0_0": ("the SSVC Coordinator publication decision table", "ssvc"),
          "human_impact_1_0_0": ("the SSVC Human Impact combination table", "ssvc"), "utility_1_0_0": ("the SSVC Utility combination table", "ssvc"),
          "public_safety_impact_1_0_0": ("the SSVC Public Safety Impact combination table", "ssvc"), "cisa_coordinator_2_0_3": ("CISA's SSVC decision tree", "cisa"),
          "cisa_bod_26_04_1_0_0": ("the CISA BOD 26-04 remediation timelines table", "cisa")}


def ssvc_logic() -> None:
    lic = "CERT/CC SSVC (see the repository's licence)"
    dps: dict[tuple[str, str], dict] = {}
    latest: dict[str, dict] = {}
    for path in G("prioritization/certcc-ssvc/data/json/decision_points/**/*.json"):
        try:
            d = json.loads(read(path))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(d, dict) or not d.get("name") or not d.get("values"):
            continue
        dps[(d["name"].lower(), d.get("version", ""))] = d
        ns = os.path.basename(os.path.dirname(path))
        if ns in ("ssvc", "cisa") and (d["name"].lower() not in latest or tuple(map(int, d["version"].split("."))) > tuple(map(int, latest[d["name"].lower()]["version"].split(".")))):
            latest[d["name"].lower()] = d
    for d in latest.values():
        emit("ssvc_decision_point", "prioritization", "A", pick(["In SSVC, what does the decision point \"{n}\" ask, and what values can it take?", "Define the SSVC decision point \"{n}\" and its values."], n=d["name"]),
             f"**{d['name']}** (version {d['version']}): {d.get('definition', '')}\n\n" + "\n".join(f"- **{v['name']}**: {v.get('definition', '')}" for v in d["values"]),
             source="certcc-ssvc decision points", key=f"ssvcdp:{d['name']}", min_a=120, licence=lic)

    def define(col: str, value: str) -> tuple[str, str]:
        m = re.match(r"^(.*?) v(\d+\.\d+\.\d+)(?: \(\w+\))?$", col.strip())
        name, ver = (m.group(1), m.group(2)) if m else (col.strip(), "")
        d = dps.get((name.lower(), ver)) or next((x for (n, _), x in dps.items() if n == name.lower()), None)
        v = next((x for x in (d or {}).get("values", []) if x["name"].lower() == value.lower()), None)
        return name, (v.get("definition", "") if v else "")

    for stem, (label, sub) in TABLES.items():
        path = os.path.join(S, "prioritization/certcc-ssvc/data/csv", sub, stem + ".csv")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8", newline="") as fh:
            rows = list(csv.reader(fh))
        cols, data = rows[0][1:], [r[1:] for r in rows[1:] if len(r) == len(rows[0])]
        table = {tuple(r[:-1]): r[-1] for r in data}
        domain = [sorted({r[i] for r in data}, key=[r[i] for r in data].index) for i in range(len(cols) - 1)]
        for r in data:
            ins, outcome = r[:-1], r[-1]
            names = [define(c, v) for c, v in zip(cols, ins)]
            oname, odef = define(cols[-1], outcome)
            flips = []
            for i, (nm, _) in enumerate(names):
                for other in domain[i]:
                    alt = table.get(tuple(ins[:i] + [other] + ins[i + 1:]))
                    if other != ins[i] and alt and alt != outcome:
                        flips.append(f"- If {nm} were **{other}**: {alt}.")
            emit("ssvc_table", "prioritization", "A",
                 pick(["Apply {l}.\n\n{i}\n\nWhat is the outcome, and which single input would change it?", "Using {l}, work out the decision for these inputs and show the sensitivity to each one.\n\n{i}"],
                      l=label, i="\n".join(f"- {nm}: {v}" for (nm, _), v in zip(names, ins))),
                 f"**Outcome: {outcome}**" + (f". {odef}" if odef and odef.lower().strip(". ") != outcome.lower() else "") + "\n\n**How each input reads:**\n\n" + "\n".join(f"- **{nm}: {v}**." + (f" {df}" if df else "") for (nm, df), v in zip(names, ins))
                 + "\n\n**Sensitivity** (one input changed, the others held):\n\n" + ("\n".join(flips[:10]) if flips else "- No single input change alters this outcome."),
                 source=f"certcc-ssvc {stem}.csv", key=f"ssvctable:{stem}:{'|'.join(ins)}", joined=True, licence=lic)
    for rel, label in (("prioritization/cisa-ssvc/cisa-ssvc-guide.txt", "CISA's SSVC guide"), ("prioritization/nist/NIST.CSWP.41-likely-exploited-vulnerabilities.txt", "NIST CSWP 41 (the Likely Exploited Vulnerabilities metric)"),
                       ("references/nist/NIST.SP.800-40r4.txt", "NIST SP 800-40r4 (Enterprise Patch Management Planning)"), ("references/nist/NIST.SP.800-30r1.txt", "NIST SP 800-30r1 (Guide for Conducting Risk Assessments)")):
        pdf_doc("prioritization_guidance", "prioritization", "A", rel.replace("/", os.sep), label, cap=45)


IMPACT = {"H": ("High", "total loss, or loss of the most critical data or function"), "L": ("Low", "partial or limited loss"), "N": ("None", "no loss")}
CVSS31 = {"AV": ("Attack Vector", {"N": ("Network", "reachable remotely across the network, including from the internet"), "A": ("Adjacent", "the attacker must be on the same physical or logical network segment, such as the local subnet or radio range"),
                                    "L": ("Local", "the attacker needs local access: a session on the machine, or a user who opens a crafted file"), "P": ("Physical", "the attacker must physically touch the device")}),
          "AC": ("Attack Complexity", {"L": ("Low", "no special conditions; the attack works repeatably"), "H": ("High", "success depends on conditions outside the attacker's control")}),
          "PR": ("Privileges Required", {"N": ("None", "no account or authentication needed"), "L": ("Low", "an ordinary user account is needed"), "H": ("High", "administrative privileges are needed")}),
          "UI": ("User Interaction", {"N": ("None", "no action by any user is needed"), "R": ("Required", "someone other than the attacker must act, such as opening a file or following a link")}),
          "S": ("Scope", {"U": ("Unchanged", "impact stays within the vulnerable component's own security authority"), "C": ("Changed", "impact reaches resources beyond the vulnerable component's security authority")}),
          "C": ("Confidentiality", IMPACT), "I": ("Integrity", IMPACT), "A": ("Availability", IMPACT)}
CVSS40 = {"AV": CVSS31["AV"], "AC": CVSS31["AC"], "AT": ("Attack Requirements", {"N": ("None", "no particular deployment or execution conditions are needed"), "P": ("Present", "the attack depends on specific conditions in the target, such as a race window or a non-default configuration")}),
          "PR": CVSS31["PR"], "UI": ("User Interaction", {"N": ("None", "no action by any user is needed"), "P": ("Passive", "limited, involuntary interaction is enough, such as viewing a page"), "A": ("Active", "a user must take specific, deliberate actions")}),
          "VC": ("Confidentiality, vulnerable system", IMPACT), "VI": ("Integrity, vulnerable system", IMPACT), "VA": ("Availability, vulnerable system", IMPACT),
          "SC": ("Confidentiality, subsequent systems", IMPACT), "SI": ("Integrity, subsequent systems", IMPACT), "SA": ("Availability, subsequent systems", IMPACT)}


def reading(m: dict[str, str]) -> str:
    av, pr, ui = m.get("AV"), m.get("PR"), m.get("UI")
    if av == "N" and pr == "N" and ui == "N":
        return "Reachable over the network with no credentials and no user action. That is the profile that gets scanned for at scale, so whether the affected service is exposed matters more than anything else here."
    if av == "N" and pr == "N":
        return "Remote and unauthenticated, but it needs a person to act, so delivery depends on getting a user to open or visit something. Exposure of the service alone is not enough."
    if av == "N":
        return f"Remote, but it needs {'an ordinary account' if pr == 'L' else 'administrative access'} first, so weak or stolen credentials are the realistic path. Review who holds that level of access."
    if av == "A":
        return "Limited to attackers on the same network segment, so segmentation and who can join that segment set the real exposure."
    if av == "L":
        return "Local: the attacker already needs a foothold or a user to open something, so this is typically a step taken after initial access. Prioritize it on shared and multi-user systems."
    return "Physical access is required, which limits it to lost, stolen or publicly accessible devices."


def cvss_breakdowns() -> None:
    per31, per40, n31, n40 = collections.Counter(), collections.Counter(), 0, 0
    for path in G("cve/nvd/*.jsonl"):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if n31 >= 1100 and n40 >= 500:
                    break
                if rng.random() > 0.35:
                    continue
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                desc = next((d["value"] for d in r.get("descriptions", []) if d.get("lang") == "en"), "")
                cwes = sorted({d["value"] for w in r.get("weaknesses", []) for d in w.get("description", []) if d.get("value", "").startswith("CWE-")})
                if len(desc) < 60:
                    continue
                for key, table, ver in (("cvssMetricV31", CVSS31, "3.1"), ("cvssMetricV40", CVSS40, "4.0")):
                    ms = r.get("metrics", {}).get(key) or []
                    if not ms:
                        continue
                    mt = next((x for x in ms if x.get("type") == "Primary"), ms[0])
                    cd = mt.get("cvssData", {})
                    vec = cd.get("vectorString", "")
                    parts = dict(p.split(":", 1) for p in vec.split("/")[1:] if ":" in p)
                    base = "/".join(f"{k}:{parts[k]}" for k in table if k in parts)
                    counter = per31 if ver == "3.1" else per40
                    if not all(k in parts and parts[k] in table[k][1] for k in table) or counter[base] >= 2 or (ver == "3.1" and n31 >= 1100) or (ver == "4.0" and n40 >= 500):
                        continue
                    counter[base] += 1
                    rows = "\n".join(f"| {table[k][0]} | {k}:{parts[k]} ({table[k][1][parts[k]][0]}) | {table[k][1][parts[k]][1].capitalize()} |" for k in table)
                    sub = f"; exploitability subscore {mt['exploitabilityScore']} of 3.9, impact subscore {mt['impactScore']} of 6.0" if ver == "3.1" and "exploitabilityScore" in mt else ""
                    emit("cvss_breakdown", "prioritization", "A",
                         pick(["Break down the CVSS score for {c}.\n\nDescription: {d}\nVector: `{v}`\n\nWhat does each metric mean, and what does that tell me about how it would be reached?",
                               "Read this CVSS {ver} vector for me and say what it implies for exposure.\n\n{c}: {d}\n`{v}`"], c=r["id"], d=cut(desc, 420), v=f"CVSS:{ver}/{base}", ver=ver),
                         f"**Base score {cd.get('baseScore')} ({str(cd.get('baseSeverity', '')).title()})**{sub}." + (f" Weakness: {', '.join(cwes[:3])}." if cwes else "")
                         + f"\n\n| Metric | Value | Meaning |\n|---|---|---|\n{rows}\n\n**What that implies:** {reading(parts)}\n\n"
                         "The score describes severity if the flaw is used, not how likely that is. Pair it with exploitation evidence (CISA's KEV catalogue, EPSS, an SSVC decision) before setting the order of work.",
                         source="nvd + cvss specification", key=f"cvss:{r['id']}:{ver}", joined=True)
                    if ver == "3.1":
                        n31 += 1
                    else:
                        n40 += 1


_VR: dict = {}          # CISA's per-CVE assessments, EPSS and KEV, loaded once by triage_sets()
RANK = {"act": 3, "attend": 2, "track*": 1, "track": 0}
ASSETS = {"high": ["an internet-facing customer portal", "a domain controller", "the payment processing service", "a hospital clinical system", "the production database cluster", "the identity provider"],
          "medium": ["an internal file server", "a CI build server", "the HR application", "a departmental web app", "an internal wiki server"],
          "low": ["a test VM", "a lab workstation", "a lobby kiosk", "a developer sandbox", "a decommissioning candidate"]}


def triage_sets() -> None:
    tree = {}
    with open(os.path.join(S, "prioritization/certcc-ssvc/data/csv/cisa/cisa_coordinator_2_0_3.csv"), encoding="utf-8") as fh:
        for row in list(csv.reader(fh))[1:]:
            tree[(row[1].replace("public poc", "poc"), row[2], row[3], row[4])] = row[5]
    epss = {}
    with gzip.open(os.path.join(S, "cve/epss/epss_scores-current.csv.gz"), "rt", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if row and row[0].startswith("CVE-"):
                epss[row[0]] = float(row[1])
    kev = {v["cveID"] for v in json.loads(read(os.path.join(S, "cve/kev/known_exploited_vulnerabilities.json"), 30_000_000))["vulnerabilities"]}
    files = G("prioritization/cisa-vulnrichment-2026/2026/**/*.json")
    pool = collections.defaultdict(list)
    for path in rng.sample(files, min(16000, len(files))):
        try:
            d = json.loads(read(path))
            cve = d["cveMetadata"]["cveId"]
            ssvc = next(m["other"]["content"] for a in d["containers"].get("adp", []) for m in a.get("metrics", []) if m.get("other", {}).get("type") == "ssvc")
            o = {k: str(v).lower() for x in ssvc["options"] for k, v in x.items()}
            desc = next(x["value"] for x in d["containers"]["cna"]["descriptions"] if x.get("lang", "en").startswith("en"))
        except Exception:  # noqa: BLE001
            continue
        e, au, ti = o.get("Exploitation"), o.get("Automatable"), o.get("Technical Impact")
        if e in ("none", "poc", "active") and au in ("yes", "no") and ti in ("partial", "total") and len(desc) > 60:
            pool[e].append((cve, cut(" ".join(desc.split()), 260), e, au, ti))
    everything = [x for v in pool.values() for x in v]
    _VR.update(epss=epss, kev=kev, everything=everything)       # reused by the scenario and deadline builders below
    if len(everything) < 50:
        return
    used, made = collections.Counter(), 0
    for _ in range(15000):
        if made >= 1800:
            break
        # exploited vulnerabilities are a small pool: draw them less often and cap reuse, so the sets do not all lead with the same few CVEs
        picks = [rng.choice(pool[e] or everything) for e in rng.choices(["active", "poc", "none"], weights=[2, 4, 5], k=4)]
        if len({p[0] for p in picks}) < 4 or any(used[p[0]] >= 4 for p in picks):
            continue
        items = []
        for cve, desc, e, au, ti in picks:
            mi = rng.choice(["low", "medium", "medium", "high", "high"])
            in_kev = cve in kev
            eff = "active" if in_kev else e
            items.append(dict(cve=cve, desc=desc, e=e, eff=eff, au=au, ti=ti, mi=mi, asset=rng.choice(ASSETS[mi]), kev=in_kev, epss=epss.get(cve), dec=tree[(eff, au, ti, mi)]))
        if len({x["dec"] for x in items}) < 2:            # nothing to rank when every item gets the same decision
            continue
        made += 1
        used.update(x["cve"] for x in items)
        order = sorted(items, key=lambda x: (-RANK[x["dec"]], not x["kev"], -(x["epss"] or 0)))
        q = pick(["I have four vulnerabilities to schedule. Rank them for remediation using CISA's SSVC tree and explain the order.\n\n{b}",
                  "Put these four findings in remediation order with SSVC, and show your working.\n\n{b}"],
                 b="\n".join(f"{i}. {x['cve']} on {x['asset']} (mission and well-being impact: {x['mi']}). {x['desc']} Exploitation: {x['e']}; automatable: {x['au']}; technical impact: {x['ti']}; "
                             f"EPSS: {('%.3f' % x['epss']) if x['epss'] is not None else 'no score'}; in KEV: {'yes' if x['kev'] else 'no'}." for i, x in enumerate(items, 1)))
        rows = "\n".join(f"| {i} | {x['cve']} | {x['eff']} | {x['au']} | {x['ti']} | {x['mi']} | **{x['dec'].capitalize()}** | {'yes' if x['kev'] else 'no'} | {('%.3f' % x['epss']) if x['epss'] is not None else 'n/a'} |"
                         for i, x in enumerate(order, 1))
        notes = ["Ranked by SSVC decision: Act first, then Attend, Track*, Track. Within the same decision, confirmed exploitation (KEV) comes first, then the higher EPSS."]
        if any(x["kev"] and x["e"] != "active" for x in items):
            notes.append("A KEV listing is confirmed exploitation, so those entries are evaluated with Exploitation = active even though the supplied value was lower: "
                         + ", ".join(x["cve"] for x in items if x["kev"] and x["e"] != "active") + ".")
        top, last = order[0], order[-1]
        notes.append(f"{top['cve']} leads: exploitation {top['eff']}, automatable {top['au']}, technical impact {top['ti']} on {top['asset']} gives {top['dec'].capitalize()}.")
        if RANK[last["dec"]] < RANK[top["dec"]]:
            notes.append(f"{last['cve']} goes last: with exploitation {last['eff']} and {last['mi']} mission impact the tree returns {last['dec'].capitalize()}, so it can follow the normal patch cycle.")
        ties = [d for d, n in collections.Counter(x["dec"] for x in items).items() if n > 1]
        if ties:
            notes.append("Equal decisions (" + ", ".join(t.capitalize() for t in ties) + ") were separated by KEV membership and then EPSS, which estimate likelihood; CVSS was not used to order them.")
        emit("triage_ranking", "prioritization", "A", q, "| Order | CVE | Exploitation | Automatable | Technical impact | Mission impact | SSVC decision | KEV | EPSS |\n|---|---|---|---|---|---|---|---|---|\n"
             + rows + "\n\n**Why this order:**\n\n" + "\n".join(f"- {n}" for n in notes), source="cisa-vulnrichment + epss + kev + cisa ssvc tree", key="triage:" + "+".join(sorted(x["cve"] for x in items)), joined=True)


EXPOSURE = {"small": ["a local service on a tightly controlled internal segment", "on an isolated management network with no route from user subnets"],
            "controlled": ["reachable from internal networks only, behind access controls", "internal, reachable from the corporate network through a VPN"],
            "open": ["reachable directly from the internet", "published to the internet through a load balancer"]}


def csv_table(sub: str, stem: str) -> tuple[list[str], list[list[str]]]:
    with open(os.path.join(S, "prioritization/certcc-ssvc/data/csv", sub, stem + ".csv"), encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    return rows[0][1:], [r[1:] for r in rows[1:] if len(r) == len(rows[0])]


def uniq(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def ssvc_scenarios() -> None:
    """Real CVEs placed in a described environment and decided with the SSVC Deployer tree. Human impact is computed, not given."""
    if not _VR.get("everything"):
        return
    dp = {}
    for path in G("prioritization/certcc-ssvc/data/json/decision_points/ssvc/*.json"):
        try:
            d = json.loads(read(path))
        except Exception:  # noqa: BLE001
            continue
        for v in d.get("values", []):
            text = str(v.get("definition", "")).strip()
            dp[(d["name"].lower(), v["name"].lower())] = text if not text or text.endswith((".", "!", "?")) else text + "."
    _, rows = csv_table("ssvc", "deployer_patch_application_priority_1_0_0")
    deploy = {tuple(r[:-1]): r[-1] for r in rows}            # (exploitation, system exposure, automatable, human impact) -> decision
    _, hrows = csv_table("ssvc", "human_impact_1_0_0")
    human = {(r[0], r[1]): r[2] for r in hrows}               # (safety impact, mission impact) -> human impact
    exposures, exploits = uniq([r[1] for r in rows]), uniq([r[0] for r in rows])
    safeties, missions = uniq([r[0] for r in hrows]), uniq([r[1] for r in hrows])
    outdef = lambda o: next((v for (n, val), v in dp.items() if val == o.lower() and "defer" in n), "")  # noqa: E731
    for _ in range(2600):
        cve, desc, e, au, _ti = rng.choice(_VR["everything"])
        in_kev = cve in _VR["kev"]
        ex = "active" if in_kev else "public poc" if e == "poc" else e
        x, s, m = rng.choice(exposures), rng.choice(safeties), rng.choice(missions)
        hi = human.get((s, m))
        out = deploy.get((ex, x, au, hi))
        if not hi or not out:
            continue
        where = rng.choice(EXPOSURE.get(x, [f"{x} exposure"]))
        s_txt, m_txt = s.replace("mef", "MEF"), m.replace("mef", "MEF")       # mission essential function: SSVC's own abbreviation
        flips = [f"- If the system's exposure were **{alt}**: {deploy[(ex, alt, au, hi)]}." for alt in exposures if alt != x and deploy.get((ex, alt, au, hi)) not in (None, out)] + \
                [f"- If exploitation became **{alt}**: {deploy[(alt, x, au, hi)]}." for alt in exploits if alt != ex and deploy.get((alt, x, au, hi)) not in (None, out)]
        od = outdef(out)
        emit("ssvc_scenario", "prioritization", "A",
             pick(["How urgently should we patch {c}? Use the SSVC Deployer tree.\n\n{d}\n\nExploitation status: {e}. Automatable: {a}.\nOur environment: the affected system is {w}. If it were compromised, the safety impact would be {s} and the mission impact would be {m}.",
                   "Work out the patch priority for {c} in our environment with SSVC.\n\n{d}\n\nKnown about the vulnerability: exploitation {e}; automatable {a}.\nKnown about us: the system is {w}; safety impact {s}; mission impact {m}."],
                  c=cve, d=desc, e=e, a=au, w=where, s=s_txt, m=m_txt),
             f"**Decision: {out}**" + (f". {od}" if od and od.lower().strip(". ") != out.lower() else "") + "\n\nWorking through the Deployer tree:\n\n"
             f"1. **Exploitation: {ex}.** {dp.get(('exploitation', ex), '')}" + (" It is listed in CISA's KEV catalogue, which is confirmed exploitation, so the supplied status is raised to active." if in_kev and e != "active" else "")
             + f"\n2. **System exposure: {x}.** {dp.get(('system exposure', x), '')} Here the system is {where}.\n3. **Automatable: {au}.** {dp.get(('automatable', au), '')}\n"
             f"4. **Human impact: {hi}.** This input is derived, not chosen: safety impact {s_txt} combined with mission impact {m_txt} gives {hi} in the SSVC human impact table.\n\n"
             "**What would change the decision** (one input changed, the others held):\n\n" + ("\n".join(flips) if flips else "- Neither a change in exposure nor in exploitation status alters it."),
             source="cisa-vulnrichment + kev + certcc-ssvc deployer and human-impact tables", key=f"scenario:{cve}", joined=True, licence="CERT/CC SSVC; CISA Vulnrichment (CC0)")


def remediation_deadlines() -> None:
    if not _VR.get("everything"):
        return
    _, rows = csv_table("cisa", "cisa_bod_26_04_1_0_0")
    table = {tuple(r[:-1]): r[-1] for r in rows}             # (in KEV, publicly exposed, automatable, technical impact) -> timeline
    for cve, desc, _e, au, ti in rng.sample(_VR["everything"], min(1500, len(_VR["everything"]))):
        kev, exposed = ("yes" if cve in _VR["kev"] else "no"), rng.choice(["yes", "no"])
        out = table.get((kev, exposed, au, ti))
        if not out:
            continue
        flip = lambda k, x: table.get((k, x, au, ti))  # noqa: E731
        notes = []
        other_x = "no" if exposed == "yes" else "yes"
        if flip(kev, other_x) != out:
            notes.append(f"- If it were{' not' if exposed == 'yes' else ''} publicly exposed: {flip(kev, other_x)}.")
        other_k = "no" if kev == "yes" else "yes"
        if flip(other_k, exposed) != out:
            notes.append(f"- If it {'were added to' if kev == 'no' else 'were not in'} the KEV catalogue: {flip(other_k, exposed)}.")
        emit("remediation_deadline", "prioritization", "A",
             pick(["What remediation deadline applies to {c} under the CISA BOD 26-04 timelines?\n\n{d}\n\nIn KEV: {k}. Publicly exposed in our environment: {x}. Automatable: {a}. Technical impact: {t}.",
                   "{c}: how long do we have to remediate under BOD 26-04?\n\n{d}\n\nKEV listed: {k}; publicly exposed: {x}; automatable: {a}; technical impact: {t}."], c=cve, d=desc, k=kev, x=exposed, a=au, t=ti),
             f"**{out[0].upper() + out[1:]}.**\n\nThe BOD 26-04 remediation timeline table, as encoded in CERT/CC's SSVC data, takes four inputs:\n\n- In KEV: {kev}\n- Publicly exposed: {exposed}\n- Automatable: {au}\n- Technical impact: {ti}\n\n"
             f"That combination maps to: {out}.\n\n**What would change it:**\n\n" + ("\n".join(notes) if notes else "- Neither exposure nor KEV status changes the timeline for this combination."),
             source="cisa-vulnrichment + kev + certcc-ssvc cisa_bod_26_04 table", key=f"deadline:{cve}", joined=True, licence="CERT/CC SSVC; CISA Vulnrichment (CC0)")


def core_tool_options() -> None:
    """Every documented option for the curated security and administration tools (round one sampled 14 per page)."""
    for path in G("man-pages/debian/text/*.txt"):
        base = os.path.basename(path)[:-4]
        name, _, sec = base.rpartition(".")
        if not name or sec[:1] not in ("1", "8"):
            continue
        p = bc.man_page(read(path))
        lines = (p.get("OPTIONS") or p.get("DESCRIPTION", "")).split("\n")
        i = n = 0
        while i < len(lines) and n < 80:
            m = bc.OPT_RE.match(lines[i])
            if not m:
                i += 1
                continue
            j, buf = i + 1, []
            while j < len(lines) and not bc.OPT_RE.match(lines[j]) and not bc.MAN_SEC.match(lines[j]) and len(buf) < 40:
                buf.append(lines[j].strip())
                j += 1
            text = " ".join(" ".join(buf).split())
            if 80 <= len(text) <= 1500:
                emit("man_option", "tools", "A", pick(["What does the `{o}` option of `{n}` do?", "In `{n}`, what is `{o}` for?", "`{n} {o}`: what does that do?"], o=m.group(1).strip(), n=name), text,
                     source=f"man:{base}", key=f"toolopt:{name}", min_a=80, licence="see the page's source package")
                n += 1
            i = j


def wiki_clean(t: str) -> str:
    t = re.sub(r"\{\{[^{}]*\}\}", "", re.sub(r"\{\{[^{}]*\}\}", "", t))
    t = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", t)
    t = re.sub(r"\[https?://\S+ ([^\]]+)\]", r"\1", t)
    t = re.sub(r"<br ?/?>", "\n", t)
    t = re.sub(r"<(math|big|small|code|tt|sub|sup|u|b|i)>|</(math|big|small|code|tt|sub|sup|u|b|i)>", "", t)
    t = re.sub(r"'{2,}", "", t)
    return clean("\n".join(re.sub(r"^[;:*#]+\s*", "- " if ln[:1] in "*#" else "", ln) for ln in html.unescape(t).split("\n")))


def rosetta_lua() -> None:
    base = os.path.join(S, "lua", "rosetta-code", "Task")
    if not os.path.isdir(base):
        return
    for task in sorted(os.listdir(base)):
        ldir = os.path.join(base, task, "Lua")
        if not os.path.isdir(ldir):
            continue
        code = next((read(os.path.join(ldir, f)).strip() for f in sorted(os.listdir(ldir)) if 80 < os.path.getsize(os.path.join(ldir, f)) < 4500), "")
        dpath = os.path.join(base, task, "00-TASK.txt")
        desc = cut(wiki_clean(read(dpath)), 700) if os.path.isfile(dpath) else ""
        if not code:
            continue
        emit("lua_program", "lua", "B", pick(["Write a Lua program for this task: {t}.{d}", "Solve this in Lua: {t}.{d}"], t=task.replace("-", " "), d=("\n\n" + desc) if len(desc) > 40 else ""),
             f"```lua\n{code}\n```", source="rosetta code", key=f"rosetta:{task}", system=False, min_a=100, licence="GNU Free Documentation License (Rosetta Code)")


def zap_help() -> None:
    for path in G("tool-docs/zap-core-help/addOns/help/src/main/javahelp/contents/**/*.html"):
        if "releases" in path.replace("\\", "/").split("/"):          # release notes date quickly and are not usage help
            continue
        raw = read(path)
        title = re.search(r"<title>(.*?)</title>", raw, re.S | re.I)
        sectioned("zap_help", "web-tools", "A", html_to_md(re.sub(r"<title>.*?</title>", "", raw, flags=re.S | re.I)), "OWASP ZAP", source="zap-core-help/" + os.path.basename(path),
                  key=f"zap:{os.path.basename(path)}", cap=6, min_body=250, licence="Apache-2.0", title=" ".join(html.unescape(title.group(1)).split()) if title else "")


STEPS_A = [("wazuh log sources (parse + alert)", wazuh_logs), ("log parsing (grok, logstash, loki, asim)", log_parsing), ("log and artifact locations", log_locations),
           ("windows event fields, sysmon, event rules", windows_logging), ("log schemas and field dictionaries", log_schemas), ("logging guidance (pdf)", logging_pdfs),
           ("lua reference and embedded lua", lua_reference), ("nmap scripting engine (lua)", nse), ("tool task cards, examples, nmap options", tool_cards),
           ("ssvc decision points and tables", ssvc_logic), ("cvss breakdowns", cvss_breakdowns), ("multi-item triage sets", triage_sets),
           ("deployer-tree scenarios", ssvc_scenarios), ("remediation deadlines", remediation_deadlines), ("full options for core tools", core_tool_options),
           ("lua programs (rosetta code)", rosetta_lua), ("zap help", zap_help)]

# ================================================================== SERVICES: what each one is, and what indicates a problem
SCM_NOTE = ("Signs of trouble in the System event log (source Service Control Manager): event 7000 or 7009 when the service fails to start or times out, 7031 or 7034 when it terminates unexpectedly, "
            "7040 when its start type is changed, and 7045 when a new service is installed.")


def plain(block: str, tag: str) -> str:
    m = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", block, re.S)      # strip tags before decoding entities, so an escaped "<Directive>" survives
    return html.unescape(" ".join(re.sub(r"<[^>]+>", "", m.group(1)).split())) if m else ""


def windows_services() -> None:
    p = os.path.join(S, "services/windows-server-docs/WindowsServerDocs/security/windows-services/security-guidelines-for-disabling-system-services-in-windows-server.md")
    for m in re.finditer(r"^### (.+?)\n(.*?)(?=^### |^## |\Z)", read(p) if os.path.isfile(p) else "", re.M | re.S):
        rows = {k.strip(): v.strip() for k, v in re.findall(r"^\| \*\*(.+?)\*\* \| (.*?) \|\s*$", m.group(2), re.M)}
        svc, desc = rows.get("Service name"), rows.get("Description")
        if not svc or not desc:
            continue
        display = re.sub(r"\s*\([^)]*\)\s*$", "", m.group(1)).strip()
        rec = rows.get("Recommendation", "")
        a = (f"**{display}** (service name `{svc}`): {desc}\n\n- Installed: {rows.get('Installation', 'not stated')}\n- Default startup type: {rows.get('Startup type', 'not stated')}\n"
             f"- Microsoft's guidance for servers: {rec or 'none given'}" + (f" ({rows['Comments']})" if rows.get("Comments") and rows["Comments"].lower() != "none" else "")
             + f"\n\nInspect it with `Get-Service -Name {svc}`, or `sc.exe qc {svc}` for its configuration and `sc.exe query {svc}` for its state.")
        if rec.lower().startswith(("ok to disable", "should be disabled")):
            a += f" To disable it: `Set-Service -Name {svc} -StartupType Disabled` and then `Stop-Service -Name {svc}`."
        emit("windows_service", "services", "A", pick(["What does the Windows service \"{d}\" ({s}) do, and can it be disabled on a server?", "Windows service `{s}`: what is it for, what is its default startup type, and what signals a problem with it?",
                                                       "Tell me about the {d} service in Windows Server."], d=display, s=svc),
             a + "\n\n" + SCM_NOTE, source="windows-server-docs (system services guidance)", key=f"winsvc:{svc}", licence="CC BY 4.0")
    p = os.path.join(S, "services/windows-service-pages/service-overview-and-network-port-requirements.txt")
    blocks = [b.split("\n") for b in re.split(r"\n\s*\n", read(p) if os.path.isfile(p) else "")]
    pending, i = [], 0
    while i < len(blocks):
        b = blocks[i]
        if b[0].startswith("System service name:") and pending and i + 1 < len(blocks) and blocks[i + 1][:1] == ["Application protocol"]:
            title = next((x[0] for x in pending if len(x) == 1 and len(x[0]) < 90), pending[0][0])
            desc = next((" ".join(x) for x in pending if len(" ".join(x)) > 120), "")
            rows, j = [], i + 2
            while j < len(blocks) and len(blocks[j]) == 3 and re.match(r"^(TCP|UDP|N/A|GRE|ESP|ICMP)", blocks[j][1].strip(), re.I):
                rows.append(blocks[j])
                j += 1
            if rows:
                emit("windows_ports", "services", "A", pick(["Which network ports does the Windows {t} service use?", "What do I need to allow through a firewall for {t} on Windows?"], t=title.strip()),
                     (cut(desc, 900) + "\n\n" if desc else "") + f"{b[0].strip()}\n\n| Application protocol | Protocol | Ports |\n|---|---|---|\n" + "\n".join(f"| {r[0].strip()} | {r[1].strip()} | {r[2].strip()} |" for r in rows),
                     source="microsoft support: service overview and network port requirements", key=f"winports:{title[:40]}", min_a=120, licence="CC BY 4.0")
            pending, i = [], j
            continue
        if len(b) == 3 and pending and re.match(r"^(TCP|UDP)", b[1].strip(), re.I):
            pending = []
        else:
            pending.append(b)
        i += 1


def systemd_units() -> None:
    lic = "LGPL-2.1-or-later (systemd)"
    for path in G("services/systemd/man/*.xml"):
        raw = read(path)
        names = [re.sub(r"<[^>]+>", "", n).strip() for n in re.findall(r"<refname>(.*?)</refname>", raw, re.S)]
        purpose = plain(raw, "refpurpose")
        secs = docbook_sections(raw)
        desc = next((markup_text(b) for t, b in secs if t.lower() == "description"), "")
        if not names or not purpose or len(desc) < 200:
            continue
        name = names[0]
        topic = "logging" if re.search(r"journal|log|coredump", name) else "services"
        if name.endswith((".service", ".socket", ".target", ".timer")):
            emit("systemd_unit", topic, "A", pick(["What does `{n}` do on a systemd Linux system, and how do I check whether it is healthy?", "What is `{n}` responsible for, and how do I inspect it?"], n=name),
                 f"**{name}**: {purpose}.\n\n{cut(desc, 2600)}\n\nTo check it: `systemctl status {name}` shows whether it is active along with its most recent log lines; `journalctl -u {name} -b` shows its log for the current boot; "
                 f"`systemctl is-failed {name}` succeeds only if the unit is in a failed state; `systemctl list-dependencies {name}` shows what it pulls in.",
                 source="systemd/man/" + os.path.basename(path), key=f"systemd:{name}", joined=True, licence=lic)
        else:
            emit("systemd_unit", topic, "A", pick(["What is `{n}` in systemd, and what is it used for?", "Explain `{n}` (systemd)."], n=name), f"**{name}**: {purpose}.\n\n{cut(desc, 3200)}",
                 source="systemd/man/" + os.path.basename(path), key=f"systemd:{name}", licence=lic)
        n = 0
        for term, body in varlist(raw):
            if term.rstrip().endswith("=") and len(body) > 150 and n < 60:
                emit("systemd_setting", topic, "A", pick(["What does the systemd setting `{t}` do?", "systemd: explain `{t}` (documented in {n})."], t=term.split(",")[0].strip(), n=name), cut(body, 3200),
                     source="systemd/man/" + os.path.basename(path), key=f"systemdopt:{term[:30]}", licence=lic)
                n += 1


def monitoring_cards() -> None:
    for readme in G("indicators/datadog-integrations/*/README.md"):
        d = os.path.dirname(readme)
        title = os.path.basename(d).replace("_", " ")
        ov = re.search(r"^## Overview\s*\n(.*?)(?=^## )", read(readme), re.M | re.S)
        overview = clean(re.sub(r"!\[[^\]]*\]\[[^\]]*\]|!\[[^\]]*\]\([^)]*\)|<[^>\n]{1,200}>", "", re.sub(r"\[([^\]]+)\]\[\d+\]", r"\1", ov.group(1)))) if ov else ""
        checks, metrics = [], []
        try:
            checks = [c for c in json.loads(read(os.path.join(d, "assets", "service_checks.json"))) if isinstance(c, dict) and c.get("check")]
        except Exception:  # noqa: BLE001
            pass
        try:
            with open(os.path.join(d, "metadata.csv"), encoding="utf-8", newline="") as fh:
                rows = [r for r in csv.DictReader(fh) if r.get("description")]
            metrics = ([r for r in rows if r.get("curated_metric")] + [r for r in rows if not r.get("curated_metric")])[:14]
        except Exception:  # noqa: BLE001
            pass
        if not checks and len(metrics) < 4:
            continue
        a = (cut(overview, 800) + "\n\n" if len(overview) > 60 else "")
        if checks:
            a += "**Health checks** (anything other than OK is the problem signal):\n\n" + "\n".join(f"- `{c['check']}` ({', '.join(c.get('statuses') or [])}): {c.get('description', '')}" for c in checks[:8]) + "\n\n"
        if metrics:
            a += "**Metrics worth watching:**\n\n" + "\n".join(f"- `{r['metric_name']}`" + (f" ({r['unit_name']})" if r.get("unit_name") else "") + f": {r['description']}" for r in metrics)
        emit("service_monitoring", "services", "B", pick(["What should I monitor for {t}, and what indicates a problem?", "Which health checks and metrics matter for {t}?"], t=title), a,
             source="datadog integrations-core", key=f"service:{title}", licence="BSD-3-Clause")
    for path in G("indicators/netdata-health/src/health/guides/**/*.md"):
        text = read(path)
        if "### Troubleshoot the alert" not in text and "### Troubleshoot" not in text:
            continue
        alert, svc = os.path.basename(path)[:-3], os.path.basename(os.path.dirname(path)).replace("_", " ")
        emit("alert_guide", "services", "B", pick(["The alert `{a}` fired ({s}). What does it mean and how do I troubleshoot it?", "How do I investigate a `{a}` alert on {s}?"], a=alert, s=svc),
             re.sub(r"^### ", "**", re.sub(r"^### (.+)$", r"### \1**", text, flags=re.M), flags=re.M).replace("### ", ""), source="netdata health guides", key=f"alert:{alert}", licence="GPL-3.0")
    order = {"DISASTER": 0, "HIGH": 1, "AVERAGE": 2, "WARNING": 3, "INFO": 4, "NOT_CLASSIFIED": 5}
    trig = re.compile(r"expression: '?(.+?)'?\n(?:\s+recovery_\w+: .*\n)*\s+name: '?(.+?)'?\n(?:\s+event_name: .*\n)?(?:\s+opdata: .*\n)?\s+priority: (\w+)\n(?:\s+description: '?([^|>\n][^\n]*?)'?\n)?")
    for path in G("indicators/zabbix-templates/templates/**/*.yaml"):
        if os.path.getsize(path) > 3_000_000:
            continue
        text = read(path, 3_000_000)
        tm = re.search(r"^\s+template: '?(.+?)'?\s*$", text, re.M)
        seen, items = set(), []
        for expr, name, prio, desc in trig.findall(text):
            if name not in seen:
                seen.add(name)
                items.append((order.get(prio, 5), name, prio, desc or "", " ".join(expr.split())))
        if not tm or len(items) < 3:
            continue
        items.sort()
        emit("service_problem_triggers", "services", "B", pick(["What conditions does Zabbix treat as problems for \"{t}\"?", "Which trigger conditions indicate trouble under the \"{t}\" monitoring template?"], t=tm.group(1)),
             "Problem conditions, most severe first:\n\n" + "\n".join(f"- **{n}** ({p.lower()})" + (f": {d}" if d else "") + (f"\n  `{e[:220]}`" if len(e) < 400 else "") for _, n, p, d, e in items[:14]),
             source="zabbix templates", key=f"service:{tm.group(1)}", licence="AGPL-3.0")


def server_directives() -> None:
    for path in G("services/nginx-docs/xml/en/docs/**/*.xml"):
        raw, mod = read(path), os.path.basename(path)[:-4]
        for name, body in re.findall(r'<directive name="([^"]+)">(.*?)</directive>', raw, re.S):
            syntax = [" ".join(re.sub(r"<[^>]+>", "", s).split()) for s in re.findall(r"<syntax[^>]*>(.*?)</syntax>", body, re.S)]
            default, ctxs = plain(body, "default"), re.findall(r"<context>(.*?)</context>", body)
            desc = markup_text(re.sub(r"<(syntax|default|context|appeared-in)\b[^>]*>.*?</\1>|<default/>", "", body, flags=re.S))
            if len(desc) < 120:
                continue
            emit("nginx_directive", "services", "A", pick(["What does the nginx directive `{n}` do? Give the syntax, default and context.", "nginx: explain `{n}`."], n=name),
                 f"**`{name}`** ({mod})\n\n- Syntax: " + "; ".join(f"`{name} {s};`".replace(" ;", ";") for s in syntax[:3]) + f"\n- Default: {f'`{name} {default};`' if default else 'none'}\n- Context: {', '.join(ctxs)}\n\n{cut(desc, 3500)}",
                 source="nginx.org docs", key=f"nginx:{name}", licence="BSD-2-Clause")
    for path in G("services/apache-httpd-docs/docs/manual/mod/*.xml"):
        raw, mod = read(path, 3_000_000), os.path.basename(path)[:-4]
        for block in re.findall(r"<directivesynopsis\b[^>]*>(.*?)</directivesynopsis>", raw, re.S):
            name, usage = plain(block, "name"), re.search(r"<usage>(.*?)</usage>", block, re.S)
            text = markup_text(usage.group(1)) if usage else ""
            if not name or len(text) < 150:
                continue
            dflt, ctxs = plain(block, "default"), ", ".join(re.findall(r"<context>(.*?)</context>", block))
            emit("apache_directive", "services", "A", pick(["What does the Apache httpd directive `{n}` do? Give the syntax, default and context.", "Apache httpd: explain `{n}`."], n=name),
                 f"**`{name}`** ({mod}): {plain(block, 'description')}\n\n- Syntax: `{plain(block, 'syntax')}`\n- Default: " + (f"`{dflt}`" if dflt else "none")
                 + f"\n- Context: {ctxs}\n\n{cut(text, 3500)}", source="apache httpd docs", key=f"apache:{name}", licence="Apache-2.0")
    p = os.path.join(S, "services/postgresql-docs/doc/src/sgml/config.sgml")
    for term, body in varlist(read(p, 3_000_000)) if os.path.isfile(p) else []:
        name = term.split(" ")[0].strip("`")
        if re.fullmatch(r"[a-z][a-z0-9_.]+", name) and len(body) > 200:
            emit("postgres_setting", "services", "A", pick(["What does the PostgreSQL setting `{n}` control?", "PostgreSQL: explain `{n}` and when to change it."], n=name), f"`{term}`\n\n{cut(body, 3500)}",
                 source="postgresql docs (config.sgml)", key=f"pg:{name}", licence="PostgreSQL License")
    for stem in ("runtime", "monitoring", "maintenance", "client-auth", "backup", "high-availability", "wal", "user-manag"):
        p = os.path.join(S, f"services/postgresql-docs/doc/src/sgml/{stem}.sgml")
        if os.path.isfile(p):
            docbook("admin_reference", "services", "A", p, "PostgreSQL administration", cap=40, licence="PostgreSQL License")
    for p in G("services/debian-handbook/en-US/*.xml"):
        docbook("admin_reference", "services", "B", p, "Debian system administration", cap=40, licence="GPL-2.0-or-later / CC BY-SA 3.0")
    rst_docs("services_docs", "services", "A", G("services/envoy-docs/docs/root/operations/**/*.rst") + G("services/envoy-docs/docs/root/intro/arch_overview/security/**/*.rst")
             + G("services/envoy-docs/docs/root/configuration/observability/**/*.rst"), "Envoy", cap_per_file=5, licence="Apache-2.0")


# ================================================================== NETWORKING
RFC_SKIP = re.compile(r"^(status of (this|the) memo|copyright|table of contents|acknowledg|authors?'? address|(normative |informative )?references|iana considerations|contributors|"
                      r"full copyright|intellectual property|requirements (language|notation)|conventions used|changes (from|since)|index)", re.I)


def networking() -> None:
    freq, iana = {}, collections.defaultdict(list)
    p = os.path.join(S, "tool-docs/nmap/nmap-services")
    for ln in (read(p, 3_000_000).split("\n") if os.path.isfile(p) else []):
        m = re.match(r"^(\S+)\t(\d+)/(tcp|udp)\t([\d.]+)", ln)
        if m and float(m.group(4)) > 0:
            freq[(int(m.group(2)), m.group(3))] = (m.group(1), float(m.group(4)))
    p = os.path.join(S, "services/iana/service-names-port-numbers.csv")
    if os.path.isfile(p):
        with open(p, encoding="utf-8", newline="") as fh:
            for row in csv.reader(fh):
                if len(row) > 3 and row[0] and row[1].isdigit() and row[2] in ("tcp", "udp") and row[3] and (row[0], row[3]) not in iana[(int(row[1]), row[2])]:
                    iana[(int(row[1]), row[2])].append((row[0], " ".join(row[3].split())))
    for proto, top in (("tcp", 900), ("udp", 300)):
        ranked = sorted(((k, v) for k, v in freq.items() if k[1] == proto), key=lambda kv: -kv[1][1])[:top]
        for rank, ((port, _), (nname, _f)) in enumerate(ranked, 1):
            reg = iana.get((port, proto), [])
            local = "`ss -ltnp`" if proto == "tcp" else "`ss -lunp`"
            win = f"`Get-NetTCPConnection -LocalPort {port} -State Listen`" if proto == "tcp" else f"`Get-NetUDPEndpoint -LocalPort {port}`"
            emit("port_card", "networking", "A", pick(["What usually runs on {P} port {n}?", "Which service is {P} port {n} associated with, and how do I confirm what is listening?", "{P}/{n}: what is it?"], P=proto.upper(), n=port),
                 f"{proto.upper()} port {port}: " + ("IANA registers it to " + "; ".join(f"`{n}` ({d})" for n, d in reg[:3]) + "." if reg else "IANA has no registered service for it.")
                 + f" Nmap's service table names it `{nname}` and ranks it number {rank} among the {proto.upper()} ports most often found open in its scan data."
                 f"\n\nA port number is a convention, not proof. Confirm what is actually listening with a version probe (`nmap -sV -p {port} <host>`) or on the host itself: {local} on Linux, {win} on Windows.",
                 source="iana registry + nmap-services", key=f"port:{proto}/{port}", joined=True, min_a=150, licence="IANA registry (CC0); Nmap Public Source License")
    for path in G("references/rfc/rfc*.txt"):
        num = re.search(r"rfc(\d+)", os.path.basename(path)).group(1)
        # drop page footers and headers; keep a paragraph break only where the text before the page break had ended a sentence
        text = re.sub(r"\n+[^\n]*\[Page \d+\][ \t]*\n\f?\n*[^\n]*\n+", "\n\x00", read(path, 3_000_000)).replace("\f", "")
        text = re.sub(r"([.:;!?)\]])\n\x00", r"\1\n\n", text).replace("\x00", "")
        head = text.split("\n")[:70]
        title = next((ln.strip() for i, ln in enumerate(head) if i > 2 and re.match(r"^ {3,}\S", ln) and 8 < len(ln.strip()) < 110 and not head[i - 1].strip()
                      and not re.search(r"Request for Comments|Category:|Obsoletes|Updates|ISSN|STD:|BCP:|^\s*(January|February|March|April|May|June|July|August|September|October|November|December) ", ln)), "")
        ms = list(re.finditer(r"^(\d+(?:\.\d+){0,2})\.?[ \t]{1,3}([A-Z][^\n]{2,70})$", text, re.M))
        n = 0
        for i, m in enumerate(ms):
            body = text[m.end():ms[i + 1].start() if i + 1 < len(ms) else len(text)]
            body = "\n".join(ln[3:] if ln.startswith("   ") else ln for ln in body.split("\n")).strip()
            if len(body) < 400 or RFC_SKIP.match(m.group(2).strip()) or n >= 14:
                continue
            emit("rfc_section", "networking", "A", pick(["RFC {r}{t}: what does section {s}, \"{h}\", specify?", "What does RFC {r}{t} say in section {s} ({h})?"], r=num, t=f" ({title})" if title else "", s=m.group(1), h=m.group(2).strip()),
                 cut(body, 5200), source=f"rfc{num}", key=f"rfc:{num}", licence="IETF Trust licence")
            n += 1


# ================================================================== KERNEL AND MEMORY
HARDEN_LABEL = {"self_protection": "kernel self-protection", "security_policy": "security policy enforcement", "cut_attack_surface": "attack surface reduction", "harden_userspace": "userspace hardening"}
HARDEN_WHO = {"defconfig": "the kernel's own default configuration", "kspp": "the Kernel Self Protection Project", "grsec": "the grsecurity guidance", "clipos": "the CLIP OS project",
              "a13xp0p0v": "the kernel-hardening-checker author", "maintainer": "kernel maintainers", "lockdown": "the kernel lockdown guidance"}


def kernel_hardening() -> None:
    p = os.path.join(S, "kernel-memory/linux-kernel-docs/Documentation/admin-guide/kernel-parameters.txt")
    cur, buf = None, []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if cur and re.fullmatch(r"[A-Za-z0-9_.\-]+=?", cur) and len(body) >= 150:
            emit("kernel_param", "kernel-memory", "A", pick(["What does the Linux kernel boot parameter `{n}` do?", "Linux kernel command line: explain `{n}` and its values."], n=cur),
                 "```\n" + cut(body, 4200) + "\n```\n\nThe bracketed tags at the start list the architectures and build options it applies to.", source="linux Documentation/admin-guide/kernel-parameters.txt",
                 key=f"kparam:{cur}", licence="GPL-2.0")
    for ln in (read(p, 4_000_000).split("\n") if os.path.isfile(p) else []):
        if re.match(r"^\t[^\t\s]", ln):
            flush()
            parts = ln.strip().split("\t", 1)
            cur, buf = parts[0].strip(), ([parts[1].strip()] if len(parts) > 1 and parts[1].strip() else [])
        elif cur and (ln.startswith("\t\t") or not ln.strip()):
            buf.append(re.sub(r"^\t{2,3}", "", ln).replace("\t", "    ").rstrip())
        else:
            flush()
            cur, buf = None, []
    flush()
    cp = G("kernel-memory/kernel-hardening-checker/**/checks.py")
    groups: dict = collections.OrderedDict()
    for kind, reason, decision, name, expected in re.findall(r"(Kconfig|Cmdline|Sysctl)Check\('(\w+)', '(\w+)', '([\w.]+)', '([^']*)'\)", read(cp[0]) if cp else ""):
        lst = groups.setdefault((kind, reason, decision), [])
        if (name, expected) not in lst:
            lst.append((name, expected))
    for (kind, reason, decision), items in groups.items():
        who, label = HARDEN_WHO.get(decision, decision), HARDEN_LABEL.get(reason, reason.replace("_", " "))
        for i in range(0, len(items), 40):
            part = items[i:i + 40]
            if len(part) < 3:
                continue
            if kind == "Kconfig":
                listing = "\n".join(f"# CONFIG_{n} is not set" if e == "is not set" else f"CONFIG_{n}={e}" for n, e in part)
                q, tail = f"Which kernel build options does {who} set for {label}?", "Compare a kernel's build configuration against the full list with `kernel-hardening-checker -c /boot/config-$(uname -r)`."
            elif kind == "Cmdline":
                listing = "\n".join(f"{n} (must not be set)" if e == "is not set" else f"{n}={e}" if e != "is set" else n for n, e in part)
                q, tail = f"Which kernel boot parameters does {who} recommend for {label}?", "See what the running kernel was booted with in `/proc/cmdline`."
            else:
                listing = "\n".join(f"{n} = {e}" for n, e in part)
                q, tail = f"Which sysctl settings does {who} recommend for {label}?", "Read a current value with `sysctl <name>`; persist changes in a file under `/etc/sysctl.d/`."
            emit("kernel_hardening_list", "kernel-memory", "B", q + (" (continued)" if i else ""), f"As checked by kernel-hardening-checker (category `{reason}`, source `{decision}`):\n\n```\n{listing}\n```\n\n{tail}",
                 source="kernel-hardening-checker checks.py", key=f"khc:{kind}:{reason}:{decision}:{i}", min_a=120, licence="GPL-3.0")
    rst_docs("kernel_sysctl", "kernel-memory", "A", G("kernel-memory/linux-kernel-docs/Documentation/admin-guide/sysctl/*.rst"), "Linux sysctl settings", cap_per_file=250, min_body=150, licence="GPL-2.0")
    rst_docs("kernel_docs", "kernel-memory", "A", G("kernel-memory/linux-kernel-docs/Documentation/admin-guide/hw-vuln/*.rst") + G("kernel-memory/linux-kernel-docs/Documentation/admin-guide/LSM/*.rst"),
             lambda p: "Linux kernel " + ("CPU vulnerability mitigations" if "hw-vuln" in p.replace("\\", "/") else "security modules"), cap_per_file=30, min_body=250, licence="GPL-2.0")
    docs("ebpf_reference", "kernel-memory", "B", G("kernel-memory/ebpf-docs/docs/linux/helper-function/*.md") + G("kernel-memory/ebpf-docs/docs/linux/map-type/*.md"), "eBPF", cap_per_file=2, licence="CC BY 4.0 / BSD-2-Clause")
    pdf_doc("memory_safety", "kernel-memory", "A", os.path.join("kernel-memory", "guidance", "CISA-case-for-memory-safe-roadmaps.txt"), "CISA's \"The Case for Memory Safe Roadmaps\"", cap=30)


# ================================================================== TESTING and BROWSER TOOLING
def testing_material() -> None:
    for path in G("testing/fuzzingbook/notebooks/*.ipynb"):
        try:
            nb = json.loads(read(path, 30_000_000))
        except Exception:  # noqa: BLE001
            continue
        parts = []
        for c in nb.get("cells", []):
            src = "".join(c.get("source", []))
            if not src.strip():
                continue
            if c.get("cell_type") == "markdown":
                parts.append(src)
            elif c.get("cell_type") == "code" and not re.match(r"\s*(import bookutils|from bookutils|# ignore|quiz\(|YouTubeVideo|import Intro_)", src):
                parts.append("```python\n" + src.strip() + "\n```")
        text = "\n\n".join(parts)
        title = (re.search(r"^# (.+)$", text, re.M) or [None, os.path.basename(path)[:-6]])[1].strip()
        sectioned("fuzzing_book", "testing", "B", text, "software fuzzing and test generation", source="fuzzingbook/" + os.path.basename(path), key=f"fuzzingbook:{os.path.basename(path)}", cap=10,
                  min_body=500, licence="CC BY-NC-SA 4.0 (text), MIT (code)", title=title)
    pdf_doc("testing_guidance", "testing", "A", os.path.join("references", "nist", "NIST.SP.800-115.txt"), "NIST SP 800-115 (Technical Guide to Information Security Testing and Assessment)", cap=45)


def browser_tooling() -> None:
    for path in G("web-tools/devtools-protocol/json/*.json"):
        try:
            doms = json.loads(read(path, 20_000_000)).get("domains", [])
        except Exception:  # noqa: BLE001
            continue
        for dom in doms:
            name = dom.get("domain")
            cmds = [c for c in dom.get("commands", []) if not c.get("deprecated")]
            if not name or dom.get("deprecated") or not cmds:
                continue
            emit("cdp_domain", "web-tools", "A", pick(["What can I do with the `{n}` domain of the Chrome DevTools Protocol?", "Chrome DevTools Protocol: what is the `{n}` domain for?"], n=name),
                 (f"{dom.get('description', '').strip()}\n\n" if dom.get("description") else "") + "**Commands:**\n" + "\n".join(f"- `{name}.{c['name']}`: {' '.join(str(c.get('description', '')).split()) or 'no description'}" for c in cmds[:36])
                 + ("\n\n**Events:**\n" + "\n".join(f"- `{name}.{e['name']}`: {' '.join(str(e.get('description', '')).split())}" for e in dom.get("events", [])[:18]) if dom.get("events") else ""),
                 source="devtools-protocol", key=f"cdp:{name}", licence="BSD-3-Clause")
            for c in cmds:
                params, rets = c.get("parameters") or [], c.get("returns") or []
                if not c.get("description") or not (params or rets):
                    continue
                fmt = lambda ps: "\n".join(f"- `{p['name']}` ({p.get('type') or p.get('$ref', '')}{', optional' if p.get('optional') else ''}): {' '.join(str(p.get('description', '')).split())}" for p in ps[:14])  # noqa: E731
                req = ", ".join(f'"{p["name"]}": <{p.get("type") or p.get("$ref", "value")}>' for p in params if not p.get("optional"))
                emit("cdp_command", "web-tools", "A", pick(["Chrome DevTools Protocol: what does `{n}` do and what does it take?", "How do I call `{n}` over the DevTools Protocol?"], n=f"{name}.{c['name']}"),
                     f"{' '.join(c['description'].split())}" + (f"\n\n**Parameters:**\n{fmt(params)}" if params else "") + (f"\n\n**Returns:**\n{fmt(rets)}" if rets else "")
                     + f"\n\nAs a protocol message:\n```json\n{{\"id\": 1, \"method\": \"{name}.{c['name']}\", \"params\": {{{req}}}}}\n```", source="devtools-protocol", key=f"cdp:{name}", min_a=150, licence="BSD-3-Clause")
    docs("playwright_api", "web-tools", "A", G("web-tools/playwright/docs/src/api/class-*.md"), "the Playwright API", cap_per_file=25, min_body=200, licence="Apache-2.0")


# ================================================================== DETECTION AND RESPONSE (defensive material only)
def val(x) -> str:
    if isinstance(x, list):
        return "; ".join(val(i) for i in x if val(i))
    if isinstance(x, dict):
        return str(x.get("@value", ""))
    return str(x or "")


def detection_cards() -> None:
    import tomllib
    names: dict[str, str] = {}
    for path in G("tool-docs/atomic-red-team/atomics/T*/T*.md"):
        m = re.match(r"# (T[\d.]+) - (.+)", read(path, 300))
        if m:
            names[m.group(1)] = m.group(2).strip()
    by: dict = collections.defaultdict(dict)

    def offer(tid: str, lang: str, title: str, code: str, fence: str) -> None:
        """Keep, per technique and language, the rule whose title best matches the technique's name; the shorter one on a tie."""
        code = code.strip()
        if not 120 <= len(code) <= 1500:
            return
        fit = len(set(re.findall(r"[a-z]{4,}", names.get(tid, "").lower())) & set(re.findall(r"[a-z]{4,}", str(title).lower())))
        cur = by[tid].get(lang)
        if cur is None or (fit, -len(code)) > (cur[3], -len(cur[1])):
            by[tid][lang] = (title, code, fence, fit)
    for path in G("tool-docs/sigma/rules/**/*.yml"):
        r = yload(path)
        if isinstance(r, dict) and r.get("title") and r.get("detection"):
            code = yaml.safe_dump({k: r[k] for k in ("title", "logsource", "detection", "level") if k in r}, sort_keys=False, allow_unicode=True, width=120)
            for tag in r.get("tags") or []:
                m = re.match(r"attack\.(t\d{4}(?:\.\d{3})?)$", str(tag))
                if m:
                    offer(m.group(1).upper(), "Sigma", r["title"], code, "yaml")
    for path in G("remediation/splunk-security-content/detections/**/*.yml"):
        r = yload(path)
        if isinstance(r, dict) and isinstance(r.get("search"), str) and r.get("status") != "deprecated":
            for tid in ((r.get("tags") or {}).get("mitre_attack_id") or []) if isinstance(r.get("tags"), dict) else []:
                offer(str(tid), "Splunk SPL", r.get("name", ""), r["search"], "spl")
    for path in G("remediation/elastic-detection-rules/rules/**/*.toml"):
        try:
            r = tomllib.loads(read(path)).get("rule", {})
        except Exception:  # noqa: BLE001
            continue
        if r.get("query") and r.get("language") in ("eql", "kuery", "esql"):
            for th in r.get("threat") or []:
                for t in th.get("technique") or []:
                    names.setdefault(t.get("id", ""), t.get("name", ""))
                    for tid in [t.get("id")] + [s.get("id") for s in t.get("subtechnique") or []]:
                        if tid:
                            offer(tid, f"Elastic {r['language'].upper()}", r.get("name", ""), r["query"], r["language"])
    for path in G("logging/azure-sentinel/Detections/**/*.yaml"):
        r = yload(path)
        if isinstance(r, dict) and isinstance(r.get("query"), str) and r.get("name"):
            for tid in r.get("relevantTechniques") or []:
                offer(str(tid), "Microsoft Sentinel KQL", r["name"], r["query"], "kql")
    for tid, langs in by.items():
        order = [k for k in ("Sigma", "Splunk SPL", "Microsoft Sentinel KQL") if k in langs] + [k for k in ("Elastic EQL", "Elastic KUERY", "Elastic ESQL") if k in langs][:1]
        order = [k for k in order if langs[k][3] >= 1] if names.get(tid) else order      # only rules that are plainly about this technique
        if len(order) < 2:
            continue
        label = f"{tid} ({names[tid]})" if names.get(tid) else tid
        emit("detection_multi", "detection", "B", pick(["Show me how to detect ATT&CK {l} in {x}.", "I need detections for ATT&CK {l}. Give me {x} versions."], l=label, x=", ".join(order[:-1]) + " and " + order[-1]),
             "\n\n".join(f"**{k}: {langs[k][0]}**\n```{langs[k][2]}\n{langs[k][1]}\n```" for k in order)
             + "\n\nThese express related ideas against different data models, and each looks at a different facet of the technique. Field names and macros are specific to each platform's schema, so map them to your own log sources and test against known-good activity before enabling alerts.",
             source="sigma + splunk security_content + elastic detection-rules + azure-sentinel", key=f"multi:{tid}", joined=True, licence="DRL 1.1 / Apache-2.0 / Elastic License 2.0 / MIT")
    for path in G("remediation/mitre-car/analytics/*.yaml"):
        r = yload(path)
        if not isinstance(r, dict) or not r.get("title") or not r.get("description"):
            continue
        impl = next((i for i in (r.get("implementations") or []) if isinstance(i, dict) and i.get("code")), None)
        cov = ", ".join(str(c.get("technique")) for c in (r.get("coverage") or []) if isinstance(c, dict) and c.get("technique"))
        a = str(r["description"]).strip() + (f"\n\nATT&CK coverage: {cov}." if cov else "")
        if impl and len(str(impl["code"])) < 2200:
            a += f"\n\n**Example implementation** ({impl.get('type') or impl.get('name') or 'pseudocode'})" + (f": {' '.join(str(impl['description']).split())}" if impl.get("description") else "") + f"\n```\n{str(impl['code']).strip()}\n```"
        emit("detection_analytic", "detection", "A", pick(["What should I look for to detect this behaviour: {t}? Give the analytic and an implementation.", "Describe a detection analytic for: {t}."], t=r["title"]), a,
             source="mitre car", key=f"car:{r.get('id')}", licence="Apache-2.0")
    p = os.path.join(S, "remediation/mitre/d3fend.json")
    for x in (json.loads(read(p, 30_000_000)).get("@graph", []) if os.path.isfile(p) else []):
        if not isinstance(x, dict) or not x.get("d3f:d3fend-id") or not x.get("d3f:definition"):
            continue
        label, did, kb, syn = val(x.get("rdfs:label")), val(x["d3f:d3fend-id"]), val(x.get("d3f:kb-article")), val(x.get("d3f:synonym"))
        tactic = val((x.get("d3f:enables") or {}).get("@id") if isinstance(x.get("d3f:enables"), dict) else "").replace("d3f:", "")
        emit("d3fend_technique", "remediation", "A", pick(["What is the MITRE D3FEND defensive technique \"{l}\" ({i}), and how does it work?", "Explain the D3FEND countermeasure {l} ({i})."], l=label, i=did),
             f"**{label}** ({did})" + (f", a technique under the {tactic} tactic" if tactic else "") + f": {val(x['d3f:definition'])}" + (f"\n\nAlso called: {syn}." if syn else "") + (f"\n\n{cut(kb, 3800)}" if kb else ""),
             source="mitre d3fend", key=f"d3fend:{did}", min_a=220, licence="MIT")


NIST_TITLES = {"800-53": "Security and Privacy Controls", "800-53A": "Assessing Security and Privacy Controls", "800-61": "Incident Response", "800-150": "Guide to Cyber Threat Information Sharing",
               "800-171": "Protecting Controlled Unclassified Information", "800-207": "Zero Trust Architecture", "800-190": "Application Container Security Guide", "800-82": "Guide to Operational Technology Security",
               "800-37": "Risk Management Framework", "800-86": "Guide to Integrating Forensic Techniques into Incident Response", "800-94": "Guide to Intrusion Detection and Prevention Systems",
               "800-123": "Guide to General Server Security", "800-63": "Digital Identity Guidelines", "800-218": "Secure Software Development Framework", "800-161": "Cybersecurity Supply Chain Risk Management",
               "800-184": "Guide for Cybersecurity Event Recovery", "800-83": "Guide to Malware Incident Prevention and Handling", "800-81": "Secure Domain Name System Deployment Guide",
               "800-52": "Guidelines for TLS Implementations", "800-57": "Recommendation for Key Management", "800-41": "Guidelines on Firewalls and Firewall Policy", "800-160": "Systems Security Engineering",
               "800-204": "Security Strategies for Microservices-based Application Systems", "800-144": "Guidelines on Security and Privacy in Public Cloud Computing", "800-145": "The NIST Definition of Cloud Computing",
               "800-125": "Guide to Security for Full Virtualization Technologies", "800-153": "Guidelines for Securing Wireless Local Area Networks", "800-177": "Trustworthy Email", "800-88": "Guidelines for Media Sanitization",
               "800-34": "Contingency Planning Guide", "800-84": "Guide to Test, Training, and Exercise Programs", "800-12": "An Introduction to Information Security", "800-137": "Information Security Continuous Monitoring",
               "800-213": "IoT Device Cybersecurity Guidance", "800-128": "Guide for Security-Focused Configuration Management", "800-39": "Managing Information Security Risk", "800-95": "Guide to Secure Web Services",
               "800-113": "Guide to SSL VPNs", "800-124": "Guidelines for Managing the Security of Mobile Devices", "800-193": "Platform Firmware Resiliency Guidelines", "800-215": "Guide to a Secure Enterprise Network Landscape",
               "800-47": "Managing the Security of Information Exchanges", "800-216": "Recommendations for Federal Vulnerability Disclosure Guidelines", "800-55": "Measurement Guide for Information Security"}
NIST_TOPIC = {"800-81": "networking", "800-52": "networking", "800-41": "networking", "800-113": "networking", "800-153": "networking", "800-215": "networking", "800-47": "networking",
              "800-190": "services", "800-204": "services", "800-144": "services", "800-145": "services", "800-125": "services", "800-123": "services", "800-94": "detection", "800-137": "detection",
              "800-37": "prioritization", "800-39": "prioritization", "800-216": "prioritization", "800-193": "kernel-memory"}
NIST_DONE = {"800-92", "800-40", "800-30", "800-115"}                # handled by the topic-specific converters above
IRM_STAGE = re.compile(r"^(preparation|identification|containment|remediation|recovery|lessons learned|aftermath)\b.{0,12}$", re.I)


def reference_pdfs() -> None:
    import glob
    for path in sorted(glob.glob(os.path.join(PDF, "references", "nist", "*.txt"))):
        stem = os.path.basename(path)[:-4]
        m = re.search(r"SP\.(\d+-\d+[A-Z]?)", stem)
        base = m.group(1) if m else ""
        if not base or base in NIST_DONE:
            continue
        label = stem.replace("NIST.SP.", "NIST SP ") + (f" ({NIST_TITLES[base]})" if base in NIST_TITLES else "")
        pdf_doc("nist_guidance", NIST_TOPIC.get(base, "remediation"), "A", os.path.relpath(path, PDF), label, cap=30)
    pdf_doc("cybok", "defense", "A", os.path.join("references", "cybok", "CyBOK_v1.1.0.txt"), "the Cyber Security Body of Knowledge (CyBOK 1.1)", cap=140, licence="Open Government Licence v3.0")
    pdf_doc("ir_playbook", "remediation", "A", os.path.join("remediation", "cisa", "incident-and-vulnerability-response-playbooks.txt"),
            "CISA's Federal Government Cybersecurity Incident and Vulnerability Response Playbooks", cap=40)
    for path in sorted(glob.glob(os.path.join(PDF, "remediation", "cert-sg-irm", "**", "*.txt"), recursive=True)):
        m = re.match(r"IRM[-_ ]?(\d+)[-_ ]*(.*)", os.path.basename(path)[:-4])
        if not m:
            continue
        name = re.sub(r"(?<=[a-z])(?=[A-Z])|[-_]+", " ", m.group(2)).strip() or f"IRM {m.group(1)}"
        stage, buf = None, []

        def flush() -> None:
            body = clean(re.sub(r"(?<![.:;!?])\n(?=[a-z(])", " ", "\n".join(buf)))
            if stage and len(body) > 250:
                emit("irm_stage", "remediation", "B", pick(["Incident response for {n}: what are the {s} steps?", "We are handling a {n} incident. What should happen in the {s} phase?"], n=lower1(name), s=stage.lower()),
                     f"**{stage}** (CERT Societe Generale incident response methodology IRM-{m.group(1)}, {name}):\n\n{cut(body, 4200)}", source="cert-sg IRM-" + m.group(1), key=f"irm:{m.group(1)}", licence="CC BY 3.0")
        for ln in read(path).replace("\f", "\n").split("\n"):
            if IRM_STAGE.match(ln.strip()):
                flush()
                stage, buf = ln.strip().split()[0].capitalize() if not ln.strip().lower().startswith("lessons") else "Lessons learned", []
            elif stage:
                buf.append(ln.strip())
        flush()


STEPS_B = [("windows services and ports", windows_services), ("systemd units and settings", systemd_units), ("service monitoring, alert guides, triggers", monitoring_cards),
           ("nginx, apache, postgresql, debian", server_directives), ("ports and RFC sections", networking), ("kernel parameters and hardening lists", kernel_hardening),
           ("fuzzing book and testing guidance", testing_material), ("devtools protocol and playwright", browser_tooling), ("detection across query languages, CAR, D3FEND", detection_cards),
           ("NIST, CyBOK and response PDFs", reference_pdfs)]


if __name__ == "__main__":
    os.makedirs(bc.OUT, exist_ok=True)
    bc.out_fh = open(os.path.join(bc.OUT, "candidates_r2.jsonl"), "w", encoding="utf-8", newline="\n")
    print("building round-two candidates", flush=True)
    for label, fn in STEPS_A + STEPS_B:
        guard(label, fn)
    bc.out_fh.close()
    kinds = {k: v for k, v in bc.stats.items() if ":" not in k}
    json.dump({"kinds": dict(sorted(kinds.items(), key=lambda kv: -kv[1])), "dropped": {k: v for k, v in bc.stats.items() if ":" in k}, "errors": dict(bc.errors)},
              open(os.path.join(bc.OUT, "candidates_r2_stats.json"), "w", encoding="utf-8"), indent=1)
    print(f"\n{sum(kinds.values())} candidates in {len(kinds)} kinds -> data/candidates_r2.jsonl")
    for e in bc.errors:
        print("  ERROR", e)
