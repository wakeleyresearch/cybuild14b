"""Stage 2 of the training-set build: prune, score, and select the best candidates.

    python -I scripts/rank_dataset.py --target 60000

Reads data/candidates.jsonl (from build_candidates.py) and writes:
  data/ranked.jsonl   every surviving candidate with its score and whether it was selected (best first per topic)
  data/train.jsonl    the selected examples in chat format      data/eval.jsonl   a held-out slice of them
  data/report.json    what was pruned and why, and how each topic quota was filled

Scoring is rule-based: it measures source trust, concrete content and variety. It does not check that an answer is correct.
"""
from __future__ import annotations

import argparse
import collections
import functools
import json
import os
import random
import re
import sys
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SYSTEM = ("You are a cybersecurity and systems operations assistant. Give accurate, practical answers with the commands, queries, "
          "configuration or code that solve the problem. State assumptions, and say so when an answer depends on context you were not given.")

# share of the final set per topic (the subjects asked for get the room; machine-templated CVE text gets almost none)
QUOTA = {"remediation": 13, "services": 10, "tools": 10, "detection": 9, "logging": 7, "prioritization": 7, "web": 7, "powershell": 6,
         "kernel-memory": 6, "testing": 5, "defense": 4.5, "reasoning": 3.5, "networking": 3, "web-tools": 3, "general": 2.5, "lua": 2,
         "vuln-intel": 1, "legal": 0.5}
TIER = {"A": 1.00, "B": 0.92, "C": 0.78, "D": 0.35}
KIND_ADJ = {"tldr_task": 0.88, "man_option": 0.9, "prom_alert": 0.88, "sigma_explain": 0.9, "kali_usage": 0.9, "quickref": 0.85, "elastic_query": 0.92,
            "ssvc_priority": 1.05, "alert_response": 1.08, "windows_event": 1.05, "tool_card": 1.05, "command_explain": 0.85, "man_section": 0.92}
KEY_CAP = {"default": 3, "tool": 6, "service": 4, "sigma": 1, "mdn": 1, "ps": 1, "legal": 14,
           # round-two keys (build_round2.py): how many examples one log source, library, document or protocol domain may contribute
           "logsrc": 6, "grok": 6, "log_parsing_docs": 25, "nselib": 12, "luafn": 2, "luajit": 12, "lua_embedding": 150, "snortlua": 12, "nse_guide": 150,
           "nmap_reference": 40, "cdp": 25, "playwright_api": 12, "systemdopt": 12, "rfc": 12, "pdf": 40, "irm": 8, "admin_reference": 30, "kernel_sysctl": 120,
           "fuzzingbook": 8, "winsvc": 1, "port": 1, "ecs": 2, "sysmon": 2, "multi": 1, "toolopt": 40, "rosetta": 1, "zap": 4, "scenario": 1, "deadline": 1}
# kinds whose answers are short by nature (one command, one rule): no penalty for brevity
SHORT_OK = {"tldr_task", "man_option", "prom_alert", "osquery_table", "kev_entry", "artifact_location", "nse_script",
            "command_explain", "port_card", "artifact_collection", "ssvc_decision_point", "sysmon_rule", "windows_ports", "kernel_hardening_list", "lua_function",
            "lua_c_api", "cdp_command", "wazuh_rule", "auditd_rules", "grok_patterns", "systemd_setting"}
# kinds whose answers share a deliberate structure: compare question and answer together, or they all look like duplicates
STRUCTURED = {"ssvc_priority", "kev_entry", "hardening_check", "prom_alert", "tldr_task", "sigma_explain",
              "ssvc_table", "triage_ranking", "cvss_breakdown", "wazuh_rule", "port_card", "cdp_command", "windows_service", "windows_ports", "command_explain",
              "sysmon_rule", "kernel_hardening_list", "tool_tasks", "ssvc_scenario", "remediation_deadline"}
# computed examples: each covers a different case by construction but shares most of its wording with its siblings, so
# near-duplicate matching would discard distinct cases. Only exact repeats are removed; the ceilings below limit volume.
EXACT_ONLY = {"ssvc_priority", "ssvc_scenario", "remediation_deadline", "triage_ranking", "cvss_breakdown", "ssvc_table", "port_card", "windows_service",
              "windows_ports", "cdp_command", "kernel_hardening_list", "wazuh_rule", "sysmon_rule", "command_explain"}
# hard ceilings so a large or formulaic source cannot crowd a topic, whatever its scores
# (round two, at the user's request: more room for prioritization reasoning, tool tasks and log locations/setup, less for
#  restated records, single-option lookups and field dictionaries)
KIND_MAX = {"kev_entry": 250, "ssvc_priority": 1400, "hardening_check": 1500, "metasploit_module": 1200, "defense_qa": 3600, "security_misc": 900,
            "tldr_task": 1800, "man_option": 800, "sigma_write": 2200, "sigma_explain": 500, "prom_alert": 500, "elastic_query": 600, "mdn": 3600,
            "windows_debugging": 900, "cve_template": 300, "exploitdb_template": 150, "advisory_template": 100, "kali_usage": 500, "quickref": 150,
            "triage_ranking": 1800, "cvss_breakdown": 1400, "ssvc_table": 450, "ssvc_scenario": 2400, "remediation_deadline": 900, "lua_program": 900, "zap_help": 300, "command_explain": 900, "tool_tasks": 3600, "man_examples": 900, "man_section": 500,
            "port_card": 450, "rfc_section": 1200, "cdp_command": 500, "playwright_api": 300, "systemd_setting": 900, "nginx_directive": 450, "apache_directive": 450,
            "postgres_setting": 300, "admin_reference": 600, "nist_guidance": 1400, "cybok": 500, "kernel_param": 700, "kernel_sysctl": 500, "fuzzing_book": 400,
            "nse_function": 900, "lua_c_api": 350, "velociraptor_artifact": 350, "artifact_collection": 600, "sysmon_rule": 600, "log_field_dictionary": 200}
# Inside a topic, room per kind relative to its size (round two, from the user's priorities). Logging: where the logs are,
# then switching logging on quickly, then acting on what the logs show; schemas, field lists and parser listings last.
# Prioritization: the computed reasoning kinds over restated records. Tools: tasks and worked examples over one-option lookups.
KIND_WEIGHT = {
    "artifact_location": 2.0, "artifact_collection": 1.8, "log_source_card": 2.0, "log_path_card": 2.0, "log_defaults": 1.6, "velociraptor_artifact": 1.2,
    "sysmon_rule": 1.5, "audit_policy": 1.8, "auditd_rules": 1.8, "logging_setup": 2.0, "systemd_setting": 1.2,
    "log_alert_conditions": 1.8, "wazuh_log_test": 1.8, "wazuh_rule": 1.4, "event_log_rule": 1.3, "windows_event": 1.3,
    "log_field_dictionary": 0.5, "ecs_fields": 0.5, "ocsf_class": 0.5, "asim_parser": 0.6, "grok_patterns": 0.6, "log_parsing_docs": 0.7,
    "log_decoder": 0.8, "windows_event_fields": 0.8, "logging_docs": 0.8,
    "ssvc_scenario": 1.6, "triage_ranking": 1.6, "cvss_breakdown": 1.4, "ssvc_table": 1.4, "remediation_deadline": 1.2, "ssvc_priority": 0.6, "kev_entry": 0.5,
    "tool_tasks": 1.5, "man_examples": 1.5, "tool_card": 1.5, "man_option": 0.5, "nmap_option": 0.7,
}
# how much of a topic one kind may take; topics fed by few sources need a looser limit to fill at all
KIND_SHARE = {"web": 0.85, "logging": 0.7, "lua": 0.9, "legal": 1.0, "prioritization": 0.75, "networking": 0.85, "web-tools": 0.7, "powershell": 0.6, "general": 0.6}
NO_REDISTRIBUTION = {"defense", "reasoning", "vuln-intel", "general"}   # these keep their quota; spare room goes to the requested subjects
# kinds that combine sources or carry remediation and "what indicates a problem" logic: taken ahead of plain reference text
PREMIUM = {"tool_card", "ssvc_priority", "attack_technique", "cwe_remediation", "windows_event", "alert_response", "alert_runbook", "service_indicators",
           "powershell_ise", "audit_policy", "linux_hardening", "capec_defence", "artifact_location", "lolbin", "codeql_fix", "incident_response",
           "kernel_hardening", "scoring_spec", "prioritization_docs", "windows_troubleshooting", "nse_script", "lua", "legal", "windows_command",
           # round two (build_round2.py)
           "log_source_card", "log_alert_conditions", "log_decoder", "wazuh_rule", "windows_event_fields", "grok_patterns", "asim_parser", "artifact_collection",
           "auditd_rules", "logging_guidance", "lua_function", "lua_embedding", "lua_config", "nse_library", "nse_function", "nse_script_source", "nse_guide",
           "tool_tasks", "man_examples", "nmap_option", "ssvc_table", "ssvc_decision_point", "triage_ranking", "cvss_breakdown", "prioritization_guidance",
           "windows_service", "windows_ports", "systemd_unit", "systemd_setting", "service_monitoring", "alert_guide", "service_problem_triggers",
           "detection_multi", "detection_analytic", "d3fend_technique", "kernel_param", "kernel_hardening_list", "irm_stage", "ssvc_scenario",
           "remediation_deadline", "lua_program"}
REFUSAL = re.compile(r"^\s*(i'?m sorry|i am sorry|i cannot|i can'?t (help|assist)|as an ai\b|sorry, but)", re.I)
STALE = re.compile(r"(September 30, 2026|December 11, 2026|January 30, 2026)")
WORD = re.compile(r"[A-Za-z0-9_./\\:-]+")


def features(q: str, a: str) -> dict:
    words = WORD.findall(a)
    head = words[:400]
    lines = a.split("\n")
    nonspace = [c for c in a[:3000] if not c.isspace()]
    return {
        "len": len(a),
        "lex": len(set(w.lower() for w in head)) / max(1, len(head)),
        "ascii": sum(c.isascii() for c in nonspace) / max(1, len(nonspace)),
        "code": a.count("```") // 2 + min(6, a.count("`") // 6),
        "steps": sum(1 for ln in lines if re.match(r"\s*(?:[-*]|\d+[.)])\s+\S", ln)),
        "specific": len(re.findall(r"(?:\b[A-Z]{2,}-\d+\b|\bT\d{4}\b|--?[a-z][\w-]+|[A-Za-z]:\\|/etc/|/var/|HKEY_|\b\d{3,5}\b)", a[:4000])),
        "repeat": 1 - len(set(ln.strip() for ln in lines if ln.strip())) / max(1, sum(1 for ln in lines if ln.strip())),
        "ends_clean": a.rstrip()[-1:] in ".!?`)]}|:*>\"'" or a.rstrip().endswith("```"),
    }


def score(c: dict, f: dict) -> float:
    s = TIER.get(c["tier"], 0.5) * KIND_ADJ.get(c["kind"], 1.0)
    if c.get("joined"):
        s *= 1.15
    n = f["len"]
    if c["kind"] not in SHORT_OK:
        s *= 0.6 if n < 200 else 0.85 if n < 400 else 1.0 if n <= 3500 else 0.93
    s *= 1 + 0.10 * min(f["code"], 4) / 4 + 0.05 * min(f["steps"], 8) / 8 + 0.05 * min(f["specific"], 10) / 10
    s *= 0.90 + 0.10 * min(1.0, f["lex"] / 0.6)
    if not f["ends_clean"]:
        s *= 0.88
    return round(s, 4)


def prune_reason(c: dict, f: dict) -> str | None:
    if REFUSAL.match(c["a"]):
        return "refusal or boilerplate"
    if f["ascii"] < 0.85:
        return "mostly non-English or binary text"
    if f["len"] > 500 and f["lex"] < 0.22:
        return "repetitive text"
    if f["repeat"] > 0.5 and f["len"] > 400:
        return "repeated lines"
    if c["topic"] == "legal" and STALE.search(c["a"]):
        return "time-sensitive legal status (belongs in lookup, not training)"
    if re.search(r"(lorem ipsum|TODO: write|coming soon)", c["a"], re.I):
        return "placeholder text"
    if c["a"].count("```") % 2:
        return "broken code block (cut from the middle of an example)"
    body = re.sub(r"```.*?```", "", c["a"], flags=re.S)
    if re.search(r"#!/(?:usr/)?bin/(?:env )?\w+ [^\n]{250,}", body):
        return "script flattened onto one line"
    if re.search(r"\{%[^%\n]{1,80}%\}", body):
        return "site template tag left in the text"
    return None


TAB_TAG = re.compile(r"\{%\s*tab\s+title=\"([^\"]+)\"\s*%\}")
SITE_TAG = re.compile(r"\{%[^%\n]{0,300}%\}")


def tidy(c: dict) -> str:
    """Remove site-generator debris the converters left in an answer, keeping the text it wrapped. Code blocks are not touched.

    Documentation sites wrap content in tags such as {% hint %}...{% endhint %} or {% tab title="x" %}; some sources carry
    unfilled placeholders ({{Fill ... Description}}) or book-index markers ({{index ...}}).
    """
    a = c["a"]
    braces = "{%" in a or "{{" in a
    tags = "<" in a and c["topic"] not in ("web", "web-tools")      # in web topics an HTML tag is usually the subject, not debris
    if not braces and not tags:
        return a
    parts = a.split("```")
    for i in range(0, len(parts), 2):                  # even pieces are prose, odd pieces are fenced code
        p = parts[i]
        if tags:
            p = re.sub(r"<br ?/?>|</?p>", "\n", re.sub(r"</?(?:code|tt|kbd)>", "`", p))
        if braces:
            p = SITE_TAG.sub("", TAB_TAG.sub(lambda m: f"**{m.group(1)}:**", p))
        if c["kind"] == "powershell_cmdlet":
            p = re.sub(r"\{\{\s*Fill [^{}\n]*\}\}", "Not documented.", p)
        elif c["kind"] == "javascript_book":
            p = re.sub(r"^\{\{figure[^\n]*\n?", "", p, flags=re.M)
            p = re.sub(r"\{\{(?:index|id|meta|docid)\b[^\n]*?\}\}\n?", "", p)
            p = re.sub(r"^(?:\{\{\w+[^\n{}]*|\w+\}\})[ \t]*$\n?", "", p, flags=re.M)
        parts[i] = p
    return re.sub(r"\n{3,}", "\n\n", "```".join(parts)).strip()


def sketch(text: str, k: int = 8) -> tuple[int, ...]:
    """MinHash over 5-word shingles: near-identical texts share most of these values."""
    w = [x.lower() for x in WORD.findall(text)[:500]]
    if len(w) < 12:
        return tuple()
    hs = sorted({zlib.crc32(" ".join(w[i:i + 5]).encode()) for i in range(len(w) - 4)})
    return tuple(hs[:k])


def template_print(a: str) -> int:
    """Same text once numbers and identifiers are masked: catches machine-templated answers."""
    t = re.sub(r"\s+", " ", re.sub(r"[0-9a-f]{6,}|\d+", "#", a.lower()))
    mid = len(t) // 2
    return zlib.crc32((t[:140] + "|" + t[mid:mid + 80]).encode())


# ------------------------------------------------------------------ questions must say which document they mean
# build_candidates.docs() asks about a document section as product + "(title)" + heading. In some collections the
# "title" it found is a section name shared by every document ("this Metasploit module (Vulnerable Application)",
# "Zabbix monitoring (Overview)"), so one question has hundreds of different answers. Those collections are renamed
# from the file itself; any question that still has more than one answer is dropped, in every kind.
SOURCES = os.path.join(ROOT, "sources")
DOC_Q = ["Explain \"{h}\" in the context of {ctx}.", "{ctx}: what do I need to know about \"{h}\"?",       # build_candidates.DOC_Q
         "Walk me through \"{h}\" ({ctx}).", "I'm working with {ctx}. Can you cover \"{h}\"?",
         "What does {ctx} say about \"{h}\"? Give me the practical details."]
DOC_RX = [re.compile("^" + re.escape(t).replace(r"\{h\}", "(?P<h>.+)").replace(r"\{ctx\}", "(?P<ctx>.+)") + "$", re.S) for t in DOC_Q]
GENERIC_TITLES = {" ".join(g.split()) for g in """overview, introduction, intro, usage, example, examples, options, syntax, notes,
    description, parameters, arguments, summary, details, configuration, setup, requirements, installation, getting started,
    quick start, readme, index, see also, references, about, general, background, contents, purpose, synopsis, vulnerable application,
    description from att ck, verification steps, options, scenarios""".replace("\n", " ").split(",")}
MASTG_WHAT = {"techniques": "technique", "tests": "test", "tests-beta": "test", "tools": "tool", "demos": "demo", "apps": "app",
              "best-practices": "best practice", "knowledge": "article"}


def norm(s: str | None) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", (s or "").casefold()).split())


def answer_id(a: str) -> int:
    """Identity of an answer: the same text once case, punctuation and spacing are ignored."""
    return zlib.crc32(norm(a)[:300].encode())


def text_id(*parts: str) -> int:
    s = "\x00".join(parts).encode()
    return zlib.crc32(s) << 32 | zlib.adler32(s)


def doc_parts(q: str) -> tuple[int, str, str] | None:
    """(template number, context, heading) for a question made from a document section, else None."""
    for i, rx in enumerate(DOC_RX):
        m = rx.match(q.strip())
        if m:
            return i, m.group("ctx"), m.group("h")
    return None


def meaning_id(q: str) -> int:
    """Questions that ask the same thing in different wording share this: a section question is its context and heading."""
    p = doc_parts(q)
    return text_id("doc", norm(p[1]), norm(p[2])) if p else text_id("q", norm(q))


def split_ctx(ctx: str) -> tuple[str, str]:
    """"rsyslog (MaxFrameSize)" -> ("rsyslog", "MaxFrameSize")."""
    if ctx.endswith(")") and " (" in ctx:
        depth = 0
        for i in range(len(ctx) - 1, -1, -1):
            depth += (ctx[i] == ")") - (ctx[i] == "(")
            if depth == 0:
                return ctx[:i].rstrip(), ctx[i + 1:-1]
    return ctx, ""


@functools.lru_cache(maxsize=None)
def _file_head(rel: str) -> tuple[str, dict]:
    """(first '# ' heading, front-matter fields) of a source file, from its first 4 KB."""
    try:
        with open(os.path.join(SOURCES, rel), encoding="utf-8", errors="replace") as fh:
            head = fh.read(4000)
    except OSError:
        return "", {}
    fm = {}
    m = re.match(r"\s*---\r?\n(.*?)\r?\n---", head, re.S)
    if m:
        for ln in m.group(1).splitlines():
            k, sep, v = ln.partition(":")
            if sep and re.fullmatch(r"[A-Za-z_]+", k.strip()):
                fm[k.strip()] = v.strip().strip("'\"")
        head = head[m.end():]
    h1 = re.search(r"^#[ \t]+(.+?)[ \t#]*$", head, re.M)
    return (h1.group(1).strip() if h1 else ""), fm


def document_name(src: str) -> tuple[str, bool] | None:
    """(name, keep the section title after it) for a document in a collection whose section titles repeat, else None."""
    m = re.search(r"metasploit-framework/documentation/modules/(.+)\.md$", src)
    if m:
        return f"the Metasploit module {m.group(1)}", False
    m = re.search(r"atomic-red-team/atomics/(T\d{4}(?:\.\d{3})?)/\1\.md$", src)
    if m:
        t = re.match(rf"{re.escape(m.group(1))}\s*-\s*(.+)$", _file_head(src)[0])     # "T1003.001 - OS Credential Dumping: LSASS Memory"
        return f"Atomic Red Team {m.group(1)}" + (f" ({t.group(1)})" if t else ""), False
    m = re.search(r"zabbix-templates/templates/.+/([^/]+)/README\.md$", src)
    if m:
        h1 = _file_head(src)[0]
        return f"the Zabbix template {h1 if h1 and norm(h1) not in GENERIC_TITLES else m.group(1)}", False
    m = re.search(r"owasp-mastg/([a-z-]+)/(?:[^/]+/)*(MAS[A-Z]*-[A-Z]+-\d+)\.md$", src)
    if m:
        fm = _file_head(src)[1]
        plat = {"android": "Android", "ios": "iOS"}.get((fm.get("platform") or "").lower(), "")
        bits = ", ".join(b for b in (fm.get("title", ""), plat) if b)
        return f"the OWASP MASTG {MASTG_WHAT.get(m.group(1), 'page')} {m.group(2)}" + (f" ({bits})" if bits else ""), False
    m = re.search(r"kernel-memory/kspp/(?:[^/]+/)*([^/]+)\.md$", src)
    if m and m.group(1) != "index":
        return f"the Kernel Self Protection Project ({m.group(1).replace('_', ' ')})", False
    m = re.search(r"rsyslog-docs/doc/source/(?:reference/parameters/([a-z0-9]+)-[^/]+|configuration/modules/([a-z0-9]+))\.rst$", src)
    if m:
        return f"rsyslog's {m.group(1) or m.group(2)} module", True
    return None


def disambiguate(q: str, src: str) -> str:
    """The question, renamed to say which document it means when it comes from one of the collections above."""
    p = doc_parts(q)
    named = document_name(src or "") if p else None
    if not named:
        return q
    i, ctx, h = p
    name, keep = named
    title = split_ctx(ctx)[1]
    if keep and title and norm(title) not in GENERIC_TITLES:
        name += f" ({title})"
    return DOC_Q[i].format(h=h, ctx=name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=60000, help="number of examples to select")
    ap.add_argument("--eval-frac", type=float, default=0.015)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--inputs", nargs="+", default=["candidates.jsonl"], help="candidate files under data/, read in this order")
    ap.add_argument("--out-dir", default=DATA, help="where to write the results (default data/; use data/r2 for a second round)")
    ap.add_argument("--holdout", help="an eval.jsonl from an earlier round: its questions are never trained on, and stay in the eval set")
    ap.add_argument("--trained", help="a train.jsonl from an earlier round: selected examples already in it go to train_replay.jsonl, the rest to train_new.jsonl")
    ap.add_argument("--loss", help="a per-topic score file from train_lora.py --eval-only: topics the model is weaker on get more room")
    ap.add_argument("--gamma", type=float, default=1.25, help="how strongly --loss shifts the topic shares (0 = not at all)")
    ap.add_argument("--boost", nargs="*", default=[], metavar="TOPIC=FACTOR", help="multiply a topic's base share, e.g. --boost lua=1.75 logging=1.2")
    ap.add_argument("--replay-frac", type=float, default=0.3, help="with --trained: also write train_round2.jsonl = the new examples plus this "
                    "fraction as many already-trained ones, so a second round refreshes earlier material instead of drifting from it")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    def identities(path):
        """An earlier round's rows by (question, source) -> the answers they gave; and every answer in the file.
        Question text alone is not an identity: one question can carry different answers from different files."""
        ids, answers = collections.defaultdict(set), set()
        if path:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    aid = answer_id(r["messages"][-1]["content"])
                    ids[text_id(r["messages"][-2]["content"], (r.get("meta") or {}).get("source") or "")].add(aid)
                    answers.add(aid)
        return ids, answers

    held_ids, held_answers = identities(args.holdout)
    trained_ids = set(identities(args.trained)[0])
    share = dict(QUOTA)
    for item in args.boost:
        t, _, f = item.partition("=")
        share[t] = share.get(t, 1) * float(f)
    if args.loss:       # importance (the base share) times weakness (loss relative to the average)
        by_t = json.load(open(args.loss, encoding="utf-8"))["by_topic"]
        mean = sum(v["loss"] for v in by_t.values()) / len(by_t)
        for t in share:
            if t in by_t:
                share[t] *= (by_t[t]["loss"] / mean) ** args.gamma

    def candidate_lines():
        for name in args.inputs:
            with open(os.path.join(DATA, name), encoding="utf-8") as fh:
                yield from fh

    pruned = collections.Counter()
    rows = []  # (score, idx, kind, topic, key, tier, sketch, is a test question, meaning id, answer id)
    seen_print: dict[int, int] = {}
    held_pick: dict[int, tuple[int, bool]] = {}     # a test row's (question, source) -> (candidate index, its answer matched)
    total = 0
    if True:
        for idx, line in enumerate(candidate_lines()):
            c = json.loads(line)
            total += 1
            c["a"] = tidy(c)
            if c["kind"] == "mdn" and c["topic"] == "networking":
                c["kind"] = "mdn_http"       # HTTP pages count toward networking, not the web-platform ceiling
            f = features(c["q"], c["a"])
            why = prune_reason(c, f)
            if why:
                pruned[why] += 1
                continue
            ident, aid = text_id(c["q"], c.get("source") or ""), answer_id(c["a"])
            if ident in held_ids:         # a test question keeps its wording; prefer the candidate whose answer matches
                matched = aid in held_ids[ident]
                if ident not in held_pick or (matched and not held_pick[ident][1]):
                    held_pick[ident] = (idx, matched)
                q2 = c["q"]
            else:
                q2 = disambiguate(c["q"], c.get("source") or "")
            sc = score(c, f)
            tp = zlib.crc32(c["kind"].encode()) ^ template_print(c["a"])
            if c["tier"] in ("C", "D") or c["kind"] in ("tldr_task", "man_option"):
                if tp in seen_print:
                    pruned["same answer template as another example"] += 1
                    continue
                seen_print[tp] = idx
            if c["kind"] in EXACT_ONLY:
                sk = tuple(zlib.crc32(f"{i}:{c['q']}\x00{c['a']}".encode()) for i in range(8))
            else:
                sk = sketch(c["q"] + " " + c["a"] if c["kind"] in STRUCTURED else c["a"])
            # kind, topic and tier repeat across hundreds of thousands of rows: share one string object each to keep memory down
            rows.append((sc, idx, sys.intern(c["kind"]), sys.intern(c["topic"]), c["key"], sys.intern(c["tier"]), sk, False, meaning_id(q2), aid))
            if total % 100000 == 0:
                print(f"  scored {total}", flush=True)
    held_idx = {i for i, _ in held_pick.values()}
    rows = [r[:7] + (r[1] in held_idx,) + r[8:] for r in rows]
    if args.holdout:
        print(f"test rows found among the candidates: {len(held_pick)} of {len(held_ids)} "
              f"({sum(m for _, m in held_pick.values())} by question, source and answer; the rest by question and source)", flush=True)

    # A question with more than one answer does not say which document, version or entry it means; training on it teaches
    # the model to answer an underspecified question with one arbitrary specific. Drop every such example (test questions
    # stay where they are), unless the answers are near-identical copies, which near-duplicate removal settles below.
    first: dict[int, int] = {}
    multi: dict[int, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r[8] in first:
            multi[r[8]].append(i)
        else:
            first[r[8]] = i
    drop: set[int] = set()
    for m, others in multi.items():
        g = [first[m]] + others
        if any(rows[x][9] != rows[y][9] and not (rows[x][6] and rows[y][6] and len(set(rows[x][6]) & set(rows[y][6])) >= 6)
               for n, x in enumerate(g) for y in g[n + 1:]):
            gone = [i for i in g if not rows[i][7]]
            drop.update(gone)
            pruned["question has more than one answer (does not say which document or entry it means)"] += len(gone)
    del first, multi
    for i, r in enumerate(rows):                 # a test answer is never trained on, whatever question carries it
        if not r[7] and r[9] in held_answers and i not in drop:
            drop.add(i)
            pruned["carries the answer of a test question"] += 1
    rows = [r for i, r in enumerate(rows) if i not in drop]
    del drop
    print(f"candidates: {total}; pruned outright: {sum(pruned.values())}; scored: {len(rows)}", flush=True)

    # Inside a topic, give more room to the kinds of example the model is weaker on, when the score file breaks loss down
    # by kind. Half the strength of the topic-level shift, at least 5 scored examples per kind, and bounded, because a
    # handful of test questions per kind is a noisy measurement.
    kind_adj: dict[str, float] = {}
    if args.loss:
        scores = json.load(open(args.loss, encoding="utf-8"))
        topic_of = {}
        for r in rows:
            topic_of.setdefault(r[2], r[3])
        for k, v in (scores.get("by_kind") or {}).items():
            base = (scores.get("by_topic") or {}).get(topic_of.get(k, ""), {}).get("loss")
            if base and v.get("n", 0) >= 5:
                kind_adj[k] = round(min(1.6, max(0.6, (v["loss"] / base) ** (args.gamma * 0.5))), 3)

    # near-duplicate removal: keep the higher-scoring of any two answers sharing 6 of 8 sketch values.
    # Earlier test questions go first, so a reworded copy of a test answer is dropped instead of being trained on.
    rows.sort(key=lambda r: (not r[7], -r[0]))
    bands: dict[tuple, list[int]] = collections.defaultdict(list)
    kept_sketch: dict[int, tuple] = {}
    survivors = []
    for r in rows:
        sk = r[6]
        dup = False
        if sk:
            for b in range(0, 8, 2):
                for other in bands.get((b, sk[b:b + 2]), ()):
                    if len(set(sk) & set(kept_sketch[other])) >= 6:
                        dup = True
                        break
                if dup:
                    break
        if dup:
            pruned["near-duplicate of a better example"] += 1
            continue
        if sk:
            kept_sketch[r[1]] = sk
            for b in range(0, 8, 2):
                bands[(b, sk[b:b + 2])].append(r[1])
        survivors.append(r)
    survivors.sort(key=lambda r: -r[0])
    print(f"after near-duplicate removal: {len(survivors)} ({sum(r[7] for r in survivors)} are earlier test questions, kept out of training)", flush=True)

    # quota-based selection, best first, with caps so no single tool/rule/page or converter dominates a topic
    by_topic = collections.defaultdict(list)
    for r in survivors:
        if not r[7]:
            by_topic[r[3]].append(r)
    weight = sum(share.get(t, 1) for t in by_topic)
    selected: set[int] = set()
    fill = {}

    scale = args.target / 60000          # the ceilings above are written for a 60,000-example set

    def take(topic: str, want: int) -> int:
        """Split the topic's room across its kinds by the square root of what each has (so a big source cannot swamp a
        small one), weight combined and remediation-focused kinds up, then take the best-scoring examples of each kind."""
        rows_t = by_topic[topic]
        have = collections.Counter(r[2] for r in rows_t)
        w = {k: (n ** 0.5) * (3.0 if k in PREMIUM else 1.0) * kind_adj.get(k, 1.0) * KIND_WEIGHT.get(k, 1.0) for k, n in have.items()}
        cap = {k: max(1, min(n, int(KIND_MAX.get(k, 10**9) * scale), int(want * KIND_SHARE.get(topic, 0.45)) + 1)) for k, n in have.items()}
        alloc = dict.fromkeys(have, 0)
        left, active = want, set(have)
        for _ in range(8):
            if left <= 0 or not active:
                break
            tw = sum(w[k] for k in active)
            for k in sorted(active):
                give = min(cap[k] - alloc[k], int(left * w[k] / tw) + 1)
                alloc[k] += give
            left = want - sum(alloc.values())
            active = {k for k in active if alloc[k] < cap[k]}
        per_key, per_kind = collections.Counter(), collections.Counter()
        for r in rows_t:
            if r[1] in selected:
                per_key[r[4]] += 1
                per_kind[r[2]] += 1
        for r in rows_t:                      # rows are already best-first
            if r[1] in selected or per_kind[r[2]] >= alloc[r[2]] or sum(per_kind.values()) >= want:
                continue
            prefix = r[4].split(":", 1)[0] if ":" in r[4] else "default"
            if r[4] and per_key[r[4]] >= KEY_CAP.get(prefix, KEY_CAP["default"]):
                continue
            selected.add(r[1])
            per_key[r[4]] += 1
            per_kind[r[2]] += 1
        return sum(per_kind.values())

    for t in by_topic:
        want = int(args.target * share.get(t, 1) / weight)
        fill[t] = {"quota": want, "available": len(by_topic[t]), "selected": take(t, want)}
    for _ in range(4):  # topics that could not fill their quota hand the room to the requested subjects that can
        short = args.target - len(selected)
        if short <= 0:
            break
        grew = False
        room = {t: fill[t]["available"] - fill[t]["selected"] for t in by_topic if t not in NO_REDISTRIBUTION}
        spare = sum(share.get(t, 1) for t, r in room.items() if r > 0) or 1
        for t, r in room.items():
            if r > 0:
                before = fill[t]["selected"]
                fill[t]["selected"] = take(t, before + min(r, int(short * share.get(t, 1) / spare) + 1))
                grew = grew or fill[t]["selected"] > before
        if not grew:
            break
    print(f"selected: {len(selected)} of target {args.target}", flush=True)

    # write everything out
    meta = {r[1]: r for r in survivors}
    rank_in_topic = {}
    for t, lst in by_topic.items():
        for i, r in enumerate(lst):
            rank_in_topic[r[1]] = i + 1
    eval_ids = set()
    by_t_sel = collections.defaultdict(list)
    for i in selected:
        by_t_sel[meta[i][3]].append(i)
    if args.eval_frac > 0:         # 0: the test set is exactly --holdout (a frozen set the model has already been scored on)
        for t, ids in by_t_sel.items():
            rng.shuffle(ids)
            eval_ids.update(ids[:max(8, int(len(ids) * args.eval_frac))])

    counts = collections.Counter()
    out = lambda name: open(os.path.join(args.out_dir, name), "w", encoding="utf-8", newline="\n")  # noqa: E731
    second_round = bool(args.trained)
    with out("ranked.jsonl") as ranked, out("train_new.jsonl" if second_round else "train.jsonl") as train, out("eval.jsonl") as ev, \
            (out("train_replay.jsonl") if second_round else open(os.devnull, "w")) as replay:
        for idx, line in enumerate(candidate_lines()):
            if idx not in meta:
                continue
            c = json.loads(line)
            c["a"] = tidy(c)
            if c["kind"] == "mdn" and c["topic"] == "networking":
                c["kind"] = "mdn_http"
            sc = meta[idx][0]
            sel = idx in selected
            held = idx in held_idx
            asked = c["q"]
            if not held:
                c["q"] = disambiguate(asked, c.get("source") or "")
            # trained in an earlier round with this same wording; a renamed question is new material
            trained = text_id(asked, c.get("source") or "") in trained_ids and c["q"] == asked
            ranked.write(json.dumps({"score": sc, "topic_rank": rank_in_topic.get(idx, 0), "selected": sel, **{k: c[k] for k in ("topic", "kind", "tier", "joined", "key", "source", "licence", "q", "a")}},
                                    ensure_ascii=False) + "\n")
            if held:                              # an earlier round's test question: keep it as a test question whatever its rank
                sel, is_eval = True, True
            else:
                is_eval = idx in eval_ids and not trained
            if sel:
                msgs = ([{"role": "system", "content": SYSTEM}] if c.get("system", True) else []) + \
                    [{"role": "user", "content": c["q"]}, {"role": "assistant", "content": c["a"]}]
                dest = ev if is_eval else replay if trained else train
                dest.write(json.dumps({"messages": msgs, "meta": {"topic": c["topic"], "kind": c["kind"], "tier": c["tier"],
                                                                 "source": c["source"], "score": sc}}, ensure_ascii=False) + "\n")
                counts["eval" if is_eval else "replay" if trained else "train"] += 1
                counts["renamed"] += c["q"] != asked
                counts["kind:" + c["kind"]] += 1
                counts["tier:" + c["tier"]] += 1
                counts["joined"] += bool(c.get("joined"))
                counts["chars"] += len(c["q"]) + len(c["a"])
    mix = None
    if second_round:        # the file to train a second round on: everything new, plus a random refresher from the earlier round
        with open(os.path.join(args.out_dir, "train_new.jsonl"), encoding="utf-8") as fh:
            # one JSON object per "\n"-terminated line. Not str.splitlines(): that also breaks on characters such as
            # U+2028 and U+0085, which can sit unescaped inside the text and would cut an example in two.
            lines = [line.rstrip("\n") for line in fh]
        keep = set(rng.sample(range(counts["replay"]), min(counts["replay"], int(len(lines) * args.replay_frac))))
        with open(os.path.join(args.out_dir, "train_replay.jsonl"), encoding="utf-8") as fh:
            refresher = [line.rstrip("\n") for i, line in enumerate(fh) if i in keep]
        mix = {"new": len(lines), "refresher_from_earlier_round": len(refresher),
               "approx_tokens_millions": round(sum(len(x) for x in lines + refresher) / 3.9 / 1e6, 1)}
        lines += refresher
        rng.shuffle(lines)
        with out("train_round2.jsonl") as fh:
            fh.write("\n".join(lines) + "\n")
    report = {"candidates": total, "pruned": dict(pruned.most_common()), "after_pruning": len(survivors), "selected": len(selected), "round2_training_file": mix,
              "train": counts["train"], "already_trained_in_earlier_round": counts["replay"], "eval": counts["eval"], "joined_examples": counts["joined"],
              "renamed_to_name_their_document": counts["renamed"],
              "topic_shares_used": {t: round(v / sum(share.values()), 4) for t, v in sorted(share.items(), key=lambda kv: -kv[1])},
              "kind_weights_from_scores": dict(sorted(kind_adj.items(), key=lambda kv: -kv[1])),
              "approx_tokens_millions": round(counts["chars"] / 3.7 / 1e6, 1),
              "by_tier": {k[5:]: v for k, v in counts.items() if k.startswith("tier:")},
              "by_topic": dict(sorted(fill.items(), key=lambda kv: -kv[1]["selected"])),
              "by_kind": dict(sorted(((k[5:], v) for k, v in counts.items() if k.startswith("kind:")), key=lambda kv: -kv[1]))}
    json.dump(report, open(os.path.join(args.out_dir, "report.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: report[k] for k in ("candidates", "after_pruning", "selected", "train", "already_trained_in_earlier_round", "eval", "joined_examples",
                                             "approx_tokens_millions", "by_tier")}, indent=1))
    try:                    # how much memory this run needed, for deciding whether it can run beside a training job
        import psutil
        m = psutil.Process().memory_info()
        print(f"peak memory: {getattr(m, 'peak_wset', m.rss) / 2**30:.2f} GiB resident, {getattr(m, 'peak_pagefile', m.vms) / 2**30:.2f} GiB committed")
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
