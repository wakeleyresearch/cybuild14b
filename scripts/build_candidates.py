"""Stage 1 of the training-set build: turn the dataset and the collected sources into candidate examples.

    python -I scripts/build_candidates.py            # writes data/candidates.jsonl and data/candidates_stats.json

Every candidate is one question/answer pair with metadata used later for pruning and ranking:
  kind    which converter made it          topic   which subject quota it counts toward
  tier    source trust: A official/primary, B curated community, C written Q&A datasets, D machine-templated
  key     the entity it is about (tool, CVE, technique, event ID ...), used to limit repeats
  joined  True when it combines several sources about the same entity
Nothing from the sources is executed; YAML is read with safe_load only. Stage 2 is rank_dataset.py.
"""
from __future__ import annotations

import collections
import csv
import glob
import gzip
import hashlib
import html
import json
import os
import random
import re
import sys
import tomllib
import xml.etree.ElementTree as ET

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = os.path.join(ROOT, "sources")
DS = os.path.join(ROOT, "cyber-security")
OUT = os.path.join(ROOT, "data")
MAX_Q, MAX_A = 2600, 6000
rng = random.Random(20261123)
out_fh = None
stats = collections.Counter()
errors = collections.Counter()
seen_hash: set[int] = set()


def G(pattern: str) -> list[str]:
    # isfile: some repositories contain folders whose names end in .md
    return sorted(p for p in glob.glob(os.path.join(S, pattern), recursive=True) if os.path.isfile(p))


def read(path: str, limit: int = 2_000_000) -> str:
    with open(path, "rb") as fh:
        raw = fh.read(limit)
    return raw.decode("utf-8", "replace").replace("\r\n", "\n").lstrip("﻿")


def clean(text: str) -> str:
    text = html.unescape(re.sub(r"<!--.*?-->", "", text, flags=re.S))
    text = re.sub(r"\{\{[<%].*?[%>]\}\}", "", text, flags=re.S)       # hugo shortcodes
    text = re.sub(r"[ \t]+\n", "\n", text.replace("\xa0", " "))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def cut(text: str, limit: int = MAX_A) -> str:
    """Trim to the limit at a paragraph (or sentence) boundary; never mid-line."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    if head.count("```") % 2:                       # do not end inside a code block
        head = head[:head.rfind("```")]
    p = head.rfind("\n\n")
    if p > limit * 0.5:
        return head[:p].rstrip()
    p = max(head.rfind(". "), head.rfind(".\n"))
    return head[:p + 1].rstrip() if p > limit * 0.4 else head[:head.rfind("\n")].rstrip()


def emit(kind: str, topic: str, tier: str, q: str, a: str, *, source: str, key: str = "", joined: bool = False,
         system: bool = True, min_a: int = 150, licence: str = "") -> None:
    q, a = clean(q), cut(clean(a))
    if len(q) < 12 or len(q) > MAX_Q or len(a) < min_a:
        stats[f"dropped:{kind}"] += 1
        return
    h = int.from_bytes(hashlib.blake2b((q + "\x00" + a).encode("utf-8", "replace"), digest_size=8).digest(), "big")
    if h in seen_hash:
        stats[f"dup:{kind}"] += 1
        return
    seen_hash.add(h)
    out_fh.write(json.dumps({"kind": kind, "topic": topic, "tier": tier, "key": key, "joined": joined, "system": system,
                             "source": source, "licence": licence, "q": q, "a": a}, ensure_ascii=False) + "\n")
    stats[kind] += 1


def pick(templates: list[str], **kw) -> str:
    return rng.choice(templates).format(**kw)


def guard(label: str, fn, *args) -> None:
    """Run one converter; a broken source must not stop the build."""
    before = sum(v for k, v in stats.items() if ":" not in k)
    try:
        fn(*args)
    except Exception as e:  # noqa: BLE001
        errors[f"{label}: {type(e).__name__}: {str(e)[:120]}"] += 1
    print(f"  {label:34s} +{sum(v for k, v in stats.items() if ':' not in k) - before}", flush=True)


# ------------------------------------------------------------------ generic: documents split by headings
SKIP_HEAD = re.compile(r"^(see also|references?|further reading|license|licence|contribut|changelog|table of contents|contents|"
                       r"credits|acknowledg|authors?|related (links|articles|content)|next steps|feedback|additional resources|"
                       r"external links|specifications|browser compatibility|index|todo|notes?)\b", re.I)
RST_ADORN = re.compile(r"^([=\-~^\"'`#*+_.:])\1{2,}\s*$")


def front_matter(text: str) -> tuple[dict, str]:
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end > 0:
            try:
                meta = yaml.safe_load(text[4:end]) or {}
            except Exception:  # noqa: BLE001
                meta = {}
            return (meta if isinstance(meta, dict) else {}), text[end + 4:].lstrip("\n")
    return {}, text


def sections(text: str) -> list[tuple[str, str]]:
    """[(heading, body)] for markdown (# ..), asciidoc (== ..) and reStructuredText (underlined) documents."""
    lines = text.split("\n")
    out, head, buf, in_code = [], "", [], False
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.lstrip().startswith(("```", "~~~")):
            in_code = not in_code
        new = None
        if not in_code:
            m = re.match(r"^(#{1,5}|={2,5})\s+(.+?)\s*[#=]*$", ln)
            if m:
                new = m.group(2)
            elif i + 1 < len(lines) and ln.strip() and len(ln) < 120 and RST_ADORN.match(lines[i + 1]) and len(lines[i + 1].strip()) >= len(ln.strip()) - 1 \
                    and not RST_ADORN.match(ln):
                new = ln.strip()
                i += 1
        if new is not None:
            if "".join(buf).strip():
                out.append((head, "\n".join(buf)))
            head, buf = re.sub(r"[`*_]|\{#.*?\}|<[^>]+>", "", new).strip(), []
        elif not (RST_ADORN.match(ln) and not buf):
            buf.append(ln)
        i += 1
    if "".join(buf).strip():
        out.append((head, "\n".join(buf)))
    return out


DOC_Q = ["Explain \"{h}\" in the context of {ctx}.", "{ctx}: what do I need to know about \"{h}\"?",
         "Walk me through \"{h}\" ({ctx}).", "I'm working with {ctx}. Can you cover \"{h}\"?",
         "What does {ctx} say about \"{h}\"? Give me the practical details."]


PRETTY = {"metasploit-framework": "Metasploit", "zap-website": "OWASP ZAP", "projectdiscovery-docs": "ProjectDiscovery tools (Nuclei, httpx, subfinder)",
          "radare2-book": "radare2", "wazuh-documentation": "Wazuh", "wireshark-guides": "Wireshark", "kali-docs": "Kali Linux", "linux-kernel-docs": "the Linux kernel",
          "sqlmap-wiki": "sqlmap", "zeek-docs": "Zeek", "llvm-sanitizer-docs": "the Clang/LLVM sanitizers", "openssf-hardening-guides": "the OpenSSF compiler hardening guides",
          "kspp": "the Kernel Self Protection Project", "sanitizers-wiki": "the compiler sanitizers (ASan, MSan, TSan)", "selinux-notebook": "SELinux", "ebpf-docs": "eBPF",
          "volatility-wiki": "Volatility", "volatility3": "Volatility 3", "systems-approach-book": "computer networking", "cnp3-ebook": "computer networking",
          "windows-exporter": "the Prometheus Windows exporter", "ubuntu-server-docs": "Ubuntu Server", "plaso-docs": "Plaso (log2timeline)", "velociraptor-docs": "Velociraptor",
          "ossem": "OSSEM", "elastic-common-schema": "the Elastic Common Schema", "windows-log-settings": "Windows event log configuration", "sysmon-modular": "Sysmon",
          "aflplusplus-docs": "AFL++", "oss-fuzz-docs": "OSS-Fuzz", "google-fuzzing": "fuzzing", "syzkaller-docs": "syzkaller", "pytest-docs": "pytest",
          "hypothesis-docs": "Hypothesis (property-based testing)", "caldera-fieldmanual": "MITRE Caldera", "javascript-testing-best-practices": "JavaScript testing",
          "codeql": "CodeQL", "payloads-all-the-things": "attack techniques used in security testing", "atomic-red-team": "Atomic Red Team", "incident-playbook": "incident response",
          "atc-react": "the RE&CT incident response framework", "ir-plan-template": "incident response planning", "certcc-ssvc": "SSVC", "lua-nginx-module": "OpenResty's lua-nginx-module",
          "kernel-hardening-checker": "Linux kernel hardening", "kernel-defence-map": "Linux kernel defences", "kubernetes-docs": "Kubernetes", "docker-docs": "Docker",
          "fluent-bit-docs": "Fluent Bit", "rsyslog-docs": "rsyslog", "vector-docs": "Vector", "falco-docs": "Falco", "tetragon-docs": "Tetragon", "kpatch": "kpatch (kernel live patching)",
          "lkrg": "the Linux Kernel Runtime Guard", "scapy": "Scapy", "coredns": "CoreDNS", "etcd-docs": "etcd", "rabbitmq-docs": "RabbitMQ", "istio-docs": "Istio",
          "prometheus-docs": "Prometheus", "opentelemetry-docs": "OpenTelemetry", "containerd-docs": "containerd", "redis-docs": "Redis", "bloodhound": "BloodHound",
          "suricata": "Suricata", "yara": "YARA", "ghidra": "Ghidra", "zabbix-templates": "Zabbix monitoring"}
JUNK_FILE = re.compile(r"^(contributing|code[-_]of[-_]conduct|changelog|changes|history|security|licen[sc]e|maintainers|governance|authors|notice|"
                       r"pull_request_template|issue_template|release[-_ ]?notes?|roadmap|support|todo)\b", re.I)


def product(path: str, default: str) -> str:
    """A readable name for the project a documentation file belongs to, from its folder under sources/."""
    rel = os.path.relpath(path, S).replace("\\", "/").split("/")
    folder = rel[1] if len(rel) > 2 else ""
    if folder in PRETTY:
        return PRETTY[folder]
    name = re.sub(r"[-_](docs?|website|wiki|documentation|book|guides?)$", "", folder).replace("-", " ").strip()
    return name.title() if name else default


def docs(kind: str, topic: str, tier: str, files: list[str], ctx, *, cap_per_file: int = 12, min_body: int = 350, licence: str = "",
         title_of=None) -> None:
    for path in files:
        if JUNK_FILE.match(os.path.basename(path)):
            continue
        try:
            meta, text = front_matter(read(path))
        except Exception:  # noqa: BLE001
            continue
        title = str(meta.get("title") or "").strip() or (title_of(path) if title_of else "")
        secs = sections(clean(text))
        if not title:
            title = next((h for h, _ in secs if h), "")
        if SKIP_HEAD.match(title) or JUNK_FILE.match(title):
            continue
        n = 0
        for head, body in secs:
            body = body.strip()
            if len(body) < min_body or SKIP_HEAD.match(head or "") or n >= cap_per_file:
                continue
            label = head if head and head.lower() != title.lower() else title
            if not label:
                continue
            context = ctx(path) if callable(ctx) else ctx
            if title and label != title:
                context = f"{context} ({title})"
            emit(kind, topic, tier, pick(DOC_Q, h=label, ctx=context), body, source=os.path.relpath(path, S).replace("\\", "/"),
                 key=f"{kind}:{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}", licence=licence)
            n += 1


# ------------------------------------------------------------------ the downloaded dataset
def core_dataset() -> None:
    def turns(conv):
        s = " ".join(t.get("value", "") for t in conv if t.get("from") == "system")
        q = "\n".join(t.get("value", "") for t in conv if t.get("from") in ("human", "user"))
        a = "\n".join(t.get("value", "") for t in conv if t.get("from") in ("gpt", "assistant"))
        return s, q, a, len(conv)

    cve_seen: set[str] = set()
    path = os.path.join(DS, "vulnerability-intelligence", "cybersec_master.jsonl")
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                s, q, a, n = turns(json.loads(line).get("conversations") or [])
            except Exception:  # noqa: BLE001
                continue
            if n > 3 or not q or not a:          # multi-turn records are left out of this single-turn build
                continue
            src = "cyber-security/cybersec_master.jsonl"
            if s.startswith("You are a highly specialized AI assistant for advanced cyber-defense"):
                emit("defense_qa", "defense", "C", q, a.replace("\\n", "\n"), source=src, licence="Apache-2.0 (Trendyol)")
            elif s.startswith("You are an advanced AI assistant specialized in cybersecurity causal reasoning"):
                emit("causal_reasoning", "reasoning", "C", q, a.replace("\\n", "\n"), source=src)
            elif re.search(r"EDB-\d+", q) or q.startswith("Review this exploit"):
                if rng.random() < 0.06:
                    emit("exploitdb_template", "vuln-intel", "D", q, a, source=src, key=(re.search(r"EDB-\d+", q + a) or [""])[0])
            else:
                m = re.search(r"CVE-\d{4}-\d+", q)
                if m:                               # templated CVE restatements: one phrasing per CVE, thinned hard
                    if m.group(0) not in cve_seen and rng.random() < 0.012:
                        cve_seen.add(m.group(0))
                        emit("cve_template", "vuln-intel", "D", q, a, source=src, key=m.group(0))
                elif re.match(r"What is vulnerability [A-Z]", q):
                    if rng.random() < 0.01:
                        emit("advisory_template", "vuln-intel", "D", q, a, source=src)
                elif rng.random() < 0.25:
                    emit("security_misc", "defense", "C", q, a, source=src)

    path = os.path.join(DS, "vulnerability-intelligence", "Cybersecurity-ShareGPT.jsonl")
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                s, q, a, n = turns(json.loads(line).get("conversations") or [])
            except Exception:  # noqa: BLE001
                continue
            if q and a:
                emit("stepwise_reasoning", "reasoning", "C", q, a, source="cyber-security/Cybersecurity-ShareGPT.jsonl")

    with open(os.path.join(DS, "code_train_no_reasoning.jsonl"), encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                msgs = json.loads(line).get("messages") or []
            except Exception:  # noqa: BLE001
                continue
            q = "\n".join(m.get("content", "") for m in msgs if m.get("role") == "user")
            a = "\n".join(m.get("content", "") for m in msgs if m.get("role") == "assistant")
            emit("general_code", "general", "C", q, a, source="cyber-security/code_train_no_reasoning.jsonl", system=False)

    # the small hand-built topic folders: instruction/input/output records in several capitalisations
    for path in glob.glob(os.path.join(DS, "*", "*.json")):
        if os.path.getsize(path) > 30 * 2**20:
            continue
        try:
            data = json.load(open(path, encoding="utf-8-sig", errors="replace"))
        except Exception:  # noqa: BLE001
            continue
        topic_dir = os.path.basename(os.path.dirname(path))
        for r in data if isinstance(data, list) else []:
            if not isinstance(r, dict):
                continue
            low = {k.lower(): v for k, v in r.items()}
            if isinstance(low.get("instruction"), str) and isinstance(low.get("output"), str):
                inp = low.get("input")
                q = low["instruction"] + ("\n\n" + inp if isinstance(inp, str) and inp.strip() else "")
                emit("topic_qa", "defense", "C", q, low["output"], source=f"cyber-security/{topic_dir}/{os.path.basename(path)}", key=topic_dir)


# ------------------------------------------------------------------ tools: tldr, man pages, Kali, joined tool cards
def tldr_pages() -> dict[str, dict]:
    tools: dict[str, dict] = {}
    for path in G("man-pages/tldr/pages/**/*.md"):
        platform = os.path.basename(os.path.dirname(path))
        if platform not in ("common", "linux", "windows", "osx"):
            continue
        text = read(path)
        name = (re.search(r"^# (.+)$", text, re.M) or [None, os.path.basename(path)[:-3]])[1].strip()
        desc = " ".join(d.strip() for d in re.findall(r"^> (.+)$", text, re.M) if not d.startswith(("More information", "See also")))
        desc = re.sub(r"`", "", desc)
        ex = []
        for d, c in re.findall(r"^- (.+?):?\n\n`(.+?)`$", text, re.M):
            d = re.sub(r"\[(\w)\]", r"\1", d).strip().rstrip(":")
            c = re.sub(r"\{\{\[(-[^|\]]+)\|[^\]]*\]\}\}", r"\1", c)      # {{[-x|--long]}} offers two spellings of one flag: keep the short one
            ex.append((d, re.sub(r"\{\{(.+?)\}\}", r"<\1>", c)))
        if not ex:
            continue
        tools.setdefault(name, {"desc": desc, "platform": platform, "ex": ex})
        plat = "" if platform == "common" else f" on {'macOS' if platform == 'osx' else platform.capitalize()}"
        for d, c in ex:
            emit("tldr_task", "tools", "B", pick(["How do I {d} with `{n}`{p}?", "`{n}`{p}: {d}. What's the command?", "Show me how to {d} using `{n}`{p}."],
                                               d=d[0].lower() + d[1:], n=name, p=plat),
                 f"```\n{c}\n```\nReplace the parts in `<angle brackets>` with your own values.", source="tldr-pages", key=f"tool:{name}", min_a=40,
                 licence="CC BY 4.0")
    return tools


MAN_SEC = re.compile(r"^([A-Z][A-Z0-9 /&-]{2,40})$", re.M)


def man_page(text: str) -> dict[str, str]:
    parts = MAN_SEC.split(text)
    return {parts[i].strip(): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


OPT_RE = re.compile(r"^\s{0,10}(-{1,2}[A-Za-z0-9][\w-]*(?:[ =][<\[]?[\w.|-]+[>\]]?)?(?:, ?-{1,2}[\w-]+(?:[ =][<\[]?[\w.|-]+[>\]]?)?)*)\s*$")


def man_pages(tldr: dict[str, dict]) -> None:
    kali_desc: dict[str, str] = {}
    for path in G("kali/kali-tools/*/index.md"):
        text = read(path)
        title = (re.search(r"^Title:\s*(.+)$", text, re.M) or [None, ""])[1].strip()
        m = re.search(r"^ ### \S+\s*\n(.*?)(?=^ \*\*Installed size)", text, re.M | re.S)
        desc = clean(re.sub(r"^ ", "", m.group(1), flags=re.M)) if m else ""
        if title and desc:
            kali_desc[title] = desc
            emit("kali_tool", "tools", "B", pick(["What is the Kali Linux tool `{t}` for, and how do I install it?", "Tell me about `{t}` in Kali."], t=title),
                 f"{desc}\n\nInstall it with `sudo apt install {title}`.", source="kali-tools", key=f"tool:{title}", licence="no explicit licence")
        for cmd, block in re.findall(r"^ ##### (\S+)\s*\n(.*?)(?=^ #####|^ ###|\Z)", text, re.M | re.S):
            code = re.search(r"```\n(.*?)```", re.sub(r"^ ", "", block, flags=re.M), re.S)
            if code and len(code.group(1)) > 200:
                emit("kali_usage", "tools", "B", pick(["Show me the usage and options for `{c}` (from the {t} package).", "What options does `{c}` accept?"], c=cmd, t=title),
                     "```\n" + cut(code.group(1).strip(), 4500) + "\n```", source="kali-tools", key=f"tool:{cmd}", licence="no explicit licence")

    done: set[str] = set()
    for folder, tier in (("kali/man-pages/text", "A"), ("man-pages/debian/text", "A"), ("man-pages/linux-man-pages/text", "A")):
        for path in G(folder + "/*.txt"):
            base = os.path.basename(path)[:-4]
            name, _, sec = base.rpartition(".")
            if base in done or not name:
                continue
            done.add(base)
            p = man_page(read(path))
            purpose = " ".join(p.get("NAME", "").split())
            if not purpose:
                continue
            syn = p.get("SYNOPSIS", "")
            desc = p.get("DESCRIPTION", "")
            body = f"**{purpose}**\n\n" + (f"Synopsis:\n```\n{cut(syn, 700)}\n```\n\n" if syn else "") + cut(desc, 2200)
            topic = "kernel-memory" if sec[:1] in ("2", "4") else "tools"
            emit("man_overview", topic, tier, pick(["What is `{n}` ({s}) and how is it invoked?", "Give me an overview of `{n}({s})`.", "What does `{n}` do? Show the synopsis."],
                                                    n=name, s=sec), body, source=f"man:{base}", key=f"tool:{name}", min_a=250)
            blob = p.get("OPTIONS") or desc
            lines, i, got = blob.split("\n"), 0, 0
            while i < len(lines) and got < 14:
                m = OPT_RE.match(lines[i])
                if m:
                    j, buf = i + 1, []
                    while j < len(lines) and not OPT_RE.match(lines[j]) and not MAN_SEC.match(lines[j]) and len(buf) < 40:
                        buf.append(lines[j].strip())
                        j += 1
                    text = " ".join(" ".join(buf).split())
                    if 80 <= len(text) <= 1500:
                        emit("man_option", "tools", tier, pick(["What does the `{o}` option of `{n}` do?", "In `{n}`, what is `{o}` for?", "`{n} {o}`: what does that do?"],
                                                              o=m.group(1).strip(), n=name), text, source=f"man:{base}", key=f"tool:{name}", min_a=80)
                        got += 1
                    i = j
                else:
                    i += 1
            # joined card: purpose + synopsis + the commands people actually use + Kali packaging
            t = tldr.get(name)
            if t and sec[:1] in ("1", "8"):
                card = f"**{purpose}**\n\n" + (f"Synopsis:\n```\n{cut(syn, 500)}\n```\n\n" if syn else "") + "Common tasks:\n\n" + \
                    "\n".join(f"- {d}:\n  `{c}`" for d, c in t["ex"][:8])
                if name in kali_desc:
                    card += f"\n\nOn Kali it ships as the `{name}` package (`sudo apt install {name}`)."
                emit("tool_card", "tools", "A", pick(["Give me a working overview of `{n}`: what it's for and the commands I'll use most.",
                                                      "I need to get productive with `{n}` quickly. Purpose, syntax and the common invocations, please."], n=name),
                     card, source=f"man:{base} + tldr-pages", key=f"tool:{name}", joined=True, min_a=300)


# ------------------------------------------------------------------ prioritization: SSVC + CVSS + EPSS + KEV, joined per CVE
EXPL = {"none": "no evidence of active exploitation and no public proof of concept", "poc": "a public proof of concept exists, but no confirmed exploitation",
        "active": "there is reliable evidence it is being exploited in the wild"}
AUTO = {"no": "an attacker cannot reliably automate the first four kill-chain steps (reconnaissance, weaponization, delivery, exploitation)",
        "yes": "an attacker can reliably automate reconnaissance through exploitation, so it can be used at scale"}
TECH = {"partial": "exploitation gives limited control of, or limited information about, the affected component",
        "total": "exploitation gives full control of the affected software or full disclosure of its information"}
LEVEL = {"track": "Track: no action needed now; remediate within standard update timelines",
         "track*": "Track*: monitor closely for changes; remediate within standard update timelines",
         "attend": "Attend: needs attention from supervisors; remediate sooner than standard timelines",
         "act": "Act: needs attention from leadership; remediate as soon as possible"}


def prioritization() -> None:
    tree = {}
    with open(os.path.join(S, "prioritization/certcc-ssvc/data/csv/cisa/cisa_coordinator_2_0_3.csv"), encoding="utf-8") as fh:
        for row in list(csv.reader(fh))[1:]:
            tree[(row[1].replace("public poc", "poc"), row[2], row[3], row[4])] = row[5]
    bod = {}
    with open(os.path.join(S, "prioritization/certcc-ssvc/data/csv/cisa/cisa_bod_26_04_1_0_0.csv"), encoding="utf-8") as fh:
        for row in list(csv.reader(fh))[1:]:
            bod[(row[1], row[2], row[3], row[4])] = row[5]
    epss = {}
    with gzip.open(os.path.join(S, "cve/epss/epss_scores-current.csv.gz"), "rt", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if row and row[0].startswith("CVE-"):
                epss[row[0]] = (float(row[1]), float(row[2]))
    kev_data = json.load(open(os.path.join(S, "cve/kev/known_exploited_vulnerabilities.json"), encoding="utf-8"))["vulnerabilities"]
    kev = {v["cveID"]: v for v in kev_data}

    for v in kev_data:
        ransom = v.get("knownRansomwareCampaignUse", "Unknown")
        emit("kev_entry", "prioritization", "A",
             pick(["Is {c} ({p}) known to be exploited, and what action is required?", "What does CISA's Known Exploited Vulnerabilities catalogue say about {c}?"],
                  c=v["cveID"], p=f"{v.get('vendorProject', '')} {v.get('product', '')}".strip()),
             f"Yes. {v['cveID']} is in CISA's Known Exploited Vulnerabilities catalogue as \"{v.get('vulnerabilityName', '')}\" (added {v.get('dateAdded', '')}).\n\n"
             f"{v.get('shortDescription', '')}\n\n**Required action:** {v.get('requiredAction', '')}\n**Federal due date:** {v.get('dueDate', '')}\n"
             f"**Known ransomware campaign use:** {ransom}\n\nCatalogue membership means exploitation is confirmed, so treat the SSVC Exploitation value as *active* when you prioritize it.",
             source="cisa-kev", key=v["cveID"])

    files = G("prioritization/cisa-vulnrichment-2026/2026/**/*.json")
    rng.shuffle(files)
    per_combo = collections.Counter()
    for path in files:
        try:
            d = json.load(open(path, encoding="utf-8"))
            cna = d["containers"]["cna"]
            cve = d["cveMetadata"]["cveId"]
            ssvc = next(m["other"]["content"] for a in d["containers"].get("adp", []) for m in a.get("metrics", [])
                        if m.get("other", {}).get("type") == "ssvc")
            opts = {k: str(v).lower() for o in ssvc["options"] for k, v in o.items()}
            e, au, ti = opts["Exploitation"], opts["Automatable"], opts["Technical Impact"]
            desc = next(x["value"] for x in cna["descriptions"] if x.get("lang", "en").startswith("en"))
        except Exception:  # noqa: BLE001
            continue
        if e not in EXPL or au not in AUTO or ti not in TECH or len(desc) < 60:
            continue
        cap = 4000 if e == "active" else 700
        if per_combo[(e, au, ti)] >= cap:
            continue
        per_combo[(e, au, ti)] += 1
        aff = cna.get("affected") or [{}]
        product = f"{aff[0].get('vendor', '')} {aff[0].get('product', '')}".strip() or "not stated"
        cvss = "not provided"
        for m in cna.get("metrics", []):
            for k in ("cvssV4_0", "cvssV3_1", "cvssV3_0"):
                if k in m:
                    cvss = f"{m[k].get('baseScore')} ({m[k].get('baseSeverity', '').title()}), {m[k].get('vectorString', '')}"
                    break
        ep = epss.get(cve)
        in_kev = cve in kev
        q = (f"Help me prioritize {cve}.\n\nDescription: {cut(desc, 700)}\nAffected: {product}\nCVSS: {cvss}\n"
             f"EPSS: {f'{ep[0]:.3f} (percentile {ep[1]:.0%})' if ep else 'no score'}\nIn CISA KEV: {'yes' if in_kev else 'no'}\n\n"
             + rng.choice(["Use SSVC and tell me how fast this needs fixing.", "Walk through the SSVC decision and give me the remediation priority.",
                           "What priority does this get under CISA's SSVC tree, and why?"]))
        lv = [tree[(e, au, ti, m)] for m in ("low", "medium", "high")]
        a = (f"**SSVC decision points for {cve}** (CISA's assessment):\n\n"
             f"- **Exploitation: {e}**. {EXPL[e].capitalize()}.\n- **Automatable: {au}**. {AUTO[au].capitalize()}.\n"
             f"- **Technical impact: {ti}**. {TECH[ti].capitalize()}.\n\n"
             "The last input is yours to supply: **mission and well-being impact**, meaning how much your organization's essential functions and public "
             "well-being depend on the affected system. CISA's decision tree then gives:\n\n"
             f"| Mission and well-being impact | Decision |\n|---|---|\n| Low | {LEVEL[lv[0]]} |\n| Medium | {LEVEL[lv[1]]} |\n| High | {LEVEL[lv[2]]} |\n\n")
        notes = []
        if in_kev and e != "active":
            notes.append("It is in the KEV catalogue, which is confirmed exploitation; re-run the tree with Exploitation = active, which raises the result.")
        if ep:
            notes.append(f"EPSS estimates a {ep[0]:.1%} chance of exploitation activity in the next 30 days, higher than {ep[1]:.0%} of scored CVEs. "
                         + ("That is high enough to watch for the Exploitation value changing." if ep[0] >= 0.1 else "That supports, but does not replace, the Exploitation value above."))
        if cvss != "not provided":
            notes.append("CVSS describes severity if exploited, not likelihood. Use it to understand impact, not to set the order of work by itself.")
        t_no, t_yes = bod.get(("yes" if in_kev else "no", "no", au, ti)), bod.get(("yes" if in_kev else "no", "yes", au, ti))
        if t_no and t_yes:
            notes.append(f"Under CISA's BOD 26-04 timelines, with KEV = {'yes' if in_kev else 'no'}: {t_yes} if the system is publicly exposed, {t_no} if it is not.")
        a += "Other signals:\n\n" + "\n".join(f"- {n}" for n in notes)
        emit("ssvc_priority", "prioritization", "A", q, a, source="cisa-vulnrichment + epss + kev + certcc-ssvc tables", key=cve, joined=True)


# ------------------------------------------------------------------ ATT&CK, CWE, CAPEC with joined detections and fixes
def strip_cites(t: str) -> str:
    return re.sub(r"\(Citation:[^)]*\)", "", re.sub(r"\[([^\]]+)\]\(https?://[^)]+\)", r"\1", t or "")).strip()


def sigma_rules() -> tuple[dict, dict]:
    by_tech, by_event = collections.defaultdict(list), collections.defaultdict(list)
    for path in G("tool-docs/sigma/rules/**/*.yml"):
        try:
            r = yaml.safe_load(read(path))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(r, dict) or not r.get("title") or not r.get("detection"):
            continue
        core = {k: r[k] for k in ("title", "status", "description", "tags", "logsource", "detection", "falsepositives", "level") if k in r}
        body = yaml.safe_dump(core, sort_keys=False, allow_unicode=True, width=120)
        desc = " ".join(str(r.get("description", "")).split())
        product = (r.get("logsource") or {}).get("product", "")
        emit("sigma_write", "detection", "B",
             pick(["Write a Sigma rule to detect this: {t}. {d}", "I need a Sigma detection for \"{t}\"{p}. {d}", "Draft a Sigma rule: {t}{p}."],
                  t=r["title"], d=desc, p=f" ({product} logs)" if product else ""),
             f"```yaml\n{body}```", source="sigma", key=f"sigma:{r.get('id', r['title'])}", licence="Detection Rule License 1.1")
        fps = r.get("falsepositives") or []
        fp = ("\n".join(f"- {f}" for f in fps) if isinstance(fps, list) else str(fps)) or "None documented."
        ls = ", ".join(f"{k}: {v}" for k, v in (r.get("logsource") or {}).items())
        emit("sigma_explain", "detection", "B", f"Explain what this Sigma rule detects, what data it needs, and what could cause false positives:\n\n```yaml\n{cut(body, 1800)}```",
             f"**What it detects:** {desc}\n\n**Log source needed:** {ls}\n\n**Severity:** {r.get('level', 'not set')}\n\n"
             f"**ATT&CK mapping:** {', '.join(t for t in (r.get('tags') or []) if str(t).startswith('attack.')) or 'none'}\n\n**Likely false positives:**\n{fp}",
             source="sigma", key=f"sigma:{r.get('id', r['title'])}", licence="Detection Rule License 1.1")
        for tag in r.get("tags") or []:
            m = re.match(r"attack\.(t\d{4}(?:\.\d{3})?)$", str(tag))
            if m:
                by_tech[m.group(1).upper()].append((r["title"], body))
        for ev in set(re.findall(r"EventID:\s*(\d{3,5})\b", body) + re.findall(r"EventID:\s*\n(?:\s*- \d+\n)*?\s*- (\d{3,5})", body)):
            by_event[ev].append(r["title"])
    return by_tech, by_event


def attack(by_tech: dict) -> None:
    atomics = collections.defaultdict(list)
    for path in G("tool-docs/atomic-red-team/atomics/T*/T*.md"):
        tid = os.path.basename(path)[:-3]
        atomics[tid] = re.findall(r"^## Atomic Test #\d+ - (.+)$", read(path), re.M)
    for dom in ("enterprise-attack", "ics-attack", "mobile-attack"):
        objs = json.load(open(os.path.join(S, "references/mitre-attack", dom + ".json"), encoding="utf-8"))["objects"]
        by_id = {o["id"]: o for o in objs}
        mit = collections.defaultdict(list)
        for o in objs:
            if o["type"] == "relationship" and o.get("relationship_type") == "mitigates" and not o.get("revoked"):
                m = by_id.get(o["source_ref"])
                if m and m["type"] == "course-of-action" and not m.get("x_mitre_deprecated"):
                    mit[o["target_ref"]].append((m["name"], strip_cites(o.get("description", ""))))
        for o in objs:
            if o["type"] != "attack-pattern" or o.get("revoked") or o.get("x_mitre_deprecated"):
                continue
            tid = next((r["external_id"] for r in o.get("external_references", []) if r.get("source_name", "").startswith("mitre-")), "")
            if not tid:
                continue
            desc = cut(strip_cites(o.get("description", "")), 1100)
            tactics = ", ".join(p["phase_name"] for p in o.get("kill_chain_phases", []))
            a = f"**{tid} {o['name']}** ({dom.replace('-attack', '')}; tactics: {tactics}; platforms: {', '.join(o.get('x_mitre_platforms', []))})\n\n{desc}\n\n"
            ms = mit.get(o["id"], [])
            a += "**Mitigations:**\n\n" + ("\n".join(f"- **{n}:** {d}" for n, d in ms[:8]) if ms else "- MITRE lists no preventive mitigation; rely on detection and response.") + "\n\n"
            det = strip_cites(o.get("x_mitre_detection", ""))
            sig = by_tech.get(tid, [])
            if det or sig:
                a += "**Detection:**\n\n" + (cut(det, 900) + "\n\n" if det else "")
                if sig:
                    a += "Existing Sigma rules for this technique include: " + "; ".join(t for t, _ in sig[:6]) + ".\n\n"
            if atomics.get(tid):
                a += "**Test your detection** with Atomic Red Team tests for " + tid + ": " + "; ".join(atomics[tid][:6]) + "."
            joined = bool(sig or atomics.get(tid))
            emit("attack_technique", "remediation", "A",
                 pick(["For ATT&CK {i} ({n}): how do I mitigate it, detect it, and test that my detection works?", "How do I defend against {n} ({i})?",
                       "Give me the mitigation and detection picture for ATT&CK technique {i} {n}."], i=tid, n=o["name"]),
                 a, source="mitre-attack" + (" + sigma + atomic-red-team" if joined else ""), key=tid, joined=joined)


def cwe_capec() -> None:
    fixes = collections.defaultdict(list)
    for path in G("testing/codeql/**/*.qhelp"):
        m = re.search(r"CWE-0*(\d+)", path.replace("\\", "/"))
        try:
            root = ET.fromstring(re.sub(r"<!DOCTYPE[^>]*>", "", read(path)))
        except Exception:  # noqa: BLE001
            continue
        txt = lambda tag: " ".join(" ".join(root.find(tag).itertext()).split()) if root.find(tag) is not None else ""  # noqa: E731
        ov, rec = txt("overview"), txt("recommendation")
        lang = path.replace("\\", "/").split("/codeql/")[1].split("/")[0]
        name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", os.path.basename(path)[:-6])
        if ov and rec:
            if m:
                fixes[m.group(1)].append((lang, name, rec))
            emit("codeql_fix", "testing", "A",
                 pick(["Static analysis flagged \"{n}\" in my {l} code. What is the problem and how do I fix it?", "How do I fix a \"{n}\" finding ({l})?"], n=name, l=lang),
                 f"**The problem:** {ov}\n\n**The fix:** {rec}", source="codeql-query-help", key=f"codeql:{lang}:{name}", licence="MIT")

    cwe_xml = G("remediation/mitre/cwec_v*.xml")
    root = ET.parse(cwe_xml[-1]).getroot()
    uri = root.tag.split("}")[0][1:]            # the schema namespace changes between CWE releases
    ns = {"c": uri}
    tx = lambda el: " ".join(" ".join(el.itertext()).split()) if el is not None else ""  # noqa: E731
    for w in root.iter(f"{{{uri}}}Weakness"):
        if w.get("Status") == "Deprecated":
            continue
        wid, name = w.get("ID"), w.get("Name")
        a = f"**CWE-{wid}: {name}**\n\n{tx(w.find('c:Description', ns))}\n\n{cut(tx(w.find('c:Extended_Description', ns)), 900)}\n\n"
        cons = [f"{tx(c.find('c:Scope', ns))}: {tx(c.find('c:Impact', ns))}" for c in w.findall("c:Common_Consequences/c:Consequence", ns)]
        if cons:
            a += "**What it can lead to:** " + "; ".join(cons[:5]) + "\n\n"
        mits = [(tx(m.find("c:Phase", ns)), tx(m.find("c:Description", ns))) for m in w.findall("c:Potential_Mitigations/c:Mitigation", ns)]
        if not mits:
            continue
        a += "**How to remediate, by phase:**\n\n" + "\n".join(f"- **{p or 'General'}:** {cut(d, 600)}" for p, d in mits[:7])
        det = [tx(d.find("c:Method", ns)) for d in w.findall("c:Detection_Methods/c:Detection_Method", ns)]
        if det:
            a += "\n\n**How to find it:** " + ", ".join(dict.fromkeys(det)) + "."
        fx = fixes.get(wid, [])
        if fx:
            a += "\n\n**Code-level guidance from static-analysis queries:**\n\n" + "\n".join(f"- *{n}* ({l}): {cut(r, 420)}" for l, n, r in fx[:4])
        emit("cwe_remediation", "remediation", "A",
             pick(["What is CWE-{i} ({n}) and how do I remediate it?", "How do I find and fix {n} (CWE-{i}) in my software?", "Remediation guidance for CWE-{i}, please."], i=wid, n=name),
             a, source="mitre-cwe" + (" + codeql-query-help" if fx else ""), key=f"CWE-{wid}", joined=bool(fx))

    croot = ET.parse(os.path.join(S, "remediation/mitre/capec_latest.xml")).getroot()
    uri = croot.tag.split("}")[0][1:]
    ns = {"c": uri}
    for p in croot.iter(f"{{{uri}}}Attack_Pattern"):
        if p.get("Status") in ("Deprecated", "Obsolete"):
            continue
        mits = [tx(m) for m in p.findall("c:Mitigations/c:Mitigation", ns)]
        if not mits:
            continue
        pre = [tx(x) for x in p.findall("c:Prerequisites/c:Prerequisite", ns)]
        ind = [tx(x) for x in p.findall("c:Indicators/c:Indicator", ns)]
        a = f"**CAPEC-{p.get('ID')}: {p.get('Name')}**\n\n{cut(tx(p.find('c:Description', ns)), 1000)}\n\n" + \
            ("**What the attacker needs:**\n" + "\n".join(f"- {x}" for x in pre[:5]) + "\n\n" if pre else "") + \
            ("**Indicators:**\n" + "\n".join(f"- {x}" for x in ind[:5]) + "\n\n" if ind else "") + \
            "**Mitigations:**\n" + "\n".join(f"- {cut(x, 500)}" for x in mits[:7])
        emit("capec_defence", "remediation", "A", pick(["How does the attack pattern \"{n}\" (CAPEC-{i}) work and how do I defend against it?", "Defences for CAPEC-{i} {n}?"],
                                                       i=p.get("ID"), n=p.get("Name")), a, source="mitre-capec", key=f"CAPEC-{p.get('ID')}")


# ------------------------------------------------------------------ detections and response
def detections() -> None:
    for path in G("remediation/splunk-security-content/detections/**/*.yml"):
        try:
            r = yaml.safe_load(read(path))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(r, dict) or not isinstance(r.get("search"), str) or r.get("status") == "deprecated":
            continue
        a = f"{r.get('description', '')}\n\n```spl\n{r['search'].strip()}\n```\n\n**How to implement:** {r.get('how_to_implement', '')}\n\n" \
            f"**Known false positives:** {r.get('known_false_positives', '')}"
        emit("splunk_detection", "detection", "B", pick(["Write a Splunk search to detect: {n}.", "I need a Splunk detection for \"{n}\". Include how to deploy it and its false positives."],
                                                         n=r.get("name", "")), a, source="splunk-security-content", key=f"splunk:{r.get('id', '')}", licence="Apache-2.0")
    for path in G("remediation/elastic-detection-rules/rules/**/*.toml"):
        try:
            r = tomllib.loads(read(path)).get("rule", {})
        except Exception:  # noqa: BLE001
            continue
        note = r.get("note") or ""
        if "remediation" in note.lower() and len(note) > 600:
            note = re.sub(r"^## Triage and analysis\s*", "", note).strip()
            note = re.sub(r"^> \*\*Disclaimer\*\*:?\s*\n(?:>.*\n?)*", "", note, flags=re.M).strip()   # the source's own authorship note, not guidance
            emit("alert_response", "remediation", "B",
                 pick(["An alert fired for \"{n}\". How do I triage it and what are the response and remediation steps?", "Give me the investigation and remediation playbook for: {n}.",
                       "How should I respond to a \"{n}\" detection?"], n=r.get("name", "")),
                 f"**What the alert means:** {' '.join(str(r.get('description', '')).split())}\n\n{note}", source="elastic-detection-rules", key=f"elastic:{r.get('rule_id', '')}",
                 licence="Elastic License 2.0")
        if r.get("query") and r.get("language") in ("eql", "kuery", "esql", "lucene"):
            emit("elastic_query", "detection", "B", f"Write an Elastic {r['language'].upper()} detection query for: {r.get('name', '')}.",
                 f"{' '.join(str(r.get('description', '')).split())}\n\n```{r['language']}\n{r['query'].strip()}\n```", source="elastic-detection-rules",
                 key=f"elastic:{r.get('rule_id', '')}", licence="Elastic License 2.0")
    for path in G("logging/azure-sentinel/Detections/**/*.yaml") + G("logging/azure-sentinel/Hunting Queries/**/*.yaml"):
        try:
            r = yaml.safe_load(read(path))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(r, dict) and r.get("query") and r.get("name") and "moved to new location" not in str(r.get("description", "")):
            emit("kql_detection", "detection", "B", pick(["Write a KQL query for Microsoft Sentinel to detect: {n}.", "Sentinel hunting query for \"{n}\"?"], n=r["name"]),
                 f"{str(r.get('description', '')).strip().strip(chr(39))}\n\n```kql\n{str(r['query']).strip()}\n```\n\nData sources: "
                 f"{', '.join(str(dt) for c in (r.get('requiredDataConnectors') or []) if isinstance(c, dict) for dt in (c.get('dataTypes') or [])) or 'see the tables in the query'}.",
                 source="azure-sentinel", key=f"sentinel:{r.get('id', '')}", licence="MIT")
    for path in G("indicators/prometheus-operator-runbooks/content/runbooks/**/*.md"):
        meta, text = front_matter(read(path))
        name = os.path.basename(path)[:-3]
        if name.startswith("_") or "## " not in text:
            continue
        emit("alert_runbook", "services", "B", pick(["The alert `{n}` is firing. What does it mean and what should I do?", "Runbook for the `{n}` alert, please."], n=name),
             re.sub(r"^# .+\n", "", text).strip(), source="prometheus-operator-runbooks", key=f"alert:{name}", licence="Apache-2.0")
    try:
        groups = yaml.safe_load(read(os.path.join(S, "indicators/awesome-prometheus-alerts/_data/rules.yml"), 20_000_000))["groups"]
    except Exception:  # noqa: BLE001
        groups = []
    for g in groups:
        for svc in g.get("services", []) if isinstance(g, dict) else []:
            rules = [r for e in svc.get("exporters", []) if isinstance(e, dict) for r in (e.get("rules") or []) if isinstance(r, dict)]
            if not rules:
                continue
            listing = "\n".join(f"- **{r.get('name')}** ({r.get('severity', 'warning')}): {r.get('description', '')}\n  `{' '.join(str(r.get('query', '')).split())}`" for r in rules[:14])
            emit("service_indicators", "services", "B",
                 pick(["What indicates a problem with {s}? Give me the conditions worth alerting on.", "Which Prometheus alerts should I have for {s}?"], s=svc["name"]),
                 f"Signs of trouble for {svc['name']}, as alert conditions:\n\n{listing}", source="awesome-prometheus-alerts", key=f"service:{svc['name']}", licence="CC BY 4.0")
            for r in rules:
                emit("prom_alert", "services", "B", f"Write a Prometheus alerting rule for {svc['name']}: {r.get('name')}.",
                     f"{r.get('description', '')}\n\n```yaml\n- alert: {re.sub(r'[^A-Za-z0-9]', '', str(r.get('name', '')).title())}\n  expr: {' '.join(str(r.get('query', '')).split())}\n"
                     f"  for: {r.get('for', '0m')}\n  labels:\n    severity: {r.get('severity', 'warning')}\n```", source="awesome-prometheus-alerts",
                     key=f"service:{svc['name']}", min_a=90, licence="CC BY 4.0")


# ------------------------------------------------------------------ Windows: events (joined with detections), commands, PowerShell, support
def windows(by_event: dict) -> None:
    for path in G("logging/windows-security-auditing/event-*.txt"):
        text = read(path)
        ev = re.search(r"event-(\d+)", path).group(1)
        title = text.split("\n", 1)[0].strip()
        desc = (re.search(r"Event Description:?\s*\n(.*?)(?=\n(?:Event XML|Required Server Roles|Field Descriptions|Security Monitoring Recommendations))", text, re.S) or [None, ""])[1]
        rec = (re.search(r"Security Monitoring Recommendations\s*\n(.*)", text, re.S) or [None, ""])[1]
        if not desc.strip():
            continue
        a = f"**{title}**\n\n{cut(desc.strip(), 1500)}"
        if rec.strip():
            a += "\n\n**What to monitor:**\n\n" + cut(rec.strip(), 2200)
        rules = by_event.get(ev, [])
        if rules:
            a += "\n\n**Existing Sigma detections that use this event:** " + "; ".join(rules[:8]) + "."
        emit("windows_event", "logging", "A",
             pick(["What does Windows Security event {e} tell me, and what should I alert on?", "Explain Windows event ID {e} and how to use it for detection.",
                   "Event {e} in the Security log: meaning, and monitoring recommendations?"], e=ev),
             a, source="microsoft-security-auditing" + (" + sigma" if rules else ""), key=f"event:{ev}", joined=bool(rules), licence="CC BY 4.0")
    for path in G("logging/windows-security-auditing/audit-*.txt"):
        text = read(path)
        emit("audit_policy", "logging", "A", f"What does the Windows audit policy \"{text.split(chr(10), 1)[0].strip()}\" log, and should I enable it?",
             text.split("\n", 1)[1] if "\n" in text else "", source="microsoft-security-auditing", key=f"audit:{os.path.basename(path)}", licence="CC BY 4.0")

    for path in G("services/windows-server-docs/WindowsServerDocs/administration/windows-commands/*.md"):
        meta, text = front_matter(read(path))
        name = os.path.basename(path)[:-3].replace("-", " ")
        if len(text) > 500 and name not in ("windows commands", "toc"):
            emit("windows_command", "services", "A", pick(["How do I use the Windows `{n}` command?", "Syntax and examples for the Windows command `{n}`?"], n=name),
                 re.sub(r"^# .+\n", "", text).strip(), source="windows-server-docs", key=f"wincmd:{name}", licence="CC BY 4.0")

    seen_cmdlet: set[str] = set()
    ps_files = G("powershell/powershell-docs/reference/7.4/**/*.md") + G("powershell/powershell-docs/reference/5.1/**/*.md") + \
        G("powershell/windows-powershell-docs/docset/winserver2025-ps/**/*.md")
    for path in ps_files:
        meta, text = front_matter(read(path))
        title = str(meta.get("title") or "")
        if "## SYNOPSIS" not in text or not re.match(r"^[A-Z][A-Za-z]+-[A-Za-z0-9]+$", title) or title in seen_cmdlet:
            continue
        seen_cmdlet.add(title)
        syn = (re.search(r"## SYNOPSIS\s*\n(.*?)(?=\n## )", text, re.S) or [None, ""])[1].strip()
        syntax = (re.search(r"## SYNTAX\s*\n(.*?)(?=\n## )", text, re.S) or [None, ""])[1].strip()
        ex = (re.search(r"## EXAMPLES\s*\n(.*?)(?=\n## )", text, re.S) or [None, ""])[1].strip()
        if not syn or not ex:
            continue
        emit("powershell_cmdlet", "powershell", "A",
             pick(["How do I use `{c}` in PowerShell? Show examples.", "What does the PowerShell cmdlet `{c}` do, and how do I call it?", "`{c}`: purpose, syntax and a couple of examples."], c=title),
             f"{syn}\n\n**Syntax:**\n\n{cut(syntax, 900)}\n\n**Examples:**\n\n{cut(ex, 3200)}", source="powershell-docs", key=f"ps:{title}", licence="CC BY 4.0")

    docs("powershell_topic", "powershell", "A", [p for p in G("powershell/powershell-docs/reference/7.4/**/About/*.md")], "PowerShell", cap_per_file=4, licence="CC BY 4.0")
    docs("powershell_ise", "powershell", "A", G("powershell/powershell-docs/reference/docs-conceptual/windows-powershell/ise/*.md"), "the Windows PowerShell ISE", cap_per_file=20,
         min_body=200, licence="CC BY 4.0")
    for path in G("powershell/ise-object-model/*.txt"):
        text = read(path)
        title = text.split("\n", 1)[0].strip()
        emit("powershell_ise", "powershell", "A", f"In the Windows PowerShell ISE scripting object model, explain {title[0].lower() + title[1:] if title.startswith('The ') else title}.",
             text.split("\n", 1)[1], source="microsoft-learn-archive", key=f"ise:{os.path.basename(path)}", licence="CC BY 4.0")
    docs("powershell_guide", "powershell", "A", G("powershell/powershell-docs/reference/docs-conceptual/**/*.md"), "PowerShell", cap_per_file=3, licence="CC BY 4.0")
    docs("pester", "testing", "B", G("testing/pester-docs/docs/**/*.md*"), "Pester, the PowerShell testing framework", cap_per_file=4, licence="MIT")

    for path in G("services/windows-support-articles/support/**/*.md"):
        meta, text = front_matter(read(path))
        title = str(meta.get("title") or "").strip()
        if not title or not re.search(r"^## (Symptoms?|Cause|Resolution|Workaround)", text, re.M):
            continue
        body = re.sub(r"_Original KB number:_.*\n", "", re.sub(r"^# .+\n", "", text)).strip()
        emit("windows_troubleshooting", "services", "A",
             pick(["On Windows I'm hitting this: {t}. What causes it and how do I fix it?", "Troubleshoot this Windows problem for me: {t}", "{t}. Cause and resolution?"], t=title.rstrip(".")),
             body, source="microsoft-support-articles", key=f"kb:{os.path.basename(path)}", licence="CC BY 4.0")
    docs("windows_server", "services", "A", G("services/windows-server-docs/WindowsServerDocs/security/**/*.md") + G("services/windows-server-docs/WindowsServerDocs/identity/ad-ds/**/*.md")
         + G("services/windows-server-docs/WindowsServerDocs/networking/**/*.md") + G("services/windows-server-docs/WindowsServerDocs/administration/*.md"),
         "Windows Server", cap_per_file=40, min_body=300, licence="CC BY 4.0")
    docs("windows_debugging", "kernel-memory", "A", G("kernel-memory/windows-debugger-docs/windows-driver-docs-pr/debugger/bug-check-*.md"),
         "Windows crash (bug check) analysis", cap_per_file=3, min_body=250, licence="CC BY 4.0")
    docs("windows_debugging", "kernel-memory", "A", G("kernel-memory/windows-debugger-docs/windows-driver-docs-pr/driversecurity/*.md")
         + G("kernel-memory/windows-debugger-docs/windows-driver-docs-pr/debuggercmds/*.md"), "Windows kernel debugging and driver security", cap_per_file=3, licence="CC BY 4.0")
    docs("sysinternals", "logging", "A", G("logging/sysinternals-docs/**/*.md"), "Sysinternals", cap_per_file=4, licence="CC BY 4.0")


# ------------------------------------------------------------------ hardening, locations, abuse catalogues
def hardening_and_locations() -> None:
    for path in G("logging/wazuh-ruleset/ruleset/sca/**/*.yml"):
        try:
            d = yaml.safe_load(read(path, 8_000_000))
        except Exception:  # noqa: BLE001
            continue
        policy = (d.get("policy") or {}).get("name", "") if isinstance(d, dict) else ""
        for c in (d.get("checks") or [])[:400] if isinstance(d, dict) else []:
            if not c.get("remediation") or not c.get("title"):
                continue
            rules = "\n".join(f"- `{r}`" for r in (c.get("rules") or [])[:6])
            emit("hardening_check", "remediation", "B",
                 pick(["How do I verify and remediate this hardening item: {t} ({p})", "{t} What is the rationale and the fix? ({p})"], t=c["title"], p=policy),
                 f"{c.get('description', '')}\n\n**Why it matters:** {c.get('rationale', '')}\n\n**Remediation:** {c.get('remediation', '')}"
                 + (f"\n\n**How a scanner checks it** (Wazuh SCA rule syntax):\n{rules}" if rules else ""),
                 source="wazuh-sca", key=f"sca:{c['title'][:60]}", licence="GPLv2; based on CIS Benchmarks")
    for path in G("remediation/compliance-as-code/linux_os/guide/**/rule.yml"):
        text = read(path)
        title = (re.search(r"^title:\s*['\"]?(.+?)['\"]?\s*$", text, re.M) or [None, ""])[1]
        blk = lambda k: clean(re.sub(r"</?(tt|pre|code|b|i)>", "`", re.sub(r"^ {4}", "", (re.search(rf"^{k}: \|-?\n((?: {{4}}.*\n|\n)+)", text, re.M) or [None, ""])[1], flags=re.M)))  # noqa: E731
        desc, why = blk("description"), blk("rationale")
        fix_path = os.path.join(os.path.dirname(path), "bash", "shared.sh")
        fix = read(fix_path) if os.path.isfile(fix_path) else ""
        fix = "\n".join(ln for ln in fix.split("\n") if not ln.startswith("# platform") and not ln.startswith("# reboot") and not ln.startswith("# strategy")
                        and not ln.startswith("# complexity") and not ln.startswith("# disruption")).strip()
        if not title or not desc or "{{" in desc + why + fix:
            continue
        emit("linux_hardening", "remediation", "A", pick(["Linux hardening: {t}. Why, and how do I apply it?", "How do I implement \"{t}\" on a Linux system?"], t=title),
             f"{desc}\n\n**Why it matters:** {why}" + (f"\n\n**Remediation script:**\n```bash\n{cut(fix, 1500)}\n```" if fix else ""),
             source="compliance-as-code", key=f"cac:{os.path.basename(os.path.dirname(path))}", licence="BSD-3-Clause")
    for path in G("logging/forensic-artifacts/**/*.yaml"):
        try:
            for d in yaml.safe_load_all(read(path, 8_000_000)):
                if not isinstance(d, dict) or not d.get("name") or not d.get("doc"):
                    continue
                locs = []
                for s in d.get("sources") or []:
                    at = s.get("attributes") or {}
                    locs += [str(x) for x in (at.get("paths") or [])] + [str(x) for x in (at.get("keys") or [])] + \
                        [str(kv.get("key")) + "\\" + str(kv.get("value", "")) for kv in (at.get("key_value_pairs") or []) if isinstance(kv, dict)] + \
                        [f"command: {at.get('cmd')} {' '.join(at.get('args') or [])}" for _ in [0] if at.get("cmd")]
                if not locs:
                    continue
                os_list = ", ".join(d.get("supported_os") or [])
                emit("artifact_location", "logging", "B",
                     pick(["Where on {o} do I find the forensic artifact \"{n}\", and what is it?", "Location of {n} on {o}?"], n=d["name"], o=os_list or "the system"),
                     f"{d['doc'].strip()}\n\n**Where it lives ({os_list}):**\n" + "\n".join(f"- `{x}`" for x in locs[:14]), source="forensic-artifacts", key=f"artifact:{d['name']}",
                     min_a=100, licence="Apache-2.0")
        except Exception:  # noqa: BLE001
            continue
    for path in G("tool-docs/lolbas/yml/**/*.yml"):
        try:
            d = yaml.safe_load(read(path))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(d, dict) or not d.get("Commands"):
            continue
        cmds = "\n".join(f"- **{c.get('Category', '')}** ({c.get('MitreID', '')}): {c.get('Description', '')}\n  `{c.get('Command', '')}`" for c in d["Commands"][:8])
        det = "\n".join(f"- {k}: {v}" for x in (d.get("Detection") or [])[:8] if isinstance(x, dict) for k, v in x.items() if v)
        paths = ", ".join(f"`{p.get('Path')}`" for p in (d.get("Full_Path") or []) if isinstance(p, dict))
        emit("lolbin", "detection", "B",
             pick(["How do attackers abuse `{n}` on Windows, and how do I detect it?", "Is `{n}` a living-off-the-land risk? What should I look for?"], n=d.get("Name", "")),
             f"`{d.get('Name')}`: {d.get('Description', '')}\n\nNormal locations: {paths}\n\n**Documented abuse:**\n{cmds}\n\n**Detection:**\n{det or '- No public detection listed; baseline normal use and alert on the command lines above.'}",
             source="lolbas", key=f"lolbin:{d.get('Name')}", licence="GPL-3.0")
    for path in G("indicators/osquery/specs/**/*.table"):
        text = read(path)
        name = (re.search(r'table_name\("([^"]+)"', text) or [None, ""])[1]
        desc = (re.search(r'description\("([^"]+)"', text) or [None, ""])[1]
        cols = re.findall(r'Column\("([^"]+)",\s*(\w+),\s*"([^"]*)"', text)
        if name and desc and cols:
            plat = os.path.basename(os.path.dirname(path))
            emit("osquery_table", "services", "B", pick(["Which osquery table shows this: {d} What columns does it have?", "How do I inspect \"{d}\" with osquery?"], d=desc),
                 f"Use the `{name}` table" + (f" ({plat})" if plat != "specs" else "") + f".\n\n" + "\n".join(f"- `{c}` ({t.lower()}): {x}" for c, t, x in cols[:25])
                 + f"\n\n```sql\nSELECT * FROM {name} LIMIT 10;\n```", source="osquery", key=f"osquery:{name}", min_a=120, licence="Apache-2.0 / GPL-2.0")
    for path in G("indicators/netdata-health/src/health/health.d/*.conf"):
        text = re.sub(r"\\\n\s*", " ", read(path))
        alerts = []
        for block in re.split(r"\n\s*\n", text):
            f = dict(re.findall(r"^\s*(\w+):\s*(.+)$", block, re.M))
            if ("template" in f or "alarm" in f) and (f.get("info") or f.get("summary")):
                alerts.append(f"- **{f.get('template') or f.get('alarm')}** on `{f.get('on', '')}`: {f.get('info') or f.get('summary')}"
                              + (f" Warn when `{f['warn']}`." if f.get("warn") else "") + (f" Critical when `{f['crit']}`." if f.get("crit") else ""))
        comp = os.path.basename(path)[:-5].replace("_", " ")
        if len(alerts) >= 2:
            emit("service_indicators", "services", "B", pick(["What health checks indicate a problem with {c}?", "What should I watch to know {c} is unhealthy?"], c=comp),
                 f"Health conditions worth alerting on for {comp}:\n\n" + "\n".join(alerts[:16]), source="netdata-health", key=f"service:{comp}", licence="GPL-3.0")
    for path in G("tool-docs/nmap/scripts/*.nse"):
        text = read(path)
        d = re.search(r"description\s*=\s*\[\[(.*?)\]\]", text, re.S)
        if not d:
            continue
        usage = re.findall(r"^--\s*@usage\s*(.*)$", text, re.M)
        cats = re.search(r"categories\s*=\s*\{([^}]*)\}", text)
        name = os.path.basename(path)[:-4]
        emit("nse_script", "lua", "A", pick(["What does the Nmap script `{n}` do, and how do I run it?", "How do I use the NSE script `{n}`?"], n=name),
             d.group(1).strip() + ("\n\n**Usage:**\n```\n" + "\n".join(u for u in usage if u.strip()) + "\n```" if any(u.strip() for u in usage) else f"\n\n**Usage:**\n```\nnmap --script {name} <target>\n```")
             + (f"\n\nCategories: {cats.group(1).strip()}" if cats else ""), source="nmap-nse", key=f"nse:{name}", licence="Nmap Public Source License")


# ------------------------------------------------------------------ web platform
def mdn_clean(t: str) -> str:
    t = re.sub(r"\{\{\s*(?:domxref|jsxref|htmlelement|cssxref|glossary|httpheader|httpmethod|httpstatus|svgelement|rfc)\(\s*\"([^\"]+)\"(?:\s*,\s*\"([^\"]*)\")?[^)]*\)\s*\}\}",
               lambda m: f"`{m.group(2) or m.group(1)}`", t, flags=re.I)
    t = re.sub(r"\{\{[^{}]*\}\}", "", t)
    t = re.sub(r"\[([^\]]+)\]\(/en-US/[^)]*\)", r"\1", t)
    return re.sub(r"\n## (Specifications|Browser compatibility|See also)\b.*", "", t, flags=re.S)


def web() -> None:
    areas = {"web/javascript": ("JavaScript", "web"), "web/api": ("the DOM and Web APIs", "web"), "web/http": ("HTTP", "networking"), "web/xml/xpath": ("XPath", "web"),
             "web/security": ("web security", "web"), "web/html": ("HTML", "web")}
    base = os.path.join(S, "web/mdn-content/files/en-us")
    for sub, (label, topic) in areas.items():
        for path in glob.glob(os.path.join(base, sub, "**", "index.md"), recursive=True):
            meta, text = front_matter(read(path))
            title = str(meta.get("title") or "").strip()
            body = mdn_clean(text).strip()
            if not title or len(body) < 500 or meta.get("status") and "deprecated" in str(meta.get("status")):
                continue
            emit("mdn", topic, "A", pick(["Explain {t} ({l}).", "{t} ({l}): explain it, with syntax and an example where they apply.", "What do I need to know about {t} in {l}?"], t=title, l=label),
                 body, source="mdn", key=f"mdn:{meta.get('slug', title)}", licence="CC BY-SA 2.5")
    docs("javascript_tutorial", "web", "B", G("web/javascript-info/**/article.md"), "JavaScript", cap_per_file=5, licence="CC BY-NC-SA")
    docs("javascript_book", "web", "B", G("web/eloquent-javascript/[0-9]*.md"), "JavaScript", cap_per_file=8, licence="CC BY-NC 3.0")
    docs("nodejs", "web", "A", G("web/nodejs-api/doc/api/*.md"), "the Node.js API", cap_per_file=10, licence="MIT")
    docs("devtools", "web-tools", "A", G("web-tools/chrome-devtools-docs/**/*.md"), "Chrome DevTools", cap_per_file=6, min_body=250, licence="CC BY 4.0")
    docs("devtools", "web-tools", "A", G("web-tools/firefox-devtools-docs/devtools/docs/user/**/*.rst"), "Firefox DevTools", cap_per_file=6, min_body=250, licence="MPL-2.0 / CC BY-SA")
    for sub, label, lic in (("everything-curl", "curl", "CC BY 4.0"), ("mitmproxy", "mitmproxy", "MIT"), ("playwright", "Playwright", "Apache-2.0"),
                            ("puppeteer/docs/guides", "Puppeteer", "Apache-2.0"), ("selenium", "Selenium WebDriver", "Apache-2.0")):
        docs("web_tooling", "web-tools", "A", G(f"web-tools/{sub}/**/*.md"), label, cap_per_file=4, licence=lic)
    for sub, label in (("owasp-wstg/document", "the OWASP Web Security Testing Guide"), ("owasp-cheatsheets/cheatsheets", "the OWASP Cheat Sheet Series"),
                       ("owasp-mastg", "the OWASP Mobile Application Security Testing Guide")):
        docs("owasp", "testing" if "sting" in label else "remediation", "A", G(f"references/{sub}/**/*.md") + G(f"tool-docs/{sub}/**/*.md"), label, cap_per_file=8, licence="CC BY-SA 4.0")
    docs("owasp", "remediation", "A", G("web/owasp-asvs/5.0/en/*.md") + G("web/owasp-top10/20*/docs/*.md") + G("web/owasp-api-security/**/*.md")
         + G("prioritization/owasp-community/pages/**/*.md"), "OWASP guidance", cap_per_file=6, licence="CC BY-SA 4.0")


# ------------------------------------------------------------------ everything else that is prose: one line per source
def prose() -> None:
    T = [  # kind, topic, tier, glob(s), context, cap per file, licence
        ("kernel_docs", "kernel-memory", "A", ["kernel-memory/linux-kernel-docs/Documentation/admin-guide/hw-vuln/*.rst", "kernel-memory/linux-kernel-docs/Documentation/admin-guide/LSM/*.rst",
                                               "kernel-memory/linux-kernel-docs/Documentation/admin-guide/sysctl/*.rst", "kernel-memory/linux-kernel-docs/Documentation/security/**/*.rst",
                                               "kernel-memory/linux-kernel-docs/Documentation/dev-tools/**/*.rst", "kernel-memory/linux-kernel-docs/Documentation/livepatch/*.rst",
                                               "kernel-memory/linux-kernel-docs/Documentation/admin-guide/mm/**/*.rst", "kernel-memory/linux-kernel-docs/Documentation/userspace-api/*.rst",
                                               "kernel-memory/linux-kernel-docs/Documentation/trace/*.rst", "kernel-memory/linux-kernel-docs/Documentation/admin-guide/*.rst",
                                               "kernel-memory/linux-kernel-docs/Documentation/mm/*.rst"], "the Linux kernel documentation", 8, "GPL-2.0"),
        ("kernel_hardening", "kernel-memory", "B", ["kernel-memory/kspp/**/*.md", "kernel-memory/openssf-hardening-guides/docs/**/*.md", "kernel-memory/sanitizers-wiki/*.md",
                                                    "kernel-memory/llvm-sanitizer-docs/clang/docs/*Sanitizer*.md", "kernel-memory/llvm-sanitizer-docs/clang/docs/ControlFlowIntegrity*.md",
                                                    "kernel-memory/llvm-sanitizer-docs/clang/docs/SafeStack.md", "kernel-memory/llvm-sanitizer-docs/clang/docs/ShadowCallStack.md",
                                                    "kernel-memory/kpatch/**/*.md", "kernel-memory/lkrg/*.md", "kernel-memory/kernel-hardening-checker/*.md",
                                                    "kernel-memory/kernel-defence-map/*.md"], "kernel and memory hardening", 10, "mixed open licences"),
        ("runtime_security", "kernel-memory", "B", ["kernel-memory/selinux-notebook/src/*.md", "kernel-memory/falco-docs/**/*.md", "kernel-memory/tetragon-docs/**/*.md",
                                                    "kernel-memory/ebpf-docs/docs/linux/concepts/**/*.md", "kernel-memory/ebpf-docs/docs/linux/program-type/*.md",
                                                    "kernel-memory/volatility-wiki/*.md", "tool-docs/volatility3/**/*.rst"], "Linux runtime security and memory forensics", 6, "mixed open licences"),
        ("logging_docs", "logging", "B", ["logging/fluent-bit-docs/**/*.md", "logging/rsyslog-docs/doc/source/**/*.rst", "logging/vector-docs/**/*.md", "logging/ossem/**/*.md",
                                          "logging/windows-log-settings/*.md", "logging/velociraptor-docs/**/*.md", "logging/plaso-docs/**/*.md", "logging/sysmon-modular/*.md",
                                          "logging/elastic-common-schema/docs/**/*.md"], "log collection and parsing", 4, "mixed open licences"),
        ("lua", "lua", "A", ["lua/lua-nginx-module/*.markdown", "lua/learnxinyminutes/lua.md", "lua/learnxinyminutes/*lua*"], "Lua", 60, "MIT / BSD / CC BY-SA"),
        ("quickref", "general", "B", ["lua/learnxinyminutes/*.md"], "this quick reference", 3, "CC BY-SA 3.0"),
        ("networking_book", "networking", "A", ["networking/systems-approach-book/**/*.rst", "networking/cnp3-ebook/**/*.rst", "networking/scapy/**/*.rst",
                                                "kernel-memory/linux-kernel-docs/Documentation/networking/*.rst"], "computer networking", 10, "CC BY 4.0 / CC BY-SA / GPL"),
        ("services_docs", "services", "A", ["services/kubernetes-docs/content/en/docs/tasks/debug/**/*.md", "services/kubernetes-docs/content/en/docs/concepts/**/*.md",
                                            "services/kubernetes-docs/content/en/docs/tasks/administer-cluster/**/*.md", "services/docker-docs/**/*.md", "services/ubuntu-server-docs/**/*.md",
                                            "services/istio-docs/content/en/docs/ops/**/*.md", "services/coredns/plugin/*/README.md", "services/etcd-docs/**/op-guide/*.md",
                                            "services/rabbitmq-docs/docs/*.md", "services/prometheus-docs/**/*.md", "services/opentelemetry-docs/content/en/docs/concepts/**/*.md",
                                            "services/containerd-docs/docs/*.md", "services/redis-docs/content/operate/oss_and_stack/management/**/*.md",
                                            "indicators/windows-exporter/docs/*.md"], "service operations", 5, "mixed open licences"),
        ("zabbix_triggers", "services", "B", ["indicators/zabbix-templates/templates/**/README.md"], "Zabbix monitoring templates", 4, "AGPL-3.0"),
        ("testing_docs", "testing", "A", ["testing/aflplusplus-docs/**/*.md", "testing/oss-fuzz-docs/docs/**/*.md", "testing/google-fuzzing/**/*.md", "testing/syzkaller-docs/docs/**/*.md",
                                          "testing/pytest-docs/doc/en/how-to/*.rst", "testing/pytest-docs/doc/en/explanation/*.rst", "testing/hypothesis-docs/**/*.rst",
                                          "testing/caldera-fieldmanual/**/*.md", "testing/javascript-testing-best-practices/readme.md", "testing/codeql/docs/codeql/**/*.rst",
                                          "kernel-memory/llvm-sanitizer-docs/llvm/docs/LibFuzzer.*"], "software and security testing", 8, "mixed open licences"),
        ("tool_docs", "tools", "A", ["tool-docs/metasploit-framework/docs/**/*.md", "tool-docs/zap-website/**/*.md", "tool-docs/sqlmap-wiki/*.md", "tool-docs/projectdiscovery-docs/**/*.md*",
                                     "tool-docs/bloodhound/**/*.md*", "tool-docs/radare2-book/src/**/*.md", "tool-docs/suricata/doc/userguide/**/*.rst", "tool-docs/yara/docs/*.rst",
                                     "tool-docs/zeek-docs/**/*.rst", "tool-docs/wazuh-documentation/source/user-manual/**/*.rst", "tool-docs/wireshark-guides/doc/wsug_src/*.adoc",
                                     "tool-docs/ghidra/**/*.md", "kali/kali-docs/**/*.md"], "this security tool's documentation", 4, "mixed open licences"),
        ("metasploit_module", "tools", "A", ["tool-docs/metasploit-framework/documentation/modules/**/*.md"], "this Metasploit module", 2, "BSD-3-Clause"),
        ("offsec_reference", "testing", "B", ["tool-docs/payloads-all-the-things/**/README.md", "tool-docs/atomic-red-team/atomics/T*/T*.md"], "security testing techniques", 5, "MIT"),
        ("incident_response", "remediation", "B", ["remediation/incident-playbook/**/*.md", "remediation/atc-react/**/*.md", "remediation/ir-plan-template/**/*.md"],
         "incident response", 6, "MIT / Apache-2.0"),
        ("prioritization_docs", "prioritization", "A", ["prioritization/certcc-ssvc/docs/**/*.md"], "SSVC (Stakeholder-Specific Vulnerability Categorization)", 8, "CC BY (see repository)"),
    ]
    for kind, topic, tier, globs, ctx, cap, lic in T:
        files = [f for g in globs for f in G(g)]
        if kind == "quickref":      # one file per language or tool: the file name is the subject
            namer = lambda p: os.path.basename(p).split(".")[0].replace("-", " ") + " (quick reference)"  # noqa: E731
        elif kind in ("lua", "metasploit_module"):
            namer = ctx
        else:
            namer = lambda p, d=ctx: product(p, d)  # noqa: E731
        docs(kind, topic, tier, files, namer, cap_per_file=cap, licence=lic)
    for name in ("lua-5.4-manual.txt", "lua-5.1-manual.txt"):
        text = read(os.path.join(S, "lua/manual", name), 4_000_000)
        ver = name[4:7]
        parts = re.split(r"\n(\d+(?:\.\d+)* – [^\n]+)\n", text)
        for i in range(1, len(parts) - 1, 2):
            emit("lua", "lua", "A", pick(["Lua {v}: explain {h}.", "From the Lua {v} reference manual, what does it say about {h}?"], v=ver, h=parts[i].split("–", 1)[1].strip()),
                 parts[i + 1], source=f"lua-{ver}-manual", key=f"lua:{ver}:{parts[i][:20]}", min_a=250, licence="MIT")
    for path in G("prioritization/first/cvss-*.txt") + G("prioritization/first/epss-*.txt"):
        text = read(path)
        parts = re.split(r"\n(\d+(?:\.\d+)*\.? [A-Z][^\n]{3,80})\n", text)
        label = "CVSS " + ("4.0" if "v4" in path else "3.1") if "cvss" in path else "EPSS"
        for i in range(1, len(parts) - 1, 2):
            emit("scoring_spec", "prioritization", "A", f"{label}: explain \"{re.sub(r'^[0-9. ]+', '', parts[i]).strip()}\".", parts[i + 1], source=os.path.basename(path),
                 key=f"spec:{label}:{parts[i][:24]}", min_a=300, licence="FIRST.org, free to use with attribution")


LEGAL_LABEL = {"17cfr229.106-item106": "17 CFR 229.106 (Regulation S-K Item 106, cybersecurity disclosure)", "45cfr164-subpartD-breach-notification": "the HIPAA Breach Notification Rule (45 CFR 164 Subpart D)",
               "16cfr314-ftc-safeguards": "the FTC Safeguards Rule (16 CFR Part 314)", "12cfr53-occ-incident-notification": "the OCC computer-security incident notification rule (12 CFR Part 53)",
               "17cfr248-regulation-s-p": "SEC Regulation S-P (17 CFR Part 248)", "ais-program-page": "CISA's Automated Indicator Sharing programme",
               "ssvc-overview": "CISA's SSVC overview", "bod-22-01-known-exploited-vulnerabilities": "CISA Binding Operational Directive 22-01",
               "bod-19-02-remediation-requirements": "CISA Binding Operational Directive 19-02"}


def legal_texts() -> None:
    """Plain-text legal and regulatory documents have no markup, so they are cut into passages at paragraph breaks."""
    files = G("legal/ecfr/*.txt") + G("legal/usc/*.txt") + G("legal/crs/*.txt") + G("legal/cisa-2015/*.txt") + G("legal/federal-register/*.txt") + \
        G("prioritization/cisa-directives/*.txt") + G("prioritization/cisa-ssvc/*.txt")
    for path in files:
        stem = os.path.basename(path)[:-4]
        label = LEGAL_LABEL.get(stem, "")
        if not label and "usc" in stem:
            label = f"6 U.S.C. § {stem[4:]} (Cybersecurity Information Sharing Act of 2015, as codified)"
        if not label and stem.startswith("IF12959"):
            label = "the Congressional Research Service brief on the Cybersecurity Information Sharing Act of 2015"
        meta_path = path[:-4] + ".json"
        if not label and os.path.isfile(meta_path):
            m = json.load(open(meta_path, encoding="utf-8"))
            label = f"the Federal Register {str(m.get('type', 'document')).lower()} \"{m.get('title', '')}\" ({m.get('citation', '')})"
        if not label:
            continue
        paras = [p.strip() for p in re.split(r"\n\s*\n", clean(read(path, 3_000_000))) if len(p.strip()) > 40]
        chunk, size, n = [], 0, 0
        for p in paras + [""]:
            if p and size + len(p) < 2800:
                chunk.append(p)
                size += len(p)
                continue
            if chunk and size > 500 and n < 24:
                lead = " ".join(chunk[0].split())[:90].rstrip(" ,;:")
                q = f"What does {label} say?" if n == 0 and len(paras) < 12 else f"In {label}, what does the passage beginning \"{lead}\" provide?"
                emit("legal", "legal", "A", q, "\n\n".join(chunk), source=os.path.relpath(path, S).replace("\\", "/"), key=f"legal:{stem}", licence="US government work")
                n += 1
            chunk, size = ([p], len(p)) if p else ([], 0)


def general_replay() -> None:
    path = os.path.join(OUT, "external", "dolly", "databricks-dolly-15k.jsonl")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            q = r["instruction"] + ("\n\n" + r["context"] if r.get("context") else "")
            emit("general_instruction", "general", "C", q, r["response"], source="dolly-15k", system=False, min_a=60, licence="CC BY-SA 3.0")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if "--add" in sys.argv:      # append the output of named converters to an existing pool, e.g. --add cwe legal
        out_fh = open(os.path.join(OUT, "candidates.jsonl"), "a", encoding="utf-8", newline="\n")
        extra = {"cwe": cwe_capec, "legal": legal_texts}
        for name in sys.argv[sys.argv.index("--add") + 1:]:
            guard(name, extra[name])
        out_fh.close()
        print({k: v for k, v in stats.items() if ":" not in k}, dict(errors))
        sys.exit(0)
    out_fh = open(os.path.join(OUT, "candidates.jsonl"), "w", encoding="utf-8", newline="\n")
    print("building candidates", flush=True)
    guard("downloaded dataset", core_dataset)
    tl: dict = {}
    def _tldr():  # noqa: E306
        tl.update(tldr_pages())
    guard("tldr tasks", _tldr)
    guard("man pages, Kali, tool cards", man_pages, tl)
    guard("prioritization (SSVC/KEV/EPSS)", prioritization)
    sig: list = [{}, {}]
    def _sigma():  # noqa: E306
        sig[0], sig[1] = sigma_rules()
    guard("sigma rules", _sigma)
    guard("ATT&CK technique cards", attack, sig[0])
    guard("CWE / CAPEC / CodeQL fixes", cwe_capec)
    guard("detections, runbooks, alerts", detections)
    guard("Windows events, commands, PS", windows, sig[1])
    guard("hardening, locations, catalogues", hardening_and_locations)
    guard("web platform and tooling", web)
    guard("prose documentation", prose)
    guard("legal and regulatory text", legal_texts)
    guard("general instruction replay", general_replay)
    out_fh.close()
    kinds = {k: v for k, v in stats.items() if ":" not in k}
    json.dump({"kinds": dict(sorted(kinds.items(), key=lambda kv: -kv[1])), "dropped": {k: v for k, v in stats.items() if ":" in k}, "errors": dict(errors)},
              open(os.path.join(OUT, "candidates_stats.json"), "w", encoding="utf-8"), indent=1)
    print(f"\n{sum(kinds.values())} candidates in {len(kinds)} kinds -> data/candidates.jsonl")
    for e, n in errors.items():
        print("  ERROR", e)
