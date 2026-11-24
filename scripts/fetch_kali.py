"""Collect man pages for the tools that ship in Kali Linux into sources/kali/.

Standard library only. Safe to re-run: finished work is skipped.

    python scripts/fetch_kali.py plan     # read Kali's package indexes and report what is available
    python scripts/fetch_kali.py debian   # rendered man pages from manpages.debian.org (tools Debian also packages)
    python scripts/fetch_kali.py debs     # remaining pages straight out of Kali's .deb packages (man source format)

Expects the Kali tool pages to be present in sources/kali/kali-tools (a clone of
https://gitlab.com/kalilinux/documentation/kali-tools). Nothing downloaded is ever executed.
"""
from __future__ import annotations

import gzip
import io
import json
import lzma
import os
import re
import sys
import tarfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_sources as fs  # noqa: E402  (shared download, manifest and HTML-to-text helpers)

KALI = os.path.join(fs.OUT, "kali")
TOOLS = os.path.join(KALI, "kali-tools")
INDEX = os.path.join(KALI, "index")
MAN = os.path.join(KALI, "man-pages")
MIRROR = "https://http.kali.org/kali"
PLAN = os.path.join(INDEX, "plan.json")
MAX_DEB = 300 * 2**20  # skip packages larger than this in the `debs` stage
MAN_RE = re.compile(r"^usr/share/man/man([1-8])/([^/\s]+?)\.(\d\w*)\.gz$")
# Commands (1), file formats (5), overviews (7) and admin commands (8). Section 3 is skipped on purpose:
# it is programming-library reference (OpenSSL's alone is 6,000+ pages), not tool usage.
KEEP_SECTIONS = {"1", "5", "7", "8"}

fs.HOST_DELAY.update({"http.kali.org": 0.2, "kali.download": 0.2})


def tool_packages() -> dict[str, str]:
    """binary package name -> tool folder, read from the kali-tools pages."""
    pkgs: dict[str, str] = {}
    for tool in sorted(os.listdir(TOOLS)):
        page = os.path.join(TOOLS, tool, "index.md")
        if not os.path.isfile(page):
            continue
        with open(page, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = re.match(r"^\s###\s+(\S+)\s*$", line)
                if m:
                    pkgs.setdefault(m.group(1), tool)
    return pkgs


def load_packages() -> dict[str, dict]:
    raw = fs.fetch("kali", f"{MIRROR}/dists/kali-rolling/main/binary-amd64/Packages.gz",
                   os.path.join(INDEX, "Packages-amd64.gz"), "Kali package index")
    out: dict[str, dict] = {}
    cur: dict[str, str] = {}
    for line in gzip.decompress(raw).decode("utf-8", "replace").splitlines() + [""]:
        if not line:
            if cur.get("Package"):
                out[cur["Package"]] = {"file": cur.get("Filename", ""), "size": int(cur.get("Size", 0) or 0), "version": cur.get("Version", "")}
            cur = {}
        elif line[0] not in " \t" and ":" in line:
            k, v = line.split(":", 1)
            if k in ("Package", "Filename", "Size", "Version"):
                cur[k] = v.strip()
    return out


def load_man_index(wanted: set[str]) -> dict[str, list[str]]:
    """package -> sorted list of 'name.section' man pages, from Kali's Contents indexes."""
    pages: dict[str, set[str]] = {}
    for arch in ("amd64",):  # Kali lists architecture-independent packages in this index too
        raw = fs.fetch("kali", f"{MIRROR}/dists/kali-rolling/main/Contents-{arch}.gz",
                       os.path.join(INDEX, f"Contents-{arch}.gz"), "Kali package contents index")
        if not raw:
            continue
        with gzip.open(io.BytesIO(raw), "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.startswith("usr/share/man/man"):
                    continue
                path, _, owners = line.rstrip("\n").rpartition(" ")
                m = MAN_RE.match(path.strip())
                if not m or m.group(1) not in KEEP_SECTIONS:
                    continue
                for owner in owners.split(","):
                    pkg = owner.rsplit("/", 1)[-1]
                    if pkg in wanted:
                        pages.setdefault(pkg, set()).add(f"{m.group(2)}.{m.group(3)}")
    return {k: sorted(v) for k, v in pages.items()}


def plan() -> dict:
    pkgs = tool_packages()
    meta = load_packages()
    man = load_man_index(set(pkgs))
    plan_data = {"packages": {p: {"tool": pkgs[p], "pages": man[p], **meta.get(p, {"file": "", "size": 0, "version": ""})} for p in sorted(man)}}
    os.makedirs(INDEX, exist_ok=True)
    with open(PLAN, "w", encoding="utf-8") as fh:
        json.dump(plan_data, fh, indent=1)
    all_pages = sorted({pg for v in man.values() for pg in v})
    sizes = sorted(v["size"] for v in plan_data["packages"].values())
    fs.log(f"tool packages on kali.org/tools : {len(pkgs)}")
    fs.log(f"  found in Kali's package index : {sum(1 for p in pkgs if p in meta)}")
    fs.log(f"  that ship man pages           : {len(man)}")
    fs.log(f"distinct man pages              : {len(all_pages)}")
    fs.log(f"download size of those packages : {sum(sizes) / 2**30:.2f} GiB total; "
           f"{sum(s for s in sizes if s <= MAX_DEB) / 2**30:.2f} GiB if packages over {MAX_DEB // 2**20} MiB are skipped "
           f"({sum(1 for s in sizes if s > MAX_DEB)} skipped)")
    top = sorted(plan_data["packages"].items(), key=lambda kv: -len(kv[1]["pages"]))[:8]
    fs.log("most pages: " + ", ".join(f"{p} ({len(v['pages'])})" for p, v in top))
    return plan_data


def read_plan() -> dict:
    if not os.path.isfile(PLAN):
        return plan()
    with open(PLAN, encoding="utf-8") as fh:
        return json.load(fh)


def safe(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def have_text(page: str) -> bool:
    return any(os.path.isfile(os.path.join(d, safe(page) + ext)) for d, ext in (
        (os.path.join(MAN, "text"), ".txt"), (os.path.join(fs.OUT, "man-pages", "debian", "text"), ".txt")))


def stage_debian() -> None:
    """Fetch rendered pages from Debian for every page name; misses are left for the `debs` stage."""
    data = read_plan()
    pages = sorted({pg for v in data["packages"].values() for pg in v["pages"]})
    miss_path = os.path.join(INDEX, "debian-misses.json")
    misses = set(json.load(open(miss_path, encoding="utf-8"))) if os.path.isfile(miss_path) else set()
    todo = [p for p in pages if not have_text(p) and p not in misses]
    fs.log(f"debian: {len(pages)} pages wanted, {len(todo)} to try ({len(misses)} known misses)")
    lic = "Debian manpages: per-package free licences (GPL/BSD/MIT etc.), see each page's source package"
    got = 0
    for i, page in enumerate(todo, 1):
        name, _, section = page.rpartition(".")
        dest = os.path.join(MAN, "html", safe(page) + ".html")
        before = len(fs.failures)
        raw = fs.fetch("kali", f"https://manpages.debian.org/{name}.{section}.en.html", dest, lic)
        if raw:
            text = fs.html_to_text(raw, "manual-text")
            if len(text) > 200:
                fs.write_text(os.path.join(MAN, "text", safe(page) + ".txt"), text)
                got += 1
        else:
            del fs.failures[before:]  # a miss here is expected, not a failure
            fs.stats["failed"] -= 1
            misses.add(page)
        if i % 100 == 0 or i == len(todo):
            fs.log(f"debian:   {i} of {len(todo)} tried, {got} found")
            with open(miss_path, "w", encoding="utf-8") as fh:
                json.dump(sorted(misses), fh)
    fs.log(f"debian: found {got}; {len(misses)} pages are not on Debian's site")


def ar_members(blob: bytes):
    """Yield (name, bytes) for each member of a .deb (ar archive)."""
    if blob[:8] != b"!<arch>\n":
        raise ValueError("not an ar archive")
    pos = 8
    while pos + 60 <= len(blob):
        head = blob[pos:pos + 60]
        name = head[:16].decode("ascii", "replace").strip().rstrip("/")
        size = int(head[48:58].decode("ascii").strip() or 0)
        pos += 60
        yield name, blob[pos:pos + size]
        pos += size + (size & 1)


def stage_debs() -> None:
    """Download Kali packages that still have missing pages and pull the man pages out of them."""
    data = read_plan()
    done_path = os.path.join(INDEX, "debs-done.json")
    done = set(json.load(open(done_path, encoding="utf-8"))) if os.path.isfile(done_path) else set()
    todo = []
    for pkg, v in data["packages"].items():
        missing = [p for p in v["pages"] if not have_text(p) and not os.path.isfile(os.path.join(MAN, "roff", safe(p)))]
        if missing and pkg not in done and v["file"]:
            todo.append((pkg, v, missing))
    big = [(p, v) for p, v, _ in todo if v["size"] > MAX_DEB]
    todo = [t for t in todo if t[1]["size"] <= MAX_DEB]
    fs.log(f"debs: {len(todo)} packages to open ({sum(t[1]['size'] for t in todo) / 2**20:.0f} MiB); "
           f"{len(big)} skipped for size: {', '.join(p for p, _ in big[:12])}")
    pages = 0
    for i, (pkg, v, missing) in enumerate(todo, 1):
        try:
            blob = fs.http_get(f"{MIRROR}/{v['file']}", timeout=300)
            payload = next(((n, b) for n, b in ar_members(blob) if n.startswith("data.tar")), None)
            if payload is None:
                raise ValueError("no data.tar member")
            name, body = payload
            if name.endswith(".xz"):
                stream = lzma.LZMAFile(io.BytesIO(body))
            elif name.endswith(".gz"):
                stream = gzip.GzipFile(fileobj=io.BytesIO(body))
            elif name.endswith(".tar"):
                stream = io.BytesIO(body)
            else:
                raise ValueError(f"unsupported compression: {name}")
            # "r|" reads the archive as a stream, so a large package is never unpacked into memory
            with tarfile.open(fileobj=stream, mode="r|") as tf:
                for m in tf:
                    mm = MAN_RE.match(m.name.lstrip("./"))
                    if not mm or not m.isfile() or mm.group(1) not in KEEP_SECTIONS:
                        continue
                    page = f"{mm.group(2)}.{mm.group(3)}"
                    dest = os.path.join(MAN, "roff", safe(page))
                    if os.path.isfile(dest):
                        continue
                    src = gzip.decompress(tf.extractfile(m).read())
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with open(dest, "wb") as fh:
                        fh.write(src)
                    fs.record("kali", dest, f"{MIRROR}/{v['file']}#{m.name}", len(src), "", f"from Kali package {pkg} {v['version']}; see the package's copyright file")
                    pages += 1
            done.add(pkg)
        except Exception as e:  # noqa: BLE001
            fs.failures.append((f"kali package {pkg}", f"{type(e).__name__}: {e}"[:160]))
        if i % 25 == 0 or i == len(todo):
            fs.log(f"debs:   {i} of {len(todo)} packages, {pages} pages extracted")
            with open(done_path, "w", encoding="utf-8") as fh:
                json.dump(sorted(done), fh)
    fs.log(f"debs: extracted {pages} man page sources")


def stage_fallback() -> None:
    """For page sources that convert_man.py could not turn into text, use Debian's rendered copy instead."""
    roff, text = os.path.join(MAN, "roff"), os.path.join(MAN, "text")
    todo = []
    for name in sorted(os.listdir(roff)):
        if os.path.isfile(os.path.join(text, name + ".txt")):
            continue
        with open(os.path.join(roff, name), "rb") as fh:
            macros = [ln.split(None, 1)[0] for ln in fh.read(4000).decode("latin-1").splitlines() if ln.startswith(".") and not ln.startswith('.\\"')]
        if macros and not (macros[0] == ".so" and len(macros) <= 2):  # skip alias stubs
            todo.append(name)
    fs.log(f"fallback: {len(todo)} unconverted pages to look up on manpages.debian.org")
    lic = "Debian manpages: per-package free licences (GPL/BSD/MIT etc.), see each page's source package"
    got, missing = 0, []
    for page in todo:
        name, _, section = page.rpartition(".")
        before = len(fs.failures)
        raw = fs.fetch("kali", f"https://manpages.debian.org/{name}.{section}.en.html", os.path.join(MAN, "html", page + ".html"), lic)
        page_text = fs.html_to_text(raw, "manual-text") if raw else ""
        if len(page_text) > 200:
            fs.write_text(os.path.join(text, page + ".txt"), page_text)
            got += 1
        else:
            del fs.failures[before:]
            if not raw:
                fs.stats["failed"] -= 1
            missing.append(page)
    fs.log(f"fallback: recovered {got}; still without text: {len(missing)} {missing}")


if __name__ == "__main__":
    stages = {"plan": plan, "debian": stage_debian, "debs": stage_debs, "fallback": stage_fallback}
    for s in sys.argv[1:] or ["plan"]:
        if s not in stages:
            sys.exit(f"unknown stage {s!r}; choose from {', '.join(stages)}")
        stages[s]()
    for path, err in fs.failures:
        fs.log(f"  FAILED {path}: {err}")
