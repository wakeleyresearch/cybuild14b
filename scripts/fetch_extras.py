"""Two sources that need special handling, kept out of fetch_sources.py for clarity.

    python scripts/fetch_extras.py

1. MITRE CAR analytics. Fetched as a zip rather than a git checkout: the repository contains file names
   Windows does not allow (in folders we do not need), so only plainly named files are extracted.
2. The Windows PowerShell ISE scripting object-model reference. It was removed from the live PowerShell
   docs repository and now exists only in Microsoft Learn's "previous versions" archive.

Standard library only; re-running skips what is already present.
"""
from __future__ import annotations

import io
import os
import re
import sys
import urllib.parse
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_sources as fs  # noqa: E402

# A path component Windows will accept: none of < > : " / \ | ? * or control characters, no trailing dot or
# space, and not a reserved device name. Anything else in an archive is skipped rather than renamed.
SAFE_NAME = re.compile(r'^(?!(?:CON|PRN|AUX|NUL|COM\d|LPT\d)(?:\.|$))[^<>:"/\\|?*\x00-\x1f]*[^<>:"/\\|?*\x00-\x1f. ]$', re.I)


def mitre_car() -> None:
    dest = os.path.join(fs.OUT, "remediation", "mitre-car")
    raw = fs.fetch("topics", "https://github.com/mitre-attack/car/archive/refs/heads/master.zip",
                   os.path.join(dest, "car-master.zip"), "MITRE CAR: Apache License 2.0 (see LICENSE in the archive)")
    if not raw:
        return
    kept = skipped = 0
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for name in zf.namelist():
            parts = name.split("/")[1:]  # drop the top-level folder
            if not parts or name.endswith("/"):
                continue
            if not (parts[0] == "analytics" or (parts[0] == "docs" and name.endswith(".md")) or parts[0] in ("LICENSE", "README.md")):
                continue
            if any(p in ("", ".", "..") or not SAFE_NAME.match(p) for p in parts):
                skipped += 1
                continue
            target = os.path.join(dest, *parts)
            if not os.path.isfile(target):
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "wb") as fh:
                    fh.write(zf.read(name))
            kept += 1
    fs.log(f"mitre-car: {kept} files present; {skipped} skipped for unusual names")


def chronicle_rules() -> None:
    """Google SecOps (Chronicle) detection rules, fetched as a zip for the same reason as MITRE CAR."""
    dest = os.path.join(fs.OUT, "logging", "chronicle-detection-rules")
    raw = fs.fetch("depth", "https://github.com/chronicle/detection-rules/archive/refs/heads/main.zip",
                   os.path.join(dest, "detection-rules-main.zip"), "chronicle/detection-rules: Apache License 2.0 (see LICENSE in the archive)")
    if not raw:
        return
    kept = skipped = 0
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for name in zf.namelist():
            parts = name.split("/")[1:]
            if not parts or name.endswith("/") or not name.endswith((".yaral", ".md", "LICENSE")):
                continue
            if any(p in ("", ".", "..") or not SAFE_NAME.match(p) for p in parts):
                skipped += 1
                continue
            target = os.path.join(dest, *parts)
            if len(target) > 240:
                skipped += 1
                continue
            if not os.path.isfile(target):
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "wb") as fh:
                    fh.write(zf.read(name))
            kept += 1
    fs.log(f"chronicle-detection-rules: {kept} files present; {skipped} skipped for unusual or over-long names")


def systemd_docs() -> None:
    """systemd man-page sources and docs via zip: the repository's test data has file names Windows rejects."""
    dest = os.path.join(fs.OUT, "services", "systemd")
    raw = fs.fetch("services", "https://github.com/systemd/systemd/archive/refs/heads/main.zip",
                   os.path.join(dest, "systemd-main.zip"), "systemd: LGPL-2.1-or-later (see LICENSES in the archive)")
    if not raw:
        return
    kept = skipped = 0
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for name in zf.namelist():
            parts = name.split("/")[1:]
            if not parts or name.endswith("/"):
                continue
            wanted = (parts[0] == "man" and len(parts) == 2 and name.endswith(".xml")) or (parts[0] == "docs" and name.endswith(".md")) \
                or parts[0] in ("README", "README.md", "LICENSE.LGPL2.1")
            if not wanted:
                continue
            if any(p in ("", ".", "..") or not SAFE_NAME.match(p) for p in parts):
                skipped += 1
                continue
            target = os.path.join(dest, *parts)
            if not os.path.isfile(target):
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "wb") as fh:
                    fh.write(zf.read(name))
            kept += 1
    fs.log(f"systemd: {kept} files present; {skipped} skipped for unusual names")


def rosetta_lua() -> None:
    """Lua solutions and their task descriptions from Rosetta Code.

    The repository holds every language and has folder names Windows rejects, so a checkout fails. The partial clone
    in sources/lua/rosetta-code still has the file listing; this reads it and downloads only the Lua files directly.
    """
    import subprocess
    repo = os.path.join(fs.OUT, "lua", "rosetta-code")
    if not os.path.isdir(os.path.join(repo, ".git")):
        subprocess.run(["git", "clone", "--quiet", "--depth", "1", "--filter=blob:none", "--no-checkout",
                        "https://github.com/acmeism/RosettaCodeData.git", repo], check=True, timeout=1800)
    head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60).stdout.strip()
    listing = subprocess.run(["git", "-C", repo, "ls-tree", "-r", "--name-only", "-z", "HEAD", "Task"], capture_output=True, timeout=900).stdout
    names = [n for n in listing.decode("utf-8", "replace").split("\0") if re.match(r"^Task/[^/]+/(Lua/[^/]+\.lua|00-TASK\.txt)$", n)]
    with_lua = {n.split("/")[1] for n in names if "/Lua/" in n}
    wanted = [n for n in names if n.split("/")[1] in with_lua and all(SAFE_NAME.match(p) for p in n.split("/"))]
    fs.HOST_DELAY["raw.githubusercontent.com"] = 0.05
    fs.log(f"rosetta-lua: {len(with_lua)} tasks have a Lua solution; {len(wanted)} files to fetch at revision {head[:10]}")
    got = 0
    for i, n in enumerate(wanted, 1):
        url = f"https://raw.githubusercontent.com/acmeism/RosettaCodeData/{head}/" + urllib.parse.quote(n)
        if fs.fetch("round2", url, os.path.join(repo, *n.split("/")), "Rosetta Code: GNU Free Documentation License 1.2"):
            got += 1
        if i % 500 == 0:
            fs.log(f"rosetta-lua:   {i} of {len(wanted)}")
    fs.log(f"rosetta-lua: {got} of {len(wanted)} files present")


def ise_object_model() -> None:
    fs.HOST_DELAY["learn.microsoft.com"] = 0.7
    base = "https://learn.microsoft.com/en-us/previous-versions/powershell/scripting/windows-powershell/ise/object-model/"
    queue = [base + s for s in (
        "the-ise-object-model-hierarchy", "purpose-of-the-windows-powershell-ise-scripting-object-model",
        "the-isemenuitem-object", "the-isemenuitemcollection-object", "the-isesnippetobject", "the-isesnippetcollection-object",
        "other-useful-scripting-objects")]
    dest = os.path.join(fs.OUT, "powershell", "ise-object-model")
    seen: set[str] = set()
    saved = 0
    while queue and len(seen) < 60:
        url = queue.pop(0)
        slug = url.rstrip("/").rsplit("/", 1)[1].lower()
        if slug in seen:
            continue
        seen.add(slug)
        raw = fs.fetch("topics", url + "?view=powershell-7.3", os.path.join(dest, "html", slug + ".html"), "Microsoft Learn documentation: CC BY 4.0")
        if not raw:
            continue
        text = fs.html_to_text(raw, "content")
        if len(text) < 400:
            text = fs.html_to_text(raw)
        fs.write_text(os.path.join(dest, slug + ".txt"), text)
        saved += 1
        for href in re.findall(r'href="([^"#?]+)', raw.decode("utf-8", "replace")):
            full = urllib.parse.urljoin(url, href)
            if "/ise/object-model/" in full and full.startswith("https://learn.microsoft.com/"):
                queue.append(full.replace("/en-us/powershell/scripting/", "/en-us/previous-versions/powershell/scripting/"))
    fs.log(f"ise-object-model: {saved} pages present")


def windows_audit_events(limit: int = 900) -> None:
    """Windows security auditing reference: every audit policy subcategory and every security event ID page.

    The source repository is no longer public, so this walks Microsoft Learn's archived copy, following
    only links that stay inside the auditing section.
    """
    fs.HOST_DELAY["learn.microsoft.com"] = 0.7
    base = "https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-10/security/threat-protection/auditing/"
    queue = [base + s for s in ("advanced-security-audit-policy-settings", "security-auditing-overview", "advanced-security-auditing-faq",
                                "basic-security-audit-policy-settings", "appendix-a-security-monitoring-recommendations-for-many-audit-events",
                                "other-events", "security-auditing-reference", "basic-security-audit-policies", "advanced-security-audit-policies")]
    dest = os.path.join(fs.OUT, "logging", "windows-security-auditing")
    seen: set[str] = set()
    saved = 0
    while queue and len(seen) < limit:
        url = queue.pop(0)
        slug = url.rstrip("/").rsplit("/", 1)[1].lower()
        if slug in seen or not SAFE_NAME.match(slug):
            continue
        seen.add(slug)
        before = len(fs.failures)
        raw = fs.fetch("depth", url, os.path.join(dest, "html", slug + ".html"), "Microsoft Learn documentation: CC BY 4.0")
        if not raw:
            del fs.failures[before:]  # dead links inside the archive are expected
            fs.stats["failed"] -= 1
            continue
        text = fs.html_to_text(raw, "content")
        if len(text) < 400:
            text = fs.html_to_text(raw)
        fs.write_text(os.path.join(dest, slug + ".txt"), text)
        saved += 1
        if saved % 100 == 0:
            fs.log(f"windows-security-auditing: {saved} pages")
        for href in re.findall(r'href="([^"#?]+)', raw.decode("utf-8", "replace")):
            if href.startswith(("http", "/", "mailto:", "..")) and "/threat-protection/auditing/" not in href:
                continue
            full = urllib.parse.urljoin(url, href)
            if full.startswith(base) and full.rstrip("/").rsplit("/", 1)[1].lower() not in seen:
                queue.append(full)
    fs.log(f"windows-security-auditing: {saved} pages present")


if __name__ == "__main__":
    mitre_car()
    chronicle_rules()
    systemd_docs()
    ise_object_model()
    windows_audit_events()
    for path, err in fs.failures:
        fs.log(f"  FAILED {path}: {err}")
