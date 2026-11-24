"""Build the zip that goes to the laptop for the lookup-library work.

    python -I scripts/make_lookup_package.py

Packs sources/ (without git remnants, downloaded archives and raw copies that already exist as text), scripts/,
the held-out evaluation set, and the hand-off documents from handoff/ into handoff/neuralproj-lookup-package.zip.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "handoff", "neuralproj-lookup-package.zip")
TOP = "neuralproj-lookup"
SKIP_DIRS = {".git", ".cache", "__pycache__"}
# raw material that the package already carries as text, and indexes only the fetch scripts need
SKIP_PATHS = ("kali/index/", "kali/man-pages/roff/", "kali/man-pages/html/", "man-pages/debian/html/", "man-pages/linux-man-pages/extracted/",
              "powershell/ise-object-model/html/", "logging/windows-security-auditing/html/")
SKIP_EXT = (".zip", ".tar.xz", ".tar.gz", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".woff", ".woff2", ".ttf", ".mp4", ".part")


def main() -> None:
    t0 = time.time()
    n = raw = 0
    skipped = 0
    with zipfile.ZipFile(OUT + ".part", "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zf:
        for name in ("README.txt", "CLAUDE.md", "PROMPT.md"):
            zf.write(os.path.join(ROOT, "handoff", name), f"{TOP}/{name}")
        zf.write(os.path.join(ROOT, "data", "eval.jsonl"), f"{TOP}/eval/eval.jsonl")
        keys = os.path.join(ROOT, "data", "eval_keys.jsonl")      # strict answer keys: the exact file or record each question is answered by
        if os.path.isfile(keys):
            zf.write(keys, f"{TOP}/eval/eval_keys.jsonl")
        for f in sorted(os.listdir(os.path.join(ROOT, "scripts"))):
            if f.endswith(".py"):
                zf.write(os.path.join(ROOT, "scripts", f), f"{TOP}/scripts/{f}")
        src = os.path.join(ROOT, "sources")
        for dp, dn, fn in os.walk(src):
            dn[:] = sorted(d for d in dn if d not in SKIP_DIRS)
            rel_dir = os.path.relpath(dp, src).replace("\\", "/")
            for f in sorted(fn):
                rel = f if rel_dir == "." else f"{rel_dir}/{f}"
                if rel.lower().endswith(SKIP_EXT) and not rel.endswith("epss_scores-current.csv.gz") or any(rel.startswith(p) for p in SKIP_PATHS):
                    skipped += 1
                    continue
                path = os.path.join(dp, f)
                try:
                    zf.write(path, f"{TOP}/sources/{rel}")
                except OSError as e:
                    skipped += 1
                    print(f"  could not add {rel}: {e}", flush=True)
                    continue
                n += 1
                raw += os.path.getsize(path)
                if n % 25000 == 0:
                    print(f"  {n} files, {raw / 2**30:.2f} GiB read, {time.time() - t0:.0f}s", flush=True)
    os.replace(OUT + ".part", OUT)
    h = hashlib.sha256()
    with open(OUT, "rb") as fh:
        for block in iter(lambda: fh.read(4 << 20), b""):
            h.update(block)
    print(f"packed {n} source files ({raw / 2**30:.2f} GiB) into {OUT}")
    print(f"skipped {skipped} files (git remnants, archives, images, raw copies)")
    print(f"zip size {os.path.getsize(OUT) / 2**30:.2f} GiB; sha256 {h.hexdigest()}; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
