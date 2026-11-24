"""Convert man-page sources (roff) into plain text.

Needs pandoc, supplied by the pypandoc-binary package without touching the project's .venv:

    uv run --no-project --with pypandoc-binary python scripts/convert_man.py

Converts:
  sources/kali/man-pages/roff/*                         -> sources/kali/man-pages/text/*.txt
  sources/man-pages/linux-man-pages/extracted/man*/*    -> sources/man-pages/linux-man-pages/text/*.txt
Already-converted pages are skipped. Alias stubs (files that only say ".so other-page") are skipped.
"""
from __future__ import annotations

import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pypandoc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "sources")
JOBS = [
    (os.path.join(SRC, "kali", "man-pages", "roff"), os.path.join(SRC, "kali", "man-pages", "text")),
    (os.path.join(SRC, "man-pages", "linux-man-pages", "extracted"), os.path.join(SRC, "man-pages", "linux-man-pages", "text")),
]


def decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def convert(src: str, dest: str) -> str:
    """Returns 'ok', 'alias', 'empty' or an error string."""
    with open(src, "rb") as fh:
        text = decode(fh.read())
    macros = [ln.split(None, 1)[0] for ln in text.splitlines() if ln.startswith(".") and not ln.startswith('.\\"')]
    if not macros:
        return "empty"
    if macros[0] == ".so" and len(macros) <= 2:
        return "alias"
    fmt = "mdoc" if any(m in (".Dd", ".Sh", ".Nm") for m in macros[:40]) else "man"
    try:
        out = pypandoc.convert_text(text, "plain", format=fmt, extra_args=["--wrap=none"])
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {str(e)[:120]}"
    out = re.sub(r"\n{3,}", "\n\n", out.replace("\r\n", "\n")).strip() + "\n"
    if len(out) < 80:
        return "empty"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(out)
    return "ok"


def main() -> None:
    work = []
    for src_dir, dest_dir in JOBS:
        if not os.path.isdir(src_dir):
            continue
        for dp, _, fn in os.walk(src_dir):
            for f in fn:
                dest = os.path.join(dest_dir, f + ".txt")
                if not os.path.isfile(dest):
                    work.append((os.path.join(dp, f), dest))
    print(f"pandoc {pypandoc.get_pandoc_version()}; {len(work)} pages to convert", flush=True)
    t0 = time.time()
    counts: dict[str, int] = {}
    errors: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, ((src, _), res) in enumerate(zip(work, pool.map(lambda w: convert(*w), work)), 1):
            key = res if res in ("ok", "alias", "empty") else "error"
            counts[key] = counts.get(key, 0) + 1
            if key == "error":
                errors.append((os.path.basename(src), res))
            if i % 500 == 0:
                print(f"  {i} of {len(work)}  {counts}", flush=True)
    print(f"done in {time.time() - t0:.0f}s: {counts}", flush=True)
    for name, err in errors[:25]:
        print(f"  ERROR {name}: {err}")
    if len(errors) > 25:
        print(f"  ... and {len(errors) - 25} more")


if __name__ == "__main__":
    sys.exit(main())
