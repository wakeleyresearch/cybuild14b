"""Extract text from the PDFs under sources/ into data/pdf_text/ (sources/ itself is left untouched).

Needs pypdf, supplied without touching the project's .venv:

    uv run --no-project --with pypdf python scripts/extract_pdfs.py

One .txt per PDF, pages separated by form feeds. Already-extracted files are skipped. Repeated page headers and
footers (lines that recur on most pages) are removed, and words hyphenated across line ends are rejoined.
"""
from __future__ import annotations

import collections
import os
import re
import sys
import time

from pypdf import PdfReader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "sources")
OUT = os.path.join(ROOT, "data", "pdf_text")


def extract(path: str) -> tuple[str, int]:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - one unreadable page should not lose the document
            pages.append("")
    # lines that appear on at least 40% of pages are running headers or footers
    seen = collections.Counter()
    for p in pages:
        for ln in {re.sub(r"\d+", "#", x.strip()) for x in p.split("\n") if x.strip()}:
            seen[ln] += 1
    limit = max(4, int(len(pages) * 0.4))
    furniture = {ln for ln, n in seen.items() if n >= limit and len(ln) < 120}
    out = []
    for p in pages:
        keep = [x.rstrip() for x in p.split("\n") if re.sub(r"\d+", "#", x.strip()) not in furniture]
        text = "\n".join(keep)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # re-join words split across lines
        out.append(text.strip())
    return "\f".join(out), len(pages)


def main() -> None:
    todo = []
    for dp, dn, fn in os.walk(SRC):
        dn[:] = [d for d in dn if d != ".git"]
        for f in fn:
            if f.lower().endswith(".pdf"):
                src = os.path.join(dp, f)
                dest = os.path.join(OUT, os.path.relpath(src, SRC))[:-4] + ".txt"
                if not os.path.isfile(dest):
                    todo.append((src, dest))
    print(f"{len(todo)} PDFs to extract", flush=True)
    t0 = time.time()
    failed = []
    for i, (src, dest) in enumerate(sorted(todo), 1):
        try:
            text, n = extract(src)
        except Exception as e:  # noqa: BLE001
            failed.append((os.path.relpath(src, SRC), f"{type(e).__name__}: {str(e)[:100]}"))
            continue
        if len(text) < 500:
            failed.append((os.path.relpath(src, SRC), f"almost no text ({len(text)} chars from {n} pages): probably scanned images"))
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print(f"  {i}/{len(todo)}  {n:4d} pages  {len(text) / 1e6:5.2f} MB  {os.path.relpath(src, SRC)}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"done in {time.time() - t0:.0f}s; {len(failed)} failed")
    for path, why in failed:
        print(f"  FAILED {path}: {why}")


if __name__ == "__main__":
    sys.exit(main())
