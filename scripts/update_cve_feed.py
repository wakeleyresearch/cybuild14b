"""Pull fresh vulnerability data into sources/cve/. Re-run it whenever you want the feeds brought up to date.

Standard library only.

    python scripts/update_cve_feed.py            # since the last run (first run: last 30 days)
    python scripts/update_cve_feed.py --days 120 # force a look-back window

Sources:
  NVD API 2.0             CVE records added or changed in the window -> sources/cve/nvd/*.jsonl (one CVE per line)
  CISA KEV catalogue      vulnerabilities known to be exploited      -> sources/cve/kev/known_exploited_vulnerabilities.json
  CISA advisories (RSS)   latest advisories                          -> sources/cve/cisa-advisories/all.xml
  EPSS scores             probability each CVE is exploited soon     -> sources/cve/epss/epss_scores-current.csv.gz
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "sources", "cve")
STATE = os.path.join(OUT, "state.json")
UA = "dataset-research-client/1.0"  # generic on purpose: no personal details leave this machine
NVD = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_PAUSE = 6.5      # NVD allows 5 requests per 30 seconds without an API key
NVD_MAX_DAYS = 119   # NVD rejects date ranges longer than 120 days
PAGE = 2000


def log(msg: str) -> None:
    print(msg, flush=True)


def http_get(url: str, timeout: int = 120, tries: int = 4) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                if resp.headers.get("Content-Encoding", "").lower() == "gzip":
                    data = gzip.decompress(data)
                return data
        except urllib.error.HTTPError as e:
            if e.code not in (403, 429, 500, 502, 503, 504) or attempt == tries - 1:
                raise
        except Exception:  # noqa: BLE001 - timeouts, resets
            if attempt == tries - 1:
                raise
        time.sleep(10 * (attempt + 1))
    raise RuntimeError("unreachable")


def save(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def iso(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def pull_nvd(start: dt.datetime, end: dt.datetime) -> int:
    """Fetch every CVE whose record changed in [start, end]. Returns the number written."""
    total_written = 0
    win_start = start
    while win_start < end:
        win_end = min(end, win_start + dt.timedelta(days=NVD_MAX_DAYS))
        name = f"nvd_{win_start:%Y%m%dT%H%M}_{win_end:%Y%m%dT%H%M}.jsonl"
        path = os.path.join(OUT, "nvd", name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        index, total, written = 0, None, 0
        with open(path + ".part", "w", encoding="utf-8", newline="\n") as fh:
            while total is None or index < total:
                q = urllib.parse.urlencode({"lastModStartDate": iso(win_start), "lastModEndDate": iso(win_end),
                                            "resultsPerPage": PAGE, "startIndex": index})
                page = json.loads(http_get(f"{NVD}?{q}"))
                total = page.get("totalResults", 0)
                items = page.get("vulnerabilities", [])
                for item in items:
                    fh.write(json.dumps(item.get("cve", item), ensure_ascii=False) + "\n")
                written += len(items)
                index += PAGE
                log(f"  NVD {win_start:%Y-%m-%d}..{win_end:%Y-%m-%d}: {written} of {total}")
                if not items:
                    break
                if index < total:
                    time.sleep(NVD_PAUSE)
        os.replace(path + ".part", path)
        total_written += written
        win_start = win_end
        if win_start < end:
            time.sleep(NVD_PAUSE)
    return total_written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, help="look back this many days instead of resuming from the last run")
    ap.add_argument("--skip-nvd", action="store_true", help="only refresh the CISA feeds")
    args = ap.parse_args()

    state = {}
    if os.path.isfile(STATE):
        with open(STATE, encoding="utf-8") as fh:
            state = json.load(fh)
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0, tzinfo=None)
    if args.days:
        start = now - dt.timedelta(days=args.days)
    elif state.get("nvd_last_end"):
        start = dt.datetime.strptime(state["nvd_last_end"], "%Y-%m-%dT%H:%M:%S")
    else:
        start = now - dt.timedelta(days=30)

    log("CISA Known Exploited Vulnerabilities catalogue")
    try:
        kev = http_get("https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json")
        save(os.path.join(OUT, "kev", "known_exploited_vulnerabilities.json"), kev)
        cat = json.loads(kev)
        log(f"  {cat.get('count', len(cat.get('vulnerabilities', [])))} entries, catalogue version {cat.get('catalogVersion')}")
    except Exception as e:  # noqa: BLE001
        log(f"  FAILED: {type(e).__name__}: {e}")

    log("CISA advisories RSS")
    try:
        rss = http_get("https://www.cisa.gov/cybersecurity-advisories/all.xml")
        save(os.path.join(OUT, "cisa-advisories", "all.xml"), rss)
        log(f"  {rss.count(b'<item>')} items")
    except Exception as e:  # noqa: BLE001
        log(f"  FAILED: {type(e).__name__}: {e}")

    log("EPSS exploit-probability scores (one score per CVE, refreshed daily)")
    for url in ("https://epss.empiricalsecurity.com/epss_scores-current.csv.gz", "https://epss.cyentia.com/epss_scores-current.csv.gz"):
        try:
            epss = http_get(url)
            if epss[:2] != b"\x1f\x8b":  # the client may already have unpacked it
                epss = gzip.compress(epss)
            save(os.path.join(OUT, "epss", "epss_scores-current.csv.gz"), epss)
            lines = gzip.decompress(epss).split(b"\n")
            log(f"  {len(lines) - 3} scores; {lines[0].decode('ascii', 'replace').lstrip('#')}")
            break
        except Exception as e:  # noqa: BLE001
            log(f"  FAILED {url}: {type(e).__name__}: {e}")

    if not args.skip_nvd:
        log(f"NVD: CVE records changed between {start:%Y-%m-%d %H:%M} and {now:%Y-%m-%d %H:%M} UTC")
        n = pull_nvd(start, now)
        state["nvd_last_end"] = now.strftime("%Y-%m-%dT%H:%M:%S")
        state["nvd_last_count"] = n
        os.makedirs(OUT, exist_ok=True)
        with open(STATE, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=1)
        log(f"NVD: wrote {n} CVE records; next run resumes from {state['nvd_last_end']} UTC")


if __name__ == "__main__":
    main()
