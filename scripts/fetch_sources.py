"""Collect openly available reference material into sources/ for the cybersecurity corpus.

Standard library only. Safe to re-run: files that already exist are skipped.

    python scripts/fetch_sources.py              # everything
    python scripts/fetch_sources.py man legal    # only some groups

Groups: man, references, legal, tooldocs, topics, depth, services
Every fetched file is recorded in sources/manifest.jsonl with its URL, size, hash and licence note.
Downloaded content is data: nothing fetched here is ever imported or executed.
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from html.parser import HTMLParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "sources")
MANIFEST = os.path.join(OUT, "manifest.jsonl")
UA = "dataset-research-client/1.0"  # generic on purpose: no personal details leave this machine
TODAY = dt.date.today().isoformat()

HOST_DELAY = {"manpages.debian.org": 0.6, "www.rfc-editor.org": 0.3, "www.govinfo.gov": 0.5,
              "nvlpubs.nist.gov": 0.5, "www.federalregister.gov": 0.5, "www.ecfr.gov": 0.5}
_last_hit: dict[str, float] = {}
stats = {"fetched": 0, "skipped": 0, "failed": 0, "bytes": 0}
failures: list[tuple[str, str]] = []


def log(msg: str) -> None:
    # Licence files sometimes start with characters the Windows console cannot print; never let that abort a fetch.
    print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def record(group: str, path: str, url: str, data_len: int, sha: str, licence: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(MANIFEST, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"group": group, "path": os.path.relpath(path, OUT).replace("\\", "/"), "url": url,
                             "bytes": data_len, "sha256": sha, "fetched": TODAY, "licence": licence}) + "\n")


def http_get(url: str, timeout: int = 90) -> bytes:
    host = urllib.parse.urlsplit(url).netloc
    wait = HOST_DELAY.get(host, 0.25) - (time.time() - _last_hit.get(host, 0))
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip", "Accept": "*/*"})
    last: Exception | None = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                if resp.headers.get("Content-Encoding", "").lower() == "gzip":
                    data = gzip.decompress(data)
            _last_hit[host] = time.time()
            return data
        except urllib.error.HTTPError as e:
            _last_hit[host] = time.time()
            if e.code < 500 and e.code != 429:
                raise
            last = e
        except Exception as e:  # noqa: BLE001 - timeouts, resets
            last = e
        time.sleep(4)
    raise last  # type: ignore[misc]


def fetch(group: str, urls: str | list[str], dest: str, licence: str) -> bytes | None:
    """Download the first URL that works to dest. Returns the bytes, or None on failure."""
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        stats["skipped"] += 1
        with open(dest, "rb") as fh:
            return fh.read()
    err = ""
    for url in ([urls] if isinstance(urls, str) else urls):
        try:
            data = http_get(url)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            continue
        if not data:
            err = "empty response"
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".part"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, dest)
        record(group, dest, url, len(data), hashlib.sha256(data).hexdigest(), licence)
        stats["fetched"] += 1
        stats["bytes"] += len(data)
        return data
    stats["failed"] += 1
    failures.append((os.path.relpath(dest, OUT).replace("\\", "/"), err))
    return None


def sparse_clone(group: str, repo: str, dest: str, keep: list[str], licence: str) -> None:
    """Shallow clone that only checks out the listed folders."""
    if os.path.isdir(os.path.join(dest, ".git")):
        stats["skipped"] += 1
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    base = ["git", "-c", "core.longpaths=true", "-c", "core.autocrlf=false"]
    try:
        subprocess.run(base + ["clone", "--depth", "1", "--filter=blob:none", "--sparse", repo, dest],
                       check=True, capture_output=True, text=True, timeout=900)
        subprocess.run(base + ["-C", dest, "sparse-checkout", "set", *keep], check=True, capture_output=True, text=True, timeout=900)
        head = subprocess.run(["git", "-C", dest, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        size = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fn in os.walk(dest) if ".git" not in dp.split(os.sep) for f in fn)
        record(group, dest, f"{repo}@{head}", size, "", licence)
        stats["fetched"] += 1
        stats["bytes"] += size
    except Exception as e:  # noqa: BLE001
        stats["failed"] += 1
        msg = getattr(e, "stderr", "") or str(e)
        failures.append((os.path.relpath(dest, OUT).replace("\\", "/"), msg.strip()[:200]))


class _Text(HTMLParser):
    """Minimal HTML -> text, optionally limited to the element with a given class."""
    BLOCK = {"p", "br", "div", "section", "h1", "h2", "h3", "h4", "li", "dt", "dd", "tr", "pre", "table", "blockquote"}
    SKIP = {"script", "style", "nav", "footer", "head"}

    def __init__(self, only_class: str | None = None):
        super().__init__(convert_charrefs=True)
        self.only, self.depth, self.skip, self.out = only_class, 0, 0, []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        if self.only:
            if self.depth:
                if tag not in ("br", "img", "hr", "input", "meta", "link"):
                    self.depth += 1
            elif self.only in (dict(attrs).get("class") or "").split():
                self.depth = 1
        if tag in self.BLOCK:
            self.out.append("\n")
        if tag in ("h1", "h2", "h3"):
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        if self.only and self.depth and tag not in ("br", "img", "hr", "input", "meta", "link"):
            self.depth -= 1
        if tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip and (not self.only or self.depth):
            self.out.append(data)


def html_to_text(raw: bytes, only_class: str | None = None) -> str:
    p = _Text(only_class)
    p.feed(raw.decode("utf-8", "replace"))
    text = "".join(p.out).replace("\xa0", " ").replace("¶", "")  # drop heading anchor marks
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


# --------------------------------------------------------------------------- man pages
MAN_PAGES = """
nmap.1 ncat.1 nping.1 masscan.8 zmap.1 hping3.8 arp-scan.1 netdiscover.8 p0f.1 nbtscan.1 onesixtyone.1 snmpwalk.1
tcpdump.8 tshark.1 wireshark.1 dumpcap.1 editcap.1 mergecap.1 capinfos.1 text2pcap.1 ngrep.8 tcpflow.1 tcpreplay.1 ssldump.1
nikto.1 sqlmap.1 gobuster.1 ffuf.1 dirb.1 wfuzz.1 whatweb.1 sslscan.1 testssl.1 wapiti.1 skipfish.1 cewl.1
hydra.1 medusa.1 ncrack.1 patator.1 john.8 hashcat.1 hashid.1 crunch.1 ophcrack.1 fcrackzip.1 pdfcrack.1
aircrack-ng.1 airodump-ng.8 aireplay-ng.8 airmon-ng.8 macchanger.1 ettercap.8 arpspoof.8 dsniff.8 bettercap.1 reaver.1
smbclient.1 rpcclient.1 smbmap.1 enum4linux.1 ldapsearch.1 nmblookup.1 net.8 kinit.1 klist.1
openssl.1ssl openssl-x509.1ssl openssl-s_client.1ssl openssl-req.1ssl openssl-enc.1ssl openssl-dgst.1ssl openssl-genpkey.1ssl
gpg.1 gpgv.1 age.1 sha256sum.1 md5sum.1 base64.1 certbot.1 keytool.1 update-ca-certificates.8 stunnel4.8
ssh.1 sshd.8 ssh-keygen.1 ssh-agent.1 ssh-add.1 ssh-copy-id.1 scp.1 sftp.1 ssh_config.5 sshd_config.5 rsync.1
sudo.8 sudoers.5 visudo.8 su.1 doas.1 pkexec.1 passwd.1 passwd.5 shadow.5 useradd.8 usermod.8 chage.1 login.defs.5
faillock.8 pwquality.conf.5 pam.conf.5 pam_unix.8 pam_faillock.8 google-authenticator.1 oathtool.1
curl.1 wget.1 nc.1 socat.1 dig.1 nslookup.1 host.1 whois.1 traceroute.1 mtr.8 ping.8 ss.8 netstat.8 lsof.8 arp.8
ip.8 ip-route.8 ip-netns.8 ip-link.8 tc.8 bridge.8 ethtool.8 iw.8 nmcli.1 resolvectl.1
iptables.8 ip6tables.8 iptables-save.8 nft.8 ufw.8 firewall-cmd.1 fail2ban-client.1 sshguard.8
openvpn.8 wg.8 wg-quick.8 tor.1 proxychains4.1 torsocks.1
ps.1 top.1 htop.1 kill.1 pkill.1 strace.1 ltrace.1 gdb.1 objdump.1 readelf.1 nm.1 strings.1 file.1 xxd.1 hexdump.1 ldd.1
binwalk.1 radare2.1 rabin2.1 rasm2.1 upx.1 patchelf.1 checksec.1 valgrind.1 afl-fuzz.8 nasm.1 objcopy.1 gcc.1
yara.1 clamscan.1 freshclam.1 chkrootkit.1 rkhunter.8 lynis.8 aide.1 debsums.1 tripwire.8 oscap.8
auditd.8 auditctl.8 ausearch.8 aureport.8 audit.rules.7 journalctl.1 systemctl.1 systemd.unit.5 systemd.service.5 systemd.exec.5
systemd-analyze.1 loginctl.1 logrotate.8 rsyslogd.8 rsyslog.conf.5 logger.1 last.1 lastlog.8 who.1 w.1 dmesg.1
setenforce.8 getenforce.8 semanage.8 restorecon.8 sestatus.8 audit2allow.1 aa-status.8 apparmor.7 apparmor_parser.8 aa-genprof.8
capsh.1 getcap.8 setcap.8 unshare.1 nsenter.1 chroot.8 firejail.1 bwrap.1 sysctl.8 sysctl.conf.5 modprobe.8 lsmod.8
chmod.1 chown.1 chattr.1 lsattr.1 setfacl.1 getfacl.1 umask.2 find.1 grep.1 gawk.1 sed.1 xargs.1 tar.1 dd.1 shred.1
mount.8 losetup.8 cryptsetup.8 lsblk.8 fdisk.8 testdisk.8 photorec.8 foremost.8 scalpel.1 fls.1 icat.1 mmls.1 fsstat.1
bulk_extractor.1 hashdeep.1 ssdeep.1 dc3dd.1 ewfacquire.1 exiftool.1p steghide.1 volatility.1
docker.1 docker-run.1 podman.1 runc.8 kubectl.1 ansible.1 ansible-playbook.1 git.1 apt.8 apt-key.8 dpkg.1 unattended-upgrade.8
crontab.1 crontab.5 cron.8 at.1 bash.1 dash.1 tmux.1 screen.1 jq.1 scapy.1 snort.8 suricata.1 zeek.8 nfdump.1
sysdig.8 bpftrace.8 perf.1 iftop.8 nethogs.8 vnstat.1 iptraf-ng.8 tcptraceroute.8 nikto.1 msfvenom.1 searchsploit.1
""".split()


def group_man() -> None:
    lic_dfsg = "Debian manpages: per-package free licences (GPL/BSD/MIT etc.), see each page's source package"
    base = os.path.join(OUT, "man-pages")

    log("man: Linux man-pages project tarball")
    tar_path = os.path.join(base, "linux-man-pages", "man-pages-6.20.tar.xz")
    data = fetch("man", "https://mirrors.edge.kernel.org/pub/linux/docs/man-pages/man-pages-6.20.tar.xz", tar_path,
                 "Linux man-pages project: per-page free licences (GPL, BSD, verbatim-copying)")
    ext_dir = os.path.join(base, "linux-man-pages", "extracted")
    if data and not os.path.isdir(ext_dir):
        n = 0
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:xz") as tf:
            for m in tf:
                parts = m.name.split("/")
                if not m.isfile() or ".." in parts or not any(re.fullmatch(r"man\d\w*", p) for p in parts):
                    continue
                rel = "/".join(parts[parts.index(next(p for p in parts if re.fullmatch(r"man\d\w*", p))):])
                rel = re.sub(r'[<>:"|?*]', "_", rel)
                target = os.path.join(ext_dir, *rel.split("/"))
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "wb") as fh:
                    fh.write(tf.extractfile(m).read())
                n += 1
        log(f"man:   extracted {n} page sources")

    log("man: tldr-pages (English)")
    zip_path = os.path.join(base, "tldr", "tldr-pages.en.zip")
    data = fetch("man", "https://github.com/tldr-pages/tldr/releases/latest/download/tldr-pages.en.zip", zip_path, "tldr-pages: CC BY 4.0")
    tldr_dir = os.path.join(base, "tldr", "pages")
    if data and not os.path.isdir(tldr_dir):
        n = 0
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                parts = name.split("/")
                if name.endswith("/") or ".." in parts or not name.endswith(".md"):
                    continue
                target = os.path.join(tldr_dir, *[re.sub(r'[<>:"|?*]', "_", p) for p in parts])
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "wb") as fh:
                    fh.write(zf.read(name))
                n += 1
        log(f"man:   extracted {n} tldr pages")

    log(f"man: {len(MAN_PAGES)} tool pages from manpages.debian.org")
    ok = 0
    for i, page in enumerate(dict.fromkeys(MAN_PAGES), 1):
        name, _, section = page.rpartition(".")
        html_path = os.path.join(base, "debian", "html", f"{page}.html")
        raw = fetch("man", [f"https://manpages.debian.org/{name}.{section}.en.html", f"https://manpages.debian.org/{name}.en.html"], html_path, lic_dfsg)
        if raw:
            txt_path = os.path.join(base, "debian", "text", f"{page}.txt")
            if not os.path.isfile(txt_path):
                text = html_to_text(raw, "manual-text")
                if len(text) > 200:
                    write_text(txt_path, text)
            ok += 1
        if i % 50 == 0:
            log(f"man:   {i} tried, {ok} found")
    log(f"man: tool pages found {ok} of {len(dict.fromkeys(MAN_PAGES))}")


# --------------------------------------------------------------------------- open references
NIST = """
SP.800-53r5 SP.800-53Ar5 SP.800-61r3 SP.800-61r2 SP.800-115 SP.800-150 SP.800-171r3 SP.800-207 SP.800-190 SP.800-82r3
SP.800-30r1 SP.800-37r2 SP.800-40r4 SP.800-86 SP.800-92 SP.800-94 SP.800-123 SP.800-63-4 SP.800-63-3 SP.800-218
SP.800-161r1-upd1 SP.800-184 SP.800-83r1 SP.800-81-2 SP.800-52r2 SP.800-57pt1r5 SP.800-41r1 SP.800-160v1r1 SP.800-160v2r1
SP.800-204 SP.800-144 SP.800-145 SP.800-125 SP.800-153 SP.800-177r1 SP.800-88r1 SP.800-34r1 SP.800-84 SP.800-12r1
SP.800-137 SP.800-181r1 SP.800-213 SP.800-128 SP.800-39 SP.800-55v1 SP.800-95 SP.800-113 SP.800-124r2 SP.800-193
SP.800-210 SP.800-215 SP.1800-25 SP.800-47r1 SP.800-216
""".split()
NIST_OTHER = {
    "NIST.CSWP.29 (Cybersecurity Framework 2.0)": "https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf",
    "NIST.AI.100-1 (AI Risk Management Framework)": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
    "NIST.AI.100-2e2025 (Adversarial ML taxonomy)": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-2e2025.pdf",
    "NIST.IR.8259 (IoT device cybersecurity)": "https://nvlpubs.nist.gov/nistpubs/ir/2020/NIST.IR.8259.pdf",
    "NIST.IR.8286 (Cybersecurity and enterprise risk)": "https://nvlpubs.nist.gov/nistpubs/ir/2020/NIST.IR.8286.pdf",
}
RFCS = """1034 1035 2196 2818 3552 3986 4033 4034 4035 4120 4251 4252 4253 4254 4301 4303 4949 5246 5280 5321 6066 6238 6454
6455 6749 6750 6797 6960 7208 7235 7469 7489 7515 7516 7517 7518 7519 7525 7540 7636 8032 8200 8252 8259 8314 8446 8484
8555 8705 8725 8996 9000 9001 9110 9112 9113 9114 9116 9325 9449 9700""".split()


def group_references() -> None:
    base = os.path.join(OUT, "references")
    log("references: OWASP Web Security Testing Guide and Cheat Sheet Series")
    sparse_clone("references", "https://github.com/OWASP/wstg.git", os.path.join(base, "owasp-wstg"), ["document"], "OWASP WSTG: CC BY-SA 4.0")
    sparse_clone("references", "https://github.com/OWASP/CheatSheetSeries.git", os.path.join(base, "owasp-cheatsheets"), ["cheatsheets"], "OWASP Cheat Sheet Series: CC BY-SA 4.0")

    log("references: MITRE ATT&CK STIX bundles")
    for dom in ("enterprise-attack", "ics-attack", "mobile-attack"):
        fetch("references", f"https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/{dom}/{dom}.json",
              os.path.join(base, "mitre-attack", f"{dom}.json"), "MITRE ATT&CK: free to use under the ATT&CK Terms of Use, with attribution")

    log("references: CyBOK 1.1")
    fetch("references", "https://www.cybok.org/wp-content/uploads/CyBOK_v1.1.0.pdf", os.path.join(base, "cybok", "CyBOK_v1.1.0.pdf"),
          "CyBOK: Open Government Licence v3.0")

    log(f"references: {len(NIST) + len(NIST_OTHER)} NIST publications")
    for doc in NIST:
        legacy = "nistspecialpublication" + doc.split(".", 1)[1]
        fetch("references", [f"https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.{doc}.pdf",
                             f"https://nvlpubs.nist.gov/nistpubs/Legacy/SP/{legacy}.pdf"],
              os.path.join(base, "nist", f"NIST.{doc}.pdf"), "NIST: US government work, public domain")
    for label, url in NIST_OTHER.items():
        fetch("references", url, os.path.join(base, "nist", url.rsplit("/", 1)[1]), "NIST: US government work, public domain")

    log(f"references: {len(RFCS)} RFCs")
    for n in RFCS:
        fetch("references", f"https://www.rfc-editor.org/rfc/rfc{n}.txt", os.path.join(base, "rfc", f"rfc{n}.txt"),
              "RFC: IETF Trust licence, copying and distribution permitted")


# --------------------------------------------------------------------------- legal and regulatory
FR_SEARCHES = [  # (search phrase, document types)
    ("Cybersecurity Risk Management, Strategy, Governance, and Incident Disclosure", ["RULE", "PRORULE"]),
    ("Cyber Incident Reporting for Critical Infrastructure Act", ["RULE", "PRORULE"]),
    ("Computer-Security Incident Notification Requirements for Banking Organizations", ["RULE"]),
    ("Standards for Safeguarding Customer Information", ["RULE"]),
    ("Regulation S-P: Privacy of Consumer Financial Information and Safeguarding Customer Information", ["RULE"]),
    ("Cybersecurity Information Sharing Act of 2015", ["NOTICE", "RULE"]),
]
ECFR = [  # (title, query string, file name, description)
    (17, "part=229&section=229.106", "17cfr229.106-item106.html", "Regulation S-K Item 106"),
    (45, "part=164&subpart=D", "45cfr164-subpartD-breach-notification.html", "HIPAA breach notification rule"),
    (16, "part=314", "16cfr314-ftc-safeguards.html", "FTC Safeguards Rule"),
    (12, "part=53", "12cfr53-occ-incident-notification.html", "OCC computer-security incident notification"),
    (17, "part=248&subpart=A", "17cfr248-regulation-s-p.html", "Regulation S-P"),
]


def group_legal() -> None:
    base = os.path.join(OUT, "legal")
    gov = "US government work, public domain"

    log("legal: CRS brief IF12959 (current and May 2026 versions)")
    for name, url in {
        "IF12959_current.html": "https://www.everycrsreport.com/reports/IF12959.html",
        "IF12959_2026-09-03.html": "https://www.everycrsreport.com/files/2026-09-03_IF12959_e88b818ccb3aca205dcf1ba233b57cd55207205b.html",
        "IF12959_2026-09-03.pdf": "https://www.everycrsreport.com/files/2026-09-03_IF12959_6e04cde8233e99f02ee3c173a029b5f915605040.pdf",
        "IF12959_2026-05-08.html": "https://www.everycrsreport.com/files/2026-05-08_IF12959_6c33670142f9cf7752f63b66f1a44700c5fc4eba.html",
    }.items():
        raw = fetch("legal", url, os.path.join(base, "crs", name), "CRS report: " + gov)
        if raw and name.endswith(".html"):
            write_text(os.path.join(base, "crs", name[:-5] + ".txt"), html_to_text(raw))

    log("legal: Cybersecurity Information Sharing Act of 2015 as codified (6 U.S.C. 1500-1510)")
    for sec in range(1500, 1511):
        raw = fetch("legal", [f"https://www.govinfo.gov/content/pkg/USCODE-2024-title6/html/USCODE-2024-title6-chap6-subchapI-sec{sec}.htm",
                              f"https://www.govinfo.gov/content/pkg/USCODE-2023-title6/html/USCODE-2023-title6-chap6-subchapI-sec{sec}.htm"],
                    os.path.join(base, "usc", f"6usc{sec}.htm"), "US Code: " + gov)
        if raw:
            write_text(os.path.join(base, "usc", f"6usc{sec}.txt"), html_to_text(raw))

    log("legal: CISA information-sharing guidance")
    cisa = os.path.join(base, "cisa-2015")
    fetch("legal", "https://www.cisa.gov/sites/default/files/2026-11/CISA-2015-pcl-final-guidelines-periodic-review-november-2026-508.pdf",
          os.path.join(cisa, "CISA-2015-pcl-final-guidelines-2026-11.pdf"), "CISA: " + gov)
    page = fetch("legal", "https://www.cisa.gov/topics/cyber-threats-and-advisories/information-sharing/automated-indicator-sharing-ais",
                 os.path.join(cisa, "ais-program-page.html"), "CISA: " + gov)
    if page:
        write_text(os.path.join(cisa, "ais-program-page.txt"), html_to_text(page))
        links = set(re.findall(r'href="([^"]+\.pdf)"', page.decode("utf-8", "replace")))
        for href in sorted(links):
            url = urllib.parse.urljoin("https://www.cisa.gov/", href)
            if urllib.parse.urlsplit(url).netloc.endswith("cisa.gov"):
                fetch("legal", url, os.path.join(cisa, re.sub(r"[^\w.-]", "_", url.rsplit("/", 1)[1])[:120]), "CISA: " + gov)

    log("legal: Federal Register rules")
    fr = os.path.join(base, "federal-register")
    seen: set[str] = set()
    for phrase, types in FR_SEARCHES:
        q = [("per_page", "20"), ("order", "relevance"), ("conditions[term]", f'"{phrase}"')] + [("conditions[type][]", t) for t in types]
        try:
            res = json.loads(http_get("https://www.federalregister.gov/api/v1/documents.json?" + urllib.parse.urlencode(q)))
        except Exception as e:  # noqa: BLE001
            stats["failed"] += 1
            failures.append((f"federal-register search: {phrase[:50]}", f"{type(e).__name__}: {e}"))
            continue
        key = re.sub(r"\W+", " ", phrase.lower()).split()[:4]
        hits = [d for d in res.get("results", []) if all(k in re.sub(r"\W+", " ", d.get("title", "").lower()) for k in key)][:4]
        for d in hits:
            num = d["document_number"]
            if num in seen:
                continue
            seen.add(num)
            try:
                meta = json.loads(http_get(f"https://www.federalregister.gov/api/v1/documents/{num}.json"))
            except Exception as e:  # noqa: BLE001
                stats["failed"] += 1
                failures.append((f"federal-register/{num}", f"{type(e).__name__}: {e}"))
                continue
            stem = f"{meta.get('publication_date', '')}_{num}"
            write_text(os.path.join(fr, stem + ".json"), json.dumps({k: meta.get(k) for k in (
                "title", "type", "citation", "document_number", "publication_date", "effective_on", "agencies", "abstract", "html_url")}, indent=1, default=str))
            if meta.get("body_html_url"):
                raw = fetch("legal", meta["body_html_url"], os.path.join(fr, stem + ".html"), "Federal Register: " + gov)
                if raw:
                    write_text(os.path.join(fr, stem + ".txt"), html_to_text(raw))
            log(f"legal:   {meta.get('citation') or num}  {meta.get('type')}  {meta.get('title', '')[:90]}")

    log("legal: current regulation text from eCFR")
    try:
        dates = {t["number"]: t.get("up_to_date_as_of") for t in json.loads(http_get("https://www.ecfr.gov/api/versioner/v1/titles.json"))["titles"]}
    except Exception as e:  # noqa: BLE001
        dates = {}
        stats["failed"] += 1
        failures.append(("ecfr titles", f"{type(e).__name__}: {e}"))
    for title, query, name, desc in ECFR:
        if not dates.get(title):
            continue
        raw = fetch("legal", f"https://www.ecfr.gov/api/renderer/v1/content/enhanced/{dates[title]}/title-{title}?{query}",
                    os.path.join(base, "ecfr", name), f"eCFR ({desc}, as of {dates[title]}): " + gov)
        if raw:
            write_text(os.path.join(base, "ecfr", name[:-5] + ".txt"), html_to_text(raw))


# --------------------------------------------------------------------------- tool documentation
def docs_clone(group: str, repo: str, dest: str, patterns: list[str]) -> None:
    """Shallow clone that checks out only files matching the patterns (gitignore syntax) plus licence files."""
    if os.path.isdir(os.path.join(dest, ".git")):
        stats["skipped"] += 1
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    base = ["git", "-c", "core.longpaths=true", "-c", "core.autocrlf=false"]
    keep = patterns + ["/LICENSE*", "/LICENCE*", "/COPYING*", "/README*"]
    repo, _, ref = repo.partition("#")  # "url#tag" pins a release tag or branch instead of the default branch
    pin = ["--branch", ref] if ref else []
    try:
        for cmd in (base + ["clone", "--quiet", "--depth", "1", "--filter=blob:none", "--no-checkout", *pin, repo, dest],
                    base + ["-C", dest, "sparse-checkout", "set", "--no-cone", *keep],
                    base + ["-C", dest, "checkout", "--quiet"]):
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
        head = subprocess.run(["git", "-C", dest, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        files = [os.path.join(dp, f) for dp, dn, fn in os.walk(dest) if ".git" not in dp.split(os.sep) for f in fn]
        size = sum(os.path.getsize(p) for p in files)
        licence = "no licence file at the top level of the repository"
        for name in sorted(os.listdir(dest)):
            if name.upper().startswith(("LICENSE", "LICENCE", "COPYING")) and os.path.isfile(os.path.join(dest, name)):
                with open(os.path.join(dest, name), encoding="utf-8", errors="replace") as fh:
                    lines = [ln.strip() for ln in fh if ln.strip()][:2]
                licence = f"{name}: " + " / ".join(lines)[:160]
                break
        record(group, dest, f"{repo}@{head}", size, "", licence)
        stats["fetched"] += 1
        stats["bytes"] += size
        log(f"tooldocs:   {os.path.basename(dest):28s} {len(files):6d} files {size / 2**20:7.1f} MiB   {licence[:70]}")
    except Exception as e:  # noqa: BLE001
        stats["failed"] += 1
        msg = getattr(e, "stderr", "") or str(e)
        failures.append((os.path.relpath(dest, OUT).replace("\\", "/"), msg.strip()[:200]))


GH = "https://github.com/"
TOOL_DOCS = [  # (folder, repository, files to keep)
    ("metasploit-framework", GH + "rapid7/metasploit-framework.git", ["/docs/**/*.md", "/documentation/**/*.md"]),
    ("zap-website", GH + "zaproxy/zaproxy-website.git", ["/site/content/docs/**/*.md", "/site/content/faq/**/*.md"]),
    ("zap-core-help", GH + "zaproxy/zap-core-help.git", ["*.html"]),
    ("sqlmap-wiki", GH + "sqlmapproject/sqlmap.wiki.git", ["*.md"]),
    ("projectdiscovery-docs", GH + "projectdiscovery/docs.git", ["*.md", "*.mdx"]),
    ("bloodhound", GH + "SpecterOps/BloodHound.git", ["*.md", "*.mdx"]),
    ("impacket", GH + "fortra/impacket.git", ["*.md", "/examples/*.py"]),
    ("ghidra", GH + "NationalSecurityAgency/ghidra.git", ["/GhidraDocs/**/*.md", "/GhidraDocs/**/*.html", "/GhidraDocs/**/*.htm", "/*.md"]),
    ("radare2-book", GH + "radareorg/radare2-book.git", ["*.md"]),
    ("volatility3", GH + "volatilityfoundation/volatility3.git", ["/doc/**/*.rst", "*.md"]),
    ("suricata", GH + "OISF/suricata.git", ["/doc/userguide/**/*.rst"]),
    ("yara", GH + "VirusTotal/yara.git", ["/docs/**/*.rst", "/*.md"]),
    ("gtfobins", GH + "GTFOBins/GTFOBins.github.io.git", ["/_gtfobins/*", "*.md"]),
    ("payloads-all-the-things", GH + "swisskyrepo/PayloadsAllTheThings.git", ["*.md"]),
    ("lolbas", GH + "LOLBAS-Project/LOLBAS.git", ["/yml/**/*.yml", "*.md"]),
    ("atomic-red-team", GH + "redcanaryco/atomic-red-team.git", ["/atomics/**/*.md", "/*.md"]),
    ("sigma", GH + "SigmaHQ/sigma.git", ["/rules/**/*.yml", "*.md"]),
    ("john", GH + "openwall/john.git", ["/doc/*"]),
    ("nmap", GH + "nmap/nmap.git", ["/docs/*.txt", "/docs/*.1", "/docs/scripting.xml", "/docs/refguide.xml", "/scripts/*.nse", "/nselib/*.lua",
                                    "/nmap-services", "/nmap-service-probes"]),
    ("hashcat", GH + "hashcat/hashcat.git", ["/docs/*", "/*.md"]),
    ("zeek-docs", GH + "zeek/zeek-docs.git", ["*.rst"]),
    ("snort3", GH + "snort3/snort3.git", ["/doc/**/*.txt", "/*.md", "/lua/*.lua"]),
    ("wazuh-documentation", GH + "wazuh/wazuh-documentation.git", ["/source/**/*.rst"]),
    ("owasp-mastg", GH + "OWASP/owasp-mastg.git", ["*.md"]),
    ("wireshark-guides", "https://gitlab.com/wireshark/wireshark.git", ["/doc/wsug_src/*.adoc", "/doc/wsdg_src/*.adoc", "/doc/man_pages/*.adoc"]),
    ("kali-tools", "https://gitlab.com/kalilinux/documentation/kali-tools.git", ["*.md"]),
    ("kali-docs", "https://gitlab.com/kalilinux/documentation/kali-docs.git", ["*.md"]),
]


def group_tooldocs() -> None:
    log(f"tooldocs: documentation from {len(TOOL_DOCS)} tool repositories")
    for folder, repo, patterns in TOOL_DOCS:
        parent = "kali" if folder.startswith("kali-") else "tool-docs"
        docs_clone("tooldocs", repo, os.path.join(OUT, parent, folder), patterns)


# --------------------------------------------------------------------------- topic knowledge
def cone_clone(group: str, repo: str, dest: str, dirs: list[str]) -> None:
    """For very large repositories: fetch only the listed directories (and nothing else of the tree)."""
    if os.path.isdir(os.path.join(dest, ".git")):
        stats["skipped"] += 1
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    base = ["git", "-c", "core.longpaths=true", "-c", "core.autocrlf=false"]
    try:
        for cmd in (base + ["clone", "--quiet", "--depth", "1", "--filter=tree:0", "--sparse", repo, dest],
                    base + ["-C", dest, "sparse-checkout", "set", *dirs]):
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600)
        head = subprocess.run(["git", "-C", dest, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        files = [os.path.join(dp, f) for dp, dn, fn in os.walk(dest) if ".git" not in dp.split(os.sep) for f in fn]
        size = sum(os.path.getsize(p) for p in files)
        record(group, dest, f"{repo}@{head} ({', '.join(dirs)})", size, "", "see the repository's licence files")
        stats["fetched"] += 1
        stats["bytes"] += size
        log(f"{group}:   {os.path.basename(dest):28s} {len(files):6d} files {size / 2**20:7.1f} MiB")
    except Exception as e:  # noqa: BLE001
        stats["failed"] += 1
        msg = getattr(e, "stderr", "") or str(e)
        failures.append((os.path.relpath(dest, OUT).replace("\\", "/"), msg.strip()[:200]))


TOPIC_DOCS = [  # (area, folder, repository, files to keep)
    # web platform: JavaScript, DOM, XPath, HTTP, browser APIs
    ("web", "mdn-content", GH + "mdn/content.git", ["/files/en-us/**/*.md"]),
    ("web", "javascript-info", GH + "javascript-tutorial/en.javascript.info.git", ["*.md"]),
    ("web", "eloquent-javascript", GH + "marijnh/Eloquent-JavaScript.git", ["/*.md"]),
    ("web", "whatwg-dom", GH + "whatwg/dom.git", ["/dom.bs"]),
    ("web", "whatwg-fetch", GH + "whatwg/fetch.git", ["/fetch.bs"]),
    ("web", "whatwg-url", GH + "whatwg/url.git", ["/url.bs"]),
    ("web", "whatwg-xhr", GH + "whatwg/xhr.git", ["/xhr.bs"]),
    ("web", "nodejs-api", GH + "nodejs/node.git", ["/doc/api/*.md"]),
    ("web", "owasp-asvs", GH + "OWASP/ASVS.git", ["/5.0/en/*.md", "/4.0/en/*.md"]),
    ("web", "owasp-top10", GH + "OWASP/Top10.git", ["/20*/docs/*.md", "/20*/docs/en/**/*.md"]),
    # web developer tools and web tooling
    ("web-tools", "chrome-devtools-docs", GH + "GoogleChrome/developer.chrome.com.git", ["/site/en/docs/devtools/**/*.md"]),
    ("web-tools", "devtools-protocol", GH + "ChromeDevTools/devtools-protocol.git", ["/json/*.json", "/pdl/**"]),
    ("web-tools", "everything-curl", GH + "bagder/everything-curl.git", ["*.md"]),
    ("web-tools", "mitmproxy", GH + "mitmproxy/mitmproxy.git", ["/docs/src/content/**/*.md", "/examples/**/*.py"]),
    ("web-tools", "playwright", GH + "microsoft/playwright.git", ["/docs/src/**/*.md"]),
    ("web-tools", "puppeteer", GH + "puppeteer/puppeteer.git", ["/docs/**/*.md"]),
    ("web-tools", "selenium", GH + "SeleniumHQ/seleniumhq.github.io.git", ["/website_and_docs/content/documentation/**/*.en.md"]),
    # networking
    ("networking", "systems-approach-book", GH + "SystemsApproach/book.git", ["*.rst"]),
    ("networking", "cnp3-ebook", GH + "cnp3/ebook.git", ["*.rst"]),
    ("networking", "scapy", GH + "secdev/scapy.git", ["/doc/scapy/**/*.rst"]),
    # Lua and other languages
    ("lua", "lua-nginx-module", GH + "openresty/lua-nginx-module.git", ["/README.markdown", "/doc/**"]),
    ("lua", "luajit", GH + "LuaJIT/LuaJIT.git", ["/doc/*.html"]),
    ("lua", "learnxinyminutes", GH + "adambard/learnxinyminutes-docs.git", ["/*.md", "/*.html.markdown"]),
    # threat remediation: playbooks, response actions, hardening rules with fixes, detection rules
    # MITRE CAR is fetched by fetch_extras.py: its repository has file names Windows rejects, so a git checkout fails.
    ("remediation", "atc-react", GH + "atc-project/atc-react.git", ["*.md", "*.yml"]),
    ("remediation", "cert-sg-irm", GH + "certsocietegenerale/IRM.git", ["/EN/**"]),
    ("remediation", "incident-playbook", GH + "austinsonger/Incident-Playbook.git", ["*.md"]),
    ("remediation", "ir-plan-template", GH + "counteractive/incident-response-plan-template.git", ["*.md"]),
    ("remediation", "compliance-as-code", GH + "ComplianceAsCode/content.git",
     ["/linux_os/guide/**/rule.yml", "/linux_os/guide/**/bash/shared.sh", "/linux_os/guide/**/ansible/shared.yml", "/applications/**/rule.yml"]),
    ("remediation", "splunk-security-content", GH + "splunk/security_content.git", ["/detections/**/*.yml", "/stories/**/*.yml"]),
    ("remediation", "elastic-detection-rules", GH + "elastic/detection-rules.git", ["/rules/**/*.toml", "/rules_building_block/**/*.toml"]),
    # PowerShell, including the ISE
    ("powershell", "powershell-docs", GH + "MicrosoftDocs/PowerShell-Docs.git",
     ["/reference/5.1/**/*.md", "/reference/7.4/**/*.md", "/reference/docs-conceptual/**/*.md"]),
    ("powershell", "windows-powershell-docs", GH + "MicrosoftDocs/windows-powershell-docs.git", ["/docset/winserver2025-ps/**/*.md"]),
    ("powershell", "psscriptanalyzer", GH + "PowerShell/PSScriptAnalyzer.git", ["/docs/**/*.md"]),
]
MORE_RFCS = """768 791 792 793 826 854 959 1122 1123 1191 1350 1812 1918 1939 2131 2328 2784 2827 2865 3022 3031 3261 3411 3704
3748 3927 3954 4271 4302 4443 4511 4632 4861 4987 5227 5389 5424 5681 5905 5925 6146 6241 6376 6762 6890 7011 7296 7348
7858 8489 9002 9051 9204 9293""".split()


def group_topics() -> None:
    log(f"topics: documentation from {len(TOPIC_DOCS) + 1} repositories")
    for area, folder, repo, patterns in TOPIC_DOCS:
        docs_clone("topics", repo, os.path.join(OUT, area, folder), patterns)
    cone_clone("topics", GH + "mozilla-firefox/firefox.git", os.path.join(OUT, "web-tools", "firefox-devtools-docs"), ["devtools/docs/user"])

    log("topics: Lua reference manuals")
    for ver in ("5.1", "5.3", "5.4"):
        raw = fetch("topics", f"https://www.lua.org/manual/{ver}/manual.html", os.path.join(OUT, "lua", "manual", f"lua-{ver}-manual.html"),
                    "Lua reference manual: freely available under the Lua (MIT) licence")
        if raw:
            write_text(os.path.join(OUT, "lua", "manual", f"lua-{ver}-manual.txt"), html_to_text(raw))

    log("topics: MITRE weakness, attack-pattern and defensive-technique catalogues")
    mitre = "MITRE: free to use under MITRE's terms of use, with attribution"
    raw = fetch("topics", "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip", os.path.join(OUT, "remediation", "mitre", "cwec_latest.xml.zip"), mitre)
    if raw:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            for name in zf.namelist():
                if name.endswith(".xml") and "/" not in name and ".." not in name:
                    target = os.path.join(OUT, "remediation", "mitre", name)
                    if not os.path.isfile(target):
                        with open(target, "wb") as fh:
                            fh.write(zf.read(name))
    fetch("topics", "https://capec.mitre.org/data/xml/capec_latest.xml", os.path.join(OUT, "remediation", "mitre", "capec_latest.xml"), mitre)
    fetch("topics", "https://d3fend.mitre.org/ontologies/d3fend.json", os.path.join(OUT, "remediation", "mitre", "d3fend.json"), "MITRE D3FEND: MIT licence")

    log("topics: CISA response playbooks")
    gov = "CISA: US government work, public domain"
    fetch("topics", ["https://www.cisa.gov/sites/default/files/2024-08/Federal_Government_Cybersecurity_Incident_and_Vulnerability_Response_Playbooks_508C.pdf",
                     "https://www.cisa.gov/sites/default/files/publications/Federal_Government_Cybersecurity_Incident_and_Vulnerability_Response_Playbooks_508C.pdf"],
          os.path.join(OUT, "remediation", "cisa", "incident-and-vulnerability-response-playbooks.pdf"), gov)
    page = fetch("topics", "https://www.cisa.gov/stopransomware/ransomware-guide", os.path.join(OUT, "remediation", "cisa", "stopransomware-guide.html"), gov)
    if page:
        write_text(os.path.join(OUT, "remediation", "cisa", "stopransomware-guide.txt"), html_to_text(page))

    log(f"topics: {len(MORE_RFCS)} more networking RFCs")
    for n in MORE_RFCS:
        fetch("topics", f"https://www.rfc-editor.org/rfc/rfc{n}.txt", os.path.join(OUT, "references", "rfc", f"rfc{n}.txt"),
              "RFC: IETF Trust licence, copying and distribution permitted")


# --------------------------------------------------------------------------- prioritization, logging, kernel/memory, testing
def page_with_pdfs(group: str, url: str, dest_dir: str, name: str, licence: str, max_pdfs: int = 8) -> None:
    """Save a web page as HTML and text, plus the PDFs it links to on the same site."""
    raw = fetch(group, url, os.path.join(dest_dir, name + ".html"), licence)
    if not raw:
        return
    write_text(os.path.join(dest_dir, name + ".txt"), html_to_text(raw))
    site = urllib.parse.urlsplit(url).netloc.split(".", 1)[-1]
    links = sorted(set(re.findall(r'href="([^"]+\.pdf)"', raw.decode("utf-8", "replace"), flags=re.I)))
    for href in links[:max_pdfs]:
        full = urllib.parse.urljoin(url, href)
        if urllib.parse.urlsplit(full).netloc.endswith(site):
            fname = re.sub(r"[^\w.-]", "_", urllib.parse.unquote(full.rsplit("/", 1)[1]))[:120]
            fetch(group, full.replace(" ", "%20"), os.path.join(dest_dir, fname), licence)


DEPTH_DOCS = [  # (area, folder, repository, files to keep)
    # how to decide what gets fixed first
    ("prioritization", "certcc-ssvc", GH + "CERTCC/SSVC.git", ["/docs/**/*.md", "/data/**/*.csv", "/data/**/*.json"]),
    ("prioritization", "cisa-vulnrichment-2026", GH + "cisagov/vulnrichment.git", ["/2026/**/*.json"]),
    ("prioritization", "owasp-community", GH + "OWASP/www-community.git", ["/pages/**/*.md"]),
    ("web", "owasp-api-security", GH + "OWASP/API-Security.git", ["/editions/2023/en/*.md"]),
    # logging: what to log, where logs live, how to parse them, what to detect in them
    ("logging", "elastic-common-schema", GH + "elastic/ecs.git", ["/docs/**/*.md", "/docs/**/*.asciidoc", "/schemas/*.yml"]),
    ("logging", "ocsf-schema", GH + "ocsf/ocsf-schema.git", ["*.json", "*.md"]),
    ("logging", "logstash-grok-patterns", GH + "logstash-plugins/logstash-patterns-core.git", ["/patterns/**"]),
    ("logging", "ossem", GH + "OTRF/OSSEM.git", ["*.md", "*.yml"]),
    ("logging", "ossem-data-dictionaries", GH + "OTRF/OSSEM-DD.git", ["*.yml", "*.yaml", "*.md"]),
    # pinned to the 4.x line: the default branch moved the decoders and rules into a different engine format
    ("logging", "wazuh-ruleset", GH + "wazuh/wazuh.git#v4.14.10", ["/ruleset/decoders/*.xml", "/ruleset/rules/*.xml", "/ruleset/sca/**/*.yml"]),
    ("logging", "fluent-bit-docs", GH + "fluent/fluent-bit-docs.git", ["*.md"]),
    ("logging", "vector-docs", GH + "vectordotdev/vector.git", ["/website/content/en/docs/**/*.md", "/website/cue/reference/remap/functions/*.cue"]),
    ("logging", "rsyslog-docs", GH + "rsyslog/rsyslog.git", ["/doc/**/*.rst"]),
    ("logging", "linux-audit-docs", GH + "linux-audit/audit-documentation.git", ["*.md"]),
    ("logging", "forensic-artifacts", GH + "ForensicArtifacts/artifacts.git", ["*.yaml", "/docs/**"]),
    ("logging", "kape-files", GH + "EricZimmerman/KapeFiles.git", ["*.tkape", "*.mkape", "*.md"]),
    ("logging", "velociraptor-artifacts", GH + "Velocidex/velociraptor.git", ["/artifacts/definitions/**/*.yaml"]),
    ("logging", "velociraptor-docs", GH + "Velocidex/velociraptor-docs.git", ["/content/docs/**/*.md"]),
    ("logging", "plaso-docs", GH + "log2timeline/plaso.git", ["/docs/sources/**/*.md"]),
    ("logging", "sysmon-modular", GH + "olafhartong/sysmon-modular.git", ["*.xml", "*.md"]),
    ("logging", "sysmon-config", GH + "SwiftOnSecurity/sysmon-config.git", ["*.xml", "*.md"]),
    ("logging", "windows-log-settings", GH + "Yamato-Security/EnableWindowsLogSettings.git", ["*.md"]),
    ("logging", "jpcert-tool-analysis", GH + "JPCERTCC/ToolAnalysisResultSheet.git", ["*.htm", "*.html", "*.md"]),
    ("logging", "azure-sentinel", GH + "Azure/Azure-Sentinel.git", ["/Parsers/**/*.yaml", "/Detections/**/*.yaml", "/Hunting Queries/**/*.yaml"]),
    # Google's chronicle/detection-rules is fetched by fetch_extras.py: it has file names Windows rejects.
    ("logging", "sysinternals-docs", GH + "MicrosoftDocs/sysinternals.git", ["*.md"]),
    ("logging", "windows-server-ad-security", GH + "MicrosoftDocs/windowsserverdocs.git", ["/WindowsServerDocs/identity/ad-ds/plan/**/*.md"]),
    # kernel and memory: hardening, mitigations, live patching, crash and memory analysis
    ("kernel-memory", "linux-kernel-docs", GH + "torvalds/linux.git",
     ["/Documentation/admin-guide/**/*.rst", "/Documentation/admin-guide/kernel-parameters.txt", "/Documentation/security/**/*.rst",
      "/Documentation/dev-tools/**/*.rst", "/Documentation/userspace-api/**/*.rst", "/Documentation/livepatch/**/*.rst",
      "/Documentation/mm/**/*.rst", "/Documentation/trace/**/*.rst", "/Documentation/bpf/**/*.rst",
      "/Documentation/networking/**/*.rst", "/Documentation/process/*.rst"]),
    ("kernel-memory", "llvm-sanitizer-docs", GH + "llvm/llvm-project.git",
     ["/clang/docs/*.md", "/clang/docs/*.rst", "/llvm/docs/LibFuzzer.*", "/llvm/docs/FuzzingLLVM.*", "/compiler-rt/docs/**"]),
    ("kernel-memory", "windows-debugger-docs", GH + "MicrosoftDocs/windows-driver-docs.git",
     ["/windows-driver-docs-pr/debugger/**/*.md", "/windows-driver-docs-pr/debuggercmds/**/*.md", "/windows-driver-docs-pr/driversecurity/**/*.md"]),
    ("kernel-memory", "kernel-hardening-checker", GH + "a13xp0p0v/kernel-hardening-checker.git", ["*.md", "*.py"]),
    ("kernel-memory", "kernel-defence-map", GH + "a13xp0p0v/linux-kernel-defence-map.git", ["*.md", "*.dot"]),
    ("kernel-memory", "kspp", GH + "KSPP/kspp.github.io.git", ["*.md"]),
    ("kernel-memory", "openssf-hardening-guides", GH + "ossf/wg-best-practices-os-developers.git", ["/docs/**/*.md"]),
    ("kernel-memory", "sanitizers-wiki", GH + "google/sanitizers.wiki.git", ["*.md"]),
    ("kernel-memory", "volatility-wiki", GH + "volatilityfoundation/volatility.wiki.git", ["*.md"]),
    ("kernel-memory", "selinux-notebook", GH + "SELinuxProject/selinux-notebook.git", ["/src/*.md"]),
    ("kernel-memory", "falco-rules", GH + "falcosecurity/rules.git", ["/rules/*.yaml", "*.md"]),
    ("kernel-memory", "falco-docs", GH + "falcosecurity/falco-website.git", ["/content/en/docs/**/*.md"]),
    ("kernel-memory", "tetragon-docs", GH + "cilium/tetragon.git", ["/docs/content/en/docs/**/*.md"]),
    ("kernel-memory", "ebpf-docs", GH + "isovalent/ebpf-docs.git", ["/docs/**/*.md"]),
    ("kernel-memory", "kpatch", GH + "dynup/kpatch.git", ["*.md", "/doc/**"]),
    ("kernel-memory", "lkrg", GH + "lkrg-org/lkrg.git", ["*.md"]),
    # testing techniques: fuzzing, static analysis, unit and property testing, adversary emulation
    ("testing", "syzkaller-docs", GH + "google/syzkaller.git", ["/docs/**/*.md"]),
    ("testing", "aflplusplus-docs", GH + "AFLplusplus/AFLplusplus.git", ["/docs/**/*.md", "/instrumentation/*.md", "/README.md"]),
    ("testing", "oss-fuzz-docs", GH + "google/oss-fuzz.git", ["/docs/**/*.md"]),
    ("testing", "google-fuzzing", GH + "google/fuzzing.git", ["/docs/**/*.md", "/tutorials/**/*.md"]),
    ("testing", "fuzzingbook", GH + "uds-se/fuzzingbook.git", ["/notebooks/*.ipynb"]),
    ("testing", "pytest-docs", GH + "pytest-dev/pytest.git", ["/doc/en/**/*.rst"]),
    ("testing", "pester-docs", GH + "pester/docs.git", ["/docs/**/*.mdx", "/docs/**/*.md"]),
    ("testing", "hypothesis-docs", GH + "HypothesisWorks/hypothesis.git", ["/hypothesis/docs/**/*.rst", "/hypothesis-python/docs/**/*.rst"]),
    ("testing", "codeql", GH + "github/codeql.git", ["/docs/codeql/**/*.rst", "*.qhelp"]),
    ("testing", "caldera-fieldmanual", GH + "mitre/fieldmanual.git", ["*.md", "*.rst"]),
    ("testing", "javascript-testing-best-practices", GH + "goldbergyoni/javascript-testing-best-practices.git", ["/readme.md"]),
]


def group_depth() -> None:
    log(f"depth: documentation from {len(DEPTH_DOCS)} repositories")
    for area, folder, repo, patterns in DEPTH_DOCS:
        docs_clone("depth", repo, os.path.join(OUT, area, folder), patterns)

    gov = "US government work, public domain"
    first = "FIRST.org: free to use with attribution (see the document's own licence statement)"
    pri = os.path.join(OUT, "prioritization")
    log("depth: prioritization methods (SSVC, CVSS, EPSS, CISA directives)")
    page_with_pdfs("depth", "https://www.cisa.gov/stakeholder-specific-vulnerability-categorization-ssvc", os.path.join(pri, "cisa-ssvc"), "ssvc-overview", "CISA: " + gov)
    fetch("depth", "https://www.cisa.gov/sites/default/files/publications/cisa-ssvc-guide%20508c.pdf", os.path.join(pri, "cisa-ssvc", "cisa-ssvc-guide.pdf"), "CISA: " + gov)
    for name, url in {
        "cvss-v4.0-specification": "https://www.first.org/cvss/v4.0/specification-document",
        "cvss-v4.0-user-guide": "https://www.first.org/cvss/v4.0/user-guide",
        "cvss-v4.0-examples": "https://www.first.org/cvss/v4.0/examples",
        "cvss-v4.0-faq": "https://www.first.org/cvss/v4.0/faq",
        "cvss-v3.1-specification": "https://www.first.org/cvss/v3.1/specification-document",
        "cvss-v3.1-user-guide": "https://www.first.org/cvss/v3.1/user-guide",
        "epss-overview": "https://www.first.org/epss/",
        "epss-faq": "https://www.first.org/epss/faq",
        "epss-data-and-statistics": "https://www.first.org/epss/data_stats",
    }.items():
        raw = fetch("depth", url, os.path.join(pri, "first", name + ".html"), first)
        if raw:
            write_text(os.path.join(pri, "first", name + ".txt"), html_to_text(raw))
    for name, url in {
        "bod-22-01-known-exploited-vulnerabilities": "https://www.cisa.gov/news-events/directives/bod-22-01-reducing-significant-risk-known-exploited-vulnerabilities",
        "bod-19-02-remediation-requirements": "https://www.cisa.gov/news-events/directives/bod-19-02-vulnerability-remediation-requirements-internet-accessible-systems",
    }.items():
        raw = fetch("depth", url, os.path.join(pri, "cisa-directives", name + ".html"), "CISA: " + gov)
        if raw:
            write_text(os.path.join(pri, "cisa-directives", name + ".txt"), html_to_text(raw))
    fetch("depth", "https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.41.pdf", os.path.join(pri, "nist", "NIST.CSWP.41-likely-exploited-vulnerabilities.pdf"), "NIST: " + gov)

    log("depth: logging guidance")
    lg = os.path.join(OUT, "logging", "guidance")
    fetch("depth", "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-92r1.ipd.pdf", os.path.join(lg, "NIST.SP.800-92r1.ipd.pdf"), "NIST: " + gov)
    fetch("depth", ["https://www.whitehouse.gov/wp-content/uploads/2021/08/M-21-31-Improving-the-Federal-Governments-Investigative-and-Remediation-Capabilities-Related-to-Cybersecurity-Incidents.pdf",
                    "https://bidenwhitehouse.archives.gov/wp-content/uploads/2021/08/M-21-31-Improving-the-Federal-Governments-Investigative-and-Remediation-Capabilities-Related-to-Cybersecurity-Incidents.pdf"],
          os.path.join(lg, "OMB-M-21-31-logging-maturity.pdf"), "OMB memorandum: " + gov)
    fetch("depth", "https://www.jpcert.or.jp/english/pub/sr/20170612ac-ir_research_en.pdf", os.path.join(lg, "JPCERT-detecting-lateral-movement-through-event-logs.pdf"),
          "JPCERT/CC report: copyright JPCERT/CC, published free of charge")
    page_with_pdfs("depth", "https://www.cisa.gov/resources-tools/resources/best-practices-event-logging-and-threat-detection", lg,
                   "cisa-best-practices-event-logging", "CISA and partner agencies: " + gov)

    log("depth: memory safety guidance")
    fetch("depth", "https://www.cisa.gov/sites/default/files/2023-12/The-Case-for-Memory-Safe-Roadmaps-508c.pdf",
          os.path.join(OUT, "kernel-memory", "guidance", "CISA-case-for-memory-safe-roadmaps.pdf"), "CISA and partner agencies: " + gov)


# --------------------------------------------------------------------------- services and what indicates a problem
SERVICE_DOCS = [  # (area, folder, repository, files to keep)
    # what the services are: Windows, Linux, and the usual microservice building blocks
    ("services", "windows-server-docs", GH + "MicrosoftDocs/windowsserverdocs.git", ["/WindowsServerDocs/**/*.md"]),
    ("services", "windows-support-articles", GH + "MicrosoftDocs/SupportArticles-docs.git",
     ["/support/windows-server/**/*.md", "/support/windows-client/**/*.md"]),
    # systemd is fetched by fetch_extras.py: its test data has file names Windows rejects.
    ("services", "debian-handbook", "https://salsa.debian.org/hertzog/debian-handbook.git", ["/en-US/*.xml"]),
    ("services", "ubuntu-server-docs", GH + "canonical/ubuntu-server-documentation.git", ["*.md", "*.rst"]),
    ("services", "apache-httpd-docs", GH + "apache/httpd.git", ["/docs/manual/**/*.xml"]),
    ("services", "nginx-docs", GH + "nginx/nginx.org.git", ["/xml/en/**/*.xml"]),
    ("services", "postgresql-docs", GH + "postgres/postgres.git", ["/doc/src/sgml/**/*.sgml"]),
    ("services", "redis-docs", GH + "redis/docs.git", ["/content/operate/**/*.md", "/content/develop/**/*.md"]),
    ("services", "docker-docs", GH + "docker/docs.git",
     ["/content/manuals/engine/**/*.md", "/content/manuals/compose/**/*.md", "/content/reference/**/*.md", "/content/guides/**/*.md"]),
    ("services", "kubernetes-docs", GH + "kubernetes/website.git", ["/content/en/docs/**/*.md"]),
    ("services", "istio-docs", GH + "istio/istio.io.git", ["/content/en/docs/**/*.md"]),
    ("services", "envoy-docs", GH + "envoyproxy/envoy.git", ["/docs/root/**/*.rst"]),
    ("services", "coredns", GH + "coredns/coredns.git", ["/plugin/**/README.md", "/README.md", "/man/*"]),
    ("services", "containerd-docs", GH + "containerd/containerd.git", ["/docs/**/*.md"]),
    ("services", "etcd-docs", GH + "etcd-io/website.git", ["/content/en/docs/v3.6/**/*.md", "/content/en/docs/v3.5/**/*.md"]),
    ("services", "rabbitmq-docs", GH + "rabbitmq/rabbitmq-website.git", ["/docs/**/*.md"]),
    ("services", "kafka-docs", GH + "apache/kafka.git", ["/docs/**/*.html", "/docs/**/*.md"]),
    ("services", "prometheus-docs", GH + "prometheus/docs.git", ["/content/docs/**/*.md", "/docs/**/*.md"]),
    ("services", "opentelemetry-docs", GH + "open-telemetry/opentelemetry.io.git", ["/content/en/docs/**/*.md"]),
    ("services", "cadvisor-docs", GH + "google/cadvisor.git", ["/docs/**/*.md"]),
    # what a problem looks like: alert rules, runbooks, health checks, host-inspection queries, abuse catalogues
    ("indicators", "awesome-prometheus-alerts", GH + "samber/awesome-prometheus-alerts.git", ["/_data/rules.yml", "/dist/rules/**/*.yml", "*.md"]),
    ("indicators", "prometheus-operator-runbooks", GH + "prometheus-operator/runbooks.git", ["/content/**/*.md"]),
    ("indicators", "kubernetes-mixin", GH + "kubernetes-monitoring/kubernetes-mixin.git", ["/alerts/*.libsonnet", "/rules/*.libsonnet", "*.md"]),
    ("indicators", "monitoring-mixins", GH + "monitoring-mixins/website.git", ["*.md", "*.yaml", "*.yml"]),
    ("indicators", "zabbix-templates", GH + "zabbix/zabbix.git", ["/templates/**/README.md", "/templates/**/*.yaml"]),
    ("indicators", "netdata-health", GH + "netdata/netdata.git",
     ["/src/health/health.d/*.conf", "/src/health/**/*.md", "/src/collectors/**/*.md", "/src/go/plugin/go.d/collector/**/*.md"]),
    ("indicators", "datadog-integrations", GH + "DataDog/integrations-core.git", ["/*/README.md", "/*/assets/service_checks.json", "/*/metadata.csv"]),
    ("indicators", "osquery", GH + "osquery/osquery.git", ["/specs/**/*.table", "/docs/wiki/**/*.md", "/packs/*.conf"]),
    ("indicators", "node-exporter", GH + "prometheus/node_exporter.git", ["*.md", "/docs/**"]),
    ("indicators", "windows-exporter", GH + "prometheus-community/windows_exporter.git", ["*.md", "/docs/**/*.md"]),
    ("indicators", "hayabusa-rules", GH + "Yamato-Security/hayabusa-rules.git", ["*.yml", "*.md"]),
    ("indicators", "loldrivers", GH + "magicsword-io/LOLDrivers.git", ["/yaml/*.yaml", "/yaml/*.yml", "*.md"]),
    ("indicators", "lolrmm", GH + "magicsword-io/LOLRMM.git", ["/yaml/*.yaml", "*.md"]),
    ("indicators", "hijacklibs", GH + "wietze/HijackLibs.git", ["/yml/**/*.yml", "*.md"]),
    ("indicators", "auditd-rules", GH + "Neo23x0/auditd.git", ["*.rules", "*.md"]),
]


def group_services() -> None:
    log(f"services: documentation from {len(SERVICE_DOCS)} repositories")
    for area, folder, repo, patterns in SERVICE_DOCS:
        docs_clone("services", repo, os.path.join(OUT, area, folder), patterns)

    log("services: port and service-name registry, Windows service pages")
    fetch("services", "https://www.iana.org/assignments/service-names-port-numbers/service-names-port-numbers.csv",
          os.path.join(OUT, "services", "iana", "service-names-port-numbers.csv"), "IANA registry: public domain (CC0)")
    for name, url in {
        "per-user-services-in-windows": "https://learn.microsoft.com/en-us/windows/application-management/per-user-services-in-windows",
        "system-services-security-guidelines": "https://learn.microsoft.com/en-us/windows-server/security/windows-services/security-guidelines-for-disabling-system-services-in-windows-server",
        "service-overview-and-network-port-requirements": "https://learn.microsoft.com/en-us/troubleshoot/windows-server/networking/service-overview-and-network-port-requirements",
    }.items():
        raw = fetch("services", url, os.path.join(OUT, "services", "windows-service-pages", name + ".html"), "Microsoft Learn documentation: CC BY 4.0")
        if raw:
            text = html_to_text(raw, "content")
            write_text(os.path.join(OUT, "services", "windows-service-pages", name + ".txt"), text if len(text) > 400 else html_to_text(raw))


# --------------------------------------------------------------------------- round two: more for the weakest topics
LS = GH + "logstash-plugins/logstash-filter-"
ROUND2_DOCS = [  # (area, folder, repository, files to keep)
    # Lua as it is embedded in security and operations tools
    ("lua", "haproxy-lua", GH + "haproxy/haproxy.git", ["/doc/lua-api/*.rst", "/doc/lua.txt", "/examples/lua/*"]),
    # complete Lua programs with a stated task each, for writing Lua rather than only reading about it
    ("lua", "rosetta-code", GH + "acmeism/RosettaCodeData.git", ["/Task/*/Lua/*", "/Task/*/00-TASK.txt", "/Task/*/00DESCRIPTION"]),
    # log parsing: the Logstash filters people actually build pipelines from, and Loki's pipeline stages and query language
    ("logging", "logstash-filter-grok", LS + "grok.git", ["/docs/*.asciidoc"]),
    ("logging", "logstash-filter-dissect", LS + "dissect.git", ["/docs/*.asciidoc"]),
    ("logging", "logstash-filter-date", LS + "date.git", ["/docs/*.asciidoc"]),
    ("logging", "logstash-filter-mutate", LS + "mutate.git", ["/docs/*.asciidoc"]),
    ("logging", "logstash-filter-kv", LS + "kv.git", ["/docs/*.asciidoc"]),
    ("logging", "logstash-filter-json", LS + "json.git", ["/docs/*.asciidoc"]),
    ("logging", "logstash-filter-csv", LS + "csv.git", ["/docs/*.asciidoc"]),
    ("logging", "logstash-filter-geoip", LS + "geoip.git", ["/docs/*.asciidoc"]),
    ("logging", "logstash-filter-syslog_pri", LS + "syslog_pri.git", ["/docs/*.asciidoc"]),
    ("logging", "loki-docs", GH + "grafana/loki.git", ["/docs/sources/send-data/promtail/stages/*.md", "/docs/sources/query/**/*.md",
                                                       "/docs/sources/send-data/promtail/*.md"]),
]
SYSLOG_RFCS = ["3164", "5425", "5426", "6587", "5848"]


def group_round2() -> None:
    log(f"round2: documentation from {len(ROUND2_DOCS)} repositories")
    for area, folder, repo, patterns in ROUND2_DOCS:
        docs_clone("round2", repo, os.path.join(OUT, area, folder), patterns)
    for n in SYSLOG_RFCS:
        fetch("round2", f"https://www.rfc-editor.org/rfc/rfc{n}.txt", os.path.join(OUT, "references", "rfc", f"rfc{n}.txt"),
              "RFC: IETF Trust licence, copying and distribution permitted")


GROUPS = {"man": group_man, "references": group_references, "legal": group_legal, "tooldocs": group_tooldocs,
          "topics": group_topics, "depth": group_depth, "services": group_services, "round2": group_round2}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(GROUPS)
    t0 = time.time()
    for g in wanted:
        if g not in GROUPS:
            sys.exit(f"unknown group {g!r}; choose from {', '.join(GROUPS)}")
        GROUPS[g]()
    log(f"\nfinished in {time.time() - t0:.0f}s: {stats['fetched']} fetched ({stats['bytes'] / 2**20:.1f} MiB), "
        f"{stats['skipped']} already present, {stats['failed']} failed")
    for path, err in failures:
        log(f"  FAILED {path}: {err}")
