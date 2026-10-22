"""
SciLit - corpus/fetch_pdfs.py
Part 1 steps 5-6: download each article's PDF and record proof of what was fetched.

Input : data/raw/articles.jsonl
Output: data/pdfs/<article id>.pdf       (not in git)
        data/raw/manifest.jsonl          one line per article: status, SHA-256, size, URL, date

Rules:
  - arXiv papers from arxiv.org, 3 s apart (arXiv's politeness rule; export.arxiv.org is
    their metadata API and was 10x slower for PDFs in our test).
  - A file still downloading after 90 s is abandoned as "timeout" and retried on the next run.
  - Other papers from their open-access PDF link, 1 s apart.
  - A reply that is not a PDF (an HTML landing page, a login page) is a failure, never saved.
  - Re-running skips papers already downloaded, so the job can be stopped and resumed.

Usage (from the repository root, with PYTHONPATH=scilit/src):
  python -m scilit.corpus.fetch_pdfs --limit 20    # test on 20 papers
  python -m scilit.corpus.fetch_pdfs               # everything (about 1.5-2 h)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

MAX_BYTES = 30_000_000
MAX_SECONDS = 90           # one slow server must not stall the whole run
HEADERS = {"User-Agent": "SciLit student project (literature-review research; contact via OPENALEX_EMAIL)"}


def pdf_url(a: dict) -> tuple[str | None, float]:
    """(URL, seconds to wait after the request). arXiv first: stable and always a PDF."""
    if a.get("arxiv_id"):
        return f"https://arxiv.org/pdf/{a['arxiv_id']}", 3.0
    return a.get("pdf_url"), 1.0


def download(session: requests.Session, url: str, dest: Path) -> dict:
    """Download one PDF. Returns the manifest fields (status ok or failed + reason)."""
    try:
        r = session.get(url, timeout=30, stream=True, headers=HEADERS)
        if r.status_code != 200:
            return {"status": "failed", "reason": f"http_{r.status_code}"}
        data, start = b"", time.monotonic()
        for chunk in r.iter_content(65536):
            data += chunk
            if len(data) > MAX_BYTES:
                return {"status": "failed", "reason": "too_large"}
            if time.monotonic() - start > MAX_SECONDS:
                return {"status": "failed", "reason": "timeout"}
    except requests.RequestException as e:
        return {"status": "failed", "reason": f"network_{type(e).__name__}"}
    if not data.startswith(b"%PDF"):                  # the only reliable test: the file's magic bytes
        return {"status": "failed", "reason": "not_a_pdf"}
    dest.write_bytes(data)
    return {"status": "ok", "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def load_manifest(path: Path) -> dict[str, dict]:
    """Latest entry per article (the file is appended to, so a retried paper has several lines)."""
    entries: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                entries[e["id"]] = e
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(description="SciLit Part 1: download PDFs")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--pdfs", default="data/pdfs")
    ap.add_argument("--limit", type=int, help="only the first N articles (test)")
    args = ap.parse_args()
    raw, pdf_dir = Path(args.raw), Path(args.pdfs)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw / "manifest.jsonl"

    articles = [json.loads(l) for l in (raw / "articles.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        articles = articles[: args.limit]
    manifest = load_manifest(manifest_path)
    # compact the file: one line per article, so it never grows with retries
    manifest_path.write_text("".join(json.dumps(e) + "\n" for e in manifest.values()), encoding="utf-8")

    todo = [a for a in articles
            if not (manifest.get(a["id"], {}).get("status") == "ok" and (pdf_dir / f"{a['id']}.pdf").exists())]
    print(f"{len(articles) - len(todo)} already downloaded, {len(todo)} to fetch.", flush=True)

    session = requests.Session()
    with manifest_path.open("a", encoding="utf-8") as mf:
        for n, a in enumerate(todo, 1):
            url, wait = pdf_url(a)
            entry = {"id": a["id"], "url": url,
                     "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            if not url:
                entry.update(status="failed", reason="no_pdf_url")
            else:
                entry.update(download(session, url, pdf_dir / f"{a['id']}.pdf"))
                time.sleep(wait)
            mf.write(json.dumps(entry) + "\n")
            mf.flush()                                   # safe to stop at any moment
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}", flush=True)

    final = load_manifest(manifest_path)
    ok = sum(1 for a in articles if final.get(a["id"], {}).get("status") == "ok")
    reasons: dict[str, int] = {}
    for a in articles:
        e = final.get(a["id"], {})
        if e.get("status") != "ok":
            reasons[e.get("reason", "?")] = reasons.get(e.get("reason", "?"), 0) + 1
    print(f"Done: {ok}/{len(articles)} PDFs. Failures: {reasons or 'none'}")


if __name__ == "__main__":
    main()
