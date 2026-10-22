"""
SciLit - corpus/validate.py
Part 2: quality gate. A PDF is parsed only if it passes five checks; every paper gets a
status and a reason, pass or fail.

  1. opens          PyMuPDF opens it, not encrypted, 3-60 pages
  2. text_layer     enough characters per page (else it is a scanned image; no OCR in scope)
  3. not_garbled    share of letters, broken glyphs "(cid:..)" / U+FFFD
  4. english        fastText language ID (lid.176) on the first pages
  5. title_on_page1 our title (corrected from arXiv) appears on page 1: catches wrong PDFs

Input : data/raw/articles.jsonl, data/raw/manifest.jsonl, data/pdfs/
Output: reports/validation_report.csv     one row per article
        reports/validation_summary.md     counts per reason (for the data card)

Usage (from the repository root, with PYTHONPATH=scilit/src):
  python -m scilit.corpus.validate
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import pymupdf
import requests
from rapidfuzz import fuzz

from scilit.corpus.collect import norm_title

LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
MIN_PAGES, MAX_PAGES = 3, 60
MIN_CHARS_PER_PAGE = 500     # a real paper page has ~3,000-5,000; a scanned page has ~0
MIN_LETTER_RATIO = 0.60      # letters among non-space characters; garbled text falls far below
MAX_BROKEN_RATIO = 0.01      # (cid:12) codes and U+FFFD replacement characters
MIN_TITLE_SCORE = 80         # fuzzy partial match of the title inside page 1 (0-100)


def load_lid(models_dir: Path):
    """fastText language-ID model (1 MB), downloaded once into models/."""
    import fasttext
    path = models_dir / "lid.176.ftz"
    if not path.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        print("Downloading fastText language model (1 MB)...", flush=True)
        r = requests.get(LID_URL, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    return fasttext.load_model(str(path))


def language(model, text: str) -> tuple[str, float]:
    """Top language and its probability. Uses the low-level call: fastText's predict()
    crashes with NumPy 2 ("Unable to avoid copy")."""
    prob, label = model.f.predict(" ".join(text.split())[:5000], 1, 0.0, "strict")[0]
    return label.replace("__label__", ""), prob


def check(pdf_path: Path, title: str, lid) -> dict:
    """Run the five checks in order; stop at the first failure (its name is the reason)."""
    row = {"pages": "", "chars_per_page": "", "letter_ratio": "", "lang": "", "lang_prob": "", "title_score": ""}
    try:
        doc = pymupdf.open(pdf_path)
    except Exception:
        return {**row, "status": "rejected", "reason": "opens"}
    if doc.needs_pass or not MIN_PAGES <= doc.page_count <= MAX_PAGES:
        return {**row, "pages": doc.page_count, "status": "rejected", "reason": "opens"}
    texts = [page.get_text() for page in doc]
    row["pages"] = doc.page_count

    full = "".join(texts)
    row["chars_per_page"] = round(len(full) / doc.page_count)
    if row["chars_per_page"] < MIN_CHARS_PER_PAGE:
        return {**row, "status": "rejected", "reason": "text_layer"}

    visible = re.sub(r"\s", "", full)
    row["letter_ratio"] = round(sum(c.isalpha() for c in visible) / max(len(visible), 1), 3)
    broken = (full.count("(cid:") * 6 + full.count("�")) / max(len(visible), 1)
    if row["letter_ratio"] < MIN_LETTER_RATIO or broken > MAX_BROKEN_RATIO:
        return {**row, "status": "rejected", "reason": "not_garbled"}

    row["lang"], prob = language(lid, " ".join(texts[:2]))
    row["lang_prob"] = round(prob, 3)
    if row["lang"] != "en":
        return {**row, "status": "rejected", "reason": "english"}

    page1 = norm_title(texts[0])
    row["title_score"] = round(fuzz.partial_ratio(norm_title(title), page1))
    if row["title_score"] < MIN_TITLE_SCORE:
        return {**row, "status": "rejected", "reason": "title_on_page1"}
    return {**row, "status": "accepted", "reason": ""}


def main() -> None:
    ap = argparse.ArgumentParser(description="SciLit Part 2: PDF quality gate")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--pdfs", default="data/pdfs")
    ap.add_argument("--models", default="models")
    ap.add_argument("--reports", default="reports")
    args = ap.parse_args()
    raw, pdf_dir, reports = Path(args.raw), Path(args.pdfs), Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)

    articles = [json.loads(l) for l in (raw / "articles.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    manifest = {}
    for line in (raw / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            e = json.loads(line)
            manifest[e["id"]] = e
    lid = load_lid(Path(args.models))

    rows = []
    for n, a in enumerate(articles, 1):
        m = manifest.get(a["id"])
        base = {"id": a["id"], "title": a["title"][:120], "themes": "|".join(a["themes"])}
        if not m:
            rows.append({**base, "status": "not_attempted", "reason": "not_downloaded_yet"})
        elif m["status"] != "ok":
            rows.append({**base, "status": "rejected", "reason": f"download_{m['reason']}"})
        else:
            rows.append({**base, **check(pdf_dir / f"{a['id']}.pdf", a["title"], lid)})
        if n % 200 == 0:
            print(f"  {n}/{len(articles)}", flush=True)

    fields = ["id", "status", "reason", "pages", "chars_per_page", "letter_ratio", "lang", "lang_prob",
              "title_score", "themes", "title"]
    with (reports / "validation_report.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    status = Counter(r["status"] for r in rows)
    reasons = Counter(r["reason"] for r in rows if r["status"] == "rejected")
    per_theme = Counter(t for r in rows if r["status"] == "accepted" for t in r["themes"].split("|"))
    lines = ["# Validation summary (Part 2)", "",
             f"- Articles: {len(rows)} · accepted: {status['accepted']} · rejected: {status['rejected']}"
             f" · not downloaded yet: {status['not_attempted']}", "",
             "| Rejection reason | Papers |", "| --- | --- |",
             *[f"| {k} | {v} |" for k, v in reasons.most_common()], "",
             "| Theme | Accepted papers |", "| --- | --- |",
             *[f"| {k} | {v} |" for k, v in sorted(per_theme.items())]]
    (reports / "validation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Accepted {status['accepted']}, rejected {status['rejected']} {dict(reasons)}, "
          f"not downloaded {status['not_attempted']}.")
    print(f"Report -> {reports / 'validation_report.csv'}")


if __name__ == "__main__":
    main()
