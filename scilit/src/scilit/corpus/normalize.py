"""
SciLit - corpus/normalize.py
Part 1 steps 2-4: metadata correction, deduplication, licence record.

Input : data/raw/candidates.jsonl  (from collect.py; OpenAlex metadata)
Output: data/raw/articles.jsonl    clean corpus, one paper per line, stable UUID per paper
        data/raw/dedup_log.jsonl   every merge and the rule that fired
        reports/data_quality.md    how much OpenAlex metadata disagreed with arXiv, duplicates, licences

Design:
  - Identity = identifiers (DOI, arXiv ID), never titles alone.
  - Title / authors / abstract come from arXiv when the paper has an arXiv ID (authoritative);
    otherwise from OpenAlex, to be checked against the PDF in Part 2.
  - Duplicates (e.g. arXiv preprint + conference version as two OpenAlex records) are merged;
    the published version is kept and the other becomes an alias.

Usage (from the repository root, with PYTHONPATH=scilit/src):
  python -m scilit.corpus.normalize
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from rapidfuzz import fuzz

from scilit.corpus import arxiv_meta
from scilit.corpus.collect import METHOD_ORDER, norm_title

TITLE_AGREE_MIN = 90     # fuzzy score above which two titles are "the same"


# --------------------------------------------------------------------------
# Step 2 · metadata correction from arXiv
# --------------------------------------------------------------------------

def apply_arxiv(papers: list[dict], truth: dict[str, dict]) -> list[dict]:
    """Overwrite title/authors/abstract with arXiv's when available; keep OpenAlex's title
    when they disagree, so the disagreement is visible and countable."""
    mismatches = []
    for p in papers:
        ax = truth.get(p.get("arxiv_id") or "")
        if not ax:
            p["title_source"] = "openalex"
            continue
        if not arxiv_meta.same_paper(p["authors"], p["year"], ax):
            # OpenAlex attached another paper's data to this ID: its reference list is untrusted
            p["openalex_conflated"] = True
            p["referenced_works"] = []
            p["year"] = ax["year"]
        score = fuzz.ratio(norm_title(p["title"]), norm_title(ax["title"]))
        if score < TITLE_AGREE_MIN:
            p["openalex_title"] = p["title"]
            mismatches.append({"openalex_id": p["openalex_id"], "arxiv_id": p["arxiv_id"],
                               "openalex_title": p["title"], "arxiv_title": ax["title"],
                               "similarity": round(score),
                               "conflated": bool(p.get("openalex_conflated"))})
        p["title"] = ax["title"]
        p["authors"] = ax["authors"][:20] or p["authors"]
        p["abstract"] = ax["abstract"] or p["abstract"]
        p["arxiv_year"] = ax["year"]
        p["title_source"] = "arxiv"
    return mismatches


# --------------------------------------------------------------------------
# Step 3 · deduplication (union-find over identifier keys)
# --------------------------------------------------------------------------

def first_author_key(p: dict) -> str:
    return arxiv_meta.surname(p["authors"][0]) if p.get("authors") else ""


def keys(p: dict) -> list[str]:
    """Every identifier that proves two records are the same paper. The title key ignores the
    year on purpose: a 2021 preprint and its 2022 conference version share title and author."""
    k = []
    if p.get("doi") and not p["doi"].startswith("10.48550/arxiv"):
        k.append("doi:" + p["doi"])
    if p.get("arxiv_id"):
        k.append("arxiv:" + p["arxiv_id"])
    t = norm_title(p["title"])
    if len(t) >= 20:                      # short titles ("Introduction") are too ambiguous
        k.append(f"title:{t}|{first_author_key(p)}")
    return k


def group_duplicates(papers: list[dict]) -> list[list[dict]]:
    parent = list(range(len(papers)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner: dict[str, int] = {}
    for i, p in enumerate(papers):
        for k in keys(p):
            if k in owner:
                parent[find(i)] = find(owner[k])
            else:
                owner[k] = i
    groups: dict[int, list[dict]] = {}
    for i, p in enumerate(papers):
        groups.setdefault(find(i), []).append(p)
    return list(groups.values())


def is_published(p: dict) -> bool:
    venue = (p.get("venue") or "").lower()
    return bool(venue) and "arxiv" not in venue


def merge_group(group: list[dict]) -> tuple[dict, dict | None]:
    """Keep the published version (else the most cited); fold the others into it."""
    keep = max(group, key=lambda p: (is_published(p), p["cited_by_count"]))
    if len(group) == 1:
        return keep, None
    others = [p for p in group if p is not keep]
    keep = dict(keep)
    keep["aliases"] = [p["openalex_id"] for p in others]
    keep["doi"] = keep.get("doi") or next((p["doi"] for p in others if p.get("doi")), None)
    for f in ("arxiv_id", "arxiv_year", "pdf_url", "license"):
        keep[f] = keep.get(f) or next((p.get(f) for p in others if p.get(f)), None)
    if not keep.get("title_source") == "arxiv":                 # prefer arXiv-corrected metadata
        ax = next((p for p in others if p.get("title_source") == "arxiv"), None)
        if ax:
            for f in ("title", "authors", "abstract", "title_source", "openalex_title"):
                if f in ax:
                    keep[f] = ax[f]
    keep["cited_by_count"] = max(p["cited_by_count"] for p in group)   # max, not sum: overlap
    keep["referenced_works"] = sorted({r for p in group for r in p["referenced_works"]})
    keep["themes"] = sorted({t for p in group for t in p["themes"]})
    found: dict[str, str] = {}
    for p in group:
        for t, m in p["found_by"].items():
            if t not in found or METHOD_ORDER[m] < METHOD_ORDER[found[t]]:
                found[t] = m
    keep["found_by"] = found
    counts = Counter(k for p in group for k in set(keys(p)))
    log = {"kept": keep["openalex_id"], "merged": keep["aliases"],
           "rule": sorted(k.split(":", 1)[0] for k, c in counts.items() if c > 1)}  # doi/arxiv/title
    return keep, log


# --------------------------------------------------------------------------
# Step 4 · licence record + final shape
# --------------------------------------------------------------------------

def finalize(p: dict) -> dict:
    """Stable ID, PDF location, licence. Open access is guaranteed upstream (collect filters);
    here we only record which licence applies, since we never redistribute the PDFs."""
    p["id"] = str(uuid5(NAMESPACE_URL, f"https://openalex.org/{p['openalex_id']}"))
    if p.get("arxiv_id"):
        p["pdf_url"] = f"https://arxiv.org/pdf/{p['arxiv_id']}"     # stable, always a PDF
        p["license"] = p.get("license") or "arxiv"
    p["license"] = p.get("license") or "unknown"
    return p


# --------------------------------------------------------------------------
# Report + main
# --------------------------------------------------------------------------

def write_report(path: Path, n_in: int, articles: list[dict], mismatches: list[dict],
                 merges: list[dict]) -> None:
    with_arxiv = sum(1 for a in articles if a.get("arxiv_id"))
    lic = Counter(a["license"] for a in articles)
    lines = [
        "# Data quality report (Part 1)", "",
        f"- Candidates in: {n_in}",
        f"- Duplicate groups merged: {len(merges)} (records removed: {sum(len(m['merged']) for m in merges)})",
        f"- Articles out: {len(articles)}",
        f"- With arXiv ID (metadata from arXiv): {with_arxiv} ({with_arxiv / max(len(articles), 1):.0%})",
        f"- OpenAlex titles that disagree with arXiv: {len(mismatches)}",
        f"- OpenAlex records that are another paper: {sum(1 for a in articles if a.get('openalex_conflated'))}", "",
        "## Licences", "", "| Licence | Papers |", "| --- | --- |",
        *[f"| {k} | {v} |" for k, v in lic.most_common()], "",
        "## Title disagreements (OpenAlex vs arXiv)", "",
    ]
    lines += [f"- {m['arxiv_id']} · OpenAlex: \"{m['openalex_title']}\" · arXiv: \"{m['arxiv_title']}\" "
              f"({m['similarity']}/100)" + (" · **another paper's record: references dropped**"
                                            if m["conflated"] else " · title only")
              for m in mismatches] or ["None."]
    conflated = [a for a in articles if a.get("openalex_conflated")]
    lines += ["", f"Records where OpenAlex attached another paper's data (authors/year differ from arXiv): "
              f"{len(conflated)}. Metadata taken from arXiv, reference list dropped."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="SciLit Part 1: correct, deduplicate, record licences")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--reports", default="reports")
    args = ap.parse_args()
    raw, reports = Path(args.raw), Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)

    papers = [json.loads(line) for line in (raw / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
              if line.strip()]
    print(f"{len(papers)} candidates. Fetching arXiv metadata for "
          f"{sum(1 for p in papers if p.get('arxiv_id'))} papers (3 s between requests)...", flush=True)
    wanted = {p["arxiv_id"] for p in papers if p.get("arxiv_id")}
    truth = arxiv_meta.fetch(list(wanted))
    if len(truth) < len(wanted):
        print(f"WARNING: arXiv returned {len(truth)}/{len(wanted)} papers; the rest keep OpenAlex "
              "metadata and are flagged title_source=openalex.", flush=True)
    mismatches = apply_arxiv(papers, truth)

    articles, merges = [], []
    for group in group_duplicates(papers):
        kept, log = merge_group(group)
        articles.append(finalize(kept))
        if log:
            merges.append(log)

    with (raw / "articles.jsonl").open("w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    with (raw / "dedup_log.jsonl").open("w", encoding="utf-8") as f:
        for m in merges:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    write_report(reports / "data_quality.md", len(papers), articles, mismatches, merges)
    print(f"Done: {len(articles)} articles ({len(merges)} duplicate groups merged, "
          f"{len(mismatches)} OpenAlex titles corrected from arXiv).")
    print(f"Report -> {reports / 'data_quality.md'}")


if __name__ == "__main__":
    main()
