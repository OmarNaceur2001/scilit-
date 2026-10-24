"""
SciLit - generation/table.py
Part 9: comparison table. Same columns for every retrieved paper (the module's reading grid);
each cell is the source sentence itself + its ID, or "not reported" (never guessed).

Usage: python -m scilit.generation.table "LoRA vs full fine-tuning in low-resource settings"
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

COLUMNS = ["question", "method", "data", "results", "limits"]


@lru_cache(maxsize=1)
def records() -> dict:
    return {r["paper_id"]: r for r in map(json.loads, Path("data/parsed/records.jsonl").open(encoding="utf-8"))}


@lru_cache(maxsize=1)
def articles() -> dict:
    return {a["id"]: a for a in map(json.loads, Path("data/raw/articles.jsonl").open(encoding="utf-8"))}


def build_table(results: list[dict]) -> list[dict]:
    """One row per paper, in ranking order (results may hold several passages of one paper)."""
    rows, seen = [], set()
    for r in results:
        if r["paper_id"] in seen:
            continue
        seen.add(r["paper_id"])
        rec, art = records().get(r["paper_id"]), articles()[r["paper_id"]]
        row = {"paper_id": r["paper_id"], "title": art["title"], "year": art.get("arxiv_year") or art["year"],
               "first_author": (art["authors"] or ["?"])[0]}
        for col in COLUMNS:
            cell = (rec or {}).get("grid", {}).get(col) or []
            row[col] = {"text": cell[0]["text"], "sentence": cell[0]["id"], "hedging": cell[0]["hedging"]} if cell else None
        row["numeric"] = (rec or {}).get("numeric", [])[:3]
        rows.append(row)
    return rows


def to_markdown(rows: list[dict]) -> str:
    head = "| Paper | " + " | ".join(c.title() for c in COLUMNS) + " |\n|" + " --- |" * (len(COLUMNS) + 1)
    cut = lambda c: (c["text"][:120] + "…" if len(c["text"]) > 120 else c["text"]).replace("|", "/") if c else "*not reported*"
    lines = [f"| {r['title'][:60]} ({r['year']}) | " + " | ".join(cut(r[c]) for c in COLUMNS) + " |" for r in rows]
    return head + "\n" + "\n".join(lines)


if __name__ == "__main__":
    from scilit.retrieval.search import Searcher
    print(to_markdown(build_table(Searcher().search(sys.argv[1], k=8))))
