"""
SciLit - evaluation/retrieval_eval.py
Part 12a: retrieval scores on the 30 test queries, against the relevance labels.

Metrics (per query, then averaged):
  recall@10  share of the query's relevant papers (grade 2) found in the top 10
  MRR        1 / rank of the first relevant paper (grade 2)
  nDCG@10    graded: rewards grade-2 papers high up, grade-1 papers less (gains 2 and 1)
Unjudged papers count as not relevant (standard with pooling). Two safeguards against that bias:
  - the ablation runs' unjudged papers are judged too (judge.py extend -> qrels_chatgpt_extra.tsv)
  - "condensed" scores (Sakai, 2007): every ranking is scored after removing unjudged papers, so no
    system is penalised for papers nobody judged. A conclusion that holds in both tables is robust.

Comparisons (ablations):
  methods   bm25 | dense | hybrid | hybrid_rerank           (English queries)
  language  hybrid_rerank on the French versions            (multilingual BGE-M3)
  chunking  structure-aware vs fixed-size chunks vs fixed-size + title, BM25 (separates the effect
            of the boundaries from the effect of the header)
Paired bootstrap over queries (10,000 resamples): hybrid_rerank vs bm25 on nDCG@10.

Usage: python -m scilit.evaluation.retrieval_eval [--qrels labels/qrels_chatgpt.tsv]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from scilit.evaluation.judge import ABLATIONS, EXTRA, queries


def load_qrels(path: str) -> dict:
    q: dict = {}
    for f in (Path(path), EXTRA):
        if f.exists():
            for qid, pid, g in csv.reader(f.open(encoding="utf-8"), delimiter="\t"):
                q.setdefault(qid, {})[pid] = int(g)
    return q


def scores(ranked: list[str], rel: dict) -> tuple[float, float, float]:
    relevant = {p for p, g in rel.items() if g == 2}
    top = ranked[:10]
    recall = len(relevant & set(top)) / len(relevant) if relevant else 0.0
    mrr = next((1 / i for i, p in enumerate(ranked, 1) if p in relevant), 0.0)
    dcg = sum(rel.get(p, 0) / math.log2(i + 1) for i, p in enumerate(top, 1))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return recall, mrr, dcg / idcg if idcg else 0.0


def run(searcher, qs, qrels, method: str, field: str = "query_en", condensed: bool = False) -> np.ndarray:
    rows = []
    for q in qs:
        rel = qrels.get(q["qid"], {})
        ranked = [r["paper_id"] for r in searcher.search(q[field], k=40 if condensed else 20, method=method)]
        if condensed:
            ranked = [p for p in ranked if p in rel][:20]
        rows.append(scores(ranked, rel))
    return np.array(rows)


def bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10000) -> float:
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(a), (n, len(a)))
    return float((a[idx].mean(1) <= b[idx].mean(1)).mean())


def main() -> None:
    from scilit.retrieval.search import Searcher
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", default="labels/qrels_chatgpt.tsv")
    args = ap.parse_args()
    qs, qrels = queries(), load_qrels(args.qrels)
    searchers = {"structure": Searcher()}
    systems = [(f"{m} (EN)", "structure", m, "query_en") for m in ["bm25", "dense", "hybrid", "hybrid_rerank"]] + ABLATIONS
    tables = {False: [], True: []}
    per_query = {}
    for name, mode, method, field in systems:
        if not Path(f"data/index/{mode}/bm25.pkl").exists():
            continue
        searchers.setdefault(mode, Searcher(mode=mode))
        for condensed in (False, True):
            r = run(searchers[mode], qs, qrels, method, field, condensed)
            tables[condensed].append((name, *r.mean(0)))
            if not condensed:
                per_query[name] = r
        print(tables[False][-1], flush=True)
    p = bootstrap(per_query["hybrid_rerank (EN)"][:, 2], per_query["bm25 (EN)"][:, 2])
    extra = sum(1 for _ in EXTRA.open(encoding="utf-8")) if EXTRA.exists() else 0
    head = ["| System | Recall@10 | MRR | nDCG@10 |", "| --- | --- | --- | --- |"]
    row = lambda t: [f"| {n} | {a:.3f} | {b:.3f} | {c:.3f} |" for n, a, b, c in t]
    lines = ["# Retrieval evaluation (Part 12a)", "",
             f"30 test queries · labels: `{args.qrels}` + {extra} extra judgments of the ablation runs", "",
             "## Standard (unjudged = not relevant)", "", *head, *row(tables[False]), "",
             "## Condensed (unjudged papers removed before scoring, Sakai 2007)", "", *head, *row(tables[True]), "",
             f"Paired bootstrap (10,000 resamples over queries), hybrid_rerank vs bm25 on nDCG@10: p = {p:.4f}"]
    Path("reports/retrieval_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path("reports/retrieval_per_query.json").write_text(json.dumps({k: v.tolist() for k, v in per_query.items()}), encoding="utf-8")
    print("\n".join(lines[4:]))


if __name__ == "__main__":
    main()
