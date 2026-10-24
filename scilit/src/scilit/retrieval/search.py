"""
SciLit - retrieval/search.py
Part 6: query -> ranked papers with their best passage.

  1. candidates : BM25 top 100 (exact words, names, numbers) + dense top 100 (meaning; BGE-M3 is
                  multilingual, so French / Arabic queries reach English papers without translation)
  2. fusion     : Reciprocal Rank Fusion, score = sum 1 / (60 + rank)  (Cormack et al., 2009):
                  uses ranks only, because BM25 scores and cosine similarities are not comparable
  3. rerank     : cross-encoder bge-reranker-v2-m3 reads query and passage TOGETHER (full attention
                  between them): more precise than comparing two separate vectors, but one forward
                  pass per pair, so only on the 50 fused candidates
  4. cut        : at most `per_paper` chunks per paper (1 = a ranking of papers, used by the
                  evaluation and the table; 2 = evidence passages, used by the review)
  5. context    : each kept chunk is returned with its previous and next chunks (small-to-big)

Methods (for the ablation): bm25 | dense | hybrid | hybrid_rerank (default)

Usage: python -m scilit.retrieval.search "LoRA vs full fine-tuning in low-resource settings"
       python -m scilit.retrieval.search "..." --method bm25 --k 5
"""
from __future__ import annotations

import argparse
import json
import pickle
from functools import cached_property
from pathlib import Path

from scilit.retrieval.build_index import tokenize

QUERY_PREFIX = {"BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: "}


class Searcher:
    def __init__(self, index_dir="data/index", mode="structure", dense_model="BAAI/bge-m3",
                 reranker="BAAI/bge-reranker-v2-m3", articles="data/raw/articles.jsonl"):
        self.dir = Path(index_dir) / mode
        self.chunks_path = Path(index_dir) / f"chunks_{mode}.jsonl"
        self.dense_name, self.rerank_name, self.articles_path = dense_model, reranker, Path(articles)

    # --- lazy loading: each resource is read once, when first needed ---
    @cached_property
    def chunks(self) -> dict:
        return {c["id"]: c for c in map(json.loads, self.chunks_path.open(encoding="utf-8"))}

    @cached_property
    def chunks_by_paper(self) -> dict:
        """paper id -> its chunk ids (the reader ranks the passages of ONE paper)."""
        by: dict = {}
        for cid, c in self.chunks.items():
            by.setdefault(c["paper_id"], []).append(cid)
        return by

    @cached_property
    def titles(self) -> dict:
        return {a["id"]: a["title"] for a in map(json.loads, self.articles_path.open(encoding="utf-8"))}

    @cached_property
    def bm25(self) -> dict:
        with (self.dir / "bm25.pkl").open("rb") as f:
            return pickle.load(f)

    @cached_property
    def dense(self):
        import faiss
        import torch
        from sentence_transformers import SentenceTransformer
        d = self.dir / self.dense_name.replace("/", "__")
        model = SentenceTransformer(self.dense_name, device="cuda" if torch.cuda.is_available() else "cpu")
        return model, faiss.read_index(str(d / "faiss.index")), json.loads((d / "ids.json").read_text())["ids"]

    @cached_property
    def reranker(self):
        import torch
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(self.rerank_name, max_length=512, device="cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            ce.model.half()
        return ce

    # --- the four stages ---
    def bm25_top(self, query: str, n: int = 100) -> list[str]:
        scores = self.bm25["bm25"].get_scores(tokenize(query))
        best = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
        return [self.bm25["ids"][i] for i in best if scores[i] > 0]

    def dense_top(self, query: str, n: int = 100) -> list[str]:
        model, index, ids = self.dense
        q = model.encode([QUERY_PREFIX.get(self.dense_name, "") + query], normalize_embeddings=True)
        _, idx = index.search(q.astype("float32"), n)
        return [ids[i] for i in idx[0] if i >= 0]

    @staticmethod
    def rrf(*rankings: list[str], k: int = 60) -> list[str]:
        score: dict[str, float] = {}
        for ranking in rankings:
            for rank, cid in enumerate(ranking, 1):
                score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank)
        return sorted(score, key=lambda c: -score[c])

    def rerank(self, query: str, ids: list[str]) -> list[str]:
        pairs = [(query, self.chunks[c]["header"] + "\n" + self.chunks[c]["text"]) for c in ids]
        scores = self.reranker.predict(pairs, batch_size=16)
        return [c for _, c in sorted(zip(scores, ids), key=lambda x: -x[0])]

    def search(self, query: str, k: int = 10, method: str = "hybrid_rerank", per_paper: int = 1) -> list[dict]:
        """One ranking of passages, cut into k results with at most `per_paper` passages per paper.
        per_paper=1 -> a ranking of PAPERS (evaluation, comparison table);
        per_paper=2 -> a ranking of EVIDENCE PASSAGES (the review: a paper can bring both its
        method and its result paragraph)."""
        if method == "bm25":
            ranked = self.bm25_top(query)
        elif method == "dense":
            ranked = self.dense_top(query)
        else:
            ranked = self.rrf(self.bm25_top(query), self.dense_top(query))
            if method == "hybrid_rerank":
                ranked = self.rerank(query, ranked[:50]) + ranked[50:]
        results, taken = [], {}
        for cid in ranked:
            c = self.chunks[cid]
            if taken.get(c["paper_id"], 0) >= per_paper:   # one long paper cannot fill the list
                continue
            taken[c["paper_id"]] = taken.get(c["paper_id"], 0) + 1
            ctx = [self.chunks[x]["text"] for x in (c["prev"], cid, c["next"]) if x]
            results.append({"rank": len(results) + 1, "paper_id": c["paper_id"],
                            "title": self.titles.get(c["paper_id"], ""), "chunk_id": cid,
                            "section": c["section_type"], "text": c["text"], "context": " ".join(ctx),
                            "sentence_ids": c["sentence_ids"]})
            if len(results) == k:
                break
        return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--method", default="hybrid_rerank", choices=["bm25", "dense", "hybrid", "hybrid_rerank"])
    args = ap.parse_args()
    for r in Searcher().search(args.query, args.k, args.method):
        print(f"{r['rank']:2}. {r['title'][:90]}\n    [{r['section']}] {r['text'][:160]}...")


if __name__ == "__main__":
    main()
