"""
SciLit — retrieval/pipeline.py
================================
Étape 6 : Pipeline BM25 → Dense → RRF

Charge les index depuis data/parsed/, fusionne via RRF,
retourne des RankedResult enrichis (chunk_text + section_type réels).

Usage :
  python -m scilit.retrieval.pipeline --query "LoRA fine-tuning" --k 5
  python -m scilit.retrieval.pipeline --query "RAG retrieval" --mode bm25
  python -m scilit.retrieval.pipeline --query "BERT NLP" --mode dense
"""
from __future__ import annotations

import argparse
import logging
from functools import lru_cache
from uuid import uuid4

from scilit.corpus.index import BM25IndexFull, DenseIndex, rrf_fusion
from scilit.corpus.parse import load_chunks
from scilit.schemas import RankedResult, SectionType

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Lazy loaders — chargement unique par session Python
# ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _bm25() -> BM25IndexFull:
    log.info("Chargement BM25 index...")
    return BM25IndexFull.load()


@lru_cache(maxsize=1)
def _dense() -> DenseIndex:
    log.info("Chargement FAISS index...")
    return DenseIndex.load()


@lru_cache(maxsize=1)
def _chunk_lookup() -> dict[str, dict]:
    """chunk_id str → {text, section_type} pour enrichir les RankedResult."""
    log.info("Chargement chunks pour lookup texte...")
    return {
        str(c.id): {"text": c.text, "section_type": c.section_type}
        for c in load_chunks()
    }


# ─────────────────────────────────────────────────────────────
# Fonction principale
# ─────────────────────────────────────────────────────────────

def search(
    query:      str,
    k:          int = 10,
    mode:       str = "rrf",
    bm25_pool:  int = 50,
    dense_pool: int = 50,
) -> list[RankedResult]:
    """
    Recherche principale.

    mode="rrf"   : BM25 + Dense fusionnés via RRF (défaut, Bronze complet)
    mode="bm25"  : BM25 seul (lexical)
    mode="dense" : Dense seul (sémantique)

    bm25_pool / dense_pool : candidats récupérés avant fusion (doit être > k).
    """
    if not query.strip():
        return []

    query_id = uuid4()
    lookup   = _chunk_lookup()

    # ── Appels aux index ────────────────────────────────────────────────────
    bm25_results  = _bm25().search(query, k=bm25_pool if mode == "rrf" else k) \
                    if mode in ("bm25", "rrf") else []
    dense_results = _dense().search(query, k=dense_pool if mode == "rrf" else k) \
                    if mode in ("dense", "rrf") else []

    # ── Helpers d'enrichissement ────────────────────────────────────────────
    def _enrich(r: RankedResult) -> RankedResult:
        info = lookup.get(str(r.chunk_id), {})
        return RankedResult(
            query_id     = query_id,
            chunk_id     = r.chunk_id,
            article_id   = r.article_id,
            section_type = info.get("section_type", SectionType.OTHER),
            chunk_text   = info.get("text", ""),
            bm25_rank    = r.bm25_rank,
            dense_rank   = r.dense_rank,
            rrf_score    = r.rrf_score,
            final_rank   = r.final_rank,
        )

    # ── BM25 seul ───────────────────────────────────────────────────────────
    if mode == "bm25":
        return [
            _enrich(RankedResult(
                query_id     = query_id,
                chunk_id     = r.chunk_id,
                article_id   = r.article_id,
                section_type = SectionType.OTHER,
                chunk_text   = "",
                bm25_rank    = r.rank,
                dense_rank   = None,
                rrf_score    = None,
                final_rank   = r.rank,
            ))
            for r in bm25_results[:k]
        ]

    # ── Dense seul ──────────────────────────────────────────────────────────
    if mode == "dense":
        return [
            _enrich(RankedResult(
                query_id     = query_id,
                chunk_id     = r.chunk_id,
                article_id   = r.article_id,
                section_type = SectionType.OTHER,
                chunk_text   = "",
                bm25_rank    = None,
                dense_rank   = r.rank,
                rrf_score    = None,
                final_rank   = r.rank,
            ))
            for r in dense_results[:k]
        ]

    # ── RRF fusion ──────────────────────────────────────────────────────────
    return [_enrich(r) for r in rrf_fusion(bm25_results, dense_results, top_n=k)]


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _print_results(results: list[RankedResult], query: str, mode: str) -> None:
    print(f"\n{'='*65}")
    print(f"  QUERY  : {query}")
    print(f"  MODE   : {mode.upper()} | TOP-{len(results)}")
    print(f"{'='*65}")
    for r in results:
        if r.rrf_score is not None:
            score = f"rrf={r.rrf_score:.4f}  bm25={r.bm25_rank}  dense={r.dense_rank}"
        elif r.bm25_rank:
            score = f"bm25_rank={r.bm25_rank}"
        else:
            score = f"dense_rank={r.dense_rank}"
        preview = r.chunk_text[:120].replace("\n", " ")
        print(f"  [{r.final_rank:2d}] {score}")
        print(f"       article={str(r.article_id)[:8]}  section={r.section_type.value}")
        print(f"       {preview}...")
        print()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    p = argparse.ArgumentParser(description="SciLit — retrieval pipeline")
    p.add_argument("--query", required=True, help="Requête de recherche")
    p.add_argument("--k",     type=int, default=5, help="Top-k résultats")
    p.add_argument(
        "--mode",
        choices=["rrf", "bm25", "dense"],
        default="rrf",
        help="Mode retrieval (défaut: rrf)",
    )
    args = p.parse_args()

    results = search(args.query, k=args.k, mode=args.mode)
    _print_results(results, args.query, args.mode)
    print("Prochain : python -m scilit.generation.template")


if __name__ == "__main__":
    main()