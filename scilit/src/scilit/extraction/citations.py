"""
SciLit — extraction/citations.py
=================================
Étape 4 : Construit InTextCitation et CitationGraph depuis refs_raw.jsonl.

Bronze : références issues des métadonnées OpenAlex (pas d'in-text parsing).
         context_sentence = placeholder, function = UNDEFINED.
Silver : in-text regex + classifieur de fonction (Lot C).

Sortie :
  data/parsed/citations.jsonl      — une InTextCitation par ligne
  data/parsed/citation_graph.json  — CitationGraph sérialisé

Usage :
  python -m scilit.extraction.citations
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from tqdm import tqdm

from scilit.schemas import (
    CitationFunction,
    CitationGraph,
    CitationGraphEdge,
    InTextCitation,
)

log = logging.getLogger(__name__)

RAW_ARTICLES  = Path("data/raw/articles.jsonl")
REFS_RAW      = Path("data/parsed/refs_raw.jsonl")
OUT_CITATIONS = Path("data/parsed/citations.jsonl")
OUT_GRAPH     = Path("data/parsed/citation_graph.json")

BRONZE_CONTEXT = "[Bronze: metadata-only reference, no in-text context]"


# ─────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────

def _load_articles() -> dict[str, dict]:
    """Returns {scilit_uuid_str → article_dict}."""
    return {
        a["id"]: a
        for line in RAW_ARTICLES.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for a in [json.loads(line)]
    }


def _load_refs() -> list[dict]:
    if not REFS_RAW.exists():
        raise FileNotFoundError(
            f"{REFS_RAW} introuvable. Lance d'abord :\n"
            "  python -m scilit.extraction.fetch_refs"
        )
    return [
        json.loads(l)
        for l in REFS_RAW.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


# ─────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────

def build_citations(
    uuid_map: dict[str, dict],
    refs: list[dict],
) -> tuple[list[InTextCitation], CitationGraph]:
    """
    1. Construit oa_to_uuid : openalex_id → scilit_uuid depuis refs_raw
       (chaque entrée contient l'openalex_id de l'article lui-même)
    2. Pour chaque article : parcourt ses referenced_works
         → interne si la cible est dans le corpus (target_article_id non-None)
         → externe sinon (target_article_id = None)
    3. Agrège les edges internes (même paire → count++)
    4. Construit CitationGraph
    """

    # ── 1. openalex_id → scilit_uuid ────────────────────────────────────────
    oa_to_uuid: dict[str, str] = {
        r["openalex_id"]: r["scilit_uuid"]
        for r in refs
        if r.get("openalex_id")
    }
    log.info(
        "Corpus : %d articles | %d avec openalex_id résolu.",
        len(uuid_map), len(oa_to_uuid),
    )

    citations:    list[InTextCitation] = []
    edge_counter: dict[tuple[str, str], int] = {}

    # ── 2. Parcours des références ───────────────────────────────────────────
    for entry in tqdm(refs, desc="Citations", unit="article", ncols=80):
        src_uuid_str = entry["scilit_uuid"]

        for oa_ref_id in entry.get("references", []):
            tgt_uuid_str = oa_to_uuid.get(oa_ref_id)  # None → hors corpus

            citations.append(InTextCitation(
                source_article_id = UUID(src_uuid_str),
                target_article_id = UUID(tgt_uuid_str) if tgt_uuid_str else None,
                target_raw_ref    = oa_ref_id,
                context_sentence  = BRONZE_CONTEXT,
                function          = CitationFunction.UNDEFINED,
                chunk_id          = None,
            ))

            if tgt_uuid_str:
                key = (src_uuid_str, tgt_uuid_str)
                edge_counter[key] = edge_counter.get(key, 0) + 1

    # ── 3. Edges agrégés ────────────────────────────────────────────────────
    edges: list[CitationGraphEdge] = [
        CitationGraphEdge(
            source_id = UUID(src),
            target_id = UUID(tgt),
            function  = CitationFunction.UNDEFINED,
            count     = cnt,
        )
        for (src, tgt), cnt in edge_counter.items()
    ]

    # ── 4. CitationGraph ────────────────────────────────────────────────────
    graph = CitationGraph(
        corpus_snapshot_date = datetime.now(timezone.utc),
        nodes                = [UUID(uid) for uid in uuid_map],
        edges                = edges,
    )

    return citations, graph


# ─────────────────────────────────────────────────────────────
# Save / Stats
# ─────────────────────────────────────────────────────────────

def _save(citations: list[InTextCitation], graph: CitationGraph) -> None:
    with OUT_CITATIONS.open("w", encoding="utf-8") as f:
        for c in citations:
            f.write(c.model_dump_json() + "\n")
    log.info("%d citations → %s", len(citations), OUT_CITATIONS)

    OUT_GRAPH.write_text(
        graph.model_dump_json(indent=2),
        encoding="utf-8",
    )
    log.info("CitationGraph → %s", OUT_GRAPH)


def _print_stats(citations: list[InTextCitation], graph: CitationGraph) -> None:
    internal = sum(1 for c in citations if c.target_article_id is not None)
    external = len(citations) - internal
    avg_refs = len(citations) / max(len(graph.nodes), 1)

    print(f"\n{'='*55}")
    print(f"  CITATION GRAPH")
    print(f"  Nœuds (articles)  : {len(graph.nodes)}")
    print(f"  Arêtes internes   : {len(graph.edges)}")
    print(f"  Densité           : {graph.density:.5f}")
    print(f"  Références totales: {len(citations)}")
    print(f"    → internes      : {internal}  ({100*internal/max(len(citations),1):.1f}%)")
    print(f"    → externes      : {external}  ({100*external/max(len(citations),1):.1f}%)")
    print(f"  Refs/article moy  : {avg_refs:.1f}")
    print(f"{'='*55}")
    print("\nProchain : python -m scilit.corpus.index --mode dense")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    uuid_map = _load_articles()
    refs     = _load_refs()
    log.info("%d entrées dans refs_raw.jsonl.", len(refs))

    citations, graph = build_citations(uuid_map, refs)
    _save(citations, graph)
    _print_stats(citations, graph)


if __name__ == "__main__":
    main()