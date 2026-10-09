"""
SciLit -- corpus/index.py
=========================
Construit les index de recherche sur les Chunks.

Bronze (--mode bm25) :
  - Tokenisation regex (lowercase, no stop-words externes)
  - BM25Okapi via rank-bm25
  - Sauvegarde : data/parsed/bm25_index.pkl + bm25_meta.json
  - Recherche : query -> list[BM25Result]

Silver (--mode dense) :
  - Embeddings allenai/specter2_base (scientifique) ou all-MiniLM-L6-v2
  - Index FAISS FlatIP (cosine apres normalisation L2)
  - Sauvegarde : data/parsed/faiss_index.bin + faiss_meta.json

Silver (--mode rrf) :
  - Fusion BM25 + Dense via Reciprocal Rank Fusion
  - Retourne des RankedResult scores + is_relevant=None

Usage :
  python -m scilit.corpus.index                           # BM25 Bronze
  python -m scilit.corpus.index --mode dense              # Dense Silver
  python -m scilit.corpus.index --search "LoRA NLP"       # Test BM25
  python -m scilit.corpus.index --search "RAG" --mode rrf # Test RRF
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any
from uuid import UUID

from tqdm import tqdm

from scilit.corpus.parse import CHUNKS_PATH, load_chunks
from scilit.schemas import BM25Result, Chunk, DenseResult, RankedResult

# =============================================================================
# CONFIG
# =============================================================================

PARSED_DIR   = Path("data/parsed")
LOG_PATH     = PARSED_DIR / "index_log.txt"

# BM25
BM25_INDEX_PATH = PARSED_DIR / "bm25_index.pkl"
BM25_META_PATH  = PARSED_DIR / "bm25_meta.json"

# Dense
FAISS_INDEX_PATH = PARSED_DIR / "faiss_index.bin"
FAISS_META_PATH  = PARSED_DIR / "faiss_meta.json"

# Modeles d'embedding
MODEL_BRONZE = "sentence-transformers/all-MiniLM-L6-v2"   # 90 MB, rapide
MODEL_SILVER = "allenai/specter2_base"                     # 440 MB, scientifique

# RRF
RRF_K = 60   # constante de lissage standard (Cormack 2009)

# Mots vides anglais (sans dependance NLTK)
STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","up","about","into","through","during","is","are","was",
    "were","be","been","being","have","has","had","do","does","did","will",
    "would","could","should","may","might","shall","this","that","these",
    "those","i","we","you","he","she","they","it","its","our","their","his",
    "her","not","no","nor","so","yet","both","either","whether","because",
    "while","although","however","therefore","thus","hence","also","then",
    "than","as","if","when","where","which","who","whom","what","how",
    "paper","show","propose","present","approach","method","model","result",
    "experiment","using","use","based","can","new","we","also","our","use",
}

# =============================================================================
# LOGGING
# =============================================================================

PARSED_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# =============================================================================
# TOKENISATION
# =============================================================================

def tokenize(text: str, remove_stop: bool = True) -> list[str]:
    """
    Tokenisation simple et efficace (Bronze).
    - Convertit en minuscules
    - Extrait les tokens alphanumeriques
    - Retire les mots vides si remove_stop=True
    - Conserve les abreviations scientifiques (LoRA, BERT, RAG…)
    """
    tokens = re.findall(r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*", text.lower())
    if remove_stop:
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return tokens

# =============================================================================
# BM25 BRONZE
# =============================================================================

class BM25Index:
    """
    Encapsule le BM25Okapi avec les metadonnees de chunks.
    Permet search(query, k) -> list[BM25Result] sans reimporter les chunks.
    """

    def __init__(self, model: Any, chunk_ids: list[str], chunk_texts: list[str]) -> None:
        self._model      = model
        self._chunk_ids  = chunk_ids
        self._chunk_texts = chunk_texts

    def search(self, query: str, k: int = 10) -> list[BM25Result]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._model.get_scores(tokens)
        # Tri decroissant, top-k
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        results = []
        for rank, (idx, score) in enumerate(ranked, start=1):
            if score <= 0:
                break
            cid = self._chunk_ids[idx]
            # article_id est le prefixe du chunk_id dans notre schema
            # Le chunk_id est un UUID, l'article_id est stocke dans la meta
            results.append(BM25Result(
                chunk_id=UUID(cid),
                article_id=UUID(self._chunk_ids[idx]),  # placeholder, remplace par meta
                score=float(score),
                rank=rank,
            ))
        return results


class BM25IndexFull:
    """
    Version complete : conserve article_id par chunk.
    C'est celle utilisee en production.
    """

    def __init__(
        self,
        model: Any,
        chunks_meta: list[dict],   # [{chunk_id, article_id, section_type, text}, ...]
    ) -> None:
        self._model = model
        self._meta  = chunks_meta   # index -> dict

    def search(self, query: str, k: int = 10) -> list[BM25Result]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores  = self._model.get_scores(tokens)
        ranked  = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        results = []
        for rank, (idx, score) in enumerate(ranked, start=1):
            if score <= 0:
                break
            m = self._meta[idx]
            results.append(BM25Result(
                chunk_id=UUID(m["chunk_id"]),
                article_id=UUID(m["article_id"]),
                score=float(score),
                rank=rank,
            ))
        return results

    def save(self, index_path: Path = BM25_INDEX_PATH, meta_path: Path = BM25_META_PATH) -> None:
        with open(index_path, "wb") as f:
            pickle.dump(self._model, f)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, ensure_ascii=False)
        log.info(f"BM25 sauvegarde : {index_path} + {meta_path}")

    @classmethod
    def load(
        cls,
        index_path: Path = BM25_INDEX_PATH,
        meta_path:  Path = BM25_META_PATH,
    ) -> BM25IndexFull:
        with open(index_path, "rb") as f:
            model = pickle.load(f)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        log.info(f"BM25 charge : {len(meta)} chunks.")
        return cls(model, meta)


def build_bm25(
    chunks: list[Chunk],
    index_path: Path = BM25_INDEX_PATH,
    meta_path:  Path = BM25_META_PATH,
) -> BM25IndexFull:
    """
    Construit et sauvegarde l'index BM25Okapi.
    Tokenise chaque chunk.text et indexe.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError("pip install rank-bm25")

    log.info(f"BM25 : tokenisation de {len(chunks)} chunks...")
    corpus_tokens: list[list[str]] = []
    meta: list[dict] = []

    for chunk in tqdm(chunks, desc="Tokenisation", unit="chunk", ncols=80):
        tokens = tokenize(chunk.text)
        corpus_tokens.append(tokens if tokens else ["<empty>"])
        meta.append({
            "chunk_id":    str(chunk.id),
            "article_id":  str(chunk.article_id),
            "section_id":  str(chunk.section_id),
            "section_type": chunk.section_type.value,
            "text":        chunk.text[:300],   # preview pour debug
            "token_count": chunk.token_count or 0,
        })

    log.info("BM25 : construction de l'index BM25Okapi...")
    model = BM25Okapi(corpus_tokens)

    index = BM25IndexFull(model, meta)
    index.save(index_path, meta_path)
    log.info(f"BM25 construit : {len(chunks)} documents indexes.")
    return index

# =============================================================================
# DENSE SILVER
# =============================================================================

class DenseIndex:
    """
    Index FAISS pour la recherche semantique dense.
    Modele : allenai/specter2_base (scientifique) ou all-MiniLM-L6-v2 (leger).
    """

    def __init__(
        self,
        faiss_index: Any,
        meta: list[dict],
        model_name: str,
        dim: int,
    ) -> None:
        self._index      = faiss_index
        self._meta       = meta
        self._model_name = model_name
        self._dim        = dim

    def search(self, query: str, k: int = 10) -> list[DenseResult]:
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            import faiss
        except ImportError:
            raise ImportError("pip install sentence-transformers faiss-cpu")

        model  = SentenceTransformer(self._model_name)
        q_emb  = model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self._index.search(q_emb, k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:
                break
            m = self._meta[idx]
            results.append(DenseResult(
                chunk_id=UUID(m["chunk_id"]),
                article_id=UUID(m["article_id"]),
                cosine_score=float(score),
                rank=rank,
            ))
        return results

    def save(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        meta_path:  Path = FAISS_META_PATH,
    ) -> None:
        import faiss
        faiss.write_index(self._index, str(index_path))
        config = {"model": self._model_name, "dim": self._dim, "n": len(self._meta)}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"config": config, "meta": self._meta}, f, ensure_ascii=False)
        log.info(f"FAISS sauvegarde : {index_path} ({len(self._meta)} vecteurs)")

    @classmethod
    def load(
        cls,
        index_path: Path = FAISS_INDEX_PATH,
        meta_path:  Path = FAISS_META_PATH,
    ) -> DenseIndex:
        import faiss
        fi = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg  = data["config"]
        meta = data["meta"]
        log.info(f"FAISS charge : {cfg['n']} vecteurs (dim={cfg['dim']}).")
        return cls(fi, meta, cfg["model"], cfg["dim"])


def build_dense(
    chunks: list[Chunk],
    model_name: str = MODEL_BRONZE,
    batch_size: int = 64,
    index_path: Path = FAISS_INDEX_PATH,
    meta_path:  Path = FAISS_META_PATH,
) -> DenseIndex:
    """
    Encode tous les chunks avec sentence-transformers et construit FAISS FlatIP.
    Normalisation L2 avant indexation -> produit scalaire = cosine similarity.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        import faiss
    except ImportError:
        raise ImportError("pip install sentence-transformers faiss-cpu")

    log.info(f"Dense : chargement du modele {model_name}...")
    model = SentenceTransformer(model_name)
    dim   = model.get_sentence_embedding_dimension()
    log.info(f"  Dimension : {dim}")

    log.info(f"Dense : encodage de {len(chunks)} chunks (batch={batch_size})...")
    texts = [c.text for c in chunks]
    meta  = [
        {
            "chunk_id":    str(c.id),
            "article_id":  str(c.article_id),
            "section_type": c.section_type.value,
            "text":        c.text[:300],
        }
        for c in chunks
    ]

    # Encodage par batch avec barre de progression
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encodage", ncols=80):
        batch = texts[i : i + batch_size]
        embs  = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(embs)

    embeddings = np.vstack(all_embeddings).astype("float32")

    log.info(f"Dense : construction de l'index FAISS FlatIP ({dim}D)...")
    index = faiss.IndexFlatIP(dim)   # produit scalaire = cosine apres L2-norm
    index.add(embeddings)

    di = DenseIndex(index, meta, model_name, dim)
    di.save(index_path, meta_path)
    log.info(f"FAISS construit : {index.ntotal} vecteurs.")
    return di

# =============================================================================
# RRF FUSION
# =============================================================================

def rrf_fusion(
    bm25_results: list[BM25Result],
    dense_results: list[DenseResult],
    k: int = RRF_K,
    top_n: int = 10,
) -> list[RankedResult]:
    """
    Reciprocal Rank Fusion (Cormack et al., 2009).
    score_RRF(d) = sum_r [ 1 / (k + rank_r(d)) ]

    Combine BM25 et Dense sans avoir a calibrer les scores.
    Retourne top_n RankedResult fusionnes.
    """
    rrf_scores: dict[str, float] = {}
    bm25_rank_map:  dict[str, int]   = {}
    dense_rank_map: dict[str, int]   = {}
    chunk_to_article: dict[str, str] = {}

    for r in bm25_results:
        cid = str(r.chunk_id)
        bm25_rank_map[cid]      = r.rank
        chunk_to_article[cid]   = str(r.article_id)
        rrf_scores[cid]         = rrf_scores.get(cid, 0.0) + 1.0 / (k + r.rank)

    for r in dense_results:
        cid = str(r.chunk_id)
        dense_rank_map[cid]     = r.rank
        chunk_to_article[cid]   = str(r.article_id)
        rrf_scores[cid]         = rrf_scores.get(cid, 0.0) + 1.0 / (k + r.rank)

    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Charger les textes si besoin (depuis meta BM25)
    results: list[RankedResult] = []
    for final_rank, (cid, score) in enumerate(sorted_chunks[:top_n], start=1):
        results.append(RankedResult(
            query_id=UUID(int=0),           # placeholder -- a remplir par l'appelant
            chunk_id=UUID(cid),
            article_id=UUID(chunk_to_article[cid]),
            section_type=__import__("scilit.schemas", fromlist=["SectionType"]).SectionType.OTHER,
            chunk_text="",                  # a remplir via lookup si besoin
            bm25_rank=bm25_rank_map.get(cid),
            dense_rank=dense_rank_map.get(cid),
            rrf_score=score,
            final_rank=final_rank,
        ))
    return results

# =============================================================================
# STATS
# =============================================================================

def print_bm25_stats(index: BM25IndexFull) -> None:
    n = len(index._meta)
    avg_tokens = sum(m.get("token_count", 0) for m in index._meta) / max(n, 1)
    print(f"\n{'='*55}")
    print(f"  BM25 INDEX : {n} chunks indexes")
    print(f"  Tokens moyens/chunk : {avg_tokens:.0f}")
    print(f"  Index sauvegarde dans : {BM25_INDEX_PATH}")
    print(f"{'='*55}\n")

# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="SciLit -- construction index")
    p.add_argument(
        "--mode",
        choices=["bm25", "dense", "rrf"],
        default="bm25",
        help="bm25=Bronze | dense=Silver | rrf=Silver fusion",
    )
    p.add_argument(
        "--search",
        type=str,
        default=None,
        help="Requete de test apres construction de l'index",
    )
    p.add_argument(
        "--k",
        type=int,
        default=10,
        help="Nombre de resultats (default: 10)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=MODEL_BRONZE,
        help=f"Modele embedding Silver (default: {MODEL_BRONZE})",
    )
    args = p.parse_args()

    chunks = load_chunks()
    if not chunks:
        print("Aucun chunk. Lance d'abord :")
        print("  python -m scilit.corpus.parse --mode abstract")
        return

    log.info(f"Mode : {args.mode} | Chunks charges : {len(chunks)}")

    # ── BM25 ──────────────────────────────────────────────────────────────────
    if args.mode in ("bm25", "rrf"):
        bm25_idx = build_bm25(chunks)
        print_bm25_stats(bm25_idx)

        if args.search:
            print(f"\nRecherche BM25 : '{args.search}'")
            results = bm25_idx.search(args.search, k=args.k)
            for r in results:
                # Retrouver le texte dans la meta
                m = bm25_idx._meta[
                    next(i for i, m in enumerate(bm25_idx._meta)
                         if m["chunk_id"] == str(r.chunk_id))
                ]
                print(f"  [{r.rank}] score={r.score:.3f} | {m['text'][:100]}...")

    # ── DENSE ──────────────────────────────────────────────────────────────────
    if args.mode in ("dense", "rrf"):
        dense_idx = build_dense(chunks, model_name=args.model)

        if args.search:
            print(f"\nRecherche Dense : '{args.search}'")
            results = dense_idx.search(args.search, k=args.k)
            for r in results:
                print(f"  [{r.rank}] score={r.cosine_score:.3f} | chunk={str(r.chunk_id)[:8]}...")

    # ── RRF ────────────────────────────────────────────────────────────────────
    if args.mode == "rrf" and args.search:
        print(f"\nFusion RRF : '{args.search}'")
        b_res = bm25_idx.search(args.search,  k=50)
        d_res = dense_idx.search(args.search, k=50)
        fused = rrf_fusion(b_res, d_res, top_n=args.k)
        for r in fused:
            print(
                f"  [{r.final_rank}] rrf={r.rrf_score:.4f} "
                f"bm25_rank={r.bm25_rank} dense_rank={r.dense_rank} "
                f"| {str(r.chunk_id)[:8]}..."
            )


if __name__ == "__main__":
    main()