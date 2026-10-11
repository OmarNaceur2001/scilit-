"""
SciLit — api/main.py
=====================
Étape 9 : API FastAPI Bronze.

Endpoints :
  GET  /                        — health check + stats corpus
  POST /search                  — retrieval BM25/Dense/RRF
  POST /generate                — génération + NLI en une passe
  GET  /review/{slug}           — lire un ReviewDraft sauvegardé
  GET  /export/bibtex/{slug}    — télécharger le .bib d'une review
  GET  /claims                  — lister/filtrer les claims

Usage :
  uvicorn scilit.api.main:app --reload --port 8000
  # puis ouvrir http://localhost:8000/docs
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title        = "SciLit API",
    description  = "Bronze literature-review assistant — ESPRIT NLP 2026-2027",
    version      = "0.1.0-bronze",
    docs_url     = "/docs",
    redoc_url    = "/redoc",
)

PARSED_DIR    = Path("data/parsed")
RAW_DIR       = Path("data/raw")
REVIEWS_DIR   = PARSED_DIR / "reviews"
CLAIMS_PATH   = PARSED_DIR / "claims.jsonl"
ARTICLES_PATH = RAW_DIR    / "articles.jsonl"


# ─────────────────────────────────────────────────────────────
# Lazy loaders (chargés une seule fois au premier appel)
# ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_retrieval():
    from scilit.retrieval.pipeline import search
    return search


@lru_cache(maxsize=1)
def _get_claims() -> list[dict]:
    if not CLAIMS_PATH.exists():
        return []
    return [
        json.loads(l)
        for l in CLAIMS_PATH.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


@lru_cache(maxsize=1)
def _get_articles() -> dict[str, dict]:
    return {
        a["id"]: a
        for line in ARTICLES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for a in [json.loads(line)]
    }


# ─────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str  = Field(..., min_length=2, example="hallucination language model")
    k:     int  = Field(10, ge=1, le=50)
    mode:  Literal["rrf", "bm25", "dense"] = "rrf"


class SearchResult(BaseModel):
    rank:         int
    chunk_id:     str
    article_id:   str
    section_type: str
    rrf_score:    float | None
    bm25_rank:    int   | None
    dense_rank:   int   | None
    preview:      str


class SearchResponse(BaseModel):
    query:   str
    mode:    str
    results: list[SearchResult]


class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=2, example="transformer attention NLP")
    k:     int = Field(20, ge=5, le=50, description="Top-k chunks retrieval")
    max_sentences_per_type: int = Field(3, ge=1, le=10)
    run_nli: bool = Field(True,  description="Lancer la vérification NLI après génération")


class GenerateResponse(BaseModel):
    slug:                    str
    title:                   str
    n_sentences:             int
    n_bibtex:                int
    hallucination_rate:      float
    support_rate:            float
    overall_hallucination_rate: float
    review_path:             str
    audit_path:              str | None


class ClaimItem(BaseModel):
    id:          str
    article_id:  str
    claim_type:  str
    hedging:     str
    text:        str
    metric_name:  str | None
    metric_value: float | None


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/", summary="Health check + corpus stats")
def root():
    n_articles = sum(
        1 for l in ARTICLES_PATH.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ) if ARTICLES_PATH.exists() else 0
    n_claims   = len(_get_claims())
    n_reviews  = len([
        f for f in REVIEWS_DIR.glob("*.json")
        if not f.stem.endswith(("_verified", "_audit"))
    ]) if REVIEWS_DIR.exists() else 0

    return {
        "status":     "ok",
        "version":    "0.1.0-bronze",
        "corpus":     {"articles": n_articles, "claims": n_claims},
        "reviews":    n_reviews,
        "index":      {
            "bm25":  (PARSED_DIR / "bm25_index.pkl").exists(),
            "faiss": (PARSED_DIR / "faiss_index.bin").exists(),
        },
    }


@app.post("/search", response_model=SearchResponse, summary="Retrieval BM25 / Dense / RRF")
def search(req: SearchRequest):
    search_fn = _get_retrieval()
    try:
        ranked = search_fn(req.query, k=req.k, mode=req.mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    results = [
        SearchResult(
            rank         = r.final_rank,
            chunk_id     = str(r.chunk_id),
            article_id   = str(r.article_id),
            section_type = r.section_type.value,
            rrf_score    = round(r.rrf_score,  4) if r.rrf_score  is not None else None,
            bm25_rank    = r.bm25_rank,
            dense_rank   = r.dense_rank,
            preview      = r.chunk_text[:200],
        )
        for r in ranked
    ]
    return SearchResponse(query=req.query, mode=req.mode, results=results)


@app.post("/generate", response_model=GenerateResponse, summary="Génération + NLI")
def generate(req: GenerateRequest):
    import re
    import unicodedata

    from scilit.generation.template import generate as gen_fn
    from scilit.generation.template import _slugify

    try:
        draft, bibtex = gen_fn(
            query                  = req.query,
            k                      = req.k,
            max_sentences_per_type = req.max_sentences_per_type,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    slug = _slugify(req.query)

    # ── Sauvegarde template ────────────────────────────────────────────────
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    review_path = REVIEWS_DIR / f"{slug}.json"
    bib_path    = REVIEWS_DIR / f"{slug}.bib"
    review_path.write_text(draft.model_dump_json(indent=2), encoding="utf-8")
    bib_path.write_text(
        "\n\n".join(e.to_bibtex() for e in bibtex),
        encoding="utf-8",
    )

    audit_path: str | None = None

    # ── NLI optionnelle ───────────────────────────────────────────────────
    if req.run_nli:
        try:
            from scilit.generation.nli_verify import (
                _load_articles_lookup,
                _load_claims_lookup,
                _load_nli_model,
                _save,
                verify_review,
            )
            claims_lookup   = _load_claims_lookup()
            articles_lookup = _load_articles_lookup()
            tokenizer, nli_model, label2id = _load_nli_model()
            updated, audit  = verify_review(
                draft, claims_lookup, articles_lookup,
                tokenizer, nli_model, label2id,
            )
            _save(updated, audit, slug)
            draft      = updated
            audit_path = str(REVIEWS_DIR / f"{slug}_audit.json")
        except Exception as e:
            log.warning("NLI skipped : %s", e)

    return GenerateResponse(
        slug                     = slug,
        title                    = draft.title,
        n_sentences              = len(draft.sentences),
        n_bibtex                 = len(bibtex),
        hallucination_rate       = draft.overall_hallucination_rate,
        support_rate             = draft.overall_support_rate,
        overall_hallucination_rate = draft.overall_hallucination_rate,
        review_path              = str(review_path),
        audit_path               = audit_path,
    )


@app.get("/review/{slug}", summary="Lire un ReviewDraft sauvegardé")
def get_review(slug: str):
    # Prefer _verified if it exists
    for suffix in ("_verified", ""):
        path = REVIEWS_DIR / f"{slug}{suffix}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail=f"Review '{slug}' introuvable.")


@app.get(
    "/export/bibtex/{slug}",
    response_class = PlainTextResponse,
    summary        = "Télécharger le BibTeX d'une review",
)
def export_bibtex(slug: str):
    path = REVIEWS_DIR / f"{slug}.bib"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"BibTeX '{slug}' introuvable.")
    return PlainTextResponse(
        content      = path.read_text(encoding="utf-8"),
        media_type   = "text/plain; charset=utf-8",
        headers      = {"Content-Disposition": f'attachment; filename="{slug}.bib"'},
    )


@app.get("/claims", response_model=list[ClaimItem], summary="Lister / filtrer les claims")
def list_claims(
    claim_type: str | None = Query(None, examples={"default": {"value": "finding"}}),
    hedging:    str | None = Query(None, examples={"default": {"value": "certain"}}),
    has_metric: bool | None = Query(None),
    limit:      int         = Query(50, ge=1, le=500),
):
    claims = _get_claims()
    if claim_type:
        claims = [c for c in claims if c.get("claim_type") == claim_type]
    if hedging:
        claims = [c for c in claims if c.get("hedging_level") == hedging]
    if has_metric is True:
        claims = [c for c in claims if c.get("metric_value") is not None]
    elif has_metric is False:
        claims = [c for c in claims if c.get("metric_value") is None]

    return [
        ClaimItem(
            id           = str(c.get("id", "")),
            article_id   = str(c.get("article_id", "")),
            claim_type   = c.get("claim_type", ""),
            hedging      = c.get("hedging_level", ""),
            text         = (c.get("text") or "")[:300],
            metric_name  = c.get("metric_name"),
            metric_value = c.get("metric_value"),
        )
        for c in claims[:limit]
    ]
