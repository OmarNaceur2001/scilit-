"""
SciLit — generation/template.py
=================================
Étape 7 : Génération de revue de littérature par gabarit (Bronze).

Bronze : pas de LLM — phrases construites depuis les Claims extraits.
         VerificationStatus = UNVERIFIABLE pour toutes les phrases (NLI = Step 8).
Silver : remplacer _template_sentence() par appel LLM (Mistral-7B-Instruct).

Pipeline :
  query → retrieval RRF (top-k chunks)
        → filtre claims par chunk_id
        → groupe par claim_type
        → gabarits déterministes
        → ReviewDraft + BibTeX

Sortie :
  data/parsed/reviews/{slug}.json   — ReviewDraft sérialisé
  data/parsed/reviews/{slug}.bib    — BibTeX des articles cités

Usage :
  python -m scilit.generation.template --query "LoRA fine-tuning NLP" --k 20
  python -m scilit.generation.template --query "hallucination LLM" --k 15
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from uuid import UUID, uuid4

from scilit import config
from scilit.retrieval.pipeline import search as retrieve
from scilit.schemas import (
    BibTeXEntry,
    Language,
    ReviewDraft,
    VerificationStatus,
    VerifiedSentence,
)

log = logging.getLogger(__name__)

PARSED_DIR    = config.PARSED_DIR
REVIEWS_DIR   = config.REVIEWS_DIR
CLAIMS_PATH   = PARSED_DIR / "claims.jsonl"
ARTICLES_PATH = config.RAW_DIR / "articles.jsonl"

PROMPT_VERSION = "v1.0-bronze"
LLM_MODEL      = "template-bronze-v1"   # Bronze = aucun LLM

# Ordre de présentation dans la revue
SECTION_ORDER = ["method", "dataset", "finding", "comparison", "limitation", "hypothesis"]


# ─────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────

def _load_claims() -> list[dict]:
    path = config.paths().parsed / "claims.jsonl"
    if not path.exists():
        raise FileNotFoundError("Lancer d'abord : python -m scilit.extraction.claims")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_articles() -> dict[str, dict]:
    """uuid_str → article dict."""
    path = config.paths().raw / "articles.jsonl"
    return {
        a["id"]: a
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for a in [json.loads(line)]
    }


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _bibtex_key(article: dict) -> str:
    authors = article.get("authors") or []
    first_author = authors[0] if authors else "Unknown"
    last_name = first_author.split(",", 1)[0] if "," in first_author else first_author.split()[-1]
    title_words = re.findall(r"[A-Za-zÀ-ÿ0-9]+", article.get("title", ""))
    first_word = next(
        (word for word in title_words if word.lower() not in {"a", "an", "the", "of", "for", "on", "in"}),
        "X",
    )
    def fold(value: str) -> str:
        return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    author_key = re.sub(r"[^A-Za-z0-9]", "", fold(last_name)) or "Unknown"
    title_key = re.sub(r"[^A-Za-z0-9]", "", fold(first_word)) or "X"
    return f"{author_key}{article.get('year', 0)}{title_key}"


def _make_bibtex(article: dict, bibtex_key: str | None = None) -> BibTeXEntry:
    source = article.get("source", {})
    venue_name = article.get("venue_name") or source.get("venue_name") or source.get("venue", "Other")
    arxiv_id = source.get("arxiv_id")
    acl_id = source.get("acl_id")
    if arxiv_id:
        entry_type = "misc"
    elif acl_id:
        entry_type = "inproceedings"
    else:
        entry_type = "article"
    return BibTeXEntry(
        article_id = UUID(article["id"]),
        bibtex_key = bibtex_key or _bibtex_key(article),
        entry_type = entry_type,
        title      = article["title"],
        authors    = article["authors"],
        year       = article["year"],
        venue      = venue_name,
        url        = source.get("url"),
        doi        = article.get("doi") or source.get("doi"),
        booktitle  = venue_name if entry_type == "inproceedings" else None,
        journal    = venue_name if entry_type == "article" else None,
        eprint     = arxiv_id,
        archive_prefix = "arXiv" if arxiv_id else None,
        howpublished = f"\\url{{https://arxiv.org/abs/{arxiv_id}}}" if arxiv_id else None,
    )


# ─────────────────────────────────────────────────────────────
# Template sentences (Bronze — déterministe, pas de LLM)
# ─────────────────────────────────────────────────────────────

def _clean_fragment(text: str, fallback: str) -> str:
    """Rejette les fragments SVO trop courts (artefacts du parsing)."""
    return text if (text and len(text.split()) >= 2) else fallback


def _template_sentence(claim: dict) -> str:
    """
    Bronze : utilise le texte brut du claim en priorité.
    Si une métrique est présente + sujet SVO valide, phrase structurée.
    Silver : remplacer par prompt LLM.
    """
    ctype    = claim.get("claim_type", "finding")
    metric_n = claim.get("metric_name")
    metric_v = claim.get("metric_value")
    raw_text = (claim.get("text") or "").strip()

    # Fragments SVO — valides seulement si ≥ 2 mots
    subj = (claim.get("subject") or "").strip()
    pred = (claim.get("predicate") or "").strip()
    subj = subj if len(subj.split()) >= 2 else ""
    pred = pred if len(pred.split()) >= 2 else ""

    # Phrase structurée si métrique chiffrée + sujet identifié
    if metric_n and metric_v is not None and subj:
        return _cap(f"{subj} achieves {metric_v:.1f} {metric_n}.")

    # Texte brut du claim (source principale Bronze)
    if raw_text:
        text = raw_text if raw_text.endswith((".", "!", "?")) else raw_text + "."
        if ctype == "limitation":
            return _cap(f"A noted limitation: {text}")
        return _cap(text)

    # Dernier recours : gabarits SVO avec fallbacks
    subj = subj or "The proposed approach"
    pred = pred or "achieves competitive results"
    if ctype == "method":
        return _cap(f"{subj} proposes to {pred}.")
    if ctype == "comparison":
        return _cap(f"{subj} outperforms baselines in {pred}.")
    if ctype == "limitation":
        return _cap(f"However, {subj} is limited by {pred}.")
    if ctype == "hypothesis":
        return _cap(f"Future work may explore {pred}.")
    if ctype == "dataset":
        return _cap(f"{subj} introduces a dataset for {pred}.")
    return _cap(f"{subj} demonstrates that {pred}.")


# ─────────────────────────────────────────────────────────────
# VerifiedSentence factory
# ─────────────────────────────────────────────────────────────

def _make_vs(
    text:                 str,
    article_ids:          list[str],
    supporting_claim_ids: list[UUID] | None = None,
) -> VerifiedSentence:
    """
    Bronze : toutes les citations restent UNVERIFIABLE.
    citations et verification_statuses ont obligatoirement la même longueur
    (validé par le model_validator du schema).
    """
    uuids    = [UUID(aid) for aid in article_ids if aid]
    statuses = [VerificationStatus.UNVERIFIABLE] * len(uuids)
    return VerifiedSentence(
        text                  = text,
        citations             = uuids,
        verification_statuses = statuses,
        supporting_claim_ids  = supporting_claim_ids or [],
        nli_score             = None,
    )


# ─────────────────────────────────────────────────────────────
# Main generation
# ─────────────────────────────────────────────────────────────

def generate(
    query:                  str,
    k:                      int = 20,
    max_sentences_per_type: int = 3,
) -> tuple[ReviewDraft, list[BibTeXEntry]]:
    """
    Génère un ReviewDraft Bronze pour la requête donnée.

    Retourne (draft, bibtex_entries) pour que le CLI puisse
    sauvegarder et afficher séparément.
    """
    articles_db = _load_articles()
    all_claims  = _load_claims()
    query_id    = uuid4()

    # ── 1. Retrieval ─────────────────────────────────────────────────────────
    log.info("Retrieval RRF : '%s' (k=%d)...", query, k)
    ranked = retrieve(query, k=k, mode="rrf")
    if not ranked:
        raise ValueError(f"Aucun résultat retrieval pour : '{query}'")

    retrieved_chunk_ids   = {str(r.chunk_id)   for r in ranked}
    retrieved_article_ids = {str(r.article_id) for r in ranked}
    log.info(
        "  %d chunks depuis %d articles.",
        len(retrieved_chunk_ids), len(retrieved_article_ids),
    )

    # ── 2. Filtrer et grouper les claims ─────────────────────────────────────
    relevant = [
        c for c in all_claims
        if c.get("chunk_id") in retrieved_chunk_ids
    ]
    log.info("  %d claims pertinents.", len(relevant))

    by_type: dict[str, list[dict]] = {t: [] for t in SECTION_ORDER}
    for c in relevant:
        ctype = c.get("claim_type", "finding")
        if ctype in by_type:
            by_type[ctype].append(c)

    # ── 3. Phrases d'introduction ────────────────────────────────────────────
    sentences: list[VerifiedSentence] = [
        _make_vs(
            f"This survey covers recent work on {query}, "
            f"drawing from {len(retrieved_article_ids)} papers.",
            [],
        )
    ]

    # ── 4. Phrases par section ───────────────────────────────────────────────
    for ctype in SECTION_ORDER:
        pool = by_type[ctype]
        if not pool:
            continue

        # Priorité aux findings avec métriques chiffrées
        if ctype == "finding":
            pool = sorted(
                pool,
                key=lambda c: (
                    c.get("metric_value") is not None,
                    c.get("metric_value") or 0.0,
                ),
                reverse=True,
            )

        for claim in pool[:max_sentences_per_type]:
            article_id = claim.get("article_id")
            support    = [UUID(claim["id"])] if claim.get("id") else []
            sentences.append(
                _make_vs(
                    _template_sentence(claim),
                    [article_id] if article_id else [],
                    support,
                )
            )

    # ── 4b. Dédupliquer (artefacts de chunks overlap) ─────────────────────
    # Filtre 1 : commence par minuscule = fragment de chunk
    # Filtre 2 : contenu (à partir du 2e mot) déjà présent dans une phrase
    #            retenue → attrape "Training, the 7B..." ⊂ "Using...training,
    #            the 7B..." et "Itions, RAG..." ⊂ "...conditions, RAG..."
    def _norm(t: str) -> str:
        return " ".join(t.lower().split())

    clean_sentences: list[VerifiedSentence] = []
    added_norms: list[str] = []

    for sent in sentences:
        if not sent.verification_statuses:          # intro / conclusion
            clean_sentences.append(sent)
            continue
        t     = sent.text.strip()
        words = t.split()
        if not words:
            continue

        # Filtre 1
        if words[0][0].islower():
            continue

        # Filtre 2 — compare le corps (sans le 1er mot) à tous les textes déjà retenus
        t_norm      = _norm(t)
        after_first = _norm(" ".join(words[1:])) if len(words) > 1 else t_norm
        if any(after_first in prev or t_norm in prev for prev in added_norms):
            continue

        clean_sentences.append(sent)
        added_norms.append(t_norm)

    sentences = clean_sentences

    # ── 5. Conclusion ────────────────────────────────────────────────────────
    n_method  = len(by_type["method"])
    n_finding = len(by_type["finding"])
    sentences.append(
        _make_vs(
            f"In summary, {n_method} methodological contributions and "
            f"{n_finding} empirical findings were identified "
            f"in the literature on {query}.",
            [],
        )
    )

    # ── 6. ReviewDraft ────────────────────────────────────────────────────────
    draft = ReviewDraft(
        query_id       = query_id,
        title          = f"Literature Review: {query.title()}",
        sentences      = sentences,
        language       = Language.EN,
        llm_model      = LLM_MODEL,
        prompt_version = PROMPT_VERSION,
    )

    # ── 7. BibTeX ─────────────────────────────────────────────────────────────
    used_keys: set[str] = set()
    bibtex_entries = []
    for aid in sorted(retrieved_article_ids):
        if aid not in articles_db:
            continue
        base_key = _bibtex_key(articles_db[aid])
        key = base_key
        suffix = ord("a")
        while key in used_keys:
            key = f"{base_key}{chr(suffix)}"
            suffix += 1
        used_keys.add(key)
        bibtex_entries.append(_make_bibtex(articles_db[aid], key))

    return draft, bibtex_entries


# ─────────────────────────────────────────────────────────────
# Save + Print
# ─────────────────────────────────────────────────────────────

def _save(draft: ReviewDraft, bibtex_entries: list[BibTeXEntry], slug: str) -> None:
    reviews_dir = config.paths().reviews
    reviews_dir.mkdir(parents=True, exist_ok=True)
    out_json = reviews_dir / f"{slug}.json"
    out_bib  = reviews_dir / f"{slug}.bib"
    out_json.write_text(draft.model_dump_json(indent=2), encoding="utf-8")
    out_bib.write_text(
        "\n\n".join(e.to_bibtex() for e in bibtex_entries),
        encoding="utf-8",
    )
    log.info("ReviewDraft → %s", out_json)
    log.info("BibTeX      → %s  (%d entrées)", out_bib, len(bibtex_entries))


def _print_review(draft: ReviewDraft, bibtex_entries: list[BibTeXEntry]) -> None:
    print(f"\n{'='*65}")
    print(f"  REVIEW : {draft.title}")
    print(f"  Modèle : {draft.llm_model}  |  Prompt : {draft.prompt_version}")
    print(f"  Phrases          : {len(draft.sentences)}")
    print(f"  Hallucination    : {draft.overall_hallucination_rate:.1%}  (Bronze=UNVERIFIABLE)")
    print(f"  Support rate     : {draft.overall_support_rate:.1%}")
    print(f"  BibTeX entrées   : {len(bibtex_entries)}")
    print(f"{'='*65}")
    for i, sent in enumerate(draft.sentences, 1):
        st = sent.verification_statuses[0].value if sent.verification_statuses else "—"
        print(f"  [{i:2d}] [{st:>10}]  {sent.text}")
    print()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main() -> None:
    config.setup_logging(config.paths().reviews / "template_log.txt")
    p = argparse.ArgumentParser(description="SciLit — template generation Bronze")
    p.add_argument("--query", required=True, help="Sujet de la revue")
    p.add_argument("--k",     type=int, default=20, help="Top-k chunks retrieval (défaut: 20)")
    p.add_argument("--max",   type=int, default=3,  help="Phrases max par type de claim (défaut: 3)")
    args = p.parse_args()

    draft, bibtex = generate(query=args.query, k=args.k, max_sentences_per_type=args.max)
    slug = _slugify(args.query)
    _save(draft, bibtex, slug)
    _print_review(draft, bibtex)
    print("Prochain : python -m scilit.generation.nli_verify")


if __name__ == "__main__":
    main()