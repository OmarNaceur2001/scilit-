"""
SciLit — generation/nli_verify.py
===================================
Étape 8 : Vérification NLI des phrases du ReviewDraft.

Modèle  : cross-encoder/nli-deberta-v3-small (~180MB, CPU)
Logique : premise = claim source → hypothesis = phrase générée
          entailment   → VERIFIED
          neutral      → EXISTS_NO_SUPPORT
          contradiction → HALLUCINATED

Bronze : utilise les supporting_claim_ids du ReviewDraft.
         Si vide → status reste UNVERIFIABLE.
Silver : vérification contre tous les chunks sources.

Sortie :
  data/parsed/reviews/{slug}_verified.json  — ReviewDraft mis à jour
  data/parsed/reviews/{slug}_audit.json     — HallucinationAudit

Usage :
  python -m scilit.generation.nli_verify --review hallucination-language-model
  python -m scilit.generation.nli_verify --all
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from tqdm import tqdm

from scilit.schemas import (
    HallucinationAudit,
    HallucinationAuditEntry,
    ReviewDraft,
    VerificationStatus,
    VerifiedSentence,
)

log = logging.getLogger(__name__)

REVIEWS_DIR   = Path("data/parsed/reviews")
CLAIMS_PATH   = Path("data/parsed/claims.jsonl")
ARTICLES_PATH = Path("data/raw/articles.jsonl")

NLI_MODEL        = "cross-encoder/nli-deberta-v3-small"
ENTAIL_THRESHOLD = 0.50
CONTRA_THRESHOLD = 0.40
MAX_PREMISES     = 3    # max claims testés par phrase (vitesse CPU)


# ─────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────

def _load_claims_lookup() -> dict[str, str]:
    """claim_uuid_str → claim text (champ 'text')."""
    lookup: dict[str, str] = {}
    for line in CLAIMS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        cid  = c.get("id")
        text = c.get("text")
        if cid and text:
            lookup[str(cid)] = text
    log.info("Claims lookup : %d entrées.", len(lookup))
    return lookup


def _load_articles_lookup() -> dict[str, dict]:
    """article_uuid_str → article dict."""
    return {
        a["id"]: a
        for line in ARTICLES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for a in [json.loads(line)]
    }


def _bibtex_key(article: dict) -> str:
    first_author = article["authors"][0].split()[-1] if article.get("authors") else "Unknown"
    first_word   = re.sub(r"[^A-Za-z]", "", article.get("title", "X").split()[0]) or "X"
    return f"{first_author}{article.get('year', 0)}{first_word}"


def _load_review(slug: str) -> ReviewDraft:
    path = REVIEWS_DIR / f"{slug}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Review introuvable : {path}\n"
            "Lance d'abord : python -m scilit.generation.template --query '...'"
        )
    return ReviewDraft.model_validate(json.loads(path.read_text(encoding="utf-8")))


# ─────────────────────────────────────────────────────────────
# NLI model
# ─────────────────────────────────────────────────────────────

def _load_nli_model(model_name: str = NLI_MODEL):
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        raise ImportError("pip install transformers torch")

    log.info("Chargement NLI : %s ...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    id2label  = model.config.id2label          # {0: 'contradiction', 1: 'entailment', ...}
    label2id  = {v.lower(): k for k, v in id2label.items()}
    log.info("Labels NLI : %s", id2label)
    return tokenizer, model, label2id


def _run_nli(
    tokenizer,
    model,
    label2id: dict[str, int],
    premise:    str,
    hypothesis: str,
) -> tuple[VerificationStatus, float]:
    """
    Retourne (VerificationStatus, entailment_proba).
    Robuste aux variantes de nommage des labels.
    """
    import torch
    import torch.nn.functional as F

    inputs = tokenizer(
        premise, hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0].tolist()

    entail_idx = label2id.get("entailment", label2id.get("entail", 1))
    contra_idx = label2id.get("contradiction", label2id.get("contradict", 0))

    entail_p = probs[entail_idx]
    contra_p = probs[contra_idx]

    if entail_p >= ENTAIL_THRESHOLD:
        return VerificationStatus.VERIFIED, entail_p
    if contra_p >= CONTRA_THRESHOLD:
        return VerificationStatus.HALLUCINATED, entail_p
    return VerificationStatus.EXISTS_NO_SUPPORT, entail_p


# ─────────────────────────────────────────────────────────────
# Core verification
# ─────────────────────────────────────────────────────────────

def verify_review(
    draft:           ReviewDraft,
    claims_lookup:   dict[str, str],
    articles_lookup: dict[str, dict],
    tokenizer,
    model,
    label2id:        dict[str, int],
) -> tuple[ReviewDraft, HallucinationAudit]:
    """
    Parcourt chaque VerifiedSentence du draft.
    - supporting_claim_ids présents → NLI (meilleur score parmi ≤3 claims)
    - sinon → UNVERIFIABLE
    Retourne (draft_mis_à_jour, HallucinationAudit).
    """
    updated_sentences: list[VerifiedSentence]     = []
    audit_entries:     list[HallucinationAuditEntry] = []

    for sent in tqdm(draft.sentences, desc="NLI verify", unit="sent", ncols=80):

        # ── Intro/conclusion sans citations → passage direct ────────────────
        if not sent.citations and not sent.supporting_claim_ids:
            updated_sentences.append(sent)
            continue

        # ── Récupérer les textes des claims supports ────────────────────────
        premise_texts = [
            claims_lookup[str(cid)]
            for cid in sent.supporting_claim_ids
            if str(cid) in claims_lookup
        ]

        # ── Pas de premise → UNVERIFIABLE ───────────────────────────────────
        if not premise_texts:
            updated_sentences.append(sent)
            for aid in sent.citations:
                article = articles_lookup.get(str(aid), {})
                audit_entries.append(HallucinationAuditEntry(
                    review_id     = draft.id,
                    sentence_text = sent.text[:200],
                    cited_key     = _bibtex_key(article) if article else str(aid)[:8],
                    status        = VerificationStatus.UNVERIFIABLE,
                    auditor       = "auto_nli",
                    note          = "no_premise_available",
                ))
            continue

        # ── NLI : tester ≤ MAX_PREMISES claims, garder le meilleur ─────────
        best_status: VerificationStatus = VerificationStatus.EXISTS_NO_SUPPORT
        best_score:  float              = 0.0

        for premise in premise_texts[:MAX_PREMISES]:
            status, score = _run_nli(tokenizer, model, label2id, premise, sent.text)
            if score > best_score:
                best_score  = score
                best_status = status
            if best_status == VerificationStatus.VERIFIED:
                break   # optimal trouvé

        # ── Mettre à jour la VerifiedSentence ───────────────────────────────
        updated_sentences.append(VerifiedSentence(
            text                  = sent.text,
            citations             = sent.citations,
            verification_statuses = [best_status] * len(sent.citations),
            supporting_claim_ids  = sent.supporting_claim_ids,
            nli_score             = round(best_score, 4),
        ))

        # ── Audit entries (une par citation) ────────────────────────────────
        for aid in sent.citations:
            article = articles_lookup.get(str(aid), {})
            audit_entries.append(HallucinationAuditEntry(
                review_id     = draft.id,
                sentence_text = sent.text[:200],
                cited_key     = _bibtex_key(article) if article else str(aid)[:8],
                status        = best_status,
                auditor       = "auto_nli",
                note          = f"nli_entail={best_score:.4f} | premises={len(premise_texts)}",
            ))

    # ── Reconstruire le ReviewDraft ─────────────────────────────────────────
    updated_draft = ReviewDraft(
        id             = draft.id,
        query_id       = draft.query_id,
        title          = draft.title,
        sentences      = updated_sentences,
        language       = draft.language,
        generated_at   = draft.generated_at,
        llm_model      = draft.llm_model,
        prompt_version = draft.prompt_version,
    )

    audit = HallucinationAudit(
        review_id  = draft.id,
        entries    = audit_entries,
        audited_at = datetime.now(timezone.utc),
    )

    return updated_draft, audit


# ─────────────────────────────────────────────────────────────
# Save + Print
# ─────────────────────────────────────────────────────────────

def _save(draft: ReviewDraft, audit: HallucinationAudit, slug: str) -> None:
    out_v = REVIEWS_DIR / f"{slug}_verified.json"
    out_a = REVIEWS_DIR / f"{slug}_audit.json"
    out_v.write_text(draft.model_dump_json(indent=2), encoding="utf-8")
    out_a.write_text(audit.model_dump_json(indent=2), encoding="utf-8")
    log.info("Verified → %s", out_v)
    log.info("Audit    → %s", out_a)


def _print_stats(draft: ReviewDraft, audit: HallucinationAudit) -> None:
    print(f"\n{'='*65}")
    print(f"  NLI AUDIT : {draft.title}")
    print(f"  Phrases           : {len(draft.sentences)}")
    print(f"  Hallucination     : {draft.overall_hallucination_rate:.1%}")
    print(f"  Support rate      : {draft.overall_support_rate:.1%}")
    print(f"  Audit entries     : {len(audit.entries)}")
    print(f"  Invented rate     : {audit.invented_rate:.1%}")
    print(f"  Soutenu rate      : {audit.support_rate:.1%}")
    print(f"{'='*65}")
    for sent in draft.sentences:
        if not sent.verification_statuses:
            label = "—"
        else:
            label = sent.verification_statuses[0].value.upper()[:14]
        nli = f"nli={sent.nli_score:.3f}" if sent.nli_score is not None else "       "
        print(f"  [{label:>14}] {nli}  {sent.text[:75]}...")
    print()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    p   = argparse.ArgumentParser(description="SciLit — NLI verification Bronze")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--review", type=str, help="Slug (ex: hallucination-language-model)")
    grp.add_argument("--all",    action="store_true", help="Toutes les reviews")
    args = p.parse_args()

    claims_lookup   = _load_claims_lookup()
    articles_lookup = _load_articles_lookup()
    tokenizer, nli_model, label2id = _load_nli_model()

    slugs = (
        [f.stem for f in REVIEWS_DIR.glob("*.json")
         if not f.stem.endswith(("_verified", "_audit"))]
        if args.all
        else [args.review]
    )

    for slug in slugs:
        log.info("Vérification : %s", slug)
        draft            = _load_review(slug)
        updated, audit   = verify_review(
            draft, claims_lookup, articles_lookup,
            tokenizer, nli_model, label2id,
        )
        _save(updated, audit, slug)
        _print_stats(updated, audit)
        print(audit.summary())

    print("\nProchain : python -m scilit.api.main")


if __name__ == "__main__":
    main()