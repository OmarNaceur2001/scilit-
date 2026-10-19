"""
SciLit — generation/nli_verify.py
===================================
Étape 8 : Vérification NLI des phrases du ReviewDraft.

Modèle  : cross-encoder/nli-deberta-v3-small (~180MB, CPU)
Logique : premise = claim source → hypothesis = phrase générée
          entailment   → VERIFIED
          neutral      → EXISTS_NO_SUPPORT
          contradiction → CONTRADICTED (the reference may still exist)

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

from tqdm import tqdm

from scilit import config
from scilit.schemas import (
    HallucinationAudit,
    HallucinationAuditEntry,
    ReviewDraft,
    VerificationStatus,
    VerifiedSentence,
)

log = logging.getLogger(__name__)

REVIEWS_DIR   = config.REVIEWS_DIR
CLAIMS_PATH   = config.PARSED_DIR / "claims.jsonl"
ARTICLES_PATH = config.RAW_DIR / "articles.jsonl"

NLI_MODEL        = "cross-encoder/nli-deberta-v3-small"
ENTAIL_THRESHOLD = 0.50
CONTRA_THRESHOLD = 0.40
MAX_PREMISES     = 3    # max claims testés par phrase (vitesse CPU)


# ─────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────

def _load_claims_lookup() -> dict[str, dict]:
    """claim_uuid_str → claim text (champ 'text')."""
    path = config.paths().parsed / "claims.jsonl"
    lookup: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        cid  = c.get("id")
        text = c.get("text")
        if cid and text:
            lookup[str(cid)] = c
    log.info("Claims lookup : %d entrées.", len(lookup))
    return lookup


def _load_articles_lookup() -> dict[str, dict]:
    """article_uuid_str → article dict."""
    path = config.paths().raw / "articles.jsonl"
    return {
        a["id"]: a
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for a in [json.loads(line)]
    }


def _load_chunks_lookup() -> dict[str, list[str]]:
    path = config.paths().parsed / "chunks.jsonl"
    lookup: dict[str, list[str]] = {}
    if not path.exists():
        return lookup
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        chunk = json.loads(line)
        lookup.setdefault(str(chunk.get("article_id")), []).append(chunk.get("text", ""))
    return lookup


def _bibtex_key(article: dict) -> str:
    first_author = article["authors"][0].split()[-1] if article.get("authors") else "Unknown"
    first_word   = re.sub(r"[^A-Za-z]", "", article.get("title", "X").split()[0]) or "X"
    return f"{first_author}{article.get('year', 0)}{first_word}"


def _load_review(slug: str) -> ReviewDraft:
    path = config.paths().reviews / f"{slug}.json"
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
    label2id  = {v.lower(): int(k) for k, v in id2label.items()}
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
        return VerificationStatus.CONTRADICTED, entail_p
    return VerificationStatus.EXISTS_NO_SUPPORT, entail_p


# ─────────────────────────────────────────────────────────────
# Core verification
# ─────────────────────────────────────────────────────────────

def verify_review(
    draft:           ReviewDraft,
    claims_lookup:   dict[str, dict | str],
    articles_lookup: dict[str, dict],
    tokenizer,
    model,
    label2id:        dict[str, int],
    chunks_lookup: dict[str, list[str]] | None = None,
    nli_fn=None,
) -> tuple[ReviewDraft, HallucinationAudit]:
    run_nli = nli_fn or (
        lambda premise, hypothesis: _run_nli(
            tokenizer, model, label2id, premise, hypothesis
        )
    )
    chunks_lookup = chunks_lookup or {}
    updated_sentences: list[VerifiedSentence] = []
    audit_entries:     list[HallucinationAuditEntry] = []

    for sent in tqdm(draft.sentences, desc="NLI verify", unit="sent", ncols=80):
        if not sent.citations and not sent.supporting_claim_ids:
            updated_sentences.append(sent)
            continue

        statuses: list[VerificationStatus] = []
        notes: list[str] = []
        max_entailment: float | None = None
        for aid in sent.citations:
            aid_text = str(aid)
            article = articles_lookup.get(aid_text)
            if article is None:
                status = VerificationStatus.REFERENCE_UNRESOLVED
                score = 0.0
                note = "article_missing_from_corpus"
            else:
                matching_claims: list[dict] = []
                missing_claim = False
                invalid_evidence = False
                for cid in sent.supporting_claim_ids:
                    claim = claims_lookup.get(str(cid))
                    if not claim:
                        missing_claim = True
                        continue
                    if isinstance(claim, str):
                        # Legacy text-only records have no article association and
                        # cannot safely verify which citation the claim belongs to.
                        missing_claim = True
                        continue
                    if str(claim.get("article_id")) == aid_text:
                        evidence_items = [
                            item for item in sent.evidence
                            if str(item.claim_id) == str(cid)
                            and str(item.article_id) == aid_text
                        ]
                        if sent.evidence:
                            source_text = (claim.get("verbatim") or claim.get("text") or "").strip()
                            if not evidence_items or any(
                                item.passage.strip() != source_text for item in evidence_items
                            ):
                                invalid_evidence = True
                                continue
                        matching_claims.append(claim)

                if not sent.supporting_claim_ids:
                    status = VerificationStatus.UNVERIFIABLE
                    score = None
                    note = "no_claim_level_evidence_attached"
                elif invalid_evidence:
                    status = VerificationStatus.UNVERIFIABLE
                    score = None
                    note = "evidence_passage_does_not_match_source_claim"
                elif not matching_claims:
                    if missing_claim:
                        status = VerificationStatus.UNVERIFIABLE
                        note = "supporting_claim_missing_or_unattributed"
                    else:
                        status = VerificationStatus.EXISTS_NO_SUPPORT
                        note = "supporting_claim_belongs_to_another_article"
                    score = None
                else:
                    scores: list[float] = []
                    candidate_statuses: list[VerificationStatus] = []
                    error_notes: list[str] = []
                    for claim in matching_claims[:MAX_PREMISES]:
                        premise = (claim.get("verbatim") or claim.get("text") or "").strip()
                        if not premise:
                            error_notes.append("source_claim_text_empty")
                            continue
                        try:
                            candidate_status, entailment = run_nli(premise, sent.text)
                        except Exception as exc:
                            log.exception("NLI failed for claim %s", claim.get("id"))
                            error_notes.append(f"nli_error:{type(exc).__name__}")
                            continue
                        candidate_statuses.append(candidate_status)
                        scores.append(float(entailment))

                    score = max(scores) if scores else None
                    if VerificationStatus.VERIFIED in candidate_statuses:
                        status = VerificationStatus.VERIFIED
                        note = "automatic_nli_entailment_indicator"
                    elif VerificationStatus.CONTRADICTED in candidate_statuses:
                        status = VerificationStatus.CONTRADICTED
                        note = "automatic_nli_contradiction_indicator"
                    elif VerificationStatus.EXISTS_NO_SUPPORT in candidate_statuses:
                        status = VerificationStatus.EXISTS_NO_SUPPORT
                        note = "automatic_nli_no_support_indicator"
                    else:
                        status = VerificationStatus.UNVERIFIABLE
                        note = ";".join(error_notes) or "no_nli_verdict"
            statuses.append(status)
            notes.append(note)
            if score is not None:
                max_entailment = score if max_entailment is None else max(max_entailment, score)
            audit_entries.append(HallucinationAuditEntry(
                review_id=draft.id,
                sentence_text=sent.text[:200],
                cited_key=_bibtex_key(article) if article else aid_text[:8],
                status=status,
                auditor="auto_nli",
                note=(f"{note} | entailment_probability={score:.4f}" if score is not None else note),
            ))

        updated_sentences.append(VerifiedSentence(
            text=sent.text,
            citations=sent.citations,
            verification_statuses=statuses,
            supporting_claim_ids=sent.supporting_claim_ids,
            nli_score=round(max_entailment, 4) if max_entailment is not None else None,
            evidence=sent.evidence,
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
    reviews_dir = config.paths().reviews
    reviews_dir.mkdir(parents=True, exist_ok=True)
    out_v = reviews_dir / f"{slug}_verified.json"
    out_a = reviews_dir / f"{slug}_audit.json"
    out_v.write_text(draft.model_dump_json(indent=2), encoding="utf-8")
    out_a.write_text(audit.model_dump_json(indent=2), encoding="utf-8")
    log.info("Verified → %s", out_v)
    log.info("Audit    → %s", out_a)


def _print_stats(draft: ReviewDraft, audit: HallucinationAudit) -> None:
    nli_statuses = {
        VerificationStatus.VERIFIED,
        VerificationStatus.EXISTS_NO_SUPPORT,
        VerificationStatus.CONTRADICTED,
    }
    nli_verdicts = sum(1 for entry in audit.entries if entry.status in nli_statuses)
    contradicted = sum(
        1 for entry in audit.entries if entry.status == VerificationStatus.CONTRADICTED
    )
    unsupported = sum(
        1 for entry in audit.entries if entry.status == VerificationStatus.EXISTS_NO_SUPPORT
    )
    unverifiable = sum(
        1 for entry in audit.entries if entry.status == VerificationStatus.UNVERIFIABLE
    )
    unresolved = sum(
        1 for entry in audit.entries
        if entry.status == VerificationStatus.REFERENCE_UNRESOLVED
    )
    legacy_ambiguous = sum(
        1 for entry in audit.entries if entry.status == VerificationStatus.HALLUCINATED
    )
    print(f"\n{'='*65}")
    print(f"  NLI AUDIT : {draft.title}")
    print(f"  Citation checks   : {len(audit.entries)} citations")
    print(f"  NLI verdicts      : {nli_verdicts}/{len(audit.entries)}")
    print(f"  NLI support       : {sum(e.status == VerificationStatus.VERIFIED for e in audit.entries)}/{nli_verdicts}")
    print(f"  Contradicted      : {contradicted}")
    print(f"  No support        : {unsupported}")
    print(f"  Unverifiable      : {unverifiable}")
    print(f"  Ref. not in corpus: {unresolved}")
    print(f"  Legacy ambiguous  : {legacy_ambiguous}")
    print("  NLI values are automatic indicators, not scientific certification.")
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
    config.setup_logging(config.paths().reviews / "nli_log.txt")
    p   = argparse.ArgumentParser(description="SciLit — NLI verification Bronze")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--review", type=str, help="Slug (ex: hallucination-language-model)")
    grp.add_argument("--all",    action="store_true", help="Toutes les reviews")
    args = p.parse_args()

    claims_lookup   = _load_claims_lookup()
    articles_lookup = _load_articles_lookup()
    chunks_lookup = _load_chunks_lookup()
    tokenizer, nli_model, label2id = _load_nli_model()

    reviews_dir = config.paths().reviews
    slugs = (
        [f.stem for f in reviews_dir.glob("*.json")
         if not f.stem.endswith(("_verified", "_audit"))]
        if args.all
        else [args.review]
    )

    for slug in slugs:
        log.info("Vérification : %s", slug)
        draft            = _load_review(slug)
        updated, audit   = verify_review(
            draft, claims_lookup, articles_lookup,
            tokenizer, nli_model, label2id, chunks_lookup,
        )
        _save(updated, audit, slug)
        _print_stats(updated, audit)
        print(audit.summary())

    print("\nProchain : python -m scilit.api.main")


if __name__ == "__main__":
    main()
