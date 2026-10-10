"""
SciLit -- extraction/claims.py
================================
Extraction des Claim depuis les Chunks.

Bronze (Lot B -- regles) :
  - spaCy en_core_web_sm : parsing de dependances, SVO
  - 6 types de claims : FINDING / METHOD / DATASET / LIMITATION /
                        COMPARISON / HYPOTHESIS
  - 3 niveaux de hedging : CERTAIN / HEDGED / SPECULATIVE
  - Extraction de metriques (BLEU, F1, accuracy…) et valeurs numeriques
  - Extraction de datasets (SQuAD, FEVER…) et de noms de modeles

Silver (Lot C -- encodeur) :
  - SciBERT fine-tune sur les claims annotes remplacera les regles
  - Voir extraction/claims_model.py (a creer au palier Silver)

Prerequis :
  pip install spacy
  python -m spacy download en_core_web_sm

Usage :
  python -m scilit.extraction.claims                   # tous les chunks
  python -m scilit.extraction.claims --max 100         # test rapide
  python -m scilit.extraction.claims --stats           # statistiques
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from uuid import UUID

from tqdm import tqdm

from scilit.corpus.parse import CHUNKS_PATH, load_chunks
from scilit.schemas import Chunk, Claim, ClaimType, HedgingLevel, SectionType

# =============================================================================
# CONFIG
# =============================================================================

PARSED_DIR   = Path("data/parsed")
CLAIMS_PATH  = PARSED_DIR / "claims.jsonl"
LOG_PATH     = PARSED_DIR / "claims_log.txt"

SPACY_MODEL  = "en_core_web_sm"
BATCH_SIZE   = 64    # chunks par batch pour nlp.pipe()

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
# LEXIQUES
# =============================================================================

# Verbes de resultats -> FINDING CERTAIN
RESULT_VERBS = {
    "achieve", "outperform", "surpass", "improve", "demonstrate",
    "show", "obtain", "reach", "attain", "yield", "produce",
    "establish", "confirm", "verify", "validate",
}

# Verbes de contribution -> METHOD
CONTRIBUTION_VERBS = {
    "propose", "introduce", "present", "describe", "develop",
    "design", "build", "implement", "create", "construct",
    "extend", "augment", "combine", "integrate",
}

# Verbes de comparaison -> COMPARISON
COMPARISON_VERBS = {
    "outperform", "surpass", "exceed", "beat", "compare",
    "versus", "against", "baseline",
}

# Marqueurs de limitation
LIMITATION_MARKERS = {
    "limit", "limitation", "restrict", "constrain", "fail",
    "cannot", "unable", "struggle", "challenging", "difficult",
    "future work", "remain", "require",
}

# Marqueurs de hedging
HEDGING_VERBS = {
    "suggest", "indicate", "appear", "seem", "tend",
    "imply", "hypothesize", "conjecture", "speculate",
}
MODAL_HEDGED      = {"may", "might", "could", "would", "should"}
MODAL_SPECULATIVE = {"possibly", "potentially", "perhaps", "presumably", "arguably"}
FUTURE_MARKERS    = {"future work", "future research", "remains to", "plan to"}

# Datasets NLP connus (extension possible)
KNOWN_DATASETS = {
    "squad", "squad2", "hotpotqa", "naturalquestions", "triviaqa",
    "msmarco", "fever", "nli", "snli", "mnli", "glue", "superglue",
    "wmt", "conll", "ontonotes", "penn treebank", "ptb",
    "cnn/dailymail", "xsum", "newsqa", "coqa", "quac",
    "boolq", "commonsenseqa", "winogrande", "hellaswag",
    "mmlu", "bbh", "gsm8k", "humaneval", "mbpp",
}

# Modeles NLP connus
KNOWN_MODELS = {
    "bert", "roberta", "deberta", "electra", "albert",
    "gpt", "gpt-2", "gpt-3", "gpt-4", "chatgpt",
    "llama", "llama-2", "llama-3", "mistral", "falcon",
    "t5", "flan-t5", "mt5", "bart", "pegasus",
    "xlm", "xlm-r", "xlmroberta", "camembert", "flaubert",
    "sentence-bert", "sbert", "specter", "scibert", "biobert",
    "lora", "qlora", "peft", "adapter",
}

# Metriques NLP
METRIC_NAMES = {
    "bleu", "rouge", "rouge-1", "rouge-2", "rouge-l",
    "f1", "accuracy", "precision", "recall",
    "em", "exact match", "mrr", "map", "ndcg",
    "perplexity", "bertscore", "meteor", "chrf",
    "comet", "bleurt", "auc", "mse", "rmse", "mae",
}

# =============================================================================
# PATTERNS REGEX
# =============================================================================

# Metrique + valeur : "achieves 92.3 F1" / "F1 score of 0.923" / "92.3% accuracy"
RE_METRIC_VALUE = re.compile(
    r"""
    (?:
        (BLEU|ROUGE[-\w]*|BERTScore|METEOR|chrF|COMET|BLEURT
        |F[-\s]?1|accuracy|precision|recall|EM|exact\s+match
        |MRR|MAP|nDCG|NDCG|perplexity|AUC|MSE|RMSE|MAE)
        [\s\w]{0,15}?              # mots entre metrique et valeur
        (\d+\.?\d*)                # valeur numerique
    |
        (\d+\.?\d*)                # ou valeur d'abord
        \s*(%|percent|points?)     # suivi de %
        (?:\s+(?:on|for|in|at))?
        [\s\w]{0,20}?
        (BLEU|ROUGE[-\w]*|F[-\s]?1|accuracy|precision|recall)?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Chiffres avec % -> potentiel resultat
RE_PERCENT = re.compile(r"\b(\d+\.?\d*)\s*(%|percent)\b", re.I)

# "N parameters" -> taille de modele
RE_PARAMS  = re.compile(r"\b(\d+\.?\d*)\s*(billion|million|B|M)\s+param", re.I)

# =============================================================================
# UTILITAIRES
# =============================================================================

def _load_spacy():
    """Charge spaCy en_core_web_sm avec message d'erreur clair."""
    try:
        import spacy
        return spacy.load(SPACY_MODEL)
    except OSError:
        raise OSError(
            f"Modele spaCy '{SPACY_MODEL}' introuvable.\n"
            f"Lance : python -m spacy download {SPACY_MODEL}"
        )
    except ImportError:
        raise ImportError("pip install spacy")


def _extract_metric(text: str) -> tuple[str | None, float | None]:
    """
    Extrait la premiere paire (nom_metrique, valeur) depuis le texte.
    Retourne (None, None) si aucune metrique trouvee.
    """
    m = RE_METRIC_VALUE.search(text)
    if m:
        groups = [g for g in m.groups() if g]
        # Chercher un nom de metrique et une valeur
        metric_name = None
        metric_val  = None
        for g in groups:
            if g.lower().replace("-", "").replace(" ", "") in {
                n.replace("-", "").replace(" ", "") for n in METRIC_NAMES
            }:
                metric_name = g
            else:
                try:
                    v = float(g)
                    if 0 < v <= 100:
                        metric_val = v
                except ValueError:
                    pass
        if metric_name or metric_val:
            return metric_name, metric_val

    # Fallback : juste un pourcentage
    mp = RE_PERCENT.search(text)
    if mp:
        try:
            return None, float(mp.group(1))
        except ValueError:
            pass
    return None, None


def _extract_dataset(text: str) -> str | None:
    """Detecte un dataset NLP connu dans le texte."""
    t = text.lower()
    for ds in KNOWN_DATASETS:
        # Recherche avec frontiere de mot
        if re.search(r"\b" + re.escape(ds) + r"\b", t):
            return ds
    return None


def _extract_model(text: str) -> str | None:
    """Detecte un nom de modele NLP connu dans le texte."""
    t = text.lower()
    for m in KNOWN_MODELS:
        if re.search(r"\b" + re.escape(m) + r"\b", t):
            # Retrouver la casse originale
            idx = t.find(m)
            return text[idx : idx + len(m)]
    return None


def _detect_hedging(sent_text: str, sent_doc) -> HedgingLevel:
    """
    Detecte le niveau de hedging d'une phrase.
    Utilise les tokens spaCy et les marqueurs lexicaux.
    """
    tl = sent_text.lower()

    # Speculatif : futur, hypothetique
    for marker in FUTURE_MARKERS:
        if marker in tl:
            return HedgingLevel.SPECULATIVE
    for word in MODAL_SPECULATIVE:
        if re.search(r"\b" + word + r"\b", tl):
            return HedgingLevel.SPECULATIVE

    # Hedge : modaux ou verbes de hedging
    for token in sent_doc:
        if token.lower_ in MODAL_HEDGED:
            return HedgingLevel.HEDGED
        if token.lemma_.lower() in HEDGING_VERBS:
            return HedgingLevel.HEDGED

    return HedgingLevel.CERTAIN


def _extract_svo(sent_doc) -> tuple[str | None, str | None, str | None]:
    """
    Extrait le triplet Sujet-Verbe-Objet depuis un document spaCy (phrase).
    Retourne (sujet, verbe_lemme, objet) ou (None, None, None).
    """
    subject = predicate = obj = None

    for token in sent_doc:
        # Verbe racine
        if token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX"):
            predicate = token.lemma_.lower()
            # Chercher sujet (nsubj) et objet (dobj, attr)
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass") and subject is None:
                    # Reconstruire le groupe nominal sujet
                    subject = " ".join(
                        t.text for t in child.subtree
                        if not t.is_punct
                    ).strip()
                if child.dep_ in ("dobj", "attr", "pobj") and obj is None:
                    obj = " ".join(
                        t.text for t in child.subtree
                        if not t.is_punct
                    ).strip()
            break

    return subject, predicate, obj


def _classify_claim(
    sent_text: str,
    sent_doc,
    subject: str | None,
    predicate: str | None,
) -> ClaimType:
    """
    Classe le type de claim en priorite :
      1. Regles explicites (marqueurs lexicaux)
      2. Verbe racine
      3. Fallback FINDING
    """
    tl = sent_text.lower()

    # LIMITATION -- priorite haute
    for marker in LIMITATION_MARKERS:
        if re.search(r"\b" + re.escape(marker) + r"\b", tl):
            return ClaimType.LIMITATION

    # DATASET
    for ds in KNOWN_DATASETS:
        if re.search(r"\b" + re.escape(ds) + r"\b", tl):
            if any(v in tl for v in ("collect", "annotat", "release", "introduc", "curate")):
                return ClaimType.DATASET

    # METHOD
    if subject and re.search(r"\b(we|our|this\s+paper|this\s+work)\b", subject.lower()):
        if predicate and predicate in CONTRIBUTION_VERBS:
            return ClaimType.METHOD
    for v in CONTRIBUTION_VERBS:
        if re.search(r"\b" + v + r"\b", tl):
            if re.search(r"\b(we|our|this\s+paper|this\s+work)\b", tl):
                return ClaimType.METHOD

    # COMPARISON
    for v in COMPARISON_VERBS:
        if re.search(r"\b" + v + r"\b", tl):
            return ClaimType.COMPARISON
    if re.search(r"\b(compared?\s+to|versus|vs\.?|over\s+the\s+baseline)\b", tl):
        return ClaimType.COMPARISON

    # HYPOTHESIS
    for word in MODAL_SPECULATIVE | FUTURE_MARKERS:
        if word in tl:
            return ClaimType.HYPOTHESIS

    # FINDING (defaut si metrique presente)
    metric_n, metric_v = _extract_metric(sent_text)
    if metric_v is not None:
        return ClaimType.FINDING

    if predicate and predicate in RESULT_VERBS:
        return ClaimType.FINDING

    return ClaimType.FINDING   # fallback

# =============================================================================
# EXTRACTION PRINCIPALE
# =============================================================================

def extract_claims_from_chunk(chunk: Chunk, nlp) -> list[Claim]:
    """
    Extrait les Claims depuis un seul Chunk.
    Une phrase = potentiellement un Claim (si elle passe les filtres).
    """
    claims: list[Claim] = []
    doc = nlp(chunk.text)

    for sent in doc.sents:
        text = sent.text.strip()

        # Filtres basiques
        if len(text) < 20 or len(text) > 600:
            continue
        tokens = [t for t in sent if not t.is_punct and not t.is_space]
        if len(tokens) < 5:
            continue

        # SVO
        subject, predicate, obj = _extract_svo(sent)

        # Hedging
        hedging = _detect_hedging(text, sent)

        # Type de claim
        claim_type = _classify_claim(text, sent, subject, predicate)

        # Metrique et dataset
        metric_name, metric_val = _extract_metric(text)
        dataset_name            = _extract_dataset(text)
        model_name              = _extract_model(text)

        # Offset dans le chunk
        char_start = sent.start_char
        char_end   = sent.end_char

        claims.append(Claim(
            article_id=chunk.article_id,
            chunk_id=chunk.id,
            section_type=chunk.section_type,
            text=text,
            verbatim=text,
            claim_type=claim_type,
            hedging_level=hedging,
            subject=subject[:200] if subject else None,
            predicate=predicate,
            **{"object": obj[:200] if obj else None},
            metric_name=metric_name,
            metric_value=metric_val,
            dataset_name=dataset_name,
            model_name=model_name,
            extraction_confidence=None,  # regles = pas de score, Silver ajoutera
        ))

    return claims


def extract_claims(
    chunks: list[Chunk],
    max_chunks: int | None = None,
    output_path: Path = CLAIMS_PATH,
) -> int:
    """
    Pipeline principal : Chunk[] -> Claim[] sauvegardes dans claims.jsonl.
    Append-safe. Utilise nlp.pipe() pour le batch processing spaCy.
    """
    nlp = _load_spacy()
    # Desactiver les composants inutiles (ner et lemmatizer suffisent)
    # Garder : tok2vec, tagger, parser, lemmatizer
    nlp.max_length = 1_000_000

    if max_chunks:
        chunks = chunks[:max_chunks]

    # Charger les chunk_ids deja traites (reprise)
    done_ids: set[str] = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        done_ids.add(str(d.get("chunk_id", "")))
                    except Exception:
                        pass
    if done_ids:
        log.info(f"Reprise : {len(done_ids)} chunks deja traites.")

    to_process = [c for c in chunks if str(c.id) not in done_ids]
    log.info(f"Extraction claims : {len(to_process)} chunks a traiter...")

    total_claims = 0
    mode = "a" if output_path.exists() else "w"

    with open(output_path, mode, encoding="utf-8") as f:
        # Batch processing avec nlp.pipe()
        texts   = [c.text for c in to_process]
        chunk_objs = to_process

        for i, doc in enumerate(tqdm(
            nlp.pipe(texts, batch_size=BATCH_SIZE),
            total=len(texts),
            desc="Claims",
            unit="chunk",
            ncols=80,
        )):
            chunk = chunk_objs[i]
            # Re-extraire les phrases depuis le doc deja parse
            claims = _extract_claims_from_doc(chunk, doc)
            for claim in claims:
                f.write(claim.model_dump_json(exclude={"embedding"}) + "\n")
                total_claims += 1

    log.info(f"Claims extraits : {total_claims} depuis {len(to_process)} chunks.")
    return total_claims


def _extract_claims_from_doc(chunk: Chunk, doc) -> list[Claim]:
    """Version de extract_claims_from_chunk qui utilise un doc deja parse."""
    claims: list[Claim] = []

    for sent in doc.sents:
        text = sent.text.strip()
        if len(text) < 20 or len(text) > 600:
            continue
        tokens = [t for t in sent if not t.is_punct and not t.is_space]
        if len(tokens) < 5:
            continue

        subject, predicate, obj = _extract_svo(sent)
        hedging    = _detect_hedging(text, sent)
        claim_type = _classify_claim(text, sent, subject, predicate)
        metric_name, metric_val = _extract_metric(text)
        dataset_name = _extract_dataset(text)
        model_name   = _extract_model(text)

        claims.append(Claim(
            article_id=chunk.article_id,
            chunk_id=chunk.id,
            section_type=chunk.section_type,
            text=text,
            verbatim=text,
            claim_type=claim_type,
            hedging_level=hedging,
            subject=subject[:200] if subject else None,
            predicate=predicate,
            **{"object": obj[:200] if obj else None},
            metric_name=metric_name,
            metric_value=metric_val,
            dataset_name=dataset_name,
            model_name=model_name,
            extraction_confidence=None,
        ))

    return claims

# =============================================================================
# CHARGEMENT
# =============================================================================

def load_claims(path: Path = CLAIMS_PATH) -> list[Claim]:
    """Charge tous les Claims depuis claims.jsonl."""
    items = []
    if not path.exists():
        return items
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                items.append(Claim.model_validate_json(line))
            except Exception:
                pass
    log.info(f"{len(items)} claims charges.")
    return items

# =============================================================================
# STATISTIQUES
# =============================================================================

def print_stats(claims: list[Claim]) -> None:
    if not claims:
        print("Aucun claim.")
        return

    types:   dict[str, int] = {}
    hedging: dict[str, int] = {}
    with_metric   = sum(1 for c in claims if c.metric_value is not None)
    with_dataset  = sum(1 for c in claims if c.dataset_name)
    with_model    = sum(1 for c in claims if c.model_name)

    for c in claims:
        k = c.claim_type.value
        types[k]          = types.get(k, 0) + 1
        h = c.hedging_level.value
        hedging[h]        = hedging.get(h, 0) + 1

    articles = len({str(c.article_id) for c in claims})

    print(f"\n{'='*55}")
    print(f"  CLAIMS : {len(claims)} depuis {articles} articles")
    print(f"  Types  : {dict(sorted(types.items(), key=lambda x:-x[1]))}")
    print(f"  Hedging: {hedging}")
    print(f"  Avec metrique : {with_metric} ({with_metric*100//len(claims)}%)")
    print(f"  Avec dataset  : {with_dataset}")
    print(f"  Avec modele   : {with_model}")
    print(f"  Claims/article: ~{len(claims)//max(articles,1)}")
    print(f"{'='*55}\n")

    # Top 5 metriques detectees
    metric_names: dict[str, int] = {}
    for c in claims:
        if c.metric_name:
            n = c.metric_name.lower()
            metric_names[n] = metric_names.get(n, 0) + 1
    if metric_names:
        top = sorted(metric_names.items(), key=lambda x: -x[1])[:5]
        print(f"  Top metriques : {top}")

# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="SciLit -- extraction claims (Bronze)")
    p.add_argument("--max",   type=int,  default=None, help="Nb max de chunks")
    p.add_argument("--input", type=Path, default=CHUNKS_PATH)
    p.add_argument("--stats", action="store_true", help="Afficher stats apres extraction")
    args = p.parse_args()

    chunks = load_chunks(args.input)
    if not chunks:
        print("Aucun chunk. Lance : python -m scilit.corpus.parse --mode abstract")
        return

    n = extract_claims(chunks, max_chunks=args.max)
    print(f"\nTermine : {n} claims extraits.")

    if args.stats or True:   # toujours afficher les stats
        claims = load_claims()
        print_stats(claims)


if __name__ == "__main__":
    main()