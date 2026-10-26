"""
SciLit - generation/verify.py
Part 11: is a sentence supported by the exact source sentences it relies on?

Used twice, with the same rules:
  - reading cards : a claim about ONE paper vs. the sentences of that paper it points to
  - synthesis     : a review sentence vs. the source sentences behind the claims it cites
The premise is always ORIGINAL text from the papers, never text written by the LLM (non-circular).

Checks, first failure wins:
  1 existence    every cited ID exists                                     -> not_found
  2 attribution  every author named in the sentence is an author of a cited paper -> misattributed
  3 numbers      every claim-carrying number of the sentence (decimals, %, multipliers such as 3x,
                 integers > 10) appears in the source sentences             -> number_mismatch
                 (a number present but attached to another condition, "4.5% for tables" vs
                 "4.5% overall", is a question of meaning: step 4's job, measured in Part 11)
  4 support      the sources entail the sentence, judged by one of two verifiers (VERIFIER):
                   "deberta"   DeBERTa-v3-base MNLI+FEVER+ANLI (184 M), a sentence-pair NLI classifier:
                               P(entailment) >= 0.5 -> supported ; P(contradiction) >= 0.5 ->
                               contradicted ; otherwise -> insufficient. Window 512 tokens.
                   "minicheck" Bespoke-MiniCheck-7B (Ollama, 4-bit), trained to decide whether a
                               sentence is supported by a grounding document (Tang et al., 2024,
                               LLM-AggreFact): "Yes" -> supported, "No" -> insufficient. 32K window.
                 Both are kept so they can be compared on the same human-checked pairs
                 (evaluation/review_eval.py); choose with SCILIT_VERIFIER or --verifier.
"""
from __future__ import annotations

import os
import re
import unicodedata
from functools import lru_cache

from scilit.corpus.arxiv_meta import surname

VERIFIER = os.environ.get("SCILIT_VERIFIER", "deberta")        # "deberta" | "minicheck"
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
MINICHECK = "bespoke-minicheck"
ENTAIL, CONTRA = 0.5, 0.5
PREMISE_TOKENS = 400            # the NLI window is 512 tokens: room left for the hypothesis
STATUSES = ["supported", "insufficient", "contradicted", "number_mismatch", "misattributed", "not_found"]


@lru_cache(maxsize=1)
def nli():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(dev).eval()
    labels = {v.lower(): k for k, v in model.config.id2label.items()}
    return tok, model, dev, labels["entailment"], labels["contradiction"]


def words(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9\-]+", t.lower()) if len(w) > 3}


def fit(hypothesis: str, sources: list[str], n_tokens) -> str:
    """Source sentences joined; if too long for the NLI window, keep the sentences sharing most
    words with the hypothesis, in their original order."""
    if n_tokens(" ".join(sources)) <= PREMISE_TOKENS:
        return " ".join(sources)
    h = words(hypothesis)
    keep, used = set(), 0
    for i in sorted(range(len(sources)), key=lambda i: -len(words(sources[i]) & h)):
        n = n_tokens(sources[i])
        if used + n <= PREMISE_TOKENS:
            keep.add(i)
            used += n
    return " ".join(sources[i] for i in sorted(keep))


def nli_scores(sources: list[str], hypothesis: str) -> tuple[float, float]:
    import torch
    tok, model, dev, ie, ic = nli()
    premise = fit(hypothesis, sources, lambda t: len(tok.tokenize(t)))
    with torch.no_grad():
        enc = tok(premise, hypothesis, truncation="only_first", max_length=512, return_tensors="pt").to(dev)
        probs = model(**enc).logits.float().softmax(-1)[0]
    return probs[ie].item(), probs[ic].item()


def strip_attribution(text: str) -> str:
    """'Hu et al. report that X' -> 'X'. WHO said it is the attribution check's job; NLI judges WHAT
    is said, against sentences that never name their own authors."""
    t = re.sub(r"^.{0,120}?\b(show|find|found|demonstrat|report|propose|note|highlight|argue|observe|suggest|indicate|conclude|claim)\w*\s+that\s+",
               "", text, flags=re.I)
    t = re.sub(r"(?:[A-Z][\w\-']+\s+)?[A-Z][\w\-']+ et al\.?,?\s*(\((19|20)\d\d\))?", "", t)
    t = re.sub(r"^\s*(the authors|this (paper|work|study))\s+", "", t.strip(), flags=re.I)
    return t[:1].upper() + t[1:] if t else text


NUMBER = re.compile(r"(?<![\w.\-])(\d+(?:\.\d+)?)\s?(%|×|x(?![a-z])|times\b|-?fold\b)?(?![\w])")


def numbers(text: str, every: bool = False) -> set[str]:
    """Numbers that carry a claim: decimals, percentages, multipliers ("3x", "3×", "3 times",
    "3-fold") and integers > 10; not years or plain small counts ("two layers", "3 seeds").
    every=True returns all of them: used on the SOURCES, where a claim's number only has to appear."""
    out = set()
    for m in NUMBER.finditer(text):
        v, unit = m.group(1), m.group(2)
        if not every and (re.fullmatch(r"(19|20)\d\d", v) or ("." not in v and not unit and int(v) <= 10)):
            continue
        out.add(v.rstrip("0").rstrip(".") if "." in v else v)
    return out


def fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def names_in(text: str, known: set[str]) -> set[str]:
    """Authors a sentence names: every "X et al.", plus capitalised, non-initial words that are the
    surname of a paper under review (so "Long contexts..." is not the author Long)."""
    etal = {surname(m) for m in re.findall(r"([A-Z][\w\-']+)\s+et\s+al", text)}
    caps = {fold(w) for w in re.findall(r"(?<!^)\b[A-Z][\w\-']+", text.strip())}
    return {n for n in etal | (caps & known) if n}


def minicheck(sources: list[str], hypothesis: str) -> bool:
    from scilit.generation.llm import ollama_text
    answer = ollama_text(f"Document: {' '.join(sources)}\nClaim: {hypothesis}", MINICHECK)
    return answer.strip().lower().startswith("yes")


def judge(sources: list[str], hypothesis: str, verifier: str | None = None) -> tuple[str, dict]:
    """Do the sources support the sentence? -> (supported | contradicted | insufficient, scores)."""
    verifier = verifier or VERIFIER
    if verifier == "minicheck":
        ok = minicheck(sources, hypothesis)
        return ("supported" if ok else "insufficient"), {"verifier": "minicheck", "supported": ok}
    ent, con = nli_scores(sources, hypothesis)
    status = "supported" if ent >= ENTAIL else "contradicted" if con >= CONTRA else "insufficient"
    return status, {"verifier": "deberta", "entailment": round(ent, 3), "contradiction": round(con, 3)}


def check(text: str, sources: list[str] | None, cited_authors: set[str], known_authors: set[str],
          verifier: str | None = None) -> tuple[str, dict]:
    """The 4 checks. sources=None means a cited ID does not exist."""
    if not sources:
        return "not_found", {}
    unknown = names_in(text, known_authors) - cited_authors
    if unknown:
        return "misattributed", {"names_not_cited": sorted(unknown)}
    missing = numbers(text) - set().union(*(numbers(s, every=True) for s in sources))
    if missing:
        return "number_mismatch", {"numbers_missing": sorted(missing)}
    return judge(sources, strip_attribution(text), verifier)


def count(items: list[dict]) -> dict:
    return {st: sum(1 for s in items if s["status"] == st) for st in STATUSES}
