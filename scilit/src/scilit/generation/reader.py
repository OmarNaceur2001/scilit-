"""
SciLit - generation/reader.py
Part 10a: READ each retrieved paper, with the research question in mind, and write a reading card.

A literature review is written the way a researcher works: read each paper, note what IT shows about
the question, then synthesise. The reader never sees fragments of several papers mixed together.

  1 the paper's OWN sentences   from the parsed document (Part 3):
        - sections that carry the paper's own content (abstract, introduction, method, experiments,
          results, discussion, conclusion, limitations, other); never related work, appendix,
          acknowledgements, ethics, references
        - minus every sentence that CITES another work (in-text citations found by the parser, or
          an "X et al." / "(Name, 2020)" / "[12]" marker): such a sentence reports someone else's
          finding, so it cannot be evidence for THIS paper
        - minus fragments (page/column breaks)
  2 the part relevant to the question   the abstract + the paper's 3 passages that the cross-encoder
        (bge-reranker-v2-m3) scores highest for the question; at most 30 sentences, in reading order;
        each sentence keeps its previous sentence as context ("this model" -> which model)
  3 the reading card   Qwen2.5-7B, temperature 0, JSON schema:
        relevant  does this paper say something that helps answer the question?
        claims    up to 4 short claims about what THIS paper does or finds, each pointing to the
                  1-3 sentence IDs that state it (the IDs are an enum in the schema: an ID that does
                  not exist cannot be generated)
  4 verification   each claim against its own source sentences + context (verify.check); only supported
        claims are passed on to the synthesis

Why not the whole paper: one paper is ~10,000 tokens, the model's window is 8,192, and models use
the middle of long inputs poorly (Liu et al., 2023, "Lost in the Middle", one of our seed papers).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from scilit.corpus.arxiv_meta import surname
from scilit.generation import verify as V
from scilit.generation.llm import MODEL, ollama

DOCUMENTS = Path("data/parsed/documents.jsonl")
OWN_SECTIONS = {"abstract", "introduction", "method", "experiments", "results", "discussion",
                "conclusion", "limitations", "other"}
TOP_PASSAGES, MAX_SENTENCES, MAX_ABSTRACT = 3, 30, 10
CITATION = re.compile(r"\bet al\b|\([^()]*\b(19|20)\d\d[a-z]?\)|\[\d+(?:\s*[,–\-]\s*\d+)*\]")

def card_schema(ids: list[str]) -> dict:
    """The JSON schema of a reading card. Sources are an ENUM of this paper's sentence IDs:
    constrained decoding makes an invented or malformed ID impossible."""
    return {"type": "object", "required": ["relevant", "claims"], "properties": {
        "relevant": {"type": "boolean"},
        "claims": {"type": "array", "maxItems": 4, "items": {"type": "object", "required": ["text", "sources"], "properties": {
            "text": {"type": "string"},
            "sources": {"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "string", "enum": ids}}}}}}}


CARD_PROMPT = """Research question: "{q}"

You are reading ONE paper: "{title}" ({year}).
Below are sentences from this paper, each with its ID.

Decide:
- "relevant": true only if this paper directly studies the topic of the research question and these
  sentences help answer it; false if it is about another topic, even a neighbouring one.
- "claims": if relevant, up to 4 claims that answer the question from THIS paper: what the paper
  does, how it works, or what it finds. Each claim is ONE short sentence (at most 35 words), starts
  with "The authors" or "This paper", copies numbers exactly, and lists in "sources" the IDs of the
  1 to 3 sentences that state it. Use only these sentences. If not relevant, return no claims.

Sentences:
{sentences}"""


# ------------------------------------------------------------------ the parsed papers

class Documents:
    """Random access to data/parsed/documents.jsonl (131 MB): one pass records the byte offset of
    each paper, then a paper is read with one seek, instead of loading the whole file in memory."""

    def __init__(self, path: Path = DOCUMENTS):
        self.path, self.offsets = path, {}
        with path.open("rb") as f:
            pos = 0
            for line in f:
                m = re.match(rb'\{"id": "([^"]+)"', line)
                if m:
                    self.offsets[m.group(1).decode()] = pos
                pos += len(line)

    def get(self, paper_id: str) -> dict:
        with self.path.open("rb") as f:
            f.seek(self.offsets[paper_id])
            return json.loads(f.readline())


@lru_cache(maxsize=1)
def documents() -> Documents:
    return Documents()


def complete(text: str) -> bool:
    return len(text) >= 40 and text.rstrip()[-1:] in ".!?" and text[:1].isupper()


def own_sentences(doc: dict) -> dict[str, tuple[str, str]]:
    """{sentence id: (section type, text)} for the sentences that are this paper's own content."""
    citing = {c["sentence"] for c in doc.get("citations", [])}
    out = {}
    for sec in doc["sections"]:
        if sec["type"] not in OWN_SECTIONS:
            continue
        for s in sec["sentences"]:
            if s["id"] not in citing and not CITATION.search(s["text"]) and complete(s["text"]):
                out[s["id"]] = (sec["type"], s["text"])
    return out


def evidence(query: str, doc: dict, searcher) -> list[tuple[str, str, str]]:
    """The question-relevant part of a paper's own content, in reading order:
    [(sentence id, text, context)], context = the previous sentence when it is also the paper's own
    content (it resolves "this model", "the fine-tuned model"... for the verifier)."""
    own = own_sentences(doc)
    order = {sid: i for i, sid in enumerate(own)}
    previous, last = {}, None
    for sec in doc["sections"]:
        for s in sec["sentences"]:
            previous[s["id"]] = last
            last = s["id"]
    picked = [sid for sid, (t, _) in own.items() if t == "abstract"][:MAX_ABSTRACT]
    candidates = [cid for cid in searcher.chunks_by_paper.get(doc["id"], [])
                  if searcher.chunks[cid]["section_type"] in OWN_SECTIONS - {"abstract"}
                  and any(sid in own for sid in searcher.chunks[cid]["sentence_ids"])]
    for cid in searcher.rerank(query, candidates)[:TOP_PASSAGES] if candidates else []:
        picked += [sid for sid in searcher.chunks[cid]["sentence_ids"] if sid in own]
    picked = sorted(dict.fromkeys(picked), key=order.get)[:MAX_SENTENCES]
    return [(sid, own[sid][1], own[previous[sid]][1] if previous.get(sid) in own else "") for sid in picked]


# ------------------------------------------------------------------ the reading card

def read_paper(query: str, meta: dict, sents: list[tuple[str, str, str]], generator: str, model: str = MODEL) -> dict:
    """-> card {relevant, claims: [{text, sources, status, checks}], sentences: {id: text}, context: {id: text}}
    Each claim is verified against its source sentences, each preceded by its context sentence."""
    if not sents:
        return {**meta, "relevant": False, "claims": [], "sentences": {}, "context": {}, "n_evidence": 0}
    text, ctx = {sid: t for sid, t, _ in sents}, {sid: c for sid, _, c in sents}
    if generator == "template":                      # baseline: the sentence closest to the question, quoted
        q = V.words(query)
        sid, best, _ = max(sents, key=lambda x: len(V.words(x[1]) & q))
        out = {"relevant": bool(V.words(best) & q), "claims": [{"text": best, "sources": [sid]}]}
    else:
        out = ollama(CARD_PROMPT.format(q=query, title=meta["title"], year=meta["year"],
                                        sentences="\n".join(f"[{sid}] {t}" for sid, t, _ in sents)),
                     card_schema(list(text)), model)
    me = {surname(meta["first_author"])}
    claims = []
    for c in out.get("claims", []) if out.get("relevant") else []:
        sources = [s for s in dict.fromkeys(c.get("sources", [])) if s in text]
        premise = [f"{ctx[s]} {text[s]}".strip() for s in sources]
        status, checks = V.check(c["text"], premise or None, me, me)
        claims.append({"text": c["text"].strip(), "sources": sources, "status": status, "checks": checks})
    used = {s for c in claims for s in c["sources"]}
    return {**meta, "relevant": bool(out.get("relevant")), "claims": claims,
            "sentences": {s: text[s] for s in used}, "context": {s: ctx[s] for s in used if ctx[s]},
            "n_evidence": len(sents)}
