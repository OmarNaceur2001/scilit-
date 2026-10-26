"""
SciLit - generation/review.py
Parts 10-11: answer a research question the way a literature review is written.

  1 find        the 8 most relevant papers (hybrid search + rerank, paper level: the ranking
                evaluated in Part 12)
  2 read        each paper on its own -> a reading card of verified claims (reader.py); the 8
                LLM drafts are sent as concurrent requests, then verified one by one
  3 abstain     if NO retrieved paper has a supported claim, the corpus does not cover the
                question: SciLit says so and lists the closest papers, instead of writing.
                With a single source it still answers, flagged "thin evidence"
  4 plan        Qwen2.5-7B only DECIDES which verified claims go together (JSON schema, claim IDs
                as an enum, temperature 0): a list of points, each = 1-3 claim IDs + its section
                ("answer" for the 1-2 points that answer the question directly, else a short aspect
                title). Choosing from a list is a selection task a 7B model does reliably; the
                citations of every sentence are fixed HERE, before any text exists
  5 write       one call per point: the model sees ONLY that point's claims (not even the question)
                and restates them in ONE sentence. It cannot cite the wrong claim (it sees no other)
                and cannot add the question's model, task or setting to it (it does not see them).
                The model never sees or writes author names: WHO said what comes from the citations
  6 verify      every sentence against the ORIGINAL source sentences behind its own claims
                (verify.check: existence, attribution, numbers, NLI)
  7 keep        only supported sentences form the review; everything else is shown with its reason

Not generated on purpose: "X finds A, while Y finds B" contrasts. No verifier can check the contrast
itself, so findings on the same aspect are placed side by side under one section title instead.
"gaps" is the model's statement of what the claims do not address: an absence, shown as such.
There is no repair step: with citations fixed before writing, a rejected sentence has nothing true
left to rescue (measured: 4 of 74 repairs accepted in the previous design).

Synthesis designs (--synthesis), compared on the SAME reading cards by review_eval replay:
  plan      steps 4-5 above (default)
  single    the previous design: one call writes the whole review and chooses its own citations
            (claim IDs as an enum), no repair. Kept as the baseline the plan design is measured against
  template  one "Surname et al. (year) report that ..." per claim. Cannot invent anything
Reading baseline (--generator template): the sentence of each paper closest to the question, quoted.

Every card keeps its evidence sentences and the LLM draft, so an experiment can re-verify the claims
(reverify) or re-synthesise (write_review) without reading the papers again.

Output: data/reviews/<slug>.json (everything) and <slug>.md (readable)
Usage : python -m scilit.generation.review "How does retrieval-augmented generation reduce hallucination?"
        python -m scilit.generation.review "..." --generator template
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scilit.corpus.arxiv_meta import surname
from scilit.generation import verify as V
from scilit.generation.llm import MODEL, ollama

from scilit.generation.reader import documents, draft_card, evidence, verify_card
from scilit.generation.table import articles, build_table, to_markdown

K_PAPERS, MAX_WORDS, READ_WORKERS = 8, 40, 4

def plan_schema(claim_ids: list[str]) -> dict:
    """Plan = which claims make one point together. "claims" is an ENUM of the existing claim IDs."""
    point = {"type": "object", "required": ["section", "claims"], "properties": {"section": {"type": "string"},
             "claims": {"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "string", "enum": claim_ids}}}}
    return {"type": "object", "required": ["points", "gaps"], "properties": {
        "points": {"type": "array", "minItems": 1, "maxItems": 8, "items": point}, "gaps": {"type": "string"}}}


PLAN_PROMPT = """Research question: "{q}"

Below are verified claims from {n} papers, each with an ID. Plan the answer of a literature review:
decide which claims go together. Do not write the review.

"points": each point lists the IDs of 1 to 3 claims that together make ONE point (the same finding,
or parts of one method or result), and the "section" it belongs to:
- the first 1 or 2 points have section "answer": the claims that most directly answer the question;
- every other point has a short section title (2 to 5 words) naming one aspect of the question;
  points about the same aspect share the same title.
Use each claim at most once. Leave out claims that do not help answer the question. Make only as
many points as the claims support.

"gaps": one sentence on what the question asks that none of these claims address, or "" if nothing.

Claims:
{claims}"""

WRITE_SCHEMA = {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}}
WRITE_PROMPT = """Restate the claims below in ONE sentence of at most 35 words, combining them if there
are several. Say only what they say: add no cause, explanation, comparison, generalisation, model,
task, dataset or setting that they do not state. Copy numbers exactly. Do not name authors, papers,
IDs or years.

Claims:
{claims}"""

SINGLE_PROMPT = """Research question: "{q}"

Below are verified claims from {n} papers, each with an ID. Write the answer of a literature review
using ONLY these claims.

- "answer": 1 or 2 sentences that directly answer the question by combining the claims.
- "aspects": 2 to 4 groups, each with a short title and 1 to 3 sentences, in this order: first HOW or
  WHY it works (mechanisms, methods), then what is found, then disagreements or limits. Compare
  papers when they agree or differ.
- "gaps": one sentence on what these claims do not show about the question.

Rules:
- Each sentence lists in "cites" the IDs of the 1 to 3 claims it combines, and says nothing they do not say.
- Each sentence has at most 35 words. Copy numbers exactly.
- Never name authors or papers in the text, and never write IDs, years or "as shown in": the
  citations in "cites" show who said what. To contrast papers, write for example "some work finds X,
  while other work finds Y" and cite each claim.

Claims:
{claims}"""


def single_schema(claim_ids: list[str]) -> dict:
    sent = {"type": "object", "required": ["text", "cites"], "properties": {"text": {"type": "string"},
            "cites": {"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "string", "enum": claim_ids}}}}
    return {"type": "object", "required": ["answer", "aspects", "gaps"], "properties": {
        "answer": {"type": "array", "minItems": 1, "maxItems": 2, "items": sent},
        "aspects": {"type": "array", "minItems": 2, "maxItems": 4, "items": {"type": "object", "required": ["title", "sentences"],
                    "properties": {"title": {"type": "string"}, "sentences": {"type": "array", "minItems": 1, "maxItems": 3, "items": sent}}}},
        "gaps": {"type": "string", "minLength": 20}}}


def meta(pid: str) -> dict:
    a = articles()[pid]
    return {"paper_id": pid, "title": a["title"], "year": a.get("arxiv_year") or a["year"],
            "first_author": (a["authors"] or ["?"])[0]}


def clean(text: str, cites: list[str]) -> dict:
    """Citations come from the plan; remove any ID, "(Ji et al., 2022)" or "as shown in" left in the text."""
    text = re.sub(r"\s*\[(C\d+(,\s*)?)+\]", "", text)
    text = re.sub(r"\s*\([^()]*(et al|\bC\d+\b|(19|20)\d\d)[^()]*\)", "", text).strip()
    text = re.sub(r",?\s*(as )?(shown|demonstrated|reported|noted|seen|found)\s+(in|by)\s*([.;])$", r"\4", text)
    return {"text": text, "cites": cites}


def claim_index(cards: list[dict]) -> dict:
    """Supported claims, numbered C1, C2... across papers, each with its paper and source sentences."""
    out, n = {}, 0
    for card in cards:
        for c in card["claims"]:
            if c["status"] == "supported":
                n += 1
                c["id"] = f"C{n}"
                out[c["id"]] = {**c, "paper_id": card["paper_id"], "author": surname(card["first_author"]),
                                "label": card["first_author"].split()[-1] if card["first_author"].split() else "?",
                                "year": card["year"], "title": card["title"],
                                "quotes": [card["sentences"][s] for s in c["sources"]],
                                "source_text": [f"{card['context'].get(s, '')} {card['sentences'][s]}".strip()
                                                for s in c["sources"]]}
    return out


def claim_blocks(claims: dict, cards: list[dict]) -> list[str]:
    """The claims grouped by paper ("Paper n: title (year)"), never with author names."""
    blocks = []
    for card in cards:
        mine = [c for c in claims.values() if c["paper_id"] == card["paper_id"]]
        if mine:
            blocks.append(f"Paper {len(blocks) + 1}: \"{card['title']}\" ({card['year']}):\n" +
                          "\n".join(f"  [{c['id']}] {c['text']}" for c in mine))
    return blocks


def plan(query: str, claims: dict, cards: list[dict], model: str) -> tuple[list[dict], str]:
    """Step 4: group claim IDs into points. Code enforces what the schema cannot: each claim used once,
    at most 2 "answer" points, and the first point answers the question."""
    blocks = claim_blocks(claims, cards)
    out = ollama(PLAN_PROMPT.format(q=query, n=len(blocks), claims="\n\n".join(blocks)), plan_schema(list(claims)), model)
    points, used, answers = [], set(), 0
    for p in out.get("points", []):
        ids = [c for c in dict.fromkeys(p.get("claims", [])) if c in claims and c not in used]
        if not ids:
            continue
        used |= set(ids)
        section = (p.get("section") or "").strip() or "Findings"
        if section.lower() == "answer":
            answers += 1
            section = "answer" if answers <= 2 else "Findings"
        points.append({"part": section, "cites": ids})
    if points and not answers:
        points[0]["part"] = "answer"
    points.sort(key=lambda p: p["part"] != "answer")          # stable: the answer first, sections in model order
    return points, out.get("gaps", "").strip()


def write(query: str, point: dict, claims: dict, model: str) -> dict:
    """Step 5: one sentence from ONE point's claims only; its citations are the point's claims."""
    text = "\n".join(f"- {claims[c]['text']}" for c in point["cites"])
    out = ollama(WRITE_PROMPT.format(claims=text), WRITE_SCHEMA, model)
    return {**clean(out.get("text", ""), point["cites"]), "part": point["part"]}


def synthesise(query: str, claims: dict, cards: list[dict], model: str) -> tuple[list[dict], str, list[dict]]:
    points, gaps = plan(query, claims, cards, model)
    with ThreadPoolExecutor(READ_WORKERS) as pool:
        sents = list(pool.map(lambda p: write(query, p, claims, model), points))
    return [s for s in sents if s["text"]], gaps, points


def synthesise_single(query: str, claims: dict, cards: list[dict], model: str) -> tuple[list[dict], str]:
    """The previous design, as a baseline: one call writes everything and picks its own citations."""
    blocks = claim_blocks(claims, cards)
    out = ollama(SINGLE_PROMPT.format(q=query, n=len(blocks), claims="\n\n".join(blocks)), single_schema(list(claims)), model)
    sents = [{**clean(s["text"], s.get("cites", [])), "part": "answer"} for s in out.get("answer", [])]
    for a in out.get("aspects", []):
        sents += [{**clean(s["text"], s.get("cites", [])), "part": a.get("title", "").strip() or "Findings"}
                  for s in a.get("sentences", [])]
    return [s for s in sents if s["text"]], out.get("gaps", "").strip()


def template_synthesis(claims: dict) -> tuple[list[dict], str]:
    return [{"text": f"{c['label']} et al. ({c['year']}) report that {c['text'][0].lower() + c['text'][1:]}",
             "cites": [cid], "part": "Extracts"} for cid, c in claims.items()], ""


def check_sentence(s: dict, claims: dict, known: set[str]) -> dict:
    cited = [claims[c] for c in s["cites"] if c in claims]
    ok = cited and len(cited) == len(s["cites"])
    sources = [t for c in cited for t in c["source_text"]] if ok else None
    if len(s["text"].split()) > MAX_WORDS * 1.5:
        return {**s, "status": "insufficient", "checks": {"too_long": len(s["text"].split())}}
    status, checks = V.check(s["text"], sources, {c["author"] for c in cited}, known)
    return {**s, "status": status, "checks": checks}


SYNTHESES = ["plan", "single", "template"]


def read(query: str, searcher, generator: str = "llm", model: str = MODEL) -> tuple[list[dict], list[dict], dict]:
    """Steps 1-2 -> (papers, cards, latency). Each card keeps its evidence and the draft."""
    t = [time.time()]
    papers = searcher.search(query, k=K_PAPERS)                     # one passage per paper = paper ranking
    docs = documents()
    t.append(time.time())
    metas = [meta(p["paper_id"]) for p in papers]
    sents = [evidence(query, docs.get(m["paper_id"]), searcher) for m in metas]          # reranker: sequential
    with ThreadPoolExecutor(READ_WORKERS) as pool:                                       # LLM drafts: concurrent
        drafts = list(pool.map(lambda i: draft_card(query, metas[i], sents[i], generator, model), range(len(metas))))
    cards = [{**verify_card(m, e, d), "evidence": e, "draft": d} for m, e, d in zip(metas, sents, drafts)]
    t.append(time.time())
    return papers, cards, {"search": round(t[1] - t[0], 1), "reading": round(t[2] - t[1], 1)}


def reverify(card: dict) -> dict:
    """The same evidence and draft, verified again with the current verifier (no LLM call)."""
    m = {k: card[k] for k in ("paper_id", "title", "year", "first_author")}
    ev = [tuple(x) for x in card["evidence"]]
    return {**verify_card(m, ev, card["draft"]), "evidence": card["evidence"], "draft": card["draft"]}


def write_review(query: str, table: list[dict], cards: list[dict], synthesis: str = "plan", model: str = MODEL) -> dict:
    """Steps 3-7 from reading cards: abstain, synthesise (one of SYNTHESES), verify every sentence."""
    t = time.time()
    claims = claim_index(cards)
    known = {surname(c["first_author"]) for c in cards} - {""}
    covered = {c["paper_id"] for c in claims.values()}
    result = {"query": query, "synthesis": synthesis, "model": model, "verifier": V.VERIFIER, "cards": cards,
              "claims": claims, "claim_counts": V.count([c for card in cards for c in card["claims"]]),
              "table": table, "abstained": not covered, "thin_evidence": len(covered) == 1}
    points: list[dict] = []
    if result["abstained"]:
        draft, gaps = [], (f"The corpus does not cover this question: {len(covered)} of {len(cards)} retrieved "
                           f"papers contain a verified claim about it.")
    elif synthesis == "plan":
        draft, gaps, points = synthesise(query, claims, cards, model)
    elif synthesis == "single":
        draft, gaps = synthesise_single(query, claims, cards, model)
    else:
        draft, gaps = template_synthesis(claims)
    sentences = [check_sentence(s, claims, known) for s in draft]
    result.update(plan=points, sentences=sentences, gaps=gaps, counts=V.count(sentences),
                  latency_s={"synthesis": round(time.time() - t, 1)})
    return result


def run_review(query: str, searcher, generator: str = "llm", model: str = MODEL, synthesis: str = "plan") -> dict:
    """The whole review, used by the command line AND the API."""
    papers, cards, lat = read(query, searcher, generator, model)
    r = write_review(query, build_table(papers), cards, "template" if generator == "template" else synthesis, model)
    r["generator"] = generator
    r["latency_s"] = {**lat, **r["latency_s"]}
    return r


def to_md(r: dict) -> str:
    kept = [s for s in r["sentences"] if s["status"] == "supported"]
    md = [f"# {r['query']}", ""]
    if r["abstained"]:
        md += [f"**{r['gaps']}**", ""]
    elif r.get("thin_evidence"):
        md += ["*Thin evidence: a single paper of the corpus addresses this question.*", ""]
    for part in dict.fromkeys(s["part"] for s in kept):
        md += [f"**{'Answer' if part == 'answer' else part}.** " +
               " ".join(f"{s['text']} [{', '.join(s['cites'])}]" for s in kept if s["part"] == part), ""]
    if r["gaps"] and not r["abstained"]:
        md += [f"*Not covered by the evidence:* {r['gaps']}", ""]
    md += ["## Removed by verification", ""] + [f"- **{s['status']}**: {s['text']} [{', '.join(s['cites'])}]"
                                                  for s in r["sentences"] if s["status"] != "supported"]
    md += ["", f"Claims (reading cards): {r['claim_counts']}", f"Synthesis: {r['counts']}", "", "## Reading cards", ""]
    for card in r["cards"]:
        md.append(f"### {card['first_author']} ({card['year']}), {card['title']} - {card['relation']}")
        md += [f"- [{c.get('id', '-')}] **{c['status']}**: {c['text']}  \n  sources: " +
               " | ".join(card["sentences"].get(s, s) for s in c["sources"]) for c in card["claims"]] + [""]
    md += ["## Comparison table", "", to_markdown(r["table"])]
    return "\n".join(md)


def main() -> None:
    from scilit.retrieval.search import Searcher
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--generator", choices=["llm", "template"], default="llm")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--synthesis", choices=SYNTHESES, default="plan")
    ap.add_argument("--verifier", choices=["deberta", "minicheck"], default=V.VERIFIER)
    args = ap.parse_args()
    V.VERIFIER = args.verifier
    r = run_review(args.query, Searcher(), args.generator, args.model, args.synthesis)
    slug = re.sub(r"[^a-z0-9]+", "-", V.fold(args.query)).strip("-")[:60] + f"_{r['synthesis']}"
    out = Path("data/reviews"); out.mkdir(parents=True, exist_ok=True)
    (out / f"{slug}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / f"{slug}.md").write_text(to_md(r), encoding="utf-8")
    print(f"claims (reading cards): {r['claim_counts']}")
    print(f"synthesis: {r['counts']} from {len(r['plan'])} planned points")
    print(f"abstained: {r['abstained']} | latency: {r['latency_s']}\n-> {out / (slug + '.md')}")


if __name__ == "__main__":
    main()
