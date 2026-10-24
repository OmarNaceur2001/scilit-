"""
SciLit - generation/review.py
Parts 10-11: answer a research question the way a literature review is written.

  1 find        the 8 most relevant papers (hybrid search + rerank, paper level: the ranking
                evaluated in Part 12)
  2 read        each paper on its own -> a reading card of verified claims (reader.py)
  3 abstain     if fewer than 2 papers have a supported claim, the corpus does not cover the
                question: SciLit says so and lists the closest papers, instead of writing
  4 synthesise  Qwen2.5-7B writes from the verified claims only (JSON schema, temperature 0):
                  answer   1-2 sentences that answer the question directly
                  aspects  2-4 groups (mechanisms, methods, findings, disagreements)
                  gaps     what the evidence does not show (states an absence: not verifiable)
                every sentence cites the 1-3 claims it combines (C1, C2...)
  5 verify      every sentence against the ORIGINAL source sentences behind the claims it cites
                (verify.check: existence, attribution, numbers, NLI)
  6 repair      a failed sentence gets ONE rewrite as a single short claim from its own sources,
                or is dropped (the RARR idea, Gao et al. 2023); the rewrite is verified again
  7 keep        only supported sentences form the review; everything else is shown with its reason

Baseline (--generator template): reading = the sentence of each paper closest to the question,
quoted; synthesis = one "Surname et al. (year) report that ..." per paper. Cannot invent anything.

Output: data/reviews/<slug>.json (everything) and <slug>.md (readable)
Usage : python -m scilit.generation.review "How does retrieval-augmented generation reduce hallucination?"
        python -m scilit.generation.review "..." --generator template | --no-repair
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from scilit.corpus.arxiv_meta import surname
from scilit.generation import verify as V
from scilit.generation.llm import MODEL, ollama
from scilit.generation.reader import documents, evidence, read_paper
from scilit.generation.table import articles, build_table, to_markdown

K_PAPERS, MIN_PAPERS, MAX_WORDS = 8, 2, 40

def schema(claim_ids: list[str]) -> dict:
    """Synthesis schema. "cites" is an ENUM of the existing claim IDs (no invented citation can be
    generated); "gaps" is required: a review always says what its evidence does not cover."""
    sent = {"type": "object", "required": ["text", "cites"], "properties": {"text": {"type": "string"},
            "cites": {"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "string", "enum": claim_ids}}}}
    return {"type": "object", "required": ["answer", "aspects", "gaps"], "properties": {
        "answer": {"type": "array", "minItems": 1, "maxItems": 2, "items": sent},
        "aspects": {"type": "array", "minItems": 2, "maxItems": 4, "items": {"type": "object", "required": ["title", "sentences"],
                    "properties": {"title": {"type": "string"}, "sentences": {"type": "array", "minItems": 1, "maxItems": 3, "items": sent}}}},
        "gaps": {"type": "string", "minLength": 20}}}


PROMPT = """Research question: "{q}"

Below are verified claims from {n} papers, each with an ID. Write the answer of a literature review
using ONLY these claims.

- "answer": 1 or 2 sentences that directly answer the question by combining the claims.
- "aspects": 2 to 4 groups (for example mechanisms, methods, findings, disagreements), each with a
  short title and 1 to 3 sentences that explain HOW or WHY, and compare papers when they agree or differ.
- "gaps": one sentence on what these claims do not show about the question.

Rules:
- Each sentence lists in "cites" the IDs of the 1 to 3 claims it combines, and says nothing they do not say.
- Each sentence has at most 35 words. Copy numbers exactly.
- Name a paper only by the surname shown ("Niu et al."). Never write IDs, years, "as shown in" or any
  citation in the text: citations go only in "cites".

Claims:
{claims}"""
REPAIR_SCHEMA = {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": {
    "type": "object", "required": ["id", "text"], "properties": {"id": {"type": "integer"}, "text": {"type": "string"}}}}}}
REPAIR_PROMPT = """Each item is a sentence from a literature review that is NOT supported by its sources.
Rewrite it as ONE short sentence (at most 35 words) that says only what its sources state, copying
numbers exactly, with no IDs or citations in the text; or answer "DROP" if the sources do not
support its point.

{items}

Return one entry per item: {{"id": <item number>, "text": "<rewritten sentence or DROP>"}}."""


def meta(pid: str) -> dict:
    a = articles()[pid]
    return {"paper_id": pid, "title": a["title"], "year": a.get("arxiv_year") or a["year"],
            "first_author": (a["authors"] or ["?"])[0]}


def clean(text: str, cites: list[str]) -> dict:
    """IDs live in "cites"; remove any the model still wrote in the text, and "(Ji et al., 2022)"."""
    cites = list(dict.fromkeys(re.findall(r"C\d+", " ".join(cites)) or re.findall(r"C\d+", text)))
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


def synthesise(query: str, claims: dict, cards: list[dict], model: str) -> tuple[list[dict], str]:
    blocks = []
    for card in cards:
        mine = [c for c in claims.values() if c["paper_id"] == card["paper_id"]]
        if mine:
            blocks.append(f"{surname(card['first_author']).title()} et al. ({card['year']}), \"{card['title']}\":\n" +
                          "\n".join(f"  [{c['id']}] {c['text']}" for c in mine))
    out = ollama(PROMPT.format(q=query, n=len(blocks), claims="\n\n".join(blocks)), schema(list(claims)), model)
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


def repair(checked: list[dict], claims: dict, known: set[str], model: str) -> list[dict]:
    todo = [i for i, s in enumerate(checked) if s["status"] != "supported" and any(c in claims for c in s["cites"])]
    if not todo:
        return checked
    items = "\n\n".join(f"Item {n}\nSentence: {checked[i]['text']}\nSources: " +
                        " ".join(t for c in checked[i]["cites"] if c in claims for t in claims[c]["source_text"])
                        for n, i in enumerate(todo))
    answers = {a["id"]: a["text"].strip() for a in ollama(REPAIR_PROMPT.format(items=items), REPAIR_SCHEMA, model)["items"]}
    out = list(checked)
    for n, i in enumerate(todo):
        s, new = out[i], answers.get(n, "DROP")
        if not new or new.upper().startswith("DROP") or len(new.split()) > MAX_WORDS:
            out[i] = {**s, "repair": "dropped"}          # a rewrite may not grow: no padding to pass the check
            continue
        fixed = check_sentence({**s, **clean(new, [c for c in s["cites"] if c in claims])}, claims, known)
        out[i] = {**fixed, "original": s["text"], "original_status": s["status"],
                  "repair": "repaired" if fixed["status"] == "supported" else "failed"}
    return out


def run_review(query: str, searcher, generator: str = "llm", model: str = MODEL, do_repair: bool = True) -> dict:
    """The whole review, used by the command line AND the API."""
    t = [time.time()]
    papers = searcher.search(query, k=K_PAPERS)                     # one passage per paper = paper ranking
    docs = documents()
    t.append(time.time())
    cards = [read_paper(query, meta(p["paper_id"]), evidence(query, docs.get(p["paper_id"]), searcher), generator, model)
             for p in papers]
    claims = claim_index(cards)
    t.append(time.time())
    known = {surname(c["first_author"]) for c in cards} - {""}
    covered = {c["paper_id"] for c in claims.values()}
    result = {"query": query, "generator": generator, "model": model if generator == "llm" else None,
              "repair": generator == "llm" and do_repair, "cards": cards, "claims": claims,
              "claim_counts": V.count([c for card in cards for c in card["claims"]]),
              "table": build_table(papers), "abstained": len(covered) < MIN_PAPERS}
    if result["abstained"]:
        sentences, gaps = [], (f"The corpus does not cover this question: {len(covered)} of {len(cards)} retrieved "
                               f"papers contain a verified claim about it.")
        before = V.count([])
    else:
        draft, gaps = synthesise(query, claims, cards, model) if generator == "llm" else template_synthesis(claims)
        sentences = [check_sentence(s, claims, known) for s in draft]
        before = V.count(sentences)
        if generator == "llm" and do_repair:
            sentences = repair(sentences, claims, known, model)
    t.append(time.time())
    result.update(sentences=sentences, gaps=gaps, counts_before_repair=before, counts=V.count(sentences),
                  latency_s={k: round(t[i + 1] - t[i], 1) for i, k in enumerate(["search", "reading", "synthesis"])})
    return result


def to_md(r: dict) -> str:
    kept = [s for s in r["sentences"] if s["status"] == "supported"]
    md = [f"# {r['query']}", ""]
    if r["abstained"]:
        md += [f"**{r['gaps']}**", ""]
    for part in dict.fromkeys(s["part"] for s in kept):
        md += [f"**{'Answer' if part == 'answer' else part}.** " +
               " ".join(f"{s['text']} [{', '.join(s['cites'])}]" for s in kept if s["part"] == part), ""]
    if r["gaps"] and not r["abstained"]:
        md += [f"*Not covered by the evidence:* {r['gaps']}", ""]
    md += ["## Removed by verification", ""] + [f"- **{s['status']}**: {s['text']}" for s in r["sentences"]
                                                  if s["status"] != "supported"]
    md += ["", f"Claims (reading cards): {r['claim_counts']}", f"Synthesis before repair: {r['counts_before_repair']}",
           f"Synthesis after repair: {r['counts']}", "", "## Reading cards", ""]
    for card in r["cards"]:
        md.append(f"### {card['first_author']} ({card['year']}), {card['title']} - "
                  f"{'relevant' if card['relevant'] else 'not relevant'}")
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
    ap.add_argument("--no-repair", action="store_true")
    args = ap.parse_args()
    r = run_review(args.query, Searcher(), args.generator, args.model, not args.no_repair)
    tag = args.generator + ("_norepair" if args.generator == "llm" and args.no_repair else "")
    slug = re.sub(r"[^a-z0-9]+", "-", V.fold(args.query)).strip("-")[:60] + f"_{tag}"
    out = Path("data/reviews"); out.mkdir(parents=True, exist_ok=True)
    (out / f"{slug}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / f"{slug}.md").write_text(to_md(r), encoding="utf-8")
    print(f"claims (reading cards): {r['claim_counts']}")
    print(f"synthesis before repair: {r['counts_before_repair']}\nsynthesis after repair : {r['counts']}")
    print(f"abstained: {r['abstained']} | latency: {r['latency_s']}\n-> {out / (slug + '.md')}")


if __name__ == "__main__":
    main()
