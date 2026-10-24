"""
SciLit - generation/review.py
Parts 10-11: write a cited state of the art, then verify every sentence against its sources.

Part 10 · generation (from numbered evidence only)
  evidence = for each retrieved paper: its best passage with context [E1..], plus its result and
             limitation sentences from the reading grid. Nothing else reaches the model.
  template  Bronze baseline: one sentence per table row, quoting the evidence (cannot invent).
  llm       Qwen2.5 (Ollama, local GPU, temperature 0 = reproducible). Contract: use only the
            evidence, cite >= 1 ID per sentence, group papers by idea, write "insufficient
            evidence" rather than guess; output is JSON checked against a schema.

Part 11 · verification (per sentence x cited evidence; non-circular: the generated sentence is
compared with the ORIGINAL passage, not with a claim it was copied from)
  1 existence    every cited ID exists in the evidence pack          -> else "not_found"
  2 attribution  a paper named in the sentence (first-author surname) owns one of the cited
                 passages                                            -> else "misattributed"
  3 numbers      every number in the sentence appears in a cited passage -> else "number_mismatch"
  4 NLI          DeBERTa-v3 (MNLI+FEVER+ANLI): premise = cited passage (2-sentence windows, best
                 window kept), hypothesis = the sentence
                 entailment >= 0.5 -> supported; contradiction >= 0.5 -> contradicted; else insufficient
  Kept in the final review: supported only. Everything else is reported, with counts.

Output: data/reviews/<slug>.json (everything) and <slug>.md (readable review + table)

Usage: python -m scilit.generation.review "LoRA vs full fine-tuning in low-resource settings"
       python -m scilit.generation.review "..." --generator template
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pysbd
import requests

from scilit.corpus.arxiv_meta import surname
from scilit.generation.table import articles, build_table, records, to_markdown

OLLAMA = "http://localhost:11434/api/chat"
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
ENTAIL, CONTRA = 0.5, 0.5
SEG = pysbd.Segmenter(language="en", clean=False)
SCHEMA = {"type": "object", "properties": {"sentences": {"type": "array", "items": {
    "type": "object", "properties": {"text": {"type": "string"}, "cites": {"type": "array", "items": {"type": "string"}}},
    "required": ["text", "cites"]}}}, "required": ["sentences"]}
PROMPT = """You write the related-work paragraph of a literature review on: "{q}".
Rules:
- Use ONLY the evidence below. Do not use outside knowledge.
- Every sentence ends with the IDs of the evidence it relies on, in the "cites" list (e.g. ["E2","E5"]).
- Group papers by idea and compare them; do not write one sentence per paper in order.
- Refer to papers by first author (e.g. "Hu et al."). Copy numbers exactly as in the evidence.
- If the evidence does not support a point, do not write it. 4 to 8 sentences.
- Answer in {lang}.
Output format (JSON only): {{"sentences": [{{"text": "Hu et al. show that ...", "cites": ["E1"]}}, ...]}}
Evidence:
{evidence}"""


def evidence_pack(results: list[dict]) -> list[dict]:
    pack, rec = [], records()
    for r in results:
        a = articles()[r["paper_id"]]
        meta = {"paper_id": r["paper_id"], "title": a["title"], "year": a.get("arxiv_year") or a["year"],
                "first_author": (a["authors"] or ["?"])[0]}
        words = r["text"].split()                        # the matched chunk only, <= 150 words: the
        pack.append({**meta, "kind": "passage",          # whole pack must fit the LLM context window
                     "text": " ".join(words[:150]) + (" ..." if len(words) > 150 else "")})
        for field in ("results", "limits"):
            for cell in (rec.get(r["paper_id"], {}).get("grid", {}).get(field) or [])[:1]:
                if cell["text"] not in r["context"]:
                    pack.append({**meta, "kind": field, "text": cell["text"]})
    for i, e in enumerate(pack, 1):
        e["id"] = f"E{i}"
    return pack


def generate_llm(query: str, pack: list[dict], model: str, lang: str) -> list[dict]:
    ev = "\n".join(f"[{e['id']}] {e['first_author']} et al. ({e['year']}), \"{e['title']}\": {e['text']}" for e in pack)
    r = requests.post(OLLAMA, json={"model": model, "stream": False, "format": SCHEMA,
                                    "options": {"temperature": 0, "num_ctx": 8192},
                                    "messages": [{"role": "user", "content": PROMPT.format(q=query, evidence=ev, lang=lang)}]},
                      timeout=600)
    r.raise_for_status()
    out = []
    for s in json.loads(r.json()["message"]["content"])["sentences"]:
        cites = re.findall(r"E\d+", " ".join(s.get("cites", []))) or re.findall(r"E\d+", s["text"])
        text = re.sub(r"\s*\[(E\d+(,\s*)?)+\]", "", s["text"]).strip()   # IDs live in "cites", not the text
        out.append({"text": text, "cites": list(dict.fromkeys(cites))})
    return out


def generate_template(pack: list[dict]) -> list[dict]:
    out = []
    for e in pack:
        if e["kind"] in ("results", "limits"):
            verb = "report that" if e["kind"] == "results" else "note as a limitation that"
            text = e["text"][0].lower() + e["text"][1:]
            out.append({"text": f"{surname(e['first_author']).title()} et al. ({e['year']}) {verb} {text}", "cites": [e["id"]]})
    return out


@lru_cache(maxsize=1)
def nli():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(dev).eval()
    labels = {v.lower(): k for k, v in model.config.id2label.items()}
    return tok, model, dev, labels["entailment"], labels["contradiction"]


def nli_scores(premises: list[str], hypothesis: str) -> tuple[float, float]:
    import torch
    tok, model, dev, ie, ic = nli()
    windows = []
    for p in premises:
        s = [x.strip() for x in SEG.segment(p) if x.strip()] or [p]
        windows += [" ".join(s[i:i + 2]) for i in range(max(len(s) - 1, 1))]
    with torch.no_grad():
        enc = tok(windows, [hypothesis] * len(windows), truncation=True, max_length=512, padding=True, return_tensors="pt").to(dev)
        probs = model(**enc).logits.float().softmax(-1)
    return probs[:, ie].max().item(), probs[:, ic].max().item()


def strip_attribution(text: str) -> str:
    """'Hu et al. (2021) report that X' -> 'X'. Who said it is checked by the attribution step;
    NLI must only judge WHAT is said, against a passage that never names its own authors."""
    t = re.sub(r"\(?[A-Z][\w\-']+ et al\.?,?\s*\(?(19|20)?\d{0,2}\)?\)?", "", text)
    t = re.sub(r"^\s*(report|show|find|found|note|observe|demonstrate|argue|propose)s?\s+(as a limitation\s+)?(that\s+)?",
               "", t.strip(), flags=re.I)
    return t[:1].upper() + t[1:] if t else text


def numbers(text: str) -> set[str]:
    """Numbers that carry a claim: decimals, percentages, integers > 10; not years or model sizes."""
    out = set()
    for m in re.finditer(r"(?<![\w.\-])(\d+(?:\.\d+)?)(%?)(?![\w])", text):
        v = m.group(1)
        if re.fullmatch(r"(19|20)\d\d", v) or ("." not in v and not m.group(2) and int(v) <= 10):
            continue
        out.add(v.rstrip("0").rstrip(".") if "." in v else v)
    return out


def fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def verify(sentences: list[dict], pack: list[dict]) -> list[dict]:
    by_id = {e["id"]: e for e in pack}
    out = []
    for s in sentences:
        cited = [by_id[c] for c in s["cites"] if c in by_id]
        row = {**s, "checks": {}}
        if not s["cites"] or len(cited) < len(s["cites"]):
            row["status"] = "not_found"                       # invented or missing evidence ID
            out.append(row)
            continue
        named = {e["paper_id"] for e in pack if surname(e["first_author"]) and
                 re.search(rf"\b{re.escape(surname(e['first_author']))}\b", fold(s["text"]))}
        if named and not named & {e["paper_id"] for e in cited}:
            row["status"] = "misattributed"
            out.append(row)
            continue
        missing = numbers(s["text"]) - set().union(*(numbers(e["text"]) for e in cited))
        row["checks"]["numbers_missing"] = sorted(missing)
        if missing:
            row["status"] = "number_mismatch"
            out.append(row)
            continue
        ent, con = nli_scores([e["text"] for e in cited], strip_attribution(s["text"]))
        row["checks"].update(entailment=round(ent, 3), contradiction=round(con, 3))
        row["status"] = "supported" if ent >= ENTAIL else "contradicted" if con >= CONTRA else "insufficient"
        out.append(row)
    return out


def main() -> None:
    from scilit.retrieval.search import Searcher
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--generator", choices=["llm", "template"], default="llm")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--lang", default="English")
    args = ap.parse_args()

    results = Searcher().search(args.query, k=args.k)
    rows, pack = build_table(results), evidence_pack(results)
    draft = generate_llm(args.query, pack, args.model, args.lang) if args.generator == "llm" else generate_template(pack)
    checked = verify(draft, pack)
    counts = {st: sum(1 for c in checked if c["status"] == st)
              for st in ["supported", "insufficient", "contradicted", "number_mismatch", "misattributed", "not_found"]}

    slug = re.sub(r"[^a-z0-9]+", "-", fold(args.query)).strip("-")[:60] + f"_{args.generator}"
    out = Path("data/reviews"); out.mkdir(parents=True, exist_ok=True)
    (out / f"{slug}.json").write_text(json.dumps({"query": args.query, "generator": args.generator, "model": args.model,
        "table": rows, "evidence": pack, "sentences": checked, "counts": counts}, ensure_ascii=False, indent=1), encoding="utf-8")
    kept = [c for c in checked if c["status"] == "supported"]
    md = [f"# {args.query}", "", " ".join(f"{c['text']} [{', '.join(c['cites'])}]" for c in kept) or "*No supported sentence.*",
          "", "## Removed by verification", ""] + [f"- **{c['status']}**: {c['text']}" for c in checked if c["status"] != "supported"] + \
         ["", "## Comparison table", "", to_markdown(rows), "", "## Evidence", ""] + \
         [f"- [{e['id']}] {e['first_author']} ({e['year']}), *{e['title']}*" for e in pack]
    (out / f"{slug}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"{len(checked)} sentences: {counts}")
    print(f"-> {out / (slug + '.md')}")


if __name__ == "__main__":
    main()
