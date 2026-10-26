"""
SciLit - api/main.py
Part 13: FastAPI app over the pipeline (the same code that produced the evaluation numbers).

  GET  /              web page (style of the original SciLit UI)
  GET  /search?q=     hybrid search -> papers + best passage
  POST /review        {query, generator: llm|template} -> answer + verification + table
  GET  /bibtex?ids=   BibTeX for paper IDs (comma-separated)
  GET  /docs          automatic API documentation

Usage: uvicorn scilit.api.main:app --port 8000   then open http://127.0.0.1:8000
"""
from __future__ import annotations

import re
import time
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel

from scilit.corpus.arxiv_meta import surname
from scilit.generation import review as R
from scilit.generation.table import articles
from scilit.retrieval.search import Searcher

app = FastAPI(title="SciLit", description="Verified literature-review assistant")
searcher = Searcher()


class ReviewRequest(BaseModel):
    query: str
    generator: str = "llm"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui():
    return (Path(__file__).parent / "ui.html").read_text(encoding="utf-8")


@app.get("/search")
def search(q: str, k: int = 10):
    t = time.time()
    res = searcher.search(q, k=k)
    return {"query": q, "results": res, "latency_s": round(time.time() - t, 2)}


@app.post("/review")
def review(req: ReviewRequest):
    """Same function as the command line: evidence passages -> plan -> sentences -> verification."""
    return R.run_review(req.query, searcher, req.generator)


@app.get("/bibtex", response_class=PlainTextResponse)
def bibtex(ids: str = Query(..., description="comma-separated paper IDs")):
    out = []
    for pid in ids.split(","):
        a = articles().get(pid.strip())
        if not a:
            continue
        year = a.get("arxiv_year") or a["year"]
        word = next((w for w in re.findall(r"[A-Za-z]+", a["title"]) if w.lower() not in {"a", "an", "the", "on", "of"}), "x")
        key = f"{surname(a['authors'][0]) if a['authors'] else 'anon'}{year}{word.lower()}"
        fields = {"title": "{" + a["title"] + "}", "author": " and ".join(a["authors"]), "year": str(year)}
        if a.get("arxiv_id"):
            fields.update(eprint=a["arxiv_id"], archivePrefix="arXiv", url=f"https://arxiv.org/abs/{a['arxiv_id']}")
        if a.get("doi") and not a["doi"].startswith("10.48550"):
            fields["doi"] = a["doi"]
        out.append(f"@article{{{key},\n" + ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items()) + "\n}")
    return "\n\n".join(out)


@app.get("/stats")
def stats():
    """Evaluation results produced by the pipeline (Part 12), read from reports/."""
    import json
    rep = Path("reports")
    read = lambda f: json.loads((rep / f).read_text(encoding="utf-8")) if (rep / f).exists() else None
    retrieval = []
    if (rep / "retrieval_eval.md").exists():
        text = (rep / "retrieval_eval.md").read_text(encoding="utf-8").split("## Condensed")[0]   # standard table
        for line in text.splitlines():
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) == 4 and re.match(r"\d", cells[1]):
                retrieval.append({"system": cells[0], "recall10": cells[1], "mrr": cells[2], "ndcg10": cells[3]})
    return {"papers": sum(1 for _ in Path("data/parsed/records.jsonl").open(encoding="utf-8")),
            "chunks": len(searcher.chunks), "retrieval": retrieval,
            "roles": read("roles_results.json"), "citfunc": read("citfunc_results.json")}
