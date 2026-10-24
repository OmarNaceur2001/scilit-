"""
SciLit - extraction/extract.py
Part 7 step 2: fill the module's reading grid for every paper. Every field stores sentence IDs
and the exact sentence text, never a paraphrase (traceability).

  question / contribution <- SciBERT role "objective"   (abstract, introduction)
  method                  <- SciBERT role "method"      (abstract, introduction, conclusion)
  results                 <- SciBERT role "result"      (abstract, results, conclusion)
  data                    <- rule: dataset / corpus / benchmark mentions (abstract, experiments)
  limits                  <- the Limitations section, or limitation cues in conclusion/discussion
  numeric                 <- rule: (metric, value, dataset) triples in result sentences
  hedging (every field)   <- cue lexicon: certain / hedged ("suggest") / speculative ("may")

Input : data/parsed/documents.jsonl, models/roles_scibert/
Output: data/parsed/records.jsonl

Usage: python -m scilit.extraction.extract
"""
from __future__ import annotations

import json
import re
from pathlib import Path

LABELS = ["background", "objective", "method", "result", "other"]
ROLE_SECTIONS = {"abstract", "introduction", "conclusion", "results", "discussion"}
FIELD_SOURCES = {"question": ("objective", {"abstract", "introduction"}),
                 "method": ("method", {"abstract", "introduction", "conclusion"}),
                 "results": ("result", {"abstract", "results", "conclusion"})}
PER_FIELD = 3                                   # most confident sentences kept per field
DATA = re.compile(r"\b(data ?sets?|corpus|corpora|benchmarks?|treebanks?)\b", re.I)
LIMIT = re.compile(r"\b(limitation|fails? to|does not|do not|cannot|restricted to|future work|remains? (an )?open)\b", re.I)
SPECULATIVE = re.compile(r"\b(may|might|could|possibly|potentially|perhaps)\b", re.I)
HEDGED = re.compile(r"\b(suggests?|appears?|seems?|likely|indicates?|tends? to)\b", re.I)
METRIC = r"(accuracy|F1|F-score|BLEU|ROUGE(?:-[12L])?|exact match|EM|MRR|nDCG(?:@\d+)?|recall(?:@\d+)?|precision|perplexity|AUC|WER)"
NUMERIC = [re.compile(METRIC + r"\s*(?:score\s*)?(?:of|=|:|is|reaches|at)?\s*(\d{1,3}(?:\.\d+)?)\s*%?", re.I),
           re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%?\s*(?:points?\s+)?(?:in\s+|of\s+)?" + METRIC, re.I)]
DATASET = re.compile(r"\bon\s+(?:the\s+)?([A-Z][\w\-.]*(?:\s[A-Z0-9][\w\-.]*)?)")


def hedging(text: str) -> str:
    return "speculative" if SPECULATIVE.search(text) else "hedged" if HEDGED.search(text) else "certain"


def numeric_triples(sid: str, text: str) -> list[dict]:
    out = []
    for i, pat in enumerate(NUMERIC):
        for m in pat.finditer(text):
            metric, value = (m.group(1), m.group(2)) if i == 0 else (m.group(2), m.group(1))
            d = DATASET.search(text[m.end():m.end() + 80]) or DATASET.search(text)
            out.append({"sentence": sid, "metric": metric.upper() if len(metric) <= 5 else metric.lower(),
                        "value": float(value), "dataset": d.group(1).rstrip(".") if d else None})
    return out


def main() -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("models/roles_scibert")
    model = AutoModelForSequenceClassification.from_pretrained("models/roles_scibert").to(dev).eval()
    if dev == "cuda":
        model.half()

    def classify(texts: list[str]) -> list[tuple[str, float]]:
        out = []
        with torch.no_grad():
            for k in range(0, len(texts), 128):
                enc = tok(texts[k:k + 128], truncation=True, max_length=96, padding=True, return_tensors="pt").to(dev)
                probs = model(**enc).logits.float().softmax(-1)
                conf, idx = probs.max(-1)
                out += [(LABELS[i], round(c, 3)) for i, c in zip(idx.tolist(), conf.tolist())]
        return out

    out_path = Path("data/parsed/records.jsonl")
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for line in Path("data/parsed/documents.jsonl").open(encoding="utf-8"):
            doc = json.loads(line)
            if not doc["structured"]:
                continue
            sents = [(s, sec["type"]) for sec in doc["sections"] for s in sec["sentences"]]
            to_classify = [(s, t) for s, t in sents if t in ROLE_SECTIONS]
            roles = classify([s["text"] for s, _ in to_classify]) if to_classify else []
            item = lambda s, conf: {"id": s["id"], "text": s["text"], "conf": conf, "hedging": hedging(s["text"])}
            grid = {}
            for field, (role, secs) in FIELD_SOURCES.items():
                cands = [(c, s) for (s, t), (r, c) in zip(to_classify, roles) if r == role and t in secs]
                grid[field] = [item(s, c) for c, s in sorted(cands, key=lambda x: -x[0])[:PER_FIELD]]
            grid["data"] = [item(s, None) for s, t in sents if t in {"abstract", "experiments"} and DATA.search(s["text"])][:PER_FIELD]
            lim = [s for s, t in sents if t == "limitations"] or \
                  [s for s, t in sents if t in {"conclusion", "discussion"} and LIMIT.search(s["text"])]
            grid["limits"] = [item(s, None) for s in lim[:PER_FIELD]]
            numeric = [tr for s, _ in to_classify for tr in numeric_triples(s["id"], s["text"])
                       if any(x["id"] == s["id"] for x in grid["results"]) or _ == "results"]
            f.write(json.dumps({"paper_id": doc["id"], "title": doc["title"], "grid": grid,
                                "numeric": numeric[:10]}, ensure_ascii=False) + "\n")
            n += 1
            if n % 200 == 0:
                print(f"  {n} papers", flush=True)
    print(f"Done: {n} paper records -> {out_path}")


if __name__ == "__main__":
    main()
