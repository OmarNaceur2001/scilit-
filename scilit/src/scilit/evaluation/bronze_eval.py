"""
SciLit - evaluation/bronze_eval.py
Bronze review: every classical component, measured on the frozen test set against a trivial
baseline, plus the error analysis. No neural model is loaded.

  Lot D  retrieval        BM25 vs random ranking (30 test queries): Recall@10, MRR, nDCG@10,
                          English and French versions of the same queries
  Lot B  sections         heading rules (parse_pdf.section_type + inheritance) vs majority class,
                          macro-F1 on labelled headings, dev and test papers separately
  Lot B  claims           cue patterns (roles.RULES) vs majority class, macro-F1, CSAbstruct dev and test
                          (deduplicated splits)
  Lot C  representations  TF-IDF word 1-2 grams + logistic regression on the same data
  Lot D  output           the Bronze pipeline (BM25 + template review, review.run_review bronze=True) on the
                          30 test queries: rate of references that exist (paper in the corpus AND quoted
                          sentence present word for word in that paper), precision@8 of the papers it
                          reads, latency; 5 out-of-scope queries: abstention rate
  errors                  categorised errors of the three rule/BM25 components -> reports/bronze_errors.csv;
                          categories come from what the rule did (which cue fired, which words matched),
                          so they are reproducible
  scope rule              out-of-scope questions: the Bronze pipeline abstains when even its best passage
                          covers too little of the question (IDF-weighted share of the query words,
                          Searcher.coverage). The threshold is chosen on labels/dev_scope.tsv (12 in-scope,
                          12 out-of-scope questions, never the test set) by tune-scope, and reported on test:
                          the 30 test questions must be answered, the 5 OUT_OF_SCOPE ones must not
  freeze                  SHA-256 and first commit date of the frozen test files

Usage
  python -m scilit.evaluation.bronze_eval sections-export   # sample headings -> labels/sections/headings.csv
  python -m scilit.evaluation.bronze_eval tune-scope         # scope threshold on dev -> configs/bronze.json
  python -m scilit.evaluation.bronze_eval run                # everything -> reports/bronze_eval.md / .json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import statistics as st
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np

SDIR = Path("labels/sections")
FROZEN = ["labels/queries.tsv", "labels/qrels_chatgpt.tsv", "labels/heldout.tsv", "configs/splits.json",
          "labels/dev_scope.tsv"]
DEV_SCOPE = Path("labels/dev_scope.tsv")
OUT_OF_SCOPE = ["Quelle est la recette du couscous ?", "Who won the 2018 FIFA World Cup?",
                "Best price for a used car in Tunis", "kifech nesta3mel el machine à laver",
                "How do I bake sourdough bread at home?"]
SECTION_LABELS = ["abstract", "introduction", "related_work", "method", "experiments", "results", "discussion",
                  "conclusion", "limitations", "ethics", "acknowledgements", "references", "appendix", "other"]
N_HEADINGS = 75                     # per split (dev, test)


def macro(gold, pred, labels) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(gold, pred, labels=sorted(set(gold)), average="macro", zero_division=0))


def acc(gold, pred) -> float:
    return sum(g == p for g, p in zip(gold, pred)) / max(len(gold), 1)


# ------------------------------------------------------------------ Lot D: retrieval

def retrieval(searcher, qrels: dict, errors: list) -> dict:
    from scilit.evaluation.judge import queries
    from scilit.evaluation.retrieval_eval import scores
    from scilit.retrieval.build_index import tokenize
    papers = sorted({c["paper_id"] for c in searcher.chunks.values()})
    rng = random.Random(0)
    out = {}
    for lang, field in (("EN", "query_en"), ("FR", "query_fr")):
        bm, rnd = [], []
        for q in queries():
            rel = qrels.get(q["qid"], {})
            res = searcher.search(q[field], k=20, method="bm25")
            ranked = [r["paper_id"] for r in res]
            bm.append(scores(ranked, rel))
            rnd.append(np.mean([scores(rng.sample(papers, min(20, len(papers))), rel) for _ in range(200)], axis=0))
            if lang == "FR":
                if scores(ranked, rel)[0] == 0:
                    errors.append(("retrieval", "French query: no shared word with English papers", q[field],
                                   (res[0]["title"] if res else "no result"), q["query_en"]))
                continue
            qt = set(tokenize(q[field]))
            for r in res[:10]:
                g = rel.get(r["paper_id"])
                if g != 0:
                    continue                                 # relevant, partly relevant, or never judged
                matched = qt & set(tokenize(searcher.chunks[r["chunk_id"]]["header"] + " " + r["text"]))
                cat = ("match inside a related-work section" if r["section"] == "related_work"
                       else "single query word matched" if len(matched) <= 1
                       else "query words matched, different topic")
                errors.append(("retrieval", cat, q[field], f"#{r['rank']} {r['title']} [{', '.join(sorted(matched))}]",
                               "judged not relevant"))
            top = {r["paper_id"] for r in res[:10]}
            for pid, g in rel.items():
                if g == 2 and pid not in top:
                    words = set()
                    for cid in searcher.chunks_by_paper.get(pid, []):
                        c = searcher.chunks[cid]
                        words |= set(tokenize(c["header"] + " " + c["text"]))
                    shared = qt & words
                    cat = ("vocabulary mismatch (<= 1 query word in the paper)" if len(shared) <= 1
                           else "relevant paper ranked below 10")
                    errors.append(("retrieval", cat, q[field], f"missed: {searcher.titles.get(pid, pid)}",
                                   f"relevant; shares [{', '.join(sorted(shared))}]"))
        b, r_ = np.array(bm).mean(0), np.array(rnd).mean(0)
        out[lang] = {"n": len(bm), "bm25": dict(zip(["recall10", "mrr", "ndcg10"], b.round(3).tolist())),
                     "random": dict(zip(["recall10", "mrr", "ndcg10"], r_.round(3).tolist()))}
    return out


# ------------------------------------------------------------------ Lot B: sections

def sections_export() -> None:
    """A fixed random sample of headings from dev and test papers, for labelling. The sheet shows the
    heading, its parent heading and the first sentence of the section, never the rule's decision."""
    from scilit.generation.reader import DOCUMENTS
    split = json.loads(Path("configs/splits.json").read_text(encoding="utf-8"))
    which = {p: "dev" for p in split["dev"]} | {p: "test" for p in split["test"]}
    pool = {"dev": [], "test": []}
    for line in DOCUMENTS.open(encoding="utf-8"):
        doc = json.loads(line)
        if doc["id"] not in which:
            continue
        parent = ""
        for k, sec in enumerate(doc["sections"]):
            if not sec.get("heading"):
                continue
            if sec.get("level", 1) == 1:
                parent = sec["heading"]
            first = sec["sentences"][0]["text"] if sec.get("sentences") else ""
            pool[which[doc["id"]]].append({"paper_id": doc["id"], "section": k, "heading": sec["heading"],
                                           "parent": "" if sec.get("level", 1) == 1 else parent,
                                           "level": sec.get("level", 1), "first_sentence": first[:200],
                                           "rule": sec["type"]})
    rng = random.Random(0)
    items = []
    for s in ("dev", "test"):
        items += [{**it, "split": s} for it in rng.sample(pool[s], min(N_HEADINGS, len(pool[s])))]
    SDIR.mkdir(parents=True, exist_ok=True)
    with (SDIR / "key.jsonl").open("w", encoding="utf-8") as f:
        for i, it in enumerate(items, 1):
            f.write(json.dumps({"item": i, **it}, ensure_ascii=False) + "\n")
    with (SDIR / "headings.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["item", "parent_heading", "heading", "first_sentence", "label"])
        w.writerows([i, it["parent"], it["heading"], it["first_sentence"], ""] for i, it in enumerate(items, 1))
    print(f"{len(items)} headings -> {SDIR}/headings.csv. Label column: one of {', '.join(SECTION_LABELS)}")


def sections(errors: list) -> dict | None:
    from scilit.corpus.parse_pdf import section_type
    if not (SDIR / "headings.csv").exists():
        return None
    gold = {int(r["item"]): r["label"].strip().lower() for r in csv.DictReader((SDIR / "headings.csv").open(encoding="utf-8-sig"))
            if r["label"].strip().lower() in SECTION_LABELS}
    key = {it["item"]: it for it in map(json.loads, (SDIR / "key.jsonl").open(encoding="utf-8"))}
    if not gold:
        return None
    dev_gold = [gold[i] for i in gold if key[i]["split"] == "dev"]
    majority = Counter(dev_gold).most_common(1)[0][0]            # chosen on dev, never on test
    out = {"majority_class": majority, "labels": "LLM-labelled headings (declared)"}
    for s in ("dev", "test"):
        ids = [i for i in sorted(gold) if key[i]["split"] == s]
        g, p = [gold[i] for i in ids], [key[i]["rule"] for i in ids]
        out[s] = {"n": len(ids), "rules_macro_f1": round(macro(g, p, SECTION_LABELS), 3), "rules_acc": round(acc(g, p), 3),
                  "majority_macro_f1": round(macro(g, [majority] * len(g), SECTION_LABELS), 3),
                  "majority_acc": round(acc(g, [majority] * len(g)), 3)}
        for i in ids:
            it, gl = key[i], gold[i]
            if it["rule"] == gl:
                continue
            direct = section_type(it["heading"])
            cat = ("heading word unknown to the lexicon -> inherited parent type" if direct is None and it["level"] > 1
                   else "heading word unknown to the lexicon -> other" if direct is None
                   else "inside appendix: forced to appendix" if direct != it["rule"]
                   else f"lexicon keyword points to another type ({direct})")
            errors.append(("sections", cat, f"{it['parent'] + ' > ' if it['parent'] else ''}{it['heading']}", it["rule"], gl))
    return out


# ------------------------------------------------------------------ Lot B / C: claims by patterns, TF-IDF

def fired(text: str) -> tuple[str, str | None]:
    """roles.rules_predict with the cue that decided (None = no cue, default background)."""
    from scilit.extraction.roles import RULES
    low = text.lower()
    for lab, pat in RULES:
        m = re.search(pat, low)
        if m:
            return lab, m.group(0)
    return "background", None


def claims(errors: list) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from scilit.extraction.roles import LABELS, dedupe, load, rules_predict
    train, dev, test = load("train"), load("dev"), load("test")
    dev, test, info = dedupe(train, dev, test)
    majority = Counter(train[1]).most_common(1)[0][0]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    lr = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced").fit(vec.fit_transform(train[0]), train[1])
    out = {"data": info, "majority_class": majority}
    for name, (x, y) in (("dev", dev), ("test", test)):
        rules, tf = rules_predict(x), lr.predict(vec.transform(x)).tolist()
        out[name] = {"n": len(y), "majority_macro_f1": round(macro(y, [majority] * len(y), LABELS), 3),
                     "rules_macro_f1": round(macro(y, rules, LABELS), 3), "rules_acc": round(acc(y, rules), 3),
                     "tfidf_macro_f1": round(macro(y, tf, LABELS), 3), "tfidf_acc": round(acc(y, tf), 3)}
        if name == "test":
            for text, g, p in zip(x, y, rules):
                if g == p:
                    continue
                lab, cue = fired(text)
                cat = (f"no cue: defaulted to background (true: {g})" if cue is None
                       else f"{lab} cue fired on {'an' if g[0] in 'aeiou' else 'a'} {g} sentence")
                errors.append(("claims (patterns)", cat, text, f"{p} [cue: {cue}]" if cue else p, g))
    return out


# ------------------------------------------------------------------ Lot D: the Bronze pipeline's output

def output(searcher, qrels: dict) -> dict:
    from scilit.evaluation.judge import queries
    from scilit.generation.reader import documents
    from scilit.generation.review import run_review
    docs = documents()
    refs = valid = 0
    lat, kept, prec, abst = [], [], [], 0
    for q in queries():
        t = time.time()
        r = run_review(q["query_en"], searcher, bronze=True)
        lat.append(time.time() - t)
        abst += r["abstained"]
        kept.append(r["counts"]["supported"])
        rel = qrels.get(q["qid"], {})
        prec.append(sum(rel.get(c["paper_id"]) == 2 for c in r["cards"]) / max(len(r["cards"]), 1))
        for s in r["sentences"]:
            for cid in s["cites"]:
                refs += 1
                c = r["claims"].get(cid)
                if not c:
                    continue
                try:
                    doc = docs.get(c["paper_id"])
                except KeyError:
                    continue
                text = {x["id"]: x["text"] for sec in doc["sections"] for x in sec["sentences"]}
                valid += all(text.get(sid) == quote for sid, quote in zip(c["sources"], c["quotes"]))
    oos = [run_review(x, searcher, bronze=True)["abstained"] for x in OUT_OF_SCOPE]
    from scilit.generation.review import scope_threshold
    answered = (len(lat) - abst) / max(len(lat), 1)
    return {"scope_threshold": scope_threshold(), "scope_balanced_accuracy": round((answered + sum(oos) / len(oos)) / 2, 3),
            "in_scope_answered": f"{len(lat) - abst}/{len(lat)}","queries": len(lat), "references": refs, "references_existing": round(valid / max(refs, 1), 3),
            "precision_at_8_relevant": round(st.mean(prec), 3), "kept_per_review": round(st.mean(kept), 2),
            "abstained": abst, "latency_s_mean": round(st.mean(lat), 2), "latency_s_max": round(max(lat), 2),
            "out_of_scope_abstained": f"{sum(oos)}/{len(oos)}"}


# ------------------------------------------------------------------ scope rule

def best_coverage(searcher, query: str) -> float:
    from scilit.generation.review import K_PAPERS
    return max((searcher.coverage(query, r["chunk_id"]) for r in searcher.search(query, k=K_PAPERS, method="bm25")),
               default=0.0)


def balanced(in_cov: list[float], out_cov: list[float], t: float) -> float:
    return (sum(c >= t for c in in_cov) / max(len(in_cov), 1) + sum(c < t for c in out_cov) / max(len(out_cov), 1)) / 2


def tune_scope() -> None:
    """Threshold with the best balanced accuracy on the dev questions; among equally good ones, the
    middle of the widest gap (largest margin on both sides)."""
    from datetime import date
    from scilit.generation.review import SCOPE_FILE
    from scilit.retrieval.search import Searcher
    searcher = Searcher()
    rows = list(csv.DictReader(DEV_SCOPE.open(encoding="utf-8"), delimiter="\t"))
    cov = [(best_coverage(searcher, r["query"]), r["in_scope"] == "1", r["query"]) for r in rows]
    in_c, out_c = [c for c, i, _ in cov if i], [c for c, i, _ in cov if not i]
    pts = sorted({c for c, _, _ in cov})
    cands = [(a + b) / 2 for a, b in zip(pts, pts[1:])] or [0.0]
    best = max(balanced(in_c, out_c, t) for t in cands)
    good = [i for i, t in enumerate(cands) if balanced(in_c, out_c, t) == best]
    t = max((cands[i] for i in good), key=lambda t: min(abs(t - c) for c, _, _ in cov))
    SCOPE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCOPE_FILE.write_text(json.dumps({"scope_threshold": round(t, 3), "chosen_on": str(DEV_SCOPE),
                                      "dev_balanced_accuracy": round(best, 3), "date": str(date.today())}, indent=1),
                          encoding="utf-8")
    for c, i, q in sorted(cov):
        print(f"  {c:.3f} {'in ' if i else 'OUT'} {'answered' if c >= t else 'abstains'}  {q}")
    print(f"threshold {t:.3f} (dev balanced accuracy {best:.3f}) -> {SCOPE_FILE}")


# ------------------------------------------------------------------ freeze proof

def freeze() -> list[dict]:
    out = []
    for f in FROZEN:
        p = Path(f)
        if not p.exists():
            continue
        try:
            date = subprocess.run(["git", "log", "--diff-filter=A", "--format=%ad", "--date=short", "--", f],
                                  capture_output=True, text=True, timeout=20).stdout.split()
        except Exception:
            date = []
        out.append({"file": f, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16],
                    "first_commit": date[-1] if date else "not committed"})
    return out


# ------------------------------------------------------------------ report

def run() -> None:
    from scilit.evaluation.retrieval_eval import load_qrels
    from scilit.retrieval.search import Searcher
    searcher, qrels, errors = Searcher(), load_qrels("labels/qrels_chatgpt.tsv"), []
    res = {"retrieval": retrieval(searcher, qrels, errors)}
    print("retrieval done", flush=True)
    res["sections"] = sections(errors)
    res["claims"] = claims(errors)
    print("rules done", flush=True)
    res["output"] = output(searcher, qrels)
    res["freeze"] = freeze()
    Path("reports").mkdir(exist_ok=True)
    with Path("reports/bronze_errors.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["component", "category", "input", "output", "reference"])
        w.writerows(errors)
    cats = Counter((e[0], e[1]) for e in errors)
    res["errors"] = {f"{c} | {k}": n for (c, k), n in sorted(cats.items())}
    Path("reports/bronze_eval.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")

    R, S, C, O = res["retrieval"], res["sections"], res["claims"], res["output"]
    md = ["# Bronze evaluation (classical components only)", "",
          "## Table 1 — Results per component on the frozen test set", "",
          "| Component | Lot | Bronze method | Trivial baseline | Metric | Baseline | Bronze | N test |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    e = R["EN"]
    md += [f"| Retrieval | D | BM25 | random ranking | Recall@10 (↑) | {e['random']['recall10']} | {e['bm25']['recall10']} | {e['n']} queries |",
           f"| Retrieval | D | BM25 | random ranking | MRR (↑) | {e['random']['mrr']} | {e['bm25']['mrr']} | {e['n']} queries |"]
    if S:
        md.append(f"| Sections | B | heading lexicon + inheritance | majority ({S['majority_class']}) | macro-F1 (↑) | "
                  f"{S['test']['majority_macro_f1']} | {S['test']['rules_macro_f1']} | {S['test']['n']} headings |")
    t = C["test"]
    md += [f"| Claims (sentence roles) | B | cue patterns | majority ({C['majority_class']}) | macro-F1 (↑) | {t['majority_macro_f1']} | {t['rules_macro_f1']} | {t['n']} sentences |",
           f"| Claims (sentence roles) | C | TF-IDF 1-2 grams + logistic regression | majority ({C['majority_class']}) | macro-F1 (↑) | {t['majority_macro_f1']} | {t['tfidf_macro_f1']} | {t['n']} sentences |",
           f"| Review by template | D | BM25 + quoted sentences | — | references that exist (↑) | — | {O['references_existing']} | {O['references']} references |",
           f"| Out-of-scope rule | E | IDF-weighted coverage ≥ {O['scope_threshold']} (chosen on dev) | always answer | balanced accuracy (↑) | 0.5 | {O['scope_balanced_accuracy']} | {O['queries']} in + {len(OUT_OF_SCOPE)} out |",
           "", "Rule components on dev and test separately (macro-F1):", "",
           "| Component | dev | test |", "| --- | --- | --- |"]
    if S:
        md.append(f"| Sections (rules) | {S['dev']['rules_macro_f1']} (n={S['dev']['n']}) | {S['test']['rules_macro_f1']} (n={S['test']['n']}) |")
    md += [f"| Claims (cue patterns) | {C['dev']['rules_macro_f1']} (n={C['dev']['n']}) | {t['rules_macro_f1']} (n={t['n']}) |",
           "", f"Bronze pipeline on the {O['queries']} test queries: {O['kept_per_review']} quoted sentences per review, "
           f"{O['precision_at_8_relevant']} of the papers read judged relevant, latency {O['latency_s_mean']} s "
           f"(max {O['latency_s_max']} s); in-scope questions answered {O['in_scope_answered']}, out-of-scope questions abstained {O['out_of_scope_abstained']}.",
           "", "## Table 2 — Results per language (same 30 queries)", "",
           "| Component | Metric | EN | FR |", "| --- | --- | --- | --- |",
           f"| Retrieval BM25 | Recall@10 | {R['EN']['bm25']['recall10']} | {R['FR']['bm25']['recall10']} |",
           f"| Retrieval BM25 | MRR | {R['EN']['bm25']['mrr']} | {R['FR']['bm25']['mrr']} |",
           "", "## Table 3 — Error analysis (all errors in reports/bronze_errors.csv)", "",
           "| Component | Category | Count |", "| --- | --- | --- |"]
    md += [f"| {c} | {k} | {n} |" for (c, k), n in sorted(cats.items(), key=lambda x: (x[0][0], -x[1]))]
    md += ["", "## Frozen test files", "", "| File | SHA-256 (16) | First commit |", "| --- | --- | --- |"]
    md += [f"| {x['file']} | {x['sha256']} | {x['first_commit']} |" for x in res["freeze"]]
    if not S:
        md += ["", "Sections not scored yet: run `sections-export`, fill labels/sections/headings.csv, run again."]
    Path("reports/bronze_eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sections-export", "tune-scope", "run"])
    a = ap.parse_args()
    {"sections-export": sections_export, "tune-scope": tune_scope, "run": run}[a.cmd]()


if __name__ == "__main__":
    main()
