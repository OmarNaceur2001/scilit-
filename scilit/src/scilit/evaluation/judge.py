"""
SciLit - evaluation/judge.py
Part 4: human relevance judgments for the test queries (TREC-style pooling).

pool  : for each query, the top-10 papers of every search method (bm25, dense, hybrid,
        hybrid_rerank) are merged and SHUFFLED, so the annotator cannot tell which system found
        which paper (no bias toward our own method). Pooling is the standard TREC practice: nobody
        can judge 1,109 papers per query.
label : shows query + title + abstract, the annotator types 0 (not relevant), 1 (partly),
        2 (relevant). Saved after every answer; quit and resume any time.

extend: the ablation runs (French queries, fixed-size chunks) were not in the pool, so papers only
        they found were never judged and counted as not relevant: a bias against them. `extend`
        collects their top-10, keeps only the (query, paper) pairs never judged, and exports them
        with the SAME instructions; `import-extra` stores the answers in labels/qrels_chatgpt_extra.tsv.

Files: labels/queries.tsv (input), labels/pool.jsonl, labels/qrels.tsv (the human judgments)

Usage: python -m scilit.evaluation.judge pool
       python -m scilit.evaluation.judge export     # batches for an external LLM judge (ChatGPT)
       python -m scilit.evaluation.judge import     # its answers -> labels/qrels_chatgpt.tsv
       python -m scilit.evaluation.judge label 50   # human judgments on 50 random items (validation)
       python -m scilit.evaluation.judge extend       # unjudged pairs of the ablation runs -> ChatGPT
       python -m scilit.evaluation.judge import-extra
"""
from __future__ import annotations

import csv
import json
import random
import sys
import textwrap
from pathlib import Path

METHODS = ["bm25", "dense", "hybrid", "hybrid_rerank"]
# ablation runs: (name, index, method, query field); judged with `extend`
ABLATIONS = [("hybrid_rerank (FR)", "structure", "hybrid_rerank", "query_fr"),
             ("bm25, fixed-size chunks", "fixed", "bm25", "query_en"),
             ("bm25, fixed-size chunks + title", "fixed_title", "bm25", "query_en")]
EXTRA = Path("labels/qrels_chatgpt_extra.tsv")
DEPTH = 10
POOL, QRELS = Path("labels/pool.jsonl"), Path("labels/qrels.tsv")


def queries() -> list[dict]:
    rows = list(csv.DictReader(Path("labels/queries.tsv").open(encoding="utf-8"), delimiter="\t"))
    return [{"qid": f"q{i:02d}", **r} for i, r in enumerate(rows, 1)]


def pool() -> None:
    from scilit.retrieval.search import Searcher
    s = Searcher()
    abstracts = {a["id"]: a["abstract"] for a in map(json.loads, Path("data/raw/articles.jsonl").open(encoding="utf-8"))}
    rng = random.Random(0)
    with POOL.open("w", encoding="utf-8") as f:
        for q in queries():
            found = {}
            for m in METHODS:
                for r in s.search(q["query_en"], k=DEPTH, method=m):
                    found.setdefault(r["paper_id"], r["title"])
            items = list(found.items())
            rng.shuffle(items)
            for pid, title in items:
                f.write(json.dumps({"qid": q["qid"], "query": q["query_en"], "paper_id": pid,
                                    "title": title, "abstract": abstracts.get(pid, "")}, ensure_ascii=False) + "\n")
            print(f"{q['qid']}: {len(items)} papers to judge", flush=True)


INSTRUCTIONS = """You are judging search results for a literature review. For each numbered item, decide whether
the paper would belong in a literature review answering its QUERY, using only the title and abstract.
  2 = relevant (directly about the question)   1 = partly relevant (related, not central)   0 = not relevant
Reply with ONLY the {n} digits, in item order, separated by commas, e.g. 2,0,1,0,...
No other text. Exactly {n} digits.
"""


def export() -> None:
    items = [json.loads(l) for l in POOL.open(encoding="utf-8")]
    out = Path("labels/chatgpt"); out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("batch_*.txt"):              # remove files from the earlier batch format
        old.unlink()
    lines = [INSTRUCTIONS.format(n=len(items)), "ITEMS:"]
    for i, it in enumerate(items, 1):
        lines.append(f"\n{i}. QUERY: {it['query']}\nTITLE: {it['title']}\nABSTRACT: {it['abstract'][:300]}")
    (out / "all_items.txt").write_text("\n".join(lines), encoding="utf-8")
    (out / "map.json").write_text(json.dumps([[it["qid"], it["paper_id"]] for it in items]), encoding="utf-8")
    print(f"{len(items)} items -> {out / 'all_items.txt'}. Paste it into ChatGPT, then paste its reply "
          f"(the digits) into {out / 'answers.txt'}.")


def import_answers() -> None:
    import re
    d = Path("labels/chatgpt")
    mapping = json.loads((d / "map.json").read_text(encoding="utf-8"))
    digits = re.findall(r"[012]", (d / "answers.txt").read_text(encoding="utf-8"))
    if len(digits) != len(mapping):                  # any shift would mis-assign every later label
        raise SystemExit(f"Got {len(digits)} digits, expected {len(mapping)}: ask ChatGPT again for exactly "
                         f"{len(mapping)} comma-separated digits.")
    with Path("labels/qrels_chatgpt.tsv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        for (qid, pid), g in zip(mapping, digits):
            w.writerow([qid, pid, g])
    print(f"{len(digits)} judgments imported -> labels/qrels_chatgpt.tsv "
          f"(2: {digits.count('2')}, 1: {digits.count('1')}, 0: {digits.count('0')})")


def label(sample: int | None = None) -> None:
    done = set()
    if QRELS.exists():
        done = {(r[0], r[1]) for r in csv.reader(QRELS.open(encoding="utf-8"), delimiter="\t")}
    items = [json.loads(l) for l in POOL.open(encoding="utf-8")]
    if sample:                                       # fixed random subset, the same on every run
        items = random.Random(1).sample(items, min(sample, len(items)))
    todo = [it for it in items if (it["qid"], it["paper_id"]) not in done]
    print(f"{len(items) - len(todo)} done, {len(todo)} to go. Keys: 0 / 1 / 2, q = quit\n")
    with QRELS.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        for n, it in enumerate(todo, 1):
            print("=" * 80)
            print(f"[{n}/{len(todo)}] QUERY: {it['query']}\n")
            print("TITLE:", it["title"])
            print(textwrap.fill(it["abstract"][:900], 80))
            while (a := input("\n0 not relevant / 1 partly / 2 relevant > ").strip().lower()) not in {"0", "1", "2", "q"}:
                pass
            if a == "q":
                print("Saved. Run the same command to continue.")
                return
            w.writerow([it["qid"], it["paper_id"], a])
            f.flush()
    print("All judged.")


def extend() -> None:
    from scilit.retrieval.search import Searcher
    judged = {(r[0], r[1]) for f in (Path("labels/qrels_chatgpt.tsv"), EXTRA) if f.exists()
              for r in csv.reader(f.open(encoding="utf-8"), delimiter="\t")}
    arts = {a["id"]: a for a in map(json.loads, Path("data/raw/articles.jsonl").open(encoding="utf-8"))}
    searchers, items = {}, {}
    for name, mode, method, field in ABLATIONS:
        if not Path(f"data/index/{mode}/bm25.pkl").exists():
            print(f"skip {name}: no index data/index/{mode}/ (run chunk + build_index --mode {mode})")
            continue
        searchers.setdefault(mode, Searcher(mode=mode))
        for q in queries():
            for r in searchers[mode].search(q[field], k=DEPTH, method=method):
                if (q["qid"], r["paper_id"]) not in judged:
                    items[(q["qid"], r["paper_id"])] = {"qid": q["qid"], "query": q["query_en"], "paper_id": r["paper_id"],
                                                        "title": arts[r["paper_id"]]["title"], "abstract": arts[r["paper_id"]]["abstract"]}
        print(f"{name}: {len(items)} unjudged pairs so far", flush=True)
    items = list(items.values())
    random.Random(0).shuffle(items)
    out = Path("labels/chatgpt"); out.mkdir(parents=True, exist_ok=True)
    lines = [INSTRUCTIONS.format(n=len(items)), "ITEMS:"]
    for i, it in enumerate(items, 1):           # always the ENGLISH query: same task as the main pool
        lines.append(f"\n{i}. QUERY: {it['query']}\nTITLE: {it['title']}\nABSTRACT: {it['abstract'][:300]}")
    (out / "extra_items.txt").write_text("\n".join(lines), encoding="utf-8")
    (out / "extra_map.json").write_text(json.dumps([[it["qid"], it["paper_id"]] for it in items]), encoding="utf-8")
    print(f"{len(items)} new pairs -> {out / 'extra_items.txt'}. Paste into ChatGPT, put its digits in "
          f"{out / 'extra_answers.txt'}, then run: import-extra")


def import_extra() -> None:
    import re
    d = Path("labels/chatgpt")
    mapping = json.loads((d / "extra_map.json").read_text(encoding="utf-8"))
    digits = re.findall(r"[012]", (d / "extra_answers.txt").read_text(encoding="utf-8"))
    if len(digits) != len(mapping):
        raise SystemExit(f"Got {len(digits)} digits, expected {len(mapping)}.")
    with EXTRA.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter="\t").writerows([qid, pid, g] for (qid, pid), g in zip(mapping, digits))
    print(f"{len(digits)} judgments -> {EXTRA}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "label":
        label(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    else:
        {"pool": pool, "export": export, "import": import_answers, "extend": extend, "import-extra": import_extra}[cmd]()
