"""
SciLit - evaluation/review_eval.py
Part 12b: evaluate the WHOLE review pipeline, stage by stage, as ONE controlled experiment.

The expensive step (search + LLM reading, ~110 s per question) runs ONCE per question set (run).
Every comparison is then replayed from those saved reading cards (replay, ~15 s per question): two
configurations differ in exactly the one thing being compared, never in what was retrieved or read.

Question sets
  test     labels/queries.tsv, 30 questions with relevance labels (retrieval, reader agreement).
           Some were inspected while designing the synthesis: "used during design" for synthesis
  heldout  labels/heldout.tsv, 12 questions nobody inspected before the final runs: the clean
           synthesis comparison. Needs no relevance labels (synthesis metrics come from the sources)

Commands
  run           search + read + synthesise every question of a set -> data/eval_runs/<tag>/qNN.json
  replay        from a run's saved cards: re-verify the claims with the current verifier, then
                synthesise again (--synthesis plan | single | template). No search, no reading.
                Replaying a run with its own settings must reproduce it: the check that replay is faithful
  report        per configuration:
                  reading    papers judged relevant; agreement with the relevance labels, 3-level
                             (relation 2/1/0 vs grade 2/1/0) and binary (relevant or partly vs not), kappa
                  claims     share of reading-card claims the verifier supports
                  synthesis  points planned, sentences, share supported, kept per review, share of the
                             kept sentences' word trigrams copied from their sources (a design that only
                             copies would score high on support while synthesising nothing), share of the
                             verified claims used, for the top 4 and the last 4 retrieved papers
                  cost       abstentions, latency of reading and of synthesis
  compare A B   the same questions under two configurations: mean difference in share supported and
                in kept sentences, paired bootstrap (10,000 resamples over questions)
  verify-set    the verifier test set for HUMAN labelling (labels/verify/), see build_verify_set
  verify-export / verify-import   the same items for ChatGPT (3 paste-sized parts) / its answers ->
                sheet_chatgpt.csv, kept apart from the human sheets and always reported as LLM labels
  verify-score  every verifier against the labels (--labels human | chatgpt), see verify_score

Usage
  python -m scilit.evaluation.review_eval run --tag test --set test
  python -m scilit.evaluation.review_eval run --tag heldout --set heldout
  python -m scilit.evaluation.review_eval replay --source heldout --tag heldout_single --synthesis single
  python -m scilit.evaluation.review_eval compare --tags heldout heldout_single
  python -m scilit.evaluation.review_eval report
  python -m scilit.evaluation.review_eval verify-set --source test
  python -m scilit.evaluation.review_eval verify-score --which deberta
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics as st
from pathlib import Path

from scilit.generation import verify as V

RUNS = Path("data/eval_runs")
VDIR = Path("labels/verify")
SETS = {"test": Path("labels/queries.tsv"), "heldout": Path("labels/heldout.tsv")}
QRELS = Path("labels/qrels_chatgpt.tsv")
GRADE = {"relevant": 2, "partly relevant": 1, "not relevant": 0}          # = judge.py grades
JUDGED = {"supported", "insufficient", "contradicted"}                     # outcomes decided by the verifier itself


def load_queries(name: str) -> list[dict]:
    rows = list(csv.DictReader(SETS[name].open(encoding="utf-8"), delimiter="\t"))
    return [{"qid": f"{'q' if name == 'test' else 'h'}{i:02d}", **r} for i, r in enumerate(rows, 1)]


def load_runs(tag: str) -> list[dict]:
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted((RUNS / tag).glob("[qh]*.json"))]


def config(tag: str) -> dict:
    return json.loads((RUNS / tag / "config.json").read_text(encoding="utf-8"))


def start(tag: str, cfg: dict) -> Path:
    """A run directory with its configuration; resuming with another configuration is refused."""
    out = RUNS / tag
    out.mkdir(parents=True, exist_ok=True)
    f = out / "config.json"
    if f.exists() and json.loads(f.read_text(encoding="utf-8")) != cfg:
        raise SystemExit(f"{out} was made with another configuration: {f.read_text()}. Use another --tag.")
    f.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return out


def save(out: Path, r: dict, tag: str) -> None:
    (out / f"{r['qid']}.json").write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
    print(f"{tag} {r['qid']}: {r['counts']['supported']}/{len(r['sentences'])} kept, {len(r['plan'])} points, "
          f"abstained={r['abstained']}, {sum(r['latency_s'].values()):.0f}s", flush=True)


# ------------------------------------------------------------------ run / replay

def run(tag: str, qset: str, generator: str, model: str, verifier: str, synthesis: str) -> None:
    from scilit.generation.review import run_review
    from scilit.retrieval.search import Searcher
    V.VERIFIER = verifier
    synthesis = synthesis if generator == "llm" else "template"
    out = start(tag, {"set": qset, "generator": generator, "model": model, "verifier": verifier,
                      "synthesis": synthesis, "replay_of": None})
    searcher = Searcher()
    for q in load_queries(qset):
        if not (out / f"{q['qid']}.json").exists():
            save(out, {**run_review(q["query_en"], searcher, generator, model, synthesis), "qid": q["qid"]}, tag)


def replay(source: str, tag: str, synthesis: str, model: str, verifier: str) -> None:
    from scilit.generation.review import reverify, write_review
    V.VERIFIER = verifier
    out = start(tag, {**config(source), "model": model, "verifier": verifier, "synthesis": synthesis, "replay_of": source})
    for r in load_runs(source):
        if (out / f"{r['qid']}.json").exists():
            continue
        if not all("draft" in c for c in r["cards"]):
            raise SystemExit(f"{source} was made before cards kept their evidence and draft: run it again.")
        new = write_review(r["query"], r["table"], [reverify(c) for c in r["cards"]], synthesis, model)
        new.update(qid=r["qid"], generator=r["generator"],
                   latency_s={"search": r["latency_s"]["search"], "reading": r["latency_s"]["reading"], **new["latency_s"]})
        save(out, new, tag)


# ------------------------------------------------------------------ report / compare

def kappa(a: list[int], b: list[int]) -> float:
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(a, b)) if len(set(a) | set(b)) > 1 else float("nan")


def trigrams(text: str) -> set[tuple]:
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}


def copied(s: dict, claims: dict) -> float:
    """Share of a sentence's word trigrams found in the source sentences behind its claims."""
    mine = trigrams(s["text"])
    src = trigrams(" ".join(t for c in s["cites"] if c in claims for t in claims[c]["source_text"]))
    return len(mine & src) / len(mine) if mine else 0.0


def supported_share(r: dict) -> float:
    return r["counts"]["supported"] / len(r["sentences"]) if r["sentences"] else 0.0


def usage(runs: list[dict], top: int | None) -> float:
    """Share of the verified claims cited by at least one synthesis sentence (kept or not), for the
    papers ranked < 4 (top=True), >= 4 (top=False) or all (None): shows whether the synthesis uses
    the evidence of every retrieved paper or mostly of the first ones."""
    used = total = 0
    for r in runs:
        rank = {c["paper_id"]: i for i, c in enumerate(r["cards"])}
        cited = {c for s in r["sentences"] for c in s["cites"]}
        for cid, c in r["claims"].items():
            if top is None or (rank.get(c["paper_id"], 0) < 4) == top:
                total += 1
                used += cid in cited
    return used / total if total else float("nan")


def report() -> None:
    qrels = {}
    if QRELS.exists():
        qrels = {(q, p): int(g) for q, p, g in csv.reader(QRELS.open(encoding="utf-8"), delimiter="\t")}
    rows = []
    for d in sorted(p for p in RUNS.iterdir() if p.is_dir() and (p / "config.json").exists()):
        runs = load_runs(d.name)
        if not runs:
            continue
        cfg = config(d.name)
        cards = [(r["qid"], c) for r in runs for c in r["cards"]]
        pairs = [(GRADE.get(c.get("relation"), 0), qrels[(q, c["paper_id"])]) for q, c in cards if (q, c["paper_id"]) in qrels]
        binary = [(int(a > 0), int(b > 0)) for a, b in pairs]
        claims = [c for _, card in cards for c in card["claims"]]
        sents = [s for r in runs for s in r["sentences"]]
        kept = [(s, r["claims"]) for r in runs for s in r["sentences"] if s["status"] == "supported"]
        rows.append({
            "config": d.name, "set": cfg.get("set", "test"), "synthesis": cfg.get("synthesis", "single+repair"),
            "verifier": cfg["verifier"], "replay_of": cfg.get("replay_of") or "-", "questions": len(runs),
            "relevant_per_q": st.mean(sum(c["relevant"] for c in r["cards"]) for r in runs),
            "reader_pairs": len(pairs),
            "reader_agree": sum(a == b for a, b in pairs) / len(pairs) if pairs else float("nan"),
            "reader_kappa": kappa([a for a, _ in pairs], [b for _, b in pairs]) if pairs else float("nan"),
            "reader_bin_agree": sum(a == b for a, b in binary) / len(binary) if binary else float("nan"),
            "reader_bin_kappa": kappa([a for a, _ in binary], [b for _, b in binary]) if binary else float("nan"),
            "claims": len(claims), "claims_supported": sum(c["status"] == "supported" for c in claims) / max(len(claims), 1),
            "points_per_q": st.mean(len(r.get("plan", [])) for r in runs),
            "sentences_per_q": len(sents) / len(runs),
            "supported": sum(s["status"] == "supported" for s in sents) / max(len(sents), 1),
            "kept_per_q": st.mean(r["counts"]["supported"] for r in runs),
            "copied_kept": st.mean(copied(s, c) for s, c in kept) if kept else float("nan"),
            "claims_used": usage(runs, None), "used_top4": usage(runs, True), "used_last4": usage(runs, False),
            "abstained": sum(r["abstained"] for r in runs), "thin_evidence": sum(bool(r.get("thin_evidence")) for r in runs),
            "reading_s": st.mean(r["latency_s"].get("search", 0) + r["latency_s"].get("reading", 0) for r in runs),
            "synthesis_s": st.mean(r["latency_s"].get("synthesis", 0) for r in runs)})
    Path("reports").mkdir(exist_ok=True)
    Path("reports/review_eval.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    table = lambda keys: ["| " + " | ".join(keys) + " |", "|" + " --- |" * len(keys)] + \
                         ["| " + " | ".join(fmt(r[k]) for k in keys) + " |" for r in rows]
    ident = ["config", "set", "synthesis", "verifier", "replay_of", "questions"]
    md = ["# Review pipeline evaluation (Part 12b)", "",
          f"Relevance labels: {QRELS} (ChatGPT, declared LLM labels; test set only).", "",
          "## Reading and claims", ""] + table(ident[:2] + ["relevant_per_q", "reader_pairs", "reader_agree", "reader_kappa",
                                                           "reader_bin_agree", "reader_bin_kappa", "claims", "claims_supported"]) + \
         ["", "## Synthesis", ""] + table(ident + ["points_per_q", "sentences_per_q", "supported", "kept_per_q", "copied_kept", "claims_used", "used_top4", "used_last4",
                                                  "abstained", "thin_evidence", "reading_s", "synthesis_s"])
    Path("reports/review_eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def compare(a: str, b: str, n: int = 10000) -> None:
    import numpy as np
    ra, rb = {r["qid"]: r for r in load_runs(a)}, {r["qid"]: r for r in load_runs(b)}
    qids = sorted(set(ra) & set(rb))
    if not qids:
        raise SystemExit("No question in common.")
    rng = np.random.default_rng(0)
    md = [f"# {a} vs {b}: {len(qids)} questions in common", "", "| metric | " + a + " | " + b + " | difference | p (one-sided) |",
          "| --- | --- | --- | --- | --- |"]
    for name, f in [("share supported", supported_share), ("kept sentences", lambda r: r["counts"]["supported"])]:
        x, y = np.array([f(ra[q]) for q in qids], float), np.array([f(rb[q]) for q in qids], float)
        d = x - y
        boots = rng.choice(d, (n, len(d))).mean(1)
        p = float((boots <= 0).mean()) if d.mean() > 0 else float((boots >= 0).mean())
        md.append(f"| {name} | {x.mean():.3f} | {y.mean():.3f} | {d.mean():+.3f} | {p:.4f} |")
    out = Path("reports") / f"compare_{a}_vs_{b}.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


# ------------------------------------------------------------------ verifier test set

HARD = {   # how a hard case is made wrong; "number" is made by rule, the others by Qwen
    "qualifier": "Remove one condition or qualifier from the sentence (for example \"for tables\", \"on small datasets\", "
                 "\"in low-resource settings\", \"up to\") so that it claims something more general. Change nothing else.",
    "entity": "Replace one method, model, dataset, language or metric named in the sentence with a different, plausible "
              "one. Change nothing else.",
    "reversed": "Reverse the finding of the sentence (for example improves -> hurts, outperforms -> underperforms, "
                "reduces -> increases), keeping the wording as close as possible.",
    "extra": "Keep the sentence as it is and add one short, plausible clause that the sources do NOT state.",
    "topic": "Write one new sentence on the same topic as the sources, stating a plausible finding that the sources do NOT state.",
}
HARD_PROMPT = """{instruction}

Sentence: {text}
Sources: {sources}

Return only the new sentence."""
HARD_PER_TYPE, REAL_PER_STRATUM = 10, 15
LABEL_HELP = """How to label (each person alone, without looking at the other sheet or discussing):
For every item, read the SOURCES, then the SENTENCE, and write in the "label" column:
  S  supported      every fact of the sentence (numbers, conditions, methods, what it applies to)
                    is stated or directly implied by the sources
  P  partly         the main point is there, but something is added, changed or generalised
  N  not supported  the sources do not say it
  C  contradicted   the sources say the opposite
Optional: a short note. Save as CSV (keep the file name)."""


def collect_pairs(tag: str) -> list[dict]:
    """Every sentence the verifier itself judged in a run, with the exact sources it was judged against."""
    out = []
    for r in load_runs(tag):
        for card in r["cards"]:
            for c in card["claims"]:
                if c["status"] in JUDGED:
                    src = [f"{card['context'].get(s, '')} {card['sentences'][s]}".strip() for s in c["sources"]]
                    out.append({"stage": "claim", "text": c["text"], "sources": src, "run_status": c["status"]})
        for s in r["sentences"]:
            if s["status"] in JUDGED and all(c in r["claims"] for c in s["cites"]):
                src = [t for c in s["cites"] for t in r["claims"][c]["source_text"]]
                out.append({"stage": "synthesis", "text": s["text"], "sources": src, "run_status": s["status"]})
    return list({(p["text"], tuple(p["sources"])): p for p in out}.values())


def wrong_number(text: str, rng: random.Random) -> tuple[str, str] | None:
    """Two claim numbers swapped ("21% to 4.5%" -> "4.5% to 21%": every number still in the sources,
    attached to the wrong thing), else one number changed."""
    spans = [m for m in V.NUMBER.finditer(text) if V.numbers(m.group(0))]
    if len(spans) >= 2 and spans[0].group(1) != spans[1].group(1):
        a, b = spans[0], spans[1]
        return (text[:a.start(1)] + b.group(1) + text[a.end(1):b.start(1)] + a.group(1) + text[b.end(1):],
                "number_swapped")
    if spans:
        m = spans[0]
        v = m.group(1)
        new = f"{float(v) * 1.3:.{len(v.split('.')[1])}f}" if "." in v else str(int(v) + max(1, round(int(v) * 0.3)))
        return text[:m.start(1)] + new + text[m.end(1):], "number_changed"
    return None


def build_verify_set(source: str, model: str, force: bool) -> None:
    """real: REAL_PER_STRATUM pairs per (claim | synthesis) x (accepted | rejected by the verifier), so
    both errors can be measured. hard: HARD_PER_TYPE per type, made from ACCEPTED sentences, so each
    has a known way to be wrong (the labellers still decide: a perturbation can fail). Shuffled; the
    sheets show neither the verifier's decision nor the type; half the items calibrate, half test."""
    from scilit.generation.llm import ollama
    if (VDIR / "items.jsonl").exists() and not force:
        raise SystemExit(f"{VDIR / 'items.jsonl'} exists (labels may refer to it). Use --force to rebuild.")
    rng = random.Random(0)
    pool = collect_pairs(source)
    rng.shuffle(pool)
    items = []
    for stage in ("claim", "synthesis"):
        for accepted in (True, False):
            group = [p for p in pool if p["stage"] == stage and (p["run_status"] == "supported") == accepted]
            items += [{**p, "origin": "real", "type": "-"} for p in group[:REAL_PER_STRATUM]]
    used = {p["text"] for p in items}
    base = [p for p in pool if p["run_status"] == "supported" and p["text"] not in used]
    with_numbers = [p for p in base if V.numbers(p["text"])]
    for p in with_numbers[:HARD_PER_TYPE]:
        text, kind = wrong_number(p["text"], rng)
        items.append({**p, "text": text, "origin": "hard", "type": kind, "made_from": p["text"]})
        used.add(p["text"])
    base = [p for p in base if p["text"] not in used]
    k = 0
    for kind, instruction in HARD.items():
        made = 0
        while made < HARD_PER_TYPE and k < len(base):
            p, k = base[k], k + 1
            new = ollama(HARD_PROMPT.format(instruction=instruction, text=p["text"], sources=" ".join(p["sources"])),
                         {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}}, model)["text"].strip()
            if new and V.fold(new) != V.fold(p["text"]):
                items.append({**p, "text": new, "origin": "hard", "type": kind, "made_from": p["text"]})
                made += 1
        print(f"hard {kind}: {made}", flush=True)
    rng.shuffle(items)
    VDIR.mkdir(parents=True, exist_ok=True)
    with (VDIR / "items.jsonl").open("w", encoding="utf-8") as f:
        for i, it in enumerate(items, 1):
            f.write(json.dumps({"item": i, "split": "calibrate" if i % 2 else "test", **it}, ensure_ascii=False) + "\n")
    for who in "AB":
        with (VDIR / f"sheet_{who}.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["item", "sources", "sentence", "label", "note"])
            w.writerows([i, " || ".join(it["sources"]), it["text"], "", ""] for i, it in enumerate(items, 1))
    (VDIR / "HOW_TO_LABEL.txt").write_text(LABEL_HELP + "\n", encoding="utf-8")
    real = sum(it["origin"] == "real" for it in items)
    print(f"{len(items)} items ({real} real, {len(items) - real} hard) -> {VDIR}/sheet_A.csv and sheet_B.csv")


def read_sheet(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    out = {}
    for row in csv.DictReader(path.open(encoding="utf-8-sig")):
        lab = (row.get("label") or "").strip().upper()[:1]
        if lab in "SPNC" and lab:
            out[int(row["item"])] = lab
    return out


GPT_PROMPT = """You are checking sentences from a literature review against their sources.
For each item, read the SOURCES, then the SENTENCE, and answer with one letter:
  S = supported: every fact in the sentence (numbers, conditions, methods, what it applies to) is
      stated or directly implied by the sources
  P = partly supported: the main point is there, but something is added, changed or generalised
  N = not supported: the sources do not say it
  C = contradicted: the sources say the opposite
Judge each item on its own. Reply with ONLY one line per item:  <item number>: <letter>
"""
GPT_PARTS = 3


def export_chatgpt() -> None:
    """The same items as the human sheets, in GPT_PARTS files small enough to paste into ChatGPT.
    Its labels are LLM labels: kept in their own sheet, always reported as such."""
    items = [json.loads(l) for l in (VDIR / "items.jsonl").open(encoding="utf-8")]
    size = -(-len(items) // GPT_PARTS)
    for k in range(GPT_PARTS):
        part = items[k * size:(k + 1) * size]
        body = "\n\n".join(f"ITEM {it['item']}\nSOURCES: {' || '.join(it['sources'])}\nSENTENCE: {it['text']}" for it in part)
        (VDIR / f"chatgpt_part{k + 1}.txt").write_text(GPT_PROMPT + "\n" + body + "\n", encoding="utf-8")
    print(f"{len(items)} items -> {VDIR}/chatgpt_part1..{GPT_PARTS}.txt. Paste each into ChatGPT, put ALL its "
          f"answers in {VDIR}/chatgpt_answers.txt, then run: verify-import")


def import_chatgpt() -> None:
    n = sum(1 for _ in (VDIR / "items.jsonl").open(encoding="utf-8"))
    found = dict(re.findall(r"(?im)^\W*(?:item\s*)?(\d+)\**\s*[:=.)\-–]\s*\**([SPNC])\b", (VDIR / "chatgpt_answers.txt").read_text(encoding="utf-8")))
    with (VDIR / "sheet_chatgpt.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["item", "label"])
        w.writerows([int(i), found[i].upper()] for i in sorted(found, key=int) if 1 <= int(i) <= n)
    missing = [i for i in range(1, n + 1) if str(i) not in found]
    print(f"{n - len(missing)} of {n} ChatGPT labels imported -> {VDIR}/sheet_chatgpt.csv" +
          (f"; missing items: {missing}" if missing else ""))


def rates(gold: list[int], pred: list[int]) -> dict:
    """1 = supported. False acceptance = a not-supported sentence the verifier lets into the review."""
    tp = sum(g and p for g, p in zip(gold, pred)); fp = sum(not g and p for g, p in zip(gold, pred))
    fn = sum(g and not p for g, p in zip(gold, pred)); tn = sum(not g and not p for g, p in zip(gold, pred))
    far, frr = fp / max(fp + tn, 1), fn / max(fn + tp, 1)
    return {"n": len(gold), "supported_gold": tp + fn, "false_acceptance": round(far, 3), "false_rejection": round(frr, 3),
            "balanced_accuracy": round(1 - (far + frr) / 2, 3), "precision_supported": round(tp / max(tp + fp, 1), 3),
            "false_accepted": fp, "false_rejected": fn}


def verify_score(which: list[str], labels: str = "human") -> None:
    """Gold = the label both labellers gave (items where they disagree are left out and counted);
    binary: S = supported, P / N / C = not. DeBERTa's entailment threshold is chosen on the
    "calibrate" half (best balanced accuracy) and every number is reported on the "test" half.
    labels="human": sheets A and B; labels="chatgpt": ChatGPT's sheet, reported as LLM labels."""
    items = {it["item"]: it for it in map(json.loads, (VDIR / "items.jsonl").open(encoding="utf-8"))}
    names = ["A", "B"] if labels == "human" else ["chatgpt"]
    sheets = {w: read_sheet(VDIR / f"sheet_{w}.csv") for w in names}
    sheets = {w: s for w, s in sheets.items() if s}
    if not sheets:
        raise SystemExit(f"No labels yet in {VDIR}: " + ", ".join(f"sheet_{w}.csv" for w in names))
    res: dict = {"labels": "human" if labels == "human" else "ChatGPT (LLM labels)", "labellers": sorted(sheets)}
    if len(sheets) == 2:
        common = sorted(set(sheets["A"]) & set(sheets["B"]))
        res["kappa_4_labels"] = round(kappa([ "SPNC".index(sheets["A"][i]) for i in common],
                                            ["SPNC".index(sheets["B"][i]) for i in common]), 3)
        res["kappa_binary"] = round(kappa([int(sheets["A"][i] == "S") for i in common],
                                          [int(sheets["B"][i] == "S") for i in common]), 3)
        res["disagreements"] = sum(sheets["A"][i] != sheets["B"][i] for i in common)
    gold = {}
    for i in items:
        labs = {s[i] for s in sheets.values() if i in s}
        if len(labs) == 1 and all(i in s for s in sheets.values()):
            gold[i] = int(labs.pop() == "S")
    res["gold_items"] = len(gold)
    gpt = read_sheet(VDIR / "sheet_chatgpt.csv") if labels == "human" else {}
    if gpt:
        common = sorted(set(gold) & set(gpt))
        res["chatgpt_vs_human_kappa_binary"] = round(kappa([gold[i] for i in common], [int(gpt[i] == "S") for i in common]), 3)
    split = {s: [i for i in sorted(gold) if items[i]["split"] == s] for s in ("calibrate", "test")}
    md = [f"# Verifier evaluation (Part 11): {res['labels']} labels", "",
          f"{len(items)} items (real pairs from the runs + hard cases with a known way to be wrong), labelled by "
          f"{'ChatGPT' if labels == 'chatgpt' else f'{len(sheets)} person(s)'}; gold = agreed labels ({len(gold)} items). Reported on the test half "
          f"({len(split['test'])} items)."]
    if "chatgpt_vs_human_kappa_binary" in res:
        md += [f"ChatGPT vs human gold: kappa {res['chatgpt_vs_human_kappa_binary']} (supported / not)."]
    if "kappa_binary" in res:
        md += [f"Agreement between labellers: kappa {res['kappa_binary']} (supported / not), {res['kappa_4_labels']} (4 labels); "
               f"{res['disagreements']} disagreements left out."]
    md += ["", "| verifier | threshold | n | false acceptance | false rejection | balanced acc. | precision (supported) |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    per_type = {}
    for name in which:
        out = {}
        for i in gold:
            it = items[i]
            status, checks = V.check(it["text"], it["sources"], set(), set(), name)
            out[i] = (status, checks.get("entailment"))
        accept = lambda i, t: out[i][0] in JUDGED and (out[i][1] >= t if out[i][1] is not None else out[i][0] == "supported")
        thresholds = [0.5]
        if name == "deberta":
            grid = [round(x * 0.05, 2) for x in range(1, 20)]
            best = max(grid, key=lambda t: rates([gold[i] for i in split["calibrate"]],
                                                  [int(accept(i, t)) for i in split["calibrate"]])["balanced_accuracy"])
            thresholds = sorted({0.5, best})
            res[f"{name}_calibrated_threshold"] = best
        for t in thresholds:
            r = rates([gold[i] for i in split["test"]], [int(accept(i, t)) for i in split["test"]])
            res[f"{name}@{t}"] = r
            md.append(f"| {name} | {t if name == 'deberta' else '-'} | {r['n']} | {r['false_acceptance']} | "
                      f"{r['false_rejection']} | {r['balanced_accuracy']} | {r['precision_supported']} |")
        for kind in sorted({it["type"] for it in items.values() if it["origin"] == "hard"}):
            wrong = [i for i in gold if items[i]["type"] == kind and not gold[i]]
            per_type.setdefault(kind, {})[name] = (sum(accept(i, 0.5) for i in wrong), len(wrong))
        print(f"{name}: done", flush=True)
    res["hard_accepted"] = {k: {n: f"{a}/{b}" for n, (a, b) in v.items()} for k, v in per_type.items()}
    md += ["", "Hard cases the labellers judged NOT supported, accepted anyway (all items, threshold 0.5):", "",
           "| type | " + " | ".join(which) + " |", "|" + " --- |" * (len(which) + 1)]
    md += [f"| {k} | " + " | ".join(f"{v[n][0]}/{v[n][1]}" for n in which) + " |" for k, v in sorted(per_type.items())]
    Path("reports").mkdir(exist_ok=True)
    Path(f"reports/verifier_eval_{labels}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    Path(f"reports/verifier_eval_{labels}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main() -> None:
    from scilit.generation.llm import MODEL
    from scilit.generation.review import SYNTHESES
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "replay", "report", "compare", "verify-set", "verify-export", "verify-import",
                                    "verify-score"])
    ap.add_argument("--tag")
    ap.add_argument("--set", choices=list(SETS), default="test")
    ap.add_argument("--source", help="run whose saved cards are reused (replay, verify-set)")
    ap.add_argument("--tags", nargs=2, help="the two configurations to compare")
    ap.add_argument("--generator", choices=["llm", "template"], default="llm")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--verifier", choices=["deberta", "minicheck"], default=V.VERIFIER)
    ap.add_argument("--synthesis", choices=SYNTHESES, default="plan")
    ap.add_argument("--which", nargs="+", default=["deberta", "minicheck"], help="verifiers to score")
    ap.add_argument("--labels", choices=["human", "chatgpt"], default="human", help="labels verify-score uses")
    ap.add_argument("--force", action="store_true", help="rebuild the verifier set (labels will no longer match)")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.tag or a.set, a.set, a.generator, a.model if a.generator == "llm" else None, a.verifier, a.synthesis)
    elif a.cmd == "replay":
        replay(a.source, a.tag, a.synthesis, a.model, a.verifier)
    elif a.cmd == "report":
        report()
    elif a.cmd == "compare":
        compare(*a.tags)
    elif a.cmd == "verify-set":
        build_verify_set(a.source, a.model, a.force)
    elif a.cmd == "verify-export":
        export_chatgpt()
    elif a.cmd == "verify-import":
        import_chatgpt()
    else:
        verify_score(a.which, a.labels)


if __name__ == "__main__":
    main()
