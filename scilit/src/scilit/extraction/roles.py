"""
SciLit - extraction/roles.py
Part 7 step 1: sentence-role classifier (background / objective / method / result / other).

Gold data: CSAbstruct (Cohan et al., 2019) - computer-science abstracts, every sentence labelled
by humans. Public, so no AI-made labels (subject rule 2.1). Official train / dev / test split,
DEDUPLICATED (dedupe below): the official splits share sentences (172 of the 1,349 test sentences,
12.7%, also appear in train), which would measure memory instead of generalisation.

Three systems, same test set, macro-F1 (rare roles count as much as frequent ones), plus micro-F1
to compare with the published 83.1 (which is micro-F1, with the whole abstract as context):
  rules    cue phrases ("we propose" -> objective, "outperforms" -> result, ...)   [Lot B]
  tfidf    TF-IDF word 1-2 grams + logistic regression                            [classical]
  scibert  allenai/scibert_scivocab_uncased fine-tuned, 3 seeds, mean +- std      [learned]
plus a paired bootstrap (1,000 resamples) SciBERT vs TF-IDF on the same test sentences, for every seed.
The saved model is the seed with the best DEV score: the test set is never used to choose anything.

Output: models/roles_scibert/ (seed chosen on dev), reports/roles_results.json

Usage: python -m scilit.extraction.roles
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

URL = "https://raw.githubusercontent.com/allenai/sequential_sentence_classification/master/data/CSAbstruct/{}.jsonl"
CACHE = Path("data/external/csabstruct")
LABELS = ["background", "objective", "method", "result", "other"]
BERT = "allenai/scibert_scivocab_uncased"
SEEDS = [13, 42, 2024]

RULES = [  # first match wins; order = most specific first
    ("objective", r"\b(we (propose|introduce|present|investigate|study|explore|address|aim)|this (paper|work|study)|in this (paper|work)|our (goal|aim))\b"),
    ("result", r"\b(outperform|achiev|improv|results? (show|demonstrat|indicat)|significantly|state-of-the-art|accuracy|f1|we (show|find|observe|demonstrate))"),
    ("method", r"\b(we (use|train|apply|employ|design|fine-tune|build|develop|model|formulate)|our (model|method|approach|framework)|consist|based on|architecture)\b"),
]


def load(split: str) -> tuple[list[str], list[str]]:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{split}.jsonl"
    if not f.exists():
        r = requests.get(URL.format(split), timeout=60)
        r.raise_for_status()
        f.write_text(r.text, encoding="utf-8")
    sents, labels = [], []
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ex = json.loads(line)
            sents += ex["sentences"]
            labels += [l.lower() for l in ex["labels"]]
    return sents, labels


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def dedupe(train, dev, test) -> tuple[tuple, tuple, dict]:
    """Remove from dev every sentence seen in train, and from test every sentence seen in train or
    dev: every reported score is then on text the model has never seen. Train is unchanged."""
    seen = {norm(x) for x in train[0]}

    def keep(split):
        idx = [i for i, x in enumerate(split[0]) if norm(x) not in seen]
        return [split[0][i] for i in idx], [split[1][i] for i in idx]
    dev2 = keep(dev)
    seen |= {norm(x) for x in dev[0]}
    test2 = keep(test)
    info = {"split": "official, deduplicated", "dev_removed": len(dev[0]) - len(dev2[0]),
            "test_removed": len(test[0]) - len(test2[0]), "dev": len(dev2[0]), "test": len(test2[0])}
    print(f"dedupe: removed {info['dev_removed']} dev and {info['test_removed']} test sentences already seen", flush=True)
    return dev2, test2, info


def rules_predict(sents: list[str]) -> list[str]:
    out = []
    for s in sents:
        low = s.lower()
        out.append(next((lab for lab, pat in RULES if re.search(pat, low)), "background"))
    return out


def macro_f1(gold, pred) -> float:
    return f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0)


def micro_f1(gold, pred) -> float:
    """= accuracy for one label per sentence; the metric the CSAbstruct paper reports."""
    return f1_score(gold, pred, labels=LABELS, average="micro", zero_division=0)


def paired_bootstrap(gold, a, b, n=1000, seed=0) -> float:
    """Share of resamples where system a does NOT beat system b (a one-sided p-value)."""
    rng = np.random.default_rng(seed)
    gold, a, b = map(np.array, (gold, a, b))
    worse = 0
    for _ in range(n):
        i = rng.integers(0, len(gold), len(gold))
        worse += macro_f1(gold[i], a[i]) <= macro_f1(gold[i], b[i])
    return worse / n


def train_scibert(train, dev, test, seed: int) -> dict:
    """Fine-tune SciBERT with one seed. The best EPOCH is chosen on dev. Returns the dev and test
    scores, the test predictions and the model (moved to CPU, so the caller can keep the best one)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dev_ = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BERT)
    model = AutoModelForSequenceClassification.from_pretrained(BERT, num_labels=len(LABELS)).to(dev_)
    lab = {l: i for i, l in enumerate(LABELS)}
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler(enabled=dev_ == "cuda")

    def batches(x, y, bs, shuffle):
        idx = list(range(len(x)))
        if shuffle:
            random.shuffle(idx)
        for k in range(0, len(idx), bs):
            j = idx[k:k + bs]
            enc = tok([x[t] for t in j], truncation=True, max_length=96, padding=True, return_tensors="pt").to(dev_)
            yield enc, torch.tensor([lab[y[t]] for t in j], device=dev_)

    def predict(x):
        model.eval(); preds = []
        with torch.no_grad(), torch.autocast(dev_, enabled=dev_ == "cuda"):
            for enc, _ in batches(x, [LABELS[0]] * len(x), 64, False):
                preds += model(**enc).logits.argmax(-1).tolist()
        return [LABELS[p] for p in preds]

    best_f1, best_state = -1.0, None
    for epoch in range(3):                         # epoch selection on DEV, never on test
        model.train()
        for enc, y in batches(*train, 32, True):
            with torch.autocast(dev_, enabled=dev_ == "cuda"):
                loss = torch.nn.functional.cross_entropy(model(**enc).logits, y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        f1 = macro_f1(dev[1], predict(dev[0]))
        print(f"  seed {seed} epoch {epoch + 1}: dev macro-F1 {f1:.3f}", flush=True)
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    test_pred = predict(test[0])
    return {"seed": seed, "dev_f1": best_f1, "test_f1": macro_f1(test[1], test_pred),
            "test_micro": micro_f1(test[1], test_pred), "test_pred": test_pred, "model": model.cpu(), "tok": tok}


def run_seeds(train, dev, test, baseline_pred: list[str], out_dir: Path) -> dict:
    """3 seeds. Reported: test mean +- std over ALL seeds. Saved model: the seed with the best DEV
    score (choosing on test would be selection on the test set). Paired bootstrap against the
    baseline for EVERY seed; the largest p is reported (the conservative one)."""
    import torch
    runs, best = [], None
    for seed in SEEDS:
        r = train_scibert(train, dev, test, seed)
        print(f"  seed {seed}: dev {r['dev_f1']:.3f} | test macro-F1 {r['test_f1']:.3f}", flush=True)
        r["p"] = paired_bootstrap(test[1], r["test_pred"], baseline_pred)
        if best is None or r["dev_f1"] > best["dev_f1"]:
            best = r
        runs.append(r)
        torch.cuda.empty_cache()
    out_dir.mkdir(parents=True, exist_ok=True)
    best["model"].save_pretrained(out_dir); best["tok"].save_pretrained(out_dir)
    test_f1 = [r["test_f1"] for r in runs]
    micro = [r["test_micro"] for r in runs]
    return {"scibert_mean": float(np.mean(test_f1)), "scibert_std": float(np.std(test_f1)),
            "scibert_micro_mean": float(np.mean(micro)), "scibert_micro_std": float(np.std(micro)),
            "scibert_seeds": {str(r["seed"]): {"dev": round(r["dev_f1"], 4), "test": round(r["test_f1"], 4),
                                               "bootstrap_p": r["p"]} for r in runs},
            "saved_seed": best["seed"], "saved_by": "best dev macro-F1",
            "bootstrap_p_scibert_vs_baseline_max": max(r["p"] for r in runs),
            "per_label_saved_model": dict(zip(LABELS, f1_score(test[1], best["test_pred"], labels=LABELS,
                                                               average=None, zero_division=0).round(3).tolist()))}


def main() -> None:
    train, dev, test = load("train"), load("dev"), load("test")
    print(f"CSAbstruct sentences (official): train {len(train[0])}, dev {len(dev[0])}, test {len(test[0])}")
    dev, test, info = dedupe(train, dev, test)
    res = {"data": info, "rules": macro_f1(test[1], rules_predict(test[0]))}
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    lr = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    lr.fit(vec.fit_transform(train[0]), train[1])
    tfidf_pred = lr.predict(vec.transform(test[0])).tolist()
    res["tfidf"] = macro_f1(test[1], tfidf_pred)
    res["tfidf_micro"] = micro_f1(test[1], tfidf_pred)
    print(f"rules {res['rules']:.3f} | tfidf {res['tfidf']:.3f}", flush=True)
    res.update(run_seeds(train, dev, test, tfidf_pred, Path("models/roles_scibert")))
    res["published"] = ("83.1 micro-F1, Cohan et al. 2019, a model that reads the whole abstract at once; "
                        "ours classifies each sentence alone")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/roles_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"rules {res['rules']:.3f} | tfidf {res['tfidf']:.3f} | scibert {res['scibert_mean']:.3f} "
          f"+- {res['scibert_std']:.3f} (micro {res['scibert_micro_mean']:.3f}) | saved seed {res['saved_seed']} "
          f"(best dev) | bootstrap p max = {res['bootstrap_p_scibert_vs_baseline_max']:.3f}")
    print("-> reports/roles_results.json, models/roles_scibert/")


if __name__ == "__main__":
    main()
