"""
SciLit - extraction/roles.py
Part 7 step 1: sentence-role classifier (background / objective / method / result / other).

Gold data: CSAbstruct (Cohan et al., 2019) - computer-science abstracts, every sentence labelled
by humans. Public, so no AI-made labels (subject rule 2.1). Official train / dev / test split.

Three systems, same test set, macro-F1 (rare roles count as much as frequent ones):
  rules    cue phrases ("we propose" -> objective, "outperforms" -> result, ...)   [Lot B]
  tfidf    TF-IDF word 1-2 grams + logistic regression                            [classical]
  scibert  allenai/scibert_scivocab_uncased fine-tuned, 3 seeds, mean +- std      [learned]
plus a paired bootstrap (1,000 resamples) SciBERT vs TF-IDF on the same test sentences.

Output: models/roles_scibert/ (best seed), reports/roles_results.json

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


def rules_predict(sents: list[str]) -> list[str]:
    out = []
    for s in sents:
        low = s.lower()
        out.append(next((lab for lab, pat in RULES if re.search(pat, low)), "background"))
    return out


def macro_f1(gold, pred) -> float:
    return f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0)


def paired_bootstrap(gold, a, b, n=1000, seed=0) -> float:
    """Share of resamples where system a does NOT beat system b (a one-sided p-value)."""
    rng = np.random.default_rng(seed)
    gold, a, b = map(np.array, (gold, a, b))
    worse = 0
    for _ in range(n):
        i = rng.integers(0, len(gold), len(gold))
        worse += macro_f1(gold[i], a[i]) <= macro_f1(gold[i], b[i])
    return worse / n


def train_scibert(train, dev, test, seed: int, out_dir: Path | None):
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
            for enc, _ in batches(x, ["other"] * len(x), 64, False):
                preds += model(**enc).logits.argmax(-1).tolist()
        return [LABELS[p] for p in preds]

    best_f1, best_state = -1.0, None
    for epoch in range(3):                         # model selection on DEV, never on test
        model.train()
        for enc, y in batches(*train, 32, True):
            with torch.autocast(dev_, enabled=dev_ == "cuda"):
                loss = torch.nn.functional.cross_entropy(model(**enc).logits, y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        f1 = macro_f1(dev[1], predict(dev[0]))
        print(f"  seed {seed} epoch {epoch + 1}: dev macro-F1 {f1:.3f}", flush=True)
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    test_pred = predict(test[0])
    if out_dir:
        model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    return macro_f1(test[1], test_pred), test_pred


def main() -> None:
    train, dev, test = load("train"), load("dev"), load("test")
    print(f"CSAbstruct sentences: train {len(train[0])}, dev {len(dev[0])}, test {len(test[0])}")
    res = {}
    res["rules"] = macro_f1(test[1], rules_predict(test[0]))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    lr = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    lr.fit(vec.fit_transform(train[0]), train[1])
    tfidf_pred = lr.predict(vec.transform(test[0])).tolist()
    res["tfidf"] = macro_f1(test[1], tfidf_pred)
    print(f"rules {res['rules']:.3f} | tfidf {res['tfidf']:.3f}", flush=True)

    scores, preds = [], []
    for seed in SEEDS:
        f1, p = train_scibert(train, dev, test, seed, None)
        scores.append(f1); preds.append(p)
        print(f"  seed {seed}: test macro-F1 {f1:.3f}", flush=True)
    best = int(np.argmax(scores))
    out = Path("models/roles_scibert")
    out.mkdir(parents=True, exist_ok=True)
    train_scibert(train, dev, test, SEEDS[best], out)        # retrain best seed and save it
    res["scibert_mean"], res["scibert_std"] = float(np.mean(scores)), float(np.std(scores))
    res["scibert_seeds"] = dict(zip(map(str, SEEDS), scores))
    res["bootstrap_p_scibert_vs_tfidf"] = paired_bootstrap(test[1], preds[best], tfidf_pred)
    res["per_label_scibert"] = dict(zip(LABELS, f1_score(test[1], preds[best], labels=LABELS, average=None, zero_division=0).round(3).tolist()))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/roles_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"rules {res['rules']:.3f} | tfidf {res['tfidf']:.3f} | scibert {res['scibert_mean']:.3f} "
          f"+- {res['scibert_std']:.3f} | bootstrap p = {res['bootstrap_p_scibert_vs_tfidf']:.3f}")
    print("-> reports/roles_results.json, models/roles_scibert/")


if __name__ == "__main__":
    main()
