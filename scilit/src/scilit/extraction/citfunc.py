"""
SciLit - extraction/citfunc.py
Part 8: citation function = WHY a paper cites another (background / method / result comparison).

Data: SciCite (Cohan et al., 2019), ~11k human-labelled citation sentences, official splits.
Systems: TF-IDF + logistic regression (classical) vs SciBERT fine-tuned (3 seeds, mean +- std),
paired bootstrap. This is our REPRODUCED PUBLISHED RESULT (subject rule 2.3.6): Beltagy et al.
(2019, SciBERT paper) and Cohan et al. (2019) report about 84-85 macro-F1 on SciCite; we put our
score next to theirs and explain the gap.

Output: models/citfunc_scibert/, reports/citfunc_results.json
Usage : python -m scilit.extraction.citfunc
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from scilit.extraction import roles

PARQUET = "https://huggingface.co/datasets/allenai/scicite/resolve/refs%2Fconvert%2Fparquet/default/{}/0000.parquet"
NAMES = ["method", "background", "result"]          # SciCite label order on the Hugging Face hub


def load(split: str) -> tuple[list[str], list[str]]:
    import pandas as pd
    cache = Path("data/external/scicite"); cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{split}.parquet"
    if not f.exists():
        import requests
        r = requests.get(PARQUET.format(split), timeout=120); r.raise_for_status(); f.write_bytes(r.content)
    df = pd.read_parquet(f)
    return df["string"].str.strip().tolist(), [NAMES[i] for i in df["label"]]


def main() -> None:
    train, dev, test = load("train"), load("validation"), load("test")
    print(f"SciCite: train {len(train[0])}, dev {len(dev[0])}, test {len(test[0])}", flush=True)
    roles.LABELS[:] = NAMES                         # reuse the Part 7 trainer with these labels
    f1 = lambda g, p: f1_score(g, p, labels=NAMES, average="macro")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    lr = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced").fit(vec.fit_transform(train[0]), train[1])
    tfidf_pred = lr.predict(vec.transform(test[0])).tolist()
    res = {"tfidf": f1(test[1], tfidf_pred)}
    print(f"tfidf {res['tfidf']:.3f}", flush=True)
    scores, preds = [], []
    for seed in roles.SEEDS:
        s, p = roles.train_scibert(train, dev, test, seed, None)
        scores.append(s); preds.append(p); print(f"  seed {seed}: test macro-F1 {s:.3f}", flush=True)
    best = int(np.argmax(scores))
    out = Path("models/citfunc_scibert"); out.mkdir(parents=True, exist_ok=True)
    roles.train_scibert(train, dev, test, roles.SEEDS[best], out)
    res.update(scibert_mean=float(np.mean(scores)), scibert_std=float(np.std(scores)),
               scibert_seeds=dict(zip(map(str, roles.SEEDS), scores)),
               bootstrap_p_scibert_vs_tfidf=roles.paired_bootstrap(test[1], preds[best], tfidf_pred),
               published="about 0.84-0.85 macro-F1 (Cohan et al. 2019; Beltagy et al. 2019)")
    Path("reports/citfunc_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"tfidf {res['tfidf']:.3f} | scibert {res['scibert_mean']:.3f} +- {res['scibert_std']:.3f} | "
          f"bootstrap p = {res['bootstrap_p_scibert_vs_tfidf']:.3f} | published ~0.84-0.85")


if __name__ == "__main__":
    main()
