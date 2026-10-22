"""
SciLit - corpus/make_splits.py
Part 4 step 1: freeze the dev / test split BEFORE any tuning (subject rule 2.3.1).

  - unit = paper (two sentences of one paper on both sides would be leakage)
  - only structured papers (parse_pdf flag); 70% dev / 30% test, stratified by main theme
  - fixed seed; the file is written once and never overwritten (use --force only to rebuild
    on purpose, and say so in experiments.md)

Output: configs/splits.json (in git, small) + its SHA-256 printed for the report.

Usage: python -m scilit.corpus.make_splits
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SEED = 42
TEST_SHARE = 0.30


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--parsed", default="data/parsed")
    ap.add_argument("--out", default="configs/splits.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists: the split is frozen. Use --force only on purpose.")

    themes = {}
    for line in Path(args.raw, "articles.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            a = json.loads(line)
            themes[a["id"]] = a["themes"][0]
    structured, excluded = [], []
    for line in Path(args.parsed, "documents.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        (structured if d["structured"] else excluded).append(d["id"])

    by_theme = defaultdict(list)
    for pid in sorted(structured):                     # sorted: same input -> same split
        by_theme[themes[pid]].append(pid)
    rng = random.Random(SEED)
    dev, test = [], []
    for theme in sorted(by_theme):
        ids = by_theme[theme]
        rng.shuffle(ids)
        k = round(len(ids) * TEST_SHARE)
        test += ids[:k]
        dev += ids[k:]

    split = {"seed": SEED, "test_share": TEST_SHARE,
             "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "dev": sorted(dev), "test": sorted(test), "excluded_unstructured": sorted(excluded)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=1), encoding="utf-8")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"dev {len(dev)} / test {len(test)} papers (excluded unstructured: {len(excluded)})")
    for theme in sorted(by_theme):
        t = sum(1 for p in by_theme[theme] if p in set(test))
        print(f"  {theme:22} dev {len(by_theme[theme]) - t:4}  test {t:4}")
    print(f"Frozen -> {out}  sha256 {digest[:16]}  (write this hash in experiments.md)")


if __name__ == "__main__":
    main()
