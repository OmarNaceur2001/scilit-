"""
SciLit - retrieval/build_index.py
Part 5 steps 2-3: the two search indexes over the chunks.

bm25  : BM25Okapi (k1=1.5, b=0.75) on header + text. Tokens keep numbers, versions and
        hyphenated names ("7b", "gpt-3.5", "bert-base", "f1"): exact matches embeddings blur.
dense : sentence embeddings of header + text, L2-normalized, in a FAISS exact inner-product
        index (= cosine similarity; exact search is milliseconds at ~50k vectors).
        Main model BAAI/bge-m3 (multilingual: French / Arabic queries need no translation);
        BAAI/bge-small-en-v1.5 is the comparison baseline.

Input : data/index/chunks_<mode>.jsonl
Output: data/index/<mode>/bm25.pkl, data/index/<mode>/<model>/faiss.index + ids.json

Usage: python -m scilit.retrieval.build_index --bm25
       python -m scilit.retrieval.build_index --dense BAAI/bge-m3
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
STOP = set("""a an the of in on to and or for is are was were be been with by from that this these
those it its as at we our us they their not no can may also than then which who what when where how
into over under such via""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


def load(chunks_path: Path) -> list[dict]:
    return [json.loads(l) for l in chunks_path.open(encoding="utf-8")]


def build_bm25(chunks: list[dict], out: Path) -> None:
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([tokenize(c["header"] + " " + c["text"]) for c in chunks], k1=1.5, b=0.75)
    with (out / "bm25.pkl").open("wb") as f:
        pickle.dump({"bm25": bm25, "ids": [c["id"] for c in chunks]}, f)
    print(f"BM25: {len(chunks)} chunks -> {out / 'bm25.pkl'}")


def build_dense(chunks: list[dict], out: Path, model_name: str, batch: int) -> None:
    import faiss
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = 512                      # chunks are <= ~300 tokens; caps memory
    if device == "cuda":
        model.half()                                # fp16: half the memory, same ranking quality
    print(f"Encoding {len(chunks)} chunks with {model_name} on {device}...", flush=True)
    emb = model.encode([c["header"] + "\n" + c["text"] for c in chunks], batch_size=batch,
                       normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
    emb = np.asarray(emb, dtype="float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    d = out / model_name.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(d / "faiss.index"))
    (d / "ids.json").write_text(json.dumps({"model": model_name, "dim": int(emb.shape[1]),
                                            "ids": [c["id"] for c in chunks]}), encoding="utf-8")
    print(f"Dense: {index.ntotal} vectors, dim {emb.shape[1]} -> {d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="structure", choices=["structure", "fixed", "fixed_title"])
    ap.add_argument("--dir", default="data/index")
    ap.add_argument("--bm25", action="store_true")
    ap.add_argument("--dense", metavar="MODEL")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    out = Path(args.dir) / args.mode
    out.mkdir(parents=True, exist_ok=True)
    chunks = load(Path(args.dir) / f"chunks_{args.mode}.jsonl")
    if args.bm25:
        build_bm25(chunks, out)
    if args.dense:
        build_dense(chunks, out, args.dense, args.batch)


if __name__ == "__main__":
    main()
