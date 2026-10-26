"""
SciLit - corpus/chunk.py
Part 5 step 1: cut parsed papers into retrieval units ("chunks") without losing context.

structure (default) : a chunk = one paragraph inside one section, never cutting a sentence;
                      paragraphs over MAX_WORDS are split at sentence ends, paragraphs under
                      MIN_WORDS are merged with the next one of the same section.
                      Each chunk gets a header "Title > Section" (prepended before indexing,
                      stored apart so quotes keep the authors' exact words), its sentence IDs
                      (traceability) and links to the previous / next chunk (small-to-big).
fixed (ablation)    : 1000-character windows with 200 overlap, ignoring structure - the
                      baseline the structure-aware chunking is measured against.
fixed_title         : the same windows, with the paper title as header. structure vs fixed changes
                      TWO things (boundaries and header); fixed_title isolates each effect:
                      fixed -> fixed_title = the header, fixed_title -> structure = the boundaries.

Input : data/parsed/documents.jsonl (structured papers only)
Output: data/index/chunks_<mode>.jsonl

Usage: python -m scilit.corpus.chunk            (or --mode fixed / fixed_title)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SKIP = {"front", "references", "acknowledgements"}
MIN_WORDS, MAX_WORDS = 40, 200
FIXED_CHARS, FIXED_OVERLAP = 1000, 200


def structure_chunks(doc: dict) -> list[dict]:
    chunks = []
    for si, sec in enumerate(doc["sections"]):
        if sec["type"] in SKIP or not sec["sentences"]:
            continue
        header = f"{doc['title']} > {sec['heading'] or sec['type'].replace('_', ' ').title()}"
        buf: list[dict] = []
        groups: list[list[dict]] = []
        by_para: dict[int, list[dict]] = {}
        for s in sec["sentences"]:
            by_para.setdefault(s["p"], []).append(s)
        for para in by_para.values():                 # paragraph order is preserved
            for s in para:
                buf.append(s)
                if sum(len(x["text"].split()) for x in buf) >= MAX_WORDS:
                    groups.append(buf)
                    buf = []
            if buf and sum(len(x["text"].split()) for x in buf) >= MIN_WORDS:
                groups.append(buf)                    # paragraph ends: close the chunk
                buf = []                              # (short ones carry over = merge)
        if buf:
            if groups and sum(len(x["text"].split()) for x in buf) < MIN_WORDS:
                groups[-1] += buf                     # tiny tail joins the previous chunk
            else:
                groups.append(buf)
        for g in groups:
            chunks.append({"paper_id": doc["id"], "section": si, "section_type": sec["type"],
                           "header": header, "text": " ".join(x["text"] for x in g),
                           "sentence_ids": [x["id"] for x in g]})
    return chunks


def fixed_chunks(doc: dict, title: bool = False) -> list[dict]:
    text = " ".join(s["text"] for sec in doc["sections"] if sec["type"] not in SKIP
                    for s in sec["sentences"])
    out, start = [], 0
    while start < len(text):
        out.append({"paper_id": doc["id"], "section": None, "section_type": None, "header": doc["title"] if title else "",
                    "text": text[start:start + FIXED_CHARS], "sentence_ids": []})
        start += FIXED_CHARS - FIXED_OVERLAP
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", default="data/parsed/documents.jsonl")
    ap.add_argument("--out", default="data/index")
    ap.add_argument("--mode", choices=["structure", "fixed", "fixed_title"], default="structure")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_docs = n = 0
    with (out / f"chunks_{args.mode}.jsonl").open("w", encoding="utf-8") as f:
        for line in Path(args.parsed).open(encoding="utf-8"):
            doc = json.loads(line)
            if not doc["structured"]:
                continue
            n_docs += 1
            chunks = structure_chunks(doc) if args.mode == "structure" else fixed_chunks(doc, args.mode == "fixed_title")
            for i, c in enumerate(chunks):
                c["id"] = f"{doc['id']}#c{i}"
                c["prev"] = f"{doc['id']}#c{i - 1}" if i else None
                c["next"] = f"{doc['id']}#c{i + 1}" if i + 1 < len(chunks) else None
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
            n += len(chunks)
    print(f"{args.mode}: {n} chunks from {n_docs} papers ({n // max(n_docs, 1)} per paper) -> "
          f"{out / f'chunks_{args.mode}.jsonl'}")


if __name__ == "__main__":
    main()
