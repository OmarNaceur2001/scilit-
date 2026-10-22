"""
SciLit - corpus/parse_pdf.py
Part 3: turn each accepted PDF into a structured paper.

  1. layout      PyMuPDF text blocks with font size, bold flag, position; two-column reading order
  2. clean       drop running headers/footers, page numbers, rotated side text (arXiv stamp),
                 tables and footnotes; fix ligatures (NFKC); rejoin words hyphenated at line ends
  3. sections    heading = short line, bold or larger, numbered ("3.2 Results") or a known word;
                 mapped to a canonical type with a lexicon; sub-sections inherit the parent type
  4. sentences   pySBD (rule-based, handles "et al.", "Fig. 3", "e.g."); every sentence gets an ID
  5. references  bibliography split into entries, linked to corpus papers (arXiv ID, DOI, title)
  6. citations   "(Hu et al., 2021)", "Hu et al. (2021)", "[3-5]" -> reference entry + citing sentence

Input : data/raw/articles.jsonl, reports/validation_report.csv, data/pdfs/
Output: data/parsed/documents.jsonl   one structured paper per line
        reports/parse_report.md       how well the parsing worked (for the report)

Usage (from the repository root, with PYTHONPATH=scilit/src):
  python -m scilit.corpus.parse_pdf --limit 20
  python -m scilit.corpus.parse_pdf
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf
import pysbd

from scilit.corpus.arxiv_meta import surname
from scilit.corpus.collect import norm_title

# ---- section lexicon: first match wins, so specific types come before generic ones ------------
SECTION_TYPES = [
    ("abstract", ["abstract"]),
    ("references", ["references", "bibliography"]),
    ("acknowledgements", ["acknowledgement", "acknowledgment"]),
    ("limitations", ["limitation"]),
    ("ethics", ["ethic", "broader impact"]),
    ("conclusion", ["conclusion", "concluding", "summary", "future work"]),
    ("related_work", ["related work", "background", "prior work", "literature", "preliminar"]),
    ("discussion", ["discussion"]),
    ("results", ["result", "evaluation", "analysis", "ablation", "comparison"]),
    ("experiments", ["experiment", "setup", "dataset", "data", "implementation", "training"]),
    ("method", ["method", "approach", "model", "framework", "architecture", "proposed", "system", "algorithm"]),
    ("introduction", ["introduction", "motivation", "overview"]),
    ("appendix", ["appendix", "supplementary"]),
]
NUMBERED = re.compile(r"^(?P<num>(?:\d{1,2}(?:\.\d{1,2}){0,3})|[A-H](?:\.\d{1,2}){0,2})\.?\s+(?P<text>[A-Z].*)$")
CAPTION = re.compile(r"^(Table|Figure|Fig\.)\s*\d+\s*[:.|]")
SEGMENTER = pysbd.Segmenter(language="en", clean=False)


def section_type(heading: str) -> str | None:
    h = heading.lower()
    for kind, words in SECTION_TYPES:
        if any(w in h for w in words):
            return kind
    return None


# --------------------------------------------------------------------------
# Step 1-2 · layout and cleaning
# --------------------------------------------------------------------------

def visual_lines(block: dict) -> list[dict]:
    """PyMuPDF splits "1" and "Introduction" into two lines at the same height: merge them."""
    merged: list[dict] = []
    for line in block.get("lines", []):
        if line.get("dir", (1, 0)) != (1, 0) and tuple(round(v) for v in line["dir"]) != (1, 0):
            continue                                            # rotated text (arXiv side stamp)
        spans = [s for s in line["spans"] if s["text"].strip()]
        if not spans:
            continue
        y = round(line["bbox"][1])
        piece = "".join(s["text"] for s in line["spans"]).strip()   # spans already carry their spaces
        if merged and abs(merged[-1]["y"] - y) <= 2:
            merged[-1]["spans"] += spans
            merged[-1]["pieces"].append(piece)
            merged[-1]["x0"] = min(merged[-1]["x0"], line["bbox"][0])
        else:
            merged.append({"y": y, "spans": spans, "pieces": [piece], "x0": line["bbox"][0]})
    for m in merged:
        m["text"] = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", " ".join(m["pieces"]))).strip()
        m["size"] = max(s["size"] for s in m["spans"])
        m["bold"] = all(s["flags"] & 16 or "Bold" in s["font"] for s in m["spans"])
    return merged


def page_blocks(page) -> list[dict]:
    """Text blocks in reading order: left column top to bottom, then right column."""
    mid = page.rect.width / 2
    blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 0]
    def column(b):                                   # full-width blocks (title, wide figure) count as left
        x0, x1 = b["bbox"][0], b["bbox"][2]
        return 1 if x0 >= mid - 10 else 0
    return sorted(blocks, key=lambda b: (column(b), b["bbox"][1]))


def repeated_lines(doc) -> set[str]:
    """Running headers/footers: the same text (digits ignored) at the top or bottom of most pages."""
    seen: Counter = Counter()
    for page in doc:
        h = page.rect.height
        keys = set()
        for b in page.get_text("dict")["blocks"]:
            if b.get("type") == 0 and (b["bbox"][1] < 0.07 * h or b["bbox"][3] > 0.93 * h):
                for line in visual_lines(b):
                    keys.add(re.sub(r"\d+", "#", line["text"].lower()))
        seen.update(keys)
    return {k for k, n in seen.items() if n >= max(3, doc.page_count // 2)}


def dehyphenate(lines: list[str], hyphenated: set[str]) -> str:
    """Join lines. A line-final hyphen is removed ("is-" + "sued" -> "issued") unless the
    hyphenated form appears elsewhere in this paper inside a line ("state-of-the-art",
    "pre-trained"), which shows the author really writes it with a hyphen."""
    out = ""
    for line in lines:
        if out.endswith("-") and line[:1].islower():
            head = re.search(r"(\w+)-$", out)
            tail = re.match(r"\w+", line)
            if head and tail and f"{head.group(1)}-{tail.group(0)}".lower() not in hyphenated:
                out = out[:-1] + line
                continue
            out += line
        else:
            out = (out + " " + line) if out else line
    return re.sub(r"\s+", " ", out).strip()


def alpha_ratio(text: str) -> float:
    visible = re.sub(r"\s", "", text)
    return sum(c.isalpha() for c in visible) / max(len(visible), 1)


# --------------------------------------------------------------------------
# Step 3-4 · sections and sentences
# --------------------------------------------------------------------------

def is_heading(line: dict, body_size: float) -> tuple[bool, str | None, int]:
    """(is it a heading?, canonical type, level). Must be short, bold or larger than body text,
    and either numbered ("3.2 Results") or a known heading word ("Abstract", "References")."""
    text = line["text"].strip()
    if len(text) > 80 or len(text.split()) > 10 or alpha_ratio(text) < 0.5 or text.endswith((".", ",", ":")):
        return False, None, 0
    if not (line["bold"] or line["size"] >= body_size + 0.8):
        return False, None, 0
    m = NUMBERED.match(text)
    if m:
        level = m.group("num").count(".") + 1
        return True, section_type(m.group("text")), level
    kind = section_type(text)
    if kind and len(text.split()) <= 4:
        return True, kind, 1
    return False, None, 0


def parse_document(pdf_path: Path, paper_id: str) -> dict:
    doc = pymupdf.open(pdf_path)
    sizes: Counter = Counter()
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            for line in visual_lines(b) if b.get("type") == 0 else []:
                sizes[round(line["size"], 1)] += len(line["text"])
    body = sizes.most_common(1)[0][0] if sizes else 10.0
    noise = repeated_lines(doc)

    # pass 1: blocks -> (kind, payload) in reading order
    items = []
    for page in doc:
        for b in page_blocks(page):
            lines = [l for l in visual_lines(b) if re.sub(r"\d+", "#", l["text"].lower()) not in noise
                     and not re.fullmatch(r"\d{1,3}", l["text"].strip())]
            if not lines:
                continue
            block_text = " ".join(l["text"] for l in lines)
            if CAPTION.match(block_text):
                items.append(("caption", block_text))
                continue
            first = lines[0]
            heading, kind, level = is_heading(first, body)
            if heading:
                items.append(("heading", (first["text"], kind, level)))
                lines = lines[1:]
                if not lines:
                    continue
            small = max(l["size"] for l in lines) < body - 1.6
            items.append(("text", ([l["text"] for l in lines], alpha_ratio(block_text), small,
                                   [(l["x0"], l["text"]) for l in lines])))

    hyphenated = {h.lower() for kind, p in items if kind == "text" for l in p[0]
                  for h in re.findall(r"[A-Za-z]+-[A-Za-z]+", l)}

    # pass 2: build sections; tables (few letters) and footnotes (small font) are dropped,
    # except inside References, which is often set in a smaller font
    sections, captions = [], []
    current = {"heading": "", "type": "front", "level": 1, "paragraphs": []}
    parent_type = "front"
    for kind, payload in items:
        if kind == "caption":
            captions.append(payload)
        elif kind == "heading":
            sections.append(current)
            text, stype, level = payload
            if stype is None:                        # numbered heading with an unknown word
                stype = parent_type if level > 1 else "other"
            if current["type"] in ("references", "appendix") and stype not in ("references",):
                stype = "appendix" if stype in ("other", "method", "experiments", "results") else stype
            if level == 1:
                parent_type = stype
            current = {"heading": text, "type": stype, "level": level, "paragraphs": []}
        else:
            lines, ratio, small, xlines = payload
            in_refs = current["type"] == "references"
            if not in_refs and (ratio < 0.55 or small):
                continue
            if in_refs:
                current.setdefault("ref_lines", []).extend(xlines)
            current["paragraphs"].append(dehyphenate(lines, hyphenated))
    sections.append(current)
    sections = [s for s in sections if s["paragraphs"] or s["heading"]]

    # sentences with IDs (references are kept as entries, not sentences)
    n = 0
    for s in sections:
        s["sentences"] = []
        if s["type"] in ("references", "front"):
            continue
        for p_idx, para in enumerate(s["paragraphs"]):
            for sent in SEGMENTER.segment(para):
                sent = sent.strip()
                if len(sent) >= 15:
                    n += 1
                    s["sentences"].append({"id": f"s{n}", "p": p_idx, "text": sent})
    return {"id": paper_id, "pages": doc.page_count, "body_font": body, "sections": sections,
            "captions": captions}


# --------------------------------------------------------------------------
# Step 5-6 · references and in-text citations
# --------------------------------------------------------------------------

YEAR = re.compile(r"\b(19[5-9]\d|20[0-3]\d)[a-z]?\b")
ARXIV = re.compile(r"(?:arxiv[:\s/]*(?:abs/)?)(\d{4}\.\d{4,5})", re.I)
DOI = re.compile(r"\b(10\.\d{4,9}/[^\s,;]+[^\s,;.])", re.I)
NUMERIC_CIT = re.compile(r"\[(\d{1,3}(?:\s*[,–—-]\s*\d{1,3})*)\]")
AUTHOR_YEAR_CIT = re.compile(
    r"([A-Z][A-Za-zÀ-ſ'\-]+)(?:\s+et\s+al\.?|\s+(?:and|&)\s+[A-Z][A-Za-zÀ-ſ'\-]+)?"
    r",?\s*\(?((?:19|20)\d{2})[a-z]?\)?")


def split_references(ref_lines: list[tuple[float, str]]) -> list[str]:
    """Bibliography lines -> entries, using the layout: a numbered entry starts with "[12]";
    otherwise an entry starts at the column's left margin and its next lines are indented
    (hanging indent, the ACL / most LaTeX styles). Works across columns and pages because the
    margin is measured per column (lines within 40 pt horizontally)."""
    entries: list[list[str]] = []
    xs = [x for x, _ in ref_lines]
    for x, text in ref_lines:
        margin = min(v for v in xs if abs(v - x) < 40)
        new = re.match(r"\s*\[\d+\]", text) or x - margin < 2
        if new or not entries:
            entries.append([text])
        else:
            entries[-1].append(text)
    out = [dehyphenate(e, set()) for e in entries]
    return [e for e in out if len(e) > 25 and YEAR.search(e)]


def link_reference(entry: str, by_arxiv: dict, by_doi: dict, titles: list[tuple[str, str]]) -> str | None:
    """Corpus paper ID for a bibliography entry: by arXiv ID, DOI, then exact normalized title."""
    m = ARXIV.search(entry)
    if m and m.group(1) in by_arxiv:
        return by_arxiv[m.group(1)]
    m = DOI.search(entry)
    if m and m.group(1).lower() in by_doi:
        return by_doi[m.group(1).lower()]
    e = norm_title(entry)
    for t, pid in titles:
        if t in e:
            return pid
    return None


def find_citations(doc: dict, refs: list[dict]) -> list[dict]:
    """In-text citation markers -> reference index (or None if it cannot be resolved)."""
    numbered = all(r["text"].lstrip().startswith("[") for r in refs[:5]) if refs else False
    index_by_num = {}
    for i, r in enumerate(refs):
        m = re.match(r"\s*\[(\d+)\]", r["text"])
        if m:
            index_by_num[int(m.group(1))] = i
    cites = []
    for s in doc["sections"]:
        for sent in s["sentences"]:
            if numbered:
                for m in NUMERIC_CIT.finditer(sent["text"]):
                    nums = set()
                    for part in m.group(1).split(","):          # "3, 5-7" -> 3, 5, 6, 7
                        bounds = [x.strip() for x in re.split(r"[–—-]", part)]
                        if len(bounds) == 2 and all(x.isdigit() for x in bounds) and 0 <= int(bounds[1]) - int(bounds[0]) < 20:
                            nums.update(range(int(bounds[0]), int(bounds[1]) + 1))
                        elif bounds[0].isdigit():
                            nums.add(int(bounds[0]))
                    for k in sorted(nums):
                        cites.append({"sentence": sent["id"], "marker": m.group(0), "ref": index_by_num.get(k)})
            else:
                for m in AUTHOR_YEAR_CIT.finditer(sent["text"]):
                    name, year = surname(m.group(1)), m.group(2)
                    ref = next((i for i, r in enumerate(refs)
                                if year in r["text"] and name and name in surname_tokens(r["text"][:120])), None)
                    if ref is not None:                    # unresolved author-year matches are usually
                        cites.append({"sentence": sent["id"], "marker": m.group(0), "ref": ref})  # not citations
    return cites


def surname_tokens(text: str) -> set[str]:
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return set(re.findall(r"[a-z\-']+", t))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="SciLit Part 3: parse accepted PDFs")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--pdfs", default="data/pdfs")
    ap.add_argument("--out", default="data/parsed")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    raw, pdf_dir, out, reports = Path(args.raw), Path(args.pdfs), Path(args.out), Path(args.reports)
    out.mkdir(parents=True, exist_ok=True)

    articles = {json.loads(l)["id"]: json.loads(l)
                for l in (raw / "articles.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    accepted = [r["id"] for r in csv.DictReader((reports / "validation_report.csv").open(encoding="utf-8"))
                if r["status"] == "accepted"]
    if args.limit:
        accepted = accepted[: args.limit]
    by_arxiv = {a["arxiv_id"]: a["id"] for a in articles.values() if a.get("arxiv_id")}
    by_doi = {a["doi"]: a["id"] for a in articles.values() if a.get("doi")}
    titles = [(norm_title(a["title"]), a["id"]) for a in articles.values() if len(a["title"].split()) >= 4]

    stats = Counter()
    types = Counter()
    with (out / "documents.jsonl").open("w", encoding="utf-8") as f:
        for n, pid in enumerate(accepted, 1):
            try:
                doc = parse_document(pdf_dir / f"{pid}.pdf", pid)
            except Exception as e:                   # one broken PDF must not stop the run
                stats["parse_error"] += 1
                print(f"  parse error {pid}: {type(e).__name__}", flush=True)
                continue
            ref_lines = [x for s in doc["sections"] if s["type"] == "references" for x in s.get("ref_lines", [])]
            for s in doc["sections"]:
                s.pop("ref_lines", None)
            refs = []
            for t in split_references(ref_lines):
                linked = link_reference(t, by_arxiv, by_doi, titles)
                refs.append({"text": t, "year": YEAR.search(t).group(1),
                             "corpus_id": linked if linked != pid else None})
            doc["title"] = articles[pid]["title"]
            kinds = {s["type"] for s in doc["sections"]}
            # a scientific paper has at least two of these; manuals or non-paper documents do not
            doc["structured"] = len(kinds & {"abstract", "introduction", "conclusion", "references"}) >= 2
            doc["references"] = refs
            doc["citations"] = find_citations(doc, refs)
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

            stats["papers"] += 1
            stats["unstructured"] += not doc["structured"]
            stats["sentences"] += sum(len(s["sentences"]) for s in doc["sections"])
            stats["references"] += len(refs)
            stats["refs_in_corpus"] += sum(1 for r in refs if r["corpus_id"])
            stats["citations"] += len(doc["citations"])
            stats["citations_linked"] += sum(1 for c in doc["citations"] if c["ref"] is not None)
            for k in ("abstract", "introduction", "conclusion", "references"):
                stats[f"has_{k}"] += k in kinds
            for s in doc["sections"]:
                types[s["type"]] += len(s["sentences"])
            if n % 100 == 0:
                print(f"  {n}/{len(accepted)}", flush=True)

    p = max(stats["papers"], 1)
    lines = ["# Parse report (Part 3)", "",
             f"- Papers parsed: {stats['papers']} (errors: {stats['parse_error']}; "
             f"without a paper structure, flagged structured=false: {stats['unstructured']})",
             f"- Sentences: {stats['sentences']} ({stats['sentences'] // p} per paper)",
             f"- Papers with Abstract / Introduction / Conclusion / References found: "
             f"{stats['has_abstract'] / p:.0%} / {stats['has_introduction'] / p:.0%} / "
             f"{stats['has_conclusion'] / p:.0%} / {stats['has_references'] / p:.0%}",
             f"- Bibliography entries: {stats['references']} ({stats['references'] // p} per paper), "
             f"of which in our corpus: {stats['refs_in_corpus']}",
             f"- In-text citations: {stats['citations']}, linked to an entry: "
             f"{stats['citations_linked'] / max(stats['citations'], 1):.0%} "
             "(author-year markers are kept only when they match an entry, so this is high by design)", "",
             "| Section type | Sentences |", "| --- | --- |",
             *[f"| {k} | {v} |" for k, v in types.most_common() if v]]
    (reports / "parse_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[2:7]))
    print(f"Output -> {out / 'documents.jsonl'}")


if __name__ == "__main__":
    main()
