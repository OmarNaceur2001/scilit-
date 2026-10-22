"""
SciLit - corpus/arxiv_meta.py
Authoritative metadata from the arXiv API (title, authors, abstract, first-version year).

Why: OpenAlex is excellent for discovery (search, references, citations) but its metadata
can be wrong; e.g. LoRA (arXiv 2106.09685) carries another paper's title in OpenAlex.
For papers with an arXiv ID, arXiv itself is the source of truth.

API: https://export.arxiv.org/api/query?id_list=...  (Atom XML, up to 100 IDs per request,
arXiv asks for at least 3 seconds between requests).
"""
from __future__ import annotations

import re
import time
import unicodedata
import xml.etree.ElementTree as ET

import requests

ARXIV_API = "https://export.arxiv.org/api/query"
BATCH = 100
DELAY = 3.0
NS = {"a": "http://www.w3.org/2005/Atom"}


def _clean(text: str | None) -> str:
    """arXiv titles and abstracts contain line breaks and double spaces."""
    return re.sub(r"\s+", " ", text or "").strip()


def parse_feed(xml_text: str) -> dict[str, dict]:
    """Atom feed -> {arxiv_id_without_version: {title, authors, abstract, year}}."""
    out: dict[str, dict] = {}
    root = ET.fromstring(xml_text)
    for entry in root.findall("a:entry", NS):
        raw_id = entry.findtext("a:id", default="", namespaces=NS)
        m = re.search(r"abs/(.+?)(v\d+)?$", raw_id)       # ".../abs/2106.09685v2" -> 2106.09685
        title = _clean(entry.findtext("a:title", default="", namespaces=NS))
        if not m or not title or title.lower() == "error":
            continue
        out[m.group(1)] = {
            "title": title,
            "authors": [_clean(a.findtext("a:name", default="", namespaces=NS))
                        for a in entry.findall("a:author", NS)],
            "abstract": _clean(entry.findtext("a:summary", default="", namespaces=NS)),
            "year": int((entry.findtext("a:published", default="0000", namespaces=NS))[:4]),
        }
    return out


def fetch(arxiv_ids: list[str], session: requests.Session | None = None) -> dict[str, dict]:
    """Metadata for many arXiv IDs, 100 per request, polite delay between requests."""
    session = session or requests.Session()
    ids = sorted({i for i in arxiv_ids if i})
    result: dict[str, dict] = {}
    for start in range(0, len(ids), BATCH):
        batch = ids[start:start + BATCH]
        for attempt in range(4):
            try:
                r = session.get(ARXIV_API, params={"id_list": ",".join(batch),
                                                   "max_results": len(batch)}, timeout=60)
                if r.status_code == 200:
                    result.update(parse_feed(r.text))
                    break
            except (requests.RequestException, ET.ParseError):
                pass                                  # network error or truncated reply: retry
            time.sleep(DELAY * (attempt + 2))         # 503 or rate limit: wait longer, retry
        else:
            print(f"WARNING: arXiv unreachable for {len(batch)} IDs; their metadata is NOT verified.",
                  flush=True)
        time.sleep(DELAY)
    return result


def surname(name: str) -> str:
    """Surname in ASCII lower case, whatever the format: "Nikhil Ghosh" or "Ghosh, Nikhil".
    Empty string when the name has no Latin letters (e.g. written only in Chinese characters)."""
    name = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()  # Müller -> Muller
    if "," in name:
        name = name.split(",")[0]
    parts = re.sub(r"[^a-z\s-]", " ", name.lower()).split()
    return parts[-1] if parts else ""


def _last_names(authors: list[str]) -> set[str]:
    return {s for s in (surname(a) for a in authors) if s}


def same_paper(oa_authors: list[str], oa_year: int | None, ax: dict) -> bool:
    """Is an OpenAlex record really the arXiv paper it claims to be?
    Titles can be wrong in OpenAlex, so we compare the AUTHORS (last names: at least half of the
    shorter list must match) and the YEAR (a preprint and its conference version can differ by
    1-2 years, more than that means another paper). If OpenAlex has no authors, we cannot tell,
    so we do not accuse the record."""
    a, b = _last_names(oa_authors), _last_names(ax["authors"])
    if a and b and len(a & b) / min(len(a), len(b)) < 0.5:
        return False
    if oa_year and ax.get("year") and abs(oa_year - ax["year"]) > 2:
        return False
    return True
