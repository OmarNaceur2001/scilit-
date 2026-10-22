"""
SciLit - corpus/collect.py
Part 0 (corpus scoping) + Part 1 steps 1-2 (metadata harvest).

Reads configs/themes.yaml and builds the list of candidate papers from OpenAlex:
  1. keyword search per theme (filters: years, open access, abstract, Computer Science, English)
  2. seed check: each seed paper is looked up by its arXiv ID or DOI, found or reported missing
  3. snowball: papers citing / cited by the found seeds, kept only if they match a theme keyword
  4. merge: one record per paper, with the list of themes and how it was found

Outputs:
  data/raw/candidates.jsonl   one paper per line
  data/raw/query_log.jsonl    every request sent to OpenAlex and its result count
  reports/coverage_report.md  papers per theme, seeds found / missing

Usage (from the repository root, with PYTHONPATH=scilit/src):
  python -m scilit.corpus.collect                 # all themes
  python -m scilit.corpus.collect --only peft     # one theme, for a quick test
  python -m scilit.corpus.collect --check-seeds   # verify every seed ID, collect nothing
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml
from rapidfuzz import fuzz

from scilit.corpus import arxiv_meta

OPENALEX = "https://api.openalex.org"
CS_FIELD_ID = "17"           # OpenAlex field "Computer Science"; checked at start-up
PER_KEYWORD = 100            # results fetched per keyword before interleaving
SEED_MATCH_MIN = 90          # fuzzy title score (0-100) to accept a seed match
SELECT = ",".join([
    "id", "doi", "title", "publication_year", "authorships", "abstract_inverted_index",
    "primary_location", "best_oa_location", "locations", "open_access", "language",
    "cited_by_count", "referenced_works", "primary_topic",
])


# --------------------------------------------------------------------------
# OpenAlex client: one place for requests, retries, politeness and logging
# --------------------------------------------------------------------------

class OpenAlexClient:
    def __init__(self, email: str | None, log_path: Path, delay: float = 0.15):
        self.email = email
        self.log_path = log_path
        self.delay = delay
        self.session = requests.Session()

    def get(self, path: str, params: dict, purpose: str) -> dict | None:
        """GET with retries. Returns None when OpenAlex has no such record (404)."""
        params = dict(params)
        if self.email:
            params["mailto"] = self.email          # OpenAlex "polite pool": faster, more reliable
        if os.environ.get("OPENALEX_API_KEY"):     # free personal key: own daily budget
            params["api_key"] = os.environ["OPENALEX_API_KEY"]
        for attempt in range(5):
            try:
                r = self.session.get(f"{OPENALEX}{path}", params=params, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    self._log(path, params, purpose, data)
                    time.sleep(self.delay)
                    return data
                if r.status_code == 404:
                    self._log(path, params, purpose, {"results": []})
                    return None
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt)           # back off: 1, 2, 4, 8, 16 s
                    continue
                r.raise_for_status()
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenAlex request failed after 5 attempts: {path} {params}")

    def _log(self, path: str, params: dict, purpose: str, data: dict) -> None:
        entry = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "purpose": purpose,
            "path": path,
            "params": {k: v for k, v in params.items() if k not in ("mailto", "select", "api_key")},
            "count": (data.get("meta") or {}).get("count"),
            "returned": len(data.get("results", [])) if "results" in data else 1,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def rebuild_abstract(inv: dict | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; put the words back in order.
    (Kept from the original SciLit collector.)"""
    if not inv:
        return ""
    positions: dict[int, str] = {}
    for word, pos_list in inv.items():
        for p in pos_list:
            positions[p] = word
    return " ".join(positions[i] for i in sorted(positions))


def norm_title(title: str) -> str:
    """Lower case, accents removed, letters and digits only: used to compare titles."""
    t = unicodedata.normalize("NFKD", title or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def arxiv_id(work: dict) -> str | None:
    m = re.search(r"10\.48550/arxiv\.(\d{4}\.\d{4,5})", (work.get("doi") or "").lower())
    if m:
        return m.group(1)
    for loc in work.get("locations") or []:
        m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", loc.get("landing_page_url") or "")
        if m:
            return m.group(1)
    return None


def matches_keyword(paper: dict, keywords: list[str]) -> bool:
    text = norm_title(paper["title"] + " " + paper["abstract"])
    return any(norm_title(k) in text for k in keywords)


def to_record(work: dict) -> dict:
    """Keep only the fields the pipeline uses, under stable names."""
    best = work.get("best_oa_location") or {}
    primary = work.get("primary_location") or {}
    topic = work.get("primary_topic") or {}
    return {
        "openalex_id": (work.get("id") or "").rsplit("/", 1)[-1],
        "doi": (work.get("doi") or "").removeprefix("https://doi.org/").lower() or None,
        "arxiv_id": arxiv_id(work),
        "title": (work.get("title") or "").strip(),
        "year": work.get("publication_year"),
        "authors": [a["author"]["display_name"] for a in work.get("authorships") or []
                    if (a.get("author") or {}).get("display_name")][:20],
        "venue": ((primary.get("source") or {}).get("display_name")),
        "abstract": rebuild_abstract(work.get("abstract_inverted_index")),
        "language": work.get("language"),
        "is_oa": (work.get("open_access") or {}).get("is_oa", False),
        "oa_url": (work.get("open_access") or {}).get("oa_url"),
        "pdf_url": best.get("pdf_url"),
        "license": best.get("license"),
        "cited_by_count": work.get("cited_by_count") or 0,
        "referenced_works": [r.rsplit("/", 1)[-1] for r in work.get("referenced_works") or []],
        "openalex_field": ((topic.get("field") or {}).get("display_name")),
    }


def filter_reason(rec: dict, cfg: dict) -> str | None:
    """The same rules as the search filters, applied to papers found another way.
    Returns None if the paper passes, else the first rule it breaks."""
    if len(rec["abstract"]) < 50:
        return "no_abstract"
    if rec["year"] is None or not cfg["year_min"] <= rec["year"] <= cfg["year_max"]:
        return f"year_{rec['year']}"
    if not rec["is_oa"]:
        return "not_open_access"
    if rec["language"] != "en":
        return f"language_{rec['language']}"
    if rec["openalex_field"] != "Computer Science":
        return f"field_{rec['openalex_field']}"
    return None


def passes_filters(rec: dict, cfg: dict) -> bool:
    return filter_reason(rec, cfg) is None


def base_filter(cfg: dict) -> str:
    return ",".join([
        f"publication_year:{cfg['year_min']}-{cfg['year_max']}",
        "open_access.is_oa:true",
        "has_abstract:true",
        "language:en",
        f"primary_topic.field.id:{CS_FIELD_ID}",
    ])


# --------------------------------------------------------------------------
# The four collection steps
# --------------------------------------------------------------------------

def keyword_search(client: OpenAlexClient, theme: dict, cfg: dict) -> list[dict]:
    """Search each keyword, then interleave the ranked lists (1st of each, 2nd of each...)
    so every keyword gets a fair share of the quota."""
    ranked_lists = []
    for kw in theme["keywords"]:
        data = client.get("/works", {
            "filter": f'{base_filter(cfg)},title_and_abstract.search:"{kw}"',
            "sort": "relevance_score:desc",
            "per-page": PER_KEYWORD,
            "select": SELECT,
        }, purpose=f"keyword:{theme['id']}:{kw}") or {}
        ranked_lists.append([to_record(w) for w in data.get("results", [])])

    picked, seen = [], set()
    for rank in range(PER_KEYWORD):
        for lst in ranked_lists:
            if rank < len(lst) and lst[rank]["openalex_id"] not in seen:
                seen.add(lst[rank]["openalex_id"])
                picked.append(lst[rank])
                if len(picked) == cfg["papers_per_theme"]:
                    return picked
    return picked


def seed_path(seed: dict) -> str | None:
    """OpenAlex lookup path for a seed's identifier (arXiv papers have DOI 10.48550/arXiv.<id>)."""
    if seed.get("arxiv"):
        return f"/works/doi:10.48550/arXiv.{seed['arxiv']}"
    if seed.get("doi"):
        return f"/works/doi:{seed['doi']}"
    return None


def lookup_seed(client: OpenAlexClient, seed: dict, purpose: str) -> tuple[dict | None, int]:
    """Find a seed's OpenAlex record by ID (exact); fall back to title search only if the seed
    has no ID. Returns (work or None, title similarity 0-100 between our title and OpenAlex's)."""
    path = seed_path(seed)
    if path:
        work = client.get(path, {"select": SELECT}, purpose=purpose)
    else:
        query = re.sub(r"[^\w\s-]", " ", seed["title"])    # "?" or ":" make OpenAlex reject the search
        data = client.get("/works", {"search": query, "per-page": 5, "select": SELECT},
                          purpose=purpose) or {}
        work = max(data.get("results", []), default=None,
                   key=lambda w: fuzz.ratio(norm_title(seed["title"]), norm_title(w.get("title") or "")))
    if work is None:
        return None, 0
    return work, round(fuzz.ratio(norm_title(seed["title"]), norm_title(work.get("title") or "")))


def find_seeds(client: OpenAlexClient, theme: dict, cfg: dict) -> tuple[list[dict], list[dict]]:
    """Each seed: found / filtered_out (with the rule it breaks) / missing.
    A title disagreement is flagged, not fatal: the ID is the identity, and OpenAlex titles
    can be wrong (normalize.py replaces them with arXiv's)."""
    found, report = [], []
    for seed in theme["seeds"]:
        work, score = lookup_seed(client, seed, purpose=f"seed:{theme['id']}")
        if work is None or (not seed_path(seed) and score < SEED_MATCH_MIN):
            report.append({"title": seed["title"], "status": "missing", "score": score})
            continue
        rec = to_record(work)
        reason = filter_reason(rec, cfg)
        # Seeds are verified by hand (ID + arXiv). OpenAlex's labels (field, language, open-access
        # flag) are wrong for some of them (RoBERTa labelled "Arts and Humanities"), so for seeds
        # only the year rule applies; the label they were kept despite is reported.
        kept_despite = reason if reason and not reason.startswith("year") else None
        ok = reason is None or kept_despite is not None
        report.append({"title": seed["title"], "status": "found" if ok else "filtered_out",
                       "reason": reason, "kept_despite": kept_despite,
                       "openalex_id": rec["openalex_id"], "score": score,
                       "title_mismatch": score < SEED_MATCH_MIN})
        if ok:
            found.append(rec)
    verify_against_arxiv(found, report)
    return found, report


def verify_against_arxiv(found: list[dict], report: list[dict]) -> None:
    """For seeds with an arXiv ID: is the OpenAlex record really that paper (same authors, year)?
    If not, OpenAlex attached another paper's data to this ID. We keep the seed (the ID is right)
    but mark the record so the snowball does not follow the other paper's references."""
    truth = arxiv_meta.fetch([p["arxiv_id"] for p in found if p.get("arxiv_id")])
    by_id = {r.get("openalex_id"): r for r in report}
    for p in found:
        ax = truth.get(p.get("arxiv_id") or "")
        if ax and not arxiv_meta.same_paper(p["authors"], p["year"], ax):
            p["openalex_conflated"] = True
            by_id[p["openalex_id"]]["conflated"] = True


def check_seeds(client: OpenAlexClient, cfg: dict, report_path: Path) -> None:
    """Verify every seed ID: what OpenAlex returns, and what arXiv (the authority) says.
    If arXiv's title differs from ours, our ID is wrong. If only OpenAlex's differs,
    OpenAlex's metadata is wrong."""
    seeds = [(t["id"], s) for t in cfg["themes"] for s in t["seeds"]]
    truth = arxiv_meta.fetch([s.get("arxiv") for _, s in seeds])
    lines = ["# Seed check", "", "| Theme | Seed | ID | arXiv title matches | OpenAlex record | OpenAlex title matches |",
             "| --- | --- | --- | --- | --- | --- |"]
    bad_ids = 0
    for tid, s in seeds:
        work, score = lookup_seed(client, s, purpose=f"check_seed:{tid}")
        ax = truth.get(s.get("arxiv") or "")
        ax_ok = "n/a (DOI)" if not s.get("arxiv") else (
            "yes" if ax and fuzz.ratio(norm_title(ax["title"]), norm_title(s["title"])) >= SEED_MATCH_MIN
            else f"NO: {ax['title'] if ax else 'not on arXiv'}")
        bad_ids += ax_ok.startswith("NO")
        oa = "missing" if work is None else (work.get("id") or "").rsplit("/", 1)[-1]
        oa_ok = "-" if work is None else ("yes" if score >= SEED_MATCH_MIN else f"NO ({score}): {work.get('title')}")
        ident = f"arXiv {s['arxiv']}" if s.get("arxiv") else f"DOI {s.get('doi')}"
        lines.append(f"| {tid} | {s['title'][:60]} | {ident} | {ax_ok} | {oa} | {oa_ok} |")
        print(f"{tid:22} {ident:28} arXiv:{ax_ok[:40]:40} OpenAlex:{oa_ok[:50]}", flush=True)
    lines += ["", f"Seeds whose ID does not match its title on arXiv: {bad_ids} (must be 0)."]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{bad_ids} wrong IDs. Report -> {report_path}")


def snowball(client: OpenAlexClient, theme: dict, cfg: dict, seeds: list[dict],
             already: set[str]) -> list[dict]:
    """Papers cited by the seeds (backward) and citing the seeds (forward), kept only if
    they pass our filters, contain a theme keyword, and are not already in this theme.
    Most-cited first, capped."""
    sb = cfg["snowball"]
    pool: dict[str, dict] = {}
    for seed in seeds:
        if seed.get("openalex_conflated"):        # record is another paper: its links are not the seed's
            continue
        refs = seed["referenced_works"]
        for i in range(0, len(refs), 50):                       # backward: 50 IDs per request
            data = client.get("/works", {"filter": "openalex_id:" + "|".join(refs[i:i + 50]),
                                         "per-page": 50, "select": SELECT},
                              purpose=f"snowball_back:{theme['id']}") or {}
            for w in data.get("results", []):
                rec = to_record(w)
                pool.setdefault(rec["openalex_id"], rec)
        data = client.get("/works", {                           # forward: most-cited citers
            "filter": f"{base_filter(cfg)},cites:{seed['openalex_id']}",
            "sort": "cited_by_count:desc", "per-page": 200, "select": SELECT,
        }, purpose=f"snowball_fwd:{theme['id']}") or {}
        for w in data.get("results", []):
            rec = to_record(w)
            pool.setdefault(rec["openalex_id"], rec)

    kept = [p for p in pool.values()
            if p["openalex_id"] not in already
            and passes_filters(p, cfg)
            and (not sb["require_keyword_match"] or matches_keyword(p, theme["keywords"]))]
    kept.sort(key=lambda p: p["cited_by_count"], reverse=True)
    return kept[: sb["max_per_theme"]]


METHOD_ORDER = {"seed": 0, "keyword": 1, "snowball": 2}   # strongest reason first


def merge(corpus: dict[str, dict], papers: list[dict], theme_id: str, method: str) -> int:
    """Add papers to the corpus; a paper already there just gains a theme. Returns new count."""
    new = 0
    for p in papers:
        entry = corpus.get(p["openalex_id"])
        if entry is None:
            entry = {**p, "themes": [], "found_by": {}}
            corpus[p["openalex_id"]] = entry
            new += 1
        if theme_id not in entry["themes"]:
            entry["themes"].append(theme_id)
        old = entry["found_by"].get(theme_id)
        if old is None or METHOD_ORDER[method] < METHOD_ORDER[old]:
            entry["found_by"][theme_id] = method
    return new


# --------------------------------------------------------------------------
# Report + main
# --------------------------------------------------------------------------

def write_report(path: Path, cfg: dict, corpus: dict, stats: dict, seed_reports: dict) -> None:
    lines = [
        "# Coverage report (Part 0)",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} · "
        f"{len(corpus)} unique papers · years {cfg['year_min']}-{cfg['year_max']}",
        "",
        "| Theme | Keyword | Seeds found | Snowball | Papers in theme |",
        "| --- | --- | --- | --- | --- |",
    ]
    for t in cfg["themes"]:
        s = stats.get(t["id"])
        if not s:
            continue
        in_theme = sum(1 for p in corpus.values() if t["id"] in p["themes"])
        found = sum(1 for r in seed_reports[t["id"]] if r["status"] == "found")
        lines.append(f"| {t['id']} | {s['keyword']} | {found}/{len(t['seeds'])} "
                     f"| {s['snowball']} | {in_theme} |")
    lines += ["", "## Seeds not in the corpus", ""]
    problems = [(tid, r) for tid, rs in seed_reports.items() for r in rs if r["status"] != "found"]
    if not problems:
        lines.append("None: every seed paper was found and passed the filters.")
    for tid, r in problems:
        why = f" · {r['reason']}" if r.get("reason") else ""
        lines.append(f"- **{tid}** · {r['status']}{why}: {r['title']}")
    flagged = [(tid, r) for tid, rs in seed_reports.items() for r in rs if r.get("title_mismatch")]
    lines += ["", "## Seeds whose OpenAlex title is wrong (fixed later from arXiv)", ""]
    lines += [f"- **{tid}** · {r['title']} (OpenAlex title similarity {r['score']}/100)"
              + (" · **record belongs to another paper: not snowballed**" if r.get("conflated") else
                 " · authors and year match arXiv: only the title is wrong")
              for tid, r in flagged] or ["None."]
    despite = [(tid, r) for tid, rs in seed_reports.items() for r in rs if r.get("kept_despite")]
    lines += ["", "## Seeds kept despite a wrong OpenAlex label", ""]
    lines += [f"- **{tid}** · OpenAlex says {r['kept_despite']}: {r['title']}" for tid, r in despite] or ["None."]
    multi = sum(1 for p in corpus.values() if len(p["themes"]) > 1)
    lines += ["", f"Papers in more than one theme: {multi}."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_cs_field(client: OpenAlexClient) -> None:
    """Our filter uses a numeric field ID; make sure it really is Computer Science."""
    data = client.get(f"/fields/{CS_FIELD_ID}", {}, purpose="check_field") or {}
    name = data.get("display_name")
    if name != "Computer Science":
        raise SystemExit(f"OpenAlex field {CS_FIELD_ID} is '{name}', not Computer Science. "
                         "Update CS_FIELD_ID in collect.py.")


def main() -> None:
    ap = argparse.ArgumentParser(description="SciLit Part 0: collect candidate papers")
    ap.add_argument("--themes", default="configs/themes.yaml")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--only", help="collect a single theme id (quick test)")
    ap.add_argument("--check-seeds", action="store_true", help="verify seed IDs only, collect nothing")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.themes).read_text(encoding="utf-8"))
    out, reports = Path(args.out), Path(args.reports)
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    log_path = out / "query_log.jsonl"
    log_path.write_text("", encoding="utf-8")                    # fresh log for this run

    client = OpenAlexClient(os.environ.get("OPENALEX_EMAIL"), log_path)
    check_cs_field(client)
    if args.check_seeds:
        check_seeds(client, cfg, reports / "seed_check.md")
        return

    themes = [t for t in cfg["themes"] if not args.only or t["id"] == args.only]
    if not themes:
        raise SystemExit(f"No theme with id '{args.only}'.")

    corpus: dict[str, dict] = {}
    stats: dict[str, dict] = {}
    seed_reports: dict[str, list] = {}
    for t in themes:
        print(f"[{t['id']}] keyword search...", flush=True)
        kw = keyword_search(client, t, cfg)
        print(f"[{t['id']}] seeds...", flush=True)
        seeds, seed_reports[t["id"]] = find_seeds(client, t, cfg)
        print(f"[{t['id']}] snowball...", flush=True)
        sb = snowball(client, t, cfg, seeds, {p["openalex_id"] for p in kw + seeds})
        merge(corpus, seeds, t["id"], "seed")
        merge(corpus, kw, t["id"], "keyword")
        merge(corpus, sb, t["id"], "snowball")
        stats[t["id"]] = {"keyword": len(kw), "snowball": len(sb)}
        print(f"[{t['id']}] keyword={len(kw)} seeds={len(seeds)}/{len(t['seeds'])} "
              f"snowball={len(sb)} corpus total={len(corpus)}", flush=True)

    with (out / "candidates.jsonl").open("w", encoding="utf-8") as f:
        for p in corpus.values():
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    write_report(reports / "coverage_report.md", cfg, corpus, stats, seed_reports)
    print(f"Done: {len(corpus)} unique papers -> {out / 'candidates.jsonl'}")
    print(f"Report -> {reports / 'coverage_report.md'}")


if __name__ == "__main__":
    main()
