"""
SciLit -- corpus/collect.py  (v3 -- OpenAlex uniquement)
=========================================================
Source unique : OpenAlex (api.openalex.org)
  - Gratuit, 10 req/s, pas de cle API, zero 429
  - Couvre ACL / EMNLP / NAACL / COLING / arXiv
  - Abstracts reconstruits depuis inverted index
  - Pagination cursor (tres rapide)

Usage :
  python -m scilit.corpus.collect             # 500 articles
  python -m scilit.corpus.collect --max 100   # test
  python -m scilit.corpus.collect --max 1000  # corpus complet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Generator

import requests
from pydantic import ValidationError
from tqdm import tqdm

from scilit.schemas import Article, ArticleSource, Language, SplitType, Venue

# =============================================================================
# CONFIG
# =============================================================================

YEAR_MIN = 2021
YEAR_MAX = 2026

OA_BASE   = "https://api.openalex.org"
OA_EMAIL  = "omar.naceur01@gmail.com"   # polite pool = pas de limite stricte
OA_PER_PAGE = 200
OA_DELAY    = 0.15                      # ~6-7 req/s (limite = 10/s)

# Concept OpenAlex : Natural Language Processing
OA_CONCEPT_NLP = "C41008148"

# Champs a recuperer (evite de telecharger tout l'objet)
OA_SELECT = ",".join([
    "id", "title", "authorships", "publication_year",
    "abstract_inverted_index", "primary_location",
    "open_access", "ids", "locations",
])

RAW_DIR       = Path("data/raw")
MANIFEST_PATH = RAW_DIR / "manifest.json"
ARTICLES_PATH = RAW_DIR / "articles.jsonl"
LOG_PATH      = RAW_DIR / "collect_log.txt"

# =============================================================================
# LOGGING
# =============================================================================

RAW_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# =============================================================================
# MANIFEST
# =============================================================================

class Manifest:
    def __init__(self, path: Path = MANIFEST_PATH) -> None:
        self.path = path
        self.seen: set[str] = set()
        if path.exists():
            try:
                data = json.loads(path.read_text("utf-8"))
                self.seen = set(data.get("ids", []))
                log.info(f"Manifest : {len(self.seen)} IDs existants.")
            except Exception:
                pass

    def contains(self, cid: str) -> bool:
        return cid in self.seen

    def add(self, cid: str) -> None:
        self.seen.add(cid)

    def save(self) -> None:
        self.path.write_text(
            json.dumps({
                "ids": sorted(self.seen),
                "total": len(self.seen),
                "updated": datetime.now().isoformat(),
            }, indent=2),
            "utf-8",
        )

# =============================================================================
# UTILITAIRES OPENALEX
# =============================================================================

def _reconstruct_abstract(inv: dict | None) -> str:
    """
    OpenAlex stocke les abstracts en inverted index :
      {"word": [pos1, pos2, ...], ...}
    On reconstruit le texte en remettant les mots dans l'ordre.
    """
    if not inv:
        return ""
    positions: dict[int, str] = {}
    for word, pos_list in inv.items():
        for p in pos_list:
            positions[p] = word
    return " ".join(positions[i] for i in sorted(positions))


def _detect_venue(work: dict) -> Venue:
    """Detecte la venue NLP depuis primary_location."""
    primary = work.get("primary_location") or {}
    source  = primary.get("source") or {}
    name    = (source.get("display_name") or "").lower()
    host    = (source.get("host_organization_name") or "").lower()
    combined = name + " " + host

    if "emnlp" in combined or "empirical methods" in combined:
        return Venue.EMNLP
    if "naacl" in combined or "north american" in combined:
        return Venue.NAACL
    if "coling" in combined or "international conference on computational linguistics" in combined:
        return Venue.COLING
    if "association for computational linguistics" in combined or "acl" in combined:
        return Venue.ACL
    if "iclr" in combined:
        return Venue.ICLR
    if "neural information processing" in combined or "neurips" in combined:
        return Venue.NEURIPS
    if "icml" in combined:
        return Venue.ICML
    return Venue.ARXIV


def _extract_arxiv_id(work: dict) -> str | None:
    """Extrait l'ID arXiv depuis les locations ou ids."""
    # IDs directs
    ids = work.get("ids") or {}
    if "arxiv" in (ids.get("arxiv") or "").lower():
        raw = ids["arxiv"].split("/abs/")[-1]
        return raw.split("v")[0] if raw else None

    # Locations
    for loc in work.get("locations", []):
        url = loc.get("landing_page_url") or ""
        if "arxiv.org/abs/" in url:
            return url.split("/abs/")[-1].split("v")[0]
    return None


def _oa_to_article(work: dict) -> Article | None:
    """Convertit un resultat OpenAlex en Article Pydantic."""
    title    = (work.get("title") or "").strip()
    year     = work.get("publication_year") or 0
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))

    # Filtres de base
    if not title or len(abstract) < 50:
        return None
    if not (YEAR_MIN <= year <= YEAR_MAX):
        return None

    # Venue et IDs
    venue     = _detect_venue(work)
    arxiv_id  = _extract_arxiv_id(work)
    oa_id     = (work.get("id") or "").replace("https://openalex.org/", "")

    # Au moins un ID externe requis
    if not arxiv_id and not oa_id:
        return None

    # URLs
    primary = work.get("primary_location") or {}
    oa_info = work.get("open_access") or {}
    pdf_url = oa_info.get("oa_url") or primary.get("pdf_url")

    if arxiv_id:
        base_url = f"https://arxiv.org/abs/{arxiv_id}"
        pdf_url  = pdf_url or f"https://arxiv.org/pdf/{arxiv_id}"
    else:
        base_url = work.get("id") or f"https://openalex.org/{oa_id}"

    if not base_url:
        return None

    # Auteurs (max 20)
    authors = []
    for a in (work.get("authorships") or [])[:20]:
        name = (a.get("author") or {}).get("display_name")
        if name:
            authors.append(name)
    if not authors:
        authors = ["Unknown"]

    try:
        source = ArticleSource(
            venue=venue,
            arxiv_id=arxiv_id,
            acl_id=None,
            url=base_url,         # type: ignore[arg-type]
            pdf_url=pdf_url,      # type: ignore[arg-type]
            license="open access",
        )
        return Article(
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
            language=Language.EN,
            source=source,
            split=SplitType.TRAIN,
        )
    except (ValidationError, Exception) as e:
        log.debug(f"OA skip {oa_id}: {e}")
        return None

# =============================================================================
# COLLECTEUR OPENALEX
# =============================================================================

def fetch_openalex(
    max_articles: int = 500,
    manifest: Manifest | None = None,
) -> Generator[Article, None, None]:
    """
    Collecte via OpenAlex cursor-pagination.
    Filtre : concept NLP (C41008148), open_access, 2021-2026.

    Performance : ~200 articles/requete, ~1 req/s => 500 articles en ~30 sec.
    """
    log.info(f"OpenAlex : cible {max_articles} articles (NLP, {YEAR_MIN}-{YEAR_MAX})...")

    filter_str = (
        f"concepts.id:{OA_CONCEPT_NLP},"
        f"publication_year:{YEAR_MIN}-{YEAR_MAX},"
        "open_access.is_oa:true,"
        "has_abstract:true"
    )

    cursor = "*"
    count  = skipped_dup = skipped_filter = 0
    pbar   = tqdm(total=max_articles, desc="OpenAlex", unit="paper", ncols=80)

    while count < max_articles:
        params = {
            "filter":   filter_str,
            "select":   OA_SELECT,
            "per-page": OA_PER_PAGE,
            "cursor":   cursor,
            "mailto":   OA_EMAIL,
            "sort":     "publication_year:desc",
        }

        try:
            r = requests.get(
                f"{OA_BASE}/works",
                params=params,
                timeout=30,
            )
            r.raise_for_status()
        except requests.HTTPError as e:
            log.error(f"OpenAlex HTTP error : {e}")
            time.sleep(10)
            continue
        except requests.RequestException as e:
            log.error(f"OpenAlex network error : {e}")
            time.sleep(5)
            break

        data    = r.json()
        results = data.get("results", [])

        if not results:
            log.info("OpenAlex : plus de resultats.")
            break

        for work in results:
            if count >= max_articles:
                break

            article = _oa_to_article(work)
            if article is None:
                skipped_filter += 1
                continue

            if manifest and manifest.contains(article.corpus_id):
                skipped_dup += 1
                continue

            if manifest:
                manifest.add(article.corpus_id)

            yield article
            count += 1
            pbar.update(1)

        # Pagination cursor
        meta   = data.get("meta") or {}
        cursor = meta.get("next_cursor")
        if not cursor:
            log.info("OpenAlex : fin de la pagination.")
            break

        time.sleep(OA_DELAY)

    pbar.close()
    log.info(
        f"OpenAlex : {count} collectes | "
        f"filtre={skipped_filter} | doublon={skipped_dup}"
    )

# =============================================================================
# PIPELINE
# =============================================================================

def collect(
    max_total: int = 500,
    output_dir: Path = RAW_DIR,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(MANIFEST_PATH)
    out_path = output_dir / "articles.jsonl"

    log.info(f"=== Collecte OpenAlex : max={max_total} ===")

    total = 0
    mode  = "a" if out_path.exists() else "w"

    with open(out_path, mode, encoding="utf-8") as f:
        for article in fetch_openalex(max_articles=max_total, manifest=manifest):
            f.write(article.model_dump_json(exclude={"raw_text", "embedding"}) + "\n")
            total += 1

    manifest.save()
    log.info(f"=== Termine : {total} articles ===")
    _summary(out_path)
    return total


def _summary(path: Path) -> None:
    if not path.exists():
        return
    years:  dict[int, int] = {}
    venues: dict[str, int] = {}
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                n += 1
                years[d.get("year", 0)] = years.get(d.get("year", 0), 0) + 1
                v = d.get("source", {}).get("venue", "?")
                venues[v] = venues.get(v, 0) + 1
            except Exception:
                continue
    print(f"\n{'='*55}")
    print(f"  CORPUS : {n} articles dans {path.name}")
    print(f"  Annees : {dict(sorted(years.items()))}")
    print(f"  Venues : {venues}")
    print(f"{'='*55}\n")


def load_articles(path: Path = ARTICLES_PATH) -> list[Article]:
    """Charge tous les Article depuis articles.jsonl."""
    arts = []
    if not path.exists():
        return arts
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                arts.append(Article.model_validate_json(line))
            except Exception:
                pass
    log.info(f"{len(arts)} articles charges.")
    return arts

# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="SciLit -- collecte corpus (OpenAlex)")
    p.add_argument("--max",    type=int,  default=500,   help="Nb articles (def: 500)")
    p.add_argument("--output", type=Path, default=RAW_DIR, help="Dossier de sortie")
    args = p.parse_args()
    n = collect(max_total=args.max, output_dir=args.output)
    print(f"\nTermine : {n} articles ajoutes.")

if __name__ == "__main__":
    main()