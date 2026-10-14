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
import json
import logging
import re
import tempfile
import time
from collections import Counter
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import requests
from tqdm import tqdm

from scilit import config
from scilit.schemas import Article, ArticleSource, Language, SplitType, Venue

# =============================================================================
# CONFIG
# =============================================================================

YEAR_MIN = 2019
YEAR_MAX = 2025

OA_BASE   = "https://api.openalex.org"
OA_EMAIL  = "omar.naceur01@gmail.com"   # polite pool = pas de limite stricte
OA_PER_PAGE = 200
OA_DELAY    = 0.15                      # ~6-7 req/s (limite = 10/s)

# Concept OpenAlex : Natural Language Processing
OA_CONCEPT_NLP = "C204321447"

# Champs a recuperer (evite de telecharger tout l'objet)
OA_SELECT = ",".join([
    "id", "title", "authorships", "publication_year",
    "abstract_inverted_index", "primary_location", "open_access", "ids",
    "locations", "doi", "cited_by_count", "referenced_works",
])

log = logging.getLogger(__name__)

# =============================================================================
# MANIFEST
# =============================================================================

class Manifest:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or config.paths().raw / "manifest.json"
        self.seen: set[str] = set()
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text("utf-8"))
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


def _extract_acl_id(work: dict) -> str | None:
    for location in work.get("locations", []):
        url = location.get("landing_page_url") or ""
        match = re.search(r"aclanthology\.org/(?:[^/?#]+/)?([^/?#]+)", url, re.I)
        if match:
            return match.group(1).removesuffix(".pdf")
    return None


def _extract_arxiv_id(work: dict) -> str | None:
    """Extrait l'ID arXiv depuis les locations ou ids."""
    pattern = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
    ids = work.get("ids") or {}
    raw_id = ids.get("arxiv") or ""
    match = pattern.search(raw_id)
    if match:
        return match.group(1)

    for loc in work.get("locations", []):
        url = loc.get("landing_page_url") or ""
        if "arxiv.org/abs/" in url:
            match = pattern.search(url)
            if match:
                return match.group(1)
    return None


def _acl_id_from_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    normalized_doi = re.sub(r"^https?://doi\.org/", "", doi.strip(), flags=re.I)
    match = re.match(r"^10\.18653/v1/(.+)$", normalized_doi, re.I)
    if not match:
        return None
    acl_id = match.group(1).strip().rstrip("/")
    if re.fullmatch(r"[a-z]\d{2}-\d{3,5}", acl_id, re.I):
        return f"{acl_id[0].upper()}{acl_id[1:]}"
    if re.fullmatch(r"\d{4}\.[a-z0-9-]+\.\d+", acl_id, re.I):
        return acl_id.lower()
    return None


def _venue_from_acl_id(acl_id: str) -> Venue:
    identifier = acl_id.lower()
    if ".emnlp-" in identifier or "findings-emnlp" in identifier:
        return Venue.EMNLP
    if ".naacl-" in identifier or "findings-naacl" in identifier:
        return Venue.NAACL
    if ".eacl-" in identifier or re.match(r"^e\d{2}-", identifier):
        return Venue.EACL
    if ".coling-" in identifier or re.match(r"^c\d{2}-", identifier):
        return Venue.COLING
    if re.match(r"^d\d{2}-", identifier):
        return Venue.CONLL
    if ".acl-" in identifier or "findings-acl" in identifier:
        return Venue.ACL
    if re.match(r"^n\d{2}-", identifier):
        return Venue.NAACL
    if re.match(r"^p\d{2}-", identifier):
        return Venue.ACL
    return Venue.OTHER


def _venue_from_name(name: str | None) -> Venue:
    value = (name or "").casefold()
    for needle, venue in (
        ("arxiv", Venue.ARXIV),
        ("emnlp", Venue.EMNLP),
        ("naacl", Venue.NAACL),
        ("eacl", Venue.EACL),
        ("conll", Venue.CONLL),
        ("coling", Venue.COLING),
        ("acl", Venue.ACL),
        ("neurips", Venue.NEURIPS),
        ("iclr", Venue.ICLR),
        ("icml", Venue.ICML),
    ):
        if needle in value:
            return venue
    return Venue.OTHER


def enrich_acl_ids(articles_path: Path | None = None) -> int:
    """Enrichit les articles existants depuis le DOI ACL sans modifier leurs IDs."""
    articles_path = articles_path or config.paths().raw / "articles.jsonl"
    if not articles_path.exists():
        raise FileNotFoundError(articles_path)

    migrated: list[str] = []
    updated_count = 0
    for line in articles_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        article = Article.model_validate_json(line)
        acl_id = _acl_id_from_doi(article.doi or article.source.doi)
        if acl_id:
            source_data = article.source.model_dump()
            source_data.update({
                "acl_id": acl_id,
                "venue": _venue_from_acl_id(acl_id),
                "url": f"https://aclanthology.org/{acl_id}",
                "pdf_url": f"https://aclanthology.org/{acl_id}.pdf",
                "license": "CC BY 4.0",
            })
            source = ArticleSource.model_validate(source_data)
            article = article.model_copy(update={"source": source})
            updated_count += 1
        elif article.source.acl_id and not re.fullmatch(
            r"(?:[A-Z]\d{2}-\d{3,5}|\d{4}\.[a-z0-9-]+\.\d+)",
            article.source.acl_id,
            re.I,
        ):
            source_data = article.source.model_dump()
            source_data.update({
                "acl_id": None,
                "venue": _venue_from_name(article.venue_name),
                "url": f"https://doi.org/{article.doi}" if article.doi else source_data["url"],
                "pdf_url": None,
                "license": article.license or "open access",
            })
            article = article.model_copy(
                update={"source": ArticleSource.model_validate(source_data)}
            )
            updated_count += 1
        migrated.append(article.model_dump_json(exclude={"raw_text", "embedding"}))

    _write_jsonl_atomically(articles_path, migrated)
    log.info("ACL IDs enrichis : %d articles.", updated_count)
    return updated_count


def _write_jsonl_atomically(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as temp:
        temp.write("".join(f"{row}\n" for row in rows))
        temp_path = Path(temp.name)
    temp_path.replace(path)


def fetch_openalex_licenses(
    openalex_ids: list[str],
    batch_size: int = 50,
    requester=None,
) -> dict[str, str | None]:
    """Récupère uniquement les champs de licence OpenAlex par lots d'IDs."""
    if not 1 <= batch_size <= 50:
        raise ValueError("batch_size doit être compris entre 1 et 50")
    requester = requester or requests.get
    licenses: dict[str, str | None] = {}
    for offset in range(0, len(openalex_ids), batch_size):
        batch = openalex_ids[offset : offset + batch_size]
        response = requester(
            f"{OA_BASE}/works",
            params={
                "filter": f"openalex_id:{'|'.join(batch)}",
                "select": "id,best_oa_location,primary_location",
                "per-page": batch_size,
                "mailto": OA_EMAIL,
            },
            timeout=30,
        )
        response.raise_for_status()
        for work in response.json().get("results", []):
            openalex_id = (work.get("id") or "").rsplit("/", 1)[-1]
            locations = (
                work.get("best_oa_location"),
                work.get("primary_location"),
            )
            licenses[openalex_id] = next(
                (
                    str(location["license"])
                    for location in locations
                    if location and location.get("license")
                ),
                None,
            )
        time.sleep(OA_DELAY)
    return licenses


def enrich_licenses(
    articles_path: Path | None = None,
    batch_size: int = 50,
    requester=None,
) -> dict[str, int]:
    """Migre la licence OpenAlex des articles existants en conservant leurs IDs."""
    articles_path = articles_path or config.paths().raw / "articles.jsonl"
    if not articles_path.exists():
        raise FileNotFoundError(articles_path)
    articles = [
        Article.model_validate_json(line)
        for line in articles_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [article.openalex_id for article in articles if article.openalex_id]
    licenses = fetch_openalex_licenses(ids, batch_size, requester)
    distribution: Counter[str] = Counter()
    migrated: list[str] = []
    for article in articles:
        license_name = licenses.get(article.openalex_id or "")
        if article.source.acl_id and not license_name:
            license_name = "cc-by-4.0"
        distribution[license_name or "unknown"] += 1
        migrated.append(
            article.model_copy(update={"license": license_name})
            .model_dump_json(exclude={"raw_text", "embedding"})
        )
    _write_jsonl_atomically(articles_path, migrated)
    log.info("Licences enrichies : %s.", dict(distribution))
    return dict(distribution)


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
    acl_id    = _extract_acl_id(work)
    oa_id     = (work.get("id") or "").replace("https://openalex.org/", "")

    primary = work.get("primary_location") or {}
    locations = work.get("locations") or []
    oa_info = work.get("open_access") or {}
    pdf_url = oa_info.get("oa_url") or primary.get("pdf_url")
    landing_url = primary.get("landing_page_url") or next(
        (location.get("landing_page_url") for location in locations
         if location.get("landing_page_url")),
        None,
    )
    if not (acl_id or arxiv_id or pdf_url or landing_url):
        return None

    if arxiv_id:
        base_url = f"https://arxiv.org/abs/{arxiv_id}"
        pdf_url  = pdf_url or f"https://arxiv.org/pdf/{arxiv_id}"
    else:
        base_url = landing_url or pdf_url

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
            venue=Venue.ACL if acl_id else venue,
            arxiv_id=arxiv_id,
            acl_id=acl_id,
            doi=(work.get("doi") or "").removeprefix("https://doi.org/") or None,
            url=base_url,         # type: ignore[arg-type]
            pdf_url=pdf_url,      # type: ignore[arg-type]
            license="open access",
        )
        return Article(
            id=uuid5(NAMESPACE_URL, f"https://openalex.org/{oa_id}"),
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
            language=Language.EN,
            source=source,
            openalex_id=oa_id or None,
            doi=(work.get("doi") or "").removeprefix("https://doi.org/") or None,
            cited_by_count=work.get("cited_by_count") or 0,
            referenced_works=[
                ref.rsplit("/", 1)[-1]
                for ref in work.get("referenced_works", [])
            ],
            venue_name=(
                (primary.get("source") or {}).get("display_name")
                or next(
                    (
                        (location.get("source") or {}).get("display_name")
                        for location in locations
                        if (location.get("source") or {}).get("display_name")
                    ),
                    None,
                )
            ),
            split=SplitType.TRAIN,
        )
    except Exception as e:
        log.debug(f"OA skip {oa_id}: {e}")
        return None

# =============================================================================
# COLLECTEUR OPENALEX
# =============================================================================

def fetch_openalex(
    max_articles: int = 1500,
    manifest: Manifest | None = None,
) -> Generator[Article, None, None]:
    """
    Collecte via OpenAlex cursor-pagination.
    Filtre : concept NLP, accès ouvert, 2019-2025.

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
            "sort":     "cited_by_count:desc",
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

            if manifest and manifest.contains(article.openalex_id or ""):
                skipped_dup += 1
                continue

            if manifest:
                if article.openalex_id:
                    manifest.add(article.openalex_id)

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
    max_total: int = 1500,
    output_dir: Path | None = None,
    fresh: bool = True,
) -> int:
    output_dir = output_dir or config.paths().raw
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    out_path = output_dir / "articles.jsonl"
    if fresh:
        out_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
    manifest = Manifest(manifest_path)

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


def load_articles(path: Path | None = None) -> list[Article]:
    """Charge tous les Article depuis articles.jsonl."""
    path = path or config.paths().raw / "articles.jsonl"
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
    raw_dir = config.paths().raw
    config.setup_logging(raw_dir / "collect_log.txt")
    p = argparse.ArgumentParser(description="SciLit -- collecte corpus (OpenAlex)")
    p.add_argument("--max", type=int, default=1500, help="Nb articles (def: 1500)")
    p.add_argument("--output", type=Path, default=raw_dir, help="Dossier de sortie")
    p.add_argument("--fresh", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--enrich-acl-ids", action="store_true")
    p.add_argument("--enrich-licenses", action="store_true")
    args = p.parse_args()
    if args.enrich_acl_ids or args.enrich_licenses:
        if args.enrich_acl_ids:
            print(f"ACL IDs enrichis : {enrich_acl_ids()}")
        if args.enrich_licenses:
            print(f"Licences : {enrich_licenses()}")
        return
    n = collect(max_total=args.max, output_dir=args.output, fresh=args.fresh)
    print(f"\nTermine : {n} articles ajoutes.")

if __name__ == "__main__":
    main()