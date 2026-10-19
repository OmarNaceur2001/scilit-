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
import time
from collections.abc import Generator
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit
from uuid import NAMESPACE_URL, uuid5

import requests
from tqdm import tqdm

from scilit import config
from scilit.schemas import (
    LICENSE_ID_FAMILY,
    Article,
    ArticleSource,
    Language,
    LicenseEvidence,
    LicenseFamily,
    LicenseId,
    LicenseInfo,
    ProvenanceLocation,
    ProvenanceProvider,
    ProvenanceRecord,
    ProvenanceRole,
    SplitType,
    Venue,
    normalize_doi_identifier,
    normalize_license_identifier,
)

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
    "best_oa_location", "locations", "doi", "cited_by_count",
    "referenced_works",
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


def _publication_locations(work: dict) -> list[dict]:
    """Return dictionary-valued primary and alternate locations without mutation."""
    if not isinstance(work, dict):
        return []

    locations: list[dict] = []
    primary = work.get("primary_location")
    if isinstance(primary, dict):
        locations.append(primary)

    alternate = work.get("locations")
    if isinstance(alternate, (list, tuple)):
        locations.extend(item for item in alternate if isinstance(item, dict))
    return locations


def _url_host(url: object) -> str:
    if not isinstance(url, str) or not url:
        return ""
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def _is_domain(host: str, domain: str) -> bool:
    return host == domain or host.endswith(f".{domain}")


def _is_aclanthology_url(url: object) -> bool:
    return _is_domain(_url_host(url), "aclanthology.org")


def _acl_policy_license(
    url: str,
    publication_year: int | None = None,
) -> LicenseId | None:
    """Identify ACL policy terms only for dated, recognized Anthology IDs."""
    if not _is_aclanthology_url(url):
        return None
    try:
        path = unquote(urlsplit(url).path).lower()
    except ValueError:
        return None

    modern_id = re.match(
        r"/((?:19|20)\d{2}\.(?:acl|emnlp|naacl)(?:-[a-z0-9]+)*\.\d+)"
        r"(?:\.pdf)?/?$",
        path,
    )
    legacy_id = re.match(r"/p(\d{2})-\d+(?:\.pdf)?/?$", path)
    if modern_id:
        year = int(path[1:5])
    elif legacy_id:
        short_year = int(legacy_id.group(1))
        year = 2000 + short_year if short_year < 90 else 1900 + short_year
    else:
        return None

    if publication_year is not None and publication_year != year:
        return None
    return LicenseId.CC_BY_NC_SA_3 if year < 2016 else LicenseId.CC_BY_4


def _parse_license(
    raw: str | None,
    landing_url: str | None = None,
    *,
    acl_policy_applies: bool | None = None,
    publication_year: int | None = None,
) -> LicenseInfo:
    """Normalize OpenAlex metadata and retain conditions and evidence."""
    family, spdx_id = normalize_license_identifier(raw)
    evidence = (
        LicenseEvidence.OPENALEX
        if family != LicenseFamily.UNKNOWN
        else LicenseEvidence.UNKNOWN
    )
    inference_source = None
    policy_conflict = None

    may_check_policy = acl_policy_applies is not False
    policy_id = (
        _acl_policy_license(landing_url, publication_year)
        if may_check_policy and landing_url
        else None
    )
    normalized = raw.strip() if isinstance(raw, str) else ""
    if normalized and policy_id is not None and family != LicenseFamily.UNKNOWN:
        expected_family = LICENSE_ID_FAMILY[policy_id]
        if family != expected_family or (
            spdx_id is not None and spdx_id != policy_id
        ):
            policy_conflict = (
                f"OpenAlex reports {raw!r}; ACL policy evidence indicates "
                f"{policy_id.value}. Explicit OpenAlex metadata was preserved."
            )
    elif not normalized and policy_id is not None:
        spdx_id = policy_id
        family = LICENSE_ID_FAMILY[policy_id]
        evidence = LicenseEvidence.ACL_POLICY
        inference_source = "https://aclanthology.org/faq/copyright/"

    return LicenseInfo(
        spdx_id=spdx_id,
        family=family,
        raw_string=raw if isinstance(raw, str) else None,
        source_url=landing_url,
        evidence=evidence,
        inference_source=inference_source,
        policy_conflict=policy_conflict,
    )


def _conference_from_text(value: object, *, acronyms: bool = True) -> Venue | None:
    """Recognize explicit conference names, not generic publisher names."""
    if not isinstance(value, str) or not value:
        return None
    text = value.lower()

    patterns: tuple[tuple[Venue, tuple[str, ...]], ...] = (
        (Venue.EMNLP, (r"\bemnlp\b", r"\bempirical methods (?:in|of) natural language processing\b")),
        (Venue.NAACL, (r"\bnaacl\b", r"\bnorth american chapter of the association for computational linguistics\b")),
        (Venue.COLING, (r"\bcoling\b", r"\binternational conference on computational linguistics\b")),
        (Venue.ICLR, (r"\biclr\b", r"\binternational conference on learning representations\b")),
        (Venue.NEURIPS, (r"\bneurips\b", r"\bneural information processing systems\b")),
        (Venue.ICML, (r"\bicml\b", r"\binternational conference on machine learning\b")),
    )
    for venue, venue_patterns in patterns:
        selected_patterns = venue_patterns if acronyms else venue_patterns[1:]
        if any(re.search(pattern, text) for pattern in selected_patterns):
            return venue

    # "Association for Computational Linguistics" alone is a publisher/host.
    # Require an ACL acronym or wording that identifies its annual meeting.
    if "anthology" not in text:
        if acronyms and re.search(r"\bacl\b", text):
            return Venue.ACL
        acl_meeting = (
            r"\b(?:annual meeting|meeting|conference|proceedings)\b"
            r".{0,100}\bassociation for computational linguistics\b"
        )
        if re.search(acl_meeting, text):
            return Venue.ACL
    return None


def _conference_from_url(url: object) -> Venue | None:
    if not isinstance(url, str) or not url:
        return None
    try:
        parsed = urlsplit(url)
    except ValueError:
        return None
    host = (parsed.hostname or "").lower()
    path = unquote(parsed.path).lower()

    # ACL Anthology IDs encode the proceedings series (e.g. 2023.emnlp-main.123).
    if _is_domain(host, "aclanthology.org"):
        event = re.search(
            r"(?:^|/)(?:19|20)\d{2}\.(acl|emnlp|naacl|coling)(?:[-./]|$)",
            path,
        )
        if event:
            return {
                "acl": Venue.ACL,
                "emnlp": Venue.EMNLP,
                "naacl": Venue.NAACL,
                "coling": Venue.COLING,
            }[event.group(1)]

    official_domains = (
        ("emnlp.org", Venue.EMNLP),
        ("naacl.org", Venue.NAACL),
        ("iclr.cc", Venue.ICLR),
        ("neurips.cc", Venue.NEURIPS),
        ("icml.cc", Venue.ICML),
    )
    for domain, venue in official_domains:
        if _is_domain(host, domain):
            return venue
    if host == "papers.nips.cc":
        return Venue.NEURIPS
    return None


def _location_venue(location: dict) -> Venue | None:
    source = location.get("source")
    if not isinstance(source, dict):
        source = {}

    # Display names describe a publication series. Host organizations are used
    # only for full conference names, never for bare publisher/organization names.
    venue = _conference_from_text(source.get("display_name"))
    if venue:
        return venue
    venue = _conference_from_url(location.get("landing_page_url"))
    if venue:
        return venue
    return _conference_from_text(
        source.get("host_organization_name"), acronyms=False
    )


def _is_arxiv_url(url: object) -> bool:
    return _is_domain(_url_host(url), "arxiv.org")


def _has_arxiv_evidence(work: dict, locations: list[dict]) -> bool:
    if _extract_arxiv_id(work):
        return True
    ids = work.get("ids")
    raw_arxiv_id = ids.get("arxiv") if isinstance(ids, dict) else None
    if _is_arxiv_url(raw_arxiv_id):
        return True
    for location in locations:
        source = location.get("source")
        if not isinstance(source, dict):
            source = {}
        display_name = source.get("display_name")
        if isinstance(display_name, str) and re.search(r"\barxiv\b", display_name, re.I):
            return True
        if _is_arxiv_url(location.get("landing_page_url")):
            return True
    return False


def _detect_venue(work: dict) -> Venue:
    """Prefer explicit venues across all locations; use arXiv only with evidence."""
    locations = _publication_locations(work)
    for location in locations:
        venue = _location_venue(location)
        if venue:
            return venue
    if isinstance(work, dict) and _has_arxiv_evidence(work, locations):
        return Venue.ARXIV
    return Venue.OTHER


def _extract_license_info(work: dict, venue: Venue, year: int) -> LicenseInfo:
    """Use the primary copy's license, falling back to OpenAlex's best OA copy."""
    primary = work.get("primary_location")
    if not isinstance(primary, dict):
        primary = {}
    best_oa = work.get("best_oa_location")
    if not isinstance(best_oa, dict):
        best_oa = {}

    acl_policy_venues = {Venue.ACL, Venue.EMNLP, Venue.NAACL}

    def acl_policy_supported(url: object) -> bool:
        return (
            isinstance(year, int)
            and venue in acl_policy_venues
            and _conference_from_url(url) == venue
            and isinstance(url, str)
            and _acl_policy_license(url, year) is not None
        )

    for location in (primary, best_oa):
        raw = location.get("license")
        if isinstance(raw, str) and raw.strip():
            landing_url = location.get("landing_page_url")
            return _parse_license(
                raw,
                landing_url if isinstance(landing_url, str) else None,
                acl_policy_applies=acl_policy_supported(landing_url),
                publication_year=year if isinstance(year, int) else None,
            )

    locations = [primary, best_oa, *_publication_locations(work)]
    landing_urls = [
        location["landing_page_url"]
        for location in locations
        if isinstance(location.get("landing_page_url"), str)
        and location["landing_page_url"]
    ]
    acl_url = next(
        (url for url in landing_urls if acl_policy_supported(url)), None
    )
    source_url = acl_url or next(iter(landing_urls), None)
    return _parse_license(
        None,
        source_url,
        acl_policy_applies=acl_url is not None,
        publication_year=year if isinstance(year, int) else None,
    )


def _extract_acl_id(work: dict) -> str | None:
    for location in _publication_locations(work):
        url = location.get("landing_page_url")
        if not isinstance(url, str) or not _is_domain(
            _url_host(url), "aclanthology.org"
        ):
            continue
        try:
            path = unquote(urlsplit(url).path).strip("/")
        except ValueError:
            continue
        if not path:
            continue
        article_id = path.split("/", 1)[0].removesuffix(".pdf")
        if article_id.lower() not in {"papers", "anthology"}:
            return article_id
    return None


def _extract_arxiv_id(work: dict) -> str | None:
    """Extrait l'ID arXiv depuis les locations ou ids."""
    pattern = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)", re.I)
    if not isinstance(work, dict):
        return None
    ids = work.get("ids")
    raw_id = ids.get("arxiv") if isinstance(ids, dict) else None
    if not isinstance(raw_id, str):
        raw_id = ""
    match = pattern.search(raw_id)
    if match:
        return match.group(1)

    for loc in _publication_locations(work):
        url = loc.get("landing_page_url")
        if isinstance(url, str) and _is_arxiv_url(url):
            match = pattern.search(url)
            if match:
                return match.group(1)
    return None


def _openalex_identity(work: dict) -> tuple[str | None, str | None]:
    """Return the supplied OpenAlex work ID and canonical URL, when available."""
    raw_id = work.get("id")
    if not isinstance(raw_id, str) or not raw_id.strip():
        return None, None
    raw_id = raw_id.strip()
    try:
        parsed = urlsplit(raw_id)
    except ValueError:
        parsed = None

    if parsed and parsed.scheme.lower() in {"http", "https"}:
        path = parsed.path.strip("/")
        if (
            (parsed.hostname or "").lower() == "openalex.org"
            and path
            and "/" not in path
            and not parsed.query
            and not parsed.fragment
        ):
            return path, raw_id
        return None, None
    if raw_id.startswith("W"):
        return raw_id, None
    return None, None


def _all_provenance_locations(work: dict) -> list[dict]:
    """Keep each distinct OpenAlex location as an independent copy record."""
    candidates = [
        work.get("primary_location"),
        work.get("best_oa_location"),
        *(work.get("locations") or []),
    ]
    result: list[dict] = []
    for location in candidates:
        if isinstance(location, dict) and location not in result:
            result.append(location)
    return result


def _location_license_info(
    location: dict,
    venue: Venue,
    publication_year: int,
) -> LicenseInfo:
    """Normalize only this location's license and its own landing-page evidence."""
    raw_license = location.get("license")
    if not isinstance(raw_license, str) or not raw_license.strip():
        raw_license = None
    landing_url = location.get("landing_page_url")
    if not isinstance(landing_url, str) or not landing_url.strip():
        landing_url = None

    acl_policy_venues = {Venue.ACL, Venue.EMNLP, Venue.NAACL}
    policy_applies = (
        venue in acl_policy_venues
        and _conference_from_url(landing_url) == venue
        and _acl_policy_license(landing_url, publication_year) is not None
    )
    return _parse_license(
        raw_license,
        landing_url,
        acl_policy_applies=policy_applies,
        publication_year=publication_year,
    )


def _provenance_record(
    work: dict,
    *,
    venue: Venue,
    publication_year: int,
    retrieved_at: datetime,
    citation_url: str,
    pdf_url: str | None,
    selected_license: LicenseInfo,
) -> ProvenanceRecord:
    provider_id, canonical_url = _openalex_identity(work)
    locations = _all_provenance_locations(work)
    license_location_index: int | None = None
    if selected_license.source_url:
        license_location_index = next(
            (
                index
                for index, location in enumerate(locations)
                if location.get("landing_page_url") == selected_license.source_url
                and _location_license_info(location, venue, publication_year)
                == selected_license
            ),
            None,
        )
    if license_location_index is None:
        license_location_index = next(
            (
                index
                for index, location in enumerate(locations)
                if (
                    _location_license_info(location, venue, publication_year)
                    == selected_license
                )
            ),
            None,
        )

    provenance_locations: list[ProvenanceLocation] = []
    selected_roles = {
        ProvenanceRole.CITATION_SOURCE: next(
            (
                index
                for index, location in enumerate(locations)
                if location.get("landing_page_url") == citation_url
            ),
            None,
        ),
        ProvenanceRole.PDF_SOURCE: next(
            (
                index
                for index, location in enumerate(locations)
                if pdf_url and location.get("pdf_url") == pdf_url
            ),
            None,
        ),
        ProvenanceRole.LICENSE_EVIDENCE: license_location_index,
    }

    for index, location in enumerate(locations):
        roles = [role for role, selected in selected_roles.items() if selected == index]
        landing_url = location.get("landing_page_url")
        location_pdf_url = location.get("pdf_url")
        version = location.get("version")
        location_id = location.get("id")
        provenance_locations.append(
            ProvenanceLocation(
                location_id=location_id if isinstance(location_id, str) else None,
                landing_page_url=(
                    landing_url if isinstance(landing_url, str) and landing_url else None
                ),
                pdf_url=(
                    location_pdf_url
                    if isinstance(location_pdf_url, str) and location_pdf_url
                    else None
                ),
                version=version if isinstance(version, str) else None,
                license=_location_license_info(location, venue, publication_year),
                roles=roles,
            )
        )

    return ProvenanceRecord(
        provider=ProvenanceProvider.OPENALEX,
        provider_record_id=provider_id,
        canonical_record_url=canonical_url,
        doi=normalize_doi_identifier(work.get("doi")),
        retrieved_at=retrieved_at,
        locations=provenance_locations,
    )


def _oa_to_article(
    work: dict,
    retrieved_at: datetime | None = None,
) -> Article | None:
    """Convertit un resultat OpenAlex en Article Pydantic."""
    if not isinstance(work, dict):
        return None
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
    oa_id, _ = _openalex_identity(work)
    if oa_id is None:
        raw_openalex_id = work.get("id")
        oa_id = (
            raw_openalex_id.replace("https://openalex.org/", "")
            if isinstance(raw_openalex_id, str)
            else ""
        )
    timestamp = retrieved_at or datetime.now(timezone.utc)
    normalized_doi = normalize_doi_identifier(work.get("doi"))
    license_info = _extract_license_info(work, venue, year)

    locations = _publication_locations(work)
    primary = locations[0] if locations else {}
    oa_info = work.get("open_access")
    if not isinstance(oa_info, dict):
        oa_info = {}
    pdf_url = oa_info.get("oa_url") or primary.get("pdf_url")

    # When OpenAlex marks arXiv primary but also records the recognized
    # proceedings location, use that publication landing page for provenance.
    venue_location = next(
        (
            location
            for location in locations
            if venue not in {Venue.ARXIV, Venue.OTHER}
            and _location_venue(location) == venue
            and isinstance(location.get("landing_page_url"), str)
            and not _is_arxiv_url(location.get("landing_page_url"))
        ),
        None,
    )
    landing_candidates = [
        location.get("landing_page_url")
        for location in locations
        if isinstance(location.get("landing_page_url"), str)
        and location.get("landing_page_url")
    ]
    publication_url = (
        venue_location.get("landing_page_url") if venue_location else None
    )
    non_arxiv_url = next(
        (url for url in landing_candidates if not _is_arxiv_url(url)), None
    )
    landing_url = publication_url or non_arxiv_url or next(
        iter(landing_candidates), None
    )
    if not (acl_id or arxiv_id or pdf_url or landing_url):
        return None

    if venue == Venue.ARXIV and arxiv_id:
        base_url = f"https://arxiv.org/abs/{arxiv_id}"
        pdf_url  = pdf_url or f"https://arxiv.org/pdf/{arxiv_id}"
    else:
        base_url = landing_url or (
            f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None
        ) or pdf_url

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
            acl_id=acl_id,
            doi=normalized_doi,
            url=base_url,         # type: ignore[arg-type]
            pdf_url=pdf_url,      # type: ignore[arg-type]
            license=license_info,
            retrieved_at=timestamp,
        )
        provenance = _provenance_record(
            work,
            venue=venue,
            publication_year=year,
            retrieved_at=timestamp,
            citation_url=base_url,
            pdf_url=pdf_url if isinstance(pdf_url, str) else None,
            selected_license=license_info,
        )
        return Article(
            id=uuid5(NAMESPACE_URL, f"https://openalex.org/{oa_id}"),
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
            language=Language.EN,
            source=source,
            provenance=[provenance],
            openalex_id=oa_id or None,
            doi=normalized_doi,
            cited_by_count=work.get("cited_by_count") or 0,
            referenced_works=[
                ref.rsplit("/", 1)[-1]
                for ref in work.get("referenced_works", [])
            ],
            venue_name=(
                (venue_location or primary).get("source", {}).get("display_name")
                if isinstance((venue_location or primary).get("source"), dict)
                else next(
                    (
                        location["source"].get("display_name")
                        for location in locations
                        if isinstance(location.get("source"), dict)
                        and location["source"].get("display_name")
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
            response_retrieved_at = datetime.now(timezone.utc)
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

            article = _oa_to_article(work, retrieved_at=response_retrieved_at)
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
    args = p.parse_args()
    n = collect(max_total=args.max, output_dir=args.output, fresh=args.fresh)
    print(f"\nTermine : {n} articles ajoutes.")

if __name__ == "__main__":
    main()
