import json
from datetime import datetime, timedelta, timezone
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest
import requests
from fastapi.testclient import TestClient
from scilit.corpus import parse as parse_module
from scilit.corpus.collect import (
    _extract_acl_id,
    _extract_arxiv_id,
    _oa_to_article,
    _parse_license,
    _reconstruct_abstract,
)
from scilit.corpus.index import rrf_fusion
from scilit.corpus.parse import (
    build_argument_parser,
    chunk_text,
    parse_abstract,
    resolve_full_text_mode,
    run_bronze,
    run_silver,
)
from scilit.extraction.citations import build_citations
from scilit.extraction.claims import _extract_metric
from scilit.generation.nli_verify import verify_review
from scilit.generation.template import _bibtex_key, _make_bibtex
from scilit.schemas import (
    Article,
    ArticleSource,
    BM25Result,
    Chunk,
    Claim,
    ClaimType,
    DenseResult,
    HedgingLevel,
    Language,
    LicenseEvidence,
    LicenseFamily,
    LicenseId,
    LicenseInfo,
    PermissionOutcome,
    ProvenanceLocation,
    ProvenanceProvider,
    ProvenanceRecord,
    ProvenanceRole,
    RankedResult,
    ReviewDraft,
    SectionType,
    Venue,
    VerificationStatus,
    VerifiedSentence,
)

from scilit import config


def make_article(index: int = 0, **values) -> Article:
    article_id = uuid5(NAMESPACE_URL, f"https://openalex.org/W{index}")
    defaults = {
        "id": article_id,
        "title": "A Study of Natural Language Processing",
        "authors": ["Ada Lovelace"],
        "year": 2022,
        "abstract": (
            "This work presents a reproducible natural language processing "
            "method and reports results on several benchmark datasets."
        ),
        "source": ArticleSource(
            venue=Venue.ACL,
            acl_id=f"2022.acl-long.{index}",
            url=f"https://aclanthology.org/2022.acl-long.{index}",
            license=LicenseInfo(
                raw_string="cc-by",
                source_url=f"https://aclanthology.org/2022.acl-long.{index}",
            ),
        ),
        "openalex_id": f"W{index}",
        "venue_name": "Annual Meeting of the ACL",
    }
    defaults.update(values)
    return Article(**defaults)


def _openalex_venue_work(
    *,
    primary_location=None,
    best_oa_location=None,
    locations=None,
    ids=None,
    title="A reproducible study of natural language processing",
):
    abstract_words = (
        "This paper describes a reproducible method for natural language processing "
        "and evaluates the approach on several benchmark datasets."
    ).split()
    return {
        "id": "https://openalex.org/WVENUE1",
        "title": title,
        "publication_year": 2023,
        "abstract_inverted_index": {
            word: [position] for position, word in enumerate(abstract_words)
        },
        "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
        "primary_location": primary_location,
        "best_oa_location": best_oa_location,
        "locations": locations or [],
        "ids": ids or {},
        "open_access": {},
    }


@pytest.mark.parametrize(
    (
        "raw",
        "expected_family",
        "expected_id",
        "redistribution",
        "commercial_use",
        "adaptations",
    ),
    [
        (
            "cc-by", LicenseFamily.CC_BY, None,
            PermissionOutcome.CONDITIONAL, PermissionOutcome.CONDITIONAL,
            PermissionOutcome.CONDITIONAL,
        ),
        (
            "cc-by-4.0", LicenseFamily.CC_BY, LicenseId.CC_BY_4,
            PermissionOutcome.CONDITIONAL, PermissionOutcome.CONDITIONAL,
            PermissionOutcome.CONDITIONAL,
        ),
        (
            "CC-BY-3.0", LicenseFamily.CC_BY, LicenseId.CC_BY_3,
            PermissionOutcome.CONDITIONAL, PermissionOutcome.CONDITIONAL,
            PermissionOutcome.CONDITIONAL,
        ),
        (
            "cc-by-nc", LicenseFamily.CC_BY_NC, None,
            PermissionOutcome.CONDITIONAL, PermissionOutcome.PROHIBITED,
            PermissionOutcome.CONDITIONAL,
        ),
        (
            "cc-by-sa", LicenseFamily.CC_BY_SA, None,
            PermissionOutcome.CONDITIONAL, PermissionOutcome.CONDITIONAL,
            PermissionOutcome.CONDITIONAL,
        ),
        (
            "cc-by-nd", LicenseFamily.CC_BY_ND, None,
            PermissionOutcome.CONDITIONAL, PermissionOutcome.CONDITIONAL,
            PermissionOutcome.PROHIBITED,
        ),
        (
            "cc-by-nc-sa", LicenseFamily.CC_BY_NC_SA, None,
            PermissionOutcome.CONDITIONAL, PermissionOutcome.PROHIBITED,
            PermissionOutcome.CONDITIONAL,
        ),
        (
            "cc-by-nc-nd", LicenseFamily.CC_BY_NC_ND, None,
            PermissionOutcome.CONDITIONAL, PermissionOutcome.PROHIBITED,
            PermissionOutcome.PROHIBITED,
        ),
        (
            "cc0", LicenseFamily.CC0, None,
            PermissionOutcome.PERMITTED, PermissionOutcome.PERMITTED,
            PermissionOutcome.PERMITTED,
        ),
        (
            "cc0-1.0", LicenseFamily.CC0, LicenseId.CC0,
            PermissionOutcome.PERMITTED, PermissionOutcome.PERMITTED,
            PermissionOutcome.PERMITTED,
        ),
        (
            "public-domain", LicenseFamily.PUBLIC_DOMAIN, None,
            PermissionOutcome.PERMITTED, PermissionOutcome.PERMITTED,
            PermissionOutcome.PERMITTED,
        ),
        (
            "apache-2.0", LicenseFamily.APACHE_2, LicenseId.APACHE_2,
            PermissionOutcome.CONDITIONAL, PermissionOutcome.CONDITIONAL,
            PermissionOutcome.CONDITIONAL,
        ),
        (
            "mit", LicenseFamily.MIT, LicenseId.MIT,
            PermissionOutcome.CONDITIONAL, PermissionOutcome.CONDITIONAL,
            PermissionOutcome.CONDITIONAL,
        ),
        (
            "all-rights-reserved", LicenseFamily.ALL_RIGHTS_RESERVED, None,
            PermissionOutcome.PROHIBITED, PermissionOutcome.PROHIBITED,
            PermissionOutcome.PROHIBITED,
        ),
        (
            "elsevier-user-license", LicenseFamily.RESTRICTED, None,
            PermissionOutcome.UNKNOWN, PermissionOutcome.UNKNOWN,
            PermissionOutcome.UNKNOWN,
        ),
    ],
)
def test_openalex_license_normalization(
    raw,
    expected_family,
    expected_id,
    redistribution,
    commercial_use,
    adaptations,
):
    license_info = _parse_license(raw)

    assert license_info.spdx_id == expected_id
    assert license_info.family == expected_family
    assert license_info.raw_string == raw
    assert license_info.is_verified is (expected_family != LicenseFamily.UNKNOWN)
    assert license_info.redistribution.outcome == redistribution
    assert license_info.commercial_use.outcome == commercial_use
    assert license_info.adaptations.outcome == adaptations


def test_license_conditions_describe_attribution_noncommercial_and_share_alike():
    cc_by = _parse_license("cc-by")
    cc_by_nc = _parse_license("cc-by-nc")
    cc_by_sa = _parse_license("cc-by-sa")
    cc_by_nd = _parse_license("cc-by-nd")
    cc_by_nc_sa = _parse_license("cc-by-nc-sa")

    assert "Attribution is required." in cc_by.redistribution.conditions
    assert "Use must be noncommercial." in cc_by_nc.redistribution.conditions
    assert "Distributed adaptations must use the same license terms." in (
        cc_by_sa.adaptations.conditions
    )
    assert cc_by_nd.adaptations.outcome == PermissionOutcome.PROHIBITED
    assert "The license does not permit distribution of adaptations." in (
        cc_by_nd.adaptations.conditions
    )
    assert "Use must be noncommercial." in cc_by_nc_sa.adaptations.conditions
    assert "Distributed adaptations must use the same license terms." in (
        cc_by_nc_sa.adaptations.conditions
    )


def test_missing_license_inference_is_conservative_and_policy_scoped():
    acl_url = "https://aclanthology.org/2023.acl-long.42/"
    inferred_acl = _parse_license(None, acl_url)
    pre_2016_acl = _parse_license(None, "https://aclanthology.org/P15-1001/")
    arxiv = _parse_license(None, "https://arxiv.org/abs/2304.01234")
    unidentified_acl = _parse_license(
        None,
        "https://aclanthology.org/2023.acl-long/",
    )
    third_party_acl = _parse_license(
        None,
        "https://aclanthology.org/2024.wassa.1/",
    )
    unknown_on_acl = _parse_license("some-unknown-thing", acl_url)
    missing = _parse_license(None)
    mismatched_date = _parse_license(
        None,
        acl_url,
        publication_year=2015,
    )
    conflicting_explicit = _parse_license("cc-by-nc", acl_url)
    legacy_source = ArticleSource(
        venue=Venue.ACL,
        acl_id="2023.acl-long.42",
        url=acl_url,
        license="open access",
    )

    assert inferred_acl.spdx_id == LicenseId.CC_BY_4
    assert inferred_acl.family == LicenseFamily.CC_BY
    assert inferred_acl.evidence == LicenseEvidence.ACL_POLICY
    assert inferred_acl.inference_source == "https://aclanthology.org/faq/copyright/"
    assert inferred_acl.is_verified is True
    assert inferred_acl.redistribution.outcome == PermissionOutcome.CONDITIONAL
    assert inferred_acl.source_url == acl_url
    assert pre_2016_acl.spdx_id == LicenseId.CC_BY_NC_SA_3
    assert pre_2016_acl.family == LicenseFamily.CC_BY_NC_SA
    assert pre_2016_acl.commercial_use.outcome == PermissionOutcome.PROHIBITED
    assert unidentified_acl.family == LicenseFamily.UNKNOWN
    assert third_party_acl.family == LicenseFamily.UNKNOWN
    assert mismatched_date.spdx_id is None
    assert mismatched_date.family == LicenseFamily.UNKNOWN
    assert arxiv.spdx_id is None
    assert arxiv.family == LicenseFamily.UNKNOWN
    assert arxiv.is_verified is False
    assert arxiv.redistribution.outcome == PermissionOutcome.UNKNOWN
    assert unknown_on_acl.spdx_id is None
    assert unknown_on_acl.family == LicenseFamily.UNKNOWN
    assert unknown_on_acl.raw_string == "some-unknown-thing"
    assert unknown_on_acl.is_verified is False
    assert unknown_on_acl.commercial_use.outcome == PermissionOutcome.UNKNOWN
    assert missing.spdx_id is None
    assert missing.family == LicenseFamily.UNKNOWN
    assert missing.is_verified is False
    assert missing.adaptations.outcome == PermissionOutcome.UNKNOWN
    assert conflicting_explicit.spdx_id is None
    assert conflicting_explicit.family == LicenseFamily.CC_BY_NC
    assert conflicting_explicit.evidence == LicenseEvidence.OPENALEX
    assert conflicting_explicit.inference_source is None
    assert conflicting_explicit.policy_conflict is not None
    assert conflicting_explicit.is_verified is False
    assert legacy_source.license.spdx_id is None
    assert legacy_source.license.family == LicenseFamily.UNKNOWN
    assert legacy_source.license.is_verified is False
    assert legacy_source.license.raw_string == "open access"


def test_legacy_structured_license_records_load_without_preserving_guessed_versions():
    legacy_openalex = LicenseInfo.model_validate(
        {
            "spdx_id": "CC-BY-4.0",
            "raw_string": "cc-by",
            "source_url": "https://doi.org/10.1234/example",
            "redistribution_permitted": True,
        }
    )
    legacy_acl_inference = LicenseInfo.model_validate(
        {
            "spdx_id": "CC-BY-4.0",
            "raw_string": None,
            "source_url": "https://aclanthology.org/2023.acl-long.42/",
            "redistribution_permitted": True,
        }
    )

    assert legacy_openalex.spdx_id is None
    assert legacy_openalex.family == LicenseFamily.CC_BY
    assert legacy_openalex.raw_string == "cc-by"
    assert legacy_openalex.is_verified is True
    assert legacy_acl_inference.spdx_id is None
    assert legacy_acl_inference.family == LicenseFamily.UNKNOWN
    assert legacy_acl_inference.is_verified is False


def test_collector_requires_policy_article_id_and_matching_publication_year():
    work = _openalex_venue_work(
        primary_location={
            "landing_page_url": "https://aclanthology.org/2023.acl-long.42/",
            "license": None,
            "source": {
                "display_name": "Proceedings of the Annual Meeting of the ACL"
            },
        }
    )
    work["publication_year"] = 2022

    article = _oa_to_article(work)

    assert article is not None
    assert article.source.license.spdx_id is None
    assert article.source.license.family == LicenseFamily.UNKNOWN
    assert article.source.license.evidence == LicenseEvidence.UNKNOWN
    assert article.source.license.is_verified is False


def test_article_license_uses_primary_location_then_best_oa_fallback():
    primary_url = "https://journal.example.org/article/primary"
    best_url = "https://repository.example.edu/article/best"
    primary_license_work = _openalex_venue_work(
        primary_location={
            "landing_page_url": primary_url,
            "license": "cc-by",
            "source": {"display_name": "Unknown Journal"},
        },
        best_oa_location={
            "landing_page_url": best_url,
            "license": "cc-by-nc",
        },
    )
    best_license_work = _openalex_venue_work(
        primary_location={
            "landing_page_url": primary_url,
            "license": None,
            "source": {"display_name": "Unknown Journal"},
        },
        best_oa_location={
            "landing_page_url": best_url,
            "license": "cc-by-nc",
        },
    )

    primary_article = _oa_to_article(primary_license_work)
    best_article = _oa_to_article(best_license_work)

    assert primary_article is not None and best_article is not None
    assert primary_article.source.license.spdx_id is None
    assert primary_article.source.license.family == LicenseFamily.CC_BY
    assert primary_article.source.license.source_url == primary_url
    assert best_article.source.license.spdx_id is None
    assert best_article.source.license.family == LicenseFamily.CC_BY_NC
    assert best_article.source.license.source_url == best_url
    assert best_article.source.license.commercial_use.outcome == PermissionOutcome.PROHIBITED


def test_provenance_record_validates_complete_openalex_evidence():
    retrieved_at = datetime(2025, 4, 3, 12, 30, tzinfo=timezone.utc)
    landing_url = "https://repository.example.org/article/123"
    license_info = _parse_license("cc-by-sa", landing_url)

    record = ProvenanceRecord(
        provider=ProvenanceProvider.OPENALEX,
        provider_record_id="W123456",
        canonical_record_url="https://openalex.org/W123456",
        doi="https://doi.org/10.1234/Example.DOI",
        retrieved_at=retrieved_at,
        locations=[
            ProvenanceLocation(
                location_id="https://openalex.org/locations/L123456",
                landing_page_url=landing_url,
                pdf_url="https://repository.example.org/article/123.pdf",
                version="publishedVersion",
                license=license_info,
                roles=[
                    ProvenanceRole.CITATION_SOURCE,
                    ProvenanceRole.LICENSE_EVIDENCE,
                ],
            )
        ],
    )

    assert record.provider_record_id == "W123456"
    assert str(record.canonical_record_url) == "https://openalex.org/W123456"
    assert record.doi == "10.1234/example.doi"
    assert record.retrieved_at == retrieved_at
    assert record.locations[0].location_id == "https://openalex.org/locations/L123456"
    assert record.locations[0].version == "publishedVersion"
    assert record.locations[0].license == license_info


def test_provenance_missing_optional_metadata_stays_absent():
    record = ProvenanceRecord(
        provider=ProvenanceProvider.OPENALEX,
        retrieved_at=datetime(2025, 4, 3, tzinfo=timezone.utc),
        locations=[ProvenanceLocation()],
    )

    assert record.provider_record_id is None
    assert record.canonical_record_url is None
    assert record.doi is None
    assert record.locations[0].location_id is None
    assert record.locations[0].landing_page_url is None
    assert record.locations[0].pdf_url is None
    assert record.locations[0].version is None
    assert record.locations[0].license is None


def test_provenance_retrieval_timestamps_must_be_aware_and_normalize_to_utc():
    with pytest.raises(ValueError, match="timezone-aware"):
        ProvenanceRecord(
            provider=ProvenanceProvider.OPENALEX,
            retrieved_at=datetime(2025, 4, 3, 12, 30),
        )

    offset_time = datetime(
        2025, 4, 3, 14, 30, tzinfo=timezone(timedelta(hours=2))
    )
    record = ProvenanceRecord(
        provider=ProvenanceProvider.OPENALEX,
        retrieved_at=offset_time,
    )
    assert record.retrieved_at == datetime(2025, 4, 3, 12, 30, tzinfo=timezone.utc)
    assert record.retrieved_at.utcoffset() == timedelta(0)


def test_provenance_keeps_multiple_location_licenses_and_roles_separate():
    citation_url = "https://publisher.example.org/paper/7"
    repository_url = "https://repository.example.edu/paper/7"
    repository_pdf = f"{repository_url}.pdf"
    publisher_license = _parse_license("cc-by-nd", citation_url)
    repository_license = _parse_license("cc-by-nc", repository_url)
    source = ArticleSource(
        venue=Venue.OTHER,
        url=citation_url,
        pdf_url=repository_pdf,
        license=repository_license,
    )
    provenance = ProvenanceRecord(
        provider=ProvenanceProvider.OPENALEX,
        provider_record_id="W777",
        retrieved_at=datetime(2025, 4, 3, tzinfo=timezone.utc),
        locations=[
            ProvenanceLocation(
                location_id="L-publisher",
                landing_page_url=citation_url,
                version="publishedVersion",
                license=publisher_license,
                roles=[ProvenanceRole.CITATION_SOURCE],
            ),
            ProvenanceLocation(
                location_id="L-repository",
                landing_page_url=repository_url,
                pdf_url=repository_pdf,
                version="acceptedVersion",
                license=repository_license,
                roles=[
                    ProvenanceRole.PDF_SOURCE,
                    ProvenanceRole.LICENSE_EVIDENCE,
                ],
            ),
        ],
    )
    article = make_article(777, source=source, provenance=[provenance])

    publisher, repository = article.provenance[0].locations
    assert str(publisher.landing_page_url) == citation_url
    assert publisher.license is not None
    assert publisher.license.raw_string == "cc-by-nd"
    assert str(repository.landing_page_url) == repository_url
    assert str(repository.pdf_url) == repository_pdf
    assert repository.license is not None
    assert repository.license.raw_string == "cc-by-nc"
    assert repository.license.commercial_use.outcome == PermissionOutcome.PROHIBITED
    assert str(article.source.url) == citation_url
    assert str(article.source.pdf_url) == repository_pdf

    mismatched_provenance = ProvenanceRecord(
        provider=ProvenanceProvider.OPENALEX,
        retrieved_at=datetime(2025, 4, 3, tzinfo=timezone.utc),
        locations=[
            ProvenanceLocation(
                landing_page_url=repository_url,
                license=publisher_license,
                roles=[ProvenanceRole.CITATION_SOURCE],
            )
        ],
    )
    with pytest.raises(ValueError, match="citation-source role"):
        make_article(778, source=source, provenance=[mismatched_provenance])


def test_oa_to_article_integrates_provider_timestamp_and_location_evidence():
    primary = {
        "id": "https://openalex.org/locations/LPRIMARY",
        "landing_page_url": "https://journal.example.org/article/primary",
        "pdf_url": "https://journal.example.org/article/primary.pdf",
        "version": "publishedVersion",
        "license": "cc-by",
        "source": {"display_name": "Unknown Journal"},
    }
    repository = {
        "id": "https://openalex.org/locations/LREPOSITORY",
        "landing_page_url": "https://repo.example.edu/article/primary",
        "pdf_url": "https://repo.example.edu/article/primary.pdf",
        "version": "acceptedVersion",
        "license": "cc-by-nc",
    }
    work = _openalex_venue_work(
        primary_location=primary,
        best_oa_location=repository,
        locations=[primary, repository],
    )
    work["doi"] = "https://doi.org/10.1234/Example.DOI"
    retrieved_at = datetime(2025, 4, 3, 12, 30, tzinfo=timezone.utc)

    article = _oa_to_article(work, retrieved_at=retrieved_at)

    assert article is not None
    assert article.openalex_id == "WVENUE1"
    assert article.doi == "10.1234/example.doi"
    assert article.source.doi == article.doi
    assert article.source.retrieved_at == retrieved_at
    assert len(article.provenance) == 1
    record = article.provenance[0]
    assert record.provider == ProvenanceProvider.OPENALEX
    assert record.provider_record_id == "WVENUE1"
    assert str(record.canonical_record_url) == "https://openalex.org/WVENUE1"
    assert record.retrieved_at == retrieved_at
    assert len(record.locations) == 2

    primary_copy, repository_copy = record.locations
    assert primary_copy.location_id == primary["id"]
    assert str(primary_copy.landing_page_url) == primary["landing_page_url"]
    assert str(primary_copy.pdf_url) == primary["pdf_url"]
    assert primary_copy.version == "publishedVersion"
    assert primary_copy.license == article.source.license
    assert primary_copy.roles == [
        ProvenanceRole.CITATION_SOURCE,
        ProvenanceRole.PDF_SOURCE,
        ProvenanceRole.LICENSE_EVIDENCE,
    ]
    assert str(repository_copy.landing_page_url) == repository["landing_page_url"]
    assert str(repository_copy.pdf_url) == repository["pdf_url"]
    assert repository_copy.version == "acceptedVersion"
    assert repository_copy.license is not None
    assert repository_copy.license.raw_string == "cc-by-nc"
    assert repository_copy.license.family == LicenseFamily.CC_BY_NC
    assert repository_copy.roles == []


def test_legacy_article_json_loads_without_fabricated_provenance_or_license():
    legacy = make_article(778).model_dump(mode="json")
    legacy.pop("provenance", None)
    legacy["source"]["license"] = "open access"

    article = Article.model_validate(legacy)

    assert article.provenance == []
    assert article.source.license.raw_string == "open access"
    assert article.source.license.family == LicenseFamily.UNKNOWN
    assert article.source.license.is_verified is False


def test_article_provenance_license_evidence_round_trips():
    inferred_license = _parse_license(
        None,
        "https://aclanthology.org/2023.acl-long.42/",
        acl_policy_applies=True,
        publication_year=2023,
    )
    article = make_article(
        779,
        source=ArticleSource(
            venue=Venue.ACL,
            acl_id="2023.acl-long.42",
            url="https://aclanthology.org/2023.acl-long.42/",
            license=inferred_license,
        ),
        provenance=[
            ProvenanceRecord(
                provider=ProvenanceProvider.OPENALEX,
                provider_record_id="W779",
                canonical_record_url="https://openalex.org/W779",
                retrieved_at=datetime(2025, 4, 3, tzinfo=timezone.utc),
                locations=[
                    ProvenanceLocation(
                        location_id="L779",
                        landing_page_url=(
                            "https://aclanthology.org/2023.acl-long.42/"
                        ),
                        license=inferred_license,
                        roles=[
                            ProvenanceRole.CITATION_SOURCE,
                            ProvenanceRole.LICENSE_EVIDENCE,
                        ],
                    )
                ],
            )
        ],
    )

    round_tripped = Article.model_validate_json(article.model_dump_json())
    round_trip_license = round_tripped.provenance[0].locations[0].license

    assert round_tripped.provenance == article.provenance
    assert round_trip_license is not None
    assert round_trip_license.spdx_id == LicenseId.CC_BY_4
    assert round_trip_license.evidence == LicenseEvidence.ACL_POLICY
    assert round_trip_license.inference_source == (
        "https://aclanthology.org/faq/copyright/"
    )
    assert round_trip_license.redistribution.outcome == PermissionOutcome.CONDITIONAL
    assert round_trip_license.adaptations == article.provenance[0].locations[0].license.adaptations


def test_unknown_publisher_is_not_guessed_to_be_arxiv():
    work = _openalex_venue_work(
        primary_location={
            "landing_page_url": "https://journal.example.org/article/123",
            "source": {
                "display_name": "Journal of Computational Methods",
                "host_organization_name": "Example Research Press",
            },
        },
        # Ignore malformed location entries without breaking conversion.
        locations=[None, "not a location", {"source": None}],
    )

    article = _oa_to_article(work)

    assert article is not None
    assert article.source.venue == Venue.OTHER
    assert article.source.arxiv_id is None


def test_emnlp_location_takes_precedence_over_primary_arxiv_location():
    expected_arxiv_id = "2304.01234"
    expected_acl_id = "2023.emnlp-main.123"
    proceedings_name = (
        "Proceedings of the 2023 Conference on Empirical Methods in "
        "Natural Language Processing"
    )
    work = _openalex_venue_work(
        primary_location={
            "landing_page_url": f"https://arxiv.org/abs/{expected_arxiv_id}",
            "source": {"display_name": "arXiv", "host_organization_name": "Cornell University"},
        },
        locations=[
            {
                "landing_page_url": f"https://aclanthology.org/{expected_acl_id}/",
                "source": {
                    "display_name": proceedings_name,
                    "host_organization_name": "Association for Computational Linguistics",
                },
            }
        ],
        ids={"arxiv": f"https://arxiv.org/abs/{expected_arxiv_id}"},
    )

    article = _oa_to_article(work)

    assert article is not None
    assert article.source.venue == Venue.EMNLP
    assert article.source.arxiv_id == expected_arxiv_id
    assert article.source.acl_id == expected_acl_id
    assert str(article.source.url) == f"https://aclanthology.org/{expected_acl_id}/"
    assert article.venue_name == proceedings_name
    assert article.source.license.spdx_id == LicenseId.CC_BY_4
    assert article.source.license.family == LicenseFamily.CC_BY
    assert article.source.license.evidence == LicenseEvidence.ACL_POLICY
    assert article.source.license.inference_source is not None
    assert article.source.license.redistribution.outcome == PermissionOutcome.CONDITIONAL
    assert article.source.license.source_url == (
        f"https://aclanthology.org/{expected_acl_id}/"
    )


def test_acl_requires_conference_evidence_not_only_publisher_organization():
    acl_work = _openalex_venue_work(
        primary_location={
            "landing_page_url": "https://aclanthology.org/2023.acl-long.42/",
            "source": {
                "display_name": (
                    "Proceedings of the 61st Annual Meeting of the "
                    "Association for Computational Linguistics"
                ),
                "host_organization_name": "Association for Computational Linguistics",
            },
        }
    )
    publisher_only_work = _openalex_venue_work(
        primary_location={
            "landing_page_url": "https://journal.example.org/article/456",
            "source": {
                "display_name": "Journal of Computational Linguistics",
                "host_organization_name": "Association for Computational Linguistics",
            },
        }
    )

    acl_article = _oa_to_article(acl_work)
    publisher_only_article = _oa_to_article(publisher_only_work)

    assert acl_article is not None
    assert acl_article.source.venue == Venue.ACL
    assert acl_article.source.license.spdx_id == LicenseId.CC_BY_4
    assert acl_article.source.license.evidence == LicenseEvidence.ACL_POLICY
    assert acl_article.source.license.redistribution.outcome == PermissionOutcome.CONDITIONAL
    assert publisher_only_article is not None
    assert publisher_only_article.source.venue == Venue.OTHER


def test_explicit_acl_license_is_preserved_and_policy_conflict_is_visible():
    acl_url = "https://aclanthology.org/2023.acl-long.42/"
    work = _openalex_venue_work(
        primary_location={
            "landing_page_url": acl_url,
            "license": "cc-by-nc",
            "source": {
                "display_name": "Proceedings of the Annual Meeting of the ACL"
            },
        }
    )

    article = _oa_to_article(work)

    assert article is not None
    assert article.source.license.raw_string == "cc-by-nc"
    assert article.source.license.family == LicenseFamily.CC_BY_NC
    assert article.source.license.spdx_id is None
    assert article.source.license.evidence == LicenseEvidence.OPENALEX
    assert article.source.license.inference_source is None
    assert article.source.license.policy_conflict is not None
    assert article.source.license.is_verified is False


def test_genuine_arxiv_record_keeps_venue_and_identifier():
    expected_arxiv_id = "2401.01234"
    work = _openalex_venue_work(
        primary_location={
            "landing_page_url": f"https://arxiv.org/abs/{expected_arxiv_id}",
            "source": {"display_name": "arXiv"},
        },
        ids={"arxiv": expected_arxiv_id},
    )

    article = _oa_to_article(work)

    assert article is not None
    assert article.source.venue == Venue.ARXIV
    assert article.source.arxiv_id == expected_arxiv_id
    assert article.source.license.spdx_id is None
    assert article.source.license.family == LicenseFamily.UNKNOWN
    assert article.source.license.is_verified is False
    assert article.source.license.redistribution.outcome == PermissionOutcome.UNKNOWN


@pytest.mark.parametrize(
    ("display_name", "expected_venue"),
    [
        (
            "Proceedings of the North American Chapter of the Association "
            "for Computational Linguistics",
            Venue.NAACL,
        ),
        (
            "Proceedings of the International Conference on Computational Linguistics",
            Venue.COLING,
        ),
        ("International Conference on Learning Representations", Venue.ICLR),
        ("Advances in Neural Information Processing Systems", Venue.NEURIPS),
        ("International Conference on Machine Learning", Venue.ICML),
    ],
)
def test_other_recognized_conference_names(display_name, expected_venue):
    work = _openalex_venue_work(
        primary_location={
            "landing_page_url": "https://proceedings.example.org/paper/1",
            "source": {"display_name": display_name},
        }
    )

    article = _oa_to_article(work)

    assert article is not None
    assert article.source.venue == expected_venue


def test_unknown_acl_anthology_event_does_not_become_acl():
    anthology_id = "2024.wassa.1"
    work = _openalex_venue_work(
        primary_location={
            "landing_page_url": f"https://aclanthology.org/{anthology_id}/",
            "source": {
                "display_name": "ACL Anthology",
                "host_organization_name": "Association for Computational Linguistics",
            },
        }
    )

    article = _oa_to_article(work)

    assert article is not None
    assert article.source.venue == Venue.OTHER
    assert article.source.acl_id == anthology_id
    assert article.source.license.spdx_id is None
    assert article.source.license.family == LicenseFamily.UNKNOWN
    assert article.source.license.is_verified is False
    assert article.source.license.redistribution.outcome == PermissionOutcome.UNKNOWN


def test_openalex_abstract_and_identifier_helpers():
    assert Language.AR.value == "ar"
    assert _reconstruct_abstract({"second": [1], "first": [0]}) == "first second"
    assert _reconstruct_abstract(None) == ""
    assert _extract_arxiv_id({
        "ids": {"arxiv": "https://arxiv.org/abs/2304.01234v3"}
    }) == "2304.01234"
    assert _extract_acl_id({
        "locations": [
            {"landing_page_url": "https://aclanthology.org/2023.acl-long.42/"}
        ]
    }) == "2023.acl-long.42"
    metric_name, metric_value = _extract_metric("The model reaches an F1 score of 92.5.")
    assert metric_name is not None
    assert metric_value == 92.5


def test_collected_article_id_is_stable_openalex_uuid():
    work = {
        "id": "https://openalex.org/W55",
        "title": "A natural language processing paper",
        "publication_year": 2023,
        "abstract_inverted_index": {
            "This": [0],
            "work": [1],
            "contains": [2],
            "a": [3],
            "sufficiently": [4],
            "long": [5],
            "abstract": [6],
            "about": [7],
            "natural": [8],
            "language": [9],
            "processing": [10],
            "research": [11],
            "methods": [12],
            "and": [13],
            "experimental": [14],
            "results": [15],
        },
        "primary_location": {
            "landing_page_url": "https://example.org/paper",
            "pdf_url": "https://example.org/paper.pdf",
            "source": {"display_name": "Example NLP venue"},
        },
        "open_access": {"oa_url": "https://example.org/paper.pdf"},
        "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
        "referenced_works": ["https://openalex.org/W44"],
        "cited_by_count": 7,
    }
    first = _oa_to_article(work)
    second = _oa_to_article(work)
    assert first is not None and second is not None
    assert first.id == uuid5(NAMESPACE_URL, "https://openalex.org/W55")
    assert first.id == second.id
    assert first.referenced_works == ["W44"]


def test_abstract_sections_chunks_and_ids_are_deterministic():
    article = make_article(10)
    sections_a, chunks_a = parse_abstract(article)
    sections_b, chunks_b = parse_abstract(article)
    assert sections_a[0].id == uuid5(article.id, "section:0")
    assert [chunk.id for chunk in chunks_a] == [chunk.id for chunk in chunks_b]
    assert all(
        chunk.id == uuid5(sections_a[0].id, f"chunk:{chunk.chunk_index}")
        for chunk in chunks_a
    )
    assert sections_a[0].id == sections_b[0].id


def test_full_text_mode_defaults_to_abstract_and_keeps_legacy_cli_options(
    tmp_path, monkeypatch
):
    parser = build_argument_parser()
    defaults = parser.parse_args([])
    assert defaults.mode is None
    assert defaults.full_text_mode is None
    assert resolve_full_text_mode(defaults.mode, defaults.full_text_mode) is False

    legacy_abstract = parser.parse_args(["--mode", "abstract"])
    legacy_pdf = parser.parse_args(["--mode", "pdf"])
    explicit_full_text = parser.parse_args(["--full-text-mode"])
    explicit_abstract = parser.parse_args(["--no-full-text-mode"])
    assert resolve_full_text_mode(legacy_abstract.mode, None) is False
    assert resolve_full_text_mode(legacy_pdf.mode, None) is True
    assert resolve_full_text_mode(None, explicit_full_text.full_text_mode) is True
    assert resolve_full_text_mode(None, explicit_abstract.full_text_mode) is False

    with pytest.raises(ValueError, match="conflicting parsing modes"):
        resolve_full_text_mode("pdf", False)
    with pytest.raises(ValueError, match="conflicting parsing modes"):
        resolve_full_text_mode("abstract", True)

    monkeypatch.setattr(
        parse_module,
        "_download_pdf",
        lambda *_args, **_kwargs: pytest.fail("abstract mode attempted a PDF download"),
    )
    article = make_article(340, abstract="ABSTRACT_ONLY_SENTINEL " + "Evidence. " * 20)
    sections_path = tmp_path / "sections.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    section_count, chunk_count = run_bronze(
        [article], sections_path=sections_path, chunks_path=chunks_path
    )
    assert section_count == 1
    assert chunk_count > 0
    saved = [json.loads(line) for line in sections_path.read_text().splitlines()]
    assert saved[0]["section_type"] == SectionType.ABSTRACT.value


def test_legacy_article_records_still_load_and_parse(tmp_path):
    record = make_article(341).model_dump(mode="json")
    record.pop("provenance", None)
    record["source"]["license"] = "cc-by"
    article = Article.model_validate(record)

    section_count, _ = run_bronze(
        [article],
        sections_path=tmp_path / "legacy-sections.jsonl",
        chunks_path=tmp_path / "legacy-chunks.jsonl",
    )
    assert section_count == 1


def _temporary_data_paths(tmp_path):
    data = tmp_path / "data"
    return config.DataPaths(
        root=tmp_path,
        data=data,
        raw=data / "raw",
        parsed=data / "parsed",
        reviews=data / "parsed" / "reviews",
    )


def _article_with_pdf(index: int, url: str = "https://example.org/paper.pdf"):
    article = make_article(
        index,
        abstract="ABSTRACT_ONLY_SENTINEL. This is only the provider abstract.",
    )
    source = article.source.model_copy(update={"pdf_url": url})
    return article.model_copy(update={"source": source})


def test_full_text_mode_extracts_only_successful_cached_pdf_text(
    tmp_path, monkeypatch, caplog
):
    import fitz

    paths = _temporary_data_paths(tmp_path)
    monkeypatch.setattr(parse_module.config, "paths", lambda: paths)
    monkeypatch.setattr(parse_module, "PDF_DELAY", 0)
    monkeypatch.setattr(
        parse_module.requests,
        "get",
        lambda *_args, **_kwargs: pytest.fail("cached PDF triggered a network request"),
    )

    article = _article_with_pdf(342)
    pdf_path = paths.raw / "pdfs" / f"{article.corpus_id}.pdf"
    pdf_path.parent.mkdir(parents=True)
    document = fitz.open()
    for page_number in range(3):
        page = document.new_page()
        if page_number == 0:
            page.insert_text((72, 72), "1 Introduction", fontsize=18)
        body = (
            "FULLTEXT_SENTINEL This full-text body describes a reproducible method. "
            * 18
        )
        page.insert_textbox((72, 100, 520, 740), body, fontsize=10)
    document.save(str(pdf_path))
    document.close()
    assert pdf_path.stat().st_size > 1000

    sections_path = paths.parsed / "sections.jsonl"
    chunks_path = paths.parsed / "chunks.jsonl"
    with caplog.at_level("INFO", logger=parse_module.__name__):
        section_count, chunk_count = run_silver(
            [article],
            max_articles=1,
            sections_path=sections_path,
            chunks_path=chunks_path,
        )

    saved = [json.loads(line) for line in sections_path.read_text().splitlines()]
    assert section_count == len(saved) > 0
    assert chunk_count > 0
    assert any("FULLTEXT_SENTINEL" in section["text"] for section in saved)
    assert all("ABSTRACT_ONLY_SENTINEL" not in section["text"] for section in saved)
    assert "processed_full_text=1" in caplog.text
    assert "processed_abstract=0" in caplog.text


def test_full_text_mode_skips_articles_without_pdf_without_abstract_fallback(
    tmp_path, monkeypatch, caplog
):
    paths = _temporary_data_paths(tmp_path)
    monkeypatch.setattr(parse_module.config, "paths", lambda: paths)
    monkeypatch.setattr(parse_module, "PDF_DELAY", 0)
    article = make_article(
        343,
        abstract="ABSTRACT_ONLY_SENTINEL. This cannot substitute for a missing PDF.",
    )
    sections_path = paths.parsed / "sections.jsonl"
    chunks_path = paths.parsed / "chunks.jsonl"

    with caplog.at_level("INFO", logger=parse_module.__name__):
        section_count, chunk_count = run_silver(
            [article],
            max_articles=1,
            sections_path=sections_path,
            chunks_path=chunks_path,
        )

    assert (section_count, chunk_count) == (0, 0)
    assert sections_path.read_text() == ""
    assert chunks_path.read_text() == ""
    assert "skipped_no_pdf=1" in caplog.text
    assert "processed_full_text=0" in caplog.text


def test_full_text_download_failure_is_recorded_without_abstract_fallback(
    tmp_path, monkeypatch, caplog
):
    paths = _temporary_data_paths(tmp_path)
    monkeypatch.setattr(parse_module.config, "paths", lambda: paths)
    monkeypatch.setattr(parse_module, "PDF_DELAY", 0)
    monkeypatch.setattr(parse_module, "_download_pdf", lambda *_args: False)
    article = _article_with_pdf(345)
    sections_path = paths.parsed / "sections.jsonl"
    chunks_path = paths.parsed / "chunks.jsonl"

    with caplog.at_level("INFO", logger=parse_module.__name__):
        section_count, chunk_count = run_silver(
            [article],
            max_articles=1,
            sections_path=sections_path,
            chunks_path=chunks_path,
        )

    assert (section_count, chunk_count) == (0, 0)
    assert sections_path.read_text() == ""
    assert chunks_path.read_text() == ""
    assert "reason=pdf_download_failed" in caplog.text
    assert "failed=1" in caplog.text
    assert "skipped_no_pdf=0" in caplog.text
    assert "processed_abstract=0" in caplog.text


@pytest.mark.parametrize(
    "failure", ["timeout", "http", "oversized", "oversized_cached"]
)
def test_pdf_download_failures_are_cleaned_up(tmp_path, monkeypatch, failure):
    destination = tmp_path / "pdfs" / "failed.pdf"

    class FakeResponse:
        def raise_for_status(self):
            if failure == "http":
                raise requests.HTTPError("test HTTP error")

        def iter_content(self, chunk_size):
            assert chunk_size == 65536
            yield b"oversized" if failure == "oversized" else b"small"

    def fake_get(_url, *, timeout, stream):
        assert timeout == parse_module.PDF_TIMEOUT
        assert stream is True
        if failure == "timeout":
            raise requests.Timeout("test timeout")
        if failure == "oversized_cached":
            pytest.fail("oversized cached PDF should be rejected before download")
        return FakeResponse()

    monkeypatch.setattr(parse_module.requests, "get", fake_get)
    if failure in {"oversized", "oversized_cached"}:
        monkeypatch.setattr(parse_module, "MAX_PDF_SIZE", 4)
    if failure == "oversized_cached":
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"oversized")

    assert parse_module._download_pdf("https://example.org/paper.pdf", destination) is False
    assert not destination.exists()


def test_malformed_cached_pdf_is_failed_and_never_replaced_by_abstract(
    tmp_path, monkeypatch, caplog
):
    paths = _temporary_data_paths(tmp_path)
    monkeypatch.setattr(parse_module.config, "paths", lambda: paths)
    monkeypatch.setattr(parse_module, "PDF_DELAY", 0)
    article = _article_with_pdf(344)
    pdf_path = paths.raw / "pdfs" / f"{article.corpus_id}.pdf"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"not a PDF " * 200)
    sections_path = paths.parsed / "sections.jsonl"
    chunks_path = paths.parsed / "chunks.jsonl"

    with caplog.at_level("INFO", logger=parse_module.__name__):
        section_count, chunk_count = run_silver(
            [article],
            max_articles=1,
            sections_path=sections_path,
            chunks_path=chunks_path,
        )

    assert (section_count, chunk_count) == (0, 0)
    assert not pdf_path.exists()
    assert sections_path.read_text() == ""
    assert "reason=no_extractable_sections" in caplog.text
    assert "failed=1" in caplog.text
    assert "processed_abstract=0" in caplog.text


def test_chunk_text_respects_overlap_and_repeatable_ids():
    article_id = uuid5(NAMESPACE_URL, "article")
    section_id = uuid5(article_id, "section:3")
    text = "Sentence with enough words to be retained. " * 20
    chunks = chunk_text(
        text,
        article_id,
        section_id,
        SectionType.ABSTRACT,
        size=150,
        overlap=30,
    )
    repeated = chunk_text(
        text,
        article_id,
        section_id,
        SectionType.ABSTRACT,
        size=150,
        overlap=30,
    )
    assert len(chunks) > 1
    assert [chunk.id for chunk in chunks] == [chunk.id for chunk in repeated]
    assert all(chunk.char_end > chunk.char_start for chunk in chunks)
    assert all(len(chunk.text) >= 30 for chunk in chunks)
    assert all(text[chunk.char_start:chunk.char_end] == chunk.text for chunk in chunks)
    assert all(
        chunk.char_start == 0 or text[chunk.char_start - 1].isspace()
        for chunk in chunks
    )


def test_claim_accepts_python_object_name_and_aware_datetime():
    article = make_article(11)
    chunk_id = uuid5(article.id, "chunk")
    claim = Claim(
        id=uuid5(chunk_id, "claim:7"),
        article_id=article.id,
        chunk_id=chunk_id,
        section_type=SectionType.ABSTRACT,
        text="The model improves the score.",
        verbatim="The model improves the score.",
        claim_type=ClaimType.FINDING,
        hedging_level=HedgingLevel.CERTAIN,
        object_="the score",
    )
    assert claim.object_ == "the score"
    assert article.source.retrieved_at.utcoffset().total_seconds() == 0


def test_bibtex_entry_types_escape_latex_and_include_metadata():
    article = make_article(
        12,
        title="A & $ # { } ~ ^ \\ Results_100%",
        authors=["Ada Lovelace & Co."],
        doi="10.1000/example",
    ).model_dump(mode="json")
    acl_entry = _make_bibtex(article)
    assert acl_entry.entry_type == "inproceedings"
    assert r"\&" in acl_entry.to_bibtex()
    assert r"\_" in acl_entry.to_bibtex()
    assert r"\%" in acl_entry.to_bibtex()
    for escaped in (
        r"\$",
        r"\#",
        r"\{",
        r"\}",
        r"\textasciitilde{}",
        r"\textasciicircum{}",
        r"\textbackslash{}",
    ):
        assert escaped in acl_entry.to_bibtex()
    assert "booktitle" in acl_entry.to_bibtex()
    assert "doi" in acl_entry.to_bibtex()
    assert _bibtex_key({
        "authors": ["Lovelace, Ada"],
        "year": 2023,
        "title": "The Évaluation of Methods",
    }) == "Lovelace2023Evaluation"

    arxiv_article = dict(article)
    arxiv_article["source"] = {
        "venue": "arxiv",
        "arxiv_id": "2304.01234",
        "url": "https://arxiv.org/abs/2304.01234",
    }
    arxiv_entry = _make_bibtex(arxiv_article)
    assert arxiv_entry.entry_type == "misc"
    assert "archivePrefix = {arXiv}" in arxiv_entry.to_bibtex()


def test_rrf_fusion_scores_and_ties_are_deterministic():
    article_id = UUID(int=2)
    chunk_ids = [UUID(int=5), UUID(int=3)]
    bm25 = [
        BM25Result(chunk_id=chunk_id, article_id=article_id, score=1, rank=1)
        for chunk_id in chunk_ids
    ]
    dense = [
        DenseResult(chunk_id=chunk_id, article_id=article_id, cosine_score=0.5, rank=1)
        for chunk_id in chunk_ids
    ]
    fused = rrf_fusion(bm25, dense, k=60, top_n=2)
    assert [result.chunk_id for result in fused] == sorted(chunk_ids)
    assert fused[0].rrf_score == 2 / 61


def _draft_with_statuses(statuses: list[VerificationStatus]) -> ReviewDraft:
    sentences = [
        VerifiedSentence(
            text=f"sentence {index}",
            citations=[UUID(int=index)],
            verification_statuses=[status],
            supporting_claim_ids=[],
        )
        for index, status in enumerate(statuses)
    ]
    return ReviewDraft(
        query_id=UUID(int=1),
        title="test review",
        generated_at=datetime.now(timezone.utc),
        llm_model="test-model",
        prompt_version="test",
        sentences=sentences,
    )


def test_review_draft_reports_each_verification_state_separately():
    """RD-01: per-state counts + coverage make '0% hallucinated' non-misleading."""
    draft = _draft_with_statuses([
        VerificationStatus.VERIFIED,
        VerificationStatus.VERIFIED,
        VerificationStatus.EXISTS_NO_SUPPORT,
        VerificationStatus.REFERENCE_UNRESOLVED,
    ])
    assert draft.overall_hallucination_rate == 0.0
    assert draft.overall_support_rate == 0.5
    assert draft.n_citations == 4
    assert draft.n_verified == 2
    assert draft.n_unsupported == 1
    assert draft.n_hallucinated == 0
    assert draft.n_reference_unresolved == 1
    assert draft.n_unverifiable == 0
    assert draft.n_evaluated == 4
    assert draft.verification_coverage == 1.0
    assert draft.evaluated_support_rate == 0.5


def test_unverified_draft_has_zero_coverage_and_is_not_a_clean_result():
    """RD-01: all-unverifiable draft must not read as '0% hallucinated = clean'."""
    draft = _draft_with_statuses([VerificationStatus.UNVERIFIABLE] * 3)
    assert draft.n_unverifiable == 3
    assert draft.n_evaluated == 0
    assert draft.verification_coverage == 0.0
    assert draft.evaluated_support_rate == 0.0
    # Legacy headline rates still report 0.0 — that is exactly the defect RD-01
    # exposes through the new explicit fields rather than changing the formula.
    assert draft.overall_hallucination_rate == 0.0
    assert draft.overall_support_rate == 0.0


def test_empty_review_metric_denominators_are_safe():
    """RD-01: no sentence must yield 0.0, never a division error."""
    draft = ReviewDraft(
        query_id=UUID(int=1), title="test review",
        generated_at=datetime.now(timezone.utc),
        llm_model="test-model", prompt_version="test", sentences=[],
    )
    assert draft.n_citations == 0
    assert draft.verification_coverage == 0.0
    assert draft.evaluated_support_rate == 0.0


def test_generate_response_exposes_honest_verification_counts():
    """RD-01: API response keeps legacy rates and adds explicit state fields."""
    from scilit.api.main import _build_generate_response

    verified_draft = _build_generate_response(
        _draft_with_statuses(
            [VerificationStatus.VERIFIED, VerificationStatus.EXISTS_NO_SUPPORT]
        ),
        "fixture", 2, "r.json", "r_audit.json",
    )
    assert verified_draft.nli_ran is True
    assert verified_draft.n_citations == 2
    assert verified_draft.n_verified == 1
    assert verified_draft.n_unsupported == 1
    assert verified_draft.verification_coverage == 1.0
    assert verified_draft.evaluated_support_rate == 0.5
    assert verified_draft.n_nli_verdicts == 2
    assert verified_draft.nli_coverage == 1.0
    assert verified_draft.nli_support_rate == 0.5
    assert len(verified_draft.sentences) == 2

    unverified = _build_generate_response(
        _draft_with_statuses([VerificationStatus.UNVERIFIABLE]),
        "fixture", 1, "r.json", None,
    )
    # The exact misleading case: old rates say "clean", new fields say "unchecked".
    assert unverified.overall_hallucination_rate == 0.0
    assert unverified.nli_ran is False
    assert unverified.verification_coverage == 0.0
    assert unverified.n_unverifiable == 1
    assert unverified.nli_coverage == 0.0


def test_generate_api_returns_clarification_status_for_rag_vs_nlp():
    from fastapi import HTTPException
    from scilit.api.main import GenerateRequest
    from scilit.api.main import generate as generate_endpoint

    with pytest.raises(HTTPException) as exc_info:
        generate_endpoint(GenerateRequest(query="RAG vs NLP"))
    assert exc_info.value.status_code == 422
    assert "RAG is a technique within NLP" in exc_info.value.detail


def test_review_endpoint_prefers_newer_unverified_generation_over_old_audit(
    tmp_path, monkeypatch
):
    from scilit.api import main as api

    paths = _temporary_data_paths(tmp_path)
    paths.reviews.mkdir(parents=True, exist_ok=True)
    (paths.reviews / "same-query.json").write_text(
        json.dumps({
            "title": "new generation",
            "generated_at": "2026-10-19T12:00:00+00:00",
            "sentences": [{"supporting_claim_ids": ["claim-without-embedded-evidence"]}],
        }),
        encoding="utf-8",
    )
    (paths.reviews / "same-query_verified.json").write_text(
        json.dumps({"title": "stale verified copy", "generated_at": "2026-10-19T11:00:00+00:00"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(api.config, "paths", lambda: paths)

    loaded = api.get_review("same-query")
    assert loaded["title"] == "new generation"
    assert loaded["evidence_complete"] is False
    assert "predates embedded exact source passages" in loaded["evidence_warning"]


def test_support_rate_and_coverage_use_distinct_denominators():
    """RD-01a: support% is over n_evaluated, coverage% over n_citations (10/8/1/1)."""
    draft = _draft_with_statuses(
        [VerificationStatus.VERIFIED] * 8
        + [VerificationStatus.EXISTS_NO_SUPPORT]
        + [VerificationStatus.UNVERIFIABLE]
    )
    assert draft.n_citations == 10
    assert draft.n_verified == 8
    assert draft.n_evaluated == 9            # 8 verified + 1 unsupported
    # support among *evaluated* -> 8/9 (~0.889): what the % badge shows.
    assert draft.evaluated_support_rate == pytest.approx(8 / 9)
    # coverage across *all citations* -> 9/10, shown separately.
    assert draft.verification_coverage == pytest.approx(0.9)
    # legacy rate uses the total denominator (8/10) -> intentionally different,
    # which is exactly the mismatch the UI must no longer hide.
    assert draft.overall_support_rate == pytest.approx(0.8)
    assert draft.evaluated_support_rate != draft.overall_support_rate


def test_schema_keeps_legacy_hallucination_labels_ambiguous():
    """Old releases mixed contradiction and missing-reference states in this value."""
    draft = _draft_with_statuses([
        VerificationStatus.VERIFIED,
        VerificationStatus.EXISTS_NO_SUPPORT,
        VerificationStatus.HALLUCINATED,
        VerificationStatus.UNVERIFIABLE,
    ])
    # Never reinterpret legacy HALLUCINATED as an unresolved reference.
    assert draft.n_reference_unresolved == 0
    assert draft.n_hallucinated == 1
    # VERIFIED + EXISTS_NO_SUPPORT are the only statuses that reached the NLI stage.
    assert draft.n_nli_verdicts == 2
    assert draft.n_reference_unresolved != draft.n_nli_verdicts
    # n_evaluated is the union of the two mechanisms, not 'checked by NLI'.
    assert draft.n_evaluated == (
        draft.n_nli_verdicts + draft.n_reference_unresolved + draft.n_hallucinated
    )
    assert draft.n_evaluated == 3


def test_generate_response_distinguishes_unavailable_from_zero_evaluable():
    """RD-01b: nli_ran separates 'unavailable' from 'ran but nothing evaluable'."""
    from scilit.api.main import _build_generate_response

    ran_nothing_evaluable = _build_generate_response(
        _draft_with_statuses([VerificationStatus.UNVERIFIABLE] * 3),
        "fixture", 1, "r.json", "r_audit.json",   # audit present => NLI ran
    )
    unavailable = _build_generate_response(
        _draft_with_statuses([VerificationStatus.UNVERIFIABLE] * 3),
        "fixture", 1, "r.json", None,             # no audit => NLI unavailable
    )
    # Both have zero coverage; only nli_ran disambiguates the two UI states.
    assert ran_nothing_evaluable.verification_coverage == 0.0
    assert unavailable.verification_coverage == 0.0
    assert ran_nothing_evaluable.nli_ran is True
    assert unavailable.nli_ran is False
    assert ran_nothing_evaluable.nli_ran != unavailable.nli_ran


def _summary_sentence(claim_id=None) -> VerifiedSentence:
    return VerifiedSentence(
        text="s",
        citations=[UUID(int=7)] if claim_id is not None else [],
        verification_statuses=[VerificationStatus.UNVERIFIABLE] if claim_id is not None else [],
        supporting_claim_ids=[claim_id] if claim_id is not None else [],
    )


def test_summary_counts_only_rendered_claims_not_full_retrieval():
    """RD-03: summary must count displayed claims, not the entire retrieved pool."""
    from scilit.generation.template import _count_rendered_claims_by_type

    method_ids = [UUID(int=i) for i in range(1, 12)]    # 11 method claims
    finding_ids = [UUID(int=i) for i in range(101, 172)]  # 71 finding claims
    claim_type_by_id = {
        **{str(cid): "method" for cid in method_ids},
        **{str(cid): "finding" for cid in finding_ids},
    }
    # Review renders 3 method sentences + 1 finding sentence (+ 1 no-claim sentence).
    sentences = [_summary_sentence(cid) for cid in method_ids[:3]]
    sentences += [_summary_sentence(finding_ids[0])]
    sentences += [_summary_sentence()]

    counts = _count_rendered_claims_by_type(sentences, claim_type_by_id)
    assert counts == {"method": 3, "finding": 1}
    # The full pool (11 method / 71 finding) must NOT leak into the reported counts.
    assert counts["method"] != len(method_ids)
    assert counts["finding"] != len(finding_ids)


def test_summary_counts_zero_when_no_claims_rendered():
    """RD-03: a pool of claims that renders nothing must count as zero."""
    from scilit.generation.template import _count_rendered_claims_by_type

    method_ids = [UUID(int=i) for i in range(1, 12)]
    claim_type_by_id = {str(cid): "method" for cid in method_ids}
    counts = _count_rendered_claims_by_type([_summary_sentence()], claim_type_by_id)
    assert counts == {"method": 0, "finding": 0}


def test_citation_graph_uses_article_referenced_works():
    source = make_article(20, referenced_works=["W21", "W99"])
    target = make_article(21)
    uuid_map = {
        str(article.id): article.model_dump(mode="json")
        for article in (source, target)
    }
    citations, graph = build_citations(uuid_map)
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert graph.edges[0].source_id == source.id
    assert graph.edges[0].target_id == target.id
    assert sum(citation.target_article_id is not None for citation in citations) == 1
    assert graph.density == 0.5


def test_nli_verifies_each_citation_independently_with_stub():
    first_id = UUID(int=100)
    second_id = UUID(int=101)
    missing_id = UUID(int=102)
    claim_id = UUID(int=200)
    second_claim_id = UUID(int=201)
    draft = ReviewDraft(
        query_id=UUID(int=300),
        title="Fixture review",
        sentences=[
            VerifiedSentence(
                text="A supported claim and another unsupported citation.",
                citations=[first_id, second_id, missing_id],
                verification_statuses=[VerificationStatus.UNVERIFIABLE] * 3,
                supporting_claim_ids=[claim_id, second_claim_id],
            )
        ],
        llm_model="test",
        prompt_version="test",
    )
    claims = {
        str(claim_id): {
            "article_id": str(first_id),
            "text": "A supported claim.",
        },
        str(second_claim_id): {
            "article_id": str(first_id),
            "text": "An even stronger supporting claim.",
        },
    }
    articles = {str(first_id): {}, str(second_id): {}}

    def nli_stub(premise, hypothesis):
        if premise == "A supported claim.":
            return VerificationStatus.VERIFIED, 0.91
        if premise == "An even stronger supporting claim.":
            return VerificationStatus.VERIFIED, 0.96
        return VerificationStatus.CONTRADICTED, 0.2

    updated, audit = verify_review(
        draft,
        claims,
        articles,
        None,
        None,
        {},
        {str(second_id): ["Contradictory chunk evidence."]},
        nli_fn=nli_stub,
    )
    statuses = updated.sentences[0].verification_statuses
    assert statuses == [
        VerificationStatus.VERIFIED,
        VerificationStatus.EXISTS_NO_SUPPORT,
        VerificationStatus.REFERENCE_UNRESOLVED,
    ]
    assert updated.sentences[0].nli_score == 0.96
    assert updated.n_reference_unresolved == 1
    assert updated.n_hallucinated == 0
    assert len(audit.entries) == 3
    assert "another_article" in audit.entries[1].note
    assert audit.entries[2].note.startswith("article_missing_from_corpus")


def test_ambiguous_rag_vs_nlp_query_asks_for_a_comparable_baseline(monkeypatch):
    from scilit.generation import template

    monkeypatch.setattr(
        template,
        "_load_articles",
        lambda: pytest.fail("ambiguous query should be rejected before loading data"),
    )
    with pytest.raises(template.QueryClarificationRequired, match="RAG is a technique"):
        template.generate("RAG vs NLP")


def test_incomplete_claim_is_not_made_into_a_complete_sentence():
    from scilit.generation.template import _template_sentence

    assert _template_sentence({"text": "The model sits atop the leaderboard, out perfo"}) is None
    assert _template_sentence({"text": "The model achieves 90.1"}) is None
    assert _template_sentence({"text": "The model achieves 90.1 accuracy."}) == (
        "The model achieves 90.1 accuracy."
    )


def test_llm_rewrite_cannot_add_numbers_or_remove_polarity():
    from scilit.generation.template import _safe_rewrite

    source = "The proposed model may not improve F1 by 2.1 points."
    assert _safe_rewrite(
        source,
        "The proposed model may improve F1 by 99.0 points.",
    ) == source
    assert _safe_rewrite(
        source,
        "The proposed system may not improve F1 by 2.1 points.",
    ) == source
    assert _safe_rewrite(source, source) == source


def test_claim_extraction_keeps_offsets_on_exact_complete_source_text():
    from types import SimpleNamespace

    from scilit.extraction.claims import _claim_span

    source = "The model improves benchmark results by 2.3 points."
    chunk = Chunk(
        id=UUID(int=9301),
        article_id=UUID(int=9302),
        section_id=UUID(int=9303),
        section_type=SectionType.RESULTS,
        text=source,
        char_start=0,
        char_end=len(source),
        chunk_index=0,
    )
    span = _claim_span(chunk, SimpleNamespace(text=source, start_char=0))
    assert span == (source, 0, len(source))
    assert source[span[1]:span[2]] == span[0]
    assert _claim_span(
        chunk, SimpleNamespace(text="The model achieves 90.1", start_char=0)
    ) is None
    overlap_chunk = chunk.model_copy(update={"char_start": 75})
    assert _claim_span(overlap_chunk, SimpleNamespace(text=source, start_char=0)) is None


def test_claim_level_relevance_rejects_dense_out_of_topic_candidates():
    from scilit.generation.template import _claim_relevance

    query = "LoRA vs full fine-tuning in low-resource NLP"
    relevant, _ = _claim_relevance(
        query,
        "DeBERTa outperforms baselines on several language understanding benchmarks.",
        "DeBERTa: Decoding-enhanced BERT with Disentangled Attention",
    )
    assert relevant is False
    relevant, score = _claim_relevance(
        query,
        "LoRA enables fine-tuning in low-resource NLP settings.",
        "Parameter-efficient LoRA fine-tuning for low-resource NLP",
    )
    assert relevant is True
    assert score > 0


def test_generation_quotes_relevant_claims_with_exact_evidence_and_deduplicates(monkeypatch):
    from scilit.generation import template

    article = make_article(807, title="LoRA fine-tuning for low-resource NLP")
    article_data = article.model_dump(mode="json")
    chunks = [
        {
            "id": str(uuid5(article.id, f"fixture-chunk-{index}")),
            "text": text,
            "rank": index + 1,
        }
        for index, text in enumerate([
            "LoRA enables fine-tuning in low-resource NLP settings.",
            "LoRA fine-tuning reduces trainable parameters in low-resource NLP tasks.",
            "DeBERTa outperforms baselines on several language understanding benchmarks.",
        ])
    ]
    claims = []
    for index, chunk in enumerate(chunks):
        source = chunk["text"]
        claims.append({
            "id": str(uuid5(UUID(chunk["id"]), f"claim-{index}")),
            "article_id": str(article.id),
            "chunk_id": chunk["id"],
            "claim_type": "finding",
            "text": source,
            "verbatim": source,
        })
    # A repeated overlap claim with a different ID must not produce a duplicate line.
    claims.append({
        **claims[0],
        "id": str(uuid5(UUID(chunks[0]["id"]), "duplicate-claim")),
    })
    ranked = [
        RankedResult(
            query_id=UUID(int=900 + index),
            chunk_id=UUID(chunk["id"]),
            article_id=article.id,
            section_type=SectionType.ABSTRACT,
            chunk_text=chunk["text"],
            bm25_rank=index + 1,
            dense_rank=index + 1,
            rrf_score=1 / (index + 1),
            final_rank=index + 1,
        )
        for index, chunk in enumerate(chunks)
    ]
    monkeypatch.setattr(template, "_load_articles", lambda: {str(article.id): article_data})
    monkeypatch.setattr(template, "_load_claims", lambda: claims)
    monkeypatch.setattr(
        template,
        "_load_chunk_metadata",
        lambda: {chunk["id"]: {"char_start": 0} for chunk in chunks},
    )
    monkeypatch.setattr(template, "retrieve", lambda *args, **kwargs: ranked)

    draft, bibtex = template.generate(
        "LoRA vs full fine-tuning in low-resource NLP",
        max_sentences_per_type=5,
    )
    cited_sentences = [sentence for sentence in draft.sentences if sentence.citations]
    texts = [sentence.text for sentence in cited_sentences]
    assert texts.count(chunks[0]["text"]) == 1
    assert chunks[1]["text"] in texts
    assert chunks[2]["text"] not in texts
    assert all(sentence.evidence[0].passage == sentence.text for sentence in cited_sentences)
    assert all(sentence.evidence[0].article_id == article.id for sentence in cited_sentences)
    assert len(bibtex) == 1


def test_comparison_review_requires_evidence_for_both_named_sides(monkeypatch):
    from scilit.generation import template

    article = make_article(808, title="Full fine-tuning for low-resource NLP")
    article_id = str(article.id)
    chunk_id = uuid5(article.id, "right-side-only")
    source = "Full fine-tuning can require substantial memory in low-resource NLP tasks."
    claim_id = uuid5(chunk_id, "claim")
    monkeypatch.setattr(
        template, "_load_articles", lambda: {article_id: article.model_dump(mode="json")}
    )
    monkeypatch.setattr(template, "_load_claims", lambda: [{
        "id": str(claim_id), "article_id": article_id, "chunk_id": str(chunk_id),
        "claim_type": "finding", "text": source, "verbatim": source,
    }])
    monkeypatch.setattr(template, "_load_chunk_metadata", lambda: {str(chunk_id): {"char_start": 0}})
    monkeypatch.setattr(template, "retrieve", lambda *args, **kwargs: [RankedResult(
        query_id=UUID(int=9801),
        chunk_id=chunk_id,
        article_id=article.id,
        section_type=SectionType.ABSTRACT,
        chunk_text=source,
        bm25_rank=1,
        dense_rank=1,
        rrf_score=0.03,
        final_rank=1,
    )])

    with pytest.raises(template.InsufficientRelevantEvidence, match="LoRA"):
        template.generate("LoRA vs full fine-tuning in low-resource NLP", k=5)


def test_nli_does_not_use_an_unlinked_chunk_as_sentence_support():
    article_id = UUID(int=9001)
    draft = ReviewDraft(
        query_id=UUID(int=9002),
        title="Fixture",
        sentences=[VerifiedSentence(
            text="A sentence with no claim-level source attached.",
            citations=[article_id],
            verification_statuses=[VerificationStatus.UNVERIFIABLE],
            supporting_claim_ids=[],
        )],
        llm_model="test",
        prompt_version="test",
    )
    updated, _ = verify_review(
        draft,
        {},
        {str(article_id): {}},
        None,
        None,
        {},
        chunks_lookup={str(article_id): ["A chunk that happens to mention this topic."]},
        nli_fn=lambda *_: pytest.fail("unlinked article chunks must never be used as proof"),
    )
    assert updated.sentences[0].verification_statuses == [VerificationStatus.UNVERIFIABLE]
    assert updated.sentences[0].nli_score is None


def test_nli_rejects_evidence_quote_that_does_not_match_claim_source():
    from scilit.schemas import ClaimEvidence

    article_id = UUID(int=9051)
    claim_id = UUID(int=9052)
    chunk_id = UUID(int=9053)
    draft = ReviewDraft(
        query_id=UUID(int=9054),
        title="Fixture",
        sentences=[VerifiedSentence(
            text="A sentence with a source passage.",
            citations=[article_id],
            verification_statuses=[VerificationStatus.UNVERIFIABLE],
            supporting_claim_ids=[claim_id],
            evidence=[ClaimEvidence(
                claim_id=claim_id,
                article_id=article_id,
                chunk_id=chunk_id,
                article_title="Fixture paper",
                passage="A different passage from the source.",
            )],
        )],
        llm_model="test",
        prompt_version="test",
    )
    updated, audit = verify_review(
        draft,
        {str(claim_id): {
            "article_id": str(article_id),
            "text": "A sentence with a source passage.",
        }},
        {str(article_id): {}},
        None,
        None,
        {},
        nli_fn=lambda *_: pytest.fail("mismatched displayed evidence must not reach NLI"),
    )
    assert updated.sentences[0].verification_statuses == [VerificationStatus.UNVERIFIABLE]
    assert "does_not_match" in audit.entries[0].note


def test_nli_contradiction_is_not_classified_as_an_invented_reference():
    article_id = UUID(int=9101)
    claim_id = UUID(int=9102)
    draft = ReviewDraft(
        query_id=UUID(int=9103),
        title="Fixture",
        sentences=[VerifiedSentence(
            text="The method decreases error rates.",
            citations=[article_id],
            verification_statuses=[VerificationStatus.UNVERIFIABLE],
            supporting_claim_ids=[claim_id],
        )],
        llm_model="test",
        prompt_version="test",
    )
    updated, _ = verify_review(
        draft,
        {str(claim_id): {"article_id": str(article_id), "text": "The method increases error rates."}},
        {str(article_id): {}},
        None,
        None,
        {},
        nli_fn=lambda *_: (VerificationStatus.CONTRADICTED, 0.04),
    )
    assert updated.sentences[0].verification_statuses == [VerificationStatus.CONTRADICTED]
    assert updated.n_contradicted == 1
    assert updated.n_reference_unresolved == 0
    assert updated.n_nli_verdicts == 1


def test_nli_runtime_failure_is_unverifiable_and_never_a_false_success():
    article_id = UUID(int=9201)
    claim_id = UUID(int=9202)
    draft = ReviewDraft(
        query_id=UUID(int=9203),
        title="Fixture",
        sentences=[VerifiedSentence(
            text="A complete source claim.",
            citations=[article_id],
            verification_statuses=[VerificationStatus.UNVERIFIABLE],
            supporting_claim_ids=[claim_id],
        )],
        llm_model="test",
        prompt_version="test",
    )

    def fail_nli(*_args):
        raise RuntimeError("model unavailable")

    updated, audit = verify_review(
        draft,
        {str(claim_id): {"article_id": str(article_id), "text": "A complete source claim."}},
        {str(article_id): {}},
        None,
        None,
        {},
        nli_fn=fail_nli,
    )
    assert updated.sentences[0].verification_statuses == [VerificationStatus.UNVERIFIABLE]
    assert updated.sentences[0].nli_score is None
    assert "nli_error:RuntimeError" in audit.entries[0].note


def test_uncited_framing_sentences_are_outside_citation_support_statistics():
    draft = _draft_with_statuses([VerificationStatus.VERIFIED])
    draft.sentences.append(VerifiedSentence(
        text="This summary sentence has no evidence attached.",
        citations=[],
        verification_statuses=[],
        supporting_claim_ids=[],
    ))
    assert draft.n_citations == 1
    assert draft.nli_support_rate == 1.0
    assert draft.nli_coverage == 1.0
    assert draft.n_sentences_without_citations == 1


def test_automatic_nli_does_not_report_invented_citation_rate():
    from scilit.schemas import HallucinationAudit, HallucinationAuditEntry

    audit = HallucinationAudit(
        review_id=UUID(int=9401),
        entries=[
            HallucinationAuditEntry(
                review_id=UUID(int=9401), sentence_text="source quote", cited_key="A",
                status=VerificationStatus.REFERENCE_UNRESOLVED, auditor="auto_nli",
            ),
            HallucinationAuditEntry(
                review_id=UUID(int=9401), sentence_text="sentence", cited_key="B",
                status=VerificationStatus.VERIFIED, auditor="auto_nli",
            ),
            HallucinationAuditEntry(
                review_id=UUID(int=9401), sentence_text="sentence", cited_key="C",
                status=VerificationStatus.CONTRADICTED, auditor="auto_nli",
            ),
        ],
    )
    assert audit.invented_rate == 0.0
    assert audit.support_rate == 0.5


def test_api_fixture_corpus_endpoints_and_bm25_search(tmp_path, monkeypatch, request):
    monkeypatch.setenv("SCILIT_DATA_DIR", str(tmp_path))
    raw_dir = tmp_path / "raw"
    parsed_dir = tmp_path / "parsed"
    raw_dir.mkdir()
    parsed_dir.mkdir()
    articles = [make_article(index) for index in range(3)]
    (raw_dir / "articles.jsonl").write_text(
        "".join(article.model_dump_json() + "\n" for article in articles),
        encoding="utf-8",
    )
    claims = []
    chunks = []
    for index, article in enumerate(articles):
        claim = {
            "id": str(uuid5(article.id, "claim:0")),
            "article_id": str(article.id),
            "chunk_id": str(uuid5(article.id, "chunk:0")),
            "claim_type": "finding",
            "hedging_level": "certain",
            "text": f"Natural language processing result number {index}. Markerfixture{index}.",
            "metric_name": None,
            "metric_value": None,
        }
        claims.append(claim)
        chunks.append(
            Chunk(
                id=UUID(claim["chunk_id"]),
                article_id=article.id,
                section_id=uuid5(article.id, "section:0"),
                section_type=SectionType.ABSTRACT,
                text=claim["text"] + " Transformers improve language processing.",
                char_start=0,
                char_end=90,
                chunk_index=0,
            )
        )
    (parsed_dir / "claims.jsonl").write_text(
        "".join(json.dumps(claim) + "\n" for claim in claims),
        encoding="utf-8",
    )
    (parsed_dir / "chunks.jsonl").write_text(
        "".join(chunk.model_dump_json() + "\n" for chunk in chunks),
        encoding="utf-8",
    )

    from scilit.corpus.index import build_bm25

    build_bm25(chunks)
    from scilit.api import main as api
    from scilit.retrieval import pipeline

    def clear_pipeline_caches():
        api._get_retrieval.cache_clear()
        pipeline._bm25.cache_clear()
        pipeline._dense.cache_clear()
        pipeline._chunk_lookup.cache_clear()

    request.addfinalizer(clear_pipeline_caches)
    api._get_retrieval.cache_clear()
    pipeline._bm25.cache_clear()
    pipeline._dense.cache_clear()
    pipeline._chunk_lookup.cache_clear()
    with TestClient(api.app) as client:
        response = client.get(
            "/search",
            params={"q": "markerfixture0", "mode": "bm25"},
        )
        assert response.status_code == 200
        assert response.json()["results"]

        article_id = str(articles[0].id)
        article_response = client.get(f"/articles/{article_id}")
        assert article_response.status_code == 200
        assert len(article_response.json()["claims"]) == 1

        bibtex_response = client.get(
            "/export/bibtex",
            params={"ids": f"{articles[0].id},{articles[1].id}"},
        )
        assert bibtex_response.status_code == 200
        assert bibtex_response.text.count("@inproceedings{") == 2
        keys = [
            line.split("{", 1)[1].split(",", 1)[0]
            for line in bibtex_response.text.splitlines()
            if line.startswith("@")
        ]
        assert len(keys) == len(set(keys))
