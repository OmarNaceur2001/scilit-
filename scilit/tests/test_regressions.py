import json
from uuid import NAMESPACE_URL, UUID, uuid5

from fastapi.testclient import TestClient
from scilit.corpus.collect import (
    _extract_acl_id,
    _extract_arxiv_id,
    _oa_to_article,
    _reconstruct_abstract,
)
from scilit.corpus.index import rrf_fusion
from scilit.corpus.parse import chunk_text, parse_abstract
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
    ReviewDraft,
    SectionType,
    Venue,
    VerificationStatus,
    VerifiedSentence,
)


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
            license="open access",
        ),
        "openalex_id": f"W{index}",
        "venue_name": "Annual Meeting of the ACL",
    }
    defaults.update(values)
    return Article(**defaults)


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
        return VerificationStatus.HALLUCINATED, 0.2

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
        VerificationStatus.HALLUCINATED,
    ]
    assert updated.sentences[0].nli_score == 0.96
    assert len(audit.entries) == 3
    assert "contradicted" in audit.entries[1].note
    assert audit.entries[2].note.startswith("article_missing_from_corpus")


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
