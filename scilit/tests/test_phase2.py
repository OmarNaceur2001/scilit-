import json
import math
from types import SimpleNamespace
from uuid import uuid4

import pytest
from scilit.evaluation.retrieval_eval import (
    METRIC_NAMES,
    append_experiment,
    mrr_at_10,
    ndcg_at_10,
    precision_at_5,
    query_metrics,
    recall_at_10,
)
from scilit.extraction.grid import build_comparison_grid
from scilit.generation.render import render_review_markdown
from scilit.schemas import (
    Article,
    ArticleSource,
    ComparativeCell,
    ComparativeRow,
    ReviewDraft,
    Venue,
    VerificationStatus,
    VerifiedSentence,
)


def _article():
    return Article(
        id=uuid4(),
        title="An Evidence-Based NLP Study",
        authors=["Ada Lovelace"],
        year=2023,
        abstract=(
            "We introduce an evidence-backed adapter method. "
            "The method is evaluated on benchmark data."
        ),
        source=ArticleSource(
            venue=Venue.ACL,
            acl_id="2023.acl-long.7",
            url="https://aclanthology.org/2023.acl-long.7",
            license="cc-by-4.0",
        ),
    )


def test_comparison_grid_uses_verbatim_evidence_with_25_word_cap():
    article = _article()
    chunk_id = uuid4()
    claim_ids = [uuid4() for _ in range(4)]
    long_method = " ".join(["method"] * 30) + "."
    claims = [
        {
            "id": claim_ids[0],
            "article_id": str(article.id),
            "claim_type": "method",
            "text": long_method,
        },
        {
            "id": claim_ids[1],
            "article_id": str(article.id),
            "claim_type": "dataset",
            "text": "We evaluate the approach on SQuAD and GLUE.",
        },
        {
            "id": claim_ids[2],
            "article_id": str(article.id),
            "claim_type": "finding",
            "text": "The approach reaches 92.5 F1 on SQuAD.",
            "metric_value": 92.5,
        },
        {
            "id": claim_ids[3],
            "article_id": str(article.id),
            "claim_type": "limitation",
            "text": "The method requires substantial compute.",
        },
    ]
    chunks = [{
        "id": chunk_id,
        "article_id": str(article.id),
        "section_type": "abstract",
        "text": article.abstract,
    }]

    row = build_comparison_grid(
        [article.id], [article], claims, [], chunks
    )[0]

    assert row.question[0].text == "We introduce an evidence-backed adapter method."
    assert row.question[0].chunk_id == chunk_id
    assert len(row.method[0].text.split()) == 25
    assert row.method[0].claim_id == claim_ids[0]
    assert row.data[0].text == "SQuAD"
    assert row.results[0].claim_id == claim_ids[2]
    assert row.limitations[0].text == claims[3]["text"]
    assert all(
        cell.verbatim
        for group in (row.question, row.method, row.data, row.results, row.limitations)
        for cell in group
    )
    with pytest.raises(ValueError, match="at most 25 words"):
        ComparativeRow(
            article_id=article.id,
            method=[ComparativeCell(
                text=" ".join(["word"] * 26), claim_id=uuid4()
            )],
        )
    limitation_section_id = uuid4()
    limitation_chunk = {
        "id": uuid4(),
        "article_id": str(article.id),
        "section_id": limitation_section_id,
        "section_type": "limitations",
        "text": "The model needs more compute.",
    }
    fallback = build_comparison_grid(
        [article.id],
        [article],
        claims[:-1],
        [{
            "id": limitation_section_id,
            "article_id": str(article.id),
            "section_type": "limitations",
            "text": "",
        }],
        [*chunks, limitation_chunk],
    )[0]
    assert fallback.limitations[0].text == limitation_chunk["text"]
    assert fallback.limitations[0].chunk_id == limitation_chunk["id"]


def test_review_renderer_numbers_first_appearance_and_optional_badges():
    first_id, second_id = uuid4(), uuid4()
    draft = ReviewDraft(
        query_id=uuid4(),
        title="A literature review",
        llm_model="template",
        prompt_version="test",
        sentences=[
            VerifiedSentence(
                text="A method was proposed.",
                theme_heading="Méthodes",
                citations=[second_id],
                verification_statuses=[VerificationStatus.VERIFIED],
                supporting_claim_ids=[],
            ),
            VerifiedSentence(
                text="Another result followed.",
                citations=[first_id, second_id],
                verification_statuses=[
                    VerificationStatus.HALLUCINATED,
                    VerificationStatus.VERIFIED,
                ],
                supporting_claim_ids=[],
            ),
        ],
    )
    articles = {
        str(first_id): {"title": "First paper", "authors": ["A. Author"], "year": 2020},
        str(second_id): {"title": "Second paper", "authors": ["B. Writer"], "year": 2021},
    }

    markdown = render_review_markdown(draft, articles)
    assert "## Méthodes" in markdown
    assert "A method was proposed. [1] ✓ verified" in markdown
    assert "Another result followed. [2] [1] ✗ hallucinated" in markdown
    assert "[1] B. Writer. (2021). Second paper" in markdown
    assert "[2] A. Author. (2020). First paper" in markdown
    assert "✓ verified" not in render_review_markdown(draft, articles, include_badges=False)


def test_review_markdown_api_reads_saved_draft(tmp_path, monkeypatch):
    from scilit.api import main

    cited_id = uuid4()
    draft = ReviewDraft(
        query_id=uuid4(),
        title="Saved review",
        llm_model="template",
        prompt_version="test",
        sentences=[
            VerifiedSentence(
                text="A supported finding.",
                citations=[cited_id],
                verification_statuses=[VerificationStatus.VERIFIED],
                supporting_claim_ids=[],
            )
        ],
    )
    (tmp_path / "fixture.json").write_text(draft.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        main.config, "paths", lambda: SimpleNamespace(reviews=tmp_path)
    )
    monkeypatch.setattr(
        main, "_get_articles", lambda: {str(cited_id): {"title": "Cited paper"}}
    )

    response = main.get_review_markdown("fixture")
    assert response.media_type == "text/markdown"
    assert "A supported finding. [1] ✓ verified" in response.body.decode()
    no_badges = main.get_review_markdown("fixture", badges=False)
    assert "✓ verified" not in no_badges.body.decode()


def test_template_generation_groups_sentences_by_claim_theme(monkeypatch):
    from scilit.generation import template

    article = _article()
    method_id, finding_id = uuid4(), uuid4()
    claims = [
        {
            "id": str(method_id),
            "article_id": str(article.id),
            "chunk_id": "chunk-method",
            "claim_type": "method",
            "text": "We introduce a lightweight adapter architecture.",
        },
        {
            "id": str(finding_id),
            "article_id": str(article.id),
            "chunk_id": "chunk-result",
            "claim_type": "finding",
            "text": "The adapter improves performance on the benchmark.",
        },
    ]
    ranked = [
        SimpleNamespace(chunk_id="chunk-method", article_id=article.id),
        SimpleNamespace(chunk_id="chunk-result", article_id=article.id),
    ]
    monkeypatch.setattr(template, "_load_articles", lambda: {str(article.id): article.model_dump()})
    monkeypatch.setattr(template, "_load_claims", lambda: claims)
    monkeypatch.setattr(template, "retrieve", lambda *args, **kwargs: ranked)

    draft, _ = template.generate("adapter methods", k=2)
    headings = [sentence.theme_heading for sentence in draft.sentences]
    assert "Méthodes" in headings
    assert "Résultats" in headings
    assert draft.sentences[0].citations == []
    assert draft.sentences[-1].citations == []


def test_table_api_formats_and_deduplicates_search_hits(monkeypatch):
    from scilit.api import main

    article = _article()
    row = build_comparison_grid(
        [article.id],
        [article],
        [],
        [],
        [{"id": uuid4(), "article_id": str(article.id), "text": article.abstract}],
    )[0]
    seen = []
    monkeypatch.setattr(
        main,
        "search",
        lambda request: SimpleNamespace(results=[
            SimpleNamespace(article_id=str(article.id)),
            SimpleNamespace(article_id=str(article.id)),
        ]),
    )
    monkeypatch.setattr(
        main, "_table_rows", lambda ids: (seen.append(ids) or [row])
    )
    monkeypatch.setattr(
        main, "_get_articles", lambda: {str(article.id): {"title": article.title}}
    )

    assert main.get_table("adapter method", 1, "json") == [row]
    assert seen[-1] == [str(article.id)]
    csv_response = main.get_table("adapter method", 1, "csv")
    md_response = main.get_table("adapter method", 1, "md")
    assert "article_id,title,question,method,data,results,limitations" in csv_response.body.decode()
    assert "| Title | question | method | data | results | limitations |" in md_response.body.decode()
    assert main.post_table({"article_ids": [str(article.id), str(article.id)]}) == [row]
    assert seen[-1] == [str(article.id)]
    article_data = article.model_dump()
    monkeypatch.setattr(
        main, "_get_articles", lambda: {str(article.id): article_data}
    )
    monkeypatch.setattr(main, "_get_claims", lambda: [])
    from scilit.corpus import parse

    monkeypatch.setattr(parse, "load_sections", lambda: [])
    monkeypatch.setattr(
        parse,
        "load_chunks",
        lambda: [{
            "id": uuid4(),
            "article_id": str(article.id),
            "text": article.abstract,
        }],
    )
    assert main.get_article(str(article.id))["grid"].article_id == article.id


def test_metrics_match_hand_computed_rankings_and_exclude_empty_qrels():
    ranking = ["a", "b", "c", "d", "e", "f"]
    relevant = {"b", "d", "f"}
    assert precision_at_5(ranking, relevant) == pytest.approx(0.4)
    assert recall_at_10(ranking, relevant) == pytest.approx(1.0)
    assert mrr_at_10(ranking, relevant) == pytest.approx(0.5)
    expected_ndcg = (
        1 / math.log2(3)
        + 1 / math.log2(5)
        + 1 / math.log2(7)
    ) / (
        1
        + 1 / math.log2(3)
        + 1 / math.log2(4)
    )
    assert ndcg_at_10(ranking, relevant) == pytest.approx(expected_ndcg)
    assert query_metrics(ranking, set()) == {
        "P@5": None,
        "Recall@10": None,
        "MRR@10": None,
        "nDCG@10": None,
    }


def test_append_experiment_records_bootstrap_seed_means(tmp_path):
    seed_means = {"42": 0.4, "123": 0.5, "999": 0.6}
    result = {
        "systems": ["bm25"],
        "qrels": {"judged": 0},
        "aggregate": {
            "bm25": {
                metric: {"bootstrap_seed_means": seed_means}
                for metric in METRIC_NAMES
            }
        },
    }
    path = tmp_path / "experiments.jsonl"

    append_experiment(result, path)

    record = json.loads(path.read_text(encoding="utf-8"))
    assert len(record["results"]) == len(METRIC_NAMES)
    assert record["results"][0]["seeds"][0]["seed"] == 42
    assert record["results"][0]["seeds"][0]["value"] == 0.4
    assert "not training seeds" in record["notes"]
