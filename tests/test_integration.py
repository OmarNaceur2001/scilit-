import pytest

from scilit import config

PATHS = config.paths()
HAS_CORPUS = all(
    (PATHS.parsed / filename).exists()
    for filename in ("chunks.jsonl", "claims.jsonl", "bm25_index.pkl")
)
pytestmark = pytest.mark.skipif(not HAS_CORPUS, reason="Bronze corpus has not been built")


def test_bm25_returns_corpus_results():
    from scilit.retrieval.pipeline import search

    assert search("natural language processing", k=5, mode="bm25")


def test_bronze_corpus_contains_claims():
    claims_path = config.paths().parsed / "claims.jsonl"
    assert claims_path.read_text(encoding="utf-8").strip()
