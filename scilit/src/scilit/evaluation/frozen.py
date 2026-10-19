"""Evaluation against a frozen, manually annotated retrieval test set.

Each JSONL line has this shape::

    {"query_id":"q001","query":"...","relevant_chunk_ids":["..."]}

The same judgments are passed to BM25, dense retrieval, and RRF. Missing or
malformed annotations are errors; the evaluator never substitutes pseudo-labels.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


class RelevanceJudgment(BaseModel):
    query_id: str = Field(min_length=1)
    query: str = Field(min_length=2)
    relevant_chunk_ids: list[str] = Field(min_length=1)


def load_relevance_judgments(path: Path) -> list[RelevanceJudgment]:
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen relevance test set is missing: {path}. "
            "Add human-annotated query/chunk judgments before reporting Recall@10 or MRR."
        )
    judgments: list[RelevanceJudgment] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = RelevanceJudgment.model_validate_json(line)
        except ValidationError as exc:
            raise ValueError(f"Invalid frozen test-set row {line_number}: {exc}") from exc
        if item.query_id in seen_ids:
            raise ValueError(f"Duplicate query_id in frozen test set: {item.query_id}")
        if len(item.relevant_chunk_ids) != len(set(item.relevant_chunk_ids)):
            raise ValueError(f"Duplicate relevant chunk IDs for query {item.query_id}")
        seen_ids.add(item.query_id)
        judgments.append(item)
    if not judgments:
        raise ValueError(f"Frozen relevance test set is empty: {path}")
    return judgments


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    if not relevant_ids:
        raise ValueError("Recall requires at least one annotated relevant chunk.")
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int = 10
) -> float:
    if not relevant_ids:
        raise ValueError("MRR requires at least one annotated relevant chunk.")
    for rank, chunk_id in enumerate(retrieved_ids[:k], 1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_modes(
    judgments: list[RelevanceJudgment],
    modes: tuple[str, ...] = ("bm25", "dense", "rrf"),
    k: int = 10,
    search_fn=None,
) -> dict[str, dict[str, list]]:
    """Evaluate every mode on exactly the same annotated queries."""
    from scilit.schemas import EvaluationObservation

    if search_fn is None:
        from scilit.retrieval.pipeline import search as search_fn
    output: dict[str, dict[str, list[EvaluationObservation]]] = {}
    for mode in modes:
        recall_observations = []
        mrr_observations = []
        for judgment in judgments:
            results = search_fn(judgment.query, k=k, mode=mode)
            retrieved = [str(result.chunk_id) for result in results[:k]]
            relevant = set(judgment.relevant_chunk_ids)
            recall_observations.append(EvaluationObservation(
                sample_id=judgment.query_id,
                metric_name=f"Recall@{k}",
                value=recall_at_k(retrieved, relevant, k),
            ))
            mrr_observations.append(EvaluationObservation(
                sample_id=judgment.query_id,
                metric_name=f"MRR@{k}",
                value=reciprocal_rank_at_k(retrieved, relevant, k),
            ))
        output[mode] = {f"Recall@{k}": recall_observations, f"MRR@{k}": mrr_observations}
    return output
