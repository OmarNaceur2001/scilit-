"""
SciLit — tests/test_evaluation.py
===================================
Regression tests for SciLit's evaluation contracts.

The official evaluator requires a frozen, human-annotated relevance set and
reports Recall@10/MRR for the same queries in BM25, dense, and RRF modes.
It refuses to substitute lexical pseudo-labels or duplicate review files.

Usage :
  pytest tests/test_evaluation.py -v
  python tests/test_evaluation.py      # mode standalone avec rapport complet
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

CLAIMS_PATH = Path("data/parsed/claims.jsonl")
REVIEWS_DIR = Path("data/parsed/reviews")


# ─────────────────────────────────────────────────────────────
# Frozen benchmark entry point
# ─────────────────────────────────────────────────────────────

def run_evaluation() -> None:
    """Delegate to the evaluator that refuses missing/unannotated test data."""
    from scilit.evaluation.bronze_eval import run_evaluation as run_frozen_evaluation
    run_frozen_evaluation()


# ─────────────────────────────────────────────────────────────
# pytest tests (§ 2.3.3 compliance)
# ─────────────────────────────────────────────────────────────

def test_component_metrics_requires_3_seeds():
    """Le validateur doit rejeter ≠ 3 graines."""
    from scilit.schemas import ComponentMetrics, SeedResult
    with pytest.raises(Exception):
        ComponentMetrics(
            component_name = "test",
            metric_name    = "p@5",
            seeds          = [SeedResult(seed=42, metric_name="p@5", value=0.8)],
        )


def test_component_metrics_accepts_3_seeds():
    from scilit.schemas import ComponentMetrics, SeedResult
    m = ComponentMetrics(
        component_name = "test",
        metric_name    = "p@5",
        seeds          = [
            SeedResult(seed=42,  metric_name="p@5", value=0.8),
            SeedResult(seed=123, metric_name="p@5", value=0.9),
            SeedResult(seed=999, metric_name="p@5", value=0.7),
        ],
    )
    assert abs(m.mean - 0.8) < 1e-9
    assert m.std > 0


def test_component_metrics_keeps_fixed_query_observations_separate_from_seeds():
    from scilit.schemas import ComponentMetrics, EvaluationObservation

    metrics = ComponentMetrics(
        component_name="rrf_retrieval",
        metric_name="Recall@10",
        observations=[
            EvaluationObservation(sample_id="q1", metric_name="Recall@10", value=0.5),
            EvaluationObservation(sample_id="q2", metric_name="Recall@10", value=1.0),
        ],
    )
    assert metrics.mean == pytest.approx(0.75)
    assert metrics.std > 0
    assert metrics.seeds == []


def test_frozen_retrieval_metrics_use_recall_and_mrr_on_the_same_queries():
    from scilit.evaluation.frozen import (
        RelevanceJudgment,
        evaluate_modes,
        recall_at_k,
        reciprocal_rank_at_k,
    )

    assert recall_at_k(["a", "b", "c"], {"b", "d"}, k=3) == pytest.approx(0.5)
    assert reciprocal_rank_at_k(["x", "b", "c"], {"b"}, k=3) == pytest.approx(0.5)
    calls = []

    def fake_search(query, *, k, mode):
        calls.append((query, k, mode))
        relevant_id = "relevant-1" if query == "first" else "relevant-2"
        returned = ["noise", relevant_id] if mode != "dense" else [relevant_id]
        return [type("Result", (), {"chunk_id": item}) for item in returned[:k]]

    judgments = [
        RelevanceJudgment(query_id="q1", query="first", relevant_chunk_ids=["relevant-1"]),
        RelevanceJudgment(query_id="q2", query="second", relevant_chunk_ids=["relevant-2"]),
    ]
    results = evaluate_modes(judgments, k=10, search_fn=fake_search)
    assert {(query, mode) for query, _, mode in calls} == {
        ("first", "bm25"), ("second", "bm25"),
        ("first", "dense"), ("second", "dense"),
        ("first", "rrf"), ("second", "rrf"),
    }
    assert results["bm25"]["Recall@10"][0].value == 1.0
    assert results["bm25"]["MRR@10"][0].value == 0.5
    assert results["dense"]["MRR@10"][0].value == 1.0


def test_missing_frozen_relevance_set_is_not_replaced_by_pseudo_labels(tmp_path):
    from scilit.evaluation.frozen import load_relevance_judgments

    with pytest.raises(FileNotFoundError, match="human-annotated"):
        load_relevance_judgments(tmp_path / "missing.jsonl")


def test_bm25_returns_results():
    from scilit.retrieval.pipeline import search
    results = search("transformer NLP", k=5, mode="bm25")
    assert len(results) > 0


def test_claims_loaded():
    claims = [
        json.loads(line)
        for line in CLAIMS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(claims) >= 5000, f"Seulement {len(claims)} claims — pipeline incomplet ?"


def test_nli_verified_review_exists():
    verified = list(REVIEWS_DIR.glob("*_verified.json"))
    assert len(verified) >= 1, "Aucune review NLI vérifiée — lancer nli_verify d'abord."


def test_ablation_result_delta_is_computed():
    """RD-07: AblationResult n'accepte que les vrais champs du schéma.

    `delta` doit être dérivé (ablated - full) et non passé au constructeur.
    Une ablation qui dégrade la métrique (ablated < full) donne un delta négatif.
    """
    from scilit.schemas import AblationResult

    degraded = AblationResult(
        component          = "retrieval",
        ablated_element    = "dense_retrieval",
        full_system_metric = 0.80,
        ablated_metric     = 0.65,
        metric_name        = "precision@5",
    )
    assert degraded.delta == pytest.approx(-0.15)

    improved = AblationResult(
        component          = "retrieval",
        ablated_element    = "dense_retrieval",
        full_system_metric = 0.60,
        ablated_metric     = 0.72,
        metric_name        = "precision@5",
    )
    assert improved.delta == pytest.approx(0.12)


def test_bronze_eval_persists_ablation_record(tmp_path, monkeypatch, capsys):
    """Frozen-set metrics and their ablation are persisted without fake seeds.

    Retrieval is mocked and the data dir is redirected to a temp path so the
    test never appends to the real experiments.jsonl.
    """
    from scilit.evaluation import bronze_eval

    # Redirect the evaluation's config.paths() to a temporary directory.
    monkeypatch.setenv("SCILIT_DATA_DIR", str(tmp_path))
    parsed = tmp_path / "parsed"
    evaluation = tmp_path / "evaluation"
    parsed.mkdir(parents=True, exist_ok=True)
    evaluation.mkdir(parents=True, exist_ok=True)
    test_set = evaluation / "frozen_relevance_test.jsonl"
    judgments = [
        {"query_id": f"q{index}", "query": f"query {index}", "relevant_chunk_ids": [f"chunk{index}"]}
        for index in range(3)
    ]
    test_set.write_text(
        "".join(json.dumps(row) + "\n" for row in judgments), encoding="utf-8"
    )

    # A pre-existing record must be preserved (append-only log).
    exp_log = parsed / "experiments.jsonl"
    exp_log.write_text(
        json.dumps({"experiment_id": "pre_existing", "ablations": []}) + "\n",
        encoding="utf-8",
    )

    # Mock scored observations, while retaining three fixed query IDs for math.
    from scilit.schemas import EvaluationObservation

    fake = {"bm25": 0.60, "dense": 0.70, "rrf": 0.80}
    monkeypatch.setattr(
        bronze_eval,
        "evaluate_modes",
        lambda judgments, k: {
            mode: {
                metric: [EvaluationObservation(
                    sample_id=row.query_id, metric_name=metric, value=value
                ) for row in judgments]
                for metric, value in ((f"Recall@{k}", fake[mode]), (f"MRR@{k}", fake[mode]))
            }
            for mode in ("bm25", "dense", "rrf")
        },
    )

    bronze_eval.run_evaluation()

    lines = exp_log.read_text(encoding="utf-8").splitlines()
    # Existing record preserved, exactly one new record appended.
    assert json.loads(lines[0])["experiment_id"] == "pre_existing"
    assert len(lines) == 2
    record = json.loads(lines[1])

    # The ablation is based on Recall@10, not keyword pseudo-precision.
    ablations = record["ablations"]
    assert len(ablations) == 1
    ab = ablations[0]
    assert ab["component"] == "retrieval"
    assert ab["ablated_element"] == "dense component removed (BM25 only vs RRF)"
    assert ab["metric_name"] == "Recall@10"
    assert ab["full_system_metric"] == pytest.approx(0.80)   # rrf mean
    assert ab["ablated_metric"] == pytest.approx(0.60)        # bm25 mean
    # delta is the computed property, consistent with the schema.
    assert ab["delta"] == pytest.approx(ab["ablated_metric"] - ab["full_system_metric"])
    assert ab["delta"] == pytest.approx(-0.20)
    assert record["dataset_split"] == "test"
    assert record["config"]["query_count"] == 3
    assert record["config"]["test_set_sha256"]
    assert all(metric["observations"] for metric in record["results"])
    assert all(not metric["seeds"] for metric in record["results"])

    # Printed result and persisted record agree.
    out = capsys.readouterr().out
    assert "-0.2" in out


if __name__ == "__main__":
    run_evaluation()
