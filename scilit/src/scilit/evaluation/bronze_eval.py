"""Run retrieval comparisons on a frozen, human-annotated test set.

Expected input: ``data/evaluation/frozen_relevance_test.jsonl``. The same query
judgments are scored for BM25, dense retrieval, and RRF using Recall@10 and MRR.
If annotations are absent, the run stops without writing fabricated metrics.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

from scilit import config
from scilit.evaluation.frozen import evaluate_modes, load_relevance_judgments

log = logging.getLogger(__name__)
K = 10


def run_evaluation() -> None:
    from scilit.schemas import (
        AblationResult,
        ComponentMetrics,
        ExperimentRecord,
        SplitType,
    )

    paths = config.paths()
    test_set_path = paths.data / "evaluation" / "frozen_relevance_test.jsonl"
    judgments = load_relevance_judgments(test_set_path)
    outputs = evaluate_modes(judgments, k=K)
    digest = hashlib.sha256(test_set_path.read_bytes()).hexdigest()

    metrics: list[ComponentMetrics] = []
    for mode in ("bm25", "dense", "rrf"):
        for metric_name, observations in outputs[mode].items():
            metrics.append(ComponentMetrics(
                component_name=f"{mode}_retrieval",
                metric_name=metric_name,
                observations=observations,
            ))

    recall_metric = {item.component_name: item for item in metrics if item.metric_name == f"Recall@{K}"}
    ablation = AblationResult(
        component="retrieval",
        ablated_element="dense component removed (BM25 only vs RRF)",
        full_system_metric=recall_metric["rrf_retrieval"].mean,
        ablated_metric=recall_metric["bm25_retrieval"].mean,
        metric_name=f"Recall@{K}",
    )
    record = ExperimentRecord(
        experiment_id=f"retrieval_test_{digest[:12]}",
        date=datetime.now(timezone.utc),
        component="retrieval_modes",
        hypothesis="Compare BM25, dense retrieval, and RRF on the same frozen relevance judgments.",
        config={
            "test_set": str(test_set_path),
            "test_set_sha256": digest,
            "query_count": len(judgments),
            "k": K,
            "modes": ["bm25", "dense", "rrf"],
            "judgment_source": "human-annotated frozen test set",
        },
        dataset_split=SplitType.TEST,
        results=metrics,
        ablations=[ablation],
        error_analysis_count=0,
        notes=(
            "All modes use the same query IDs and human relevance labels. "
            "This record evaluates retrieval only; automatic NLI scores are not "
            "human-validated citation-support results. Do not tune on this frozen set."
        ),
    )

    output_path = paths.parsed / "experiments.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json() + "\n")

    print("\nFrozen retrieval evaluation — SciLit")
    print(f"Test set: {len(judgments)} queries | SHA-256 {digest}")
    print(f"{'Mode':<18} {'Metric':<12} {'Mean':>8} {'Std':>8}")
    for item in metrics:
        print(f"{item.component_name:<18} {item.metric_name:<12} {item.mean:>8.3f} {item.std:>8.3f}")
    print(f"RRF vs BM25 delta Recall@{K}: {ablation.delta:+.3f}")
    print("Citation support and NLI certification are not evaluated by this run.")
    log.info("Frozen retrieval experiment recorded at %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_evaluation()
