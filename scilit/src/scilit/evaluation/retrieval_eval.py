from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scilit import config

log = logging.getLogger(__name__)
MODEL_JUDGE = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_SPECTER = "allenai/specter2_base"
JUDGE_NAME = "qwen2.5-1.5b-instruct"
SYSTEMS = ("bm25", "dense", "rrf", "bm25_title_abs", "specter2_abs")
BOOTSTRAP_SEEDS = (42, 123, 999)
METRIC_NAMES = ("P@5", "Recall@10", "MRR@10", "nDCG@10")
PROMPT_PATH = Path("benchmark/judge_prompt.md")
QUERY_PATH = Path("benchmark/queries.jsonl")
POOL_SYSTEMS = ("bm25", "dense", "bm25_title_abs", "specter2_abs")
POOL_SYSTEM_IDS = {
    "bm25": "S1",
    "dense": "S2",
    "bm25_title_abs": "S3",
    "specter2_abs": "S4",
}


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def precision_at_5(ranking: list[str], relevant: set[str]) -> float:
    return sum(item in relevant for item in ranking[:5]) / 5


def recall_at_10(ranking: list[str], relevant: set[str]) -> float | None:
    if not relevant:
        return None
    return sum(item in relevant for item in ranking[:10]) / len(relevant)


def mrr_at_10(ranking: list[str], relevant: set[str]) -> float:
    return next(
        (1 / rank for rank, item in enumerate(ranking[:10], 1) if item in relevant),
        0.0,
    )


def ndcg_at_10(ranking: list[str], relevant: set[str]) -> float:
    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, item in enumerate(ranking[:10], 1)
        if item in relevant
    )
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(10, len(relevant)) + 1))
    return dcg / ideal if ideal else 0.0


def query_metrics(ranking: list[str], relevant: set[str]) -> dict[str, float | None]:
    if not relevant:
        return {name: None for name in METRIC_NAMES}
    return {
        "P@5": precision_at_5(ranking, relevant),
        "Recall@10": recall_at_10(ranking, relevant),
        "MRR@10": mrr_at_10(ranking, relevant),
        "nDCG@10": ndcg_at_10(ranking, relevant),
    }


def _bootstrap_ci(values: list[float], seed: int, samples: int = 1000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=np.float64)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = rng.choice(data, size=len(data), replace=True).mean()
    return [float(value) for value in np.percentile(means, [2.5, 97.5])]


def _bootstrap_mean(values: list[float], seed: int, samples: int = 1000) -> float:
    if not values:
        return 0.0
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=np.float64)
    return float(np.mean([
        rng.choice(data, size=len(data), replace=True).mean()
        for _ in range(samples)
    ]))


def aggregate_system(
    per_query: list[dict],
    judged_coverage: list[float],
    component_name: str,
) -> dict:
    from scilit.schemas import ComponentMetrics, SeedResult

    aggregate = {}
    for metric in METRIC_NAMES:
        values = [
            row[metric] for row in per_query if row[metric] is not None
        ]
        seed_means = {
            str(seed): _bootstrap_mean(values, seed)
            for seed in BOOTSTRAP_SEEDS
        }
        component_metrics = ComponentMetrics(
            component_name=component_name,
            metric_name=metric,
            seeds=[
                SeedResult(
                    seed=seed,
                    metric_name=metric,
                    value=seed_means[str(seed)],
                )
                for seed in BOOTSTRAP_SEEDS
            ],
        )
        aggregate[metric] = {
            "mean": float(np.mean(values)) if values else 0.0,
            "ci95": _bootstrap_ci(values, 42),
            "bootstrap_seed_means": seed_means,
            "component_metrics": component_metrics.model_dump(),
        }
    aggregate["judged@10"] = (
        float(np.mean(judged_coverage)) if judged_coverage else 0.0
    )
    aggregate["queries_with_relevant"] = sum(
        row["relevant_count"] > 0 for row in per_query
    )
    aggregate["queries_with_zero_relevant"] = sum(
        row["relevant_count"] == 0 for row in per_query
    )
    return aggregate


def _deduplicate_article_results(results, depth: int = 15) -> list[str]:
    output = []
    seen = set()
    for result in results:
        article_id = str(result.article_id)
        if article_id not in seen:
            seen.add(article_id)
            output.append(article_id)
            if len(output) == depth:
                break
    return output


def _title_abstract_index(articles: list[dict]):
    from rank_bm25 import BM25Okapi

    from scilit.corpus.index import tokenize

    tokenized = [
        tokenize(f"{article.get('title', '')} {article.get('abstract', '')}")
        for article in articles
    ]
    return BM25Okapi(tokenized), tokenize


class Specter2Index:
    def __init__(self, articles: list[dict], index_dir: Path):
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(2)
        index_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_path = index_dir / "specter2_article_embeddings.npy"
        self.meta_path = index_dir / "specter2_article_embeddings.json"
        self.articles = articles
        ids = [str(item["id"]) for item in articles]
        if self.embeddings_path.exists() and self.meta_path.exists():
            saved_ids = json.loads(self.meta_path.read_text(encoding="utf-8"))["article_ids"]
            if saved_ids == ids:
                self.embeddings = np.load(self.embeddings_path).astype(np.float32)
            else:
                self.embeddings = None
        else:
            self.embeddings = None
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_SPECTER)
        self.model = AutoModel.from_pretrained(
            MODEL_SPECTER, low_cpu_mem_usage=True
        ).to("cpu").eval()
        if self.embeddings is None:
            texts = [
                f"{item.get('title', '')} [SEP] {item.get('abstract', '')}"
                for item in articles
            ]
            self.embeddings = self._encode(texts)
            np.save(self.embeddings_path, self.embeddings.astype(np.float16))
            self.meta_path.write_text(
                json.dumps({
                    "model": MODEL_SPECTER,
                    "article_ids": ids,
                    "text": "title [SEP] abstract",
                    "pooling": "last_hidden_state CLS",
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _encode(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        import torch

        encoded = []
        for offset in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                texts[offset : offset + batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.inference_mode():
                output = self.model(**batch)
            vectors = output.last_hidden_state[:, 0, :].float()
            encoded.append(torch.nn.functional.normalize(vectors, p=2, dim=1).cpu().numpy())
        return np.concatenate(encoded).astype(np.float32)

    def rank(self, query: str, depth: int = 15) -> list[str]:
        query_vector = self._encode([f"{query} [SEP] "])[0]
        matrix = self.embeddings.astype(np.float32)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        scores = matrix @ query_vector
        ranks = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        return [str(self.articles[index]["id"]) for index in ranks[:depth]]


def _system_rankings(
    query: str,
    articles: list[dict],
    title_index,
    title_tokenize,
    specter: Specter2Index,
    depth: int = 15,
) -> dict[str, list[str]]:
    from scilit.retrieval.pipeline import search

    chunk_results = {
        "bm25": search(query, k=100, mode="bm25"),
        "dense": search(query, k=100, mode="dense"),
        "rrf": search(query, k=100, mode="rrf", bm25_pool=100, dense_pool=100),
    }
    rankings = {
        system: _deduplicate_article_results(results, depth)
        for system, results in chunk_results.items()
    }
    scores = title_index.get_scores(title_tokenize(query))
    title_order = sorted(
        range(len(scores)), key=lambda index: (-scores[index], index)
    )
    rankings["bm25_title_abs"] = [
        str(articles[index]["id"]) for index in title_order[:depth]
    ]
    rankings["specter2_abs"] = specter.rank(query, depth)
    return rankings


def run_pool(
    query_path: Path = QUERY_PATH,
    output_path: Path | None = None,
) -> list[dict]:
    paths = config.paths()
    output_path = output_path or paths.data / "eval" / "pool.jsonl"
    queries = read_jsonl(query_path)
    articles = read_jsonl(paths.raw / "articles.jsonl")
    article_map = {str(item["id"]): item for item in articles}
    title_index, tokenize_query = _title_abstract_index(articles)
    specter = Specter2Index(articles, paths.data / "indexes")
    pooled: dict[tuple[str, str], dict] = {}
    for query in queries:
        qid = query["qid"]
        text = query.get("query_en", query.get("en", ""))
        rankings = _system_rankings(
            text, articles, title_index, tokenize_query, specter
        )
        for system in POOL_SYSTEMS:
            article_ids = rankings[system]
            for article_id in article_ids:
                key = (qid, article_id)
                row = pooled.setdefault(key, {
                    "qid": qid,
                    "article_id": article_id,
                    "title": article_map[article_id].get("title", ""),
                    "abstract": article_map[article_id].get("abstract", "")[:1500],
                    "pooled_by": [],
                })
                system_id = POOL_SYSTEM_IDS[system]
                if system_id not in row["pooled_by"]:
                    row["pooled_by"].append(system_id)
    rows = list(pooled.values())
    write_jsonl(output_path, rows)
    return rows


def _prompt_templates(path: Path = PROMPT_PATH) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    system_match = re.search(r'^System: "([^"]+)"', text, re.M)
    user_match = re.search(r'User:\n"""(.*?)"""', text, re.S)
    if not system_match or not user_match:
        raise ValueError(f"Judge prompt format not recognized: {path}")
    return system_match.group(1), user_match.group(1)


def _yes_no_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    def ids_for(variants: tuple[str, ...]) -> list[int]:
        ids = []
        for variant in variants:
            tokens = tokenizer.encode(variant, add_special_tokens=False)
            if tokens:
                ids.append(tokens[0])
        return sorted(set(ids))

    yes_ids = ids_for(("yes", "Yes", " yes", " Yes"))
    no_ids = ids_for(("no", "No", " no", " No"))
    if not yes_ids or not no_ids:
        raise ValueError("Tokenizer has no yes/no first-token variants")
    return yes_ids, no_ids


def judge_pool(
    pool_path: Path | None = None,
    output_path: Path | None = None,
    prompt_path: Path = PROMPT_PATH,
    model_name: str = MODEL_JUDGE,
    batch_size: int = 2,
) -> tuple[list[dict], float]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    paths = config.paths()
    pool_path = pool_path or paths.data / "eval" / "pool.jsonl"
    output_path = output_path or paths.data / "eval" / "qrels_silver.jsonl"
    pool = read_jsonl(pool_path)
    qrels = {
        (row["qid"], row["article_id"]): row
        for row in read_jsonl(output_path)
    } if output_path.exists() else {}
    pending = [
        row for row in pool
        if (row["qid"], row["article_id"]) not in qrels
    ]
    start_time = time.monotonic()
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to("cpu").eval()
    yes_ids, no_ids = _yes_no_token_ids(tokenizer)
    system_prompt, user_template = _prompt_templates(prompt_path)
    queries = {
        row["qid"]: row for row in read_jsonl(QUERY_PATH)
    }
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_template.format(
                            query_en=queries[row["qid"]].get(
                                "query_en", queries[row["qid"]].get("en", "")
                            ),
                            title=row.get("title", ""),
                            abstract_first_1500_chars=row.get("abstract", "")[:1500],
                        ),
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        inputs = tokenizer(
            prompts, padding=True, truncation=True, max_length=2048,
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = model(**inputs).logits[:, -1, :].float()
        yes_score = torch.logsumexp(logits[:, yes_ids], dim=1)
        no_score = torch.logsumexp(logits[:, no_ids], dim=1)
        p_yes = torch.softmax(torch.stack((yes_score, no_score), dim=1), dim=1)[:, 0]
        for row, probability in zip(batch, p_yes.tolist()):
            key = (row["qid"], row["article_id"])
            qrels[key] = {
                "qid": row["qid"],
                "article_id": row["article_id"],
                "p_yes": float(probability),
                "relevant": probability >= 0.5,
                "judge": JUDGE_NAME,
                "pooled_by": row["pooled_by"],
            }
        write_jsonl(
            output_path,
            [qrels[(row["qid"], row["article_id"])] for row in pool
             if (row["qid"], row["article_id"]) in qrels],
        )
        del inputs, logits, yes_score, no_score, p_yes
    runtime = time.monotonic() - start_time
    (paths.data / "eval" / "judge_runtime.json").write_text(
        json.dumps({"seconds": runtime, "judged": len(qrels), "model": model_name}, indent=2),
        encoding="utf-8",
    )
    return list(qrels.values()), runtime


def _rankings_for_systems(
    queries: list[dict], articles: list[dict], depth: int = 100
) -> dict[str, dict[str, list[str]]]:
    title_index, tokenize_query = _title_abstract_index(articles)
    specter = Specter2Index(articles, config.paths().data / "indexes")
    results = {system: {} for system in SYSTEMS}
    for query in queries:
        qid = query["qid"]
        text = query.get("query_en", query.get("en", ""))
        rankings = _system_rankings(
            text, articles, title_index, tokenize_query, specter, depth
        )
        for system in SYSTEMS:
            results[system][qid] = rankings[system]
    return results


def evaluate(
    systems: tuple[str, ...] = SYSTEMS,
    query_path: Path = QUERY_PATH,
    qrels_path: Path | None = None,
    output_path: Path | None = None,
) -> dict:
    paths = config.paths()
    qrels_path = qrels_path or paths.data / "eval" / "qrels_silver.jsonl"
    output_path = output_path or paths.data / "eval" / "retrieval_bronze.json"
    queries = read_jsonl(query_path)
    articles = read_jsonl(paths.raw / "articles.jsonl")
    qrels_rows = read_jsonl(qrels_path)
    relevant = defaultdict(set)
    judged = defaultdict(set)
    for row in qrels_rows:
        judged[row["qid"]].add(row["article_id"])
        if row["relevant"]:
            relevant[row["qid"]].add(row["article_id"])
    rankings = _rankings_for_systems(queries, articles)
    per_query = {}
    aggregate = {}
    for system in systems:
        rows, coverage = [], []
        for query in queries:
            qid = query["qid"]
            ranking = rankings[system][qid]
            metric_values = query_metrics(ranking, relevant[qid])
            coverage.append(
                len(set(ranking[:10]) & judged[qid]) / min(10, len(ranking))
                if ranking else 0.0
            )
            rows.append({
                "qid": qid,
                "ranking": ranking[:10],
                "relevant_count": len(relevant[qid]),
                "judged@10": coverage[-1],
                **metric_values,
            })
        per_query[system] = rows
        aggregate[system] = aggregate_system(rows, coverage, system)
    result = {
        "systems": list(systems),
        "per_query": per_query,
        "aggregate": aggregate,
        "qrels": {
            "judged": len(qrels_rows),
            "relevant": sum(row["relevant"] for row in qrels_rows),
            "relevant_per_query": {
                query["qid"]: len(relevant[query["qid"]]) for query in queries
            },
            "queries_with_zero_relevant": sum(
                not relevant[query["qid"]] for query in queries
            ),
        },
        "bootstrap": {
            "resamples": 1000,
            "ci_seed": 42,
            "component_seeds": BOOTSTRAP_SEEDS,
            "seed_semantics": (
                "Deterministic bootstrap seeds over queries, not training seeds."
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def evaluate_robustness(
    query_path: Path = QUERY_PATH,
    qrels_path: Path | None = None,
    output_path: Path | None = None,
) -> dict:
    paths = config.paths()
    qrels_path = qrels_path or paths.data / "eval" / "qrels_silver.jsonl"
    output_path = output_path or paths.data / "eval" / "robustness.json"
    queries = read_jsonl(query_path)
    articles = read_jsonl(paths.raw / "articles.jsonl")
    qrels = read_jsonl(qrels_path)
    relevant = defaultdict(set)
    for row in qrels:
        if row["relevant"]:
            relevant[row["qid"]].add(row["article_id"])
    title_index, tokenize_query = _title_abstract_index(articles)
    specter = Specter2Index(articles, paths.data / "indexes")
    from scilit.retrieval.pipeline import search

    records = {system: [] for system in SYSTEMS}
    for query in queries:
        qid = query["qid"]
        original = query.get("query_en", query.get("en", ""))
        variants = query.get("reformulations", [])
        texts = [original, *variants]
        top_by_system: dict[str, list[list[str]]] = {system: [] for system in SYSTEMS}
        for text in texts:
            chunk_res = {
                "bm25": search(text, k=100, mode="bm25"),
                "dense": search(text, k=100, mode="dense"),
                "rrf": search(text, k=100, mode="rrf", bm25_pool=100, dense_pool=100),
            }
            ranked = {
                name: _deduplicate_article_results(value, 10)
                for name, value in chunk_res.items()
            }
            scores = title_index.get_scores(tokenize_query(text))
            order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
            ranked["bm25_title_abs"] = [
                str(articles[index]["id"]) for index in order[:10]
            ]
            ranked["specter2_abs"] = specter.rank(text, 10)
            for system in SYSTEMS:
                top_by_system[system].append(ranked[system])
        for system in SYSTEMS:
            original_top, reform_top = top_by_system[system][0], top_by_system[system][1:]
            recalls = [
                recall_at_10(ranking, relevant[qid])
                for ranking in reform_top
            ]
            valid_recalls = [value for value in recalls if value is not None]
            jaccard = [
                len(set(original_top) & set(ranking)) / len(set(original_top) | set(ranking))
                if set(original_top) | set(ranking) else 1.0
                for ranking in reform_top
            ]
            records[system].append({
                "qid": qid,
                "original_recall@10": recall_at_10(original_top, relevant[qid]),
                "reformulation_mean_recall@10": (
                    float(np.mean(valid_recalls)) if valid_recalls else None
                ),
                "top10_jaccard": float(np.mean(jaccard)) if jaccard else None,
            })
    summary = {}
    for system, rows in records.items():
        original_values = [row["original_recall@10"] for row in rows
                           if row["original_recall@10"] is not None]
        reform_values = [row["reformulation_mean_recall@10"] for row in rows
                         if row["reformulation_mean_recall@10"] is not None]
        overlap_values = [row["top10_jaccard"] for row in rows
                          if row["top10_jaccard"] is not None]
        summary[system] = {
            "mean_original_recall@10": float(np.mean(original_values)) if original_values else 0.0,
            "mean_reformulation_recall@10": float(np.mean(reform_values)) if reform_values else 0.0,
            "mean_top10_jaccard": float(np.mean(overlap_values)) if overlap_values else 0.0,
            "per_query": rows,
        }
    output = {"systems": summary}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def append_experiment(result: dict, path: Path | None = None) -> None:
    from scilit.schemas import ComponentMetrics, ExperimentRecord, SeedResult, SplitType

    path = path or config.paths().parsed / "experiments.jsonl"
    metrics = []
    for system, values in result["aggregate"].items():
        for name in METRIC_NAMES:
            metric = values[name]
            metrics.append(ComponentMetrics(
                component_name=system,
                metric_name=name,
                seeds=[
                    SeedResult(
                        seed=seed,
                        metric_name=name,
                        value=metric["bootstrap_seed_means"][str(seed)],
                    )
                    for seed in BOOTSTRAP_SEEDS
                ],
            ))
    record = ExperimentRecord(
        experiment_id="exp_002_bronze_retrieval",
        date=datetime.now(timezone.utc),
        component="silver_retrieval_evaluation",
        hypothesis="Lexical, dense, hybrid, title-abstract BM25 and SPECTER2 retrieval differ on judged literature-review relevance.",
        config={
            "systems": list(result["systems"]),
            "judge": JUDGE_NAME,
            "bootstrap_resamples": 1000,
            "qrels": result["qrels"]["judged"],
        },
        dataset_split=SplitType.TEST,
        results=metrics,
        error_analysis_count=0,
        notes=(
            "The three seeds are deterministic bootstrap resampling seeds over "
            "queries, not training seeds. Phase 2 silver Qwen judgments; "
            "error samples remain uncategorized."
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(record.model_dump_json() + "\n")


def generate_error_samples(eval_dir: Path | None = None) -> dict[str, int]:
    paths = config.paths()
    eval_dir = eval_dir or paths.data / "eval"
    claims = read_jsonl(paths.parsed / "claims.jsonl")
    chunks = {
        row["id"]: row.get("text", "")
        for row in read_jsonl(paths.parsed / "chunks.jsonl")
    }
    sampled_claims = random.Random(42).sample(claims, min(40, len(claims)))
    claim_rows = []
    for claim in sampled_claims:
        chunk_text = chunks.get(str(claim.get("chunk_id")), "")
        phrase = claim.get("verbatim") or claim.get("text") or ""
        position = chunk_text.find(phrase) if phrase else -1
        center = position if position >= 0 else 0
        start = max(0, center - 200)
        end = min(len(chunk_text), center + len(phrase) + 200)
        claim_rows.append({
            "claim_id": claim.get("id"),
            "claim_type": claim.get("claim_type"),
            "hedging": claim.get("hedging_level", claim.get("hedging")),
            "text": claim.get("text", ""),
            "chunk_context": chunk_text[start:end],
        })
    write_jsonl(eval_dir / "error_samples_claims.jsonl", claim_rows)

    query_rows = read_jsonl(QUERY_PATH)
    query_by_id = {row["qid"]: row for row in query_rows}
    articles = {
        str(row["id"]): row for row in read_jsonl(paths.raw / "articles.jsonl")
    }
    qrels_rows = read_jsonl(eval_dir / "qrels_silver.jsonl")
    qrels = {(row["qid"], row["article_id"]): row for row in qrels_rows}
    relevant_by_query = defaultdict(set)
    for row in qrels_rows:
        if row["relevant"]:
            relevant_by_query[row["qid"]].add(row["article_id"])
    retrieval = json.loads(
        (eval_dir / "retrieval_bronze.json").read_text(encoding="utf-8")
    )
    retrieval_rows = []
    for row in retrieval["per_query"]["rrf"]:
        if row["Recall@10"] is None or row["Recall@10"] >= 0.5:
            continue
        qid = row["qid"]
        missed = sorted(
            relevant_by_query[qid] - set(row["ranking"]),
            key=lambda article_id: (
                -qrels[(qid, article_id)]["p_yes"], article_id
            ),
        )[:3]
        top = row["ranking"][:5]
        retrieval_rows.append({
            "qid": qid,
            "query": query_by_id[qid].get("query_en", query_by_id[qid].get("en", "")),
            "top5_rrf": [
                {
                    "article_id": article_id,
                    "title": articles[article_id].get("title", ""),
                    "p_yes": qrels.get((qid, article_id), {}).get("p_yes"),
                }
                for article_id in top
            ],
            "missed_relevant": [
                {
                    "article_id": article_id,
                    "title": articles[article_id].get("title", ""),
                    "abstract": articles[article_id].get("abstract", "")[:300],
                }
                for article_id in missed
            ],
        })
    write_jsonl(eval_dir / "error_samples_retrieval.jsonl", retrieval_rows)

    relevant_rows = [row for row in qrels_rows if row["relevant"]]
    nonrelevant_rows = [row for row in qrels_rows if not row["relevant"]]
    rng = random.Random(42)
    audit_rows = []
    for group in (relevant_rows, nonrelevant_rows):
        audit_rows.extend(rng.sample(group, min(20, len(group))))
    audit_sample = []
    for row in audit_rows:
        article = articles[row["article_id"]]
        audit_sample.append({
            "qid": row["qid"],
            "query": query_by_id[row["qid"]].get(
                "query_en", query_by_id[row["qid"]].get("en", "")
            ),
            "article_id": row["article_id"],
            "title": article.get("title", ""),
            "abstract": article.get("abstract", "")[:600],
            "p_yes": row["p_yes"],
            "relevant": row["relevant"],
        })
    write_jsonl(eval_dir / "judge_audit_sample.jsonl", audit_sample)
    return {
        "claims": len(claim_rows),
        "retrieval_errors": len(retrieval_rows),
        "audit_relevant": min(20, len(relevant_rows)),
        "audit_nonrelevant": min(20, len(nonrelevant_rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SciLit judged retrieval evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    pool_parser = subparsers.add_parser("pool")
    pool_parser.add_argument("--queries", type=Path, default=QUERY_PATH)
    pool_parser.add_argument("--output", type=Path)
    judge_parser = subparsers.add_parser("judge")
    judge_parser.add_argument("--pool", type=Path)
    judge_parser.add_argument("--output", type=Path)
    judge_parser.add_argument("--prompt", type=Path, default=PROMPT_PATH)
    judge_parser.add_argument("--model", default=MODEL_JUDGE)
    judge_parser.add_argument("--batch-size", type=int, default=2)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--systems", default=",".join(SYSTEMS))
    evaluate_parser.add_argument("--queries", type=Path, default=QUERY_PATH)
    evaluate_parser.add_argument("--qrels", type=Path)
    args = parser.parse_args()
    if args.command == "pool":
        rows = run_pool(args.queries, args.output)
        print(f"pooled {len(rows)} qid/article pairs")
    elif args.command == "judge":
        qrels, runtime = judge_pool(
            args.pool, args.output, args.prompt, args.model, args.batch_size
        )
        print(f"judged {len(qrels)} pairs in {runtime:.1f}s")
    else:
        systems = tuple(item.strip() for item in args.systems.split(",") if item.strip())
        result = evaluate(systems, args.queries, args.qrels)
        evaluate_robustness(args.queries, args.qrels)
        append_experiment(result)
        samples = generate_error_samples()
        print(f"error samples: {samples}")
        print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
