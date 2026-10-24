# Retrieval evaluation (Part 12a)

30 test queries · labels: `labels/qrels_chatgpt.tsv`

| System | Recall@10 | MRR | nDCG@10 |
| --- | --- | --- | --- |
| bm25 (EN) | 0.652 | 0.950 | 0.823 |
| dense (EN) | 0.621 | 0.888 | 0.786 |
| hybrid (EN) | 0.654 | 0.883 | 0.818 |
| hybrid_rerank (EN) | 0.684 | 0.928 | 0.827 |
| hybrid_rerank (FR) | 0.486 | 0.864 | 0.634 |
| bm25, fixed-size chunks | 0.513 | 0.893 | 0.673 |

Paired bootstrap (10,000 resamples over queries), hybrid_rerank vs bm25 on nDCG@10: p = 0.4231
