# Relevance judge (silver qrels) — lead-authored, do not modify
Model: Qwen/Qwen2.5-1.5B-Instruct (transformers, CPU, greedy, chat template).
System: "You are an expert NLP researcher judging paper relevance for a literature review."
User:
"""Research question: {query_en}

Paper title: {title}
Abstract: {abstract_first_1500_chars}

Would this paper be useful to cite in a literature review answering the research question? It is relevant only if it directly studies, proposes, evaluates or compares the methods or problems central to the question. Sharing generic keywords is not enough.
Answer with exactly one word: yes or no."""
Score: from logits of the first generated token after the assistant prefix, p = softmax over the two token ids of "yes" and "no" (also sum " yes"/"Yes" variants: take logsumexp over {"yes","Yes"," yes"," Yes"} vs same for no). relevant = p_yes >= 0.5. Store p_yes.

# Pooling (article level), depth 15 per system, union:
S1 BM25 over chunks (existing index) -> dedupe to articles in rank order
S2 MiniLM dense over chunks (existing FAISS) -> dedupe
S3 BM25 over title+abstract (article level, rank_bm25, same tokenizer as index.py)
S4 allenai/specter2_base [CLS] embeddings of "title [SEP] abstract" (article level, cosine), query embedded same way (title=query, empty abstract)
Queries judged with the EN text only. qrels file: data/eval/qrels_silver.jsonl {qid, article_id, p_yes, relevant, judge:"qwen2.5-1.5b-instruct", pooled_by:[S..]}.

# Metrics (article-level ranking: retrieve 100 chunks, dedupe by article keeping first rank; article-level systems rank articles directly)
P@5, Recall@10 = |rel ∩ top10| / |rel| (queries with |rel|=0 excluded and counted/reported), MRR@10 (1/rank of first relevant in top10 else 0), nDCG@10 binary gains, ideal from |rel|.
Report mean over queries + bootstrap 95% CI (1000 resamples of queries). "3 seeds" for deterministic systems = 3 bootstrap seeds 42,123,999 each giving a bootstrap-mean (state this explicitly); for stochastic components seeds are real training/sampling seeds.
Unjudged retrieved docs count as non-relevant; report judged@10 coverage per system.
