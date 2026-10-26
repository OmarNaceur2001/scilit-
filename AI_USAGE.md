# Declaration of AI assistant use

| Assistant | Used for | How the output was checked |
| --- | --- | --- |
| Claude (Anthropic) | Help with code, debugging, analysis of results, documentation | Every change run and evaluated on the frozen test sets; numbers reported only from the scripts' outputs |
| ChatGPT (OpenAI) | Relevance labels for retrieval (`labels/qrels_chatgpt.tsv`), labels of the verifier test set (`labels/verify/sheet_chatgpt.csv`) | Declared as LLM labels in every report; 52 retrieval pairs also labelled by a human (kappa 0.64 relevant / not relevant) |
| Claude (Anthropic) | Labels of 150 section headings (`labels/sections/headings.csv`) | Declared as LLM labels in `reports/bronze_eval.md` |

Models used inside the system (Qwen2.5-7B, Bespoke-MiniCheck-7B, BGE-M3, bge-reranker-v2-m3, SciBERT,
DeBERTa-v3) are components of SciLit, not assistants; they are described in the README. The Bronze
pipeline and its demo use none of them.
