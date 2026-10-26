# SciLit

**A literature-review assistant for NLP research that only writes what it can prove.**

Ask a research question. SciLit finds the relevant papers in a corpus of 1,109 NLP papers, reads each one, writes a short review, and checks every sentence against the original text. A sentence that the sources do not support is removed, and every remaining sentence links back to the exact sentences in the PDF.

Everything runs locally and uses open models only.

## How it works

```
question
   │
   ├─ 1. Retrieve     BM25 + BGE-M3 dense search, fused with RRF, reranked by bge-reranker-v2-m3
   │                  → the 8 most relevant papers
   ├─ 2. Read         per paper: its own sentences only (no related work, no citing sentences),
   │                  the abstract + the 3 passages most relevant to the question
   │                  → Qwen2.5-7B writes a reading card: is the paper relevant, and up to 4 claims,
   │                    each pointing to its source sentences
   ├─ 3. Verify       every claim is checked against its source sentences
   │                  (numbers, attribution, Bespoke-MiniCheck-7B fact checker)
   ├─ 4. Plan         Qwen groups the verified claims into points (claim IDs only, no text yet)
   ├─ 5. Write        one sentence per point, written from that point's claims only
   └─ 6. Verify       every sentence is checked against the original source sentences;
                      unsupported sentences are removed and shown with the reason
```

Design choices that matter:

- **Citations are fixed before any text is written.** The model picks claim IDs from a closed list (constrained JSON decoding), then writes from those claims only. It cannot cite a source that does not exist or the wrong one.
- **The model never sees author names.** Who said what comes from the citations, so a result cannot be attributed to the wrong paper.
- **The verifier reads the original PDF sentences, never the model's own text.**

The corpus pipeline: OpenAlex + arXiv metadata, deduplication, a PDF quality gate, a layout-aware PyMuPDF parser (sections, sentences, references, in-text citations) and structure-aware chunking. A fine-tuned SciBERT labels sentence roles (objective, method, result…) to fill a comparison table across papers.

## Results

| What | Data | Result |
|---|---|---|
| Retrieval, nDCG@10 | 30 test questions | hybrid + rerank **0.827**, BM25 0.823 (difference not significant, p = 0.42) |
| Sentence roles, macro-F1 | CSAbstruct, deduplicated splits | SciBERT **0.749 ± 0.005**, TF-IDF 0.691, rules 0.357 (p ≤ 0.001) |
| Citation function, macro-F1 | SciCite | SciBERT **0.855 ± 0.017**, published 0.855 (reproduced) |
| Review sentences supported | 30 test + 12 held-out questions | **95% / 93%**, vs 53% / 67% for single-call generation on the same reading cards (p < 0.001, DeBERTa verifier) |
| Verifier accuracy | 113 sentence–source pairs, ChatGPT-labelled | MiniCheck: balanced accuracy **0.88**, 93% of accepted sentences supported; DeBERTa-v3 NLI: 0.62, 69% |
| Review with MiniCheck | 30 test + 12 held-out questions | **92% / 92%** of sentences accepted, 82–84% of verified claims used |

How these numbers were measured:

- Classifiers: 3 seeds, model chosen on dev, reported on test. Sentences that appear in both train and test in the official CSAbstruct splits (12.7% of test) were removed.
- Review: both designs are run on identical saved reading cards, so the difference comes from the synthesis step only. The held-out questions were written after the design was fixed and never inspected before the final run.
- "Supported" is the verifier's judgement. The verifiers themselves were tested on real pairs plus hard cases (changed or swapped numbers, reversed findings, swapped entities, plausible but unstated facts); MiniCheck replaced DeBERTa as the default after this test.
- Retrieval relevance labels were produced by an LLM and are declared as such.

## Quick start

Requirements: Python 3.10+, [Ollama](https://ollama.com), a GPU with 8 GB is enough.

```bash
pip install -e .
ollama pull qwen2.5:7b
ollama pull bespoke-minicheck
```

Build the corpus and indexes (one time):

```bash
python -m scilit.corpus.collect          # OpenAlex search, seeds, citation snowball
python -m scilit.corpus.normalize        # metadata correction, deduplication
python -m scilit.corpus.fetch_pdfs
python -m scilit.corpus.validate         # PDF quality gate
python -m scilit.corpus.parse_pdf
python -m scilit.corpus.make_splits      # frozen dev / test split
python -m scilit.corpus.chunk
python -m scilit.retrieval.build_index --bm25
python -m scilit.retrieval.build_index --dense BAAI/bge-m3
python -m scilit.extraction.roles        # train SciBERT sentence roles
python -m scilit.extraction.extract      # reading grid for the comparison table
```

Ask a question:

```bash
python -m scilit.generation.review "How does retrieval-augmented generation reduce hallucination?"
```

Or use the web interface:

```bash
uvicorn scilit.api.main:app --port 8000      # then open http://127.0.0.1:8000
```

`collect` uses the OpenAlex API: set `OPENALEX_API_KEY` in your environment (see `.env.example`). Never commit keys.

## Evaluation

```bash
python -m scilit.evaluation.retrieval_eval                          # retrieval metrics
python -m scilit.evaluation.review_eval run --set test              # full pipeline, saves reading cards
python -m scilit.evaluation.review_eval replay --source test --tag test_single --synthesis single
python -m scilit.evaluation.review_eval compare --tags test test_single
python -m scilit.evaluation.review_eval report
python -m scilit.evaluation.review_eval verify-set --source test    # verifier test set for human labelling
python -m scilit.evaluation.review_eval verify-score --which deberta minicheck
```

The slow step (search + reading) runs once; every comparison replays the saved reading cards, so two configurations differ in exactly one thing.

## Project layout

```
scilit/src/scilit/
  corpus/       collection, metadata, PDF validation and parsing, chunking
  retrieval/    BM25 and dense indexes, hybrid search and reranking
  extraction/   SciBERT sentence roles, citation function, reading grid
  generation/   reader, synthesis, verifier, comparison table, LLM client
  evaluation/   retrieval and end-to-end evaluation, verifier test set
  api/          FastAPI server and web interface
configs/        themes and seed papers
labels/         test and held-out questions, relevance labels, verifier labels
reports/        evaluation results
```

## Limitations

- The corpus has 1,109 papers: a question outside it gets a thin answer or none. SciLit says so instead of inventing.
- The reader sees about 30 sentences per paper; findings only in tables or appendices can be missed.
- The verifier is not perfect: on the test set MiniCheck still accepts about 8% of unsupported sentences.
- The LLM runs at temperature 0, but GPU inference is not bitwise reproducible: averages are stable, individual reviews can vary slightly.
- English only for now: questions in French retrieve English papers, but reviews are written in English.
