# SciLit

SciLit is a Python research-literature pipeline for collecting papers, indexing text, retrieving passages, extracting claims/citations, generating a template-based literature-review draft, and optionally auditing citation support with NLI.

## Configure environment

Python 3.10 or newer is required.

1. Create and activate a virtual environment:
   ```bash
   .venv\Scripts\Activate.ps1
   ```
2. Install the package and its dependencies from the repository root:

   ```bash
   pip install -e ".[dev]"
   ```

3. Copy `.env.example` to `.env` and edit the values. The app loads `.env` automatically at startup; already-exported environment variables take precedence over `.env`.

   ```bash
   # Windows PowerShell
   Copy-Item .env.example .env

   # macOS / Linux
   cp .env.example .env
   ```

4. Start the API so `API_HOST`, `API_PORT`, and `APP_ENV` are applied:

   ```bash
   python -m scilit.api.main 
   or 
   .\.venv\Scripts\python.exe -m scilit.api.main
   ```

   Interactive API docs: `http://127.0.0.1:8000/docs` (adjust the host/port to your `.env`).
   SciLit user interface: `http://127.0.0.1:8000/ui`.

## Environment variables

- `APP_NAME`: FastAPI title.
- `APP_ENV`: set to `development` to enable Uvicorn reload when starting with `python -m scilit.api.main`; other values disable reload.
- `API_HOST`, `API_PORT`: API bind address and port.
- `DEFAULT_TOP_K`: default retrieval depth for API, CLI retrieval, and review generation (valid range: 1–50).
- `RETRIEVAL_MODE`: `rrf`, `bm25`, or `dense`.
- `DATA_DIR`: corpus/index/review data directory. Relative paths are resolved from the repository root.
- `SCILIT_ROOT` and the legacy `SCILIT_DATA_DIR` remain supported for existing setups.
- `OPENAI_API_KEY`, `GROQ_API_KEY`: loaded for future/provider integrations. The current Bronze review generator is deterministic/template-based and does not call either LLM provider.

**Security:** `.env` is ignored by Git. Do not commit API keys; commit `.env.example` with blank values instead.

## Existing data

By default, SciLit expects its corpus and indexes under `data/raw/` and `data/parsed/`. Configure `DATA_DIR` if the data is stored elsewhere.

## Evidence and verification behavior

- Generation checks the query at claim level after retrieval. A retrieved chunk does not make every claim in that chunk relevant.
- Claims must be complete sentences and recoverable verbatim from their retrieved source chunk. SciLit does not add punctuation to a fragment or invent a fallback result sentence.
- Repeated source claims are deduplicated, and no more than two claims from one paper are selected for a review.
- Each generated evidence sentence stores its claim ID, article ID, exact source passage, article link, and parsed section offsets when available. Open `/ui` to inspect the passage below each sentence.
- A comparison between RAG and NLP is rejected with a clarification prompt because RAG is a technique within NLP.

NLI outputs are automatic indicators, not scientific certification. SciLit reports NLI-supported, no-support, contradicted, unresolved-reference, and unverifiable cases separately. It checks only claims explicitly linked to the cited article; it does not use arbitrary article chunks as substitute evidence. Sentences without citations are excluded from citation support rates.

### French and Arabic queries

The original query is retained in the review. For French or Arabic queries, SciLit also creates an English translation locally with an open Helsinki-NLP/OPUS-MT model, then searches with both the original and translated query and merges their results. The translation model is downloaded from Hugging Face on first use if it is not already cached; query text is not sent to a translation API. Override the model IDs with `SCILIT_TRANSLATION_MODEL_FR` or `SCILIT_TRANSLATION_MODEL_AR`. If loading or translation fails, SciLit logs that fact and continues with the original query rather than bypassing evidence checks.

Query-term matching recognizes Arabic script, French accents, and English technical terms. A translated query can improve retrieval and claim relevance, but it cannot create source evidence: claims still need to match the retrieved article and current chunk text verbatim.

### Generation diagnostics

Each generation writes a structured `generation_stages` log entry without the raw query, query terms, or source passages. Retrieval counts distinguish returned chunks from chunks with current metadata, matching article IDs, and matching source text. `associated_with_retrieved_chunks` counts claims whose chunk IDs occur in the retrieved set; `not_associated_with_retrieved_chunks` is the remaining corpus-wide claim count, not a claim rejection. `complete`, `verbatim_in_chunk`, and `relevance_pass` are cumulative stage-pass counts. Relevance diagnostics report distinct query-token counts before and after filtering, per-variant claim-overlap histograms, title-context passes, and aggregate rejection reasons without recording text. A relevance pass requires at least one meaningful query term in the claim itself; for longer queries a second distinct term may corroborate the claim from the article title, but the title cannot pass by itself. The `rejections` map assigns each associated claim to only its first failing check; its comparison count applies only when a comparison gate rejects a set of otherwise relevant claims. `final_eligible` counts source-backed claims actually included in the generated draft. Offset counters distinguish valid offsets from an invalid offset recovered by an exact text search. Comparison counters report the number of source candidates per side and missing sides without logging the side text.

Retrieval also logs index metadata checks against current `chunks.jsonl` (unknown chunk IDs, article-ID mismatches, and text-preview mismatches). Empty retrieval and retrieved-but-ineligible evidence have distinct 422 errors; the latter includes claim-stage counts. Unexpected generation failures are logged server-side and returned as a generic API error. SciLit does not rebuild the corpus or indexes automatically.

## Frozen retrieval evaluation

Run the scientific retrieval comparison with:

```powershell
python -m scilit.evaluation.bronze_eval
```

The command requires a human-annotated, frozen file at `data/evaluation/frozen_relevance_test.jsonl` (or the configured `DATA_DIR`). Each non-empty line must contain a unique `query_id`, a `query`, and at least one annotated `relevant_chunk_ids` value. BM25, dense retrieval, and RRF are evaluated on that same set using Recall@10 and MRR. The file's SHA-256 is stored with the experiment record. If the annotations are absent or invalid, the evaluator stops without substituting keyword pseudo-labels.

The current repository does not include that human-annotated test set. Results from the old keyword-overlap smoke checks and NLI model scores must not be presented as validated scientific metrics.

## Collection provenance

Newly collected `Article` records include a work-level `ProvenanceRecord` for
OpenAlex and separate typed entries for each known source location. Each
location keeps its own landing page, PDF URL, version, and `LicenseInfo`;
location roles identify which copy backs the citation URL, PDF URL, and
`ArticleSource.license` evidence. `ArticleSource` remains the pipeline's
selected-source summary, and license permission outcomes remain separate from
provenance.

`ProvenanceRecord.retrieved_at` is the authoritative UTC retrieval time. The
legacy `ArticleSource.retrieved_at` field mirrors it for newly collected
records. The OpenAlex DOI in provenance is the normalized evidence value;
`Article.doi` and `ArticleSource.doi` retain the same normalized value for
existing callers. Missing provider fields remain absent, while inferred
licenses retain their evidence and inference source in `LicenseInfo`.

Historical articles without provenance continue to load with an empty
provenance list. No corpus migration or recollection is performed as part of
loading; controlled recollection is needed to populate provenance for those
records.

## Abstract and full-text parsing

Parsing defaults to abstracts and makes no PDF requests:

```powershell
python -m scilit.corpus.parse
python -m scilit.corpus.parse --mode abstract
```

Full-text extraction must be selected explicitly. The legacy `--mode pdf`
option remains supported, as does the explicit flag:

```powershell
python -m scilit.corpus.parse --full-text-mode
python -m scilit.corpus.parse --mode pdf --max 50
```

`--mode abstract` and `--no-full-text-mode` select abstract parsing. Supplying
conflicting values to `--mode` and `--full-text-mode` is an error. Full-text
mode uses the PDF URL recorded for an article or its arXiv identifier, caches
PDFs under `data/raw/pdfs/`, enforces the configured request timeout and
20 MB size limit, and uses PyMuPDF with heuristic section detection. It does
not substitute the provider abstract when no PDF is available or extraction
fails. Articles without a PDF are skipped; download, malformed-PDF, and
no-text failures are logged and produce no sections or chunks.

Abstract-only retrieval is limited to information present in abstracts.
Methods, datasets, numerical results, or limitations omitted from an abstract
cannot reliably be recovered, and a citation audit against abstracts is not
equivalent to checking the complete paper. Literature reviews based on this
mode can omit evidence that appears only in the main text. Inspect the run log
with `Get-Content data/parsed/parse_log.txt` in PowerShell, or open that file
in a text editor.

When no eligible cached PDF exists, full-text mode needs network access to the
article's existing supported PDF URL. PyMuPDF must be installed; no model or
additional provider is used. A readable or open-access PDF does not by itself
establish permission to redistribute it, and parsing does not change the
location-specific license or provenance evidence. PDF extraction can miss
papers or sections, misread tables and numerical results, and does not verify
the scientific accuracy of extracted claims or their citation support.

Each run records its requested and effective mode, input and selected article
counts, processed abstract or full-text article counts, failures, missing-PDF
skips, sections, and chunks in `data/parsed/parse_log.txt`. The JSONL outputs
are append-oriented aggregate files, so they may contain records from earlier
runs or both modes; the run summary in the log is the per-run count. Use a
separate `DATA_DIR` when you need isolated outputs for a full-text run.

The corpus snapshot inspected on 2026-10-16 has a parse log from 2026-10-14
recording 1,500 abstract articles, 1,500 sections, and 4,062 chunks. The
sections in that snapshot are all marked `abstract`; it should not be described
as a full-text corpus. Abstracts omit the body of each paper. Full-text output
depends on an accessible PDF and is not comprehensive: the current font-size
and heading-pattern heuristics can misclassify sections, and extraction does
not reliably preserve reading order, tables, figures, equations, or other page
layout. A successful extraction does not guarantee that every scientific
claim or section was captured.
