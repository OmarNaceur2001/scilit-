# SciLit

SciLit is a Python research-literature pipeline for collecting papers, indexing text, retrieving passages, extracting claims/citations, generating a template-based literature-review draft, and optionally auditing citation support with NLI.

## Configure environment

Python 3.10 or newer is required.

1. Create and activate a virtual environment.
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
   ```

   Interactive API docs: `http://127.0.0.1:8000/docs` (adjust the host/port to your `.env`).

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
