#!/usr/bin/env python3
"""Connect the SciLit project to .env without altering research logic.

Run from anywhere:
    python scilit_env_setup.py /path/to/scilit-

The script checks every expected source snippet before writing any source file.
It never overwrites an existing .env file. Existing secrets stay local.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ENV_EXAMPLE = '''# SciLit environment configuration
# Copy this file to .env and edit values for your machine.
# Never commit .env or paste real API keys into GitHub.

# Application
APP_NAME=SciLit
APP_ENV=development
API_HOST=127.0.0.1
API_PORT=8000

# Retrieval
DEFAULT_TOP_K=10
RETRIEVAL_MODE=rrf

# Optional providers. The current Bronze pipeline uses templates, not an LLM.
OPENAI_API_KEY=
GROQ_API_KEY=

# Data directory; relative paths are resolved from the repository root.
DATA_DIR=data
'''

README = '''# SciLit

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
'''

CONFIG = '''from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load the repository-root .env before reading any application settings.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_PROJECT_ROOT / ".env", override=False)

ROOT = Path(os.environ.get("SCILIT_ROOT", _PROJECT_ROOT)).expanduser()
if not ROOT.is_absolute():
    ROOT = (_PROJECT_ROOT / ROOT).resolve()

APP_NAME = os.environ.get("APP_NAME", "SciLit").strip() or "SciLit"
APP_ENV = os.environ.get("APP_ENV", "development").strip().lower() or "development"
API_HOST = os.environ.get("API_HOST", "127.0.0.1").strip() or "127.0.0.1"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; received {raw!r}") from exc


API_PORT = _env_int("API_PORT", 8000)
if not 1 <= API_PORT <= 65535:
    raise ValueError("API_PORT must be between 1 and 65535")

DEFAULT_TOP_K = _env_int("DEFAULT_TOP_K", 10)
if not 1 <= DEFAULT_TOP_K <= 50:
    raise ValueError("DEFAULT_TOP_K must be between 1 and 50")

RETRIEVAL_MODE = os.environ.get("RETRIEVAL_MODE", "rrf").strip().lower() or "rrf"
if RETRIEVAL_MODE not in {"rrf", "bm25", "dense"}:
    raise ValueError("RETRIEVAL_MODE must be one of: rrf, bm25, dense")

# Keys are loaded for provider integrations; Bronze currently does not call an LLM.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


def _resolve_data_dir(root: Path = ROOT) -> Path:
    # Keep the existing SCILIT_DATA_DIR override for backward compatibility.
    legacy_value = os.environ.get("SCILIT_DATA_DIR")
    if legacy_value:
        legacy_path = Path(legacy_value).expanduser()
        # Preserve the legacy variable's historical relative-to-working-directory behavior.
        return legacy_path if legacy_path.is_absolute() else legacy_path.resolve()

    configured = Path(os.environ.get("DATA_DIR", "data")).expanduser()
    return configured if configured.is_absolute() else (root / configured).resolve()


DATA_DIR = _resolve_data_dir()
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
REVIEWS_DIR = PARSED_DIR / "reviews"


@dataclass(frozen=True)
class DataPaths:
    root: Path
    data: Path
    raw: Path
    parsed: Path
    reviews: Path


def paths() -> DataPaths:
    root_value = Path(os.environ.get("SCILIT_ROOT", str(ROOT))).expanduser()
    root = root_value if root_value.is_absolute() else (_PROJECT_ROOT / root_value).resolve()
    data = _resolve_data_dir(root)
    return DataPaths(
        root=root,
        data=data,
        raw=data / "raw",
        parsed=data / "parsed",
        reviews=data / "parsed" / "reviews",
    )


def setup_logging(log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.insert(0, logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )
'''


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one match for {label}, found {count}. "
            "No project files have been written; this repository version may differ."
        )
    return source.replace(old, new, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", nargs="?", default=".", help="Path to the SciLit repository root")
    args = parser.parse_args()
    root = Path(args.repo).resolve()

    paths = {
        "config": root / "scilit/src/scilit/config.py",
        "api": root / "scilit/src/scilit/api/main.py",
        "retrieval": root / "scilit/src/scilit/retrieval/pipeline.py",
        "template": root / "scilit/src/scilit/generation/template.py",
        "readme": root / "README.md",
    }
    missing = [str(path.relative_to(root)) for path in paths.values() if not path.is_file()]
    if missing:
        print("Not a recognized SciLit repository root. Missing: " + ", ".join(missing), file=sys.stderr)
        return 2

    original = {name: path.read_text(encoding="utf-8") for name, path in paths.items()}
    if original["readme"].strip() != "# scilit-":
        print("README.md differs from the expected minimal README; refusing to overwrite it automatically.", file=sys.stderr)
        return 2
    updated = dict(original)
    updated["config"] = CONFIG

    api = updated["api"]
    api = replace_once(
        api,
        '    title        = "SciLit API",\n    description  = "Bronze literature-review assistant — ESPRIT NLP 2026-2027",',
        '    title        = config.APP_NAME,\n    description  = f"{config.APP_NAME} literature-review assistant — ESPRIT NLP 2026-2027",',
        "FastAPI app metadata",
    )
    api = replace_once(api, 'k:     int  = Field(10, ge=1, le=50)', 'k:     int  = Field(config.DEFAULT_TOP_K, ge=1, le=50)', "SearchRequest default top-k")
    api = replace_once(api, 'mode:  Literal["rrf", "bm25", "dense"] = "rrf"', 'mode:  Literal["rrf", "bm25", "dense"] = config.RETRIEVAL_MODE', "SearchRequest retrieval mode")
    api = replace_once(api, 'k:     int = Field(20, ge=5, le=50, description="Top-k chunks retrieval")', 'k:     int = Field(config.DEFAULT_TOP_K, ge=5, le=50, description="Top-k chunks retrieval")', "GenerateRequest default top-k")
    if 'if __name__ == "__main__":' not in api:
        api = api.rstrip() + '''\n\n\nif __name__ == "__main__":\n    import uvicorn\n\n    uvicorn.run(\n        "scilit.api.main:app",\n        host=config.API_HOST,\n        port=config.API_PORT,\n        reload=config.APP_ENV == "development",\n    )\n'''
    updated["api"] = api

    retrieval = updated["retrieval"]
    retrieval = replace_once(retrieval, '    k:          int = 10,\n    mode:       str = "rrf",', '    k:          int = config.DEFAULT_TOP_K,\n    mode:       str = config.RETRIEVAL_MODE,', "retrieval function defaults")
    retrieval = replace_once(retrieval, 'p.add_argument("--k",     type=int, default=5, help="Top-k résultats")', 'p.add_argument("--k",     type=int, default=config.DEFAULT_TOP_K, help="Top-k résultats")', "retrieval CLI top-k")
    retrieval = replace_once(retrieval, '        default="rrf",\n        help="Mode retrieval (défaut: rrf)",', '        default=config.RETRIEVAL_MODE,\n        help="Mode retrieval lu depuis RETRIEVAL_MODE (.env)",', "retrieval CLI mode")
    updated["retrieval"] = retrieval

    template = updated["template"]
    template = replace_once(template, '    k:                      int = 20,', '    k:                      int = config.DEFAULT_TOP_K,', "generation function default top-k")
    template = replace_once(template, 'ranked = retrieve(query, k=k, mode="rrf")', 'ranked = retrieve(query, k=k, mode=config.RETRIEVAL_MODE)', "generation retrieval mode")
    template = replace_once(template, 'p.add_argument("--k",     type=int, default=20, help="Top-k chunks retrieval (défaut: 20)")', 'p.add_argument("--k",     type=int, default=config.DEFAULT_TOP_K, help="Top-k chunks retrieval from .env")', "generation CLI top-k")
    updated["template"] = template

    # Current README is only a title; replace it with concise setup documentation.
    updated["readme"] = README

    # Preflight all replacements before touching any files.
    changed = [name for name in paths if updated[name] != original[name]]
    for name in changed:
        paths[name].write_text(updated[name], encoding="utf-8")

    env_example_path = root / ".env.example"
    created_example = not env_example_path.exists()
    if created_example:
        env_example_path.write_text(ENV_EXAMPLE, encoding="utf-8")
    env_path = root / ".env"
    created_env = False
    if not env_path.exists():
        env_path.write_text(ENV_EXAMPLE, encoding="utf-8")
        created_env = True

    print("SciLit environment integration applied.")
    print("Updated: " + ", ".join(str(paths[name].relative_to(root)) for name in changed))
    print("Created .env.example" if created_example else "Kept existing .env.example unchanged")
    print("Created local .env with blank provider keys" if created_env else "Kept existing .env unchanged")
    print("No retrieval, extraction, generation, or NLI algorithms were changed.")
    print("Review git diff, then run: pytest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
