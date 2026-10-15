from __future__ import annotations

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
