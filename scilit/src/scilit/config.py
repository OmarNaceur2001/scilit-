from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(os.environ.get("SCILIT_ROOT", Path(__file__).resolve().parents[3]))
DATA_DIR = Path(os.environ.get("SCILIT_DATA_DIR", ROOT / "data"))
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
    root = Path(os.environ.get("SCILIT_ROOT", ROOT))
    data = Path(os.environ.get("SCILIT_DATA_DIR", DATA_DIR))
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
