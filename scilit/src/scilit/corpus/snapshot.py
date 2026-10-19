"""Create and verify reproducible, offline corpus snapshot manifests."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from scilit.corpus import collect, parse
from scilit.schemas import (
    Article,
    LicenseEvidence,
    LicenseFamily,
    normalize_doi_identifier,
)

SCHEMA_VERSION = 1
REQUIRED_ARTICLE_PATH = "raw/articles.jsonl"
EXPECTED_ARTIFACTS = (
    "raw/manifest.json",
    "raw/collect_log.txt",
    "parsed/sections.jsonl",
    "parsed/chunks.jsonl",
    "parsed/parse_log.txt",
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class SnapshotError(ValueError):
    """A corpus or manifest cannot be safely snapshotted or verified."""


class SoftwareIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "scilit"
    version: str = "unknown"
    git_commit: str = "unknown"
    git_worktree_dirty: bool | None = None


class SnapshotFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    role: Literal["source_input", "derived", "metadata"]
    artifact_type: str
    size_bytes: int = Field(ge=0)
    sha256: str
    record_count: int | None = Field(default=None, ge=0)
    derived_from_paths: list[str] = Field(default_factory=list)
    derived_from_snapshot_id: str | None = None

    @model_validator(mode="after")
    def validate_relative_paths(self) -> SnapshotFile:
        _validate_relative_path(self.path)
        if not SHA256_PATTERN.fullmatch(self.sha256):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        for path in self.derived_from_paths:
            _validate_relative_path(path)
        return self


class ArticleStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: int = 0
    unique_ids: int = 0
    unique_dois: int = 0
    with_doi: int = 0
    missing_abstracts: int = 0
    missing_source_urls: int = 0
    without_usable_pdf_source: int = 0
    with_provenance: int = 0
    without_provenance: int = 0
    missing_license_metadata: int = 0


class LicenseStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identified_with_evidence: int = 0
    explicit_openalex_evidence: int = 0
    inferred_acl_policy_evidence: int = 0
    unknown_or_unverified: int = 0
    policy_conflicts: int = 0
    family_counts: dict[str, int] = Field(default_factory=dict)


class CorpusStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    articles: ArticleStatistics
    licenses: LicenseStatistics


class SnapshotManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    snapshot_id: str
    created_at: datetime
    software: SoftwareIdentity
    corpus_root_layout: list[str]
    collection_config: dict[str, Any]
    parsing_config: dict[str, Any]
    parsing_mode: Literal["abstract_only", "full_text", "mixed", "unknown"]
    files: list[SnapshotFile]
    statistics: CorpusStatistics
    is_complete: bool
    missing_expected_artifacts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_manifest_shape(self) -> SnapshotManifest:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        if self.created_at.utcoffset().total_seconds() != 0:
            raise ValueError("created_at must be UTC")
        paths = [item.path for item in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("files must have unique, sorted relative paths")
        if self.is_complete != (not self.missing_expected_artifacts):
            raise ValueError("is_complete must reflect missing expected artifacts")
        if self.missing_expected_artifacts != sorted(
            set(self.missing_expected_artifacts)
        ):
            raise ValueError("missing expected artifacts must be unique and sorted")
        return self


def _validate_relative_path(value: str) -> PurePosixPath:
    """Reject non-canonical and escaping manifest paths."""
    if not value or "\\" in value or "\x00" in value:
        raise ValueError("manifest paths must be non-empty POSIX relative paths")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("manifest path must stay within the corpus root")
    if ":" in path.parts[0]:
        raise ValueError("manifest path cannot contain a drive or URI prefix")
    return path


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _identity_payload(manifest: SnapshotManifest) -> dict[str, Any]:
    """Return stable content/configuration fields, excluding run details."""
    payload = manifest.model_dump(mode="json", exclude={"snapshot_id", "created_at", "warnings"})
    software = payload["software"]
    software.pop("git_worktree_dirty", None)
    return payload


def calculate_snapshot_id(manifest: SnapshotManifest) -> str:
    digest = hashlib.sha256(_canonical_json(_identity_payload(manifest))).hexdigest()
    return f"sha256:{digest}"


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _scan_files(corpus_root: Path) -> list[Path]:
    files: list[Path] = []
    for folder_name in ("raw", "parsed"):
        folder = corpus_root / folder_name
        if not folder.exists():
            continue
        if folder.is_symlink():
            raise SnapshotError(f"unsafe symlink in corpus: {folder_name}")
        if not folder.is_dir():
            raise SnapshotError(f"expected a directory: {folder_name}")
        stack = [folder]
        while stack:
            current = stack.pop()
            try:
                entries = sorted(os.scandir(current), key=lambda entry: entry.name)
            except OSError as exc:
                relative = current.relative_to(corpus_root).as_posix()
                raise SnapshotError(f"cannot read {relative}: {exc.strerror}") from exc
            for entry in entries:
                entry_path = Path(entry.path)
                relative = entry_path.relative_to(corpus_root).as_posix()
                if entry.is_symlink():
                    raise SnapshotError(f"unsafe symlink in corpus: {relative}")
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry_path)
                elif entry.is_file(follow_symlinks=False):
                    files.append(entry_path)
                else:
                    raise SnapshotError(f"unsupported filesystem entry: {relative}")
    return sorted(files, key=lambda item: item.relative_to(corpus_root).as_posix())


def _artifact_details(relative: str) -> tuple[str, str, list[str]]:
    if relative == REQUIRED_ARTICLE_PATH:
        return "source_input", "article_records", []
    if relative.startswith("raw/pdfs/"):
        return "source_input", "cached_pdf_source", []
    if relative == "raw/manifest.json":
        return "metadata", "collection_manifest", []
    if relative.startswith("raw/"):
        return "metadata", "collection_log", []
    if relative == "parsed/sections.jsonl":
        return "derived", "parsed_sections", [REQUIRED_ARTICLE_PATH]
    if relative == "parsed/chunks.jsonl":
        return "derived", "parsed_chunks", ["parsed/sections.jsonl"]
    if relative.startswith("parsed/") and relative.endswith(".log.txt"):
        return "metadata", "processing_log", []
    if relative.startswith("parsed/") and Path(relative).name in {
        "bm25_index.pkl",
        "faiss_index.bin",
    }:
        return "derived", "derived_index", ["parsed/chunks.jsonl"]
    if relative.startswith("parsed/") and Path(relative).name in {
        "bm25_meta.json",
        "faiss_meta.json",
    }:
        return "derived", "derived_index_metadata", ["parsed/chunks.jsonl"]
    if relative.startswith("parsed/"):
        return "derived", "parsed_or_review_output", ["raw/articles.jsonl"]
    return "metadata", "other", []


class _ArticleAccumulator:
    def __init__(self) -> None:
        self.ids: set[str] = set()
        self.dois: set[str] = set()
        self.records = 0
        self.with_doi = 0
        self.missing_abstracts = 0
        self.missing_source_urls = 0
        self.without_usable_pdf_source = 0
        self.with_provenance = 0
        self.missing_license_metadata = 0
        self.identified = 0
        self.explicit = 0
        self.inferred = 0
        self.unknown = 0
        self.conflicts = 0
        self.families: Counter[str] = Counter()

    @property
    def without_provenance(self) -> int:
        return self.records - self.with_provenance

    def add(self, record: dict[str, Any], line_number: int, relative: str) -> None:
        source_data = record.get("source")
        if not isinstance(source_data, dict):
            raise SnapshotError(f"{relative}:{line_number}: article source must be an object")
        normalized = dict(record)
        normalized_source = dict(source_data)
        missing_license = "license" not in normalized_source or normalized_source["license"] is None
        if missing_license:
            normalized_source["license"] = {"family": "unknown", "evidence": "unknown"}
        if normalized.get("abstract") is None:
            normalized["abstract"] = ""
        normalized["source"] = normalized_source
        try:
            article = Article.model_validate(normalized)
        except ValidationError as exc:
            first_error = exc.errors()[0]
            location = ".".join(str(part) for part in first_error["loc"])
            message = str(first_error["msg"])
            raise SnapshotError(
                f"{relative}:{line_number}: invalid Article field {location}: {message}"
            ) from exc

        self.records += 1
        self.ids.add(str(article.id))
        doi = normalize_doi_identifier(article.doi or article.source.doi)
        if doi:
            self.with_doi += 1
            self.dois.add(doi)
        if not article.abstract.strip():
            self.missing_abstracts += 1
        if not article.source.url:
            self.missing_source_urls += 1
        if not (article.source.pdf_url or article.source.arxiv_id):
            self.without_usable_pdf_source += 1
        if article.provenance:
            self.with_provenance += 1
        if missing_license:
            self.missing_license_metadata += 1

        license_info = article.source.license
        family = license_info.family.value
        self.families[family] += 1
        has_evidence = (
            license_info.family != LicenseFamily.UNKNOWN
            and license_info.evidence in {LicenseEvidence.OPENALEX, LicenseEvidence.ACL_POLICY}
            and license_info.policy_conflict is None
        )
        if has_evidence:
            self.identified += 1
        else:
            self.unknown += 1
        if has_evidence and license_info.evidence == LicenseEvidence.OPENALEX:
            self.explicit += 1
        if has_evidence and license_info.evidence == LicenseEvidence.ACL_POLICY:
            self.inferred += 1
        if license_info.policy_conflict:
            self.conflicts += 1

    def statistics(self) -> CorpusStatistics:
        articles = ArticleStatistics(
            records=self.records,
            unique_ids=len(self.ids),
            unique_dois=len(self.dois),
            with_doi=self.with_doi,
            missing_abstracts=self.missing_abstracts,
            missing_source_urls=self.missing_source_urls,
            without_usable_pdf_source=self.without_usable_pdf_source,
            with_provenance=self.with_provenance,
            without_provenance=self.without_provenance,
            missing_license_metadata=self.missing_license_metadata,
        )
        licenses = LicenseStatistics(
            identified_with_evidence=self.identified,
            explicit_openalex_evidence=self.explicit,
            inferred_acl_policy_evidence=self.inferred,
            unknown_or_unverified=self.unknown,
            policy_conflicts=self.conflicts,
            family_counts=dict(sorted(self.families.items())),
        )
        return CorpusStatistics(articles=articles, licenses=licenses)


def _parse_jsonl(path: Path, relative: str, article_stats: _ArticleAccumulator) -> int:
    count = 0
    is_articles = relative == REQUIRED_ARTICLE_PATH
    try:
        with path.open("r", encoding="utf-8", errors="strict") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise SnapshotError(
                        f"{relative}:{line_number}: malformed JSON: {exc}"
                    ) from exc
                if is_articles:
                    if not isinstance(value, dict):
                        raise SnapshotError(
                            f"{relative}:{line_number}: Article record must be a JSON object"
                        )
                    article_stats.add(value, line_number, relative)
                count += 1
    except UnicodeDecodeError as exc:
        raise SnapshotError(f"{relative}: invalid UTF-8: {exc}") from exc
    except OSError as exc:
        raise SnapshotError(f"cannot read {relative}: {exc.strerror}") from exc
    return count


def _validate_json(path: Path, relative: str) -> None:
    try:
        with path.open("r", encoding="utf-8", errors="strict") as stream:
            json.load(stream)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        line_number = getattr(exc, "lineno", "?")
        raise SnapshotError(f"{relative}:{line_number}: malformed JSON: {exc}") from exc
    except OSError as exc:
        raise SnapshotError(f"cannot read {relative}: {exc.strerror}") from exc


def _parsing_mode(corpus_root: Path) -> tuple[str, list[str]]:
    log_path = corpus_root / "parsed" / "parse_log.txt"
    if not log_path.is_file():
        return "unknown", ["Parsing mode is unknown: parsed/parse_log.txt is missing."]
    text = log_path.read_text(encoding="utf-8", errors="replace")
    modes: set[str] = set()
    for value in re.findall(r"effective_mode\s*=\s*(abstract|full_text|pdf)", text):
        modes.add("abstract_only" if value == "abstract" else "full_text")
    if not modes:
        for value in re.findall(r"Mode\s*:\s*(abstract|pdf)", text, flags=re.IGNORECASE):
            modes.add("abstract_only" if value.lower() == "abstract" else "full_text")
    if len(modes) > 1:
        return "mixed", ["Parsing log records more than one parsing mode."]
    if modes:
        return next(iter(modes)), []
    return "unknown", ["Parsing mode is unknown from the available parsing log."]


def _software_identity() -> tuple[SoftwareIdentity, list[str]]:
    warnings: list[str] = []
    try:
        version = importlib.metadata.version("scilit")
    except importlib.metadata.PackageNotFoundError:
        project_file = Path(__file__).resolve().parents[4] / "pyproject.toml"
        match = re.search(
            r"(?ms)^\[project\]\s.*?^version\s*=\s*['\"]([^'\"]+)['\"]",
            project_file.read_text(encoding="utf-8") if project_file.is_file() else "",
        )
        version = match.group(1) if match else "unknown"
    if version == "unknown":
        warnings.append("SciLit software version is unavailable.")

    repository = next(
        (parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()),
        None,
    )
    commit = "unknown"
    dirty: bool | None = None
    if repository is not None:
        safe_path = repository.as_posix()
        result = subprocess.run(
            ["git", "-c", f"safe.directory={safe_path}", "-C", str(repository), "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            commit = result.stdout.strip()
            status = subprocess.run(
                ["git", "-c", f"safe.directory={safe_path}", "-C", str(repository), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=False,
            )
            if status.returncode == 0:
                dirty = bool(status.stdout.strip())
    if commit == "unknown":
        warnings.append("Git commit identity is unavailable.")
    elif dirty:
        warnings.append("Git worktree has uncommitted changes; git_commit identifies HEAD only.")
    return SoftwareIdentity(version=version, git_commit=commit, git_worktree_dirty=dirty), warnings


def _configuration() -> tuple[dict[str, Any], dict[str, Any]]:
    collection = {
        "provider": "OpenAlex",
        "api_base": collect.OA_BASE,
        "concept_id": collect.OA_CONCEPT_NLP,
        "publication_year_min": collect.YEAR_MIN,
        "publication_year_max": collect.YEAR_MAX,
        "filter": (
            f"concepts.id:{collect.OA_CONCEPT_NLP},"
            f"publication_year:{collect.YEAR_MIN}-{collect.YEAR_MAX},"
            "open_access.is_oa:true,has_abstract:true"
        ),
        "select_fields": collect.OA_SELECT.split(","),
        "page_size": collect.OA_PER_PAGE,
        "sort": "cited_by_count:desc",
        "target_limit": "unknown: not persisted with article records",
        "retrieval_timestamp": "per-record provenance when present",
    }
    parsing = {
        "chunk_size_chars": parse.CHUNK_SIZE_CHARS,
        "chunk_overlap_chars": parse.CHUNK_OVERLAP_CHARS,
        "pdf_timeout_seconds": parse.PDF_TIMEOUT,
        "max_pdf_size_bytes": parse.MAX_PDF_SIZE,
        "section_patterns": [
            {"pattern": pattern.pattern, "flags": pattern.flags, "section": section.value}
            for pattern, section in parse.SECTION_PATTERNS
        ],
    }
    return collection, parsing


def build_snapshot(corpus_dir: Path) -> SnapshotManifest:
    """Validate and describe raw/parsed corpus files without modifying them."""
    corpus_root = corpus_dir.expanduser().resolve()
    if not corpus_root.is_dir():
        raise SnapshotError("corpus root must be an existing directory")
    files = _scan_files(corpus_root)
    relative_paths = {path.relative_to(corpus_root).as_posix() for path in files}
    if REQUIRED_ARTICLE_PATH not in relative_paths:
        raise SnapshotError(f"required input is missing: {REQUIRED_ARTICLE_PATH}")
    if (corpus_root / REQUIRED_ARTICLE_PATH).stat().st_size == 0:
        raise SnapshotError(f"required input is empty: {REQUIRED_ARTICLE_PATH}")

    article_stats = _ArticleAccumulator()
    records: list[SnapshotFile] = []
    for path in files:
        relative = path.relative_to(corpus_root).as_posix()
        role, artifact_type, derived_from = _artifact_details(relative)
        record_count: int | None = None
        if path.suffix.lower() == ".jsonl":
            record_count = _parse_jsonl(path, relative, article_stats)
        elif path.suffix.lower() == ".json":
            _validate_json(path, relative)
        checksum, size = _hash_file(path)
        records.append(
            SnapshotFile(
                path=relative,
                role=role,
                artifact_type=artifact_type,
                size_bytes=size,
                sha256=checksum,
                record_count=record_count,
                derived_from_paths=derived_from,
                # Existing index formats do not persist a corpus snapshot ID.
                derived_from_snapshot_id=None,
            )
        )

    if article_stats.records == 0:
        raise SnapshotError(f"required input has no Article records: {REQUIRED_ARTICLE_PATH}")

    missing = sorted(path for path in EXPECTED_ARTIFACTS if path not in relative_paths)
    warnings: list[str] = []
    if missing:
        warnings.append("Expected companion artifacts are missing: " + ", ".join(missing))
    mode, mode_warnings = _parsing_mode(corpus_root)
    warnings.extend(mode_warnings)
    software, software_warnings = _software_identity()
    warnings.extend(software_warnings)
    if any(item.artifact_type == "derived_index" for item in records):
        warnings.append(
            "Derived index files do not identify the source snapshot that produced them."
        )
    if article_stats.without_usable_pdf_source:
        warnings.append("Some article records have no usable PDF source recorded.")
    if article_stats.unknown:
        warnings.append("Unknown or unverified license evidence is present.")
    if article_stats.without_provenance:
        warnings.append("Some legacy article records have no provenance record.")
    collection_config, parsing_config = _configuration()
    manifest = SnapshotManifest(
        snapshot_id="pending",
        created_at=datetime.now(timezone.utc),
        software=software,
        corpus_root_layout=["raw/", "parsed/"],
        collection_config=collection_config,
        parsing_config=parsing_config,
        parsing_mode=mode,  # type: ignore[arg-type]
        files=records,
        statistics=article_stats.statistics(),
        is_complete=not missing,
        missing_expected_artifacts=missing,
        warnings=sorted(set(warnings)),
    )
    manifest.snapshot_id = calculate_snapshot_id(manifest)
    return manifest


def _resolve_manifest_path(root: Path, relative: str) -> Path:
    posix = _validate_relative_path(relative)
    candidate = root.joinpath(*posix.parts)
    current = root
    for part in posix.parts:
        current = current / part
        if current.is_symlink():
            raise SnapshotError(f"unsafe symlink path: {relative}")
    try:
        candidate.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise SnapshotError(f"manifest path escapes or is missing: {relative}") from exc
    if not candidate.is_file():
        raise SnapshotError(f"referenced file is missing or not a file: {relative}")
    return candidate


def verify_snapshot(corpus_dir: Path, manifest_path: Path) -> SnapshotManifest:
    """Verify file paths, sizes, checksums, schema, and the canonical ID."""
    root = corpus_dir.expanduser().resolve()
    if not root.is_dir():
        raise SnapshotError("corpus root must be an existing directory")
    try:
        raw = manifest_path.expanduser().read_text(encoding="utf-8")
        manifest = SnapshotManifest.model_validate_json(raw)
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise SnapshotError(f"cannot load valid snapshot manifest: {exc}") from exc
    if manifest.schema_version != SCHEMA_VERSION:
        raise SnapshotError(
            f"unsupported manifest schema version: {manifest.schema_version}"
        )
    expected_id = calculate_snapshot_id(manifest)
    if manifest.snapshot_id != expected_id:
        raise SnapshotError("snapshot identifier does not match canonical manifest content")
    for item in manifest.files:
        path = _resolve_manifest_path(root, item.path)
        checksum, size = _hash_file(path)
        if size != item.size_bytes:
            raise SnapshotError(
                f"size mismatch for {item.path}: expected {item.size_bytes}, found {size}"
            )
        if checksum != item.sha256:
            raise SnapshotError(f"SHA-256 mismatch for {item.path}")
    return manifest


def _is_under(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def write_snapshot(manifest: SnapshotManifest, corpus_dir: Path, output: Path) -> None:
    """Atomically write a manifest without replacing corpus source artifacts."""
    root = corpus_dir.expanduser().resolve()
    target = output.expanduser().absolute()
    resolved_target = target.resolve(strict=False)
    if target.is_symlink():
        raise SnapshotError("refusing to write through a symlink output path")
    for folder in (root / "raw", root / "parsed"):
        if _is_under(resolved_target, folder.resolve(strict=False)):
            raise SnapshotError("manifest output must be outside raw/ and parsed/ inputs")
    if resolved_target == root:
        raise SnapshotError("manifest output must be a file outside the corpus input folders")

    if target.exists():
        try:
            existing = SnapshotManifest.model_validate_json(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValidationError):
            raise SnapshotError("refusing to overwrite an existing non-manifest file") from None
        if existing.schema_version != SCHEMA_VERSION:
            raise SnapshotError("refusing to overwrite a manifest with an unsupported schema")

    target.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json")
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=target.parent,
            prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except OSError as exc:
        raise SnapshotError(f"cannot atomically write manifest: {exc.strerror}") from exc
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _summary(manifest: SnapshotManifest) -> str:
    articles = manifest.statistics.articles
    return (
        f"snapshot={manifest.snapshot_id} articles={articles.records} "
        f"unique_ids={articles.unique_ids} unique_dois={articles.unique_dois} "
        f"mode={manifest.parsing_mode} complete={str(manifest.is_complete).lower()} "
        f"warnings={len(manifest.warnings)}"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or verify a SciLit corpus snapshot.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="validate corpus files and write a manifest")
    create.add_argument("--corpus-dir", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="verify a manifest against local files")
    verify.add_argument("--corpus-dir", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        if args.command == "create":
            manifest = build_snapshot(args.corpus_dir)
            write_snapshot(manifest, args.corpus_dir, args.output)
            print(f"Created {args.output.name}: {_summary(manifest)}")
            for warning in manifest.warnings:
                print(f"warning: {warning}")
        else:
            manifest = verify_snapshot(args.corpus_dir, args.manifest)
            print(f"Verified {args.manifest.name}: {_summary(manifest)}")
        return 0
    except SnapshotError as exc:
        print(f"snapshot error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
