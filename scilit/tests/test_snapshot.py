from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest
from pydantic import ValidationError
from scilit.corpus.snapshot import (
    SnapshotError,
    SnapshotFile,
    SnapshotManifest,
    build_snapshot,
    calculate_snapshot_id,
    main,
    verify_snapshot,
    write_snapshot,
)
from scilit.schemas import Article


def _article(
    index: int,
    *,
    license_info: dict | str | None = "cc-by-4.0",
    include_license: bool = True,
    include_abstract: bool = True,
    provenance: list[dict] | None = None,
) -> dict:
    source = {
        "venue": "other",
        "url": f"https://example.org/paper/{index}",
    }
    if include_license:
        source["license"] = license_info
    record = {
        "id": str(uuid5(NAMESPACE_URL, f"https://openalex.org/W{index}")),
        "title": f"Study {index}",
        "authors": ["Ada Lovelace"],
        "year": 2022,
        "source": source,
        "doi": f"10.1234/example.{index}",
    }
    if include_abstract:
        record["abstract"] = "A reproducible study."
    if provenance is not None:
        record["provenance"] = provenance
    return record


def _provenance() -> list[dict]:
    return [
        {
            "provider": "openalex",
            "provider_record_id": "W1",
            "retrieved_at": "2026-01-02T03:04:05Z",
        }
    ]


def _corpus(
    root: Path,
    records: list[dict] | None = None,
    *,
    reverse_creation: bool = False,
    with_index: bool = False,
) -> Path:
    files = {
        "raw/articles.jsonl": "".join(
            json.dumps(item, sort_keys=True) + "\n"
            for item in (records or [_article(1, provenance=_provenance())])
        ),
        "raw/manifest.json": '{"ids": [], "total": 0}\n',
        "raw/collect_log.txt": "collection complete\n",
        "parsed/sections.jsonl": '{"section_type": "abstract"}\n',
        "parsed/chunks.jsonl": '{"text": "abstract"}\n',
        "parsed/parse_log.txt": "Mode : abstract | Articles : 1\n",
    }
    if with_index:
        files["parsed/bm25_index.pkl"] = "index bytes"
        files["parsed/bm25_meta.json"] = '{"documents": 1}\n'
    items = list(files.items())
    if reverse_creation:
        items.reverse()
    for relative, contents in items:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    return root


def _manifest_file(manifest: SnapshotManifest, relative: str) -> SnapshotFile:
    return next(item for item in manifest.files if item.path == relative)


def test_snapshot_id_is_stable_across_different_directories(tmp_path: Path) -> None:
    first = build_snapshot(_corpus(tmp_path / "one"))
    second = build_snapshot(_corpus(tmp_path / "another"))
    assert first.snapshot_id == second.snapshot_id


def test_content_change_changes_checksum_and_snapshot_id(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "corpus")
    before = build_snapshot(root)
    article_path = root / "raw/articles.jsonl"
    article_path.write_text(
        json.dumps(_article(1, provenance=_provenance()) | {"abstract": "changed"}) + "\n",
        encoding="utf-8",
    )
    after = build_snapshot(root)
    assert _manifest_file(before, "raw/articles.jsonl").sha256 != _manifest_file(
        after, "raw/articles.jsonl"
    ).sha256
    assert before.snapshot_id != after.snapshot_id


def test_creation_timestamp_is_separate_from_snapshot_id(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "corpus")
    first = build_snapshot(root)
    second = build_snapshot(root)
    assert first.created_at != second.created_at
    assert first.snapshot_id == second.snapshot_id
    assert calculate_snapshot_id(first) == calculate_snapshot_id(second)


def test_file_creation_order_does_not_change_manifest_content(tmp_path: Path) -> None:
    first = build_snapshot(_corpus(tmp_path / "first"))
    second = build_snapshot(_corpus(tmp_path / "second", reverse_creation=True))
    assert first.snapshot_id == second.snapshot_id
    assert [item.path for item in first.files] == [item.path for item in second.files]
    first_data = first.model_dump(mode="json", exclude={"created_at"})
    second_data = second.model_dump(mode="json", exclude={"created_at"})
    assert first_data == second_data


def test_jsonl_article_counts_and_legacy_records_load(tmp_path: Path) -> None:
    records = [
        _article(1, license_info="open access"),
        _article(2, include_license=False, include_abstract=False),
    ]
    manifest = build_snapshot(_corpus(tmp_path / "corpus", records))
    stats = manifest.statistics.articles
    assert stats.records == 2
    assert stats.unique_ids == 2
    assert stats.unique_dois == 2
    assert stats.without_provenance == 2
    assert stats.missing_abstracts == 1
    assert stats.missing_license_metadata == 1
    legacy = Article.model_validate(
        _article(3, license_info="open access", provenance=None)
    )
    assert legacy.provenance == []
    assert legacy.source.license.is_verified is False


def test_missing_provenance_emits_warning(tmp_path: Path) -> None:
    manifest = build_snapshot(
        _corpus(tmp_path / "corpus", [_article(1, provenance=None)])
    )

    assert manifest.statistics.articles.without_provenance == 1
    assert "Some legacy article records have no provenance record." in manifest.warnings


def test_license_and_provenance_statistics_preserve_unknown_states(
    tmp_path: Path,
) -> None:
    records = [
        _article(1, license_info={"raw_string": "cc-by-4.0"}, provenance=_provenance()),
        _article(
            2,
            license_info={
                "spdx_id": "CC-BY-NC-SA-3.0",
                "evidence": "acl_policy",
                "inference_source": "https://aclanthology.org/faq/copyright/",
            },
        ),
        _article(3, license_info="open access"),
        _article(4, include_license=False),
    ]
    manifest = build_snapshot(_corpus(tmp_path / "corpus", records))
    stats = manifest.statistics
    assert stats.articles.with_provenance == 1
    assert stats.articles.without_provenance == 3
    assert stats.articles.missing_license_metadata == 1
    assert stats.licenses.identified_with_evidence == 2
    assert stats.licenses.explicit_openalex_evidence == 1
    assert stats.licenses.inferred_acl_policy_evidence == 1
    assert stats.licenses.unknown_or_unverified == 2
    assert stats.licenses.family_counts == {"cc-by": 1, "cc-by-nc-sa": 1, "unknown": 2}


def test_malformed_jsonl_reports_path_and_line(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "corpus")
    valid_article = json.dumps(_article(1, provenance=_provenance()))
    (root / "raw/articles.jsonl").write_text(
        valid_article + "\n{bad json}\n",
        encoding="utf-8",
    )
    with pytest.raises(SnapshotError, match=r"raw/articles\.jsonl:2: malformed JSON"):
        build_snapshot(root)


def test_derived_indexes_are_not_source_inputs(tmp_path: Path) -> None:
    manifest = build_snapshot(_corpus(tmp_path / "corpus", with_index=True))
    index = _manifest_file(manifest, "parsed/bm25_index.pkl")
    assert index.role == "derived"
    assert index.artifact_type == "derived_index"
    assert index.derived_from_paths == ["parsed/chunks.jsonl"]
    assert index.derived_from_snapshot_id is None
    assert any("do not identify the source snapshot" in item for item in manifest.warnings)


def test_manifest_paths_reject_traversal_and_absolute_paths() -> None:
    for unsafe in ("../outside.jsonl", "C:/outside.jsonl", "/outside.jsonl", "raw\\file"):
        with pytest.raises(ValidationError):
            SnapshotFile(
                path=unsafe,
                role="source_input",
                artifact_type="article_records",
                size_bytes=0,
                sha256="0" * 64,
            )


def test_snapshot_rejects_symlink_in_corpus(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "corpus")
    external = tmp_path / "outside.txt"
    external.write_text("outside", encoding="utf-8")
    link = root / "raw" / "escape.txt"
    try:
        link.symlink_to(external)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available")
    with pytest.raises(SnapshotError, match="unsafe symlink"):
        build_snapshot(root)


def test_verify_detects_missing_and_modified_files(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "corpus")
    manifest = build_snapshot(root)
    output = tmp_path / "snapshot.json"
    write_snapshot(manifest, root, output)
    (root / "raw/articles.jsonl").unlink()
    with pytest.raises(SnapshotError, match="missing"):
        verify_snapshot(root, output)

    _corpus(root)
    (root / "raw/articles.jsonl").write_text("modified\n", encoding="utf-8")
    with pytest.raises(SnapshotError, match="size mismatch|SHA-256 mismatch"):
        verify_snapshot(root, output)


def test_verify_detects_bad_checksum_and_unsupported_schema(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "corpus")
    manifest = build_snapshot(root)
    output = tmp_path / "snapshot.json"
    data = manifest.model_dump(mode="json")
    data["files"][0]["sha256"] = "0" * 64
    altered = SnapshotManifest.model_validate(data)
    altered.snapshot_id = calculate_snapshot_id(altered)
    output.write_text(altered.model_dump_json(), encoding="utf-8")
    with pytest.raises(SnapshotError, match="SHA-256 mismatch"):
        verify_snapshot(root, output)

    data = manifest.model_dump(mode="json")
    data["schema_version"] = 999
    data["snapshot_id"] = "unsupported"
    output.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SnapshotError, match="unsupported manifest schema"):
        verify_snapshot(root, output)


def test_verify_detects_invalid_canonical_snapshot_id(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "corpus")
    data = build_snapshot(root).model_dump(mode="json")
    data["snapshot_id"] = "sha256:" + "0" * 64
    output = tmp_path / "snapshot.json"
    output.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SnapshotError, match="identifier does not match"):
        verify_snapshot(root, output)


def test_cli_create_verify_invalid_input_and_output_safety(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _corpus(tmp_path / "corpus")
    output = tmp_path / "snapshot.json"
    assert main(["create", "--corpus-dir", str(root), "--output", str(output)]) == 0
    assert "Created snapshot.json" in capsys.readouterr().out
    assert main(["verify", "--corpus-dir", str(root), "--manifest", str(output)]) == 0
    assert "Verified snapshot.json" in capsys.readouterr().out
    assert main(
        [
            "create",
            "--corpus-dir",
            str(tmp_path / "missing"),
            "--output",
            str(tmp_path / "invalid.json"),
        ]
    ) == 2
    assert not (tmp_path / "invalid.json").exists()
    assert main(
        [
            "create",
            "--corpus-dir",
            str(root),
            "--output",
            str(root / "raw" / "articles.jsonl"),
        ]
    ) == 2
    assert main(["verify", "--corpus-dir", str(root), "--manifest", str(tmp_path / "absent.json")]) == 2


def test_empty_or_missing_required_article_input_fails(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    (root / "raw").mkdir(parents=True)
    with pytest.raises(SnapshotError, match="required input is missing"):
        build_snapshot(root)
    (root / "raw" / "articles.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(SnapshotError, match="required input is empty"):
        build_snapshot(root)


def test_snapshot_creation_timestamp_is_aware_utc(tmp_path: Path) -> None:
    manifest = build_snapshot(_corpus(tmp_path / "corpus"))
    assert manifest.created_at.tzinfo is not None
    assert manifest.created_at.utcoffset() == datetime.now(timezone.utc).utcoffset()
