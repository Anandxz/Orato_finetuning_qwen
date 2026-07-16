from __future__ import annotations

import json
from pathlib import Path

import pytest

from orato_asr.data.build_splits import (
    build_splits,
    load_split_config,
    validate_split_directory,
)
from orato_asr.data.manifest import iter_manifest
from orato_asr.exceptions import ManifestError, ManifestValidationError


def _create_processed(root: Path, *, rows_per_dataset: int = 18) -> Path:
    processed = root / "processed"
    for dataset_index, dataset in enumerate(("calls", "read", "mixed")):
        directory = processed / dataset
        (directory / "audio").mkdir(parents=True)
        rows = []
        for index in range(rows_per_dataset):
            audio = directory / "audio" / f"{index}.flac"
            audio.write_bytes(b"not-copied-audio")
            rows.append(
                {
                    "audio_filepath": f"audio/{index}.flac",
                    "text": f"मुझे appointment {index} के लिए चाहिए",
                    "duration": float(index % 6 + 1),
                    "language": "hinglish" if index % 2 else "hindi",
                    "domain": "call" if dataset == "calls" else "read_speech",
                    "speaker_id": f"{dataset}-speaker-{index // 2}",
                    "session_id": f"{dataset}-session-{index // 3}",
                    "tags": ["call_like"] if dataset == "calls" else ["clean"],
                }
            )
        (directory / "manifest.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
    return processed


def _config(
    root: Path,
    processed: Path,
    *,
    version: str = "v1",
    seed: int = 42,
    require_audio: bool = True,
) -> Path:
    path = root / f"split-{version}-{seed}.yaml"
    path.write_text(
        f"""name: split_all
version: {version}
seed: {seed}
storage:
  data_root: {processed}
  split_root: {root / 'splits'}
ratios:
  train: 0.8
  validation: 0.1
  test: 0.1
datasets:
  include: [\"*\"]
  exclude: []
grouping:
  priority: [session_id, source_id, speaker_id, audio_filepath]
stratification:
  fields: [dataset, language, domain, contains_number, duration_bucket, tags]
validation:
  require_audio_exists: {str(require_audio).lower()}
  fail_on_duplicate_audio_path: true
""",
        encoding="utf-8",
    )
    return path


def test_discovery_determinism_group_safety_fingerprints_and_no_copy(tmp_path: Path) -> None:
    processed = _create_processed(tmp_path)
    source_audio = sorted(processed.rglob("*.flac"))
    source_hashes = {path: path.read_bytes() for path in source_audio}
    config = load_split_config(
        _config(tmp_path, processed), project_root=tmp_path
    )

    first = build_splits(config)
    first_bytes = {
        split: (config.output_directory / f"{split}.jsonl").read_bytes()
        for split in ("train", "validation", "test")
    }
    second = build_splits(config, overwrite=True)

    assert first["split_fingerprint"] == second["split_fingerprint"]
    assert first_bytes == {
        split: (config.output_directory / f"{split}.jsonl").read_bytes()
        for split in first_bytes
    }
    assert first["group_leakage_count"] == 0
    assert sum(values["records"] for values in first["splits"].values()) == 54
    assert all(
        abs(values["distribution_deviation"]["duration_ratio"]) < 0.12
        for values in first["splits"].values()
    )
    assert not list(config.output_directory.rglob("*.flac"))
    assert {path: path.read_bytes() for path in source_audio} == source_hashes

    all_records = [
        record
        for split in ("train", "validation", "test")
        for record in iter_manifest(config.output_directory / f"{split}.jsonl")
    ]
    assert all(not Path(record.audio_filepath).is_absolute() for record in all_records)
    assert all(record.metadata["dataset"] in {"calls", "read", "mixed"} for record in all_records)
    validation = validate_split_directory(
        config.output_directory,
        project_root=tmp_path,
        data_root=str(processed),
        check_audio=True,
    )
    assert validation["status"] == "success"


def test_seed_changes_assignment_and_version_requires_overwrite(tmp_path: Path) -> None:
    processed = _create_processed(tmp_path)
    first_config = load_split_config(_config(tmp_path, processed, version="v1", seed=1), project_root=tmp_path)
    first = build_splits(first_config)
    with pytest.raises(ManifestError, match="overwrite"):
        build_splits(first_config)

    second_config = load_split_config(_config(tmp_path, processed, version="v2", seed=2), project_root=tmp_path)
    second = build_splits(second_config)
    assert first["split_fingerprint"] != second["split_fingerprint"]
    assert first["output_manifest_checksums"] != second["output_manifest_checksums"]


def test_missing_invalid_and_duplicate_records_fail_without_output(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    directory = processed / "broken"
    directory.mkdir(parents=True)
    (directory / "manifest.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"audio_filepath": "audio/a.flac", "text": "", "duration": 1.0},
                {"audio_filepath": "audio/b.mp3", "text": "x", "duration": -1.0},
                {"audio_filepath": "audio/c.flac", "text": "x", "duration": 1.0},
                {"audio_filepath": "audio/c.flac", "text": "y", "duration": 1.0},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    config = load_split_config(
        _config(tmp_path, processed, require_audio=False), project_root=tmp_path
    )
    with pytest.raises(ManifestValidationError, match="error"):
        build_splits(config)
    assert not config.output_directory.exists()


def test_legacy_absolute_and_blob_paths_normalize_to_dataset_suffix(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    directory = processed / "legacy"
    directory.mkdir(parents=True)
    rows = []
    for index in range(12):
        locator = (
            f"/old/machine/processed/legacy/audio/{index}.flac"
            if index % 2 == 0
            else f"https://account.blob.core.windows.net/data/processed/legacy/audio/{index}.flac"
        )
        rows.append(
            {
                "audio_filepath": locator,
                "text": f"sample {index}",
                "duration": 1.0,
                "speaker_id": f"speaker-{index}",
            }
        )
    (directory / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    config = load_split_config(
        _config(tmp_path, processed, require_audio=False), project_root=tmp_path
    )

    report = build_splits(config)
    records = [
        record
        for split in ("train", "validation", "test")
        for record in iter_manifest(config.output_directory / f"{split}.jsonl")
    ]
    assert report["total_records"] == 12
    assert all(record.audio_filepath.startswith("legacy/audio/") for record in records)
