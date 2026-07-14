from __future__ import annotations

import json
from pathlib import Path

import pytest

from orato_asr.audio import DecodedAudio
from orato_asr.data.manifest import iter_manifest, iter_manifest_events, write_manifest
from orato_asr.data.overlap import check_overlap
from orato_asr.data.schema import ManifestRecord, parse_record
from orato_asr.data.selection import SelectionOptions, select_manifest
from orato_asr.data.summary import summarize_manifest
from orato_asr.data.validation import validate_manifest
from orato_asr.exceptions import ManifestError, ManifestValidationError


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return path


def _decoded(path: str | Path, *, duration: float = 1.0) -> DecodedAudio:
    return DecodedAudio(
        path=Path(path),
        samples=None,
        original_sample_rate=8_000,
        sample_rate=16_000,
        original_channels=2,
        channels=1,
        duration_seconds=duration,
        downmixed=True,
        resampled=True,
    )


def test_manifest_is_strict_streaming_and_writer_is_atomic(tmp_path: Path) -> None:
    source = _write_manifest(
        tmp_path / "source.jsonl",
        [
            {"audio_filepath": "audio/a.wav", "text": "नमस्ते hello", "metadata": {"dataset": "x"}},
            {"audio_filepath": "audio/b.flac", "text": "okay", "duration": 1.5},
        ],
    )
    original = source.read_bytes()

    records = list(iter_manifest(source))
    assert records[0].metadata["dataset"] == "x"
    destination = write_manifest(iter(records), tmp_path / "derived.jsonl")

    assert source.read_bytes() == original
    assert list(iter_manifest(destination))[1].duration == 1.5
    with pytest.raises(ManifestError, match="local paths"):
        write_manifest(iter(records), "https://example.test/derived.jsonl")

    malformed = tmp_path / "bad.jsonl"
    malformed.write_text("\n{\"audio_filepath\": \"x.wav\", \"text\": \"x\", \"extra\": 1}\n", encoding="utf-8")
    events = list(iter_manifest_events(malformed))
    assert len(events) == 1
    assert events[0].line_number == 2
    assert isinstance(events[0].error, ManifestValidationError)


def test_record_allows_only_canonical_fields_and_nested_metadata() -> None:
    with pytest.raises(ManifestValidationError, match="unsupported top-level"):
        parse_record({"audio_filepath": "a.wav", "text": "x", "speaker": "no"})
    with pytest.raises(ManifestValidationError, match="positive finite"):
        parse_record({"audio_filepath": "a.wav", "text": "x", "duration": True})

    record = parse_record(
        {"audio_filepath": "a.wav", "text": "x", "metadata": {"nested": {"allowed": True}}}
    )
    assert record.as_dict()["metadata"] == {"nested": {"allowed": True}}


def test_validation_reports_remote_script_duplicate_and_decoder_findings(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.jsonl",
        [
            {"audio_filepath": "audio/a.wav", "text": "hello", "language": "hi", "duration": 2.0},
            {"audio_filepath": "audio/a.wav", "text": "hello", "duration": 2.0},
            {"audio_filepath": "azureml://datastores/private/paths/a.flac?sig=secret", "text": "नमस्ते"},
        ],
    )
    report = validate_manifest(
        manifest,
        project_root=tmp_path,
        check_audio=True,
        audio_decoder=lambda path: _decoded(path, duration=1.0),
    )
    codes = {finding.code for finding in report.findings}
    assert {"possibly_romanized_hindi", "duplicate_audio_path", "duplicate_transcript", "duration_mismatch", "remote_audio_not_locally_verified"} <= codes
    assert report.remote_records == 1
    assert all("sig=secret" not in (finding.audio_filepath or "") for finding in report.findings)


def test_summary_selection_and_overlap_are_deterministic(tmp_path: Path) -> None:
    rows = [
        {"audio_filepath": "audio/a.wav", "text": "नमस्ते A", "duration": 1.0, "recording_id": "r1", "speaker_id": "s1", "source": "x"},
        {"audio_filepath": "audio/b.wav", "text": "hello B", "duration": 2.0, "recording_id": "r2", "speaker_id": "s2", "source": "x"},
        {"audio_filepath": "audio/c.wav", "text": "hello B", "duration": 3.0, "recording_id": "r3", "speaker_id": "s3", "source": "y"},
    ]
    manifest = _write_manifest(tmp_path / "source.jsonl", rows)
    summary = summarize_manifest(manifest, project_root=tmp_path)
    assert summary.as_dict()["total_duration_seconds"] == 6.0
    assert summary.as_dict()["script_distribution"]["mixed_devanagari_latin"] == 1

    first = select_manifest(
        manifest,
        tmp_path / "first.jsonl",
        options=SelectionOptions(max_samples=2, seed=7, shuffled=True),
    )
    second = select_manifest(
        manifest,
        tmp_path / "second.jsonl",
        options=SelectionOptions(max_samples=2, seed=7, shuffled=True),
    )
    assert [row.audio_filepath for row in iter_manifest(first["output_manifest"])] == [
        row.audio_filepath for row in iter_manifest(second["output_manifest"])
    ]
    with pytest.raises(ManifestError, match="declared duration"):
        no_duration = _write_manifest(tmp_path / "no_duration.jsonl", [{"audio_filepath": "a.wav", "text": "x"}])
        select_manifest(no_duration, tmp_path / "bad-select.jsonl", options=SelectionOptions(max_duration_seconds=1.0))

    evaluation = _write_manifest(
        tmp_path / "evaluation.jsonl",
        [
            {"audio_filepath": "audio/a.wav", "text": "different", "recording_id": "other", "speaker_id": "s1"},
            {"audio_filepath": "audio/z.wav", "text": "hello B", "recording_id": "r2", "speaker_id": "z"},
        ],
    )
    overlap = check_overlap(manifest, evaluation, project_root=tmp_path, disallow_speaker_overlap=True)
    assert overlap.counts["audio_path"] == 1
    assert overlap.counts["recording_id"] == 1
    assert overlap.counts["speaker_id"] == 1
    assert overlap.counts["transcript"] == 2
    assert overlap.prohibited_count == 3
    assert all(len(example.value) == 16 for example in overlap.examples)
