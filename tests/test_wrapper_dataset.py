from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from orato_asr.exceptions import AudioValidationError, ManifestError, TrainingError
from orato_asr.training import data as training_data
from orato_asr.training.data import LazyWrapperTrainingDataset, prepare_training_manifest


def _write_manifest(path: Path, records: list[dict[str, object]], *, blank: bool = False) -> Path:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    if blank:
        lines.insert(1, "")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _decoder(
    durations: dict[str, float], calls: list[Path], *, corrupt: set[str] | None = None
):
    corrupt = corrupt or set()

    def decode(path: Path) -> SimpleNamespace:
        calls.append(path)
        if path.name in corrupt:
            raise AudioValidationError(f"corrupt test clip: {path.name}")
        return SimpleNamespace(
            duration_seconds=durations[path.name],
            samples=f"decoded:{path.name}:{len(calls)}",
        )

    return decode


def test_manifest_preflight_filters_duration_and_reports_truthful_totals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _write_manifest(
        tmp_path / "train.jsonl",
        [
            {"audio_filepath": "audio/short.wav", "text": "हाँ", "language": "Hindi"},
            {"audio_filepath": "audio/first.wav", "text": "जी", "language": "Hindi"},
            {"audio_filepath": "audio/capped.wav", "text": "okay", "language": "English"},
            {"audio_filepath": "audio/long.wav", "text": "नहीं", "language": "Hindi"},
        ],
    )
    calls: list[Path] = []
    monkeypatch.setattr(
        training_data,
        "decode_audio",
        _decoder(
            {"short.wav": 0.5, "first.wav": 2.0, "capped.wav": 3.0, "long.wav": 7.0},
            calls,
        ),
    )

    prepared = prepare_training_manifest(
        manifest,
        project_root=tmp_path,
        minimum_duration_seconds=1.0,
        maximum_duration_seconds=6.0,
        max_samples=1,
        max_hours=2.0,
    )
    summary = prepared.as_dict()

    assert len(calls) == 4
    assert prepared.total_samples == 4
    assert prepared.total_duration_seconds == pytest.approx(12.5)
    assert prepared.eligible_samples == 2
    assert prepared.eligible_duration_seconds == pytest.approx(5.0)
    assert prepared.selected_duration_seconds == pytest.approx(2.0)
    assert prepared.capped_samples == 1
    assert prepared.capped_duration_seconds == pytest.approx(3.0)
    assert [(item.duration_seconds, item.reason) for item in prepared.excluded] == [
        (0.5, "below_minimum_duration"),
        (7.0, "above_maximum_duration"),
    ]
    assert prepared.total_duration_seconds == pytest.approx(
        prepared.eligible_duration_seconds
        + sum(item.duration_seconds for item in prepared.excluded)
    )
    assert prepared.eligible_duration_seconds == pytest.approx(
        prepared.selected_duration_seconds + prepared.capped_duration_seconds
    )
    assert summary["total_audio_hours"] == pytest.approx(12.5 / 3600)
    assert summary["eligible_audio_hours"] == pytest.approx(5.0 / 3600)
    assert summary["selected_audio_hours"] == pytest.approx(2.0 / 3600)


def test_lazy_dataset_recovers_utf8_record_by_byte_offset_and_decodes_on_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _write_manifest(
        tmp_path / "train.jsonl",
        [
            {"audio_filepath": "audio/one.wav", "text": "मुझे appointment चाहिए", "language": "hi"},
            {
                "audio_filepath": "audio/two.flac",
                "text": "okay",
                "language": "English",
                "source": "owner",
                "speaker_id": "speaker-2",
                "recording_id": "recording-2",
                "domain": "appointments",
                "split": "train",
                "metadata": {"sample_id": "source-sample-2"},
            },
        ],
        blank=True,
    )
    calls: list[Path] = []
    monkeypatch.setattr(
        training_data,
        "decode_audio",
        _decoder({"one.wav": 1.5, "two.flac": 2.5}, calls),
    )
    prepared = prepare_training_manifest(
        manifest,
        project_root=tmp_path,
        minimum_duration_seconds=None,
        maximum_duration_seconds=6.0,
        max_samples=None,
        max_hours=1.0,
    )

    assert [item.line_number for item in prepared.selected] == [1, 3]
    assert all(not hasattr(item, "transcript") for item in prepared.selected)
    assert len(calls) == 2
    dataset = LazyWrapperTrainingDataset(prepared, project_root=tmp_path)
    assert len(dataset) == 2
    assert len(calls) == 2

    second = dataset[1]

    assert len(calls) == 3
    assert calls[-1] == (tmp_path / "audio/two.flac").resolve()
    assert second.sample_id == prepared.selected[1].sample_id
    assert second.line_number == 3
    assert second.transcript == "okay"
    assert second.language == "English"
    assert second.duration_seconds == 2.5
    assert second.audio == "decoded:two.flac:3"
    assert second.source == "owner"
    assert second.speaker_id == "speaker-2"
    assert second.recording_id == "recording-2"
    assert second.domain == "appointments"
    assert second.split == "train"
    assert second.metadata == {"sample_id": "source-sample-2"}


def test_lazy_dataset_rejects_manifest_record_changed_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "train.jsonl"
    original = {
        "audio_filepath": "audio/one.wav",
        "text": "yes",
        "language": "English",
    }
    manifest = _write_manifest(path, [original])
    calls: list[Path] = []
    monkeypatch.setattr(
        training_data,
        "decode_audio",
        _decoder({"one.wav": 1.0}, calls),
    )
    prepared = prepare_training_manifest(
        manifest,
        project_root=tmp_path,
        minimum_duration_seconds=None,
        maximum_duration_seconds=6.0,
        max_samples=None,
        max_hours=1.0,
    )
    changed = dict(original)
    changed["text"] = "no!"
    _write_manifest(path, [changed])

    with pytest.raises(TrainingError, match="manifest record changed after preflight"):
        LazyWrapperTrainingDataset(prepared, project_root=tmp_path)[0]


def test_remote_locator_is_rejected_without_decode_or_secret_leakage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _write_manifest(
        tmp_path / "remote.jsonl",
        [
            {
                "audio_filepath": "https://private.example/container/audio.wav?sig=secret",
                "text": "नमस्ते",
                "language": "Hindi",
            }
        ],
    )
    calls: list[Path] = []
    monkeypatch.setattr(
        training_data,
        "decode_audio",
        _decoder({}, calls),
    )

    with pytest.raises(TrainingError, match="requires local audio") as raised:
        prepare_training_manifest(
            manifest,
            project_root=tmp_path,
            minimum_duration_seconds=None,
            maximum_duration_seconds=6.0,
            max_samples=None,
            max_hours=1.0,
        )

    assert calls == []
    assert "sig=secret" not in str(raised.value)


def test_training_mode_rejects_explicit_evaluation_split_before_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _write_manifest(
        tmp_path / "evaluation.jsonl",
        [
            {
                "audio_filepath": "audio/eval.wav",
                "text": "यह evaluation है",
                "language": "Hindi",
                "split": "evaluation",
            }
        ],
    )
    calls: list[Path] = []
    monkeypatch.setattr(
        training_data,
        "decode_audio",
        _decoder({"eval.wav": 1.0}, calls),
    )

    with pytest.raises(TrainingError, match="non-training split 'evaluation'"):
        prepare_training_manifest(
            manifest,
            project_root=tmp_path,
            minimum_duration_seconds=None,
            maximum_duration_seconds=6.0,
            max_samples=None,
            max_hours=1.0,
            require_training_split=True,
        )

    assert calls == []


def test_corrupt_audio_fails_preflight_with_manifest_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _write_manifest(
        tmp_path / "corrupt.jsonl",
        [
            {"audio_filepath": "audio/good.wav", "text": "हाँ"},
            {"audio_filepath": "audio/bad.wav", "text": "नहीं"},
        ],
    )
    calls: list[Path] = []
    monkeypatch.setattr(
        training_data,
        "decode_audio",
        _decoder({"good.wav": 1.0, "bad.wav": 1.0}, calls, corrupt={"bad.wav"}),
    )

    with pytest.raises(TrainingError, match=r"corrupt\.jsonl:2: invalid training audio"):
        prepare_training_manifest(
            manifest,
            project_root=tmp_path,
            minimum_duration_seconds=None,
            maximum_duration_seconds=6.0,
            max_samples=None,
            max_hours=1.0,
        )
    assert [path.name for path in calls] == ["good.wav", "bad.wav"]


def test_declared_duration_must_match_decoded_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _write_manifest(
        tmp_path / "duration.jsonl",
        [
            {
                "audio_filepath": "audio/a.wav",
                "text": "हाँ",
                "duration": 2.0,
            }
        ],
    )
    monkeypatch.setattr(
        training_data,
        "decode_audio",
        lambda _path: SimpleNamespace(duration_seconds=2.26, samples=object()),
    )

    with pytest.raises(TrainingError, match="differs from decoded audio"):
        prepare_training_manifest(
            manifest,
            project_root=tmp_path,
            minimum_duration_seconds=None,
            maximum_duration_seconds=6.0,
            max_samples=None,
            max_hours=1.0,
        )


def test_malformed_jsonl_is_line_numbered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text('{"audio_filepath":"audio/a.wav","text":"हाँ"}\n{bad\n')
    monkeypatch.setattr(
        training_data,
        "decode_audio",
        lambda _path: SimpleNamespace(duration_seconds=1.0, samples=object()),
    )

    with pytest.raises(ManifestError, match=r"bad\.jsonl:2: invalid UTF-8 JSONL record"):
        prepare_training_manifest(
            manifest,
            project_root=tmp_path,
            minimum_duration_seconds=None,
            maximum_duration_seconds=6.0,
            max_samples=None,
            max_hours=1.0,
        )


@pytest.mark.parametrize("duration", [0.0, float("nan"), float("inf")])
def test_decoder_duration_must_remain_positive_and_finite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    duration: float,
) -> None:
    manifest = _write_manifest(
        tmp_path / "train.jsonl",
        [{"audio_filepath": "audio/a.wav", "text": "हाँ"}],
    )
    monkeypatch.setattr(
        training_data,
        "decode_audio",
        lambda _path: SimpleNamespace(duration_seconds=duration, samples=object()),
    )

    with pytest.raises(TrainingError, match="decoded duration must be positive and finite"):
        prepare_training_manifest(
            manifest,
            project_root=tmp_path,
            minimum_duration_seconds=None,
            maximum_duration_seconds=6.0,
            max_samples=None,
            max_hours=1.0,
        )
