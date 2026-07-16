from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from orato_asr.exceptions import TrainingError
from orato_asr.training.official_h100 import (
    _latest_checkpoint,
    load_official_h100_config,
    prepare_official_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_h100_config_is_bounded_and_official() -> None:
    config = load_official_h100_config(
        PROJECT_ROOT / "configs" / "train_wrapper_official_sft_h100_20hr.yaml"
    )

    assert config.model_id == "Qwen/Qwen3-ASR-0.6B"
    assert config.train_max_hours == 20.0
    assert config.per_device_batch_size == 1
    assert config.gradient_accumulation_steps == 8


def test_checked_in_one_hour_config_is_a_separate_smoke_profile() -> None:
    config = load_official_h100_config(
        PROJECT_ROOT / "configs" / "train_wrapper_official_sft_h100_1hr.yaml"
    )

    assert config.profile_name == "official_sft_h100_1hr"
    assert config.train_max_hours == 1.0
    assert config.validation_max_hours == 0.1
    assert config.save_steps == 25


def test_h100_config_rejects_model_revision_drift(tmp_path: Path) -> None:
    payload = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "train_wrapper_official_sft_h100_20hr.yaml").read_text(
            encoding="utf-8"
        )
    )
    payload["model"]["revision"] = "main"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(TrainingError, match="pinned non-hf wrapper"):
        load_official_h100_config(path)


def test_prepare_official_jsonl_is_bounded_and_preserves_mixed_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    for name in ("first.flac", "long.flac", "last.flac"):
        (processed / name).write_bytes(b"not-decoded-during-selection")
    source = tmp_path / "train.jsonl"
    rows = [
        {
            "audio_filepath": "first.flac",
            "text": "#incomplete मुझे appointment चाहिए",
            "duration": 2.0,
            "language": "hi",
            "source": "Gram_Vaani",
            "dataset_folder": "GV_Train_100h",
            "original_audio_id": "02-17185-01",
            "original_audio_path": "raw/gram_vaani/GV_Train_100h/Audio/02-17185-01.mp3",
            "sample_rate": 16000,
            "split": "train",
        },
        {
            "audio_filepath": "long.flac",
            "text": "यह clip बहुत लंबी है",
            "duration": 31.0,
            "language": "Hindi",
            "split": "train",
        },
        {
            "audio_filepath": "last.flac",
            "text": "okay",
            "duration": 3.0,
            "language": "en",
            "split": "train",
        },
    ]
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    monkeypatch.setenv("ORATO_DATA_ROOT", str(processed))
    monkeypatch.setenv("ORATO_STORAGE_BACKEND", "local")
    destination = tmp_path / "prepared.jsonl"

    report = prepare_official_jsonl(
        source,
        destination,
        split="train",
        max_hours=0.001,
        max_audio_seconds=30.0,
        project_root=tmp_path,
    )

    prepared = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert report["selected_rows"] == 1
    assert report["skipped_above_max_audio_seconds"] == 1
    assert report["skipped_over_hour_cap"] == 1
    assert report["accepted_source_extra_fields"] == [
        "dataset_folder",
        "original_audio_id",
        "original_audio_path",
        "sample_rate",
    ]
    assert prepared[0]["reference"] == "#incomplete मुझे appointment चाहिए"
    assert prepared[0]["text"] == "language Hindi<asr_text>#incomplete मुझे appointment चाहिए"
    assert prepared[0]["audio"] == str((processed / "first.flac").resolve())


def test_prepare_official_jsonl_rejects_cross_split_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "audio.flac").write_bytes(b"x")
    source = tmp_path / "train.jsonl"
    source.write_text(
        json.dumps(
            {
                "audio_filepath": "audio.flac",
                "text": "नहीं",
                "duration": 1.0,
                "language": "hi",
                "split": "test",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ORATO_DATA_ROOT", str(processed))

    with pytest.raises(TrainingError, match="input contains split"):
        prepare_official_jsonl(
            source,
            tmp_path / "out.jsonl",
            split="train",
            max_hours=1.0,
            max_audio_seconds=30.0,
            project_root=tmp_path,
        )


def test_latest_checkpoint_uses_highest_step(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-9").mkdir()
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-invalid").mkdir()

    assert _latest_checkpoint(tmp_path) == tmp_path / "checkpoint-100"


def test_azure_h100_job_reuses_registered_wrapper_environment() -> None:
    path = PROJECT_ROOT / "azureml" / "jobs" / "official-sft-h100-20hr.yml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert payload["environment"] == "azureml:orato-qwen3-asr-wrapper-lora:2"
    assert payload["inputs"]["processed_data"]["mode"] == "ro_mount"
    assert payload["inputs"]["split_data"]["mode"] == "ro_mount"
    assert payload["outputs"]["training_output"]["mode"] == "rw_mount"
    assert "CUDA_VISIBLE_DEVICES=0" in payload["command"]
    assert "official_h100 run" in payload["command"]
    assert "${{inputs.split_data}}/train.jsonl" in payload["command"]
    assert "${{inputs.split_data}}/valid.jsonl" in payload["command"]
    assert "split_all/v1" not in payload["command"]
    assert "--resume" in payload["command"]


def test_one_hour_job_has_an_isolated_output_and_profile() -> None:
    path = PROJECT_ROOT / "azureml" / "jobs" / "official-sft-h100-1hr.yml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert payload["environment"] == "azureml:orato-qwen3-asr-wrapper-lora:2"
    assert "official-sft-h100-1hr" in payload["outputs"]["training_output"]["path"]
    assert "train_wrapper_official_sft_h100_1hr.yaml" in payload["command"]
    assert payload["limits"]["timeout"] == 21600
