from __future__ import annotations

import os
import json
import shutil
from pathlib import Path

import pytest

from orato_asr.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_real_native_model_transcription_when_owner_audio_is_supplied() -> None:
    audio_value = os.environ.get("ORATO_ASR_TEST_AUDIO")
    if not audio_value:
        pytest.skip("ORATO_ASR_TEST_AUDIO is unset; no legal non-PII audio was supplied")

    from orato_asr.audio import decode_audio
    from orato_asr.models.qwen3_asr import Qwen3ASREngine

    config = load_config(ROOT / "configs" / "local_tiny.yaml", project_root=ROOT)
    values = config.as_dict()
    audio = decode_audio(audio_value)
    engine = Qwen3ASREngine(
        device=values["inference"]["device"],
        precision=values["inference"]["precision"],
        cache_dir=values["paths"]["model_cache_dir"],
        offline=values["inference"]["offline"],
        language=values["inference"]["language_hint"],
        max_new_tokens=values["inference"]["max_new_tokens"],
    )
    try:
        result = engine.transcribe(audio)
    finally:
        engine.close()

    assert result.status == "success"
    assert result.transcript and any(character.isalnum() for character in result.transcript)
    assert result.load_seconds > 0
    assert result.inference_seconds > 0
    assert result.real_time_factor is not None and result.real_time_factor > 0
    assert result.model["revision"] == "6aa69c382e2b426eee1f5870d4c95859a74b6445"


@pytest.mark.integration
def test_real_baseline_when_owner_manifest_is_supplied() -> None:
    manifest_value = os.environ.get("ORATO_ASR_TEST_MANIFEST")
    if not manifest_value:
        pytest.skip("ORATO_ASR_TEST_MANIFEST is unset; no legal local manifest was supplied")

    from orato_asr.evaluation.baseline import BaselineOptions, run_baseline

    config = load_config(ROOT / "configs" / "local_tiny.yaml", project_root=ROOT)
    run_name = "integration-manifest-temporary"
    try:
        result = run_baseline(
            manifest_value,
            config,
            options=BaselineOptions(
                run_name=run_name,
                max_samples=3,
                offline=True,
                overwrite=True,
            ),
        )
        assert result.status == "completed"
        predictions = [
            json.loads(line)
            for line in (result.run_directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert 1 <= len(predictions) <= 3
        assert all(row["status"] == "success" for row in predictions)
        assert all(any(character.isalnum() for character in row["transcript"]) for row in predictions)
        assert all(float(row["inference_seconds"]) > 0 for row in predictions)
        assert all(row["model"]["revision"] == "6aa69c382e2b426eee1f5870d4c95859a74b6445" for row in predictions)
    finally:
        shutil.rmtree(
            ROOT / "reports" / "local_tiny" / "evaluation" / run_name,
            ignore_errors=True,
        )
