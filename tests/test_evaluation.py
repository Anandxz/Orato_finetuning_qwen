from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from orato_asr.audio import DecodedAudio
from orato_asr.config import load_config
from orato_asr.evaluation.baseline import BaselineOptions, run_baseline
from orato_asr.evaluation.metrics import aggregate_predictions, compute_text_metrics, edit_counts
from orato_asr.evaluation.normalization import NormalizationOptions, normalize_standard

ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[build-system]\nrequires = []\nbuild-backend = 'setuptools.build_meta'\n", encoding="utf-8")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "reports").mkdir()
    values = yaml.safe_load((ROOT / "configs" / "local_tiny.yaml").read_text(encoding="utf-8"))
    values["paths"] = {
        "output_dir": "outputs/unit",
        "reports_dir": "reports/unit",
        "model_cache_dir": "outputs/cache",
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return load_config(path, project_root=tmp_path)


def _manifest(tmp_path: Path, texts: list[str]) -> Path:
    path = tmp_path / "manifest.jsonl"
    rows = [
        {"audio_filepath": f"audio/{index}.wav", "text": text, "duration": 1.0}
        for index, text in enumerate(texts)
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return path


def _decoded(path: str | Path) -> DecodedAudio:
    return DecodedAudio(Path(path), None, 16_000, 16_000, 1, 1, 1.0, False, False)


class _Result:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "success",
            "transcript": self.transcript,
            "language": "hi",
            "model": {"id": "Qwen/Qwen3-ASR-0.6B-hf", "revision": "pinned"},
            "device": "cpu",
            "precision": "float32",
            "timing": {"load_seconds": 0.1, "inference_seconds": 0.2, "real_time_factor": 0.2},
            "peak_cuda_memory_bytes": None,
            "warnings": [],
        }


class _Engine:
    created = 0
    closed = 0

    def __init__(self, *, predictions: list[str], **_: Any) -> None:
        type(self).created += 1
        self.predictions = iter(predictions)

    def transcribe(self, _: DecodedAudio) -> _Result:
        return _Result(next(self.predictions))

    def close(self) -> None:
        type(self).closed += 1


def _engine_factory(predictions: list[str]):
    return lambda **kwargs: _Engine(predictions=predictions, **kwargs)


def test_normalization_and_edit_metrics_preserve_mixed_script() -> None:
    options = NormalizationOptions(remove_punctuation=True, lowercase_latin=True)
    assert normalize_standard("  नमस्ते—HELLO!  ", options) == "नमस्ते hello"
    assert edit_counts(["a", "b"], ["a", "c", "b"]).insertions == 1
    metrics = compute_text_metrics("Hello दुनिया!", "hello दुनिया", options=options)
    assert metrics["wer"] == 0.0
    assert metrics["cer"] == 0.0
    totals = aggregate_predictions([{"status": "success", **metrics, "audio_duration_seconds": 1.0, "inference_seconds": 0.5, "real_time_factor": 0.5}])
    assert totals["normalized"]["wer"] == 0.0
    assert totals["failure_rate"] == 0.0


def test_baseline_persists_resumes_and_writes_reports(tmp_path: Path) -> None:
    _Engine.created = _Engine.closed = 0
    config = _config(tmp_path)
    manifest = _manifest(tmp_path, ["hello", "नमस्ते"])
    first = run_baseline(
        manifest,
        config,
        options=BaselineOptions(run_name="resume", max_samples=1),
        engine_factory=_engine_factory(["hello"]),
        audio_decoder=_decoded,
    )
    resumed = run_baseline(
        manifest,
        config,
        options=BaselineOptions(run_name="resume", max_samples=2, resume=True),
        engine_factory=_engine_factory(["नमस्ते"]),
        audio_decoder=_decoded,
    )
    assert first.status == resumed.status == "completed"
    assert resumed.summary["resumed_samples"] == 1
    assert resumed.metrics["successful_samples"] == 2
    assert _Engine.created == _Engine.closed == 2
    for name in ("run_config.json", "summary.json", "predictions.jsonl", "failures.jsonl", "metrics.json", "metrics.csv", "worst_examples.json", "README.md"):
        assert (resumed.run_directory / name).exists()


def test_baseline_stops_on_early_identical_predictions(tmp_path: Path) -> None:
    config = _config(tmp_path)
    manifest = _manifest(tmp_path, ["one", "two", "three", "four", "five", "six"])
    result = run_baseline(
        manifest,
        config,
        options=BaselineOptions(run_name="collapse"),
        engine_factory=_engine_factory(["same"] * 6),
        audio_decoder=_decoded,
    )
    assert result.status == "stopped"
    assert result.exit_code == 1
    assert result.stopped_reason == "early_collapse_identical_predictions"
    assert result.metrics["samples"] == 5


def test_baseline_continue_policy_records_per_sample_failure(tmp_path: Path) -> None:
    config = _config(tmp_path)
    manifest = _manifest(tmp_path, ["one"])
    result = run_baseline(
        manifest,
        config,
        options=BaselineOptions(run_name="failure"),
        engine_factory=_engine_factory(["unused"]),
        audio_decoder=lambda _: (_ for _ in ()).throw(OSError("decoder unavailable")),
    )
    assert result.status == "completed"
    assert result.exit_code == 0
    assert result.metrics["failed_samples"] == 1
    assert (result.run_directory / "failures.jsonl").read_text(encoding="utf-8")
