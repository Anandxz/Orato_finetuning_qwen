from __future__ import annotations

import os
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
PROFILE_PATHS = sorted(
    path
    for path in CONFIG_DIR.glob("*.yaml")
    if not path.name.startswith("train_wrapper_")
)


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "orato_asr.cli", *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_preflight(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "scripts/preflight.py", *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_version_succeeds() -> None:
    result = _run_cli("--version")

    assert result.returncode == 0
    assert result.stdout.strip() == "0.1.0"
    assert result.stderr == ""


@pytest.mark.parametrize("profile_path", PROFILE_PATHS, ids=lambda path: path.stem)
def test_config_show_succeeds_for_every_profile(profile_path: Path) -> None:
    result = _run_cli("config", "show", "--config", str(profile_path))

    assert result.returncode == 0, result.stderr
    assert "schema_version: 3" in result.stdout
    assert "Qwen/Qwen3-ASR-0.6B-hf" in result.stdout
    assert "integration_track: transformers_native" in result.stdout
    assert str(ROOT / "outputs") in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize("profile_path", PROFILE_PATHS, ids=lambda path: path.stem)
def test_config_validate_succeeds_for_every_profile(profile_path: Path) -> None:
    result = _run_cli("config", "validate", "--config", str(profile_path))

    assert result.returncode == 0, result.stderr
    assert "Configuration is valid:" in result.stdout
    assert result.stderr == ""


def test_invalid_config_returns_nonzero_and_actionable_error(tmp_path: Path) -> None:
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text("schema_version: 1\n", encoding="utf-8")

    result = _run_cli("config", "validate", "--config", str(invalid_config))

    assert result.returncode == 2
    assert result.stdout == ""
    assert "Configuration error:" in result.stderr
    assert "missing required keys" in result.stderr
    assert "Traceback" not in result.stderr


def test_doctor_runs_without_gpu_azure_or_internet() -> None:
    result = _run_cli("doctor")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[OK] Python version" in result.stdout
    assert "[OK] Package import" in result.stdout
    assert "[OK] Operating system: Linux" in result.stdout
    assert "[OK] Linux/WSL detection:" in result.stdout
    assert "[OK] Reports write access" in result.stdout
    assert "[OK] Outputs write access" in result.stdout
    assert (
        "Use 'doctor --ml' for Qwen dependency and CUDA checks"
        in result.stdout
    )
    assert result.stderr == ""
    assert not list((ROOT / "reports").glob(".orato-write-check-*"))
    assert not list((ROOT / "outputs").glob(".orato-write-check-*"))


def test_preflight_reports_selected_profile() -> None:
    result = _run_preflight("--config", "configs/local_tiny.yaml")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[INFO] Profile name: local_tiny" in result.stdout
    assert "[INFO] Hardware mode: rtx_3050_6gb (cuda_then_cpu)" in result.stdout
    assert "[INFO] Scale intent: local_pipeline_qualification" in result.stdout
    assert "[INFO] GPU count: 1" in result.stdout
    assert "[INFO] Distributed: false" in result.stdout
    assert "Use --inference for native Qwen dependency/device qualification" in result.stdout
    assert result.stderr == ""


def test_preflight_invalid_config_has_no_traceback(tmp_path: Path) -> None:
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text("schema_version: 1\n", encoding="utf-8")

    result = _run_preflight("--config", str(invalid_config))

    assert result.returncode == 2
    assert "Preflight configuration error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_data_commands_use_explicit_local_reports_and_exit_codes(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "evaluation.jsonl"
    train.write_text(
        json.dumps({"audio_filepath": "audio/a.wav", "text": "hello", "recording_id": "same"}) + "\n",
        encoding="utf-8",
    )
    evaluation.write_text(
        json.dumps({"audio_filepath": "audio/a.wav", "text": "hello", "recording_id": "same"}) + "\n",
        encoding="utf-8",
    )
    report = tmp_path / "validation.json"
    result = _run_cli("data", "validate", "--manifest", str(train), "--report", str(report))
    assert result.returncode == 0, result.stderr
    assert json.loads(report.read_text(encoding="utf-8"))["records"] == 1

    overlap = _run_cli(
        "data", "check-overlap", "--train-manifest", str(train), "--evaluation-manifest", str(evaluation)
    )
    assert overlap.returncode == 1
    assert "audio_path" in overlap.stdout

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("not json\n", encoding="utf-8")
    invalid = _run_cli("data", "validate", "--manifest", str(malformed), "--report", str(tmp_path / "bad.json"))
    assert invalid.returncode == 2
    assert "Traceback" not in invalid.stderr


def test_baseline_cli_returns_one_for_a_guard_stopped_run(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import orato_asr.cli as cli
    import orato_asr.evaluation.baseline as baseline

    fake_result = SimpleNamespace(
        exit_code=1,
        as_dict=lambda: {"status": "stopped", "stopped_reason": "early_collapse_identical_predictions"},
    )
    monkeypatch.setattr(cli, "load_config", lambda _: object())
    monkeypatch.setattr(baseline, "run_baseline", lambda *_, **__: fake_result)

    assert cli.main(["evaluate", "baseline", "--manifest", "unused.jsonl", "--run-name", "stopped"]) == 1
    assert '"status": "stopped"' in capsys.readouterr().out


def test_cli_help_exposes_native_inference_commands() -> None:
    result = _run_cli("--help")

    assert result.returncode == 0
    assert "model" in result.stdout
    assert "transcribe" in result.stdout
    assert "doctor" in result.stdout


def test_unloaded_model_info_reports_pins_without_model_loading() -> None:
    result = _run_cli("model", "info", "--config", "configs/local_tiny.yaml")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "success"
    assert payload["model"]["status"] == "not_loaded"
    assert payload["model"]["id"] == "Qwen/Qwen3-ASR-0.6B-hf"
    assert payload["model"]["revision"].startswith("6aa69c")
    assert result.stderr == ""


def test_distributed_model_load_is_rejected_before_loading_dependencies() -> None:
    result = _run_cli(
        "model",
        "info",
        "--config",
        "configs/h100_8gpu.yaml",
        "--load",
    )

    assert result.returncode == 1
    assert "single-process only" in result.stderr
    assert "Traceback" not in result.stderr


def test_transcribe_failure_writes_sanitized_json(tmp_path: Path) -> None:
    output = tmp_path / "failure.json"
    result = _run_cli(
        "transcribe",
        "--audio",
        str(tmp_path / "missing.wav"),
        "--output-json",
        str(output),
    )

    assert result.returncode == 1
    assert "Transcription failed:" in result.stderr
    assert "Traceback" not in result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert "does not exist" in payload["error"]


def test_transcribe_rejects_uri_cache_before_audio_decode(tmp_path: Path) -> None:
    result = _run_cli(
        "transcribe",
        "--audio",
        str(tmp_path / "missing.wav"),
        "--cache-dir",
        "https://example.invalid/cache",
    )

    assert result.returncode == 1
    assert "not a URI" in result.stderr
    assert "Traceback" not in result.stderr


def test_doctor_ml_reports_dependency_state_without_traceback(tmp_path: Path) -> None:
    report = tmp_path / "environment.json"
    result = _run_cli("doctor", "--ml", "--json", str(report))

    assert result.returncode in {0, 1}
    assert "[INFO] CUDA available:" in result.stdout
    assert "Dependency torch" in result.stdout
    assert "Traceback" not in result.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["model"]["integration_track"] == "transformers_native"
    assert "pytorch" in payload


def test_inference_preflight_writes_structured_report(tmp_path: Path) -> None:
    result = _run_preflight(
        "--inference",
        "--device",
        "cpu",
        "--report-dir",
        str(tmp_path),
    )

    assert result.returncode in {0, 1}
    assert "Inference preflight:" in result.stdout
    assert "Traceback" not in result.stderr
    payload = json.loads(
        (tmp_path / "inference_preflight.json").read_text(encoding="utf-8")
    )
    assert payload["model"]["id"] == "Qwen/Qwen3-ASR-0.6B-hf"
    assert payload["device_requested"] == "cpu"
    assert payload["model_load_requested"] is False
    assert payload["checks"]["output_writable"]["ok"] is True
    assert not list((ROOT / "outputs").glob(".orato-inference-preflight-*"))
