from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
PROFILE_PATHS = sorted(CONFIG_DIR.glob("*.yaml"))


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
    assert "schema_version: 1" in result.stdout
    assert "Qwen/Qwen3-ASR-0.6B" in result.stdout
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
        "Qwen model, CUDA, Azure, audio, and training checks are not implemented yet"
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
    assert "Qwen model and training qualification have not yet run" in result.stdout
    assert result.stderr == ""


def test_preflight_invalid_config_has_no_traceback(tmp_path: Path) -> None:
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text("schema_version: 1\n", encoding="utf-8")

    result = _run_preflight("--config", str(invalid_config))

    assert result.returncode == 2
    assert "Preflight configuration error:" in result.stderr
    assert "Traceback" not in result.stderr
