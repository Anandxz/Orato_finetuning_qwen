from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from orato_asr.exceptions import TrainingError
from orato_asr.training.memory import (
    BYTES_PER_GIB,
    MemoryGuardConfig,
    MemoryGuardError,
    build_failure_metadata,
    capture_memory_snapshot,
    convert_cuda_oom,
    enforce_memory_guard,
    evaluate_memory_guard,
)
from orato_asr.training.reporting import (
    append_atomic_jsonl,
    build_training_summary,
    render_cto_smoke_summary,
    render_training_run_readme,
    resolve_training_run_directories,
    sanitize_report_payload,
    write_atomic_csv,
    write_atomic_json,
    write_atomic_jsonl,
    write_selected_sample_ids,
    write_training_run_readme,
)

ROOT = Path(__file__).resolve().parents[1]


class _CudaOOM(RuntimeError):
    pass


class _FakeCuda:
    OutOfMemoryError = _CudaOOM

    def is_available(self) -> bool:
        return True

    def memory_allocated(self, device: int) -> int:
        assert device == 0
        return 4 * BYTES_PER_GIB

    def memory_reserved(self, device: int) -> int:
        assert device == 0
        return 5 * BYTES_PER_GIB

    def max_memory_allocated(self, device: int) -> int:
        assert device == 0
        return 5 * BYTES_PER_GIB

    def max_memory_reserved(self, device: int) -> int:
        assert device == 0
        return 5 * BYTES_PER_GIB + 128


_TORCH = SimpleNamespace(cuda=_FakeCuda(), OutOfMemoryError=_CudaOOM)


def _snapshot(**kwargs: object):
    return capture_memory_snapshot(
        "before_backward",
        torch_module=_TORCH,
        meminfo_reader=lambda: "MemTotal: 8388608 kB\nMemAvailable: 4194304 kB\n",
        clock=lambda: datetime(2026, 7, 14, tzinfo=timezone.utc),
        **kwargs,
    )


def test_memory_snapshot_is_lazy_structured_and_filters_current_process() -> None:
    snapshot = _snapshot(
        current_pid=10,
        cuda_process_detector=lambda device, pid: [
            {"pid": pid, "used_memory_bytes": 4 * BYTES_PER_GIB, "name": "this"},
            {"pid": 20, "used_memory_bytes": 700 * 1024**2, "name": " other  job "},
        ],
    )
    payload = snapshot.as_dict()

    assert payload["captured_at_utc"] == "2026-07-14T00:00:00Z"
    assert payload["system_ram"] == {
        "status": "available",
        "source": "/proc/meminfo",
        "total_bytes": 8 * BYTES_PER_GIB,
        "available_bytes": 4 * BYTES_PER_GIB,
        "used_bytes": 4 * BYTES_PER_GIB,
        "used_ratio": 0.5,
    }
    assert payload["cuda"]["allocated_bytes"] == 4 * BYTES_PER_GIB
    assert payload["cuda"]["reserved_bytes"] == 5 * BYTES_PER_GIB
    assert payload["cuda"]["peak_reserved_bytes"] == 5 * BYTES_PER_GIB + 128
    assert payload["cuda_process_check"] == {
        "status": "checked",
        "other_processes": [
            {"pid": 20, "used_memory_bytes": 700 * 1024**2, "name": "other job"}
        ],
    }


def test_memory_guard_reports_all_reasons_and_can_abort() -> None:
    snapshot = _snapshot(
        current_pid=10,
        cuda_process_detector=lambda _device, _pid: [
            {"pid": 20, "used_memory_bytes": 700 * 1024**2}
        ],
    )
    guard = MemoryGuardConfig(
        minimum_available_system_bytes=5 * BYTES_PER_GIB,
        gpu_safety_limit_bytes=5 * BYTES_PER_GIB,
        large_cuda_process_bytes=512 * 1024**2,
        require_cuda_process_check=True,
    )
    result = evaluate_memory_guard(snapshot, guard)
    codes = {item["code"] for item in result.violations}

    assert result.safe is False
    assert codes == {
        "system_ram_below_minimum",
        "gpu_memory_at_or_above_safety_limit",
        "other_large_cuda_process_active",
    }
    with pytest.raises(MemoryGuardError) as raised:
        enforce_memory_guard(snapshot, guard)
    assert raised.value.metadata["status"] == "blocked"
    assert "No CPU or disk fallback" in str(raised.value)


def test_memory_guard_can_report_without_raising_and_requires_process_hook() -> None:
    snapshot = _snapshot()
    guard = MemoryGuardConfig(
        minimum_available_system_bytes=1,
        gpu_safety_limit_bytes=6 * BYTES_PER_GIB,
        abort_on_threshold=False,
        require_cuda_process_check=True,
    )
    result = enforce_memory_guard(snapshot, guard)
    assert result.safe is False
    assert [item["code"] for item in result.violations] == [
        "cuda_process_check_unavailable"
    ]


def test_oom_conversion_is_structured_sanitized_and_requires_fresh_process() -> None:
    source = _CudaOOM(
        "CUDA out of memory at https://private.example/audio?sig=secret "
        "using hf_abcdefghijklmnop"
    )
    snapshot = _snapshot()
    metadata = build_failure_metadata(
        source,
        stage="backward",
        snapshot=snapshot,
        torch_module=_TORCH,
    )
    converted = convert_cuda_oom(
        source,
        stage="backward",
        snapshot=snapshot,
        torch_module=_TORCH,
    )

    assert metadata["category"] == "cuda_out_of_memory"
    assert metadata["requires_fresh_process_before_retry"] is True
    assert metadata["cpu_fallback_attempted"] is False
    assert "private.example" not in metadata["message"]
    assert "hf_abcdefghijklmnop" not in metadata["message"]
    assert converted.metadata == metadata
    assert "no CPU fallback" in str(converted)


def test_importing_memory_and_reporting_does_not_import_torch() -> None:
    code = (
        "import sys; "
        "import orato_asr.training.memory, orato_asr.training.reporting; "
        "print('torch' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False"


def test_run_directories_are_traversal_safe_and_created_under_configured_roots(
    tmp_path: Path,
) -> None:
    (tmp_path / "outputs" / "training").mkdir(parents=True)
    (tmp_path / "reports" / "training").mkdir(parents=True)
    directories = resolve_training_run_directories(
        project_root=tmp_path,
        output_root="outputs/training",
        reports_root=tmp_path / "reports" / "training",
        run_name="qwen06b_lora_one_step",
        create=True,
    )

    assert directories.output_directory == (
        tmp_path / "outputs" / "training" / "qwen06b_lora_one_step"
    )
    assert directories.report_directory.is_dir()
    assert directories.adapter_directory.is_dir()
    assert directories.verification_directory.is_dir()
    with pytest.raises(TrainingError, match="run name"):
        resolve_training_run_directories(
            project_root=tmp_path,
            output_root="outputs/training",
            reports_root="reports/training",
            run_name="../escape",
        )
    with pytest.raises(TrainingError, match="inside"):
        resolve_training_run_directories(
            project_root=tmp_path,
            output_root=tmp_path / "outside",
            reports_root="reports/training",
            run_name="safe",
        )


def test_atomic_json_jsonl_csv_and_selected_id_writes(tmp_path: Path) -> None:
    json_path = write_atomic_json(
        tmp_path / "summary.json",
        {
            "text": "मुझे appointment चाहिए",
            "access_token": "secret-value",
            "endpoint": "https://private.example/blob?sig=secret",
        },
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload == {
        "access_token": "[redacted]",
        "endpoint": "https://private.example/blob",
        "text": "मुझे appointment चाहिए",
    }
    with pytest.raises(TrainingError, match="overwrite"):
        write_atomic_json(json_path, {"changed": True})

    jsonl_path = write_atomic_jsonl(
        tmp_path / "events.jsonl",
        ({"step": step} for step in (1, 2)),
    )
    append_atomic_jsonl(jsonl_path, {"step": 3})
    assert [json.loads(line) for line in jsonl_path.read_text().splitlines()] == [
        {"step": 1},
        {"step": 2},
        {"step": 3},
    ]

    csv_path = write_atomic_csv(
        tmp_path / "metrics.csv",
        [{"step": 1, "loss": 2.5}, {"step": 2, "loss": 2.0}],
        fieldnames=("step", "loss"),
    )
    with csv_path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == [
            {"step": "1", "loss": "2.5"},
            {"step": "2", "loss": "2.0"},
        ]

    selected = write_selected_sample_ids(
        [
            {
                "sample_id": "abc",
                "line_number": 7,
                "duration_seconds": 1.25,
                "audio_filepath": "/private/audio.wav",
                "text": "must not be copied",
            }
        ],
        tmp_path / "selected_sample_ids.jsonl",
    )
    selected_payload = json.loads(selected.read_text(encoding="utf-8"))
    assert selected_payload == {
        "audio_filepath": "/private/audio.wav",
        "duration_seconds": 1.25,
        "manifest_line": 7,
        "sample_id": "abc",
    }
    assert "text" not in selected.read_text(encoding="utf-8")


def test_reports_reject_binary_payloads_and_nonfinite_numbers() -> None:
    with pytest.raises(TrainingError, match="binary bytes"):
        sanitize_report_payload({"audio": b"RIFF"})
    with pytest.raises(TrainingError, match="NaN or Inf"):
        sanitize_report_payload({"loss": float("nan")})


def test_training_summary_distinguishes_total_eligible_selected_and_consumed() -> None:
    summary = build_training_summary(
        status="one_step_completed",
        total_manifest_samples=120,
        total_manifest_duration_seconds=7200,
        eligible_samples=100,
        eligible_duration_seconds=3600,
        selected_samples=100,
        selected_duration_seconds=3600,
        consumed_samples=8,
        unique_consumed_samples=8,
        consumed_audio_seconds=48,
        microsteps=8,
        optimizer_steps=1,
        per_device_batch_size=1,
        gradient_accumulation_steps=8,
        runtime_seconds=80,
        complete_epoch_performed=False,
    )

    assert summary["dataset"]["total_manifest_hours"] == 2.0
    assert summary["dataset"]["eligible_hours"] == 1.0
    assert summary["consumption"]["samples"] == 8
    assert summary["consumption"]["microsteps"] == 8
    assert summary["consumption"]["optimizer_steps"] == 1
    assert summary["consumption"]["complete_epoch_performed"] is False
    assert summary["consumption"]["estimated_complete_epoch_runtime_seconds"] == 6000
    assert summary["claims"]["full_input_manifest_consumed"] is False
    with pytest.raises(ValueError, match="complete eligible epoch"):
        build_training_summary(
            status="invalid",
            total_manifest_samples=100,
            total_manifest_duration_seconds=100,
            eligible_samples=100,
            eligible_duration_seconds=100,
            selected_samples=100,
            selected_duration_seconds=100,
            consumed_samples=8,
            unique_consumed_samples=8,
            consumed_audio_seconds=8,
            microsteps=8,
            optimizer_steps=1,
            per_device_batch_size=1,
            gradient_accumulation_steps=8,
            runtime_seconds=10,
            complete_epoch_performed=True,
        )


def test_cto_summary_keeps_unverified_fields_pending_and_disclaims_accuracy() -> None:
    summary = build_training_summary(
        status="one_step_completed",
        total_manifest_samples=120,
        total_manifest_duration_seconds=7200,
        eligible_samples=100,
        eligible_duration_seconds=3600,
        selected_samples=100,
        selected_duration_seconds=3600,
        consumed_samples=8,
        unique_consumed_samples=8,
        consumed_audio_seconds=48,
        microsteps=8,
        optimizer_steps=1,
        per_device_batch_size=1,
        gradient_accumulation_steps=8,
        runtime_seconds=80,
        complete_epoch_performed=False,
    )
    report = render_cto_smoke_summary(
        summary,
        {
            "model": {"id": "Qwen/Qwen3-ASR-0.6B", "backend": "qwen-asr"},
            "lora": {"rank": 4, "targets": "text q_proj/v_proj"},
            "trainable_parameters": 1234,
            "adapter": {"saved": True, "reloaded": None},
        },
    )

    assert "Supplied manifest: 120 samples, 2.0000 hours" in report
    assert "Eligible after filtering: 100 samples, 1.0000 hours" in report
    assert "Actually consumed: 8 sample uses (8 unique), 48.000 seconds" in report
    assert "Complete eligible epoch: no" in report
    assert "Adapter saved / fresh-reloaded: yes / PENDING" in report
    assert "Starting / ending loss: PENDING" in report
    assert "does not establish accuracy improvement" in report
    assert "full-manifest coverage" in report


def _readme_summary() -> dict[str, object]:
    return build_training_summary(
        status="smoke_completed",
        total_manifest_samples=120,
        total_manifest_duration_seconds=7200,
        eligible_samples=100,
        eligible_duration_seconds=3600,
        selected_samples=80,
        selected_duration_seconds=3000,
        consumed_samples=8,
        unique_consumed_samples=8,
        consumed_audio_seconds=48,
        microsteps=8,
        optimizer_steps=1,
        per_device_batch_size=1,
        gradient_accumulation_steps=8,
        runtime_seconds=80,
        complete_epoch_performed=False,
    )


def test_training_run_readme_reports_measured_facts_and_nonclaims() -> None:
    report = render_training_run_readme(
        _readme_summary(),
        {
            "losses": {"initial": 2.5, "final": 2.0},
            "gradient_norms": [1.25, 0.75],
            "memory": {
                "peak_cuda_allocated_bytes": 4 * BYTES_PER_GIB,
                "peak_cuda_reserved_bytes": 5 * BYTES_PER_GIB,
                "peak_system_used_bytes": 6 * BYTES_PER_GIB,
            },
            "adapter": {
                "saved": True,
                "fresh_process_reload": True,
                "path": "https://private.example/adapter?sig=secret",
            },
            "predictions": {
                "base": "मुझे appointment चाहिए hf_abcdefghijklmnop",
                "adapter": "मुझे appointment चाहिए",
            },
            "metrics": {"adapter_wer": 0.25, "base_wer": 0.5},
        },
    )

    assert "Supplied manifest: 120 samples, 7200.000 seconds (2.0000 hours)" in report
    assert "Eligible after duration filtering: 100 samples, 3600.000 seconds (1.0000 hours)" in report
    assert "Selected by configured caps: 80 samples, 3000.000 seconds" in report
    assert "Actually consumed: 8 sample uses (8 unique), 48.000 seconds (0.0133 hours)" in report
    assert "Microsteps / optimizer steps: 8 / 1" in report
    assert "Per-device batch / accumulation / effective batch: 1 / 8 / 8" in report
    assert "Starting / ending loss: 2.5 / 2" in report
    assert "Gradient norms: 1.25, 0.75" in report
    assert "Peak CUDA allocated: 4.000 GiB" in report
    assert "Peak CUDA reserved: 5.000 GiB" in report
    assert "Peak system RAM used: 6.000 GiB" in report
    assert "Adapter saved: yes" in report
    assert "Adapter fresh-process reload: yes" in report
    assert "Adapter path: https://private.example/adapter" in report
    assert "sig=secret" not in report
    assert "hf_abcdefghijklmnop" not in report
    assert "[redacted-token]" in report
    assert "Evaluation metrics: adapter_wer=0.25, base_wer=0.5" in report
    assert "Estimated complete-epoch runtime: 6000.000 seconds" in report
    assert "Full supplied manifest consumed: no" in report
    assert "does not establish accuracy improvement" in report
    assert "only the explicitly reported consumed duration was used" in report


def test_training_run_readme_keeps_missing_evidence_pending() -> None:
    report = render_training_run_readme(_readme_summary())

    assert "Starting / ending loss: PENDING" in report
    assert "Gradient norms: PENDING — not measured" in report
    assert "Peak CUDA allocated: PENDING — not measured" in report
    assert "Adapter saved: PENDING — not verified" in report
    assert "Base prediction: PENDING — not captured" in report
    assert "Evaluation metrics: PENDING — not measured" in report


def test_training_run_readme_writer_is_atomic_sanitized_and_explicit_overwrite(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "README.md"
    written = write_training_run_readme(
        _readme_summary(),
        {"adapter": {"saved": True, "path": "https://host/path?token=private"}},
        destination,
    )

    assert written == destination.resolve()
    assert destination.read_text(encoding="utf-8").endswith("\n")
    assert "?token=private" not in destination.read_text(encoding="utf-8")
    assert not list(tmp_path.glob(".README.md.*.tmp"))
    with pytest.raises(TrainingError, match="overwrite"):
        write_training_run_readme(_readme_summary(), None, destination)
    write_training_run_readme(
        _readme_summary(),
        {"adapter": {"saved": False}},
        destination,
        overwrite=True,
    )
    assert "Adapter saved: no" in destination.read_text(encoding="utf-8")


def test_training_run_readme_rejects_nonfinite_or_binary_evidence() -> None:
    with pytest.raises(TrainingError, match="NaN or Inf"):
        render_training_run_readme(
            _readme_summary(),
            {"gradient_norms": [float("inf")]},
        )
    with pytest.raises(TrainingError, match="binary bytes"):
        render_training_run_readme(
            _readme_summary(),
            {"predictions": {"base": b"audio bytes"}},
        )
