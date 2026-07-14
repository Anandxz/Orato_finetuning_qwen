"""Safe, portable reporting primitives for bounded wrapper-LoRA runs."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from orato_asr.exceptions import TrainingError

_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_TOKEN = re.compile(r"\bhf_[A-Za-z0-9]{12,}\b")
_SENSITIVE_KEYS = {
    "access_token",
    "api_key",
    "authorization",
    "credential",
    "credentials",
    "password",
    "refresh_token",
    "sas_token",
    "secret",
}


@dataclass(frozen=True, slots=True)
class TrainingRunDirectories:
    """Validated output/report locations for one training run."""

    run_name: str
    output_directory: Path
    report_directory: Path
    adapter_directory: Path
    verification_directory: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "run_name": self.run_name,
            "output_directory": str(self.output_directory),
            "report_directory": str(self.report_directory),
            "adapter_directory": str(self.adapter_directory),
            "verification_directory": str(self.verification_directory),
        }


def validate_run_name(run_name: str) -> str:
    """Accept a compact filename component, never a path or hidden directory."""

    if not isinstance(run_name, str) or not _RUN_NAME.fullmatch(run_name):
        raise TrainingError(
            "Training run name must start with an ASCII letter or digit and contain "
            "only letters, digits, '.', '_', or '-' (maximum 128 characters)"
        )
    if run_name in {".", ".."}:
        raise TrainingError("Training run name must not be '.' or '..'")
    return run_name


def resolve_training_run_directories(
    *,
    project_root: str | Path,
    output_root: str | Path,
    reports_root: str | Path,
    run_name: str,
    create: bool = False,
    allow_existing: bool = False,
) -> TrainingRunDirectories:
    """Resolve a run strictly below configured repository output/report roots."""

    if type(create) is not bool or type(allow_existing) is not bool:
        raise ValueError("create and allow_existing must be true or false")
    safe_name = validate_run_name(run_name)
    project = Path(project_root).expanduser().resolve()
    output_base = _configured_root(
        output_root,
        project_root=project,
        allowed_root=project / "outputs",
        label="output_root",
    )
    report_base = _configured_root(
        reports_root,
        project_root=project,
        allowed_root=project / "reports",
        label="reports_root",
    )
    output_directory = (output_base / safe_name).resolve()
    report_directory = (report_base / safe_name).resolve()
    _require_descendant(output_directory, output_base, "training output directory")
    _require_descendant(report_directory, report_base, "training report directory")

    directories = TrainingRunDirectories(
        run_name=safe_name,
        output_directory=output_directory,
        report_directory=report_directory,
        adapter_directory=output_directory / "adapter",
        verification_directory=output_directory / "verification",
    )
    if create:
        existing = [
            path
            for path in (output_directory, report_directory)
            if path.exists()
        ]
        if existing and not allow_existing:
            listed = ", ".join(str(path) for path in existing)
            raise TrainingError(
                f"Training run already exists; choose a new run name or explicitly "
                f"allow resume: {listed}"
            )
        for path in (
            output_directory,
            report_directory,
            directories.adapter_directory,
            directories.verification_directory,
        ):
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise TrainingError(f"Could not create training directory {path}: {exc}") from exc
    return directories


def write_atomic_json(
    path: str | Path,
    payload: object,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write sanitized, strict UTF-8 JSON."""

    normalized = sanitize_report_payload(payload)
    text = json.dumps(
        normalized,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    return _atomic_text(path, text, overwrite=overwrite)


def write_json_atomic(
    payload: object,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Payload-first alias matching the project's existing manifest writer."""

    return write_atomic_json(path, payload, overwrite=overwrite)


def write_atomic_jsonl(
    path: str | Path,
    rows: Iterable[object],
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically replace a JSONL file while consuming rows one at a time."""

    destination = _local_destination(path)
    return _atomic_stream(
        destination,
        (
            json.dumps(
                sanitize_report_payload(row),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        overwrite=overwrite,
    )


def append_atomic_jsonl(path: str | Path, row: object) -> Path:
    """Append one row by atomically replacing the file, never a partial line."""

    destination = _local_destination(path)
    normalized = sanitize_report_payload(row)
    line = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            if destination.exists():
                with destination.open("rb") as existing:
                    for chunk in iter(lambda: existing.read(1024 * 1024), b""):
                        temporary.write(chunk)
            temporary.write(line.encode("utf-8"))
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        _fsync_directory(destination.parent)
    except OSError as exc:
        raise TrainingError(f"Could not append JSONL report {destination}: {exc}") from exc
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return destination


def write_atomic_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, object]],
    *,
    fieldnames: Sequence[str],
    overwrite: bool = False,
) -> Path:
    """Atomically write a deterministic UTF-8 CSV with an explicit schema."""

    if not fieldnames or any(not isinstance(item, str) or not item for item in fieldnames):
        raise ValueError("fieldnames must contain non-empty strings")
    if len(set(fieldnames)) != len(fieldnames):
        raise ValueError("fieldnames must not contain duplicates")
    destination = _local_destination(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise TrainingError(f"Refusing to overwrite existing report: {destination}")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            writer = csv.DictWriter(
                temporary,
                fieldnames=list(fieldnames),
                extrasaction="raise",
            )
            writer.writeheader()
            for row in rows:
                normalized = sanitize_report_payload(row)
                if not isinstance(normalized, Mapping):
                    raise TypeError("CSV rows must be mappings")
                writer.writerow(
                    {
                        key: _csv_value(normalized.get(key))
                        for key in fieldnames
                    }
                )
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        _fsync_directory(destination.parent)
    except (csv.Error, OSError, TypeError, ValueError) as exc:
        raise TrainingError(f"Could not write CSV report {destination}: {exc}") from exc
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return destination


def write_selected_sample_ids(
    samples: Iterable[object],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write lightweight selected-sample references, never transcripts or audio."""

    def rows() -> Iterable[dict[str, Any]]:
        seen: set[str] = set()
        allowed = {
            "sample_id",
            "manifest_line",
            "line_number",
            "duration_seconds",
            "audio_filepath",
        }
        for item in samples:
            if isinstance(item, Mapping):
                raw = dict(item)
            else:
                serializer = getattr(item, "as_dict", None)
                if not callable(serializer):
                    raise TypeError("Selected sample rows must be mappings or expose as_dict()")
                raw = serializer()
            sample_id = raw.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise TrainingError("Selected sample row is missing a non-empty sample_id")
            if sample_id in seen:
                raise TrainingError(f"Selected sample ID is duplicated: {sample_id}")
            seen.add(sample_id)
            row = {key: value for key, value in raw.items() if key in allowed}
            if "line_number" in row and "manifest_line" not in row:
                row["manifest_line"] = row.pop("line_number")
            yield row

    return write_atomic_jsonl(path, rows(), overwrite=overwrite)


def build_training_summary(
    *,
    status: str,
    total_manifest_samples: int,
    total_manifest_duration_seconds: int | float,
    eligible_samples: int,
    eligible_duration_seconds: int | float,
    selected_samples: int,
    selected_duration_seconds: int | float,
    consumed_samples: int,
    unique_consumed_samples: int,
    consumed_audio_seconds: int | float,
    microsteps: int,
    optimizer_steps: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    runtime_seconds: int | float | None,
    complete_epoch_performed: bool,
    epoch_estimate_runtime_seconds: int | float | None = None,
) -> dict[str, Any]:
    """Build a summary that cannot conflate supplied, eligible, and consumed data."""

    if not isinstance(status, str) or not status.strip():
        raise ValueError("status must be a non-empty string")
    counts = {
        "total_manifest_samples": total_manifest_samples,
        "eligible_samples": eligible_samples,
        "selected_samples": selected_samples,
        "consumed_samples": consumed_samples,
        "unique_consumed_samples": unique_consumed_samples,
        "microsteps": microsteps,
        "optimizer_steps": optimizer_steps,
    }
    for name, value in counts.items():
        _nonnegative_int(value, name)
    _positive_int(per_device_batch_size, "per_device_batch_size")
    _positive_int(gradient_accumulation_steps, "gradient_accumulation_steps")
    durations = {
        "total_manifest_duration_seconds": total_manifest_duration_seconds,
        "eligible_duration_seconds": eligible_duration_seconds,
        "selected_duration_seconds": selected_duration_seconds,
        "consumed_audio_seconds": consumed_audio_seconds,
    }
    for name, value in durations.items():
        _nonnegative_number(value, name)
    if runtime_seconds is not None:
        _nonnegative_number(runtime_seconds, "runtime_seconds")
    if epoch_estimate_runtime_seconds is not None:
        _nonnegative_number(
            epoch_estimate_runtime_seconds,
            "epoch_estimate_runtime_seconds",
        )
    if type(complete_epoch_performed) is not bool:
        raise ValueError("complete_epoch_performed must be true or false")

    if not total_manifest_samples >= eligible_samples >= selected_samples:
        raise ValueError("sample counts must satisfy total >= eligible >= selected")
    if unique_consumed_samples > consumed_samples:
        raise ValueError("unique_consumed_samples cannot exceed consumed_samples")
    if unique_consumed_samples > selected_samples:
        raise ValueError("unique_consumed_samples cannot exceed selected_samples")
    if optimizer_steps > microsteps:
        raise ValueError("optimizer_steps cannot exceed microsteps")
    if (
        float(eligible_duration_seconds) > float(total_manifest_duration_seconds) + 1e-9
        or float(selected_duration_seconds) > float(eligible_duration_seconds) + 1e-9
    ):
        raise ValueError("durations must satisfy total >= eligible >= selected")
    if complete_epoch_performed and unique_consumed_samples < eligible_samples:
        raise ValueError(
            "A complete eligible epoch cannot be reported before every eligible sample "
            "has been consumed at least once"
        )

    runtime = None if runtime_seconds is None else float(runtime_seconds)
    estimate_runtime = (
        runtime
        if epoch_estimate_runtime_seconds is None
        else float(epoch_estimate_runtime_seconds)
    )
    epoch_estimate = _estimate_epoch_runtime(
        runtime_seconds=estimate_runtime,
        consumed_audio_seconds=float(consumed_audio_seconds),
        eligible_duration_seconds=float(eligible_duration_seconds),
        unique_consumed_samples=unique_consumed_samples,
        eligible_samples=eligible_samples,
    )
    epoch_fraction = (
        min(1.0, unique_consumed_samples / eligible_samples)
        if eligible_samples
        else None
    )
    full_manifest_consumed = (
        complete_epoch_performed
        and eligible_samples == total_manifest_samples
        and math.isclose(
            float(eligible_duration_seconds),
            float(total_manifest_duration_seconds),
            rel_tol=0,
            abs_tol=1e-6,
        )
    )
    return {
        "status": status.strip(),
        "dataset": {
            "total_manifest_samples": total_manifest_samples,
            "total_manifest_duration_seconds": float(total_manifest_duration_seconds),
            "total_manifest_hours": float(total_manifest_duration_seconds) / 3600,
            "eligible_samples": eligible_samples,
            "eligible_duration_seconds": float(eligible_duration_seconds),
            "eligible_hours": float(eligible_duration_seconds) / 3600,
            "selected_samples": selected_samples,
            "selected_duration_seconds": float(selected_duration_seconds),
            "selected_hours": float(selected_duration_seconds) / 3600,
        },
        "consumption": {
            "samples": consumed_samples,
            "unique_samples": unique_consumed_samples,
            "audio_duration_seconds": float(consumed_audio_seconds),
            "audio_hours": float(consumed_audio_seconds) / 3600,
            "microsteps": microsteps,
            "optimizer_steps": optimizer_steps,
            "per_device_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": per_device_batch_size
            * gradient_accumulation_steps,
            "runtime_seconds": runtime,
            "optimizer_loop_runtime_seconds": (
                None
                if epoch_estimate_runtime_seconds is None
                else float(epoch_estimate_runtime_seconds)
            ),
            "optimizer_samples_per_second": (
                consumed_samples / estimate_runtime
                if estimate_runtime is not None and estimate_runtime > 0
                else None
            ),
            "optimizer_audio_seconds_per_second": (
                float(consumed_audio_seconds) / estimate_runtime
                if estimate_runtime is not None and estimate_runtime > 0
                else None
            ),
            "complete_epoch_performed": complete_epoch_performed,
            "eligible_epoch_fraction_by_unique_samples": epoch_fraction,
            "estimated_complete_epoch_runtime_seconds": epoch_estimate,
        },
        "claims": {
            "full_input_manifest_consumed": full_manifest_consumed,
            "accuracy_improvement_demonstrated": False,
            "production_ready": False,
            "h100_qualified": False,
        },
    }


def render_cto_smoke_summary(
    summary: Mapping[str, Any],
    facts: Mapping[str, Any] | None = None,
) -> str:
    """Render a concise CTO note with explicit pending and non-claim fields."""

    facts = {} if facts is None else dict(facts)
    dataset = _mapping(summary.get("dataset"), "summary.dataset")
    consumption = _mapping(summary.get("consumption"), "summary.consumption")
    dependencies = facts.get("dependencies")
    dependency_text = _mapping_inline(dependencies) if isinstance(dependencies, Mapping) else _pending("versions not captured")
    machine = facts.get("machine")
    machine_text = _mapping_inline(machine) if isinstance(machine, Mapping) else _pending("machine specification not captured")
    model = facts.get("model")
    model_text = _mapping_inline(model) if isinstance(model, Mapping) else _pending("model metadata not captured")
    lora = facts.get("lora")
    lora_text = _mapping_inline(lora) if isinstance(lora, Mapping) else _pending("LoRA configuration not captured")
    memory = facts.get("memory") if isinstance(facts.get("memory"), Mapping) else {}
    adapter = facts.get("adapter") if isinstance(facts.get("adapter"), Mapping) else {}
    predictions = facts.get("predictions") if isinstance(facts.get("predictions"), Mapping) else {}
    losses = facts.get("losses") if isinstance(facts.get("losses"), Mapping) else {}

    limitations = facts.get("known_limitations")
    if isinstance(limitations, str):
        limitation_lines = [limitations]
    elif isinstance(limitations, Sequence) and not isinstance(limitations, (bytes, bytearray)):
        limitation_lines = [str(item) for item in limitations if str(item).strip()]
    else:
        limitation_lines = []
    limitation_lines.append(
        "A 5–10-step smoke run is pipeline evidence only; it does not establish "
        "accuracy improvement, production readiness, official Qwen LoRA support, "
        "full-manifest coverage, or H100 qualification."
    )

    lines = [
        "# CTO LoRA smoke summary",
        "",
        f"Status: {_inline(summary.get('status'))}",
        "",
        "## Objective",
        "",
        _inline(facts.get("objective") or "Qualify a memory-safe wrapper-LoRA training lifecycle on the laptop."),
        "",
        "## Environment",
        "",
        f"- Machine: {machine_text}",
        f"- Model: {model_text}",
        f"- Wrapper and dependencies: {dependency_text}",
        "",
        "## Dataset and actual consumption",
        "",
        f"- Supplied manifest: {_format_count(dataset.get('total_manifest_samples'))} samples, {_format_hours(dataset.get('total_manifest_duration_seconds'))}",
        f"- Eligible after filtering: {_format_count(dataset.get('eligible_samples'))} samples, {_format_hours(dataset.get('eligible_duration_seconds'))}",
        f"- Actually consumed: {_format_count(consumption.get('samples'))} sample uses ({_format_count(consumption.get('unique_samples'))} unique), {_format_seconds(consumption.get('audio_duration_seconds'))}",
        f"- Microsteps / optimizer steps: {_format_count(consumption.get('microsteps'))} / {_format_count(consumption.get('optimizer_steps'))}",
        f"- Complete eligible epoch: {_yes_no_pending(consumption.get('complete_epoch_performed'))}",
        f"- Estimated complete-epoch runtime: {_format_seconds(consumption.get('estimated_complete_epoch_runtime_seconds'), pending='PENDING — insufficient runtime evidence')}",
        "",
        "## LoRA and measured evidence",
        "",
        f"- Configuration: {lora_text}",
        f"- Trainable parameters: {_format_count(facts.get('trainable_parameters'), pending='PENDING — not measured')}",
        f"- Trainable percentage: {_format_ratio_percent(facts.get('trainable_percentage'))}",
        f"- Starting / ending loss: {_format_number(losses.get('initial'))} / {_format_number(losses.get('final'))}",
        f"- Peak CUDA allocated / reserved: {_format_bytes(memory.get('peak_cuda_allocated_bytes'))} / {_format_bytes(memory.get('peak_cuda_reserved_bytes'))}",
        f"- Peak system RAM used: {_format_bytes(memory.get('peak_system_used_bytes'))}",
        f"- Runtime: {_format_seconds(consumption.get('runtime_seconds'))}",
        f"- Adapter saved / fresh-reloaded: {_yes_no_pending(adapter.get('saved'))} / {_yes_no_pending(adapter.get('reloaded'))}",
        f"- Adapter path: {_inline(adapter.get('path'))}",
        f"- Example base prediction: {_inline(predictions.get('base'))}",
        f"- Example adapter prediction: {_inline(predictions.get('adapter'))}",
        "",
        "## Known limitations",
        "",
        *(f"- {_inline(item)}" for item in limitation_lines),
        "",
        "## Next step",
        "",
        _inline(
            facts.get("next_step")
            or "H100 full-parameter or longer LoRA qualification after reviewing this evidence."
        ),
        "",
    ]
    return "\n".join(lines)


def write_cto_smoke_summary(
    summary: Mapping[str, Any],
    facts: Mapping[str, Any] | None,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write ``CTO_SMOKE_SUMMARY.md`` content."""

    return _atomic_text(
        path,
        render_cto_smoke_summary(summary, facts),
        overwrite=overwrite,
    )


def render_training_run_readme(
    summary: Mapping[str, Any],
    facts: Mapping[str, Any] | None = None,
) -> str:
    """Render a factual per-run README, distinct from the concise CTO note.

    Missing measurements remain explicitly pending.  All supplied values pass
    through the report sanitizer before any text is rendered.
    """

    safe_summary = _mapping(
        sanitize_report_payload(summary),
        "summary",
    )
    safe_facts = _mapping(
        sanitize_report_payload({} if facts is None else facts),
        "facts",
    )
    dataset = _mapping(safe_summary.get("dataset"), "summary.dataset")
    consumption = _mapping(safe_summary.get("consumption"), "summary.consumption")
    claims_value = safe_summary.get("claims")
    claims = claims_value if isinstance(claims_value, Mapping) else {}
    training_value = safe_facts.get("training")
    training = training_value if isinstance(training_value, Mapping) else {}
    losses_value = safe_facts.get("losses")
    losses = losses_value if isinstance(losses_value, Mapping) else training
    memory_value = safe_facts.get("memory")
    memory = memory_value if isinstance(memory_value, Mapping) else training
    adapter_value = safe_facts.get("adapter")
    adapter = adapter_value if isinstance(adapter_value, Mapping) else {}
    predictions_value = safe_facts.get("predictions")
    predictions = predictions_value if isinstance(predictions_value, Mapping) else {}
    metrics_value = safe_facts.get("metrics")
    metrics = metrics_value if isinstance(metrics_value, Mapping) else {}
    gradients = safe_facts.get("gradient_norms")
    if gradients is None:
        gradients = safe_facts.get("gradients", training.get("gradient_norms"))
    initial_loss = losses.get("initial", losses.get("initial_loss"))
    final_loss = losses.get("final", losses.get("final_loss"))
    reloaded = (
        adapter.get("reloaded")
        if "reloaded" in adapter
        else adapter.get("fresh_process_reload")
    )

    lines = [
        "# Wrapper LoRA training run",
        "",
        f"Status: {_inline(safe_summary.get('status'))}",
        "",
        "## Dataset accounting",
        "",
        f"- Supplied manifest: {_format_count(dataset.get('total_manifest_samples'))} samples, {_format_seconds(dataset.get('total_manifest_duration_seconds'))} ({_format_hours(dataset.get('total_manifest_duration_seconds'))})",
        f"- Eligible after duration filtering: {_format_count(dataset.get('eligible_samples'))} samples, {_format_seconds(dataset.get('eligible_duration_seconds'))} ({_format_hours(dataset.get('eligible_duration_seconds'))})",
        f"- Selected by configured caps: {_format_count(dataset.get('selected_samples'))} samples, {_format_seconds(dataset.get('selected_duration_seconds'))} ({_format_hours(dataset.get('selected_duration_seconds'))})",
        f"- Actually consumed: {_format_count(consumption.get('samples'))} sample uses ({_format_count(consumption.get('unique_samples'))} unique), {_format_seconds(consumption.get('audio_duration_seconds'))} ({_format_hours(consumption.get('audio_duration_seconds'))})",
        f"- Complete eligible epoch performed: {_yes_no_pending(consumption.get('complete_epoch_performed'))}",
        f"- Eligible epoch fraction by unique samples: {_format_fraction(consumption.get('eligible_epoch_fraction_by_unique_samples'))}",
        f"- Estimated complete-epoch runtime: {_format_seconds(consumption.get('estimated_complete_epoch_runtime_seconds'), pending='PENDING — insufficient runtime evidence')}",
        "",
        "## Training evidence",
        "",
        f"- Microsteps / optimizer steps: {_format_count(consumption.get('microsteps'))} / {_format_count(consumption.get('optimizer_steps'))}",
        f"- Per-device batch / accumulation / effective batch: {_format_count(consumption.get('per_device_batch_size'))} / {_format_count(consumption.get('gradient_accumulation_steps'))} / {_format_count(consumption.get('effective_batch_size'))}",
        f"- Starting / ending loss: {_format_number(initial_loss)} / {_format_number(final_loss)}",
        f"- Gradient norms: {_format_evidence(gradients, pending='PENDING — not measured')}",
        f"- Wall runtime: {_format_seconds(consumption.get('runtime_seconds'))}",
        f"- Optimizer-loop runtime used for epoch estimate: {_format_seconds(consumption.get('optimizer_loop_runtime_seconds'))}",
        f"- Optimizer throughput (sample uses/s; audio s/s): {_format_number(consumption.get('optimizer_samples_per_second'))} / {_format_number(consumption.get('optimizer_audio_seconds_per_second'))}",
        "",
        "## Memory evidence",
        "",
        f"- Peak CUDA allocated: {_format_bytes(memory.get('peak_cuda_allocated_bytes'))}",
        f"- Peak CUDA reserved: {_format_bytes(memory.get('peak_cuda_reserved_bytes'))}",
        f"- Peak system RAM used: {_format_bytes(memory.get('peak_system_used_bytes'))}",
        "",
        "## Adapter and evaluation evidence",
        "",
        f"- Adapter saved: {_yes_no_pending(adapter.get('saved'))}",
        f"- Adapter fresh-process reload: {_yes_no_pending(reloaded)}",
        f"- Adapter path: {_inline(adapter.get('path'))}",
        f"- Base prediction: {_inline(predictions.get('base'))}",
        f"- Adapter prediction: {_inline(predictions.get('adapter'))}",
        f"- Evaluation metrics: {_format_evidence(metrics, pending='PENDING — not measured')}",
        "",
        "## Interpretation boundaries",
        "",
        f"- Full supplied manifest consumed: {_yes_no_pending(claims.get('full_input_manifest_consumed'))}",
        "- Supplied, eligible, and selected durations are context; only the explicitly reported consumed duration was used by this run.",
        "- A bounded smoke run does not establish accuracy improvement, production readiness, official Qwen LoRA support, or H100 qualification.",
        "- Base-versus-adapter predictions and metrics are evidence only when populated after explicit fresh-process verification.",
        "",
    ]
    return "\n".join(lines)


def write_training_run_readme(
    summary: Mapping[str, Any],
    facts: Mapping[str, Any] | None,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write a sanitized human-readable per-run ``README.md``."""

    return _atomic_text(
        path,
        render_training_run_readme(summary, facts),
        overwrite=overwrite,
    )


def sanitize_report_payload(value: object, *, _key: str | None = None) -> Any:
    """Create a JSON-safe copy, redacting credentials and rejecting binary audio."""

    if _key is not None and _key.lower() in _SENSITIVE_KEYS:
        return "[redacted]"
    if value is None or type(value) in (bool, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TrainingError("Reports must not contain NaN or Inf")
        return value
    if isinstance(value, str):
        return _sanitize_string(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise TrainingError("Reports must never contain audio or other binary bytes")
    if is_dataclass(value) and not isinstance(value, type):
        return sanitize_report_payload(asdict(value), _key=_key)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TrainingError("Report mappings must use string keys")
            normalized[key] = sanitize_report_payload(item, _key=key)
        return normalized
    if isinstance(value, (list, tuple)):
        return [sanitize_report_payload(item) for item in value]
    raise TrainingError(f"Report value is not JSON serializable: {type(value).__name__}")


def _configured_root(
    value: str | Path,
    *,
    project_root: Path,
    allowed_root: Path,
    label: str,
) -> Path:
    raw = str(value).strip()
    if not raw:
        raise TrainingError(f"Configured {label} must be a non-empty local path")
    parsed = urlsplit(raw)
    if parsed.scheme or parsed.netloc:
        raise TrainingError(f"Configured {label} must not be a URI")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    candidate = candidate.resolve()
    allowed = allowed_root.resolve()
    _require_descendant(candidate, allowed, f"configured {label}", allow_equal=True)
    return candidate


def _require_descendant(
    path: Path,
    root: Path,
    label: str,
    *,
    allow_equal: bool = False,
) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise TrainingError(f"{label} must remain inside {root}; received {path}") from exc
    if not allow_equal and relative == Path("."):
        raise TrainingError(f"{label} must be below, not equal to, {root}")


def _local_destination(path: str | Path) -> Path:
    raw = str(path).strip()
    parsed = urlsplit(raw)
    if not raw or parsed.scheme or parsed.netloc:
        raise TrainingError("Training reports require an explicit local output path")
    return Path(raw).expanduser().resolve()


def _atomic_text(path: str | Path, text: str, *, overwrite: bool) -> Path:
    destination = _local_destination(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise TrainingError(f"Refusing to overwrite existing report: {destination}")
    return _atomic_stream(destination, (text,), overwrite=overwrite)


def _atomic_stream(
    destination: Path,
    chunks: Iterable[str],
    *,
    overwrite: bool,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise TrainingError(f"Refusing to overwrite existing report: {destination}")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            for chunk in chunks:
                temporary.write(chunk)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        _fsync_directory(destination.parent)
    except (OSError, TypeError, ValueError) as exc:
        raise TrainingError(f"Could not write training report {destination}: {exc}") from exc
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return destination


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _sanitize_string(value: str) -> str:
    sanitized = _TOKEN.sub("[redacted-token]", value)
    stripped = sanitized.strip()
    parsed = urlsplit(stripped)
    if parsed.scheme and parsed.netloc and (parsed.query or parsed.fragment):
        sanitized = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    return sanitized


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return value


def _estimate_epoch_runtime(
    *,
    runtime_seconds: float | None,
    consumed_audio_seconds: float,
    eligible_duration_seconds: float,
    unique_consumed_samples: int,
    eligible_samples: int,
) -> float | None:
    if runtime_seconds is None or runtime_seconds <= 0:
        return None
    if consumed_audio_seconds > 0 and eligible_duration_seconds > 0:
        return runtime_seconds * eligible_duration_seconds / consumed_audio_seconds
    if unique_consumed_samples > 0 and eligible_samples > 0:
        return runtime_seconds * eligible_samples / unique_consumed_samples
    return None


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainingError(f"{label} must be a mapping")
    return value


def _mapping_inline(value: Mapping[object, object]) -> str:
    if not value:
        return _pending("not captured")
    return ", ".join(
        f"{_inline(key)}={_inline(item)}"
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
    )


def _format_evidence(value: object, *, pending: str) -> str:
    if value is None:
        return pending
    if isinstance(value, Mapping):
        return _mapping_inline(value) if value else pending
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return pending
        return ", ".join(
            _format_number(item)
            if type(item) in (int, float)
            else _inline(item)
            for item in value
        )
    return _inline(value)


def _inline(value: object) -> str:
    if value is None or value == "":
        return _pending("not captured")
    if isinstance(value, bool):
        return "yes" if value else "no"
    return _sanitize_string(" ".join(str(value).split()))[:1000]


def _pending(reason: str) -> str:
    return f"PENDING — {reason}"


def _format_count(value: object, *, pending: str = "PENDING — not recorded") -> str:
    if type(value) is not int:
        return pending
    return f"{value:,}"


def _format_number(value: object) -> str:
    if type(value) not in (int, float) or not math.isfinite(value):
        return _pending("not measured")
    return f"{float(value):.6g}"


def _format_hours(seconds: object) -> str:
    if type(seconds) not in (int, float) or not math.isfinite(seconds):
        return _pending("duration not recorded")
    return f"{float(seconds) / 3600:.4f} hours"


def _format_seconds(
    seconds: object,
    *,
    pending: str = "PENDING — not measured",
) -> str:
    if type(seconds) not in (int, float) or not math.isfinite(seconds):
        return pending
    return f"{float(seconds):.3f} seconds"


def _format_bytes(value: object) -> str:
    if type(value) is not int or value < 0:
        return _pending("not measured")
    return f"{value / (1024**3):.3f} GiB ({value:,} bytes)"


def _format_ratio_percent(value: object) -> str:
    if type(value) not in (int, float) or not math.isfinite(value):
        return _pending("not measured")
    return f"{float(value):.6f}%"


def _format_fraction(value: object) -> str:
    if type(value) not in (int, float) or not math.isfinite(value):
        return _pending("not measured")
    return f"{float(value):.6f}"


def _yes_no_pending(value: object) -> str:
    if type(value) is bool:
        return "yes" if value else "no"
    return _pending("not verified")


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_number(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


__all__ = [
    "TrainingRunDirectories",
    "append_atomic_jsonl",
    "build_training_summary",
    "render_cto_smoke_summary",
    "render_training_run_readme",
    "resolve_training_run_directories",
    "sanitize_report_payload",
    "validate_run_name",
    "write_atomic_csv",
    "write_atomic_json",
    "write_atomic_jsonl",
    "write_cto_smoke_summary",
    "write_json_atomic",
    "write_selected_sample_ids",
    "write_training_run_readme",
]
