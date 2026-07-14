"""Incremental, one-model baseline evaluation for local manifest audio."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from ..audio import DecodedAudio, decode_audio
from ..config import ProjectConfig
from ..data.manifest import iter_manifest, manifest_fingerprint, write_json_atomic
from ..data.schema import ManifestRecord, display_audio_locator, resolve_local_audio_path
from ..exceptions import EvaluationError
from .metrics import aggregate_predictions, compute_text_metrics
from .normalization import NormalizationOptions, is_blank, is_punctuation_only

_SAFE_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")


@dataclass(frozen=True, slots=True)
class BaselineOptions:
    """Optional CLI adjustments to the strict profile baseline policy."""

    run_name: str
    device: str | None = None
    max_samples: int | None = None
    max_duration_seconds: float | None = None
    resume: bool | None = None
    overwrite: bool | None = None
    error_policy: str | None = None
    offline: bool | None = None

    def validate(self) -> None:
        if not _SAFE_RUN_NAME.fullmatch(self.run_name):
            raise EvaluationError(
                "Baseline run name must use 1-80 letters, numbers, dots, underscores, "
                "or hyphens and cannot start with punctuation"
            )
        if self.device is not None and self.device not in {"auto", "cpu", "cuda"}:
            raise EvaluationError("Baseline device must be auto, cpu, or cuda")
        if self.max_samples is not None and (
            type(self.max_samples) is not int or self.max_samples <= 0
        ):
            raise EvaluationError("Baseline max_samples must be a positive integer")
        if self.max_duration_seconds is not None and (
            type(self.max_duration_seconds) not in (int, float)
            or self.max_duration_seconds <= 0
        ):
            raise EvaluationError("Baseline max_duration_seconds must be a positive number")
        if self.error_policy is not None and self.error_policy not in {"continue", "stop"}:
            raise EvaluationError("Baseline error policy must be continue or stop")
        if self.resume is True and self.overwrite is True:
            raise EvaluationError("Baseline resume and overwrite cannot both be enabled")


@dataclass(frozen=True, slots=True)
class BaselineRunResult:
    run_directory: Path
    status: str
    stopped_reason: str | None
    summary: dict[str, Any]
    metrics: dict[str, Any]

    @property
    def exit_code(self) -> int:
        return 1 if self.status == "stopped" else 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "stopped_reason": self.stopped_reason,
            "run_directory": str(self.run_directory),
            "summary": self.summary,
            "metrics": self.metrics,
        }


def run_baseline(
    manifest: str | Path,
    config: ProjectConfig,
    *,
    options: BaselineOptions,
    engine_factory: Callable[..., Any] | None = None,
    audio_decoder: Callable[[str | Path], DecodedAudio] = decode_audio,
) -> BaselineRunResult:
    """Evaluate a local manifest once, persisting every completed row immediately.

    The implementation deliberately has no downloader.  Remote Azure/blob rows are
    persisted as per-sample failures under the configured ``continue`` policy.
    """

    options.validate()
    values = config.as_dict()
    _require_single_process(values)
    policy = values["evaluation"]["baseline"]
    resolved = _resolved_options(options, values)
    source = Path(manifest).expanduser().resolve()
    fingerprint = manifest_fingerprint(source)
    run_directory = _run_directory(config, options.run_name)
    existing_rows = _prepare_run_directory(
        run_directory,
        resume=resolved["resume"],
        overwrite=resolved["overwrite"],
        manifest=source,
        manifest_fingerprint=fingerprint,
    )
    normalization = NormalizationOptions(
        remove_punctuation=bool(policy["remove_punctuation"]),
        lowercase_latin=bool(policy["lowercase_latin"]),
    )
    run_config = _run_config_payload(
        config,
        source,
        fingerprint,
        options,
        resolved,
        normalization,
    )
    config_path = run_directory / "run_config.json"
    if not config_path.exists():
        write_json_atomic(run_config, config_path)

    prediction_path = run_directory / "predictions.jsonl"
    failure_path = run_directory / "failures.jsonl"
    completed_ids = {str(row.get("sample_id")) for row in existing_rows if row.get("sample_id")}
    rows = list(existing_rows)
    skipped_resume = len(existing_rows)
    skipped_duration_limit = 0
    accumulated_duration = sum(
        float(row.get("audio_duration_seconds") or 0.0)
        for row in rows
        if row.get("status") == "success"
    )
    engine: Any = None
    stopped_reason = _collapse_reason(rows, resolved)

    try:
        with prediction_path.open("a", encoding="utf-8") as predictions, failure_path.open(
            "a", encoding="utf-8"
        ) as failures:
            for record in iter_manifest(source):
                sample_id = _sample_id(fingerprint, record)
                if sample_id in completed_ids:
                    continue
                if resolved["max_samples"] is not None and len(rows) >= resolved["max_samples"]:
                    break
                if stopped_reason is not None:
                    break

                row, failure = _evaluate_record(
                    record,
                    sample_id=sample_id,
                    project_root=config.project_root,
                    resolved=resolved,
                    normalization=normalization,
                    engine=engine,
                    engine_factory=engine_factory,
                    audio_decoder=audio_decoder,
                )
                engine = row.pop("_engine", engine)

                if row.get("status") == "success":
                    duration = float(row["audio_duration_seconds"])
                    if (
                        resolved["max_duration_seconds"] is not None
                        and accumulated_duration + duration > resolved["max_duration_seconds"]
                    ):
                        skipped_duration_limit += 1
                        continue

                _append_jsonl(predictions, row)
                rows.append(row)
                completed_ids.add(sample_id)
                if row.get("status") == "success":
                    accumulated_duration += float(row["audio_duration_seconds"])
                if failure:
                    _append_jsonl(failures, row)
                    if resolved["error_policy"] == "stop":
                        stopped_reason = "error_policy_stop_after_sample_failure"
                        break
                stopped_reason = _collapse_reason(rows, resolved)
                if stopped_reason is not None:
                    break
    finally:
        if engine is not None:
            _close_engine(engine)

    metrics = aggregate_predictions(rows)
    summary = {
        "manifest": str(source),
        "manifest_fingerprint": fingerprint,
        "status": "stopped" if stopped_reason else "completed",
        "stopped_reason": stopped_reason,
        "attempted_samples": len(rows),
        "resumed_samples": skipped_resume,
        "new_samples": len(rows) - skipped_resume,
        "skipped_for_duration_limit": skipped_duration_limit,
        "configured_max_samples": resolved["max_samples"],
        "configured_max_duration_seconds": resolved["max_duration_seconds"],
        "error_policy": resolved["error_policy"],
    }
    metrics = {**metrics, "normalization": normalization.as_dict(), "run_status": summary["status"]}
    _write_final_reports(run_directory, summary, metrics, rows, int(policy["worst_example_count"]))
    return BaselineRunResult(
        run_directory=run_directory,
        status=summary["status"],
        stopped_reason=stopped_reason,
        summary=summary,
        metrics=metrics,
    )


def _resolved_options(options: BaselineOptions, values: dict[str, Any]) -> dict[str, Any]:
    policy = values["evaluation"]["baseline"]
    max_duration = options.max_duration_seconds
    if max_duration is None and policy["max_audio_hours"] is not None:
        max_duration = float(policy["max_audio_hours"]) * 3600
    resolved = {
        "device": options.device or values["inference"]["device"],
        "precision": values["inference"]["precision"],
        "cache_dir": values["paths"]["model_cache_dir"],
        "offline": bool(values["inference"]["offline"]) if options.offline is None else options.offline,
        "language": values["inference"]["language_hint"],
        "max_new_tokens": values["inference"]["max_new_tokens"],
        "max_samples": options.max_samples if options.max_samples is not None else policy["max_samples"],
        "max_duration_seconds": max_duration,
        "resume": bool(policy["resume"]) if options.resume is None else options.resume,
        "overwrite": bool(policy["overwrite"]) if options.overwrite is None else options.overwrite,
        "error_policy": options.error_policy or policy["error_policy"],
        "early_check_samples": int(policy["early_check_samples"]),
        "blank_output_stop_threshold": int(policy["blank_output_stop_threshold"]),
        "punctuation_only_stop_threshold": int(policy["punctuation_only_stop_threshold"]),
        "identical_prediction_stop_threshold": int(policy["identical_prediction_stop_threshold"]),
    }
    if resolved["resume"] and resolved["overwrite"]:
        raise EvaluationError("Baseline resume and overwrite cannot both be enabled")
    return resolved


def _evaluate_record(
    record: ManifestRecord,
    *,
    sample_id: str,
    project_root: Path,
    resolved: dict[str, Any],
    normalization: NormalizationOptions,
    engine: Any,
    engine_factory: Callable[..., Any] | None,
    audio_decoder: Callable[[str | Path], DecodedAudio],
) -> tuple[dict[str, Any], bool]:
    base = _base_row(record, sample_id)
    try:
        local_path = resolve_local_audio_path(record, project_root)
        if local_path is None:
            raise EvaluationError("Remote audio is structurally supported but cannot be evaluated without a local WAV or FLAC file")
        audio = audio_decoder(local_path)
        active_engine = engine
        if active_engine is None:
            if engine_factory is None:
                from ..models.qwen3_asr import Qwen3ASREngine

                engine_factory = Qwen3ASREngine
            active_engine = engine_factory(
                device=resolved["device"],
                precision=resolved["precision"],
                cache_dir=resolved["cache_dir"],
                offline=resolved["offline"],
                language=resolved["language"],
                max_new_tokens=resolved["max_new_tokens"],
            )
        # Preserve ownership even if inference itself raises, so the outer
        # lifecycle always releases a partially-loaded CUDA model.
        engine = active_engine
        result = active_engine.transcribe(audio)
        payload = result.as_dict() if hasattr(result, "as_dict") else dict(result)
        if payload.get("status") != "success":
            raise EvaluationError(str(payload.get("error") or "Model returned a non-success inference result"))
        transcript = str(payload.get("transcript") or "")
        text_metrics = compute_text_metrics(record.text, transcript, options=normalization)
        timing = payload.get("timing") or {}
        return (
            {
                **base,
                "status": "success",
                "error": None,
                "audio_duration_seconds": float(audio.duration_seconds),
                "transcript": transcript,
                "language": payload.get("language"),
                "device": payload.get("device"),
                "precision": payload.get("precision"),
                "model": payload.get("model"),
                "load_seconds": timing.get("load_seconds"),
                "inference_seconds": timing.get("inference_seconds"),
                "real_time_factor": timing.get("real_time_factor"),
                "peak_cuda_memory_bytes": payload.get("peak_cuda_memory_bytes"),
                "warnings": payload.get("warnings") or [],
                **text_metrics,
                "_engine": active_engine,
            },
            False,
        )
    except Exception as exc:
        return ({**base, "status": "failure", "error": _sanitize_error(exc), "_engine": engine}, True)


def _base_row(record: ManifestRecord, sample_id: str) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "manifest_line": record.line_number,
        "audio_filepath": display_audio_locator(record.audio_filepath),
        "reference": record.text,
        "declared_duration_seconds": record.duration,
        "language_hint": record.language,
        "source": record.source,
        "speaker_id": record.speaker_id,
        "recording_id": record.recording_id,
        "domain": record.domain,
        "split": record.split,
    }


def _collapse_reason(rows: Iterable[dict[str, Any]], resolved: dict[str, Any]) -> str | None:
    successful = [row for row in rows if row.get("status") == "success"]
    count = int(resolved["early_check_samples"])
    if len(successful) < count:
        return None
    early = successful[:count]
    blank = sum(bool(row.get("blank_prediction")) for row in early)
    punctuation = sum(bool(row.get("punctuation_only_prediction")) for row in early)
    predictions = [str(row.get("transcript") or "") for row in early]
    identical = max(Counter(predictions).values(), default=0)
    if blank >= resolved["blank_output_stop_threshold"]:
        return "early_collapse_blank_predictions"
    if punctuation >= resolved["punctuation_only_stop_threshold"]:
        return "early_collapse_punctuation_only_predictions"
    if identical >= resolved["identical_prediction_stop_threshold"]:
        return "early_collapse_identical_predictions"
    return None


def _prepare_run_directory(
    run_directory: Path,
    *,
    resume: bool,
    overwrite: bool,
    manifest: Path,
    manifest_fingerprint: str,
) -> list[dict[str, Any]]:
    if run_directory.exists() and overwrite:
        shutil.rmtree(run_directory)
    if resume:
        if not run_directory.is_dir():
            raise EvaluationError(f"Cannot resume because baseline report directory does not exist: {run_directory}")
        config_path = run_directory / "run_config.json"
        try:
            previous = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvaluationError(f"Cannot resume without a readable run_config.json: {exc}") from exc
        if previous.get("manifest_fingerprint") != manifest_fingerprint or previous.get("manifest") != str(manifest):
            raise EvaluationError("Cannot resume: manifest path or fingerprint differs from the existing baseline run")
        return _read_jsonl(run_directory / "predictions.jsonl")
    if run_directory.exists() and any(run_directory.iterdir()):
        raise EvaluationError(f"Baseline report directory already exists; use --resume or --overwrite: {run_directory}")
    run_directory.mkdir(parents=True, exist_ok=True)
    return []


def _run_directory(config: ProjectConfig, run_name: str) -> Path:
    reports_dir = Path(config.as_dict()["paths"]["reports_dir"]).resolve()
    base = (reports_dir / "evaluation").resolve()
    destination = (base / run_name).resolve()
    try:
        destination.relative_to(base)
    except ValueError as exc:
        raise EvaluationError("Baseline run name resolved outside reports/evaluation") from exc
    return destination


def _run_config_payload(
    config: ProjectConfig,
    manifest: Path,
    fingerprint: str,
    options: BaselineOptions,
    resolved: dict[str, Any],
    normalization: NormalizationOptions,
) -> dict[str, Any]:
    return {
        "manifest": str(manifest),
        "manifest_fingerprint": fingerprint,
        "profile_config": config.as_dict(),
        "cli_options": {
            "run_name": options.run_name,
            "device": options.device,
            "max_samples": options.max_samples,
            "max_duration_seconds": options.max_duration_seconds,
            "resume": options.resume,
            "overwrite": options.overwrite,
            "error_policy": options.error_policy,
            "offline": options.offline,
        },
        "resolved_options": resolved,
        "normalization": normalization.as_dict(),
    }


def _write_final_reports(
    run_directory: Path,
    summary: dict[str, Any],
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
    worst_count: int,
) -> None:
    write_json_atomic(summary, run_directory / "summary.json", overwrite=True)
    write_json_atomic(metrics, run_directory / "metrics.json", overwrite=True)
    worst = _worst_examples(rows, worst_count)
    write_json_atomic(worst, run_directory / "worst_examples.json", overwrite=True)
    _write_metrics_csv(metrics, run_directory / "metrics.csv")
    _write_readme(run_directory, summary, metrics)


def _worst_examples(rows: list[dict[str, Any]], count: int) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "success"]
    failures = [row for row in rows if row.get("status") != "success"]
    successful.sort(
        key=lambda row: (
            -1.0 if row.get("wer") is None else -float(row["wer"]),
            -1.0 if row.get("cer") is None else -float(row["cer"]),
            int(row.get("manifest_line") or 0),
        )
    )
    return {"failures": failures[:count], "worst_successful_examples": successful[:count]}


def _write_metrics_csv(metrics: dict[str, Any], destination: Path) -> None:
    flattened = {
        "samples": metrics["samples"],
        "successful_samples": metrics["successful_samples"],
        "failed_samples": metrics["failed_samples"],
        "failure_rate": metrics["failure_rate"],
        "exact_match_rate": metrics["exact_match_rate"],
        "blank_prediction_rate": metrics["blank_prediction_rate"],
        "punctuation_only_prediction_rate": metrics["punctuation_only_prediction_rate"],
        "normalized_wer": metrics["normalized"]["wer"],
        "normalized_cer": metrics["normalized"]["cer"],
        "raw_wer": metrics["raw"]["wer"],
        "raw_cer": metrics["raw"]["cer"],
        "total_audio_duration_seconds": metrics["total_audio_duration_seconds"],
        "total_inference_seconds": metrics["total_inference_seconds"],
        "average_real_time_factor": metrics["average_real_time_factor"],
    }
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened))
        writer.writeheader()
        writer.writerow(flattened)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _write_readme(run_directory: Path, summary: dict[str, Any], metrics: dict[str, Any]) -> None:
    text = (
        "# Base evaluation report\n\n"
        f"Status: **{summary['status']}**. This report contains references, predictions, "
        "and source paths, but never audio bytes, credentials, access tokens, or environment secrets.\n\n"
        "- `run_config.json` — pinned profile and selected evaluation options\n"
        "- `predictions.jsonl` — incrementally persisted sample results\n"
        "- `failures.jsonl` — per-sample failures under the configured policy\n"
        "- `metrics.json` / `metrics.csv` — decimal WER/CER and aggregate timing\n"
        "- `worst_examples.json` — bounded error examples\n\n"
        f"Completed samples: {metrics['samples']}; successful samples: {metrics['successful_samples']}.\n"
    )
    temporary = run_directory / ".README.md.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, run_directory / "README.md")


def _append_jsonl(handle: Any, row: dict[str, Any]) -> None:
    payload = {key: value for key, value in row.items() if key != "_engine"}
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict) or not value.get("sample_id"):
                    raise ValueError("row is not an object with sample_id")
                rows.append(value)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Could not resume from {path}: {exc}") from exc
    return rows


def _sample_id(manifest_fingerprint: str, record: ManifestRecord) -> str:
    canonical = f"{manifest_fingerprint}\0{record.line_number}\0{record.audio_filepath}\0{record.text}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _close_engine(engine: Any) -> None:
    close = getattr(engine, "close", None)
    if callable(close):
        close()


def _require_single_process(values: dict[str, Any]) -> None:
    hardware = values["hardware"]
    if hardware["distributed"] or hardware["process_count"] != 1:
        raise EvaluationError("Baseline evaluation is single-process only in this milestone")


def _sanitize_error(error: BaseException) -> str:
    from ..models.qwen3_asr import sanitize_error

    return sanitize_error(error)
