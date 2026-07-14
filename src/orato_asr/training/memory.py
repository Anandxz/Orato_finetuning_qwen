"""Lazy memory observation and hard guards for wrapper training.

This module deliberately accepts a Torch-like object from the training entry
point.  Importing it never imports PyTorch, starts CUDA, or runs ``nvidia-smi``.
The optional CUDA-process detector is likewise injected so unit tests and
platform-specific launchers can supply a bounded, read-only implementation.
"""

from __future__ import annotations

import gc
import math
import os
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orato_asr.exceptions import MemorySafetyError, TrainingOOMError

BYTES_PER_GIB = 1024**3
DEFAULT_LARGE_CUDA_PROCESS_BYTES = 512 * 1024**2

MemoryInfoReader = Callable[[], str]
CudaProcessDetector = Callable[[int, int], Iterable[Mapping[str, Any]]]


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """One JSON-serializable memory observation at a named training stage."""

    stage: str
    captured_at_utc: str
    system_ram: Mapping[str, Any]
    cuda: Mapping[str, Any]
    cuda_process_check: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "captured_at_utc": self.captured_at_utc,
            "system_ram": dict(self.system_ram),
            "cuda": dict(self.cuda),
            "cuda_process_check": {
                **self.cuda_process_check,
                "other_processes": [
                    dict(item)
                    for item in self.cuda_process_check.get("other_processes", ())
                ],
            },
        }


@dataclass(frozen=True, slots=True)
class MemoryGuardConfig:
    """Thresholds checked immediately before a major training stage."""

    minimum_available_system_bytes: int
    gpu_safety_limit_bytes: int
    abort_on_threshold: bool = True
    reject_other_large_cuda_processes: bool = True
    large_cuda_process_bytes: int = DEFAULT_LARGE_CUDA_PROCESS_BYTES
    require_cuda_process_check: bool = False

    def __post_init__(self) -> None:
        _nonnegative_integer(
            self.minimum_available_system_bytes,
            "minimum_available_system_bytes",
        )
        _positive_integer(self.gpu_safety_limit_bytes, "gpu_safety_limit_bytes")
        _positive_integer(self.large_cuda_process_bytes, "large_cuda_process_bytes")
        for field_name in (
            "abort_on_threshold",
            "reject_other_large_cuda_processes",
            "require_cuda_process_check",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise ValueError(f"{field_name} must be true or false")


@dataclass(frozen=True, slots=True)
class MemoryGuardResult:
    """Structured guard outcome suitable for a memory-events report."""

    safe: bool
    stage: str
    violations: tuple[Mapping[str, Any], ...]
    snapshot: MemorySnapshot

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "safe" if self.safe else "blocked",
            "stage": self.stage,
            "violations": [dict(item) for item in self.violations],
            "snapshot": self.snapshot.as_dict(),
        }


class MemoryGuardError(MemorySafetyError):
    """Memory-safety stop carrying a machine-readable guard result."""

    def __init__(self, result: MemoryGuardResult) -> None:
        self.result = result
        self.metadata = result.as_dict()
        codes = ", ".join(str(item["code"]) for item in result.violations)
        super().__init__(
            f"Memory guard blocked stage {result.stage!r}: {codes}. "
            "No CPU or disk fallback was attempted."
        )


class StructuredTrainingOOMError(TrainingOOMError):
    """CUDA OOM normalized with reportable failure metadata."""

    def __init__(self, message: str, metadata: Mapping[str, Any]) -> None:
        self.metadata = dict(metadata)
        super().__init__(message)


def gib_to_bytes(value: int | float) -> int:
    """Convert a finite positive GiB value to bytes without accepting booleans."""

    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("GiB value must be a finite positive number")
    return int(value * BYTES_PER_GIB)


def capture_memory_snapshot(
    stage: str,
    *,
    torch_module: Any | None = None,
    device_index: int = 0,
    capture_system_ram: bool = True,
    meminfo_reader: MemoryInfoReader | None = None,
    psutil_module: Any | None = None,
    cuda_process_detector: CudaProcessDetector | None = None,
    current_pid: int | None = None,
    clock: Callable[[], datetime] | None = None,
) -> MemorySnapshot:
    """Capture RAM, CUDA allocator, and optional peer-process measurements.

    Linux ``/proc/meminfo`` is the dependency-free default.  A caller may
    inject psutil for a non-Linux platform, but this module never imports it.
    The detector receives ``(device_index, current_pid)`` and must return only
    lightweight process metadata; process memory is not inspected here.
    """

    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("stage must be a non-empty string")
    if type(device_index) is not int or device_index < 0:
        raise ValueError("device_index must be a non-negative integer")
    if type(capture_system_ram) is not bool:
        raise ValueError("capture_system_ram must be true or false")

    captured = (clock or (lambda: datetime.now(timezone.utc)))()
    if captured.tzinfo is None:
        captured = captured.replace(tzinfo=timezone.utc)
    timestamp = captured.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    system = _capture_system_ram(
        enabled=capture_system_ram,
        meminfo_reader=meminfo_reader,
        psutil_module=psutil_module,
    )
    cuda = _capture_cuda(torch_module, device_index)
    process_check = _capture_cuda_processes(
        cuda_process_detector,
        device_index=device_index,
        current_pid=os.getpid() if current_pid is None else current_pid,
    )
    return MemorySnapshot(
        stage=stage.strip(),
        captured_at_utc=timestamp,
        system_ram=system,
        cuda=cuda,
        cuda_process_check=process_check,
    )


def evaluate_memory_guard(
    snapshot: MemorySnapshot,
    config: MemoryGuardConfig,
) -> MemoryGuardResult:
    """Evaluate hard thresholds without changing CUDA or process state."""

    violations: list[Mapping[str, Any]] = []
    system = snapshot.system_ram
    available = system.get("available_bytes")
    if config.minimum_available_system_bytes:
        if type(available) is not int:
            violations.append(
                {
                    "code": "system_ram_measurement_unavailable",
                    "required_available_bytes": config.minimum_available_system_bytes,
                }
            )
        elif available < config.minimum_available_system_bytes:
            violations.append(
                {
                    "code": "system_ram_below_minimum",
                    "available_bytes": available,
                    "required_available_bytes": config.minimum_available_system_bytes,
                }
            )

    cuda = snapshot.cuda
    if cuda.get("status") != "available":
        violations.append(
            {
                "code": "cuda_memory_measurement_unavailable",
                "cuda_status": cuda.get("status", "unknown"),
                "gpu_safety_limit_bytes": config.gpu_safety_limit_bytes,
            }
        )
    else:
        allocated = _int_or_zero(cuda.get("allocated_bytes"))
        reserved = _int_or_zero(cuda.get("reserved_bytes"))
        observed = max(allocated, reserved)
        if observed >= config.gpu_safety_limit_bytes:
            violations.append(
                {
                    "code": "gpu_memory_at_or_above_safety_limit",
                    "allocated_bytes": allocated,
                    "reserved_bytes": reserved,
                    "observed_bytes": observed,
                    "gpu_safety_limit_bytes": config.gpu_safety_limit_bytes,
                }
            )

    process_check = snapshot.cuda_process_check
    check_status = process_check.get("status")
    if config.require_cuda_process_check and check_status != "checked":
        violations.append(
            {
                "code": "cuda_process_check_unavailable",
                "process_check_status": check_status or "unknown",
            }
        )
    if config.reject_other_large_cuda_processes and check_status == "checked":
        large_processes = [
            item
            for item in process_check.get("other_processes", ())
            if _int_or_zero(item.get("used_memory_bytes"))
            >= config.large_cuda_process_bytes
        ]
        if large_processes:
            violations.append(
                {
                    "code": "other_large_cuda_process_active",
                    "minimum_large_process_bytes": config.large_cuda_process_bytes,
                    "processes": [dict(item) for item in large_processes],
                }
            )

    return MemoryGuardResult(
        safe=not violations,
        stage=snapshot.stage,
        violations=tuple(violations),
        snapshot=snapshot,
    )


def enforce_memory_guard(
    snapshot: MemorySnapshot,
    config: MemoryGuardConfig,
) -> MemoryGuardResult:
    """Return a safe result or stop before the stage when configured to abort."""

    result = evaluate_memory_guard(snapshot, config)
    if not result.safe and config.abort_on_threshold:
        raise MemoryGuardError(result)
    return result


def reset_cuda_memory_tracking(torch_module: Any) -> None:
    """Collect Python garbage, clear unused CUDA cache, and reset peak counters."""

    gc.collect()
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not bool(cuda.is_available()):
        raise MemorySafetyError("CUDA must be available before wrapper training")
    cuda.empty_cache()
    cuda.reset_peak_memory_stats()


def is_cuda_oom(error: BaseException, torch_module: Any | None = None) -> bool:
    """Recognize a CUDA OOM without importing Torch."""

    candidates: list[type[BaseException]] = []
    if torch_module is not None:
        for owner in (torch_module, getattr(torch_module, "cuda", None)):
            candidate = getattr(owner, "OutOfMemoryError", None)
            if isinstance(candidate, type) and issubclass(candidate, BaseException):
                candidates.append(candidate)
    if candidates and isinstance(error, tuple(candidates)):
        return True
    message = str(error).lower()
    return "cuda" in message and "out of memory" in message


def build_failure_metadata(
    error: BaseException,
    *,
    stage: str,
    snapshot: MemorySnapshot | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Normalize an expected training failure without leaking URLs or tokens."""

    oom = is_cuda_oom(error, torch_module)
    if isinstance(error, MemoryGuardError):
        category = "memory_safety_guard"
    elif oom:
        category = "cuda_out_of_memory"
    else:
        category = "training_failure"
    payload: dict[str, Any] = {
        "status": "failed",
        "stage": stage,
        "category": category,
        "exception_type": type(error).__name__,
        "message": _sanitize_error(error),
        "cuda_oom": oom,
        "cpu_fallback_attempted": False,
        "requires_fresh_process_before_retry": oom,
    }
    if snapshot is not None:
        payload["memory"] = snapshot.as_dict()
    if isinstance(error, MemoryGuardError):
        payload["guard"] = error.metadata
    return payload


def convert_cuda_oom(
    error: BaseException,
    *,
    stage: str,
    snapshot: MemorySnapshot | None = None,
    torch_module: Any | None = None,
) -> StructuredTrainingOOMError:
    """Convert only a verified CUDA OOM into the project's training exception."""

    if not is_cuda_oom(error, torch_module):
        raise ValueError("convert_cuda_oom received an error that is not a CUDA OOM")
    metadata = build_failure_metadata(
        error,
        stage=stage,
        snapshot=snapshot,
        torch_module=torch_module,
    )
    return StructuredTrainingOOMError(
        f"CUDA ran out of memory during {stage}; no CPU fallback was attempted. "
        "Release the process before applying one documented fallback.",
        metadata,
    )


def _capture_system_ram(
    *,
    enabled: bool,
    meminfo_reader: MemoryInfoReader | None,
    psutil_module: Any | None,
) -> dict[str, Any]:
    if not enabled:
        return {"status": "not_checked", "source": None}
    if psutil_module is not None:
        try:
            memory = psutil_module.virtual_memory()
            total = int(memory.total)
            available = int(memory.available)
            return _system_payload(total, available, "psutil")
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return {
                "status": "unavailable",
                "source": "psutil",
                "error": _sanitize_error(exc),
            }

    reader = meminfo_reader or (
        lambda: Path("/proc/meminfo").read_text(encoding="utf-8")
    )
    try:
        values = _parse_meminfo(reader())
        total = values["MemTotal"]
        available = values.get("MemAvailable")
        if available is None:
            available = (
                values.get("MemFree", 0)
                + values.get("Buffers", 0)
                + values.get("Cached", 0)
                + values.get("SReclaimable", 0)
                - values.get("Shmem", 0)
            )
        return _system_payload(total, available, "/proc/meminfo")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return {
            "status": "unavailable",
            "source": "/proc/meminfo",
            "error": _sanitize_error(exc),
        }


def _parse_meminfo(contents: str) -> dict[str, int]:
    if not isinstance(contents, str):
        raise TypeError("meminfo reader must return text")
    values: dict[str, int] = {}
    for line in contents.splitlines():
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        parts = raw.split()
        if not parts:
            continue
        value = int(parts[0])
        unit = parts[1].lower() if len(parts) > 1 else "bytes"
        if unit == "kb":
            value *= 1024
        elif unit not in {"b", "byte", "bytes"}:
            raise ValueError(f"Unsupported /proc/meminfo unit {unit!r}")
        values[name] = value
    if "MemTotal" not in values:
        raise ValueError("/proc/meminfo does not contain MemTotal")
    return values


def _system_payload(total: int, available: int, source: str) -> dict[str, Any]:
    if total <= 0 or available < 0 or available > total:
        raise ValueError("System memory counters are inconsistent")
    used = total - available
    return {
        "status": "available",
        "source": source,
        "total_bytes": total,
        "available_bytes": available,
        "used_bytes": used,
        "used_ratio": used / total,
    }


def _capture_cuda(torch_module: Any | None, device_index: int) -> dict[str, Any]:
    if torch_module is None:
        return {"status": "not_checked", "available": None, "device_index": device_index}
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        return {"status": "unavailable", "available": False, "device_index": device_index}
    try:
        if not bool(cuda.is_available()):
            return {
                "status": "unavailable",
                "available": False,
                "device_index": device_index,
            }
        return {
            "status": "available",
            "available": True,
            "device_index": device_index,
            "allocated_bytes": int(cuda.memory_allocated(device_index)),
            "reserved_bytes": int(cuda.memory_reserved(device_index)),
            "peak_allocated_bytes": int(cuda.max_memory_allocated(device_index)),
            "peak_reserved_bytes": int(cuda.max_memory_reserved(device_index)),
        }
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "status": "error",
            "available": None,
            "device_index": device_index,
            "error": _sanitize_error(exc),
        }


def _capture_cuda_processes(
    detector: CudaProcessDetector | None,
    *,
    device_index: int,
    current_pid: int,
) -> dict[str, Any]:
    if detector is None:
        return {"status": "not_checked", "other_processes": []}
    try:
        processes: list[dict[str, Any]] = []
        for raw in detector(device_index, current_pid):
            if not isinstance(raw, Mapping):
                raise TypeError("CUDA process detector rows must be mappings")
            pid = raw.get("pid")
            used = raw.get("used_memory_bytes")
            if type(pid) is not int or pid <= 0:
                raise ValueError("CUDA process detector pid must be a positive integer")
            if pid == current_pid:
                continue
            if type(used) is not int or used < 0:
                raise ValueError(
                    "CUDA process detector used_memory_bytes must be non-negative"
                )
            process: dict[str, Any] = {"pid": pid, "used_memory_bytes": used}
            name = raw.get("name")
            if isinstance(name, str) and name.strip():
                process["name"] = " ".join(name.split())[:200]
            processes.append(process)
        processes.sort(key=lambda item: (int(item["pid"]), int(item["used_memory_bytes"])))
        return {"status": "checked", "other_processes": processes}
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "status": "error",
            "error": _sanitize_error(exc),
            "other_processes": [],
        }


def _sanitize_error(error: BaseException) -> str:
    message = " ".join(str(error).split())
    message = re.sub(
        r"(?:https?|azureml|azure|az|blob|abfs|abfss|wasb|wasbs)://\S+",
        "[redacted-url]",
        message,
    )
    message = re.sub(r"\bhf_[A-Za-z0-9]{12,}\b", "[redacted-token]", message)
    return message[:1000]


def _int_or_zero(value: object) -> int:
    return value if type(value) is int and value >= 0 else 0


def _nonnegative_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


__all__ = [
    "BYTES_PER_GIB",
    "DEFAULT_LARGE_CUDA_PROCESS_BYTES",
    "MemoryGuardConfig",
    "MemoryGuardError",
    "MemoryGuardResult",
    "MemorySnapshot",
    "StructuredTrainingOOMError",
    "build_failure_metadata",
    "capture_memory_snapshot",
    "convert_cuda_oom",
    "enforce_memory_guard",
    "evaluate_memory_guard",
    "gib_to_bytes",
    "is_cuda_oom",
    "reset_cuda_memory_tracking",
]
