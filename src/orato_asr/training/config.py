"""Strict configuration for the isolated Qwen wrapper LoRA backend.

This loader intentionally does not extend :mod:`orato_asr.config`.  Native
inference and wrapper training use different checkpoints, dependency stacks,
and processor contracts, so accepting either shape through one schema would
make an accidental backend mix much harder to detect.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from orato_asr.exceptions import ConfigError
from orato_asr.paths import PathSafetyError, find_project_root, resolve_repository_path

WRAPPER_CONFIG_SCHEMA_VERSION = 1
WRAPPER_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
WRAPPER_MODEL_REVISION = "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
WRAPPER_BACKEND = "qwen_asr_wrapper"

_TOP_LEVEL_KEYS = {
    "schema_version",
    "model",
    "method",
    "data",
    "training",
    "memory",
    "runtime",
    "paths",
}

_SECTION_KEYS = {
    "model": {"id", "revision", "backend", "dtype"},
    "method": {
        "type",
        "rank",
        "alpha",
        "dropout",
        "bias",
        "freeze_audio_encoder",
        "target_scope",
    },
    "data": {
        "min_audio_seconds",
        "max_audio_seconds",
        "max_samples",
        "max_hours",
        "num_workers",
        "pin_memory",
        "persistent_workers",
    },
    "training": {
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "max_optimizer_steps",
        "learning_rate",
        "weight_decay",
        "warmup_steps",
        "max_grad_norm",
        "gradient_checkpointing",
        "use_cache",
        "seed",
        "log_every_optimizer_steps",
        "save_every_optimizer_steps",
    },
    "memory": {
        "gpu_safety_limit_gb",
        "minimum_available_system_ram_gb",
        "abort_on_threshold",
        "capture_system_ram",
    },
    "runtime": {"device", "distributed", "cpu_fallback", "run_kind"},
    "paths": {"output_root", "reports_root", "model_cache_dir"},
}


@dataclass(frozen=True, slots=True)
class WrapperTrainingConfig:
    """Recursively immutable, validated wrapper-LoRA configuration."""

    source_path: Path
    project_root: Path
    _values: Mapping[str, object]

    @property
    def values(self) -> Mapping[str, object]:
        """Return the immutable configuration mapping."""

        return self._values

    def as_dict(self) -> dict[str, Any]:
        """Return a detached serialization-safe copy of the configuration."""

        thawed = _thaw(self._values)
        if not isinstance(thawed, dict):  # The root is validated as a mapping.
            raise TypeError("Wrapper training configuration root is not a mapping")
        return thawed


def load_wrapper_training_config(
    path: str | Path,
    project_root: str | Path | None = None,
) -> WrapperTrainingConfig:
    """Load and strictly validate the standalone wrapper-LoRA schema."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise ConfigError(f"Wrapper training configuration does not exist: {source_path}")
    if not source_path.is_file():
        raise ConfigError(
            f"Wrapper training configuration path is not a file: {source_path}"
        )

    root = _resolve_project_root(source_path, project_root)
    try:
        with source_path.open("r", encoding="utf-8") as config_file:
            loaded = yaml.safe_load(config_file)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid wrapper training YAML in {source_path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(
            f"Could not read wrapper training configuration {source_path}: {exc}"
        ) from exc

    if not isinstance(loaded, dict):
        raise ConfigError("Wrapper training configuration root must be a YAML mapping")

    values = copy.deepcopy(loaded)
    _validate_and_resolve(values, root)
    return WrapperTrainingConfig(
        source_path=source_path,
        project_root=root,
        _values=_freeze(values),
    )


def _resolve_project_root(
    source_path: Path,
    project_root: str | Path | None,
) -> Path:
    if project_root is not None:
        root = Path(project_root).expanduser().resolve()
        if not (root / "pyproject.toml").is_file():
            raise ConfigError(f"Project root must contain pyproject.toml: {root}")
        return root

    for anchor in (source_path.parent, Path.cwd(), Path(__file__).resolve().parent):
        try:
            return find_project_root(anchor)
        except PathSafetyError:
            continue
    raise ConfigError(
        "Could not locate the project root; run from the repository or pass "
        "project_root explicitly"
    )


def _validate_and_resolve(values: dict[str, Any], project_root: Path) -> None:
    _validate_keys(values, _TOP_LEVEL_KEYS, "wrapper training configuration root")

    schema_version = values["schema_version"]
    if type(schema_version) is not int or schema_version != WRAPPER_CONFIG_SCHEMA_VERSION:
        raise ConfigError(
            "schema_version must be integer "
            f"{WRAPPER_CONFIG_SCHEMA_VERSION} for wrapper training; received "
            f"{schema_version!r}"
        )

    runtime_section = _section(values, "runtime")
    runtime_section.setdefault("run_kind", "smoke")
    sections = {name: _section(values, name) for name in _SECTION_KEYS}
    for name, section in sections.items():
        _validate_keys(section, _SECTION_KEYS[name], f"section {name!r}")

    model = sections["model"]
    _exact(model["id"], "model.id", WRAPPER_MODEL_ID)
    _exact(model["revision"], "model.revision", WRAPPER_MODEL_REVISION)
    _exact(model["backend"], "model.backend", WRAPPER_BACKEND)
    _exact(model["dtype"], "model.dtype", "bfloat16")

    method = sections["method"]
    _exact(method["type"], "method.type", "lora")
    rank = _positive_int(method["rank"], "method.rank")
    if rank not in {2, 4}:
        raise ConfigError("method.rank must be 4, or 2 only for the documented OOM fallback")
    alpha = _positive_int(method["alpha"], "method.alpha")
    if alpha not in {8, 16}:
        raise ConfigError("method.alpha must be either 8 or 16")
    if alpha < rank:
        raise ConfigError("method.alpha must be greater than or equal to method.rank")
    dropout = _finite_number(method["dropout"], "method.dropout")
    if not 0.0 <= dropout < 1.0:
        raise ConfigError("method.dropout must be at least 0 and less than 1")
    _exact(method["bias"], "method.bias", "none")
    _required_boolean(
        method["freeze_audio_encoder"],
        "method.freeze_audio_encoder",
        required=True,
    )
    _exact(
        method["target_scope"],
        "method.target_scope",
        "text_decoder_attention_qv_only",
    )

    data = sections["data"]
    min_audio_seconds = _optional_nonnegative_number(
        data["min_audio_seconds"], "data.min_audio_seconds"
    )
    max_audio_seconds = _positive_number(
        data["max_audio_seconds"], "data.max_audio_seconds"
    )
    run_kind = _exact_one_of(
        sections["runtime"]["run_kind"],
        "runtime.run_kind",
        {"smoke", "full_epoch"},
    )
    maximum_audio_cap = 10 if run_kind == "smoke" else 30
    if max_audio_seconds > maximum_audio_cap:
        raise ConfigError(
            "data.max_audio_seconds must not exceed "
            f"{maximum_audio_cap} for runtime.run_kind={run_kind!r}"
        )
    if min_audio_seconds is not None and min_audio_seconds >= max_audio_seconds:
        raise ConfigError(
            "data.min_audio_seconds must be less than data.max_audio_seconds"
        )
    _optional_positive_int(data["max_samples"], "data.max_samples")
    max_hours = _positive_number(data["max_hours"], "data.max_hours")
    if run_kind == "full_epoch" and max_hours > 8:
        raise ConfigError("data.max_hours must not exceed 8 for the local full-epoch run")
    num_workers = _nonnegative_int(data["num_workers"], "data.num_workers")
    if num_workers != 0:
        raise ConfigError("data.num_workers must be 0 for the laptop smoke run")
    _required_boolean(data["pin_memory"], "data.pin_memory", required=False)
    _required_boolean(
        data["persistent_workers"], "data.persistent_workers", required=False
    )

    training = sections["training"]
    batch_size = _positive_int(
        training["per_device_batch_size"], "training.per_device_batch_size"
    )
    if batch_size != 1:
        raise ConfigError("training.per_device_batch_size must be 1 for laptop safety")
    accumulation = _positive_int(
        training["gradient_accumulation_steps"],
        "training.gradient_accumulation_steps",
    )
    if accumulation > 8:
        raise ConfigError(
            "training.gradient_accumulation_steps must not exceed 8 for this smoke profile"
        )
    max_steps = _positive_int(
        training["max_optimizer_steps"], "training.max_optimizer_steps"
    )
    maximum_step_cap = 20 if run_kind == "smoke" else 1000
    if max_steps > maximum_step_cap:
        raise ConfigError(
            "training.max_optimizer_steps must not exceed "
            f"{maximum_step_cap} for runtime.run_kind={run_kind!r}"
        )
    _positive_number(training["learning_rate"], "training.learning_rate")
    _nonnegative_number(training["weight_decay"], "training.weight_decay")
    warmup_steps = _nonnegative_int(training["warmup_steps"], "training.warmup_steps")
    if warmup_steps != 0:
        raise ConfigError("training.warmup_steps must be 0 for the direct smoke loop")
    _positive_number(training["max_grad_norm"], "training.max_grad_norm")
    _required_boolean(
        training["gradient_checkpointing"],
        "training.gradient_checkpointing",
        required=True,
    )
    _required_boolean(training["use_cache"], "training.use_cache", required=False)
    _nonnegative_int(training["seed"], "training.seed")
    _positive_int(
        training["log_every_optimizer_steps"],
        "training.log_every_optimizer_steps",
    )
    _positive_int(
        training["save_every_optimizer_steps"],
        "training.save_every_optimizer_steps",
    )

    memory = sections["memory"]
    safety_limit = _positive_number(
        memory["gpu_safety_limit_gb"], "memory.gpu_safety_limit_gb"
    )
    if safety_limit > 5.3:
        raise ConfigError(
            "memory.gpu_safety_limit_gb must not exceed 5.3 on the 6 GB laptop GPU"
        )
    minimum_ram = _positive_number(
        memory["minimum_available_system_ram_gb"],
        "memory.minimum_available_system_ram_gb",
    )
    if minimum_ram > 4.0:
        raise ConfigError(
            "memory.minimum_available_system_ram_gb must not exceed 4.0 on this "
            "roughly 8 GB laptop"
        )
    _required_boolean(
        memory["abort_on_threshold"], "memory.abort_on_threshold", required=True
    )
    _required_boolean(
        memory["capture_system_ram"], "memory.capture_system_ram", required=True
    )

    runtime = sections["runtime"]
    _exact_one_of(runtime["run_kind"], "runtime.run_kind", {"smoke", "full_epoch"})
    _exact(runtime["device"], "runtime.device", "cuda")
    _required_boolean(runtime["distributed"], "runtime.distributed", required=False)
    _required_boolean(runtime["cpu_fallback"], "runtime.cpu_fallback", required=False)

    paths = sections["paths"]
    for key, allowed_directory in (
        ("output_root", "outputs"),
        ("reports_root", "reports"),
    ):
        try:
            paths[key] = resolve_repository_path(
                paths[key],
                project_root=project_root,
                allowed_directory=allowed_directory,
            )
        except PathSafetyError as exc:
            raise ConfigError(f"paths.{key} {exc}") from exc

    if paths["model_cache_dir"] is not None:
        try:
            paths["model_cache_dir"] = resolve_repository_path(
                paths["model_cache_dir"],
                project_root=project_root,
                allowed_directory="outputs",
            )
        except PathSafetyError as exc:
            raise ConfigError(f"paths.model_cache_dir {exc}") from exc


def _section(values: dict[str, Any], name: str) -> dict[str, Any]:
    section = values.get(name)
    if not isinstance(section, dict):
        raise ConfigError(f"Required section {name!r} must be a YAML mapping")
    return section


def _validate_keys(
    mapping: Mapping[object, object],
    expected_keys: set[str],
    label: str,
) -> None:
    non_string = [key for key in mapping if not isinstance(key, str)]
    if non_string:
        raise ConfigError(f"{label} contains non-string keys: {non_string!r}")
    actual = set(mapping)
    missing = sorted(expected_keys - actual)
    unknown = sorted(actual - expected_keys)
    if missing:
        raise ConfigError(f"{label} is missing required keys: {', '.join(missing)}")
    if unknown:
        raise ConfigError(f"{label} contains unknown keys: {', '.join(unknown)}")


def _non_empty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty string")
    return value


def _exact(value: object, field: str, expected: str) -> str:
    selected = _non_empty_string(value, field)
    if selected != expected:
        raise ConfigError(f"{field} must be {expected!r}; received {selected!r}")
    return selected


def _exact_one_of(value: object, field: str, expected: set[str]) -> str:
    selected = _non_empty_string(value, field)
    if selected not in expected:
        raise ConfigError(
            f"{field} must be one of {', '.join(sorted(expected))}; received {selected!r}"
        )
    return selected


def _required_boolean(value: object, field: str, *, required: bool) -> bool:
    if type(value) is not bool:
        raise ConfigError(f"{field} must be true or false; received {value!r}")
    if value is not required:
        raise ConfigError(f"{field} must be {str(required).lower()} for this smoke profile")
    return value


def _positive_int(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ConfigError(f"{field} must be a positive integer; received {value!r}")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ConfigError(f"{field} must be a non-negative integer; received {value!r}")
    return value


def _optional_positive_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field)


def _finite_number(value: object, field: str) -> float | int:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ConfigError(f"{field} must be a finite number; received {value!r}")
    return value


def _positive_number(value: object, field: str) -> float | int:
    number = _finite_number(value, field)
    if number <= 0:
        raise ConfigError(f"{field} must be positive; received {value!r}")
    return number


def _nonnegative_number(value: object, field: str) -> float | int:
    number = _finite_number(value, field)
    if number < 0:
        raise ConfigError(f"{field} must be non-negative; received {value!r}")
    return number


def _optional_nonnegative_number(value: object, field: str) -> float | int | None:
    if value is None:
        return None
    return _nonnegative_number(value, field)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return copy.deepcopy(value)
