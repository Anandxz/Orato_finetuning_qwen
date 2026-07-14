"""Strict, lightweight YAML configuration loading."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from .exceptions import ConfigError, ConfigValidationError
from .paths import PathSafetyError, find_project_root, resolve_repository_path

SCHEMA_VERSION = 1
TRANSCRIPT_POLICY = "mixed_devanagari_hindi_latin_english_v1"

_TOP_LEVEL_KEYS = {
    "schema_version",
    "profile",
    "model",
    "transcript",
    "data",
    "hardware",
    "training",
    "checkpointing",
    "evaluation",
    "paths",
}

_SECTION_KEYS = {
    "profile": {"name", "intent", "training_status"},
    "model": {"id"},
    "transcript": {
        "policy",
        "hindi_script",
        "english_script",
        "roman_hinglish_primary",
        "short_utterances_allowed",
    },
    "data": {
        "max_samples",
        "max_audio_hours",
        "raw_data_mutation_enabled",
        "synthetic_data_enabled",
    },
    "hardware": {
        "device_preference",
        "accelerator",
        "gpu_count",
        "process_count",
        "node_count",
        "distributed",
        "rank_zero_writes",
        "launcher",
    },
    "training": {
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "full_parameter_fit_guaranteed",
    },
    "checkpointing": {"enabled", "save_steps", "keep_last", "resume_enabled"},
    "evaluation": {
        "validation_enabled",
        "validation_steps",
        "full_evaluation_enabled",
        "reporting_enabled",
    },
    "paths": {"output_dir", "reports_dir"},
}


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """An immutable validated project configuration."""

    source_path: Path
    project_root: Path
    _values: Mapping[str, object]

    @property
    def values(self) -> Mapping[str, object]:
        """Return the recursively immutable configuration mapping."""

        return self._values

    def as_dict(self) -> dict[str, Any]:
        """Return a detached mutable copy suitable for YAML or JSON output."""

        thawed = _thaw(self._values)
        if not isinstance(thawed, dict):  # Defensive: the root is validated as a map.
            raise TypeError("Configuration root is not a mapping")
        return thawed


def load_config(
    path: str | Path,
    project_root: str | Path | None = None,
) -> ProjectConfig:
    """Load, strictly validate, resolve, and freeze a YAML configuration."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise ConfigError(f"Configuration file does not exist: {source_path}")
    if not source_path.is_file():
        raise ConfigError(f"Configuration path is not a file: {source_path}")

    root = _resolve_project_root(source_path, project_root)

    try:
        with source_path.open("r", encoding="utf-8") as config_file:
            loaded = yaml.safe_load(config_file)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {source_path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read configuration {source_path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ConfigError("Configuration root must be a YAML mapping")

    values = copy.deepcopy(loaded)
    try:
        _validate_and_resolve(values, root)
    except ConfigValidationError:
        raise
    except ConfigError as exc:
        raise ConfigValidationError(str(exc)) from exc
    return ProjectConfig(
        source_path=source_path,
        project_root=root,
        _values=_freeze(values),
    )


def _resolve_project_root(source_path: Path, project_root: str | Path | None) -> Path:
    if project_root is not None:
        root = Path(project_root).expanduser().resolve()
        if not (root / "pyproject.toml").is_file():
            raise ConfigError(
                f"Project root must contain pyproject.toml: {root}"
            )
        return root

    anchors = (source_path.parent, Path.cwd(), Path(__file__).resolve().parent)
    for anchor in anchors:
        try:
            return find_project_root(anchor)
        except PathSafetyError:
            continue

    raise ConfigError(
        "Could not locate the project root; run from the repository or pass "
        "project_root explicitly"
    )


def _validate_and_resolve(values: dict[str, Any], project_root: Path) -> None:
    _validate_keys(values, _TOP_LEVEL_KEYS, "configuration root")

    schema_version = values["schema_version"]
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise ConfigError(
            f"schema_version must be integer {SCHEMA_VERSION}; received "
            f"{schema_version!r}"
        )

    sections = {
        name: _section(values, name)
        for name in _SECTION_KEYS
    }
    for name, section in sections.items():
        _validate_keys(section, _SECTION_KEYS[name], f"section {name!r}")

    profile = sections["profile"]
    _non_empty_string(profile["name"], "profile.name")
    _non_empty_string(profile["intent"], "profile.intent")
    if profile["training_status"] != "not_implemented":
        raise ConfigError(
            "profile.training_status must be 'not_implemented' in the foundation milestone"
        )

    model = sections["model"]
    _non_empty_string(model["id"], "model.id")

    transcript = sections["transcript"]
    if transcript["policy"] != TRANSCRIPT_POLICY:
        raise ConfigError(
            f"transcript.policy must be {TRANSCRIPT_POLICY!r}; received "
            f"{transcript['policy']!r}"
        )
    if transcript["hindi_script"] != "devanagari":
        raise ConfigError(
            "transcript.hindi_script must be 'devanagari' for the canonical target"
        )
    if transcript["english_script"] != "latin":
        raise ConfigError(
            "transcript.english_script must be 'latin' for the canonical target"
        )
    roman_hinglish_primary = _boolean(
        transcript["roman_hinglish_primary"],
        "transcript.roman_hinglish_primary",
    )
    if roman_hinglish_primary:
        raise ConfigError(
            "transcript.roman_hinglish_primary must be false for version 1"
        )
    short_utterances_allowed = _boolean(
        transcript["short_utterances_allowed"],
        "transcript.short_utterances_allowed",
    )
    if not short_utterances_allowed:
        raise ConfigError(
            "transcript.short_utterances_allowed must be true; valid short utterances "
            "must not be removed by word count"
        )

    data = sections["data"]
    max_samples = _optional_positive_int(data["max_samples"], "data.max_samples")
    max_audio_hours = _optional_positive_number(
        data["max_audio_hours"], "data.max_audio_hours"
    )
    if max_samples is None and max_audio_hours is None:
        raise ConfigError(
            "At least one of data.max_samples or data.max_audio_hours must be set"
        )
    raw_data_mutation_enabled = _boolean(
        data["raw_data_mutation_enabled"], "data.raw_data_mutation_enabled"
    )
    if raw_data_mutation_enabled:
        raise ConfigError("data.raw_data_mutation_enabled must be false")
    synthetic_data_enabled = _boolean(
        data["synthetic_data_enabled"], "data.synthetic_data_enabled"
    )
    if synthetic_data_enabled:
        raise ConfigError(
            "data.synthetic_data_enabled must be false in the foundation milestone"
        )

    hardware = sections["hardware"]
    _non_empty_string(hardware["device_preference"], "hardware.device_preference")
    _non_empty_string(hardware["accelerator"], "hardware.accelerator")
    gpu_count = _positive_int(hardware["gpu_count"], "hardware.gpu_count")
    process_count = _positive_int(
        hardware["process_count"], "hardware.process_count"
    )
    node_count = _positive_int(hardware["node_count"], "hardware.node_count")
    distributed = _boolean(hardware["distributed"], "hardware.distributed")
    rank_zero_writes = _boolean(
        hardware["rank_zero_writes"], "hardware.rank_zero_writes"
    )
    launcher = _non_empty_string(hardware["launcher"], "hardware.launcher")

    if not distributed and process_count != 1:
        raise ConfigError(
            "hardware.process_count must be 1 when hardware.distributed is false"
        )
    if distributed and process_count != gpu_count * node_count:
        raise ConfigError(
            "Distributed process count must equal gpu_count * node_count; "
            f"received {process_count} processes, {gpu_count} GPUs, and "
            f"{node_count} nodes"
        )
    if distributed and not rank_zero_writes:
        raise ConfigError(
            "hardware.rank_zero_writes must be true for distributed shared outputs"
        )
    if distributed and launcher != "torchrun":
        raise ConfigError(
            "hardware.launcher must be 'torchrun' for the planned distributed profile"
        )
    if not distributed and launcher != "single_process":
        raise ConfigError(
            "hardware.launcher must be 'single_process' when distributed is false"
        )
    if gpu_count == 8 and (
        node_count != 1 or process_count != 8 or not distributed or not rank_zero_writes
    ):
        raise ConfigError(
            "Eight-GPU profiles require one node, eight processes, distributed mode, "
            "and rank-zero shared writes"
        )
    if gpu_count == 1 and process_count == 8:
        raise ConfigError("A single-GPU profile cannot declare eight processes")

    training = sections["training"]
    _positive_int(
        training["per_device_batch_size"], "training.per_device_batch_size"
    )
    _positive_int(
        training["gradient_accumulation_steps"],
        "training.gradient_accumulation_steps",
    )
    _positive_int(training["max_steps"], "training.max_steps")
    full_parameter_fit_guaranteed = _boolean(
        training["full_parameter_fit_guaranteed"],
        "training.full_parameter_fit_guaranteed",
    )
    if full_parameter_fit_guaranteed:
        raise ConfigError(
            "training.full_parameter_fit_guaranteed must be false until training is qualified"
        )

    checkpointing = sections["checkpointing"]
    checkpoints_enabled = _boolean(
        checkpointing["enabled"], "checkpointing.enabled"
    )
    resume_enabled = _boolean(
        checkpointing["resume_enabled"], "checkpointing.resume_enabled"
    )
    if checkpoints_enabled:
        _positive_int(checkpointing["save_steps"], "checkpointing.save_steps")
        _positive_int(checkpointing["keep_last"], "checkpointing.keep_last")
    elif (
        checkpointing["save_steps"] is not None
        or checkpointing["keep_last"] is not None
        or resume_enabled
    ):
        raise ConfigError(
            "Disabled checkpointing requires null save_steps, null keep_last, and "
            "resume_enabled=false"
        )

    evaluation = sections["evaluation"]
    validation_enabled = _boolean(
        evaluation["validation_enabled"], "evaluation.validation_enabled"
    )
    full_evaluation_enabled = _boolean(
        evaluation["full_evaluation_enabled"],
        "evaluation.full_evaluation_enabled",
    )
    _boolean(evaluation["reporting_enabled"], "evaluation.reporting_enabled")
    if validation_enabled:
        _positive_int(
            evaluation["validation_steps"], "evaluation.validation_steps"
        )
    elif evaluation["validation_steps"] is not None:
        raise ConfigError(
            "evaluation.validation_steps must be null when validation is disabled"
        )
    if full_evaluation_enabled and not validation_enabled:
        raise ConfigError(
            "Full evaluation requires evaluation.validation_enabled=true"
        )

    paths = sections["paths"]
    for key, allowed_directory in (
        ("output_dir", "outputs"),
        ("reports_dir", "reports"),
    ):
        try:
            paths[key] = resolve_repository_path(
                paths[key],
                project_root=project_root,
                allowed_directory=allowed_directory,
            )
        except PathSafetyError as exc:
            raise ConfigError(f"paths.{key} {exc}") from exc


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
    non_string_keys = [key for key in mapping if not isinstance(key, str)]
    if non_string_keys:
        raise ConfigError(f"{label} contains non-string keys: {non_string_keys!r}")

    actual_keys = set(mapping)
    missing = sorted(expected_keys - actual_keys)
    unknown = sorted(actual_keys - expected_keys)
    if missing:
        raise ConfigError(f"{label} is missing required keys: {', '.join(missing)}")
    if unknown:
        raise ConfigError(f"{label} contains unknown keys: {', '.join(unknown)}")


def _non_empty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty string")
    return value


def _boolean(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise ConfigError(f"{field} must be true or false; received {value!r}")
    return value


def _positive_int(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ConfigError(f"{field} must be a positive integer; received {value!r}")
    return value


def _optional_positive_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field)


def _optional_positive_number(value: object, field: str) -> float | int | None:
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ConfigError(f"{field} must be a positive finite number; received {value!r}")
    return value


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
