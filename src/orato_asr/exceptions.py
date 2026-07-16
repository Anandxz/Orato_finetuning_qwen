"""Small shared exception hierarchy for foundation utilities."""

from __future__ import annotations


class OratoASRError(Exception):
    """Base exception for expected Orato ASR project failures."""


class ConfigError(OratoASRError, ValueError):
    """Raised when a project configuration cannot be loaded."""


class ConfigValidationError(ConfigError):
    """Raised when loaded configuration values violate the schema."""


class PathSafetyError(OratoASRError, ValueError):
    """Raised when a repository-managed path is unsafe or unsupported."""


class StorageError(OratoASRError):
    """Raised when configured local or Azure-backed data cannot be resolved."""


class PreflightError(OratoASRError):
    """Raised when an expected preflight operation cannot be completed."""


class DependencyError(OratoASRError, ImportError):
    """Raised when the explicitly selected inference stack is unavailable."""


class AudioValidationError(OratoASRError, ValueError):
    """Raised when a local audio input is unsafe or cannot be decoded."""


class DeviceSelectionError(OratoASRError, ValueError):
    """Raised when a requested device or precision is unsupported."""


class ModelLoadError(OratoASRError):
    """Raised when the pinned native model or processor cannot be loaded."""


class InferenceError(OratoASRError):
    """Raised when native model inference cannot complete."""


class InferenceOOMError(InferenceError):
    """Raised when CUDA exhausts device memory without a silent fallback."""


class ManifestError(OratoASRError, ValueError):
    """Raised when a canonical JSONL manifest cannot be read or written."""


class ManifestValidationError(ManifestError):
    """Raised when manifest records violate the canonical schema."""


class EvaluationError(OratoASRError):
    """Raised when baseline evaluation cannot complete safely."""


class BaselineStoppedError(EvaluationError):
    """Raised when baseline early-collapse protection stops a run."""


class TrainingError(OratoASRError):
    """Raised when the explicitly requested training path cannot proceed safely."""


class WrapperCompatibilityError(TrainingError):
    """Raised when qwen-asr wrapper targets, labels, or model structure disagree."""


class TrainingOOMError(TrainingError):
    """Raised when a training stage exhausts CUDA memory without fallback."""


class MemorySafetyError(TrainingError):
    """Raised before a stage would cross a configured memory guard."""


class AdapterVerificationError(TrainingError):
    """Raised when an adapter cannot be safely loaded and qualified."""


# Descriptive aliases keep the public meaning clear without duplicate classes.
ProjectConfigurationError = ConfigError
ConfigurationValidationError = ConfigValidationError
UnsafePathError = PathSafetyError

__all__ = [
    "ConfigError",
    "ConfigValidationError",
    "ConfigurationValidationError",
    "AudioValidationError",
    "DependencyError",
    "DeviceSelectionError",
    "InferenceError",
    "InferenceOOMError",
    "ModelLoadError",
    "ManifestError",
    "ManifestValidationError",
    "EvaluationError",
    "BaselineStoppedError",
    "TrainingError",
    "WrapperCompatibilityError",
    "TrainingOOMError",
    "MemorySafetyError",
    "AdapterVerificationError",
    "OratoASRError",
    "PathSafetyError",
    "StorageError",
    "PreflightError",
    "ProjectConfigurationError",
    "UnsafePathError",
]
