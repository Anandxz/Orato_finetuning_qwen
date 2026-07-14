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


class PreflightError(OratoASRError):
    """Raised when an expected preflight operation cannot be completed."""


# Descriptive aliases keep the public meaning clear without duplicate classes.
ProjectConfigurationError = ConfigError
ConfigurationValidationError = ConfigValidationError
UnsafePathError = PathSafetyError

__all__ = [
    "ConfigError",
    "ConfigValidationError",
    "ConfigurationValidationError",
    "OratoASRError",
    "PathSafetyError",
    "PreflightError",
    "ProjectConfigurationError",
    "UnsafePathError",
]
