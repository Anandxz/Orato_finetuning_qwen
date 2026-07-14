"""Evaluation normalization, metrics, and baseline-model execution."""

from .metrics import compute_text_metrics
from .normalization import NormalizationOptions, normalize_standard

__all__ = ["NormalizationOptions", "compute_text_metrics", "normalize_standard"]
