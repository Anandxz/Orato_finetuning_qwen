"""Narrow, non-mutating local WAV/FLAC decoding for Qwen3-ASR inference."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import AudioValidationError, DependencyError

TARGET_SAMPLE_RATE = 16_000
SUPPORTED_SUFFIXES = {".wav", ".flac"}


@dataclass(frozen=True, slots=True)
class DecodedAudio:
    path: Path
    samples: Any
    original_sample_rate: int
    sample_rate: int
    original_channels: int
    channels: int
    duration_seconds: float
    downmixed: bool
    resampled: bool

    def metadata(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "original_sample_rate": self.original_sample_rate,
            "sample_rate": self.sample_rate,
            "original_channels": self.original_channels,
            "channels": self.channels,
            "duration_seconds": self.duration_seconds,
            "downmixed": self.downmixed,
            "resampled": self.resampled,
        }


def decode_audio(path: str | Path) -> DecodedAudio:
    """Decode local audio to mono float32 at 16 kHz without changing the source."""

    audio_path = Path(path).expanduser().resolve()
    if audio_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise AudioValidationError("Audio must be a local WAV or FLAC file")
    if not audio_path.exists():
        raise AudioValidationError(f"Audio file does not exist: {audio_path}")
    if not audio_path.is_file():
        raise AudioValidationError(f"Audio path is not a regular file: {audio_path}")
    if audio_path.stat().st_size <= 0:
        raise AudioValidationError(f"Audio file is empty: {audio_path}")

    soundfile = _import_dependency("soundfile", "soundfile==0.14.0")
    try:
        samples, sample_rate = soundfile.read(
            str(audio_path), dtype="float32", always_2d=True
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise AudioValidationError(f"Could not decode audio {audio_path}: {exc}") from exc

    if type(sample_rate) is not int or sample_rate <= 0:
        raise AudioValidationError(f"Decoded audio has an invalid sample rate: {sample_rate!r}")
    if getattr(samples, "ndim", None) != 2 or samples.shape[0] <= 0:
        raise AudioValidationError("Decoded audio contains no samples")
    original_channels = int(samples.shape[1])
    if original_channels <= 0:
        raise AudioValidationError("Decoded audio contains no channels")
    duration = float(samples.shape[0]) / sample_rate
    if not math.isfinite(duration) or duration <= 0:
        raise AudioValidationError("Decoded audio duration must be positive and finite")

    downmixed = original_channels > 1
    mono = samples.mean(axis=1, dtype="float32") if downmixed else samples[:, 0]
    resampled = sample_rate != TARGET_SAMPLE_RATE
    if resampled:
        soxr = _import_dependency("soxr", "soxr==1.1.0")
        try:
            mono = soxr.resample(mono, sample_rate, TARGET_SAMPLE_RATE, quality="HQ")
        except (RuntimeError, ValueError) as exc:
            raise AudioValidationError(f"Could not resample audio to 16 kHz: {exc}") from exc

    if getattr(mono, "size", 0) <= 0:
        raise AudioValidationError("Decoded audio contains no usable samples")
    mono = mono.astype("float32", copy=False)
    return DecodedAudio(
        path=audio_path,
        samples=mono,
        original_sample_rate=sample_rate,
        sample_rate=TARGET_SAMPLE_RATE,
        original_channels=original_channels,
        channels=1,
        duration_seconds=duration,
        downmixed=downmixed,
        resampled=resampled,
    )


def _import_dependency(module_name: str, requirement: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except (ImportError, OSError) as exc:
        raise DependencyError(
            f"Missing audio dependency {requirement}; install requirements/inference.txt"
        ) from exc
