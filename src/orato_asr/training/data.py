"""Lazy, bounded-memory canonical-manifest consumption for wrapper training."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from ..audio import decode_audio
from ..data.manifest import manifest_fingerprint
from ..data.schema import (
    ManifestRecord,
    display_audio_locator,
    parse_record,
    resolve_local_audio_path,
)
from ..exceptions import AudioValidationError, ManifestError, TrainingError
from .official_sft import WrapperSample


@dataclass(frozen=True, slots=True)
class TrainingSampleRef:
    """Lightweight pointer to one validated local manifest record."""

    sample_id: str
    line_number: int
    byte_offset: int
    duration_seconds: float
    audio_locator: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "manifest_line": self.line_number,
            "duration_seconds": self.duration_seconds,
            "audio_filepath": self.audio_locator,
        }


@dataclass(frozen=True, slots=True)
class ExcludedTrainingSample:
    sample_id: str
    line_number: int
    audio_locator: str
    duration_seconds: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "manifest_line": self.line_number,
            "audio_filepath": self.audio_locator,
            "duration_seconds": self.duration_seconds,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class PreparedTrainingManifest:
    manifest_path: Path
    fingerprint: str
    total_samples: int
    total_duration_seconds: float
    eligible_samples: int
    eligible_duration_seconds: float
    selected: tuple[TrainingSampleRef, ...]
    excluded: tuple[ExcludedTrainingSample, ...]
    capped_samples: int
    capped_duration_seconds: float

    @property
    def selected_duration_seconds(self) -> float:
        return sum(item.duration_seconds for item in self.selected)

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.fingerprint,
            "total_samples": self.total_samples,
            "total_duration_seconds": self.total_duration_seconds,
            "total_audio_hours": self.total_duration_seconds / 3600,
            "eligible_samples": self.eligible_samples,
            "eligible_duration_seconds": self.eligible_duration_seconds,
            "eligible_audio_hours": self.eligible_duration_seconds / 3600,
            "selected_samples": len(self.selected),
            "selected_duration_seconds": self.selected_duration_seconds,
            "selected_audio_hours": self.selected_duration_seconds / 3600,
            "capped_samples": self.capped_samples,
            "capped_duration_seconds": self.capped_duration_seconds,
            "excluded": [item.as_dict() for item in self.excluded],
        }


def prepare_training_manifest(
    manifest: str | Path,
    *,
    project_root: Path,
    minimum_duration_seconds: float | None,
    maximum_duration_seconds: float,
    max_samples: int | None,
    max_hours: float,
    require_training_split: bool = False,
) -> PreparedTrainingManifest:
    """Validate every local clip while retaining only offsets and small metadata."""

    if minimum_duration_seconds is not None and (
        type(minimum_duration_seconds) not in (int, float)
        or not math.isfinite(float(minimum_duration_seconds))
        or float(minimum_duration_seconds) < 0
    ):
        raise TrainingError("minimum_duration_seconds must be non-negative and finite")
    if not _positive_finite(maximum_duration_seconds):
        raise TrainingError("maximum_duration_seconds must be positive and finite")
    if minimum_duration_seconds is not None and minimum_duration_seconds > maximum_duration_seconds:
        raise TrainingError("minimum audio duration cannot exceed maximum audio duration")
    if max_samples is not None and (type(max_samples) is not int or max_samples <= 0):
        raise TrainingError("max_samples must be null or a positive integer")
    if not _positive_finite(max_hours):
        raise TrainingError("max_hours must be positive and finite")
    if type(require_training_split) is not bool:
        raise TrainingError("require_training_split must be true or false")

    source = Path(manifest).expanduser().resolve()
    fingerprint = manifest_fingerprint(source)
    eligible: list[TrainingSampleRef] = []
    excluded: list[ExcludedTrainingSample] = []
    total_samples = 0
    total_duration = 0.0

    for offset, record in _iter_indexed_records(source):
        total_samples += 1
        if require_training_split and record.split is not None:
            split = record.split.strip().casefold()
            if split not in {"train", "training"}:
                raise TrainingError(
                    f"{source}:{record.line_number}: training rejects explicit "
                    f"non-training split {record.split!r}"
                )
        if record.is_remote:
            raise TrainingError(
                f"{source}:{record.line_number}: wrapper training requires local audio; "
                f"remote locator {display_audio_locator(record.audio_filepath)!r} is unsupported"
            )
        audio_path = resolve_local_audio_path(record, project_root)
        assert audio_path is not None
        try:
            decoded = decode_audio(audio_path)
        except AudioValidationError as exc:
            raise TrainingError(
                f"{source}:{record.line_number}: invalid training audio: {exc}"
            ) from exc
        decoded_duration = decoded.duration_seconds
        if not _positive_finite(decoded_duration):
            raise TrainingError(
                f"{source}:{record.line_number}: decoded duration must be positive and finite"
            )
        duration = float(decoded_duration)
        del decoded
        if record.duration is not None and abs(record.duration - duration) > 0.25:
            raise TrainingError(
                f"{source}:{record.line_number}: declared duration differs from "
                "decoded audio by more than 0.25 seconds"
            )
        total_duration += duration
        sample_id = stable_training_sample_id(fingerprint, record)
        display_path = display_audio_locator(record.audio_filepath)
        reason: str | None = None
        if minimum_duration_seconds is not None and duration < minimum_duration_seconds:
            reason = "below_minimum_duration"
        elif duration > maximum_duration_seconds:
            reason = "above_maximum_duration"
        if reason is not None:
            excluded.append(
                ExcludedTrainingSample(
                    sample_id,
                    record.line_number or 0,
                    display_path,
                    duration,
                    reason,
                )
            )
            continue
        eligible.append(
            TrainingSampleRef(
                sample_id,
                record.line_number or 0,
                offset,
                duration,
                display_path,
            )
        )

    if total_samples == 0:
        raise TrainingError("Training manifest contains no records")
    eligible_duration = sum(item.duration_seconds for item in eligible)
    if not eligible:
        raise TrainingError(
            "No training samples remain after applying the configured duration filter"
        )

    selected: list[TrainingSampleRef] = []
    selected_duration = 0.0
    duration_cap = max_hours * 3600
    capped_samples = 0
    capped_duration = 0.0
    for item in eligible:
        if max_samples is not None and len(selected) >= max_samples:
            capped_samples += 1
            capped_duration += item.duration_seconds
            continue
        if selected_duration + item.duration_seconds > duration_cap:
            capped_samples += 1
            capped_duration += item.duration_seconds
            continue
        selected.append(item)
        selected_duration += item.duration_seconds
    if not selected:
        raise TrainingError("Configured sample/hour bounds selected no training records")

    return PreparedTrainingManifest(
        manifest_path=source,
        fingerprint=fingerprint,
        total_samples=total_samples,
        total_duration_seconds=total_duration,
        eligible_samples=len(eligible),
        eligible_duration_seconds=eligible_duration,
        selected=tuple(selected),
        excluded=tuple(excluded),
        capped_samples=capped_samples,
        capped_duration_seconds=capped_duration,
    )


class LazyWrapperTrainingDataset(Sequence[WrapperSample]):
    """Recover and decode only the sample requested by the current microstep."""

    def __init__(self, prepared: PreparedTrainingManifest, *, project_root: Path) -> None:
        self.prepared = prepared
        self.project_root = project_root.expanduser().resolve()

    def __len__(self) -> int:
        return len(self.prepared.selected)

    def __getitem__(self, index: int) -> WrapperSample:
        ref = self.prepared.selected[index]
        record = _record_at(self.prepared.manifest_path, ref)
        recovered_sample_id = stable_training_sample_id(self.prepared.fingerprint, record)
        if recovered_sample_id != ref.sample_id:
            raise TrainingError(
                f"Sample {ref.sample_id} manifest record changed after preflight; "
                "source data must remain immutable"
            )
        path = resolve_local_audio_path(record, self.project_root)
        if path is None:
            raise TrainingError(
                f"Sample {ref.sample_id} unexpectedly resolved to remote audio"
            )
        try:
            decoded = decode_audio(path)
        except AudioValidationError as exc:
            raise TrainingError(
                f"Sample {ref.sample_id} at manifest line {ref.line_number} failed decoding: {exc}"
            ) from exc
        if not _positive_finite(decoded.duration_seconds):
            raise TrainingError(
                f"Sample {ref.sample_id} decoded duration must be positive and finite"
            )
        if abs(float(decoded.duration_seconds) - ref.duration_seconds) > 0.001:
            raise TrainingError(
                f"Sample {ref.sample_id} duration changed after preflight; source data must remain immutable"
            )
        return WrapperSample(
            sample_id=ref.sample_id,
            audio=decoded.samples,
            duration_seconds=decoded.duration_seconds,
            transcript=record.text,
            language=record.language,
            line_number=ref.line_number,
            source=record.source,
            speaker_id=record.speaker_id,
            recording_id=record.recording_id,
            domain=record.domain,
            split=record.split,
            metadata=record.metadata,
        )


def stable_training_sample_id(fingerprint: str, record: ManifestRecord) -> str:
    record_payload = json.dumps(
        record.as_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    payload = "\0".join(
        (
            fingerprint,
            str(record.line_number or 0),
            record_payload,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _iter_indexed_records(source: Path) -> Iterator[tuple[int, ManifestRecord]]:
    try:
        with source.open("rb") as handle:
            line_number = 0
            while True:
                offset = handle.tell()
                raw_line = handle.readline()
                if not raw_line:
                    break
                line_number += 1
                if not raw_line.strip():
                    continue
                try:
                    decoded = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ManifestError(
                        f"{source}:{line_number}: invalid UTF-8 JSONL record: {exc}"
                    ) from exc
                yield offset, parse_record(
                    decoded, manifest_path=source, line_number=line_number
                )
    except OSError as exc:
        raise ManifestError(f"Could not index training manifest {source}: {exc}") from exc


def _record_at(source: Path, ref: TrainingSampleRef) -> ManifestRecord:
    try:
        with source.open("rb") as handle:
            handle.seek(ref.byte_offset)
            raw_line = handle.readline().decode("utf-8")
        decoded = json.loads(raw_line)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingError(
            f"Could not recover sample {ref.sample_id} at manifest line {ref.line_number}: {exc}"
        ) from exc
    return parse_record(decoded, manifest_path=source, line_number=ref.line_number)


def _positive_finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value)) and float(value) > 0


__all__ = [
    "ExcludedTrainingSample",
    "LazyWrapperTrainingDataset",
    "PreparedTrainingManifest",
    "TrainingSampleRef",
    "prepare_training_manifest",
    "stable_training_sample_id",
]
