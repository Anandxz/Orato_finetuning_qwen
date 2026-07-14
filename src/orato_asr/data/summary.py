"""Streaming dataset summaries using canonical manifest records."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..audio import DecodedAudio, decode_audio
from .manifest import iter_manifest_events
from .schema import ManifestRecord, display_audio_locator, script_classification
from .validation import (
    DEFAULT_DURATION_TOLERANCE_SECONDS,
    Finding,
    inspect_local_audio,
    normalized_audio_locator,
    transcript_findings,
)

PERCENTILE_RESERVOIR_SIZE = 10_000


@dataclass(slots=True)
class DatasetSummary:
    manifest: Path
    check_audio: bool
    hash_local_audio: bool
    records: int = 0
    total_duration_seconds: float = 0.0
    duration_count: int = 0
    minimum_duration_seconds: float | None = None
    maximum_duration_seconds: float | None = None
    language_counts: Counter[str] = field(default_factory=Counter)
    source_counts: Counter[str] = field(default_factory=Counter)
    domain_counts: Counter[str] = field(default_factory=Counter)
    sample_rate_counts: Counter[str] = field(default_factory=Counter)
    channel_counts: Counter[str] = field(default_factory=Counter)
    format_counts: Counter[str] = field(default_factory=Counter)
    script_counts: Counter[str] = field(default_factory=Counter)
    missing_optional_fields: Counter[str] = field(default_factory=Counter)
    findings: list[Finding] = field(default_factory=list)
    duplicate_path_count: int = 0
    duplicate_content_count: int = 0
    duplicate_transcript_count: int = 0
    _durations: list[float] = field(default_factory=list)
    _seen_paths: set[str] = field(default_factory=set)
    _seen_hashes: set[str] = field(default_factory=set)
    _seen_transcripts: set[str] = field(default_factory=set)
    _random: random.Random = field(default_factory=lambda: random.Random(0))

    def add_duration(self, duration: float) -> None:
        self.total_duration_seconds += duration
        self.duration_count += 1
        self.minimum_duration_seconds = duration if self.minimum_duration_seconds is None else min(self.minimum_duration_seconds, duration)
        self.maximum_duration_seconds = duration if self.maximum_duration_seconds is None else max(self.maximum_duration_seconds, duration)
        if len(self._durations) < PERCENTILE_RESERVOIR_SIZE:
            self._durations.append(duration)
        else:
            replacement = self._random.randrange(self.duration_count)
            if replacement < PERCENTILE_RESERVOIR_SIZE:
                self._durations[replacement] = duration

    def as_dict(self) -> dict[str, Any]:
        severity_counts = Counter(finding.severity for finding in self.findings)
        return {
            "manifest": str(self.manifest),
            "check_audio": self.check_audio,
            "hash_local_audio": self.hash_local_audio,
            "records": self.records,
            "total_duration_seconds": self.total_duration_seconds,
            "audio_hours": self.total_duration_seconds / 3600,
            "duration_count": self.duration_count,
            "minimum_duration_seconds": self.minimum_duration_seconds,
            "maximum_duration_seconds": self.maximum_duration_seconds,
            "mean_duration_seconds": self.total_duration_seconds / self.duration_count if self.duration_count else None,
            "duration_percentiles_seconds": _percentiles(self._durations),
            "duration_percentile_method": {
                "method": "exact" if self.duration_count <= PERCENTILE_RESERVOIR_SIZE else "deterministic_reservoir_sample",
                "sample_size": len(self._durations),
            },
            "sample_rate_counts": dict(sorted(self.sample_rate_counts.items())),
            "channel_counts": dict(sorted(self.channel_counts.items())),
            "audio_format_counts": dict(sorted(self.format_counts.items())),
            "language_counts": dict(sorted(self.language_counts.items())),
            "source_counts": dict(sorted(self.source_counts.items())),
            "domain_counts": dict(sorted(self.domain_counts.items())),
            "speaker_count": self.records - self.missing_optional_fields["speaker_id"],
            "missing_optional_field_counts": dict(sorted(self.missing_optional_fields.items())),
            "script_distribution": dict(sorted(self.script_counts.items())),
            "severity_counts": dict(sorted(severity_counts.items())),
            "structural_errors": severity_counts["error"],
            "media_errors": sum(1 for finding in self.findings if finding.code == "local_audio_invalid"),
            "warning_counts": severity_counts["warning"],
            "duplicate_path_count": self.duplicate_path_count,
            "duplicate_content_count": self.duplicate_content_count,
            "duplicate_transcript_count": self.duplicate_transcript_count,
            "findings": [finding.as_dict() for finding in self.findings],
        }


def summarize_manifest(
    path: str | Path,
    *,
    project_root: str | Path,
    check_audio: bool = False,
    hash_local_audio: bool = False,
    audio_decoder: Callable[[str | Path], DecodedAudio] = decode_audio,
) -> DatasetSummary:
    """Summarize a manifest while decoding at most one local audio file at a time."""

    source = Path(path).expanduser().resolve()
    root = Path(project_root).expanduser().resolve()
    # Hashing requires opening the file, so it intentionally implies the same
    # local decoder validation as --check-audio.
    check_audio = check_audio or hash_local_audio
    summary = DatasetSummary(source, check_audio, hash_local_audio)
    for event in iter_manifest_events(source):
        if event.error is not None:
            summary.findings.append(Finding("error", "manifest_record_invalid", str(event.error), event.line_number))
            continue
        assert event.record is not None
        record = event.record
        _add_record(summary, record, root, audio_decoder)
    return summary


def _add_record(
    summary: DatasetSummary,
    record: ManifestRecord,
    project_root: Path,
    audio_decoder: Callable[[str | Path], DecodedAudio],
) -> None:
    summary.records += 1
    for finding in transcript_findings(record):
        summary.findings.append(finding)
    summary.script_counts[script_classification(record.text)] += 1
    for field, counter in (("language", summary.language_counts), ("source", summary.source_counts), ("domain", summary.domain_counts)):
        value = getattr(record, field)
        if value is None:
            summary.missing_optional_fields[field] += 1
        else:
            counter[value] += 1
    for field in ("speaker_id", "recording_id", "split"):
        if getattr(record, field) is None:
            summary.missing_optional_fields[field] += 1
    locator = normalized_audio_locator(record, project_root)
    if locator in summary._seen_paths:
        summary.duplicate_path_count += 1
        summary.findings.append(Finding("warning", "duplicate_audio_path", "Audio locator is duplicated", record.line_number, display_audio_locator(record.audio_filepath)))
    summary._seen_paths.add(locator)
    transcript_hash = _hash_text(record.text)
    if transcript_hash in summary._seen_transcripts:
        summary.duplicate_transcript_count += 1
        summary.findings.append(Finding("warning", "duplicate_transcript", "Transcript text is duplicated", record.line_number, display_audio_locator(record.audio_filepath)))
    summary._seen_transcripts.add(transcript_hash)
    summary.format_counts[Path(record.audio_filepath).suffix.lower().lstrip(".") or "unknown"] += 1

    if record.is_remote:
        summary.findings.append(Finding("warning", "remote_audio_not_locally_verified", "Remote audio was not downloaded or locally verified", record.line_number, display_audio_locator(record.audio_filepath)))
        if record.duration is not None:
            summary.add_duration(record.duration)
        return
    if not summary.check_audio:
        if record.duration is not None:
            summary.add_duration(record.duration)
        return
    inspection, findings = inspect_local_audio(
        record,
        project_root=project_root,
        hash_local_audio=summary.hash_local_audio,
        duration_tolerance_seconds=DEFAULT_DURATION_TOLERANCE_SECONDS,
        audio_decoder=audio_decoder,
    )
    summary.findings.extend(findings)
    if inspection.decoded is None:
        return
    decoded = inspection.decoded
    summary.add_duration(decoded.duration_seconds)
    summary.sample_rate_counts[str(decoded.original_sample_rate)] += 1
    summary.channel_counts[str(decoded.original_channels)] += 1
    if inspection.content_hash is not None:
        if inspection.content_hash in summary._seen_hashes:
            summary.duplicate_content_count += 1
            summary.findings.append(Finding("warning", "duplicate_audio_content", "Local audio content hash is duplicated", record.line_number, display_audio_locator(record.audio_filepath)))
        summary._seen_hashes.add(inspection.content_hash)


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p90": None, "p95": None, "p99": None}
    ordered = sorted(values)
    return {key: _quantile(ordered, fraction) for key, fraction in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99))}


def _quantile(values: list[float], fraction: float) -> float:
    index = (len(values) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def _hash_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()
