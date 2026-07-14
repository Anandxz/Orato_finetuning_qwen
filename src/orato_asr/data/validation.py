"""Structural and local-media validation for canonical manifests."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urlsplit

from ..audio import DecodedAudio, decode_audio
from ..exceptions import AudioValidationError, DependencyError, ManifestError
from .manifest import ManifestEvent, iter_manifest_events
from .schema import (
    ManifestRecord,
    display_audio_locator,
    is_remote_locator,
    lexical_content,
    resolve_local_audio_path,
    script_classification,
)

DEFAULT_DURATION_TOLERANCE_SECONDS = 0.25
_REPEATED_PUNCTUATION = re.compile(r"([!?…。,.])\1{3,}")
_REPEATED_CHARACTER = re.compile(r"(\S)\1{5,}")
_LONG_WHITESPACE = re.compile(r"\s{4,}")


@dataclass(frozen=True, slots=True)
class Finding:
    severity: str
    code: str
    message: str
    line_number: int | None = None
    audio_filepath: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "line_number": self.line_number,
            "audio_filepath": self.audio_filepath,
        }


@dataclass(frozen=True, slots=True)
class AudioInspection:
    local_path: Path | None
    decoded: DecodedAudio | None
    content_hash: str | None

    @property
    def duration_seconds(self) -> float | None:
        return self.decoded.duration_seconds if self.decoded is not None else None


@dataclass(slots=True)
class ValidationReport:
    manifest: Path
    check_audio: bool
    hash_local_audio: bool
    duration_tolerance_seconds: float
    records: int = 0
    remote_records: int = 0
    findings: list[Finding] = field(default_factory=list)
    duplicate_path_count: int = 0
    duplicate_content_count: int = 0
    duplicate_transcript_count: int = 0

    @property
    def severity_counts(self) -> Counter[str]:
        return Counter(finding.severity for finding in self.findings)

    @property
    def has_errors(self) -> bool:
        return bool(self.severity_counts["error"])

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest": str(self.manifest),
            "check_audio": self.check_audio,
            "hash_local_audio": self.hash_local_audio,
            "duration_tolerance_seconds": self.duration_tolerance_seconds,
            "records": self.records,
            "remote_records": self.remote_records,
            "severity_counts": dict(sorted(self.severity_counts.items())),
            "duplicate_path_count": self.duplicate_path_count,
            "duplicate_content_count": self.duplicate_content_count,
            "duplicate_transcript_count": self.duplicate_transcript_count,
            "findings": [finding.as_dict() for finding in self.findings],
        }


def validate_manifest(
    path: str | Path,
    *,
    project_root: str | Path,
    check_audio: bool = False,
    hash_local_audio: bool = False,
    duration_tolerance_seconds: float = DEFAULT_DURATION_TOLERANCE_SECONDS,
    audio_decoder: Callable[[str | Path], DecodedAudio] = decode_audio,
) -> ValidationReport:
    """Validate every JSONL row without loading raw audio beyond one sample."""

    if not math.isfinite(duration_tolerance_seconds) or duration_tolerance_seconds < 0:
        raise ManifestError("duration tolerance must be a non-negative finite number")
    source = Path(path).expanduser().resolve()
    root = Path(project_root).expanduser().resolve()
    # Hashing is a local-media operation.  Make the flag useful on its own
    # rather than silently returning no content-hash findings.
    check_audio = check_audio or hash_local_audio
    report = ValidationReport(
        manifest=source,
        check_audio=check_audio,
        hash_local_audio=hash_local_audio,
        duration_tolerance_seconds=duration_tolerance_seconds,
    )
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    seen_transcripts: set[str] = set()

    for event in iter_manifest_events(source):
        if event.error is not None:
            report.add(_error_from_event(event))
            continue
        assert event.record is not None
        record = event.record
        report.records += 1
        for finding in transcript_findings(record):
            report.add(finding)

        locator = normalized_audio_locator(record, root)
        if locator in seen_paths:
            report.duplicate_path_count += 1
            report.add(_finding("warning", "duplicate_audio_path", "Audio locator is duplicated", record))
        else:
            seen_paths.add(locator)
        transcript_digest = _digest_text(record.text)
        if transcript_digest in seen_transcripts:
            report.duplicate_transcript_count += 1
            report.add(_finding("warning", "duplicate_transcript", "Transcript text is duplicated", record))
        else:
            seen_transcripts.add(transcript_digest)

        if record.is_remote:
            report.remote_records += 1
            report.add(_finding("warning", "remote_audio_not_locally_verified", "Remote audio was not downloaded or locally verified", record))
            continue
        if not check_audio:
            continue
        inspection, findings = inspect_local_audio(
            record,
            project_root=root,
            hash_local_audio=hash_local_audio,
            duration_tolerance_seconds=duration_tolerance_seconds,
            audio_decoder=audio_decoder,
        )
        for finding in findings:
            report.add(finding)
        if inspection.content_hash is not None:
            if inspection.content_hash in seen_hashes:
                report.duplicate_content_count += 1
                report.add(_finding("warning", "duplicate_audio_content", "Local audio content hash is duplicated", record))
            else:
                seen_hashes.add(inspection.content_hash)
    return report


def inspect_local_audio(
    record: ManifestRecord,
    *,
    project_root: Path,
    hash_local_audio: bool = False,
    duration_tolerance_seconds: float = DEFAULT_DURATION_TOLERANCE_SECONDS,
    audio_decoder: Callable[[str | Path], DecodedAudio] = decode_audio,
) -> tuple[AudioInspection, list[Finding]]:
    """Inspect one local record through the existing decoder and return findings."""

    if record.is_remote:
        return AudioInspection(None, None, None), [
            _finding("warning", "remote_audio_not_locally_verified", "Remote audio was not downloaded or locally verified", record)
        ]
    local_path = resolve_local_audio_path(record, project_root)
    assert local_path is not None
    try:
        decoded = audio_decoder(local_path)
    except (AudioValidationError, DependencyError, OSError) as exc:
        return AudioInspection(local_path, None, None), [
            _finding("error", "local_audio_invalid", str(exc), record)
        ]

    findings: list[Finding] = []
    if decoded.original_channels > 1:
        findings.append(_finding("warning", "stereo_or_multichannel_audio", "Audio will be downmixed in memory", record))
    if decoded.original_sample_rate != 16_000:
        findings.append(_finding("warning", "non_16khz_audio", "Audio will be resampled to 16 kHz in memory", record))
    if record.duration is not None:
        difference = abs(record.duration - decoded.duration_seconds)
        if difference > duration_tolerance_seconds:
            findings.append(
                _finding(
                    "warning",
                    "duration_mismatch",
                    f"Declared duration differs from measured duration by {difference:.3f} seconds",
                    record,
                )
            )
    try:
        content_hash = _file_hash(local_path) if hash_local_audio else None
    except AudioValidationError as exc:
        findings.append(_finding("error", "local_audio_hash_failed", str(exc), record))
        content_hash = None
    return AudioInspection(local_path, decoded, content_hash), findings


def transcript_findings(record: ManifestRecord) -> list[Finding]:
    """Warn on policy anomalies without rewriting canonical transcript text."""

    findings: list[Finding] = []
    script = script_classification(record.text)
    language = (record.language or "").casefold()
    if language in {"hi", "hindi"} and script == "latin_only":
        findings.append(_finding("warning", "possibly_romanized_hindi", "Hindi-labelled transcript has Latin script but no Devanagari", record))
    if _REPEATED_PUNCTUATION.search(record.text):
        findings.append(_finding("warning", "excessive_repeated_punctuation", "Transcript has excessive repeated punctuation", record))
    if _REPEATED_CHARACTER.search(record.text):
        findings.append(_finding("warning", "excessive_repeated_character", "Transcript has excessive repeated characters", record))
    if _LONG_WHITESPACE.search(record.text):
        findings.append(_finding("warning", "suspicious_whitespace", "Transcript has long whitespace runs", record))
    if not lexical_content(record.text):
        findings.append(_finding("warning", "empty_after_lexical_normalization", "Transcript has no letters or numbers after lexical normalization", record))
    if record.duration is not None and len(record.text.strip()) / record.duration > 60:
        findings.append(_finding("warning", "text_duration_mismatch", "Transcript length appears unusually high for declared duration", record))
    return findings


def normalized_audio_locator(record: ManifestRecord, project_root: Path) -> str:
    """Create a stable local/remote duplicate key without contacting remote storage."""

    if not is_remote_locator(record.audio_filepath):
        local = resolve_local_audio_path(record, project_root)
        assert local is not None
        return f"local:{local}"
    parsed = urlsplit(record.audio_filepath)
    return f"remote:{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path}"


def _error_from_event(event: ManifestEvent) -> Finding:
    assert event.error is not None
    return Finding("error", "manifest_record_invalid", str(event.error), event.line_number, None)


def _finding(severity: str, code: str, message: str, record: ManifestRecord) -> Finding:
    return Finding(severity, code, message, record.line_number, display_audio_locator(record.audio_filepath))


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise AudioValidationError(f"Could not hash local audio {path}: {exc}") from exc
    return digest.hexdigest()
