"""Deterministic, bounded-metadata manifest subset selection."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..exceptions import ManifestError
from .manifest import write_json_atomic, write_manifest
from .schema import ManifestRecord, parse_record


@dataclass(frozen=True, slots=True)
class SelectionOptions:
    max_samples: int | None = None
    max_duration_seconds: float | None = None
    min_duration_seconds: float | None = None
    maximum_duration_seconds: float | None = None
    source: str | None = None
    domain: str | None = None
    language: str | None = None
    seed: int = 0
    shuffled: bool = False

    def validate(self) -> None:
        if self.max_samples is None and self.max_duration_seconds is None:
            raise ManifestError("Selection requires --max-samples, --max-seconds, or --max-hours")
        for field in ("max_samples",):
            value = getattr(self, field)
            if value is not None and (type(value) is not int or value <= 0):
                raise ManifestError(f"{field} must be a positive integer")
        for field in ("max_duration_seconds", "min_duration_seconds", "maximum_duration_seconds"):
            value = getattr(self, field)
            if value is not None and (type(value) not in (int, float) or value <= 0):
                raise ManifestError(f"{field} must be a positive number")
        if (
            self.min_duration_seconds is not None
            and self.maximum_duration_seconds is not None
            and self.min_duration_seconds > self.maximum_duration_seconds
        ):
            raise ManifestError("minimum duration cannot exceed maximum duration")


@dataclass(frozen=True, slots=True)
class _Candidate:
    line_number: int
    offset: int | None
    duration: float | None
    source: str | None
    domain: str | None
    language: str | None


def select_manifest(
    manifest: str | Path,
    output: str | Path,
    *,
    options: SelectionOptions,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a deterministic derived manifest and its JSON sidecar summary."""

    options.validate()
    source = Path(manifest).expanduser().resolve()
    # Keep only small row metadata and byte offsets in memory.  In particular,
    # do not decode audio or retain transcript text merely to choose a subset.
    candidates = _indexed_candidates(source)
    if options.shuffled:
        random.Random(options.seed).shuffle(candidates)
    selected: list[ManifestRecord] = []
    eligible_count = 0
    skipped_duration_limit = 0
    total_duration = 0.0
    for candidate in candidates:
        if not _matches(candidate, options):
            continue
        eligible_count += 1
        # Continue scanning lightweight metadata after the sample cap so the
        # sidecar reports the actual eligible population, not just the prefix
        # needed to construct the result.
        if options.max_samples is not None and len(selected) >= options.max_samples:
            continue
        if options.max_duration_seconds is not None and candidate.duration is None:
            raise ManifestError(
                f"{source}:{candidate.line_number}: duration-limited selection requires a declared duration"
            )
        duration = candidate.duration or 0.0
        if options.max_duration_seconds is not None and total_duration + duration > options.max_duration_seconds:
            skipped_duration_limit += 1
            continue
        record = _record_for_candidate(source, candidate)
        selected.append(record)
        total_duration += duration

    destination = write_manifest(iter(selected), output, overwrite=overwrite)
    sidecar = destination.with_suffix(destination.suffix + ".summary.json")
    report = {
        "source_manifest": str(source),
        "output_manifest": str(destination),
        "selection": {
            "max_samples": options.max_samples,
            "max_duration_seconds": options.max_duration_seconds,
            "min_duration_seconds": options.min_duration_seconds,
            "maximum_duration_seconds": options.maximum_duration_seconds,
            "source": options.source,
            "domain": options.domain,
            "language": options.language,
            "seed": options.seed,
            "order": "shuffled" if options.shuffled else "source",
        },
        "eligible_records": eligible_count,
        "selected_records": len(selected),
        "selected_duration_seconds": total_duration,
        "selected_audio_hours": total_duration / 3600,
        "skipped_for_duration_limit": skipped_duration_limit,
        "source_smaller_than_requested": _source_smaller_than_requested(options, len(selected), total_duration),
    }
    write_json_atomic(report, sidecar, overwrite=overwrite)
    return report


def _indexed_candidates(source: Path) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    try:
        with source.open("rb") as handle:
            line_number = 0
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                line_number += 1
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ManifestError(f"{source}:{line_number}: invalid JSON while indexing selection: {exc}") from exc
                record = parse_record(raw, manifest_path=source, line_number=line_number)
                candidates.append(_candidate_from_record(record, offset=offset))
    except OSError as exc:
        raise ManifestError(f"Could not index manifest {source}: {exc}") from exc
    return candidates


def _candidate_from_record(record: ManifestRecord, *, offset: int | None) -> _Candidate:
    return _Candidate(
        line_number=record.line_number or 0,
        offset=offset,
        duration=record.duration,
        source=record.source,
        domain=record.domain,
        language=record.language,
    )


def _matches(candidate: _Candidate, options: SelectionOptions) -> bool:
    if options.source is not None and candidate.source != options.source:
        return False
    if options.domain is not None and candidate.domain != options.domain:
        return False
    if options.language is not None and candidate.language != options.language:
        return False
    if options.min_duration_seconds is not None and (candidate.duration is None or candidate.duration < options.min_duration_seconds):
        return False
    if options.maximum_duration_seconds is not None and (candidate.duration is None or candidate.duration > options.maximum_duration_seconds):
        return False
    return True


def _record_for_candidate(source: Path, candidate: _Candidate) -> ManifestRecord:
    if candidate.offset is None:
        raise ManifestError(f"Selection index is missing byte offset for line {candidate.line_number}")
    try:
        with source.open("rb") as handle:
            handle.seek(candidate.offset)
            line = handle.readline().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ManifestError(f"Could not recover manifest record at line {candidate.line_number}: {exc}") from exc
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Could not parse indexed record at line {candidate.line_number}: {exc}") from exc
    return parse_record(raw, manifest_path=source, line_number=candidate.line_number)


def _source_smaller_than_requested(options: SelectionOptions, selected: int, duration: float) -> bool:
    if options.max_samples is not None and selected < options.max_samples:
        return True
    return bool(options.max_duration_seconds is not None and duration < options.max_duration_seconds)
