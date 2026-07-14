"""Leakage-oriented overlap checks between train and evaluation manifests."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..exceptions import ManifestError
from .manifest import iter_manifest
from .schema import ManifestRecord, resolve_local_audio_path
from .validation import normalized_audio_locator


@dataclass(frozen=True, slots=True)
class OverlapExample:
    category: str
    train_line: int
    eval_line: int
    value: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "train_line": self.train_line,
            "eval_line": self.eval_line,
            "value": self.value,
        }


@dataclass(slots=True)
class OverlapReport:
    train_manifest: Path
    eval_manifest: Path
    hash_local_audio: bool
    disallow_speaker_overlap: bool
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    examples: list[OverlapExample] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def prohibited_count(self) -> int:
        categories = ("audio_path", "audio_content", "recording_id")
        prohibited = sum(self.counts.get(category, 0) for category in categories)
        if self.disallow_speaker_overlap:
            prohibited += self.counts.get("speaker_id", 0)
        return prohibited

    def add(self, category: str, train_line: int, eval_line: int, value: str) -> None:
        self.counts[category] += 1
        if len(self.examples) < 100:
            self.examples.append(OverlapExample(category, train_line, eval_line, value))

    def as_dict(self) -> dict[str, Any]:
        return {
            "train_manifest": str(self.train_manifest),
            "eval_manifest": str(self.eval_manifest),
            "hash_local_audio": self.hash_local_audio,
            "disallow_speaker_overlap": self.disallow_speaker_overlap,
            "overlap_counts": dict(sorted(self.counts.items())),
            "prohibited_overlap_count": self.prohibited_count,
            "examples": [example.as_dict() for example in self.examples],
            "warnings": self.warnings,
        }


def check_overlap(
    train_manifest: str | Path,
    eval_manifest: str | Path,
    *,
    project_root: str | Path,
    hash_local_audio: bool = False,
    disallow_speaker_overlap: bool = False,
) -> OverlapReport:
    """Compare lightweight identifiers; remote locators are never fetched."""

    train_path = Path(train_manifest).expanduser().resolve()
    evaluation_path = Path(eval_manifest).expanduser().resolve()
    root = Path(project_root).expanduser().resolve()
    report = OverlapReport(train_path, evaluation_path, hash_local_audio, disallow_speaker_overlap)
    indices: dict[str, dict[str, list[ManifestRecord]]] = {
        "audio_path": defaultdict(list),
        "recording_id": defaultdict(list),
        "speaker_id": defaultdict(list),
        "transcript": defaultdict(list),
        "audio_content": defaultdict(list),
    }
    for record in iter_manifest(train_path):
        indices["audio_path"][normalized_audio_locator(record, root)].append(record)
        if record.recording_id is not None:
            indices["recording_id"][record.recording_id].append(record)
        if record.speaker_id is not None:
            indices["speaker_id"][record.speaker_id].append(record)
        indices["transcript"][_text_hash(record.text)].append(record)
        if hash_local_audio:
            content_hash = _optional_local_hash(record, root, report)
            if content_hash is not None:
                indices["audio_content"][content_hash].append(record)

    for record in iter_manifest(evaluation_path):
        _compare(report, "audio_path", indices["audio_path"], normalized_audio_locator(record, root), record)
        if record.recording_id is not None:
            _compare(report, "recording_id", indices["recording_id"], record.recording_id, record)
        if record.speaker_id is not None:
            _compare(report, "speaker_id", indices["speaker_id"], record.speaker_id, record)
        _compare(report, "transcript", indices["transcript"], _text_hash(record.text), record)
        if hash_local_audio:
            content_hash = _optional_local_hash(record, root, report)
            if content_hash is not None:
                _compare(report, "audio_content", indices["audio_content"], content_hash, record)
    return report


def _compare(
    report: OverlapReport,
    category: str,
    index: dict[str, list[ManifestRecord]],
    key: str,
    evaluation_record: ManifestRecord,
) -> None:
    for train_record in index.get(key, []):
        # IDs and remote locators may themselves be private.  The report needs
        # a stable correlation token, not the source value.
        report.add(
            category,
            train_record.line_number or 0,
            evaluation_record.line_number or 0,
            hashlib.sha256(key.encode("utf-8")).hexdigest()[:16],
        )


def _optional_local_hash(record: ManifestRecord, project_root: Path, report: OverlapReport) -> str | None:
    local_path = resolve_local_audio_path(record, project_root)
    if local_path is None:
        warning = f"Remote audio at line {record.line_number} was not content-hashed"
        if warning not in report.warnings:
            report.warnings.append(warning)
        return None
    try:
        digest = hashlib.sha256()
        with local_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as exc:
        raise ManifestError(f"Could not hash local audio for overlap check: {local_path}: {exc}") from exc


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
