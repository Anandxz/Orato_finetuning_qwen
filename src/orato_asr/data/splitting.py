"""Deterministic, group-safe stratified splitting for owner manifests.

The source import may contain dataset-specific top-level extensions.  Derived
files always use the strict canonical schema; extensions are retained beneath
``metadata`` and the source file is never rewritten.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..exceptions import ManifestError, ManifestValidationError
from .schema import TOP_LEVEL_FIELDS, ManifestRecord, parse_record

DEFAULT_CATEGORIES = (
    "hi_clean",
    "hinglish",
    "call_like",
    "numbers_entities",
)
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class SplitOptions:
    """Validated split policy."""

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    category_field: str = "eval_category"
    categories: tuple[str, ...] = DEFAULT_CATEGORIES

    def validate(self) -> None:
        ratios = (self.train_ratio, self.val_ratio, self.test_ratio)
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in ratios):
            raise ManifestError("Split ratios must be positive finite numbers")
        if not math.isclose(sum(ratios), 1.0, rel_tol=0, abs_tol=1e-9):
            raise ManifestError("Train, validation, and test ratios must sum to 1")
        if type(self.seed) is not int or self.seed < 0:
            raise ManifestError("Split seed must be a non-negative integer")
        if not isinstance(self.category_field, str) or not self.category_field.strip():
            raise ManifestError("Category field must be a non-empty string")
        if not self.categories or len(set(self.categories)) != len(self.categories):
            raise ManifestError("Split categories must be non-empty and unique")


@dataclass(frozen=True, slots=True)
class _ImportedRow:
    source_line: int
    category: str
    record: ManifestRecord
    group_identifiers: tuple[str, ...]


@dataclass(slots=True)
class _Group:
    rows: list[_ImportedRow]
    category_counts: Counter[str]
    category_durations: Counter[str]

    @property
    def duration(self) -> float:
        return float(sum(self.category_durations.values()))


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def split_owner_manifest(
    source_manifest: str | Path,
    *,
    train_output: str | Path,
    val_output: str | Path,
    test_output: str | Path,
    summary_output: str | Path,
    options: SplitOptions = SplitOptions(),
    overwrite: bool = False,
) -> dict[str, Any]:
    """Canonicalize and split every source row exactly once.

    Speaker, recording, and original-recording identifiers are unioned before
    assignment, so a connected recording group cannot cross splits.  The
    greedy assignment targets both category counts and category durations.
    """

    options.validate()
    source = Path(source_manifest).expanduser().resolve()
    if not source.is_file():
        raise ManifestError(f"Source manifest is not a regular file: {source}")
    destinations = {
        "train": _local_path(train_output),
        "val": _local_path(val_output),
        "test": _local_path(test_output),
    }
    summary_path = _local_path(summary_output)
    all_destinations = (*destinations.values(), summary_path)
    if len(set(all_destinations)) != len(all_destinations):
        raise ManifestError("Split manifest and summary output paths must be distinct")
    if source in all_destinations:
        raise ManifestError("Split outputs must not replace the immutable source manifest")
    if not overwrite:
        existing = [str(path) for path in all_destinations if path.exists()]
        if existing:
            raise ManifestError(
                "Refusing to overwrite existing split output(s) without --overwrite: "
                + ", ".join(existing)
            )

    imported = _read_source(source, options)
    groups = _build_groups(imported)
    assignments = _assign_groups(groups, imported, options)
    ordered = {
        split: _balanced_record_order(
            [row for group in assignments[split] for row in group.rows],
            categories=options.categories,
            seed=options.seed + index,
            longest_first=split == "train",
        )
        for index, split in enumerate(SPLIT_NAMES)
    }
    _validate_assignment(imported, assignments, ordered, options)
    summary = _summary(source, imported, groups, assignments, ordered, options, destinations)
    _write_bundle(ordered, destinations, summary, summary_path)
    return summary


def _read_source(source: Path, options: SplitOptions) -> list[_ImportedRow]:
    rows: list[_ImportedRow] = []
    try:
        handle = source.open("r", encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"Could not read source manifest {source}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestValidationError(
                    f"{source}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(raw, dict):
                raise ManifestValidationError(
                    f"{source}:{line_number}: source row must be a JSON object"
                )
            category = raw.get(options.category_field)
            if category not in options.categories:
                raise ManifestValidationError(
                    f"{source}:{line_number}: {options.category_field} must be one of "
                    + ", ".join(options.categories)
                )
            canonical = _canonicalize(raw, line_number, category, options)
            record = parse_record(
                canonical, manifest_path=source, line_number=line_number
            )
            rows.append(
                _ImportedRow(
                    source_line=line_number,
                    category=category,
                    record=record,
                    group_identifiers=_group_identifiers(raw, line_number),
                )
            )
    if not rows:
        raise ManifestError("Source manifest contains no records")
    present = {row.category for row in rows}
    missing = sorted(set(options.categories) - present)
    if missing:
        raise ManifestError(
            "Source manifest is missing requested split categories: " + ", ".join(missing)
        )
    return rows


def _canonicalize(
    raw: dict[str, Any],
    line_number: int,
    category: str,
    options: SplitOptions,
) -> dict[str, Any]:
    existing_metadata = raw.get("metadata", {})
    if not isinstance(existing_metadata, dict):
        raise ManifestValidationError(
            f"Source manifest line {line_number}: metadata must be a JSON object"
        )
    metadata = dict(existing_metadata)
    for key, value in raw.items():
        if key not in TOP_LEVEL_FIELDS:
            if key in metadata and metadata[key] != value:
                raise ManifestValidationError(
                    f"Source manifest line {line_number}: conflicting metadata key {key!r}"
                )
            metadata[key] = value
    if "split" in raw:
        metadata["source_split"] = raw["split"]
    metadata["source_line_number"] = line_number
    metadata[options.category_field] = category

    canonical = {
        key: value
        for key, value in raw.items()
        if key in TOP_LEVEL_FIELDS and key not in {"metadata", "split"}
    }
    canonical["metadata"] = metadata
    # The final split label is filled after assignment.
    return canonical


def _group_identifiers(raw: dict[str, Any], line_number: int) -> tuple[str, ...]:
    namespace = str(raw.get("dataset") or raw.get("source") or "unknown")
    identifiers: list[str] = []
    for field in ("speaker_id", "recording_id", "original_audio_path"):
        value = raw.get(field)
        if isinstance(value, str) and value.strip():
            identifiers.append(f"{namespace}:{field}:{value.strip()}")
    if not identifiers:
        identifiers.append(f"row:{line_number}")
    return tuple(identifiers)


def _build_groups(rows: list[_ImportedRow]) -> list[_Group]:
    sets = _DisjointSet(len(rows))
    owners: dict[str, int] = {}
    for index, row in enumerate(rows):
        for identifier in row.group_identifiers:
            previous = owners.setdefault(identifier, index)
            sets.union(index, previous)
    members: dict[int, list[_ImportedRow]] = defaultdict(list)
    for index, row in enumerate(rows):
        members[sets.find(index)].append(row)
    groups: list[_Group] = []
    for group_rows in members.values():
        counts: Counter[str] = Counter()
        durations: Counter[str] = Counter()
        for row in group_rows:
            counts[row.category] += 1
            durations[row.category] += float(row.record.duration or 0.0)
        groups.append(_Group(group_rows, counts, durations))
    return groups


def _assign_groups(
    groups: list[_Group],
    rows: list[_ImportedRow],
    options: SplitOptions,
) -> dict[str, list[_Group]]:
    ratios = dict(
        zip(SPLIT_NAMES, (options.train_ratio, options.val_ratio, options.test_ratio))
    )
    total_counts = Counter(row.category for row in rows)
    total_durations: Counter[str] = Counter()
    for row in rows:
        total_durations[row.category] += float(row.record.duration or 0.0)
    targets = {
        split: {
            "counts": {cat: total_counts[cat] * ratios[split] for cat in options.categories},
            "durations": {
                cat: total_durations[cat] * ratios[split] for cat in options.categories
            },
        }
        for split in SPLIT_NAMES
    }
    assigned = {split: [] for split in SPLIT_NAMES}
    current_counts = {split: Counter() for split in SPLIT_NAMES}
    current_durations = {split: Counter() for split in SPLIT_NAMES}
    randomizer = random.Random(options.seed)
    tie_breakers = {id(group): randomizer.random() for group in groups}
    ordered_groups = sorted(
        groups,
        key=lambda group: (
            -group.duration,
            -len(group.rows),
            tie_breakers[id(group)],
        ),
    )
    split_order = list(SPLIT_NAMES)
    randomizer.shuffle(split_order)
    for group in ordered_groups:
        best = min(
            split_order,
            key=lambda split: _assignment_score(
                split,
                group,
                current_counts,
                current_durations,
                targets,
                options.categories,
            ),
        )
        assigned[best].append(group)
        current_counts[best].update(group.category_counts)
        current_durations[best].update(group.category_durations)
    return assigned


def _assignment_score(
    candidate: str,
    group: _Group,
    counts: dict[str, Counter[str]],
    durations: dict[str, Counter[str]],
    targets: dict[str, dict[str, dict[str, float]]],
    categories: tuple[str, ...],
) -> float:
    score = 0.0
    for split in SPLIT_NAMES:
        for category in categories:
            count = counts[split][category]
            duration = durations[split][category]
            if split == candidate:
                count += group.category_counts[category]
                duration += group.category_durations[category]
            count_target = targets[split]["counts"][category]
            duration_target = targets[split]["durations"][category]
            score += ((count - count_target) / max(1.0, count_target)) ** 2
            # Audio duration is the training exposure unit.  Weight it above row
            # count so a split cannot look balanced merely by receiving many
            # unusually short clips.
            score += 4.0 * (
                (duration - duration_target) / max(1.0, duration_target)
            ) ** 2
    return score


def _balanced_record_order(
    rows: list[_ImportedRow],
    *,
    categories: tuple[str, ...],
    seed: int,
    longest_first: bool,
) -> list[_ImportedRow]:
    randomizer = random.Random(seed)
    buckets: dict[str, deque[_ImportedRow]] = {}
    for category in categories:
        values = [row for row in rows if row.category == category]
        randomizer.shuffle(values)
        buckets[category] = deque(values)
    ordered: list[_ImportedRow] = []
    if longest_first and rows:
        longest = max(rows, key=lambda row: (float(row.record.duration or 0), -row.source_line))
        buckets[longest.category].remove(longest)
        ordered.append(longest)
    while any(buckets.values()):
        for category in categories:
            if buckets[category]:
                ordered.append(buckets[category].popleft())
    return ordered


def _validate_assignment(
    imported: list[_ImportedRow],
    assignments: dict[str, list[_Group]],
    ordered: dict[str, list[_ImportedRow]],
    options: SplitOptions,
) -> None:
    source_lines = [row.source_line for row in imported]
    output_lines = [row.source_line for split in SPLIT_NAMES for row in ordered[split]]
    if len(output_lines) != len(source_lines) or set(output_lines) != set(source_lines):
        raise ManifestError("Split assignment did not preserve every source row exactly once")
    group_splits: dict[int, set[str]] = defaultdict(set)
    for split, groups in assignments.items():
        for group in groups:
            group_splits[id(group)].add(split)
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ManifestError("A linked speaker/recording group crossed split boundaries")
    for split in SPLIT_NAMES:
        present = {row.category for row in ordered[split]}
        missing = set(options.categories) - present
        if missing:
            raise ManifestError(
                f"Split {split} is missing categories: {', '.join(sorted(missing))}"
            )


def _summary(
    source: Path,
    imported: list[_ImportedRow],
    groups: list[_Group],
    assignments: dict[str, list[_Group]],
    ordered: dict[str, list[_ImportedRow]],
    options: SplitOptions,
    destinations: dict[str, Path],
) -> dict[str, Any]:
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    result: dict[str, Any] = {
        "status": "success",
        "source_manifest": str(source),
        "source_manifest_sha256": source_sha,
        "seed": options.seed,
        "category_field": options.category_field,
        "categories": list(options.categories),
        "ratios": {
            "train": options.train_ratio,
            "val": options.val_ratio,
            "test": options.test_ratio,
        },
        "source_rows": len(imported),
        "source_duration_seconds": sum(float(row.record.duration or 0) for row in imported),
        "linked_groups": len(groups),
        "group_leakage_count": 0,
        "splits": {},
    }
    for split in SPLIT_NAMES:
        rows = ordered[split]
        counts = Counter(row.category for row in rows)
        durations: Counter[str] = Counter()
        for row in rows:
            durations[row.category] += float(row.record.duration or 0)
        result["splits"][split] = {
            "output_manifest": str(destinations[split]),
            "rows": len(rows),
            "duration_seconds": sum(durations.values()),
            "duration_hours": sum(durations.values()) / 3600,
            "linked_groups": len(assignments[split]),
            "category_counts": {cat: counts[cat] for cat in options.categories},
            "category_duration_seconds": {
                cat: durations[cat] for cat in options.categories
            },
        }
    return result


def _write_bundle(
    ordered: dict[str, list[_ImportedRow]],
    destinations: dict[str, Path],
    summary: dict[str, Any],
    summary_path: Path,
) -> None:
    temporary_paths: list[tuple[Path, Path]] = []
    try:
        for split in SPLIT_NAMES:
            destination = destinations[split]
            destination.parent.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            )
            temporary = Path(handle.name)
            try:
                for row in ordered[split]:
                    payload = row.record.as_dict()
                    payload["split"] = split
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                handle.close()
            temporary_paths.append((temporary, destination))
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{summary_path.name}.",
            suffix=".tmp",
            dir=summary_path.parent,
            delete=False,
        )
        temporary = Path(handle.name)
        try:
            json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()
        temporary_paths.append((temporary, summary_path))
        for temporary, destination in temporary_paths:
            os.replace(temporary, destination)
    except OSError as exc:
        raise ManifestError(f"Could not write split manifest bundle: {exc}") from exc
    finally:
        for temporary, _destination in temporary_paths:
            temporary.unlink(missing_ok=True)


def _local_path(value: str | Path) -> Path:
    raw = str(value).strip()
    parsed = urlsplit(raw)
    if not raw or parsed.scheme or parsed.netloc:
        raise ManifestError("Split outputs must be explicit local paths")
    return Path(raw).expanduser().resolve()
