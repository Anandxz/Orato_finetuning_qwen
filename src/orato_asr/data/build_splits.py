"""Configuration-driven, leakage-safe split generation across processed datasets."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit

import yaml

from ..exceptions import ManifestError, ManifestValidationError, StorageError
from ..paths import DataPathResolver, StorageSettings, normalize_logical_path
from .manifest import manifest_fingerprint, write_json_atomic
from .schema import OPTIONAL_FIELDS, REQUIRED_FIELDS, ManifestRecord, parse_record

SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_GROUPING = (
    "session_id",
    "call_id",
    "source_id",
    "video_id",
    "speaker_id",
    "recording_id",
    "audio_filepath",
)
SUPPORTED_AUDIO_EXTENSIONS = frozenset({".wav", ".flac"})
_NUMBER_WORDS = frozenset(
    {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "शून्य", "एक", "दो", "तीन", "चार", "पांच", "पाँच", "छह", "सात", "आठ", "नौ", "दस",
    }
)


@dataclass(frozen=True, slots=True)
class SplitBuildConfig:
    source_path: Path
    name: str
    version: str
    seed: int
    ratios: Mapping[str, float]
    storage: StorageSettings
    explicit_manifests: tuple[str, ...]
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    defaults: Mapping[str, Mapping[str, Any]]
    grouping_priority: tuple[str, ...]
    stratification_fields: tuple[str, ...]
    duration_buckets: tuple[tuple[str, float | None, float | None], ...]
    require_audio_exists: bool
    fail_on_duplicate_audio_path: bool

    @property
    def output_directory(self) -> Path:
        return self.storage.split_root / self.name / self.version

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "seed": self.seed,
            "ratios": dict(self.ratios),
            "manifests": list(self.explicit_manifests),
            "datasets": {"include": list(self.include), "exclude": list(self.exclude)},
            "defaults": self.defaults,
            "grouping_priority": list(self.grouping_priority),
            "stratification_fields": list(self.stratification_fields),
            "duration_buckets": [list(value) for value in self.duration_buckets],
            "validation": {
                "require_audio_exists": self.require_audio_exists,
                "fail_on_duplicate_audio_path": self.fail_on_duplicate_audio_path,
            },
        }


@dataclass(frozen=True, slots=True)
class _Row:
    record: ManifestRecord
    dataset: str
    record_id: str
    group_identifiers: tuple[str, ...]
    features: tuple[str, ...]
    duration_bucket: str
    contains_number: bool


@dataclass(slots=True)
class _Group:
    key: str
    rows: list[_Row]
    duration: float
    feature_counts: Counter[str]
    feature_durations: Counter[str]


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parents = list(range(size))

    def find(self, value: int) -> int:
        while self.parents[value] != value:
            self.parents[value] = self.parents[self.parents[value]]
            value = self.parents[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parents[right] = left


def load_split_config(
    path: str | Path,
    *,
    project_root: Path,
    data_root: str | None = None,
    split_root: str | Path | None = None,
    seed: int | None = None,
    train_ratio: float | None = None,
    validation_ratio: float | None = None,
    test_ratio: float | None = None,
) -> SplitBuildConfig:
    """Load and validate a split YAML with optional CLI root/seed overrides."""

    source = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ManifestError(f"Could not load split configuration {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ManifestError("Split configuration must be a YAML mapping")
    name = _identifier(raw.get("name"), "name")
    version = _identifier(raw.get("version"), "version")
    selected_seed = raw.get("seed", 42) if seed is None else seed
    if type(selected_seed) is not int or selected_seed < 0:
        raise ManifestError("Split seed must be a non-negative integer")
    ratios_raw = raw.get("ratios", {})
    if not isinstance(ratios_raw, dict):
        raise ManifestError("ratios must be a mapping")
    ratios = {
        "train": ratios_raw.get("train", 0.8) if train_ratio is None else train_ratio,
        "validation": (
            ratios_raw.get("validation", ratios_raw.get("val", 0.1))
            if validation_ratio is None
            else validation_ratio
        ),
        "test": ratios_raw.get("test", 0.1) if test_ratio is None else test_ratio,
    }
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in ratios.values()):
        raise ManifestError("All split ratios must be positive finite numbers")
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9):
        raise ManifestError("Train, validation, and test ratios must sum to 1")

    storage_raw = raw.get("storage", {})
    if not isinstance(storage_raw, dict):
        raise ManifestError("storage must be a mapping")
    configured_data_root = data_root or storage_raw.get("data_root")
    configured_split_root = split_root or storage_raw.get("split_root")
    configured_cache_root = storage_raw.get("cache_root")
    configured_backend = storage_raw.get("backend")
    storage = StorageSettings.from_environment(
        project_root=project_root,
        data_root=str(configured_data_root) if configured_data_root is not None else None,
        split_root=configured_split_root,
        cache_root=configured_cache_root,
        backend=str(configured_backend) if configured_backend is not None else None,
    )

    manifests = raw.get("manifests", [])
    if not isinstance(manifests, list) or any(not isinstance(value, str) for value in manifests):
        raise ManifestError("manifests must be a list of paths or az:// locators")
    datasets = raw.get("datasets", {})
    if not isinstance(datasets, dict):
        raise ManifestError("datasets must be a mapping")
    include = _string_tuple(datasets.get("include", ["*"]), "datasets.include")
    exclude = _string_tuple(datasets.get("exclude", []), "datasets.exclude")
    defaults = datasets.get("defaults", {})
    if not isinstance(defaults, dict) or any(not isinstance(value, dict) for value in defaults.values()):
        raise ManifestError("datasets.defaults must map dataset names to mappings")

    grouping = raw.get("grouping", {})
    if not isinstance(grouping, dict):
        raise ManifestError("grouping must be a mapping")
    priority = _string_tuple(grouping.get("priority", list(DEFAULT_GROUPING)), "grouping.priority")
    if "audio_filepath" not in priority:
        raise ManifestError("grouping.priority must end with or include audio_filepath")
    stratification = raw.get("stratification", {})
    if not isinstance(stratification, dict):
        raise ManifestError("stratification must be a mapping")
    fields = _string_tuple(
        stratification.get(
            "fields", ["dataset", "language", "domain", "contains_number", "duration_bucket"]
        ),
        "stratification.fields",
    )
    buckets_raw = stratification.get(
        "duration_buckets",
        {"short": [None, 5.0], "medium": [5.0, 15.0], "long": [15.0, None]},
    )
    buckets = _parse_duration_buckets(buckets_raw)
    validation = raw.get("validation", {})
    if not isinstance(validation, dict):
        raise ManifestError("validation must be a mapping")
    return SplitBuildConfig(
        source_path=source,
        name=name,
        version=version,
        seed=selected_seed,
        ratios=ratios,
        storage=storage,
        explicit_manifests=tuple(manifests),
        include=include,
        exclude=exclude,
        defaults=defaults,
        grouping_priority=priority,
        stratification_fields=fields,
        duration_buckets=buckets,
        require_audio_exists=_bool_option(validation, "require_audio_exists", False),
        fail_on_duplicate_audio_path=_bool_option(
            validation, "fail_on_duplicate_audio_path", True
        ),
    )


def build_splits(config: SplitBuildConfig, *, overwrite: bool = False) -> dict[str, Any]:
    """Discover, normalize, group, assign, validate, and atomically write a split."""

    output = config.output_directory
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise ManifestError(
            f"Refusing to overwrite split version without --overwrite: {output}"
        )
    resolver = DataPathResolver(config.storage)
    manifest_locators = _select_manifests(config, resolver)
    if not manifest_locators:
        raise ManifestError("No processed dataset manifests matched the split configuration")
    rows, diagnostics, source_fingerprints = _read_sources(
        config, resolver, manifest_locators
    )
    if diagnostics["errors"]:
        examples = "; ".join(diagnostics["errors"][:5])
        raise ManifestValidationError(
            f"Split input validation found {len(diagnostics['errors'])} error(s): {examples}"
        )
    groups = _build_groups(rows)
    assignments = _assign_groups(groups, rows, config)
    _validate_assignments(rows, assignments)
    ordered = {
        split: sorted(
            (row for group in assignments[split] for row in group.rows),
            key=lambda row: (row.dataset, row.record_id, row.record.audio_filepath),
        )
        for split in SPLIT_NAMES
    }
    row_group_keys = {
        id(row): group.key
        for split in SPLIT_NAMES
        for group in assignments[split]
        for row in group.rows
    }

    output.mkdir(parents=True, exist_ok=True)
    manifests = {split: output / f"{split}.jsonl" for split in SPLIT_NAMES}
    for destination in manifests.values():
        if destination.exists() and overwrite:
            destination.unlink()
    for split, destination in manifests.items():
        _write_rows(ordered[split], split, destination, row_group_keys)
    output_checksums = {
        split: manifest_fingerprint(path) for split, path in manifests.items()
    }
    fingerprint = _final_fingerprint(config, source_fingerprints, output_checksums)
    warnings = _representation_warnings(assignments)
    report = _build_report(
        config,
        rows,
        groups,
        assignments,
        diagnostics,
        warnings,
        source_fingerprints,
        output_checksums,
        fingerprint,
    )
    report_path = output / "split_report.json"
    config_path = output / "split_config.yaml"
    fingerprint_path = output / "split_fingerprint.txt"
    for path in (report_path, config_path, fingerprint_path):
        if path.exists() and overwrite:
            path.unlink()
    write_json_atomic(report, report_path)
    recorded_config = config.fingerprint_payload()
    recorded_config["generation"] = {
        "timestamp_utc": report["generation_timestamp_utc"],
        "source_manifest_fingerprints": dict(source_fingerprints),
        "git_commit": report["git_commit"],
    }
    _write_text_atomic(
        config_path,
        yaml.safe_dump(recorded_config, allow_unicode=True, sort_keys=False),
    )
    _write_text_atomic(fingerprint_path, fingerprint + "\n")
    return report


def validate_split_directory(
    split_directory: str | Path,
    *,
    project_root: Path,
    data_root: str | None = None,
    check_audio: bool = False,
) -> dict[str, Any]:
    """Validate manifest integrity, grouping isolation, checksums, and resolution."""

    directory = Path(split_directory).expanduser().resolve()
    report_path = directory / "split_report.json"
    try:
        expected = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Could not read {report_path}: {exc}") from exc
    settings = StorageSettings.from_environment(project_root=project_root, data_root=data_root)
    resolver = DataPathResolver(settings)
    errors: list[str] = []
    seen_audio: dict[str, str] = {}
    group_splits: dict[str, set[str]] = defaultdict(set)
    checksums: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split in SPLIT_NAMES:
        path = directory / f"{split}.jsonl"
        if not path.is_file():
            errors.append(f"Missing manifest: {path}")
            continue
        checksums[split] = manifest_fingerprint(path)
        count = 0
        for line_number, raw in _iter_raw_jsonl(path):
            try:
                record = parse_record(raw, manifest_path=path, line_number=line_number)
            except ManifestValidationError as exc:
                errors.append(str(exc))
                continue
            count += 1
            if record.split != split:
                errors.append(f"{path}:{line_number}: split field is not {split!r}")
            previous = seen_audio.get(record.audio_filepath)
            if previous is not None:
                errors.append(
                    f"Duplicate audio path in split bundle: {record.audio_filepath} ({previous}, {split})"
                )
            else:
                seen_audio[record.audio_filepath] = split
            metadata = dict(record.metadata)
            group_key = metadata.get("split_group_key")
            if isinstance(group_key, str):
                group_splits[group_key].add(split)
            try:
                normalize_logical_path(record.audio_filepath)
                if check_audio:
                    local = resolver.resolve(record.audio_filepath)
                    if not local.is_file() or local.stat().st_size <= 0:
                        errors.append(f"Missing or empty audio: {record.audio_filepath}")
            except StorageError as exc:
                errors.append(f"Could not resolve {record.audio_filepath}: {exc}")
        counts[split] = count
    leaking = sorted(key for key, splits in group_splits.items() if len(splits) > 1)
    if leaking:
        errors.append(f"{len(leaking)} grouping key(s) cross split boundaries")
    expected_checksums = expected.get("output_manifest_checksums", {})
    for split, checksum in checksums.items():
        if expected_checksums.get(split) != checksum:
            errors.append(f"Checksum mismatch for {split}.jsonl")
    return {
        "status": "success" if not errors else "failed",
        "split_directory": str(directory),
        "records": counts,
        "group_leakage_count": len(leaking),
        "checksums": checksums,
        "errors": errors,
    }


def _select_manifests(
    config: SplitBuildConfig, resolver: DataPathResolver
) -> list[tuple[str, str]]:
    locators = list(config.explicit_manifests) or resolver.discover_manifest_locators()
    selected: list[tuple[str, str]] = []
    for locator in sorted(set(locators)):
        dataset = _dataset_from_manifest(locator, resolver)
        if not _included(dataset, config.include, config.exclude):
            continue
        selected.append((dataset, locator))
    return selected


def _dataset_from_manifest(
    locator: str, resolver: DataPathResolver
) -> str:
    parsed = urlsplit(locator)
    if parsed.scheme == "az":
        root = resolver.azure_root
        assert root is not None
        path = PurePosixPath(parsed.path.lstrip("/"))
        try:
            relative = path.relative_to(PurePosixPath(root.blob_path))
        except ValueError:
            relative = path
        if len(relative.parts) < 2:
            raise ManifestError(f"Cannot infer dataset name from manifest locator: {locator}")
        return relative.parts[0]
    path = Path(locator)
    if not path.is_absolute() and resolver.local_root is not None:
        path = resolver.local_root / normalize_logical_path(locator)
    if resolver.local_root is not None:
        try:
            relative = path.expanduser().resolve().relative_to(resolver.local_root)
            if len(relative.parts) >= 2:
                return relative.parts[0]
        except ValueError:
            pass
    return path.parent.name


def _read_sources(
    config: SplitBuildConfig,
    resolver: DataPathResolver,
    manifests: list[tuple[str, str]],
) -> tuple[list[_Row], dict[str, Any], dict[str, str]]:
    rows: list[_Row] = []
    errors: list[str] = []
    duplicate_audio: list[str] = []
    duplicate_record_ids: list[str] = []
    transcript_counts: Counter[str] = Counter()
    metadata_duplicate_counts: Counter[str] = Counter()
    seen_audio: set[str] = set()
    seen_record_ids: set[str] = set()
    source_fingerprints: dict[str, str] = {}
    missing_metadata: Counter[str] = Counter()

    for dataset, locator in manifests:
        source = _localize_config_manifest(locator, resolver)
        source_fingerprints[f"{dataset}:{PurePosixPath(locator).name}"] = manifest_fingerprint(source)
        defaults = dict(config.defaults.get("*", {}))
        defaults.update(config.defaults.get(dataset, {}))
        for line_number, raw in _iter_raw_jsonl(source):
            try:
                row = _normalize_row(
                    raw,
                    dataset=dataset,
                    source=str(locator),
                    line_number=line_number,
                    defaults=defaults,
                    config=config,
                    resolver=resolver,
                )
            except (ManifestError, StorageError) as exc:
                errors.append(str(exc))
                continue
            audio = row.record.audio_filepath
            if audio in seen_audio:
                duplicate_audio.append(audio)
                if config.fail_on_duplicate_audio_path:
                    errors.append(f"Duplicate audio path: {audio}")
            seen_audio.add(audio)
            if row.record_id in seen_record_ids:
                duplicate_record_ids.append(row.record_id)
                errors.append(f"Duplicate stable record ID: {row.record_id}")
            seen_record_ids.add(row.record_id)
            transcript_counts[row.record.text] += 1
            metadata_duplicate_counts[
                f"{row.record.duration:.6f}\0{row.record.text}"
            ] += 1
            for field in ("language", "domain", "speaker_id", "session_id", "source_id"):
                value = row.record.as_dict().get(field) or row.record.metadata.get(field)
                if value in (None, "unknown"):
                    missing_metadata[field] += 1
            rows.append(row)
    if not rows:
        errors.append("No valid source records were loaded")
    diagnostics = {
        "errors": errors,
        "invalid_record_count": len(errors),
        "duplicate_audio_paths": sorted(set(duplicate_audio)),
        "duplicate_stable_record_ids": sorted(set(duplicate_record_ids)),
        "exact_duplicate_transcript_count": sum(value - 1 for value in transcript_counts.values() if value > 1),
        "possible_duplicate_audio_metadata_count": sum(
            value - 1 for value in metadata_duplicate_counts.values() if value > 1
        ),
        "unknown_metadata_counts": dict(sorted(missing_metadata.items())),
    }
    return rows, diagnostics, source_fingerprints


def _normalize_row(
    raw: object,
    *,
    dataset: str,
    source: str,
    line_number: int,
    defaults: Mapping[str, Any],
    config: SplitBuildConfig,
    resolver: DataPathResolver,
) -> _Row:
    prefix = f"{source}:{line_number}: "
    if not isinstance(raw, dict):
        raise ManifestValidationError(prefix + "manifest row must be a JSON object")
    if not isinstance(raw.get("audio_filepath"), str) or not raw["audio_filepath"].strip():
        raise ManifestValidationError(prefix + "audio_filepath must be non-empty")
    if not isinstance(raw.get("text"), str) or not raw["text"].strip():
        raise ManifestValidationError(prefix + "text must be non-empty")
    duration = raw.get("duration")
    if type(duration) not in (int, float) or not math.isfinite(float(duration)) or float(duration) <= 0:
        raise ManifestValidationError(prefix + "duration must be a positive finite number")
    locator = _logical_source_audio(raw["audio_filepath"], dataset, resolver)
    suffix = PurePosixPath(locator).suffix.lower()
    if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ManifestValidationError(
            prefix + f"unsupported audio extension {suffix!r}; expected WAV or FLAC"
        )
    if config.require_audio_exists:
        local = resolver.resolve(locator)
        if not local.is_file() or local.stat().st_size <= 0:
            raise ManifestValidationError(prefix + f"audio is missing or empty: {locator}")

    existing_metadata = raw.get("metadata", {})
    if not isinstance(existing_metadata, dict):
        raise ManifestValidationError(prefix + "metadata must be a mapping")
    metadata = dict(existing_metadata)
    for key, value in raw.items():
        if key not in REQUIRED_FIELDS | OPTIONAL_FIELDS:
            if key in metadata and metadata[key] != value:
                raise ManifestValidationError(prefix + f"conflicting metadata field {key!r}")
            metadata[key] = value
    language = raw.get("language", defaults.get("language", "unknown"))
    domain = raw.get("domain", defaults.get("domain", "unknown"))
    if not isinstance(language, str) or not language.strip():
        language = "unknown"
    if not isinstance(domain, str) or not domain.strip():
        domain = "unknown"
    contains_number = _contains_number(raw["text"])
    tags = metadata.get("tags", defaults.get("tags", []))
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        raise ManifestValidationError(prefix + "tags must be a list of strings")
    if contains_number and "contains_number" not in tags:
        tags = [*tags, "contains_number"]
    metadata.update(
        {
            "dataset": dataset,
            "contains_number": contains_number,
            "tags": sorted(set(tags)),
            "source_manifest": PurePosixPath(source).name,
            "source_line_number": line_number,
        }
    )
    canonical: dict[str, Any] = {
        "audio_filepath": locator,
        "text": raw["text"],
        "duration": float(duration),
        "language": language,
        "domain": domain,
        "source": raw.get("source", dataset),
        "metadata": metadata,
    }
    for field in ("speaker_id", "recording_id"):
        value = raw.get(field)
        if value is not None:
            canonical[field] = value
    record = parse_record(canonical)
    record_id_source = _metadata_value(raw, metadata, "record_id", "id", "utt_id")
    record_id = (
        f"{dataset}:{record_id_source}"
        if record_id_source is not None
        else hashlib.sha256(f"{dataset}\0{locator}".encode()).hexdigest()[:24]
    )
    bucket = _duration_bucket(float(duration), config.duration_buckets)
    group_identifiers: list[str] = []
    for field in config.grouping_priority:
        if field == "audio_filepath":
            value: object = locator
        else:
            value = _metadata_value(raw, metadata, field)
        if isinstance(value, (str, int)) and str(value).strip():
            group_identifiers.append(f"{dataset}:{field}:{str(value).strip()}")
    features = _features(
        config.stratification_fields,
        dataset=dataset,
        language=language,
        domain=domain,
        contains_number=contains_number,
        duration_bucket=bucket,
        tags=metadata["tags"],
    )
    return _Row(
        record=record,
        dataset=dataset,
        record_id=record_id,
        group_identifiers=tuple(group_identifiers),
        features=features,
        duration_bucket=bucket,
        contains_number=contains_number,
    )


def _build_groups(rows: list[_Row]) -> list[_Group]:
    sets = _DisjointSet(len(rows))
    owners: dict[str, int] = {}
    for index, row in enumerate(rows):
        for identifier in row.group_identifiers:
            previous = owners.setdefault(identifier, index)
            sets.union(index, previous)
    members: dict[int, list[_Row]] = defaultdict(list)
    for index, row in enumerate(rows):
        members[sets.find(index)].append(row)
    groups: list[_Group] = []
    for values in members.values():
        feature_counts: Counter[str] = Counter()
        feature_durations: Counter[str] = Counter()
        for row in values:
            for feature in row.features:
                feature_counts[feature] += 1
                feature_durations[feature] += float(row.record.duration or 0)
        keys = sorted(identifier for row in values for identifier in row.group_identifiers)
        groups.append(
            _Group(
                key=hashlib.sha256("\0".join(keys).encode()).hexdigest()[:24],
                rows=values,
                duration=sum(float(row.record.duration or 0) for row in values),
                feature_counts=feature_counts,
                feature_durations=feature_durations,
            )
        )
    return groups


def _assign_groups(
    groups: list[_Group], rows: list[_Row], config: SplitBuildConfig
) -> dict[str, list[_Group]]:
    total_duration = sum(float(row.record.duration or 0) for row in rows)
    feature_counts: Counter[str] = Counter(feature for row in rows for feature in row.features)
    feature_durations: Counter[str] = Counter()
    for row in rows:
        for feature in row.features:
            feature_durations[feature] += float(row.record.duration or 0)
    current_counts = {split: 0 for split in SPLIT_NAMES}
    current_durations = {split: 0.0 for split in SPLIT_NAMES}
    current_features = {split: Counter() for split in SPLIT_NAMES}
    current_feature_durations = {split: Counter() for split in SPLIT_NAMES}
    assigned = {split: [] for split in SPLIT_NAMES}
    randomizer = random.Random(config.seed)
    tie = {group.key: randomizer.random() for group in groups}
    ordered = sorted(groups, key=lambda group: (-group.duration, -len(group.rows), tie[group.key]))
    split_order = list(SPLIT_NAMES)
    randomizer.shuffle(split_order)
    for group in ordered:
        best = min(
            split_order,
            key=lambda split: _score_assignment(
                split,
                group,
                config,
                len(rows),
                total_duration,
                feature_counts,
                feature_durations,
                current_counts,
                current_durations,
                current_features,
                current_feature_durations,
            ),
        )
        assigned[best].append(group)
        current_counts[best] += len(group.rows)
        current_durations[best] += group.duration
        current_features[best].update(group.feature_counts)
        current_feature_durations[best].update(group.feature_durations)
    return assigned


def _score_assignment(
    candidate: str,
    group: _Group,
    config: SplitBuildConfig,
    total_rows: int,
    total_duration: float,
    feature_counts: Counter[str],
    feature_durations: Counter[str],
    counts: dict[str, int],
    durations: dict[str, float],
    current_features: dict[str, Counter[str]],
    current_feature_durations: dict[str, Counter[str]],
) -> float:
    score = 0.0
    for split in SPLIT_NAMES:
        ratio = config.ratios[split]
        count = counts[split] + (len(group.rows) if split == candidate else 0)
        duration = durations[split] + (group.duration if split == candidate else 0)
        score += ((count - total_rows * ratio) / max(1.0, total_rows * ratio)) ** 2
        score += 6.0 * ((duration - total_duration * ratio) / max(1.0, total_duration * ratio)) ** 2
        for feature in feature_counts:
            feature_count = current_features[split][feature]
            feature_duration = current_feature_durations[split][feature]
            if split == candidate:
                feature_count += group.feature_counts[feature]
                feature_duration += group.feature_durations[feature]
            score += 0.5 * (
                (feature_count - feature_counts[feature] * ratio)
                / max(1.0, feature_counts[feature] * ratio)
            ) ** 2
            score += (
                (feature_duration - feature_durations[feature] * ratio)
                / max(1.0, feature_durations[feature] * ratio)
            ) ** 2
    return score


def _validate_assignments(
    rows: list[_Row], assignments: Mapping[str, list[_Group]]
) -> None:
    output_rows = [row for split in SPLIT_NAMES for group in assignments[split] for row in group.rows]
    if len(output_rows) != len(rows) or {row.record_id for row in output_rows} != {row.record_id for row in rows}:
        raise ManifestError("Split assignment did not preserve every record exactly once")
    if any(not assignments[split] for split in SPLIT_NAMES):
        raise ManifestError("At least one output split is empty; more independent groups are required")
    owners: dict[str, str] = {}
    for split in SPLIT_NAMES:
        for group in assignments[split]:
            previous = owners.setdefault(group.key, split)
            if previous != split:
                raise ManifestError("Group leakage detected during assignment")


def _write_rows(
    rows: Iterable[_Row],
    split: str,
    destination: Path,
    row_group_keys: Mapping[int, str],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                payload = row.record.as_dict()
                payload["split"] = split
                metadata = dict(payload.get("metadata", {}))
                metadata["split_group_key"] = row_group_keys[id(row)]
                metadata["stable_record_id"] = row.record_id
                metadata["duration_bucket"] = row.duration_bucket
                payload["metadata"] = metadata
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

def _build_report(
    config: SplitBuildConfig,
    rows: list[_Row],
    groups: list[_Group],
    assignments: Mapping[str, list[_Group]],
    diagnostics: Mapping[str, Any],
    warnings: list[str],
    source_fingerprints: Mapping[str, str],
    output_checksums: Mapping[str, str],
    fingerprint: str,
) -> dict[str, Any]:
    total_duration = sum(float(row.record.duration or 0) for row in rows)
    report: dict[str, Any] = {
        "status": "success",
        "name": config.name,
        "version": config.version,
        "seed": config.seed,
        "ratios": dict(config.ratios),
        "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(config.source_path.parent),
        "total_records": len(rows),
        "total_duration_seconds": total_duration,
        "total_groups": len(groups),
        "group_leakage_count": 0,
        "diagnostics": dict(diagnostics),
        "warnings": warnings,
        "source_manifest_fingerprints": dict(source_fingerprints),
        "output_manifest_checksums": dict(output_checksums),
        "split_fingerprint": fingerprint,
        "splits": {},
    }
    total_feature_counts: Counter[str] = Counter(feature for row in rows for feature in row.features)
    for split in SPLIT_NAMES:
        split_rows = [row for group in assignments[split] for row in group.rows]
        duration = sum(float(row.record.duration or 0) for row in split_rows)
        dataset_counts = Counter(row.dataset for row in split_rows)
        dataset_durations: Counter[str] = Counter()
        language = Counter(row.record.language or "unknown" for row in split_rows)
        domain = Counter(row.record.domain or "unknown" for row in split_rows)
        numbers = Counter(str(row.contains_number).lower() for row in split_rows)
        buckets = Counter(row.duration_bucket for row in split_rows)
        feature_counts = Counter(feature for row in split_rows for feature in row.features)
        for row in split_rows:
            dataset_durations[row.dataset] += float(row.record.duration or 0)
        report["splits"][split] = {
            "records": len(split_rows),
            "duration_seconds": duration,
            "groups": len(assignments[split]),
            "dataset_records": dict(sorted(dataset_counts.items())),
            "dataset_duration_seconds": dict(sorted(dataset_durations.items())),
            "language_distribution": dict(sorted(language.items())),
            "domain_distribution": dict(sorted(domain.items())),
            "contains_number_distribution": dict(sorted(numbers.items())),
            "duration_bucket_distribution": dict(sorted(buckets.items())),
            "distribution_deviation": {
                "record_ratio": len(split_rows) / len(rows) - config.ratios[split],
                "duration_ratio": duration / total_duration - config.ratios[split],
                "max_feature_record_ratio_deviation": max(
                    (
                        abs(feature_counts[feature] / count - config.ratios[split])
                        for feature, count in total_feature_counts.items()
                    ),
                    default=0.0,
                ),
            },
        }
    return report


def _representation_warnings(
    assignments: Mapping[str, list[_Group]],
) -> list[str]:
    feature_groups: dict[str, set[str]] = defaultdict(set)
    for split_groups in assignments.values():
        for group in split_groups:
            for feature in group.feature_counts:
                feature_groups[feature].add(group.key)
    warnings = [
        f"Feature {feature!r} has only {len(keys)} independent group(s), so representation in every split is not guaranteed"
        for feature, keys in sorted(feature_groups.items())
        if len(keys) < len(SPLIT_NAMES)
    ]
    return warnings


def _final_fingerprint(
    config: SplitBuildConfig,
    sources: Mapping[str, str],
    outputs: Mapping[str, str],
) -> str:
    payload = {
        "config": config.fingerprint_payload(),
        "sources": dict(sorted(sources.items())),
        "outputs": dict(sorted(outputs.items())),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _logical_source_audio(value: str, dataset: str, resolver: DataPathResolver) -> str:
    try:
        logical = resolver.logical_path(value)
    except StorageError:
        portable_suffix = _portable_dataset_suffix(value, dataset)
        if portable_suffix is None:
            logical = resolver.logical_path(value, dataset=dataset)
        else:
            logical = portable_suffix
    if logical == dataset or logical.startswith(f"{dataset}/"):
        return logical
    return normalize_logical_path(f"{dataset}/{logical}")


def _localize_config_manifest(locator: str, resolver: DataPathResolver) -> Path:
    parsed = urlsplit(locator)
    if parsed.scheme:
        return resolver.localize_manifest(locator)
    path = Path(locator).expanduser()
    if path.is_absolute():
        return path.resolve()
    if resolver.local_root is not None:
        return (resolver.local_root / normalize_logical_path(locator)).resolve()
    return resolver.localize_manifest(locator)


def _portable_dataset_suffix(value: str, dataset: str) -> str | None:
    """Recover dataset-relative suffixes from legacy absolute/Blob-style paths."""

    parsed = urlsplit(value)
    candidate = parsed.path if parsed.scheme else value
    parts = PurePosixPath(candidate.replace("\\", "/")).parts
    matching = [index for index, part in enumerate(parts) if part == dataset]
    if not matching:
        return None
    suffix = parts[matching[-1] :]
    if len(suffix) < 2:
        return None
    return normalize_logical_path(str(PurePosixPath(*suffix)))


def _iter_raw_jsonl(path: Path) -> Iterable[tuple[int, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    yield line_number, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ManifestValidationError(
                        f"{path}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc
    except OSError as exc:
        raise ManifestError(f"Could not read manifest {path}: {exc}") from exc


def _features(
    fields: tuple[str, ...],
    *,
    dataset: str,
    language: str,
    domain: str,
    contains_number: bool,
    duration_bucket: str,
    tags: list[str],
) -> tuple[str, ...]:
    values: dict[str, object] = {
        "dataset": dataset,
        "language": language,
        "domain": domain,
        "contains_number": str(contains_number).lower(),
        "duration_bucket": duration_bucket,
    }
    features: list[str] = []
    for field in fields:
        if field == "tags":
            features.extend(f"tags={tag}" for tag in tags)
        elif field in values:
            features.append(f"{field}={values[field]}")
        else:
            raise ManifestError(f"Unsupported stratification field: {field!r}")
    return tuple(features)


def _metadata_value(raw: Mapping[str, Any], metadata: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = raw.get(key, metadata.get(key))
        if value not in (None, ""):
            return value
    return None


def _contains_number(text: str) -> bool:
    if any(character.isdigit() for character in text):
        return True
    words = {word.casefold().strip(".,!?;:()[]{}") for word in text.split()}
    return bool(words & _NUMBER_WORDS)


def _duration_bucket(
    duration: float,
    buckets: tuple[tuple[str, float | None, float | None], ...],
) -> str:
    for name, minimum, maximum in buckets:
        if (minimum is None or duration >= minimum) and (maximum is None or duration < maximum):
            return name
    raise ManifestError(f"Duration {duration} does not match a configured bucket")


def _parse_duration_buckets(value: object) -> tuple[tuple[str, float | None, float | None], ...]:
    if not isinstance(value, dict) or not value:
        raise ManifestError("duration_buckets must be a non-empty mapping")
    result: list[tuple[str, float | None, float | None]] = []
    for name, bounds in value.items():
        if not isinstance(name, str) or not isinstance(bounds, list) or len(bounds) != 2:
            raise ManifestError("Each duration bucket must be name: [minimum, maximum]")
        minimum, maximum = bounds
        if minimum is not None and (type(minimum) not in (int, float) or minimum < 0):
            raise ManifestError(f"Invalid minimum for duration bucket {name!r}")
        if maximum is not None and (type(maximum) not in (int, float) or maximum <= 0):
            raise ManifestError(f"Invalid maximum for duration bucket {name!r}")
        if minimum is not None and maximum is not None and minimum >= maximum:
            raise ManifestError(f"Invalid bounds for duration bucket {name!r}")
        result.append((name, float(minimum) if minimum is not None else None, float(maximum) if maximum is not None else None))
    return tuple(result)


def _included(dataset: str, include: tuple[str, ...], exclude: tuple[str, ...]) -> bool:
    return ("*" in include or dataset in include) and dataset not in exclude


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(character in value for character in "/\\"):
        raise ManifestError(f"Split {label} must be a non-empty path-safe string")
    return value.strip()


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ManifestError(f"{label} must be a list of non-empty strings")
    if len(set(value)) != len(value):
        raise ManifestError(f"{label} contains duplicate values")
    return tuple(value)


def _bool_option(mapping: Mapping[str, Any], name: str, default: bool) -> bool:
    value = mapping.get(name, default)
    if type(value) is not bool:
        raise ManifestError(f"validation.{name} must be true or false")
    return value


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    except OSError as exc:
        raise ManifestError(f"Could not write {path}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _git_commit(start: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=start,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def format_split_summary(report: Mapping[str, Any]) -> str:
    lines = ["split       records   hours     groups", "-----------  --------  --------  ------"]
    splits = report.get("splits", {})
    for split in SPLIT_NAMES:
        values = splits[split]
        lines.append(
            f"{split:<11}  {values['records']:>8}  {values['duration_seconds'] / 3600:>8.3f}  {values['groups']:>6}"
        )
    lines.append(f"fingerprint: {report['split_fingerprint']}")
    return "\n".join(lines)


__all__ = [
    "SplitBuildConfig",
    "build_splits",
    "format_split_summary",
    "load_split_config",
    "validate_split_directory",
]
