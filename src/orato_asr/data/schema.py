"""Canonical JSONL manifest records and structural field validation."""

from __future__ import annotations

import math
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping
from urllib.parse import urlsplit

from ..exceptions import ManifestValidationError

if TYPE_CHECKING:
    from ..paths import DataPathResolver

REQUIRED_FIELDS = frozenset({"audio_filepath", "text"})
OPTIONAL_FIELDS = frozenset(
    {
        "duration",
        "language",
        "source",
        "speaker_id",
        "recording_id",
        "domain",
        "split",
        "metadata",
    }
)
TOP_LEVEL_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS
_IDENTIFIER_FIELDS = ("language", "source", "speaker_id", "recording_id", "domain", "split")
_REMOTE_SCHEMES = {
    "azureml",
    "azure",
    "az",
    "blob",
    "https",
    "http",
    "abfs",
    "abfss",
    "wasb",
    "wasbs",
}
_CONTROL_PATTERN = re.compile(r"[\x00-\x1f\x7f-\x9f]")


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    """A validated record while preserving source text and path verbatim."""

    audio_filepath: str
    text: str
    duration: float | None = None
    language: str | None = None
    source: str | None = None
    speaker_id: str | None = None
    recording_id: str | None = None
    domain: str | None = None
    split: str | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    manifest_path: Path | None = None
    line_number: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return canonical JSON-compatible values without reader provenance."""

        values: dict[str, Any] = {
            "audio_filepath": self.audio_filepath,
            "text": self.text,
        }
        for key in (
            "duration",
            "language",
            "source",
            "speaker_id",
            "recording_id",
            "domain",
            "split",
        ):
            value = getattr(self, key)
            if value is not None:
                values[key] = value
        if self.metadata:
            values["metadata"] = dict(self.metadata)
        return values

    @property
    def is_remote(self) -> bool:
        return is_remote_locator(self.audio_filepath)


def parse_record(
    value: object,
    *,
    manifest_path: Path | None = None,
    line_number: int | None = None,
) -> ManifestRecord:
    """Validate one JSON object against the deliberately strict schema."""

    prefix = _location_prefix(manifest_path, line_number)
    if not isinstance(value, dict):
        raise ManifestValidationError(f"{prefix}manifest line must be a JSON object")
    non_string_keys = [key for key in value if not isinstance(key, str)]
    if non_string_keys:
        raise ManifestValidationError(f"{prefix}manifest has non-string keys")
    keys = set(value)
    missing = sorted(REQUIRED_FIELDS - keys)
    unknown = sorted(keys - TOP_LEVEL_FIELDS)
    if missing:
        raise ManifestValidationError(
            f"{prefix}manifest is missing required fields: {', '.join(missing)}"
        )
    if unknown:
        raise ManifestValidationError(
            f"{prefix}manifest contains unsupported top-level fields: {', '.join(unknown)}"
        )

    audio_filepath = _non_empty_string(value["audio_filepath"], "audio_filepath", prefix)
    text = _non_empty_string(value["text"], "text", prefix)
    if _CONTROL_PATTERN.search(text):
        raise ManifestValidationError(f"{prefix}text contains control characters")

    duration = value.get("duration")
    if duration is not None:
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
            raise ManifestValidationError(f"{prefix}duration must be a positive finite number")
        duration = float(duration)

    identifiers: dict[str, str | None] = {}
    for field in _IDENTIFIER_FIELDS:
        raw = value.get(field)
        if raw is None:
            identifiers[field] = None
            continue
        identifier = _non_empty_string(raw, field, prefix)
        if _CONTROL_PATTERN.search(identifier):
            raise ManifestValidationError(f"{prefix}{field} contains control characters")
        identifiers[field] = identifier

    metadata = value.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ManifestValidationError(f"{prefix}metadata must be a JSON object")
    if any(not isinstance(key, str) for key in metadata):
        raise ManifestValidationError(f"{prefix}metadata contains non-string keys")

    return ManifestRecord(
        audio_filepath=audio_filepath,
        text=text,
        duration=duration,
        language=identifiers["language"],
        source=identifiers["source"],
        speaker_id=identifiers["speaker_id"],
        recording_id=identifiers["recording_id"],
        domain=identifiers["domain"],
        split=identifiers["split"],
        metadata=MappingProxyType(dict(metadata)),
        manifest_path=manifest_path,
        line_number=line_number,
    )


def is_remote_locator(value: str) -> bool:
    """Return whether a locator is a future remote URI, without resolving it."""

    parsed = urlsplit(value)
    return parsed.scheme.lower() in _REMOTE_SCHEMES and bool(parsed.scheme)


def resolve_local_audio_path(
    record: ManifestRecord,
    project_root: Path,
    *,
    data_resolver: "DataPathResolver | None" = None,
) -> Path | None:
    """Resolve through the configured data root while preserving legacy behavior."""

    if data_resolver is not None:
        return data_resolver.resolve(record.audio_filepath)
    if "ORATO_DATA_ROOT" in os.environ:
        from ..paths import resolver_from_environment

        return resolver_from_environment(project_root=project_root).resolve(
            record.audio_filepath
        )

    if record.is_remote:
        return None
    raw = Path(record.audio_filepath)
    return raw.expanduser().resolve() if raw.is_absolute() else (project_root / raw).resolve()


def display_audio_locator(value: str) -> str:
    """Retain a useful remote path while removing URI queries such as SAS tokens."""

    if not is_remote_locator(value):
        return value
    parsed = urlsplit(value)
    if parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    return f"{parsed.scheme}:{parsed.path}"


def script_classification(text: str) -> str:
    """Classify transcript letters for compact dataset-summary reporting."""

    has_devanagari = any("\u0900" <= character <= "\u097f" for character in text)
    has_latin = any("A" <= character <= "Z" or "a" <= character <= "z" for character in text)
    if has_devanagari and has_latin:
        return "mixed_devanagari_latin"
    if has_devanagari:
        return "devanagari_only"
    if has_latin:
        return "latin_only"
    return "other_or_unknown"


def lexical_content(text: str) -> str:
    """Return letters/numbers only for non-destructive transcript policy checks."""

    return "".join(
        character for character in text if unicodedata.category(character)[0] in {"L", "N"}
    )


def _non_empty_string(value: object, field: str, prefix: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{prefix}{field} must be a non-empty string")
    return value


def _location_prefix(path: Path | None, line_number: int | None) -> str:
    if path is None:
        return ""
    if line_number is None:
        return f"{path}: "
    return f"{path}:{line_number}: "
