"""Streaming canonical JSONL manifest reading and atomic derived-manifest writing."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import urlsplit

from ..exceptions import ManifestError, ManifestValidationError
from .schema import ManifestRecord, parse_record


@dataclass(frozen=True, slots=True)
class ManifestEvent:
    line_number: int
    record: ManifestRecord | None
    error: ManifestError | None


def iter_manifest(
    path: str | Path,
    *,
    skip_blank_lines: bool = True,
) -> Iterator[ManifestRecord]:
    """Yield validated records in file order and stop on the first malformed line."""

    for event in iter_manifest_events(path, skip_blank_lines=skip_blank_lines):
        if event.error is not None:
            raise event.error
        assert event.record is not None
        yield event.record


def iter_manifest_events(
    path: str | Path,
    *,
    skip_blank_lines: bool = True,
) -> Iterator[ManifestEvent]:
    """Yield line-numbered records or expected parsing/schema errors."""

    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise ManifestError(f"Manifest file does not exist: {source}")
    if not source.is_file():
        raise ManifestError(f"Manifest path is not a regular file: {source}")
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip() and skip_blank_lines:
                    continue
                if not line.strip():
                    yield ManifestEvent(
                        line_number,
                        None,
                        ManifestValidationError(f"{source}:{line_number}: blank JSONL line"),
                    )
                    continue
                try:
                    decoded = json.loads(line)
                except json.JSONDecodeError as exc:
                    yield ManifestEvent(
                        line_number,
                        None,
                        ManifestValidationError(
                            f"{source}:{line_number}: invalid JSON: {exc.msg}"
                        ),
                    )
                    continue
                try:
                    record = parse_record(
                        decoded, manifest_path=source, line_number=line_number
                    )
                except ManifestValidationError as exc:
                    yield ManifestEvent(line_number, None, exc)
                else:
                    yield ManifestEvent(line_number, record, None)
    except OSError as exc:
        raise ManifestError(f"Could not read manifest {source}: {exc}") from exc


def write_manifest(
    records: Iterator[ManifestRecord] | list[ManifestRecord],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write a derived JSONL manifest without rewriting source input."""

    destination = _local_destination(path)
    if destination.exists() and not overwrite:
        raise ManifestError(
            f"Refusing to overwrite existing manifest without --overwrite: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            for record in records:
                temporary.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
    except OSError as exc:
        raise ManifestError(f"Could not write manifest {destination}: {exc}") from exc
    finally:
        if temporary_name is not None:
            temporary_path = Path(temporary_name)
            if temporary_path.exists():
                temporary_path.unlink(missing_ok=True)
    return destination


def write_json_atomic(
    payload: object,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write a UTF-8 JSON sidecar at an explicit local path."""

    destination = _local_destination(path)
    if destination.exists() and not overwrite:
        raise ManifestError(
            f"Refusing to overwrite existing JSON without --overwrite: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(payload, temporary, ensure_ascii=False, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
    except OSError as exc:
        raise ManifestError(f"Could not write JSON {destination}: {exc}") from exc
    finally:
        if temporary_name is not None:
            temporary_path = Path(temporary_name)
            if temporary_path.exists():
                temporary_path.unlink(missing_ok=True)
    return destination


def manifest_fingerprint(path: str | Path) -> str:
    """Return a content fingerprint for safe baseline resume checks."""

    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ManifestError(f"Could not fingerprint manifest {source}: {exc}") from exc
    return digest.hexdigest()


def _local_destination(path: str | Path) -> Path:
    raw = str(path).strip()
    parsed = urlsplit(raw)
    if not raw or parsed.scheme or parsed.netloc:
        raise ManifestError("Derived manifest and report outputs must be explicit local paths")
    return Path(raw).expanduser().resolve()
