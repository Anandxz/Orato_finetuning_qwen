"""Safe repository path handling for local generated artifacts."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from .exceptions import PathSafetyError


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the nearest ancestor containing this project's ``pyproject.toml``."""

    candidate = Path.cwd() if start is None else Path(start)
    candidate = candidate.expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for directory in (candidate, *candidate.parents):
        if (directory / "pyproject.toml").is_file():
            return directory

    raise PathSafetyError(
        f"Could not find a project root containing pyproject.toml from {candidate}"
    )


def resolve_repository_path(
    value: object,
    *,
    project_root: Path,
    allowed_directory: str,
    create: bool = False,
) -> Path:
    """Resolve a relative local path and keep it within an allowed directory."""

    if allowed_directory not in {"outputs", "reports"}:
        raise PathSafetyError(
            "allowed_directory must be either 'outputs' or 'reports'; received "
            f"{allowed_directory!r}"
        )

    if not isinstance(value, str) or not value.strip():
        raise PathSafetyError("must be a non-empty relative path string")

    raw_value = value.strip()
    parsed = urlsplit(raw_value)
    if parsed.scheme or parsed.netloc:
        raise PathSafetyError(
            f"URI values are not supported for local paths: {raw_value!r}"
        )

    relative_path = Path(raw_value)
    if relative_path.is_absolute() or raw_value.startswith("~"):
        raise PathSafetyError(f"absolute paths are not allowed: {raw_value!r}")

    root = project_root.expanduser().resolve()
    allowed_root = (root / allowed_directory).resolve()
    resolved = (root / relative_path).resolve()

    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise PathSafetyError(
            f"must resolve inside {allowed_root}; received {raw_value!r}"
        ) from exc

    if create:
        try:
            resolved.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PathSafetyError(
                f"could not create repository-managed directory {resolved}: {exc}"
            ) from exc

    return resolved


def resolve_output_dir(
    value: object,
    *,
    project_root: Path,
    create: bool = False,
) -> Path:
    """Resolve a path under the repository's ``outputs`` directory."""

    return resolve_repository_path(
        value,
        project_root=project_root,
        allowed_directory="outputs",
        create=create,
    )


def resolve_reports_dir(
    value: object,
    *,
    project_root: Path,
    create: bool = False,
) -> Path:
    """Resolve a path under the repository's ``reports`` directory."""

    return resolve_repository_path(
        value,
        project_root=project_root,
        allowed_directory="reports",
        create=create,
    )
