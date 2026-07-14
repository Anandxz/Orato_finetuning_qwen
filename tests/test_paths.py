from __future__ import annotations

from pathlib import Path

import pytest

from orato_asr.exceptions import PathSafetyError, UnsafePathError
from orato_asr.paths import (
    find_project_root,
    resolve_output_dir,
    resolve_reports_dir,
    resolve_repository_path,
)

ROOT = Path(__file__).resolve().parents[1]


def test_repository_root_resolves_from_nested_file() -> None:
    assert find_project_root(ROOT / "src" / "orato_asr" / "paths.py") == ROOT


def test_relative_output_and_report_paths_resolve_under_repository() -> None:
    assert resolve_output_dir("outputs/run-1", project_root=ROOT) == (
        ROOT / "outputs" / "run-1"
    )
    assert resolve_reports_dir("reports/run-1", project_root=ROOT) == (
        ROOT / "reports" / "run-1"
    )


def test_directory_creation_requires_explicit_request(tmp_path: Path) -> None:
    output_path = resolve_output_dir("outputs/run", project_root=tmp_path)
    assert not output_path.exists()

    created_path = resolve_output_dir(
        "outputs/run",
        project_root=tmp_path,
        create=True,
    )
    assert created_path.is_dir()


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../outside",
        "outputs/../../outside",
        "/tmp/outside",
        "azureml://datastores/example/paths/output",
        "https://example.invalid/output",
        "wasbs://container@example.blob.core.windows.net/output",
    ],
)
def test_traversal_and_remote_uris_are_rejected(unsafe_path: str) -> None:
    with pytest.raises(PathSafetyError):
        resolve_output_dir(unsafe_path, project_root=ROOT)


def test_generic_path_helper_restricts_managed_directory_name() -> None:
    with pytest.raises(PathSafetyError, match="allowed_directory"):
        resolve_repository_path(
            "outside/run",
            project_root=ROOT,
            allowed_directory="outside",
        )


def test_unsafe_path_alias_is_the_public_path_error() -> None:
    assert UnsafePathError is PathSafetyError


def test_path_source_contains_no_user_specific_home() -> None:
    source = (ROOT / "src" / "orato_asr" / "paths.py").read_text(encoding="utf-8")
    assert str(Path.home()) not in source
