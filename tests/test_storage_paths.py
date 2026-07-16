from __future__ import annotations

import io
from pathlib import Path

import pytest

from orato_asr.exceptions import StorageError
from orato_asr.paths import (
    DataPathResolver,
    StorageSettings,
    cache_path_for_blob,
    normalize_logical_path,
    parse_azure_blob_uri,
)


class _BlobStore:
    def __init__(self, payload: bytes = b"audio") -> None:
        self.payload = payload
        self.downloads = 0

    def download_blob(self, container: str, blob_path: str, handle: io.BufferedWriter) -> None:
        assert container == "data"
        assert blob_path == "processed/set/audio/a.flac"
        self.downloads += 1
        handle.write(self.payload)

    def list_blobs(self, container: str, prefix: str) -> list[str]:
        return [
            "processed/set/manifest.jsonl",
            "processed/set/audio/a.flac",
        ]


def _settings(tmp_path: Path, *, root: str, backend: str) -> StorageSettings:
    return StorageSettings(
        data_root=root,
        split_root=tmp_path / "splits",
        cache_root=tmp_path / "cache",
        backend=backend,
        azure_account_name="account",
    )


def test_local_linux_and_mounted_paths_resolve_from_processed_root(tmp_path: Path) -> None:
    processed = tmp_path / "mnt" / "orato-data" / "processed"
    resolver = DataPathResolver(_settings(tmp_path, root=str(processed), backend="local"))

    assert resolver.resolve("set/audio/a.flac") == processed / "set/audio/a.flac"
    assert resolver.logical_path(str(processed / "set/audio/a.flac")) == "set/audio/a.flac"


def test_windows_paths_and_backslashes_normalize_portably(tmp_path: Path) -> None:
    resolver = DataPathResolver(
        _settings(tmp_path, root="D:/Orato/data/processed", backend="local")
    )

    assert normalize_logical_path(r"set\audio\a.flac") == "set/audio/a.flac"
    assert (
        resolver.logical_path(r"D:\Orato\data\processed\set\audio\a.flac")
        == "set/audio/a.flac"
    )
    with pytest.raises(StorageError, match="outside"):
        resolver.logical_path(r"C:\private\a.flac")

    settings = StorageSettings.from_environment(
        project_root=tmp_path, data_root="D:/Orato/data/processed"
    )
    assert settings.backend == "local"


def test_azure_uri_parsing_and_cache_path_are_deterministic(tmp_path: Path) -> None:
    parsed = parse_azure_blob_uri("az://data/processed/set/audio/a.flac")
    assert (parsed.container, parsed.blob_path) == (
        "data",
        "processed/set/audio/a.flac",
    )
    first = cache_path_for_blob(
        tmp_path, account_name="account", container="data", blob_path=parsed.blob_path
    )
    second = cache_path_for_blob(
        tmp_path, account_name="account", container="data", blob_path=parsed.blob_path
    )
    assert first == second
    assert first.is_relative_to(tmp_path)
    with pytest.raises(StorageError, match="query"):
        parse_azure_blob_uri("az://data/a.flac?sig=secret")


def test_blob_download_is_atomic_and_cache_is_reused(tmp_path: Path) -> None:
    store = _BlobStore()
    resolver = DataPathResolver(
        _settings(tmp_path, root="az://data/processed", backend="azure_blob"),
        blob_store=store,
    )

    first = resolver.resolve("set/audio/a.flac")
    second = resolver.resolve("set/audio/a.flac")

    assert first == second
    assert first.read_bytes() == b"audio"
    assert store.downloads == 1
    assert not list(first.parent.glob("*.tmp"))
    assert not list(first.parent.glob("*.lock"))


def test_blob_manifest_discovery_does_not_download_audio(tmp_path: Path) -> None:
    store = _BlobStore()
    resolver = DataPathResolver(
        _settings(tmp_path, root="az://data/processed", backend="azure_blob"),
        blob_store=store,
    )

    assert resolver.discover_manifest_locators() == [
        "az://data/processed/set/manifest.jsonl"
    ]
    assert store.downloads == 0
