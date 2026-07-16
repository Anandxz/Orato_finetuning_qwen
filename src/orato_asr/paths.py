"""Safe repository paths and environment-independent processed-data access."""

from __future__ import annotations

import importlib
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO, Iterable, Protocol
from urllib.parse import urlsplit

from .exceptions import PathSafetyError, StorageError

LOGGER = logging.getLogger(__name__)
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
_SUPPORTED_DATA_BACKENDS = {"local", "azure_blob"}


@dataclass(frozen=True, slots=True)
class AzureBlobLocation:
    """Credential-free identity for one Azure Blob object or prefix."""

    container: str
    blob_path: str


class BlobStore(Protocol):
    """Minimal interface used by the resolver and mocked by offline tests."""

    def download_blob(
        self, container: str, blob_path: str, handle: BinaryIO
    ) -> None: ...

    def list_blobs(self, container: str, prefix: str) -> Iterable[str]: ...


@dataclass(frozen=True, slots=True)
class StorageSettings:
    """Portable storage roots loaded from explicit values or environment."""

    data_root: str
    split_root: Path
    cache_root: Path
    backend: str
    azure_account_name: str | None = None

    @classmethod
    def from_environment(
        cls,
        *,
        project_root: Path,
        data_root: str | None = None,
        split_root: str | Path | None = None,
        cache_root: str | Path | None = None,
        backend: str | None = None,
    ) -> "StorageSettings":
        selected_data_root = (data_root or os.environ.get("ORATO_DATA_ROOT") or "data/processed").strip()
        selected_backend = (backend or os.environ.get("ORATO_STORAGE_BACKEND") or "").strip().lower()
        data_scheme = (
            ""
            if _WINDOWS_ABSOLUTE.match(selected_data_root)
            else urlsplit(selected_data_root).scheme.lower()
        )
        if not selected_backend:
            selected_backend = "azure_blob" if data_scheme == "az" else "local"
        if selected_backend not in _SUPPORTED_DATA_BACKENDS:
            raise StorageError(
                "ORATO_STORAGE_BACKEND must be 'local' or 'azure_blob'; "
                f"received {selected_backend!r}"
            )
        if selected_backend == "azure_blob":
            parse_azure_blob_uri(selected_data_root)
        elif data_scheme:
            raise StorageError(
                "The local storage backend requires a filesystem ORATO_DATA_ROOT"
            )
        selected_split_root = Path(
            split_root or os.environ.get("ORATO_SPLIT_ROOT") or "data/splits"
        ).expanduser()
        if not selected_split_root.is_absolute():
            selected_split_root = project_root / selected_split_root
        selected_cache_root = Path(
            cache_root or os.environ.get("ORATO_CACHE_ROOT") or "outputs/data_cache"
        ).expanduser()
        if not selected_cache_root.is_absolute():
            selected_cache_root = project_root / selected_cache_root
        return cls(
            data_root=selected_data_root,
            split_root=selected_split_root.resolve(),
            cache_root=selected_cache_root.resolve(),
            backend=selected_backend,
            azure_account_name=(
                os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
                or os.environ.get("AZURE_STORAGE_ACCOUNT")
                or None
            ),
        )


class DataPathResolver:
    """Resolve logical processed-data paths and localize Azure blobs on demand."""

    def __init__(
        self,
        settings: StorageSettings,
        *,
        blob_store: BlobStore | None = None,
        lock_timeout_seconds: float = 300.0,
    ) -> None:
        self.settings = settings
        self._blob_store = blob_store
        self.lock_timeout_seconds = lock_timeout_seconds
        self._local_root = (
            Path(settings.data_root).expanduser().resolve()
            if settings.backend == "local"
            else None
        )
        self._windows_root = (
            PureWindowsPath(settings.data_root)
            if settings.backend == "local" and _WINDOWS_ABSOLUTE.match(settings.data_root)
            else None
        )
        self._azure_root = (
            parse_azure_blob_uri(settings.data_root)
            if settings.backend == "azure_blob"
            else None
        )

    @property
    def local_root(self) -> Path | None:
        return self._local_root

    @property
    def azure_root(self) -> AzureBlobLocation | None:
        return self._azure_root

    def resolve(self, locator: str) -> Path:
        """Return a safe local path, downloading a blob atomically if required."""

        value = locator.strip()
        if not value:
            raise StorageError("Audio locator must be a non-empty string")
        scheme = urlsplit(value).scheme.lower()
        if scheme == "az":
            return self._localize(parse_azure_blob_uri(value))
        if scheme:
            raise StorageError(
                f"Unsupported direct data URI scheme {scheme!r}; use az:// or an Azure ML mount"
            )
        if self._local_root is not None:
            return self._resolve_local(value)
        assert self._azure_root is not None
        logical = normalize_logical_path(value)
        blob_path = _join_posix(self._azure_root.blob_path, logical)
        return self._localize(AzureBlobLocation(self._azure_root.container, blob_path))

    def logical_path(self, locator: str, *, dataset: str | None = None) -> str:
        """Normalize a source locator into a portable processed-root-relative path."""

        value = locator.strip()
        if not value:
            raise StorageError("Audio locator must be a non-empty string")
        if self._local_root is not None:
            if self._windows_root is not None and _WINDOWS_ABSOLUTE.match(value):
                try:
                    relative = PureWindowsPath(value).relative_to(self._windows_root)
                except ValueError as exc:
                    raise StorageError("Windows audio path is outside ORATO_DATA_ROOT") from exc
                return normalize_logical_path(str(relative))
            relative = _relative_to_local_root(value, self._local_root)
            if relative is not None:
                return normalize_logical_path(relative)
        elif urlsplit(value).scheme.lower() == "az":
            location = parse_azure_blob_uri(value)
            assert self._azure_root is not None
            if location.container != self._azure_root.container:
                raise StorageError("Blob locator uses a different container than ORATO_DATA_ROOT")
            try:
                relative = PurePosixPath(location.blob_path).relative_to(
                    PurePosixPath(self._azure_root.blob_path)
                )
            except ValueError as exc:
                raise StorageError("Blob locator is outside ORATO_DATA_ROOT") from exc
            return normalize_logical_path(str(relative))
        if urlsplit(value).scheme:
            raise StorageError(
                "Remote source audio must use az:// and match the configured data root"
            )
        logical = normalize_logical_path(value)
        if dataset and not (logical == dataset or logical.startswith(f"{dataset}/")):
            logical = normalize_logical_path(f"{dataset}/{logical}")
        return logical

    def discover_manifest_locators(self) -> list[str]:
        """Return sorted manifest locators without downloading audio."""

        if self._local_root is not None:
            if not self._local_root.is_dir():
                raise StorageError(f"Processed-data root does not exist: {self._local_root}")
            return [
                str(path)
                for path in sorted(self._local_root.rglob("manifest*.jsonl"))
                if path.is_file()
            ]
        assert self._azure_root is not None
        prefix = self._azure_root.blob_path.rstrip("/") + "/"
        return [
            f"az://{self._azure_root.container}/{name}"
            for name in sorted(self._store().list_blobs(self._azure_root.container, prefix))
            if PurePosixPath(name).name.startswith("manifest") and name.endswith(".jsonl")
        ]

    def localize_manifest(self, locator: str) -> Path:
        return self.resolve(locator)

    def _resolve_local(self, value: str) -> Path:
        assert self._local_root is not None
        relative = _relative_to_local_root(value, self._local_root)
        if relative is not None:
            return (self._local_root / normalize_logical_path(relative)).resolve()
        if Path(value).expanduser().is_absolute():
            return Path(value).expanduser().resolve()
        logical = normalize_logical_path(value)
        destination = (self._local_root / logical).resolve()
        _require_within(destination, self._local_root, "Logical data path")
        return destination

    def _localize(self, location: AzureBlobLocation) -> Path:
        account = self.settings.azure_account_name
        if not account and os.environ.get("AZURE_STORAGE_CONNECTION_STRING") is None:
            raise StorageError(
                "Direct Azure Blob access requires AZURE_STORAGE_ACCOUNT_NAME or "
                "AZURE_STORAGE_CONNECTION_STRING"
            )
        account_key = account or "connection-string-account"
        relative = Path("azure_blob") / _safe_component(account_key) / _safe_component(location.container)
        for component in PurePosixPath(location.blob_path).parts:
            if component not in {"", ".", "/"}:
                relative /= _safe_component(component)
        destination = (self.settings.cache_root / relative).resolve()
        _require_within(destination, self.settings.cache_root, "Azure cache path")
        if _valid_cached_file(destination):
            LOGGER.debug("Azure data cache hit: %s", destination)
            return destination
        LOGGER.debug("Azure data cache miss: %s", destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        lock_path = destination.with_name(destination.name + ".lock")
        acquired = _acquire_lock(lock_path, self.lock_timeout_seconds)
        try:
            if _valid_cached_file(destination):
                return destination
            temporary_name: str | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    prefix=f".{destination.name}.",
                    suffix=".tmp",
                    dir=destination.parent,
                    delete=False,
                ) as temporary:
                    temporary_name = temporary.name
                    self._store().download_blob(
                        location.container, location.blob_path, temporary
                    )
                    temporary.flush()
                    os.fsync(temporary.fileno())
                temporary_path = Path(temporary_name)
                if not _valid_cached_file(temporary_path):
                    raise StorageError(
                        f"Downloaded Azure blob is empty: az://{location.container}/{location.blob_path}"
                    )
                os.replace(temporary_path, destination)
            finally:
                if temporary_name:
                    Path(temporary_name).unlink(missing_ok=True)
        finally:
            if acquired:
                lock_path.unlink(missing_ok=True)
        return destination

    def _store(self) -> BlobStore:
        if self._blob_store is None:
            self._blob_store = AzureBlobStore(self.settings.azure_account_name)
        return self._blob_store


class AzureBlobStore:
    """Small optional-SDK adapter using managed identity or an environment secret."""

    def __init__(self, account_name: str | None) -> None:
        try:
            blob_module = importlib.import_module("azure.storage.blob")
        except (ImportError, OSError) as exc:
            raise StorageError(
                "Direct Azure Blob access requires the optional azure dependencies: "
                "pip install -e '.[azure]'"
            ) from exc
        connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if connection_string:
            self._service = blob_module.BlobServiceClient.from_connection_string(
                connection_string
            )
            return
        if not account_name:
            raise StorageError("AZURE_STORAGE_ACCOUNT_NAME is required for managed identity")
        try:
            identity_module = importlib.import_module("azure.identity")
        except (ImportError, OSError) as exc:
            raise StorageError("Managed identity requires azure-identity") from exc
        credential = identity_module.DefaultAzureCredential()
        self._service = blob_module.BlobServiceClient(
            account_url=f"https://{account_name}.blob.core.windows.net",
            credential=credential,
        )

    def download_blob(
        self, container: str, blob_path: str, handle: BinaryIO
    ) -> None:
        try:
            stream = self._service.get_blob_client(container, blob_path).download_blob()
            stream.readinto(handle)
        except Exception as exc:  # Azure exception classes remain optional.
            raise StorageError(
                f"Could not download az://{container}/{blob_path}: {exc}"
            ) from exc

    def list_blobs(self, container: str, prefix: str) -> Iterable[str]:
        try:
            for blob in self._service.get_container_client(container).list_blobs(
                name_starts_with=prefix
            ):
                yield str(blob.name)
        except Exception as exc:  # Azure exception classes remain optional.
            raise StorageError(
                f"Could not list az://{container}/{prefix}: {exc}"
            ) from exc


def parse_azure_blob_uri(value: str) -> AzureBlobLocation:
    parsed = urlsplit(value.strip())
    if parsed.scheme.lower() != "az" or not parsed.netloc:
        raise StorageError("Azure Blob roots must use az://<container>/<path>")
    if parsed.query or parsed.fragment:
        raise StorageError("Azure Blob URIs must not contain query strings or fragments")
    blob_path = str(PurePosixPath(parsed.path.lstrip("/")))
    if blob_path == ".":
        blob_path = ""
    if any(part in {"..", ""} for part in PurePosixPath(blob_path).parts):
        raise StorageError("Azure Blob URI contains an unsafe path")
    return AzureBlobLocation(parsed.netloc, blob_path)


def normalize_logical_path(value: str | Path) -> str:
    """Return a slash-separated, relative logical path with no traversal."""

    raw = str(value).strip().replace("\\", "/")
    if not raw or raw.startswith("/") or _WINDOWS_ABSOLUTE.match(raw):
        raise StorageError(f"Logical data path must be relative: {value!r}")
    path = PurePosixPath(raw)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise StorageError(f"Logical data path contains unsafe segments: {value!r}")
    return str(path)


def cache_path_for_blob(
    cache_root: Path, *, account_name: str, container: str, blob_path: str
) -> Path:
    """Expose deterministic cache mapping for validation and tests."""

    settings = StorageSettings(
        data_root=f"az://{container}",
        split_root=cache_root / "splits",
        cache_root=cache_root.resolve(),
        backend="azure_blob",
        azure_account_name=account_name,
    )
    location = AzureBlobLocation(container, normalize_logical_path(blob_path))
    account_key = settings.azure_account_name or "connection-string-account"
    relative = Path("azure_blob") / _safe_component(account_key) / _safe_component(container)
    for component in PurePosixPath(location.blob_path).parts:
        relative /= _safe_component(component)
    return (settings.cache_root / relative).resolve()


def resolver_from_environment(
    *, project_root: Path, data_root: str | None = None
) -> DataPathResolver:
    return DataPathResolver(
        StorageSettings.from_environment(project_root=project_root, data_root=data_root)
    )


def _relative_to_local_root(value: str, root: Path) -> str | None:
    if _WINDOWS_ABSOLUTE.match(value):
        root_windows = PureWindowsPath(str(root))
        value_windows = PureWindowsPath(value)
        try:
            return str(value_windows.relative_to(root_windows))
        except ValueError:
            return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        return None
    try:
        return str(candidate.resolve().relative_to(root))
    except ValueError:
        return None


def _join_posix(prefix: str, relative: str) -> str:
    return str(PurePosixPath(prefix) / PurePosixPath(relative)) if prefix else relative


def _require_within(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise StorageError(f"{label} escapes configured root {root}") from exc


def _safe_component(value: str) -> str:
    if value in {"", ".", ".."} or "/" in value or "\\" in value:
        raise StorageError(f"Unsafe storage path component: {value!r}")
    return value


def _valid_cached_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _acquire_lock(path: Path, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise StorageError(f"Timed out waiting for data-cache lock: {path}")
            time.sleep(0.05)
        else:
            os.close(descriptor)
            return True


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
