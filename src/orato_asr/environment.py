"""Lazy, JSON-serializable environment reporting for inference qualification."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import INTEGRATION_TRACK, MODEL_ID, MODEL_REVISION, PROCESSOR_REVISION

PACKAGE_NAMES = (
    "torch",
    "transformers",
    "numpy",
    "soundfile",
    "soxr",
    "huggingface-hub",
    "tokenizers",
    "safetensors",
)
QWEN_REPOSITORY_COMMIT = "7c6daf77a2421100f5fb066495372c00129d39ff"
TRANSFORMERS_TAG_COMMIT = "6af945f436d85f2b0c5dff9b14feccd27b1d470b"


def collect_environment(
    project_root: str | Path | None = None,
    *,
    include_ml: bool = True,
) -> dict[str, Any]:
    """Collect platform and optional PyTorch/CUDA facts without eager ML imports."""

    root = Path.cwd() if project_root is None else Path(project_root)
    root = root.expanduser().resolve()
    release = platform.release()
    is_wsl = platform.system() == "Linux" and (
        "microsoft" in release.lower() or "WSL_INTEROP" in os.environ
    )
    packages: dict[str, dict[str, str]] = {}
    for package in PACKAGE_NAMES:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = {"status": "unavailable", "version": "unavailable"}
        else:
            packages[package] = {"status": "available", "version": version}

    report: dict[str, Any] = {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "supported": (3, 11) <= sys.version_info[:2] < (3, 14),
            "recommended": sys.version_info[:2] == (3, 12),
        },
        "platform": {
            "system": platform.system(),
            "release": release,
            "machine": platform.machine(),
            "wsl": is_wsl,
        },
        "packages": packages,
        "model": {
            "integration_track": INTEGRATION_TRACK,
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "processor_revision": PROCESSOR_REVISION,
            "qwen_repository_commit": QWEN_REPOSITORY_COMMIT,
            "transformers_tag_commit": TRANSFORMERS_TAG_COMMIT,
        },
        "project": {
            "root": str(root),
            "commit": _git_commit(root),
        },
        "pytorch": {"status": "not_checked"},
        "cuda": {"status": "not_checked"},
        "gpus": [],
    }
    if include_ml:
        _add_torch_details(report)
    return report


def _add_torch_details(report: dict[str, Any]) -> None:
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        report["pytorch"] = {
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
        }
        report["cuda"] = {"status": "unavailable", "available": False}
        return

    report["pytorch"] = {
        "status": "available",
        "version": str(getattr(torch, "__version__", "unknown")),
    }
    cuda = getattr(torch, "cuda", None)
    available = bool(cuda is not None and cuda.is_available())
    report["cuda"] = {
        "status": "available" if available else "unavailable",
        "available": available,
        "torch_cuda_version": str(getattr(getattr(torch, "version", None), "cuda", None)),
    }
    if not available:
        return

    gpus: list[dict[str, Any]] = []
    for index in range(cuda.device_count()):
        properties = cuda.get_device_properties(index)
        gpus.append(
            {
                "index": index,
                "name": cuda.get_device_name(index),
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": f"{properties.major}.{properties.minor}",
                "bfloat16_supported": bool(cuda.is_bf16_supported()),
            }
        )
    report["gpus"] = gpus


def _git_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 else "unavailable"
