from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from orato_asr import environment


def test_environment_reports_missing_ml_as_structured_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def version(name: str) -> str:
        if name == "torch":
            raise environment.importlib.metadata.PackageNotFoundError(name)
        return "test-version"

    def missing_torch(name: str) -> object:
        assert name == "torch"
        raise ImportError("torch unavailable")

    monkeypatch.setattr(environment.importlib.metadata, "version", version)
    monkeypatch.setattr(environment.importlib, "import_module", missing_torch)
    monkeypatch.setattr(environment, "_git_commit", lambda _: "abc123")

    report = environment.collect_environment(tmp_path)

    assert report["packages"]["torch"] == {
        "status": "unavailable",
        "version": "unavailable",
    }
    assert report["pytorch"]["status"] == "unavailable"
    assert report["cuda"] == {"status": "unavailable", "available": False}
    assert report["gpus"] == []
    assert report["model"]["integration_track"] == "transformers_native"
    assert report["model"]["revision"].startswith("6aa69c")
    assert report["project"]["commit"] == "abc123"


def test_environment_reports_cuda_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda _: SimpleNamespace(
            total_memory=6_442_450_944, major=8, minor=6
        ),
        get_device_name=lambda _: "Fake RTX",
        is_bf16_supported=lambda: True,
    )
    torch = SimpleNamespace(__version__="2.11.0+cu128", cuda=cuda, version=SimpleNamespace(cuda="12.8"))
    monkeypatch.setattr(environment.importlib, "import_module", lambda _: torch)
    monkeypatch.setattr(environment.importlib.metadata, "version", lambda _: "1")

    report = environment.collect_environment(include_ml=True)

    assert report["pytorch"]["version"] == "2.11.0+cu128"
    assert report["cuda"]["available"] is True
    assert report["gpus"] == [
        {
            "index": 0,
            "name": "Fake RTX",
            "total_memory_bytes": 6_442_450_944,
            "compute_capability": "8.6",
            "bfloat16_supported": True,
        }
    ]


def test_environment_can_skip_torch_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(environment.importlib.metadata, "version", lambda _: "1")

    def unexpected(_: str) -> object:
        raise AssertionError("torch import was not expected")

    monkeypatch.setattr(environment.importlib, "import_module", unexpected)
    report = environment.collect_environment(include_ml=False)

    assert report["pytorch"] == {"status": "not_checked"}
