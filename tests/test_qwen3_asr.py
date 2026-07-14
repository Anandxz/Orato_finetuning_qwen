from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from orato_asr.audio import DecodedAudio
from orato_asr.exceptions import (
    DependencyError,
    DeviceSelectionError,
    InferenceError,
    InferenceOOMError,
    ModelLoadError,
)
from orato_asr.models import qwen3_asr


class _FakeCUDA:
    class OutOfMemoryError(RuntimeError):
        pass

    def __init__(self, available: bool, bf16: bool = True) -> None:
        self.available = available
        self.bf16 = bf16
        self.empty_cache_calls = 0

    def is_available(self) -> bool:
        return self.available

    def is_bf16_supported(self) -> bool:
        return self.bf16

    def reset_peak_memory_stats(self) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def max_memory_allocated(self) -> int:
        return 1234

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _FakeTorch:
    float32 = "float32-dtype"
    float16 = "float16-dtype"
    bfloat16 = "bfloat16-dtype"

    def __init__(self, available: bool = True, bf16: bool = True) -> None:
        self.cuda = _FakeCUDA(available, bf16)
        self.OutOfMemoryError = self.cuda.OutOfMemoryError

    @staticmethod
    def inference_mode() -> object:
        return nullcontext()


@pytest.mark.parametrize(
    ("available", "bf16", "device", "precision", "expected"),
    [
        (False, False, "auto", "auto", ("cpu", "float32")),
        (True, True, "auto", "auto", ("cuda", "bfloat16")),
        (True, False, "cuda", "auto", ("cuda", "float16")),
        (True, True, "cuda", "float16", ("cuda", "float16")),
    ],
)
def test_runtime_selection(
    available: bool,
    bf16: bool,
    device: str,
    precision: str,
    expected: tuple[str, str],
) -> None:
    selected = qwen3_asr.select_runtime(device, precision, _FakeTorch(available, bf16))
    assert (selected.device, selected.precision) == expected


@pytest.mark.parametrize(
    ("device", "precision", "available", "message"),
    [
        ("cuda", "auto", False, "CUDA was requested"),
        ("cpu", "float16", False, "CPU inference"),
        ("cuda", "bfloat16", True, "does not report BF16"),
        ("cuda", "float32", True, "CUDA float32"),
    ],
)
def test_unsupported_runtime_fails_without_fallback(
    device: str, precision: str, available: bool, message: str
) -> None:
    torch = _FakeTorch(available, bf16=False)
    with pytest.raises(DeviceSelectionError, match=message):
        qwen3_asr.select_runtime(device, precision, torch)


class _InputIds:
    shape = (1, 4)


class _Inputs(dict[str, object]):
    def __init__(self) -> None:
        super().__init__(input_ids=_InputIds(), input_features="features")
        self.to_call: tuple[str, object] | None = None

    def to(self, device: str, dtype: object) -> "_Inputs":
        self.to_call = (device, dtype)
        return self


class _Generated:
    def __init__(self) -> None:
        self.slice_key: object = None

    def __getitem__(self, key: object) -> str:
        self.slice_key = key
        return "generated-token-ids"


class _Processor:
    def __init__(self) -> None:
        self.request: dict[str, object] | None = None
        self.inputs = _Inputs()
        self.decode_call: tuple[object, str] | None = None

    def apply_transcription_request(self, **kwargs: object) -> _Inputs:
        self.request = kwargs
        return self.inputs

    def decode(self, ids: object, return_format: str) -> list[dict[str, str]]:
        self.decode_call = (ids, return_format)
        return [{"language": "Hindi", "transcription": "नमस्ते Orato"}]


class _Model:
    def __init__(self) -> None:
        self.to_call: str | None = None
        self.eval_called = False
        self.generate_kwargs: dict[str, object] | None = None
        self.generated = _Generated()

    def to(self, device: str) -> "_Model":
        self.to_call = device
        return self

    def eval(self) -> None:
        self.eval_called = True

    def generate(self, **kwargs: object) -> _Generated:
        self.generate_kwargs = kwargs
        return self.generated


def _fake_transformers(processor: _Processor, model: _Model) -> tuple[object, list[tuple[str, tuple[object, ...], dict[str, object]]]]:
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    class ProcessorLoader:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> _Processor:
            calls.append(("processor", args, kwargs))
            return processor

    class ModelLoader:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> _Model:
            calls.append(("model", args, kwargs))
            return model

    return SimpleNamespace(
        AutoProcessor=ProcessorLoader,
        AutoModelForMultimodalLM=ModelLoader,
    ), calls


def test_pinned_loader_and_official_generation_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = _FakeTorch(available=True, bf16=True)
    processor, model = _Processor(), _Model()
    transformers, calls = _fake_transformers(processor, model)

    def fake_import(name: str) -> object:
        return torch if name == "torch" else transformers

    monkeypatch.setattr(qwen3_asr.importlib, "import_module", fake_import)
    engine = qwen3_asr.Qwen3ASREngine(
        device="cuda",
        precision="auto",
        cache_dir=tmp_path / "cache",
        offline=True,
        language="hi",
        max_new_tokens=256,
    )
    samples = object()
    decoded = DecodedAudio(
        path=tmp_path / "input.wav",
        samples=samples,
        original_sample_rate=8_000,
        sample_rate=16_000,
        original_channels=2,
        channels=1,
        duration_seconds=2.0,
        downmixed=True,
        resampled=True,
    )

    result = engine.transcribe(decoded)

    assert [call[0] for call in calls] == ["processor", "model"]
    for kind, args, kwargs in calls:
        assert args == ("Qwen/Qwen3-ASR-0.6B-hf",)
        assert kwargs["revision"] == "6aa69c382e2b426eee1f5870d4c95859a74b6445"
        assert kwargs["local_files_only"] is True
        assert kwargs["cache_dir"] == str((tmp_path / "cache").resolve())
        assert "device_map" not in kwargs
        if kind == "model":
            assert kwargs["dtype"] == torch.bfloat16
    assert model.to_call == "cuda"
    assert model.eval_called is True
    assert processor.request == {"audio": samples, "language": "hi"}
    assert processor.inputs.to_call == ("cuda", torch.bfloat16)
    assert model.generate_kwargs is not None
    assert model.generate_kwargs["max_new_tokens"] == 256
    assert model.generate_kwargs["do_sample"] is False
    assert processor.decode_call == ("generated-token-ids", "parsed")
    assert result.transcript == "नमस्ते Orato"
    assert result.language == "Hindi"
    assert result.device == "cuda"
    assert result.precision == "bfloat16"
    assert result.peak_cuda_memory_bytes == 1234
    assert result.real_time_factor is not None
    assert result.warnings == (
        "Audio was downmixed to mono in memory",
        "Audio was resampled to 16 kHz in memory",
    )
    payload = result.as_dict()
    assert payload["model"]["revision"].startswith("6aa69c")
    assert payload["status"] == "success"
    engine.close()
    assert torch.cuda.empty_cache_calls >= 1


def test_loader_cuda_oom_becomes_actionable_project_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = _FakeTorch(available=True)

    class ProcessorLoader:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> object:
            return object()

    class ModelLoader:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> object:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory token hf_secretsecret")

    transformers = SimpleNamespace(
        AutoProcessor=ProcessorLoader, AutoModelForMultimodalLM=ModelLoader
    )
    monkeypatch.setattr(
        qwen3_asr.importlib,
        "import_module",
        lambda name: torch if name == "torch" else transformers,
    )
    engine = qwen3_asr.Qwen3ASREngine(
        device="cuda",
        precision="auto",
        cache_dir=tmp_path,
        offline=False,
        language=None,
        max_new_tokens=256,
    )

    with pytest.raises(InferenceOOMError, match="no CPU fallback"):
        engine.load()


def test_missing_model_dependencies_are_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        qwen3_asr.importlib,
        "import_module",
        lambda _: (_ for _ in ()).throw(ImportError("missing")),
    )
    engine = qwen3_asr.Qwen3ASREngine(
        device="cpu",
        precision="auto",
        cache_dir=tmp_path,
        offline=False,
        language=None,
        max_new_tokens=256,
    )

    with pytest.raises(DependencyError, match="requirements/inference.txt"):
        engine.load()


def test_missing_offline_cache_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = _FakeTorch(available=False)

    class ProcessorLoader:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> object:
            raise OSError("file not found in local cache")

    transformers = SimpleNamespace(
        AutoProcessor=ProcessorLoader,
        AutoModelForMultimodalLM=object(),
    )
    monkeypatch.setattr(
        qwen3_asr.importlib,
        "import_module",
        lambda name: torch if name == "torch" else transformers,
    )
    engine = qwen3_asr.Qwen3ASREngine(
        device="cpu",
        precision="auto",
        cache_dir=tmp_path / "missing-cache",
        offline=True,
        language=None,
        max_new_tokens=256,
    )

    with pytest.raises(ModelLoadError, match="offline cache"):
        engine.load()


def test_language_validation_and_error_sanitization() -> None:
    assert qwen3_asr.validate_language("Hindi") == "Hindi"
    assert qwen3_asr.validate_language("hi") == "hi"
    assert qwen3_asr.validate_language(None) is None
    with pytest.raises(InferenceError, match="Unsupported"):
        qwen3_asr.validate_language("Klingon")
    message = qwen3_asr.sanitize_error(
        RuntimeError("failed https://private.example/path hf_abcdefghijklmnop")
    )
    assert "https://" not in message
    assert "hf_" not in message
