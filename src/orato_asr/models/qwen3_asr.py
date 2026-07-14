"""Pinned native Transformers integration for one-shot Qwen3-ASR inference."""

from __future__ import annotations

import importlib
import importlib.metadata
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..audio import DecodedAudio
from ..config import INTEGRATION_TRACK, MODEL_ID, MODEL_REVISION, PROCESSOR_REVISION
from ..exceptions import (
    DependencyError,
    DeviceSelectionError,
    InferenceError,
    InferenceOOMError,
    ModelLoadError,
)

REQUIRED_VERSIONS = {
    "torch": "2.11.0",
    "transformers": "5.13.0",
    "numpy": "2.4.2",
    "soundfile": "0.14.0",
    "soxr": "1.1.0",
    "huggingface-hub": "1.23.0",
    "tokenizers": "0.22.2",
    "safetensors": "0.8.0",
}
SUPPORTED_LANGUAGES = {
    "ar", "Arabic", "cs", "Czech", "da", "Danish", "de", "German",
    "el", "Greek", "en", "English", "es", "Spanish", "fa", "Persian",
    "fi", "Finnish", "fil", "Filipino", "fr", "French", "hi", "Hindi",
    "hu", "Hungarian", "id", "Indonesian", "it", "Italian", "ja", "Japanese",
    "ko", "Korean", "mk", "Macedonian", "ms", "Malay", "nl", "Dutch",
    "pl", "Polish", "pt", "Portuguese", "ro", "Romanian", "ru", "Russian",
    "sv", "Swedish", "th", "Thai", "tr", "Turkish", "vi", "Vietnamese",
    "yue", "Cantonese", "zh", "Chinese",
}


@dataclass(frozen=True, slots=True)
class RuntimeSelection:
    device: str
    precision: str
    dtype: Any


@dataclass(frozen=True, slots=True)
class InferenceResult:
    status: str
    error: str | None
    audio: dict[str, Any]
    transcript: str | None
    language: str | None
    model: dict[str, str]
    device: str
    precision: str
    load_seconds: float
    inference_seconds: float
    real_time_factor: float | None
    peak_cuda_memory_bytes: int | None
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "error": self.error,
            "audio": dict(self.audio),
            "transcript": self.transcript,
            "language": self.language,
            "model": dict(self.model),
            "device": self.device,
            "precision": self.precision,
            "timing": {
                "load_seconds": self.load_seconds,
                "inference_seconds": self.inference_seconds,
                "real_time_factor": self.real_time_factor,
            },
            "peak_cuda_memory_bytes": self.peak_cuda_memory_bytes,
            "warnings": list(self.warnings),
        }


def dependency_status() -> dict[str, dict[str, str | bool]]:
    """Report whether the exact direct inference pins are installed."""

    result: dict[str, dict[str, str | bool]] = {}
    for package, required in REQUIRED_VERSIONS.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = {
                "available": False,
                "installed": "unavailable",
                "required": required,
                "matches": False,
            }
        else:
            result[package] = {
                "available": True,
                "installed": installed,
                "required": required,
                "matches": installed.split("+")[0] == required,
            }
    return result


def validate_language(language: str | None) -> str | None:
    if language is None:
        return None
    normalized = language.strip()
    if not normalized:
        raise InferenceError("Language hint must be non-empty when provided")
    if normalized not in SUPPORTED_LANGUAGES:
        raise InferenceError(
            f"Unsupported Qwen3-ASR language hint {normalized!r}; use a documented "
            "language name/code such as Hindi/hi or English/en, or omit the hint"
        )
    return normalized


def select_runtime(device: str, precision: str, torch_module: Any) -> RuntimeSelection:
    """Resolve an explicit supported device/dtype combination without fallback."""

    if device not in {"auto", "cpu", "cuda"}:
        raise DeviceSelectionError(f"Unsupported device {device!r}")
    if precision not in {"auto", "float32", "float16", "bfloat16"}:
        raise DeviceSelectionError(f"Unsupported precision {precision!r}")

    cuda_available = bool(torch_module.cuda.is_available())
    resolved_device = "cuda" if device == "auto" and cuda_available else (
        "cpu" if device == "auto" else device
    )
    if resolved_device == "cuda" and not cuda_available:
        raise DeviceSelectionError(
            "CUDA was requested but torch.cuda.is_available() is false; check the "
            "NVIDIA driver and install the official PyTorch CUDA 12.8 wheel"
        )

    if resolved_device == "cpu":
        if precision in {"float16", "bfloat16"}:
            raise DeviceSelectionError(
                "CPU inference supports only precision auto or float32 in this milestone"
            )
        return RuntimeSelection("cpu", "float32", torch_module.float32)

    bf16_supported = bool(torch_module.cuda.is_bf16_supported())
    if precision == "auto":
        resolved_precision = "bfloat16" if bf16_supported else "float16"
    else:
        resolved_precision = precision
    if resolved_precision == "bfloat16" and not bf16_supported:
        raise DeviceSelectionError(
            "bfloat16 was requested but this CUDA device does not report BF16 support"
        )
    if resolved_precision == "float32":
        raise DeviceSelectionError(
            "CUDA float32 inference is excluded from this unquantized qualification; "
            "use auto, bfloat16, or float16"
        )
    return RuntimeSelection(
        "cuda",
        resolved_precision,
        getattr(torch_module, resolved_precision),
    )


class Qwen3ASREngine:
    """One pinned native processor/model pair with explicit lifecycle control."""

    def __init__(
        self,
        *,
        device: str,
        precision: str,
        cache_dir: str | Path,
        offline: bool,
        language: str | None,
        max_new_tokens: int,
    ) -> None:
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
            raise InferenceError("max_new_tokens must be an integer from 1 through 4096")
        self.requested_device = device
        self.requested_precision = precision
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.offline = bool(offline)
        self.language = validate_language(language)
        self.max_new_tokens = max_new_tokens
        self.processor: Any = None
        self.model: Any = None
        self.torch: Any = None
        self.selection: RuntimeSelection | None = None
        self.load_seconds = 0.0

    def load(self) -> dict[str, Any]:
        """Load the pinned processor and model with no device_map or Accelerate."""

        if self.model is not None:
            return self.model_info()
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except (ImportError, OSError) as exc:
            raise DependencyError(
                "Native inference dependencies are unavailable; install torch==2.11.0 "
                "from the official CPU/CUDA index and requirements/inference.txt"
            ) from exc

        selection = select_runtime(
            self.requested_device, self.requested_precision, torch
        )
        loader_kwargs = {
            "revision": MODEL_REVISION,
            "cache_dir": str(self.cache_dir),
            "local_files_only": self.offline,
        }
        started = time.perf_counter()
        try:
            processor = transformers.AutoProcessor.from_pretrained(
                MODEL_ID, **loader_kwargs
            )
            model = transformers.AutoModelForMultimodalLM.from_pretrained(
                MODEL_ID,
                dtype=selection.dtype,
                **loader_kwargs,
            )
            model = model.to(selection.device)
            model.eval()
        except Exception as exc:
            self._release(torch)
            if _is_cuda_oom(exc, torch):
                raise InferenceOOMError(
                    "CUDA ran out of memory while loading the unquantized model; no CPU "
                    "fallback was attempted. Retry on a larger GPU."
                ) from exc
            mode = "offline cache" if self.offline else "Hugging Face model access"
            raise ModelLoadError(
                f"Could not load pinned {MODEL_ID}@{MODEL_REVISION} from {mode}: "
                f"{sanitize_error(exc)}"
            ) from exc

        self.torch = torch
        self.processor = processor
        self.model = model
        self.selection = selection
        self.load_seconds = time.perf_counter() - started
        return self.model_info()

    def model_info(self) -> dict[str, Any]:
        return {
            "status": "loaded" if self.model is not None else "not_loaded",
            "integration_track": INTEGRATION_TRACK,
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "processor_revision": PROCESSOR_REVISION,
            "device": self.selection.device if self.selection else self.requested_device,
            "precision": (
                self.selection.precision if self.selection else self.requested_precision
            ),
            "cache_dir": str(self.cache_dir),
            "offline": self.offline,
            "load_seconds": self.load_seconds,
        }

    def transcribe(self, audio: DecodedAudio) -> InferenceResult:
        if self.model is None:
            self.load()
        assert self.processor is not None
        assert self.model is not None
        assert self.torch is not None
        assert self.selection is not None

        warnings: list[str] = []
        if audio.downmixed:
            warnings.append("Audio was downmixed to mono in memory")
        if audio.resampled:
            warnings.append("Audio was resampled to 16 kHz in memory")
        if self.selection.device == "cuda":
            self.torch.cuda.reset_peak_memory_stats()
            self.torch.cuda.synchronize()

        request_kwargs: dict[str, Any] = {"audio": audio.samples}
        if self.language is not None:
            request_kwargs["language"] = self.language
        started = time.perf_counter()
        try:
            inputs = self.processor.apply_transcription_request(**request_kwargs)
            inputs = inputs.to(self.selection.device, self.selection.dtype)
            input_length = inputs["input_ids"].shape[1]
            with self.torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            if self.selection.device == "cuda":
                self.torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            generated_ids = output_ids[:, input_length:]
            parsed = self.processor.decode(generated_ids, return_format="parsed")[0]
            transcript = str(parsed.get("transcription") or "").strip()
            language = parsed.get("language")
            peak_memory = (
                int(self.torch.cuda.max_memory_allocated())
                if self.selection.device == "cuda"
                else None
            )
        except Exception as exc:
            if _is_cuda_oom(exc, self.torch):
                raise InferenceOOMError(
                    "CUDA ran out of memory during generation; no CPU fallback was attempted"
                ) from exc
            if isinstance(exc, ValueError) and self.language is not None:
                raise InferenceError(
                    f"The processor rejected language hint {self.language!r}: "
                    f"{sanitize_error(exc)}"
                ) from exc
            raise InferenceError(f"Native Qwen3-ASR inference failed: {sanitize_error(exc)}") from exc

        return InferenceResult(
            status="success",
            error=None,
            audio=audio.metadata(),
            transcript=transcript,
            language=str(language) if language is not None else None,
            model={
                "integration_track": INTEGRATION_TRACK,
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "processor_revision": PROCESSOR_REVISION,
            },
            device=self.selection.device,
            precision=self.selection.precision,
            load_seconds=self.load_seconds,
            inference_seconds=elapsed,
            real_time_factor=elapsed / audio.duration_seconds,
            peak_cuda_memory_bytes=peak_memory,
            warnings=tuple(warnings),
        )

    def close(self) -> None:
        self._release(self.torch)

    def _release(self, torch: Any) -> None:
        self.model = None
        self.processor = None
        self.selection = None
        if torch is not None and getattr(torch, "cuda", None) is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (RuntimeError, AttributeError):
                pass
        self.torch = None

    def __enter__(self) -> "Qwen3ASREngine":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def sanitize_error(error: BaseException) -> str:
    """Remove URLs and token-shaped values from expected CLI/report errors."""

    message = " ".join(str(error).split())
    message = re.sub(r"https?://\S+", "[redacted-url]", message)
    message = re.sub(r"\bhf_[A-Za-z0-9]{12,}\b", "[redacted-token]", message)
    return message[:1000]


def _is_cuda_oom(error: BaseException, torch: Any) -> bool:
    candidates = [getattr(torch, "OutOfMemoryError", None)]
    cuda = getattr(torch, "cuda", None)
    candidates.append(getattr(cuda, "OutOfMemoryError", None))
    exception_types = tuple(candidate for candidate in candidates if isinstance(candidate, type))
    return bool(exception_types and isinstance(error, exception_types)) or (
        "cuda" in str(error).lower() and "out of memory" in str(error).lower()
    )
