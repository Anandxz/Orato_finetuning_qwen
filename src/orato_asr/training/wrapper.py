"""Lazy lifecycle for the isolated official qwen-asr wrapper backend."""

from __future__ import annotations

import gc
import importlib
import importlib.metadata
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..audio import DecodedAudio
from ..exceptions import (
    DependencyError,
    TrainingError,
    TrainingOOMError,
    WrapperCompatibilityError,
)
from ..models.qwen3_asr import sanitize_error
from .config import WRAPPER_MODEL_ID, WRAPPER_MODEL_REVISION
from .official_sft import patch_outer_forward, qwen_language_name, validate_finite_inputs

WRAPPER_REQUIRED_VERSIONS = {
    "torch": "2.11.0",
    "qwen-asr": "0.0.6",
    "transformers": "4.57.6",
    "accelerate": "1.12.0",
    "peft": "0.19.1",
    "numpy": "2.3.5",
    "soundfile": "0.14.0",
    "soxr": "1.1.0",
    "librosa": "0.11.0",
    "qwen-omni-utils": "0.0.9",
    "nagisa": "0.2.11",
    "soynlp": "0.0.493",
    "gradio": "6.17.3",
    "flask": "3.1.3",
    "pytz": "2026.2",
    "huggingface-hub": "0.36.0",
    "tokenizers": "0.22.2",
    "safetensors": "0.8.0",
    "sox": "1.5.0",
}


@dataclass(slots=True)
class LoadedWrapper:
    """One loaded wrapper/model/processor pair with explicit cleanup."""

    wrapper: Any
    model: Any
    processor: Any
    torch: Any
    snapshot_path: Path
    load_seconds: float

    def close(self) -> None:
        self.wrapper = None
        self.model = None
        self.processor = None
        release_cuda(self.torch)

    def __enter__(self) -> "LoadedWrapper":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def wrapper_dependency_status() -> dict[str, dict[str, str | bool]]:
    result: dict[str, dict[str, str | bool]] = {}
    for distribution, required in WRAPPER_REQUIRED_VERSIONS.items():
        try:
            installed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = {
                "available": False,
                "installed": "unavailable",
                "required": required,
                "matches": False,
            }
        else:
            result[distribution] = {
                "available": True,
                "installed": installed,
                "required": required,
                "matches": installed.split("+")[0] == required,
            }
    return result


def require_wrapper_dependencies() -> dict[str, dict[str, str | bool]]:
    status = wrapper_dependency_status()
    mismatches = [
        f"{name}={values['installed']} (required {values['required']})"
        for name, values in status.items()
        if not values["matches"]
    ]
    if mismatches:
        raise DependencyError(
            "Wrapper-LoRA environment is incomplete or inconsistent: "
            + "; ".join(mismatches)
            + ". Activate .venv-qwen-wrapper and install requirements/wrapper-lora.txt."
        )
    return status


def resolve_wrapper_snapshot(
    *,
    cache_dir: str | Path | None,
    offline: bool,
) -> Path:
    """Resolve one exact snapshot so model and processor cannot drift revisions."""

    try:
        hub = importlib.import_module("huggingface_hub")
    except (ImportError, OSError) as exc:
        raise DependencyError("huggingface-hub is unavailable in the wrapper environment") from exc
    kwargs: dict[str, Any] = {
        "repo_id": WRAPPER_MODEL_ID,
        "revision": WRAPPER_MODEL_REVISION,
        "local_files_only": offline,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(Path(cache_dir).expanduser().resolve())
    try:
        snapshot = hub.snapshot_download(**kwargs)
    except Exception as exc:
        mode = "local cache" if offline else "Hugging Face model access"
        raise TrainingError(
            f"Could not resolve pinned wrapper model {WRAPPER_MODEL_ID}@"
            f"{WRAPPER_MODEL_REVISION} from {mode}: {sanitize_error(exc)}"
        ) from exc
    resolved = Path(snapshot).expanduser().resolve()
    if resolved.name != WRAPPER_MODEL_REVISION:
        raise WrapperCompatibilityError(
            f"Resolved wrapper snapshot {resolved.name!r} does not match pinned revision "
            f"{WRAPPER_MODEL_REVISION}"
        )
    return resolved


def load_wrapper_processor(
    *, cache_dir: str | Path | None = None, offline: bool = True
) -> tuple[Any, Path]:
    """Load only the exact official processor after qwen-asr registration."""

    require_wrapper_dependencies()
    snapshot = resolve_wrapper_snapshot(cache_dir=cache_dir, offline=offline)
    try:
        importlib.import_module("qwen_asr")
        transformers = importlib.import_module("transformers")
        processor = transformers.AutoProcessor.from_pretrained(
            str(snapshot), fix_mistral_regex=True, local_files_only=True
        )
    except Exception as exc:
        raise WrapperCompatibilityError(
            f"Could not load pinned wrapper processor: {sanitize_error(exc)}"
        ) from exc
    if processor.__class__.__name__ != "Qwen3ASRProcessor":
        raise WrapperCompatibilityError(
            f"Expected Qwen3ASRProcessor; received {processor.__class__.__name__}"
        )
    return processor, snapshot


def load_wrapper_model(
    *,
    cache_dir: str | Path | None = None,
    offline: bool = True,
    training: bool,
) -> LoadedWrapper:
    """Load one BF16 CUDA wrapper model with no device-map/offload fallback."""

    require_wrapper_dependencies()
    try:
        torch = importlib.import_module("torch")
        qwen_asr = importlib.import_module("qwen_asr")
    except (ImportError, OSError) as exc:
        raise DependencyError("Torch or qwen-asr is unavailable in the wrapper environment") from exc
    if not bool(torch.cuda.is_available()):
        raise TrainingError("Wrapper training requires CUDA; no CPU fallback is permitted")
    if not bool(torch.cuda.is_bf16_supported()):
        raise TrainingError("Wrapper training requires CUDA BF16 support on this profile")

    release_cuda(torch)
    torch.cuda.reset_peak_memory_stats()
    snapshot = resolve_wrapper_snapshot(cache_dir=cache_dir, offline=offline)
    started = time.perf_counter()
    wrapper: Any = None
    try:
        wrapper = qwen_asr.Qwen3ASRModel.from_pretrained(
            str(snapshot),
            dtype=torch.bfloat16,
            device_map=None,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        model = wrapper.model
        if model.__class__.__name__ != "Qwen3ASRForConditionalGeneration":
            raise WrapperCompatibilityError(
                f"Expected Qwen3ASRForConditionalGeneration; received {model.__class__.__name__}"
            )
        if wrapper.processor.__class__.__name__ != "Qwen3ASRProcessor":
            raise WrapperCompatibilityError(
                f"Expected Qwen3ASRProcessor; received {wrapper.processor.__class__.__name__}"
            )
        if not training and bool(getattr(model.__class__, "_forward_patched", False)):
            raise WrapperCompatibilityError(
                "Wrapper inference cannot reuse a process whose model class was patched "
                "for training; start a fresh process"
            )
        model = model.to("cuda")
        wrapper.model = model
        wrapper.device = torch.device("cuda")
        wrapper.dtype = torch.bfloat16
        if training:
            patch_outer_forward(model)
            model.train()
            _set_use_cache(model, False)
        else:
            model.eval()
        torch.cuda.synchronize()
    except Exception as exc:
        wrapper = None
        release_cuda(torch)
        if _is_cuda_oom(exc, torch):
            raise TrainingOOMError(
                "CUDA OOM while loading the unquantized wrapper model; no CPU or disk "
                "offload was attempted"
            ) from exc
        if isinstance(exc, WrapperCompatibilityError):
            raise
        raise TrainingError(f"Could not load wrapper model: {sanitize_error(exc)}") from exc

    return LoadedWrapper(
        wrapper=wrapper,
        model=model,
        processor=wrapper.processor,
        torch=torch,
        snapshot_path=snapshot,
        load_seconds=time.perf_counter() - started,
    )


def wrapper_inference(
    loaded: LoadedWrapper,
    audio: DecodedAudio,
    *,
    language: str | None,
) -> dict[str, Any]:
    """Run one deterministic wrapper transcription for compatibility evidence."""

    torch = loaded.torch
    loaded.model.eval()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            results = loaded.wrapper.transcribe(
                audio=(audio.samples, audio.sample_rate),
                language=None if language is None else qwen_language_name(language),
            )
        torch.cuda.synchronize()
    except Exception as exc:
        if _is_cuda_oom(exc, torch):
            raise TrainingOOMError("CUDA OOM during wrapper compatibility inference") from exc
        raise TrainingError(f"Wrapper inference failed: {sanitize_error(exc)}") from exc
    if len(results) != 1:
        raise WrapperCompatibilityError(
            f"Wrapper inference returned {len(results)} results for one sample"
        )
    transcript = str(getattr(results[0], "text", "") or "").strip()
    result_language = str(getattr(results[0], "language", "") or "").strip()
    if not transcript:
        raise WrapperCompatibilityError("Wrapper inference produced an empty transcript")
    return {
        "status": "success",
        "transcript": transcript,
        "language": result_language,
        "inference_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def move_batch_to_cuda(inputs: Mapping[str, Any], torch: Any) -> dict[str, Any]:
    """Move one official batch without casting integer IDs or masks."""

    moved: dict[str, Any] = {}
    for name, value in inputs.items():
        if not bool(torch.is_tensor(value)):
            moved[name] = value
        elif bool(value.is_floating_point()):
            moved[name] = value.to(device="cuda", dtype=torch.bfloat16)
        else:
            moved[name] = value.to(device="cuda")
    validate_finite_inputs(moved, torch)
    return moved


def finite_forward(
    model: Any,
    inputs: Mapping[str, Any],
    torch: Any,
    *,
    no_grad: bool,
) -> dict[str, Any]:
    """Run one official forward and reject missing, NaN, or infinite loss."""

    started = time.perf_counter()
    try:
        context = torch.no_grad() if no_grad else _null_context()
        with context:
            outputs = model(**inputs)
        loss = getattr(outputs, "loss", None)
        if loss is None:
            raise TrainingError("Wrapper model forward returned no loss")
        loss_value = float(loss.detach().float().item())
        if not math.isfinite(loss_value):
            raise TrainingError(f"Wrapper model produced non-finite loss {loss_value!r}")
        torch.cuda.synchronize()
    except Exception as exc:
        if _is_cuda_oom(exc, torch):
            raise TrainingOOMError("CUDA OOM during wrapper finite-forward stage") from exc
        if isinstance(exc, TrainingError):
            raise
        raise TrainingError(f"Wrapper forward failed: {sanitize_error(exc)}") from exc
    return {
        "loss": loss_value,
        "loss_tensor": loss,
        "forward_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def release_cuda(torch: Any | None) -> None:
    gc.collect()
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except (RuntimeError, AttributeError):
        pass


def _set_use_cache(model: Any, value: bool) -> None:
    thinker = getattr(model, "thinker", None)
    for candidate in (model, thinker, getattr(thinker, "model", None)):
        config = getattr(candidate, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = value
        text_config = getattr(config, "text_config", None)
        if text_config is not None and hasattr(text_config, "use_cache"):
            text_config.use_cache = value


class _null_context:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:
        return None


def _is_cuda_oom(error: BaseException, torch: Any) -> bool:
    candidates = (
        getattr(torch, "OutOfMemoryError", None),
        getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None),
    )
    types = tuple(item for item in candidates if isinstance(item, type))
    return bool(types and isinstance(error, types)) or (
        "cuda" in str(error).lower() and "out of memory" in str(error).lower()
    )


__all__ = [
    "LoadedWrapper",
    "WRAPPER_REQUIRED_VERSIONS",
    "finite_forward",
    "load_wrapper_model",
    "load_wrapper_processor",
    "move_batch_to_cuda",
    "release_cuda",
    "require_wrapper_dependencies",
    "resolve_wrapper_snapshot",
    "wrapper_dependency_status",
    "wrapper_inference",
]
