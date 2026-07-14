"""Strict LoRA targeting and trainability checks for the Qwen wrapper model.

This module deliberately imports neither PyTorch nor PEFT at import time.  The
real PEFT package is loaded only when :func:`inject_lora` is called; tests and
preflight inspection can supply a small compatible fake instead.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from orato_asr.exceptions import DependencyError, WrapperCompatibilityError

_AUDIO_ENCODER_PATH = "thinker.audio_tower"
_TEXT_DECODER_PATH = "thinker.model"
_EMBEDDINGS_PATH = "thinker.model.embed_tokens"
_OUTPUT_HEAD_PATH = "thinker.lm_head"
_TEXT_LAYER_CONTAINER_PATH = "thinker.model.layers"
_APPROVED_MODULE_RE = re.compile(
    r"^thinker\.model\.layers\.(0|[1-9][0-9]*)\.self_attn\.(q_proj|v_proj)$"
)
_QV_CANDIDATE_RE = re.compile(r"(?:^|\.)(?:q_proj|v_proj)$")
_MLP_CANDIDATE_RE = re.compile(
    r"^thinker\.model\.layers\.(?:0|[1-9][0-9]*)\.mlp\."
    r"(?:gate_proj|up_proj|down_proj)$"
)


@dataclass(frozen=True, slots=True)
class ModuleInventoryEntry:
    """One named module without retaining another reference to the module."""

    name: str
    class_name: str

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "class_name": self.class_name}


@dataclass(frozen=True, slots=True)
class ParameterInventoryEntry:
    """Small, JSON-safe metadata for one named parameter."""

    name: str
    element_count: int
    requires_grad: bool
    shape: tuple[int, ...]
    dtype: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "element_count": self.element_count,
            "requires_grad": self.requires_grad,
            "shape": list(self.shape),
            "dtype": self.dtype,
        }


@dataclass(frozen=True, slots=True)
class LoRAModuleInventory:
    """Verified wrapper topology and exact text-decoder LoRA allowlist."""

    audio_encoder_path: str
    text_decoder_path: str
    embeddings_path: str
    output_head_path: str
    text_layer_count: int
    approved_module_paths: tuple[str, ...]
    rejected_qv_candidate_paths: tuple[str, ...]
    text_mlp_paths: tuple[str, ...]
    modules: tuple[ModuleInventoryEntry, ...]
    parameters: tuple[ParameterInventoryEntry, ...]
    total_parameter_count: int
    initially_trainable_parameter_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "audio_encoder_path": self.audio_encoder_path,
            "text_decoder_path": self.text_decoder_path,
            "embeddings_path": self.embeddings_path,
            "output_head_path": self.output_head_path,
            "text_layer_count": self.text_layer_count,
            "approved_module_paths": list(self.approved_module_paths),
            "rejected_qv_candidate_paths": list(self.rejected_qv_candidate_paths),
            "text_mlp_paths": list(self.text_mlp_paths),
            "module_count": len(self.modules),
            "parameter_tensor_count": len(self.parameters),
            "total_parameter_count": self.total_parameter_count,
            "initially_trainable_parameter_count": (
                self.initially_trainable_parameter_count
            ),
            "modules": [entry.as_dict() for entry in self.modules],
            "parameters": [entry.as_dict() for entry in self.parameters],
        }


@dataclass(frozen=True, slots=True)
class TrainableParameterAudit:
    """Proof that PEFT exposed only the exact approved adapter parameters."""

    total_parameter_count: int
    trainable_parameter_count: int
    frozen_parameter_count: int
    trainable_percentage: float
    trainable_parameter_tensor_count: int
    audio_parameter_count: int
    audio_trainable_parameter_count: int
    text_decoder_base_parameter_count: int
    text_decoder_base_trainable_parameter_count: int
    trainable_parameter_names: tuple[str, ...]
    approved_module_paths: tuple[str, ...]
    trainable_elements_by_module: Mapping[str, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "total_parameter_count": self.total_parameter_count,
            "trainable_parameter_count": self.trainable_parameter_count,
            "frozen_parameter_count": self.frozen_parameter_count,
            "trainable_percentage": self.trainable_percentage,
            "trainable_parameter_tensor_count": self.trainable_parameter_tensor_count,
            "audio_parameter_count": self.audio_parameter_count,
            "audio_trainable_parameter_count": self.audio_trainable_parameter_count,
            "text_decoder_base_parameter_count": (
                self.text_decoder_base_parameter_count
            ),
            "text_decoder_base_trainable_parameter_count": (
                self.text_decoder_base_trainable_parameter_count
            ),
            "trainable_parameter_names": list(self.trainable_parameter_names),
            "approved_module_paths": list(self.approved_module_paths),
            "trainable_elements_by_module": dict(self.trainable_elements_by_module),
        }


@dataclass(frozen=True, slots=True)
class LoRAInjectionResult:
    """Injected model plus the evidence needed by reports and training guards."""

    model: Any
    inventory: LoRAModuleInventory
    audit: TrainableParameterAudit
    peft_configuration: Mapping[str, object]
    gradient_checkpointing: bool
    gradient_checkpointing_use_reentrant: bool | None

    def as_dict(self) -> dict[str, object]:
        return {
            "inventory": self.inventory.as_dict(),
            "trainability": self.audit.as_dict(),
            "peft_configuration": dict(self.peft_configuration),
            "gradient_checkpointing": self.gradient_checkpointing,
            "gradient_checkpointing_use_reentrant": (
                self.gradient_checkpointing_use_reentrant
            ),
        }


@dataclass(frozen=True, slots=True)
class FrozenParameterChecksums:
    """Checksums of representative frozen parameters around an optimizer step."""

    checksums: Mapping[str, str]

    def as_dict(self) -> dict[str, object]:
        return {
            "algorithm": "sha256",
            "coverage": "bounded deterministic slices of representative tensors",
            "checksums": dict(self.checksums),
        }


@dataclass(frozen=True, slots=True)
class OptimizerParameterAudit:
    """Identity-level validation of parameters handed to an optimizer."""

    optimizer_parameter_tensor_count: int
    trainable_parameter_tensor_count: int
    duplicate_parameter_references: int

    def as_dict(self) -> dict[str, int]:
        return {
            "optimizer_parameter_tensor_count": self.optimizer_parameter_tensor_count,
            "trainable_parameter_tensor_count": self.trainable_parameter_tensor_count,
            "duplicate_parameter_references": self.duplicate_parameter_references,
        }


def discover_lora_inventory(model: Any) -> LoRAModuleInventory:
    """Discover and validate the exact wrapper text-decoder Q/V topology.

    Exact paths are intentional.  Both the audio tower and text decoder contain
    modules named ``q_proj`` and ``v_proj``; broad suffix matching would adapt
    the audio encoder and is rejected by design.
    """

    named_modules = _named_modules(model)
    modules_by_name = {name: module for name, module in named_modules}
    required_paths = (
        _AUDIO_ENCODER_PATH,
        _TEXT_DECODER_PATH,
        _EMBEDDINGS_PATH,
        _OUTPUT_HEAD_PATH,
        _TEXT_LAYER_CONTAINER_PATH,
    )
    missing = [path for path in required_paths if path not in modules_by_name]
    if missing:
        raise WrapperCompatibilityError(
            "Wrapper model is missing required module path(s): " + ", ".join(missing)
        )

    layers = getattr(modules_by_name[_TEXT_DECODER_PATH], "layers", None)
    try:
        layer_count = len(layers)
    except (TypeError, AttributeError):
        raise WrapperCompatibilityError(
            "Wrapper text decoder thinker.model has no finite layers collection"
        ) from None
    if type(layer_count) is not int or layer_count <= 0:
        raise WrapperCompatibilityError(
            "Wrapper text decoder must contain at least one decoder layer"
        )

    approved_by_layer: dict[int, set[str]] = {}
    approved_paths: list[str] = []
    qv_candidates: list[str] = []
    text_mlp_paths: list[str] = []
    for name, _module in named_modules:
        match = _APPROVED_MODULE_RE.fullmatch(name)
        if match is not None:
            layer_index = int(match.group(1))
            projection = match.group(2)
            approved_by_layer.setdefault(layer_index, set()).add(projection)
            approved_paths.append(name)
        if _QV_CANDIDATE_RE.search(name):
            qv_candidates.append(name)
        if _MLP_CANDIDATE_RE.fullmatch(name):
            text_mlp_paths.append(name)

    expected_indices = set(range(layer_count))
    discovered_indices = set(approved_by_layer)
    if discovered_indices != expected_indices:
        missing_indices = sorted(expected_indices - discovered_indices)
        unexpected_indices = sorted(discovered_indices - expected_indices)
        details: list[str] = []
        if missing_indices:
            details.append(f"missing layer indices {missing_indices}")
        if unexpected_indices:
            details.append(f"unexpected layer indices {unexpected_indices}")
        raise WrapperCompatibilityError(
            "Wrapper text-decoder LoRA topology is not contiguous: "
            + "; ".join(details)
        )
    for layer_index in range(layer_count):
        projections = approved_by_layer[layer_index]
        if projections != {"q_proj", "v_proj"}:
            raise WrapperCompatibilityError(
                f"Text decoder layer {layer_index} must contain exactly q_proj and "
                f"v_proj at the approved self_attn paths; found {sorted(projections)}"
            )

    approved_paths = sorted(approved_paths, key=_approved_path_sort_key)
    if len(approved_paths) != layer_count * 2:
        raise WrapperCompatibilityError(
            "Wrapper text decoder must expose exactly two approved LoRA modules per layer"
        )
    approved_set = set(approved_paths)
    rejected_qv_paths = sorted(
        path for path in qv_candidates if path not in approved_set
    )
    unexpected_qv_paths = [
        path
        for path in rejected_qv_paths
        if not path.startswith(_AUDIO_ENCODER_PATH + ".")
    ]
    if unexpected_qv_paths:
        raise WrapperCompatibilityError(
            "Wrapper contains Q/V projection candidates outside the verified text "
            "self-attention and audio-tower paths: "
            + ", ".join(unexpected_qv_paths)
        )

    named_parameters = _named_parameters(model)
    parameter_names = {name for name, _parameter in named_parameters}
    missing_weights = [
        f"{path}.weight"
        for path in approved_paths
        if f"{path}.weight" not in parameter_names
    ]
    if missing_weights:
        raise WrapperCompatibilityError(
            "Approved LoRA projection module(s) lack an exact weight parameter: "
            + ", ".join(missing_weights)
        )

    module_entries = tuple(
        ModuleInventoryEntry(name, _class_name(module))
        for name, module in sorted(named_modules, key=lambda item: item[0])
    )
    parameter_entries = tuple(
        ParameterInventoryEntry(
            name=name,
            element_count=_parameter_numel(parameter, name),
            requires_grad=bool(getattr(parameter, "requires_grad", False)),
            shape=_parameter_shape(parameter),
            dtype=str(getattr(parameter, "dtype", "unknown")),
        )
        for name, parameter in sorted(named_parameters, key=lambda item: item[0])
    )
    total = sum(entry.element_count for entry in parameter_entries)
    trainable = sum(
        entry.element_count for entry in parameter_entries if entry.requires_grad
    )
    return LoRAModuleInventory(
        audio_encoder_path=_AUDIO_ENCODER_PATH,
        text_decoder_path=_TEXT_DECODER_PATH,
        embeddings_path=_EMBEDDINGS_PATH,
        output_head_path=_OUTPUT_HEAD_PATH,
        text_layer_count=layer_count,
        approved_module_paths=tuple(approved_paths),
        rejected_qv_candidate_paths=tuple(rejected_qv_paths),
        text_mlp_paths=tuple(sorted(text_mlp_paths)),
        modules=module_entries,
        parameters=parameter_entries,
        total_parameter_count=total,
        initially_trainable_parameter_count=trainable,
    )


def inject_lora(
    model: Any,
    *,
    rank: int = 4,
    alpha: int = 16,
    dropout: float = 0.05,
    bias: str = "none",
    peft_module: Any | None = None,
    gradient_checkpointing: bool = True,
) -> LoRAInjectionResult:
    """Freeze the base model and inject PEFT into only exact text Q/V paths.

    The caller must apply Qwen's official outer-forward patch before this
    function.  Gradient checkpointing is enabled *after* adapter injection and
    explicitly requests the non-reentrant implementation.
    """

    _validate_lora_settings(rank, alpha, dropout, bias)
    if type(gradient_checkpointing) is not bool:
        raise WrapperCompatibilityError(
            "gradient_checkpointing must be explicitly true or false"
        )
    if not bool(getattr(model.__class__, "_forward_patched", False)):
        raise WrapperCompatibilityError(
            "Apply the official Qwen wrapper outer-forward patch before LoRA injection"
        )
    inventory = discover_lora_inventory(model)
    for _name, parameter in _named_parameters(model):
        parameter.requires_grad = False

    peft = peft_module if peft_module is not None else _import_peft()
    for attribute in ("LoraConfig", "TaskType", "get_peft_model"):
        if not hasattr(peft, attribute):
            raise WrapperCompatibilityError(
                f"PEFT integration object lacks required attribute {attribute}"
            )
    task_type = getattr(peft.TaskType, "CAUSAL_LM", None)
    if task_type is None:
        raise WrapperCompatibilityError("PEFT TaskType.CAUSAL_LM is unavailable")

    configuration_values: dict[str, object] = {
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": float(dropout),
        "bias": bias,
        "target_modules": list(inventory.approved_module_paths),
        "task_type": task_type,
        "inference_mode": False,
    }
    try:
        peft_configuration = peft.LoraConfig(**configuration_values)
        injected_model = peft.get_peft_model(model, peft_configuration)
    except WrapperCompatibilityError:
        raise
    except Exception as exc:
        raise WrapperCompatibilityError(
            f"PEFT failed to inject the exact Qwen text-decoder LoRA allowlist: {exc}"
        ) from exc
    if injected_model is None:
        raise WrapperCompatibilityError("PEFT returned no model after LoRA injection")

    checkpointing_use_reentrant: bool | None = None
    if gradient_checkpointing:
        enable_non_reentrant_gradient_checkpointing(injected_model)
        checkpointing_use_reentrant = False
    audit = audit_trainable_parameters(
        injected_model,
        inventory.approved_module_paths,
    )
    report_configuration = {
        **configuration_values,
        "task_type": str(task_type),
        "target_modules": list(inventory.approved_module_paths),
    }
    return LoRAInjectionResult(
        model=injected_model,
        inventory=inventory,
        audit=audit,
        peft_configuration=MappingProxyType(report_configuration),
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_use_reentrant=checkpointing_use_reentrant,
    )


def enable_non_reentrant_gradient_checkpointing(model: Any) -> None:
    """Enable checkpointing using the memory-safe non-reentrant variant."""

    enable = getattr(model, "gradient_checkpointing_enable", None)
    if not callable(enable):
        raise WrapperCompatibilityError(
            "Injected wrapper model does not support gradient checkpointing"
        )
    try:
        enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception as exc:
        raise WrapperCompatibilityError(
            "Could not enable non-reentrant gradient checkpointing after LoRA "
            f"injection: {exc}"
        ) from exc
    for candidate in (
        model,
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
    ):
        config = getattr(candidate, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False


def audit_trainable_parameters(
    model: Any,
    approved_module_paths: Sequence[str],
) -> TrainableParameterAudit:
    """Abort unless every and only expected A/B adapter weight is trainable."""

    approved = tuple(approved_module_paths)
    _validate_approved_path_sequence(approved)
    expected_parts = {
        (module_path, matrix)
        for module_path in approved
        for matrix in ("A", "B")
    }
    observed_parts: dict[tuple[str, str], str] = {}
    trainable_names: list[str] = []
    elements_by_module = {module_path: 0 for module_path in approved}
    total = 0
    trainable = 0
    audio_total = 0
    audio_trainable = 0
    text_base_total = 0
    text_base_trainable = 0

    for name, parameter in _named_parameters(model):
        count = _parameter_numel(parameter, name)
        is_trainable = bool(getattr(parameter, "requires_grad", False))
        total += count
        if _contains_module_path(name, _AUDIO_ENCODER_PATH):
            audio_total += count
            if is_trainable:
                audio_trainable += count
        lora_part = _parse_approved_lora_parameter(name, approved)
        is_text_parameter = _contains_module_path(name, _TEXT_DECODER_PATH)
        if is_text_parameter and lora_part is None:
            text_base_total += count
            if is_trainable:
                text_base_trainable += count
        if not is_trainable:
            continue
        trainable += count
        trainable_names.append(name)
        if lora_part is None:
            raise WrapperCompatibilityError(
                f"Unexpected trainable parameter outside exact LoRA allowlist: {name}"
            )
        if lora_part in observed_parts:
            raise WrapperCompatibilityError(
                "Multiple trainable adapter tensors represent "
                f"{lora_part[0]} lora_{lora_part[1]}: "
                f"{observed_parts[lora_part]}, {name}"
            )
        observed_parts[lora_part] = name
        elements_by_module[lora_part[0]] += count

    if audio_trainable != 0:
        raise WrapperCompatibilityError(
            f"Audio encoder has {audio_trainable} trainable parameters; expected zero"
        )
    if text_base_trainable != 0:
        raise WrapperCompatibilityError(
            "Base text-decoder weights remain trainable; only LoRA A/B weights are allowed"
        )
    missing = sorted(expected_parts - set(observed_parts))
    unexpected = sorted(set(observed_parts) - expected_parts)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(
                "missing "
                + ", ".join(f"{path}.lora_{matrix}" for path, matrix in missing)
            )
        if unexpected:
            details.append(
                "unexpected "
                + ", ".join(
                    f"{path}.lora_{matrix}" for path, matrix in unexpected
                )
            )
        raise WrapperCompatibilityError(
            "LoRA trainable tensor coverage is invalid: " + "; ".join(details)
        )
    if trainable <= 0 or total <= 0:
        raise WrapperCompatibilityError(
            "LoRA injection produced no trainable parameters or an empty model"
        )

    return TrainableParameterAudit(
        total_parameter_count=total,
        trainable_parameter_count=trainable,
        frozen_parameter_count=total - trainable,
        trainable_percentage=(trainable / total) * 100.0,
        trainable_parameter_tensor_count=len(trainable_names),
        audio_parameter_count=audio_total,
        audio_trainable_parameter_count=audio_trainable,
        text_decoder_base_parameter_count=text_base_total,
        text_decoder_base_trainable_parameter_count=text_base_trainable,
        trainable_parameter_names=tuple(sorted(trainable_names)),
        approved_module_paths=approved,
        trainable_elements_by_module=MappingProxyType(dict(elements_by_module)),
    )


def optimizer_trainable_parameters(model: Any) -> tuple[Any, ...]:
    """Return the exact adapter-only parameter tuple for optimizer creation."""

    parameters = tuple(
        parameter
        for _name, parameter in _named_parameters(model)
        if bool(getattr(parameter, "requires_grad", False))
    )
    if not parameters:
        raise WrapperCompatibilityError(
            "Cannot create an optimizer because the model has no trainable parameters"
        )
    identities = [id(parameter) for parameter in parameters]
    if len(set(identities)) != len(identities):
        raise WrapperCompatibilityError(
            "Model named_parameters returned duplicate trainable parameter references"
        )
    return parameters


def capture_trainable_parameter_copies(model: Any) -> Mapping[str, Any]:
    """Capture small adapter-only CPU copies before an optimizer step."""

    captured: dict[str, Any] = {}
    for name, parameter in _named_parameters(model):
        if not bool(getattr(parameter, "requires_grad", False)):
            continue
        try:
            captured[name] = parameter.detach().cpu().clone()
        except Exception as exc:
            raise WrapperCompatibilityError(
                f"Could not snapshot trainable adapter parameter {name}: {exc}"
            ) from exc
    if not captured:
        raise WrapperCompatibilityError(
            "Cannot snapshot adapter changes because no parameters are trainable"
        )
    return MappingProxyType(captured)


def verify_trainable_parameter_change(
    model: Any,
    before: Mapping[str, Any],
    *,
    tensor_equal: Any,
) -> None:
    """Require at least one exact trainable adapter tensor to change."""

    if not isinstance(before, Mapping) or not before:
        raise WrapperCompatibilityError("Pre-step adapter snapshot is empty or invalid")
    if not callable(tensor_equal):
        raise WrapperCompatibilityError("Adapter tensor equality function is unavailable")
    current = dict(_named_parameters(model))
    changed = False
    for name, previous in before.items():
        parameter = current.get(name)
        if parameter is None:
            raise WrapperCompatibilityError(
                f"Trainable adapter parameter disappeared after optimizer step: {name}"
            )
        if not bool(getattr(parameter, "requires_grad", False)):
            raise WrapperCompatibilityError(
                f"Adapter parameter became frozen after optimizer step: {name}"
            )
        try:
            equal = bool(tensor_equal(previous, parameter.detach().cpu()))
        except Exception as exc:
            raise WrapperCompatibilityError(
                f"Could not compare adapter parameter {name}: {exc}"
            ) from exc
        changed = changed or not equal
    if not changed:
        raise WrapperCompatibilityError(
            "Optimizer steps did not change any LoRA adapter parameter"
        )


def validate_optimizer_parameter_ids(
    optimizer: Any,
    model: Any,
) -> OptimizerParameterAudit:
    """Require optimizer groups to contain each trainable adapter exactly once."""

    expected = optimizer_trainable_parameters(model)
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, (list, tuple)):
        raise WrapperCompatibilityError("Optimizer has no parameter-group sequence")
    optimizer_parameters: list[Any] = []
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping) or "params" not in group:
            raise WrapperCompatibilityError(
                f"Optimizer parameter group {index} lacks a params collection"
            )
        try:
            optimizer_parameters.extend(list(group["params"]))
        except TypeError as exc:
            raise WrapperCompatibilityError(
                f"Optimizer parameter group {index} params are not iterable"
            ) from exc

    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
    expected_ids = {id(parameter) for parameter in expected}
    duplicate_count = len(optimizer_ids) - len(set(optimizer_ids))
    optimizer_id_set = set(optimizer_ids)
    if duplicate_count:
        raise WrapperCompatibilityError(
            f"Optimizer contains {duplicate_count} duplicate parameter reference(s)"
        )
    if optimizer_id_set != expected_ids:
        missing_count = len(expected_ids - optimizer_id_set)
        unexpected_count = len(optimizer_id_set - expected_ids)
        raise WrapperCompatibilityError(
            "Optimizer parameters do not exactly equal trainable LoRA parameters: "
            f"missing={missing_count}, unexpected_or_frozen={unexpected_count}"
        )
    return OptimizerParameterAudit(
        optimizer_parameter_tensor_count=len(optimizer_parameters),
        trainable_parameter_tensor_count=len(expected),
        duplicate_parameter_references=0,
    )


def capture_representative_frozen_checksums(model: Any) -> FrozenParameterChecksums:
    """Hash one deterministic frozen parameter from each critical base scope."""

    # Qwen ties the LM head to the token embedding.  PyTorch's default
    # named_parameters() iterator removes duplicate tensor aliases, which can
    # hide thinker.lm_head.weight even though that exact module must remain
    # frozen.  Retain aliases for this bounded checksum audit only; inventory
    # and trainable counts continue to use the de-duplicated iterator.
    named = _named_parameters_with_aliases(model)
    scopes = (
        ("audio_encoder", _AUDIO_ENCODER_PATH),
        ("text_decoder", _TEXT_LAYER_CONTAINER_PATH),
        ("embeddings", _EMBEDDINGS_PATH),
        ("output_head", _OUTPUT_HEAD_PATH),
    )
    selected: dict[str, str] = {}
    for scope_name, module_path in scopes:
        candidates = [
            (name, parameter)
            for name, parameter in named
            if _contains_module_path(name, module_path)
            and ".lora_" not in name
            and not bool(getattr(parameter, "requires_grad", False))
        ]
        if not candidates:
            raise WrapperCompatibilityError(
                f"No frozen parameter is available for representative {scope_name} checksum"
            )
        name, parameter = sorted(candidates, key=lambda item: item[0])[0]
        selected[name] = _parameter_checksum(parameter, name)
    return FrozenParameterChecksums(MappingProxyType(selected))


def verify_frozen_parameter_checksums(
    model: Any,
    expected: FrozenParameterChecksums | Mapping[str, str],
) -> None:
    """Abort if a representative base weight changed or became trainable."""

    expected_values = (
        dict(expected.checksums)
        if isinstance(expected, FrozenParameterChecksums)
        else dict(expected)
    )
    parameters = dict(_named_parameters_with_aliases(model))
    for name, expected_checksum in expected_values.items():
        parameter = parameters.get(name)
        if parameter is None:
            raise WrapperCompatibilityError(
                f"Representative frozen parameter disappeared after optimizer step: {name}"
            )
        if bool(getattr(parameter, "requires_grad", False)):
            raise WrapperCompatibilityError(
                f"Representative base parameter unexpectedly became trainable: {name}"
            )
        actual = _parameter_checksum(parameter, name)
        if actual != expected_checksum:
            raise WrapperCompatibilityError(
                f"Frozen base parameter changed during adapter optimization: {name}"
            )


def _validate_lora_settings(rank: int, alpha: int, dropout: float, bias: str) -> None:
    if type(rank) is not int or rank not in {2, 4}:
        raise WrapperCompatibilityError("LoRA rank must be 4, or 2 for the final OOM fallback")
    if type(alpha) is not int or alpha not in {8, 16} or alpha < rank:
        raise WrapperCompatibilityError("LoRA alpha must be 8 or 16 and at least rank")
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise WrapperCompatibilityError("LoRA dropout must be a finite number")
    if not math.isfinite(float(dropout)) or not 0.0 <= float(dropout) < 1.0:
        raise WrapperCompatibilityError("LoRA dropout must be at least 0 and less than 1")
    if bias != "none":
        raise WrapperCompatibilityError("LoRA bias must be exactly 'none'")


def _import_peft() -> Any:
    try:
        return importlib.import_module("peft")
    except (ImportError, OSError) as exc:
        raise DependencyError(
            "PEFT is unavailable; use the isolated .venv-qwen-wrapper environment "
            "and install requirements/wrapper-lora.txt"
        ) from exc


def _validate_approved_path_sequence(paths: Sequence[str]) -> None:
    if not paths:
        raise WrapperCompatibilityError("The exact LoRA module allowlist is empty")
    if len(set(paths)) != len(paths):
        raise WrapperCompatibilityError("The exact LoRA module allowlist contains duplicates")
    by_layer: dict[int, set[str]] = {}
    for path in paths:
        match = _APPROVED_MODULE_RE.fullmatch(path)
        if match is None:
            raise WrapperCompatibilityError(
                f"Unsafe LoRA module path outside exact text Q/V allowlist: {path}"
            )
        by_layer.setdefault(int(match.group(1)), set()).add(match.group(2))
    indices = sorted(by_layer)
    if indices != list(range(len(indices))):
        raise WrapperCompatibilityError(
            "The LoRA allowlist must cover contiguous text decoder layers from zero"
        )
    for index in indices:
        if by_layer[index] != {"q_proj", "v_proj"}:
            raise WrapperCompatibilityError(
                f"The LoRA allowlist must contain exactly q_proj/v_proj for layer {index}"
            )


def _parse_approved_lora_parameter(
    parameter_name: str,
    approved_paths: Sequence[str],
) -> tuple[str, str] | None:
    for module_path in approved_paths:
        for prefix in ("", "base_model.model."):
            stem = prefix + module_path
            if not parameter_name.startswith(stem):
                continue
            suffix = parameter_name[len(stem) :]
            match = re.fullmatch(r"\.lora_([AB])\.([^.]+)\.weight", suffix)
            if match is not None:
                return module_path, match.group(1)
    return None


def _contains_module_path(parameter_name: str, module_path: str) -> bool:
    start = parameter_name.find(module_path)
    while start >= 0:
        before_ok = start == 0 or parameter_name[start - 1] == "."
        end = start + len(module_path)
        after_ok = end == len(parameter_name) or parameter_name[end] == "."
        if before_ok and after_ok:
            return True
        start = parameter_name.find(module_path, start + 1)
    return False


def _named_modules(model: Any) -> list[tuple[str, Any]]:
    method = getattr(model, "named_modules", None)
    if not callable(method):
        raise WrapperCompatibilityError("Wrapper model does not expose named_modules()")
    try:
        values = list(method())
    except Exception as exc:
        raise WrapperCompatibilityError(f"Could not inventory wrapper modules: {exc}") from exc
    if not all(isinstance(item, tuple) and len(item) == 2 for item in values):
        raise WrapperCompatibilityError("Wrapper named_modules() returned invalid entries")
    names = [name for name, _module in values]
    if any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
        raise WrapperCompatibilityError(
            "Wrapper named_modules() returned invalid or duplicate names"
        )
    return values


def _named_parameters(model: Any) -> list[tuple[str, Any]]:
    method = getattr(model, "named_parameters", None)
    if not callable(method):
        raise WrapperCompatibilityError("Wrapper model does not expose named_parameters()")
    try:
        values = list(method())
    except Exception as exc:
        raise WrapperCompatibilityError(f"Could not inventory wrapper parameters: {exc}") from exc
    if not all(isinstance(item, tuple) and len(item) == 2 for item in values):
        raise WrapperCompatibilityError("Wrapper named_parameters() returned invalid entries")
    names = [name for name, _parameter in values]
    if any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
        raise WrapperCompatibilityError(
            "Wrapper named_parameters() returned invalid or duplicate names"
        )
    return values


def _named_parameters_with_aliases(model: Any) -> list[tuple[str, Any]]:
    """Return named parameters while retaining tied-weight module aliases."""

    method = getattr(model, "named_parameters", None)
    if not callable(method):
        raise WrapperCompatibilityError("Wrapper model does not expose named_parameters()")
    try:
        values = list(method(remove_duplicate=False))
    except TypeError:
        # Lightweight test doubles and older compatible module-like objects may
        # not expose PyTorch's remove_duplicate keyword.
        return _named_parameters(model)
    except Exception as exc:
        raise WrapperCompatibilityError(
            f"Could not inventory wrapper parameter aliases: {exc}"
        ) from exc
    if not all(isinstance(item, tuple) and len(item) == 2 for item in values):
        raise WrapperCompatibilityError(
            "Wrapper named_parameters(remove_duplicate=False) returned invalid entries"
        )
    names = [name for name, _parameter in values]
    if any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
        raise WrapperCompatibilityError(
            "Wrapper named_parameters(remove_duplicate=False) returned invalid or "
            "duplicate names"
        )
    return values


def _parameter_numel(parameter: Any, name: str) -> int:
    numel = getattr(parameter, "numel", None)
    if not callable(numel):
        raise WrapperCompatibilityError(f"Parameter {name} has no numel()")
    try:
        count = numel()
    except Exception as exc:
        raise WrapperCompatibilityError(f"Could not count parameter {name}: {exc}") from exc
    if type(count) is not int or count < 0:
        raise WrapperCompatibilityError(f"Parameter {name} returned invalid numel {count!r}")
    return count


def _parameter_shape(parameter: Any) -> tuple[int, ...]:
    shape = getattr(parameter, "shape", ())
    try:
        return tuple(int(value) for value in shape)
    except (TypeError, ValueError):
        return ()


def _parameter_checksum(parameter: Any, name: str) -> str:
    custom = getattr(parameter, "checksum_bytes", None)
    if callable(custom):
        payload = custom()
        if not isinstance(payload, bytes):
            raise WrapperCompatibilityError(
                f"Parameter {name} checksum_bytes() did not return bytes"
            )
    else:
        try:
            tensor = parameter.detach().reshape(-1)
            element_count = _parameter_numel(parameter, name)
            # Hash bounded slices, not an entire embedding or projection.  A
            # full BF16-to-FP32 CPU copy can consume hundreds of MiB and would
            # defeat the laptop memory guard this evidence is meant to serve.
            if element_count <= 768:
                ranges = ((0, element_count),)
            else:
                midpoint = max(0, (element_count // 2) - 128)
                ranges = (
                    (0, 256),
                    (midpoint, min(element_count, midpoint + 256)),
                    (element_count - 256, element_count),
                )
            payload_parts: list[bytes] = []
            for start, end in ranges:
                sample = tensor[start:end].cpu().contiguous()
                try:
                    sample_bytes = sample.numpy().tobytes()
                except (TypeError, ValueError):
                    sample_bytes = sample.float().numpy().tobytes()
                payload_parts.append(f"{start}:{end}|".encode() + sample_bytes)
            payload = b"".join(payload_parts)
        except Exception as exc:
            raise WrapperCompatibilityError(
                f"Could not checksum frozen parameter {name}: {exc}"
            ) from exc
    metadata = f"{getattr(parameter, 'dtype', 'unknown')}|{_parameter_shape(parameter)}|".encode()
    return hashlib.sha256(metadata + payload).hexdigest()


def _class_name(module: Any) -> str:
    cls = module.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _approved_path_sort_key(path: str) -> tuple[int, int]:
    match = _APPROVED_MODULE_RE.fullmatch(path)
    if match is None:  # Guarded before sorting.
        return (2**31 - 1, 2**31 - 1)
    return (int(match.group(1)), 0 if match.group(2) == "q_proj" else 1)


__all__ = [
    "FrozenParameterChecksums",
    "LoRAInjectionResult",
    "LoRAModuleInventory",
    "ModuleInventoryEntry",
    "OptimizerParameterAudit",
    "ParameterInventoryEntry",
    "TrainableParameterAudit",
    "audit_trainable_parameters",
    "capture_trainable_parameter_copies",
    "capture_representative_frozen_checksums",
    "discover_lora_inventory",
    "enable_non_reentrant_gradient_checkpointing",
    "inject_lora",
    "optimizer_trainable_parameters",
    "validate_optimizer_parameter_ids",
    "verify_frozen_parameter_checksums",
    "verify_trainable_parameter_change",
]
