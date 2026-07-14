from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from orato_asr.exceptions import WrapperCompatibilityError
from orato_asr.training.lora import (
    audit_trainable_parameters,
    capture_representative_frozen_checksums,
    capture_trainable_parameter_copies,
    discover_lora_inventory,
    inject_lora,
    optimizer_trainable_parameters,
    validate_optimizer_parameter_ids,
    verify_frozen_parameter_checksums,
    verify_trainable_parameter_change,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeParameter:
    def __init__(
        self,
        element_count: int,
        *,
        requires_grad: bool = True,
        value: int = 1,
    ) -> None:
        self._element_count = element_count
        self.requires_grad = requires_grad
        self.shape = (element_count,)
        self.dtype = "fake.float32"
        self.value = value

    def numel(self) -> int:
        return self._element_count

    def checksum_bytes(self) -> bytes:
        return bytes([self.value]) * self._element_count


class FakeModule:
    pass


class FakeWrapperModel:
    _forward_patched = True

    def __init__(self, *, layer_count: int = 2) -> None:
        text_model = FakeModule()
        text_model.layers = [FakeModule() for _ in range(layer_count)]
        self._modules: dict[str, Any] = {
            "": self,
            "thinker": FakeModule(),
            "thinker.audio_tower": FakeModule(),
            "thinker.audio_tower.layers": FakeModule(),
            "thinker.audio_tower.layers.0": FakeModule(),
            "thinker.audio_tower.layers.0.self_attn": FakeModule(),
            "thinker.audio_tower.layers.0.self_attn.q_proj": FakeModule(),
            "thinker.audio_tower.layers.0.self_attn.v_proj": FakeModule(),
            "thinker.model": text_model,
            "thinker.model.embed_tokens": FakeModule(),
            "thinker.model.layers": FakeModule(),
            "thinker.lm_head": FakeModule(),
        }
        self._parameters: dict[str, FakeParameter] = {
            "thinker.audio_tower.layers.0.self_attn.q_proj.weight": FakeParameter(12),
            "thinker.audio_tower.layers.0.self_attn.v_proj.weight": FakeParameter(12),
            "thinker.model.embed_tokens.weight": FakeParameter(30),
            "thinker.lm_head.weight": FakeParameter(30),
        }
        for layer_index in range(layer_count):
            base = f"thinker.model.layers.{layer_index}"
            module_names = (
                f"{base}",
                f"{base}.self_attn",
                f"{base}.self_attn.q_proj",
                f"{base}.self_attn.k_proj",
                f"{base}.self_attn.v_proj",
                f"{base}.self_attn.o_proj",
                f"{base}.mlp",
                f"{base}.mlp.gate_proj",
                f"{base}.mlp.up_proj",
                f"{base}.mlp.down_proj",
            )
            self._modules.update({name: FakeModule() for name in module_names})
            for suffix in (
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ):
                self._parameters[f"{base}.{suffix}"] = FakeParameter(16)

    def named_modules(self):
        return list(self._modules.items())

    def named_parameters(self):
        return list(self._parameters.items())


class FakePeftModel:
    def __init__(
        self,
        base: FakeWrapperModel,
        target_modules: list[str],
        events: list[str],
        *,
        omit_last_b: bool = False,
        unexpected_trainable: bool = False,
    ) -> None:
        self.config = SimpleNamespace(use_cache=True)
        self.checkpointing_kwargs: dict[str, Any] | None = None
        self.events = events
        self.base_model = SimpleNamespace(
            config=SimpleNamespace(use_cache=True),
            model=SimpleNamespace(config=SimpleNamespace(use_cache=True)),
        )
        self._parameters: dict[str, FakeParameter] = {}
        for name, parameter in base.named_parameters():
            if any(name == f"{target}.weight" for target in target_modules):
                target_name = name.removesuffix(".weight")
                wrapped_name = f"base_model.model.{target_name}.base_layer.weight"
            else:
                wrapped_name = f"base_model.model.{name}"
            self._parameters[wrapped_name] = parameter
        for target_index, target in enumerate(target_modules):
            for matrix in ("A", "B"):
                if omit_last_b and target_index == len(target_modules) - 1 and matrix == "B":
                    continue
                name = f"base_model.model.{target}.lora_{matrix}.default.weight"
                self._parameters[name] = FakeParameter(8, requires_grad=True, value=3)
        if unexpected_trainable:
            name = "base_model.model.thinker.audio_tower.extra.weight"
            self._parameters[name] = FakeParameter(4, requires_grad=True, value=4)

    def named_parameters(self):
        return list(self._parameters.items())

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        self.events.append("checkpointing")
        self.checkpointing_kwargs = kwargs


class FakeLoraConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.values = kwargs


class FakePeft:
    class TaskType:
        CAUSAL_LM = "CAUSAL_LM"

    LoraConfig = FakeLoraConfig

    def __init__(
        self,
        *,
        omit_last_b: bool = False,
        unexpected_trainable: bool = False,
    ) -> None:
        self.events: list[str] = []
        self.configuration: FakeLoraConfig | None = None
        self.omit_last_b = omit_last_b
        self.unexpected_trainable = unexpected_trainable

    def get_peft_model(
        self, model: FakeWrapperModel, configuration: FakeLoraConfig
    ) -> FakePeftModel:
        self.events.append("inject")
        self.configuration = configuration
        return FakePeftModel(
            model,
            configuration.values["target_modules"],
            self.events,
            omit_last_b=self.omit_last_b,
            unexpected_trainable=self.unexpected_trainable,
        )


def test_inventory_uses_exact_full_text_paths_and_rejects_audio_candidates() -> None:
    inventory = discover_lora_inventory(FakeWrapperModel())

    assert inventory.text_layer_count == 2
    assert inventory.approved_module_paths == (
        "thinker.model.layers.0.self_attn.q_proj",
        "thinker.model.layers.0.self_attn.v_proj",
        "thinker.model.layers.1.self_attn.q_proj",
        "thinker.model.layers.1.self_attn.v_proj",
    )
    assert inventory.rejected_qv_candidate_paths == (
        "thinker.audio_tower.layers.0.self_attn.q_proj",
        "thinker.audio_tower.layers.0.self_attn.v_proj",
    )
    assert set(inventory.text_mlp_paths) == {
        f"thinker.model.layers.{index}.mlp.{projection}"
        for index in range(2)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    assert inventory.total_parameter_count > 0
    assert inventory.as_dict()["module_count"] == len(inventory.modules)


def test_inventory_rejects_missing_projection_and_noncontiguous_structure() -> None:
    missing_projection = FakeWrapperModel()
    del missing_projection._modules["thinker.model.layers.1.self_attn.v_proj"]
    with pytest.raises(WrapperCompatibilityError, match="exactly q_proj and v_proj"):
        discover_lora_inventory(missing_projection)

    missing_layer = FakeWrapperModel(layer_count=3)
    for name in list(missing_layer._modules):
        if name.startswith("thinker.model.layers.1"):
            del missing_layer._modules[name]
    with pytest.raises(WrapperCompatibilityError, match="not contiguous"):
        discover_lora_inventory(missing_layer)


def test_inventory_requires_exact_projection_weight_parameters() -> None:
    model = FakeWrapperModel()
    del model._parameters["thinker.model.layers.0.self_attn.q_proj.weight"]

    with pytest.raises(WrapperCompatibilityError, match="lack an exact weight"):
        discover_lora_inventory(model)


def test_inventory_rejects_unverified_non_audio_qv_candidate() -> None:
    model = FakeWrapperModel()
    model._modules["thinker.model.layers.0.cross_attn.q_proj"] = FakeModule()

    with pytest.raises(WrapperCompatibilityError, match="outside the verified"):
        discover_lora_inventory(model)


def test_inject_lora_freezes_base_uses_causal_lm_and_checkpoints_afterward() -> None:
    model = FakeWrapperModel()
    peft = FakePeft()

    result = inject_lora(model, peft_module=peft)

    assert peft.events == ["inject", "checkpointing"]
    assert peft.configuration is not None
    assert peft.configuration.values == {
        "r": 4,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": list(result.inventory.approved_module_paths),
        "task_type": "CAUSAL_LM",
        "inference_mode": False,
    }
    assert all(not parameter.requires_grad for parameter in model._parameters.values())
    assert result.model.checkpointing_kwargs == {
        "gradient_checkpointing_kwargs": {"use_reentrant": False}
    }
    assert result.model.config.use_cache is False
    assert result.model.base_model.config.use_cache is False
    assert result.model.base_model.model.config.use_cache is False
    assert result.audit.trainable_parameter_tensor_count == 8
    assert result.audit.audio_trainable_parameter_count == 0
    assert result.audit.text_decoder_base_trainable_parameter_count == 0
    assert result.audit.trainable_parameter_count == 64
    assert result.audit.trainable_percentage < 50
    assert set(result.audit.trainable_elements_by_module.values()) == {16}
    assert result.as_dict()["gradient_checkpointing_use_reentrant"] is False


def test_injection_requires_official_forward_patch_first() -> None:
    class UnpatchedWrapper(FakeWrapperModel):
        _forward_patched = False

    with pytest.raises(WrapperCompatibilityError, match="forward patch"):
        inject_lora(UnpatchedWrapper(), peft_module=FakePeft())


@pytest.mark.parametrize(
    ("peft", "message"),
    [
        (FakePeft(omit_last_b=True), "missing"),
        (FakePeft(unexpected_trainable=True), "Unexpected trainable"),
    ],
)
def test_injection_aborts_for_incomplete_or_unexpected_trainability(
    peft: FakePeft, message: str
) -> None:
    with pytest.raises(WrapperCompatibilityError, match=message):
        inject_lora(
            FakeWrapperModel(),
            peft_module=peft,
            gradient_checkpointing=False,
        )


def test_audit_rejects_broad_or_audio_module_allowlists() -> None:
    model = FakeWrapperModel()

    with pytest.raises(WrapperCompatibilityError, match="Unsafe LoRA module path"):
        audit_trainable_parameters(model, ["q_proj", "v_proj"])
    with pytest.raises(WrapperCompatibilityError, match="Unsafe LoRA module path"):
        audit_trainable_parameters(
            model,
            [
                "thinker.audio_tower.layers.0.self_attn.q_proj",
                "thinker.audio_tower.layers.0.self_attn.v_proj",
            ],
        )


def test_optimizer_ids_are_exactly_trainable_adapters_once() -> None:
    result = inject_lora(
        FakeWrapperModel(),
        peft_module=FakePeft(),
        gradient_checkpointing=False,
    )
    parameters = optimizer_trainable_parameters(result.model)
    optimizer = SimpleNamespace(param_groups=[{"params": list(parameters)}])

    audit = validate_optimizer_parameter_ids(optimizer, result.model)

    assert audit.optimizer_parameter_tensor_count == len(parameters)
    with pytest.raises(WrapperCompatibilityError, match="duplicate"):
        validate_optimizer_parameter_ids(
            SimpleNamespace(param_groups=[{"params": [*parameters, parameters[0]]}]),
            result.model,
        )
    frozen = next(
        parameter
        for parameter in result.model._parameters.values()
        if not parameter.requires_grad
    )
    with pytest.raises(WrapperCompatibilityError, match="unexpected_or_frozen=1"):
        validate_optimizer_parameter_ids(
            SimpleNamespace(param_groups=[{"params": [*parameters, frozen]}]),
            result.model,
        )


def test_one_optimizer_step_must_change_an_adapter_copy() -> None:
    class CopyParameter(FakeParameter):
        def detach(self) -> "CopyParameter":
            return self

        def cpu(self) -> "CopyParameter":
            return self

        def clone(self) -> "CopyParameter":
            return CopyParameter(
                self._element_count,
                requires_grad=self.requires_grad,
                value=self.value,
            )

    class AdapterOnlyModel:
        def __init__(self) -> None:
            self.adapter = CopyParameter(4, requires_grad=True, value=1)
            self.frozen = CopyParameter(4, requires_grad=False, value=7)

        def named_parameters(self):
            return [("adapter.weight", self.adapter), ("base.weight", self.frozen)]

    model = AdapterOnlyModel()
    before = capture_trainable_parameter_copies(model)
    with pytest.raises(WrapperCompatibilityError, match="did not change"):
        verify_trainable_parameter_change(
            model,
            before,
            tensor_equal=lambda left, right: left.value == right.value,
        )

    # Simulate the exact observable effect required from one optimizer step.
    model.adapter.value += 1
    verify_trainable_parameter_change(
        model,
        before,
        tensor_equal=lambda left, right: left.value == right.value,
    )
    assert model.frozen.value == 7


def test_representative_frozen_checksums_cover_scopes_and_detect_mutation() -> None:
    result = inject_lora(
        FakeWrapperModel(),
        peft_module=FakePeft(),
        gradient_checkpointing=False,
    )
    checksums = capture_representative_frozen_checksums(result.model)

    assert len(checksums.checksums) == 4
    verify_frozen_parameter_checksums(result.model, checksums)
    changed_name = next(iter(checksums.checksums))
    result.model._parameters[changed_name].value += 1
    with pytest.raises(WrapperCompatibilityError, match="Frozen base parameter changed"):
        verify_frozen_parameter_checksums(result.model, checksums)


def test_representative_checksums_retain_tied_output_head_alias() -> None:
    shared = FakeParameter(30, requires_grad=False)

    class TiedAliasModel:
        def __init__(self) -> None:
            self.parameters = [
                ("thinker.audio_tower.weight", FakeParameter(4, requires_grad=False)),
                (
                    "thinker.model.layers.0.weight",
                    FakeParameter(4, requires_grad=False),
                ),
                ("thinker.model.embed_tokens.weight", shared),
                ("thinker.lm_head.weight", shared),
            ]

        def named_parameters(self, *, remove_duplicate: bool = True):
            if remove_duplicate:
                return self.parameters[:-1]
            return self.parameters

    model = TiedAliasModel()
    checksums = capture_representative_frozen_checksums(model)

    assert "thinker.lm_head.weight" in checksums.checksums
    verify_frozen_parameter_checksums(model, checksums)
    shared.value += 1
    with pytest.raises(WrapperCompatibilityError, match="Frozen base parameter changed"):
        verify_frozen_parameter_checksums(model, checksums)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rank": 8}, "rank"),
        ({"alpha": 4}, "alpha"),
        ({"dropout": True}, "dropout"),
        ({"dropout": float("nan")}, "dropout"),
        ({"bias": "all"}, "bias"),
        ({"gradient_checkpointing": 1}, "gradient_checkpointing"),
    ],
)
def test_lora_hyperparameters_are_strict(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(WrapperCompatibilityError, match=message):
        inject_lora(FakeWrapperModel(), peft_module=FakePeft(), **kwargs)


def test_lora_module_import_is_free_of_heavy_dependencies() -> None:
    script = """
import json
import sys
import orato_asr.training.lora
names = ('torch', 'peft', 'transformers', 'qwen_asr')
print(json.dumps({name: name in sys.modules for name in names}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "torch": False,
        "peft": False,
        "transformers": False,
        "qwen_asr": False,
    }
