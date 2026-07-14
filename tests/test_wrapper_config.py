from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
import yaml

from orato_asr.exceptions import ConfigError
from orato_asr.training.config import (
    WRAPPER_MODEL_REVISION,
    WrapperTrainingConfig,
    load_wrapper_training_config,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "configs" / "train_wrapper_lora_laptop_smoke.yaml"


def _profile() -> dict[str, Any]:
    loaded = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _write(tmp_path: Path, values: object) -> Path:
    path = tmp_path / "wrapper.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def test_committed_wrapper_profile_loads_separately_and_resolves_paths() -> None:
    config = load_wrapper_training_config(PROFILE_PATH, project_root=ROOT)

    assert isinstance(config, WrapperTrainingConfig)
    assert config.values["schema_version"] == 1
    assert config.values["model"] == {
        "id": "Qwen/Qwen3-ASR-0.6B",
        "revision": WRAPPER_MODEL_REVISION,
        "backend": "qwen_asr_wrapper",
        "dtype": "bfloat16",
    }
    paths = config.values["paths"]
    assert paths["output_root"] == (ROOT / "outputs" / "training").resolve()
    assert paths["reports_root"] == (ROOT / "reports" / "training").resolve()
    assert paths["model_cache_dir"] is None


def test_wrapper_config_is_recursively_immutable_and_as_dict_is_detached() -> None:
    config = load_wrapper_training_config(PROFILE_PATH, project_root=ROOT)

    assert isinstance(config.values, MappingProxyType)
    assert isinstance(config.values["method"], MappingProxyType)
    with pytest.raises(TypeError):
        config.values["method"]["rank"] = 2  # type: ignore[index]

    copy = config.as_dict()
    copy["method"]["rank"] = 2
    assert config.values["method"]["rank"] == 4
    assert copy["paths"]["output_root"] == str((ROOT / "outputs/training").resolve())


@pytest.mark.parametrize("schema_version", [True, 0, 2, "1"])
def test_wrapper_schema_version_is_strict(
    tmp_path: Path, schema_version: object
) -> None:
    profile = _profile()
    profile["schema_version"] = schema_version

    with pytest.raises(ConfigError, match="schema_version"):
        load_wrapper_training_config(_write(tmp_path, profile), project_root=ROOT)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "Qwen/Qwen3-ASR-0.6B-hf"),
        ("revision", "main"),
        ("backend", "transformers_native"),
        ("dtype", "float16"),
    ],
)
def test_wrapper_model_contract_cannot_be_mixed(
    tmp_path: Path, field: str, value: str
) -> None:
    profile = _profile()
    profile["model"][field] = value

    with pytest.raises(ConfigError, match=f"model.{field}"):
        load_wrapper_training_config(_write(tmp_path, profile), project_root=ROOT)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("type", "qlora"),
        ("rank", 3),
        ("rank", True),
        ("alpha", 4),
        ("dropout", -0.1),
        ("dropout", 1.0),
        ("bias", "all"),
        ("freeze_audio_encoder", False),
        ("target_scope", "all_qv_modules"),
    ],
)
def test_only_constrained_lora_method_is_accepted(
    tmp_path: Path, field: str, value: object
) -> None:
    profile = _profile()
    profile["method"][field] = value

    with pytest.raises(ConfigError, match=f"method.{field}"):
        load_wrapper_training_config(_write(tmp_path, profile), project_root=ROOT)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_audio_seconds", -1),
        ("max_audio_seconds", 0),
        ("max_audio_seconds", 10.1),
        ("max_samples", 0),
        ("max_samples", True),
        ("max_hours", 0),
        ("num_workers", 1),
        ("pin_memory", True),
        ("persistent_workers", True),
    ],
)
def test_data_settings_remain_bounded_and_lazy(
    tmp_path: Path, field: str, value: object
) -> None:
    profile = _profile()
    profile["data"][field] = value

    with pytest.raises(ConfigError, match=f"data.{field}"):
        load_wrapper_training_config(_write(tmp_path, profile), project_root=ROOT)


def test_minimum_duration_must_be_below_maximum(tmp_path: Path) -> None:
    profile = _profile()
    profile["data"]["min_audio_seconds"] = 6

    with pytest.raises(ConfigError, match="min_audio_seconds"):
        load_wrapper_training_config(_write(tmp_path, profile), project_root=ROOT)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("per_device_batch_size", 2),
        ("gradient_accumulation_steps", 0),
        ("gradient_accumulation_steps", 9),
        ("max_optimizer_steps", 0),
        ("max_optimizer_steps", 21),
        ("learning_rate", 0),
        ("weight_decay", -0.1),
        ("warmup_steps", -1),
        ("warmup_steps", 1),
        ("max_grad_norm", 0),
        ("gradient_checkpointing", False),
        ("use_cache", True),
        ("seed", -1),
        ("log_every_optimizer_steps", 0),
        ("save_every_optimizer_steps", 0),
    ],
)
def test_training_values_are_type_checked_and_memory_safe(
    tmp_path: Path, field: str, value: object
) -> None:
    profile = _profile()
    profile["training"][field] = value

    with pytest.raises(ConfigError, match=f"training.{field}"):
        load_wrapper_training_config(_write(tmp_path, profile), project_root=ROOT)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gpu_safety_limit_gb", 0),
        ("gpu_safety_limit_gb", 5.31),
        ("minimum_available_system_ram_gb", 0),
        ("minimum_available_system_ram_gb", 4.1),
        ("abort_on_threshold", False),
        ("capture_system_ram", False),
    ],
)
def test_memory_guards_cannot_be_disabled(
    tmp_path: Path, field: str, value: object
) -> None:
    profile = _profile()
    profile["memory"][field] = value

    with pytest.raises(ConfigError, match=f"memory.{field}"):
        load_wrapper_training_config(_write(tmp_path, profile), project_root=ROOT)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("device", "cpu"),
        ("device", "auto"),
        ("distributed", True),
        ("cpu_fallback", True),
    ],
)
def test_runtime_is_cuda_only_single_process_without_cpu_fallback(
    tmp_path: Path, field: str, value: object
) -> None:
    profile = _profile()
    profile["runtime"][field] = value

    with pytest.raises(ConfigError, match=f"runtime.{field}"):
        load_wrapper_training_config(_write(tmp_path, profile), project_root=ROOT)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("output_root", "/tmp/run"),
        ("output_root", "outputs/../outside"),
        ("reports_root", "outputs/reports"),
        ("reports_root", "https://example.test/reports"),
        ("model_cache_dir", "reports/model-cache"),
    ],
)
def test_generated_paths_must_stay_in_their_repository_roots(
    tmp_path: Path, field: str, value: str
) -> None:
    profile = _profile()
    profile["paths"][field] = value

    with pytest.raises(ConfigError, match=f"paths.{field}"):
        load_wrapper_training_config(_write(tmp_path, profile), project_root=ROOT)


def test_optional_custom_model_cache_must_stay_beneath_outputs(tmp_path: Path) -> None:
    profile = _profile()
    profile["paths"]["model_cache_dir"] = "outputs/model-cache-wrapper"

    config = load_wrapper_training_config(_write(tmp_path, profile), project_root=ROOT)

    assert config.values["paths"]["model_cache_dir"] == (
        ROOT / "outputs/model-cache-wrapper"
    ).resolve()


@pytest.mark.parametrize("location", ["root", "section"])
def test_unknown_keys_are_rejected_at_every_level(tmp_path: Path, location: str) -> None:
    profile = _profile()
    if location == "root":
        profile["native_inference"] = True
    else:
        profile["method"]["surprise"] = True

    with pytest.raises(ConfigError, match="unknown keys"):
        load_wrapper_training_config(_write(tmp_path, profile), project_root=ROOT)


def test_missing_required_field_is_rejected(tmp_path: Path) -> None:
    profile = _profile()
    del profile["model"]["revision"]

    with pytest.raises(ConfigError, match="missing required keys: revision"):
        load_wrapper_training_config(_write(tmp_path, profile), project_root=ROOT)


@pytest.mark.parametrize("contents", ["- not\n- a\n- mapping\n", "model: ["])
def test_non_mapping_and_malformed_yaml_fail(tmp_path: Path, contents: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigError):
        load_wrapper_training_config(path, project_root=ROOT)


def test_loading_does_not_create_directories_or_rewrite_yaml() -> None:
    before = PROFILE_PATH.read_bytes()

    load_wrapper_training_config(PROFILE_PATH, project_root=ROOT)

    assert PROFILE_PATH.read_bytes() == before
