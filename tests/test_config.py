from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
import yaml

from orato_asr.config import ConfigError, ProjectConfig, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
PROFILE_PATHS = sorted(CONFIG_DIR.glob("*.yaml"))


def _local_profile() -> dict[str, Any]:
    loaded = yaml.safe_load((CONFIG_DIR / "local_tiny.yaml").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _write_config(tmp_path: Path, values: object, name: str = "test.yaml") -> Path:
    config_path = tmp_path / name
    config_path.write_text(
        yaml.safe_dump(values, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


@pytest.mark.parametrize("profile_path", PROFILE_PATHS, ids=lambda path: path.stem)
def test_every_committed_profile_loads_and_resolves_paths(profile_path: Path) -> None:
    config = load_config(profile_path, project_root=ROOT)

    assert isinstance(config, ProjectConfig)
    output_dir = config.values["paths"]["output_dir"]  # type: ignore[index]
    reports_dir = config.values["paths"]["reports_dir"]  # type: ignore[index]
    model_cache_dir = config.values["paths"]["model_cache_dir"]  # type: ignore[index]
    assert isinstance(output_dir, Path) and output_dir.is_absolute()
    assert isinstance(reports_dir, Path) and reports_dir.is_absolute()
    assert output_dir.is_relative_to(ROOT / "outputs")
    assert reports_dir.is_relative_to(ROOT / "reports")
    assert isinstance(model_cache_dir, Path) and model_cache_dir.is_absolute()
    assert model_cache_dir.is_relative_to(ROOT / "outputs")


def test_profiles_pin_native_model_and_inference_defaults() -> None:
    for profile_path in PROFILE_PATHS:
        values = load_config(profile_path, project_root=ROOT).as_dict()
        assert values["schema_version"] == 2
        assert values["model"] == {
            "integration_track": "transformers_native",
            "id": "Qwen/Qwen3-ASR-0.6B-hf",
            "revision": "6aa69c382e2b426eee1f5870d4c95859a74b6445",
            "processor_revision": "6aa69c382e2b426eee1f5870d4c95859a74b6445",
        }
        assert values["inference"]["max_new_tokens"] == 256
        assert values["inference"]["precision"] == "auto"


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("data", "max_samples", 0),
        ("data", "max_samples", -1),
        ("data", "max_audio_hours", -0.5),
        ("hardware", "gpu_count", 0),
        ("hardware", "process_count", -1),
        ("training", "per_device_batch_size", 0),
        ("training", "gradient_accumulation_steps", -1),
        ("training", "max_steps", 0),
    ],
)
def test_non_positive_numeric_values_fail(
    tmp_path: Path,
    section: str,
    field: str,
    value: int | float,
) -> None:
    profile = _local_profile()
    profile[section][field] = value

    with pytest.raises(ConfigError, match=field):
        load_config(_write_config(tmp_path, profile), project_root=ROOT)


def test_boolean_is_not_accepted_as_an_integer(tmp_path: Path) -> None:
    profile = _local_profile()
    profile["training"]["max_steps"] = True

    with pytest.raises(ConfigError, match="training.max_steps"):
        load_config(_write_config(tmp_path, profile), project_root=ROOT)


def test_empty_model_id_fails(tmp_path: Path) -> None:
    profile = _local_profile()
    profile["model"]["id"] = "   "

    with pytest.raises(ConfigError, match="model.id"):
        load_config(_write_config(tmp_path, profile), project_root=ROOT)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("integration_track", "qwen_asr_wrapper"),
        ("id", "Qwen/Qwen3-ASR-0.6B"),
        ("revision", "wrong"),
        ("processor_revision", "wrong"),
    ],
)
def test_native_track_rejects_mixed_model_metadata(
    tmp_path: Path, field: str, value: str
) -> None:
    profile = _local_profile()
    profile["model"][field] = value

    with pytest.raises(ConfigError, match=f"model.{field}"):
        load_config(_write_config(tmp_path, profile), project_root=ROOT)


@pytest.mark.parametrize("field", ["revision", "processor_revision"])
def test_native_track_requires_revisions(tmp_path: Path, field: str) -> None:
    profile = _local_profile()
    del profile["model"][field]

    with pytest.raises(ConfigError, match="missing required keys"):
        load_config(_write_config(tmp_path, profile), project_root=ROOT)


@pytest.mark.parametrize("value", [0, -1, True, 4097])
def test_invalid_generation_limit_fails(tmp_path: Path, value: object) -> None:
    profile = _local_profile()
    profile["inference"]["max_new_tokens"] = value

    with pytest.raises(ConfigError, match="max_new_tokens"):
        load_config(_write_config(tmp_path, profile), project_root=ROOT)


@pytest.mark.parametrize("precision", ["float16", "bfloat16"])
def test_cpu_half_precision_fails(tmp_path: Path, precision: str) -> None:
    profile = _local_profile()
    profile["inference"].update({"device": "cpu", "precision": precision})

    with pytest.raises(ConfigError, match="CPU inference"):
        load_config(_write_config(tmp_path, profile), project_root=ROOT)


@pytest.mark.parametrize(
    ("section", "field", "unsafe_value"),
    [
        ("transcript", "roman_hinglish_primary", True),
        ("transcript", "short_utterances_allowed", False),
        ("data", "raw_data_mutation_enabled", True),
        ("data", "synthetic_data_enabled", True),
        ("training", "full_parameter_fit_guaranteed", True),
    ],
)
def test_transcript_and_data_safety_flags_cannot_be_relaxed(
    tmp_path: Path,
    section: str,
    field: str,
    unsafe_value: bool,
) -> None:
    profile = _local_profile()
    profile[section][field] = unsafe_value

    with pytest.raises(ConfigError, match=field):
        load_config(_write_config(tmp_path, profile), project_root=ROOT)


@pytest.mark.parametrize("profile_path", PROFILE_PATHS, ids=lambda path: path.stem)
def test_profiles_preserve_short_utterances_and_raw_data(profile_path: Path) -> None:
    config = load_config(profile_path, project_root=ROOT).as_dict()

    assert config["transcript"]["short_utterances_allowed"] is True
    assert config["transcript"]["roman_hinglish_primary"] is False
    assert config["data"]["raw_data_mutation_enabled"] is False
    assert config["data"]["synthetic_data_enabled"] is False


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("checkpointing", "save_steps", 0),
        ("checkpointing", "keep_last", -1),
        ("evaluation", "validation_steps", 0),
    ],
)
def test_invalid_checkpoint_or_validation_cadence_fails(
    tmp_path: Path,
    section: str,
    field: str,
    value: int,
) -> None:
    profile = yaml.safe_load(
        (CONFIG_DIR / "h100_smoke.yaml").read_text(encoding="utf-8")
    )
    profile[section][field] = value

    with pytest.raises(ConfigError, match=field):
        load_config(_write_config(tmp_path, profile), project_root=ROOT)


@pytest.mark.parametrize("location", ["top", "section"])
def test_unknown_keys_fail(tmp_path: Path, location: str) -> None:
    profile = _local_profile()
    if location == "top":
        profile["surprise"] = True
    else:
        profile["model"]["surprise"] = "unqualified"

    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(_write_config(tmp_path, profile), project_root=ROOT)


def test_missing_config_file_fails_clearly(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    with pytest.raises(ConfigError, match=f"does not exist: {missing}"):
        load_config(missing, project_root=ROOT)


@pytest.mark.parametrize("contents", ["profile: [", "- not\n- a\n- mapping\n"])
def test_malformed_or_non_mapping_yaml_fails(tmp_path: Path, contents: str) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(config_path, project_root=ROOT)


def test_eight_gpus_require_distributed_mode(tmp_path: Path) -> None:
    profile = _local_profile()
    profile["hardware"].update(
        {"gpu_count": 8, "process_count": 1, "distributed": False}
    )

    with pytest.raises(ConfigError, match="Eight-GPU|process_count"):
        load_config(_write_config(tmp_path, profile), project_root=ROOT)


def test_single_gpu_rejects_eight_processes(tmp_path: Path) -> None:
    profile = _local_profile()
    profile["hardware"].update(
        {
            "gpu_count": 1,
            "process_count": 8,
            "node_count": 8,
            "distributed": True,
            "launcher": "torchrun",
        }
    )

    with pytest.raises(ConfigError, match="single-GPU"):
        load_config(_write_config(tmp_path, profile), project_root=ROOT)


def test_distributed_outputs_require_rank_zero_writes(tmp_path: Path) -> None:
    profile = yaml.safe_load(
        (CONFIG_DIR / "h100_8gpu.yaml").read_text(encoding="utf-8")
    )
    profile["hardware"]["rank_zero_writes"] = False

    with pytest.raises(ConfigError, match="rank_zero_writes"):
        load_config(_write_config(tmp_path, profile), project_root=ROOT)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../outside",
        "/tmp/outside",
        "azureml://datastores/example/paths/output",
        "https://example.invalid/output",
    ],
)
def test_unsafe_or_uri_output_paths_fail(tmp_path: Path, unsafe_path: str) -> None:
    profile = _local_profile()
    profile["paths"]["output_dir"] = unsafe_path

    with pytest.raises(ConfigError, match="paths.output_dir"):
        load_config(_write_config(tmp_path, profile), project_root=ROOT)


def test_loaded_configuration_is_recursively_immutable() -> None:
    config = load_config(CONFIG_DIR / "local_tiny.yaml", project_root=ROOT)

    assert isinstance(config.values, MappingProxyType)
    with pytest.raises(TypeError):
        config.values["schema_version"] = 2  # type: ignore[index]
    with pytest.raises(TypeError):
        config.values["model"]["id"] = "changed"  # type: ignore[index]

    detached = config.as_dict()
    detached["model"]["id"] = "changed"
    assert config.values["model"]["id"] == "Qwen/Qwen3-ASR-0.6B-hf"  # type: ignore[index]


def test_loading_does_not_modify_yaml_or_create_directories(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    config_dir = project_root / "configs"
    config_dir.mkdir()

    profile = _local_profile()
    profile["paths"]["output_dir"] = "outputs/not-created"
    profile["paths"]["reports_dir"] = "reports/not-created"
    profile["paths"]["model_cache_dir"] = "outputs/model-cache-not-created"
    config_path = _write_config(config_dir, profile)
    original = config_path.read_bytes()

    config = load_config(config_path, project_root=project_root)

    assert config_path.read_bytes() == original
    assert not (project_root / "outputs").exists()
    assert not (project_root / "reports").exists()
    assert not (project_root / "outputs" / "model-cache-not-created").exists()
    assert config.as_dict()["paths"]["output_dir"] == str(
        project_root / "outputs" / "not-created"
    )
