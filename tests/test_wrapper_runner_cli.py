from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from orato_asr.exceptions import (
    AdapterVerificationError,
    ConfigError,
    DependencyError,
    ManifestError,
    TrainingError,
    WrapperCompatibilityError,
)
from orato_asr.training import runner, wrapper


ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PROFILE = ROOT / "configs" / "train_wrapper_lora_laptop_smoke.yaml"
HEAVY_MODULES = {
    "accelerate",
    "datasets",
    "librosa",
    "peft",
    "qwen_asr",
    "soundfile",
    "soxr",
    "torch",
    "transformers",
}


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "orato_asr.cli", *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (("--help",), ("wrapper", "train")),
        (("wrapper", "--help"), ("inspect",)),
        (("wrapper", "inspect", "--help"), ("--offline", "--json")),
        (
            ("train", "--help"),
            ("wrapper-preflight", "lora-one-step", "lora-smoke", "verify-adapter"),
        ),
        (
            ("train", "wrapper-preflight", "--help"),
            ("--train-manifest", "--eval-manifest", "--device", "--json"),
        ),
        (
            ("train", "lora-one-step", "--help"),
            ("--train-manifest", "--run-name", "--device", "--output-json"),
        ),
        (
            ("train", "lora-smoke", "--help"),
            (
                "--max-optimizer-steps",
                "--allow-without-one-step-evidence",
                "--output-json",
            ),
        ),
        (
            ("train", "verify-adapter", "--help"),
            ("--run-dir", "--eval-manifest", "--max-samples", "--output-json"),
        ),
    ],
)
def test_wrapper_and_training_help_surfaces(
    arguments: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    result = _run_cli(*arguments)

    assert result.returncode == 0, result.stderr
    assert all(value in result.stdout for value in expected)
    assert result.stderr == ""


def test_cli_wrapper_and_runner_imports_remain_free_of_heavy_modules() -> None:
    code = (
        "import sys; "
        "import orato_asr.cli; "
        "import orato_asr.training.wrapper; "
        "import orato_asr.training.runner; "
        f"forbidden={HEAVY_MODULES!r}; "
        "loaded=sorted(forbidden.intersection(sys.modules)); "
        "raise SystemExit('unexpected heavy imports: ' + ', '.join(loaded) if loaded else 0)"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_wrapper_dependency_status_reports_exact_pins_and_local_torch_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = dict(wrapper.WRAPPER_REQUIRED_VERSIONS)
    installed["torch"] = "2.11.0+cu128"

    monkeypatch.setattr(
        wrapper.importlib.metadata,
        "version",
        lambda distribution: installed[distribution],
    )

    status = wrapper.wrapper_dependency_status()

    assert set(status) == set(wrapper.WRAPPER_REQUIRED_VERSIONS)
    assert status["torch"] == {
        "available": True,
        "installed": "2.11.0+cu128",
        "required": "2.11.0",
        "matches": True,
    }
    assert status["qwen-asr"]["required"] == "0.0.6"
    assert status["transformers"]["required"] == "4.57.6"
    assert status["accelerate"]["required"] == "1.12.0"
    assert status["peft"]["required"] == "0.19.1"
    assert all(values["matches"] for values in status.values())


def test_wrapper_dependency_status_and_requirement_failure_are_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = dict(wrapper.WRAPPER_REQUIRED_VERSIONS)
    installed["peft"] = "0.18.0"

    def fake_version(distribution: str) -> str:
        if distribution == "qwen-asr":
            raise importlib.metadata.PackageNotFoundError(distribution)
        return installed[distribution]

    monkeypatch.setattr(wrapper.importlib.metadata, "version", fake_version)

    status = wrapper.wrapper_dependency_status()

    assert status["qwen-asr"] == {
        "available": False,
        "installed": "unavailable",
        "required": "0.0.6",
        "matches": False,
    }
    assert status["peft"]["matches"] is False
    with pytest.raises(DependencyError) as raised:
        wrapper.require_wrapper_dependencies()
    assert "qwen-asr=unavailable (required 0.0.6)" in str(raised.value)
    assert "peft=0.18.0 (required 0.19.1)" in str(raised.value)
    assert ".venv-qwen-wrapper" in str(raised.value)
    assert "requirements/wrapper-lora.txt" in str(raised.value)


def test_wrapper_training_disables_cache_on_nested_text_decoder() -> None:
    text_config = SimpleNamespace(use_cache=True)
    thinker = SimpleNamespace(
        config=SimpleNamespace(use_cache=True, text_config=text_config),
        model=SimpleNamespace(config=SimpleNamespace(use_cache=True)),
    )
    model = SimpleNamespace(config=SimpleNamespace(use_cache=True), thinker=thinker)

    wrapper._set_use_cache(model, False)

    assert model.config.use_cache is False
    assert thinker.config.use_cache is False
    assert thinker.config.text_config.use_cache is False
    assert thinker.model.config.use_cache is False


def test_snapshot_resolution_uses_the_exact_revision_and_offline_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / wrapper.WRAPPER_MODEL_REVISION
    snapshot.mkdir()
    calls: list[dict[str, Any]] = []

    def snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(snapshot)

    fake_hub = SimpleNamespace(snapshot_download=snapshot_download)
    monkeypatch.setattr(wrapper, "require_wrapper_dependencies", lambda: {})
    monkeypatch.setattr(wrapper.importlib, "import_module", lambda name: fake_hub)

    resolved = wrapper.resolve_wrapper_snapshot(cache_dir=tmp_path / "cache", offline=True)

    assert resolved == snapshot.resolve()
    assert calls == [
        {
            "repo_id": wrapper.WRAPPER_MODEL_ID,
            "revision": wrapper.WRAPPER_MODEL_REVISION,
            "local_files_only": True,
            "cache_dir": str((tmp_path / "cache").resolve()),
        }
    ]


def test_snapshot_resolution_rejects_a_revision_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drifted = tmp_path / "main"
    drifted.mkdir()
    fake_hub = SimpleNamespace(snapshot_download=lambda **_: str(drifted))
    monkeypatch.setattr(wrapper.importlib, "import_module", lambda name: fake_hub)

    with pytest.raises(WrapperCompatibilityError, match="does not match pinned revision"):
        wrapper.resolve_wrapper_snapshot(cache_dir=None, offline=False)


def _patch_cli_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Any, dict[str, Any]]:
    import orato_asr.training as training

    config = SimpleNamespace(
        project_root=tmp_path,
        as_dict=lambda: {
            "paths": {"reports_root": str(tmp_path / "reports" / "training")}
        },
    )
    calls: dict[str, Any] = {}
    monkeypatch.setattr(
        training,
        "load_wrapper_training_config",
        lambda path: calls.setdefault("config_path", path) and config,
    )
    return config, calls


def test_wrapper_inspect_cli_dispatches_and_writes_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import orato_asr.cli as cli

    config, calls = _patch_cli_training(monkeypatch, tmp_path)
    output = tmp_path / "inspect.json"

    def fake_inspect(received: Any, *, offline: bool) -> dict[str, Any]:
        calls["inspect"] = (received, offline)
        return {"status": "success", "backend": "qwen_asr_wrapper"}

    monkeypatch.setattr(runner, "inspect_wrapper", fake_inspect)

    exit_code = cli.main(
        [
            "wrapper",
            "inspect",
            "--config",
            str(WRAPPER_PROFILE),
            "--offline",
            "--json",
            str(output),
        ]
    )

    assert exit_code == 0
    assert calls["config_path"] == str(WRAPPER_PROFILE)
    assert calls["inspect"] == (config, True)
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "success"
    assert '"backend": "qwen_asr_wrapper"' in capsys.readouterr().out


def test_wrapper_preflight_cli_dispatches_all_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import orato_asr.cli as cli

    config, calls = _patch_cli_training(monkeypatch, tmp_path)
    train_manifest = tmp_path / "train.jsonl"
    eval_manifest = tmp_path / "eval.jsonl"
    output = tmp_path / "preflight.json"

    def fake_preflight(received: Any, **kwargs: Any) -> dict[str, Any]:
        calls["preflight"] = (received, kwargs)
        return {"status": "success", "decision": "wrapper_0.6b_compatible"}

    monkeypatch.setattr(runner, "run_wrapper_preflight", fake_preflight)

    exit_code = cli.main(
        [
            "train",
            "wrapper-preflight",
            "--config",
            str(WRAPPER_PROFILE),
            "--train-manifest",
            str(train_manifest),
            "--eval-manifest",
            str(eval_manifest),
            "--offline",
            "--json",
            str(output),
        ]
    )

    assert exit_code == 0
    assert calls["preflight"] == (
        config,
        {
            "train_manifest": train_manifest,
            "eval_manifest": eval_manifest,
            "offline": True,
        },
    )
    assert json.loads(output.read_text(encoding="utf-8"))["decision"] == (
        "wrapper_0.6b_compatible"
    )
    assert '"status": "success"' in capsys.readouterr().out


def test_wrapper_preflight_uses_safe_configured_default_report_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import orato_asr.cli as cli

    _patch_cli_training(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runner,
        "run_wrapper_preflight",
        lambda *_, **__: {
            "status": "success",
            "decision": "wrapper_0.6b_compatible",
        },
    )

    exit_code = cli.main(
        [
            "train",
            "wrapper-preflight",
            "--train-manifest",
            str(tmp_path / "train.jsonl"),
        ]
    )

    assert exit_code == 0
    report = (
        tmp_path
        / "reports"
        / "training"
        / cli.DEFAULT_WRAPPER_COMPATIBILITY_REPORT
    )
    assert json.loads(report.read_text(encoding="utf-8")) == {
        "status": "success",
        "decision": "wrapper_0.6b_compatible",
    }


def test_wrapper_preflight_default_report_captures_sanitized_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import orato_asr.cli as cli

    _patch_cli_training(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runner,
        "run_wrapper_preflight",
        lambda *_, **__: (_ for _ in ()).throw(
            TrainingError("wrapper compatibility failed")
        ),
    )

    exit_code = cli.main(
        [
            "train",
            "wrapper-preflight",
            "--train-manifest",
            str(tmp_path / "train.jsonl"),
        ]
    )

    assert exit_code == 1
    assert "Training command failed: wrapper compatibility failed" in (
        capsys.readouterr().err
    )
    report = (
        tmp_path
        / "reports"
        / "training"
        / cli.DEFAULT_WRAPPER_COMPATIBILITY_REPORT
    )
    assert json.loads(report.read_text(encoding="utf-8")) == {
        "status": "error",
        "error": "wrapper compatibility failed",
    }


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (
            (
                "lora-one-step",
                "--train-manifest",
                "train.jsonl",
                "--run-name",
                "one-step",
                "--offline",
            ),
            {
                "train_manifest": Path("train.jsonl"),
                "run_name": "one-step",
                "optimizer_steps": 1,
                "one_step_mode": True,
                "offline": True,
            },
        ),
        (
            (
                "lora-smoke",
                "--train-manifest",
                "train.jsonl",
                "--run-name",
                "smoke",
                "--max-optimizer-steps",
                "10",
                "--allow-without-one-step-evidence",
            ),
            {
                "train_manifest": Path("train.jsonl"),
                "run_name": "smoke",
                "optimizer_steps": 10,
                "one_step_mode": False,
                "offline": False,
                "allow_without_one_step_evidence": True,
            },
        ),
    ],
)
def test_lora_cli_dispatches_the_bounded_training_modes(
    arguments: tuple[str, ...],
    expected: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import orato_asr.cli as cli

    config, calls = _patch_cli_training(monkeypatch, tmp_path)

    def fake_training(received: Any, **kwargs: Any) -> dict[str, Any]:
        calls["training"] = (received, kwargs)
        return {"status": "success", "optimizer_steps": kwargs["optimizer_steps"]}

    monkeypatch.setattr(runner, "run_lora_training", fake_training)

    output = tmp_path / f"{arguments[0]}.json"

    exit_code = cli.main(["train", *arguments, "--output-json", str(output)])

    assert exit_code == 0
    assert calls["training"] == (config, expected)
    assert json.loads(capsys.readouterr().out)["status"] == "success"
    assert json.loads(output.read_text(encoding="utf-8"))["optimizer_steps"] == (
        expected["optimizer_steps"]
    )


@pytest.mark.parametrize(
    "arguments",
    [
        (
            "lora-one-step",
            "--train-manifest",
            "unused.jsonl",
            "--run-name",
            "one-step",
        ),
        (
            "lora-smoke",
            "--train-manifest",
            "unused.jsonl",
            "--run-name",
            "smoke",
            "--allow-without-one-step-evidence",
        ),
    ],
)
def test_lora_cli_requested_json_captures_sanitized_failures(
    arguments: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import orato_asr.cli as cli

    _patch_cli_training(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runner,
        "run_lora_training",
        lambda *_, **__: (_ for _ in ()).throw(TrainingError("bounded run failed")),
    )
    output = tmp_path / "failure.json"

    exit_code = cli.main(
        ["train", *arguments, "--output-json", str(output)]
    )

    assert exit_code == 1
    assert "Training command failed: bounded run failed" in capsys.readouterr().err
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "status": "error",
        "error": "bounded run failed",
    }


def test_verify_adapter_cli_dispatches_and_writes_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import orato_asr.cli as cli

    config, calls = _patch_cli_training(monkeypatch, tmp_path)
    run_dir = tmp_path / "outputs" / "training" / "smoke"
    eval_manifest = tmp_path / "eval.jsonl"
    output = tmp_path / "verification.json"

    def fake_verify(received: Any, **kwargs: Any) -> dict[str, Any]:
        calls["verify"] = (received, kwargs)
        return {"status": "success", "fresh_process_pid": 123}

    monkeypatch.setattr(runner, "verify_adapter", fake_verify)

    exit_code = cli.main(
        [
            "train",
            "verify-adapter",
            "--run-dir",
            str(run_dir),
            "--eval-manifest",
            str(eval_manifest),
            "--max-samples",
            "2",
            "--offline",
            "--output-json",
            str(output),
        ]
    )

    assert exit_code == 0
    assert calls["verify"] == (
        config,
        {
            "run_directory": run_dir,
            "eval_manifest": eval_manifest,
            "max_samples": 2,
            "offline": True,
        },
    )
    assert json.loads(output.read_text(encoding="utf-8"))["fresh_process_pid"] == 123


@pytest.mark.parametrize(
    ("failure", "expected_exit", "expected_label"),
    [
        (ConfigError("bad wrapper profile"), 2, "Training configuration error:"),
        (ManifestError("bad manifest row"), 2, "Training input error:"),
        (TrainingError("finite-forward failed"), 1, "Training command failed:"),
    ],
)
def test_training_cli_maps_expected_failures_to_stable_exit_codes(
    failure: BaseException,
    expected_exit: int,
    expected_label: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import orato_asr.cli as cli
    import orato_asr.training as training

    if isinstance(failure, ConfigError):
        monkeypatch.setattr(
            training,
            "load_wrapper_training_config",
            lambda _: (_ for _ in ()).throw(failure),
        )
    else:
        _patch_cli_training(monkeypatch, tmp_path)
        monkeypatch.setattr(
            runner,
            "run_wrapper_preflight",
            lambda *_, **__: (_ for _ in ()).throw(failure),
        )
    output = tmp_path / "failure.json"

    exit_code = cli.main(
        [
            "train",
            "wrapper-preflight",
            "--train-manifest",
            "unused.jsonl",
            "--json",
            str(output),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == expected_exit
    assert expected_label in captured.err
    assert "Traceback" not in captured.err
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == {"status": "error", "error": str(failure)}


def test_runner_inspection_closes_the_loaded_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    values = {
        "paths": {"model_cache_dir": str(tmp_path / "cache")},
        "memory": {"capture_system_ram": True},
    }
    config = SimpleNamespace(as_dict=lambda: values)
    inventory = SimpleNamespace(as_dict=lambda: {"approved_module_paths": ["q", "v"]})
    loaded = SimpleNamespace(
        model=object(),
        wrapper=SimpleNamespace(__class__=SimpleNamespace(__name__="FakeWrapper")),
        processor=SimpleNamespace(__class__=SimpleNamespace(__name__="FakeProcessor")),
        torch=object(),
        snapshot_path=tmp_path / wrapper.WRAPPER_MODEL_REVISION,
        close=lambda: events.append("close"),
    )
    monkeypatch.setattr(runner, "load_wrapper_model", lambda **_: loaded)
    monkeypatch.setattr(runner, "discover_lora_inventory", lambda _: inventory)
    monkeypatch.setattr(
        runner,
        "wrapper_dependency_status",
        lambda: {"qwen-asr": {"installed": "0.0.6", "matches": True}},
    )
    monkeypatch.setattr(
        runner,
        "_snapshot",
        lambda *_: SimpleNamespace(as_dict=lambda: {"stage": "wrapper_inspect"}),
    )

    result = runner.inspect_wrapper(config, offline=True)

    assert result["status"] == "success"
    assert result["inventory"] == {"approved_module_paths": ["q", "v"]}
    assert result["memory"] == {"stage": "wrapper_inspect"}
    assert events == ["close"]


def test_runner_preflight_releases_each_model_and_keeps_stages_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    values = {
        "paths": {"model_cache_dir": str(tmp_path / "cache")},
        "memory": {"capture_system_ram": True},
    }
    config = SimpleNamespace(
        project_root=tmp_path,
        as_dict=lambda: values,
    )
    prepared = SimpleNamespace(
        selected=[SimpleNamespace(sample_id="sample-1")],
        as_dict=lambda: {"eligible_samples": 1},
    )
    sample = SimpleNamespace(language="Hindi")

    class FakeLoaded:
        def __init__(self, name: str) -> None:
            self.name = name
            self.wrapper = object()
            self.model = object()
            self.processor = object()
            self.torch = object()
            self.snapshot_path = tmp_path / wrapper.WRAPPER_MODEL_REVISION
            self.load_seconds = 0.25

        def close(self) -> None:
            events.append(f"close:{self.name}")

    loaded_models = [FakeLoaded("inference"), FakeLoaded("forward")]

    def fake_load(**kwargs: Any) -> FakeLoaded:
        events.append(f"load:{kwargs['training']}")
        if kwargs["training"]:
            assert events[-2] == "close:inference"
        return loaded_models.pop(0)

    inventory = SimpleNamespace(
        text_layer_count=1,
        approved_module_paths=("thinker.model.layers.0.self_attn.q_proj",),
        rejected_qv_candidate_paths=("thinker.audio_tower.layers.0.q_proj",),
        audio_encoder_path="thinker.audio_tower",
        text_decoder_path="thinker.model",
        embeddings_path="thinker.model.embed_tokens",
        output_head_path="thinker.lm_head",
    )
    collated = SimpleNamespace(
        inspection={"prefix_fully_masked": True, "supervised_label_tokens": 3},
        inputs={"input_ids": object()},
    )
    monkeypatch.setattr(runner, "_prepare", lambda *_: prepared)
    monkeypatch.setattr(runner, "_overlap", lambda *_: {"prohibited_count": 0})
    monkeypatch.setattr(
        runner,
        "LazyWrapperTrainingDataset",
        lambda *_args, **_kwargs: [sample],
    )
    monkeypatch.setattr(runner, "_sample_audio_path", lambda *_: tmp_path / "audio.wav")
    monkeypatch.setattr(runner, "decode_audio", lambda _: object())
    monkeypatch.setattr(runner, "load_wrapper_model", fake_load)
    monkeypatch.setattr(
        runner,
        "wrapper_inference",
        lambda *_, **__: {"status": "success", "transcript": "हाँ"},
    )
    monkeypatch.setattr(
        runner,
        "load_wrapper_processor",
        lambda **_: (object(), tmp_path / wrapper.WRAPPER_MODEL_REVISION),
    )
    monkeypatch.setattr(runner, "collate_official_single", lambda *_: collated)
    monkeypatch.setattr(runner, "_enforce_guard", lambda *_: None)
    monkeypatch.setattr(runner, "move_batch_to_cuda", lambda inputs, _: inputs)
    monkeypatch.setattr(
        runner,
        "finite_forward",
        lambda *_, **__: {
            "loss": 1.25,
            "loss_tensor": object(),
            "peak_cuda_allocated_bytes": 100,
        },
    )
    monkeypatch.setattr(runner, "discover_lora_inventory", lambda _: inventory)
    monkeypatch.setattr(
        runner,
        "_snapshot",
        lambda *_: SimpleNamespace(as_dict=lambda: {"stage": "base_forward_complete"}),
    )
    monkeypatch.setattr(runner, "wrapper_dependency_status", lambda: {})

    result = runner.run_wrapper_preflight(
        config,
        train_manifest=tmp_path / "train.jsonl",
        eval_manifest=tmp_path / "eval.jsonl",
        offline=True,
    )

    assert events == ["load:False", "close:inference", "load:True", "close:forward"]
    assert result["decision"] == "wrapper_0.6b_compatible"
    assert result["optimizer_state_allocated"] is False
    assert result["stages"]["official_collator"]["prefix_fully_masked"] is True
    assert "loss_tensor" not in result["stages"]["base_finite_forward"]
    assert result["stages"]["module_inventory"]["text_layer_count"] == 1


@pytest.mark.parametrize(
    ("one_step_mode", "optimizer_steps", "message"),
    [
        (True, 2, "exactly one optimizer step"),
        (False, 1, "exactly 5 or 10"),
        (False, 20, "exactly 5 or 10"),
    ],
)
def test_runner_rejects_unbounded_modes_before_importing_torch(
    one_step_mode: bool,
    optimizer_steps: int,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: imported.append(name),
    )

    with pytest.raises(TrainingError, match=message):
        runner.run_lora_training(
            SimpleNamespace(),
            train_manifest="unused.jsonl",
            run_name="unused",
            optimizer_steps=optimizer_steps,
            one_step_mode=one_step_mode,
        )

    assert imported == []


def _compatibility_payload(fingerprint: str) -> dict[str, Any]:
    return {
        "status": "success",
        "decision": "wrapper_0.6b_compatible",
        "model": {
            "id": wrapper.WRAPPER_MODEL_ID,
            "revision": wrapper.WRAPPER_MODEL_REVISION,
            "backend": "qwen_asr_wrapper",
        },
        "dataset": {"manifest_sha256": fingerprint},
        "stages": {
            "wrapper_inference": {"transcript": "ठीक है"},
            "official_collator": {
                "prefix_fully_masked": True,
                "padding_fully_masked": True,
                "labels_match_target_token_ids": True,
                "decoded_supervised_target_matches": True,
                "supervised_label_tokens": 4,
            },
            "base_finite_forward": {"loss": 1.25},
        },
    }


def test_training_requires_exact_same_manifest_compatibility_evidence(
    tmp_path: Path,
) -> None:
    report_root = tmp_path / "reports" / "training"
    report_root.mkdir(parents=True)
    path = report_root / "wrapper_0.6b_compatibility.json"
    path.write_text(json.dumps(_compatibility_payload("abc")), encoding="utf-8")

    payload = runner._require_compatibility_evidence(
        report_root, manifest_fingerprint="abc"
    )

    assert payload["decision"] == "wrapper_0.6b_compatible"
    with pytest.raises(TrainingError, match="target/masking contract"):
        runner._require_compatibility_evidence(
            report_root, manifest_fingerprint="different"
        )
    broken = _compatibility_payload("abc")
    broken["stages"]["official_collator"]["prefix_fully_masked"] = False
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(TrainingError, match="target/masking contract"):
        runner._require_compatibility_evidence(report_root, manifest_fingerprint="abc")


def test_smoke_evidence_must_match_manifest_model_method_and_stage_e(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs" / "training"
    evidence_dir = output_root / "one"
    evidence_dir.mkdir(parents=True)
    method = {
        "rank": 4,
        "alpha": 16,
        "dropout": 0.05,
        "target_scope": "text_decoder_attention_qv_only",
    }
    evidence = {
        "status": "success",
        "optimizer_steps": 1,
        "adapter_saved": True,
        "manifest_sha256": "abc",
        "model_id": wrapper.WRAPPER_MODEL_ID,
        "model_revision": wrapper.WRAPPER_MODEL_REVISION,
        "method": method,
        "stage_e_backward_without_optimizer": True,
    }
    path = evidence_dir / "one_step_evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    runner._require_one_step_evidence(
        output_root, manifest_fingerprint="abc", method=method
    )

    evidence["stage_e_backward_without_optimizer"] = False
    path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(TrainingError, match="lora-one-step"):
        runner._require_one_step_evidence(
            output_root, manifest_fingerprint="abc", method=method
        )


def test_ten_step_requires_verified_safe_five_step_evidence(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs" / "training"
    run_dir = output_root / "five"
    run_dir.mkdir(parents=True)
    method = {
        "rank": 4,
        "alpha": 16,
        "dropout": 0.05,
        "target_scope": "text_decoder_attention_qv_only",
    }
    summary = {
        "status": "smoke_completed",
        "dataset_identity": {"manifest_sha256": "abc"},
        "method": method,
        "model": {
            "id": wrapper.WRAPPER_MODEL_ID,
            "revision": wrapper.WRAPPER_MODEL_REVISION,
            "backend": "qwen_asr_wrapper",
        },
        "consumption": {"optimizer_steps": 5},
        "training": {
            "initial_loss": 1.0,
            "final_loss": 0.9,
            "gradient_norms": [0.5] * 5,
            "peak_cuda_allocated_bytes": 100,
            "peak_cuda_reserved_bytes": 120,
        },
        "adapter": {"saved": True, "fresh_process_reload": True},
        "verification": {"status": "success"},
    }
    path = run_dir / "run_summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")

    runner._require_five_step_evidence(
        output_root,
        manifest_fingerprint="abc",
        method=method,
        gpu_safety_limit_bytes=200,
    )

    summary["training"]["peak_cuda_reserved_bytes"] = 200
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(TrainingError, match="verified five-step"):
        runner._require_five_step_evidence(
            output_root,
            manifest_fingerprint="abc",
            method=method,
            gpu_safety_limit_bytes=200,
        )


def test_memory_csv_and_peaks_retain_stage_measurements() -> None:
    events = [
        {
            "stage": "stage_e_after_backward",
            "captured_at_utc": "2026-07-14T00:00:00Z",
            "system_ram": {
                "total_bytes": 800,
                "available_bytes": 300,
                "used_bytes": 500,
            },
            "cuda": {
                "allocated_bytes": 100,
                "reserved_bytes": 120,
                "peak_allocated_bytes": 130,
                "peak_reserved_bytes": 150,
            },
            "cuda_process_check": {"status": "checked", "other_processes": []},
        }
    ]

    rows = runner._memory_csv_rows(events)

    assert rows[0]["stage"] == "stage_e_after_backward"
    assert rows[0]["system_used_bytes"] == 500
    assert runner._peak_memory_values(events) == (130, 150, 500)


def test_training_memory_guard_requires_a_successful_cuda_process_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "memory": {
            "minimum_available_system_ram_gb": 1.0,
            "gpu_safety_limit_gb": 5.3,
            "abort_on_threshold": True,
        }
    }

    assert runner._guard_config(values).require_cuda_process_check is True
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    with pytest.raises(OSError, match="non-zero"):
        runner._cuda_processes(0, 123)


def test_verify_adapter_updates_fresh_process_reports_with_fakes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "outputs" / "training"
    reports_root = tmp_path / "reports" / "training"
    run_dir = output_root / "smoke"
    adapter_dir = run_dir / "adapter"
    adapter_dir.mkdir(parents=True)
    approved = [
        "thinker.model.layers.0.self_attn.q_proj",
        "thinker.model.layers.0.self_attn.v_proj",
    ]
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "target_modules": approved,
                "r": 4,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            }
        ),
        encoding="utf-8",
    )
    (adapter_dir / "orato_adapter_metadata.json").write_text(
        json.dumps(
            {
                "base_model_id": wrapper.WRAPPER_MODEL_ID,
                "base_model_revision": wrapper.WRAPPER_MODEL_REVISION,
                "backend": "qwen_asr_wrapper",
                "qwen_sft_commit": runner.QWEN_SFT_COMMIT,
                "optimizer_steps": 5,
                "approved_module_paths": approved,
                "rank": 4,
                "alpha": 16,
                "dropout": 0.05,
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "status": "smoke_completed",
        "dataset": {
            "total_manifest_samples": 1,
            "total_manifest_duration_seconds": 1.0,
            "eligible_samples": 1,
            "eligible_duration_seconds": 1.0,
            "selected_samples": 1,
            "selected_duration_seconds": 1.0,
        },
        "consumption": {
            "samples": 1,
            "unique_samples": 1,
            "audio_duration_seconds": 1.0,
            "microsteps": 5,
            "optimizer_steps": 5,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "effective_batch_size": 1,
            "runtime_seconds": 2.0,
            "complete_epoch_performed": True,
            "eligible_epoch_fraction_by_unique_samples": 1.0,
            "estimated_complete_epoch_runtime_seconds": 2.0,
        },
        "claims": {"full_input_manifest_consumed": True},
        "adapter": {"saved": True, "fresh_process_reload": False},
        "report_facts": {
            "losses": {"initial": 1.0, "final": 0.9},
            "adapter": {"saved": True, "reloaded": False},
            "known_limitations": [
                "Fresh-process adapter verification remains a separate required command."
            ],
        },
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    (run_dir / "environment.json").write_text(
        json.dumps({"process_id": os.getpid() + 1}), encoding="utf-8"
    )
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"fake")
    manifest = tmp_path / "eval.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "audio_filepath": str(audio_path),
                "text": "ठीक है",
                "language": "Hindi",
                "split": "evaluation",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    config = SimpleNamespace(
        project_root=tmp_path,
        as_dict=lambda: {
            "paths": {
                "output_root": str(output_root),
                "reports_root": str(reports_root),
                "model_cache_dir": None,
            }
        },
    )
    monkeypatch.setattr(
        runner,
        "decode_audio",
        lambda _: SimpleNamespace(duration_seconds=1.0, samples=[0.0], sample_rate=16000),
    )

    class Loaded:
        def __init__(self, tag: str) -> None:
            self.tag = tag
            self.model = object()
            self.wrapper = SimpleNamespace(model=self.model)

        def close(self) -> None:
            pass

    loaded = iter((Loaded("base"), Loaded("adapter")))
    monkeypatch.setattr(runner, "load_wrapper_model", lambda **_: next(loaded))
    monkeypatch.setattr(
        runner,
        "wrapper_inference",
        lambda item, *_args, **_kwargs: {
            "transcript": "ठीक है" if item.tag == "base" else "ठीक है जी"
        },
    )

    class FakeAdapter:
        def to(self, _device: str) -> "FakeAdapter":
            return self

        def eval(self) -> None:
            pass

    fake_peft = SimpleNamespace(
        PeftModel=SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: FakeAdapter()
        )
    )
    real_import = runner.importlib.import_module
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: fake_peft if name == "peft" else real_import(name),
    )

    result = runner.verify_adapter(
        config,
        run_directory=run_dir,
        eval_manifest=manifest,
        max_samples=1,
        offline=True,
    )

    assert result["status"] == "success"
    assert result["aggregate_metrics"]["adapter"]["successful_samples"] == 1
    updated = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert updated["adapter"]["fresh_process_reload"] is True
    assert updated["report_facts"]["adapter"]["reloaded"] is True
    report_dir = reports_root / "smoke"
    assert (report_dir / "base_vs_adapter.json").is_file()
    assert "ठीक है जी" in (report_dir / "CTO_SMOKE_SUMMARY.md").read_text(
        encoding="utf-8"
    )
    assert "fresh-process reload: yes" in (
        report_dir / "README.md"
    ).read_text(encoding="utf-8")

    (run_dir / "environment.json").write_text(
        json.dumps({"process_id": os.getpid()}), encoding="utf-8"
    )
    with pytest.raises(AdapterVerificationError, match="fresh process"):
        runner.verify_adapter(
            config,
            run_directory=run_dir,
            eval_manifest=manifest,
            max_samples=1,
            offline=True,
        )
