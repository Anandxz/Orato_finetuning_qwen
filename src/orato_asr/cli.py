"""Command-line configuration, qualification, and one-shot inference utilities."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit

import yaml

from .config import ConfigError, ProjectConfig, load_config
from .environment import collect_environment
from .exceptions import (
    EvaluationError,
    ManifestError,
    OratoASRError,
    PathSafetyError,
    StorageError,
)
from .paths import find_project_root
from .version import __version__

DEFAULT_CONFIG = "configs/local_tiny.yaml"
DEFAULT_WRAPPER_CONFIG = "configs/train_wrapper_lora_laptop_smoke.yaml"
DEFAULT_WRAPPER_COMPATIBILITY_REPORT = "wrapper_0.6b_compatibility.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orato-asr",
        description="Orato native Qwen3-ASR qualification and inference utilities",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    config_parser = commands.add_parser("config", help="Inspect configuration profiles")
    config_commands = config_parser.add_subparsers(dest="config_command", required=True)
    show_parser = config_commands.add_parser("show", help="Show a resolved profile")
    show_parser.add_argument("--config", required=True)
    validate_parser = config_commands.add_parser("validate", help="Validate a profile")
    validate_parser.add_argument("--config", required=True)

    doctor = commands.add_parser("doctor", help="Run foundation or ML environment checks")
    doctor.add_argument("--ml", action="store_true", help="Check exact inference pins and CUDA")
    doctor.add_argument("--json", type=Path, help="Write a sanitized JSON environment report")

    model = commands.add_parser("model", help="Inspect the pinned model integration")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    model_info = model_commands.add_parser("info", help="Report model metadata without loading by default")
    model_info.add_argument("--config", default=DEFAULT_CONFIG)
    model_info.add_argument("--device", choices=("auto", "cpu", "cuda"))
    model_info.add_argument("--load", action="store_true", help="Explicitly load processor and model")
    model_info.add_argument("--json", type=Path, help="Write the result as JSON")

    transcribe = commands.add_parser("transcribe", help="Transcribe one local WAV or FLAC file")
    transcribe.add_argument("--audio", required=True, type=Path)
    transcribe.add_argument("--config", default=DEFAULT_CONFIG)
    transcribe.add_argument("--device", choices=("auto", "cpu", "cuda"))
    transcribe.add_argument(
        "--precision", choices=("auto", "float32", "float16", "bfloat16")
    )
    transcribe.add_argument("--language")
    transcribe.add_argument("--cache-dir", type=str)
    transcribe.add_argument("--offline", action="store_true")
    transcribe.add_argument("--max-new-tokens", type=int)
    transcribe.add_argument("--output-json", type=Path)

    data = commands.add_parser("data", help="Validate, summarize, select, and compare JSONL manifests")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    data_validate = data_commands.add_parser("validate", help="Validate a canonical JSONL manifest")
    data_validate.add_argument("--manifest", required=True, type=Path)
    data_validate.add_argument("--check-audio", action="store_true")
    data_validate.add_argument("--hash-local-audio", action="store_true")
    data_validate.add_argument("--duration-tolerance-seconds", type=float, default=0.25)
    data_validate.add_argument("--report", required=True, type=Path)
    data_summarize = data_commands.add_parser("summarize", help="Write a streaming manifest summary")
    data_summarize.add_argument("--manifest", required=True, type=Path)
    data_summarize.add_argument("--check-audio", action="store_true")
    data_summarize.add_argument("--hash-local-audio", action="store_true")
    data_summarize.add_argument("--output", required=True, type=Path)
    data_select = data_commands.add_parser("select", help="Write a deterministic derived manifest")
    data_select.add_argument("--manifest", required=True, type=Path)
    data_select.add_argument("--output", required=True, type=Path)
    data_select.add_argument("--max-samples", type=int)
    duration_limit = data_select.add_mutually_exclusive_group()
    duration_limit.add_argument("--max-seconds", type=float)
    duration_limit.add_argument("--max-hours", type=float)
    data_select.add_argument("--min-duration-seconds", type=float)
    data_select.add_argument("--max-duration-seconds", type=float)
    data_select.add_argument("--source")
    data_select.add_argument("--domain")
    data_select.add_argument("--language")
    data_select.add_argument("--seed", type=int, default=0)
    data_select.add_argument("--shuffled", action="store_true")
    data_select.add_argument("--overwrite", action="store_true")
    data_split = data_commands.add_parser(
        "split",
        help="Canonicalize and group-safely stratify an owner manifest",
    )
    data_split.add_argument("--manifest", required=True, type=Path)
    data_split.add_argument("--train-output", required=True, type=Path)
    data_split.add_argument("--val-output", required=True, type=Path)
    data_split.add_argument("--test-output", required=True, type=Path)
    data_split.add_argument("--summary-output", required=True, type=Path)
    data_split.add_argument("--train-ratio", type=float, default=0.8)
    data_split.add_argument("--val-ratio", type=float, default=0.1)
    data_split.add_argument("--test-ratio", type=float, default=0.1)
    data_split.add_argument("--seed", type=int, default=42)
    data_split.add_argument("--category-field", default="eval_category")
    data_split.add_argument("--overwrite", action="store_true")
    data_build_splits = data_commands.add_parser(
        "build-splits",
        help="Build a versioned leakage-safe split across processed datasets",
    )
    data_build_splits.add_argument("--config", required=True, type=Path)
    data_build_splits.add_argument("--data-root")
    data_build_splits.add_argument("--split-root", type=Path)
    data_build_splits.add_argument("--seed", type=int)
    data_build_splits.add_argument("--train-ratio", type=float)
    data_build_splits.add_argument("--validation-ratio", type=float)
    data_build_splits.add_argument("--test-ratio", type=float)
    data_build_splits.add_argument("--overwrite", action="store_true")
    data_validate_splits = data_commands.add_parser(
        "validate-splits", help="Validate a versioned split bundle"
    )
    data_validate_splits.add_argument("--split-dir", required=True, type=Path)
    data_validate_splits.add_argument("--data-root")
    data_validate_splits.add_argument("--check-audio", action="store_true")
    data_overlap = data_commands.add_parser("check-overlap", help="Check train/evaluation leakage")
    data_overlap.add_argument("--train-manifest", required=True, type=Path)
    data_overlap.add_argument("--evaluation-manifest", required=True, type=Path)
    data_overlap.add_argument("--hash-local-audio", action="store_true")
    data_overlap.add_argument("--disallow-speaker-overlap", action="store_true")
    data_overlap.add_argument("--output", type=Path)

    evaluate = commands.add_parser("evaluate", help="Run dependency-free baseline evaluation orchestration")
    evaluate_commands = evaluate.add_subparsers(dest="evaluate_command", required=True)
    baseline = evaluate_commands.add_parser("baseline", help="Evaluate local WAV/FLAC records with the pinned model")
    baseline.add_argument("--manifest", required=True, type=Path)
    baseline.add_argument("--config", default=DEFAULT_CONFIG)
    baseline.add_argument("--run-name", required=True)
    baseline.add_argument("--device", choices=("auto", "cpu", "cuda"))
    baseline.add_argument("--max-samples", type=int)
    duration_limit = baseline.add_mutually_exclusive_group()
    duration_limit.add_argument("--max-seconds", type=float)
    duration_limit.add_argument("--max-hours", type=float)
    run_mode = baseline.add_mutually_exclusive_group()
    run_mode.add_argument("--resume", action="store_true")
    run_mode.add_argument("--overwrite", action="store_true")
    baseline.add_argument("--error-policy", choices=("continue", "stop"))
    baseline.add_argument("--offline", action="store_true")

    wrapper = commands.add_parser(
        "wrapper", help="Inspect the isolated qwen-asr wrapper backend"
    )
    wrapper_commands = wrapper.add_subparsers(dest="wrapper_command", required=True)
    wrapper_inspect = wrapper_commands.add_parser(
        "inspect", help="Load the wrapper and inventory exact LoRA candidates"
    )
    wrapper_inspect.add_argument("--config", default=DEFAULT_WRAPPER_CONFIG)
    wrapper_inspect.add_argument("--offline", action="store_true")
    wrapper_inspect.add_argument("--json", type=Path)

    train = commands.add_parser(
        "train", help="Run isolated wrapper compatibility and LoRA smoke checks"
    )
    train_commands = train.add_subparsers(dest="train_command", required=True)
    wrapper_preflight = train_commands.add_parser(
        "wrapper-preflight", help="Validate wrapper targets and finite base loss"
    )
    wrapper_preflight.add_argument("--config", default=DEFAULT_WRAPPER_CONFIG)
    wrapper_preflight.add_argument("--train-manifest", required=True, type=Path)
    wrapper_preflight.add_argument("--eval-manifest", type=Path)
    wrapper_preflight.add_argument("--device", choices=("cuda",), default="cuda")
    wrapper_preflight.add_argument("--offline", action="store_true")
    wrapper_preflight.add_argument("--json", type=Path)

    one_step = train_commands.add_parser(
        "lora-one-step", help="Run exactly one verified adapter optimizer step"
    )
    one_step.add_argument("--config", default=DEFAULT_WRAPPER_CONFIG)
    one_step.add_argument("--train-manifest", required=True, type=Path)
    one_step.add_argument("--run-name", required=True)
    one_step.add_argument("--device", choices=("cuda",), default="cuda")
    one_step.add_argument("--offline", action="store_true")
    one_step.add_argument("--output-json", type=Path)

    smoke = train_commands.add_parser(
        "lora-smoke", help="Run a bounded five- or ten-step LoRA smoke"
    )
    smoke.add_argument("--config", default=DEFAULT_WRAPPER_CONFIG)
    smoke.add_argument("--train-manifest", required=True, type=Path)
    smoke.add_argument("--eval-manifest", type=Path)
    smoke.add_argument("--run-name", required=True)
    smoke.add_argument("--max-optimizer-steps", type=int, choices=(5, 10), default=5)
    smoke.add_argument("--device", choices=("cuda",), default="cuda")
    smoke.add_argument("--offline", action="store_true")
    smoke.add_argument("--output-json", type=Path)
    smoke.add_argument(
        "--allow-without-one-step-evidence",
        action="store_true",
        help="Development-only override; the default requires successful one-step evidence",
    )

    full = train_commands.add_parser(
        "lora-full",
        help="Run one complete selected-data LoRA epoch after verified smoke evidence",
    )
    full.add_argument("--config", default="configs/train_wrapper_lora_full_epoch.yaml")
    full.add_argument("--train-manifest", required=True, type=Path)
    full.add_argument("--eval-manifest", required=True, type=Path)
    full.add_argument("--run-name", required=True)
    full.add_argument("--device", choices=("cuda",), default="cuda")
    full.add_argument("--offline", action="store_true")
    full.add_argument("--output-json", type=Path)

    verify = train_commands.add_parser(
        "verify-adapter", help="Fresh-process adapter reload and bounded evaluation"
    )
    verify.add_argument("--config", default=DEFAULT_WRAPPER_CONFIG)
    verify.add_argument("--run-dir", required=True, type=Path)
    verify.add_argument("--eval-manifest", required=True, type=Path)
    verify.add_argument("--max-samples", type=int, default=3)
    verify.add_argument("--device", choices=("cuda",), default="cuda")
    verify.add_argument("--offline", action="store_true")
    verify.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        return run_doctor(include_ml=args.ml, json_path=args.json)
    if args.command == "config":
        return _run_config(args)
    if args.command == "model" and args.model_command == "info":
        return _run_model_info(args)
    if args.command == "transcribe":
        return _run_transcribe(args)
    if args.command == "data":
        return _run_data(args)
    if args.command == "evaluate" and args.evaluate_command == "baseline":
        return _run_baseline(args)
    if args.command == "wrapper" and args.wrapper_command == "inspect":
        return _run_wrapper_inspect(args)
    if args.command == "train":
        return _run_train(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


def _run_config(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    if args.config_command == "show":
        print(yaml.safe_dump(config.as_dict(), allow_unicode=True, sort_keys=False), end="")
    else:
        print(f"Configuration is valid: {config.source_path}")
    return 0


def _run_model_info(args: argparse.Namespace) -> int:
    output_path = args.json
    try:
        config = load_config(args.config)
        values = config.as_dict()
        inference = values["inference"]
        if args.load:
            _require_single_process(config)
        from .models.qwen3_asr import Qwen3ASREngine, dependency_status

        device = args.device or inference["device"]
        engine = Qwen3ASREngine(
            device=device,
            precision=inference["precision"],
            cache_dir=values["paths"]["model_cache_dir"],
            offline=inference["offline"],
            language=inference["language_hint"],
            max_new_tokens=inference["max_new_tokens"],
        )
        try:
            info = engine.load() if args.load else engine.model_info()
            payload = {"status": "success", "model": info, "dependencies": dependency_status()}
        finally:
            engine.close()
    except (ConfigError, OratoASRError, OSError) as exc:
        return _report_failure("Model qualification failed", exc, output_path)
    try:
        _emit_json(payload, output_path)
    except OSError as exc:
        print(f"Could not write model JSON: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_transcribe(args: argparse.Namespace) -> int:
    output_path = args.output_json
    try:
        config = load_config(args.config)
        _require_single_process(config)
        values = config.as_dict()
        inference = values["inference"]
        cache_dir = (
            _explicit_local_path(args.cache_dir, "cache directory")
            if args.cache_dir is not None
            else Path(values["paths"]["model_cache_dir"])
        )
        from .audio import decode_audio
        from .models.qwen3_asr import Qwen3ASREngine

        audio = decode_audio(args.audio)
        engine = Qwen3ASREngine(
            device=args.device or inference["device"],
            precision=args.precision or inference["precision"],
            cache_dir=cache_dir,
            offline=args.offline or inference["offline"],
            language=args.language if args.language is not None else inference["language_hint"],
            max_new_tokens=(
                args.max_new_tokens
                if args.max_new_tokens is not None
                else inference["max_new_tokens"]
            ),
        )
        try:
            payload = engine.transcribe(audio).as_dict()
        finally:
            engine.close()
    except (ConfigError, OratoASRError, OSError) as exc:
        return _report_failure("Transcription failed", exc, output_path)
    try:
        _emit_json(payload, output_path)
    except OSError as exc:
        print(f"Could not write transcription JSON: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_data(args: argparse.Namespace) -> int:
    try:
        project_root = find_project_root(Path.cwd())
        from .data.manifest import write_json_atomic

        if args.data_command == "validate":
            from .data.validation import validate_manifest

            report = validate_manifest(
                args.manifest,
                project_root=project_root,
                check_audio=args.check_audio,
                hash_local_audio=args.hash_local_audio,
                duration_tolerance_seconds=args.duration_tolerance_seconds,
            )
            payload = report.as_dict()
            write_json_atomic(payload, args.report)
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            return 2 if report.has_errors else 0
        if args.data_command == "summarize":
            from .data.summary import summarize_manifest

            summary = summarize_manifest(
                args.manifest,
                project_root=project_root,
                check_audio=args.check_audio,
                hash_local_audio=args.hash_local_audio,
            )
            payload = summary.as_dict()
            write_json_atomic(payload, args.output)
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            return 2 if payload["structural_errors"] or payload["media_errors"] else 0
        if args.data_command == "select":
            from .data.selection import SelectionOptions, select_manifest

            maximum = args.max_seconds
            if args.max_hours is not None:
                maximum = args.max_hours * 3600
            report = select_manifest(
                args.manifest,
                args.output,
                options=SelectionOptions(
                    max_samples=args.max_samples,
                    max_duration_seconds=maximum,
                    min_duration_seconds=args.min_duration_seconds,
                    maximum_duration_seconds=args.max_duration_seconds,
                    source=args.source,
                    domain=args.domain,
                    language=args.language,
                    seed=args.seed,
                    shuffled=args.shuffled,
                ),
                overwrite=args.overwrite,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.data_command == "split":
            from .data.splitting import SplitOptions, split_owner_manifest

            report = split_owner_manifest(
                args.manifest,
                train_output=args.train_output,
                val_output=args.val_output,
                test_output=args.test_output,
                summary_output=args.summary_output,
                options=SplitOptions(
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio,
                    seed=args.seed,
                    category_field=args.category_field,
                ),
                overwrite=args.overwrite,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.data_command == "build-splits":
            from .data.build_splits import (
                build_splits,
                format_split_summary,
                load_split_config,
            )

            config = load_split_config(
                args.config,
                project_root=project_root,
                data_root=args.data_root,
                split_root=args.split_root,
                seed=args.seed,
                train_ratio=args.train_ratio,
                validation_ratio=args.validation_ratio,
                test_ratio=args.test_ratio,
            )
            report = build_splits(config, overwrite=args.overwrite)
            print(format_split_summary(report))
            return 0
        if args.data_command == "validate-splits":
            from .data.build_splits import validate_split_directory

            report = validate_split_directory(
                args.split_dir,
                project_root=project_root,
                data_root=args.data_root,
                check_audio=args.check_audio,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if report["status"] == "success" else 2
        if args.data_command == "check-overlap":
            from .data.overlap import check_overlap

            report = check_overlap(
                args.train_manifest,
                args.evaluation_manifest,
                project_root=project_root,
                hash_local_audio=args.hash_local_audio,
                disallow_speaker_overlap=args.disallow_speaker_overlap,
            )
            payload = report.as_dict()
            if args.output is not None:
                write_json_atomic(payload, args.output)
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            return 1 if report.prohibited_count else 0
    except (ManifestError, PathSafetyError, StorageError, OSError, ValueError) as exc:
        print(f"Data command failed: {_sanitize_cli_error(exc)}", file=sys.stderr)
        return 2
    raise AssertionError(f"Unsupported data command: {args.data_command}")


def _run_baseline(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        from .evaluation.baseline import BaselineOptions, run_baseline

        max_duration = args.max_seconds
        if args.max_hours is not None:
            max_duration = args.max_hours * 3600
        result = run_baseline(
            args.manifest,
            config,
            options=BaselineOptions(
                run_name=args.run_name,
                device=args.device,
                max_samples=args.max_samples,
                max_duration_seconds=max_duration,
                resume=True if args.resume else None,
                overwrite=True if args.overwrite else None,
                error_policy=args.error_policy,
                offline=True if args.offline else None,
            ),
        )
    except (ConfigError, EvaluationError, ManifestError, PathSafetyError, OSError, ValueError) as exc:
        print(f"Baseline evaluation failed: {_sanitize_cli_error(exc)}", file=sys.stderr)
        return 2
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return result.exit_code


def _run_wrapper_inspect(args: argparse.Namespace) -> int:
    try:
        from .training import load_wrapper_training_config
        from .training.runner import inspect_wrapper

        config = load_wrapper_training_config(args.config)
        payload = inspect_wrapper(config, offline=args.offline)
        _emit_json(payload, args.json)
    except ConfigError as exc:
        return _report_failure(
            "Wrapper configuration error", exc, args.json, exit_code=2
        )
    except (OratoASRError, OSError) as exc:
        return _report_failure("Wrapper inspection failed", exc, args.json)
    return 0


def _run_train(args: argparse.Namespace) -> int:
    output_path = getattr(args, "json", None) or getattr(args, "output_json", None)
    try:
        from .training import load_wrapper_training_config
        from .training.runner import (
            run_lora_training,
            run_wrapper_preflight,
            verify_adapter,
        )

        config = load_wrapper_training_config(args.config)
        if args.train_command == "wrapper-preflight":
            if output_path is None:
                output_path = (
                    Path(config.as_dict()["paths"]["reports_root"])
                    / DEFAULT_WRAPPER_COMPATIBILITY_REPORT
                )
            payload = run_wrapper_preflight(
                config,
                train_manifest=args.train_manifest,
                eval_manifest=args.eval_manifest,
                offline=args.offline,
            )
        elif args.train_command == "lora-one-step":
            payload = run_lora_training(
                config,
                train_manifest=args.train_manifest,
                run_name=args.run_name,
                optimizer_steps=1,
                one_step_mode=True,
                offline=args.offline,
            )
        elif args.train_command == "lora-smoke":
            if args.eval_manifest is not None:
                from .data.overlap import check_overlap

                overlap = check_overlap(
                    args.train_manifest,
                    args.eval_manifest,
                    project_root=config.project_root,
                    hash_local_audio=True,
                )
                if overlap.prohibited_count:
                    raise EvaluationError(
                        f"Train/evaluation overlap has {overlap.prohibited_count} prohibited finding(s)"
                    )
            payload = run_lora_training(
                config,
                train_manifest=args.train_manifest,
                run_name=args.run_name,
                optimizer_steps=args.max_optimizer_steps,
                one_step_mode=False,
                offline=args.offline,
                allow_without_one_step_evidence=args.allow_without_one_step_evidence,
            )
        elif args.train_command == "lora-full":
            from .data.overlap import check_overlap

            overlap = check_overlap(
                args.train_manifest,
                args.eval_manifest,
                project_root=config.project_root,
                hash_local_audio=True,
            )
            if overlap.prohibited_count:
                raise EvaluationError(
                    f"Train/evaluation overlap has {overlap.prohibited_count} prohibited finding(s)"
                )
            payload = run_lora_training(
                config,
                train_manifest=args.train_manifest,
                run_name=args.run_name,
                optimizer_steps=None,
                one_step_mode=False,
                offline=args.offline,
                full_epoch_mode=True,
            )
        elif args.train_command == "verify-adapter":
            payload = verify_adapter(
                config,
                run_directory=args.run_dir,
                eval_manifest=args.eval_manifest,
                max_samples=args.max_samples,
                offline=args.offline,
            )
        else:  # pragma: no cover - argparse owns the choices.
            raise AssertionError(f"Unsupported train command: {args.train_command}")
        _emit_json(payload, output_path)
    except ConfigError as exc:
        return _report_failure(
            "Training configuration error", exc, output_path, exit_code=2
        )
    except (ManifestError, PathSafetyError, ValueError) as exc:
        return _report_failure(
            "Training input error", exc, output_path, exit_code=2
        )
    except (OratoASRError, OSError) as exc:
        return _report_failure("Training command failed", exc, output_path)
    return 0


def run_doctor(*, include_ml: bool = False, json_path: Path | None = None) -> int:
    """Run foundation checks and optionally inspect the pinned ML environment."""

    failures = 0
    version = platform.python_version()
    supported_python = (3, 11) <= sys.version_info[:2] < (3, 14)
    failures += _print_check(supported_python, "Python version", f"{version} (supported: >=3.11,<3.14; recommended: 3.12)")

    try:
        import orato_asr
        import_detail, import_ok = f"orato_asr {orato_asr.__version__}", True
    except (ImportError, AttributeError) as exc:
        import_detail, import_ok = str(exc), False
    failures += _print_check(import_ok, "Package import", import_detail)

    try:
        project_root = find_project_root(Path.cwd())
    except PathSafetyError as exc:
        project_root = None
        failures += _print_check(False, "Project root", str(exc))
    else:
        failures += _print_check(True, "Project root", str(project_root))

    if project_root is not None:
        config_directory = project_root / "configs"
        config_available = config_directory.is_dir() and os.access(config_directory, os.R_OK)
        failures += _print_check(config_available, "Configuration directory", f"{config_directory} (readable: {config_available})")
        for label, relative_path in (("Reports", "reports"), ("Outputs", "outputs")):
            directory = project_root / relative_path
            failures += _print_check(directory.is_dir(), f"{label} directory", str(directory))
            writable, detail = _temporary_write_check(directory)
            failures += _print_check(writable, f"{label} write access", detail)

    system, release = platform.system(), platform.release()
    is_linux = system == "Linux"
    is_wsl = is_linux and ("microsoft" in release.lower() or "WSL_INTEROP" in os.environ)
    failures += _print_check(is_linux, "Operating system", f"{system} {release}")
    failures += _print_check(is_linux, "Linux/WSL detection", "WSL" if is_wsl else "Linux" if is_linux else "not Linux/WSL")
    print(f"[INFO] Current working directory: {Path.cwd()}")

    report = collect_environment(project_root or Path.cwd(), include_ml=include_ml)
    if include_ml:
        from .models.qwen3_asr import dependency_status
        statuses = dependency_status()
        for package, status in statuses.items():
            ok = bool(status["matches"])
            failures += _print_check(ok, f"Dependency {package}", f"installed={status['installed']}; required={status['required']}")
        cuda = report["cuda"]
        print(f"[INFO] CUDA available: {str(bool(cuda.get('available'))).lower()}")
        print("[INFO] Azure and training checks remain unimplemented.")
    else:
        print("[INFO] Use 'doctor --ml' for Qwen dependency and CUDA checks; Azure and training checks remain unimplemented.")

    if json_path is not None:
        try:
            _write_json(json_path, report)
        except OSError as exc:
            print(f"Could not write doctor JSON: {exc}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


def _require_single_process(config: ProjectConfig) -> None:
    hardware = config.as_dict()["hardware"]
    if hardware["distributed"] or hardware["process_count"] != 1:
        raise OratoASRError(
            "Native inference in this milestone is single-process only; select a "
            "non-distributed profile"
        )


def _explicit_local_path(value: str, label: str) -> Path:
    parsed = urlsplit(value.strip())
    if not value.strip() or parsed.scheme or parsed.netloc:
        raise PathSafetyError(f"{label} must be an explicit local filesystem path, not a URI")
    return Path(value).expanduser().resolve()


def _emit_json(payload: dict[str, Any], output_path: Path | None) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if output_path is not None:
        _write_json(output_path, payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _report_failure(
    label: str,
    error: BaseException,
    output_path: Path | None,
    *,
    exit_code: int = 1,
) -> int:
    from .models.qwen3_asr import sanitize_error
    message = sanitize_error(error)
    print(f"{label}: {message}", file=sys.stderr)
    if output_path is not None:
        try:
            _write_json(output_path, {"status": "error", "error": message})
        except OSError as write_error:
            print(f"Could not write failure JSON: {write_error}", file=sys.stderr)
    return exit_code


def _sanitize_cli_error(error: BaseException) -> str:
    """Use the existing URI/token-redaction policy for data/evaluation errors."""

    from .models.qwen3_asr import sanitize_error

    return sanitize_error(error)


def _print_check(ok: bool, label: str, detail: str) -> int:
    print(f"[{'OK' if ok else 'FAIL'}] {label}: {detail}")
    return 0 if ok else 1


def _temporary_write_check(directory: Path) -> tuple[bool, str]:
    if not directory.is_dir():
        return False, f"directory does not exist: {directory}"
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".orato-write-check-", dir=directory) as temporary_file:
            temporary_file.write("ok\n")
            temporary_file.flush()
    except OSError as exc:
        return False, f"temporary-file check failed in {directory}: {exc}"
    return True, f"temporary-file check passed in {directory}"


if __name__ == "__main__":
    raise SystemExit(main())
