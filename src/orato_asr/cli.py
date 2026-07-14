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
from .exceptions import EvaluationError, ManifestError, OratoASRError, PathSafetyError
from .paths import find_project_root
from .version import __version__

DEFAULT_CONFIG = "configs/local_tiny.yaml"


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
    except (ManifestError, PathSafetyError, OSError, ValueError) as exc:
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


def _report_failure(label: str, error: BaseException, output_path: Path | None) -> int:
    from .models.qwen3_asr import sanitize_error
    message = sanitize_error(error)
    print(f"{label}: {message}", file=sys.stderr)
    if output_path is not None:
        try:
            _write_json(output_path, {"status": "error", "error": message})
        except OSError as write_error:
            print(f"Could not write failure JSON: {write_error}", file=sys.stderr)
    return 1


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
