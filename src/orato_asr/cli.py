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
from .exceptions import OratoASRError, PathSafetyError
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
