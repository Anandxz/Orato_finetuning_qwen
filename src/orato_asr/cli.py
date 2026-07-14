"""Command-line interface for foundation validation."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import yaml

from .config import ConfigError, load_config
from .paths import PathSafetyError, find_project_root
from .version import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="orato-asr",
        description="Orato Qwen3-ASR project foundation utilities",
    )
    parser.add_argument("--version", action="version", version=__version__)

    commands = parser.add_subparsers(dest="command", required=True)
    config_parser = commands.add_parser("config", help="Inspect configuration profiles")
    config_commands = config_parser.add_subparsers(
        dest="config_command", required=True
    )

    show_parser = config_commands.add_parser(
        "show", help="Validate and show a resolved configuration"
    )
    show_parser.add_argument("--config", required=True, help="Path to a YAML profile")

    validate_parser = config_commands.add_parser(
        "validate", help="Validate a configuration"
    )
    validate_parser.add_argument(
        "--config", required=True, help="Path to a YAML profile"
    )

    commands.add_parser("doctor", help="Run lightweight foundation checks")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        return run_doctor()

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.config_command == "show":
        print(
            yaml.safe_dump(
                config.as_dict(),
                allow_unicode=True,
                sort_keys=False,
            ),
            end="",
        )
        return 0

    if args.config_command == "validate":
        print(f"Configuration is valid: {config.source_path}")
        return 0

    parser.error(f"Unknown configuration command: {args.config_command}")
    return 2


def run_doctor() -> int:
    """Run read-only checks that do not require internet, GPU, or cloud access."""

    failures = 0
    version = platform.python_version()
    supported_python = (3, 11) <= sys.version_info[:2] < (3, 14)
    failures += _print_check(
        supported_python,
        "Python version",
        f"{version} (supported range: >=3.11,<3.14)",
    )

    try:
        import orato_asr

        import_detail = f"orato_asr {orato_asr.__version__}"
        import_ok = True
    except (ImportError, AttributeError) as exc:
        import_detail = str(exc)
        import_ok = False
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
        config_available = config_directory.is_dir() and os.access(
            config_directory, os.R_OK
        )
        failures += _print_check(
            config_available,
            "Configuration directory",
            f"{config_directory} (readable: {config_available})",
        )

        for label, relative_path in (
            ("Reports", "reports"),
            ("Outputs", "outputs"),
        ):
            directory = project_root / relative_path
            directory_available = directory.is_dir()
            failures += _print_check(
                directory_available,
                f"{label} directory",
                str(directory),
            )
            writable, write_detail = _temporary_write_check(directory)
            failures += _print_check(
                writable,
                f"{label} write access",
                write_detail,
            )

    system = platform.system()
    release = platform.release()
    is_linux = system == "Linux"
    is_wsl = is_linux and (
        "microsoft" in release.lower() or "WSL_INTEROP" in os.environ
    )
    failures += _print_check(
        is_linux,
        "Operating system",
        f"{system} {release}",
    )
    linux_environment = "WSL" if is_wsl else "Linux" if is_linux else "not Linux/WSL"
    failures += _print_check(
        is_linux,
        "Linux/WSL detection",
        linux_environment,
    )

    print(f"[INFO] Current working directory: {Path.cwd()}")
    print(
        "[INFO] Qwen model, CUDA, Azure, audio, and training checks are not "
        "implemented yet."
    )
    return 1 if failures else 0


def _print_check(ok: bool, label: str, detail: str) -> int:
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {label}: {detail}")
    return 0 if ok else 1


def _temporary_write_check(directory: Path) -> tuple[bool, str]:
    if not directory.is_dir():
        return False, f"directory does not exist: {directory}"

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=".orato-write-check-",
            dir=directory,
        ) as temporary_file:
            temporary_file.write("ok\n")
            temporary_file.flush()
    except OSError as exc:
        return False, f"temporary-file check failed in {directory}: {exc}"

    return True, f"temporary-file check passed in {directory}"


if __name__ == "__main__":
    raise SystemExit(main())
