"""Run foundation checks and optional native inference qualification."""

from __future__ import annotations

import argparse
import importlib
import json
import tempfile
import sys
from pathlib import Path
from typing import Any, Sequence

from orato_asr.cli import run_doctor
from orato_asr.config import ConfigError, load_config
from orato_asr.exceptions import OratoASRError
from orato_asr.models.qwen3_asr import (
    Qwen3ASREngine,
    dependency_status,
    sanitize_error,
    select_runtime,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local_tiny.yaml")
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args(argv)

    inference_mode = args.inference or args.load_model
    doctor_status = run_doctor(include_ml=inference_mode)
    try:
        config = load_config(Path(args.config))
    except ConfigError as exc:
        print(f"Preflight configuration error: {exc}", file=sys.stderr)
        return 2

    values = config.as_dict()
    profile, hardware, data = values["profile"], values["hardware"], values["data"]
    scale_limit = (
        f"max_samples={data['max_samples']}"
        if data["max_samples"] is not None
        else f"max_audio_hours={data['max_audio_hours']}"
    )
    print(f"[OK] Configuration: {config.source_path}")
    print(f"[INFO] Profile name: {profile['name']}")
    print(f"[INFO] Hardware mode: {hardware['accelerator']} ({hardware['device_preference']})")
    print(f"[INFO] Scale intent: {profile['intent']} ({scale_limit})")
    print(f"[INFO] GPU count: {hardware['gpu_count']}")
    print(f"[INFO] Distributed: {str(hardware['distributed']).lower()}")

    if not inference_mode:
        print("[INFO] Use --inference for native Qwen dependency/device qualification.")
        return doctor_status

    report: dict[str, Any] = {
        "status": "pending",
        "config": str(config.source_path),
        "profile": profile["name"],
        "model": values["model"],
        "device_requested": args.device or values["inference"]["device"],
        "model_load_requested": args.load_model,
        "checks": {},
    }
    failures = 0
    if hardware["distributed"] or hardware["process_count"] != 1:
        report["checks"]["single_process"] = {
            "ok": False,
            "detail": "This inference milestone supports only one process",
        }
        failures += 1
    else:
        report["checks"]["single_process"] = {"ok": True, "detail": "one process"}

    dependencies = dependency_status()
    report["checks"]["dependencies"] = dependencies
    if not all(bool(item["matches"]) for item in dependencies.values()):
        failures += 1

    outputs_root = config.project_root / "outputs"
    output_writable = False
    output_detail = f"directory does not exist: {outputs_root}"
    if outputs_root.is_dir():
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=".orato-inference-preflight-",
                dir=outputs_root,
            ) as temporary_file:
                temporary_file.write("ok\n")
                temporary_file.flush()
        except OSError as exc:
            output_detail = f"temporary-file check failed: {sanitize_error(exc)}"
        else:
            output_writable = True
            output_detail = f"temporary-file check passed in {outputs_root}"
    report["checks"]["output_writable"] = {
        "ok": output_writable,
        "detail": output_detail,
    }
    if not output_writable:
        failures += 1

    cache_dir = Path(values["paths"]["model_cache_dir"])
    offline = bool(values["inference"]["offline"])
    cache_ok = cache_dir.is_dir() if offline else cache_dir.parent.is_dir()
    report["checks"]["cache"] = {
        "ok": cache_ok,
        "path": str(cache_dir),
        "offline": offline,
        "state": "present" if cache_dir.is_dir() else "not_created",
    }
    if not cache_ok:
        failures += 1

    selection = None
    try:
        torch = importlib.import_module("torch")
        selection = select_runtime(
            args.device or values["inference"]["device"],
            values["inference"]["precision"],
            torch,
        )
    except (ImportError, OSError, OratoASRError) as exc:
        report["checks"]["device"] = {"ok": False, "detail": sanitize_error(exc)}
        failures += 1
    else:
        report["checks"]["device"] = {
            "ok": True,
            "resolved_device": selection.device,
            "resolved_precision": selection.precision,
        }

    for module_name in ("soundfile", "soxr"):
        try:
            importlib.import_module(module_name)
        except (ImportError, OSError) as exc:
            report["checks"][f"decoder_{module_name}"] = {
                "ok": False,
                "detail": sanitize_error(exc),
            }
            failures += 1
        else:
            report["checks"][f"decoder_{module_name}"] = {"ok": True}

    if args.load_model and failures == 0:
        engine = Qwen3ASREngine(
            device=args.device or values["inference"]["device"],
            precision=values["inference"]["precision"],
            cache_dir=cache_dir,
            offline=offline,
            language=values["inference"]["language_hint"],
            max_new_tokens=values["inference"]["max_new_tokens"],
        )
        try:
            report["checks"]["model_load"] = {"ok": True, "detail": engine.load()}
        except OratoASRError as exc:
            report["checks"]["model_load"] = {"ok": False, "detail": sanitize_error(exc)}
            failures += 1
        finally:
            engine.close()
    else:
        report["checks"]["model_load"] = {
            "ok": not args.load_model,
            "detail": "not requested" if not args.load_model else "skipped after failed prerequisite",
        }

    report["status"] = "pass" if failures == 0 and doctor_status == 0 else "fail"
    if args.report_dir is not None:
        destination = args.report_dir.expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        report_path = destination / "inference_preflight.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[INFO] Inference report: {report_path}")
    print(f"[{'OK' if report['status'] == 'pass' else 'FAIL'}] Inference preflight: {report['status']}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
