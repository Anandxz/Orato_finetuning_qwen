"""Run foundation checks and optionally summarize a validated profile."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from orato_asr.cli import run_doctor
from orato_asr.config import ConfigError, load_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional YAML profile to validate and report")
    args = parser.parse_args(argv)

    doctor_status = run_doctor()
    if args.config is None:
        return doctor_status

    try:
        config = load_config(Path(args.config))
    except ConfigError as exc:
        print(f"Preflight configuration error: {exc}", file=sys.stderr)
        return 2

    values = config.as_dict()
    profile = values["profile"]
    hardware = values["hardware"]
    data = values["data"]
    scale_limit = (
        f"max_samples={data['max_samples']}"
        if data["max_samples"] is not None
        else f"max_audio_hours={data['max_audio_hours']}"
    )

    print(f"[OK] Configuration: {config.source_path}")
    print(f"[INFO] Profile name: {profile['name']}")
    print(
        "[INFO] Hardware mode: "
        f"{hardware['accelerator']} ({hardware['device_preference']})"
    )
    print(f"[INFO] Scale intent: {profile['intent']} ({scale_limit})")
    print(f"[INFO] GPU count: {hardware['gpu_count']}")
    print(f"[INFO] Distributed: {str(hardware['distributed']).lower()}")
    print("[INFO] Qwen model and training qualification have not yet run.")
    return doctor_status


if __name__ == "__main__":
    raise SystemExit(main())
