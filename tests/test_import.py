from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import orato_asr

ROOT = Path(__file__).resolve().parents[1]


def test_package_import_and_version() -> None:
    assert orato_asr.__version__ == "0.1.0"
    assert callable(orato_asr.load_config)


def test_package_import_does_not_load_heavy_libraries() -> None:
    forbidden = {
        "accelerate",
        "azure",
        "datasets",
        "librosa",
        "mlflow",
        "peft",
        "qwen_asr",
        "soundfile",
        "soxr",
        "torch",
        "transformers",
    }
    code = (
        "import sys; import orato_asr; "
        f"forbidden={forbidden!r}; "
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
