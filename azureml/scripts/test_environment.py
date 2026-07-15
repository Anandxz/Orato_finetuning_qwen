from __future__ import annotations

import importlib.metadata
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


EXPECTED_VERSIONS = {
    "qwen-asr": "0.0.6",
    "transformers": "4.57.6",
    "accelerate": "1.12.0",
    "peft": "0.19.1",
    "soundfile": "0.14.0",
    "librosa": "0.11.0",
    "soxr": "1.1.0",
    "huggingface-hub": "0.36.0",
}


def verify_versions() -> None:
    print("\nPackage versions:")

    for package, expected in EXPECTED_VERSIONS.items():
        installed = importlib.metadata.version(package)
        print(f"  {package}: {installed}")

        if installed != expected:
            raise RuntimeError(
                f"{package} version mismatch: "
                f"expected {expected}, found {installed}"
            )


def verify_audio() -> None:
    sample_rate = 16_000
    duration_seconds = 0.25

    samples = np.zeros(
        int(sample_rate * duration_seconds),
        dtype=np.float32,
    )

    with tempfile.TemporaryDirectory() as directory:
        audio_path = Path(directory) / "environment_test.wav"
        sf.write(audio_path, samples, sample_rate)

        decoded, decoded_rate = sf.read(
            audio_path,
            dtype="float32",
        )

        if decoded_rate != sample_rate:
            raise RuntimeError(
                f"Unexpected sample rate: {decoded_rate}"
            )

        if len(decoded) != len(samples):
            raise RuntimeError("Audio round-trip length mismatch")

    print("Audio write/read test: passed")


def main() -> None:
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print(
            "No GPU detected. This is expected on the CPU "
            "qualification compute."
        )

    verify_versions()

    # Verify wrapper imports.
    import accelerate  # noqa: F401
    import librosa  # noqa: F401
    import peft  # noqa: F401
    import qwen_asr  # noqa: F401
    import transformers  # noqa: F401

    print("Qwen wrapper imports: passed")

    # The repository source is supplied separately by the Azure job.
    import orato_asr

    print("Project import:", Path(orato_asr.__file__).resolve())

    ffmpeg_path = shutil.which("ffmpeg")
    sox_path = shutil.which("sox")

    if not ffmpeg_path:
        raise RuntimeError("ffmpeg is unavailable")

    if not sox_path:
        raise RuntimeError("sox is unavailable")

    print("ffmpeg:", ffmpeg_path)
    print("sox:", sox_path)

    verify_audio()

    print("\nENVIRONMENT_RUNTIME_TEST_PASSED")


if __name__ == "__main__":
    main()
