FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime@sha256:eee11b3b3872a8c838e35ef48f08b2d5def2080902c7f666831310ca1a0ef2be

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_XET=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    VIRTUAL_ENV=/opt/orato-venv

USER root

# System packages needed for audio processing and Python venv creation.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        ffmpeg \
        libsndfile1 \
        sox \
        libsox-fmt-all \
        python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Verify that the base image already contains the expected CUDA PyTorch.
RUN python - <<'PY'
import torch

print("Base PyTorch:", torch.__version__)
print("Base CUDA build:", torch.version.cuda)

if not torch.__version__.startswith("2.11.0"):
    raise RuntimeError(
        f"Expected PyTorch 2.11.0, found {torch.__version__}"
    )

if torch.version.cuda != "12.8":
    raise RuntimeError(
        f"Expected CUDA build 12.8, found {torch.version.cuda}"
    )
PY

# Use a project-specific venv rather than modifying Ubuntu's protected Python.
# --system-site-packages makes the venv reuse CUDA PyTorch from the base image.
RUN python -m venv \
        --system-site-packages \
        "${VIRTUAL_ENV}" && \
    "${VIRTUAL_ENV}/bin/python" -m pip install \
        --upgrade \
        pip \
        setuptools \
        wheel

ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

COPY requirements.txt /tmp/wrapper-lora-requirements.txt

RUN python -m pip install \
        -r /tmp/wrapper-lora-requirements.txt && \
    python -m pip check

# Import qualification only. The image-build machine does not need a GPU.
RUN python - <<'PY'
import importlib.metadata as metadata
import sys

import torch
import transformers
import accelerate
import peft
import qwen_asr
import soundfile
import librosa

print("Python:", sys.version)
print("Python executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("qwen-asr:", metadata.version("qwen-asr"))
print("Transformers:", transformers.__version__)
print("Accelerate:", accelerate.__version__)
print("PEFT:", peft.__version__)
print("SoundFile:", soundfile.__version__)
print("Librosa:", librosa.__version__)

# CUDA can be unavailable while building the image because image builds
# normally run without an attached GPU. Runtime GPU checks happen in a job.
print("CUDA available during image build:", torch.cuda.is_available())
print("ENVIRONMENT_BUILD_QUALIFICATION_PASSED")
PY

WORKDIR /workspace
