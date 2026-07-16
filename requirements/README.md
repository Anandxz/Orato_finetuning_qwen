# Dependency policy

The native Transformers integration uses Python 3.12, PyTorch 2.11.0,
Transformers 5.13.0, and the exact package pins in `inference.txt`. The core
package stays lightweight; inference dependencies are an explicit environment
choice.

Install PyTorch first from an official index, then the inference group:

```bash
# NVIDIA CUDA 12.8
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# Or CPU only
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu

python -m pip install -r requirements/inference.txt
python -m pip install -e ".[dev]"
```

`train-candidate.txt` records Accelerate and Datasets versions for a later
training qualification. It is not part of the default install.

Wrapper LoRA qualification uses a deliberately separate Python 3.12
environment because `qwen-asr==0.0.6` requires Transformers 4.57.6 and the
pre-1.0 Hugging Face Hub line. Those versions are incompatible with the
native inference environment's Transformers 5.13.0/Hub 1.23.0 combination.

```bash
python3.12 -m venv .venv-qwen-wrapper
source .venv-qwen-wrapper/bin/activate
mkdir -p outputs/pip-tmp outputs/pip-cache-wrapper
export TMPDIR="$PWD/outputs/pip-tmp"
export PIP_CACHE_DIR="$PWD/outputs/pip-cache-wrapper"

python -m pip install torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/wrapper-lora.txt
python -m pip install -e ".[dev]"
python -m pip check
```

The wrapper group pins PEFT but excludes Datasets because the project uses a
lazy canonical-manifest dataset and a direct bounded loop. `qwen-asr` declares
Gradio and Flask as mandatory package dependencies even though this project
does not implement or start a web UI. FlashAttention, vLLM, bitsandbytes,
QLoRA, DeepSpeed, Azure SDKs, and MLflow remain absent.

The inspected `qwen-asr==0.0.6` wheel matches official Qwen source commit
`7c6daf77a2421100f5fb066495372c00129d39ff`; its qualified wheel SHA-256 is
`b9c55a38413298f3a990a4475467399daec6e8f4172363053fc42e2166c2dfd3`.

The native track must use `Qwen/Qwen3-ASR-0.6B-hf` with Transformers native
classes. The older `Qwen/Qwen3-ASR-0.6B`/`qwen-asr` wrapper is the isolated
training backend and must never be mixed with the native checkpoint or classes.

The native environment still does not install `qwen-asr` or PEFT. Application,
training, and notebook code never installs or upgrades packages.

For CPU-only Azure data and split qualification, use
`requirements/cpu-validation.txt`. It adds the audio decoder stack and optional
Azure Identity/Blob clients, but deliberately omits PyTorch, Transformers,
qwen-asr, PEFT, and model downloads. Azure packages are only needed for direct
`az://` localization; Azure ML mounts and local paths use the local backend.
