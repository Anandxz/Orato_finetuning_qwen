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

The native track must use `Qwen/Qwen3-ASR-0.6B-hf` with Transformers native
classes. The older `Qwen/Qwen3-ASR-0.6B`/`qwen-asr` wrapper is a separate later
alternative and must never be mixed with the native checkpoint or classes.

Not installed in this milestone: `qwen-asr`, vLLM, FlashAttention, serving
frameworks, PEFT, Azure SDKs, and MLflow. Application, training, and notebook
code must never install or upgrade packages.
