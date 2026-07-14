# Orato Qwen3-ASR

Orato is building a reproducible Python workflow for adapting Qwen3-ASR to
Hindi, English, and natural Hindi-English code-switching in real-time
voice-agent calls. The canonical transcript style uses Devanagari for Hindi
and Latin script for English:

```text
मुझे appointment next Monday के लिए reschedule करना है
```

See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for the durable business, data,
training, evaluation, and deployment context.

## Current status

This milestone selects the native Transformers track, adds one-shot base
inference, and adds a dependency-free manifest and base-evaluation pipeline.
It provides:

- Strict schema-v3 configuration profiles, including bounded baseline policy.
- Lazy environment, dependency, CUDA, and GPU reporting.
- Non-mutating float32 audio decoding, mono downmixing, and 16 kHz resampling.
- Explicit native processor/model loading and deterministic transcription.
- Streaming JSONL validation, summary, selection, and split-overlap checks.
- Raw and standard normalized WER/CER plus incremental baseline reports.
- CLI model inspection, ML doctor, inference preflight, data commands, and sanitized JSON.
- Dependency-free unit tests plus owner-audio/manifest-gated integration tests.

The selected model is `Qwen/Qwen3-ASR-0.6B-hf` at revision
`6aa69c382e2b426eee1f5870d4c95859a74b6445`, loaded only with native
`AutoProcessor` and `AutoModelForMultimodalLM` classes. The older
`Qwen/Qwen3-ASR-0.6B`/`qwen-asr` wrapper is a separate alternative and must not
be mixed with this integration.

Fine-tuning, checkpoints, Azure authentication/downloads, distributed
inference, FlashAttention, quantization, and serving are not implemented.
Configuration training values remain unqualified examples, not recommended
hyperparameters or memory-fit claims.

## Repository structure

```text
configs/             Hardware and data-scale profiles
requirements/        Explicit inference pins and later training candidates
scripts/             Thin preflight entry point
src/orato_asr/       Authoritative package and native inference code
src/orato_asr/data/  Strict manifest and leakage-checking utilities
src/orato_asr/evaluation/  Text metrics and incremental base evaluation
tests/               Offline unit tests and gated integration test
outputs/             Ignored model caches and generated artifacts
reports/             Ignored sanitized qualification evidence
```

## Python 3.12 inference environment

Python 3.12 is the recommended and qualified target. The lightweight package
metadata supports Python 3.11 through 3.13. Use a dedicated environment and
install PyTorch from its official wheel index before the inference group:

```bash
python3.12 -m venv .venv-inference
source .venv-inference/bin/activate
python -m pip install --upgrade pip

# NVIDIA CUDA 12.8 environment
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# For a CPU-only environment, replace the command above with:
# python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu

python -m pip install -r requirements/inference.txt
python -m pip install -e ".[dev]"
python -m pip check
```

The selected direct pins are Transformers 5.13.0, NumPy 2.4.2, SoundFile
0.14.0, soxr 1.1.0, huggingface-hub 1.23.0, tokenizers 0.22.2, and
safetensors 0.8.0. Accelerate 1.14.0 and Datasets 5.0.0 are recorded only as
future training candidates. `qwen-asr`, vLLM, FlashAttention, serving
frameworks, PEFT, Azure SDKs, and MLflow are not installed.

Application and training code never installs or upgrades packages.

## Command-line usage

Foundation and configuration checks:

```bash
orato-asr --version
orato-asr config show --config configs/local_tiny.yaml
orato-asr config validate --config configs/local_tiny.yaml
orato-asr doctor
orato-asr doctor --ml --json reports/environment/environment_report.json
```

Model inspection does not load or download anything unless `--load` is given:

```bash
orato-asr model info --config configs/local_tiny.yaml
orato-asr model info --config configs/local_tiny.yaml --load --device cuda
```

One-shot transcription accepts only a local WAV or FLAC file:

```bash
orato-asr transcribe \
  --audio /path/to/legal_non_pii.wav \
  --config configs/local_tiny.yaml \
  --device auto \
  --output-json reports/environment/base_inference_result.json
```

`auto` selects CUDA when PyTorch reports it available and otherwise CPU.
CPU always uses float32. CUDA `auto` uses bfloat16 when supported and float16
otherwise. Explicit unavailable CUDA or unsupported precision fails without a
CPU fallback. Hugging Face tokens remain environment-managed and are never
stored in YAML or reports. `--offline` requires the pinned model to be present
in the selected local cache.

Inference preflight checks the exact pins, decoder libraries, model metadata,
cache state, device, CUDA, and output writability without loading the model:

```bash
python scripts/preflight.py --inference --device auto \
  --report-dir reports/environment
python scripts/preflight.py --inference --device cuda --load-model \
  --report-dir reports/environment
```

## Manifests and base evaluation

The canonical manifest is UTF-8 JSON Lines. Every row has exactly
`audio_filepath` and `text` plus these optional top-level fields: `duration`,
`language`, `source`, `speaker_id`, `recording_id`, `domain`, `split`, and
`metadata`. Dataset-specific extensions belong only inside `metadata`.

```json
{"audio_filepath":"audio/call-0001.flac","text":"मुझे appointment reschedule करना है","duration":3.42,"language":"hi","source":"calls","recording_id":"call-0001","metadata":{"split_source":"owner-reviewed"}}
```

Local audio may be an absolute path or repository-relative path. `azureml:`,
Blob-style, and HTTP(S) locators are accepted structurally but are never
downloaded, authenticated, or validated locally. Do not put a SAS token or a
private URL in a manifest intended for shared reports. This repository does
not copy or document owner Azure identifiers.

```bash
orato-asr data validate --manifest /private/eval.jsonl --check-audio \
  --report reports/evaluation/manifest_validation.json
orato-asr data summarize --manifest /private/eval.jsonl \
  --output reports/evaluation/manifest_summary.json
orato-asr data select --manifest /private/eval.jsonl \
  --output /private/eval_ten.jsonl --max-samples 10 --shuffled --seed 17
orato-asr data check-overlap --train-manifest /private/train.jsonl \
  --evaluation-manifest /private/eval.jsonl --hash-local-audio \
  --output reports/evaluation/overlap.json
```

`data validate` returns `2` for malformed rows or invalid checked media.
Overlap of normalized audio paths, local content hashes, or recording IDs is
prohibited and returns `1`; repeated transcript text is informational. Speaker
overlap becomes prohibited only with `--disallow-speaker-overlap`.

Run a bounded base evaluation only against local readable WAV/FLAC records:

```bash
orato-asr evaluate baseline --manifest /private/eval.jsonl \
  --config configs/local_tiny.yaml --run-name base-eval-001 \
  --max-samples 10 --offline
```

Each run writes ignored files below `reports/<profile>/evaluation/<run-name>/`:
the resolved config, summary, incrementally persisted predictions and
failures, metrics JSON/CSV, worst examples, and a short README. Use `--resume`
only with the same immutable manifest; use `--overwrite` explicitly to replace
a run. The normal `continue` policy records individual failures and still
completes. A blank, punctuation-only, or identical-prediction collapse in the
first five successful samples stops the run and returns `1`.

Metrics are decimal ratios, never unlabeled percentages. Raw WER/CER compare
source text; standard WER/CER apply NFKC, whitespace collapse, punctuation
canonicalization, and ASCII Latin lowercasing while preserving Devanagari,
numbers, and code-switching. CER compares Unicode code points excluding
whitespace. Punctuation remains by default and can be removed only through
the configured baseline policy.

## Configuration profiles

- `local_tiny.yaml`: RTX 3050/CPU qualification, at most 50 samples, inference
  device `auto`.
- `h100_smoke.yaml`: planned one-hour, single-H100 work; inference device
  `cuda`.
- `h100_100hr.yaml`: planned 100-hour, single-H100 work; inference device
  `cuda`.
- `h100_8gpu.yaml`: planned one-node/eight-H100 training capability. It remains
  valid configuration but is rejected by the current single-process inference
  command.

All profiles pin the same native model/processor revisions, use precision
`auto`, default to automatic language detection, cap generation at 256 new
tokens, and keep the model cache under `outputs/`. Profile paths are strictly
repository-relative and configuration loading never creates directories or
rewrites YAML.

## Tests and real-inference gate

```bash
python -m pytest
```

Unit tests require no model, network, GPU, or audio. A real integration test
only runs when the owner supplies legal, non-PII audio:

```bash
ORATO_ASR_TEST_AUDIO=/path/to/legal_non_pii.wav \
  python -m pytest -m integration -v
```

The bounded offline baseline integration check is separately owner-gated:

```bash
ORATO_ASR_TEST_MANIFEST=/path/to/legal_local_eval.jsonl \
  python -m pytest -m integration -v
```

No audio, manifest, prediction, or Azure locator is generated or committed. A
skipped integration test is not evidence that transcription or evaluation
succeeded.

## Security and limitations

Never commit credentials, private URLs, PII, audio, datasets, model weights,
checkpoints, model caches, manifests, or generated predictions. Raw source
datasets are immutable. Audio conversion occurs in memory and never changes
the source. Reports retain transcript text and source paths for review, but
strip URI query strings and must not contain secrets.

The unquantized model's memory fit on an RTX 3050 6 GB and a roughly 8 GB host
remains an explicit qualification question. A CUDA OOM is reported as a real
blocker with no silent CPU retry. FlashAttention and quantization are excluded,
and local success would not prove H100 training behavior.
