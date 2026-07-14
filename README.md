# Orato Qwen3-ASR

Orato is building a reproducible Python workflow for adapting
`Qwen/Qwen3-ASR-0.6B` to Hindi, English, and natural Hindi-English
code-switching in real-time voice-agent calls.

The canonical transcript style uses Devanagari for Hindi and Latin script for
English, for example:

```text
मुझे appointment next Monday के लिए reschedule करना है
```

See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for the durable business, data,
training, evaluation, and deployment context.

## Current status

This repository currently provides only the project foundation:

- A small installable Python package.
- Four illustrative configuration profiles.
- Configuration validation and safe local path resolution.
- A lightweight command-line interface and environment doctor.
- Offline, CPU-only foundational tests.

Qwen loading, audio and manifest handling, training, evaluation, checkpoints,
Azure integration, distributed launch, and GPU checks are **not implemented in
this milestone**. The configuration values are unqualified examples, not
recommended hyperparameters or claims about memory fit.

## Future execution modes

One eventual training entry point is intended to serve:

- Local CPU validation and reporting checks.
- RTX 3050 6 GB pipeline qualification with at most tiny workloads where they
  fit.
- Single-H100 tiny, one-hour, and approximately 100-hour experiments.
- One-node, eight-H100 distributed execution after the single-GPU path works.
- Approximately 1,000-hour capability after smaller runs are qualified.

The local profile does not claim that full-parameter Qwen3-ASR fine-tuning fits
in 6 GB VRAM. The eight-GPU profile describes a planned capability only.

## Repository structure

```text
configs/             Illustrative hardware and data-scale profiles
requirements/        Dependency qualification notes for future milestones
scripts/             Thin command-line wrappers
src/orato_asr/       Authoritative Python package
tests/               Inexpensive offline tests
outputs/             Generated run artifacts (ignored except .gitkeep)
reports/             Generated reports (ignored except .gitkeep)
```

Future reports are expected to include resolved configuration and environment
information, metrics, predictions, checkpoint verification, and PNG graphs.
Generated artifacts remain outside version control.

## Development installation

Python 3.12 is recommended. The package supports Python 3.11 through 3.13.
Create and activate an isolated environment before installing dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Dependency installation belongs in environment setup, never in application,
training, or notebook code.

## Command-line usage

```bash
orato-asr --version
orato-asr config show --config configs/local_tiny.yaml
orato-asr config validate --config configs/local_tiny.yaml
orato-asr doctor
```

The doctor performs only foundation checks: Python and package availability,
repository directories, platform/WSL detection, and the current working
directory. It does not inspect Qwen, CUDA, Azure, or training readiness.

`scripts/preflight.py` invokes the same doctor implementation:

```bash
python scripts/preflight.py
python scripts/preflight.py --config configs/local_tiny.yaml
```

With `--config`, preflight also validates the profile and reports its name,
hardware mode, scale intent, GPU count, and distributed status. It does not
perform model or training qualification.

## Running tests

After installing the development extra, run the offline test suite:

```bash
python -m pytest
```

Tests require no internet, GPU, Azure account, Qwen model, audio, or external
dataset.

## Configuration profiles

- `local_tiny.yaml`: local RTX 3050/CPU qualification, capped at 50 samples.
- `h100_smoke.yaml`: planned single-H100 run with about one hour of audio.
- `h100_100hr.yaml`: planned single-H100 run with about 100 hours of audio.
- `h100_8gpu.yaml`: planned one-node, eight-H100 capability at roughly
  1,000-hour scale.

Profiles use a strict schema. Unknown keys, unsafe output paths, non-positive
numeric values, and inconsistent GPU/process settings fail with actionable
errors. Loading or displaying a profile never creates directories or rewrites
the source YAML. The profiles also lock Hindi to Devanagari, English to Latin
script, Roman Hinglish as a non-primary target, valid short utterances as
allowed, and raw-data mutation and synthetic data as disabled.

## Security and data safety

Never commit credentials, private URLs, PII, audio, datasets, model weights,
checkpoints, generated predictions, or local environment files. Raw source
datasets are immutable and must never be modified by future training code.

Copy `.env.example` only for local setup and keep real values in an ignored
`.env` file. No Azure or Hugging Face credentials are required by the current
foundation.
