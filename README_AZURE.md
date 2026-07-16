# Azure CPU setup and data qualification

This runbook validates the repository, processed manifests, portable audio
paths, split generation, and a small number of real audio files on an Azure ML
CPU compute instance. It does **not** run LoRA training: those commands require
CUDA, and CPU success does not prove H100 memory or throughput behavior.

Processed audio stays in place. Generated split manifests live under a
separate split root and contain paths such as
`rasa_hindi/audio/example.flac`, resolved through `ORATO_DATA_ROOT`.

## 1. Compute identity and storage access

Prefer a system- or user-assigned managed identity. Grant it `Storage Blob
Data Reader` on the storage account/container. Never put a storage key, SAS
token, or connection string in Git.

```bash
az login --identity
az account show --output table
```

If the compute instance uses user authentication, use interactive `az login`.
See Azure's current [compute-instance](https://learn.microsoft.com/azure/machine-learning/how-to-create-compute-instance?view=azureml-api-2)
and [authentication](https://learn.microsoft.com/azure/machine-learning/how-to-setup-authentication?view=azureml-api-2)
guides.

## 2. Clone and create the CPU validation environment

Keep the clone under your Azure ML user files:

```bash
cd ~/cloudfiles/code/Users/<your-user-name>
git clone https://github.com/Anandxz/Orato_finetuning_qwen.git
cd Orato_finetuning_qwen

python3.12 -m venv .venv-azure-cpu
source .venv-azure-cpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/cpu-validation.txt
python -m pip install -e ".[dev]"
python -m pip check

orato-asr --version
orato-asr doctor
python -m pytest
```

If `python3.12` is unavailable, create a dedicated Conda environment instead:

```bash
conda create --name orato-azure-cpu python=3.12 -y
conda activate orato-azure-cpu
```

Installation happens only here, never from application, training, or notebook
code. This CPU environment omits PyTorch and the Qwen model because this pass
is data/pipeline qualification.

## 3A. Interactive mode: direct Blob plus cache

Use the container and account names from private Azure configuration:

```bash
export AZURE_STORAGE_ACCOUNT_NAME="<storage-account-name>"
export AZURE_STORAGE_CONTAINER="<container-name>"
export ORATO_STORAGE_BACKEND="azure_blob"
export ORATO_DATA_ROOT="az://${AZURE_STORAGE_CONTAINER}/processed"
export ORATO_SPLIT_ROOT="$PWD/data/splits"
export ORATO_CACHE_ROOT="/mnt/resource/orato-data-cache"

mkdir -p "$ORATO_CACHE_ROOT" outputs/cpu-smoke reports/data
```

`DefaultAzureCredential` uses the compute identity. An environment-only
`AZURE_STORAGE_CONNECTION_STRING` is also supported, but it must never be
written to `.env`, YAML, notebooks, logs, or Git.

Build and validate `split_all`. Direct Blob mode downloads only source
manifests here; it does not copy audio into the split directory.

```bash
orato-asr data build-splits --config configs/splits/split_all.yaml

orato-asr data validate-splits \
  --split-dir "$ORATO_SPLIT_ROOT/split_all/v1"
```

Then localize and decode three bounded samples. This proves credentials, cache
writes, path resolution, and SoundFile decoding without scanning all audio:

```bash
orato-asr data select \
  --manifest "$ORATO_SPLIT_ROOT/split_all/v1/train.jsonl" \
  --output outputs/cpu-smoke/train-three.jsonl \
  --max-samples 3

orato-asr data validate \
  --manifest outputs/cpu-smoke/train-three.jsonl \
  --check-audio \
  --report reports/data/azure_cpu_audio_validation.json
```

Expected results:

- `build-splits` prints train/validation/test rows, hours, group counts, and a
  fingerprint.
- `validate-splits` reports `status: success` and zero group leakage.
- Bounded audio validation returns 0 and caches three blobs.
- Rebuilding with `--overwrite` produces the same fingerprint when inputs and
  configuration are unchanged.

## 3B. Azure ML job mode: read-only mount

Use the long Azure ML datastore URI as a `uri_folder` input and mount it
read-only. Override the placeholders when submitting the checked-in job:

```bash
export ORATO_AZUREML_PROCESSED_URI="<full-azureml-datastore-uri-to-processed>"
export ORATO_CPU_COMPUTE="<cpu-compute-name>"

az ml environment create -f azureml/environments/wrapper-lora-v2.yml

az ml job create -f azureml/jobs/build-splits-cpu.yml \
  --set inputs.processed_data.path="$ORATO_AZUREML_PROCESSED_URI" \
  --set compute="azureml:${ORATO_CPU_COMPUTE}"
```

Azure ML supports datastore URIs for folder data and mount/download job input
modes; see [data assets](https://learn.microsoft.com/azure/machine-learning/how-to-create-data-assets?view=azureml-api-2)
and [job data access](https://learn.microsoft.com/azure/machine-learning/how-to-read-write-data-v2?view=azureml-api-2).
The job writes the split bundle to an Azure ML output, never into `processed/`.

## 4. Split contents and versions

```text
split_all/v1/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
├── split_config.yaml
├── split_report.json
└── split_fingerprint.txt
```

The builder keeps linked session/call/source/speaker groups together, reports
duplicates and missing metadata, and balances primarily by audio duration and
secondarily by row count plus overlapping metadata features. Repeated
transcripts are reported but retained because call-centre phrases can
legitimately repeat.

To create another experiment/version, change `name` or `version`. Existing
versions require explicit `--overwrite`:

```bash
orato-asr data build-splits --config configs/splits/split1.yaml
```

## 5. H100 hand-off

Use the same immutable processed root and generated split manifests. For H100
throughput, prefer a read-only Azure ML mount or stage selected data to
node-local NVMe; per-sample Blob download is a correctness fallback, not the
recommended training path.

Retain the split fingerprint/report, exact passing test summary, bounded audio
validation report, and Azure environment qualification result. The wrapper
training commands remain CUDA-only. Do not run them on CPU or claim H100
readiness until the H100 environment/profile, one-step backward pass, adapter
save, fresh-process reload, and fixed-sample predictions pass.

The current upstream Qwen repository now publishes
[`finetuning/qwen3_asr_sft.py`](https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning)
with single-GPU and `torchrun` usage. Before the H100 run, reconcile and qualify
this repository's training backend against that current official SFT contract;
the CPU data milestone does not authorize or validate an older/custom training
path.

## Troubleshooting

- `AuthorizationPermissionMismatch`: grant the compute identity Storage Blob
  Data Reader and allow time for role propagation.
- Missing Azure modules: install `requirements/cpu-validation.txt` in the
  active environment.
- Duplicate paths: do not include both `manifest_normal.jsonl` and
  `manifest_full.jsonl` unless they are proven disjoint.
- Mounted missing files: `ORATO_DATA_ROOT` must directly contain the dataset
  folders.
- Changed fingerprint: compare source-manifest fingerprints and recorded
  `split_config.yaml` before allocating H100.
