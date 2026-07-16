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

## 5. Single-H100 20-hour smoke training

The H100 job is implemented as full BF16 supervised fine-tuning of the pinned
`Qwen/Qwen3-ASR-0.6B` wrapper checkpoint. It uses Qwen's official target,
processor, prefix masking, and forward contract. This is intentionally not the
laptop LoRA runner, and it does not inherit the laptop's 5.3 GiB memory guard.

The checked-in job selects at most 20 training audio hours and 0.5 validation
hours, rejects clips over 30 seconds, uses one visible H100, batch size 1 with
gradient accumulation 8, and trains for one epoch. It stops on non-finite loss
or gradients, keeps the latest two resumable checkpoints, saves a final full
model, then loads that model in a new Python process and transcribes three
validation samples. It does not claim accuracy improvement from this smoke run.

### 5.1 Reuse the existing Azure environment

The environment you already created is the reusable Docker runtime for jobs.
Azure ML pulls version 2 onto the GPU compute, so the job does not create a
virtual environment or run `pip install`. The source repository is uploaded
separately, which means changes to this training module do not require a new
image while its package requirements stay unchanged.

Confirm that the registered version exists:

```bash
az ml environment show \
  --name orato-qwen3-asr-wrapper-lora \
  --version 2 \
  --output table
```

Only if version 2 is not registered in this workspace, create it once:

```bash
az ml environment create -f azureml/environments/wrapper-lora-v2.yml
```

The H100 job references it directly as
`azureml:orato-qwen3-asr-wrapper-lora:2`. Do not repeat the manual installation
commands from the CPU compute-instance section inside the GPU job.

### 5.2 Set the three Azure values

Use the processed-folder URI already qualified on CPU. Point `ORATO_SPLIT_URI`
at the owner-created `splitting/` datastore folder:

```bash
export ORATO_AZUREML_PROCESSED_URI="<full-azureml-datastore-uri-to-processed>"
export ORATO_SPLIT_URI="<full-azureml-datastore-uri-to-splitting>"
export ORATO_H100_COMPUTE="<h100-compute-name>"
```

The short workspace-relative form is:

```bash
export ORATO_SPLIT_URI="azureml://datastores/workspaceblobstore/paths/splitting"
```

The mounted folder must contain `train.jsonl`, `valid.jsonl`, and `test.jsonl`
at its root. This smoke job trains from `train.jsonl`, evaluates and performs
fresh-process predictions from `valid.jsonl`, and does not read `test.jsonl`.

Owner-created rows may contain provenance fields such as `dataset_folder`,
`original_audio_id`, `original_audio_path`, and `sample_rate`. The H100 input
adapter accepts those fields without making them model inputs. It uses
`audio_filepath`, `duration`, `text`, `language`, `source`, and `split`.
For example, `gram_vaani/audio/example.flac` resolves to that path beneath the
read-only mounted `processed/` input. Transcript text, including markers such
as `#incomplete`, is preserved exactly.

### 5.3 Submit and watch the GPU job

Run this from the cloned repository after pulling the latest commit:

```bash
git pull --ff-only

H100_JOB=$(az ml job create \
  -f azureml/jobs/official-sft-h100-20hr.yml \
  --set inputs.processed_data.path="$ORATO_AZUREML_PROCESSED_URI" \
  --set inputs.split_data.path="$ORATO_SPLIT_URI" \
  --set compute="azureml:${ORATO_H100_COMPUTE}" \
  --query name --output tsv)

echo "$H100_JOB"
az ml job stream --name "$H100_JOB"
```

The job pins GPU visibility to device 0, so a multi-GPU H100 VM still runs this
single-GPU milestone safely. Its 36-hour Azure ML timeout is a wall-clock safety
limit, not an expected runtime. Do not submit two copies simultaneously because
this profile deliberately uses stable output paths for resume.

Check status later or cancel a bad run:

```bash
az ml job show --name "$H100_JOB" --query status --output tsv
az ml job cancel --name "$H100_JOB"
```

### 5.4 What success looks like

The durable training output contains:

```text
prepared/train.jsonl
prepared/validation.jsonl
run_contract.json
trainer/checkpoint-*/
trainer/trainer_state.json
final/
training_summary.json
verification.json
```

Treat the run as technically successful only when the Azure job is `Completed`,
`training_summary.json` has `status: trained`, and `verification.json` has
`status: success` with three non-empty, non-collapsed predictions. The same job
can be resubmitted after interruption; `--resume` selects the highest durable
`trainer/checkpoint-*` directory.

The current official source contract is Qwen's
[`finetuning/qwen3_asr_sft.py`](https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning).
This milestone remains single-GPU. Eight-GPU distributed training is still out
of scope until this job and its fresh-process checkpoint verification pass.

### 5.5 Recommended first run: one hour

Before the 20-hour run, submit the isolated one-hour profile. It selects at
most one training audio hour and 0.1 validation hours, saves every 25 optimizer
steps, and has a six-hour wall-clock limit:

```bash
H100_JOB=$(az ml job create \
  -f azureml/jobs/official-sft-h100-1hr.yml \
  --set inputs.processed_data.path="$ORATO_AZUREML_PROCESSED_URI" \
  --set inputs.split_data.path="$ORATO_SPLIT_URI" \
  --set compute="azureml:${ORATO_H100_COMPUTE}" \
  --query name --output tsv)

az ml job stream --name "$H100_JOB"
```

The one-hour job writes to `official-sft-h100-1hr`, while the longer jobs write
to separate 20-hour and 100-hour locations. Never resume a longer dataset run
from the one-hour checkpoint. After the one-hour job and `verification.json`
pass, use the separate 20-hour or 100-hour job.

### 5.6 Run the isolated 100-hour profile

The 100-hour profile selects at most 100 training hours and one validation
hour. It saves every 200 optimizer steps, keeps the latest three checkpoints,
has a 72-hour wall-clock limit, and writes only to
`official-sft-h100-100hr`:

```bash
H100_JOB=$(az ml job create \
  -f azureml/jobs/official-sft-h100-100hr.yml \
  --set inputs.processed_data.path="$ORATO_AZUREML_PROCESSED_URI" \
  --set inputs.split_data.path="$ORATO_SPLIT_URI" \
  --set compute="azureml:${ORATO_H100_COMPUTE}" \
  --query name --output tsv)

echo "$H100_JOB"
az ml job stream --name "$H100_JOB"
```

Resubmitting this same 100-hour job can resume its highest durable
`checkpoint-*`. It cannot see the one-hour checkpoints because its output path
is different. Do not run two copies of the 100-hour job simultaneously.

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
