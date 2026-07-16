# Orato Qwen3-ASR Project Context

## Purpose of this document

This file is the durable context for future work on the Orato Hindi-Hinglish ASR training project. Read it before planning or changing the repository.

The project is intentionally practical. It must produce a reproducible Python training and evaluation workflow, not a large governance system. An earlier attempt at extensive architecture audits, issue matrices, promotion boards, and repeated Git-foundation reviews was abandoned. Do not recreate it.

## Business objective

Orato is building speech recognition for real-time voice agents serving Indian users. The initial system should support calling workflows such as:

- Customer support.
- Dental and hospital reception.
- Appointment booking, cancellation, and rescheduling.
- Insurance enquiries and follow-ups.
- Sales and lead qualification.
- General customer-service calls.

The long-term product should be reusable across industries while retaining domain-specific vocabulary and evaluation.

## ASR objective

The selected version-1 integration target is:

`Qwen/Qwen3-ASR-0.6B-hf`

The repository uses the native Transformers track for inference. Model and
processor are both pinned to revision
`6aa69c382e2b426eee1f5870d4c95859a74b6445`. The qualified dependency target
is Python 3.12, PyTorch 2.11.0 with the official CUDA 12.8 or matching CPU
wheel, Transformers 5.13.0, NumPy 2.4.2, SoundFile 0.14.0, soxr 1.1.0,
huggingface-hub 1.23.0, tokenizers 0.22.2, and safetensors 0.8.0. The recorded
upstream references are Qwen repository commit
`7c6daf77a2421100f5fb066495372c00129d39ff` and Transformers v5.13.0 tag commit
`6af945f436d85f2b0c5dff9b14feccd27b1d470b`.

Base inference is intentionally unquantized and excludes FlashAttention,
vLLM, serving frameworks, PEFT, Azure, and MLflow. Current inference is
single-process. Native one-shot CUDA/BF16 model load and inference have been
qualified on the RTX 3050 6 GB; failures must never silently fall back to CPU.

Training is a separate coherent backend using the non-`-hf`
`Qwen/Qwen3-ASR-0.6B` wrapper checkpoint. It never mixes wrapper classes,
processor inputs, or dependencies into native inference.

The model should eventually recognize:

- Hindi.
- English.
- Natural Hindi-English code-switching.
- Indian and regional accents.
- Telephone-band audio.
- Realistic background noise and channel distortion.
- Short, incomplete, corrected, or interrupted utterances.
- Names, cities, organizations, medicines, procedures, and other domain terms.
- Phone numbers, dates, times, amounts, identifiers, and confirmation phrases.

This is not a plan to train an ASR model from scratch. Version 1 adapts and evaluates an existing open-weight foundation model.

## Canonical transcript policy

The primary ASR target uses mixed script:

- Hindi words use Devanagari.
- English words use Latin script.

Example:

`मुझे appointment next Monday के लिए reschedule करना है`

Additional rules:

- Roman Hinglish is not the primary version-1 ASR target. It may be explored later as a separate model experiment or produced by downstream transliteration.
- Do not train the same audio against conflicting transcript styles without an explicitly designed experiment.
- Spoken numbers and dates may remain in acoustic/verbatim form initially. Inverse text normalization should be a separate, testable layer.
- Preserve negations, meaningful corrections, and repeated critical information.
- Valid short utterances must not be removed merely because of word count. Examples include `हाँ`, `नहीं`, `जी`, `चार`, `okay`, names, and cities.
- Transcript normalization must be versioned once implemented.

## Data principles

- Raw source audio and source manifests are immutable.
- Every training row must have a trustworthy audio-to-transcript mapping.
- Localized audio filenames must be collision-safe; basenames alone are insufficient across datasets.
- Train, validation, and evaluation splits must prevent duplicate audio and same-session leakage where metadata permits.
- Evaluation data must remain separate from training, augmentation, synthetic generation, and pseudo-labelling inputs.
- Real conversational and telephone speech has higher product value than large volumes of clean read speech.
- Synthetic speech may later target rare names and vocabulary, but it must not replace real-call evaluation.
- Dataset mixture, duration limits, sample limits, and random seeds must be configurable.
- Large manifests should eventually support bounded-memory or streaming access rather than requiring all rows in memory.
- Dataset licences, commercial-use rights, consent, and PII risk must be checked before production training. Exact permissions are not yet confirmed.

### Current manifest boundary

The repository now supports strict, UTF-8 JSONL manifests for local data work.
Canonical rows require `audio_filepath` and `text`; optional `duration`,
language/source/speaker/recording/domain/split identifiers, and nested
`metadata` support deterministic validation, selection, and leakage checks.
Absolute and repository-relative local audio paths remain supported for
backward compatibility. New split manifests use processed-root-relative
logical paths. A central resolver supports local/Azure-mounted roots and an
optional direct `az://` Blob backend with atomic deterministic caching;
managed identity or environment-managed credentials are required. Other
remote locators remain structural-only. Do not commit real Azure identifiers,
private URLs, SAS tokens, manifests, audio, or data-derived reports.

Duplicate normalized audio paths, content hashes when locally requested, and
recording IDs are leakage; repeated transcript text is only informational.
Speaker overlap is optional policy because its meaning depends on the dataset.

## Execution targets

The project should use the same core Python path across execution environments. Hardware profiles may change precision, batch size, worker count, accumulation, launch method, and distributed settings, but they should not create separate training implementations.

### Local CPU

Use for inexpensive checks such as configuration validation, manifest validation, imports, report generation, and other non-training tests.

### RTX 3050 laptop with 6 GB VRAM

Treat the laptop primarily as a pipeline qualification system:

- Validate manifests and configuration.
- Load and preprocess a small number of audio clips.
- Run base-model inference when memory permits.
- Attempt a one-batch forward or backward check only when it fits.
- Exercise save/reload and reporting utilities where possible.

Do not claim that full-parameter Qwen3-ASR-0.6B supervised fine-tuning fits in 6 GB VRAM. Local success proves code-path correctness only to the extent actually exercised; it does not prove H100 performance or training quality.

### One NVIDIA H100

This is the first real training target. It should eventually support:

- A tiny integration run.
- An approximately one-hour smoke dataset.
- Medium-scale training, including an approximately 100-hour selected dataset.
- Validation, checkpointing, resume, evaluation, and reporting.

### Eight NVIDIA H100 GPUs

Add this only after the single-H100 path works reliably. The intended initial topology is one node, one process per GPU, using a standard distributed mechanism supported by the verified training stack. Shared artifacts must be written safely, normally by rank zero.

The approximately 1,000-hour scale is a capability target, not a confirmed first training run.

## Training direction

Version 1 must follow the current official Qwen3-ASR supervised fine-tuning implementation as closely as practical. Before implementation, verify the exact official repository, training entry point, dataset schema, package compatibility, model revision, and checkpoint lifecycle against current primary documentation.

The official target, processor, masking, and forward contract must be proven
before any project adaptation. The training path must not introduce:

- Automatic LoRA target discovery.
- A custom ASR architecture.
- A custom processor.
- A custom target convention that conflicts with the official data contract.
- Unverified label masking or collation.
- Notebook-only training logic.

The pinned native Transformers label helper failed that contract, so it is not
used for training. The isolated wrapper backend preserves Qwen's official
supervised target and collator while applying an explicitly approved,
full-path text-attention Q/V LoRA allowlist for the 6 GB laptop qualification.
This is a controlled project experiment, not an official Qwen LoRA recipe or
a runtime fallback. It is not accepted until backward, adapter-only save, a
fresh-process reload, and fixed-sample inference all pass.

The local runner also has a separate one-epoch mode for an owner-approved,
canonical local manifest. It is gated by an exact-manifest compatibility run,
one optimizer step, a five-step smoke, and fresh-process adapter verification.
It keeps the same batch-one/accumulation-eight LoRA contract and hard memory
guards. Full-epoch execution must consume each selected training row once;
validation and test rows remain excluded from optimization.

## Training engineering principles

- Authoritative logic belongs in version-controlled Python modules and command-line entry points.
- Notebooks may inspect data and visualize results, but must not own the training implementation.
- Install and qualify dependencies before allocating expensive GPUs.
- Training code must never run `pip install` or silently upgrade packages.
- Pin Python, PyTorch/CUDA compatibility, Qwen package/repository revision, model revision, and important dependencies after qualification.
- Use configuration for sample count, audio-hour limits, steps, batch sizing, precision, evaluation cadence, checkpoint cadence, and hardware profile.
- Run an inexpensive preflight and a tiny forward/backward qualification before a longer training run.
- Stop immediately on NaN or infinite loss, gradients, or other critical numeric failures.
- Save enough state to resume a supported training run.
- A saved checkpoint is not accepted until a new process loads it and produces meaningful predictions.
- Transcribe a few fixed samples before starting a long evaluation. Abort early on blank, identical, repetitive, or punctuation-only collapse.
- Preserve resolved configuration, environment versions, dataset identity, and model revision with every run.

## Azure Machine Learning direction

Azure Machine Learning is the planned cloud execution environment.

The future workflow should provide:

- A versioned Azure ML Environment built before H100 allocation.
- A reproducible job entry point rather than an interactive training notebook.
- Managed identity or another deliberately selected authentication method.
- Readable data inputs and persistent output locations.
- Durable checkpoints, metrics, predictions, logs, and reports.
- Cheap preflight or smaller-compute qualification before H100 use.
- H100 x1 success before H100 x8 work.
- Fast failure and compute shutdown when qualification checks fail.

No real Azure subscription IDs, workspace names, storage keys, SAS tokens, private Blob URLs, or credentials belong in this repository.

## Evaluation requirements

The system must compare the base model and trained checkpoint on the same immutable evaluation inputs and normalization policy.

Planned recognition metrics include:

- Word Error Rate (WER).
- Character Error Rate (CER).
- Mixed Error Rate (MER) for mixed-script speech.
- Hindi-token and English-token error rates.
- Entity and number accuracy.
- Blank-output and repetition rates.
- Hallucination indicators, including silence/non-speech cases.

Results should also be grouped where metadata permits by dataset, evaluation category, audio duration, acoustic condition, and other relevant slices.

Planned training and systems metrics include:

- Training and validation loss.
- Learning rate.
- Gradient norm.
- Step and data-loading time.
- Samples and audio seconds processed per second.
- GPU allocated/reserved memory and utilization.
- Checkpoint timing and skipped/corrupt sample counts.

Planned static graphs include:

- Training and validation loss.
- Learning-rate schedule.
- Gradient norm.
- Step time and throughput.
- GPU memory and utilization.
- Base-versus-fine-tuned WER/CER/MER.
- Metrics by dataset, category, and duration.
- Dataset sampling distribution and error-type distribution.

### Current base-evaluation boundary

The current baseline runner evaluates only the selected base model on local
WAV/FLAC records. It uses one model instance per run, persists sample JSONL
results incrementally, supports safe resume by deterministic manifest-line
sample ID, and releases CUDA resources at completion. It reports raw and
standard-normalized WER/CER, edit counts, exact/blank/punctuation/failure
rates, timing, and real-time factor. Standard comparison uses NFKC,
whitespace collapse, punctuation canonicalization, Latin lowercasing, and
optional punctuation removal without transliterating Devanagari or
code-switching.

The default run is intentionally capped at ten samples. It records individual
failures under a `continue` policy and stops early after five successful
predictions if blank, punctuation-only, or identical output reaches the
configured threshold. Fine-tuning evaluation, trained checkpoints, MER,
entity metrics, graphs, Azure execution, and distributed evaluation remain
unimplemented.

MLflow is a likely experiment tracker for Azure, while CSV, JSON, Markdown, prediction JSONL, and PNG outputs should remain portable. The exact tracking implementation is deferred until the core model lifecycle works.

## Lessons from the failed notebook

The earlier notebook is evidence that a short script can still be operationally unsafe. Owner-reported observations included runtime package installation, a custom path that diverged from Qwen's official fine-tuning approach, mutable notebook state, NaN validation loss, punctuation-only output after training, lengthy evaluation after obvious collapse, and inconsistent authentication behaviour.

These observations do not establish one definitive root cause. They establish engineering requirements:

- Prove the official path before customization.
- Separate environment setup, data preparation, training, checkpoint verification, and evaluation.
- Treat one finite forward pass as necessary but not sufficient.
- Fail on NaN or Inf.
- Inspect a few predictions before full evaluation.
- Verify checkpoints from a fresh process.
- Keep an exact record of software and model revisions.

## Immediate milestone sequence

1. Create a minimal Python package, configuration profiles, CLI shell, tests, and Git safety rules.
2. Qualify and pin the official Qwen3-ASR dependency stack using current primary documentation.
3. Implement manifest loading, validation, bounded selection, and a tiny human-verified fixture contract.
4. Implement base-model loading and inference through the selected official Qwen path.
5. Complete a tiny supervised lifecycle: preprocess, forward, backward, save, exit, fresh reload, and fixed-sample inference.
6. Add evaluation metrics, early-collapse checks, portable reports, and graphs.
7. Package a versioned Azure ML Environment and run inexpensive preflight checks.
8. Run the H100 x1 tiny qualification and one-hour smoke training.
9. Scale to a selected 100-hour experiment after reviewing smoke results.
10. Add H100 x8 distributed execution only when single-GPU behaviour is stable and scaling is justified.

Each milestone should deliver working, tested behavior. Do not replace implementation progress with large planning packages.

## Confirmed decisions

- Initial native inference model: `Qwen/Qwen3-ASR-0.6B-hf` at the pinned
  revision recorded above; the older wrapper checkpoint is a separate later
  alternative.
- Initial training direction: official supervised fine-tuning path first.
- Native Transformers 5.13 label construction is not used for training. The
  separate training backend is `Qwen/Qwen3-ASR-0.6B` revision
  `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` with `qwen-asr==0.0.6`,
  Transformers 4.57.6, Accelerate 1.12.0, and PEFT 0.19.1 in its own
  environment. The inspected `qwen-asr` 0.0.6 wheel source matches Qwen commit
  `7c6daf77a2421100f5fb066495372c00129d39ff`; the qualified wheel SHA-256 is
  `b9c55a38413298f3a990a4475467399daec6e8f4172363053fc42e2166c2dfd3`.
- The wrapper's official supervised contract has been confirmed on a legal
  owner sample: prompt/audio prefix masked with `-100`, padding masked, and
  `language Hindi<asr_text><raw transcript>` plus EOS supervised. A real
  finite base loss and a LoRA-injected no-gradient forward passed on the RTX
  3050. No optimizer step or adapter verification has been claimed because an
  owner training manifest was not supplied to this milestone run.
- Laptop LoRA is a bounded project experiment using exact decoder-attention
  Q/V paths, not an official Qwen LoRA recipe. It does not replace the official
  target/collator contract or authorize QLoRA.
- Canonical transcript style: Devanagari Hindi plus Latin-script English.
- Roman Hinglish and inverse text normalization are separate concerns.
- Python modules and command-line jobs own the training implementation.
- The laptop is primarily for qualification; H100 x1 is the first real training target.
- H100 x8 comes after H100 x1.
- The same core code path should serve all hardware and data scales.
- Reports, predictions, checkpoint verification, and graphs are required outputs.
- Runtime dependency installation, unchecked NaN continuation, and unverified checkpoint reuse are prohibited.
- The former foundation-audit bureaucracy is not part of this project.

## Deferred decisions

- Whether longer LoRA runs improve owner-set metrics after the one/five-step
  lifecycle and fresh-process adapter verification are completed.
- Exact distributed launcher and optimization stack for H100 x8.
- Exact MLflow/Azure tracking implementation.
- Streaming-serving architecture and endpointing.
- Synthetic-data, pseudo-labelling, distillation, and dynamic-vocabulary experiments.
- Roman Hinglish direct-ASR experiments.
- Inverse text normalization implementation.
- Industry-specific adapters or checkpoints.

## Unknowns requiring validation

- Whether verified wrapper-LoRA backward plus AdamW fits on the RTX 3050 with
  an owner-approved local training manifest.
- Whether full-parameter official wrapper training fits any useful local slice;
  no such laptop claim is currently intended.
- Exact Azure workspace, compute SKU availability, quota, identity, storage, and cost constraints.
- Actual dataset inventory, file schema, verified audio hours, licences, consent, PII status, and split metadata.
- The initial human-verified evaluation set and acceptance thresholds.
- H100 batch sizes, throughput, memory behavior, and distributed scaling efficiency.
- The precise checkpoint format and supported resume semantics of the selected official stack.

Do not convert an unknown into a confident implementation assumption. Resolve it with primary documentation, inexpensive tests, or owner input when that milestone begins.
