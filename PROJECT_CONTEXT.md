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

The first model target is:

`Qwen/Qwen3-ASR-0.6B`

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

The first working vertical slice should not introduce:

- Custom LoRA or PEFT.
- Automatic LoRA target discovery.
- A custom ASR architecture.
- A custom processor.
- A custom target convention that conflicts with the official data contract.
- Unverified label masking or collation.
- Notebook-only training logic.

LoRA may be investigated after official supervised fine-tuning completes a tiny train-save-fresh-reload-infer lifecycle. It must be a controlled research branch, not a fallback silently selected at runtime.

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

- Initial model: `Qwen/Qwen3-ASR-0.6B`.
- Initial training direction: official supervised fine-tuning path first.
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

- Whether LoRA/PEFT provides a useful later alternative to full supervised fine-tuning.
- Exact distributed launcher and optimization stack for H100 x8.
- Exact MLflow/Azure tracking implementation.
- Streaming-serving architecture and endpointing.
- Synthetic-data, pseudo-labelling, distillation, and dynamic-vocabulary experiments.
- Roman Hinglish direct-ASR experiments.
- Inverse text normalization implementation.
- Industry-specific adapters or checkpoints.

## Unknowns requiring validation

- The current official Qwen3-ASR training API, compatible versions, repository commit, and model revision.
- Whether the full official training step fits or partially fits on the RTX 3050.
- Exact Azure workspace, compute SKU availability, quota, identity, storage, and cost constraints.
- Actual dataset inventory, file schema, verified audio hours, licences, consent, PII status, and split metadata.
- The initial human-verified evaluation set and acceptance thresholds.
- H100 batch sizes, throughput, memory behavior, and distributed scaling efficiency.
- The precise checkpoint format and supported resume semantics of the selected official stack.

Do not convert an unknown into a confident implementation assumption. Resolve it with primary documentation, inexpensive tests, or owner input when that milestone begins.
