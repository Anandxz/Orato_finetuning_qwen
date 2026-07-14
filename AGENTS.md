# Agent Instructions

- Read `PROJECT_CONTEXT.md` and `README.md` before making changes.
- Keep each task focused on the requested milestone; do not create unrelated files or features.
- Use current official Qwen3-ASR primary documentation for model, processor, training, and checkpoint behavior. Verify version-sensitive facts before implementation.
- Implement the official supervised fine-tuning path before custom LoRA, processors, targets, masking, or model changes.
- Keep authoritative training logic in Python modules and command-line entry points, not notebooks.
- Never install or upgrade packages from application, training, or notebook code.
- Explain and constrain every new heavy dependency. Do not silently change dependency or model revisions.
- Preserve the mixed-script transcript policy: Devanagari Hindi and Latin-script English. Do not reject valid short utterances by word count alone.
- Never modify raw datasets. Do not commit credentials, private URLs, PII, audio, datasets, model weights, checkpoints, or generated predictions.
- Stop training on NaN or Inf. Verify saved checkpoints from a fresh process and inspect a few predictions before long evaluation.
- Run relevant tests and report the exact results; do not claim success for checks that were not executed.
- Prefer simple, testable modules over speculative abstractions and extra planning documents.
- Do not recreate the abandoned foundation-audit, issue-matrix, or Git-governance system.
