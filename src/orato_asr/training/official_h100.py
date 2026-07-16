"""Bounded single-H100 supervised fine-tuning using Qwen's official contract.

The heavy ML imports are intentionally lazy.  CPU tests can validate the
configuration and deterministic manifest preparation without installing the
training environment.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import yaml

from ..data.manifest import manifest_fingerprint
from ..data.schema import ManifestRecord, lexical_content, parse_record
from ..exceptions import DependencyError, ManifestValidationError, TrainingError
from ..paths import resolver_from_environment
from .config import WRAPPER_MODEL_ID, WRAPPER_MODEL_REVISION
from .official_sft import (
    QWEN_SFT_COMMIT,
    WrapperSample,
    collate_official_single,
    qwen_language_name,
    serialize_official_target,
)
from .reporting import write_atomic_json, write_atomic_jsonl
from .wrapper import release_cuda, resolve_wrapper_snapshot

OFFICIAL_SFT_SHA256 = "6f3e5cc530d9da9ff405144fc94a5c4790018eaf956dbc623634818160418976"
_SUPPORTED_SPLITS = {
    "train": {"train", "training"},
    "validation": {"validation", "valid", "val", "development", "dev"},
}
_CANONICAL_SPLIT_FIELDS = {
    "audio_filepath",
    "text",
    "duration",
    "language",
    "source",
    "speaker_id",
    "recording_id",
    "domain",
    "split",
    "metadata",
}


@dataclass(frozen=True, slots=True)
class OfficialH100Config:
    profile_name: str
    model_id: str
    model_revision: str
    train_max_hours: float
    validation_max_hours: float
    max_audio_seconds: float
    per_device_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    epochs: float
    logging_steps: int
    evaluation_steps: int
    save_steps: int
    save_total_limit: int
    dataloader_num_workers: int
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_official_h100_config(path: str | Path) -> OfficialH100Config:
    """Load the deliberately small, strict single-H100 profile."""

    source = Path(path).expanduser().resolve()
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TrainingError(f"Could not read H100 training config {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrainingError("H100 training config must be a YAML object")
    _require_keys(payload, {"schema_version", "profile", "upstream", "model", "data", "hardware", "training", "checkpointing"}, "config")
    if payload["schema_version"] != 1:
        raise TrainingError("Official H100 config schema_version must be 1")
    profile = _mapping(payload["profile"], "profile")
    upstream = _mapping(payload["upstream"], "upstream")
    model = _mapping(payload["model"], "model")
    data = _mapping(payload["data"], "data")
    hardware = _mapping(payload["hardware"], "hardware")
    training = _mapping(payload["training"], "training")
    checkpointing = _mapping(payload["checkpointing"], "checkpointing")
    if upstream.get("qwen_repository_commit") != QWEN_SFT_COMMIT:
        raise TrainingError("Official Qwen repository commit does not match the qualified SFT contract")
    if upstream.get("sft_script_sha256") != OFFICIAL_SFT_SHA256:
        raise TrainingError("Official Qwen SFT script SHA-256 does not match the qualified source")
    if model.get("id") != WRAPPER_MODEL_ID or model.get("revision") != WRAPPER_MODEL_REVISION:
        raise TrainingError("H100 SFT must use the pinned non-hf wrapper model and revision")
    if hardware != {
        "accelerator": "h100",
        "gpu_count": 1,
        "distributed": False,
        "precision": "bfloat16",
    }:
        raise TrainingError("This milestone requires exactly one H100, BF16, and no distributed launcher")
    config = OfficialH100Config(
        profile_name=_string(profile.get("name"), "profile.name"),
        model_id=WRAPPER_MODEL_ID,
        model_revision=WRAPPER_MODEL_REVISION,
        train_max_hours=_positive_float(data.get("train_max_hours"), "data.train_max_hours"),
        validation_max_hours=_positive_float(data.get("validation_max_hours"), "data.validation_max_hours"),
        max_audio_seconds=_positive_float(data.get("max_audio_seconds"), "data.max_audio_seconds"),
        per_device_batch_size=_positive_int(training.get("per_device_batch_size"), "training.per_device_batch_size"),
        gradient_accumulation_steps=_positive_int(training.get("gradient_accumulation_steps"), "training.gradient_accumulation_steps"),
        learning_rate=_positive_float(training.get("learning_rate"), "training.learning_rate"),
        epochs=_positive_float(training.get("epochs"), "training.epochs"),
        logging_steps=_positive_int(training.get("logging_steps"), "training.logging_steps"),
        evaluation_steps=_positive_int(training.get("evaluation_steps"), "training.evaluation_steps"),
        save_steps=_positive_int(checkpointing.get("save_steps"), "checkpointing.save_steps"),
        save_total_limit=_positive_int(checkpointing.get("keep_last"), "checkpointing.keep_last"),
        dataloader_num_workers=_nonnegative_int(training.get("dataloader_num_workers"), "training.dataloader_num_workers"),
        seed=_nonnegative_int(training.get("seed"), "training.seed"),
    )
    if not 1.0 <= config.train_max_hours <= 100.0:
        raise TrainingError("H100 training profiles must select between 1 and 100 training audio hours")
    if config.max_audio_seconds > 30.0:
        raise TrainingError("H100 smoke clips are capped at 30 seconds")
    if config.per_device_batch_size != 1:
        raise TrainingError("Official prefix masking is qualified here only at batch size 1")
    return config


def prepare_official_jsonl(
    source_manifest: str | Path,
    destination: str | Path,
    *,
    split: str,
    max_hours: float,
    max_audio_seconds: float,
    project_root: str | Path,
) -> dict[str, Any]:
    """Create a deterministic bounded official-SFT JSONL without changing source data."""

    if split not in _SUPPORTED_SPLITS:
        raise TrainingError(f"Unsupported prepared split {split!r}")
    source = Path(source_manifest).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    resolver = resolver_from_environment(project_root=Path(project_root).resolve())
    cap_seconds = _positive_float(max_hours, "max_hours") * 3600.0
    selected_seconds = 0.0
    selected: list[dict[str, Any]] = []
    skipped_long = 0
    skipped_over_cap = 0
    scanned = 0
    accepted_extra_fields: set[str] = set()
    for record, extra_fields in _iter_external_split_manifest(source):
        scanned += 1
        accepted_extra_fields.update(extra_fields)
        if record.split is not None and record.split.strip().casefold() not in _SUPPORTED_SPLITS[split]:
            raise TrainingError(
                f"{source}:{record.line_number}: {split} input contains split {record.split!r}"
            )
        if record.duration is None:
            raise TrainingError(
                f"{source}:{record.line_number}: duration is required for bounded H100 selection"
            )
        duration = float(record.duration)
        if duration > max_audio_seconds:
            skipped_long += 1
            continue
        if selected_seconds + duration > cap_seconds:
            skipped_over_cap += 1
            continue
        audio_path = resolver.resolve(record.audio_filepath)
        if not audio_path.is_file():
            raise TrainingError(f"{source}:{record.line_number}: audio file is missing: {audio_path}")
        language = _training_language(record.language)
        selected.append(
            {
                "audio": str(audio_path),
                "text": serialize_official_target(record.text, language),
                "reference": record.text,
                "language": language,
                "duration": duration,
                "source_line": record.line_number,
            }
        )
        selected_seconds += duration
        if selected_seconds >= cap_seconds - 0.001:
            break
    if not selected:
        raise TrainingError(f"No {split} rows remain after applying the configured bounds")
    write_atomic_jsonl(destination_path, selected, overwrite=True)
    return {
        "source_manifest": str(source),
        "source_manifest_sha256": manifest_fingerprint(source),
        "prepared_manifest": str(destination_path),
        "split": split,
        "scanned_rows": scanned,
        "selected_rows": len(selected),
        "selected_seconds": selected_seconds,
        "selected_hours": selected_seconds / 3600.0,
        "skipped_above_max_audio_seconds": skipped_long,
        "skipped_over_hour_cap": skipped_over_cap,
        "accepted_source_extra_fields": sorted(accepted_extra_fields),
    }


class OfficialSingleSampleCollator:
    """Decode and collate one row with the already-qualified official target mask."""

    def __init__(self, processor: Any, librosa_module: Any) -> None:
        self.processor = processor
        self.librosa = librosa_module

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        if len(rows) != 1:
            raise TrainingError("Official H100 collator requires batch size exactly 1")
        row = rows[0]
        audio, sample_rate = self.librosa.load(str(row["audio"]), sr=16000, mono=True)
        if sample_rate != 16000:
            raise TrainingError(f"Audio decoder returned unexpected sample rate {sample_rate}")
        sample = WrapperSample(
            sample_id=f"line-{int(row['source_line'])}",
            audio=audio,
            duration_seconds=float(row["duration"]),
            transcript=str(row["reference"]),
            language=_training_language(row.get("language")),
            line_number=int(row["source_line"]),
        )
        return collate_official_single(self.processor, sample).inputs


class PreparedJsonlDataset(Sequence[Mapping[str, Any]]):
    """Small in-memory index for the bounded derived JSONL used by Trainer."""

    def __init__(self, path: str | Path) -> None:
        self.rows = _read_jsonl(Path(path).expanduser().resolve())
        if not self.rows:
            raise TrainingError("Prepared H100 dataset contains no rows")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return self.rows[index]


def train_official_h100(args: argparse.Namespace) -> int:
    """Prepare bounded inputs and execute one resumable full-parameter SFT run."""

    config = load_official_h100_config(args.config)
    _require_single_h100_runtime()
    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    prepared_dir = output_dir / "prepared"
    trainer_dir = output_dir / "trainer"
    final_dir = output_dir / "final"
    for path in (prepared_dir, trainer_dir, cache_dir):
        path.mkdir(parents=True, exist_ok=True)
    train_report = prepare_official_jsonl(
        args.train_manifest,
        prepared_dir / "train.jsonl",
        split="train",
        max_hours=config.train_max_hours,
        max_audio_seconds=config.max_audio_seconds,
        project_root=args.project_root,
    )
    validation_report = prepare_official_jsonl(
        args.validation_manifest,
        prepared_dir / "validation.jsonl",
        split="validation",
        max_hours=config.validation_max_hours,
        max_audio_seconds=config.max_audio_seconds,
        project_root=args.project_root,
    )
    write_atomic_json(
        output_dir / "run_contract.json",
        {
            "status": "prepared",
            "config": config.as_dict(),
            "official_qwen_repository_commit": QWEN_SFT_COMMIT,
            "official_sft_script_sha256": OFFICIAL_SFT_SHA256,
            "train": train_report,
            "validation": validation_report,
        },
        overwrite=True,
    )

    torch = _import_required("torch")
    transformers = _import_required("transformers")
    librosa = _import_required("librosa")
    qwen_asr = _import_required("qwen_asr")
    snapshot = resolve_wrapper_snapshot(cache_dir=cache_dir, offline=args.offline)
    started = time.time()
    wrapper = qwen_asr.Qwen3ASRModel.from_pretrained(
        str(snapshot),
        dtype=torch.bfloat16,
        device_map=None,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model = wrapper.model
    from .official_sft import patch_outer_forward

    patch_outer_forward(model)
    model.config.use_cache = False
    model.train()
    train_dataset = PreparedJsonlDataset(prepared_dir / "train.jsonl")
    validation_dataset = PreparedJsonlDataset(prepared_dir / "validation.jsonl")
    training_arguments = transformers.TrainingArguments(
        output_dir=str(trainer_dir),
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        bf16=True,
        tf32=True,
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        logging_nan_inf_filter=False,
        eval_strategy="steps",
        eval_steps=config.evaluation_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=False,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        seed=config.seed,
        data_seed=config.seed,
        report_to="none",
        save_safetensors=True,
    )
    trainer_class = _finite_trainer_class(transformers, torch)
    trainer = trainer_class(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=OfficialSingleSampleCollator(wrapper.processor, librosa),
        processing_class=wrapper.processor.tokenizer,
    )
    checkpoint = _latest_checkpoint(trainer_dir) if args.resume else None
    result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    trainer.save_model(str(final_dir))
    wrapper.processor.save_pretrained(str(final_dir))
    trainer.save_state()
    metrics = dict(result.metrics)
    metrics["wall_seconds"] = time.time() - started
    metrics["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    metrics["peak_cuda_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    write_atomic_json(
        output_dir / "training_summary.json",
        {
            "status": "trained",
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "final_checkpoint": str(final_dir),
            "resumed_from": str(checkpoint) if checkpoint else None,
            "metrics": metrics,
            "versions": _runtime_versions(),
        },
        overwrite=True,
    )
    return 0


def verify_official_checkpoint(args: argparse.Namespace) -> int:
    """Load the saved checkpoint in a fresh process and transcribe fixed rows."""

    _require_single_h100_runtime()
    torch = _import_required("torch")
    qwen_asr = _import_required("qwen_asr")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_dir():
        raise TrainingError(f"Final checkpoint does not exist: {checkpoint}")
    wrapper = qwen_asr.Qwen3ASRModel.from_pretrained(
        str(checkpoint),
        dtype=torch.bfloat16,
        device_map=None,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    wrapper.model = wrapper.model.to("cuda")
    wrapper.device = torch.device("cuda")
    wrapper.dtype = torch.bfloat16
    wrapper.model.eval()
    rows = _read_jsonl(Path(args.manifest).expanduser().resolve())[: args.max_samples]
    if not rows:
        raise TrainingError("Verification manifest contains no rows")
    predictions: list[dict[str, Any]] = []
    for row in rows:
        language = _training_language(row.get("language"))
        results = wrapper.transcribe(
            audio=str(row["audio"]),
            language=None if language is None else qwen_language_name(language),
        )
        if len(results) != 1:
            raise TrainingError("Checkpoint verification returned an unexpected result count")
        prediction = str(getattr(results[0], "text", "") or "").strip()
        if not lexical_content(prediction):
            raise TrainingError("Checkpoint verification produced blank or punctuation-only text")
        predictions.append(
            {
                "audio": row["audio"],
                "reference": row["reference"],
                "prediction": prediction,
                "language": str(getattr(results[0], "language", "") or ""),
            }
        )
    if len(predictions) >= 2 and len({item["prediction"] for item in predictions}) == 1:
        raise TrainingError("Checkpoint verification produced identical text for every sample")
    report_path = Path(args.report).expanduser().resolve()
    write_atomic_json(
        report_path,
        {
            "status": "success",
            "checkpoint": str(checkpoint),
            "fresh_process_pid": os.getpid(),
            "sample_count": len(predictions),
            "predictions": predictions,
            "versions": _runtime_versions(),
        },
        overwrite=True,
    )
    return 0


def run_and_verify(args: argparse.Namespace) -> int:
    """Run training, release its process, then verify via a new Python process."""

    train_args = argparse.Namespace(**vars(args))
    result = train_official_h100(train_args)
    if result:
        return result
    release_cuda(sys.modules.get("torch"))
    command = [
        sys.executable,
        "-m",
        "orato_asr.training.official_h100",
        "verify",
        "--checkpoint",
        str(Path(args.output_dir).expanduser().resolve() / "final"),
        "--manifest",
        str(Path(args.output_dir).expanduser().resolve() / "prepared" / "validation.jsonl"),
        "--report",
        str(Path(args.output_dir).expanduser().resolve() / "verification.json"),
        "--max-samples",
        str(args.verify_samples),
    ]
    return subprocess.run(command, check=False).returncode


def _finite_trainer_class(transformers: Any, torch: Any) -> type:
    class FiniteTrainer(transformers.Trainer):
        def compute_loss(self, model: Any, inputs: Any, return_outputs: bool = False, num_items_in_batch: Any = None) -> Any:
            result = super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
            loss = result[0] if return_outputs else result
            if not bool(torch.isfinite(loss.detach()).all().item()):
                raise TrainingError("Training stopped because loss is NaN or Inf")
            return result

        def training_step(self, model: Any, inputs: Any, num_items_in_batch: Any = None) -> Any:
            loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
            if not bool(torch.isfinite(loss.detach()).all().item()):
                raise TrainingError("Training stopped because step loss is NaN or Inf")
            if bool(self.accelerator.sync_gradients):
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
                        raise TrainingError(f"Training stopped because gradient {name!r} is NaN or Inf")
            return loss

    return FiniteTrainer


def _latest_checkpoint(root: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    if root.is_dir():
        for path in root.iterdir():
            match = re.fullmatch(r"checkpoint-(\d+)", path.name)
            if match and path.is_dir():
                candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _require_single_h100_runtime() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise TrainingError("Today's H100 job is single-process; WORLD_SIZE must be 1")
    torch = _import_required("torch")
    if not bool(torch.cuda.is_available()) or int(torch.cuda.device_count()) != 1:
        raise TrainingError("Official H100 SFT requires exactly one visible CUDA GPU")
    if not bool(torch.cuda.is_bf16_supported()):
        raise TrainingError("Official H100 SFT requires CUDA BF16 support")
    name = str(torch.cuda.get_device_name(0))
    if "H100" not in name.upper():
        raise TrainingError(f"Expected one H100 GPU; CUDA reported {name!r}")


def _training_language(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if normalized.casefold() in {"", "none", "unknown", "und"}:
        return None
    qwen_language_name(normalized)
    return normalized


def _runtime_versions() -> dict[str, str]:
    from importlib import metadata

    values = {"python": platform.python_version()}
    for name in ("torch", "qwen-asr", "transformers", "accelerate", "librosa"):
        try:
            values[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            values[name] = "unavailable"
    return values


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError(f"Could not read prepared manifest {path}: {exc}") from exc


def _iter_external_split_manifest(
    path: Path,
) -> Iterator[tuple[ManifestRecord, tuple[str, ...]]]:
    """Read owner-created split rows while adapting provenance-only fields.

    The source split format may include fields such as ``dataset_folder``,
    ``original_audio_id``, ``original_audio_path``, and ``sample_rate``. They
    are recorded in the preparation report but are not model inputs.
    """

    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TrainingError(
                        f"{path}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc
                if not isinstance(raw, dict):
                    raise TrainingError(f"{path}:{line_number}: split row must be a JSON object")
                canonical = {
                    key: value for key, value in raw.items() if key in _CANONICAL_SPLIT_FIELDS
                }
                extra_fields = tuple(sorted(set(raw) - _CANONICAL_SPLIT_FIELDS))
                try:
                    record = parse_record(
                        canonical,
                        manifest_path=path,
                        line_number=line_number,
                    )
                except ManifestValidationError as exc:
                    raise TrainingError(f"Invalid split row: {exc}") from exc
                yield record, extra_fields
    except OSError as exc:
        raise TrainingError(f"Could not read split manifest {path}: {exc}") from exc


def _import_required(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except (ImportError, OSError) as exc:
        raise DependencyError(f"Required H100 training dependency {name!r} is unavailable") from exc


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TrainingError(f"{label} must be a YAML object")
    return value


def _require_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise TrainingError(f"{label} keys must be exactly: {', '.join(sorted(expected))}")


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingError(f"{label} must be a non-empty string")
    return value.strip()


def _positive_float(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) <= 0:
        raise TrainingError(f"{label} must be positive and finite")
    return float(value)


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise TrainingError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise TrainingError(f"{label} must be a non-negative integer")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Official single-H100 Qwen3-ASR SFT")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Train and fresh-process verify")
    run.add_argument("--config", required=True)
    run.add_argument("--train-manifest", required=True)
    run.add_argument("--validation-manifest", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--cache-dir", required=True)
    run.add_argument("--project-root", default=".")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--offline", action="store_true")
    run.add_argument("--verify-samples", type=int, default=3, choices=range(2, 11))
    run.set_defaults(handler=run_and_verify)
    verify = subparsers.add_parser("verify", help="Fresh-process checkpoint verification")
    verify.add_argument("--checkpoint", required=True)
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--report", required=True)
    verify.add_argument("--max-samples", type=int, default=3, choices=range(2, 11))
    verify.set_defaults(handler=verify_official_checkpoint)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (DependencyError, TrainingError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        try:
            release_cuda(sys.modules.get("torch"))
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OFFICIAL_SFT_SHA256",
    "OfficialH100Config",
    "PreparedJsonlDataset",
    "load_official_h100_config",
    "prepare_official_jsonl",
]
