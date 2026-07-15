"""Memory-bounded wrapper compatibility, LoRA smoke, and adapter verification."""

from __future__ import annotations

import gc
import importlib
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from ..audio import decode_audio
from ..data.manifest import iter_manifest
from ..data.overlap import check_overlap
from ..data.schema import resolve_local_audio_path
from ..evaluation.metrics import aggregate_predictions, compute_text_metrics
from ..evaluation.normalization import (
    NormalizationOptions,
    is_blank,
    is_punctuation_only,
)
from ..models.qwen3_asr import sanitize_error
from ..exceptions import (
    AdapterVerificationError,
    TrainingError,
    TrainingOOMError,
)
from .config import (
    WRAPPER_BACKEND,
    WRAPPER_MODEL_ID,
    WRAPPER_MODEL_REVISION,
    WrapperTrainingConfig,
)
from .data import LazyWrapperTrainingDataset, PreparedTrainingManifest, prepare_training_manifest
from .lora import (
    capture_representative_frozen_checksums,
    capture_trainable_parameter_copies,
    discover_lora_inventory,
    inject_lora,
    optimizer_trainable_parameters,
    validate_optimizer_parameter_ids,
    verify_frozen_parameter_checksums,
    verify_trainable_parameter_change,
)
from .memory import (
    MemoryGuardConfig,
    build_failure_metadata,
    capture_memory_snapshot,
    enforce_memory_guard,
    gib_to_bytes,
    reset_cuda_memory_tracking,
)
from .official_sft import QWEN_SFT_COMMIT, collate_official_single
from .reporting import (
    append_atomic_jsonl,
    build_training_summary,
    resolve_training_run_directories,
    write_atomic_csv,
    write_atomic_json,
    write_atomic_jsonl,
    write_cto_smoke_summary,
    write_selected_sample_ids,
    write_training_run_readme,
)
from .wrapper import (
    finite_forward,
    load_wrapper_model,
    load_wrapper_processor,
    move_batch_to_cuda,
    release_cuda,
    wrapper_dependency_status,
    wrapper_inference,
)


def inspect_wrapper(config: WrapperTrainingConfig, *, offline: bool = True) -> dict[str, Any]:
    """Load only the wrapper backend and report the verified LoRA topology."""

    values = config.as_dict()
    loaded = load_wrapper_model(
        cache_dir=values["paths"]["model_cache_dir"],
        offline=offline,
        training=True,
    )
    try:
        inventory = discover_lora_inventory(loaded.model)
        return {
            "status": "success",
            "backend": WRAPPER_BACKEND,
            "model": _model_metadata(loaded.snapshot_path),
            "wrapper_class": loaded.wrapper.__class__.__name__,
            "model_class": loaded.model.__class__.__name__,
            "processor_class": loaded.processor.__class__.__name__,
            "dependencies": wrapper_dependency_status(),
            "inventory": inventory.as_dict(),
            "memory": _snapshot("wrapper_inspect", loaded.torch, values).as_dict(),
        }
    finally:
        loaded.close()


def run_wrapper_preflight(
    config: WrapperTrainingConfig,
    *,
    train_manifest: str | Path,
    eval_manifest: str | Path | None,
    offline: bool = True,
) -> dict[str, Any]:
    """Run the no-update wrapper compatibility ladder through base finite loss."""

    values = config.as_dict()
    prepared = _prepare(config, train_manifest)
    overlap = _overlap(config, train_manifest, eval_manifest)
    dataset = LazyWrapperTrainingDataset(prepared, project_root=config.project_root)
    sample = dataset[0]
    stages: dict[str, Any] = {}

    # Stage A is released completely before processor/collator qualification.
    loaded = load_wrapper_model(
        cache_dir=values["paths"]["model_cache_dir"], offline=offline, training=False
    )
    try:
        audio_path = _sample_audio_path(prepared, config.project_root)
        stages["wrapper_inference"] = wrapper_inference(
            loaded, decode_audio(audio_path), language=sample.language
        )
        stages["wrapper_inference"]["load_seconds"] = loaded.load_seconds
    finally:
        loaded.close()

    processor, snapshot = load_wrapper_processor(
        cache_dir=values["paths"]["model_cache_dir"], offline=offline
    )
    collated = collate_official_single(processor, sample)
    stages["official_collator"] = dict(collated.inspection)
    del processor, collated
    gc.collect()

    # Stage C proves a real finite loss before any LoRA adapter exists.
    loaded = load_wrapper_model(
        cache_dir=values["paths"]["model_cache_dir"], offline=offline, training=True
    )
    collated = None
    inputs = None
    try:
        _enforce_guard("base_forward", loaded.torch, values)
        collated = collate_official_single(loaded.processor, sample)
        inputs = move_batch_to_cuda(collated.inputs, loaded.torch)
        forward = finite_forward(loaded.model, inputs, loaded.torch, no_grad=True)
        forward.pop("loss_tensor", None)
        inventory = discover_lora_inventory(loaded.model)
        stages["base_finite_forward"] = forward
        stages["module_inventory"] = {
            "text_layer_count": inventory.text_layer_count,
            "approved_lora_module_paths": list(inventory.approved_module_paths),
            "rejected_qv_candidate_paths": list(inventory.rejected_qv_candidate_paths),
            "audio_encoder_path": inventory.audio_encoder_path,
            "text_decoder_path": inventory.text_decoder_path,
            "embeddings_path": inventory.embeddings_path,
            "output_head_path": inventory.output_head_path,
        }
        memory = _snapshot("base_forward_complete", loaded.torch, values).as_dict()
    finally:
        inputs = None
        collated = None
        loaded.close()

    return {
        "status": "success",
        "decision": "wrapper_0.6b_compatible",
        "model": _model_metadata(snapshot),
        "dependencies": wrapper_dependency_status(),
        "dataset": prepared.as_dict(),
        "overlap": overlap,
        "stages": stages,
        "memory": memory,
        "optimizer_state_allocated": False,
        "backward_fit": "unproven_until_lora-one-step",
    }


def run_lora_training(
    config: WrapperTrainingConfig,
    *,
    train_manifest: str | Path,
    run_name: str,
    optimizer_steps: int | None,
    one_step_mode: bool,
    offline: bool = True,
    allow_without_one_step_evidence: bool = False,
    full_epoch_mode: bool = False,
) -> dict[str, Any]:
    """Run one step, a bounded smoke, or one complete selected-data epoch."""

    if type(one_step_mode) is not bool:
        raise TrainingError("one_step_mode must be true or false")
    if type(full_epoch_mode) is not bool:
        raise TrainingError("full_epoch_mode must be true or false")
    if one_step_mode and full_epoch_mode:
        raise TrainingError("One-step and full-epoch modes are mutually exclusive")
    if one_step_mode and optimizer_steps != 1:
        raise TrainingError("lora-one-step must run exactly one optimizer step")
    if not one_step_mode and not full_epoch_mode and optimizer_steps not in {5, 10}:
        raise TrainingError("lora-smoke permits exactly 5 or 10 optimizer steps")

    values = config.as_dict()
    prepared = _prepare(config, train_manifest, require_training_split=True)
    accumulation_configured = int(values["training"]["gradient_accumulation_steps"])
    if full_epoch_mode:
        if values["runtime"].get("run_kind") != "full_epoch":
            raise TrainingError(
                "lora-full requires a wrapper config with runtime.run_kind=full_epoch"
            )
        optimizer_steps = math.ceil(len(prepared.selected) / accumulation_configured)
    if type(optimizer_steps) is not int or optimizer_steps <= 0:
        raise TrainingError("Optimizer steps must be a positive integer")
    if optimizer_steps > int(values["training"]["max_optimizer_steps"]):
        raise TrainingError("Requested optimizer steps exceed the configured maximum")
    compatibility = _require_compatibility_evidence(
        Path(values["paths"]["reports_root"]),
        manifest_fingerprint=prepared.fingerprint,
    )
    if not one_step_mode and not allow_without_one_step_evidence:
        _require_one_step_evidence(
            Path(values["paths"]["output_root"]),
            manifest_fingerprint=prepared.fingerprint,
            method=values["method"],
        )
    if full_epoch_mode:
        _require_five_step_evidence(
            Path(values["paths"]["output_root"]),
            manifest_fingerprint=prepared.fingerprint,
            method=values["method"],
            gpu_safety_limit_bytes=gib_to_bytes(
                values["memory"]["gpu_safety_limit_gb"]
            ),
        )
    elif not one_step_mode and optimizer_steps == 10:
        _require_five_step_evidence(
            Path(values["paths"]["output_root"]),
            manifest_fingerprint=prepared.fingerprint,
            method=values["method"],
            gpu_safety_limit_bytes=gib_to_bytes(
                values["memory"]["gpu_safety_limit_gb"]
            ),
        )

    directories = resolve_training_run_directories(
        project_root=config.project_root,
        output_root=values["paths"]["output_root"],
        reports_root=values["paths"]["reports_root"],
        run_name=run_name,
        create=True,
    )
    dataset = LazyWrapperTrainingDataset(prepared, project_root=config.project_root)
    write_atomic_json(
        directories.output_directory / "resolved_config.json",
        values,
    )
    write_atomic_json(
        directories.output_directory / "environment.json",
        _environment_payload(),
    )
    write_atomic_json(
        directories.output_directory / "dataset_summary.json", prepared.as_dict()
    )
    write_atomic_json(
        directories.output_directory / "compatibility_report.json", compatibility
    )
    write_selected_sample_ids(
        prepared.selected,
        directories.output_directory / "selected_sample_ids.jsonl",
    )

    memory_path = directories.output_directory / "memory_events.jsonl"
    metrics_path = directories.output_directory / "training_metrics.jsonl"
    failures_path = directories.output_directory / "failures.jsonl"
    write_atomic_jsonl(metrics_path, ())
    write_atomic_jsonl(failures_path, ())
    loaded = None
    model = None
    injection = None
    optimizer = None
    trainable = None
    first = None
    sample = None
    base_batch = None
    base_inputs = None
    base_forward = None
    lora_batch = None
    lora_inputs = None
    backward_batch = None
    backward_inputs = None
    backward_output = None
    backward_loss = None
    batch = None
    inputs = None
    output = None
    loss = None
    adapter_before = None
    started = time.perf_counter()
    consumed_ids: set[str] = set()
    consumed_samples = 0
    consumed_audio = 0.0
    metrics: list[dict[str, Any]] = []
    memory_events: list[dict[str, Any]] = []
    try:
        torch = importlib.import_module("torch")
        seed = int(values["training"]["seed"])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        reset_cuda_memory_tracking(torch)
        _guard_and_record(
            "before_model_load", torch, values, memory_path, memory_events
        )
        loaded = load_wrapper_model(
            cache_dir=values["paths"]["model_cache_dir"],
            offline=offline,
            training=True,
        )
        _guard_and_record(
            "after_model_load", loaded.torch, values, memory_path, memory_events
        )

        # The official base loss is proved again before this run mutates adapter state.
        _guard_and_record(
            "before_base_collate", loaded.torch, values, memory_path, memory_events
        )
        first = dataset[0]
        base_batch = collate_official_single(loaded.processor, first)
        _guard_and_record(
            "after_base_collate", loaded.torch, values, memory_path, memory_events
        )
        base_inputs = move_batch_to_cuda(base_batch.inputs, loaded.torch)
        _guard_and_record(
            "before_base_forward", loaded.torch, values, memory_path, memory_events
        )
        base_forward = finite_forward(loaded.model, base_inputs, loaded.torch, no_grad=True)
        base_loss = float(base_forward["loss"])
        base_forward.pop("loss_tensor", None)
        _guard_and_record(
            "after_base_forward", loaded.torch, values, memory_path, memory_events
        )
        del base_inputs, base_batch
        loaded.torch.cuda.empty_cache()

        method = values["method"]
        injection = inject_lora(
            loaded.model,
            rank=method["rank"],
            alpha=method["alpha"],
            dropout=method["dropout"],
            bias=method["bias"],
            gradient_checkpointing=values["training"]["gradient_checkpointing"],
        )
        model = injection.model
        loaded.model = model
        loaded.wrapper.model = model
        write_atomic_json(
            directories.output_directory / "module_inventory.json",
            injection.inventory.as_dict(),
        )
        write_atomic_json(
            directories.output_directory / "trainable_parameters.json",
            injection.as_dict(),
        )
        _guard_and_record(
            "after_lora_injection", loaded.torch, values, memory_path, memory_events
        )

        trainable = optimizer_trainable_parameters(model)
        # Stage D: prove the injected LoRA model has a finite official loss
        # before gradients or optimizer state exist.
        _guard_and_record(
            "stage_d_before_collate", loaded.torch, values, memory_path, memory_events
        )
        lora_batch = collate_official_single(loaded.processor, first)
        _guard_and_record(
            "stage_d_after_collate", loaded.torch, values, memory_path, memory_events
        )
        lora_inputs = move_batch_to_cuda(lora_batch.inputs, loaded.torch)
        _guard_and_record(
            "stage_d_before_forward", loaded.torch, values, memory_path, memory_events
        )
        lora_forward = finite_forward(model, lora_inputs, loaded.torch, no_grad=True)
        lora_forward.pop("loss_tensor", None)
        _guard_and_record(
            "stage_d_after_forward", loaded.torch, values, memory_path, memory_events
        )
        del lora_inputs, lora_batch
        loaded.torch.cuda.empty_cache()

        # Stage E: backward must fit and produce finite adapter gradients before
        # AdamW is constructed.  This keeps optimizer state out of the memory
        # measurement and makes the ladder ordering mechanically verifiable.
        model.zero_grad(set_to_none=True)
        backward_batch = collate_official_single(loaded.processor, first)
        backward_inputs = move_batch_to_cuda(backward_batch.inputs, loaded.torch)
        _guard_and_record(
            "stage_e_before_backward", loaded.torch, values, memory_path, memory_events
        )
        try:
            backward_output = model(**backward_inputs)
            backward_loss = getattr(backward_output, "loss", None)
            if backward_loss is None:
                raise TrainingError("LoRA backward qualification returned no loss")
            backward_loss_value = float(backward_loss.detach().float().item())
            if not math.isfinite(backward_loss_value):
                raise TrainingError(
                    f"Non-finite Stage E LoRA loss {backward_loss_value!r}"
                )
            backward_loss.backward()
        except Exception as exc:
            if _is_cuda_oom(exc, loaded.torch):
                raise TrainingOOMError(
                    "CUDA OOM during Stage E LoRA backward; retry only in a fresh process"
                ) from exc
            raise
        _require_finite_gradients(model, loaded.torch, "Stage E")
        backward_gradient_norm = _unclipped_gradient_norm(trainable, loaded.torch)
        _guard_and_record(
            "stage_e_after_backward", loaded.torch, values, memory_path, memory_events
        )
        model.zero_grad(set_to_none=True)
        del backward_inputs, backward_batch, backward_output, backward_loss
        loaded.torch.cuda.empty_cache()
        _guard_and_record(
            "stage_e_after_gradient_clear",
            loaded.torch,
            values,
            memory_path,
            memory_events,
        )
        ladder = {
            "stage_d_lora_finite_forward": {
                **lora_forward,
                "optimizer_state_allocated": False,
            },
            "stage_e_lora_backward_without_optimizer": {
                "loss": backward_loss_value,
                "gradient_norm": backward_gradient_norm,
                "finite_gradients": True,
                "optimizer_state_allocated": False,
            },
        }
        write_atomic_json(
            directories.output_directory / "lora_memory_ladder.json", ladder
        )

        frozen_before = capture_representative_frozen_checksums(model)
        adapter_before = capture_trainable_parameter_copies(model)
        _guard_and_record(
            "before_optimizer_creation",
            loaded.torch,
            values,
            memory_path,
            memory_events,
        )
        optimizer = loaded.torch.optim.AdamW(
            trainable,
            lr=float(values["training"]["learning_rate"]),
            weight_decay=float(values["training"]["weight_decay"]),
        )
        optimizer_audit = validate_optimizer_parameter_ids(optimizer, model)
        _guard_and_record(
            "after_optimizer_creation",
            loaded.torch,
            values,
            memory_path,
            memory_events,
        )

        accumulation = 1 if one_step_mode else accumulation_configured
        target_microsteps = (
            len(dataset) if full_epoch_mode else optimizer_steps * accumulation
        )
        microstep = 0
        optimizer.zero_grad(set_to_none=True)
        optimization_started = time.perf_counter()
        for step in range(1, optimizer_steps + 1):
            _guard_and_record(
                f"optimizer_step_{step}_start",
                loaded.torch,
                values,
                memory_path,
                memory_events,
            )
            step_losses: list[float] = []
            step_started = time.perf_counter()
            step_sample_ids: list[str] = []
            step_accumulation = min(accumulation, target_microsteps - microstep)
            for _ in range(step_accumulation):
                sample = dataset[microstep] if full_epoch_mode else dataset[microstep % len(dataset)]
                batch = collate_official_single(loaded.processor, sample)
                inputs = move_batch_to_cuda(batch.inputs, loaded.torch)
                _guard_and_record(
                    f"optimizer_step_{step}_microstep_{microstep + 1}_before_backward",
                    loaded.torch,
                    values,
                    memory_path,
                    memory_events,
                )
                try:
                    output = model(**inputs)
                    loss = getattr(output, "loss", None)
                    if loss is None:
                        raise TrainingError("LoRA model returned no training loss")
                    loss_value = float(loss.detach().float().item())
                    if not math.isfinite(loss_value):
                        raise TrainingError(f"Non-finite training loss at microstep {microstep + 1}")
                    (loss / step_accumulation).backward()
                except Exception as exc:
                    if _is_cuda_oom(exc, loaded.torch):
                        raise TrainingOOMError(
                            f"CUDA OOM during backward at optimizer step {step}; retry in a fresh process"
                        ) from exc
                    raise
                step_losses.append(loss_value)
                step_sample_ids.append(sample.sample_id)
                consumed_ids.add(sample.sample_id)
                consumed_samples += 1
                consumed_audio += sample.duration_seconds
                microstep += 1
                _guard_and_record(
                    f"optimizer_step_{step}_microstep_{microstep}_after_backward",
                    loaded.torch,
                    values,
                    memory_path,
                    memory_events,
                )
                del inputs, batch, output, loss

            _require_finite_gradients(model, loaded.torch, step)
            gradient_norm = loaded.torch.nn.utils.clip_grad_norm_(
                trainable, float(values["training"]["max_grad_norm"])
            )
            gradient_norm_value = float(gradient_norm.detach().float().item())
            if not math.isfinite(gradient_norm_value):
                raise TrainingError(f"Non-finite gradient norm at optimizer step {step}")
            _guard_and_record(
                f"optimizer_step_{step}_before_update",
                loaded.torch,
                values,
                memory_path,
                memory_events,
            )
            optimizer.step()
            _require_finite_trainable_parameters(model, loaded.torch, step)
            optimizer.zero_grad(set_to_none=True)
            loaded.torch.cuda.synchronize()
            _guard_and_record(
                f"optimizer_step_{step}_after_update",
                loaded.torch,
                values,
                memory_path,
                memory_events,
            )
            event = {
                "optimizer_step": step,
                "microsteps_completed": microstep,
                "loss": sum(step_losses) / len(step_losses),
                "gradient_norm_before_clip": gradient_norm_value,
                "learning_rate": float(values["training"]["learning_rate"]),
                "sample_ids": step_sample_ids,
                "step_seconds": time.perf_counter() - step_started,
                "cuda_allocated_bytes": int(loaded.torch.cuda.memory_allocated()),
                "cuda_reserved_bytes": int(loaded.torch.cuda.memory_reserved()),
                "peak_cuda_allocated_bytes": int(loaded.torch.cuda.max_memory_allocated()),
                "peak_cuda_reserved_bytes": int(loaded.torch.cuda.max_memory_reserved()),
            }
            append_atomic_jsonl(metrics_path, event)
            metrics.append(event)
        optimization_runtime = time.perf_counter() - optimization_started

        verify_trainable_parameter_change(
            model,
            adapter_before,
            tensor_equal=loaded.torch.equal,
        )
        verify_frozen_parameter_checksums(model, frozen_before)
        model.save_pretrained(str(directories.adapter_directory), safe_serialization=True)
        adapter_metadata = {
            "format": "peft_adapter_only",
            "base_model_id": WRAPPER_MODEL_ID,
            "base_model_revision": WRAPPER_MODEL_REVISION,
            "backend": WRAPPER_BACKEND,
            "qwen_sft_commit": QWEN_SFT_COMMIT,
            "approved_module_paths": list(injection.inventory.approved_module_paths),
            "rank": method["rank"],
            "alpha": method["alpha"],
            "dropout": method["dropout"],
            "optimizer_steps": optimizer_steps,
            "full_epochs": 1 if full_epoch_mode else 0,
            "manifest_sha256": prepared.fingerprint,
        }
        write_atomic_json(
            directories.adapter_directory / "orato_adapter_metadata.json",
            adapter_metadata,
        )
        _guard_and_record(
            "training_complete", loaded.torch, values, memory_path, memory_events
        )
        runtime = time.perf_counter() - started
        summary = build_training_summary(
            status=(
                "one_step_completed"
                if one_step_mode
                else "full_epoch_completed"
                if full_epoch_mode
                else "smoke_completed"
            ),
            total_manifest_samples=prepared.total_samples,
            total_manifest_duration_seconds=prepared.total_duration_seconds,
            eligible_samples=prepared.eligible_samples,
            eligible_duration_seconds=prepared.eligible_duration_seconds,
            selected_samples=len(prepared.selected),
            selected_duration_seconds=prepared.selected_duration_seconds,
            consumed_samples=consumed_samples,
            unique_consumed_samples=len(consumed_ids),
            consumed_audio_seconds=consumed_audio,
            microsteps=microstep,
            optimizer_steps=optimizer_steps,
            per_device_batch_size=1,
            gradient_accumulation_steps=accumulation,
            runtime_seconds=runtime,
            complete_epoch_performed=len(consumed_ids) >= prepared.eligible_samples,
            epoch_estimate_runtime_seconds=optimization_runtime,
        )
        peak_allocated, peak_reserved, peak_system_used = _peak_memory_values(
            memory_events
        )
        result = {
            **summary,
            "model": _model_metadata(loaded.snapshot_path),
            "dataset_identity": {"manifest_sha256": prepared.fingerprint},
            "method": {
                "rank": method["rank"],
                "alpha": method["alpha"],
                "dropout": method["dropout"],
                "target_scope": method["target_scope"],
            },
            "dependencies": wrapper_dependency_status(),
            "base_finite_loss": base_loss,
            "memory_ladder": ladder,
            "training": {
                "initial_loss": metrics[0]["loss"],
                "final_loss": metrics[-1]["loss"],
                "gradient_norms": [event["gradient_norm_before_clip"] for event in metrics],
                "peak_cuda_allocated_bytes": peak_allocated,
                "peak_cuda_reserved_bytes": peak_reserved,
                "peak_system_ram_used_bytes": peak_system_used,
                "optimizer": "AdamW",
                "seed": seed,
                "optimizer_parameter_audit": optimizer_audit.as_dict(),
            },
            "lora": injection.as_dict(),
            "adapter": {
                "saved": True,
                "path": str(directories.adapter_directory),
                "fresh_process_reload": False,
            },
            "verification": "pending_explicit_fresh_process_command",
        }
        write_atomic_csv(
            directories.report_directory / "metrics.csv",
            metrics,
            fieldnames=(
                "optimizer_step",
                "microsteps_completed",
                "loss",
                "gradient_norm_before_clip",
                "learning_rate",
                "sample_ids",
                "step_seconds",
                "cuda_allocated_bytes",
                "cuda_reserved_bytes",
                "peak_cuda_allocated_bytes",
                "peak_cuda_reserved_bytes",
            ),
        )
        memory_csv_rows = _memory_csv_rows(memory_events)
        write_atomic_csv(
            directories.report_directory / "memory.csv",
            memory_csv_rows,
            fieldnames=(
                "stage",
                "captured_at_utc",
                "system_total_bytes",
                "system_available_bytes",
                "system_used_bytes",
                "cuda_allocated_bytes",
                "cuda_reserved_bytes",
                "cuda_peak_allocated_bytes",
                "cuda_peak_reserved_bytes",
                "cuda_process_check_status",
                "other_cuda_processes",
            ),
        )
        facts = {
            "machine": _machine_payload(),
            "model": _model_metadata(loaded.snapshot_path),
            "dependencies": {k: v["installed"] for k, v in wrapper_dependency_status().items()},
            "lora": {
                "rank": method["rank"],
                "alpha": method["alpha"],
                "dropout": method["dropout"],
                "scope": method["target_scope"],
            },
            "trainable_parameters": injection.audit.trainable_parameter_count,
            "trainable_percentage": injection.audit.trainable_percentage,
            "losses": {"initial": metrics[0]["loss"], "final": metrics[-1]["loss"]},
            "gradient_norms": [
                event["gradient_norm_before_clip"] for event in metrics
            ],
            "memory": {
                "peak_cuda_allocated_bytes": peak_allocated,
                "peak_cuda_reserved_bytes": peak_reserved,
                "peak_system_used_bytes": peak_system_used,
            },
            "adapter": {
                "saved": True,
                "reloaded": False,
                "path": str(directories.adapter_directory),
            },
            "known_limitations": [
                "Fresh-process adapter verification remains a separate required command.",
                "LoRA is a controlled project experiment, not an officially documented Qwen LoRA recipe.",
                *(
                    []
                    if full_epoch_mode
                    else ["A bounded smoke run does not establish full-dataset coverage."]
                ),
            ],
        }
        result["report_facts"] = facts
        # Write summaries only after all report facts are assembled.
        write_atomic_json(
            directories.output_directory / "run_summary.json",
            result,
            overwrite=True,
        )
        write_atomic_json(
            directories.report_directory / "summary.json",
            result,
            overwrite=True,
        )
        write_cto_smoke_summary(
            summary,
            facts,
            directories.report_directory / "CTO_SMOKE_SUMMARY.md",
        )
        write_training_run_readme(
            summary,
            facts,
            directories.report_directory / "README.md",
        )
        if one_step_mode:
            write_atomic_json(
                directories.output_directory / "one_step_evidence.json",
                {
                    "status": "success",
                    "optimizer_steps": 1,
                    "adapter_saved": True,
                    "manifest_sha256": prepared.fingerprint,
                    "model_id": WRAPPER_MODEL_ID,
                    "model_revision": WRAPPER_MODEL_REVISION,
                    "method": {
                        "rank": method["rank"],
                        "alpha": method["alpha"],
                        "dropout": method["dropout"],
                        "target_scope": method["target_scope"],
                    },
                    "stage_e_backward_without_optimizer": True,
                },
            )
        return result
    except Exception as exc:
        snapshot = None
        torch = getattr(loaded, "torch", None) if loaded is not None else None
        if torch is not None:
            snapshot = _snapshot("failure", torch, values)
        failure = build_failure_metadata(
            exc,
            stage=(
                "lora_one_step"
                if one_step_mode
                else "lora_full_epoch"
                if full_epoch_mode
                else "lora_smoke"
            ),
            snapshot=snapshot,
            torch_module=torch,
        )
        append_atomic_jsonl(failures_path, failure)
        if isinstance(exc, TrainingError):
            raise
        raise TrainingError(
            f"Unexpected LoRA training failure: {sanitize_error(exc)}"
        ) from exc
    finally:
        torch_module = getattr(loaded, "torch", None) if loaded is not None else None
        optimizer = None
        trainable = None
        adapter_before = None
        injection = None
        model = None
        first = None
        sample = None
        base_batch = None
        base_inputs = None
        base_forward = None
        lora_batch = None
        lora_inputs = None
        backward_batch = None
        backward_inputs = None
        backward_output = None
        backward_loss = None
        batch = None
        inputs = None
        output = None
        loss = None
        if loaded is not None:
            loaded.close()
            loaded = None
        gc.collect()
        release_cuda(torch_module)


def verify_adapter(
    config: WrapperTrainingConfig,
    *,
    run_directory: str | Path,
    eval_manifest: str | Path,
    max_samples: int,
    offline: bool = True,
) -> dict[str, Any]:
    """Fresh-process base/adapter reload and deterministic prediction comparison."""

    if type(max_samples) is not int or max_samples < 1:
        raise AdapterVerificationError("Adapter verification max_samples must be positive")
    values = config.as_dict()
    run_dir = Path(run_directory).expanduser().resolve()
    output_root = Path(values["paths"]["output_root"]).resolve()
    try:
        run_dir.relative_to(output_root)
    except ValueError as exc:
        raise AdapterVerificationError(
            f"Run directory must be beneath {output_root}"
        ) from exc
    adapter_dir = run_dir / "adapter"
    required = (
        adapter_dir / "adapter_config.json",
        adapter_dir / "adapter_model.safetensors",
        adapter_dir / "orato_adapter_metadata.json",
    )
    for path in required:
        if not path.is_file() or path.stat().st_size <= 0:
            raise AdapterVerificationError(f"Adapter file is missing or empty: {path}")
    for forbidden in (adapter_dir / "model.safetensors", adapter_dir / "pytorch_model.bin"):
        if forbidden.exists():
            raise AdapterVerificationError(
                f"Adapter directory unexpectedly contains a full-model weight file: {forbidden}"
            )
    metadata = _read_json_object(required[2], "adapter metadata")
    adapter_config = _read_json_object(required[0], "PEFT adapter configuration")
    if (
        metadata.get("base_model_id") != WRAPPER_MODEL_ID
        or metadata.get("base_model_revision") != WRAPPER_MODEL_REVISION
        or metadata.get("backend") != WRAPPER_BACKEND
        or metadata.get("qwen_sft_commit") != QWEN_SFT_COMMIT
        or type(metadata.get("optimizer_steps")) is not int
        or metadata["optimizer_steps"] <= 0
    ):
        raise AdapterVerificationError("Adapter metadata does not match the pinned wrapper base")
    approved = metadata.get("approved_module_paths")
    configured_targets = adapter_config.get("target_modules")
    if (
        not isinstance(approved, list)
        or not approved
        or not all(isinstance(value, str) for value in approved)
        or not isinstance(configured_targets, list)
        or set(configured_targets) != set(approved)
        or adapter_config.get("r") != metadata.get("rank")
        or adapter_config.get("lora_alpha") != metadata.get("alpha")
        or adapter_config.get("lora_dropout") != metadata.get("dropout")
        or adapter_config.get("bias") != "none"
        or adapter_config.get("task_type") != "CAUSAL_LM"
    ):
        raise AdapterVerificationError(
            "PEFT adapter configuration does not match the recorded exact LoRA allowlist"
        )

    run_summary_path = run_dir / "run_summary.json"
    run_summary = _read_json_object(run_summary_path, "training run summary")
    if run_summary.get("status") not in {
        "one_step_completed",
        "smoke_completed",
        "full_epoch_completed",
    }:
        raise AdapterVerificationError(
            "Adapter verification requires a completed one-step or smoke run summary"
        )
    training_environment = _read_json_object(
        run_dir / "environment.json", "training environment"
    )
    if training_environment.get("process_id") == os.getpid():
        raise AdapterVerificationError(
            "Adapter verification must run in a fresh process after training exits"
        )

    records: list[tuple[Any, Path, float]] = []
    for record in iter_manifest(eval_manifest):
        path = resolve_local_audio_path(record, config.project_root)
        if path is None:
            raise AdapterVerificationError("Adapter verification requires local evaluation audio")
        try:
            audio = decode_audio(path)
        except Exception as exc:
            raise AdapterVerificationError(
                f"Could not validate local evaluation audio at manifest line "
                f"{record.line_number}: {exc}"
            ) from exc
        records.append((record, path, float(audio.duration_seconds)))
        del audio
        if len(records) >= max_samples:
            break
    if not records:
        raise AdapterVerificationError("Evaluation manifest contains no records")

    base_results: list[dict[str, Any]] = []
    loaded = load_wrapper_model(
        cache_dir=values["paths"]["model_cache_dir"], offline=offline, training=False
    )
    try:
        for record, path, _duration in records:
            audio = decode_audio(path)
            try:
                base_results.append(
                    wrapper_inference(loaded, audio, language=record.language)
                )
            finally:
                del audio
    finally:
        loaded.close()

    adapter_results: list[dict[str, Any]] = []
    loaded = load_wrapper_model(
        cache_dir=values["paths"]["model_cache_dir"], offline=offline, training=True
    )
    adapter_model = None
    try:
        peft = importlib.import_module("peft")
        adapter_model = peft.PeftModel.from_pretrained(
            loaded.model, str(adapter_dir), is_trainable=False
        ).to("cuda")
        adapter_model.eval()
        loaded.model = adapter_model
        loaded.wrapper.model = adapter_model
        for record, path, _duration in records:
            audio = decode_audio(path)
            try:
                adapter_results.append(
                    wrapper_inference(loaded, audio, language=record.language)
                )
            finally:
                del audio
    except Exception as exc:
        if isinstance(exc, AdapterVerificationError):
            raise
        raise AdapterVerificationError(f"Could not load or infer with adapter: {exc}") from exc
    finally:
        adapter_model = None
        loaded.close()

    adapter_predictions = [str(result["transcript"]) for result in adapter_results]
    if any(is_blank(value) or is_punctuation_only(value) for value in adapter_predictions):
        raise AdapterVerificationError("Adapter produced blank or punctuation-only output")
    if len(adapter_predictions) > 1 and len(set(adapter_predictions)) == 1:
        raise AdapterVerificationError("Adapter produced identical-output collapse")
    options = NormalizationOptions()
    rows = []
    base_metric_rows: list[dict[str, Any]] = []
    adapter_metric_rows: list[dict[str, Any]] = []
    for (record, _path, duration), base_result, adapter_result in zip(
        records, base_results, adapter_results
    ):
        base = str(base_result["transcript"])
        adapter = str(adapter_result["transcript"])
        base_seconds = float(base_result.get("inference_seconds") or 0.0)
        adapter_seconds = float(adapter_result.get("inference_seconds") or 0.0)
        base_metrics = compute_text_metrics(record.text, base, options=options)
        adapter_metrics = compute_text_metrics(record.text, adapter, options=options)
        category = record.metadata.get("eval_category")
        rows.append(
            {
                "manifest_line": record.line_number,
                "audio_filepath": record.audio_filepath,
                "audio_duration_seconds": duration,
                "category": category,
                "reference": record.text,
                "base_prediction": base,
                "adapter_prediction": adapter,
                "base_inference_seconds": base_seconds,
                "adapter_inference_seconds": adapter_seconds,
                "base_real_time_factor": base_seconds / duration if duration else None,
                "adapter_real_time_factor": adapter_seconds / duration if duration else None,
                "base_metrics": base_metrics,
                "adapter_metrics": adapter_metrics,
            }
        )
        base_metric_rows.append(
            {
                "status": "success",
                **base_metrics,
                "audio_duration_seconds": duration,
                "inference_seconds": base_seconds,
                "real_time_factor": base_seconds / duration if duration else None,
            }
        )
        adapter_metric_rows.append(
            {
                "status": "success",
                **adapter_metrics,
                "audio_duration_seconds": duration,
                "inference_seconds": adapter_seconds,
                "real_time_factor": adapter_seconds / duration if duration else None,
            }
        )
    base_aggregate = aggregate_predictions(
        base_metric_rows
    )
    adapter_aggregate = aggregate_predictions(
        adapter_metric_rows
    )
    categories = sorted(
        {str(row["category"]) for row in rows if row.get("category") is not None}
    )
    category_metrics: dict[str, Any] = {}
    for category in categories:
        indexes = [
            index for index, row in enumerate(rows) if str(row.get("category")) == category
        ]
        category_metrics[category] = {
            "base": aggregate_predictions([base_metric_rows[index] for index in indexes]),
            "adapter": aggregate_predictions(
                [adapter_metric_rows[index] for index in indexes]
            ),
        }
    report = {
        "status": "success",
        "fresh_process_pid": os.getpid(),
        "model": _model_metadata(None),
        "adapter_path": str(adapter_dir),
        "samples": rows,
        "aggregate_metrics": {
            "base": base_aggregate,
            "adapter": adapter_aggregate,
        },
        "category_metrics": category_metrics,
        "non_empty_predictions": True,
        "punctuation_only_collapse": False,
        "identical_output_collapse": False,
    }
    verification_dir = run_dir / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)
    write_atomic_json(verification_dir / "adapter_verification.json", report, overwrite=True)
    report_dir = Path(values["paths"]["reports_root"]) / run_dir.name
    report_dir.mkdir(parents=True, exist_ok=True)
    write_atomic_json(report_dir / "base_vs_adapter.json", report, overwrite=True)

    adapter_summary = run_summary.get("adapter")
    if not isinstance(adapter_summary, dict):
        adapter_summary = {}
        run_summary["adapter"] = adapter_summary
    adapter_summary["fresh_process_reload"] = True
    run_summary["verification"] = {
        "status": "success",
        "fresh_process_pid": os.getpid(),
        "samples": len(rows),
        "aggregate_metrics": report["aggregate_metrics"],
        "report_path": str(report_dir / "base_vs_adapter.json"),
    }
    facts = run_summary.get("report_facts")
    if not isinstance(facts, dict):
        facts = {}
        run_summary["report_facts"] = facts
    fact_adapter = facts.get("adapter")
    if not isinstance(fact_adapter, dict):
        fact_adapter = {}
        facts["adapter"] = fact_adapter
    fact_adapter.update(
        {"saved": True, "reloaded": True, "path": str(adapter_dir)}
    )
    facts["predictions"] = {
        "base": str(base_results[0]["transcript"]),
        "adapter": adapter_predictions[0],
    }
    facts["metrics"] = report["aggregate_metrics"]
    limitations = facts.get("known_limitations")
    if isinstance(limitations, list):
        facts["known_limitations"] = [
            value
            for value in limitations
            if not str(value).startswith("Fresh-process adapter verification remains")
        ]
    write_atomic_json(run_summary_path, run_summary, overwrite=True)
    write_atomic_json(report_dir / "summary.json", run_summary, overwrite=True)
    write_cto_smoke_summary(
        run_summary,
        facts,
        report_dir / "CTO_SMOKE_SUMMARY.md",
        overwrite=True,
    )
    write_training_run_readme(
        run_summary,
        facts,
        report_dir / "README.md",
        overwrite=True,
    )
    return report


def _prepare(
    config: WrapperTrainingConfig,
    manifest: str | Path,
    *,
    require_training_split: bool = False,
) -> PreparedTrainingManifest:
    data = config.as_dict()["data"]
    return prepare_training_manifest(
        manifest,
        project_root=config.project_root,
        minimum_duration_seconds=data["min_audio_seconds"],
        maximum_duration_seconds=data["max_audio_seconds"],
        max_samples=data["max_samples"],
        max_hours=data["max_hours"],
        require_training_split=require_training_split,
    )


def _overlap(
    config: WrapperTrainingConfig,
    train_manifest: str | Path,
    eval_manifest: str | Path | None,
) -> dict[str, Any] | None:
    if eval_manifest is None:
        return None
    report = check_overlap(
        train_manifest,
        eval_manifest,
        project_root=config.project_root,
        hash_local_audio=True,
    )
    if report.prohibited_count:
        raise TrainingError(
            f"Train/evaluation overlap contains {report.prohibited_count} prohibited finding(s)"
        )
    return report.as_dict()


def _sample_audio_path(prepared: PreparedTrainingManifest, project_root: Path) -> Path:
    ref = prepared.selected[0]
    # The lazy dataset already proved this offset. Recover through its public sample
    # only for decoded arrays; inference requires the original local path.
    for record in iter_manifest(prepared.manifest_path):
        if record.line_number == ref.line_number:
            path = resolve_local_audio_path(record, project_root)
            if path is None:
                break
            return path
    raise TrainingError(f"Could not recover local audio for selected sample {ref.sample_id}")


def _guard_config(values: Mapping[str, Any]) -> MemoryGuardConfig:
    memory = values["memory"]
    return MemoryGuardConfig(
        minimum_available_system_bytes=gib_to_bytes(
            memory["minimum_available_system_ram_gb"]
        ),
        gpu_safety_limit_bytes=gib_to_bytes(memory["gpu_safety_limit_gb"]),
        abort_on_threshold=memory["abort_on_threshold"],
        reject_other_large_cuda_processes=True,
        require_cuda_process_check=True,
    )


def _snapshot(stage: str, torch: Any, values: Mapping[str, Any]) -> Any:
    return capture_memory_snapshot(
        stage,
        torch_module=torch,
        capture_system_ram=values["memory"]["capture_system_ram"],
        cuda_process_detector=_cuda_processes,
    )


def _enforce_guard(stage: str, torch: Any, values: Mapping[str, Any]) -> Any:
    snapshot = _snapshot(stage, torch, values)
    enforce_memory_guard(snapshot, _guard_config(values))
    return snapshot


def _guard_and_record(
    stage: str,
    torch: Any,
    values: Mapping[str, Any],
    path: Path,
    events: list[dict[str, Any]] | None = None,
) -> Any:
    snapshot = _enforce_guard(stage, torch, values)
    payload = snapshot.as_dict()
    append_atomic_jsonl(path, payload)
    if events is not None:
        events.append(payload)
    return snapshot


def _cuda_processes(device_index: int, current_pid: int) -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
                f"--id={device_index}",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        raise OSError("Could not execute nvidia-smi CUDA process check") from None
    if result.returncode != 0:
        raise OSError("nvidia-smi CUDA process check returned a non-zero status")
    processes = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            pid, memory_mib = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if pid != current_pid:
            processes.append(
                {"pid": pid, "used_memory_bytes": memory_mib * 1024 * 1024}
            )
    return processes


def _require_finite_gradients(model: Any, torch: Any, step: int | str) -> None:
    stage = f"optimizer step {step}" if type(step) is int else str(step)
    found = False
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        found = True
        if not bool(torch.isfinite(parameter.grad).all().item()):
            raise TrainingError(f"Non-finite gradient in {name} at {stage}")
    if not found:
        raise TrainingError(f"No LoRA gradients were produced at {stage}")


def _unclipped_gradient_norm(parameters: tuple[Any, ...], torch: Any) -> float:
    norm = torch.nn.utils.clip_grad_norm_(parameters, float("inf"))
    value = float(norm.detach().float().item())
    if not math.isfinite(value):
        raise TrainingError("Stage E produced a non-finite LoRA gradient norm")
    return value


def _require_finite_trainable_parameters(model: Any, torch: Any, step: int) -> None:
    found = False
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        found = True
        if not bool(torch.isfinite(parameter).all().item()):
            raise TrainingError(
                f"Non-finite LoRA parameter {name} after optimizer step {step}"
            )
    if not found:
        raise TrainingError("No trainable LoRA parameters remain after optimizer step")


def _require_compatibility_evidence(
    report_root: Path,
    *,
    manifest_fingerprint: str,
) -> dict[str, Any]:
    path = report_root / "wrapper_0.6b_compatibility.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TrainingError(
            "A successful wrapper-preflight report for this training manifest is "
            f"required before LoRA backward: {path}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError(f"Could not read wrapper-preflight evidence {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrainingError("Wrapper-preflight evidence must be a JSON object")
    model = payload.get("model")
    dataset = payload.get("dataset")
    stages = payload.get("stages")
    if not isinstance(model, dict) or not isinstance(dataset, dict) or not isinstance(stages, dict):
        raise TrainingError("Wrapper-preflight evidence is missing model, dataset, or stage metadata")
    collator = stages.get("official_collator")
    forward = stages.get("base_finite_forward")
    inference = stages.get("wrapper_inference")
    if not isinstance(collator, dict) or not isinstance(forward, dict) or not isinstance(inference, dict):
        raise TrainingError("Wrapper-preflight evidence is missing a mandatory compatibility stage")
    loss = forward.get("loss")
    valid = (
        payload.get("status") == "success"
        and payload.get("decision") == "wrapper_0.6b_compatible"
        and model.get("id") == WRAPPER_MODEL_ID
        and model.get("revision") == WRAPPER_MODEL_REVISION
        and model.get("backend") == WRAPPER_BACKEND
        and dataset.get("manifest_sha256") == manifest_fingerprint
        and collator.get("prefix_fully_masked") is True
        and collator.get("padding_fully_masked") is True
        and collator.get("labels_match_target_token_ids") is True
        and collator.get("decoded_supervised_target_matches") is True
        and type(collator.get("supervised_label_tokens")) is int
        and collator["supervised_label_tokens"] > 0
        and type(loss) in (int, float)
        and math.isfinite(float(loss))
        and isinstance(inference.get("transcript"), str)
        and bool(inference["transcript"].strip())
    )
    if not valid:
        raise TrainingError(
            "Wrapper-preflight evidence does not prove the exact model, manifest, "
            "target/masking contract, inference, and finite base loss"
        )
    return payload


def _require_one_step_evidence(
    output_root: Path,
    *,
    manifest_fingerprint: str,
    method: Mapping[str, Any],
) -> None:
    candidates = sorted(output_root.glob("*/one_step_evidence.json"))
    for path in reversed(candidates):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        expected_method = {
            "rank": method["rank"],
            "alpha": method["alpha"],
            "dropout": method["dropout"],
            "target_scope": method["target_scope"],
        }
        if (
            payload.get("status") == "success"
            and payload.get("optimizer_steps") == 1
            and payload.get("adapter_saved") is True
            and payload.get("manifest_sha256") == manifest_fingerprint
            and payload.get("model_id") == WRAPPER_MODEL_ID
            and payload.get("model_revision") == WRAPPER_MODEL_REVISION
            and payload.get("method") == expected_method
            and payload.get("stage_e_backward_without_optimizer") is True
        ):
            return
    raise TrainingError(
        "A successful lora-one-step evidence file is required before lora-smoke; "
        "run the one-step command first"
    )


def _require_five_step_evidence(
    output_root: Path,
    *,
    manifest_fingerprint: str,
    method: Mapping[str, Any],
    gpu_safety_limit_bytes: int,
) -> None:
    expected_method = {
        "rank": method["rank"],
        "alpha": method["alpha"],
        "dropout": method["dropout"],
        "target_scope": method["target_scope"],
    }
    for path in reversed(sorted(output_root.glob("*/run_summary.json"))):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        consumption = payload.get("consumption")
        training = payload.get("training")
        adapter = payload.get("adapter")
        verification = payload.get("verification")
        model = payload.get("model")
        identity = payload.get("dataset_identity")
        if not all(
            isinstance(value, dict)
            for value in (consumption, training, adapter, verification, model, identity)
        ):
            continue
        losses = (training.get("initial_loss"), training.get("final_loss"))
        gradients = training.get("gradient_norms")
        finite = all(
            type(value) in (int, float) and math.isfinite(float(value))
            for value in losses
        ) and isinstance(gradients, list) and bool(gradients) and all(
            type(value) in (int, float) and math.isfinite(float(value))
            for value in gradients
        )
        peaks = (
            training.get("peak_cuda_allocated_bytes"),
            training.get("peak_cuda_reserved_bytes"),
        )
        memory_safe = all(
            type(value) is int and 0 <= value < gpu_safety_limit_bytes
            for value in peaks
        )
        if (
            payload.get("status") == "smoke_completed"
            and consumption.get("optimizer_steps") == 5
            and identity.get("manifest_sha256") == manifest_fingerprint
            and payload.get("method") == expected_method
            and model.get("id") == WRAPPER_MODEL_ID
            and model.get("revision") == WRAPPER_MODEL_REVISION
            and model.get("backend") == WRAPPER_BACKEND
            and adapter.get("saved") is True
            and adapter.get("fresh_process_reload") is True
            and verification.get("status") == "success"
            and finite
            and memory_safe
        ):
            return
    raise TrainingError(
        "A verified five-step smoke with matching manifest/method, finite loss and "
        "gradients, safe peak memory, adapter save, and fresh-process reload is "
        "required before a ten-step or full-epoch run"
    )


def _memory_csv_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        system = event.get("system_ram")
        cuda = event.get("cuda")
        process = event.get("cuda_process_check")
        system = system if isinstance(system, dict) else {}
        cuda = cuda if isinstance(cuda, dict) else {}
        process = process if isinstance(process, dict) else {}
        rows.append(
            {
                "stage": event.get("stage"),
                "captured_at_utc": event.get("captured_at_utc"),
                "system_total_bytes": system.get("total_bytes"),
                "system_available_bytes": system.get("available_bytes"),
                "system_used_bytes": system.get("used_bytes"),
                "cuda_allocated_bytes": cuda.get("allocated_bytes"),
                "cuda_reserved_bytes": cuda.get("reserved_bytes"),
                "cuda_peak_allocated_bytes": cuda.get("peak_allocated_bytes"),
                "cuda_peak_reserved_bytes": cuda.get("peak_reserved_bytes"),
                "cuda_process_check_status": process.get("status"),
                "other_cuda_processes": process.get("other_processes", []),
            }
        )
    return rows


def _peak_memory_values(events: list[dict[str, Any]]) -> tuple[int, int, int]:
    peak_allocated = 0
    peak_reserved = 0
    peak_system_used = 0
    for row in _memory_csv_rows(events):
        for key, current in (
            ("cuda_peak_allocated_bytes", peak_allocated),
            ("cuda_peak_reserved_bytes", peak_reserved),
            ("system_used_bytes", peak_system_used),
        ):
            value = row.get(key)
            if type(value) is int and value > current:
                if key == "cuda_peak_allocated_bytes":
                    peak_allocated = value
                elif key == "cuda_peak_reserved_bytes":
                    peak_reserved = value
                else:
                    peak_system_used = value
    return peak_allocated, peak_reserved, peak_system_used


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AdapterVerificationError(f"Missing {label}: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterVerificationError(f"Could not read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdapterVerificationError(f"{label.capitalize()} must be a JSON object")
    return value


def _model_metadata(snapshot: Path | None) -> dict[str, Any]:
    return {
        "id": WRAPPER_MODEL_ID,
        "revision": WRAPPER_MODEL_REVISION,
        "backend": WRAPPER_BACKEND,
        "snapshot_path": str(snapshot) if snapshot is not None else None,
        "qwen_sft_commit": QWEN_SFT_COMMIT,
    }


def _environment_payload() -> dict[str, Any]:
    return {
        "process_id": os.getpid(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": _machine_payload(),
        "dependencies": wrapper_dependency_status(),
        "model": _model_metadata(None),
    }


def _machine_payload() -> dict[str, Any]:
    return {
        "platform": platform.system(),
        "release": platform.release(),
        "gpu": "NVIDIA RTX 3050 6 GB laptop target",
        "execution": "WSL/Linux" if "microsoft" in platform.release().lower() else "Linux",
    }


def _is_cuda_oom(error: BaseException, torch: Any) -> bool:
    candidates = (
        getattr(torch, "OutOfMemoryError", None),
        getattr(torch.cuda, "OutOfMemoryError", None),
    )
    types = tuple(item for item in candidates if isinstance(item, type))
    return bool(types and isinstance(error, types)) or (
        "cuda" in str(error).lower() and "out of memory" in str(error).lower()
    )


__all__ = [
    "inspect_wrapper",
    "run_lora_training",
    "run_wrapper_preflight",
    "verify_adapter",
]
