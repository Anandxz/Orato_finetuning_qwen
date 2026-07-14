from __future__ import annotations

import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_real_wrapper_preflight_uses_owner_training_manifest() -> None:
    """Run the real wrapper contract only when the owner opts in explicitly."""

    train_manifest = os.environ.get("ORATO_ASR_TRAIN_MANIFEST")
    if not train_manifest:
        pytest.skip("ORATO_ASR_TRAIN_MANIFEST is not set")

    # Keep all wrapper/Torch imports after the gate so the default suite is
    # dependency-free and never loads a model or touches CUDA.
    from orato_asr.training import load_wrapper_training_config
    from orato_asr.training.runner import run_wrapper_preflight

    evaluation_manifest = os.environ.get("ORATO_ASR_EVAL_MANIFEST")
    config = load_wrapper_training_config(
        ROOT / "configs" / "train_wrapper_lora_laptop_smoke.yaml"
    )
    result = run_wrapper_preflight(
        config,
        train_manifest=Path(train_manifest),
        eval_manifest=(
            Path(evaluation_manifest) if evaluation_manifest else None
        ),
        offline=True,
    )

    assert result["status"] == "success"
    assert result["decision"] == "wrapper_0.6b_compatible"
    assert result["optimizer_state_allocated"] is False
    assert result["backward_fit"] == "unproven_until_lora-one-step"
    collator = result["stages"]["official_collator"]
    assert collator["prefix_fully_masked"] is True
    assert collator["padding_fully_masked"] is True
    assert collator["supervised_label_tokens"] > 0
    assert result["stages"]["base_finite_forward"]["loss"] >= 0
