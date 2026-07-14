"""Official qwen-asr supervised target and single-sample collation contract.

This module deliberately has no Torch, Transformers, qwen-asr, or PEFT import
at module import time.  Callers inject the processor and Torch module from the
isolated wrapper environment.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from ..exceptions import WrapperCompatibilityError

ASR_TEXT_TOKEN = "<asr_text>"
IGNORE_INDEX = -100
QWEN_SFT_COMMIT = "7c6daf77a2421100f5fb066495372c00129d39ff"


@dataclass(frozen=True, slots=True)
class WrapperSample:
    """One decoded sample retained only for the current collator call."""

    sample_id: str
    audio: Any
    duration_seconds: float
    transcript: str
    language: str | None
    line_number: int
    source: str | None = None
    speaker_id: str | None = None
    recording_id: str | None = None
    domain: str | None = None
    split: str | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True, slots=True)
class CollatedWrapperBatch:
    """Official processor inputs plus a sanitized correctness inspection."""

    inputs: Any
    inspection: Mapping[str, Any]
    serialized_target: str


def qwen_language_name(language: str | None) -> str:
    """Map canonical manifest language metadata to Qwen's target language name."""

    if language is None:
        return "None"
    normalized = language.strip().casefold().replace("_", "-")
    if not normalized:
        raise WrapperCompatibilityError(
            "Manifest language must be omitted, not blank, when genuinely unavailable"
        )
    primary_language = normalized.partition("-")[0]
    if normalized in {"hindi", "hinglish", "hindi-english"} or primary_language == "hi":
        return "Hindi"
    if normalized == "english" or primary_language == "en":
        return "English"
    if normalized in {"none", "unknown", "und"}:
        raise WrapperCompatibilityError(
            "Use a null manifest language only when language is genuinely unavailable"
        )
    raise WrapperCompatibilityError(
        f"Unsupported wrapper training language {language!r}; use Hindi/hi, "
        "English/en, or null"
    )


def serialize_official_target(transcript: str, language: str | None) -> str:
    """Return Qwen's exact wrapper SFT target without normalizing transcript text."""

    if not isinstance(transcript, str) or not transcript.strip():
        raise WrapperCompatibilityError("Training transcript must be a non-empty string")
    return f"language {qwen_language_name(language)}{ASR_TEXT_TOKEN}{transcript}"


def build_prefix_messages(audio: Any = None) -> list[dict[str, Any]]:
    """Build the exact system-plus-audio messages used by Qwen's SFT script."""

    return [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio}]},
    ]


def build_official_prefix(processor: Any) -> str:
    """Render the official assistant-generation prefix through the wrapper processor."""

    rendered = processor.apply_chat_template(
        [build_prefix_messages(None)],
        add_generation_prompt=True,
        tokenize=False,
    )
    if isinstance(rendered, (list, tuple)) and len(rendered) == 1:
        rendered = rendered[0]
    if not isinstance(rendered, str) or not rendered:
        raise WrapperCompatibilityError(
            "Wrapper processor did not produce one non-empty official prefix"
        )
    return rendered


def collate_official_single(
    processor: Any,
    sample: WrapperSample,
    *,
    include_private_text: bool = False,
) -> CollatedWrapperBatch:
    """Collate exactly one sample using Qwen's official prefix/masking algorithm.

    The upstream collator's simple prefix slice is only safe for batch size one
    with the wrapper processor's current left-padding behaviour.  This laptop
    milestone therefore rejects batching at this boundary rather than silently
    generalizing an incorrect mask.
    """

    if not math.isfinite(sample.duration_seconds) or sample.duration_seconds <= 0:
        raise WrapperCompatibilityError("Decoded sample duration must be positive and finite")
    prefix = build_official_prefix(processor)
    target = serialize_official_target(sample.transcript, sample.language)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise WrapperCompatibilityError("Wrapper processor has no tokenizer")
    eos = getattr(tokenizer, "eos_token", None)
    if not isinstance(eos, str) or not eos:
        raise WrapperCompatibilityError("Wrapper tokenizer has no EOS token")

    full_text = prefix + target + eos
    common = {
        "audio": [sample.audio],
        "return_tensors": "pt",
        "padding": True,
        "truncation": False,
    }
    full_inputs = processor(text=[full_text], **common)
    prefix_inputs = processor(text=[prefix], **common)
    for name, values in (("full", full_inputs), ("prefix", prefix_inputs)):
        if "input_ids" not in values or "attention_mask" not in values:
            raise WrapperCompatibilityError(
                f"Wrapper processor {name} output lacks input_ids or attention_mask"
            )
        if _shape(values["input_ids"])[0] != 1:
            raise WrapperCompatibilityError("Wrapper SFT collator requires batch size exactly 1")

    prefix_len = int(prefix_inputs["attention_mask"].sum(dim=1).tolist()[0])
    full_ids = full_inputs["input_ids"]
    sequence_len = _shape(full_ids)[1]
    if not 0 < prefix_len < sequence_len:
        raise WrapperCompatibilityError(
            f"Invalid official prefix boundary {prefix_len} for sequence length {sequence_len}"
        )

    labels = full_ids.clone()
    labels[0, :prefix_len] = IGNORE_INDEX
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        labels[labels == pad_id] = IGNORE_INDEX
    full_inputs["labels"] = labels

    label_values = labels[0].tolist()
    input_values = full_ids[0].tolist()
    supervised_positions = [i for i, value in enumerate(label_values) if value != IGNORE_INDEX]
    if not supervised_positions:
        raise WrapperCompatibilityError("Official wrapper batch contains no supervised label token")
    if any(position < prefix_len for position in supervised_positions):
        raise WrapperCompatibilityError("Prompt or audio-prefix token remains supervised")
    if any(label_values[i] != input_values[i] for i in supervised_positions):
        raise WrapperCompatibilityError(
            "Supervised labels do not match vocabulary input IDs at the same positions"
        )
    if pad_id is not None and any(
        input_values[i] == pad_id and label_values[i] != IGNORE_INDEX
        for i in range(sequence_len)
    ):
        raise WrapperCompatibilityError("Padding label was not masked with -100")

    supervised_ids = [label_values[i] for i in supervised_positions]
    expected_ids = _token_ids(tokenizer, target + eos)
    if supervised_ids != expected_ids:
        raise WrapperCompatibilityError(
            "Supervised token IDs do not exactly equal serialized target plus EOS"
        )
    decoded_target = tokenizer.decode(
        supervised_ids[:-1],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded_target != target:
        raise WrapperCompatibilityError(
            "Supervised target does not decode back to the exact serialized transcript"
        )

    ignored = sum(value == IGNORE_INDEX for value in label_values)
    inspection: dict[str, Any] = {
        "sample_ids": [sample.sample_id],
        "manifest_lines": [sample.line_number],
        "audio_duration_seconds": sample.duration_seconds,
        "input_shapes": {key: _shape(value) for key, value in full_inputs.items()},
        "prefix_token_count": prefix_len,
        "supervised_label_tokens": len(supervised_ids),
        "ignored_label_tokens": ignored,
        "padding_label_tokens": sum(value == pad_id for value in input_values)
        if pad_id is not None
        else 0,
        "target_language": qwen_language_name(sample.language),
        "target_character_count": len(target),
        "target_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
        "decoded_supervised_target_matches": True,
        "labels_match_target_token_ids": True,
        "prefix_fully_masked": True,
        "padding_fully_masked": True,
        "truncation": False,
        "batch_size": 1,
        "special_token_ids": {
            "pad": pad_id,
            "eos": getattr(tokenizer, "eos_token_id", None),
            "asr_text": _single_token_id(tokenizer, ASR_TEXT_TOKEN),
        },
    }
    if include_private_text:
        inspection["raw_transcript"] = sample.transcript
        inspection["serialized_target"] = target
        inspection["decoded_supervised_target"] = decoded_target
    return CollatedWrapperBatch(full_inputs, inspection, target)


def validate_finite_inputs(inputs: Mapping[str, Any], torch_module: Any) -> None:
    """Reject a processor batch containing non-finite floating tensors."""

    for name, value in inputs.items():
        if not bool(getattr(torch_module, "is_tensor")(value)):
            continue
        if bool(value.is_floating_point()) and not bool(torch_module.isfinite(value).all().item()):
            raise WrapperCompatibilityError(f"Processor output {name!r} contains NaN or Inf")


def patch_outer_forward(model: Any) -> None:
    """Apply Qwen's official wrapper SFT forward delegation verbatim in behaviour."""

    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return
    thinker = getattr(model, "thinker", None)
    if thinker is None or not hasattr(thinker, "forward"):
        raise WrapperCompatibilityError(
            "Wrapper model has no thinker.forward required by official SFT"
        )

    def forward(
        self: Any,
        input_ids: Any = None,
        attention_mask: Any = None,
        input_features: Any = None,
        feature_attention_mask: Any = None,
        labels: Any = None,
        **kwargs: Any,
    ) -> Any:
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def _single_token_id(tokenizer: Any, token: str) -> int | None:
    values = _token_ids(tokenizer, token)
    return values[0] if len(values) == 1 else None


def _shape(value: Any) -> list[int]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return []
    return [int(item) for item in shape]


__all__ = [
    "ASR_TEXT_TOKEN",
    "IGNORE_INDEX",
    "QWEN_SFT_COMMIT",
    "CollatedWrapperBatch",
    "WrapperSample",
    "build_official_prefix",
    "build_prefix_messages",
    "collate_official_single",
    "patch_outer_forward",
    "qwen_language_name",
    "serialize_official_target",
    "validate_finite_inputs",
]
