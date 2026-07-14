from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from orato_asr.exceptions import WrapperCompatibilityError
from orato_asr.training.official_sft import (
    ASR_TEXT_TOKEN,
    IGNORE_INDEX,
    WrapperSample,
    build_prefix_messages,
    collate_official_single,
    qwen_language_name,
    serialize_official_target,
    validate_finite_inputs,
)


class _Row:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def tolist(self) -> list[int]:
        return list(self.values)


class _Reduction:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def tolist(self) -> list[int]:
        return list(self.values)


@dataclass
class _Mask:
    values: list[list[bool]]


class _Tensor:
    """Small two-dimensional tensor fake for dependency-free collator tests."""

    def __init__(self, values: list[list[int]]) -> None:
        self.values = [list(row) for row in values]
        self.shape = (len(self.values), len(self.values[0]))

    def clone(self) -> "_Tensor":
        return _Tensor(self.values)

    def sum(self, *, dim: int) -> _Reduction:
        assert dim == 1
        return _Reduction([sum(row) for row in self.values])

    def __getitem__(self, key: object) -> _Row:
        assert type(key) is int
        return _Row(self.values[key])  # type: ignore[index]

    def __setitem__(self, key: object, value: int) -> None:
        if isinstance(key, _Mask):
            for row_index, row in enumerate(key.values):
                for column_index, selected in enumerate(row):
                    if selected:
                        self.values[row_index][column_index] = value
            return
        row_index, column_key = key  # type: ignore[misc]
        assert type(row_index) is int and isinstance(column_key, slice)
        start, stop, step = column_key.indices(len(self.values[row_index]))
        for column_index in range(start, stop, step):
            self.values[row_index][column_index] = value

    def __eq__(self, other: object) -> _Mask:  # type: ignore[override]
        return _Mask([[item == other for item in row] for row in self.values])


class _Tokenizer:
    eos_token = "<eos>"
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self) -> None:
        self._piece_to_id = {self.eos_token: self.eos_token_id, ASR_TEXT_TOKEN: 3}
        self._id_to_piece = {value: key for key, value in self._piece_to_id.items()}
        self._next_id = 100

    def _encode(self, text: str) -> list[int]:
        values: list[int] = []
        offset = 0
        while offset < len(text):
            special = next(
                (
                    piece
                    for piece in (ASR_TEXT_TOKEN, self.eos_token)
                    if text.startswith(piece, offset)
                ),
                None,
            )
            if special is not None:
                values.append(self._piece_to_id[special])
                offset += len(special)
                continue
            piece = text[offset]
            if piece not in self._piece_to_id:
                self._piece_to_id[piece] = self._next_id
                self._id_to_piece[self._next_id] = piece
                self._next_id += 1
            values.append(self._piece_to_id[piece])
            offset += 1
        return values

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": self._encode(text)}

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(self._id_to_piece[token_id] for token_id in token_ids)


class _Processor:
    prefix = "<official-prefix>"

    def __init__(self, *, modality_ids_instead_of_tokens: bool = False) -> None:
        self.tokenizer = _Tokenizer()
        self.modality_ids_instead_of_tokens = modality_ids_instead_of_tokens
        self.processor_calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: object, **kwargs: object) -> list[str]:
        assert messages == [build_prefix_messages(None)]
        assert kwargs == {"add_generation_prompt": True, "tokenize": False}
        return [self.prefix]

    def __call__(self, *, text: list[str], **kwargs: object) -> dict[str, _Tensor]:
        self.processor_calls.append({"text": text, **kwargs})
        assert kwargs["return_tensors"] == "pt"
        assert kwargs["padding"] is True
        assert kwargs["truncation"] is False
        assert len(kwargs["audio"]) == 1  # type: ignore[arg-type]
        token_ids = self.tokenizer._encode(text[0])
        if text[0] != self.prefix:
            if self.modality_ids_instead_of_tokens:
                prefix_length = len(self.tokenizer._encode(self.prefix))
                token_ids = token_ids[:prefix_length] + [0, 3]
            token_ids.append(self.tokenizer.pad_token_id)
            attention = [1] * (len(token_ids) - 1) + [0]
        else:
            attention = [1] * len(token_ids)
        return {
            "input_ids": _Tensor([token_ids]),
            "attention_mask": _Tensor([attention]),
        }


def _sample(
    transcript: str = "मुझे appointment reschedule करनी है",
    language: str | None = "Hindi",
    *,
    duration: float = 1.25,
) -> WrapperSample:
    return WrapperSample(
        sample_id="sample-1",
        audio=object(),
        duration_seconds=duration,
        transcript=transcript,
        language=language,
        line_number=7,
    )


@pytest.mark.parametrize(
    ("transcript", "language", "expected"),
    [
        (
            "मुझे appointment reschedule करनी है",
            "Hinglish",
            "language Hindi<asr_text>मुझे appointment reschedule करनी है",
        ),
        ("हाँ", "hi", "language Hindi<asr_text>हाँ"),
        ("जी", "Hindi", "language Hindi<asr_text>जी"),
        ("okay", "English", "language English<asr_text>okay"),
        ("a", "en", "language English<asr_text>a"),
        ("चार", None, "language None<asr_text>चार"),
    ],
)
def test_exact_official_target_preserves_mixed_and_short_transcripts(
    transcript: str, language: str | None, expected: str
) -> None:
    assert serialize_official_target(transcript, language) == expected


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("hi_IN", "Hindi"),
        ("Hindi-English", "Hindi"),
        ("EN", "English"),
        (None, "None"),
    ],
)
def test_language_prefix_mapping(language: str | None, expected: str) -> None:
    assert qwen_language_name(language) == expected


@pytest.mark.parametrize("language", ["", "unknown", "und", "French"])
def test_unsupported_or_placeholder_language_is_rejected(language: str) -> None:
    with pytest.raises(WrapperCompatibilityError):
        qwen_language_name(language)


def test_single_sample_collator_masks_prefix_and_padding_exactly() -> None:
    processor = _Processor()
    sample = _sample()

    collated = collate_official_single(processor, sample, include_private_text=True)

    target = "language Hindi<asr_text>मुझे appointment reschedule करनी है"
    input_ids = collated.inputs["input_ids"][0].tolist()
    labels = collated.inputs["labels"][0].tolist()
    prefix_count = collated.inspection["prefix_token_count"]
    assert collated.serialized_target == target
    assert labels[:prefix_count] == [IGNORE_INDEX] * prefix_count
    assert input_ids[-1] == processor.tokenizer.pad_token_id
    assert labels[-1] == IGNORE_INDEX
    assert [value for value in labels if value != IGNORE_INDEX] == processor.tokenizer._encode(
        target + processor.tokenizer.eos_token
    )
    assert collated.inspection["supervised_label_tokens"] > 0
    assert collated.inspection["prefix_fully_masked"] is True
    assert collated.inspection["padding_fully_masked"] is True
    assert collated.inspection["decoded_supervised_target"] == target
    assert processor.processor_calls[0]["audio"] == [sample.audio]


def test_collator_rejects_modality_type_ids_as_vocabulary_labels() -> None:
    processor = _Processor(modality_ids_instead_of_tokens=True)

    with pytest.raises(
        WrapperCompatibilityError,
        match="Supervised token IDs do not exactly equal serialized target plus EOS",
    ):
        collate_official_single(processor, _sample())


@pytest.mark.parametrize("duration", [0.0, -1.0, float("nan"), float("inf")])
def test_collator_rejects_nonpositive_or_nonfinite_duration(duration: float) -> None:
    with pytest.raises(WrapperCompatibilityError, match="positive and finite"):
        collate_official_single(_Processor(), _sample(duration=duration))


class _FloatingTensor:
    def __init__(self, *, finite: bool) -> None:
        self.finite = finite

    def is_floating_point(self) -> bool:
        return True


class _BooleanResult:
    def __init__(self, value: bool) -> None:
        self.value = value

    def all(self) -> "_BooleanResult":
        return self

    def item(self) -> bool:
        return self.value


class _FakeTorch:
    @staticmethod
    def is_tensor(value: object) -> bool:
        return isinstance(value, _FloatingTensor)

    @staticmethod
    def isfinite(value: _FloatingTensor) -> _BooleanResult:
        return _BooleanResult(value.finite)


def test_processor_inputs_must_be_finite() -> None:
    validate_finite_inputs(
        {"input_features": _FloatingTensor(finite=True), "metadata": object()},
        _FakeTorch,
    )

    with pytest.raises(WrapperCompatibilityError, match="input_features.*NaN or Inf"):
        validate_finite_inputs(
            {"input_features": _FloatingTensor(finite=False)},
            _FakeTorch,
        )
