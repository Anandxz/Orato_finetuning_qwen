"""Small, dependency-free ASR edit-distance metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Any, Sequence

from .normalization import NormalizationOptions, is_blank, is_punctuation_only, normalize_standard, raw_comparable_text


@dataclass(frozen=True, slots=True)
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_length: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float | None:
        return self.errors / self.reference_length if self.reference_length else None

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "reference_length": self.reference_length,
            "errors": self.errors,
            "rate": self.rate,
        }


def edit_counts(reference: Sequence[str], prediction: Sequence[str]) -> EditCounts:
    """Return deterministic Levenshtein operation counts with stable tie breaking."""

    rows = len(reference) + 1
    columns = len(prediction) + 1
    matrix: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(columns)] for _ in range(rows)
    ]
    for row in range(1, rows):
        matrix[row][0] = (row, 0, row, 0)
    for column in range(1, columns):
        matrix[0][column] = (column, 0, 0, column)
    for row in range(1, rows):
        for column in range(1, columns):
            if reference[row - 1] == prediction[column - 1]:
                matrix[row][column] = matrix[row - 1][column - 1]
                continue
            substitution = matrix[row - 1][column - 1]
            deletion = matrix[row - 1][column]
            insertion = matrix[row][column - 1]
            candidates = (
                (substitution[0] + 1, substitution[1] + 1, substitution[2], substitution[3]),
                (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3]),
                (insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1),
            )
            matrix[row][column] = min(candidates, key=lambda value: (value[0], value[1], value[2], value[3]))
    _, substitutions, deletions, insertions = matrix[-1][-1]
    return EditCounts(substitutions, deletions, insertions, len(reference))


def compute_text_metrics(
    reference: str,
    prediction: str,
    *,
    options: NormalizationOptions,
) -> dict[str, Any]:
    """Calculate raw and standard WER/CER without changing stored source text."""

    raw_reference = raw_comparable_text(reference)
    raw_prediction = raw_comparable_text(prediction)
    normalized_reference = normalize_standard(reference, options)
    normalized_prediction = normalize_standard(prediction, options)
    raw_word = edit_counts(raw_reference.split(), raw_prediction.split())
    raw_character = edit_counts(_characters(raw_reference), _characters(raw_prediction))
    word = edit_counts(normalized_reference.split(), normalized_prediction.split())
    character = edit_counts(_characters(normalized_reference), _characters(normalized_prediction))
    return {
        "raw_reference": raw_reference,
        "raw_prediction": raw_prediction,
        "normalized_reference": normalized_reference,
        "normalized_prediction": normalized_prediction,
        "raw_wer": raw_word.rate,
        "raw_cer": raw_character.rate,
        "wer": word.rate,
        "cer": character.rate,
        "word_edits": word.as_dict(),
        "character_edits": character.as_dict(),
        "raw_word_edits": raw_word.as_dict(),
        "raw_character_edits": raw_character.as_dict(),
        "exact_match": normalized_reference == normalized_prediction,
        "blank_prediction": is_blank(prediction),
        "punctuation_only_prediction": is_punctuation_only(prediction),
        "empty_normalized_reference": not normalized_reference,
    }


def aggregate_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate persisted prediction rows, including partial resumed evaluations."""

    successful = [row for row in rows if row.get("status") == "success"]
    failures = [row for row in rows if row.get("status") != "success"]
    word = _aggregate_edit_counts(successful, "word_edits")
    character = _aggregate_edit_counts(successful, "character_edits")
    raw_word = _aggregate_edit_counts(successful, "raw_word_edits")
    raw_character = _aggregate_edit_counts(successful, "raw_character_edits")
    durations = [float(row["audio_duration_seconds"]) for row in successful if row.get("audio_duration_seconds") is not None]
    inference = [float(row["inference_seconds"]) for row in successful if row.get("inference_seconds") is not None]
    rtfs = [float(row["real_time_factor"]) for row in successful if row.get("real_time_factor") is not None]
    return {
        "samples": len(rows),
        "successful_samples": len(successful),
        "failed_samples": len(failures),
        "failure_rate": len(failures) / len(rows) if rows else None,
        "exact_match_rate": _rate(sum(bool(row.get("exact_match")) for row in successful), len(successful)),
        "blank_prediction_rate": _rate(sum(bool(row.get("blank_prediction")) for row in successful), len(successful)),
        "punctuation_only_prediction_rate": _rate(sum(bool(row.get("punctuation_only_prediction")) for row in successful), len(successful)),
        "normalized": {"wer": word.rate, "cer": character.rate, "word_edits": word.as_dict(), "character_edits": character.as_dict()},
        "raw": {"wer": raw_word.rate, "cer": raw_character.rate, "word_edits": raw_word.as_dict(), "character_edits": raw_character.as_dict()},
        "total_audio_duration_seconds": sum(durations),
        "total_inference_seconds": sum(inference),
        "average_real_time_factor": sum(rtfs) / len(rtfs) if rtfs else None,
        "median_real_time_factor": median(rtfs) if rtfs else None,
    }


def _characters(value: str) -> list[str]:
    return [character for character in value if not character.isspace()]


def _aggregate_edit_counts(rows: list[dict[str, Any]], key: str) -> EditCounts:
    totals = Counter()
    for row in rows:
        values = row.get(key) or {}
        for field in ("substitutions", "deletions", "insertions", "reference_length"):
            totals[field] += int(values.get(field) or 0)
    return EditCounts(totals["substitutions"], totals["deletions"], totals["insertions"], totals["reference_length"])


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
