"""Explicit, non-transliterating transcript views for fair ASR evaluation."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        "…": "...",
        "，": ",",
        "。": "।",
        "؟": "?",
    }
)


@dataclass(frozen=True, slots=True)
class NormalizationOptions:
    """The exact standard-comparison behaviour stored with every run."""

    remove_punctuation: bool = False
    lowercase_latin: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": "standard",
            "unicode": "NFKC",
            "collapse_whitespace": True,
            "remove_punctuation": self.remove_punctuation,
            "lowercase_latin": self.lowercase_latin,
            "transliteration": False,
            "inverse_text_normalization": False,
        }


def raw_comparable_text(text: str) -> str:
    """Retain the source comparison text, changing only line-ending mechanics."""

    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_standard(text: str, options: NormalizationOptions) -> str:
    """Normalize comparison mechanics without transliterating Hindi or English."""

    normalized = unicodedata.normalize("NFKC", raw_comparable_text(text)).translate(
        _PUNCTUATION_TRANSLATION
    )
    if options.lowercase_latin:
        normalized = "".join(
            character.lower() if character.isascii() and character.isalpha() else character
            for character in normalized
        )
    if options.remove_punctuation:
        normalized = "".join(
            " " if unicodedata.category(character).startswith("P") else character
            for character in normalized
        )
    return _WHITESPACE.sub(" ", normalized).strip()


def is_blank(text: str) -> bool:
    return not text.strip()


def is_punctuation_only(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and not any(
        unicodedata.category(character)[0] in {"L", "N"} for character in stripped
    )
