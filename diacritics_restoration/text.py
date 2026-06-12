from __future__ import annotations

import re
import unicodedata
from typing import Iterable


CEDILLA_TO_COMMA = str.maketrans(
    {
        "ş": "ș",
        "Ş": "Ș",
        "ţ": "ț",
        "Ţ": "Ț",
    }
)

DIACRITIC_TO_BASE = str.maketrans(
    {
        "ă": "a",
        "Ă": "A",
        "â": "a",
        "Â": "A",
        "î": "i",
        "Î": "I",
        "ș": "s",
        "Ș": "S",
        "ț": "t",
        "Ț": "T",
        "ş": "s",
        "Ş": "S",
        "ţ": "t",
        "Ţ": "T",
    }
)

DIACRITIC_CHARS = set("ăĂâÂîÎșȘțȚşŞţŢ")
POSSIBLE_DIACRITIC_BASES = set("aAiIsStT")

TOKEN_RE = re.compile(r"\w+|[^\w\s]|\s+", re.UNICODE)
WORD_RE = re.compile(r"\w+", re.UNICODE)


def normalize_text(text: str, collapse_spaces: bool = True) -> str:
    """Normalize Unicode, Romanian comma-below chars, BOMs, CRLFs and spaces."""

    text = unicodedata.normalize("NFC", text or "")
    text = text.translate(CEDILLA_TO_COMMA)
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    if collapse_spaces:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def remove_diacritics(text: str) -> str:
    return normalize_text(text, collapse_spaces=False).translate(DIACRITIC_TO_BASE)


def normalize_for_constraint(text: str) -> str:
    return remove_diacritics(normalize_text(text))


def safety_constraint_ok(source_without_diacritics: str, restored: str) -> bool:
    return normalize_for_constraint(source_without_diacritics) == normalize_for_constraint(restored)


def tokenize_preserving_space(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def tokenize_lm(text: str) -> list[str]:
    return [tok for tok in TOKEN_RE.findall(text) if not tok.isspace()]


def is_word(token: str) -> bool:
    return bool(WORD_RE.fullmatch(token)) and any(ch.isalpha() for ch in token)


def apply_source_case(source_token: str, restored_lower: str) -> str:
    """Apply a simple Romanian-safe case shape from source to a lowercase form."""

    if not restored_lower:
        return restored_lower
    if source_token.isupper():
        return restored_lower.upper()
    if source_token[:1].isupper() and source_token[1:].islower():
        return restored_lower[:1].upper() + restored_lower[1:]
    return restored_lower


def base_form(token: str) -> str:
    return remove_diacritics(token).lower()


def chunk_text(text: str, max_chars: int = 220) -> list[str]:
    """Split long input into chunks while preserving the separator text."""

    text = normalize_text(text)
    if len(text) <= max_chars:
        return [text] if text else []

    pieces = re.split(r"(\n+|(?<=[.!?])\s+)", text)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        if len(current) + len(piece) <= max_chars:
            current += piece
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(piece) <= max_chars:
            current = piece
            continue
        words = re.split(r"(\s+)", piece)
        for word in words:
            if len(current) + len(word) > max_chars and current:
                chunks.append(current)
                current = ""
            current += word
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def has_diacritic(text: str) -> bool:
    return any(ch in DIACRITIC_CHARS for ch in text)


def pairwise_equal_base(source_tokens: Iterable[str], target_tokens: Iterable[str]) -> bool:
    return [base_form(tok) for tok in source_tokens] == [base_form(tok) for tok in target_tokens]

