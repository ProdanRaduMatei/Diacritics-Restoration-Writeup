from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

from .data import TextPair
from .text import (
    DIACRITIC_CHARS,
    POSSIBLE_DIACRITIC_BASES,
    normalize_text,
    remove_diacritics,
    safety_constraint_ok,
    tokenize_lm,
)


@dataclass
class Prediction:
    doc_id: str
    source: str
    target: str
    prediction: str
    layer: str


def char_accuracy(prediction: str, target: str) -> float:
    pred = normalize_text(prediction)
    gold = normalize_text(target)
    denom = max(len(pred), len(gold), 1)
    correct = sum(a == b for a, b in zip(pred, gold))
    return correct / denom


def word_accuracy(prediction: str, target: str) -> float:
    pred_tokens = tokenize_lm(normalize_text(prediction))
    gold_tokens = tokenize_lm(normalize_text(target))
    denom = max(len(pred_tokens), len(gold_tokens), 1)
    correct = sum(a == b for a, b in zip(pred_tokens, gold_tokens))
    return correct / denom


def diacritic_position_accuracy(source: str, prediction: str, target: str) -> float:
    src = normalize_text(source)
    pred = normalize_text(prediction)
    gold = normalize_text(target)
    denom = 0
    correct = 0
    for i, gold_ch in enumerate(gold):
        src_ch = src[i] if i < len(src) else ""
        pred_ch = pred[i] if i < len(pred) else ""
        base = remove_diacritics(gold_ch)
        is_relevant = gold_ch in DIACRITIC_CHARS or src_ch in POSSIBLE_DIACRITIC_BASES or base in POSSIBLE_DIACRITIC_BASES
        if not is_relevant:
            continue
        denom += 1
        correct += int(pred_ch == gold_ch)
    return correct / max(denom, 1)


def evaluate_predictions(predictions: list[Prediction], max_examples: int = 20) -> dict:
    n = max(1, len(predictions))
    layer_counts = Counter(pred.layer for pred in predictions)
    metrics = {
        "items": len(predictions),
        "char_accuracy": sum(char_accuracy(p.prediction, p.target) for p in predictions) / n,
        "word_accuracy": sum(word_accuracy(p.prediction, p.target) for p in predictions) / n,
        "diacritic_position_accuracy": sum(
            diacritic_position_accuracy(p.source, p.prediction, p.target) for p in predictions
        )
        / n,
        "sentence_exact_match": sum(normalize_text(p.prediction) == normalize_text(p.target) for p in predictions) / n,
        "constraint_violation_rate": sum(
            not safety_constraint_ok(p.source, p.prediction) for p in predictions
        )
        / n,
        "layer_counts": dict(layer_counts),
        "examples": [],
    }
    for item in predictions:
        if normalize_text(item.prediction) == normalize_text(item.target):
            continue
        metrics["examples"].append(
            {
                "doc_id": item.doc_id,
                "source": item.source,
                "target": item.target,
                "prediction": item.prediction,
                "layer": item.layer,
            }
        )
        if len(metrics["examples"]) >= max_examples:
            break
    return metrics


def evaluate_pairs(
    pairs: list[TextPair],
    restore_fn: Callable[[str], tuple[str, str]],
) -> dict:
    predictions: list[Prediction] = []
    for pair in pairs:
        prediction, layer = restore_fn(pair.source)
        predictions.append(Prediction(pair.doc_id, pair.source, pair.target, prediction, layer))
    return evaluate_predictions(predictions)

