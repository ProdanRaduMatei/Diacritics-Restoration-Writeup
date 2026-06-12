from __future__ import annotations

import difflib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .text import normalize_text, remove_diacritics


@dataclass(frozen=True)
class TextPair:
    doc_id: str
    source: str
    target: str


def read_raw_pairs(raw_dir: str | Path) -> list[TextPair]:
    raw_path = Path(raw_dir)
    pairs: list[TextPair] = []
    for target_path in sorted(raw_path.glob("*.gt.txt")):
        source_path = Path(str(target_path) + ".bak")
        if not source_path.exists():
            continue
        source = source_path.read_text(encoding="utf-8", errors="replace")
        target = target_path.read_text(encoding="utf-8", errors="replace")
        pairs.append(TextPair(target_path.stem.replace(".gt", ""), source, target))
    return pairs


def _similarity(source: str, target: str) -> float:
    # Compare source with target stripped of diacritics; large gaps mean bad pairs.
    return difflib.SequenceMatcher(None, source, remove_diacritics(target)).ratio()


def clean_pairs(
    pairs: Iterable[TextPair],
    *,
    min_similarity: float = 0.98,
    max_length_ratio: float = 1.15,
    max_chars: int = 512,
) -> tuple[list[TextPair], dict]:
    # Keep cleaning conservative: remove corruption, but do not try OCR correction here.
    kept: list[TextPair] = []
    seen: set[tuple[str, str]] = set()
    report = {
        "input_pairs": 0,
        "kept_pairs": 0,
        "removed_empty": 0,
        "removed_replacement_char": 0,
        "removed_length": 0,
        "removed_similarity": 0,
        "removed_duplicate": 0,
        "examples": {
            "length": [],
            "similarity": [],
            "duplicate": [],
            "replacement_char": [],
        },
    }

    for pair in pairs:
        report["input_pairs"] += 1
        source = normalize_text(pair.source)
        target = normalize_text(pair.target)

        if not source or not target:
            report["removed_empty"] += 1
            continue

        if "\ufffd" in source or "\ufffd" in target:
            report["removed_replacement_char"] += 1
            if len(report["examples"]["replacement_char"]) < 5:
                report["examples"]["replacement_char"].append(pair.doc_id)
            continue

        shorter = max(1, min(len(source), len(target)))
        longer = max(len(source), len(target))
        if longer > max_chars or longer / shorter > max_length_ratio:
            report["removed_length"] += 1
            if len(report["examples"]["length"]) < 5:
                report["examples"]["length"].append(
                    {
                        "doc_id": pair.doc_id,
                        "source_len": len(source),
                        "target_len": len(target),
                    }
                )
            continue

        sim = _similarity(source, target)
        if sim < min_similarity:
            report["removed_similarity"] += 1
            if len(report["examples"]["similarity"]) < 5:
                report["examples"]["similarity"].append(
                    {
                        "doc_id": pair.doc_id,
                        "similarity": round(sim, 4),
                        "source": source[:120],
                        "target": target[:120],
                    }
                )
            continue

        key = (source, target)
        if key in seen:
            report["removed_duplicate"] += 1
            if len(report["examples"]["duplicate"]) < 5:
                report["examples"]["duplicate"].append(pair.doc_id)
            continue
        seen.add(key)
        kept.append(TextPair(pair.doc_id, source, target))

    report["kept_pairs"] = len(kept)
    return kept, report


def split_by_document(
    pairs: list[TextPair],
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
) -> dict[str, list[TextPair]]:
    # Split whole documents, not random lines, to avoid near-duplicate leakage.
    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * train_ratio)
    valid_end = train_end + int(n * valid_ratio)
    return {
        "train": shuffled[:train_end],
        "valid": shuffled[train_end:valid_end],
        "test": shuffled[valid_end:],
    }


def write_jsonl(path: str | Path, pairs: Iterable[TextPair]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(asdict(pair), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[TextPair]:
    pairs: list[TextPair] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pairs.append(TextPair(obj["doc_id"], obj["source"], obj["target"]))
    return pairs


def write_cleaning_report(path: str | Path, report: dict, splits: dict[str, list[TextPair]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Cleaning report",
        "===============",
        "",
        f"Input pairs: {report['input_pairs']}",
        f"Kept pairs: {report['kept_pairs']}",
        f"Removed empty: {report['removed_empty']}",
        f"Removed replacement char: {report['removed_replacement_char']}",
        f"Removed length: {report['removed_length']}",
        f"Removed similarity: {report['removed_similarity']}",
        f"Removed duplicate: {report['removed_duplicate']}",
        "",
        "Split sizes",
        "-----------",
    ]
    for name, split_pairs in splits.items():
        lines.append(f"{name}: {len(split_pairs)}")
    lines.extend(["", "Examples", "--------", json.dumps(report["examples"], ensure_ascii=False, indent=2)])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
