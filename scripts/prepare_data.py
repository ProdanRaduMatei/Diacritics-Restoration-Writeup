from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diacritics_restoration.data import (
    clean_pairs,
    read_raw_pairs,
    split_by_document,
    write_cleaning_report,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Romanian diacritics restoration data.")
    parser.add_argument("--raw-dir", default="training")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--report", default="reports/cleaning_report.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-similarity", type=float, default=0.98)
    parser.add_argument("--max-length-ratio", type=float, default=1.15)
    parser.add_argument("--max-chars", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_pairs = read_raw_pairs(args.raw_dir)
    clean, report = clean_pairs(
        raw_pairs,
        min_similarity=args.min_similarity,
        max_length_ratio=args.max_length_ratio,
        max_chars=args.max_chars,
    )
    splits = split_by_document(clean, seed=args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, pairs in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", pairs)

    split_manifest = {
        "seed": args.seed,
        "splits": {name: [pair.doc_id for pair in pairs] for name, pairs in splits.items()},
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_cleaning_report(args.report, report, splits)
    print(json.dumps({**report, "split_sizes": {k: len(v) for k, v in splits.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
