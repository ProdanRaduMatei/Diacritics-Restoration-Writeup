from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diacritics_restoration.baselines import FrequencyDictionary, NGramLanguageModel
from diacritics_restoration.data import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frequency dictionary and n-gram LM baselines.")
    parser.add_argument("--train", default="data/processed/train.jsonl")
    parser.add_argument("--out-dir", default="artifacts/baselines")
    parser.add_argument("--ngram-order", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--ambiguity-report", default="reports/ambiguity_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = read_jsonl(args.train)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dictionary = FrequencyDictionary.fit(pairs)
    dictionary.save(out_dir / "frequency_dictionary.json")

    lm = NGramLanguageModel(order=args.ngram_order, alpha=args.alpha).fit([pair.target for pair in pairs])
    lm.save(out_dir / "ngram_lm.json")

    ambiguity = dictionary.ambiguity_inventory(min_count=2)
    Path(args.ambiguity_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.ambiguity_report).write_text(
        json.dumps({"ambiguous_forms": ambiguity[:200]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "train_pairs": len(pairs),
                "dictionary_entries": len(dictionary.variants),
                "ambiguous_entries": len(ambiguity),
                "ngram_order": lm.order,
                "vocab_size": len(lm.vocab),
                "skipped_alignment": getattr(dictionary, "skipped_alignment", 0),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
