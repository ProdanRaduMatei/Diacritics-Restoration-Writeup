from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diacritics_restoration.inference import RestorationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore Romanian diacritics for plain text.")
    parser.add_argument("text", nargs="*", help="Text without diacritics. Reads stdin if omitted.")
    parser.add_argument("--model", default="artifacts/model/transformer.pt")
    parser.add_argument("--dictionary", default="artifacts/baselines/frequency_dictionary.json")
    parser.add_argument("--ngram", default="artifacts/baselines/ngram_lm.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fallback", choices=["dictionary", "ngram"], default="dictionary")
    parser.add_argument("--source", action="store_true", help="Print which layer produced the output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = " ".join(args.text) if args.text else sys.stdin.read()
    pipeline = RestorationPipeline(
        model_path=args.model,
        dictionary_path=args.dictionary,
        ngram_path=args.ngram,
        device=args.device,
        fallback_strategy=args.fallback,
    )
    result = pipeline.restore(text)
    print(result.text)
    if args.source:
        print(f"[source={result.source} constraint_ok={result.constraint_ok}]", file=sys.stderr)


if __name__ == "__main__":
    main()
