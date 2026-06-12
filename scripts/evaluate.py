from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diacritics_restoration.baselines import FrequencyDictionary, NGramLanguageModel, rerank_with_ngram
from diacritics_restoration.data import TextPair, read_jsonl
from diacritics_restoration.evaluation import evaluate_pairs
from diacritics_restoration.inference import RestorationPipeline
from diacritics_restoration.linguistic import write_linguistic_report


HARD_CASES = [
    (
        "Am pus peste peste legume.",
        "Am pus pește peste legume.",
        "The first `peste` is a noun and the second is a preposition; corpus frequency alone prefers `peste`.",
    ),
    (
        "Fata sta in fata casei.",
        "Fata stă în fața casei.",
        "`fata` can be `fata`, `fată`, `fața`; the second position is forced by the prepositional phrase.",
    ),
    (
        "Mana dreapta era in mana medicului.",
        "Mâna dreaptă era în mâna medicului.",
        "`mana` differs between indefinite `mână` and definite `mâna`; local syntax matters.",
    ),
    (
        "Para dulce nu e acelasi lucru cu o para.",
        "Para dulce nu e același lucru cu o pară.",
        "`para` may stay plain in named/foreign contexts, but fruit needs `pară`.",
    ),
    (
        "Casa veche nu era acasa.",
        "Casa veche nu era acasă.",
        "`casa/casă` and `acasă` show that token frequency is not enough; phrase role is important.",
    ),
    (
        "Sa-si ia cartea sau sa o lase?",
        "Să-și ia cartea sau să o lase?",
        "Hyphenated clitics test whether the system handles punctuation-adjacent words.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Romanian diacritics restoration systems.")
    parser.add_argument("--test", default="data/processed/test.jsonl")
    parser.add_argument("--dictionary", default="artifacts/baselines/frequency_dictionary.json")
    parser.add_argument("--ngram", default="artifacts/baselines/ngram_lm.json")
    parser.add_argument("--model", default="artifacts/model/transformer.pt")
    parser.add_argument("--fallback", choices=["dictionary", "ngram"], default="dictionary")
    parser.add_argument("--out", default="reports/evaluation.json")
    parser.add_argument("--hard-cases-md", default="reports/hard_cases.md")
    parser.add_argument("--linguistic-md", default="reports/linguistic_report.md")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def write_hard_cases(path: str | Path, rows: list[dict]) -> None:
    lines = [
        "# Manual Hard Cases",
        "",
        "| source | expected | prediction | layer | constraint | why hard |",
        "|---|---|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['source']}`",
                    f"`{row['expected']}`",
                    f"`{row['prediction']}`",
                    row["layer"],
                    str(row["constraint_ok"]),
                    row["why"],
                ]
            )
            + " |"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    pairs = read_jsonl(args.test)

    dictionary = FrequencyDictionary.load(args.dictionary)
    lm = NGramLanguageModel.load(args.ngram)

    results = {
        "frequency_dictionary": evaluate_pairs(
            pairs,
            lambda source: (dictionary.restore(source), "frequency_dictionary"),
        ),
        "ngram_reranker": evaluate_pairs(
            pairs,
            lambda source: (rerank_with_ngram(source, dictionary, lm), "ngram_reranker"),
        ),
    }

    pipeline = RestorationPipeline(
        model_path=args.model,
        dictionary_path=args.dictionary,
        ngram_path=args.ngram,
        device=args.device,
        fallback_strategy=args.fallback,
    )
    results["pipeline"] = evaluate_pairs(
        pairs,
        lambda source: (
            (result := pipeline.restore(source)).text,
            result.source,
        ),
    )

    hard_rows = []
    for source, expected, why in HARD_CASES:
        result = pipeline.restore(source)
        hard_rows.append(
            {
                "source": source,
                "expected": expected,
                "prediction": result.text,
                "layer": result.source,
                "constraint_ok": result.constraint_ok,
                "why": why,
            }
        )
    results["hard_cases"] = hard_rows

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_hard_cases(args.hard_cases_md, hard_rows)
    write_linguistic_report(args.dictionary, HARD_CASES, args.linguistic_md)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
