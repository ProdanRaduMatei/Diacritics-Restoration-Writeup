from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diacritics_restoration.inference import RestorationPipeline


BENCH_TEXTS = [
    "Cum sa-ti poti inchipui, de pilda, un oras",
    "Fata sta in fata casei si tine cartea in mana.",
    "Am pus peste peste legume.",
    "Sa-si ia cartea sau sa o lase?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark local inference latency and artifact size.")
    parser.add_argument("--model", default="artifacts/model/transformer.pt")
    parser.add_argument("--dictionary", default="artifacts/baselines/frequency_dictionary.json")
    parser.add_argument("--ngram", default="artifacts/baselines/ngram_lm.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fallback", choices=["dictionary", "ngram"], default="dictionary")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--out", default="reports/benchmark.json")
    return parser.parse_args()


def size_mb(path: str | Path) -> float:
    p = Path(path)
    return round(p.stat().st_size / (1024 * 1024), 3) if p.exists() else 0.0


def main() -> None:
    args = parse_args()
    pipeline = RestorationPipeline(
        model_path=args.model,
        dictionary_path=args.dictionary,
        ngram_path=args.ngram,
        device=args.device,
        fallback_strategy=args.fallback,
    )
    timings = []
    sources = []
    for _ in range(args.repeats):
        for text in BENCH_TEXTS:
            start = time.perf_counter()
            result = pipeline.restore(text)
            timings.append((time.perf_counter() - start) * 1000)
            sources.append(result.source)
    payload = {
        "device": args.device,
        "runs": len(timings),
        "latency_ms_mean": round(statistics.mean(timings), 3),
        "latency_ms_p50": round(statistics.median(timings), 3),
        "latency_ms_p95": round(sorted(timings)[int(0.95 * (len(timings) - 1))], 3),
        "model_size_mb": size_mb(args.model),
        "dictionary_size_mb": size_mb(args.dictionary),
        "ngram_size_mb": size_mb(args.ngram),
        "sources": {source: sources.count(source) for source in sorted(set(sources))},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
