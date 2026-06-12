from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .data import TextPair
from .text import (
    apply_source_case,
    base_form,
    has_diacritic,
    is_word,
    pairwise_equal_base,
    tokenize_lm,
    tokenize_preserving_space,
)


def _json_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in counter.items()}


@dataclass
class FrequencyDictionary:
    variants: dict[str, Counter[str]]

    @classmethod
    def fit(cls, pairs: list[TextPair]) -> "FrequencyDictionary":
        variants: dict[str, Counter[str]] = defaultdict(Counter)
        skipped_alignment = 0
        for pair in pairs:
            source_tokens = tokenize_preserving_space(pair.source)
            target_tokens = tokenize_preserving_space(pair.target)
            if len(source_tokens) != len(target_tokens) or not pairwise_equal_base(source_tokens, target_tokens):
                skipped_alignment += 1
                continue
            for source_token, target_token in zip(source_tokens, target_tokens):
                if not is_word(source_token) or not is_word(target_token):
                    continue
                variants[base_form(source_token)][target_token.lower()] += 1
        model = cls(dict(variants))
        model.skipped_alignment = skipped_alignment  # type: ignore[attr-defined]
        return model

    def candidates(self, token: str, top_k: int = 4) -> list[str]:
        forms = self.variants.get(base_form(token))
        if not forms:
            return [token]
        candidates = [apply_source_case(token, form) for form, _ in forms.most_common(top_k)]
        if token not in candidates:
            candidates.append(token)
        return candidates

    def log_prior(self, source_token: str, candidate: str) -> float:
        forms = self.variants.get(base_form(source_token))
        if not forms:
            return 0.0
        candidate_key = candidate.lower()
        total = sum(forms.values())
        return math.log((forms.get(candidate_key, 0) + 1.0) / (total + len(forms) + 1.0))

    def restore(self, text: str) -> str:
        restored: list[str] = []
        for token in tokenize_preserving_space(text):
            if is_word(token):
                restored.append(self.candidates(token, top_k=1)[0])
            else:
                restored.append(token)
        return "".join(restored)

    def lexicon_constraint_ok(self, source_text: str, restored_text: str) -> bool:
        source_tokens = tokenize_preserving_space(source_text)
        restored_tokens = tokenize_preserving_space(restored_text)
        if len(source_tokens) != len(restored_tokens):
            return False
        for source_token, restored_token in zip(source_tokens, restored_tokens):
            if not is_word(source_token):
                if source_token != restored_token:
                    return False
                continue
            if not is_word(restored_token) or base_form(source_token) != base_form(restored_token):
                return False
            forms = self.variants.get(base_form(source_token))
            if forms:
                if restored_token.lower() not in forms:
                    return False
            elif has_diacritic(restored_token):
                return False
        return True

    def ambiguity_inventory(self, min_count: int = 2) -> list[dict]:
        rows = []
        for base, counter in self.variants.items():
            if len(counter) < 2:
                continue
            total = sum(counter.values())
            if total < min_count:
                continue
            rows.append(
                {
                    "base": base,
                    "total": total,
                    "forms": counter.most_common(),
                    "entropy": _entropy(counter),
                }
            )
        rows.sort(key=lambda row: (row["entropy"], row["total"]), reverse=True)
        return rows

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "variants": {base: _json_counter(counter) for base, counter in self.variants.items()},
            "skipped_alignment": getattr(self, "skipped_alignment", 0),
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FrequencyDictionary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({base: Counter(forms) for base, forms in payload["variants"].items()})


def _entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if not total:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter.values())


class NGramLanguageModel:
    def __init__(self, order: int = 4, alpha: float = 0.1) -> None:
        self.order = order
        self.alpha = alpha
        self.counts: list[Counter[tuple[str, ...]]] = [Counter() for _ in range(order + 1)]
        self.context_counts: list[Counter[tuple[str, ...]]] = [Counter() for _ in range(order + 1)]
        self.vocab: set[str] = set()

    def fit(self, sentences: list[str]) -> "NGramLanguageModel":
        for sentence in sentences:
            tokens = ["<s>"] * (self.order - 1) + [tok.lower() for tok in tokenize_lm(sentence)] + ["</s>"]
            self.vocab.update(tok for tok in tokens if tok not in {"<s>", "</s>"})
            for n in range(1, self.order + 1):
                for i in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[i : i + n])
                    self.counts[n][ngram] += 1
                    if n > 1:
                        self.context_counts[n][ngram[:-1]] += 1
        self.vocab.update({"<unk>", "</s>"})
        return self

    def score_next(self, context: tuple[str, ...], token: str) -> float:
        token = token.lower()
        vocab_size = max(1, len(self.vocab))
        for n in range(min(self.order, len(context) + 1), 1, -1):
            ctx = context[-(n - 1) :]
            context_count = self.context_counts[n].get(ctx, 0)
            if context_count:
                ngram = ctx + (token,)
                count = self.counts[n].get(ngram, 0)
                return math.log((count + self.alpha) / (context_count + self.alpha * vocab_size))
        unigram_count = self.counts[1].get((token,), 0)
        total = sum(self.counts[1].values())
        return math.log((unigram_count + self.alpha) / (total + self.alpha * vocab_size))

    def sentence_score(self, text: str) -> float:
        context = ("<s>",) * (self.order - 1)
        score = 0.0
        for token in [tok.lower() for tok in tokenize_lm(text)] + ["</s>"]:
            score += self.score_next(context, token)
            context = (context + (token,))[-(self.order - 1) :]
        return score

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "order": self.order,
            "alpha": self.alpha,
            "vocab": sorted(self.vocab),
            "counts": [
                {"\t".join(key): value for key, value in counter.items()}
                for counter in self.counts
            ],
            "context_counts": [
                {"\t".join(key): value for key, value in counter.items()}
                for counter in self.context_counts
            ],
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NGramLanguageModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        lm = cls(order=payload["order"], alpha=payload["alpha"])
        lm.vocab = set(payload["vocab"])
        lm.counts = [Counter() for _ in range(lm.order + 1)]
        lm.context_counts = [Counter() for _ in range(lm.order + 1)]
        for i, counter in enumerate(payload["counts"]):
            lm.counts[i] = Counter({tuple(key.split("\t")) if key else tuple(): value for key, value in counter.items()})
        for i, counter in enumerate(payload["context_counts"]):
            lm.context_counts[i] = Counter(
                {tuple(key.split("\t")) if key else tuple(): value for key, value in counter.items()}
            )
        return lm


def rerank_with_ngram(
    text: str,
    dictionary: FrequencyDictionary,
    lm: NGramLanguageModel,
    *,
    beam_size: int = 8,
    candidate_top_k: int = 4,
    dictionary_weight: float = 1.5,
) -> str:
    tokens = tokenize_preserving_space(text)
    start_context = ("<s>",) * (lm.order - 1)
    beams: list[tuple[float, tuple[str, ...], list[str]]] = [(0.0, start_context, [])]

    for token in tokens:
        if token.isspace():
            beams = [(score, context, out + [token]) for score, context, out in beams]
            continue

        if is_word(token):
            candidates = dictionary.candidates(token, top_k=candidate_top_k)
        else:
            candidates = [token]

        next_beams: list[tuple[float, tuple[str, ...], list[str]]] = []
        for score, context, out in beams:
            for candidate in candidates:
                lm_token = candidate.lower()
                prior = dictionary.log_prior(token, candidate) if is_word(token) else 0.0
                next_score = score + lm.score_next(context, lm_token) + dictionary_weight * prior
                next_context = (context + (lm_token,))[-(lm.order - 1) :]
                next_beams.append((next_score, next_context, out + [candidate]))
        next_beams.sort(key=lambda item: item[0], reverse=True)
        beams = next_beams[:beam_size]

    finished: list[tuple[float, list[str]]] = []
    for score, context, out in beams:
        finished.append((score + lm.score_next(context, "</s>"), out))
    finished.sort(key=lambda item: item[0], reverse=True)
    return "".join(finished[0][1])
