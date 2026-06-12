from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from .baselines import FrequencyDictionary, NGramLanguageModel, rerank_with_ngram
from .model import Seq2SeqTransformer, TransformerConfig
from .text import chunk_text, normalize_text, safety_constraint_ok
from .tokenization import SentencePieceTokenizer


SourceName = Literal["transformer", "ngram_fallback", "dictionary_fallback", "identity"]
FallbackStrategy = Literal["dictionary", "ngram"]


@dataclass
class RestorationResult:
    text: str
    source: SourceName
    constraint_ok: bool


class TransformerRestorer:
    """Thin wrapper around the trained checkpoint and tokenizer."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.device = torch.device(device)
        self.tokenizer = SentencePieceTokenizer(checkpoint["tokenizer_model"])
        self.config = TransformerConfig.from_dict(checkpoint["config"])
        self.model = Seq2SeqTransformer(self.config)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def restore(self, text: str, max_new_tokens: int | None = None) -> str:
        ids = self.tokenizer.encode(normalize_text(text))
        source = torch.tensor([ids], dtype=torch.long, device=self.device)
        max_tokens = max_new_tokens or min(self.config.max_positions - 1, len(ids) + 32)
        generated = self.model.generate(source, max_new_tokens=max_tokens)
        return normalize_text(self.tokenizer.decode(generated[0].tolist()))


class RestorationPipeline:
    """Production path: try the model, then fall back to safer local baselines."""

    def __init__(
        self,
        *,
        model_path: str | Path | None = "artifacts/model/transformer.pt",
        dictionary_path: str | Path | None = "artifacts/baselines/frequency_dictionary.json",
        ngram_path: str | Path | None = "artifacts/baselines/ngram_lm.json",
        device: str = "cpu",
        beam_size: int = 8,
        fallback_strategy: FallbackStrategy = "dictionary",
    ) -> None:
        self.beam_size = beam_size
        self.fallback_strategy = fallback_strategy
        self.model: TransformerRestorer | None = None
        self.dictionary: FrequencyDictionary | None = None
        self.ngram: NGramLanguageModel | None = None

        if model_path and Path(model_path).exists():
            self.model = TransformerRestorer(model_path, device=device)
        if dictionary_path and Path(dictionary_path).exists():
            self.dictionary = FrequencyDictionary.load(dictionary_path)
        if ngram_path and Path(ngram_path).exists():
            self.ngram = NGramLanguageModel.load(ngram_path)

    def restore(self, text: str) -> RestorationResult:
        normalized = normalize_text(text)
        if not normalized:
            return RestorationResult("", "identity", True)

        restored_chunks: list[str] = []
        used_sources: list[SourceName] = []
        for chunk in chunk_text(normalized):
            # Each chunk is validated independently so long texts cannot drift.
            result = self._restore_chunk(chunk)
            restored_chunks.append(result.text)
            used_sources.append(result.source)

        restored = "".join(restored_chunks)
        ok = safety_constraint_ok(normalized, restored)
        source: SourceName = "transformer" if all(src == "transformer" for src in used_sources) else used_sources[-1]
        return RestorationResult(restored, source, ok)

    def _restore_chunk(self, text: str) -> RestorationResult:
        if self.model is not None:
            generated = self.model.restore(text)
            if self._model_output_ok(text, generated):
                return RestorationResult(generated, "transformer", True)

        fallback = self.fallback(text)
        return fallback

    def _model_output_ok(self, text: str, generated: str) -> bool:
        # First reject rewrites, then reject unseen diacritized forms for known bases.
        if not safety_constraint_ok(text, generated):
            return False
        if self.dictionary is None:
            return True
        return self.dictionary.lexicon_constraint_ok(text, generated)

    def fallback(self, text: str) -> RestorationResult:
        # Dictionary-first is the default because it beat the n-gram reranker on test.
        if self.fallback_strategy == "dictionary":
            dictionary_result = self._dictionary_fallback(text)
            if dictionary_result.constraint_ok and dictionary_result.source != "identity":
                return dictionary_result
            ngram_result = self._ngram_fallback(text)
            if ngram_result.constraint_ok and ngram_result.source != "identity":
                return ngram_result
            return RestorationResult(text, "identity", safety_constraint_ok(text, text))

        ngram_result = self._ngram_fallback(text)
        if ngram_result.constraint_ok and ngram_result.source != "identity":
            return ngram_result
        dictionary_result = self._dictionary_fallback(text)
        if dictionary_result.constraint_ok and dictionary_result.source != "identity":
            return dictionary_result
        return RestorationResult(text, "identity", safety_constraint_ok(text, text))

    def _ngram_fallback(self, text: str) -> RestorationResult:
        if self.dictionary is not None and self.ngram is not None:
            restored = rerank_with_ngram(text, self.dictionary, self.ngram, beam_size=self.beam_size)
            if safety_constraint_ok(text, restored):
                return RestorationResult(restored, "ngram_fallback", True)
        return RestorationResult(text, "identity", safety_constraint_ok(text, text))

    def _dictionary_fallback(self, text: str) -> RestorationResult:
        if self.dictionary is not None:
            restored = self.dictionary.restore(text)
            if safety_constraint_ok(text, restored):
                return RestorationResult(restored, "dictionary_fallback", True)
        return RestorationResult(text, "identity", safety_constraint_ok(text, text))
