from __future__ import annotations

from pathlib import Path
from typing import Iterable

import sentencepiece as spm


class SentencePieceTokenizer:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.processor = spm.SentencePieceProcessor()
        self.processor.load(str(self.model_path))
        self.pad_id = self.processor.pad_id()
        self.unk_id = self.processor.unk_id()
        self.bos_id = self.processor.bos_id()
        self.eos_id = self.processor.eos_id()

    @property
    def vocab_size(self) -> int:
        return self.processor.vocab_size()

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = list(self.processor.encode(text, out_type=int))
        if add_special:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        special = {self.pad_id, self.bos_id, self.eos_id}
        clean_ids = [int(idx) for idx in ids if int(idx) not in special]
        return self.processor.decode(clean_ids)


def train_sentencepiece(
    texts: Iterable[str],
    *,
    output_dir: str | Path,
    vocab_size: int = 8000,
    model_type: str = "bpe",
    prefix: str = "ro_diacritics",
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_prefix = output_path / prefix

    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(texts),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        character_coverage=1.0,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        hard_vocab_limit=False,
        input_sentence_size=0,
    )
    return model_prefix.with_suffix(".model")

