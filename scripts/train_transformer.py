from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from diacritics_restoration.data import TextPair, read_jsonl
from diacritics_restoration.model import Seq2SeqTransformer, TransformerConfig, count_parameters
from diacritics_restoration.tokenization import SentencePieceTokenizer, train_sentencepiece


PRESETS = {
    "tiny": {
        "vocab_size": 2000,
        "d_model": 96,
        "nhead": 4,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "dim_feedforward": 384,
        "batch_size": 64,
    },
    "small": {
        "vocab_size": 4000,
        "d_model": 192,
        "nhead": 4,
        "num_encoder_layers": 3,
        "num_decoder_layers": 3,
        "dim_feedforward": 768,
        "batch_size": 64,
    },
    "recommended": {
        "vocab_size": 8000,
        "d_model": 256,
        "nhead": 4,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        "dim_feedforward": 1024,
        "batch_size": 64,
    },
}


class DiacriticsDataset(Dataset):
    def __init__(
        self,
        pairs: list[TextPair],
        tokenizer: SentencePieceTokenizer,
        *,
        max_positions: int,
    ) -> None:
        self.items: list[tuple[list[int], list[int]]] = []
        for pair in pairs:
            source = tokenizer.encode(pair.source)
            target = tokenizer.encode(pair.target)
            if len(source) <= max_positions and len(target) <= max_positions:
                self.items.append((source, target))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return self.items[index]


def collate_batch(batch: list[tuple[list[int], list[int]]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_src = max(len(src) for src, _ in batch)
    max_tgt = max(len(tgt) for _, tgt in batch)
    sources = torch.full((len(batch), max_src), pad_id, dtype=torch.long)
    targets = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)
    for row, (src, tgt) in enumerate(batch):
        sources[row, : len(src)] = torch.tensor(src, dtype=torch.long)
        targets[row, : len(tgt)] = torch.tensor(tgt, dtype=torch.long)
    return sources, targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight encoder-decoder Transformer.")
    parser.add_argument("--train", default="data/processed/train.jsonl")
    parser.add_argument("--valid", default="data/processed/valid.jsonl")
    parser.add_argument("--out-dir", default="artifacts/model")
    parser.add_argument("--tokenizer-dir", default="artifacts/tokenizer")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="recommended")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-positions", type=int, default=256)
    parser.add_argument("--max-train-docs", type=int, default=0)
    parser.add_argument("--max-valid-docs", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_loss(
    model: Seq2SeqTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for source, target in loader:
            source = source.to(device)
            target = target.to(device)
            logits = model(source, target[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), target[:, 1:].reshape(-1))
            losses.append(float(loss.item()))
    return sum(losses) / max(1, len(losses))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    preset = PRESETS[args.preset]
    train_pairs = read_jsonl(args.train)
    valid_pairs = read_jsonl(args.valid)
    if args.max_train_docs:
        train_pairs = train_pairs[: args.max_train_docs]
    if args.max_valid_docs:
        valid_pairs = valid_pairs[: args.max_valid_docs]

    tokenizer_model = train_sentencepiece(
        [pair.source for pair in train_pairs] + [pair.target for pair in train_pairs],
        output_dir=args.tokenizer_dir,
        vocab_size=preset["vocab_size"],
    )
    tokenizer = SentencePieceTokenizer(tokenizer_model)

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        d_model=preset["d_model"],
        nhead=preset["nhead"],
        num_encoder_layers=preset["num_encoder_layers"],
        num_decoder_layers=preset["num_decoder_layers"],
        dim_feedforward=preset["dim_feedforward"],
        dropout=args.dropout,
        max_positions=args.max_positions,
    )

    train_ds = DiacriticsDataset(train_pairs, tokenizer, max_positions=args.max_positions)
    valid_ds = DiacriticsDataset(valid_pairs, tokenizer, max_positions=args.max_positions)
    train_loader = DataLoader(
        train_ds,
        batch_size=preset["batch_size"],
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_id),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=preset["batch_size"],
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_id),
    )

    device = choose_device(args.device)
    model = Seq2SeqTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_valid = float("inf")
    log_rows: list[dict] = []
    started = time.perf_counter()
    print(
        json.dumps(
            {
                "device": str(device),
                "preset": args.preset,
                "train_items": len(train_ds),
                "valid_items": len(valid_ds),
                "parameters": count_parameters(model),
                "vocab_size": tokenizer.vocab_size,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for source, target in train_loader:
            source = source.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(source, target[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), target[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        train_loss = sum(losses) / max(1, len(losses))
        valid_loss = evaluate_loss(model, valid_loader, criterion, device)
        row = {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss}
        log_rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if valid_loss < best_valid:
            best_valid = valid_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config.to_dict(),
                    "tokenizer_model": str(Path(args.tokenizer_dir) / "ro_diacritics.model"),
                    "preset": args.preset,
                    "epoch": epoch,
                    "valid_loss": valid_loss,
                    "parameters": count_parameters(model),
                },
                out_dir / "transformer.pt",
            )

    training_log = {
        "preset": args.preset,
        "epochs": args.epochs,
        "best_valid_loss": best_valid,
        "seconds": round(time.perf_counter() - started, 3),
        "rows": log_rows,
    }
    (out_dir / "training_log.json").write_text(
        json.dumps(training_log, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
