from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import torch
from torch import nn

warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True",
    category=UserWarning,
)


@dataclass
class TransformerConfig:
    vocab_size: int
    pad_id: int = 0
    bos_id: int = 2
    eos_id: int = 3
    d_model: int = 256
    nhead: int = 4
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_positions: int = 256

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TransformerConfig":
        return cls(**data)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.position_embedding = nn.Embedding(config.max_positions, config.d_model)
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.generator = nn.Linear(config.d_model, config.vocab_size)
        self.generator.weight = self.embedding.weight

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq_len = tokens.shape
        if seq_len > self.config.max_positions:
            raise ValueError(f"Sequence length {seq_len} exceeds max_positions={self.config.max_positions}")
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch, seq_len)
        return self.embedding(tokens) + self.position_embedding(positions)

    @staticmethod
    def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, source: torch.Tensor, target_input: torch.Tensor) -> torch.Tensor:
        src_padding = source.eq(self.config.pad_id)
        tgt_padding = target_input.eq(self.config.pad_id)
        tgt_mask = self._causal_mask(target_input.size(1), target_input.device)
        output = self.transformer(
            self._embed(source),
            self._embed(target_input),
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding,
            tgt_key_padding_mask=tgt_padding,
            memory_key_padding_mask=src_padding,
        )
        return self.generator(output)

    @torch.no_grad()
    def generate(
        self,
        source: torch.Tensor,
        *,
        max_new_tokens: int = 160,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        self.eval()
        generated = torch.full(
            (source.size(0), 1),
            fill_value=self.config.bos_id,
            dtype=torch.long,
            device=source.device,
        )
        for _ in range(max_new_tokens):
            logits = self.forward(source, generated)[:, -1, :]
            if temperature and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if torch.all(next_token.squeeze(1).eq(self.config.eos_id)):
                break
        return generated


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())
