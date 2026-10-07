"""Fixed hashed-token pooling dual tower, intentionally small for serial trials."""

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .lexical import tokenize


@dataclass(frozen=True)
class BusinessConfig:
    vocab_size: int = 32768
    embedding_dim: int = 64
    output_dim: int = 64
    query_max_tokens: int = 64
    item_max_tokens: int = 96
    learning_rate: float = 0.002
    weight_decay: float = 0.0
    temperature: float = 0.1
    shared_towers: bool = False

    def __post_init__(self):
        for name in ("vocab_size", "embedding_dim", "output_dim", "query_max_tokens", "item_max_tokens"):
            if getattr(self, name) < (2 if name == "vocab_size" else 1):
                raise ValueError(f"Invalid {name}")
        if self.learning_rate <= 0 or self.temperature <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid business optimization configuration")
        if not isinstance(self.shared_towers, bool):
            raise ValueError("shared_towers must be boolean")


def encode_texts(texts, max_tokens, vocab_size):
    result = np.zeros((len(texts), max_tokens), dtype=np.int64)
    for i, text in enumerate(texts):
        tokens = tokenize(text)[:max_tokens]
        for j, token in enumerate(tokens):
            # Python's process-randomized hash is deliberately not used.
            value = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little")
            result[i, j] = 1 + value % (vocab_size - 1)
    return torch.from_numpy(result)


class Tower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
        self.projection = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim), nn.ReLU(),
            nn.Linear(config.embedding_dim, config.output_dim),
        )

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)
        mask = token_ids.ne(0).unsqueeze(-1)
        pooled = (embedded * mask).sum(dim=-2) / mask.sum(dim=-2).clamp_min(1)
        return F.normalize(self.projection(pooled), dim=-1)


class DualTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.query = Tower(config)
        self.item = self.query if config.shared_towers else Tower(config)

    def forward(self, query_tokens, item_tokens):
        query_vectors = self.query(query_tokens)
        item_vectors = self.item(item_tokens)
        return torch.einsum("bd,bnd->bn", query_vectors, item_vectors) / self.config.temperature


def initialize_model(config, seed):
    # CPU construction gives identical initialization independent of device/strategy.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return DualTower(config)


def model_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
