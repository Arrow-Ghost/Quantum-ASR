"""Spoken-language-ID heads on top of the frozen encoder embedding.

HybridLID    : LayerNorm -> Linear(D, n) -> pi*tanh -> QuantumLayer(n qubits) -> Linear(n, 2)
ClassicalLID : identical, but the quantum layer is replaced by a tanh-MLP with
               (approximately) the same number of trainable parameters.
Comparing the two isolates the contribution of the quantum circuit.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .config import Config
from .quantum.layer import QuantumLayer

N_CLASSES = 2
LABELS = ["hi", "en"]


class HybridLID(nn.Module):
    kind = "hybrid"

    def __init__(self, in_dim: int, n_qubits: int, n_layers: int, entangler: str = "ring", dropout: float = 0.1,
                 backend: str = "auto", device: torch.device | None = None, cuda_threads: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(in_dim, n_qubits)
        self.quantum = QuantumLayer(n_qubits, n_layers, entangler, backend, device, cuda_threads)
        self.head = nn.Linear(n_qubits, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = math.pi * torch.tanh(self.proj(self.drop(self.norm(x))))
        return self.head(self.quantum(z))

    def quantum_parameters(self):
        return list(self.quantum.parameters())


class ClassicalLID(nn.Module):
    kind = "classical"

    def __init__(self, in_dim: int, n_qubits: int, n_layers: int, dropout: float = 0.1, **_):
        super().__init__()
        n = n_qubits
        q_params = 5 * n * n_layers
        hidden = max(2, round((q_params - n) / (2 * n + 1)))  # n->h->n MLP ~ same #params
        self.norm = nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(in_dim, n)
        self.mlp = nn.Sequential(nn.Linear(n, hidden), nn.Tanh(), nn.Linear(hidden, n), nn.Tanh())
        self.head = nn.Linear(n, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = math.pi * torch.tanh(self.proj(self.drop(self.norm(x))))
        return self.head(self.mlp(z))

    def quantum_parameters(self):
        return list(self.mlp.parameters())


def build_model(kind: str, in_dim: int, cfg: Config, device: torch.device) -> nn.Module:
    q = cfg.quantum
    common = dict(in_dim=in_dim, n_qubits=int(q.n_qubits), n_layers=int(q.n_layers), dropout=float(cfg.train.dropout))
    if kind == "hybrid":
        return HybridLID(entangler=q.entangler, backend=q.backend, device=device,
                         cuda_threads=int(q.cuda_block_threads), **common).to(device)
    if kind == "classical":
        return ClassicalLID(**common).to(device)
    raise ValueError(kind)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def load_lid_checkpoint(path, device: torch.device, backend: str | None = None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    from .config import _wrap

    cfg = _wrap(cfg)
    if backend:
        cfg.quantum.backend = backend
    model = build_model(ckpt["kind"], ckpt["in_dim"], cfg, device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
