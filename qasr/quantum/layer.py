"""nn.Module wrapper: input features -> circuit angles -> simulator -> <Z_q>."""
from __future__ import annotations

import math

import torch
from torch import nn

from .backends import make_backend
from .circuit import build_reuploading_circuit


class QuantumLayer(nn.Module):
    """Variational data re-uploading circuit.

    x [B, n] (already squashed to (-pi, pi)) is re-uploaded in every layer as
    RY(scale[l,q] * x_q + bias[l,q]); each layer then applies trainable
    RZ-RY-RZ rotations and a CNOT ring. Output: <Z_q> in [-1, 1], shape [B, n].
    Trainable parameters: 5 * n_qubits * n_layers.
    """

    def __init__(self, n_qubits: int, n_layers: int, entangler: str = "ring", backend: str = "auto",
                 device: torch.device | None = None, cuda_threads: int = 256):
        super().__init__()
        self.n, self.L = n_qubits, n_layers
        self.spec = build_reuploading_circuit(n_qubits, n_layers, entangler)
        self.enc_scale = nn.Parameter(torch.ones(n_layers, n_qubits))
        self.enc_bias = nn.Parameter(torch.zeros(n_layers, n_qubits))
        self.weights = nn.Parameter(torch.empty(n_layers, 3, n_qubits).uniform_(-math.pi, math.pi) * 0.1)
        self._backend_pref = backend
        self._cuda_threads = cuda_threads
        self.backend = None
        if device is not None:
            self.bind(device)

    def bind(self, device: torch.device) -> "QuantumLayer":
        self.backend = make_backend(self.spec, device, self._backend_pref, self._cuda_threads)
        return self

    @property
    def backend_name(self) -> str:
        return self.backend.name if self.backend else "unbound"

    def angles(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        enc = x[:, None, :] * self.enc_scale[None] + self.enc_bias[None]           # [B, L, n]
        w = self.weights[None].expand(B, -1, -1, -1)                                 # [B, L, 3, n]
        return torch.cat([enc[:, :, None, :], w], dim=2).reshape(B, self.spec.n_angles)  # layout l*4n+blk*n+q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend is None:
            self.bind(x.device)
        return self.backend(self.angles(x.float()))
