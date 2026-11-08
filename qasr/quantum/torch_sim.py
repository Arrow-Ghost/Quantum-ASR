"""Reference batched statevector simulator in pure PyTorch (CPU or GPU).

Differentiable through autograd. It is the always-available fallback and the
ground truth the C++ / CUDA kernels are tested against.
"""
from __future__ import annotations

import torch

from .circuit import CNOT, RX, RY, CircuitSpec


class TorchSimulator:
    def __init__(self, spec: CircuitSpec):
        self.spec = spec
        self._perm_cache: dict[tuple, torch.Tensor] = {}
        self._sign_cache: dict[tuple, torch.Tensor] = {}

    def _cnot_perm(self, c: int, t: int, device) -> torch.Tensor:
        key = (c, t, str(device))
        if key not in self._perm_cache:
            idx = torch.arange(2 ** self.spec.n_qubits, device=device)
            flip = ((idx >> c) & 1).bool()
            self._perm_cache[key] = torch.where(flip, idx ^ (1 << t), idx)
        return self._perm_cache[key]

    def _z_signs(self, device) -> torch.Tensor:
        key = (str(device),)
        if key not in self._sign_cache:
            n = self.spec.n_qubits
            idx = torch.arange(2 ** n, device=device)
            bits = torch.stack([(idx >> q) & 1 for q in range(n)], dim=1)  # [D, n]
            self._sign_cache[key] = (1 - 2 * bits).to(torch.float32)
        return self._sign_cache[key]

    def state(self, angles: torch.Tensor) -> torch.Tensor:
        spec, n = self.spec, self.spec.n_qubits
        B, D = angles.shape[0], 2 ** n
        psi = torch.zeros(B, D, dtype=torch.complex64, device=angles.device)
        psi[:, 0] = 1.0
        for t, q0, q1, a in zip(spec.gate_type, spec.q0, spec.q1, spec.angle_idx):
            if t == CNOT:
                psi = psi[:, self._cnot_perm(q0, q1, angles.device)]
                continue
            th = angles[:, a].to(torch.float32).view(B, 1, 1)
            c, s = torch.cos(th / 2), torch.sin(th / 2)
            v = psi.view(B, 2 ** (n - q0 - 1), 2, 2 ** q0)
            v0, v1 = v[:, :, 0, :], v[:, :, 1, :]
            if t == RY:
                n0, n1 = c * v0 - s * v1, s * v0 + c * v1
            elif t == RX:
                mis = torch.complex(torch.zeros_like(s), -s)
                n0, n1 = c * v0 + mis * v1, mis * v0 + c * v1
            else:  # RZ
                ph = torch.complex(c, -s)
                n0, n1 = v0 * ph, v1 * ph.conj()
            psi = torch.stack([n0, n1], dim=2).reshape(B, D)
        return psi

    def expz(self, angles: torch.Tensor) -> torch.Tensor:
        psi = self.state(angles)
        probs = psi.real ** 2 + psi.imag ** 2  # [B, D]
        return probs @ self._z_signs(angles.device)  # [B, n]
