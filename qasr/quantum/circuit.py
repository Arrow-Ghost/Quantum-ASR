"""Circuit description shared by every simulator backend.

A circuit is a flat gate list. Each rotation gate reads its angle from a
per-sample angle vector  angles[b, angle_idx]  so that one description covers
both data-encoding gates (angle depends on the input) and trainable gates
(angle is a weight). Conventions (identical in torch / C++ / CUDA):

  * qubit q  <->  bit q of the basis-state index (little-endian, like Qiskit)
  * RX(t) = exp(-i t X / 2), RY(t) = exp(-i t Y / 2), RZ(t) = exp(-i t Z / 2)
  * CNOT(control=q0, target=q1)
  * output = <Z_q> for every qubit, exact (no shot sampling)
"""
from __future__ import annotations

from dataclasses import dataclass, field

RX, RY, RZ, CNOT = 0, 1, 2, 3
GATE_NAMES = {RX: "rx", RY: "ry", RZ: "rz", CNOT: "cx"}


@dataclass
class CircuitSpec:
    n_qubits: int
    n_layers: int
    gate_type: list[int] = field(default_factory=list)
    q0: list[int] = field(default_factory=list)
    q1: list[int] = field(default_factory=list)
    angle_idx: list[int] = field(default_factory=list)
    n_angles: int = 0

    @property
    def n_gates(self) -> int:
        return len(self.gate_type)

    def add(self, gtype: int, q0: int, q1: int = -1, angle: int = -1) -> None:
        self.gate_type.append(gtype)
        self.q0.append(q0)
        self.q1.append(q1)
        self.angle_idx.append(angle)

    def summary(self) -> str:
        n_rot = sum(1 for g in self.gate_type if g != CNOT)
        return (f"{self.n_qubits} qubits, {self.n_layers} layers, {self.n_gates} gates "
                f"({n_rot} rotations, {self.n_gates - n_rot} CNOT), statevector 2^{self.n_qubits}="
                f"{2 ** self.n_qubits} amplitudes")


def build_reuploading_circuit(n_qubits: int, n_layers: int, entangler: str = "ring") -> CircuitSpec:
    """Data re-uploading ansatz. Per layer l and qubit q (angle layout
    idx = l*4n + block*n + q):

        RY(enc[l,q])  ->  RZ(w0[l,q]) RY(w1[l,q]) RZ(w2[l,q])  ->  CNOT entangler
    """
    if not 1 <= n_qubits <= 12:
        raise ValueError("n_qubits must be in [1, 12] (CUDA kernel keeps the state in shared memory)")
    n = n_qubits
    spec = CircuitSpec(n_qubits=n, n_layers=n_layers)
    for l in range(n_layers):
        base = l * 4 * n
        for q in range(n):
            spec.add(RY, q, angle=base + q)                 # data encoding (re-uploaded)
        for q in range(n):
            spec.add(RZ, q, angle=base + n + q)
            spec.add(RY, q, angle=base + 2 * n + q)
            spec.add(RZ, q, angle=base + 3 * n + q)
        if n > 1:
            pairs = [(q, (q + 1) % n) for q in range(n)] if (entangler == "ring" and n > 2) else \
                    [(q, q + 1) for q in range(n - 1)]
            for c, t in pairs:
                spec.add(CNOT, c, t)
    spec.n_angles = 4 * n * n_layers
    return spec
