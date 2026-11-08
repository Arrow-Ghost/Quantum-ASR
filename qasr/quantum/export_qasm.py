"""Export the trained LID circuit as OpenQASM 3 (runs in Qiskit / on IBM hardware).

Two files:
  lid_circuit_parametric.qasm : the 10 input features are `input float` values,
                                trained weights are baked in as constants
  lid_circuit_sample.qasm     : fully bound circuit for one real test clip
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from ..config import Config, checkpoints_dir, results_dir
from ..features import load_split
from ..models import load_lid_checkpoint
from ..utils import get_logger
from .circuit import CNOT, GATE_NAMES

log = get_logger("qasr.qasm")


def _emit(spec, angle_expr, header_inputs: list[str]) -> str:
    out = ["OPENQASM 3.0;", 'include "stdgates.inc";', ""]
    out += header_inputs
    out += [f"qubit[{spec.n_qubits}] q;", f"bit[{spec.n_qubits}] c;", ""]
    for t, q0, q1, a in zip(spec.gate_type, spec.q0, spec.q1, spec.angle_idx):
        if t == CNOT:
            out.append(f"cx q[{q0}], q[{q1}];")
        else:
            out.append(f"{GATE_NAMES[t]}({angle_expr(a)}) q[{q0}];")
    out += ["", "// measure in Z; <Z_q> = P(c_q=0) - P(c_q=1) feeds the classical head", "c = measure q;"]
    return "\n".join(out) + "\n"


def export(cfg: Config) -> list[Path]:
    ck = checkpoints_dir(cfg) / "hybrid_best.pt"
    model, meta = load_lid_checkpoint(ck, torch.device("cpu"), backend="torch")
    ql = model.quantum
    spec, n = ql.spec, ql.n
    scale = ql.enc_scale.detach().numpy()
    bias = ql.enc_bias.detach().numpy()
    w = ql.weights.detach().numpy()

    def param_expr(a: int) -> str:
        l, rem = divmod(a, 4 * n)
        blk, q = divmod(rem, n)
        if blk == 0:
            return f"{scale[l, q]:.8f} * x{q} + {bias[l, q]:.8f}"
        return f"{w[l, blk - 1, q]:.8f}"

    rd = results_dir(cfg)
    p1 = rd / "lid_circuit_parametric.qasm"
    inputs = [f"// x_q = pi * tanh(Linear(LayerNorm(embedding)))_q  (see qasr/models.py)"] + \
             [f"input float[64] x{q};" for q in range(n)] + [""]
    p1.write_text(_emit(spec, param_expr, inputs), encoding="utf-8")

    paths = [p1]
    test = load_split(cfg, "test")
    if test is not None:
        x = torch.from_numpy(test["X"][:1, 0]).float()
        with torch.no_grad():
            z = math.pi * torch.tanh(model.proj(model.norm(x)))
            ang = ql.angles(z)[0].numpy()
        p2 = rd / "lid_circuit_sample.qasm"
        p2.write_text(_emit(spec, lambda a: f"{ang[a]:.8f}",
                            [f"// test clip {test['ids'][0]} (label {['hi', 'en'][int(test['y'][0])]})", ""]),
                      encoding="utf-8")
        paths.append(p2)
    log.info("OpenQASM 3 written: %s", ", ".join(str(p) for p in paths))
    return paths
