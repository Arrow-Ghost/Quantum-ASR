"""Simulator backends behind one interface:   expz = backend(angles[B, n_angles]) -> [B, n_qubits]

  cuda_kernel : custom CUDA kernel (qsim_cuda.cu) compiled by NVRTC via CuPy,
                adjoint gradients, statevector in shared memory   (fastest)
  cpu_kernel  : custom C++ kernel (qsim_cpu.cpp) compiled with g++/clang/MSVC,
                OpenMP over the batch, loaded with ctypes
  torch       : pure PyTorch reference (autograd), runs everywhere

`auto` picks the fastest backend that compiles AND passes a numerical self-test
against the torch reference, otherwise falls back to torch.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from ..utils import get_logger
from .circuit import CircuitSpec
from .torch_sim import TorchSimulator

log = get_logger("qasr.quantum")
KDIR = Path(__file__).parent / "kernels"
BUILD_DIR = KDIR / "build"


# ====================================================================== torch
class TorchBackend:
    name = "torch"

    def __init__(self, spec: CircuitSpec, device: torch.device):
        self.sim = TorchSimulator(spec)

    def __call__(self, angles: torch.Tensor) -> torch.Tensor:
        return self.sim.expz(angles)


# ================================================================= CPU kernel
def _compile_cpu_kernel() -> Path:
    src = KDIR / "qsim_cpu.cpp"
    # key on source AND machine: a -march=native build from another CPU could crash (e.g. AVX-512)
    machine = f"{platform.node()}|{platform.processor()}|{platform.machine()}".encode()
    digest = hashlib.md5(src.read_bytes() + machine).hexdigest()[:10]
    ext = ".dll" if os.name == "nt" else (".dylib" if platform.system() == "Darwin" else ".so")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    out = BUILD_DIR / f"qsim_cpu_{digest}{ext}"
    if out.exists():
        return out
    cmds = []
    for cxx in ("g++", "clang++"):
        if shutil.which(cxx):
            base = [cxx, "-O3", "-std=c++17", "-shared", "-fPIC", str(src), "-o", str(out)]
            arch = ["-march=native"] if platform.machine().lower() in ("x86_64", "amd64") else []
            static = ["-static-libgcc", "-static-libstdc++"] if os.name == "nt" else []
            cmds.append(base + arch + ["-fopenmp"] + static)
            cmds.append(base + static)  # retry without OpenMP / march
    if shutil.which("cl"):
        cmds.append(["cl", "/nologo", "/O2", "/openmp", "/LD", "/EHsc", str(src), f"/Fe:{out}",
                     f"/Fo:{BUILD_DIR}{os.sep}"])
    if not cmds:
        raise RuntimeError("no C++ compiler (g++, clang++ or cl) found on PATH")
    errors = []
    for cmd in cmds:
        env = os.environ.copy()
        exe = shutil.which(cmd[0])
        if exe:  # MinGW's cc1plus needs its own bin dir on PATH to load its DLLs
            env["PATH"] = str(Path(exe).parent) + os.pathsep + env.get("PATH", "")
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode == 0 and out.exists():
            log.info("compiled CPU kernel: %s", " ".join(cmd[:2]))
            return out
        errors.append(r.stderr[-800:])
    raise RuntimeError("CPU kernel compilation failed:\n" + "\n".join(errors))


class _CpuFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, angles, backend):
        a = np.ascontiguousarray(angles.detach().to("cpu", torch.float32).numpy())
        B = a.shape[0]
        out = np.empty((B, backend.n), dtype=np.float32)
        backend.lib.qsim_forward(_fp(a), B, backend.n_angles, backend.n, backend.G, *backend.gate_ptrs, _fp(out))
        ctx.backend = backend
        ctx.save_for_backward(angles)
        return torch.from_numpy(out).to(angles.device)

    @staticmethod
    def backward(ctx, grad_out):
        (angles,) = ctx.saved_tensors
        be = ctx.backend
        a = np.ascontiguousarray(angles.detach().to("cpu", torch.float32).numpy())
        g = np.ascontiguousarray(grad_out.detach().to("cpu", torch.float32).numpy())
        ga = np.zeros_like(a)
        be.lib.qsim_backward(_fp(a), _fp(g), a.shape[0], be.n_angles, be.n, be.G, *be.gate_ptrs, _fp(ga))
        return torch.from_numpy(ga).to(angles.device), None


def _fp(x: np.ndarray):
    return x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class CpuKernelBackend:
    name = "cpu_kernel"

    def __init__(self, spec: CircuitSpec, device: torch.device):
        path = _compile_cpu_kernel()
        if os.name == "nt":
            gxx = shutil.which("g++")
            if gxx:  # libgomp DLL lives next to MinGW g++
                os.add_dll_directory(str(Path(gxx).parent))
        self.lib = ctypes.CDLL(str(path))
        i32p, f32p, c_int = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float), ctypes.c_int
        self.lib.qsim_forward.argtypes = [f32p, c_int, c_int, c_int, c_int, i32p, i32p, i32p, i32p, f32p]
        self.lib.qsim_backward.argtypes = [f32p, f32p, c_int, c_int, c_int, c_int, i32p, i32p, i32p, i32p, f32p]
        self.n, self.n_angles, self.G = spec.n_qubits, spec.n_angles, spec.n_gates
        self._gate_arrays = [np.ascontiguousarray(np.asarray(v, dtype=np.int32))
                             for v in (spec.gate_type, spec.q0, spec.q1, spec.angle_idx)]
        self.gate_ptrs = [x.ctypes.data_as(i32p) for x in self._gate_arrays]

    def __call__(self, angles: torch.Tensor) -> torch.Tensor:
        return _CpuFn.apply(angles, self)


# ================================================================ CUDA kernel
class _CudaFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, angles, backend):
        a = angles.detach().to(torch.float32).contiguous()
        out = torch.empty((a.shape[0], backend.n), device=a.device, dtype=torch.float32)
        backend.launch_forward(a, out)
        ctx.backend = backend
        ctx.save_for_backward(a)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (a,) = ctx.saved_tensors
        g = grad_out.to(torch.float32).contiguous()
        ga = torch.zeros_like(a)
        ctx.backend.launch_backward(a, g, ga)
        return ga, None


class CudaKernelBackend:
    name = "cuda_kernel"

    def __init__(self, spec: CircuitSpec, device: torch.device, threads: int = 256):
        import cupy as cp

        if device.type != "cuda":
            raise RuntimeError("cuda_kernel backend needs a CUDA device")
        self.cp = cp
        self.n, self.n_angles, self.G = spec.n_qubits, spec.n_angles, spec.n_gates
        D = 2 ** self.n
        self.threads = int(min(threads, max(32, D // 2)))
        self.threads = 1 << (self.threads.bit_length() - 1)  # power of two for the reduction
        code = (KDIR / "qsim_cuda.cu").read_text(encoding="utf-8")
        opts = ("--use_fast_math", "-std=c++14")
        self.k_fwd = cp.RawKernel(code, "qsim_forward", options=opts)
        self.k_bwd = cp.RawKernel(code, "qsim_backward", options=opts)
        self.smem_fwd = D * 8 + self.threads * 4
        self.smem_bwd = 2 * D * 8 + self.threads * 4
        for k, sm in ((self.k_fwd, self.smem_fwd), (self.k_bwd, self.smem_bwd)):
            if sm > 48 * 1024:
                k.max_dynamic_shared_size_bytes = sm  # opt-in (Ampere/Ada: up to ~99 KB)
        with cp.cuda.Device(device.index or 0):
            self.gates = [cp.asarray(np.asarray(v, dtype=np.int32)) for v in
                          (spec.gate_type, spec.q0, spec.q1, spec.angle_idx)]
        self.i32 = np.int32

    def _stream(self):
        return self.cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)

    def launch_forward(self, a: torch.Tensor, out: torch.Tensor) -> None:
        cp = self.cp
        with self._stream():
            self.k_fwd((a.shape[0],), (self.threads,),
                       (cp.from_dlpack(a), self.i32(self.n_angles), self.i32(self.n), self.i32(self.G), *self.gates,
                        cp.from_dlpack(out)), shared_mem=self.smem_fwd)

    def launch_backward(self, a: torch.Tensor, g: torch.Tensor, ga: torch.Tensor) -> None:
        cp = self.cp
        with self._stream():
            self.k_bwd((a.shape[0],), (self.threads,),
                       (cp.from_dlpack(a), cp.from_dlpack(g), self.i32(self.n_angles), self.i32(self.n),
                        self.i32(self.G), *self.gates, cp.from_dlpack(ga)), shared_mem=self.smem_bwd)

    def __call__(self, angles: torch.Tensor) -> torch.Tensor:
        return _CudaFn.apply(angles, self)


# ================================================================== selection
def self_test(backend, spec: CircuitSpec, device: torch.device, batch: int = 7, tol: float = 2e-4) -> float:
    """Compare forward values and adjoint gradients with the torch reference."""
    gen = torch.Generator().manual_seed(0)
    angles = (torch.rand(batch, spec.n_angles, generator=gen) * 6.283 - 3.1415).to(device)
    w = torch.randn(batch, spec.n_qubits, generator=gen).to(device)
    ref = TorchSimulator(spec)
    a1 = angles.clone().requires_grad_(True)
    (ref.expz(a1) * w).sum().backward()
    a2 = angles.clone().requires_grad_(True)
    out2 = backend(a2)
    (out2 * w).sum().backward()
    with torch.no_grad():
        err_f = (ref.expz(angles) - out2).abs().max().item()
        err_g = (a1.grad - a2.grad).abs().max().item()
    err = max(err_f, err_g)
    if err > tol:
        raise AssertionError(f"{backend.name}: max error {err:.2e} (forward {err_f:.2e}, grad {err_g:.2e})")
    return err


def make_backend(spec: CircuitSpec, device: torch.device, preference: str = "auto", threads: int = 256):
    if preference == "auto":
        order = ["cuda_kernel", "torch"] if device.type == "cuda" else ["cpu_kernel", "torch"]
    else:
        order = [preference]
    last_err = None
    for name in order:
        try:
            if name == "torch":
                be = TorchBackend(spec, device)
            elif name == "cpu_kernel":
                be = CpuKernelBackend(spec, device)
            elif name == "cuda_kernel":
                be = CudaKernelBackend(spec, device, threads)
            else:
                raise ValueError(f"unknown backend {name}")
            if name != "torch":
                err = self_test(be, spec, device)
                log.info("quantum backend %s passed self-test (max err %.1e)", name, err)
            else:
                log.info("quantum backend torch (reference)")
            return be
        except Exception as e:  # fall through to the next backend
            last_err = e
            if preference != "auto":
                raise
            log.warning("quantum backend %s unavailable: %s", name, str(e).splitlines()[0][:200])
    raise RuntimeError(f"no quantum backend available: {last_err}")
