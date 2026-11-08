"""Command line entry point:  python -m qasr <command> [options]

  all          full pipeline: access check -> self-test -> download -> features ->
               train (hybrid + classical) -> report -> QASM export -> model prefetch -> benchmark
  check-access verify the Hugging Face token can read every gated AI4Bharat repo
  selftest     environment report + quantum backend correctness/speed
  download     stream the dataset subsets to data/
  features     extract + cache encoder embeddings
  train        train LID heads           (--model hybrid|classical|both)
  report       write results/REPORT.md
  export-qasm  write the trained circuit as OpenQASM 3
  prefetch     download + warm up all live-inference models
  benchmark    offline end-to-end test (LID acc, WER, latency)
  live         live microphone translation (--file x.wav to replay a recording)
  devices      list audio input devices
  smoketest    synthetic end-to-end run, no downloads (CPU OK)
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time

from .config import load_config, results_dir
from .utils import get_logger, write_json

log = get_logger("qasr")


# ------------------------------------------------------------------ commands
def cmd_selftest(cfg) -> bool:
    import torch

    from .quantum.backends import make_backend
    from .quantum.circuit import build_reuploading_circuit

    env = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
           "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env.update(gpu=p.name, vram_gb=round(p.total_memory / 2 ** 30, 1), compute_capability=f"{p.major}.{p.minor}")
    for mod in ("transformers", "datasets", "cupy", "onnxruntime", "silero_vad", "sounddevice"):
        try:
            env[mod] = getattr(__import__(mod), "__version__", "installed")
        except Exception as e:
            env[mod] = f"MISSING ({type(e).__name__})"
    env["cxx"] = next((c for c in ("g++", "clang++", "cl") if shutil.which(c)), None)
    for k, v in env.items():
        log.info("  %-20s %s", k, v)

    q = cfg.quantum
    spec = build_reuploading_circuit(int(q.n_qubits), int(q.n_layers), q.entangler)
    log.info("circuit: %s", spec.summary())
    targets = [("torch", torch.device("cpu")), ("cpu_kernel", torch.device("cpu"))]
    if torch.cuda.is_available():
        targets += [("torch", torch.device("cuda")), ("cuda_kernel", torch.device("cuda"))]
    bench = []
    for name, dev in targets:
        try:
            be = make_backend(spec, dev, name, int(q.cuda_block_threads))
            B = 512 if dev.type == "cuda" else 128
            x = torch.rand(B, spec.n_angles, device=dev, requires_grad=True)
            be(x).sum().backward()  # warm-up / JIT
            reps = 5 if dev.type == "cuda" else 2
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                be(x).sum().backward()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000 / reps
            bench.append({"backend": name, "device": str(dev), "batch": B, "fwd_bwd_ms": round(ms, 2),
                          "samples_per_s": round(B / ms * 1000)})
            log.info("  %-12s %-5s B=%-4d fwd+bwd %8.2f ms  (%d samples/s)", name, dev.type, B, ms, B / ms * 1000)
        except Exception as e:
            bench.append({"backend": name, "device": str(dev), "error": str(e).splitlines()[0][:300]})
            log.warning("  %-12s %-5s FAILED: %s", name, dev.type, str(e).splitlines()[0][:200])
    write_json(results_dir(cfg) / "selftest.json", {"env": env, "quantum_backends": bench})
    return any("error" not in b for b in bench)


def cmd_smoketest(cfg) -> None:
    """Synthetic 'hi' (low, slow harmonics) vs 'en' (high, fast) clips -> full training path."""
    import numpy as np

    from .audio_io import SR, save_flac
    from .config import apply_overrides, resolve
    from .features import extract_features
    from .quantum.export_qasm import export
    from .train import train_one, write_report

    root = resolve("data/_smoke")
    if root.exists():
        shutil.rmtree(root)
    apply_overrides(cfg, ["paths.data_dir=data/_smoke", "paths.checkpoints_dir=data/_smoke/checkpoints",
                          "paths.results_dir=data/_smoke/results", "features.encoder=mel_stats",
                          "features.num_workers=0", "features.views_train=2", "quantum.n_qubits=4",
                          "quantum.n_layers=2", "train.epochs=4", "train.batch_size=32", "train.patience=10"])
    rng = np.random.default_rng(0)
    rows = []
    for i in range(160):
        lang = "hi" if i % 2 == 0 else "en"
        dur = rng.uniform(1.5, 4.0)
        t = np.arange(int(dur * SR)) / SR
        f0 = rng.uniform(110, 160) if lang == "hi" else rng.uniform(190, 260)
        rate = 3.0 if lang == "hi" else 6.0
        env = 0.5 * (1 + np.sin(2 * np.pi * rate * t))
        x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6)) * env * 0.1
        x = (x + rng.standard_normal(len(t)) * 0.01).astype(np.float32)
        split = ["train"] * 8 + ["val", "test"]
        rel = f"clips/{lang}_{i:04d}.flac"
        save_flac(root / rel, x)
        rows.append({"id": f"{lang}_{i:04d}", "path": rel, "source": f"synthetic_{lang}", "lang": lang,
                     "label": 0 if lang == "hi" else 1, "speaker": str(i), "split": split[(i // 2) % 10],
                     "duration": dur, "text": ""})
    with open(root / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    extract_features(cfg)
    for kind in ("hybrid", "classical"):
        res = train_one(cfg, kind)
        log.info("smoketest %s: test acc %.3f", kind, res["test"]["accuracy"])
    write_report(cfg)
    export(cfg)
    try:
        from .live.vad import Segmenter

        seg = Segmenter(cfg.live)
        seg.push(np.zeros(int(cfg.live.chunk_samples), dtype=np.float32))
        log.info("smoketest: Silero VAD OK")
    except ImportError:
        log.info("smoketest: silero-vad not installed here (fine for training-only machines)")
    log.info("SMOKETEST PASSED - outputs in %s", root)


def cmd_all(cfg, args) -> None:
    from .access import check_access
    from .data.download import build_manifest
    from .features import extract_features
    from .quantum.export_qasm import export
    from .train import train_one, write_report

    t0 = time.time()
    if not args.skip_access and not check_access(cfg):
        sys.exit("Fix the Hugging Face access problems listed above, then re-run.")
    if not cmd_selftest(cfg):
        sys.exit("No working quantum backend - see the self-test output above.")
    if not args.skip_download:
        build_manifest(cfg, scale=args.scale)
    extract_features(cfg)
    for kind in cfg.train.models:
        train_one(cfg, kind)
    write_report(cfg)
    export(cfg)
    if not args.skip_live:
        from .benchmark import run_benchmark

        run_benchmark(cfg, n_per_lang=args.bench_n)
    log.info("ALL DONE in %.1f min. Results: %s  |  Start live translation with: live.ps1 (or live.sh)",
             (time.time() - t0) / 60, results_dir(cfg))


# ---------------------------------------------------------------------- main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m qasr", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["all", "check-access", "selftest", "download", "features", "train", "report",
                                        "export-qasm", "prefetch", "benchmark", "live", "devices", "smoketest"])
    ap.add_argument("--config", default=None, help="YAML config (default configs/default.yaml)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set quantum.n_qubits=8 (repeatable)")
    ap.add_argument("--quick", action="store_true",
                    help="small trial run: 5%% of the data, 5 epochs, 20 benchmark clips")
    ap.add_argument("--scale", type=float, default=1.0, help="fraction of each source's max_items to download")
    ap.add_argument("--only", nargs="*", help="download only these source names")
    ap.add_argument("--model", default="both", choices=["hybrid", "classical", "both"])
    ap.add_argument("--file", help="live: replay this audio file instead of the microphone")
    ap.add_argument("--speak", action="store_true", help="live: speak the translation (TTS)")
    ap.add_argument("--device", type=int, help="live: input device index (see `devices`)")
    ap.add_argument("--bench-n", type=int, default=50, help="benchmark clips per language")
    ap.add_argument("--skip-access", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-live", action="store_true", help="all: skip model prefetch + benchmark")
    args = ap.parse_args(argv)

    if args.quick:
        args.overrides = ["train.epochs=5"] + args.overrides
        args.scale = min(args.scale, 0.05)
        args.bench_n = min(args.bench_n, 10)
    cfg = load_config(args.config, args.overrides)
    if args.device is not None:
        cfg.live.input_device = args.device
    from .utils import seed_everything

    seed_everything(int(cfg.project.seed))

    c = args.command
    if c in ("all", "download", "features", "train", "benchmark", "selftest"):
        import logging
        from datetime import datetime

        log_dir = results_dir(cfg) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{c}_{datetime.now():%Y%m%d_%H%M%S}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
        logging.getLogger().addHandler(fh)
        log.info("command: %s | overrides: %s", " ".join(sys.argv[1:]), args.overrides)
    if c == "all":
        cmd_all(cfg, args)
    elif c == "check-access":
        from .access import check_access

        sys.exit(0 if check_access(cfg) else 1)
    elif c == "selftest":
        sys.exit(0 if cmd_selftest(cfg) else 1)
    elif c == "download":
        from .data.download import build_manifest

        build_manifest(cfg, scale=args.scale, only=args.only)
    elif c == "features":
        from .features import extract_features

        extract_features(cfg)
    elif c == "train":
        from .train import train_one, write_report

        for kind in (["hybrid", "classical"] if args.model == "both" else [args.model]):
            train_one(cfg, kind)
        write_report(cfg)
    elif c == "report":
        from .train import write_report

        write_report(cfg)
    elif c == "export-qasm":
        from .quantum.export_qasm import export

        export(cfg)
    elif c == "prefetch":
        from .live.pipeline import LivePipeline

        LivePipeline(cfg)
        log.info("All live models downloaded and warmed up.")
    elif c == "benchmark":
        from .benchmark import run_benchmark

        run_benchmark(cfg, n_per_lang=args.bench_n)
    elif c == "live":
        from .live.pipeline import run_live

        run_live(cfg, file=args.file, speak=True if args.speak else None)
    elif c == "devices":
        from .live.pipeline import list_devices

        list_devices()
    elif c == "smoketest":
        cmd_smoketest(cfg)


if __name__ == "__main__":
    main()
