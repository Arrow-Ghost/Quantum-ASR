"""Train + evaluate the hybrid quantum LID head and its classical baseline."""
from __future__ import annotations

import copy
import json
import math
import time

import numpy as np
import torch
from torch import nn

from .config import Config, checkpoints_dir, results_dir
from .features import load_split
from .models import LABELS, build_model, count_params
from .utils import get_logger, pick_device, seed_everything, write_json

log = get_logger("qasr.train")


def _metrics(y_true: np.ndarray, prob_en: np.ndarray, sources: np.ndarray | None = None) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

    y_pred = (prob_en >= 0.5).astype(np.int64)
    m = {"n": int(len(y_true)), "accuracy": float(accuracy_score(y_true, y_pred)),
         "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
         "confusion[true][pred]": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
         "labels": LABELS}
    if len(np.unique(y_true)) == 2:
        m["roc_auc"] = float(roc_auc_score(y_true, prob_en))
    if sources is not None:
        m["accuracy_by_source"] = {str(s): float((y_pred[sources == s] == y_true[sources == s]).mean())
                                   for s in np.unique(sources)}
    return m


@torch.inference_mode()
def predict(model: nn.Module, X: torch.Tensor, batch: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, X.shape[0], batch):
        logits = model(X[i:i + batch].float())
        out.append(torch.softmax(logits, -1)[:, 1].float().cpu())
    return torch.cat(out).numpy()


@torch.inference_mode()
def measure_latency(model: nn.Module, in_dim: int, device: torch.device, runs: int = 200) -> dict:
    model.eval()
    x = torch.randn(1, in_dim, device=device)
    for _ in range(20):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t.append((time.perf_counter() - t0) * 1000)
    return {"device": str(device), "batch": 1, "mean_ms": float(np.mean(t)), "p95_ms": float(np.percentile(t, 95))}


def _to_device(split: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.from_numpy(split["X"]).to(device)  # float16, [N, V, D]
    y = torch.from_numpy(split["y"]).to(device)
    return X, y


def train_one(cfg: Config, kind: str) -> dict:
    seed_everything(int(cfg.project.seed))
    device = pick_device(cfg.project.device)
    tr, va = load_split(cfg, "train"), load_split(cfg, "val")
    if tr is None or va is None:
        raise FileNotFoundError("features missing - run `python -m qasr features` first")
    Xtr, ytr = _to_device(tr, device)
    Xva, yva = _to_device(va, device)
    N, V, D = Xtr.shape
    t = cfg.train

    # class-balanced sampling + optional per-source up-weighting
    src = tr["source"]
    w = np.ones(N, dtype=np.float64)
    for c in (0, 1):
        mask = tr["y"] == c
        w[mask] = 1.0 / max(1, mask.sum())
    for s, f in (t.get("source_weight") or {}).items():
        w[src == s] *= float(f)
    weights = torch.from_numpy(w / w.sum()).to(device)

    model = build_model(kind, D, cfg, device)
    qparams = set(id(p) for p in model.quantum_parameters()) if kind == "hybrid" else set()
    groups = [{"params": [p for p in model.parameters() if id(p) not in qparams], "lr": float(t.lr)}]
    if qparams:
        groups.append({"params": [p for p in model.parameters() if id(p) in qparams], "lr": float(t.lr_quantum),
                       "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, weight_decay=float(t.weight_decay))
    steps_per_epoch = math.ceil(N / int(t.batch_size))
    total = steps_per_epoch * int(t.epochs)
    warm = steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=float(t.label_smoothing))

    backend = getattr(getattr(model, "quantum", None), "backend_name", "n/a")
    log.info("[%s] params=%d (quantum/matched part=%d) | train %d clips x %d views | dim %d | backend=%s",
             kind, count_params(model), sum(p.numel() for p in model.quantum_parameters()), N, V, D, backend)

    best, best_f1, bad, history = None, -1.0, 0, []
    t_start = time.time()
    for epoch in range(1, int(t.epochs) + 1):
        model.train()
        idx = torch.multinomial(weights, N, replacement=True)
        views = torch.randint(0, V, (N,), device=device)
        tot, correct, seen = 0.0, 0, 0
        t0 = time.time()
        for i in range(0, N, int(t.batch_size)):
            bi, bv = idx[i:i + int(t.batch_size)], views[i:i + int(t.batch_size)]
            x, y = Xtr[bi, bv].float(), ytr[bi]
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(t.grad_clip))
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            correct += (logits.argmax(-1) == y).sum().item()
            seen += len(y)
        pv = predict(model, Xva[:, 0])
        vm = _metrics(va["y"], pv)
        rec = {"epoch": epoch, "train_loss": tot / seen, "train_acc": correct / seen, "val_acc": vm["accuracy"],
               "val_f1": vm["macro_f1"], "sec": round(time.time() - t0, 1)}
        history.append(rec)
        log.info("[%s] epoch %3d | loss %.4f | train acc %.4f | val acc %.4f | val F1 %.4f | %.1fs", kind, epoch,
                 rec["train_loss"], rec["train_acc"], rec["val_acc"], rec["val_f1"], rec["sec"])
        if vm["macro_f1"] > best_f1 + 1e-4:
            best_f1, bad, best = vm["macro_f1"], 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= int(t.patience):
                log.info("[%s] early stop (no val improvement for %d epochs)", kind, bad)
                break
    model.load_state_dict(best)
    train_minutes = (time.time() - t_start) / 60

    # ---- evaluation
    results = {"kind": kind, "params_total": count_params(model),
               "params_quantum_or_matched": sum(p.numel() for p in model.quantum_parameters()),
               "backend": backend, "best_val_f1": best_f1, "epochs_run": len(history),
               "train_minutes": round(train_minutes, 2), "val": _metrics(va["y"], predict(model, Xva[:, 0]))}
    for split in ("test", "test_ood"):
        s = load_split(cfg, split)
        if s is not None:
            Xs, _ = _to_device(s, device)
            results[split] = _metrics(s["y"], predict(model, Xs[:, 0]), s["source"])
    results["latency_head"] = measure_latency(model, D, device)

    ck = checkpoints_dir(cfg) / f"{kind}_best.pt"
    torch.save({"kind": kind, "in_dim": D, "state_dict": model.state_dict(),
                "config": json.loads(json.dumps(cfg)), "metrics": results}, ck)
    rd = results_dir(cfg)
    write_json(rd / f"metrics_{kind}.json", results)
    write_json(rd / f"history_{kind}.json", history)
    _plot(history, rd / f"curves_{kind}.png", kind)
    log.info("[%s] saved %s | test acc %.4f | OOD acc %s", kind, ck, results.get("test", {}).get("accuracy", float("nan")),
             f"{results['test_ood']['accuracy']:.4f}" if "test_ood" in results else "n/a")
    return results


def _plot(history: list[dict], path, kind: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    e = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(e, [h["train_loss"] for h in history])
    ax[0].set_title(f"{kind}: train loss")
    ax[1].plot(e, [h["train_acc"] for h in history], label="train acc")
    ax[1].plot(e, [h["val_f1"] for h in history], label="val macro-F1")
    ax[1].legend()
    ax[1].set_title(f"{kind}: accuracy")
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_report(cfg: Config) -> None:
    rd = results_dir(cfg)
    rows = []
    for kind in ("hybrid", "classical"):
        p = rd / f"metrics_{kind}.json"
        if p.exists():
            rows.append(json.loads(p.read_text(encoding="utf-8")))
    if not rows:
        return
    lines = ["# LID training report", "",
             f"Circuit: {cfg.quantum.n_qubits} qubits x {cfg.quantum.n_layers} layers ({cfg.quantum.entangler}), "
             f"features: {cfg.features.encoder}/{cfg.features.pooling}", "",
             "| model | params (total / quantum-or-matched) | backend | val F1 | test acc | test F1 | OOD acc | OOD F1 "
             "| head latency (ms) | train min |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        te, od = r.get("test", {}), r.get("test_ood", {})
        lines.append(
            f"| {r['kind']} | {r['params_total']} / {r['params_quantum_or_matched']} | {r['backend']} | "
            f"{r['best_val_f1']:.4f} | {te.get('accuracy', float('nan')):.4f} | {te.get('macro_f1', float('nan')):.4f} | "
            f"{od.get('accuracy', float('nan')):.4f} | {od.get('macro_f1', float('nan')):.4f} | "
            f"{r['latency_head']['mean_ms']:.3f} | {r['train_minutes']} |")
    lines += ["", "Per-source test accuracy (look for a source the model fails on = dataset bias):", ""]
    for r in rows:
        lines.append(f"- **{r['kind']}**: " + ", ".join(
            f"{k} {v:.3f}" for k, v in {**r.get('test', {}).get('accuracy_by_source', {}),
                                        **r.get('test_ood', {}).get('accuracy_by_source', {})}.items()))
    (rd / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("report: %s", rd / "REPORT.md")
