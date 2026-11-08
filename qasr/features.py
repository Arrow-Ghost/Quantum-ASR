"""Frozen-encoder feature extraction, computed ONCE and cached to .npz.

Training clips get several augmented random crops ("views") so the quantum
head sees crop-length / gain / noise variation without re-running the encoder
every epoch. Val/test clips get one deterministic center crop.
"""
from __future__ import annotations

import math
import zlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .audio_io import SR, center_crop, load_audio
from .config import Config, data_dir
from .utils import get_logger, pick_device, read_jsonl

log = get_logger("qasr.features")
SPLITS = ("train", "val", "test", "test_ood")


# ---------------------------------------------------------------- embedders
class WhisperEmbedder:
    """Mean(+std) pooled Whisper encoder states over the *valid* frames."""

    frame_samples = 320  # 50 encoder frames per second

    def __init__(self, model_name: str, device: torch.device, pooling: str = "mean_std", encoder=None):
        from transformers import WhisperFeatureExtractor

        self.fe = WhisperFeatureExtractor.from_pretrained(model_name)
        if encoder is None:
            from transformers import WhisperModel

            encoder = WhisperModel.from_pretrained(model_name).encoder
        self.device = device
        self.dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.encoder = encoder.to(device=device, dtype=self.dtype).eval()
        self.pooling = pooling
        d = self.encoder.config.d_model
        self.dim = d * (2 if pooling == "mean_std" else 1)

    def _mel(self, wavs: list[np.ndarray]) -> torch.Tensor:
        try:
            out = self.fe(wavs, sampling_rate=SR, return_tensors="pt", device=str(self.device))
        except TypeError:
            out = self.fe(wavs, sampling_rate=SR, return_tensors="pt")
        return out.input_features

    @torch.inference_mode()
    def __call__(self, wavs: list[np.ndarray]) -> np.ndarray:
        feats = self._mel(wavs).to(self.device, self.dtype)
        h = self.encoder(feats).last_hidden_state.float()  # [B, 1500, D]
        T = h.shape[1]
        lens = torch.tensor([min(T, max(1, math.ceil(len(w) / self.frame_samples))) for w in wavs],
                            device=h.device, dtype=torch.float32)
        mask = (torch.arange(T, device=h.device)[None, :] < lens[:, None]).float().unsqueeze(-1)
        mean = (h * mask).sum(1) / lens[:, None]
        if self.pooling == "mean_std":
            var = (((h - mean[:, None, :]) ** 2) * mask).sum(1) / lens[:, None]
            out = torch.cat([mean, torch.sqrt(var + 1e-5)], dim=-1)
        else:
            out = mean
        return out.cpu().numpy()


class MelStatsEmbedder:
    """No pretrained network: mean+std of 80 log-mel bands (research ablation)."""

    def __init__(self, device: torch.device, n_mels: int = 80):
        import torchaudio

        self.mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=400, hop_length=160, n_mels=n_mels).to(device)
        self.device = device
        self.dim = 2 * n_mels

    @torch.inference_mode()
    def __call__(self, wavs: list[np.ndarray]) -> np.ndarray:
        outs = []
        for w in wavs:
            m = torch.log(self.mel(torch.from_numpy(w).to(self.device)) + 1e-6)  # [n_mels, T]
            outs.append(torch.cat([m.mean(1), m.std(1)]).cpu().numpy())
        return np.stack(outs)


def build_embedder(cfg: Config, device: torch.device, encoder=None):
    f = cfg.features
    if f.encoder == "whisper_base":
        return WhisperEmbedder(f.whisper_model, device, f.pooling, encoder=encoder)
    if f.encoder == "mel_stats":
        return MelStatsEmbedder(device)
    raise ValueError(f"unknown features.encoder {f.encoder}")


# ------------------------------------------------------------- augmentation
def make_view(x: np.ndarray, rng: np.random.Generator, train: bool, f: Config) -> np.ndarray:
    if not train:
        return center_crop(x, int(f.eval_crop_s * SR))
    L = int(rng.uniform(f.crop_min_s, f.crop_max_s) * SR)
    if len(x) > L:
        s = int(rng.integers(0, len(x) - L + 1))
        x = x[s:s + L]
    x = x * (10 ** (rng.uniform(-f.aug_gain_db, f.aug_gain_db) / 20.0))
    if rng.random() < f.aug_noise_prob:
        snr = rng.uniform(*f.aug_snr_db)
        p_sig = float(np.mean(x ** 2)) + 1e-10
        noise = rng.standard_normal(len(x)).astype(np.float32)
        if rng.random() < 0.5:  # brown-ish noise (fans, traffic) instead of white
            noise = np.cumsum(noise)
            noise -= noise.mean()
            noise /= (np.std(noise) + 1e-8)
        x = x + noise * math.sqrt(p_sig / (10 ** (snr / 10.0)))
    return np.clip(x, -1, 1).astype(np.float32)


class ViewDataset(Dataset):
    def __init__(self, rows: list[dict], root: Path, n_views: int, train: bool, fcfg: Config, seed: int):
        self.rows, self.root, self.V, self.train, self.f, self.seed = rows, root, n_views, train, fcfg, seed

    def __len__(self) -> int:
        return len(self.rows) * self.V

    def __getitem__(self, i: int):
        r = self.rows[i // self.V]
        rng = np.random.default_rng(zlib.crc32(f"{self.seed}:{r['id']}:{i % self.V}".encode()))
        x = load_audio(self.root / r["path"])
        return make_view(x, rng, self.train, self.f), i


def _collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


# ---------------------------------------------------------------- extraction
def feature_dir(cfg: Config) -> Path:
    return data_dir(cfg) / "features" / f"{cfg.features.encoder}_{cfg.features.pooling}"


def extract_features(cfg: Config, rows: list[dict] | None = None) -> Path:
    root = data_dir(cfg)
    rows = rows if rows is not None else read_jsonl(root / "manifest.jsonl")
    if not rows:
        raise FileNotFoundError(f"No manifest at {root / 'manifest.jsonl'} - run `python -m qasr download` first.")
    device = pick_device(cfg.project.device)
    out_dir = feature_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    embedder = None
    f = cfg.features
    for split in SPLITS:
        srows = [r for r in rows if r["split"] == split]
        if not srows:
            continue
        out = out_dir / f"{split}.npz"
        V = int(f.views_train) if split == "train" else 1
        if out.exists():
            with np.load(out, allow_pickle=False) as z:
                if z["X"].shape[0] == len(srows) and z["X"].shape[1] == V:
                    log.info("features/%s: cached (%d clips x %d views)", split, len(srows), V)
                    continue
        if embedder is None:
            embedder = build_embedder(cfg, device)
            log.info("Encoder %s on %s, embedding dim %d", f.encoder, device, embedder.dim)
        ds = ViewDataset(srows, root, V, split == "train", f, int(cfg.project.seed))
        workers = int(f.num_workers)
        dl = DataLoader(ds, batch_size=int(f.batch_size), shuffle=False, num_workers=workers,
                        collate_fn=_collate, persistent_workers=False, prefetch_factor=4 if workers else None)
        X = np.zeros((len(srows), V, embedder.dim), dtype=np.float16)
        from tqdm import tqdm

        for wavs, idx in tqdm(dl, desc=f"features/{split}", unit="batch"):
            emb = embedder(wavs)
            for e, i in zip(emb, idx):
                X[i // V, i % V] = e
        np.savez(out, X=X, y=np.array([r["label"] for r in srows], dtype=np.int64),
                 source=np.array([r["source"] for r in srows]), ids=np.array([r["id"] for r in srows]))
        log.info("features/%s: saved %s %s", split, out, X.shape)
    return out_dir


def load_split(cfg: Config, split: str) -> dict | None:
    p = feature_dir(cfg) / f"{split}.npz"
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
