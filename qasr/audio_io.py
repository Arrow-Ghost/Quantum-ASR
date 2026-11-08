"""Audio decoding / resampling / saving. Avoids torchcodec + system FFmpeg:
soundfile first (wav/flac/ogg/mp3), PyAV (bundled FFmpeg) as fallback."""
from __future__ import annotations

import io
from math import gcd
from pathlib import Path

import numpy as np

SR = 16000


def to_mono_float32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        # soundfile -> (frames, channels); PyAV -> (channels, frames)
        x = x.mean(axis=1) if x.shape[0] > x.shape[1] else x.mean(axis=0)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    return x.astype(np.float32, copy=False)


def resample(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    if sr_in == sr_out:
        return x
    from scipy.signal import resample_poly

    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def _decode_soundfile(data: bytes) -> tuple[np.ndarray, int]:
    import soundfile as sf

    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    return x, sr


def _decode_pyav(data: bytes) -> tuple[np.ndarray, int]:
    import av

    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SR)
        chunks = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    x = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    return x.astype(np.float32), SR


def decode_bytes(data: bytes) -> np.ndarray:
    """Decode any common audio container to 16 kHz mono float32 in [-1, 1]."""
    try:
        x, sr = _decode_soundfile(data)
    except Exception:
        x, sr = _decode_pyav(data)
    x = to_mono_float32(x)
    return resample(x, sr, SR)


def load_audio(path: str | Path) -> np.ndarray:
    import soundfile as sf

    try:
        x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        with open(path, "rb") as f:
            return decode_bytes(f.read())
    return resample(to_mono_float32(x), sr, SR)


def save_flac(path: str | Path, x: np.ndarray, sr: int = SR) -> None:
    import soundfile as sf

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(x, -1.0, 1.0)
    sf.write(str(path), x, sr, format="FLAC", subtype="PCM_16")


def center_crop(x: np.ndarray, max_len: int) -> np.ndarray:
    if len(x) <= max_len:
        return x
    start = (len(x) - max_len) // 2
    return x[start:start + max_len]


def rms_normalize(x: np.ndarray, target_dbfs: float = -23.0) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
    gain = 10 ** (target_dbfs / 20.0) / max(rms, 1e-6)
    return np.clip(x * min(gain, 30.0), -1.0, 1.0).astype(np.float32)
