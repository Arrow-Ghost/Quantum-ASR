"""Stream AI4Bharat (+ LibriSpeech) speech data from Hugging Face and store a
balanced, speaker-diverse subset as 16 kHz mono FLAC + JSONL manifests.

Only the rows we keep are decoded; nothing close to the full ~155 GB of Hindi
corpora is downloaded. The run is resumable: re-running skips clips that are
already on disk.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ..audio_io import SR, center_crop, decode_bytes, save_flac
from ..config import Config, data_dir
from ..utils import get_logger, read_jsonl, write_json

log = get_logger("qasr.download")

LANG_LABEL = {"hi": 0, "en": 1}


def _speaker_key(row: dict, cols: list[str], source: str, fallback: str) -> str:
    if not cols:
        return f"{source}:{fallback}"
    return source + ":" + "|".join(str(row.get(c, "")) for c in cols)


def _assign_split(speaker: str, ratio: list[float], seed: int) -> str:
    h = int(hashlib.md5(f"{seed}:{speaker}".encode("utf-8")).hexdigest(), 16) % 10_000 / 10_000.0
    if h < ratio[0]:
        return "train"
    if h < ratio[0] + ratio[1]:
        return "val"
    return "test"


def _open_stream(src: Config, seed: int, buffer: int):
    from datasets import Audio, load_dataset

    ds = load_dataset(src.repo, src.config, split=src.split, streaming=True)
    keep = [src.audio_col] + ([src.text_col] if src.get("text_col") else []) + list(src.get("speaker_cols") or [])
    try:
        ds = ds.select_columns([c for c in keep if c])
    except Exception:
        pass
    ds = ds.cast_column(src.audio_col, Audio(decode=False))
    return ds.shuffle(seed=seed, buffer_size=buffer)


def download_source(cfg: Config, src: Config, scale: float = 1.0, max_retries: int = 6) -> list[dict]:
    root = data_dir(cfg)
    man_path = root / "manifests" / f"{src.name}.jsonl"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(man_path)
    have = {r["id"] for r in rows}
    target = max(1, int(round(src.max_items * scale)))
    if len(rows) >= target:
        log.info("%-20s already complete (%d clips)", src.name, len(rows))
        return rows[:target]

    per_speaker = Counter(r["speaker"] for r in rows)
    d = cfg.data
    min_len = int(d.min_duration_s * SR)
    max_len = int(d.max_store_s * SR)
    seed = int(cfg.project.seed)
    attempt = 0
    log.info("%-20s streaming %s [%s/%s] -> target %d clips (have %d)",
             src.name, src.repo, src.config, src.split, target, len(rows))

    from tqdm import tqdm

    pbar = tqdm(total=target, initial=len(rows), desc=src.name, unit="clip")
    with open(man_path, "a", encoding="utf-8") as fman:
        while len(rows) < target:
            try:
                stream = _open_stream(src, seed, int(d.shuffle_buffer))
                for idx, row in enumerate(stream):
                    if len(rows) >= target:
                        break
                    cid = f"{src.name}_{idx:07d}"
                    if cid in have:
                        continue
                    spk = _speaker_key(row, list(src.get("speaker_cols") or []), src.name, cid)
                    if per_speaker[spk] >= int(d.max_per_speaker):
                        continue
                    audio = row.get(src.audio_col) or {}
                    data = audio.get("bytes")
                    if data is None and audio.get("path"):
                        with open(audio["path"], "rb") as f:
                            data = f.read()
                    if not data:
                        continue
                    try:
                        x = decode_bytes(data)
                    except Exception as e:  # corrupt clip -> skip
                        log.debug("decode failed for %s: %s", cid, e)
                        continue
                    if len(x) < min_len or not np.isfinite(x).all() or np.abs(x).max() < 1e-4:
                        continue
                    x = center_crop(x, max_len)
                    rel = Path("clips") / src.name / f"{cid}.flac"
                    save_flac(root / rel, x)
                    split = src.get("force_split") or _assign_split(spk, list(d.split_ratio), seed)
                    rec = {
                        "id": cid, "path": rel.as_posix(), "source": src.name, "lang": src.lang,
                        "label": LANG_LABEL[src.lang], "speaker": spk, "split": split,
                        "duration": round(len(x) / SR, 3),
                        "text": str(row.get(src.get("text_col") or "", "") or ""),
                    }
                    fman.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fman.flush()
                    rows.append(rec)
                    have.add(cid)
                    per_speaker[spk] += 1
                    pbar.update(1)
                else:
                    if len(rows) < target:
                        log.warning("%s: stream exhausted at %d/%d clips (speaker cap %d). Continuing.",
                                    src.name, len(rows), target, d.max_per_speaker)
                    break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                attempt += 1
                msg = str(e)
                if "gated" in msg.lower() or "401" in msg or "403" in msg:
                    pbar.close()
                    raise RuntimeError(
                        f"No access to {src.repo}. Open https://huggingface.co/datasets/{src.repo} "
                        f"while logged in, accept the terms, then run `hf auth login`.") from e
                if attempt > max_retries:
                    pbar.close()
                    raise
                wait = min(60, 5 * attempt)
                log.warning("%s: stream error (%s). Retry %d/%d in %ds ...", src.name, msg[:200], attempt,
                            max_retries, wait)
                time.sleep(wait)
    pbar.close()
    return rows


def build_manifest(cfg: Config, scale: float = 1.0, only: list[str] | None = None) -> Path:
    root = data_dir(cfg)
    root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for src in cfg.data.sources:
        if only and src.name not in only:
            continue
        all_rows.extend(download_source(cfg, src, scale=scale))
    out = root / "manifest.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_json(root / "dataset_stats.json", dataset_stats(all_rows))
    log.info("Manifest written: %s (%d clips)", out, len(all_rows))
    return out


def dataset_stats(rows: list[dict]) -> dict:
    stats: dict = {"total_clips": len(rows), "total_hours": round(sum(r["duration"] for r in rows) / 3600, 2),
                   "by_split_lang": {}, "by_source": {}}
    for r in rows:
        k = f"{r['split']}/{r['lang']}"
        s = stats["by_split_lang"].setdefault(k, {"clips": 0, "hours": 0.0})
        s["clips"] += 1
        s["hours"] += r["duration"] / 3600
        t = stats["by_source"].setdefault(r["source"], {"clips": 0, "hours": 0.0, "speakers": set()})
        t["clips"] += 1
        t["hours"] += r["duration"] / 3600
        t["speakers"].add(r["speaker"])
    for v in stats["by_split_lang"].values():
        v["hours"] = round(v["hours"], 2)
    for v in stats["by_source"].values():
        v["hours"] = round(v["hours"], 2)
        v["speakers"] = len(v["speakers"])
    return stats
