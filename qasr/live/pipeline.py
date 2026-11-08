"""Live speech translation:  mic -> VAD -> quantum LID -> ASR -> MT -> screen (+TTS).

Threads
  audio callback  : copies 32 ms frames into a queue (never blocks)
  main thread     : runs Silero VAD, cuts utterances
  worker thread   : LID -> ASR -> MT on the GPU, prints results
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ..audio_io import SR, load_audio
from ..config import Config, results_dir
from ..utils import Timer, get_logger, pick_device

log = get_logger("qasr.live")

C = {"hi": "\033[38;5;208m", "en": "\033[38;5;39m", "dim": "\033[2m", "b": "\033[1m", "r": "\033[0m"}
OTHER = {"hi": "en", "en": "hi"}


class LivePipeline:
    def __init__(self, cfg: Config, device: torch.device | None = None):
        from .asr import EnglishASR, HindiASR
        from .lid import LanguageIdentifier
        from .translate import Translator

        self.cfg = cfg
        lv = cfg.live
        self.device = device or pick_device(cfg.project.device)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        t0 = time.time()
        log.info("Loading models on %s ...", self.device)
        self.lid = LanguageIdentifier(cfg, self.device)
        self.asr = {"hi": HindiASR(cfg, self.device), "en": EnglishASR(cfg, self.device)}
        self.mt = {"hi": Translator(lv.mt_hi_en, "hi", "en", self.device, lv.mt_num_beams, lv.mt_max_length),
                   "en": Translator(lv.mt_en_hi, "en", "hi", self.device, lv.mt_num_beams, lv.mt_max_length)}
        self.min_conf = float(lv.lid_min_confidence)
        self.prev_lang = "hi"
        self.timer = Timer()
        self.warmup()
        log.info("All models ready in %.1fs", time.time() - t0)

    def warmup(self) -> None:
        rng = np.random.default_rng(0)
        wav = (rng.standard_normal(SR * 2) * 0.01).astype(np.float32)
        for lang in ("hi", "en"):
            try:
                self.asr[lang](wav)
            except Exception as e:
                log.warning("warm-up %s ASR: %s", lang, e)
        self.lid(wav)
        self.mt["hi"]("नमस्ते")
        self.mt["en"]("hello")
        self.timer = Timer()

    def process(self, wav: np.ndarray, force_lang: str | None = None) -> dict:
        sync = self.device.type == "cuda"
        t0 = time.perf_counter()
        with self.timer("lid", sync):
            lang, conf = self.lid(wav)
        raw_lang = lang
        if force_lang:
            lang = force_lang
        elif conf < self.min_conf:
            lang = self.prev_lang  # ambiguous -> keep the conversation's current language
        self.prev_lang = lang
        with self.timer(f"asr_{lang}", sync):
            text = self.asr[lang](wav)
        with self.timer(f"mt_{lang}_{OTHER[lang]}", sync):
            translation = self.mt[lang](text) if text else ""
        total = (time.perf_counter() - t0) * 1000
        self.timer.records.setdefault("total_processing", []).append(total)
        return {"lang": lang, "lid_raw": raw_lang, "lid_conf": round(conf, 4), "text": text,
                "translation": translation, "target": OTHER[lang], "audio_s": round(len(wav) / SR, 2),
                "processing_ms": round(total, 1)}


def _print_result(r: dict, endpoint_ms: float) -> None:
    src, tgt = r["lang"], r["target"]
    perceived = r["processing_ms"] + endpoint_ms
    print(f"{C['dim']}[{src.upper()}->{tgt.upper()} | LID {r['lid_conf']:.2f} | audio {r['audio_s']:.1f}s | "
          f"compute {r['processing_ms']:.0f} ms | ~{perceived:.0f} ms after you stopped]{C['r']}")
    print(f"  {C[src]}{r['text'] or '(nothing recognised)'}{C['r']}")
    if r["translation"]:
        print(f"  {C['b']}{C[tgt]}{r['translation']}{C['r']}")
    print(flush=True)


def _session_file(cfg: Config) -> Path | None:
    if not cfg.live.log_sessions:
        return None
    d = results_dir(cfg) / "live_sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"session_{datetime.now():%Y%m%d_%H%M%S}.jsonl"


def list_devices() -> None:
    import sounddevice as sd

    print(sd.query_devices())
    print(f"\nDefault input device: {sd.default.device[0]}")


def run_live(cfg: Config, file: str | None = None, speak: bool | None = None) -> None:
    from .vad import Segmenter

    if os.name == "nt":
        os.system("")  # enable ANSI colours in the Windows console
    pipe = LivePipeline(cfg)
    seg = Segmenter(cfg.live)
    endpoint_ms = float(cfg.live.min_silence_ms)
    speak = cfg.live.speak if speak is None else speak
    speaker = None
    if speak:
        from .tts import Speaker

        speaker = Speaker()
    sess = _session_file(cfg)
    chunk = int(cfg.live.chunk_samples)

    def handle(wav: np.ndarray) -> None:
        r = pipe.process(wav)
        _print_result(r, endpoint_ms)
        if sess:
            with open(sess, "a", encoding="utf-8") as f:
                f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), **r},
                                   ensure_ascii=False) + "\n")
        if speaker and r["translation"]:
            speaker.say(r["translation"], r["target"])

    if file:  # offline replay of a recording through the same streaming path
        x = load_audio(file)
        print(f"{C['b']}Replaying {file} ({len(x) / SR:.1f}s){C['r']}\n")
        for i in range(0, len(x) - chunk + 1, chunk):
            for s in seg.push(x[i:i + chunk]):
                handle(s)
        tail = seg.flush()
        if tail is not None:
            handle(tail)
        _print_latency(pipe)
        return

    import sounddevice as sd

    q_audio: queue.Queue = queue.Queue()
    q_seg: queue.Queue = queue.Queue()
    stop = threading.Event()

    def callback(indata, frames, t, status):
        if status:
            log.debug("audio status: %s", status)
        q_audio.put(indata[:, 0].copy())

    def worker():
        while not stop.is_set():
            try:
                wav = q_seg.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                handle(wav)
            except Exception as e:
                log.exception("processing failed: %s", e)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    dev = cfg.live.input_device
    print(f"{C['b']}Listening (device: {dev if dev is not None else 'default'}). Speak Hindi or English. "
          f"Ctrl+C to stop.{C['r']}\n", flush=True)
    try:
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=chunk, device=dev,
                            callback=callback, latency="low"):
            pending = np.zeros(0, dtype=np.float32)
            while True:
                pending = np.concatenate([pending, q_audio.get()])
                while len(pending) >= chunk:
                    frame, pending = pending[:chunk], pending[chunk:]
                    if speaker and speaker.busy.is_set():
                        seg.reset()
                        continue
                    for s in seg.push(frame):
                        q_seg.put(s)
    except KeyboardInterrupt:
        print("\nStopping ...")
    finally:
        stop.set()
        th.join(timeout=5)
        _print_latency(pipe)
        if sess:
            print(f"Session log: {sess}")


def _print_latency(pipe: LivePipeline) -> None:
    s = pipe.timer.summary()
    if not s:
        return
    print(f"\n{C['b']}Latency per stage (ms){C['r']}")
    for k, v in s.items():
        print(f"  {k:<18} n={v['n']:<4} mean={v['mean_ms']:7.1f}  p50={v['p50_ms']:7.1f}  p95={v['p95_ms']:7.1f}")
