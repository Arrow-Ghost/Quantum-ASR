"""Streaming utterance segmentation with Silero VAD (32 ms frames).

Latency knob: `min_silence_ms` - how long the speaker must pause before the
utterance is closed and sent to LID -> ASR -> MT.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch

from ..config import Config


class Segmenter:
    def __init__(self, live: Config, sr: int = 16000):
        from silero_vad import load_silero_vad

        self.model = load_silero_vad()
        self.sr = sr
        self.chunk = int(live.chunk_samples)
        ms = self.chunk / sr * 1000.0
        self.start_th = float(live.vad_start_threshold)
        self.end_th = float(live.vad_end_threshold)
        self.sil_chunks = max(1, round(float(live.min_silence_ms) / ms))
        self.pre_chunks = max(1, round(float(live.pre_roll_ms) / ms))
        self.min_chunks = max(1, round(float(live.min_speech_ms) / ms))
        self.max_chunks = max(2, round(float(live.max_segment_s) * 1000.0 / ms))
        self.keep_tail = max(1, round(100.0 / ms))  # keep ~100 ms of trailing silence
        self.reset()

    def reset(self) -> None:
        self.model.reset_states()
        self.pre: deque = deque(maxlen=self.pre_chunks)
        self.buf: list[np.ndarray] = []
        self.in_speech = False
        self.start_count = 0
        self.silence = 0
        self.speech_chunks = 0
        self.last_prob = 0.0

    @torch.inference_mode()
    def prob(self, chunk: np.ndarray) -> float:
        self.last_prob = float(self.model(torch.from_numpy(chunk.astype(np.float32)), self.sr).item())
        return self.last_prob

    def _close(self) -> np.ndarray | None:
        drop = max(0, self.silence - self.keep_tail)
        chunks = self.buf[:len(self.buf) - drop] if drop else self.buf
        seg = np.concatenate(chunks) if chunks and self.speech_chunks >= self.min_chunks else None
        self.buf, self.in_speech, self.start_count, self.silence, self.speech_chunks = [], False, 0, 0, 0
        self.pre.clear()
        return seg

    def push(self, chunk: np.ndarray) -> list[np.ndarray]:
        """Feed exactly `chunk_samples` samples; returns finished utterances."""
        out = []
        p = self.prob(chunk)
        if not self.in_speech:
            self.pre.append(chunk)
            self.start_count = self.start_count + 1 if p >= self.start_th else 0
            if self.start_count >= 2:
                self.in_speech, self.buf, self.silence = True, list(self.pre), 0
                self.speech_chunks = self.start_count
        else:
            self.buf.append(chunk)
            if p < self.end_th:
                self.silence += 1
            else:
                self.silence = 0
                self.speech_chunks += 1
            if self.silence >= self.sil_chunks or len(self.buf) >= self.max_chunks:
                seg = self._close()
                if seg is not None:
                    out.append(seg)
        return out

    def flush(self) -> np.ndarray | None:
        return self._close() if self.in_speech else None
