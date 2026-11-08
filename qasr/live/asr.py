"""Speech recognition: AI4Bharat IndicConformer / IndicWav2Vec for Hindi,
(distil-)Whisper for English. Each is loaded once and kept warm on the GPU."""
from __future__ import annotations

import numpy as np
import torch

from ..config import Config
from ..utils import add_cuda_dll_dirs, get_logger

log = get_logger("qasr.asr")


class HindiASR:
    def __init__(self, cfg: Config, device: torch.device):
        self.kind = cfg.live.hindi_asr
        self.device = device
        if self.kind == "indicconformer":
            add_cuda_dll_dirs()  # onnxruntime-gpu needs CUDA/cuDNN DLLs (shipped with torch)
            from transformers import AutoModel

            self.decoding = cfg.live.hindi_asr_decoding
            self.model = AutoModel.from_pretrained(cfg.models.indicconformer, trust_remote_code=True)
            try:
                import onnxruntime as ort

                log.info("IndicConformer loaded (onnxruntime providers available: %s)", ort.get_available_providers())
            except Exception:
                pass
        elif self.kind == "indicwav2vec":
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

            name = cfg.models.indicwav2vec
            self.proc = Wav2Vec2Processor.from_pretrained(name)
            dtype = torch.float16 if device.type == "cuda" else torch.float32
            self.model = Wav2Vec2ForCTC.from_pretrained(name, torch_dtype=dtype).to(device).eval()
            log.info("IndicWav2Vec-Hindi loaded on %s", device)
        else:
            raise ValueError(f"unknown live.hindi_asr {self.kind}")

    @torch.inference_mode()
    def __call__(self, wav: np.ndarray) -> str:
        if self.kind == "indicconformer":
            out = self.model(torch.from_numpy(wav).float().unsqueeze(0), "hi", self.decoding)
            if isinstance(out, (list, tuple)):
                out = out[0] if out else ""
            return str(out).strip()
        x = self.proc(wav, sampling_rate=16000, return_tensors="pt").input_values.to(self.device, self.model.dtype)
        ids = self.model(x).logits.argmax(-1)
        return self.proc.batch_decode(ids)[0].strip()


class EnglishASR:
    def __init__(self, cfg: Config, device: torch.device):
        self.backend = cfg.live.english_asr_backend
        self.device = device
        if self.backend == "faster_whisper":
            add_cuda_dll_dirs()
            from faster_whisper import WhisperModel

            self.model = WhisperModel(cfg.models.faster_whisper_en, device="cuda" if device.type == "cuda" else "cpu",
                                      compute_type="float16" if device.type == "cuda" else "int8")
        else:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            name = cfg.live.english_asr
            self.proc = WhisperProcessor.from_pretrained(name)
            self.dtype = torch.float16 if device.type == "cuda" else torch.float32
            try:
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    name, torch_dtype=self.dtype, attn_implementation="sdpa").to(device).eval()
            except Exception:
                self.model = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=self.dtype).to(device).eval()
        log.info("English ASR: %s (%s)", cfg.live.english_asr if self.backend != "faster_whisper"
                 else cfg.models.faster_whisper_en, self.backend)

    @torch.inference_mode()
    def __call__(self, wav: np.ndarray) -> str:
        if self.backend == "faster_whisper":
            segs, _ = self.model.transcribe(wav, language="en", beam_size=1, vad_filter=False,
                                            condition_on_previous_text=False, without_timestamps=True)
            return " ".join(s.text.strip() for s in segs).strip()
        feats = self.proc(wav, sampling_rate=16000, return_tensors="pt").input_features.to(self.device, self.dtype)
        ids = self.model.generate(feats, language="en", task="transcribe", num_beams=1, max_new_tokens=160)
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
