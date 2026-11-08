"""Spoken language identification (Hindi vs English) for the live pipeline.

mode=quantum : Whisper-base encoder embedding -> trained hybrid quantum head
mode=whisper : Whisper-base's own language token (fallback before training)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..config import Config, _wrap, resolve
from ..features import MelStatsEmbedder, WhisperEmbedder
from ..models import load_lid_checkpoint
from ..utils import get_logger

log = get_logger("qasr.lid")


class LanguageIdentifier:
    def __init__(self, cfg: Config, device: torch.device):
        from transformers import WhisperForConditionalGeneration, WhisperTokenizer

        self.device = device
        name = cfg.features.whisper_model
        self.mode = cfg.live.lid_mode
        ck = resolve(cfg.live.lid_checkpoint)
        if self.mode == "quantum" and not Path(ck).exists():
            log.warning("LID checkpoint %s not found -> using Whisper language token instead. "
                        "Train first (train.ps1) to use the quantum LID.", ck)
            self.mode = "whisper"

        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.whisper = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=dtype).to(device).eval()
        self.tok = WhisperTokenizer.from_pretrained(name)
        self.sot = self.tok.convert_tokens_to_ids("<|startoftranscript|>")
        self.lang_ids = [self.tok.convert_tokens_to_ids("<|hi|>"), self.tok.convert_tokens_to_ids("<|en|>")]
        self.embedder = WhisperEmbedder(name, device, "mean", encoder=self.whisper.model.encoder)

        if self.mode == "quantum":
            self.head, meta = load_lid_checkpoint(ck, device)
            fcfg = _wrap(meta["config"]).features
            if fcfg.encoder == "whisper_base":
                self.embedder = WhisperEmbedder(fcfg.whisper_model, device, fcfg.pooling,
                                                encoder=self.whisper.model.encoder)
            else:
                self.embedder = MelStatsEmbedder(device)
            backend = getattr(getattr(self.head, "quantum", None), "backend_name", "n/a")
            log.info("LID: %s head (%s), quantum backend=%s, test acc=%s", meta["kind"], ck.name, backend,
                     meta.get("metrics", {}).get("test", {}).get("accuracy"))
        else:
            log.info("LID: whisper language-token fallback")

    @torch.inference_mode()
    def __call__(self, wav: np.ndarray) -> tuple[str, float]:
        if self.mode == "quantum":
            emb = torch.from_numpy(self.embedder([wav])).to(self.device)
            p_en = torch.softmax(self.head(emb), -1)[0, 1].item()
        else:
            feats = self.embedder._mel([wav]).to(self.device, self.whisper.dtype)
            dec = torch.tensor([[self.sot]], device=self.device)
            logits = self.whisper(input_features=feats, decoder_input_ids=dec).logits[0, -1, self.lang_ids]
            p_en = torch.softmax(logits.float(), -1)[1].item()
        return ("en", p_en) if p_en >= 0.5 else ("hi", 1.0 - p_en)
