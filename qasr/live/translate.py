"""AI4Bharat IndicTrans2 (distilled 200M) translation, Hindi <-> English."""
from __future__ import annotations

import re
import unicodedata

import torch

from ..utils import get_logger

log = get_logger("qasr.mt")
CODES = {"hi": "hin_Deva", "en": "eng_Latn"}


def _load_indic_processor():
    for mod in ("IndicTransToolkit.processor", "IndicTransToolkit"):
        try:
            m = __import__(mod, fromlist=["IndicProcessor"])
            return m.IndicProcessor(inference=True)
        except Exception:
            continue
    return None


def _basic_preprocess(text: str, src: str, tgt: str) -> str:
    """Minimal stand-in for IndicProcessor (Hindi is already Devanagari)."""
    t = unicodedata.normalize("NFC", text)
    t = re.sub(r"\s+", " ", t).strip()
    return f"{src} {tgt} {t}"


class Translator:
    def __init__(self, model_name: str, src: str, tgt: str, device: torch.device, num_beams: int = 1,
                 max_length: int = 256):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.src, self.tgt = CODES[src], CODES[tgt]
        self.device, self.num_beams, self.max_length = device, int(num_beams), int(max_length)
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True,
                                                           torch_dtype=dtype).to(device).eval()
        self.ip = _load_indic_processor()
        self.use_cache = True
        log.info("MT %s -> %s: %s (IndicTransToolkit: %s)", src, tgt, model_name, "yes" if self.ip else "fallback")

    def _generate(self, inputs):
        return self.model.generate(**inputs, use_cache=self.use_cache, min_length=0, max_length=self.max_length,
                                   num_beams=self.num_beams, num_return_sequences=1)

    @torch.inference_mode()
    def __call__(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        batch = (self.ip.preprocess_batch([text], src_lang=self.src, tgt_lang=self.tgt) if self.ip
                 else [_basic_preprocess(text, self.src, self.tgt)])
        inputs = self.tok(batch, truncation=True, padding="longest", return_tensors="pt",
                          return_attention_mask=True).to(self.device)
        try:
            gen = self._generate(inputs)
        except Exception as e:  # some transformers versions break IndicTrans2's KV cache
            if not self.use_cache:
                raise
            log.warning("generate(use_cache=True) failed (%s); retrying without KV cache", str(e)[:120])
            self.use_cache = False
            gen = self._generate(inputs)
        dec = self.tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if self.ip:
            dec = self.ip.postprocess_batch(dec, lang=self.tgt)
        return dec[0].strip()
