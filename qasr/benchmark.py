"""Offline end-to-end benchmark on held-out clips: LID accuracy, ASR WER/CER,
per-stage latency, and example translations."""
from __future__ import annotations

import random
import re
import unicodedata

import numpy as np

from .audio_io import load_audio
from .config import Config, data_dir, results_dir
from .utils import get_logger, read_jsonl, write_json

log = get_logger("qasr.bench")


def _norm(text: str, lang: str) -> str:
    t = unicodedata.normalize("NFC", text or "").lower()
    t = re.sub(r"[।॥.,!?;:\"'()\[\]{}\-–—…]", " ", t)  # incl. Devanagari danda
    return re.sub(r"\s+", " ", t).strip()


def run_benchmark(cfg: Config, n_per_lang: int = 50, splits=("test", "test_ood")) -> dict:
    import jiwer

    from .live.pipeline import LivePipeline

    rows = [r for r in read_jsonl(data_dir(cfg) / "manifest.jsonl") if r["split"] in splits and r.get("text")]
    if not rows:
        raise FileNotFoundError("no test clips with transcripts in the manifest - run download first")
    rnd = random.Random(int(cfg.project.seed))
    pick = []
    for lang in ("hi", "en"):
        cand = [r for r in rows if r["lang"] == lang]
        rnd.shuffle(cand)
        pick += cand[:n_per_lang]
    pipe = LivePipeline(cfg)
    root = data_dir(cfg)
    lid_ok, refs, hyps, examples = 0, {"hi": [], "en": []}, {"hi": [], "en": []}, []
    for r in pick:
        wav = load_audio(root / r["path"])
        out = pipe.process(wav)
        lid_ok += int(out["lid_raw"] == r["lang"])
        if out["lang"] == r["lang"]:  # WER only when routed to the right recogniser
            refs[r["lang"]].append(_norm(r["text"], r["lang"]) or "-")
            hyps[r["lang"]].append(_norm(out["text"], r["lang"]) or "-")
        if len(examples) < 20:
            examples.append({"id": r["id"], "ref": r["text"], **out})
    res = {"n": len(pick), "lid_accuracy": lid_ok / len(pick), "latency_ms": pipe.timer.summary(), "asr": {},
           "examples": examples}
    for lang in ("hi", "en"):
        if refs[lang]:
            res["asr"][lang] = {"n": len(refs[lang]), "wer": float(jiwer.wer(refs[lang], hyps[lang])),
                                "cer": float(jiwer.cer(refs[lang], hyps[lang]))}
    write_json(results_dir(cfg) / "benchmark.json", res)
    lat = res["latency_ms"].get("total_processing", {})
    log.info("benchmark: LID acc %.3f | WER %s | compute mean %.0f ms p95 %.0f ms", res["lid_accuracy"],
             {k: round(v["wer"], 3) for k, v in res["asr"].items()}, lat.get("mean_ms", np.nan), lat.get("p95_ms", np.nan))
    return res
