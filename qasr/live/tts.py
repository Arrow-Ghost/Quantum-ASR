"""Optional offline text-to-speech for the translation (Windows SAPI via pyttsx3).
While speaking, the pipeline ignores the microphone so it does not translate itself."""
from __future__ import annotations

import queue
import threading

from ..utils import get_logger

log = get_logger("qasr.tts")


class Speaker:
    def __init__(self):
        self.q: queue.Queue = queue.Queue()
        self.busy = threading.Event()
        self.ok = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.ok.wait(timeout=10)

    def _run(self):
        try:
            import pyttsx3

            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
        except Exception as e:
            log.warning("TTS unavailable (%s)", e)
            self.ok.set()
            return
        self.ok.set()

        def pick(lang: str):
            keys = ["hindi", "hi_in", "hi-in", "kalpana", "hemant"] if lang == "hi" else ["english", "en_", "en-"]
            for v in voices:
                blob = (v.name + " " + " ".join(str(x) for x in (getattr(v, "languages", []) or []))).lower()
                if any(k in blob for k in keys):
                    return v.id
            return None

        while True:
            text, lang = self.q.get()
            if text is None:
                break
            vid = pick(lang)
            if vid is None and lang == "hi":
                log.warning("No Hindi SAPI voice installed (Settings > Time & language > Speech > add Hindi).")
            if vid:
                engine.setProperty("voice", vid)
            self.busy.set()
            try:
                engine.say(text)
                engine.runAndWait()
            finally:
                self.busy.clear()

    def say(self, text: str, lang: str) -> None:
        if text:
            self.busy.set()
            self.q.put((text, lang))
