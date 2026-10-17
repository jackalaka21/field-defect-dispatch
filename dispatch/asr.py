"""Faster-Whisper wrapper tuned for short, noisy, jargon-heavy field memos."""
from __future__ import annotations

import math
from pathlib import Path

from .assets import Asset
from .schemas import Transcript
from .taxonomy import GLOSSARY


def build_prompt(registry: dict[str, Asset]) -> str:
    """Priming text: Whisper imitates the style and vocabulary of its prompt."""
    ids = ", ".join(sorted(registry)[:12])
    return ("Concrete structure inspection memo. Terms: " + ", ".join(GLOSSARY)
            + f". Asset IDs such as {ids}. Example: Found severe rebar corrosion near pump housing B4.")


class Transcriber:
    def __init__(self, registry: dict[str, Asset], size: str = "small.en", device: str = "cuda",
                 compute_type: str = "int8_float16", beam_size: int = 5):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(size, device=device, compute_type=compute_type)
        self.prompt = build_prompt(registry)
        self.hotwords = " ".join(GLOSSARY)
        self.beam_size = beam_size

    def transcribe(self, audio: str | Path, use_prompt: bool = True) -> Transcript:
        kw = dict(beam_size=self.beam_size, vad_filter=True, condition_on_previous_text=False, temperature=0.0)
        if use_prompt:
            kw |= dict(initial_prompt=self.prompt, hotwords=self.hotwords)
        segments, _ = self.model.transcribe(str(audio), **kw)
        segs = list(segments)
        text = " ".join(s.text.strip() for s in segs).strip()
        if not segs:
            return Transcript(text="", confidence=0.0)
        dur = sum(max(1e-3, s.end - s.start) for s in segs)
        avg_lp = sum(s.avg_logprob * max(1e-3, s.end - s.start) for s in segs) / dur
        no_speech = max(s.no_speech_prob for s in segs)
        return Transcript(text=text, confidence=float(max(0.0, min(1.0, math.exp(avg_lp) * (1.0 - no_speech)))))
