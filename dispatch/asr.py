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


def _expose_cuda_libs() -> None:
    """ctranslate2 needs cuBLAS/cuDNN 12; the CUDA build of PyTorch already ships them. Make them
    discoverable instead of requiring a separate CUDA toolkit install."""
    import os
    import sys

    if sys.platform != "win32":
        return
    try:
        import torch

        lib = Path(torch.__file__).parent / "lib"
        if lib.is_dir():
            os.add_dll_directory(str(lib))
            os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        pass


def load_audio(path: str | Path, sr: int = 16000):
    """Mono float32 at 16 kHz. Decoding WAV ourselves avoids a faster-whisper / PyAV 19 incompatibility
    (`metadata_errors`); other formats fall back to faster-whisper's own decoder."""
    import numpy as np

    if Path(path).suffix.lower() != ".wav":
        return str(path)
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd

    wav, native = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if native != sr:
        g = gcd(sr, native)
        wav = resample_poly(wav, sr // g, native // g).astype(np.float32)
    return wav


class Transcriber:
    def __init__(self, registry: dict[str, Asset], size: str = "small.en", device: str = "cuda",
                 compute_type: str = "int8_float16", beam_size: int = 5):
        _expose_cuda_libs()
        from faster_whisper import WhisperModel

        self.model = WhisperModel(size, device=device, compute_type=compute_type)
        self.prompt = build_prompt(registry)
        self.hotwords = " ".join(GLOSSARY)
        self.beam_size = beam_size

    def transcribe(self, audio: str | Path, use_prompt: bool = True) -> Transcript:
        kw = dict(beam_size=self.beam_size, vad_filter=True, condition_on_previous_text=False, temperature=0.0)
        if use_prompt:
            kw |= dict(initial_prompt=self.prompt, hotwords=self.hotwords)
        segments, _ = self.model.transcribe(load_audio(audio), **kw)
        segs = list(segments)
        text = " ".join(s.text.strip() for s in segs).strip()
        if not segs:
            return Transcript(text="", confidence=0.0)
        dur = sum(max(1e-3, s.end - s.start) for s in segs)
        avg_lp = sum(s.avg_logprob * max(1e-3, s.end - s.start) for s in segs) / dur
        no_speech = max(s.no_speech_prob for s in segs)
        return Transcript(text=text, confidence=float(max(0.0, min(1.0, math.exp(avg_lp) * (1.0 - no_speech)))))
