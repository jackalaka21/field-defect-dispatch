"""Synthetic voice memos: local text-to-speech plus field noise, for measuring ASR on domain terms.

TTS speech is cleaner and more uniform than real technicians, so ASR numbers from these memos are an
optimistic upper bound. They are still useful for A/B comparisons (prompt vs no prompt, model size).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def tts_to_wav(text: str, path: Path, rate: int = 170, voice_index: int = 0) -> None:
    import pyttsx3

    eng = pyttsx3.init()
    voices = eng.getProperty("voices")
    if voices:
        eng.setProperty("voice", voices[voice_index % len(voices)].id)
    eng.setProperty("rate", rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    eng.save_to_file(text, str(path))
    eng.runAndWait()
    eng.stop()


def add_field_noise(wav: np.ndarray, sr: int, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Broadband noise plus a 100 Hz machinery hum at the requested signal-to-noise ratio."""
    wav = wav.astype(np.float64)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    sig_pow = float(np.mean(wav**2)) + 1e-12
    t = np.arange(len(wav)) / sr
    noise = 0.7 * rng.normal(size=len(wav)) + 0.5 * np.sin(2 * np.pi * 100 * t) + 0.25 * np.sin(2 * np.pi * 200 * t)
    noise *= np.sqrt(sig_pow / (10 ** (snr_db / 10)) / (float(np.mean(noise**2)) + 1e-12))
    out = wav + noise
    return (out / max(1.0, float(np.abs(out).max()))).astype(np.float32)


def make_memos(texts: list[str], outdir: Path, snr_db: float | None = 10.0, seed: int = 0) -> list[Path]:
    import soundfile as sf

    rng = np.random.default_rng(seed)
    paths: list[Path] = []
    for i, text in enumerate(texts):
        clean = outdir / f"memo_{i:04d}.wav"
        tts_to_wav(text, clean, voice_index=i % 2)
        if snr_db is not None:
            wav, sr = sf.read(str(clean))
            sf.write(str(clean), add_field_noise(wav, sr, snr_db, rng), sr)
        paths.append(clean)
    return paths
