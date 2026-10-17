import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
pytest.importorskip("scipy")

from dispatch.asr import build_prompt, load_audio  # noqa: E402
from dispatch.assets import default_registry  # noqa: E402


def test_load_audio_resamples_to_16k_mono_float32(tmp_path):
    p = tmp_path / "a.wav"
    stereo = np.stack([np.sin(np.linspace(0, 50, 8000)), np.zeros(8000)], axis=1)
    sf.write(str(p), stereo, 8000)
    out = load_audio(p)
    assert out.dtype == np.float32 and out.ndim == 1
    assert abs(len(out) - 16000) <= 2  # 1 s at 8 kHz -> 16 kHz


def test_non_wav_is_left_to_the_library_decoder(tmp_path):
    assert load_audio(tmp_path / "memo.m4a") == str(tmp_path / "memo.m4a")


def test_prompt_contains_domain_terms_and_ids():
    prompt = build_prompt(default_registry())
    assert "rebar" in prompt and "B4" in prompt
