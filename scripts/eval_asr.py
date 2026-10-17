"""A/B the speech stage on synthetic noisy memos: plain Whisper vs prompted Whisper vs prompted + repair.

Reports word error rate, domain keyword error rate, and asset-ID exact match. TTS audio is cleaner than
real technicians, so absolute numbers are optimistic; the *differences* between variants are the point.

    python scripts/eval_asr.py --n 80 --size small.en --snr 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dispatch.assets import default_registry  # noqa: E402
from dispatch.metrics import keyword_error_rate, word_error_rate  # noqa: E402
from dispatch.synth.audio import make_memos  # noqa: E402
from dispatch.synth.cases import generate_cases  # noqa: E402
from dispatch.taxonomy import GLOSSARY  # noqa: E402
from dispatch.text import extract_cues, repair  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--size", default="small.en")
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--audio-dir", type=Path, default=Path("data/synth_audio"))
    ap.add_argument("--out", default="reports/asr_ab.json")
    a = ap.parse_args()

    from dispatch.asr import Transcriber

    reg = default_registry()
    cases = generate_cases(a.n, seed=123, asr_error=0.0)  # error-free text: the *audio* carries the difficulty
    paths = make_memos([c.clean_text for c in cases], a.audio_dir, snr_db=a.snr)
    asr = Transcriber(reg, size=a.size, device=a.device)
    kw = set(GLOSSARY)
    variants = {"plain": lambda p: asr.transcribe(p, use_prompt=False).text,
                "prompted": lambda p: asr.transcribe(p, use_prompt=True).text,
                "prompted+repair": lambda p: repair(asr.transcribe(p, use_prompt=True).text)}
    report = {}
    for name, fn in variants.items():
        wer, miss, tot, asset_ok = [], 0, 0, 0
        for c, p in zip(cases, paths):
            hyp = fn(p)
            wer.append(word_error_rate(c.clean_text, hyp))
            m, t = keyword_error_rate(c.clean_text, hyp, kw)
            miss, tot = miss + m, tot + t
            asset_ok += extract_cues(hyp, reg).asset_id == c.asset.asset_id
        report[name] = {"wer": sum(wer) / len(wer), "keyword_error_rate": miss / max(1, tot),
                        "asset_id_exact": asset_ok / len(cases)}
        print(f"{name:16s} WER={report[name]['wer']:.3f}  KER={report[name]['keyword_error_rate']:.3f}  "
              f"asset-ID={report[name]['asset_id_exact']:.3f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"model": a.size, "snr_db": a.snr, "n": a.n, "variants": report,
                                       "note": "TTS audio: optimistic absolute numbers"}, indent=2))


if __name__ == "__main__":
    main()
