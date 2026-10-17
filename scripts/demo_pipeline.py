"""End-to-end demo on real components: a photo + a spoken memo -> ticket JSON.

    python scripts/demo_pipeline.py --image data/defects_v2/images/image_0000115.jpg \
        --say "Found severe rebar corrosion near pump housing B4"

The detector, thresholds and calibration come from the CODEBRIM-trained model. The fusion model was fitted on the
hybrid fixture (real detector behaviour + synthetic voice and labels), so priorities demonstrate the plumbing and
are not a validated judgement.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402

from dispatch.asr import Transcriber  # noqa: E402
from dispatch.assets import default_registry  # noqa: E402
from dispatch.calibration import Calibrator  # noqa: E402
from dispatch.fusion import PriorityModel, load_operating_point  # noqa: E402
from dispatch.pipeline import DispatchPipeline  # noqa: E402
from dispatch.synth.audio import make_memos  # noqa: E402
from dispatch.vision import DefectDetector, load_thresholds  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--say", help="text to synthesise as the technician's memo")
    ap.add_argument("--audio", type=Path, help="an existing .wav memo instead of --say")
    ap.add_argument("--weights", default="weights/detector.pt")
    ap.add_argument("--thresholds", default="reports/detector_thresholds.json")
    ap.add_argument("--whisper", default="large-v3")
    a = ap.parse_args()

    audio = a.audio or make_memos([a.say], Path("data/demo_audio"), snr_db=15.0)[0]
    reg = default_registry()
    cost, cfg = load_operating_point("reports/operating_point_bank.json")
    model = PriorityModel(joblib.load("weights/fusion_bank.joblib"), cost, cfg)
    pipe = DispatchPipeline(DefectDetector(a.weights, load_thresholds(a.thresholds),
                                                      calibrator=Calibrator.load("reports/detector_calibration.json")), Transcriber(reg, size=a.whisper),
                            model=model, registry=reg, cfg=cfg)
    ev = pipe.evidence(a.image, audio)
    print("transcript cues:", ev.cues.model_dump())
    print("detections     :", [(d.cls.value, round(d.conf, 2)) for d in ev.detections])
    print(pipe.run(a.image, audio, "T-DEMO-1").model_dump_json(indent=2))


if __name__ == "__main__":
    main()
