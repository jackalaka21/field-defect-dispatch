import numpy as np
import pytest

from dispatch.pipeline import DispatchPipeline
from dispatch.schemas import Detection, Transcript
from dispatch.synth.audio import add_field_noise
from dispatch.taxonomy import DefectClass, Priority
from dispatch.vision import iou, merge_detections, tile_windows


@pytest.mark.parametrize("w,h", [(640, 640), (1280, 720), (4032, 3024), (700, 650)])
def test_tiles_cover_the_whole_image_within_bounds(w, h):
    wins = tile_windows(w, h, 640, 0.2)
    covered = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in wins:
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
        covered[y0:y1, x0:x1] = True
    assert covered.all()


def test_tile_overlap_validation():
    with pytest.raises(ValueError):
        tile_windows(100, 100, 50, 1.0)


def test_iou_and_class_wise_nms():
    a = (0.1, 0.1, 0.5, 0.5)
    assert iou(a, a) == pytest.approx(1.0)
    assert iou(a, (0.6, 0.6, 0.9, 0.9)) == 0.0
    dets = [Detection(cls=DefectClass.SPALLING, conf=0.9, box=a),
            Detection(cls=DefectClass.SPALLING, conf=0.6, box=(0.11, 0.11, 0.5, 0.5)),  # duplicate -> dropped
            Detection(cls=DefectClass.EXPOSED_REBAR, conf=0.7, box=a)]  # other class -> kept
    kept = merge_detections(dets)
    assert [(d.cls, d.conf) for d in kept] == [(DefectClass.SPALLING, 0.9), (DefectClass.EXPOSED_REBAR, 0.7)]


def test_noise_hits_requested_snr():
    rng = np.random.default_rng(0)
    sr = 16000
    wav = np.sin(2 * np.pi * 440 * np.arange(sr) / sr) * 0.5
    noisy = add_field_noise(wav, sr, 10.0, rng)
    scale = noisy.std() / (wav.std() + 1e-9)
    noise = noisy / scale - wav
    snr = 10 * np.log10(np.mean(wav**2) / np.mean(noise**2))
    assert 8.0 < snr < 12.0


class FakeDetector:
    def detect(self, image):
        return [Detection(cls=DefectClass.EXPOSED_REBAR, conf=0.91, box=(0.2, 0.2, 0.6, 0.6))]


class FakeASR:
    def __init__(self, text, conf=0.9):
        self.t = Transcript(text=text, confidence=conf)

    def transcribe(self, audio):
        return self.t


def test_pipeline_headline_example_end_to_end_without_models():
    pipe = DispatchPipeline(FakeDetector(), FakeASR("Found severe rebar corrosion near pump housing be four"))
    t = pipe.run("photo.jpg", "memo.wav", "T-100")
    assert t.priority in (Priority.P1, Priority.P2)
    assert t.asset_id == "B4"
    assert t.category.value == "Structural Hazard"
    assert t.work_order_id in {"WO-STR-01", "WO-STR-02"}


def test_pipeline_flags_conflicts_and_unknown_assets_for_review():
    pipe = DispatchPipeline(FakeDetector(), FakeASR("minor efflorescence somewhere", conf=0.9))
    t = pipe.run("photo.jpg", "memo.wav", "T-101")
    assert t.needs_review
    assert any("asset" in r for r in t.review_reasons)
    assert any("different defects" in r for r in t.review_reasons)
    assert t.priority.rank <= Priority.P2.rank  # the confident photo still pins the floor
