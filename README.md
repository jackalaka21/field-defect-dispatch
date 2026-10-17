# Field Defect Dispatch

Photo + voice memo from a field technician → validated, prioritised maintenance ticket.

```
photo ──► YOLOv8s ─► per-class thresholds ─► isotonic calibration ─┐
                                                                   ├─► evidence ─► cost-sensitive priority ─► Pydantic ticket
memo  ──► Faster-Whisper large-v3 (primed) ─► glossary repair ─────┘                + hard safety floor        (+ review flag)
                                       └─► cues: defect, severity, urgency, asset ID (checked against a registry)
```

Example (real photo, real detector, real Whisper on a synthesised memo; the photo was in the detector's training set,
so this checks the plumbing, not accuracy):

```
memo : "Found severe rebar corrosion near pump housing be four. This needs attention immediately."
→ asset B4 (from "be four"), priority P1, Structural Hazard, WO-STR-01 "Immediate shoring and rebar passivation",
  needs_review = true  ("photo and voice name different defects": the photo's top detection was spalling 0.95)
```

## What is real and what is not

| Part | Data | Status |
|---|---|---|
| Detector (YOLOv8s) | **Real**: CODEBRIM bridge photos with boxes | trained, cross-validated, calibrated |
| Speech (Whisper large-v3) | Synthetic: text-to-speech + noise | measured; real memos needed for absolute numbers |
| Voice cues, severity, priority labels | **Synthetic** | a latent "expert" policy; no real technician labels exist yet |
| Fusion / review routing | Hybrid: real detector behaviour + synthetic voice and labels | machinery validated, not real-world accuracy |

No result below says how accurate the system is on real technicians' memos or on real priority decisions.

## Results

### Detector — CODEBRIM, YOLOv8s, 640 px, 5-fold CV grouped by blocks of 20 consecutive photos

1,554 photos: 1,016 with annotated defects and 538 with no annotation (clean-looking concrete, used as negatives; a
few may hold faint unlabelled stains). Classes: exposed rebar, rebar corrosion (CODEBRIM's "corrosion stain"),
spalling, structural crack (CODEBRIM's generic "crack"), efflorescence. **Hairline crack and water leak have no
labels in CODEBRIM and are untrained.**

- **Box level** (IoU 0.5): mean mAP50 **0.369 ± 0.030**, mAP50-95 0.169. Weak; every fold was still improving at
  its last epoch (50). CODEBRIM boxes are large and loose, which makes tight localisation hard.
- **Image level** ("is this defect in the photo?"), which is what the pipeline consumes. Pooled out-of-fold scores;
  threshold = highest precision that still meets a per-class recall target (hazards 0.90–0.95):

| Class | AP | AUROC | precision at target recall | false-positive rate on clean photos |
|---|---|---|---|---|
| exposed rebar | 0.939 | 0.971 | 0.73 @ 0.95 | 9.3% |
| rebar corrosion | 0.896 | 0.938 | 0.76 @ 0.90 | 8.4% |
| spalling | 0.916 | 0.944 | 0.74 @ 0.90 | 10.4% |
| structural crack | 0.833 | 0.895 | 0.60 @ 0.95 | 30.5% |
| efflorescence | 0.723 | 0.888 | 0.53 @ 0.75 | 13.6% |

At these recall-targeted thresholds **45% of clean photos are flagged for some class** (98% of defect photos are).
The detector says reliably *what* is in a defect photo; as a clean/defective gate it is noisy, so the pipeline
leans on voice agreement and human review.
- **Calibration:** raw YOLO scores are not probabilities. A per-class isotonic curve fitted on out-of-fold scores
  cuts cross-fitted expected calibration error from 0.07–0.23 to 0.01–0.02. It encodes this dataset's class
  prevalence (20–40% of photos); refit on production photos before trusting absolute levels.
- **Resolution probe:** 960 px vs 640 px on one fold was a tie at image level (mixed ±0.02) for ~4–5× the compute, so
  640 px was kept.

### Speech — Whisper large-v3, 80 synthetic memos, 10 dB noise

| Variant | word error | domain-keyword error | asset-ID exact |
|---|---|---|---|
| plain | 0.229 | 3.6% | 90.0% |
| domain prompt + hotwords | 0.177 | 0.6% | 91.2% |
| prompt + glossary repair | 0.179 | 0.6% | 91.2% |

Text-to-speech is cleaner than real technicians, so absolute numbers are optimistic; the prompt effect is the
finding. The repair step has not been shown to help yet (clean audio rarely produces the near-miss spellings it fixes).

### Fusion — nested CV grouped by site, then one look at an unseen holdout

With the real detector's out-of-fold behaviour replayed (`--bank`; dev cases draw from CV folds 0–2, holdout cases
from folds 3–4, so no photo is shared), 4,000 dev + 1,500 holdout cases:

| System (CV, mean ± std) | accuracy | critical-miss | mean cost |
|---|---|---|---|
| deterministic rules | 0.512 ± 0.008 | 0.153 ± 0.080 | 0.914 |
| safety floor only | 0.600 ± 0.010 | 1.000 | 1.341 |
| model, argmax, no floor | 0.744 ± 0.014 | 0.591 ± 0.093 | 0.982 |
| model, argmax + floor | 0.628 ± 0.010 | 0.591 ± 0.093 | 0.915 |
| **model, cost-sensitive + floor** | 0.536 ± 0.010 | **0.168 ± 0.101** | **0.727 ± 0.066** |

- Accuracy and safety trade off: plain argmax is most accurate and misses ~60% of true P1 tickets because P1 is rare.
- Operating point chosen on pooled out-of-fold probabilities (cost scale 2×, review threshold 0.55): holdout
  **review rate 52%**, raw critical-miss 11.3%, **critical misses that escape human review 4.3%** (target 3%, missed),
  wasteful dispatch (≥2 levels too urgent) 13.5%.
- With the original fully simulated detector the same pipeline reached 2.8% escaped misses at a 28% review rate
  (`reports/fusion_cv.json`). That was too optimistic: the real detector's false alarms roughly double the review
  workload.
- Oracle check on the simulated fixture: with the true class, severity and asset a classifier reaches 0.856
  accuracy against 0.761 from noisy evidence. The gap is information lost upstream, not fusion model capacity.
- A unit test caught a real bug: dismissive wording could push a confident photo hit below its safety floor.

## Reproduce

Python 3.12 (3.14 has no `ctranslate2` wheel). GPU torch needs the CUDA index.

```bash
py -3.12 -m venv .venv && .venv/Scripts/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pytest -q

# detector (needs data/raw/codebrim/CODEBRIM_original_images.zip from Zenodo record 2620293, ~8.3 GB)
python scripts/prepare_codebrim.py --zip data/raw/codebrim/CODEBRIM_original_images.zip --out data/defects_v2 --with-negatives
python scripts/make_groups.py --data data/defects_v2 --block 20
python scripts/train_detector_cv.py --data data/defects_v2 --groups-file groups_block.csv --weights yolov8s.pt --epochs 50 --out runs/detector_cv_v2
python scripts/eval_detector_oof.py --data data/defects_v2 --cv-dir runs/detector_cv_v2 --groups-file groups_block.csv --out reports/detector_image_level_v2.json
python scripts/fit_calibration.py --scores reports/detector_image_level_v2.npz --eval reports/detector_image_level_v2.json
python scripts/train_detector_cv.py --data data/defects_v2 --groups-file groups_block.csv --final-only --epochs 60 --out runs/detector_final

python scripts/eval_asr.py --n 80 --size large-v3      # Whisper weights download on first use
python scripts/cv_fusion.py --bank --out reports/fusion_cv_bank.json --op-out reports/operating_point_bank.json
python scripts/demo_pipeline.py --image data/defects_v2/images/image_0001489.jpg --say "Found severe rebar corrosion near pump housing be four"
```

Windows notes: the CODEBRIM zip stores 8 GB of offsets modulo 2³² (no zip64), which `scripts/prepare_codebrim.py`
repairs. `faster-whisper` 1.2.1 and PyAV 19 disagree, so WAV files are decoded with `soundfile`; ctranslate2 finds
cuBLAS/cuDNN through the CUDA PyTorch build.

## Data licence and citation

CODEBRIM is licensed for **non-commercial research and educational use only**, may not be redistributed, and models
trained on it may be shared only under the same terms. The data, weights and derived score files are therefore not
committed (`data/`, `weights/`, `reports/*.npz` are git-ignored). Please cite:

> Mundt, Majumder, Murali, Panetsos, Ramesh. *Meta-learning Convolutional Neural Architectures for Multi-target Concrete
> Defect Classification with the COncrete DEfect BRidge IMage Dataset.* CVPR 2019.

See `docs/FEASIBILITY.md` for the risk assessment and remaining work.
