# Field Defect Dispatch

Takes a technician's inspection photo and voice memo and produces a validated, prioritised maintenance ticket.

```
photo ──► YOLOv8s (+TTA) ─► per-class box statistics ─► logistic image-level scorer ─┐
                                                                                     ├─► evidence ─► cost-sensitive priority ─► Pydantic ticket
memo  ──► Faster-Whisper large-v3 (primed) ─► glossary repair ─► cues ───────────────┘               + hard safety floor        (+ review flag)
                                       (defect, severity, urgency, asset ID checked against a registry)
```

Two example runs, using real photos, the real detector and real Whisper on a synthesised memo. Every CODEBRIM photo
was in the final detector's training set, so these show that the pieces connect. They say nothing about accuracy.

```
Photo with exposed bars and spalling. Memo: "Found severe rebar corrosion near pump housing be four. This needs attention immediately."
  Photo: exposed_rebar 0.97, rebar_corrosion 0.98, spalling 0.99. Asset: B4 (from "be four").
  Ticket: P1, Structural Hazard, WO-STR-01 "Immediate shoring and rebar passivation", needs_review (priority confidence 0.50).

Clean concrete photo. Memo: "All clear on wall W2. No defect found."
  Photo: structural_crack 0.26, below the floor.
  Ticket: P4, WO-NON-01 "No action required", no review.
```

## What is real and what is synthetic

| Part | Data | Status |
|---|---|---|
| Detector (YOLOv8s) | Real: CODEBRIM bridge photos with boxes | trained, cross-validated, calibrated |
| Speech (Whisper large-v3) | Synthetic: text-to-speech plus noise | measured; real memos needed for absolute numbers |
| Voice cues, severity, priority labels | Synthetic | a latent "expert" policy; no real technician labels exist yet |
| Fusion and review routing | Hybrid: real detector behaviour, synthetic voice and labels | the machinery is tested; real-world accuracy is unknown |

No result below says how accurate the system is on real technicians' memos or on real priority decisions.

## Results

### Detector (CODEBRIM, YOLOv8s, 640 px, 5-fold CV grouped by blocks of 20 consecutive photos)

The set has 1,554 photos: 1,016 with annotated defects and 538 with no annotation. The 538 look like plain concrete
and serve as negatives, though a few may hold faint unlabelled stains. Classes are exposed rebar, rebar corrosion
(CODEBRIM's "corrosion stain"), spalling, structural crack (CODEBRIM's generic "crack") and efflorescence. CODEBRIM has
no labels for hairline crack or water leak, so those two classes are untrained.

The fusion stage needs image-level evidence (is this defect in the photo, and how sure), not pixel-accurate boxes, so
the detector work targeted that. Box-level quality is mediocre and barely moved: mean mAP50 is 0.377 ± 0.023 (mAP50-95 0.173) after
retraining, against 0.369 ± 0.030 before. CODEBRIM's boxes are large and loose.

Image-level results use pooled out-of-fold predictions. Each class threshold is the highest-precision point that still
reaches a per-class recall target of 0.90 to 0.95. "Clean flagged" is the share of the 538 clean photos flagged for any class.

| Step | mean AP | mean AUROC | clean photos flagged |
|---|---|---|---|
| baseline: 50 epochs, max box confidence | 0.861 | 0.926 | 45.4% |
| + test-time augmentation | 0.871 | 0.932 | 38.8% |
| + longer retrain (100 epochs, higher classification loss, lower box loss, cosine schedule) | 0.892 | 0.942 | 31.6% |
| + learned image-level scorer (final) | 0.898 | 0.943 | 22.5% |

| Final class results | AP | AUROC | precision at target recall | false-positive rate on clean photos | scorer calibration error |
|---|---|---|---|---|---|
| exposed rebar | 0.950 | 0.977 | 0.84 @ 0.95 | 2.6% | 0.012 |
| rebar corrosion | 0.910 | 0.943 | 0.79 @ 0.90 | 5.0% | 0.019 |
| spalling | 0.926 | 0.950 | 0.77 @ 0.90 | 6.9% | 0.023 |
| structural crack | 0.903 | 0.929 | 0.64 @ 0.95 | 16.5% | 0.022 |
| efflorescence | 0.802 | 0.915 | 0.72 @ 0.75 | 4.3% | 0.019 |

Defect photos are flagged for some class 96.3% of the time. The scorer is a per-class logistic regression over every
class's box statistics (max confidence, sum of the top three, box count, top-box area). It was evaluated
leave-one-fold-out across the detector's own CV folds. Its outputs are probabilities with calibration error of at most
0.023, so no separate calibration step is needed.

Limits: every metric comes from the fold models, while the shipped model trains on all the data and its scores shift
slightly. Prevalence here is 20 to 40% per class, and the calibrated probabilities carry that prevalence with them.
Structural crack is still the weakest class.

Two things did not help. A 960 px input tied with 640 px at image level for 4 to 5 times the compute, and the retraining
recipe did almost nothing for box-level mAP.

### Speech (Whisper large-v3, 80 synthetic memos, 10 dB noise)

| Variant | word error | domain-keyword error | asset-ID exact |
|---|---|---|---|
| plain | 0.229 | 3.6% | 90.0% |
| domain prompt + hotwords | 0.177 | 0.6% | 91.2% |
| prompt + glossary repair | 0.179 | 0.6% | 91.2% |

Text-to-speech is cleaner than real technicians, so the absolute numbers are optimistic. The gap between the first two
rows is the useful result. The repair step has shown no benefit so far, because clean audio rarely produces the
near-miss spellings it corrects.

### Fusion (nested CV grouped by site, then one look at an unseen holdout)

The pipeline replays the real detector's out-of-fold behaviour (`--bank`). Development cases draw photos from CV
folds 0 to 2 and holdout cases from folds 3 and 4, so no photo appears in both. Each seed uses 4,000 development and
1,500 holdout cases. Voice and labels are synthetic.

Final detector (retrained, with scorer), CV on the development cases, mean ± std over folds:

| System | accuracy | critical-miss | mean cost |
|---|---|---|---|
| deterministic rules | 0.519 | 0.141 | 0.866 |
| safety floor only | 0.594 | 1.000 | 1.334 |
| model, argmax, no floor | 0.738 | 0.614 ± 0.041 | 1.007 |
| model, argmax + floor | 0.614 | 0.614 ± 0.041 | 0.952 |
| model, cost-sensitive + floor | 0.534 ± 0.020 | 0.142 ± 0.046 | 0.705 |

On the unseen holdout, with the operating point chosen on out-of-fold data (4 seeds, mean ± sd), 1.8% ± 1.2% of
critical misses escape human review (per seed 2.6, 0.0, 3.3 and 1.4; the goal is 3% or less). The review rate is
38% ± 9%. Critical misses before review are 4.2% ± 1.8%, and 15% of tickets are dispatched two or more levels too urgent.

- Detector improvements are large at detector level and invisible at fusion level. The earlier detector variants
  gave 1.5% ± 1.1% (raw scores with isotonic calibration) and 1.1% ± 0.8% (TTA plus scorer) escaped misses at 33 to 39%
  review. Each holdout has about 100 critical cases, so seed noise of 1 to 2 points swamps the difference, and the
  synthetic voice and label noise sets the floor on review load.
- Plain argmax is the most accurate model and still misses about 60% of true P1 tickets, because P1 is rare.
- Testing caught two bugs. Dismissive wording could push a confident photo below its safety floor. The review rule
  also compared the voice only with the photo's top class, so photos showing several defects raised false "different
  defects" flags. After that fix, the review rate on the seed-0 holdout (real detector, no scorer) fell from 52% to 34%,
  though the fix also added two features.
- On the simulated fixture, a classifier given the true class, severity and asset reaches 0.856 accuracy against 0.761
  from the noisy evidence. The remaining gap comes from information lost upstream.

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
python scripts/train_detector_cv.py --data data/defects_v2 --groups-file groups_block.csv --weights yolov8s.pt \
    --epochs 100 --patience 25 --box 4.0 --cls 1.0 --cos-lr --close-mosaic 15 --out runs/detector_cv_v3
python scripts/eval_detector_oof.py --data data/defects_v2 --cv-dir runs/detector_cv_v3 --groups-file groups_block.csv \
    --augment --out reports/detector_image_level_v3.json
python scripts/fit_stacker.py --scores reports/detector_image_level_v3.npz     # writes weights/stacker.joblib, thresholds
python scripts/train_detector_cv.py --data data/defects_v2 --groups-file groups_block.csv --final-only \
    --epochs 100 --patience 25 --box 4.0 --cls 1.0 --cos-lr --close-mosaic 15 --out runs/detector_final_v3

python scripts/eval_asr.py --n 80 --size large-v3      # Whisper weights download on first use
python scripts/cv_fusion.py --bank --bank-npz reports/detector_final_oof.npz --bank-thresholds reports/stacker_thresholds.json \
    --bank-calibration none --out reports/fusion_cv_bank_stacker.json --op-out reports/operating_point_bank_stacker.json
python scripts/demo_pipeline.py --image data/defects_v2/images/image_0001489.jpg --say "Found severe rebar corrosion near pump housing be four"
```

Windows notes: the CODEBRIM zip stores 8 GB of offsets modulo 2³² (no zip64), and `scripts/prepare_codebrim.py` repairs
that. `faster-whisper` 1.2.1 and PyAV 19 are incompatible, so WAV files are decoded with `soundfile`. ctranslate2 finds
cuBLAS and cuDNN through the CUDA PyTorch build. The `reports/*_v2*` and `exp_*` files record earlier steps.

## Data licence and citation

CODEBRIM is licensed for non-commercial research and educational use only. It may not be redistributed, and models
trained on it may be shared only under the same terms. The data, weights and derived score files are therefore not
committed (`data/`, `weights/`, `reports/*.npz` and `reports/*.npy` are git-ignored). Please cite:

> Mundt, Majumder, Murali, Panetsos, Ramesh. *Meta-learning Convolutional Neural Architectures for Multi-target Concrete
> Defect Classification with the COncrete DEfect BRidge IMage Dataset.* CVPR 2019.

`docs/FEASIBILITY.md` covers the risks and remaining work.
