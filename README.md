# Field Defect Dispatch

Photo + voice memo from a field technician → validated, prioritised maintenance ticket.

```
photo ──► YOLOv8s (+TTA) ─► per-class box statistics ─► logistic image-level scorer ─┐
                                                                                     ├─► evidence ─► cost-sensitive priority ─► Pydantic ticket
memo  ──► Faster-Whisper large-v3 (primed) ─► glossary repair ─► cues ───────────────┘               + hard safety floor        (+ review flag)
                                       (defect, severity, urgency, asset ID checked against a registry)
```

Examples (real photos, real detector, real Whisper on a synthesised memo; every CODEBRIM photo was in the final
detector's training set, so these check the plumbing, not accuracy):

```
photo with exposed bars + spalling, memo: "Found severe rebar corrosion near pump housing be four. This needs attention immediately."
→ photo: exposed_rebar 0.97, rebar_corrosion 0.98, spalling 0.99 · asset B4 (from "be four")
→ P1, Structural Hazard, WO-STR-01 "Immediate shoring and rebar passivation", needs_review (priority confidence 0.50)

clean concrete photo, memo: "All clear on wall W2. No defect found."
→ photo: structural_crack 0.26 (below the floor) · P4, WO-NON-01 "No action required", no review
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

The pipeline consumes image-level evidence ("is this defect in the photo, how sure"), not pixel-accurate boxes, so
that is what was optimised. Box-level quality is mediocre and barely changed: mean mAP50 0.377 ± 0.023
(mAP50-95 0.173) after retraining vs 0.369 ± 0.030 before. CODEBRIM boxes are large and loose.

Image-level results on pooled out-of-fold data (threshold = highest precision that still meets a per-class recall
target of 0.90–0.95; "clean flagged" = share of the 538 clean photos flagged for any class):

| Step | mean AP | mean AUROC | clean photos flagged |
|---|---|---|---|
| baseline: 50 epochs, max box confidence | 0.861 | 0.926 | 45.4% |
| + test-time augmentation | 0.871 | 0.932 | 38.8% |
| + longer retrain (100 epochs, higher classification / lower box loss, cosine schedule) | 0.892 | 0.942 | 31.6% |
| + learned image-level scorer (**final**) | **0.898** | **0.943** | **22.5%** |

| Final class results | AP | AUROC | precision at target recall | false-positive rate on clean photos | scorer calibration error |
|---|---|---|---|---|---|
| exposed rebar | 0.950 | 0.977 | 0.84 @ 0.95 | 2.6% | 0.012 |
| rebar corrosion | 0.910 | 0.943 | 0.79 @ 0.90 | 5.0% | 0.019 |
| spalling | 0.926 | 0.950 | 0.77 @ 0.90 | 6.9% | 0.023 |
| structural crack | 0.903 | 0.929 | 0.64 @ 0.95 | 16.5% | 0.022 |
| efflorescence | 0.802 | 0.915 | 0.72 @ 0.75 | 4.3% | 0.019 |

Defect photos are flagged for some class 96.3% of the time. The learned scorer is a per-class logistic regression
over every class's box statistics (max confidence, sum of top-3, box count, top-box area), evaluated by
leave-one-fold-out over the detector's own CV folds. Its outputs are probabilities (calibration error ≤ 0.023), so
no separate calibration step is needed. Caveats: all metrics are from the fold models (the shipped model is trained
on all data and its scores shift slightly); prevalence here is 20–40% per class and calibrated probabilities encode
it; structural crack remains the weak class.

Things that did not help: 960 px input vs 640 px (a tie at image level for 4–5× the compute); box-level gains from
retraining (negligible).

### Speech — Whisper large-v3, 80 synthetic memos, 10 dB noise

| Variant | word error | domain-keyword error | asset-ID exact |
|---|---|---|---|
| plain | 0.229 | 3.6% | 90.0% |
| domain prompt + hotwords | 0.177 | 0.6% | 91.2% |
| prompt + glossary repair | 0.179 | 0.6% | 91.2% |

Text-to-speech is cleaner than real technicians, so absolute numbers are optimistic; the prompt effect is the
finding. The repair step has not been shown to help yet (clean audio rarely produces the near-miss spellings it fixes).

### Fusion — nested CV grouped by site, then one look at an unseen holdout

The real detector's out-of-fold behaviour is replayed (`--bank`): dev cases draw photos from CV folds 0–2, holdout
cases from folds 3–4, so no photo is shared. 4,000 dev + 1,500 holdout cases per seed. Voice and labels are synthetic.

Final detector (retrained + scorer), CV on the dev cases (mean ± std over folds):

| System | accuracy | critical-miss | mean cost |
|---|---|---|---|
| deterministic rules | 0.519 | 0.141 | 0.866 |
| safety floor only | 0.594 | 1.000 | 1.334 |
| model, argmax, no floor | 0.738 | 0.614 ± 0.041 | 1.007 |
| model, argmax + floor | 0.614 | 0.614 ± 0.041 | 0.952 |
| **model, cost-sensitive + floor** | 0.534 ± 0.020 | **0.142 ± 0.046** | **0.705** |

Unseen holdout at an operating point chosen on out-of-fold data (4 seeds, mean ± sd): **1.8% ± 1.2% of critical
misses escape human review** (per seed 2.6, 0.0, 3.3, 1.4; goal ≤ 3%), at a **review rate of 38% ± 9%**. Raw
critical-miss before review is 4.2% ± 1.8%; wasteful dispatch (≥ 2 levels too urgent) 15%.

- The detector improvements are large at detector level but **not distinguishable at fusion level**: the earlier
  detector variants gave 1.5% ± 1.1% (raw+isotonic) and 1.1% ± 0.8% (TTA+scorer) escaped at 33–39% review. Each holdout
  has only ~100 critical cases, so seed noise (±1–2 points) swamps the difference; the synthetic voice and label noise
  set the floor on review load.
- Accuracy and safety trade off: plain argmax is most accurate yet misses ~60% of true P1 tickets because P1 is rare.
- Bugs caught by testing: dismissive wording could push a confident photo below its safety floor, and the review rule
  compared the voice only with the photo's top class, so multi-defect photos raised false "different defects" flags
  (fixing it cut review load markedly).
- Oracle check on the simulated fixture: with the true class, severity and asset a classifier reaches 0.856
  accuracy vs 0.761 from noisy evidence, so the remaining gap is information lost upstream, not model capacity.

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

Windows notes: the CODEBRIM zip stores 8 GB of offsets modulo 2³² (no zip64), which `scripts/prepare_codebrim.py`
repairs. `faster-whisper` 1.2.1 and PyAV 19 disagree, so WAV files are decoded with `soundfile`; ctranslate2 finds
cuBLAS/cuDNN through the CUDA PyTorch build. `reports/*_v2*` and `exp_*` files document earlier steps.

## Data licence and citation

CODEBRIM is licensed for **non-commercial research and educational use only**, may not be redistributed, and models
trained on it may be shared only under the same terms. The data, weights and derived score files are therefore not
committed (`data/`, `weights/`, `reports/*.npz` and `reports/*.npy` are git-ignored). Please cite:

> Mundt, Majumder, Murali, Panetsos, Ramesh. *Meta-learning Convolutional Neural Architectures for Multi-target Concrete
> Defect Classification with the COncrete DEfect BRidge IMage Dataset.* CVPR 2019.

See `docs/FEASIBILITY.md` for the risk assessment and remaining work.
