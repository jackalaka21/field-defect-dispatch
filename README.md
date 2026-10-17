# Field Defect Dispatch

Photo + voice memo from a field technician → validated, prioritised maintenance ticket.

```
photo ──► YOLOv8 (tiled, per-class thresholds) ─┐
                                                ├─► evidence ─► cost-sensitive priority ─► Pydantic ticket ─► (Plane API)
memo  ──► Faster-Whisper (primed) ─► repair ────┘                + hard safety floor
                                       └─► cues: defect, severity, urgency, asset ID (registry-checked)
```

## What is built

| Stage | Module | Accuracy levers |
|---|---|---|
| Detection | `dispatch/vision.py`, `scripts/train_detector_cv.py` | site-grouped k-fold fine-tuning, tiled inference for small defects, **per-class thresholds tuned on pooled out-of-fold predictions to a recall target** (hazards 0.90–0.95) |
| Speech | `dispatch/asr.py`, `dispatch/text.py`, `dispatch/assets.py` | domain prompt + hotwords, VAD, glossary repair, spoken-ID resolution (`be four` → `B4`) validated against an asset registry |
| Fusion | `dispatch/fusion.py` | cross-modal agreement/conflict features, calibrated probabilities, expected-cost decision, **safety floor the model can never go below** |
| Ticket | `dispatch/writer.py`, `dispatch/schemas.py` | Pydantic schema, work orders chosen from a fixed catalog, optional LLM finaliser that is retried on invalid output and cannot lower priority or invent a work order |
| Evaluation | `dispatch/metrics.py`, `scripts/` | critical-miss rate, cost, escaped-miss rate after human review, WER and domain keyword error rate |

## Results so far — read this before quoting any number

**Everything below was produced on a synthetic fixture** (`dispatch/synth/`). It validates the machinery
(leakage-safe CV, cost-sensitive decisions, safety floor, review routing). It says nothing about real-world
accuracy, because no real labelled photos or technician memos have been used yet.

Fusion stage, 5-fold nested CV grouped by site (4,000 dev cases), then one look at 1,500 cases from unseen sites:

| System (CV, mean ± std) | accuracy | critical-miss | mean cost |
|---|---|---|---|
| deterministic rules | 0.596 ± 0.007 | 0.196 ± 0.031 | 0.949 |
| safety floor only | 0.649 ± 0.013 | 1.000 | 1.532 |
| model, argmax, no floor | 0.761 ± 0.014 | 0.596 ± 0.050 | 0.998 |
| model, argmax + floor | 0.689 ± 0.014 | 0.596 ± 0.050 | 0.927 |
| **model, cost-sensitive + floor** | 0.593 ± 0.024 | **0.119 ± 0.016** | **0.618 ± 0.034** |

- Accuracy and safety trade off. Plain argmax is the most accurate and misses ~60% of true P1s because P1 is rare.
  The cost-sensitive decision gives up accuracy to cut the critical-miss rate five-fold.
- Oracle ceiling: given the *true* class, severity and asset, a classifier reaches 0.856 argmax accuracy against
  0.761 from the pipeline's noisy evidence. The ~10-point gap is information lost upstream (the simulated detector
  fires on 68% of cases, the voice names the true defect on 80%), so the remaining headroom is in the detector and
  the speech stage, not in the fusion model (logistic regression and gradient boosting tie).
- The operating point (cost scale 3×, review threshold 0.5) was chosen on pooled out-of-fold probabilities by
  minimising review workload plus wasteful dispatch subject to escaped critical misses ≤ 2%. On the unseen-site
  holdout: raw critical-miss 9.2%, **escaped critical-miss 2.8%** (flagged tickets go to a human), review rate 28.5%,
  wasteful (≥2 levels too urgent) 8.9%. Probabilities are well calibrated (ECE 0.02–0.03).
- Real findings from testing, not from tuning: the rule baseline originally let dismissive wording push a confident
  photo hit below its safety floor; a unit test caught it and `rule_priority` now clamps to the floor.

The detector CV path was smoke-tested on a generated toy image set (no downloads, trained from scratch, offline).
Those numbers are meaningless; the run only proves the fold/predict/match/threshold path works.

## Reproduce

Python 3.12 (3.14 lacks `ctranslate2` wheels). GPU torch needs the CUDA index.

```bash
py -3.12 -m venv .venv && .venv/Scripts/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pytest -q
python scripts/cv_fusion.py                      # writes reports/fusion_cv.json, reports/operating_point.json
python scripts/eval_asr.py --n 80                # needs Whisper weights (downloaded on first use)
python scripts/train_detector_cv.py --data data/defects --weights yolov8s.pt --folds 5 --final
```

See `docs/FEASIBILITY.md` for the feasibility assessment, risks and the order of remaining work.
