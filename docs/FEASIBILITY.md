# Feasibility assessment

## Verdict

Feasible on the target machine (RTX 3070 Ti Laptop, 8 GB) and now demonstrated end to end on real components: a
CODEBRIM-trained YOLOv8s detector, Whisper large-v3 and the fusion/ticket stage run on a real photo plus a spoken
memo. **What is not demonstrated is real-world accuracy**: there are no real technician memos and no expert priority
labels, so the voice and priority parts of every number are synthetic. See the README for what is real.

## What the work established

| Question | Finding |
|---|---|
| Is the detector good enough to name the defect in a photo? | Yes at image level (mean AP 0.898, AUROC 0.915–0.977, recall targets met out of fold); no at box level (mAP50 0.38). |
| Can it gate clean vs defective photos? | Better but not reliably: clean photos flagged fell from 45% to 22.5% (TTA, retraining, learned scorer) at ~96% defect recall; structural crack is the weak class (16.5%). |
| Do raw scores work as probabilities? | No (calibration error 0.07–0.23). The logistic image-level scorer outputs probabilities with error ≤ 0.023, but they encode this dataset's prevalence. |
| Does higher resolution help? | Not measurably at image level (960 px tie with 640 px for ~4–5× compute). |
| Does domain prompting help speech? | Yes on synthetic memos: word error 0.229 → 0.177, keyword error 3.6% → 0.6%. |
| Does a better detector improve fusion? | Not distinguishably: 1.1–1.8% of critical misses escape review across detector variants (4 seeds), at 33–39% review. The synthetic voice/label layer and ~100 critical holdout cases dominate. |
| Can fusion hit ≤3% escaped critical misses? | Yes on average (1.8% ± 1.2%, worst seed 3.3%), at a 38% ± 9% review rate, on the hybrid fixture only. |

## Hardware budget (8 GB)

| Stage | Setting | Observed |
|---|---|---|
| YOLOv8s training | 640 px, batch 8, AMP | ~2.5 GB, 20–45 s/epoch |
| YOLOv8s inference | single pass | < 1 GB |
| Whisper large-v3 | `int8_float16` | ~3 GB |

Stages run sequentially per ticket; training and inference should not overlap.

## Risks

| Risk | Impact | Status |
|---|---|---|
| No real memos / expert priority labels | fusion accuracy unknown | open; the main gap |
| CODEBRIM licence is non-commercial, no redistribution | data, weights, score files stay local | handled in `.gitignore` and README |
| Hairline crack and water leak have no training data | those classes cannot be detected | open; need another dataset or own photos |
| CODEBRIM's generic "crack" is mapped to `structural_crack` | cosmetic cracks over-escalated | accepted, conservative |
| Detector false alarms on clean photos (22.5% flagged) | human-review load | reduced from 45%; more clean negatives and a larger model would help |
| Scorer and thresholds fitted on fold-model scores, applied to the final model trained on all data | small distribution shift | accepted (standard k-fold practice); refit on a production sample |
| Tiled inference not validated | score shift vs the scorer | `DefectDetector.tile=None` by default; `ImageLevelDetector` uses one pass + TTA as validated |
| Label noise: unlabelled stains among the 538 negatives | FPR slightly pessimistic | noted |
| Python 3.14 / PyAV 19 / CUDA-lib incompatibilities | install failures | worked around and documented |

## Remaining work, by expected gain

1. Real technician memos and expert priority labels (even 200 cases) to replace the synthetic voice/label layer.
2. Try a larger model (yolov8m needs a new weights download), add more clean negatives, and rerun the image-level
   evaluation; structural crack (precision 0.64 at 0.95 recall) has the most headroom.
3. Collect photos of hairline cracks and water leaks, or drop those classes from the taxonomy.
4. Re-tune the operating point on real fusion data; aim for ≤3% escaped critical misses at a tolerable review rate.
5. Plane API client with idempotency keys and deduplication; drift monitoring; shadow-mode rollout.
