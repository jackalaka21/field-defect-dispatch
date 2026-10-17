# Feasibility assessment

## Verdict

Feasible on the target machine (RTX 3070 Ti Laptop, 8 GB) and now demonstrated end to end on real components: a
CODEBRIM-trained YOLOv8s detector, Whisper large-v3 and the fusion/ticket stage run on a real photo plus a spoken
memo. **What is not demonstrated is real-world accuracy**: there are no real technician memos and no expert priority
labels, so the voice and priority parts of every number are synthetic. See the README for what is real.

## What the work established

| Question | Finding |
|---|---|
| Is the detector good enough to name the defect in a photo? | Yes at image level (AUROC 0.89–0.97, recall targets met out of fold); no at box level (mAP50 0.37). |
| Can it gate clean vs defective photos? | Poorly: at recall-targeted thresholds 45% of clean photos are flagged. Structural-crack alone 30%. |
| Do raw scores work as probabilities? | No (calibration error 0.07–0.23). Isotonic calibration fixes it to ~0.01–0.02, but is prevalence-specific. |
| Does higher resolution help? | Not measurably at image level (960 px tie with 640 px for ~4–5× compute). |
| Does domain prompting help speech? | Yes on synthetic memos: word error 0.229 → 0.177, keyword error 3.6% → 0.6%. |
| Can fusion hit ≤3% escaped critical misses? | With the real detector's false alarms: 4.3% at a 52% review rate on the holdout. Not yet. |

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
| Detector false alarms on clean photos (45% flagged) | high human-review load | open; more clean negatives, longer training, better threshold policy |
| Thresholds/calibration come from fold models, applied to the final model fitted on all data | small distribution shift | accepted (standard k-fold practice); refit on a production sample |
| Tiled inference not validated | score shift vs calibration | `tile=None` by default; enable only after refitting |
| Label noise: unlabelled stains among the 538 negatives | FPR slightly pessimistic | noted |
| Python 3.14 / PyAV 19 / CUDA-lib incompatibilities | install failures | worked around and documented |

## Remaining work, by expected gain

1. Real technician memos and expert priority labels (even 200 cases) to replace the synthetic voice/label layer.
2. Train longer (every CV fold's best epoch was its last) and try a larger model; add more clean negatives to cut
   false alarms.
3. Collect photos of hairline cracks and water leaks, or drop those classes from the taxonomy.
4. Re-tune the operating point on real fusion data; aim for ≤3% escaped critical misses at a tolerable review rate.
5. Plane API client with idempotency keys and deduplication; drift monitoring; shadow-mode rollout.
