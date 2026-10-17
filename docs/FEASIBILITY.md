# Feasibility assessment

## Verdict

Feasible on the target machine (RTX 3070 Ti Laptop, 8 GB) as a research-grade pipeline. The software and
evaluation design are done and tested. **Real accuracy is gated by data, not code**: there is no in-domain labelled
photo set and no real technician audio yet.

## Hardware budget (8 GB)

| Stage | Setting | Approx. VRAM |
|---|---|---|
| YOLOv8s training | imgsz 640, batch 8, AMP | 4–6 GB |
| YOLOv8s inference (tiled) | batch 1 | < 1 GB |
| Faster-Whisper `small.en` / `large-v3` | `int8_float16` | ~1 GB / ~3 GB |
| LLM finaliser | API call, or a 4-bit 7B locally | 0 / ~5 GB |

Stages run sequentially per ticket, so inference never needs more than the largest stage. Training and inference
should not overlap. Note: the machine shows ~1.5 GB already in use by other processes.

## Risks and how each is handled

| Risk | Impact | Mitigation |
|---|---|---|
| No labelled in-domain photos | detector accuracy unknown | k-fold fine-tuning script is ready and smoke-tested; public concrete-defect sets (candidates: CODEBRIM, SDNET2018) need their licence and label format checked, and are classification rather than box labels, so some boxes must be annotated |
| Priority ground truth is subjective | any accuracy number depends on who labelled | fusion is trained and scored on expert-labelled tickets only; synthetic labels are a fixture |
| Class taxonomy gap (photo says `exposed_rebar`, voice says "corrosion") | modalities appear to conflict | `rebar_corrosion` and `spalling` are first-class detector classes; conflict → human review rather than a guess |
| Python 3.14 has no `ctranslate2` wheel | speech stage cannot install | project pinned to a Python 3.12 venv |
| Rare critical class (P1 ≈ 7%) | argmax models ignore it | cost-sensitive decision, safety floor, recall-targeted thresholds, escaped-miss metric |
| LLM nondeterminism | invalid or downgraded tickets | deterministic writer is default; LLM output is schema-validated, retried, floor-checked and falls back to the deterministic ticket |
| TTS audio is cleaner than real memos | ASR numbers optimistic | used only for A/B comparisons (prompt vs none); real memos needed for absolute WER |

## Evaluation design (why the numbers can be trusted once real data exists)

- **Grouped splits.** Folds are grouped by site/inspection so near-duplicate photos never straddle train and
  validation.
- **Nested tuning.** Model choice runs inside each training fold; the operating point is chosen on pooled
  out-of-fold probabilities; the holdout is looked at once.
- **Metrics that match the harm.** Critical-miss rate, cost under an asymmetric matrix, and the *escaped* critical-miss
  rate after human review, in addition to accuracy and macro-F1.
- **Always report mean ± std over folds**, never a single split.

## Remaining work, in order of expected accuracy gain

1. Assemble real data: photos with box labels, 100+ real or recorded memos, and expert priority labels per case.
2. Run `train_detector_cv.py` (YOLOv8s, 5 folds) and adopt the OOF-tuned `thresholds.json`.
3. Run `eval_asr.py` on real memos: compare `small.en` vs `large-v3`, prompt vs none, and tune the repair step on the
   observed errors.
4. Re-run `cv_fusion.py` on real evidence; the simulator in `dispatch/synth/` is then used only for tests.
5. Shadow-mode deployment against the Plane API before any ticket is auto-created.

## Planned, not built

- Plane API client (create/idempotency key/deduplicate by asset + defect + time window).
- Drift monitoring (class mix, mean confidence, review rate).
- Conformal calibration of the detector scores.
