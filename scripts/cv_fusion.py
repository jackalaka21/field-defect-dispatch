"""Nested, site-grouped cross-validation of the fusion stage.

Outer folds estimate generalisation; the inner loop picks the estimator and its hyper-parameters by
mean expected cost (not accuracy), so tuning never sees the outer validation fold. A final model is
refit on all development data and scored once on a held-out set of unseen sites.

    python scripts/cv_fusion.py --n 4000 --holdout 1500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import replace  # noqa: E402

from dispatch.fusion import (CostMatrix, FusionConfig, PriorityModel, build_matrix, expected_cost_rank,  # noqa: E402
                             review_reasons, rule_priority, safety_floor)
from dispatch.metrics import expected_calibration_error, priority_metrics  # noqa: E402
from dispatch.synth.cases import generate_cases  # noqa: E402

COST = CostMatrix()
CFG = FusionConfig()


def candidates():
    for c in (0.1, 1.0, 10.0):
        yield f"logreg(C={c})", lambda c=c: make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
    for depth in (3, None):
        for lr in (0.05, 0.1):
            yield (f"hgb(depth={depth},lr={lr})",
                   lambda d=depth, l=lr: HistGradientBoostingClassifier(max_depth=d, learning_rate=l, max_iter=200,
                                                                        l2_regularization=1.0, random_state=0))


def decide(proba, evidence, use_floor=True, cost_sensitive=True):
    out = []
    for p, ev in zip(proba, evidence):
        r = expected_cost_rank(p, COST) if cost_sensitive else int(p.argmax())
        if use_floor:
            r = min(r, safety_floor(ev, CFG).rank)
        out.append(r)
    return np.array(out)


def inner_select(X, y, groups, evidence, seed):
    """Pick the candidate with the lowest out-of-fold mean cost on the training portion only."""
    best, best_cost = None, np.inf
    skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    for name, make in candidates():
        costs = []
        for tr, va in skf.split(X, y, groups):
            m = PriorityModel(make(), COST, CFG).fit(X[tr], y[tr])
            pred = decide(m.predict_proba(X[va]), [evidence[i] for i in va])
            costs.append(priority_metrics(y[va], pred, COST)["mean_cost"])
        if np.mean(costs) < best_cost:
            best, best_cost = (name, make), float(np.mean(costs))
    return best


def operating_metrics(proba, evidence, y, under_scale, review_prob):
    """Production view: flagged tickets go to a human (assumed correct); what matters is what escapes."""
    cost = CostMatrix(under=tuple(u * under_scale for u in COST.under), over=COST.over)
    cfg = replace(CFG, review_prob=review_prob)
    pred = np.array([min(expected_cost_rank(p, cost), safety_floor(e, cfg).rank) for p, e in zip(proba, evidence)])
    flagged = np.array([bool(review_reasons(e, float(p.max()), cfg)) for p, e in zip(proba, evidence)])
    p1 = y == 0
    return {"under_scale": under_scale, "review_prob": review_prob,
            "review_rate": float(flagged.mean()),
            "critical_miss_raw": float((p1 & (pred != 0)).sum() / max(1, p1.sum())),
            "critical_miss_escaped": float((p1 & (pred != 0) & ~flagged).sum() / max(1, p1.sum())),
            "auto_accuracy": float((pred[~flagged] == y[~flagged]).mean()) if (~flagged).any() else float("nan"),
            "over_escalation": float((pred < y).mean()),
            "wasteful_rate": float((pred <= y - 2).mean()),  # dispatched two or more levels too urgent
            "p1_precision": float(((pred == 0) & p1).sum() / max(1, (pred == 0).sum()))}


def pick_operating_point(proba, evidence, y, max_escaped=0.02, waste_weight=2.0):
    """Lowest (review workload + weighted wasteful dispatch) whose escaped critical misses stay under the
    target. The target sits below the 3% goal as a margin: the choice is made on finite OOF data."""
    grid = [operating_metrics(proba, evidence, y, s, t) for s in (0.5, 1.0, 2.0, 3.0, 4.0)
            for t in (0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8)]
    ok = [g for g in grid if g["critical_miss_escaped"] <= max_escaped]
    best = min(ok, key=lambda g: g["review_rate"] + waste_weight * g["wasteful_rate"]) if ok else min(
        grid, key=lambda g: g["critical_miss_escaped"])
    return best, grid


def summarise(rows):
    keys = rows[0].keys()
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))} for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--holdout", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="reports/fusion_cv.json")
    ap.add_argument("--bank", action="store_true", help="use real out-of-fold detector behaviour (hybrid fixture)")
    ap.add_argument("--op-out", default="reports/operating_point.json")
    a = ap.parse_args()

    bank = None
    if a.bank:
        from dispatch.synth.bank import DetectorBank

        bank = DetectorBank("reports/detector_image_level_v2.npz", "reports/detector_thresholds.json",
                            "reports/detector_calibration.json")
    dev = generate_cases(a.n, seed=a.seed, bank=bank.view("dev") if bank else None)
    ev = [c.evidence for c in dev]
    X = build_matrix(ev)
    y = np.array([c.true_priority.rank for c in dev])
    groups = np.array([c.site for c in dev])

    systems = {"rules": [], "floor_only": [], "model_argmax_nofloor": [], "model_argmax_floor": [],
               "model_cost_floor": []}
    chosen, review = [], []
    oof = np.zeros((len(y), 4))
    skf = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
    for k, (tr, va) in enumerate(skf.split(X, y, groups)):
        name, make = inner_select(X[tr], y[tr], groups[tr], [ev[i] for i in tr], seed=k)
        chosen.append(name)
        m = PriorityModel(make(), COST, CFG).fit(X[tr], y[tr])
        proba = m.predict_proba(X[va])
        oof[va] = proba
        ev_va = [ev[i] for i in va]
        systems["rules"].append(priority_metrics(y[va], np.array([rule_priority(e, CFG).rank for e in ev_va]), COST))
        systems["floor_only"].append(priority_metrics(y[va], np.array([safety_floor(e, CFG).rank for e in ev_va]), COST))
        systems["model_argmax_nofloor"].append(priority_metrics(y[va], decide(proba, ev_va, False, False), COST))
        systems["model_argmax_floor"].append(priority_metrics(y[va], decide(proba, ev_va, True, False), COST))
        pred = decide(proba, ev_va, True, True)
        systems["model_cost_floor"].append({**priority_metrics(y[va], pred, COST),
                                            "ece": expected_calibration_error(proba, y[va])})
        flags = np.array([d.needs_review for d in m.decide_many(ev_va)])
        review.append({"review_rate": float(flags.mean()),
                       "auto_accuracy": float((pred[~flags] == y[va][~flags]).mean()) if (~flags).any() else float("nan"),
                       "flagged_accuracy": float((pred[flags] == y[va][flags]).mean()) if flags.any() else float("nan")})
        print(f"fold {k}: chose {name}")

    # operating point (cost scale + review threshold) chosen on pooled out-of-fold probabilities only
    point, grid = pick_operating_point(oof, ev, y)

    # final refit on all development data, one look at an untouched holdout of unseen sites
    name, make = inner_select(X, y, groups, ev, seed=99)
    final = PriorityModel(make(), COST, CFG).fit(X, y)
    import joblib

    Path("weights").mkdir(exist_ok=True)
    joblib.dump(final.estimator, "weights/fusion_bank.joblib" if a.bank else "weights/fusion.joblib")  # local only (gitignored); rerun this script to rebuild
    test = generate_cases(a.holdout, seed=a.seed + 1000, bank=bank.view("holdout") if bank else None)
    ev_t = [c.evidence for c in test]
    y_t = np.array([c.true_priority.rank for c in test])
    proba_t = final.predict_proba(build_matrix(ev_t))
    holdout_point = operating_metrics(proba_t, ev_t, y_t, point["under_scale"], point["review_prob"])
    holdout_default = operating_metrics(proba_t, ev_t, y_t, 1.0, CFG.review_prob)
    holdout = {
        "model": name,
        "operating_point": holdout_point,
        "default_point": holdout_default,
        "model_cost_floor": priority_metrics(y_t, decide(proba_t, ev_t, True, True), COST),
        "rules": priority_metrics(y_t, np.array([rule_priority(e, CFG).rank for e in ev_t]), COST),
        "ece": expected_calibration_error(proba_t, y_t),
    }

    report = {"n_dev": a.n, "n_holdout": a.holdout, "folds": a.folds, "chosen_per_fold": chosen,
              "cv": {k: summarise(v) for k, v in systems.items()}, "review_routing": summarise(review),
              "operating_point_oof": point, "operating_grid_oof": grid,
              "holdout": holdout, "note": "synthetic fixture: validates the machinery, not real-world accuracy"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))

    print(f"\n{a.folds}-fold site-grouped CV (mean ± std)")
    print(f"{'system':24s} {'acc':>13s} {'macroF1':>13s} {'crit-miss':>13s} {'cost':>13s}")
    for s, r in report["cv"].items():
        print(f"{s:24s} " + " ".join(f"{r[k]['mean']:6.3f}±{r[k]['std']:.3f}" for k in
                                       ("accuracy", "macro_f1", "critical_miss_rate", "mean_cost")))
    fmt = lambda d: {k: round(v, 3) for k, v in d.items()}  # noqa: E731
    print("\nchosen operating point (pooled out-of-fold):", fmt(point))
    print("holdout at chosen point :", fmt(holdout_point))
    print("holdout at default point:", fmt(holdout_default))
    Path(a.op_out).write_text(json.dumps(
        {"under_scale": point["under_scale"], "review_prob": point["review_prob"]}, indent=2))
    print(f"\nholdout ({name}):", {k: round(v, 3) for k, v in holdout["model_cost_floor"].items()})
    print("holdout rules   :", {k: round(v, 3) for k, v in holdout["rules"].items()})


if __name__ == "__main__":
    main()
