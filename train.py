

from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from features import FEATURE_COLUMNS, build_feature_frame

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")

N_SPLITS = 5
RANDOM_STATE = 0
FALLBACK_TIMEOUT_S = 1.6 

def make_base_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.08,
        max_iter=200,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )


def class_weights(y: np.ndarray) -> np.ndarray:
    """Sample weights inverse to class frequency (mean weight == 1)."""
    y = np.asarray(y).astype(int)
    w = np.ones(len(y), dtype=float)
    for c in np.unique(y):
        w[y == c] = len(y) / (len(np.unique(y)) * max((y == c).sum(), 1))
    return w


def load_labels(data_dirs: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    frames, dirs = [], []
    for d in data_dirs:
        p = os.path.join(d, "labels.csv")
        if not os.path.exists(p):
            raise FileNotFoundError(f"no labels.csv in {d}")
        df = pd.read_csv(p)
        df["__dir"] = d
        # turn_id is only unique within a folder; namespace it for grouping.
        df["turn_id"] = df["turn_id"].astype(str)
        df["__gid"] = os.path.basename(os.path.normpath(d)) + "/" + df["turn_id"]
        frames.append(df)
        dirs.append(d)
    return pd.concat(frames, ignore_index=True), dirs



def turn_level_curve(meta: pd.DataFrame, y: np.ndarray, p: np.ndarray,
                     thresholds: np.ndarray) -> pd.DataFrame:
    d = meta.copy()
    d["y"], d["p"] = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    d = d.sort_values(["__gid", "pause_index"])

    turns = []
    for gid, g in d.groupby("__gid", sort=False):
        eot = g[g["y"] == 1]
        t_eot = float(eot["pause_start"].iloc[0]) if len(eot) else float(g["pause_start"].max())
        turns.append((gid, g["p"].to_numpy(), g["y"].to_numpy(),
                      g["pause_start"].to_numpy(float), t_eot))

    out = []
    for thr in thresholds:
        cuts, delays = 0, []
        for _, p_, y_, ts_, t_eot in turns:
            fired = np.flatnonzero(p_ >= thr)
            if len(fired) == 0:
                delays.append(FALLBACK_TIMEOUT_S)
                continue
            k = fired[0]
            if y_[k] == 0:
                cuts += 1
            delays.append(max(ts_[k] - t_eot, 0.0))
        out.append({
            "threshold": float(thr),
            "false_cut_rate": cuts / max(len(turns), 1),
            "mean_delay_s": float(np.mean(delays)),
            "p50_delay_s": float(np.median(delays)),
            "n_turns": len(turns),
        })
    return pd.DataFrame(out)


def operating_point(curve: pd.DataFrame, max_fcr: float = 0.05) -> dict:
    ok = curve[curve["false_cut_rate"] <= max_fcr]
    if not len(ok):
        return {}
    return ok.sort_values("mean_delay_s").iloc[0].to_dict()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", nargs="+", required=True,
                    help="one or more folders, each containing labels.csv (english hindi)")
    ap.add_argument("--model_out", default="eot_model.joblib")
    ap.add_argument("--skip_importance", action="store_true")
    args = ap.parse_args()

    os.makedirs(ART, exist_ok=True)
    labels, _ = load_labels(args.data_dir)
    print(f"[data] {len(labels)} pauses / {labels['__gid'].nunique()} turns")

    Xs, Ms = [], []
    for d, sub in labels.groupby("__dir", sort=False):
        print(f"[features] {d}")
        X, M = build_feature_frame(d, sub)
        M["__gid"] = sub["__gid"].to_numpy()
        M["label"] = sub["label"].to_numpy()  
        Xs.append(X)
        Ms.append(M)
    X = pd.concat(Xs, ignore_index=True)
    M = pd.concat(Ms, ignore_index=True)
    y = (M["label"].astype(str).str.lower() == "eot").astype(int).to_numpy()
    groups = M["__gid"].to_numpy()
    print(f"[data] eot={y.sum()}  hold={(1 - y).sum()}  ({y.mean():.1%} positive)")

    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(y), dtype=float)
    for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
        inner = list(GroupKFold(n_splits=min(N_SPLITS, len(np.unique(groups[tr]))))
                     .split(X.iloc[tr], y[tr], groups[tr]))
        m = CalibratedClassifierCV(make_base_model(), method="isotonic", cv=inner)
        m.fit(X.iloc[tr], y[tr], sample_weight=class_weights(y[tr]))
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        print(f"[cv] fold {k}: AUC={roc_auc_score(y[te], oof[te]):.4f}")

    metrics = {
        "oof_roc_auc": float(roc_auc_score(y, oof)),
        "oof_pr_auc": float(average_precision_score(y, oof)),
    }
    thresholds = np.unique(np.round(np.linspace(0.01, 0.99, 197), 4))
    curve = turn_level_curve(M, y, oof, thresholds)
    curve.to_csv(os.path.join(ART, "delay_vs_cutoff.csv"), index=False)
    op = operating_point(curve, 0.05)
    metrics["operating_point_fcr<=5%"] = op
    print(f"[oof] ROC-AUC={metrics['oof_roc_auc']:.4f} PR-AUC={metrics['oof_pr_auc']:.4f}")
    if op:
        print(f"[oof] @FCR<=5%: thr={op['threshold']:.3f} "
              f"mean_delay={op['mean_delay_s']:.3f}s fcr={op['false_cut_rate']:.3%}")

    pd.DataFrame({
        "turn_id": M["turn_id"], "pause_index": M["pause_index"],
        "gid": M["__gid"], "y": y, "p_eot": oof,
    }).to_csv(os.path.join(ART, "oof.csv"), index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
        ax.plot(curve["false_cut_rate"] * 100, curve["mean_delay_s"], lw=2)
        ax.axvline(5, ls="--", c="crimson", lw=1, label="5% budget")
        if op:
            ax.plot(op["false_cut_rate"] * 100, op["mean_delay_s"], "o", c="crimson")
        ax.set_xlim(0, 25)
        ax.set_xlabel("false-cutoff rate (%)")
        ax.set_ylabel("mean response delay (s)")
        ax.set_title("Delay vs. false-cutoff (out-of-fold)")
        ax.grid(alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(ART, "delay_vs_cutoff.png"))
    except Exception as e:  # plotting is optional
        print(f"[warn] plot skipped: {e}")

    if not args.skip_importance:
        tr, te = list(GroupKFold(n_splits=N_SPLITS).split(X, y, groups))[-1]
        base = make_base_model().fit(X.iloc[tr], y[tr], sample_weight=class_weights(y[tr]))
        r = permutation_importance(base, X.iloc[te], y[te], n_repeats=5,
                                   random_state=RANDOM_STATE, scoring="average_precision")
        pd.DataFrame({"feature": FEATURE_COLUMNS,
                      "importance": r.importances_mean,
                      "std": r.importances_std}) \
            .sort_values("importance", ascending=False) \
            .to_csv(os.path.join(ART, "importance.csv"), index=False)

    full_cv = list(GroupKFold(n_splits=N_SPLITS).split(X, y, groups))
    final = CalibratedClassifierCV(make_base_model(), method="isotonic", cv=full_cv)
    final.fit(X, y, sample_weight=class_weights(y))

    bundle = {
        "model": final,
        "feature_columns": FEATURE_COLUMNS,
        "threshold_at_5pct": float(op["threshold"]) if op else 0.5,
        "metrics": metrics,
        "sklearn_version": __import__("sklearn").__version__,
    }
    out = os.path.join(HERE, args.model_out)
    joblib.dump(bundle, out, compress=3)
    with open(os.path.join(ART, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[save] {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()