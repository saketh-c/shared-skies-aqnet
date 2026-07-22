"""Model interpretation artifacts for AQNet: SHAP summary + permutation report.

Both functions guard their optional dependencies (shap, matplotlib) and skip
gracefully with a printed reason — interpretation is a nice-to-have layered on
top of the pipeline, never a reason for a Colab run to crash. Nothing here
touches the target or the folds; both functions operate on an already-fitted
model and a caller-chosen evaluation slice.

The permutation report uses grouped-honest data by convention: pass X/y from
held-out fold rows (e.g. the OOF slice of one LOSO fold), not training rows,
so importances reflect generalization rather than memorization.
"""
import os
import sys
import json

import numpy as np
import pandas as pd

# ── Path bootstrap (identical across aqnet modules) ─────────────────────────

_AQNET_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_AQNET_DIR))
_DEEP_DIR = os.path.join(_ROOT, "research", "deeplearning")
for _p in (_AQNET_DIR, _DEEP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Permutation-report settings (kept as constants so the pinned signature of
# permutation_report stays exact).
N_REPEATS = 5
PERM_SEED = 0


# ── Helpers ─────────────────────────────────────────────────────────────────

def _r2(y_true, y_pred):
    """Plain R² over finite pairs; NaN when undefined."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() < 2:
        return float("nan")
    yt, yp = y_true[ok], y_pred[ok]
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - float(np.sum((yp - yt) ** 2)) / ss_tot


def _jsonable(x):
    """Replace non-finite floats with None so the report is strict JSON."""
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (int, np.integer)):
        return int(x)
    return x


def _ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


# ── SHAP summary ────────────────────────────────────────────────────────────

def shap_summary(fitted_lgbm, X_sample, out_png):
    """Beeswarm SHAP summary plot for a fitted tree model (LightGBM expected).

    Parameters
    ----------
    fitted_lgbm : a fitted tree estimator shap.TreeExplainer accepts
        (LGBMRegressor / Booster; XGBoost and CatBoost also work).
    X_sample : pd.DataFrame
        Feature rows to explain. Keep it a sample (a few thousand rows) —
        TreeExplainer cost scales with rows x trees.
    out_png : str
        Destination path for the rendered PNG.

    Returns the out_png path on success, or None (with a printed reason) when
    shap / matplotlib is unavailable or explanation fails — callers can then
    simply skip the artifact.
    """
    try:
        import shap
    except ImportError:
        print("[interpret] shap not installed — skipping SHAP summary "
              "(pip install shap)")
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless-safe: no display on Colab/CI
        import matplotlib.pyplot as plt
    except ImportError:
        print("[interpret] matplotlib not installed — skipping SHAP summary")
        return None

    try:
        explainer = shap.TreeExplainer(fitted_lgbm)
        values = explainer.shap_values(X_sample)
        if isinstance(values, list):  # some shap versions wrap regression output
            values = values[0]
    except Exception as e:
        print(f"[interpret] SHAP explanation failed ({e}) — skipping")
        return None

    _ensure_parent_dir(out_png)
    plt.figure()
    shap.summary_plot(values, X_sample, show=False)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"[interpret] SHAP summary -> {out_png}")
    return out_png


# ── Permutation importance ──────────────────────────────────────────────────

def permutation_report(model_predict, X, y, features, out_json):
    """Permutation importance (delta R² per feature) written as strict JSON.

    Parameters
    ----------
    model_predict : callable(pd.DataFrame) -> 1-D array
        Prediction closure over the fitted model (e.g. lambda F:
        models_tabular.predict_full(fitted, F)). Called once for the baseline
        and N_REPEATS times per feature, so subsample X for large frames.
    X : pd.DataFrame or 2-D array
        Feature rows; arrays are wrapped into a DataFrame with `features`.
    y : 1-D array
        Ground truth aligned to X.
    features : list[str]
        Columns to permute (and the column order model_predict expects).
    out_json : str
        Destination path for the JSON report.

    Each feature's column is shuffled N_REPEATS times (fixed seed) while all
    others stay intact; the drop in R² versus the unpermuted baseline is that
    feature's importance. Values near zero mean the model does not rely on the
    feature (or its signal is duplicated by correlated features — permutation
    importance splits credit across correlated groups).

    Returns out_json. Requires only numpy/pandas, so it never skips.
    """
    if isinstance(X, pd.DataFrame):
        X = X.copy()
    else:
        X = pd.DataFrame(np.asarray(X), columns=list(features))
    features = [f for f in features if f in X.columns]
    y = np.asarray(y, dtype=np.float64)

    base_pred = np.asarray(model_predict(X[features]), dtype=np.float64).ravel()
    base_r2 = _r2(y, base_pred)
    print(f"[interpret] permutation baseline R2 {base_r2:.4f} on {len(y):,} rows"
          if np.isfinite(base_r2) else
          f"[interpret] permutation baseline R2 undefined on {len(y):,} rows")

    rng = np.random.default_rng(PERM_SEED)
    rows = []
    for feat in features:
        orig = X[feat].to_numpy(copy=True)
        deltas = []
        for _ in range(N_REPEATS):
            X[feat] = orig[rng.permutation(len(X))]
            pred = np.asarray(model_predict(X[features]), dtype=np.float64).ravel()
            deltas.append(base_r2 - _r2(y, pred))
        X[feat] = orig
        deltas = np.asarray(deltas, dtype=np.float64)
        rows.append({
            "feature": feat,
            "delta_r2_mean": float(np.nanmean(deltas)),
            "delta_r2_std": float(np.nanstd(deltas)),
        })

    rows.sort(key=lambda r: -(r["delta_r2_mean"]
                              if np.isfinite(r["delta_r2_mean"]) else -np.inf))
    report = {
        "base_r2": base_r2,
        "n": int(len(y)),
        "n_repeats": N_REPEATS,
        "seed": PERM_SEED,
        "importances": rows,
    }
    _ensure_parent_dir(out_json)
    with open(out_json, "w") as f:
        json.dump(_jsonable(report), f, indent=2)
    print(f"[interpret] permutation report -> {out_json}")
    return out_json
