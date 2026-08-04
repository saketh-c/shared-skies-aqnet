"""AQNet v2 T1: the statistical skeleton on the FRM scale
(stage `skeleton`, DESIGN S6).

v1's Tier-1 was a 4-model GBM blend on Barkjohn-corrected PA rows whose
pooled LOSO R^2 was ~93% temporal signal, whose residual kriging earned a
composite weight of exactly 0.000 (device-nugget-dominated residuals: no
per-unit random effect existed to absorb it), and whose spatial R^2 on the
FRM-comparable scale was 0.049. T1 answers each measured defect:

  * Candidate A (primary): GPBoost -- LightGBM fixed effects over
    frame2.feature_columns() + a Matern-3/2 GP (Vecchia, m = 30) on
    equirectangular-km coordinates + grouped random effects (unit intercept
    absorbing the device nugget; date intercept absorbing regional shocks).
    Location enters ONLY through the GP: raw lat/lon are banned features
    (frame2 asserts). Precision weights w (AQS 1/sigma_FRM^2, PA
    lam * s2/(s2 + cal_var)) enter the boosting loss via gpb.Dataset(weight=);
    GPBoost estimates the GP/RE covariance parameters unweighted -- a
    documented limitation, not a silent one.
  * Candidate B (documented budget escape + mandatory paired baseline
    forever): the v1 models_tabular.train_cv 4-model ensemble refit on this
    frame (fold_col_overrides = the leakage-free per-fold neighbor/t0/target
    columns from frame2's nbr_overrides npz contract) PLUS a per-day GP
    residual krige of the blend's OOF residuals (fusion.residual_kriging_oof:
    train-fold residuals only, kriged to test rows, 0.0 when no same-day
    train point, added where finite). train_cv clips predictions at 0 --
    fine here because y is FRM-scale PM mass, NEVER reuse it for signed
    residual targets (models_tabular audit, clip-at-0 caveat).

Fold protocol (DESIGN S2; folds2.json is the single authority):
  per outer fold k in 0..4:
    training pool = rows with outer_fold != k and not vault
                    (vault = vault sites + all rows >= VAULT_DATE_START);
    OOF for training-pool rows via the 10 nested unit-grouped LOSO folds
    (folds2 loso_fold[k]); held-out fold-k AQS rows are scored by a fit on
    the FULL fold-k training pool (outer-system overrides).
  Canonical oof: AQS rows take their OWN outer fold's holdout score
  (fold_provenance = k); PA rows take the mean over the 5 per-outer LOSO
  OOFs -- every contributing fit excluded that PA unit (LOSO), so the mean
  is still out-of-fold; it is smoother than any single system's OOF, which
  is documented, not hidden (fold_provenance = -1 for PA rows).

BUDGET ESCAPE (pre-registered, decision recorded in weights_json): the
(outer 0, LOSO 0) candidate-A fit is timed; if the fit alone exceeds
ESCAPE_FIT_SECONDS (12 min -- 55 planned fits would project past 10 h),
candidate B supplies the full OOF grid and candidate A is fit exactly once
per outer fold (5 fits, full training pool, no LOSO) to supply gp_var and
the unit random-effect covariance diagnostics. Under escape, PA-row gp_var
comes from a fit that saw the unit (in-sample diagnostic, flagged); the
decision, timings and winner are all in weights_json.

Candidate selection: A vs B scored on the outer-0 inner SELECTION rows
(folds2 inner_role[0] == 0 -- never the confirmation rows, which belong to
the admission bootstrap) via precision-weighted R^2/RMSE on identical rows.
The winner becomes T1; the loser's canonical OOF ships forever as
per_model_baseline_{name}.

Output config2.artifact("oof_tier1.npz") (INTERFACES schema):
  oof f8[n], gp_var f8[n] (NaN when gpboost unavailable),
  per_model_{name} f8[n] (candidate B blend members), oof_f{k} f8[n]
  (per-outer-system OOF, NaN outside that system's evaluation),
  per_model_baseline_{loser} f8[n], weights_json (json str: decision log,
  winner, timings, blend weights, lambda), fold_provenance i1[n].

gpboost 1.7 API used (documented against gpboost 1.7.1.1 on Phoenix):
  gpb.GPModel(gp_coords=..., cov_function="matern", cov_fct_shape=1.5,
              gp_approx="vecchia", num_neighbors=30, group_data=...,
              likelihood="gaussian")
  gpb.train(params=..., train_set=gpb.Dataset(X, label=y, weight=w),
            gp_model=..., num_boost_round=...)
  bst.predict(data=..., gp_coords_pred=..., group_data_pred=...,
              predict_var=True, pred_latent=False)
    -> {"response_mean": ..., "response_var": ...} (older releases return
    {"fixed_effect", "random_effect_mean", ...}; both forms handled).
  Unseen groups at prediction time get random effect 0 with the group
  variance added to response_var -- exactly the honest behaviour for
  held-out units.

Run from anywhere:
    python skeleton.py [--quick]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import config2
import frame2

# ── Guarded heavy imports (v1 models_tabular style) ─────────────────────────
try:
    import gpboost as gpb
    HAS_GPBOOST = True
except ImportError:
    gpb = None
    HAS_GPBOOST = False
    print("[aqnet2] skeleton: gpboost not installed -- candidate A "
          "unavailable, candidate B becomes T1 (pip install gpboost)")

# ── Constants ───────────────────────────────────────────────────────────────

CAND_A = "gpboost"            # GPBoost Vecchia + unit/day random effects
CAND_B = "ensemble_krige"     # v1 4-model blend + per-day residual krige

NUM_BOOST_ROUND = 500
QUICK_BOOST_ROUND = 60
VECCHIA_NEIGHBORS = 30
MATERN_SHAPE = 1.5            # Matern-3/2 (DESIGN S6)
PRED_CHUNK = 100_000          # bound prediction memory; chunks are exact

# Budget escape (pre-registered): a single LOSO fit past this projects the
# 55-fit full protocol (5 outer x (10 LOSO + 1 full-pool)) past 10 h.
ESCAPE_FIT_SECONDS = 720.0
ESCAPE_BUDGET_HOURS = 10.0

# PA weight multiplier grid, tuned on the (outer 0, LOSO 0) split scored on
# outer-0 SELECTION rows (DESIGN S6: lam frozen from T0/T1 metrics before
# any deep tier trains). 1.0 first: the timing/escape fit doubles as its
# first grid point. Frozen at 1.0 under escape (no LOSO split to tune on).
LAMBDA_GRID = (1.0, 0.5, 2.0)

QUICK_OUTER_FOLDS = 2
QUICK_LOSO_FOLDS = 4
QUICK_MODELS = {"lgbm": {"n_estimators": 300}, "rf": {"n_estimators": 100}}

VAULT_DATE_START = getattr(config2, "VAULT_DATE_START", "2026-01-01")

# Candidate npz artifact names for the nbr_overrides files (producer:
# features stage / frame2.save_overrides; first existing wins).
OUTER_OVERRIDE_NAMES = ("nbr_overrides_outer.npz", "nbr_overrides.npz")
LOSO_OVERRIDE_NAMES = ("nbr_overrides_loso_f{k}.npz",   # pipeline2 canonical
                       "nbr_overrides_loso{k}.npz",
                       "nbr_overrides_loso_{k}.npz")


def _say(msg):
    print(f"[aqnet2] skeleton: {msg}", flush=True)


# ── v1 module bootstrap (lazy; pipeline2 owns the canonical one) ────────────

_V1_CACHE = {}


def _v1_modules():
    """Import v1 models_tabular + fusion via a sys.path bootstrap.

    Lazy so this module imports cleanly with no heavy deps; cached so the
    (import-time) GPU probe and warnings filter in models_tabular run once.
    Returns (models_tabular, fusion) or (None, None) with a printed reason.
    """
    if "mods" in _V1_CACHE:
        return _V1_CACHE["mods"]
    for p in (config2.V1_DIR, config2.PIPELINE_DIR,
              getattr(config2, "DL_DIR", None)):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    try:
        import models_tabular as mt
        import fusion as fus
        _V1_CACHE["mods"] = (mt, fus)
    except Exception as e:  # noqa: BLE001 -- candidate B degrades, loudly
        _say(f"v1 modules unavailable ({e!r}) -- candidate B cannot run")
        _V1_CACHE["mods"] = (None, None)
    return _V1_CACHE["mods"]


# ── folds2.json consumption ─────────────────────────────────────────────────

def load_folds_checked(frame, path=None):
    """Load folds2.json, preferring folds2.load_folds (content-hash verify).

    A frame/folds mismatch silently corrupts every downstream fold, so the
    hash check is the point of folds2.load_folds; when folds2 is not
    importable we fall back to raw JSON with the (weaker) n_rows check and
    say so. Missing file or Phase-1-only content refuses to run.
    """
    p = path or config2.artifact("folds2.json")
    if not os.path.exists(p):
        raise SystemExit(
            f"[aqnet2] skeleton: folds2.json not found at {p} -- run folds2 "
            "(Phase 2) first; T1 without the fold system would be a leak.")
    try:
        import folds2
        folds = folds2.load_folds(p, frame)   # raises on hash mismatch
    except ImportError:
        _say("WARNING: folds2 module not importable -- raw JSON load with "
             "n_rows check only (content hash NOT verified)")
        with open(p, "r", encoding="utf-8") as fh:
            folds = json.load(fh)
    n = int(folds.get("n_rows", -1))
    if n != len(frame):
        raise SystemExit(
            f"[aqnet2] skeleton: folds2.json n_rows={n} != frame rows "
            f"{len(frame)} -- rebuild folds2 against this frame.")
    for key in ("outer_fold", "loso_fold"):
        if key not in folds:
            raise SystemExit(
                f"[aqnet2] skeleton: folds2.json lacks '{key}' -- Phase 2 "
                "row-level arrays are required for the T1 fold protocol.")
    return folds


def _per_outer(folds, key, k):
    """folds[key][k] with str/int key tolerance; None when absent."""
    sub = folds.get(key)
    if not isinstance(sub, dict):
        return None
    arr = sub.get(str(k), sub.get(k))
    return None if arr is None else np.asarray(arr, dtype=int)


def _vault_mask(frame, folds):
    """Vault rows: vault-site units + every row in the vault period."""
    vault_units = sorted("aqs_" + str(s)
                         for s in folds.get("vault_sites", []))
    m = frame["unit_id"].isin(vault_units).to_numpy()
    m |= (frame["date"] >= pd.Timestamp(VAULT_DATE_START)).to_numpy()
    return m


def _build_loso_folds(assign, pool_mask, max_folds=None):
    """Positional (train_idx, test_idx) LOSO pairs RESTRICTED to the pool.

    folds2.folds_from_assign would place assign == -1 rows (which include
    fold-k AQS rows and the vault) in every train split; here rows outside
    the training pool never appear on either side. Pool rows with assign -1
    are always-train (v1 semantics). Returns (pairs, fold_ids) with fold_ids
    sorted (dtype-stable) so pair order == override key order.
    """
    ids = sorted(int(v) for v in np.unique(assign[pool_mask]) if v >= 0)
    if max_folds is not None:
        ids = ids[:max_folds]
    pairs = []
    for fid in ids:
        te = pool_mask & (assign == fid)
        tr = pool_mask & (assign != fid)
        pairs.append((np.flatnonzero(tr), np.flatnonzero(te)))
    return pairs, ids


# ── Overrides (frame2 f{fold}__{col} npz contract) ──────────────────────────

def _load_override_npz(names, n_rows, k=None, quick=False):
    """First existing candidate artifact -> {fold: {col: arr}}.

    Absence is a HARD ERROR in full mode: full-pool neighbor/t0 columns are
    the exact v1 leak the overrides exist to close, and a full run that
    silently degraded would still write a valid-looking oof_tier1.npz
    (review finding). --quick may degrade with the loud v1-style warning.
    """
    for name in names:
        path = config2.artifact(name.format(k=k) if k is not None else name)
        if os.path.exists(path):
            ov = frame2.load_overrides(path, n_rows)
            _say(f"overrides loaded: {os.path.basename(path)} "
                 f"({len(ov)} folds)")
            return ov, path
    if not quick and os.environ.get("AQNET2_ALLOW_FULLPOOL") != "1":
        raise SystemExit(
            f"[aqnet2] skeleton: no nbr_overrides npz found ({names}, "
            f"k={k}) — a FULL run may not fall back to full-pool features "
            "(fold-k FRM would leak into training). Run the features stage "
            "first, or set AQNET2_ALLOW_FULLPOOL=1 to accept the leak "
            "explicitly.")
    _say(f"WARNING: no nbr_overrides npz found ({names}, k={k}) -- "
         "full-pool neighbor/t0 columns will be used (leak-adjacent; run "
         "the features stage overrides first for honest OOF)")
    return {}, None


def _system_target(frame, ov_loso, ov_outer_k, k):
    """Fold-aware FRM-scale target for outer system k.

    Priority: the 'y' override emitted by frame2.neighbor_overrides (PA rows
    swapped to pa_cal_f{k}; identical across the LOSO folds of one system)
    -> outer-system 'y' override -> local pa_cal_f{k} swap -> frame y
    (announced: full-calibration target, deployment view).
    """
    for cols in (ov_loso or {}).values():
        if "y" in cols:
            return np.asarray(cols["y"], dtype=np.float64)
    if ov_outer_k and "y" in ov_outer_k:
        return np.asarray(ov_outer_k["y"], dtype=np.float64)
    y = frame["y"].to_numpy(dtype=np.float64).copy()
    col = f"pa_cal_f{k}"
    if col in frame.columns:
        is_pa = (frame["unit_type"] == "pa").to_numpy()
        y[is_pa] = frame[col].to_numpy(dtype=np.float64)[is_pa]
    else:
        _say(f"WARNING: no fold-aware target for outer {k} "
             f"({col} absent, no 'y' override) -- using frame y")
    return y


def _feature_override_map(ov, fold_ids, feat_set):
    """{fold_index: {col: arr}} for train_cv, feature columns only.

    train_cv keys fold_col_overrides by POSITION in the folds list; the npz
    is keyed by fold id -- fold_ids (sorted) is the bridge. Non-feature keys
    ('y') are dropped here, not left for train_cv's skip-with-print path.
    """
    out = {}
    for i, fid in enumerate(fold_ids):
        cols = {c: a for c, a in (ov.get(fid) or {}).items() if c in feat_set}
        if cols:
            out[i] = cols
    return out


def _apply_feature_overrides(X0, features, cols):
    """Copy of the base matrix with override columns swapped in (all rows)."""
    if not cols:
        return X0
    pos = {c: j for j, c in enumerate(features)}
    X = X0.copy()
    for c, arr in cols.items():
        if c in pos:
            X[:, pos[c]] = np.asarray(arr, dtype=np.float64)
    return X


# ── Geometry / groups for the GP ────────────────────────────────────────────

def equirect_km(lat, lon):
    """Equirectangular projection to km, referenced to the TX bbox centre.

    x = R cos(lat0) dlon, y = R dlat. Adequate over the TX extent for a
    30-neighbor Vecchia GP (max metric distortion ~ 1% at the bbox corners);
    deterministic and dependency-free where an EPSG:3083 transform would
    drag in pyproj (not a guaranteed dep on Phoenix).
    """
    bb = config2.TX_BBOX
    lat0 = 0.5 * (bb["lat_min"] + bb["lat_max"])
    lon0 = 0.5 * (bb["lon_min"] + bb["lon_max"])
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    x = 6371.0 * np.radians(lon - lon0) * np.cos(np.radians(lat0))
    y = 6371.0 * np.radians(lat - lat0)
    return np.column_stack([x, y])


def _gp_groups(frame):
    """(n, 2) object array [unit_id, date-ordinal] for the grouped REs."""
    unit = frame["unit_id"].astype(str).to_numpy(dtype=object)
    days = (frame["date"] - pd.Timestamp(config2.DATE_START)).dt.days
    day = np.array(["d" + str(int(v)) for v in days.to_numpy()], dtype=object)
    return np.column_stack([unit, day])


# ── GPBoost fitting / prediction (candidate A core) ─────────────────────────

def _gpb_params(seed):
    return {"objective": "regression_l2", "learning_rate": 0.05,
            "num_leaves": 63, "min_data_in_leaf": 60, "lambda_l2": 3.0,
            "verbose": -1, "seed": int(seed)}


def _fit_gpb(X, y, w, coords, groups, nbr, seed):
    gp_model = gpb.GPModel(gp_coords=coords, cov_function="matern",
                           cov_fct_shape=MATERN_SHAPE,
                           gp_approx="vecchia",
                           num_neighbors=VECCHIA_NEIGHBORS,
                           group_data=groups, likelihood="gaussian",
                           # gpboost >= 1.7: weights live on GPModel for the
                           # GPBoost algorithm (Dataset weight alone raises).
                           weights=w)
    try:
        gp_model.set_prediction_data(num_neighbors_pred=VECCHIA_NEIGHBORS)
    except Exception:  # noqa: BLE001 -- older API; defaults are fine
        pass
    ds = gpb.Dataset(X, label=y)
    bst = gpb.train(params=_gpb_params(seed), train_set=ds,
                    gp_model=gp_model, num_boost_round=int(nbr))
    return bst, gp_model


def _predict_gpb(bst, X, coords, groups):
    """(mean, var) at query rows, chunked. Mean clipped at 0 (PM mass);
    variance untouched. Handles both gpboost return forms."""
    means, vars_ = [], []
    for lo in range(0, len(X), PRED_CHUNK):
        sl = slice(lo, lo + PRED_CHUNK)
        out = bst.predict(data=X[sl], gp_coords_pred=coords[sl],
                          group_data_pred=groups[sl],
                          predict_var=True, pred_latent=False)
        if isinstance(out, dict):
            m = out.get("response_mean")
            if m is None:
                m = (np.asarray(out["fixed_effect"], dtype=np.float64)
                     + np.asarray(out["random_effect_mean"],
                                  dtype=np.float64))
            v = out.get("response_var")
            if v is None:
                v = out.get("random_effect_cov")
        else:
            m, v = out, None
        m = np.asarray(m, dtype=np.float64).ravel()
        v = (np.asarray(v, dtype=np.float64).ravel() if v is not None
             else np.full(len(m), np.nan))
        means.append(m)
        vars_.append(v)
    return (np.clip(np.concatenate(means), 0.0, None),
            np.concatenate(vars_))


def _cov_pars_jsonable(gp_model):
    """Fitted covariance parameters (unit/day RE variances, GP var/range,
    error term) as a plain dict -- the unit-random-effects diagnostic."""
    try:
        cp = gp_model.get_cov_pars()
        if hasattr(cp, "to_dict"):
            d = cp.to_dict()
            first = next(iter(d.values())) if d else {}
            if isinstance(first, dict):    # DataFrame orient: {col: {row: v}}
                return {str(c): float(list(v.values())[0])
                        for c, v in d.items() if len(v)}
            return {str(k): float(v) for k, v in d.items()}
        arr = np.asarray(cp, dtype=np.float64).ravel()
        return {f"par_{i}": float(v) for i, v in enumerate(arr)}
    except Exception as e:  # noqa: BLE001 -- diagnostics never kill the run
        return {"error": repr(e)}


# ── Candidate A driver ──────────────────────────────────────────────────────

def run_candidate_a(ctx):
    """GPBoost protocol with the pre-registered budget escape.

    Returns {"available", "escape", "oof_f", "var_f", "lambda", "cov_pars",
    "timings", "notes"}. oof_f[k]/var_f[k] are full-length arrays, NaN
    outside system k's evaluation rows.
    """
    n = ctx["n"]
    res = {"available": False, "escape": False,
           "oof_f": {k: np.full(n, np.nan) for k in ctx["ks"]},
           "var_f": {k: np.full(n, np.nan) for k in ctx["ks"]},
           "lambda": {"grid": list(ctx["lam_grid"]), "chosen": 1.0,
                      "scores": {}},
           "cov_pars": {}, "timings": {}, "notes": []}
    if not HAS_GPBOOST:
        res["notes"].append("gpboost not installed")
        return res
    res["available"] = True

    frame, features, X0 = ctx["frame"], ctx["features"], ctx["X0"]
    coords, groups, w0 = ctx["coords"], ctx["groups"], ctx["w0"]
    is_pa = ctx["is_pa"]
    nbr, seed = ctx["nbr"], config2.SEED
    t_stage = time.time()

    def _fit_rows(idx, y, w):
        ok = np.isfinite(y[idx]) & np.isfinite(w[idx]) & (w[idx] > 0)
        return idx[ok]

    def _w_lam(lam):
        w = w0.copy()
        w[is_pa] *= float(lam)
        return w

    # ── Timing fit + lambda grid on the (outer ks[0], LOSO 0) split ──
    k0 = ctx["ks"][0]
    pairs0, ids0 = ctx["loso"][k0]
    if not pairs0:
        res["notes"].append(f"no LOSO folds for outer {k0} -- forcing the "
                            "escape path")
        res["escape"] = True
    lam_preds = {}
    if pairs0:
        y0 = ctx["y_by_k"][k0]
        ov0 = _feature_override_map(ctx["loso_ov"][k0], ids0,
                                    set(features)).get(0, {})
        X_f = _apply_feature_overrides(X0, features, ov0)
        tr0, te0 = pairs0[0]
        sel0 = ctx["sel_mask"]
        for lam in ctx["lam_grid"]:
            w = _w_lam(lam)
            fit_idx = _fit_rows(tr0, y0, w)
            t0 = time.time()
            bst, gpm = _fit_gpb(X_f[fit_idx], y0[fit_idx], w[fit_idx],
                                coords[fit_idx], groups[fit_idx], nbr, seed)
            fit_s = time.time() - t0
            mean, var = _predict_gpb(bst, X_f[te0], coords[te0], groups[te0])
            sc_mask = sel0[te0] & np.isfinite(y0[te0]) & np.isfinite(mean)
            if sc_mask.sum() < 10:
                sc_mask = np.isfinite(y0[te0]) & np.isfinite(mean)
            m = _weighted_metrics(y0[te0][sc_mask], mean[sc_mask],
                                  w0[te0][sc_mask])
            res["lambda"]["scores"][str(lam)] = m
            lam_preds[lam] = (mean, var)
            _say(f"lambda={lam}: (outer {k0}, LOSO 0) fit {fit_s:.1f}s "
                 f"wrmse={m['wrmse']} wr2={m['wr2']} n={m['n']}")
            if lam == 1.0:
                planned = sum(len(ctx["loso"][k][0]) + 1 for k in ctx["ks"])
                res["timings"]["first_fit_seconds"] = round(fit_s, 1)
                res["timings"]["planned_fits"] = planned
                res["timings"]["projected_hours"] = round(
                    fit_s * planned / 3600.0, 2)
                if fit_s > ESCAPE_FIT_SECONDS:
                    res["escape"] = True
                    res["notes"].append(
                        f"escape: first fit {fit_s:.0f}s > "
                        f"{ESCAPE_FIT_SECONDS:.0f}s (projected "
                        f"{res['timings']['projected_hours']} h > "
                        f"{ESCAPE_BUDGET_HOURS} h)")
                    break
            del bst, gpm

    if res["escape"] and not ctx["allow_escape"]:
        res["escape"] = False
        res["notes"].append("escape requested but candidate B is "
                            "unavailable -- full A protocol runs anyway")

    if not res["escape"] and res["lambda"]["scores"]:
        def _key(lam):
            m = res["lambda"]["scores"][str(lam)]
            return m["wrmse"] if m["wrmse"] is not None else np.inf
        best = min(lam_preds, key=_key)
        res["lambda"]["chosen"] = float(best)
        _say(f"lambda frozen at {best}")
    else:
        res["notes"].append("lambda frozen at 1.0 (escape / no grid)")
    w_best = _w_lam(res["lambda"]["chosen"])

    n_fits = 0
    for k in ctx["ks"]:
        y_k = ctx["y_by_k"][k]
        pool_idx = np.flatnonzero(ctx["pool_mask"][k])
        hold_idx = np.flatnonzero(ctx["hold_mask"][k])

        if not res["escape"]:
            pairs, ids = ctx["loso"][k]
            ov_map = _feature_override_map(ctx["loso_ov"][k], ids,
                                           set(features))
            for j, (tr, te) in enumerate(pairs):
                if k == k0 and j == 0 and res["lambda"]["chosen"] in lam_preds:
                    mean, var = lam_preds[res["lambda"]["chosen"]]
                else:
                    X_f = _apply_feature_overrides(X0, features,
                                                   ov_map.get(j, {}))
                    fit_idx = _fit_rows(tr, y_k, w_best)
                    bst, _g = _fit_gpb(X_f[fit_idx], y_k[fit_idx],
                                       w_best[fit_idx], coords[fit_idx],
                                       groups[fit_idx], nbr, seed)
                    mean, var = _predict_gpb(bst, X_f[te], coords[te],
                                             groups[te])
                    del bst, _g
                    n_fits += 1
                res["oof_f"][k][te] = mean
                res["var_f"][k][te] = var
            _say(f"outer {k}: LOSO OOF complete ({len(pairs)} folds)")

        # Full-training-pool fit -> fold-k AQS holdout (+ diagnostics).
        X_out = _apply_feature_overrides(X0, features,
                                         {c: a for c, a in
                                          (ctx["outer_ov"].get(k) or {})
                                          .items() if c in set(features)})
        fit_idx = _fit_rows(pool_idx, y_k, w_best)
        t0 = time.time()
        bst, gpm = _fit_gpb(X_out[fit_idx], y_k[fit_idx], w_best[fit_idx],
                            coords[fit_idx], groups[fit_idx], nbr, seed)
        n_fits += 1
        res["cov_pars"][str(k)] = _cov_pars_jsonable(gpm)
        if len(hold_idx):
            mean, var = _predict_gpb(bst, X_out[hold_idx], coords[hold_idx],
                                     groups[hold_idx])
            res["oof_f"][k][hold_idx] = mean
            res["var_f"][k][hold_idx] = var
        _say(f"outer {k}: full-pool fit {time.time() - t0:.1f}s, "
             f"{len(hold_idx):,} holdout rows scored")
        if res["escape"] and k == ctx["ks"][0]:
            # PA-row gp_var diagnostic: this fit SAW every PA unit, so the
            # variance is in-sample for the unit RE -- flagged, never hidden.
            pa_idx = np.flatnonzero(is_pa)
            if len(pa_idx):
                _m, v = _predict_gpb(bst, X_out[pa_idx], coords[pa_idx],
                                     groups[pa_idx])
                res["var_f"][k][pa_idx] = v
                res["notes"].append("escape: PA-row gp_var is an in-sample "
                                    "diagnostic (unit seen by the fit)")
        del bst, gpm

    res["timings"]["n_fits_after_probe"] = n_fits
    res["timings"]["total_seconds"] = round(time.time() - t_stage, 1)
    return res


# ── Candidate B driver ──────────────────────────────────────────────────────

def run_candidate_b(ctx):
    """v1 train_cv blend + per-day residual krige, per outer system.

    Returns {"available", "oof_f", "per_model_f", "blend_meta", "timings"}.
    train_cv carries no sample weights (v1 contract) -- candidate B fits are
    unweighted; the precision weights enter only at selection scoring.
    """
    n = ctx["n"]
    res = {"available": False, "oof_f": {k: np.full(n, np.nan)
                                         for k in ctx["ks"]},
           "per_model_f": {}, "blend_meta": {}, "timings": {}}
    mt, fus = _v1_modules()
    if mt is None:
        return res
    res["available"] = True

    frame, features = ctx["frame"], ctx["features"]
    feat_set = set(features)
    t_stage = time.time()

    for k in ctx["ks"]:
        pairs, ids = ctx["loso"][k]
        if not pairs:
            _say(f"outer {k}: no LOSO folds -- candidate B skips the system")
            continue
        y_k = ctx["y_by_k"][k]
        pool_idx = np.flatnonzero(ctx["pool_mask"][k])
        hold_idx = np.flatnonzero(ctx["hold_mask"][k])

        df_k = frame[["date", "lat", "lon"] + features].copy()
        df_k["target"] = y_k
        df_k["sensor_id"] = frame["unit_id"].astype(str)

        ov_map = _feature_override_map(ctx["loso_ov"][k], ids, feat_set)
        cv = mt.train_cv(df_k, features, pairs, target="target",
                         models=ctx["models_spec"],
                         fold_col_overrides=ov_map, return_fitted=False)
        blend = cv["oof_lofo"].copy()   # headline: leave-one-fold-out blend

        # Holdout: outer-system overrides baked into a full-length copy,
        # fit on the whole training pool, per-model + blend predictions.
        out_ov = {c: a for c, a in (ctx["outer_ov"].get(k) or {}).items()
                  if c in feat_set}
        df_out = df_k.copy()
        for c, arr in out_ov.items():
            df_out[c] = np.asarray(arr, dtype=np.float64)
        bundle = mt.fit_full(df_out.iloc[pool_idx].reset_index(drop=True),
                             features, models=ctx["models_spec"])
        hold_pm = {}
        b_hold = np.zeros(0)
        if len(hold_idx):
            Xh = df_out.iloc[hold_idx][features].to_numpy(dtype=np.float64)
            b_hold = mt.predict_full(bundle, Xh)
            med = bundle["impute_medians"]
            for name, est in bundle["models"].items():
                if name == "rf":
                    Xi = Xh.copy()
                    bad = ~np.isfinite(Xi)
                    Xi[bad] = np.broadcast_to(med, Xi.shape)[bad]
                    p = est.predict(Xi)
                else:
                    p = est.predict(Xh)
                hold_pm[name] = np.maximum(0.0,
                                           np.asarray(p, dtype=np.float64))

        # Per-day GP residual krige of the blend's OOF residuals: LOSO folds
        # give the training-pool corrections; one extra (pool -> holdout)
        # fold kriges the SAME honest OOF residuals to the fold-k AQS rows.
        # Train residuals with NaN blend (always-train LOSO rows) are
        # dropped inside the engine; "no same-day train point" is 0.0.
        kr_folds = list(pairs) + ([(pool_idx, hold_idx)]
                                  if len(hold_idx) else [])
        kr = fus.residual_kriging_oof(df_k, blend, kr_folds)

        b = blend
        add = np.isfinite(kr) & np.isfinite(b)
        b[add] += kr[add]
        if len(hold_idx):
            bh = b_hold.copy()
            addh = np.isfinite(kr[hold_idx])
            bh[addh] += kr[hold_idx][addh]
            b[hold_idx] = bh
        res["oof_f"][k] = b

        for name, arr in cv["per_model_oof"].items():
            pm = arr.copy()
            if len(hold_idx) and name in hold_pm:
                pm[hold_idx] = hold_pm[name]
            res["per_model_f"].setdefault(name, {})[k] = pm

        res["blend_meta"][str(k)] = {
            "weights_pooled": cv["weights"],
            "weights_lofo": cv["weights_lofo"],
            "internal_cv_r2": bundle.get("internal_cv_r2"),
        }
        del df_k, df_out, bundle, cv
        _say(f"outer {k}: candidate B OOF complete "
             f"({len(pairs)} LOSO folds + holdout + krige)")

    res["timings"]["total_seconds"] = round(time.time() - t_stage, 1)
    return res


# ── Scoring / assembly ──────────────────────────────────────────────────────

def _weighted_metrics(y, p, w):
    """Precision-weighted R^2 / RMSE on finite rows (paired rows upstream)."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    good = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w > 0)
    y, p, w = y[good], p[good], w[good]
    if len(y) < 2:
        return {"wr2": None, "wrmse": None, "n": int(len(y))}
    sw = w.sum()
    ybar = float((w * y).sum() / sw)
    sse = float((w * (y - p) ** 2).sum())
    sst = float((w * (y - ybar) ** 2).sum())
    return {"wr2": (1.0 - sse / sst) if sst > 0 else None,
            "wrmse": float(np.sqrt(sse / sw)),
            "n": int(len(y))}


def _nanmean_stack(arrs, n):
    """Row-wise mean over finite entries; NaN where none (no warnings)."""
    s = np.zeros(n, dtype=np.float64)
    c = np.zeros(n, dtype=np.float64)
    for a in arrs:
        f = np.isfinite(a)
        s[f] += a[f]
        c[f] += 1.0
    out = np.full(n, np.nan)
    nz = c > 0
    out[nz] = s[nz] / c[nz]
    return out


def assemble_canonical(oof_f, ctx):
    """Canonical per-row array: AQS rows -> own outer fold's holdout value;
    PA rows -> mean over the per-outer LOSO values (each contributor
    excluded the unit: still OOF). NaN where never evaluated (vault; folds
    outside a --quick run)."""
    n = ctx["n"]
    nan = np.full(n, np.nan)
    out = nan.copy()
    mean_all = _nanmean_stack([oof_f.get(k, nan) for k in ctx["ks"]], n)
    pa = ctx["is_pa"]
    out[pa] = mean_all[pa]
    for k in ctx["ks"]:
        hm = ctx["hold_mask"][k]
        out[hm] = oof_f.get(k, nan)[hm]
    return out


def _jsonable(obj):
    """Recursively convert numpy scalars and NaN to JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ── Serving bundles (validate2 vault/bare-site + pipeline2 export hook) ─────
#
# One picklable candidate-B fit per outer fold (t1_serving_f{k}.pkl) plus a
# deployment full model (t1_serving_full.pkl). The fold-k bundle trains on
# fold-k's TRAINING pool only, with the fold-k neighbor/target overrides
# applied, so scoring a fold-k AQS site through it is honest site-holdout.
# Serving is ALWAYS candidate B (fit_full ensemble): a GPBoost winner cannot
# be pickled portably across venv rebuilds, and validate2's parity report
# measures any winner/serving gap rather than hiding it (bundle records the
# caveat).

def _fit_serving_bundle(ctx, k, quick):
    """Fit one serving bundle. k = outer fold int, or None for the full
    deployment model (all non-vault rows, deployment-view features)."""
    mt, _ = _v1_modules()
    if mt is None:
        raise RuntimeError("v1 models_tabular unavailable — no serving fit")
    frame = ctx["frame"]
    features = ctx["features"]
    df = frame[features].copy()
    if k is None:
        mask = np.ones(ctx["n"], dtype=bool)
        y = frame["y"].to_numpy(dtype=np.float64).copy()
    else:
        mask = ctx["pool_mask"][k]
        y = ctx["y_by_k"][k].copy()
        # Outer-fold override columns (train-pool neighbor blocks, fold t0).
        oov = ctx["outer_ov"].get(k) if isinstance(ctx["outer_ov"], dict) \
            else None
        if oov:
            for col, arr in oov.items():
                if col in df.columns and col != "y":
                    df[col] = np.asarray(arr, dtype=np.float64)
    vault = _vault_mask(frame, ctx["folds"])
    mask = mask & ~vault
    fit_df = df.loc[mask].copy()
    fit_df["target"] = y[mask]
    fit_df["sensor_id"] = frame.loc[mask, "unit_id"].astype(str).to_numpy()
    fit_df["date"] = frame.loc[mask, "date"].to_numpy()
    fitted = mt.fit_full(fit_df, features, models=ctx.get("models_spec"))
    return {"kind": "fit_full", "outer_k": k, "features": features,
            "n_train": int(mask.sum()), "quick": bool(quick),
            "caveat": "serving is candidate B regardless of the OOF winner; "
                      "validate2 parity_report measures the gap"}, fitted


def write_serving_bundles(ctx, quick):
    """Persist t1_serving_f{k}.pkl for each run outer fold + full model.
    Failure is loud but non-fatal: export refits when bundles are absent."""
    import pickle
    jobs = [(k, f"t1_serving_f{k}.pkl") for k in ctx["ks"]]
    jobs.append((None, "t1_serving_full.pkl"))
    for k, name in jobs:
        dest = config2.artifact(name)
        if os.path.exists(dest) and os.environ.get("FORCE") != "1":
            _say(f"serving bundle {name} exists -- keep")
            continue
        t0 = time.time()
        try:
            meta, fitted = _fit_serving_bundle(ctx, k, quick)
        except Exception as e:  # noqa: BLE001 — bundles must not kill OOF
            import traceback
            traceback.print_exc()
            _say(f"WARNING serving bundle {name} failed ({e}) -- export "
                 "will refit")
            continue
        tmp = dest + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"meta": meta, "fitted": fitted}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, dest)
        _say(f"wrote {name} (n_train={meta['n_train']:,}, "
             f"{time.time() - t0:.0f}s)")


def load_fold_models():
    """{int k: bundle, 'full': bundle} from t1_serving_*.pkl (may be {}).
    The consumer contract validate2/pipeline2 rely on."""
    import glob as _glob
    import pickle
    out = {}
    for path in sorted(_glob.glob(os.path.join(config2.ARTIFACTS_DIR,
                                               "t1_serving_*.pkl"))):
        stem = os.path.basename(path)[len("t1_serving_"):-len(".pkl")]
        try:
            with open(path, "rb") as f:
                bundle = pickle.load(f)
        except Exception as e:  # unreadable bundle: skip loudly, never guess
            _say(f"WARNING unreadable serving bundle {path}: {e}")
            continue
        out["full" if stem == "full" else int(stem[1:])] = bundle
    return out


def predict_fold(bundle, df):
    """Predict PM2.5 at rows of df (must carry the bundle's feature columns;
    NaN allowed — boosters handle natively, RF uses stored medians)."""
    mt, _ = _v1_modules()
    if mt is None:
        raise RuntimeError("v1 models_tabular unavailable — cannot predict")
    fitted = bundle["fitted"] if "fitted" in bundle else bundle
    return mt.predict_full(fitted, df)


# ── Stage driver ────────────────────────────────────────────────────────────

def run_skeleton(quick=False, frame_path=None, folds_path=None,
                 out_path=None):
    dest = out_path or config2.artifact("oof_tier1.npz")
    if os.path.exists(dest) and os.environ.get("FORCE") != "1":
        _say(f"{dest} exists (FORCE=1 to rebuild) -- skip")
        return 0

    print("[aqnet2] ── stage: skeleton ──")
    t_stage = time.time()
    fpath = frame_path or config2.artifact("frame_truth.parquet")
    if not os.path.exists(fpath):
        raise SystemExit(f"[aqnet2] skeleton: frame not found at {fpath} -- "
                         "run the features stage first.")
    frame = pd.read_parquet(fpath)
    frame["date"] = pd.to_datetime(frame["date"]).dt.as_unit("ns")
    frame["unit_id"] = frame["unit_id"].astype(str)
    for col in ("y", "w"):
        if col not in frame.columns:
            raise SystemExit(f"[aqnet2] skeleton: frame lacks '{col}'")
    folds = load_folds_checked(frame, folds_path)

    n = len(frame)
    features = frame2.feature_columns(frame)
    _say(f"{n:,} rows, {len(features)} features")

    outer_assign = np.asarray(folds["outer_fold"], dtype=int)
    vault = _vault_mask(frame, folds)
    is_pa = (frame["unit_type"] == "pa").to_numpy()
    ks = sorted(int(v) for v in np.unique(outer_assign) if v >= 0)
    if quick:
        ks = ks[:QUICK_OUTER_FOLDS]
        _say(f"--quick: outer folds {ks}, LOSO capped at {QUICK_LOSO_FOLDS}")
    if not ks:
        raise SystemExit("[aqnet2] skeleton: no outer folds in folds2.json")

    pool_mask, hold_mask, loso, loso_ov, y_by_k = {}, {}, {}, {}, {}
    outer_ov, outer_ov_path = _load_override_npz(OUTER_OVERRIDE_NAMES, n,
                                                 quick=quick)
    loso_ov_paths = {}
    max_loso = QUICK_LOSO_FOLDS if quick else None
    for k in ks:
        pool_mask[k] = (outer_assign != k) & ~vault
        hold_mask[k] = (outer_assign == k) & ~vault
        assert not (pool_mask[k] & vault).any(), "vault row in training pool"
        arr = _per_outer(folds, "loso_fold", k)
        if arr is None:
            raise SystemExit(f"[aqnet2] skeleton: folds2.json loso_fold has "
                             f"no entry for outer fold {k}")
        loso[k] = _build_loso_folds(arr, pool_mask[k], max_loso)
        loso_ov[k], loso_ov_paths[k] = _load_override_npz(
            LOSO_OVERRIDE_NAMES, n, k=k, quick=quick)
        y_by_k[k] = _system_target(frame, loso_ov[k], outer_ov.get(k), k)

    # Selection rows: outer ks[0] inner_role == 0 (never confirmation rows).
    k0 = ks[0]
    role0 = _per_outer(folds, "inner_role", k0)
    if role0 is None:
        _say(f"WARNING: folds2.json has no inner_role[{k0}] -- selection "
             "falls back to the whole outer-0 training pool")
        sel_mask = pool_mask[k0].copy()
    else:
        sel_mask = (role0 == 0) & pool_mask[k0]

    mt, _fus = _v1_modules()
    ctx = {
        "frame": frame, "n": n, "features": features, "folds": folds,
        "X0": frame[features].to_numpy(dtype=np.float64),
        "w0": frame["w"].to_numpy(dtype=np.float64),
        "coords": equirect_km(frame["lat"].to_numpy(),
                              frame["lon"].to_numpy()),
        "groups": _gp_groups(frame),
        "is_pa": is_pa, "ks": ks, "pool_mask": pool_mask,
        "hold_mask": hold_mask, "loso": loso, "loso_ov": loso_ov,
        "outer_ov": outer_ov, "y_by_k": y_by_k, "sel_mask": sel_mask,
        "nbr": QUICK_BOOST_ROUND if quick else NUM_BOOST_ROUND,
        "lam_grid": (1.0,) if quick else LAMBDA_GRID,
        "models_spec": QUICK_MODELS if quick else None,
        "allow_escape": mt is not None,
    }
    if not HAS_GPBOOST and mt is None:
        raise SystemExit("[aqnet2] skeleton: neither gpboost nor the v1 "
                         "modules are importable -- no candidate can run.")

    a = run_candidate_a(ctx)
    b = run_candidate_b(ctx)

    # ── Candidate selection on outer-k0 SELECTION rows (paired) ──
    y0, w0 = y_by_k[k0], ctx["w0"]
    comparison = None
    if a["escape"]:
        decision, winner = "budget_escape", CAND_B
    elif not a["available"]:
        decision, winner = "gpboost_unavailable", CAND_B
    elif not b["available"]:
        decision, winner = "candidate_b_unavailable", CAND_A
    else:
        both = (sel_mask & np.isfinite(y0)
                & np.isfinite(a["oof_f"][k0]) & np.isfinite(b["oof_f"][k0]))
        ma = _weighted_metrics(y0[both], a["oof_f"][k0][both], w0[both])
        mb = _weighted_metrics(y0[both], b["oof_f"][k0][both], w0[both])
        comparison = {"outer_k": k0, "n_rows": int(both.sum()),
                      CAND_A: ma, CAND_B: mb}
        if ma["wr2"] is None or mb["wr2"] is None:
            decision = "selection_underpowered"
            winner = CAND_A if ma["wr2"] is not None else CAND_B
        else:
            decision = "selection"
            # Higher weighted R2 wins; exact tie goes to A (the primary).
            winner = CAND_A if ma["wr2"] >= mb["wr2"] else CAND_B
        _say(f"selection on {int(both.sum()):,} rows: "
             f"{CAND_A} wr2={ma['wr2']}  {CAND_B} wr2={mb['wr2']}")
    loser = CAND_B if winner == CAND_A else CAND_A
    _say(f"decision={decision}  T1={winner}  baseline={loser}")

    win_res, lose_res = (a, b) if winner == CAND_A else (b, a)
    oof = assemble_canonical(win_res["oof_f"], ctx)
    gp_var = (assemble_canonical(a["var_f"], ctx) if a["available"]
              else np.full(n, np.nan))
    baseline = (assemble_canonical(lose_res["oof_f"], ctx)
                if lose_res["available"] else None)

    prov = outer_assign.astype(np.int8).copy()
    prov[is_pa] = -1
    prov[vault] = -1

    aqs_scored = np.zeros(n, dtype=bool)
    for k in ks:
        aqs_scored |= hold_mask[k]
    canon_aqs = _weighted_metrics(frame["y"].to_numpy()[aqs_scored],
                                  oof[aqs_scored], w0[aqs_scored])
    _say(f"canonical T1 at held-out AQS rows: wr2={canon_aqs['wr2']} "
         f"wrmse={canon_aqs['wrmse']} n={canon_aqs['n']}")

    weights = {
        "candidates": {
            CAND_A: "gpboost: lgbm fixed effects + matern-3/2 vecchia(m=30) "
                    "GP + unit/day random effects, precision-weighted",
            CAND_B: "v1 train_cv blend (unweighted, clip>=0) + per-day "
                    "residual krige of OOF residuals (train-fold only)",
        },
        "decision": decision, "winner": winner, "loser": loser,
        "escape": {"triggered": bool(a["escape"]),
                   "threshold_seconds": ESCAPE_FIT_SECONDS,
                   "budget_hours": ESCAPE_BUDGET_HOURS,
                   **a["timings"]},
        "selection": comparison,
        "lambda": a["lambda"],
        "blend": b["blend_meta"],
        "gpboost": {"available": a["available"],
                    "cov_pars_by_outer_fold": a["cov_pars"],
                    "params": _gpb_params(config2.SEED) if HAS_GPBOOST
                    else None,
                    "notes": a["notes"]},
        "canonical_aqs_metrics": canon_aqs,
        "nbr_overrides": {"outer": outer_ov_path or "missing",
                          "loso": {str(k): loso_ov_paths[k] or "missing"
                                   for k in ks}},
        "timings_seconds": {"candidate_a": a["timings"].get("total_seconds"),
                            "candidate_b": b["timings"].get("total_seconds"),
                            "stage": round(time.time() - t_stage, 1)},
        "outer_folds_run": ks, "quick": bool(quick),
        "n_rows": n, "n_features": len(features), "seed": config2.SEED,
    }

    payload = {"oof": oof, "gp_var": gp_var,
               "fold_provenance": prov,
               "weights_json": np.array(json.dumps(_jsonable(weights)))}
    for k in ks:
        payload[f"oof_f{k}"] = win_res["oof_f"][k]
    if b["available"]:
        for name, per_k in b["per_model_f"].items():
            payload[f"per_model_{name}"] = assemble_canonical(per_k, ctx)
    if baseline is not None:
        payload[f"per_model_baseline_{loser}"] = baseline

    tmp = dest + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, dest)
    _say(f"wrote {dest} ({len(payload)} arrays, "
         f"{int(np.isfinite(oof).sum()):,}/{n:,} OOF rows finite)")

    # Serving bundles AFTER the OOF npz: the expensive artifact is already
    # checkpointed, and a bundle failure must never invalidate it.
    write_serving_bundles(ctx, quick)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="AQNet v2 T1 statistical skeleton (GPBoost + "
                    "candidate-B ensemble)")
    ap.add_argument("--quick", action="store_true",
                    help="2 outer folds, LOSO capped at 4, small rounds")
    ap.add_argument("--frame", default=None,
                    help="frame_truth.parquet (default: artifacts/v2)")
    ap.add_argument("--folds", default=None,
                    help="folds2.json (default: artifacts/v2)")
    ap.add_argument("--out", default=None,
                    help="output npz (default: artifacts/v2/oof_tier1.npz)")
    args = ap.parse_args(argv)
    return run_skeleton(quick=args.quick, frame_path=args.frame,
                        folds_path=args.folds, out_path=args.out)


if __name__ == "__main__":
    sys.exit(main())
