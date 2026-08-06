"""
uq.py
AQNet v2 uncertainty stage: signed-residual quantile heads refit on the
DEPLOYED composed predictor, unit-level CQR scores, and NexCP
(nonexchangeable) weighted conformal deltas per coverage-density bin.

WHY this replaces v1's interval machinery (DESIGN §10, §0):
  * v1 fed ~77k sensor-day rows into split conformal as if independent; the
    effective sample is the number of UNITS (sensors/sites), so v1's
    row-level intervals were overconfident. Here the conformal score is
    unit-level — one number per unit, the unit's own finite-sample CQR
    quantile (fusion.conformal_intervals applied within-unit) — so honest
    n = number of calibration units, and NexCP weighting handles the
    non-exchangeability of a spatially/temporally distant query.
  * v1's conformal_recenter GRAFTED Tier-1 quantile band half-widths onto a
    different predictor's center — a build error in v2. The quantile heads
    here are refit on the residual of the deployed composite (y − oof_final)
    and uq_params.json records fitted_against_tier_hash = sha256 of the
    oof_composite.npz bytes, so a band that outlived its predictor is
    detectable, not silent.
  * models_tabular.train_quantile_cv clips predictions >= 0 — correct for
    PM2.5 levels, WRONG for signed residuals (a band that cannot go below
    its center has zero lower width). The small trainer below copies its
    early-stopping (temporal-tail) and crossing-repair (per-row monotone
    rearrangement) idioms WITHOUT the clip.

Calibration discipline (DESIGN §2): scores come ONLY from folds2
conformal_unit rows — a 25% PA-sensor set plus one confirmation-side AQS
fold, disjoint by construction from every selection/confirmation row — and
those rows are excluded from quantile-head training in every fold. Vault
units and the vault period never enter anything. Conformal PA sensors are in
no outer test fold (outer_fold == -1, always-train elsewhere) but ARE
excluded from every fold's training here, so their band is predicted with
the mean over all fold models — honest for them, and recorded.

NexCP: delta(q) = weighted conformal quantile of the unit scores with
w_u ∝ exp(−d_u/rho_s) · exp(−dt_u/tau), d_u = km from the query to the
unit centroid, dt_u = days from the reference date (window end) to the
unit's last observation; the query itself contributes the +1 mass at +inf
(Barber et al. NexCP construction), so an isolated query honestly gets an
infinite delta rather than a borrowed one. Deltas are computed per
coverage-density bin — the same 3 bins as the tier stratum contract
(nbr_pacal_count_50km: 0 / 1–3 / >= 4). The stage artifact records the
uniform-weight (rho -> inf limit) per-bin deltas as the serving fallback;
query-adaptive serving calls nexcp_delta directly.

Ship window (DESIGN §10): site-level coverage of [lo − delta, hi + delta]
on held-out outer rows must land in [0.88, 0.93]; the check is RECORDED
here (ship gating is validate2's job). Honest intervals will be wider than
v1's — pre-registered as the correct outcome.

Artifacts (config2.artifact):
  uq_params.json     unit scores table, NexCP params (rho_s, tau, alpha),
                     delta per coverage bin, fitted_against_tier_hash,
                     ship-window check
  quantile_oof.npz   lo f8[n], hi f8[n], q50 f8[n] (NaN where the band was
                     never honestly predicted), delta_bin i1[n] (per-row
                     coverage bin)

Run (stage CLI, idempotent, FORCE=1 to re-run):
    python uq.py [--quick]
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

import config2
import frame2

# ── Guarded heavy import (v1 models_tabular style) ──────────────────────────
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    lgb = None
    HAS_LGBM = False
    print("[uq] lightgbm not installed -- quantile heads degrade to NaN "
          "bands (pip install lightgbm)")

# ── Constants (config-style, DESIGN §10 defaults) ───────────────────────────
ALPHA = 0.10                      # 90% nominal intervals
Q_LEVELS = (0.05, 0.5, 0.95)
RHO_S_KM = 150.0                  # NexCP spatial decay scale
TAU_DAYS = 365.0                  # NexCP staleness decay scale
SHIP_COVERAGE = (0.88, 0.93)      # site-level coverage ship window
N_ESTIMATORS = 2000
QUICK_N_ESTIMATORS = 150
COVERAGE_BIN_LABELS = ("no_pa_50km", "pa_1_3_50km", "pa_ge4_50km")
COVERAGE_COUNT_COL = "nbr_pacal_count_50km"


def _say(msg):
    print(f"[aqnet2] uq: {msg}", flush=True)


# ── v1 fusion.conformal_intervals (lazy file-path import, frame2 idiom) ─────
_FUSION_CACHE = []


def _load_fusion():
    """v1 fusion module loaded by file path (no sys.path bootstrap needed;
    pipeline2 owns that). Cached; None with a printed degradation message
    when the load fails — the inline replica below takes over."""
    if _FUSION_CACHE:
        return _FUSION_CACHE[0]
    mod = None
    try:
        path = os.path.join(config2.V1_DIR, "fusion.py")
        spec = importlib.util.spec_from_file_location("aq2_v1_fusion", path)
        mod = importlib.util.module_from_spec(spec)
        buf = io.StringIO()  # fusion prints pykrige availability at import
        with contextlib.redirect_stdout(buf):
            spec.loader.exec_module(mod)
    except Exception as e:
        mod = None
        print(f"[uq] v1 fusion.py unavailable ({e}) -- using the inline "
              "conformal-quantile replica (identical math)")
    _FUSION_CACHE.append(mod)
    return mod


def _cqr_delta(y, lo, hi, alpha):
    """Per-group CQR conformal quantile. Reuses fusion.conformal_intervals
    (the kept-for-v2 pure function; audited identical math) with its
    per-call print suppressed — one line per unit would flood the stage
    log. Inline replica when the v1 module cannot be loaded."""
    fusion = _load_fusion()
    if fusion is not None:
        with contextlib.redirect_stdout(io.StringIO()):
            return float(fusion.conformal_intervals(y, lo, hi, alpha=alpha))
    y = np.asarray(y, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not ok.any():
        raise ValueError("conformal delta: no finite calibration rows")
    scores = np.maximum(lo[ok] - y[ok], y[ok] - hi[ok])
    n = len(scores)
    q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, q_level, method="higher"))


# ── Coverage-density bins (tier stratum contract: 0 / 1–3 / >= 4) ───────────
def coverage_bin_from_count(count):
    """nbr_pacal_count_50km -> bin i1: 0 = no PA within 50 km, 1 = 1–3,
    2 = >= 4. A NaN count means the neighbor block itself was absent —
    'no local information' is bin 0 by the same semantics frame2 uses for
    empty pools (an availability statement, not a fill)."""
    c = np.asarray(count, dtype=np.float64)
    b = np.zeros(c.shape, dtype=np.int8)
    b[c >= 1] = 1
    b[c >= 4] = 2
    return b


# ── Unit-level scores (INTERFACES frozen API) ───────────────────────────────
def unit_scores(y, pred, sigma, unit_id, mask, alpha=ALPHA):
    """One CQR conformal score per unit — honest n = units, not rows.

    y        target (FRM scale)
    pred     band center (the deployed composite prediction)
    sigma    band geometry: shape (n,) = symmetric half-width
             (band = pred ± sigma), or shape (n, 2) = SIGNED offsets
             [lo_off, hi_off] (band = pred + lo_off .. pred + hi_off —
             the quantile-residual convention, lo_off typically negative)
    unit_id  per-row unit ids (cast to str)
    mask     boolean row mask — which rows participate (the caller passes
             the folds2 conformal_unit rows)

    Score s_u = the unit's own finite-sample CQR quantile of its row scores
    max(lo − y, y − hi) at miscoverage alpha (fusion.conformal_intervals
    applied within-unit) — one number per unit, computed over that unit's
    rows only, so between-unit dependence can no longer masquerade as
    sample size. Units iterate in SORTED id order (dtype-stable).

    Returns DataFrame[unit_id(str), score, n_rows]; rows with non-finite
    y/lo/hi are dropped per unit (NaN-honest), units with no usable rows
    are absent. Empty input -> empty DataFrame with a printed warning.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    pr = np.asarray(pred, dtype=np.float64)
    if len(pr) != n:
        raise ValueError(f"pred has length {len(pr)}, expected {n}")
    sig = np.asarray(sigma, dtype=np.float64)
    if sig.ndim == 1:
        if len(sig) != n:
            raise ValueError(f"sigma has length {len(sig)}, expected {n}")
        lo, hi = pr - sig, pr + sig
    elif sig.ndim == 2 and sig.shape == (n, 2):
        lo, hi = pr + sig[:, 0], pr + sig[:, 1]
    elif sig.ndim == 2 and sig.shape == (2, n):
        lo, hi = pr + sig[0], pr + sig[1]
    else:
        raise ValueError(f"sigma must be shape ({n},), ({n}, 2) or (2, {n}); "
                         f"got {sig.shape}")
    ids = np.asarray(unit_id).astype(str)
    if len(ids) != n:
        raise ValueError(f"unit_id has length {len(ids)}, expected {n}")
    m = np.asarray(mask).astype(bool)
    if len(m) != n:
        raise ValueError(f"mask has length {len(m)}, expected {n}")

    ok = m & np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    units = np.unique(ids[ok])  # np.unique returns sorted — dtype-stable
    if len(units) == 0:
        _say("WARNING: unit_scores received no usable rows -- empty score "
             "table")
        return pd.DataFrame({"unit_id": pd.Series([], dtype=str),
                             "score": pd.Series([], dtype=np.float64),
                             "n_rows": pd.Series([], dtype=np.int64)})
    scores, counts = [], []
    for u in units:
        rows = ok & (ids == u)
        scores.append(_cqr_delta(y[rows], lo[rows], hi[rows], alpha))
        counts.append(int(rows.sum()))
    _say(f"unit scores: {len(units)} units, "
         f"median rows/unit {int(np.median(counts))}")
    return pd.DataFrame({"unit_id": units.astype(str),
                         "score": np.asarray(scores, dtype=np.float64),
                         "n_rows": np.asarray(counts, dtype=np.int64)})


def attach_unit_meta(scores_df, frame):
    """Join the per-unit metadata nexcp_delta needs onto a unit_scores
    table: centroid lat/lon (mean over the unit's rows), last_date (max
    observation date — staleness anchor), coverage_bin (bin of the unit's
    MEDIAN nbr_pacal_count_50km, the unit's typical density regime)."""
    cols = {"unit_id": frame["unit_id"].astype(str),
            "lat": frame["lat"].astype(float),
            "lon": frame["lon"].astype(float),
            "date": pd.to_datetime(frame["date"])}
    if COVERAGE_COUNT_COL in frame.columns:
        cols["_count"] = frame[COVERAGE_COUNT_COL].astype(float)
    else:
        _say(f"WARNING: frame has no {COVERAGE_COUNT_COL} column -- all "
             "units fall into coverage bin 0")
        cols["_count"] = np.nan
    g = pd.DataFrame(cols).groupby("unit_id", sort=True).agg(
        lat=("lat", "mean"), lon=("lon", "mean"), last_date=("date", "max"),
        _med_count=("_count", "median")).reset_index()
    g["coverage_bin"] = coverage_bin_from_count(g["_med_count"].to_numpy())
    g = g.drop(columns=["_med_count"])
    out = scores_df.copy()
    out["unit_id"] = out["unit_id"].astype(str)
    g["unit_id"] = g["unit_id"].astype(str)
    return out.merge(g, on="unit_id", how="left")


# ── NexCP weighted conformal quantile ───────────────────────────────────────
def _weighted_conformal_quantile(scores, weights, alpha):
    """Weighted conformal quantile with the query's +1 mass at +inf.

    Normalized weights p_u = w_u / (sum(w) + 1); the remaining mass is the
    query's own point mass at +inf (its unnormalized weight is 1: distance
    and staleness are zero at the query itself). Returns the smallest score
    s with cumulative mass >= 1 - alpha, or +inf when the calibration mass
    cannot reach it — an isolated query gets an honest infinite delta,
    never a borrowed finite one.
    """
    s = np.asarray(scores, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if len(s) != len(w):
        raise ValueError(f"scores/weights length mismatch: {len(s)} vs "
                         f"{len(w)}")
    ok = np.isfinite(s) & np.isfinite(w) & (w >= 0.0)
    s, w = s[ok], w[ok]
    if len(s) == 0:
        return float("inf")
    p = w / (w.sum() + 1.0)
    order = np.argsort(s, kind="stable")
    cum = np.cumsum(p[order])
    idx = int(np.searchsorted(cum, 1.0 - float(alpha), side="left"))
    if idx >= len(s):
        return float("inf")
    return float(s[order][idx])


def nexcp_delta(scores_df, query_latlon, coverage_bin, rho_s=RHO_S_KM,
                tau=TAU_DAYS, alpha=ALPHA, ref_date=None):
    """Query-adaptive NexCP conformal delta for one coverage-density bin.

    scores_df     unit score table carrying unit_id, score, lat, lon,
                  last_date, coverage_bin (unit_scores + attach_unit_meta)
    query_latlon  (lat, lon) of the query
    coverage_bin  the query's bin (coverage_bin_from_count of its own
                  nbr_pacal_count_50km) — only same-bin units calibrate it
    rho_s, tau    decay scales: w_u = exp(-d_u/rho_s) * exp(-dt_u/tau),
                  d_u km to the unit centroid, dt_u days from ref_date
                  (default config2.DATE_END, the window end) back to the
                  unit's last observation (clipped at 0)

    Returns the weighted conformal quantile at 1 - alpha; +inf when the
    bin is empty or the weighted mass cannot reach the level (see
    _weighted_conformal_quantile — default-honest, never default-tight).
    """
    req = {"unit_id", "score", "lat", "lon", "last_date", "coverage_bin"}
    missing = req - set(scores_df.columns)
    if missing:
        raise ValueError(f"scores_df is missing columns {sorted(missing)} "
                         "-- build it with unit_scores + attach_unit_meta")
    sub = scores_df.loc[
        scores_df["coverage_bin"].astype(int) == int(coverage_bin)]
    if len(sub) == 0:
        _say(f"nexcp_delta: no calibration units in coverage bin "
             f"{int(coverage_bin)} -- delta inf")
        return float("inf")
    qlat, qlon = float(query_latlon[0]), float(query_latlon[1])
    d_km = frame2._haversine_km(np.full(len(sub), qlat),
                                np.full(len(sub), qlon),
                                sub["lat"].to_numpy(dtype=np.float64),
                                sub["lon"].to_numpy(dtype=np.float64))
    ref = pd.Timestamp(ref_date if ref_date is not None
                       else config2.DATE_END)
    last = pd.to_datetime(sub["last_date"])
    dt_days = np.maximum(
        (ref - last).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0,
        0.0)
    w = np.exp(-d_km / float(rho_s)) * np.exp(-dt_days / float(tau))
    return _weighted_conformal_quantile(sub["score"].to_numpy(dtype=float),
                                        w, alpha)


# ── Lineage ─────────────────────────────────────────────────────────────────
def composite_hash(path):
    """sha256 over the raw oof_composite.npz FILE BYTES — the band's
    fitted_against lineage. Any regeneration of the composite (even
    metric-identical) changes the hash; a grafted band is a build error,
    and this is the tripwire validate2 checks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Signed-residual quantile trainer (small, clip-free) ─────────────────────
def _es_tail_split(n, dates=None):
    """Temporal-tail early-stopping split (models_tabular idiom): most
    recent ~10% of rows by date, capped 50k, floored 100; (idx, None) for
    small folds."""
    if n < 200:
        return np.arange(n), None
    order = (np.argsort(dates, kind="stable") if dates is not None
             else np.arange(n))
    n_es = min(max(int(round(0.1 * n)), 100), 50000, n // 2)
    return order[:-n_es], order[-n_es:]


def _apply_overrides(X, features, overrides, k):
    """Fold k's f{k}__{col} neighbor overrides swapped into a COPY of X
    (full-length arrays, train AND test rows — the v1 npz contract).
    Non-feature columns ('y', absent t0 cols) are skipped."""
    ov = (overrides or {}).get(k)
    if not ov:
        return X
    pos = {c: j for j, c in enumerate(features)}
    Xf = X.copy()
    applied = 0
    for col, arr in ov.items():
        if col not in pos:
            continue
        arr = np.asarray(arr, dtype=np.float64)
        if arr.shape != (X.shape[0],):
            raise ValueError(f"override for '{col}' fold {k} has shape "
                             f"{arr.shape}, expected ({X.shape[0]},)")
        Xf[:, pos[col]] = arr
        applied += 1
    if applied:
        _say(f"fold {k}: {applied} neighbor-override columns applied")
    return Xf


def fit_quantile_heads(frame, oof_final, folds, tier_hash, overrides=None,
                       quick=False, alpha=ALPHA):
    """Cross-fit LightGBM quantile heads for the SIGNED residual
    y − oof_final on frame features, per outer fold with neighbor
    overrides. models_tabular.train_quantile_cv's early-stopping and
    crossing-repair idioms are copied WITHOUT its >= 0 clip (signed
    residuals must be free to go negative — a clipped lower head would
    collapse the band's lower width to zero).

    Training rows per fold k: outer_fold != k, finite residual, NOT a
    conformal_unit row, NOT a vault row/period (calibration and vault
    disjointness — DESIGN §2). Predictions:
      outer_fold == k rows           fold-k model (standard OOF)
      conformal PA rows (outer -1)   mean over ALL fold models — honest,
                                     since every fold excluded them
      everything else                NaN (never honestly scorable)

    Returns {"oof_q": {q: f8[n] residual quantiles}, "lo", "hi", "q50"
    (oof_final + residual quantiles), "tier_hash", "meta"}. Degrades to
    all-NaN bands with a printed message when lightgbm is absent.
    """
    n = len(frame)
    of = np.asarray(oof_final, dtype=np.float64)
    if len(of) != n:
        raise ValueError(f"oof_final has length {len(of)}, expected {n}")
    y = frame["y"].to_numpy(dtype=np.float64)
    resid = y - of

    qs = sorted(float(q) for q in Q_LEVELS)
    oof_q = {q: np.full(n, np.nan) for q in qs}
    meta = {"tier_hash": str(tier_hash), "quantiles": qs,
            "alpha": float(alpha), "quick": bool(quick),
            "lightgbm": bool(HAS_LGBM)}

    def _bundle():
        rows = np.column_stack([oof_q[q] for q in qs])
        fin = np.all(np.isfinite(rows), axis=1)
        rows[fin] = np.sort(rows[fin], axis=1)  # crossing repair, NO clip
        for j, q in enumerate(qs):
            oof_q[q] = rows[:, j]
        return {"oof_q": oof_q,
                "lo": of + oof_q[qs[0]],
                "hi": of + oof_q[qs[-1]],
                "q50": of + oof_q[0.5] if 0.5 in oof_q else of.copy(),
                "tier_hash": str(tier_hash), "meta": meta}

    if not HAS_LGBM:
        _say("lightgbm unavailable -- quantile heads return NaN bands")
        meta["note"] = "lightgbm_unavailable"
        return _bundle()

    feats = frame2.feature_columns(frame)
    X = frame[feats].to_numpy(dtype=np.float64)
    dates_ns = pd.to_datetime(frame["date"]).to_numpy()

    outer = np.asarray(folds["outer_fold"], dtype=int)
    if len(outer) != n:
        raise ValueError(f"outer_fold length {len(outer)} != frame rows {n}")
    ks = sorted(int(v) for v in np.unique(outer) if v >= 0)
    if not ks:
        raise ValueError("no outer folds in folds2.json")

    conf_arr = folds.get("conformal_unit")
    conf_unit = (np.asarray(conf_arr, dtype=int) > 0 if conf_arr is not None
                 else np.zeros(n, bool))
    if conf_arr is None:
        _say("WARNING: no conformal_unit array -- quantile training cannot "
             "exclude calibration rows (folds2 must provide it)")
    vault = frame2._as_unit_set((folds or {}).get("vault_sites", []))
    ids = frame["unit_id"].astype(str)
    vault_m = (ids.isin(vault).to_numpy() if vault else np.zeros(n, bool))
    vstart = pd.Timestamp(getattr(config2, "VAULT_DATE_START", "2026-01-01"))
    vault_m = vault_m | (pd.to_datetime(frame["date"]) >= vstart).to_numpy()

    orphan = conf_unit & (outer < 0)  # conformal PA sensors: never OOF
    rounds = QUICK_N_ESTIMATORS if quick else N_ESTIMATORS
    orph_acc = {q: np.zeros(int(orphan.sum())) for q in qs}
    _say(f"quantile heads: {len(ks)} outer folds x {len(qs)} quantiles, "
         f"{int(orphan.sum()):,} conformal-orphan rows get the fold-mean "
         f"band")

    for k in ks:
        X_fold = _apply_overrides(X, feats, overrides, k)
        tr = ((outer != k) & np.isfinite(resid) & ~conf_unit & ~vault_m)
        te = outer == k
        tr_idx = np.flatnonzero(tr)
        head, tail = _es_tail_split(len(tr_idx), dates_ns[tr_idx])
        _say(f"fold {k}: {len(tr_idx):,} train rows, "
             f"{int(te.sum()):,} test rows")
        for q in qs:
            est = lgb.LGBMRegressor(
                objective="quantile", alpha=q,
                n_estimators=int(rounds), learning_rate=0.03, num_leaves=127,
                min_child_samples=60, subsample=0.7, subsample_freq=1,
                colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=3.0,
                n_jobs=-1, random_state=int(config2.SEED), verbose=-1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # sklearn name chatter
                if tail is not None:
                    h_rows, t_rows = tr_idx[head], tr_idx[tail]
                    est.fit(X_fold[h_rows], resid[h_rows],
                            eval_set=[(X_fold[t_rows], resid[t_rows])],
                            eval_metric="quantile",
                            callbacks=[lgb.early_stopping(100,
                                                          verbose=False)])
                else:
                    est.fit(X_fold[tr_idx], resid[tr_idx])
                oof_q[q][te] = est.predict(X_fold[te])   # NO clip: signed
                if orphan.any():
                    orph_acc[q] += est.predict(X_fold[orphan])

    if orphan.any():
        for q in qs:
            oof_q[q][orphan] = orph_acc[q] / len(ks)

    meta.update({"n_folds": len(ks), "n_estimators": rounds,
                 "features": list(feats),
                 "overrides_used": bool(overrides),
                 "excluded_from_training": {
                     "conformal_rows": int(conf_unit.sum()),
                     "vault_rows": int(vault_m.sum())},
                 "orphan_rows_fold_mean": int(orphan.sum())})
    return _bundle()


# ── Stage wiring ────────────────────────────────────────────────────────────
def _load_folds(path, frame):
    """folds2.load_folds when importable (hash-verified); raw JSON with an
    n_rows check otherwise (calibrate idiom, warned)."""
    if not os.path.exists(path):
        raise SystemExit(f"[aqnet2] uq: {path} missing -- run the folds "
                         "stage first")
    try:
        import folds2
        return folds2.load_folds(path, frame)
    except ImportError as e:
        print(f"[uq] folds2 module unavailable ({e}) -- loading folds2.json "
              "raw (content hash NOT verified)")
    with open(path, "r", encoding="utf-8") as fh:
        folds = json.load(fh)
    if int(folds.get("n_rows", -1)) != len(frame):
        raise SystemExit(f"[aqnet2] uq: folds2.json n_rows "
                         f"{folds.get('n_rows')} != frame rows {len(frame)}")
    return folds


def _load_overrides_if_any(n_rows, path=None):
    cands = [path] if path else [config2.artifact("nbr_overrides_outer.npz"),
                                 config2.artifact("nbr_overrides.npz")]
    for p in cands:
        if p and os.path.exists(p):
            _say(f"neighbor overrides loaded from {p}")
            return frame2.load_overrides(p, n_rows)
    _say("WARNING: no outer-fold neighbor-overrides npz found -- quantile "
         "heads see deployment-view neighbor columns (recorded)")
    return None


def _bin_deltas(scores_meta, alpha):
    """Per-coverage-bin uniform-weight conformal deltas (the rho -> inf
    NexCP limit — the serving fallback the artifact records). An empty bin
    inherits the pooled all-units delta, recorded as such (conservative
    fallback, never silent)."""
    all_scores = scores_meta["score"].to_numpy(dtype=np.float64)
    pooled = _weighted_conformal_quantile(all_scores,
                                          np.ones(len(all_scores)), alpha)
    out = {}
    for b, label in enumerate(COVERAGE_BIN_LABELS):
        sub = scores_meta.loc[scores_meta["coverage_bin"].astype(int) == b,
                              "score"].to_numpy(dtype=np.float64)
        if len(sub):
            d = _weighted_conformal_quantile(sub, np.ones(len(sub)), alpha)
            src = "bin"
        else:
            d, src = pooled, "pooled_fallback"
        out[str(b)] = {"label": label, "delta": d if np.isfinite(d) else None,
                       "n_units": int(len(sub)), "source": src}
        _say(f"bin {b} ({label}): delta={d:.4g} n_units={len(sub)} "
             f"source={src}")
    # Monotone enforcement (methodology-audit fix): bin 0 = sparsest
    # coverage = largest expected error. A sparse bin whose delta came out
    # SMALLER than a denser bin's (sampling noise on ~tens of units, or
    # the pooled fallback dominated by dense units) would under-cover
    # exactly where the product matters. Enforce delta_0 >= delta_1 >=
    # delta_2 by propagating the running max from dense to sparse; both
    # raw and enforced values are recorded.
    prev = None
    for b in range(len(COVERAGE_BIN_LABELS) - 1, -1, -1):
        e = out[str(b)]
        raw = e["delta"]
        if raw is not None:
            enforced = raw if prev is None else max(raw, prev)
            if enforced != raw:
                e["delta_raw_pre_monotone"] = raw
                e["source"] += "+monotone_raise"
                _say(f"bin {b}: delta raised {raw:.4g} -> {enforced:.4g} "
                     "(monotone-in-sparsity enforcement)")
            e["delta"] = enforced
            prev = enforced
    out["pooled"] = {"label": "all_units",
                     "delta": pooled if np.isfinite(pooled) else None,
                     "n_units": int(len(all_scores)), "source": "pooled"}
    out["monotone_enforced"] = True
    return out


def _ship_window_check(frame, folds, y, lo, hi, row_bin, deltas,
                       conf_unit, vault_m):
    """Site-level coverage of [lo - delta_bin, hi + delta_bin] on held-out
    outer rows (not conformal, not vault). Recorded, not gated — validate2
    owns the ship decision."""
    outer = np.asarray(folds["outer_fold"], dtype=int)
    d_row = np.full(len(frame), np.nan)
    for b in range(len(COVERAGE_BIN_LABELS)):
        d = deltas[str(b)]["delta"]
        if d is not None:
            d_row[row_bin == b] = float(d)
    ev = ((outer >= 0) & ~conf_unit & ~vault_m & np.isfinite(y)
          & np.isfinite(lo) & np.isfinite(hi) & np.isfinite(d_row))
    result = {"window": list(SHIP_COVERAGE), "n_rows": int(ev.sum()),
              "n_sites": 0, "site_coverage": None, "in_window": None}
    if not ev.any():
        _say("ship-window check: no evaluable held-out rows -- recorded as "
             "undefined")
        return result
    covered = ((y[ev] >= lo[ev] - d_row[ev])
               & (y[ev] <= hi[ev] + d_row[ev])).astype(np.float64)
    units = frame["unit_id"].astype(str).to_numpy()[ev]
    per_site = pd.DataFrame({"unit_id": units, "cov": covered}).groupby(
        "unit_id", sort=True)["cov"].mean()
    cov = float(per_site.mean())
    result.update({"n_sites": int(len(per_site)), "site_coverage": cov,
                   "in_window": bool(SHIP_COVERAGE[0] <= cov
                                     <= SHIP_COVERAGE[1])})
    _say(f"ship-window check: site coverage {cov:.4f} over "
         f"{len(per_site)} sites -- in [{SHIP_COVERAGE[0]}, "
         f"{SHIP_COVERAGE[1]}]: {result['in_window']}")
    return result


def _jsonable(obj):
    """JSON-safe conversion (calibrate idiom): numpy scalars -> python,
    non-finite -> None, Timestamps -> ISO date strings."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    return obj


def run_uq(quick=False, folds_path=None, overrides_path=None):
    dest_params = config2.artifact("uq_params.json")
    dest_npz = config2.artifact("quantile_oof.npz")
    if (os.path.exists(dest_params) and os.path.exists(dest_npz)
            and os.environ.get("FORCE") != "1"):
        _say(f"{dest_params} exists (FORCE=1 to rebuild) -- skip")
        return 0

    print("[aqnet2] ── stage: uq ──")
    frame_path = config2.artifact("frame_truth.parquet")
    if not os.path.exists(frame_path):
        raise SystemExit("[aqnet2] uq: frame_truth.parquet missing -- run "
                         "the features stage first")
    frame = pd.read_parquet(frame_path)
    folds = _load_folds(folds_path or config2.artifact("folds2.json"), frame)

    comp_path = config2.artifact("oof_composite.npz")
    if not os.path.exists(comp_path):
        raise SystemExit("[aqnet2] uq: oof_composite.npz missing -- run the "
                         "gates stage first")
    tier_hash = composite_hash(comp_path)
    with np.load(comp_path) as z:
        oof_final = np.asarray(z["oof_final"], dtype=np.float64)
    if len(oof_final) != len(frame):
        raise SystemExit(f"[aqnet2] uq: oof_final length {len(oof_final)} "
                         f"!= frame rows {len(frame)}")
    _say(f"fitted_against_tier_hash = {tier_hash[:16]}... "
         f"(sha256 of oof_composite.npz bytes)")

    overrides = _load_overrides_if_any(len(frame), overrides_path)
    qh = fit_quantile_heads(frame, oof_final, folds, tier_hash,
                            overrides=overrides, quick=quick)
    lo, hi, q50 = qh["lo"], qh["hi"], qh["q50"]
    y = frame["y"].to_numpy(dtype=np.float64)

    # ── Unit scores on the conformal calibration rows only ──
    conf_arr = folds.get("conformal_unit")
    n = len(frame)
    conf_unit = (np.asarray(conf_arr, dtype=int) > 0 if conf_arr is not None
                 else np.zeros(n, bool))
    vault = frame2._as_unit_set(folds.get("vault_sites", []))
    ids = frame["unit_id"].astype(str)
    vault_m = ids.isin(vault).to_numpy() if vault else np.zeros(n, bool)
    vstart = pd.Timestamp(getattr(config2, "VAULT_DATE_START", "2026-01-01"))
    vault_m = vault_m | (pd.to_datetime(frame["date"]) >= vstart).to_numpy()

    offsets = np.column_stack([lo - oof_final, hi - oof_final])
    scores = unit_scores(y, oof_final, offsets, frame["unit_id"],
                         conf_unit & ~vault_m, alpha=ALPHA)
    scores_meta = attach_unit_meta(scores, frame)

    deltas = _bin_deltas(scores_meta, ALPHA)
    if COVERAGE_COUNT_COL in frame.columns:
        row_bin = coverage_bin_from_count(
            frame[COVERAGE_COUNT_COL].to_numpy(dtype=np.float64))
    else:
        row_bin = np.zeros(n, dtype=np.int8)
    ship = _ship_window_check(frame, folds, y, lo, hi, row_bin, deltas,
                              conf_unit, vault_m)

    # ── Artifacts (atomic) ──
    tmp = dest_npz + ".tmp.npz"
    np.savez_compressed(tmp, lo=lo.astype(np.float64),
                        hi=hi.astype(np.float64),
                        q50=q50.astype(np.float64),
                        delta_bin=row_bin.astype(np.int8))
    os.replace(tmp, dest_npz)
    _say(f"wrote {dest_npz} (lo/hi/q50 f8[{n}], delta_bin i1[{n}])")

    params = {
        "alpha": ALPHA,
        "rho_s_km": RHO_S_KM,
        "tau_days": TAU_DAYS,
        "ref_date": str(config2.DATE_END),
        "coverage_bins": {"column": COVERAGE_COUNT_COL,
                          "edges": "0 / 1-3 / >=4",
                          "labels": list(COVERAGE_BIN_LABELS)},
        "delta_by_bin": deltas,
        "fitted_against_tier_hash": tier_hash,
        "ship_window": ship,
        "unit_scores": scores_meta.to_dict(orient="records"),
        "n_calibration_units": int(len(scores_meta)),
        "quantile_meta": qh["meta"],
        "quick": bool(quick),
    }
    tmp = dest_params + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(params), fh, indent=2, sort_keys=True)
    os.replace(tmp, dest_params)
    _say(f"wrote {dest_params}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="AQNet v2 unit-level NexCP conformal UQ (stage: uq)")
    ap.add_argument("--quick", action="store_true",
                    help="fewer boosting rounds (smoke test)")
    ap.add_argument("--folds", default=None,
                    help="path to folds2.json (default: config2.ARTIFACTS_DIR)")
    ap.add_argument("--overrides", default=None,
                    help="outer-fold neighbor-overrides npz (optional)")
    args = ap.parse_args(argv)
    return run_uq(quick=args.quick, folds_path=args.folds,
                  overrides_path=args.overrides)


if __name__ == "__main__":
    sys.exit(main())
