"""Evaluation harness for AQNet: honest folds, metrics, spatial baselines, and
EPA AQS external validation.

Everything here is deliberately conservative about leakage:

  * Folds are grouped by sensor (the LOSO ethos of the production ensemble) or
    by spatial block (KMeans regions over sensor coordinates), never by random
    rows — random row splits leak spatial autocorrelation and inflate R².
  * Interpolation baselines (nearest / IDW / per-day ordinary kriging) are
    strictly out-of-fold: a test row is only ever interpolated from that
    fold's TRAIN sensors on the SAME day.
  * EPA AQS FRM/FEM data is EXTERNAL VALIDATION ONLY. external_aqs_validation
    builds the model's feature vector at AQS site-days with
    features.build_site_features — PurpleAir sensors and gridded products
    alone, with neighbor aggregates over the corrected-target pool (training
    units) — AQS concentrations never appear in any model input; they are
    only the ground truth predictions are scored against.

Folds are lists of (train_idx, test_idx) POSITIONAL index arrays into the
training frame's row order, shared with models_tabular.py and fusion.py.

Run order: pipeline_colab.py drives these functions; nothing here trains a
model or writes an artifact on import.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

# ── Path bootstrap (identical across aqnet modules) ─────────────────────────

_AQNET_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_AQNET_DIR))
_DEEP_DIR = os.path.join(_ROOT, "research", "deeplearning")
_PIPELINE_DIR = os.path.join(_ROOT, "pipeline")
for _p in (_AQNET_DIR, _DEEP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _PIPELINE_DIR not in sys.path:
    sys.path.append(_PIPELINE_DIR)  # appended last; features.py resolves
    # pipeline/neighbor_features through it when imported lazily from here

import config

# ── Constants ───────────────────────────────────────────────────────────────

EARTH_R_KM = 6371.0

# EPA 2024 daily PM2.5 AQI breakpoints (µg/m³, 24-hour average, concentrations
# truncated to 0.1 µg/m³ before categorization). Source: EPA AQI Technical
# Assistance Document EPA-454/B-24-002 (May 2024), implementing the 2024 PM
# NAAQS final rule (89 FR 16202, March 6, 2024):
#   Good            0.0 –   9.0
#   Moderate        9.1 –  35.4
#   USG            35.5 –  55.4   (Unhealthy for Sensitive Groups)
#   Unhealthy      55.5 – 125.4
#   Very Unhealthy 125.5 – 225.4
#   Hazardous      225.5+
AQI_UPPER_BOUNDS = np.array([9.0, 35.4, 55.4, 125.4, 225.4])
AQI_CATEGORIES = ["Good", "Moderate", "USG", "Unhealthy",
                  "Very Unhealthy", "Hazardous"]
EXCEEDANCE_THRESHOLD = 35.4  # > 35.4 µg/m³ = USG or worse


# ── Small shared helpers ────────────────────────────────────────────────────

def _latlon(df):
    """(lat, lon) float64 arrays; accepts either lat/lon or latitude/longitude."""
    lat_col = "lat" if "lat" in df.columns else "latitude"
    lon_col = "lon" if "lon" in df.columns else "longitude"
    return (df[lat_col].to_numpy(dtype=np.float64),
            df[lon_col].to_numpy(dtype=np.float64))


def _norm_days(df):
    """Normalized (midnight) datetime64 array for the frame's date column."""
    return pd.to_datetime(df["date"]).dt.normalize().to_numpy()


def _group_positions(keys, positions):
    """{key -> np.ndarray of `positions` entries whose aligned key matches}.

    `keys` and `positions` are equal-length arrays; positions are row indices
    into the full frame. Used to bucket fold indices by day.
    """
    positions = np.asarray(positions)
    grp = pd.Series(np.arange(len(positions))).groupby(
        pd.Series(np.asarray(keys))).indices
    return {k: positions[np.asarray(v)] for k, v in grp.items()}


def _idw_predict(p_coords_rad, p_vals, q_coords_rad, k=8, power=2.0):
    """Haversine k-NN inverse-distance interpolation; k=1 is nearest-value.

    Coordinates are already in radians. Distances are floored at 1 m so a
    query point sitting exactly on a pool point takes (almost exactly) that
    point's value instead of dividing by zero.
    """
    from sklearn.neighbors import BallTree

    tree = BallTree(p_coords_rad, metric="haversine")
    k_eff = int(min(k, len(p_vals)))
    dist, ind = tree.query(q_coords_rad, k=k_eff)
    if k_eff == 1:
        return np.asarray(p_vals, dtype=np.float64)[ind[:, 0]]
    d_km = dist * EARTH_R_KM
    w = 1.0 / np.maximum(d_km, 1e-3) ** power
    vals = np.asarray(p_vals, dtype=np.float64)[ind]
    return (w * vals).sum(axis=1) / w.sum(axis=1)


# ── Fold constructors ───────────────────────────────────────────────────────

def make_loso_folds(df, n_folds=10, seed=42):
    """Grouped K-fold over sensor_id (leave-sensors-out).

    Every sensor's rows land in exactly one test fold, so each fold scores the
    model on sensors it never trained on — the protocol behind the production
    LOSO benchmark. Sensors are shuffled with `seed` before being dealt into
    folds (a deterministic, version-proof shuffled GroupKFold). Returns
    [(train_idx, test_idx), ...] positional index arrays.
    """
    sensors = df["sensor_id"].to_numpy()
    uniq = np.unique(sensors)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_folds = int(min(n_folds, len(uniq)))
    all_idx = np.arange(len(df))
    folds = []
    for part in np.array_split(uniq, n_folds):
        te_mask = np.isin(sensors, part)
        folds.append((all_idx[~te_mask], all_idx[te_mask]))
    return folds


def make_spatial_block_folds(df, n_blocks=5, seed=42):
    """Leave-one-region-out folds: KMeans blocks over unique sensor coords.

    Harsher than LOSO — a held-out sensor loses not just itself but its whole
    geographic neighborhood, so neighbor features can't lean on 25 km context.
    This is the split that measures true spatial extrapolation. Returns the
    same [(train_idx, test_idx), ...] structure as make_loso_folds.
    """
    from sklearn.cluster import KMeans

    sensors = df["sensor_id"].to_numpy()
    lat, lon = _latlon(df)
    site = (pd.DataFrame({"sensor_id": sensors, "lat": lat, "lon": lon})
            .groupby("sensor_id", as_index=False)
            .mean(numeric_only=True))
    n_blocks = int(min(n_blocks, len(site)))
    km = KMeans(n_clusters=n_blocks, random_state=seed, n_init=10)
    labels = km.fit_predict(site[["lat", "lon"]].to_numpy(dtype=np.float64))
    sensor_block = dict(zip(site["sensor_id"].to_numpy(), labels))
    row_block = np.array([sensor_block[s] for s in sensors])
    all_idx = np.arange(len(df))
    folds = []
    for b in range(n_blocks):
        te_mask = row_block == b
        if te_mask.any() and (~te_mask).any():
            folds.append((all_idx[~te_mask], all_idx[te_mask]))
    return folds


def temporal_split(df, cutoff=None):
    """(train_idx, test_idx): rows strictly before `cutoff` vs at/after it.

    Default cutoff is config.TEMPORAL_CUTOFF. Measures forward-in-time
    generalization (all sensors seen, future days unseen) — complementary to
    the spatial splits, not a substitute for them.
    """
    cutoff = pd.Timestamp(cutoff if cutoff is not None else config.TEMPORAL_CUTOFF)
    d = pd.to_datetime(df["date"]).dt.normalize()
    te_mask = (d >= cutoff).to_numpy()
    all_idx = np.arange(len(df))
    return all_idx[~te_mask], all_idx[te_mask]


# ── Metrics ─────────────────────────────────────────────────────────────────

def metrics(y_true, y_pred):
    """r2 / rmse / mae / bias / n over finite (y_true, y_pred) pairs.

    bias is mean(pred - true): positive = systematic over-prediction. Rows
    where either side is NaN (e.g. a baseline had no same-day neighbors) are
    dropped and reflected in the returned n.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(ok.sum())
    if n == 0:
        return {"r2": float("nan"), "rmse": float("nan"), "mae": float("nan"),
                "bias": float("nan"), "n": 0}
    yt, yp = y_true[ok], y_pred[ok]
    err = yp - yt
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return {
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "n": n,
    }


def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=0, cluster=None):
    """Percentile-bootstrap 95% CIs for R² and RMSE via paired resampling.

    cluster : optional array of per-row cluster labels (sensor ids). When
    given, unique clusters are resampled with replacement and every row of a
    drawn cluster enters the replicate (cluster bootstrap) — the honest CI
    when rows within a sensor are correlated, which sensor-day panels always
    are. Default None keeps the iid row bootstrap.

    Returns {"r2": (lo, hi), "rmse": (lo, hi)}. NaN pairs are dropped first;
    degenerate resamples (zero variance) contribute NaN and are ignored by
    the percentile via nanpercentile.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[ok], y_pred[ok]
    n = len(yt)
    nan_pair = (float("nan"), float("nan"))
    if n < 2:
        return {"r2": nan_pair, "rmse": nan_pair}
    rng = np.random.default_rng(seed)

    cluster_rows = None
    if cluster is not None:
        cl = np.asarray(cluster)[ok]
        cluster_rows = list(
            pd.Series(np.arange(n)).groupby(pd.Series(cl)).indices.values())
        if len(cluster_rows) < 2:
            cluster_rows = None  # one cluster: fall back to iid rows

    r2s = np.full(n_boot, np.nan)
    rmses = np.full(n_boot, np.nan)
    for b in range(n_boot):
        if cluster_rows is not None:
            picks = rng.integers(0, len(cluster_rows), len(cluster_rows))
            idx = np.concatenate([cluster_rows[i] for i in picks])
        else:
            idx = rng.integers(0, n, n)
        t, p = yt[idx], yp[idx]
        err = p - t
        rmses[b] = np.sqrt(np.mean(err ** 2))
        ss_tot = np.sum((t - t.mean()) ** 2)
        if ss_tot > 0:
            r2s[b] = 1.0 - np.sum(err ** 2) / ss_tot
    r2_lo, r2_hi = np.nanpercentile(r2s, [2.5, 97.5])
    rm_lo, rm_hi = np.nanpercentile(rmses, [2.5, 97.5])
    return {"r2": (float(r2_lo), float(r2_hi)),
            "rmse": (float(rm_lo), float(rm_hi))}


def morans_i(residuals, lat, lon, k=8):
    """Moran's I of residuals under row-standardized k-NN weights (haversine).

    Positive values mean spatially clustered residuals — structure the model
    failed to absorb (and the signal residual kriging can harvest); values
    near E[I] = -1/(n-1) indicate spatial randomness. Pass residuals for a
    single day, or per-sensor aggregated residuals: pooling many days at the
    same coordinates makes the k-NN graph degenerate.

    With row-standardized weights (each of the k neighbors gets 1/k) the
    statistic reduces to sum(z_i * zbar_nbr_i) / sum(z_i^2).
    """
    from sklearn.neighbors import BallTree

    r = np.asarray(residuals, dtype=np.float64)
    la = np.asarray(lat, dtype=np.float64)
    lo = np.asarray(lon, dtype=np.float64)
    ok = np.isfinite(r) & np.isfinite(la) & np.isfinite(lo)
    r, la, lo = r[ok], la[ok], lo[ok]
    n = len(r)
    if n < k + 2:
        return float("nan")
    z = r - r.mean()
    denom = float(np.sum(z ** 2))
    if denom <= 0:
        return float("nan")
    coords = np.radians(np.column_stack([la, lo]))
    tree = BallTree(coords, metric="haversine")
    dist, ind = tree.query(coords, k=k + 1)
    # Drop self by INDEX, not by position: with duplicate coordinates the
    # zero-distance ties may reorder, so push the self entry to the end of the
    # sort key and keep the nearest k genuine neighbors.
    self_mask = ind == np.arange(n)[:, None]
    sort_key = np.where(self_mask, np.inf, dist)
    take = np.argsort(sort_key, axis=1, kind="stable")[:, :k]
    nbrs = np.take_along_axis(ind, take, axis=1)
    lag = z[nbrs].mean(axis=1)
    return float(np.sum(z * lag) / denom)


def morans_i_daily(residuals, lat, lon, day_ids, k=8, min_sensors=15):
    """Per-day Moran's I of residuals, summarized across days.

    morans_i is only meaningful within a single day (pooling days stacks many
    rows on identical coordinates and degenerates the k-NN graph); this
    wrapper applies it day by day. Days with fewer than min_sensors rows with
    finite residuals are skipped — tiny days make the statistic wildly noisy.

    residuals/lat/lon/day_ids: aligned per-row arrays (day_ids is any
    hashable day key, e.g. normalized datetime64).

    Returns {"mean", "median", "iqr", "n_days"} over the per-day I values
    (iqr = 75th - 25th percentile); all NaN with n_days=0 when no day
    qualifies.
    """
    r = np.asarray(residuals, dtype=np.float64)
    la = np.asarray(lat, dtype=np.float64)
    lo = np.asarray(lon, dtype=np.float64)
    days = np.asarray(day_ids)
    ok = np.isfinite(r) & np.isfinite(la) & np.isfinite(lo)

    vals = []
    by_day = pd.Series(np.arange(len(r))).groupby(pd.Series(days)).indices
    for d, pos in by_day.items():
        pos = np.asarray(pos)
        pos = pos[ok[pos]]
        if len(pos) < min_sensors:
            continue
        i_d = morans_i(r[pos], la[pos], lo[pos], k=k)
        if np.isfinite(i_d):
            vals.append(i_d)
    if not vals:
        return {"mean": float("nan"), "median": float("nan"),
                "iqr": float("nan"), "n_days": 0}
    v = np.asarray(vals, dtype=np.float64)
    q25, q75 = np.percentile(v, [25.0, 75.0])
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "iqr": float(q75 - q25), "n_days": int(len(v))}


def aqi_category(pm25):
    """Daily PM2.5 (µg/m³) -> 0..5 AQI category index (see AQI_CATEGORIES).

    Follows EPA rounding convention: concentrations are truncated (not
    rounded) to 0.1 µg/m³ before comparison against the breakpoints; negative
    inputs are clipped to 0.
    """
    v = np.maximum(np.asarray(pm25, dtype=np.float64), 0.0)
    v = np.floor(v * 10.0 + 1e-9) / 10.0
    return np.searchsorted(AQI_UPPER_BOUNDS, v, side="left")


def aqi_category_metrics(y_true, y_pred):
    """Categorical agreement under the EPA 2024 daily PM2.5 AQI breakpoints.

    Returns category accuracy, macro-F1 over the categories present in
    y_true, and precision/recall for the exceedance event (> 35.4 µg/m³,
    i.e. USG or worse) — the operational question "would the model have
    flagged the bad day?". Precision/recall are NaN when undefined (no
    predicted / no observed exceedances respectively).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(ok.sum())
    if n == 0:
        return {"category_accuracy": float("nan"), "macro_f1": float("nan"),
                "exceedance_precision": float("nan"),
                "exceedance_recall": float("nan"),
                "n": 0, "n_exceedance_true": 0}
    ct = aqi_category(y_true[ok])
    cp = aqi_category(y_pred[ok])

    f1s = []
    for c in np.unique(ct):
        tp = int(np.sum((cp == c) & (ct == c)))
        fp = int(np.sum((cp == c) & (ct != c)))
        fn = int(np.sum((cp != c) & (ct == c)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)

    exc_t = ct >= 2  # USG or worse == > 35.4 µg/m³ (Moderate tops out at 35.4)
    exc_p = cp >= 2
    tp = int(np.sum(exc_p & exc_t))
    n_pred_pos = int(exc_p.sum())
    n_true_pos = int(exc_t.sum())
    return {
        "category_accuracy": float(np.mean(ct == cp)),
        "macro_f1": float(np.mean(f1s)),
        "exceedance_precision": tp / n_pred_pos if n_pred_pos > 0 else float("nan"),
        "exceedance_recall": tp / n_true_pos if n_true_pos > 0 else float("nan"),
        "n": n,
        "n_exceedance_true": n_true_pos,
    }


def strata_metrics(y, pred, smoke_flag, dust_vals):
    """Metrics stratified by aerosol regime: smoke / dust / clean.

    smoke_flag : per-row HMS smoke tier (rows with flag > 0 are "smoke";
                 NaN counts as no smoke).
    dust_vals  : per-row dust proxy (CAMS dust or merra2_dust25); "dust" rows
                 exceed the 75th percentile of the finite dust values. When
                 no finite dust value exists the dust stratum is empty.
    "clean" is everything that is neither smoke nor dust (smoke and dust may
    overlap — they are diagnostic strata, not a partition of blame).

    Returns {"smoke": metrics(...), "dust": metrics(...), "clean": ...} —
    the strata where satellite-driven components should earn their keep.
    """
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    smoke = np.nan_to_num(np.asarray(smoke_flag, dtype=np.float64), nan=0.0) > 0
    dust_vals = np.asarray(dust_vals, dtype=np.float64)
    finite_dust = dust_vals[np.isfinite(dust_vals)]
    if len(finite_dust):
        thr = float(np.percentile(finite_dust, 75.0))
        dust = np.isfinite(dust_vals) & (dust_vals > thr)
    else:
        dust = np.zeros(len(y), dtype=bool)
    clean = ~smoke & ~dust
    return {"smoke": metrics(y[smoke], pred[smoke]),
            "dust": metrics(y[dust], pred[dust]),
            "clean": metrics(y[clean], pred[clean])}


def spatial_temporal_r2(y, pred, sensor_ids):
    """Decompose skill into between-sensor and within-sensor components.

    spatial_r2:  R² of per-sensor mean(pred) against per-sensor mean(y) —
                 does the model rank LOCATIONS correctly (the exposure-
                 assessment question)?
    temporal_r2: R² of the anomalies (y - site mean) vs (pred - pred site
                 mean), pooled over rows — does the model track day-to-day
                 VARIATION at a site once its level is removed?

    A model can score a high pooled R² almost entirely on spatial contrast;
    this decomposition makes that visible. NaN pairs are dropped first.
    Returns {"spatial_r2", "temporal_r2", "n_sensors", "n"}.
    """
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    sid = np.asarray(sensor_ids)
    ok = np.isfinite(y) & np.isfinite(pred)
    if not ok.any():
        return {"spatial_r2": float("nan"), "temporal_r2": float("nan"),
                "n_sensors": 0, "n": 0}
    d = pd.DataFrame({"sid": sid[ok], "y": y[ok], "pred": pred[ok]})
    site = d.groupby("sid")[["y", "pred"]].mean()
    d = d.join(site, on="sid", rsuffix="_site")
    return {
        "spatial_r2": metrics(site["y"], site["pred"])["r2"],
        "temporal_r2": metrics(d["y"] - d["y_site"],
                               d["pred"] - d["pred_site"])["r2"],
        "n_sensors": int(len(site)),
        "n": int(len(d)),
    }


# ── Out-of-fold spatial baselines ───────────────────────────────────────────
# These answer "does the model beat plain geostatistics?" — every test row is
# interpolated from the SAME DAY's train-fold sensors only, so the baselines
# face exactly the fold discipline the model does.

def _baseline_interp(df, folds, k, power=2.0):
    """Shared engine for the nearest / IDW baselines (per fold, per day)."""
    lat, lon = _latlon(df)
    coords = np.radians(np.column_stack([lat, lon]))
    y = df["target"].to_numpy(dtype=np.float64)
    day = _norm_days(df)
    oof = np.full(len(df), np.nan)
    for tr, te in folds:
        tr = np.asarray(tr)
        te = np.asarray(te)
        tr_ok = tr[np.isfinite(y[tr])]
        tr_by_day = _group_positions(day[tr_ok], tr_ok)
        te_by_day = _group_positions(day[te], te)
        for d, q_pos in te_by_day.items():
            p_pos = tr_by_day.get(d)
            if p_pos is None or len(p_pos) == 0:
                continue
            oof[q_pos] = _idw_predict(coords[p_pos], y[p_pos],
                                      coords[q_pos], k=k, power=power)
    return oof


def baseline_nearest(df, folds):
    """Nearest same-day train-fold sensor value. The floor any model must beat."""
    return _baseline_interp(df, folds, k=1)


def baseline_idw(df, folds, k=8):
    """Inverse-distance-squared mean of the k nearest same-day train sensors."""
    return _baseline_interp(df, folds, k=k)


def _import_ordinary_kriging():
    """pykrige's OrdinaryKriging class, or None when pykrige is absent."""
    try:
        from pykrige.ok import OrdinaryKriging
        return OrdinaryKriging
    except ImportError:
        return None


def _krige_or_idw(p_lat, p_lon, p_vals, q_lat, q_lon, ok_cls):
    """One day's ordinary-kriging solve with IDW(k=8) fallback.

    The shared engine behind baseline_kriging and krige_to_sites. Kriging
    (exponential variogram, geographic coordinates) is attempted only when
    pykrige is importable (ok_cls not None), >= 5 train points exist, and the
    field is not near-constant; masked/non-finite kriging output or any
    numerical failure inside the variogram fit / solve falls back to IDW.
    No clipping here — callers clip when the field is non-negative (PM2.5
    yes, residuals no). Returns (pred, used_kriging).
    """
    p_lat = np.asarray(p_lat, dtype=np.float64)
    p_lon = np.asarray(p_lon, dtype=np.float64)
    p_vals = np.asarray(p_vals, dtype=np.float64)
    q_lat = np.asarray(q_lat, dtype=np.float64)
    q_lon = np.asarray(q_lon, dtype=np.float64)
    if (ok_cls is not None and len(p_vals) >= 5
            and float(np.ptp(p_vals)) > 1e-9):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ok_model = ok_cls(
                    p_lon, p_lat, p_vals,
                    variogram_model="exponential",
                    coordinates_type="geographic",
                    enable_plotting=False, verbose=False)
                z, _ = ok_model.execute("points", q_lon, q_lat)
            z = np.asarray(np.ma.filled(z, np.nan), dtype=np.float64)
            if np.all(np.isfinite(z)):
                return z, True
        except Exception:
            pass
    pred = _idw_predict(np.radians(np.column_stack([p_lat, p_lon])), p_vals,
                        np.radians(np.column_stack([q_lat, q_lon])),
                        k=8, power=2.0)
    return pred, False


def baseline_kriging(df, folds, max_train_per_day=150):
    """Per-day ordinary kriging of train-fold values, evaluated at test rows.

    pykrige OrdinaryKriging with an exponential variogram on geographic
    coordinates (via the shared _krige_or_idw engine). Train points are
    subsampled to max_train_per_day (kriging solves a dense system per day)
    with a fixed rng for reproducibility. Any per-day failure — singular
    variogram, near-constant field, too few points, or pykrige missing
    entirely — falls back to IDW (k=8) for that day. Predictions are clipped
    at 0: kriging happily extrapolates below zero, PM2.5 cannot.
    """
    ok_cls = _import_ordinary_kriging()
    if ok_cls is None:
        print("[baseline_kriging] pykrige not installed — every day falls back "
              "to IDW(k=8). pip install pykrige for the true kriging baseline.")

    lat, lon = _latlon(df)
    y = df["target"].to_numpy(dtype=np.float64)
    day = _norm_days(df)
    oof = np.full(len(df), np.nan)
    rng = np.random.default_rng(0)
    n_fall, n_days = 0, 0
    for tr, te in folds:
        tr = np.asarray(tr)
        te = np.asarray(te)
        tr_ok = tr[np.isfinite(y[tr])]
        tr_by_day = _group_positions(day[tr_ok], tr_ok)
        te_by_day = _group_positions(day[te], te)
        for d, q_pos in te_by_day.items():
            p_pos = tr_by_day.get(d)
            if p_pos is None or len(p_pos) == 0:
                continue
            n_days += 1
            if len(p_pos) > max_train_per_day:
                p_pos = rng.choice(p_pos, size=max_train_per_day, replace=False)
            pred, used_kriging = _krige_or_idw(lat[p_pos], lon[p_pos], y[p_pos],
                                               lat[q_pos], lon[q_pos], ok_cls)
            if not used_kriging and ok_cls is not None:
                n_fall += 1
            oof[q_pos] = np.maximum(pred, 0.0)
    if n_fall:
        print(f"[baseline_kriging] {n_fall}/{n_days} fold-days fell back to IDW")
    return oof


def krige_to_sites(train_lat, train_lon, train_vals, site_lat, site_lon,
                   max_train=150):
    """Single-day ordinary kriging of scattered values to arbitrary sites.

    The per-day engine behind baseline_kriging exposed standalone, for the
    pipeline's deployment-mode Tier-3-at-AQS assembly: per day, krige the
    full-data training residuals to the AQS site coordinates. pykrige
    ordinary kriging (exponential variogram, geographic coordinates) with
    IDW(k=8) fallback on any failure; train points with non-finite
    coordinates/values are dropped, the rest subsampled to max_train with a
    fixed rng. Output is NOT clipped at zero — residual fields are
    legitimately negative. Returns all-NaN when no usable train point exists.
    """
    t_lat = np.asarray(train_lat, dtype=np.float64)
    t_lon = np.asarray(train_lon, dtype=np.float64)
    t_val = np.asarray(train_vals, dtype=np.float64)
    s_lat = np.asarray(site_lat, dtype=np.float64)
    s_lon = np.asarray(site_lon, dtype=np.float64)
    ok = np.isfinite(t_val) & np.isfinite(t_lat) & np.isfinite(t_lon)
    if not ok.any():
        return np.full(len(s_lat), np.nan)
    t_lat, t_lon, t_val = t_lat[ok], t_lon[ok], t_val[ok]
    if len(t_val) > max_train:
        keep = np.random.default_rng(0).choice(len(t_val), size=max_train,
                                               replace=False)
        t_lat, t_lon, t_val = t_lat[keep], t_lon[keep], t_val[keep]
    pred, _ = _krige_or_idw(t_lat, t_lon, t_val, s_lat, s_lon,
                            _import_ordinary_kriging())
    return pred


def baseline_column(df, col, offset=0.0):
    """A raw prior column (e.g. cams_pm25 or geoscf_pm25) used directly as the
    prediction — the "is the ML model beating the CTM?" baseline. Needs no
    folds because the column was never fit to the target.

    offset optionally mean-debiases the prior before scoring: the caller
    computes offset = mean(prior - target) on TRAIN rows only and the column
    is scored as (col - offset) — the "one free constant" variant that
    separates a prior's spatial-pattern skill from its calibration bias.
    Default 0.0 keeps the raw prior.
    """
    if col not in df.columns:
        print(f"[baseline_column] '{col}' not in frame — returning all-NaN")
        return np.full(len(df), np.nan)
    return df[col].to_numpy(dtype=np.float64) - float(offset)


# ── EPA AQS external validation ─────────────────────────────────────────────

def external_aqs_validation(predict_fn, aqs_parquet, geoscf_parquet=None,
                            merra2_parquet=None, correction="barkjohn",
                            save_rows_to=None):
    """Score a fitted AQNet predictor against EPA AQS FRM/FEM daily PM2.5.

    Feature assembly is delegated to features.build_site_features — the one
    site-day feature builder shared with the training side. Each AQS FRM/FEM
    monitor is treated as a VIRTUAL location: every feature comes from
    PurpleAir sensors, gridded products, and tract data alone, and the
    neighbor aggregates pool CORRECTED-target sensor-days (the same units the
    model was trained on). AQS concentrations never appear in any model
    input — pm25_aqs rides along solely as the ground-truth column
    (methodology rule: AQS is external validation only, never training or
    feature input). AQS site-days outside the PurpleAir date coverage are
    dropped (no same-day context exists for them).

    Parameters
    ----------
    predict_fn : callable(pd.DataFrame) -> 1-D array
        Takes the feature frame restricted to features.feature_columns(...)
        (e.g. a closure over models_tabular.predict_full or the Tier-3 meta
        predictor) and returns predictions in µg/m³.
    aqs_parquet : str
        Output of data_external.fetch_aqs_daily_tx — [site_id, date,
        pm25_aqs, lat, lon]. Used ONLY as ground truth.
    geoscf_parquet, merra2_parquet : str or None
        Pass the SAME parquets the model was trained with so the AQS feature
        vector carries the same external columns.
    correction : str
        Target-correction method the model was trained with, forwarded to
        build_site_features so the neighbor pool aggregates matching units.

    Returns a dict: pooled regression metrics (+ cluster-bootstrap CIs over
    monitors), EPA-2024 AQI category metrics, per-year breakdown, site/row
    counts, and "feature_builder" recording the assembly path. Because AQS
    monitors are FRM/FEM reference instruments never seen in training, this
    is a genuinely external accuracy estimate for the corrected-PurpleAir
    target scale.
    """
    import features

    aqs = pd.read_parquet(aqs_parquet)
    aqs["date"] = pd.to_datetime(aqs["date"]).dt.normalize()
    aqs = aqs.dropna(subset=["pm25_aqs", "lat", "lon"]).reset_index(drop=True)

    # Corrected PurpleAir pool, built once: it defines the date coverage AND
    # feeds build_site_features so neighbor aggregates are in target units.
    pool = features.apply_target_correction(features.load_sensor_days(),
                                            method=correction)
    in_range = aqs["date"].isin(pd.unique(pool["date"]))
    n_drop = int((~in_range).sum())
    if n_drop:
        print(f"[aqs] dropping {n_drop:,} AQS site-days outside PurpleAir date coverage")
    aqs = aqs[in_range].reset_index(drop=True)
    print(f"[aqs] building features at {len(aqs):,} site-days "
          f"({aqs['site_id'].nunique()} monitors) via features.build_site_features")

    X = features.build_site_features(
        aqs[["site_id", "date", "lat", "lon", "pm25_aqs"]],
        correction=correction, geoscf_parquet=geoscf_parquet,
        merra2_parquet=merra2_parquet, pool=pool)
    cols = features.feature_columns(X)
    y_true = X["pm25_aqs"].to_numpy(dtype=np.float64)
    y_pred = np.asarray(predict_fn(X[cols]), dtype=np.float64).ravel()
    if len(y_pred) != len(X):
        raise ValueError(f"predict_fn returned {len(y_pred)} predictions "
                         f"for {len(X)} AQS site-days")

    out = metrics(y_true, y_pred)
    out["feature_builder"] = "features.build_site_features"
    out["n_sites"] = int(X["site_id"].nunique())
    out["n_pred_nan"] = int(np.sum(~np.isfinite(y_pred)))
    out["bootstrap_ci"] = bootstrap_ci(y_true, y_pred,
                                       cluster=X["site_id"].to_numpy())
    out["aqi"] = aqi_category_metrics(y_true, y_pred)

    by_year = {}
    years = pd.to_datetime(X["date"]).dt.year.to_numpy()
    for yr in sorted(np.unique(years)):
        m = years == yr
        by_year[int(yr)] = metrics(y_true[m], y_pred[m])
    out["by_year"] = by_year

    # Per-site-day predictions for downstream figures (F12 maps/scatter).
    if save_rows_to:
        rows = X[["site_id", "date", "lat", "lon", "pm25_aqs"]].copy()
        rows["pred_tier1"] = y_pred
        rows.to_parquet(save_rows_to, index=False)
        print(f"[aqs] per-site-day predictions -> {save_rows_to}")
    return out
