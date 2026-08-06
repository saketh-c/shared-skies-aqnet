"""Two-network FRM-scale training frame + THE single point-feature builder.

frame2 replaces v1's build_training_frame / build_site_features pair with one
construction that closes three measured v1 defects (DESIGN.md §0, §6):

  * Single-builder parity. v1 built training features one way and AQS-site
    features another (IDW'd PA met vs measured met, tract-centroid elevation
    vs sensor elevation) — a train/serve covariate shift that fed the −2.7 to
    −3.8 µg/m³ AQS bias. Here EVERY feature — training row or arbitrary
    (lat, lon, date) query — flows through build_point_features(); the frame
    builder itself calls it, so parity holds by construction and is unit
    tested (tests/test_frame2.py).
  * No location fingerprints, no silent fills. Raw lat/lon and
    dist_to_nearest_sensor are structurally excluded from the feature list
    (location enters only through the T1 GP). Empty neighbor pools yield NaN
    plus an explicit *_avail_* indicator — v1's grand-mean fallback taught
    the model a statewide mean instead of "no local information".
  * Deployment-honest neighbor blocks. Same-day FRM neighbor features do not
    exist (serving has no same-day FRM feed): the FRM block is lag-1/lag-7
    ONLY, and _neighbor_block() asserts against a same-day FRM call. Lags
    shift the QUERY date, so a query at t can only ever see pool values from
    t − lag (the embargo is tested, not assumed).

Vault airlock: every pool builder takes exclude_units and, whenever a folds
context is supplied, asserts that no vault unit (folds2.json vault_sites) and
no vault-period row (>= 2026-01-01) contributes to any pool. The assert is
the guard — passing exclude_units correctly is the caller's job; forgetting
is a crash, never a leak.

Met is grid-sourced (ERA5 / MERRA-2 / met-extra by-cell parquets), never the
PA on-board columns. HR statics join from pipeline/static_covariates.parquet
when present (graceful NaN + warning when absent). Demographic EJScreen
columns are dropped at every join and asserted out of the returned feature
list.

Fold-nested consumption: the frame's own columns are the deployment view
(pa_cal_full targets, full-pool neighbor blocks, full-fit T0). Training
consumption goes through neighbor_overrides(), which recomputes the neighbor
blocks, the fold-aware PA target (pa_cal_f{k}) and the fold-k T0 prior per
fold against TRAIN-only pools, honoring the v1 f{fold}__{col} npz contract.

Smoke test (prints frame shape + coverage):
    python research/aqnet2/frame2.py --quick
"""
import os
import json
import argparse
import importlib.util

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import config2

# ── v1 shared BallTree neighbor implementation (single source of truth) ─────
# Loaded by file path so this module needs no sys.path bootstrap (pipeline2
# owns that); pipeline/neighbor_features.py is a committed, stable v1 asset.

_NF_PATH = os.path.join(config2.PIPELINE_DIR, "neighbor_features.py")
_spec = importlib.util.spec_from_file_location("aq2_neighbor_features", _NF_PATH)
_nf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_nf)
compute_neighbor_features_df = _nf.compute_neighbor_features_df

# ── Constants ───────────────────────────────────────────────────────────────

EARTH_R_KM = 6371.0
SIGMA_FRM = float(config2.SIGMA_FRM)

# All data from this date onward belongs to the one-shot vault (DESIGN §2).
VAULT_DATE_START = getattr(config2, "VAULT_DATE_START", "2026-01-01")

PA_RADII_KM = (25, 50, 100)      # calibrated-PA neighbor block radii
FRM_RADIUS_KM = 50               # FRM block: 50 km only, lagged only
FRM_LAGS = (1, 7)
PA_LAGS = (1,)

# Texas coast reference points (Brownsville -> Sabine Pass), identical to v1
# features.py so dist_to_coast stays byte-compatible across versions.
# Domain note (EXPANSION.md): west7 coast reference points are NOT yet
# registered; until they are, dist_to_coast keeps this Gulf reference in
# every domain rather than inventing unregistered coordinates here.
TX_COAST_POINTS = [
    (25.97, -97.50),  # Brownsville
    (27.80, -97.40),  # Corpus Christi
    (28.93, -95.97),  # Freeport
    (29.30, -94.79),  # Galveston
    (29.70, -93.90),  # Sabine Pass
]

# Feature-name contract. A column is a model feature iff it matches a prefix
# or an exact name below; identity/target/weight columns never match. Raw
# coordinates and demographics are BANNED outright (asserted, not filtered).
FEATURE_PREFIXES = ("t0_", "nbr_", "era5_", "merra2_", "geoscf_", "cams_",
                    "maiac_", "st_")
FEATURE_EXACT = frozenset({
    "aod", "dust", "hms_smoke", "dist_to_coast",
    "doy_sin", "doy_cos", "dow_sin", "dow_cos",
    "shortwave", "et0", "cloud_cover",
})
BANNED_FEATURES = (set(config2.EXCLUDED_DEMOGRAPHIC)
                   | {"lat", "lon", "latitude", "longitude",
                      "dist_to_nearest_sensor"})

# Default committed by-cell products (v1 parquets remain live inputs) used
# when an external_paths dict does not name them explicitly. The statics
# default is tx-ONLY: the committed lattice is Texas-bbox, so a non-tx
# domain must get its statics from the data-stage registry (fetchers2
# writes the domain-stamped path) or degrade loudly to NaN st_* via the
# build_pools warning — a default here would be a silent wrong-domain fill.
_DEFAULT_EXTERNAL = {
    "cams": os.path.join(config2.PIPELINE_DIR, "airquality_by_cell.parquet"),
    "met_extra": os.path.join(config2.PIPELINE_DIR, "met_extra_by_cell.parquet"),
    "hms_grid": os.path.join(config2.PIPELINE_DIR, "hms_grid.parquet"),
    "statics": (os.path.join(config2.PIPELINE_DIR, "static_covariates.parquet")
                if config2.DOMAIN == "tx" else None),
    "pa_daily": os.path.join(config2.PIPELINE_DIR, "purpleair_full_dataset.parquet"),
}
_GRIDDED_KEYS = ("geoscf", "merra2", "cams", "era5", "met_extra", "maiac",
                 "hms_grid")


# ── Geometry / small helpers ────────────────────────────────────────────────

def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Inputs may be scalars or numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2)
    return 2.0 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _min_dist_to_points(lats, lons, points):
    """For each (lat, lon), the min haversine km to any reference point."""
    out = np.full(len(lats), np.inf)
    for plat, plon in points:
        out = np.minimum(out, _haversine_km(lats, lons, plat, plon))
    return out


def _norm_dates(dates):
    """Any date-like sequence -> normalized (midnight) DatetimeIndex."""
    return pd.DatetimeIndex(pd.to_datetime(dates)).normalize()


def _as_unit_set(units):
    """Normalize a vault/exclusion id list to a set covering both raw site
    ids and their 'aqs_'/'pa_'-prefixed unit_id forms."""
    out = set()
    for u in (units or ()):
        s = str(u)
        out.add(s)
        if not s.startswith(("aqs_", "pa_")):
            out.add(f"aqs_{s}")
            out.add(f"pa_{s}")
    return out


def _vault_units(folds):
    """The vault unit-id set from a folds2.json-shaped dict (or None)."""
    if not folds:
        return set()
    return _as_unit_set(folds.get("vault_sites", []))


def _assert_no_vault(pool, vault_units, name):
    """The vault airlock: a pool that contains any vault unit is a build
    error, never a warning — DESIGN §2 exclusion list is load-bearing."""
    if not vault_units or pool is None or not len(pool):
        return
    present = set(pool["unit_id"].astype(str)) & vault_units
    assert not present, (
        f"vault airlock breach: vault units {sorted(present)[:5]} present in "
        f"the {name} pool — exclude_units must carry the folds2 vault list")


# ── Loaders ─────────────────────────────────────────────────────────────────

def load_gridded(path):
    """A by-cell product parquet -> standardized [lat, lon, date, values...].

    Accepts either (lat, lon) or v1-style (cell_lat, cell_lon) coordinate
    columns; dates are normalized to midnight.
    """
    g = pd.read_parquet(path)
    if "cell_lat" in g.columns:
        g = g.rename(columns={"cell_lat": "lat", "cell_lon": "lon"})
    g["date"] = pd.to_datetime(g["date"]).dt.normalize()
    return g


def load_aqs(path, start=None, end=None):
    """AQS daily FRM/FEM parquet [site_id, date, pm25_aqs, lat, lon] with the
    date window applied and NaN observations dropped."""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    n0 = len(df)
    df = df.dropna(subset=["pm25_aqs"]).reset_index(drop=True)
    if len(df) < n0:
        print(f"[frame2] aqs: dropped {n0 - len(df):,} rows with NaN pm25_aqs")
    df["site_id"] = df["site_id"].astype(str)
    return df


def load_pa_calibrated(path, external_paths=None, start=None, end=None):
    """pa_calibrated.parquet with sensor coordinates guaranteed.

    The calibrate stage is expected to carry lat/lon through; if it did not,
    coordinates are joined from the committed PA daily parquet
    (external_paths['pa_daily'], one coordinate pair per sensor).
    """
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    df = df.reset_index(drop=True)

    if "lat" not in df.columns or "lon" not in df.columns:
        ext = dict(_DEFAULT_EXTERNAL)
        ext.update(external_paths or {})
        pa_path = ext.get("pa_daily")
        if not pa_path or not os.path.exists(pa_path):
            raise FileNotFoundError(
                "pa_calibrated.parquet has no lat/lon and no pa_daily parquet "
                "is available to join sensor coordinates from")
        pa = pd.read_parquet(pa_path, columns=["sensor_id", "latitude", "longitude"])
        coords = (pa.dropna(subset=["latitude", "longitude"])
                    .drop_duplicates("sensor_id"))
        coords = coords.rename(columns={"latitude": "lat", "longitude": "lon"})
        # calibrate writes sensor_id as str; the committed PA parquet carries
        # int64 — normalize both sides (audited pandas-3 join hazard).
        df["sensor_id"] = df["sensor_id"].astype(str)
        coords["sensor_id"] = coords["sensor_id"].astype(str)
        df = df.merge(coords, on="sensor_id", how="left")
        n_bad = int(df["lat"].isna().sum())
        if n_bad:
            print(f"[frame2] dropped {n_bad:,} calibrated rows with no sensor "
                  f"coordinates")
            df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    n0 = len(df)
    df = df.dropna(subset=["pa_cal_full"]).reset_index(drop=True)
    if len(df) < n0:
        print(f"[frame2] pa: dropped {n0 - len(df):,} rows with NaN pa_cal_full")
    return df


# ── Gridded / static joins (v1 join mechanics, generalized) ────────────────

def gridded_join(df, ext, value_cols=None):
    """Left-join a gridded product onto df by NEAREST (lat, lon) cell on the
    same day (cKDTree in degree space — v1 convention, adequate at Texas
    latitudes and robust to grids with missing cells). Rows whose date is
    absent from the product stay NaN. Existing df columns are never
    overwritten."""
    if value_cols is None:
        value_cols = [c for c in ext.columns if c not in ("lat", "lon", "date")]
    value_cols = [c for c in value_cols if c in ext.columns and c not in df.columns]
    if not value_cols or not len(ext):
        return df
    cells = ext[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    tree = cKDTree(cells[["lat", "lon"]].to_numpy(dtype=np.float64))
    query = np.column_stack([df["lat"].to_numpy(dtype=np.float64),
                             df["lon"].to_numpy(dtype=np.float64)])
    _, idx = tree.query(query, k=1)

    out = df.copy()
    out["_cell_lat"] = cells["lat"].to_numpy()[idx]
    out["_cell_lon"] = cells["lon"].to_numpy()[idx]
    sub = ext[["lat", "lon", "date"] + list(value_cols)].rename(
        columns={"lat": "_cell_lat", "lon": "_cell_lon"})
    sub = sub.drop_duplicates(["_cell_lat", "_cell_lon", "date"])
    out = out.merge(sub, on=["_cell_lat", "_cell_lon", "date"], how="left")
    return out.drop(columns=["_cell_lat", "_cell_lon"])


def hms_join(df, hms):
    """HMS polygon-raster join: nearest cell, same day. Inside the raster's
    date coverage a missing row means NO smoke polygon (0); outside coverage
    the tier is honestly NaN — data absence is not 'no smoke'."""
    out = gridded_join(df, hms, value_cols=["hms_smoke"])
    if "hms_smoke" not in out.columns:
        out["hms_smoke"] = np.nan
        return out
    dmin, dmax = hms["date"].min(), hms["date"].max()
    in_cov = (out["date"] >= dmin) & (out["date"] <= dmax)
    fill = in_cov & out["hms_smoke"].isna()
    if fill.any():
        out.loc[fill, "hms_smoke"] = 0.0
    out["hms_smoke"] = out["hms_smoke"].astype(np.float64)
    return out


def statics_join(df, statics):
    """HR-static covariates by nearest point (0.01-degree raster table or
    site table). Columns are prefixed 'st_' (unless already), demographic
    columns are dropped before the join, and a 'year' column — the NEI
    year-key — selects, per query row, the latest static year <= the query
    year (else the earliest available)."""
    st = statics.copy()
    if "latitude" in st.columns and "lat" not in st.columns:
        st = st.rename(columns={"latitude": "lat", "longitude": "lon"})
    demo = [c for c in st.columns if c in set(config2.EXCLUDED_DEMOGRAPHIC)]
    if demo:
        print(f"[frame2] statics: dropping demographic columns {demo} "
              f"(never model inputs)")
        st = st.drop(columns=demo)

    year_col = "year" if "year" in st.columns else None
    value_cols = [c for c in st.columns if c not in ("lat", "lon", "year")]
    rename = {c: (c if c.startswith("st_") else f"st_{c}") for c in value_cols}
    st = st.rename(columns=rename)
    value_cols = list(rename.values())

    out = df.copy()
    for c in value_cols:
        out[c] = np.nan
    qxy = np.column_stack([out["lat"].to_numpy(dtype=np.float64),
                           out["lon"].to_numpy(dtype=np.float64)])

    if year_col is None:
        tree = cKDTree(st[["lat", "lon"]].to_numpy(dtype=np.float64))
        _, idx = tree.query(qxy, k=1)
        for c in value_cols:
            out[c] = st[c].to_numpy()[idx]
        return out

    years = np.sort(st["year"].unique())
    qy = out["date"].dt.year.to_numpy()
    # Latest static year <= query year; queries before the first year use it.
    pick = np.searchsorted(years, qy, side="right") - 1
    pick = years[np.clip(pick, 0, len(years) - 1)]
    for yr in np.unique(pick):
        sub = st[st["year"] == yr].reset_index(drop=True)
        rows = np.flatnonzero(pick == yr)
        tree = cKDTree(sub[["lat", "lon"]].to_numpy(dtype=np.float64))
        _, idx = tree.query(qxy[rows], k=1)
        for c in value_cols:
            out.loc[out.index[rows], c] = sub[c].to_numpy()[idx]
    return out


# ── Pool builders (every one carries the exclude_units airlock) ─────────────

def _apply_pool_exclusions(pool, exclude_units, fold_ctx, name):
    """Shared airlock: drop excluded units, drop vault-period rows whenever a
    folds context is supplied, then ASSERT no vault unit survived."""
    ex = _as_unit_set(exclude_units)
    if ex:
        pool = pool[~pool["unit_id"].astype(str).isin(ex)]
    fold_ctx = fold_ctx or {}
    vault = _as_unit_set(fold_ctx.get("vault_units"))
    if vault and not fold_ctx.get("allow_vault_period", False):
        pool = pool[pool["date"] < pd.Timestamp(VAULT_DATE_START)]
    pool = pool.reset_index(drop=True)
    _assert_no_vault(pool, vault, name)
    return pool


def build_pa_pool(pa_daily, exclude_units=(), fold_ctx=None,
                  value_col="pa_cal_full"):
    """Calibrated-PA sensor-day pool [unit_id, lat, lon, date, value].

    value_col picks the calibration column (pa_cal_f{k} for fold-nested
    consumption, pa_cal_full for deployment) so neighbor aggregates are
    always on the FRM scale of the consuming fold.
    """
    if value_col not in pa_daily.columns:
        raise KeyError(f"pa pool value column {value_col!r} missing from the "
                       f"calibrated parquet")
    pool = pd.DataFrame({
        "unit_id": "pa_" + pa_daily["sensor_id"].astype(str),
        "lat": pa_daily["lat"].astype(float),
        "lon": pa_daily["lon"].astype(float),
        "date": pd.to_datetime(pa_daily["date"]).dt.normalize(),
        "value": pa_daily[value_col].astype(float),
    })
    return _apply_pool_exclusions(pool, exclude_units, fold_ctx, "pa")


def build_frm_pool(aqs_daily, exclude_units=(), fold_ctx=None,
                   value_col="pm25_aqs"):
    """FRM site-day pool [unit_id, lat, lon, date, value] — consumed ONLY by
    the lagged nbr_frm block (same-day FRM features do not exist)."""
    pool = pd.DataFrame({
        "unit_id": "aqs_" + aqs_daily["site_id"].astype(str),
        "lat": aqs_daily["lat"].astype(float),
        "lon": aqs_daily["lon"].astype(float),
        "date": pd.to_datetime(aqs_daily["date"]).dt.normalize(),
        "value": aqs_daily[value_col].astype(float),
    })
    return _apply_pool_exclusions(pool, exclude_units, fold_ctx, "frm")


def build_pools(calibrated_parquet=None, external_paths=None, exclude_units=(),
                fold_ctx=None, pa_daily=None, aqs_daily=None, t0_models=None,
                pa_value_col=None, start=None, end=None):
    """Assemble the pools dict build_point_features consumes.

    pools = {"pa": DataFrame, "frm": DataFrame,
             "gridded": {name: DataFrame[lat, lon, date, values...]},
             "statics": DataFrame | None,
             "t0": {outer_k | "full": downscaler model} | None}

    pa_daily / aqs_daily may be passed pre-loaded (the frame builder and the
    tests do); otherwise they are read from calibrated_parquet and
    external_paths["aqs"]. exclude_units + fold_ctx thread the vault airlock
    into every pool. pa_value_col defaults to pa_cal_f{outer_k} when fold_ctx
    names an outer fold, else pa_cal_full.
    """
    ext = dict(_DEFAULT_EXTERNAL)
    ext.update(external_paths or {})
    fold_ctx = fold_ctx or {}

    if pa_daily is None:
        if calibrated_parquet is None:
            raise ValueError("build_pools needs pa_daily or calibrated_parquet")
        pa_daily = load_pa_calibrated(calibrated_parquet, ext, start, end)
    if aqs_daily is None:
        aqs_path = ext.get("aqs")
        if not aqs_path or not os.path.exists(aqs_path):
            raise FileNotFoundError("external_paths['aqs'] is required (FRM "
                                    "truth rows + lagged FRM pool)")
        aqs_daily = load_aqs(aqs_path, start, end)

    if pa_value_col is None:
        outer_k = fold_ctx.get("outer_k")
        pa_value_col = (f"pa_cal_f{outer_k}"
                        if outer_k is not None and f"pa_cal_f{outer_k}" in pa_daily.columns
                        else "pa_cal_full")

    gridded = {}
    for key in _GRIDDED_KEYS:
        path = ext.get(key)
        if path is None:
            continue
        if not os.path.exists(path):
            print(f"[frame2] gridded product missing, skipped: {key} ({path})")
            continue
        gridded[key] = load_gridded(path)

    statics = None
    st_path = ext.get("statics")
    if st_path and os.path.exists(st_path):
        statics = pd.read_parquet(st_path)
    else:
        print("[frame2] WARNING: static_covariates.parquet absent — HR "
              "statics unavailable, st_* features will be missing")

    return {
        "pa": build_pa_pool(pa_daily, exclude_units, fold_ctx, pa_value_col),
        "frm": build_frm_pool(aqs_daily, exclude_units, fold_ctx),
        "gridded": gridded,
        "statics": statics,
        "t0": t0_models,
    }


# ── Neighbor blocks ─────────────────────────────────────────────────────────

def _neighbor_block(q, pool, prefix, lag=0, keep_radii=(25, 50, 100),
                    with_std=False):
    """One neighbor feature block via the shared v1 BallTree implementation,
    with v1's silent fallback fills REMOVED.

    q: DataFrame[lat, lon, date, unit_id]; pool: [unit_id, lat, lon, date,
    value]. lag shifts the QUERY date back, so a query at date t aggregates
    pool values from exactly t − lag (the lag embargo — tested). Leave-self-
    out by unit_id applies at every lag: a unit's own history never enters
    its features, keeping training rows comparable to bare-site queries.

    Zero-neighbor rows: mean NaN, count 0, avail 0 — the explicit
    availability indicator replaces v1's pool-grand-mean fill. The 50 km std
    is NaN with no neighbors and 0.0 for a singleton (v1 value).
    """
    assert not (prefix == "nbr_frm" and lag == 0), (
        "same-day FRM features are excluded by construction (DESIGN §6): "
        "serving has no same-day FRM feed")
    lagsuf = f"_lag{lag}" if lag else ""
    n = len(q)
    out = {}

    have_pool = pool is not None and len(pool) > 0
    if have_pool:
        vals = pd.to_numeric(pool["value"], errors="coerce").to_numpy(np.float64)
        bad = ~np.isfinite(vals)
        if bad.any():
            print(f"[frame2] {prefix}{lagsuf} pool: dropped {int(bad.sum()):,} "
                  f"rows with non-finite value")
            pool = pool[~bad].reset_index(drop=True)
        have_pool = len(pool) > 0

    if not have_pool:
        for r in keep_radii:
            out[f"{prefix}_{r}km{lagsuf}"] = np.full(n, np.nan)
            out[f"{prefix}_count_{r}km{lagsuf}"] = np.zeros(n)
            out[f"{prefix}_avail_{r}km{lagsuf}"] = np.zeros(n)
        if with_std and 50 in keep_radii:
            out[f"{prefix}_std_50km{lagsuf}"] = np.full(n, np.nan)
        return out

    qq = pd.DataFrame({
        "latitude": q["lat"].to_numpy(dtype=np.float64),
        "longitude": q["lon"].to_numpy(dtype=np.float64),
        "date": _norm_dates(q["date"]) - pd.Timedelta(days=lag),
        "sensor_id": q["unit_id"].astype(str).to_numpy(),
        "value": np.nan,
    })
    pp = pd.DataFrame({
        "latitude": pool["lat"].to_numpy(dtype=np.float64),
        "longitude": pool["lon"].to_numpy(dtype=np.float64),
        "date": _norm_dates(pool["date"]),
        "sensor_id": pool["unit_id"].astype(str).to_numpy(),
        "value": pool["value"].to_numpy(dtype=np.float64),
    })
    res = compute_neighbor_features_df(qq, pp, target_col="value")

    for r in keep_radii:
        cnt = res[f"nbr_count_{r}km"].astype(np.float64)
        mean = np.where(cnt > 0, res[f"nbr_pm25_{r}km"], np.nan)
        out[f"{prefix}_{r}km{lagsuf}"] = mean
        out[f"{prefix}_count_{r}km{lagsuf}"] = cnt
        out[f"{prefix}_avail_{r}km{lagsuf}"] = (cnt > 0).astype(np.float64)
    if with_std and 50 in keep_radii:
        cnt50 = res["nbr_count_50km"].astype(np.float64)
        out[f"{prefix}_std_50km{lagsuf}"] = np.where(
            cnt50 > 0, res["nbr_std_50km"], np.nan)
    return out


# ── THE single point-feature builder (R5 serving path) ──────────────────────

def build_point_features(lats, lons, dates, pools, fold_ctx=None):
    """Assemble the full v2 feature vector at arbitrary (lat, lon, date).

    Training rows and serving queries share this path bit-for-bit — the
    parity contract (DESIGN §6). Returns a DataFrame with identity columns
    (lat, lon, date, unit_id) plus every feature; run feature_columns() on
    the result for the model-input list.

    pools: see build_pools(). fold_ctx (all keys optional):
      unit_ids     per-query unit ids for leave-self-out (training rows);
                   arbitrary queries default to synthetic never-colliding ids
      outer_k      outer fold whose T0 downscaler to evaluate (None -> full)
      vault_units  folds2 vault list — pools are RE-asserted against it here
      allow_vault_period  validate-stage escape hatch for scoring 2026 rows

    Portable features: per-stream CTM/reanalysis + ERA5 grid met + HMS tier +
    HR statics + dist_to_coast + doy/dow harmonics + T0 prior (+ per-stream
    debiased CTMs) when downscaler models are supplied. Interpolating
    features: calibrated-PA neighbor block (same-day + lag-1) and the FRM
    block lag-1/lag-7 ONLY, each with explicit availability indicators.
    """
    fold_ctx = fold_ctx or {}
    q = pd.DataFrame({
        "lat": np.asarray(lats, dtype=np.float64),
        "lon": np.asarray(lons, dtype=np.float64),
        "date": _norm_dates(dates),
    })
    unit_ids = fold_ctx.get("unit_ids")
    if unit_ids is None:
        unit_ids = np.array([f"q{i}" for i in range(len(q))])
    q["unit_id"] = np.asarray(unit_ids).astype(str)

    # Re-assert the airlock on the pools we were handed: the builder is the
    # last line of defense, whoever assembled them.
    vault = _as_unit_set(fold_ctx.get("vault_units"))
    for name in ("pa", "frm"):
        pool = pools.get(name)
        if pool is not None and len(pool):
            _assert_no_vault(pool, vault, name)

    # ── Gridded products (nearest cell, same day; NaN-honest) ──
    for name in sorted(pools.get("gridded", {})):
        gdf = pools["gridded"][name]
        if name == "hms_grid":
            q = hms_join(q, gdf)
        else:
            q = gridded_join(q, gdf)
    if "hms_smoke" not in q.columns:
        print("[frame2] WARNING: no hms_grid product — hms_smoke left NaN")
        q["hms_smoke"] = np.nan

    # ── HR statics ──
    statics = pools.get("statics")
    if statics is not None and len(statics):
        q = statics_join(q, statics)
    else:
        print("[frame2] WARNING: HR statics unavailable at query time; "
              "st_* features absent")

    # ── Portable geometry + calendar ──
    q["dist_to_coast"] = _min_dist_to_points(
        q["lat"].to_numpy(), q["lon"].to_numpy(), TX_COAST_POINTS)
    doy = q["date"].dt.dayofyear
    dow = q["date"].dt.dayofweek
    q["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    q["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
    q["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    q["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # ── Neighbor blocks (coverage-gated; FRM lagged only) ──
    pa_pool = pools.get("pa")
    frm_pool = pools.get("frm")
    blocks = {}
    blocks.update(_neighbor_block(q, pa_pool, "nbr_pacal", lag=0,
                                  keep_radii=PA_RADII_KM, with_std=True))
    for lag in PA_LAGS:
        blocks.update(_neighbor_block(q, pa_pool, "nbr_pacal", lag=lag,
                                      keep_radii=PA_RADII_KM, with_std=True))
    for lag in FRM_LAGS:
        blocks.update(_neighbor_block(q, frm_pool, "nbr_frm", lag=lag,
                                      keep_radii=(FRM_RADIUS_KM,),
                                      with_std=False))
    for col, arr in blocks.items():
        q[col] = arr

    # ── T0 prior (per-stream debiased CTMs + precision-weighted combination)
    t0_models = pools.get("t0") or {}
    if t0_models:
        import priors  # lazy: priors imports frame2 at module level
        outer_k = fold_ctx.get("outer_k")
        model = t0_models.get(outer_k) if outer_k is not None else None
        if model is None:
            model = t0_models.get("full")
        if outer_k is not None and t0_models.get(outer_k) is None:
            # Never silently substitute the full-fit model inside a fold —
            # that would leak the fold's sites through the prior.
            print(f"[frame2] WARNING: no T0 downscaler for outer fold "
                  f"{outer_k}; t0_* features left NaN for this context")
            model = None
        if model is not None:
            ctm = {s: (q[s].to_numpy(dtype=np.float64) if s in q.columns
                       else np.full(len(q), np.nan))
                   for s in priors.STREAMS}
            t0, _pattern, per_stream = priors.evaluate_prior(
                model, q["lat"].to_numpy(), q["lon"].to_numpy(),
                q["date"], ctm)
            q["t0_prior"] = t0
            for s, pred in per_stream.items():
                q[f"t0_{s}"] = pred
        else:
            q["t0_prior"] = np.nan

    _assert_feature_hygiene(q.columns)
    return q


def _is_feature(col):
    if col in BANNED_FEATURES:
        return False
    return col in FEATURE_EXACT or any(col.startswith(p) for p in FEATURE_PREFIXES)


def _assert_feature_hygiene(columns):
    """No demographic and no raw-coordinate column may ever qualify as a
    feature; a joined product that smuggles one in is a build error."""
    demo = [c for c in columns
            if c in set(config2.EXCLUDED_DEMOGRAPHIC) and _is_feature(c)]
    assert not demo, f"demographic columns must never enter features: {demo}"
    coords = [c for c in ("lat", "lon", "latitude", "longitude")
              if _is_feature(c)]
    assert not coords, f"raw coordinates must never be features: {coords}"


def feature_columns(df):
    """Model-input columns of a frame/feature DataFrame, in column order.

    Presence-dependent (products that were not fetched contribute nothing),
    with the two hard methodological asserts: no demographic column and no
    raw lat/lon may ever be returned. dist_to_nearest_sensor (a v1 network
    fingerprint) is banned alongside them.
    """
    cols = [c for c in df.columns if _is_feature(c)]
    banned = [c for c in cols if c in BANNED_FEATURES]
    assert not banned, f"banned columns in the feature list: {banned}"
    demo = [c for c in cols if c in set(config2.EXCLUDED_DEMOGRAPHIC)]
    assert not demo, f"demographic columns must never enter the feature list: {demo}"
    return cols


# ── Training frame (two networks, one target scale) ─────────────────────────

def build_frame_truth(calibrated_parquet, external_paths, quick=False,
                      folds=None, lam=1.0, t0_models=None):
    """FRM-scale training frame: AQS site-days UNION calibrated-PA sensor-days.

      AQS rows:  y = pm25_aqs (FRM),        w = 1 / SIGMA_FRM^2
      PA rows:   y = pa_cal_full (FRM scale), w = lam * SIGMA_FRM^2
                                                / (SIGMA_FRM^2 + cal_var)

    lam is the PA down-weighting multiplier, tuned later on inner selection
    folds (frozen before any deep tier trains); the default 1.0 is the
    pre-tuning frame. All feature columns come from build_point_features()
    on pools built from the same inputs — the parity contract.

    folds (a folds2.json-shaped dict) supplies the vault context: when given,
    vault sites and vault-period rows are excluded from every pool (rows
    themselves stay in the frame so validate can score them). When folds is
    None — the bootstrap pass that folds2.build_folds itself consumes — pools
    are full and a loud notice is printed; every training consumer must use
    the fold-nested neighbor_overrides() columns, never the bootstrap ones.

    The frame keeps pa_cal_f{k} (+ pa_cal_f{k}_{j}) columns as passthrough so
    fold-aware targets can be swapped in via the overrides contract. Fold-
    aware t0 columns likewise arrive via neighbor_overrides(); t0_models
    (dict, e.g. {"full": model}) controls the frame's own deployment-view
    t0_* columns — None scans the artifacts dir, {} disables.
    """
    ext = dict(_DEFAULT_EXTERNAL)
    ext.update(external_paths or {})
    start, end = config2.DATE_START, config2.DATE_END
    if quick:
        # Fixed pre-vault summer window (v1 pipeline_colab QUICK_* convention,
        # shared with calibrate/priors). A trailing-92-days window would land
        # entirely inside the vault period (>= VAULT_DATE_START), where the
        # pool airlock drops every neighbor/lag source and a quick smoke run
        # degenerates to featureless rows.
        start, end = "2024-07-01", "2024-09-30"
        print(f"[frame2] --quick: window {start} .. {end}")

    pa = load_pa_calibrated(calibrated_parquet, ext, start, end)
    aqs_path = ext.get("aqs")
    if not aqs_path or not os.path.exists(aqs_path):
        raise FileNotFoundError("external_paths['aqs'] is required to build "
                                "the truth frame")
    aqs = load_aqs(aqs_path, start, end)

    # ── Base rows ──
    aqs_rows = pd.DataFrame({
        "unit_id": "aqs_" + aqs["site_id"].astype(str),
        "unit_type": "aqs",
        "network": "FRM",
        "date": aqs["date"],
        "lat": aqs["lat"].astype(float),
        "lon": aqs["lon"].astype(float),
        "y": aqs["pm25_aqs"].astype(float),
        "w": 1.0 / SIGMA_FRM ** 2,
        "cal_var": 0.0,
    })
    s2 = SIGMA_FRM ** 2
    cal_var = pa["cal_var"].astype(float)
    pa_rows = pd.DataFrame({
        "unit_id": "pa_" + pa["sensor_id"].astype(str),
        "unit_type": "pa",
        "network": "PA",
        "date": pa["date"],
        "lat": pa["lat"].astype(float),
        "lon": pa["lon"].astype(float),
        "y": pa["pa_cal_full"].astype(float),
        "w": lam * s2 / (s2 + cal_var),
        "cal_var": cal_var,
    })
    # Passthrough columns for fold-aware consumption + provenance.
    for c in pa.columns:
        if c.startswith("pa_cal_f") or c in ("pa_raw", "channel_reconstructed",
                                             "dist_to_nearest_frm"):
            pa_rows[c] = pa[c].to_numpy()

    base = pd.concat([aqs_rows, pa_rows], ignore_index=True)
    base = base.sort_values(["unit_id", "date"], kind="mergesort").reset_index(drop=True)

    n_bad_w = int(base["w"].isna().sum())
    if n_bad_w:
        print(f"[frame2] WARNING: {n_bad_w:,} rows have NaN weight "
              f"(missing cal_var); trainers will drop them")

    # ── Vault context ──
    vault = _vault_units(folds)
    if folds is None:
        # A no-folds frame carries vault-period rows in its pools and would
        # still pass feature hygiene — a review-flagged foot-gun. The
        # pipeline always supplies folds; a bootstrap frame must be opted
        # into explicitly.
        if os.environ.get("AQNET2_ALLOW_NOFOLDS") != "1":
            raise SystemExit(
                "[frame2] build_frame_truth called without folds — this "
                "builds a vault-contaminated bootstrap frame. Build "
                "folds2.json first (the calibrate stage does), or set "
                "AQNET2_ALLOW_NOFOLDS=1 to accept a bootstrap frame "
                "explicitly.")
        print("[frame2] NOTE: no folds supplied — bootstrap frame with FULL "
              "pools (AQNET2_ALLOW_NOFOLDS=1); training consumption must go "
              "through neighbor_overrides() once folds2.json exists")
    fold_ctx = {"unit_ids": base["unit_id"].to_numpy()}
    if vault:
        fold_ctx["vault_units"] = sorted(vault)

    if t0_models is None:
        t0_models = _scan_t0_models()

    pools = build_pools(external_paths=ext, exclude_units=sorted(vault),
                        fold_ctx=fold_ctx, pa_daily=pa, aqs_daily=aqs,
                        t0_models=t0_models, start=start, end=end)

    feats = build_point_features(base["lat"].to_numpy(), base["lon"].to_numpy(),
                                 base["date"], pools, fold_ctx)
    fcols = feature_columns(feats)
    frame = pd.concat([base, feats[fcols].reset_index(drop=True)], axis=1)
    print(f"[frame2] frame: {len(frame):,} rows "
          f"({int((frame['unit_type'] == 'aqs').sum()):,} AQS / "
          f"{int((frame['unit_type'] == 'pa').sum()):,} PA), "
          f"{len(fcols)} features")
    return frame


def _scan_t0_models():
    """Load any T0 downscaler artifacts already produced (deployment view).
    Returns {} when the priors stage has not run yet — t0_* columns are then
    simply absent from the bootstrap frame (presence-dependent features)."""
    try:
        import priors
        models = priors.load_fold_models()
    except Exception as e:  # missing artifacts dir, no npz yet
        print(f"[frame2] NOTE: no T0 downscaler artifacts loaded ({e}); "
              f"t0_* features deferred")
        return {}
    if not models:
        print("[frame2] NOTE: no T0 downscaler artifacts yet; t0_* features "
              "deferred until the priors stage runs")
    return models


# ── Fold-nested overrides (v1 f{fold}__{col} npz contract) ─────────────────

def neighbor_overrides(frame, folds, fold_key):
    """Recompute every fold-sensitive column against TRAIN-only pools.

    fold_key selects the fold system inside the folds2.json-shaped dict:
      "outer_fold" / "outer"            outer spatial folds (t0 + pa_cal f{k})
      "spatial_block_fold" / "spatial_block"   descriptive block folds
      "loso:{k}"                        the 10 LOSO folds nested in outer k

    For each fold f: pool = that fold's TRAIN rows minus vault units minus
    vault-period rows (asserted), PA values taken from the fold-aware
    calibration column, and — for outer/loso systems — t0_* re-evaluated with
    the fold's own downscaler (never the full fit: a missing fold model means
    the t0 override is skipped loudly, not substituted). Also emits a fold-
    aware "y" (PA rows -> pa_cal_f{k}).

    Returns {fold: {col: full-length np.ndarray}} aligned to frame row order,
    exactly what the v1 npz contract persists as f{fold}__{col}
    (save_overrides / load_overrides).
    """
    try:
        import folds2
        folds_from_assign = folds2.folds_from_assign
    except ImportError as e:
        raise RuntimeError(f"neighbor_overrides requires folds2 ({e})") from e

    assign, cal_k_of_fold, t0_enabled = _resolve_fold_key(frame, folds, fold_key)
    fold_pairs = folds_from_assign(assign)
    vault = _vault_units(folds)
    is_pa = (frame["unit_type"] == "pa").to_numpy()

    t0_models = {}
    if t0_enabled:
        try:
            import priors
            t0_models = priors.load_fold_models()
        except Exception as e:
            print(f"[frame2] overrides: T0 models unavailable ({e}); "
                  f"t0 overrides skipped")

    base = frame.reset_index(drop=True)
    q = base[["lat", "lon", "date", "unit_id"]].copy()
    out = {}
    print(f"[frame2] per-fold override recompute: {len(fold_pairs)} folds x "
          f"{len(base):,} rows ({fold_key})")

    # In the loso:{k} system, assign == -1 marks outer-fold-k AQS holdout
    # rows (plus vault material) — folds_from_assign treats -1 as
    # always-train, which is right for the OUTER system (PA rows are
    # legitimate always-train pool members) but a LEAK here: fold-k FRM
    # history would enter every within-k LOSO pool and contaminate the
    # residuals the deep tiers fine-tune on, inside the very chain that
    # later scores fold-k sites. Strip -1 rows from loso pools.
    loso_system = str(fold_key).startswith("loso")
    assign_arr = np.asarray(assign)

    for f, (train_idx, _test_idx) in enumerate(fold_pairs):
        ck = cal_k_of_fold(f)
        tr = np.asarray(train_idx)
        if loso_system:
            n_holdout = int((assign_arr[tr] == -1).sum())
            if n_holdout and f == 0:
                print(f"[frame2] overrides ({fold_key}): excluding "
                      f"{n_holdout:,} always-train rows (outer-fold holdout "
                      f"+ vault) from every pool")
            tr = tr[assign_arr[tr] != -1]
        pool_rows = base.iloc[tr]
        fold_ctx = {"vault_units": sorted(vault)} if vault else {}

        cal_col = f"pa_cal_f{ck}" if ck is not None else "pa_cal_full"
        if cal_col not in base.columns:
            if ck is not None:
                print(f"[frame2] overrides fold {f}: {cal_col} missing; "
                      f"using pa_cal_full for the PA pool")
            cal_col = "pa_cal_full" if "pa_cal_full" in base.columns else "y"

        pa_src = pool_rows[pool_rows["unit_type"] == "pa"]
        pa_pool = pd.DataFrame({
            "unit_id": pa_src["unit_id"].astype(str),
            "lat": pa_src["lat"].astype(float),
            "lon": pa_src["lon"].astype(float),
            "date": pa_src["date"],
            "value": pa_src[cal_col].astype(float)
            if cal_col in pa_src.columns else pa_src["y"].astype(float),
        })
        pa_pool = _apply_pool_exclusions(pa_pool, vault, fold_ctx, "pa")

        frm_src = pool_rows[pool_rows["unit_type"] == "aqs"]
        frm_pool = pd.DataFrame({
            "unit_id": frm_src["unit_id"].astype(str),
            "lat": frm_src["lat"].astype(float),
            "lon": frm_src["lon"].astype(float),
            "date": frm_src["date"],
            "value": frm_src["y"].astype(float),
        })
        frm_pool = _apply_pool_exclusions(frm_pool, vault, fold_ctx, "frm")

        cols = {}
        cols.update(_neighbor_block(q, pa_pool, "nbr_pacal", lag=0,
                                    keep_radii=PA_RADII_KM, with_std=True))
        for lag in PA_LAGS:
            cols.update(_neighbor_block(q, pa_pool, "nbr_pacal", lag=lag,
                                        keep_radii=PA_RADII_KM, with_std=True))
        for lag in FRM_LAGS:
            cols.update(_neighbor_block(q, frm_pool, "nbr_frm", lag=lag,
                                        keep_radii=(FRM_RADIUS_KM,),
                                        with_std=False))

        # Fold-aware PA target.
        if ck is not None and f"pa_cal_f{ck}" in base.columns:
            y = base["y"].to_numpy(dtype=np.float64).copy()
            y[is_pa] = base[f"pa_cal_f{ck}"].to_numpy(dtype=np.float64)[is_pa]
            cols["y"] = y

        # Fold-nested T0 (outer/loso systems only; exact fold model or skip).
        if t0_enabled and t0_models:
            model = t0_models.get(ck)
            if model is None:
                print(f"[frame2] overrides fold {f}: no T0 downscaler f{ck}; "
                      f"t0 override skipped")
            else:
                import priors
                ctm = {s: (base[s].to_numpy(dtype=np.float64)
                           if s in base.columns else np.full(len(base), np.nan))
                       for s in priors.STREAMS}
                t0, _pat, per_stream = priors.evaluate_prior(
                    model, base["lat"].to_numpy(), base["lon"].to_numpy(),
                    base["date"], ctm)
                cols["t0_prior"] = t0
                for s, pred in per_stream.items():
                    cols[f"t0_{s}"] = pred

        out[f] = cols
    return out


def _resolve_fold_key(frame, folds, fold_key):
    """(assign array, fold -> calibration/t0 outer-k mapping, t0_enabled)."""
    n = len(frame)
    key = str(fold_key)
    if key in ("outer", "outer_fold"):
        assign = np.asarray(folds["outer_fold"], dtype=int)
        cal_k = lambda f: f  # noqa: E731 — fold f IS outer fold f
        t0_enabled = True
    elif key in ("spatial_block", "spatial_block_fold"):
        assign = np.asarray(folds["spatial_block_fold"], dtype=int)
        cal_k = lambda f: None  # noqa: E731 — descriptive folds, full cal
        t0_enabled = False
    elif key.startswith("loso:"):
        k = int(key.split(":", 1)[1])
        loso = folds["loso_fold"]
        arr = loso.get(str(k), loso.get(k)) if isinstance(loso, dict) else None
        if arr is None:
            raise KeyError(f"folds['loso_fold'] has no entry for outer fold {k}")
        assign = np.asarray(arr, dtype=int)
        cal_k = lambda f, _k=k: _k  # noqa: E731 — nested in outer k
        t0_enabled = True
    else:
        raise ValueError(f"unknown fold_key {fold_key!r}; expected 'outer_fold'"
                         f", 'spatial_block_fold', or 'loso:{{k}}'")
    if len(assign) != n:
        raise ValueError(f"fold assignment length {len(assign)} != frame rows {n}")
    return assign, cal_k, t0_enabled


def save_overrides(overrides, path):
    """Persist {fold: {col: array}} as the f{fold}__{col} npz (atomic)."""
    payload = {}
    for f, cols in overrides.items():
        for c, arr in cols.items():
            payload[f"f{int(f)}__{c}"] = np.asarray(arr, dtype=np.float64)
    tmp = str(path) + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)
    print(f"[frame2] overrides saved -> {path} ({len(payload)} arrays)")


def load_overrides(path, n_rows):
    """Rebuild {fold: {col: array}} from the npz; wrong-length arrays are a
    hard error (row misalignment silently corrupts every downstream fold)."""
    out = {}
    with np.load(path) as z:
        for key in z.files:
            fold_s, col = key.split("__", 1)
            arr = z[key]
            if len(arr) != n_rows:
                raise ValueError(f"override {key} has length {len(arr)}, "
                                 f"expected {n_rows}")
            out.setdefault(int(fold_s[1:]), {})[col] = arr
    return out


# ── CLI smoke test / stage entry ────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Build (and optionally save) the AQNet v2 truth frame.")
    ap.add_argument("--calibrated", default=config2.artifact("pa_calibrated.parquet"),
                    help="pa_calibrated.parquet from the calibrate stage")
    ap.add_argument("--external-paths", default=config2.artifact("external_paths.json"),
                    help="external_paths.json from the data stage")
    ap.add_argument("--folds", default=config2.artifact("folds2.json"),
                    help="folds2.json (vault context); bootstrap frame if absent")
    ap.add_argument("--out", default=config2.artifact("frame_truth.parquet"))
    ap.add_argument("--overrides-out", default=None,
                    help="also write outer-fold overrides npz to this path")
    ap.add_argument("--lam", type=float, default=1.0,
                    help="PA row weight multiplier (tuned later on selection folds)")
    ap.add_argument("--quick", action="store_true",
                    help="restrict to the last 3 months for a fast smoke test")
    args = ap.parse_args()

    ext = {}
    if os.path.exists(args.external_paths):
        with open(args.external_paths) as fh:
            ext = {k: v for k, v in json.load(fh).items() if v}
    folds = None
    if os.path.exists(args.folds):
        with open(args.folds) as fh:
            folds = json.load(fh)
        print(f"[frame2] folds context loaded ({len(folds.get('vault_sites', []))} "
              f"vault sites); content-hash verification is folds2.load_folds's "
              f"job once the frame exists")

    frame = build_frame_truth(args.calibrated, ext, quick=args.quick,
                              folds=folds, lam=args.lam)
    feats = feature_columns(frame)
    cov = frame[feats].notna().mean().sort_values()
    low = cov[cov < 1.0]
    if len(low):
        print("feature coverage below 100% (NaN allowed; models decide):")
        for name, frac in low.items():
            print(f"  {name:28s} {frac * 100:6.2f}%")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    frame.to_parquet(args.out, index=False)
    print(f"saved -> {args.out}")

    if args.overrides_out and folds is not None and "outer_fold" in folds:
        ov = neighbor_overrides(frame, folds, "outer_fold")
        save_overrides(ov, args.overrides_out)


if __name__ == "__main__":
    main()
