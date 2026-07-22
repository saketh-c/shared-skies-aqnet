"""Sensor-day feature assembly for AQNet.

Builds the tabular training frame for the AQNet research model entirely from
files the main pipeline already produces — no new API pulls. The construction
mirrors the production 38-feature build (pipeline/06 + pipeline/03) with two
deliberate departures:

  * Demographic EJScreen columns (config.EXCLUDED_DEMOGRAPHIC) are excluded
    from every feature set — only the physical source-proximity EJ features
    (traffic / superfund / RMP / diesel-PM proximity) are model inputs.
    build_training_frame(keep_demographics=True) can retain the demographic
    columns as REFERENCE columns (used solely by the ablation stage), but
    feature_columns() never returns them.
  * The regression target is the Barkjohn et al. (2021) corrected PurpleAir
    PM2.5 (corrections.apply_target_correction), not the raw ATM reading.

Sources (all repo-relative):
  pipeline/purpleair_full_dataset.parquet   sensor-day PM2.5 + met + EJ + time
  pipeline/hms_smoke_by_sensor.parquet      NOAA HMS smoke tier (0-3)
  pipeline/airquality_by_cell.parquet       CAMS aod + cams_pm25 + dust (0.5-deg)
  pipeline/elevations.json                  sensor + tract elevations
  pipeline/sensor_tx_membership.csv         in-Texas target membership
  backend/static/tract_lookup.parquet       tract centroids + EJScreen columns

The parquet already carries engineered met, EJ, time-encoding and interaction
columns from pipeline/06; those are REUSED, not recomputed. Note its lat/lon
columns are TRACT CENTROIDS from the EJ join — this module re-derives lat/lon
from the sensor coordinates (latitude/longitude) so every downstream consumer
gets true sensor positions.

Neighbor features are same-day, leave-self-out, multi-radius aggregates and
reuse pipeline/neighbor_features.compute_neighbor_features_df — the single
source of truth shared with the production trainer. For cross-validation,
neighbor_features_per_fold() recomputes every nbr_* column against pools
restricted to each fold's TRAIN rows, so the tabular tier can score held-out
sensors without cross-fold neighbor leakage. External CTM/reanalysis
products (GEOS-CF, MERRA-2 parquets from data_external.py) attach by nearest
grid cell on the same day and stay NaN wherever a product has no coverage.

build_site_features() assembles the SAME feature vector at arbitrary
(lat, lon, date) site-days — used for external validation at EPA AQS monitor
locations, where every feature must come from PurpleAir / CAMS / tract data
and never from the AQS measurements themselves.

Smoke test (prints frame shape + coverage):
    python research/aqnet/features.py --quick
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ── Import bootstrap (works from any cwd: Colab, repo root, module import) ──

_AQNET_DIR = os.path.dirname(os.path.abspath(__file__))
_RESEARCH_DIR = os.path.dirname(_AQNET_DIR)
_ROOT = os.path.dirname(_RESEARCH_DIR)
for _p in (os.path.join(_ROOT, "pipeline"),
           os.path.join(_RESEARCH_DIR, "deeplearning"),
           _AQNET_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from corrections import apply_target_correction
from neighbor_features import compute_neighbor_features_df

# ── Paths & constants ───────────────────────────────────────────────────────

PA_DATASET = os.path.join(config.ROOT, "pipeline", "purpleair_full_dataset.parquet")
AQ_BY_CELL = os.path.join(config.ROOT, "pipeline", "airquality_by_cell.parquet")
HMS_SMOKE = os.path.join(config.ROOT, "pipeline", "hms_smoke_by_sensor.parquet")
ELEVATIONS = os.path.join(config.ROOT, "pipeline", "elevations.json")
MEMBERSHIP = os.path.join(config.ROOT, "pipeline", "sensor_tx_membership.csv")
TRACT_LOOKUP = os.path.join(config.ROOT, "backend", "static", "tract_lookup.parquet")

EARTH_R_KM = 6371.0

# Physical (non-demographic) EJScreen source-proximity features.
EJ_PHYSICAL = ["traffic_proximity", "superfund_proximity",
               "rmp_proximity", "diesel_pm_proximity"]

# Sensor-day meteorology columns (Open-Meteo point pulls in pipeline/06).
MET_COLS = ["temperature", "humidity", "pressure", "wind_speed", "precipitation"]

# Texas coast reference points (Brownsville -> Sabine Pass), identical to the
# production trainer (pipeline/03_train_enhanced.py TX_COAST_POINTS) so
# dist_to_coast is byte-compatible with the deployed feature.
TX_COAST_POINTS = [
    (25.97, -97.50),  # Brownsville
    (27.80, -97.40),  # Corpus Christi
    (28.93, -95.97),  # Freeport
    (29.30, -94.79),  # Galveston
    (29.70, -93.90),  # Sabine Pass
]


# ── Geometry helpers ────────────────────────────────────────────────────────

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


def _nearest_other_sensor_km(sensors):
    """{sensor_id: km to nearest OTHER sensor} over unique sensor sites.

    Constant per sensor and computed from the network geometry only, so it is
    LOSO-safe (mirrors the production dist_to_nearest_sensor).
    """
    lat_r = np.radians(sensors["lat"].to_numpy(dtype=np.float64))
    lon_r = np.radians(sensors["lon"].to_numpy(dtype=np.float64))
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2.0) ** 2)
    d = 2.0 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    np.fill_diagonal(d, np.inf)
    return dict(zip(sensors["sensor_id"], d.min(axis=1)))


def _nearest_cell_join(df, ext, value_cols, cell_lat_col, cell_lon_col):
    """Left-join a gridded product onto df by NEAREST (lat, lon) cell, same day.

    Nearest is resolved with a cKDTree in degree space (adequate at Texas
    latitudes; same convention as research/deeplearning/dataset.py), which
    also handles grids with missing cells — unlike a round-to-cell join.
    Rows whose date is absent from the product stay NaN.
    """
    if not value_cols:
        return df
    ext = ext.copy()
    ext["date"] = pd.to_datetime(ext["date"]).dt.normalize()
    cells = ext[[cell_lat_col, cell_lon_col]].drop_duplicates().reset_index(drop=True)
    tree = cKDTree(cells[[cell_lat_col, cell_lon_col]].to_numpy(dtype=np.float64))
    query = np.column_stack([df["lat"].to_numpy(dtype=np.float64),
                             df["lon"].to_numpy(dtype=np.float64)])
    _, idx = tree.query(query, k=1)

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["_cell_lat"] = cells[cell_lat_col].to_numpy()[idx]
    out["_cell_lon"] = cells[cell_lon_col].to_numpy()[idx]
    sub = ext[[cell_lat_col, cell_lon_col, "date"] + list(value_cols)].rename(
        columns={cell_lat_col: "_cell_lat", cell_lon_col: "_cell_lon"})
    sub = sub.drop_duplicates(["_cell_lat", "_cell_lon", "date"])
    out = out.merge(sub, on=["_cell_lat", "_cell_lon", "date"], how="left")
    return out.drop(columns=["_cell_lat", "_cell_lon"])


# ── Small loaders ───────────────────────────────────────────────────────────

def _tract_lookup():
    """Tract centroids + EJScreen columns with a canonical zero-filled GEOID."""
    tl = pd.read_parquet(TRACT_LOOKUP)
    tl["GEOID"] = tl["GEOID"].astype(str).str.zfill(11)
    return tl


def _load_elevations():
    """(sensor elevations {int id: m}, tract elevations {GEOID str: m})."""
    with open(ELEVATIONS) as f:
        payload = json.load(f)
    sens = {}
    for k, v in payload.get("sensors", {}).items():
        try:
            sens[int(float(k))] = float(v)
        except (TypeError, ValueError):
            continue
    tracts = {str(k): float(v) for k, v in payload.get("tracts", {}).items()
              if v is not None}
    return sens, tracts


def _out_of_texas_ids():
    """Sensor ids audited as OUTSIDE Texas — neighbor pool only, never targets.

    Sensors absent from the membership audit default to in-Texas (the same
    conservative-keep convention as the production trainer).
    """
    if not os.path.exists(MEMBERSHIP):
        print("[features] sensor_tx_membership.csv not found; keeping all sensors as targets")
        return set()
    mem = pd.read_csv(MEMBERSHIP)
    sid = pd.to_numeric(mem["sensor_id"], errors="coerce")
    in_tx = mem["in_tx"].astype(str).str.lower() == "true"
    return set(sid[~in_tx & sid.notna()].astype("int64").tolist())


def _add_time_and_interactions(df):
    """Production time encodings + interaction terms, computing ONLY missing
    columns so engineered values already in the parquet are reused verbatim."""
    if "month" not in df.columns:
        df["month"] = df["date"].dt.month
    if "dow" not in df.columns:
        df["dow"] = df["date"].dt.dayofweek
    if "day_of_year" not in df.columns:
        df["day_of_year"] = df["date"].dt.dayofyear
    if "month_sin" not in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    if "month_cos" not in df.columns:
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    if "dow_sin" not in df.columns:
        df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    if "dow_cos" not in df.columns:
        df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    if "doy_sin" not in df.columns:
        df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    if "doy_cos" not in df.columns:
        df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    if "temp_x_humidity" not in df.columns and {"temperature", "humidity"} <= set(df.columns):
        df["temp_x_humidity"] = df["temperature"] * df["humidity"] / 100.0
    if "wind_x_temp" not in df.columns and {"wind_speed", "temperature"} <= set(df.columns):
        df["wind_x_temp"] = df["wind_speed"] * df["temperature"] / 100.0
    return df


def _static_physical_features():
    """PHYSICAL_FEATURES minus the neighbor group (added by add_neighbor_features)."""
    return [f for f in config.PHYSICAL_FEATURES if not f.startswith("nbr_")]


# ── Sensor-day assembly ─────────────────────────────────────────────────────

def load_sensor_days(keep_demographics=False):
    """Sensor-day rows with sensor_id, date, lat, lon, pm25, GEOID, elevation
    plus every non-neighbor PHYSICAL_FEATURE computable from repo files, and
    any config.EXTRA_FEATURES column (CAMS dust) the aerosol parquet carries.

    Missing values are left as NaN (imputation is a modeling decision, not a
    data-assembly one). Demographic EJScreen columns are dropped here and never
    reappear downstream — unless keep_demographics=True retains them as
    REFERENCE columns for the ablation stage; feature_columns() still never
    returns them.
    """
    df = pd.read_parquet(PA_DATASET)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df[(df["date"] >= pd.Timestamp(config.DATE_START))
            & (df["date"] <= pd.Timestamp(config.DATE_END))]
    df = df.dropna(subset=["pm25"]).reset_index(drop=True)

    # Canonical coordinates. The parquet's lat/lon columns are TRACT CENTROIDS
    # from the pipeline/06 EJ join; replace them with the sensor coordinates.
    df = df.drop(columns=[c for c in ("lat", "lon") if c in df.columns])
    df["lat"] = df["latitude"].astype(float)
    df["lon"] = df["longitude"].astype(float)

    # Demographic EJScreen columns are excluded from AQNet features everywhere;
    # keep_demographics=True retains them ONLY as ablation reference columns.
    if not keep_demographics:
        df = df.drop(columns=[c for c in config.EXCLUDED_DEMOGRAPHIC
                              if c in df.columns])

    if "GEOID" in df.columns:
        df["GEOID"] = df["GEOID"].astype(str).str.zfill(11)

    # ── Physical EJ proximity (reuse parquet values; tract_lookup fallback) ──
    ej_missing = [c for c in EJ_PHYSICAL if c not in df.columns]
    if ej_missing and "GEOID" in df.columns:
        tl = _tract_lookup()
        df = df.merge(tl[["GEOID"] + ej_missing], on="GEOID", how="left")

    # ── Time encodings + interactions (reuse parquet columns when present) ──
    df = _add_time_and_interactions(df)

    # ── Spatial context ──
    sensors = (df.drop_duplicates("sensor_id")[["sensor_id", "lat", "lon"]]
               .reset_index(drop=True))
    if "dist_to_nearest_sensor" not in df.columns:
        nearest = _nearest_other_sensor_km(sensors)
        df["dist_to_nearest_sensor"] = df["sensor_id"].map(nearest).astype(float)
    if "dist_to_coast" not in df.columns:
        df["dist_to_coast"] = _min_dist_to_points(
            df["lat"].to_numpy(), df["lon"].to_numpy(), TX_COAST_POINTS)

    # ── NOAA HMS smoke tier (missing row = no smoke polygon = 0) ──
    if "hms_smoke" not in df.columns and os.path.exists(HMS_SMOKE):
        hms = pd.read_parquet(HMS_SMOKE)
        sid = pd.to_numeric(hms["sensor_id"], errors="coerce")
        hms = hms[sid.notna()].copy()
        hms["sensor_id"] = sid[sid.notna()].astype("int64")
        hms["date"] = pd.to_datetime(hms["date"]).dt.normalize()
        hms = hms.drop_duplicates(["sensor_id", "date"])
        df = df.merge(hms[["sensor_id", "date", "hms_smoke"]],
                      on=["sensor_id", "date"], how="left")
    if "hms_smoke" in df.columns:
        df["hms_smoke"] = df["hms_smoke"].fillna(0).astype("int16")
    else:
        df["hms_smoke"] = np.int16(0)

    # ── CAMS aerosol (aod + cams_pm25 + dust; archive starts 2022-08 -> NaN
    # before). dust joins by the same nearest cell as a config.EXTRA_FEATURES
    # column (tabular + dust-strata validation); NaN-tolerant — rows outside
    # the archive stay NaN and feature_columns() lists it only when present.
    aq_cols = [c for c in ("aod", "cams_pm25", "dust") if c not in df.columns]
    if aq_cols and os.path.exists(AQ_BY_CELL):
        aq = pd.read_parquet(AQ_BY_CELL)
        df = _nearest_cell_join(df, aq, [c for c in aq_cols if c in aq.columns],
                                "cell_lat", "cell_lon")

    # ── Elevation (sensor value where surveyed, else its tract centroid) ──
    if "elevation" not in df.columns and os.path.exists(ELEVATIONS):
        sens_elev, tract_elev = _load_elevations()
        elev = df["sensor_id"].map(sens_elev)
        if "GEOID" in df.columns:
            elev = elev.fillna(df["GEOID"].map(tract_elev))
        df["elevation"] = elev.astype(float)
    elif "elevation" not in df.columns:
        df["elevation"] = np.nan

    # ── Guarantee every non-neighbor physical feature exists (NaN allowed) ──
    static_feats = _static_physical_features()
    for f in static_feats:
        if f not in df.columns:
            print(f"[features] WARNING: {f} not computable from repo files; left NaN")
            df[f] = np.nan

    keep = ["sensor_id", "date", "lat", "lon", "pm25", "GEOID", "elevation"]
    keep += [f for f in static_feats if f not in keep]
    keep += [c for c in config.EXTRA_FEATURES if c in df.columns and c not in keep]
    if keep_demographics:
        keep += [c for c in config.EXCLUDED_DEMOGRAPHIC
                 if c in df.columns and c not in keep]
    return df[keep].reset_index(drop=True)


# ── Neighbor features ───────────────────────────────────────────────────────

def add_neighbor_features(df, pool_df=None, value_col=None):
    """Same-day, leave-self-out, multi-radius neighbor PM2.5 features.

    Adds nbr_pm25_25km, nbr_count_25km, nbr_pm25_50km, nbr_count_50km,
    nbr_std_50km, nbr_pm25_100km, nbr_count_100km via the shared BallTree
    implementation in pipeline/neighbor_features.py (haversine, one tree per
    date, self-exclusion by sensor_id) — a query row's own sensor never enters
    its aggregates, and only same-day values are used.

    pool_df optionally supplies the reference sensor readings (default: df
    itself). value_col picks the aggregated column; by default the corrected
    "target" when the pool has one, else raw "pm25", so neighbor features stay
    in the same units as the regression target. Pool rows with a non-finite
    value_col (e.g. NaN Barkjohn targets from missing humidity) are dropped
    from the POOL before any aggregation — they would otherwise poison the
    same-day means — and the drop count is printed. Query rows without
    value_col (e.g. AQS site-days, which have no PurpleAir reading) get NaN
    there and fall through to the pool-mean fallback for zero-neighbor rows.
    """
    q = df.copy()
    p = df.copy() if pool_df is None else pool_df.copy()
    if value_col is None:
        value_col = "target" if "target" in p.columns else "pm25"

    for frame in (q, p):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        if "latitude" not in frame.columns:
            frame["latitude"] = frame["lat"]
        if "longitude" not in frame.columns:
            frame["longitude"] = frame["lon"]

    # Drop non-finite pool readings BEFORE pooling (query rows keep their
    # positions; only the reference set shrinks).
    p_vals = pd.to_numeric(p[value_col], errors="coerce").to_numpy(dtype=np.float64)
    p_bad = ~np.isfinite(p_vals)
    if p_bad.any():
        print(f"[features] neighbor pool: dropped {int(p_bad.sum()):,} rows "
              f"with non-finite {value_col}")
        p = p[~p_bad].reset_index(drop=True)

    if "sensor_id" not in q.columns:
        q["sensor_id"] = ["q%d" % i for i in range(len(q))]
    if value_col not in q.columns:
        q[value_col] = np.nan

    nbr = compute_neighbor_features_df(q, p, target_col=value_col)
    out = df.copy()
    for col, arr in nbr.items():
        out[col] = arr
    return out


def neighbor_features_per_fold(df, folds):
    """Recompute every nbr_* column against TRAIN-fold-only neighbor pools.

    For each fold i in ``folds`` (a sequence of (train_idx, test_idx)
    positional-index pairs over df's row order), all nbr_* columns are
    recomputed for EVERY row of df with the pool restricted to that fold's
    TRAIN rows (query = all rows, pool = train rows; the leave-self-out rule
    still applies within the pool via sensor_id). Held-out sensors therefore
    never contribute to any neighbor aggregate used to score them — the
    cross-validation analogue of the production LOSO leakage fix in
    pipeline/03_train_enhanced.loso_cv, sharing the same single-source
    implementation (pipeline/neighbor_features.compute_neighbor_features_df).

    Aggregates the corrected "target" column when present (else raw "pm25")
    and drops non-finite pool readings per fold, both matching
    add_neighbor_features so full-pool and per-fold columns stay in the same
    units.

    Returns {fold_index: {column_name: np.ndarray}} with FULL-LENGTH arrays
    aligned to df row order — models_tabular.train_cv applies them to both
    the train and test rows of each fold via fold_col_overrides, and
    pipeline_colab persists them as f{fold}__{col} npz entries.
    """
    base = df.copy().reset_index(drop=True)
    base["date"] = pd.to_datetime(base["date"]).dt.normalize()
    if "latitude" not in base.columns:
        base["latitude"] = base["lat"]
    if "longitude" not in base.columns:
        base["longitude"] = base["lon"]
    value_col = "target" if "target" in base.columns else "pm25"

    print(f"[features] per-fold neighbor recompute: {len(folds)} folds x "
          f"{len(base):,} rows (value column: {value_col})")
    out = {}
    for i, (train_idx, _test_idx) in enumerate(folds):
        pool = base.iloc[np.asarray(train_idx)].reset_index(drop=True)
        vals = pd.to_numeric(pool[value_col], errors="coerce").to_numpy(dtype=np.float64)
        bad = ~np.isfinite(vals)
        if bad.any():
            print(f"[features] fold {i}: dropped {int(bad.sum()):,} rows with "
                  f"non-finite {value_col} from the neighbor pool")
            pool = pool[~bad].reset_index(drop=True)
        out[i] = compute_neighbor_features_df(base, pool, target_col=value_col)
    return out


# ── External CTM / reanalysis products ──────────────────────────────────────

def attach_external(df, geoscf_parquet=None, merra2_parquet=None):
    """Attach GEOS-CF / MERRA-2 features by nearest grid cell, same day.

    Each parquet (from data_external.py) has [date, lat, lon, <features>] on a
    regular grid; rows are matched to their nearest cell with a cKDTree in
    degree space and joined on the exact date. Features stay NaN wherever a
    product has no coverage (missing dates, columns, or file not provided).
    """
    out = df
    for path, names in ((geoscf_parquet, config.GEOSCF_FEATURES),
                        (merra2_parquet, config.MERRA2_FEATURES)):
        if path is None:
            continue
        if not os.path.exists(path):
            print(f"[features] external parquet missing, skipped: {path}")
            continue
        ext = pd.read_parquet(path)
        have = [c for c in names if c in ext.columns and c not in out.columns]
        out = _nearest_cell_join(out, ext, have, "lat", "lon")
        for c in names:
            if c not in out.columns:
                out[c] = np.nan
    return out


# ── Training frame ──────────────────────────────────────────────────────────

def build_training_frame(correction="barkjohn", geoscf_parquet=None,
                         merra2_parquet=None, in_texas_only=True,
                         start=None, end=None, keep_demographics=False):
    """Full sensor-day training frame: assembly + target correction.

    Order matters: the target correction is applied FIRST so the neighbor
    features aggregate corrected values (same units as the target), and the
    neighbor features are computed BEFORE the in-Texas filter so border-state
    sensors stay in the pool as neighbors (production convention) while never
    becoming prediction targets. start/end optionally narrow the date window
    (used by smoke tests).

    keep_demographics=True retains the four config.EXCLUDED_DEMOGRAPHIC
    columns in the frame as REFERENCE columns — used ONLY by the ablation
    stage, which builds its plus_demographics feature list explicitly.
    feature_columns() never returns them regardless.

    Returns sensor_id, date, lat, lon, target (+ raw pm25, GEOID, elevation,
    in_tx for reference) and every feature column; run feature_columns() on
    the result to get the model input list.
    """
    df = load_sensor_days(keep_demographics=keep_demographics)
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    df = df.reset_index(drop=True)

    df = apply_target_correction(df, method=correction)
    df = add_neighbor_features(df)
    df = attach_external(df, geoscf_parquet=geoscf_parquet,
                         merra2_parquet=merra2_parquet)

    out_ids = _out_of_texas_ids()
    df["in_tx"] = ~df["sensor_id"].isin(out_ids)
    if in_texas_only:
        n_out = int((~df["in_tx"]).sum())
        if n_out:
            print(f"[features] dropped {n_out:,} out-of-Texas rows "
                  f"({df.loc[~df['in_tx'], 'sensor_id'].nunique()} sensors) "
                  f"after neighbor-pool use")
        df = df[df["in_tx"]].reset_index(drop=True)

    feats = feature_columns(df)
    keep = ["sensor_id", "date", "lat", "lon", "target",
            "pm25", "GEOID", "elevation", "in_tx"]
    keep += [f for f in feats if f not in keep]
    if keep_demographics:
        keep += [c for c in config.EXCLUDED_DEMOGRAPHIC
                 if c in df.columns and c not in keep]
    return df[keep].reset_index(drop=True)


def feature_columns(df):
    """Model input columns: PHYSICAL_FEATURES, then whichever
    config.EXTRA_FEATURES are present (CAMS dust), then external features.

    Raises if any physical feature is missing and ASSERTS that the RETURNED
    list contains no excluded demographic column — the hard methodological
    guarantee that demographics never enter prediction. The check is on the
    returned list (not the frame), so frames built with
    keep_demographics=True may carry the demographic REFERENCE columns; they
    are still never returned here.
    """
    missing = [f for f in config.PHYSICAL_FEATURES if f not in df.columns]
    if missing:
        raise KeyError(f"frame is missing physical features: {missing} "
                       f"(build with load_sensor_days + add_neighbor_features)")
    feats = list(config.PHYSICAL_FEATURES)
    feats += [c for c in config.EXTRA_FEATURES if c in df.columns]
    feats += [c for c in list(config.GEOSCF_FEATURES) + list(config.MERRA2_FEATURES)
              if c in df.columns]
    banned = [c for c in feats if c in config.EXCLUDED_DEMOGRAPHIC]
    assert not banned, f"demographic columns must never enter the feature list: {banned}"
    return feats


# ── Arbitrary site-days (external AQS validation) ───────────────────────────

def _sample_sensor_fields(sites, pool, cols=MET_COLS, k=8, power=2.0):
    """Same-day sampling of sensor-day fields onto arbitrary points.

    Continuous columns are inverse-distance-weighted from the k nearest
    same-day sensors (degree space, matching the deep-learning gridder's
    convention); the ordinal hms_smoke tier is taken from the single nearest
    sensor so category levels are not blurred. Returns ({col: values}, hms).
    """
    n = len(sites)
    out = {c: np.full(n, np.nan) for c in cols}
    hms = np.zeros(n, dtype=np.float64)
    s_lat = sites["lat"].to_numpy(dtype=np.float64)
    s_lon = sites["lon"].to_numpy(dtype=np.float64)
    pool_by_date = {d: g for d, g in pool.groupby("date")}

    for d, q_idx in sites.groupby("date").indices.items():
        g = pool_by_date.get(d)
        if g is None or not len(g):
            continue
        tree = cKDTree(g[["lat", "lon"]].to_numpy(dtype=np.float64))
        kk = int(min(k, len(g)))
        dist, idx = tree.query(np.column_stack([s_lat[q_idx], s_lon[q_idx]]), k=kk)
        if kk == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        w = 1.0 / np.maximum(dist, 1e-6) ** power
        for c in cols:
            vals = g[c].to_numpy(dtype=np.float64)[idx]
            finite = np.isfinite(vals)
            wv = np.where(finite, w, 0.0)
            denom = wv.sum(axis=1)
            est = (wv * np.where(finite, vals, 0.0)).sum(axis=1) / np.maximum(denom, 1e-12)
            out[c][q_idx] = np.where(denom > 0, est, np.nan)
        if "hms_smoke" in g.columns:
            hms[q_idx] = g["hms_smoke"].to_numpy(dtype=np.float64)[idx[:, 0]]
    return out, hms


def build_site_features(sites, correction="barkjohn", geoscf_parquet=None,
                        merra2_parquet=None, pool=None):
    """Assemble the SAME feature vector at arbitrary (lat, lon, date) site-days.

    Built for external validation at EPA AQS monitor locations: every feature
    comes from PurpleAir sensor-days, CAMS cells, tract data, and the optional
    CTM/reanalysis parquets — NEVER from AQS measurements, keeping reference
    data strictly out of feature computation. Feature construction mirrors
    build_training_frame:

      met (5 cols)          same-day IDW from the k=8 nearest PurpleAir sensors
      hms_smoke             same-day nearest-sensor tier (0 = no smoke)
      aod/cams_pm25/dust    nearest 0.5-deg CAMS cell, same day
      EJ physical, elevation nearest tract centroid (GEOID)
      dist_to_*             coast reference points / nearest PurpleAir sensor
      neighbor features     leave-self-out over the PurpleAir pool (corrected)
      GEOS-CF / MERRA-2     nearest grid cell, same day (when parquets given)

    sites needs columns lat, lon, date (site_id optional and passed through).
    pool overrides the PurpleAir sensor-day pool — default is
    load_sensor_days() with the same target correction applied, so neighbor
    aggregates are in corrected-target units exactly as in training.
    """
    s = sites.copy().reset_index(drop=True)
    if "lat" not in s.columns and "latitude" in s.columns:
        s["lat"] = s["latitude"]
    if "lon" not in s.columns and "longitude" in s.columns:
        s["lon"] = s["longitude"]
    s["date"] = pd.to_datetime(s["date"]).dt.normalize()
    s["lat"] = s["lat"].astype(float)
    s["lon"] = s["lon"].astype(float)
    if "latitude" not in s.columns:
        s["latitude"] = s["lat"]
    if "longitude" not in s.columns:
        s["longitude"] = s["lon"]
    # Site ids must never collide with PurpleAir sensor ids (self-exclusion in
    # the neighbor computation compares sensor_id as strings).
    if "sensor_id" not in s.columns:
        base = (s["site_id"].astype(str) if "site_id" in s.columns
                else pd.Series(np.arange(len(s)).astype(str), index=s.index))
        s["sensor_id"] = "site_" + base

    if pool is None:
        pool = apply_target_correction(load_sensor_days(), method=correction)

    # ── Same-day sensor fields (met + smoke) ──
    met, hms = _sample_sensor_fields(s, pool)
    for c in MET_COLS:
        s[c] = met[c]
    s["hms_smoke"] = hms.astype("int16")

    # ── CAMS aerosol (dust included, matching load_sensor_days) ──
    if os.path.exists(AQ_BY_CELL):
        aq = pd.read_parquet(AQ_BY_CELL)
        s = _nearest_cell_join(s, aq, [c for c in ("aod", "cams_pm25", "dust")
                                       if c not in s.columns and c in aq.columns],
                               "cell_lat", "cell_lon")

    # ── Tract join: GEOID, physical EJ proximity, elevation ──
    tl = _tract_lookup()
    tree = cKDTree(tl[["lat", "lon"]].to_numpy(dtype=np.float64))
    _, idx = tree.query(np.column_stack([s["lat"].to_numpy(dtype=np.float64),
                                         s["lon"].to_numpy(dtype=np.float64)]), k=1)
    s["GEOID"] = tl["GEOID"].to_numpy()[idx]
    for c in EJ_PHYSICAL:
        if c not in s.columns and c in tl.columns:
            s[c] = tl[c].to_numpy()[idx]
    if "elevation" not in s.columns and os.path.exists(ELEVATIONS):
        _, tract_elev = _load_elevations()
        s["elevation"] = s["GEOID"].map(tract_elev).astype(float)

    # ── Spatial context ──
    sensors = (pool.drop_duplicates("sensor_id")[["sensor_id", "lat", "lon"]]
               .reset_index(drop=True))
    stree = cKDTree(sensors[["lat", "lon"]].to_numpy(dtype=np.float64))
    kk = int(min(8, len(sensors)))
    _, sidx = stree.query(np.column_stack([s["lat"].to_numpy(dtype=np.float64),
                                           s["lon"].to_numpy(dtype=np.float64)]), k=kk)
    if kk == 1:
        sidx = sidx[:, None]
    # Degree-space candidates, exact haversine for the reported km value.
    cand_lat = sensors["lat"].to_numpy()[sidx]
    cand_lon = sensors["lon"].to_numpy()[sidx]
    d_km = _haversine_km(s["lat"].to_numpy()[:, None], s["lon"].to_numpy()[:, None],
                         cand_lat, cand_lon)
    s["dist_to_nearest_sensor"] = d_km.min(axis=1)
    s["dist_to_coast"] = _min_dist_to_points(
        s["lat"].to_numpy(), s["lon"].to_numpy(), TX_COAST_POINTS)

    # ── Time encodings + interactions (from the sampled met) ──
    s = _add_time_and_interactions(s)

    # ── Neighbor features from the PurpleAir pool only ──
    s = add_neighbor_features(s, pool_df=pool)

    # ── External CTM / reanalysis ──
    s = attach_external(s, geoscf_parquet=geoscf_parquet,
                        merra2_parquet=merra2_parquet)

    for f in _static_physical_features():
        if f not in s.columns:
            print(f"[features] WARNING: {f} not computable at sites; left NaN")
            s[f] = np.nan
    return s


# ── CLI smoke test ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Build (and optionally save) the AQNet training frame.")
    ap.add_argument("--correction", default="barkjohn", choices=["barkjohn", "raw"],
                    help="target correction method (default barkjohn)")
    ap.add_argument("--geoscf-parquet", default=None, help="GEOS-CF parquet from data_external.py")
    ap.add_argument("--merra2-parquet", default=None, help="MERRA-2 parquet from data_external.py")
    ap.add_argument("--all-sensors", action="store_true",
                    help="keep out-of-Texas sensors as rows (default: in-Texas targets only)")
    ap.add_argument("--quick", action="store_true",
                    help="restrict to the last 120 days for a fast smoke test")
    ap.add_argument("--out", default=None, help="optional output parquet path")
    args = ap.parse_args()

    start = None
    if args.quick:
        start = (pd.Timestamp(config.DATE_END) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
        print(f"[features] --quick: window {start} .. {config.DATE_END}")

    df = build_training_frame(correction=args.correction,
                              geoscf_parquet=args.geoscf_parquet,
                              merra2_parquet=args.merra2_parquet,
                              in_texas_only=not args.all_sensors,
                              start=start)
    feats = feature_columns(df)
    print(f"training frame: {len(df):,} rows x {len(df.columns)} cols, "
          f"{df['sensor_id'].nunique()} sensors, "
          f"{df['date'].min().date()} .. {df['date'].max().date()}")
    print(f"features ({len(feats)}): {feats}")
    cov = df[feats].notna().mean().sort_values()
    low = cov[cov < 1.0]
    if len(low):
        print("feature coverage below 100% (NaN allowed; models decide imputation):")
        for name, frac in low.items():
            print(f"  {name:24s} {frac * 100:6.2f}%")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        df.to_parquet(args.out, index=False)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
