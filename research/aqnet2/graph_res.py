"""AQNet v2 T2: masked graph-attention residual (stages `graphpre`, `graphres`).

Residual target r1 = y - T1_oof (DESIGN S7). The between-site workhorse:
a shielded graph transformer over PurpleAir stations that learns to
reconstruct a masked station (or an AQS query node) from its airshed
neighborhood, then is fine-tuned per (outer k, inner j) fold to predict the
T1 residual at held-out units.

Why this shape (v1 evidence):
  * v1's residual kriging earned a composite weight of exactly 0.000 -- the
    device nugget plus isotropic stationary kernels had nothing to add over
    Tier-1. The graph attention replaces the stationary kernel with a learned
    non-stationary decay (edge-feature-biased logits: distance, wind
    alignment, terrain) and the nugget with h_rel, a per-station reliability
    latent learned from each station's own embargoed disagreement history.
  * v1's UNet consumed same-day network state without a deployment story.
    Here the deployment-honest rule is structural: AQS observations NEVER
    appear as input nodes (serving has no same-day FRM feed), vault sites
    never appear at all (asserted against folds2 vault_sites), and shielded
    attention means only observed PA nodes emit keys/values.
  * Full-window masking: a masked or query node has its ENTIRE observation
    window replaced by a learned [MASK] embedding and its h_rel by a learned
    null vector, so the pretext task is spatial interpolation, never
    own-history extrapolation (DESIGN S7).

Leakage rules implemented here (load-bearing):
  * Pretraining consumes RAW PA only (pa_raw from pa_calibrated.parquet, or
    the committed PA parquet ATM channel) -- no FRM-derived labels, no
    calibration, no T0. Calibration enters at fine-tune, per fold, through
    the pa_cal_f{k}_{j} nested columns.
  * h_rel is computed from days <= t - (TEMPORAL_EMBARGO_DAYS + 1) only;
    hrel_day_indices() carries the unit-testable no-future-leak assert.
  * The vault (sites + the >= VAULT_DATE_START period) is never trained on,
    never an input, and gets avail=0 / oof_r=NaN in oof_tier2.npz.
  * 100-km hard zero (BUILD_NOTES contract #5): rows with
    nbr_pacal_avail_100km == 0 get avail=0 AND oof_r=NaN, as do rows whose
    graph neighborhood is empty after shielding.

Fold-assignment epistemic ensemble: the per-(k, j) fine-tuned model set IS
the ensemble (seed ensembles rejected -- DESIGN S7); sigma in oof_tier2.npz
is the model's heteroscedastic head combined with the across-member spread
in quadrature.

Budget (one RTX6000, DESIGN S13): pretrain 150 epochs over ~1800 day-graphs
at 4 days/batch (~68k steps) fits <= 4 h; 20 fine-tunes x 40 epochs with a
seeded 50% per-epoch day subsample at 8 days/batch (~90k steps total) fit
<= 6 h. Checkpoints are atomic (tmp + os.replace), carry
{model, optimizer, scheduler, scaler, rng_state, epoch, cfg, fold_id}, are
written every epoch AND every 30 min, and `--resume` autodetects *_last.pt.

CLI:
    python graph_res.py graphpre  [--quick] [--resume] [--variant temporal] [--study]
    python graph_res.py graphres  [--quick] [--resume] [--variant temporal]
    python graph_res.py predict   [--quick] [--variant temporal]
Sentinels: graphpre -> graphpre[_temporal]_marker.json (+ last.pt);
graphres -> oof_tier2[_temporal].npz. FORCE=1 re-runs.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd

import config2

# -- Guarded heavy import (module must import with no torch installed) -------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    F = None
    HAS_TORCH = False
    print("[aqnet2] graph_res: torch not installed -- graph build/caching "
          "works, training and prediction need torch (pip install torch)")

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    cKDTree = None
    HAS_SCIPY = False
    print("[aqnet2] graph_res: scipy not installed -- brute-force kNN "
          "fallback in graph build (pip install scipy)")

# -- Constants ---------------------------------------------------------------

N_AIRSHEDS = 12            # k-means airshed clusters (DESIGN S7)
KNN_K = 10                 # in-edges per node from candidate stations
CAND_POOL = 64             # nearest-station candidate pool before filtering
CROSS_MAX_KM = 150.0       # cross-airshed edge only if closer than this
CROSS_MAX_DELEV_M = 500.0  # ... and |delta elevation| below this

D_MODEL = 128
N_HEADS = 8
N_LAYERS = 4
D_FF = 1152                # pre-LN FFN width (sizes the ~1.5M param budget)
D_OBS = 64                 # observation-window embedding width
D_REL = 32                 # h_rel GRU hidden width
OBS_WINDOW = 7             # trailing days of own observations (t .. t-6)

HREL_WINDOW = 30           # embargoed residual-history window length
HREL_EMBARGO_DAYS = int(config2.TEMPORAL_EMBARGO_DAYS) + 1   # days <= t-8

PRETRAIN_EPOCHS = 150
FINETUNE_EPOCHS = 40
FINETUNE_DAY_FRAC = 0.5    # seeded per-epoch day subsample (budget device)
PRETRAIN_BATCH_DAYS = 4
FINETUNE_BATCH_DAYS = 8
PREDICT_BATCH_DAYS = 16
QUICK_EPOCHS = 2
QUICK_WINDOW_DAYS = 92
QUICK_OUTER_FOLDS = 2

LR_PRETRAIN = 3e-4
LR_FINETUNE = 1e-4
WEIGHT_DECAY = 1e-3
GRAD_CLIP = 5.0            # v1 models_deep convention
WARMUP_EPOCHS = 5          # v1 _make_scheduler convention
CKPT_EVERY_SEC = 1800      # 30-min wall-clock checkpoint (DESIGN S13)

MASK_FRAC_LO, MASK_FRAC_HI = 0.20, 0.40   # pretrain station-mask fraction
BALL_KM_LO, BALL_KM_HI = 50.0, 150.0      # structured ball-mask radius
FINETUNE_PA_MASK_FRAC = 0.30              # PA-task mask fraction at finetune
AQS_TASK_WEIGHT = 0.8      # 4:1 mask-the-AQS-site oversampling, expectation form
PINBALL_WEIGHT = 0.25      # q05/q95 auxiliary weight (DESIGN S7)
HUBER_DELTA_NORM = 1.0     # Huber delta in NORMALIZED obs space (v1's 15
                           # ug/m3 was raw-space; ~1 sigma post log1p+z)
LOGVAR_CLAMP = 8.0
LEAKAGE_GAP_R2 = 0.01      # fold-pure-vs-shared decision threshold
STUDY_EPOCHS = 20

EARTH_R_KM = 6371.0
VAULT_DATE_START = getattr(config2, "VAULT_DATE_START", "2026-01-01")

# Node met channels: gridded/reanalysis columns already joined into the frame
# (deployment-available everywhere). Wind pair candidates first present wins;
# the rest are taken when present, order fixed here for determinism.
WIND_CANDIDATES = [("merra2_u10", "merra2_v10"),
                   ("merra2_u10m", "merra2_v10m"),
                   ("era5_u10", "era5_v10"),
                   ("era5_u10m", "era5_v10m")]
MET_CANDIDATES = ["merra2_t2m", "merra2_t2m_c", "merra2_qv2m", "merra2_pblh",
                  "era5_t2m", "shortwave", "et0", "cloud_cover", "hms_smoke",
                  "aod", "dust"]
MAX_MET_COLS = 10
ELEV_CANDIDATES = ["st_elev", "st_elevation", "st_elevation_m", "st_dem"]


def _say(msg):
    print(f"[aqnet2] graph_res: {msg}", flush=True)


def _banner(name):
    print(f"[aqnet2] " + f"── stage: {name} "
          + "─" * max(0, 58 - len(name)), flush=True)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


def _write_json_atomic(payload, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
    os.replace(tmp, path)


def _save_npz_atomic(path, **arrays):
    tmp = str(path) + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _dates_ns(s):
    """Any date-like series -> normalized datetime64[ns] (AQS parquets are
    datetime64[us] under pandas 3 -- normalize on every load)."""
    return pd.to_datetime(s).dt.normalize().astype("datetime64[ns]")


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2)
    return 2.0 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _equirect_xy_km(lat, lon, lat0, lon0):
    """Equirectangular projection to km around (lat0, lon0) -- adequate at
    Texas latitudes for kNN/k-means (v1 degree-space cKDTree precedent,
    upgraded to km so the wind features share units)."""
    x = (np.asarray(lon, dtype=np.float64) - lon0) * 111.320 * math.cos(
        math.radians(lat0))
    y = (np.asarray(lat, dtype=np.float64) - lat0) * 110.574
    return x, y


def hrel_day_indices(t_idx):
    """Day-axis indices of the h_rel window for day t_idx.

    The window is the HREL_WINDOW days ENDING at t_idx - HREL_EMBARGO_DAYS
    (config2.TEMPORAL_EMBARGO_DAYS + 1) -- the embargo guarantees no residual
    from t-7..t can inform day t's reliability latent. The assert is the
    unit-testable no-future-leak contract (tests/ call this directly).
    Indices may be negative (pre-history); callers mask them invalid.
    """
    last = int(t_idx) - HREL_EMBARGO_DAYS
    idx = np.arange(last - HREL_WINDOW + 1, last + 1)
    assert idx.max() <= t_idx - HREL_EMBARGO_DAYS, (
        "h_rel future leak: window reaches past the embargo")
    return idx


# -- Seeded numpy k-means (no sklearn dependency) ----------------------------

def _kmeans(X, k, seed, iters=60):
    """Plain Lloyd k-means, seeded via default_rng, deterministic given the
    (sorted-id-ordered) row order of X. Returns (labels, centroids)."""
    n = len(X)
    k = int(min(k, n))
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)          # X rows arrive in sorted-station order
    cent = X[order[:k]].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        d2 = ((X[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
        new = d2.argmin(axis=1)
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        for c in range(k):
            m = labels == c
            if m.any():
                cent[c] = X[m].mean(axis=0)
    return labels, cent


# -- Data loading ------------------------------------------------------------

def _default_paths():
    return {
        "frame": config2.artifact("frame_truth.parquet"),
        "folds": config2.artifact("folds2.json"),
        "tier1": config2.artifact("oof_tier1.npz"),
        "pa_calibrated": config2.artifact("pa_calibrated.parquet"),
        "pa_parquet": os.path.join(config2.PIPELINE_DIR,
                                   "purpleair_full_dataset.parquet"),
    }


def _load_folds_dict(path, n_rows=None):
    """folds2.json as a plain dict (site + row-level keys). Content-hash
    verification is folds2.load_folds's job; here only the n_rows guard
    (row misalignment silently corrupts every fold) is enforced."""
    if not os.path.exists(path):
        raise SystemExit(f"[aqnet2] graph_res: {path} missing -- run the "
                         f"folds/features stages first")
    with open(path, encoding="utf-8") as fh:
        folds = json.load(fh)
    if n_rows is not None and folds.get("n_rows") not in (None, n_rows):
        raise SystemExit(
            f"[aqnet2] graph_res: folds2.json n_rows={folds.get('n_rows')} "
            f"!= frame rows {n_rows} -- stale folds artifact")
    return folds


def load_raw_pa(paths, start, end):
    """RAW PA sensor-days for pretraining: sensor_id(str), date(ns), lat,
    lon, pa_raw. Prefers pa_calibrated.parquet's pa_raw column (identical
    archive, coordinates already joined by frame2 conventions); falls back
    to the committed PA parquet ATM channel. NO FRM-derived columns are read
    on either path -- the pretrain input contract (DESIGN S7)."""
    cal_path = paths["pa_calibrated"]
    if os.path.exists(cal_path):
        df = pd.read_parquet(cal_path)
        if "pa_raw" not in df.columns:
            raise SystemExit(f"[aqnet2] graph_res: {cal_path} lacks pa_raw")
        df = df[["sensor_id", "date", "pa_raw"]
                + [c for c in ("lat", "lon") if c in df.columns]].copy()
        src = "pa_calibrated.pa_raw"
    else:
        pa_path = paths["pa_parquet"]
        if not os.path.exists(pa_path):
            raise SystemExit("[aqnet2] graph_res: neither pa_calibrated."
                             "parquet nor the PA parquet exists")
        # latitude/longitude are the SENSOR coords; lat/lon in this parquet
        # are TRACT CENTROIDS (audited hazard) -- never touch the latter.
        df = pd.read_parquet(pa_path, columns=["sensor_id", "date", "pm25",
                                               "latitude", "longitude"])
        df = df.rename(columns={"pm25": "pa_raw", "latitude": "lat",
                                "longitude": "lon"})
        src = "purpleair_full_dataset.pm25 (ATM)"
    df["sensor_id"] = df["sensor_id"].astype(str)
    df["date"] = _dates_ns(df["date"])
    if "lat" not in df.columns or "lon" not in df.columns:
        pa = pd.read_parquet(paths["pa_parquet"],
                             columns=["sensor_id", "latitude", "longitude"])
        pa["sensor_id"] = pa["sensor_id"].astype(str)
        coords = (pa.dropna(subset=["latitude", "longitude"])
                    .drop_duplicates("sensor_id")
                    .rename(columns={"latitude": "lat", "longitude": "lon"}))
        df = df.merge(coords, on="sensor_id", how="left")
    df = df[(df["date"] >= pd.Timestamp(start))
            & (df["date"] <= pd.Timestamp(end))]
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    _say(f"raw PA source: {src} ({len(df):,} sensor-days, "
         f"{df['sensor_id'].nunique():,} sensors)")
    return df


def _pick_wind_cols(columns):
    for u, v in WIND_CANDIDATES:
        if u in columns and v in columns:
            return u, v
    cols = set(columns)
    for c in sorted(cols):
        for tag in ("u10", "u10m"):
            if c.endswith("_" + tag):
                v = c[: -len(tag)] + "v" + tag[1:]
                if v in cols:
                    return c, v
    return None, None


def _pick_elev_col(columns):
    for c in ELEV_CANDIDATES:
        if c in columns:
            return c
    return None


def _met_cols(columns, wind_uv):
    met = [c for c in MET_CANDIDATES if c in columns]
    met = [c for c in met if c not in wind_uv][:MAX_MET_COLS - 2]
    return list(wind_uv) + met if wind_uv[0] is not None else met


def _fill_matrix(n_units, n_days, uidx, didx, values, dtype=np.float32):
    """Scatter (unit, day) values into a NaN matrix; frame unit-days are
    unique by construction so later-write-wins is moot."""
    m = np.full((n_units, n_days), np.nan, dtype=dtype)
    m[uidx, didx] = np.asarray(values, dtype=dtype)
    return m


# -- Context: node universe, matrices, graph ---------------------------------

def build_context(frame, folds, cfg, paths=None):
    """Everything the trainers/predictor share, built once.

    Node universe = sorted PA stations (raw-PA archive, non-vault window)
    + sorted AQS query sites from the frame. Deployment-honest rule asserted
    here: no vault unit may enter the universe, AQS nodes never emit K/V
    (is_station mask). All per-day matrices are [n_units, n_days] float32
    with NaN as the only missingness representation.
    """
    paths = dict(_default_paths(), **(paths or {}))
    quick = bool(cfg.get("quick"))
    temporal = cfg.get("variant") == "temporal"

    end = min(pd.Timestamp(config2.DATE_END),
              pd.Timestamp(VAULT_DATE_START) - pd.Timedelta(days=1))
    if temporal:
        end = min(end, pd.Timestamp(config2.TEMPORAL_CUTOFF)
                  - pd.Timedelta(days=1))
    start = pd.Timestamp(config2.DATE_START)
    if quick:
        # Fixed pre-vault summer window — the SAME quick window as frame2/
        # calibrate/priors/field_res. A trailing window here intersected the
        # quick calibrated parquet (2024 Q3) on zero days, leaving an empty
        # station universe (smoke4 argmin crash).
        start = pd.Timestamp("2024-07-01")
        end = min(end, pd.Timestamp("2024-09-30"))
        _say(f"--quick: window {start.date()} .. {end.date()}")
    dates = pd.date_range(start, end, freq="D")
    n_days = len(dates)
    # Keys are pd.Timestamp (iterating the DatetimeIndex), NOT datetime64 --
    # the v1 dataset.py Timestamp-vs-datetime64 hash-mismatch hazard
    # (audit 07-grids S1) that silently empties every .map() lookup.
    day_of = {d: i for i, d in enumerate(dates)}

    raw = load_raw_pa(paths, start, end)
    station_ids = np.sort(raw["sensor_id"].unique().astype(str))
    st_unit_ids = np.array(["pa_" + s for s in station_ids])

    aqs = frame.loc[frame["unit_type"] == "aqs",
                    ["unit_id", "lat", "lon"]].copy()
    aqs["unit_id"] = aqs["unit_id"].astype(str)
    vault = set(str(v) for v in folds.get("vault_sites", []))
    vault |= ({f"aqs_{v}" for v in list(vault)}
              | {f"pa_{v}" for v in list(vault)})
    aqs = aqs[~aqs["unit_id"].isin(vault)]
    q_unit_ids = np.sort(aqs["unit_id"].unique())

    unit_ids = np.concatenate([st_unit_ids, q_unit_ids])
    # Deployment-honest assert: vault sites never appear at all (INTERFACES).
    breach = sorted(set(unit_ids.tolist()) & vault)
    assert not breach, f"vault units in the T2 node universe: {breach[:5]}"
    n_st = len(st_unit_ids)
    if n_st == 0:
        raise SystemExit(
            f"[aqnet2] graph_res: zero PA stations in the window "
            f"{start.date()}..{end.date()} — the raw-PA source and the "
            "graph window do not overlap (check pa_calibrated.parquet's "
            "window vs this stage's).")
    n_units = len(unit_ids)
    unit_index = {u: i for i, u in enumerate(unit_ids)}
    is_station = np.zeros(n_units, dtype=bool)
    is_station[:n_st] = True

    # Coordinates: stations from the raw archive, queries from the frame.
    st_coords = raw.drop_duplicates("sensor_id").set_index("sensor_id")
    lat = np.full(n_units, np.nan)
    lon = np.full(n_units, np.nan)
    lat[:n_st] = st_coords.loc[station_ids, "lat"].to_numpy(dtype=np.float64)
    lon[:n_st] = st_coords.loc[station_ids, "lon"].to_numpy(dtype=np.float64)
    if len(q_unit_ids):
        q_first = aqs.drop_duplicates("unit_id").set_index("unit_id")
        lat[n_st:] = q_first.loc[q_unit_ids, "lat"].to_numpy(dtype=np.float64)
        lon[n_st:] = q_first.loc[q_unit_ids, "lon"].to_numpy(dtype=np.float64)

    # Raw observation matrix [n_st, n_days].
    st_pos = {s: i for i, s in enumerate(station_ids)}
    ridx = raw["sensor_id"].map(st_pos).to_numpy(dtype=np.int64)
    rday = raw["date"].map(day_of)
    ok = rday.notna().to_numpy()
    obs_raw = _fill_matrix(n_st, n_days, ridx[ok],
                           rday.to_numpy()[ok].astype(np.int64),
                           raw["pa_raw"].to_numpy(dtype=np.float32)[ok])

    # Frame row alignment (unit idx, day idx); rows off-universe/off-window
    # get -1 and are structurally unavailable downstream.
    f_uid = frame["unit_id"].astype(str).map(unit_index)
    f_day = pd.Series(_dates_ns(frame["date"])).map(day_of)
    row_unit = f_uid.fillna(-1).to_numpy(dtype=np.int64)
    row_day = f_day.fillna(-1).to_numpy(dtype=np.int64)

    # Node met / wind / elevation from the frame's gridded-join columns
    # (deployment-available; the frame is the single-builder parity source).
    wind_u, wind_v = _pick_wind_cols(frame.columns)
    if wind_u is None:
        _say("WARNING: no u10/v10 wind columns in the frame -- wind edge "
             "features disabled (wind_ok=0), airsheds cluster on coords+elev")
    met_cols = _met_cols(frame.columns, (wind_u, wind_v))
    elev_col = _pick_elev_col(frame.columns)
    if elev_col is None:
        _say("WARNING: no st_ elevation column in the frame -- delta-elev "
             "edge filter degrades to distance-only, node elev = 0")

    inb = (row_unit >= 0) & (row_day >= 0)
    met = np.full((n_units, n_days, max(len(met_cols), 1)), np.nan,
                  dtype=np.float32)
    for ci, c in enumerate(met_cols):
        met[row_unit[inb], row_day[inb], ci] = (
            frame[c].to_numpy(dtype=np.float32)[inb])
    u_mat = v_mat = None
    if wind_u is not None:
        u_mat = _fill_matrix(n_units, n_days, row_unit[inb], row_day[inb],
                             frame[wind_u].to_numpy(dtype=np.float64)[inb])
        v_mat = _fill_matrix(n_units, n_days, row_unit[inb], row_day[inb],
                             frame[wind_v].to_numpy(dtype=np.float64)[inb])
    elev = np.zeros(n_units, dtype=np.float64)
    elev_ok = np.zeros(n_units, dtype=np.float64)
    if elev_col is not None:
        ev = np.full(n_units, np.nan)
        ev[row_unit[inb]] = frame[elev_col].to_numpy(dtype=np.float64)[inb]
        elev_ok = np.isfinite(ev).astype(np.float64)
        elev = np.where(np.isfinite(ev), ev, 0.0)

    graph = build_graph(lat, lon, elev, elev_ok, is_station, u_mat, v_mat,
                        st_unit_ids, q_unit_ids, cfg)

    # h_rel source: raw-obs residual vs graph-neighborhood median (raw PA on
    # BOTH stages -- no FRM-derived data ever enters the reliability latent).
    R = _residual_vs_neighborhood(obs_raw, graph["nb_idx_st"])
    R_univ = np.full((n_units, n_days), np.nan, dtype=np.float32)
    R_univ[:n_st] = R

    doy = dates.dayofyear.to_numpy()
    dow = dates.dayofweek.to_numpy()
    harmonics = np.stack([np.sin(2 * np.pi * doy / 365.0),
                          np.cos(2 * np.pi * doy / 365.0),
                          np.sin(2 * np.pi * dow / 7.0),
                          np.cos(2 * np.pi * dow / 7.0)],
                         axis=1).astype(np.float32)

    ctx = {
        "dates": dates, "n_days": n_days, "day_of": day_of,
        "unit_ids": unit_ids, "unit_index": unit_index, "n_st": n_st,
        "n_units": n_units, "is_station": is_station,
        "lat": lat, "lon": lon, "elev": elev, "elev_ok": elev_ok,
        "obs_raw": obs_raw, "met": met, "met_cols": met_cols,
        "u_mat": u_mat, "v_mat": v_mat, "wind_cols": (wind_u, wind_v),
        "graph": graph, "hrel": R_univ, "harmonics": harmonics,
        "row_unit": row_unit, "row_day": row_day,
        "vault": vault, "paths": paths,
        "stations_sha": hashlib.sha256(
            "|".join(station_ids.tolist()).encode()).hexdigest()[:16],
    }
    _say(f"context: {n_st:,} stations + {len(q_unit_ids)} query sites, "
         f"{n_days} days, {len(graph['edge_src']):,} edges, "
         f"met={met_cols}")
    return ctx


def _residual_vs_neighborhood(obs, nb_idx):
    """[n_st, n_days] residual of each station's raw obs vs the median of
    its graph neighbors' raw obs, NaN wherever either side is missing --
    the h_rel disagreement signal."""
    n_st, n_days = obs.shape
    pad = np.concatenate([obs, np.full((1, n_days), np.nan, obs.dtype)],
                         axis=0)
    nb = pad[nb_idx]                      # [n_st, K, n_days], -1 -> NaN row
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(nb, axis=1)
    return (obs - med).astype(np.float32)


def build_graph(lat, lon, elev, elev_ok, is_station, u_mat, v_mat,
                st_unit_ids, q_unit_ids, cfg):
    """Airsheds + directed in-edges (numpy, cached npz under CACHE_DIR).

    Airsheds: N_AIRSHEDS k-means clusters over (equirect-km coords,
    climatological mean u10/v10, elevation), z-scored, seeded, rows in
    sorted-station order (dtype-stable determinism). Edge rule: for every
    node, candidates are STATIONS ONLY (queries never emit K/V) that are
    either in the same airshed or pass the cross-airshed filter
    (d < 150 km AND |delta elev| < 500 m); edges go to the KNN_K nearest
    candidates. This reads DESIGN S7's two clauses as one bounded kNN --
    documented interpretation, in-degree <= 10 by construction.
    """
    n_units = len(lat)
    n_st = int(is_station.sum())
    lat0 = float(np.nanmean(lat[:n_st]))
    lon0 = float(np.nanmean(lon[:n_st]))
    x, y = _equirect_xy_km(lat, lon, lat0, lon0)

    key_src = "|".join([
        ",".join(st_unit_ids.tolist()), ",".join(q_unit_ids.tolist()),
        f"k{KNN_K}a{N_AIRSHEDS}c{CROSS_MAX_KM}e{CROSS_MAX_DELEV_M}",
        f"seed{config2.SEED}", str(cfg.get("quick", False)),
        str(cfg.get("variant", "full"))])
    key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
    cache = os.path.join(config2.CACHE_DIR, f"graph_t2_{n_st}_{key}.npz")
    if os.path.exists(cache) and os.environ.get("FORCE") != "1":
        with np.load(cache) as z:
            g = {k: z[k] for k in z.files}
        _say(f"graph cache hit: {cache}")
        g["nb_idx_st"] = g["nb_idx"][:n_st]
        return g

    # Climatological wind per node (mean over days; NaN -> 0 post z-score).
    if u_mat is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            uc = np.nanmean(u_mat, axis=1)
            vc = np.nanmean(v_mat, axis=1)
    else:
        uc = np.zeros(n_units)
        vc = np.zeros(n_units)

    feats = np.column_stack([x, y, uc, vc, elev])
    mu = np.nanmean(np.where(np.isfinite(feats[:n_st]), feats[:n_st], np.nan),
                    axis=0)
    sd = np.nanstd(np.where(np.isfinite(feats[:n_st]), feats[:n_st], np.nan),
                   axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
    z = (feats - mu) / sd
    z = np.where(np.isfinite(z), z, 0.0)

    k_air = int(min(N_AIRSHEDS, max(2, n_st // 3)))
    st_lab, cent = _kmeans(z[:n_st], k_air, seed=config2.SEED)
    airshed = np.empty(n_units, dtype=np.int64)
    airshed[:n_st] = st_lab
    if n_units > n_st:
        d2 = ((z[n_st:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
        airshed[n_st:] = d2.argmin(axis=1)

    # Candidate pool: CAND_POOL nearest stations per node.
    st_xy = np.column_stack([x[:n_st], y[:n_st]])
    all_xy = np.column_stack([x, y])
    kq = int(min(CAND_POOL, n_st))
    if HAS_SCIPY:
        tree = cKDTree(st_xy)
        dist, cand = tree.query(all_xy, k=kq)
        if kq == 1:
            dist, cand = dist[:, None], cand[:, None]
    else:
        d_all = np.sqrt(((all_xy[:, None, :] - st_xy[None, :, :]) ** 2)
                        .sum(axis=2))
        cand = np.argsort(d_all, axis=1)[:, :kq]
        dist = np.take_along_axis(d_all, cand, axis=1)

    src_l, dst_l = [], []
    for i in range(n_units):
        ci, di = cand[i], dist[i]
        not_self = ~((ci == i) & (i < n_st))
        ci, di = ci[not_self], di[not_self]
        same = airshed[ci] == airshed[i]
        d_elev = np.abs(elev[ci] - elev[i])
        # Missing elevation on either end degrades the filter to distance
        # only (elev_ok carries the honesty flag into the node features).
        elev_gate = np.where((elev_ok[ci] > 0) & (elev_ok[i] > 0),
                             d_elev < CROSS_MAX_DELEV_M, True)
        keep = same | ((di < CROSS_MAX_KM) & elev_gate)
        ci, di = ci[keep], di[keep]
        take = np.argsort(di, kind="stable")[:KNN_K]
        for s in ci[take]:
            src_l.append(int(s))
            dst_l.append(i)
    edge_src = np.asarray(src_l, dtype=np.int64)
    edge_dst = np.asarray(dst_l, dtype=np.int64)

    dkm = _haversine_km(lat[edge_src], lon[edge_src],
                        lat[edge_dst], lon[edge_dst])
    ex = x[edge_dst] - x[edge_src]
    ey = y[edge_dst] - y[edge_src]
    norm = np.sqrt(ex ** 2 + ey ** 2)
    norm = np.where(norm > 1e-9, norm, 1.0)
    g = {
        "edge_src": edge_src, "edge_dst": edge_dst,
        "dist_km": dkm.astype(np.float32),
        "ex": (ex / norm).astype(np.float32),
        "ey": (ey / norm).astype(np.float32),
        "delta_elev_km": ((elev[edge_dst] - elev[edge_src]) / 1000.0)
        .astype(np.float32),
        "same_airshed": (airshed[edge_src] == airshed[edge_dst])
        .astype(np.float32),
        "airshed": airshed,
        "nb_idx": _neighbor_lists(edge_src, edge_dst, n_units),
    }
    _save_npz_atomic(cache, **g)
    _say(f"graph built: {len(edge_src):,} edges, {k_air} airsheds "
         f"-> cached {cache}")
    g["nb_idx_st"] = g["nb_idx"][:n_st]
    return g


def _neighbor_lists(edge_src, edge_dst, n_units):
    """[n_units, KNN_K] in-neighbor station indices, -1 padded."""
    nb = np.full((n_units, KNN_K), -1, dtype=np.int64)
    fill = np.zeros(n_units, dtype=np.int64)
    for s, d in zip(edge_src, edge_dst):
        if fill[d] < KNN_K:
            nb[d, fill[d]] = s
            fill[d] += 1
    return nb


# -- Normalization stats (finite-only, v1 prefill-stats convention) ----------

def compute_norm_stats(ctx):
    """Finite-pixel-only mean/std (float64 accumulate) for obs (log1p
    space), met channels and h_rel residuals -- the v1
    _compute_norm_stats_prefill convention: stats BEFORE any masking, NaN
    never contributes. Stored in the checkpoint cfg for serve parity."""
    obs = ctx["obs_raw"]
    fin = np.isfinite(obs)
    lo = np.log1p(np.clip(obs[fin].astype(np.float64), 0.0, None))
    obs_mu = float(lo.mean()) if lo.size else 0.0
    obs_sd = float(lo.std()) if lo.size else 1.0
    obs_sd = max(obs_sd, 1e-6)

    met = ctx["met"]
    met_mu, met_sd = [], []
    for ci in range(met.shape[2]):
        v = met[:, :, ci]
        f = np.isfinite(v)
        if f.any():
            met_mu.append(float(v[f].astype(np.float64).mean()))
            met_sd.append(max(float(v[f].astype(np.float64).std()), 1e-6))
        else:
            met_mu.append(0.0)
            met_sd.append(1.0)

    r = ctx["hrel"]
    fr = np.isfinite(r)
    rel_sd = max(float(np.abs(r[fr].astype(np.float64)).mean()) * 1.2533,
                 1e-6) if fr.any() else 1.0   # ~std under normality
    return {"obs_mu": obs_mu, "obs_sd": obs_sd,
            "met_mu": met_mu, "met_sd": met_sd,
            "rel_sd": rel_sd,
            "elev_scale_km": 2.0}


def _norm_obs(x, stats):
    return (np.log1p(np.clip(x, 0.0, None)) - stats["obs_mu"]) / stats["obs_sd"]


# -- Model (defined only when torch is present) ------------------------------

if HAS_TORCH:

    def _segment_softmax(logits, seg, n_seg):
        """Softmax over edges grouped by destination node (fp32 for
        stability under AMP). Segments with zero edges are simply never
        indexed -- their nodes are handled by the has_nbrs mask."""
        logits = logits.float()
        m = torch.full((n_seg, logits.shape[1]), float("-inf"),
                       device=logits.device)
        m = m.scatter_reduce(0, seg.unsqueeze(1).expand_as(logits), logits,
                             reduce="amax", include_self=True)
        ex = torch.exp(logits - m[seg])
        den = torch.zeros((n_seg, logits.shape[1]), device=logits.device)
        den = den.index_add(0, seg, ex)
        return ex / den[seg].clamp_min(1e-12)

    class _EdgeAttention(nn.Module):
        """One shielded, edge-biased attention head bundle over an edge
        list. Only kv-eligible sources reach this module (edges are
        pre-filtered per forward), so shielding is structural."""

        def __init__(self, d, heads):
            super().__init__()
            self.h = heads
            self.dh = d // heads
            self.q = nn.Linear(d, d)
            self.k = nn.Linear(d, d)
            self.v = nn.Linear(d, d)
            self.o = nn.Linear(d, d)

        def forward(self, x, src, dst, ebias):
            n = x.shape[0]
            q = self.q(x).view(n, self.h, self.dh)
            k = self.k(x).view(n, self.h, self.dh)
            v = self.v(x).view(n, self.h, self.dh)
            logits = (q[dst] * k[src]).sum(-1) / math.sqrt(self.dh) + ebias
            a = _segment_softmax(logits, dst, n)          # [E, H] fp32
            msg = a.unsqueeze(-1) * v[src].float()        # [E, H, dh]
            out = torch.zeros(n, self.h, self.dh, device=x.device)
            out = out.index_add(0, dst, msg)
            return self.o(out.reshape(n, self.h * self.dh).to(x.dtype))

    class _Block(nn.Module):
        """Pre-LN transformer block: LN -> shielded attention -> residual;
        LN -> FFN -> residual (DESIGN S7 architecture)."""

        def __init__(self, d, heads, d_ff):
            super().__init__()
            self.ln1 = nn.LayerNorm(d)
            self.attn = _EdgeAttention(d, heads)
            self.ln2 = nn.LayerNorm(d)
            self.ffn = nn.Sequential(nn.Linear(d, d_ff), nn.SiLU(),
                                     nn.Linear(d_ff, d))

        def forward(self, x, src, dst, ebias):
            x = x + self.attn(self.ln1(x), src, dst, ebias)
            return x + self.ffn(self.ln2(x))

    class GraphResNet(nn.Module):
        """Masked graph-attention residual network (~1.5M params at the
        default dims; the constructor prints the exact count).

        Inputs per node: an OBS_WINDOW own-observation window (value +
        finite flag per day), met channels (value + finite flag), static
        scalars (elev, elev_ok, doy/dow harmonics) and the h_rel sequence.
        A masked or query node has its observation embedding replaced by
        the learned [MASK] vector and its h_rel by the learned null vector
        for the ENTIRE window -- full-window masking (DESIGN S7).
        """

        def __init__(self, n_met, d=D_MODEL, heads=N_HEADS,
                     layers=N_LAYERS, d_ff=D_FF):
            super().__init__()
            self.obs_mlp = nn.Sequential(
                nn.Linear(2 * OBS_WINDOW, D_OBS), nn.SiLU(),
                nn.Linear(D_OBS, D_OBS))
            self.mask_embed = nn.Parameter(torch.zeros(D_OBS))
            self.rel_gru = nn.GRU(input_size=2, hidden_size=D_REL,
                                  batch_first=True)
            self.null_rel = nn.Parameter(torch.zeros(D_REL))
            n_static = 2 + 4                      # elev, elev_ok, harmonics
            self.embed = nn.Sequential(
                nn.Linear(D_OBS + D_REL + 2 * n_met + n_static, d),
                nn.SiLU(), nn.Linear(d, d))
            self.edge_mlp = nn.Sequential(        # shared across layers
                nn.Linear(6, 64), nn.SiLU(), nn.Linear(64, heads))
            self.blocks = nn.ModuleList(
                [_Block(d, heads, d_ff) for _ in range(layers)])
            self.ln_out = nn.LayerNorm(d)
            self.head = nn.Linear(d, 4)           # mu, logvar, q05, q95
            n_par = sum(p.numel() for p in self.parameters())
            print(f"[aqnet2] graph_res: GraphResNet params = {n_par:,}",
                  flush=True)

        def forward(self, batch):
            x_obs = batch["x_obs"]
            masked = batch["masked"]
            e = self.obs_mlp(x_obs)
            e = torch.where(masked.unsqueeze(1),
                            self.mask_embed.to(e.dtype).expand_as(e), e)
            _, hn = self.rel_gru(batch["hrel_seq"])
            hr = hn.squeeze(0)
            hr = torch.where(masked.unsqueeze(1),
                             self.null_rel.to(hr.dtype).expand_as(hr), hr)
            h = self.embed(torch.cat(
                [e, hr, batch["x_met"], batch["x_static"]], dim=1))

            src, dst = batch["src"], batch["dst"]
            keep = batch["kv_ok"][src]
            src_k, dst_k = src[keep], dst[keep]
            ebias = self.edge_mlp(batch["efeat"][keep])
            for blk in self.blocks:
                h = blk(h, src_k, dst_k, ebias)

            deg = torch.zeros(h.shape[0], device=h.device)
            deg = deg.index_add(0, dst_k,
                                torch.ones_like(dst_k, dtype=torch.float32))
            out = self.head(self.ln_out(h))
            mu = out[:, 0]
            logvar = out[:, 1].clamp(-LOGVAR_CLAMP, LOGVAR_CLAMP)
            q05, q95 = out[:, 2], out[:, 3]
            return mu, logvar, q05, q95, deg > 0


# -- Training scaffolding (v1 models_deep idioms, extended per contract) -----

def _resolve_device(device="auto", quick=False):
    """v1 _resolve_device: auto -> cuda -> mps -> cpu; AMP CUDA-only.

    In FULL mode a CPU fallback is a hard error unless AQNET2_ALLOW_CPU=1:
    the Phoenix cu13x-wheel incident showed cuda_ok=False training silently
    on CPU for hours (review finding + observed 2026-08-04). Quick/smoke
    runs may use any device.
    """
    if not HAS_TORCH:
        raise RuntimeError("torch is required for T2 training/prediction "
                           "(pip install torch)")
    if device != "auto":
        dev = torch.device(device)
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        mps = getattr(torch.backends, "mps", None)
        dev = (torch.device("mps") if mps is not None and mps.is_available()
               else torch.device("cpu"))
    if (not quick and dev.type != "cuda"
            and os.environ.get("AQNET2_ALLOW_CPU") != "1"):
        raise SystemExit(
            f"[aqnet2] graph_res: resolved device is {dev.type!r} in FULL "
            "mode — a full training run must not silently fall back to CPU "
            "(check the torch CUDA wheel vs the node driver; cu126 works on "
            "Phoenix). Set AQNET2_ALLOW_CPU=1 to override deliberately.")
    _say(f"device: {dev.type}"
         + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda"
            else ""))
    return dev


def _make_scheduler(optimizer, epochs, warmup_epochs=WARMUP_EPOCHS):
    """v1 models_deep._make_scheduler: LinearLR warmup then cosine, stepped
    once per epoch."""
    sched = torch.optim.lr_scheduler
    warm = sched.LinearLR(optimizer,
                          start_factor=1.0 / max(warmup_epochs, 1),
                          end_factor=1.0, total_iters=warmup_epochs)
    if epochs > warmup_epochs:
        cos = sched.CosineAnnealingLR(optimizer,
                                      T_max=epochs - warmup_epochs)
        return sched.SequentialLR(optimizer, [warm, cos],
                                  milestones=[warmup_epochs])
    return warm


def _make_grad_scaler(enabled):
    """v1 models_deep._make_grad_scaler: torch.amp with legacy fallback."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda")
    import contextlib
    return contextlib.nullcontext()


def _rng_capture(np_rng):
    state = {"python": random.getstate(),
             "numpy": np_rng.bit_generator.state,
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _rng_restore(state, np_rng):
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np_rng.bit_generator.state = state["numpy"]
    if "torch" in state:
        torch.set_rng_state(torch.as_tensor(state["torch"],
                                            dtype=torch.uint8))
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except Exception as e:
            _say(f"cuda rng restore skipped ({e})")


def _ckpt_path(stem):
    return config2.artifact(f"{stem}_last.pt", "graph")


def save_checkpoint(stem, model, optimizer, scheduler, scaler, rng_state,
                    epoch, cfg, fold_id):
    """Atomic checkpoint per BUILD_NOTES contract #7: tmp + os.replace,
    exactly the frozen key set. v1 train_fusion_unet lacked optimizer/
    scheduler/RNG state and atomic writes -- this is the extension, not a
    copy."""
    path = _ckpt_path(stem)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng_state": rng_state,
        "epoch": int(epoch),
        "cfg": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "fold_id": fold_id,
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(stem):
    """Autodetect + load {stem}_last.pt (the --resume contract). Returns
    None when absent."""
    path = _ckpt_path(stem)
    if not os.path.exists(path):
        return None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    missing = [k for k in ("model", "optimizer", "scheduler", "scaler",
                           "rng_state", "epoch", "cfg", "fold_id")
               if k not in ck]
    if missing:
        raise RuntimeError(f"checkpoint {path} missing keys {missing} -- "
                           f"not a v2-contract checkpoint")
    _say(f"resume: loaded {path} (epoch {ck['epoch']})")
    return ck


# -- Batching ----------------------------------------------------------------

def _cal_obs_matrix(ctx, col):
    """Calibrated observation matrix [n_st, n_days] from pa_calibrated
    .parquet column `col` (fold-nested calibration enters ONLY here, at
    fine-tune/predict -- never at pretrain). Cached per column in ctx."""
    cache = ctx.setdefault("_cal_mats", {})
    if col in cache:
        return cache[col]
    if "_cal_df" not in ctx:
        path = ctx["paths"]["pa_calibrated"]
        if not os.path.exists(path):
            raise SystemExit(f"[aqnet2] graph_res: {path} missing -- "
                             f"fine-tune needs the calibrate stage")
        df = pd.read_parquet(path)
        df["sensor_id"] = df["sensor_id"].astype(str)
        df["date"] = _dates_ns(df["date"])
        pos = {u[3:]: i for i, u in
               enumerate(ctx["unit_ids"][:ctx["n_st"]].tolist())}
        df["_si"] = df["sensor_id"].map(pos)
        df["_di"] = df["date"].map(ctx["day_of"])
        df = df.dropna(subset=["_si", "_di"]).reset_index(drop=True)
        ctx["_cal_df"] = df
    df = ctx["_cal_df"]
    use = col
    if use not in df.columns:
        for fb in (use.rsplit("_", 1)[0], "pa_cal_full"):
            if fb in df.columns:
                _say(f"WARNING: {col} missing from pa_calibrated.parquet; "
                     f"using {fb}")
                use = fb
                break
        else:
            raise SystemExit(f"[aqnet2] graph_res: no usable calibration "
                             f"column for {col}")
    m = _fill_matrix(ctx["n_st"], ctx["n_days"],
                     df["_si"].to_numpy(dtype=np.int64),
                     df["_di"].to_numpy(dtype=np.int64),
                     df[use].to_numpy(dtype=np.float32))
    cache[col] = m
    return m


def _build_batch(ctx, stats, day_idx, obs_st, masked_units_by_day, device):
    """Tensors for a batch of whole day-graphs (batch = days).

    obs_st: [n_st, n_days] observation source (raw at pretrain, fold-nested
    calibrated at fine-tune). masked_units_by_day: {day -> int array of
    unit indices task-masked that day}. Query (AQS) nodes are ALWAYS
    masked; stations with no day-t observation keep their own window but
    never emit K/V (shielded attention).
    """
    n_st, n_units = ctx["n_st"], ctx["n_units"]
    n_days = ctx["n_days"]
    B = len(day_idx)
    N = n_units
    is_st = ctx["is_station"]
    g = ctx["graph"]
    E = len(g["edge_src"])

    if "_obs_pad_id" not in ctx or ctx["_obs_pad_id"] != id(obs_st):
        ctx["_obs_pad"] = np.concatenate(
            [np.full((n_st, OBS_WINDOW - 1), np.nan, np.float32), obs_st],
            axis=1)
        ctx["_obs_pad_id"] = id(obs_st)
    obs_pad = ctx["_obs_pad"]

    x_obs = np.zeros((B, N, 2 * OBS_WINDOW), dtype=np.float32)
    x_met = np.zeros((B, N, 2 * ctx["met"].shape[2]), dtype=np.float32)
    x_static = np.zeros((B, N, 6), dtype=np.float32)
    hrel_seq = np.zeros((B, N, HREL_WINDOW, 2), dtype=np.float32)
    masked = np.zeros((B, N), dtype=bool)
    kv_ok = np.zeros((B, N), dtype=bool)
    efeat = np.zeros((B, E, 6), dtype=np.float32)

    elev_n = (ctx["elev"] / (stats["elev_scale_km"] * 1000.0)
              ).astype(np.float32)
    src, dst = g["edge_src"], g["edge_dst"]
    e_logd = (np.log1p(g["dist_km"]) / math.log1p(CROSS_MAX_KM)
              ).astype(np.float32)

    for b, t in enumerate(day_idx):
        w = obs_pad[:, t:t + OBS_WINDOW]              # oldest -> day t
        fin = np.isfinite(w)
        wn = np.where(fin, _norm_obs(w, stats), 0.0)
        x_obs[b, :n_st, :OBS_WINDOW] = wn
        x_obs[b, :n_st, OBS_WINDOW:] = fin.astype(np.float32)

        mt = ctx["met"][:, t, :]
        mfin = np.isfinite(mt)
        mn = np.where(
            mfin,
            (mt - np.asarray(stats["met_mu"], np.float32))
            / np.asarray(stats["met_sd"], np.float32), 0.0)
        M = mt.shape[1]
        x_met[b, :, :M] = mn
        x_met[b, :, M:] = mfin.astype(np.float32)

        x_static[b, :, 0] = elev_n
        x_static[b, :, 1] = ctx["elev_ok"]
        x_static[b, :, 2:6] = ctx["harmonics"][t]

        idx = hrel_day_indices(t)
        valid = idx >= 0
        seq = np.full((N, HREL_WINDOW), np.nan, np.float32)
        if valid.any():
            seq[:, valid] = ctx["hrel"][:, idx[valid]]
        sfin = np.isfinite(seq)
        hrel_seq[b, :, :, 0] = np.where(sfin, seq / stats["rel_sd"], 0.0)
        hrel_seq[b, :, :, 1] = sfin.astype(np.float32)

        tm = masked_units_by_day.get(t)
        day_masked = np.zeros(N, dtype=bool)
        day_masked[~is_st] = True                     # queries always masked
        if tm is not None and len(tm):
            day_masked[np.asarray(tm, dtype=np.int64)] = True
        masked[b] = day_masked
        obs_t_fin = np.zeros(N, dtype=bool)
        obs_t_fin[:n_st] = np.isfinite(obs_st[:, t])
        kv_ok[b] = is_st & obs_t_fin & ~day_masked

        efeat[b, :, 0] = e_logd
        if ctx["u_mat"] is not None:
            u = ctx["u_mat"][dst, t]
            v = ctx["v_mat"][dst, t]
            spd = np.sqrt(u ** 2 + v ** 2)
            wok = np.isfinite(spd) & (spd > 1e-9)
            align = np.where(wok,
                             (g["ex"] * u + g["ey"] * v)
                             / np.where(wok, spd, 1.0), 0.0)
            efeat[b, :, 1] = np.where(wok, align, 0.0)
            efeat[b, :, 2] = np.where(np.isfinite(spd), spd / 10.0, 0.0)
            efeat[b, :, 5] = wok.astype(np.float32)
        efeat[b, :, 3] = g["delta_elev_km"] / 0.5
        efeat[b, :, 4] = g["same_airshed"]

    offs = (np.arange(B, dtype=np.int64) * N)[:, None]
    src_b = (src[None, :] + offs).ravel()
    dst_b = (dst[None, :] + offs).ravel()

    def T(a, dt=torch.float32):
        return torch.as_tensor(a, dtype=dt, device=device)

    return {
        "x_obs": T(x_obs.reshape(B * N, -1)),
        "x_met": T(x_met.reshape(B * N, -1)),
        "x_static": T(x_static.reshape(B * N, -1)),
        "hrel_seq": T(hrel_seq.reshape(B * N, HREL_WINDOW, 2)),
        "masked": T(masked.reshape(-1), torch.bool),
        "kv_ok": T(kv_ok.reshape(-1), torch.bool),
        "src": T(src_b, torch.int64),
        "dst": T(dst_b, torch.int64),
        "efeat": T(efeat.reshape(B * E, 6)),
        "B": B, "N": N,
    }


# -- Losses ------------------------------------------------------------------

def _gaussian_nll(mu, logvar, y):
    return 0.5 * (logvar + (y - mu) ** 2 * torch.exp(-logvar))


def _pinball(q, y, tau):
    d = y - q
    return torch.maximum(tau * d, (tau - 1.0) * d)


def _head_loss(mu, logvar, q05, q95, y, w=None, huber=False):
    """NLL (+ optional Huber) + 0.25 * pinball(q05, q95), precision-weighted
    (weights normalized to mean 1 so the loss scale is comparable across
    tasks). NaN targets never reach here -- callers pre-mask."""
    per = _gaussian_nll(mu, logvar, y)
    if huber:
        per = per + F.huber_loss(mu, y, reduction="none",
                                 delta=HUBER_DELTA_NORM)
    per = per + PINBALL_WEIGHT * (_pinball(q05, y, 0.05)
                                  + _pinball(q95, y, 0.95))
    if w is not None:
        per = per * (w / w.mean().clamp_min(1e-12))
    return per.mean()


# -- Pretrain mask sampling --------------------------------------------------

def _sample_pretrain_mask(ctx, t, rng, obs_st):
    """20-40% of day-t observed stations: half uniform, half a structured
    50-150 km ball around a random observed center (teaches sigma to widen
    off-support, DESIGN S7). Deterministic given the rng stream; candidate
    arrays come from flatnonzero over the SORTED station axis."""
    obs_idx = np.flatnonzero(np.isfinite(obs_st[:, t]))
    if len(obs_idx) < 3:
        return obs_idx[:0]
    frac = rng.uniform(MASK_FRAC_LO, MASK_FRAC_HI)
    n_mask = max(1, int(round(frac * len(obs_idx))))
    n_uni = n_mask // 2
    chosen = set(rng.choice(obs_idx, size=n_uni, replace=False).tolist()
                 if n_uni else [])
    n_str = n_mask - len(chosen)
    center = int(rng.choice(obs_idx))
    r = rng.uniform(BALL_KM_LO, BALL_KM_HI)
    d = _haversine_km(ctx["lat"][obs_idx], ctx["lon"][obs_idx],
                      ctx["lat"][center], ctx["lon"][center])
    ball = obs_idx[np.argsort(d, kind="stable")]
    ball = ball[np.sort(d, kind="stable") <= r]
    for s in ball[:n_str]:
        chosen.add(int(s))
    short = n_mask - len(chosen)
    if short > 0:
        rest = np.setdiff1d(obs_idx, np.fromiter(chosen, dtype=np.int64))
        if len(rest):
            extra = rng.choice(rest, size=min(short, len(rest)),
                               replace=False)
            chosen.update(int(s) for s in extra)
    return np.sort(np.fromiter(chosen, dtype=np.int64))


# -- cfg ---------------------------------------------------------------------

def make_cfg(stage="graphpre", quick=False, variant="full", resume=False,
             device="auto", epochs=None, study=False):
    """Plain-dict run config. Keys starting with '_' are runtime context
    (never persisted into checkpoints)."""
    if epochs is None:
        if quick:
            epochs = QUICK_EPOCHS
        else:
            epochs = PRETRAIN_EPOCHS if stage == "graphpre" \
                else FINETUNE_EPOCHS
    return {"stage": stage, "quick": bool(quick), "variant": variant,
            "resume": bool(resume), "device": device, "epochs": int(epochs),
            "study": bool(study), "seed": int(config2.SEED)}


def _variant_suffix(cfg):
    return "_temporal" if cfg.get("variant") == "temporal" else ""


def _get_ctx(cfg):
    if cfg.get("_ctx") is not None:
        return cfg["_ctx"]
    paths = _default_paths()
    if not os.path.exists(paths["frame"]):
        raise SystemExit(f"[aqnet2] graph_res: {paths['frame']} missing -- "
                         f"run the features stage first")
    frame = pd.read_parquet(paths["frame"])
    folds = _load_folds_dict(paths["folds"], n_rows=len(frame))
    ctx = build_context(frame, folds, cfg)
    ctx["frame"] = frame
    ctx["folds"] = folds
    cfg["_ctx"] = ctx
    return ctx


def _unit_map_from_rows(ctx, row_values, name):
    """Row-level fold array -> unit-level map [n_units], with the
    unit-grouped-consistency assert (a unit split across folds would break
    the epistemic-ensemble semantics)."""
    row_unit = ctx["row_unit"]
    vals = np.asarray(row_values, dtype=np.int64)
    valid = row_unit >= 0
    df = pd.DataFrame({"u": row_unit[valid], "f": vals[valid]})
    bad = int((df.groupby("u")["f"].nunique() > 1).sum())
    assert bad == 0, (f"{name}: {bad} units have inconsistent fold "
                      f"assignments across rows")
    m = np.full(ctx["n_units"], -1, dtype=np.int64)
    m[row_unit[valid]] = vals[valid]
    return m


def _t1_residual(ctx, k):
    """r1 = y - T1 oof for outer fold k as (r1_mat, w_mat) [n_units,
    n_days]. Prefers a per-fold oof_f{k} array when skeleton wrote one;
    falls back to the cross-fit 'oof' (which IS the fold-k model's
    prediction on fold-k rows by construction)."""
    key = ("_r1", int(k))
    if key in ctx:
        return ctx[key]
    if "_t1z" not in ctx:
        p = ctx["paths"]["tier1"]
        if not os.path.exists(p):
            raise SystemExit(f"[aqnet2] graph_res: {p} missing -- run the "
                             f"skeleton stage first")
        with np.load(p, allow_pickle=False) as z:
            ctx["_t1z"] = {kk: z[kk] for kk in z.files}
    z = ctx["_t1z"]
    frame = ctx["frame"]
    base = z.get(f"oof_f{k}", z.get("oof"))
    if base is None or len(base) != len(frame):
        raise SystemExit("[aqnet2] graph_res: oof_tier1.npz misaligned with "
                         "the frame")
    y = frame["y"].to_numpy(dtype=np.float64)
    r1 = y - np.asarray(base, dtype=np.float64)
    w = frame["w"].to_numpy(dtype=np.float64)
    ru, rd = ctx["row_unit"], ctx["row_day"]
    ok = (ru >= 0) & (rd >= 0) & np.isfinite(r1)
    r1_mat = _fill_matrix(ctx["n_units"], ctx["n_days"], ru[ok], rd[ok],
                          r1[ok], dtype=np.float32)
    w_mat = _fill_matrix(ctx["n_units"], ctx["n_days"], ru[ok], rd[ok],
                         np.where(np.isfinite(w[ok]), w[ok], np.nan),
                         dtype=np.float32)
    ctx[key] = (r1_mat, w_mat)
    return ctx[key]


def _r1_norm_stats(ctx, ks):
    """Robust residual scale over the outer folds in play (stored in every
    fine-tune checkpoint cfg; predictions un-normalize through it)."""
    vals = []
    for k in ks:
        r1_mat, _ = _t1_residual(ctx, k)
        f = np.isfinite(r1_mat)
        if f.any():
            vals.append(r1_mat[f].astype(np.float64))
    v = np.concatenate(vals) if vals else np.zeros(1)
    mu = float(np.median(v))
    sd = float(1.4826 * np.median(np.abs(v - mu)))
    return mu, max(sd, 1e-3)


def _train_day_list(ctx, obs_st, min_obs=5):
    """Days with at least min_obs observed stations (sorted day indices --
    shuffles downstream operate on this sorted array, dtype-stable)."""
    counts = np.isfinite(obs_st).sum(axis=0)
    days = np.flatnonzero(counts >= min_obs)
    if len(days) == 0:
        _say(f"WARNING: no day has >= {min_obs} observed stations; "
             f"falling back to any-observation days")
        days = np.flatnonzero(counts > 0)
    return days.astype(np.int64)


def _fit_setup(cfg, ctx, model, lr, epochs):
    device = _resolve_device(cfg.get("device", "auto"), quick=bool(cfg.get("quick")))
    torch.manual_seed(cfg["seed"])
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=WEIGHT_DECAY)
    sched = _make_scheduler(opt, epochs)
    use_amp = device.type == "cuda"
    scaler = _make_grad_scaler(use_amp)
    return device, model, opt, sched, scaler, use_amp


def _step(model, batch, loss_fn, opt, scaler, use_amp, device):
    opt.zero_grad(set_to_none=True)
    with _autocast(device):
        out = model(batch)
        loss = loss_fn(out, batch)
    if loss is None:
        return None
    if use_amp:
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
    return float(loss.detach().cpu())


# -- Public API: pretrain ----------------------------------------------------

def pretrain(cfg):
    """Stage `graphpre`: masked-station reconstruction of RAW PA values.

    No FRM-derived inputs, no calibration, no T0 (DESIGN S7). Per training
    day: 20-40% of observed stations masked (half uniform, half structured
    ball), loss = Huber + Gaussian NLL + 0.25*pinball on the masked
    stations, in normalized log1p space. `--variant temporal` restricts to
    days < TEMPORAL_CUTOFF (the sole basis for temporal-holdout claims).
    cfg['_exclude_station_idx'] (leakage study) removes stations from both
    the loss and the K/V pool. Returns the last-checkpoint path.
    """
    ctx = _get_ctx(cfg)
    stem = cfg.get("_stem") or ("graphpre" + _variant_suffix(cfg))
    epochs = int(cfg["epochs"])
    stats = compute_norm_stats(ctx)
    obs_st = ctx["obs_raw"]
    excl = np.asarray(cfg.get("_exclude_station_idx", []), dtype=np.int64)

    run_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    run_cfg.update({"norm_stats": stats, "met_cols": ctx["met_cols"],
                    "stations_sha": ctx["stations_sha"], "stem": stem,
                    "n_met": ctx["met"].shape[2],
                    "obs_source": "pa_raw"})

    torch.manual_seed(cfg["seed"])       # seeded BEFORE weight init
    model = GraphResNet(n_met=ctx["met"].shape[2])
    device, model, opt, sched, scaler, use_amp = _fit_setup(
        cfg, ctx, model, LR_PRETRAIN, epochs)
    rng = np.random.default_rng(cfg["seed"])

    start_epoch = 0
    if cfg.get("resume"):
        ck = load_checkpoint(stem)
        if ck is not None:
            model.load_state_dict(ck["model"])
            if ck["optimizer"]:
                opt.load_state_dict(ck["optimizer"])
            if ck["scheduler"]:
                sched.load_state_dict(ck["scheduler"])
            if ck["scaler"] and use_amp:
                scaler.load_state_dict(ck["scaler"])
            _rng_restore(ck["rng_state"], rng)
            start_epoch = int(ck["epoch"]) + 1

    days = _train_day_list(ctx, obs_st)
    bs = PRETRAIN_BATCH_DAYS
    _say(f"pretrain[{stem}]: {len(days)} day-graphs, epochs "
         f"{start_epoch}..{epochs - 1}, batch {bs} days, device {device}")

    def loss_fn(out, batch):
        mu, logvar, q05, q95, has_nbrs = out
        sel = batch["_loss_mask"]
        tgt = batch["_target"]
        pick = sel & has_nbrs & torch.isfinite(tgt)
        if not bool(pick.any()):
            return None
        return _head_loss(mu[pick], logvar[pick], q05[pick], q95[pick],
                          tgt[pick], huber=True)

    last_save = time.time()
    path = _ckpt_path(stem)
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        order = days.copy()
        rng.shuffle(order)                 # rng over the SORTED day array
        tot, nb = 0.0, 0
        for s in range(0, len(order), bs):
            chunk = order[s:s + bs]
            masks = {}
            for t in chunk:
                m = _sample_pretrain_mask(ctx, int(t), rng, obs_st)
                if len(excl):
                    m = np.union1d(m, excl)
                masks[int(t)] = m
            batch = _build_batch(ctx, stats, [int(t) for t in chunk],
                                 obs_st, masks, device)
            B, N = batch["B"], batch["N"]
            lm = np.zeros((B, N), dtype=bool)
            tg = np.zeros((B, N), dtype=np.float32)
            for b, t in enumerate(chunk):
                m = np.setdiff1d(masks[int(t)], excl)
                lm[b, m] = True
                col = np.where(np.isfinite(obs_st[:, int(t)]),
                               _norm_obs(obs_st[:, int(t)], stats), np.nan)
                tg[b, :ctx["n_st"]] = col
            batch["_loss_mask"] = torch.as_tensor(lm.reshape(-1),
                                                  device=device)
            batch["_target"] = torch.as_tensor(tg.reshape(-1), device=device)
            val = _step(model, batch, loss_fn, opt, scaler, use_amp, device)
            if val is not None:
                tot += val
                nb += 1
            if time.time() - last_save > CKPT_EVERY_SEC:
                save_checkpoint(stem, model, opt, sched, scaler,
                                _rng_capture(rng), epoch - 1, run_cfg, None)
                last_save = time.time()
        sched.step()
        save_checkpoint(stem, model, opt, sched, scaler, _rng_capture(rng),
                        epoch, run_cfg, None)
        last_save = time.time()
        _say(f"pretrain[{stem}] epoch {epoch + 1}/{epochs} "
             f"loss {tot / max(nb, 1):.4f} "
             f"lr {opt.param_groups[0]['lr']:.2e} "
             f"({time.time() - t0:.0f}s)")

    marker = config2.artifact(f"{stem}_marker.json", "graph")
    _write_json_atomic({"stage": "graphpre", "stem": stem,
                        "variant": cfg.get("variant"), "epochs": epochs,
                        "quick": cfg.get("quick", False),
                        "completed": True, "ckpt": path,
                        "stations_sha": ctx["stations_sha"],
                        "finished_utc": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, marker)
    _say(f"pretrain[{stem}] done -> {path}")
    return path


# -- Public API: finetune ----------------------------------------------------

def finetune(cfg, fold):
    """Stage `graphres`, one (outer k, inner j) member of the
    fold-assignment epistemic ensemble. fold = (k, j).

    Target r1 = y - T1_oof (oof_f{k} when skeleton wrote it). Inputs use
    the fold-nested calibration column pa_cal_f{k}_{j}. Loss units are the
    inner TRAIN folds (inner != j, != -1); held-out inner-j units and
    outer-fold-k sites never contribute loss. Per batch the loss is
    0.8 * L(AQS train queries) + 0.2 * L(masked train PA stations) -- the
    expectation form of the 4:1 mask-the-AQS-site task oversampling
    (DESIGN S7), lower-variance than literal task sampling and identical in
    expectation. Precision weights w come from the frame. Idempotent: an
    existing finished checkpoint short-circuits unless FORCE=1.
    """
    k, j = int(fold[0]), int(fold[1])
    ctx = _get_ctx(cfg)
    folds = ctx["folds"]
    suffix = _variant_suffix(cfg)
    stem = cfg.get("_stem") or f"graphres{suffix}_f{k}_{j}"
    epochs = int(cfg["epochs"])

    ck0 = load_checkpoint(stem) if (cfg.get("resume")
                                    or os.environ.get("FORCE") != "1") \
        else None
    if (ck0 is not None and int(ck0["epoch"]) >= epochs - 1
            and os.environ.get("FORCE") != "1"):
        _say(f"finetune[{stem}]: already finished (epoch {ck0['epoch']}) "
             f"-- skip (FORCE=1 to redo)")
        return _ckpt_path(stem)

    # Init from the (variant-matched) pretrain checkpoint.
    init = cfg.get("_init_stem") or ("graphpre" + suffix)
    pre = load_checkpoint(init)
    if pre is None:
        raise SystemExit(f"[aqnet2] graph_res: pretrain checkpoint "
                         f"{_ckpt_path(init)} missing -- run graphpre first")
    stats = pre["cfg"]["norm_stats"]
    assert pre["cfg"].get("stations_sha") == ctx["stations_sha"], (
        "pretrain checkpoint station universe differs from the current "
        "context -- stale graphpre artifact (FORCE=1 the graphpre stage)")

    r1_mat, w_mat = _t1_residual(ctx, k)
    r_mu, r_sd = _r1_norm_stats(ctx, [k])
    obs_st = _cal_obs_matrix(ctx, f"pa_cal_f{k}_{j}")

    inner = _unit_map_from_rows(ctx, folds["inner_fold"][str(k)],
                                f"inner_fold[{k}]")
    outer = _unit_map_from_rows(ctx, folds["outer_fold"], "outer_fold")
    is_st = ctx["is_station"]
    train_unit = (inner >= 0) & (inner != j) & (outer != k)
    train_pa = train_unit & is_st
    train_aqs = train_unit & ~is_st
    heldout = inner == j
    _say(f"finetune[{stem}]: {int(train_pa.sum())} train PA units, "
         f"{int(train_aqs.sum())} train AQS units, "
         f"{int(heldout.sum())} held-out inner-{j} units")

    run_cfg = {kk: v for kk, v in cfg.items() if not kk.startswith("_")}
    run_cfg.update({"norm_stats": stats, "met_cols": ctx["met_cols"],
                    "stations_sha": ctx["stations_sha"], "stem": stem,
                    "n_met": ctx["met"].shape[2],
                    "cal_col": f"pa_cal_f{k}_{j}",
                    "r1_mu": r_mu, "r1_sd": r_sd,
                    "pretrain_stem": init})

    model = GraphResNet(n_met=ctx["met"].shape[2])
    model.load_state_dict(pre["model"])
    device, model, opt, sched, scaler, use_amp = _fit_setup(
        cfg, ctx, model, LR_FINETUNE, epochs)
    rng = np.random.default_rng(cfg["seed"] + 1000 * k + j)

    start_epoch = 0
    if ck0 is not None and cfg.get("resume"):
        model.load_state_dict(ck0["model"])
        if ck0["optimizer"]:
            opt.load_state_dict(ck0["optimizer"])
        if ck0["scheduler"]:
            sched.load_state_dict(ck0["scheduler"])
        if ck0["scaler"] and use_amp:
            scaler.load_state_dict(ck0["scaler"])
        _rng_restore(ck0["rng_state"], rng)
        start_epoch = int(ck0["epoch"]) + 1

    r1n = np.where(np.isfinite(r1_mat), (r1_mat - r_mu) / r_sd, np.nan)
    days = _train_day_list(ctx, obs_st)
    # Only days with at least one train-unit target are trainable.
    has_tgt = np.isfinite(r1n[train_unit][:, days]).any(axis=0)
    days = days[has_tgt]
    if len(days) == 0:
        _say(f"WARNING: finetune[{stem}] has no trainable days in the "
             f"window -- writing an untrained checkpoint (quick smoke only)")
    bs = FINETUNE_BATCH_DAYS

    def loss_fn(out, batch):
        mu, logvar, q05, q95, has_nbrs = out
        tgt, wts = batch["_target"], batch["_w"]
        parts = []
        for sel, wgt in ((batch["_aqs_mask"], AQS_TASK_WEIGHT),
                         (batch["_pa_mask"], 1.0 - AQS_TASK_WEIGHT)):
            pick = sel & has_nbrs & torch.isfinite(tgt)
            if bool(pick.any()):
                parts.append(wgt * _head_loss(
                    mu[pick], logvar[pick], q05[pick], q95[pick],
                    tgt[pick], w=wts[pick], huber=False))
        if not parts:
            return None
        return sum(parts)

    last_save = time.time()
    path = _ckpt_path(stem)
    fold_id = f"f{k}_{j}"
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        order = days.copy()
        rng.shuffle(order)
        n_take = max(1, int(round(len(order) * FINETUNE_DAY_FRAC))) \
            if not cfg.get("quick") else len(order)
        order = order[:n_take]
        tot, nb = 0.0, 0
        for s in range(0, len(order), bs):
            chunk = [int(t) for t in order[s:s + bs]]
            masks = {}
            pa_masked = {}
            for t in chunk:
                cand = np.flatnonzero(train_pa
                                      & np.isfinite(r1n[:, t])
                                      & np.pad(np.isfinite(obs_st[:, t]),
                                               (0, ctx["n_units"]
                                                - ctx["n_st"])))
                n_m = max(1, int(round(FINETUNE_PA_MASK_FRAC * len(cand)))) \
                    if len(cand) else 0
                m = (rng.choice(cand, size=n_m, replace=False)
                     if n_m else cand[:0])
                pa_masked[t] = np.sort(m)
                masks[t] = pa_masked[t]
            batch = _build_batch(ctx, stats, chunk, obs_st, masks, device)
            B, N = batch["B"], batch["N"]
            aqs_m = np.zeros((B, N), dtype=bool)
            pa_m = np.zeros((B, N), dtype=bool)
            tg = np.full((B, N), np.nan, dtype=np.float32)
            ww = np.ones((B, N), dtype=np.float32)
            for b, t in enumerate(chunk):
                aqs_m[b] = train_aqs
                pa_m[b, pa_masked[t]] = True
                tg[b] = r1n[:, t]
                wcol = w_mat[:, t]
                ww[b] = np.where(np.isfinite(wcol), wcol, np.nan)
            batch["_aqs_mask"] = torch.as_tensor(aqs_m.reshape(-1),
                                                 device=device)
            batch["_pa_mask"] = torch.as_tensor(pa_m.reshape(-1),
                                                device=device)
            tgt = np.where(np.isfinite(ww), tg, np.nan)   # NaN w -> dropped
            batch["_target"] = torch.as_tensor(tgt.reshape(-1),
                                               device=device)
            batch["_w"] = torch.as_tensor(
                np.where(np.isfinite(ww), ww, 1.0).reshape(-1),
                device=device)
            val = _step(model, batch, loss_fn, opt, scaler, use_amp, device)
            if val is not None:
                tot += val
                nb += 1
            if time.time() - last_save > CKPT_EVERY_SEC:
                save_checkpoint(stem, model, opt, sched, scaler,
                                _rng_capture(rng), epoch - 1, run_cfg,
                                fold_id)
                last_save = time.time()
        sched.step()
        save_checkpoint(stem, model, opt, sched, scaler, _rng_capture(rng),
                        epoch, run_cfg, fold_id)
        last_save = time.time()
        _say(f"finetune[{stem}] epoch {epoch + 1}/{epochs} "
             f"loss {tot / max(nb, 1):.4f} ({time.time() - t0:.0f}s)")

    if cfg.get("_eval_heldout"):
        r2 = _eval_heldout_r2(cfg, ctx, model, device, stats, run_cfg,
                              obs_st, r1n, w_mat, heldout, inner, j)
        _say(f"finetune[{stem}] held-out inner-{j} weighted R^2 = {r2:.4f}")
        run_cfg["heldout_r2"] = r2
        save_checkpoint(stem, model, opt, sched, scaler, _rng_capture(rng),
                        epochs - 1, run_cfg, fold_id)
    _say(f"finetune[{stem}] done -> {path}")
    return path


def _eval_heldout_r2(cfg, ctx, model, device, stats, run_cfg, obs_st, r1n,
                     w_mat, heldout, inner, j):
    """Weighted R^2 of normalized r1 at held-out inner-j units (masked as
    queries, exactly the OOF protocol). Diagnostic + leakage study metric."""
    model.eval()
    days = _train_day_list(ctx, obs_st)
    ho_idx = np.flatnonzero(heldout)
    ys, ps, ws = [], [], []
    with torch.no_grad():
        for s in range(0, len(days), PREDICT_BATCH_DAYS):
            chunk = [int(t) for t in days[s:s + PREDICT_BATCH_DAYS]]
            masks = {t: ho_idx[ho_idx < ctx["n_st"]] for t in chunk}
            batch = _build_batch(ctx, stats, chunk, obs_st, masks, device)
            with _autocast(device):
                mu, logvar, q05, q95, has_nbrs = model(batch)
            N = batch["N"]
            mu = mu.float().cpu().numpy().reshape(len(chunk), N)
            hn = has_nbrs.cpu().numpy().reshape(len(chunk), N)
            for b, t in enumerate(chunk):
                tgt = r1n[ho_idx, t]
                w = w_mat[ho_idx, t]
                fin = np.isfinite(tgt) & np.isfinite(w) & hn[b, ho_idx]
                if fin.any():
                    ys.append(tgt[fin])
                    ps.append(mu[b, ho_idx[fin]])
                    ws.append(w[fin])
    model.train()
    if not ys:
        return float("nan")
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    w = np.concatenate(ws)
    ybar = np.average(y, weights=w)
    ss_res = np.average((y - p) ** 2, weights=w)
    ss_tot = np.average((y - ybar) ** 2, weights=w)
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


# -- Public API: predict_oof -------------------------------------------------

def _normalize_ckpts(ckpts):
    """Accept {(k, j): path} or {'f0_1': path} / {'graphres_f0_1': path}."""
    out = {}
    for key, p in ckpts.items():
        if isinstance(key, tuple):
            k, j = int(key[0]), int(key[1])
        else:
            tail = str(key).rsplit("f", 1)[-1]
            k_s, j_s = tail.split("_")
            k, j = int(k_s), int(j_s)
        out[(k, j)] = p
    return out


def predict_oof(frame, folds, ckpts, cfg=None, dest=None):
    """Cross-fit T2 OOF over every frame row -> oof_tier2.npz
    {oof_r f8 (NaN where unavailable), sigma f8, avail u1, pattern_id i1}.

    Ensemble semantics (the fold-assignment epistemic ensemble, DESIGN S7):
      AQS row in outer fold k  -> members (k, j) for all inner j (none saw
                                  fold-k sites);
      PA row                   -> per outer k, member (k, j_k) where j_k is
                                  the unit's inner fold under k (that model
                                  held the unit out); units folds2 excluded
                                  everywhere (j_k = -1) average the four
                                  fold-k members (all are OOF for them).
    A row is available only if EVERY member produced a finite prediction
    with a non-empty shielded neighborhood. avail = 0 (and oof_r = NaN)
    additionally where nbr_pacal_avail_100km == 0 (BUILD_NOTES #5 hard
    zero) and on vault rows (sites or the vault period -- never touched).
    sigma = sqrt(mean member sigma^2 + across-member spread) (quadrature).
    pattern_id: 1 = PA neighbors within 50 km, 2 = only 50-100 km,
    0 = unavailable.
    """
    if not HAS_TORCH:
        raise RuntimeError("torch is required for predict_oof")
    cfg = dict(cfg or make_cfg("graphres"))
    ctx = cfg.get("_ctx")
    if ctx is None:
        ctx = build_context(frame, folds, cfg)
        ctx["frame"] = frame
        ctx["folds"] = folds
        cfg["_ctx"] = ctx
    suffix = _variant_suffix(cfg)
    dest = dest or config2.artifact(f"oof_tier2{suffix}.npz")
    device = _resolve_device(cfg.get("device", "auto"), quick=bool(cfg.get("quick")))
    cmap = _normalize_ckpts(ckpts)
    n = len(frame)
    ru, rd = ctx["row_unit"], ctx["row_day"]
    is_pa_row = (frame["unit_type"].astype(str) == "pa").to_numpy()
    valid = (ru >= 0) & (rd >= 0)

    ks = sorted({k for k, _ in cmap})
    js_of = {k: sorted(j for kk, j in cmap if kk == k) for k in ks}
    outer_row = np.asarray(folds["outer_fold"], dtype=np.int64)
    inner_row = {k: np.asarray(folds["inner_fold"][str(k)], dtype=np.int64)
                 for k in ks}
    inner_unit = {k: _unit_map_from_rows(ctx, inner_row[k],
                                         f"inner_fold[{k}]") for k in ks}

    mu_m, sg_m, ok_m = {}, {}, {}
    for (k, j), path in sorted(cmap.items()):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ccfg = ck["cfg"]
        assert ccfg.get("stations_sha") == ctx["stations_sha"], (
            f"checkpoint {path} was trained on a different station "
            f"universe -- rebuild (FORCE=1)")
        stats = ccfg["norm_stats"]
        r_mu = float(ccfg.get("r1_mu", 0.0))
        r_sd = float(ccfg.get("r1_sd", 1.0))
        model = GraphResNet(n_met=int(ccfg["n_met"]))
        model.load_state_dict(ck["model"])
        model = model.to(device)
        model.eval()
        obs_st = _cal_obs_matrix(ctx, ccfg.get("cal_col",
                                               f"pa_cal_f{k}_{j}"))

        sel = valid & (((~is_pa_row) & (outer_row == k))
                       | (is_pa_row & np.isin(inner_row[k], (j, -1))))
        rows = np.flatnonzero(sel)
        mu_a = np.full(n, np.nan)
        sg_a = np.full(n, np.nan)
        ok_a = np.zeros(n, dtype=bool)
        if len(rows):
            by_day = {}
            for r in rows:
                by_day.setdefault(int(rd[r]), []).append(int(r))
            day_list = sorted(by_day)
            hold_pa = np.flatnonzero((inner_unit[k] == j)
                                     & ctx["is_station"])
            with torch.no_grad():
                for s in range(0, len(day_list), PREDICT_BATCH_DAYS):
                    chunk = day_list[s:s + PREDICT_BATCH_DAYS]
                    masks = {}
                    for t in chunk:
                        qs = np.asarray([ru[r] for r in by_day[t]],
                                        dtype=np.int64)
                        masks[t] = np.union1d(hold_pa,
                                              qs[qs < ctx["n_st"]])
                    batch = _build_batch(ctx, stats, chunk, obs_st, masks,
                                         device)
                    with _autocast(device):
                        mu, logvar, _q5, _q95, has_nbrs = model(batch)
                    N = batch["N"]
                    mu = mu.float().cpu().numpy().reshape(len(chunk), N)
                    lv = logvar.float().cpu().numpy().reshape(len(chunk), N)
                    hn = has_nbrs.cpu().numpy().reshape(len(chunk), N)
                    for b, t in enumerate(chunk):
                        for r in by_day[t]:
                            node = ru[r]
                            mu_a[r] = mu[b, node] * r_sd + r_mu
                            sg_a[r] = math.exp(0.5 * lv[b, node]) * r_sd
                            ok_a[r] = bool(hn[b, node])
        mu_m[(k, j)], sg_m[(k, j)], ok_m[(k, j)] = mu_a, sg_a, ok_a
        _say(f"predict: member f{k}_{j} scored {len(rows):,} rows")

    # -- Combine members per row (all-finite required: honesty over reach).
    mu_k = np.full((len(ks), n), np.nan)
    s2_k = np.full((len(ks), n), np.nan)
    ok_k = np.zeros((len(ks), n), dtype=bool)
    for ki, k in enumerate(ks):
        MU = np.stack([mu_m[(k, j)] for j in js_of[k]])
        SG = np.stack([sg_m[(k, j)] for j in js_of[k]])
        OK = np.stack([ok_m[(k, j)] for j in js_of[k]])
        aqs_sel = (~is_pa_row) & valid & (outer_row == k)
        all_ok = np.isfinite(MU).all(axis=0) & OK.all(axis=0)
        comb_mu = MU.mean(axis=0)
        comb_s2 = (SG ** 2).mean(axis=0) + MU.var(axis=0)
        mu_k[ki, aqs_sel & all_ok] = comb_mu[aqs_sel & all_ok]
        s2_k[ki, aqs_sel & all_ok] = comb_s2[aqs_sel & all_ok]
        ok_k[ki, aqs_sel & all_ok] = True
        # PA rows: exact held-out member; folds2-excluded units (-1) take
        # the fold-k member mean (every member is OOF for them).
        for jj, j in enumerate(js_of[k]):
            sel = is_pa_row & valid & (inner_row[k] == j)
            pick = sel & np.isfinite(MU[jj]) & OK[jj]
            mu_k[ki, pick] = MU[jj][pick]
            s2_k[ki, pick] = SG[jj][pick] ** 2
            ok_k[ki, pick] = True
        sel = is_pa_row & valid & (inner_row[k] == -1)
        pick = sel & all_ok
        mu_k[ki, pick] = comb_mu[pick]
        s2_k[ki, pick] = comb_s2[pick]
        ok_k[ki, pick] = True

    oof_r = np.full(n, np.nan)
    sigma = np.full(n, np.nan)
    avail = np.zeros(n, dtype=np.uint8)

    aqs_rows = (~is_pa_row) & valid & (outer_row >= 0)
    k_of_row = np.searchsorted(np.asarray(ks), outer_row)
    have_k = np.isin(outer_row, ks)
    sel = aqs_rows & have_k
    idx = np.flatnonzero(sel)
    ki = k_of_row[idx]
    good = ok_k[ki, idx]
    oof_r[idx[good]] = mu_k[ki[good], idx[good]]
    sigma[idx[good]] = np.sqrt(s2_k[ki[good], idx[good]])
    avail[idx[good]] = 1

    pa_rows = is_pa_row & valid
    all_k_ok = ok_k.all(axis=0)
    pick = pa_rows & all_k_ok
    if pick.any():
        oof_r[pick] = mu_k[:, pick].mean(axis=0)
        sigma[pick] = np.sqrt(s2_k[:, pick].mean(axis=0)
                              + mu_k[:, pick].var(axis=0))
        avail[pick] = 1

    # -- Structural unavailability (BUILD_NOTES #5 + vault airlock).
    if "nbr_pacal_avail_100km" not in frame.columns:
        raise SystemExit("[aqnet2] graph_res: frame lacks "
                         "nbr_pacal_avail_100km -- cannot enforce the "
                         "100-km hard zero")
    a100 = frame["nbr_pacal_avail_100km"].to_numpy(dtype=np.float64)
    a50 = (frame["nbr_pacal_avail_50km"].to_numpy(dtype=np.float64)
           if "nbr_pacal_avail_50km" in frame.columns
           else np.zeros(n))
    hard_zero = ~(a100 > 0)
    vault_row = (frame["unit_id"].astype(str).isin(ctx["vault"]).to_numpy()
                 | (_dates_ns(frame["date"])
                    >= pd.Timestamp(VAULT_DATE_START)).to_numpy())
    kill = hard_zero | vault_row | ~valid
    avail[kill] = 0
    oof_r[kill] = np.nan
    sigma[kill] = np.nan

    pattern_id = np.where(a50 > 0, 1, np.where(a100 > 0, 2, 0))
    pattern_id = np.where(avail > 0, pattern_id, 0).astype(np.int8)
    assert np.isnan(oof_r[avail == 0]).all(), (
        "avail=0 rows must carry NaN oof_r (no fill values, ever)")

    _save_npz_atomic(dest, oof_r=oof_r.astype(np.float64),
                     sigma=sigma.astype(np.float64),
                     avail=avail.astype(np.uint8),
                     pattern_id=pattern_id.astype(np.int8))
    meta = {"n_rows": n, "n_avail": int(avail.sum()),
            "n_pattern1": int((pattern_id == 1).sum()),
            "n_pattern2": int((pattern_id == 2).sum()),
            "members": {f"f{k}_{j}": str(p)
                        for (k, j), p in sorted(cmap.items())},
            "variant": cfg.get("variant"), "quick": cfg.get("quick", False)}
    _write_json_atomic(meta, dest.replace(".npz", "_meta.json"))
    _say(f"oof_tier2: {int(avail.sum()):,}/{n:,} rows available "
         f"-> {dest}")
    return {"oof_r": oof_r, "sigma": sigma, "avail": avail,
            "pattern_id": pattern_id, "path": dest}


# -- Leakage study (graphpre --study) ----------------------------------------

def run_leakage_study(cfg):
    """Fold-pure-vs-shared pretraining leakage magnitude on outer fold 0
    (DESIGN S7: the decision is recorded, not assumed). Fold-pure = the
    pretrain never sees PA stations within VAULT_BUFFER_KM of any fold-0
    AQS site (spatial leakage channel; pretraining has no FRM labels, so
    proximity is the only conduit). Both arms get a short pretrain + a
    (k=0, j=0) fine-tune; the gap in held-out inner-0 weighted R^2 decides
    whether full 5x fold-pure pretraining is required. Writes
    leakage_study.json {gap_r2, decision, ...}."""
    ctx = _get_ctx(cfg)
    epochs = QUICK_EPOCHS if cfg.get("quick") else STUDY_EPOCHS
    outer_unit = _unit_map_from_rows(ctx, ctx["folds"]["outer_fold"],
                                     "outer_fold")
    f0 = np.flatnonzero((outer_unit == 0) & ~ctx["is_station"])
    if len(f0) == 0:
        _say("leakage study: no fold-0 AQS sites in the window -- skipped")
        return None
    n_st = ctx["n_st"]
    dmin = np.full(n_st, np.inf)
    for q in f0:
        dmin = np.minimum(dmin, _haversine_km(
            ctx["lat"][:n_st], ctx["lon"][:n_st],
            ctx["lat"][q], ctx["lon"][q]))
    excl = np.flatnonzero(dmin <= float(config2.VAULT_BUFFER_KM))
    _say(f"leakage study: {len(excl)} stations within "
         f"{config2.VAULT_BUFFER_KM} km of a fold-0 site excluded in the "
         f"fold-pure arm")

    r2 = {}
    for name, ex in (("shared", np.array([], dtype=np.int64)),
                     ("fold_pure", excl)):
        pcfg = dict(cfg)
        pcfg.update({"_stem": f"graphpre_study_{name}", "epochs": epochs,
                     "_exclude_station_idx": ex, "study": False})
        pretrain(pcfg)
        fcfg = dict(cfg)
        fcfg.update({"_stem": f"graphres_study_{name}_f0_0",
                     "_init_stem": f"graphpre_study_{name}",
                     "epochs": max(QUICK_EPOCHS, epochs // 2),
                     "_eval_heldout": True, "study": False})
        finetune(fcfg, (0, 0))
        ck = load_checkpoint(f"graphres_study_{name}_f0_0")
        r2[name] = float(ck["cfg"].get("heldout_r2", float("nan")))

    gap = r2["shared"] - r2["fold_pure"]
    decision = ("fold_pure_required" if np.isfinite(gap)
                and gap > LEAKAGE_GAP_R2 else "shared_pretrain_ok")
    payload = {"gap_r2": gap, "decision": decision,
               "r2_shared": r2["shared"], "r2_fold_pure": r2["fold_pure"],
               "threshold": LEAKAGE_GAP_R2,
               "n_excluded_stations": int(len(excl)),
               "buffer_km": float(config2.VAULT_BUFFER_KM),
               "epochs": epochs, "quick": cfg.get("quick", False)}
    dest = config2.artifact("leakage_study.json")
    _write_json_atomic(payload, dest)
    _say(f"leakage study: gap_r2={gap:+.4f} -> {decision} ({dest})")
    return payload


# -- Stage runners -----------------------------------------------------------

def _outer_ks(folds, quick):
    vals = sorted({int(v) for v in
                   folds.get("outer_fold_of_site", {}).values()
                   if int(v) >= 0})
    if not vals:
        vals = list(range(int(config2.OUTER_N_FOLDS)))
    return vals[:QUICK_OUTER_FOLDS] if quick else vals


def run_graphpre(args):
    cfg = make_cfg("graphpre", quick=args.quick, variant=args.variant,
                   resume=args.resume, device=args.device,
                   epochs=args.epochs, study=args.study)
    suffix = _variant_suffix(cfg)
    marker = config2.artifact(f"graphpre{suffix}_marker.json", "graph")
    ckpt = _ckpt_path("graphpre" + suffix)
    study_dest = config2.artifact("leakage_study.json")
    done = os.path.exists(marker) and os.path.exists(ckpt)
    study_done = (not args.study) or os.path.exists(study_dest)
    if done and study_done and os.environ.get("FORCE") != "1":
        _say(f"{marker} exists (FORCE=1 to rebuild) -- skip")
        return 0
    _banner("graphpre")
    t0 = time.time()
    if not (done and os.environ.get("FORCE") != "1"):
        pretrain(cfg)
    else:
        _say(f"pretrain checkpoint exists -- study only")
        _get_ctx(cfg)
    if args.study and (os.environ.get("FORCE") == "1"
                       or not os.path.exists(study_dest)):
        run_leakage_study(cfg)
    _say(f"── stage graphpre done in {time.time() - t0:.1f}s")
    return 0


def run_graphres(args):
    cfg = make_cfg("graphres", quick=args.quick, variant=args.variant,
                   resume=args.resume, device=args.device,
                   epochs=args.epochs)
    suffix = _variant_suffix(cfg)
    dest = config2.artifact(f"oof_tier2{suffix}.npz")
    if os.path.exists(dest) and os.environ.get("FORCE") != "1":
        _say(f"{dest} exists (FORCE=1 to rebuild) -- skip")
        return 0
    _banner("graphres")
    t0 = time.time()
    ctx = _get_ctx(cfg)
    ks = _outer_ks(ctx["folds"], cfg["quick"])
    js = list(range(int(config2.INNER_N_FOLDS)))
    _say(f"fine-tuning {len(ks)}x{len(js)} fold-assignment ensemble "
         f"members (outer {ks}, inner {js})")
    ckpts = {}
    for k in ks:
        for j in js:
            ckpts[(k, j)] = finetune(cfg, (k, j))
    predict_oof(ctx["frame"], ctx["folds"], ckpts, cfg=cfg, dest=dest)
    _say(f"── stage graphres done in {time.time() - t0:.1f}s")
    return 0


def run_predict(args):
    """Re-run predict_oof from existing fine-tune checkpoints (no
    training). FORCE=1 overwrites the existing npz."""
    cfg = make_cfg("graphres", quick=args.quick, variant=args.variant,
                   device=args.device)
    suffix = _variant_suffix(cfg)
    dest = config2.artifact(f"oof_tier2{suffix}.npz")
    if os.path.exists(dest) and os.environ.get("FORCE") != "1":
        _say(f"{dest} exists (FORCE=1 to rebuild) -- skip")
        return 0
    _banner("graphres-predict")
    ctx = _get_ctx(cfg)
    ks = _outer_ks(ctx["folds"], cfg["quick"])
    js = list(range(int(config2.INNER_N_FOLDS)))
    ckpts = {}
    for k in ks:
        for j in js:
            p = _ckpt_path(f"graphres{suffix}_f{k}_{j}")
            if os.path.exists(p):
                ckpts[(k, j)] = p
    if not ckpts:
        raise SystemExit("[aqnet2] graph_res: no graphres checkpoints "
                         "found -- run the graphres stage first")
    missing = [(k, j) for k in ks for j in js if (k, j) not in ckpts]
    if missing:
        _say(f"WARNING: missing fine-tune checkpoints {missing} -- rows "
             f"needing them will be unavailable")
    predict_oof(ctx["frame"], ctx["folds"], ckpts, cfg=cfg, dest=dest)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="AQNet v2 T2 masked graph-attention residual "
                    "(pretrain / per-fold finetune / OOF predict)")
    ap.add_argument("stage", choices=["graphpre", "graphres", "predict"])
    ap.add_argument("--quick", action="store_true",
                    help="3-month window, 2 outer folds, 2 epochs")
    ap.add_argument("--resume", action="store_true",
                    help="autodetect and resume from *_last.pt")
    ap.add_argument("--variant", choices=["full", "temporal"],
                    default="full",
                    help="temporal: train strictly before TEMPORAL_CUTOFF")
    ap.add_argument("--study", action="store_true",
                    help="graphpre only: fold-pure-vs-shared leakage study")
    ap.add_argument("--device", default="auto",
                    help="auto|cuda|cpu|mps")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the stage epoch budget")
    args = ap.parse_args(argv)
    if args.stage == "graphpre":
        return run_graphpre(args)
    if args.stage == "graphres":
        return run_graphres(args)
    return run_predict(args)


if __name__ == "__main__":
    sys.exit(main())
