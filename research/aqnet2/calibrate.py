"""AQNet v2 S1: learned Kennedy-O'Hagan PurpleAir->FRM calibration
(stage `calibrate`, DESIGN S4).

v1 trained on Barkjohn-corrected PurpleAir and scored R^2 ~ 0 at the AQS
monitors with a -2.7..-3.8 ug/m3 bias: fixed US-wide cf_1 constants applied
to an ATM archive, no drift term, no smoke-regime term. This stage replaces
those constants with a calibration LEARNED on the Texas colocation pairs
(colocate.py), in the Kennedy-O'Hagan multi-fidelity form

    y_FRM(s,t) = rho(x_cal) * y_PA(s',t) + delta(x_cal)
                 + b0_j + b1_j * y_PA + g_{j,year} + eps

where the GBM fixed effects over x_cal (which includes y_PA) absorb
rho() * y_PA + delta() jointly, and grouped random effects give each sensor
a partially-pooled intercept/slope (the device nugget that zeroed v1's
residual kriging) plus per-sensor-year drift intercepts.

Implementation chain (first available wins; every rung prints why it fell):
  1. GPBoost: LightGBM fixed effects + grouped REs (sensor intercept+slope,
     sensor-year intercept), Gaussian NLL.
  2. LightGBM linear-tree fixed effects + statsmodels MixedLM on the
     residuals (per-sensor random intercept+slope; the sensor-year RE is
     dropped in this rung -- sensor_age_days in x_cal is the drift proxy).
  3. Pure LightGBM (no random effects).
  4. Weighted linear pm25 ~ pa + rh + t (numpy; emergency only, so this
     module functions with zero optional deps).

Leakage rules implemented here (load-bearing, DESIGN S2/S4):
  - Vault airlock: vault sites from folds2.json never appear in any
    calibration pair, in any fit, full included. Asserted, not assumed.
  - Nested refits: pa_cal_full, pa_cal_f{k} (all pairs touching outer-fold-k
    AQS sites dropped), pa_cal_f{k}_{j} (fold-k plus inner-(k,j) sites
    dropped). A site's pairs never influence its own fold's column.
  - No fill values: NaN raw PA yields NaN calibrated PA; missing humidity is
    left NaN for the tree models' native missing handling and propagates to
    NaN through the linear forms.

cal_var honesty: a second (quantile-spread) model gives a heteroscedastic
cal_var_model, then `cal_var_final = max(cal_var_model, cal_var_floor(d))`
with a monotone floor in the distance to the nearest FRM reference, so
sensors far off the colocation support genuinely shrink in T1 training
weight. channel-reconstructed rows get cal_var inflated by a fixed factor
(DESIGN S4 cf_1 fallback policy).

Gate G0 (LOLO over pairing sites): the learned calibration must beat BOTH
published Barkjohn AND the AMT-2024-style multilinear RH+T form refit on the
TX pairs, on LOLO RMSE and |bias|; otherwise the nested columns are produced
from the best refit linear form (never published constants on ATM).

Run from anywhere:
    python calibrate.py [--quick]
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

import config2
import colocate

# ── Guarded heavy imports (v1 models_tabular style) ─────────────────────────
try:
    import gpboost as gpb
    HAS_GPBOOST = True
except ImportError:
    gpb = None
    HAS_GPBOOST = False
    print("[aqnet2] calibrate: gpboost not installed -- falling back to "
          "LightGBM+MixedLM chain (pip install gpboost)")

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    lgb = None
    HAS_LGBM = False
    print("[aqnet2] calibrate: lightgbm not installed -- tree rungs "
          "unavailable (pip install lightgbm)")

try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    smf = None
    HAS_STATSMODELS = False
    print("[aqnet2] calibrate: statsmodels not installed -- MixedLM random "
          "effects unavailable (pip install statsmodels)")

# ── Constants ───────────────────────────────────────────────────────────────

PRIMARY_PAIR_KM = 10.0        # primary calibration radius (DESIGN S4)
SENSITIVITY_PAIR_KM = 25.0    # sensitivity arm
PAIR_WEIGHT_SCALE_KM = 5.0    # pair weight exp(-d / 5 km)
SMOKE_OVERSAMPLE = 5          # smoke-day pairs appear 5x in mean fits

# cf_1 channel policy (DESIGN S4): the committed parquet is ATM-only. Below
# ~20 ug/m3 the PA firmware's ATM and cf_1 estimates coincide, so ATM rows
# under the threshold are treated as cf_1; above it the channel is flagged
# reconstructed, cal_var is inflated, and downstream exceedance labelling
# must exclude the row (enforced in exceed.py, flagged here).
CF1_RECON_UGM3 = 20.0
CHANNEL_RECON_VAR_FACTOR = 4.0   # 2x sigma for reconstructed-channel rows

# Off-support cal_var floor: a * expm1(d / 50 km) + b. Conservative by
# construction: b = 1.0 (ug/m3)^2 minimum calibration variance even at a
# colocated sensor; a = 4.0 puts the floor at ~7.9 at 50 km, ~26.6 at
# 100 km, ~215 at 200 km, i.e. a 200-km-from-any-FRM sensor's precision
# weight sigma_FRM^2/(sigma_FRM^2 + cal_var) collapses to ~0.01.
VAR_FLOOR_A = 4.0
VAR_FLOOR_SCALE_KM = 50.0
VAR_FLOOR_B = 1.0
CAL_VAR_MIN = 0.25            # never claim better than +-0.5 ug/m3

VAR_QUANTILES = (0.16, 0.84)  # +-1 sigma band for the variance model
NUM_BOOST_ROUND = 300
QUICK_BOOST_ROUND = 60
QUICK_START, QUICK_END = "2024-07-01", "2024-09-30"   # v1 --quick window
QUICK_LOLO_MAX_SITES = 8

# Barkjohn et al. (2021) US-wide constants -- the v1 target correction
# (research/aqnet/corrections.py). Kept here as the published baseline the
# learned calibration must beat; NEVER used as a training target in v2.
BARKJOHN_SLOPE = 0.524
BARKJOHN_RH_COEF = -0.0862
BARKJOHN_INTERCEPT = 5.75

# x_cal per DESIGN S4. `pa` is the raw PA reading; pair-only features take
# deployment conventions at apply time (dist_km = 0: the calibrated value
# estimates a hypothetically colocated monitor; is_fem = 0: FRM reference
# scale is the primary standard).
CAL_FEATURES = ["pa", "rh", "t", "dewpoint", "hms_smoke",
                "doy_sin", "doy_cos", "doy_sin2", "doy_cos2",
                "sensor_age_days", "dist_km", "channel_reconstructed",
                "is_fem", "urban"]

URBAN_TRACT_POP = 5000.0      # crude tract-population urban proxy (see below)

HMS_PARQUET = os.path.join(config2.PIPELINE_DIR, "hms_smoke_by_sensor.parquet")

_FORM_BY_BASELINE = {"barkjohn_refit": "linear_rh", "amt_rht": "linear_rht"}
_LINEAR_COLS = {"linear_rh": ("pa", "rh"), "linear_rht": ("pa", "rh", "t")}


def _say(msg):
    print(f"[aqnet2] calibrate: {msg}")


# ── Shared feature transforms ───────────────────────────────────────────────

def dewpoint_c(t_c, rh_pct):
    """Magnus dewpoint (deg C) from temperature (deg C) and RH (%).

    RH is clipped to [1, 100] inside the log only (domain guard on a
    deterministic transform, not a data fill); NaN inputs propagate.
    """
    t = np.asarray(t_c, dtype=np.float64)
    rh = np.clip(np.asarray(rh_pct, dtype=np.float64), 1.0, 100.0)
    gamma = np.log(rh / 100.0) + 17.625 * t / (243.04 + t)
    return 243.04 * gamma / (17.625 - gamma)


def add_time_features(df):
    """Add doy harmonic columns (2 harmonics) in place; returns df."""
    doy = pd.to_datetime(df["date"]).dt.dayofyear.to_numpy(dtype=np.float64)
    ang = 2.0 * np.pi * doy / 365.25
    df["doy_sin"] = np.sin(ang)
    df["doy_cos"] = np.cos(ang)
    df["doy_sin2"] = np.sin(2.0 * ang)
    df["doy_cos2"] = np.cos(2.0 * ang)
    return df


# ── Loaders ─────────────────────────────────────────────────────────────────

def load_pa_daily(pa_parquet=None, hms_parquet=None, start=None, end=None):
    """Sensor-day frame with x_cal inputs from the committed PA parquet.

    Columns: sensor_id(str), date, lat, lon, pa_raw, rh, t, dewpoint,
    hms_smoke, sensor_age_days, channel_reconstructed, urban, doy harmonics.

    Channel policy: if the parquet carries a `pm25_cf1` column (post
    `data-pa` refetch) it is preferred, with per-row ATM fallback flagged
    reconstructed above CF1_RECON_UGM3; on the ATM-only committed parquet
    every row >= CF1_RECON_UGM3 is flagged reconstructed (DESIGN S4).

    Urban flag: tract population >= URBAN_TRACT_POP -- a crude proxy until
    fetchers2 lands AQS location-setting metadata; it exists so the learned
    form can separate urban/rural RH regimes, not as a precise land-use
    variable.
    """
    path = pa_parquet or colocate.PA_PARQUET
    pa = pd.read_parquet(path)
    cols = {"sensor_id", "date", "pm25", "latitude", "longitude",
            "humidity", "temperature"}
    missing = cols - set(pa.columns)
    if missing:
        raise ValueError(f"PA parquet {path} missing columns {sorted(missing)}")

    out = pd.DataFrame({
        "sensor_id": pa["sensor_id"].astype(str),
        "date": pd.to_datetime(pa["date"]).dt.normalize(),
        "lat": pa["latitude"].astype(np.float64),
        "lon": pa["longitude"].astype(np.float64),
        "rh": pa["humidity"].astype(np.float64),
        "t": pa["temperature"].astype(np.float64),
    })
    atm = pa["pm25"].astype(np.float64).to_numpy()
    if "pm25_cf1" in pa.columns:
        cf1 = pa["pm25_cf1"].astype(np.float64).to_numpy()
        recon = ~np.isfinite(cf1) & np.isfinite(atm) & (atm >= CF1_RECON_UGM3)
        out["pa_raw"] = np.where(np.isfinite(cf1), cf1, atm)
        out["channel_reconstructed"] = recon.astype(np.float64)
    else:
        out["pa_raw"] = atm
        out["channel_reconstructed"] = (
            np.isfinite(atm) & (atm >= CF1_RECON_UGM3)).astype(np.float64)

    if "POPULATION" in pa.columns:
        popn = pa["POPULATION"].astype(np.float64).to_numpy()
        out["urban"] = (popn >= URBAN_TRACT_POP).astype(np.float64)
    else:
        out["urban"] = 0.0

    out["dewpoint"] = dewpoint_c(out["t"], out["rh"])
    add_time_features(out)

    # HMS smoke tier by (sensor, day); absence of a smoke polygon is tier 0
    # by that product's semantics (v1 convention), not a fill.
    hpath = hms_parquet or HMS_PARQUET
    if os.path.exists(hpath):
        hms = pd.read_parquet(hpath)
        hms["sensor_id"] = hms["sensor_id"].astype(str)
        hms["date"] = pd.to_datetime(hms["date"]).dt.normalize()
        out = out.merge(hms[["sensor_id", "date", "hms_smoke"]],
                        on=["sensor_id", "date"], how="left")
        out["hms_smoke"] = out["hms_smoke"].fillna(0).astype(np.float64)
    else:
        _say(f"HMS parquet not found at {hpath} -- hms_smoke tier 0 everywhere")
        out["hms_smoke"] = 0.0

    first = out.groupby("sensor_id")["date"].transform("min")
    out["sensor_age_days"] = (out["date"] - first).dt.days.astype(np.float64)

    if start is not None:
        out = out[out["date"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["date"] <= pd.Timestamp(end)]
    return out.reset_index(drop=True)


def load_aqs_daily(aqs_parquet=None, start=None, end=None):
    """AQS site-day frame [site_id(str), date, pm25_aqs, lat, lon, is_fem].

    The v1 daily parquet does not retain the sampling method; until the
    hardened fetchers2 AQS table (POC + method) lands, is_fem degrades to a
    constant 0.0 (announced once) -- a constant column is inert in every
    model form used here.
    """
    path = aqs_parquet
    if path is None:
        # Prefer the fetchers2-hardened v2 table (carries is_fem + urban
        # metadata this stage's x_cal actually uses); fall back to the
        # committed v1 parquet with the constant-column degradation below.
        import glob as _glob
        _v2 = sorted(_glob.glob(os.path.join(
            config2.DATA_DIR, "aqs_daily_tx_v2_*.parquet")))
        path = _v2[-1] if _v2 else colocate.AQS_PARQUET
    aq = pd.read_parquet(path)
    aq["site_id"] = aq["site_id"].astype(str)
    aq["date"] = pd.to_datetime(aq["date"]).dt.normalize()
    if "is_fem" in aq.columns:
        aq["is_fem"] = aq["is_fem"].astype(np.float64)
    elif "method" in aq.columns:
        aq["is_fem"] = (aq["method"].astype(str).str.upper()
                        .str.contains("FEM")).astype(np.float64)
    else:
        _say("AQS parquet has no method column -- is_fem constant 0.0 "
             "(hardened fetcher not yet consumed)")
        aq["is_fem"] = 0.0
    if start is not None:
        aq = aq[aq["date"] >= pd.Timestamp(start)]
    if end is not None:
        aq = aq[aq["date"] <= pd.Timestamp(end)]
    keep = ["site_id", "date", "pm25_aqs", "lat", "lon", "is_fem"]
    return aq[keep].reset_index(drop=True)


def dist_to_nearest_frm(pa_daily, aqs_daily, exclude_sites=()):
    """Per-row distance (km) from each sensor to its nearest usable FRM site.

    `exclude_sites` (normally the vault) are not usable references: they
    must not tighten any sensor's cal_var floor (DESIGN S2 vault airlock).
    """
    ex = {str(s) for s in exclude_sites}
    sites = (aqs_daily[~aqs_daily["site_id"].isin(ex)]
             .groupby("site_id")[["lat", "lon"]].median())
    sens = pa_daily.groupby("sensor_id")[["lat", "lon"]].median()
    if len(sites) == 0:
        raise ValueError("no usable FRM sites for the cal_var floor")
    d = colocate.haversine_km(sens["lat"].to_numpy()[:, None],
                              sens["lon"].to_numpy()[:, None],
                              sites["lat"].to_numpy()[None, :],
                              sites["lon"].to_numpy()[None, :])
    nearest = pd.Series(d.min(axis=1), index=sens.index)
    return pa_daily["sensor_id"].map(nearest).to_numpy(dtype=np.float64)


# ── folds2.json consumption (site-level views) ──────────────────────────────

def load_fold_sites(path=None):
    """Load folds2.json as a plain dict.

    calibrate runs before the training frame exists, so it cannot go through
    folds2.load_folds (which verifies the frame content hash); it consumes
    only the site-level keys: vault_sites, outer_fold_of_site and (if
    present) inner_fold_of_site.
    """
    p = path or config2.artifact("folds2.json")
    if not os.path.exists(p):
        raise SystemExit(
            f"[aqnet2] calibrate: folds2.json not found at {p} -- build the "
            "fold system (folds2.py) before calibrate; nested calibration "
            "without folds would be a leak, refusing to run.")
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def vault_site_set(folds):
    return {str(s) for s in (folds or {}).get("vault_sites", [])}


def outer_fold_ids(folds):
    m = (folds or {}).get("outer_fold_of_site", {})
    return sorted({int(v) for v in m.values() if int(v) >= 0})


def inner_fold_of_site(folds, outer_k, _warned=[False]):
    """{site_id: inner fold j} for the AQS sites inside outer fold k's
    inner split.

    Preferred source: an explicit site-level `inner_fold_of_site` map in
    folds2.json ({outer_k: {site_id: j}}). If folds2.json only carries the
    row-level inner_fold arrays (which need the frame to interpret), fall
    back to a deterministic seeded assignment over the remaining non-vault
    sites: stable across runs, honest (whole site groups are still
    excluded), but it must MATCH folds2's row-level assignment for the
    inner columns to line up with selection/confirmation downstream -- the
    derived map is therefore recorded in calibration_report.json for audit.
    """
    folds = folds or {}
    explicit = folds.get("inner_fold_of_site")
    if explicit is not None:
        mk = explicit.get(str(outer_k), explicit.get(outer_k))
        if mk:
            return {str(s): int(j) for s, j in mk.items()}
    if not _warned[0]:
        _say("WARNING: folds2.json has no site-level inner_fold_of_site map; "
             "deriving a deterministic inner assignment (seeded) -- verify it "
             "matches folds2's row-level inner folds (recorded in report)")
        _warned[0] = True
    vault = vault_site_set(folds)
    outer = {str(s): int(f)
             for s, f in folds.get("outer_fold_of_site", {}).items()}
    remaining = sorted(s for s, f in outer.items()
                       if f != int(outer_k) and s not in vault)
    rng = np.random.default_rng(config2.SEED + 100003 * (int(outer_k) + 1))
    order = rng.permutation(len(remaining))
    n_inner = int(getattr(config2, "INNER_N_FOLDS", 4))
    return {remaining[idx]: int(pos % n_inner)
            for pos, idx in enumerate(order)}


def excluded_site_set(folds, outer_k=None, inner_j=None):
    """AQS sites whose pairs must be dropped for a given nested fit.

    Always includes the vault. outer_k adds fold-k sites; inner_j (requires
    outer_k) adds the inner-(k, j) sites. This is the single source of truth
    for 'drop every pair touching that fold's AQS sites'.
    """
    folds = folds or {}
    excl = vault_site_set(folds)
    if inner_j is not None and outer_k is None:
        raise ValueError("inner_j requires outer_k")
    if outer_k is not None:
        outer = folds.get("outer_fold_of_site", {})
        excl |= {str(s) for s, f in outer.items() if int(f) == int(outer_k)}
        if inner_j is not None:
            inner = inner_fold_of_site(folds, outer_k)
            excl |= {s for s, j in inner.items() if j == int(inner_j)}
    return excl


# ── Pair frame ──────────────────────────────────────────────────────────────

def build_cal_frame(pairs, pa_daily, aqs_daily, max_dist_km=PRIMARY_PAIR_KM):
    """Pair-day training frame for the KO calibration.

    One row per (site, sensor, day) with y = pm25_aqs, x_cal features, and
    the distance-decayed precision weight w = exp(-d/5km) / sigma_FRM^2.
    Rows require finite (y, pa, rh, t): humidity/temperature are core to
    every candidate form and the design forbids imputation.
    """
    p = pairs.loc[pairs["dist_km"] <= max_dist_km,
                  ["site_id", "sensor_id", "dist_km"]].copy()
    p["site_id"] = p["site_id"].astype(str)
    p["sensor_id"] = p["sensor_id"].astype(str)

    aq = aqs_daily[["site_id", "date", "pm25_aqs", "is_fem"]].copy()
    aq["site_id"] = aq["site_id"].astype(str)
    cal = p.merge(aq, on="site_id", how="inner")
    cal = cal.merge(pa_daily.drop(columns=["lat", "lon"], errors="ignore"),
                    on=["sensor_id", "date"], how="inner")
    cal = cal.rename(columns={"pm25_aqs": "y", "pa_raw": "pa"})

    ok = (np.isfinite(cal["y"]) & np.isfinite(cal["pa"])
          & np.isfinite(cal["rh"]) & np.isfinite(cal["t"]))
    cal = cal[ok].reset_index(drop=True)

    cal["year"] = pd.to_datetime(cal["date"]).dt.year.astype(int)
    cal["sensor_year"] = (cal["sensor_id"] + "_" + cal["year"].astype(str))
    cal["w"] = (np.exp(-cal["dist_km"].to_numpy() / PAIR_WEIGHT_SCALE_KM)
                / float(config2.SIGMA_FRM) ** 2)
    return cal


def _oversample_index(cal, seed):
    """Training index with smoke-day rows (hms_smoke >= 1) repeated 5x."""
    base = np.arange(len(cal))
    smoke = base[cal["hms_smoke"].to_numpy() >= 1]
    idx = np.concatenate([base] + [smoke] * (SMOKE_OVERSAMPLE - 1))
    return np.random.default_rng(seed).permutation(idx)


def _X(df, feats=CAL_FEATURES):
    return df[list(feats)].to_numpy(dtype=np.float64)


# ── Linear forms (baselines, G0 fallback, emergency rung) ──────────────────

def barkjohn_published(df):
    """Published Barkjohn et al. (2021) form on raw PA + RH, clipped at 0."""
    out = (BARKJOHN_SLOPE * df["pa"].to_numpy(dtype=np.float64)
           + BARKJOHN_RH_COEF * df["rh"].to_numpy(dtype=np.float64)
           + BARKJOHN_INTERCEPT)
    return np.clip(out, 0.0, None)


def fit_linear_form(cal, cols, sample_idx=None):
    """Weighted least squares y ~ 1 + cols on the pair frame.

    cols=("pa","rh") is the Barkjohn FORM refit on TX pairs; ("pa","rh","t")
    is the AMT-2024-style multilinear RH+T baseline, its coefficients fit on
    the TX pairs (a refit pm25 ~ pa + rh + t baseline, not published
    constants).
    """
    rows = cal if sample_idx is None else cal.iloc[sample_idx]
    Xd = np.column_stack([np.ones(len(rows))]
                         + [rows[c].to_numpy(dtype=np.float64) for c in cols])
    y = rows["y"].to_numpy(dtype=np.float64)
    sw = np.sqrt(rows["w"].to_numpy(dtype=np.float64))
    coef, *_ = np.linalg.lstsq(Xd * sw[:, None], y * sw, rcond=None)
    return {"cols": list(cols), "coef": [float(c) for c in coef]}


def predict_linear_form(form, df):
    out = np.full(len(df), form["coef"][0], dtype=np.float64)
    for c, col in zip(form["coef"][1:], form["cols"]):
        out = out + c * df[col].to_numpy(dtype=np.float64)
    return np.clip(out, 0.0, None)


# ── Model fitting rungs ─────────────────────────────────────────────────────

def _lgb_params(objective, seed, **extra):
    p = {"objective": objective, "learning_rate": 0.05, "num_leaves": 31,
         "min_data_in_leaf": 20, "verbose": -1, "seed": int(seed)}
    p.update(extra)
    return p


def _fit_gpboost(cal, os_idx, nbr, seed):
    tr = cal.iloc[os_idx]
    gp_model = gpb.GPModel(
        group_data=np.column_stack([tr["sensor_id"].to_numpy(dtype=object),
                                    tr["sensor_year"].to_numpy(dtype=object)]),
        group_rand_coef_data=tr["pa"].to_numpy(dtype=np.float64)
                               .reshape(-1, 1),
        ind_effect_group_rand_coef=[1],   # random slope on sensor_id
        likelihood="gaussian")
    ds = gpb.Dataset(_X(tr), label=tr["y"].to_numpy(dtype=np.float64),
                     weight=tr["w"].to_numpy(dtype=np.float64))
    bst = gpb.train(params=_lgb_params("regression_l2", seed),
                    train_set=ds, gp_model=gp_model, num_boost_round=nbr)
    _gpb_predict(bst, cal.iloc[:5])   # API smoke check inside the try
    return bst


def _gpb_predict(bst, df):
    out = bst.predict(
        data=_X(df),
        group_data_pred=np.column_stack([df["sensor_id"].to_numpy(dtype=object),
                                         df["sensor_year"].to_numpy(dtype=object)]),
        group_rand_coef_data_pred=df["pa"].to_numpy(dtype=np.float64)
                                    .reshape(-1, 1),
        predict_var=False, pred_latent=False)
    if isinstance(out, dict):
        mean = out.get("response_mean")
        if mean is None:
            mean = (np.asarray(out["fixed_effect"], dtype=np.float64)
                    + np.asarray(out["random_effect_mean"], dtype=np.float64))
        return np.asarray(mean, dtype=np.float64)
    return np.asarray(out, dtype=np.float64)


def _fit_lgbm(cal, os_idx, nbr, seed, linear_tree=True):
    tr = cal.iloc[os_idx]
    params = _lgb_params("regression", seed)
    if linear_tree:
        params["linear_tree"] = True
    ds = lgb.Dataset(_X(tr), label=tr["y"].to_numpy(dtype=np.float64),
                     weight=tr["w"].to_numpy(dtype=np.float64))
    return lgb.train(params, ds, num_boost_round=nbr)


def _fit_mixedlm_re(cal, gbm):
    """Per-sensor random intercept+slope on the GBM residuals.

    Fit on the unique (non-oversampled) training rows. Returns
    (fe_intercept, {sensor_id: (b0, b1)}).
    """
    resid = (cal["y"].to_numpy(dtype=np.float64)
             - gbm.predict(_X(cal)))
    dfr = pd.DataFrame({"r": resid,
                        "pa": cal["pa"].to_numpy(dtype=np.float64),
                        "sensor_id": cal["sensor_id"].to_numpy()})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = smf.mixedlm("r ~ 1", dfr, groups=dfr["sensor_id"],
                         re_formula="~pa")
        res = md.fit(method="lbfgs", reml=True, maxiter=200)
    fe = float(res.fe_params.get("Intercept", 0.0))
    re = {str(g): (float(s.get("Group", 0.0)), float(s.get("pa", 0.0)))
          for g, s in res.random_effects.items()}
    return fe, re


def _fit_var_model(cal, nbr, seed):
    """Second-stage heteroscedastic cal_var model (quantile spread).

    Two LightGBM quantile heads at +-1 sigma; cal_var_model =
    ((q84 - q16)/2)^2. The FRM measurement noise stays inside the spread --
    a deliberate conservative inflation, never subtracted out. Without
    lightgbm the model degrades to the weighted residual variance of the
    mean fit (scalar), announced.
    """
    if HAS_LGBM:
        heads = []
        for q in VAR_QUANTILES:
            ds = lgb.Dataset(_X(cal),
                             label=cal["y"].to_numpy(dtype=np.float64),
                             weight=cal["w"].to_numpy(dtype=np.float64))
            params = _lgb_params("quantile", seed, alpha=float(q),
                                 num_leaves=15, min_data_in_leaf=40)
            heads.append(lgb.train(params, ds,
                                   num_boost_round=min(nbr, 200)))
        return {"kind": "quantile", "lo": heads[0], "hi": heads[1],
                "scalar": None}
    _say("variance model degraded to scalar residual variance "
         "(lightgbm unavailable)")
    return {"kind": "scalar", "lo": None, "hi": None, "scalar": None}


class CalModel:
    """Fitted calibration: mean form + heteroscedastic variance machinery.

    predict(df) / predict_var(df) consume any frame carrying CAL_FEATURES
    plus sensor_id / sensor_year (for the random-effect forms). Predictions
    are clipped at 0 (PM mass) and NaN wherever `pa` is NaN -- the
    no-fill rule holds through application.
    """

    def __init__(self, kind, meta):
        self.kind = kind
        self.meta = meta
        self.booster = None          # gpboost or lightgbm mean model
        self.fe_intercept = 0.0      # mixedlm rung
        self.re_sensor = {}          # {sensor_id: (b0, b1)}
        self.linear = None           # linear-form dict
        self.var_model = None
        self.var_scalar = float(CAL_VAR_MIN)

    # -- mean ---------------------------------------------------------------
    def predict(self, df):
        pa = df["pa"].to_numpy(dtype=np.float64)
        if self.kind == "gpboost":
            out = _gpb_predict(self.booster, df)
        elif self.kind == "lgbm_mixedlm":
            out = self.booster.predict(_X(df)) + self.fe_intercept
            b = np.array([self.re_sensor.get(s, (0.0, 0.0))
                          for s in df["sensor_id"].astype(str)],
                         dtype=np.float64).reshape(-1, 2)
            out = out + b[:, 0] + b[:, 1] * pa
        elif self.kind == "lgbm":
            out = self.booster.predict(_X(df))
        elif self.kind in _LINEAR_COLS:
            out = predict_linear_form(self.linear, df)
        else:
            raise ValueError(f"unknown CalModel kind {self.kind!r}")
        out = np.asarray(out, dtype=np.float64)
        out = np.clip(out, 0.0, None)
        out[~np.isfinite(pa)] = np.nan
        return out

    # -- variance (model term only; the floor is applied by the caller) ----
    def predict_var(self, df):
        if self.var_model is not None and self.var_model["kind"] == "quantile":
            lo = self.var_model["lo"].predict(_X(df))
            hi = self.var_model["hi"].predict(_X(df))
            var = ((hi - lo) / 2.0) ** 2
        else:
            var = np.full(len(df), self.var_scalar, dtype=np.float64)
        return np.maximum(np.asarray(var, dtype=np.float64), CAL_VAR_MIN)


def _fit_on_frame(cal, num_boost_round=None, seed=None, model_form="learned"):
    """Fit the calibration chain on an already-filtered pair frame."""
    nbr = int(num_boost_round or NUM_BOOST_ROUND)
    seed = config2.SEED if seed is None else int(seed)
    if len(cal) == 0:
        raise ValueError("empty calibration frame")
    os_idx = _oversample_index(cal, seed)
    meta = {"n_pair_days": int(len(cal)),
            "n_sites": int(cal["site_id"].nunique()),
            "n_sensors": int(cal["sensor_id"].nunique()),
            "sites": sorted(cal["site_id"].unique().tolist()),
            "messages": []}

    model = None
    if model_form == "learned":
        if HAS_GPBOOST:
            try:
                bst = _fit_gpboost(cal, os_idx, nbr, seed)
                model = CalModel("gpboost", meta)
                model.booster = bst
            except Exception as e:      # noqa: BLE001 -- fall down the chain
                meta["messages"].append(f"gpboost failed: {e!r}")
                _say(f"gpboost rung failed ({e!r}) -- falling back")
        if model is None and HAS_LGBM and HAS_STATSMODELS:
            try:
                gbm = _fit_lgbm(cal, os_idx, nbr, seed, linear_tree=True)
                fe, re = _fit_mixedlm_re(cal, gbm)
                model = CalModel("lgbm_mixedlm", meta)
                model.booster, model.fe_intercept, model.re_sensor = gbm, fe, re
            except Exception as e:      # noqa: BLE001
                meta["messages"].append(f"lgbm+mixedlm failed: {e!r}")
                _say(f"lgbm+mixedlm rung failed ({e!r}) -- falling back")
        if model is None and HAS_LGBM:
            try:
                gbm = _fit_lgbm(cal, os_idx, nbr, seed, linear_tree=True)
                model = CalModel("lgbm", meta)
                model.booster = gbm
                meta["messages"].append("no random effects (pure LightGBM)")
            except Exception as e:      # noqa: BLE001
                meta["messages"].append(f"lgbm failed: {e!r}")
                _say(f"pure-lgbm rung failed ({e!r}) -- falling back")
        if model is None:
            _say("all tree rungs unavailable -- emergency weighted linear "
                 "pa+rh+t form")
            model_form = "linear_rht"

    if model is None:
        cols = _LINEAR_COLS[model_form]
        model = CalModel(model_form, meta)
        model.linear = fit_linear_form(cal, cols, sample_idx=os_idx)

    model.var_model = _fit_var_model(cal, nbr, seed)
    if model.var_model["kind"] == "scalar":
        resid = cal["y"].to_numpy(dtype=np.float64) - model.predict(cal)
        w = cal["w"].to_numpy(dtype=np.float64)
        good = np.isfinite(resid)
        model.var_scalar = float(max(
            np.average(resid[good] ** 2, weights=w[good]), CAL_VAR_MIN))
    meta["kind"] = model.kind
    return model


# ── Public API (frozen signatures, INTERFACES.md) ──────────────────────────

def build_pairs(*args, **kwargs):
    """Pair-table construction lives in colocate.py (single owner)."""
    return colocate.build_pairs(*args, **kwargs)


def fit_calibration(pairs, pa_daily, folds, outer_k=None, inner_j=None,
                    aqs_daily=None, max_dist_km=PRIMARY_PAIR_KM,
                    num_boost_round=None, seed=None, model_form="learned"):
    """Fit one nested KO calibration.

    outer_k / inner_j select the nested exclusion (None/None = pa_cal_full,
    which still excludes the vault). Every pair touching an excluded AQS
    site is dropped BEFORE fitting; the vault airlock is asserted on the
    resulting frame.
    """
    if aqs_daily is None:
        aqs_daily = load_aqs_daily()
    excl = excluded_site_set(folds, outer_k, inner_j)
    vault = vault_site_set(folds)

    cal = build_cal_frame(pairs, pa_daily, aqs_daily, max_dist_km)
    cal = cal[~cal["site_id"].isin(excl)].reset_index(drop=True)
    touched = set(cal["site_id"])
    assert not (touched & vault), (
        f"vault airlock breach: vault sites {sorted(touched & vault)} "
        "present in calibration pairs")
    assert not (touched & excl), "fold exclusion failed to drop pair sites"
    if len(cal) == 0:
        raise ValueError(
            f"no calibration pairs left after excluding {sorted(excl)} "
            f"at <= {max_dist_km} km")

    model = _fit_on_frame(cal, num_boost_round, seed, model_form)
    model.meta.update({"outer_k": outer_k, "inner_j": inner_j,
                       "excluded_sites": sorted(excl),
                       "max_dist_km": float(max_dist_km)})
    return model


def cal_var_floor(dist_to_nearest_pair_km):
    """Monotone off-support cal_var floor (ug/m3)^2.

    floor(d) = VAR_FLOOR_A * expm1(d / VAR_FLOOR_SCALE_KM) + VAR_FLOOR_B.
    Strictly increasing in d, floor(0) = VAR_FLOOR_B; constants documented
    at the top of this module. Applied as max(model, floor) so the
    heteroscedastic model can only ever ADD variance off-support, never
    remove it.
    """
    d = np.maximum(np.asarray(dist_to_nearest_pair_km, dtype=np.float64), 0.0)
    return VAR_FLOOR_A * np.expm1(d / VAR_FLOOR_SCALE_KM) + VAR_FLOOR_B


def apply_calibration(model, pa_daily):
    """Calibrate every sensor-day; returns (pm25_cal, cal_var).

    Apply-time conventions (documented in the module docstring): dist_km = 0
    (estimate for a hypothetically colocated monitor), is_fem = 0 (FRM
    reference scale). cal_var = max(model, floor(dist_to_nearest_frm)),
    inflated for channel-reconstructed rows; NaN raw PA yields NaN for both
    outputs (no fill).
    """
    q = pa_daily.copy()
    q["pa"] = q["pa_raw"].astype(np.float64)
    q["dist_km"] = 0.0
    q["is_fem"] = 0.0
    if "year" not in q.columns:
        q["year"] = pd.to_datetime(q["date"]).dt.year.astype(int)
    if "sensor_year" not in q.columns:
        q["sensor_year"] = q["sensor_id"].astype(str) + "_" + \
            q["year"].astype(str)

    mean = model.predict(q)
    var = model.predict_var(q)
    if "dist_to_nearest_frm" in q.columns:
        d = q["dist_to_nearest_frm"].to_numpy(dtype=np.float64)
    else:
        d = np.zeros(len(q), dtype=np.float64)
    var = np.maximum(var, cal_var_floor(d))
    recon = q["channel_reconstructed"].to_numpy(dtype=np.float64) > 0
    var = np.where(recon, var * CHANNEL_RECON_VAR_FACTOR, var)
    var = np.where(np.isfinite(mean), var, np.nan)
    return mean, var


# ── LOLO validation + gate G0 ───────────────────────────────────────────────

def _metrics(y, pred):
    good = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[good], pred[good]
    if len(y) == 0:
        return {"rmse": None, "mae": None, "bias": None, "r2": None, "n": 0}
    err = pred - y
    sst = float(np.sum((y - y.mean()) ** 2))
    return {"rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)),
            "r2": float(1.0 - np.sum(err ** 2) / sst) if sst > 0 else None,
            "n": int(len(y))}


def lolo_validate(pairs, pa_daily=None, aqs_daily=None, folds=None,
                  max_dist_km=PRIMARY_PAIR_KM, num_boost_round=None,
                  seed=None, max_sites=None):
    """Leave-one-location-out over pairing AQS sites.

    For each site: refit the learned chain AND the refit linear baselines on
    all other sites' pairs, predict that site's pair-days, pool. All methods
    are scored on the identical row set (paired comparison). Returns the
    metric dict consumed by calibration_report.json, including the G0
    verdict: learned must beat BOTH published Barkjohn AND the TX-refit
    AMT-style RH+T form on LOLO RMSE and |bias|.
    """
    seed = config2.SEED if seed is None else int(seed)
    if pa_daily is None:
        pa_daily = load_pa_daily()
    if aqs_daily is None:
        aqs_daily = load_aqs_daily()
    vault = vault_site_set(folds)

    cal = build_cal_frame(pairs, pa_daily, aqs_daily, max_dist_km)
    cal = cal[~cal["site_id"].isin(vault)].reset_index(drop=True)
    assert not (set(cal["site_id"]) & vault), "vault airlock breach in LOLO"
    if len(cal) == 0:
        raise ValueError("no calibration pairs for LOLO")

    sites = (cal.groupby("site_id").size().sort_values(ascending=False)
             .index.tolist())
    if max_sites is not None:
        sites = sorted(sites[:max_sites])
    site_arr = cal["site_id"].to_numpy()
    scored = np.isin(site_arr, sites)

    n = len(cal)
    preds = {m: np.full(n, np.nan)
             for m in ("learned", "barkjohn", "barkjohn_refit", "amt_rht")}
    kinds = set()
    for s in sites:
        te = site_arr == s
        tr_frame = cal[~te].reset_index(drop=True)
        te_frame = cal[te]
        if len(tr_frame) == 0 or len(te_frame) == 0:
            continue
        model = _fit_on_frame(tr_frame, num_boost_round, seed, "learned")
        kinds.add(model.kind)
        preds["learned"][te] = model.predict(te_frame)
        preds["barkjohn"][te] = barkjohn_published(te_frame)
        os_idx = _oversample_index(tr_frame, seed)
        for name, form_cols in (("barkjohn_refit", ("pa", "rh")),
                                ("amt_rht", ("pa", "rh", "t"))):
            form = fit_linear_form(tr_frame, form_cols, sample_idx=os_idx)
            preds[name][te] = predict_linear_form(form, te_frame)

    y = cal["y"].to_numpy(dtype=np.float64)
    methods = {name: _metrics(y[scored], p[scored])
               for name, p in preds.items()}
    if any(m["n"] == 0 for m in methods.values()):
        raise ValueError("LOLO produced zero scored rows for at least one "
                         "method -- cannot evaluate gate G0")

    years = cal["year"].to_numpy()
    by_year = {}
    for yr in sorted(np.unique(years[scored]).tolist()):
        sel = scored & (years == yr)
        row = {"n": int(sel.sum())}
        for name, p in preds.items():
            good = sel & np.isfinite(p) & np.isfinite(y)
            row[name] = (float(np.mean(p[good] - y[good]))
                         if good.any() else None)
        by_year[str(int(yr))] = row

    per_site = {}
    for s in sites:
        sel = site_arr == s
        per_site[s] = {
            "n": int(sel.sum()),
            "rmse_learned": _metrics(y[sel], preds["learned"][sel])["rmse"],
            "rmse_barkjohn": _metrics(y[sel], preds["barkjohn"][sel])["rmse"],
        }

    ml, mb, ma = (methods["learned"], methods["barkjohn"], methods["amt_rht"])
    criteria = {
        "rmse_beats_barkjohn": bool(ml["rmse"] < mb["rmse"]),
        "rmse_beats_amt_rht": bool(ml["rmse"] < ma["rmse"]),
        "bias_beats_barkjohn": bool(abs(ml["bias"]) < abs(mb["bias"])),
        "bias_beats_amt_rht": bool(abs(ml["bias"]) < abs(ma["bias"])),
    }
    passed = all(criteria.values())
    fallback = min(("barkjohn_refit", "amt_rht"),
                   key=lambda m: methods[m]["rmse"])
    return {
        "max_dist_km": float(max_dist_km),
        "n_sites": int(len(sites)),
        "n_pair_days": int(scored.sum()),
        "model_kinds": sorted(kinds),
        "methods": methods,
        "by_year_bias": by_year,
        "per_site": per_site,
        "g0": {"verdict": "pass" if passed else "fail",
               "criteria": criteria,
               "fallback_form": fallback,
               "production_form": "learned" if passed else fallback},
    }


# ── Stage driver ────────────────────────────────────────────────────────────

def _weight_by_distance_band(pa_daily, cal_var):
    """Mandatory diagnostic: precision weight by distance-to-FRM band."""
    sigma2 = float(config2.SIGMA_FRM) ** 2
    d = pa_daily["dist_to_nearest_frm"].to_numpy(dtype=np.float64)
    good = np.isfinite(cal_var)
    w = sigma2 / (sigma2 + cal_var)
    bands = [(0, 5), (5, 10), (10, 25), (25, 50), (50, 100), (100, np.inf)]
    table = []
    for lo, hi in bands:
        sel = good & (d >= lo) & (d < hi)
        table.append({
            "band_km": f"{lo}-{'inf' if np.isinf(hi) else int(hi)}",
            "n_rows": int(sel.sum()),
            "n_sensors": int(pa_daily.loc[sel, "sensor_id"].nunique()),
            "mean_cal_var": float(np.mean(cal_var[sel])) if sel.any() else None,
            "mean_weight": float(np.mean(w[sel])) if sel.any() else None,
        })
    return table


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


def run_calibrate(quick=False, folds_path=None):
    dest_parquet = config2.artifact("pa_calibrated.parquet")
    dest_report = config2.artifact("calibration_report.json")
    if (os.path.exists(dest_parquet) and os.path.exists(dest_report)
            and os.environ.get("FORCE") != "1"):
        _say(f"{dest_parquet} exists (FORCE=1 to rebuild) -- skip")
        return 0

    print("[aqnet2] ── stage: calibrate ──")
    folds = load_fold_sites(folds_path)
    vault = vault_site_set(folds)
    nbr = QUICK_BOOST_ROUND if quick else NUM_BOOST_ROUND

    pairs_path = config2.artifact("colocation_pairs.parquet")
    if os.path.exists(pairs_path):
        pairs = pd.read_parquet(pairs_path)
    else:
        _say("colocation_pairs.parquet missing -- building in-process "
             "(run the colocate stage to persist it)")
        pairs = colocate.build_pairs()

    start, end = (QUICK_START, QUICK_END) if quick else (None, None)
    pa_daily = load_pa_daily(start=start, end=end)
    aqs_daily = load_aqs_daily(start=start, end=end)
    pa_daily["dist_to_nearest_frm"] = dist_to_nearest_frm(
        pa_daily, aqs_daily, exclude_sites=vault)

    # -- Gate G0 ------------------------------------------------------------
    lolo = lolo_validate(pairs, pa_daily, aqs_daily, folds,
                         max_dist_km=PRIMARY_PAIR_KM, num_boost_round=nbr,
                         max_sites=QUICK_LOLO_MAX_SITES if quick else None)
    production = lolo["g0"]["production_form"]
    model_form = "learned" if production == "learned" \
        else _FORM_BY_BASELINE[production]
    _say(f"G0 verdict: {lolo['g0']['verdict']} -- production form "
         f"'{production}'")

    sensitivity = None
    if not quick:
        sensitivity = lolo_validate(pairs, pa_daily, aqs_daily, folds,
                                    max_dist_km=SENSITIVITY_PAIR_KM,
                                    num_boost_round=nbr)

    # -- Nested refits ------------------------------------------------------
    out = pa_daily[["sensor_id", "date", "pa_raw",
                    "channel_reconstructed", "dist_to_nearest_frm"]].copy()
    fits_meta = {}
    n_inner = int(getattr(config2, "INNER_N_FOLDS", 4))
    ks = outer_fold_ids(folds)
    jobs = [("pa_cal_full", None, None)]
    jobs += [(f"pa_cal_f{k}", k, None) for k in ks]
    jobs += [(f"pa_cal_f{k}_{j}", k, j) for k in ks for j in range(n_inner)]

    cal_var_full = None
    for col, k, j in jobs:
        model = fit_calibration(pairs, pa_daily, folds, outer_k=k, inner_j=j,
                                aqs_daily=aqs_daily,
                                max_dist_km=PRIMARY_PAIR_KM,
                                num_boost_round=nbr, model_form=model_form)
        mean, var = apply_calibration(model, pa_daily)
        out[col] = mean
        if col == "pa_cal_full":
            cal_var_full = var
        fits_meta[col] = {"kind": model.meta["kind"],
                          "n_pair_days": model.meta["n_pair_days"],
                          "n_sites": model.meta["n_sites"],
                          "excluded_sites": model.meta["excluded_sites"]}
        _say(f"{col}: kind={model.meta['kind']} "
             f"pairs={model.meta['n_pair_days']} "
             f"sites={model.meta['n_sites']}")
    out["cal_var"] = cal_var_full

    order = (["sensor_id", "date", "pa_raw"]
             + [c for c, _, _ in jobs]
             + ["cal_var", "channel_reconstructed", "dist_to_nearest_frm"])
    out = out[order]
    tmp = dest_parquet + ".tmp"
    out.to_parquet(tmp, index=False)
    os.replace(tmp, dest_parquet)
    _say(f"wrote {len(out)} sensor-days x {len(jobs)} nested columns "
         f"-> {dest_parquet}")

    report = {
        "quick": bool(quick),
        "max_dist_km": PRIMARY_PAIR_KM,
        "production_form": production,
        "model_form": model_form,
        "lolo": lolo,
        "sensitivity_25km": sensitivity,
        "g0": lolo["g0"],
        "by_year_bias": lolo["by_year_bias"],
        "weight_by_distance_band": _weight_by_distance_band(
            pa_daily, cal_var_full),
        "channel_reconstructed_frac": float(
            np.mean(out["channel_reconstructed"].to_numpy() > 0)),
        "vault_sites_excluded": sorted(vault),
        "inner_fold_of_site": {
            str(k): inner_fold_of_site(folds, k) for k in ks},
        "inner_fold_source": ("folds2" if folds.get("inner_fold_of_site")
                              else "derived_fallback"),
        "fits": fits_meta,
        "columns_written": [c for c, _, _ in jobs],
        "cal_var_floor": {"a": VAR_FLOOR_A, "scale_km": VAR_FLOOR_SCALE_KM,
                          "b": VAR_FLOOR_B, "min": CAL_VAR_MIN,
                          "recon_factor": CHANNEL_RECON_VAR_FACTOR},
    }
    tmp = dest_report + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(report), fh, indent=2)
    os.replace(tmp, dest_report)
    _say(f"wrote {dest_report}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="AQNet v2 S1 Kennedy-O'Hagan PA calibration")
    ap.add_argument("--quick", action="store_true",
                    help="3-month window, fewer boosting rounds, capped LOLO")
    ap.add_argument("--folds", default=None,
                    help="path to folds2.json (default: artifacts/v2)")
    args = ap.parse_args(argv)
    return run_calibrate(quick=args.quick, folds_path=args.folds)


if __name__ == "__main__":
    sys.exit(main())
