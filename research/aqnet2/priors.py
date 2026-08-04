"""AQNet v2 T0: EPA-Downscaler-lineage debiased CTM prior (stage `priors`,
DESIGN S5).

v1 fed raw GEOS-CF / CAMS PM2.5 straight into the GBMs as covariates and
never anchored them to the FRM scale; off the PA network the models fell
back to what those raw fields plus a statewide mean could say (bare-site
AQS R^2 ~ 0, -2.7..-3.8 ug/m3 bias — DESIGN S0). T0 replaces that with the
Berrocal/Gelfand/Holland (2010) EPA downscaler form, per CTM stream c:

    y_FRM(s, t) = beta0_c(s, season) + beta1_c(s, season) * CTM_c(s, t) + eps

The beta fields live on a LOW-RANK spatial basis: a thin-plate RBF
(phi(u) = u^2 log u, u = haversine_km / RBF_SCALE_KM) over ~200 seeded
quasi-regular nodes covering TX_BBOX (a 15 x 14 cell-center lattice — the
half-cell inset is the "trim" — with seeded jitter, config2.SEED), plus an
unpenalized global intercept/slope pair. Both betas interact with the four
meteorological seasons (DJF/MAM/JJA/SON coefficient blocks; the seasonal
WLS problems decouple exactly, so each season is solved separately).
Ridge-penalized weighted least squares, weights 1 / SIGMA_FRM^2, on
non-vault, non-held-out AQS site-days ONLY — per outer fold k (fold-k
sites excluded) plus a "full" fit (vault excluded only). A season with
fewer than MIN_SEASON_ROWS rows keeps NaN coefficients: its predictions
are NaN, honestly absent, never a zero-fill.

T0(q, t) is the precision-weighted combination of the AVAILABLE streams
(per-stream precision = 1 / residual variance of that stream's own fit).
A missing stream is absent from the combination, never filled; pattern_id
is the availability bitmask (geoscf_pm25=1, cams_pm25=2,
merra2_pm25_proxy=4). Rows where no stream is available get t0 = NaN and
pattern 0 — compose.py treats absence structurally, so NaN honesty here is
load-bearing.

Import direction (frozen — audit/05-frame2 S4): priors imports frame2 at
MODULE level (load_aqs / load_gridded / gridded_join); frame2 imports
priors only lazily inside functions. Do not invert.

Artifacts (config2.artifact): prior_downscaler_f{k}.npz per outer fold,
prior_downscaler_full.npz (the stage sentinel), priors_report.json with
descriptive per-fold held-out ("LOLO-ish") per-stream R^2/bias on the
fold's own sites. NOTE: oof_tier0.npz ({oof_t0, pattern_id} per
INTERFACES) is NOT written here — pipeline2's `features` stage evaluates
each fold's downscaler at the frame rows and writes it, because the OOF
vector is defined over frame row order, which does not exist until that
stage builds the frame.

Run from anywhere (FORCE=1 re-runs past the sentinel):
    python priors.py [--quick]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import config2
import frame2  # module-level on purpose: the sanctioned cycle direction

# ── Frozen stream contract (INTERFACES / BUILD_NOTES contract 2) ────────────

STREAMS = ["geoscf_pm25", "cams_pm25", "merra2_pm25_proxy"]

# pattern_id bit for STREAMS[i] is 1 << i: geoscf=1, cams=2, merra2=4.
STREAM_BITS = {s: 1 << i for i, s in enumerate(STREAMS)}

# external_paths.json key that carries each stream's by-cell parquet; the
# stream name is the required value COLUMN inside that parquet (frame2's
# self-prefixed gridded-column convention).
STREAM_SOURCE = {"geoscf_pm25": "geoscf", "cams_pm25": "cams",
                 "merra2_pm25_proxy": "merra2"}

# ── Constants ───────────────────────────────────────────────────────────────

EARTH_R_KM = 6371.0
SIGMA_FRM = float(config2.SIGMA_FRM)
VAULT_DATE_START = getattr(config2, "VAULT_DATE_START", "2026-01-01")

NODE_GRID = (15, 14)          # (n_lon, n_lat) lattice -> 210 (~200) nodes
NODE_JITTER_FRAC = 0.25       # seeded jitter, fraction of a lattice cell
RBF_SCALE_KM = 200.0          # thin-plate length scale (conditioning only)
RIDGE_LAM = 1e-2              # relative ridge on the unit-diagonal normal eqs
GLOBAL_PEN_FACTOR = 1e-6      # near-unpenalized global intercept/slope cols
MIN_SEASON_ROWS = 10          # below this a season keeps NaN coefficients
VAR_MIN = 1e-3                # residual-variance floor (precision cap)

SEASON_NAMES = ("DJF", "MAM", "JJA", "SON")

# --quick window: fixed 3 months WELL BEFORE the vault period. frame2's
# trailing-92-day quick window would land entirely inside the vault period
# (>= 2026-01-01) and the vault-period exclusion would empty the fit.
QUICK_START, QUICK_END = "2024-07-01", "2024-09-30"
QUICK_N_FOLDS = 2


def _say(msg):
    print(f"[aqnet2] priors: {msg}")


def _banner():
    """Stage banner (BUILD_NOTES logging contract). The box-drawing chars
    fall back to ASCII on consoles that cannot encode them (Windows cp1252
    dev shells) — a log-encoding quirk must never kill the stage."""
    try:
        print("[aqnet2] ── stage: priors ──", flush=True)
    except UnicodeEncodeError:
        print("[aqnet2] -- stage: priors --", flush=True)


# ── Geometry / basis ────────────────────────────────────────────────────────

def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (broadcasting; frame2's formula)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2)
    return 2.0 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def make_basis_nodes(seed=None):
    """Seeded quasi-regular node lattice over TX_BBOX -> (M, 2) [lat, lon].

    15 x 14 cell centers (the half-cell inset from the bbox edge is the
    'trim' — no node sits on the boundary), jittered by up to a quarter
    cell with numpy default_rng(seed), clipped back to the bbox. A pure
    function of the seed, so every fold model and the full model share one
    basis and their coefficient fields are directly comparable.
    """
    seed = config2.SEED if seed is None else int(seed)
    rng = np.random.default_rng(seed)
    bb = config2.TX_BBOX
    n_lon, n_lat = NODE_GRID
    dlon = (bb["lon_max"] - bb["lon_min"]) / n_lon
    dlat = (bb["lat_max"] - bb["lat_min"]) / n_lat
    lons = bb["lon_min"] + dlon * (np.arange(n_lon) + 0.5)
    lats = bb["lat_min"] + dlat * (np.arange(n_lat) + 0.5)
    glon, glat = np.meshgrid(lons, lats)
    lat = glat.ravel() + rng.uniform(-NODE_JITTER_FRAC, NODE_JITTER_FRAC,
                                     glat.size) * dlat
    lon = glon.ravel() + rng.uniform(-NODE_JITTER_FRAC, NODE_JITTER_FRAC,
                                     glon.size) * dlon
    lat = np.clip(lat, bb["lat_min"], bb["lat_max"])
    lon = np.clip(lon, bb["lon_min"], bb["lon_max"])
    return np.column_stack([lat, lon])


def _tps_basis(lats, lons, nodes, scale_km):
    """Thin-plate RBF basis matrix (n, M+1): [1 | phi(r_1) ... phi(r_M)].

    phi(u) = u^2 log(u) with phi(0) = 0, u = haversine_km / scale_km.
    Non-finite coordinates yield NaN basis rows (and therefore NaN
    predictions), never a silent zero.
    """
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    d = _haversine_km(lats[:, None], lons[:, None],
                      nodes[None, :, 0], nodes[None, :, 1])
    u = d / float(scale_km)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.where(u > 0.0, u * u * np.log(u), 0.0)
    phi[~np.isfinite(u)] = np.nan
    return np.concatenate([np.ones((len(lats), 1)), phi], axis=1)


def _season_of(dates):
    """Meteorological season index per date: DJF=0, MAM=1, JJA=2, SON=3."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    return (np.asarray(idx.month, dtype=np.int64) % 12) // 3


def _unique_basis(lats, lons, nodes, scale_km):
    """(B_unique, inverse) so repeated coordinates evaluate the RBF once."""
    coords = np.column_stack([np.asarray(lats, dtype=np.float64),
                              np.asarray(lons, dtype=np.float64)])
    uc, inv = np.unique(coords, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    return _tps_basis(uc[:, 0], uc[:, 1], nodes, scale_km), inv


# ── Per-stream ridge-WLS fit (season-decoupled) ─────────────────────────────

def _fit_stream(Brow, season, y, ctm, w, lam, name, label):
    """One stream's downscaler: per-season ridge WLS of y on [B | B*ctm].

    The season blocks share no coefficients, so the joint block-diagonal
    WLS problem decouples exactly into four solves of 2(M+1) columns. The
    normal equations are rescaled to unit diagonal before the ridge is
    added (lam is therefore scale-free); the global intercept and global
    slope columns (0 and M+1) get lam * GLOBAL_PEN_FACTOR so a spatially
    constant beta field is essentially unshrunk — the RBF columns only
    model DEVIATIONS from the global calibration.

    Returns None when no season reaches MIN_SEASON_ROWS (stream absent),
    else {"coef" (4, 2, M+1) with NaN blocks for unfit seasons,
    "resid_var", "precision", "r2_insample", "n_rows", "seasons_fit"}.
    """
    m1 = Brow.shape[1]
    coef = np.full((4, 2, m1), np.nan)
    usable = np.isfinite(y) & np.isfinite(ctm) & np.isfinite(Brow).all(axis=1)
    seasons_fit = []

    for q in range(4):
        m = usable & (season == q)
        n_q = int(m.sum())
        if n_q < MIN_SEASON_ROWS:
            if n_q:
                _say(f"{label}/{name}: season {SEASON_NAMES[q]} has only "
                     f"{n_q} rows (< {MIN_SEASON_ROWS}) -- left NaN")
            continue
        X = np.concatenate([Brow[m], Brow[m] * ctm[m][:, None]], axis=1)
        sw = np.sqrt(w[m])
        Xw = X * sw[:, None]
        A = Xw.T @ Xw
        b = Xw.T @ (y[m] * sw)
        norm = np.sqrt(np.maximum(np.diag(A), 1e-12))
        dinv = 1.0 / norm
        As = A * dinv[:, None] * dinv[None, :]
        pen = np.full(2 * m1, float(lam))
        pen[0] = pen[m1] = float(lam) * GLOBAL_PEN_FACTOR
        As[np.diag_indices_from(As)] += pen
        beta = np.linalg.solve(As, b * dinv) * dinv
        coef[q, 0] = beta[:m1]
        coef[q, 1] = beta[m1:]
        seasons_fit.append(SEASON_NAMES[q])

    if not seasons_fit:
        return None

    # Weighted residual variance over all fitted rows -> the stream's
    # precision in the T0 combination (linear fit residuals; the >= 0 clip
    # applied at evaluation time is a physicality guard, not part of the
    # statistical model).
    r_all, w_all = [], []
    for q in range(4):
        if np.isnan(coef[q]).all():
            continue
        m = usable & (season == q)
        pred = Brow[m] @ coef[q, 0] + (Brow[m] @ coef[q, 1]) * ctm[m]
        r_all.append(y[m] - pred)
        w_all.append(w[m])
    r = np.concatenate(r_all)
    ww = np.concatenate(w_all)
    resid_var = float(max(np.average(r ** 2, weights=ww), VAR_MIN))
    yy = np.concatenate([y[usable & (season == q)] for q in range(4)
                         if not np.isnan(coef[q]).all()])
    sst = float(np.average((yy - np.average(yy, weights=ww)) ** 2,
                           weights=ww))
    r2 = 1.0 - float(np.average(r ** 2, weights=ww)) / sst if sst > 0 else None
    return {"coef": coef, "resid_var": resid_var,
            "precision": 1.0 / resid_var, "r2_insample": r2,
            "n_rows": int(len(r)), "seasons_fit": seasons_fit}


def fit_downscaler(table, exclude_sites=(), seed=None, lam=RIDGE_LAM,
                   label="full"):
    """Fit the T0 downscaler on non-vault, non-held-out AQS site-days.

    table: DataFrame [site_id, date, lat, lon, pm25_aqs, <stream cols>]
    (missing stream columns are treated as all-NaN — that stream is then
    absent from the model, never filled). exclude_sites is the vault plus,
    for fold-k models, the fold's own sites; vault-PERIOD rows
    (date >= VAULT_DATE_START) are always dropped — the one-shot vault is
    both a site set and a time window (DESIGN S2).
    """
    seed = config2.SEED if seed is None else int(seed)
    df = table.copy()
    df["site_id"] = df["site_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    ex = {str(s) for s in exclude_sites}
    if ex:
        df = df[~df["site_id"].isin(ex)]
    df = df[df["date"] < pd.Timestamp(VAULT_DATE_START)]
    df = df.dropna(subset=["pm25_aqs"]).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"fit_downscaler({label}): no AQS site-days left "
                         f"after exclusions")
    breach = set(df["site_id"]) & ex
    if breach:
        raise AssertionError(f"exclusion airlock breach in fit_downscaler"
                             f"({label}): {sorted(breach)[:5]}")

    nodes = make_basis_nodes(seed)
    B_u, inv = _unique_basis(df["lat"].to_numpy(), df["lon"].to_numpy(),
                             nodes, RBF_SCALE_KM)
    Brow = B_u[inv]
    season = _season_of(df["date"])
    y = df["pm25_aqs"].to_numpy(dtype=np.float64)
    w = np.full(len(df), 1.0 / SIGMA_FRM ** 2)

    streams, coef, precision, resid_var = [], {}, {}, {}
    per_stream_meta = {}
    for s in STREAMS:
        ctm = (df[s].to_numpy(dtype=np.float64) if s in df.columns
               else np.full(len(df), np.nan))
        n_fin = int(np.isfinite(ctm).sum())
        if n_fin == 0:
            _say(f"{label}: stream {s} has no finite values -- absent from "
                 f"this model (never filled)")
            continue
        res = _fit_stream(Brow, season, y, ctm, w, lam, s, label)
        if res is None:
            _say(f"{label}: stream {s} fit no season -- absent")
            continue
        streams.append(s)
        coef[s] = res["coef"]
        precision[s] = res["precision"]
        resid_var[s] = res["resid_var"]
        per_stream_meta[s] = {k: res[k] for k in
                              ("n_rows", "r2_insample", "resid_var",
                               "precision", "seasons_fit")}
        _say(f"{label}/{s}: n={res['n_rows']:,} seasons={res['seasons_fit']} "
             f"r2_insample={res['r2_insample']:.3f} "
             f"resid_var={res['resid_var']:.2f}")
    if not streams:
        _say(f"WARNING: {label}: NO stream could be fit -- this model "
             f"evaluates to all-NaN t0 (pattern 0 everywhere)")

    return {"nodes": nodes, "rbf_scale_km": float(RBF_SCALE_KM),
            "ridge_lam": float(lam), "streams": streams, "coef": coef,
            "precision": precision, "resid_var": resid_var,
            "meta": {"label": str(label), "seed": int(seed),
                     "n_rows": int(len(df)),
                     "n_sites": int(df["site_id"].nunique()),
                     "excluded_sites": sorted(ex),
                     "streams": per_stream_meta}}


# ── Evaluation (frozen API — frame2 calls this) ─────────────────────────────

def evaluate_prior(model, lats, lons, dates, ctm):
    """Evaluate the T0 prior at arbitrary points.

    ctm: {stream_name: f8[n]} — the already-joined CTM values at the query
    points (frame2 passes NaN vectors for absent streams). Returns
    (t0 f8[n], pattern_id i1[n], per_stream {stream: f8[n]}):

      * per_stream[s]: beta0_s(q, season) + beta1_s(q, season) * ctm_s,
        clipped at 0 (PM mass), NaN wherever the ctm value is NaN, the
        season was never fit, the coordinates are non-finite, or the
        stream is not in the model. EVERY name in STREAMS is present so
        the t0_{stream} column set is schema-stable across fold models.
      * t0: precision-weighted combination of the available streams; NaN
        when none is available (never a fill).
      * pattern_id: int8 bitmask (STREAM_BITS) of the streams that
        contributed to t0 — t0 is finite exactly where pattern_id > 0.
    """
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    n = len(lats)
    if len(lons) != n:
        raise ValueError(f"lats/lons length mismatch: {n} vs {len(lons)}")
    season = _season_of(dates)
    if len(season) != n:
        raise ValueError(f"dates length {len(season)} != n points {n}")

    fitted = [str(s) for s in model["streams"]]
    for s in fitted:
        prec = float(model["precision"][s])
        if not np.isfinite(prec) or prec <= 0:
            raise AssertionError(f"stream {s} has invalid precision {prec!r}"
                                 f" -- corrupt model")

    B_u, inv = _unique_basis(lats, lons, model["nodes"],
                             float(model["rbf_scale_km"]))

    per_stream = {}
    num = np.zeros(n)
    den = np.zeros(n)
    pattern = np.zeros(n, dtype=np.int8)
    for i, s in enumerate(STREAMS):
        vals = ctm.get(s)
        vals = (np.full(n, np.nan) if vals is None
                else np.asarray(vals, dtype=np.float64))
        if len(vals) != n:
            raise ValueError(f"ctm[{s!r}] length {len(vals)} != {n}")
        pred = np.full(n, np.nan)
        if s in fitted:
            coef = np.asarray(model["coef"][s], dtype=np.float64)
            for q in range(4):
                m = season == q
                if not m.any() or np.isnan(coef[q]).all():
                    continue
                f0 = B_u @ coef[q, 0]
                f1 = B_u @ coef[q, 1]
                pred[m] = f0[inv[m]] + f1[inv[m]] * vals[m]
            pred = np.clip(pred, 0.0, None)   # PM mass; NaN passes through
        per_stream[s] = pred
        if s in fitted:
            fin = np.isfinite(pred)
            prec = float(model["precision"][s])
            num[fin] += prec * pred[fin]
            den[fin] += prec
            pattern[fin] |= np.int8(1 << i)

    t0 = np.full(n, np.nan)
    ok = den > 0
    t0[ok] = num[ok] / den[ok]
    return t0, pattern, per_stream


# ── Persistence (npz; atomic tmp + os.replace) ──────────────────────────────

def save_model(model, path):
    """Persist one downscaler as npz: basis node coords, per-stream
    per-season beta coefficients, per-stream precision (and residual
    variance), ridge lambda, streams present."""
    payload = {
        "nodes": np.asarray(model["nodes"], dtype=np.float64),
        "rbf_scale_km": np.float64(model["rbf_scale_km"]),
        "ridge_lam": np.float64(model["ridge_lam"]),
        "streams": np.asarray(list(model["streams"]), dtype=str),
    }
    for s in model["streams"]:
        payload[f"coef__{s}"] = np.asarray(model["coef"][s], dtype=np.float64)
        payload[f"precision__{s}"] = np.float64(model["precision"][s])
        payload[f"resid_var__{s}"] = np.float64(model["resid_var"][s])
    tmp = str(path) + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)
    _say(f"saved -> {path} ({len(model['streams'])} streams)")


def load_model(path):
    """Rebuild a model dict from save_model's npz (evaluate-ready)."""
    with np.load(path, allow_pickle=False) as z:
        streams = [str(s) for s in z["streams"]]
        model = {"nodes": np.asarray(z["nodes"], dtype=np.float64),
                 "rbf_scale_km": float(z["rbf_scale_km"]),
                 "ridge_lam": float(z["ridge_lam"]),
                 "streams": streams,
                 "coef": {s: np.asarray(z[f"coef__{s}"], dtype=np.float64)
                          for s in streams},
                 "precision": {s: float(z[f"precision__{s}"])
                               for s in streams},
                 "resid_var": {s: float(z[f"resid_var__{s}"])
                               for s in streams},
                 "meta": {"path": str(path)}}
    return model


def load_fold_models():
    """{int k: model, "full": model} from the artifacts dir; {} when absent.

    Missing individual files are skipped (a partial dict is legitimate mid
    --quick runs); frame2 refuses to substitute the full model inside a
    fold, so a missing fold npz degrades to NaN t0 there, never to a leak.
    """
    models = {}
    for k in range(int(config2.OUTER_N_FOLDS)):
        p = config2.artifact(f"prior_downscaler_f{k}.npz")
        if os.path.exists(p):
            models[k] = load_model(p)
    p = config2.artifact("prior_downscaler_full.npz")
    if os.path.exists(p):
        models["full"] = load_model(p)
    return models


# ── Fitting-data assembly (external_paths + folds2 Phase-1 + AQS) ──────────

def load_fit_table(external_paths=None, start=None, end=None):
    """AQS site-days with each CTM stream joined at the site coordinates.

    Columns: site_id(str), date(datetime64[ns]), lat, lon, pm25_aqs, plus
    one column per STREAMS entry (all-NaN when that product is missing —
    the stream is then absent from the fit, never filled). Joins go
    through frame2.load_gridded + frame2.gridded_join (nearest 0.1-degree
    cell, same day) so T0's fitting-time CTM values match the frame's
    serving-time columns by construction.
    """
    ext = dict(frame2._DEFAULT_EXTERNAL)
    ext.update({k: v for k, v in (external_paths or {}).items() if v})

    aqs_path = ext.get("aqs")
    if not aqs_path or not os.path.exists(aqs_path):
        raise FileNotFoundError("external_paths['aqs'] is required to fit "
                                "the T0 downscaler (FRM truth rows)")
    aqs = frame2.load_aqs(aqs_path, start, end)
    df = aqs[["site_id", "date", "pm25_aqs", "lat", "lon"]].copy()
    # AQS parquet dates may be datetime64[us]; joins require one resolution.
    df["date"] = df["date"].astype("datetime64[ns]")

    for s in STREAMS:
        key = STREAM_SOURCE[s]
        path = ext.get(key)
        if not path or not os.path.exists(path):
            _say(f"stream {s}: no '{key}' product ({path}) -- unavailable")
            df[s] = np.nan
            continue
        g = frame2.load_gridded(path)
        if s not in g.columns:
            _say(f"stream {s}: column missing from {path} -- unavailable")
            df[s] = np.nan
            continue
        g = g[["lat", "lon", "date", s]].copy()
        g["date"] = g["date"].astype("datetime64[ns]")
        df = frame2.gridded_join(df, g, value_cols=[s])
        if s not in df.columns:            # empty product edge case
            df[s] = np.nan
        cov = float(np.isfinite(df[s].to_numpy(dtype=np.float64)).mean())
        _say(f"stream {s}: joined from {key}, coverage {cov * 100:.1f}%")
    return df


def _load_fold_sites(path=None):
    """folds2.json Phase-1 (raw json): vault_sites + outer_fold_of_site.

    priors runs before the training frame exists, so it cannot verify the
    content hash (folds2.load_folds's job); it consumes only the
    site-level keys — the same convention as calibrate.load_fold_sites.
    Refuses to run without folds: fold-k prior models fit on fold-k sites
    would leak those sites through every t0 feature downstream.
    """
    p = path or config2.artifact("folds2.json")
    if not os.path.exists(p):
        raise SystemExit(
            f"[aqnet2] priors: folds2.json not found at {p} -- build the "
            "fold system (folds2.py) first; un-nested T0 priors would be "
            "a leak, refusing to run.")
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ── Descriptive held-out metrics (priors_report.json) ───────────────────────

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


def holdout_metrics(model, df):
    """Score a model on held-out AQS rows (descriptive, per stream +
    combined + availability-pattern counts)."""
    ctm = {s: (df[s].to_numpy(dtype=np.float64) if s in df.columns
               else np.full(len(df), np.nan)) for s in STREAMS}
    t0, pattern, per_stream = evaluate_prior(
        model, df["lat"].to_numpy(), df["lon"].to_numpy(), df["date"], ctm)
    y = df["pm25_aqs"].to_numpy(dtype=np.float64)
    vals, cnts = np.unique(pattern, return_counts=True)
    return {"combined": _metrics(y, t0),
            "per_stream": {s: _metrics(y, p) for s, p in per_stream.items()},
            "pattern_counts": {str(int(v)): int(c)
                               for v, c in zip(vals, cnts)}}


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


# ── Stage driver ────────────────────────────────────────────────────────────

def run_priors(quick=False, external_paths_path=None, folds_path=None,
               lam=RIDGE_LAM):
    dest_full = config2.artifact("prior_downscaler_full.npz")
    dest_report = config2.artifact("priors_report.json")
    if (os.path.exists(dest_full) and os.path.exists(dest_report)
            and os.environ.get("FORCE") != "1"):
        _say(f"{dest_full} exists (FORCE=1 to rebuild) -- skip")
        return 0

    _banner()
    folds = _load_fold_sites(folds_path)
    vault = {str(s) for s in folds.get("vault_sites", [])}
    outer = {str(s): int(k)
             for s, k in folds.get("outer_fold_of_site", {}).items()}
    ks = sorted({k for k in outer.values() if k >= 0})
    if not ks:
        raise SystemExit("[aqnet2] priors: folds2.json has no "
                         "outer_fold_of_site assignments -- refusing to run")
    start, end = config2.DATE_START, config2.DATE_END
    if quick:
        ks = ks[:QUICK_N_FOLDS]
        start, end = QUICK_START, QUICK_END
        _say(f"--quick: folds {ks} + full, window {start}..{end}")

    ext = {}
    p = external_paths_path or config2.artifact("external_paths.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as fh:
            ext = {k: v for k, v in json.load(fh).items() if v}
    else:
        _say(f"external_paths.json not found at {p} -- committed defaults "
             f"only (geoscf/merra2 streams will be unavailable)")

    table = load_fit_table(ext, start, end)
    _say(f"fit table: {len(table):,} AQS site-days, "
         f"{table['site_id'].nunique()} sites, vault={len(vault)} sites")

    report_folds = {}
    for k in ks:
        fold_sites = {s for s, f in outer.items() if f == k}
        model = fit_downscaler(table, exclude_sites=vault | fold_sites,
                               lam=lam, label=f"f{k}")
        save_model(model, config2.artifact(f"prior_downscaler_f{k}.npz"))
        hold = table[table["site_id"].astype(str).isin(fold_sites)
                     & (table["date"] < pd.Timestamp(VAULT_DATE_START))]
        entry = {"n_train_rows": model["meta"]["n_rows"],
                 "n_train_sites": model["meta"]["n_sites"],
                 "n_holdout_rows": int(len(hold)),
                 "n_holdout_sites": int(hold["site_id"].nunique()),
                 "streams_fit": model["streams"]}
        if len(hold):
            entry.update(holdout_metrics(model, hold))
            comb = entry["combined"]
            _say(f"fold {k} held-out: n={comb['n']:,} "
                 f"r2={comb['r2']} bias={comb['bias']}")
        else:
            _say(f"fold {k}: no held-out rows in window -- metrics skipped")
        report_folds[str(k)] = entry

    full_model = fit_downscaler(table, exclude_sites=vault, lam=lam,
                                label="full")
    save_model(full_model, dest_full)

    report = {
        "quick": bool(quick),
        "seed": int(config2.SEED),
        "window": [start, end],
        "streams": list(STREAMS),
        "stream_bits": dict(STREAM_BITS),
        "n_nodes": int(len(full_model["nodes"])),
        "rbf_scale_km": float(RBF_SCALE_KM),
        "ridge_lam": float(lam),
        "min_season_rows": int(MIN_SEASON_ROWS),
        "vault_sites_excluded": sorted(vault),
        "folds": report_folds,
        "full": full_model["meta"],
        "note": ("held-out metrics are descriptive (LOLO-ish, fold-k sites "
                 "scored by the fold-k model). oof_tier0.npz is written by "
                 "pipeline2's features stage, which evaluates these fold "
                 "models at the frame rows -- not by this stage."),
    }
    tmp = dest_report + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(report), fh, indent=2)
    os.replace(tmp, dest_report)
    _say(f"wrote {dest_report}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="AQNet v2 T0 EPA-Downscaler-lineage debiased CTM prior")
    ap.add_argument("--quick", action="store_true",
                    help="3-month window, first 2 outer folds + full")
    ap.add_argument("--external-paths", default=None,
                    help="path to external_paths.json (default: artifacts/v2)")
    ap.add_argument("--folds", default=None,
                    help="path to folds2.json (default: artifacts/v2)")
    ap.add_argument("--lam", type=float, default=RIDGE_LAM,
                    help="relative ridge penalty on the RBF coefficients")
    args = ap.parse_args(argv)
    return run_priors(quick=args.quick,
                      external_paths_path=args.external_paths,
                      folds_path=args.folds, lam=args.lam)


if __name__ == "__main__":
    sys.exit(main())
