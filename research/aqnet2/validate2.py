"""
validate2.py
AQNet v2 pre-registered validation battery (stage `validate`, DESIGN SS11).

Everything reported here was pre-registered before any tier trained; the
battery only READS artifacts produced by earlier stages and never refits a
model. v1's evidence shaped every block:

  P1  Held-out-AQS R2/RMSE/MAE/bias on the outer folds, for the composite
      and the cumulative ladder (T0, T1, T1+T2, T1+T2+T3, composite), with
      site-cluster bootstrap CIs. v1 reported a pooled LOSO R2 (~93%%
      temporal by construction) as its headline; here the headline is the
      spatially honest outer-fold number and the LOSO figure is retained
      ONLY as a labeled diagnostic. By-year bias flatness and the per-year
      attenuation slope b (WLS y ~ a + b*pred) make v1's -2.7..-3.8 ug/m3
      AQS bias mechanism permanently visible; the per-fold and per-block
      tables name the worst block (v1's block-3 R2 of -1.8 stays visible).
  P2  Between-site R2 on per-site means + Spearman rank-rho — the exposure
      assessment question — in a with-network arm (composite OOF) and a
      BARE-SITE arm: features rebuilt at held-out AQS site-days with every
      PA sensor within 5 km of the site excluded from all pools
      (frame2.build_point_features + build_pools with exclude_units), scored
      with the fold's T1 model + T0 while the deep tiers are structurally
      closed (their serve-time residuals do not exist off-network). The
      bare-site number therefore covers the T0+T1 core and is a LOWER BOUND
      for the full ladder — the only number that answers "can the system
      rank unmonitored locations".
  P3  Exceedance precision/recall/F1 at FRM labels from oof_exceed.npz at
      both NAAQS thresholds; channel_reconstructed rows excluded (their
      labels are synthetic above 20 ug/m3 — BUILD_NOTES scope decision 1).
  P4  Interval coverage/width, SITE-level (honest n = sites, not the 77k
      dependent rows v1 counted), per coverage-density bin and pooled, with
      the pre-registered ship-window verdict: coverage in [0.88, 0.93].

  VAULT  The one-shot second sample. Evaluated ONLY here, exactly once,
      gated by a vault_opened.json marker: when the marker is absent (and
      the run is not --quick and the serving hook is available) the vault
      sites + vault period are scored through the serving path (bag of T1
      fold models over the frame's deployment-view features; deep tiers
      structurally closed at serve time) and the marker records git sha +
      time.time(). When the marker exists this stage REFUSES to recompute
      and reports the cached metrics_vault.json verbatim.

  Baselines  All paired against the composite on identical rows via
      compose.admission_test — v1's _paired_delta_r2_ci template promoted
      to one-sided margins from power_analysis.json: per-day ordinary
      kriging on calibrated PA (validation.krige_to_sites, train = PA
      only), T0 alone, T1 itself, the T1 candidate loser(s) (per_model_*
      arrays in oof_tier1.npz), persistence (site lag-1 FRM), site
      climatology (per-site train-period mean, scored post-cutoff only),
      and raw + mean-debiased CTM streams.

  Structural audits
      monotone_report.json  composite must be BIT-identical to T1 on every
          row where no gate opened (tier_mask row empty) — the passthrough
          guarantee compose.apply_gates promises, verified on the shipped
          arrays, not on a synthetic test.
      parity_report.json    serving-path spot check: a seeded 2,000-row
          sample of inner site-days is rebuilt through
          frame2.build_point_features against deployment-view pools and
          must match the frame's stored features bit-for-bit; plus the
          gates.json re-application over the tier npzs must reproduce
          oof_composite.npz.

v1 reuse is a LAZY import of research/aqnet/validation.py (metrics,
bootstrap_ci, morans_i_daily, aqi_category_metrics, strata_metrics,
spatial_temporal_r2, krige_to_sites) — pure array functions, audited safe.
When the v1 tree or sklearn is absent those diagnostics degrade to absent
with a printed message; the primary P1/P2 numbers use local numpy-only
implementations so the battery never silently thins its headline.

Outputs (config2.artifact): SUMMARY.md, metrics_outer.json,
metrics_vault.json, metrics_baselines.json, metrics_temporal.json,
metrics_strata.json, permutation_report.json, monotone_report.json,
parity_report.json. Sentinel: SUMMARY.md + parity_report.json (FORCE=1
re-runs).

Run:
    python validate2.py [--quick]
"""
import os
import re
import json
import time
import argparse

import numpy as np
import pandas as pd

import config2
import compose

# ── Constants ───────────────────────────────────────────────────────────────

VAULT_DATE_START = getattr(config2, "VAULT_DATE_START", "2026-01-01")
SHIP_COVERAGE_WINDOW = (0.88, 0.93)     # pre-registered site-level window
CTM_STREAMS = ("geoscf_pm25", "cams_pm25", "merra2_pm25_proxy")
PARITY_SAMPLE_N = 2000
_SUMMARY_MAX_ROWS = 160

# Candidate gates.json tier keys (the gates stage owns the exact spelling;
# scanning candidates keeps the battery robust to either convention and the
# resolved key is recorded in the report).
_TIER_KEYS = {2: ("tier2", "t2", "T2", "2", "graph", "graph_res"),
              3: ("tier3", "t3", "T3", "3", "field", "field_res")}

_QUICK_N_BOOT = 200
_QUICK_KRIGE_DAYS = 25
_QUICK_BARE_SITES_PER_FOLD = 2
_QUICK_PARITY_N = 200

_V1_VAL = None
_V1_VAL_TRIED = False


def _say(msg):
    print(f"[aqnet2] {msg}", flush=True)


# ── JSON helpers (v1 pipeline_colab idioms) ─────────────────────────────────

def _jsonable(obj):
    """Recursively convert numpy scalars / NaN to JSON-safe values."""
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
    return obj


def _write_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(obj), fh, indent=2)
    os.replace(tmp, path)
    _say(f"wrote {path}")


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ── v1 validation.py lazy import (audited safe: pure array functions) ──────

def _v1():
    """research/aqnet/validation.py, imported once, or None with a printed
    degradation message (missing tree / missing sklearn)."""
    global _V1_VAL, _V1_VAL_TRIED
    if _V1_VAL_TRIED:
        return _V1_VAL
    _V1_VAL_TRIED = True
    import sys
    for p in (config2.V1_DIR, config2.PIPELINE_DIR, config2.DL_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import validation as _val
        _V1_VAL = _val
    except Exception as e:
        _say(f"validate: v1 validation.py unavailable ({e}) — bootstrap CIs, "
             f"Moran's I, strata, AQI and kriging baselines degrade to absent")
        _V1_VAL = None
    return _V1_VAL


# ── Local metric primitives (numpy-only; primaries never degrade) ───────────

def _metrics(y, pred):
    """{"r2","rmse","mae","bias","n"} over jointly finite rows (v1 shape)."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(p)
    n = int(ok.sum())
    if n == 0:
        nan = float("nan")
        return {"r2": nan, "rmse": nan, "mae": nan, "bias": nan, "n": 0}
    yy, pp = y[ok], p[ok]
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    r2 = 1.0 - float(np.sum((pp - yy) ** 2)) / ss_tot if ss_tot > 0 \
        else float("nan")
    return {"r2": r2,
            "rmse": float(np.sqrt(np.mean((pp - yy) ** 2))),
            "mae": float(np.mean(np.abs(pp - yy))),
            "bias": float(np.mean(pp - yy)),
            "n": n}


def _bootstrap_ci(y, pred, clusters, n_boot):
    """v1 site-cluster bootstrap CI when importable, else None."""
    val = _v1()
    if val is None:
        return None
    try:
        return val.bootstrap_ci(np.asarray(y, float), np.asarray(pred, float),
                                n_boot=n_boot, seed=0,
                                cluster=np.asarray(clusters))
    except Exception as e:
        _say(f"validate: bootstrap_ci failed ({e})")
        return None


def _wls_line(y, x, w):
    """WLS y ~ a + b*x. Returns {"a","b","n"} or None (<3 usable rows)."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(w) & (w > 0)
    if int(ok.sum()) < 3:
        return None
    y, x, w = y[ok], x[ok], w[ok]
    sw = w.sum()
    xm = float(np.sum(w * x) / sw)
    ym = float(np.sum(w * y) / sw)
    ssx = float(np.sum(w * (x - xm) ** 2))
    if ssx <= 1e-12:
        return None
    b = float(np.sum(w * (x - xm) * (y - ym)) / ssx)
    return {"a": ym - b * xm, "b": b, "n": int(len(y))}


def _spearman(a, b):
    """Spearman rank-rho over jointly finite rows (pandas average ranks)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3:
        return float("nan")
    ra = pd.Series(a[ok]).rank(method="average").to_numpy(dtype=np.float64)
    rb = pd.Series(b[ok]).rank(method="average").to_numpy(dtype=np.float64)
    if ra.std() <= 0 or rb.std() <= 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _between_site(y, pred, sites):
    """Per-site-mean R2 + Spearman rho — the P2 exposure-ranking metric."""
    y = np.asarray(y, float)
    p = np.asarray(pred, float)
    ok = np.isfinite(y) & np.isfinite(p)
    if int(ok.sum()) < 3:
        return {"between_site_r2": float("nan"),
                "spearman_rho": float("nan"), "n_sites": 0, "n_rows": 0}
    df = pd.DataFrame({"s": np.asarray(sites)[ok].astype(str),
                       "y": y[ok], "p": p[ok]})
    g = df.groupby("s", sort=True).mean()
    my = g["y"].to_numpy()
    mp = g["p"].to_numpy()
    ss = float(np.sum((my - my.mean()) ** 2))
    r2 = 1.0 - float(np.sum((mp - my) ** 2)) / ss if ss > 0 else float("nan")
    return {"between_site_r2": r2, "spearman_rho": _spearman(mp, my),
            "n_sites": int(len(g)), "n_rows": int(ok.sum())}


def _prf(label, flag):
    """Precision/recall/F1 from boolean arrays (NaN-honest denominators)."""
    tp = int(np.sum(flag & label))
    fp = int(np.sum(flag & ~label))
    fn = int(np.sum(~flag & label))
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * prec * rec / (prec + rec)
          if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0
          else float("nan"))
    return {"precision": prec, "recall": rec, "f1": f1,
            "n_true_pos_labels": tp + fn, "n_pred_pos": tp + fp}


def _bits_equal(a, b):
    """(all_equal, n_diff) — BIT-level comparison of two f8 arrays (NaN
    payloads and signed zeros included; float == would call NaN != NaN)."""
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    b = np.ascontiguousarray(np.asarray(b, dtype=np.float64))
    if a.shape != b.shape:
        return False, int(max(a.size, b.size))
    diff = a.view(np.int64) != b.view(np.int64)
    return not bool(diff.any()), int(diff.sum())


# ── Artifact loading ────────────────────────────────────────────────────────

def _load_npz(path, required=False):
    """npz -> {key: array} (metadata keys pass through; row alignment is
    checked per-key by _check_rows at the point of use)."""
    if not os.path.exists(path):
        if required:
            raise SystemExit(f"[aqnet2] required artifact missing: {path} — "
                             f"run the producing stage first")
        _say(f"validate: optional artifact absent: {os.path.basename(path)}")
        return None
    out = {}
    with np.load(path, allow_pickle=False) as z:
        for k in z.files:
            out[k] = z[k]
    return out


def _check_rows(npz, keys, n_rows, label):
    """Row-length mismatch on a per-row key is a HARD error — silent
    misalignment corrupts every number downstream (v1 audit gotcha 10)."""
    if npz is None:
        return npz
    for k in keys:
        if k not in npz:
            raise ValueError(f"{label} lacks required key '{k}' "
                             f"(keys: {sorted(npz)})")
        if npz[k].shape[0] != n_rows:
            raise ValueError(f"{label}[{k}] has length {npz[k].shape[0]}, "
                             f"expected {n_rows}")
    return npz


def _load_frame(path):
    frame = pd.read_parquet(path)
    # AQS dates may arrive as datetime64[us] (pandas 3 audit note) —
    # normalize everything to [ns] midnight once, here.
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["date"] = frame["date"].astype("datetime64[ns]")
    frame["unit_id"] = frame["unit_id"].astype(str)
    return frame.reset_index(drop=True)


def _load_folds(path, frame):
    """folds2.load_folds (content-hash verified) when importable; degraded
    json.load + row-count check otherwise (printed)."""
    try:
        import folds2
        return folds2.load_folds(path, frame)
    except ImportError as e:
        _say(f"validate: folds2 unavailable ({e}) — loading folds2.json "
             f"WITHOUT content-hash verification (row-count check only)")
    folds = _read_json(path)
    if folds is None:
        raise SystemExit(f"[aqnet2] required artifact missing: {path}")
    if int(folds.get("n_rows", -1)) != len(frame):
        raise ValueError(f"folds2.json n_rows {folds.get('n_rows')} != frame "
                         f"rows {len(frame)}")
    return folds


def _stratum_from_count(count50):
    """Coverage-density stratum ids from nbr_pacal_count_50km — the SAME
    binning the gates stage uses (BUILD_NOTES frozen contract 4):
    0 -> no PA within 50 km, 1 -> 1-3 sensors, 2 -> >=4."""
    c = np.asarray(count50, dtype=np.float64)
    c = np.where(np.isfinite(c), c, 0.0)
    out = np.zeros(len(c), dtype=np.int64)
    out[(c >= 1) & (c <= 3)] = 1
    out[c >= 4] = 2
    return out


def _find_tier_gates(gates, tier_no):
    """(key, per-tier gates subdict) or (None, None)."""
    if not gates:
        return None, None
    for k in _TIER_KEYS[tier_no]:
        if k in gates:
            return k, gates[k]
    return None, None


# ── Serving hooks (skeleton T1 bag / priors T0) ─────────────────────────────

def _skeleton_hook():
    """Optional serving protocol: skeleton.load_fold_models() -> {k: bundle}
    plus a per-bundle predictor. Accepted predictors, in order:
    skeleton.predict_fold(bundle, df) / skeleton.predict_points(bundle, df),
    else a models_tabular fit_full-shaped bundle scored via v1
    models_tabular.predict_full. df is the FULL row DataFrame (features +
    lat/lon/date/unit_id — GPBoost bundles need coordinates). Returns
    {"models": {...}, "predict": callable} or None with a printed message."""
    try:
        import skeleton
    except Exception as e:
        _say(f"validate: skeleton module unavailable ({e}) — T1 serving "
             f"hooks disabled (vault deferred, bare-site arm degrades to T0)")
        return None
    load = getattr(skeleton, "load_fold_models", None)
    predfn = (getattr(skeleton, "predict_fold", None)
              or getattr(skeleton, "predict_points", None))
    if load is None:
        _say("validate: skeleton.load_fold_models missing — T1 serving "
             "hooks disabled")
        return None
    try:
        models = load()
    except Exception as e:
        _say(f"validate: skeleton.load_fold_models failed ({e}) — T1 "
             f"serving hooks disabled")
        return None
    if not models:
        _say("validate: no skeleton fold models on disk — T1 serving hooks "
             "disabled")
        return None

    def _predict(bundle, df):
        if predfn is not None:
            return np.asarray(predfn(bundle, df), dtype=np.float64)
        if isinstance(bundle, dict) and "models" in bundle:
            import sys
            for p in (config2.V1_DIR, config2.PIPELINE_DIR):
                if p not in sys.path:
                    sys.path.insert(0, p)
            import models_tabular
            return np.asarray(models_tabular.predict_full(bundle, df),
                              dtype=np.float64)
        raise RuntimeError("no usable predictor for skeleton bundle")

    return {"models": models, "predict": _predict}


def _priors_models():
    try:
        import priors
        return priors.load_fold_models() or {}
    except Exception as e:
        _say(f"validate: priors fold models unavailable ({e})")
        return {}


def _bag_mean(preds):
    """Row-wise mean over a list of prediction arrays, NaN where no member
    is finite (never a fill)."""
    stack = np.vstack(preds)
    fin = np.isfinite(stack)
    cnt = fin.sum(axis=0)
    ssum = np.where(fin, stack, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = ssum / cnt
    out[cnt == 0] = np.nan
    return out


# ── Cumulative ladder (compose.apply_gates over the tier npzs) ─────────────

def _apply_tier(inc, tier_npz, tier_gates, stratum_id, label, notes):
    """One ladder rung. Defensive against incumbent holes: rows whose
    incumbent is non-finite are structurally closed (recorded), because
    apply_gates rightly refuses to modify a non-finite incumbent. Contract
    breaches in the tier npz (avail == 1 with non-finite oof_r) are recorded
    loudly and closed — the battery reports, it does not crash."""
    inc = np.asarray(inc, dtype=np.float64)
    res = np.asarray(tier_npz["oof_r"], dtype=np.float64)
    avail = np.asarray(tier_npz["avail"]).astype(bool)
    pat = np.asarray(tier_npz["pattern_id"])
    breach = avail & ~np.isfinite(res)
    if breach.any():
        notes.append(f"{label}: CONTRACT BREACH — {int(breach.sum())} rows "
                     f"with avail==1 but non-finite oof_r; closed for the "
                     f"ladder (upstream bug, investigate)")
    hole = avail & ~np.isfinite(inc)
    if hole.any():
        notes.append(f"{label}: {int(hole.sum())} available rows have a "
                     f"non-finite incumbent (vault/never-scored rows); "
                     f"structurally closed")
    av = avail & np.isfinite(inc) & np.isfinite(res)
    res2 = np.where(av, res, np.nan)
    return compose.apply_gates(inc, res2, av, pat, stratum_id, tier_gates)


def _cumulative_ladder(npz1, npzs_deep, gates, stratum_id, comp, notes):
    """Ordered {arm: prediction array} — T1, +T2, +T3, composite."""
    arms = {}
    t1 = np.asarray(npz1["oof"], dtype=np.float64)
    arms["t1"] = t1
    cur = t1
    for tier_no in (2, 3):
        npz = npzs_deep.get(tier_no)
        key, tg = _find_tier_gates(gates, tier_no)
        arm = "t1_t2" if tier_no == 2 else "t1_t2_t3"
        if npz is None:
            notes.append(f"tier{tier_no}: oof npz absent — ladder rung "
                         f"skipped (arm carries the previous rung)")
            arms[arm] = cur
            continue
        if tg is None:
            notes.append(f"tier{tier_no}: no gates.json entry — every gate "
                         f"closed, rung is exact passthrough")
            arms[arm] = cur
            continue
        notes.append(f"tier{tier_no}: gates.json key '{key}'")
        cur = _apply_tier(cur, npz, tg, stratum_id, f"tier{tier_no}", notes)
        arms[arm] = cur
    arms["composite"] = np.asarray(comp["oof_final"], dtype=np.float64)
    return arms


# ── P1 / P2 / P3 / P4 ──────────────────────────────────────────────────────

def _p1_outer(frame, folds, arms, npz0, heldout, outer, n_boot):
    y = frame["y"].to_numpy(dtype=np.float64)
    w = frame["w"].to_numpy(dtype=np.float64)
    sites = frame["unit_id"].to_numpy()
    years = frame["date"].dt.year.to_numpy()
    h = heldout
    val = _v1()

    ladder = {}
    full_arms = {}
    if npz0 is not None:
        full_arms["t0"] = np.asarray(npz0["oof_t0"], dtype=np.float64)
    full_arms.update(arms)
    for name, pred in full_arms.items():
        entry = {"metrics": _metrics(y[h], pred[h]),
                 "bootstrap_ci": _bootstrap_ci(y[h], pred[h], sites[h],
                                               n_boot)}
        if val is not None:
            try:
                entry["spatial_temporal"] = val.spatial_temporal_r2(
                    y[h], pred[h], sites[h])
            except Exception as e:
                _say(f"validate: spatial_temporal_r2 failed for {name} ({e})")
        ladder[name] = entry

    compv = full_arms["composite"]
    by_year = {}
    for yr in sorted(np.unique(years[h]).tolist()):
        m = h & (years == yr)
        row = _metrics(y[m], compv[m])
        att = _wls_line(y[m], compv[m], w[m])
        row["attenuation_a"] = att["a"] if att else None
        row["attenuation_b"] = att["b"] if att else None
        by_year[int(yr)] = row

    per_fold = {}
    outer_ids = sorted(int(v) for v in np.unique(outer[h]))
    for k in outer_ids:
        m = h & (outer == k)
        row = _metrics(y[m], compv[m])
        row["n_sites"] = int(pd.unique(sites[m]).size)
        per_fold[f"outer_{k}"] = row
    worst_outer = _worst_named(per_fold)

    per_block = {}
    blocks = np.asarray(folds.get("spatial_block_fold", []), dtype=np.int64) \
        if folds.get("spatial_block_fold") is not None else None
    if blocks is not None and len(blocks) == len(frame):
        for b in sorted(int(v) for v in np.unique(blocks[h]) if v >= 0):
            m = h & (blocks == b)
            row = _metrics(y[m], compv[m])
            row["n_sites"] = int(pd.unique(sites[m]).size)
            per_block[f"block_{b}"] = row
    worst_block = _worst_named(per_block)

    return {"ladder": ladder, "by_year": by_year,
            "per_outer_fold": per_fold, "worst_outer_fold": worst_outer,
            "per_spatial_block": per_block, "worst_spatial_block": worst_block}


def _worst_named(table):
    """Name of the lowest-R2 entry — the worst block is NAMED, per DESIGN
    SS11 (v1's block-3 R2 of -1.8 must stay visible)."""
    worst, worst_r2 = None, np.inf
    for name, row in table.items():
        r2 = row.get("r2")
        if r2 is not None and np.isfinite(r2) and r2 < worst_r2:
            worst, worst_r2 = name, r2
    return worst


def _p2_with_network(frame, arms, npz0, heldout):
    y = frame["y"].to_numpy(dtype=np.float64)
    sites = frame["unit_id"].to_numpy()
    out = {}
    for name in ("composite", "t1"):
        out[name] = _between_site(y[heldout], arms[name][heldout],
                                  sites[heldout])
    if npz0 is not None:
        out["t0"] = _between_site(
            y[heldout], np.asarray(npz0["oof_t0"], float)[heldout],
            sites[heldout])
    return out


def _p3_exceedance(frame, exceed_npz, exceed_model, heldout):
    """Exceedance P/R/F1 at FRM labels from the exceed head's calibrated
    OOF probabilities, both thresholds, channel_reconstructed excluded."""
    if exceed_npz is None:
        return {"absent": True,
                "note": "oof_exceed.npz not produced (exceed stage)"}
    y = frame["y"].to_numpy(dtype=np.float64)
    recon = np.zeros(len(frame), dtype=bool)
    if "channel_reconstructed" in frame.columns:
        cr = pd.to_numeric(frame["channel_reconstructed"],
                           errors="coerce").to_numpy(dtype=np.float64)
        recon = np.isfinite(cr) & (cr > 0)
    rows = heldout & ~recon & np.isfinite(y)

    probs = _scan_prob_keys(exceed_npz)
    if not probs:
        return {"absent": True,
                "note": f"no prob_* keys recognized in oof_exceed.npz "
                        f"(keys: {sorted(exceed_npz.keys())})"}
    out = {"n_rows": int(rows.sum()),
           "n_excluded_channel_reconstructed": int((heldout & recon).sum())}
    for thr, (key, prob) in sorted(probs.items()):
        tau, tau_src = _frozen_tau(exceed_model, thr)
        p = np.asarray(prob, dtype=np.float64)
        if len(p) != len(frame):
            raise ValueError(f"oof_exceed.npz[{key}] has length {len(p)}, "
                             f"expected {len(frame)}")
        ok = rows & np.isfinite(p)
        label = y[ok] > thr
        flag = p[ok] >= tau
        entry = _prf(label, flag)
        entry.update({"threshold_ugm3": thr, "prob_key": key,
                      "decision_tau": tau, "tau_source": tau_src,
                      "n_scored": int(ok.sum())})
        out[f"thr_{thr:g}"] = entry
    return out


def _scan_prob_keys(exceed_npz):
    """{threshold: (key, array)} by scanning npz keys for prob_* names."""
    out = {}
    for key in exceed_npz:
        m = re.search(r"prob[a-z]*[_]?([0-9]+(?:[._][0-9]+)?)", key.lower())
        if not m:
            continue
        try:
            v = float(m.group(1).replace("_", "."))
        except ValueError:
            continue
        for thr in config2.EXCEED_THRESHOLDS:
            if abs(v - thr) < 0.5 and thr not in out:
                out[thr] = (key, exceed_npz[key])
    return out


def _frozen_tau(exceed_model, thr):
    """The frozen decision threshold for P(y > thr); 0.5 fallback printed."""
    if exceed_model:
        for name in ("frozen_threshold", "threshold", "tau",
                     "decision_threshold"):
            v = exceed_model.get(name)
            if isinstance(v, dict):
                for kk in (str(thr), f"{thr:g}", f"thr_{thr:g}"):
                    if kk in v and isinstance(v[kk], (int, float)):
                        return float(v[kk]), f"exceed_model.{name}.{kk}"
            elif isinstance(v, (int, float)):
                return float(v), f"exceed_model.{name}"
    _say(f"validate: no frozen threshold found for {thr:g} — using 0.5")
    return 0.5, "default_0.5"


def _p4_intervals(frame, q_npz, uq_params, heldout, stratum_id):
    """Site-level interval coverage/width, per coverage bin + pooled, with
    the pre-registered [0.88, 0.93] ship-window verdict."""
    if q_npz is None:
        return {"absent": True,
                "note": "quantile_oof.npz not produced (uq stage)"}
    lo, hi, keys = _quantile_bounds(q_npz)
    if lo is None or hi is None:
        return {"absent": True,
                "note": f"no lower/upper quantile keys recognized "
                        f"(keys: {sorted(q_npz.keys())})"}
    y = frame["y"].to_numpy(dtype=np.float64)
    sites = frame["unit_id"].to_numpy()
    delta_map, delta_src = _uq_delta(uq_params)

    lo = np.asarray(lo, dtype=np.float64).copy()
    hi = np.asarray(hi, dtype=np.float64).copy()
    if len(lo) != len(frame) or len(hi) != len(frame):
        raise ValueError(f"quantile_oof.npz arrays have length "
                         f"{len(lo)}/{len(hi)}, expected {len(frame)}")
    for b in np.unique(stratum_id):
        d = delta_map.get(str(int(b)), delta_map.get("__pooled__", 0.0))
        m = stratum_id == b
        lo[m] = lo[m] - d
        hi[m] = hi[m] + d

    rows = heldout & np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if int(rows.sum()) == 0:
        return {"absent": True, "note": "no scorable interval rows"}
    cover = (y >= lo) & (y <= hi)
    width = hi - lo

    def site_level(mask):
        df = pd.DataFrame({"s": sites[mask], "c": cover[mask].astype(float),
                           "w": width[mask]})
        g = df.groupby("s", sort=True).mean()
        return {"coverage": float(g["c"].mean()),
                "mean_width": float(g["w"].mean()),
                "n_sites": int(len(g)), "n_rows": int(mask.sum())}

    pooled = site_level(rows)
    per_bin = {}
    for b in sorted(int(v) for v in np.unique(stratum_id[rows])):
        per_bin[f"bin_{b}"] = site_level(rows & (stratum_id == b))
    lo_w, hi_w = SHIP_COVERAGE_WINDOW
    cov = pooled["coverage"]
    verdict = "ship" if lo_w <= cov <= hi_w else \
        ("under_covered" if cov < lo_w else "over_covered")
    return {"quantile_keys": keys, "delta_source": delta_src,
            "pooled_site_level": pooled, "per_coverage_bin": per_bin,
            "ship_window": list(SHIP_COVERAGE_WINDOW),
            "ship_verdict": verdict,
            "fitted_against_tier_hash":
                (uq_params or {}).get("fitted_against_tier_hash")}


def _quantile_bounds(q_npz):
    """(lo, hi, {name: key}) — lower/upper quantile arrays by key scan."""
    named = {}
    for key in q_npz:
        kl = key.lower().replace("oof_", "")
        m = re.fullmatch(r"q[_]?([0-9]+(?:[._][0-9]+)?)", kl)
        if m:
            q = float(m.group(1).replace("_", "."))
            if q > 1.0:
                q = q / 100.0
            named[q] = key
        elif kl in ("lo", "lower"):
            named[0.05] = key
        elif kl in ("hi", "upper"):
            named[0.95] = key
    lows = [q for q in named if q <= 0.25]
    highs = [q for q in named if q >= 0.75]
    if not lows or not highs:
        return None, None, {}
    ql, qh = min(lows), max(highs)
    return (q_npz[named[ql]], q_npz[named[qh]],
            {"lower": named[ql], "upper": named[qh]})


def _uq_delta(uq_params):
    """{bin_key: delta} from uq_params.json ('delta per coverage bin')."""
    if not uq_params:
        _say("validate: uq_params.json absent — conformal delta 0 applied")
        return {"__pooled__": 0.0}, "absent_delta_zero"
    for name in ("delta_by_bin", "delta_per_bin", "delta_by_coverage_bin",
                 "delta"):
        v = uq_params.get(name)
        if isinstance(v, dict):
            return ({str(k): float(x) for k, x in v.items()
                     if isinstance(x, (int, float))},
                    f"uq_params.{name}")
        if isinstance(v, (int, float)):
            return {"__pooled__": float(v)}, f"uq_params.{name}"
    _say("validate: no delta key recognized in uq_params.json — delta 0")
    return {"__pooled__": 0.0}, "unrecognized_delta_zero"


# ── Bare-site arm (P2 co-primary) ───────────────────────────────────────────

def _bare_site_arm(frame, folds, ext, heldout, outer, hook, quick):
    """Rebuild features at held-out AQS site-days with every PA sensor
    within 5 km of the site excluded (plus the fold's own held-out sites and
    the vault), then score the fold's T1 model + T0. Deep tiers are
    structurally closed off-network, so this number covers the T0+T1 core —
    a LOWER BOUND for the full ladder, reported as such."""
    import frame2
    note = ("bare-site arm covers the T0+T1 core with deep tiers "
            "structurally closed (no serve-time residuals off-network); it "
            "is a lower bound for the full ladder")
    cal_path = config2.artifact("pa_calibrated.parquet")
    if not os.path.exists(cal_path):
        return {"absent": True, "note": "pa_calibrated.parquet missing"}
    try:
        pa_daily = frame2.load_pa_calibrated(cal_path, ext)
        aqs_path = ext.get("aqs")
        aqs_daily = frame2.load_aqs(aqs_path) if aqs_path and \
            os.path.exists(aqs_path) else None
        if aqs_daily is None:
            return {"absent": True, "note": "external_paths['aqs'] missing"}
        pools_base = frame2.build_pools(external_paths=ext,
                                        pa_daily=pa_daily,
                                        aqs_daily=aqs_daily)
    except Exception as e:
        return {"absent": True, "note": f"pool assembly failed: {e}"}

    t0_models = _priors_models()
    vault = sorted(set(str(s) for s in folds.get("vault_sites", [])))
    site_of = {str(s): int(k)
               for s, k in (folds.get("outer_fold_of_site") or {}).items()}

    # PA sensor coordinates (one pair per sensor) for the 5-km exclusion.
    pa_xy = (pa_daily.drop_duplicates("sensor_id")
             .loc[:, ["sensor_id", "lat", "lon"]].reset_index(drop=True))
    pa_ids = pa_xy["sensor_id"].astype(str).to_numpy()
    pa_lat = pa_xy["lat"].to_numpy(dtype=np.float64)
    pa_lon = pa_xy["lon"].to_numpy(dtype=np.float64)

    y = frame["y"].to_numpy(dtype=np.float64)
    sites_arr = frame["unit_id"].to_numpy()
    preds_t1 = np.full(len(frame), np.nan)
    preds_t0 = np.full(len(frame), np.nan)
    n_sites_done = 0

    fold_sites = {}
    for s, k in site_of.items():
        fold_sites.setdefault(k, []).append(s)

    for k in sorted(fold_sites):
        sites_k = sorted(fold_sites[k])
        if quick:
            sites_k = sites_k[:_QUICK_BARE_SITES_PER_FOLD]
        bundle = (hook["models"].get(k) if hook else None)
        for site in sites_k:
            uid = f"aqs_{site}" if not site.startswith("aqs_") else site
            rows = np.flatnonzero(heldout & (sites_arr == uid))
            if len(rows) == 0:
                continue
            slat = float(frame["lat"].to_numpy()[rows[0]])
            slon = float(frame["lon"].to_numpy()[rows[0]])
            near = frame2._haversine_km(pa_lat, pa_lon, slat, slon) <= 5.0
            ex = set(vault) | set(sites_k) | \
                {f"pa_{s}" for s in pa_ids[near]}
            fold_ctx = {"outer_k": k, "vault_units": vault,
                        "unit_ids": np.repeat(uid, len(rows))}
            try:
                pools = dict(pools_base)
                cal_col = (f"pa_cal_f{k}"
                           if f"pa_cal_f{k}" in pa_daily.columns
                           else "pa_cal_full")
                pools["pa"] = frame2.build_pa_pool(
                    pa_daily, exclude_units=sorted(ex), fold_ctx=fold_ctx,
                    value_col=cal_col)
                pools["frm"] = frame2.build_frm_pool(
                    aqs_daily, exclude_units=sorted(ex), fold_ctx=fold_ctx)
                # frame2 never substitutes the full fit inside a fold: a
                # missing fold-k downscaler leaves t0_* NaN, loudly.
                pools["t0"] = t0_models
                feats = frame2.build_point_features(
                    np.repeat(slat, len(rows)), np.repeat(slon, len(rows)),
                    frame["date"].iloc[rows], pools, fold_ctx)
            except Exception as e:
                _say(f"validate: bare-site rebuild failed for {uid} ({e})")
                continue
            if "t0_prior" in feats.columns:
                preds_t0[rows] = feats["t0_prior"].to_numpy(dtype=np.float64)
            if bundle is not None and hook is not None:
                try:
                    preds_t1[rows] = hook["predict"](bundle, feats)
                except Exception as e:
                    _say(f"validate: bare-site T1 predict failed for {uid} "
                         f"({e})")
            n_sites_done += 1

    arm = "t0_plus_t1" if np.isfinite(preds_t1).any() else \
        ("t0_only" if np.isfinite(preds_t0).any() else "absent")
    if arm == "absent":
        return {"absent": True, "note": "no bare-site predictions produced "
                                        "(hooks unavailable)", "n_sites": 0}
    if arm == "t0_only":
        note += "; T1 hook unavailable this run — DEGRADED to T0 only"
    out = {"arm": arm, "note": note, "n_sites": n_sites_done,
           "quick_capped": bool(quick)}
    h = heldout
    if np.isfinite(preds_t1).any():
        out["t1"] = _between_site(y[h], preds_t1[h], sites_arr[h])
        out["t1"]["pooled"] = _metrics(y[h], preds_t1[h])
    if np.isfinite(preds_t0).any():
        out["t0"] = _between_site(y[h], preds_t0[h], sites_arr[h])
        out["t0"]["pooled"] = _metrics(y[h], preds_t0[h])
    return out


# ── Vault (one-shot) ────────────────────────────────────────────────────────

def _git_sha():
    for env in ("AQNET2_GIT_SHA", "GIT_SHA"):
        v = os.environ.get(env)
        if v:
            return v.strip()
    p = config2.artifact("git_sha.txt")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            return fh.read().strip()
    return "unknown"


def _vault_stage(frame, folds, hook, quick, n_boot):
    """One-shot vault evaluation, marker-gated.

    Marker present  -> REFUSE to recompute; report cached metrics_vault.json.
    Marker absent   -> if quick or the T1 serving hook is missing, DEFER
                       (never burn the one-shot on a degraded pass);
                       otherwise score vault sites + vault period through
                       the serving path (bag of T1 fold models on the
                       frame's deployment-view features; deep tiers
                       structurally closed at serve time; T4 not applied —
                       recorded) and write the marker.
    """
    marker_path = config2.artifact("vault_opened.json")
    dest = config2.artifact("metrics_vault.json")
    marker = _read_json(marker_path)
    if marker is not None:
        _say("validate: vault_opened.json exists — REFUSING to recompute "
             "the one-shot vault; reporting cached numbers")
        cached = _read_json(dest)
        if cached is None:
            return {"status": "opened_but_metrics_missing", "marker": marker,
                    "note": "vault_opened.json exists but metrics_vault.json "
                            "is gone — NOT recomputing (one-shot integrity); "
                            "restore the artifact from backup"}
        cached["status"] = "cached"
        cached["marker"] = marker
        return cached
    if quick:
        return {"status": "deferred",
                "note": "--quick never opens the vault (the one-shot must "
                        "not be spent on a smoke test)"}
    if hook is None:
        return {"status": "deferred",
                "note": "T1 serving hook unavailable — refusing to spend "
                        "the one-shot on a degraded (T0-only) pass"}

    vault_sites = set(str(s) for s in folds.get("vault_sites", []))
    vault_uids = {f"aqs_{s}" if not s.startswith("aqs_") else s
                  for s in vault_sites}
    aqs_mask = (frame["unit_type"] == "aqs").to_numpy()
    uid = frame["unit_id"].to_numpy()
    dates = frame["date"].to_numpy()
    in_vault_site = aqs_mask & np.isin(uid, sorted(vault_uids))
    in_vault_period = aqs_mask & (dates >= np.datetime64(VAULT_DATE_START))
    vmask = in_vault_site | in_vault_period
    if int(vmask.sum()) == 0:
        return {"status": "deferred",
                "note": "no vault rows in the frame (site list empty or "
                        "period outside the window)"}

    y = frame["y"].to_numpy(dtype=np.float64)
    sites = frame["unit_id"].to_numpy()
    feats_df = frame.loc[vmask]
    preds = []
    for k in sorted(kk for kk in hook["models"] if isinstance(kk, int)):
        try:
            preds.append(hook["predict"](hook["models"][k], feats_df))
        except Exception as e:
            _say(f"validate: vault T1 fold-{k} predict failed ({e})")
    if not preds:
        return {"status": "deferred",
                "note": "every T1 fold predict failed — vault untouched"}
    serve = np.full(len(frame), np.nan)
    serve[vmask] = _bag_mean(preds)

    def block(mask, name):
        return {"metrics": _metrics(y[mask], serve[mask]),
                "bootstrap_ci": _bootstrap_ci(y[mask], serve[mask],
                                              sites[mask], n_boot),
                "between_site": _between_site(y[mask], serve[mask],
                                              sites[mask]),
                "n_rows": int(mask.sum()),
                "n_sites": int(pd.unique(sites[mask]).size)}

    result = {
        "status": "opened",
        "serve": "t1_fold_bag_mean + deep tiers structurally closed + "
                 "T4 not applied (recorded, not hidden)",
        "n_fold_models": len(preds),
        "vault_sites": block(in_vault_site, "sites"),
        "vault_period": block(in_vault_period, "period"),
        "combined": block(vmask, "combined"),
        "vault_site_list": sorted(vault_sites),
        "vault_period_start": VAULT_DATE_START,
    }
    _write_json(dest, result)
    marker = {"opened_unix_time": time.time(),
              "git_sha": _git_sha(),
              "n_rows_scored": int(vmask.sum()),
              "n_fold_models": len(preds),
              "inputs": {name: os.path.getmtime(config2.artifact(name))
                         for name in ("frame_truth.parquet", "folds2.json",
                                      "oof_tier1.npz", "gates.json")
                         if os.path.exists(config2.artifact(name))}}
    _write_json(marker_path, marker)
    _say("validate: VAULT OPENED — vault_opened.json written; this "
         "evaluation will never be recomputed")
    result["marker"] = marker
    return result


# ── Baselines (paired vs composite, admission-test margins) ────────────────

def _baselines(frame, npz0, npz1, comp, heldout, outer, quick, n_boot):
    y = frame["y"].to_numpy(dtype=np.float64)
    sites = frame["unit_id"].to_numpy()
    dates = frame["date"].to_numpy()
    compv = np.asarray(comp["oof_final"], dtype=np.float64)
    h = heldout
    margins = _read_json(config2.artifact("power_analysis.json"))
    seed = int(config2.SEED)

    preds = {}
    notes = {}

    if npz0 is not None:
        preds["t0_alone"] = np.asarray(npz0["oof_t0"], dtype=np.float64)
    preds["t1"] = np.asarray(npz1["oof"], dtype=np.float64)
    notes["t1"] = ("the incumbent itself — the composite must be "
                   "non-inferior to it by construction")
    for key in sorted(npz1):
        if key.startswith("per_model_"):
            preds[f"t1_candidate_{key[len('per_model_'):]}"] = \
                np.asarray(npz1[key], dtype=np.float64)
    if "weights_json" in npz1:
        try:
            notes["t1_weights"] = json.loads(str(npz1["weights_json"]))
        except Exception:
            pass

    preds["persistence_frm_lag1"] = _persistence(frame)
    clim, clim_note = _climatology(frame)
    preds["site_climatology"] = clim
    notes["site_climatology"] = clim_note

    for s in CTM_STREAMS:
        if s in frame.columns:
            raw = frame[s].to_numpy(dtype=np.float64)
            preds[f"ctm_raw_{s}"] = raw
            preds[f"ctm_debiased_{s}"] = _mean_debias(raw, y, h, outer)
        else:
            notes[f"ctm_{s}"] = "stream column absent from the frame"

    krig, krig_note = _kriging_baseline(frame, heldout, outer, quick)
    if krig is not None:
        preds["kriging_pa_daily"] = krig
    notes["kriging_pa_daily"] = krig_note

    out = {"margins_source": ("power_analysis.json" if margins else
                              "absent_tight_defaults"),
           "notes": notes, "baselines": {}}
    for name, pred in preds.items():
        entry = {"metrics": _metrics(y[h], pred[h])}
        try:
            entry["admission_vs_composite"] = compose.admission_test(
                y[h], pred[h], compv[h], sites[h], margins,
                n_boot=n_boot, seed=seed)
            entry["direction"] = ("pass means the composite beats this "
                                  "baseline within pre-registered margins")
        except Exception as e:
            entry["admission_vs_composite"] = {"error": str(e)}
        out["baselines"][name] = entry
    return out


def _persistence(frame):
    """Site lag-1 FRM: yesterday's monitor value at the same site."""
    aqs = frame["unit_type"].to_numpy() == "aqs"
    src = pd.DataFrame({
        "u": frame["unit_id"].to_numpy()[aqs],
        "d": frame["date"].to_numpy()[aqs] + np.timedelta64(1, "D"),
        "p": frame["y"].to_numpy(dtype=np.float64)[aqs]})
    src = src.drop_duplicates(["u", "d"])
    q = pd.DataFrame({"u": frame["unit_id"].to_numpy(),
                      "d": frame["date"].to_numpy()})
    q = q.merge(src, on=["u", "d"], how="left")
    return q["p"].to_numpy(dtype=np.float64)


def _climatology(frame):
    """Per-site train-period (< TEMPORAL_CUTOFF) mean, scored on
    post-cutoff rows ONLY (NaN elsewhere) — using a mean of the very rows
    being scored would be self-prediction, not a baseline."""
    cutoff = np.datetime64(config2.TEMPORAL_CUTOFF)
    aqs = frame["unit_type"].to_numpy() == "aqs"
    dates = frame["date"].to_numpy()
    uid = frame["unit_id"].to_numpy()
    y = frame["y"].to_numpy(dtype=np.float64)
    tr = aqs & (dates < cutoff) & np.isfinite(y)
    means = pd.DataFrame({"u": uid[tr], "y": y[tr]}) \
        .groupby("u", sort=True)["y"].mean()
    pred = np.full(len(frame), np.nan)
    post = dates >= cutoff
    mapped = pd.Series(uid[post]).map(means).to_numpy(dtype=np.float64)
    pred[post] = mapped
    return pred, ("per-site pre-cutoff mean; finite only on rows >= "
                  f"{config2.TEMPORAL_CUTOFF} (paired test runs there)")


def _mean_debias(raw, y, heldout, outer):
    """Constant offset per outer fold, estimated on the OTHER folds'
    held-out AQS rows (OOF-honest debiasing)."""
    out = np.full(len(raw), np.nan)
    for k in sorted(int(v) for v in np.unique(outer[heldout])):
        rows_k = heldout & (outer == k)
        rows_o = heldout & (outer != k)
        d = raw[rows_o] - y[rows_o]
        d = d[np.isfinite(d)]
        if len(d) < 3:
            continue
        out[rows_k] = raw[rows_k] - float(d.mean())
    return out


def _kriging_baseline(frame, heldout, outer, quick):
    """Per-day ordinary kriging on calibrated PA at AQS points
    (validation.krige_to_sites; train = PA only, fold-aware calibration
    column, vault period excluded from the train pool)."""
    val = _v1()
    if val is None or not hasattr(val, "krige_to_sites"):
        return None, "v1 krige_to_sites unavailable — baseline absent"
    is_pa = frame["unit_type"].to_numpy() == "pa"
    dates = frame["date"].to_numpy()
    lat = frame["lat"].to_numpy(dtype=np.float64)
    lon = frame["lon"].to_numpy(dtype=np.float64)
    pa_ok = is_pa & (dates < np.datetime64(VAULT_DATE_START))
    pred = np.full(len(frame), np.nan)
    rng = np.random.default_rng(config2.SEED)
    n_days_done, n_fail = 0, 0
    for k in sorted(int(v) for v in np.unique(outer[heldout])):
        col = f"pa_cal_f{k}" if f"pa_cal_f{k}" in frame.columns else \
            ("pa_cal_full" if "pa_cal_full" in frame.columns else "y")
        vals = pd.to_numeric(frame[col], errors="coerce") \
            .to_numpy(dtype=np.float64)
        rows_k = heldout & (outer == k)
        days = np.sort(np.unique(dates[rows_k]))
        if quick and len(days) > _QUICK_KRIGE_DAYS:
            days = np.sort(rng.choice(days, _QUICK_KRIGE_DAYS,
                                      replace=False))
        for d in days:
            te = rows_k & (dates == d)
            tr = pa_ok & (dates == d) & np.isfinite(vals)
            if int(tr.sum()) < 5 or int(te.sum()) == 0:
                continue
            try:
                pred[te] = val.krige_to_sites(lat[tr], lon[tr], vals[tr],
                                              lat[te], lon[te])
                n_days_done += 1
            except Exception:
                n_fail += 1
    note = (f"per-day OK on calibrated PA (fold-aware column), "
            f"{n_days_done} fold-days kriged, {n_fail} failed"
            + ("; --quick capped days per fold" if quick else ""))
    return pred, note


# ── Diagnostics ─────────────────────────────────────────────────────────────

def _diagnostics(frame, npz1, compv, heldout, n_boot):
    y = frame["y"].to_numpy(dtype=np.float64)
    lat = frame["lat"].to_numpy(dtype=np.float64)
    lon = frame["lon"].to_numpy(dtype=np.float64)
    out = {}

    t1 = np.asarray(npz1["oof"], dtype=np.float64)
    out["pooled_loso_diagnostic"] = {
        "label": "~93% temporal, diagnostic only",
        "note": ("pooled OOF over BOTH networks; dominated by PA rows and "
                 "temporal structure — never a headline (v1's mistake)"),
        "metrics": _metrics(y, t1)}

    val = _v1()
    if val is not None:
        h = heldout
        resid = compv - y
        day_ids = frame["date"].to_numpy().astype("datetime64[D]") \
            .astype(np.int64)
        try:
            out["morans_i_daily"] = val.morans_i_daily(
                resid[h], lat[h], lon[h], day_ids[h])
        except Exception as e:
            _say(f"validate: morans_i_daily failed ({e})")
        try:
            out["aqi"] = val.aqi_category_metrics(y[h], compv[h])
        except Exception as e:
            _say(f"validate: aqi_category_metrics failed ({e})")
    else:
        out["morans_i_daily"] = {"absent": True, "note": "v1 unavailable"}
    return out


def _strata(frame, compv, heldout):
    val = _v1()
    if val is None:
        return {"absent": True, "note": "v1 strata_metrics unavailable"}
    y = frame["y"].to_numpy(dtype=np.float64)
    smoke = (frame["hms_smoke"].to_numpy(dtype=np.float64)
             if "hms_smoke" in frame.columns else np.zeros(len(frame)))
    dust = (frame["dust"].to_numpy(dtype=np.float64)
            if "dust" in frame.columns else np.full(len(frame), np.nan))
    h = heldout
    try:
        return {"strata": val.strata_metrics(y[h], compv[h], smoke[h],
                                             dust[h]),
                "n_rows": int(h.sum())}
    except Exception as e:
        return {"absent": True, "note": f"strata_metrics failed: {e}"}


def _permutation_report(frame, hook, heldout, outer, quick):
    """OOF-only permutation importance on the T1 winner: features permuted
    within each fold's HELD-OUT rows, re-predicted with that fold's model
    (which never saw those rows), pooled R2 drop. Grouped portable vs
    interpolating via config2.split_feature_sets, plus top-15 singles."""
    if hook is None:
        return {"absent": True,
                "note": "skeleton serving hook unavailable — permutation "
                        "importance needs the fold models"}
    import frame2
    fcols = frame2.feature_columns(frame)
    portable, interp = config2.split_feature_sets(fcols)
    y = frame["y"].to_numpy(dtype=np.float64)
    models = {k: v for k, v in hook["models"].items() if isinstance(k, int)}
    folds_present = sorted(k for k in models
                           if (heldout & (outer == k)).any())
    if not folds_present:
        return {"absent": True, "note": "no fold model covers held-out rows"}

    rng = np.random.default_rng(config2.SEED)

    def predict_all(perm_cols):
        pred = np.full(len(frame), np.nan)
        for k in folds_present:
            rows = np.flatnonzero(heldout & (outer == k))
            sub = frame.iloc[rows].copy()
            for c in perm_cols:
                v = sub[c].to_numpy()
                sub.loc[:, c] = v[rng.permutation(len(v))]
            try:
                pred[rows] = hook["predict"](models[k], sub)
            except Exception as e:
                _say(f"validate: permutation predict failed fold {k} ({e})")
                return None
        return pred

    base = predict_all(())
    if base is None:
        return {"absent": True, "note": "base OOF re-prediction failed"}
    base_r2 = _metrics(y[heldout], base[heldout])["r2"]
    match = _metrics(y[heldout], base[heldout])

    def delta(cols):
        p = predict_all(tuple(cols))
        if p is None:
            return float("nan")
        return base_r2 - _metrics(y[heldout], p[heldout])["r2"]

    groups = {"portable": delta(portable),
              "interpolating": delta(interp)}
    singles = {}
    if quick:
        singles_note = "--quick skips single-feature scan (groups only)"
    else:
        singles_note = "delta pooled R2 when the single column is permuted"
        for c in fcols:
            singles[c] = delta([c])
    top15 = sorted(((c, d) for c, d in singles.items() if np.isfinite(d)),
                   key=lambda kv: -kv[1])[:15]
    return {"base_r2": base_r2, "base_reprediction_metrics": match,
            "n_features": len(fcols),
            "groups": groups, "groups_note":
                "delta pooled R2 when the whole set is permuted jointly",
            "top15_single_features": {c: d for c, d in top15},
            "singles_note": singles_note, "seed": int(config2.SEED)}


# ── Structural audits ───────────────────────────────────────────────────────

def _monotone_report(npz1, comp):
    """composite == T1 BIT-identical wherever no gate opened — the
    structural passthrough guarantee, audited on the shipped arrays."""
    t1 = np.asarray(npz1["oof"], dtype=np.float64)
    final = np.asarray(comp["oof_final"], dtype=np.float64)
    mask = np.asarray(comp["tier_mask"])
    if mask.ndim != 2:
        return {"passed": False,
                "note": f"tier_mask has ndim {mask.ndim}, expected 2"}
    cols = list(range(mask.shape[1]))
    convention = "all_columns"
    if mask.shape[1] > 1 and bool((mask[:, 0] != 0).all()):
        # Column 0 marks the always-on incumbent — ignore it for "closed".
        cols = cols[1:]
        convention = "column0_is_incumbent_ignored"
    closed = mask[:, cols].sum(axis=1) == 0 if cols else \
        np.ones(len(final), dtype=bool)
    equal, n_diff = _bits_equal(final[closed], t1[closed])
    open_frac = {f"col_{j}": float((mask[:, j] != 0).mean())
                 for j in range(mask.shape[1])}
    out = {"passed": bool(equal), "n_closed_rows": int(closed.sum()),
           "n_bit_mismatch": n_diff, "tier_mask_convention": convention,
           "tier_mask_open_fraction_by_column": open_frac,
           "note": ("composite must be BIT-identical to T1 on every row "
                    "where all tier gates are closed/unavailable")}
    if not equal:
        d = final[closed] - t1[closed]
        d = d[np.isfinite(d)]
        out["max_abs_diff_finite"] = float(np.max(np.abs(d))) if len(d) \
            else None
        _say(f"validate: MONOTONE AUDIT FAILED — {n_diff} closed rows "
             f"differ from T1 at the bit level")
    return out


def _parity_report(frame, folds, npz1, npzs_deep, gates, comp, stratum_id,
                   ext, quick):
    """Serving-path parity: (a) seeded sample of inner site-days rebuilt
    through frame2.build_point_features must match the stored features
    bit-for-bit; (b) re-applying gates.json over the tier npzs must
    reproduce oof_composite.npz (post-T3 exact, or post-T4 via a
    deterministic t4_recalibrate replay — the T4 recipe is owned by the
    gates stage and both attempts are recorded)."""
    report = {"feature_parity": _feature_parity(frame, folds, ext, quick),
              "gates_reapplication": _gates_parity(
                  frame, npz1, npzs_deep, gates, comp, stratum_id)}
    report["passed"] = bool(
        report["feature_parity"].get("passed") is True
        and report["gates_reapplication"].get("passed") is True)
    return report


def _feature_parity(frame, folds, ext, quick):
    try:
        import frame2
    except Exception as e:
        return {"passed": None, "note": f"frame2 unimportable ({e})"}
    n_sample = _QUICK_PARITY_N if quick else PARITY_SAMPLE_N
    vault_uids = frame2._as_unit_set(folds.get("vault_sites", []))
    uid = frame["unit_id"].to_numpy()
    dates = frame["date"].to_numpy()
    inner = ~np.isin(uid, sorted(vault_uids)) & \
        (dates < np.datetime64(VAULT_DATE_START))
    pool_idx = np.flatnonzero(inner)
    if len(pool_idx) == 0:
        return {"passed": None, "note": "no inner rows to sample"}
    rng = np.random.default_rng(config2.SEED)
    pick = rng.choice(pool_idx, size=min(n_sample, len(pool_idx)),
                      replace=False)
    pick = np.sort(pick)

    cal_path = config2.artifact("pa_calibrated.parquet")
    if not os.path.exists(cal_path):
        return {"passed": None, "note": "pa_calibrated.parquet missing"}
    aqs_path = ext.get("aqs")
    if not aqs_path or not os.path.exists(aqs_path):
        return {"passed": None, "note": "external_paths['aqs'] missing"}
    try:
        # Deployment-view pools, EXACTLY as build_frame_truth assembles them
        # (full window from the frame itself so quick/full frames both
        # replicate their own construction).
        start = str(pd.Timestamp(dates.min()).date())
        end = str(pd.Timestamp(dates.max()).date())
        pa_daily = frame2.load_pa_calibrated(cal_path, ext, start, end)
        aqs_daily = frame2.load_aqs(aqs_path, start, end)
        fold_ctx = {"unit_ids": uid[pick]}
        vault_sorted = sorted(vault_uids)
        if vault_sorted:
            fold_ctx["vault_units"] = vault_sorted
        t0_models = {}
        scan = getattr(frame2, "_scan_t0_models", None)
        if scan is not None:
            t0_models = scan()
        pools = frame2.build_pools(external_paths=ext,
                                   exclude_units=vault_sorted,
                                   fold_ctx=fold_ctx, pa_daily=pa_daily,
                                   aqs_daily=aqs_daily, t0_models=t0_models,
                                   start=start, end=end)
        rebuilt = frame2.build_point_features(
            frame["lat"].to_numpy()[pick], frame["lon"].to_numpy()[pick],
            frame["date"].iloc[pick], pools, fold_ctx)
    except Exception as e:
        return {"passed": None, "note": f"rebuild failed: {e}"}

    fcols = frame2.feature_columns(frame)
    mismatched, skipped = {}, []
    for c in fcols:
        if c not in rebuilt.columns:
            skipped.append(c)
            continue
        a = frame[c].to_numpy(dtype=np.float64)[pick]
        b = rebuilt[c].to_numpy(dtype=np.float64)
        equal, n_diff = _bits_equal(a, b)
        if not equal:
            both = np.isfinite(a) & np.isfinite(b)
            mismatched[c] = {
                "n_bit_mismatch": n_diff,
                "max_abs_diff_finite": (float(np.max(np.abs(a[both]
                                                            - b[both])))
                                        if both.any() else None),
                "nan_pattern_mismatch": int(np.sum(np.isfinite(a)
                                                   != np.isfinite(b)))}
    passed = not mismatched
    if not passed:
        _say(f"validate: FEATURE PARITY FAILED on {len(mismatched)} "
             f"columns: {sorted(mismatched)[:8]}")
    return {"passed": passed, "n_sampled_rows": int(len(pick)),
            "seed": int(config2.SEED), "window": [start, end],
            "n_feature_columns": len(fcols),
            "skipped_columns_not_rebuilt": skipped,
            "mismatched_columns": mismatched,
            "note": ("stored frame features vs frame2.build_point_features "
                     "rebuild, BIT-for-bit on a seeded inner-row sample")}


def _gates_parity(frame, npz1, npzs_deep, gates, comp, stratum_id):
    final = np.asarray(comp["oof_final"], dtype=np.float64)
    notes = []
    cur = np.asarray(npz1["oof"], dtype=np.float64)
    try:
        for tier_no in (2, 3):
            npz = npzs_deep.get(tier_no)
            key, tg = _find_tier_gates(gates, tier_no)
            if npz is None or tg is None:
                notes.append(f"tier{tier_no}: "
                             f"{'npz absent' if npz is None else 'no gates'}"
                             f" — passthrough")
                continue
            # Raw application first (exact contract); defensive fallback
            # recorded if the raw airlock trips.
            res = np.asarray(npz["oof_r"], dtype=np.float64)
            avail = np.asarray(npz["avail"]).astype(bool)
            pat = np.asarray(npz["pattern_id"])
            try:
                cur = compose.apply_gates(cur, res, avail, pat, stratum_id,
                                          tg)
            except AssertionError as e:
                notes.append(f"tier{tier_no}: raw apply_gates raised "
                             f"({e}); defensively-masked application used")
                cur = _apply_tier(cur, npz, tg, stratum_id,
                                  f"tier{tier_no}", notes)
    except Exception as e:
        return {"passed": None, "note": f"re-application failed: {e}",
                "chain_notes": notes}

    equal, n_diff = _bits_equal(final, cur)
    if equal:
        return {"passed": True, "chain_matches": "post_t3",
                "chain_notes": notes,
                "note": "gates re-application reproduces oof_final "
                        "bit-for-bit (no T4 modification present)"}
    # T4 replay: t4_recalibrate is deterministic given (y, pred, clusters,
    # seed) — if the gates stage ran it on the full frame this reproduces
    # the shipped array exactly.
    try:
        y = frame["y"].to_numpy(dtype=np.float64)
        clusters = frame["unit_id"].to_numpy()
        recal, _params = compose.t4_recalibrate(y, cur, clusters)
        equal4, n_diff4 = _bits_equal(final, recal)
        if equal4:
            return {"passed": True, "chain_matches": "post_t4_replay",
                    "chain_notes": notes,
                    "note": "oof_final == post-T3 chain + deterministic "
                            "t4_recalibrate replay, bit-for-bit"}
    except Exception as e:
        notes.append(f"t4 replay failed: {e}")
        n_diff4 = None
    both = np.isfinite(final) & np.isfinite(cur)
    _say(f"validate: GATES RE-APPLICATION MISMATCH — {n_diff} rows differ "
         f"post-T3 (T4 replay did not close it)")
    return {"passed": False, "chain_matches": False, "chain_notes": notes,
            "n_bit_mismatch_post_t3": n_diff,
            "n_bit_mismatch_post_t4_replay": n_diff4,
            "max_abs_diff_finite": (float(np.max(np.abs(final[both]
                                                        - cur[both])))
                                    if both.any() else None),
            "note": ("the T4 application recipe is owned by the gates "
                     "stage; neither exact post-T3 nor the deterministic "
                     "T4 replay reproduced oof_final — investigate before "
                     "shipping")}


# ── Temporal-variant metrics ───────────────────────────────────────────────

def _temporal_metrics(frame, folds, n_rows):
    """Temporally-pure variant numbers IF the temporal-variant tier
    artifacts exist (DESIGN SS2: temporal-holdout claims may only be made
    from them); marked absent otherwise."""
    names = ["oof_tier1_temporal.npz", "oof_tier2_temporal.npz",
             "oof_tier3_temporal.npz", "oof_composite_temporal.npz"]
    present = {n: config2.artifact(n) for n in names
               if os.path.exists(config2.artifact(n))}
    if not present:
        return {"absent": True,
                "note": ("no temporal-variant tier artifacts on disk — "
                         "temporal-holdout claims may NOT be made from the "
                         "spatial-OOF numbers (7-day embargo, DESIGN SS2)")}
    tmask = np.asarray(folds.get("temporal_is_test", []), dtype=np.int64)
    if len(tmask) != n_rows:
        return {"absent": True, "note": "temporal_is_test length mismatch"}
    aqs = (frame["unit_type"] == "aqs").to_numpy()
    rows = aqs & (tmask == 1)
    y = frame["y"].to_numpy(dtype=np.float64)
    out = {"n_test_rows": int(rows.sum()),
           "embargo_days": int(config2.TEMPORAL_EMBARGO_DAYS)}
    for name, path in sorted(present.items()):
        z = _load_npz(path)
        arr = None
        for key in ("oof_final", "oof", "oof_r", "oof_t0"):
            if key in z:
                arr = np.asarray(z[key], dtype=np.float64)
                break
        if arr is None:
            out[name] = {"note": f"no prediction key in {sorted(z)}"}
            continue
        if len(arr) != n_rows:
            out[name] = {"note": f"length {len(arr)} != frame rows "
                                 f"{n_rows} — misaligned, skipped"}
            continue
        out[name] = _metrics(y[rows], arr[rows])
    return out


# ── SUMMARY.md (v1 write_summary pattern) ──────────────────────────────────

_SUMMARY_SECTIONS = [
    ("metrics_outer.json", "P1-P4 — outer-fold battery"),
    ("metrics_vault.json", "Vault (one-shot second sample)"),
    ("metrics_baselines.json", "Paired baselines vs composite"),
    ("metrics_temporal.json", "Temporally-pure variants"),
    ("metrics_strata.json", "Strata (smoke / dust / clean)"),
    ("permutation_report.json", "OOF permutation importance (T1)"),
    ("monotone_report.json", "Monotone structural audit"),
    ("parity_report.json", "Serving-path parity"),
]


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _flatten(obj, prefix=""):
    rows = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            rows.extend(_flatten(v, f"{prefix}{k}."))
    elif isinstance(obj, (list, tuple)):
        if (len(obj) <= 4 and all(
                isinstance(x, (int, float, str, bool, type(None)))
                for x in obj)):
            rows.append((prefix.rstrip("."), json.dumps(obj)))
        else:
            rows.append((prefix.rstrip("."),
                         f"[{len(obj)} items — see the JSON artifact]"))
    else:
        rows.append((prefix.rstrip("."), _fmt(obj)))
    return rows


def write_summary(quick):
    """Auto-generate SUMMARY.md from whatever metrics artifacts exist —
    only computed numbers appear; absent sections are listed, never filled."""
    stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(time.time()))
    lines = [
        "# AQNet v2 — Validation Summary",
        "",
        f"Auto-generated by validate2.py at {stamp}. Every number was "
        "computed by this run's stages from the artifacts alongside this "
        "file; nothing is hand-entered." + (" **QUICK MODE — smoke-test "
                                            "signal only.**" if quick
                                            else ""),
        "",
        "Headline discipline: the pre-registered primaries are the "
        "outer-fold held-out-AQS numbers (P1), the between-site / rank-rho "
        "pair with its bare-site co-primary (P2), exceedance at FRM labels "
        "(P3) and site-level interval coverage (P4). The pooled LOSO figure "
        "is a labeled ~93%-temporal diagnostic, never a headline.",
        "",
    ]
    for fname, title in _SUMMARY_SECTIONS:
        lines.append(f"## {title}")
        lines.append("")
        obj = _read_json(config2.artifact(fname))
        if obj is None:
            lines.append(f"_`{fname}` not present — its block was skipped "
                         "(see the run log)._")
            lines.append("")
            continue
        rows = _flatten(obj)
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for key, val in rows[:_SUMMARY_MAX_ROWS]:
            lines.append(f"| `{key}` | {val} |")
        if len(rows) > _SUMMARY_MAX_ROWS:
            lines.append(f"| ... | _{len(rows) - _SUMMARY_MAX_ROWS} more "
                         f"rows in `{fname}`_ |")
        lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for name in ("frame_truth.parquet", "folds2.json", "oof_tier0.npz",
                 "oof_tier1.npz", "oof_tier2.npz", "oof_tier3.npz",
                 "gates.json", "oof_composite.npz", "oof_exceed.npz",
                 "quantile_oof.npz", "uq_params.json", "vault_opened.json"):
        mark = "present" if os.path.exists(config2.artifact(name)) \
            else "absent"
        lines.append(f"- `{name}` — {mark}")
    lines.append("")
    path = config2.artifact("SUMMARY.md")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
    os.replace(tmp, path)
    _say(f"wrote {path}")


# ── Stage entry ─────────────────────────────────────────────────────────────

def run_validate(quick=False, frame_path=None, folds_path=None):
    dest_summary = config2.artifact("SUMMARY.md")
    dest_parity = config2.artifact("parity_report.json")
    if (os.path.exists(dest_summary) and os.path.exists(dest_parity)
            and os.environ.get("FORCE") != "1"):
        _say(f"{dest_summary} exists (FORCE=1 to rebuild) -- skip")
        return 0

    print("[aqnet2] ── stage: validate " + "─" * 44)
    t0 = time.time()
    n_boot = _QUICK_N_BOOT if quick else 1000

    frame_path = frame_path or config2.artifact("frame_truth.parquet")
    folds_path = folds_path or config2.artifact("folds2.json")
    if not os.path.exists(frame_path):
        raise SystemExit(f"[aqnet2] {frame_path} missing — run the features "
                         f"stage first")
    frame = _load_frame(frame_path)
    folds = _load_folds(folds_path, frame)
    n = len(frame)

    npz0 = _check_rows(_load_npz(config2.artifact("oof_tier0.npz")),
                       ("oof_t0",), n, "oof_tier0.npz")
    npz1 = _check_rows(_load_npz(config2.artifact("oof_tier1.npz"),
                                 required=True),
                       ("oof",), n, "oof_tier1.npz")
    npzs_deep = {
        k: _check_rows(_load_npz(config2.artifact(f"oof_tier{k}.npz")),
                       ("oof_r", "avail", "pattern_id"), n,
                       f"oof_tier{k}.npz")
        for k in (2, 3)}
    comp = _check_rows(_load_npz(config2.artifact("oof_composite.npz"),
                                 required=True),
                       ("oof_final", "tier_mask"), n, "oof_composite.npz")
    gates = None
    gates_path = config2.artifact("gates.json")
    if os.path.exists(gates_path):
        try:
            gates = compose.load_gates(gates_path)
        except Exception as e:
            _say(f"validate: gates.json failed validation ({e}) — ladder "
                 f"rungs treated as closed")
    exceed_npz = _load_npz(config2.artifact("oof_exceed.npz"))
    exceed_model = _read_json(config2.artifact("exceed_model.json"))
    q_npz = _load_npz(config2.artifact("quantile_oof.npz"))
    uq_params = _read_json(config2.artifact("uq_params.json"))
    ext = {}
    ep = _read_json(config2.artifact("external_paths.json"))
    if ep:
        ext = {k: v for k, v in ep.items() if v}

    aqs_mask = (frame["unit_type"] == "aqs").to_numpy()
    outer = np.asarray(folds["outer_fold"], dtype=np.int64)
    heldout = aqs_mask & (outer >= 0)
    _say(f"held-out AQS rows: {int(heldout.sum()):,} over "
         f"{int(pd.unique(frame['unit_id'].to_numpy()[heldout]).size)} "
         f"sites, {len(np.unique(outer[heldout]))} outer folds")

    if "nbr_pacal_count_50km" not in frame.columns:
        raise SystemExit("[aqnet2] frame lacks nbr_pacal_count_50km — "
                         "cannot form the gate stratum ids")
    stratum_id = _stratum_from_count(frame["nbr_pacal_count_50km"])

    hook = _skeleton_hook()

    # ── Ladder + P1 ──
    ladder_notes = []
    arms = _cumulative_ladder(npz1, npzs_deep, gates, stratum_id, comp,
                              ladder_notes)
    compv = arms["composite"]
    metrics_outer = {"quick": bool(quick), "n_boot": n_boot,
                     "n_heldout_rows": int(heldout.sum()),
                     "ladder_notes": ladder_notes}
    metrics_outer.update(_p1_outer(frame, folds, arms, npz0, heldout, outer,
                                   n_boot))

    # ── P2 ──
    metrics_outer["between_site"] = {
        "with_network": _p2_with_network(frame, arms, npz0, heldout),
        "bare_site": _bare_site_arm(frame, folds, ext, heldout, outer,
                                    hook, quick)}

    # ── P3 / P4 / diagnostics ──
    metrics_outer["exceedance"] = _p3_exceedance(frame, exceed_npz,
                                                 exceed_model, heldout)
    metrics_outer["intervals"] = _p4_intervals(frame, q_npz, uq_params,
                                               heldout, stratum_id)
    metrics_outer.update(_diagnostics(frame, npz1, compv, heldout, n_boot))
    _write_json(config2.artifact("metrics_outer.json"), metrics_outer)

    # ── Strata ──
    _write_json(config2.artifact("metrics_strata.json"),
                _strata(frame, compv, heldout))

    # ── Baselines ──
    _write_json(config2.artifact("metrics_baselines.json"),
                _baselines(frame, npz0, npz1, comp, heldout, outer, quick,
                           n_boot))

    # ── Vault (one-shot; writes its own artifact + marker when it opens) ──
    vault = _vault_stage(frame, folds, hook, quick, n_boot)
    if vault.get("status") != "opened":
        # opened path already wrote metrics_vault.json before the marker.
        _write_json(config2.artifact("metrics_vault.json"), vault)

    # ── Temporal variants ──
    _write_json(config2.artifact("metrics_temporal.json"),
                _temporal_metrics(frame, folds, n))

    # ── Permutation importance ──
    _write_json(config2.artifact("permutation_report.json"),
                _permutation_report(frame, hook, heldout, outer, quick))

    # ── Structural audits ──
    monotone = _monotone_report(npz1, comp)
    _write_json(config2.artifact("monotone_report.json"), monotone)
    parity = _parity_report(frame, folds, npz1, npzs_deep, gates, comp,
                            stratum_id, ext, quick)
    _write_json(config2.artifact("parity_report.json"), parity)

    write_summary(quick)
    if not monotone.get("passed"):
        _say("WARNING: monotone audit failed — the composite is NOT a "
             "structural passthrough where gates are closed")
    if parity.get("passed") is False:
        _say("WARNING: serving parity failed — ship criterion not met")
    _say(f"── stage validate done in {time.time() - t0:.1f}s")
    _say(f"artifacts in {config2.ARTIFACTS_DIR}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="AQNet v2 pre-registered validation battery")
    ap.add_argument("--quick", action="store_true",
                    help="capped bootstraps/kriging/bare-site/parity; "
                         "NEVER opens the vault")
    ap.add_argument("--frame", default=None,
                    help="frame_truth.parquet (default: artifacts/v2)")
    ap.add_argument("--folds", default=None,
                    help="folds2.json (default: artifacts/v2)")
    args = ap.parse_args(argv)
    return run_validate(quick=args.quick, frame_path=args.frame,
                        folds_path=args.folds)


if __name__ == "__main__":
    raise SystemExit(main())
