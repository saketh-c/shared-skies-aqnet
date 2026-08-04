"""AQNet v2 stage driver — the single CLI over research/aqnet2 (DESIGN §13).

Stages, in DAG order (each communicates with the others ONLY via artifacts
under research/aqnet2/artifacts/v2):

  audit      S0: channel provenance, colocation inventory, cf_1 cost
             arithmetic (skip decision), gate POWER ANALYSIS on v1 residuals
             (power_analysis.json margins size every admission test), and the
             T2 kill-switch probe (per-day kriging of v1 T1 OOF residuals
             through compose.admission_test vs the zero-residual baseline).
  data-pa    cf_1 refetch decision stage (BUILD scope: SKIPPED — decision
             artifact only; reconstruction fallback lives in calibrate).
  data       external fetches (SLV met, GEOS-CF, MERRA-2 reassembly, HMS
             grid) via fetchers2; writes/completes external_paths.json.
  statics    HR statics bundle (DEM/TIGER/NEI/WorldPop/NLCD) via fetchers2.
  colocate   (site, sensor) pair table — colocate.py.
  calibrate  S1 Kennedy-O'Hagan PA calibration (ensures folds2.json Phase 1
             exists first — nested calibration without folds is a leak).
  priors     T0 EPA-Downscaler debiased CTM prior — priors.py.
  features   truth frame + Phase-2 folds + per-fold neighbor overrides
             (outer / per-outer LOSO / spatial-block) + oof_tier0.npz.
  skeleton   T1 GPBoost (candidate-B escape documented) — oof_tier1.npz.
  graphpre / graphres    T2 graph-attention residual — oof_tier2.npz.
  fieldpre / fieldres    T3 masked-MAE field + INR decoder — oof_tier3.npz.
  gates      the composition harness: compose.fit_gate/apply_gates ladder
             (T1 -> +T2 -> +T3 -> T4 recal), gates.json + t4_params.json +
             oof_composite.npz.
  exceed     cross-fit exceedance head — exceed.py.
  uq         NexCP conformal + quantile refit with lineage — uq.py.
  validate   the full battery — validate2.py.
  export     serving bundle (bag-of-fold T1 + gates + uq params +
             predict_points) + demo surface + export_manifest.json.
  all        everything above in order.

Idempotency contract (BUILD_NOTES #10): sentinels live HERE — each stage
exits 0 fast when its sentinel artifacts exist and FORCE != "1" — AND as
mirrored file checks in slurm2/aq2-*.sbatch, so preempted chains resume by
resubmission (embers preemption is CANCEL-not-requeue). The v1 exception is
kept: the three data stages are cache-driven (month chunks + .failed.json
sidecars resume mid-fetch), so `data` carries no exit-0 sentinel.

Delegation: sibling modules are imported lazily inside their stage functions
(v1 pattern — a missing heavy dep kills only the stage that needs it). The
GPU tiers are driven through their FROZEN typed API (pretrain(cfg) /
finetune(cfg, fold) / predict_oof(frame, folds, ckpts)) rather than argv,
because INTERFACES freezes those names while the CLI flag surface is
module-private. Other stages are driven run_<stage>(quick=...) first, then
main(argv) — the calibrate.py convention.

Run:  python research/aqnet2/pipeline2.py <stage> [--quick] [--resume]
      FORCE=1 python research/aqnet2/pipeline2.py <stage>   # re-run
"""
import os
import sys
import json
import time
import glob
import shutil
import pickle
import inspect
import argparse
import importlib
import traceback

import numpy as np
import pandas as pd

# ── Path bootstrap (INTERFACES: pipeline2 owns sys.path setup) ──────────────
# Order matters: aqnet2 first, then v1 aqnet, then pipeline, then
# deeplearning — inserted in reverse so the earlier entries win lookups.

_AQNET2_DIR = os.path.dirname(os.path.abspath(__file__))
_RESEARCH_DIR = os.path.dirname(_AQNET2_DIR)
_ROOT = os.path.dirname(_RESEARCH_DIR)
_V1_DIR = os.path.join(_RESEARCH_DIR, "aqnet")
_DL_DIR = os.path.join(_RESEARCH_DIR, "deeplearning")
_PIPE_DIR = os.path.join(_ROOT, "pipeline")
for _p in (_DL_DIR, _PIPE_DIR, _V1_DIR, _AQNET2_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config2
from config2 import artifact

# ── Console: force UTF-8 so banners render identically everywhere ───────────
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

# ── Quick-mode knobs (BUILD_NOTES #10: 3-month window, 2 outer folds,
#    2 epochs, LOSO 4 — the window/folds shrink is enacted by the delegate
#    modules; these are the pipeline-local caps) ────────────────────────────
QUICK_N_BOOT = 120
FULL_N_BOOT = 400
QUICK_KRIGE_DAY_CAP = 30
FULL_KRIGE_DAY_CAP = 150
QUICK_POWER_ROW_CAP = 50_000
QUICK_EPOCHS = 2

EXCEED_THRESHOLD = 35.4
MDE_DELTA_GRID = (0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05)

# Stage sentinels (also mirrored as shell checks in slurm2/aq2-*.sbatch —
# keep the two lists in sync). Names are artifact() basenames; the empty
# tuples are the deliberately sentinel-free cache-driven stages (v1 aq-data
# precedent: rerunning RESUMES a partial fetch instead of trusting it).
SENTINELS = {
    "audit": ("power_analysis.json", "audit_report.json"),
    "data-pa": ("data_pa_decision.json",),
    "data": (),
    "statics": (),           # sentinel is pipeline/static_covariates.parquet
    "colocate": ("colocation_pairs.parquet",),
    "calibrate": ("pa_calibrated.parquet", "calibration_report.json"),
    "priors": ("prior_downscaler_f0.npz",),
    "features": ("frame_truth.parquet", "folds2.json",
                 "nbr_overrides_outer.npz", "oof_tier0.npz"),
    "skeleton": ("oof_tier1.npz",),
    "graphpre": ("graph_pretrain.json",),
    "graphres": ("oof_tier2.npz",),
    "fieldpre": ("field_pretrain.json",),
    "fieldres": ("oof_tier3.npz",),
    "gates": ("gates.json", "oof_composite.npz"),
    "exceed": ("exceed_model.json", "oof_exceed.npz"),
    "uq": ("uq_params.json", "quantile_oof.npz"),
    "validate": ("SUMMARY.md", "parity_report.json"),
    "export": ("export_manifest.json",),
}


# ── Logging / small helpers (v1 pipeline_colab idioms, [aqnet2] prefix) ─────

def _say(msg):
    print(f"[aqnet2] {msg}", flush=True)


def _skip(stage, what, why):
    print(f"[aqnet2] {stage}: SKIPPED {what} — {why}", flush=True)


def _jsonable(o):
    """json default= hook tolerant of numpy scalars and stray objects."""
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)


def _write_json(path, obj):
    """Atomic JSON write (tmp + os.replace) — every artifact writer here."""
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_jsonable)
    os.replace(tmp, path)
    _say(f"wrote {path}")


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_npz(path, **arrays):
    """Atomic compressed npz write (tmp already ends in .npz for savez)."""
    tmp = str(path) + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    _say(f"wrote {path}")


def _force():
    return os.environ.get("FORCE") == "1"


def _skip_if_done(name):
    """Exit-0-fast sentinel check. Message format is FROZEN (the sbatch
    scripts echo the identical sentence, so logs read the same either way)."""
    names = SENTINELS.get(name) or ()
    if not names or _force():
        return False
    if all(os.path.exists(artifact(n)) for n in names):
        _say(f"{name} outputs exist — skipping (FORCE=1 to re-run)")
        return True
    return False


def _fold_keyed(d, k):
    """folds2.json dict sub-maps may key by int or str; accept both."""
    if not isinstance(d, dict):
        return None
    return d.get(str(k), d.get(int(k) if str(k).lstrip("-").isdigit() else k))


def _folds_from_assign(assign):
    """[(train_idx, test_idx)] from a per-row test-fold id; -1 always-train.

    COPIED from v1 pipeline_colab.py (BUILD_NOTES contract #8: pipeline_colab
    must never be imported — it has import-time side effects)."""
    assign = np.asarray(assign, dtype=np.int64)
    folds = []
    for k in sorted(int(v) for v in np.unique(assign[assign >= 0])):
        test = np.where(assign == k)[0]
        train = np.where(assign != k)[0]
        folds.append((train, test))
    return folds


# ── Delegation machinery ────────────────────────────────────────────────────

def _import_stage_module(name, stage):
    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise SystemExit(
            f"[aqnet2] {stage}: stage module '{name}' is unavailable "
            f"({e}) — build/pull it before running this stage") from e


def _call_with_supported(fn, **kw):
    """Call fn passing only the keyword args its signature accepts."""
    try:
        params = inspect.signature(fn).parameters
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD
                   for p in params.values()):
            kw = {k: v for k, v in kw.items() if k in params}
    except (TypeError, ValueError):
        pass
    return fn(**kw)


def _delegate(modname, stage, attrs, argv, quick=False, resume=False,
              required=True):
    """Run a sibling stage module: typed run_* attr first (the calibrate.py
    `run_calibrate(quick=...)` convention), else main(argv).

    required=False degrades to a printed SKIP (used by the data stages so a
    committed-data smoke run works before fetchers2 lands)."""
    try:
        mod = _import_stage_module(modname, stage)
    except SystemExit:
        if required:
            raise
        _skip(stage, f"{modname} delegation", "module not importable yet")
        return None
    for a in attrs:
        fn = getattr(mod, a, None)
        if fn is not None and callable(fn):
            _say(f"{stage}: delegating to {modname}.{a}()")
            return _call_with_supported(fn, quick=quick, resume=resume)
    if hasattr(mod, "main"):
        _say(f"{stage}: delegating to {modname}.main({argv})")
        try:
            rc = mod.main(argv)
        except SystemExit as e:
            rc = e.code
        if rc not in (0, None):
            raise SystemExit(rc)
        return rc
    msg = f"{modname} exposes none of {attrs} and no main()"
    if required:
        raise SystemExit(f"[aqnet2] {stage}: {msg}")
    _skip(stage, f"{modname} delegation", msg)
    return None


def _quick_argv(args, extra=()):
    argv = list(extra)
    if args.quick:
        argv.append("--quick")
    return argv


# ── external_paths.json (data-stage output registry, v1 precedent) ──────────

def _ensure_external_paths():
    """Complete external_paths.json from well-known committed locations.

    fetchers2 is the primary writer; this fills any key it left out from the
    committed v1 products so a frame build never silently loses a stream.
    Keys per BUILD_NOTES contract #1: aqs, pa_daily, geoscf, merra2, cams,
    era5, met_extra, maiac, hms_grid (+ statics). Only existing files are
    recorded — an absent product is an absent key, never a dangling path."""
    dest = artifact("external_paths.json")
    paths = _read_json(dest) or {}
    fallbacks = {
        "aqs": [os.path.join(config2.DATA_DIR, "aqs_daily_tx.parquet"),
                os.path.join(config2.V1_DIR, "data", "aqs_daily_tx.parquet")],
        "pa_daily": [os.path.join(config2.PIPELINE_DIR,
                                  "purpleair_full_dataset.parquet")],
        "geoscf": sorted(glob.glob(os.path.join(
            config2.V1_DIR, "data", "geoscf_pm25_*.parquet"))),
        "merra2": [os.path.join(config2.DATA_DIR, "merra2_daily.parquet")]
                  + sorted(glob.glob(os.path.join(
                      config2.V1_DIR, "data", "merra2_*.parquet"))),
        "cams": [os.path.join(config2.PIPELINE_DIR,
                              "airquality_by_cell.parquet")],
        "era5": [os.path.join(config2.DATA_DIR, "era5_by_cell.parquet")],
        "met_extra": [os.path.join(config2.PIPELINE_DIR,
                                   "met_extra_by_cell.parquet")],
        "maiac": [os.path.join(config2.DATA_DIR, "maiac_aod_by_cell.parquet")],
        "hms_grid": [os.path.join(config2.PIPELINE_DIR, "hms_grid.parquet"),
                     os.path.join(config2.DATA_DIR, "hms_grid.parquet")],
        "statics": [os.path.join(config2.PIPELINE_DIR,
                                 "static_covariates.parquet")],
    }
    changed = False
    for key, cands in fallbacks.items():
        if paths.get(key) and os.path.exists(paths[key]):
            continue
        for c in cands:
            if c and os.path.exists(c):
                paths[key] = c
                changed = True
                break
    if changed or not os.path.exists(dest):
        _write_json(dest, paths)
    missing = [k for k in ("aqs", "pa_daily") if not paths.get(k)]
    if missing:
        _say(f"WARNING: external_paths.json is missing required keys "
             f"{missing} — the frame build will fail until data lands")
    return paths


# ── Audit: power analysis machinery ─────────────────────────────────────────

def _load_v1_residual_source(quick=False):
    """(y, oof, clusters, source_name) from the best available v1 run.

    Order (task contract): cluster-local Aug-1 artifacts (oof_meta.npz +
    training_frame.parquet), then the committed results npz, then None (the
    caller falls to the analytic-CI fallback). The frame supplies y (target)
    and clusters (sensor_id, cast str — pandas 3 join hygiene)."""
    art = os.path.join(config2.V1_DIR, "artifacts")
    frame_p = os.path.join(art, "training_frame.parquet")
    for npz_name in ("oof_meta.npz", "oof_tier1.npz"):
        npz_p = os.path.join(art, npz_name)
        if not (os.path.exists(npz_p) and os.path.exists(frame_p)):
            continue
        df = pd.read_parquet(frame_p)
        with np.load(npz_p, allow_pickle=False) as z:
            key = ("oof_meta" if "oof_meta" in z.files
                   else "oof_lofo" if "oof_lofo" in z.files else "oof")
            if key not in z.files:
                continue
            oof = np.asarray(z[key], dtype=np.float64)
        if len(oof) != len(df):
            _say(f"audit: {npz_name} length {len(oof)} != frame "
                 f"{len(df)} — ignoring this source")
            continue
        y = df["target"].to_numpy(dtype=np.float64)
        clusters = df["sensor_id"].astype(str).to_numpy()
        if quick and len(y) > QUICK_POWER_ROW_CAP:
            rng = np.random.default_rng(config2.SEED)
            pick = np.sort(rng.choice(len(y), QUICK_POWER_ROW_CAP,
                                      replace=False))
            y, oof, clusters = y[pick], oof[pick], clusters[pick]
        return y, oof, clusters, f"v1_artifacts/{npz_name}:{key}"
    # Committed-results npz (usable only if it is self-contained).
    npz_p = os.path.join(_ROOT, "results", "definitive_texas_202607",
                         "oof_meta.npz")
    if os.path.exists(npz_p):
        with np.load(npz_p, allow_pickle=False) as z:
            if {"oof_meta", "y", "sensor_id"} <= set(z.files):
                return (np.asarray(z["y"], dtype=np.float64),
                        np.asarray(z["oof_meta"], dtype=np.float64),
                        z["sensor_id"].astype(str),
                        "results/definitive_texas_202607/oof_meta.npz")
        _say("audit: committed oof_meta.npz is not self-contained "
             "(no y/sensor_id) — falling through")
    return None, None, None, None


def _cluster_suff(y, pred, inv, k_clusters, thresh=EXCEED_THRESHOLD):
    """Per-cluster sufficient stats: everything the bootstrap needs so each
    draw is a bincount-free vectorized reduction (n ~ 300k, B ~ 400)."""
    n_c = np.bincount(inv, minlength=k_clusters).astype(np.float64)
    sy = np.bincount(inv, weights=y, minlength=k_clusters)
    syy = np.bincount(inv, weights=y * y, minlength=k_clusters)
    sse = np.bincount(inv, weights=(pred - y) ** 2, minlength=k_clusters)
    et, ep = y > thresh, pred > thresh
    tp = np.bincount(inv, weights=(et & ep).astype(np.float64),
                     minlength=k_clusters)
    fp = np.bincount(inv, weights=(~et & ep).astype(np.float64),
                     minlength=k_clusters)
    fn = np.bincount(inv, weights=(et & ~ep).astype(np.float64),
                     minlength=k_clusters)
    denom = np.maximum(n_c, 1.0)
    return {"n": n_c, "sy": sy, "syy": syy, "sse": sse,
            "tp": tp, "fp": fp, "fn": fn,
            "my": sy / denom,
            "mp": np.bincount(inv, weights=pred,
                              minlength=k_clusters) / denom}


def _boot_deltas(sa, sb, picks):
    """Paired per-draw deltas (pooled R2, between-cluster R2, exceedance F1)
    from sufficient stats; picks is (n_boot, K) resampled cluster indices —
    both predictors are scored on identical resampled rows (paired)."""
    n = sa["n"][picks].sum(axis=1)
    sy = sa["sy"][picks].sum(axis=1)
    syy = sa["syy"][picks].sum(axis=1)
    sst = syy - sy * sy / np.maximum(n, 1.0)
    sst = np.where(sst > 0, sst, np.nan)
    d_pool = (sa["sse"][picks].sum(axis=1)
              - sb["sse"][picks].sum(axis=1)) / sst

    my = sa["my"][picks]
    mybar = np.nanmean(my, axis=1, keepdims=True)
    sst_s = np.nansum((my - mybar) ** 2, axis=1)
    sst_s = np.where(sst_s > 0, sst_s, np.nan)
    r2a = 1.0 - np.nansum((sa["mp"][picks] - my) ** 2, axis=1) / sst_s
    r2b = 1.0 - np.nansum((sb["mp"][picks] - my) ** 2, axis=1) / sst_s
    d_spat = r2b - r2a

    def _f1(s):
        tp = s["tp"][picks].sum(axis=1)
        den = 2.0 * tp + s["fp"][picks].sum(axis=1) + s["fn"][picks].sum(axis=1)
        return np.where(den > 0, 2.0 * tp / np.maximum(den, 1.0), np.nan)

    return d_pool, d_spat, _f1(sb) - _f1(sa)


def _one_sided_lb(draws):
    draws = draws[np.isfinite(draws)]
    if len(draws) < 20:
        return None
    return float(np.percentile(draws, 5.0))


def _power_margins(y, oof, clusters, n_boot, seed):
    """Bootstrap minimum-detectable deltas at the ACTUAL cluster counts.

    Mechanism: inject a known improvement by shrinking the v1 residuals —
    alt = y - c (y - oof) with c = sqrt(1 - d / (1 - R2_ref)) raises pooled
    R2 by exactly d — then run the same one-sided paired cluster bootstrap
    the admission test uses (compose: lb95 = 5th pct) and take the smallest
    d whose lower bound clears zero. The spatial/exceedance margins are the
    between-cluster-R2 / F1 point deltas realized at their own minimal
    detectable shrink, i.e. margins are expressed in each metric's units.
    Detection-on-the-actual-draw is the operational power criterion (the
    admission test will face exactly this resampling noise). Keys of the
    returned margins dict are FROZEN (audit/03-compose.md §3): any other
    spelling silently falls back to compose.DEFAULT_MARGINS."""
    ok = np.isfinite(y) & np.isfinite(oof)
    y, oof, clusters = y[ok], oof[ok], np.asarray(clusters)[ok]
    uniq, inv = np.unique(clusters.astype(str), return_inverse=True)
    k = len(uniq)
    out = {"n_rows": int(len(y)), "n_clusters": int(k), "n_boot": int(n_boot),
           "seed": int(seed), "delta_grid": list(MDE_DELTA_GRID), "curve": []}
    if k < 2 or len(y) < 10:
        out["note"] = "insufficient rows/clusters — default margins"
        return {"pooled_r2": 0.01, "spatial_r2": 0.01,
                "exceedance_f1": 0.02}, out

    sst = float(np.sum((y - y.mean()) ** 2))
    r2_ref = 1.0 - float(np.sum((oof - y) ** 2)) / sst if sst > 0 else np.nan
    out["r2_ref"] = r2_ref
    if not np.isfinite(r2_ref) or r2_ref >= 1.0:
        out["note"] = "degenerate reference fit — default margins"
        return {"pooled_r2": 0.01, "spatial_r2": 0.01,
                "exceedance_f1": 0.02}, out

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, k, size=(n_boot, k))
    sa = _cluster_suff(y, oof, inv, k)
    has_exceed = bool((y > EXCEED_THRESHOLD).any())

    mde = {"pooled_r2": None, "spatial_r2": None, "exceedance_f1": None}
    for d in MDE_DELTA_GRID:
        if d >= (1.0 - r2_ref) * 0.999:
            break
        c = float(np.sqrt(1.0 - d / (1.0 - r2_ref)))
        alt = y - c * (y - oof)
        sb = _cluster_suff(y, alt, inv, k)
        d_pool, d_spat, d_f1 = _boot_deltas(sa, sb, picks)
        # Point deltas on the full (un-resampled) data.
        spat_pt = _spatial_r2_point(sa, sb)
        f1_pt = _f1_point(sa, sb) if has_exceed else None
        row = {"delta_injected": d,
               "pooled": {"point": d, "lb95": _one_sided_lb(d_pool)},
               "spatial": {"point": spat_pt, "lb95": _one_sided_lb(d_spat)},
               "exceedance_f1": {"point": f1_pt,
                                 "lb95": _one_sided_lb(d_f1)
                                 if has_exceed else None}}
        out["curve"].append(row)
        if mde["pooled_r2"] is None and (row["pooled"]["lb95"] or 0) > 0:
            mde["pooled_r2"] = d
        if (mde["spatial_r2"] is None and spat_pt is not None
                and spat_pt > 0 and (row["spatial"]["lb95"] or 0) > 0):
            mde["spatial_r2"] = spat_pt
        if (has_exceed and mde["exceedance_f1"] is None
                and f1_pt is not None and f1_pt > 0
                and (row["exceedance_f1"]["lb95"] or 0) > 0):
            mde["exceedance_f1"] = f1_pt

    margins = {
        "pooled_r2": float(mde["pooled_r2"] if mde["pooled_r2"] is not None
                           else max(MDE_DELTA_GRID)),
        "spatial_r2": float(mde["spatial_r2"] if mde["spatial_r2"] is not None
                            else max(MDE_DELTA_GRID)),
        "exceedance_f1": float(mde["exceedance_f1"]
                               if mde["exceedance_f1"] is not None else 0.02),
    }
    if mde["pooled_r2"] is None or mde["spatial_r2"] is None:
        out["note"] = ("no grid delta cleared the bootstrap lower bound for "
                       "every metric — margin pinned to the grid max; gates "
                       "will be conservative (passthrough-biased), which is "
                       "the designed safe direction")
    if not has_exceed:
        out["exceedance_note"] = ("no rows above the 35.4 threshold in the "
                                  "residual source — default F1 margin 0.02")
    return margins, out


def _spatial_r2_point(sa, sb):
    my, w = sa["my"], sa["n"] > 0
    if w.sum() < 3:
        return None
    my = my[w]
    sst = float(np.sum((my - my.mean()) ** 2))
    if sst <= 0:
        return None
    r2a = 1.0 - float(np.sum((sa["mp"][w] - my) ** 2)) / sst
    r2b = 1.0 - float(np.sum((sb["mp"][w] - my) ** 2)) / sst
    return float(r2b - r2a)


def _f1_point(sa, sb):
    def f1(s):
        tp, fp, fn = s["tp"].sum(), s["fp"].sum(), s["fn"].sum()
        den = 2.0 * tp + fp + fn
        return 2.0 * tp / den if den > 0 else np.nan
    a, b = f1(sa), f1(sb)
    if not (np.isfinite(a) and np.isfinite(b)):
        return None
    return float(b - a)


def _analytic_margins():
    """Fallback when no v1 residual arrays exist locally: size margins from
    the committed run's bootstrap CI widths (results/*/metrics_loso.json).

    mde_pooled ~= the CI half-width (a shift of that size is what the v1
    bootstrap could just resolve). Spatial scales it by sqrt(n_pa_clusters /
    n_aqs_sites) ~= 3 (62 monitors vs ~530 sensors — the spatial test has
    far fewer effective units). Exceedance falls to compose's default."""
    for run in ("phoenix_202608", "barkjohn", "raw"):
        p = os.path.join(_ROOT, "results", run, "metrics_loso.json")
        d = _read_json(p)
        if not d:
            continue
        try:
            lo, hi = d["tier1_blend"]["bootstrap_ci"]["r2"][:2]
            half = abs(float(hi) - float(lo)) / 2.0
        except (KeyError, TypeError, IndexError, ValueError):
            continue
        if not np.isfinite(half) or half <= 0:
            continue
        margins = {"pooled_r2": float(half),
                   "spatial_r2": float(min(half * 3.0, 0.2)),
                   "exceedance_f1": 0.02}
        return margins, {"source": p, "ci_half_width_r2": half,
                         "spatial_scaling": 3.0,
                         "note": "analytic fallback — no v1 residual arrays"}
    margins = {"pooled_r2": 0.01, "spatial_r2": 0.01, "exceedance_f1": 0.02}
    return margins, {"source": None,
                     "note": "no committed metrics found — compose defaults"}


def _t2_killswitch(y, oof, clusters, margins, quick):
    """Per-day kriging of v1 T1 OOF residuals through the admission test.

    pred_a = the T1 OOF itself (the zero-residual baseline), pred_b = T1 +
    kriged same-day residuals. If even properly-fit kriging cannot clear the
    admission margins on v1 residuals, spatial residual structure is too
    weak for T2 to be a shipping tier (DESIGN §3) and its GPU budget is
    re-scoped to a research arm. Day-capped for the audit budget (the probe
    estimates, the real gate decides)."""
    result = {"kriging_passes": False, "advisory": False}
    try:
        import fusion  # v1 module; pykrige optional inside (IDW fallback)
        import compose
    except ImportError as e:
        result.update(advisory=True, reason=f"module unavailable: {e}")
        return result

    art = os.path.join(config2.V1_DIR, "artifacts")
    frame_p = os.path.join(art, "training_frame.parquet")
    folds_p = os.path.join(art, "folds.json")
    if not os.path.exists(frame_p):
        result.update(advisory=True,
                      reason="no v1 training_frame.parquet — probe needs "
                             "row coordinates/dates; T2 keeps its budget "
                             "provisionally (advisory, not a verdict)")
        return result
    df = pd.read_parquet(frame_p)
    if len(df) != len(y):
        result.update(advisory=True, reason="v1 frame/oof length mismatch")
        return result

    lat_c = "lat" if "lat" in df.columns else "latitude"
    lon_c = "lon" if "lon" in df.columns else "longitude"
    dates = pd.to_datetime(df["date"]).dt.normalize().to_numpy()

    folds_meta = _read_json(folds_p)
    if folds_meta and "loso_fold" in folds_meta:
        assign = np.asarray(folds_meta["loso_fold"], dtype=np.int64)
    else:
        # Deterministic grouped 5-fold fallback over SORTED unit ids
        # (BUILD_NOTES #12: shuffles operate on sorted, dtype-stable arrays).
        units = np.sort(np.unique(np.asarray(clusters).astype(str)))
        rng = np.random.default_rng(config2.SEED)
        perm = rng.permutation(len(units))
        fold_of = {u: int(p % 5) for u, p in zip(units, perm)}
        assign = np.array([fold_of[str(c)] for c in clusters], dtype=np.int64)

    ok = np.isfinite(y) & np.isfinite(oof)
    day_cap = QUICK_KRIGE_DAY_CAP if quick else FULL_KRIGE_DAY_CAP
    days = np.unique(dates[ok])
    if len(days) > day_cap:
        rng = np.random.default_rng(config2.SEED)
        days = np.sort(rng.choice(days, size=day_cap, replace=False))
    sub = ok & np.isin(dates, days)
    _say(f"audit: T2 kill-switch probe on {int(sub.sum()):,} rows / "
         f"{len(days)} days")

    dsub = pd.DataFrame({
        "lat": df.loc[sub, lat_c].to_numpy(dtype=np.float64),
        "lon": df.loc[sub, lon_c].to_numpy(dtype=np.float64),
        "date": pd.to_datetime(df.loc[sub, "date"].to_numpy()),
        "target": y[sub],
    })
    fold_list = _folds_from_assign(assign[sub])
    rk = fusion.residual_kriging_oof(dsub, oof[sub], fold_list,
                                     max_train_per_day=150)
    pred_b = oof[sub] + np.where(np.isfinite(rk), rk, np.nan)
    test = compose.admission_test(y[sub], oof[sub], pred_b,
                                  np.asarray(clusters)[sub], margins,
                                  n_boot=500, seed=config2.SEED)
    result.update(kriging_passes=(test.get("decision") == "pass"),
                  admission=test, n_rows=int(sub.sum()),
                  n_days=int(len(days)))
    return result


# ── Stage: audit ────────────────────────────────────────────────────────────

def stage_audit(args):
    """S0 — no data spend before this passes (DESIGN §3)."""
    if _skip_if_done("audit"):
        return

    report = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
              "quick": bool(args.quick),
              "python": sys.version.split()[0],
              "numpy": np.__version__, "pandas": pd.__version__}

    # 1) PA channel provenance (the committed parquet is single-column ATM).
    pa_path = os.path.join(config2.PIPELINE_DIR,
                           "purpleair_full_dataset.parquet")
    n_pa_rows = None
    if os.path.exists(pa_path):
        import pyarrow.parquet as pq
        meta = pq.ParquetFile(pa_path)
        cols = [c for c in meta.schema_arrow.names]
        n_pa_rows = int(meta.metadata.num_rows)
        report["pa_channel"] = {
            "path": pa_path, "n_rows": n_pa_rows,
            "pm25_columns": [c for c in cols if "pm25" in c.lower()
                             or c == "pm25"],
            "single_pm25_atm": ("pm25" in cols
                                and not any(c.startswith("pm25_")
                                            for c in cols)),
        }
    else:
        _skip("audit", "PA channel provenance", f"{pa_path} not found")
        report["pa_channel"] = None

    # 2) Colocation inventory (pure geometry, from colocate.py).
    try:
        import colocate
        pairs_p = artifact("colocation_pairs.parquet")
        pairs = (pd.read_parquet(pairs_p) if os.path.exists(pairs_p)
                 else colocate.build_pairs())
        report["colocation_inventory"] = colocate.pair_inventory(pairs)
        _say(f"audit: colocation inventory {report['colocation_inventory']}")
    except Exception as e:
        traceback.print_exc()
        _skip("audit", "colocation inventory", f"{type(e).__name__}: {e}")
        report["colocation_inventory"] = None

    # 3) cf_1 refetch cost arithmetic — records the BUILD scope decision.
    #    PurpleAir history API: ~2 points/row read + per-request overhead;
    #    412,507 sensor-days ~= 467 sensors x ~5.3y of daily history ->
    #    O(10^6) points and multi-day wall clock at polite rates. Decision
    #    (BUILD_NOTES scope #1): SKIP; reconstruct pa_cf1 := ATM below
    #    20 ug/m3, channel_reconstructed=1 above (excluded from exceedance
    #    labels, inflated cal_var — enforced in calibrate/exceed).
    n_rows = n_pa_rows or 412_507
    report["cf1_refetch"] = {
        "decision": "skip_refetch",
        "n_sensor_days": n_rows,
        "est_api_points": int(n_rows * 2.5),
        "est_wall_clock_days": round(n_rows / 150_000, 1),
        "fallback": "pa_cf1 := ATM below 20 ug/m3; channel_reconstructed=1 "
                    "above (rows excluded from exceedance labels, cal_var "
                    "inflated)",
    }

    # 4) Gate power analysis on v1 residuals.
    seed = config2.SEED
    n_boot = QUICK_N_BOOT if args.quick else FULL_N_BOOT
    y, oof, clusters, source = _load_v1_residual_source(quick=args.quick)
    if y is not None:
        margins, detail = _power_margins(y, oof, clusters, n_boot, seed)
        detail["source"] = source
    else:
        margins, detail = _analytic_margins()
    _say(f"audit: margins {margins} (source: {detail.get('source')})")

    # 5) T2 kill-switch probe.
    if y is not None:
        t2 = _t2_killswitch(y, oof, clusters, margins, args.quick)
    else:
        t2 = {"kriging_passes": False, "advisory": True,
              "reason": "no v1 residual arrays available locally — probe "
                        "not run; T2 budget decision deferred to the "
                        "cluster run (advisory, not a verdict)"}
    _say(f"audit: t2_killswitch kriging_passes={t2['kriging_passes']}"
         + (" (advisory)" if t2.get("advisory") else ""))

    _write_json(artifact("power_analysis.json"), {
        "mde_pooled_r2": margins["pooled_r2"],
        "mde_site_r2": margins["spatial_r2"],
        # Keys inside `margins` are FROZEN: pooled_r2 / spatial_r2 /
        # exceedance_f1 — compose._resolve_margins silently substitutes
        # defaults for any other spelling (audit/03-compose.md §3/§7.6).
        "margins": margins,
        "power_detail": detail,
        "t2_killswitch": t2,
    })
    _write_json(artifact("audit_report.json"), report)


# ── Stages: data-pa / data / statics ────────────────────────────────────────

def stage_data_pa(args):
    """cf_1 refetch stage — scope-decision artifact only (BUILD_NOTES #1).

    fetchers2 may implement A/B-channel QC later; today the stage records
    the audited skip decision so the DAG edge exists and downstream stages
    can read WHY there is no cf_1 archive."""
    if _skip_if_done("data-pa"):
        return
    _delegate("fetchers2", "data-pa", ("run_data_pa",), _quick_argv(
        args, ["data-pa"]), quick=args.quick, required=False)
    dest = artifact("data_pa_decision.json")
    if not os.path.exists(dest):
        audit = _read_json(artifact("audit_report.json")) or {}
        _write_json(dest, {
            "decision": "skip_refetch",
            "basis": audit.get("cf1_refetch")
            or "audit stage not run — BUILD_NOTES scope decision #1 applies",
            "reconstruction": "pa_cf1 := ATM below 20 ug/m3, "
                              "channel_reconstructed=1 above; reconstructed "
                              "rows excluded from exceedance labels and get "
                              "inflated cal_var",
        })


def stage_data(args):
    """External fetches via fetchers2 (month-chunked + .failed.json sidecar
    resume — hence NO exit-0 sentinel), then complete external_paths.json."""
    _delegate("fetchers2", "data", ("run_data",), _quick_argv(
        args, ["data"]), quick=args.quick, required=False)
    _ensure_external_paths()


def stage_statics(args):
    """HR statics bundle via fetchers2.statics; sentinel is the committed
    pipeline/static_covariates.parquet (DESIGN §12.6), not an $ART file."""
    statics_p = os.path.join(config2.PIPELINE_DIR, "static_covariates.parquet")
    if os.path.exists(statics_p) and not _force():
        _say("statics outputs exist — skipping (FORCE=1 to re-run)")
        return
    _delegate("fetchers2", "statics", ("run_statics",), _quick_argv(
        args, ["statics"]), quick=args.quick, required=False)
    if os.path.exists(statics_p):
        _say(f"statics: {statics_p} present")
    else:
        _say("statics: WARNING static_covariates.parquet still absent — "
             "st_* features will be NaN (frame2 degrades loudly)")
    _ensure_external_paths()


# ── Stages: colocate / calibrate / priors ───────────────────────────────────

def stage_colocate(args):
    if _skip_if_done("colocate"):
        return
    import colocate
    rc = colocate.main(_quick_argv(args))
    if rc not in (0, None):
        raise SystemExit(rc)


def _ensure_site_folds():
    """Phase-1 folds2.json (site-level keys) — calibrate's prerequisite.

    Nested calibration without the fold system would be a leak, so this is
    created here rather than trusted to exist (folds2.build_site_folds is a
    pure function of seed + site ids + coords — contract #3)."""
    path = artifact("folds2.json")
    if os.path.exists(path):
        return
    import folds2
    _say("calibrate: folds2.json missing — building Phase-1 site folds")
    fn = folds2.build_site_folds
    try:
        res = _call_with_supported(fn, seed=config2.SEED, path=path)
    except TypeError:
        res = fn()
    if not os.path.exists(path):
        if isinstance(res, dict):
            _write_json(path, res)
        else:
            raise SystemExit("[aqnet2] calibrate: folds2.build_site_folds "
                             "neither wrote folds2.json nor returned a dict")


def stage_calibrate(args):
    if _skip_if_done("calibrate"):
        return
    _ensure_site_folds()
    import calibrate
    rc = calibrate.run_calibrate(quick=args.quick)
    if rc not in (0, None):
        raise SystemExit(rc)


def stage_priors(args):
    if _skip_if_done("priors"):
        return
    _delegate("priors", "priors", ("run_priors",), _quick_argv(args),
              quick=args.quick)


# ── Stage: features ─────────────────────────────────────────────────────────

def stage_features(args):
    """Truth frame + Phase-2 folds + per-fold overrides + oof_tier0.npz.

    oof_tier0.npz is written HERE (not in priors) because honest T0 OOF
    needs the FOLD-AWARE t0_prior columns that only the outer-override
    recompute produces: each AQS row takes the t0 evaluated by its OWN outer
    fold's downscaler (which never saw that fold's sites); PA rows carry
    outer_fold = -1 and take the frame's deployment-view (full-fit)
    t0_prior — they are never outer-scored, so their T0 entry is
    informational (tier_mask column 0), never an admission quantity.
    pattern_id encodes per-row CTM stream availability as a bitmask over
    priors.STREAMS (bit i set iff stream i is finite at that row)."""
    if _skip_if_done("features"):
        return
    import frame2
    import folds2

    ext = _ensure_external_paths()
    calibrated = artifact("pa_calibrated.parquet")
    if not os.path.exists(calibrated):
        raise SystemExit("[aqnet2] features: pa_calibrated.parquet not "
                         "found — run the calibrate stage first.")

    raw_folds = _read_json(artifact("folds2.json"))  # Phase-1 (or prior full)
    frame = frame2.build_frame_truth(calibrated, ext, quick=args.quick,
                                     folds=raw_folds)

    _say("features: building Phase-2 folds (row-level arrays + content hash)")
    folds = folds2.build_folds(frame, seed=config2.SEED)
    folds2.save_folds(folds, artifact("folds2.json"))

    dest_frame = artifact("frame_truth.parquet")
    tmp = dest_frame + ".tmp"
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, dest_frame)
    _say(f"wrote {dest_frame} ({len(frame):,} rows)")

    # ── Per-fold neighbor/t0/target overrides (v1 f{fold}__{col} contract) ──
    ov_outer = frame2.neighbor_overrides(frame, folds, "outer_fold")
    frame2.save_overrides(ov_outer, artifact("nbr_overrides_outer.npz"))

    outer = np.asarray(folds["outer_fold"], dtype=np.int64)
    outer_ids = sorted(int(k) for k in np.unique(outer[outer >= 0]))
    for k in outer_ids:
        try:
            ov_k = frame2.neighbor_overrides(frame, folds, f"loso:{k}")
        except KeyError as e:
            _skip("features", f"LOSO overrides for outer fold {k}", str(e))
            continue
        frame2.save_overrides(ov_k, artifact(f"nbr_overrides_loso_f{k}.npz"))

    ov_blk = frame2.neighbor_overrides(frame, folds, "spatial_block_fold")
    frame2.save_overrides(ov_blk, artifact("nbr_overrides_block.npz"))

    # ── oof_tier0.npz from the outer overrides' fold-aware t0 columns ──
    n = len(frame)
    oof_t0 = np.full(n, np.nan)
    for k in outer_ids:
        cols = ov_outer.get(k) or {}
        t0k = cols.get("t0_prior")
        if t0k is None:
            _say(f"features: outer fold {k} override has no t0_prior "
                 f"(priors stage incomplete?) — its rows stay NaN")
            continue
        rows = outer == k
        oof_t0[rows] = np.asarray(t0k, dtype=np.float64)[rows]
    pa_rows = outer < 0
    if "t0_prior" in frame.columns:
        full_t0 = frame["t0_prior"].to_numpy(dtype=np.float64)
        oof_t0[pa_rows] = full_t0[pa_rows]
    else:
        _say("features: frame has no t0_prior column — T0 OOF for "
             "PA/vault rows stays NaN")

    # pattern_id: CTM-stream availability bitmask at each row.
    try:
        import priors
        streams = list(priors.STREAMS)
    except ImportError:
        streams = ["geoscf_pm25", "cams_pm25", "merra2_pm25_proxy"]
        _say("features: priors unimportable — using the contract STREAMS "
             "order for pattern_id")
    pattern = np.zeros(n, dtype=np.int8)
    for i, s in enumerate(streams):
        if s in frame.columns:
            avail = np.isfinite(frame[s].to_numpy(dtype=np.float64))
            pattern |= (avail.astype(np.int8) << i)
    _write_npz(artifact("oof_tier0.npz"), oof_t0=oof_t0, pattern_id=pattern)
    n_fin = int(np.isfinite(oof_t0).sum())
    _say(f"features: oof_tier0 finite on {n_fin:,}/{n:,} rows, "
         f"{len(np.unique(pattern))} availability patterns")


# ── Stages: skeleton + GPU tiers ────────────────────────────────────────────

def stage_skeleton(args):
    if _skip_if_done("skeleton"):
        return
    _delegate("skeleton", "skeleton", ("run_skeleton",), _quick_argv(args),
              quick=args.quick, resume=args.resume)


def _tier_cfg(args, extra=None):
    """The cfg dict handed to graph_res/field_res pretrain/finetune.

    Keys are pipeline-chosen (the frozen interface fixes the function names,
    not the cfg schema): quick, resume, seed, epochs (None = module
    default; --quick pins the contract 2)."""
    cfg = {"quick": bool(args.quick), "resume": bool(args.resume) or True,
           "seed": config2.SEED,
           "epochs": QUICK_EPOCHS if args.quick else None}
    cfg.update(extra or {})
    return cfg


def _load_frame_and_folds2():
    import folds2
    frame_p = artifact("frame_truth.parquet")
    if not os.path.exists(frame_p):
        raise SystemExit("[aqnet2] frame_truth.parquet not found — run the "
                         "features stage first.")
    frame = pd.read_parquet(frame_p)
    folds = folds2.load_folds(artifact("folds2.json"), frame)  # verifies hash
    return frame, folds


def _outer_ids(folds):
    outer = np.asarray(folds["outer_fold"], dtype=np.int64)
    return sorted(int(k) for k in np.unique(outer[outer >= 0]))


def _tier_argv(args, subcommand):
    """argv for a tier module's own main(): its stage drivers already own
    the (outer k, inner j) fine-tune loop, checkpoint stems, sentinels and
    npz assembly — pipeline2 delegates instead of re-implementing them
    (finetune()'s fold argument is a (k, j) tuple; a scalar-k loop here
    would both crash and under-build the epistemic ensemble)."""
    argv = [subcommand]
    if args.quick:
        argv.append("--quick")
    argv.append("--resume")   # always resume-capable: embers CANCELs mid-run
    return argv


def _stage_pretrain(args, modname, marker):
    if _skip_if_done(marker):
        return
    mod = _import_stage_module(modname, marker)
    sub = "graphpre" if modname == "graph_res" else "fieldpre"
    rc = mod.main(_tier_argv(args, sub))
    if rc not in (0, None):
        raise SystemExit(f"[aqnet2] {marker}: {modname}.main returned {rc}")
    _write_json(artifact(SENTINELS[marker][0]),
                {"driver": f"{modname}.main {sub}", "quick": bool(args.quick),
                 "seed": config2.SEED,
                 "finished": time.strftime("%Y-%m-%d %H:%M:%S")})


def _stage_residual(args, modname, stage, pre_marker, oof_name):
    if _skip_if_done(stage):
        return
    mod = _import_stage_module(modname, stage)
    sub = "graphres" if modname == "graph_res" else "fieldres"
    rc = mod.main(_tier_argv(args, sub))
    if rc not in (0, None):
        raise SystemExit(f"[aqnet2] {stage}: {modname}.main returned {rc}")
    if not os.path.exists(artifact(oof_name)):
        raise SystemExit(f"[aqnet2] {stage}: module driver completed but "
                         f"{oof_name} was not written")


def stage_graphpre(args):
    _stage_pretrain(args, "graph_res", "graphpre")


def stage_graphres(args):
    _stage_residual(args, "graph_res", "graphres", "graphpre",
                    "oof_tier2.npz")


def stage_fieldpre(args):
    _stage_pretrain(args, "field_res", "fieldpre")


def stage_fieldres(args):
    _stage_residual(args, "field_res", "fieldres", "fieldpre",
                    "oof_tier3.npz")


# ── Stage: gates (the composition harness) ──────────────────────────────────

def _pooled_roles(frame, folds):
    """Selection/confirmation masks pooled across outer-fold contexts.

    folds2 inner_role[k] marks every row 0 (sel) / 1 (conf) / 2 (excluded)
    within outer context k. A row's role here is taken from its OWN outer
    context: AQS rows carry outer_fold = k and are role 2 in context k (they
    are that chain's held-out truth — outer folds are descriptive, they
    never fit or admit gates), so ALL outer-assigned AQS rows are excluded
    from gate fitting/admission — documented, deliberate. PA rows (outer
    -1) appear in every context's inner split; their pooled role is the
    context-consistent vote (sel in some context and conf in another would
    break the sel/conf cluster disjointness compose asserts, so any
    mixed-role unit is demoted to excluded, loudly). Vault units and
    conformal-calibration units are excluded outright (DESIGN §2: conformal
    rows are never admission rows)."""
    n = len(frame)
    outer = np.asarray(folds["outer_fold"], dtype=np.int64)
    inner_role = folds.get("inner_role") or {}
    sel_vote = np.zeros(n, dtype=bool)
    conf_vote = np.zeros(n, dtype=bool)
    for k in sorted(inner_role.keys(), key=str):
        rk = np.asarray(_fold_keyed(inner_role, k), dtype=np.int64)
        if len(rk) != n:
            raise SystemExit(f"[aqnet2] gates: inner_role[{k}] length "
                             f"{len(rk)} != frame rows {n}")
        sel_vote |= rk == 0
        conf_vote |= rk == 1
    role = np.full(n, 2, dtype=np.int64)
    role[sel_vote & ~conf_vote] = 0
    role[conf_vote & ~sel_vote] = 1
    n_mixed = int((sel_vote & conf_vote).sum())
    if n_mixed:
        _say(f"gates: {n_mixed:,} rows voted both sel and conf across "
             f"contexts — excluded (folds2 inner assignment is not "
             f"context-consistent; verify folds2.build_folds)")
    role[outer >= 0] = 2  # AQS fold-k rows: own context holds them out

    unit = frame["unit_id"].astype(str).to_numpy()
    vault = {str(s) for s in folds.get("vault_sites", [])}
    vault |= {f"aqs_{s}" for s in vault if not str(s).startswith("aqs_")}
    role[np.isin(unit, sorted(vault))] = 2
    conf_unit = np.asarray(folds.get("conformal_unit", np.zeros(n)),
                           dtype=np.int64)
    role[conf_unit == 1] = 2

    # Unit purity: a unit split across sel and conf breaks the paired
    # cluster bootstrap; demote such units wholesale.
    for r_a, r_b in ((0, 1),):
        u_sel = set(unit[role == r_a])
        u_conf = set(unit[role == r_b])
        both = u_sel & u_conf
        if both:
            _say(f"gates: {len(both)} units in both halves — excluded")
            role[np.isin(unit, sorted(both))] = 2
    return role == 0, role == 1


def _stratum_from_frame(frame):
    """Coverage-density strata from nbr_pacal_count_50km: 0 / 1-3 / >=4
    (BUILD_NOTES contract #4 — the fit_gate stratum_id source)."""
    col = "nbr_pacal_count_50km"
    if col not in frame.columns:
        _say(f"gates: WARNING {col} missing — single stratum 0")
        return np.zeros(len(frame), dtype=np.int8)
    cnt = np.nan_to_num(frame[col].to_numpy(dtype=np.float64), nan=0.0)
    stratum = np.zeros(len(frame), dtype=np.int8)
    stratum[(cnt >= 1) & (cnt <= 3)] = 1
    stratum[cnt >= 4] = 2
    return stratum


def _load_tier_npz(name, n):
    p = artifact(name)
    if not os.path.exists(p):
        return None
    out = {}
    with np.load(p, allow_pickle=False) as z:
        for k in z.files:
            out[k] = z[k]
    length = len(out.get("oof_r", out.get("oof", out.get("oof_t0", []))))
    if length != n:
        raise SystemExit(f"[aqnet2] gates: {name} length {length} != frame "
                         f"rows {n} — stale artifact, re-run its stage")
    return out


def _gate_step(y, incumbent, tier, stratum, clusters, sel, conf, margins,
               tier_name, a100):
    """One fit_gate/apply_gates rung. Enforces the 100-km hard alpha=0
    contract (BUILD_NOTES #5) BEFORE fitting, and masks availability to rows
    where the incumbent itself is finite (a residual cannot compose onto a
    missing incumbent — those rows stay NaN, honestly)."""
    import compose
    avail = tier["avail"].astype(bool)
    viol = avail & (a100 == 0)
    if viol.any():
        raise AssertionError(
            f"{tier_name}: avail=1 on {int(viol.sum())} rows with no PA "
            f"station within {config2.GATE_MAX_DIST_KM:.0f} km — the tier "
            f"violated the hard-zero contract (predict_oof must zero these)")
    av = avail & np.isfinite(incumbent)
    n_dropped = int((avail & ~av).sum())
    if n_dropped:
        _say(f"gates: {tier_name}: {n_dropped:,} available rows have no "
             f"finite incumbent — masked out (no composition target)")
    resid = np.where(av, tier["oof_r"], np.nan)
    gate = compose.fit_gate(y, incumbent, resid, av,
                            tier["pattern_id"], stratum, clusters,
                            sel, conf, margins)
    composed = compose.apply_gates(incumbent, resid, av,
                                   tier["pattern_id"], stratum, gate)
    applied = av & np.isfinite(incumbent) & np.isfinite(composed)
    applied &= ~np.isclose(composed, incumbent, rtol=0.0, atol=0.0,
                           equal_nan=True)
    _say(f"gates: {tier_name}: n_open={gate.n_open} "
         f"applied_rows={int(applied.sum()):,}")
    return gate, composed, applied


def stage_gates(args):
    """T1 -> (+T2) -> (+T3) -> T4 ladder through compose (DESIGN §9)."""
    if _skip_if_done("gates"):
        return
    import compose
    frame, folds = _load_frame_and_folds2()
    n = len(frame)

    t1 = _load_tier_npz("oof_tier1.npz", n)
    if t1 is None:
        raise SystemExit("[aqnet2] gates: oof_tier1.npz not found — run the "
                         "skeleton stage first.")
    t0 = _load_tier_npz("oof_tier0.npz", n)
    t2 = _load_tier_npz("oof_tier2.npz", n)
    t3 = _load_tier_npz("oof_tier3.npz", n)

    y = frame["y"].to_numpy(dtype=np.float64)
    clusters = frame["unit_id"].astype(str).to_numpy()
    stratum = _stratum_from_frame(frame)
    sel, conf = _pooled_roles(frame, folds)
    _say(f"gates: sel rows {int(sel.sum()):,} / conf rows "
         f"{int(conf.sum()):,} (clusters: {len(np.unique(clusters[sel]))} "
         f"sel / {len(np.unique(clusters[conf]))} conf)")
    margins = _read_json(artifact("power_analysis.json"))
    if margins is None:
        _say("gates: WARNING power_analysis.json missing — compose will "
             "warn and use its default margins")

    a100_col = "nbr_pacal_avail_100km"
    if a100_col in frame.columns:
        a100 = np.nan_to_num(frame[a100_col].to_numpy(dtype=np.float64),
                             nan=0.0)
    else:
        _say(f"gates: WARNING {a100_col} missing — 100-km hard-zero assert "
             f"degraded to avail-as-declared")
        a100 = np.ones(n)

    f1 = np.asarray(t1["oof"], dtype=np.float64)
    gates_out, tier_mask = {}, np.zeros((n, 4), dtype=np.uint8)
    # Column 0: T0 informational (finite OOF prior at the row) — T0 is the
    # extrapolation floor UNDER T1, never a gated additive rung.
    if t0 is not None:
        tier_mask[:, 0] = np.isfinite(
            np.asarray(t0["oof_t0"], dtype=np.float64)).astype(np.uint8)

    current = f1
    if t2 is not None:
        gate2, current, applied2 = _gate_step(
            y, f1, t2, stratum, clusters, sel, conf, margins, "tier2", a100)
        gates_out["t2"] = gate2
        tier_mask[:, 1] = applied2.astype(np.uint8)
    else:
        _skip("gates", "tier2 rung", "oof_tier2.npz absent — passthrough")

    f2 = current
    if t3 is not None:
        gate3, current, applied3 = _gate_step(
            y, f2, t3, stratum, clusters, sel, conf, margins, "tier3", a100)
        gates_out["t3"] = gate3
        tier_mask[:, 2] = applied3.astype(np.uint8)
    else:
        _skip("gates", "tier3 rung", "oof_tier3.npz absent — passthrough")

    # T4 last: declared affine recalibration rung, cross-fit by cluster.
    # Its params go to t4_params.json, NEVER gates.json — "slope_clip"
    # deliberately trips compose's forbidden-key scanner (audit/03 §5).
    f3 = current
    recal, t4_params = compose.t4_recalibrate(y, f3, clusters,
                                              seed=config2.SEED)
    applied4 = np.isfinite(recal) & np.isfinite(f3) & (recal != f3)
    tier_mask[:, 3] = applied4.astype(np.uint8)
    oof_final = recal
    _say(f"gates: T4 modified {int(applied4.sum()):,} rows "
         f"(slopes {[round(f['b'], 3) for f in t4_params['folds']]})")

    if gates_out:
        compose.save_gates(gates_out, artifact("gates.json"))
    else:
        # No residual tier delivered an OOF — an all-closed gates.json keeps
        # the artifact contract (load_gates-able, expresses zero opens).
        _write_json(artifact("gates.json"), {})
        _say("gates: no residual tiers available — empty gates.json "
             "(composite == T1 + T4)")
    _write_json(artifact("t4_params.json"), t4_params)
    _write_npz(artifact("oof_composite.npz"),
               oof_final=oof_final, tier_mask=tier_mask)
    fin = np.isfinite(oof_final)
    _say(f"gates: composite finite on {int(fin.sum()):,}/{n:,} rows; "
         f"tier_mask col sums {tier_mask.sum(axis=0).tolist()}")


# ── Stages: exceed / uq / validate ──────────────────────────────────────────

def stage_exceed(args):
    if _skip_if_done("exceed"):
        return
    _delegate("exceed", "exceed", ("run_exceed",), _quick_argv(args),
              quick=args.quick)


def stage_uq(args):
    if _skip_if_done("uq"):
        return
    _delegate("uq", "uq", ("run_uq",), _quick_argv(args), quick=args.quick)


def stage_validate(args):
    if _skip_if_done("validate"):
        return
    _delegate("validate2", "validate", ("run_validate",), _quick_argv(args),
              quick=args.quick)


# ── Stage: export (serving bundle) ──────────────────────────────────────────

_BUNDLE_DIRNAME = "serving_bundle"
_BUNDLE_COPY = ("gates.json", "t4_params.json", "uq_params.json",
                "folds2.json", "power_analysis.json",
                "calibration_report.json", "export_manifest.json")


def _bundle_dir():
    d = os.path.join(config2.ARTIFACTS_DIR, _BUNDLE_DIRNAME)
    os.makedirs(d, exist_ok=True)
    return d


def _fit_serving_models(frame, folds, quick):
    """Bag-of-fold T1 serving models into the bundle.

    If the skeleton stage persisted its fitted candidate-B objects
    (t1_serving_f{k}.pkl in artifacts), they are copied; otherwise the
    models are RE-FIT here per outer fold via v1 models_tabular.fit_full on
    fold-aware columns (outer overrides applied, fold-k sites and vault rows
    excluded from training) and STORED — serving must never depend on
    objects that only lived in a dead process (DESIGN §11: bag of fold
    models, never a full refit at query time)."""
    import frame2
    bundle = _bundle_dir()
    pre = sorted(glob.glob(os.path.join(config2.ARTIFACTS_DIR,
                                        "t1_serving_f*.pkl")))
    if pre:
        for p in pre:
            shutil.copy2(p, os.path.join(bundle, os.path.basename(p)))
        _say(f"export: copied {len(pre)} persisted skeleton serving models")
        return [os.path.basename(p) for p in pre], "skeleton_persisted"

    try:
        import models_tabular
    except ImportError as e:
        _skip("export", "T1 serving refit",
              f"models_tabular deps unavailable ({e})")
        return [], "unavailable"

    ov_path = artifact("nbr_overrides_outer.npz")
    overrides = (frame2.load_overrides(ov_path, len(frame))
                 if os.path.exists(ov_path) else {})
    feats = frame2.feature_columns(frame)
    outer = np.asarray(folds["outer_fold"], dtype=np.int64)
    unit = frame["unit_id"].astype(str).to_numpy()
    vault = {str(s) for s in folds.get("vault_sites", [])}
    vault |= {f"aqs_{s}" for s in set(vault)}
    non_vault = ~np.isin(unit, sorted(vault))

    ks = _outer_ids(folds)
    if quick:
        ks = ks[:1]
    written = []
    jobs = [(k, f"t1_serving_f{k}.pkl") for k in ks]
    jobs.append((None, "t1_serving_full.pkl"))
    for k, fname in jobs:
        sub = frame.copy()
        if k is not None:
            for col, arr in (overrides.get(k) or {}).items():
                if col in sub.columns or col == "y":
                    sub[col] = np.asarray(arr, dtype=np.float64)
            train = non_vault & (outer != k)
        else:
            train = non_vault
        sub = sub.loc[train].reset_index(drop=True)
        tr = sub.copy()
        tr["target"] = tr["y"].to_numpy(dtype=np.float64)
        tr["sensor_id"] = tr["unit_id"].astype(str)
        _say(f"export: fitting serving bundle "
             f"{'fold ' + str(k) if k is not None else 'full'} "
             f"({len(tr):,} rows)")
        fitted = models_tabular.fit_full(tr, feats)
        dest = os.path.join(_bundle_dir(), fname)
        tmp = dest + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump({"fitted": fitted, "outer_fold": k,
                         "features": feats}, fh)
        os.replace(tmp, dest)
        written.append(fname)
    return written, "refit_candidate_b"


def predict_points(lats, lons, dates, bundle_dir=None):
    """Serve PM2.5 point predictions from the exported bundle.

    Path parity: features come from frame2.build_point_features on full
    deployment pools (pa_cal_full targets, full-fit T0) — the identical
    builder training used. Prediction = mean over the bag of per-outer-fold
    T1 models (epistemic bagging; the full-fit model only if no fold models
    exist), then the T4 affine with fold-averaged (a, b). T2/T3 residual
    tiers are not servable from this CPU bundle: their avail is structurally
    0 at query time, so the gate composition is exact passthrough — never a
    fill (DESIGN §1)."""
    import frame2
    try:
        import models_tabular
    except ImportError as e:
        raise SystemExit(f"[aqnet2] predict_points: models_tabular "
                         f"unavailable ({e})")
    bundle = bundle_dir or _bundle_dir()
    ext = _ensure_external_paths()
    folds = _read_json(artifact("folds2.json")) or {}
    vault = sorted({str(s) for s in folds.get("vault_sites", [])})

    t0_models = {}
    try:
        import priors
        t0_models = priors.load_fold_models()
    except Exception as e:
        _say(f"predict_points: T0 models unavailable ({e}) — t0_* NaN")
    pools = frame2.build_pools(
        calibrated_parquet=artifact("pa_calibrated.parquet"),
        external_paths=ext, exclude_units=vault,
        fold_ctx={"vault_units": vault}, t0_models=t0_models)
    feats_df = frame2.build_point_features(lats, lons, dates, pools,
                                           {"vault_units": vault})

    fold_pkls = sorted(glob.glob(os.path.join(bundle, "t1_serving_f*.pkl")))
    if not fold_pkls:
        full = os.path.join(bundle, "t1_serving_full.pkl")
        if not os.path.exists(full):
            raise SystemExit("[aqnet2] predict_points: no serving models in "
                             f"{bundle} — run the export stage")
        fold_pkls = [full]
    preds = []
    for p in fold_pkls:
        with open(p, "rb") as fh:
            b = pickle.load(fh)
        cols = [c for c in b["features"] if c in feats_df.columns]
        missing = [c for c in b["features"] if c not in feats_df.columns]
        for c in missing:
            feats_df[c] = np.nan  # absent product -> NaN, models impute
        preds.append(models_tabular.predict_full(
            b["fitted"], feats_df[b["features"]]))
        del cols
    pred = np.nanmean(np.vstack(preds), axis=0)

    t4 = _read_json(artifact("t4_params.json"))
    if t4 and t4.get("folds"):
        a = float(np.mean([f["a"] for f in t4["folds"]]))
        b = float(np.mean([f["b"] for f in t4["folds"]]))
        pred = a + b * pred
    return pred


def stage_export(args):
    if _skip_if_done("export"):
        return
    frame, folds2_full = None, None
    try:
        frame, folds2_full = _load_frame_and_folds2()
    except SystemExit:
        raise
    bundle = _bundle_dir()

    copied = []
    for name in _BUNDLE_COPY:
        src = artifact(name)
        if name != "export_manifest.json" and os.path.exists(src):
            shutil.copy2(src, os.path.join(bundle, name))
            copied.append(name)
    models, model_source = _fit_serving_models(frame, folds2_full,
                                               args.quick)

    # Demo surface: last 30 days at 0.1 deg over TX_BBOX (quick: 0.5 deg x
    # 3 days) through the same predict_points serving path.
    demo_name = None
    try:
        bbox = config2.TX_BBOX
        step = 0.5 if args.quick else config2.GRID_DEG
        n_days = 3 if args.quick else 30
        glat = np.round(np.arange(bbox["lat_min"], bbox["lat_max"] + step / 2,
                                  step), 6)
        glon = np.round(np.arange(bbox["lon_min"], bbox["lon_max"] + step / 2,
                                  step), 6)
        end = pd.to_datetime(frame["date"]).max()
        days = pd.date_range(end - pd.Timedelta(days=n_days - 1), end,
                             freq="D")
        gg_lat, gg_lon = np.meshgrid(glat, glon, indexing="ij")
        flat_lat = np.tile(gg_lat.ravel(), len(days))
        flat_lon = np.tile(gg_lon.ravel(), len(days))
        flat_day = np.repeat(days.to_numpy(), gg_lat.size)
        _say(f"export: demo surface {len(flat_lat):,} queries "
             f"({len(days)} days x {gg_lat.size} cells)")
        pred = predict_points(flat_lat, flat_lon, flat_day,
                              bundle_dir=bundle)
        demo = pd.DataFrame({"lat": flat_lat, "lon": flat_lon,
                             "date": flat_day, "pm25_pred": pred})
        demo_name = "demo_surface.parquet"
        dest = os.path.join(bundle, demo_name)
        tmp = dest + ".tmp"
        demo.to_parquet(tmp, index=False)
        os.replace(tmp, dest)
        _say(f"wrote {dest}")
    except Exception as e:
        traceback.print_exc()
        _skip("export", "demo surface", f"{type(e).__name__}: {e}")

    git_sha = None
    sha_p = os.path.join(config2.ARTIFACTS_DIR, "git_sha.txt")
    if os.path.exists(sha_p):
        with open(sha_p, encoding="utf-8") as fh:
            git_sha = fh.read().strip()
    _write_json(artifact("export_manifest.json"), {
        "bundle_dir": bundle,
        "copied_artifacts": copied,
        "t1_models": models,
        "t1_model_source": model_source,
        "demo_surface": demo_name,
        "predict_api": "python -c \"import pipeline2; "
                       "pipeline2.predict_points(lats, lons, dates)\"",
        "t2_t3_serving": "not bundled: residual tiers have avail=0 at query "
                         "time -> exact gate passthrough (never a fill)",
        "git_sha": git_sha,
        "quick": bool(args.quick),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    })


# ── CLI ─────────────────────────────────────────────────────────────────────

_STAGES = {
    "audit": stage_audit,
    "data-pa": stage_data_pa,
    "data": stage_data,
    "statics": stage_statics,
    "colocate": stage_colocate,
    "calibrate": stage_calibrate,
    "priors": stage_priors,
    "features": stage_features,
    "skeleton": stage_skeleton,
    "graphpre": stage_graphpre,
    "graphres": stage_graphres,
    "fieldpre": stage_fieldpre,
    "fieldres": stage_fieldres,
    "gates": stage_gates,
    "exceed": stage_exceed,
    "uq": stage_uq,
    "validate": stage_validate,
    "export": stage_export,
}

_STAGE_ORDER = ["audit", "data-pa", "data", "statics", "colocate",
                "calibrate", "priors", "features", "skeleton", "graphpre",
                "graphres", "fieldpre", "fieldres", "gates", "exceed", "uq",
                "validate", "export"]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="AQNet v2 stage driver (FRM-anchored residual ladder).")
    ap.add_argument("stage", choices=_STAGE_ORDER + ["all"],
                    help="stage to run ('all' runs the full DAG in order)")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 3-month window, 2 outer folds, "
                         "2 epochs, LOSO 4 (threaded to every delegate)")
    ap.add_argument("--resume", action="store_true",
                    help="GPU/long stages: resume from last.pt checkpoints "
                         "(chain links pass this; harmless elsewhere)")
    args = ap.parse_args(argv)

    stages = _STAGE_ORDER if args.stage == "all" else [args.stage]
    for name in stages:
        t0 = time.time()
        _say(f"── stage: {name} " + "─" * max(0, 58 - len(name)))
        _STAGES[name](args)
        _say(f"── stage {name} done in {time.time() - t0:.1f}s")
    _say(f"artifacts in {config2.ARTIFACTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
