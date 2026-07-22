"""AQNet research pipeline — stage-based CLI over the research/aqnet modules.

Runs the three-tier AQNet stack end to end, Colab-friendly:

  data      fetch external open datasets: EPA AQS daily FRM/FEM PM2.5
            (EXTERNAL validation only — never enters training), NASA GEOS-CF
            surface PM2.5 (OPeNDAP), MERRA-2 aerosol species + PBLH
            (earthaccess; degrades gracefully without credentials)
  features  assemble the sensor-day training frame (Barkjohn-corrected
            target, physical features only, leave-self-out neighbor
            aggregates, external CTM joins), freeze the CV folds, and
            precompute per-fold neighbor-feature overrides (pools restricted
            to each LOSO fold's TRAIN sensors) for the tabular tier
  tabular   Tier 1 — GBM ensemble LOSO cross-validation with the per-fold
            neighbor overrides, pooled AND leave-one-fold-out simplex blends
            (LOFO is the headline: fold f's weights never see fold f's rows),
            plus LightGBM quantile heads
  deep      Tier 2 — FusionUNet on the extended gridded stack (reuses
            research/deeplearning; AMP + warmup/cosine LR + crop augmentation
            live in models_deep; --batch-size overrides the auto batch)
  fuse      Tier 3 — residual kriging of the LOFO blend's OOF errors, a
            cross-fitted non-negative ridge meta-learner over {tier1, rk,
            unet} (U-Net part masked to its held-out validation-site pixels),
            a final ridge fit on meta-train sensors, and split-conformal
            recentering of the quantile band on Tier 3 (delta calibrated on
            the sensor-disjoint calibration split; coverage AND mean width
            reported on the meta-train cross-fit rows, disjoint from delta)
  ablation  Tier-1 feature-set ablations on the SAME frozen LOSO folds:
            primary / plus_demographics / no_external / no_neighbor, with
            paired delta-R2 cluster-bootstrap CIs (plus_demographics exists
            ONLY here — demographics are never inputs anywhere else)
  validate  LOSO / spatial-block / temporal metrics with CLUSTER (per-sensor)
            bootstrap CIs, per-day Moran's I, AQI-category skill for Tier 1
            and the Tier-3 cross-fit, smoke/dust/clean strata, spatial vs
            temporal R2 decomposition, a Tier-2 U-Net row, interpolation +
            raw-CTM baselines (incl. mean-debiased variants), external EPA
            AQS validation with a deployment-mode Tier-3 row, SHAP +
            permutation importance, and an auto-generated SUMMARY.md
  all       every stage above, in order (--skip-ablation drops the ablation)

Every artifact lands in research/aqnet/artifacts/. Stages are restartable:
each reads only files earlier stages wrote, so a crashed run resumes at the
failed stage. Stages print what they skipped and why (missing credentials,
missing optional dependencies, absent upstream artifacts) instead of dying.
Every number in the metrics artifacts and SUMMARY.md is computed by this
run — nothing is hand-entered.

Run from the repo root (paths are derived from this file, so any cwd works):
    python research/aqnet/pipeline_colab.py all --quick   # small-window smoke test
    python research/aqnet/pipeline_colab.py all           # full run (GPU for deep)
"""
import os
import sys
import gc
import json
import time
import argparse
import traceback

import numpy as np
import pandas as pd

# ── Path bootstrap (identical in Colab and locally) ─────────────────────────

_AQNET_DIR = os.path.dirname(os.path.abspath(__file__))
_DEEP_DIR = os.path.normpath(os.path.join(_AQNET_DIR, os.pardir, "deeplearning"))
for _p in (_AQNET_DIR, _DEEP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from config import artifact

# ── Quick-mode settings (smoke tests: small window, coarse grid, few folds) ─

QUICK_START = "2024-07-01"
QUICK_END = "2024-09-30"
QUICK_TEMPORAL_CUTOFF = "2024-09-01"
QUICK_AQS_YEARS = [2024]
QUICK_LOSO_FOLDS = 4
QUICK_BLOCK_FOLDS = 3
QUICK_EPOCHS = 3
QUICK_GRID_DEG = 0.2

FULL_LOSO_FOLDS = 10
FULL_BLOCK_FOLDS = 5
FULL_EPOCHS = 100

SHAP_SAMPLE_ROWS = 2000
PERMUTATION_SAMPLE_ROWS = 5000
KRIGE_MAX_TRAIN_PER_DAY = 150  # cap for the deployment-mode Tier-3 kriging


# ── Small helpers ────────────────────────────────────────────────────────────

# Windows consoles can default stdout to cp1252, which cannot encode the
# box-drawing characters used in progress banners. Force UTF-8 (replace on
# failure) so the pipeline behaves identically on Colab, Linux, and Windows.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def _say(msg):
    print(f"[aqnet] {msg}", flush=True)


def _skip(stage, what, why):
    print(f"[aqnet] {stage}: SKIPPED {what} — {why}", flush=True)


def _jsonable(o):
    """json.dumps default= hook tolerant of numpy scalars and stray objects."""
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_jsonable)
    _say(f"wrote {path}")


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _window(args):
    if args.quick:
        return QUICK_START, QUICK_END
    return config.DATE_START, config.DATE_END


def _grid_deg(args):
    if args.grid_deg is not None:
        return float(args.grid_deg)
    return QUICK_GRID_DEG if args.quick else config.GRID_DEG


def _epochs(args):
    if args.epochs is not None:
        return int(args.epochs)
    return QUICK_EPOCHS if args.quick else FULL_EPOCHS


def _external_paths():
    """Paths recorded by the data stage ({} when the stage has not run)."""
    return _read_json(artifact("external_paths.json")) or {}


def _folds_from_assign(assign):
    """Rebuild [(train_idx, test_idx), ...] from a per-row test-fold id."""
    assign = np.asarray(assign, dtype=np.int64)
    folds = []
    for k in sorted(int(k) for k in np.unique(assign[assign >= 0])):
        test = np.where(assign == k)[0]
        train = np.where(assign != k)[0]
        folds.append((train, test))
    return folds


def _load_frame_and_folds():
    frame_path = artifact("training_frame.parquet")
    folds_path = artifact("folds.json")
    if not os.path.exists(frame_path):
        raise SystemExit("[aqnet] training_frame.parquet not found — run the "
                         "features stage first.")
    if not os.path.exists(folds_path):
        raise SystemExit("[aqnet] folds.json not found — run the features "
                         "stage first.")
    df = pd.read_parquet(frame_path)
    folds_meta = _read_json(folds_path)
    if folds_meta["n_rows"] != len(df):
        raise SystemExit("[aqnet] folds.json row count does not match "
                         "training_frame.parquet — re-run the features stage.")
    return df, folds_meta


def _load_nbr_overrides(n_rows, stage):
    """Per-fold neighbor overrides from nbr_overrides_loso.npz, or None.

    The features stage saves arrays named f{fold}__{column}; this rebuilds the
    {fold: {column: full-length array}} dict models_tabular.train_cv expects.
    Missing file degrades to None with a printed warning (full-pool neighbor
    columns are then used — the pre-override behavior); wrong-length arrays
    are a hard error because they mean the frame was rebuilt after the
    overrides were computed.
    """
    path = artifact("nbr_overrides_loso.npz")
    if not os.path.exists(path):
        _skip(stage, "per-fold neighbor overrides",
              "nbr_overrides_loso.npz not found (re-run the features stage); "
              "falling back to full-pool neighbor columns")
        return None
    z = np.load(path)
    overrides = {}
    for key in z.files:
        if "__" not in key or not key.startswith("f"):
            continue
        fold_s, col = key.split("__", 1)
        try:
            fold = int(fold_s[1:])
        except ValueError:
            continue
        arr = np.asarray(z[key], dtype=np.float64)
        if arr.shape != (n_rows,):
            raise SystemExit(
                "[aqnet] nbr_overrides_loso.npz arrays do not match the "
                "training frame — re-run the features stage.")
        overrides.setdefault(fold, {})[col] = arr
    if not overrides:
        _skip(stage, "per-fold neighbor overrides",
              "nbr_overrides_loso.npz holds no f{fold}__{col} arrays")
        return None
    n_cols = len(next(iter(overrides.values())))
    _say(f"{stage}: per-fold neighbor overrides loaded "
         f"({len(overrides)} folds x {n_cols} columns)")
    return overrides


def _finite_metrics(validation, y, pred):
    """Metrics restricted to rows where both target and prediction are finite."""
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(pred)
    if not ok.any():
        return {"r2": None, "rmse": None, "mae": None, "bias": None, "n": 0,
                "note": "no finite prediction/target pairs"}
    return validation.metrics(y[ok], pred[ok])


def _find_estimator(fitted, name):
    """Locate a fitted estimator by model name inside a fit_full() result."""
    if not isinstance(fitted, dict):
        return None
    obj = fitted.get(name)
    if hasattr(obj, "predict"):
        return obj
    for value in fitted.values():
        if isinstance(value, dict):
            obj = value.get(name)
            if hasattr(obj, "predict"):
                return obj
    return None


def _apply_ridge_json(ridge, parts, n):
    """Apply a persisted Tier-3 ridge ({cols, coef, intercept, col_fill}).

    Mirrors fusion.predict_meta: NaN part values are replaced by that part's
    meta-training mean (col_fill) so a prediction is produced everywhere, and
    the output is clipped at 0 (PM2.5 is non-negative).
    """
    cols = list(ridge.get("cols") or [])
    coef = np.asarray(ridge.get("coef") or [], dtype=np.float64)
    if not cols or coef.shape != (len(cols),):
        raise ValueError("persisted ridge is unusable (cols/coef mismatch)")
    intercept = float(ridge.get("intercept", 0.0))
    fills = ridge.get("col_fill") or {}
    M = np.column_stack([
        np.asarray(parts.get(c, np.full(n, np.nan)), dtype=np.float64)
        for c in cols])
    for j, c in enumerate(cols):
        col = M[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            col[bad] = float(fills.get(c, 0.0))
    return np.maximum(0.0, M @ coef + intercept)


# ── Stage: data ──────────────────────────────────────────────────────────────

def stage_data(args):
    """Fetch/caches the external open datasets and record their paths."""
    import data_external

    start, end = _window(args)
    paths = _external_paths()
    paths.update({"quick": bool(args.quick), "start": start, "end": end})

    # EPA AQS daily PM2.5 — held out for EXTERNAL validation only. It is
    # never read by the features or tabular stages.
    years = QUICK_AQS_YEARS if args.quick else config.AQS_YEARS
    try:
        paths["aqs"] = data_external.fetch_aqs_daily_tx(years=years)
        _say(f"data: EPA AQS daily PM2.5 (years {years[0]}-{years[-1]}) -> "
             f"{paths['aqs']}")
    except Exception as e:
        paths["aqs"] = None
        _skip("data", "EPA AQS download", f"{type(e).__name__}: {e}")

    # NASA GEOS-CF surface PM2.5 via OPeNDAP.
    if args.skip_geoscf:
        paths["geoscf"] = None
        _skip("data", "GEOS-CF fetch", "--skip-geoscf was passed")
    else:
        try:
            paths["geoscf"] = data_external.fetch_geoscf_pm25(start, end)
            _say(f"data: GEOS-CF surface PM2.5 {start}..{end} -> "
                 f"{paths['geoscf']}")
        except Exception as e:
            paths["geoscf"] = None
            _skip("data", "GEOS-CF fetch",
                  f"{type(e).__name__}: {e} (geoscf_pm25 will be NaN)")

    # MERRA-2 aerosol species + PBLH via earthaccess. fetch_merra2 itself
    # returns None (with printed instructions) when credentials are absent.
    if args.skip_merra2:
        paths["merra2"] = None
        _skip("data", "MERRA-2 fetch", "--skip-merra2 was passed")
    else:
        try:
            paths["merra2"] = data_external.fetch_merra2(start, end)
            if paths["merra2"] is None:
                _skip("data", "MERRA-2 fetch",
                      "no Earthdata credentials (see instructions above); "
                      "MERRA-2 columns will be NaN")
            else:
                _say(f"data: MERRA-2 species + PBLH -> {paths['merra2']}")
        except Exception as e:
            paths["merra2"] = None
            _skip("data", "MERRA-2 fetch", f"{type(e).__name__}: {e}")

    _write_json(artifact("external_paths.json"), paths)


# ── Stage: features ──────────────────────────────────────────────────────────

def stage_features(args):
    """Build the training frame, freeze the CV folds, precompute overrides."""
    import features
    import validation

    ext = _external_paths()
    geoscf = ext.get("geoscf")
    merra2 = ext.get("merra2")
    if geoscf is None:
        _skip("features", "GEOS-CF join",
              "no parquet recorded by the data stage (geoscf_pm25 -> NaN)")
    if merra2 is None:
        _skip("features", "MERRA-2 join",
              "no parquet recorded by the data stage (merra2_* -> NaN)")

    _say(f"features: building training frame (correction={args.correction})")
    df = features.build_training_frame(correction=args.correction,
                                       geoscf_parquet=geoscf,
                                       merra2_parquet=merra2)

    start, end = _window(args)
    d = pd.to_datetime(df["date"])
    df = df[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))]
    df = df.reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit(f"[aqnet] no training rows in window {start}..{end}")
    _say(f"features: {len(df):,} sensor-day rows, "
         f"{df['sensor_id'].nunique()} sensors, window {start}..{end}")

    df.to_parquet(artifact("training_frame.parquet"), index=False)
    _say(f"wrote {artifact('training_frame.parquet')}")

    # Freeze the folds now so every later stage (and any re-run) sees the
    # exact same splits.
    n_loso = QUICK_LOSO_FOLDS if args.quick else FULL_LOSO_FOLDS
    n_block = QUICK_BLOCK_FOLDS if args.quick else FULL_BLOCK_FOLDS
    cutoff = QUICK_TEMPORAL_CUTOFF if args.quick else config.TEMPORAL_CUTOFF

    loso_folds = validation.make_loso_folds(df, n_folds=n_loso)
    loso_assign = np.full(len(df), -1, dtype=np.int64)
    for k, (_, test) in enumerate(loso_folds):
        loso_assign[test] = k

    block_assign = np.full(len(df), -1, dtype=np.int64)
    for k, (_, test) in enumerate(
            validation.make_spatial_block_folds(df, n_blocks=n_block)):
        block_assign[test] = k

    _, temporal_test = validation.temporal_split(df, cutoff=cutoff)
    temporal_is_test = np.zeros(len(df), dtype=np.int64)
    temporal_is_test[np.asarray(temporal_test, dtype=np.int64)] = 1

    _write_json(artifact("folds.json"), {
        "n_rows": len(df),
        "quick": bool(args.quick),
        "correction": args.correction,
        "loso_n_folds": n_loso,
        "loso_fold": loso_assign.tolist(),
        "block_n_folds": n_block,
        "spatial_block_fold": block_assign.tolist(),
        "temporal_cutoff": cutoff,
        "temporal_is_test": temporal_is_test.tolist(),
    })

    # Per-fold neighbor overrides for the frozen LOSO folds: every nbr_*
    # column recomputed for EVERY row against a pool restricted to that
    # fold's TRAIN rows, so held-out sensors are scored without cross-fold
    # neighbor leakage. The full-pool nbr_* columns stay in the frame but are
    # used ONLY by fit_full / external-AQS (deployment-mode predictions).
    try:
        _say(f"features: recomputing neighbor features per LOSO fold "
             f"({len(loso_folds)} folds)")
        overrides = features.neighbor_features_per_fold(df, loso_folds)
        payload = {}
        for fold, colmap in overrides.items():
            for col, arr in colmap.items():
                payload[f"f{int(fold)}__{col}"] = np.asarray(
                    arr, dtype=np.float64)
        np.savez_compressed(artifact("nbr_overrides_loso.npz"), **payload)
        n_cols = len(next(iter(overrides.values()))) if overrides else 0
        _say(f"wrote {artifact('nbr_overrides_loso.npz')} "
             f"({len(overrides)} folds x {n_cols} columns)")
    except Exception as e:
        traceback.print_exc()
        _skip("features", "per-fold neighbor overrides",
              f"{type(e).__name__}: {e} (tabular/ablation will fall back to "
              "full-pool neighbor columns)")


# ── Stage: tabular (Tier 1) ──────────────────────────────────────────────────

def stage_tabular(args):
    """LOSO cross-validate the GBM ensemble and the quantile heads.

    Both train_cv and train_quantile_cv receive the per-fold neighbor
    overrides so the nbr_* features a fold sees were pooled from that fold's
    TRAIN sensors only. The headline OOF is the leave-one-fold-out blend
    ("oof_lofo"): fold f's simplex weights are fit on the OTHER folds' OOF
    rows, so no fold influences its own blend. The pooled-weight blend is
    persisted alongside (fit_full uses pooled weights for deployment).
    """
    import features
    import validation
    import models_tabular

    df, folds_meta = _load_frame_and_folds()
    cols = features.feature_columns(df)
    loso = _folds_from_assign(folds_meta["loso_fold"])
    y = df["target"].to_numpy(dtype=np.float64)
    overrides = _load_nbr_overrides(len(df), "tabular")
    _say(f"tabular: {len(df):,} rows x {len(cols)} features, "
         f"{len(loso)} LOSO folds")

    res = models_tabular.train_cv(df, cols, loso,
                                  fold_col_overrides=overrides)
    payload = {
        "oof": np.asarray(res["oof"], dtype=np.float64),
        "oof_lofo": np.asarray(res["oof_lofo"], dtype=np.float64),
        "y": y,
        "weights_json": np.array(json.dumps(res["weights"], default=_jsonable)),
        "weights_lofo_json": np.array(
            json.dumps(res.get("weights_lofo"), default=_jsonable)),
        "fold_metrics_json": np.array(
            json.dumps(res["fold_metrics"], default=_jsonable)),
        "features_json": np.array(json.dumps(cols)),
        "used_fold_overrides": np.int8(overrides is not None),
    }
    for name, arr in res["per_model_oof"].items():
        payload[f"per_model_{name}"] = np.asarray(arr, dtype=np.float64)
    np.savez_compressed(artifact("oof_tier1.npz"), **payload)
    _say(f"wrote {artifact('oof_tier1.npz')}")

    m_lofo = _finite_metrics(validation, y, res["oof_lofo"])
    m_pool = _finite_metrics(validation, y, res["oof"])
    _say(f"tabular: LOFO blend (headline) LOSO OOF r2={m_lofo['r2']} "
         f"rmse={m_lofo['rmse']} (n={m_lofo['n']:,})")
    _say(f"tabular: pooled-weight blend LOSO OOF r2={m_pool['r2']} "
         f"rmse={m_pool['rmse']}; pooled weights={res['weights']}")

    try:
        qres = models_tabular.train_quantile_cv(df, cols, loso,
                                                fold_col_overrides=overrides)
        qpayload = {"quantiles": np.array(sorted(qres["oof_q"]),
                                          dtype=np.float64)}
        for q, arr in qres["oof_q"].items():
            qpayload[f"q{int(round(float(q) * 100)):02d}"] = np.asarray(
                arr, dtype=np.float64)
        np.savez_compressed(artifact("quantile_oof.npz"), **qpayload)
        _say(f"wrote {artifact('quantile_oof.npz')}")
    except Exception as e:
        _skip("tabular", "quantile heads",
              f"{type(e).__name__}: {e} (conformal intervals will be skipped)")


# ── Stage: deep (Tier 2) ─────────────────────────────────────────────────────

def stage_deep(args):
    """Train FusionUNet on the extended gridded stack; cache the stack.

    The stack cache filename embeds the run tag, grid resolution, target
    correction, date window, and which optional channel groups (ctm / merra2
    / flags) were available at build time, so a stack built without MERRA-2
    is never silently reused once the credentials arrive.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        _skip("deep", "FusionUNet training", "torch is not installed")
        return

    try:
        import grids
        import models_deep
        from dataset import load_cache, save_cache  # research/deeplearning

        ext = _external_paths()
        start, end = _window(args)
        gd = _grid_deg(args)
        tag = "quick" if args.quick else "full"
        grp_tag = "-".join(
            g for g, present in (("ctm", ext.get("geoscf")),
                                 ("merra2", ext.get("merra2")),
                                 ("flags", True)) if present)
        stack_path = os.path.join(
            config.CACHE_DIR,
            f"extended_stack_{tag}_{gd:g}deg_{args.correction}_"
            f"{start.replace('-', '')}_{end.replace('-', '')}_{grp_tag}.npz")

        if os.path.exists(stack_path):
            _say(f"deep: loading cached stack {stack_path}")
            stack = load_cache(stack_path)
        else:
            _say(f"deep: building extended stack {start}..{end} at {gd} deg "
                 f"(correction={args.correction})")
            stack = grids.build_extended_stack(
                start=start, end=end, grid_deg=gd,
                geoscf_parquet=ext.get("geoscf"),
                merra2_parquet=ext.get("merra2"),
                correction=args.correction)
            save_cache(stack, stack_path)
            _say(f"deep: cached stack to {stack_path}")
        for name in ("ctm", "merra2"):
            if name not in stack["channels"]:
                _skip("deep", f"'{name}' channel group",
                      "its external parquet was not available at build time")
        if "flags" not in stack["channels"]:
            _skip("deep", "'flags' channel group",
                  "stack predates the availability-flag channels (delete the "
                  "cache file to rebuild)")

        ckpt_dir = artifact("unet")
        os.makedirs(ckpt_dir, exist_ok=True)
        res = models_deep.train_fusion_unet(stack, epochs=_epochs(args),
                                            batch_size=args.batch_size,
                                            device=args.device,
                                            checkpoint_dir=ckpt_dir)
        _write_json(artifact("unet_train.json"), {
            "best": res["best"],
            "ckpt": res["ckpt"],
            "epochs_requested": _epochs(args),
            "batch_size": args.batch_size,
            "grid_deg": gd,
            "correction": args.correction,
            "window": [start, end],
            "stack_cache": stack_path,
            "history": res.get("history"),
        })
        _say(f"deep: best holdout metrics {res['best']} -> {res['ckpt']}")
    except Exception as e:
        traceback.print_exc()
        _skip("deep", "FusionUNet training",
              f"{type(e).__name__}: {e} (fuse/validate will run without the "
              "U-Net part)")


# ── Stage: fuse (Tier 3) ─────────────────────────────────────────────────────

def stage_fuse(args):
    """Residual kriging + cross-fitted meta-learner + conformal recentering.

    Leakage discipline: every meta input is strictly out-of-fold ({"tier1":
    LOFO blend OOF, "rk": kriged residual of that blend, "unet": pixel OOF
    masked to the U-Net's held-out validation-site pixels}); the meta
    prediction on meta-train sensors is itself cross-fitted over grouped
    sensor folds; the final ridge is fit only on meta-train rows; the
    conformal delta is computed only on the sensor-disjoint calibration
    split, and its coverage/width are reported on the meta-train cross-fit
    rows (disjoint from delta fitting). Nothing here ever sees EPA AQS data.
    """
    import fusion
    import validation

    df, folds_meta = _load_frame_and_folds()
    t1_path = artifact("oof_tier1.npz")
    if not os.path.exists(t1_path):
        raise SystemExit("[aqnet] oof_tier1.npz not found — run the tabular "
                         "stage first.")
    t1 = np.load(t1_path)
    if "oof_lofo" not in t1.files:
        raise SystemExit("[aqnet] oof_tier1.npz predates the LOFO blend — "
                         "re-run the tabular stage.")
    tier1 = np.asarray(t1["oof_lofo"], dtype=np.float64)
    if len(tier1) != len(df):
        raise SystemExit("[aqnet] oof_tier1.npz length does not match the "
                         "training frame — re-run features then tabular.")
    y = df["target"].to_numpy(dtype=np.float64)
    n = len(df)
    loso = _folds_from_assign(folds_meta["loso_fold"])

    parts = {"tier1": tier1}

    # Residual kriging of the LOFO blend's errors (train-fold residuals only).
    # The kriged residual enters the meta as its OWN part ("rk") — the ridge
    # learns its weight; no tier1+rk sum is ever built here.
    try:
        parts["rk"] = fusion.residual_kriging_oof(df, tier1, loso)
        _say("fuse: residual kriging done (part 'rk')")
    except Exception as e:
        _skip("fuse", "residual kriging", f"{type(e).__name__}: {e}")

    # U-Net pixel predictions, masked to the checkpoint's held-out validation
    # sites: only rows whose pixel id (row*W+col) is in ckpt["val_sites"] are
    # out-of-sample for the U-Net, so only those rows may feed the meta.
    unet_info = _read_json(artifact("unet_train.json"))
    if unet_info is None:
        _skip("fuse", "U-Net meta input",
              "no unet_train.json (deep stage not run or it was skipped)")
    elif not os.path.exists(unet_info.get("ckpt", "")):
        _skip("fuse", "U-Net meta input",
              f"checkpoint {unet_info.get('ckpt')} not found")
    else:
        try:
            import torch
            import models_deep
            from dataset import load_cache
            stack = load_cache(unet_info["stack_cache"])
            ckpt = torch.load(unet_info["ckpt"], map_location="cpu",
                              weights_only=False)
            px = models_deep.unet_pixel_oof(df, stack, ckpt)

            g = float(stack["grid_deg"])
            lat0 = float(stack["lat"][0])
            lon0 = float(stack["lon"][0])
            height, width = len(stack["lat"]), len(stack["lon"])
            la = df["lat"].to_numpy(dtype=np.float64)
            lo = df["lon"].to_numpy(dtype=np.float64)
            finite = np.isfinite(la) & np.isfinite(lo)
            r = np.full(n, -1, dtype=np.int64)
            c = np.full(n, -1, dtype=np.int64)
            r[finite] = np.rint((la[finite] - lat0) / g).astype(np.int64)
            c[finite] = np.rint((lo[finite] - lon0) / g).astype(np.int64)
            on_grid = (r >= 0) & (r < height) & (c >= 0) & (c < width)
            pid = np.where(on_grid, r * width + c, -1)
            val_sites = np.asarray(ckpt.get("val_sites", []), dtype=np.int64)
            keep = on_grid & np.isin(pid, val_sites)
            parts["unet"] = np.where(keep, px, np.nan)
            _say(f"fuse: U-Net part masked to "
                 f"{int(np.isfinite(parts['unet']).sum()):,} rows on "
                 f"{len(val_sites)} held-out validation-site pixels")
        except Exception as e:
            traceback.print_exc()
            _skip("fuse", "U-Net meta input", f"{type(e).__name__}: {e}")

    # Sensor-disjoint split: meta-learner trains on one set of sensors,
    # conformal calibration uses the other (methodology rule 4).
    sensors = df["sensor_id"].astype(str).to_numpy()
    uniq = np.unique(sensors)
    rng = np.random.default_rng(42)
    rng.shuffle(uniq)
    n_cal = max(1, int(round(0.25 * len(uniq))))
    cal_set = set(uniq[:n_cal].tolist())
    is_cal = np.array([s in cal_set for s in sensors], dtype=bool)
    mt = ~is_cal
    _say(f"fuse: {len(uniq) - n_cal} meta-train sensors / "
         f"{n_cal} calibration sensors "
         f"({int(mt.sum()):,} / {int(is_cal.sum()):,} rows)")

    # Cross-fitted meta on META-TRAIN sensors: grouped 4-fold over sensor
    # ids, so the Tier-3 prediction on every meta-train row is itself
    # out-of-fold. Rows never in any fold stay NaN.
    tier3_crossfit = np.full(n, np.nan)
    crossfit_cols = []
    mt_idx = np.where(mt)[0]
    try:
        sub_parts = {name: np.asarray(arr, dtype=np.float64)[mt_idx]
                     for name, arr in parts.items()}
        cf_pred, _cf_models, crossfit_cols = fusion.cross_fit_meta(
            y[mt_idx], sub_parts, groups=sensors[mt_idx], n_folds=4, seed=42)
        tier3_crossfit[mt_idx] = np.asarray(cf_pred, dtype=np.float64)
        m_cf = _finite_metrics(validation, y[mt_idx], tier3_crossfit[mt_idx])
        _say(f"fuse: cross-fit meta on meta-train sensors r2={m_cf['r2']} "
             f"rmse={m_cf['rmse']} (n={m_cf['n']:,}); parts "
             f"{list(crossfit_cols)}")
    except Exception as e:
        traceback.print_exc()
        _skip("fuse", "cross-fit meta", f"{type(e).__name__}: {e}")

    # Final ridge on ALL meta-train OOF rows; scores the calibration sensors
    # and serves as the deployment-mode combiner (Tier-3 at AQS).
    meta, used_cols = fusion.stack_meta(y, parts, mask=mt)
    pred_meta = fusion.predict_meta(meta, parts)
    m = _finite_metrics(validation, y[is_cal], pred_meta[is_cal])
    _say(f"fuse: final ridge on held-out calibration sensors r2={m['r2']} "
         f"rmse={m['rmse']} (n={m['n']:,}); parts used: {list(used_cols)}")

    # Conformal: recenter the Tier-1 quantile band on the Tier-3 prediction
    # (cross-fit values on meta-train rows, final-ridge values on calibration
    # rows), calibrate the CQR delta on the calibration split only, then
    # report empirical coverage AND mean width on the meta-train cross-fit
    # rows — disjoint from the rows that fit delta.
    delta = np.nan
    coverage_mt = np.nan
    width_mt = np.nan
    n_cal_rows = 0
    n_mt_rows = 0
    tier3_center = np.where(is_cal, pred_meta, tier3_crossfit)
    lo_c = np.full(n, np.nan)
    hi_c = np.full(n, np.nan)
    q_path = artifact("quantile_oof.npz")
    if not os.path.exists(q_path):
        _skip("fuse", "conformal calibration",
              "quantile_oof.npz not found (tabular quantile heads skipped)")
    else:
        q = np.load(q_path)
        if not {"q05", "q50", "q95"} <= set(q.files):
            _skip("fuse", "conformal calibration",
                  "quantile_oof.npz lacks q05/q50/q95 arrays")
        else:
            lo, hi = fusion.conformal_recenter(tier3_center, q["q05"],
                                               q["q50"], q["q95"])
            okc = is_cal & np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
            if not okc.any():
                _skip("fuse", "conformal calibration",
                      "no finite recentered-band rows on the calibration split")
            else:
                delta = float(fusion.conformal_intervals(
                    y[okc], lo[okc], hi[okc], alpha=0.1))
                n_cal_rows = int(okc.sum())
                lo_c, hi_c = lo - delta, hi + delta
                okm = mt & np.isfinite(y) & np.isfinite(lo_c) & np.isfinite(hi_c)
                if okm.any():
                    coverage_mt = float(np.mean((y[okm] >= lo_c[okm])
                                                & (y[okm] <= hi_c[okm])))
                    width_mt = float(np.mean(hi_c[okm] - lo_c[okm]))
                    n_mt_rows = int(okm.sum())
                _say(f"fuse: conformal delta={delta:.4f} "
                     f"(alpha=0.1, {n_cal_rows:,} calibration rows); "
                     f"meta-train cross-fit coverage={coverage_mt:.4f} "
                     f"mean width={width_mt:.4f} ({n_mt_rows:,} rows)")

    payload = {
        "oof_meta": np.asarray(pred_meta, dtype=np.float64),
        "tier3_crossfit": tier3_crossfit,
        "tier3_center": np.asarray(tier3_center, dtype=np.float64),
        "y": y,
        "is_calibration": is_cal.astype(np.int8),
        "conformal_delta": np.float64(delta),
        "conformal_lo": lo_c,
        "conformal_hi": hi_c,
        "conformal_coverage_meta_train": np.float64(coverage_mt),
        "conformal_mean_width_meta_train": np.float64(width_mt),
        "conformal_n_calibration": np.int64(n_cal_rows),
        "conformal_n_meta_train": np.int64(n_mt_rows),
        "used_cols_json": np.array(json.dumps(list(used_cols))),
        "crossfit_cols_json": np.array(json.dumps(list(crossfit_cols))),
    }
    for name, arr in parts.items():
        payload[f"part_{name}"] = np.asarray(arr, dtype=np.float64)
    try:
        payload["meta_coef_json"] = np.array(json.dumps({
            "coef": np.asarray(meta.coef_).ravel().tolist(),
            "intercept": float(np.asarray(meta.intercept_).ravel()[0]),
            "cols": list(used_cols),
            "col_fill": {k: float(v)
                         for k, v in getattr(meta, "col_fill_", {}).items()},
        }))
    except Exception:
        pass  # meta model without exposed coefficients — weights not persisted
    np.savez_compressed(artifact("oof_meta.npz"), **payload)
    _say(f"wrote {artifact('oof_meta.npz')}")


# ── Stage: ablation ──────────────────────────────────────────────────────────

def _paired_delta_r2_ci(y, pred_ref, pred_alt, clusters, n_boot=1000, seed=0):
    """Cluster-bootstrap CI for R2(alt) - R2(ref), paired on shared rows.

    Unique clusters (sensor ids) are resampled with replacement; both
    predictions are scored on the SAME resampled rows, so the delta
    distribution reflects the paired error structure rather than two
    independent bootstraps. Rows where either prediction or y is non-finite
    are excluded up front (both variants face identical rows).
    """
    y = np.asarray(y, dtype=np.float64)
    pr = np.asarray(pred_ref, dtype=np.float64)
    pa = np.asarray(pred_alt, dtype=np.float64)
    cl = np.asarray(clusters)
    ok = np.isfinite(y) & np.isfinite(pr) & np.isfinite(pa)
    nan_out = {"delta_r2": float("nan"), "ci95": (float("nan"), float("nan")),
               "n_boot": n_boot, "n_clusters": 0, "n_rows": int(ok.sum())}
    if ok.sum() < 3:
        return nan_out
    y, pr, pa, cl = y[ok], pr[ok], pa[ok], cl[ok]

    def _r2(pred):
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot <= 0:
            return float("nan")
        return 1.0 - float(np.sum((pred - y) ** 2)) / ss_tot

    point = _r2(pa) - _r2(pr)
    idx_by = pd.Series(np.arange(len(y))).groupby(pd.Series(cl)).indices
    pools = [np.asarray(v) for v in idx_by.values()]
    rng = np.random.default_rng(seed)
    deltas = np.full(n_boot, np.nan)
    for b in range(n_boot):
        pick = rng.integers(0, len(pools), len(pools))
        rows = np.concatenate([pools[i] for i in pick])
        t = y[rows]
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        if ss_tot <= 0:
            continue
        deltas[b] = (float(np.sum((pr[rows] - t) ** 2))
                     - float(np.sum((pa[rows] - t) ** 2))) / ss_tot
    lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    return {"delta_r2": float(point), "ci95": (float(lo), float(hi)),
            "n_boot": n_boot, "n_clusters": len(pools), "n_rows": int(len(y))}


def stage_ablation(args):
    """Tier-1 feature-set ablations on the SAME frozen LOSO folds.

    Variants (quick mode runs only the first two):
      primary            the current feature set (with per-fold neighbor
                         overrides), identical to the tabular stage
      plus_demographics  primary + the 4 demographic EJScreen columns.
                         The frame is REBUILT with keep_demographics=True and
                         the feature list is constructed explicitly here,
                         deliberately bypassing feature_columns() (which
                         never returns demographics). This variant exists
                         ONLY as an ablation — demographics are never model
                         inputs anywhere else in AQNet.
      no_external        primary minus every geoscf_* / merra2_* column
      no_neighbor        primary minus every nbr_* column (overrides skipped)

    Each variant is scored by its LOFO-blend LOSO OOF; deltas vs primary are
    paired cluster-bootstrap delta-R2 CIs (clusters = sensors). Everything
    lands in metrics_ablation.json.
    """
    import features
    import validation
    import models_tabular

    df, folds_meta = _load_frame_and_folds()
    loso = _folds_from_assign(folds_meta["loso_fold"])
    y = df["target"].to_numpy(dtype=np.float64)
    sensor_ids = df["sensor_id"].astype(str).to_numpy()
    cols = features.feature_columns(df)
    overrides = _load_nbr_overrides(len(df), "ablation")
    correction = folds_meta.get("correction") or args.correction

    variants = ["primary", "plus_demographics"]
    if not args.quick:
        variants += ["no_external", "no_neighbor"]
    _say(f"ablation: variants {variants} on {len(loso)} frozen LOSO folds"
         + (" (quick: shrunken variant set)" if args.quick else ""))

    # plus_demographics needs a frame that still CARRIES the demographic
    # columns (the frozen frame never does) — rebuild with the same window
    # and verify row-for-row alignment so the frozen folds stay valid.
    demo_frame = None
    demo_cols = None
    if "plus_demographics" in variants:
        try:
            ext = _external_paths()
            _say("ablation: rebuilding frame with keep_demographics=True "
                 "(reference columns for this variant only)")
            demo_frame = features.build_training_frame(
                correction=correction,
                geoscf_parquet=ext.get("geoscf"),
                merra2_parquet=ext.get("merra2"),
                keep_demographics=True)
            start, end = _window(args)
            dd = pd.to_datetime(demo_frame["date"])
            demo_frame = demo_frame[(dd >= pd.Timestamp(start))
                                    & (dd <= pd.Timestamp(end))]
            demo_frame = demo_frame.reset_index(drop=True)
            aligned = (
                len(demo_frame) == len(df)
                and np.array_equal(
                    demo_frame["sensor_id"].astype(str).to_numpy(),
                    sensor_ids)
                and np.array_equal(
                    pd.to_datetime(demo_frame["date"]).to_numpy(),
                    pd.to_datetime(df["date"]).to_numpy()))
            if not aligned:
                raise RuntimeError(
                    "rebuilt keep_demographics frame does not align with the "
                    "frozen training frame (source data changed since the "
                    "features stage?)")
            demo_present = [c for c in config.EXCLUDED_DEMOGRAPHIC
                            if c in demo_frame.columns]
            # Explicit list, bypassing feature_columns(): the ONE place in
            # AQNet where demographics may enter a model, as an ablation.
            demo_cols = cols + demo_present
            _say(f"ablation: plus_demographics adds {demo_present}")
        except Exception as e:
            traceback.print_exc()
            _skip("ablation", "plus_demographics variant",
                  f"{type(e).__name__}: {e}")
            demo_frame = None

    results = {}
    preds = {}
    for name in variants:
        if name == "primary":
            frame, vcols, ov = df, cols, overrides
        elif name == "plus_demographics":
            if demo_frame is None:
                continue
            frame, vcols, ov = demo_frame, demo_cols, overrides
        elif name == "no_external":
            vcols = [c for c in cols
                     if not (c.startswith("geoscf_")
                             or c.startswith("merra2_"))]
            if len(vcols) == len(cols):
                # No external columns were dropped, so this variant would be
                # byte-identical to primary and its delta-R2 a meaningless 0.
                _skip("ablation", "variant 'no_external'",
                      "the frame carries no geoscf_*/merra2_* columns "
                      "(external sources not fetched) — the variant would "
                      "duplicate 'primary'")
                continue
            frame, ov = df, overrides
        elif name == "no_neighbor":
            vcols = [c for c in cols if not c.startswith("nbr_")]
            frame, ov = df, None  # no neighbor columns -> no overrides
        else:
            continue

        _say(f"ablation: variant '{name}' ({len(vcols)} features)")
        try:
            res = models_tabular.train_cv(frame, vcols, loso,
                                          fold_col_overrides=ov,
                                          return_fitted=False)
        except Exception as e:
            traceback.print_exc()
            _skip("ablation", f"variant '{name}'", f"{type(e).__name__}: {e}")
            continue
        pred = np.asarray(res["oof_lofo"], dtype=np.float64)
        preds[name] = pred
        ok = np.isfinite(y) & np.isfinite(pred)
        entry = {"n_features": len(vcols),
                 "used_fold_overrides": ov is not None,
                 "loso_lofo": _finite_metrics(validation, y, pred)}
        if ok.any():
            entry["bootstrap_ci"] = validation.bootstrap_ci(
                y[ok], pred[ok], cluster=sensor_ids[ok])
        results[name] = entry
        del res
        gc.collect()

    if "primary" in preds:
        for name, pred in preds.items():
            if name == "primary":
                continue
            results[name]["delta_r2_vs_primary"] = _paired_delta_r2_ci(
                y, preds["primary"], pred, sensor_ids)
    else:
        _skip("ablation", "paired delta-R2 CIs", "primary variant failed")

    note = ("plus_demographics exists ONLY as an ablation: demographic "
            "EJScreen columns are never model inputs in the primary AQNet "
            "configuration, any other stage, or any deployed model.")
    _write_json(artifact("metrics_ablation.json"), {
        "quick": bool(args.quick),
        "loso_n_folds": folds_meta.get("loso_n_folds"),
        "target_correction": correction,
        "variants": results,
        "note": note,
    })
    _say(f"ablation: NOTE — {note}")
    del demo_frame
    gc.collect()


# ── Stage: validate ──────────────────────────────────────────────────────────

def stage_validate(args):
    """Compute every metrics artifact this run can support, then SUMMARY.md."""
    import features
    import validation
    import models_tabular

    df, folds_meta = _load_frame_and_folds()
    y = df["target"].to_numpy(dtype=np.float64)
    cols = features.feature_columns(df)
    loso = _folds_from_assign(folds_meta["loso_fold"])
    rng = np.random.default_rng(0)

    lat = df["lat"].to_numpy(dtype=np.float64)
    lon = df["lon"].to_numpy(dtype=np.float64)
    days = pd.to_datetime(df["date"]).dt.normalize().to_numpy()
    sensor_ids = df["sensor_id"].astype(str).to_numpy()

    def enrich(y_true, pred, e_lat=None, e_lon=None, e_days=None,
               cluster=None):
        """metrics + CLUSTER bootstrap CI + AQI skill (+ per-day Moran's I)."""
        y_true = np.asarray(y_true, dtype=np.float64)
        pred = np.asarray(pred, dtype=np.float64)
        ok = np.isfinite(y_true) & np.isfinite(pred)
        if not ok.any():
            return {"n": 0, "note": "no finite prediction/target pairs"}
        m = validation.metrics(y_true[ok], pred[ok])
        m["bootstrap_ci"] = validation.bootstrap_ci(
            y_true[ok], pred[ok],
            cluster=(np.asarray(cluster)[ok] if cluster is not None else None))
        if cluster is not None:
            m["bootstrap_ci"]["cluster"] = "sensor"
        m["aqi"] = validation.aqi_category_metrics(y_true[ok], pred[ok])
        if e_lat is not None and e_days is not None:
            resid = y_true[ok] - pred[ok]
            m["morans_i_daily"] = validation.morans_i_daily(
                resid, np.asarray(e_lat)[ok], np.asarray(e_lon)[ok],
                np.asarray(e_days)[ok])
        return m

    # ── LOSO metrics: Tier-1 blends, per-model, Tier-2/3, strata, R2 split ──
    out = {"n_rows": len(df), "quick": bool(folds_meta.get("quick")),
           "correction": folds_meta.get("correction"),
           "loso_n_folds": folds_meta.get("loso_n_folds")}
    lofo = None
    t1_path = artifact("oof_tier1.npz")
    if os.path.exists(t1_path):
        t1 = np.load(t1_path)
        if "oof_lofo" in t1.files:
            lofo = np.asarray(t1["oof_lofo"], dtype=np.float64)
        else:
            lofo = np.asarray(t1["oof"], dtype=np.float64)
            out["tier1_note"] = ("oof_lofo absent from oof_tier1.npz — "
                                 "pooled blend used; re-run the tabular stage")
        out["tier1_blend"] = enrich(y, lofo, lat, lon, days, sensor_ids)
        out["tier1_blend"]["blend"] = "leave-one-fold-out (headline)"
        out["tier1_blend"]["weights_pooled"] = json.loads(
            t1["weights_json"].item())
        if "weights_lofo_json" in t1.files:
            out["tier1_blend"]["weights_lofo"] = json.loads(
                t1["weights_lofo_json"].item())
        out["tier1_blend_pooled_weights"] = _finite_metrics(
            validation, y, t1["oof"])
        out["tier1_per_model"] = {
            k[len("per_model_"):]: _finite_metrics(validation, y, t1[k])
            for k in t1.files if k.startswith("per_model_")}

        # Strata: smoke-flagged rows, dusty rows (top quartile of the dust
        # column), and the clean remainder — computed on the headline OOF.
        try:
            smoke = (df["hms_smoke"].to_numpy(dtype=np.float64)
                     if "hms_smoke" in df.columns
                     else np.zeros(len(df)))
            dust_src = None
            for cand in ("dust", "merra2_dust25"):
                if cand in df.columns and df[cand].notna().any():
                    dust_src = cand
                    break
            dust_vals = (df[dust_src].to_numpy(dtype=np.float64)
                         if dust_src else np.full(len(df), np.nan))
            out["strata_tier1_lofo"] = validation.strata_metrics(
                y, lofo, smoke, dust_vals)
            out["strata_tier1_lofo"]["dust_column"] = dust_src or "none"
        except Exception as e:
            _skip("validate", "strata metrics", f"{type(e).__name__}: {e}")

        # Spatial vs temporal R2 decomposition of the headline OOF.
        out["spatial_temporal_r2"] = {}
        try:
            ok = np.isfinite(y) & np.isfinite(lofo)
            out["spatial_temporal_r2"]["tier1_lofo"] = (
                validation.spatial_temporal_r2(y[ok], lofo[ok],
                                               sensor_ids[ok]))
        except Exception as e:
            _skip("validate", "spatial/temporal R2 (tier1)",
                  f"{type(e).__name__}: {e}")
    else:
        _skip("validate", "Tier-1 LOSO metrics", "oof_tier1.npz not found")

    q_path = artifact("quantile_oof.npz")
    if os.path.exists(q_path):
        q = np.load(q_path)
        if "q05" in q.files and "q95" in q.files:
            ok = (np.isfinite(y) & np.isfinite(q["q05"])
                  & np.isfinite(q["q95"]))
            if ok.any():
                cov = float(np.mean((y[ok] >= q["q05"][ok])
                                    & (y[ok] <= q["q95"][ok])))
                out["tier1_quantiles"] = {
                    "empirical_coverage_q05_q95": cov,
                    "n": int(ok.sum())}

    meta_path = artifact("oof_meta.npz")
    if os.path.exists(meta_path):
        mz = np.load(meta_path)
        pred_meta = mz["oof_meta"]
        is_cal = mz["is_calibration"].astype(bool)
        out["tier3_meta_all_rows"] = enrich(y, pred_meta, lat, lon, days,
                                            sensor_ids)
        out["tier3_meta_calibration_sensors_only"] = enrich(
            y[is_cal], pred_meta[is_cal], cluster=sensor_ids[is_cal])
        out["tier3_meta_parts"] = json.loads(mz["used_cols_json"].item())

        # Tier-3 cross-fit rows (fully OOF on meta-train sensors): AQI and
        # cluster-bootstrap CIs ride along inside enrich.
        if "tier3_crossfit" in mz.files:
            cf = np.asarray(mz["tier3_crossfit"], dtype=np.float64)
            out["tier3_crossfit_meta_train"] = enrich(y, cf, lat, lon, days,
                                                      sensor_ids)
            try:
                ok = np.isfinite(y) & np.isfinite(cf)
                out.setdefault("spatial_temporal_r2", {})["tier3_crossfit"] = (
                    validation.spatial_temporal_r2(y[ok], cf[ok],
                                                   sensor_ids[ok]))
            except Exception as e:
                _skip("validate", "spatial/temporal R2 (tier3)",
                      f"{type(e).__name__}: {e}")
        else:
            _skip("validate", "Tier-3 cross-fit metrics",
                  "oof_meta.npz predates cross_fit_meta — re-run fuse")

        # Tier-2 U-Net row: metrics on the rows where the fuse stage kept the
        # pixel OOF (the U-Net's held-out validation-site sensor-days).
        if "part_unet" in mz.files:
            out["tier2_unet"] = enrich(y, mz["part_unet"], lat, lon, days,
                                       sensor_ids)
            out["tier2_unet"]["note"] = ("finite only on the U-Net's "
                                         "held-out validation-site rows")
        else:
            _skip("validate", "Tier-2 U-Net metrics",
                  "no part_unet in oof_meta.npz (deep stage skipped)")

        delta = float(mz["conformal_delta"])
        if np.isfinite(delta):
            out["conformal"] = {
                "alpha": 0.1,
                "delta": delta,
                "n_calibration": int(mz["conformal_n_calibration"])
                if "conformal_n_calibration" in mz.files else None,
                "note": ("band recentered on Tier-3 (CQR); delta fit on "
                         "calibration sensors; coverage/width on the "
                         "DISJOINT meta-train cross-fit rows"),
            }
            if "conformal_coverage_meta_train" in mz.files:
                out["conformal"]["coverage_meta_train_crossfit"] = float(
                    mz["conformal_coverage_meta_train"])
            if "conformal_mean_width_meta_train" in mz.files:
                out["conformal"]["mean_width_meta_train_crossfit"] = float(
                    mz["conformal_mean_width_meta_train"])
            if "conformal_n_meta_train" in mz.files:
                out["conformal"]["n_meta_train_crossfit"] = int(
                    mz["conformal_n_meta_train"])
    else:
        _skip("validate", "Tier-3 meta metrics", "oof_meta.npz not found "
              "(run the fuse stage)")
    _write_json(artifact("metrics_loso.json"), out)
    del out
    gc.collect()

    # ── Spatial-block CV (retrains the tabular tier on region folds) ──
    # return_fitted=False: per-fold estimators are dropped as scored (RAM).
    # Neighbor columns here are the frame's full-pool values (per-fold
    # overrides exist only for the frozen LOSO folds), as before.
    blocks = _folds_from_assign(folds_meta["spatial_block_fold"])
    try:
        _say(f"validate: spatial-block CV ({len(blocks)} region folds)")
        res_b = models_tabular.train_cv(df, cols, blocks,
                                        return_fitted=False)
        _write_json(artifact("metrics_spatial_block.json"), {
            "n_blocks": len(blocks),
            "tier1_blend_lofo": enrich(y, res_b["oof_lofo"], lat, lon, days,
                                       sensor_ids),
            "tier1_blend_pooled": _finite_metrics(validation, y, res_b["oof"]),
            "weights": res_b["weights"],
            "fold_metrics": res_b["fold_metrics"],
            "note": ("neighbor features use the frame's full-pool columns "
                     "(per-fold overrides are computed for the LOSO folds)"),
        })
        del res_b
        gc.collect()
    except Exception as e:
        _skip("validate", "spatial-block CV", f"{type(e).__name__}: {e}")

    # ── Temporal holdout (train before cutoff, test after) ──
    is_test = np.asarray(folds_meta["temporal_is_test"], dtype=bool)
    train_idx = np.where(~is_test)[0]
    test_idx = np.where(is_test)[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        _skip("validate", "temporal holdout",
              f"empty split at cutoff {folds_meta.get('temporal_cutoff')} "
              f"({len(train_idx)} train / {len(test_idx)} test rows)")
    else:
        try:
            _say(f"validate: temporal holdout "
                 f"(cutoff {folds_meta.get('temporal_cutoff')})")
            res_t = models_tabular.train_cv(df, cols, [(train_idx, test_idx)],
                                            return_fitted=False)
            _write_json(artifact("metrics_temporal.json"), {
                "cutoff": folds_meta.get("temporal_cutoff"),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "tier1_blend": enrich(y[test_idx],
                                      np.asarray(res_t["oof"])[test_idx],
                                      lat[test_idx], lon[test_idx],
                                      days[test_idx], sensor_ids[test_idx]),
            })
            del res_t
            gc.collect()
        except Exception as e:
            _skip("validate", "temporal holdout", f"{type(e).__name__}: {e}")

    # ── Baselines: interpolation-only, raw CTM priors, debiased priors ──
    baselines = {}
    for name, fn in [("nearest_sensor",
                      lambda: validation.baseline_nearest(df, loso)),
                     ("idw_k8", lambda: validation.baseline_idw(df, loso)),
                     ("ordinary_kriging",
                      lambda: validation.baseline_kriging(df, loso))]:
        try:
            _say(f"validate: baseline {name}")
            baselines[name] = enrich(y, fn(), lat, lon, days, sensor_ids)
        except Exception as e:
            _skip("validate", f"baseline {name}", f"{type(e).__name__}: {e}")
    for col in ("cams_pm25", "geoscf_pm25", "merra2_pm25_proxy"):
        if col in df.columns and df[col].notna().any():
            baselines[f"raw_{col}"] = enrich(
                y, validation.baseline_column(df, col), lat, lon, days,
                sensor_ids)
            # Mean-debiased variant: subtract the frame-wide mean bias of the
            # prior vs the target — a favorable-to-baseline adjustment that
            # asks "does the ML model beat even a bias-corrected CTM?".
            vals = df[col].to_numpy(dtype=np.float64)
            okp = np.isfinite(vals) & np.isfinite(y)
            offset = float(np.mean(vals[okp] - y[okp])) if okp.any() else 0.0
            baselines[f"raw_{col}_debiased_mean"] = enrich(
                y, validation.baseline_column(df, col, offset=offset),
                lat, lon, days, sensor_ids)
            baselines[f"raw_{col}_debiased_mean"]["offset"] = offset
        else:
            _skip("validate", f"baseline raw_{col}",
                  "column absent or all-NaN in the training frame")
    _write_json(artifact("metrics_baselines.json"), baselines)
    del baselines
    gc.collect()

    # ── External EPA AQS validation (data the models never trained on) ──
    fitted = None
    predict_fn = None
    ext = _external_paths()
    aqs = ext.get("aqs")
    if aqs and os.path.exists(aqs):
        try:
            _say("validate: fitting full-data ensemble for external AQS "
                 "validation")
            fitted = models_tabular.fit_full(df, cols)
            predict_fn = lambda X: models_tabular.predict_full(fitted, X)  # noqa: E731
            # Quick mode trains on a 3-month window, so scoring against the
            # full multi-year AQS record would measure out-of-window
            # extrapolation, not model skill. Subset AQS to the quick window.
            if args.quick:
                _aqs_df = pd.read_parquet(aqs)
                _aqs_df["date"] = pd.to_datetime(_aqs_df["date"])
                _aqs_df = _aqs_df[(_aqs_df["date"] >= QUICK_START)
                                  & (_aqs_df["date"] <= QUICK_END)]
                aqs = artifact("aqs_quick_subset.parquet")
                _aqs_df.to_parquet(aqs, index=False)
                _say(f"validate: quick mode — AQS subset to "
                     f"{QUICK_START}..{QUICK_END} ({len(_aqs_df):,} site-days)")
            m = validation.external_aqs_validation(
                predict_fn, aqs, geoscf_parquet=ext.get("geoscf"),
                merra2_parquet=ext.get("merra2"),
                save_rows_to=artifact("external_aqs_predictions.parquet"))
            m["note"] = ("EPA AQS FRM/FEM monitors are fully held out: "
                         "never used in training or feature computation "
                         "for training rows.")
            if args.quick:
                m["note"] += (" QUICK MODE: AQS subset to the quick window; "
                              "smoke-test signal only.")

            # Deployment-mode Tier-3 at AQS site-days: full-data Tier-1
            # prediction + per-day kriging of the full-data training
            # residuals to the sites + the U-Net surface value at each
            # site's pixel/day (NaN -> ridge col_fill), combined by the
            # final ridge persisted by the fuse stage.
            try:
                m["tier3"] = _tier3_at_aqs(args, df, y, days, lat, lon,
                                           fitted, aqs, ext, folds_meta)
            except Exception as e:
                traceback.print_exc()
                _skip("validate", "deployment-mode Tier-3 at AQS",
                      f"{type(e).__name__}: {e}")
            _write_json(artifact("metrics_external_aqs.json"), m)
        except Exception as e:
            traceback.print_exc()
            _skip("validate", "external AQS validation",
                  f"{type(e).__name__}: {e}")
    else:
        _skip("validate", "external AQS validation",
              "no AQS parquet recorded by the data stage")
    gc.collect()

    # ── Interpretability: SHAP summary + permutation importance ──
    try:
        import interpret
        if fitted is None:
            _say("validate: fitting full-data ensemble for interpretability")
            fitted = models_tabular.fit_full(df, cols)
            predict_fn = lambda X: models_tabular.predict_full(fitted, X)  # noqa: E731
        lgbm_est = _find_estimator(fitted, "lgbm")
        if lgbm_est is None:
            _skip("validate", "SHAP summary",
                  "no fitted LightGBM model in the full-data ensemble")
        else:
            sample = df[cols].sample(min(SHAP_SAMPLE_ROWS, len(df)),
                                     random_state=0)
            interpret.shap_summary(lgbm_est, sample,
                                   artifact("shap_summary.png"))
            _say(f"wrote {artifact('shap_summary.png')}")
        pick = rng.choice(len(df), min(PERMUTATION_SAMPLE_ROWS, len(df)),
                          replace=False)
        interpret.permutation_report(predict_fn, df.iloc[pick][cols],
                                     y[pick], cols,
                                     artifact("permutation_report.json"))
        _say(f"wrote {artifact('permutation_report.json')}")
    except Exception as e:
        _skip("validate", "interpretability report",
              f"{type(e).__name__}: {e}")
    del fitted
    gc.collect()

    write_summary()


def _tier3_at_aqs(args, df, y, days, lat, lon, fitted, aqs_parquet,
                  ext, folds_meta):
    """Deployment-mode Tier-3 metrics at EPA AQS site-days.

    Components (all full-data — this is the deployed configuration, not a
    cross-validated one, hence the explicit label):
      tier1  full-data ensemble prediction at each AQS site-day, on features
             built by features.build_site_features (PurpleAir/CAMS/tract
             sources only — AQS values never enter any input)
      rk     per-day validation.krige_to_sites of the full-data model's
             training residuals (0.0 on days with no training residuals)
      unet   U-Net surface value at each site's pixel/day when the cached
             stack + checkpoint exist, else NaN (the ridge fills NaN parts
             with their meta-training means)
    combined by the final Ridge persisted in oof_meta.npz.
    """
    import features
    import validation
    import models_tabular

    meta_path = artifact("oof_meta.npz")
    if not os.path.exists(meta_path):
        raise RuntimeError("oof_meta.npz not found — run the fuse stage")
    mz = np.load(meta_path)
    if "meta_coef_json" not in mz.files:
        raise RuntimeError("oof_meta.npz carries no persisted ridge "
                           "coefficients — re-run the fuse stage")
    ridge = json.loads(mz["meta_coef_json"].item())

    sites = pd.read_parquet(aqs_parquet)
    correction = folds_meta.get("correction") or args.correction
    _say(f"validate: Tier-3 at AQS — building site features at "
         f"{len(sites):,} site-days (correction={correction})")
    sf = features.build_site_features(sites, correction=correction,
                                     geoscf_parquet=ext.get("geoscf"),
                                     merra2_parquet=ext.get("merra2"))

    # Guard: every feature the fitted bundle expects must exist (NaN is fine
    # — the boosters handle it; the RF imputes with training medians).
    missing = [c for c in fitted["features"] if c not in sf.columns]
    if missing:
        print(f"[aqnet] validate: Tier-3 at AQS — site frame lacks {missing}; "
              f"left NaN")
        for c in missing:
            sf[c] = np.nan

    tier1_site = np.asarray(models_tabular.predict_full(fitted, sf),
                            dtype=np.float64)

    # Per-day kriging of full-data training residuals to the AQS sites.
    resid_full = y - np.asarray(models_tabular.predict_full(fitted, df),
                                dtype=np.float64)
    ok_res = np.isfinite(resid_full)
    day_to_rows = pd.Series(np.arange(len(df))).groupby(
        pd.Series(days)).indices
    site_days = pd.to_datetime(sf["date"]).dt.normalize().to_numpy()
    s_lat = sf["lat"].to_numpy(dtype=np.float64)
    s_lon = sf["lon"].to_numpy(dtype=np.float64)
    rk_site = np.zeros(len(sf), dtype=np.float64)
    site_by_day = pd.Series(np.arange(len(sf))).groupby(
        pd.Series(site_days)).indices
    n_kriged = 0
    for day, s_idx in site_by_day.items():
        s_idx = np.asarray(s_idx)
        tr = day_to_rows.get(day)
        if tr is None:
            continue  # neutral 0.0: no same-day training residuals
        tr = np.asarray(tr)
        tr = tr[ok_res[tr]]
        if len(tr) == 0:
            continue
        rk_site[s_idx] = validation.krige_to_sites(
            lat[tr], lon[tr], resid_full[tr], s_lat[s_idx], s_lon[s_idx],
            max_train=KRIGE_MAX_TRAIN_PER_DAY)
        n_kriged += 1
    _say(f"validate: Tier-3 at AQS — kriged residuals on {n_kriged} days")

    # U-Net surface value at each site's pixel/day (NaN when unavailable —
    # the ridge col_fill covers those rows).
    unet_site = np.full(len(sf), np.nan)
    unet_info = _read_json(artifact("unet_train.json"))
    if (unet_info and os.path.exists(unet_info.get("ckpt", ""))
            and os.path.exists(unet_info.get("stack_cache", ""))):
        try:
            import models_deep
            from dataset import load_cache
            stack = load_cache(unet_info["stack_cache"])
            unet_site = models_deep.unet_pixel_oof(sf, stack,
                                                   unet_info["ckpt"])
            _say(f"validate: Tier-3 at AQS — U-Net surface at "
                 f"{int(np.isfinite(unet_site).sum()):,} site-days")
        except Exception as e:
            _skip("validate", "U-Net surface at AQS sites",
                  f"{type(e).__name__}: {e} (ridge col_fill used)")
    else:
        _skip("validate", "U-Net surface at AQS sites",
              "no checkpoint/stack cache (ridge col_fill used)")

    tier3_site = _apply_ridge_json(ridge, {"tier1": tier1_site,
                                           "rk": rk_site,
                                           "unet": unet_site}, len(sf))

    y_true = sf["pm25_aqs"].to_numpy(dtype=np.float64)
    site_ids = (sf["site_id"].astype(str).to_numpy()
                if "site_id" in sf.columns
                else np.arange(len(sf)).astype(str))
    ok = np.isfinite(y_true) & np.isfinite(tier3_site)
    block = _finite_metrics(validation, y_true, tier3_site)
    if ok.any():
        block["bootstrap_ci"] = validation.bootstrap_ci(
            y_true[ok], tier3_site[ok], cluster=site_ids[ok])
        block["bootstrap_ci"]["cluster"] = "site"
        block["aqi"] = validation.aqi_category_metrics(y_true[ok],
                                                       tier3_site[ok])
    block["label"] = "deployment-mode Tier-3: full-data components"
    block["ridge_cols"] = list(ridge.get("cols") or [])
    block["n_unet_finite"] = int(np.isfinite(unet_site).sum())
    block["n_site_days"] = int(len(sf))

    # Merge Tier-3 predictions into the per-site-day parquet (figure F12).
    pred_path = artifact("external_aqs_predictions.parquet")
    if os.path.exists(pred_path) and "site_id" in sf.columns:
        t3 = sf[["site_id", "date"]].copy()
        t3["pred_tier3"] = tier3_site
        merged = pd.read_parquet(pred_path)
        merged = merged.drop(columns=["pred_tier3"], errors="ignore").merge(
            t3, on=["site_id", "date"], how="left")
        merged.to_parquet(pred_path, index=False)
    return block


# ── SUMMARY.md ───────────────────────────────────────────────────────────────

_SUMMARY_SECTIONS = [
    ("metrics_loso.json", "Leave-one-sensor-out (LOSO) cross-validation"),
    ("metrics_spatial_block.json", "Spatial-block cross-validation"),
    ("metrics_temporal.json", "Temporal holdout"),
    ("metrics_ablation.json",
     "Ablation study — Tier 1 on the frozen LOSO folds"),
    ("metrics_baselines.json", "Interpolation and raw-CTM baselines"),
    ("metrics_external_aqs.json",
     "External EPA AQS validation (never trained on)"),
    ("unet_train.json", "Tier 2 — FusionUNet grouped-site-holdout training"),
]

_SUMMARY_MAX_ROWS = 160


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _flatten(obj, prefix=""):
    """Flatten nested dicts to (dotted-key, printable-value) rows."""
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


def write_summary():
    """Auto-generate SUMMARY.md from whatever metrics artifacts exist.

    Only computed numbers appear here; sections whose stage did not run are
    listed as absent rather than filled in. The new blocks — ablation,
    strata, spatial/temporal decomposition, per-day Moran's I, conformal
    coverage + width, the Tier-2 U-Net row, and Tier-3 at AQS — ride inside
    the flattened metrics files above.
    """
    lines = [
        "# AQNet — Run Summary",
        "",
        f"Auto-generated by pipeline_colab.py on "
        f"{pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}. "
        "Every number below was computed by this run's stages from the "
        "metrics artifacts alongside this file; nothing is hand-entered.",
        "",
        "Production baseline for context: the deployed Shared Skies 4-model "
        "tree ensemble reports LOSO R2 = 0.7136 (`models/metrics.json`). "
        "That number describes the production system, not AQNet.",
        "",
    ]
    for fname, title in _SUMMARY_SECTIONS:
        lines.append(f"## {title}")
        lines.append("")
        obj = _read_json(artifact(fname))
        if obj is None:
            lines.append(f"_`{fname}` not present — its stage was not run "
                         "or was skipped (see the run log)._")
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
    others = ["training_frame.parquet", "folds.json",
              "nbr_overrides_loso.npz", "oof_tier1.npz",
              "quantile_oof.npz", "oof_meta.npz", "shap_summary.png",
              "permutation_report.json"]
    lines.append("## Artifacts")
    lines.append("")
    for name in others:
        mark = "present" if os.path.exists(artifact(name)) else "absent"
        lines.append(f"- `{name}` — {mark}")
    lines.append("")
    path = artifact("SUMMARY.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    _say(f"wrote {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

_STAGES = {
    "data": stage_data,
    "features": stage_features,
    "tabular": stage_tabular,
    "deep": stage_deep,
    "fuse": stage_fuse,
    "ablation": stage_ablation,
    "validate": stage_validate,
}

_STAGE_ORDER = ["data", "features", "tabular", "deep", "fuse", "ablation",
                "validate"]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="AQNet three-tier PM2.5 research pipeline "
                    "(offline research track — not the live map).")
    ap.add_argument("stage", choices=_STAGE_ORDER + ["all"],
                    help="pipeline stage to run ('all' runs every stage "
                         "in order)")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: %s..%s window, %g-deg grid, %d LOSO "
                         "folds, %d epochs, 2-variant ablation"
                         % (QUICK_START, QUICK_END, QUICK_GRID_DEG,
                            QUICK_LOSO_FOLDS, QUICK_EPOCHS))
    ap.add_argument("--skip-merra2", action="store_true",
                    help="do not attempt the MERRA-2 fetch (no Earthdata "
                         "credentials needed; merra2_* features become NaN)")
    ap.add_argument("--skip-geoscf", action="store_true",
                    help="do not attempt the GEOS-CF OPeNDAP fetch "
                         "(geoscf_pm25 becomes NaN)")
    ap.add_argument("--skip-ablation", action="store_true",
                    help="omit the ablation stage from an 'all' run")
    ap.add_argument("--epochs", type=int, default=None,
                    help="FusionUNet epochs (default %d, or %d with --quick)"
                         % (FULL_EPOCHS, QUICK_EPOCHS))
    ap.add_argument("--batch-size", type=int, default=None,
                    help="FusionUNet batch size (default: auto — larger on "
                         "CUDA, small on CPU)")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda", "mps"],
                    help="deep-stage device (auto: cuda, else Apple mps, "
                         "else cpu)")
    ap.add_argument("--grid-deg", type=float, default=None,
                    help="deep-stage grid resolution in degrees (default "
                         "%g, or %g with --quick)" % (config.GRID_DEG,
                                                      QUICK_GRID_DEG))
    ap.add_argument("--correction", choices=["barkjohn", "raw"],
                    default="barkjohn",
                    help="target correction: Barkjohn et al. (2021) "
                         "PurpleAir correction (default) or raw ATM as a "
                         "sensitivity option")
    args = ap.parse_args(argv)

    stages = _STAGE_ORDER if args.stage == "all" else [args.stage]
    if args.stage == "all" and args.skip_ablation:
        stages = [s for s in stages if s != "ablation"]
        _say("--skip-ablation: the ablation stage is omitted from this run")
    for name in stages:
        t0 = time.time()
        _say(f"── stage: {name} " + "─" * max(0, 58 - len(name)))
        _STAGES[name](args)
        _say(f"── stage {name} done in {time.time() - t0:.1f}s")
    _say(f"artifacts in {config.ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
