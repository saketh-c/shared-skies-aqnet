"""Result-dependent figures for AQNet (F08-F22).

Every function reads ONLY pipeline artifacts under artifacts/ (plus the
tract GeoJSON for light-gray map context), renders one figure to the shared
visualization standard in fig_style, and returns the list of written paths.
When a required artifact is missing the function prints a clear skip message
and returns None -- nothing is ever fabricated.

Quick-mode guard: when the artifacts carry the pipeline's quick flag, every
render is forced to preview=True so the QUICK-MODE watermark is stamped; a
quick-mode number can never end up in a clean figure.

Signatures: def FXX_name(mode="paper", preview=False) -> list[str] | None.
Run ``python fig_results.py`` to test-render everything (preview by default).
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from fig_style import (COLORS, SEQUENTIAL_CMAP, FIG_W, annotate_metrics,
                       save_fig, set_style)

ART = os.path.join(_HERE, "artifacts")
ROOT = os.path.dirname(os.path.dirname(_HERE))
TRACTS_GEOJSON = os.path.join(ROOT, "backend", "static",
                              "texas_all_tracts.geojson")

UNITS = "PM2.5 (ug/m3)"
INK = "#333333"
INK2 = "#555555"

# EPA 2024 daily PM2.5 AQI breakpoints (upper bounds, ug/m3) -- mirrors
# validation.AQI_UPPER_BOUNDS / AQI_CATEGORIES so figure categories match
# the pipeline's metrics exactly.
AQI_UPPER_BOUNDS = np.array([9.0, 35.4, 55.4, 125.4, 225.4])
AQI_CATEGORIES = ["Good", "Moderate", "USG", "Unhealthy",
                  "Very Unhealthy", "Hazardous"]


# ── artifact access ─────────────────────────────────────────────────────────

def _path(name):
    return os.path.join(ART, name)


def _json(name):
    p = _path(name)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _npz(name):
    p = _path(name)
    if not os.path.exists(p):
        return None
    return np.load(p, allow_pickle=False)


def _parquet(name):
    p = _path(name)
    if not os.path.exists(p):
        return None
    return pd.read_parquet(p)


def _skip(figname, missing):
    print(f"[fig_results] {figname}: SKIPPED - missing artifact(s): {missing}")
    return None


_QUICK = None


def _quick():
    """True when the artifacts on disk came from a quick-mode run."""
    global _QUICK
    if _QUICK is None:
        q = False
        for name in ("folds.json", "metrics_loso.json"):
            d = _json(name)
            if isinstance(d, dict) and d.get("quick"):
                q = True
                break
        _QUICK = q
        if q:
            print("[fig_results] quick-mode artifacts detected - all renders "
                  "forced to preview=True (watermarked)")
    return _QUICK


def _guard_preview(preview):
    return bool(preview or _quick())


def _lw(mode):
    return 1.6 if mode == "paper" else 3.0


def _regression_metrics(y, p):
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    sst = float(np.sum((y - y.mean()) ** 2))
    sse = float(np.sum((y - p) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    return r2, float(np.sqrt(sse / len(y))), int(len(y))


# ── map context (tract polygons, loaded once, drawn rasterized) ─────────────

_TRACT_VERTS = None


def _tract_verts():
    global _TRACT_VERTS
    if _TRACT_VERTS is None:
        if not os.path.exists(TRACTS_GEOJSON):
            _TRACT_VERTS = []
            print(f"[fig_results] note: {TRACTS_GEOJSON} not found - maps "
                  "render without tract context")
        else:
            with open(TRACTS_GEOJSON, "r", encoding="utf-8") as fh:
                gj = json.load(fh)
            verts = []
            for feat in gj.get("features", []):
                geom = feat.get("geometry") or {}
                gtype, coords = geom.get("type"), geom.get("coordinates")
                if gtype == "Polygon":
                    polys = [coords]
                elif gtype == "MultiPolygon":
                    polys = coords
                else:
                    continue
                for poly in polys:
                    if poly:
                        verts.append(np.asarray(poly[0], dtype=np.float64))
            _TRACT_VERTS = verts
    return _TRACT_VERTS


def _map_context(ax):
    """Light-gray tract polygons (rasterized) + lat/lon axis cosmetics."""
    from matplotlib.collections import PolyCollection

    verts = _tract_verts()
    if verts:
        pc = PolyCollection(verts, facecolor="#f0f0f0", edgecolor="#d9d9d9",
                            linewidth=0.15, rasterized=True, zorder=0)
        ax.add_collection(pc)
    ax.set_aspect(1.0 / np.cos(np.radians(31.0)))
    ax.grid(False)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")


def _tag(ax, text):
    ax.set_title(text, loc="left", color=INK)


# ── F08: main results forest plot ───────────────────────────────────────────

def F08_main_results(mode="paper", preview=False):
    """Forest plot: R2 (with cluster-bootstrap 95% CI) and RMSE per method."""
    m_loso = _json("metrics_loso.json")
    m_base = _json("metrics_baselines.json")
    if m_loso is None or m_base is None:
        return _skip("F08_main_results",
                     "metrics_loso.json / metrics_baselines.json")
    preview = _guard_preview(preview)
    set_style(mode)

    def row(label, src, key, color_key, open_marker):
        m = src.get(key)
        if not isinstance(m, dict) or "r2" not in m:
            return None
        ci = m.get("bootstrap_ci") or {}
        return dict(label=label, color=COLORS[color_key], open=open_marker,
                    r2=float(m["r2"]), rmse=float(m["rmse"]),
                    r2_ci=tuple(ci.get("r2", (np.nan, np.nan))),
                    rmse_ci=tuple(ci.get("rmse", (np.nan, np.nan))))

    spec = [
        ("Nearest sensor", m_base, "nearest_sensor", "baseline_nearest", False),
        ("IDW (k=8)", m_base, "idw_k8", "baseline_idw", False),
        ("Ordinary kriging", m_base, "ordinary_kriging", "baseline_kriging", False),
        ("CAMS raw", m_base, "raw_cams_pm25", "prior_cams", False),
        ("CAMS debiased", m_base, "raw_cams_pm25_debiased_mean", "prior_cams", True),
        ("GEOS-CF raw", m_base, "raw_geoscf_pm25", "prior_geoscf", False),
        ("GEOS-CF debiased", m_base, "raw_geoscf_pm25_debiased_mean", "prior_geoscf", True),
        ("MERRA-2 raw", m_base, "raw_merra2_pm25", "prior_merra2", False),
        ("MERRA-2 debiased", m_base, "raw_merra2_pm25_debiased_mean", "prior_merra2", True),
        ("Tier-1 LOFO blend", m_loso, "tier1_blend", "tier1", False),
        ("Tier-2 U-Net (val sites)", m_loso, "tier2_unet", "tier2", False),
        ("Tier-3 cross-fit", m_loso, "tier3_crossfit_meta_train", "tier3", False),
    ]
    rows = [r for r in (row(*s) for s in spec) if r is not None]
    if not rows:
        return _skip("F08_main_results", "no scored methods in metrics files")

    n = len(rows)
    ys = np.arange(n)[::-1]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, sharey=True, figsize=(FIG_W["double"], 0.34 * n + 1.2))

    def forest(ax, val_key, ci_key, xmin=None):
        for r, y in zip(rows, ys):
            v = r[val_key]
            lo, hi = r[ci_key]
            mfc = "white" if r["open"] else r["color"]
            clipped = xmin is not None and v < xmin
            if not clipped and np.isfinite(lo) and np.isfinite(hi):
                lo_c = lo if xmin is None else max(lo, xmin)
                ax.plot([lo_c, hi], [y, y], color=r["color"], lw=1.3,
                        solid_capstyle="butt", zorder=3)
                for end, orig in ((lo_c, lo), (hi, hi)):
                    if xmin is None or orig >= xmin:
                        ax.plot([end, end], [y - 0.16, y + 0.16],
                                color=r["color"], lw=1.1, zorder=3)
            if clipped:
                ax.plot([xmin + 0.015], [y], marker="<", ms=6,
                        color=r["color"], ls="none", zorder=4,
                        clip_on=False)
                ax.text(xmin + 0.05, y, f"{v:.1f}", va="center", ha="left",
                        color=INK2, fontsize=plt.rcParams["legend.fontsize"])
            else:
                ax.plot([v], [y], marker="o", ms=5.5, mfc=mfc,
                        mec=r["color"], mew=1.1, ls="none", zorder=5)
        ax.set_yticks(ys)
        ax.set_yticklabels([r["label"] for r in rows], color=INK)
        ax.set_ylim(-0.7, n - 0.3)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True)

    xmin = -0.5
    forest(ax1, "r2", "r2_ci", xmin=xmin)
    prod = 0.7136
    ax1.axvline(prod, ls="--", lw=1.0, color=INK2, zorder=2)
    ax1.text(prod, 1.01, f"production baseline\n(LOSO R2 = {prod:.4f})",
             transform=ax1.get_xaxis_transform(), ha="center", va="bottom",
             color=INK2, fontsize=plt.rcParams["legend.fontsize"])
    ax1.set_xlim(xmin - 0.03, max(0.9, prod + 0.12))
    ax1.set_xlabel("R2 (cluster-bootstrap 95% CI)")
    _tag(ax1, "(a) R2")

    forest(ax2, "rmse", "rmse_ci")
    hi_all = [r["rmse_ci"][1] for r in rows if np.isfinite(r["rmse_ci"][1])]
    ax2.set_xlim(0, 1.06 * max([r["rmse"] for r in rows] + hi_all))
    ax2.set_xlabel(f"RMSE (ug/m3)")
    _tag(ax2, "(b) RMSE")

    fig.tight_layout()
    paths = save_fig(fig, "F08_main_results", mode, preview=preview)
    plt.close(fig)
    return paths


# ── F09: predicted vs observed hexbin ───────────────────────────────────────

def F09_pred_vs_obs(mode="paper", preview=False):
    """Hexbin density of OOF predicted vs observed for Tier-1 and Tier-3."""
    z1 = _npz("oof_tier1.npz")
    zm = _npz("oof_meta.npz")
    if z1 is None and zm is None:
        return _skip("F09_pred_vs_obs", "oof_tier1.npz / oof_meta.npz")
    m_loso = _json("metrics_loso.json") or {}
    preview = _guard_preview(preview)
    set_style(mode)

    panels = []
    if z1 is not None:
        panels.append(("Tier-1 LOFO (OOF)", np.asarray(z1["y"]),
                       np.asarray(z1["oof_lofo"]),
                       m_loso.get("tier1_blend")))
    if zm is not None and "tier3_crossfit" in zm.files:
        y3 = np.asarray(zm["y"])
        p3 = np.asarray(zm["tier3_crossfit"])
        if "is_calibration" in zm.files:
            mt = np.asarray(zm["is_calibration"]).astype(bool)
            y3, p3 = y3[~mt], p3[~mt]
        panels.append(("Tier-3 cross-fit (OOF)", y3, p3,
                       m_loso.get("tier3_crossfit_meta_train")))
    if not panels:
        return _skip("F09_pred_vs_obs", "no OOF prediction arrays")

    fig, axes = plt.subplots(1, len(panels),
                             figsize=(FIG_W["double"], 3.3), squeeze=False)
    vmax = 0.0
    for _, y, p, _m in panels:
        ok = np.isfinite(y) & np.isfinite(p)
        vmax = max(vmax, float(np.nanpercentile(y[ok], 99.9)),
                   float(np.nanpercentile(p[ok], 99.9)))
    vmax = float(np.ceil(vmax / 5.0) * 5.0)

    for i, (ax, (title, y, p, m)) in enumerate(zip(axes[0], panels)):
        ok = np.isfinite(y) & np.isfinite(p)
        hb = ax.hexbin(y[ok], p[ok], gridsize=40, cmap=SEQUENTIAL_CMAP,
                       bins="log", mincnt=1, extent=(0, vmax, 0, vmax),
                       linewidths=0.2)
        ax.plot([0, vmax], [0, vmax], ls="--", lw=1.0, color=INK2, zorder=3)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlabel(f"Observed {UNITS}")
        ax.set_ylabel(f"Predicted {UNITS}")
        _tag(ax, f"({'ab'[i]}) {title}")
        if isinstance(m, dict):
            r2, rmse, nn = m.get("r2"), m.get("rmse"), m.get("n")
        else:
            r2, rmse, nn = _regression_metrics(y[ok], p[ok])
        annotate_metrics(ax, {"R2": f"{r2:.3f}", "RMSE": f"{rmse:.2f} ug/m3",
                              "n": f"{nn:,}"})
        fig.colorbar(hb, ax=ax, shrink=0.85, label="site-days")

    fig.tight_layout()
    paths = save_fig(fig, "F09_pred_vs_obs", mode, preview=preview)
    plt.close(fig)
    return paths


# ── F10: spatial-block CV ───────────────────────────────────────────────────

def F10_spatial_block(mode="paper", preview=False):
    """Per-region-fold R2 bars + KMeans block map of sensors."""
    m_sb = _json("metrics_spatial_block.json")
    if m_sb is None:
        return _skip("F10_spatial_block", "metrics_spatial_block.json")
    fold_metrics = m_sb.get("fold_metrics") or []
    if not fold_metrics:
        return _skip("F10_spatial_block",
                     "fold_metrics in metrics_spatial_block.json")
    folds = _json("folds.json")
    tf = _parquet("training_frame.parquet")
    has_map = (folds is not None and tf is not None
               and "spatial_block_fold" in folds
               and len(folds["spatial_block_fold"]) == len(tf))
    if not has_map:
        print("[fig_results] F10_spatial_block: folds.json/training_frame "
              "unavailable - rendering bars panel only")
    preview = _guard_preview(preview)
    set_style(mode)

    ncols = 2 if has_map else 1
    fig, axes = plt.subplots(
        1, ncols, figsize=(FIG_W["double"] if has_map else FIG_W["single"] + 1.2, 3.3),
        squeeze=False, gridspec_kw={"width_ratios": [1.05, 1][:ncols]})
    ax = axes[0][0]

    model_order = [m for m in ("lgbm", "xgb", "rf")
                   if m in fold_metrics[0].get("models", {})]
    # Tier-1 component models: shades of the tier-1 hue + hatches (identity
    # is never color-alone).
    shades = {"lgbm": "#93c4e4", "xgb": "#4795c9", "rf": COLORS["tier1"]}
    hatches = {"lgbm": "", "xgb": "///", "rf": "xxx"}
    x = np.arange(len(fold_metrics))
    w = 0.75 / max(1, len(model_order))
    for j, mname in enumerate(model_order):
        vals = [fm["models"][mname]["r2"] for fm in fold_metrics]
        ax.bar(x + (j - (len(model_order) - 1) / 2.0) * w, vals,
               width=w * 0.88, color=shades.get(mname, COLORS["neutral"]),
               hatch=hatches.get(mname, ""), edgecolor=INK, linewidth=0.4,
               label=mname.upper())
    ax.axhline(0, color=INK2, lw=0.8)
    all_r2 = [fm["models"][m]["r2"] for fm in fold_metrics for m in model_order]
    span = max(all_r2) - min(min(all_r2), 0.0)
    for i, fm in enumerate(fold_metrics):
        top = max(0.0, max(fm["models"][m]["r2"] for m in model_order))
        ax.text(x[i], top + 0.03 * span, f"n={fm.get('n_test', 0):,}",
                ha="center", va="bottom", color=INK2,
                fontsize=plt.rcParams["legend.fontsize"])
    ax.set_ylim(min(min(all_r2), 0.0) - 0.05 * span,
                max(all_r2) + 0.22 * span)
    ax.set_xticks(x)
    ax.set_xticklabels([f"block {fm.get('fold', i)}"
                        for i, fm in enumerate(fold_metrics)])
    ax.set_xlabel("Spatial block fold (KMeans regions)")
    ax.set_ylabel("R2")
    ax.xaxis.grid(False)
    ax.legend(loc="upper left", frameon=False, title="Tier-1 components")
    _tag(ax, "(a) Held-out R2 per spatial block")

    if has_map:
        axm = axes[0][1]
        sb = np.asarray(folds["spatial_block_fold"])
        g = (tf.assign(block=sb)
               .groupby("sensor_id")
               .agg(lat=("lat", "first"), lon=("lon", "first"),
                    block=("block", "first")))
        blocks = sorted(g["block"].unique())
        cmap = mpl.colormaps[SEQUENTIAL_CMAP]
        bcolors = cmap(np.linspace(0.12, 0.88, len(blocks)))
        markers = ["o", "s", "^", "D", "v", "P"]
        _map_context(axm)
        for i, b in enumerate(blocks):
            sub = g[g["block"] == b]
            axm.scatter(sub["lon"], sub["lat"], s=24, color=bcolors[i],
                        marker=markers[i % len(markers)], edgecolor="white",
                        linewidth=0.3, zorder=3,
                        label=f"block {b} ({len(sub)} sensors)")
        axm.set_xlim(-107.2, -93.1)
        axm.set_ylim(25.4, 36.9)
        axm.legend(loc="lower left", frameon=True, framealpha=0.85,
                   edgecolor="#cccccc")
        _tag(axm, "(b) KMeans spatial blocks")

    fig.tight_layout()
    paths = save_fig(fig, "F10_spatial_block", mode, preview=preview)
    plt.close(fig)
    return paths


# ── F11: temporal holdout ───────────────────────────────────────────────────

def F11_temporal(mode="paper", preview=False):
    """Daily statewide mean observed vs OOF-predicted, train period shaded."""
    tf = _parquet("training_frame.parquet")
    z1 = _npz("oof_tier1.npz")
    if tf is None or z1 is None:
        return _skip("F11_temporal", "training_frame.parquet / oof_tier1.npz")
    m_t = _json("metrics_temporal.json")
    folds = _json("folds.json") or {}
    cutoff = (m_t or {}).get("cutoff") or folds.get("temporal_cutoff")
    if cutoff is None:
        return _skip("F11_temporal",
                     "temporal cutoff (metrics_temporal.json / folds.json)")
    zm = _npz("oof_meta.npz")
    preview = _guard_preview(preview)
    set_style(mode)

    daily = pd.DataFrame({
        "date": pd.to_datetime(tf["date"]).to_numpy(),
        "obs": np.asarray(z1["y"], dtype=float),
        "t1": np.asarray(z1["oof_lofo"], dtype=float),
    })
    if zm is not None and "tier3_crossfit" in zm.files:
        daily["t3"] = np.asarray(zm["tier3_crossfit"], dtype=float)
    daily = daily.groupby("date", as_index=False).mean().sort_values("date")
    cutoff_ts = pd.Timestamp(cutoff)

    fig, ax = plt.subplots(figsize=(FIG_W["double"], 2.9))
    ax.axvspan(daily["date"].min(), cutoff_ts, color=COLORS["train_shade"],
               zorder=0)
    lw = _lw(mode)
    ax.plot(daily["date"], daily["obs"], color=COLORS["observed"], lw=lw,
            ls="-", label="Observed (statewide sensor mean)")
    ax.plot(daily["date"], daily["t1"], color=COLORS["tier1"], lw=lw,
            ls="--", label="Tier-1 LOFO OOF")
    if "t3" in daily.columns:
        ax.plot(daily["date"], daily["t3"], color=COLORS["tier3"], lw=lw,
                ls=":", label="Tier-3 cross-fit OOF")
    ax.axvline(cutoff_ts, ls="--", lw=0.9, color=INK2, zorder=2)
    ax.text(cutoff_ts, 0.02, f" cutoff {cutoff} ",
            transform=ax.get_xaxis_transform(), rotation=90, ha="right",
            va="bottom", color=INK2, fontsize=plt.rcParams["legend.fontsize"])

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=COLORS["train_shade"],
                         label="train period (temporal split)"))
    labels.append("train period (temporal split)")
    ax.legend(handles, labels, loc="upper right", frameon=True,
              framealpha=0.85, edgecolor="#cccccc")

    if m_t and isinstance(m_t.get("tier1_blend"), dict):
        t = m_t["tier1_blend"]
        annotate_metrics(ax, {
            "Temporal-test R2 (Tier-1)": f"{t['r2']:.3f}",
            "RMSE": f"{t['rmse']:.2f} ug/m3",
            "n test": f"{t['n']:,}"})
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Daily mean {UNITS}")
    locator = mpl.dates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mpl.dates.ConciseDateFormatter(locator))

    fig.tight_layout()
    paths = save_fig(fig, "F11_temporal", mode, preview=preview)
    plt.close(fig)
    return paths


# ── F12: external EPA AQS validation ────────────────────────────────────────

_AQS_PRED_CANDIDATES = ("external_aqs_predictions.parquet",
                        "external_aqs_preds.parquet")


def F12_external_aqs(mode="paper", preview=False):
    """(a) pred vs AQS scatter, (b) per-site R2 map, (c) bias histogram.

    Needs a per-row external prediction artifact (site_id, date, lat, lon,
    pm25_aqs + pred_tier1/pred_tier3 columns); the aggregate-only
    metrics_external_aqs.json cannot support these panels.
    """
    m_ext = _json("metrics_external_aqs.json")
    pred_df = None
    for name in _AQS_PRED_CANDIDATES:
        pred_df = _parquet(name)
        if pred_df is not None:
            break
    if pred_df is None:
        return _skip("F12_external_aqs",
                     f"per-row AQS predictions ({_AQS_PRED_CANDIDATES[0]})")

    def first_col(cands):
        for c in cands:
            if c in pred_df.columns:
                return c
        return None

    obs_col = first_col(["pm25_aqs", "obs", "y"])
    t1_col = first_col(["pred_tier1", "tier1", "pred_full", "pred"])
    t3_col = first_col(["pred_tier3", "tier3"])
    if obs_col is None or (t1_col is None and t3_col is None):
        return _skip("F12_external_aqs",
                     "pm25_aqs + prediction columns in the AQS artifact")
    preview = _guard_preview(preview)
    set_style(mode)

    obs = pred_df[obs_col].to_numpy(dtype=float)
    series = []
    if t1_col is not None:
        series.append(("Tier-1", pred_df[t1_col].to_numpy(dtype=float),
                       COLORS["tier1"], "o"))
    if t3_col is not None:
        series.append(("Tier-3", pred_df[t3_col].to_numpy(dtype=float),
                       COLORS["tier3"], "^"))

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(FIG_W["double"], 2.9),
        gridspec_kw={"width_ratios": [1, 1.15, 1]})

    vmax = float(np.nanpercentile(
        np.concatenate([obs] + [p for _, p, _, _ in series]), 99.5))
    for label, p, col, mk in series:
        ax1.scatter(obs, p, s=7, color=col, marker=mk, alpha=0.35,
                    linewidths=0, label=label, rasterized=True)
    ax1.plot([0, vmax], [0, vmax], ls="--", lw=1.0, color=INK2, zorder=3)
    ax1.set_xlim(0, vmax)
    ax1.set_ylim(0, vmax)
    ax1.set_aspect("equal")
    ax1.set_xlabel(f"AQS observed {UNITS}")
    ax1.set_ylabel(f"Predicted {UNITS}")
    ax1.legend(loc="lower right", frameon=False)
    _tag(ax1, "(a) Held-out AQS monitors")
    if m_ext:
        box = {}
        if "r2" in m_ext:
            box["Tier-1 R2"] = f"{m_ext['r2']:.3f}"
        t3m = m_ext.get("tier3")
        if isinstance(t3m, dict) and "r2" in t3m:
            box["Tier-3 R2"] = f"{t3m['r2']:.3f}"
        if "n" in m_ext:
            box["n"] = f"{m_ext['n']:,}"
        if box:
            annotate_metrics(ax1, box)

    # (b) per-site R2 map for the best available tier row.
    map_label, map_pred = series[-1][0], series[-1][1]
    per_site = []
    d = pred_df.assign(_pred=map_pred)
    for sid, sub in d.groupby("site_id"):
        yv = sub[obs_col].to_numpy(dtype=float)
        pv = sub["_pred"].to_numpy(dtype=float)
        ok = np.isfinite(yv) & np.isfinite(pv)
        if ok.sum() >= 8:
            r2, _, _ = _regression_metrics(yv[ok], pv[ok])
            per_site.append((float(sub["lon"].iloc[0]),
                             float(sub["lat"].iloc[0]), r2))
    _map_context(ax2)
    if per_site:
        lons, lats, r2s = map(np.asarray, zip(*per_site))
        vmin = min(0.0, float(np.floor(np.nanmin(r2s) * 10) / 10))
        sc = ax2.scatter(lons, lats, c=r2s, cmap=SEQUENTIAL_CMAP,
                         vmin=vmin, vmax=1.0, s=26, edgecolor="white",
                         linewidth=0.3, zorder=3)
        fig.colorbar(sc, ax=ax2, shrink=0.85, label=f"per-site R2 ({map_label})")
    ax2.set_xlim(-107.2, -93.1)
    ax2.set_ylim(25.4, 36.9)
    _tag(ax2, "(b) Per-site R2")

    # (c) bias histogram.
    for label, p, col, _mk in series:
        bias = p - obs
        bias = bias[np.isfinite(bias)]
        ax3.hist(bias, bins=40, histtype="step" if label == "Tier-3" else "bar",
                 color=col, alpha=0.85 if label == "Tier-3" else 0.55,
                 label=label, linewidth=_lw(mode) * 0.8)
    ax3.axvline(0, ls="--", lw=0.9, color=INK2)
    ax3.set_xlabel("Bias, predicted - observed (ug/m3)")
    ax3.set_ylabel("site-days")
    ax3.legend(loc="upper left", frameon=False)
    _tag(ax3, "(c) Bias at AQS monitors")

    fig.tight_layout()
    paths = save_fig(fig, "F12_external_aqs", mode, preview=preview)
    plt.close(fig)
    return paths


# ── F13: conformal intervals ────────────────────────────────────────────────

def F13_conformal(mode="paper", preview=False):
    """(a) coverage vs nominal, (b) width distribution, (c) example month."""
    zm = _npz("oof_meta.npz")
    if zm is None or "conformal_lo" not in getattr(zm, "files", []):
        return _skip("F13_conformal", "oof_meta.npz (conformal_lo/hi)")
    preview = _guard_preview(preview)
    set_style(mode)

    y = np.asarray(zm["y"], dtype=float)
    center = np.asarray(zm["tier3_center"], dtype=float)
    lo = np.asarray(zm["conformal_lo"], dtype=float)
    hi = np.asarray(zm["conformal_hi"], dtype=float)
    delta = float(np.asarray(zm["conformal_delta"]))
    cal = np.asarray(zm["is_calibration"]).astype(bool)
    ev = ~cal
    # Reconstruct the pre-conformal CQR band widths from the artifacts.
    w_lo = center - lo - delta
    w_hi = hi - center - delta

    m_conf = (_json("metrics_loso.json") or {}).get("conformal", {})
    alpha0 = float(m_conf.get("alpha", 0.1))
    achieved = float(np.asarray(zm["conformal_coverage_meta_train"]))

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(FIG_W["double"], 2.8),
        gridspec_kw={"width_ratios": [1, 1, 1.35]})

    # (a) empirical coverage vs nominal, recalibrating delta on the stored
    # calibration scores for a grid of nominal levels.
    s = np.maximum((center - w_lo) - y, y - (center + w_hi))[cal]
    s = s[np.isfinite(s)]
    if s.size:
        alphas = np.unique(np.round(np.concatenate(
            [np.linspace(0.02, 0.5, 25), [alpha0]]), 4))
        nominal, empirical = [], []
        n_cal = s.size
        for a in alphas:
            q = min(1.0, (1.0 - a) * (1.0 + 1.0 / n_cal))
            d_a = float(np.quantile(s, q, method="higher"))
            lo_a = center - w_lo - d_a
            hi_a = center + w_hi + d_a
            nominal.append(1.0 - a)
            empirical.append(float(np.mean((y[ev] >= lo_a[ev])
                                           & (y[ev] <= hi_a[ev]))))
        ax1.plot([0.45, 1.0], [0.45, 1.0], ls="--", lw=1.0, color=INK2,
                 label="ideal")
        ax1.plot(nominal, empirical, color=COLORS["tier3"], lw=_lw(mode),
                 marker="o", ms=3.5, markevery=3, label="split conformal")
    ax1.plot([1.0 - alpha0], [achieved], marker="o", ms=7,
             color=COLORS["tier3"], mec="white", mew=1.0, ls="none",
             zorder=5)
    ax1.annotate(f"{achieved:.3f} at nominal {1 - alpha0:.2f}",
                 (1.0 - alpha0, achieved), xytext=(-8, -14),
                 textcoords="offset points", ha="right", color=INK,
                 fontsize=plt.rcParams["legend.fontsize"])
    ax1.set_xlim(0.45, 1.02)
    ax1.set_ylim(0.45, 1.02)
    ax1.set_xlabel("Nominal coverage")
    ax1.set_ylabel("Empirical coverage")
    ax1.legend(loc="lower right", frameon=False)
    _tag(ax1, "(a) Coverage")

    # (b) interval-width distribution on the evaluation (meta-train) rows.
    widths = (hi - lo)[ev]
    widths = widths[np.isfinite(widths)]
    ax2.hist(widths, bins=40, color=COLORS["tier3"], alpha=0.85,
             edgecolor="white", linewidth=0.3)
    mean_w = float(np.mean(widths))
    ax2.axvline(mean_w, ls="--", lw=1.0, color=INK2)
    ax2.text(mean_w, 0.97, f" mean {mean_w:.2f}",
             transform=ax2.get_xaxis_transform(), ha="left", va="top",
             color=INK2, fontsize=plt.rcParams["legend.fontsize"])
    ax2.set_xlabel(f"{int(round((1 - alpha0) * 100))}% interval width (ug/m3)")
    ax2.set_ylabel("site-days")
    _tag(ax2, "(b) Interval width")

    # (c) one example month at one sensor (band + observations).
    tf = _parquet("training_frame.parquet")
    if tf is not None and len(tf) == len(y):
        d = pd.DataFrame({
            "sensor_id": tf["sensor_id"].to_numpy(),
            "date": pd.to_datetime(tf["date"]).to_numpy(),
            "y": y, "center": center, "lo": lo, "hi": hi})
        d["ym"] = pd.PeriodIndex(d["date"], freq="M")
        pick = d.groupby(["sensor_id", "ym"]).size().idxmax()
        sub = d[(d["sensor_id"] == pick[0])
                & (d["ym"] == pick[1])].sort_values("date")
        ax3.fill_between(sub["date"], sub["lo"], sub["hi"],
                         color=COLORS["tier3"], alpha=0.22, linewidth=0,
                         label=f"{int(round((1 - alpha0) * 100))}% conformal band")
        ax3.plot(sub["date"], sub["center"], color=COLORS["tier3"],
                 lw=_lw(mode), label="Tier-3 prediction")
        ax3.plot(sub["date"], sub["y"], ls="none", marker="o", ms=3.5,
                 color=COLORS["observed"], label="observed")
        mon = pick[1].strftime("%b %Y")
        _tag(ax3, f"(c) Sensor {pick[0]}, {mon}")
        ax3.set_ylabel(UNITS)
        ax3.set_xlabel("Date")
        locator = mpl.dates.AutoDateLocator()
        ax3.xaxis.set_major_locator(locator)
        ax3.xaxis.set_major_formatter(mpl.dates.ConciseDateFormatter(locator))
        ax3.legend(loc="upper right", frameon=True, framealpha=0.85,
                   edgecolor="#cccccc")
    else:
        ax3.axis("off")
        ax3.text(0.5, 0.5, "training_frame.parquet unavailable",
                 ha="center", va="center", color=INK2,
                 transform=ax3.transAxes)

    fig.tight_layout()
    paths = save_fig(fig, "F13_conformal", mode, preview=preview)
    plt.close(fig)
    return paths


# ── F14: ablation deltas ────────────────────────────────────────────────────

def F14_ablation(mode="paper", preview=False):
    """Paired delta-R2 (variant minus primary) with cluster-bootstrap CIs."""
    m_ab = _json("metrics_ablation.json")
    if m_ab is None or not isinstance(m_ab.get("variants"), dict):
        return _skip("F14_ablation", "metrics_ablation.json")
    variants = m_ab["variants"]
    items = [(k, v) for k, v in variants.items()
             if isinstance(v, dict) and "delta_r2_vs_primary" in v]
    if not items:
        return _skip("F14_ablation",
                     "variants with delta_r2_vs_primary in metrics_ablation.json")
    order = ["plus_demographics", "no_external", "no_neighbor"]
    items.sort(key=lambda kv: (order.index(kv[0]) if kv[0] in order else 99,
                               kv[0]))
    preview = _guard_preview(preview)
    set_style(mode)

    n = len(items)
    fig, ax = plt.subplots(figsize=(FIG_W["single"] + 0.8, 0.55 * n + 1.5))
    ys = np.arange(n)[::-1]
    labels = []
    span = 0.0
    for (name, v), yy in zip(items, ys):
        d = v["delta_r2_vs_primary"]
        val = float(d["delta_r2"])
        lo, hi = [float(x) for x in d.get("ci95", (np.nan, np.nan))]
        ax.errorbar([val], [yy], xerr=[[val - lo], [hi - val]], fmt="o",
                    ms=5.5, color=COLORS["neutral"], ecolor=COLORS["neutral"],
                    elinewidth=1.3, capsize=2.5, zorder=4)
        span = max(span, abs(val), abs(lo), abs(hi))
        lab = name.replace("_", " ")
        if name == "plus_demographics":
            lab += "\n(ablation-only)"
        labels.append(lab)
    ax.axvline(0, ls="--", lw=1.0, color=INK2, zorder=2)
    ax.text(0, 1.01, "primary", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", color=INK2,
            fontsize=plt.rcParams["legend.fontsize"])
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, color=INK)
    ax.set_ylim(-0.7, n - 0.3)
    ax.yaxis.grid(False)
    lim = span * 1.25 if span > 0 else 0.01
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Delta R2 vs primary (LOSO, cluster-bootstrap 95% CI)")

    import textwrap

    d0 = items[0][1]["delta_r2_vs_primary"]
    note = ("plus_demographics is ablation-only: demographic covariates are "
            "never inputs to any deployed AQNet model.")
    if "n_boot" in d0:
        note += (f" Bootstrap: {d0['n_boot']:,} resamples over "
                 f"{d0.get('n_clusters', '?')} sensor clusters.")
    fig.text(0.02, 0.0, "\n".join(textwrap.wrap(note, width=68)),
             ha="left", va="top", color=INK2,
             fontsize=plt.rcParams["legend.fontsize"])

    fig.tight_layout()
    paths = save_fig(fig, "F14_ablation", mode, preview=preview)
    plt.close(fig)
    return paths


# ── F15: event strata ───────────────────────────────────────────────────────

def F15_strata(mode="paper", preview=False):
    """Grouped bars of R2 and RMSE across smoke/dust/clean strata."""
    m_loso = _json("metrics_loso.json")
    if m_loso is None:
        return _skip("F15_strata", "metrics_loso.json")
    tier_of = [("tier1", "Tier-1 LOFO", COLORS["tier1"], ""),
               ("tier2", "Tier-2 U-Net", COLORS["tier2"], "\\\\"),
               ("tier3", "Tier-3", COLORS["tier3"], "//")]
    series = []
    for key, val in m_loso.items():
        if not (key.startswith("strata_") and isinstance(val, dict)):
            continue
        label, color, hatch = key, COLORS["neutral"], ""
        for sub, lab, col, hat in tier_of:
            if sub in key:
                label, color, hatch = lab, col, hat
                break
        strata = {k: v for k, v in val.items()
                  if isinstance(v, dict) and "r2" in v}
        if strata:
            series.append((label, color, hatch, strata))
    if not series:
        return _skip("F15_strata", "strata_* blocks in metrics_loso.json")
    preview = _guard_preview(preview)
    set_style(mode)

    strata_names = [s for s in ("smoke", "dust", "clean")
                    if s in series[0][3]]
    strata_names += [s for s in series[0][3] if s not in strata_names]
    x = np.arange(len(strata_names))
    w = 0.7 / len(series)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W["double"], 2.8))
    for ax, metric, ylab, tag in ((ax1, "r2", "R2", "(a) R2"),
                                  (ax2, "rmse", "RMSE (ug/m3)", "(b) RMSE")):
        all_vals = [strata[s].get(metric, np.nan)
                    for _, _, _, strata in series for s in strata_names]
        vmax = np.nanmax([0.0] + all_vals)
        pad = 0.03 * (vmax if vmax > 0 else 1.0)
        for j, (label, color, hatch, strata) in enumerate(series):
            xs = x + (j - (len(series) - 1) / 2.0) * w
            vals = [strata[s].get(metric, np.nan) for s in strata_names]
            ax.bar(xs, vals, width=w * 0.9, color=color, hatch=hatch,
                   edgecolor=INK, linewidth=0.4, label=label)
            for xi, s in zip(xs, strata_names):
                v = strata[s].get(metric, np.nan)
                nn = strata[s].get("n")
                if np.isfinite(v) and nn is not None:
                    ax.text(xi, max(v, 0.0) + pad, f"n={nn:,}", ha="center",
                            va="bottom", color=INK2, rotation=90,
                            fontsize=plt.rcParams["legend.fontsize"] * 0.9)
        if any(np.nan_to_num(v) < 0 for v in all_vals):
            ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in strata_names])
        ax.set_xlabel("Stratum")
        ax.set_ylabel(ylab)
        ax.xaxis.grid(False)
        if len(series) >= 2:
            ax.legend(loc="upper right", frameon=False)
        _tag(ax, tag)
        # headroom for the rotated n= labels
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, max(y1, vmax + 12 * pad))

    fig.tight_layout()
    paths = save_fig(fig, "F15_strata", mode, preview=preview)
    plt.close(fig)
    return paths


# ── F16: spatial vs temporal R2 decomposition ───────────────────────────────

def F16_decomposition(mode="paper", preview=False):
    """Scatter of spatial R2 vs temporal R2 per tier (direct-labeled)."""
    m_loso = _json("metrics_loso.json")
    d = (m_loso or {}).get("spatial_temporal_r2")
    if not isinstance(d, dict) or not d:
        return _skip("F16_decomposition",
                     "spatial_temporal_r2 in metrics_loso.json")
    preview = _guard_preview(preview)
    set_style(mode)

    def style_of(key):
        if "tier1" in key:
            return "Tier-1 LOFO", COLORS["tier1"], "o"
        if "tier2" in key or "unet" in key:
            return "Tier-2 U-Net", COLORS["tier2"], "s"
        if "tier3" in key:
            return "Tier-3 cross-fit", COLORS["tier3"], "^"
        return key, COLORS["neutral"], "D"

    fig, ax = plt.subplots(figsize=(FIG_W["single"], 3.4))
    ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color=INK2, zorder=1)
    # stagger direct labels so near-coincident points stay readable
    offsets = [(8, 8), (8, -16), (-8, 8), (-8, -16)]
    i = 0
    for key, v in d.items():
        if not isinstance(v, dict):
            continue
        sx, ty = v.get("spatial_r2"), v.get("temporal_r2")
        if sx is None or ty is None:
            continue
        label, color, marker = style_of(key)
        ns = v.get("n_sensors")
        ax.scatter([sx], [ty], s=55, color=color, marker=marker,
                   edgecolor="white", linewidth=0.5, zorder=4,
                   label=f"{label}" + (f" ({ns} sensors)" if ns else ""))
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(label, (sx, ty), xytext=(dx, dy),
                    textcoords="offset points", color=INK,
                    ha="left" if dx > 0 else "right",
                    fontsize=plt.rcParams["legend.fontsize"])
        i += 1
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Spatial R2 (across-sensor means)")
    ax.set_ylabel("Temporal R2 (within-sensor deviations)")
    ax.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    paths = save_fig(fig, "F16_decomposition", mode, preview=preview)
    plt.close(fig)
    return paths


# ── F20: per-day Moran's I distribution ─────────────────────────────────────

def _haversine_km(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    dla = la[:, None] - la[None, :]
    dlo = lo[:, None] - lo[None, :]
    a = (np.sin(dla / 2.0) ** 2
         + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlo / 2.0) ** 2)
    return 2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _morans_i_day(r, lat, lon, k=8):
    """Row-standardized k-NN Moran's I -- mirrors validation.morans_i."""
    n = len(r)
    if n < k + 2:
        return float("nan")
    z = r - r.mean()
    denom = float(np.sum(z ** 2))
    if denom <= 0:
        return float("nan")
    dist = _haversine_km(lat, lon)
    np.fill_diagonal(dist, np.inf)
    idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
    lag = z[idx].mean(axis=1)
    return float(np.sum(z * lag) / denom)


def _morans_daily(resid, lat, lon, days, k=8, min_sensors=15):
    r = np.asarray(resid, dtype=float)
    la = np.asarray(lat, dtype=float)
    lo = np.asarray(lon, dtype=float)
    ok = np.isfinite(r) & np.isfinite(la) & np.isfinite(lo)
    vals = []
    for _, pos in pd.Series(np.arange(len(r))).groupby(pd.Series(days)).indices.items():
        pos = np.asarray(pos)
        pos = pos[ok[pos]]
        if len(pos) < min_sensors:
            continue
        v = _morans_i_day(r[pos], la[pos], lo[pos], k=k)
        if np.isfinite(v):
            vals.append(v)
    return np.asarray(vals, dtype=float)


def F20_morans(mode="paper", preview=False):
    """Violin of per-day residual Moran's I with median/IQR annotated."""
    tf = _parquet("training_frame.parquet")
    z1 = _npz("oof_tier1.npz")
    if tf is None or z1 is None:
        return _skip("F20_morans", "training_frame.parquet / oof_tier1.npz")
    zm = _npz("oof_meta.npz")
    preview = _guard_preview(preview)
    set_style(mode)

    lat = tf["lat"].to_numpy(dtype=float)
    lon = tf["lon"].to_numpy(dtype=float)
    days = pd.to_datetime(tf["date"]).dt.normalize().to_numpy()
    y = np.asarray(z1["y"], dtype=float)

    groups = []
    v1 = _morans_daily(y - np.asarray(z1["oof_lofo"], dtype=float),
                       lat, lon, days)
    if v1.size:
        groups.append(("Tier-1 LOFO", COLORS["tier1"], v1))
    if zm is not None and "tier3_crossfit" in zm.files:
        p3 = np.asarray(zm["tier3_crossfit"], dtype=float)
        r3 = y - p3
        if "is_calibration" in zm.files:
            r3 = np.where(np.asarray(zm["is_calibration"]).astype(bool),
                          np.nan, r3)  # meta-train rows only, as in metrics
        v3 = _morans_daily(r3, lat, lon, days)
        if v3.size:
            groups.append(("Tier-3 cross-fit", COLORS["tier3"], v3))
    if not groups:
        return _skip("F20_morans", "no day passed the min-sensor threshold")

    fig, ax = plt.subplots(figsize=(FIG_W["single"] + 0.5, 3.2))
    pos = np.arange(1, len(groups) + 1, dtype=float)
    vp = ax.violinplot([g[2] for g in groups], positions=pos,
                       widths=0.7, showextrema=False)
    for body, (_, color, _v) in zip(vp["bodies"], groups):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.55)
    for p, (label, color, v) in zip(pos, groups):
        q25, med, q75 = np.percentile(v, [25, 50, 75])
        ax.vlines(p, q25, q75, color=INK, lw=3.5, zorder=4)
        ax.plot([p], [med], marker="o", ms=5.5, mfc="white", mec=INK,
                mew=1.2, ls="none", zorder=5)
        ax.text(p + 0.06, float(v.max()),
                f"median {med:.3f}\nIQR {q75 - q25:.3f}\n{v.size} days",
                ha="left", va="top", color=INK2,
                fontsize=plt.rcParams["legend.fontsize"])
    ax.axhline(0, ls="--", lw=0.9, color=INK2, zorder=1)
    ax.text(0.99, 0.02, "I = 0: no spatial autocorrelation",
            transform=ax.transAxes, ha="right", va="bottom", color=INK2,
            fontsize=plt.rcParams["legend.fontsize"])
    ax.set_xticks(pos)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("Per-day Moran's I of residuals (k=8)")
    ax.xaxis.grid(False)

    fig.tight_layout()
    paths = save_fig(fig, "F20_morans", mode, preview=preview)
    plt.close(fig)
    return paths


# ── F22: AQI categories & exceedances ───────────────────────────────────────

def _aqi_category(pm25):
    """EPA 2024 category index; mirrors validation.aqi_category (truncation)."""
    v = np.maximum(np.asarray(pm25, dtype=np.float64), 0.0)
    v = np.floor(v * 10.0 + 1e-9) / 10.0
    return np.searchsorted(AQI_UPPER_BOUNDS, v, side="left")


def F22_aqi(mode="paper", preview=False):
    """(a) AQI category confusion heatmap, (b) exceedance precision/recall."""
    z1 = _npz("oof_tier1.npz")
    if z1 is None:
        return _skip("F22_aqi", "oof_tier1.npz")
    zm = _npz("oof_meta.npz")
    m_loso = _json("metrics_loso.json") or {}
    preview = _guard_preview(preview)
    set_style(mode)

    y = np.asarray(z1["y"], dtype=float)
    p1 = np.asarray(z1["oof_lofo"], dtype=float)
    ok = np.isfinite(y) & np.isfinite(p1)
    ct = _aqi_category(y[ok])
    cp = _aqi_category(p1[ok])
    k = len(AQI_CATEGORIES)
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (ct, cp), 1)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(FIG_W["double"], 3.3),
        gridspec_kw={"width_ratios": [1.3, 1]})

    cmap = mpl.colormaps[SEQUENTIAL_CMAP].copy()
    cmap.set_bad("#f2f2f2")
    norm = LogNorm(vmin=1, vmax=max(1, cm.max()))
    im = ax1.imshow(np.ma.masked_equal(cm, 0), cmap=cmap, norm=norm,
                    origin="upper", aspect="equal")
    row_sums = cm.sum(axis=1)
    small = plt.rcParams["legend.fontsize"] * 0.85
    for i in range(k):
        for j in range(k):
            c = cm[i, j]
            if c == 0:
                continue
            rgba = cmap(norm(c))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ink = "#f2f2f2" if lum < 0.45 else "#333333"
            pct = c / row_sums[i] if row_sums[i] else 0.0
            ax1.text(j, i, f"{c:,}\n{pct:.0%}", ha="center", va="center",
                     color=ink, fontsize=small)
    ax1.set_xticks(range(k))
    ax1.set_xticklabels(AQI_CATEGORIES, rotation=30, ha="right")
    ax1.set_yticks(range(k))
    ax1.set_yticklabels(AQI_CATEGORIES)
    ax1.set_xlabel("Predicted AQI category")
    ax1.set_ylabel("Observed AQI category")
    ax1.grid(False)
    fig.colorbar(im, ax=ax1, shrink=0.8, label="site-days")
    _tag(ax1, "(a) AQI confusion - Tier-1 LOFO OOF")

    # (b) exceedance (> 35.4 ug/m3) precision / recall per tier.
    def exceed_pr(yy, pp):
        t = _aqi_category(yy) >= 2
        h = _aqi_category(pp) >= 2
        tp = int(np.sum(t & h))
        prec = tp / int(h.sum()) if h.sum() > 0 else float("nan")
        rec = tp / int(t.sum()) if t.sum() > 0 else float("nan")
        return prec, rec, int(t.sum())

    tiers = [("Tier-1 LOFO", COLORS["tier1"], "", *exceed_pr(y[ok], p1[ok]))]
    if zm is not None and "tier3_crossfit" in zm.files:
        y3 = np.asarray(zm["y"], dtype=float)
        p3 = np.asarray(zm["tier3_crossfit"], dtype=float)
        if "is_calibration" in zm.files:
            keep = ~np.asarray(zm["is_calibration"]).astype(bool)
            y3, p3 = y3[keep], p3[keep]
        ok3 = np.isfinite(y3) & np.isfinite(p3)
        tiers.append(("Tier-3 cross-fit", COLORS["tier3"], "//",
                      *exceed_pr(y3[ok3], p3[ok3])))

    x = np.arange(2)
    w = 0.7 / len(tiers)
    small = plt.rcParams["legend.fontsize"]
    for j, (label, color, hatch, prec, rec, n_true) in enumerate(tiers):
        xs = x + (j - (len(tiers) - 1) / 2.0) * w
        for xi, v in zip(xs, (prec, rec)):
            if np.isfinite(v):
                ax2.bar([xi], [v], width=w * 0.9, color=color, hatch=hatch,
                        edgecolor=INK, linewidth=0.4)
                ax2.text(xi, v + 0.03, f"{v:.2f}", ha="center", va="bottom",
                         color=INK2, fontsize=small)
            else:
                ax2.text(xi, 0.02, "undef.", rotation=90, ha="center",
                         va="bottom", color=INK2, fontsize=small)
    handles = [Patch(facecolor=c, hatch=h, edgecolor=INK, linewidth=0.4,
                     label=lab) for lab, c, h, _p, _r, _n in tiers]
    ax2.legend(handles=handles, loc="upper right", frameon=False)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Precision", "Recall"])
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Score")
    n_true_obs = tiers[0][5]
    ax2.set_xlabel("Exceedance: PM2.5 > 35.4 ug/m3\n"
                   f"(n observed = {n_true_obs})")
    ax2.xaxis.grid(False)
    _tag(ax2, "(b) Exceedance detection")

    fig.tight_layout()
    paths = save_fig(fig, "F22_aqi", mode, preview=preview)
    plt.close(fig)
    return paths


# ── driver ──────────────────────────────────────────────────────────────────

ALL_FIGURES = [F08_main_results, F09_pred_vs_obs, F10_spatial_block,
               F11_temporal, F12_external_aqs, F13_conformal, F14_ablation,
               F15_strata, F16_decomposition, F20_morans, F22_aqi]


def render_all(mode="paper", preview=True):
    out = {}
    for fn in ALL_FIGURES:
        try:
            out[fn.__name__] = fn(mode=mode, preview=preview)
        except Exception as e:  # keep going; report at the end
            print(f"[fig_results] {fn.__name__}: FAILED - "
                  f"{type(e).__name__}: {e}")
            out[fn.__name__] = None
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", default="paper", choices=["paper", "poster"])
    ap.add_argument("--final", action="store_true",
                    help="write non-preview outputs (still forced to preview "
                         "when the artifacts are quick-mode)")
    args = ap.parse_args()
    results = render_all(mode=args.mode, preview=not args.final)
    for name, paths in results.items():
        print(f"{name}: {paths if paths else 'skipped/failed'}")
