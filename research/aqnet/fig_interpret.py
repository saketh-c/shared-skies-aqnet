"""Interpretability figures for AQNet (shared figure standard).

Figures
-------
F17_importance       Top-15 permutation feature importance (delta-R2) from
                     artifacts/permutation_report.json, plus a mean-|SHAP|
                     panel when `shap` + `lightgbm` are installed and the
                     training frame artifact exists (a quick LGBM is refit
                     on the frame purely to attribute it; the panel title
                     says so).
F18_attention        Per-source fusion attention-weight maps from the
                     Tier-2 FusionUNet checkpoint, for the highest-HMS-smoke
                     day and a clean day inside the cached extended stack.
F19_surfaces         Predicted PM2.5 surfaces for the same two days, shared
                     viridis scale, sensor pixels overlaid.
F21_learning_curves  Training loss and validation RMSE vs epoch from the
                     per-epoch history in artifacts/unet_train.json (skips
                     gracefully when the artifact has no history).

Every figure function takes (mode="paper"|"poster", preview=bool), returns
the list of written paths, or None on a graceful skip (with a printed
"[skip]" reason). Missing artifacts never raise. All styling comes from
fig_style; renders from quick-mode artifacts must be run with preview=True
so save_fig watermarks them.
"""
import glob
import json
import os
import sys

# ── Sibling imports (aqnet + deep-learning track), Colab-safe ───────────────

_AQNET_DIR = os.path.dirname(os.path.abspath(__file__))
_DL_DIR = os.path.join(os.path.dirname(_AQNET_DIR), "deeplearning")
for _p in (_DL_DIR, _AQNET_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

import config
from fig_style import (COLORS, FIG_W, SEQUENTIAL_CMAP, annotate_metrics,
                       save_fig, set_style)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _skip(fig_name, reason):
    """Print a graceful-skip message and return None (the skip sentinel)."""
    print(f"[skip] {fig_name}: {reason}")
    return None


def _scale(mode):
    """Figure-size multiplier: poster mode roughly doubles the canvas."""
    return 1.0 if mode == "paper" else 1.9


def _fig_width(mode, column="double"):
    return FIG_W[column] if mode == "paper" else FIG_W["poster"]


_FEATURE_LABELS = {
    "latitude": "Latitude",
    "longitude": "Longitude",
    "dist_to_coast": "Distance to coast",
    "dist_to_nearest_sensor": "Distance to nearest sensor",
    "rmp_proximity": "RMP facility proximity",
    "diesel_pm_proximity": "Diesel PM proximity",
    "traffic_proximity": "Traffic proximity",
    "superfund_proximity": "Superfund proximity",
    "temp_x_humidity": "Temperature x humidity",
    "wind_x_temp": "Wind x temperature",
    "humidity": "Relative humidity",
    "pressure": "Surface pressure",
    "temperature": "Temperature",
    "precipitation": "Precipitation",
    "wind_speed": "Wind speed",
    "aod": "Aerosol optical depth",
    "cams_pm25": "CAMS PM2.5",
    "geoscf_pm25": "GEOS-CF PM2.5",
    "dust": "CAMS dust",
    "hms_smoke": "HMS smoke density",
    "elevation": "Elevation",
    "day_of_year": "Day of year",
    "doy_sin": "Day-of-year (sin)",
    "doy_cos": "Day-of-year (cos)",
    "dow": "Day of week",
    "dow_sin": "Day-of-week (sin)",
    "dow_cos": "Day-of-week (cos)",
    "month": "Month",
    "month_sin": "Month (sin)",
    "month_cos": "Month (cos)",
}


def clean_feature_name(name):
    """Human-readable label for a raw feature column name."""
    if name in _FEATURE_LABELS:
        return _FEATURE_LABELS[name]
    for prefix, label in (("nbr_pm25_", "Neighbor PM2.5"),
                          ("nbr_std_", "Neighbor PM2.5 s.d."),
                          ("nbr_count_", "Neighbor count"),
                          ("merra2_", "MERRA-2 ")):
        if name.startswith(prefix):
            rest = name[len(prefix):]
            if rest.endswith("km"):
                return f"{label} ({rest[:-2]} km)"
            return f"{label}{rest.replace('_', ' ')}".strip()
    return name.replace("_", " ").capitalize()


# ── F17: feature importance ─────────────────────────────────────────────────

def _shap_importance(features, top_n=15, max_rows=4000, seed=42):
    """Mean-|SHAP| per feature from a quick LGBM refit on the training frame.

    Returns [(feature, mean_abs_shap), ...] (descending) or None when shap /
    lightgbm are unavailable, the frame artifact is missing, or anything in
    the recompute fails. The refit exists only to attribute the tabular
    feature set with SHAP; it is NOT the production Tier-1 blend and the
    panel is labeled accordingly.
    """
    frame_path = config.artifact("training_frame.parquet")
    if not os.path.exists(frame_path):
        print("  note: no training_frame.parquet -> permutation panel only")
        return None
    try:
        import lightgbm as lgb
        import shap
    except ImportError as exc:
        print(f"  note: SHAP recompute unavailable ({exc}) -> "
              "permutation panel only")
        return None
    try:
        df = pd.read_parquet(frame_path, columns=list(features) + ["target"])
        X = df[list(features)]
        y = df["target"].to_numpy()
        model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                  num_leaves=63, subsample=0.9,
                                  colsample_bytree=0.9, random_state=seed,
                                  n_jobs=-1, verbose=-1)
        model.fit(X, y)
        rng = np.random.default_rng(seed)
        take = min(max_rows, len(X))
        idx = rng.choice(len(X), size=take, replace=False)
        sample = X.iloc[np.sort(idx)]
        values = shap.TreeExplainer(model).shap_values(sample)
        mean_abs = np.abs(np.asarray(values)).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:top_n]
        return [(features[i], float(mean_abs[i])) for i in order]
    except Exception as exc:  # any shap/lgbm hiccup degrades gracefully
        print(f"  note: SHAP recompute failed ({type(exc).__name__}: {exc}) "
              "-> permutation panel only")
        return None


def _barh_importance(ax, names, values, errors=None):
    """Neutral-hue horizontal importance bars, largest at the top."""
    pos = np.arange(len(names))[::-1]
    ax.barh(pos, values, height=0.72, color=COLORS.get("neutral", "#555555"),
            xerr=errors, error_kw={"ecolor": "#333333", "capsize": 2,
                                   "linewidth": 0.8})
    ax.set_yticks(pos)
    ax.set_yticklabels(names)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.yaxis.grid(False)
    ax.margins(x=0.02)


def F17_importance(mode="paper", preview=False):
    """Top-15 feature importance: permutation delta-R2 (+ SHAP if possible)."""
    path = config.artifact("permutation_report.json")
    if not os.path.exists(path):
        return _skip("F17_importance", f"missing artifact {path}")
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    imps = sorted(report.get("importances", []),
                  key=lambda r: -r["delta_r2_mean"])
    if not imps:
        return _skip("F17_importance", "permutation_report.json has no "
                                       "importances")
    top = imps[:15]
    all_features = [r["feature"] for r in imps]
    shap_top = _shap_importance(all_features, top_n=15)

    set_style(mode)
    s = _scale(mode)
    n_panels = 2 if shap_top else 1
    width = (FIG_W["double"] if n_panels == 2 else FIG_W["single"]) * s
    fig, axes = plt.subplots(1, n_panels, figsize=(width, 3.6 * s),
                             layout="constrained")
    axes = np.atleast_1d(axes)

    names = [clean_feature_name(r["feature"]) for r in top]
    values = [r["delta_r2_mean"] for r in top]
    errors = [r.get("delta_r2_std", 0.0) for r in top]
    _barh_importance(axes[0], names, values, errors)
    axes[0].set_xlabel("Decrease in R2 (dimensionless)")
    n_rep = report.get("n_repeats")
    rep_txt = f" (mean of {n_rep} shuffles)" if n_rep else ""
    tag = "(a) " if n_panels == 2 else ""
    axes[0].set_title(f"{tag}Permutation importance{rep_txt}")
    if n_panels == 2:
        s_names = [clean_feature_name(f) for f, _ in shap_top]
        s_values = [v for _, v in shap_top]
        _barh_importance(axes[1], s_names, s_values)
        axes[1].set_title("(b) Mean |SHAP| (LGBM refit)")
        axes[1].set_xlabel("Mean |SHAP| (ug/m3)")

    return save_fig(fig, "F17_importance", mode, preview=preview)


# ── Shared U-Net inference for F18 / F19 ────────────────────────────────────

_SOURCE_TITLES = {
    "aerosol": "Aerosol",
    "smoke": "Smoke",
    "meteorology": "Meteorology",
    "static": "Static",
    "temporal": "Temporal",
    "ctm": "GEOS-CF",
    "merra2": "MERRA-2",
    "flags": "Flags",
}

_UNET_CACHE = {"loaded": False, "value": None}


def _find_stack_cache():
    """Locate the extended-stack cache: unet_train.json's path, else newest."""
    info_path = config.artifact("unet_train.json")
    if os.path.exists(info_path):
        try:
            with open(info_path, encoding="utf-8") as f:
                cached = json.load(f).get("stack_cache")
            if cached and os.path.exists(cached):
                return cached
        except (OSError, ValueError):
            pass
    hits = glob.glob(os.path.join(config.CACHE_DIR, "extended_stack_*.npz"))
    if not hits:
        return None
    return max(hits, key=os.path.getmtime)


def _select_days(stack):
    """(smoke_day, clean_day) indices: max HMS smoke vs min smoke (ties
    broken by lowest mean CAMS PM2.5)."""
    channels = stack["channels"]
    smoke_arr = stack["groups"]["smoke"]
    s_idx = (channels["smoke"].index("hms_smoke")
             if "hms_smoke" in channels["smoke"] else 0)
    with np.errstate(all="ignore"):
        smoke = np.nanmean(smoke_arr[:, s_idx], axis=(1, 2))
    cams = np.zeros(len(smoke))
    if "cams_pm25" in channels.get("aerosol", []):
        c_idx = channels["aerosol"].index("cams_pm25")
        with np.errstate(all="ignore"):
            cams = np.nanmean(stack["groups"]["aerosol"][:, c_idx], axis=(1, 2))
    smoke = np.where(np.isfinite(smoke), smoke, 0.0)
    cams = np.where(np.isfinite(cams), cams, 0.0)
    order_clean = np.lexsort((cams, smoke))       # ascending smoke, then CAMS
    smoke_day = int(np.lexsort((-cams, -smoke))[0])
    clean_day = int(order_clean[0])
    if clean_day == smoke_day and len(order_clean) > 1:
        clean_day = int(order_clean[1])
    return smoke_day, clean_day


def _load_unet_inference():
    """Checkpoint + cached stack -> surfaces and attention for the two focus
    days. Memoized; returns None (after printing why) on any graceful skip."""
    if _UNET_CACHE["loaded"]:
        return _UNET_CACHE["value"]
    _UNET_CACHE["loaded"] = True
    name = "F18/F19 (U-Net inference)"

    try:
        import torch
        import models as dl_models          # research/deeplearning
        import dataset as dl_dataset        # research/deeplearning
    except ImportError as exc:
        return _skip(name, f"deep stack unavailable ({exc})")

    ckpt_path = os.path.join(config.ARTIFACTS_DIR, "unet",
                             "fusion_unet_best.pt")
    if not os.path.exists(ckpt_path):
        return _skip(name, f"missing checkpoint {ckpt_path}")
    stack_path = _find_stack_cache()
    if stack_path is None:
        return _skip(name, "no cache/extended_stack_*.npz found (run the "
                           "deep stage to build it)")

    stack = dl_dataset.load_cache(stack_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Grid / channel-group compatibility (mismatch = stale cache, not a bug).
    if (len(stack["lat"]) != len(ckpt["lat"])
            or len(stack["lon"]) != len(ckpt["lon"])
            or abs(float(stack["grid_deg"]) - float(ckpt["grid_deg"])) > 1e-9):
        return _skip(name, f"stack grid ({len(stack['lat'])}x"
                           f"{len(stack['lon'])} at {stack['grid_deg']} deg) "
                           "does not match the checkpoint grid")
    for gname, n_ch in ckpt["group_channels"].items():
        arr = stack["groups"].get(gname)
        if arr is None or arr.shape[1] != n_ch:
            return _skip(name, f"stack channel group {gname!r} missing or "
                               "mismatched vs the checkpoint")

    smoke_day, clean_day = _select_days(stack)
    sel = [smoke_day, clean_day]
    dates = pd.DatetimeIndex(stack["dates"])
    day_labels = [f"Smoke day {dates[smoke_day].date()}",
                  f"Clean day {dates[clean_day].date()}"]
    print(f"  U-Net focus days: {day_labels[0]} / {day_labels[1]} "
          f"(stack {os.path.basename(stack_path)})")

    # Fancy indexing copies, so checkpoint-time fills/normalization never
    # touch the loaded stack arrays.
    sub = {g: stack["groups"][g][sel] for g in ckpt["group_channels"]}
    dl_dataset.fill_missing(sub, fill_values=ckpt["fill_values"])
    dl_dataset.apply_norm_stats(sub, ckpt["norm_stats"])

    model = dl_models.FusionUNet(ckpt["group_channels"],
                                 embed_dim=ckpt["embed_dim"],
                                 base_width=ckpt["base_width"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        surf, attn = model({g: torch.from_numpy(a) for g, a in sub.items()})

    lat, lon = stack["lat"], stack["lon"]
    obs = stack["obs"]
    pix = np.unique(obs["row"].astype(np.int64) * len(lon) + obs["col"])
    value = {
        "surfaces": surf.squeeze(1).numpy(),            # (2, H, W)
        "attention": attn.numpy(),                      # (2, S, H, W)
        "sources": list(ckpt["group_channels"]),
        "day_labels": day_labels,
        "lat": lat,
        "lon": lon,
        "grid_deg": float(stack["grid_deg"]),
        "sensor_lat": lat[pix // len(lon)],
        "sensor_lon": lon[pix % len(lon)],
    }
    _UNET_CACHE["value"] = value
    return value


def _map_extent(inf):
    g = inf["grid_deg"]
    lat, lon = inf["lat"], inf["lon"]
    extent = [lon[0] - g / 2, lon[-1] + g / 2,
              lat[0] - g / 2, lat[-1] + g / 2]
    aspect = 1.0 / np.cos(np.deg2rad(float(np.mean(lat))))
    return extent, aspect


# ── F18: attention maps ─────────────────────────────────────────────────────

def F18_attention(mode="paper", preview=False):
    """Per-source attention-weight maps, smoke day vs clean day."""
    inf = _load_unet_inference()
    if inf is None:
        return None
    set_style(mode)
    sources = inf["sources"]
    n_src = len(sources)

    # Poster fonts scale ~2.1x while FIG_W["poster"] is only ~1.67x the
    # double-column width, so the 7-column small-multiple grid gets extra
    # canvas in poster mode to keep the same relative text density.
    width = FIG_W["double"] if mode == "paper" else FIG_W["poster"] * 1.25
    fig, axes = plt.subplots(2, n_src, figsize=(width, 0.34 * width),
                             sharex=True, sharey=True, layout="constrained")
    extent, aspect = _map_extent(inf)
    im = None
    for i in range(2):                                   # rows: days
        for j in range(n_src):                           # cols: sources
            ax = axes[i, j]
            im = ax.imshow(inf["attention"][i, j], origin="lower",
                           extent=extent, vmin=0.0, vmax=1.0,
                           cmap=SEQUENTIAL_CMAP, interpolation="nearest")
            ax.set_aspect(aspect)
            ax.grid(False)
            ax.set_xticks([-105, -95])
            ax.set_yticks([27, 31, 35])
            ax.tick_params(length=2)
            if i == 0:
                ax.set_title(_SOURCE_TITLES.get(sources[j], sources[j]),
                             fontsize=plt.rcParams["xtick.labelsize"])
        label, date = inf["day_labels"][i].rsplit(" ", 1)
        axes[i, 0].set_ylabel(f"({'ab'[i]}) {label}\n{date}",
                              fontsize=plt.rcParams["xtick.labelsize"])
    fig.supxlabel("Longitude (deg E)",
                  fontsize=plt.rcParams["axes.labelsize"])
    fig.supylabel("Latitude (deg N)",
                  fontsize=plt.rcParams["axes.labelsize"])
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.015)
    cbar.set_label("Attention weight (0-1)")
    return save_fig(fig, "F18_attention", mode, preview=preview)


# ── F19: predicted surfaces ─────────────────────────────────────────────────

def F19_surfaces(mode="paper", preview=False):
    """Predicted PM2.5 surfaces for the two focus days, sensors overlaid."""
    inf = _load_unet_inference()
    if inf is None:
        return None
    set_style(mode)
    s = _scale(mode)
    surfaces = inf["surfaces"]
    vmax = max(float(np.nanpercentile(surfaces, 99.5)), 1.0)

    width = _fig_width(mode, "double")
    fig, axes = plt.subplots(1, 2, figsize=(width, 0.46 * width),
                             sharey=True, layout="constrained")
    extent, aspect = _map_extent(inf)
    im = None
    for i, ax in enumerate(axes):
        im = ax.imshow(surfaces[i], origin="lower", extent=extent,
                       vmin=0.0, vmax=vmax, cmap=SEQUENTIAL_CMAP,
                       interpolation="nearest")
        ax.scatter(inf["sensor_lon"], inf["sensor_lat"], s=4 * s ** 2,
                   c=COLORS["observed"], edgecolors="white",
                   linewidths=0.25 * s, label="Sensor pixel", zorder=3)
        ax.set_aspect(aspect)
        ax.grid(False)
        ax.set_xticks([-105, -100, -95])
        ax.set_title(f"({'ab'[i]}) {inf['day_labels'][i]}")
        ax.set_xlabel("Longitude (deg E)")
        annotate_metrics(ax, {"domain mean":
                              f"{float(np.nanmean(surfaces[i])):.1f} ug/m3"})
    axes[0].set_ylabel("Latitude (deg N)")
    axes[0].legend(loc="lower left", frameon=True, framealpha=0.85,
                   handletextpad=0.4)
    cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.015)
    cbar.set_label("PM2.5 (ug/m3)")
    return save_fig(fig, "F19_surfaces", mode, preview=preview)


# ── F21: learning curves ────────────────────────────────────────────────────

def _render_learning_curves(history, best_epoch, mode):
    """Two stacked single-axis panels: train loss, then validation RMSE."""
    s = _scale(mode)
    epochs = [h["epoch"] for h in history]
    loss = [h["train_loss"] for h in history]
    val = [(h["epoch"], h["val"]["rmse"]) for h in history
           if isinstance(h.get("val"), dict) and "rmse" in h["val"]]
    color = COLORS["tier2"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W["single"] * s,
                                                  4.2 * s),
                                   sharex=True, layout="constrained")
    ax1.plot(epochs, loss, color=color, marker="o", markersize=2.5 * s)
    ax1.set_ylabel("Training loss\n(masked Huber, ug/m3)")
    ax1.set_title("(a) Tier-2 U-Net training loss")

    if val:
        ve, vr = zip(*val)
        ax2.plot(ve, vr, color=color, marker="o", markersize=2.5 * s)
    ax2.set_ylabel("Validation RMSE (ug/m3)")
    ax2.set_xlabel("Epoch")
    ax2.set_title("(b) Held-out-site RMSE")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    if best_epoch is not None:
        for ax in (ax1, ax2):
            ax.axvline(best_epoch, color="#999999", linestyle="--",
                       linewidth=0.8 * s, zorder=1)
        best_rmse = dict(val).get(best_epoch)
        if best_rmse is not None:
            ax2.plot([best_epoch], [best_rmse], marker="o",
                     markersize=5 * s, markerfacecolor="none",
                     markeredgecolor=color, linestyle="none")
            ax2.annotate(f"best (epoch {best_epoch})",
                         xy=(best_epoch, best_rmse),
                         xytext=(4 * s, 4 * s), textcoords="offset points",
                         fontsize=plt.rcParams["legend.fontsize"],
                         color="#333333")
    return fig


def F21_learning_curves(mode="paper", preview=False):
    """FusionUNet loss and validation RMSE vs epoch, best epoch marked."""
    path = config.artifact("unet_train.json")
    if not os.path.exists(path):
        return _skip("F21_learning_curves", f"missing artifact {path}")
    with open(path, encoding="utf-8") as f:
        info = json.load(f)
    history = [h for h in (info.get("history") or [])
               if isinstance(h, dict) and "epoch" in h]
    if len(history) < 2:
        return _skip("F21_learning_curves",
                     "unet_train.json has no per-epoch history (quick-mode "
                     "artifact); rerun the deep stage with history persisted")
    best = info.get("best") or {}
    best_epoch = best.get("epoch")
    if best_epoch is None:
        rmses = [(h["val"]["rmse"], h["epoch"]) for h in history
                 if isinstance(h.get("val"), dict) and "rmse" in h["val"]]
        best_epoch = min(rmses)[1] if rmses else None

    set_style(mode)
    fig = _render_learning_curves(history, best_epoch, mode)
    return save_fig(fig, "F21_learning_curves", mode, preview=preview)


FIGURES = {
    "F17_importance": F17_importance,
    "F18_attention": F18_attention,
    "F19_surfaces": F19_surfaces,
    "F21_learning_curves": F21_learning_curves,
}


if __name__ == "__main__":
    for _fname, _fn in FIGURES.items():
        print(f"-- {_fname}")
        _out = _fn(mode="paper", preview=True)
        if _out:
            for _p in _out:
                print(f"   wrote {_p}")
