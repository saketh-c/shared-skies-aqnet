"""fig_architecture.py — data-independent AQNet schematics (F01-F04).

Renders the four architecture/protocol diagrams as pure-matplotlib schematics
(patches + FancyArrowPatch + annotate; no graphviz, no seaborn):

  F01_three_tier_architecture  data sources -> Tier1/Tier2 internals -> Tier3
                               stack -> outputs, with dashed red leakage
                               separators (EPA AQS fully outside training)
  F02_fusionunet_architecture  channel groups -> per-source encoders ->
                               per-pixel softmax attention (alpha equation) ->
                               squeeze-excite -> depth-4 U-Net with skip
                               arrows and channel widths -> softplus surface
  F03_data_provenance          every source flowing into features/channels/
                               validation, color-coded training vs
                               external-only leakage status
  F04_validation_protocol      LOSO folds, spatial blocks on Texas, temporal
                               split, external AQS, and the Tier-3 meta
                               cross-fit/calibration sensor split

Every number annotated here is read from (or verified against) the code:
config.py (dates, grid, feature contract), models_tabular.py (Huber delta 10,
2000-round cap, early-stop tail, simplex/LOFO blend, quantile heads),
research/deeplearning/models.py + grids.py (channel groups/counts, embed 32,
U-Net widths 32-512, SE 32->8->32, softplus), models_deep.py (Huber delta 15,
AdamW/warmup/cosine, 96 px crops, modality dropout 0.15, 20% site holdout),
fusion.py + pipeline_colab.py (per-day kriging <=150 pts, Ridge alpha=1
coef>=0, 4 cross-fit sensor folds, 75/25 meta/calibration split, CQR
alpha=0.1), validation.py (LOSO GroupKFold(10), KMeans(5) spatial blocks,
temporal cutoff 2025-01-01, external AQS).

F04's spatial panel draws real sensor sites (pipeline/sensor_tx_membership.csv)
over the Texas tract polygons (backend/static/texas_all_tracts.geojson,
rasterized light-gray context); everything else is data-independent.

Run (from anywhere):
    python research/aqnet/fig_architecture.py
"""
import os
import sys
import json

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Patch
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D

_AQNET_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_AQNET_DIR))
if _AQNET_DIR not in sys.path:
    sys.path.insert(0, _AQNET_DIR)

# ── Shared style (fig_style.py contract; local fallback keeps this module
#    runnable before/without that file, implementing the same contract) ──────

try:
    import fig_style as _FS
    COLORS = _FS.COLORS
    FIG_W = _FS.FIG_W
    set_style = _FS.set_style
    save_fig = _FS.save_fig
    _STYLE_SOURCE = "fig_style.py"
except Exception:  # pragma: no cover - exercised only pre-fig_style
    _STYLE_SOURCE = "internal fallback (fig_style.py not importable)"

    COLORS = {
        "tier1": "#0072B2", "tier2": "#E69F00", "tier3": "#009E73",
        "baseline_nearest": "#9a9a9a", "baseline_idw": "#7a7a7a",
        "baseline_kriging": "#5a5a5a",
        "prior_cams": "#56B4E9", "prior_geoscf": "#D55E00",
        "prior_merra2": "#CC79A7",
        "observed": "#333333", "train_shade": "#e8e8e8",
    }
    FIG_W = {"single": 3.5, "double": 7.2, "poster": 12.0}

    def set_style(mode="paper"):
        paper = mode == "paper"
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8.5 if paper else 18.0,
            "axes.titlesize": 10 if paper else 21,
            "axes.labelsize": 9 if paper else 19,
            "xtick.labelsize": 8 if paper else 17,
            "ytick.labelsize": 8 if paper else 17,
            "legend.fontsize": 8 if paper else 17,
            "axes.spines.top": False, "axes.spines.right": False,
            "grid.alpha": 0.25, "grid.linewidth": 0.5,
            "axes.axisbelow": True, "savefig.dpi": 300,
        })

    def save_fig(fig, name, mode="paper", preview=False):
        out_dir = os.path.join(_AQNET_DIR, "figures")
        if preview:
            out_dir = os.path.join(out_dir, "preview_quick")
            fig.text(0.5, 0.5, "QUICK-MODE PREVIEW — NOT RESULTS",
                     rotation=30, ha="center", va="center",
                     fontsize=26, color="#999999", alpha=0.35, zorder=1000)
        os.makedirs(out_dir, exist_ok=True)
        stem = name + ("_poster" if mode == "poster" else "")
        paths = []
        for ext in ("pdf", "png", "svg"):
            p = os.path.join(out_dir, f"{stem}.{ext}")
            fig.savefig(p, dpi=300, bbox_inches="tight")
            paths.append(p)
        return paths


def _c(key, default):
    """COLORS lookup that survives a missing key in an older fig_style."""
    try:
        return COLORS[key]
    except Exception:
        return default


INK = "#333333"       # primary neutral ink (text)
INK2 = "#555555"      # secondary neutral ink
EDGE = "#777777"      # neutral box edges
FILL = "#f5f5f5"      # neutral box fill
LEAK_RED = "#c62828"  # leakage-boundary red (separator lines/edges only)

TIER1 = _c("tier1", "#0072B2")
TIER2 = _c("tier2", "#E69F00")
TIER3 = _c("tier3", "#009E73")
P_CAMS = _c("prior_cams", "#56B4E9")
P_GEOSCF = _c("prior_geoscf", "#D55E00")
P_MERRA2 = _c("prior_merra2", "#CC79A7")
TRAIN_SHADE = _c("train_shade", "#e8e8e8")

# Repo data used only by F04's spatial-blocks panel.
MEMBERSHIP_CSV = os.path.join(_ROOT, "pipeline", "sensor_tx_membership.csv")
TRACTS_GEOJSON = os.path.join(_ROOT, "backend", "static",
                              "texas_all_tracts.geojson")


# ── Scaling helpers (poster geometry = 12/7.2 of paper; fonts ~1.75x) ───────

def _scales(mode):
    geo = FIG_W["poster"] / FIG_W["double"] if mode == "poster" else 1.0
    fs = 1.75 if mode == "poster" else 1.0
    return {
        "geo": geo,
        "title": 9.6 * fs, "head": 7.4 * fs, "body": 6.4 * fs,
        "tiny": 5.6 * fs,
        "lw_box": 0.9 * geo, "lw_chip": 0.7 * geo, "lw_arrow": 1.0 * geo,
        "lw_sep": 1.2 * geo, "ams": 7.0 * geo,
    }


def _canvas(mode, ratio):
    """Figure + full-bleed schematic axes; canvas coords are 0-100 x 0-100*r."""
    w = FIG_W["poster"] if mode == "poster" else FIG_W["double"]
    fig = plt.figure(figsize=(w, w * ratio))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * ratio)
    ax.axis("off")
    return fig, ax


def _box(ax, x, y, w, h, fc=FILL, ec=EDGE, lw=0.9, ls="-", z=2, round_=0.8):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={round_}",
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=z)
    ax.add_patch(p)
    return p


def _text(ax, x, y, s, size, color=INK, ha="center", va="center",
          weight="normal", ls=1.25, z=5, rot=0, style="normal"):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
                   fontweight=weight, linespacing=ls, zorder=z, rotation=rot,
                   fontstyle=style)


def _arrow(ax, p0, p1, color=INK2, lw=1.0, ls="-", rad=0.0, ams=7.0, z=4,
           alpha=1.0, style="-|>"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ams,
                        color=color, lw=lw, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=z,
                        alpha=alpha, shrinkA=1.5, shrinkB=1.5)
    ax.add_patch(a)
    return a


# ── F04 data helpers (real repo data for the spatial panel) ─────────────────

_TRACTS = {"polys": None, "tried": False}
_SENSORS = {"xy": None, "tried": False}


def _load_tract_polys():
    """Exterior rings of every Texas tract polygon as (N,2) lon/lat arrays."""
    if _TRACTS["tried"]:
        return _TRACTS["polys"]
    _TRACTS["tried"] = True
    if not os.path.exists(TRACTS_GEOJSON):
        print(f"[fig_architecture] tract geojson missing, map context "
              f"skipped: {TRACTS_GEOJSON}")
        return None
    with open(TRACTS_GEOJSON, encoding="utf-8") as f:
        gj = json.load(f)
    polys = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry") or {}
        gtype, coords = geom.get("type"), geom.get("coordinates")
        if gtype == "Polygon":
            rings = [coords[0]] if coords else []
        elif gtype == "MultiPolygon":
            rings = [p[0] for p in coords if p]
        else:
            rings = []
        for ring in rings:
            arr = np.asarray(ring, dtype=np.float64)
            if arr.ndim == 2 and len(arr) >= 3:
                polys.append(arr[:, :2])
    _TRACTS["polys"] = polys or None
    print(f"[fig_architecture] loaded {len(polys):,} tract rings for context")
    return _TRACTS["polys"]


def _load_sensor_xy():
    """(lon, lat) of in-Texas PurpleAir supervision sensors, or None."""
    if _SENSORS["tried"]:
        return _SENSORS["xy"]
    _SENSORS["tried"] = True
    if not os.path.exists(MEMBERSHIP_CSV):
        return None
    df = pd.read_csv(MEMBERSHIP_CSV)
    df = df[df["in_tx"].astype(str).str.lower() == "true"]
    df = df.drop_duplicates("sensor_id")
    xy = np.column_stack([df["longitude"].to_numpy(dtype=float),
                          df["latitude"].to_numpy(dtype=float)])
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    _SENSORS["xy"] = xy if len(xy) else None
    return _SENSORS["xy"]


def _kmeans(xy, k=5, seed=42, iters=100):
    """Plain-numpy Lloyd's KMeans (deterministic; mirrors the protocol's
    KMeans(5)-over-sensor-coords idea without an sklearn dependency)."""
    rng = np.random.default_rng(seed)
    cent = xy[rng.choice(len(xy), size=k, replace=False)].copy()
    lab = np.zeros(len(xy), dtype=int)
    for _ in range(iters):
        d = ((xy[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        new_lab = d.argmin(1)
        if (new_lab == lab).all() and _ > 0:
            break
        lab = new_lab
        for j in range(k):
            sel = lab == j
            if sel.any():
                cent[j] = xy[sel].mean(0)
    return lab, cent


def _convex_hull(pts):
    """Monotone-chain convex hull; returns hull vertices (M,2)."""
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def _half(seq):
        out = []
        for p in seq:
            while len(out) >= 2:
                a, b = out[-1] - out[-2], p - out[-2]
                if a[0] * b[1] - a[1] * b[0] > 0:
                    break
                out.pop()
            out.append(p)
        return out

    lower = _half(pts)
    upper = _half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def _pad_hull(hull, frac=0.10):
    c = hull.mean(0)
    return c + (hull - c) * (1.0 + frac)


# ═══════════════════════════════════════════════════════════════════════════
# F01 — three-tier system architecture
# ═══════════════════════════════════════════════════════════════════════════

def fig_f01(mode="paper"):
    S = _scales(mode)
    fig, ax = _canvas(mode, ratio=0.62)  # canvas 100 x 62

    _text(ax, 1.2, 60.3, "AQNet — three-tier PM2.5 fusion "
          "(Texas, 0.1° daily grid, 2021-01-01 → 2026-05-01)",
          S["title"], ha="left", weight="bold")

    # ── Data-source column (training side) ──
    sources = [
        ("PurpleAir", "PM2.5 target + sensor met", EDGE),
        ("Open-Meteo · ERA5", "daily met · SW/ET0/cloud", EDGE),
        ("NASA POWER", "met gap-fill fallback", EDGE),
        ("CAMS", "aod · cams_pm25 · dust", P_CAMS),
        ("GEOS-CF", "geoscf_pm25 chem. prior", P_GEOSCF),
        ("MERRA-2", "aerosol species · PBLH", P_MERRA2),
        ("NOAA HMS", "smoke plumes (tier 0–3)", EDGE),
        ("EJScreen — physical", "source proximity only", EDGE),
        ("TIGERweb · elevation", "tracts · terrain", EDGE),
    ]
    sx, sw, sh, sgap = 1.5, 17.5, 3.7, 0.85
    y_top = 55.6
    for i, (name, sub, ec) in enumerate(sources):
        y = y_top - i * (sh + sgap) - sh
        _box(ax, sx, y, sw, sh, fc=FILL, ec=ec, lw=S["lw_box"])
        _text(ax, sx + sw / 2, y + sh * 0.66, name, S["body"], weight="bold")
        _text(ax, sx + sw / 2, y + sh * 0.27, sub, S["tiny"], color=INK2)
    y_src_bot = y_top - len(sources) * (sh + sgap)

    # Collector spine -> tier arrows
    spine_x = sx + sw + 1.6
    ax.plot([spine_x, spine_x], [y_src_bot + 1.5, y_top - 1.0],
            color="#bbbbbb", lw=S["lw_box"], zorder=1)
    for i in range(len(sources)):
        yc = y_top - i * (sh + sgap) - sh / 2
        ax.plot([sx + sw, spine_x], [yc, yc], color="#bbbbbb",
                lw=0.6 * S["geo"], zorder=1)
    _arrow(ax, (spine_x, 47.0), (23.0, 47.0), lw=S["lw_arrow"], ams=S["ams"])
    _arrow(ax, (spine_x, 27.5), (23.0, 27.5), lw=S["lw_arrow"], ams=S["ams"])

    # ── Tier 1 ──
    t1 = dict(x=23, y=38, w=37, h=18)
    _box(ax, t1["x"], t1["y"], t1["w"], t1["h"], fc="#f2f7fc", ec=TIER1,
         lw=1.4 * S["geo"])
    _text(ax, t1["x"] + t1["w"] / 2, 54.4, "Tier 1 — tabular ensemble",
          S["head"], weight="bold")
    _text(ax, t1["x"] + t1["w"] / 2, 52.4,
          "features: 34 physical + dust + GEOS-CF + MERRA-2 (9)  →  ≤ 45",
          S["tiny"], color=INK2)
    chips = [("LightGBM", "Huber δ=10"), ("XGBoost", "ps.-Huber δ=10"),
             ("CatBoost", "depth 8"), ("RandomForest", "500 trees")]
    cw, cgap = 8.35, 0.55
    cx0 = t1["x"] + (t1["w"] - 4 * cw - 3 * cgap) / 2
    for j, (nm, sub) in enumerate(chips):
        x = cx0 + j * (cw + cgap)
        _box(ax, x, 46.7, cw, 4.4, fc="white", ec=TIER1, lw=S["lw_chip"],
             round_=0.5)
        _text(ax, x + cw / 2, 46.7 + 3.05, nm, S["tiny"], weight="bold")
        _text(ax, x + cw / 2, 46.7 + 1.35, sub, S["tiny"], color=INK2)
    _text(ax, t1["x"] + t1["w"] / 2, 45.6,
          "per-fold early stop on ~10% temporal tail (2000-round cap)",
          S["tiny"], color=INK2)
    _box(ax, 24.2, 39.2, 19.6, 4.6, fc="white", ec=TIER1, lw=S["lw_chip"],
         round_=0.5)
    _text(ax, 34.0, 42.4, "LOFO simplex blend", S["tiny"], weight="bold")
    _text(ax, 34.0, 40.6, "w ≥ 0, Σw = 1 (fold-out weights)", S["tiny"],
          color=INK2)
    _box(ax, 45.0, 39.2, 13.8, 4.6, fc="white", ec=TIER1, lw=S["lw_chip"],
         ls=(0, (3, 2)), round_=0.5)
    _text(ax, 51.9, 42.4, "LGBM quantile heads", S["tiny"], weight="bold")
    _text(ax, 51.9, 40.6, "q05 · q50 · q95 (for CQR)", S["tiny"], color=INK2)

    # ── Tier 2 ──
    t2 = dict(x=23, y=19.5, w=37, h=16)
    _box(ax, t2["x"], t2["y"], t2["w"], t2["h"], fc="#fdf6ea", ec=TIER2,
         lw=1.4 * S["geo"])
    tcx = t2["x"] + t2["w"] / 2
    _text(ax, tcx, 33.7, "Tier 2 — FusionUNet on 0.1° daily grids",
          S["head"], weight="bold")
    for k, line in enumerate([
            "8 source groups · 31 channels · sparse sensor supervision",
            "per-source encoders → per-pixel softmax attention → SE gate",
            "U-Net depth 4 (32→64→128→256→512) → softplus surface",
            "masked Huber δ=15 · modality dropout 0.15 · 96 px crops",
            "AdamW + warmup/cosine · AMP · 20% of sensor sites held out"]):
        _text(ax, tcx, 31.5 - 2.35 * k, line, S["tiny"], color=INK2)

    # ── Tier 3 ──
    t3 = dict(x=62, y=14, w=27, h=42)
    _box(ax, t3["x"], t3["y"], t3["w"], t3["h"], fc="#f0faf6", ec=TIER3,
         lw=1.4 * S["geo"])
    t3c = t3["x"] + t3["w"] / 2
    _text(ax, t3c, 54.3, "Tier 3 — cross-fitted stack\n+ conformal intervals",
          S["head"], weight="bold")
    parts = [
        ("tier1 — LOFO blend (OOF)", None, TIER1, "-"),
        ("rk — per-day kriged OOF residuals",
         "exp. variogram · ≤150 pts/day", EDGE, "-"),
        ("unet — Tier-2 OOF pixels",
         "held-out validation sites only", TIER2, "-"),
    ]
    py = 47.4
    for nm, sub, ec, ls in parts:
        h = 4.2 if sub else 3.2
        _box(ax, t3["x"] + 1.3, py - h, t3["w"] - 2.6, h, fc="white", ec=ec,
             lw=S["lw_chip"], ls=ls, round_=0.5)
        if sub:
            _text(ax, t3c, py - h + 2.85, nm, S["tiny"], weight="bold")
            _text(ax, t3c, py - h + 1.15, sub, S["tiny"], color=INK2)
        else:
            _text(ax, t3c, py - h / 2, nm, S["tiny"], weight="bold")
        py -= h + 0.9
    _arrow(ax, (t3c, py + 0.7), (t3c, py - 0.7), lw=S["lw_arrow"],
           ams=S["ams"])
    _box(ax, t3["x"] + 1.3, py - 5.6, t3["w"] - 2.6, 4.6, fc="white",
         ec=TIER3, lw=S["lw_chip"], round_=0.5)
    _text(ax, t3c, py - 2.4, "grouped Ridge — α=1, coef ≥ 0", S["tiny"],
          weight="bold")
    _text(ax, t3c, py - 4.2, "cross-fit over 4 sensor folds", S["tiny"],
          color=INK2)
    py -= 6.4
    _arrow(ax, (t3c, py + 0.7), (t3c, py - 0.7), lw=S["lw_arrow"],
           ams=S["ams"])
    _box(ax, t3["x"] + 1.3, py - 5.6, t3["w"] - 2.6, 4.6, fc="white",
         ec=TIER3, lw=S["lw_chip"], round_=0.5)
    _text(ax, t3c, py - 2.4, "CQR split-conformal interval", S["tiny"],
          weight="bold")
    _text(ax, t3c, py - 4.2, "recenter on Tier 3 · δ @ α=0.1 → 90%",
          S["tiny"], color=INK2)
    py -= 6.6
    ax.plot([t3["x"] + 1.3, t3["x"] + t3["w"] - 1.3], [py, py],
            color=LEAK_RED, lw=0.9 * S["geo"], ls=(0, (3, 2)), zorder=3)
    _text(ax, t3c, py - 1.5,
          "sensor-disjoint split (seed 42):",
          S["tiny"], color=INK2)
    _text(ax, t3c, py - 3.2, "75% meta-train · 25% calibration",
          S["tiny"], color=INK2)

    # Tier -> Tier3 arrows (all stacked inputs are strictly out-of-fold)
    _arrow(ax, (60.0, 47.0), (t3["x"], 47.0), lw=S["lw_arrow"], ams=S["ams"])
    _text(ax, 61.0, 47.9, "OOF", S["tiny"], color=INK2).set_bbox(
        dict(fc="white", ec="none", pad=0.3))
    _arrow(ax, (60.0, 27.5), (t3["x"], 27.5), lw=S["lw_arrow"], ams=S["ams"])
    _text(ax, 61.0, 28.4, "OOF", S["tiny"], color=INK2).set_bbox(
        dict(fc="white", ec="none", pad=0.3))

    # ── Outputs ──
    _box(ax, 91.0, 29.5, 8.6, 13.5, fc="white", ec=INK, lw=S["lw_box"])
    _text(ax, 95.3, 40.9, "Outputs", S["body"], weight="bold")
    _text(ax, 95.3, 36.5, "daily PM2.5\nsurface\n(µg/m³)", S["tiny"],
          color=INK2)
    _text(ax, 95.3, 31.8, "point + 90%\ninterval", S["tiny"], color=INK2)
    _arrow(ax, (t3["x"] + t3["w"], 36.2), (91.0, 36.2), lw=S["lw_arrow"],
           ams=S["ams"])

    # ── Leakage boundary + external-validation strip ──
    ax.plot([0.5, 99.5], [11.5, 11.5], color=LEAK_RED, lw=S["lw_sep"],
            ls=(0, (5, 3)), zorder=3)
    _text(ax, 99.2, 12.3, "training side ↑", S["tiny"], color=INK2,
          ha="right")
    _text(ax, 99.2, 10.6, "external validation only ↓", S["tiny"],
          color=INK2, ha="right")
    _box(ax, sx, 3.0, sw, 6.0, fc="#fdf3f3", ec=LEAK_RED, lw=S["lw_box"],
         ls=(0, (4, 2)))
    _text(ax, sx + sw / 2, 7.1, "EPA AQS FRM/FEM", S["body"], weight="bold")
    _text(ax, sx + sw / 2, 4.7, "62 monitors · external only",
          S["tiny"], color=INK2)
    _box(ax, 27, 3.0, 44, 6.0, fc="white", ec=LEAK_RED, lw=S["lw_box"],
         ls=(0, (4, 2)))
    _text(ax, 49, 7.1, "External validation", S["body"], weight="bold")
    _text(ax, 49, 4.7, "Tier-3 predicted at AQS site-days, scored vs "
          "FRM/FEM (86,316 site-days)", S["tiny"], color=INK2)
    _arrow(ax, (sx + sw, 6.0), (27, 6.0), color=LEAK_RED, ls=(0, (4, 2)),
           lw=S["lw_arrow"], ams=S["ams"])
    _text(ax, 23.0, 7.0, "truth", S["tiny"], color=INK2)
    _arrow(ax, (75.5, 14.0), (58.0, 9.2), lw=S["lw_arrow"], ams=S["ams"],
           rad=-0.18)
    _text(ax, 74.0, 10.6, "predictions", S["tiny"], color=INK2)
    _text(ax, 49, 1.3, "leakage boundary — AQS never enters any training, "
          "feature, or calibration set", S["tiny"], color=INK2,
          style="italic")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# F02 — FusionUNet architecture
# ═══════════════════════════════════════════════════════════════════════════

def fig_f02(mode="paper"):
    S = _scales(mode)
    fig, ax = _canvas(mode, ratio=0.66)  # canvas 100 x 66

    _text(ax, 1.2, 64.4, "Tier 2 — FusionUNet: fusion → per-pixel "
          "attention → U-Net → PM2.5 surface",
          S["title"], ha="left", weight="bold")

    # ── Channel groups (counts from grids.build_extended_stack) ──
    groups = [
        ("aerosol", 3, "aod · cams_pm25 · dust", P_CAMS),
        ("smoke", 1, "hms_smoke (0–3)", EDGE),
        ("meteorology", 8, "T·RH·P·wind·prcp·SW·ET0·cld", EDGE),
        ("static", 1, "elevation", EDGE),
        ("temporal", 4, "doy + dow sin/cos", EDGE),
        ("ctm", 1, "geoscf_pm25", P_GEOSCF),
        ("merra2", 6, "dust25·oc·bc·so4·ss25·pblh", P_MERRA2),
        ("flags", 7, "1 availability bit / source", EDGE),
    ]
    gx, gw, gh, ggap = 1.2, 15.2, 5.35, 1.05
    y_top = 57.6
    _text(ax, 8.3, 59.0, "8 groups · 31 channels", S["tiny"], color=INK2)
    enc_x, enc_w, enc_h = 20.0, 8.6, 3.0
    for i, (name, n, chans, ec) in enumerate(groups):
        y = y_top - i * (gh + ggap) - gh
        ls = (0, (3, 2)) if name == "flags" else "-"
        _box(ax, gx, y, gw, gh, fc=FILL, ec=ec, lw=S["lw_box"], ls=ls)
        _text(ax, gx + gw / 2, y + gh * 0.68, f"{name} — {n}", S["body"],
              weight="bold")
        _text(ax, gx + gw / 2, y + gh * 0.26, chans, S["tiny"], color=INK2)
        # per-source encoder
        yc = y + gh / 2
        _box(ax, enc_x, yc - enc_h / 2, enc_w, enc_h, fc="white", ec=TIER2,
             lw=S["lw_chip"], round_=0.5)
        _text(ax, enc_x + enc_w / 2, yc, "enc$_s$", S["tiny"])
        _arrow(ax, (gx + gw, yc), (enc_x, yc), lw=0.7 * S["geo"],
               ams=0.8 * S["ams"], color=INK2, alpha=0.8)
        _arrow(ax, (enc_x + enc_w, yc), (33.5, min(max(yc, 34.5), 51.0)),
               lw=0.7 * S["geo"], ams=0.8 * S["ams"], color=INK2, alpha=0.6,
               rad=0.0)
    _text(ax, 25.8, 59.0,
          "2×[conv3×3·GN·SiLU] → $e_s \\in \\mathbb{R}^{32}$",
          S["tiny"], color=INK2)

    # ── Per-pixel softmax attention ──
    _box(ax, 33.5, 32.5, 24.0, 20.5, fc="#fdf6ea", ec=TIER2,
         lw=1.3 * S["geo"])
    acx = 45.5
    _text(ax, acx, 50.8, "per-pixel source attention (S = 8)", S["body"],
          weight="bold")
    _text(ax, acx, 47.9, r"$z_s(p) = \mathrm{conv}_{3\times 3}(e_s(p))$",
          S["body"])
    _text(ax, acx, 43.9,
          r"$\alpha_s(p) = \dfrac{\exp z_s(p)}{\sum_{s'} \exp z_{s'}(p)}$",
          S["body"])
    _text(ax, acx, 39.3, r"$F(p) = \sum_s \alpha_s(p)\, e_s(p)$", S["body"])
    _text(ax, acx, 36.2, r"$\sum_s \alpha_s(p) = 1$ — attention maps "
          "returned", S["tiny"], color=INK2)
    _text(ax, acx, 34.3, "for interpretability  (B, 8, H, W)", S["tiny"],
          color=INK2)

    # ── Squeeze-excite gate ──
    _arrow(ax, (acx, 32.5), (acx, 30.3), lw=S["lw_arrow"], ams=S["ams"])
    _box(ax, 33.5, 20.5, 24.0, 9.8, fc="#fdf6ea", ec=TIER2,
         lw=1.3 * S["geo"])
    _text(ax, acx, 28.2, "squeeze-excite channel gate", S["body"],
          weight="bold")
    _text(ax, acx, 25.4,
          r"$g = \sigma(W_2\, \mathrm{SiLU}(W_1\, \mathrm{GAP}(F)))$",
          S["body"])
    _text(ax, acx, 22.6, r"$F \leftarrow F \odot g$   (32 → 8 → 32)",
          S["tiny"], color=INK2)
    _arrow(ax, (57.5, 25.4), (63.0, 49.0), lw=S["lw_arrow"], ams=S["ams"],
           rad=-0.35)
    _text(ax, 56.0, 18.6, "fused $F$ (B,32,H,W)", S["tiny"],
          color=INK2, ha="left")

    # ── U-Net (depth 4; widths from UNet(base_width=32)) ──
    enc_col_x, dec_col_x, blk_w = 63.0, 92.2, 6.2
    levels = [  # (y, h, enc label, dec label, spatial)
        (46.0, 11.0, "32", "32", "H×W"),
        (36.5, 7.5, "64", "64", "H/2"),
        (28.5, 6.0, "128", "128", "H/4"),
        (21.0, 5.0, "256", "256", "H/8"),
    ]
    for (y, h, el, dl, sp) in levels:
        _box(ax, enc_col_x, y, blk_w, h, fc="#eef2f7", ec=INK2,
             lw=S["lw_box"], round_=0.4)
        _text(ax, enc_col_x + blk_w / 2, y + h * 0.60, el, S["body"],
              weight="bold")
        _text(ax, enc_col_x + blk_w / 2, y + h * 0.24, sp, S["tiny"],
              color=INK2)
        _box(ax, dec_col_x, y, blk_w, h, fc="#eef2f7", ec=INK2,
             lw=S["lw_box"], round_=0.4)
        _text(ax, dec_col_x + blk_w / 2, y + h * 0.60, dl, S["body"],
              weight="bold")
        _text(ax, dec_col_x + blk_w / 2, y + h * 0.24, sp, S["tiny"],
              color=INK2)
        # skip connection
        _arrow(ax, (enc_col_x + blk_w, y + h / 2), (dec_col_x, y + h / 2),
               ls=(0, (3, 2)), lw=0.8 * S["geo"], ams=0.8 * S["ams"],
               color="#888888", alpha=0.9)
    _text(ax, (enc_col_x + dec_col_x + blk_w) / 2, 52.6, "skip → concat",
          S["tiny"], color=INK2)
    # pooling arrows down the encoder / up the decoder
    for (y0, h0), (y1, h1) in zip([(46.0, 11.0), (36.5, 7.5), (28.5, 6.0)],
                                  [(36.5, 7.5), (28.5, 6.0), (21.0, 5.0)]):
        _arrow(ax, (enc_col_x + blk_w / 2, y0), (enc_col_x + blk_w / 2,
               y1 + h1), lw=0.8 * S["geo"], ams=0.8 * S["ams"], color=INK2)
        _arrow(ax, (dec_col_x + blk_w / 2, y1 + h1), (dec_col_x + blk_w / 2,
               y0), lw=0.8 * S["geo"], ams=0.8 * S["ams"], color=INK2)
    _text(ax, enc_col_x + blk_w + 0.7, 45.0, "maxpool /2", S["tiny"],
          color=INK2, ha="left").set_bbox(dict(fc="white", ec="none",
                                               pad=0.3))
    _text(ax, dec_col_x + blk_w + 0.9, 41.5, "bilinear ×2 · 1×1", S["tiny"],
          color=INK2, rot=270)
    # bottleneck
    _box(ax, 72.5, 11.0, 16.5, 5.2, fc="#e3e9f0", ec=INK2, lw=S["lw_box"],
         round_=0.4)
    _text(ax, 80.75, 14.3, "bottleneck 512", S["body"], weight="bold")
    _text(ax, 80.75, 12.4, "H/16", S["tiny"], color=INK2)
    _arrow(ax, (enc_col_x + blk_w / 2, 21.0), (73.5, 14.0),
           lw=0.8 * S["geo"], ams=0.8 * S["ams"], rad=0.25, color=INK2)
    _arrow(ax, (88.2, 14.0), (dec_col_x + blk_w / 2, 21.0),
           lw=0.8 * S["geo"], ams=0.8 * S["ams"], rad=0.25, color=INK2)
    _text(ax, 80.75, 9.4, "DoubleConv = 2×[conv3×3 · GroupNorm · SiLU] · "
          "input padded to ×16", S["tiny"], color=INK2)
    # output head
    _arrow(ax, (dec_col_x + blk_w / 2, 57.0), (dec_col_x + blk_w / 2, 58.6),
           lw=S["lw_arrow"], ams=S["ams"])
    _box(ax, 77.0, 58.8, 22.5, 4.7, fc="white", ec=TIER2, lw=S["lw_box"])
    _text(ax, 88.25, 62.1, "1×1 conv → softplus", S["tiny"], weight="bold")
    _text(ax, 88.25, 60.2, "PM2.5 surface — 1×H×W (µg/m³, ≥ 0)", S["tiny"],
          color=INK2)

    # ── Training notes ──
    _text(ax, 1.2, 5.2, "training: masked Huber (δ = 15) at sensor pixels "
          "only · AdamW (lr 1e-3, wd 1e-3) · 5-epoch warmup → cosine · "
          "AMP · 96 px random crops", S["tiny"], color=INK2, ha="left")
    _text(ax, 1.2, 3.3, "modality dropout: each source group zeroed w.p. "
          "0.15 per batch after normalization — flags exempt, so the "
          "availability signal survives", S["tiny"], color=INK2, ha="left")
    _text(ax, 1.2, 1.4, "flags group: 1 binary channel per source group, "
          "1 when the source reported that day pre-fill (flag-and-fill)",
          S["tiny"], color=INK2, ha="left")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# F03 — data provenance & leakage status
# ═══════════════════════════════════════════════════════════════════════════

def fig_f03(mode="paper"):
    S = _scales(mode)
    fig, ax = _canvas(mode, ratio=0.74)  # canvas 100 x 74

    _text(ax, 1.2, 72.3, "AQNet data provenance — training inputs vs "
          "external-validation-only sources",
          S["title"], ha="left", weight="bold")

    # legend (flow color code)
    handles = [
        Line2D([], [], color=INK2, lw=1.2 * S["geo"], label="training flow"),
        Line2D([], [], color=LEAK_RED, lw=1.2 * S["geo"], ls=(0, (4, 2)),
               label="external-validation-only flow"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False,
              fontsize=S["tiny"], bbox_to_anchor=(0.995, 0.955),
              handlelength=2.4)

    # ── Sources (left) ──
    N1, N2, N3, N4 = "n1", "n2", "n3", "n4"
    sources = [
        ("PurpleAir", "ATM PM2.5 + RH + sensor met", EDGE, [N1, N2, N3]),
        ("Open-Meteo · ERA5", "daily met · shortwave/ET0/cloud", EDGE,
         [N2, N3]),
        ("NASA POWER", "met gap-fill (quota fallback)", EDGE, [N2]),
        ("CAMS", "aod · cams_pm25 · dust", P_CAMS, [N2, N3]),
        ("GEOS-CF", "geoscf_pm25 (chem. transport)", P_GEOSCF, [N2, N3]),
        ("MERRA-2", "aerosol species · PBLH", P_MERRA2, [N2, N3]),
        ("NOAA HMS", "smoke plumes (tier 0–3)", EDGE, [N2, N3]),
        ("EJScreen — physical", "proximity metrics only", EDGE, [N2]),
        ("TIGERweb", "tract geometry · centroids", EDGE, [N2, N3]),
        ("Elevation", "sensor + tract terrain", EDGE, [N2, N3]),
        ("EPA AQS FRM/FEM", "62 monitors · 86,316 site-days", LEAK_RED,
         [N4]),
    ]
    sx, sw, sh, sgap = 1.2, 22.5, 4.5, 0.95
    y_top = 66.8
    centers = {}
    for i, (name, sub, ec, feeds) in enumerate(sources):
        y = y_top - i * (sh + sgap) - sh
        external = ec is LEAK_RED
        _box(ax, sx, y, sw, sh, fc="#fdf3f3" if external else FILL, ec=ec,
             lw=S["lw_box"], ls=(0, (4, 2)) if external else "-")
        _text(ax, sx + sw / 2, y + sh * 0.66, name, S["body"], weight="bold")
        _text(ax, sx + sw / 2, y + sh * 0.27, sub, S["tiny"], color=INK2)
        centers[i] = (sx + sw, y + sh / 2, feeds)
    y_aqs_top = y_top - 10 * (sh + sgap)  # top of the AQS box

    # full-width leakage separator just above the AQS row
    y_sep = y_aqs_top + 0.45
    ax.plot([0.5, 99.5], [y_sep, y_sep], color=LEAK_RED, lw=S["lw_sep"],
            ls=(0, (5, 3)), zorder=3)
    _text(ax, 29.5, y_sep + 1.2, "training ↑", S["tiny"], color=INK2)
    _text(ax, 29.5, y_sep - 1.4, "external-only ↓", S["tiny"], color=INK2)

    # ── Middle nodes ──
    nx, nw = 36.5, 26.0
    nodes = {
        N1: dict(y=57.5, h=7.0, head="Supervision target",
                 lines=["Barkjohn-corrected PurpleAir", "PM2.5 (µg/m³)"],
                 ec=EDGE, ext=False),
        N2: dict(y=40.5, h=10.0, head="Tier-1 tabular features",
                 lines=["34 physical + dust + GEOS-CF + MERRA-2 (9)",
                        "+ same-day leave-self-out neighbor PM2.5"],
                 ec=TIER1, ext=False),
        N3: dict(y=24.5, h=10.0, head="Tier-2 grid channels",
                 lines=["8 groups · 31 channels @ 0.1°",
                        "flags = per-source availability bits"],
                 ec=TIER2, ext=False),
        N4: dict(y=4.5, h=7.0, head="External truth",
                 lines=["never a feature, target,", "or calibration input"],
                 ec=LEAK_RED, ext=True),
    }
    for key, nd in nodes.items():
        _box(ax, nx, nd["y"], nw, nd["h"], fc="#fdf3f3" if nd["ext"]
             else "white", ec=nd["ec"], lw=1.2 * S["geo"],
             ls=(0, (4, 2)) if nd["ext"] else "-")
        _text(ax, nx + nw / 2, nd["y"] + nd["h"] - 1.7, nd["head"],
              S["body"], weight="bold")
        for k, line in enumerate(nd["lines"]):
            _text(ax, nx + nw / 2, nd["y"] + nd["h"] - 3.6 - 1.8 * k, line,
                  S["tiny"], color=INK2)

    # source -> node arrows (entry points spread along each node's left edge)
    entry_count = {k: 0 for k in nodes}
    inbound = {k: sum(1 for *_r, feeds in sources if k in feeds)
               for k in nodes}
    for i, (_n, _s, ec, feeds) in enumerate(sources):
        x0, y0, _ = centers[i]
        for key in feeds:
            nd = nodes[key]
            j = entry_count[key]
            entry_count[key] += 1
            frac = (j + 1) / (inbound[key] + 1)
            y1 = nd["y"] + nd["h"] * (1 - frac)
            external = key == N4
            _arrow(ax, (x0, y0), (nx, y1),
                   color=LEAK_RED if external else INK2,
                   ls=(0, (4, 2)) if external else "-",
                   lw=0.7 * S["geo"], ams=0.75 * S["ams"],
                   alpha=0.9 if external else 0.55,
                   rad=0.06 if y1 < y0 else -0.06)

    # ── Consumers (right) ──
    rx, rw = 73.0, 23.5
    r_t1 = dict(y=47.0, h=6.5)
    r_t2 = dict(y=38.0, h=6.5)
    r_t3 = dict(y=27.0, h=6.5)
    r_ev = dict(y=4.5, h=7.0)
    _box(ax, rx, r_t1["y"], rw, r_t1["h"], fc="#f2f7fc", ec=TIER1,
         lw=1.2 * S["geo"])
    _text(ax, rx + rw / 2, r_t1["y"] + 4.3, "Tier 1 ensemble", S["body"],
          weight="bold")
    _text(ax, rx + rw / 2, r_t1["y"] + 1.9, "4 learners + LOFO blend "
          "+ q-heads", S["tiny"], color=INK2)
    _box(ax, rx, r_t2["y"], rw, r_t2["h"], fc="#fdf6ea", ec=TIER2,
         lw=1.2 * S["geo"])
    _text(ax, rx + rw / 2, r_t2["y"] + 4.3, "Tier 2 FusionUNet", S["body"],
          weight="bold")
    _text(ax, rx + rw / 2, r_t2["y"] + 1.9, "masked Huber at sensor pixels",
          S["tiny"], color=INK2)
    _box(ax, rx, r_t3["y"], rw, r_t3["h"], fc="#f0faf6", ec=TIER3,
         lw=1.2 * S["geo"])
    _text(ax, rx + rw / 2, r_t3["y"] + 4.3, "Tier 3 stack", S["body"],
          weight="bold")
    _text(ax, rx + rw / 2, r_t3["y"] + 1.9, "consumes Tier-1/-2 OOF only",
          S["tiny"], color=INK2)
    _box(ax, rx, r_ev["y"], rw, r_ev["h"], fc="#fdf3f3", ec=LEAK_RED,
         lw=1.2 * S["geo"], ls=(0, (4, 2)))
    _text(ax, rx + rw / 2, r_ev["y"] + 4.8, "External validation", S["body"],
          weight="bold")
    _text(ax, rx + rw / 2, r_ev["y"] + 2.3, "Tier-3 scored at AQS site-days",
          S["tiny"], color=INK2)

    # node -> consumer arrows
    _arrow(ax, (nx + nw, 60.5), (rx, r_t1["y"] + 5.2), lw=0.9 * S["geo"],
           ams=0.85 * S["ams"], rad=-0.12, alpha=0.8)
    _arrow(ax, (nx + nw, 58.8), (rx, r_t2["y"] + 5.0), lw=0.9 * S["geo"],
           ams=0.85 * S["ams"], rad=-0.22, alpha=0.8)
    _text(ax, 67.5, 57.2, "target y", S["tiny"], color=INK2)
    _arrow(ax, (nx + nw, 45.5), (rx, r_t1["y"] + 2.3), lw=0.9 * S["geo"],
           ams=0.85 * S["ams"], alpha=0.8)
    _arrow(ax, (nx + nw, 30.5), (rx, r_t2["y"] + 2.3), lw=0.9 * S["geo"],
           ams=0.85 * S["ams"], alpha=0.8)
    # Tier1 -> Tier3 down the outer right lane (around Tier2)
    _arrow(ax, (rx + rw, r_t1["y"] + 1.8), (rx + rw, r_t3["y"] + 4.7),
           lw=0.8 * S["geo"], ams=0.8 * S["ams"], alpha=0.8, rad=0.18)
    _arrow(ax, (rx + rw / 2, r_t2["y"]), (rx + rw / 2, r_t3["y"]
           + r_t3["h"]), lw=0.8 * S["geo"], ams=0.8 * S["ams"], alpha=0.8)
    _arrow(ax, (rx + rw / 2, r_t3["y"]), (rx + rw / 2, r_ev["y"]
           + r_ev["h"]), lw=0.9 * S["geo"], ams=0.85 * S["ams"], alpha=0.8)
    _text(ax, rx + rw / 2 + 1.4, 19.5, "predictions", S["tiny"], color=INK2,
          ha="left")
    _arrow(ax, (nx + nw, 8.0), (rx, 8.0), color=LEAK_RED, ls=(0, (4, 2)),
           lw=0.9 * S["geo"], ams=0.85 * S["ams"])
    _text(ax, (nx + nw + rx) / 2, 9.0, "truth", S["tiny"], color=INK2)

    _text(ax, 1.2, 1.2, "EJScreen demographic columns (ejf_score, % people "
          "of color, % low income, % linguistically isolated) are excluded "
          "from every model input; used only in the demographics ablation.",
          S["tiny"], color=INK2, ha="left", style="italic")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# F04 — validation protocol
# ═══════════════════════════════════════════════════════════════════════════

def _f04_loso_panel(ax, S):
    """(a) LOSO — GroupKFold(10) over sensor_id."""
    ax.set_xlim(-2.4, 11.0)
    ax.set_ylim(-1.8, 11.6)
    ax.axis("off")
    ax.set_title("(a) LOSO — GroupKFold(10) over sensors",
                 fontsize=S["head"], color=INK, pad=4 * S["geo"], loc="left")
    for i in range(10):
        y = 9.3 - i
        for j in range(10):
            fc = "#8c8c8c" if j == i else TRAIN_SHADE
            ax.add_patch(Rectangle((j, y), 0.94, 0.74, fc=fc, ec="white",
                                   lw=0.4 * S["geo"]))
        ax.text(-0.25, y + 0.37, f"fold {i + 1}", fontsize=S["tiny"],
                color=INK2, ha="right", va="center")
    ax.text(5.0, -0.6, "unique sensors → 10 groups (shuffled, seed 42)",
            fontsize=S["tiny"], color=INK2, ha="center")
    ax.text(5.0, -1.5, "held-out sensors never inform their own fold",
            fontsize=S["tiny"], color=INK2, ha="center", style="italic")
    ax.legend(handles=[Patch(fc=TRAIN_SHADE, ec="#cccccc", label="train"),
                       Patch(fc="#8c8c8c", label="held-out sensors")],
              loc="upper left", bbox_to_anchor=(0.0, 1.02), frameon=False,
              fontsize=S["tiny"], ncol=2, handlelength=1.1,
              columnspacing=0.9, borderaxespad=0.0)


def _f04_spatial_panel(ax, S):
    """(b) spatial blocks — KMeans(5) over real sensor sites on Texas."""
    ax.set_title("(b) spatial blocks — KMeans(5)",
                 fontsize=S["head"], color=INK, pad=4 * S["geo"], loc="left")
    ax.set_aspect(1.15)
    ax.axis("off")
    polys = _load_tract_polys()
    if polys is not None:
        ax.add_collection(PolyCollection(
            polys, facecolors="#ececec", edgecolors="#dcdcdc",
            linewidths=0.15 * S["geo"], rasterized=True, zorder=1))
    xy = _load_sensor_xy()
    lab, _cent = _kmeans(xy, k=5, seed=42)
    counts = np.bincount(lab, minlength=5)
    j_hold = int(np.argmax(counts))
    for j in range(5):
        pts = xy[lab == j]
        if len(pts) >= 3:
            hull = _pad_hull(_convex_hull(pts), 0.12)
            ax.add_patch(plt.Polygon(hull, closed=True, fill=False,
                                     ec="#999999", lw=0.7 * S["geo"],
                                     ls=(0, (3, 2)), zorder=3))
            cx, cy = pts.mean(0)
            ax.text(cx, cy, str(j + 1), fontsize=S["body"], color=INK2,
                    ha="center", va="center", zorder=6, fontweight="bold",
                    bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.75))
    tr = xy[lab != j_hold]
    ho = xy[lab == j_hold]
    ax.scatter(tr[:, 0], tr[:, 1], s=6.5 * S["geo"] ** 2, c="#8f8f8f",
               marker="o", linewidths=0, zorder=4, label="train sites")
    ax.scatter(ho[:, 0], ho[:, 1], s=8.5 * S["geo"] ** 2, facecolors="none",
               edgecolors=INK, marker="o", linewidths=0.6 * S["geo"],
               zorder=4, label="held-out block")
    ax.set_xlim(-107.2, -93.1)
    ax.set_ylim(24.6, 37.0)
    ax.legend(loc="lower left", frameon=False, fontsize=S["tiny"],
              handletextpad=0.3, borderaxespad=0.2)
    ax.text(-93.4, 36.3, "leave-one-region-out;\nsites: repo membership "
            "file", fontsize=S["tiny"], color=INK2, ha="right", va="top")


def _f04_temporal_panel(ax, S):
    """(c) temporal holdout on the 2021 → 2026-05 window."""
    ax.set_title("(c) temporal holdout — cutoff 2025-01-01",
                 fontsize=S["head"], color=INK, pad=4 * S["geo"], loc="left")
    x0, x1, xc = 2021.0, 2026.0 + 4.0 / 12.0, 2025.0
    ax.axvspan(x0, xc, color=TRAIN_SHADE, zorder=1)
    ax.axvspan(xc, x1, facecolor="#fafafa", hatch="///",
               edgecolor="#bbbbbb", linewidth=0, zorder=1)
    ax.axvline(xc, color=INK, lw=1.0 * S["geo"], ls=(0, (4, 2)), zorder=3)
    ax.text((x0 + xc) / 2, 0.60, "train\n2021-01-01 … 2024-12-31",
            fontsize=S["tiny"], color=INK, ha="center", va="center", zorder=4)
    ax.text((xc + x1) / 2 + 0.10, 0.60, "test\n≥ 2025-01-01",
            fontsize=S["tiny"], color=INK, ha="center", va="center",
            zorder=4, bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.7))
    ax.set_xlim(x0 - 0.08, x1 + 0.08)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks(range(2021, 2027))
    ax.set_xticklabels([str(y) for y in range(2021, 2027)],
                       fontsize=S["tiny"], color=INK)
    ax.set_xlabel("date (year)", fontsize=S["tiny"], color=INK)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, width=0.6 * S["geo"],
                   labelsize=S["tiny"])


def _f04_aqs_panel(ax, S):
    """(d) external validation against EPA AQS."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("(d) external validation — EPA AQS FRM/FEM",
                 fontsize=S["head"], color=INK, pad=4 * S["geo"], loc="left")
    _box(ax, 0.3, 3.4, 4.0, 4.6, fc=FILL, ec=TIER3, lw=1.1 * S["geo"],
         round_=0.25)
    _text(ax, 2.3, 6.8, "AQNet Tier 3", S["body"], weight="bold")
    _text(ax, 2.3, 5.5, "trained on PurpleAir", S["tiny"], color=INK2)
    _text(ax, 2.3, 4.4, "sensor-days only", S["tiny"], color=INK2)
    _box(ax, 6.0, 3.4, 3.8, 4.6, fc="#fdf3f3", ec=LEAK_RED,
         lw=1.1 * S["geo"], ls=(0, (4, 2)), round_=0.25)
    _text(ax, 7.9, 6.8, "EPA AQS", S["body"], weight="bold")
    _text(ax, 7.9, 5.5, "62 monitors", S["tiny"], color=INK2)
    _text(ax, 7.9, 4.4, "86,316 site-days", S["tiny"], color=INK2)
    ax.plot([5.15, 5.15], [3.2, 9.2], color=LEAK_RED, lw=1.1 * S["geo"],
            ls=(0, (5, 3)), zorder=3)
    _text(ax, 5.15, 9.6, "leakage boundary", S["tiny"], color=INK2)
    _arrow(ax, (4.3, 6.2), (6.0, 6.2), lw=S["lw_arrow"], ams=S["ams"])
    _text(ax, 5.15, 6.9, "predict", S["tiny"], color=INK2)
    _text(ax, 5.15, 2.4, "scored vs FRM/FEM truth — AQS never used in "
          "training,\nfeatures, stacking, or conformal calibration",
          S["tiny"], color=INK2)


def _f04_meta_panel(ax, S):
    """(e) Tier-3 meta split: cross-fit vs conformal calibration sensors."""
    ax.set_xlim(-0.03, 1.06)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("(e) Tier-3 meta split — cross-fit vs calibration",
                 fontsize=S["head"], color=INK, pad=4 * S["geo"], loc="left")
    y0, h = 4.0, 2.4
    # meta-train 75%, subdivided into the 4 grouped cross-fit folds
    for j in range(4):
        x = j * 0.1875
        fc = "#bdbdbd" if j == 1 else TRAIN_SHADE
        ax.add_patch(Rectangle((x, y0), 0.1875, h, fc=fc, ec="white",
                               lw=0.5 * S["geo"], zorder=2))
    ax.add_patch(Rectangle((0.75, y0), 0.25, h, fc="#fafafa", ec="#bbbbbb",
                           hatch="///", lw=0.6 * S["geo"], zorder=2))
    ax.plot([0.75, 0.75], [y0 - 1.5, y0 + h + 1.5], color=LEAK_RED,
            lw=1.0 * S["geo"], ls=(0, (4, 2)), zorder=3)
    _text(ax, 0.75, y0 + h + 2.1, "sensor-disjoint (seed 42)", S["tiny"],
          color=INK2)
    _text(ax, 0.375, y0 + h + 1.0, "meta-train sensors — 75%", S["tiny"],
          color=INK)
    _text(ax, 0.875, y0 + h + 1.0, "calibration — 25%", S["tiny"],
          color=INK)
    _text(ax, 0.34, y0 - 1.1, "cross-fit Ridge (coef ≥ 0), 4 grouped "
          "folds:\ncombiner never sees its own sensor's rows",
          S["tiny"], color=INK2)
    _text(ax, 0.9, y0 - 1.1, "CQR conformal δ\n(α = 0.1 → 90%)",
          S["tiny"], color=INK2)
    _text(ax, 0.5, 9.2, "unique sensors →", S["tiny"], color=INK2)
    _text(ax, 0.5, 0.4, "one sub-fold at a time is held out inside "
          "meta-train (darker block)", S["tiny"], color=INK2,
          style="italic")


def fig_f04(mode="paper"):
    if _load_sensor_xy() is None:
        print(f"[fig_architecture] SKIP F04_validation_protocol — sensor "
              f"membership file missing: {MEMBERSHIP_CSV}")
        return None
    S = _scales(mode)
    w = FIG_W["poster"] if mode == "poster" else FIG_W["double"]
    fig = plt.figure(figsize=(w, w * 0.78))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.82],
                          left=0.055, right=0.985, top=0.90, bottom=0.045,
                          wspace=0.55, hspace=0.42)
    fig.suptitle("AQNet validation protocol — four axes plus the Tier-3 "
                 "meta split", fontsize=S["title"], color=INK,
                 fontweight="bold", x=0.055, ha="left", y=0.975)
    _f04_loso_panel(fig.add_subplot(gs[0, 0:2]), S)
    _f04_spatial_panel(fig.add_subplot(gs[0, 2:4]), S)
    _f04_temporal_panel(fig.add_subplot(gs[0, 4:6]), S)
    _f04_aqs_panel(fig.add_subplot(gs[1, 0:3]), S)
    _f04_meta_panel(fig.add_subplot(gs[1, 3:6]), S)
    return fig


# ── Driver ──────────────────────────────────────────────────────────────────

FIGURES = {
    "F01_three_tier_architecture": fig_f01,
    "F02_fusionunet_architecture": fig_f02,
    "F03_data_provenance": fig_f03,
    "F04_validation_protocol": fig_f04,
}


def main():
    print(f"[fig_architecture] style source: {_STYLE_SOURCE}")
    rendered = []
    for mode in ("paper", "poster"):
        set_style(mode)
        for name, fn in FIGURES.items():
            fig = fn(mode)
            if fig is None:
                continue
            paths = save_fig(fig, name, mode, preview=False)
            plt.close(fig)
            rendered.extend(paths)
            print(f"  rendered {name} [{mode}] -> "
                  f"{os.path.basename(paths[0])} (+png/svg)")
    print(f"[fig_architecture] wrote {len(rendered)} files")
    return rendered


if __name__ == "__main__":
    main()
