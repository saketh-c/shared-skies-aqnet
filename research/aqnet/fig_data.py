"""Data-description figures for AQNet (F05-F07), rendered from repo data only.

F05_study_domain      Texas tract outlines (light-gray context, rasterized)
                      with PurpleAir sensors (dots sized/colored by data-days,
                      viridis), the EPA AQS monitors (open triangles, neutral
                      ink), the 0.1-deg grid extent rectangle, and counts.
F06_data_coverage     (a) per-source coverage windows with the temporal-holdout
                      cutoff marked; (b) active-sensor count per month from the
                      PurpleAir parquet (the survivorship caveat visualized).
F07_barkjohn_correction  (a) raw ATM vs Barkjohn-corrected PM2.5 density hexbin
                      with the 1:1 line; (b) correction magnitude vs relative
                      humidity (binned means with an IQR band). ~50k sampled
                      real sensor-day rows.

Every figure renders in both "paper" and "poster" modes through fig_style
(set_style / FIG_W / save_fig with preview=False — these are final renders
from full repo data, not quick-mode artifacts). If an input file is missing
the figure prints a clear skip message and returns None.

Run:  python research/aqnet/fig_data.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import config
from corrections import (BARKJOHN_INTERCEPT, BARKJOHN_RH_COEF, BARKJOHN_SLOPE,
                         barkjohn_correct)
from fig_style import COLORS, FIG_W, SEQUENTIAL_CMAP, save_fig, set_style

# ── Input paths (repo data only) ────────────────────────────────────────────

PA_PARQUET = os.path.join(config.ROOT, "pipeline", "purpleair_full_dataset.parquet")
AQS_PARQUET = os.path.join(config.DATA_DIR, "aqs_daily_tx.parquet")
TRACTS_GEOJSON = os.path.join(config.ROOT, "backend", "static",
                              "texas_all_tracts.geojson")

# CAMS global atmospheric-composition forecasts are public from this date, so
# the cams_pm25 / dust channels only exist from here onward (F06 panel a).
CAMS_START = "2022-08-03"

_NEUTRAL = "#555555"
_INK = "#333333"


def _require(path, fig_name):
    """True if path exists; otherwise print the standard skip message."""
    if os.path.exists(path):
        return True
    print(f"[skip] {fig_name}: missing input {path}")
    return False


def _scale(mode):
    """Linear size factor for poster mode (fonts scale ~2.1x in set_style)."""
    return 1.0 if mode == "paper" else 18.0 / 8.5


def _lw(mode):
    """Main line width: ~1.6pt paper / 3pt poster."""
    return 1.6 if mode == "paper" else 3.0


def _panel_tag(ax, tag):
    """Short bold panel tag (a/b/...) in neutral ink, top-left of the axes."""
    ax.text(0.0, 1.02, f"({tag})", transform=ax.transAxes, ha="left",
            va="bottom", fontweight="bold", color=_INK)


# ── Cached loaders ──────────────────────────────────────────────────────────

_CACHE = {}


def _load_pa():
    """PurpleAir sensor-day rows (the raw ATM pm25 column plus humidity)."""
    if "pa" not in _CACHE:
        df = pd.read_parquet(PA_PARQUET, columns=[
            "sensor_id", "date", "pm25", "humidity", "latitude", "longitude"])
        df["date"] = pd.to_datetime(df["date"])
        _CACHE["pa"] = df
    return _CACHE["pa"]


def _load_aqs():
    if "aqs" not in _CACHE:
        df = pd.read_parquet(AQS_PARQUET)
        df["date"] = pd.to_datetime(df["date"])
        _CACHE["aqs"] = df
    return _CACHE["aqs"]


def _load_tract_polys():
    """Exterior rings of every Texas tract as a list of (N, 2) lon/lat arrays.

    Loaded once per process; drawn as a single rasterized PolyCollection so
    the vector outputs stay small.
    """
    if "tracts" not in _CACHE:
        import json
        with open(TRACTS_GEOJSON, encoding="utf-8") as f:
            gj = json.load(f)
        polys = []
        for feat in gj["features"]:
            geom = feat.get("geometry") or {}
            if geom.get("type") == "Polygon":
                polys.append(np.asarray(geom["coordinates"][0], dtype=np.float64))
            elif geom.get("type") == "MultiPolygon":
                for part in geom["coordinates"]:
                    polys.append(np.asarray(part[0], dtype=np.float64))
        _CACHE["tracts"] = polys
    return _CACHE["tracts"]


# ── F05: study domain ───────────────────────────────────────────────────────

def fig_study_domain(mode="paper"):
    """Texas domain map: tracts, PurpleAir sensors, AQS monitors, grid extent."""
    name = "F05_study_domain"
    for path in (PA_PARQUET, AQS_PARQUET, TRACTS_GEOJSON):
        if not _require(path, name):
            return None
    set_style(mode)
    sc = _scale(mode)

    pa = _load_pa()
    sensors = (pa.groupby("sensor_id")
                 .agg(lat=("latitude", "median"), lon=("longitude", "median"),
                      days=("date", "size"))
                 .reset_index())
    aqs = _load_aqs()
    monitors = aqs.groupby("site_id")[["lat", "lon"]].median().reset_index()

    bbox = config.TX_BBOX
    width = FIG_W["double" if mode == "paper" else "poster"]
    fig, ax = plt.subplots(figsize=(width, width * 0.78))
    ax.grid(False)

    # Tract outlines: light-gray context, rasterized to keep vectors small.
    ax.add_collection(PolyCollection(
        _load_tract_polys(), facecolors="none", edgecolors="#c9c9c9",
        linewidths=0.25 * sc, rasterized=True, zorder=1))

    # 0.1-deg model grid extent.
    ax.add_patch(Rectangle(
        (bbox["lon_min"], bbox["lat_min"]),
        bbox["lon_max"] - bbox["lon_min"], bbox["lat_max"] - bbox["lat_min"],
        facecolor="none", edgecolor=_NEUTRAL, linewidth=0.9 * sc,
        linestyle="--", zorder=2))

    # PurpleAir sensors: dot color AND area encode data-days (viridis).
    d = sensors["days"].to_numpy(dtype=float)
    smin, smax = 8.0 * sc**2, 42.0 * sc**2
    sizes = smin + (smax - smin) * (d - d.min()) / max(d.max() - d.min(), 1.0)
    pts = ax.scatter(sensors["lon"], sensors["lat"], c=d, s=sizes,
                     cmap=SEQUENTIAL_CMAP, alpha=0.85, linewidths=0.3 * sc,
                     edgecolors="white", zorder=3)

    # EPA AQS monitors: open triangles in neutral ink (never trained on).
    ax.scatter(monitors["lon"], monitors["lat"], marker="^", s=30.0 * sc**2,
               facecolors="none", edgecolors=_INK, linewidths=0.8 * sc,
               zorder=4)

    pad = 0.35
    ax.set_xlim(bbox["lon_min"] - pad, bbox["lon_max"] + pad)
    ax.set_ylim(bbox["lat_min"] - pad, bbox["lat_max"] + pad)
    mid_lat = 0.5 * (bbox["lat_min"] + bbox["lat_max"])
    ax.set_aspect(1.0 / np.cos(np.deg2rad(mid_lat)))
    ax.set_xlabel("Longitude (\N{DEGREE SIGN}E)")
    ax.set_ylabel("Latitude (\N{DEGREE SIGN}N)")

    cbar = fig.colorbar(pts, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Data-days per sensor (dot color and size)")
    cbar.outline.set_visible(False)

    mid = plt.get_cmap(SEQUENTIAL_CMAP)(0.6)
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=6 * sc,
               markerfacecolor=mid, markeredgecolor="white",
               markeredgewidth=0.3 * sc,
               label=f"PurpleAir sensor (n={len(sensors)})"),
        Line2D([], [], marker="^", linestyle="none", markersize=7 * sc,
               markerfacecolor="none", markeredgecolor=_INK,
               markeredgewidth=0.8 * sc,
               label=f"EPA AQS monitor (n={len(monitors)})"),
        Line2D([], [], linestyle="--", color=_NEUTRAL, linewidth=0.9 * sc,
               label=f"{config.GRID_DEG}\N{DEGREE SIGN} grid extent"),
    ]
    # Above the axes: every in-map corner contains sensors or monitors.
    leg = ax.legend(handles=handles, loc="lower left",
                    bbox_to_anchor=(0.0, 1.005), ncol=2, frameon=False,
                    borderaxespad=0.0, handlelength=1.4, columnspacing=1.2)
    for txt in leg.get_texts():
        txt.set_color(_INK)

    paths = save_fig(fig, name, mode, preview=False)
    plt.close(fig)
    return paths


# ── F06: data coverage ──────────────────────────────────────────────────────

def fig_data_coverage(mode="paper"):
    """(a) source coverage windows + holdout cutoff; (b) sensors per month."""
    name = "F06_data_coverage"
    for path in (PA_PARQUET, AQS_PARQUET):
        if not _require(path, name):
            return None
    set_style(mode)
    sc, lw = _scale(mode), _lw(mode)

    pa = _load_pa()
    aqs = _load_aqs()
    start = pd.Timestamp(config.DATE_START)
    end = pd.Timestamp(config.DATE_END)
    cutoff = pd.Timestamp(config.TEMPORAL_CUTOFF)

    # Coverage windows (top row first). PA/AQS windows come from the actual
    # data files; the gridded sources span the study window except CAMS,
    # which is only public from CAMS_START.
    rows = [
        ("PurpleAir",  pa["date"].min(),  pa["date"].max(),  COLORS["observed"]),
        ("EPA AQS",    aqs["date"].min(), aqs["date"].max(), "#777777"),
        ("CAMS",       pd.Timestamp(CAMS_START), end,        COLORS["prior_cams"]),
        ("GEOS-CF",    start, end,                           COLORS["prior_geoscf"]),
        ("MERRA-2",    start, end,                           COLORS["prior_merra2"]),
        ("HMS smoke",  start, end,                           "#aaaaaa"),
    ]

    width = FIG_W["double" if mode == "paper" else "poster"]
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(width, width * 0.62), sharex=True,
        gridspec_kw={"height_ratios": [0.75, 1.0], "hspace": 0.28})

    # (a) coverage windows -------------------------------------------------
    ax_a.axvspan(start, cutoff, color=COLORS["train_shade"], zorder=0)
    ys = np.arange(len(rows))[::-1]
    for y, (label, s, e, color) in zip(ys, rows):
        ax_a.barh(y, (e - s).days, left=mdates.date2num(s), height=0.55,
                  color=color, edgecolor="none", zorder=3)
    ax_a.set_yticks(ys)
    ax_a.set_yticklabels([r[0] for r in rows])
    ax_a.axvline(mdates.date2num(cutoff), color=_INK, linestyle="--",
                 linewidth=lw * 0.7, zorder=4)
    ax_a.set_ylim(-0.7, len(rows) - 0.3)
    ax_a.set_ylabel("Source")
    ax_a.grid(axis="y", visible=False)
    # Direct-label the one notable late start (CAMS).
    ax_a.annotate(CAMS_START, xy=(mdates.date2num(pd.Timestamp(CAMS_START)),
                                  ys[2] + 0.35),
                  xytext=(2 * sc, 2 * sc), textcoords="offset points",
                  color=_NEUTRAL, fontsize=plt.rcParams["xtick.labelsize"])
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=COLORS["train_shade"],
                  edgecolor="none", label="training window"),
        Line2D([], [], color=_INK, linestyle="--", linewidth=lw * 0.7,
               label=f"temporal holdout cutoff ({config.TEMPORAL_CUTOFF})"),
    ]
    # Legend above the panel: every row inside the axes is occupied by bars.
    ax_a.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0),
                ncol=2, frameon=False, borderaxespad=0.0,
                handlelength=1.6, columnspacing=1.2)
    _panel_tag(ax_a, "a")

    # (b) active sensors per month ----------------------------------------
    months = pa["date"].values.astype("datetime64[M]")
    active = (pd.DataFrame({"month": months, "sensor_id": pa["sensor_id"]})
                .groupby("month")["sensor_id"].nunique())
    mdates_x = pd.to_datetime(active.index)
    ax_b.bar(mdates_x, active.values, width=22, align="center",
             color=COLORS["observed"], edgecolor="none", zorder=3)
    ax_b.axvline(mdates.date2num(cutoff), color=_INK, linestyle="--",
                 linewidth=lw * 0.7, zorder=4)
    ax_b.set_ylabel("Active sensors (count / month)")
    ax_b.set_xlabel("Date")
    ax_b.grid(axis="x", visible=False)
    first_n, last_n = int(active.iloc[0]), int(active.iloc[-1])
    peak_n = int(active.max())
    ax_b.text(0.02, 0.96,
              (f"{first_n} sensors in {str(active.index[0])[:7]} → "
               f"{peak_n} at peak;\nonly sensors reaching the assembled\n"
               f"dataset appear, so early coverage\nis survivorship-biased"),
              transform=ax_b.transAxes, va="top", ha="left", color=_NEUTRAL,
              fontsize=plt.rcParams["xtick.labelsize"])
    _panel_tag(ax_b, "b")

    ax_b.set_xlim(start - pd.Timedelta(days=45), end + pd.Timedelta(days=45))
    ax_b.xaxis.set_major_locator(mdates.YearLocator())
    ax_b.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    paths = save_fig(fig, name, mode, preview=False)
    plt.close(fig)
    return paths


# ── F07: Barkjohn correction ────────────────────────────────────────────────

def fig_barkjohn_correction(mode="paper", n_sample=50_000, seed=42):
    """(a) raw vs corrected hexbin with 1:1; (b) correction vs RH, IQR band."""
    name = "F07_barkjohn_correction"
    if not _require(PA_PARQUET, name):
        return None
    set_style(mode)
    sc, lw = _scale(mode), _lw(mode)

    pa = _load_pa()
    ok = pa["pm25"].notna() & pa["humidity"].notna()
    df = pa.loc[ok, ["pm25", "humidity"]]
    if len(df) > n_sample:
        df = df.sample(n=n_sample, random_state=seed)
    raw = df["pm25"].to_numpy(dtype=np.float64)
    rh = df["humidity"].to_numpy(dtype=np.float64)
    corrected = barkjohn_correct(raw, rh)
    delta = corrected - raw

    width = FIG_W["double" if mode == "paper" else "poster"]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(width, width * 0.44))
    fig.subplots_adjust(wspace=0.42 if mode == "paper" else 0.60)

    # (a) raw ATM vs corrected density ------------------------------------
    hi = float(np.percentile(raw, 99.9))
    hb = ax_a.hexbin(raw, corrected, gridsize=55, cmap=SEQUENTIAL_CMAP,
                     bins="log", mincnt=1, extent=(0, hi, 0, hi),
                     linewidths=0.1)
    ax_a.plot([0, hi], [0, hi], linestyle="--", color=_NEUTRAL,
              linewidth=lw * 0.7, zorder=4)
    # Direct label beside the line, in the empty triangle above it.
    ax_a.text(hi * 0.38, hi * 0.45, "1:1", color=_NEUTRAL, ha="right",
              va="bottom", fontsize=plt.rcParams["xtick.labelsize"])
    ax_a.set_xlim(0, hi)
    ax_a.set_ylim(0, hi)
    ax_a.set_xlabel("Raw PurpleAir ATM PM2.5 (ug/m3)")
    ax_a.set_ylabel("Barkjohn-corrected PM2.5 (ug/m3)")
    # Short lines so the note fits the empty triangle above the 1:1 line.
    ax_a.text(0.03, 0.97,
              (f"corrected = {BARKJOHN_SLOPE}·raw\n"
               f"    − {abs(BARKJOHN_RH_COEF)}·RH + {BARKJOHN_INTERCEPT}\n"
               f"n = {len(df):,} sensor-days\n(sampled)"),
              transform=ax_a.transAxes, va="top", ha="left", color=_INK,
              fontsize=plt.rcParams["xtick.labelsize"])
    cbar = fig.colorbar(hb, ax=ax_a, shrink=0.9, pad=0.02)
    cbar.set_label("Sensor-days per hex")
    cbar.outline.set_visible(False)
    _panel_tag(ax_a, "a")

    # (b) correction magnitude vs RH --------------------------------------
    edges = np.arange(0.0, 102.5, 2.5)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(rh, edges) - 1
    mean = np.full(centers.shape, np.nan)
    q25 = np.full(centers.shape, np.nan)
    q75 = np.full(centers.shape, np.nan)
    for i in range(len(centers)):
        vals = delta[idx == i]
        if len(vals) >= 30:                      # thin bins are unstable
            mean[i] = vals.mean()
            q25[i], q75[i] = np.percentile(vals, [25, 75])
    good = np.isfinite(mean)
    ax_b.axhline(0.0, color="#bbbbbb", linestyle=":", linewidth=lw * 0.5,
                 zorder=1)
    ax_b.fill_between(centers[good], q25[good], q75[good], color="#999999",
                      alpha=0.32, linewidth=0, zorder=2,
                      label="IQR (25–75%)")
    ax_b.plot(centers[good], mean[good], color=COLORS["observed"],
              linewidth=lw, zorder=3, label="binned mean")
    ax_b.set_xlim(0, 100)
    ax_b.set_xlabel("Relative humidity (%)")
    ax_b.set_ylabel("Correction, corrected − raw (ug/m3)")
    leg = ax_b.legend(loc="upper right", frameon=True, framealpha=0.9,
                      edgecolor="#cccccc")
    for txt in leg.get_texts():
        txt.set_color(_INK)
    _panel_tag(ax_b, "b")

    paths = save_fig(fig, name, mode, preview=False)
    plt.close(fig)
    return paths


# ── Driver ──────────────────────────────────────────────────────────────────

FIGURES = (fig_study_domain, fig_data_coverage, fig_barkjohn_correction)


def render_all(modes=("paper", "poster")):
    """Render every data figure in every mode; returns written paths."""
    written = []
    for mode in modes:
        for fn in FIGURES:
            paths = fn(mode=mode)
            if paths:
                written.extend(paths)
                for p in paths:
                    print(f"[wrote] {p}")
    return written


if __name__ == "__main__":
    render_all()
