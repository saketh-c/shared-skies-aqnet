"""Shared figure style for AQNet publication figures.

This module is the single source of truth for how every AQNet figure looks:
the fixed entity->color mapping, the sequential/diverging colormaps, the
rcParams for paper and poster modes, the journal column widths, and the one
sanctioned save path (which enforces the quick-mode preview watermark).

Every figure script imports this module FIRST — importing it locks the Agg
backend before pyplot is touched — then calls::

    import fig_style
    fig_style.set_style("paper")
    fig, ax = plt.subplots(figsize=(fig_style.FIG_W["single"], 2.6))
    ...
    fig_style.save_fig(fig, "f08_main_results_forest", mode="paper",
                       preview=True)   # preview=True for quick-mode artifacts

Rules encoded here (see FIGURES.md for the full standard):
  * Color follows the entity, never the position: series colors come from
    COLORS by name and are never cycled or repainted.
  * Sequential surfaces use viridis; diverging (bias/residual) maps use
    RdBu_r centered on zero. Never rainbow/jet.
  * Any render produced from quick-mode artifacts MUST go through
    ``save_fig(..., preview=True)`` so it lands in figures/preview_quick/
    with a diagonal "QUICK-MODE PREVIEW — NOT RESULTS" watermark.
"""

import math
import os

import matplotlib

# Lock the non-interactive backend before anyone imports pyplot.
matplotlib.use("Agg", force=True)

__all__ = [
    "COLORS", "SEQUENTIAL_CMAP", "DIVERGING_CMAP", "FIG_W",
    "FIGURES_DIR", "PREVIEW_DIR", "WATERMARK_TEXT",
    "set_style", "save_fig", "annotate_metrics",
]

# ── Paths ───────────────────────────────────────────────────────────────────

AQNET_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(AQNET_DIR, "figures")
PREVIEW_DIR = os.path.join(FIGURES_DIR, "preview_quick")

# ── Palette (fixed ENTITY -> hex; color follows the entity, never position) ─
#
# Okabe–Ito derived; validated for CVD separation. Identity is never carried
# by color alone: multi-series plots also use distinct markers/linestyles and
# a legend (>= 2 series) or direct labels.

COLORS = {
    # Models (the three tiers)
    "tier1": "#0072B2",             # blue
    "tier2": "#E69F00",             # orange
    "tier3": "#009E73",             # green
    # Classical interpolation baselines (gray ramp, darker = stronger method)
    "baseline_nearest": "#9a9a9a",
    "baseline_idw": "#7a7a7a",
    "baseline_kriging": "#5a5a5a",
    # CTM priors
    "prior_cams": "#56B4E9",        # sky blue
    "prior_geoscf": "#D55E00",      # vermillion
    "prior_merra2": "#CC79A7",      # pink (last slot)
    # Non-series inks
    "observed": "#333333",          # observations / 1:1 lines / reference data
    "train_shade": "#e8e8e8",       # shaded train period in time-series panels
    # Semantic accents, used per-figure where tiers are not co-plotted
    "positive": "#009E73",          # favorable direction (e.g. ablation gains)
    "neutral": "#555555",           # neutral annotation ink
}

# Colormaps: sequential for PM2.5 magnitude surfaces (perceptually uniform,
# CVD-safe); diverging for bias/residual maps centered on zero with a
# near-white neutral midpoint.
SEQUENTIAL_CMAP = "viridis"
DIVERGING_CMAP = "RdBu_r"

# Figure widths in inches (journal column widths + poster panel).
FIG_W = {"single": 3.5, "double": 7.2, "poster": 12.0}

# Watermark stamped on every quick-mode preview render.
WATERMARK_TEXT = "QUICK-MODE PREVIEW — NOT RESULTS"

# ── Mode parameters ─────────────────────────────────────────────────────────
# paper: base 8.5pt (title 10 / label 9 / tick 8 / legend 8), thin 1.6pt lines.
# poster: base 18pt (~2.1x scale), 3pt lines.

_MODE_PARAMS = {
    "paper": {
        "base": 8.5, "title": 10, "label": 9, "tick": 8, "legend": 8,
        "line": 1.6, "marker": 5.0, "axes_lw": 0.8, "grid_lw": 0.5,
        "capsize": 2.5,
    },
    "poster": {
        "base": 18, "title": 21, "label": 19, "tick": 17, "legend": 17,
        "line": 3.0, "marker": 10.5, "axes_lw": 1.6, "grid_lw": 1.0,
        "capsize": 4.0,
    },
}


def set_style(mode="paper"):
    """Apply the shared matplotlib rcParams for ``mode`` ("paper"|"poster").

    Sans-serif (DejaVu Sans), top/right spines off, light grid drawn behind
    the data (alpha 0.25), 300 dpi raster output, TrueType fonts embedded in
    PDF/PS so text stays editable in vector editors.
    """
    if mode not in _MODE_PARAMS:
        raise ValueError(f"mode must be 'paper' or 'poster', got {mode!r}")
    p = _MODE_PARAMS[mode]
    matplotlib.rcParams.update({
        # Type
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": p["base"],
        "axes.titlesize": p["title"],
        "figure.titlesize": p["title"],
        "axes.labelsize": p["label"],
        "xtick.labelsize": p["tick"],
        "ytick.labelsize": p["tick"],
        "legend.fontsize": p["legend"],
        "axes.unicode_minus": True,
        # Frame: only left/bottom spines
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": p["axes_lw"],
        "xtick.major.width": p["axes_lw"],
        "ytick.major.width": p["axes_lw"],
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Grid: recessive, behind the data
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.25,
        "grid.linewidth": p["grid_lw"],
        "grid.color": "#b0b0b0",
        # Marks: thin lines, visible markers, capped error bars
        "lines.linewidth": p["line"],
        "lines.markersize": p["marker"],
        "errorbar.capsize": p["capsize"],
        "legend.frameon": False,
        # Output
        "savefig.dpi": 300,
        "figure.dpi": 110,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def _stamp_watermark(fig):
    """Diagonal translucent quick-mode watermark across the whole figure."""
    if getattr(fig, "_quickmode_watermarked", False):
        return
    w = fig.get_figwidth()
    h = fig.get_figheight()
    diag = math.hypot(w, h)
    fig.text(
        0.5, 0.5, WATERMARK_TEXT,
        transform=fig.transFigure,
        rotation=math.degrees(math.atan2(h, w)),
        ha="center", va="center",
        fontsize=max(9.0, 3.3 * diag),
        fontweight="bold",
        color="#888888", alpha=0.22,
        zorder=1000,
    )
    fig._quickmode_watermarked = True


def save_fig(fig, name, mode="paper", preview=False):
    """Save ``fig`` under the shared conventions; return the written paths.

    preview=False  -> <aqnet>/figures/<name>.{pdf,png,svg}   (png at 300 dpi)
    preview=True   -> <aqnet>/figures/preview_quick/... and the figure is
                      stamped with the diagonal quick-mode watermark BEFORE
                      saving. Every render sourced from quick-mode artifacts
                      must use preview=True.

    mode="poster" appends ``_poster`` to the name. Always bbox_inches="tight".
    """
    if mode not in _MODE_PARAMS:
        raise ValueError(f"mode must be 'paper' or 'poster', got {mode!r}")
    if mode == "poster":
        name = f"{name}_poster"

    out_dir = PREVIEW_DIR if preview else FIGURES_DIR
    os.makedirs(out_dir, exist_ok=True)

    if preview:
        _stamp_watermark(fig)

    paths = []
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths


def annotate_metrics(ax, metrics):
    """Standard metrics text box: top-left, small, neutral ink.

    ``metrics`` is an ordered mapping of label -> value (e.g.
    {"R2": 0.71, "RMSE": 3.2, "n": 86316}). Floats are shown to three
    decimals; everything else is str()'d. Returns the Text artist.
    """
    lines = []
    for key, val in metrics.items():
        if isinstance(val, float):
            text = f"{val:,.0f}" if abs(val) >= 1000 else f"{val:.3f}"
        else:
            text = f"{val}"
        lines.append(f"{key} = {text}")
    return ax.text(
        0.02, 0.98, "\n".join(lines),
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=matplotlib.rcParams["font.size"] * 0.92,
        color="#333333",
        linespacing=1.35,
        zorder=6,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "#cccccc",
            "linewidth": 0.5,
        },
    )
