"""AQNet v3 poster figures (F13-F17) — the WEST7-specific stories.

Reads the validated v3 artifacts (AQNET2_DOMAIN=west7) plus the shipped v2
results; the only hardcoded inputs are the cold-transfer prelim (job
11724017, v3_prelim.py) and the published-literature comparison points
(RELATED_WORK.md, each traceable to its citation there). Palette is the
figs2 house set (Okabe-Ito derived), CVD-validated; the pink slot is
unused to keep every adjacent pair above the deutan floor.

Run:  AQNET2_DOMAIN=west7 python figs3_poster.py
"""
import json
import os

import numpy as np
import pandas as pd

import config2

FIG_DIR = os.path.join(config2.ARTIFACTS_DIR, "figures")
BLUE, VERM, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#8c8c8c"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
        "font.family": "DejaVu Sans",
    })
    return plt


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, name)
    fig.savefig(out, bbox_inches="tight")
    print(f"[figs3] wrote {out}")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _load(name, base=None):
    path = os.path.join(base, name) if base else config2.artifact(name)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── F13: honest-protocol positioning ────────────────────────────────────────

def fig13_positioning():
    """Published models under their honest spatial protocols vs AQNet.

    Literature points from RELATED_WORK.md section 1 (each cited there);
    AQNet points read from the shipped run artifacts.
    """
    plt = _plt()
    mo3 = _load("metrics_outer.json")
    v3 = _load("metrics_vault.json")
    v2dir = os.path.join(config2.ROOT, "results", "v2_texas_202608")
    mo2 = _load("metrics_outer.json", v2dir)
    v2 = _load("metrics_vault.json", v2dir)

    lit = [
        ("Di 2019 vs independent\nsmoke monitors (Considine)", 0.07),
        ("STARQ 2026, Europe\nunseen stations x times", 0.242),
        ("Aurora 1.3B at stations\n(China eval, range mid)", 0.305),
        ("van Donkelaar 2024\nbuffered cluster CV (biweekly)", 0.36),
        ("ACAG V6 CNN, N. America\nspatial CV", 0.57),
        ("DeepAir 2025, California\nstation-grouped CV", 0.583),
    ]
    aq = [
        ("AQNet v2 Texas\nheld-out sites", mo2["ladder"]["composite"]["metrics"]["r2"],
         mo2["ladder"]["composite"]["bootstrap_ci"]["r2"],
         v2["vault_sites"]["metrics"]["r2"]),
        ("AQNet v3 WEST7\nheld-out sites", mo3["ladder"]["composite"]["metrics"]["r2"],
         mo3["ladder"]["composite"]["bootstrap_ci"]["r2"],
         v3["vault_sites"]["metrics"]["r2"]),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    y = 0
    for name, r2 in lit:
        ax.plot(r2, y, "o", color=GRAY, ms=8)
        ax.text(r2 + 0.015, y, f"{r2:.2f}", va="center", fontsize=8.5,
                color="#444444")
        ax.text(-0.02, y, name, ha="right", va="center", fontsize=8.2)
        y += 1
    for name, r2, ci, vault in aq:
        ax.plot([ci[0], ci[1]], [y, y], color=BLUE, lw=2.4, solid_capstyle="round")
        ax.plot(r2, y, "o", color=BLUE, ms=9, zorder=5)
        ax.plot(vault, y, "D", color=VERM, ms=8, zorder=6)
        ax.text(ci[1] + 0.015, y, f"{r2:.2f} (vault {vault:.2f})",
                va="center", fontsize=8.5, color="#1a1a1a")
        ax.text(-0.02, y, name, ha="right", va="center", fontsize=8.6,
                fontweight="bold")
        y += 1
    ax.plot([], [], "o", color=GRAY, label="published, honest spatial protocol")
    ax.plot([], [], "o", color=BLUE, label="AQNet (95% CI)")
    ax.plot([], [], "D", color=VERM, label="one-shot sealed vault")
    ax.legend(fontsize=8.2, frameon=False, loc="lower right")
    ax.set_yticks([])
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("R² at spatially held-out / independent sites")
    ax.set_title("PM2.5 models under honest validation — AQNet is the only entry\n"
                 "with a sealed vault confirming its estimate", fontsize=10.5)
    _save(fig, "F13_positioning.png")


# ── F14: retrain vs cold transfer, per state ────────────────────────────────

def fig14_transfer():
    """Per-state held-out R2: WEST7-retrained T1 vs the Texas v2 model
    applied cold (prelim job 11724017). Retrained bars recomputed from
    oof_tier1 so the figure tracks the shipped artifact."""
    plt = _plt()
    t1 = np.load(config2.artifact("oof_tier1.npz"), allow_pickle=True)
    fr = pd.read_parquet(config2.artifact("frame_truth.parquet"),
                         columns=["unit_type", "unit_id", "y"])
    y = fr.y.to_numpy()
    aqs = (fr.unit_type == "aqs").to_numpy()
    st = fr.unit_id.str[4:6].to_numpy()
    o1 = t1["oof"]

    def r2_of(code):
        m = aqs & (st == code) & np.isfinite(o1) & np.isfinite(y)
        return 1 - np.sum((y[m] - o1[m]) ** 2) / np.sum((y[m] - y[m].mean()) ** 2)

    cold = {"06": 0.0785, "53": 0.0329, "08": 0.0743, "49": 0.13,
            "32": -0.0064, "04": -0.0217}          # v3_prelim.py, job 11724017
    states = [("06", "CA"), ("32", "NV"), ("49", "UT"), ("08", "CO"),
              ("53", "WA"), ("04", "AZ")]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    x = np.arange(len(states))
    re = [r2_of(c) for c, _ in states]
    co = [cold[c] for c, _ in states]
    ax.bar(x - 0.19, co, width=0.34, color=GRAY,
           label="Texas model, cold transfer")
    ax.bar(x + 0.19, re, width=0.34, color=BLUE,
           label="WEST7 retrained (held-out sites)")
    for i, (c, r) in enumerate(zip(co, re)):
        ax.text(i - 0.19, max(c, 0) + 0.008, f"{c:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.19, max(r, 0) + 0.008, f"{r:.2f}", ha="center", fontsize=8)
    ax.axhline(0, color="#000000", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in states])
    ax.set_ylabel("held-out R²")
    ax.legend(fontsize=8.4, frameon=False, loc="upper right")
    ax.set_title("Regional models do not travel: in-domain training vs transfer",
                 fontsize=10.5)
    _save(fig, "F14_transfer.png")


# ── F15: the deep-tier verdict forest plot ──────────────────────────────────

def fig15_verdict():
    """Admission deltas with CIs from gates.json — the headline finding."""
    plt = _plt()
    g = _load("gates.json")
    rows = []
    t2p1 = g["t2"]["1"]["__global__"]["test"]["metrics"]["pooled_r2"]
    rows.append(("Graph tier, sensor-dense regime\n(Texas: monitors + PurpleAir)",
                 t2p1["delta"], t2p1["ci"], "significant — refused on\nspatial-axis certification"))
    t2p2 = g["t2"]["2"]["__global__"]["test"]["metrics"]["pooled_r2"]
    rows.append(("Graph tier, monitors-only regime\n(six new states)",
                 t2p2["delta"], t2p2["ci"], "null"))
    t3 = g["t3"]["0"]["__global__"]["test"]["metrics"]["pooled_r2"]
    rows.append(("Neural field tier (all regimes)",
                 t3["delta"], t3["ci"], "null"))
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    for i, (name, d, ci, verdict) in enumerate(rows):
        sig = ci[0] is not None and ci[0] > 0
        color = GREEN if sig else GRAY
        ax.plot([ci[0], ci[1]], [i, i], color=color, lw=2.6,
                solid_capstyle="round")
        ax.plot(d, i, "o", color=color, ms=9, zorder=5)
        ax.text(-0.085, i, name, ha="right", va="center", fontsize=8.6)
        ax.text(max(ci[1], d) + 0.004, i, verdict, va="center", fontsize=8,
                color="#444444")
    ax.axvline(0, color="#000000", lw=1.0)
    ax.set_yticks([])
    ax.set_xlim(-0.08, 0.12)
    ax.invert_yaxis()
    ax.set_xlabel("Δ pooled R² vs incumbent (95% cluster-bootstrap CI)")
    ax.set_title("What deep learning adds, measured honestly:\n"
                 "real, small, and sensor-density-dependent", fontsize=10.5)
    _save(fig, "F15_verdict.png")


# ── F16: the residual structure deep tiers feed on ─────────────────────────

def fig16_residual_autocorr():
    """Neighbor correlation of T1 held-out residuals by distance band.
    Recomputed from oof_tier1 (300 sampled days, seed 42 — matches the
    session diagnostic)."""
    plt = _plt()
    t1 = np.load(config2.artifact("oof_tier1.npz"), allow_pickle=True)
    fr = pd.read_parquet(config2.artifact("frame_truth.parquet"),
                         columns=["unit_type", "date", "lat", "lon", "y"])
    m = (fr.unit_type == "aqs").to_numpy() & np.isfinite(t1["oof"]) \
        & np.isfinite(fr.y.to_numpy())
    df = fr[m].copy()
    df["r"] = (fr.y.to_numpy() - t1["oof"])[m]
    rng = np.random.default_rng(42)
    days = rng.choice(df.date.unique(), size=300, replace=False)
    bands = [(0, 50), (50, 150), (150, 400)]
    acc = {b: [] for b in bands}
    for d in days:
        dd = df[df.date == d]
        if len(dd) < 20:
            continue
        lat, lon, r = (dd[c].to_numpy() for c in ("lat", "lon", "r"))
        km = 111.0
        dist = np.sqrt(((lat[:, None] - lat[None, :]) * km) ** 2
                       + ((lon[:, None] - lon[None, :]) * km * 0.84) ** 2)
        upper = np.arange(len(r))[:, None] < np.arange(len(r))[None, :]
        for lo, hi in bands:
            i, j = np.where((dist > lo) & (dist <= hi) & upper)
            if len(i) > 5:
                acc[(lo, hi)].append(np.corrcoef(r[i], r[j])[0, 1])
    vals = [float(np.mean(acc[b])) for b in bands]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x = np.arange(len(bands))
    ax.bar(x, vals, width=0.55, color=[VERM, VERM, GRAY])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["< 50 km", "50–150 km", "150–400 km"])
    ax.set_ylabel("neighbor correlation of held-out residuals")
    ax.set_title("The skeleton's errors are spatially organized at exactly\n"
                 "the graph tier's edge length scales", fontsize=10.5)
    ax.axvspan(-0.5, 1.5, color=VERM, alpha=0.06)
    ax.text(0.5, max(vals) * 0.92, "graph edges span here", ha="center",
            fontsize=8.2, color=VERM)
    _save(fig, "F16_residual_autocorr.png")


# ── F17: the honesty ladder, v2 and v3 ─────────────────────────────────────

def fig17_ladder():
    """Pooled-LOSO diagnostic vs strict spatial vs vault, both runs."""
    plt = _plt()
    mo3 = _load("metrics_outer.json")
    v3 = _load("metrics_vault.json")
    v2dir = os.path.join(config2.ROOT, "results", "v2_texas_202608")
    mo2 = _load("metrics_outer.json", v2dir)
    v2 = _load("metrics_vault.json", v2dir)
    rungs = ["pooled LOSO\n(diagnostic)", "spatially blocked\nheld-out sites",
             "sealed vault\n(one shot)"]
    vals2 = [mo2["pooled_loso_diagnostic"]["metrics"]["r2"],
             mo2["ladder"]["composite"]["metrics"]["r2"],
             v2["vault_sites"]["metrics"]["r2"]]
    vals3 = [mo3["pooled_loso_diagnostic"]["metrics"]["r2"],
             mo3["ladder"]["composite"]["metrics"]["r2"],
             v3["vault_sites"]["metrics"]["r2"]]
    x = np.arange(len(rungs))
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.plot(x, vals2, "o-", color=GREEN, lw=2, ms=9, label="v2 Texas")
    ax.plot(x, vals3, "o-", color=BLUE, lw=2, ms=9, label="v3 WEST7")
    for xs, vs, c in ((x, vals2, GREEN), (x, vals3, BLUE)):
        for xi, vi in zip(xs, vs):
            ax.text(xi + 0.06, vi, f"{vi:.2f}", fontsize=8.6, color=c,
                    va="center")
    ax.set_xticks(x)
    ax.set_xticklabels(rungs, fontsize=8.6)
    ax.set_ylabel("R²")
    ax.set_ylim(0, 0.65)
    ax.legend(fontsize=8.6, frameon=False)
    ax.set_title("The validation ladder: the honest number survives its own\n"
                 "sealed test in both domains", fontsize=10.5)
    _save(fig, "F17_ladder.png")


if __name__ == "__main__":
    print(f"[figs3] domain={config2.DOMAIN} artifacts={config2.ARTIFACTS_DIR}")
    fig13_positioning()
    fig14_transfer()
    fig15_verdict()
    fig16_residual_autocorr()
    fig17_ladder()
