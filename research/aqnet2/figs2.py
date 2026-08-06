"""AQNet v2 publication figure system.

Reads ONLY the validate2/calibrate/compose artifacts (metrics_*.json,
calibration_report.json, gates.json, uq_params.json, oof npzs) and renders
the v2 figure set under artifacts/v2/figures/ (PNG, 200 dpi). Every figure
degrades gracefully when its source artifact is absent — a missing input
yields a skipped figure with a printed reason, never a crash, so the module
can run mid-pipeline on partial results.

Figures:
  F01 residual-ladder architecture schematic (pure matplotlib, no data)
  F02 calibration: learned KO vs Barkjohn vs AMT — LOLO RMSE/|bias| bars
  F03 calibration by-year bias flatness (the v1 drift, closed or not)
  F04 outer-fold held-out-site R2 ladder (T0 -> T1 -> +T2 -> +T3 -> +T4)
      with site-cluster bootstrap CIs; vault point overlaid when present
  F05 admission-gate outcomes: per (tier, pattern, stratum) alpha heatmap
  F06 per-site bias/R2 map (lat/lon scatter, diverging bias colormap)
  F07 attenuation slope by year (b per year, clip band shaded)
  F08 interval coverage/width per coverage-density bin (ship window shaded)
  F09 exceedance precision/recall at both thresholds vs baseline
  F10 baseline forest plot: paired delta-R2 CIs vs composite
  F11 grouped permutation importance (portable vs interpolating)
  F12 spatial/temporal R2 decomposition per tier

Run:  python figs2.py [--only F04,F10] [--dpi 200]
"""
import argparse
import json
import os
import sys

import numpy as np

import config2

FIG_DIR = os.path.join(config2.ARTIFACTS_DIR, "figures")

# House style: colorblind-safe, print-safe.
STYLE = {
    "t0": "#8c8c8c", "t1": "#0072B2", "t2": "#009E73", "t3": "#D55E00",
    "t4": "#CC79A7", "composite": "#000000", "baseline": "#E69F00",
    "good": "#009E73", "bad": "#D55E00", "band": "#DDDDDD",
}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 9.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "font.family": "DejaVu Sans",
    })
    return plt


def _load_json(name):
    path = config2.artifact(name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        print(f"[figs2] unreadable {name}: {e}")
        return None


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, name)
    fig.savefig(out, bbox_inches="tight")
    print(f"[figs2] wrote {out}")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _skip(name, why):
    print(f"[figs2] SKIP {name} — {why}")


# ── F01 architecture schematic ──────────────────────────────────────────────

def fig01_architecture():
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.axis("off")
    rungs = [
        ("T0  debiased CTM prior", "extrapolation floor — queryable anywhere", STYLE["t0"]),
        ("T1  GPBoost skeleton", "GBM + Matern GP + unit/day random effects\nFRM-anchored two-network target", STYLE["t1"]),
        ("T2  graph-attention residual", "masked, shielded, airshed-wired\n(admission-gated)", STYLE["t2"]),
        ("T3  field residual + INR decoder", "masked multimodal pretrain, point head\n(admission-gated)", STYLE["t3"]),
        ("T4  slope recalibration", "declared rung, b clipped to [0.8, 1.25]", STYLE["t4"]),
    ]
    y = 0.88
    for title, sub, color in rungs:
        ax.add_patch(plt.Rectangle((0.06, y - 0.135), 0.62, 0.15,
                                   facecolor=color, alpha=0.14,
                                   edgecolor=color, linewidth=1.4))
        ax.text(0.08, y - 0.032, title, fontsize=10.5, fontweight="bold",
                color=color, va="top")
        ax.text(0.08, y - 0.075, sub, fontsize=8, va="top", color="#333333")
        if y < 0.88:
            ax.annotate("", xy=(0.37, y + 0.015), xytext=(0.37, y + 0.052),
                        arrowprops=dict(arrowstyle="-|>", color="#555555"))
        y -= 0.192
    ax.text(0.72, 0.86, "F_k(q) = F_{k-1}(q) + m_k(q) · α_k[p(q)] · r̂_k(q)",
            fontsize=10, style="italic")
    for i, line in enumerate([
            "frozen incumbent — coefficient exactly 1",
            "m_k ∈ {0,1}: structural zero, never fill",
            "α from cross-fit, power-calibrated admission",
            "default closed; α = 0 beyond 100 km",
            "worst reachable outcome: T1 alone"]):
        ax.text(0.72, 0.76 - 0.075 * i, "• " + line, fontsize=8.4, color="#333333")
    ax.set_title("AQNet v2 — structurally monotone residual ladder", fontsize=12)
    _save(fig, "F01_architecture.png")


# ── F02/F03 calibration ─────────────────────────────────────────────────────

def fig02_calibration_methods():
    rep = _load_json("calibration_report.json")
    if not rep or "lolo" not in rep or not rep["lolo"].get("methods"):
        return _skip("F02", "calibration_report.json absent/incomplete")
    plt = _plt()
    methods = rep["lolo"]["methods"]
    order = [m for m in ("learned", "barkjohn", "barkjohn_refit", "amt_rht") if m in methods]
    rmse = [methods[m].get("rmse") for m in order]
    bias = [abs(methods[m].get("bias") or np.nan) for m in order]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    colors = [STYLE["t1"] if m == "learned" else STYLE["baseline"] for m in order]
    for ax, vals, label in ((axes[0], rmse, "LOLO RMSE (µg/m³)"),
                            (axes[1], bias, "LOLO |bias| (µg/m³)")):
        vals = [v if v is not None else np.nan for v in vals]
        ax.bar(range(len(order)), vals, color=colors, width=0.62)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([m.replace("_", "\n") for m in order], fontsize=8)
        ax.set_ylabel(label)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    verdict = (rep.get("g0") or {}).get("verdict", "?")
    fig.suptitle(f"PA→FRM calibration, leave-one-site-out (G0: {verdict})", fontsize=11)
    _save(fig, "F02_calibration_methods.png")


def fig03_byyear_bias():
    rep = _load_json("calibration_report.json")
    by = (rep or {}).get("by_year_bias") or {}
    if not by:
        return _skip("F03", "by_year_bias absent")
    plt = _plt()
    years = sorted(by)
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    method_names = sorted({k for y in by.values() for k in y if k != "n"})
    palette = {"learned": STYLE["t1"], "barkjohn": STYLE["baseline"],
               "barkjohn_refit": "#B8860B", "amt_rht": "#666666"}
    for m in method_names:
        vals = [by[y].get(m) for y in years]
        ax.plot(years, [v if v is not None else np.nan for v in vals],
                marker="o", ms=4, label=m, color=palette.get(m, "#999999"))
    ax.axhline(0.0, color="#000000", lw=0.8)
    ax.set_ylabel("mean bias vs FRM (µg/m³)")
    ax.set_xlabel("year")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("By-year calibration bias — the v1 drift, and whether v2 closed it", fontsize=10)
    _save(fig, "F03_byyear_bias.png")


# ── F04 ladder performance ──────────────────────────────────────────────────

def fig04_ladder():
    mo = _load_json("metrics_outer.json")
    if not mo:
        return _skip("F04", "metrics_outer.json absent")
    plt = _plt()
    # validate2 layout: mo["ladder"][chain] = {"metrics": {r2,...},
    # "bootstrap_ci": {"r2": [lo, hi], ...}, "spatial_temporal": {...}}.
    lad = mo.get("ladder") or {}
    ladder_keys = [k for k in ("t0", "t1", "t1_t2", "t1_t2_t3", "composite")
                   if isinstance(lad.get(k), dict)
                   and (lad[k].get("metrics") or {}).get("r2") is not None]
    if not ladder_keys:
        return _skip("F04", "no ladder entries in metrics_outer.json")
    labels = {"t0": "T0", "t1": "T1", "t1_t2": "+T2", "t1_t2_t3": "+T3",
              "composite": "composite\n(+T4)"}
    colors = [STYLE.get(k.split("_")[-1], STYLE["composite"]) for k in ladder_keys]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for i, k in enumerate(ladder_keys):
        r2 = lad[k]["metrics"]["r2"]
        ci = (lad[k].get("bootstrap_ci") or {}).get("r2") or (None, None)
        ax.bar(i, r2, color=colors[i], width=0.6)
        if ci and ci[0] is not None:
            ax.errorbar(i, r2, yerr=[[r2 - ci[0]], [ci[1] - r2]],
                        color="#222222", capsize=3, lw=1.1)
        ax.text(i, r2 + 0.01, f"{r2:.3f}", ha="center", fontsize=8.4)
    vault = _load_json("metrics_vault.json")
    vr2 = (((vault or {}).get("vault_sites") or {}).get("metrics")
           or {}).get("r2")
    if vr2 is not None:
        ax.axhline(vr2, color=STYLE["bad"], lw=1.2, ls="--",
                   label=f"vault sites (one-shot): {vr2:.3f}")
        ax.legend(fontsize=8, frameon=False)
    ax.set_xticks(range(len(ladder_keys)))
    ax.set_xticklabels([labels[k] for k in ladder_keys])
    ax.set_ylabel("held-out-site R² (outer folds)")
    ax.set_title("Ladder performance at truly held-out AQS sites", fontsize=10.5)
    _save(fig, "F04_ladder.png")


# ── F05 gates ───────────────────────────────────────────────────────────────

def fig05_gates():
    path = config2.artifact("gates.json")
    if not os.path.exists(path):
        return _skip("F05", "gates.json absent")
    with open(path, encoding="utf-8") as f:
        gates = json.load(f)
    rows = []
    for tier, patterns in gates.items():
        for pat, strata in patterns.items():
            for stratum, entry in strata.items():
                rows.append((tier, pat, stratum, float(entry.get("alpha", 0.0)),
                             (entry.get("test") or {}).get("decision", "?")))
    if not rows:
        return _skip("F05", "gates.json empty")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 1.2))
    for i, (tier, pat, stratum, alpha, decision) in enumerate(rows):
        color = STYLE["good"] if alpha > 0 else STYLE["bad"]
        ax.barh(i, max(alpha, 0.015), color=color, alpha=0.85, height=0.62)
        ax.text(1.02, i, f"{tier} | p={pat} | s={stratum} — {decision}",
                va="center", fontsize=7.6, transform=ax.get_yaxis_transform())
    ax.set_yticks([])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("admitted α (0 = passthrough)")
    ax.set_title("Admission gates: what earned its way into the ladder", fontsize=10.5)
    ax.invert_yaxis()
    _save(fig, "F05_gates.png")


# ── F06 site map ────────────────────────────────────────────────────────────

def fig06_site_map():
    # validate2 emits per_outer_fold / per_spatial_block (no per-site
    # coordinate table lives in the artifacts), so this panel shows the
    # composite's R2 and bias across both spatial partitions instead of a
    # lat/lon scatter.
    mo = _load_json("metrics_outer.json")
    pof = (mo or {}).get("per_outer_fold") or {}
    psb = (mo or {}).get("per_spatial_block") or {}
    if not pof and not psb:
        return _skip("F06", "per_outer_fold/per_spatial_block absent")
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
    for ax, tab, ttl in ((axes[0], pof, "outer folds"),
                         (axes[1], psb, "spatial blocks")):
        keys = sorted(tab)
        r2 = [tab[k].get("r2", np.nan) for k in keys]
        bias = [tab[k].get("bias", np.nan) for k in keys]
        nsit = [tab[k].get("n_sites") for k in keys]
        x = np.arange(len(keys))
        bars = ax.bar(x, r2, width=0.6,
                      color=[STYLE["good"] if (v == v and v > 0)
                             else STYLE["bad"] for v in r2])
        for i, (b, bi, ns) in enumerate(zip(bars, bias, nsit)):
            ax.text(i, max(b.get_height(), 0) + 0.02,
                    f"bias {bi:+.1f}\n{ns} sites", ha="center", fontsize=7)
        ax.axhline(0.0, color="#000000", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace("outer_", "fold ")
                            .replace("block_", "block ") for k in keys],
                           fontsize=8)
        ax.set_title(f"held-out R² by {ttl}", fontsize=9.5)
    axes[0].set_ylabel("R²")
    fig.suptitle("Composite skill across spatial partitions", fontsize=10.5)
    _save(fig, "F06_fold_blocks.png")


# ── F07 attenuation ─────────────────────────────────────────────────────────

def fig07_attenuation():
    # attenuation lives inside by_year: attenuation_a / attenuation_b.
    mo = _load_json("metrics_outer.json")
    byy = (mo or {}).get("by_year") or {}
    att = {y: d for y, d in byy.items()
           if isinstance(d, dict) and d.get("attenuation_b") is not None}
    if not att:
        return _skip("F07", "by_year attenuation absent")
    plt = _plt()
    years = sorted(att)
    b = [att[y].get("attenuation_b") for y in years]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    lo, hi = config2.T4_SLOPE_CLIP
    ax.axhspan(lo, hi, color=STYLE["band"], alpha=0.5,
               label=f"T4 clip [{lo}, {hi}]")
    ax.axhline(1.0, color="#000000", lw=0.8)
    ax.plot(years, [v if v is not None else np.nan for v in b], marker="o",
            color=STYLE["t1"])
    ax.set_ylabel("regression slope b (y ~ a + b·pred)")
    ax.set_xlabel("year")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Attenuation by year — the failure mode T4 exists to fix", fontsize=10)
    _save(fig, "F07_attenuation.png")


# ── F08 coverage ────────────────────────────────────────────────────────────

def fig08_coverage():
    # coverage lives in metrics_outer.intervals (validate2), keyed
    # per_coverage_bin with a declared ship_window.
    mo = _load_json("metrics_outer.json")
    iv = (mo or {}).get("intervals") or {}
    cov = iv.get("per_coverage_bin")
    if not cov:
        return _skip("F08", "intervals.per_coverage_bin absent")
    win = iv.get("ship_window") or [0.88, 0.93]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    if isinstance(cov, dict):
        bins = sorted(cov)
        c = [cov[b].get("coverage") if isinstance(cov[b], dict) else cov[b] for b in bins]
        w = [cov[b].get("mean_width") if isinstance(cov[b], dict) else np.nan for b in bins]
        ax.axhspan(win[0], win[1], color=STYLE["band"], alpha=0.6,
                   label=f"ship window [{win[0]}, {win[1]}]")
        ax.axhline(0.90, color="#000000", lw=0.8, ls=":")
        ax.bar(range(len(bins)), c, color=STYLE["t1"], width=0.55)
        for i, (ci, wi) in enumerate(zip(c, w)):
            if ci is not None and np.isfinite(ci):
                lbl = f"{ci:.3f}" + (f"\nw={wi:.1f}" if wi is not None and np.isfinite(wi) else "")
                ax.text(i, ci + 0.004, lbl, ha="center", fontsize=7.6)
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels([f"bin {b}" for b in bins], fontsize=8)
        ax.set_ylim(0.75, 1.0)
    ax.set_ylabel("site-level empirical coverage (α=0.10)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Conformal coverage by coverage-density bin", fontsize=10.5)
    _save(fig, "F08_coverage.png")


# ── F09 exceedance ──────────────────────────────────────────────────────────

def fig09_exceedance():
    # exceed_model.json per_threshold[thr]["admission"] carries the paired
    # head-vs-thresholded-composite F1 test (point_new = head,
    # point_ref = thresholded composite, decision).
    ex = _load_json("exceed_model.json")
    per = (ex or {}).get("per_threshold") or {}
    rows = []
    for t in sorted(per, key=lambda s: float(s)):
        adm = (per[t] or {}).get("admission") or {}
        if adm.get("point_new") is not None:
            rows.append((t, adm.get("point_ref"), adm.get("point_new"),
                         adm.get("decision", "?"), adm.get("ci")))
    if not rows:
        return _skip("F09", "no per-threshold admission metrics in exceed_model.json")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    x = np.arange(len(rows))
    ax.bar(x - 0.17, [r[1] for r in rows], width=0.3,
           label="thresholded composite F1", color=STYLE["t1"])
    ax.bar(x + 0.17, [r[2] for r in rows], width=0.3,
           label="dedicated head F1", color=STYLE["t2"])
    for i, (t, ref, new, dec, _ci) in enumerate(rows):
        ax.text(i, max(ref or 0, new or 0) + 0.004, dec, ha="center",
                fontsize=7.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"> {r[0]} µg/m³" for r in rows])
    ax.set_ylabel("exceedance F1 (valid labels only)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Exceedance: dedicated head vs thresholded composite "
                 "(admission-tested)", fontsize=9.5)
    _save(fig, "F09_exceedance.png")


# ── F10 baselines forest ────────────────────────────────────────────────────

def fig10_baselines():
    mb = _load_json("metrics_baselines.json")
    if not mb:
        return _skip("F10", "metrics_baselines.json absent")
    plt = _plt()
    rows = []
    # validate2 layout: mb["baselines"][name]["admission_vs_composite"]
    # ["metrics"]["pooled_r2"] carries delta (composite - baseline) + ci.
    for name, m in (mb.get("baselines") or {}).items():
        if not isinstance(m, dict):
            continue
        pr = ((m.get("admission_vs_composite") or {}).get("metrics")
              or {}).get("pooled_r2") or {}
        if pr.get("delta") is not None:
            rows.append((name, pr["delta"], pr.get("ci") or (None, None)))
    if not rows:
        return _skip("F10", "no baseline entries")
    paired = [r for r in rows if len(r) == 3]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * max(len(paired), 1) + 1.4))
    for i, (name, delta, ci) in enumerate(paired):
        color = STYLE["good"] if (ci[0] is not None and ci[0] > 0) else "#777777"
        ax.plot([ci[0], ci[1]], [i, i], color=color, lw=2.0)
        ax.plot(delta, i, "o", color=color, ms=5)
        ax.text(-0.012, i, name, ha="right", va="center", fontsize=8,
                transform=ax.get_yaxis_transform())
    ax.axvline(0.0, color="#000000", lw=0.9)
    ax.set_yticks([])
    ax.set_xlabel("ΔR² (composite − baseline), site-cluster bootstrap 95% CI")
    ax.set_title("Composite vs classical baselines, paired on identical rows", fontsize=10)
    ax.invert_yaxis()
    _save(fig, "F10_baselines.png")


# ── F11 permutation importance ──────────────────────────────────────────────

def fig11_permutation():
    pr = _load_json("permutation_report.json")
    top = ((pr or {}).get("top15_single_features")
           or (pr or {}).get("top_features") or (pr or {}).get("features"))
    if not top:
        return _skip("F11", "permutation_report.json absent/empty")
    plt = _plt()
    items = sorted(top.items(), key=lambda kv: kv[1], reverse=True)[:15] \
        if isinstance(top, dict) else top[:15]
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = [STYLE["t2"] if config2.is_interp_feature(n) else STYLE["t1"] for n in names]
    fig, ax = plt.subplots(figsize=(5.8, 0.32 * len(names) + 1.2))
    ax.barh(range(len(names)), vals, color=colors, height=0.62)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("OOF permutation ΔR²")
    ax.set_title("What the skeleton actually uses (blue portable, green interpolating)",
                 fontsize=9.5)
    _save(fig, "F11_permutation.png")


# ── F12 spatial/temporal decomposition ──────────────────────────────────────

def fig12_decomposition():
    # spatial/temporal decomposition lives per ladder chain:
    # mo["ladder"][chain]["spatial_temporal"].
    mo = _load_json("metrics_outer.json")
    lad = (mo or {}).get("ladder") or {}
    dec = {k: v.get("spatial_temporal") for k, v in lad.items()
           if isinstance(v, dict) and isinstance(v.get("spatial_temporal"),
                                                 dict)}
    if not dec:
        return _skip("F12", "spatial_temporal decomposition absent")
    plt = _plt()
    tiers = list(dec.keys())
    sp = [dec[t].get("spatial_r2") for t in tiers]
    te = [dec[t].get("temporal_r2") for t in tiers]
    x = np.arange(len(tiers))
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.bar(x - 0.18, [v if v is not None else np.nan for v in sp], width=0.34,
           label="between-site (spatial) R²", color=STYLE["t2"])
    ax.bar(x + 0.18, [v if v is not None else np.nan for v in te], width=0.34,
           label="within-site (temporal) R²", color=STYLE["t1"])
    ax.set_xticks(x)
    ax.set_xticklabels(tiers, fontsize=8.5)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Where the skill lives — v1's headline hid spatial_r2 = 0.05",
                 fontsize=9.5)
    _save(fig, "F12_decomposition.png")


# ── CLI ─────────────────────────────────────────────────────────────────────

_ALL = {
    "F01": fig01_architecture, "F02": fig02_calibration_methods,
    "F03": fig03_byyear_bias, "F04": fig04_ladder, "F05": fig05_gates,
    "F06": fig06_site_map, "F07": fig07_attenuation, "F08": fig08_coverage,
    "F09": fig09_exceedance, "F10": fig10_baselines,
    "F11": fig11_permutation, "F12": fig12_decomposition,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None,
                    help="comma-separated figure ids (e.g. F04,F10)")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args(argv)
    wanted = ([f.strip().upper() for f in args.only.split(",")]
              if args.only else list(_ALL))
    print(f"[aqnet2] ── stage: figures " + "─" * 40)
    import matplotlib
    matplotlib.rcParams["savefig.dpi"] = args.dpi
    for fid in wanted:
        fn = _ALL.get(fid)
        if fn is None:
            print(f"[figs2] unknown figure id {fid}")
            continue
        try:
            fn()
        except Exception as e:  # a bad artifact never kills the whole set
            import traceback
            traceback.print_exc()
            print(f"[figs2] FAILED {fid}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
