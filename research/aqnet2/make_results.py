"""Assemble the committable results package for an AQNet v2 run.

Reads ONLY the run artifacts under artifacts/v2 (nothing hand-entered — the
definitive-run discipline), then:

  1. copies the small committable artifact set into results/<run-name>/
     (metrics_*.json, SUMMARY.md, calibration_report.json, gates.json,
     t4_params.json, power_analysis.json, audit_report.json, uq_params.json,
     exceed_model.json, parity/monotone reports, versions.txt, git_sha.txt),
  2. renders the figs2 figure set and copies the PNGs alongside,
  3. writes RESULTS.md — the model card — with every number read from the
     artifacts at assembly time; absent metrics are written as "not
     computed", never guessed.

Run (from anywhere, after the validate stage):
    python research/aqnet2/make_results.py --run-name v2_texas_202608
"""
import argparse
import json
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config2  # noqa: E402

COMMIT_SET = [
    "SUMMARY.md", "metrics_outer.json", "metrics_vault.json",
    "metrics_baselines.json", "metrics_temporal.json", "metrics_strata.json",
    "permutation_report.json", "monotone_report.json", "parity_report.json",
    "calibration_report.json", "gates.json", "t4_params.json",
    "power_analysis.json", "audit_report.json", "uq_params.json",
    "exceed_model.json", "priors_report.json", "data_pa_decision.json",
    "leakage_study.json", "export_manifest.json",
    "versions.txt", "git_sha.txt",
]


def _read(name):
    p = config2.artifact(name)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _fmt(v, nd=4):
    if v is None:
        return "not computed"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int,)):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _g(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return default if cur is None else cur


def build_results_md(run_name):
    mo = _read("metrics_outer.json") or {}
    mv = _read("metrics_vault.json") or {}
    mb = _read("metrics_baselines.json") or {}
    cal = _read("calibration_report.json") or {}
    t1w = {}
    try:
        import numpy as np
        z = np.load(config2.artifact("oof_tier1.npz"), allow_pickle=False)
        t1w = json.loads(str(z["weights_json"]))
    except Exception:
        pass
    gates = _read("gates.json") or {}
    pa = _read("power_analysis.json") or {}
    uq = _read("uq_params.json") or {}
    ex = _read("exceed_model.json") or {}
    par = _read("parity_report.json") or {}
    mono = _read("monotone_report.json") or {}
    sha = None
    p = config2.artifact("git_sha.txt")
    if os.path.exists(p):
        sha = open(p).read().strip()[:12]

    n_open = sum(1 for tier in gates.values() if isinstance(tier, dict)
                 for pat in tier.values() if isinstance(pat, dict)
                 for s, e in pat.items()
                 if s != "__global__" and isinstance(e, dict)
                 and float(e.get("alpha", 0)) > 0)

    lines = []
    a = lines.append
    a(f"# AQNet {config2.DOMAIN_SPEC['artifacts']} — {run_name} — "
      f"Model Card & Results")
    a("")
    a(f"Every number in this document was computed by the pipeline and read "
      f"from `research/aqnet2/artifacts/{config2.DOMAIN_SPEC['artifacts']}` "
      f"at assembly time by "
      f"`make_results.py`; nothing is hand-entered. Absent metrics read "
      f"\"not computed\".")
    a("")
    a(f"- **Pipeline git sha:** {sha or 'not recorded'}")
    a(f"- **Target:** EPA AQS FRM/FEM anchor + calibrated PurpleAir "
      f"covariate stream (v2 inversion of the v1 target)")
    a(f"- **Calibration (G0):** {_g(cal, 'g0', 'verdict', default='not computed')} "
      f"— production form `{_g(cal, 'production_form', default='?')}`")
    a(f"- **T1 decision:** {_g(t1w, 'decision', default='not computed')} — "
      f"winner `{_g(t1w, 'winner', default='?')}` "
      f"(baseline `{_g(t1w, 'loser', default='?')}`)")
    a(f"- **Admission gates open:** {n_open} (tier, pattern, stratum) "
      f"entries across {len(gates)} tiers")
    a(f"- **Serving parity:** {_g(par, 'passed', default='not computed')} "
      f"({_g(par, 'gates_reapplication', 'chain_matches', default='?')})")
    a(f"- **Monotone audit:** "
      f"{_g(mono, 'passed', default=_g(mono, 'note', default='not computed'))}")
    a("")

    a("## P1 — Held-out AQS sites (outer folds)")
    a("")
    a("| Arm | R² | RMSE | MAE | Bias |")
    a("|---|---|---|---|---|")
    ladder = _g(mo, "ladder", default=mo)
    for key, label in (("t0", "T0 prior"), ("t1", "T1 skeleton"),
                       ("t1_t2", "+T2 graph"), ("t1_t2_t3", "+T3 field"),
                       ("composite", "Composite (+T4)")):
        m = _g(ladder, key) if isinstance(ladder, dict) else None
        if isinstance(m, dict):
            a(f"| {label} | {_fmt(m.get('r2'))} | {_fmt(m.get('rmse'), 3)} "
              f"| {_fmt(m.get('mae'), 3)} | {_fmt(m.get('bias'), 3)} |")
        else:
            a(f"| {label} | not computed | | | |")
    a("")

    by_year = _g(mo, "by_year_bias") or _g(mo, "by_year")
    if isinstance(by_year, dict) and by_year:
        a("### By-year bias (flatness is the ship criterion the v1 drift "
          "failed)")
        a("")
        a("| Year | Bias | Attenuation b |")
        a("|---|---|---|")
        att = _g(mo, "attenuation_by_year") or {}
        for yr in sorted(by_year):
            b = by_year[yr]
            bias = b.get("bias") if isinstance(b, dict) else b
            slope = _g(att, yr, "b")
            a(f"| {yr} | {_fmt(bias, 3)} | {_fmt(slope, 3)} |")
        a("")

    a("## P2 — Between-site skill (the v1 failure axis)")
    a("")
    p2 = _g(mo, "p2") or {}
    wn = _g(p2, "with_network") or {}
    bs = _g(p2, "bare_site") or {}
    a(f"- With-network between-site R²: "
      f"{_fmt(_g(wn, 'composite', 'between_site_r2') or _g(wn, 'between_site_r2'))}"
      f" ; Spearman ρ: "
      f"{_fmt(_g(wn, 'composite', 'spearman_rho') or _g(wn, 'spearman_rho'))}")
    a(f"- Bare-site arm (PA within 5 km excluded; T0+T1 core): R² "
      f"{_fmt(_g(bs, 'between_site_r2'))} ; ρ "
      f"{_fmt(_g(bs, 'spearman_rho'))} "
      f"({_g(bs, 'note', default='')})")
    a("")

    a("## P3 — Exceedance (v1 recall was 0.068 → 0.000)")
    a("")
    p3 = _g(mo, "exceedance") or {}
    for thr_key in sorted(k for k in p3 if str(k).startswith("thr_")):
        e = p3[thr_key]
        if isinstance(e, dict):
            a(f"- {thr_key}: precision {_fmt(e.get('precision'), 3)}, "
              f"recall {_fmt(e.get('recall'), 3)}, F1 {_fmt(e.get('f1'), 3)} "
              f"(n={_fmt(e.get('n_scored'))}, source {e.get('tau_source')})")
    if not any(str(k).startswith("thr_") for k in p3):
        a("- not computed")
    a("")

    a("## P4 — Conformal intervals")
    a("")
    p4 = _g(mo, "intervals") or {}
    pooled = _g(p4, "pooled_site_level") or {}
    a(f"- Site-level coverage: {_fmt(pooled.get('coverage'))} "
      f"(ship window {_g(p4, 'ship_window', default='[0.88, 0.93]')}, "
      f"verdict **{_g(p4, 'ship_verdict', default='not computed')}**), "
      f"mean width {_fmt(pooled.get('mean_width'), 2)} µg/m³ over "
      f"{_fmt(pooled.get('n_sites'))} sites")
    a(f"- Quantile-band lineage vs live composite: "
      f"{_g(p4, 'lineage_ok', default='not checked')}")
    a("")

    a("## Vault (one-shot second sample)")
    a("")
    if mv.get("status") == "opened" or _g(mv, "composite"):
        c = _g(mv, "composite") or {}
        a(f"- Composite at vault sites/period: R² {_fmt(c.get('r2'))}, "
          f"RMSE {_fmt(c.get('rmse'), 3)}, bias {_fmt(c.get('bias'), 3)} "
          f"(n={_fmt(c.get('n'))})")
    else:
        a(f"- {_g(mv, 'status', default=_g(mv, 'note', default='not opened'))}")
    a("")

    a("## Baselines (paired on identical rows)")
    a("")
    a("| Baseline | R² | ΔR² vs composite [95% CI] |")
    a("|---|---|---|")
    if isinstance(mb, dict):
        for name in sorted(mb):
            m = mb[name]
            if not isinstance(m, dict):
                continue
            d = _g(m, "paired_delta_r2") or {}
            ci = d.get("ci95") or [None, None]
            a(f"| {name} | {_fmt(m.get('r2'))} | "
              f"{_fmt(d.get('delta_r2'))} "
              f"[{_fmt(ci[0])}, {_fmt(ci[1])}] |")
    a("")

    a("## Audit-stage registrations")
    a("")
    a(f"- Margins: {json.dumps(_g(pa, 'margins', default={}))} "
      f"(source: {_g(pa, 'margins_source', default='?')})")
    a(f"- T2 kill-switch (advisory): kriging_passes = "
      f"{_g(pa, 't2_killswitch', 'kriging_passes', default='not computed')}")
    a(f"- Conformal δ by coverage bin: "
      f"{json.dumps(_g(uq, 'delta_by_bin', default='not computed'))[:200]}")
    thr_frozen = {k: _g(v, 'frozen', default=None) if isinstance(v, dict)
                  else None for k, v in (_g(ex, "per_threshold") or {}).items()}
    if any(v is not None for v in thr_frozen.values()):
        a(f"- Exceedance frozen thresholds: {json.dumps(thr_frozen)[:200]}")
    a("")
    a("## Known limitations of this run (recorded, not hidden)")
    a("")
    a("- cf_1 refetch skipped (DESIGN §4 fallback): channel-reconstructed "
      "rows are excluded from exceedance labels and carry inflated "
      "calibration variance.")
    a("- GPBoost candidate A hit a NaN-likelihood instability (suspected "
      "unit-RE/GP identifiability at repeated coordinates) — T1 is the "
      "pre-registered candidate-B escape; decision + exception recorded in "
      "oof_tier1.npz weights_json.")
    a("- NLCD impervious fractions unavailable (no public endpoint "
      "answered); NEI 2023 not yet published (2020 year-keyed values used).")
    a("- Temporal-holdout claims deferred to the temporally-pure pretrain "
      "variants (ABLATION_PLAN A11), per DESIGN §2.")
    a("- Admission gates are decided on calibrated-PA unit clusters (AQS "
      "rows are outer-held everywhere): gate deltas are a SAFETY mechanism "
      "and are never quoted as FRM-scale effect sizes; every 'tier X "
      "helps' claim is sourced from outer folds + vault only.")
    a("- Admission's exceedance-F1 axis is inert under the cf_1 skip "
      "(reconstructed labels excluded => metric undefined => honestly "
      "skipped); tail-behavior protection at admission activates with "
      "ABLATION_PLAN A16.")
    a("- The bare-site arm is 5-km-denuded (nbr features beyond 5 km "
      "remain), covers the T0+T1 core, and lower-bounds — not answers — "
      "the unmonitored-location question (A17 registers the radius sweep).")
    a("- Vault one-shot certifies the T1-core serving path (deep tiers "
      "closed at serve, T4 not applied); composite increments below the "
      "registered MDEs are not resolvable at n=50 sites.")
    a("- Conformal coverage is a two-level empirical window, not a "
      "finite-sample theorem; NexCP rho/tau are fixed heuristics; per-bin "
      "deltas are monotone-enforced and bin-0 claims are suppressed when "
      "the bin is empty.")
    return "\n".join(lines) + "\n"


# Shipped run names, keyed by config2.DOMAIN. tx is FROZEN: the committed
# v2 package lives at results/v2_texas_202608. Domains without a shipped
# name default to "<artifacts>_<domain>" (override with --run-name).
_RUN_NAME_DEFAULTS = {"tx": "v2_texas_202608"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-name",
                    default=_RUN_NAME_DEFAULTS.get(
                        config2.DOMAIN,
                        f"{config2.DOMAIN_SPEC['artifacts']}_{config2.DOMAIN}"))
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args(argv)

    root = os.path.dirname(os.path.dirname(_HERE))
    dest = os.path.join(root, "results", args.run_name)
    os.makedirs(dest, exist_ok=True)

    copied = []
    for name in COMMIT_SET:
        src = config2.artifact(name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, name))
            copied.append(name)
    print(f"[make_results] copied {len(copied)} artifacts -> {dest}")

    if not args.no_figs:
        try:
            import figs2
            figs2.main([])
            figdir = os.path.join(config2.ARTIFACTS_DIR, "figures")
            if os.path.isdir(figdir):
                fdest = os.path.join(dest, "figures")
                os.makedirs(fdest, exist_ok=True)
                for f in sorted(os.listdir(figdir)):
                    if f.endswith(".png"):
                        shutil.copy2(os.path.join(figdir, f),
                                     os.path.join(fdest, f))
                print(f"[make_results] figures -> {fdest}")
        except Exception as e:  # noqa: BLE001 — figures never block results
            print(f"[make_results] WARNING figures failed: {e}")

    md = build_results_md(args.run_name)
    with open(os.path.join(dest, "RESULTS.md"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write(md)
    print(f"[make_results] wrote {os.path.join(dest, 'RESULTS.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
