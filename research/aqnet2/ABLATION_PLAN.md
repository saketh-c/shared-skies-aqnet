# AQNet v2 — Pre-registered ablation plan

DESIGN §13 requires the itemized ablation budget to be published **before the
first full submission**; this file is that registration. Every arm re-runs
the affected stages with one lever moved and writes into its own
`artifacts/v2/<arm>/` namespace (the flat-namespace overwrite hazard closed
by config2.artifact(sub=...)). No arm ever ships; `plus_demographics` in
particular exists only to quantify the cost of the demographic exclusion and
is never a reported model.

| # | Arm | Lever | Stages re-run | Compute class | Status at first submission |
|---|-----|-------|---------------|---------------|----------------------------|
| A1 | `no_lags` | interpolating lag features dropped from T1 | features (view), skeleton | CPU ~2 h | queued (run after primary skeleton) |
| A2 | `no_statics` | st_* columns dropped | features (view), skeleton | CPU ~2 h | queued |
| A3 | `plus_demographics` | 4 EJScreen demographic columns added (ablation-only assert) | skeleton | CPU ~2 h | queued |
| A4 | `atm_vs_cf1` | calibration on reconstructed cf_1 vs ATM input | calibrate, skeleton | CPU ~3 h | queued |
| A5 | `no_pa_truth_only` | AQS-only training rows (lambda = 0) | skeleton | CPU ~1 h | queued |
| A6 | `no_pretrain_t2` | graphres from random init | graphres | GPU ~6 h | registered, post-primary |
| A7 | `no_pretrain_t3` | fieldres INR on random-init encoder | fieldres | GPU ~6 h | registered, post-primary |
| A8 | `naive_knn_graph` | T2 edges = plain 10-NN, no airsheds/wind | graphres | GPU ~6 h | registered, post-primary |
| A9 | `inr_vs_pixel` | T3 pixel-snapped head vs INR | fieldres | GPU ~6 h | registered, post-primary |
| A10 | `grid_only_pretrain` | T3 pretrain without the obs raster | fieldpre+fieldres | GPU ~10 h | registered, post-primary |
| A11 | `temporal_pure` | temporally-pure pretrain variants (< 2025) | graphpre/graphres/fieldpre/fieldres `--variant temporal` | GPU ~20 h | registered — REQUIRED before any temporal-holdout claim (metrics_temporal.json stays "absent" until run) |

Decision rules bound to ablations (registered here, executed by the primary
run's own gates):

* **Leakage-magnitude study** (graphpre `--study`): shared-vs-fold-pure
  pretraining on outer fold 0; gap_r2 > 0.01 forces fold-pure pretraining
  for every admitted T2 claim. The decision lands in
  `leakage_study.json` either way.
* **T2 kill-switch** (audit stage): if kriging-on-residuals cannot pass the
  admission test on v1 residuals, T2 runs as a research arm and its GPU
  budget is re-scoped (recorded in `power_analysis.json.t2_killswitch`).
* **Skeleton budget escape**: candidate-A fit projection over 10 h triggers
  the pre-registered candidate-B escape; the decision and timings are in
  `oof_tier1.npz weights_json`.

Total registered budget ≈ 150–250 GPU-h (≈ 2× the primary), embers-shaped;
`inferno` remains the named contingency for `fieldpre` only. CPU arms
(A1–A5) run immediately after the primary CPU chain completes; GPU arms
(A6–A11) queue after the primary GPU tiers, in the order above.
