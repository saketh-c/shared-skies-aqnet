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

## Amendments from the 2026-08-04 methodology audit (registered pre-gates)

| # | Arm / action | Why | Compute |
|---|--------------|-----|---------|
| A12 | `fold_geometry_seeds` — 3-seed refold of folds2 + T1-only rerun | single-seed outer/vault/airshed geometry is a real variance term at 50 sites | CPU ~6 h |
| A13 | `unit_intercept_b` — candidate B + partially-pooled MixedLM unit intercept on blend residuals | the device-nugget fix (CHANGES #8) currently lives only in demoted candidate A | CPU ~3 h |
| A14 | `v1_rescored_frm` — v1 Tier-1 re-scored under the identical FRM protocol (DESIGN §11 contract item, deferred from the primary run: requires a v1 serve path against the v2 frame) | the single comparison a reviewer asks for first | CPU ~4 h |
| A15 | external-product comparison — van Donkelaar V5 / EPA Downscaler surfaces scored at the vault sites (post-freeze; no refit) | positions against published products instead of only internal baselines | CPU ~2 h |
| A16 | tail-only cf_1 refetch (flagged rows only, ~hours not days) + calibration/exceedance re-run | the smoke/high-PM regime currently rests on the reconstructed ATM channel — the single highest-value data acquisition; requires PurpleAir API spend (OWNER DECISION) | fetch hours + CPU ~3 h |
| A17 | bare-site radius sweep (5/25/50 km denudation) + admitted-tier bare-site scoring with graph inputs masked | "5-km-denuded" must not be read as "unmonitored"; T2's 150-km graph is not neutralized by a 5-km strip | CPU+GPU ~6 h |

Deviation (2026-08-05, primary run): the T1 candidate-A budget escape was
fired by operator override (`AQNET2_FORCE_ESCAPE=1`) rather than by the
in-band timing probe — the probe measures the lambda=1.0 fit, but the FIRST
lambda fit alone exceeded 3.4 h wall on the full frame (job 11699055),
projecting ~190 h against the registered 10 h budget; the probe could
therefore never reach its own checkpoint. The decision label, evidence and
env override are recorded in `oof_tier1.npz weights_json`. Candidate A
remains a registered post-primary arm (A13 covers the unit-intercept
follow-up).

Deviations registered (disclosed, not silently accepted): outer folds enforce
size bounds but not the CBSA-share cap; vault stratification omits urbanicity
(metadata absent pre-refetch); the 7-day temporal embargo is enforced by
backward-only lags rather than a cutoff gap — temporal claims remain gated on
the A11 temporally-pure variants; T4 is cross-fit by cluster but not nested
per outer fold (2-parameter global affine; per-outer sensitivity to be
reported with A12); admission's exceedance-F1 axis is inert under the cf_1
skip (labels excluded ⇒ metric undefined ⇒ honestly skipped) until A16 runs.
