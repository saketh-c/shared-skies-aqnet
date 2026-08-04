# AQNet v1 → v2 — Change log with evidence

Every change below cites the v1 evidence that forced it. "Results" refers to
the committed dual-target full runs in `results/`; module references are v1
paths under `research/aqnet/`.

## Target and truth

| # | Change | Why (v1 evidence) |
|---|---|---|
| 1 | Target scale inverted: EPA AQS FRM/FEM anchors training; PurpleAir becomes a calibrated, variance-weighted covariate/pseudo-label stream | v1 trained on Barkjohn-corrected PA and scored R² ≈ 0 at AQS with −2.7…−3.8 µg/m³ bias (bias² = 21–32% of MSE); the corrected-target run was *more* biased at AQS than raw — PA-side correction cannot fix FRM agreement |
| 2 | Fixed Barkjohn correction replaced by a learned Kennedy–O'Hagan calibration (per-sensor random effects, sensor-year drift terms, smoke interactions), LOLO-gated against Barkjohn AND the AMT-2024 RH+T form | Barkjohn's cf_1 coefficients were applied to an ATM archive; by-year AQS bias drifts monotonically −3.04 → −4.28 (2021→2025) — a static 2021 correction cannot be right at both ends |
| 3 | cf_1 channel refetch (audited first; reconstruction fallback flagged and excluded from exceedance labels) | ATM diverges from cf_1 above ~25 µg/m³ — exactly the smoke/exceedance regime where v1 collapsed (recall 0.068 → 0.000 through the stack) |
| 4 | AQS quarantine replaced by nested honesty: outer spatially-blocked site folds + a 12-site one-shot vault + bare-site scoring arm | Pure quarantine guaranteed the target scale could never be anchored to truth; nested site-holdout keeps external validation honest while fixing the scale |

## Composition (the fusion fix)

| # | Change | Why |
|---|---|---|
| 5 | Late ridge stack deleted; replaced by an additive residual ladder with the incumbent coefficient frozen at 1 | v1's ridge shrank Tier-1's coefficient conditional on U-Net availability (fit on 17% of rows) — paired ΔR² −0.050/−0.031, CIs excluding 0 |
| 6 | Mean-fill is unrepresentable: availability masks dispatch to exact passthrough; `fit_gate` asserts NaN on unavailable rows; `gates.json` cannot express an incumbent coefficient or fill value; a synthetic 17%-finite component is a permanent CI regression test | The U-Net column was mean-filled on 82.7% of rows — the direct −0.05 mechanism |
| 7 | Admission gates: cross-fit α ∈ [0,1] per coverage pattern, selection/confirmation split of inner data, one-sided paired cluster-bootstrap with power-calibrated margins, shrink-to-zero for unvalidated strata, hard α = 0 beyond 100 km | v1 shipped its stack untested; the paired test that post-hoc diagnosed the regression is promoted to a pre-deployment gate |
| 8 | Residual kriging deleted as a component; per-unit random effects added to T1 instead | rk ridge weight was exactly 0.000 on both targets — LOSO residuals are device-nugget-dominated; the missing term was the unit random effect, not better kriging |
| 9 | T4 slope recalibration added as a *declared* reweighting rung (b clipped to [0.8, 1.25], cross-fit) | Slope attenuation (b < 1) drives the by-year bias drift and high-PM-year R² collapse; no v1 component addressed it |

## Spatial skill

| # | Change | Why |
|---|---|---|
| 10 | Raw lat/lon and dist_to_nearest_sensor removed from tabular features; location enters only through the GP | Permutation: static geo features ≈ 0.24 of signal as site fingerprints — memorization that manufactured within-network spatial R² which died at AQS (0.049 corrected-scale) |
| 11 | HR static covariates added (NLCD, TIGER roads, NEI emissions, 30 m DEM, population) | The model had no sub-0.5° spatial covariates; between-site skill requires site-portable predictors |
| 12 | T2 masked graph-attention interpolator with airshed/wind-conditioned edges, shielded attention, full-window masking, 4:1 mask-the-AQS fine-tune | v1's only spatial mechanism was neighbor averaging (~60% of signal); kriging was within 0.05 R² of the whole model; naive kNN over Texas smooths across airsheds |
| 13 | ERA5 grid-sourced meteorology everywhere; single `build_point_features` builder for train and serve | v1 measured a train/serve met covariate shift: PA on-board met at training, interpolated met at AQS scoring |
| 14 | INR point decoder; supervision at true coordinates | v1's `rint()` pixel snapping collapsed co-located urban sensors into one supervised value — a measured cap on spatial resolution |
| 15 | Coverage-density-gated passthrough to a debiased-CTM floor (T0) | Spatial-block fold 3: R² −1.8 — trees and deep interpolators both fail off sensor support; extrapolation must degrade to a portable prior, visibly |

## Deep learning

| # | Change | Why |
|---|---|---|
| 16 | Masked multimodal pretraining with the observation raster inside the masking objective and a discrepancy head (predict obs − CTM) | v1's U-Net learned representation from ~208 supervised pixels (val R² = kriging parity 0.5430); pretraining must not become a CTM autoencoder (CTM biases are the thing being corrected) |
| 17 | Per-channel finite masks replace per-group flag channels; no fill in any model-facing path | v1's flag channels were dead inputs; AOD missingness is non-random and mean-filling injects wrong values on exactly the days that matter |
| 18 | Fold-honest pretraining (raw PA inputs; leakage-magnitude study before paying 5× fold-pure cost; temporally-pure variant for temporal claims) | Shared pretraining on FRM-derived labels/surfaces is cross-fold leakage; pretraining across 2025 leaks the attenuation stress test |
| 19 | Checkpoint selection by sensor-bootstrap-smoothed between-site metrics | v1 selected epoch 81 on a 52-pixel val set inside a ±0.03–0.05 noise band |
| 20 | FusionUNet frozen as a paired benchmark, never in the chain | Its ceiling in v1's configuration was interpolation parity; it remains the R9 reference |

## Uncertainty and tails

| # | Change | Why |
|---|---|---|
| 21 | Conformal scores at unit level with NexCP weighting; calibration units disjoint from all selection rows; wider honest intervals pre-registered | v1 treated 77k dependent rows as exchangeable (honest n ≈ 85 sensors) → systematic 0.88 vs 0.90 undercoverage on both targets |
| 22 | Quantile heads refit on the deployed composed predictor, with artifact lineage checks | v1 grafted Tier-1's band onto Tier-3's center — no guarantee, wrong error distribution |
| 23 | Dedicated cross-fit exceedance classifier inside the admission harness | Regression-path exceedance recall was 0.068 → 0.000 through the stack; Huber down-weights exactly the events the product exists to catch |
| 24 | Epistemic spread from fold-assignment ensembles, not seed ensembles | With ~340 sensors, epistemic error is dominated by which-sensors-held-out; seed ensembles agree and are jointly wrong off-network |

## Protocol and infrastructure

| # | Change | Why |
|---|---|---|
| 25 | Primary metrics: held-out-site R²/bias + between-site R² + Spearman ρ + bare-site arm + exceedance + coverage; pooled LOSO demoted to a labeled diagnostic | Pooled R² is ~93% temporal; v1's headline metric could not see the spatial failure |
| 26 | OOF-only permutation importance | v1's permutation baseline (R² 0.79) was scored on training rows |
| 27 | Per-fold neighbor recompute extended to spatial-block and temporal folds; 7-day temporal embargo | v1's block/temporal reruns used full-pool neighbor features — a real leak |
| 28 | folds2.json content hash; window-stamped AQS cache + sidecar; artifacts/v2 namespace | Row-count-only staleness checks and the quick/full AQS cache aliasing were documented hazards; the flat artifact namespace could silently overwrite |
| 29 | Serving = bag of fold models with a mandatory serving-parity stage | v1 deployed full-data refits whose in-sample residual scale mismatched the OOF scale its combiner was fit on |
| 30 | Audit-first stage ordering + gate power analysis + T2 kill-switch probe before GPU spend | Channel provenance, colocation inventory, API cost, and detectable-effect size all determine whether downstream stages are worth their budget — measured before commitment, not assumed |
