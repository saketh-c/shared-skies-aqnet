# AQNet v3 — v3_west7_202608 — Model Card & Results

Every number in this document was computed by the pipeline and read from `research/aqnet2/artifacts/v3` at assembly time by `make_results.py`; nothing is hand-entered. Absent metrics read "not computed".

- **Pipeline git sha:** f401fa3cc46d
- **Target:** EPA AQS FRM/FEM anchor + calibrated PurpleAir covariate stream (v2 inversion of the v1 target)
- **Calibration (G0):** fail — production form `amt_rht`
- **T1 decision:** budget_escape — winner `ensemble_krige` (baseline `gpboost`)
- **Admission gates open:** 0 (tier, pattern, stratum) entries across 2 tiers
- **Serving parity:** True (post_t4_replay)
- **Monotone audit:** True

## P1 — Held-out AQS sites (outer folds)

| Arm | R² | RMSE | MAE | Bias |
|---|---|---|---|---|
| T0 prior | not computed | not computed | not computed | not computed |
| T1 skeleton | not computed | not computed | not computed | not computed |
| +T2 graph | not computed | not computed | not computed | not computed |
| +T3 field | not computed | not computed | not computed | not computed |
| Composite (+T4) | not computed | not computed | not computed | not computed |

### By-year bias (flatness is the ship criterion the v1 drift failed)

| Year | Bias | Attenuation b |
|---|---|---|
| 2021 | 0.047 | not computed |
| 2022 | 0.109 | not computed |
| 2023 | 0.269 | not computed |
| 2024 | 0.251 | not computed |
| 2025 | 0.337 | not computed |

## P2 — Between-site skill (the v1 failure axis)

- With-network between-site R²: not computed ; Spearman ρ: not computed
- Bare-site arm (PA within 5 km excluded; T0+T1 core): R² not computed ; ρ not computed ()

## P3 — Exceedance (v1 recall was 0.068 → 0.000)

- thr_35.4: precision 0.463, recall 0.302, F1 0.366 (n=406,265, source npz_flag:flag_35.4)
- thr_9: precision 0.566, recall 0.603, F1 0.584 (n=406,265, source npz_flag:flag_9)

## P4 — Conformal intervals

- Site-level coverage: 0.9470 (ship window [0.88, 0.93], verdict **over_covered**), mean width 19.24 µg/m³ over 234 sites
- Quantile-band lineage vs live composite: True

## Vault (one-shot second sample)

- Composite at vault sites/period: R² not computed, RMSE not computed, bias not computed (n=not computed)

## Baselines (paired on identical rows)

| Baseline | R² | ΔR² vs composite [95% CI] |
|---|---|---|
| baselines | not computed | not computed [not computed, not computed] |
| notes | not computed | not computed [not computed, not computed] |

## Audit-stage registrations

- Margins: {"pooled_r2": 0.005, "spatial_r2": 0.010305996659858097, "exceedance_f1": 0.02} (source: ?)
- T2 kill-switch (advisory): kriging_passes = False
- Conformal δ by coverage bin: {"0": {"delta": 5.125480796025766, "delta_raw_pre_monotone": 3.136114295789482, "label": "no_pa_50km", "n_units": 58, "source": "bin+monotone_raise"}, "1": {"delta": 5.125480796025766, "label": "pa_1_

## Known limitations of this run (recorded, not hidden)

- cf_1 refetch skipped (DESIGN §4 fallback): channel-reconstructed rows are excluded from exceedance labels and carry inflated calibration variance.
- GPBoost candidate A hit a NaN-likelihood instability (suspected unit-RE/GP identifiability at repeated coordinates) — T1 is the pre-registered candidate-B escape; decision + exception recorded in oof_tier1.npz weights_json.
- NLCD impervious fractions unavailable (no public endpoint answered); NEI 2023 not yet published (2020 year-keyed values used).
- Temporal-holdout claims deferred to the temporally-pure pretrain variants (ABLATION_PLAN A11), per DESIGN §2.
- Admission gates are decided on calibrated-PA unit clusters (AQS rows are outer-held everywhere): gate deltas are a SAFETY mechanism and are never quoted as FRM-scale effect sizes; every 'tier X helps' claim is sourced from outer folds + vault only.
- Admission's exceedance-F1 axis is inert under the cf_1 skip (reconstructed labels excluded => metric undefined => honestly skipped); tail-behavior protection at admission activates with ABLATION_PLAN A16.
- The bare-site arm is 5-km-denuded (nbr features beyond 5 km remain), covers the T0+T1 core, and lower-bounds — not answers — the unmonitored-location question (A17 registers the radius sweep).
- Vault one-shot certifies the T1-core serving path (deep tiers closed at serve, T4 not applied); composite increments below the registered MDEs are not resolvable at n=50 sites.
- Conformal coverage is a two-level empirical window, not a finite-sample theorem; NexCP rho/tau are fixed heuristics; per-bin deltas are monotone-enforced and bin-0 claims are suppressed when the bin is empty.
