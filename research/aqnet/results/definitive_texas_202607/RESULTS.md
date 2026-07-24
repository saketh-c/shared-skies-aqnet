# AQNet — Definitive Texas Run (July 2026) — Model Card & Results

**What this is.** The definitive, full-window run of the AQNet three-tier PM2.5 research model for Texas: 2021-01-01 → 2026-05-01, Barkjohn-corrected PurpleAir target, all external data sources fetched live (EPA AQS, NASA GEOS-CF, NASA MERRA-2), full validation battery. Every number in this document was computed by the pipeline and read from `research/aqnet/artifacts/` (`SUMMARY.md` auto-generated 2026-07-24 08:09 UTC and the `metrics_*.json` files it tabulates). Nothing is hand-entered or estimated; absent metrics are marked "not computed".

- **Run completed:** 2026-07-24 (artifacts stamped 08:09 UTC)
- **Hardware:** local workstation — NVIDIA RTX 4090 (24 GB), AMD Ryzen 9 7950X, 124 GB RAM. GPU used by XGBoost (`device='cuda'`), CatBoost (`task_type='GPU'`), and the Tier-2 deep stage (AMP).
- **Stage wall-clock** (from `logs/definitive_run.log` "stage … done" lines):

| Stage | Wall-clock |
|---|---|
| data | 31,420.0 s |
| features | 241.7 s |
| tabular | 940.6 s |
| deep | 776.1 s |
| fuse | 178.3 s |
| ablation | 1,829.7 s |
| validate | 909.2 s |
| **Total (logged stages)** | **36,295.6 s ≈ 10.08 h** |

The data stage (network-bound external fetches, dominated by MERRA-2 hourly granules and GEOS-CF OPeNDAP retries) accounts for 86.6% of the total. The log shows the data stage was invoked more than once (restartable stages; month-chunk caches are the cache of record), so compute-stage times reflect warm external caches.

---

## 1 Architecture summary

Three tiers, strictly out-of-fold end to end:

- **Tier 1 — tabular GBM ensemble.** LightGBM, XGBoost, CatBoost, and Random Forest cross-validated with GroupKFold over sensors (10 LOSO folds); neighbor features recomputed per fold from train-fold pools. Headline blend is a **leave-one-fold-out (LOFO) simplex blend** (each fold's convex weights fit on the other folds' OOF rows; the pooled-OOF blend is also reported). LightGBM quantile heads (q05/q50/q95) provide the raw interval band. Primary feature set: 45 features.
- **Tier 2 — FusionUNet** on the daily 0.1° gridded stack: **7,339,153 parameters**, **8 per-source attention groups** (aerosol 3, smoke 1, meteorology 8, static 1, temporal 4, ctm 1, merra2 6, flags 7 = **31 channels**), per-pixel softmax attention across sources + squeeze-excite gate → depth-4 U-Net → softplus. Masked **Huber loss (δ = 15)** at sensor pixels only, AMP on GPU, AdamW with 5-epoch linear warmup → cosine decay, random 96×96 crops, modality dropout 0.15. Binary flag channels mark per-source data availability each day (flag-and-fill), so attention can learn day-conditional trust.
- **Tier 3 — geostatistical fusion.** Per-fold, per-day **ordinary kriging of Tier-1 residuals** (train-fold residuals only, pykrige, max 150 train points/day); a **cross-fit positive Ridge meta-learner** over the OOF parts `[tier1, rk, unet]` (4 grouped folds over 255 meta-train sensors; U-Net part masked to held-out validation-site pixels only); **CQR-style conformal intervals** — quantile band recentered on Tier-3, widening δ calibrated on 85 sensor-disjoint calibration sensors, coverage/width reported on the disjoint meta-train cross-fit rows.

**Demographic exclusion.** `ejf_score`, `pct_people_of_color`, `pct_low_income`, and `pct_ling_isolated` are excluded from prediction everywhere; physical EJScreen source-proximity features (traffic, Superfund, RMP, diesel PM) are retained. A `plus_demographics` arm exists **only** as an ablation (§4.8) — never as a reported or deployed model.

---

## 2 Data — coverage achieved this run

| Source | Role | Coverage achieved (this run) |
|---|---|---|
| PurpleAir sensor-days (Barkjohn-corrected) | Training target + sensor meteorology | 467 sensors / 412,507 sensor-days in the neighbor pool; **293,579 in-Texas training rows from 340 sensors** after dropping 118,928 out-of-Texas rows (127 sensors); 36 non-finite-target rows are additionally dropped from the neighbor pool, leaving the 293,543 evaluated rows (293,579 − 36) |
| NASA GEOS-CF surface PM2.5 (OPeNDAP) | CTM prior feature + Tier-2 channel + baseline | **61/61 archive-available months, 4,423,760 cell-days**, 2021-01-01 → 2026-01-02 (verified from the final window-stamped parquet) |
| NASA MERRA-2 aerosol species + PBLH (earthaccess) | Features + Tier-2 channels | **65/65 months, 942,348 cell-days**, 2021-01-01 → 2026-05-01 |
| EPA AQS daily FRM/FEM PM2.5 | **External validation only — never trained on** | **86,316 site-days at 62 sites** (exceptional-event rows dropped; ≥75% sub-daily completeness) |
| CAMS AOD / PM2.5 / dust | Aerosol features/channels | From 2022-08-03 (archive start); earlier days flag-filled |
| ERA5 extras, NOAA HMS smoke, elevation, EJScreen physical proximity | Features/channels | Repo pipeline caches (full window) |

**GEOS-CF 2026 gap.** The GEOS-CF archive ends 2026-01-02; months 2026-02 → 2026-05 are recorded as failed in the `.failed.json` sidecar and are the only months missing. The gap is handled by the Tier-2 **flag channels** (ctm flag = 0 on filled days) and by NaN-tolerant tabular features. Sixteen months initially failed mid-run with OPeNDAP time-axis errors ("year 30180", non-monotonic index); sidecar-driven retries recovered all of them except the four post-archive 2026 months (see §5, the `data_external.py` fixes).

---

## 3 Headline results

Target: Barkjohn-corrected PM2.5 (µg/m³). All CIs are cluster (per-sensor) bootstrap 95% intervals unless noted. `n` = evaluated rows.

### 3.1 Leave-one-sensor-out CV (10 folds, 293,579 rows)

| Model | R² | RMSE | MAE | Bias | n | R² 95% CI | RMSE 95% CI |
|---|---|---|---|---|---|---|---|
| **Tier-1 LOFO blend (headline)** | **0.6000** | **2.740** | **1.492** | 0.0170 | 293,543 | [0.5363, 0.6655] | [2.404, 3.128] |
| Tier-1 pooled-weights blend | 0.6018 | 2.734 | 1.488 | 0.0141 | 293,543 | not computed | not computed |
| Tier-1 LightGBM | 0.5822 | 2.800 | 1.539 | 0.0084 | 293,543 | not computed | not computed |
| Tier-1 XGBoost | 0.5861 | 2.787 | 1.550 | 0.0099 | 293,543 | not computed | not computed |
| Tier-1 CatBoost | 0.5861 | 2.787 | 1.548 | 0.0338 | 293,543 | not computed | not computed |
| Tier-1 Random Forest | 0.6003 | 2.739 | 1.490 | 0.0127 | 293,543 | not computed | not computed |
| Tier-2 U-Net (held-out validation-site rows only) | 0.5007 | 3.049 | 1.843 | −0.3832 | 50,688 | [0.3624, 0.6235] | [2.430, 3.857] |
| Tier-3 cross-fit meta (meta-train sensors, OOF) | 0.5614 | 2.817 | 1.767 | −0.0646 | 216,416 | [0.5020, 0.6161] | [2.528, 3.149] |
| Tier-3 final ridge — all rows | 0.5554 | 2.889 | 1.744 | −0.0426 | 293,543 | [0.4971, 0.6087] | [2.586, 3.262] |
| Tier-3 final ridge — calibration sensors only | 0.5335 | 3.105 | 1.705 | −0.0250 | 77,127 | [0.4098, 0.6566] | [2.414, 3.953] |

Pooled-OOF simplex weights (the pooled-weights blend row above): RF 0.7296, XGB 0.1798, CatBoost 0.0906, LGBM 0.0000. The headline LOFO blend instead uses 10 per-fold weight vectors (`weights_lofo`), which vary across folds — RF 0.6808–0.7533, XGB 0.0000–0.2541, CatBoost 0.0000–0.1533; LGBM is 0.0000 in nine folds but receives 0.1013 in fold 10 (where XGB ≈ 0). Tier-2 training itself (grouped-site holdout): best val R² 0.5394 / RMSE 2.839 / MAE 1.790 (n = 37,058) at epoch 72 of 100.

**Intervals.** Raw Tier-1 quantile band (q05–q95) empirical coverage: 0.6610 (n = 293,543) — under-covering, as expected pre-conformal. Conformal (α = 0.10): δ = 1.767 fit on 77,127 calibration rows; on the disjoint meta-train cross-fit rows (n = 216,416) **coverage 0.8828, mean width 7.080 µg/m³**.

### 3.2 Interpolation and raw-CTM baselines (same LOSO rows)

| Baseline | R² | RMSE | MAE | Bias | n | AQI cat. acc. | R² 95% CI |
|---|---|---|---|---|---|---|---|
| Nearest sensor | 0.2859 | 3.661 | 1.909 | 0.0431 | 293,543 | 0.8964 | [0.1808, 0.3822] |
| IDW (k=8) | 0.5242 | 2.988 | 1.590 | 0.0346 | 293,543 | 0.9116 | [0.4501, 0.5918] |
| Ordinary kriging | 0.5431 | 2.928 | 1.597 | 0.0230 | 293,543 | 0.9145 | [0.4744, 0.6084] |
| Raw CAMS PM2.5 | −0.9569 | 6.059 | 4.611 | 3.522 | 251,433 | 0.6780 | [−1.143, −0.7810] |
| CAMS mean-debiased (offset 3.522) | −0.2956 | 4.930 | 3.530 | −0.0000 | 251,433 | 0.8052 | [−0.3529, −0.2464] |
| Raw GEOS-CF PM2.5 | −8.348 | 13.09 | 10.25 | 9.747 | 259,421 | 0.4092 | [−9.532, −7.165] |
| GEOS-CF mean-debiased (offset 9.747) | −3.165 | 8.737 | 6.608 | −0.0000 | 259,421 | 0.7285 | [−3.534, −2.785] |
| Raw MERRA-2 PM2.5 proxy | −5.676 | 11.19 | 8.632 | 8.102 | 293,543 | 0.3862 | [−6.440, −4.822] |
| MERRA-2 mean-debiased (offset 8.102) | −2.178 | 7.723 | 5.090 | 0.0000 | 293,543 | 0.7742 | [−2.440, −1.885] |

All baseline CIs are sensor-clustered bootstrap 95% intervals from `metrics_baselines.json` (RMSE CIs are computed there as well). Note the ordinary-kriging R² CI [0.4744, 0.6084] overlaps Tier-1's [0.5363, 0.6655] and its upper bound exceeds Tier-1's point estimate (0.6000); no paired-difference CI was computed for the baselines (see §4.2). Macro F1 and exceedance precision/recall for the six CTM rows are tabulated in §3.6.

### 3.3 Spatial-block CV (5 region blocks, 293,543 rows)

| Model | R² | RMSE | MAE | Bias | R² 95% CI | RMSE 95% CI |
|---|---|---|---|---|---|---|
| Tier-1 LOFO blend | 0.4960 | 3.076 | 1.880 | 0.2776 | [0.4370, 0.5551] | [2.759, 3.447] |
| Tier-1 pooled blend | 0.5070 | 3.042 | 1.853 | 0.3021 | not computed | not computed |

Spatial-block pooled weights: RF 0.7006, XGB 0.2994, CatBoost 0.0000, LGBM 0.0000. AQI category accuracy 0.8986, macro-F1 0.3900, exceedance precision nan / recall 0.0000 (336 true exceedances); residual Moran's I daily mean 0.1094 (median 0.0886, IQR 0.1548, 1,947 days). Per-block fold heterogeneity is severe (see §4.5): fold R² by block = 0.4291–0.5811 for the three large blocks vs **−1.979 to −0.0299** for the two small blocks (n_test = 4,672 and 6,204).

### 3.4 Temporal holdout (train < 2025-01-01; n_train 166,170, n_test 127,409)

Tier-1 LOFO blend: **R² 0.6646, RMSE 2.347, MAE 1.287**, bias 0.0701 (n = 127,373); R² 95% CI [0.5920, 0.7218], RMSE 95% CI [2.106, 2.607]; AQI category accuracy 0.9357, macro-F1 0.4162, exceedance recall 0.0000 (105 true exceedances); Moran's I daily mean 0.0446 (median 0.0338, IQR 0.0738, 486 days — also in the §3.9 table).

### 3.5 External EPA AQS validation (never trained on; 86,316 site-days, 62 sites)

| Model (deployment mode) | R² | RMSE | MAE | Bias | n | R² 95% CI |
|---|---|---|---|---|---|---|
| Full-data Tier-1 ensemble at AQS sites | −0.0488 | 6.302 | 4.092 | −3.570 | 86,316 | [−0.1231, 0.0329] |
| **Deployment-mode Tier-3** (full-data tier1 + per-day kriged residuals + U-Net surface, final ridge) | −0.0488 | 6.302 | 4.097 | −3.640 | 86,316 | [−0.1216, 0.0284] (site-clustered) |

Tier-3 row: `ridge_cols = [tier1, rk, unet]`, U-Net finite at all 86,316 site-days, 0 NaN predictions. AQI category accuracy 0.6936 (Tier-1) / 0.6906 (Tier-3), macro-F1 0.2005 (Tier-1) / 0.1968 (Tier-3); Tier-1 exceedance precision 1.0000 at recall 0.0048, Tier-3 exceedance recall 0.0000 (418 true exceedances).

Per-year breakdown (Tier-1 deployment mode):

| Year | R² | RMSE | MAE | Bias | n |
|---|---|---|---|---|---|
| 2021 | −0.1276 | 5.093 | 3.641 | −3.039 | 14,148 |
| 2022 | −0.1729 | 5.552 | 3.807 | −3.227 | 16,071 |
| 2023 | 0.0079 | 5.305 | 3.855 | −3.376 | 16,372 |
| 2024 | 0.0835 | 6.731 | 4.437 | −3.976 | 17,519 |
| 2025 | −0.1772 | 7.947 | 4.690 | −4.280 | 17,977 |
| 2026 | 0.1691 | 6.441 | 3.633 | −2.711 | 4,229 |

The systematic negative bias (−2.7 to −4.3 µg/m³ every year) is consistent with the Barkjohn-corrected PurpleAir target sitting below FRM/FEM daily values and with the UTC-vs-local-day convention mismatch (§6 of REPORT.md); the near-zero R² reflects a PurpleAir-trained model evaluated against a differently-instrumented, urban-weighted, 62-site network.

### 3.6 AQI-category skill (EPA 2024 breakpoints, LOSO rows)

| Model | Category accuracy | Macro F1 | Exceed. precision | Exceed. recall | n (true exceed.) |
|---|---|---|---|---|---|
| Tier-1 LOFO blend | 0.9196 | 0.4391 | 0.7419 | 0.0685 | 336 |
| Tier-3 all rows | 0.9051 | 0.3813 | nan | 0.0000 | 336 |
| Tier-3 cross-fit (meta-train) | 0.9019 | 0.3762 | 0.6667 | 0.0103 | 194 |
| Tier-3 calibration sensors only | 0.9145 | 0.3953 | nan | 0.0000 | 142 |
| Tier-2 U-Net (holdout rows) | 0.8989 | 0.3885 | nan | 0.0000 | 35 |
| Nearest sensor | 0.8964 | 0.4390 | 0.1175 | 0.1101 | 336 |
| IDW k=8 | 0.9116 | 0.4613 | 0.3600 | 0.1071 | 336 |
| Ordinary kriging | 0.9145 | 0.4336 | 0.3774 | 0.0595 | 336 |
| Raw CAMS PM2.5 | 0.6780 | 0.2912 | 0.0000 | 0.0000 | 262 |
| CAMS mean-debiased | 0.8052 | 0.3119 | 0.0000 | 0.0000 | 262 |
| Raw GEOS-CF PM2.5 | 0.4092 | 0.1993 | 0.0019 | 0.0461 | 304 |
| GEOS-CF mean-debiased | 0.7285 | 0.2980 | 0.0020 | 0.0132 | 304 |
| Raw MERRA-2 proxy | 0.3862 | 0.1886 | 0.0017 | 0.0268 | 336 |
| MERRA-2 mean-debiased | 0.7742 | 0.3064 | 0.0025 | 0.0179 | 336 |

### 3.7 Strata (Tier-1 LOFO, LOSO rows)

| Stratum | R² | RMSE | MAE | Bias | n |
|---|---|---|---|---|---|
| Smoke (HMS) | 0.6387 | 2.801 | 1.540 | 0.0045 | 116,826 |
| Dust | 0.6213 | 2.656 | 1.450 | 0.0203 | 61,212 |
| Clean | 0.5406 | 2.682 | 1.468 | 0.0269 | 146,736 |

### 3.8 Spatial vs temporal R² decomposition

| Model | Spatial R² (sensor means) | Temporal R² (anomalies) | n_sensors | n |
|---|---|---|---|---|
| Tier-1 LOFO | 0.0491 | 0.6743 | 340 | 293,543 |
| Tier-3 cross-fit | 0.1559 | 0.6251 | 255 | 216,416 |

### 3.9 Residual Moran's I (per-day, mean / median / IQR / n_days)

| Model | Mean | Median | IQR | n_days |
|---|---|---|---|---|
| Tier-1 LOFO (LOSO) | −0.0564 | −0.0597 | 0.0387 | 1,947 |
| Tier-3 all rows | 0.0113 | −0.0038 | 0.0753 | 1,947 |
| Tier-3 cross-fit | 0.0093 | −0.0040 | 0.0815 | 1,947 |
| Tier-2 U-Net (holdout) | 0.0353 | 0.0041 | 0.1403 | 1,709 |
| Tier-1 LOFO (spatial-block) | 0.1094 | 0.0886 | 0.1548 | 1,947 |
| Tier-1 LOFO (temporal holdout) | 0.0446 | 0.0338 | 0.0738 | 486 |
| Nearest sensor | −0.0844 | −0.0885 | 0.0270 | 1,947 |
| IDW k=8 | −0.0955 | −0.0976 | 0.0224 | 1,947 |
| Ordinary kriging | −0.0600 | −0.0826 | 0.0607 | 1,947 |
| Raw GEOS-CF | 0.4993 | 0.5138 | 0.3681 | 1,828 |
| Raw CAMS | 0.4079 | 0.4026 | 0.2965 | 1,367 |
| Raw MERRA-2 proxy | 0.4113 | 0.3967 | 0.3250 | 1,947 |

### 3.10 Ablation — Tier-1 on the frozen LOSO folds (paired ΔR², 1,000-rep cluster bootstrap over 340 sensors)

| Variant | n_features | R² | RMSE | MAE | ΔR² vs primary | ΔR² 95% CI |
|---|---|---|---|---|---|---|
| **primary** | 45 | 0.6000 | 2.740 | 1.492 | — | — |
| plus_demographics (ablation-only) | 49 | 0.5988 | 2.744 | 1.498 | −0.0012 | [−0.0099, 0.0071] |
| no_external (drop geoscf_*/merra2_*) | 35 | 0.5961 | 2.753 | 1.506 | −0.0039 | [−0.0063, −0.0021] |
| no_neighbor (drop nbr_*) | 38 | 0.5898 | 2.775 | 1.556 | −0.0102 | [−0.0243, 0.0064] |

### 3.11 Permutation importance (full-data model; base R² 0.7892 on n = 5,000, 5 repeats, seed 0)

Top 10 by mean ΔR² when permuted: `nbr_pm25_100km` 0.2392, `nbr_pm25_25km` 0.1497, `nbr_pm25_50km` 0.1246, `rmp_proximity` 0.05872, `latitude` 0.05023, `longitude` 0.04029, `dist_to_coast` 0.03258, `dist_to_nearest_sensor` 0.03091, `traffic_proximity` 0.02496, `diesel_pm_proximity` 0.02473.

---

## 4 Honest interpretation

1. **Do not compare these R² values to the production 0.7136.** The production ensemble's LOSO R² = 0.7136 is on the **raw ATM target**; AQNet's primary target is Barkjohn-corrected, which compresses target variance **4.03× (69.71 → 17.32 µg²/m⁶** — repo README), and the feature and sensor sets differ. R² on different targets is not comparable; the `--correction raw` sensitivity run is the like-for-like comparison and was not part of this run.
2. **What the baselines show.** Tier-1 LOSO R² 0.6000 clearly beats nearest-sensor (0.2859, CI [0.1808, 0.3822]) on identical rows, and its point estimate sits above IDW (0.5242, CI [0.4501, 0.5918]) and ordinary kriging (0.5431, CI [0.4744, 0.6084]). The kriging CI overlaps Tier-1's [0.5363, 0.6655] and its upper bound (0.6084) exceeds Tier-1's point estimate, and no paired-difference CI was computed for the baselines, so the margin over kriging is suggestive rather than statistically resolved. The paired ablation shows the external CTM features contribute (ΔR² −0.0039, CI [−0.0063, −0.0021] excluding 0); the neighbor-feature ablation is inconclusive (ΔR² −0.0102, CI [−0.0243, +0.0064] spanning 0), and unlike the other variants the `no_neighbor` run used `used_fold_overrides = false`. Raw CTMs are far worse than useless as direct predictors here (GEOS-CF R² −8.348; even mean-debiased −3.165) — their value is as features/channels, not surfaces.
3. **The U-Net earns real weight in the stack.** The cross-fit positive Ridge assigns the U-Net 0.251–0.413 across the four meta folds (final full-fit: intercept 0.083, tier1 0.660, rk 0.000, unet 0.338) even though its standalone holdout R² (0.5007) trails Tier-1 — it contributes complementary, spatially structured signal.
4. **The rk = 0 finding.** Residual kriging received **exactly zero weight in every cross-fit fold and in the final ridge**. After Tier-1's leave-self-out neighbor features, the LOFO residuals carry no additional daily spatial structure that kriging can exploit — corroborated by Tier-1's residual Moran's I being slightly *negative* (mean −0.0564). Tier-3's headline gain over Tier-1 is therefore null on these axes (0.5554 vs 0.6000 all-rows), and Tier-3's value in this run is (a) the modest spatial-R² improvement (0.1559 vs 0.0491) and (b) carrying the calibrated conformal band.
5. **Fold heterogeneity is real and reported.** Spatial-block CV: the two small blocks (n_test 4,672 and 6,204) have strongly negative per-model R² (down to −1.979) while the three large blocks sit at 0.43–0.58; the pooled 0.4960 hides this. The Tier-1 LOSO bootstrap CI [0.5363, 0.6655] and the Tier-2 CI [0.3624, 0.6235] are wide because per-sensor performance varies. Spatial extrapolation to sensor-free regions is the weakest axis — consistent with the tiny spatial R² (0.0491) of Tier-1.
6. **Exceedance detection is poor everywhere.** With only 336 true exceedance days in 293,543 LOSO rows, the best exceedance recall of any model is 0.11 (nearest sensor, at 0.12 precision); Tier-1 manages precision 0.7419 at recall 0.0685. These models smooth; none is an exceedance alarm.
7. **External AQS is a hard, honest check** and comes back near zero R² with a systematic −3.6 µg/m³ bias. It bounds transportability claims: AQNet as trained predicts *corrected PurpleAir*, not FRM/FEM.

---

## 5 Reproducibility

**Command** (repo root; stages restartable, run individually in sequence):

```bash
python research/aqnet/pipeline_colab.py data
python research/aqnet/pipeline_colab.py features
python research/aqnet/pipeline_colab.py tabular
python research/aqnet/pipeline_colab.py deep
python research/aqnet/pipeline_colab.py fuse
python research/aqnet/pipeline_colab.py ablation
python research/aqnet/pipeline_colab.py validate
```

Defaults used: `--correction barkjohn`, `--grid-deg 0.1`, window 2021-01-01 → 2026-05-01, 10 LOSO folds, 100 U-Net epochs (best checkpoint at epoch 72), auto batch size (CUDA).

**Seeds:** fold construction seed 42; bootstrap seed 0; U-Net trainer seeds torch/numpy (residual GPU convolution nondeterminism may perturb Tier 2 slightly between runs).

**Key package versions** (`artifacts/versions.txt`): numpy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0, scipy 1.18.0, lightgbm 4.7.0, xgboost 3.3.0, catboost 1.2.10, torch 2.11.0+cu128, xarray 2026.7.0, netCDF4 1.7.4, pydap 3.5.10, PyKrige 1.7.3, earthaccess 0.18.0, shap 0.52.0.

**The two `data_external.py` fixes this run depended on** (commit `de1f73e`, "Fix GEOS-CF OPeNDAP fetch: pydap fallback + GrADS time-axis decode"): (1) NASA's GrADS Data Server rejects libnetCDF's DAP client, so `_open_geoscf` falls back to **pydap with an explicit `dap2://` URL** (auto-negotiation behind the load balancer intermittently misparses the axis — the "year 30180" errors in the log); (2) the server's "days since 1-1-1" **GrADS time axis is decoded manually** (`decode_times=False`, `fromordinal(floor(v) − 1) + frac`) because GrADS day counts run one ahead of Python's proleptic-Gregorian ordinal.

**Caches:** window-stamped final parquets (`geoscf_pm25_20210101_20260501.parquet`, `merra2_daily_tx_20210101_20260501.parquet`, `aqs_daily_tx.parquet` — paths recorded in `artifacts/external_paths.json`), month chunks as the cache of record, `.failed.json` sidecars retried first on the next run; Tier-2 stack cache `cache/extended_stack_full_0.1deg_barkjohn_20210101_20260501_ctm-merra2-flags.npz`; checkpoint `artifacts/unet/fusion_unet_best.pt`.

---

## 6 Limitations

Summarized from REPORT.md §7, plus run-specific notes:

- **Sensor measurement and siting.** PurpleAir optics undercount coarse dust (dust stratum reported separately), the linear Barkjohn correction degrades under extreme smoke, and the residentially-sited network over-represents affluent urban areas, with survivorship bias in the archive itself.
- **Spatial/temporal support.** 0.1° (~11 km) predictions are area averages, not personal exposures; UTC-day aggregation vs AQS local-time days slightly and knowingly penalizes the external comparison.
- **Conformal guarantees are marginal, not conditional** — average coverage (0.8828 observed at α = 0.10) is controlled, but coverage may vary by region, season, or concentration.
- **LOSO optimism where sensors cluster** — bounded by the spatial-block (R² 0.4960) and external-AQS (R² −0.0488) axes, which are the honest lower envelope; external AQS is 62 urban-weighted sites, limiting rural power.
- **Intended use.** Research and monitoring-investment prioritization only; not regulatory attainment, enforcement, or individual medical decisions. Demographic variables never enter prediction.
- **Run-specific.** GEOS-CF ends 2026-01-02, so 2026-02 → 2026-05 have no CTM channel (flag-filled; the four months remain in the failed sidecar); CAMS channels begin 2022-08-03 (earlier days flag-filled); 16 GEOS-CF months initially failed on OPeNDAP time-axis errors and were recovered by sidecar retries; exceedance-day skill is weak in all models (336 true exceedances in 293,543 rows).

---

*Generated from `research/aqnet/artifacts/` of the 2026-07-24 definitive run. Production live-map metrics (`models/metrics.json`, LOSO R² 0.7136) describe the deployed system, not AQNet.*
