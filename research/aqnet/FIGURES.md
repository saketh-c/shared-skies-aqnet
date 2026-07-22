# AQNet Figure Plan

The complete figure inventory for the AQNet paper (main text + SI) and poster.
Every figure script imports `fig_style.py` first (it locks the Agg backend and
owns the palette, rcParams, column widths, and save path) and writes through
`fig_style.save_fig(...)` — nothing else may touch `figures/`.

## Conventions (binding)

- **Quick-mode watermark.** Every render sourced from the current quick-mode
  artifacts (3-month smoke run) uses `save_fig(..., preview=True)`: it lands in
  `figures/preview_quick/` (gitignored) with the diagonal
  "QUICK-MODE PREVIEW — NOT RESULTS" watermark. No quick-mode number ever
  appears in a non-watermarked figure. After the full Colab run regenerates
  `artifacts/`, the same scripts re-run with `preview=False` to produce finals.
- **Citable numbers.** The only citable number today is the **production
  baseline (LOSO R² = 0.7136)**; wherever it is drawn as a reference line it is
  labeled "production baseline". Bracketed `[X.XX]` tokens in captions are
  placeholders filled only from full-run artifacts.
- **Identity.** Colors come from `fig_style.COLORS` by entity name (tier1 blue,
  tier2 orange, tier3 green, baselines gray ramp, priors sky/vermillion/pink,
  observed dark ink) and are never cycled; every multi-series plot also uses
  distinct markers/linestyles plus a legend. Sequential surfaces = viridis;
  bias/residual maps = RdBu_r centered on zero. One axis per plot; small
  multiples instead of dual scales.
- **Maps.** Tract/state polygons from `backend/static/texas_all_tracts.geojson`
  as a light-gray context layer (`rasterized=True`), data on top, colorbar
  labeled "PM2.5 (ug/m3)"; no basemap tiles.
- **No fabricated data.** Figures render only from repo data files or
  `artifacts/`; a missing artifact prints a clear skip message and returns None.

## Status legend

| Status | Meaning |
|---|---|
| **renders-now (final)** | Uses only repo data or is a schematic — final quality today, no watermark. |
| **renders-now (preview)** | Renders today from quick-mode artifacts, watermarked into `preview_quick/`; final after Colab. |
| **post-Colab** | Needs outputs the quick run did not produce (or produced only degenerately); a watermarked partial preview may exist. |

## Figure inventory

| ID | Filename | Title | Draft caption | Source | Status | Paper | Poster |
|----|----------|-------|---------------|--------|--------|-------|--------|
| F01 | `f01_architecture_three_tier` | AQNet three-tier fusion architecture | Overview of the AQNet three-tier fusion architecture. Tier 1 blends gradient-boosted and random-forest learners over physical features with a leave-one-fold-out simplex weighting and LGBM quantile heads; Tier 2 is a multi-source FusionUNet producing gridded PM2.5 surfaces; Tier 3 combines both with per-day kriged residuals through a cross-fitted grouped ridge and adds split-conformal (CQR) intervals re-centered on the Tier-3 mean. Arrows show data flow; no demographic variable enters any tier. | Schematic (drawn programmatically; structure verified against `models_tabular.py`, `models_deep.py`, `fusion.py`) | renders-now (final) | Main | Yes — simplified banner version |
| F02 | `f02_fusion_unet` | FusionUNet architecture | Detail of the Tier-2 FusionUNet. Per-source convolutional encoders ingest channel groups — aerosol (AOD, CAMS PM2.5, dust), smoke, meteorology (8), static terrain, temporal encodings, GEOS-CF, MERRA-2 (6), and availability flags — a per-pixel softmax attention fuses sources, and a squeeze-excite block feeds a depth-4 GroupNorm U-Net whose softplus head yields a non-negative PM2.5 surface. Training uses masked Huber(15) loss at sensor pixels with modality dropout, AdamW warmup+cosine, AMP, and 96-px crops on the 0.1° grid. | Schematic (verified against `models_deep.py`, `grids.py`) | renders-now (final) | SI | No |
| F03 | `f03_data_provenance` | Data provenance and leakage firewalls | Provenance of every data stream and the firewalls that prevent leakage. Barkjohn-corrected PurpleAir sensor-days supply supervision; CAMS, GEOS-CF, MERRA-2, HMS smoke, meteorology, and terrain supply predictors; EPA AQS observations are quarantined for external validation and never enter training. Shaded boundaries mark where grouping by sensor and by fold blocks information flow between training and evaluation, including the masking of U-Net supervision pixels to validation sites in Tier 3. | Schematic (verified against `config.py`, `data_external.py`, `validation.py`) | renders-now (final) | SI | No |
| F04 | `f04_validation_protocol` | Validation protocol schematic | The five evaluation protocols applied to one frozen pipeline: leave-one-site-out GroupKFold(10), KMeans(5) spatial-block folds, a temporal holdout (train < 2025-01-01, test ≥ 2025-01-01), external EPA AQS comparison, and classical baselines (nearest-neighbor, IDW, kriging, raw and debiased CTM priors). Every reported score carries a sensor-cluster bootstrap confidence interval. | Schematic; fold geometry optionally illustrated from `artifacts/folds.json` | renders-now (final) | SI | No |
| F05 | `f05_study_domain` | Study domain and monitoring networks | Study domain: Texas on the 0.1° analysis grid (25.6–36.7° N, 107.0–93.3° W). Census tracts drawn as light-gray context; PurpleAir sensors sized by days of record, and the 62 EPA AQS regulatory monitors as distinct markers. AQS sites are reserved for external validation and never used in training. | Repo data: `backend/static/texas_all_tracts.geojson`, `pipeline/purpleair_full_dataset.parquet`, `pipeline/sensor_tx_membership.csv`, `research/aqnet/data/aqs_daily_tx.parquet`, `config.py` (bbox/grid) | renders-now (final) | Main | Yes — intro panel |
| F06 | `f06_data_coverage` | Data coverage timeline | Data coverage across the 2021-01-01 to 2026-05-01 study window. Daily counts of reporting PurpleAir sensors, with availability spans of each external source (CAMS, GEOS-CF, MERRA-2, HMS smoke) as horizontal bars beneath; the temporal-holdout cutoff (2025-01-01) is marked and the training period shaded. | Repo data: `pipeline/purpleair_full_dataset.parquet`, `config.py` dates; external-source spans confirmed from full-run `artifacts/external_paths.json` | renders-now (final; external-source rows re-checked post-Colab) | SI | No |
| F07 | `f07_barkjohn_correction` | Barkjohn correction effect | Effect of the Barkjohn et al. (2021) correction on PurpleAir PM2.5. Panels show raw versus corrected concentration as a function of relative humidity and the resulting distributional shift; the correction removes the humidity-driven overestimate of the raw signal before any modeling. | Repo data: `pipeline/purpleair_full_dataset.parquet` + `corrections.py` | renders-now (final) | SI | No |
| F08 | `f08_main_results_forest` | Main results: models vs baselines | Cross-validated performance of all models and baselines under leave-one-site-out evaluation. Filled markers show R² with 95% sensor-cluster bootstrap confidence intervals for Tiers 1–3; open markers show nearest-neighbor, IDW, kriging, and raw/debiased CTM priors. The dashed reference line is the production baseline (LOSO R² = 0.7136), the pre-existing system this work must beat. | Artifacts: `metrics_loso.json`, `metrics_baselines.json` | renders-now (preview) | Main | Yes — hero results panel |
| F09 | `f09_pred_vs_obs_density` | Predicted vs observed density panels | Predicted versus observed daily PM2.5 as 2-D log-count density panels (viridis) for Tiers 1–3 under LOSO validation, with the 1:1 line in neutral ink. Per-panel R², RMSE, and mean bias are annotated via the standard metrics box. Density shading avoids overplotting across [N] sensor-day pairs. | Artifacts: `oof_tier1.npz`, `oof_meta.npz` (Tier-2/3 OOF columns as stored there) | renders-now (preview) | Main | No |
| F10 | `f10_spatial_block` | Spatial-block generalization | Spatial-block generalization. (a) The five KMeans sensor blocks over the Texas domain with tract outlines as context; (b) per-block R² with 95% bootstrap CIs for Tier 3 against the strongest interpolation baseline. Whole blocks are held out, so scores measure transfer to unmonitored regions. | Artifacts: `metrics_spatial_block.json`, `folds.json`; coords from `pipeline/purpleair_full_dataset.parquet`; tracts geojson | renders-now (preview) | Main | No |
| F11 | `f11_temporal_holdout` | Temporal holdout time series | Temporal holdout: domain-average observed and predicted PM2.5 through the test period (≥ 2025-01-01), with the training period shaded light gray. Observations in dark ink, Tier 3 in green with its 90% CQR interval band; residual statistics annotated. Quick-mode previews use the smoke-run window's internal cutoff, not the paper cutoff. | Artifacts: `metrics_temporal.json`, `oof_meta.npz`, `training_frame.parquet`, `quantile_oof.npz` | renders-now (preview; paper cutoff only post-Colab) | SI | No |
| F12 | `f12_external_aqs` | External EPA AQS validation | External validation against EPA AQS: 86,316 site-days at 62 regulatory monitors never used in training. (a) Predicted versus observed density scatter with 1:1 line; (b) per-site mean bias mapped over Texas (RdBu_r, centered at zero); (c) distribution of site-level bias. Agreement with an independent reference network is the strongest evidence the exposure fields transfer beyond the PurpleAir network. | Artifacts: `metrics_external_aqs.json` (+ `aqs_quick_subset.parquet` in preview); repo `research/aqnet/data/aqs_daily_tx.parquet` for site coordinates; tracts geojson | renders-now (preview) | Main | Yes — panel (a) only |
| F13 | `f13_conformal_calibration` | Conformal interval calibration | Calibration of the split-conformal (CQR) intervals re-centered on Tier 3. (a) Empirical coverage versus nominal level with the identity line; (b) mean interval width versus nominal level, stratified by concentration. Calibrated intervals let downstream health analyses propagate exposure uncertainty instead of treating fields as exact. | Artifacts: `quantile_oof.npz`, `oof_meta.npz` | renders-now (preview) | Main | No |
| F14 | `f14_ablation_deltas` | Feature-group ablation | Ablation of the input configuration: change in LOSO R² for plus_demographics, no_external, and no_neighbor relative to the primary configuration, with 95% bootstrap CIs on the deltas. Demographic variables are excluded from the primary model by design; this panel shows their inclusion is not what drives skill, while external CTM and neighbor features carry measurable signal. | Artifacts: `metrics_ablation.json` | renders-now (preview) | SI | No |
| F15 | `f15_event_strata` | Performance by event stratum | Performance stratified by event type: smoke-impacted, dust-impacted, and clean days. Grouped bars show per-stratum R² (with RMSE in a companion panel) and bootstrap CIs for Tier 3 and the interpolation baselines. Event days are exactly where interpolation degrades and multi-source fusion pays off. | Artifacts: strata blocks in `metrics_loso.json`; event flags from `training_frame.parquet` | renders-now (preview) | SI | No |
| F16 | `f16_r2_decomposition` | Spatial vs temporal R² decomposition | Decomposition of skill into spatial (between-site) and temporal (within-site) components for each model. Separating the two shows whether a model merely reproduces the network's long-term spatial pattern or also tracks day-to-day variation at a fixed location — the component that matters for acute-exposure epidemiology. | Artifacts: decomposition fields in `metrics_loso.json` / `metrics_spatial_block.json` | renders-now (preview) | SI | No |
| F17 | `f17_feature_importance` | Feature importance (SHAP + permutation) | Feature importance from two independent views: (a) SHAP summary for the Tier-1 blend, restyled to the shared palette; (b) grouped permutation importance evaluated on held-out folds. Agreement between attribution methods identifies the dominant drivers [ordering filled from full run]. | Artifacts: `permutation_report.json` (panel b renders now); panel (a) needs SHAP values exported by the full run — the quick run saved only the rasterized `shap_summary.png` | post-Colab (panel b previews now) | SI | No |
| F18 | `f18_unet_attention` | U-Net source attention: smoke vs clean day | Per-source softmax attention from the FusionUNet on a smoke-impacted day versus a clean day. Columns show each source group's attention surface on a shared viridis scale; on smoke days weight shifts toward the smoke and aerosol channels, evidence the fusion is physically interpretable rather than a black box. | Artifacts: `unet/fusion_unet_best.pt` + gridded inputs from `research/aqnet/cache/` (inference via `models_deep.py`/`grids.py`) | renders-now (preview; smoke-window days only) | Main | Yes |
| F19 | `f19_exposure_surfaces` | Example daily exposure surfaces | Example daily PM2.5 exposure surfaces from the full three-tier pipeline for contrasting days (a regional smoke event; a clean winter day), rendered in viridis with tract boundaries as light-gray context and PurpleAir observations overplotted as circles on the identical color scale (colorbar "PM2.5 (ug/m3)"). Close agreement between circles and field indicates spatial fidelity beyond the sensor pixels. | Artifacts: full Tier-3 gridded surfaces (post-Colab); preview from `unet/fusion_unet_best.pt` + `cache/` grids for the smoke window | post-Colab (watermarked preview possible) | Main | Yes — visual hero |
| F20 | `f20_morans_i` | Residual spatial autocorrelation | Distribution of per-day Moran's I computed on model residuals at sensor locations, for Tier 3 versus interpolation baselines. Residual autocorrelation concentrated near zero indicates the model has absorbed the spatially structured signal rather than leaving it in the errors, supporting use of the surfaces at unmonitored tracts. | Artifacts: `oof_tier1.npz`, `oof_meta.npz`; coords from `pipeline/purpleair_full_dataset.parquet` | renders-now (preview) | SI | No |
| F21 | `f21_unet_learning_curves` | FusionUNet learning curves | FusionUNet optimization under the AdamW warmup+cosine schedule. (a) Training and validation masked-Huber loss per epoch; (b) validation R² at sensor pixels, with the selected best-validation checkpoint marked. Curves diagnose fit quality of the sparse-supervision objective. | Artifacts: `unet_train.json` | renders-now (preview) | SI | No |
| F22 | `f22_aqi_confusion_pr` | AQI categories and exceedance detection | Public-health utility of the fields. (a) Confusion matrix of daily AQI categories derived from predicted versus observed PM2.5; (b) precision–recall curve for exceedance detection at the 24-h 35 µg/m³ threshold. Category-level fidelity, not continuous R² alone, determines usefulness for advisories and health-alert applications. | Artifacts: `oof_tier1.npz`, `oof_meta.npz` (Tier-3 OOF + observations) | renders-now (preview) | SI | Yes — panel (a) only |

## Placement summary

- **Main text (9):** F01, F05, F08, F09, F10, F12, F13, F18, F19.
- **SI (13):** F02, F03, F04, F06, F07, F11, F14, F15, F16, F17, F20, F21, F22.
- **Poster (7):** F01 (banner), F05, F08 (hero), F12a, F18, F19 (visual hero),
  F22a — all regenerated with `set_style("poster")` /
  `save_fig(..., mode="poster")` at `FIG_W["poster"]` width.

## Regeneration

1. Quick pass (today): run each figure script with `preview=True`; outputs land
   watermarked in `figures/preview_quick/` (gitignored).
2. After the full Colab run refreshes `artifacts/`: re-run the same scripts
   with `preview=False`; finals land in `figures/` as `.pdf` + `.png` (300 dpi)
   + `.svg`.
3. Fill every `[bracketed]` caption placeholder from full-run artifacts only.
