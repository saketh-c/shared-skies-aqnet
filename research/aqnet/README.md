# AQNet — Three-Tier PM2.5 Research Model

A publication-grade **offline research track** of the Shared Skies Initiative
that fuses tabular gradient-boosted ensembles, the existing FusionUNet
deep-learning surface model, and geostatistical post-processing into one
rigorously validated PM2.5 estimation stack for Texas. AQNet does **not**
serve the live map — the production site runs the 4-model tree ensemble in
`models/` (LOSO R² = 0.7136, the production baseline; see
`models/metrics.json`).

## The three tiers

```
 Tier 1  tabular GBM ensemble ──▶ per-model out-of-fold predictions
         (LGBM/XGB/CatBoost/RF)     + LOFO simplex blend + quantile heads
                                          │ strictly out-of-fold
 Tier 2  FusionUNet on the gridded ──▶ per-pixel PM2.5 surface, sampled
         0.1° stack (+ GEOS-CF /        at sensor pixels
         MERRA-2 / dust / flag            │
         channels)                        ▼
 Tier 3  residual kriging of Tier-1 errors + cross-fit ridge meta-learner
         over the OOF parts + CQR-style conformal prediction intervals
```

- **Tier 1** cross-validates LightGBM, XGBoost, CatBoost, and Random Forest
  with GroupKFold over sensors (leave-one-sensor-out ethos). Neighbor
  features are recomputed **per fold** from train-fold pools, so no test
  sensor ever feeds its own fold's neighbor aggregates. The headline blend
  is **leave-one-fold-out (LOFO)**: each fold's simplex weights are fit on
  the other folds' OOF rows only (the pooled-OOF blend is also reported).
  LightGBM quantile heads (q05/q50/q95) provide the raw interval band.
  XGBoost and CatBoost switch to their GPU modes automatically when CUDA
  is visible.
- **Tier 2** reuses `research/deeplearning`'s `FusionUNet` (per-source
  spatial-attention fusion → U-Net, masked sparse supervision at sensor
  pixels) on an extended channel stack — GEOS-CF, MERRA-2, a CAMS dust
  plane, day-of-week encodings, and per-source binary **flag channels**
  that mark which source groups had real data each day (flag-and-fill, so
  the attention can learn day-conditional trust). Training uses AMP on GPU,
  AdamW with linear warmup then cosine decay, a masked Huber loss, random
  training crops, and modality dropout. Checkpoints stay compatible with
  `research/deeplearning/export_surface.py`.
- **Tier 3** krigs the Tier-1 residuals (train-fold residuals only), then
  **cross-fits** a ridge meta-learner over the OOF parts (Tier-1 LOFO
  blend, kriged residual, U-Net pixel prediction) with grouped K-fold over
  the meta-training sensors, so the meta prediction is itself fully
  out-of-fold. Prediction intervals recenter the quantile band on the
  Tier-3 prediction and calibrate a CQR-style conformal widening (Romano
  et al., 2019) on calibration sensors disjoint from meta training.

## Quickstart

**Colab (recommended):** open `colab_shared_skies_aqnet.ipynb`, switch to a
GPU runtime, and run top to bottom. The notebook clones the repo, optionally
symlinks its output directories into Google Drive (so a lost session costs
nothing), installs `requirements.txt`, optionally logs into NASA Earthdata
for MERRA-2, runs a `--quick` smoke test, hard-checks the GPU, then runs the
full stage-by-stage pipeline and renders every metrics artifact at the end.

**Locally, from the repo root:**

```bash
pip install -r research/aqnet/requirements.txt

# End-to-end smoke test: 3-month window, 0.2° grid, 4 folds, 3 epochs
python research/aqnet/pipeline_colab.py all --quick --skip-merra2

# Full run (GPU recommended for the deep stage)
python research/aqnet/pipeline_colab.py data
python research/aqnet/pipeline_colab.py features
python research/aqnet/pipeline_colab.py tabular
python research/aqnet/pipeline_colab.py deep
python research/aqnet/pipeline_colab.py fuse
python research/aqnet/pipeline_colab.py ablation
python research/aqnet/pipeline_colab.py validate
```

Flags: `--quick` (smoke test), `--skip-merra2` / `--skip-geoscf` (skip an
external fetch), `--skip-ablation` (drop the ablation stage from `all`),
`--epochs N`, `--batch-size N` (deep stage; default auto — 32 on CUDA, 8 on
CPU), `--grid-deg D`, `--correction barkjohn|raw`. Stages are restartable —
each reads only what earlier stages wrote to `artifacts/`, and every stage
prints what it skipped and why.

**Window-aware caches (quick-then-full is safe).** Final external-data
caches embed the requested date window in the filename (e.g.
`geoscf_pm25_20210101_20260501.parquet`) and are validated to actually
cover that window before reuse; per-month chunks are the cache of record,
so a full run after a `--quick` run reassembles from chunks rather than
trusting a 3-month file. Months that failed to fetch are recorded in a
`.failed.json` sidecar next to the cache and retried on the next run
instead of being silently baked in.

## Files

| File | Purpose |
|---|---|
| `config.py` | Paths, Texas bbox/grid, date window, feature lists (physical + `dust` extra; MERRA-2 incl. PBLH max/min), `artifact()` helper |
| `corrections.py` | Barkjohn et al. (2021) PurpleAir correction; `raw` kept as a sensitivity option |
| `data_external.py` | EPA AQS daily PM2.5 (event/completeness-filtered), GEOS-CF OPeNDAP, MERRA-2 via earthaccess — window-stamped caches + failed-month sidecars |
| `features.py` | Sensor-day training frame, leave-self-out neighbor features + per-fold recompute, external CTM joins |
| `validation.py` | Fold builders, metrics + cluster bootstrap CIs, per-day Moran's I, strata metrics, spatial/temporal R² split, AQI-category skill, baselines, external AQS validation |
| `models_tabular.py` | Tier 1: model registry, LOSO `train_cv` with per-fold neighbor overrides, LOFO + pooled simplex blends, quantile heads, GPU boosters, full-data fit |
| `grids.py` | Extends `research/deeplearning/dataset.py`'s stack with GEOS-CF/MERRA-2/dust channels, day-of-week planes, and per-source availability flags |
| `models_deep.py` | Tier 2: FusionUNet training wrapper (AMP, warmup + cosine LR, masked Huber, random crops, modality dropout) + per-row pixel predictions |
| `fusion.py` | Tier 3: residual kriging, cross-fit ridge stacking, CQR-style conformal intervals |
| `interpret.py` | SHAP summary + permutation importance (guarded for missing libs) |
| `pipeline_colab.py` | Stage-based CLI: `data / features / tabular / deep / fuse / ablation / validate / all` |
| `colab_shared_skies_aqnet.ipynb` | One-click Colab runner (Drive mount, GPU gate) + results display |

Artifacts land in `research/aqnet/artifacts/` (training frame, frozen folds,
per-fold neighbor overrides, OOF arrays, checkpoints, `metrics_*.json`
including `metrics_ablation.json`, `shap_summary.png`, and an
auto-generated `SUMMARY.md` that tabulates only computed numbers).

## Data sources

| Source | File / access | Role |
|---|---|---|
| PurpleAir sensor-days (2021-01..2026-05, 467 sensors) | `pipeline/purpleair_full_dataset.parquet` | Training target (Barkjohn-corrected) + sensor meteorology |
| CAMS AOD / PM2.5 / dust (from 2022-08-03) | `pipeline/airquality_by_cell.parquet` | Aerosol features/channels — `aod`, `cams_pm25`, and `dust` (tabular feature **and** grid channel) |
| ERA5 extras (shortwave, ET0, cloud cover) | `pipeline/met_extra_by_cell.parquet` | Meteorology features/channels |
| NOAA HMS smoke tiers | `pipeline/hms_smoke_by_sensor.parquet` | Smoke feature/channel (also defines the smoke stratum) |
| Elevations, EJScreen physical source-proximity, tract lookup | `pipeline/elevations.json`, `backend/static/tract_lookup.parquet`, `ejscreendata.xls` | Static features |
| **EPA AQS daily FRM/FEM PM2.5** | public zips via `data_external.fetch_aqs_daily_tx` | **External validation only — never trained on** (exceptional-event rows dropped; sub-daily rows require ≥75% observation completeness) |
| **NASA GEOS-CF** (GEOS-Chem chemistry) | public OPeNDAP via `fetch_geoscf_pm25` | CTM prior feature + Tier-2 channel + baseline (raw and mean-debiased) |
| **MERRA-2 aerosol species + PBLH (daily mean/max/min)** | earthaccess (free Earthdata login) via `fetch_merra2` | Optional features/channels; PBLH max/min are tabular-only; NaN when unavailable |

## Methodology (what makes it publishable)

- **No demographic model inputs.** `ejf_score`, `pct_people_of_color`,
  `pct_low_income`, and `pct_ling_isolated` are excluded from prediction
  everywhere (`features.feature_columns` asserts this). Physical EJScreen
  source-proximity features (traffic, Superfund, RMP, diesel PM) are kept.
  A `plus_demographics` ablation variant quantifies what the exclusion
  costs — it exists **only** as an ablation, never as a reported model.
- **Corrected target.** PurpleAir ATM readings are corrected per Barkjohn
  et al. (2021, AMT 14:4617): `pm25 = 0.524·atm − 0.0862·RH + 5.75`, clipped
  at 0. `--correction raw` re-runs everything on the raw channel as a
  sensitivity analysis.
- **External validation is external.** EPA AQS monitors never enter
  training or feature computation for training rows; they are only ever
  predicted against. The AQS table itself is quality-filtered
  (exceptional-event exclusions, observation-completeness threshold).
- **Leakage discipline.** Neighbor features are leave-self-out, same-day
  only, and recomputed per CV fold from train-fold pools. The headline
  blend weights are leave-one-fold-out. The Tier-3 meta-learner is
  cross-fit over sensor groups so its own prediction is out-of-fold;
  residual kriging uses train-fold residuals only; conformal calibration
  uses a sensor-disjoint split from meta training, and coverage/width are
  reported on rows disjoint from that calibration.
- **Validation battery.** LOSO GroupKFold, spatial-block (region) CV,
  temporal holdout (train < 2025-01-01), external AQS — each with
  **cluster (per-sensor) bootstrap CIs**, per-day residual Moran's I
  (summarized as mean/median/IQR over days), smoke/dust/clean strata
  metrics, a spatial-vs-temporal R² decomposition, and EPA 2024
  AQI-category skill — against nearest-sensor, IDW, ordinary-kriging, and
  raw + mean-debiased CTM baselines. External AQS additionally gets a
  **deployment-mode Tier-3 row**: full-data Tier-1 at AQS site-days,
  per-day kriging of full-data training residuals to the AQS sites, and
  the U-Net surface value at each site's pixel, combined by the final
  ridge.
- **Ablation on frozen folds.** The `ablation` stage re-runs Tier-1 on the
  same frozen LOSO folds and target for `primary`, `plus_demographics`,
  `no_external` (drop `geoscf_*`/`merra2_*`), and `no_neighbor` (drop
  `nbr_*`), reporting per-variant metrics plus paired ΔR² vs `primary`
  with cluster bootstrap CIs (`metrics_ablation.json`).
- **No invented numbers.** Nothing in this directory quotes an AQNet
  accuracy figure; `SUMMARY.md` is generated from computed metrics only.
  The only citable number is the production ensemble's LOSO R² = 0.7136,
  which describes the live system, not AQNet.

## Expected runtimes

Network speed dominates the data stage; the CPU long poles are `tabular`,
`ablation`, and `validate`. GPU boosters (XGBoost/CatBoost GPU modes, AMP
in the deep stage) enable automatically when CUDA is visible.

- `data` — minutes for AQS zips + GEOS-CF; the **first full-window MERRA-2
  fetch can take hours** (bulky hourly granules). Month chunks are cached
  and final files are window-stamped, so you pay that fetch once.
- `features` — minutes to tens of minutes (BallTree neighbor features over
  ~400K rows, plus the per-fold neighbor recompute).
- `tabular` — **hours**: 4 models × 10 LOSO folds with per-fold neighbor
  overrides, plus quantile heads. LightGBM/RF stay CPU-bound; XGBoost and
  CatBoost use the GPU when available.
- `deep` — roughly **1–1.5 h on a Colab T4** at 0.1° with AMP and the
  larger auto-selected batch size (`--batch-size N` to override). CPU is
  impractical for the full window; `--quick` (0.2°, 3 epochs) finishes in
  minutes.
- `fuse` — tens of minutes (per-day kriging + cross-fit meta-learner +
  conformal calibration).
- `ablation` — **hours** (a Tier-1 re-run per variant on the frozen folds);
  `--skip-ablation` drops it from `all`.
- `validate` — **hours**: spatial/temporal CV retrains, baselines including
  per-day kriging, strata and per-day Moran's I, external AQS with the
  deployment-mode Tier-3 row, SHAP.

## Status — honest

- Code-complete research track; syntax-checked, designed to run on Colab.
- No AQNet accuracy numbers are quoted anywhere in this directory because
  none have been finalized — run the pipeline and read
  `artifacts/SUMMARY.md` for the numbers your run actually produced.
- The production live map is unaffected: it continues to serve the
  4-model tree ensemble.
