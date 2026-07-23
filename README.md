# Shared Skies Initiative — AQNet

Historical PM2.5 reconstruction for Texas: a three-tier fusion research model that
estimates daily ground-level PM2.5 at 0.1° resolution across the state.

This is the **offline research track** of the Shared Skies Initiative. The live
real-time map is a separate project and a separate repository
([shared-skies-live-map](https://github.com/saketh-c/shared-skies-live-map)). AQNet does not serve
it, and is deliberately free to use data sources (GEOS-CF OPeNDAP, MERRA-2 reanalysis,
per-day kriging) whose latency makes them impractical in a live serving loop.

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

See [`research/aqnet/README.md`](research/aqnet/README.md) for the full design and
[`research/aqnet/REPORT.md`](research/aqnet/REPORT.md) for the methodology writeup.

## Quickstart

```bash
pip install -r research/aqnet/requirements.txt

# End-to-end smoke test (3-month window, 0.2° grid, 4 folds, 3 epochs)
python research/aqnet/pipeline_colab.py all --quick --skip-merra2 --skip-geoscf

# Full run, stage by stage
python research/aqnet/pipeline_colab.py data
python research/aqnet/pipeline_colab.py features
python research/aqnet/pipeline_colab.py tabular
python research/aqnet/pipeline_colab.py deep --device mps   # or cuda / cpu
python research/aqnet/pipeline_colab.py fuse
python research/aqnet/pipeline_colab.py ablation
python research/aqnet/pipeline_colab.py validate
```

Artifacts land in `research/aqnet/artifacts/` — `metrics_*.json` plus an
auto-generated `SUMMARY.md` that tabulates only computed numbers.

**Colab:** open `research/aqnet/colab_shared_skies_aqnet.ipynb`, switch to a GPU
runtime, and run top to bottom.

## Methodology notes

- **No demographic model inputs.** `ejf_score`, `pct_people_of_color`,
  `pct_low_income`, and `pct_ling_isolated` are excluded from prediction everywhere,
  to avoid circularity in downstream environmental-justice analysis. Physical
  source-proximity features (traffic, Superfund, RMP, diesel PM) are retained.
- **Corrected target.** PurpleAir ATM readings are corrected per Barkjohn et al.
  (2021): `pm25 = 0.524·atm − 0.0862·RH + 5.75`, clipped at 0. `--correction raw`
  re-runs on the raw channel as a sensitivity analysis.
- **External validation is external.** EPA AQS FRM/FEM monitors never enter training
  or feature computation — they are only ever predicted against.
- **Corrected-target scores are not comparable to raw-target scores.** The correction
  compresses target variance 4.03× (69.71 → 17.32 µg²/m⁶) and the feature and sensor
  sets differ. Use `--correction raw` for any like-for-like comparison.

## Reconstruction window

| Window | Quality |
|---|---|
| Aug 2022 → May 2026 | Full — all channels available |
| Jan 2021 → Aug 2022 | Good — CAMS aerosol absent, marked by availability-flag channels |
| Before 2021 | Not supported by the current data pull |

The binding constraint is PurpleAir sensor density (93 sensors statewide in 2021Q1,
439 by 2025Q4), not the covariates.

## Data inputs

Training data ships with the repo under `pipeline/`. External sources — EPA AQS,
NASA GEOS-CF, and MERRA-2 (free Earthdata login) — are fetched at runtime and cached;
`--skip-geoscf` / `--skip-merra2` let a run proceed without them.

## Status

Code-complete and verified to run end-to-end. No AQNet accuracy numbers are quoted
anywhere in this repository because none have been finalized — run the pipeline and
read `research/aqnet/artifacts/SUMMARY.md` for the numbers your run produced.

## Authors

AQNet (`research/aqnet`) and the FusionUNet deep-learning track
(`research/deeplearning`) were architected and developed by **[@saketh-c](https://github.com/saketh-c)** and **[@nathantantexas](https://github.com/nathantantexas)**.

Part of the Shared Skies Initiative, which also produces the live real-time PM2.5 map at
[sharedskiesinitiative.org](https://sharedskiesinitiative.org/real-time-map).

This repository was split out of the main Shared Skies repository so the research
track could continue as its own project; the original AQNet commit history remains in
[shared-skies-live-map](https://github.com/saketh-c/shared-skies-live-map).
