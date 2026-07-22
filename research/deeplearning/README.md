# Deep-Learning Research Track — Fusion U-Net PM2.5 Surfaces

An experimental track of the Shared Skies Initiative that learns **continuous
PM2.5 fields** for Texas instead of per-tract point predictions. A
multi-source attention-fusion U-Net ingests daily gridded satellite,
smoke, meteorology, terrain, and seasonal channels and outputs a full
PM2.5 surface, supervised only at the sparse pixels where PurpleAir
sensors exist.

## Architecture

```
 aerosol ─┐
 smoke   ─┤   per-source        per-pixel softmax           U-Net (depth 4,
 met     ─┼─▶ conv encoders ──▶ attention across sources ──▶ GroupNorm, bilinear ──▶ PM2.5
 static  ─┤   (shared embed)    + squeeze-excite fusion      upsampling, softplus)   surface
 temporal─┘                          │
                                     └──▶ attention maps (which source the
                                          model trusts at each pixel)
```

- **SpatialAttentionFusion** — each source group gets its own small conv
  encoder into a shared embedding space; a softmax across sources at every
  pixel weights the embeddings, and a squeeze-excite block re-weights the
  fused channels. The attention maps are returned for interpretability.
- **UNet** — standard encoder-decoder with skip connections, depth 4,
  GroupNorm (stable at small batch sizes), bilinear upsampling, and a
  softplus head so predicted PM2.5 is non-negative.
- **FusionUNet** — fusion module feeding the U-Net.

## Input channels

All channels come from files the main pipeline already produces — no new
data pulls. Grid: 0.1° lat/lon over the Texas tract extent (bounds from
`backend/static/tract_lookup.parquet`).

| Group | Channels | Source | Gridding |
|---|---|---|---|
| aerosol | aod, cams_pm25 | `pipeline/airquality_by_cell.parquet` (0.5° CAMS) | nearest cell |
| smoke | hms_smoke (0–3 tier) | `pipeline/hms_smoke_by_sensor.parquet` (NOAA HMS) | nearest sensor |
| meteorology | temperature, humidity, pressure, wind_speed, precipitation | `pipeline/purpleair_full_dataset.parquet` (sensor-day) | IDW (k=8) |
| meteorology | shortwave, et0, cloud_cover | `pipeline/met_extra_by_cell.parquet` (0.5° ERA5) | nearest cell |
| static | elevation | `pipeline/elevations.json` (tract centroids) | IDW (k=4) |
| temporal | doy_sin, doy_cos | derived from date | constant planes |

Supervision: raw PurpleAir ATM-channel PM2.5 at sensor pixels (in-Texas
sensors only, per `pipeline/sensor_tx_membership.csv` — the same convention
the production ensemble uses for training targets).

## Files

| File | Purpose |
|---|---|
| `models.py` | UNet, SpatialAttentionFusion, FusionUNet |
| `dataset.py` | Builds the daily gridded stack + sparse supervision records; `.npz` caching |
| `train.py` | Masked-MSE training with grouped **site** holdout; R2/RMSE/MAE reporting; checkpoints |
| `export_surface.py` | Runs a checkpoint over a date range, writes `.npz` surface + JSON manifest |

## How to run

From the repo root:

```bash
# 1. Install (torch per your platform/CUDA — see pytorch.org)
pip install -r research/deeplearning/requirements.txt

# 2. Build the gridded dataset cache (a few minutes; ~1.5 GB in RAM at 0.1°)
python research/deeplearning/dataset.py --out research/deeplearning/cache/texas_grid.npz

# 3. Train (GPU recommended; auto-detects CUDA)
python research/deeplearning/train.py --epochs 40 --lr 1e-3 --holdout-frac 0.2 \
    --cache research/deeplearning/cache/texas_grid.npz

# 4. Export predicted surfaces for a date range
python research/deeplearning/export_surface.py \
    --checkpoint research/deeplearning/checkpoints/fusion_unet_best.pt \
    --start 2025-06-01 --end 2025-06-30 \
    --out research/deeplearning/surfaces/pm25_june2025
```

On Colab: upload/clone the repo, `pip install -r research/deeplearning/requirements.txt`,
and run the same commands with a GPU runtime. For quick CPU experiments,
use `--grid-deg 0.2` and a shorter `--start/--end` window in both
`dataset.py` and `train.py`.

## Design notes

- **Sparse supervision.** The loss is masked MSE evaluated only at grid
  pixels containing sensors; the network still predicts every pixel, so the
  spatial structure of the inputs is what generalizes the surface between
  sensors.
- **Grouped site holdout.** Validation holds out whole sensor *sites*
  (grid pixels and every sensor sharing them, across all days) — the same
  leave-one-site-out ethos as the production ensemble's GroupKFold-over-sensors
  CV. Random day splits are deliberately not offered because they leak
  spatial information.
- **No train/serve skew.** Fill values and normalization statistics are
  computed once at training time and stored in the checkpoint;
  `export_surface.py` reuses them verbatim.
- **Data coverage.** CAMS aerosol history starts 2022-08-03 (the Open-Meteo
  archive limit); earlier days get mean-filled aerosol channels. The CAMS
  `dust` column exists but is not currently used as a channel.

## Status — honest

- This is the **deep-learning research track** of Shared Skies.
- **Code-complete**: models, dataset builder, training, and export are
  implemented and syntax-checked.
- **Training runs on GPU** (e.g., Google Colab); CPU works but is slow at
  the full 0.1° resolution.
- The **production live map currently serves the 4-model tree ensemble**
  (Random Forest, LightGBM, XGBoost, CatBoost). **U-Net surfaces are not
  yet served live.**
- No accuracy numbers are quoted here because none have been finalized for
  this track; `train.py` prints R2/RMSE/MAE on held-out sensor sites for
  any run you do.
