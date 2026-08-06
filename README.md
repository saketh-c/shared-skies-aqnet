# Shared Skies Initiative - AQNet

Historical PM2.5 reconstruction for Texas: a research framework that fuses
PurpleAir low-cost sensors, chemical-transport model output (CAMS, GEOS-CF),
MERRA-2 reanalysis, meteorology, smoke plumes, and high-resolution static
covariates into daily ground-level PM2.5 fields at 0.1° resolution.

This is the **offline research track** of the Shared Skies Initiative. The live
real-time map is a separate project and a separate repository
([shared-skies-live-map](https://github.com/saketh-c/shared-skies-live-map)).
AQNet does not serve it, and is deliberately free to use data sources whose
latency makes them impractical in a live serving loop.

**The current system is AQNet v2** (`research/aqnet2/`). The original v1
(`research/aqnet/`) is superseded: its final full-scale validation showed the
architecture did not deliver, and every v2 design decision cites that evidence.
The original v1 README is preserved, clearly marked as outdated, at the bottom
of this file.

## Where things stand (August 6, 2026)

**v1 is finished and its verdict is in.** The definitive leak-free full run
(Aug 2026, GT PACE Phoenix) is committed in
[`results/phoenix_202608/`](results/phoenix_202608/SUMMARY.md):

- Pooled leave-one-sensor-out R² = **0.597** (Tier-1 blend); the three-tier
  stack scored *lower* (0.583), and a paired test showed the stack degraded
  accuracy (ΔR² ≈ −0.05, CI excluding zero). The deployed production 4-model
  tree ensemble it was meant to improve on reports LOSO R² = 0.714.
- The pooled headline is ~93% temporal signal. Between-site **spatial R² = 0.05**;
  permutation analysis traced within-network spatial skill to site-fingerprint
  memorization.
- On fully held-out EPA FRM/FEM monitors: **R² = −0.06** with −3.6 µg/m³ bias,
  drifting worse year over year — a static PurpleAir correction cannot anchor
  the target scale to truth.
- The novel components were the weakest parts: residual kriging earned a ridge
  weight of exactly 0.000, and the FusionUNet (R² 0.41) underperformed ordinary
  kriging (0.54). Exceedance-day recall collapsed to ~0 through the stack.

These failure modes are invisible to the pooled cross-validation commonly used
in the low-cost-sensor literature — that finding, and the mechanisms behind it,
are v1's real contribution.

**v2 is complete: the full production run is validated and shipped** (GT PACE
Phoenix, Aug 2026). Everything is in
[`results/v2_texas_202608/`](results/v2_texas_202608/RESULTS.md) — model card,
all validated metrics artifacts, and the 12-figure set. Headlines:

- **Held-out FRM sites** (62,223 site-days, 1000-resample CIs): composite
  R² = **0.33** [0.20–0.46], RMSE 5.2 µg/m³ — more than double the debiased
  chemical-transport floor (0.15). Pooled leave-site-out diagnostic
  R² = 0.52 / RMSE 2.9. Smoke-day stratum R² = 0.44 at +0.003 bias.
- **The one-shot vault held**: the 12 buffered sites nothing ever touched
  scored R² = **0.335** — matching the cross-validation estimate almost
  exactly. The headline numbers are not fold-tuning artifacts.
- **Calibration study** (201,230 held-out sensor–monitor pair-days,
  2021–2026): published national PurpleAir correction constants carry a
  **−4.5 µg/m³ Texas bias**; the shipped Texas-refit RH+T form cuts
  held-out-site bias to +0.13 µg/m³. The pre-registered gate rejected a
  learned nonparametric form that improved RMSE but tripled bias.
- **The admission gates did their job**: both deep residual tiers (graph
  attention, neural field) showed positive point estimates that could not be
  statistically confirmed at 49 site-clusters — so neither ships. The served
  model is the physics prior + gradient-boosted ensemble + daily kriging,
  and every "fail" verdict is recorded in the artifacts, not hidden.
- The calibrated-PurpleAir neighbor covariates are the single most important
  feature family (permutation importance ~10× the next feature) — the
  FRM-anchored, sensors-as-covariates inversion is the design bet that paid.

**v3 is in progress** (`research/aqnet2/EXPANSION.md`): the WEST7 domain
expansion (CA/TX/WA/CO/UT/NV/AZ, ~285 FRM sites) built to give the deep tiers
an admission test with real statistical power — the Texas run could detect
effects of ~0.08 R²; WEST7 detects ~0.03. Phase 1 runs without new PurpleAir
acquisition (Texas keeps its calibrated archive; new states run the portable
feature set). Same codebase, same gates, domain-switched
(`AQNET2_DOMAIN=west7`), artifacts under `v3`.

## AQNet v2 — design summary

v2 is not v1 with bugs fixed; it is the architecture the v1 evidence specifies.
[`research/aqnet2/CHANGES.md`](research/aqnet2/CHANGES.md) lists all 30 changes,
each citing the v1 measurement that forced it. Full design in
[`research/aqnet2/DESIGN.md`](research/aqnet2/DESIGN.md) and module contracts in
[`research/aqnet2/INTERFACES.md`](research/aqnet2/INTERFACES.md). Highlights:

- **Truth-anchored target.** EPA AQS FRM/FEM anchors training; PurpleAir becomes
  a calibrated, variance-weighted covariate/pseudo-label stream via a learned
  Kennedy–O'Hagan-style calibration (per-sensor random effects, sensor-year
  drift, smoke interactions), gated against Barkjohn and AMT-2024 forms.
- **Honest validation by construction.** Nested spatially-blocked site folds, a
  12-site one-shot vault, a bare-site scoring arm, 7-day temporal embargo, and
  per-fold neighbor recompute everywhere. Pooled LOSO is demoted to a labeled
  diagnostic; primary metrics are held-out-site accuracy, between-site R²,
  exceedance detection, and interval coverage.
- **Admission-gated composition.** The late ridge stack is deleted. Components
  join an additive residual ladder only by passing pre-registered,
  power-calibrated admission gates (cross-fit α per coverage pattern,
  selection/confirmation split, one-sided paired cluster-bootstrap); mean-fill
  is unrepresentable by design.
- **Portable spatial skill.** Raw lat/lon and distance-to-sensor removed from
  tabular features; high-resolution static covariates (DEM, roads, NEI
  emissions, population) added; a masked graph-attention interpolator with
  airshed/wind-conditioned edges replaces naive neighbor averaging; off-support
  predictions degrade visibly to a debiased-CTM floor.
- **Honest uncertainty and tails.** Unit-level weighted conformal intervals
  (calibration units disjoint from selection), epistemic spread from
  fold-assignment ensembles, and a dedicated cross-fit exceedance classifier.
- **No demographic model inputs**, as in v1: demographic variables are excluded
  from prediction everywhere to avoid circularity in downstream
  environmental-justice analysis.

## Repository layout

| Path | What it is |
|---|---|
| `research/aqnet2/` | **Current system (v2)** — modules, design docs, change log |
| `slurm2/` | v2 PACE Phoenix job chain |
| `results/` | Committed run results — `v2_texas_202608/` is the shipped v2 run (model card, metrics, figures, Dallas 1 km demo); `phoenix_202608/` is the definitive v1 run |
| `research/aqnet/` | v1 (superseded) — kept intact for reproducibility |
| `research/deeplearning/` | FusionUNet track (v1-era; benchmark-only in v2) |
| `slurm/` | v1 PACE job chain |
| `pipeline/` | Training data shipped with the repo |

## Authors

AQNet was architected and developed by
**[@saketh-c](https://github.com/saketh-c)** and
**[@nathantantexas](https://github.com/nathantantexas)**.

Part of the Shared Skies Initiative, which also produces the live real-time
PM2.5 map at
[sharedskiesinitiative.org](https://sharedskiesinitiative.org/real-time-map).

This repository was split out of the main Shared Skies repository so the
research track could continue as its own project; the original AQNet commit
history remains in
[shared-skies-live-map](https://github.com/saketh-c/shared-skies-live-map).

---

# Archived: original v1 README

> **⚠️ OUTDATED — kept for historical reference only.** Everything below
> describes AQNet v1 (`research/aqnet/`), which has been superseded by v2 (see
> above). In particular, the "Status" claim that no accuracy numbers exist in
> the repository is no longer true: the definitive v1 numbers are committed in
> [`results/phoenix_202608/`](results/phoenix_202608/SUMMARY.md), and they are
> the reason v1 was retired.

Historical PM2.5 reconstruction for Texas: a three-tier fusion research model that
estimates daily ground-level PM2.5 at 0.1° resolution across the state.

### The three tiers

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

### Quickstart

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

### Methodology notes

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

### Reconstruction window

| Window | Quality |
|---|---|
| Aug 2022 → May 2026 | Full — all channels available |
| Jan 2021 → Aug 2022 | Good — CAMS aerosol absent, marked by availability-flag channels |
| Before 2021 | Not supported by the current data pull |

The binding constraint is PurpleAir sensor density (93 sensors statewide in 2021Q1,
439 by 2025Q4), not the covariates.

### Data inputs

Training data ships with the repo under `pipeline/`. External sources — EPA AQS,
NASA GEOS-CF, and MERRA-2 (free Earthdata login) — are fetched at runtime and cached;
`--skip-geoscf` / `--skip-merra2` let a run proceed without them.

### Status (historical)

Code-complete and verified to run end-to-end. ~~No AQNet accuracy numbers are quoted
anywhere in this repository because none have been finalized~~ — *superseded: final
v1 numbers are now committed in `results/phoenix_202608/`.*
