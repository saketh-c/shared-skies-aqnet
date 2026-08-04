# AQNet v2 — Design

FRM-anchored multi-fidelity residual ladder with a structurally monotone
composition harness, learned PurpleAir calibration, graph-attention and
pretrained-field residual tiers, and a continuous point decoder.

This document is the implementation contract for `research/aqnet2/`. It was
synthesized from a four-way independent architecture tournament, each design
adversarially red-teamed and scored by a three-judge panel, on top of a
quantitative failure diagnosis of v1's two full runs (`results/`). Every design
decision below traces to a measured v1 defect or a named methodological
requirement; the tournament's critique fixes are baked in as the design, not
appended as amendments.

## 0. What v1's evidence established (the constraints this design answers)

1. **Target miscalibration dominates.** v1 trained on Barkjohn-corrected
   PurpleAir and scored R² ≈ 0 at the 62 EPA AQS FRM/FEM monitors with a
   −2.7 to −3.8 µg/m³ bias (bias² = 21–32% of AQS MSE), worsening
   monotonically 2021→2025. The Barkjohn coefficients are cf_1-channel
   constants applied to an ATM-channel archive; drift and smoke-regime
   failure compound it. A perfect interpolator of a miscalibrated field
   scores ~0 at truth.
2. **No between-site skill.** Spatial (between-site) R² = 0.049 on the
   FRM-comparable scale (0.18 raw, partly humidity-artifact prediction).
   ~60% of Tier-1 signal is neighbor interpolation, ~24% static site
   fingerprints (raw lat/lon among top features). Pooled LOSO R² is ~93%
   temporal signal and must never again drive selection.
3. **The fusion regression was a wiring defect, not a fusion verdict.** The
   v1 ridge was fit on the 17% of rows where all components were finite,
   shrank Tier-1's coefficient conditional on the U-Net being informative,
   then mean-filled the U-Net on 83% of rows: paired ΔR² −0.050/−0.031,
   CIs excluding 0; exceedance recall → 0. The kriged-residual weight of
   exactly 0.000 is structural (LOSO residuals are device-nugget-dominated
   because v1 had no per-unit random effect).
4. **Uncertainty was dishonest.** Quantile heads covered 0.66 at nominal
   0.90; conformal calibration treated 77k dependent rows as independent
   (honest n ≈ 85 sensors); one model's band was grafted onto another's
   center. Coverage landed at 0.88.
5. **Resolution was capped by grid snapping.** `rint((lat−lat0)/0.1°)`
   collapsed co-located urban sensors into single supervised pixels.

## 1. System overview

```
S0 audit      channel/provenance audit (no spend before it passes)
S1 calibrate  learned Kennedy–O'Hagan PA→FRM calibration (nested per fold)
T0 priors     EPA-Downscaler-debiased CTM prior — the extrapolation floor
T1 skeleton   GPBoost: GBM fixed effects + Matérn GP + unit/day random
              effects, trained on FRM-scale targets (AQS + calibrated PA)
T2 graphres   masked graph-attention residual (between-site skill)
T3 fieldres   masked-pretrained multimodal field residual + INR point decoder
T4 recal      declared slope-recalibration rung (attenuation fix, b clipped)
   compose    gated additive composition (structural monotonicity)
   exceed     cross-fit exceedance classifier (inside the admission harness)
   uq         heteroscedastic + unit-level NexCP conformal intervals
   validate   outer folds + one-shot vault + bare-site arm
   export     bag-of-fold-models serving bundle, predict(lat, lon, date)
```

The deployed predictor is a residual ladder:

    F_k(q) = F_{k-1}(q) + m_k(q) · α_k[p(q)] · r̂_k(q)

with three structural guarantees, each independently sufficient to prevent a
v1-class regression:

- **Frozen incumbent.** F_{k−1} enters with coefficient exactly 1. No
  combiner anywhere can rescale a previous tier. (T4 is the sole, declared
  exception: an affine recalibration rung whose slope is clipped to
  [0.8, 1.25] and cross-fit; it is documented as a reweighting rung, not
  smuggled through the invariant.)
- **Structural zero, never fill.** m_k(q) ∈ {0,1} is component availability;
  m_k = 0 yields bit-identical passthrough. No fill value exists in the
  composition path; `compose.fit_gate` asserts `np.isnan(r_k[~avail]).all()`
  and `gates.json` cannot express an incumbent coefficient. A synthetic
  17%-finite weak component ships as a **permanent regression test** that
  must produce composite ≡ passthrough on uncovered rows.
- **Cross-fit, power-calibrated admission gates.** α_k[p] ∈ [0,1] per
  coverage pattern p, fit on OOF quantities from the inner **selection**
  half, accepted only by a one-sided paired unit-cluster bootstrap on the
  disjoint inner **confirmation** half (selection and test never share
  rows), with margins sized by a pre-registered power analysis on v1
  residuals. Default is closed. A stratum with fewer than 5 held-out
  cluster units shrinks toward **zero** (passthrough), never toward the
  global α. Hard α = 0 beyond 100 km from any live station.

Worst reachable outcome of the entire ladder: T1. Worst reachable outcome of
T1: the classical baseline it must beat. There is a pre-committed
ship-the-classical branch if nothing passes.

## 2. Fold system (folds2.json — one unified system, every stage nested in it)

Units = 467 PA sensors ∪ 62 AQS sites (unit_id, unit_type).

- **VAULT (the one-shot second sample):** 12 AQS sites, stratified by
  region × urbanicity × mean PM, each ≥ 30 km from every non-vault site,
  plus all data from 2026-01-01 onward. Excluded from calibration pairs,
  training rows, neighbor pools, graph nodes, lag rasters, gate fitting,
  hyperparameters, checkpoint selection, conformal calibration — everything.
  Touched exactly once, by `validate`, after the configuration is frozen.
  `folds2.json` stores the vault site list; the graph/pool builders carry an
  assert against it.
- **OUTER: 5 spatially-blocked folds** over the remaining 50 AQS sites.
  Operational definition: constrained k-means on EPSG:3083 coordinates with
  a max-40%-of-fold-per-CBSA constraint; the assignment map is a committed
  artifact. Every non-vault site is scored exactly once by a chain that
  never touched it (calibration, priors, features, tiers, gates, UQ all
  nested per outer fold).
- **INNER (per outer fold): selection/confirmation split.** Remaining AQS
  sites + all PA sensors → 4 unit-grouped folds; folds 1–2 = selection
  (hyperparameters, λ, α gate fitting), folds 3–4 = confirmation (the
  admission bootstrap). No gate is ever tested on rows that chose it.
  Conformal calibration units are a disjoint 25% PA-sensor set plus one
  confirmation-side AQS fold **never used in admission evaluation**.
- **T1 OOF: 10 unit-grouped LOSO folds nested within each outer fold**
  (stated honestly: 5 × 10 fits; the documented budget escape is the
  GBM + per-day-kriging fallback if joint GPBoost fits stall).
- **Spatial-block folds (5) and temporal holdout** (train < 2025-01-01,
  test ≥, 7-day embargo on all lagged features). Both get per-fold neighbor
  recompute (closing v1's leak where block/temporal reruns used full-pool
  neighbors). Temporal-holdout claims may only be made from the
  temporally-pure pretrain/fine-tune variants (§6, §7).
- `folds2.json` carries a sha256 content hash over (unit_id, date, target);
  every consumer verifies it (row-count checks alone are insufficient).

Colocated-twin rule (bare-site honesty): when scoring any held-out AQS site,
a **bare-site arm** removes all PA units within 5 km from graphs, neighbor
pools, and lag rasters; it is reported alongside the with-network number as a
co-primary metric. It is the only number that answers "can the system rank
unmonitored locations."

## 3. S0 — audit (stage `audit`)

Before any data spend: verify PA channel provenance (the committed parquet has
a single `pm25` column documented as ATM), enumerate the colocation inventory
(measured from committed data: ~8 pairs ≤ 1 km, ~105 ≤ 5 km, ~320 ≤ 10 km,
~1,041 ≤ 25 km), compute the PurpleAir API points cost and wall-clock for the
cf_1 refetch, and run the **gate power analysis**: bootstrap the minimum
detectable Δ (pooled held-out-site R² and between-site R²) at the actual
cluster counts using v1 residuals. Admission margins in §9 are set from this
artifact (`power_analysis.json`), not hand-picked. Also run the **T2
kill-switch probe**: empirical variogram + Moran's I of v1-style T1 OOF
residuals, and kriging-on-residuals through the admission test. If kriging
cannot pass, T2 runs as a research arm only and its GPU budget is re-scoped.

## 4. S1 — learned PA calibration (stage `calibrate`)

Kennedy–O'Hagan multi-fidelity form, daily:

    y_FRM(s,t) = ρ(x_cal) · y_PA(s',t) + δ(x_cal) + b0_j + b1_j · y_PA + g_{j,year} + ε

- Pairs: (AQS site s, PA sensor s′) with d ≤ 10 km primary (25 km
  sensitivity arm), weight exp(−d/5 km). Smoke-day pairs oversampled 5×.
- x_cal: RH, T, dewpoint, hms_smoke tier, doy harmonics, sensor_age_days,
  pair distance, channel_reconstructed flag, FEM-vs-FRM method indicator,
  urban flag. Per-sensor random intercept/slope with partial pooling;
  per-sensor-year drift intercepts.
- Implementation: GPBoost (LightGBM fixed effects + grouped REs, Gaussian
  NLL); fallback: LightGBM linear-tree + MixedLM.
- **Nesting (non-negotiable):** refit per (outer k, inner j) —
  `pa_cal_f{k,j}` columns — excluding all pairs touching fold-(k,j) sites,
  plus `pa_cal_f{k}` for outer scoring and `pa_cal_full` for deployment.
  ~20 refits of a cheap CPU stage.
- **cal_var floor (off-support honesty):**
  `cal_var_final = max(cal_var_model, floor(dist_to_nearest_pair))` with a
  conservative monotone floor, so rural sensors far from any FRM reference
  genuinely shrink in training weight. The weight-by-distance-band
  diagnostic table is a mandatory artifact.
- **Gate G0 (LOLO over pairing sites):** the learned calibration must beat
  BOTH Barkjohn AND the AMT-2024 multilinear RH+T form (the real SOTA
  baseline, not the strawman) on LOLO bias and RMSE; else fall back to the
  best of those forms refit on TX pairs (never published cf_1 constants on
  ATM). By-year bias flatness is a ship criterion.
- cf_1 refetch policy: month-chunked resumable fetcher with failed-month
  sidecars in its own multi-day stage (`data-pa`); fallback reconstruction
  (pa_cf1 := ATM below 20 µg/m³, flag above) is allowed for training but
  **reconstructed rows are excluded from exceedance labels** and get
  inflated cal_var — the fallback cannot silently poison the tail product.

## 5. T0 — debiased CTM prior (stage `priors`)

Per stream c ∈ {geoscf_pm25, cams_pm25, merra2_pm25_proxy}: EPA-Downscaler
lineage, y_FRM(s,t) = β0_c(s) + β1_c(s)·CTM_c(s,t), β fields on a ~200-node
SPDE/thin-plate low-rank basis with season interaction, ridge-penalized WLS on
inner AQS sites only, per outer fold. T0(q,t) = precision-weighted combination
of **available** streams (a missing stream is absent, never filled).
Continuous, full-coverage, queryable anywhere; the ladder's extrapolation
floor.

## 6. T1 — statistical skeleton (stage `skeleton`)

- **Target:** FRM scale. Rows = AQS site-days (weight 1/σ²_FRM, σ_FRM ≈ 1.5)
  ∪ calibrated-PA sensor-days with precision weights
  w = σ²_FRM / (σ²_FRM + cal_var_final), scaled by λ tuned on inner
  selection folds (λ frozen from T0/T1 metrics before any deep tier trains).
- **Features (single-builder parity contract):** every feature computable
  identically at arbitrary (lat, lon, date) through one
  `build_point_features` path — training rows and serving queries share it.
  - Portable set (extrapolation-safe): T0 prior + per-stream debiased CTMs;
    ERA5 grid-sourced met (replacing PA-IDW met — a measured train/serve
    covariate shift in v1); MERRA-2 species + PBLH + u10/v10; HMS smoke from
    a polygon raster (queryable off-network); MAIAC AOD (NaN-honest);
    HR statics: 30 m DEM elevation, NLCD impervious/developed fractions at
    1/5 km buffers, TIGER road density, NEI point emissions at 5/20 km
    (year-keyed), population density, dist_to_coast; doy/dow harmonics.
  - Interpolating set (coverage-gated): calibrated-PA neighbor block
    nbr_pacal_{25,50,100} km (+ counts, std, lag-1 versions) and the
    FRM neighbor block **lagged only** — nbr_frm_50km_lag1 / _lag7
    (same-day FRM is not available at serve time; the v1-style train/serve
    contradiction is excluded by construction). Empty-pool encoding is an
    explicit availability indicator, never a silent fill.
  - **Excluded:** raw lat/lon (location enters only through the GP),
    dist_to_nearest_sensor as a GBM feature, all four demographic variables
    (assert retained; `plus_demographics` stays ablation-only).
- **Model:** GPBoost — LightGBM fixed effects + Matérn-3/2 GP (Vecchia,
  m = 30, EPSG:3083) + grouped random effects: unit intercept (absorbs the
  device nugget that zeroed v1's residual kriging) + day effect (regional
  shocks). Candidate B: v1's 4-model `train_cv` ensemble refit on this frame
  + per-day GP residual krige. T1 = inner-selection winner; the loser is a
  mandatory paired baseline forever.
- OOF via the nested LOSO folds (§2); neighbor blocks recomputed per fold
  through the `nbr_overrides` npz contract (keys `f{fold}__{col}`).

## 7. T2 — graph-attention residual (stages `graphpre`, `graphres`)

Residual target r1 = y − T1_oof. The between-site workhorse.

- **Deployment-honest input rule:** PA stations are the only observation
  inputs; AQS sites appear only as query/target nodes (serving has no
  same-day FRM feed).
- **Graph:** nodes = PA stations + queries. Airsheds = 12 k-means clusters
  on (projected coords, climatological wind, elevation). Edges: k = 10 NN
  within airshed; cross-airshed only if d < 150 km and |Δelev| < 500 m.
  Edge features: log-distance, bearing vs same-day wind (advection
  alignment), wind speed, Δelevation, same-airshed/CBSA flags.
- **Architecture:** 4-layer pre-LN transformer, d = 128, 8 heads, attention
  logits biased by an MLP of edge features (learned non-stationary distance
  decay); **shielded attention** (only observed nodes emit keys/values; a
  masked or query station has ALL days replaced by [MASK] and a learned
  null h_rel embedding — full-window masking, not day-t-only, so the task
  is spatial interpolation, not own-history extrapolation). Empty
  neighborhood ⇒ m2 = 0 ⇒ passthrough. ~1.5 M params. h_rel (amortized
  station-reliability latent from an embargoed trailing window) with a unit
  test asserting no future-day leakage and null embedding at scored nodes.
- **Pretraining (`graphpre`):** masked-station reconstruction on **raw**
  PA values (calibration applied only at fine-tune, per fold — pretraining
  consumes no FRM-derived labels and no T0), 20–40% masking per day, half
  uniform, half structured 50–150 km ball masking (teaches σ to widen
  off-support). Fold-purity: a one-fold leakage-magnitude study (shared vs
  fold-pure pretrain) decides whether 5× fold-pure pretraining is required;
  the decision is recorded, not assumed. A temporally-pure (< 2025) variant
  is the sole basis for temporal-holdout claims.
- **Fine-tune (`graphres`):** per inner fold, predict r1 at held-out units;
  4:1 oversampling of mask-the-AQS-site tasks (reconstruct the FRM-scale
  residual from PA + context — the deployment task); Gaussian NLL + pinball
  auxiliary heads; precision-weighted. The nested exclusion runs double as
  the **fold-assignment epistemic ensemble** (seed ensembles rejected:
  epistemic spread must come from which-units-held-out).

## 8. T3 — pretrained field residual + INR decoder (stages `fieldpre`, `fieldres`)

Residual target r2 = y − F2_oof.

- **Stack v2** (0.1° grid, extends the v1 npz schema additively): ctm,
  merra2 (incl. winds), ERA5 met, HMS polygon raster, MAIAC aerosol,
  raw-PA observation raster + obs-count, HR-static planes, per-CHANNEL
  finite-mask planes (v1's per-group flags retired). No pa_cal raster and
  no T2 surfaces in the shared pretrain stack (they are FRM-informed);
  fold-honest inputs only.
- **Pretraining (`fieldpre`):** masked multimodal autoencoder — 60% patch
  masking + whole-channel-group drops, loss only on finite pixels (no
  mean-fill anywhere), the **observation raster inside the masking
  objective**, and a **discrepancy head** predicting obs − CTM at masked
  stations so the representation encodes CTM error structure instead of
  becoming a CTM autoencoder. Temporally-pure variant for temporal claims.
- **Decoder (`fieldres`):** INR point head — query (lat, lon, t) →
  multi-scale bilinear samples of encoder features + HR statics at the
  exact point + band-limited Fourier features of the sub-cell offset
  (wavelengths 5–200 km, hard cap against between-sensor hallucination) →
  MLP → (r̂2, log σ²). Supervision at TRUE station coordinates — no
  rint() snapping; co-located urban sensors get distinct supervision.
  Per-inner-fold fine-tunes give OOF at 100% of rows. Coverage-density is
  part of T3's dispatch pattern (not only its strata), so off-support rows
  structurally pass through. v1's FusionUNet, frozen, is the mandatory
  paired benchmark.

## 9. Composition, admission, exceedance (stages `gates`, `exceed`)

- `compose.py` contract: `fit_gate(r_k, avail_mask, pattern_id, ...)`
  asserts NaN on unavailable rows; per-pattern α on selection folds;
  admission on confirmation folds: one-sided paired unit-cluster bootstrap
  (1000 draws) requiring **non-inferiority on every primary metric**
  (pooled held-out-site R², between-site R², exceedance F1, within margins
  from `power_analysis.json`) **plus CI-separated superiority on at least
  one**. Unseen pattern → α = 0. Strata below 5 held-out clusters → α = 0.
  `gates.json` schema: `{tier: {pattern: {stratum: alpha, test: {...}}}}` —
  structurally unable to express an incumbent coefficient or a fill value.
- Outer folds never gate shipping; they are descriptive. The vault is
  consumed once, after freeze.
- **Exceedance head:** cross-fit LightGBM classifier P(y > 35.4) (and
  > 9.0), tail oversampling, isotonic-calibrated on selection OOF,
  threshold frozen on confirmation folds, **inside the admission harness**
  with its own paired test against a thresholded-composite baseline.
  Decoupled from the regression path so tier acceptance cannot break it.

## 10. Uncertainty (stage `uq`)

σ̂²(q) = GP predictive variance (T1) ⊕ T2/T3 heteroscedastic heads ⊕
fold-assignment ensemble spread ⊕ propagated cal_var. Quantile heads are
refit on the **deployed composed predictor's** cross-fit OOF (artifact
lineage records the tier hash they were fit against — grafted bands are a
build error). Conformal: **unit-level scores** (one score per sensor/site;
honest n = units), NexCP weighting w ∝ exp(−d/ρ_s)·exp(−Δt/τ) evaluated at
the query, δ per coverage-density bin, calibration units disjoint from all
selection/confirmation rows. Ship window: site-level coverage ∈ [0.88, 0.93]
on outer folds. Honest intervals will be wider than v1's — pre-registered as
the correct outcome.

## 11. Validation (stage `validate`) and ship rules

Primary (pre-registered): (P1) held-out-AQS R²/RMSE/bias on outer folds +
the vault, site-cluster bootstrap CIs, by-year bias flatness, attenuation
slope per year; (P2) between-site R² **and Spearman rank-ρ** on the FRM
scale — with-network AND bare-site arms; (P3) exceedance precision/recall/F1
at FRM labels; (P4) interval coverage/width, site-level. Pooled LOSO R² is a
labeled ~93%-temporal diagnostic. Secondary: spatial/temporal decomposition
per tier, per-block tables (worst block named — v1's block-3 R² −1.8 stays
visible), Moran's I, strata, OOF-only permutation importance.

Baselines, all paired (unit-cluster bootstrap, identical rows): properly-fit
per-day ordinary kriging on calibrated PA, NNGP/GPBoost-GP, T0 alone, v1
Tier-1 re-scored on the identical FRM protocol, v1 FusionUNet, persistence,
site climatology. Every deep rung vs its classical counterpart and vs T1.

Serving parity is a ship criterion: the full serving path (bag of fold
models — never a full refit — through `build_point_features` and
`gates.json`) is run over all inner site-days and must agree with the
admission-tested OOF composite within bootstrap CI per site.

Ship decision tree: if no gate opens, ship T1 (or the classical winner) with
the negative results reported — pre-committed, not improvised.

## 12. Data plan (stage `data`, `data-pa`, `statics`, `colocate`)

All new fetchers follow the v1 month-chunk + `.failed.json` sidecar +
window-stamped-final contract. New/changed:

1. PA cf_1 refetch (`data-pa`, own multi-day resumable stage; API cost
   arithmetic in the audit; A/B channel QC: drop days with |A−B|/mean > 0.3).
2. AQS fetcher hardened: window-stamped dest + sidecar (closes the
   documented quick/full cache-poisoning hazard); POC + method retained;
   site metadata (location setting) kept.
3. ERA5 full point-met (t2m, rh/d2m, u10, v10, precip + existing extras) so
   met is grid-sourced everywhere.
4. MERRA-2 + M2T1NXSLV U10M/V10M/T2M.
5. HMS polygon → 0.1° cell raster (`hms_grid.parquet`).
6. Committed statics (`pipeline/static_covariates.parquet` + 0.01° raster
   npz + build scripts): NLCD 2021 fractions, TIGER road density, NEI
   2020/2023 point emissions (year-keyed join), WorldPop, SRTM 30 m DEM.
7. MAIAC MCD19A2 1-km AOD — stretch, promoted only on a positive fold-0
   between-site ablation.
8. `colocation_pairs.parquet` (site, sensor, distance, shared days).

Committed v1 parquets all remain live inputs. `artifacts/v2/` namespace
(subdir per correction/config) fixes the v1 flat-namespace overwrite hazard.

## 13. Phoenix plan

Conventions: v1 stage-CLI (sentinels, FORCE=1, `--quick`, venv bootstrap +
git_sha stamping in the data stage), embers QOS unless noted, **every** GPU
loop writes atomic (tmp + os.replace) checkpoints with optimizer/scheduler/
RNG state every 30 min AND every epoch, first checkpoint inside the 1-h
protected window; long trainings are afterany-chained ≤ 8 h jobs that exit 0
on sentinel. embers preemption is CANCEL-not-requeue: a killed chain head
requires resubmission of `submit.sh` (idempotent; completed stages no-op) —
stated plainly, since submissions are manual.

DAG: audit → data-pa → data → statics → colocate → calibrate → priors →
features → skeleton → graphpre → graphres → fieldpre → fieldres → gates →
exceed → uq → validate → export.

| Stage | Partition | Budget |
|---|---|---|
| audit, colocate, calibrate, priors, gates, exceed, uq | cpu-small | minutes–4 h each |
| data, data-pa, statics | cpu-small | 6–8 h chunks, resumable |
| features | cpu-small 64–96 G | 3–5 h (nested nbr recompute) |
| skeleton | cpu-small 24c/64 G | 5 × folds, checkpoint per fold; kriging fallback documented |
| graphpre + graphres | gpu-rtx6000 | ~15–25 GPU-h (+ fold-pure multiplier only if the leakage study demands it) |
| fieldpre + fieldres | gpu-rtx6000/a100 | ~30–45 GPU-h |
| validate | cpu-small 96 G | 3–6 h |
| export | gpu-rtx6000 | scoped: latest-N-days surfaces + on-demand queries (no 3-billion-query fantasy) |

Itemized ablation budget (mandatory, published before first submission):
no-pretrain T2/T3, grid-only pretrain, naive-kNN graph, INR-vs-pixel head,
ATM-vs-cf_1 calibration, no-PA (truth-only), no-lags, no-statics,
plus_demographics, temporally-pure variants — ~2× the primary GPU budget;
total ≈ 150–250 GPU-h, free-tier shaped; inferno is a named contingency for
`fieldpre` only.

## 14. Reuse map (v1 asset → disposition)

**Kept as-is:** data_external chunk/sidecar/window-stamp machinery; GEOS-CF
and MERRA-2 fetchers; committed pipeline parquets; `neighbor_features.py` (+
its brute-force regression-test pattern); folds freeze/replay discipline +
`nbr_overrides` npz contract; `models_tabular` registry/train_cv/simplex
blend/quantile machinery (T1 candidate B + baselines); validation.py metric
battery (bootstrap CIs, Moran's I, AQI skill, strata, spatial/temporal
decomposition, kriging baselines, `_paired_delta_r2_ci` → promoted to the
gate test); UNet blocks + AMP/warmup-cosine scaffolding + pre-fill norm
stats; `conformal_intervals` pure function; stage/sentinel/sbatch
conventions; demographic exclusion assert + ablation.

**Modified:** corrections.py gains `method="ko_cal"` (signature preserved —
the method string threads through folds/caches); `build_training_frame` →
`build_frame_truth` (two-network rows, weights, lags, statics, no raw
lat/lon); `build_site_features` → `build_point_features` (single-builder
parity, the R5 serving path); AQS fetcher; `make_loso_folds` → unit
generalization + outer/vault builders; quantile heads refit on the deployed
predictor; `export_surface` → INR query API.

**Deleted (cause):** `stack_meta`/`cross_fit_meta`/`predict_meta` and every
mean-fill site (the −0.05 mechanism); `residual_kriging_oof` as a component
(nugget-structural zero; the per-day kriging engine survives as baseline);
`conformal_recenter` grafting; Barkjohn-as-target (ablation arm only);
SpatialAttentionFusion softmax-over-sources + dead flag channels (FusionUNet
frozen as benchmark only); raw lat/lon features; in-sample permutation/SHAP.

## 15. Risks (honest)

1. Semi-colocation confounding may cap calibration quality — G0 gate +
   fallback keeps the ladder valid; the floor still clears v1 (bias² term).
2. cf_1 refetch cost — audited before commitment; reconstruction fallback
   with tail-label exclusion.
3. 62 (→ 50 non-vault) AQS sites → wide CIs and limited gate power — the
   power analysis sizes margins; passthrough is the designed safe outcome;
   rank-ρ metrics for small n.
4. Deep tiers may gate to zero — then v2 ships as an anchored statistical
   model that already fixes the dominant error, and the negative result is
   reported against strong baselines (a defensible finding).
5. True between-site variance in Texas may be modest — report the variance
   decomposition; do not promise a number.
6. embers churn — checkpoint discipline + manual resubmission stated;
   inferno contingency named.
7. Serving parity drift — a ship criterion with its own stage, not a hope.

## 16. File layout (research/aqnet2/)

```
DESIGN.md CHANGES.md            this contract + the v1→v2 delta log
config2.py                      paths, domains, feature contracts, vault registry
folds2.py                       unified fold system, vault, content hash
fetchers2.py                    cf_1 / AQS-hardened / ERA5 / SLV / HMS-grid / statics
colocate.py calibrate.py        pair table; KO calibration (nested) + G0 gate
priors.py                       T0 downscaler
frame2.py                       build_frame_truth + build_point_features (parity)
skeleton.py                     T1 GPBoost + candidate-B ensemble
graph_res.py                    T2 pretrain + fine-tune + graph cache
field_res.py                    T3 MAE pretrain + INR decoder
compose.py                      gates, admission tests, gates.json, regression test
exceed.py uq.py                 exceedance head; NexCP conformal + quantile refit
validate2.py                    battery extensions: vault, bare-site, rank-ρ, parity
pipeline2.py                    stage CLI (audit … export), --quick
tests/                          leakage + harness unit tests (incl. 17%-finite test)
slurm/ (repo slurm2/)           aq2-*.sbatch + submit.sh
```
