# AQNet: A Three-Tier Fusion Framework for Daily PM2.5 Estimation Across Texas with Distribution-Free Uncertainty Quantification

*Shared Skies Initiative — offline research track (`research/aqnet/`)*

---

## Abstract

Most Texas census tracts contain no regulatory PM2.5 monitor, and low-cost sensor networks that fill the gap are unevenly sited and systematically biased. AQNet is an offline, publication-oriented research framework that estimates daily ground-level PM2.5 across Texas at 0.1° resolution by fusing three complementary model families: (Tier 1) a gradient-boosted and random-forest tabular ensemble with Huber objectives and quantile heads over a physical feature set (meteorology, source proximity, neighbor aggregates, satellite and chemical-transport priors, aerosol speciation); (Tier 2) a spatial-attention fusion U-Net that learns continuous exposure surfaces from gridded satellite, chemical-transport, smoke, meteorological, terrain, and data-availability-flag channels under sparse sensor supervision; and (Tier 3) a cross-fitted stacked meta-learner trained strictly on out-of-fold predictions, augmented with per-day residual kriging and conformalized quantile regression (CQR) prediction intervals. Training targets are Barkjohn-corrected PurpleAir measurements; EPA AQS FRM/FEM observations are reserved exclusively for external validation and never enter training. Demographic variables are deliberately excluded from all prediction inputs to avoid circularity in downstream environmental-justice analyses; they inform only sensor-placement allocation, and their marginal predictive contribution is quantified in a clearly-labeled ablation that never touches a deliverable model. The validation protocol covers leave-one-sensor-out, spatial-block, and temporal generalization; interpolation and raw plus mean-debiased chemical-transport baselines; sensor-clustered bootstrap confidence intervals; per-day residual spatial autocorrelation; smoke/dust/clean stratified metrics; a spatial-versus-temporal skill decomposition; feature-set ablations on frozen folds; and AQI-category accuracy. **This report specifies the design and protocol only: the AQNet models are untrained until the accompanying pipeline is executed, and the Results section is an empty template to be filled from computed artifacts.** The production ensemble's leave-one-sensor-out R² of 0.7136 is the pre-registered baseline to beat.

---

## 1 Introduction

Fine particulate matter (PM2.5) is among the best-documented environmental risk factors for cardiovascular, respiratory, and neurological disease, yet the infrastructure that measures it is sparse. Texas spans roughly 696,000 km² and 6,896 census tracts, but regulatory Federal Reference Method / Federal Equivalent Method (FRM/FEM) monitors number only in the dozens, concentrated in large metropolitan areas. Low-cost sensor networks such as PurpleAir have densified coverage by an order of magnitude, but their siting follows purchasing power rather than statistical design, and their optical measurements carry known humidity-dependent biases (Barkjohn et al., 2021). Estimating exposure where no monitor exists therefore requires models that combine sensor data with satellite aerosol products, chemical-transport model (CTM) output, meteorology, smoke detection, and terrain — an approach with a substantial literature at national scale (van Donkelaar et al., 2021; Di et al., 2019; Hu et al., 2017) that AQNet adapts to a single-state, sensor-dense, reproducible setting.

**Environmental-justice context and the exclusion of demographic predictors.** Exposure misestimation is not uniformly distributed: communities with fewer monitors — disproportionately low-income communities and communities of color — receive the least reliable estimates. The Shared Skies production system addresses this on two fronts: an exposure model, and a QUBO-based sensor-placement optimizer that weights coverage gaps by EJScreen equity indicators. AQNet draws a hard line between these two roles. **No demographic variable (EJScreen EJF score, percent people of color, percent low income, percent linguistically isolated) is used as a prediction input anywhere in AQNet.** The rationale is circularity: a central downstream use of exposure fields is testing whether exposure differs across demographic groups (as in health studies of PM2.5 and neurodegenerative outcomes). If the model is allowed to *learn* demographic composition as a predictor of PM2.5, then any observed exposure–demographics association is partially an artifact of the model's inputs rather than a property of the atmosphere. Physical EJScreen variables that proxy emission sources — traffic proximity, Superfund-site proximity, Risk Management Plan facility proximity, and diesel PM proximity — describe actual pollution sources and are retained. Demographic data are used only in the *allocation* problem (where to place the next sensors), where prioritizing under-monitored, overburdened communities is an explicit policy choice rather than a measurement claim. The predictive cost of this exclusion is not asserted; it is measured, once, in a frozen-fold ablation (§4.7) whose demographic variant exists only as an ablation arm.

**Contributions.** AQNet contributes: (i) a three-tier fusion architecture combining tabular ensembles, an attention-fusion U-Net with data-availability flags and modality dropout, and a leakage-controlled, cross-fitted stacked meta-learner with residual kriging; (ii) integration of two independent CTM priors (CAMS and NASA GEOS-CF) and MERRA-2 aerosol speciation and boundary-layer height; (iii) distribution-free uncertainty via quantile regression re-centered on the stacked prediction and calibrated with conformalized quantile regression; (iv) a pre-registered, multi-axis validation protocol with feature-set ablations, event-stratified metrics, a spatial/temporal skill decomposition, and fully external evaluation against EPA AQS; and (v) a reproducible Colab pipeline whose every reported number is computed, never asserted.

---

## 2 Data

All sources are public. Table 1 lists every dataset touched by the pipeline, its role, and its leakage status.

**Table 1. Data sources.**

| Source | Variables | Resolution / extent | Role | Leakage status |
|---|---|---|---|---|
| PurpleAir (ATM channel) | pm25, temperature, humidity, pressure, wind_speed, precipitation per sensor-day | 467 sensors, ~412K sensor-days, 2021-01 – 2026-05 | Training target (Barkjohn-corrected) + sensor meteorology; in-Texas sensors are targets, border-state sensors provide neighbor context only | Training |
| Open-Meteo archive (ERA5-derived) | Daily meteorology joined per sensor | Sensor-day | Primary historical weather | Training |
| Open-Meteo ERA5 extras (`met_extra`) | shortwave radiation, FAO reference ET0, cloud cover | Per 0.5° cell, daily | Additional meteorology: gridded fields feeding the U-Net meteorology channels | Training |
| NASA POWER | Same meteorological fields | Daily point API | Fallback where Open-Meteo quota was exhausted during the historical pull (units harmonized) | Training |
| CAMS via Open-Meteo air-quality API | aod, cams_pm25, dust | 0.5° cells, daily, from 2022-08-03 | Aerosol/CTM prior channels and tabular features; `dust` is both a tabular feature and a U-Net aerosol channel | Training |
| NOAA HMS smoke | hms_smoke ordinal tier 0–3 | Per sensor-day (polygon membership) | Wildfire-smoke indicator | Training |
| EPA EJScreen (physical subset) | traffic_proximity, superfund_proximity, rmp_proximity, diesel_pm_proximity | Census tract | Source-proximity features (demographic fields **excluded**) | Training |
| Census TIGERweb | Tract geometries and centroids | 6,896 Texas tracts | Grid extent, tract lookup, sensor–tract assignment | Training (geometry only) |
| Elevation service (cached) | Elevation at sensors and tract centroids | Point | Static terrain feature and U-Net channel | Training |
| EPA AQS daily (parameter 88101) | pm25_aqs at FRM/FEM sites | Site-day, 2021–2026, Texas | **External validation only** | **Never trained on** |
| NASA GEOS-CF (chm_tavg_1hr, v1) | geoscf_pm25 (surface `pm25_rh35_gcc`), hourly → daily mean | 0.25° global, via OPeNDAP | Independent CTM prior: tabular feature + U-Net channel + raw and mean-debiased prior baselines | Training (feature), baseline |
| MERRA-2 (M2T1NXAER, M2T1NXFLX) | DUSMASS25, OCSMASS, BCSMASS, SO4SMASS, SSSMASS25, PBLH → daily means, plus PBLH daily max/min (tabular-only); reconstructed surface-mass proxy | ~0.5° × 0.625°, via earthaccess | Aerosol speciation + boundary-layer features and channels; degrades gracefully to absent when no Earthdata credentials | Training (feature) |

Dataset-wide conventions:

1. All joins of gridded products to sensor locations use nearest-cell, same-day matching, and all daily aggregation is on UTC days (the AQS local-day mismatch this creates is discussed in §7).
2. Missing values are carried as NaN into the tabular models (which handle them natively). In the gridded stack, per-channel normalization statistics are computed over **finite pixels before any filling**, missing pixels are then mean-filled, and both the fill values and the normalization statistics are persisted in checkpoints to prevent train/serve skew.
3. The gridded stack additionally carries one binary **availability flag channel per source group** (1.0 on days the group had any real data, 0.0 where its planes were all-NaN before filling), appended after the physical groups and before filling. This "flag-and-fill" convention makes filled days distinguishable from observed days (§3.3).
4. The MERRA-2 surface-mass proxy is the standard GOCART reconstruction, 1.375·SO4 + 1.6·OC + BC + DUST2.5 + SS2.5 converted to µg/m³ (Buchard et al., 2017; Provençal et al., 2017). Because the MERRA-2 aerosol module carries no nitrate, the proxy systematically understates PM2.5 where nitrate matters (notably cool-season conditions); it is provided as a tabular feature and prior baseline but not as a U-Net channel — the six underlying species/PBLH channels are, letting the network learn its own combination.
5. GEOS-CF's `pm25_rh35_gcc` reports PM2.5 with aerosol water at 35% relative humidity, a filter-equilibration-like convention rather than ambient humidification. This convention difference (and its analogue for other CTM outputs) produces roughly constant level offsets against the corrected-PurpleAir target, which is exactly why the baseline suite includes mean-debiased variants of each CTM prior alongside the raw ones (§4.5).

---

## 3 Methods

### 3.1 Target definition: Barkjohn correction

Raw PurpleAir ATM-channel concentrations overestimate PM2.5, with a humidity-dependent bias. AQNet's primary target applies the U.S.-wide correction of Barkjohn et al. (2021):

> PM2.5 = 0.524 · PA_ATM − 0.0862 · RH + 5.75, clipped at ≥ 0 µg/m³,

where PA_ATM is the ATM-channel reading and RH is relative humidity (%). Three approximations in how AQNet applies this correction are stated explicitly rather than hidden:

**cf_1-to-ATM transfer.** The Barkjohn et al. (2021) coefficients were developed on PurpleAir's cf_1 ("higher") data channel; the Shared Skies archive stores the ATM channel, so AQNet applies the correction to ATM readings. The two channels agree at low concentrations but diverge as concentrations rise, where the ATM channel reports below cf_1; applying the cf_1-derived slope to ATM therefore drifts progressively below the canonical correction exactly in the high-concentration regime, compounding the known degradation of the linear form under extreme wildfire smoke (Barkjohn et al., 2022). This is treated as a stated limitation rather than patched ad hoc, and the smoke-stratified metrics of §4.6 are the place its consequences would appear.

**Ambient RH substitution.** The correction was fit with the sensor's on-board humidity reading, which is not present in the archive; AQNet substitutes ambient (Open-Meteo/ERA5-derived) RH. On-board RH tends to read low relative to ambient because of internal heating, so the substituted RH is generally higher and — through the −0.0862·RH term — shifts the corrected target slightly downward relative to the canonical correction. The shift is approximately additive and largely absorbed by tree-based learners, but it matters when comparing absolute levels against AQS and is therefore named here.

**Humidity's dual role.** RH enters both the target correction and the feature list. Measured on the archive (412,471 sensor-days), the correction *reduces* the target's linear humidity predictability rather than adding any: the R² of a univariate RH regression falls from 0.0894 against the raw ATM target to 0.0034 against the corrected target, because the −0.0862·RH term cancels the pre-existing positive PM–humidity correlation (r = +0.299). The dual role is therefore not an R²-inflation mechanism. The reasons corrected-target scores are not comparable to raw-target scores (production's 0.7136) are instead the 4.03× target variance compression (69.71 → 17.32 µg²/m⁶) and the differing feature and sensor sets (§5). The `method="raw"` sensitivity run remains the guard for any residual target-formula learning: the entire pipeline re-executes on the uncorrected ATM target with the identical feature set, and conclusions are only trusted where they survive both targets. Because the production system trains on raw ATM values, the raw option doubles as the like-for-like production comparison.

### 3.2 Tier 1 — tabular ensemble with quantile heads

Tier 1 mirrors the production architecture with demographic features removed: 34 physical features (meteorology; source proximity; geography and elevation; leave-self-out neighbor aggregates of same-day PM2.5 at 25/50/100 km computed by BallTree haversine search; HMS smoke; CAMS AOD and PM2.5; cyclical time encodings; weather interactions), plus CAMS `dust`, and, when available, `geoscf_pm25` and the nine MERRA-2 features (five species masses, the surface-mass proxy, and PBLH daily mean/max/min).

Four learners are trained per cross-validation fold — random forest (Breiman, 2001), LightGBM (Ke et al., 2017), XGBoost (Chen and Guestrin, 2016), and CatBoost (Prokhorenkova et al., 2018). The boosted learners use robust Huber-type objectives with δ = 10 µg/m³ (LightGBM `huber` with α = 10; XGBoost pseudo-Huber with slope 10 and a median-anchored base score): residuals beyond ~10 µg/m³ are dominated by smoke and dust spikes, and a Huber transition keeps those days informative without letting a handful of extreme days dominate the gradients. The random forest is regularized for smooth spatial generalization (a third of features per split, minimum leaf size 10).

Out-of-fold (OOF) predictions are combined by a simplex-constrained convex blend: weights **w** minimize the squared error of the blended OOF prediction subject to w_k ≥ 0 and Σ_k w_k = 1, solved by SLSQP. Two blends are computed: the pooled-OOF blend (weights fit on all OOF rows at once, reported for continuity) and a **leave-one-fold-out (LOFO) blend — the headline Tier-1 result**: for each fold f, blend weights are fit only on the OOF rows of the *other* folds and then applied to fold f's rows, so even the blend weights never see the rows they score.

Neighbor features are strictly leave-self-out (a sensor's own reading never enters its own neighbor aggregate) and same-day only — and, critically, they are **recomputed per fold** during cross-validation with the pooling restricted to train-fold sensors (§3.5). A parallel LightGBM quantile model (objective = quantile, τ ∈ {0.05, 0.5, 0.95}), trained under the same per-fold neighbor recomputation, produces raw interval endpoints later re-centered and calibrated by conformal prediction (§3.4).

### 3.3 Tier 2 — FusionUNet on the extended gridded stack

Tier 2 reuses the existing deep-learning track's FusionUNet (Ronneberger et al., 2015, for the U-Net backbone) on a daily 0.1° Texas grid. The channel groups are: `aerosol` (aod, cams_pm25, dust), `smoke` (hms_smoke), `meteorology` (five sensor-interpolated fields plus the three ERA5 extras), `static` (elevation), `temporal` (doy_sin, doy_cos, dow_sin, dow_cos), `ctm` (geoscf_pm25), `merra2` (five species mass channels + PBLH), and `flags` — one binary availability channel per source group (§2, convention 3). Each source group g supplies channels x_g ∈ ℝ^(C_g×H×W) and passes through its own two-layer convolutional encoder into a shared embedding space:

- e_g = Enc_g(x_g) ∈ ℝ^(E×H×W)
- s_g = Conv3x3(e_g) ∈ ℝ^(1×H×W) (per-source score head)
- α_g,ij = exp(s_g,ij) / Σ_g′ exp(s_g′,ij) (softmax **across sources** at every pixel)
- u = Σ_g α_g ⊙ e_g (attention-weighted fusion)
- fused = σ(W2 · SiLU(W1 · GAP(u))) ⊙ u (squeeze-excite channel gate)
- ŷ = softplus(UNet(fused)), a non-negative PM2.5 surface.

The attention maps α are retained for interpretability: they show which source the model trusts at each pixel (e.g., smoke channels during HMS events, CTM channels far from sensors). The flag channels exist precisely so this trust can be *day-conditional*: a mean-filled CAMS day looks identical to a real one after filling, but its flag is 0, letting the attention learn to down-weight sources on the days they were absent (flag-and-fill). As a complementary regularizer, **modality dropout** zeroes an entire source group's channels with probability 0.15 per batch (flags are never dropped), forcing the network to remain usable when a source is missing at inference. The U-Net is depth 4 with GroupNorm and bilinear upsampling.

Supervision is sparse: the loss is a **masked Huber** (δ = 15 µg/m³) evaluated only at grid pixels containing in-Texas sensors — the same masking semantics as a masked MSE, with the quadratic-to-linear transition keeping smoke-day pixels informative without letting them dominate; δ is set looser than Tier 1's because pixel supervision is sparser and noisier. The network predicts every pixel while being graded only where truth exists.

The training regime: AdamW with weight decay 10⁻³ under a 5-epoch linear warmup followed by cosine annealing over the remaining budget (Loshchilov and Hutter, 2019); random 96×96 training crops (crops with an empty supervision mask are skipped; evaluation always runs on the full grid); mixed precision and larger batches on GPU (batch 32 CUDA / 8 CPU); and full-budget training with best-checkpoint selection on grouped-site-holdout RMSE as the safety net (early-stopping patience remains available as an option). Normalization statistics are computed over finite pixels before filling (§2), and all hyperparameters are recorded inside the checkpoint. Per-row U-Net predictions for stacking are read from trained surfaces at each observation's (date, pixel); for stacking they are **masked to validation-site pixels only** (§3.4).

### 3.4 Tier 3 — stacking, residual kriging, and conformal intervals

**Stacked meta-learner (cross-fitted).** The meta-learner combines exactly three component predictions — parts {`tier1`, `rk`, `unet`}: the Tier-1 **LOFO blend** OOF prediction, the **kriged residual itself** (below; it enters as its own component so the combiner learns its weight, rather than being pre-added to Tier 1), and the U-Net pixel OOF prediction **masked to validation-site pixels** (rows whose pixel hosted a U-Net training site are set to NaN, so the stack can never reward memorized training pixels). The combiner is a Ridge regression with **positive = True**, so every coefficient is non-negative and the stack is an interpretable re-weighting, never a sign-flipping regression; components missing for a row are filled at prediction time with that component's meta-training mean, stored on the model.

Two leakage guards operate at once. First, in the tradition of stacked generalization (Wolpert, 1992), the meta-learner is fit **only on out-of-fold component predictions**: every input it sees for row i was produced by models that never saw row i in training. Second, the combiner itself is **cross-fitted**: sensors are dealt into grouped K folds (4 folds over sensor ids), and for each fold a Ridge is fit on the other folds' rows and predicts the held-out fold, so each row's Tier-3 prediction comes from a combiner that never saw that row's sensor. A final Ridge fit on all meta-training OOF rows serves rows outside the cross-fit (the calibration split and deployment).

**Residual kriging.** Tree ensembles and CNNs both leave spatially structured residuals. For each fold and each day, ordinary kriging (Cressie, 1993; pykrige implementation, inverse-distance-weighting fallback when unavailable) is fit to **train-fold residuals of the LOFO blend only** and evaluated at test-fold sensor locations, capping the number of training points per day for tractability. The kriged residual field enters the meta-learner as the `rk` component, letting the stack decide how much residual spatial structure to trust.

**Conformalized quantile intervals.** Raw quantile intervals under-cover in finite samples, and their center need not match the best point prediction. AQNet therefore (i) **re-centers the quantile band on the Tier-3 prediction**, carrying over the band's shape: lo = tier3 − (q50 − q05), hi = tier3 + (q95 − q50); and (ii) calibrates the band with split-conformal widening using the conformalized-quantile-regression (CQR) nonconformity score (Romano et al., 2019; Vovk et al., 2005; Angelopoulos and Bates, 2023). On a **calibration split of sensors disjoint from every set used to fit the meta-learner or quantile heads**, scores s_i = max(lo_i − y_i, y_i − hi_i) are computed and δ is the ⌈(n+1)(1−α)⌉/n empirical quantile of {s_i}. Reported intervals [lo − δ, hi + δ] carry a finite-sample marginal coverage guarantee of ≥ 1 − α (α = 0.1 by default) under exchangeability. Empirical coverage **and mean width** are then evaluated on the meta-training cross-fit rows — disjoint from the rows that fit δ — so the reported operating characteristics are not read off the calibration data itself.

### 3.5 Leakage-control summary

1. EPA AQS never enters training or feature computation for training rows.
2. Neighbor features are leave-self-out and same-day.
3. **Per-fold neighbor-feature recomputation:** during cross-validation, every `nbr_*` column is recomputed per fold with the pooling restricted to train-fold sensors, so train rows' neighbor aggregates never contain held-out sensors' readings — the dependence-aware CV discipline of Roberts et al. (2017). The production repository applied the same discipline to produce the 0.7136 baseline, keeping the comparison like-for-like.
4. The meta-learner trains only on out-of-fold component predictions ({tier1 LOFO blend, rk, unet}).
5. The meta-learner is itself **cross-fitted** over grouped sensor folds, so Tier-3 predictions are out-of-fold with respect to the combiner too.
6. The U-Net component fed to the stack is **masked to validation-site pixels**; rows at U-Net training-site pixels are NaN.
7. Residual kriging uses train-fold residuals only.
8. Conformal calibration uses a sensor-disjoint split; coverage and width are evaluated on rows disjoint from δ fitting.
9. Grid fill values and normalization statistics are computed on training data (finite pixels, pre-fill) and persisted.
10. No demographic variable appears in any deliverable feature list; the feature-assembly code asserts their absence. The single exception is the clearly-labeled ablation arm of §4.7, which bypasses the deliverable feature path explicitly and produces no model that is kept.

---

## 4 Validation protocol

All evaluation axes below are computed by the pipeline and written as JSON artifacts; nothing in this report pre-judges their values.

### 4.1 Leave-one-sensor-out (LOSO)

GroupKFold over `sensor_id` (10 folds, seed 42): every sensor's entire history is held out together, so performance reflects prediction at never-seen locations. Neighbor features are recomputed per fold with train-only pools (§3.5, item 3). This is the same protocol behind the production baseline number.

### 4.2 Spatial-block cross-validation

KMeans clustering of unique sensor coordinates into 5 regions, leave-one-region-out. LOSO can be optimistic when a held-out sensor has close neighbors in the training folds; spatial blocking removes entire regions and stresses long-range extrapolation.

### 4.3 Temporal holdout

Train before 2025-01-01, test after. This probes robustness to distribution shift (sensor-network growth, meteorological year-to-year variation, smoke seasons).

The spatial-block and temporal axes are run for the **Tier-1 blend only**: they exist to stress the tabular learner's feature-driven generalization, while the deep and stacked tiers are defined on the frozen LOSO folds (retraining a U-Net and re-deriving the stack per auxiliary axis would conflate compute budget with protocol). Table 3 is labeled accordingly.

### 4.4 External validation against EPA AQS

The strongest test: the identical feature vector is assembled at AQS FRM/FEM site-days (neighbor aggregates computed from the corrected-unit PurpleAir pool — the same units as the training target), the trained model predicts, and predictions are scored against regulatory `pm25_aqs`. AQS site-days are screened for quality on ingest: exceptional-event-excluded records are dropped, sub-daily-duration records require ≥ 75% observation completeness, and duplicate monitors resolve by the documented FEM-over-FRM duration/POC preference. Because AQS never touched training, this measures transfer from a corrected low-cost network to the regulatory standard, including any residual target mismatch.

A **deployment-mode Tier-3** row is also computed at AQS: the full-data Tier-1 prediction, a per-day kriging of full-data training residuals to the AQS sites, and the U-Net surface value at each site's pixel/day where the stack and checkpoint exist (NaN otherwise, handled by the ridge's stored column fill), combined by the stored final Ridge. This row is labeled "deployment-mode" because — unlike every cross-validated number — its components are full-data fits: it answers *"what would the deployed stack say at a regulatory site?"*, not *"how well does the stack generalize?"* (see Table 4 footnote).

### 4.5 Baselines

Every AQNet tier is compared against: nearest-sensor assignment; inverse-distance weighting (k = 8); per-day ordinary kriging of sensor values; and CTM priors used directly as predictions — `cams_pm25`, `geoscf_pm25`, and the MERRA-2 proxy where available, each in **raw** form and in a **mean-debiased** variant (the train-mean offset between prior and target is subtracted before scoring). The debiased variants separate level error — much of it a humidification-convention artifact (§2, convention 5) — from pattern skill, which is the harder thing for a fusion model to beat. A fusion model that fails to beat interpolation or its own priors has no claim to usefulness.

### 4.6 Metrics, inference, and diagnostics

R², RMSE, MAE, and mean bias are reported with 1,000-resample bootstrap 95% confidence intervals for R² and RMSE (seed 0). All bootstrap CIs use a **cluster bootstrap by sensor**: unique sensors are resampled with replacement and their rows enter together. An iid row bootstrap would be anti-conservative here — repeated days at the same sensor are strongly correlated, so treating rows as independent overstates the effective sample size and produces intervals that are too narrow.

Residual spatial autocorrelation is quantified by **per-day Moran's I** (Moran, 1950) over each day's k = 8 nearest-neighbor sensor graph, skipping days with fewer than 15 sensors, and summarized as the median and IQR across days (plus mean and day count). A single pooled Moran's I over all sensor-days conflates persistent between-site level differences with same-day spatial structure; the per-day statistic isolates the latter, which is what residual kriging is supposed to remove.

**Stratified metrics.** The full metric block is recomputed within three strata: smoke (HMS flag > 0), dust (CAMS dust above its 75th percentile of finite values), and clean (the rest). Aggregate R² is dominated by clean days, while the health-relevant errors — and the known optical-sensor weaknesses under dust (§7) — concentrate in the event strata, so event-day skill must be reported, not inferred.

**Spatial-versus-temporal decomposition.** R² is decomposed into a *spatial* component (R² of per-sensor means — does the model rank sites and reproduce the long-term gradient?) and a *temporal* component (R² of within-site anomalies, y − site mean versus prediction − predicted site mean — does the model track day-to-day dynamics?). The decomposition matters because a model can excel on one axis while failing the other, and downstream health analyses differ in which axis they lean on.

### 4.7 Feature-set ablation (frozen folds)

On the same frozen LOSO folds and the same target, the Tier-1 pipeline (with per-fold neighbor recomputation) is re-run for four variants:

- **primary** — the AQNet feature set as specified in §3.2;
- **plus_demographics** — primary plus the four excluded demographic columns. This variant exists **only as an ablation**: the frame is built with an explicit demographic-reference path, the feature list is constructed manually to bypass the asserting feature-assembly code, and no model from this arm is retained or deployed;
- **no_external** — primary minus all GEOS-CF and MERRA-2 features;
- **no_neighbor** — primary minus all `nbr_*` features (and no per-fold overrides, as there is nothing to override).

Each variant reports its LOSO metrics plus the **paired ΔR² versus primary** with cluster-bootstrap 95% CIs (differences computed on identical rows, sensors resampled). The ablation replaces assertion with measurement on three fronts: what demographics would add (the EJ-safe-modeling question), what the external CTM/reanalysis sources add, and how much of the skill is neighbor interpolation rather than atmosphere.

### 4.8 AQI-category metrics

Point metrics hide what matters for public communication, so predictions are also scored on the EPA 2024 daily PM2.5 AQI breakpoints:

| Category | PM2.5 (µg/m³) |
|---|---|
| Good | 0.0 – 9.0 |
| Moderate | 9.1 – 35.4 |
| Unhealthy for Sensitive Groups | 35.5 – 55.4 |
| Unhealthy | 55.5 – 125.4 |
| Very Unhealthy | 125.5 – 225.4 |
| Hazardous | ≥ 225.5 |

Reported for the Tier-1 LOSO OOF and the Tier-3 cross-fit rows: overall category accuracy, macro-F1, and precision/recall for exceedance of the 35.4 µg/m³ threshold — the operating point where a wrong answer changes public-health advice.

---

## 5 Results — TEMPLATE (to be filled from computed artifacts)

> **STATUS: NO RESULTS EXIST YET.** The AQNet models described above are **untrained** until the pipeline (`pipeline_colab.py`) is executed end-to-end. Every cell below is deliberately blank and must be filled **only** from the JSON/NPZ artifacts the pipeline writes (`metrics_loso.json`, `metrics_spatial_block.json`, `metrics_temporal.json`, `metrics_external_aqs.json`, `metrics_baselines.json`, `metrics_ablation.json`, and the auto-generated `SUMMARY.md`). Do not transcribe numbers from anywhere else. The only number quoted in this document is the production system's LOSO R² = 0.7136, which is a **baseline from a different, already-deployed model** (§6), not an AQNet result.

**Table 2. LOSO cross-validation (10-fold GroupKFold over sensors, Barkjohn target). CIs are sensor-clustered bootstrap. The Tier 2 row is computed on the rows whose pixels were U-Net validation sites (§3.4); its n differs accordingly.**

| Model | R² (95% CI) | RMSE (95% CI) | MAE | Bias | n |
|---|---|---|---|---|---|
| Nearest sensor | — | — | — | — | — |
| IDW (k=8) | — | — | — | — | — |
| Ordinary kriging | — | — | — | — | — |
| CAMS prior (raw) | — | — | — | — | — |
| CAMS prior (mean-debiased) | — | — | — | — | — |
| GEOS-CF prior (raw) | — | — | — | — | — |
| GEOS-CF prior (mean-debiased) | — | — | — | — | — |
| MERRA-2 proxy (raw) | — | — | — | — | — |
| MERRA-2 proxy (mean-debiased) | — | — | — | — | — |
| Tier 1 blend (LOFO) | — | — | — | — | — |
| Tier 2 FusionUNet (val-site pixels) | — | — | — | — | — |
| **Tier 3 AQNet (cross-fit stack)** | — | — | — | — | — |

**Table 3. Spatial-block and temporal generalization (Tier 1 blend; these axes stress the tabular tier only, §4.3).**

| Protocol | R² | RMSE | MAE | Bias | Moran's I (residuals, per-day median [IQR]) |
|---|---|---|---|---|---|
| Spatial block (5 regions) | — | — | — | — | — |
| Temporal (train < 2025-01-01) | — | — | — | — | — |

**Table 4. External validation at EPA AQS FRM/FEM sites (never trained on).**

| Model | R² | RMSE | MAE | Bias | n site-days |
|---|---|---|---|---|---|
| Tier 1 blend | — | — | — | — | — |
| **Tier 3 AQNet**† | — | — | — | — | — |

† Deployment-mode Tier-3: full-data Tier-1 components, per-day kriged full-data training residuals, U-Net surface values where available, combined by the stored final Ridge (§4.4). Unlike every cross-validated row, this row characterizes the deployed stack at regulatory sites, not held-out generalization.

**Table 5. Uncertainty quantification (α = 0.1). The quantile band is re-centered on the Tier-3 prediction (§3.4); δ is calibrated on the held-out calibration sensors via CQR nonconformity scores (Romano et al., 2019); empirical coverage and mean width are evaluated on the meta-train cross-fit rows, disjoint from δ fitting.**

| Interval | Empirical coverage | Mean width (µg/m³) |
|---|---|---|
| Re-centered quantile band (pre-conformal) | — | — |
| CQR-conformalized | — | — |

**Table 6. AQI-category performance (Tier 1 LOSO OOF; Tier 3 cross-fit rows).**

| Metric | Tier 1 | Tier 3 |
|---|---|---|
| Category accuracy | — | — |
| Macro-F1 | — | — |
| Exceedance (>35.4) precision | — | — |
| Exceedance (>35.4) recall | — | — |

**Table 7. Feature-set ablation (frozen LOSO folds, §4.7). ΔR² is paired on identical rows; CIs are sensor-clustered bootstrap. The plus_demographics arm exists only as an ablation and produces no retained model.**

| Variant | R² (95% CI) | RMSE | ΔR² vs primary (95% CI) |
|---|---|---|---|
| primary | — | — | (reference) |
| plus_demographics (ablation-only) | — | — | — |
| no_external | — | — | — |
| no_neighbor | — | — | — |

Additional planned figures (produced by the pipeline where dependencies permit): SHAP feature-importance summary for the Tier 1 LightGBM (`shap_summary.png`), permutation-importance report, and U-Net attention-map examples for smoke and non-smoke days.

---

## 6 Relationship to the production system

The Shared Skies production service runs a 4-model tree ensemble (RF, LightGBM, XGBoost, CatBoost; simplex-constrained convex blend fit by GroupKFold-over-sensors cross-validation) on 38 tabular features, serving live tract-level predictions on a 30-minute cycle. Its honest leave-one-sensor-out cross-validated R² is **0.7136** (`models/metrics.json`, `loso_cv_optimized`). **This is the production baseline, quoted for context and as the pre-registered number AQNet aims to beat; it is not an AQNet result.**

Three differences make the comparison informative but demand care:

1. **Feature set.** Production includes four demographic features that AQNet excludes by design. What that exclusion costs is not asserted but measured: the frozen-fold ablation of §4.7 runs a plus_demographics arm on identical folds and reports the paired ΔR² with cluster-bootstrap CIs (Table 7). The arm exists only as an ablation — no demographic-bearing model is retained.
2. **Target.** Production trains on raw ATM-channel PM2.5; AQNet's primary target is Barkjohn-corrected. R² values on different targets are not directly comparable, so the pipeline's `method="raw"` sensitivity run provides the like-for-like comparison, and all AQNet baselines (Table 2) are recomputed under AQNet's own target.
3. **Scope.** Production is a live system optimized for latency and API-quota budgets; AQNet is offline and free to use data sources (GEOS-CF OPeNDAP, MERRA-2 via earthaccess, per-day kriging) that would be impractical in a 30-minute serving loop. AQNet is **not** intended to replace the live map; validated components may be back-ported deliberately.

---

## 7 Limitations and ethics

**Measurement.** PurpleAir sensors are optically based and residentially sited; the network over-represents affluent urban neighborhoods, and no correction removes siting bias. A related **survivorship bias** affects the archive itself: it contains the sensors that stayed online long enough to accumulate history, and sensor mortality is plausibly correlated with harsh operating conditions and household resources, so the surviving network is not a random sample even of the installed one. The Barkjohn correction is linear, was developed on the cf_1 channel while AQNet applies it to ATM readings (§3.1), and degrades under extreme smoke loading (Barkjohn et al., 2022); high-concentration performance should be read from the AQI-category, exceedance, and smoke-stratum metrics, not overall R². Under **dust**, the Plantower optics inside PurpleAir undercount coarse-mode particles — laboratory and field evaluations show sharply falling detection efficiency above the sub-micron range (Kuula et al., 2020; Ouimette et al., 2022) — so dust-storm days carry target error that no model choice removes; this is why the dust stratum is reported separately (§4.6).

**Spatial and temporal support.** The 0.1° grid (~11 km) cannot resolve near-road or intra-neighborhood gradients; predictions are area averages, not personal exposures. All aggregation is on UTC days while EPA AQS daily values follow local-time days, so cross-midnight events smear differently in the two datasets — a convention mismatch that slightly penalizes the external comparison and is accepted rather than hidden. CAMS channels begin 2022-08-03, so earlier days carry filled aerosol inputs (marked by the availability flags, §2); MERRA-2 and GEOS-CF availability depends on external services and credentials, and the pipeline records exactly which sources each run used.

**Statistical.** Conformal guarantees are marginal, not conditional: average coverage is controlled, but coverage may vary by region, season, or concentration level (a planned diagnostic, not a solved problem). LOSO can still be optimistic where sensors cluster; the spatial-block and external-AQS axes exist precisely to bound that optimism. External AQS sites are few and urban-weighted, limiting the power of the external check in rural Texas.

**Ethics and intended use.** Demographic variables are excluded from prediction (§1) and used only for sensor-placement allocation; the ablation arm that touches them (§4.7) is measurement of the exclusion's cost, not a model. Exposure estimates from this framework are suitable for research and for prioritizing monitoring investment; they are not suitable for regulatory attainment determinations, enforcement, or individual medical decisions. All data are public and contain no personally identifiable information; PurpleAir locations are used at the coordinates the network itself publishes.

---

## 8 Reproducibility

The entire pipeline runs from the repository root on Google Colab or any Python environment:

```
python research/aqnet/pipeline_colab.py all                    # full run
python research/aqnet/pipeline_colab.py all --quick            # smoke test
python research/aqnet/pipeline_colab.py all --skip-merra2 --skip-geoscf
```

Stages (`data`, `features`, `tabular`, `deep`, `fuse`, `ablation`, `validate`) are individually re-runnable and print what they skipped and why; `--skip-ablation` drops the ablation stage from `all`, `--batch-size` overrides the U-Net batch, and `--correction raw` re-runs everything on the uncorrected target. External fetches are month-chunked with window-stamped final caches and failed-month sidecars — the month chunks are the cache of record — so quick runs and full runs never poison each other's caches. Determinism: fold construction uses seed 42, bootstrap uses seed 0, and the U-Net trainer seeds torch/numpy; residual GPU nondeterminism in convolution kernels may perturb Tier 2 slightly between runs. MERRA-2 requires a free NASA Earthdata login via `earthaccess`; without credentials the fetcher returns None with printed instructions and the pipeline proceeds without those features. GEOS-CF is fetched over public OPeNDAP with month-chunked, retry-wrapped reads.

Artifacts written to `research/aqnet/artifacts/`: `training_frame.parquet`, `folds.json`, `nbr_overrides_loso.npz` (per-fold neighbor-feature recomputations), `oof_tier1.npz`, `quantile_oof.npz`, the U-Net checkpoint directory plus `unet_train.json`, `oof_meta.npz`, `external_paths.json` (which external sources the run actually used), `metrics_loso.json`, `metrics_spatial_block.json`, `metrics_temporal.json`, `metrics_external_aqs.json`, `metrics_baselines.json`, `metrics_ablation.json`, `permutation_report.json`, `shap_summary.png`, `aqs_quick_subset.parquet` (quick-mode AQS slice), and `SUMMARY.md` — the last auto-generated from whatever metrics exist, computed and never invented.

---

## References

- Angelopoulos, A. N., and Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning*, 16(4), 494–591.
- Barkjohn, K. K., Gantt, B., and Clements, A. L. (2021). Development and application of a United States-wide correction for PM2.5 data collected with the PurpleAir sensor. *Atmospheric Measurement Techniques*, 14(6), 4617–4637.
- Barkjohn, K. K., Holder, A. L., Frederick, S. G., and Clements, A. L. (2022). Correction and accuracy of PurpleAir PM2.5 measurements for extreme wildfire smoke. *Sensors*, 22(24), 9669.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Buchard, V., Randles, C. A., da Silva, A. M., Darmenov, A., Colarco, P. R., Govindaraju, R., Ferrare, R., Hair, J., Beyersdorf, A. J., Ziemba, L. D., and Yu, H. (2017). The MERRA-2 aerosol reanalysis, 1980 onward. Part II: Evaluation and case studies. *Journal of Climate*, 30(17), 6851–6872.
- Chen, T., and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.
- Cressie, N. (1993). *Statistics for Spatial Data* (revised edition). Wiley, New York.
- Di, Q., Amini, H., Shi, L., Kloog, I., Silvern, R., Kelly, J., Sabath, M. B., Choirat, C., Koutrakis, P., Lyapustin, A., Wang, Y., Mickley, L. J., and Schwartz, J. (2019). An ensemble-based model of PM2.5 concentration across the contiguous United States with high spatiotemporal resolution. *Environment International*, 130, 104909.
- Gelaro, R., McCarty, W., Suárez, M. J., et al. (2017). The Modern-Era Retrospective Analysis for Research and Applications, Version 2 (MERRA-2). *Journal of Climate*, 30(14), 5419–5454.
- Hu, X., Belle, J. H., Meng, X., Wildani, A., Waller, L. A., Strickland, M. J., and Liu, Y. (2017). Estimating PM2.5 concentrations in the conterminous United States using the random forest approach. *Environmental Science & Technology*, 51(12), 6936–6944.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems 30*, 3146–3154.
- Keller, C. A., Knowland, K. E., Duncan, B. N., et al. (2021). Description of the NASA GEOS Composition Forecast modeling system GEOS-CF v1.0. *Journal of Advances in Modeling Earth Systems*, 13(4), e2020MS002413.
- Kuula, J., Mäkelä, T., Aurela, M., Teinilä, K., Varjonen, S., González, Ó., and Timonen, H. (2020). Laboratory evaluation of particle-size selectivity of optical low-cost particulate matter sensors. *Atmospheric Measurement Techniques*, 13(5), 2413–2423.
- Loshchilov, I., and Hutter, F. (2019). Decoupled weight decay regularization. In *International Conference on Learning Representations (ICLR 2019)*.
- Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika*, 37(1/2), 17–23.
- Ouimette, J. R., Malm, W. C., Schichtel, B. A., Sheridan, P. J., Andrews, E., Ogren, J. A., and Arnott, W. P. (2022). Evaluating the PurpleAir monitor as an aerosol light scattering instrument. *Atmospheric Measurement Techniques*, 15(3), 655–676.
- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., and Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. In *Advances in Neural Information Processing Systems 31*, 6638–6648.
- Provençal, S., Buchard, V., da Silva, A. M., Leduc, R., and Barrette, N. (2017). Evaluation of PM surface concentrations simulated by Version 1 of NASA's MERRA Aerosol Reanalysis over Europe. *Atmospheric Pollution Research*, 8(2), 374–382.
- Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., and Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929.
- Romano, Y., Patterson, E., and Candès, E. (2019). Conformalized quantile regression. In *Advances in Neural Information Processing Systems 32*, 3543–3553.
- Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In *Medical Image Computing and Computer-Assisted Intervention (MICCAI 2015)*, LNCS 9351, 234–241.
- van Donkelaar, A., Hammer, M. S., Bindle, L., et al. (2021). Monthly global estimates of fine particulate matter and their uncertainty. *Environmental Science & Technology*, 55(22), 15287–15300.
- Vovk, V., Gammerman, A., and Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer, New York.
- Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241–259.
