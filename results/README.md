# AQNet — Full-Scale Dual-Target Run Results

Two complete end-to-end executions of the AQNet three-tier pipeline (all six
stages, no `--quick`, no skipped stages), one per training target:

| Run | Target | Artifacts |
|---|---|---|
| 1 | **barkjohn** — Barkjohn et al. (2021)-corrected PurpleAir PM2.5 | [`results/barkjohn/`](barkjohn/) |
| 2 | **raw** — uncorrected PurpleAir ATM PM2.5 (production-comparable) | [`results/raw/`](raw/) |

Executed 2026-07-23 on a Windows 11 desktop: **AMD Ryzen 7 3700X (8c/16t),
NVIDIA RTX 2070 SUPER 8 GB, 16 GB RAM**. Python 3.12.10 venv,
**torch 2.11.0+cu128 (CUDA 12.8)** — full environment in
[`versions.txt`](versions.txt). Every number below is copied from the
pipeline's own `metrics_*.json` artifacts in this directory; nothing is
hand-entered. `n` differs across rows because each protocol scores a
different row set (stated per table).

---

## A. Run summary

Per-stage wall-clock (seconds), from the stage logs:

| Stage | Run 1 (barkjohn) | Run 2 (raw) |
|---|---:|---:|
| features | 191 | 196 |
| tabular | 1,969 | 1,937 |
| deep | 2,129 | 2,149 |
| fuse | 178 | 203 |
| ablation | 5,000 | 5,028 |
| validate | 1,907 | 1,943 |
| **total** | **11,374 s ≈ 3 h 10 m** | **11,456 s ≈ 3 h 11 m** |

External data acquisition (shared by both runs, one-time): EPA AQS cached
in-repo; GEOS-CF ≈ 2.2 h; MERRA-2 ≈ 0.6 h + validation probes (see §I for
why both fetchers needed intervention).

GPU usage verified in the logs of every relevant stage: tabular/ablation/
validate print `GPU detected via torch.cuda (NVIDIA GeForce RTX 2070
SUPER) — xgb device='cuda', catboost task_type='GPU'`; both deep stages
print `Device: cuda  batch_size 32  amp on` and ran the full 100-epoch
budget with no OOM (~18 s/epoch).

Environment notes (see §I for full list): `num_workers=0` patch applied
for the run and **reverted before commit** (Windows spawn-based DataLoader
workers would duplicate the ~3.5 GB grid stack per worker); pandas pinned
to 2.3.x; `h5py` and `dask` installed to complete the MERRA-2 streaming
stack that `requirements.txt` omits.

## B. Headline accuracy — both targets side by side

LOSO protocol, sensor-clustered bootstrap 95% CIs. **Corrected-target and
raw-target scores are not comparable to each other** (the correction
compresses target variance ~4×; RMSE/MAE are on different scales).

| Model (row basis) | barkjohn R² [95% CI] | barkjohn RMSE [95% CI] / MAE | raw R² [95% CI] | raw RMSE [95% CI] / MAE |
|---|---|---|---|---|
| **Tier 1 — LOFO blend** (all LOSO OOF rows; n=293,543 / 293,579) | **0.5995** [0.536, 0.665] | 2.742 [2.406, 3.129] / 1.493 | **0.6141** [0.545, 0.680] | 5.273 [4.642, 6.021] / 2.872 |
| **Tier 2 — FusionUNet** (U-Net held-out-site sensor-day rows; n=50,688 / 50,695) | 0.5082 [0.354, 0.641] | 3.026 [2.378, 3.882] / 1.820 | 0.4809 [0.268, 0.634] | 6.068 [4.763, 7.813] / 3.692 |
| **Tier 3 — cross-fit stack** (meta-train sensors; n=216,416 / 216,441) | 0.5583 [0.496, 0.614] | 2.827 [2.535, 3.172] / 1.774 | 0.5922 [0.522, 0.653] | 5.332 [4.747, 5.998] / 3.237 |

(MAE bootstrap CIs are not computed by the pipeline; RMSE CIs are
sensor-clustered like the R² CIs.)

Supporting rows:

| Item | barkjohn | raw |
|---|---|---|
| Tier-1 per-model R² (lgbm / xgb / catboost / rf) | 0.583 / 0.586 / 0.584 / **0.600** | 0.599 / 0.600 / 0.603 / **0.614** |
| Pooled blend weights (lgbm/xgb/catboost/rf) | 0.00 / 0.23 / 0.00 / 0.77 | 0.00 / 0.18 / 0.16 / 0.67 |
| Tier-2 pixel-level site-holdout best (from `barkjohn/unet_train.json`, `raw/unet_train.json`) | R² 0.5430, RMSE 2.828 (epoch 81) | R² 0.5167, RMSE 5.694 (epoch 63) |
| Tier-3 final ridge on calibration sensors (n=77,127 / 77,138) | 0.5355 [0.411, 0.659] | 0.5690 [0.439, 0.697] |
| Spatial-block CV, Tier-1 (5 regions) | 0.4916 [0.433, 0.551] | 0.5009 [0.440, 0.564] |
| Temporal holdout, Tier-1 (test ≥ 2025-01-01; n=127,373 / 127,409) | 0.6640 [0.591, 0.721] | 0.6822 [0.610, 0.738] |
| Spatial vs temporal R² decomposition (Tier-1) | **spatial 0.049** / temporal 0.674 | **spatial 0.180** / temporal 0.686 |
| Per-day residual Moran's I, Tier-1 (median over 1,947 days) | −0.061 | −0.061 |

## C. Production comparison (raw target only — the single valid comparison)

The production Shared Skies ensemble reports **LOSO R² = 0.7136** on the
raw ATM target (as documented in this repository's README; the
`models/metrics.json` it cites lives in the production repo, not here, so
that figure is taken on trust). AQNet on the same uncorrected target:

| Model | LOSO R² | 95% CI | vs production 0.7136 |
|---|---|---|---|
| AQNet Tier-1 LOFO blend | 0.6141 | [0.545, **0.680**] | **−0.0995** |
| AQNet Tier-3 cross-fit | 0.5922 | [0.522, 0.653] | −0.1214 |

**AQNet does not beat the production baseline.** The shortfall is
statistically clear, not CI noise: the production number (0.7136) lies
*above the entire* sensor-clustered 95% CI of the AQNet Tier-1 blend
(upper bound 0.680). Caveats that keep this honest rather than damning:
the two systems were trained on different data vintages and sensor
populations, and production's 38 features include the four demographic
columns AQNet excludes by design — though §G shows those columns
contribute ≈ nothing here, so they do not explain the gap.

Do **not** compare the barkjohn-run numbers to 0.7136 — the corrected
target has ~4× less variance and the comparison is invalid.

## D. External validation — EPA AQS (62 FRM/FEM monitors, 86,316 site-days, never trained on)

| Row | barkjohn R² [95% CI] | barkjohn RMSE / bias | raw R² [95% CI] | raw RMSE / bias |
|---|---|---|---|---|
| Tier-1 (full-data fit) | **−0.048** [−0.123, +0.033] | 6.300 / −3.572 | **+0.077** [+0.002, +0.156] | 5.912 / −2.716 |
| Tier-3 (deployment-mode) | **−0.066** [−0.139, +0.010] | 6.352 / −3.756 | **+0.088** [−0.004, +0.177] | 5.876 / +1.122 |

By year (Tier-1 R²): barkjohn −0.13 / −0.17 / +0.01 / +0.08 / −0.18 /
+0.17 for 2021–2026; raw +0.07 / −0.03 / +0.16 / +0.22 / −0.09 / +0.21.

AQI-category skill at AQS (Tier-3): category accuracy 0.688 (barkjohn) /
0.708 (raw); exceedance (>35.4 µg/m³, 418 true site-days) recall 0.000
(barkjohn) / 0.297 (raw).

**Honest verdict: the models show essentially no site-day-level skill at
regulatory monitors.** R² against AQS is ≈ 0 on both targets (slightly
negative on corrected, ≈ +0.08 on raw), with a persistent negative bias
(−2.7 to −3.8 µg/m³ except deployment-mode raw at +1.1). Within-network
LOSO skill (§B) does not transfer to the FRM/FEM network in these runs.
This is the pre-registered strongest test, and the models fail it.

## E. Baseline comparison — does AQNet beat plain interpolation? (scientifically the key table)

Same LOSO fold discipline for the interpolation baselines; CTM priors need
no folds. All rows scored against each run's own target.

| Method | barkjohn R² | barkjohn RMSE | barkjohn n | raw R² | raw RMSE | raw n |
|---|---:|---:|---:|---:|---:|---:|
| Nearest sensor | 0.2859 | 3.661 | 293,543 | 0.3188 | 7.006 | 293,579 |
| IDW (k=8) | 0.5242 | 2.988 | 293,543 | 0.5416 | 5.748 | 293,579 |
| Ordinary kriging | 0.5430 | 2.929 | 293,543 | **0.5681** | 5.579 | 293,579 |
| CAMS prior (raw) | −0.9569 | 6.059 | 251,433 | 0.1317 | 7.905 | 251,469 |
| CAMS prior (mean-debiased) | −0.2956 | 4.930 | 251,433 | 0.1498 | 7.822 | 251,469 |
| GEOS-CF prior (raw) | −8.3483 | 13.090 | 259,421 | −0.7619 | 11.080 | 259,421 |
| GEOS-CF prior (mean-debiased) | −3.1649 | 8.737 | 259,421 | −0.3833 | 9.818 | 259,421 |
| MERRA-2 proxy (raw) | −5.6757 | 11.193 | 293,543 | −0.4424 | 10.196 | 293,579 |
| MERRA-2 proxy (mean-debiased) | −2.1784 | 7.723 | 293,543 | −0.2787 | 9.600 | 293,579 |
| **AQNet Tier-1 LOFO** | **0.5995** | **2.742** | 293,543 | **0.6141** | **5.273** | 293,579 |
| AQNet Tier-3 cross-fit | 0.5583 | 2.827 | 216,416 | 0.5922 | 5.332 | 216,441 |

Note the differing n: the CTM-prior rows cover only the days each product
exists (CAMS starts 2022-08; GEOS-CF ends 2026-01) — ~12–14% fewer rows
than the interpolation/AQNet rows, so those comparisons are not on an
identical row set.

Plain statement: **Tier-1 beats the best simple interpolation (ordinary
kriging) on both targets in point estimate, by modest margins** — ΔR² ≈
+0.057 (barkjohn) and +0.046 (raw), RMSE better by ~0.19 / 0.31 µg/m³ —
**but this win is not CI-separated**: the unpaired sensor-cluster CIs
overlap substantially (kriging [0.475, 0.608] vs Tier-1 [0.536, 0.665]
on barkjohn; [0.498, 0.635] vs [0.545, 0.680] on raw), and the pipeline
computes no paired model-vs-kriging test, so the margin over kriging
should be treated as suggestive, not established — unlike the §C
shortfall and §F fusion results, which are CI-separated. Every CTM prior
used directly as a prediction is far worse than interpolation — even
mean-debiased — so the raw-prior pattern skill at Texas daily scale is
poor; their value, such as it is, comes only as features (§G).

## F. Does fusion actually help? — No; it statistically significantly hurts.

Computed post-hoc from the runs' own OOF artifacts (`oof_tier1.npz`,
`oof_meta.npz`) on **identical rows**, paired ΔR² = R²(alt) − R²(Tier-1),
1,000-resample cluster bootstrap over sensors, seed 0
([`fusion_paired.json`](fusion_paired.json)):

| Comparison (identical rows) | barkjohn ΔR² [95% CI] | raw ΔR² [95% CI] |
|---|---|---|
| Tier-3 cross-fit vs Tier-1 (cross-fit rows; n≈216k, 255 sensors) | **−0.0500** [−0.0737, −0.0243] | **−0.0306** [−0.0515, −0.0076] |
| Tier-2 U-Net vs Tier-1 (U-Net val-site rows; n≈51k, 52 sensors) | −0.0631 [−0.1286, −0.0022] | −0.1004 [−0.1551, −0.0392] |

On both targets the Tier-3 stack is **worse than Tier-1 alone, and the CI
excludes zero** — the fusion claim is not merely unsupported, it is
contradicted at the 95% level in this configuration. Two observed
mechanisms visible in the artifacts: (i) the ridge gives the kriged
residual **zero weight** on both targets (`rk: 0.000`) — residual kriging
adds nothing once Tier-1 already carries leave-self-out neighbor
features; (ii) the ridge is fit on the ~34k rows where all three
components are finite (U-Net validation-site rows) yet gives the U-Net —
the weakest component — weight 0.32 (barkjohn) / 0.22 (raw), and at
prediction time the U-Net column is mean-filled on the 82.7% of rows
where it is masked (only 50,695 of 293,579 rows sit on U-Net
validation-site pixels; the all-components-finite ridge-fit set is
smaller still at ~34k rows), degrading the stack everywhere else.
(Paired-test config: 1,000 cluster-bootstrap resamples over sensors,
seed 0 — recorded here because `fusion_paired.json` stores only the
results.)

## G. Ablations (frozen LOSO folds; paired ΔR² vs primary, sensor-cluster bootstrap CIs)

| Variant | barkjohn R² | barkjohn ΔR² vs primary [95% CI] | raw R² | raw ΔR² vs primary [95% CI] |
|---|---:|---|---:|---|
| primary (45 features) | 0.5995 | (reference) | 0.6143¹ | (reference) |
| plus_demographics (49) | 0.5984 | −0.0011 [−0.0099, +0.0072] — n.s. | 0.6115 | −0.0027 [−0.0120, +0.0056] — n.s. |
| no_external (35) | 0.5959 | **−0.0036 [−0.0064, −0.0016]** — significant | 0.6108 | **−0.0035 [−0.0062, −0.0015]** — significant |
| no_neighbor (38) | 0.5885 | −0.0110 [−0.0259, +0.0062] — n.s. | 0.6080 | −0.0063 [−0.0194, +0.0093] — n.s. |

Readings: **demographic columns contribute nothing** (point estimates are
*negative*; the design exclusion costs no accuracy). The external
CTM/reanalysis features (GEOS-CF + MERRA-2) contribute a real but tiny
+0.0036 (barkjohn) / +0.0035 (raw) R². Removing all neighbor features costs more in
point estimate (−0.006 to −0.011) but is not significant under sensor
clustering — despite permutation importance ranking `nbr_pm25_100km` /
`25km` / `50km` as the top three features on both targets (correlated
features absorb the signal when the whole group is dropped and models
retrain).

¹ The ablation stage re-trains `primary` on the same frozen folds, so its
raw R² (0.61426) differs from §B's headline (0.61415) in the 4th decimal
— run-to-run nondeterminism of the GPU boosters, not an inconsistency.

## H. Uncertainty quantification (α = 0.1, nominal 90%)

| Item | barkjohn | raw |
|---|---|---|
| Pre-conformal q05–q95 band, empirical coverage (all LOSO rows) | 0.661 | 0.647 |
| Conformal δ (CQR, calibrated on 77k sensor-disjoint rows) | 1.759 µg/m³ | 2.981 µg/m³ |
| Conformalized coverage (meta-train cross-fit rows, disjoint from δ) | **0.882** | **0.880** |
| Mean interval width | 7.06 µg/m³ | 12.68 µg/m³ |

Conformalization lifts coverage from ~0.65 to ~0.88 but both runs land
~2 points **under** the 0.90 nominal — consistent with exchangeability
violation across sensors (calibration sensors ≠ evaluation sensors).

## I. Anything that looks wrong / caveats (read before citing numbers)

1. **The repo's GEOS-CF fetcher cannot work as shipped** against current
   xarray: the GrADS DODS server advertises `days since 1-1-1`, decodes
   ~500 years off onto cftime objects, and `sel(time=Timestamp)` throws
   `different calendars` for every month. Data was fetched by a sidecar
   that writes the repo's own cache-of-record month chunks with an
   **empirically calibrated** regular hourly axis (anchor = the
   documented stream start 2018-01-01 00:30 UTC). Validation evidence
   (measurements transcribed in
   [`geoscf_axis_calibration.md`](geoscf_axis_calibration.md); the
   repo's own GEOS-CF reduction cannot run at all, so no repo-code
   cross-comparison is possible for this source): the lag-correlation
   scan of daily Texas means against PurpleAir peaks at the chosen
   anchor (r = 0.711, cleanly separated from ±1-day neighbors), and the
   full-record sanity check preserved in the sidecar log peaks at lag 0
   vs CAMS (r = +0.286 vs +0.219/+0.212 at ±1 day). The DODS axis also
   carries 32 corrupted time values (all early-2018, outside the study
   window; per-glitch list in the same note), and **the server's assim
   stream ends 2026-01-02** — Feb–May 2026 has no GEOS-CF anywhere, and
   those days are NaN/flagged, as the design intends.
2. **The repo's MERRA-2 streaming path has never been runnable as
   shipped**: it requires `h5py` and `dask`, neither in
   `requirements.txt` (installed for the run); the account needed the
   GES DISC EULA/application authorization (one-time browser step); and
   the on-prem `goldsmr4` server no longer hosts 2025+ granules (fetched
   from the Earthdata-Cloud OPeNDAP endpoint instead). MERRA-2 data was
   fetched via server-side OPeNDAP subsetting and validated two ways:
   a single-day comparison against a raw downloaded granule reduced
   locally (observed max relative difference 0.0 on every variable —
   the gate threshold was rtol 1e-5; the observed result was exact),
   and a full-month (202101) comparison against the repo's **own**
   earthaccess-streamed `_merra2_month` reduction — identical keys, max
   relative difference ≈ 2×10⁻⁷ (float32 summation-order noise; PBLH
   max/min bit-identical). Full-window coverage is complete (942,348
   cell-days, zero nulls).
3. **External AQS transfer ≈ zero** (§D) with persistent under-prediction
   (−2.7 to −3.8 µg/m³). Note the corrected-target run is *more* negative
   at AQS than the raw run despite the Barkjohn correction nominally
   moving PurpleAir toward FRM scale.
4. **Fusion hurts** (§F): `rk` weight is exactly 0 on both targets and the
   stack underperforms Tier-1 with CIs excluding zero.
5. **Spatial skill is very low**: spatial (between-site) R² is 0.049
   (barkjohn) / 0.180 (raw) vs temporal 0.674 / 0.686 — the model tracks
   day-to-day variation well but ranks *locations* poorly, which matters
   for exposure-mapping uses; consistent with the weak AQS transfer.
6. **Exceedance detection on the corrected target is poor**: recall 0.068
   (Tier-1 barkjohn; 336 true events) vs 0.398 on raw (3,959 events) —
   variance compression plus Huber loss blunts event sensitivity.
7. Permutation baseline R² (0.790/0.800) is a **full-data fit scored on
   a sample of its own training rows** — an in-sample diagnostic; do not
   read it as generalization.
8. Environment deviations, all documented: pandas pinned to 2.3.3 (major
   3.0.5 was resolved by pip and this 2025-era codebase predates it);
   `h5py`, `dask` added; the `num_workers=0` patch was applied for the
   run and reverted before this commit. `versions.txt` reflects the final
   environment.
9. Wall-clock context: pipeline compute was ≈ 6.4 h total on this
   hardware; the overnight window was dominated by the two account-level
   blockers above (token + EULA, requiring user action) and the external
   fetch engineering, not by the pipeline itself.

---

### File map

- `barkjohn/`, `raw/` — each run's `SUMMARY.md`, all `metrics_*.json`,
  `permutation_report.json`, `shap_summary.png`, `unet_train.json`
  (Tier-2 best metrics + full per-epoch history)
- `fusion_paired.json` — Section-F paired tests (computed from the runs'
  OOF artifacts; seed 0, 1,000 cluster-bootstrap resamples)
- `geoscf_axis_calibration.md` — measured evidence for the GEOS-CF time
  axis repair (§I item 1)
- `versions.txt` — `pip freeze` of the exact environment
