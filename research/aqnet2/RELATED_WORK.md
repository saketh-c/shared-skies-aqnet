# AQNet in the literature: positioning, novelty audit, and adoptions

Compiled 2026-08-08 from a four-track survey (satellite/CNN estimation
family; GNN/attention family; authoritative exposure products and the
validation-critique literature; frontier methods: INRs, foundation models,
conformal UQ, multi-fidelity calibration). Full citations at the end of
each section; every claim below traces to a published source.

## 1. The field's validation problem, quantified by its own authors

Pooled or sample-based cross-validation, the customary headline protocol,
overstates out-of-network skill by a wide, measured margin:

| Study | Random/pooled CV | Honest spatial protocol |
|---|---|---|
| Geoi-DBN (Li 2017/2018, China) | R ~ 0.94 | R2 0.84 site-CV, 0.58 at 110 km from network |
| Wei 2023 global 1 km (4D-STET) | 0.91 sample | 0.87 station, 0.79 grid, 0.76 state-blocked |
| van Donkelaar 2024 N. America | 0.62-0.66 random | 0.36-0.43 buffered cluster CV (BLeCO) |
| ACAG V6 CNN (Shen 2024) | 0.86 global spatial-CV | 0.57 North America regional |
| Di 2019 ensemble (Harvard 1 km) | 0.86 site-random CV | 0.07 vs independent USFS smoke monitors (Considine 2023); 0.00 on smoke days |
| DeepAir (Guo 2025, California) | -- | 0.58 station-grouped CV |
| STARQ (Wang 2026, Europe) | -- | 0.24 at unseen stations x unseen times |
| Aurora 1.3B foundation model | -- | 0.23-0.38 PM2.5 at Chinese stations |

AQNet's numbers belong in the right-hand column by construction: v2 Texas
strict spatially-blocked R2 0.33 (vault-confirmed at 0.335), pooled LOSO
diagnostic 0.52 (labeled, never headline); v3 WEST7 strict spatial 0.32
pooled across seven states (CA 0.36) against a near-zero CTM floor outside
Texas. Under matched honesty AQNet is competitive with the flagship
products (BLeCO 0.36 biweekly) and ahead of the strictest published
protocol (STARQ 0.24), while being the only entry with a sealed one-shot
vault agreeing with its cross-validation estimate.

## 2. Novelty audit: what no surveyed work does

1. Pre-registered admission gates / non-inferiority margins / sealed
   holdout in AQ exposure modeling: not found anywhere 2015-2026. Nearest
   analogs are real-time forecast competitions (Forecast Rodeo, AI Weather
   Quest) and prospective ungauged-basin evaluation in hydrology (Nearing
   2024). CNN-family survey: no paper statistically gates components;
   ablations are point-estimate deltas. GNN survey: 1 of 13 papers runs any
   significance test (AirFormer, on run-to-run noise only); zero test
   deltas at held-out stations.
2. A fair three-way comparison of graph deep learning vs gradient boosting
   vs kriging at held-out sites: zero published instances. Papers with any
   GBM baseline: 3 of 13 (fed pointwise features without neighbor access);
   with classical kriging: 4 of 13; with both: none. AQNet's T2-vs-T1
   admission test (graph attention vs boosted ensemble + daily kriging,
   spatially blocked, cluster-bootstrap significance) is that comparison.
3. Three-way gated fusion of regulatory monitors + low-cost sensors + CTM:
   flagged an unclaimed niche by the frontier survey. Closest pairs:
   Baltimore multi-network Bayesian calibration (monitors+LCS), Hamburg
   robust multi-fidelity GP (monitors+LCS), GHAP (CTM-as-feature, no
   gating).
4. Conformal intervals validated at held-out US regulatory sites: no
   product ships this. The two applied conformal-AQ papers (Africa 2026,
   GeoConformal wildfire 2026) document the marginal guarantee FAILING
   under spatial shift; AQNet's P4 site-level coverage battery at held-out
   sites is the artifact they call for.
5. A coordinate-network (INR) decoder trained against monitors and
   validated at held-out stations: unclaimed. The one AQ-native INR
   (HF-SDF, npj 2025) validates against gridded reanalysis, never
   stations. AQNet's T3 is architecturally first in this slot even though
   the WEST7 gates refused it (a result, not a defect).

## 3. Corrections this survey forces on our own claims

* The Texas calibration-bias finding has partial precedent and must cite
  it: AMT 2024 published a southeastern-US high-humidity regional refit
  (RH+T multilinear, 16-23% gains over national constants); Jaffe 2023
  documented ~6x dust-event underestimation; Barkjohn 2021 itself flags
  southern-state undersampling. Correct claim: the largest-sample regional
  quantification for the south-central US (201k held-out pair-days) with
  pre-registered, gate-adjudicated form selection.
* Neighbor-monitor features are standard in the strongest US models (Di,
  Hu, DeepAir). Our distinction is leakage handling: DeepAir leaves open
  whether held-out monitors were excluded from its nearest-neighbor
  features; AQNet recomputes every neighbor feature per fold
  (nbr_overrides), which should be stated as the contrast.
* Raw-coordinate ban: independently supported by Wang 2025 (location
  encoders): raw lat/lon helps within-region and hurts cross-region
  transfer. Cite as external validation of the design rule.

## 4. Adoptions (concrete, ordered by value/cost)

1. Report the full validation ladder (sample -> station -> spatial-block ->
   state) plus a distance-decay curve: R2 vs distance-to-nearest-training-
   site (Li & Shen 2018 buffered protocol; ACAG B-LOO shows information
   leaks to ~150 km). Cheap: computable from existing OOF arrays.
2. Print the matched-protocol comparison table (section 1) in the paper;
   it is the strongest positioning artifact available.
3. Register an external-truth ablation (A18): score the frozen WEST7
   composite against USFS AirFire mobile smoke monitors (Considine 2023
   protocol), which sit outside the AQS network entirely. Public data;
   no refit; the hardest exam in the literature.
4. Adopt the Barkjohn 2022 piecewise extension for the >300 ug/m3
   nonlinearity in any future smoke-regime calibration work (A16 family).
5. Frame coverage-pattern gating in Meyer & Pebesma (2022) area-of-
   applicability terms; our pattern-conditional gates are an operational
   implementation of that proposal.
6. Cite the Considine/Just/Ploton critique line as the motivation for the
   harness; cite Wei 2023 as the best-practice ladder precedent.

## 5. Key sources

Products: van Donkelaar 2021 (ES&T 55:15287), Shen 2024 (ACS EST Air),
van Donkelaar 2024 (ACS EST Air, BLeCO), EPA FAQSD/Downscaler (Berrocal
2010/2012/2020), Di 2019 (Environ Int 130:104909), Wei 2023 (Nat Commun
14). Critiques: Considine 2023 (EST 57), Just 2020 (Atmos Environ), Meyer
& Pebesma 2022 (Nat Commun 13:2208), Li & Shen 2018 (arXiv:1812.00135),
Linnenbrink 2026 (arXiv:2605.13689). Calibration: Barkjohn 2021 (AMT
14:4617), Barkjohn 2022 (Sensors 22:9669), Jaffe 2023 (AMT 16:1311), AMT
2024 (17:6735, high-RH regional), Wallace 2022 (Sensors). CNN family: Di
2016 (EST), Hu 2017 (EST), Li 2017 (GRL), Park 2020 (Environ Pollut), Yan
2020 (Environ Int), Li/Habre 2020 (Environ Int), Kayastha 2024 (AIES),
Guo 2025 DeepAir (MLST), Wang 2026 STARQ (arXiv:2607.05292), Gupta 2024
(arXiv:2404.07308). GNN family: Qi 2019 GC-LSTM (STOTEN), Wang 2020
PM2.5-GNN (SIGSPATIAL), Chen 2021 GAGNN (TKDD), Liang 2023 AirFormer
(AAAI), Hettige 2024 AirPhyNet (ICLR), Appleby 2020 KCN (AAAI), Wu 2021
IGNNK (AAAI), Xu 2025 KITS (AAAI), Wang 2025 SPIN (arXiv:2511.16013),
Wang 2025 AirRadar (AAAI), Iyer 2022 (npj Clim Atmos Sci 5:76). Frontier:
Zhang 2025 HF-SDF (npj Clim Atmos Sci), Bodnar 2025 Aurora (Nature), Gui
2026 AI-GAMFS (Nature), Keller 2021 GEOS-CF (JAMES), TransNet 2026 (npj
Clean Air), conformal-AQ (Atmosphere 17:692; arXiv:2604.22787), Hamburg
multi-fidelity (arXiv:2511.15942), Baltimore unified calibration
(arXiv:2412.13034), Wang 2025 location encoders (arXiv:2505.18461).
