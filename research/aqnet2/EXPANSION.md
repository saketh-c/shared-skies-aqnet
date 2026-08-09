# AQNet v3 — pre-registered multistate domain expansion (WEST7)

Registered 2026-08-06, before any multistate data acquisition. Purpose: give
the deep residual tiers (T2 graph-attention, T3 neural field) an admission
test with the statistical power the Texas domain could not provide. v3 is a
DOMAIN configuration of the v2 codebase (same pipeline, same gates), not a
fork: `AQNET2_DOMAIN=west7`, artifacts under `artifacts/v3`.

## Phase 1 amendment (owner-approved 2026-08-06): NO new PurpleAir

Phase 1 runs WITHOUT PurpleAir acquisition for the six new states: Texas
keeps its existing calibrated PA archive; the new states run on the
portable feature set (the registered coverage-gated split — absent
covariates stay NaN, never filled). The deep-tier admission test is powered
by FRM clusters (~285) and pairs T2/T3 against T1 on identical rows, so it
is valid without PA. This also sharpens the claim under test: does deep
spatial learning add confirmable skill in a monitors-only network — the
regime most of the world is in. PA for the new states is a registered
LATER upgrade (owner gate: API spend), expected to raise the incumbent T1
and re-open every gate for re-adjudication.

Sensor-volume note (owner flag): California alone has O(10k) PA sensors;
when the PA phase runs, acquisition uses the registered selection rule —
every sensor within 25 km of an FRM monitor (calibration pairs) plus a
per-10-km-cell cap elsewhere, long-record preferred (covariate field) —
cutting CA to ~2-4k sensors with negligible modeling loss (the nbr_pacal_*
features average within radii and saturate with density).

Honest expectation, stated pre-run: pooled R2 may land BELOW the Texas-only
number even as the model improves — CA terrain, coastal gradients and
extreme smoke make the exam harder (the v1 0.71 -> v2 0.33 lesson). The
run's claims are the paired ladder deltas and the admission verdicts, not
"the number went up".

## Why (from the v2_texas_202608 primary run)

The T2 raw pooled preview was +0.066 R2 over T1 at held-out sites, but the
admission machinery — correctly — could not confirm it: at 49 site-clusters
the pooled-R2 CI half-width in the paired tests ran ~0.077, so a true
+0.05-0.07 effect is statistically invisible. The gates returned
"insufficient evidence at this density", not "no effect". This document
registers the follow-up rather than quietly retrying until something passes.

## Power targets (from the v2 test geometry, half-width ~ 1/sqrt(n))

| detectable delta (pooled R2) | clusters needed |
|---|---|
| 0.07 | ~60 |
| 0.05 | ~120 |
| 0.03 | ~320 |

## Domain: WEST7

CA (126 FRM/FEM PM2.5 sites with >=90 obs-days in 2024), TX (53), WA (22),
CO (24), UT (23), NV (20), AZ (~18) — ~285 sites, shared western airsheds,
smoke + dust + urban regimes, and the densest PurpleAir coverage in the
country (the calibrated-PA covariate is v2's top feature; the expansion must
not starve it). ~5.5x the Texas cluster count -> MDE ~0.033: a true
+0.05 deep-tier effect becomes confirmably visible.

Registered hypothesis: at >=250 clusters, T2 clears admission
(non-inferiority + superiority on pooled_r2) iff its Texas preview reflected
a real effect >=0.04. Either outcome is reported.

## What changes in code (no methodology changes)

* config2: BBOX/state lists become per-domain config (TX_BBOX -> DOMAIN);
  artifacts namespace `v2ms`.
* fetchers2: AQS fetch parameterized by state list (same API); statics
  fetchers take the domain bbox; MERRA-2/GEOS-CF subsetting widens.
* folds2: same fold machinery over ~285 sites (vault scales to ~30 sites,
  same buffer rule; outer folds ~8-10).
* calibrate: per-state colocation pairing, same G0 protocol (registered:
  the calibration form is re-adjudicated on the multistate pair record).
* GPU tiers: identical architectures; graphpre/fieldpre may warm-start from
  the Texas encoders (documented; self-supervised pretraining continues on
  the expanded data either way).

## Registered v3 run defaults

* `AQNET2_FORCE_ESCAPE=1`: T1 candidate A (GPBoost Vecchia) is escaped by
  default in v3. Evidence: the v2 full run's first candidate-A fit exceeded
  3.4 h wall (~190 h projected vs the 10 h budget), and the v3 QUICK smoke's
  47k-row timing fit alone ran >50 min without completing — at 5-8x v2 scale
  the full protocol is out of budget by orders of magnitude. Candidate B
  (ensemble + krige, the v2 shipped T1) carries the ladder; the decision is
  recorded per-run in oof_tier1.npz weights_json as decision=budget_escape.
* GEOS-CF and the Texas-scoped statics population column are ABSENT for the
  west7 states (loud omission, never a Texas fill) until domain-wide
  fetchers are written; the T0 prior runs on its remaining streams.

## Registered v4 feature additions (require refit + re-adjudication)

* EDGAR v8 gridded sectoral emissions (0.1 deg, global, public): adds
  area-source emission covariates (traffic, residential, industry) beside
  the existing NEI point sources and road-density proxies. Globally
  available, so it also serves the portability claim. Fetch is free;
  admission of any resulting skill change goes through the gates like
  everything else.
* Neighbor-feature continuous taper. Implemented 2026-08-09 as exponential
  distance decay w = exp(-d/(r/2)) with support truncated at the registered
  radius r, emitted as additional gated columns (env AQNET2_NBR_TAPER,
  default off) beside the hard-cutoff originals. Truncation keeps the
  neighbor set identical to the cutoff variant so the A/B ablation isolates
  the weighting alone; the information-horizon seam is therefore softened
  (weight exp(-2) at the cutoff), not fully removed. Adjudicated: support
  parity for clean ablation outweighs full seam removal; revisit only if
  the tapered variant admits and seams remain visible in served maps.

## Decision gates (owner)

1. PurpleAir historical acquisition. LESSON RECORDED 2026-08-09: PurpleAir
   bills per FIELD VALUE, not per row. Correct cost formula: rows x
   n_fields = points; verify against the account dashboard at one-sensor
   scale before any volume pull. The original per-row formula here was
   wrong by 6x and exhausted the owner's ~60M point allotment at 43% of
   the Tier B pull. Any future acquisition is BLOCKED until the owner
   approves a quote computed in field-values and dashboard-verified.
2. Compute: full multistate run ~5-8x Texas CPU + GPU hours; inferno-class
   spend needs re-authorization at that scale.

## Timeline (once gates clear)

Config + fetch ~4-6 days (PA fetch is the long pole), primary chain ~3-5
days of compute, admission verdict + report ~1 day. ~2-3 weeks calendar.
