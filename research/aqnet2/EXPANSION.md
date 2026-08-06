# AQNet v2-multistate — pre-registered domain expansion

Registered 2026-08-06, before any multistate data acquisition. Purpose: give
the deep residual tiers (T2 graph-attention, T3 neural field) an admission
test with the statistical power the Texas domain could not provide.

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

## Decision gates (owner)

1. PurpleAir historical acquisition for 6 new states — API point spend.
   Cost formula: sensors x months x (rows/month) x per-row point price;
   precise quote requires one cheap sensor-index call per state bbox plus
   current pricing. BLOCKED until owner approves the quoted number.
2. Compute: full multistate run ~5-8x Texas CPU + GPU hours; inferno-class
   spend needs re-authorization at that scale.

## Timeline (once gates clear)

Config + fetch ~4-6 days (PA fetch is the long pole), primary chain ~3-5
days of compute, admission verdict + report ~1 day. ~2-3 weeks calendar.
