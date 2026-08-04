"""Shared configuration for AQNet v2 — the FRM-anchored residual ladder.

Sibling of v1's research/aqnet/config.py, with the v2 additions that encode
methodology as data (DESIGN.md is the contract; INTERFACES.md freezes the
names exported here):

  * The fold-system constants (vault size/buffer, outer/inner/LOSO counts)
    live here so folds2.py and every consumer agree by construction.
  * The gate constants (alpha grid, min clusters, 100-km hard-zero radius,
    T4 slope clip) are data, not prose — compose.py reads them directly.
  * EXCLUDED_DEMOGRAPHIC carries the same four names as v1; frame2.py bans
    them (plus raw lat/lon and dist_to_nearest_sensor) from every feature
    list and asserts at build time.

Directories are created on import so any module can run first, locally or on
Phoenix. artifact() gains an optional subdir over v1's version because the
v2 namespace is per-config (DESIGN §12: the v1 flat namespace could silently
overwrite a raw-target run with a barkjohn one).
"""
import os

# ── Paths ───────────────────────────────────────────────────────────────────

# aqnet2 sits at <ROOT>/research/aqnet2
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AQNET2_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(AQNET2_DIR, "data")
CACHE_DIR = os.path.join(AQNET2_DIR, "cache")
ARTIFACTS_DIR = os.path.join(AQNET2_DIR, "artifacts", "v2")
V1_DIR = os.path.join(os.path.dirname(AQNET2_DIR), "aqnet")
DL_DIR = os.path.join(os.path.dirname(AQNET2_DIR), "deeplearning")
PIPELINE_DIR = os.path.join(ROOT, "pipeline")

for _d in (DATA_DIR, CACHE_DIR, ARTIFACTS_DIR):
    os.makedirs(_d, exist_ok=True)


def artifact(name, sub=""):
    """Absolute path for a named artifact under ARTIFACTS_DIR[/sub].

    The subdir is created on demand so producers can namespace per config
    (e.g. artifact("metrics_loso.json", "ablation_no_statics")).
    """
    base = os.path.join(ARTIFACTS_DIR, sub) if sub else ARTIFACTS_DIR
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, name)


# ── Domain & dates (identical to v1 — the reconstruction window) ────────────

TX_BBOX = {"lat_min": 25.6, "lat_max": 36.7, "lon_min": -107.0, "lon_max": -93.3}
GRID_DEG = 0.1

DATE_START = "2021-01-01"
DATE_END = "2026-05-01"
TEMPORAL_CUTOFF = "2025-01-01"   # temporal holdout: train < cutoff, test >=
TEMPORAL_EMBARGO_DAYS = 7        # lagged features embargoed at the cutoff

# ── Fold system (DESIGN §2; folds2.py is the sole builder) ──────────────────

VAULT_N_SITES = 12          # one-shot AQS vault, touched once by validate
VAULT_BUFFER_KM = 30.0      # every vault site >= this far from non-vault
VAULT_DATE_START = "2026-01-01"  # vault period: all data from here onward
OUTER_N_FOLDS = 5           # spatially-blocked outer folds over AQS sites
INNER_N_FOLDS = 4           # folds 0-1 selection, 2-3 confirmation
LOSO_N_FOLDS = 10           # unit-grouped LOSO nested within each outer fold
SEED = 42

# ── Feature contract ────────────────────────────────────────────────────────

# Demographic EJScreen columns — excluded from prediction everywhere.
# (Physical source-proximity features remain; the ablation arm that adds
# these back is built manually and never ships.)
EXCLUDED_DEMOGRAPHIC = ["ejf_score", "pct_people_of_color", "pct_low_income",
                        "pct_ling_isolated"]

# Feature-set split (DESIGN §6). frame2.feature_columns() emits the concrete
# names; these prefix contracts classify them. The interpolating set is
# coverage-gated downstream (its absence must degrade to the portable set,
# never to a fill), so the split is data other modules can assert against.
INTERP_FEATURE_PREFIXES = ("nbr_",)
PORTABLE_FEATURE_PREFIXES = ("t0_", "era5_", "merra2_", "geoscf_", "cams_",
                             "maiac_", "st_")
PORTABLE_FEATURES_EXACT = ["aod", "dust", "hms_smoke", "dist_to_coast",
                           "doy_sin", "doy_cos", "dow_sin", "dow_cos",
                           "shortwave", "et0", "cloud_cover"]


def is_interp_feature(name):
    """True when a feature belongs to the coverage-gated interpolating set."""
    return any(name.startswith(p) for p in INTERP_FEATURE_PREFIXES)


def split_feature_sets(columns):
    """Partition feature names -> (portable, interpolating), order preserved."""
    portable, interp = [], []
    for c in columns:
        (interp if is_interp_feature(c) else portable).append(c)
    return portable, interp


# Kept as plain lists for INTERFACES conformance/documentation; the live
# split always comes from split_feature_sets over frame2.feature_columns().
PORTABLE_FEATURES = list(PORTABLE_FEATURES_EXACT)
INTERP_FEATURES = []  # populated per-frame; nbr_* names are frame-dependent

# ── Targets, weights, gates (DESIGN §1, §6, §9) ─────────────────────────────

SIGMA_FRM = 1.5             # FRM/FEM daily measurement sigma (µg/m³)

GATE_ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
GATE_MIN_CLUSTERS = 5       # strata below this shrink to zero (passthrough)
GATE_MAX_DIST_KM = 100.0    # hard alpha = 0 beyond this from any live station
T4_SLOPE_CLIP = (0.8, 1.25)

EXCEED_THRESHOLDS = (9.0, 35.4)   # EPA 2024 annual / daily NAAQS breakpoints

# Conformal calibration units: disjoint PA-sensor fraction (DESIGN §2).
CONFORMAL_PA_FRAC = 0.25
