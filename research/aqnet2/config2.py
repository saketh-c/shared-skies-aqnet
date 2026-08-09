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

# ── Domain (AQNet v2 = tx, AQNet v3 = west7; EXPANSION.md) ──────────────────
#
# The domain is selected by env (AQNET2_DOMAIN) so one codebase serves both
# runs. Everything domain-dependent — state list, bbox, artifacts namespace,
# fold-system sizes, the AQS parquet stem — resolves from DOMAIN_SPEC here;
# no consumer may hardcode "tx"/"v2".

DOMAIN = os.environ.get("AQNET2_DOMAIN", "tx").lower()

_DOMAINS = {
    # AQNet v2 primary (results/v2_texas_202608)
    "tx": {
        "states": ["48"],                      # TX
        "bbox": {"lat_min": 25.6, "lat_max": 36.7,
                 "lon_min": -107.0, "lon_max": -93.3},
        "artifacts": "v2",
        "aqs_stem": "aqs_daily_tx_v2",         # frozen: matches shipped data
        "vault_n_sites": 12,
        "outer_n_folds": 5,
        "pa_states": ["48"],
    },
    # AQNet v3 Phase 1 (EXPANSION.md): CA TX WA CO UT NV AZ, no new PA.
    "west7": {
        "states": ["06", "48", "53", "08", "49", "32", "04"],
        "bbox": {"lat_min": 25.6, "lat_max": 49.1,
                 "lon_min": -124.8, "lon_max": -93.3},
        "artifacts": "v3",
        "aqs_stem": "aqs_daily_west7_v3",
        "vault_n_sites": 30,
        "outer_n_folds": 8,
        "pa_states": ["48"],                   # Phase 1: TX archive only
    },
}
if DOMAIN not in _DOMAINS:
    raise SystemExit(f"[config2] unknown AQNET2_DOMAIN {DOMAIN!r} "
                     f"(known: {sorted(_DOMAINS)})")
DOMAIN_SPEC = _DOMAINS[DOMAIN]
STATE_FIPS = list(DOMAIN_SPEC["states"])
PA_STATE_FIPS = list(DOMAIN_SPEC["pa_states"])
AQS_STEM = DOMAIN_SPEC["aqs_stem"]

# ── Paths ───────────────────────────────────────────────────────────────────

# aqnet2 sits at <ROOT>/research/aqnet2
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AQNET2_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(AQNET2_DIR, "data")
CACHE_DIR = os.path.join(AQNET2_DIR, "cache")
# AQNET2_ARTIFACTS_TAG namespaces a run's artifacts away from the domain
# default so a new chain (e.g. v4) can never overwrite a shipped bundle.
# Unset = the frozen per-domain default.
_ART_TAG = os.environ.get("AQNET2_ARTIFACTS_TAG", "").strip()
if _ART_TAG and not all(c.isalnum() or c in "_-" for c in _ART_TAG):
    raise SystemExit(f"[config2] bad AQNET2_ARTIFACTS_TAG {_ART_TAG!r} "
                     "(letters, digits, _ and - only)")
ARTIFACTS_DIR = os.path.join(AQNET2_DIR, "artifacts",
                             _ART_TAG or DOMAIN_SPEC["artifacts"])
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


def canonical_aqs_path():
    """The one AQS daily parquet every consumer must read, or None.

    Preference: the data-stage registry (external_paths.json 'aqs' entry,
    written under the widest-window rule), then the widest-window (never
    lexicographically-last) v2 parquet under DATA_DIR. A narrow quick-window
    refetch such as aqs_daily_tx_v2_2024_2024.parquet sorts AFTER the full
    aqs_daily_tx_v2_2021_2026.parquet, so any sorted(glob)[-1] consumer
    silently trains on one year — folds2 and calibrate both did until every
    caller was routed through here. Callers fall back to their own committed
    v1 defaults on None.
    """
    import glob as _glob
    import json as _json
    import re as _re
    reg = os.path.join(ARTIFACTS_DIR, "external_paths.json")
    try:
        with open(reg) as fh:
            p = _json.load(fh).get("aqs")
        if p and os.path.exists(p):
            return p
    except Exception:
        pass

    def _span_mtime(p):
        m = _re.search(r"_(\d{4})_(\d{4})\.parquet$", os.path.basename(p))
        span = (int(m.group(2)) - int(m.group(1))) if m else -1
        try:
            mt = os.path.getmtime(p)
        except OSError:
            mt = 0.0
        return (span, mt)

    v2s = sorted(_glob.glob(os.path.join(DATA_DIR, AQS_STEM + "_*.parquet")),
                 key=_span_mtime, reverse=True)
    return v2s[0] if v2s else None


# ── Domain bbox & dates ─────────────────────────────────────────────────────

# Name kept from v1/v2 for the many existing consumers; semantically this is
# THE DOMAIN bbox (tx: Texas; west7: the WEST7 envelope).
TX_BBOX = dict(DOMAIN_SPEC["bbox"])
GRID_DEG = 0.1

DATE_START = "2021-01-01"
DATE_END = "2026-05-01"
TEMPORAL_CUTOFF = "2025-01-01"   # temporal holdout: train < cutoff, test >=
TEMPORAL_EMBARGO_DAYS = 7        # lagged features embargoed at the cutoff

# ── Fold system (DESIGN §2; folds2.py is the sole builder) ──────────────────

VAULT_N_SITES = int(DOMAIN_SPEC["vault_n_sites"])  # one-shot AQS vault
VAULT_BUFFER_KM = 30.0      # every vault site >= this far from non-vault
VAULT_DATE_START = "2026-01-01"  # vault period: all data from here onward
OUTER_N_FOLDS = int(DOMAIN_SPEC["outer_n_folds"])  # spatially-blocked outer
INNER_N_FOLDS = 4           # folds 0-1 selection, 2-3 confirmation
LOSO_N_FOLDS = 10           # unit-grouped LOSO nested within each outer fold
# AQNET2_SEED_OFFSET shifts every seeded draw for a new run generation.
# The v3 vault sites are revealed (their one-shot numbers are published),
# so a rerun at the same seed would reproduce the same "sealed" vault; a
# registered nonzero offset draws fresh folds and a fresh vault. Unset =
# 0 = frozen behavior.
_SEED_OFF = os.environ.get("AQNET2_SEED_OFFSET", "0").strip() or "0"
try:
    _SEED_OFF = int(_SEED_OFF)
except ValueError:
    raise SystemExit(f"[config2] bad AQNET2_SEED_OFFSET {_SEED_OFF!r} "
                     "(integer required)")
SEED = 42 + _SEED_OFF

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
