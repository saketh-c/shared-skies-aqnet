"""Shared configuration for AQNet, the offline research PM2.5 track.

AQNet (research/aqnet) is a publication-oriented three-tier fusion model
built alongside — never inside — the production system that powers the live
map. This module centralizes the paths, Texas domain, date bounds, and the
feature-name contract that every sibling module imports.

Two methodology rules are encoded here as data rather than prose:

  * EXCLUDED_DEMOGRAPHIC names are never model inputs anywhere in AQNet.
    PHYSICAL_FEATURES is the production 38-feature list with those names
    removed; features.feature_columns() asserts against the exclusion list.
  * EPA AQS observations are reserved for external validation only, so the
    AQS pull years live here but no AQS-derived feature name does.

Directories (data/, cache/, artifacts/) are created on import so any module
can be run first, locally or in Colab.
"""
import os
import json

# ── Paths ───────────────────────────────────────────────────────────────────

# aqnet sits at <ROOT>/research/aqnet
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AQNET_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(AQNET_DIR, "data")
CACHE_DIR = os.path.join(AQNET_DIR, "cache")
ARTIFACTS_DIR = os.path.join(AQNET_DIR, "artifacts")
for _d in (DATA_DIR, CACHE_DIR, ARTIFACTS_DIR):
    os.makedirs(_d, exist_ok=True)


def artifact(name):
    """Absolute path for a named artifact under ARTIFACTS_DIR."""
    return os.path.join(ARTIFACTS_DIR, name)


# ── Domain & dates ──────────────────────────────────────────────────────────

TX_BBOX = {"lat_min": 25.6, "lat_max": 36.7, "lon_min": -107.0, "lon_max": -93.3}
GRID_DEG = 0.1

DATE_START = "2021-01-01"
DATE_END = "2026-05-01"
TEMPORAL_CUTOFF = "2025-01-01"  # temporal_split: train < cutoff, test >= cutoff

# ── Feature contract ────────────────────────────────────────────────────────

# Demographic EJScreen columns — excluded from prediction everywhere in AQNet.
# (Physical source-proximity features such as traffic_proximity remain.)
EXCLUDED_DEMOGRAPHIC = ["ejf_score", "pct_people_of_color", "pct_low_income",
                        "pct_ling_isolated"]

_FEATURE_NAMES_JSON = os.path.join(ROOT, "models", "feature_names.json")

# Mirror of models/feature_names.json (the production 38 features), used only
# if that file is unavailable (e.g. a partial checkout on a worker).
_PRODUCTION_FEATURES_FALLBACK = [
    "humidity", "temperature", "pressure", "wind_speed", "precipitation",
    "ejf_score", "pct_people_of_color", "pct_low_income",
    "traffic_proximity", "superfund_proximity", "rmp_proximity",
    "diesel_pm_proximity", "pct_ling_isolated",
    "latitude", "longitude", "dist_to_nearest_sensor", "dist_to_coast",
    "nbr_pm25_25km", "nbr_count_25km", "nbr_pm25_50km", "nbr_count_50km",
    "nbr_std_50km", "nbr_pm25_100km", "nbr_count_100km",
    "hms_smoke", "aod", "cams_pm25",
    "month", "dow", "day_of_year",
    "month_sin", "month_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "temp_x_humidity", "wind_x_temp",
]


def _load_production_features():
    """The production feature list, read from models/feature_names.json."""
    try:
        with open(_FEATURE_NAMES_JSON) as f:
            names = json.load(f)
        if isinstance(names, list) and names:
            return [str(n) for n in names]
    except (OSError, ValueError):
        pass
    return list(_PRODUCTION_FEATURES_FALLBACK)


PRODUCTION_FEATURES = _load_production_features()

# Production list minus the demographic exclusions, order preserved (34 names).
PHYSICAL_FEATURES = [f for f in PRODUCTION_FEATURES if f not in EXCLUDED_DEMOGRAPHIC]

_leak = set(PHYSICAL_FEATURES) & set(EXCLUDED_DEMOGRAPHIC)
if _leak:
    raise RuntimeError(f"Demographic features leaked into PHYSICAL_FEATURES: {sorted(_leak)}")

# Extra tabular features beyond the production physical list. "dust" is the
# CAMS dust-aerosol column that features.load_sensor_days joins by nearest
# cell alongside aod/cams_pm25; feature_columns() appends whichever of these
# are present after the physical features, before the externals.
EXTRA_FEATURES = ["dust"]

# External-data feature names (NaN wherever a source was not fetched).
# merra2_pblh_max / merra2_pblh_min are TABULAR-ONLY: the U-Net merra2
# channel group keeps its original six channels, so grids.py must exclude
# them (alongside merra2_pm25_proxy) when deriving its channel list.
MERRA2_FEATURES = ["merra2_dust25", "merra2_oc", "merra2_bc", "merra2_so4",
                   "merra2_ss25", "merra2_pm25_proxy", "merra2_pblh",
                   "merra2_pblh_max", "merra2_pblh_min"]
GEOSCF_FEATURES = ["geoscf_pm25"]

# ── External sources ────────────────────────────────────────────────────────

AQS_YEARS = list(range(2021, 2027))
GEOSCF_OPENDAP = ("https://opendap.nccs.nasa.gov/dods/gmao/geos-cf/assim/"
                  "chm_tavg_1hr_g1440x721_v1")
