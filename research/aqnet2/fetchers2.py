"""AQNet v2 fetchers (stages `data-pa`, `data`, `statics`) — DESIGN §12.

Everything here follows the v1 data_external.py contract (month chunks that
cache independently, {dest}.failed.json sidecars so a partial assembly is
never mistaken for a complete one, window-stamped final filenames so a
--quick pull never masquerades as the full window, atomic tmp+os.replace
writes). The v1 machinery itself (_download, _month_edges, sidecar helpers,
fetch_aqs_daily_tx, fetch_geoscf_pm25, fetch_merra2) is REUSED via a lazy
sys.path import of config2.V1_DIR rather than reimplemented.

What v2 adds (audit 06-pace.md is the ground truth for why):

  fetch_aqs_v2          hardened AQS refetch. The committed v1 parquet
                        (86,316 rows) lacks POC, method and site metadata,
                        so S1's FEM-vs-FRM flag and the urban/rural flag are
                        impossible from it. Same quality filters as v1
                        (Event Type "Excluded" dropped, sub-daily rows need
                        Observation Percent >= 75), plus retained POC,
                        is_fem from Method Name, and Location Setting from
                        the aqs_sites.zip listing. Window-stamped dest
                        closes the documented v1 quick/full cache-poisoning
                        hazard (DESIGN §12.2). REQUIRED: hard-fails when no
                        year could be fetched.
  fetch_merra2_slv      M2T1NXSLV T2M/QV2M/U10M/V10M/PS via GES DISC
                        OPeNDAP subsetting (bbox-clipped requests — the SLV
                        collection is NOT in the 909G granule cache and no
                        full-granule downloads are sanctioned, BUILD_NOTES
                        scope decision 2). ERA5 point met is blocked on the
                        cluster (no ~/.cdsapirc), so this is the final grid
                        met gap-filler behind pipeline/met_extra_by_cell.
                        OPTIONAL: failure degrades to a sidecar + NaN
                        features downstream.
  fetch_merra2_combined ensures the v1 aerosol parquet exists (reassembled
                        from the granule cache by v1 fetch_merra2 — no
                        re-download) and outer-merges the SLV parquet on
                        (lat, lon, date). Both collections share the native
                        0.5 x 0.625 grid; frame2 joins nearest-cell so a
                        one-sided merge is still usable.
  fetch_geoscf_domain   GEOS-CF surface PM2.5 daily means over the DOMAIN
                        bbox (config2.TX_BBOX), month-chunked to a
                        domain-stamped cache and a domain-stamped final
                        [lat, lon, date, geoscf_pm25]. The endpoint routing
                        (assim v1 tree before GEOSCF_V2_START, ana v2 tree
                        after) and the netCDF4-then-pydap open with its
                        hand-rebuilt GrADS time axis are v1's committed
                        'union GEOS-CF fix', reused via the bridge; only
                        the bbox clip is re-ported, because v1's is frozen
                        to Texas. Run for NON-tx domains only: tx keeps the
                        committed v1 Texas parquet (write_external_paths
                        candidates unchanged). OPTIONAL: warn-and-continue.
  fetch_hms_grid        NOAA HMS smoke polygons -> 0.1-degree cell raster
                        (pyshp + matplotlib.path, no GDAL). frame2.hms_join
                        treats a missing (cell, day) INSIDE the coverage
                        window as tier 0 (no polygon = no smoke), so only
                        smoke-positive cells are materialized; a 404 day is
                        an unmapped day and is documented as ~ no smoke.
  fetch_statics         committed statics bundle -> pipeline/
                        static_covariates.parquet on a 0.01-degree lattice
                        (DEM, TIGER road density, NEI point emissions
                        year-keyed, population-density proxy, NLCD
                        impervious attempt). Each sub-source is independent:
                        one failing leaves its columns absent, never filled.
  fetch_edgar_domain    EDGAR v8.1 (EU JRC) global 0.1-degree ANNUAL
                        sector-aggregated (TOTALS) emission gridmaps,
                        PM2.5 + NOx + SO2 for the latest published year,
                        bbox-subset to a domain-stamped STATIC parquet
                        [lat, lon, edgar_pm25, edgar_nox, edgar_so2]
                        (tonnes per cell per year). A NEW v4 feature
                        source registered under the 'edgar' registry key
                        for ALL domains, tx included; no shipped consumer
                        reads the key, so adding it changes no current
                        pipeline behavior. Pollutants are independent: a
                        failing one leaves its column absent (sidecar,
                        retried next call), never filled. OPTIONAL:
                        warn-and-continue.
  write_external_paths  the data-stage output registry frame2 consumes
                        (artifact external_paths.json); only keys whose
                        files exist are written.
  data_pa_decision      records the cf_1 refetch SKIP decision (BUILD_NOTES
                        scope decision 1) as an artifact so the `data-pa`
                        stage is a decision, not a fetch.

NaN honesty: NaN is the only missingness representation anywhere in these
outputs. A zero in st_nei_* means "no facility within the radius" (a fact),
never "we could not fetch NEI" (that leaves the column absent).

Run from anywhere:
    python fetchers2.py data [--quick] [--start YYYY-MM-DD --end YYYY-MM-DD]
    python fetchers2.py statics [--quick]
    python fetchers2.py data-pa
    python fetchers2.py all [--quick]
"""

import argparse
import glob
import io
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pandas as pd

import config2

# Banner glyphs are U+2500; Windows cp1252 consoles cannot encode them
# (v1 pipeline_colab precedent).
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

# ── Guarded heavy imports (v1 models_tabular style) ─────────────────────────
try:
    import shapefile as _pyshp          # pyshp
    HAS_PYSHP = True
except ImportError:
    _pyshp = None
    HAS_PYSHP = False
    print("[aqnet2] fetchers2: pyshp not installed -- HMS grid and TIGER "
          "roads unavailable (pip install pyshp)")

try:
    import rasterio as _rasterio
    HAS_RASTERIO = True
except ImportError:
    _rasterio = None
    HAS_RASTERIO = False
    print("[aqnet2] fetchers2: rasterio not installed -- NLCD impervious "
          "sub-source will be skipped (pip install rasterio)")

try:
    from matplotlib.path import Path as _MplPath
    HAS_MPL = True
except ImportError:
    _MplPath = None
    HAS_MPL = False
    print("[aqnet2] fetchers2: matplotlib not installed -- HMS polygon "
          "rasterization unavailable (pip install matplotlib)")

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    cKDTree = None
    HAS_SCIPY = False
    print("[aqnet2] fetchers2: scipy not installed -- NEI radius sums and "
          "population proxy unavailable (pip install scipy)")

# ── Constants ───────────────────────────────────────────────────────────────

QUICK_START, QUICK_END = "2024-07-01", "2024-09-30"   # v1 --quick window
QUICK_AQS_YEARS = [2024]
AQS_YEARS_FULL = list(range(2021, 2027))              # = v1 config.AQS_YEARS

AQS_SITES_URL = "https://aqs.epa.gov/aqsweb/airdata/aqs_sites.zip"

# GES DISC OPeNDAP per-day endpoint; stream numbers vary by production era
# (100/200/300/400) and reprocessing (401) — probed per day, cached per month.
SLV_URL = ("https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/"
           "M2T1NXSLV.5.12.4/{y}/{m:02d}/"
           "MERRA2_{s}.tavg1_2d_slv_Nx.{ymd}.nc4")
SLV_VARS = ["T2M", "QV2M", "U10M", "V10M", "PS"]
SLV_OUT_COLS = ["lat", "lon", "date", "merra2_t2m", "merra2_rh2m",
                "merra2_u10", "merra2_v10", "merra2_ps"]

HMS_BASE = ("https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/"
            "Smoke_Polygons/Shapefile/{y}/{m:02d}/hms_smoke{ymd}")
HMS_TIER = {"light": 1, "medium": 2, "heavy": 3}
HMS_TIER_NUM = {5.0: 1, 16.0: 2, 27.0: 3}   # legacy numeric Density codes

TIGER_ROADS_URL = ("https://www2.census.gov/geo/tiger/TIGER2023/PRISECROADS/"
                   "tl_2023_{fips}_prisecroads.zip")

ETOPO_30S_URL = ("https://www.ngdc.noaa.gov/thredds/fileServer/global/"
                 "ETOPO2022/30s/30s_surface_elev_netcdf/"
                 "ETOPO_2022_v1_30s_N90W180_surface.nc")
ETOPO_60S_URL = ("https://www.ngdc.noaa.gov/thredds/fileServer/global/"
                 "ETOPO2022/60s/60s_surface_elev_netcdf/"
                 "ETOPO_2022_v1_60s_N90W180_surface.nc")

NLCD_IMPERV_URLS = [
    "https://storage.googleapis.com/mrlc/Annual_NLCD_FctImp_2021_CU_C1V1.tif",
    "https://s3-us-west-2.amazonaws.com/mrlc/Annual_NLCD_FctImp_2021_CU_C1V1.tif",
    ("https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/"
     "Annual_NLCD_FctImp_2021_CU_C1V1.tif"),
]

NEI_YEARS = (2020, 2023)

# EPA region by state FIPS, for selecting members of the NEI by-regions
# fallback zip (the facility-level summary zips are national and need no
# selection). Covers every state a configured domain names; an unmapped
# FIPS falls back to parsing ALL regional members — a safe superset, since
# facilities are bbox-filtered afterwards.
EPA_REGION_BY_FIPS = {"04": "9", "06": "9", "08": "8", "32": "9",
                      "48": "6", "49": "8", "53": "10"}
STATICS_STEP = 0.01
STATICS_STEP_QUICK = 0.05
ROAD_SAMPLE_KM = 0.1          # polyline resample spacing (~100 m)
KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON_EQ = 111.320

# cf_1 policy constants recorded by data_pa_decision(). These mirror
# calibrate.py (CF1_RECON_UGM3, CHANNEL_RECON_VAR_FACTOR) — duplicated here
# so the decision artifact does not import calibrate's heavy chain.
CF1_RECON_UGM3 = 20.0
CHANNEL_RECON_VAR_FACTOR = 4.0

# EDGAR v8.1 air-pollutant annual gridmaps (EU JRC, public, no auth):
# https://edgar.jrc.ec.europa.eu/dataset_ap81. One zip per (pollutant,
# year) under the JRC open-data tree, ~16 MB each, holding a single global
# 0.1-degree netCDF (member v8.1_FT2022_AP_{poll}_{year}_TOTALS_emi.nc,
# variable `emissions`, units Tonnes per cell per year, cell centers
# -89.95..89.95 / -179.95..179.95). Pattern and content verified against
# the live directory listing and the PM2.5 2022 file on 2026-08-09. The
# release is frozen at FT2022 ("fast track" through 2022), so the latest
# year is a constant, not a probe: a newer year implies a new release id
# and therefore a new URL, never a new value of {year} here.
EDGAR_URL = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EDGAR/"
             "datasets/v81_FT2022_AP_new/{poll}/TOTALS/emi_nc/"
             "v8.1_FT2022_AP_{poll}_{year}_TOTALS_emi_nc.zip")
EDGAR_YEAR = 2022
EDGAR_POLLUTANTS = [("PM2.5", "edgar_pm25"), ("NOx", "edgar_nox"),
                    ("SO2", "edgar_so2")]


def _say(msg):
    print(f"[aqnet2] fetchers2: {msg}", flush=True)


def _banner(name):
    print(f"[aqnet2] ── stage: {name} " + "─" * max(0, 58 - len(name)),
          flush=True)


# ── v1 bridge (reuse, never reimplement) ────────────────────────────────────

_V1_DX = None


def _v1():
    """The v1 data_external module, imported lazily via config2.V1_DIR.

    v1 owns the battle-tested download/chunk/sidecar machinery and the
    GEOS-CF + MERRA-2 aerosol fetchers (DESIGN §14 "kept as-is"). Importing
    it also imports v1 config from the same directory — V1_DIR is inserted
    at sys.path position 0 so no other `config` module can shadow it."""
    global _V1_DX
    if _V1_DX is not None:
        return _V1_DX
    if config2.V1_DIR not in sys.path:
        sys.path.insert(0, config2.V1_DIR)
    import data_external
    _V1_DX = data_external
    return _V1_DX


# ── Shared helpers ──────────────────────────────────────────────────────────

def _atomic_parquet(df, dest):
    tmp = dest + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, dest)


def _atomic_json(obj, dest):
    tmp = dest + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, dest)


def _jsonable(obj):
    """Recursively convert numpy scalars and NaN to JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _dstem(base):
    """Domain-stamped filename stem for a bbox-dependent product.

    The tx domain keeps the shipped v2 names byte-for-byte (the frozen run's
    finals, chunk caches and registry stay valid and reproducible); every
    other domain appends its name. data/, cache/ and pipeline/ are SHARED
    across domains (only ARTIFACTS_DIR is namespaced), so an unstamped name
    would let a Texas-bbox file silently serve — or be served by — a wider
    domain. This is the aqs/statics window-stamp precedent (DESIGN §12.2)
    applied to space."""
    return base if config2.DOMAIN == "tx" else f"{base}_{config2.DOMAIN}"


def _dcache(name):
    """Domain-stamped CACHE_DIR subdir (created) for bbox-dependent chunks.

    Chunk parquets carry bbox-clipped content under bare {YYYYMM} names, so
    the DIRECTORY must be domain-stamped or a wider-domain reassembly would
    silently reuse Texas-bbox chunks. Raw downloads that are national/global
    (HMS day shapefiles, ETOPO, TIGER per-state zips, NEI summaries) stay in
    shared, unstamped dirs on purpose — their content does not depend on the
    bbox."""
    d = os.path.join(config2.CACHE_DIR, _dstem(name))
    os.makedirs(d, exist_ok=True)
    return d


def _probe_download(url, dest, attempts=3):
    """Download url to dest, distinguishing hard absence from transience.

    Returns (path_or_None, status) with status in {"ok", "absent", "error"}.
    Unlike v1 _download (which retries a 404 four times — correct for AirData
    zips that appear late, wasteful for HMS days that simply never existed),
    a 404/403 short-circuits as "absent" on the first attempt. Never leaves
    a partial file."""
    if os.path.exists(dest):
        return dest, "ok"
    tmp = dest + ".part"
    req = urllib.request.Request(
        url, headers={"User-Agent": "shared-skies-aqnet/2.0"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=180) as r, \
                    open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, dest)
            return dest, "ok"
        except urllib.error.HTTPError as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            if e.code in (403, 404, 410):
                return None, "absent"
            print(f"    download attempt {attempt + 1}/{attempts} "
                  f"failed: HTTP {e.code}")
            time.sleep(10 * (attempt + 1))
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            print(f"    download attempt {attempt + 1}/{attempts} failed: {e}")
            time.sleep(10 * (attempt + 1))
    return None, "error"


def _ensure_dodsrc():
    """Create ~/.dodsrc pointing the netCDF4 DAP client at ~/.netrc.

    GES DISC OPeNDAP requires Earthdata auth; the netCDF4 C library only
    honors ~/.netrc when a ~/.dodsrc names it (HTTP.NETRC) together with a
    cookie jar (HTTP.COOKIEJAR) for the URS redirect dance. ~/.netrc already
    exists on both machines (audit 06 §5); this creates the missing half."""
    home = os.path.expanduser("~")
    rc = os.path.join(home, ".dodsrc")
    if not os.path.exists(rc):
        with open(rc, "w", encoding="utf-8") as fh:
            fh.write(f"HTTP.NETRC={os.path.join(home, '.netrc')}\n")
            fh.write(f"HTTP.COOKIEJAR={os.path.join(home, '.urs_cookies')}\n")
        _say(f"created {rc} (HTTP.NETRC + HTTP.COOKIEJAR for OPeNDAP auth)")
    return rc


def _grid_axes(step, decimals=5):
    """Ascending (lat_axis, lon_axis) of cell centers covering TX_BBOX.

    Centers land on multiples of `step` starting at the bbox minima; at
    step=0.1 this reproduces the committed met_extra_by_cell cell centers
    (26.0, -100.0, ... — audit 06 §4) so frame2's nearest-cell join is
    exact, not approximate."""
    bb = config2.TX_BBOX
    lat = np.round(np.arange(bb["lat_min"], bb["lat_max"] + step / 2, step),
                   decimals)
    lon = np.round(np.arange(bb["lon_min"], bb["lon_max"] + step / 2, step),
                   decimals)
    return lat, lon


def _box_sum(a, hl, hw):
    """Edge-truncated box sum of 2-D array a over (2*hl+1) x (2*hw+1) cells."""
    h, w = a.shape
    p = np.zeros((h + 1, w + 1), dtype=np.float64)
    p[1:, 1:] = np.cumsum(np.cumsum(a, axis=0), axis=1)
    r0 = np.clip(np.arange(h) - hl, 0, h)
    r1 = np.clip(np.arange(h) + hl + 1, 0, h)
    c0 = np.clip(np.arange(w) - hw, 0, w)
    c1 = np.clip(np.arange(w) + hw + 1, 0, w)
    return (p[np.ix_(r1, c1)] - p[np.ix_(r0, c1)]
            - p[np.ix_(r1, c0)] + p[np.ix_(r0, c0)])


def _bilinear(axis_lat, axis_lon, z, qlat, qlon):
    """Bilinear sample of regular grid z[lat, lon] at query points (NaN off
    the grid; NaN corners propagate — no fill)."""
    fi = np.interp(qlat, axis_lat, np.arange(len(axis_lat)),
                   left=np.nan, right=np.nan)
    fj = np.interp(qlon, axis_lon, np.arange(len(axis_lon)),
                   left=np.nan, right=np.nan)
    out = np.full(qlat.shape, np.nan)
    ok = np.isfinite(fi) & np.isfinite(fj)
    i0 = np.floor(fi[ok]).astype(np.int64)
    j0 = np.floor(fj[ok]).astype(np.int64)
    i0 = np.clip(i0, 0, len(axis_lat) - 2)
    j0 = np.clip(j0, 0, len(axis_lon) - 2)
    di = fi[ok] - i0
    dj = fj[ok] - j0
    z = np.asarray(z, dtype=np.float64)
    out[ok] = (z[i0, j0] * (1 - di) * (1 - dj)
               + z[i0 + 1, j0] * di * (1 - dj)
               + z[i0, j0 + 1] * (1 - di) * dj
               + z[i0 + 1, j0 + 1] * di * dj)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 1. AQS hardened refetch (REQUIRED)
# ═════════════════════════════════════════════════════════════════════════════

_AQS_USECOLS_V2 = ["State Code", "County Code", "Site Num", "Parameter Code",
                   "POC", "Latitude", "Longitude", "Date Local",
                   "Sample Duration", "Arithmetic Mean", "Event Type",
                   "Observation Percent", "Method Name"]
_AQS_DAILY_DURATIONS = {"24-HR BLK AVG", "24 HOUR"}
_AQS_MIN_OBS_PCT = 75
_DUR_RANK = {"24-HR BLK AVG": 0, "1 HOUR": 1}
_URBAN_SETTING = "URBAN AND CENTER CITY"


def _aqs_site_meta(zip_dir, dx):
    """Domain-state rows of the AQS site listing -> [site_id,
    location_setting].

    aqs_sites.zip is the station-metadata companion of the AirData daily
    files; Location Setting is the EPA's own urban classification. States
    come from config2.STATE_FIPS (tx: 48 only, unchanged). Returns None
    (with a message) when the listing cannot be fetched — is_fem and pm25
    survive, urban/location_setting become NaN."""
    zp = dx._download(AQS_SITES_URL, os.path.join(zip_dir, "aqs_sites.zip"))
    if zp is None:
        _say("aqs_sites.zip unavailable -- urban/location_setting will be NaN")
        return None
    want = {"State Code", "County Code", "Site Number", "Location Setting"}
    s = pd.read_csv(zp, usecols=lambda c: c in want, dtype=str,
                    low_memory=False)
    s = s[s["State Code"].isin(config2.STATE_FIPS)].copy()
    s["site_id"] = (s["State Code"].str.zfill(2)
                    + s["County Code"].str.zfill(3)
                    + s["Site Number"].str.zfill(4))
    s["location_setting"] = s["Location Setting"].str.strip().str.upper()
    s = s.drop_duplicates("site_id")[["site_id", "location_setting"]]
    s = s.reset_index(drop=True)
    _say(f"aqs_sites listing: {len(s)} sites in "
         f"{len(config2.STATE_FIPS)} domain state(s)")
    return s


def _aqs_is_fem(method_name, dur_rank):
    """FEM indicator: Method Name string first, sample duration as fallback.

    Rule 1 (explicit): a Method Name containing "FEM" -> 1, "FRM" -> 0.
    Rule 2 (fallback, printed): AirData method names often omit the literal
    designation (e.g. "R & P Model 2025 ... - Gravimetric" is an FRM with no
    'FRM' substring), so undetermined rows fall back to the duration
    heuristic v1's fetch docstring documents: continuous durations
    (24-HR BLK AVG, 1 HOUR — dur_rank 0/1) report every day from hourly
    data and are FEMs; everything else (the filter "24 HOUR" schedule) is
    FRM. Every row therefore gets 0/1 — no NaN, because the fallback is a
    total function of duration, not a fill."""
    name = method_name.fillna("").astype(str).str.upper()
    out = np.full(len(name), np.nan)
    out[name.str.contains("FEM", regex=False).to_numpy()] = 1.0
    out[name.str.contains("FRM", regex=False).to_numpy()] = 0.0
    undet = ~np.isfinite(out)
    if undet.any():
        _say(f"aqs: {int(undet.sum()):,} rows lack FRM/FEM in Method Name "
             "-- classified by sample duration (continuous=FEM)")
        out[undet] = (dur_rank.to_numpy()[undet] <= 1).astype(np.float64)
    return out


def fetch_aqs_v2(years=None, dest=None):
    """Hardened EPA AQS daily PM2.5 for the domain states -> parquet path.

    Reuses the v1 AirData zip cache (cache/aqs under V1_DIR) and the v1
    reduction: keep the config2.STATE_FIPS State Codes / Parameter 88101,
    average rows sharing (site, date, POC, duration) — AirData repeats them
    once per pollutant standard — then the preferred duration (24-HR BLK
    AVG, else 1 HOUR, else other) and lowest POC win per (site, date).
    Quality filters are identical to v1: Event Type "Excluded" rows
    dropped, sub-daily durations need Observation Percent >= 75.

    Multistate note: the AirData zips are NATIONAL files (one download per
    year, shared across domains in the v1 zip cache), so "loop the states"
    is an isin() filter over State Code, not per-state downloads — v1
    fetch_aqs_daily_tx takes no state argument and hardcodes 48 internally,
    which is why the reduction lives here. site_id embeds the state FIPS,
    so the (site, date) dedupe below already covers the multistate concat.

    v2 hardening over v1 fetch_aqs_daily_tx (audit 06 §4, item f):
      * POC, Method Name and Sample Duration are RETAINED through the
        reduction so is_fem is derivable (v1 discarded them);
      * the aqs_sites.zip listing joins location_setting -> urban;
      * dest embeds the domain stem AND the year window
        (config2.AQS_STEM + _{y0}_{y1}.parquet under config2.DATA_DIR;
        tx: aqs_daily_tx_v2_..., byte-identical to the shipped run),
        closing the v1 quick/full cache-poisoning hazard (DESIGN §12.2);
      * failed years land in a {dest}.failed.json sidecar and are retried
        on the next call instead of silently truncating the record.

    Output columns: [site_id (str), date (datetime64[ns]), pm25_aqs, lat,
    lon, poc (int), is_fem (0/1), urban (1/0/NaN), location_setting].
    urban is 1.0 for "URBAN AND CENTER CITY", 0.0 for SUBURBAN/RURAL, NaN
    when the site is absent from the listing. REQUIRED source: raises
    RuntimeError when no year could be fetched."""
    dx = _v1()
    years = sorted(list(years) if years is not None else AQS_YEARS_FULL)
    dest = dest or os.path.join(
        config2.DATA_DIR,
        f"{config2.AQS_STEM}_{years[0]}_{years[-1]}.parquet")

    prev_failed = dx._read_failed_months(dest)
    if os.path.exists(dest) and not prev_failed:
        _say(f"aqs: using cached {dest}")
        return dest
    if prev_failed:
        _say(f"aqs: cached {dest} is missing year(s) {prev_failed} -- "
             "retrying them first")

    zip_dir = os.path.join(dx.config.CACHE_DIR, "aqs")   # shared v1 zip cache
    os.makedirs(zip_dir, exist_ok=True)
    sites = _aqs_site_meta(zip_dir, dx)

    order = ([y for y in years if str(y) in set(prev_failed)]
             + [y for y in years if str(y) not in set(prev_failed)])
    frames, failed = [], []
    for y in order:
        zp = dx._download(dx.AQS_URL.format(year=y),
                          os.path.join(zip_dir, f"daily_88101_{y}.zip"))
        if zp is None:
            _say(f"aqs {y}: download failed (year may not be published yet)")
            failed.append(str(y))
            continue
        # low_memory=False avoids the pandas C-parser defect v1 documents
        # (chunked dtype inference raising IndexError on the 2024 file).
        d = pd.read_csv(zp, usecols=_AQS_USECOLS_V2, low_memory=False,
                        dtype={"State Code": str, "County Code": str,
                               "Site Num": str})
        d = d[d["State Code"].isin(config2.STATE_FIPS)
              & (d["Parameter Code"] == 88101)]
        if len(d):
            frames.append(d)
        n_by = d["State Code"].value_counts()
        _say(f"aqs {y}: {len(d):,} rows ("
             + ", ".join(f"{s}={int(n_by.get(s, 0)):,}"
                         for s in config2.STATE_FIPS) + ")")
    if not frames:
        raise RuntimeError("aqs: no data retrieved for any requested year "
                           "(REQUIRED source -- data stage cannot proceed).")

    d = pd.concat(frames, ignore_index=True)
    n0 = len(d)
    d = d[d["Event Type"] != "Excluded"]
    n_event = n0 - len(d)
    incomplete = (~d["Sample Duration"].isin(_AQS_DAILY_DURATIONS)
                  & ~(d["Observation Percent"] >= _AQS_MIN_OBS_PCT))
    d = d[~incomplete].copy()
    _say(f"aqs quality: dropped {n_event:,} Event Type 'Excluded' rows and "
         f"{int(incomplete.sum()):,} sub-daily rows with Observation "
         f"Percent < {_AQS_MIN_OBS_PCT}")
    if not len(d):
        raise RuntimeError("aqs: no rows survived the quality filters.")

    d["site_id"] = (d["State Code"].str.zfill(2)
                    + d["County Code"].str.zfill(3)
                    + d["Site Num"].str.zfill(4))
    # AQS dates arrive as strings; normalize explicitly to datetime64[ns]
    # (the committed v1 parquet is datetime64[us] — audit 06 §4 item h).
    d["date"] = (pd.to_datetime(d["Date Local"]).dt.normalize()
                 .astype("datetime64[ns]"))
    d["dur_rank"] = d["Sample Duration"].map(_DUR_RANK).fillna(2).astype(int)

    g = (d.groupby(["site_id", "date", "POC", "dur_rank"], as_index=False)
          .agg(pm25_aqs=("Arithmetic Mean", "mean"),
               lat=("Latitude", "first"), lon=("Longitude", "first"),
               method_name=("Method Name", "first")))
    g = (g.sort_values(["site_id", "date", "dur_rank", "POC"])
          .drop_duplicates(["site_id", "date"], keep="first")
          .reset_index(drop=True))
    g["poc"] = g["POC"].astype("int64")
    g["is_fem"] = _aqs_is_fem(g["method_name"], g["dur_rank"])
    _say(f"aqs: is_fem mean {float(np.nanmean(g['is_fem'])):.3f} over "
         f"{len(g):,} site-days")

    g["site_id"] = g["site_id"].astype(str)
    if sites is not None:
        sites = sites.copy()
        sites["site_id"] = sites["site_id"].astype(str)
        g = g.merge(sites, on="site_id", how="left")
    else:
        g["location_setting"] = pd.Series([pd.NA] * len(g), dtype="object")
    loc = g["location_setting"].astype("string").str.upper()
    urban = np.full(len(g), np.nan)
    urban[(loc == _URBAN_SETTING).fillna(False).to_numpy()] = 1.0
    urban[loc.isin(["SUBURBAN", "RURAL"]).fillna(False).to_numpy()] = 0.0
    g["urban"] = urban
    n_unk = int((~np.isfinite(urban)).sum())
    if n_unk:
        _say(f"aqs: {n_unk:,} site-days have no location_setting -- "
             "urban stays NaN (never filled)")

    out = g[["site_id", "date", "pm25_aqs", "lat", "lon", "poc", "is_fem",
             "urban", "location_setting"]].reset_index(drop=True)
    _atomic_parquet(out, dest)
    dx._write_failed_months(dest, failed)
    _say(f"aqs: saved {dest}: {len(out):,} site-days, "
         f"{out['site_id'].nunique()} sites, "
         f"{out['date'].min().date()} .. {out['date'].max().date()}")
    return dest


# ═════════════════════════════════════════════════════════════════════════════
# 2. MERRA-2 SLV met via GES DISC OPeNDAP subsetting (OPTIONAL -> flag)
# ═════════════════════════════════════════════════════════════════════════════

def _slv_streams(ts):
    """Candidate MERRA-2 stream numbers for a date, most likely first.

    Production streams: 100 (1980-91), 200 (1992-2000), 300 (2001-10),
    400 (2011+); 401 covers reprocessed spans (e.g. mid-2021). The order is
    a probe list, not an assumption — every day tries the survivors."""
    y = pd.Timestamp(ts).year
    if y >= 2011:
        return ["400", "401", "300", "200", "100"]
    if y >= 2001:
        return ["300", "400", "401", "200", "100"]
    if y >= 1992:
        return ["200", "300", "100", "400", "401"]
    return ["100", "200", "300", "400", "401"]


def _open_dap(url):
    """Open an OPeNDAP URL via xarray: netCDF4 engine first, pydap fallback
    (the v1 _open_geoscf pattern; pydap is absent on the cluster, so its
    failure message names the real fix)."""
    import xarray as xr
    try:
        return xr.open_dataset(url, engine="netcdf4")
    except Exception as e_nc:
        try:
            return xr.open_dataset(
                url.replace("https://", "dap2://").replace("http://",
                                                           "dap2://"),
                engine="pydap")
        except Exception as e_dap:
            raise RuntimeError(
                f"netcdf4 engine failed ({e_nc}); pydap fallback failed "
                f"({e_dap})")


def _slv_day(ts, stream_hint):
    """One day of SLV daily means over the TX bbox -> (DataFrame, stream).

    Only the bbox subset crosses the wire: xarray's DAP backend translates
    .sel + .load into constrained requests, so a day costs a few hundred KB,
    not a 400 MB granule (BUILD_NOTES scope decision 2)."""
    bb = config2.TX_BBOX
    streams = list(dict.fromkeys([stream_hint] + _slv_streams(ts))) \
        if stream_hint else _slv_streams(ts)
    last_err = None
    for s in streams:
        url = SLV_URL.format(y=ts.year, m=ts.month, s=s,
                             ymd=ts.strftime("%Y%m%d"))
        try:
            ds = _open_dap(url)
        except Exception as e:
            last_err = e
            continue
        try:
            sub = ds[SLV_VARS].sel(
                lat=slice(bb["lat_min"], bb["lat_max"]),
                lon=slice(bb["lon_min"], bb["lon_max"]))
            day = sub.mean("time", skipna=True).load()
        finally:
            ds.close()
        df = day.to_dataframe().reset_index()
        df["date"] = pd.Timestamp(ts).normalize()
        # Daily-mean RH from daily-mean T2M/QV2M/PS (Magnus over the mixing
        # ratio; an approximation of mean(hourly RH) — documented, adequate
        # for a met covariate).
        t_c = df["T2M"].to_numpy(dtype=np.float64) - 273.15
        q = df["QV2M"].to_numpy(dtype=np.float64)
        ps = df["PS"].to_numpy(dtype=np.float64)
        es = 610.94 * np.exp(17.625 * t_c / (t_c + 243.04))     # Pa
        w = q / (1.0 - q)
        ws = 0.622 * es / (ps - es)
        rh = np.clip(100.0 * w / ws, 1.0, 100.0)                # NaN passes
        out = pd.DataFrame({
            "lat": df["lat"].to_numpy(dtype=np.float64),
            "lon": df["lon"].to_numpy(dtype=np.float64),
            "date": df["date"],
            "merra2_t2m": t_c,
            "merra2_rh2m": rh,
            "merra2_u10": df["U10M"].to_numpy(dtype=np.float64),
            "merra2_v10": df["V10M"].to_numpy(dtype=np.float64),
            "merra2_ps": ps,                                     # Pa
        })
        keep = np.isfinite(
            out[["merra2_t2m", "merra2_u10", "merra2_v10",
                 "merra2_ps"]].to_numpy()).any(axis=1)
        return out[keep].reset_index(drop=True), s
    raise RuntimeError(f"all streams failed for {ts.date()}: {last_err}")


def _slv_month(lo, hi):
    """One calendar month of SLV daily means, or raise (any failing day
    fails the month — a partial month must never be baked into a chunk)."""
    days = pd.date_range(lo.normalize(), hi.normalize(), freq="D")
    frames, hint = [], None
    for ts in days:
        df = None
        last = None
        for attempt in range(2):
            try:
                df, hint = _slv_day(ts, hint)
                break
            except Exception as e:
                last = e
                time.sleep(10 * (attempt + 1))
        if df is None:
            raise RuntimeError(f"SLV day {ts.date()} failed: {last}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def fetch_merra2_slv(start, end, dest=None):
    """MERRA-2 M2T1NXSLV daily met over the TX bbox -> parquet path or None.

    T2M/QV2M/U10M/V10M/PS hourly fields, bbox-subset per day through GES
    DISC OPeNDAP (never full granules), reduced to daily means; merra2_rh2m
    derived from QV2M/T2M/PS (Magnus/mixing-ratio, clipped 1..100).
    Month-chunked under cache/merra2_slv with the v1 sidecar contract; the
    final dest embeds the window. Auth: ~/.netrc plus ~/.dodsrc (created
    here when missing).

    Output columns: [lat, lon, date, merra2_t2m (degC), merra2_rh2m (%),
    merra2_u10, merra2_v10 (m/s), merra2_ps (Pa)] on the native 0.5 x 0.625
    grid. Both the final (merra2_slv_{domain}_{window}) and the chunk dir
    are domain-stamped — the content is bbox-clipped, and data/ + cache/
    are shared across domains (tx keeps merra2_slv_tx_*, byte-identical).
    OPTIONAL source: total failure returns None (downstream features
    stay NaN); failed months land in the sidecar and are retried first on
    the next call."""
    dx = _v1()
    dest = dest or os.path.join(
        config2.DATA_DIR,
        f"merra2_slv_{config2.DOMAIN}_{dx._window_tag(start, end)}.parquet")
    prev_failed = []
    if os.path.exists(dest):
        prev_failed = dx._read_failed_months(dest)
        if prev_failed:
            _say(f"merra2-slv: cached {dest} is missing month(s) "
                 f"{prev_failed} -- retrying them first")
        elif dx._covers_window(dest, start, end):
            _say(f"merra2-slv: using cached {dest}")
            return dest
        else:
            _say(f"merra2-slv: cached {dest} does not cover {start}..{end} "
                 "-- reassembling from month chunks")
    _ensure_dodsrc()
    chunk_dir = _dcache("merra2_slv")   # bbox-clipped chunks: domain-stamped

    frames, failed = [], []
    edges = dx._retry_failed_first(dx._month_edges(start, end), prev_failed)
    for m, (lo, hi) in enumerate(edges):
        tag = lo.strftime("%Y%m")
        cp = os.path.join(chunk_dir, f"merra2slv_{tag}.parquet")
        if os.path.exists(cp):
            frames.append(pd.read_parquet(cp))
            continue
        df = None
        try:
            df = _slv_month(lo, hi)
        except Exception as e:
            print(f"  merra2-slv {tag}: {e}")
        if df is None:
            failed.append(tag)
            continue
        _atomic_parquet(df, cp)   # atomic: a preempted chunk must never be trusted
        frames.append(df)
        _say(f"merra2-slv {tag}: {len(df):,} cell-days "
             f"({m + 1}/{len(edges)} months)")

    if failed:
        _say(f"merra2-slv: {len(failed)} month(s) failed and were skipped: "
             f"{failed} (recorded in the sidecar; retried first next call)")
    if not frames:
        _say("merra2-slv: no months could be fetched -- continuing without "
             "SLV met (features stay NaN)")
        return None

    out = pd.concat(frames, ignore_index=True)
    out = out[(out["date"] >= pd.Timestamp(start))
              & (out["date"] <= pd.Timestamp(end))]
    out = out.sort_values(["date", "lat", "lon"]).reset_index(drop=True)
    _atomic_parquet(out, dest)
    dx._write_failed_months(dest, failed)
    _say(f"merra2-slv: saved {dest}: {len(out):,} cell-days, "
         f"{out['date'].min().date()} .. {out['date'].max().date()}")
    return dest


def fetch_merra2_combined(start, end, dest=None):
    """v1 aerosol MERRA-2 + v2 SLV met, outer-merged -> parquet path or None.

    The v1 fetch_merra2 call reassembles its parquet from the 909G granule
    cache when the month chunks or granules exist (earthaccess skips
    already-downloaded files) — no re-download of the archive. Both
    collections live on the native 0.5 x 0.625 grid, so the outer merge on
    (lat, lon, date) is exact (coords rounded to 5 decimals to kill float
    representation jitter); rows present on one side only carry NaN for the
    other side's columns — frame2 joins nearest-cell, so mixed availability
    degrades honestly per column. OPTIONAL: returns None when neither side
    is available.

    Domain note: the v1 aerosol parquet is Texas-bbox by construction (v1
    config.TX_BBOX is frozen — widening it would poison v1's own
    merra2_daily_tx_* namespace), so under a non-tx domain the aerosol
    columns are honest NaN outside Texas AND the SLV side is REQUIRED:
    frame2's nearest-cell join has no distance cap, so an aerosol-only
    combined under a wider domain would silently smear Texas cells across
    the whole domain. The final is domain-stamped for the same reason."""
    dx = _v1()
    dest = dest or os.path.join(
        config2.DATA_DIR,
        f"{_dstem('merra2_combined')}_{dx._window_tag(start, end)}.parquet")
    if os.path.exists(dest) and dx._covers_window(dest, start, end) \
            and not dx._read_failed_months(dest) \
            and os.environ.get("FORCE") != "1":
        # FORCE=1 rebuilds (review finding: SLV repairs could otherwise
        # never propagate into a once-written combined parquet).
        _say(f"merra2-combined: using cached {dest}")
        return dest

    aer_path = None
    try:
        aer_path = dx.fetch_merra2(start, end)
    except Exception as e:
        _say(f"merra2-combined: v1 aerosol fetch raised ({e}) -- "
             "continuing with SLV only")
    slv_path = fetch_merra2_slv(start, end)
    if config2.DOMAIN != "tx" and slv_path is None:
        _say("merra2-combined: SLV side unavailable and the v1 aerosol "
             "parquet is Texas-bbox -- refusing an aerosol-only combined "
             f"for domain {config2.DOMAIN!r} (nearest-cell joins would "
             "smear Texas cells across the domain); skipped")
        return None

    parts = []
    for name, path in (("aerosol", aer_path), ("slv", slv_path)):
        if path is None or not os.path.exists(path):
            _say(f"merra2-combined: {name} side unavailable")
            continue
        df = pd.read_parquet(path)
        df["lat"] = np.round(df["lat"].astype(np.float64), 5)
        df["lon"] = np.round(df["lon"].astype(np.float64), 5)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize() \
            .astype("datetime64[ns]")
        parts.append(df)
    if not parts:
        _say("merra2-combined: neither aerosol nor SLV available -- skipped")
        return None
    out = parts[0]
    if len(parts) == 2:
        out = out.merge(parts[1], on=["lat", "lon", "date"], how="outer")
    out = out.sort_values(["date", "lat", "lon"]).reset_index(drop=True)
    _atomic_parquet(out, dest)
    _say(f"merra2-combined: saved {dest}: {len(out):,} cell-days, "
         f"{len(out.columns) - 3} value columns")
    return dest


# ═════════════════════════════════════════════════════════════════════════════
# 2b. GEOS-CF surface PM2.5 over the domain bbox (OPTIONAL, non-tx)
# ═════════════════════════════════════════════════════════════════════════════

def _geoscf_domain_bbox(da):
    """Clip a GEOS-CF DataArray to config2.TX_BBOX (THE DOMAIN bbox),
    tolerating either latitude order.

    Port of v1 _bbox_slice, which is hardwired to v1 config.TX_BBOX; that
    box is frozen (widening it would poison v1's own geoscf_pm25_* cache
    namespace), so the domain clip must resolve from config2 here. Same
    either-order tolerance as v1: the GrADS trees serve ascending latitude
    today, but against a descending axis an ascending slice silently yields
    an empty subset, hence the size-0 re-probe with reversed bounds."""
    bb = config2.TX_BBOX
    sub = da.sel(lat=slice(bb["lat_min"], bb["lat_max"]),
                 lon=slice(bb["lon_min"], bb["lon_max"]))
    if sub.sizes.get("lat", 0) == 0:      # descending latitude axis
        sub = da.sel(lat=slice(bb["lat_max"], bb["lat_min"]),
                     lon=slice(bb["lon_min"], bb["lon_max"]))
    return sub


def _geoscf_domain_month(lo, hi):
    """One month of GEOS-CF daily-mean surface PM2.5 over the domain, or
    raise (a partial month must never be baked into a chunk).

    Everything except the clip is v1's committed 'union GEOS-CF fix',
    reused via the bridge rather than reimplemented:

      * dx._geoscf_url routes the month to the assim v1 chm_tavg_1hr tree
        before config.GEOSCF_V2_START and to the v2 ana tree after it (the
        v1 collection stops at 2026-01-02 and the v2 collection is a
        rolling ~1-year window, so neither covers the full window alone);
      * dx._open_geoscf prefers the netCDF4 engine but VALIDATES its time
        axis (netCDF4 can open the GrADS server and still mis-decode the
        "days since 1-1-1" axis onto a garbage scale) and falls back to
        pydap pinned to dap2://, rebuilding the axis by hand from GrADS
        ordinals;
      * dx._geoscf_var probes pm25_rh35_gcc / pm25_rh35 and friends, since
        the two trees publish disjoint variable names.

    Hourly fields are clipped to the domain bbox, sliced to the month, and
    reduced to daily means on the native 0.25-degree grid; only the subset
    crosses the wire. Rows whose value is non-finite are dropped (loud
    omission: an absent cell-day stays absent, never filled)."""
    dx = _v1()
    ds = dx._open_geoscf(dx._geoscf_url(lo))
    try:
        da = ds[dx._geoscf_var(ds)]
        if "lev" in da.dims:
            da = da.isel(lev=0)
        da = _geoscf_domain_bbox(da).sel(time=slice(lo, hi))
        daily = da.resample(time="1D").mean(skipna=True).load()
    finally:
        ds.close()
    df = daily.rename("geoscf_pm25").to_dataframe().reset_index()
    df = df.rename(columns={"time": "date"})
    df["date"] = (pd.to_datetime(df["date"]).dt.normalize()
                  .astype("datetime64[ns]"))
    df = df[np.isfinite(df["geoscf_pm25"])]
    return df[["lat", "lon", "date", "geoscf_pm25"]].reset_index(drop=True)


def fetch_geoscf_domain(start, end, dest=None):
    """GEOS-CF surface PM2.5 daily means over the domain bbox -> path or
    None.

    The domain-wide sibling of v1 fetch_geoscf_pm25 (which stays Texas-only
    and untouched; tx runs keep registering its committed parquet from
    V1_DIR/data). Month-chunked under the domain-stamped _dcache("geoscf")
    dir with bare geoscf_{YYYYMM}.parquet chunk names (the merra2_slv
    precedent: bbox-clipped content in a shared cache/ tree must live in a
    stamped DIRECTORY, or a wider-domain reassembly would silently reuse
    Texas-bbox chunks); an existing chunk file is trusted and skipped, so
    an interrupted pull resumes where it stopped. Each missing month gets
    3 attempts with backoff, v1 style; a failing month lands in the
    {dest}.failed.json sidecar and is retried first on the next call, never
    baked in silently. All writes are atomic (tmp + os.replace).

    The final is DATA_DIR/{_dstem('geoscf_pm25')}_{window}.parquet (west7:
    geoscf_pm25_west7_YYYYMMDD_YYYYMMDD.parquet), exactly the pattern
    write_external_paths already globs for the domain's 'geoscf' key; an
    existing final is trusted only when its sidecar is clean and its dates
    cover the window, otherwise it is reassembled from the month chunks,
    which are the cache of record. Auth is not needed for the public GrADS
    server; ~/.netrc and ~/.dodsrc on the cluster are harmless to it.

    Output columns: [lat, lon, date, geoscf_pm25] (ug/m3) on the native
    0.25-degree grid. OPTIONAL source: when no month can be fetched this
    returns None instead of raising (v1 raises; here downstream geoscf_*
    features stay NaN and run_data continues)."""
    dx = _v1()
    dest = dest or os.path.join(
        config2.DATA_DIR,
        f"{_dstem('geoscf_pm25')}_{dx._window_tag(start, end)}.parquet")
    prev_failed = []
    if os.path.exists(dest):
        prev_failed = dx._read_failed_months(dest)
        if prev_failed:
            _say(f"geoscf: cached {dest} is missing month(s) {prev_failed} "
                 "-- retrying them first")
        elif dx._covers_window(dest, start, end):
            _say(f"geoscf: using cached {dest}")
            return dest
        else:
            _say(f"geoscf: cached {dest} does not cover {start}..{end} -- "
                 "reassembling from month chunks")
    chunk_dir = _dcache("geoscf")   # bbox-clipped chunks: domain-stamped

    frames, failed = [], []
    edges = dx._retry_failed_first(dx._month_edges(start, end), prev_failed)
    for m, (lo, hi) in enumerate(edges):
        tag = lo.strftime("%Y%m")
        cp = os.path.join(chunk_dir, f"geoscf_{tag}.parquet")
        if os.path.exists(cp):
            frames.append(pd.read_parquet(cp))
            continue
        df = None
        for attempt in range(3):
            try:
                df = _geoscf_domain_month(lo, hi)
                break
            except Exception as e:
                print(f"  geoscf {tag} attempt {attempt + 1}/3: {e}")
                time.sleep(15 * (attempt + 1))
        if df is None:
            failed.append(tag)
            continue
        _atomic_parquet(df, cp)   # atomic: a preempted chunk must never be trusted
        frames.append(df)
        _say(f"geoscf {tag}: {len(df):,} cell-days "
             f"({m + 1}/{len(edges)} months)")

    if failed:
        _say(f"geoscf: {len(failed)} month(s) failed and were skipped: "
             f"{failed} (recorded in the sidecar; retried first next call)")
    if not frames:
        _say("geoscf: no months could be fetched -- continuing without "
             "GEOS-CF (geoscf_pm25 stays NaN)")
        return None

    out = pd.concat(frames, ignore_index=True)
    out = out[(out["date"] >= pd.Timestamp(start))
              & (out["date"] <= pd.Timestamp(end))]
    if not len(out):
        _say("geoscf: assembled chunks hold no rows inside the window -- "
             "no final written (geoscf_pm25 stays NaN)")
        return None
    out = out.sort_values(["date", "lat", "lon"]).reset_index(drop=True)
    _atomic_parquet(out, dest)
    dx._write_failed_months(dest, failed)
    _say(f"geoscf: saved {dest}: {len(out):,} cell-days, "
         f"{out['date'].min().date()} .. {out['date'].max().date()}")
    return dest


# ═════════════════════════════════════════════════════════════════════════════
# 3. HMS smoke polygons -> 0.1-degree cell raster (OPTIONAL)
# ═════════════════════════════════════════════════════════════════════════════

def _hms_empty():
    return pd.DataFrame({"cell_lat": pd.Series(dtype="float64"),
                         "cell_lon": pd.Series(dtype="float64"),
                         "date": pd.Series(dtype="datetime64[ns]"),
                         "hms_smoke": pd.Series(dtype="int8")})


def _hms_tier(value):
    """Density attribute -> tier (Light 1 / Medium 2 / Heavy 3, else 1).

    Newer shapefiles carry the words, older ones numeric codes (5/16/27
    ug/m3 nominal); anything unrecognized is conservatively Light."""
    s = str(value).strip().lower()
    if s in HMS_TIER:
        return HMS_TIER[s]
    try:
        return HMS_TIER_NUM.get(float(s), 1)
    except (TypeError, ValueError):
        return 1


def _hms_reader_for_day(ts, raw_dir):
    """pyshp Reader for one day, or None when the day is unmapped (404).

    Tries hms_smoke{ymd}.zip first, then the loose .shp/.dbf/.shx triplet.
    Raises on transient (non-404) failure so the month is marked failed
    rather than silently recorded as smoke-free."""
    ymd = ts.strftime("%Y%m%d")
    base = HMS_BASE.format(y=ts.year, m=ts.month, ymd=ymd)
    zdest = os.path.join(raw_dir, f"hms_smoke{ymd}.zip")
    path, status = _probe_download(base + ".zip", zdest)
    if status == "ok":
        with zipfile.ZipFile(path) as zf:
            members = {os.path.splitext(n)[1].lower(): n
                       for n in zf.namelist()
                       if os.path.splitext(n)[1].lower() in
                       (".shp", ".dbf", ".shx")}
            if ".shp" not in members or ".dbf" not in members:
                raise RuntimeError(f"hms zip for {ymd} lacks shp/dbf members")
            return _pyshp.Reader(
                shp=io.BytesIO(zf.read(members[".shp"])),
                dbf=io.BytesIO(zf.read(members[".dbf"])),
                shx=(io.BytesIO(zf.read(members[".shx"]))
                     if ".shx" in members else None))
    if status == "error":
        raise RuntimeError(f"hms zip download failed for {ymd}")
    # 404 on the zip: probe the loose triplet before declaring the day absent
    parts = {}
    for ext in (".shp", ".dbf", ".shx"):
        p, st = _probe_download(base + ext,
                                os.path.join(raw_dir, f"hms_smoke{ymd}{ext}"))
        if st == "error":
            raise RuntimeError(f"hms {ext} download failed for {ymd}")
        if st == "ok":
            parts[ext] = p
    if ".shp" not in parts or ".dbf" not in parts:
        return None            # genuinely unmapped day (~ no smoke, documented)
    return _pyshp.Reader(shp=open(parts[".shp"], "rb"),
                         dbf=open(parts[".dbf"], "rb"),
                         shx=(open(parts[".shx"], "rb")
                              if ".shx" in parts else None))


def _hms_day_rows(ts, reader, pts):
    """Rasterize one day's polygons -> smoke-positive cell rows.

    Containment is matplotlib.path.contains_points over the 0.1-degree cell
    CENTERS (a center-in-polygon raster, the standard mask convention at
    this resolution); multi-ring shapes are OR-ed per ring, so interior
    holes count as covered — smoke plumes essentially never have holes and
    over-counting a hole is the conservative direction for an exposure
    tier. Per-cell tier is the MAX over overlapping polygons."""
    fields = [f[0] for f in reader.fields if f[0] != "DeletionFlag"]
    try:
        di = fields.index("Density")
    except ValueError:
        di = None
    tier = np.zeros(pts.shape[0], dtype=np.int8)
    for sr in reader.iterShapeRecords():
        shp = sr.shape
        if not getattr(shp, "points", None):
            continue
        t = _hms_tier(sr.record[di]) if di is not None else 1
        xy = np.asarray(shp.points, dtype=np.float64)     # (lon, lat)
        x0, y0 = xy.min(axis=0)
        x1, y1 = xy.max(axis=0)
        cand = np.flatnonzero((pts[:, 0] >= x0) & (pts[:, 0] <= x1)
                              & (pts[:, 1] >= y0) & (pts[:, 1] <= y1))
        if not len(cand):
            continue
        parts = list(shp.parts) + [len(xy)]
        inside = np.zeros(len(cand), dtype=bool)
        for a, b in zip(parts[:-1], parts[1:]):
            ring = xy[a:b]
            if len(ring) < 3:
                continue
            inside |= _MplPath(ring).contains_points(pts[cand])
        hit = cand[inside]
        tier[hit] = np.maximum(tier[hit], np.int8(t))
    pos = np.flatnonzero(tier > 0)
    if not len(pos):
        return _hms_empty()
    return pd.DataFrame({
        "cell_lat": pts[pos, 1],
        "cell_lon": pts[pos, 0],
        "date": pd.Timestamp(ts).normalize(),
        "hms_smoke": tier[pos],
    })


def fetch_hms_grid(start, end, dest=None):
    """NOAA HMS smoke polygons -> 0.1-degree domain cell raster or None.

    Per-day shapefiles (zip, else loose triplet) parsed with pyshp — no
    GDAL — and rasterized by point-in-polygon over the domain bbox cell
    centers. Rows exist ONLY for smoke-positive cells: frame2.hms_join
    treats a missing (cell, day) inside the raster's coverage window as
    tier 0 (no polygon = no smoke) and anything outside coverage as NaN,
    so absence encodes exactly one thing. A 404 day is an unmapped day and
    is treated as no-smoke (documented limitation: HMS analyst coverage,
    not physical absence); a transiently failing day fails its whole month
    into the sidecar. Month-chunked and resumable; OPTIONAL source. The
    final and the chunk dir are domain-stamped (bbox-clipped content in
    shared dirs; tx keeps hms_grid_*, byte-identical); the raw per-day
    shapefiles are national and deliberately SHARED across domains.

    Output columns: [cell_lat, cell_lon, date, hms_smoke (int8 1..3)]."""
    if not (HAS_PYSHP and HAS_MPL):
        _say("hms: pyshp/matplotlib missing -- skipping HMS grid "
             "(hms_smoke stays NaN outside the by-sensor v1 product)")
        return None
    dx = _v1()
    dest = dest or os.path.join(
        config2.DATA_DIR,
        f"{_dstem('hms_grid')}_{dx._window_tag(start, end)}.parquet")
    prev_failed = []
    if os.path.exists(dest):
        prev_failed = dx._read_failed_months(dest)
        if prev_failed:
            _say(f"hms: cached {dest} is missing month(s) {prev_failed} -- "
                 "retrying them first")
        elif dx._covers_window(dest, start, end):
            _say(f"hms: using cached {dest}")
            return dest
        else:
            _say(f"hms: cached {dest} does not cover {start}..{end} -- "
                 "reassembling from month chunks")

    chunk_dir = _dcache("hms")          # bbox-rasterized chunks: stamped
    raw_dir = os.path.join(config2.CACHE_DIR, "hms", "raw")   # national: shared
    os.makedirs(raw_dir, exist_ok=True)
    lat_axis, lon_axis = _grid_axes(config2.GRID_DEG, decimals=4)
    glon, glat = np.meshgrid(lon_axis, lat_axis)
    pts = np.column_stack([glon.ravel(), glat.ravel()])   # (x=lon, y=lat)

    frames, failed = [], []
    edges = dx._retry_failed_first(dx._month_edges(start, end), prev_failed)
    for m, (lo, hi) in enumerate(edges):
        tag = lo.strftime("%Y%m")
        cp = os.path.join(chunk_dir, f"hms_grid_{tag}.parquet")
        if os.path.exists(cp):
            frames.append(pd.read_parquet(cp))
            continue
        month_rows, absent, err = [], 0, None
        for ts in pd.date_range(lo.normalize(), hi.normalize(), freq="D"):
            try:
                reader = _hms_reader_for_day(ts, raw_dir)
            except Exception as e:
                err = e
                break
            if reader is None:
                absent += 1
                continue
            try:
                month_rows.append(_hms_day_rows(ts, reader, pts))
            finally:
                try:
                    reader.close()
                except Exception:
                    pass
        if err is not None:
            print(f"  hms {tag}: {err}")
            failed.append(tag)
            continue
        chunk = (pd.concat([_hms_empty()] + month_rows, ignore_index=True)
                 if month_rows else _hms_empty())
        chunk.to_parquet(cp, index=False)
        frames.append(chunk)
        _say(f"hms {tag}: {len(chunk):,} smoke cell-days, {absent} unmapped "
             f"day(s) ({m + 1}/{len(edges)} months)")

    if failed:
        _say(f"hms: {len(failed)} month(s) failed and were skipped: {failed} "
             "(recorded in the sidecar; retried first next call)")
    if not frames or not sum(len(f) for f in frames):
        _say("hms: no smoke rows assembled -- skipping final (hms_smoke "
             "stays NaN)")
        return None

    out = pd.concat(frames, ignore_index=True)
    out = out[(out["date"] >= pd.Timestamp(start))
              & (out["date"] <= pd.Timestamp(end))]
    out = out.sort_values(["date", "cell_lat", "cell_lon"]) \
             .reset_index(drop=True)
    out["hms_smoke"] = out["hms_smoke"].astype("int8")
    _atomic_parquet(out, dest)
    dx._write_failed_months(dest, failed)
    _say(f"hms: saved {dest}: {len(out):,} smoke cell-days, "
         f"{out['date'].min().date()} .. {out['date'].max().date()}")
    return dest


# ═════════════════════════════════════════════════════════════════════════════
# 4. Committed statics -> pipeline/static_covariates.parquet (OPTIONAL)
# ═════════════════════════════════════════════════════════════════════════════

def _statics_cache():
    # Shared across domains on purpose: every raw file here is
    # bbox-independent (ETOPO is global, TIGER zips are keyed per state
    # FIPS, NEI summaries are national). The bbox-dependent statics OUTPUT
    # is domain-stamped in fetch_statics instead.
    d = os.path.join(config2.CACHE_DIR, "statics")
    os.makedirs(d, exist_ok=True)
    return d


def _static_dem(lat_axis, lon_axis, qlat, qlon, quick):
    """st_elev: ETOPO 2022 surface elevation (m), bilinear at the lattice.

    Single-file netCDF download from NOAA NCEI THREDDS (30 arcsec primary,
    60 arcsec fallback; --quick prefers 60s for bandwidth). 30 m NASADEM is
    deferred by design: at a ~1.1 km lattice spacing a 30 m DEM adds
    nothing over 30 arcsec (~900 m) sampling."""
    try:
        from netCDF4 import Dataset
    except ImportError:
        raise RuntimeError("netCDF4 not installed (pip install netCDF4)")
    cache = _statics_cache()
    urls = ([ETOPO_60S_URL, ETOPO_30S_URL] if quick
            else [ETOPO_30S_URL, ETOPO_60S_URL])
    path = None
    for url in urls:
        dest = os.path.join(cache, os.path.basename(url))
        path, status = _probe_download(url, dest, attempts=2)
        if status == "ok":
            break
        _say(f"statics/dem: {os.path.basename(url)} unavailable ({status})")
        path = None
    if path is None:
        raise RuntimeError("no ETOPO file could be downloaded")

    bb = config2.TX_BBOX
    ds = Dataset(path)
    try:
        vlat = next(ds.variables[n] for n in ("lat", "latitude", "y")
                    if n in ds.variables)
        vlon = next(ds.variables[n] for n in ("lon", "longitude", "x")
                    if n in ds.variables)
        vz = next(ds.variables[n] for n in ("z", "elevation", "Band1")
                  if n in ds.variables)
        glat = np.asarray(vlat[:], dtype=np.float64)
        glon = np.asarray(vlon[:], dtype=np.float64)
        flip = bool(len(glat) > 1 and glat[0] > glat[-1])
        alat = glat[::-1] if flip else glat       # _bilinear needs ascending
        i0 = max(0, int(np.searchsorted(alat, bb["lat_min"] - 0.1)) - 1)
        i1 = min(len(alat), int(np.searchsorted(alat, bb["lat_max"] + 0.1)) + 1)
        j0 = max(0, int(np.searchsorted(glon, bb["lon_min"] - 0.1)) - 1)
        j1 = min(len(glon), int(np.searchsorted(glon, bb["lon_max"] + 0.1)) + 1)
        zi0, zi1 = (len(alat) - i1, len(alat) - i0) if flip else (i0, i1)
        z = np.ma.filled(np.asarray(vz[zi0:zi1, j0:j1], dtype=np.float64),
                         np.nan)
        if flip:
            z = z[::-1, :]
    finally:
        ds.close()
    elev = _bilinear(alat[i0:i1], glon[j0:j1], z, qlat, qlon)
    _say(f"statics/dem: sampled {os.path.basename(path)}; "
         f"{int(np.isfinite(elev).sum()):,}/{len(elev):,} lattice points")
    return elev


def _static_roads(lat_axis, lon_axis):
    """(st_road_km_1km, st_road_km_5km): TIGER 2023 PRISECROADS density.

    One state file per config2.STATE_FIPS (tx: 48 only, unchanged), each
    cached per state under cache/statics (zip + extracted dir keyed by
    FIPS), so a rerun skips already-fetched states. A failing state RAISES
    and fails the whole roads sub-source: a partial-domain histogram would
    encode "zero km of road" over the missing state — a structural-zero
    lie, exactly what the never-fill principle forbids. TIGER state files
    clip at the state boundary, so roads in out-of-domain states inside the
    bbox are uncounted (same edge limitation the tx run has at NM/OK/LA).

    Each polyline is resampled every ~100 m; each sample carries its share
    of segment length (km) into a 2-D histogram on the lattice cells; box
    sums over +-1 km / +-5 km half-widths approximate km-of-road within the
    radius. Documented approximations: square window ~ circle, equirect
    segment lengths, sample quantization at 100 m — all sub-cell effects at
    a 0.01-degree lattice."""
    if not HAS_PYSHP:
        raise RuntimeError("pyshp not installed (pip install pyshp)")
    cache = _statics_cache()

    slat, slon, sw = [], [], []
    for fips in config2.STATE_FIPS:
        stem = f"tl_2023_{fips}_prisecroads"
        zp, status = _probe_download(TIGER_ROADS_URL.format(fips=fips),
                                     os.path.join(cache, stem + ".zip"))
        if status != "ok":
            raise RuntimeError(
                f"TIGER prisecroads download {status} for state {fips}")
        exdir = os.path.join(cache, stem)
        if not os.path.isdir(exdir):
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(exdir + ".tmp")
            os.replace(exdir + ".tmp", exdir)
        shp = glob.glob(os.path.join(exdir, "*.shp"))
        if not shp:
            raise RuntimeError(f"no .shp member in TIGER zip for state {fips}")

        n_before = len(sw)
        rd = _pyshp.Reader(shp[0])
        for shape in rd.iterShapes():
            xy = np.asarray(shape.points, dtype=np.float64)
            if len(xy) < 2:
                continue
            parts = list(shape.parts) + [len(xy)]
            for a, b in zip(parts[:-1], parts[1:]):
                p = xy[a:b]
                if len(p) < 2:
                    continue
                lon0, lat0 = p[:-1, 0], p[:-1, 1]
                lon1, lat1 = p[1:, 0], p[1:, 1]
                latm = 0.5 * (lat0 + lat1)
                dxk = (lon1 - lon0) * KM_PER_DEG_LON_EQ \
                    * np.cos(np.radians(latm))
                dyk = (lat1 - lat0) * KM_PER_DEG_LAT
                seg = np.hypot(dxk, dyk)
                n = np.maximum(1, np.ceil(seg / ROAD_SAMPLE_KM)
                               .astype(np.int64))
                reps = np.repeat(np.arange(len(seg)), n)
                offs = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
                frac = (offs + 0.5) / np.repeat(n, n)
                slat.append(lat0[reps] + (lat1 - lat0)[reps] * frac)
                slon.append(lon0[reps] + (lon1 - lon0)[reps] * frac)
                sw.append(np.repeat(seg / n, n))
        rd.close()
        if len(sw) == n_before:
            raise RuntimeError(
                f"TIGER shapefile for state {fips} yielded no polylines")
    if not slat:
        raise RuntimeError("TIGER shapefiles yielded no polylines")
    slat = np.concatenate(slat)
    slon = np.concatenate(slon)
    sw = np.concatenate(sw)

    step_lat = float(lat_axis[1] - lat_axis[0])
    step_lon = float(lon_axis[1] - lon_axis[0])
    lat_edges = np.append(lat_axis - step_lat / 2, lat_axis[-1] + step_lat / 2)
    lon_edges = np.append(lon_axis - step_lon / 2, lon_axis[-1] + step_lon / 2)
    hist, _, _ = np.histogram2d(slat, slon, bins=[lat_edges, lon_edges],
                                weights=sw)
    mean_lat = float(np.mean(lat_axis))
    km_lat = KM_PER_DEG_LAT * step_lat
    km_lon = KM_PER_DEG_LON_EQ * math.cos(math.radians(mean_lat)) * step_lon
    r1 = (max(1, int(round(1.0 / km_lat))), max(1, int(round(1.0 / km_lon))))
    r5 = (max(1, int(round(5.0 / km_lat))), max(1, int(round(5.0 / km_lon))))
    km1 = _box_sum(hist, *r1).ravel()
    km5 = _box_sum(hist, *r5).ravel()
    _say(f"statics/roads: {sw.sum():,.0f} km of prisecroads sampled at "
         f"{len(sw):,} points; windows +-{r1} / +-{r5} cells")
    return km1, km5


def _nei_columns(cols):
    """Case-insensitive column resolution across NEI summary vintages."""
    low = {c.strip().lower(): c for c in cols}

    def pick(cands):
        for c in cands:
            if c in low:
                return low[c]
        return None

    return {
        "lat": pick(["site latitude", "latitude msr", "latitude", "lat"]),
        "lon": pick(["site longitude", "longitude msr", "longitude", "lon"]),
        "poll": pick(["pollutant code", "pollutant cd", "pollutant"]),
        "emis": pick(["total emissions", "emissions",
                      "total emissions (tons)"]),
        "uom": pick(["emissions uom", "uom", "emissions unit of measure"]),
    }


def _nei_facilities(year, cache):
    """Facility PM25-PRI short tons at (lat, lon) for one NEI year.

    Tries the facility-level summary zips first (smaller, national), then
    the by-regions process file — its members are selected by the domain
    states' EPA regions (EPA_REGION_BY_FIPS; tx: region 6 only, unchanged)
    and parsed FIRST-PRODUCTIVE-MEMBER-PER-REGION: a later member matching
    only already-served regions (e.g. a second per-sector file for region 6)
    is skipped so no facility is double-counted, while every remaining
    region of a multistate domain is still parsed. Facilities are bbox-filtered on
    coordinates rather than state strings (robust across vintages), so
    bbox fringes outside the domain states' regions are covered only by
    the national summary path — a documented fallback limitation. Unknown
    emission units are dropped with a printed count, never assumed."""
    urls = [
        f"https://gaftp.epa.gov/air/nei/{year}/data_summaries/"
        f"Facility%20Level%20by%20Pollutant.zip",
        f"https://gaftp.epa.gov/air/nei/{year}/data_summaries/"
        f"{year}nei_facility_level_by_pollutant.zip",
        f"https://gaftp.epa.gov/air/nei/{year}/data_summaries/"
        f"{year}nei_facility_process_byregions.zip",
    ]
    zp = None
    for url in urls:
        dest = os.path.join(cache,
                            f"nei_{year}_" + os.path.basename(url)
                            .replace("%20", "_"))
        zp, status = _probe_download(url, dest, attempts=2)
        if status == "ok":
            break
        _say(f"statics/nei {year}: {url.rsplit('/', 1)[-1]} {status}")
        zp = None
    if zp is None:
        raise RuntimeError(f"no NEI {year} summary zip reachable")

    bb = config2.TX_BBOX
    acc = {}
    with zipfile.ZipFile(zp) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not members:
            raise RuntimeError(f"NEI {year} zip has no csv members")
        regional = [n for n in members if "region" in n.lower()]
        if regional:
            want = {EPA_REGION_BY_FIPS[f] for f in config2.STATE_FIPS
                    if f in EPA_REGION_BY_FIPS}
            unmapped = [f for f in config2.STATE_FIPS
                        if f not in EPA_REGION_BY_FIPS]
            if unmapped:
                _say(f"statics/nei {year}: state(s) {unmapped} lack an EPA "
                     "region mapping -- parsing every regional member")
            hits = [n for n in regional
                    if any(r in os.path.basename(n) for r in want)]
            members = regional if unmapped else (hits or regional)
        done_regions = set()
        for name in members:
            # Regional path: at most ONE productive member per wanted region.
            # A member matching only regions that already had a productive
            # member is a per-sector duplicate — parsing it would sum the
            # same facilities twice. A member matching no wanted region
            # (unmapped-FIPS superset fallback) is always parsed.
            matched = ({r for r in want if r in os.path.basename(name)}
                       if regional else set())
            if matched and matched <= done_regions:
                continue
            with zf.open(name) as fh:
                head = pd.read_csv(fh, nrows=0)
            cols = _nei_columns(head.columns)
            if not all(cols[k] for k in ("lat", "lon", "poll", "emis")):
                continue
            use = [v for v in cols.values() if v]
            member_rows = 0
            with zf.open(name) as fh:
                for chunk in pd.read_csv(fh, usecols=use, chunksize=250_000,
                                         low_memory=False):
                    c = chunk
                    poll = c[cols["poll"]].astype(str).str.strip().str.upper()
                    c = c[poll == "PM25-PRI"]
                    if not len(c):
                        continue
                    lat = pd.to_numeric(c[cols["lat"]], errors="coerce")
                    lon = pd.to_numeric(c[cols["lon"]], errors="coerce")
                    emis = pd.to_numeric(c[cols["emis"]], errors="coerce")
                    if cols["uom"]:
                        uom = c[cols["uom"]].astype(str).str.strip().str.upper()
                        bad = ~uom.isin(["TON", "TONS"]) & ~(uom == "LB")
                        if bad.any():
                            _say(f"statics/nei {year}: dropped "
                                 f"{int(bad.sum())} rows with unknown UOM")
                        emis = emis.where(~bad)
                        emis = emis.where(uom != "LB", emis / 2000.0)
                    keep = (np.isfinite(lat) & np.isfinite(lon)
                            & np.isfinite(emis)
                            & (lat >= bb["lat_min"]) & (lat <= bb["lat_max"])
                            & (lon >= bb["lon_min"]) & (lon <= bb["lon_max"]))
                    member_rows += int(keep.sum())
                    for la, lo_, em in zip(lat[keep], lon[keep], emis[keep]):
                        key = (round(float(la), 5), round(float(lo_), 5))
                        acc[key] = acc.get(key, 0.0) + float(em)
            # National summary path: first productive member wins (v2
            # behavior). Regional path: first productive member PER wanted
            # region — the member's regions are marked served so per-sector
            # duplicates are skipped above, while the loop continues to the
            # remaining regions of a multistate domain (breaking here would
            # drop them).
            if not member_rows:
                continue
            if not regional:
                break
            done_regions |= matched
    if not acc:
        raise RuntimeError(f"NEI {year}: no PM25-PRI facilities parsed")
    pts = np.asarray(list(acc.keys()), dtype=np.float64)
    tons = np.asarray(list(acc.values()), dtype=np.float64)
    _say(f"statics/nei {year}: {len(tons):,} facility points, "
         f"{tons.sum():,.0f} short tons PM25-PRI")
    return pts, tons


def _static_nei(qlat, qlon, cache):
    """{year: (st_nei_pm25_5km, st_nei_pm25_20km)} facility-sum arrays.

    Per lattice point: sum of facility PM25-PRI short tons within 5 km and
    20 km (cKDTree on equirect-projected km coordinates; the loop runs per
    FACILITY — thousands — against a lattice tree, not per lattice point).
    A zero is the honest 'no facility within radius'; a failed year is
    absent from the dict, never zero-filled."""
    if not HAS_SCIPY:
        raise RuntimeError("scipy not installed (pip install scipy)")
    lat0 = float(np.mean(qlat))
    kx = KM_PER_DEG_LON_EQ * math.cos(math.radians(lat0))
    q_xy = np.column_stack([qlon * kx, qlat * KM_PER_DEG_LAT])
    tree = cKDTree(q_xy)
    out = {}
    for year in NEI_YEARS:
        try:
            pts, tons = _nei_facilities(year, cache)
        except Exception as e:
            _say(f"statics/nei {year}: SKIPPED -- {e}")
            continue
        f_xy = np.column_stack([pts[:, 1] * kx, pts[:, 0] * KM_PER_DEG_LAT])
        s5 = np.zeros(len(qlat))
        s20 = np.zeros(len(qlat))
        for i in range(len(f_xy)):
            for radius, arr in ((5.0, s5), (20.0, s20)):
                idx = tree.query_ball_point(f_xy[i], r=radius)
                if idx:
                    arr[idx] += tons[i]
        out[int(year)] = (s5, s20)
    return out


def _static_pop(qlat, qlon):
    """st_pop_density: nearest-tract people-per-km2 proxy.

    Population comes from the committed purpleair parquet (POPULATION per
    GEOID at tract-centroid lat/lon — NOT the sensor latitude/longitude,
    audit 06 §4). Tract area is unknown here, so it is approximated from
    centroid spacing: area_i ~ pi * d3_i^2 / 3 with d3 the distance to the
    3rd-nearest centroid (union of purpleair + backend tract_lookup
    centroids for spacing). ROUGH by construction — a coverage-density
    proxy, not a census areal density — and documented as such.

    Texas-scoped by its sources: the committed purpleair parquet and the
    backend tract lookup carry Texas tracts only, and the nearest-tract
    query has no distance cap, so a wider domain would silently inherit
    Texas densities everywhere. Non-tx domains therefore RAISE (the column
    stays absent — honest) until a domain-wide tract source exists."""
    if not HAS_SCIPY:
        raise RuntimeError("scipy not installed (pip install scipy)")
    if config2.DOMAIN != "tx":
        raise RuntimeError(
            "committed tract-centroid sources are Texas-scoped; a "
            f"nearest-tract proxy would smear Texas densities across "
            f"domain {config2.DOMAIN!r} -- st_pop_density stays absent")
    pa_path = os.path.join(config2.PIPELINE_DIR,
                           "purpleair_full_dataset.parquet")
    if not os.path.exists(pa_path):
        raise RuntimeError(f"{pa_path} missing")
    pa = pd.read_parquet(pa_path, columns=["GEOID", "lat", "lon",
                                           "POPULATION"])
    pa["GEOID"] = pa["GEOID"].astype(str)
    tr = pa.drop_duplicates("GEOID").reset_index(drop=True)
    tr = tr[np.isfinite(tr["lat"]) & np.isfinite(tr["lon"])]

    cent = tr[["lat", "lon"]].to_numpy(dtype=np.float64)
    lookup = os.path.join(config2.ROOT, "backend", "static",
                          "tract_lookup.parquet")
    union = [cent]
    if os.path.exists(lookup):
        try:
            tl = pd.read_parquet(lookup)
            cols = {c.lower(): c for c in tl.columns}
            la, lo = cols.get("lat"), cols.get("lon")
            if la and lo:
                extra = tl[[la, lo]].to_numpy(dtype=np.float64)
                union.append(extra[np.isfinite(extra).all(axis=1)])
                _say(f"statics/pop: tract_lookup adds {len(union[-1]):,} "
                     "centroids for spacing")
        except Exception as e:
            _say(f"statics/pop: tract_lookup unreadable ({e}) -- "
                 "using purpleair centroids only")
    allc = np.vstack(union)

    lat0 = float(np.mean(qlat))
    kx = KM_PER_DEG_LON_EQ * math.cos(math.radians(lat0))

    def to_km(a):
        return np.column_stack([a[:, 1] * kx, a[:, 0] * KM_PER_DEG_LAT])

    union_tree = cKDTree(to_km(allc))
    d, _ = union_tree.query(to_km(cent), k=4)     # self + 3 nearest
    d3 = np.maximum(d[:, 3], 0.1)                 # km; floor vs coincident
    area = math.pi * d3 ** 2 / 3.0
    dens = tr["POPULATION"].to_numpy(dtype=np.float64) / area
    dens = np.clip(dens, 0.0, None)

    known_tree = cKDTree(to_km(cent))
    _, idx = known_tree.query(np.column_stack([qlon * kx,
                                               qlat * KM_PER_DEG_LAT]), k=1)
    out = dens[idx]
    _say(f"statics/pop: {len(tr):,} tracts -> proxy density, median "
         f"{float(np.median(out)):.1f} people/km2")
    return out


def _static_imperv(qlat, qlon):
    """st_imperv_1km: Annual NLCD 2021 fractional impervious, ~1 km mean.

    COG range-reads only (rasterio + a public endpoint that answers); the
    TX window is read decimated with average resampling to ~1 km, then
    nearest-sampled at the lattice. Any failure -> RuntimeError -> the
    column is absent (attempt-with-graceful-skip, BUILD_NOTES decision 3)."""
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio not installed")
    from rasterio.enums import Resampling
    from rasterio.warp import transform as rio_transform
    from rasterio.windows import from_bounds

    bb = config2.TX_BBOX
    last = None
    for url in NLCD_IMPERV_URLS:
        try:
            with _rasterio.open(url) as ds:
                xs, ys = rio_transform(
                    "EPSG:4326", ds.crs,
                    [bb["lon_min"], bb["lon_max"], bb["lon_min"],
                     bb["lon_max"]],
                    [bb["lat_min"], bb["lat_min"], bb["lat_max"],
                     bb["lat_max"]])
                win = from_bounds(min(xs), min(ys), max(xs), max(ys),
                                  ds.transform)
                scale = max(1, int(round(1000.0 / abs(ds.res[0]))))
                out_h = max(1, int(math.ceil(win.height / scale)))
                out_w = max(1, int(math.ceil(win.width / scale)))
                data = ds.read(1, window=win, out_shape=(out_h, out_w),
                               resampling=Resampling.average, masked=True)
                arr = np.ma.filled(data.astype(np.float64), np.nan)
                arr[arr > 100.0] = np.nan       # 250/127 nodata codes
                wt = ds.window_transform(win)
                ax = wt.a * (win.width / out_w)
                ey = wt.e * (win.height / out_h)
                qx, qy = rio_transform("EPSG:4326", ds.crs,
                                       list(map(float, qlon)),
                                       list(map(float, qlat)))
                col = np.floor((np.asarray(qx) - wt.c) / ax).astype(np.int64)
                row = np.floor((np.asarray(qy) - wt.f) / ey).astype(np.int64)
                ok = ((row >= 0) & (row < out_h) & (col >= 0) & (col < out_w))
                out = np.full(len(qlat), np.nan)
                out[ok] = arr[row[ok], col[ok]]
                _say(f"statics/imperv: sampled {url.rsplit('/', 1)[-1]}, "
                     f"{int(np.isfinite(out).sum()):,} finite lattice points")
                return out
        except Exception as e:
            last = e
            _say(f"statics/imperv: {url.rsplit('/', 1)[-1]} failed ({e})")
    raise RuntimeError(f"no NLCD endpoint answered ({last})")


def fetch_statics(quick=False, dest=None):
    """Committed statics -> pipeline/static_covariates.parquet path or None.

    A 0.01-degree lattice over the domain bbox (0.05 under --quick, written
    to a _quick-suffixed file so a smoke-test lattice can never poison the
    committed statics; non-tx domains get a domain-stamped file for the
    same reason — pipeline/ is shared, and the committed Texas
    static_covariates.parquet must neither be overwritten by nor served to
    a wider domain)
    with columns lat, lon [, year] + whichever of st_elev, st_road_km_1km,
    st_road_km_5km, st_nei_pm25_5km, st_nei_pm25_20km, st_pop_density,
    st_imperv_1km could be built. Sub-sources are INDEPENDENT: one failing
    leaves its columns absent (frame2 then simply has no such feature),
    never NaN-backfilled into pretend-coverage, and never zero-filled.

    Year semantics (deliberate deviation from the draft NaN-year schema):
    frame2.statics_join's year-keyed picker takes, per query row, the
    latest static year <= the query year via searchsorted over the sorted
    unique years — a NaN year row can NEVER be selected by that picker
    (NaN sorts last and searchsorted of any finite year lands before it).
    So when NEI succeeds, the year-invariant columns are REPLICATED into
    every year subset (year=2020 rows and year=2023 rows each carry the
    full column set); when NEI fails entirely the year column is omitted
    and statics_join uses its single-tree branch. Either way every written
    row is reachable."""
    step = STATICS_STEP_QUICK if quick else STATICS_STEP
    dest = dest or os.path.join(
        config2.PIPELINE_DIR,
        _dstem("static_covariates") + ("_quick" if quick else "")
        + ".parquet")
    if os.path.exists(dest) and os.environ.get("FORCE") != "1":
        _say(f"statics: {dest} exists (FORCE=1 to rebuild) -- skip")
        return dest

    lat_axis, lon_axis = _grid_axes(step)
    glat, glon = np.meshgrid(lat_axis, lon_axis, indexing="ij")
    qlat = glat.ravel()
    qlon = glon.ravel()
    _say(f"statics: lattice {len(lat_axis)} x {len(lon_axis)} = "
         f"{len(qlat):,} points at {step} deg")
    cache = _statics_cache()

    cols = {}
    try:
        cols["st_elev"] = _static_dem(lat_axis, lon_axis, qlat, qlon, quick)
    except Exception as e:
        _say(f"statics/dem: SKIPPED -- {e}")
    try:
        km1, km5 = _static_roads(lat_axis, lon_axis)
        cols["st_road_km_1km"] = km1
        cols["st_road_km_5km"] = km5
    except Exception as e:
        _say(f"statics/roads: SKIPPED -- {e}")
    try:
        cols["st_pop_density"] = _static_pop(qlat, qlon)
    except Exception as e:
        _say(f"statics/pop: SKIPPED -- {e}")
    try:
        cols["st_imperv_1km"] = _static_imperv(qlat, qlon)
    except Exception as e:
        _say(f"statics/imperv: SKIPPED -- {e}")
    try:
        nei = _static_nei(qlat, qlon, cache)
    except Exception as e:
        _say(f"statics/nei: SKIPPED -- {e}")
        nei = {}

    if not cols and not nei:
        _say("statics: every sub-source failed -- nothing written")
        return None

    base = {"lat": qlat, "lon": qlon}
    base.update(cols)
    if nei:
        blocks = []
        for year in sorted(nei):
            s5, s20 = nei[year]
            blk = pd.DataFrame(base)
            blk["year"] = np.int64(year)
            blk["st_nei_pm25_5km"] = s5
            blk["st_nei_pm25_20km"] = s20
            blocks.append(blk)
        out = pd.concat(blocks, ignore_index=True)
    else:
        out = pd.DataFrame(base)
    _atomic_parquet(out, dest)
    _say(f"statics: saved {dest}: {len(out):,} rows, columns "
         f"{[c for c in out.columns if c not in ('lat', 'lon', 'year')]}"
         + (f", years {sorted(nei)}" if nei else ", no year key"))
    if quick:
        _say("statics: QUICK lattice -- frame2's default statics path is "
             "pipeline/static_covariates.parquet; pass the _quick file "
             "explicitly for smoke tests only")
    return dest


# ═════════════════════════════════════════════════════════════════════════════
# 4b. EDGAR v8.1 annual emission gridmaps -> domain static parquet (OPTIONAL)
# ═════════════════════════════════════════════════════════════════════════════

def _edgar_cache():
    """Shared unstamped CACHE_DIR subdir for EDGAR raw downloads (created).

    The per-(pollutant, year) zips and the .nc members extracted beside
    them are GLOBAL 0.1-degree grids, so the raw cache is deliberately
    shared across domains (the statics/HMS raw precedent: content that
    does not depend on the bbox lives unstamped); only the bbox-subset
    final under DATA_DIR carries the domain stamp."""
    d = os.path.join(config2.CACHE_DIR, "edgar")
    os.makedirs(d, exist_ok=True)
    return d


def _edgar_pollutant_frame(poll, col, year, cache):
    """One EDGAR TOTALS pollutant, bbox-subset -> DataFrame [lat, lon, col].

    Downloads the per-year zip (resumable: an existing zip, and separately
    an already-extracted .nc, are trusted and skipped), extracts the single
    .nc member atomically (tmp + os.replace) and reads it with
    xarray+netcdf4. The `emissions` variable is tonnes of substance per
    0.1-degree cell per year on a global grid (verified 2026-08-09 on the
    v8.1 PM2.5 2022 file); longitudes are normalized to [-180, 180) and
    both axes sorted ascending before the bbox .sel, guarding against a
    future vintage shipping 0..360. Raises on any failure: the caller
    records the pollutant in the sidecar and leaves its column ABSENT,
    never filled."""
    import xarray as xr
    url = EDGAR_URL.format(poll=poll, year=year)
    zdest = os.path.join(cache, os.path.basename(url))
    ncdest = os.path.splitext(zdest)[0] + ".nc"
    if not os.path.exists(ncdest):
        zp, status = _probe_download(url, zdest)
        if status != "ok":
            raise RuntimeError(f"download {status} for {url}")
        with zipfile.ZipFile(zp) as zf:
            members = [n for n in zf.namelist() if n.lower().endswith(".nc")]
            if len(members) != 1:
                raise RuntimeError(
                    f"{os.path.basename(zp)} holds {len(members)} .nc "
                    "members (expected exactly 1)")
            tmp = ncdest + ".tmp"
            with zf.open(members[0]) as src, open(tmp, "wb") as out:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            os.replace(tmp, ncdest)

    bb = config2.TX_BBOX
    ds = xr.open_dataset(ncdest, engine="netcdf4")
    try:
        if "emissions" in ds.data_vars:
            da = ds["emissions"]
        elif len(ds.data_vars) == 1:
            da = ds[list(ds.data_vars)[0]]
        else:
            raise RuntimeError(
                f"no 'emissions' variable in {os.path.basename(ncdest)} "
                f"(data_vars: {sorted(ds.data_vars)})")
        if float(np.asarray(da["lon"]).max()) > 180.0:
            da = da.assign_coords(lon=((da["lon"] + 180.0) % 360.0) - 180.0)
        da = da.sortby("lat").sortby("lon")
        sub = da.sel(lat=slice(bb["lat_min"], bb["lat_max"]),
                     lon=slice(bb["lon_min"], bb["lon_max"])).load()
    finally:
        ds.close()
    df = sub.to_dataframe(name=col).reset_index()[["lat", "lon", col]]
    if not len(df):
        raise RuntimeError(f"bbox subset of {poll} {year} is empty")
    # Round the native centers so the per-pollutant merge on (lat, lon) is
    # exact (the merra2_combined float-jitter precedent).
    df["lat"] = np.round(df["lat"].astype(np.float64), 5)
    df["lon"] = np.round(df["lon"].astype(np.float64), 5)
    df[col] = df[col].astype(np.float64)
    _say(f"edgar {poll} {year}: {len(df):,} cells, "
         f"{float(np.nansum(df[col].to_numpy())):,.0f} t/yr in the "
         "domain bbox")
    return df


def fetch_edgar_domain(year=None, dest=None):
    """EDGAR v8.1 annual TOTALS emissions over the domain bbox -> path/None.

    EU JRC EDGAR v8.1 air-pollutant release (public, no auth): global
    0.1-degree ANNUAL sector-aggregated (TOTALS) emission gridmaps, one
    ~16 MB netCDF zip per (pollutant, year) at EDGAR_URL (pattern verified
    against the live JRC open-data directory listing, 2026-08-09). PM2.5,
    NOx and SO2 are fetched for the latest published year (EDGAR_YEAR =
    2022; the release is frozen at FT2022, so newer data would be a new
    release id, not a larger {year}), each bbox-subset to config2.TX_BBOX
    and outer-merged on the rounded native cell centers, which the three
    files share by construction.

    A NEW v4 feature source, registered as the 'edgar' key in
    write_external_paths for ALL domains including tx: no shipped consumer
    reads the key, so adding it changes no current pipeline behavior and
    the frozen v2 run stays reproducible. Pollutants are INDEPENDENT: a
    failing one leaves its column absent (never filled), is announced, and
    is recorded in the {dest}.failed.json sidecar so a partial parquet is
    never mistaken for a complete one and the missing pollutant is retried
    on the next call; when no pollutant can be fetched this returns None
    (OPTIONAL source: run_data warns and continues).

    Static product, so the final is year-stamped rather than
    window-stamped: DATA_DIR/{_dstem('edgar_v81')}_{year}.parquet (tx:
    edgar_v81_2022.parquet, west7: edgar_v81_west7_2022.parquet); the
    stamped stem plus the `_[0-9]*` glob tail keeps each domain's registry
    from ever admitting the other's file. Raw zips live in the shared
    cache/edgar dir because the grids are global. Output columns: [lat,
    lon, edgar_pm25, edgar_nox, edgar_so2] in tonnes of substance per cell
    per year, minus any pollutant that failed. A zero is EDGAR's own "no
    emission in this cell", a fact, never a fill."""
    dx = _v1()
    year = int(year or EDGAR_YEAR)
    dest = dest or os.path.join(
        config2.DATA_DIR, f"{_dstem('edgar_v81')}_{year}.parquet")
    prev_failed = dx._read_failed_months(dest)
    if os.path.exists(dest) and not prev_failed \
            and os.environ.get("FORCE") != "1":
        _say(f"edgar: using cached {dest}")
        return dest
    if prev_failed:
        _say(f"edgar: cached {dest} is missing column(s) {prev_failed} -- "
             "retrying them (already-cached pollutants reassemble from the "
             "shared raw cache)")
    cache = _edgar_cache()

    frames, failed = [], []
    for poll, col in EDGAR_POLLUTANTS:
        try:
            frames.append(_edgar_pollutant_frame(poll, col, year, cache))
        except Exception as e:
            _say(f"edgar {poll} {year}: SKIPPED -- {e} ({col} stays "
                 "absent, never filled)")
            failed.append(col)
    if not frames:
        _say("edgar: no pollutant could be fetched -- nothing written "
             "(edgar_* features stay absent)")
        return None

    out = frames[0]
    for df in frames[1:]:
        out = out.merge(df, on=["lat", "lon"], how="outer")
    out = out.sort_values(["lat", "lon"]).reset_index(drop=True)
    _atomic_parquet(out, dest)
    dx._write_failed_months(dest, failed)
    _say(f"edgar: saved {dest}: {len(out):,} cells, columns "
         f"{[c for c in out.columns if c not in ('lat', 'lon')]}")
    return dest


# ═════════════════════════════════════════════════════════════════════════════
# 5. external_paths.json + data-pa decision artifact
# ═════════════════════════════════════════════════════════════════════════════

def _window_span(path):
    """(span_days, mtime) sort key for window-stamped filenames; files
    without a parsable _{YYYYMMDD}_{YYYYMMDD} tail sort by mtime alone."""
    stem = os.path.splitext(os.path.basename(path))[0]
    toks = stem.split("_")
    span = 0
    if len(toks) >= 2 and toks[-1].isdigit() and toks[-2].isdigit():
        try:
            span = (pd.Timestamp(toks[-1]) - pd.Timestamp(toks[-2])).days
        except ValueError:
            span = 0
    return (span, os.path.getmtime(path))


def _best_glob(patterns, require_rows=False):
    """Widest-window (then newest) existing match, or None. Files whose
    sidecar records failures still win over nothing — the sidecar is the
    caller's warning, not a disqualification."""
    hits = []
    for pat in patterns:
        hits.extend(glob.glob(pat))
    hits = sorted(set(hits), key=_window_span, reverse=True)
    for h in hits:
        if require_rows:
            try:
                if not len(pd.read_parquet(h, columns=["date"])):
                    continue
            except Exception:
                continue
        return h
    return None


def write_external_paths():
    """Write artifact external_paths.json -> its path (the frame2 registry).

    Keys {aqs, pa_daily, geoscf, merra2, cams, met_extra, hms_grid, edgar,
    pa_v4_daily, pa_v4_pairs}
    (plus `statics` for non-tx domains), each included ONLY when its file
    exists
    (frame2 skips missing keys loudly; an absent key is honest degradation,
    a dead path is a crash). merra2 prefers the combined parquet, then
    aerosol-only, then SLV-only.

    Domain routing: every bbox-dependent candidate resolves through the
    domain-stamped stems, and the `_[0-9]*` glob tails only admit
    window-tagged files — so a west7 final in the shared data/ dir can
    never be registered for tx, nor vice versa. The v1 Texas products
    (V1_DIR/data geoscf + merra2 aerosol, the committed by-cell pipeline
    parquets, pipeline statics) are offered to the tx domain ONLY: frame2's
    nearest-cell joins carry no distance cap, so registering a Texas-bbox
    file for a wider domain would silently smear Texas values across it.
    pa_daily stays unstamped by design (EXPANSION Phase 1: the TX archive
    is the PA source for every domain, config2.PA_STATE_FIPS). Under a
    non-tx domain the 'geoscf' key resolves to fetch_geoscf_domain's
    domain-stamped final (geoscf_pm25_{domain}_{window}.parquet); under tx
    the v1 Texas candidates keep first priority, unchanged.

    `edgar` (v4 feature source) is registered for EVERY domain, tx
    included: its final is domain-stamped by fetch_edgar_domain, and the
    key is NEW, so no shipped consumer reads it and the current pipeline
    behavior is unchanged by its presence.

    `pa_v4_daily` / `pa_v4_pairs` (v4 PurpleAir archive ingest,
    pa_v4_ingest.py) follow the same precedent: NEW keys, domain-stamped
    window-stamped finals, and the only readers sit behind
    AQNET2_PA_SOURCE=v4 (default v2), so registering them changes no
    current pipeline behavior either."""
    v1_data = os.path.join(config2.V1_DIR, "data")
    tx = config2.DOMAIN == "tx"
    geoscf_pats = [os.path.join(config2.DATA_DIR,
                                _dstem("geoscf_pm25") + "_[0-9]*.parquet")]
    merra2_fallback_pats = [os.path.join(
        config2.DATA_DIR,
        f"merra2_slv_{config2.DOMAIN}_[0-9]*.parquet")]
    if tx:
        geoscf_pats.insert(0, os.path.join(v1_data,
                                           "geoscf_pm25_[0-9]*.parquet"))
        merra2_fallback_pats.insert(0, os.path.join(
            v1_data, "merra2_daily_tx_[0-9]*.parquet"))
    cand = {
        "aqs": _best_glob(
            [os.path.join(config2.DATA_DIR,
                          config2.AQS_STEM + "_*.parquet")]),
        "pa_daily": os.path.join(config2.PIPELINE_DIR,
                                 "purpleair_full_dataset.parquet"),
        "geoscf": _best_glob(geoscf_pats),
        "merra2": _best_glob(
            [os.path.join(config2.DATA_DIR,
                          _dstem("merra2_combined") + "_[0-9]*.parquet")]) \
            or _best_glob(merra2_fallback_pats),
        "cams": os.path.join(config2.PIPELINE_DIR,
                             _dstem("airquality_by_cell") + ".parquet"),
        "met_extra": os.path.join(config2.PIPELINE_DIR,
                                  _dstem("met_extra_by_cell") + ".parquet"),
        "hms_grid": _best_glob(
            [os.path.join(config2.DATA_DIR,
                          _dstem("hms_grid") + "_[0-9]*.parquet")],
            require_rows=True),
        "edgar": _best_glob(
            [os.path.join(config2.DATA_DIR,
                          _dstem("edgar_v81") + "_[0-9]*.parquet")]),
        "pa_v4_daily": _best_glob(
            [os.path.join(config2.DATA_DIR,
                          _dstem("pa_v4_daily") + "_[0-9]*.parquet")]),
        "pa_v4_pairs": _best_glob(
            [os.path.join(config2.DATA_DIR,
                          _dstem("pa_v4_pairs") + "_[0-9]*.parquet")]),
    }
    if not tx:
        # frame2's default statics path is the committed Texas lattice;
        # route non-tx frames to the domain-stamped file (or omit, loudly).
        cand["statics"] = os.path.join(
            config2.PIPELINE_DIR, _dstem("static_covariates") + ".parquet")
    paths = {}
    for key, path in cand.items():
        if path and os.path.exists(path):
            paths[key] = path
            _say(f"external_paths: {key} -> {path}")
        else:
            _say(f"external_paths: {key} -- no file found (key omitted; "
                 "downstream features stay NaN)")
    dest = config2.artifact("external_paths.json")
    _atomic_json(paths, dest)
    _say(f"wrote {dest} ({len(paths)} keys)")
    return dest


def data_pa_decision():
    """Write artifact data_pa_decision.json -> its path (stage `data-pa`).

    BUILD_NOTES scope decision 1: the cf_1 refetch is SKIPPED. The
    committed PurpleAir archive is ATM-only; below ~20 ug/m3 the PA
    firmware's ATM and cf_1 estimates coincide, so ATM is treated as cf_1
    under the threshold and flagged channel_reconstructed=1 above it, with
    cal_var inflated (calibrate.py) and reconstructed rows excluded from
    exceedance labels (exceed.py). This artifact records the decision so
    the DAG stage is auditable, and carries the API-cost arithmetic as
    explicit placeholders to be filled before any future refetch is
    reconsidered — a placeholder None is a visible unknown, not a fill."""
    dest = config2.artifact("data_pa_decision.json")
    payload = {
        "stage": "data-pa",
        "decision": "SKIP_REFETCH",
        "policy": "cf1_reconstruction_fallback",
        "design_ref": "DESIGN section 4 sanctioned fallback; section 12.1",
        "build_notes_ref": "BUILD_NOTES scope decision 1",
        "rationale": (
            "Committed archive is ATM-only (single pm25 column, audit 06 "
            "section 4). Below the reconstruction threshold the PA "
            "firmware's ATM and cf_1 outputs coincide, so pa_cf1 := ATM "
            "there; above it channel_reconstructed=1, cal_var is inflated "
            "by channel_recon_var_factor in calibrate.py, and exceed.py "
            "excludes reconstructed rows from exceedance labels. A full "
            "cf_1 refetch of the multi-year archive is deferred until the "
            "API cost arithmetic below is filled in and approved."),
        "cf1_recon_threshold_ugm3": CF1_RECON_UGM3,
        "channel_recon_var_factor": CHANNEL_RECON_VAR_FACTOR,
        "exceedance_label_policy":
            "channel_reconstructed rows excluded (enforced in exceed.py)",
        "api_cost": {
            "n_sensors": None,
            "n_sensor_days": None,
            "points_per_request": None,
            "requests_needed": None,
            "est_cost_usd": None,
            "note": ("fill from the PurpleAir API price sheet before any "
                     "future refetch decision; ~412k committed sensor-days "
                     "is the lower bound on volume"),
        },
        "created": pd.Timestamp.now().strftime("%Y-%m-%d"),
    }
    _atomic_json(_jsonable(payload), dest)
    _say(f"wrote {dest}")
    return dest


# ═════════════════════════════════════════════════════════════════════════════
# 6. Stage runners + CLI
# ═════════════════════════════════════════════════════════════════════════════

def run_data_pa():
    dest = config2.artifact("data_pa_decision.json")
    if os.path.exists(dest) and os.environ.get("FORCE") != "1":
        _say(f"{dest} exists (FORCE=1 to rebuild) -- skip")
        return 0
    _banner("data-pa")
    data_pa_decision()
    return 0


def run_data(quick=False, start=None, end=None, years=None):
    """Stage `data`: AQS v2 (REQUIRED) + SLV/combined MERRA-2 + HMS grid +
    EDGAR v8.1 annual emissions (OPTIONAL, warn-and-continue) + the
    external_paths.json registry. Non-tx domains also run
    fetch_geoscf_domain (OPTIONAL, never REQUIRED): tx keeps registering
    the committed v1 Texas GEOS-CF parquet instead, so the shipped v2
    run's fetch sequence stays byte-identical. EDGAR runs for EVERY domain
    (it is a new v4 feature source with a new registry key no shipped
    consumer reads), and on an already-complete tx setup the stage
    sentinel above still short-circuits first, so nothing about the
    frozen run re-executes without FORCE=1."""
    sentinel = config2.artifact("external_paths.json")
    have_aqs = _best_glob(
        [os.path.join(config2.DATA_DIR, config2.AQS_STEM + "_*.parquet")])
    if (os.path.exists(sentinel) and have_aqs
            and os.environ.get("FORCE") != "1"):
        _say(f"{sentinel} + aqs v2 parquet exist (FORCE=1 to re-run) -- skip")
        return 0
    _banner("data")
    t0 = time.time()
    if quick:
        start, end = QUICK_START, QUICK_END
        years = years or QUICK_AQS_YEARS
        _say(f"quick mode: window {start}..{end}, AQS years {years}")
    else:
        start = start or config2.DATE_START
        end = end or config2.DATE_END
        years = years or AQS_YEARS_FULL

    fetch_aqs_v2(years=years)          # REQUIRED: exceptions propagate

    optional = [("merra2-slv", lambda: fetch_merra2_slv(start, end)),
                ("merra2-combined",
                 lambda: fetch_merra2_combined(start, end)),
                ("hms", lambda: fetch_hms_grid(start, end)),
                ("edgar", fetch_edgar_domain)]   # static: window-independent
    if config2.DOMAIN != "tx":
        # tx deliberately absent: its 'geoscf' registry key resolves to the
        # committed v1 Texas parquet (write_external_paths candidates), and
        # a refetch here would only duplicate a frozen product.
        optional.append(("geoscf", lambda: fetch_geoscf_domain(start, end)))
    for label, fn in optional:
        try:
            if fn() is None:
                _say(f"{label}: unavailable -- downstream features stay NaN")
        except Exception as e:
            _say(f"{label}: SKIPPED -- {e}")

    write_external_paths()
    _say(f"── stage data done in {time.time() - t0:.1f}s")
    return 0


def run_statics(quick=False):
    dest = os.path.join(
        config2.PIPELINE_DIR,
        _dstem("static_covariates") + ("_quick" if quick else "")
        + ".parquet")
    if os.path.exists(dest) and os.environ.get("FORCE") != "1":
        _say(f"{dest} exists (FORCE=1 to rebuild) -- skip")
        return 0
    _banner("statics")
    t0 = time.time()
    fetch_statics(quick=quick, dest=dest)
    _say(f"── stage statics done in {time.time() - t0:.1f}s")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="AQNet v2 external-data fetchers (DESIGN section 12)")
    ap.add_argument("stage", choices=["data-pa", "data", "statics", "all"],
                    help="which stage to run")
    ap.add_argument("--quick", action="store_true",
                    help="3-month window, coarse statics lattice")
    ap.add_argument("--start", default=None,
                    help="first date YYYY-MM-DD (default config2.DATE_START)")
    ap.add_argument("--end", default=None,
                    help="last date YYYY-MM-DD, inclusive "
                         "(default config2.DATE_END)")
    ap.add_argument("--years", type=int, nargs="*", default=None,
                    help="AQS years (default 2021..2026; --quick: 2024)")
    args = ap.parse_args(argv)

    rc = 0
    if args.stage in ("data-pa", "all"):
        rc = run_data_pa() or rc
    if args.stage in ("data", "all"):
        rc = run_data(quick=args.quick, start=args.start, end=args.end,
                      years=args.years) or rc
    if args.stage in ("statics", "all"):
        rc = run_statics(quick=args.quick) or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
