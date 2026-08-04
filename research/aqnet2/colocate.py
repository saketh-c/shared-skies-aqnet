"""AQNet v2 colocation inventory (stage `colocate`).

Builds `colocation_pairs.parquet`: every (AQS site, PurpleAir sensor) pair
within 25 km, with the great-circle distance and the number of days on which
both units report. This table is the sole pairing source for the S1
Kennedy-O'Hagan calibration (calibrate.py) and the audit's colocation
inventory (DESIGN S0/S12). v1 had no such table -- the Barkjohn constants
were applied network-wide with no colocated fitting at all, which is the
root of the v1 target-miscalibration defect (DESIGN S0.1).

Why 25 km when the primary calibration radius is 10 km: the wider table is
cheap, lets calibrate.py run its 25 km sensitivity arm without re-scanning
the raw parquets, and gives the audit stage the full distance histogram it
needs for the pair-inventory check.

Deliberately fold-agnostic: vault exclusion is NOT applied here. The vault
airlock lives in calibrate.py (which consumes folds2.json), so that this
inventory stays a pure geometric fact about the committed data and cannot go
stale when folds are rebuilt.

Run from anywhere:
    python colocate.py [--quick] [--max-km 25]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

import config2

# ── Input paths (committed v1 data; see diag_data.md for schemas) ──────────
PA_PARQUET = os.path.join(config2.PIPELINE_DIR, "purpleair_full_dataset.parquet")
AQS_PARQUET = os.path.join(config2.V1_DIR, "data", "aqs_daily_tx.parquet")

MAX_PAIR_KM = 25.0
EARTH_RADIUS_KM = 6371.0088


# ── Geometry ────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Inputs broadcast (degrees, float64)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(a, dtype=np.float64))
                              for a in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


# ── Loaders (id normalization: sensor_id and site_id are always str) ───────

def load_sensor_index(pa_parquet=None):
    """Per-sensor location + observed-date table from the committed PA parquet.

    Returns a DataFrame [sensor_id(str), lat, lon] and a dict
    {sensor_id: np.ndarray of datetime64[D] observed dates}. Sensor
    coordinates are the per-sensor median of the reported latitude/longitude
    (they are static in the committed parquet; median is robust if a future
    refetch ever jitters them).
    """
    path = pa_parquet or PA_PARQUET
    pa = pd.read_parquet(path, columns=["sensor_id", "date",
                                        "latitude", "longitude"])
    pa["sensor_id"] = pa["sensor_id"].astype(str)
    pa["date"] = pd.to_datetime(pa["date"]).dt.normalize()
    loc = (pa.groupby("sensor_id", sort=True)[["latitude", "longitude"]]
             .median().reset_index()
             .rename(columns={"latitude": "lat", "longitude": "lon"}))
    dates = {sid: np.unique(g.to_numpy().astype("datetime64[D]"))
             for sid, g in pa.groupby("sensor_id", sort=True)["date"]}
    return loc, dates


def load_site_index(aqs_parquet=None):
    """Per-site location + observed-date table from the v1 AQS daily parquet.

    Returns a DataFrame [site_id(str), lat, lon] and {site_id: dates array}.
    """
    path = aqs_parquet or AQS_PARQUET
    aq = pd.read_parquet(path, columns=["site_id", "date", "lat", "lon"])
    aq["site_id"] = aq["site_id"].astype(str)
    aq["date"] = pd.to_datetime(aq["date"]).dt.normalize()
    loc = (aq.groupby("site_id", sort=True)[["lat", "lon"]]
             .median().reset_index())
    dates = {sid: np.unique(g.to_numpy().astype("datetime64[D]"))
             for sid, g in aq.groupby("site_id", sort=True)["date"]}
    return loc, dates


# ── Pair table ──────────────────────────────────────────────────────────────

def build_pairs(pa_parquet=None, aqs_parquet=None, max_dist_km=MAX_PAIR_KM):
    """All (site, sensor) pairs within max_dist_km, with shared-day counts.

    Returns a DataFrame [site_id, sensor_id, dist_km, n_shared_days], sorted
    by (site_id, dist_km). n_shared_days counts calendar days on which BOTH
    the AQS site and the PA sensor have an observation -- the number of
    usable pair-days before any QC.
    """
    site_loc, site_dates = load_site_index(aqs_parquet)
    sens_loc, sens_dates = load_sensor_index(pa_parquet)

    # 62 x 467 full distance matrix -- trivially small, no tree needed.
    d = haversine_km(site_loc["lat"].to_numpy()[:, None],
                     site_loc["lon"].to_numpy()[:, None],
                     sens_loc["lat"].to_numpy()[None, :],
                     sens_loc["lon"].to_numpy()[None, :])
    si, pi = np.nonzero(d <= max_dist_km)

    rows = []
    for a, b in zip(si, pi):
        site = site_loc["site_id"].iloc[a]
        sens = sens_loc["sensor_id"].iloc[b]
        shared = np.intersect1d(site_dates[site], sens_dates[sens],
                                assume_unique=True).size
        rows.append((site, sens, float(d[a, b]), int(shared)))

    pairs = pd.DataFrame(rows, columns=["site_id", "sensor_id",
                                        "dist_km", "n_shared_days"])
    pairs = pairs.sort_values(["site_id", "dist_km"]).reset_index(drop=True)
    return pairs


def pair_inventory(pairs):
    """Distance-band pair counts (the DESIGN S3 audit numbers)."""
    return {f"pairs_le_{int(km)}km": int((pairs["dist_km"] <= km).sum())
            for km in (1, 5, 10, 25)}


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description="AQNet v2 colocation pair table")
    ap.add_argument("--quick", action="store_true",
                    help="accepted for stage-CLI uniformity; the pair table "
                         "is window-independent so quick == full here")
    ap.add_argument("--max-km", type=float, default=MAX_PAIR_KM)
    ap.add_argument("--pa-parquet", default=None)
    ap.add_argument("--aqs-parquet", default=None)
    args = ap.parse_args(argv)

    dest = config2.artifact("colocation_pairs.parquet")
    if os.path.exists(dest) and os.environ.get("FORCE") != "1":
        print(f"[aqnet2] colocate: {dest} exists (FORCE=1 to rebuild) -- skip")
        return 0

    print("[aqnet2] ── stage: colocate ──")
    pairs = build_pairs(args.pa_parquet, args.aqs_parquet, args.max_km)
    tmp = dest + ".tmp"
    pairs.to_parquet(tmp, index=False)
    os.replace(tmp, dest)

    inv = pair_inventory(pairs)
    print(f"[aqnet2] colocate: wrote {len(pairs)} pairs -> {dest}")
    for k, v in inv.items():
        print(f"[aqnet2] colocate:   {k} = {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
