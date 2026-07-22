"""Build daily gridded training tensors for the deep-learning PM2.5 track.

Assembles a (days x channels x H x W) stack on a regular 0.1-degree lat/lon
grid spanning the Texas census-tract extent, built entirely from files the
main pipeline already produces:

  aerosol      pipeline/airquality_by_cell.parquet    [aod, cams_pm25]
  smoke        pipeline/hms_smoke_by_sensor.parquet   [hms_smoke]  (0-3 tier)
  meteorology  pipeline/purpleair_full_dataset.parquet
               [temperature, humidity, pressure, wind_speed, precipitation]
               + pipeline/met_extra_by_cell.parquet   [shortwave, et0, cloud_cover]
  static       pipeline/elevations.json (tract elevations) [elevation]
  temporal     day-of-year sin/cos planes             [doy_sin, doy_cos]

Supervision is SPARSE: PurpleAir PM2.5 readings exist only at sensor pixels,
so the builder also emits (day, row, col, pm25, sensor_id) observation
records that train.py turns into target grids + masks for a masked loss.
Only in-Texas sensors (pipeline/sensor_tx_membership.csv, in_tx=True) are
used as supervision targets, mirroring the production training setup.

Gridding is deliberately simple: nearest-cell for the 0.5-degree by-cell
products and the sensor smoke tiers, inverse-distance weighting for the
scattered sensor meteorology and tract-centroid elevations. Distances are
computed in degree space, which is adequate at Texas latitudes for a
0.1-degree grid.

Run:
    python research/deeplearning/dataset.py --out research/deeplearning/cache/texas_grid.npz
"""
import os
import json
import math
import argparse

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ── Paths & constants ───────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PA_DATASET = os.path.join(ROOT, "pipeline", "purpleair_full_dataset.parquet")
AQ_BY_CELL = os.path.join(ROOT, "pipeline", "airquality_by_cell.parquet")
MET_EXTRA = os.path.join(ROOT, "pipeline", "met_extra_by_cell.parquet")
HMS_SMOKE = os.path.join(ROOT, "pipeline", "hms_smoke_by_sensor.parquet")
ELEVATIONS = os.path.join(ROOT, "pipeline", "elevations.json")
MEMBERSHIP = os.path.join(ROOT, "pipeline", "sensor_tx_membership.csv")
TRACT_LOOKUP = os.path.join(ROOT, "backend", "static", "tract_lookup.parquet")

# Source groups and their channel order. Every channel name below is a real
# column in the files above (temporal planes are derived from the date).
GROUP_CHANNELS = {
    "aerosol": ["aod", "cams_pm25"],
    "smoke": ["hms_smoke"],
    "meteorology": ["temperature", "humidity", "pressure", "wind_speed",
                    "precipitation", "shortwave", "et0", "cloud_cover"],
    "static": ["elevation"],
    "temporal": ["doy_sin", "doy_cos"],
}

MET_SENSOR_COLS = ["temperature", "humidity", "pressure", "wind_speed", "precipitation"]
MET_EXTRA_COLS = ["shortwave", "et0", "cloud_cover"]


# ── Grid construction ───────────────────────────────────────────────────────

def build_grid(grid_deg=0.1, pad_cells=1):
    """Regular lat/lon axes covering the Texas tract-centroid extent.

    Bounds come from backend/static/tract_lookup.parquet (6,896 tracts),
    snapped outward to grid_deg multiples and padded by pad_cells.
    Returns (lat_axis, lon_axis), both ascending.
    """
    tl = pd.read_parquet(TRACT_LOOKUP, columns=["lat", "lon"])
    g = grid_deg
    lat_lo = math.floor(float(tl["lat"].min()) / g - pad_cells) * g
    lat_hi = math.ceil(float(tl["lat"].max()) / g + pad_cells) * g
    lon_lo = math.floor(float(tl["lon"].min()) / g - pad_cells) * g
    lon_hi = math.ceil(float(tl["lon"].max()) / g + pad_cells) * g
    lat_axis = np.round(np.arange(lat_lo, lat_hi + g / 2, g), 6)
    lon_axis = np.round(np.arange(lon_lo, lon_hi + g / 2, g), 6)
    return lat_axis.astype(np.float64), lon_axis.astype(np.float64)


def _grid_points(lat_axis, lon_axis):
    """Flattened (N, 2) array of grid-cell centers as (lat, lon) rows."""
    glat, glon = np.meshgrid(lat_axis, lon_axis, indexing="ij")
    return np.column_stack([glat.ravel(), glon.ravel()])


# ── Interpolation helpers ───────────────────────────────────────────────────

def _idw(pt_lat, pt_lon, values, grid_pts, shape, k=8, power=2.0):
    """Inverse-distance-weighted interpolation of scattered points to grid.

    NaN values are dropped before interpolation; a fully-NaN input yields a
    NaN plane (filled later by fill_missing). k=1 degenerates to nearest.
    """
    values = np.asarray(values, dtype=np.float64)
    pt_lat = np.asarray(pt_lat, dtype=np.float64)
    pt_lon = np.asarray(pt_lon, dtype=np.float64)
    ok = np.isfinite(values)
    if not ok.any():
        return np.full(shape, np.nan, dtype=np.float32)
    tree = cKDTree(np.column_stack([pt_lat[ok], pt_lon[ok]]))
    k_eff = int(min(k, ok.sum()))
    dist, idx = tree.query(grid_pts, k=k_eff)
    if k_eff == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    w = 1.0 / np.maximum(dist, 1e-6) ** power
    est = (w * values[ok][idx]).sum(axis=1) / w.sum(axis=1)
    return est.reshape(shape).astype(np.float32)


def _nearest(pt_lat, pt_lon, values, grid_pts, shape):
    """Nearest-point gridding (IDW with k=1) — used for by-cell products and
    the ordinal smoke tiers, where averaging would blur category levels."""
    return _idw(pt_lat, pt_lon, values, grid_pts, shape, k=1)


# ── Per-source loaders ──────────────────────────────────────────────────────

def _load_by_date(path, columns):
    """Read a parquet, normalize its date column, and index frames by day."""
    df = pd.read_parquet(path, columns=columns)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return {d: sub for d, sub in df.groupby("date")}


def build_elevation_plane(grid_pts, shape):
    """Static elevation plane (meters) IDW-gridded from the 6,896 tract
    centroid elevations in pipeline/elevations.json ("tracts" section)."""
    with open(ELEVATIONS) as f:
        elev = json.load(f)["tracts"]
    tl = pd.read_parquet(TRACT_LOOKUP, columns=["GEOID", "lat", "lon"])
    tl["elevation"] = tl["GEOID"].astype(str).map(elev)
    tl = tl.dropna(subset=["elevation"])
    return _idw(tl["lat"].values, tl["lon"].values, tl["elevation"].values,
                grid_pts, shape, k=4)


def _texas_sensor_ids(pa):
    """Sensor ids allowed as supervision targets (in-Texas only, matching the
    production convention that border-state sensors are context, not targets)."""
    if os.path.exists(MEMBERSHIP):
        mem = pd.read_csv(MEMBERSHIP)
        in_tx = mem["in_tx"].astype(str).str.lower() == "true"
        return set(mem.loc[in_tx, "sensor_id"].astype("int64"))
    return set(int(s) for s in pa["sensor_id"].unique())


# ── Assembly ────────────────────────────────────────────────────────────────

def build_dataset(start=None, end=None, grid_deg=0.1, idw_k=8, idw_power=2.0,
                  verbose=True):
    """Build the full gridded stack plus sparse supervision records.

    Parameters
    ----------
    start, end : str or None
        Optional inclusive date bounds (e.g. "2022-08-03"); None = all days
        present in the PurpleAir dataset.
    grid_deg : float
        Grid resolution in degrees (0.1 default, ~11 km cells).
    idw_k, idw_power : int, float
        Neighbors and power for the IDW meteorology interpolation.

    Returns
    -------
    dict with keys:
        groups   {name: float32 (D, C, H, W)}  raw physical values, NaN where
                 a source has no data that day (fill with fill_missing)
        channels GROUP_CHANNELS copy
        lat, lon grid axes (ascending)
        dates    np.datetime64[ns] array, length D
        obs      {"day", "row", "col", "pm25", "sensor"} flat numpy arrays
        grid_deg float
    """
    lat_axis, lon_axis = build_grid(grid_deg)
    H, W = len(lat_axis), len(lon_axis)
    shape = (H, W)
    grid_pts = _grid_points(lat_axis, lon_axis)

    pa = pd.read_parquet(PA_DATASET, columns=[
        "sensor_id", "date", "pm25", "latitude", "longitude"] + MET_SENSOR_COLS)
    pa["date"] = pd.to_datetime(pa["date"]).dt.normalize()

    dates = np.sort(pa["date"].unique())
    if start is not None:
        dates = dates[dates >= np.datetime64(pd.Timestamp(start))]
    if end is not None:
        dates = dates[dates <= np.datetime64(pd.Timestamp(end))]
    D = len(dates)
    if D == 0:
        raise SystemExit("No days selected — check --start/--end against the dataset range.")
    if verbose:
        print(f"Grid {H}x{W} at {grid_deg} deg, {D} days "
              f"({pd.Timestamp(dates[0]).date()} .. {pd.Timestamp(dates[-1]).date()})")

    # Source frames indexed by day.
    aq_by_date = _load_by_date(AQ_BY_CELL, ["cell_lat", "cell_lon", "date", "aod", "cams_pm25"])
    mx_by_date = _load_by_date(MET_EXTRA, ["cell_lat", "cell_lon", "date"] + MET_EXTRA_COLS)
    pa_by_date = {d: sub for d, sub in pa.groupby("date")}

    hms = pd.read_parquet(HMS_SMOKE)
    hms["date"] = pd.to_datetime(hms["date"]).dt.normalize()
    hms["sensor_id"] = hms["sensor_id"].astype("float64").astype("int64")
    hms_by_date = {d: sub for d, sub in hms.groupby("date")}

    # Sensor sites (coordinates are constant per sensor).
    sensors = pa.drop_duplicates("sensor_id")[["sensor_id", "latitude", "longitude"]]
    sensors = sensors.reset_index(drop=True)
    sensor_pos = {int(s): i for i, s in enumerate(sensors["sensor_id"])}
    s_lat = sensors["latitude"].values
    s_lon = sensors["longitude"].values

    # Allocate channel groups.
    groups = {name: np.full((D, len(chs), H, W), np.nan, dtype=np.float32)
              for name, chs in GROUP_CHANNELS.items()}

    # Static: one elevation plane tiled over days.
    elev_plane = build_elevation_plane(grid_pts, shape)
    groups["static"][:] = elev_plane[None, None]

    # Daily channels.
    for i, d in enumerate(dates):
        # dates holds numpy datetime64; the *_by_date dicts are keyed by the
        # pd.Timestamp keys groupby produces. The two hash differently even
        # though they compare equal, so a raw .get(d) misses every day —
        # normalize once per iteration.
        day = pd.Timestamp(d)
        # aerosol: nearest 0.5-deg CAMS cell
        sub = aq_by_date.get(day)
        if sub is not None:
            c_lat = sub["cell_lat"].values
            c_lon = sub["cell_lon"].values
            groups["aerosol"][i, 0] = _nearest(c_lat, c_lon, sub["aod"].values, grid_pts, shape)
            groups["aerosol"][i, 1] = _nearest(c_lat, c_lon, sub["cams_pm25"].values, grid_pts, shape)

        # smoke: HMS tier per sensor (0 where no polygon), nearest-sensor grid
        smoke_vals = np.zeros(len(sensors), dtype=np.float64)
        sub = hms_by_date.get(day)
        if sub is not None:
            for sid, tier in zip(sub["sensor_id"], sub["hms_smoke"]):
                pos = sensor_pos.get(int(sid))
                if pos is not None:
                    smoke_vals[pos] = float(tier)
        groups["smoke"][i, 0] = _nearest(s_lat, s_lon, smoke_vals, grid_pts, shape)

        # meteorology: IDW from sensor-day readings ...
        sub = pa_by_date.get(day)
        if sub is not None:
            for j, col in enumerate(MET_SENSOR_COLS):
                groups["meteorology"][i, j] = _idw(
                    sub["latitude"].values, sub["longitude"].values,
                    sub[col].values, grid_pts, shape, k=idw_k, power=idw_power)
        # ... plus nearest 0.5-deg cell for the PBL-proxy extras
        sub = mx_by_date.get(day)
        if sub is not None:
            c_lat = sub["cell_lat"].values
            c_lon = sub["cell_lon"].values
            for j, col in enumerate(MET_EXTRA_COLS):
                groups["meteorology"][i, len(MET_SENSOR_COLS) + j] = _nearest(
                    c_lat, c_lon, sub[col].values.astype(np.float64), grid_pts, shape)

        # temporal: constant day-of-year sin/cos planes
        ang = 2.0 * math.pi * day.dayofyear / 365.25
        groups["temporal"][i, 0] = math.sin(ang)
        groups["temporal"][i, 1] = math.cos(ang)

        if verbose and (i + 1) % 200 == 0:
            print(f"  gridded {i + 1}/{D} days")

    # Sparse supervision: in-Texas sensor readings snapped to grid pixels.
    tx_ids = _texas_sensor_ids(pa)
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    o = pa[pa["pm25"].notna() & pa["sensor_id"].isin(tx_ids)].copy()
    o["day"] = o["date"].map(date_to_idx)
    o = o.dropna(subset=["day"])
    rows = np.rint((o["latitude"].values - lat_axis[0]) / grid_deg).astype(np.int64)
    cols = np.rint((o["longitude"].values - lon_axis[0]) / grid_deg).astype(np.int64)
    inb = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    obs = {
        "day": o["day"].values.astype(np.int64)[inb],
        "row": rows[inb],
        "col": cols[inb],
        "pm25": o["pm25"].values.astype(np.float32)[inb],
        "sensor": o["sensor_id"].values.astype(np.int64)[inb],
    }
    if verbose:
        n_sites = len(np.unique(obs["row"] * W + obs["col"]))
        print(f"Supervision: {len(obs['pm25']):,} readings from "
              f"{len(np.unique(obs['sensor']))} sensors at {n_sites} grid pixels")

    return {
        "groups": groups,
        "channels": {k: list(v) for k, v in GROUP_CHANNELS.items()},
        "lat": lat_axis,
        "lon": lon_axis,
        "dates": dates.astype("datetime64[ns]"),
        "obs": obs,
        "grid_deg": float(grid_deg),
    }


# ── Missing data & normalization ────────────────────────────────────────────

def fill_missing(groups, fill_values=None):
    """Replace NaN pixels in each channel, in place.

    With fill_values=None the per-channel mean over all finite pixels is used
    (and returned, so training can persist it in the checkpoint). Passing a
    stored {group: [values]} dict reuses the training-time fills at export
    time, avoiding train/serve skew. Returns the fill values actually used.
    """
    used = {}
    for name, arr in groups.items():
        chan_fills = []
        for c in range(arr.shape[1]):
            ch = arr[:, c]
            bad = ~np.isfinite(ch)
            if fill_values is not None:
                fv = float(fill_values[name][c])
            else:
                good = ch[~bad]
                fv = float(good.mean()) if good.size else 0.0
            if bad.any():
                ch[bad] = fv
            chan_fills.append(fv)
        used[name] = chan_fills
    return used


def compute_norm_stats(groups):
    """Per-channel mean/std over all days and pixels (std floored at 1e-6)."""
    stats = {}
    for name, arr in groups.items():
        mean = arr.mean(axis=(0, 2, 3), dtype=np.float64)
        std = arr.std(axis=(0, 2, 3), dtype=np.float64)
        std = np.maximum(std, 1e-6)
        stats[name] = {"mean": mean.tolist(), "std": std.tolist()}
    return stats


def apply_norm_stats(groups, stats):
    """Standardize each channel in place with the given mean/std stats."""
    for name, arr in groups.items():
        mean = np.asarray(stats[name]["mean"], dtype=np.float32)[None, :, None, None]
        std = np.asarray(stats[name]["std"], dtype=np.float32)[None, :, None, None]
        arr -= mean
        arr /= std


# ── Cache I/O ───────────────────────────────────────────────────────────────

def save_cache(data, path):
    """Write a built dataset to a compressed .npz cache."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "lat": data["lat"],
        "lon": data["lon"],
        "dates_ns": data["dates"].astype("datetime64[ns]").astype(np.int64),
        "grid_deg": np.float64(data["grid_deg"]),
        "channels_json": np.array(json.dumps(data["channels"])),
    }
    for key, arr in data["obs"].items():
        payload[f"obs_{key}"] = arr
    for name, arr in data["groups"].items():
        payload[f"group_{name}"] = arr
    np.savez_compressed(path, **payload)


def load_cache(path):
    """Load a dataset cache written by save_cache; mirrors build_dataset()."""
    z = np.load(path, allow_pickle=False)
    channels = json.loads(z["channels_json"].item())
    return {
        "groups": {name: z[f"group_{name}"] for name in channels},
        "channels": channels,
        "lat": z["lat"],
        "lon": z["lon"],
        "dates": z["dates_ns"].astype("datetime64[ns]"),
        "obs": {key: z[f"obs_{key}"] for key in ("day", "row", "col", "pm25", "sensor")},
        "grid_deg": float(z["grid_deg"]),
    }


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Build the gridded Texas dataset cache.")
    ap.add_argument("--out", default=os.path.join(here, "cache", "texas_grid.npz"),
                    help="output .npz cache path")
    ap.add_argument("--grid-deg", type=float, default=0.1, help="grid resolution (degrees)")
    ap.add_argument("--start", default=None, help="first date (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="last date (YYYY-MM-DD)")
    args = ap.parse_args()

    data = build_dataset(start=args.start, end=args.end, grid_deg=args.grid_deg)
    save_cache(data, args.out)
    n_ch = sum(len(v) for v in data["channels"].values())
    print(f"Saved {args.out}: {len(data['dates'])} days x {n_ch} channels x "
          f"{len(data['lat'])}x{len(data['lon'])} grid, "
          f"{len(data['obs']['pm25']):,} supervision readings")


if __name__ == "__main__":
    main()
