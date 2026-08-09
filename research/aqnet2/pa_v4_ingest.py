"""AQNet v4 PurpleAir archive ingestion (stage `data-pa-v4`).

Consumes the raw pa_v4 archive that fetch_pa_v4.py pulls once and keeps
forever (DATA_DIR/pa_v4/A/{sensor_index}.parquet 6-hour tier-A rows,
DATA_DIR/pa_v4/B/{sensor_index}.parquet daily tier-B rows, plus the
pa_selection.parquet sensor metadata table) and produces two
window-stamped, domain-stamped tables under DATA_DIR:

  pa_v4_daily   one row per (sensor, local calendar day) with the
                EPA/Barkjohn dual-channel QC applied per sensor-day:
                pa_cf1 = (pm2.5_cf_1_a + pm2.5_cf_1_b) / 2 and the day is
                EXCLUDED (qc_pass False, kept in the table for audit) when
                |A - B| > 5 ug/m3 AND the relative difference
                |A - B| / ((A + B) / 2) > 61% -- one branch alone never
                excludes, and a zero denominator is guarded (identical
                zero channels pass with relative difference 0). Rows with
                a cf_1 channel outside [0, 1000] or humidity outside
                [0, 100] are non-physical and drop before aggregation
                (NaN humidity is missingness, not a violation: it
                propagates to NaN pa_rh, never a drop). pa_atm carries the
                same A/B channel-mean rule applied to the atm columns.
                Tier A rows map to local days via a fixed offset of
                round(longitude / 15) hours and a valid day needs >= 3 of
                its 4 six-hour blocks (n_blocks recorded); tier B rows are
                daily aggregates already and pass through with
                n_blocks = 4 nominal and tz_approx = True (they are
                UTC-day aligned, not local-day aligned).
  pa_v4_pairs   tier-A QC-passing sensor-days inner-joined to same-day
                FRM observations: eligibility is the selection table's
                frm_km <= 10 km gate, a sensor pairs with EVERY AQS site
                within 10 km (the v2 pairing semantics: one sensor near
                two sites contributes pairs to both), and dist_km is the
                haversine distance to each paired site (it can differ
                from frm_km by metadata vintage; frm_km stays the
                registered gate).

A third, OPTIONAL final rides along when an hms_grid raster exists:

  hms_by_sensor_v4  daily HMS smoke density per v4-fleet sensor (nearest
                    raster cell within one cell pitch, dense tier-0 rows
                    over the raster's coverage window) -- the v4 analogue
                    of the committed v2 hms_smoke_by_sensor product,
                    which calibrate's v4 branch prefers when present.

All finals are registered by fetchers2.write_external_paths under the
NEW keys 'pa_v4_daily' / 'pa_v4_pairs' / 'hms_by_sensor_v4' -- like the
'edgar' precedent, no shipped consumer reads them, so their presence
changes no current pipeline behavior.

AQNET2_PA_SOURCE switch (pa_source()): default 'v2' changes NOTHING --
the tx domain and the shipped v3 behavior stay byte-identical. Under
'v4', three consumers route here instead of the committed v2 archive:
colocate.build_pairs derives the calibration pair table from pa_v4_pairs
(load_pairs_table), calibrate.load_pa_daily builds its sensor-day frame
from pa_v4_daily (load_daily_for_cal, true cf_1 channel means so the v2
ATM reconstruction policy retires), and frame2.load_pa_calibrated joins
sensor coordinates from pa_v4_daily (sensor_coords) -- so the calibrated
parquet feeding frame2's nbr_pacal covariate pool is built from, and
located by, the v4 tables end to end.

The heavy archive lives on the cluster; this module runs there. Locally
it is exercised by synthetic-fixture unit tests only
(tests/test_pa_v4_ingest.py).

Run from anywhere (resumable: outputs that already exist and cover the
window are skipped, FORCE=1 rebuilds):
    python pa_v4_ingest.py [--start YYYY-MM-DD --end YYYY-MM-DD]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

import config2

# Raw archive layout (fetch_pa_v4.py contract, verified on the cluster).
ARCHIVE_DIR = os.path.join(config2.DATA_DIR, "pa_v4")
SELECTION_PARQUET = os.path.join(ARCHIVE_DIR, "pa_selection.parquet")
RAW_COLUMNS = ["time_stamp", "humidity", "temperature", "pm2.5_atm_a",
               "pm2.5_atm_b", "pm2.5_cf_1_a", "pm2.5_cf_1_b",
               "sensor_index"]

# EPA/Barkjohn sensor-day channel-agreement QC: exclusion requires BOTH
# branches (absolute AND relative disagreement).
AB_MAX_DIFF_UGM3 = 5.0
AB_MAX_REL_DIFF = 0.61
CF1_MIN, CF1_MAX = 0.0, 1000.0
RH_MIN, RH_MAX = 0.0, 100.0

MIN_BLOCKS_A = 3            # tier A: >= 3 of 4 six-hour blocks make a day
N_BLOCKS_NOMINAL = 4        # tier B daily rows pass through at 4 nominal
PAIR_KM = 10.0              # selection frm_km eligibility gate for pairs
EARTH_RADIUS_KM = 6371.0088

PA_SOURCES = ("v2", "v4")

# Set (never cleared in-process) by load_pairs_table when a caller asks for
# a wider radius than the pairs product can contain -- the product only
# holds sensors gated at selection frm_km <= PAIR_KM, so a wider request
# cannot widen the pair set. Consumers may inspect it after a call.
PAIRS_GATE_EXCEEDED = False

DAILY_STEM = "pa_v4_daily"
PAIRS_STEM = "pa_v4_pairs"
HMS_BY_SENSOR_STEM = "hms_by_sensor_v4"
# hms_by_sensor_v4 join cap: a sensor maps to its nearest hms_grid cell
# only within ONE cell pitch per axis; farther means the raster never
# rasterized that neighborhood and no coverage is claimed.
HMS_CELL_PITCH_DEG = float(config2.GRID_DEG)
DAILY_COLUMNS = ["sensor_index", "lat", "lon", "date", "pa_cf1", "pa_atm",
                 "pa_rh", "pa_t", "n_blocks", "tier", "qc_pass", "tz_approx"]
PAIRS_COLUMNS = DAILY_COLUMNS + ["site_id", "frm_pm25", "dist_km"]


def _say(msg):
    print(f"[aqnet2] pa_v4_ingest: {msg}", flush=True)


def _atomic_parquet(df, dest):
    tmp = dest + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, dest)


def _dstem(base):
    """Domain-stamped filename stem (tx keeps bare names, DESIGN 12.2).

    Mirrors fetchers2._dstem; duplicated so the v2-frozen consumers that
    import this module for pa_source() never pull the fetchers2
    guarded-import chain (same precedent as the cf_1 constants fetchers2
    mirrors from calibrate).
    """
    return base if config2.DOMAIN == "tx" else f"{base}_{config2.DOMAIN}"


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Inputs broadcast (degrees, float64)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(a, dtype=np.float64))
                              for a in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def pa_source():
    """The active PurpleAir archive generation from AQNET2_PA_SOURCE.

    'v2' (the default) changes NOTHING: every consumer keeps the shipped
    read path byte-identically (the tx domain is FROZEN). 'v4' routes
    colocate.build_pairs, calibrate.load_pa_daily and frame2's calibrated
    coordinate join to the tables this module builds. An unknown value is
    a loud config error, never a silent default.
    """
    src = os.environ.get("AQNET2_PA_SOURCE", "v2").strip().lower()
    if src not in PA_SOURCES:
        raise SystemExit(f"[aqnet2] pa_v4_ingest: unknown AQNET2_PA_SOURCE "
                         f"{src!r} (known: {sorted(PA_SOURCES)})")
    return src


# -- QC + local-day aggregation ---------------------------------------------

def local_day_offset_hours(lon):
    """Fixed local-day UTC offset in whole hours: round(longitude / 15).

    A fixed solar offset (not a tz database) is deliberate: PurpleAir
    sensors carry no timezone metadata, and a sub-hour error at the day
    boundary moves at most one 6-hour block, which the 3-of-4
    completeness rule absorbs.
    """
    return int(np.round(float(lon) / 15.0))


def qc_screen_rows(raw):
    """Row-level non-physical screen (pre-aggregation).

    Keeps rows with BOTH cf_1 channels finite inside [0, 1000] and
    humidity inside [0, 100] when reported; NaN humidity survives (it is
    missingness and propagates to NaN pa_rh, never a drop). Returns
    (kept, n_dropped). Requiring both cf_1 channels is the point of the
    v4 archive: a single-channel day cannot take the A/B agreement test.
    """
    a = pd.to_numeric(raw["pm2.5_cf_1_a"], errors="coerce").to_numpy(np.float64)
    b = pd.to_numeric(raw["pm2.5_cf_1_b"], errors="coerce").to_numpy(np.float64)
    rh = pd.to_numeric(raw["humidity"], errors="coerce").to_numpy(np.float64)
    ok = (np.isfinite(a) & np.isfinite(b)
          & (a >= CF1_MIN) & (a <= CF1_MAX)
          & (b >= CF1_MIN) & (b <= CF1_MAX)
          & ~((rh < RH_MIN) | (rh > RH_MAX)))
    kept = raw[ok].reset_index(drop=True)
    return kept, int(len(raw) - len(kept))


def channel_day_qc(cf1_a, cf1_b):
    """Day-level A/B channel agreement (EPA/Barkjohn convention).

    A sensor-day is EXCLUDED only when |A - B| > AB_MAX_DIFF_UGM3 AND the
    relative difference |A - B| / ((A + B) / 2) > AB_MAX_REL_DIFF -- the
    AND is load-bearing: high-concentration days tolerate absolute
    disagreement, clean days tolerate relative disagreement. A zero
    denominator is guarded: it only occurs at A = B = 0 (channels are
    non-negative after the row screen), where the difference is 0 and the
    day passes. Returns (channel_mean, qc_pass ndarray of bool).
    """
    a = np.asarray(cf1_a, dtype=np.float64)
    b = np.asarray(cf1_b, dtype=np.float64)
    mean = (a + b) / 2.0
    diff = np.abs(a - b)
    rel = np.divide(diff, mean, out=np.zeros_like(diff),
                    where=(np.isfinite(mean)) & (mean > 0))
    qc = ~((diff > AB_MAX_DIFF_UGM3) & (rel > AB_MAX_REL_DIFF))
    qc &= np.isfinite(mean)
    return mean, qc


def _empty_days():
    return pd.DataFrame({
        "date": pd.Series(dtype="datetime64[ns]"),
        "pa_cf1": pd.Series(dtype=np.float64),
        "pa_atm": pd.Series(dtype=np.float64),
        "pa_rh": pd.Series(dtype=np.float64),
        "pa_t": pd.Series(dtype=np.float64),
        "n_blocks": pd.Series(dtype=np.int64),
        "qc_pass": pd.Series(dtype=bool),
        "tz_approx": pd.Series(dtype=bool),
    })


def _empty_daily():
    out = _empty_days()
    out.insert(0, "sensor_index", pd.Series(dtype=np.int64))
    out["lat"] = pd.Series(dtype=np.float64)
    out["lon"] = pd.Series(dtype=np.float64)
    out["tier"] = pd.Series(dtype=str)
    return out[DAILY_COLUMNS]


def _empty_pairs():
    out = _empty_daily()
    out["site_id"] = pd.Series(dtype=str)
    out["frm_pm25"] = pd.Series(dtype=np.float64)
    out["dist_km"] = pd.Series(dtype=np.float64)
    return out[PAIRS_COLUMNS]


def sensor_days(raw, lon, tier):
    """One sensor's archive rows -> QC'd local-calendar-day aggregates.

    Tier A (6-hour rows): rows map to local days via the fixed
    round(lon / 15) hour offset; a valid day needs >= MIN_BLOCKS_A of its
    4 blocks (invalid days never become rows). Tier B rows are daily
    UTC-day aggregates already: passthrough with n_blocks = 4 nominal and
    tz_approx = True. The A/B channel QC applies at the day level in both
    tiers; qc_pass records it (failing days STAY in the table for audit,
    consumers filter).

    Returns [date, pa_cf1, pa_atm, pa_rh, pa_t, n_blocks, qc_pass,
    tz_approx] (possibly empty).
    """
    kept, _n_drop = qc_screen_rows(raw)
    if not len(kept):
        return _empty_days()
    ts = pd.to_datetime(pd.to_numeric(kept["time_stamp"]), unit="s")
    if tier == "A":
        local = ts + pd.Timedelta(hours=local_day_offset_hours(lon))
        day = local.dt.normalize()
        tz_approx = False
    else:
        day = ts.dt.normalize()
        tz_approx = True

    g = pd.DataFrame({
        "date": day.to_numpy(),
        "cf1_a": pd.to_numeric(kept["pm2.5_cf_1_a"],
                               errors="coerce").to_numpy(np.float64),
        "cf1_b": pd.to_numeric(kept["pm2.5_cf_1_b"],
                               errors="coerce").to_numpy(np.float64),
        "atm_a": pd.to_numeric(kept["pm2.5_atm_a"],
                               errors="coerce").to_numpy(np.float64),
        "atm_b": pd.to_numeric(kept["pm2.5_atm_b"],
                               errors="coerce").to_numpy(np.float64),
        "rh": pd.to_numeric(kept["humidity"],
                            errors="coerce").to_numpy(np.float64),
        "t": pd.to_numeric(kept["temperature"],
                           errors="coerce").to_numpy(np.float64),
        "ts": pd.to_numeric(kept["time_stamp"]).to_numpy(np.int64),
    }).groupby("date")
    agg = g.agg(cf1_a=("cf1_a", "mean"), cf1_b=("cf1_b", "mean"),
                atm_a=("atm_a", "mean"), atm_b=("atm_b", "mean"),
                pa_rh=("rh", "mean"), pa_t=("t", "mean"),
                n_blocks=("ts", "nunique")).reset_index()
    if tier == "A":
        agg = agg[agg["n_blocks"] >= MIN_BLOCKS_A].reset_index(drop=True)
    else:
        agg["n_blocks"] = N_BLOCKS_NOMINAL

    cf1_mean, qc = channel_day_qc(agg["cf1_a"], agg["cf1_b"])
    atm_mean = (agg["atm_a"].to_numpy(np.float64)
                + agg["atm_b"].to_numpy(np.float64)) / 2.0
    return pd.DataFrame({
        "date": pd.to_datetime(agg["date"]).astype("datetime64[ns]"),
        "pa_cf1": cf1_mean,
        "pa_atm": atm_mean,
        "pa_rh": agg["pa_rh"].to_numpy(np.float64),
        "pa_t": agg["pa_t"].to_numpy(np.float64),
        "n_blocks": agg["n_blocks"].astype(np.int64),
        "qc_pass": qc,
        "tz_approx": bool(tz_approx),
    })


# -- Table builders ---------------------------------------------------------

def load_selection(selection_path=None):
    """pa_selection.parquet with the columns this module relies on checked."""
    path = selection_path or SELECTION_PARQUET
    sel = pd.read_parquet(path)
    need = {"sensor_index", "latitude", "longitude", "frm_km"}
    missing = need - set(sel.columns)
    if missing:
        raise ValueError(f"selection table {path} missing columns "
                         f"{sorted(missing)}")
    sel = sel.copy()
    sel["sensor_index"] = sel["sensor_index"].astype(np.int64)
    return sel


def build_daily(archive_dir=None, selection=None, start=None, end=None):
    """Assemble the QC'd local-day table over every archived sensor.

    Sensors whose archive file is absent (fetch still in progress on the
    cluster) or zero-row (fetch_pa_v4's no-data completion marker) are
    counted and skipped, never an error: ingest is resumable over a
    partially-pulled archive and simply covers less until the pull
    finishes.
    """
    archive_dir = archive_dir or ARCHIVE_DIR
    if selection is None:
        selection = load_selection()
    t_lo = pd.Timestamp(start or config2.DATE_START)
    t_hi = pd.Timestamp(end or config2.DATE_END)

    frames = []
    n_absent = n_empty = 0
    for i, srow in enumerate(selection.itertuples(index=False), 1):
        si = int(srow.sensor_index)
        # A sensor's tier is which archive subdirectory holds its file:
        # the fetcher assigned tiers at run time and the selection table's
        # tier column predates that split ('hourly'/'daily' scoping labels),
        # so disk is the only authoritative record.
        tier = path = None
        for t in ("A", "B"):
            p = os.path.join(archive_dir, t, f"{si}.parquet")
            if os.path.exists(p):
                tier, path = t, p
                break
        if path is None:
            n_absent += 1
            continue
        raw = pd.read_parquet(path)
        if not len(raw):
            n_empty += 1
            continue
        days = sensor_days(raw, float(srow.longitude), tier)
        days = days[(days["date"] >= t_lo) & (days["date"] <= t_hi)]
        if not len(days):
            continue
        days = days.copy()
        days.insert(0, "sensor_index", np.int64(si))
        days["lat"] = float(srow.latitude)
        days["lon"] = float(srow.longitude)
        days["tier"] = tier
        frames.append(days)
        if i % 1000 == 0:
            _say(f"daily: {i:,}/{len(selection):,} sensors scanned "
                 f"({sum(len(f) for f in frames):,} sensor-days so far)")
    if n_absent or n_empty:
        _say(f"daily: skipped {n_absent:,} not-yet-fetched and {n_empty:,} "
             f"zero-row sensors (resume-safe; re-run after the pull)")
    if not frames:
        return _empty_daily()
    out = pd.concat(frames, ignore_index=True)[DAILY_COLUMNS]
    return (out.sort_values(["sensor_index", "date"], kind="mergesort")
               .reset_index(drop=True))


def build_pairs_table(daily, selection, aqs_daily):
    """Tier-A QC-passing sensor-days paired to same-day FRM observations.

    Eligibility is tier-A membership taken from the daily table (which
    derives it from the archive layout) plus the selection table's
    frm_km <= PAIR_KM gate; a sensor
    then pairs with EVERY AQS site within PAIR_KM (full haversine matrix;
    eligible sensors x sites is small), matching the v2 pairing semantics
    so the calibration audit and distance histogram stay comparable: one
    sensor near two sites contributes pairs to both, and dist_km is the
    distance to each paired site. Duplicate same-site same-day FRM
    observations (multi-POC) average to one truth value so a
    (sensor, site, day) row appears exactly once.
    """
    need = {"site_id", "date", "lat", "lon"}
    missing = need - set(aqs_daily.columns)
    if missing:
        raise ValueError(f"AQS daily table missing columns {sorted(missing)}")
    if "pm25_aqs" in aqs_daily.columns:
        frm_col = "pm25_aqs"
    elif "pm25" in aqs_daily.columns:
        frm_col = "pm25"
    else:
        raise ValueError("AQS daily table has neither pm25_aqs nor pm25")

    aq = aqs_daily[["site_id", "date", frm_col, "lat", "lon"]].copy()
    aq["site_id"] = aq["site_id"].astype(str)
    aq["date"] = pd.to_datetime(aq["date"]).dt.normalize()
    sites = aq.groupby("site_id")[["lat", "lon"]].median().reset_index()

    a_sensors = daily.loc[daily["tier"].astype(str) == "A", "sensor_index"]
    elig = selection[selection["sensor_index"].isin(set(a_sensors))
                     & (selection["frm_km"].astype(np.float64) <= PAIR_KM)]
    if not len(elig) or not len(sites) or not len(daily):
        return _empty_pairs()

    d = _haversine_km(elig["latitude"].to_numpy()[:, None],
                      elig["longitude"].to_numpy()[:, None],
                      sites["lat"].to_numpy()[None, :],
                      sites["lon"].to_numpy()[None, :])
    ei, sj = np.nonzero(d <= PAIR_KM)
    if not len(ei):
        return _empty_pairs()
    assign = pd.DataFrame({
        "sensor_index": elig["sensor_index"].to_numpy(np.int64)[ei],
        "site_id": sites["site_id"].to_numpy()[sj],
        "dist_km": d[ei, sj],
    })

    rows = daily[(daily["tier"].astype(str) == "A")
                 & daily["qc_pass"].astype(bool)
                 & daily["sensor_index"].isin(assign["sensor_index"])]
    if not len(rows):
        return _empty_pairs()
    rows = rows.merge(assign, on="sensor_index", how="inner")

    obs = (aq.dropna(subset=[frm_col])
             .rename(columns={frm_col: "frm_pm25"})
             .groupby(["site_id", "date"], as_index=False)["frm_pm25"].mean())
    pairs = rows.merge(obs, on=["site_id", "date"], how="inner")
    return (pairs[PAIRS_COLUMNS]
            .sort_values(["sensor_index", "date", "site_id"], kind="mergesort")
            .reset_index(drop=True))


def build_hms_by_sensor(coords, hms):
    """Daily HMS smoke density per v4 sensor from the gridded raster.

    coords: [sensor_id(str), lat, lon] (sensor_coords()). hms: the
    hms_grid product [lat|cell_lat, lon|cell_lon, date, hms_smoke], which
    holds rows ONLY for smoke-positive cells (fetchers2.fetch_hms_grid:
    inside the raster's coverage window a missing cell-day means no smoke
    polygon). Each sensor maps to its nearest raster cell, capped at ONE
    cell pitch per axis (HMS_CELL_PITCH_DEG): a sensor farther than that
    from every cell the raster ever marked sits outside the rasterized
    envelope and is left out of the table, so no coverage is claimed for
    it (calibrate keeps it NaN). The output is DENSE over the raster's
    coverage window -- one row per (covered sensor, day) with hms_smoke 0
    on no-polygon days -- so a consumer's left-join is NaN exactly where
    coverage genuinely ends, mirroring the committed v2 by-sensor
    product's semantics for the v4 fleet.

    Returns [sensor_id(str), date, hms_smoke(int8)] sorted by
    (sensor_id, date), empty when no sensor maps or the raster is empty.
    """
    hms = hms.rename(columns={"cell_lat": "lat", "cell_lon": "lon"})
    need = {"lat", "lon", "date", "hms_smoke"}
    missing = need - set(hms.columns)
    if missing:
        raise ValueError(f"hms_grid product missing columns "
                         f"{sorted(missing)}")
    empty = pd.DataFrame({"sensor_id": pd.Series(dtype=str),
                          "date": pd.Series(dtype="datetime64[ns]"),
                          "hms_smoke": pd.Series(dtype=np.int8)})
    if not len(hms) or not len(coords):
        return empty
    hms = hms.copy()
    hms["date"] = pd.to_datetime(hms["date"]).dt.normalize()

    # Nearest raster cell per sensor, capped at one pitch per axis. Lazy
    # scipy import: this module stays importable by the pa_source()-only
    # consumers without the scipy dependency chain.
    from scipy.spatial import cKDTree
    cells = hms[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    tree = cKDTree(cells[["lat", "lon"]].to_numpy(dtype=np.float64))
    q = np.column_stack([coords["lat"].to_numpy(dtype=np.float64),
                         coords["lon"].to_numpy(dtype=np.float64)])
    _, idx = tree.query(q, k=1)
    clat = cells["lat"].to_numpy()[idx]
    clon = cells["lon"].to_numpy()[idx]
    tol = HMS_CELL_PITCH_DEG + 1e-6
    within = ((np.abs(clat - q[:, 0]) <= tol)
              & (np.abs(clon - q[:, 1]) <= tol))
    n_out = int((~within).sum())
    if n_out:
        _say(f"hms_by_sensor: {n_out:,} sensors beyond one cell pitch of "
             f"the raster -- left uncovered (no coverage claim)")
    if not within.any():
        return empty
    assign = pd.DataFrame({
        "sensor_id": coords["sensor_id"].astype(str).to_numpy()[within],
        "lat": clat[within],
        "lon": clon[within],
    })

    # Smoke-positive sensor-days at the assigned cells, then a dense fill
    # over the raster's coverage window (missing cell-day = tier 0).
    sm = assign.merge(hms[["lat", "lon", "date", "hms_smoke"]]
                      .drop_duplicates(["lat", "lon", "date"]),
                      on=["lat", "lon"], how="inner")
    days = pd.date_range(hms["date"].min(), hms["date"].max(), freq="D")
    sids = assign["sensor_id"].to_numpy()
    out = pd.DataFrame({
        "sensor_id": np.repeat(sids, len(days)),
        "date": np.tile(days.to_numpy(), len(sids)),
    })
    out = out.merge(sm[["sensor_id", "date", "hms_smoke"]],
                    on=["sensor_id", "date"], how="left")
    out["hms_smoke"] = (out["hms_smoke"].fillna(0)
                        .astype(np.float64).astype(np.int8))
    return (out.sort_values(["sensor_id", "date"], kind="mergesort")
               .reset_index(drop=True))


# -- Output resolution (window-stamped finals + data-stage registry) --------

def _window_tag(start, end):
    return f"{pd.Timestamp(start):%Y%m%d}_{pd.Timestamp(end):%Y%m%d}"


def _covering_final(base, start, end, out_dir=None):
    """An existing final whose window stamp covers [start, end], or None.

    The stamp is authoritative (the aqs window-stamp precedent): a
    narrower re-run must never mask a wider product, and a wider product
    legitimately serves a narrower request.
    """
    root = out_dir or config2.DATA_DIR
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    best = None
    pat = os.path.join(root, _dstem(base) + "_[0-9]*_[0-9]*.parquet")
    for p in sorted(glob.glob(pat)):
        toks = os.path.splitext(os.path.basename(p))[0].split("_")
        try:
            f_lo, f_hi = pd.Timestamp(toks[-2]), pd.Timestamp(toks[-1])
        except (ValueError, IndexError):
            continue
        if f_lo <= lo and f_hi >= hi:
            best = p
    return best


def _widest_final(base):
    """Widest-window (then newest) final under DATA_DIR, or None (the
    fetchers2._best_glob ordering, mirrored for the same no-import
    reason as _dstem)."""
    pat = os.path.join(config2.DATA_DIR, _dstem(base) + "_[0-9]*.parquet")

    def _key(p):
        toks = os.path.splitext(os.path.basename(p))[0].split("_")
        try:
            span = (pd.Timestamp(toks[-1]) - pd.Timestamp(toks[-2])).days
        except (ValueError, IndexError):
            span = -1
        return (span, os.path.getmtime(p))

    hits = sorted(glob.glob(pat), key=_key, reverse=True)
    return hits[0] if hits else None


def _registry():
    """external_paths.json as a dict ({} when the data stage has not run)."""
    try:
        with open(config2.artifact("external_paths.json")) as fh:
            return json.load(fh)
    except Exception:
        return {}


def daily_path(explicit=None):
    """Resolve the pa_v4_daily table: explicit arg (must exist), else the
    registry key, else the widest final under DATA_DIR; None when absent."""
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"pa_v4_daily table not found: {explicit}")
        return explicit
    p = _registry().get("pa_v4_daily")
    if p and os.path.exists(p):
        return p
    return _widest_final(DAILY_STEM)


def pairs_path(explicit=None):
    """Resolve the pa_v4_pairs table (same preference order as daily_path)."""
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"pa_v4_pairs table not found: {explicit}")
        return explicit
    p = _registry().get("pa_v4_pairs")
    if p and os.path.exists(p):
        return p
    return _widest_final(PAIRS_STEM)


def hms_by_sensor_path(explicit=None):
    """Resolve the hms_by_sensor_v4 table (same preference order as
    daily_path); None when absent -- calibrate then falls back to the
    committed v2-fleet by-sensor product with its NaN accounting."""
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(
                f"hms_by_sensor_v4 table not found: {explicit}")
        return explicit
    p = _registry().get("hms_by_sensor_v4")
    if p and os.path.exists(p):
        return p
    return _widest_final(HMS_BY_SENSOR_STEM)


def _hms_grid_path():
    """The hms_grid raster to build hms_by_sensor_v4 from: the data-stage
    registry entry, else the widest domain-stamped final under DATA_DIR,
    else the committed v1 raster; None when nothing exists (hms_grid is an
    OPTIONAL source, so the builder skips, announced)."""
    p = _registry().get("hms_grid")
    if p and os.path.exists(p):
        return p
    p = _widest_final("hms_grid")
    if p:
        return p
    p = os.path.join(config2.PIPELINE_DIR, "hms_grid.parquet")
    return p if os.path.exists(p) else None


# -- Consumer-side loaders (the AQNET2_PA_SOURCE=v4 read path) --------------

def load_daily(path=None, qc_only=True, tiers=None, start=None, end=None):
    """pa_v4_daily rows, QC-passing by default.

    qc_only keeps only qc_pass sensor-days: QC failures exist in the
    table for audit, never in models. tiers optionally restricts to a
    subset of {"A", "B"}.

    Domain envelope gate: the archive's selection universe is built
    OUTSIDE this repo (fetch_pa_v4's pa_selection.parquet), so nothing
    upstream guarantees its sensors sit inside config2.TX_BBOX (the
    domain bbox); an out-of-envelope sensor would silently receive
    bbox-edge covariates from frame2's uncapped nearest-cell join and
    enter frames and folds as a legitimate unit. Every consumer reads
    through here, so the filter is applied once and announced, never
    silent. NaN coordinates fail the gate (a locationless sensor cannot
    be placed in any frame).
    """
    p = daily_path(path)
    if p is None:
        raise FileNotFoundError(
            "no pa_v4_daily table found (run `python pa_v4_ingest.py` on the "
            "archive host, or register 'pa_v4_daily' in external_paths.json)")
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    bb = config2.TX_BBOX
    in_bbox = (df["lat"].astype(np.float64)
                 .between(bb["lat_min"], bb["lat_max"])
               & df["lon"].astype(np.float64)
                 .between(bb["lon_min"], bb["lon_max"]))
    if not bool(in_bbox.all()):
        n_sens = int(df.loc[~in_bbox, "sensor_index"].nunique())
        _say(f"bbox: dropped {n_sens:,} sensors "
             f"({int((~in_bbox).sum()):,} sensor-days) outside the "
             f"{config2.DOMAIN} domain envelope")
        df = df[in_bbox]
    if qc_only:
        df = df[df["qc_pass"].astype(bool)]
    if tiers:
        df = df[df["tier"].astype(str).isin(set(tiers))]
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


def sensor_coords(path=None):
    """[sensor_id(str), lat, lon], one row per sensor, for frame2's
    calibrated-pool coordinate join (selection coordinates are constant
    per sensor, so first occurrence is exact, not approximate)."""
    df = load_daily(path, qc_only=False)
    coords = df.drop_duplicates("sensor_index")
    return pd.DataFrame({
        "sensor_id": coords["sensor_index"].astype(np.int64).astype(str),
        "lat": coords["lat"].astype(np.float64),
        "lon": coords["lon"].astype(np.float64),
    }).reset_index(drop=True)


def load_daily_for_cal(path=None, start=None, end=None):
    """The calibrate.load_pa_daily base frame from the v4 daily table.

    Columns [sensor_id(str), date, lat, lon, rh, t, pa_raw,
    channel_reconstructed, urban]: pa_raw is the TRUE dual-channel cf_1
    mean, so channel_reconstructed is identically 0.0 (the v2 ATM
    reconstruction policy retires under v4) and urban keeps the v2
    constant-0.0 degradation. calibrate adds dewpoint, harmonics, HMS and
    sensor age downstream, unchanged.
    """
    df = load_daily(path, qc_only=True, start=start, end=end)
    return pd.DataFrame({
        "sensor_id": df["sensor_index"].astype(np.int64).astype(str),
        "date": df["date"],
        "lat": df["lat"].astype(np.float64),
        "lon": df["lon"].astype(np.float64),
        "rh": df["pa_rh"].astype(np.float64),
        "t": df["pa_t"].astype(np.float64),
        "pa_raw": df["pa_cf1"].astype(np.float64),
        "channel_reconstructed": 0.0,
        "urban": 0.0,
    })


def load_pairs_table(max_dist_km=PAIR_KM, path=None):
    """colocate.build_pairs-shaped pair table from pa_v4_pairs.

    Returns [site_id(str), sensor_id(str), dist_km, n_shared_days] sorted
    by (site_id, dist_km) -- the exact schema calibrate consumes.
    n_shared_days counts QC-passing pair-days (the v4 table only holds
    those), a stricter count than the v2 pre-QC inventory by design.

    max_dist_km beyond PAIR_KM cannot widen the pair set (the product is
    gated at selection frm_km <= PAIR_KM and pairing distance <= PAIR_KM):
    such a request warns loudly and sets PAIRS_GATE_EXCEEDED so consumers
    never mistake the returned table for a genuinely wider inventory.
    """
    global PAIRS_GATE_EXCEEDED
    if float(max_dist_km) > PAIR_KM:
        PAIRS_GATE_EXCEEDED = True
        _say(f"WARNING: max_dist_km={float(max_dist_km):g} exceeds the "
             f"pairs product gate PAIR_KM={PAIR_KM:g} -- pa_v4_pairs only "
             f"holds sensors within {PAIR_KM:g} km of an AQS site, so the "
             f"wider request returns the same gated pair set")
    p = pairs_path(path)
    if p is None:
        raise FileNotFoundError(
            "no pa_v4_pairs table found (run `python pa_v4_ingest.py` on the "
            "archive host, or register 'pa_v4_pairs' in external_paths.json)")
    pr = pd.read_parquet(p, columns=["site_id", "sensor_index",
                                     "dist_km", "date"])
    pr = pr[pr["dist_km"].astype(np.float64) <= float(max_dist_km)]
    g = (pr.groupby(["site_id", "sensor_index"], as_index=False)
           .agg(dist_km=("dist_km", "first"),
                n_shared_days=("date", "nunique")))
    out = pd.DataFrame({
        "site_id": g["site_id"].astype(str),
        "sensor_id": g["sensor_index"].astype(np.int64).astype(str),
        "dist_km": g["dist_km"].astype(np.float64),
        "n_shared_days": g["n_shared_days"].astype(np.int64),
    })
    return (out.sort_values(["site_id", "dist_km"])
               .reset_index(drop=True))


# -- Stage runner + CLI -----------------------------------------------------

def run_ingest(start=None, end=None, archive_dir=None, selection_path=None,
               aqs_path=None, out_dir=None):
    """Build pa_v4_daily then pa_v4_pairs (plus, when an hms_grid raster
    exists, hms_by_sensor_v4), resumably.

    An existing final whose window stamp covers [start, end] is reused
    (FORCE=1 rebuilds); pairs reuse a freshly-skipped daily. Returns
    (daily_final_path, pairs_final_path); the hms side product resolves
    via hms_by_sensor_path().
    """
    start = start or config2.DATE_START
    end = end or config2.DATE_END
    root = out_dir or config2.DATA_DIR
    force = os.environ.get("FORCE") == "1"
    tag = _window_tag(start, end)

    selection = None

    def _sel():
        nonlocal selection
        if selection is None:
            selection = load_selection(selection_path)
        return selection

    dest_daily = os.path.join(root, f"{_dstem(DAILY_STEM)}_{tag}.parquet")
    daily = None
    have = None if force else _covering_final(DAILY_STEM, start, end, root)
    if have:
        _say(f"daily: {have} covers {start}..{end} "
             f"(FORCE=1 to rebuild) -- skip")
        dest_daily = have
    else:
        _say(f"daily: aggregating {len(_sel()):,} sensors from "
             f"{archive_dir or ARCHIVE_DIR} ({start}..{end})")
        daily = build_daily(archive_dir, _sel(), start, end)
        _atomic_parquet(daily, dest_daily)
        n_a = int((daily["tier"].astype(str) == "A").sum())
        _say(f"daily: wrote {len(daily):,} sensor-days (tier A {n_a:,} / "
             f"tier B {len(daily) - n_a:,}, qc_pass "
             f"{int(daily['qc_pass'].astype(bool).sum()):,}) -> {dest_daily}")

    dest_pairs = os.path.join(root, f"{_dstem(PAIRS_STEM)}_{tag}.parquet")
    have = None if force else _covering_final(PAIRS_STEM, start, end, root)
    if have:
        _say(f"pairs: {have} covers {start}..{end} "
             f"(FORCE=1 to rebuild) -- skip")
        dest_pairs = have
    else:
        if daily is None:
            daily = pd.read_parquet(dest_daily)
            daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
        # A reused daily final may cover a WIDER window than [start, end];
        # the pairs window stamp is authoritative (_covering_final), so the
        # frame is filtered to the request before pairing -- a pairs final
        # must never hold days outside its stamp.
        t_lo, t_hi = pd.Timestamp(start), pd.Timestamp(end)
        daily = daily[(daily["date"] >= t_lo) & (daily["date"] <= t_hi)]
        aqs_p = aqs_path or config2.canonical_aqs_path()
        if not aqs_p or not os.path.exists(aqs_p):
            raise FileNotFoundError(
                "no AQS daily parquet resolvable "
                "(config2.canonical_aqs_path) -- pairs need FRM truth; run "
                "the data stage first")
        pairs = build_pairs_table(daily, _sel(), pd.read_parquet(aqs_p))
        _atomic_parquet(pairs, dest_pairs)
        _say(f"pairs: wrote {len(pairs):,} pair-days "
             f"({pairs['sensor_index'].nunique()} sensors x "
             f"{pairs['site_id'].nunique()} sites) -> {dest_pairs}")

    # hms_by_sensor_v4: the v4-fleet analogue of the committed v2
    # hms_smoke_by_sensor product (which calibrate's v4 branch otherwise
    # falls back to, leaving the wider fleet NaN). OPTIONAL: hms_grid is
    # itself an optional source, so absence skips loudly, never errors.
    # The final is stamped with the window it actually covers (the request
    # intersected with the raster's coverage), so the covering-final rule
    # never mistakes a short raster for a full-window product.
    hms_grid_p = _hms_grid_path()
    if hms_grid_p is None:
        _say("hms_by_sensor: no hms_grid product found -- skipping "
             "(calibrate falls back to the committed v2-fleet table)")
        return dest_daily, dest_pairs
    hms = pd.read_parquet(hms_grid_p)
    hms = hms.rename(columns={"cell_lat": "lat", "cell_lon": "lon"})
    hms["date"] = pd.to_datetime(hms["date"]).dt.normalize()
    if not len(hms):
        _say(f"hms_by_sensor: {hms_grid_p} is empty -- skipping")
        return dest_daily, dest_pairs
    cov_lo = max(pd.Timestamp(start), hms["date"].min())
    cov_hi = min(pd.Timestamp(end), hms["date"].max())
    if cov_lo > cov_hi:
        _say(f"hms_by_sensor: {hms_grid_p} does not overlap "
             f"{start}..{end} -- skipping")
        return dest_daily, dest_pairs
    hms = hms[(hms["date"] >= cov_lo) & (hms["date"] <= cov_hi)]
    dest_hms = os.path.join(
        root, f"{_dstem(HMS_BY_SENSOR_STEM)}_{_window_tag(cov_lo, cov_hi)}"
              ".parquet")
    have = None if force else _covering_final(HMS_BY_SENSOR_STEM,
                                              cov_lo, cov_hi, root)
    if have:
        _say(f"hms_by_sensor: {have} covers {cov_lo:%Y-%m-%d}.."
             f"{cov_hi:%Y-%m-%d} (FORCE=1 to rebuild) -- skip")
    else:
        # sensor_coords routes through load_daily, so the domain bbox
        # gate applies to the covered fleet by construction.
        coords = sensor_coords(dest_daily)
        by_sensor = build_hms_by_sensor(coords, hms)
        _atomic_parquet(by_sensor, dest_hms)
        n_smoke = int((by_sensor["hms_smoke"] > 0).sum())
        _say(f"hms_by_sensor: wrote {len(by_sensor):,} sensor-days "
             f"({by_sensor['sensor_id'].nunique():,} sensors, "
             f"{n_smoke:,} smoke-positive) -> {dest_hms}")
    return dest_daily, dest_pairs


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="AQNet v4 PurpleAir archive ingestion "
                    "(QC + local-day aggregation + FRM pairs)")
    ap.add_argument("--start", default=None,
                    help="first date YYYY-MM-DD (default config2.DATE_START)")
    ap.add_argument("--end", default=None,
                    help="last date YYYY-MM-DD, inclusive "
                         "(default config2.DATE_END)")
    ap.add_argument("--archive-dir", default=None,
                    help=f"raw pa_v4 archive root (default {ARCHIVE_DIR})")
    ap.add_argument("--selection", default=None,
                    help=f"pa_selection.parquet (default {SELECTION_PARQUET})")
    ap.add_argument("--aqs-parquet", default=None,
                    help="AQS daily parquet (default canonical_aqs_path)")
    args = ap.parse_args(argv)

    run_ingest(start=args.start, end=args.end, archive_dir=args.archive_dir,
               selection_path=args.selection, aqs_path=args.aqs_parquet)

    # Refresh the data-stage registry so the NEW keys become visible to
    # consumers; inert to the shipped pipeline (nothing reads them under
    # AQNET2_PA_SOURCE=v2). Imported lazily: fetchers2's guarded imports
    # print degradation notices that do not belong in library importers.
    import fetchers2
    fetchers2.write_external_paths()
    return 0


if __name__ == "__main__":
    sys.exit(main())
