# GEOS-CF DODS time-axis repair — measured evidence

The GEOS-CF GrADS-DODS aggregation
(`opendap.nccs.nasa.gov/dods/gmao/geos-cf/assim/chm_tavg_1hr_g1440x721_v1`)
advertises `time: days since 1-1-1 00:00:0.0` with no calendar attribute.
Current xarray decodes this ~500 years off (cftime `DatetimeGregorian`
objects around year 1518) and `sel(time=pd.Timestamp)` raises
`cannot compare ... different calendars` — the repository's
`data_external._geoscf_month` therefore fails on every month, and no
repo-code reduction of this source can be produced for comparison. All
measurements below are from the calibration probes run on 2026-07-23
before any bulk fetch; the sidecar's full-record check is preserved in
its run log.

## Raw-axis probe (decode_times=False)

- `time` size 70,164; units `days since 1-1-1 00:00:0.0`; calendar absent
- `time[0]` = 736696.0208333 (i.e. integer day + 00:30), step exactly
  1/24 day everywhere except the glitches below
- **32 irregular steps**, all isolated single corrupted values confined
  to indices ≤ 243 (early January 2018 — outside the 2021–2026 study
  window). Examples: idx 39 (−23 h then +25 h), idx 42→43 (jump to
  2005-10-04 and back), idx 96→97 (jump to year 2121 and back), idx 157
  (fractional time 14:09). Each glitch sits between neighbors that
  continue a perfect hourly progression, so the true axis is regular
  hourly.
- Reconstruction: `time[i] = ANCHOR + i hours`.

## Anchor calibration (lag-correlation of daily Texas-mean pm25_rh35_gcc)

Candidate anchor A maps `time[0]` → 2018-01-03 00:30. Scans of the
GEOS-CF daily Texas mean against two independently-dated references
(lag = days added to anchor-A dates; top entries as measured):

**Summer window 2024-06-01..2024-07-31 (61 days) vs PurpleAir raw pm25:**

| lag | r |
|---|---|
| **−2 d** | **+0.7109** |
| −1 d | +0.6311 |
| −3 d | +0.6196 |
| +0 d | +0.5575 |

**Winter/spring window 2024-01-01..2024-06-30 (182 days):**
vs PurpleAir best −3 (r=+0.333) with −2 at +0.303; vs CAMS best **−2**
(r=+0.303); CAMS-vs-PurpleAir mutual lag **0** (r=+0.387) — the
references are mutually aligned, and an earlier −11 peak in the summer
CAMS scan is a synoptic alias.

Chosen anchor: **anchor A − 2 days → `time[0]` = 2018-01-01 00:30 UTC**,
which is exactly the documented GEOS-CF v1 assim stream start. Axis end
under this anchor: 2026-01-02 11:30 — the stream stops there, so
2026-02..2026-05 genuinely has no GEOS-CF data on the server.

## Post-fetch full-record sanity (preserved in the sidecar log)

After all 61 non-empty months were fetched, daily Texas-mean GEOS-CF vs
CAMS over the whole overlapping record (n = 1,248 days):

| lag | r |
|---|---|
| −1 d | +0.2194 |
| **+0 d** | **+0.2855** |
| +1 d | +0.2118 |

Lag 0 is the maximum, consistent with a correctly dated axis. (The
modest magnitude is expected — these are two different models' daily
Texas means across all seasons; the lag *ordering*, not the absolute r,
is the dating evidence.)

The sidecar wrote the repository's own cache-of-record chunk format
(`research/aqnet/cache/geoscf/geoscf_YYYYMM.parquet`), and the
repository's `fetch_geoscf_pm25` assembled the final window parquet from
those chunks unchanged: 4,423,760 cell-days, 2021-01-01..2026-01-02.
