"""pa_v4_ingest unit tests -- synthetic fixtures only, no network, no repo
data (the real pa_v4 archive lives on the cluster and is unreachable from
this machine by design).

What is frozen here:

  * EPA/Barkjohn channel QC: the sensor-day exclusion is the AND of the
    absolute (> 5 ug/m3) and relative (> 61%) A/B disagreement branches;
    either branch alone never excludes, the zero-denominator case is
    guarded, and non-physical rows (cf_1 outside [0, 1000], humidity
    outside [0, 100]) drop before aggregation while NaN humidity survives
    as missingness.
  * Local-day aggregation: the fixed round(lon / 15) hour offset assigns
    rows to local calendar days (tested via a block that crosses the UTC
    day boundary), tier A days need >= 3 of 4 six-hour blocks, tier B
    rows pass through UTC-day aligned with n_blocks = 4 and tz_approx.
  * Pairs: tier-A QC-passing days only, the selection frm_km <= 10 km
    gate, EVERY AQS site within 10 km paired (v2 semantics: one sensor
    near two sites contributes pairs to both) with per-site haversine
    dist_km, and a same-day inner join against FRM observations. A pairs
    final never holds days outside its window stamp, even when rebuilt
    from a wider covering daily; load_pairs_table warns loudly (and sets
    PAIRS_GATE_EXCEEDED) when asked for a radius beyond the product gate.
  * AQNET2_PA_SOURCE: the default 'v2' is a no-op (the v4 loaders are
    provably never touched by colocate/calibrate/frame2), 'v4' routes all
    three consumers to the new tables, and unknown values fail loudly.
    The persisted colocation-pairs artifact is namespaced per source
    (colocate.pairs_artifact) so stale cross-source artifacts are never
    consumed; calibrate skips its 25 km sensitivity arm under v4 and
    zero-fills hms_smoke only for sensors the committed product covers.
  * Domain envelope: load_daily drops sensors outside config2.TX_BBOX
    (the domain bbox) with an announced count, never silently, so an
    out-of-domain archive sensor can reach no frame or fold;
    sensor_coords inherits the gate.
  * hms_by_sensor_v4: build_hms_by_sensor joins the v4 sensor
    coordinates to the hms_grid raster at the nearest cell within ONE
    cell pitch (farther sensors get no coverage claim) and emits dense
    tier-0 rows over the raster's coverage window; run_ingest writes and
    stamps the final; calibrate's v4 branch prefers it when present and
    falls back to the committed v2-fleet table's covered-sensors-only
    accounting when absent.
  * graph_res coordinate routing: under v4, load_raw_pa joins T2
    pretrain coordinates from pa_v4_ingest.sensor_coords (announcing
    rows dropped for missing coords), never from the committed v2
    parquet, so v4-only sensors survive into the station universe; the
    default v2 join is untouched.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import pytest

import calibrate
import colocate
import config2
import frame2
import graph_res
import pa_v4_ingest

RAW_COLS = ["time_stamp", "humidity", "temperature", "pm2.5_atm_a",
            "pm2.5_atm_b", "pm2.5_cf_1_a", "pm2.5_cf_1_b", "sensor_index"]


def _epoch(s):
    return int(pd.Timestamp(s).timestamp())


def _raw_frame(rows, sensor_index):
    """rows: (utc_ts_str, cf1_a, cf1_b, rh); atm channels are 0.9x cf1."""
    recs = [(_epoch(ts), rh, 25.0, 0.9 * a, 0.9 * b, a, b, sensor_index)
            for ts, a, b, rh in rows]
    return pd.DataFrame(recs, columns=RAW_COLS)


def _selection_frame(rows):
    """rows: (sensor_index, lat, lon, frm_km, tier)."""
    return pd.DataFrame({
        "sensor_index": np.array([r[0] for r in rows], dtype=np.int64),
        "date_created": _epoch("2021-01-01"),
        "last_seen": _epoch("2026-01-01"),
        "location_type": 0,
        "latitude": [r[1] for r in rows],
        "longitude": [r[2] for r in rows],
        "box": "tx",
        "frm_km": [float(r[3]) for r in rows],
        "days": 365,
        "tier": [r[4] for r in rows],
        "cell": "c0",
    })


# Sensor 1 (tier A, lon -97 -> offset -6): local day D spans UTC
# [D 06:00, D+1 06:00), so its four blocks sit at UTC D 06/12/18 and
# D+1 00:00.
_S1_ROWS = (
    # 2024-06-10: 4 blocks, last block (UTC 06-11 00:00) distinctive ->
    # proves the offset pulls it into the 06-10 local day (mean 12.5).
    [("2024-06-10 06:00", 10.0, 10.0, 40.0),
     ("2024-06-10 12:00", 10.0, 10.0, 40.0),
     ("2024-06-10 18:00", 10.0, 10.0, 40.0),
     ("2024-06-11 00:00", 20.0, 20.0, 40.0)]
    # 2024-06-11: 4 blocks failing BOTH QC branches (|2-20|=18 > 5 and
    # 18/11 = 164% > 61%) -> qc_pass False, stays in the table.
    + [("2024-06-11 06:00", 2.0, 20.0, 40.0),
       ("2024-06-11 12:00", 2.0, 20.0, 40.0),
       ("2024-06-11 18:00", 2.0, 20.0, 40.0),
       ("2024-06-12 00:00", 2.0, 20.0, 40.0)]
    # 2024-06-12: 3 of 4 blocks (valid), one NaN humidity (kept as
    # missingness; pa_rh means over the finite two).
    + [("2024-06-12 06:00", 8.0, 8.0, 40.0),
       ("2024-06-12 12:00", 8.0, 8.0, np.nan),
       ("2024-06-12 18:00", 8.0, 8.0, 40.0)]
    # 2024-06-13: only 2 blocks -> below MIN_BLOCKS_A, day never emitted.
    + [("2024-06-13 06:00", 8.0, 8.0, 40.0),
       ("2024-06-13 12:00", 8.0, 8.0, 40.0)]
)


@pytest.fixture
def synth_archive(tmp_path):
    """Tiny on-disk pa_v4 archive + selection + AQS parquets."""
    arch = tmp_path / "pa_v4"
    (arch / "A").mkdir(parents=True)
    (arch / "B").mkdir()

    _raw_frame(_S1_ROWS, 1).to_parquet(arch / "A" / "1.parquet", index=False)
    # Sensor 2: tier A but frm_km 20 -> daily yes, pairs no.
    s2 = [(f"2024-06-10 {h:02d}:00", 5.0, 5.0, 30.0) for h in (6, 12, 18)]
    s2.append(("2024-06-11 00:00", 5.0, 5.0, 30.0))
    _raw_frame(s2, 2).to_parquet(arch / "A" / "2.parquet", index=False)
    # Sensor 3: tier B daily rows (UTC-day aligned passthrough).
    s3 = [("2024-06-10 00:00", 7.0, 7.0, 50.0),
          ("2024-06-11 00:00", 7.0, 7.0, 50.0)]
    _raw_frame(s3, 3).to_parquet(arch / "B" / "3.parquet", index=False)
    # Sensor 4: zero-row archive file (fetch_pa_v4 completion marker).
    pd.DataFrame({c: pd.Series(dtype=np.float64) for c in RAW_COLS}) \
        .to_parquet(arch / "B" / "4.parquet", index=False)
    # Sensor 5: in the selection but not yet fetched (no file).

    sel = _selection_frame([
        (1, 30.0, -97.0, 5.0, "A"),
        (2, 30.5, -97.0, 20.0, "A"),
        (3, 30.2, -97.2, 1.0, "B"),
        (4, 30.3, -97.3, 50.0, "B"),
        (5, 30.4, -97.4, 50.0, "A"),
    ])
    sel_path = arch / "pa_selection.parquet"
    sel.to_parquet(sel_path, index=False)

    aqs = pd.DataFrame({
        "site_id": ["48_siteX", "48_siteX", "48_siteY"],
        "date": pd.to_datetime(["2024-06-10", "2024-06-11", "2024-06-10"]),
        "pm25_aqs": [9.5, 30.0, 50.0],
        "lat": [30.005, 30.005, 31.0],
        "lon": [-97.0, -97.0, -97.0],
    })
    aqs_path = tmp_path / "aqs_daily.parquet"
    aqs.to_parquet(aqs_path, index=False)

    return {"archive": str(arch), "selection": str(sel_path),
            "selection_df": sel, "aqs": str(aqs_path), "aqs_df": aqs,
            "tmp": tmp_path}


# ── QC rule ────────────────────────────────────────────────────────────────

def test_qc_exclusion_requires_both_branches():
    a = np.array([100.0, 1.0, 2.0, 0.0, 10.0])
    b = np.array([106.0, 4.0, 20.0, 0.0, 10.0])
    mean, qc = pa_v4_ingest.channel_day_qc(a, b)
    assert qc[0]        # abs branch alone (diff 6 > 5, rel ~5.8%) -> keep
    assert qc[1]        # rel branch alone (120% > 61%, diff 3 <= 5) -> keep
    assert not qc[2]    # both branches (diff 18, rel 164%) -> exclude
    assert qc[3]        # zero denominator guarded: identical zeros pass
    assert mean[3] == 0.0
    assert qc[4]        # identical channels trivially pass
    np.testing.assert_allclose(mean, (a + b) / 2.0)


def test_qc_screen_drops_nonphysical_rows_keeps_nan_humidity():
    rows = [
        ("2024-06-10 06:00", 10.0, 10.0, 40.0),     # good
        ("2024-06-10 12:00", 1500.0, 10.0, 40.0),   # cf1_a > 1000 -> drop
        ("2024-06-10 18:00", 10.0, -0.5, 40.0),     # cf1_b < 0 -> drop
        ("2024-06-11 00:00", 10.0, np.nan, 40.0),   # missing channel -> drop
        ("2024-06-11 06:00", 10.0, 10.0, 150.0),    # rh > 100 -> drop
        ("2024-06-11 12:00", 10.0, 10.0, -5.0),     # rh < 0 -> drop
        ("2024-06-11 18:00", 10.0, 10.0, np.nan),   # NaN rh -> KEPT
    ]
    kept, n_drop = pa_v4_ingest.qc_screen_rows(_raw_frame(rows, 9))
    assert n_drop == 5
    assert len(kept) == 2
    got_ts = set(kept["time_stamp"].astype(np.int64))
    assert got_ts == {_epoch("2024-06-10 06:00"), _epoch("2024-06-11 18:00")}


# ── Local-day aggregation ──────────────────────────────────────────────────

def test_local_day_offset_values():
    assert pa_v4_ingest.local_day_offset_hours(0.0) == 0
    assert pa_v4_ingest.local_day_offset_hours(-90.0) == -6
    assert pa_v4_ingest.local_day_offset_hours(-97.4) == -6
    assert pa_v4_ingest.local_day_offset_hours(-119.0) == -8
    assert pa_v4_ingest.local_day_offset_hours(141.0) == 9


def test_sensor_days_block_rule_and_day_assignment():
    days = pa_v4_ingest.sensor_days(_raw_frame(_S1_ROWS, 1), -97.0, "A")
    got = {pd.Timestamp(d).strftime("%Y-%m-%d"): r
           for d, r in zip(days["date"], days.to_dict("records"))}
    # 06-13 had only 2 of 4 blocks: never emitted.
    assert set(got) == {"2024-06-10", "2024-06-11", "2024-06-12"}

    d0 = got["2024-06-10"]
    assert d0["n_blocks"] == 4
    assert bool(d0["qc_pass"])
    # The UTC 06-11 00:00 block belongs to the 06-10 LOCAL day (offset -6):
    # mean of (10, 10, 10, 20) = 12.5, not 10.
    assert d0["pa_cf1"] == pytest.approx(12.5)
    assert d0["pa_atm"] == pytest.approx(0.9 * 12.5)
    assert not bool(d0["tz_approx"])

    d1 = got["2024-06-11"]
    assert d1["n_blocks"] == 4
    assert not bool(d1["qc_pass"])          # both QC branches fired
    assert d1["pa_cf1"] == pytest.approx(11.0)   # mean stays recorded

    d2 = got["2024-06-12"]
    assert d2["n_blocks"] == 3              # 3-of-4 rule: valid
    assert bool(d2["qc_pass"])
    assert d2["pa_rh"] == pytest.approx(40.0)    # NaN rh was missingness


def test_sensor_days_tier_b_passthrough_is_utc_aligned():
    rows = [("2024-06-10 00:00", 7.0, 7.0, 50.0),
            ("2024-06-11 00:00", 7.0, 7.0, 50.0)]
    days = pa_v4_ingest.sensor_days(_raw_frame(rows, 3), -97.2, "B")
    assert len(days) == 2
    # No local-day shift for tier B: dates stay the UTC days despite the
    # negative-longitude offset that would move midnight rows back a day.
    assert set(pd.to_datetime(days["date"]).dt.strftime("%Y-%m-%d")) \
        == {"2024-06-10", "2024-06-11"}
    assert set(days["n_blocks"]) == {pa_v4_ingest.N_BLOCKS_NOMINAL}
    assert days["tz_approx"].all()
    assert days["qc_pass"].all()


def test_build_daily_integration(synth_archive):
    daily = pa_v4_ingest.build_daily(synth_archive["archive"],
                                     synth_archive["selection_df"],
                                     "2024-06-01", "2024-06-30")
    assert list(daily.columns) == pa_v4_ingest.DAILY_COLUMNS
    # Sensors 4 (zero-row) and 5 (not fetched) skipped without error.
    assert set(daily["sensor_index"]) == {1, 2, 3}
    assert len(daily) == 6                      # 3 + 1 + 2 sensor-days
    s1 = daily[daily["sensor_index"] == 1]
    assert (s1["tier"] == "A").all()
    assert int(s1["qc_pass"].astype(bool).sum()) == 2
    s3 = daily[daily["sensor_index"] == 3]
    assert (s3["tier"] == "B").all()
    assert s3["tz_approx"].astype(bool).all()
    assert (daily.loc[daily["sensor_index"] == 1, "lat"] == 30.0).all()


def test_build_daily_ignores_selection_tier_labels(synth_archive):
    """Tier comes from the archive layout, never the selection column.

    The real pa_selection.parquet carries pre-redesign scoping labels
    ('hourly'/'daily') in its tier column; the fetcher assigned actual
    tiers at run time, recorded only as the A/ or B/ subdirectory. This
    reproduces the 2026-08-09 zero-row ingest: with those labels the old
    code found no A/hourly/*.parquet files and skipped every sensor.
    """
    sel = synth_archive["selection_df"].copy()
    sel["tier"] = ["hourly", "hourly", "daily", "daily", "hourly"]
    daily = pa_v4_ingest.build_daily(synth_archive["archive"], sel,
                                     "2024-06-01", "2024-06-30")
    assert set(daily["sensor_index"]) == {1, 2, 3}
    assert (daily.loc[daily["sensor_index"] == 1, "tier"] == "A").all()
    assert (daily.loc[daily["sensor_index"] == 3, "tier"] == "B").all()
    aqs = synth_archive["aqs_df"]
    pairs = pa_v4_ingest.build_pairs_table(daily, sel, aqs)
    assert len(pairs)
    assert set(pairs["sensor_index"]) == {1}


# ── Pairs ──────────────────────────────────────────────────────────────────

def test_build_pairs_join_correctness(synth_archive):
    daily = pa_v4_ingest.build_daily(synth_archive["archive"],
                                     synth_archive["selection_df"],
                                     "2024-06-01", "2024-06-30")
    pairs = pa_v4_ingest.build_pairs_table(daily,
                                           synth_archive["selection_df"],
                                           synth_archive["aqs_df"])
    assert list(pairs.columns) == pa_v4_ingest.PAIRS_COLUMNS
    # Only sensor 1 qualifies (tier A, frm_km 5 <= 10); of its QC-passing
    # days (06-10, 06-12) only 06-10 has a same-day FRM observation, and
    # the QC-failing 06-11 must not pair even though an observation exists.
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row["sensor_index"] == 1
    assert row["site_id"] == "48_siteX"          # siteY is ~111 km away
    assert pd.Timestamp(row["date"]) == pd.Timestamp("2024-06-10")
    assert row["frm_pm25"] == pytest.approx(9.5)
    assert row["pa_cf1"] == pytest.approx(12.5)
    expect_km = pa_v4_ingest._haversine_km(30.0, -97.0, 30.005, -97.0)
    assert row["dist_km"] == pytest.approx(expect_km)
    assert row["dist_km"] < 1.0


def test_build_pairs_all_sites_within_gate():
    """v2 pairing semantics: one sensor near two sites pairs with BOTH."""
    daily = pd.DataFrame({
        "sensor_index": np.array([1], dtype=np.int64),
        "lat": [30.0], "lon": [-97.0],
        "date": pd.to_datetime(["2024-06-10"]),
        "pa_cf1": [12.0], "pa_atm": [10.8],
        "pa_rh": [40.0], "pa_t": [25.0],
        "n_blocks": np.array([4], dtype=np.int64),
        "tier": ["A"], "qc_pass": [True], "tz_approx": [False],
    })[pa_v4_ingest.DAILY_COLUMNS]
    sel = _selection_frame([(1, 30.0, -97.0, 5.0, "A")])
    aqs = pd.DataFrame({
        "site_id": ["near", "mid", "far"],
        "date": pd.to_datetime(["2024-06-10"] * 3),
        "pm25_aqs": [9.5, 11.0, 50.0],
        "lat": [30.005, 30.05, 30.2],       # ~0.56 / ~5.6 / ~22 km north
        "lon": [-97.0, -97.0, -97.0],
    })
    pairs = pa_v4_ingest.build_pairs_table(daily, sel, aqs)
    # Both in-gate sites pair (never only the nearest); 'far' is outside.
    assert len(pairs) == 2
    assert set(pairs["site_id"]) == {"near", "mid"}
    by_site = pairs.set_index("site_id")
    for sid, slat in (("near", 30.005), ("mid", 30.05)):
        expect_km = pa_v4_ingest._haversine_km(30.0, -97.0, slat, -97.0)
        assert by_site.loc[sid, "dist_km"] == pytest.approx(expect_km)
        assert by_site.loc[sid, "pa_cf1"] == pytest.approx(12.0)
    assert by_site.loc["near", "frm_pm25"] == pytest.approx(9.5)
    assert by_site.loc["mid", "frm_pm25"] == pytest.approx(11.0)


def test_run_ingest_pairs_never_exceed_window_stamp(tmp_path, monkeypatch):
    """The review's exact scenario: a covering daily + a deleted pairs
    final + a narrower re-run must yield a pairs final whose content
    honors its own window stamp."""
    monkeypatch.delenv("FORCE", raising=False)
    arch = tmp_path / "pa_v4"
    (arch / "A").mkdir(parents=True)
    rows = ([(f"2024-06-10 {h:02d}:00", 10.0, 10.0, 40.0)
             for h in (6, 12, 18)]
            + [("2024-06-11 00:00", 10.0, 10.0, 40.0)]
            + [(f"2024-06-25 {h:02d}:00", 15.0, 15.0, 40.0)
               for h in (6, 12, 18)]
            + [("2024-06-26 00:00", 15.0, 15.0, 40.0)])
    _raw_frame(rows, 1).to_parquet(arch / "A" / "1.parquet", index=False)
    sel_path = arch / "pa_selection.parquet"
    _selection_frame([(1, 30.0, -97.0, 5.0, "A")]) \
        .to_parquet(sel_path, index=False)
    aqs_path = tmp_path / "aqs_daily.parquet"
    pd.DataFrame({
        "site_id": ["sX", "sX"],
        "date": pd.to_datetime(["2024-06-10", "2024-06-25"]),
        "pm25_aqs": [9.5, 12.0],
        "lat": [30.005, 30.005], "lon": [-97.0, -97.0],
    }).to_parquet(aqs_path, index=False)
    out = tmp_path / "data"
    out.mkdir()
    kw = dict(archive_dir=str(arch), selection_path=str(sel_path),
              aqs_path=str(aqs_path), out_dir=str(out))

    d1, p1 = pa_v4_ingest.run_ingest("2024-06-01", "2024-06-30", **kw)
    wide = pd.read_parquet(p1)
    assert set(pd.to_datetime(wide["date"]).dt.strftime("%Y-%m-%d")) \
        == {"2024-06-10", "2024-06-25"}

    # Interrupted run: the pairs final is gone, the covering daily remains.
    os.remove(p1)
    d2, p2 = pa_v4_ingest.run_ingest("2024-06-05", "2024-06-15", **kw)
    assert d2 == d1                              # covering daily reused
    assert os.path.basename(p2) == "pa_v4_pairs_20240605_20240615.parquet"
    narrow = pd.read_parquet(p2)
    dates = pd.to_datetime(narrow["date"])
    assert (dates >= pd.Timestamp("2024-06-05")).all()
    assert (dates <= pd.Timestamp("2024-06-15")).all()
    # The 06-25 pair-day of the wider daily must NOT leak into the
    # narrower stamp; the in-window 06-10 pair-day survives.
    assert set(dates.dt.strftime("%Y-%m-%d")) == {"2024-06-10"}


def test_load_pairs_table_warns_beyond_product_gate(tmp_path, monkeypatch,
                                                    capsys):
    pr = pd.DataFrame({
        "site_id": ["s1", "s1"],
        "sensor_index": np.array([1, 1], dtype=np.int64),
        "dist_km": [0.5, 0.5],
        "date": pd.to_datetime(["2024-06-10", "2024-06-11"]),
    })
    p = tmp_path / "pa_v4_pairs_20240601_20240630.parquet"
    pr.to_parquet(p, index=False)
    monkeypatch.setattr(pa_v4_ingest, "PAIRS_GATE_EXCEEDED", False)

    tbl10 = pa_v4_ingest.load_pairs_table(max_dist_km=10.0, path=str(p))
    assert pa_v4_ingest.PAIRS_GATE_EXCEEDED is False
    assert "WARNING" not in capsys.readouterr().out

    tbl25 = pa_v4_ingest.load_pairs_table(max_dist_km=25.0, path=str(p))
    assert pa_v4_ingest.PAIRS_GATE_EXCEEDED is True
    out = capsys.readouterr().out
    assert "WARNING" in out and "PAIR_KM=10" in out
    # The wider request cannot widen the gated pair set.
    pd.testing.assert_frame_equal(tbl25, tbl10)


# ── Resumable stage runner + loaders ───────────────────────────────────────

def test_run_ingest_resume_force_and_loaders(synth_archive, monkeypatch):
    out = synth_archive["tmp"] / "data"
    out.mkdir()
    monkeypatch.delenv("FORCE", raising=False)
    kw = dict(archive_dir=synth_archive["archive"],
              selection_path=synth_archive["selection"],
              aqs_path=synth_archive["aqs"], out_dir=str(out))

    d1, p1 = pa_v4_ingest.run_ingest("2024-06-01", "2024-06-30", **kw)
    assert os.path.basename(d1) == "pa_v4_daily_20240601_20240630.parquet"
    assert os.path.basename(p1) == "pa_v4_pairs_20240601_20240630.parquet"
    mt_d, mt_p = os.path.getmtime(d1), os.path.getmtime(p1)

    # Same window: both finals reused untouched.
    d2, p2 = pa_v4_ingest.run_ingest("2024-06-01", "2024-06-30", **kw)
    assert (d2, p2) == (d1, p1)
    assert os.path.getmtime(d2) == mt_d and os.path.getmtime(p2) == mt_p

    # Narrower window: covered by the existing stamp, still reused.
    d3, p3 = pa_v4_ingest.run_ingest("2024-06-05", "2024-06-20", **kw)
    assert (d3, p3) == (d1, p1)

    # FORCE=1 rebuilds in place.
    time.sleep(0.05)
    monkeypatch.setenv("FORCE", "1")
    d4, _p4 = pa_v4_ingest.run_ingest("2024-06-01", "2024-06-30", **kw)
    assert d4 == d1 and os.path.getmtime(d4) > mt_d

    # Consumer-side loaders against the finals (explicit paths, no registry).
    tbl = pa_v4_ingest.load_pairs_table(path=p1)
    assert list(tbl.columns) == ["site_id", "sensor_id", "dist_km",
                                 "n_shared_days"]
    assert len(tbl) == 1
    assert tbl.loc[0, "sensor_id"] == "1"
    assert tbl.loc[0, "n_shared_days"] == 1

    cal = pa_v4_ingest.load_daily_for_cal(path=d1)
    assert list(cal.columns) == ["sensor_id", "date", "lat", "lon", "rh",
                                 "t", "pa_raw", "channel_reconstructed",
                                 "urban"]
    assert (cal["channel_reconstructed"] == 0.0).all()
    assert len(cal) == 5                # the QC-failing sensor-day is out
    coords = pa_v4_ingest.sensor_coords(path=d1)
    assert set(coords["sensor_id"]) == {"1", "2", "3"}


# ── AQNET2_PA_SOURCE switch ────────────────────────────────────────────────

def _boom(*_a, **_k):
    raise AssertionError("v4 loader touched under AQNET2_PA_SOURCE=v2")


def test_pa_source_default_and_validation(monkeypatch):
    monkeypatch.delenv("AQNET2_PA_SOURCE", raising=False)
    assert pa_v4_ingest.pa_source() == "v2"
    monkeypatch.setenv("AQNET2_PA_SOURCE", "v2")
    assert pa_v4_ingest.pa_source() == "v2"
    monkeypatch.setenv("AQNET2_PA_SOURCE", "V4")
    assert pa_v4_ingest.pa_source() == "v4"
    monkeypatch.setenv("AQNET2_PA_SOURCE", "v5")
    with pytest.raises(SystemExit):
        pa_v4_ingest.pa_source()


def _committed_style_pa(tmp_path):
    """A committed-v2-shaped PA parquet (ATM-only, sensor_id/latitude)."""
    df = pd.DataFrame({
        "sensor_id": [42, 42],
        "date": pd.to_datetime(["2024-06-10", "2024-06-11"]),
        "pm25": [10.0, 25.0],
        "latitude": [30.0, 30.0],
        "longitude": [-97.0, -97.0],
        "humidity": [40.0, 45.0],
        "temperature": [25.0, 26.0],
    })
    p = tmp_path / "purpleair_committed.parquet"
    df.to_parquet(p, index=False)
    return str(p)


def test_v2_default_is_a_noop_for_all_consumers(tmp_path, monkeypatch):
    monkeypatch.delenv("AQNET2_PA_SOURCE", raising=False)
    monkeypatch.setattr(pa_v4_ingest, "load_pairs_table", _boom)
    monkeypatch.setattr(pa_v4_ingest, "load_daily_for_cal", _boom)
    monkeypatch.setattr(pa_v4_ingest, "sensor_coords", _boom)

    # colocate.build_pairs: default args take the geometric v2 path.
    site_loc = pd.DataFrame({"site_id": ["s1"], "lat": [30.0], "lon": [-97.0]})
    site_dates = {"s1": np.array(["2024-06-10", "2024-06-11"],
                                 dtype="datetime64[D]")}
    sens_loc = pd.DataFrame({"sensor_id": ["p1"], "lat": [30.003],
                             "lon": [-97.0]})
    sens_dates = {"p1": np.array(["2024-06-10", "2024-06-11"],
                                 dtype="datetime64[D]")}
    monkeypatch.setattr(colocate, "load_site_index",
                        lambda p=None: (site_loc, site_dates))
    monkeypatch.setattr(colocate, "load_sensor_index",
                        lambda p=None: (sens_loc, sens_dates))
    pairs = colocate.build_pairs()
    assert len(pairs) == 1
    assert pairs.loc[0, "n_shared_days"] == 2

    # calibrate.load_pa_daily: the shipped ATM/cf1-threshold policy holds,
    # whether the env is unset (default) or explicitly v2.
    pa_path = _committed_style_pa(tmp_path)
    for env in (None, "v2"):
        if env is None:
            monkeypatch.delenv("AQNET2_PA_SOURCE", raising=False)
        else:
            monkeypatch.setenv("AQNET2_PA_SOURCE", env)
        out = calibrate.load_pa_daily(pa_parquet=pa_path)
        assert list(out["pa_raw"]) == [10.0, 25.0]
        # ATM-only archive: rows >= 20 ug/m3 are channel-reconstructed.
        assert list(out["channel_reconstructed"]) == [0.0, 1.0]

    # An explicit pa_parquet wins even under v4 (the documented override).
    monkeypatch.setenv("AQNET2_PA_SOURCE", "v4")
    out = calibrate.load_pa_daily(pa_parquet=pa_path)
    assert list(out["channel_reconstructed"]) == [0.0, 1.0]
    monkeypatch.delenv("AQNET2_PA_SOURCE", raising=False)

    # frame2.load_pa_calibrated: the v2 coordinate join reads pa_daily.
    calp = tmp_path / "pa_calibrated.parquet"
    pd.DataFrame({"sensor_id": ["42"],
                  "date": pd.to_datetime(["2024-06-10"]),
                  "pa_cal_full": [9.0], "cal_var": [1.0]}) \
        .to_parquet(calp, index=False)
    df = frame2.load_pa_calibrated(str(calp),
                                   external_paths={"pa_daily": pa_path})
    assert df.loc[0, "lat"] == pytest.approx(30.0)
    assert df.loc[0, "lon"] == pytest.approx(-97.0)


def test_v4_switch_routes_all_three_consumers(tmp_path, monkeypatch):
    monkeypatch.setenv("AQNET2_PA_SOURCE", "v4")

    # colocate.build_pairs -> pa_v4_ingest.load_pairs_table.
    marker = pd.DataFrame({"site_id": ["m"], "sensor_id": ["77"],
                           "dist_km": [1.0], "n_shared_days": [3]})
    monkeypatch.setattr(pa_v4_ingest, "load_pairs_table",
                        lambda *a, **k: marker)
    assert colocate.build_pairs() is marker

    # calibrate.load_pa_daily -> load_daily_for_cal, then the shared
    # feature tail (dewpoint, harmonics, HMS tier, sensor age) unchanged.
    base = pd.DataFrame({
        "sensor_id": ["1", "1"],
        "date": pd.to_datetime(["2024-06-10", "2024-06-11"]),
        "lat": [30.0, 30.0], "lon": [-97.0, -97.0],
        "rh": [40.0, 50.0], "t": [25.0, 26.0],
        "pa_raw": [12.5, 8.0],
        "channel_reconstructed": [0.0, 0.0], "urban": [0.0, 0.0],
    })
    monkeypatch.setattr(pa_v4_ingest, "load_daily_for_cal",
                        lambda *a, **k: base.copy())
    out = calibrate.load_pa_daily()
    for col in ("dewpoint", "doy_sin", "doy_cos", "doy_sin2", "doy_cos2",
                "hms_smoke", "sensor_age_days"):
        assert col in out.columns, col
    assert list(out["pa_raw"]) == [12.5, 8.0]
    assert (out["channel_reconstructed"] == 0.0).all()
    assert list(out["sensor_age_days"]) == [0.0, 1.0]

    # frame2.load_pa_calibrated -> sensor_coords for the pool coordinates.
    monkeypatch.setattr(
        pa_v4_ingest, "sensor_coords",
        lambda p=None: pd.DataFrame({"sensor_id": ["42"], "lat": [31.5],
                                     "lon": [-99.0]}))
    calp = tmp_path / "pa_calibrated.parquet"
    pd.DataFrame({"sensor_id": ["42"],
                  "date": pd.to_datetime(["2024-06-10"]),
                  "pa_cal_full": [9.0], "cal_var": [1.0]}) \
        .to_parquet(calp, index=False)
    df = frame2.load_pa_calibrated(str(calp), external_paths={})
    assert df.loc[0, "lat"] == pytest.approx(31.5)
    assert df.loc[0, "lon"] == pytest.approx(-99.0)


# ── Source-namespaced colocation artifact + 25 km arm + HMS coverage ───────

def test_pairs_artifact_namespaced_by_pa_source(tmp_path, monkeypatch):
    monkeypatch.setattr(config2, "ARTIFACTS_DIR", str(tmp_path))
    pd.DataFrame({"site_id": ["v2s"], "sensor_id": ["1"],
                  "dist_km": [1.0], "n_shared_days": [2]}) \
        .to_parquet(tmp_path / "colocation_pairs.parquet", index=False)
    pd.DataFrame({"site_id": ["v4s"], "sensor_id": ["9"],
                  "dist_km": [0.5], "n_shared_days": [7]}) \
        .to_parquet(tmp_path / "colocation_pairs_v4.parquet", index=False)

    # With BOTH artifacts present each source resolves and reads its own.
    monkeypatch.delenv("AQNET2_PA_SOURCE", raising=False)
    assert os.path.basename(colocate.pairs_artifact()) \
        == "colocation_pairs.parquet"
    assert list(pd.read_parquet(colocate.pairs_artifact())["site_id"]) \
        == ["v2s"]
    monkeypatch.setenv("AQNET2_PA_SOURCE", "v4")
    assert os.path.basename(colocate.pairs_artifact()) \
        == "colocation_pairs_v4.parquet"
    assert list(pd.read_parquet(colocate.pairs_artifact())["site_id"]) \
        == ["v4s"]


def test_run_calibrate_reads_own_pairs_and_skips_25km_arm(tmp_path,
                                                          monkeypatch):
    """run_calibrate wiring: with BOTH pair artifacts present the stage
    consumes only the active source's table, and under v4 the 25 km
    sensitivity arm is skipped with the recorded reason (never a
    degenerate comparison of near-identical gated pair sets)."""
    monkeypatch.delenv("FORCE", raising=False)
    calls = []

    def fake_lolo(pairs, pa_daily=None, aqs_daily=None, folds=None, **kw):
        calls.append((float(kw.get("max_dist_km")),
                      tuple(sorted(set(pairs["site_id"])))))
        return {"max_dist_km": float(kw.get("max_dist_km")),
                "methods": {}, "by_year_bias": {}, "per_site": {},
                "g0": {"verdict": "pass", "criteria": {},
                       "fallback_form": "amt_rht",
                       "production_form": "learned"}}

    class _FakeModel:
        meta = {"kind": "fake", "n_pair_days": 1, "n_sites": 1,
                "excluded_sites": []}

    pa = pd.DataFrame({
        "sensor_id": ["1", "1"],
        "date": pd.to_datetime(["2024-06-10", "2024-06-11"]),
        "pa_raw": [12.0, 8.0],
        "channel_reconstructed": [0.0, 0.0],
    })
    aq = pd.DataFrame({"site_id": ["sX"],
                       "date": pd.to_datetime(["2024-06-10"]),
                       "pm25_aqs": [9.0], "lat": [30.0], "lon": [-97.0],
                       "is_fem": [0.0]})
    monkeypatch.setattr(calibrate, "load_fold_sites",
                        lambda p=None: {"vault_sites": [],
                                        "outer_fold_of_site": {}})
    monkeypatch.setattr(calibrate, "load_pa_daily",
                        lambda start=None, end=None: pa.copy())
    monkeypatch.setattr(calibrate, "load_aqs_daily",
                        lambda start=None, end=None: aq.copy())
    monkeypatch.setattr(calibrate, "dist_to_nearest_frm",
                        lambda *a, **k: np.zeros(2))
    monkeypatch.setattr(calibrate, "lolo_validate", fake_lolo)
    monkeypatch.setattr(calibrate, "fit_calibration",
                        lambda *a, **k: _FakeModel())
    monkeypatch.setattr(calibrate, "apply_calibration",
                        lambda m, q: (np.full(len(q), 9.0),
                                      np.full(len(q), 1.0)))

    for env, expect_site, expect_kms in (
            ("v4", "v4site", [calibrate.PRIMARY_PAIR_KM]),
            (None, "v2site", [calibrate.PRIMARY_PAIR_KM,
                              calibrate.SENSITIVITY_PAIR_KM])):
        art = tmp_path / f"artifacts_{env or 'v2'}"
        art.mkdir()
        monkeypatch.setattr(config2, "ARTIFACTS_DIR", str(art))
        pd.DataFrame({"site_id": ["v2site"], "sensor_id": ["1"],
                      "dist_km": [1.0], "n_shared_days": [2]}) \
            .to_parquet(art / "colocation_pairs.parquet", index=False)
        pd.DataFrame({"site_id": ["v4site"], "sensor_id": ["1"],
                      "dist_km": [0.5], "n_shared_days": [2]}) \
            .to_parquet(art / "colocation_pairs_v4.parquet", index=False)
        if env is None:
            monkeypatch.delenv("AQNET2_PA_SOURCE", raising=False)
        else:
            monkeypatch.setenv("AQNET2_PA_SOURCE", env)

        calls.clear()
        assert calibrate.run_calibrate(quick=False) == 0
        assert [s for _, s in calls] == [(expect_site,)] * len(expect_kms)
        assert [k for k, _ in calls] == expect_kms
        with open(os.path.join(str(art), "calibration_report.json"),
                  encoding="utf-8") as fh:
            rep = json.load(fh)
        if env == "v4":
            assert rep["sensitivity_25km"] == {
                "skipped": "pa_v4 pairs product is gated at 10 km"}
        else:
            assert rep["sensitivity_25km"]["max_dist_km"] \
                == calibrate.SENSITIVITY_PAIR_KM
    monkeypatch.delenv("AQNET2_PA_SOURCE", raising=False)


def test_sensitivity_25km_skip_record(monkeypatch):
    monkeypatch.delenv("AQNET2_PA_SOURCE", raising=False)
    assert calibrate.sensitivity_25km_skip() is None
    monkeypatch.setenv("AQNET2_PA_SOURCE", "v4")
    assert calibrate.sensitivity_25km_skip() == {
        "skipped": "pa_v4 pairs product is gated at 10 km"}


def test_v4_hms_zero_fill_only_for_covered_sensors(tmp_path, monkeypatch,
                                                   capsys):
    """The committed hms_smoke_by_sensor product spans the v2 fleet: under
    v4, sensor-days of sensors ABSENT from it stay NaN (announced), while
    covered sensors keep the tier-0 zero-fill; the v2 path is untouched."""
    monkeypatch.setenv("AQNET2_PA_SOURCE", "v4")
    base = pd.DataFrame({
        "sensor_id": ["1", "1", "2"],
        "date": pd.to_datetime(["2024-06-10", "2024-06-11", "2024-06-10"]),
        "lat": [30.0] * 3, "lon": [-97.0] * 3,
        "rh": [40.0, 50.0, 45.0], "t": [25.0, 26.0, 24.0],
        "pa_raw": [12.5, 8.0, 5.0],
        "channel_reconstructed": [0.0] * 3, "urban": [0.0] * 3,
    })
    monkeypatch.setattr(pa_v4_ingest, "load_daily_for_cal",
                        lambda *a, **k: base.copy())
    hpath = tmp_path / "hms_smoke_by_sensor.parquet"
    pd.DataFrame({"sensor_id": ["1"],
                  "date": pd.to_datetime(["2024-06-10"]),
                  "hms_smoke": [2.0]}).to_parquet(hpath, index=False)

    out = calibrate.load_pa_daily(hms_parquet=str(hpath))
    got = {(r["sensor_id"], pd.Timestamp(r["date"]).strftime("%m-%d")):
           r["hms_smoke"] for _, r in out.iterrows()}
    assert got[("1", "06-10")] == 2.0     # covered sensor, product value
    assert got[("1", "06-11")] == 0.0     # covered sensor, no-polygon day
    assert np.isnan(got[("2", "06-10")])  # sensor absent from the product
    assert "hms_smoke: 1 sensor-days left NaN" in capsys.readouterr().out

    # v2 path byte-identical: uncovered sensors still zero-fill.
    monkeypatch.delenv("AQNET2_PA_SOURCE", raising=False)
    pa_path = _committed_style_pa(tmp_path)
    out2 = calibrate.load_pa_daily(pa_parquet=pa_path,
                                   hms_parquet=str(hpath))
    assert (out2["hms_smoke"] == 0.0).all()


# ── Domain bbox gate ───────────────────────────────────────────────────────

def _daily_table(rows):
    """rows: (sensor_index, lat, lon, date_str) -> a DAILY_COLUMNS frame."""
    return pd.DataFrame({
        "sensor_index": np.array([r[0] for r in rows], dtype=np.int64),
        "lat": [r[1] for r in rows], "lon": [r[2] for r in rows],
        "date": pd.to_datetime([r[3] for r in rows]),
        "pa_cf1": 10.0, "pa_atm": 9.0, "pa_rh": 40.0, "pa_t": 25.0,
        "n_blocks": np.int64(4), "tier": "A",
        "qc_pass": True, "tz_approx": False,
    })[pa_v4_ingest.DAILY_COLUMNS]


def test_load_daily_bbox_gate_drops_out_of_domain_sensors(tmp_path, capsys):
    p = tmp_path / "pa_v4_daily_20240601_20240630.parquet"
    _daily_table([
        (1, 30.0, -97.0, "2024-06-10"),      # inside the tx bbox
        (2, 45.0, -120.0, "2024-06-10"),     # far outside (WA-ish)
        (2, 45.0, -120.0, "2024-06-11"),
        (3, 30.1, -80.0, "2024-06-10"),      # lat in, lon out -> out
    ]).to_parquet(p, index=False)

    df = pa_v4_ingest.load_daily(path=str(p))
    assert set(df["sensor_index"]) == {1}
    out = capsys.readouterr().out
    assert "bbox: dropped 2 sensors (3 sensor-days)" in out
    assert "domain envelope" in out

    # sensor_coords routes through load_daily and inherits the gate.
    coords = pa_v4_ingest.sensor_coords(path=str(p))
    assert set(coords["sensor_id"]) == {"1"}

    # An all-inside table stays silent (no spurious announcement).
    p2 = tmp_path / "pa_v4_daily_in_20240601_20240630.parquet"
    _daily_table([(1, 30.0, -97.0, "2024-06-10")]).to_parquet(p2, index=False)
    capsys.readouterr()
    df2 = pa_v4_ingest.load_daily(path=str(p2))
    assert len(df2) == 1
    assert "bbox" not in capsys.readouterr().out


# ── hms_by_sensor_v4 builder + calibrate preference ────────────────────────

def test_build_hms_by_sensor_pitch_cap_and_dense_coverage(capsys):
    coords = pd.DataFrame({
        "sensor_id": ["1", "2"],
        "lat": [30.02, 31.0],       # 1: within a pitch of cell (30, -97)
        "lon": [-97.03, -96.0],     # 2: > 1 deg from every raster cell
    })
    hms = pd.DataFrame({
        "cell_lat": [30.0, 32.0], "cell_lon": [-97.0, -95.0],
        "date": pd.to_datetime(["2024-06-10", "2024-06-12"]),
        "hms_smoke": np.array([2, 1], dtype=np.int8),
    })
    out = pa_v4_ingest.build_hms_by_sensor(coords, hms)
    # Only the in-pitch sensor is covered; its rows are DENSE over the
    # raster's coverage window with explicit tier-0 no-polygon days.
    assert set(out["sensor_id"]) == {"1"}
    got = {pd.Timestamp(d).strftime("%m-%d"): int(v)
           for d, v in zip(out["date"], out["hms_smoke"])}
    assert got == {"06-10": 2, "06-11": 0, "06-12": 0}
    assert "1 sensors beyond one cell pitch" in capsys.readouterr().out


def test_run_ingest_writes_hms_by_sensor_final(synth_archive, monkeypatch):
    out = synth_archive["tmp"] / "data_hms"
    out.mkdir()
    monkeypatch.delenv("FORCE", raising=False)
    hmsp = synth_archive["tmp"] / "hms_grid_synth.parquet"
    pd.DataFrame({"cell_lat": [30.0], "cell_lon": [-97.0],
                  "date": pd.to_datetime(["2024-06-10"]),
                  "hms_smoke": np.array([3], dtype=np.int8)}) \
        .to_parquet(hmsp, index=False)
    monkeypatch.setattr(pa_v4_ingest, "_hms_grid_path", lambda: str(hmsp))
    kw = dict(archive_dir=synth_archive["archive"],
              selection_path=synth_archive["selection"],
              aqs_path=synth_archive["aqs"], out_dir=str(out))

    pa_v4_ingest.run_ingest("2024-06-01", "2024-06-30", **kw)
    # Stamped with the window it ACTUALLY covers (the raster's), not the
    # request; resolvable through the standard path resolver.
    final = out / "hms_by_sensor_v4_20240610_20240610.parquet"
    assert final.exists()
    assert pa_v4_ingest.hms_by_sensor_path(str(final)) == str(final)
    tbl = pd.read_parquet(final)
    # Of sensors 1/2/3 only sensor 1 (30.0, -97.0) sits within one cell
    # pitch of the single raster cell; 2 (30.5) and 3 (30.2, -97.2) do not.
    assert set(tbl["sensor_id"]) == {"1"}
    assert tbl["hms_smoke"].tolist() == [3]

    # Resume: a covering final is reused untouched.
    mt = final.stat().st_mtime
    pa_v4_ingest.run_ingest("2024-06-01", "2024-06-30", **kw)
    assert final.stat().st_mtime == mt


def test_calibrate_prefers_v4_hms_product_with_fallback(tmp_path,
                                                        monkeypatch, capsys):
    monkeypatch.setenv("AQNET2_PA_SOURCE", "v4")
    base = pd.DataFrame({
        "sensor_id": ["1", "1", "1", "2"],
        "date": pd.to_datetime(["2024-06-10", "2024-06-11",
                                "2024-06-20", "2024-06-10"]),
        "lat": [30.0] * 4, "lon": [-97.0] * 4,
        "rh": [40.0] * 4, "t": [25.0] * 4,
        "pa_raw": [12.5, 8.0, 6.0, 5.0],
        "channel_reconstructed": [0.0] * 4, "urban": [0.0] * 4,
    })
    monkeypatch.setattr(pa_v4_ingest, "load_daily_for_cal",
                        lambda *a, **k: base.copy())
    # Dense v4 by-sensor product: sensor 1 covered on 06-10/06-11 only.
    v4p = tmp_path / "hms_by_sensor_v4_20240610_20240611.parquet"
    pd.DataFrame({"sensor_id": ["1", "1"],
                  "date": pd.to_datetime(["2024-06-10", "2024-06-11"]),
                  "hms_smoke": np.array([2, 0], dtype=np.int8)}) \
        .to_parquet(v4p, index=False)
    monkeypatch.setattr(pa_v4_ingest, "hms_by_sensor_path",
                        lambda explicit=None: str(v4p))

    out = calibrate.load_pa_daily()
    got = {(r["sensor_id"], pd.Timestamp(r["date"]).strftime("%m-%d")):
           r["hms_smoke"] for _, r in out.iterrows()}
    assert got[("1", "06-10")] == 2.0     # product value
    assert got[("1", "06-11")] == 0.0     # explicit tier-0 row, no fill
    assert np.isnan(got[("1", "06-20")])  # outside the coverage window
    assert np.isnan(got[("2", "06-10")])  # uncovered sensor
    txt = capsys.readouterr().out
    assert "v4 by-sensor product" in txt
    assert "2 sensor-days outside" in txt

    # Fallback (no v4 product): the committed covered-sensors-only
    # accounting, unchanged -- covered sensors zero-fill everywhere.
    monkeypatch.setattr(pa_v4_ingest, "hms_by_sensor_path",
                        lambda explicit=None: None)
    cpath = tmp_path / "hms_smoke_by_sensor.parquet"
    pd.DataFrame({"sensor_id": ["1"],
                  "date": pd.to_datetime(["2024-06-10"]),
                  "hms_smoke": [2.0]}).to_parquet(cpath, index=False)
    monkeypatch.setattr(calibrate, "HMS_PARQUET", str(cpath))
    out2 = calibrate.load_pa_daily()
    got2 = {(r["sensor_id"], pd.Timestamp(r["date"]).strftime("%m-%d")):
            r["hms_smoke"] for _, r in out2.iterrows()}
    assert got2[("1", "06-10")] == 2.0
    assert got2[("1", "06-11")] == 0.0
    assert got2[("1", "06-20")] == 0.0    # covered fill, v2-product rules
    assert np.isnan(got2[("2", "06-10")])
    assert "left NaN" in capsys.readouterr().out
    monkeypatch.delenv("AQNET2_PA_SOURCE", raising=False)


# ── graph_res T2 coordinate routing ────────────────────────────────────────

def test_graph_res_raw_pa_coords_route_by_source(tmp_path, monkeypatch,
                                                 capsys):
    calp = tmp_path / "pa_calibrated.parquet"
    pd.DataFrame({"sensor_id": ["42", "777", "999"],
                  "date": pd.to_datetime(["2024-06-10"] * 3),
                  "pa_raw": [10.0, 11.0, 12.0],
                  "pa_cal_full": [9.0] * 3}).to_parquet(calp, index=False)
    paths = {"pa_calibrated": str(calp),
             "pa_parquet": str(tmp_path / "does_not_exist.parquet")}

    # v4: coords come from sensor_coords; the committed parquet path does
    # not even exist, proving it is never consulted. The v4-only sensor
    # 999 SURVIVES; 777 (no coordinates anywhere) drops with a count.
    monkeypatch.setenv("AQNET2_PA_SOURCE", "v4")
    monkeypatch.setattr(
        pa_v4_ingest, "sensor_coords",
        lambda path=None: pd.DataFrame({"sensor_id": ["42", "999"],
                                        "lat": [30.0, 31.0],
                                        "lon": [-97.0, -96.0]}))
    df = graph_res.load_raw_pa(paths, "2024-06-01", "2024-06-30")
    assert set(df["sensor_id"]) == {"42", "999"}
    assert df.loc[df["sensor_id"] == "999", "lat"].iloc[0] \
        == pytest.approx(31.0)
    assert "dropping 1 in-window sensor-days" in capsys.readouterr().out

    # v2 default: the shipped committed-parquet join, byte-identical --
    # sensors absent from it (777, 999) drop with no v4 announcement.
    monkeypatch.delenv("AQNET2_PA_SOURCE", raising=False)
    paths["pa_parquet"] = _committed_style_pa(tmp_path)
    df2 = graph_res.load_raw_pa(paths, "2024-06-01", "2024-06-30")
    assert set(df2["sensor_id"]) == {"42"}
    assert "pa_v4_daily coordinates" not in capsys.readouterr().out
