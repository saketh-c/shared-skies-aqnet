"""Registered v4 neighbor-taper variant (EXPANSION.md, AQNET2_NBR_TAPER).

The frozen-domain contract is the anchor: with the switch off (the default)
every existing feature column is byte-identical, and with it on the tapered
means arrive ONLY as additional '_tp'-suffixed columns beside the hard-cutoff
nbr_pacal / nbr_frm blocks, never as mutations, so an ablation A/B run can
compare both variants inside one frame. The kernel is exponential distance
decay w = exp(-d / tau) with tau = radius / 2 (the registration leaves the
form open; frame2.taper_weights documents the choice), applied over the same
neighbor set as the hard-cutoff mean.

All synthetic pools (conftest / inline); no repo data files, no network.
"""
import numpy as np
import pandas as pd
import pytest

import config2
import frame2


def _query_frame():
    """30 queries over the conftest pool window: 20 inside the sensor
    cluster, 10 hundreds of km away (honest zero-coverage rows)."""
    dates = pd.date_range("2024-06-01", periods=10, freq="D")
    lats = np.concatenate([30.05 + 0.01 * np.arange(10),
                           30.10 + 0.01 * np.arange(10),
                           33.50 + 0.10 * np.arange(10)])
    lons = np.concatenate([np.full(10, -97.90),
                           np.full(10, -97.90),
                           np.full(10, -94.50)])
    return lats, lons, pd.DatetimeIndex(list(dates) * 3)


# ── (1) the switch: default off, validated values, loud unknowns ────────────

def test_switch_default_off_and_unknown_raises(monkeypatch):
    monkeypatch.delenv("AQNET2_NBR_TAPER", raising=False)
    assert frame2.nbr_taper_enabled() is False
    for v in ("0", "off", "false", ""):
        monkeypatch.setenv("AQNET2_NBR_TAPER", v)
        assert frame2.nbr_taper_enabled() is False
    for v in ("1", "on", "true"):
        monkeypatch.setenv("AQNET2_NBR_TAPER", v)
        assert frame2.nbr_taper_enabled() is True
    monkeypatch.setenv("AQNET2_NBR_TAPER", "banana")
    with pytest.raises(SystemExit):
        frame2.nbr_taper_enabled()


# ── (2) off = identical frames; on = ONLY additional '_tp' columns ──────────

def test_off_identical_on_adds_only_suffixed_columns(pools, monkeypatch):
    lats, lons, qdates = _query_frame()

    monkeypatch.delenv("AQNET2_NBR_TAPER", raising=False)
    f_off = frame2.build_point_features(lats, lons, qdates, pools)
    monkeypatch.setenv("AQNET2_NBR_TAPER", "1")
    f_on = frame2.build_point_features(lats, lons, qdates, pools)

    # Off: not a single taper column exists.
    assert not any(c.endswith(frame2.TAPER_SUFFIX) for c in f_off.columns)

    # On: the added columns are exactly the registered tapered means, one
    # per (block, radius, lag) beside the existing hard-cutoff blocks.
    added = [c for c in f_on.columns if c not in set(f_off.columns)]
    expected = ([f"nbr_pacal_{r}km_tp" for r in frame2.PA_RADII_KM]
                + [f"nbr_pacal_{r}km_lag{l}_tp" for l in frame2.PA_LAGS
                   for r in frame2.PA_RADII_KM]
                + [f"nbr_frm_{frame2.FRM_RADIUS_KM}km_lag{l}_tp"
                   for l in frame2.FRM_LAGS])
    assert sorted(added) == sorted(expected)

    # Every pre-existing column survives, in order, byte-identical.
    shared_in_on = [c for c in f_on.columns if c in set(f_off.columns)]
    assert shared_in_on == list(f_off.columns)
    for c in f_off.columns:
        a = f_off[c].to_numpy()
        b = f_on[c].to_numpy()
        if a.dtype.kind in "fiu" and b.dtype.kind in "fiu":
            assert np.array_equal(a.astype(np.float64), b.astype(np.float64),
                                  equal_nan=True), f"taper-on mutated {c}"
        else:
            assert list(a) == list(b), f"taper-on mutated {c}"

    # The tapered means are features in the interpolating set, and they are
    # NaN exactly where the (identical) neighbor set is empty.
    fcols = frame2.feature_columns(f_on)
    for tp_col in expected:
        assert tp_col in fcols
        assert config2.is_interp_feature(tp_col)
        avail_col = tp_col.replace("_tp", "").replace(
            "nbr_pacal_", "nbr_pacal_avail_").replace(
            "nbr_frm_", "nbr_frm_avail_")
        tp = f_on[tp_col].to_numpy(dtype=np.float64)
        av = f_on[avail_col].to_numpy(dtype=np.float64)
        assert (np.isnan(tp) == (av == 0.0)).all(), (
            f"{tp_col} NaN pattern diverges from {avail_col}")


def test_empty_pool_emits_nan_taper_columns(pools, monkeypatch):
    monkeypatch.setenv("AQNET2_NBR_TAPER", "1")
    p2 = dict(pools)
    p2["pa"] = pools["pa"].iloc[0:0]
    f = frame2.build_point_features([30.0, 30.2], [-97.9, -97.9],
                                    ["2024-06-03", "2024-06-04"], p2)
    for suf in ("", "_lag1"):
        col = f"nbr_pacal_50km{suf}_tp"
        assert col in f.columns
        assert np.isnan(f[col].to_numpy()).all()


# ── (3) taper weights strictly monotone decreasing in distance ──────────────

def test_taper_weights_monotone_decreasing():
    d = np.linspace(0.0, 200.0, 401)
    for r in frame2.PA_RADII_KM:
        w = frame2.taper_weights(d, r)
        assert w[0] == 1.0
        assert (w > 0.0).all()
        assert (np.diff(w) < 0.0).all(), f"taper not monotone at r={r}"
        # tau = r / 2: weight at the hard cutoff is exp(-2).
        assert abs(frame2.taper_weights(float(r), r) - np.exp(-2.0)) < 1e-12


# ── (4) hand-computed 2-sensor example, exact to 1e-9 ───────────────────────

def test_two_sensor_hand_computed_taper(monkeypatch):
    monkeypatch.setenv("AQNET2_NBR_TAPER", "1")
    d0 = pd.Timestamp("2024-06-03")
    qlat, qlon = 30.0, -97.9
    s1_lat, s2_lat = 30.05, 30.15          # ~5.6 km and ~16.7 km due north
    v1, v2 = 10.0, 20.0

    q = pd.DataFrame({"lat": [qlat], "lon": [qlon], "date": [d0],
                      "unit_id": ["q0"]})
    pool = pd.DataFrame({
        "unit_id": ["pa_A", "pa_B"],
        "lat": [s1_lat, s2_lat],
        "lon": [qlon, qlon],
        "date": [d0, d0],
        "value": [v1, v2],
    })
    out = frame2._neighbor_block(q, pool, "nbr_pacal", lag=0,
                                 keep_radii=frame2.PA_RADII_KM, with_std=True)

    d1 = float(frame2._haversine_km(qlat, qlon, s1_lat, qlon))
    d2 = float(frame2._haversine_km(qlat, qlon, s2_lat, qlon))
    assert d1 < d2 < 25.0                  # both inside every radius
    for r in frame2.PA_RADII_KM:
        tau = r / 2.0
        w1, w2 = np.exp(-d1 / tau), np.exp(-d2 / tau)
        expected = (w1 * v1 + w2 * v2) / (w1 + w2)
        got = float(out[f"nbr_pacal_{r}km_tp"][0])
        assert abs(got - expected) < 1e-9, (
            f"r={r}: tapered mean {got!r} != hand-computed {expected!r}")
        # The nearer, lower-value sensor dominates: strictly below the
        # uniform mean the hard-cutoff column reports.
        assert float(out[f"nbr_pacal_{r}km"][0]) == (v1 + v2) / 2.0
        assert got < (v1 + v2) / 2.0


# ── (5) lag semantics carry over: tp sees pool values at t - lag only ───────

def test_taper_lag_embargo_and_singleton(monkeypatch):
    monkeypatch.setenv("AQNET2_NBR_TAPER", "1")
    d0 = pd.Timestamp("2024-06-03")
    d1 = d0 + pd.Timedelta(days=1)
    pa_pool = pd.DataFrame({"unit_id": ["pa_9001"], "lat": [30.0],
                            "lon": [-97.9], "date": [d0], "value": [10.0]})
    pools_min = {"pa": pa_pool, "frm": pa_pool.iloc[0:0], "gridded": {},
                 "statics": None, "t0": None}
    f = frame2.build_point_features([30.001, 30.001], [-97.9, -97.9],
                                    [d0, d1], pools_min)
    r0, r1 = f.iloc[0], f.iloc[1]
    # Singleton neighbor: weights cancel, tapered mean equals the value
    # (to FP division roundoff, far inside the 1e-9 contract).
    assert abs(r0["nbr_pacal_25km_tp"] - 10.0) < 1e-9
    assert np.isnan(r0["nbr_pacal_25km_lag1_tp"])
    # At t = d1 the d0 value is visible ONLY through the lag-1 tp column.
    assert np.isnan(r1["nbr_pacal_25km_tp"])
    assert abs(r1["nbr_pacal_25km_lag1_tp"] - 10.0) < 1e-9


# ── (6) FP boundary: tp NaN pattern tracks avail bit-exactly at the cutoff ──

def test_taper_nan_matches_avail_at_radius_boundary(monkeypatch):
    """_taper_means queries the BallTree at the hard-cutoff pass's radius
    (max(RADII_KM)) and masks down per keep-radius, so a neighbor sitting
    at the 50 km FRM cutoff is included or excluded by the IDENTICAL
    d_km <= r comparison in both passes: 'tp is NaN exactly where avail
    is 0' holds on both sides of the boundary."""
    monkeypatch.setenv("AQNET2_NBR_TAPER", "1")
    d0 = pd.Timestamp("2024-06-03")
    qlat, qlon = 30.0, -97.9
    # Bisect a pool latitude due north to land as close to exactly 50 km
    # as float latitude arithmetic allows, then test one point per side.
    lo, hi = qlat + 0.40, qlat + 0.50
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if float(frame2._haversine_km(qlat, qlon, mid, qlon)) <= 50.0:
            lo = mid
        else:
            hi = mid
    for plat in (lo, hi):
        q = pd.DataFrame({"lat": [qlat], "lon": [qlon], "date": [d0],
                          "unit_id": ["q0"]})
        pool = pd.DataFrame({"unit_id": ["frm_A"], "lat": [plat],
                             "lon": [qlon],
                             "date": [d0 - pd.Timedelta(days=1)],
                             "value": [10.0]})
        out = frame2._neighbor_block(q, pool, "nbr_frm", lag=1,
                                     keep_radii=(50,), with_std=True)
        tp = np.asarray(out["nbr_frm_50km_lag1_tp"], dtype=np.float64)
        av = np.asarray(out["nbr_frm_avail_50km_lag1"], dtype=np.float64)
        assert (np.isnan(tp) == (av == 0.0)).all()
        # When the neighbor is in, the singleton tapered mean is its value.
        if av[0] == 1.0:
            assert abs(tp[0] - 10.0) < 1e-9
