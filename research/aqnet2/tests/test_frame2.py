"""frame2 unit tests: parity, deployment-honest lags, availability, hygiene.

The parity contract (DESIGN §6) is single-builder determinism: EVERY feature
— training row or arbitrary (lat, lon, date) query — flows through
build_point_features(), so calling it twice on the same inputs must be
identical column-for-column. The availability tests pin the v2 fix for v1's
grand-mean fallback (BUILD_NOTES audited fact #9: neighbor_features.py
silently fills zero-neighbor rows; frame2 must mask count == 0 back to NaN).
The lag test pins the embargo mechanics: lags shift the QUERY date, so a
query at t sees pool values from exactly t - lag and nothing newer.

All synthetic pools from conftest; no repo data files, no network.
"""
import numpy as np
import pandas as pd
import pytest

import config2
import frame2


# ── (1) parity: same query twice -> identical features ──────────────────────

def test_build_point_features_parity(pools):
    dates = pd.date_range("2024-06-01", periods=10, freq="D")
    # 30 triplets: 10 inside the sensor cluster, 10 ~30 km out, 10 hundreds
    # of km away (honest zero-coverage rows).
    lats = np.concatenate([30.05 + 0.01 * np.arange(10),
                           30.30 + 0.01 * np.arange(10),
                           33.50 + 0.10 * np.arange(10)])
    lons = np.concatenate([np.full(10, -97.90),
                           np.full(10, -97.90),
                           np.full(10, -94.50)])
    qdates = pd.DatetimeIndex(list(dates) * 3)

    f1 = frame2.build_point_features(lats, lons, qdates, pools)
    f2 = frame2.build_point_features(lats, lons, qdates, pools)

    assert list(f1.columns) == list(f2.columns)
    fcols = frame2.feature_columns(f1)
    assert fcols == frame2.feature_columns(f2)
    assert len(f1) == 30
    for c in fcols:
        a = f1[c].to_numpy(dtype=np.float64)
        b = f2[c].to_numpy(dtype=np.float64)
        assert np.array_equal(a, b, equal_nan=True), f"parity broke on {c}"

    # Coverage sanity: near rows have PA neighbors, far rows honestly none.
    assert (f1["nbr_pacal_avail_50km"].to_numpy()[:10] == 1.0).all()
    assert (f1["nbr_pacal_avail_100km"].to_numpy()[20:] == 0.0).all()
    assert np.isnan(f1["nbr_pacal_100km"].to_numpy()[20:]).all()


# ── (2) same-day FRM neighbor features are forbidden by construction ────────

def test_same_day_frm_neighbor_forbidden(pools):
    q = pd.DataFrame({"lat": [30.0], "lon": [-97.9],
                      "date": [pd.Timestamp("2024-06-03")],
                      "unit_id": ["q0"]})
    with pytest.raises(AssertionError):
        frame2._neighbor_block(q, pools["frm"], "nbr_frm", lag=0,
                               keep_radii=(frame2.FRM_RADIUS_KM,))
    # The sanctioned lagged path works and emits the frozen names.
    out = frame2._neighbor_block(q, pools["frm"], "nbr_frm", lag=1,
                                 keep_radii=(frame2.FRM_RADIUS_KM,))
    for name in ("nbr_frm_50km_lag1", "nbr_frm_count_50km_lag1",
                 "nbr_frm_avail_50km_lag1"):
        assert name in out


# ── (3) availability: empty pool and zero-neighbor rows are NaN, never fill ─

def test_empty_pa_pool_yields_nan_not_fill(pools):
    p2 = dict(pools)
    p2["pa"] = pools["pa"].iloc[0:0]
    f = frame2.build_point_features([30.0, 30.2], [-97.9, -97.9],
                                    ["2024-06-03", "2024-06-04"], p2)
    for suf in ("", "_lag1"):
        assert np.isnan(f[f"nbr_pacal_50km{suf}"].to_numpy()).all()
        assert (f[f"nbr_pacal_count_50km{suf}"].to_numpy() == 0.0).all()
        assert (f[f"nbr_pacal_avail_50km{suf}"].to_numpy() == 0.0).all()
    assert np.isnan(f["nbr_pacal_std_50km"].to_numpy()).all()


def test_far_query_masked_not_grand_mean_filled(pools):
    # v1 compute_neighbor_features_df grand-mean-fills zero-neighbor rows;
    # frame2 must mask count == 0 back to NaN even when the pool is rich on
    # that very day (BUILD_NOTES audited fact #9).
    f = frame2.build_point_features([35.9], [-93.6], ["2024-06-03"], pools)
    row = f.iloc[0]
    for r in (25, 50, 100):
        assert row[f"nbr_pacal_count_{r}km"] == 0.0
        assert row[f"nbr_pacal_avail_{r}km"] == 0.0
        assert np.isnan(row[f"nbr_pacal_{r}km"]), (
            "zero-neighbor mean was filled — v1's grand-mean fallback is "
            "leaking through the mask")


# ── (4) feature hygiene: coords and demographics never become features ──────

def test_feature_hygiene_statics(pools):
    n_st = 25
    statics = pd.DataFrame({
        "lat": np.linspace(29.9, 30.3, n_st),
        "lon": np.full(n_st, -97.9),
        "elevation": np.linspace(150.0, 300.0, n_st),
        "ejf_score": np.linspace(0.0, 1.0, n_st),   # demographic — must drop
    })
    assert "ejf_score" in config2.EXCLUDED_DEMOGRAPHIC
    p2 = dict(pools)
    p2["statics"] = statics
    f = frame2.build_point_features([30.0, 30.1], [-97.9, -97.9],
                                    ["2024-06-03", "2024-06-04"], p2)
    fcols = frame2.feature_columns(f)
    assert "st_elevation" in fcols
    assert np.isfinite(f["st_elevation"].to_numpy()).all()
    # Demographic column dropped at the join — never even prefixed in.
    assert not any("ejf_score" in c for c in f.columns)
    # Raw coordinates stay identity columns, never features.
    for banned in ("lat", "lon", "latitude", "longitude",
                   "dist_to_nearest_sensor"):
        assert banned not in fcols
    # feature_columns re-filters any frame that carries demographics.
    fake = f.copy()
    fake["pct_low_income"] = 1.0
    assert "pct_low_income" not in frame2.feature_columns(fake)


# ── (5) lag semantics: t-1 pool value visible at t ONLY via the lag1 col ────

def test_lag_semantics_pool_shifted_into_view():
    d0 = pd.Timestamp("2024-06-03")
    d1 = d0 + pd.Timedelta(days=1)
    pa_pool = pd.DataFrame({"unit_id": ["pa_9001"], "lat": [30.0],
                            "lon": [-97.9], "date": [d0], "value": [10.0]})
    pools_min = {"pa": pa_pool, "frm": pa_pool.iloc[0:0], "gridded": {},
                 "statics": None, "t0": None}
    f = frame2.build_point_features([30.001, 30.001], [-97.9, -97.9],
                                    [d0, d1], pools_min)
    r0, r1 = f.iloc[0], f.iloc[1]

    # Query at t = d0: pool value is same-day visible at lag 0 ...
    assert r0["nbr_pacal_25km"] == 10.0
    assert r0["nbr_pacal_count_25km"] == 1.0
    assert r0["nbr_pacal_avail_25km"] == 1.0
    assert r0["nbr_pacal_std_50km"] == 0.0        # singleton std, v1 value
    # ... and invisible at lag 1 (d0 - 1 has no pool rows).
    assert np.isnan(r0["nbr_pacal_25km_lag1"])
    assert r0["nbr_pacal_count_25km_lag1"] == 0.0
    assert r0["nbr_pacal_avail_25km_lag1"] == 0.0

    # Query at t = d1: the d0 value is gone from lag 0 ...
    assert np.isnan(r1["nbr_pacal_25km"])
    assert r1["nbr_pacal_avail_25km"] == 0.0
    # ... and appears exactly once, through the lag1 column.
    assert r1["nbr_pacal_25km_lag1"] == 10.0
    assert r1["nbr_pacal_count_25km_lag1"] == 1.0
    assert r1["nbr_pacal_avail_25km_lag1"] == 1.0


# ── vault airlock (pool builders + the builder's re-assert) ─────────────────

def test_vault_airlock_pool_builders(pools):
    aqs_like = pd.DataFrame({
        "site_id": ["site6", "site7"],
        "lat": [30.0, 30.1],
        "lon": [-97.8, -97.9],
        "date": [pd.Timestamp("2024-06-03")] * 2,
        "pm25_aqs": [9.0, 11.0],
    })
    # A vault unit surviving into a pool is a build error, never a warning.
    with pytest.raises(AssertionError):
        frame2.build_frm_pool(aqs_like, exclude_units=(),
                              fold_ctx={"vault_units": ["site7"]})
    pool = frame2.build_frm_pool(aqs_like, exclude_units=["site7"],
                                 fold_ctx={"vault_units": ["site7"]})
    assert set(pool["unit_id"]) == {"aqs_site6"}
    # build_point_features re-asserts on handed pools (last line of defense).
    bad = {"pa": pools["pa"], "frm": frame2.build_frm_pool(aqs_like),
           "gridded": {}, "statics": None, "t0": None}
    with pytest.raises(AssertionError):
        frame2.build_point_features([30.0], [-97.9], ["2024-06-03"], bad,
                                    fold_ctx={"vault_units": ["site7"]})


# ── fixture contract (frame_truth + folds shapes other suites rely on) ──────

def test_fixture_frame_and_folds_contract(frame_truth, folds):
    assert len(frame_truth) == 200
    assert list(frame_truth.columns[:9]) == [
        "unit_id", "unit_type", "network", "date", "lat", "lon",
        "y", "w", "cal_var"]
    fcols = frame2.feature_columns(frame_truth)
    assert "nbr_pacal_50km" in fcols
    assert "nbr_pacal_avail_50km" in fcols
    # Availability convention: value NaN exactly where avail == 0.
    v = frame_truth["nbr_pacal_50km"].to_numpy(dtype=np.float64)
    av = frame_truth["nbr_pacal_avail_50km"].to_numpy(dtype=np.float64)
    assert (np.isnan(v) == (av == 0.0)).all()

    assert folds["n_rows"] == 200
    for key in ("outer_fold", "spatial_block_fold", "temporal_is_test",
                "conformal_unit"):
        assert len(folds[key]) == 200
    for k in map(str, range(config2.OUTER_N_FOLDS)):
        assert len(folds["inner_fold"][k]) == 200
        assert len(folds["inner_role"][k]) == 200
        assert len(folds["loso_fold"][k]) == 200
    outer = np.asarray(folds["outer_fold"])
    is_pa = (frame_truth["unit_type"] == "pa").to_numpy()
    assert (outer[is_pa] == -1).all(), "PA rows must be -1 (always-train)"
    vault_units = {f"aqs_{s}" for s in folds["vault_sites"]}
    vault_rows = frame_truth["unit_id"].isin(vault_units).to_numpy()
    assert vault_rows.any()
    assert (outer[vault_rows] == -1).all(), "vault rows must be -1"
