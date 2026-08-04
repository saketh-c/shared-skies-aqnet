"""calibrate unit tests on fully synthetic colocation pairs.

Ground truth: y = 0.5*pa + 0.1*rh + 2 with small per-sensor offsets on the
PA readings — 20 sensors, 5 sites, 60 days. The truth is deliberately NOT
Barkjohn (0.524*pa - 0.0862*rh + 5.75), so any learned/refit form must beat
the published constants in LOLO — pinning gate G0's direction of pull.

All fits run the numpy linear rungs: the numpy_only fixture monkeypatches
the module-level HAS_GPBOOST / HAS_LGBM / HAS_STATSMODELS degradation
switches to False (they are read at call time inside _fit_on_frame and
_fit_var_model), which exercises the exact chain calibrate ships with when
no tree library is installed — deterministic and dependency-free.

No repo data files, no network, no folds2.json (folds=None paths only).
"""
import numpy as np
import pandas as pd
import pytest

import config2
import calibrate

N_SITES, N_SENSORS, N_DAYS = 5, 20, 60
TRUE_INTERCEPT, TRUE_PA, TRUE_RH = 2.0, 0.5, 0.1


@pytest.fixture
def numpy_only(monkeypatch):
    """Force the emergency numpy rungs (the documented degradation chain)."""
    monkeypatch.setattr(calibrate, "HAS_GPBOOST", False)
    monkeypatch.setattr(calibrate, "HAS_LGBM", False)
    monkeypatch.setattr(calibrate, "HAS_STATSMODELS", False)


@pytest.fixture
def cal_synth():
    """pairs / pa_daily / aqs_daily with a known linear truth.

    Site-day bases (pa_base, rh_base, t_base) drive both networks: the AQS
    truth is linear in the site-day base, each sensor reads the base plus a
    small fixed offset (the device nugget the KO form models), so per-row
    y = 0.5*pa + 0.1*rh + 2 - 0.5*offset_s + tiny noise. One PA reading is
    planted NaN — it must be DROPPED from the pair frame, never filled.
    """
    rng = np.random.default_rng(config2.SEED)
    dates = pd.date_range("2024-01-01", periods=N_DAYS, freq="D")
    site_ids = [f"S{i}" for i in range(N_SITES)]
    site_lat = 29.0 + 0.6 * np.arange(N_SITES)
    site_lon = -98.0 + 0.6 * np.arange(N_SITES)

    pa_base = rng.uniform(4.0, 40.0, (N_SITES, N_DAYS))
    rh_base = rng.uniform(20.0, 80.0, (N_SITES, N_DAYS))
    t_base = rng.uniform(5.0, 35.0, (N_SITES, N_DAYS))
    y_site = (TRUE_INTERCEPT + TRUE_PA * pa_base + TRUE_RH * rh_base
              + rng.normal(0.0, 0.05, (N_SITES, N_DAYS)))

    aqs_daily = pd.DataFrame({
        "site_id": np.repeat(site_ids, N_DAYS),
        "date": np.tile(dates, N_SITES),
        "pm25_aqs": y_site.ravel(),
        "lat": np.repeat(site_lat, N_DAYS),
        "lon": np.repeat(site_lon, N_DAYS),
        "is_fem": 0.0,
    })

    sensor_ids = [str(1000 + i) for i in range(N_SENSORS)]
    site_of = np.arange(N_SENSORS) % N_SITES
    off = rng.normal(0.0, 0.2, N_SENSORS)
    frames = []
    for i, sn in enumerate(sensor_ids):
        s = site_of[i]
        frames.append(pd.DataFrame({
            "sensor_id": sn,
            "date": dates,
            "lat": site_lat[s] + 0.004,
            "lon": site_lon[s] + 0.004,
            "pa_raw": pa_base[s] + off[i] + rng.normal(0.0, 0.05, N_DAYS),
            "rh": rh_base[s],
            "t": t_base[s],
        }))
    pa_daily = pd.concat(frames, ignore_index=True)
    pa_daily["sensor_id"] = pa_daily["sensor_id"].astype(str)
    pa_daily["dewpoint"] = calibrate.dewpoint_c(pa_daily["t"], pa_daily["rh"])
    calibrate.add_time_features(pa_daily)
    pa_daily["hms_smoke"] = 0.0
    smoke_days = set(dates[-5:])
    pa_daily.loc[pa_daily["date"].isin(smoke_days), "hms_smoke"] = 1.0
    first = pa_daily.groupby("sensor_id")["date"].transform("min")
    pa_daily["sensor_age_days"] = ((pa_daily["date"] - first)
                                   .dt.days.astype(np.float64))
    pa_daily["channel_reconstructed"] = 0.0
    pa_daily["urban"] = 0.0
    # Planted NaN raw reading: NaN is the only missingness representation.
    pa_daily.loc[pa_daily.index[7], "pa_raw"] = np.nan

    pairs = pd.DataFrame({
        "site_id": [site_ids[site_of[i]] for i in range(N_SENSORS)],
        "sensor_id": sensor_ids,
        "dist_km": 0.5 + 0.05 * np.arange(N_SENSORS),
        "n_shared_days": N_DAYS,
    })
    return {"pairs": pairs, "pa_daily": pa_daily, "aqs_daily": aqs_daily}


# ── (1) build_cal_frame row structure + weights ─────────────────────────────

def test_build_cal_frame_rows_and_weights(cal_synth):
    cal = calibrate.build_cal_frame(cal_synth["pairs"],
                                    cal_synth["pa_daily"],
                                    cal_synth["aqs_daily"],
                                    max_dist_km=10.0)
    # One row per (site, sensor, day) minus the planted-NaN reading.
    assert len(cal) == N_SENSORS * N_DAYS - 1
    for col in ("site_id", "sensor_id", "dist_km", "date", "y", "pa", "rh",
                "t", "w", "year", "sensor_year", "hms_smoke", "is_fem"):
        assert col in cal.columns, f"pair frame lost column {col}"
    ok = (np.isfinite(cal["y"]) & np.isfinite(cal["pa"])
          & np.isfinite(cal["rh"]) & np.isfinite(cal["t"]))
    assert ok.all(), "rows without finite (y, pa, rh, t) must be dropped"
    # Distance-decayed precision weight, exactly as documented.
    expect_w = (np.exp(-cal["dist_km"].to_numpy(dtype=np.float64)
                       / calibrate.PAIR_WEIGHT_SCALE_KM)
                / float(config2.SIGMA_FRM) ** 2)
    assert np.allclose(cal["w"].to_numpy(dtype=np.float64), expect_w)
    # sensor_year drift key, str-typed ids (pandas 3 join hygiene).
    assert (cal["sensor_year"]
            == cal["sensor_id"] + "_" + cal["year"].astype(str)).all()
    assert (cal["sensor_id"].astype(str) == cal["sensor_id"]).all()
    assert cal["site_id"].nunique() == N_SITES
    assert cal["sensor_id"].nunique() == N_SENSORS


# ── (2) the numpy linear rung recovers the known coefficients ───────────────

def test_linear_rung_recovers_truth(cal_synth, numpy_only):
    cal = calibrate.build_cal_frame(cal_synth["pairs"],
                                    cal_synth["pa_daily"],
                                    cal_synth["aqs_daily"],
                                    max_dist_km=10.0)
    model = calibrate._fit_on_frame(cal, num_boost_round=10,
                                    seed=config2.SEED,
                                    model_form="linear_rh")
    assert model.kind == "linear_rh"
    b0, b_pa, b_rh = model.linear["coef"]
    assert abs(b0 - TRUE_INTERCEPT) < 0.2
    assert abs(b_pa - TRUE_PA) < 0.02
    assert abs(b_rh - TRUE_RH) < 0.01

    # "learned" with every tree switch off must land on the emergency
    # linear_rht rung (the documented degradation chain), not crash.
    m2 = calibrate._fit_on_frame(cal, num_boost_round=10,
                                 seed=config2.SEED, model_form="learned")
    assert m2.kind == "linear_rht"

    # Predictions: clipped >= 0 and NaN exactly where pa is NaN (no fill).
    q = cal.head(4).copy()
    q.loc[q.index[0], "pa"] = np.nan
    pred = m2.predict(q)
    assert np.isnan(pred[0])
    assert np.isfinite(pred[1:]).all()
    assert (pred[1:] >= 0.0).all()

    # Variance model degraded to the announced scalar, floored.
    v = m2.predict_var(cal.head(8))
    assert (v >= calibrate.CAL_VAR_MIN).all()


# ── (3) lolo_validate: frozen schema + learned beats published Barkjohn ─────

def test_lolo_validate_schema_and_beats_barkjohn(cal_synth, numpy_only):
    res = calibrate.lolo_validate(cal_synth["pairs"],
                                  cal_synth["pa_daily"],
                                  cal_synth["aqs_daily"],
                                  folds=None, max_dist_km=10.0)
    assert set(res) == {"max_dist_km", "n_sites", "n_pair_days",
                        "model_kinds", "methods", "by_year_bias",
                        "per_site", "g0"}
    assert set(res["methods"]) == {"learned", "barkjohn", "barkjohn_refit",
                                   "amt_rht"}
    for m in res["methods"].values():
        assert set(m) == {"rmse", "mae", "bias", "r2", "n"}
        assert m["n"] > 0
    assert set(res["g0"]) == {"verdict", "criteria", "fallback_form",
                              "production_form"}
    assert res["g0"]["verdict"] in ("pass", "fail")
    assert set(res["g0"]["criteria"]) == {
        "rmse_beats_barkjohn", "rmse_beats_amt_rht",
        "bias_beats_barkjohn", "bias_beats_amt_rht"}
    assert res["n_sites"] == N_SITES
    assert len(res["per_site"]) == N_SITES
    assert res["model_kinds"] == ["linear_rht"]   # emergency rung under LOLO
    assert "2024" in res["by_year_bias"]

    # The truth is not Barkjohn — the learned form must beat the published
    # constants on both LOLO RMSE and |bias| on this synthetic.
    ml, mb = res["methods"]["learned"], res["methods"]["barkjohn"]
    assert ml["rmse"] < mb["rmse"]
    assert abs(ml["bias"]) < abs(mb["bias"])


# ── (4) cal_var_floor: monotone, conservative, floor(0) = 1.0 ───────────────

def test_cal_var_floor_monotone():
    d = np.array([0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0])
    v = calibrate.cal_var_floor(d)
    assert v[0] == calibrate.VAR_FLOOR_B
    assert v[0] == 1.0
    assert (np.diff(v) > 0.0).all(), "floor must be strictly increasing"
    # Negative distances clip to 0 (never a bonus below the minimum).
    assert calibrate.cal_var_floor(np.array([-3.0]))[0] == 1.0
    # Documented anchors: ~7.9 at 50 km, ~26.6 at 100 km.
    assert abs(v[5] - (4.0 * np.expm1(1.0) + 1.0)) < 1e-12
    assert abs(v[6] - (4.0 * np.expm1(2.0) + 1.0)) < 1e-12


# ── (5) apply_calibration: NaN passthrough + reconstructed-channel inflation

def test_apply_calibration_nan_and_recon_inflation():
    model = calibrate.CalModel("linear_rh", {"kind": "linear_rh"})
    model.linear = {"cols": ["pa", "rh"],
                    "coef": [TRUE_INTERCEPT, TRUE_PA, TRUE_RH]}
    df = pd.DataFrame({
        "sensor_id": ["1", "1", "2", "2", "3"],
        "date": pd.to_datetime(["2024-01-01"] * 5),
        "pa_raw": [10.0, np.nan, 10.0, 10.0, 10.0],
        "rh": [50.0] * 5,
        "channel_reconstructed": [0.0, 0.0, 0.0, 1.0, 0.0],
        "dist_to_nearest_frm": [0.0, 0.0, 0.0, 0.0, 100.0],
    })
    mean, var = calibrate.apply_calibration(model, df)

    assert mean[0] == pytest.approx(TRUE_INTERCEPT + TRUE_PA * 10.0
                                    + TRUE_RH * 50.0)
    # NaN raw PA -> NaN calibrated mean AND NaN cal_var (no fill, ever).
    assert np.isnan(mean[1])
    assert np.isnan(var[1])
    # Colocated sensor: scalar model var (0.25) is lifted to floor(0) = 1.0.
    assert var[2] == pytest.approx(calibrate.cal_var_floor(np.array([0.0]))[0])
    # channel_reconstructed = 1 inflates cal_var by the fixed factor.
    assert var[3] == pytest.approx(var[2]
                                   * calibrate.CHANNEL_RECON_VAR_FACTOR)
    # Off-support sensor: the monotone floor dominates.
    assert var[4] == pytest.approx(
        calibrate.cal_var_floor(np.array([100.0]))[0])
    assert var[4] > var[2]
