"""Unit tests for priors.py — the T0 EPA-Downscaler-lineage debiased CTM
prior (DESIGN S5).

Covers the four contract points named in the build brief:
  1. Synthetic linear-in-CTM data -> evaluate_prior recovers the betas
     (beta0 via ctm=0 queries, beta1 via the ctm=1 minus ctm=0 delta) and
     the held-out predictions, within tolerance.
  2. A missing stream drops its pattern_id bit; t0 stays finite whenever
     at least one stream is present (single-stream combination == that
     stream's prediction).
  3. Rows with no stream at all -> t0 NaN, pattern 0 (never a fill).
  4. save/load npz roundtrip reproduces evaluate_prior bit-for-bit, and
     load_fold_models returns {} when the artifacts are absent.

Runs under pytest or directly:  python tests/test_priors.py
(no heavy optional deps required — numpy/pandas only, plus frame2's
module chain, which priors imports at module level by contract).
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config2   # noqa: E402
import priors    # noqa: E402

BETA0, BETA1 = 2.0, 0.8


def _synthetic_table(seed=7, n_sites=40, noise=0.3, two_streams=False):
    """AQS-like site-day table with y exactly linear in the geoscf stream:
    y = BETA0 + BETA1 * geoscf + N(0, noise). Dates cover all 4 seasons.
    two_streams additionally populates cams_pm25 (its own linear relation
    to y, so its fitted betas differ from geoscf's)."""
    rng = np.random.default_rng(seed)
    bb = config2.TX_BBOX
    lat = rng.uniform(bb["lat_min"] + 0.5, bb["lat_max"] - 0.5, n_sites)
    lon = rng.uniform(bb["lon_min"] + 0.5, bb["lon_max"] - 0.5, n_sites)
    dates = pd.date_range("2024-01-05", "2024-12-20", freq="3D")
    site = np.repeat(np.arange(n_sites), len(dates))
    g = rng.uniform(2.0, 40.0, len(site))
    y = BETA0 + BETA1 * g + rng.normal(0.0, noise, len(site))
    table = pd.DataFrame({
        "site_id": np.asarray([f"s{i}" for i in site]),
        "date": np.tile(dates.to_numpy(), n_sites),
        "lat": lat[site],
        "lon": lon[site],
        "pm25_aqs": y,
        "geoscf_pm25": g,
        "cams_pm25": np.nan,
        "merra2_pm25_proxy": np.nan,
    })
    if two_streams:
        # cams carries the same signal on a different scale: y is linear in
        # cams with beta0=2-0.8*5/0.5... (exact values irrelevant; the fit
        # recovers whatever they are), plus its own noise.
        table["cams_pm25"] = (g + 5.0) * 0.5 + rng.normal(0.0, 0.05,
                                                          len(site))
    return table


def _queries(n=200, seed=1):
    rng = np.random.default_rng(seed)
    qlat = rng.uniform(27.0, 35.0, n)
    qlon = rng.uniform(-105.0, -95.0, n)
    qdates = pd.to_datetime("2024-01-15") + pd.to_timedelta(
        rng.integers(0, 330, n), unit="D")
    return qlat, qlon, qdates


def _ctm(n, **finite):
    """ctm dict with all STREAMS NaN except the named finite arrays."""
    out = {s: np.full(n, np.nan) for s in priors.STREAMS}
    out.update({k: np.asarray(v, dtype=np.float64)
                for k, v in finite.items()})
    return out


def test_recovers_betas():
    table = _synthetic_table()
    model = priors.fit_downscaler(table, exclude_sites=(), label="test")
    assert model["streams"] == ["geoscf_pm25"]

    n = 200
    qlat, qlon, qdates = _queries(n)
    # beta0 field: prediction at ctm = 0.
    t0_a, pat_a, ps_a = priors.evaluate_prior(
        model, qlat, qlon, qdates, _ctm(n, geoscf_pm25=np.zeros(n)))
    b0_hat = ps_a["geoscf_pm25"]
    assert np.isfinite(b0_hat).all()
    assert (pat_a == 1).all()
    assert np.mean(np.abs(b0_hat - BETA0)) < 0.25, np.mean(np.abs(b0_hat - BETA0))
    # beta1 field: delta between ctm = 1 and ctm = 0.
    _, _, ps_b = priors.evaluate_prior(
        model, qlat, qlon, qdates, _ctm(n, geoscf_pm25=np.ones(n)))
    b1_hat = ps_b["geoscf_pm25"] - b0_hat
    assert np.mean(np.abs(b1_hat - BETA1)) < 0.1, np.mean(np.abs(b1_hat - BETA1))
    # Prediction recovery at fresh points with realistic ctm levels.
    rng = np.random.default_rng(3)
    g = rng.uniform(2.0, 40.0, n)
    t0, pat, ps = priors.evaluate_prior(
        model, qlat, qlon, qdates, _ctm(n, geoscf_pm25=g))
    truth = BETA0 + BETA1 * g
    rmse = float(np.sqrt(np.mean((t0 - truth) ** 2)))
    assert rmse < 0.5, rmse
    # Single-stream combination is that stream's prediction.
    np.testing.assert_allclose(t0, ps["geoscf_pm25"])
    # Schema stability: every stream name appears in per_stream.
    assert sorted(ps) == sorted(priors.STREAMS)
    assert np.isnan(ps["cams_pm25"]).all()


def test_missing_stream_drops_bit_and_no_stream_is_nan():
    table = _synthetic_table(two_streams=True)
    model = priors.fit_downscaler(table, exclude_sites=(), label="test2")
    assert model["streams"] == ["geoscf_pm25", "cams_pm25"]

    n = 30
    qlat, qlon, qdates = _queries(n, seed=5)
    g = np.full(n, 10.0)
    c = (g + 5.0) * 0.5
    g[20:] = np.nan            # rows 20..29: NO stream at all
    c[10:] = np.nan            # rows 10..19: geoscf only
    t0, pattern, ps = priors.evaluate_prior(
        model, qlat, qlon, qdates, _ctm(n, geoscf_pm25=g, cams_pm25=c))

    assert (pattern[:10] == 3).all(), pattern[:10]        # geoscf|cams = 1|2
    assert (pattern[10:20] == 1).all(), pattern[10:20]    # cams bit dropped
    assert (pattern[20:] == 0).all(), pattern[20:]        # nothing available
    # t0 finite exactly where pattern > 0; NaN rows are NaN, never filled.
    assert np.isfinite(t0[:20]).all()
    assert np.isnan(t0[20:]).all()
    # Single-stream rows: the combination IS the surviving stream.
    np.testing.assert_allclose(t0[10:20], ps["geoscf_pm25"][10:20])
    # Two-stream rows: combination lies between the two stream predictions.
    lo = np.minimum(ps["geoscf_pm25"][:10], ps["cams_pm25"][:10])
    hi = np.maximum(ps["geoscf_pm25"][:10], ps["cams_pm25"][:10])
    assert ((t0[:10] >= lo - 1e-9) & (t0[:10] <= hi + 1e-9)).all()
    # per_stream stays NaN-honest per stream.
    assert np.isnan(ps["cams_pm25"][10:]).all()
    assert np.isfinite(ps["cams_pm25"][:10]).all()


def test_save_load_roundtrip():
    table = _synthetic_table(two_streams=True)
    model = priors.fit_downscaler(table, exclude_sites=(), label="rt")
    n = 60
    qlat, qlon, qdates = _queries(n, seed=11)
    rng = np.random.default_rng(13)
    g = rng.uniform(2.0, 40.0, n)
    g[::7] = np.nan
    ctm = _ctm(n, geoscf_pm25=g, cams_pm25=(g + 5.0) * 0.5)

    t0_a, pat_a, ps_a = priors.evaluate_prior(model, qlat, qlon, qdates, ctm)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "prior_downscaler_test.npz")
        priors.save_model(model, path)
        m2 = priors.load_model(path)
    assert m2["streams"] == model["streams"]
    assert m2["ridge_lam"] == model["ridge_lam"]
    np.testing.assert_array_equal(m2["nodes"], model["nodes"])
    for s in model["streams"]:
        np.testing.assert_array_equal(m2["coef"][s], model["coef"][s])
        assert m2["precision"][s] == model["precision"][s]

    t0_b, pat_b, ps_b = priors.evaluate_prior(m2, qlat, qlon, qdates, ctm)
    np.testing.assert_array_equal(t0_a, t0_b)          # NaN == NaN here
    np.testing.assert_array_equal(pat_a, pat_b)
    for s in priors.STREAMS:
        np.testing.assert_array_equal(ps_a[s], ps_b[s])


def test_load_fold_models_absent_then_roundtrip():
    table = _synthetic_table()
    model = priors.fit_downscaler(table, exclude_sites=(), label="lfm")
    old = config2.ARTIFACTS_DIR
    try:
        with tempfile.TemporaryDirectory() as td:
            config2.ARTIFACTS_DIR = td
            assert priors.load_fold_models() == {}    # absent -> {}
            priors.save_model(
                model, config2.artifact("prior_downscaler_f0.npz"))
            priors.save_model(
                model, config2.artifact("prior_downscaler_full.npz"))
            models = priors.load_fold_models()
            assert set(models) == {0, "full"}          # int keys + "full"
            assert models[0]["streams"] == model["streams"]
    finally:
        config2.ARTIFACTS_DIR = old


def test_vault_and_fold_sites_excluded_from_fit():
    table = _synthetic_table()
    held = {"s0", "s1", "s2"}
    model = priors.fit_downscaler(table, exclude_sites=held, label="excl")
    assert model["meta"]["n_sites"] == table["site_id"].nunique() - len(held)
    assert sorted(held) == model["meta"]["excluded_sites"]


if __name__ == "__main__":
    tests = [test_recovers_betas,
             test_missing_stream_drops_bit_and_no_stream_is_nan,
             test_save_load_roundtrip,
             test_load_fold_models_absent_then_roundtrip,
             test_vault_and_fold_sites_excluded_from_fit]
    for t in tests:
        print(f"-- {t.__name__}")
        t()
    print(f"all {len(tests)} priors tests passed")
