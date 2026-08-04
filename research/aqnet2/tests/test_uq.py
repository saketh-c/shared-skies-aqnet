"""Synthetic tests for uq.py — no repo artifacts, no heavy deps required.

Covers the four INTERFACES/testing-floor claims for the uq module:
  1. unit_scores returns exactly one row per unit (honest n = units)
  2. nexcp_delta is monotone (non-increasing) in rho_s for a far query —
     as the spatial decay loosens, distant calibration units gain mass and
     the query's +inf point mass shrinks
  3. split-conformal coverage on iid synthetic data lands near 1 - alpha
     (unit-level scores, uniform-weight NexCP limit)
  4. the fitted_against lineage hash changes whenever the composite npz
     bytes change

Run directly (pytest-free) or via pytest:
    python tests/test_uq.py
    pytest tests/test_uq.py
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uq  # noqa: E402


def _make_scores_df(n_units, scores, lat, lon, coverage_bin=0,
                    last_date="2026-05-01"):
    return pd.DataFrame({
        "unit_id": [f"u{i}" for i in range(n_units)],
        "score": np.asarray(scores, dtype=np.float64),
        "n_rows": np.full(n_units, 30, dtype=np.int64),
        "lat": np.full(n_units, float(lat)),
        "lon": np.full(n_units, float(lon)),
        "last_date": pd.to_datetime([last_date] * n_units),
        "coverage_bin": np.full(n_units, int(coverage_bin), dtype=np.int8),
    })


def test_unit_scores_one_row_per_unit():
    rng = np.random.default_rng(42)
    n_units, per = 7, 40
    n = n_units * per
    unit_id = np.repeat([f"s{i:02d}" for i in range(n_units)], per)
    y = rng.normal(10.0, 2.0, n)
    pred = y + rng.normal(0.0, 0.5, n)
    sigma = np.full(n, 1.0)
    df = uq.unit_scores(y, pred, sigma, unit_id, np.ones(n, bool))

    assert len(df) == n_units, f"expected {n_units} rows, got {len(df)}"
    assert sorted(df["unit_id"]) == sorted(set(unit_id))
    assert df["unit_id"].is_unique
    assert np.isfinite(df["score"].to_numpy()).all()
    assert (df["n_rows"].to_numpy() == per).all()

    # Masked-out unit disappears; NaN rows drop per-unit (NaN-honest).
    mask = unit_id != "s00"
    y2 = y.copy()
    y2[unit_id == "s01"] = np.nan
    df2 = uq.unit_scores(y2, pred, sigma, unit_id, mask)
    assert len(df2) == n_units - 2  # s00 masked, s01 all-NaN
    assert "s00" not in set(df2["unit_id"])
    assert "s01" not in set(df2["unit_id"])

    # Signed-offset band form agrees with the symmetric form.
    offsets = np.column_stack([np.full(n, -1.0), np.full(n, 1.0)])
    df3 = uq.unit_scores(y, pred, offsets, unit_id, np.ones(n, bool))
    assert np.allclose(df3["score"].to_numpy(), df["score"].to_numpy())


def test_nexcp_delta_monotone_in_rho_s_for_far_query():
    rng = np.random.default_rng(42)
    n_units = 40
    scores = rng.uniform(0.0, 1.0, n_units)
    # Units clustered near Austin; query far away (~650 km).
    df = _make_scores_df(n_units, scores, lat=30.3, lon=-97.7,
                         coverage_bin=1)
    query = (33.6, -103.5)
    ref = "2026-05-01"  # == last_date, so staleness is zero

    deltas = [uq.nexcp_delta(df, query, 1, rho_s=r, tau=uq.TAU_DAYS,
                             ref_date=ref)
              for r in (25.0, 150.0, 1000.0, 1e8)]
    for a, b in zip(deltas, deltas[1:]):
        assert a >= b, f"delta not non-increasing in rho_s: {deltas}"
    # Tight decay: the query's +inf mass dominates -> honest inf.
    assert np.isinf(deltas[0])
    # Loose decay approaches the uniform-weight conformal quantile.
    assert np.isfinite(deltas[-1])
    assert 0.0 <= deltas[-1] <= 1.0

    # Empty bin -> inf (default-honest, never default-tight).
    assert np.isinf(uq.nexcp_delta(df, query, 2, ref_date=ref))


def test_conformal_coverage_iid_within_tolerance():
    rng = np.random.default_rng(42)
    # Many rows per unit so the within-unit finite-sample correction is
    # small and the test isolates validity; the construction is honestly
    # CONSERVATIVE (corrections at both levels), so the tolerance is
    # asymmetric around the nominal 0.90: coverage must never fall below
    # it by more than noise and may exceed it moderately.
    n_units, per = 80, 200
    n = n_units * per
    unit_id = np.repeat([f"c{i:03d}" for i in range(n_units)], per)
    y = rng.normal(0.0, 1.0, n)
    pred = np.zeros(n)
    sigma = np.ones(n)  # raw band ±1 under-covers (~68%) -> delta widens it

    scores = uq.unit_scores(y, pred, sigma, unit_id, np.ones(n, bool),
                            alpha=0.1)
    df = _make_scores_df(len(scores), scores["score"].to_numpy(),
                         lat=30.0, lon=-97.0, coverage_bin=0)
    # Uniform-weight NexCP limit: query at the units, huge decay scales.
    delta = uq.nexcp_delta(df, (30.0, -97.0), 0, rho_s=1e9, tau=1e9,
                           alpha=0.1, ref_date="2026-05-01")
    assert np.isfinite(delta) and delta > 0.0

    y_new = rng.normal(0.0, 1.0, 20000)
    cover = float(np.mean(np.abs(y_new) <= 1.0 + delta))
    assert 0.87 <= cover <= 0.96, (
        f"iid coverage {cover:.4f} outside tolerance for nominal 0.90 "
        f"(delta={delta:.4f})")


def test_lineage_hash_changes_with_composite_bytes():
    with tempfile.TemporaryDirectory() as td:
        p1 = os.path.join(td, "oof_composite.npz")
        p2 = os.path.join(td, "oof_composite_b.npz")
        rng = np.random.default_rng(42)
        oof = rng.normal(10.0, 3.0, 500)
        mask = np.ones((500, 4), dtype=np.uint8)
        np.savez_compressed(p1, oof_final=oof, tier_mask=mask)
        h1 = uq.composite_hash(p1)
        assert isinstance(h1, str) and len(h1) == 64
        # Same file, same bytes -> same hash (pure function of bytes).
        assert uq.composite_hash(p1) == h1
        # One value regenerated -> different bytes -> different hash.
        oof2 = oof.copy()
        oof2[0] += 1e-6
        np.savez_compressed(p2, oof_final=oof2, tier_mask=mask)
        h2 = uq.composite_hash(p2)
        assert h1 != h2, "lineage hash must change when composite bytes do"


if __name__ == "__main__":
    tests = [
        test_unit_scores_one_row_per_unit,
        test_nexcp_delta_monotone_in_rho_s_for_far_query,
        test_conformal_coverage_iid_within_tolerance,
        test_lineage_hash_changes_with_composite_bytes,
    ]
    for t in tests:
        t()
        print(f"[test_uq] PASS {t.__name__}")
    print(f"[test_uq] all {len(tests)} tests passed")
