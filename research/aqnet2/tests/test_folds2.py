"""folds2 invariants on synthetic data only -- no repo parquets touched.

Covers the INTERFACES.md testing floor for folds2: folds_from_assign -1
semantics, the vault buffer invariant, content-hash mismatch detection,
Phase-1 determinism, inner-fold disjointness (a site never appears in its
own outer fold's calibration split -- the calibrate.py alignment contract),
and selection/confirmation row- AND cluster-disjointness per outer fold.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config2  # noqa: E402
import folds2   # noqa: E402


# ── Synthetic geography ─────────────────────────────────────────────────────
# 5x5 grid spaced 0.7 deg (~67-77 km -- every grid site individually
# satisfies a 30 km buffer) plus a 5-site cluster spaced ~2.8 km apart (no
# cluster site can EVER be vaulted greedily: at acceptance time its cluster
# neighbors are still non-vault and < 30 km away).

def _site_df(with_cluster=True):
    rows = []
    i = 0
    for a in range(5):
        for b in range(5):
            rows.append((f"S{i:03d}", 29.0 + a * 0.7, -100.0 + b * 0.7))
            i += 1
    if with_cluster:
        for j in range(5):
            rows.append((f"C{j:03d}", 33.5, -96.5 + j * 0.03))
    df = pd.DataFrame(rows, columns=["site_id", "lat", "lon"])
    df["pm_mean"] = np.linspace(4.0, 16.0, len(df))
    df["n_days"] = 400
    return df


def _assignments(seed=config2.SEED, n_vault=5, n_outer=3):
    return folds2.site_assignments(_site_df(), seed=seed, n_vault=n_vault,
                                   buffer_km=30.0, n_outer=n_outer,
                                   n_inner=config2.INNER_N_FOLDS)


def _frame(sf, site_df, n_pa=8, n_days=8, start="2024-03-01"):
    """Synthetic two-network frame matching the frame2 identity contract."""
    dates = pd.date_range(start, periods=n_days, freq="D")
    rng = np.random.default_rng(0)
    loc = site_df.set_index("site_id")
    rows = []
    for s in sorted(sf["outer_fold_of_site"]):
        for d in dates:
            rows.append(("aqs_" + s, "aqs", d, float(loc.loc[s, "lat"]),
                         float(loc.loc[s, "lon"]),
                         float(rng.uniform(2.0, 30.0))))
    for i in range(n_pa):
        for d in dates:
            rows.append((f"pa_{1000 + i}", "pa", d, 30.0 + 0.2 * i,
                         -99.0 + 0.2 * i, float(rng.uniform(2.0, 30.0))))
    fr = pd.DataFrame(rows, columns=["unit_id", "unit_type", "date",
                                     "lat", "lon", "y"])
    return (fr.sort_values(["unit_id", "date"], kind="mergesort")
              .reset_index(drop=True))


# ── folds_from_assign ───────────────────────────────────────────────────────

def test_folds_from_assign_minus_one_always_train():
    assign = np.array([-1, 0, 0, 1, 2, -1, 2])
    folds = folds2.folds_from_assign(assign)
    assert len(folds) == 3
    for k, (tr, te) in enumerate(folds):
        # -1 rows train in EVERY fold, are NEVER test rows
        assert 0 in tr and 5 in tr
        assert 0 not in te and 5 not in te
        assert set(te.tolist()) == {i for i, a in enumerate(assign) if a == k}
        assert not (set(tr.tolist()) & set(te.tolist()))
        assert set(tr.tolist()) | set(te.tolist()) == set(range(len(assign)))


def test_folds_from_assign_all_minus_one_is_empty():
    assert folds2.folds_from_assign(np.array([-1, -1, -1])) == []


# ── Vault ───────────────────────────────────────────────────────────────────

def test_vault_buffer_invariant_on_synthetic_coords():
    df = _site_df()
    sf = _assignments()
    vault = sf["vault_sites"]
    assert len(vault) == 5
    # dense-cluster sites are un-vaultable: each has a < 30 km non-vault
    # neighbor at acceptance time
    assert not any(v.startswith("C") for v in vault)
    # the invariant proper: every vault site >= 30 km from every FINAL
    # non-vault site
    loc = df.set_index("site_id")
    nonvault = [s for s in df["site_id"] if s not in set(vault)]
    for v in vault:
        for s in nonvault:
            d = float(folds2.haversine_km(loc.loc[v, "lat"],
                                          loc.loc[v, "lon"],
                                          loc.loc[s, "lat"],
                                          loc.loc[s, "lon"]))
            assert d >= 30.0, (v, s, d)
    # vault sites carry -1 in the (total) outer map
    assert all(sf["outer_fold_of_site"][v] == -1 for v in vault)


# ── Phase-1 determinism + calibrate contract ────────────────────────────────

def test_phase1_determinism_two_calls_identical():
    a = _assignments(seed=7)
    b = _assignments(seed=7)
    assert a == b
    c = _assignments(seed=8)
    assert a != c  # the seed is load-bearing


def test_inner_matches_calibrate_derived_fallback():
    """The emitted site-level inner map must equal calibrate.py's derived
    fallback bit-for-bit (rng SEED + 100003*(k+1), permutation of the
    sorted remaining list, pos % INNER_N_FOLDS)."""
    sf = _assignments(seed=config2.SEED)
    vault = set(sf["vault_sites"])
    omap = sf["outer_fold_of_site"]
    n_inner = int(config2.INNER_N_FOLDS)
    outer_ids = sorted({f for f in omap.values() if f >= 0})
    assert outer_ids
    for k in outer_ids:
        remaining = sorted(s for s, f in omap.items()
                           if f != k and s not in vault)
        rng = np.random.default_rng(config2.SEED + 100003 * (k + 1))
        order = rng.permutation(len(remaining))
        derived = {remaining[ix]: int(pos % n_inner)
                   for pos, ix in enumerate(order)}
        assert sf["inner_fold_of_site"][str(k)] == derived


def test_inner_disjointness_site_never_in_own_fold():
    sf = _assignments()
    vault = set(sf["vault_sites"])
    omap = sf["outer_fold_of_site"]
    for k in sorted({f for f in omap.values() if f >= 0}):
        inner_sites = set(sf["inner_fold_of_site"][str(k)])
        own = {s for s, f in omap.items() if f == k}
        assert not (own & inner_sites)       # never in own fold's split
        assert not (vault & inner_sites)     # vault airlock
        # inner values span 0..n_inner-1 only
        js = set(sf["inner_fold_of_site"][str(k)].values())
        assert js <= set(range(config2.INNER_N_FOLDS))


# ── Phase 2 row-level semantics ─────────────────────────────────────────────

def test_build_folds_row_semantics_and_sel_conf_disjointness():
    df = _site_df()
    sf = _assignments(n_vault=4)
    fr = _frame(sf, df)
    folds = folds2.build_folds(fr, seed=config2.SEED, site_folds=sf, n_loso=4)

    assert folds["n_rows"] == len(fr)
    assert folds["content_hash"] == folds2.content_hash(fr)

    outer = np.asarray(folds["outer_fold"])
    utype = fr["unit_type"].to_numpy()
    uid = fr["unit_id"].astype(str).to_numpy()
    vmask = folds2.vault_row_mask(fr, folds)

    # PA rows and vault rows are -1 in outer_fold
    assert (outer[utype == "pa"] == -1).all()
    assert (outer[vmask] == -1).all()
    # non-vault AQS rows carry their site's fold
    omap = folds["outer_fold_of_site"]
    for i in np.where((utype == "aqs") & ~vmask)[0]:
        assert outer[i] == omap[uid[i][4:]]

    outer_ids = sorted({f for f in omap.values() if f >= 0})
    for k in outer_ids:
        sk = str(k)
        ik = np.asarray(folds["inner_fold"][sk])
        rk = np.asarray(folds["inner_role"][sk])
        lk = np.asarray(folds["loso_fold"][sk])

        # fold-k outer rows and vault rows excluded from the inner split
        assert (ik[outer == k] == -1).all()
        assert (ik[vmask] == -1).all()
        # role encoding: 0 sel (inner 0-1), 1 conf (2-3), 2 excluded (-1)
        assert ((rk == 2) == (ik == -1)).all()
        assert ((rk == 0) == ((ik >= 0)
                              & (ik < config2.INNER_N_FOLDS // 2))).all()
        assert ((rk == 1) == (ik >= config2.INNER_N_FOLDS // 2)).all()
        # PA rows are covered by the inner split (selection/confirmation
        # masks must span PA too)
        assert (ik[(utype == "pa") & ~vmask] >= 0).all()

        # sel/conf are row-disjoint (encoding) AND cluster-disjoint
        sel_units = set(uid[rk == 0])
        conf_units = set(uid[rk == 1])
        assert not (sel_units & conf_units)

        # LOSO: -1 exactly on fold-k test rows + vault; training rows dealt
        assert (lk[outer == k] == -1).all()
        assert (lk[vmask] == -1).all()
        train = (outer != k) & ~vmask
        assert (lk[train] >= 0).all()
        assert (lk[train] < 4).all()
        # unit-grouped: one fold per unit
        for u in set(uid[train]):
            assert len(set(lk[(uid == u) & train])) == 1

    # spatial blocks are total over rows; temporal split matches the cutoff
    sb = np.asarray(folds["spatial_block_fold"])
    assert (sb >= 0).all()
    tt = np.asarray(folds["temporal_is_test"])
    expect = (pd.to_datetime(fr["date"])
              >= pd.Timestamp(config2.TEMPORAL_CUTOFF)).to_numpy()
    assert (tt.astype(bool) == expect).all()

    # conformal units: unit-constant, never vault, PA share matches the
    # configured fraction
    cu = np.asarray(folds["conformal_unit"])
    assert set(cu.tolist()) <= {0, 1}
    assert (cu[vmask] == 0).all()
    conf_pa = {u for u in set(uid[cu == 1]) if u.startswith("pa_")}
    n_pa = len({u for u in uid if u.startswith("pa_")})
    assert len(conf_pa) == int(round(config2.CONFORMAL_PA_FRAC * n_pa))
    for u in set(uid[cu == 1]):
        assert len(set(cu[uid == u])) == 1


def test_build_folds_deterministic():
    df = _site_df()
    sf = _assignments(n_vault=4)
    fr = _frame(sf, df)
    a = folds2.build_folds(fr, seed=config2.SEED, site_folds=sf, n_loso=4)
    b = folds2.build_folds(fr, seed=config2.SEED, site_folds=sf, n_loso=4)
    assert a == b


# ── Persistence + hash guard ────────────────────────────────────────────────

def test_hash_mismatch_raises(tmp_path):
    df = _site_df()
    sf = _assignments(n_vault=4)
    fr = _frame(sf, df)
    folds = folds2.build_folds(fr, seed=config2.SEED, site_folds=sf, n_loso=4)
    p = str(tmp_path / "folds2.json")
    folds2.save_folds(folds, p)

    # clean round-trip
    loaded = folds2.load_folds(p, fr)
    assert loaded["content_hash"] == folds["content_hash"]
    assert loaded["outer_fold"] == folds["outer_fold"]

    # a single mutated y must be caught
    bad = fr.copy()
    bad.loc[0, "y"] = float(bad.loc[0, "y"]) + 1.0
    with pytest.raises(ValueError):
        folds2.load_folds(p, bad)

    # a row-count change must be caught
    with pytest.raises(ValueError):
        folds2.load_folds(p, fr.iloc[:-1])

    # a Phase-1-only json (content_hash == "") must be rejected too
    phase1 = {"n_rows": len(fr), "content_hash": "", "seed": config2.SEED}
    p1 = str(tmp_path / "phase1.json")
    folds2.save_folds(phase1, p1)
    with pytest.raises(ValueError):
        folds2.load_folds(p1, fr)
    # ... while the raw loader serves Phase-1 consumers without a frame
    assert folds2.load_folds_raw(p1)["content_hash"] == ""


def test_content_hash_row_order_invariant():
    df = _site_df()
    sf = _assignments(n_vault=4)
    fr = _frame(sf, df)
    shuffled = fr.sample(frac=1.0, random_state=3).reset_index(drop=True)
    assert folds2.content_hash(fr) == folds2.content_hash(shuffled)
    bad = fr.copy()
    bad.loc[0, "y"] = float(bad.loc[0, "y"]) + 1e-3   # visible at 6 dp
    assert folds2.content_hash(fr) != folds2.content_hash(bad)
