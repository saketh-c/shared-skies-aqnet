"""Permanent regression tests for compose.py — the composition airlock.

The 17%-finite test is THE mandated permanent regression (INTERFACES.md
"Testing floor"): v1's fusion stage fit its combiner on the ~17% of rows
where every component happened to be finite and MEAN-FILLED the missing
component elsewhere — measured paired delta-R2 of -0.050/-0.031 with CIs
excluding zero and exceedance recall driven to ~0. The v2 contract is that
a component finite on 17% of rows leaves the OTHER 83% bit-identical to the
incumbent (np.ndarray.tobytes equality — NaN payloads and signed zeros
count), and that a component whose held-out skill is negative is rejected
with alpha 0 and its decision recorded.

All synthetic, seeded from config2.SEED; no repo files, no network.
"""
import json
import os

import numpy as np
import pytest

import config2
import compose

# Explicit margins (the power_analysis.json "margins" subdict shape) so no
# safe-default warning fires; one test passes the full payload form instead
# to exercise compose._resolve_margins payload detection.
MARGINS = {"pooled_r2": 0.01, "spatial_r2": 0.01, "exceedance_f1": 0.02}
PAYLOAD_MARGINS = {"mde_pooled_r2": 0.02, "mde_site_r2": 0.05,
                   "margins": dict(MARGINS)}

N_UNITS, N_DAYS = 40, 30


def _synth(avail_frac=0.5, res_mode="skilled", seed=config2.SEED):
    """Clustered synthetic: the incumbent misses a site-level signal.

    y = 10 + site_eff + noise; incumbent = y - 0.8*site_eff + noise, so the
    true residual is a per-cluster offset a component can genuinely learn.
    res_mode:
      "skilled"   residual = true error + N(0, 0.1) on available rows
      "anti"      residual = -true error + noise (harmful everywhere)
      "sel_only"  skilled on selection units, harmful on confirmation units
                  (skill that does not generalize — must die in admission)
    Availability is an exact row count (round(avail_frac * n)) drawn from a
    seeded permutation; residual is NaN wherever avail == 0 (the airlock).
    Selection = units 0..19, confirmation = units 20..39 (row- AND
    cluster-disjoint, as fit_gate asserts).
    """
    rng = np.random.default_rng(seed)
    n = N_UNITS * N_DAYS
    unit = np.repeat(np.arange(N_UNITS), N_DAYS)
    site_eff = rng.normal(0.0, 3.0, N_UNITS)
    y = 10.0 + site_eff[unit] + rng.normal(0.0, 1.0, n)
    signal = 0.8 * site_eff[unit]
    inc = y - signal + rng.normal(0.0, 0.3, n)

    n_avail = int(round(avail_frac * n))
    perm = rng.permutation(n)
    avail = np.zeros(n, dtype=bool)
    avail[perm[:n_avail]] = True

    sel = unit < N_UNITS // 2
    conf = ~sel
    true_err = y - inc
    res = np.full(n, np.nan)
    if res_mode == "skilled":
        res[avail] = true_err[avail] + rng.normal(0.0, 0.1, n_avail)
    elif res_mode == "anti":
        res[avail] = -true_err[avail] + rng.normal(0.0, 0.1, n_avail)
    elif res_mode == "sel_only":
        g = avail & sel
        b = avail & conf
        res[g] = true_err[g] + rng.normal(0.0, 0.1, int(g.sum()))
        res[b] = -true_err[b] + rng.normal(0.0, 0.1, int(b.sum()))
    else:
        raise ValueError(res_mode)

    return {"y": y, "inc": inc, "res": res, "avail": avail,
            "pattern": np.ones(n, dtype=np.int8),
            "stratum": np.zeros(n, dtype=np.int8),
            "unit": unit, "sel": sel, "conf": conf, "n": n}


def _fit(d, margins=MARGINS):
    return compose.fit_gate(d["y"], d["inc"], d["res"], d["avail"],
                            d["pattern"], d["stratum"], d["unit"],
                            d["sel"], d["conf"], margins)


def _apply(d, gates, pattern=None, stratum=None):
    return compose.apply_gates(
        d["inc"], d["res"], d["avail"],
        d["pattern"] if pattern is None else pattern,
        d["stratum"] if stratum is None else stratum, gates)


# ── (1) THE mandated permanent regression: 17%-finite component ─────────────

def test_17pct_finite_component_bit_identical_passthrough():
    d = _synth(avail_frac=0.17, res_mode="skilled")
    y, inc, res, avail = d["y"], d["inc"], d["res"], d["avail"]
    assert int(avail.sum()) == int(round(0.17 * d["n"]))

    # Plant hostile payloads on uncovered rows: NaN y (unusable, dropped by
    # fit_gate), NaN incumbents and negative zeros. Bit-identity must carry
    # all of them through untouched — a recomputation or fill would not.
    holes = np.flatnonzero(~avail)[:8]
    y[holes] = np.nan
    inc[holes[:3]] = np.nan
    inc[holes[3:6]] = -0.0

    gr = _fit(d)
    entry = gr.patterns["1"]["0"]
    assert entry["alpha"] > 0.0, (
        "a genuinely skilled 17%-coverage component should open its gate — "
        "otherwise this regression test is vacuous")

    out = _apply(d, gr)
    assert out[~avail].tobytes() == inc[~avail].tobytes(), (
        "composite is not bit-identical to the incumbent on the 83% of rows "
        "the component does not cover — the v1 mean-fill defect is back")
    assert not np.array_equal(out[avail], inc[avail]), (
        "gate open but no covered row was composed")


# ── (2) negative-skill component rejected, decision recorded ────────────────

def test_negative_skill_component_rejected():
    # (a) Harmful everywhere: the selection grid search already refuses it
    # (strict-improvement scan, ties to smaller alpha) and records why.
    d = _synth(avail_frac=0.5, res_mode="anti")
    gr = _fit(d)
    g = gr.patterns["1"][compose.GLOBAL_KEY]
    assert g["alpha"] == 0.0
    assert g["test"]["decision"] == "closed_selection_alpha_zero"
    assert g["test"]["alpha_candidate"] == 0.0
    s = gr.patterns["1"]["0"]
    assert s["alpha"] == 0.0
    assert s["test"]["decision"] == "closed_global_failed"
    assert gr.n_open == 0
    out = _apply(d, gr)
    assert out.tobytes() == d["inc"].tobytes(), (
        "closed gates must be exact passthrough everywhere")

    # (b) Skill that does not generalize: selection likes it (alpha > 0
    # candidate) but the held-out confirmation clusters are hurt — the
    # one-sided paired cluster bootstrap must fail it, alpha forced to 0.
    d2 = _synth(avail_frac=0.5, res_mode="sel_only")
    gr2 = _fit(d2)
    g2 = gr2.patterns["1"][compose.GLOBAL_KEY]
    assert g2["alpha"] == 0.0
    assert g2["test"]["decision"] == "fail"
    assert g2["test"]["alpha_candidate"] is not None
    assert g2["test"]["alpha_candidate"] > 0.0
    assert gr2.n_open == 0
    out2 = _apply(d2, gr2)
    assert out2.tobytes() == d2["inc"].tobytes()


# ── (3) genuinely skilled component admitted, composite beats incumbent ─────

def test_skilled_component_admitted_and_improves():
    d = _synth(avail_frac=0.6, res_mode="skilled")
    gr = _fit(d, margins=PAYLOAD_MARGINS)   # full power_analysis payload form
    assert gr.margins == MARGINS            # resolved from the subdict
    g = gr.patterns["1"][compose.GLOBAL_KEY]
    assert g["test"]["decision"] == "pass"
    assert g["alpha"] > 0.0
    s = gr.patterns["1"]["0"]
    assert s["alpha"] > 0.0
    assert gr.n_open == 1

    out = _apply(d, gr)
    av = d["avail"]
    mse_inc = float(np.mean((d["inc"][av] - d["y"][av]) ** 2))
    mse_out = float(np.mean((out[av] - d["y"][av]) ** 2))
    assert mse_out < mse_inc, "admitted composite must beat the incumbent"
    assert out[~av].tobytes() == d["inc"][~av].tobytes()


# ── (4) admission determinism: same seed -> identical gates.json ────────────

def test_admission_determinism(tmp_path):
    a = _synth(avail_frac=0.5, res_mode="skilled")
    b = _synth(avail_frac=0.5, res_mode="skilled")   # rebuilt, same seed
    gr1 = _fit(a)
    gr2 = _fit(b)
    p1 = str(tmp_path / "gates_run1.json")
    p2 = str(tmp_path / "gates_run2.json")
    compose.save_gates({"tier2": gr1}, p1)
    compose.save_gates({"tier2": gr2}, p2)
    j1 = compose.load_gates(p1)
    j2 = compose.load_gates(p2)
    assert j1 == j2, "same seed + same data must serialize identical gates"
    # And a refit on the very same arrays matches the dataclass too.
    gr3 = _fit(a)
    assert gr3.patterns == gr1.patterns


# ── (5) forbidden-key scanner ───────────────────────────────────────────────

def test_forbidden_key_scanner_rejects_coef(tmp_path):
    good = {"alpha": 0.5, "test": {"decision": "pass"}}
    bad = {"tier2": {"1": {"0": dict(good, coef=1.2)}}}
    path = str(tmp_path / "gates_bad.json")
    with pytest.raises(AssertionError):
        compose.save_gates(bad, path)
    assert not os.path.exists(path), (
        "an invalid gate set must never reach disk")

    # Load side too: a hand-edited gates.json cannot smuggle one in.
    hand = str(tmp_path / "gates_hand.json")
    with open(hand, "w", encoding="utf-8") as fh:
        json.dump({"tier2": {"1": {"0": {
            "alpha": 0.0,
            "test": {"decision": "pass", "coef": 3.0}}}}}, fh)
    with pytest.raises(AssertionError):
        compose.load_gates(hand)


# ── (6) T4 slope clipped into config2.T4_SLOPE_CLIP ─────────────────────────

def test_t4_slope_clipped_on_attenuated_preds(tmp_path):
    rng = np.random.default_rng(config2.SEED)
    n = N_UNITS * N_DAYS
    unit = np.repeat(np.arange(N_UNITS), N_DAYS)
    y = 10.0 + rng.normal(0.0, 3.0, N_UNITS)[unit] + rng.normal(0.0, 1.0, n)
    pred = 0.5 * y + 4.0 + rng.normal(0.0, 0.5, n)   # attenuated: b_raw ~ 1.8
    pred[[5, 250, 999]] = np.nan

    recal, params = compose.t4_recalibrate(y, pred, unit)
    lo, hi = (float(config2.T4_SLOPE_CLIP[0]), float(config2.T4_SLOPE_CLIP[1]))
    assert params["slope_clip"] == [lo, hi]
    for f in params["folds"]:
        assert lo <= f["b"] <= hi, "fitted slope escaped T4_SLOPE_CLIP"
    assert all(f["b_raw"] > hi for f in params["folds"])
    assert any(f["clipped"] and f["b"] == hi for f in params["folds"])

    fin = np.isfinite(pred)
    assert np.isnan(recal[~fin]).all(), "T4 must pass missingness through"
    assert np.isfinite(recal[fin]).all()

    # T4 params deliberately trip the gates.json scanner ("slope" token):
    # its coefficients live in their OWN artifact, never the gate file.
    with pytest.raises(AssertionError):
        compose.save_gates({"t4": params}, str(tmp_path / "gates_t4.json"))


# ── (7) unseen pattern at apply time -> exact passthrough ───────────────────

def test_unseen_pattern_exact_passthrough():
    d = _synth(avail_frac=0.6, res_mode="skilled")
    gr = _fit(d)
    assert gr.patterns["1"]["0"]["alpha"] > 0.0
    n = d["n"]

    # Entirely unseen pattern: default-closed, whole array bit-identical.
    unseen_p = np.full(n, 2, dtype=np.int8)
    out = _apply(d, gr, pattern=unseen_p)
    assert out.tobytes() == d["inc"].tobytes()

    # Unseen STRATUM under a seen pattern: passthrough too — dispatch never
    # falls back to the global alpha (audit-only entry).
    unseen_s = np.full(n, 7, dtype=np.int8)
    out2 = _apply(d, gr, stratum=unseen_s)
    assert out2.tobytes() == d["inc"].tobytes()

    # Mixed: unseen-pattern half untouched, seen half composed.
    mix = d["pattern"].copy()
    mix[: n // 2] = 2
    out3 = _apply(d, gr, pattern=mix)
    assert out3[: n // 2].tobytes() == d["inc"][: n // 2].tobytes()
    assert not np.array_equal(out3[n // 2:], d["inc"][n // 2:])
