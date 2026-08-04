"""
compose.py
AQNet v2 composition harness: per-(pattern, stratum) gates, one-sided paired
cluster-bootstrap admission tests, gates.json serialization, and the declared
T4 slope-recalibration rung.

WHY this module is the most correctness-critical in v2: v1's fusion stage fit
its Ridge combiner on the ~17% of rows where every component happened to be
finite, then MEAN-FILLED the missing U-Net component on the other ~83% at
prediction time. The measured cost was a paired delta-R2 of -0.050/-0.031
with CIs excluding zero and exceedance recall driven to ~0 — a wiring defect,
not a fusion verdict. v2 therefore replaces "learned combiner + fill" with a
structurally monotone residual ladder

    F_k(q) = F_{k-1}(q) + m_k(q) * alpha_k[p(q), s(q)] * r_k(q)

whose three safety properties are enforced HERE, mechanically:

  1. Frozen incumbent. The incumbent enters with coefficient exactly 1.
     apply_gates copies the incumbent array and only ever ADDS a gated
     residual on rows where the component is available and its gate is open.
     gates.json cannot express an incumbent coefficient: save_gates and
     load_gates run a forbidden-key scan (coef/intercept/fill/weight/...)
     over every key at every depth, and apply_gates reads no numeric field
     but "alpha" (clamped to [0, 1]).
  2. Structural zero, never fill. m_k = 0 (avail == 0) yields BIT-IDENTICAL
     passthrough: those rows are never written after the initial copy, so
     even NaN payloads and signed zeros survive untouched. fit_gate and
     apply_gates hard-assert that the residual is NaN on every unavailable
     row AND finite on every available one, so a fill value cannot ride in
     through the residual channel either.
  3. Cross-fit, power-calibrated admission. Alphas are chosen by grid search
     on the inner SELECTION rows and admitted only by a one-sided paired
     unit-cluster bootstrap on the disjoint CONFIRMATION rows (row AND
     cluster disjointness are asserted, not assumed). Admission requires
     non-inferiority on every defined primary metric (pooled R2,
     between-site R2, exceedance F1) within margins injected from
     power_analysis.json, PLUS CI-separated superiority on at least one.
     Default is closed: unseen patterns, unseen strata, strata with fewer
     than config2.GATE_MIN_CLUSTERS held-out clusters, and any admission
     failure all resolve to alpha = 0 (exact passthrough). Small strata
     shrink toward ZERO, never toward the global alpha.

The bootstrap mechanics reuse v1's `_paired_delta_r2_ci` template
(pipeline_colab.py, ablation stage): resample unique clusters with
replacement, score BOTH predictors on the SAME resampled rows, and read the
one-sided bound off the paired delta distribution. Here it is generalized to
the three primary metrics via per-cluster sufficient statistics.

T4 is the sole, declared exception to the frozen-incumbent invariant: an
affine recalibration rung (attenuation fix), cross-fit over clusters, slope
clipped to config2.T4_SLOPE_CLIP with the intercept refit under the clipped
slope. Its parameters live in their own artifact — the gates.json key scan
intentionally rejects them (the "slope_clip" key trips the scanner), so T4
coefficients can never be smuggled into the gate file.

All contract checks raise AssertionError explicitly (never bare `assert`
statements), so `python -O` cannot strip the airlock.

Run from research/aqnet2 (smoke test on synthetic data, no repo files
touched):
    python compose.py
"""

import json
import os
from dataclasses import dataclass

import numpy as np

import config2

# ── Module constants ────────────────────────────────────────────────────────
EXCEED_THRESHOLD = 35.4  # µg/m³, USG boundary — v1 validation.py convention
GLOBAL_KEY = "__global__"
PRIMARY_METRICS = ("pooled_r2", "spatial_r2", "exceedance_f1")
# Safe-default non-inferiority margins (max tolerated degradation) used when
# power_analysis.json margins are absent. Deliberately TIGHT: a missing power
# analysis must make admission harder, never easier.
DEFAULT_MARGINS = {"pooled_r2": 0.01, "spatial_r2": 0.01,
                   "exceedance_f1": 0.02}
# Any gates.json key containing one of these tokens (case-insensitive, any
# depth) is rejected on save AND load — the schema is structurally unable to
# express an incumbent coefficient or a fill value.
_FORBIDDEN_KEY_TOKENS = ("coef", "intercept", "fill", "weight", "beta",
                         "scale", "slope", "bias", "offset", "gain")


# ── Array coercion helpers ──────────────────────────────────────────────────
def _as_f8(x, name):
    """1-D float64 view/copy of x; raises on non-1-D input."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    return arr


def _as_bool(x, name, n):
    arr = np.asarray(x)
    if arr.ndim != 1 or len(arr) != n:
        raise ValueError(f"{name} must be 1-D of length {n}, "
                         f"got shape {arr.shape}")
    return arr.astype(bool)


def _check_len(arr, n, name):
    if len(arr) != n:
        raise ValueError(f"{name} has length {len(arr)}, expected {n}")


def _gate_key(v):
    """Canonical string key for a pattern/stratum id.

    fit_gate and apply_gates MUST agree on this mapping or a fitted gate
    would silently fail to match at apply time (default-closed would mask
    the bug as passthrough). Integer-valued floats collapse to their int
    form so np.int8(3), 3 and 3.0 all key as "3".
    """
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        fv = float(v)
        if np.isfinite(fv) and fv.is_integer():
            return str(int(fv))
        return repr(fv)
    return str(v)


def _jsonable(obj):
    """Recursively convert to JSON-safe python types.

    numpy scalars -> python scalars; non-finite floats -> None (gates.json
    must be strict-JSON loadable everywhere); arrays/tuples -> lists.
    """
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if np.isfinite(f) else None
    return obj


# ── Margins (power_analysis.json injection with safe defaults) ──────────────
def _resolve_margins(margins):
    """Normalize the margins argument to {metric: non-negative float}.

    Accepts the full power_analysis.json dict (uses its "margins" subdict),
    a bare margins dict, or None. Missing metrics fall back to
    DEFAULT_MARGINS — tight on purpose (see module constant).
    """
    if margins is None:
        print("[admission] no margins provided — using tight safe defaults "
              f"{DEFAULT_MARGINS}")
        src = {}
    elif isinstance(margins, dict) and isinstance(margins.get("margins"),
                                                  dict):
        src = margins["margins"]
    elif isinstance(margins, dict):
        src = margins
    else:
        raise ValueError("margins must be a dict (power_analysis.json "
                         "payload or its 'margins' subdict) or None")
    out = {}
    for m in PRIMARY_METRICS:
        v = float(src.get(m, DEFAULT_MARGINS[m]))
        if not np.isfinite(v) or v < 0.0:
            raise AssertionError(
                f"admission margin for '{m}' must be a finite non-negative "
                f"float, got {v} — a negative margin would invert the "
                "non-inferiority test")
        out[m] = v
    return out


# ── Airlock asserts (shared by fit_gate and apply_gates) ────────────────────
def _assert_residual_airlock(residual, avail, where):
    """The structural-zero contract: NaN outside coverage, finite inside.

    A finite residual on an unavailable row means some upstream path
    manufactured a value where the component has none — the exact v1
    mean-fill defect. A NaN residual on an available row means the
    availability mask lies. Both are hard errors, never warnings.
    """
    bad_fill = np.isfinite(residual[~avail])
    if bad_fill.any():
        raise AssertionError(
            f"{where}: {int(bad_fill.sum())} unavailable rows carry a "
            "finite residual — a fill value entered the composition path "
            "(residual_oof must be NaN wherever avail == 0)")
    bad_hole = ~np.isfinite(residual[avail])
    if bad_hole.any():
        raise AssertionError(
            f"{where}: {int(bad_hole.sum())} available rows carry a "
            "non-finite residual — avail == 1 requires a finite component "
            "value")


# ── Admission test: one-sided paired unit-cluster bootstrap ─────────────────
def _f1(tp, fp, fn):
    """F1 from counts; NaN when undefined (no true or predicted positives)."""
    denom = 2.0 * tp + fp + fn
    return float("nan") if denom <= 0 else 2.0 * tp / denom


def admission_test(y, pred_a, pred_b, clusters, margins,
                   n_boot=1000, seed=42, exceed_valid=None):
    """One-sided paired unit-cluster bootstrap admission test.

    pred_a is the incumbent (reference), pred_b the challenger composite.
    Unique clusters (units) are resampled with replacement n_boot times;
    both predictors are scored on the SAME resampled rows (v1
    `_paired_delta_r2_ci` template), so the delta distribution reflects the
    paired error structure. Primary metrics:

      pooled_r2      pooled R2 over rows
      spatial_r2     between-cluster R2 (per-cluster mean pred vs mean y —
                     the exposure-assessment question)
      exceedance_f1  F1 of the y > 35.4 µg/m³ event

    Pass requires, on every DEFINED metric, one-sided non-inferiority
    (5th-percentile paired delta > -margin) AND CI-separated superiority
    (5th-percentile delta > 0) on at least one. pooled_r2 must be defined
    or the test fails outright. A metric is "defined" when its point delta
    is finite and at least half the bootstrap draws could evaluate it
    (rare-event F1 in a sparse stratum honestly reports as undefined and is
    skipped, recorded in "skipped"). Default is closed: every degenerate
    input path returns decision "fail".

    margins: power_analysis.json payload / margins subdict / None (safe
    defaults). Returns a JSON-safe dict with headline "delta"/"ci" (pooled
    R2) plus per-metric detail — exactly what gates.json embeds as "test".

    exceed_valid: optional bool array (len(y)) marking rows whose exceedance
    LABEL is trustworthy. Rows with exceed_valid False keep contributing to
    the R2 metrics but are removed from the exceedance tp/fp/fn counts —
    the DESIGN §4 rule that channel-reconstructed PA rows never define
    exceedance events (their high-PM values are the reconstructed ATM
    channel, biased exactly in the event regime). With every positive
    invalid, exceedance_f1 honestly reports as undefined and is skipped.
    """
    y = _as_f8(y, "y")
    pa = _as_f8(pred_a, "pred_a")
    pb = _as_f8(pred_b, "pred_b")
    cl = np.asarray(clusters)
    n_in = len(y)
    _check_len(pa, n_in, "pred_a")
    _check_len(pb, n_in, "pred_b")
    _check_len(cl, n_in, "clusters")
    if exceed_valid is None:
        ev = np.ones(n_in, dtype=bool)
    else:
        ev = np.asarray(exceed_valid).astype(bool)
        _check_len(ev, n_in, "exceed_valid")
    margins = _resolve_margins(margins)

    base = {"decision": "fail", "delta": None, "ci": [None, None],
            "lb95": None, "n_clusters": 0, "n_rows": 0,
            "n_boot": int(n_boot), "seed": int(seed), "margins": margins,
            "metrics": {}, "skipped": list(PRIMARY_METRICS), "reasons": []}

    ok = np.isfinite(y) & np.isfinite(pa) & np.isfinite(pb)
    if int(ok.sum()) < 3:
        base["reasons"].append(f"only {int(ok.sum())} jointly-finite rows")
        return _jsonable(base)
    y, pa, pb, cl, ev = y[ok], pa[ok], pb[ok], cl[ok], ev[ok]
    n = len(y)

    uniq, inv = np.unique(cl, return_inverse=True)
    n_cl = len(uniq)
    if n_cl < 2:
        base["reasons"].append(f"only {n_cl} cluster(s) — paired cluster "
                               "bootstrap needs >= 2")
        base["n_rows"] = n
        return _jsonable(base)

    # Per-cluster sufficient statistics: every bootstrap metric is a pure
    # function of these sums, so 1000 draws cost 1000 fancy-index reductions.
    n_c = np.bincount(inv).astype(np.float64)
    sy = np.bincount(inv, weights=y)
    syy = np.bincount(inv, weights=y * y)
    sse_a = np.bincount(inv, weights=(pa - y) ** 2)
    sse_b = np.bincount(inv, weights=(pb - y) ** 2)
    my = sy / n_c
    ma = np.bincount(inv, weights=pa) / n_c
    mb = np.bincount(inv, weights=pb) / n_c
    # Exceedance counts only on label-valid rows (exceed_valid semantics in
    # the docstring); an all-invalid input zeroes every count and the F1
    # metric reports undefined -> skipped, never a fabricated 0-vs-0 tie.
    exc_t = (y > EXCEED_THRESHOLD) & ev
    exc_a = (pa > EXCEED_THRESHOLD) & ev
    exc_b = (pb > EXCEED_THRESHOLD) & ev
    n_ev = int(ev.sum())
    tp_a = np.bincount(inv, weights=(exc_a & exc_t).astype(np.float64))
    fp_a = np.bincount(inv, weights=(exc_a & ~exc_t & ev).astype(np.float64))
    fn_a = np.bincount(inv, weights=(~exc_a & exc_t).astype(np.float64))
    tp_b = np.bincount(inv, weights=(exc_b & exc_t).astype(np.float64))
    fp_b = np.bincount(inv, weights=(exc_b & ~exc_t & ev).astype(np.float64))
    fn_b = np.bincount(inv, weights=(~exc_b & exc_t).astype(np.float64))

    # Point estimates on the full confirmation rows.
    point = {}
    ss_tot = float(syy.sum() - sy.sum() ** 2 / n)
    if ss_tot > 0:
        r2a = 1.0 - float(sse_a.sum()) / ss_tot
        r2b = 1.0 - float(sse_b.sum()) / ss_tot
        point["pooled_r2"] = (r2a, r2b)
    ss_sp = float(np.sum((my - my.mean()) ** 2))
    if ss_sp > 0:
        point["spatial_r2"] = (1.0 - float(np.sum((ma - my) ** 2)) / ss_sp,
                               1.0 - float(np.sum((mb - my) ** 2)) / ss_sp)
    f1a = _f1(tp_a.sum(), fp_a.sum(), fn_a.sum())
    f1b = _f1(tp_b.sum(), fp_b.sum(), fn_b.sum())
    if np.isfinite(f1a) and np.isfinite(f1b):
        point["exceedance_f1"] = (f1a, f1b)

    # Paired cluster bootstrap (seeded; identical rows for both predictors).
    rng = np.random.default_rng(seed)
    boot = {m: np.full(n_boot, np.nan) for m in PRIMARY_METRICS}
    for b in range(int(n_boot)):
        pick = rng.integers(0, n_cl, n_cl)
        nb = n_c[pick].sum()
        syb = sy[pick].sum()
        sst = syy[pick].sum() - syb * syb / nb
        if sst > 0:
            boot["pooled_r2"][b] = (sse_a[pick].sum()
                                    - sse_b[pick].sum()) / sst
        myp = my[pick]
        ssp = np.sum((myp - myp.mean()) ** 2)
        if ssp > 0:
            boot["spatial_r2"][b] = (np.sum((ma[pick] - myp) ** 2)
                                     - np.sum((mb[pick] - myp) ** 2)) / ssp
        f1ab = _f1(tp_a[pick].sum(), fp_a[pick].sum(), fn_a[pick].sum())
        f1bb = _f1(tp_b[pick].sum(), fp_b[pick].sum(), fn_b[pick].sum())
        if np.isfinite(f1ab) and np.isfinite(f1bb):
            boot["exceedance_f1"][b] = f1bb - f1ab

    metrics, skipped, reasons = {}, [], []
    for m in PRIMARY_METRICS:
        draws = boot[m]
        fin = np.isfinite(draws)
        n_fin = int(fin.sum())
        pt = point.get(m)
        delta = (pt[1] - pt[0]) if pt is not None else float("nan")
        defined = bool(np.isfinite(delta) and n_fin >= int(n_boot) // 2)
        lb = ub = float("nan")
        if n_fin:
            lb, ub = np.percentile(draws[fin], [5.0, 95.0])
        det = {"delta": float(delta) if np.isfinite(delta) else None,
               "lb95": float(lb) if np.isfinite(lb) else None,
               "ci": [float(lb) if np.isfinite(lb) else None,
                      float(ub) if np.isfinite(ub) else None],
               "margin": margins[m],
               "point_ref": pt[0] if pt is not None else None,
               "point_new": pt[1] if pt is not None else None,
               "n_boot_defined": n_fin,
               "defined": defined,
               "non_inferior": bool(lb > -margins[m]) if defined else None,
               "superior": bool(lb > 0.0) if defined else None}
        metrics[m] = det
        if not defined:
            skipped.append(m)

    defined_ms = [m for m in PRIMARY_METRICS if metrics[m]["defined"]]
    if "pooled_r2" not in defined_ms:
        decision = "fail"
        reasons.append("pooled_r2 undefined (degenerate target variance or "
                       "unstable bootstrap) — default closed")
    else:
        not_ni = [m for m in defined_ms if not metrics[m]["non_inferior"]]
        any_sup = any(metrics[m]["superior"] for m in defined_ms)
        if not_ni:
            decision = "fail"
            reasons.append("non-inferiority failed on: " + ", ".join(not_ni))
        elif not any_sup:
            decision = "fail"
            reasons.append("no CI-separated superiority on any defined "
                           "primary metric")
        else:
            decision = "pass"

    head = metrics["pooled_r2"]
    out = {"decision": decision,
           "delta": head["delta"],
           "ci": list(head["ci"]),
           "lb95": head["lb95"],
           "n_clusters": int(n_cl),
           "n_rows": int(n),
           "n_exceed_valid": int(n_ev),
           "n_boot": int(n_boot),
           "seed": int(seed),
           "margins": margins,
           "metrics": metrics,
           "skipped": skipped,
           "reasons": reasons}
    return _jsonable(out)


# ── Alpha grid search (selection rows only) ─────────────────────────────────
def _grid_search_alpha(y, inc, res, mask):
    """SSE-minimizing alpha over config2.GATE_ALPHA_GRID on `mask` rows.

    The grid is scanned in ascending order with a STRICT improvement rule,
    so ties resolve to the smaller alpha — the default-closed bias applies
    inside the grid search too. 0.0 is always a candidate.
    """
    grid = sorted({float(a) for a in config2.GATE_ALPHA_GRID} | {0.0})
    for a in grid:
        if not 0.0 <= a <= 1.0:
            raise AssertionError(
                f"GATE_ALPHA_GRID entry {a} outside [0, 1] — gates cannot "
                "rescale the incumbent or overshoot the residual")
    yy, ii, rr = y[mask], inc[mask], res[mask]
    best_alpha, best_sse = 0.0, np.inf
    for a in grid:
        sse = float(np.sum((yy - (ii + a * rr)) ** 2))
        if sse < best_sse:
            best_sse, best_alpha = sse, a
    return best_alpha


def _closed_entry(decision, n_conf_clusters, alpha_candidate=None):
    """A gate entry closed WITHOUT an admission run (schema-complete)."""
    return {"alpha": 0.0,
            "test": {"delta": None, "ci": [None, None],
                     "n_clusters": int(n_conf_clusters),
                     "decision": str(decision),
                     "alpha_candidate": alpha_candidate}}


# ── fit_gate ────────────────────────────────────────────────────────────────
@dataclass
class GateResult:
    """Fitted gates for ONE tier, ready for gates.json.

    patterns    {pattern_key: {"__global__": entry, stratum_key: entry}}
                where entry = {"alpha": float in [0,1], "test": dict}.
                This is exactly the per-tier subdict of gates.json; assemble
                the artifact as {"tier2": result.patterns, ...}.
    margins     resolved non-inferiority margins used by every admission run
    alpha_grid  the candidate grid searched on selection rows
    n_sel_rows / n_conf_rows   usable (finite-y) row counts per half
    n_open      stratum entries (excluding __global__) with alpha > 0
    """
    patterns: dict
    margins: dict
    alpha_grid: list
    n_sel_rows: int
    n_conf_rows: int
    n_open: int


def fit_gate(y, incumbent_oof, residual_oof, avail, pattern_id, stratum_id,
             clusters, sel_mask, conf_mask, margins, exceed_valid=None):
    """Fit per-(pattern, stratum) gates for one tier's residual component.

    y             target (FRM scale), full frame length
    incumbent_oof F_{k-1} out-of-fold predictions (the frozen incumbent)
    residual_oof  r_k out-of-fold predictions; MUST be NaN wherever
                  avail == 0 and finite wherever avail == 1 (asserted)
    avail         {0,1} component availability mask m_k
    pattern_id    coverage-pattern id per row (drives alpha dispatch)
    stratum_id    stratum id per row (regime/density refinement)
    clusters      unit ids (sensors/sites) — the bootstrap resampling unit
    sel_mask      inner SELECTION rows (alpha grid search happens here)
    conf_mask     inner CONFIRMATION rows (admission happens ONLY here)
    margins       power_analysis.json payload / margins subdict / None

    Hierarchy per pattern: a GLOBAL gate (all strata pooled) is grid-searched
    on selection rows and admission-tested on confirmation rows. Strata are
    tested ONLY if the global gate passes; each stratum then gets its own
    grid search + admission on its own (pattern, stratum) rows. A stratum
    with fewer than config2.GATE_MIN_CLUSTERS held-out confirmation clusters
    is forced to alpha = 0 (shrink toward passthrough, NEVER toward the
    global alpha). Rows are only ever gated by their stratum entry — the
    "__global__" entry is the hierarchical gatekeeper and audit record, not
    a fallback alpha.

    Leakage asserts (all raise AssertionError):
      - residual airlock (NaN outside coverage, finite inside)
      - sel/conf share no rows AND no clusters (no gate is tested on rows —
        or units — that chose it)
      - incumbent finite on every usable selection/confirmation row

    Returns a GateResult; apply with apply_gates(..., result.patterns).
    """
    y = _as_f8(y, "y")
    n = len(y)
    inc = _as_f8(incumbent_oof, "incumbent_oof")
    res = _as_f8(residual_oof, "residual_oof")
    _check_len(inc, n, "incumbent_oof")
    _check_len(res, n, "residual_oof")
    av = _as_bool(avail, "avail", n)
    sel = _as_bool(sel_mask, "sel_mask", n)
    conf = _as_bool(conf_mask, "conf_mask", n)
    pattern = np.asarray(pattern_id)
    stratum = np.asarray(stratum_id)
    cl = np.asarray(clusters)
    _check_len(pattern, n, "pattern_id")
    _check_len(stratum, n, "stratum_id")
    _check_len(cl, n, "clusters")
    if exceed_valid is None:
        ev_all = np.ones(n, dtype=bool)
    else:
        ev_all = np.asarray(exceed_valid).astype(bool)
        _check_len(ev_all, n, "exceed_valid")
    margins = _resolve_margins(margins)

    # ── Airlock + selection/confirmation separation ──
    _assert_residual_airlock(res, av, "fit_gate")
    overlap = sel & conf
    if overlap.any():
        raise AssertionError(
            f"fit_gate: {int(overlap.sum())} rows are in BOTH selection and "
            "confirmation — no gate may be tested on rows that chose it")
    shared = np.intersect1d(np.unique(cl[sel]), np.unique(cl[conf]))
    if len(shared):
        raise AssertionError(
            f"fit_gate: {len(shared)} clusters appear in both selection and "
            f"confirmation halves (e.g. {shared[:5].tolist()}) — inner "
            "folds are unit-grouped; a shared unit leaks selection "
            "information into the admission test")

    usable = np.isfinite(y)
    n_dropped = int(((sel | conf) & ~usable).sum())
    if n_dropped:
        print(f"[fit_gate] dropping {n_dropped} sel/conf rows with "
              "non-finite y")
    sel_u = sel & usable
    conf_u = conf & usable
    if not np.isfinite(inc[sel_u | conf_u]).all():
        raise AssertionError(
            "fit_gate: incumbent_oof is non-finite on usable "
            "selection/confirmation rows — the ladder floor must cover "
            "100% of scored rows")

    grid = sorted({float(a) for a in config2.GATE_ALPHA_GRID} | {0.0})
    min_cl = int(config2.GATE_MIN_CLUSTERS)
    pats_out = {}
    n_open = 0

    pat_ids = np.unique(pattern[av & (sel_u | conf_u)])
    if len(pat_ids) == 0:
        print("[fit_gate] no available rows in selection/confirmation — "
              "all gates closed (passthrough tier)")
    for p in pat_ids:
        in_p = av & (pattern == p)
        sel_p = in_p & sel_u
        conf_p = in_p & conf_u
        n_cc = len(np.unique(cl[conf_p]))
        entry_map = {}

        # ── Global (pattern-level) gate ──
        if not sel_p.any():
            g_entry = _closed_entry("closed_no_selection_rows", n_cc)
        else:
            alpha_g = _grid_search_alpha(y, inc, res, sel_p)
            if alpha_g <= 0.0:
                g_entry = _closed_entry("closed_selection_alpha_zero", n_cc,
                                        alpha_candidate=alpha_g)
            elif n_cc < min_cl:
                g_entry = _closed_entry("closed_min_clusters", n_cc,
                                        alpha_candidate=alpha_g)
            else:
                challenger = inc[conf_p] + alpha_g * res[conf_p]
                t = admission_test(y[conf_p], inc[conf_p], challenger,
                                   cl[conf_p], margins,
                                   n_boot=1000, seed=config2.SEED,
                                   exceed_valid=ev_all[conf_p])
                t["alpha_candidate"] = alpha_g
                g_entry = {"alpha": alpha_g if t["decision"] == "pass"
                           else 0.0, "test": t}
        g_open = g_entry["alpha"] > 0.0
        entry_map[GLOBAL_KEY] = g_entry
        print(f"[fit_gate] pattern {_gate_key(p)}: global "
              f"decision={g_entry['test']['decision']} "
              f"alpha={g_entry['alpha']:g} conf_clusters={n_cc}")

        # ── Strata: tested only if the global gate passed ──
        strat_ids = np.unique(stratum[in_p & (sel_u | conf_u)])
        for s in strat_ids:
            if _gate_key(s) == GLOBAL_KEY:
                raise AssertionError(
                    f"fit_gate: stratum id {s!r} collides with the reserved "
                    f"'{GLOBAL_KEY}' key — rename the stratum; the global "
                    "entry is an audit record, never a dispatch target")
            in_ps = in_p & (stratum == s)
            sel_ps = in_ps & sel_u
            conf_ps = in_ps & conf_u
            n_scc = len(np.unique(cl[conf_ps]))
            if not g_open:
                s_entry = _closed_entry("closed_global_failed", n_scc)
            elif n_scc < min_cl:
                # Shrink toward ZERO, never toward the global alpha.
                s_entry = _closed_entry("closed_min_clusters", n_scc)
            elif not sel_ps.any():
                s_entry = _closed_entry("closed_no_selection_rows", n_scc)
            else:
                alpha_s = _grid_search_alpha(y, inc, res, sel_ps)
                if alpha_s <= 0.0:
                    s_entry = _closed_entry("closed_selection_alpha_zero",
                                            n_scc, alpha_candidate=alpha_s)
                else:
                    challenger = inc[conf_ps] + alpha_s * res[conf_ps]
                    t = admission_test(y[conf_ps], inc[conf_ps], challenger,
                                       cl[conf_ps], margins,
                                       n_boot=1000, seed=config2.SEED,
                                       exceed_valid=ev_all[conf_ps])
                    t["alpha_candidate"] = alpha_s
                    s_entry = {"alpha": alpha_s if t["decision"] == "pass"
                               else 0.0, "test": t}
            if s_entry["alpha"] > 0.0:
                n_open += 1
            entry_map[_gate_key(s)] = s_entry
            print(f"[fit_gate]   stratum {_gate_key(s)}: "
                  f"decision={s_entry['test']['decision']} "
                  f"alpha={s_entry['alpha']:g} conf_clusters={n_scc}")
        pats_out[_gate_key(p)] = entry_map

    n_strata = sum(len(v) - 1 for v in pats_out.values())
    print(f"[fit_gate] {len(pats_out)} patterns, {n_open}/{max(n_strata, 0)} "
          f"strata open, sel rows {int(sel_u.sum()):,}, "
          f"conf rows {int(conf_u.sum()):,}")
    return GateResult(patterns=_jsonable(pats_out), margins=margins,
                      alpha_grid=list(grid), n_sel_rows=int(sel_u.sum()),
                      n_conf_rows=int(conf_u.sum()), n_open=int(n_open))


# ── apply_gates ─────────────────────────────────────────────────────────────
def _tier_map(gates):
    """Normalize the gates argument to a per-tier {pattern: {stratum: entry}}.

    Accepts a GateResult or the per-tier dict. Passing the FULL gates.json
    dict ({tier: {pattern: ...}}) is detected by shape and rejected loudly —
    under default-closed it would otherwise silently gate everything to
    passthrough, masking the wiring bug.
    """
    g = getattr(gates, "patterns", gates)
    if not isinstance(g, dict):
        raise ValueError("gates must be a GateResult or a per-tier dict "
                         "{pattern: {stratum: {'alpha': ...}}}")
    for pat_map in g.values():
        if not isinstance(pat_map, dict):
            raise ValueError("malformed gates dict (pattern value is not "
                             "a dict)")
        for entry in pat_map.values():
            if isinstance(entry, dict) and "alpha" not in entry:
                raise ValueError(
                    "apply_gates received what looks like the FULL "
                    "gates.json dict ({tier: {pattern: ...}}); pass "
                    "gates[tier] — the per-tier pattern map — instead "
                    "(default-closed would silently passthrough otherwise)")
    return g


def _alpha_row(gmap, av, pattern, stratum, n):
    """Per-row alpha under strict default-closed dispatch.

    Unseen pattern, unseen stratum, and closed gates all yield 0.0. Rows
    dispatch ONLY on their stratum entry — the reserved '__global__' audit
    entry is skipped even if a stratum id collides with it. The only numeric
    field ever read is "alpha", asserted into [0, 1].
    """
    alpha_row = np.zeros(n, dtype=np.float64)
    for p in np.unique(pattern[av]):
        pat_map = gmap.get(_gate_key(p))
        if not pat_map:
            continue  # unseen pattern -> exact passthrough (default closed)
        rows_p = av & (pattern == p)
        for s in np.unique(stratum[rows_p]):
            key = _gate_key(s)
            if key == GLOBAL_KEY:
                continue  # audit entry, never a dispatch target
            entry = pat_map.get(key)
            if entry is None:
                continue  # unseen stratum -> passthrough (never global alpha)
            a = float(entry["alpha"])
            if not np.isfinite(a) or not 0.0 <= a <= 1.0:
                raise AssertionError(
                    f"apply_gates: alpha {a} for pattern {_gate_key(p)} "
                    f"stratum {key} outside [0, 1] — gates cannot "
                    "rescale the incumbent")
            if a > 0.0:
                alpha_row[rows_p & (stratum == s)] = a
    return alpha_row


def apply_gates(incumbent, residual, avail, pattern_id, stratum_id, gates):
    """Compose one tier: incumbent + alpha[pattern, stratum] * residual.

    Returns a NEW array. Rows with avail == 0, rows whose (pattern, stratum)
    is unseen in `gates`, and rows whose gate is closed (alpha == 0) carry
    the incumbent BIT-IDENTICALLY: they are never written after the initial
    copy, never recomputed, never filled. The residual airlock asserts run
    here too — serving must obey the same contract as fitting.

    gates: GateResult or the per-tier dict (gates_json[tier]). The only
    numeric field ever read is "alpha", clamped-asserted to [0, 1]; nothing
    else in the file can influence the composition.
    """
    inc = _as_f8(incumbent, "incumbent")
    n = len(inc)
    res = _as_f8(residual, "residual")
    _check_len(res, n, "residual")
    av = _as_bool(avail, "avail", n)
    pattern = np.asarray(pattern_id)
    stratum = np.asarray(stratum_id)
    _check_len(pattern, n, "pattern_id")
    _check_len(stratum, n, "stratum_id")
    gmap = _tier_map(gates)

    _assert_residual_airlock(res, av, "apply_gates")

    out = inc.copy()  # bit-preserving; untouched rows stay bit-identical
    if not av.any():
        return out

    alpha_row = _alpha_row(gmap, av, pattern, stratum, n)
    mod = av & (alpha_row > 0.0)
    if mod.any():
        if not np.isfinite(inc[mod]).all():
            raise AssertionError(
                "apply_gates: non-finite incumbent on rows selected for "
                "composition — the frozen incumbent must be finite wherever "
                "a gate opens")
        out[mod] = inc[mod] + alpha_row[mod] * res[mod]
    return out


# ── gates.json save/load ────────────────────────────────────────────────────
def _scan_forbidden_keys(obj, trail="gates"):
    """Reject any key (any depth) resembling an incumbent coefficient or
    fill value. Runs on save AND load, so a hand-edited gates.json cannot
    smuggle one in either."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            for tok in _FORBIDDEN_KEY_TOKENS:
                if tok in kl:
                    raise AssertionError(
                        f"gates.json airlock: key '{trail}.{k}' resembles "
                        f"an incumbent coefficient / fill value (token "
                        f"'{tok}') — the gate schema is structurally "
                        "forbidden from expressing one")
            _scan_forbidden_keys(v, f"{trail}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _scan_forbidden_keys(v, f"{trail}[{i}]")


def _validate_gates(gates):
    """Structural validation of {tier: {pattern: {stratum: entry}}}."""
    if not isinstance(gates, dict):
        raise AssertionError("gates.json payload must be a dict "
                             "{tier: {pattern: {stratum: entry}}}")
    _scan_forbidden_keys(gates)
    for tier, pats in gates.items():
        if not isinstance(pats, dict):
            raise AssertionError(f"gates.json: tier '{tier}' must map to a "
                                 "dict of patterns")
        for pat, strata in pats.items():
            if not isinstance(strata, dict):
                raise AssertionError(f"gates.json: '{tier}.{pat}' must map "
                                     "to a dict of strata")
            for strat, entry in strata.items():
                where = f"'{tier}.{pat}.{strat}'"
                if (not isinstance(entry, dict) or "alpha" not in entry
                        or "test" not in entry):
                    raise AssertionError(
                        f"gates.json: {where} must be an entry dict with "
                        "'alpha' and 'test' keys")
                a = entry["alpha"]
                if (isinstance(a, bool) or not isinstance(a, (int, float))
                        or not np.isfinite(float(a))
                        or not 0.0 <= float(a) <= 1.0):
                    raise AssertionError(
                        f"gates.json: {where} alpha {a!r} must be a finite "
                        "number in [0, 1]")
                t = entry["test"]
                if not isinstance(t, dict) or "decision" not in t:
                    raise AssertionError(
                        f"gates.json: {where} test must be a dict carrying "
                        "a 'decision' string")


def save_gates(gates, path):
    """Write gates.json atomically (tmp + os.replace, v1 checkpoint style).

    gates: {tier: GateResult-or-per-tier-dict}. Validation (forbidden-key
    scan, alpha bounds, schema shape) runs BEFORE the write — an invalid
    gate set never reaches disk.
    """
    norm = {str(t): getattr(v, "patterns", v) for t, v in gates.items()}
    _validate_gates(norm)
    payload = _jsonable(norm)
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)
    n_pat = sum(len(v) for v in payload.values())
    print(f"[compose] wrote {path} ({len(payload)} tiers, {n_pat} patterns)")
    return path


def load_gates(path):
    """Load and re-validate gates.json.

    The load-side validation is the last line of defense: it asserts that
    no key resembling an incumbent coefficient or fill value exists anywhere
    in the file, that every alpha is a finite number in [0, 1], and that the
    {tier: {pattern: {stratum: entry}}} shape holds. Returns the dict; pass
    gates[tier] to apply_gates.
    """
    with open(path, "r", encoding="utf-8") as fh:
        gates = json.load(fh)
    _validate_gates(gates)
    return gates


# ── T4: the declared slope-recalibration rung ───────────────────────────────
def t4_recalibrate(y, pred, clusters, n_folds=4, seed=config2.SEED):
    """Cross-fit affine recalibration — the sole, DECLARED exception to the
    frozen-incumbent invariant (DESIGN §1): an attenuation fix, documented
    as a reweighting rung rather than smuggled through a gate.

    Unique clusters are shuffled (seeded) and dealt into n_folds groups.
    For each fold the affine map y ~ a + b * pred is fit by least squares on
    the OTHER folds' rows only, the slope b is clipped to
    config2.T4_SLOPE_CLIP, the intercept is REFIT under the clipped slope
    (a = mean(y) - b_clipped * mean(pred), so clipping cannot introduce a
    bias), and the map is applied to the held-out fold's rows only — every
    recalibrated value comes from parameters that never saw its cluster.

    Degenerate folds (fewer than 3 usable training rows, or ~zero predictor
    variance) fall back to the identity map (a=0, b=1), recorded per fold.

    Returns (recal, params):
      recal   full-length array; NaN exactly where pred is non-finite
              (never fills) — a passthrough of missingness, not a value.
      params  JSON-safe dict {"rung": "T4", "slope_clip": [lo, hi],
              "n_folds", "seed", "folds": [{"a", "b", "b_raw", "clipped",
              "n_train_rows", "n_apply_rows", "note"}...]}. Persist it in
              its OWN artifact: the gates.json forbidden-key scan rejects
              "slope_clip" by design, so T4 coefficients cannot ride into
              the gate file.
    """
    y = _as_f8(y, "y")
    n = len(y)
    pr = _as_f8(pred, "pred")
    _check_len(pr, n, "pred")
    cl = np.asarray(clusters)
    _check_len(cl, n, "clusters")
    lo, hi = (float(config2.T4_SLOPE_CLIP[0]),
              float(config2.T4_SLOPE_CLIP[1]))
    if not (0.0 < lo <= 1.0 <= hi):
        raise AssertionError(
            f"T4_SLOPE_CLIP {config2.T4_SLOPE_CLIP} must bracket 1.0 with a "
            "positive lower bound — the rung is a recalibration, not a "
            "free rescale")

    ok = np.isfinite(y) & np.isfinite(pr)
    uniq = np.unique(cl)
    if len(uniq) < 2:
        raise ValueError("t4_recalibrate: need >= 2 clusters for cross-fit")
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_folds = int(min(n_folds, len(uniq)))

    out = np.full(n, np.nan)
    folds_meta = []
    for fi, grp in enumerate(np.array_split(uniq, n_folds)):
        te = np.isin(cl, grp)
        tr = ok & ~te
        note = "ok"
        if int(tr.sum()) < 3:
            a_, b_raw, b_ = 0.0, 1.0, 1.0
            note = "identity_insufficient_rows"
        else:
            px, ty = pr[tr], y[tr]
            dx = px - px.mean()
            ssx = float(np.sum(dx * dx))
            if ssx <= 1e-12:
                a_, b_raw, b_ = 0.0, 1.0, 1.0
                note = "identity_degenerate_variance"
            else:
                b_raw = float(np.sum(dx * (ty - ty.mean())) / ssx)
                b_ = min(max(b_raw, lo), hi)
                a_ = float(ty.mean() - b_ * px.mean())
        apply_rows = te & np.isfinite(pr)
        out[apply_rows] = a_ + b_ * pr[apply_rows]
        folds_meta.append({"a": float(a_), "b": float(b_),
                           "b_raw": float(b_raw),
                           "clipped": bool(b_ != b_raw),
                           "n_train_rows": int(tr.sum()),
                           "n_apply_rows": int(apply_rows.sum()),
                           "note": note})
    n_clipped = sum(1 for f in folds_meta if f["clipped"])
    print(f"[t4] cross-fit affine recalibration: {n_folds} folds, "
          f"{n_clipped} slope(s) clipped to [{lo:g}, {hi:g}]")
    params = {"rung": "T4",
              "declared": "affine slope-recalibration — the sole declared "
                          "exception to the frozen-incumbent invariant",
              "slope_clip": [lo, hi],
              "n_folds": n_folds,
              "seed": int(seed),
              "folds": folds_meta}
    return out, _jsonable(params)


# ── Smoke test (synthetic data — no repo files touched) ─────────────────────
if __name__ == "__main__":
    import tempfile

    rng = np.random.default_rng(config2.SEED)
    n_units, n_days = 40, 30
    n = n_units * n_days
    unit = np.repeat(np.arange(n_units), n_days)
    site_eff = rng.normal(0.0, 3.0, n_units)
    y_demo = 10.0 + site_eff[unit] + rng.normal(0.0, 1.0, n)
    signal = 0.8 * site_eff[unit]
    inc_demo = y_demo - signal + rng.normal(0.0, 0.3, n)

    avail_demo = rng.random(n) < 0.5
    res_demo = np.full(n, np.nan)
    res_demo[avail_demo] = signal[avail_demo] + rng.normal(
        0.0, 0.2, int(avail_demo.sum()))
    pat = np.ones(n, dtype=np.int8)
    strat = np.zeros(n, dtype=np.int8)
    sel = unit < n_units // 2
    conf = ~sel

    gr = fit_gate(y_demo, inc_demo, res_demo, avail_demo, pat, strat,
                  unit, sel, conf, margins=None)
    comp = apply_gates(inc_demo, res_demo, avail_demo, pat, strat, gr)
    same = comp[~avail_demo].tobytes() == inc_demo[~avail_demo].tobytes()
    print(f"[smoke] passthrough bit-identical on uncovered rows: {same}")
    mse_inc = float(np.mean((inc_demo[avail_demo] - y_demo[avail_demo]) ** 2))
    mse_cmp = float(np.mean((comp[avail_demo] - y_demo[avail_demo]) ** 2))
    print(f"[smoke] covered-row MSE incumbent={mse_inc:.3f} "
          f"composite={mse_cmp:.3f}")

    fd, tmp_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        save_gates({"tier_demo": gr}, tmp_path)
        loaded = load_gates(tmp_path)
        print(f"[smoke] gates.json roundtrip ok: "
              f"{sorted(loaded['tier_demo'].keys())}")
    finally:
        os.unlink(tmp_path)

    recal, t4p = t4_recalibrate(y_demo, 0.5 * y_demo + 4.0
                                + rng.normal(0.0, 0.5, n), unit)
    print(f"[smoke] t4 slopes: {[f['b'] for f in t4p['folds']]}")
