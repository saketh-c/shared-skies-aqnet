"""
exceed.py
AQNet v2 decoupled exceedance head: cross-fit LightGBM classifiers with
isotonic calibration, confirmation-frozen decision thresholds, and a paired
cluster-bootstrap admission test against the thresholded composite.

WHY a decoupled head (DESIGN §9): v1's fusion wiring drove exceedance recall
to ~0 — the mean-filled meta-learner regressed toward the middle of the
distribution exactly where the tail mattered. v2 therefore never derives its
shipped exceedance call from the regression path implicitly: the head is its
own cross-fit classifier, and it must BEAT the obvious baseline — simply
thresholding the composite, I(oof_final > thr) — inside the same admission
harness compose.py uses, or the recorded decision is "fail" and downstream
ships the thresholded composite. Tier acceptance can therefore never break
exceedance and the head can never ship on vibes.

Contracts honored here:
  * channel_reconstructed exclusion (calibrate.py DESIGN §4): PA rows whose
    cf_1 channel was reconstructed from ATM (>= ~20 µg/m³ — i.e. exactly the
    exceedance-relevant range) carry a label the calibration cannot vouch
    for. Those rows are excluded from classifier LABELS everywhere — train,
    isotonic calibration, threshold freezing, admission evaluation. They may
    still be PREDICTED (features are unaffected); they are never scored.
    A NaN channel_reconstructed flag (AQS rows, where the flag does not
    apply) counts as not-reconstructed — that is flag semantics, not a fill.
  * Fold nesting (DESIGN §2): per outer fold k the classifier trains on the
    inner SELECTION rows (inner_role[k] == 0), the isotonic map is fit on
    selection OUT-OF-FOLD raw scores (cross-fit across the selection inner
    folds, so the calibrator never sees in-sample scores), and the decision
    threshold is FROZEN on the disjoint CONFIRMATION rows (inner_role == 1)
    by max-F1. OOF probabilities are then emitted for outer-fold-k rows
    only — every scored row was held out from everything that shaped its
    score. Conformal-calibration rows (folds2 conformal_unit) and vault
    rows/units are excluded from every training, calibration, and admission
    set (belt and braces on top of folds2's own role assignment).
  * Features: frame2.feature_columns(frame) minus nothing, PLUS the
    composite OOF prediction oof_final as one extra column ("composite_oof").
    oof_final is itself cross-fit (the gates stage's OOF composite), so no
    row's composite feature was produced by a chain that saw the row.
  * Tail oversampling: positive-label (exceedance) rows appear 5x in every
    classifier fit (calibrate.py smoke-oversample idiom — seeded permuted
    index repetition, never synthetic rows).
  * Admission: one-sided paired unit-cluster bootstrap of the F1 delta,
    mechanics shared with compose.admission_test (per-cluster sufficient
    statistics, identical resampled rows for both predictors); the
    exceedance_f1 margin comes from power_analysis.json via
    compose._resolve_margins so the key contract stays single-sourced.
    Decision "pass" requires CI-separated superiority (lb95 > 0) of the
    head over the thresholded composite; default is closed.

Degradation: without lightgbm the classifier degrades to the raw composite
score routed through the same isotonic map (printed loudly, recorded in the
artifact); admission then compares calibrated-thresholded-composite against
plain-thresholded-composite and will almost surely stay closed — the honest
outcome, not a crash.

Artifacts (config2.artifact):
  exceed_model.json   per threshold: per-fold isotonic curves, frozen
                      decision thresholds, confirmation F1, admission result
  oof_exceed.npz      prob_{thr:g} f8[n] calibrated OOF probabilities (NaN
                      where a row was never honestly scored), flag_{thr:g}
                      u1[n] (flags are meaningful ONLY where prob is finite)

Run (stage CLI, idempotent, FORCE=1 to re-run):
    python exceed.py [--quick]
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

import config2
import compose
import frame2

# ── Guarded heavy import (v1 models_tabular style) ──────────────────────────
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    lgb = None
    HAS_LGBM = False
    print("[exceed] lightgbm not installed -- classifier degrades to the "
          "isotonic-calibrated composite score (pip install lightgbm)")

# ── Constants ───────────────────────────────────────────────────────────────
COMPOSITE_FEATURE = "composite_oof"   # oof_final appended as the last column
TAIL_OVERSAMPLE = 5                   # positive rows appear 5x in fits
N_ESTIMATORS = 500
QUICK_N_ESTIMATORS = 60
N_BOOT = 1000
DEFAULT_DECISION_THRESHOLD = 0.5      # only when confirmation is degenerate


def _say(msg):
    print(f"[aqnet2] exceed: {msg}", flush=True)


def _thr_key(thr):
    """Canonical artifact key for a threshold: 9.0 -> '9', 35.4 -> '35.4'."""
    return f"{float(thr):g}"


# ── Fold-dict access helpers ────────────────────────────────────────────────
def _per_fold_array(folds, key, k, n):
    """folds[key][k] as an int array of length n (str/int keys both work)."""
    d = folds.get(key)
    if not isinstance(d, dict):
        raise KeyError(f"folds2.json is missing the per-outer-fold dict "
                       f"'{key}'")
    arr = d.get(str(k), d.get(k))
    if arr is None:
        raise KeyError(f"folds['{key}'] has no entry for outer fold {k}")
    a = np.asarray(arr, dtype=int)
    if len(a) != n:
        raise ValueError(f"folds['{key}'][{k}] has length {len(a)}, "
                         f"expected {n}")
    return a


def _vault_row_mask(frame, folds):
    """Rows belonging to vault units OR the vault period (DESIGN §2: the
    vault is excluded from everything until validate)."""
    vault = frame2._as_unit_set((folds or {}).get("vault_sites", []))
    ids = frame["unit_id"].astype(str)
    m = ids.isin(vault).to_numpy() if vault else np.zeros(len(frame), bool)
    vstart = pd.Timestamp(getattr(config2, "VAULT_DATE_START", "2026-01-01"))
    m = m | (pd.to_datetime(frame["date"]) >= vstart).to_numpy()
    return m


def _conformal_row_mask(frame, folds):
    """conformal_unit rows — reserved for uq.py, never used here."""
    arr = (folds or {}).get("conformal_unit")
    if arr is None:
        print("[exceed] WARNING: folds2.json has no 'conformal_unit' array "
              "-- assuming none (uq calibration rows cannot be excluded)")
        return np.zeros(len(frame), bool)
    a = np.asarray(arr, dtype=int)
    if len(a) != len(frame):
        raise ValueError(f"conformal_unit length {len(a)} != frame rows "
                         f"{len(frame)}")
    return a > 0


def _reconstructed_mask(frame):
    """channel_reconstructed > 0 per row. NaN (AQS rows — the flag does not
    apply) counts as not-reconstructed: flag semantics, not a fill."""
    if "channel_reconstructed" not in frame.columns:
        print("[exceed] WARNING: frame has no channel_reconstructed column "
              "-- treating all rows as clean-channel (calibrate stage "
              "flagged none)")
        return np.zeros(len(frame), bool)
    cr = frame["channel_reconstructed"].to_numpy(dtype=np.float64)
    return np.nan_to_num(cr, nan=0.0) > 0


# ── Feature matrix + per-fold overrides (models_tabular idioms) ─────────────
def _apply_overrides(X, features, overrides, k):
    """Swap fold k's override columns into a COPY of X (v1 f{fold}__{col}
    contract: full-length arrays applied to train AND test rows). Non-feature
    override cols ('y', t0 columns absent from this frame) are skipped."""
    ov = (overrides or {}).get(k)
    if not ov:
        return X
    pos = {c: j for j, c in enumerate(features)}
    Xf = X.copy()
    applied = 0
    for col, arr in ov.items():
        if col not in pos:
            continue  # e.g. the fold-aware "y" override — not a feature here
        arr = np.asarray(arr, dtype=np.float64)
        if arr.shape != (X.shape[0],):
            raise ValueError(f"override for '{col}' fold {k} has shape "
                             f"{arr.shape}, expected ({X.shape[0]},)")
        Xf[:, pos[col]] = arr
        applied += 1
    if applied:
        _say(f"fold {k}: {applied} neighbor-override columns applied")
    return Xf


def _es_tail_split(n, dates=None):
    """Fold-internal early-stopping split: most recent ~10% of rows by date
    (models_tabular._es_tail_split idiom). (idx, None) for small folds."""
    if n < 200:
        return np.arange(n), None
    order = (np.argsort(dates, kind="stable") if dates is not None
             else np.arange(n))
    n_es = min(max(int(round(0.1 * n)), 100), 50000, n // 2)
    return order[:-n_es], order[-n_es:]


def _oversample_positive(idx, pos_mask, seed, factor=TAIL_OVERSAMPLE):
    """Index array with positive rows repeated `factor`x, seeded-permuted
    (calibrate.py _oversample_index idiom). idx must be sorted ascending —
    the shuffle operates on a dtype-stable, deterministic base order."""
    idx = np.sort(np.asarray(idx))
    pos = idx[pos_mask[idx]]
    if factor <= 1 or len(pos) == 0:
        return idx
    rep = np.concatenate([idx] + [pos] * (int(factor) - 1))
    rng = np.random.default_rng(seed)
    return rep[rng.permutation(len(rep))]


# ── Classifier (guarded LightGBM; composite-score fallback) ─────────────────
def _fit_classifier(X, labels, train_idx, dates_ns, rounds, seed):
    """Fit one binary classifier on train_idx rows with 5x tail oversampling
    and a temporal-tail early-stopping split (carved BEFORE oversampling so
    the eval set never contains duplicated rows).

    Returns a model tuple:
      ("lgbm", booster)        raw score = predict_proba[:, 1]
      ("const", p)             degenerate single-class training labels
      ("composite", None)      lightgbm unavailable — raw score is the
                               composite_oof column (last column of X)
    """
    tr = np.sort(np.asarray(train_idx))
    ytr = labels[tr].astype(np.int64)
    if not HAS_LGBM:
        return ("composite", None)
    if ytr.sum() == 0 or ytr.sum() == len(ytr):
        p = float(ytr.mean()) if len(ytr) else 0.0
        _say(f"degenerate training labels (all {p:g}) -- constant classifier")
        return ("const", p)

    d_tr = dates_ns[tr] if dates_ns is not None else None
    head, tail = _es_tail_split(len(tr), d_tr)
    head_rows = tr[head]
    os_rows = _oversample_positive(head_rows, labels, seed)
    est = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=int(rounds), learning_rate=0.05, num_leaves=63,
        min_child_samples=40, subsample=0.7, subsample_freq=1,
        colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=3.0,
        n_jobs=-1, random_state=int(seed), verbose=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn feature-name chatter
        if tail is not None:
            tail_rows = tr[tail]
            est.fit(X[os_rows], labels[os_rows].astype(np.int64),
                    eval_set=[(X[tail_rows],
                               labels[tail_rows].astype(np.int64))],
                    eval_metric="binary_logloss",
                    callbacks=[lgb.early_stopping(100, verbose=False)])
        else:
            os_all = _oversample_positive(tr, labels, seed)
            est.fit(X[os_all], labels[os_all].astype(np.int64))
    return ("lgbm", est)


def _raw_score(model, X, rows):
    """Raw (pre-isotonic) score for the given rows under the model tuple."""
    kind, obj = model
    rows = np.asarray(rows)
    if kind == "lgbm":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # sklearn feature-name chatter
            return obj.predict_proba(X[rows])[:, 1].astype(np.float64)
    if kind == "const":
        return np.full(len(rows), float(obj))
    if kind == "composite":
        return X[rows, -1].astype(np.float64)  # composite_oof column
    raise ValueError(f"unknown classifier kind {kind!r}")


# ── Isotonic calibration (self-contained PAVA — curves are the artifact) ────
def _pava_fit(x, y):
    """Pool-adjacent-violators isotonic fit of P(label|score).

    Returns a JSON-serializable curve {"x": [...], "y": [...], "note": str}
    of strictly-increasing knots; apply with _iso_apply (linear interpolation
    between block knots, clamped at the ends — sklearn semantics without the
    dependency, so the exact serving curve lives in exceed_model.json).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 10 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return {"x": [0.0, 1.0], "y": [0.0, 1.0],
                "note": "identity_insufficient_data"}
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    # Blocks: [sum_y, weight, x_first, x_last]; merge while means decrease.
    blocks = []
    for xi, yi in zip(xs, ys):
        blocks.append([yi, 1.0, xi, xi])
        while (len(blocks) > 1 and
               blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]):
            s, w, x0, x1 = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += w
            blocks[-1][3] = x1
    kx, ky = [], []
    for s, w, x0, x1 in blocks:
        m = s / w
        for xi in (x0, x1):
            if not kx or xi > kx[-1]:
                kx.append(float(xi))
                ky.append(float(m))
    if len(kx) < 2:
        kx = [kx[0], kx[0] + 1e-9]
        ky = [ky[0], ky[0]]
    return {"x": kx, "y": ky, "note": "ok"}


def _iso_apply(curve, scores):
    """Calibrated probability via the stored isotonic curve. NaN propagates
    (np.interp maps NaN input to NaN — never a fill)."""
    p = np.interp(np.asarray(scores, dtype=np.float64),
                  np.asarray(curve["x"]), np.asarray(curve["y"]))
    return np.clip(p, 0.0, 1.0)


# ── Threshold freezing (max-F1 on confirmation rows) ────────────────────────
def _max_f1_threshold(labels, probs):
    """(threshold, f1_at_threshold, note) maximizing F1 of `prob >= t` over
    the candidate probabilities. The recorded F1 is recomputed under the
    frozen >= rule so artifact and serving agree exactly."""
    ok = np.isfinite(probs)
    lb = np.asarray(labels, bool)[ok]
    pr = np.asarray(probs, dtype=np.float64)[ok]
    if len(lb) == 0 or lb.sum() == 0:
        return DEFAULT_DECISION_THRESHOLD, None, "no_positive_confirmation_rows"
    order = np.argsort(-pr, kind="stable")
    ps, ls = pr[order], lb[order].astype(np.float64)
    tp = np.cumsum(ls)
    fp = np.cumsum(1.0 - ls)
    fn = float(ls.sum()) - tp
    f1 = 2.0 * tp / np.maximum(2.0 * tp + fp + fn, 1e-12)
    # Only cuts at the LAST occurrence of each distinct prob are realizable
    # under the `prob >= t` rule (ties all flip together).
    last = np.r_[ps[1:] != ps[:-1], True]
    cand = np.flatnonzero(last)
    best = cand[np.argmax(f1[cand])]
    t = float(ps[best])
    flag = pr >= t
    tp_f = float(np.sum(flag & lb))
    fp_f = float(np.sum(flag & ~lb))
    fn_f = float(np.sum(~flag & lb))
    f1_f = compose._f1(tp_f, fp_f, fn_f)
    return t, (float(f1_f) if np.isfinite(f1_f) else None), "ok"


# ── Admission: paired cluster bootstrap of F1 (compose mechanics) ───────────
def admission_f1(labels, flag_ref, flag_new, clusters, margin,
                 n_boot=N_BOOT, seed=None):
    """One-sided paired unit-cluster bootstrap of the F1 delta.

    Single-metric specialization of compose.admission_test (same per-cluster
    sufficient statistics, same identical-resampled-rows pairing). With one
    metric the compose pass rule — non-inferiority on all defined metrics
    plus CI-separated superiority on at least one — collapses to superiority:
    decision "pass" iff lb95(delta F1) > 0. non_inferior (lb95 > -margin) is
    reported alongside for the record. Default closed on every degenerate
    path (compose convention).
    """
    seed = config2.SEED if seed is None else int(seed)
    lb_arr = np.asarray(labels, bool)
    fr = np.asarray(flag_ref, bool)
    fn_ = np.asarray(flag_new, bool)
    cl = np.asarray(clusters)
    n = len(lb_arr)
    for name, a in (("flag_ref", fr), ("flag_new", fn_), ("clusters", cl)):
        if len(a) != n:
            raise ValueError(f"{name} has length {len(a)}, expected {n}")
    margin = float(margin)
    if not np.isfinite(margin) or margin < 0.0:
        raise AssertionError(f"F1 admission margin must be finite >= 0, "
                             f"got {margin}")

    base = {"decision": "fail", "metric": "exceedance_f1", "delta": None,
            "ci": [None, None], "lb95": None, "margin": margin,
            "point_ref": None, "point_new": None, "n_clusters": 0,
            "n_rows": int(n), "n_boot": int(n_boot), "seed": int(seed),
            "n_boot_defined": 0, "non_inferior": None, "superior": None,
            "reasons": []}
    if n < 3:
        base["reasons"].append(f"only {n} evaluation rows")
        return compose._jsonable(base)
    uniq, inv = np.unique(cl, return_inverse=True)
    n_cl = len(uniq)
    base["n_clusters"] = int(n_cl)
    if n_cl < 2:
        base["reasons"].append(f"only {n_cl} cluster(s) -- paired cluster "
                               "bootstrap needs >= 2")
        return compose._jsonable(base)

    def _counts(flag):
        tp = np.bincount(inv, weights=(flag & lb_arr).astype(np.float64))
        fp = np.bincount(inv, weights=(flag & ~lb_arr).astype(np.float64))
        fn = np.bincount(inv, weights=(~flag & lb_arr).astype(np.float64))
        return tp, fp, fn

    tp_a, fp_a, fn_a = _counts(fr)
    tp_b, fp_b, fn_b = _counts(fn_)
    f1a = compose._f1(tp_a.sum(), fp_a.sum(), fn_a.sum())
    f1b = compose._f1(tp_b.sum(), fp_b.sum(), fn_b.sum())
    delta = f1b - f1a

    rng = np.random.default_rng(seed)
    draws = np.full(int(n_boot), np.nan)
    for b in range(int(n_boot)):
        pick = rng.integers(0, n_cl, n_cl)
        d_a = compose._f1(tp_a[pick].sum(), fp_a[pick].sum(),
                          fn_a[pick].sum())
        d_b = compose._f1(tp_b[pick].sum(), fp_b[pick].sum(),
                          fn_b[pick].sum())
        if np.isfinite(d_a) and np.isfinite(d_b):
            draws[b] = d_b - d_a
    fin = np.isfinite(draws)
    n_fin = int(fin.sum())
    defined = bool(np.isfinite(delta) and n_fin >= int(n_boot) // 2)
    lb95 = ub95 = float("nan")
    if n_fin:
        lb95, ub95 = np.percentile(draws[fin], [5.0, 95.0])

    out = dict(base)
    out.update({
        "delta": float(delta) if np.isfinite(delta) else None,
        "ci": [float(lb95) if np.isfinite(lb95) else None,
               float(ub95) if np.isfinite(ub95) else None],
        "lb95": float(lb95) if np.isfinite(lb95) else None,
        "point_ref": float(f1a) if np.isfinite(f1a) else None,
        "point_new": float(f1b) if np.isfinite(f1b) else None,
        "n_boot_defined": n_fin,
        "non_inferior": bool(lb95 > -margin) if defined else None,
        "superior": bool(lb95 > 0.0) if defined else None,
    })
    if not defined:
        out["reasons"].append("F1 delta undefined (rare event or unstable "
                              "bootstrap) -- default closed")
    elif lb95 > 0.0:
        out["decision"] = "pass"
    else:
        out["reasons"].append("head not CI-separated superior to the "
                              "thresholded composite -- ship the baseline")
    return compose._jsonable(out)


# ── fit_exceed (INTERFACES frozen API + optional kwargs, calibrate style) ───
def fit_exceed(frame, oof_final, folds, thresholds=config2.EXCEED_THRESHOLDS,
               overrides=None, margins=None, quick=False):
    """Fit the decoupled exceedance head. See the module docstring for the
    full protocol; per outer fold k and threshold thr:

      1. selection rows (inner_role[k]==0, labels valid) train the classifier
      2. selection OOF raw scores (cross-fit across selection inner folds)
         fit the isotonic map — the calibrator never sees in-sample scores
      3. the decision threshold is frozen on confirmation rows by max-F1
      4. calibrated OOF probabilities are emitted for outer-fold-k rows

    Label validity = finite y AND channel_reconstructed != 1 (calibrate.py
    contract) AND not vault AND not conformal_unit. `overrides` is the
    optional {fold: {col: array}} f{k}__{col} neighbor-override dict
    (frame2.neighbor_overrides contract); `margins` the power_analysis.json
    payload (compose._resolve_margins semantics).

    Returns {"prob": {key: f8[n]}, "flag": {key: u1[n]}, "model": dict}
    where key = _thr_key(thr) and "model" is the exceed_model.json payload.
    """
    n = len(frame)
    of = np.asarray(oof_final, dtype=np.float64)
    if len(of) != n:
        raise ValueError(f"oof_final has length {len(of)}, expected {n}")
    y = frame["y"].to_numpy(dtype=np.float64)
    clusters = frame["unit_id"].astype(str).to_numpy()
    dates_ns = pd.to_datetime(frame["date"]).to_numpy()

    feats = frame2.feature_columns(frame)
    X_base = frame[feats].to_numpy(dtype=np.float64)
    X_all = np.column_stack([X_base, of])
    feat_names = list(feats) + [COMPOSITE_FEATURE]

    outer = np.asarray(folds["outer_fold"], dtype=int)
    if len(outer) != n:
        raise ValueError(f"outer_fold length {len(outer)} != frame rows {n}")
    ks = sorted(int(v) for v in np.unique(outer) if v >= 0)
    if not ks:
        raise ValueError("no outer folds in folds2.json -- nothing to "
                         "cross-fit")

    vault = _vault_row_mask(frame, folds)
    conf_unit = _conformal_row_mask(frame, folds)
    recon = _reconstructed_mask(frame)
    label_valid = np.isfinite(y) & ~recon & ~vault & ~conf_unit
    n_recon_dropped = int((np.isfinite(y) & recon).sum())
    _say(f"{n_recon_dropped:,} finite-y rows excluded from labels "
         f"(channel_reconstructed -- calibrate contract); "
         f"{int(label_valid.sum()):,} labelable rows remain")

    rounds = QUICK_N_ESTIMATORS if quick else N_ESTIMATORS
    m_f1 = compose._resolve_margins(margins)["exceedance_f1"]
    thr_list = [float(t) for t in thresholds]

    prob_out, flag_out, per_thr = {}, {}, {}
    for k in ks:
        role = _per_fold_array(folds, "inner_role", k, n)
        inner = _per_fold_array(folds, "inner_fold", k, n)
        sel = (role == 0) & label_valid
        conf = (role == 1) & label_valid
        test = outer == k
        X_fold = _apply_overrides(X_all, feat_names, overrides, k)
        _say(f"outer fold {k}: sel {int(sel.sum()):,} / conf "
             f"{int(conf.sum()):,} / test {int(test.sum()):,} rows")

        for thr in thr_list:
            key = _thr_key(thr)
            if key not in prob_out:
                prob_out[key] = np.full(n, np.nan)
                flag_out[key] = np.zeros(n, dtype=np.uint8)
                per_thr[key] = {"folds": {}}
            labels = y > thr
            seed_k = int(config2.SEED) + 101 * (k + 1)
            note = "ok"

            # 2. Selection OOF raw scores for the isotonic map.
            sel_idx = np.flatnonzero(sel)
            sel_raw = np.full(n, np.nan)
            sel_js = sorted(int(j) for j in np.unique(inner[sel]) if j >= 0)
            if len(sel_js) >= 2:
                for j in sel_js:
                    tr_j = np.flatnonzero(sel & (inner != j))
                    te_j = np.flatnonzero(sel & (inner == j))
                    m_j = _fit_classifier(X_fold, labels, tr_j, dates_ns,
                                          rounds, seed_k + j)
                    sel_raw[te_j] = _raw_score(m_j, X_fold, te_j)
            else:
                note = "insample_isotonic_single_selection_fold"
                _say(f"fold {k} thr {key}: {note}")
                m_j = _fit_classifier(X_fold, labels, sel_idx, dates_ns,
                                      rounds, seed_k)
                sel_raw[sel_idx] = _raw_score(m_j, X_fold, sel_idx)

            curve = _pava_fit(sel_raw[sel_idx], labels[sel_idx].astype(float))

            # 1./3./4. Final classifier, frozen threshold, OOF emission.
            model = _fit_classifier(X_fold, labels, sel_idx, dates_ns,
                                    rounds, seed_k)
            conf_idx = np.flatnonzero(conf)
            p_conf = _iso_apply(curve, _raw_score(model, X_fold, conf_idx))
            t_k, f1_k, t_note = _max_f1_threshold(labels[conf_idx], p_conf)
            test_idx = np.flatnonzero(test)
            p_test = _iso_apply(curve, _raw_score(model, X_fold, test_idx))
            prob_out[key][test_idx] = p_test
            flag_out[key][test_idx] = np.where(
                np.isfinite(p_test) & (p_test >= t_k), 1, 0).astype(np.uint8)

            per_thr[key]["folds"][str(k)] = {
                "iso_x": curve["x"], "iso_y": curve["y"],
                "iso_note": curve["note"],
                "decision_threshold": float(t_k),
                "conf_f1": f1_k,
                "threshold_note": t_note,
                "classifier": model[0],
                "n_sel_rows": int(sel.sum()),
                "n_conf_rows": int(conf.sum()),
                "n_sel_pos": int(labels[sel].sum()),
                "n_conf_pos": int(labels[conf].sum()),
                "note": note,
            }

    # ── Admission per threshold: head flags vs thresholded composite ──
    for thr in thr_list:
        key = _thr_key(thr)
        labels = y > thr
        ev = (np.isfinite(prob_out[key]) & label_valid & np.isfinite(of))
        base_flag = of[ev] > thr
        head_flag = flag_out[key][ev] > 0
        test = admission_f1(labels[ev], base_flag, head_flag, clusters[ev],
                            m_f1, n_boot=N_BOOT, seed=config2.SEED)
        per_thr[key]["admission"] = test
        per_thr[key]["baseline"] = f"I(oof_final > {key})"
        per_thr[key]["n_eval_rows"] = int(ev.sum())
        _say(f"thr {key}: admission decision={test['decision']} "
             f"F1 base={test['point_ref']} head={test['point_new']} "
             f"lb95={test['lb95']} over {test['n_clusters']} clusters")

    model_json = {
        "thresholds": thr_list,
        "npz_keys": {k: [f"prob_{k}", f"flag_{k}"] for k in per_thr},
        "features": feat_names,
        "lightgbm": bool(HAS_LGBM),
        "tail_oversample": TAIL_OVERSAMPLE,
        "n_estimators": rounds,
        "seed": int(config2.SEED),
        "quick": bool(quick),
        "label_exclusions": {
            "channel_reconstructed_rows": n_recon_dropped,
            "vault_rows": int(vault.sum()),
            "conformal_unit_rows": int(conf_unit.sum()),
        },
        "margin_exceedance_f1": m_f1,
        "per_threshold": per_thr,
    }
    return {"prob": prob_out, "flag": flag_out,
            "model": compose._jsonable(model_json)}


# ── Stage wiring ────────────────────────────────────────────────────────────
def _load_folds(path, frame):
    """folds2.load_folds when the module exists (content-hash verified);
    raw-JSON fallback with an n_rows check otherwise (calibrate idiom)."""
    if not os.path.exists(path):
        raise SystemExit(f"[aqnet2] exceed: {path} missing -- run the folds "
                         "stage first (cross-fitting without folds would be "
                         "a leak, refusing to run)")
    try:
        import folds2
        return folds2.load_folds(path, frame)
    except ImportError as e:
        print(f"[exceed] folds2 module unavailable ({e}) -- loading "
              "folds2.json raw (content hash NOT verified)")
    with open(path, "r", encoding="utf-8") as fh:
        folds = json.load(fh)
    if int(folds.get("n_rows", -1)) != len(frame):
        raise SystemExit(f"[aqnet2] exceed: folds2.json n_rows "
                         f"{folds.get('n_rows')} != frame rows {len(frame)}")
    return folds


def _load_overrides_if_any(n_rows, path=None):
    """Outer-fold neighbor overrides npz when present; degrade with a
    warning otherwise (the head then sees the deployment-view neighbor
    columns — recorded, never silent)."""
    cands = [path] if path else [config2.artifact("nbr_overrides_outer.npz"),
                                 config2.artifact("nbr_overrides.npz")]
    for p in cands:
        if p and os.path.exists(p):
            _say(f"neighbor overrides loaded from {p}")
            return frame2.load_overrides(p, n_rows)
    _say("WARNING: no outer-fold neighbor-overrides npz found -- classifier "
         "sees deployment-view neighbor columns (recorded in the artifact)")
    return None


def run_exceed(quick=False, folds_path=None, overrides_path=None):
    dest_model = config2.artifact("exceed_model.json")
    dest_npz = config2.artifact("oof_exceed.npz")
    if (os.path.exists(dest_model) and os.path.exists(dest_npz)
            and os.environ.get("FORCE") != "1"):
        _say(f"{dest_model} exists (FORCE=1 to rebuild) -- skip")
        return 0

    print("[aqnet2] ── stage: exceed ──")
    frame_path = config2.artifact("frame_truth.parquet")
    if not os.path.exists(frame_path):
        raise SystemExit("[aqnet2] exceed: frame_truth.parquet missing -- "
                         "run the features stage first")
    frame = pd.read_parquet(frame_path)
    folds = _load_folds(folds_path or config2.artifact("folds2.json"), frame)

    comp_path = config2.artifact("oof_composite.npz")
    if not os.path.exists(comp_path):
        raise SystemExit("[aqnet2] exceed: oof_composite.npz missing -- "
                         "run the gates stage first")
    with np.load(comp_path) as z:
        oof_final = np.asarray(z["oof_final"], dtype=np.float64)
    if len(oof_final) != len(frame):
        raise SystemExit(f"[aqnet2] exceed: oof_final length "
                         f"{len(oof_final)} != frame rows {len(frame)}")

    margins = None
    pa_path = config2.artifact("power_analysis.json")
    if os.path.exists(pa_path):
        with open(pa_path, "r", encoding="utf-8") as fh:
            margins = json.load(fh)
    else:
        _say("WARNING: power_analysis.json missing -- compose safe-default "
             "margins in force")

    overrides = _load_overrides_if_any(len(frame), overrides_path)
    result = fit_exceed(frame, oof_final, folds, overrides=overrides,
                        margins=margins, quick=quick)
    result["model"]["overrides_used"] = bool(overrides)

    payload = {}
    for key in result["prob"]:
        payload[f"prob_{key}"] = result["prob"][key].astype(np.float64)
        payload[f"flag_{key}"] = result["flag"][key].astype(np.uint8)
    tmp = dest_npz + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, dest_npz)
    _say(f"wrote {dest_npz} ({sorted(payload)})")

    tmp = dest_model + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(result["model"], fh, indent=2, sort_keys=True)
    os.replace(tmp, dest_model)
    _say(f"wrote {dest_model}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="AQNet v2 decoupled exceedance head (stage: exceed)")
    ap.add_argument("--quick", action="store_true",
                    help="fewer boosting rounds (smoke test)")
    ap.add_argument("--folds", default=None,
                    help="path to folds2.json (default: artifacts/v2)")
    ap.add_argument("--overrides", default=None,
                    help="outer-fold neighbor-overrides npz (optional)")
    args = ap.parse_args(argv)
    return run_exceed(quick=args.quick, folds_path=args.folds,
                      overrides_path=args.overrides)


if __name__ == "__main__":
    sys.exit(main())
