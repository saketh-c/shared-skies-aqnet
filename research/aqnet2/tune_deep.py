"""Registered v4 deep-tier hyperparameter search (stages graphtune/fieldtune).

EXPANSION.md "Registered v4 deep-tier hyperparameter search (2026-08-09,
pre-launch)" is the binding protocol; this module is its executable form and
freezes the search spaces at commit time:

  * Search data: strictly the inner SELECTION folds (inner folds 0-1) of
    outer contexts 0-2; candidates are scored on the inner CONFIRMATION
    folds (2-3) of the SAME contexts, so later outer folds never inform
    selection. Outer-fold test rows and the vault (sites and the
    >= VAULT_DATE_START period) are never read by any trial: the driver
    passes explicit fold lists into the tiers' existing fine-tune
    machinery, NaN-masks the residual targets at every outer-test/vault
    row BEFORE any trial statistic runs, and hard-asserts the row masks
    (selection_masks) per (candidate, context).
  * Selection: ONE config per tier, chosen globally. The winner maximizes
    the MEAN inner-confirmation weighted R^2 across contexts 0-2; the
    folds are spatially blocked by construction (folds2), so this is the
    registered spatially-blocked selection metric. Candidate 0 is ALWAYS
    the current default config (graph_res/field_res.tier_defaults()) and
    remains the registered fallback on search failure or tie.
  * Budgets: T2 24 candidates, T3 16, random search over the frozen
    spaces below. The search surrenders its remaining budget when the
    selection objective plateaus (PLATEAU_*, the registered
    inner-selection early stop); trials that crash or score NaN are
    recorded as failed and can never win.
  * The winning config then runs the UNCHANGED pre-registered admission
    machinery (same gates, same alpha caps, same coverage-pattern
    conditioning). Tuning changes the candidate, never the test.

Trial mechanics (reusing the tiers' existing inner-fold fit paths):

  T2  graph_res.finetune per context k with _train_inner_folds=[0, 1] and
      _eval_inner_folds=[2, 3]. The (k, 2) nested calibration column feeds
      the trial inputs: it is the closest existing column to a
      selection-pure calibration (its own fold-2 exclusion covers half the
      confirmation split); the residual fold-3 overlap touches INPUTS
      only, is identical for every candidate, and involves no outer-test
      or vault row. Architecture-changing candidates get a short shared
      trial pretrain (TRIAL_PRETRAIN_EPOCHS_T2, cached per architecture);
      default-architecture candidates initialize from the production
      graphpre checkpoint. That asymmetry biases the search toward the
      incumbent architecture, which is the conservative direction: the
      registered fallback IS the incumbent, so a challenger that cannot
      overcome it was never going to be adopted anyway.
  T3  field_res.finetune per context k with the same fold lists. Encoder
      features are built once per (encoder checkpoint, fourier ladder)
      and shared across contexts and candidates. mask_ratio is a pretrain
      parameter, so non-default draws (a discrete grid, so trial
      pretrains cache by value) get a short trial pretrain
      (TRIAL_PRETRAIN_EPOCHS_T3) with private checkpoint paths.

Seeds (folds2 precedent: every draw is default_rng(config2.SEED + a
documented salt)):

  candidate draws   SEED + T2_DRAW_SALT / T3_DRAW_SALT, one fresh
                    generator per call, so the candidate list is
                    deterministic and re-derivable on resume;
  trial fits        SEED + TRIAL_FIT_SALT * (candidate_index + 1);
  trial pretrains   SEED + TUNE_PRETRAIN_SALT, candidate-independent so
                    equal configurations produce identical (cacheable)
                    pretrains.

Artifacts: config2.artifact("tune_t2.json") / artifact("tune_t3.json")
with the winning config, the full trial table (config, per-context scores,
mean, wall seconds, status), the protocol echo and the seed provenance.
In-flight progress lives in tune_t{2,3}_partial.json until completion, so
the sentinel (the final artifact, checked by pipeline2.SENTINELS and the
slurm2 pre-checks) can never be satisfied by a half-run search; a
resubmitted job resumes from the partial table after verifying its seed
and candidate provenance (a mismatch discards the partial AND every trial
checkpoint, so a restarted search can never score a stale candidate's
result). Every trial checkpoint carries candidate_hash (config + trial
seed + space fingerprint) in both its filename stem and its stored cfg;
resume/skip honors only a matching hash. A completed artifact skips
unless FORCE=1.

A --quick run is a machinery smoke, never a search: it writes
tune_t{2,3}_quick.json (and _quick-stemmed trial checkpoints), so it can
never satisfy the production sentinel or seed a full search, and both
tiers' load_tuned_hp refuse any artifact whose payload says quick.

CLI:
    python tune_deep.py --tier t2|t3 [--quick]
Sentinels: tune_t2.json / tune_t3.json. FORCE=1 re-runs.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

import config2
import folds2

# ── Protocol constants (EXPANSION.md; frozen at commit) ─────────────────────

KS = (0, 1, 2)            # outer contexts searched; later folds never inform
TRAIN_INNER = (0, 1)      # inner SELECTION folds (trial training data)
EVAL_INNER = (2, 3)       # inner CONFIRMATION folds (candidate scoring)
TRIAL_CAL_J = 2           # T2 nested calibration column: pa_cal_f{k}_2

BUDGET = {"t2": 24, "t3": 16}
QUICK_BUDGET = 3          # smoke budget; a quick run is never a search
QUICK_KS = (0, 1)         # quick smoke keeps the 2-outer-fold convention

ARTIFACT = {"t2": "tune_t2.json", "t3": "tune_t3.json"}

# Search-level early stop (the registered inner-selection plateau): the
# remaining budget is surrendered when the best mean confirmation R^2 has
# not improved by MIN_DELTA over the last PATIENCE completed candidates.
PLATEAU_PATIENCE = {"t2": 6, "t3": 4}
PLATEAU_MIN_DELTA = 0.002
# Trial-level early stop handed to the fine-tune loops (_early_stop hook).
TRIAL_ES = {"patience": 5, "min_delta": 1e-3}
# Score differences at float-noise scale are ties; the registered default
# keeps the seat on a tie (EXPANSION.md fallback clause).
TIE_EPS = 1e-6

# Short trial pretrains for candidates whose pretrain-side parameters
# differ from the defaults (T2 architecture + knn; T3 mask ratio).
# Deliberately small: enough signal to RANK candidates inside the 24 h
# wall, cached across candidates that share the varied values. The
# winner's production pretrain runs at the full budget in its own stage.
TRIAL_PRETRAIN_EPOCHS_T2 = 12
TRIAL_PRETRAIN_EPOCHS_T3 = 12

# ── Seed salts (folds2 precedent: default_rng(config2.SEED + salt)) ─────────

T2_DRAW_SALT = 620017     # T2 candidate draws
T3_DRAW_SALT = 730013     # T3 candidate draws
TRIAL_FIT_SALT = 810019   # per-trial fit seed: SEED + salt * (index + 1)
TUNE_PRETRAIN_SALT = 916003   # trial pretrains, candidate-independent

# ── Frozen search spaces ────────────────────────────────────────────────────
#
# Registered clause: "random search over the spaces frozen in tune_deep.py
# at its commit". Every range is a documented span around the current
# graph_res/field_res defaults (the candidate-0 value in parentheses).
# Tuples of values are uniform choices; (lo, hi) pairs are ranges whose
# draw form is documented per key.

T2_SPACE = {
    # transformer width (128); d_ff scales with the drawn width below
    "d_model": (96, 128, 160, 192),
    # attention heads (8); drawn among divisors of the drawn width
    "n_heads": (4, 8),
    # pre-LN blocks (4)
    "n_layers": (3, 4, 5, 6),
    # FFN width multiplier (9.0 = D_FF / D_MODEL); d_ff = round(m * width)
    "d_ff_mult": (6.0, 12.0),
    # residual-branch dropout (0.0), uniform
    "dropout": (0.0, 0.3),
    # fine-tune AdamW learning rate (1e-4), log-uniform
    "lr_finetune": (3e-5, 3e-4),
    # fine-tune AdamW weight decay (1e-3), log-uniform
    "weight_decay": (1e-4, 1e-2),
    # graph in-degree, in-edges per node (10)
    "knn_k": (6, 8, 10, 12, 15),
    # fine-tune epochs (40), uniform int; the trial early stop may end
    # a fit sooner on a loss plateau
    "finetune_epochs": (16, 48),
}

T3_SPACE = {
    # Fourier wavelengths per axis (4)
    "fourier_n": (3, 4, 5, 6),
    # shortest wavelength in km (5.0), log-uniform; 5 km is the registered
    # between-sensor-hallucination hard cap and therefore the immovable
    # lower bound of this range (field_res.fourier_wavelengths)
    "fourier_min_km": (5.0, 25.0),
    # INR MLP hidden width (256)
    "inr_width": (128, 192, 256, 384),
    # INR hidden layers (3)
    "inr_depth": (2, 3, 4),
    # INR AdamW learning rate (1e-3), log-uniform
    "ft_lr": (3e-4, 3e-3),
    # pretrain patch-mask ratio (0.60); discrete grid so the per-value
    # trial pretrains cache across candidates
    "mask_ratio": (0.45, 0.60, 0.75),
    # INR fine-tune epochs (40), uniform int
    "ft_epochs": (16, 48),
}

# Protocol echo written into every tune artifact.
PROTOCOL = {
    "registered": "EXPANSION.md: Registered v4 deep-tier hyperparameter "
                  "search (2026-08-09, pre-launch)",
    "search_data": "inner SELECTION folds (0-1) of outer contexts 0-2 only",
    "scoring": "inner CONFIRMATION folds (2-3) of the same contexts; "
               "winner = best mean spatially-blocked weighted R^2",
    "fallback": "candidate 0 = the current default config; wins on search "
                "failure or tie",
    "never_read": "outer-fold test rows and the vault (sites and the >= "
                  "VAULT_DATE_START period)",
}


def _say(msg):
    print(f"[aqnet2] tune_deep: {msg}", flush=True)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


def _write_json_atomic(payload, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
    os.replace(tmp, path)


# ── Candidate draws ─────────────────────────────────────────────────────────

def _log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _draw_t2(rng):
    import graph_res
    d_model = int(rng.choice(T2_SPACE["d_model"]))
    heads = [h for h in T2_SPACE["n_heads"] if d_model % h == 0]
    hp = dict(graph_res.tier_defaults())
    hp.update({
        "d_model": d_model,
        "n_heads": int(rng.choice(heads)),
        "n_layers": int(rng.choice(T2_SPACE["n_layers"])),
        "d_ff": int(round(rng.uniform(*T2_SPACE["d_ff_mult"]) * d_model)),
        "dropout": round(float(rng.uniform(*T2_SPACE["dropout"])), 4),
        "lr_finetune": round(_log_uniform(rng, *T2_SPACE["lr_finetune"]), 8),
        "weight_decay": round(_log_uniform(rng, *T2_SPACE["weight_decay"]),
                              8),
        "knn_k": int(rng.choice(T2_SPACE["knn_k"])),
        "finetune_epochs": int(rng.integers(
            T2_SPACE["finetune_epochs"][0],
            T2_SPACE["finetune_epochs"][1] + 1)),
    })
    return hp


def _draw_t3(rng):
    import field_res
    hp = dict(field_res.tier_defaults())
    hp.update({
        "fourier_n": int(rng.choice(T3_SPACE["fourier_n"])),
        "fourier_min_km": round(_log_uniform(rng,
                                             *T3_SPACE["fourier_min_km"]), 4),
        "inr_width": int(rng.choice(T3_SPACE["inr_width"])),
        "inr_depth": int(rng.choice(T3_SPACE["inr_depth"])),
        "ft_lr": round(_log_uniform(rng, *T3_SPACE["ft_lr"]), 8),
        "mask_ratio": float(rng.choice(T3_SPACE["mask_ratio"])),
        "ft_epochs": int(rng.integers(T3_SPACE["ft_epochs"][0],
                                      T3_SPACE["ft_epochs"][1] + 1)),
    })
    return hp


def draw_candidates(tier, budget=None):
    """Deterministic candidate list for a tier.

    Index 0 is ALWAYS the tier's current default config (the registered
    fallback); indices 1..budget-1 are seeded random draws from the frozen
    space -- one fresh default_rng(config2.SEED + tier draw salt) per
    call, so two calls (and a resumed search) derive the identical list.
    """
    tier = str(tier).lower()
    if tier == "t2":
        import graph_res
        salt, draw, default = T2_DRAW_SALT, _draw_t2, graph_res.tier_defaults
    elif tier == "t3":
        import field_res
        salt, draw, default = T3_DRAW_SALT, _draw_t3, field_res.tier_defaults
    else:
        raise SystemExit(f"[aqnet2] tune_deep: unknown tier {tier!r}")
    budget = int(budget if budget is not None else BUDGET[tier])
    rng = np.random.default_rng(config2.SEED + salt)
    return [default()] + [draw(rng) for _ in range(budget - 1)]


# ── Trial provenance ────────────────────────────────────────────────────────

def space_fingerprint(tier):
    """Stable digest of a tier's frozen search space, so a space edit
    changes every candidate hash and stale trial checkpoints rebuild."""
    space = T2_SPACE if str(tier).lower() == "t2" else T3_SPACE
    blob = json.dumps(_jsonable(space), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


def candidate_hash(tier, idx, hp):
    """Per-candidate provenance hash: the candidate config, its trial-fit
    seed and the frozen-space fingerprint. Embedded in every trial
    checkpoint's filename stem AND its stored cfg (tune_hash), so
    resume/skip can only honor a checkpoint that provably belongs to the
    current candidate list -- anything else rebuilds."""
    payload = {"tier": str(tier).lower(), "config": _jsonable(hp),
               "seed": int(config2.SEED + TRIAL_FIT_SALT * (int(idx) + 1)),
               "space": space_fingerprint(tier)}
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def _t2_trial_stem(idx, hp, quick, k):
    """T2 trial checkpoint stem. _quick keeps smoke checkpoints in their
    own namespace so they can never resume or skip a full trial."""
    h = candidate_hash("t2", idx, hp)
    return f"graphtune{'_quick' if quick else ''}_t{idx:02d}_{h}_f{k}"


def _t3_trial_name(idx, hp, quick, k):
    """T3 trial checkpoint filename (same namespacing as T2)."""
    h = candidate_hash("t3", idx, hp)
    return f"fieldtune{'_quick' if quick else ''}_t{idx:02d}_{h}_f{k}.pt"


def _clear_trial_checkpoints(tier):
    """Remove one tier's trial checkpoints (fits and cached trial
    pretrains) on partial provenance drift, so a restarted search can
    never score a stale candidate's result. The trial stems are private
    namespaces (graphtune_*/graphpre_tune_*/fieldtune_*/fieldpre_tune_*);
    production checkpoints are structurally out of reach."""
    if str(tier).lower() == "t2":
        pats = [config2.artifact("graphtune_*", "graph"),
                config2.artifact("graphpre_tune_*", "graph")]
    else:
        pats = [config2.artifact("fieldtune_*", "field"),
                config2.artifact("fieldpre_tune_*", "field"),
                config2.artifact("fieldpre_tune_*.json")]
    n = 0
    for pat in pats:
        for p in glob.glob(pat):
            os.remove(p)
            n += 1
    if n:
        _say(f"provenance drift: cleared {n} stale {tier} trial "
             f"checkpoint file(s)")


def _finite_score(trial):
    """True when a trial row carries a finite confirmation score."""
    s = (trial or {}).get("mean_r2")
    return s is not None and bool(np.isfinite(s))


# ── Protocol masks ──────────────────────────────────────────────────────────

def selection_masks(frame, folds, k):
    """Row masks (train, eval) for outer context k under the protocol.

    train = the inner SELECTION folds (TRAIN_INNER), eval = the inner
    CONFIRMATION folds (EVAL_INNER), both minus every outer-fold-k row and
    every vault row (folds2.vault_row_mask: vault units AND the vault
    period). Hard-asserted here -- this is the driver's provable
    no-outer-test/no-vault record, unit-tested against synthetic folds
    whose inner arrays deliberately cover vault units.
    """
    n = len(frame)
    outer = np.asarray(folds["outer_fold"], dtype=np.int64)
    inner_map = folds["inner_fold"]
    inner = np.asarray(inner_map.get(str(k), inner_map.get(int(k))),
                       dtype=np.int64)
    if len(outer) != n or len(inner) != n:
        raise SystemExit(f"[aqnet2] tune_deep: folds arrays misaligned with "
                         f"the frame ({len(outer)}/{len(inner)} vs {n} rows)")
    vrow = folds2.vault_row_mask(frame, folds)

    keep = (outer != int(k)) & ~vrow
    train = keep & np.isin(inner, np.asarray(TRAIN_INNER, dtype=np.int64))
    ev = keep & np.isin(inner, np.asarray(EVAL_INNER, dtype=np.int64))

    assert not (train & ev).any(), "selection/confirmation rows overlap"
    for name, m in (("train", train), ("eval", ev)):
        assert not (m & (outer == int(k))).any(), (
            f"outer-fold-{k} test rows in the {name} mask")
        assert not (m & vrow).any(), f"vault rows in the {name} mask"
    return train, ev


# ── Winner selection / plateau ──────────────────────────────────────────────

def select_winner(trials):
    """(winner_index, is_fallback) from the trial table.

    The highest finite mean_r2 wins; equal scores resolve to the LOWEST
    index, so candidate 0 (the registered default) wins exact ties
    outright. A best challenger within TIE_EPS of candidate 0's score is
    a tie too and triggers the registered fallback, as does a search with
    no finite score anywhere.
    """
    scored = [t for t in trials
              if t.get("mean_r2") is not None
              and np.isfinite(t["mean_r2"])]
    if not scored:
        return 0, True
    best = scored[0]
    for t in scored[1:]:
        if t["mean_r2"] > best["mean_r2"]:
            best = t
    if int(best["index"]) == 0:
        return 0, False
    c0 = next((t for t in scored if int(t["index"]) == 0), None)
    if c0 is not None and best["mean_r2"] <= c0["mean_r2"] + TIE_EPS:
        return 0, True
    return int(best["index"]), False


def plateau_stop(trials, tier):
    """True when the last PLATEAU_PATIENCE[tier] completed candidates
    failed to improve the best mean confirmation R^2 seen before them by
    PLATEAU_MIN_DELTA -- the registered inner-selection plateau. The
    remaining budget is surrendered, never re-drawn."""
    pat = int(PLATEAU_PATIENCE[tier])
    if len(trials) <= pat:
        return False

    def _score(t):
        s = t.get("mean_r2")
        return s if (s is not None and np.isfinite(s)) else -np.inf

    best_head = max(_score(t) for t in trials[:-pat])
    if not np.isfinite(best_head):
        return False
    best_tail = max(_score(t) for t in trials[-pat:])
    return best_tail <= best_head + PLATEAU_MIN_DELTA


# ── T2 trial ────────────────────────────────────────────────────────────────

def _mask_t2_residual(ctx, k, frame, folds):
    """NaN-mask the fold-k residual/weight matrices at every outer-fold-k
    and vault row BEFORE any trial statistic runs.

    graph_res normalizes r1 over the finite cells of these cached
    matrices (_r1_norm_stats), so an unmasked cache would let outer-test
    rows into a trial constant -- the protocol forbids ANY read. The
    masked pair is pre-seeded into ctx's cache slot, which _t1_residual
    then returns to every downstream consumer. Idempotent; returns the
    masked cell count.
    """
    import graph_res
    r1_mat, w_mat = graph_res._t1_residual(ctx, int(k))
    outer = np.asarray(folds["outer_fold"], dtype=np.int64)
    vrow = folds2.vault_row_mask(frame, folds)
    kill = (((outer == int(k)) | vrow)
            & (ctx["row_unit"] >= 0) & (ctx["row_day"] >= 0))
    if kill.any():
        r1_mat = r1_mat.copy()
        w_mat = w_mat.copy()
        r1_mat[ctx["row_unit"][kill], ctx["row_day"][kill]] = np.nan
        w_mat[ctx["row_unit"][kill], ctx["row_day"][kill]] = np.nan
        ctx[("_r1", int(k))] = (r1_mat, w_mat)
    return int(kill.sum())


def _t2_trial_pretrain(hp, quick, base):
    """Short shared pretrain for a non-default T2 architecture, cached per
    (architecture, knn_k) so equal draws reuse one checkpoint. Seeded from
    SEED + TUNE_PRETRAIN_SALT, candidate-independent: identical
    configurations must produce identical pretrains."""
    import graph_res
    key_cfg = dict(graph_res.arch_of(hp), knn_k=int(hp["knn_k"]))
    key = hashlib.sha256(json.dumps(key_cfg, sort_keys=True).encode()
                         ).hexdigest()[:8]
    stem = f"graphpre_tune{'_quick' if quick else ''}_{key}"
    epochs = graph_res.QUICK_EPOCHS if quick else TRIAL_PRETRAIN_EPOCHS_T2
    ck = graph_res.load_checkpoint(stem)
    if ck is not None and int(ck["epoch"]) >= epochs - 1:
        _say(f"t2 trial pretrain {stem}: complete -- reused")
        return stem
    cfg = dict(base)
    cfg["stage"] = "graphpre"
    cfg["seed"] = int(config2.SEED + TUNE_PRETRAIN_SALT)
    cfg["epochs"] = epochs
    cfg["resume"] = True
    cfg["_stem"] = stem
    _say(f"t2 trial pretrain {stem}: {epochs} epochs ({key_cfg})")
    graph_res.pretrain(cfg)
    return stem


def _run_t2_trial(idx, hp, quick, ks, caches):
    """One T2 candidate: per outer context k, fine-tune on the SELECTION
    folds and score weighted R^2 on the CONFIRMATION folds. Returns
    {str(k): r2}. The graph context is built once per knn_k and shared
    across candidates."""
    import graph_res
    seed = int(config2.SEED + TRIAL_FIT_SALT * (idx + 1))
    base = graph_res.make_cfg("graphres", quick=quick, resume=True)
    base["seed"] = seed
    base["hp"] = dict(hp)
    base["epochs"] = (graph_res.QUICK_EPOCHS if quick
                      else int(hp["finetune_epochs"]))
    ctx_cache = caches.setdefault("t2_ctx", {})
    knn = int(hp["knn_k"])
    if knn in ctx_cache:
        base["_ctx"] = ctx_cache[knn]
    ctx = graph_res._get_ctx(base)
    ctx_cache[knn] = ctx
    frame, folds = ctx["frame"], ctx["folds"]

    if graph_res.arch_of(hp) == graph_res.arch_of(graph_res.tier_defaults()):
        init_stem = "graphpre"     # the production pretrain checkpoint
    else:
        init_stem = _t2_trial_pretrain(hp, quick, base)

    fold_r2 = {}
    for k in ks:
        tr_rows, ev_rows = selection_masks(frame, folds, k)
        n_masked = _mask_t2_residual(ctx, k, frame, folds)
        cfg = dict(base)
        # Candidate provenance: the hash lives in the stem AND the stored
        # cfg (tune_hash persists -- no underscore), so graph_res can only
        # resume or skip a checkpoint belonging to THIS candidate.
        cfg["tune_hash"] = candidate_hash("t2", idx, hp)
        cfg["_stem"] = _t2_trial_stem(idx, hp, quick, k)
        cfg["_init_stem"] = init_stem
        cfg["_train_inner_folds"] = list(TRAIN_INNER)
        cfg["_eval_inner_folds"] = list(EVAL_INNER)
        cfg["_eval_heldout"] = True
        cfg["_early_stop"] = dict(TRIAL_ES)
        _say(f"t2 candidate {idx} context {k}: {int(tr_rows.sum()):,} "
             f"selection rows / {int(ev_rows.sum()):,} confirmation rows "
             f"({n_masked:,} outer-test/vault residual cells masked)")
        graph_res.finetune(cfg, (k, TRIAL_CAL_J))
        ck = graph_res.load_checkpoint(cfg["_stem"])
        fold_r2[str(k)] = float(ck["cfg"].get("heldout_r2", float("nan")))
    return fold_r2


# ── T3 trial ────────────────────────────────────────────────────────────────

def _masked_shared_t3(shared, k):
    """Shallow copy of the T3 shared dict with the residual targets
    NaN-masked at every outer-fold-k and vault row -- the field-tier
    equivalent of _mask_t2_residual. The loss and score masks already
    exclude these rows; the masking makes any read structurally
    impossible."""
    kill = (np.asarray(shared["outer"]) == int(k)) | shared["vault"]
    out = dict(shared)
    r2 = shared["r2"].copy()
    r2[kill] = np.nan
    out["r2"] = r2
    out["r2_by_k"] = {}
    for kk, v in (shared.get("r2_by_k") or {}).items():
        v2 = v.copy()
        v2[kill] = np.nan
        out["r2_by_k"][kk] = v2
    return out


def _t3_trial_pretrain(hp, quick):
    """Short trial pretrain for a non-default mask ratio, cached by value
    (the mask-ratio grid is discrete for exactly this reason) under
    private checkpoint paths. Seeded from SEED + TUNE_PRETRAIN_SALT."""
    import field_res
    tag = f"m{int(round(float(hp['mask_ratio']) * 100)):02d}"
    if quick:
        tag += "_quick"     # smoke pretrains never seed a full search
    last = config2.artifact(f"fieldpre_tune_{tag}_last.pt", "field")
    best = config2.artifact(f"fieldpre_tune_{tag}_best.pt", "field")
    state = config2.artifact(f"fieldpre_tune_{tag}_state.json")
    if os.path.exists(state):
        pre = best if os.path.exists(best) else last
        _say(f"t3 trial pretrain {tag}: complete -- reused ({pre})")
        return pre
    cfg = {"quick": quick, "resume": True, "hp": dict(hp),
           "seed": int(config2.SEED + TUNE_PRETRAIN_SALT),
           "epochs": (field_res.QUICK_EPOCHS if quick
                      else TRIAL_PRETRAIN_EPOCHS_T3),
           "_paths": (last, best, state)}
    _say(f"t3 trial pretrain {tag}: {cfg['epochs']} epochs "
         f"(mask_ratio {hp['mask_ratio']})")
    return field_res.pretrain(cfg)


def _run_t3_trial(idx, hp, quick, ks, caches):
    """One T3 candidate: INR heads fit on the SELECTION folds per context
    and scored on the CONFIRMATION folds. Encoder features are cached per
    (encoder checkpoint, fourier ladder) and shared across contexts and
    candidates."""
    import field_res
    torch = field_res._require_torch()[0]
    defaults = field_res.tier_defaults()
    seed = int(config2.SEED + TRIAL_FIT_SALT * (idx + 1))

    pre_path = None
    if float(hp["mask_ratio"]) != float(defaults["mask_ratio"]):
        pre_path = _t3_trial_pretrain(hp, quick)

    wl = tuple(field_res.fourier_wavelengths(dict(defaults, **hp)))
    skey = (pre_path or "default", wl)
    shared_cache = caches.setdefault("t3_shared", {})
    if skey not in shared_cache:
        cfg_s = {"quick": quick, "hp": dict(hp), "seed": seed}
        if pre_path:
            cfg_s["pretrain_ckpt"] = pre_path
        shared_cache[skey] = field_res._build_shared(cfg_s)
    shared = shared_cache[skey]
    frame, folds = shared["frame"], shared["folds"]

    fold_r2 = {}
    for k in ks:
        tr_rows, ev_rows = selection_masks(frame, folds, k)
        sh_k = _masked_shared_t3(shared, k)
        # Candidate provenance: hash in the checkpoint filename AND its
        # stored cfg, exactly as in the T2 trial (field_res honors it).
        cfg = {"quick": quick, "hp": dict(hp), "seed": seed,
               "tune_hash": candidate_hash("t3", idx, hp),
               "ft_epochs": (field_res.QUICK_EPOCHS if quick
                             else int(hp["ft_epochs"])),
               "ft_lr": float(hp["ft_lr"]),
               "_out_path": config2.artifact(
                   _t3_trial_name(idx, hp, quick, k), "field"),
               "_train_inner_folds": list(TRAIN_INNER),
               "_eval_inner_folds": list(EVAL_INNER),
               "_early_stop": dict(TRIAL_ES)}
        if pre_path:
            cfg["pretrain_ckpt"] = pre_path
        _say(f"t3 candidate {idx} context {k}: {int(tr_rows.sum()):,} "
             f"selection rows / {int(ev_rows.sum()):,} confirmation rows")
        path = field_res.finetune(cfg, (k, TRIAL_CAL_J), shared=sh_k)
        ck = field_res._load_ckpt(torch, path)
        fold_r2[str(k)] = float(ck["cfg"].get("heldout_r2", float("nan")))
    return fold_r2


# ── Search driver ───────────────────────────────────────────────────────────

def run_search(tier, quick=False):
    """Run (or resume) one tier's search; write the tune artifact."""
    tier = str(tier).lower()
    if tier not in BUDGET:
        raise SystemExit(f"[aqnet2] tune_deep: unknown tier {tier!r}")
    dest = config2.artifact(ARTIFACT[tier])
    if quick:
        # A quick smoke must never satisfy the production sentinel: it
        # gets its own artifact name (and the tiers' load_tuned_hp refuse
        # a quick payload regardless of where it lands).
        dest = dest.replace(".json", "_quick.json")
    if os.path.exists(dest) and os.environ.get("FORCE") != "1":
        _say(f"{dest} exists -- skipping (FORCE=1 to re-run)")
        return 0
    budget = QUICK_BUDGET if quick else BUDGET[tier]
    ks = list(QUICK_KS if quick else KS)
    cands = draw_candidates(tier, budget)
    part_path = dest.replace(".json", "_partial.json")

    trials = []
    if os.path.exists(part_path) and os.environ.get("FORCE") != "1":
        with open(part_path, encoding="utf-8") as fh:
            part = json.load(fh)
        same = (part.get("seed") == config2.SEED
                and part.get("quick") == bool(quick)
                and part.get("candidates") == _jsonable(cands))
        if same:
            trials = list(part.get("trials") or [])
            _say(f"resuming from {part_path} ({len(trials)}/{budget} "
                 f"candidates done)")
        else:
            _say(f"{part_path} provenance mismatch (seed/quick/candidate "
                 f"drift) -- discarding it and restarting the search")
            _clear_trial_checkpoints(tier)

    trial_fn = _run_t2_trial if tier == "t2" else _run_t3_trial
    caches = {}

    def _trial_row(idx):
        hp = cands[idx]
        t0 = time.time()
        row = {"index": idx, "config": dict(hp), "fold_r2": None,
               "mean_r2": None, "wall_sec": None, "status": "failed",
               "error": None}
        try:
            fold_r2 = trial_fn(idx, hp, quick, ks, caches)
            vals = [fold_r2[str(k)] for k in ks]
            if all(v is not None and np.isfinite(v) for v in vals):
                row.update(fold_r2=fold_r2, mean_r2=float(np.mean(vals)),
                           status="ok", error=None)
            else:
                row.update(fold_r2=fold_r2,
                           error="non-finite confirmation score")
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:  # noqa: BLE001 -- a failed trial never wins
            traceback.print_exc()
            row["error"] = f"{type(e).__name__}: {e}"
        row["wall_sec"] = round(time.time() - t0, 1)
        return row

    def _save_partial():
        _write_json_atomic({"tier": tier, "seed": int(config2.SEED),
                            "quick": bool(quick), "candidates": cands,
                            "trials": trials}, part_path)

    early_stopped = False
    for idx in range(len(trials), budget):
        if plateau_stop(trials, tier):
            early_stopped = True
            _say(f"inner-selection plateau after {len(trials)} candidates "
                 f"-- surrendering the remaining budget")
            break
        _say(f"candidate {idx}/{budget - 1} ({tier}): {cands[idx]}")
        row = _trial_row(idx)
        trials.append(row)
        _save_partial()
        _say(f"candidate {idx}: status={row['status']} "
             f"mean_r2={row['mean_r2']} ({row['wall_sec']}s)")

    # Candidate 0 holds the registered fallback seat: if it ended unscored
    # (e.g. a preemption-scarred trial), any scored challenger would win
    # with no tie protection. Retry it once before selection; if it stays
    # unscored, proceed but say so loudly and record it in the artifact.
    default_unscored = False
    i0 = next((i for i, t in enumerate(trials)
               if int(t.get("index", -1)) == 0), None)
    if i0 is not None and not _finite_score(trials[i0]):
        _say("candidate 0 (the registered default) is unscored -- "
             "retrying it once before selection")
        row = _trial_row(0)
        row["retried"] = True
        trials[i0] = row
        _save_partial()
        if not _finite_score(row):
            default_unscored = True
            _say("WARNING: candidate 0 (the registered default) is STILL "
                 "unscored after a retry -- selection proceeds without "
                 "the fallback's score (default_unscored=true in the "
                 "artifact)")

    win_idx, fallback = select_winner(trials)
    payload = {
        "tier": tier, "quick": bool(quick), "completed": True,
        "protocol": PROTOCOL,
        "seed": int(config2.SEED),
        "seed_salts": {
            "draw": T2_DRAW_SALT if tier == "t2" else T3_DRAW_SALT,
            "trial_fit": TRIAL_FIT_SALT,
            "trial_pretrain": TUNE_PRETRAIN_SALT,
        },
        "outer_contexts": ks,
        "train_inner_folds": list(TRAIN_INNER),
        "eval_inner_folds": list(EVAL_INNER),
        "budget": budget, "n_run": len(trials),
        "early_stopped": early_stopped,
        "default_unscored": bool(default_unscored),
        "winner_index": int(win_idx),
        "winner_is_fallback": bool(fallback),
        "winner": dict(cands[win_idx]),
        "trials": trials,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json_atomic(payload, dest)
    if os.path.exists(part_path):
        os.remove(part_path)
    _say(f"winner: candidate {win_idx}"
         + (" (registered fallback)" if fallback else "")
         + f" -> {dest}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Registered v4 deep-tier hyperparameter search "
                    "(T2 graph / T3 field; EXPANSION.md protocol)")
    ap.add_argument("--tier", required=True, choices=["t2", "t3"])
    ap.add_argument("--quick", action="store_true",
                    help="smoke: tiny budget, 2 outer contexts, 2-epoch "
                         "fits -- a machinery check, never a search; "
                         "writes tune_t{2,3}_quick.json, never the "
                         "production sentinel")
    args = ap.parse_args(argv)
    return run_search(args.tier, quick=args.quick)


if __name__ == "__main__":
    sys.exit(main())
