"""Registered v4 deep-tier hyperparameter search driver (tune_deep.py).

The frozen-behavior contract is the anchor: candidate 0 of every draw IS
the tiers' current default config, absent tune artifacts resolve to the
exact module constants (byte-identical plumbing), and the protocol masks
provably exclude outer-fold test rows and the vault from every trial.

All synthetic fixtures (conftest frame_truth/folds plus inline trial
tables); no repo data files, no network, no torch.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

import config2
import field_res
import graph_res
import pipeline2
import tune_deep


# ── (1) deterministic draws, candidate 0 = the current defaults ─────────────

def test_draws_deterministic_and_candidate0_is_default():
    for tier, defaults in (("t2", graph_res.tier_defaults()),
                           ("t3", field_res.tier_defaults())):
        a = tune_deep.draw_candidates(tier)
        b = tune_deep.draw_candidates(tier)
        assert a == b, f"{tier} draws are not deterministic under the seed"
        assert a[0] == defaults, f"{tier} candidate 0 != current defaults"
        assert len(a) == tune_deep.BUDGET[tier]


def test_t2_draws_stay_inside_the_frozen_space():
    for c in tune_deep.draw_candidates("t2")[1:]:
        assert c["d_model"] in tune_deep.T2_SPACE["d_model"]
        assert c["n_heads"] in tune_deep.T2_SPACE["n_heads"]
        assert c["d_model"] % c["n_heads"] == 0
        assert c["n_layers"] in tune_deep.T2_SPACE["n_layers"]
        lo, hi = tune_deep.T2_SPACE["d_ff_mult"]
        assert lo * c["d_model"] - 1 <= c["d_ff"] <= hi * c["d_model"] + 1
        lo, hi = tune_deep.T2_SPACE["dropout"]
        assert lo <= c["dropout"] <= hi
        lo, hi = tune_deep.T2_SPACE["lr_finetune"]
        assert lo <= c["lr_finetune"] <= hi
        lo, hi = tune_deep.T2_SPACE["weight_decay"]
        assert lo <= c["weight_decay"] <= hi
        assert c["knn_k"] in tune_deep.T2_SPACE["knn_k"]
        lo, hi = tune_deep.T2_SPACE["finetune_epochs"]
        assert lo <= c["finetune_epochs"] <= hi


def test_t3_draws_stay_inside_the_frozen_space():
    for c in tune_deep.draw_candidates("t3")[1:]:
        assert c["fourier_n"] in tune_deep.T3_SPACE["fourier_n"]
        lo, hi = tune_deep.T3_SPACE["fourier_min_km"]
        assert lo <= c["fourier_min_km"] <= hi
        assert c["fourier_min_km"] >= 5.0   # the hallucination hard cap
        assert c["inr_width"] in tune_deep.T3_SPACE["inr_width"]
        assert c["inr_depth"] in tune_deep.T3_SPACE["inr_depth"]
        lo, hi = tune_deep.T3_SPACE["ft_lr"]
        assert lo <= c["ft_lr"] <= hi
        assert c["mask_ratio"] in tune_deep.T3_SPACE["mask_ratio"]
        lo, hi = tune_deep.T3_SPACE["ft_epochs"]
        assert lo <= c["ft_epochs"] <= hi


# ── (2) winner selection: higher mean wins; tie/all-failed -> candidate 0 ───

def _trial(i, mean):
    return {"index": i, "mean_r2": mean,
            "status": "ok" if mean is not None else "failed"}


def test_winner_prefers_higher_mean():
    idx, fb = tune_deep.select_winner(
        [_trial(0, 0.10), _trial(1, 0.30), _trial(2, 0.20)])
    assert idx == 1 and fb is False


def test_winner_default_keeps_seat_when_best():
    idx, fb = tune_deep.select_winner([_trial(0, 0.30), _trial(1, 0.10)])
    assert idx == 0 and fb is False


def test_winner_tie_falls_back_to_candidate0():
    idx, _fb = tune_deep.select_winner([_trial(0, 0.20), _trial(1, 0.20)])
    assert idx == 0
    idx, fb = tune_deep.select_winner(
        [_trial(0, 0.20), _trial(1, 0.20 + tune_deep.TIE_EPS / 2)])
    assert idx == 0 and fb is True


def test_winner_all_failed_falls_back_to_candidate0():
    idx, fb = tune_deep.select_winner([_trial(0, None), _trial(1, None)])
    assert idx == 0 and fb is True
    idx, fb = tune_deep.select_winner(
        [_trial(0, float("nan")), _trial(1, None)])
    assert idx == 0 and fb is True


def test_winner_failed_default_still_loses_to_a_scored_challenger():
    idx, fb = tune_deep.select_winner([_trial(0, None), _trial(1, 0.05)])
    assert idx == 1 and fb is False


def test_plateau_surrenders_budget_only_after_patience():
    pat = tune_deep.PLATEAU_PATIENCE["t2"]
    rising = [_trial(i, 0.1 + 0.05 * i) for i in range(pat + 2)]
    assert tune_deep.plateau_stop(rising, "t2") is False
    flat = [_trial(0, 0.30)] + [_trial(i + 1, 0.10) for i in range(pat)]
    assert tune_deep.plateau_stop(flat, "t2") is True
    assert tune_deep.plateau_stop(flat[:pat], "t2") is False


# ── (3) protocol masks: no outer-test rows, no vault rows, ever ─────────────

def test_selection_masks_exclude_outer_test_and_vault(frame_truth, folds):
    frame = frame_truth.copy()
    # Plant vault-PERIOD rows so the period airlock is exercised too; the
    # synthetic folds deliberately leave vault units inside the inner
    # arrays, so the vault-unit exclusion must come from the mask builder.
    frame.loc[frame.index[:5], "date"] = pd.Timestamp("2026-02-01")
    vault_units = {f"aqs_{s}" for s in folds["vault_sites"]}
    vault_rows = frame["unit_id"].astype(str).isin(vault_units).to_numpy()
    period_rows = (frame["date"]
                   >= pd.Timestamp(config2.VAULT_DATE_START)).to_numpy()
    outer = np.asarray(folds["outer_fold"], dtype=np.int64)
    for k in tune_deep.KS:
        train, ev = tune_deep.selection_masks(frame, folds, k)
        inner = np.asarray(folds["inner_fold"][str(k)], dtype=np.int64)
        assert train.any() and ev.any()
        assert not (train & ev).any()
        assert not (train & (outer == k)).any()
        assert not (ev & (outer == k)).any()
        for m in (train, ev):
            assert not (m & vault_rows).any(), "vault-unit rows leaked"
            assert not (m & period_rows).any(), "vault-period rows leaked"
        assert np.isin(inner[train], tune_deep.TRAIN_INNER).all()
        assert np.isin(inner[ev], tune_deep.EVAL_INNER).all()


def test_selection_masks_reject_misaligned_folds(frame_truth, folds):
    bad = dict(folds)
    bad["outer_fold"] = folds["outer_fold"][:-1]
    with pytest.raises(SystemExit):
        tune_deep.selection_masks(frame_truth, bad, 0)


# ── (4) config plumbing: absent artifact = the constants; present = applied ─

def test_hp_defaults_equal_module_constants():
    d = graph_res.tier_defaults()
    assert d["d_model"] == graph_res.D_MODEL
    assert d["n_heads"] == graph_res.N_HEADS
    assert d["n_layers"] == graph_res.N_LAYERS
    assert d["d_ff"] == graph_res.D_FF
    assert d["dropout"] == 0.0
    assert d["lr_finetune"] == graph_res.LR_FINETUNE
    assert d["weight_decay"] == graph_res.WEIGHT_DECAY
    assert d["knn_k"] == graph_res.KNN_K
    assert d["finetune_epochs"] == graph_res.FINETUNE_EPOCHS
    assert graph_res._hp({}) == d
    assert graph_res._hp(None) == d

    f = field_res.tier_defaults()
    assert f["fourier_n"] == len(field_res.FOURIER_WAVELENGTHS_KM)
    assert f["fourier_min_km"] == field_res.FOURIER_WAVELENGTHS_KM[0]
    assert f["inr_width"] == field_res.INR_HIDDEN
    assert f["inr_depth"] == field_res.INR_DEPTH
    assert f["ft_lr"] == field_res.LR_FINETUNE
    assert f["mask_ratio"] == field_res.MASK_RATIO
    assert f["ft_epochs"] == field_res.FINETUNE_EPOCHS
    assert field_res._hp({}) == f


def test_hp_overrides_apply_and_leave_the_rest_at_defaults():
    over = graph_res._hp({"hp": {"d_model": 192, "knn_k": 15}})
    assert over["d_model"] == 192 and over["knn_k"] == 15
    assert over["n_heads"] == graph_res.N_HEADS
    assert over["lr_finetune"] == graph_res.LR_FINETUNE
    fo = field_res._hp({"hp": {"inr_depth": 4}})
    assert fo["inr_depth"] == 4
    assert fo["inr_width"] == field_res.INR_HIDDEN


def test_fourier_wavelengths_default_is_the_exact_frozen_list():
    f = field_res.tier_defaults()
    assert (field_res.fourier_wavelengths(f)
            == list(field_res.FOURIER_WAVELENGTHS_KM))
    wl = field_res.fourier_wavelengths(dict(f, fourier_n=5,
                                            fourier_min_km=8.0))
    assert len(wl) == 5
    assert wl[0] == pytest.approx(8.0)
    assert wl[-1] == pytest.approx(field_res.FOURIER_WAVELENGTHS_KM[-1])
    assert all(a < b for a, b in zip(wl, wl[1:]))


def test_load_tuned_hp_absent_and_incomplete_mean_defaults(tmp_path,
                                                           monkeypatch):
    monkeypatch.setattr(config2, "ARTIFACTS_DIR", str(tmp_path))
    assert graph_res.load_tuned_hp() is None
    assert field_res.load_tuned_hp() is None
    # An in-flight (incomplete) artifact must never leak into a run.
    with open(tmp_path / "tune_t2.json", "w", encoding="utf-8") as fh:
        json.dump({"completed": False, "winner": {"d_model": 192}}, fh)
    assert graph_res.load_tuned_hp() is None


def test_load_tuned_hp_present_applies_winner(tmp_path, monkeypatch):
    monkeypatch.setattr(config2, "ARTIFACTS_DIR", str(tmp_path))
    winner = dict(graph_res.tier_defaults(), d_model=192, lr_finetune=3e-4)
    with open(tmp_path / "tune_t2.json", "w", encoding="utf-8") as fh:
        json.dump({"completed": True, "winner_index": 3,
                   "winner_is_fallback": False, "winner": winner}, fh)
    hp = graph_res.load_tuned_hp()
    assert hp["d_model"] == 192 and hp["lr_finetune"] == 3e-4
    assert hp["n_heads"] == graph_res.N_HEADS   # unnamed keys stay default
    w3 = dict(field_res.tier_defaults(), mask_ratio=0.75)
    with open(tmp_path / "tune_t3.json", "w", encoding="utf-8") as fh:
        json.dump({"completed": True, "winner_index": 1,
                   "winner_is_fallback": False, "winner": w3}, fh)
    hp3 = field_res.load_tuned_hp()
    assert hp3["mask_ratio"] == 0.75
    assert hp3["inr_width"] == field_res.INR_HIDDEN


def test_run_search_skips_completed_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(config2, "ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.delenv("FORCE", raising=False)
    with open(tmp_path / "tune_t2.json", "w", encoding="utf-8") as fh:
        json.dump({"completed": True, "winner_index": 0,
                   "winner": graph_res.tier_defaults()}, fh)
    assert tune_deep.run_search("t2") == 0   # resume contract: exit-0 fast


# ── (5) pipeline2 stage order and sentinel registration ─────────────────────

def test_pipeline_stage_order_and_sentinels():
    order = pipeline2._STAGE_ORDER
    assert order.index("graphtune") == order.index("graphpre") + 1
    assert order.index("graphres") == order.index("graphtune") + 1
    assert order.index("fieldtune") == order.index("fieldpre") + 1
    assert order.index("fieldres") == order.index("fieldtune") + 1
    assert pipeline2.SENTINELS["graphtune"] == ("tune_t2.json",)
    assert pipeline2.SENTINELS["fieldtune"] == ("tune_t3.json",)
    assert callable(pipeline2._STAGES["graphtune"])
    assert callable(pipeline2._STAGES["fieldtune"])
    # The sentinel names match the artifacts the tiers' loaders read.
    assert pipeline2.SENTINELS["graphtune"][0] == graph_res.TUNE_ARTIFACT
    assert pipeline2.SENTINELS["fieldtune"][0] == field_res.TUNE_ARTIFACT


# ── (6) trial provenance: candidate hash, stems, drift hygiene ──────────────

def test_candidate_hash_tracks_config_index_space_and_seed(monkeypatch):
    hp = graph_res.tier_defaults()
    orig_space = tune_deep.T2_SPACE
    h = tune_deep.candidate_hash("t2", 0, hp)
    assert h == tune_deep.candidate_hash("t2", 0, dict(hp))   # stable
    assert h != tune_deep.candidate_hash("t2", 1, hp)         # trial seed
    assert h != tune_deep.candidate_hash("t2", 0, dict(hp, d_model=192))
    assert h != tune_deep.candidate_hash("t3", 0, hp)         # tier
    # A frozen-space edit changes every hash (stale checkpoints rebuild).
    monkeypatch.setattr(tune_deep, "T2_SPACE",
                        dict(orig_space, knn_k=(6, 8)))
    assert h != tune_deep.candidate_hash("t2", 0, hp)
    # So does a SEED change (the trial-fit seed is part of the payload).
    monkeypatch.setattr(tune_deep, "T2_SPACE", orig_space)
    monkeypatch.setattr(config2, "SEED", config2.SEED + 1)
    assert h != tune_deep.candidate_hash("t2", 0, hp)


def test_trial_stems_carry_hash_and_quick_namespace():
    hp = graph_res.tier_defaults()
    h = tune_deep.candidate_hash("t2", 3, hp)
    full = tune_deep._t2_trial_stem(3, hp, False, 0)
    quick = tune_deep._t2_trial_stem(3, hp, True, 0)
    assert h in full and h in quick
    assert full.startswith("graphtune_t03_")
    assert quick.startswith("graphtune_quick_t03_")
    assert full != quick   # a smoke can never resume or skip a full trial
    hp3 = field_res.tier_defaults()
    h3 = tune_deep.candidate_hash("t3", 2, hp3)
    name = tune_deep._t3_trial_name(2, hp3, False, 1)
    qname = tune_deep._t3_trial_name(2, hp3, True, 1)
    assert h3 in name and name.endswith("_f1.pt")
    assert name.startswith("fieldtune_t02_")
    assert qname.startswith("fieldtune_quick_t02_")


def test_clear_trial_checkpoints_scopes_to_trial_namespaces(tmp_path,
                                                            monkeypatch):
    monkeypatch.setattr(config2, "ARTIFACTS_DIR", str(tmp_path))
    graph = tmp_path / "graph"
    field = tmp_path / "field"
    graph.mkdir()
    field.mkdir()
    stale_t2 = [graph / "graphtune_t01_deadbeef00_f0_last.pt",
                graph / "graphpre_tune_cafe0123_last.pt"]
    stale_t3 = [field / "fieldtune_t00_ab12cd34ef_f0.pt",
                field / "fieldpre_tune_m45_last.pt",
                tmp_path / "fieldpre_tune_m45_state.json"]
    keep = [graph / "graphpre_last.pt", graph / "graphres_f0_0_last.pt",
            field / "fieldpre_primary_last.pt",
            field / "fieldres_primary_f0_0.pt",
            tmp_path / "fieldpre_state.json"]
    for p in stale_t2 + stale_t3 + keep:
        p.write_bytes(b"x")
    tune_deep._clear_trial_checkpoints("t2")
    tune_deep._clear_trial_checkpoints("t3")
    for p in stale_t2 + stale_t3:
        assert not p.exists(), f"stale trial file survived: {p.name}"
    for p in keep:
        assert p.exists(), f"production file removed: {p.name}"


def test_run_search_drift_restart_clears_stale_trial_checkpoints(
        tmp_path, monkeypatch):
    monkeypatch.setattr(config2, "ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.delenv("FORCE", raising=False)
    monkeypatch.setitem(tune_deep.BUDGET, "t2", 2)
    graph = tmp_path / "graph"
    graph.mkdir()
    stale = graph / "graphtune_t01_deadbeef00_f0_last.pt"
    stale.write_bytes(b"x")
    with open(tmp_path / "tune_t2_partial.json", "w",
              encoding="utf-8") as fh:
        json.dump({"seed": config2.SEED + 1, "quick": False,
                   "candidates": [], "trials": [{"index": 0,
                                                 "mean_r2": 0.9}]}, fh)
    monkeypatch.setattr(
        tune_deep, "_run_t2_trial",
        lambda idx, hp, quick, ks, caches: {str(k): 0.1 for k in ks})
    assert tune_deep.run_search("t2") == 0
    assert not stale.exists(), "drift restart left a stale trial checkpoint"
    with open(tmp_path / "tune_t2.json", encoding="utf-8") as fh:
        art = json.load(fh)
    assert art["n_run"] == 2   # the drifted partial's trial was discarded


# ── (7) checkpoint completeness and resume guards ───────────────────────────

def test_t2_trial_checkpoint_completeness_guards():
    trial_cfg = {"tune_hash": "abc", "_eval_heldout": True}
    done = {"epoch": 9, "cfg": {"tune_hash": "abc", "heldout_r2": 0.2}}
    assert graph_res._trial_ckpt_complete(done, trial_cfg, 10) is True
    # Preemption between the last-epoch save and the eval save: the
    # checkpoint must read INCOMPLETE (rebuild), never permanently failed.
    no_eval = {"epoch": 9, "cfg": {"tune_hash": "abc"}}
    assert graph_res._trial_ckpt_complete(no_eval, trial_cfg, 10) is False
    # Another candidate's checkpoint (hash mismatch) can never skip.
    stale = {"epoch": 9, "cfg": {"tune_hash": "zzz", "heldout_r2": 0.2}}
    assert graph_res._trial_ckpt_complete(stale, trial_cfg, 10) is False
    short = {"epoch": 5, "cfg": {"tune_hash": "abc", "heldout_r2": 0.2}}
    assert graph_res._trial_ckpt_complete(short, trial_cfg, 10) is False
    assert graph_res._trial_ckpt_complete(None, trial_cfg, 10) is False
    # Production cfgs carry neither key: the frozen skip semantics.
    assert graph_res._trial_ckpt_complete({"epoch": 9, "cfg": {}}, {},
                                          10) is True


def test_resume_guards_discard_mismatched_checkpoints():
    d = graph_res.tier_defaults()
    ck_default = {"cfg": {}}    # pre-tune checkpoint: no hp recorded
    assert graph_res._resume_arch_matches(ck_default, d) is True
    tuned = dict(d, d_model=160, d_ff=1440)
    assert graph_res._resume_arch_matches(ck_default, tuned) is False
    ck_tuned = {"cfg": {"hp": dict(tuned)}}
    assert graph_res._resume_arch_matches(ck_tuned, tuned) is True
    assert graph_res._resume_arch_matches(ck_tuned, d) is False
    # Non-architecture keys never trigger a discard.
    assert graph_res._resume_arch_matches(ck_default,
                                          dict(d, knn_k=15)) is True
    # fieldpre mirror: mask_ratio decides; absent = the frozen default.
    assert field_res._resume_mask_matches({"mask_ratio": 0.60}, 0.60) is True
    assert field_res._resume_mask_matches({"mask_ratio": 0.60}, 0.45) \
        is False
    assert field_res._resume_mask_matches({}, field_res.MASK_RATIO) is True


def test_t3_encoder_parity_exits_loudly_on_mask_mismatch():
    hp = dict(field_res.tier_defaults(), mask_ratio=0.45)
    with pytest.raises(SystemExit) as ei:
        field_res._assert_encoder_parity(hp, {"mask_ratio": 0.60}, "enc.pt")
    msg = str(ei.value)
    assert "FORCE=1" in msg and "fieldpre" in msg and "mask_ratio" in msg
    # Parity passes when the encoder matches, and for default hp against a
    # pre-tune checkpoint (recorded default) -- the frozen contract.
    field_res._assert_encoder_parity(hp, {"mask_ratio": 0.45}, "enc.pt")
    field_res._assert_encoder_parity(
        field_res.tier_defaults(), {"mask_ratio": field_res.MASK_RATIO},
        "enc.pt")


# ── (8) quick smoke isolation: artifact naming + loader refusal ─────────────

def test_load_tuned_hp_refuses_quick_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(config2, "ARTIFACTS_DIR", str(tmp_path))
    winner = dict(graph_res.tier_defaults(), d_model=192)
    with open(tmp_path / "tune_t2.json", "w", encoding="utf-8") as fh:
        json.dump({"completed": True, "quick": True, "winner_index": 1,
                   "winner": winner}, fh)
    assert graph_res.load_tuned_hp() is None
    w3 = dict(field_res.tier_defaults(), mask_ratio=0.75)
    with open(tmp_path / "tune_t3.json", "w", encoding="utf-8") as fh:
        json.dump({"completed": True, "quick": True, "winner_index": 1,
                   "winner": w3}, fh)
    assert field_res.load_tuned_hp() is None


def test_run_search_quick_writes_only_the_quick_artifact(tmp_path,
                                                         monkeypatch):
    monkeypatch.setattr(config2, "ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.delenv("FORCE", raising=False)
    monkeypatch.setattr(
        tune_deep, "_run_t2_trial",
        lambda idx, hp, quick, ks, caches: {str(k): 0.1 + 0.01 * idx
                                            for k in ks})
    assert tune_deep.run_search("t2", quick=True) == 0
    assert (tmp_path / "tune_t2_quick.json").exists()
    assert not (tmp_path / "tune_t2.json").exists()
    with open(tmp_path / "tune_t2_quick.json", encoding="utf-8") as fh:
        art = json.load(fh)
    assert art["quick"] is True and art["completed"] is True
    # The production loader would refuse this payload even if it were
    # copied over the sentinel name by hand.
    os.replace(tmp_path / "tune_t2_quick.json", tmp_path / "tune_t2.json")
    assert graph_res.load_tuned_hp() is None
    os.replace(tmp_path / "tune_t2.json", tmp_path / "tune_t2_quick.json")
    # A completed quick artifact short-circuits only another quick run...
    assert tune_deep.run_search("t2", quick=True) == 0   # fast quick skip
    # ...while a full search still runs and writes the real sentinel.
    monkeypatch.setitem(tune_deep.BUDGET, "t2", 2)
    assert tune_deep.run_search("t2") == 0
    with open(tmp_path / "tune_t2.json", encoding="utf-8") as fh:
        assert json.load(fh)["quick"] is False


# ── (9) driver bookkeeping: the candidate-0 fallback seat ───────────────────

def test_run_search_retries_unscored_candidate0_once(tmp_path, monkeypatch):
    monkeypatch.setattr(config2, "ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.delenv("FORCE", raising=False)
    monkeypatch.setitem(tune_deep.BUDGET, "t2", 2)
    calls = {"n0": 0}

    def fake_trial(idx, hp, quick, ks, caches):
        if idx == 0:
            calls["n0"] += 1
            if calls["n0"] == 1:   # preemption-scarred first attempt
                return {str(k): float("nan") for k in ks}
            return {str(k): 0.30 for k in ks}
        return {str(k): 0.10 for k in ks}

    monkeypatch.setattr(tune_deep, "_run_t2_trial", fake_trial)
    assert tune_deep.run_search("t2") == 0
    with open(tmp_path / "tune_t2.json", encoding="utf-8") as fh:
        art = json.load(fh)
    assert calls["n0"] == 2
    row0 = next(t for t in art["trials"] if t["index"] == 0)
    assert row0["status"] == "ok" and row0.get("retried") is True
    assert art["default_unscored"] is False
    assert art["winner_index"] == 0 and art["winner_is_fallback"] is False
    assert not (tmp_path / "tune_t2_partial.json").exists()


def test_run_search_records_default_unscored_after_failed_retry(
        tmp_path, monkeypatch):
    monkeypatch.setattr(config2, "ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.delenv("FORCE", raising=False)
    monkeypatch.setitem(tune_deep.BUDGET, "t2", 2)
    calls = {"n0": 0}

    def fake_trial(idx, hp, quick, ks, caches):
        if idx == 0:
            calls["n0"] += 1
            return {str(k): float("nan") for k in ks}
        return {str(k): 0.10 for k in ks}

    monkeypatch.setattr(tune_deep, "_run_t2_trial", fake_trial)
    assert tune_deep.run_search("t2") == 0
    with open(tmp_path / "tune_t2.json", encoding="utf-8") as fh:
        art = json.load(fh)
    assert calls["n0"] == 2            # exactly one retry, never a loop
    assert art["default_unscored"] is True
    assert art["winner_index"] == 1    # the scored challenger takes the seat
