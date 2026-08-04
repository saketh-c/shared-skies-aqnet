# AQNet v2 — Frozen cross-module interfaces

Implementation contract companion to DESIGN.md. Every module MUST conform to
the signatures and artifact schemas below; deviations are integration bugs.
Modules use flat same-directory imports (`import config2`, `import folds2`);
`pipeline2.py` owns the sys.path bootstrap (v1 `models_deep.py` pattern).
Target: Python 3.12 (cluster venv). Heavy deps (gpboost, torch, statsmodels)
behind try-import guards with graceful degradation messages, v1 style.

## config2.py

```python
ROOT, AQNET2_DIR, DATA_DIR, CACHE_DIR          # aqnet2-local data/ cache/
ARTIFACTS_DIR                                   # <AQNET2_DIR>/artifacts/v2
V1_DIR                                          # ../aqnet (reuse committed data + modules)
PIPELINE_DIR                                    # <ROOT>/pipeline
def artifact(name: str, sub: str = "") -> str   # ARTIFACTS_DIR[/sub]/name, dirs created
TX_BBOX, GRID_DEG = 0.1, DATE_START = "2021-01-01", DATE_END = "2026-05-01"
TEMPORAL_CUTOFF = "2025-01-01"; TEMPORAL_EMBARGO_DAYS = 7
VAULT_N_SITES = 12; VAULT_BUFFER_KM = 30.0; OUTER_N_FOLDS = 5
INNER_N_FOLDS = 4                               # folds 0-1 selection, 2-3 confirmation
LOSO_N_FOLDS = 10; SEED = 42
EXCLUDED_DEMOGRAPHIC = [...]                    # same 4 names as v1
PORTABLE_FEATURES: list[str]; INTERP_FEATURES: list[str]   # per DESIGN §6
SIGMA_FRM = 1.5
GATE_ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
GATE_MIN_CLUSTERS = 5; GATE_MAX_DIST_KM = 100.0
T4_SLOPE_CLIP = (0.8, 1.25)
```

## folds2.py

```python
def build_folds(frame: pd.DataFrame, seed: int = config2.SEED) -> dict
def save_folds(folds: dict, path: str) -> None
def load_folds(path: str, frame: pd.DataFrame) -> dict   # verifies content_hash; raises on mismatch
def content_hash(frame) -> str                            # sha256 over sorted (unit_id, date, y)
def folds_from_assign(assign: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]  # v1 semantics, -1 = always-train
```

`folds2.json` schema (all per-row arrays are lists over frame rows, row order
= frame order):

```json
{"n_rows": int, "content_hash": str, "seed": int,
 "vault_sites": [site_id...],
 "outer_fold_of_site": {site_id: int},
 "outer_fold": [int per row, -1 for PA rows and vault rows],
 "inner_fold": {"k": [int per row]},          // per outer fold k: 0..3, -1 = fold-k outer rows
 "inner_role": {"k": [0|1|2 per row]},        // 0 sel, 1 conf, 2 excluded
 "loso_fold": {"k": [int per row]},           // nested 10-fold LOSO within outer k
 "spatial_block_fold": [int per row],
 "temporal_is_test": [0|1 per row],
 "conformal_unit": [0|1 per row]}
```

## frame2.py

```python
def build_frame_truth(calibrated_parquet, external_paths, quick=False) -> pd.DataFrame
# columns: unit_id, unit_type ("pa"|"aqs"), network ("PA"|"FRM"), date, lat, lon,
#          y (FRM scale), w (precision weight), cal_var, + all feature columns
def build_point_features(lats, lons, dates, pools, fold_ctx=None) -> pd.DataFrame
# THE single-builder parity path: identical for training rows and arbitrary queries.
# pools carries neighbor/lag sources honoring fold exclusions + vault airlock.
def neighbor_overrides(frame, folds, fold_key) -> dict  # {fold: {col: np.ndarray}}, v1 npz contract f{fold}__{col}
```

## Stage artifacts (all under config2.artifact())

| Artifact | Producer | Schema |
|---|---|---|
| `power_analysis.json` | audit | {"mde_pooled_r2": f, "mde_site_r2": f, "margins": {...}, "t2_killswitch": {"kriging_passes": bool, ...}} |
| `colocation_pairs.parquet` | colocate | site_id, sensor_id, dist_km, n_shared_days |
| `pa_calibrated.parquet` | calibrate | sensor_id, date, pa_raw, pa_cal_full, pa_cal_f{k} (k=0..4), pa_cal_f{k}_{j} (j=0..3), cal_var, channel_reconstructed, dist_to_nearest_frm |
| `calibration_report.json` | calibrate | LOLO metrics vs barkjohn + amt_rht baselines, G0 verdict, by-year bias |
| `prior_downscaler_f{k}.npz` | priors | basis node coords, per-stream beta coefficients, precision weights |
| `oof_tier0.npz` | priors | oof_t0 f8[n], pattern_id i1[n] |
| `oof_tier1.npz` | skeleton | oof f8[n], gp_var f8[n], per_model_{name} f8[n], weights_json str, fold_provenance i1[n] |
| `oof_tier2.npz` | graph_res | oof_r f8[n] (NaN where unavailable), sigma f8[n], avail u1[n], pattern_id i1[n] |
| `oof_tier3.npz` | field_res | same keys as tier2 |
| `gates.json` | compose | {"tier": {"pattern": {"stratum": {"alpha": f, "test": {"delta": f, "ci": [f,f], "n_clusters": int, "decision": str}}}}} — NO incumbent coefficient, NO fill value expressible |
| `oof_composite.npz` | compose | oof_final f8[n], tier_mask u1[n, n_tiers] |
| `exceed_model.json` + `oof_exceed.npz` | exceed | calibrated probs, frozen threshold, admission test result |
| `uq_params.json` + `quantile_oof.npz` | uq | unit scores, NexCP params (rho_s, tau), delta per coverage bin, fitted_against_tier_hash |
| `SUMMARY.md`, `metrics_*.json`, `monotone_report.json`, `parity_report.json` | validate | v1-style tables + vault + bare-site + per-block + attenuation |

## compose.py

```python
def fit_gate(y, incumbent_oof, residual_oof, avail, pattern_id, stratum_id,
             clusters, sel_mask, conf_mask, margins) -> GateResult
# asserts np.isnan(residual_oof[~avail]).all(); alpha grid search on sel rows;
# admission: one-sided paired cluster bootstrap on conf rows; default closed.
def apply_gates(incumbent, residual, avail, pattern_id, stratum_id, gates) -> np.ndarray
# unseen pattern -> alpha 0 (exact passthrough); never fills.
def admission_test(y, pred_a, pred_b, clusters, margins, n_boot=1000, seed=42) -> dict
def save_gates / load_gates(path)
```

`tests/test_compose.py` MUST include the permanent regression test: a
synthetic component finite on 17% of rows must produce composite ≡ incumbent
(bit-identical) on the other 83%, and must be rejected by admission when its
finite-row skill is negative.

## calibrate.py

```python
def build_pairs(...) -> pd.DataFrame                      # colocate.py may own this
def fit_calibration(pairs, pa_daily, folds, outer_k=None, inner_j=None) -> CalModel
def apply_calibration(model, pa_daily) -> (pm25_cal, cal_var)
def cal_var_floor(dist_to_nearest_pair_km) -> np.ndarray  # monotone, conservative
def lolo_validate(pairs, ...) -> dict                     # vs barkjohn + amt_rht forms
```

## graph_res.py / field_res.py

Both expose: `pretrain(cfg) -> ckpt_path`, `finetune(cfg, fold) -> ckpt_path`,
`predict_oof(frame, folds, ckpts) -> dict` (writes their oof npz), plus
`main()` CLI mirroring v1 `models_deep.py`. Checkpoints: atomic tmp+os.replace,
save every 30 min AND every epoch, keys {model, optimizer, scheduler, scaler,
rng_state, epoch, cfg, fold_id}; `--resume` auto-detects `last.pt`.
graph_res deployment-honest rule: AQS observations never appear as input
nodes; vault sites never appear at all (assert against folds2 vault list).

## uq.py / exceed.py

```python
def unit_scores(y, pred, sigma, unit_id, mask) -> pd.DataFrame   # one score per unit
def nexcp_delta(scores_df, query_latlon, coverage_bin, rho_s, tau) -> float
def fit_quantile_heads(frame, oof_final, folds, tier_hash) -> ...  # records lineage
def fit_exceed(frame, oof_final, folds, thresholds=(9.0, 35.4)) -> ...
```

## pipeline2.py

Stages, in order: `audit data-pa data statics colocate calibrate priors
features skeleton graphpre graphres fieldpre fieldres gates exceed uq
validate export all`. Each stage: idempotent (exits 0 fast if its sentinel
artifact exists and inputs unchanged), `FORCE=1` re-runs, `--quick` shrinks
(3-month window, 2 outer folds, 2 epochs). Communication between stages ONLY
via artifacts. v1 conventions for logging (`[aqnet2] ── stage: X ──`).

## slurm2/

`aq2-<stage>.sbatch` per GPU/long stage + `submit.sh` building the afterok
DAG per DESIGN §13; `common.sh` shared preamble (account gts-ar70, embers,
venv activate, scratch paths). Names are frozen — renames break resume.

## Testing floor

`tests/` must cover: folds2 vault/buffer/hash invariants; compose passthrough
+ 17%-finite regression + admission determinism (seeded); calibrate LOLO
plumbing with synthetic pairs; frame2 parity (build_point_features at
training coordinates reproduces training features); no module imports fail
when optional heavy deps are absent (torch/gpboost guarded).
