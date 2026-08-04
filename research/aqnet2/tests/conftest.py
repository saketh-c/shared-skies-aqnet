"""Shared fixtures for the aqnet2 test suite.

sys.path bootstrap: pytest may be launched from anywhere (repo root, the
aqnet2 dir, or the tests dir), so the three import roots the flat aqnet2
modules assume — research/aqnet2 itself, research/aqnet (v1 reuse) and
ROOT/pipeline (neighbor_features.py) — are prepended here, mirroring the
pipeline2.py bootstrap contract (INTERFACES.md).

Fixtures are tiny and fully synthetic: no repo data files, no network.
frame_truth reproduces the exact identity columns frame2.build_frame_truth
emits (unit_id/unit_type/network/date/lat/lon/y/w/cal_var) plus a handful of
feature columns honoring the availability convention (avail == 0 <=> value
NaN, count 0 — NaN is the only missingness representation, DESIGN §6).

heavy_dep_blocker is the import airlock for tests/test_imports.py: a
meta-path finder that raises ImportError for every heavy optional dependency
(torch, gpboost, statsmodels, lightgbm, ...), with sys.modules purged and
restored around it so each import test exercises a genuinely fresh import.
"""
import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
AQNET2_DIR = os.path.dirname(_TESTS_DIR)
_RESEARCH_DIR = os.path.dirname(AQNET2_DIR)
ROOT = os.path.dirname(_RESEARCH_DIR)

for _p in (AQNET2_DIR,
           os.path.join(_RESEARCH_DIR, "aqnet"),
           os.path.join(ROOT, "pipeline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import pytest

import config2

SIGMA_FRM = float(config2.SIGMA_FRM)

# Heavy optional deps every aqnet2 module must import WITHOUT (BUILD_NOTES
# hard rule: the cluster venv lacks pydap/rasterio, local machines may lack
# all of them; an import-time heavy dependency is a contract bug).
HEAVY_DEPS = ("torch", "gpboost", "statsmodels", "lightgbm", "xgboost",
              "catboost", "shapefile", "rasterio", "netCDF4", "pydap")

# The flat aqnet2 module namespace (INTERFACES.md). Purged around the
# blocker so already-imported modules cannot mask an import-time dependency.
AQNET2_MODULES = ("config2", "folds2", "fetchers2", "priors", "skeleton",
                  "graph_res", "field_res", "compose", "calibrate",
                  "colocate", "frame2", "exceed", "uq", "validate2",
                  "pipeline2")

# Synthetic frame geometry: 8 AQS sites x 10 days + 12 PA sensors x 10 days
# = exactly 200 rows.
N_SITES, N_SENSORS, N_DAYS = 8, 12, 10


def _site_ids():
    return [f"site{i}" for i in range(N_SITES)]


def _sensor_ids():
    return [str(1000 + i) for i in range(N_SENSORS)]


class _HeavyDepBlocker:
    """Meta-path finder that refuses every blocked top-level package.

    Raising ImportError from find_spec (rather than returning None) stops
    the import machinery immediately, exactly as if the package were not
    installed — the try-import guards in the aqnet2 modules must catch it
    and print their degradation message.
    """

    def __init__(self, blocked):
        self.blocked = frozenset(blocked)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.blocked:
            raise ImportError(
                f"import of {fullname!r} blocked by the heavy-dep test "
                "airlock (every aqnet2 module must import without it)")
        return None


@pytest.fixture
def heavy_dep_blocker():
    """Hide every heavy optional dep; purge + restore module state around it.

    Yields the blocked-name tuple. Both the heavy deps and the aqnet2
    modules are removed from sys.modules first (so importlib.import_module
    re-executes module code under the blocker) and the originals are put
    back afterwards so later tests see the unmodified modules.
    """
    saved = {}
    for name in list(sys.modules):
        top = name.split(".")[0]
        if top in HEAVY_DEPS or top in AQNET2_MODULES:
            saved[name] = sys.modules.pop(name)
    finder = _HeavyDepBlocker(HEAVY_DEPS)
    sys.meta_path.insert(0, finder)
    try:
        yield HEAVY_DEPS
    finally:
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            pass
        for name in list(sys.modules):
            top = name.split(".")[0]
            if top in HEAVY_DEPS or top in AQNET2_MODULES:
                del sys.modules[name]
        sys.modules.update(saved)


@pytest.fixture
def frame_truth():
    """200-row two-network synthetic frame (frame2 column contract).

    Identity columns exactly as build_frame_truth emits them, sorted by
    (unit_id, date) with mergesort like the real builder. Feature columns
    include the nbr_pacal_50km block with its availability triple: value is
    NaN exactly where avail == 0 and count == 0 — never a fill.
    """
    rng = np.random.default_rng(config2.SEED)
    dates = pd.date_range("2024-06-01", periods=N_DAYS, freq="D")
    s2 = SIGMA_FRM ** 2
    recs = []
    for i, sid in enumerate(_site_ids()):
        lat, lon = 29.5 + 0.2 * i, -98.5 + 0.2 * i
        for d in dates:
            recs.append((f"aqs_{sid}", "aqs", "FRM", d, lat, lon,
                         float(np.clip(rng.normal(12.0, 3.0), 0.0, None)),
                         1.0 / s2, 0.0))
    for i, sn in enumerate(_sensor_ids()):
        lat, lon = 29.55 + 0.15 * i, -98.45 + 0.15 * i
        for d in dates:
            cv = float(rng.uniform(0.5, 8.0))
            recs.append((f"pa_{sn}", "pa", "PA", d, lat, lon,
                         float(np.clip(rng.normal(11.0, 3.5), 0.0, None)),
                         s2 / (s2 + cv), cv))
    frame = pd.DataFrame(recs, columns=["unit_id", "unit_type", "network",
                                        "date", "lat", "lon", "y", "w",
                                        "cal_var"])
    frame = (frame.sort_values(["unit_id", "date"], kind="mergesort")
                  .reset_index(drop=True))
    n = len(frame)

    # PA neighbor block at 25/50 km (+ availability triples + 50 km std).
    for r, p_avail in ((25, 0.6), (50, 0.8)):
        avail = rng.random(n) < p_avail
        cnt = np.where(avail, rng.integers(1, 6, n), 0).astype(np.float64)
        frame[f"nbr_pacal_{r}km"] = np.where(avail,
                                             rng.normal(10.0, 2.0, n), np.nan)
        frame[f"nbr_pacal_count_{r}km"] = cnt
        frame[f"nbr_pacal_avail_{r}km"] = (cnt > 0).astype(np.float64)
        if r == 50:
            frame["nbr_pacal_std_50km"] = np.where(
                cnt == 0, np.nan,
                np.where(cnt == 1, 0.0, np.abs(rng.normal(1.0, 0.3, n))))
    # Lagged FRM block (lag-1 only here; lag-0 FRM must never exist).
    avail_f = rng.random(n) < 0.9
    cnt_f = np.where(avail_f, rng.integers(1, 4, n), 0).astype(np.float64)
    frame["nbr_frm_50km_lag1"] = np.where(avail_f,
                                          rng.normal(11.0, 2.0, n), np.nan)
    frame["nbr_frm_count_50km_lag1"] = cnt_f
    frame["nbr_frm_avail_50km_lag1"] = (cnt_f > 0).astype(np.float64)
    # Portable exact-name features.
    frame["hms_smoke"] = 0.0
    frame["dist_to_coast"] = rng.uniform(50.0, 400.0, n)
    doy = frame["date"].dt.dayofyear.to_numpy(dtype=np.float64)
    dow = frame["date"].dt.dayofweek.to_numpy(dtype=np.float64)
    frame["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    frame["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
    frame["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    frame["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return frame


@pytest.fixture
def pools():
    """Synthetic pools dict for build_point_features (no repo files).

    12 PA sensors spaced ~2 km apart around (30.0, -97.9) and 8 FRM sites
    nearby, 10 days each, deterministic values — queries near the cluster
    see neighbors at every radius; queries hundreds of km away honestly see
    none. Shapes follow frame2.build_pools: [unit_id, lat, lon, date, value].
    """
    dates = pd.date_range("2024-06-01", periods=N_DAYS, freq="D")
    pa_recs, frm_recs = [], []
    for i, sn in enumerate(_sensor_ids()):
        lat, lon = 30.0 + 0.02 * i, -97.90
        for j, d in enumerate(dates):
            pa_recs.append((f"pa_{sn}", lat, lon, d, 8.0 + 0.5 * i + 0.1 * j))
    for i, sid in enumerate(_site_ids()):
        lat, lon = 30.0 + 0.03 * i, -97.85
        for j, d in enumerate(dates):
            frm_recs.append((f"aqs_{sid}", lat, lon, d,
                             9.0 + 0.4 * i + 0.1 * j))
    cols = ["unit_id", "lat", "lon", "date", "value"]
    return {"pa": pd.DataFrame(pa_recs, columns=cols),
            "frm": pd.DataFrame(frm_recs, columns=cols),
            "gridded": {},
            "statics": None,
            "t0": None}


@pytest.fixture
def folds(frame_truth):
    """folds2.json-shaped dict with the INTERFACES row-level keys.

    Row arrays align to frame_truth row order. site7 is the vault; PA rows
    and vault rows carry -1 in outer_fold (v1 always-train semantics).
    Deterministic pure function of the frame — no rng needed.
    """
    frame = frame_truth
    n = len(frame)
    sites = _site_ids()
    vault = [sites[-1]]
    outer_of_site = {s: (i % config2.OUTER_N_FOLDS)
                     for i, s in enumerate(sites[:-1])}
    outer_of_site[sites[-1]] = -1

    unit = frame["unit_id"].astype(str).to_numpy()
    site_of_row = np.array([u[4:] if u.startswith("aqs_") else ""
                            for u in unit])
    outer = np.full(n, -1, dtype=int)
    for s, k in outer_of_site.items():
        outer[site_of_row == s] = k

    uniq = sorted(set(unit))
    uidx = {u: i for i, u in enumerate(uniq)}
    row_uidx = np.array([uidx[u] for u in unit])

    inner, role, loso = {}, {}, {}
    for k in range(config2.OUTER_N_FOLDS):
        ik = np.where(outer == k, -1, row_uidx % config2.INNER_N_FOLDS)
        inner[str(k)] = ik.tolist()
        role[str(k)] = np.where(ik < 0, 2, np.where(ik < 2, 0, 1)).tolist()
        loso[str(k)] = (row_uidx % config2.LOSO_N_FOLDS).tolist()

    conf_units = {"pa_1000", "pa_1001", "pa_1002"}
    conformal = frame["unit_id"].isin(conf_units).astype(int).tolist()
    return {"n_rows": n,
            "content_hash": "synthetic-fixture",
            "seed": config2.SEED,
            "vault_sites": vault,
            "outer_fold_of_site": outer_of_site,
            "outer_fold": outer.tolist(),
            "inner_fold": inner,
            "inner_role": role,
            "loso_fold": loso,
            "spatial_block_fold": (row_uidx % 6).tolist(),
            "temporal_is_test": [0] * n,
            "conformal_unit": conformal}
