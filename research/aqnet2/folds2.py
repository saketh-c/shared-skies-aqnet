"""AQNet v2 unified fold system (DESIGN S2) -- folds2.json, both phases.

v1's fold story was the root of three audited hazards (audit notes, 2026-08):
folds were positional indices guarded only by an n_rows check (a rebuilt or
re-sorted frame silently invalidated folds.json AND every nbr_overrides npz);
the spatial-block and temporal reruns reused full-pool neighbor features (a
measured leak); and there was no held-out second sample at all -- every AQS
site had been touched by model selection before the final numbers were run.
folds2 closes all three with ONE builder every stage nests inside:

  * VAULT: config2.VAULT_N_SITES AQS sites, stratified by region x mean-PM
    tercile, each >= config2.VAULT_BUFFER_KM from every non-vault site, plus
    all data from config2.VAULT_DATE_START onward. Touched once, by
    validate2, after the configuration freezes.
  * OUTER: config2.OUTER_N_FOLDS spatially-blocked folds over the remaining
    AQS sites (seeded k-means on equirectangular-km coordinates + size
    balancing -- see _equirect_xy for the documented EPSG:3083 deviation).
  * INNER per outer fold: remaining AQS sites AND all PA sensors dealt into
    config2.INNER_N_FOLDS unit-grouped folds; folds 0-1 = selection, 2-3 =
    confirmation (inner_role). The AQS dealing REPRODUCES calibrate.py's
    derived fallback bit-for-bit (rng SEED + 100003*(outer_k+1), permutation
    of the sorted remaining site list, position % 4) and is emitted as the
    site-level `inner_fold_of_site` map, closing the audited contract gap:
    calibrate's pa_cal_f{k}_{j} columns and the row-level selection/
    confirmation masks here now align by construction, not by luck.
  * LOSO per outer fold: config2.LOSO_N_FOLDS unit-grouped folds over the
    outer-k TRAINING units (v1 make_loso_folds mechanics: shuffle the sorted
    unit array, np.array_split).
  * Spatial-block folds, temporal holdout, and the conformal-unit flag.

Two phases, one artifact (folds2.json):

  Phase 1 (this module's CLI; runs BEFORE any frame exists): site-level keys
  only -- vault_sites, outer_fold_of_site, inner_fold_of_site -- with
  n_rows=0 and content_hash="". calibrate.py consumes this via raw json
  (calibrate.load_fold_sites); it cannot go through load_folds because there
  is no frame to hash yet.

  Phase 2 (build_folds(frame), called by the features stage): verifies the
  Phase-1 site-level assignments still reproduce from the current data
  (recompute + compare; a mismatch is a hard error, because calibrate's
  nested columns were fit against the Phase-1 map), then fills every
  row-level key of the INTERFACES.md schema plus the sha256 content hash
  over sorted (unit_id, date, y) triplets. Every consumer verifies the hash
  through load_folds -- row counts alone are insufficient (v1 gotcha 10).

Fold-id semantics (v1 pipeline_colab._folds_from_assign, replicated in
folds_from_assign below): ids >= 0 are test folds; -1 means always-train,
NEVER a test row. NOTE the deliberate asymmetry for the vault: vault rows
carry -1 in every per-row array (they must never be scored by a fold), but
-1 alone would leave them in every training split -- the airlock is enforced
by the separate vault mechanisms (vault_sites + frame2's pool asserts +
vault_row_mask below). Consumers selecting training rows MUST additionally
drop vault_row_mask(frame, folds) rows; frame2's pool builders assert this.

Seeds: every random draw is a fresh numpy default_rng seeded from
config2.SEED plus a documented per-purpose salt, and every shuffle operates
on a SORTED id array (dtype-stable; v1 gotcha 1: reordering the unique-id
array changes fold membership).

Run (Phase 1, idempotent, FORCE=1 to rebuild):
    python folds2.py [--quick]
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

import config2

# ── Input paths (committed v1 data; fetchers2 may later write a v2 AQS) ─────

PA_PARQUET = os.path.join(config2.PIPELINE_DIR, "purpleair_full_dataset.parquet")
AQS_V1_PARQUET = os.path.join(config2.V1_DIR, "data", "aqs_daily_tx.parquet")

# ── Geometry constants ──────────────────────────────────────────────────────

EARTH_RADIUS_KM = 6371.0088          # identical to colocate.py: fold geometry
                                     # and pair geometry must agree
REF_LAT_DEG = 31.0                   # TX mid-latitude for the equirect proj
KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON = 111.320 * float(np.cos(np.radians(REF_LAT_DEG)))

VAULT_DATE_START = getattr(config2, "VAULT_DATE_START", "2026-01-01")

N_SPATIAL_BLOCKS = 5                 # v1 make_spatial_block_folds default
KMEANS_N_INIT = 10
KMEANS_MAX_ITER = 100

# Outer-fold size bounds on the real 50-site problem (DESIGN S2). For tiny
# synthetic/--quick site sets the bounds are scaled to stay feasible
# (lo = min(6, n//k), hi = max(14, ceil(n/k))) -- documented deviation.
OUTER_SIZE_LO = 6
OUTER_SIZE_HI = 14

# ── Seed salts (all draws: default_rng(config2.SEED + salt)) ────────────────
# INNER_SITE_SALT is FROZEN: calibrate.inner_fold_of_site's derived fallback
# uses default_rng(config2.SEED + 100003 * (outer_k + 1)) and the emitted
# site-level map must equal it bit-for-bit. Every other salt just keeps the
# independent draws on visibly distinct seeds.

INNER_SITE_SALT = 100003             # FROZEN -- calibrate.py contract
INNER_PA_SALT = 7919                 # PA sensors: SEED + 7919, k-independent
                                     # (one role per unit across contexts --
                                     # see the dealing site in build_folds)
LOSO_SALT = 400009                   # per outer k: SEED + 400009 * (k + 1)
OUTER_KMEANS_SALT = 131
SPATIAL_KMEANS_SALT = 271
CONFORMAL_SALT = 900007


def _say(msg):
    print(f"[aqnet2] folds2: {msg}", flush=True)


# ── Geometry ────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Inputs broadcast (degrees, float64)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(a, dtype=np.float64))
                              for a in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def _equirect_xy(lat, lon):
    """Equirectangular km coordinates for clustering.

    DESIGN S2 names EPSG:3083 (TX Albers). We approximate with an
    equirectangular projection at REF_LAT_DEG = 31 N -- documented
    deviation: across the TX bbox the relative distance distortion vs Albers
    is small (single-digit percent), it needs no pyproj/GDAL dependency, and
    -- decisive for a fold system -- the arithmetic is bit-reproducible
    everywhere numpy runs. Offsets are irrelevant to k-means distances.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    return np.column_stack([lon * KM_PER_DEG_LON, lat * KM_PER_DEG_LAT])


# ── Seeded k-means (hand-rolled: fold assignments must be version-proof) ────

def _kmeans(xy, k, seed, n_init=KMEANS_N_INIT, max_iter=KMEANS_MAX_ITER):
    """Lloyd's k-means with seeded restarts, pure numpy.

    sklearn.KMeans results can shift across library versions (init and tie
    handling changed more than once); a fold system must be a pure function
    of (seed, data), so the ~50-point clustering is done here. Empty
    clusters are reseeded to the farthest point (deterministic). Returns the
    int assignment of the best-inertia restart.
    """
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    k = int(min(k, n))
    assert k >= 1, "kmeans needs at least one point"
    rng = np.random.default_rng(seed)
    best_inertia, best_assign = np.inf, None
    for _ in range(int(n_init)):
        cents = xy[rng.choice(n, size=k, replace=False)].copy()
        assign = np.full(n, -1, dtype=np.int64)
        for _ in range(int(max_iter)):
            d2 = ((xy[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
            new = d2.argmin(axis=1)
            for c in range(k):
                m = new == c
                if m.any():
                    cents[c] = xy[m].mean(axis=0)
                else:
                    far = int(d2.min(axis=1).argmax())
                    cents[c] = xy[far]
                    new[far] = c
            if (new == assign).all():
                break
            assign = new
        inertia = float(((xy - cents[assign]) ** 2).sum())
        if inertia < best_inertia - 1e-9:
            best_inertia, best_assign = inertia, assign.copy()
    return best_assign


def _balance_sizes(xy, assign, n_folds, lo, hi, ids):
    """Move sites between folds until every fold size is within [lo, hi].

    Moves are deterministic: fix the most undersized fold first by pulling
    the donor site nearest to its centroid (donors keep >= lo members), then
    shrink oversized folds by pushing their member nearest to another
    fold's centroid (receivers stay < hi before the move). Ties break on the
    sorted site id, then the fold id. Each move strictly reduces the total
    bound violation, so the loop terminates whenever (lo, hi) is feasible
    (lo * k <= n <= hi * k); an infeasible request degrades with a WARNING
    rather than looping.
    """
    xy = np.asarray(xy, dtype=np.float64)
    assign = np.asarray(assign, dtype=np.int64).copy()
    ids = [str(s) for s in ids]
    n = len(assign)
    for _ in range(1000):
        sizes = np.bincount(assign, minlength=n_folds)
        cents = np.zeros((n_folds, 2))
        for f in range(n_folds):
            m = assign == f
            cents[f] = xy[m].mean(axis=0) if m.any() else xy.mean(axis=0)
        under = [f for f in range(n_folds) if sizes[f] < lo]
        over = [f for f in range(n_folds) if sizes[f] > hi]
        if not under and not over:
            return assign
        if under:
            f = min(under, key=lambda g: (sizes[g], g))
            best = None
            for i in range(n):
                g = int(assign[i])
                if g == f or sizes[g] <= lo:
                    continue
                key = (float(((xy[i] - cents[f]) ** 2).sum()), ids[i])
                if best is None or key < best[0]:
                    best = (key, i)
            if best is None:
                break
            assign[best[1]] = f
        else:
            f = int(max(over, key=lambda g: (sizes[g], -g)))
            best = None
            for i in np.where(assign == f)[0]:
                for g in range(n_folds):
                    if g == f or sizes[g] >= hi:
                        continue
                    key = (float(((xy[i] - cents[g]) ** 2).sum()), ids[i], g)
                    if best is None or key < best[0]:
                        best = (key, int(i), g)
            if best is None:
                break
            assign[best[1]] = best[2]
    _say(f"WARNING: fold size balance did not converge to [{lo}, {hi}] "
         f"(sizes {np.bincount(assign, minlength=n_folds).tolist()}) -- "
         "bounds infeasible for this site count; keeping best-effort split")
    return assign


# ── Content hash (the load_folds guard) ─────────────────────────────────────

def content_hash(frame):
    """sha256 over the SORTED (unit_id, date, y) triplets of a frame.

    Row-order invariant by construction (v1 gotcha 10: folds are positional,
    and a silently re-sorted frame is the measured failure mode this guards
    against -- n_rows checks cannot catch it). y is rounded to 6 decimal
    places so the hash survives lossy float round-trips through non-parquet
    intermediates; -0.0 is normalized to 0.0 and NaN encodes as the literal
    "nan" (NaN is data here, never a fill).
    """
    for c in ("unit_id", "date", "y"):
        if c not in frame.columns:
            raise ValueError(f"content_hash: frame lacks required column {c!r}")
    uid = frame["unit_id"].astype(str).to_numpy()
    dts = (pd.to_datetime(frame["date"]).dt.normalize()
           .dt.strftime("%Y-%m-%d").to_numpy())
    y = np.asarray(frame["y"], dtype=np.float64)
    y = np.round(y, 6) + 0.0          # + 0.0 folds -0.0 into 0.0
    parts = [f"{u}|{d}|" + ("nan" if np.isnan(v) else format(v, ".6f"))
             for u, d, v in zip(uid, dts, y)]
    parts.sort()
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


# ── v1 fold semantics (pipeline_colab._folds_from_assign, promoted) ─────────

def folds_from_assign(assign):
    """Rebuild [(train_idx, test_idx), ...] from a per-row test-fold id.

    EXACT v1 semantics (research/aqnet/pipeline_colab.py:159): iterate the
    sorted unique ids >= 0 as test folds; every row not in the test fold --
    including every -1 row -- lands on the train side. -1 therefore means
    always-train, never-scored. Positional indices into frame row order.
    """
    assign = np.asarray(assign, dtype=np.int64)
    folds = []
    for k in sorted(int(v) for v in np.unique(assign[assign >= 0])):
        test = np.where(assign == k)[0]
        train = np.where(assign != k)[0]
        folds.append((train, test))
    return folds


# ── Index loaders (Phase 1 inputs) ──────────────────────────────────────────

def load_aqs_site_index(aqs_parquet=None):
    """Per-site index [site_id(str), lat, lon, pm_mean, n_days].

    Source preference: an explicit path argument, then a fetchers2-hardened
    v2 copy at config2.artifact("aqs_daily_tx.parquet") when present, then
    the committed v1 parquet. Coordinates are per-site medians; dates are
    normalized to datetime64[ns] (the committed AQS parquet is
    datetime64[us] -- audited pandas-3 hazard).
    """
    path = aqs_parquet
    if path is None:
        # fetchers2 canonical (year-window-stamped under DATA_DIR), then a
        # hand-placed artifact copy, then the committed v1 parquet. Site
        # set/coords/pm25 are identical across v1 and v2 copies (v2 adds
        # metadata columns), so fold assignments are stable either way.
        import glob as _glob
        v2s = sorted(_glob.glob(os.path.join(
            config2.DATA_DIR, "aqs_daily_tx_v2_*.parquet")))
        art = config2.artifact("aqs_daily_tx.parquet")
        path = (v2s[-1] if v2s
                else art if os.path.exists(art) else AQS_V1_PARQUET)
    if not os.path.exists(path):
        raise FileNotFoundError(f"AQS daily parquet not found: {path}")
    aq = pd.read_parquet(path, columns=["site_id", "date", "pm25_aqs",
                                        "lat", "lon"])
    aq = aq.copy()
    aq["site_id"] = aq["site_id"].astype(str)
    aq["date"] = (pd.to_datetime(aq["date"]).dt.normalize()
                  .astype("datetime64[ns]"))
    out = (aq.groupby("site_id", sort=True)
             .agg(lat=("lat", "median"), lon=("lon", "median"),
                  pm_mean=("pm25_aqs", "mean"), n_days=("date", "nunique"))
             .reset_index())
    return out


def load_pa_sensor_index(pa_parquet=None):
    """Per-sensor index [sensor_id(str), lat, lon] from the committed PA
    parquet.

    Sensor coordinates are the latitude/longitude columns -- NOT lat/lon,
    which in that parquet are TRACT CENTROIDS (audited schema trap). Phase 1
    uses this only as an inventory check; PA sensors receive their fold
    assignments at Phase 2 from the frame's own unit list.
    """
    path = pa_parquet or PA_PARQUET
    if not os.path.exists(path):
        raise FileNotFoundError(f"PA daily parquet not found: {path}")
    pa = pd.read_parquet(path, columns=["sensor_id", "latitude", "longitude"])
    pa = pa.copy()
    pa["sensor_id"] = pa["sensor_id"].astype(str)
    out = (pa.groupby("sensor_id", sort=True)[["latitude", "longitude"]]
             .median().reset_index()
             .rename(columns={"latitude": "lat", "longitude": "lon"}))
    return out


# ── Site-level assignments (the Phase-1 pure function) ──────────────────────

def _stratum_ids(lat, lon, pm):
    """Region x mean-PM stratum id per site.

    Region = 4 quadrants of the site cloud via the median-lat/median-lon
    split; PM = terciles of the site mean. DESIGN S2 asks for region x
    urbanicity x mean PM; urbanicity metadata (AQS location-setting) is not
    in the committed parquet until the fetchers2 hardened refetch lands, so
    the stratification is region x tercile -- documented deviation, recorded
    here rather than silently approximated. NaN site means (a site with no
    finite pm25) fall deterministically into the top tercile.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    pm = np.asarray(pm, dtype=np.float64)
    quad = ((lat >= float(np.median(lat))).astype(np.int64) * 2
            + (lon >= float(np.median(lon))).astype(np.int64))
    finite = np.isfinite(pm)
    if int(finite.sum()) >= 3:
        qs = np.quantile(pm[finite], [1.0 / 3.0, 2.0 / 3.0])
        terc = np.searchsorted(qs, pm, side="right")   # NaN -> 2
        terc = np.clip(terc, 0, 2).astype(np.int64)
    else:
        terc = np.zeros(len(pm), dtype=np.int64)
    return quad * 3 + terc


def _select_vault(dist, strat, cand_order, n_vault, buffer_km, ids):
    """Greedy deterministic vault selection (positions into ids).

    Iterates candidates in the seeded-shuffled sorted-id order; a candidate
    is accepted iff, after hypothetically moving it into the vault, it is
    >= buffer_km from every remaining NON-vault site. Accepting a site only
    shrinks earlier members' constraint sets, so the final configuration
    satisfies the invariant exactly as checked. Stratified sweeps run first
    (round-robin targets over the non-empty strata); if strata cannot fill
    the vault, the stratum constraint is relaxed -- the BUFFER never is
    (DESIGN S2). Sweeps repeat until no progress: a candidate that failed
    the buffer early can become eligible after its close neighbor vaults.
    """
    n = len(ids)
    n_vault = int(min(n_vault, n))
    strata = sorted(set(int(s) for s in strat))
    counts = {s: int((strat == s).sum()) for s in strata}
    targets = {s: 0 for s in strata}
    left = n_vault
    while left > 0:
        moved = False
        for s in strata:
            if left == 0:
                break
            if targets[s] < counts[s]:
                targets[s] += 1
                left -= 1
                moved = True
        if not moved:
            break

    in_vault = np.zeros(n, dtype=bool)
    taken = {s: 0 for s in strata}
    vault = []

    def _buffer_ok(c):
        nv = ~in_vault                # fresh copy: the non-vault set ...
        nv[c] = False                 # ... after hypothetically vaulting c
        if not nv.any():
            return True
        return float(dist[c, nv].min()) >= buffer_km

    for respect_strata in (True, False):
        progressed = True
        while progressed and len(vault) < n_vault:
            progressed = False
            for c in cand_order:
                if len(vault) >= n_vault:
                    break
                c = int(c)
                if in_vault[c]:
                    continue
                s = int(strat[c])
                if respect_strata and taken[s] >= targets[s]:
                    continue
                if _buffer_ok(c):
                    in_vault[c] = True
                    taken[s] += 1
                    vault.append(c)
                    progressed = True
    if len(vault) < n_vault:
        _say(f"WARNING: vault filled {len(vault)}/{n_vault} sites -- the "
             f"{buffer_km:g} km buffer is never relaxed (DESIGN S2); "
             "proceeding with the smaller vault")
    return vault


def site_assignments(site_df, seed=config2.SEED, n_vault=None, buffer_km=None,
                     n_outer=None, n_inner=None):
    """Pure function (seed, site ids, coords, mean pm) -> site-level folds.

    Returns {"vault_sites", "outer_fold_of_site", "inner_fold_of_site"}.
    outer_fold_of_site is TOTAL over the input sites: vault sites carry -1
    (calibrate.outer_fold_ids filters >= 0; its fallback derivation filters
    the vault set -- both behaviors are exercised against this exact shape).

    The inner AQS dealing per outer fold k is byte-identical to
    calibrate.inner_fold_of_site's derived fallback when seed ==
    config2.SEED (the only production case): rng = default_rng(seed +
    100003 * (k + 1)), permutation of the sorted remaining non-vault
    non-fold-k site list, position % n_inner. calibrate prefers the
    explicit map emitted here, so alignment holds even off-default.
    """
    n_vault = config2.VAULT_N_SITES if n_vault is None else int(n_vault)
    buffer_km = (config2.VAULT_BUFFER_KM if buffer_km is None
                 else float(buffer_km))
    n_outer = config2.OUTER_N_FOLDS if n_outer is None else int(n_outer)
    n_inner = config2.INNER_N_FOLDS if n_inner is None else int(n_inner)

    for c in ("site_id", "lat", "lon", "pm_mean"):
        if c not in site_df.columns:
            raise ValueError(f"site_assignments: site_df lacks column {c!r}")
    df = site_df.copy()
    df["site_id"] = df["site_id"].astype(str)
    df = df.sort_values("site_id", kind="mergesort").reset_index(drop=True)
    ids = df["site_id"].to_numpy()
    if len(set(ids)) != len(ids):
        raise ValueError("site_assignments: duplicate site_id in site index")
    lat = df["lat"].to_numpy(dtype=np.float64)
    lon = df["lon"].to_numpy(dtype=np.float64)
    pm = df["pm_mean"].to_numpy(dtype=np.float64)
    n = len(ids)
    if n == 0:
        raise ValueError("site_assignments: empty site index")

    dist = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    strat = _stratum_ids(lat, lon, pm)

    # Vault: candidate order = seeded shuffle of the SORTED site-id array.
    rng = np.random.default_rng(seed)
    cand_order = rng.permutation(n)
    vault_pos = _select_vault(dist, strat, cand_order, n_vault, buffer_km, ids)
    vault = sorted(str(ids[p]) for p in vault_pos)

    # Outer: seeded k-means over the remaining sites + size balance.
    in_vault = np.zeros(n, dtype=bool)
    in_vault[list(vault_pos)] = True
    rem_pos = np.where(~in_vault)[0]
    n_rem = len(rem_pos)
    if n_rem == 0:
        raise ValueError("site_assignments: vault consumed every site -- "
                         "no sites left for outer folds")
    k = int(min(n_outer, n_rem))
    xy = _equirect_xy(lat[rem_pos], lon[rem_pos])
    assign = _kmeans(xy, k, seed + OUTER_KMEANS_SALT)
    lo = min(OUTER_SIZE_LO, n_rem // k)
    hi = max(OUTER_SIZE_HI, -(-n_rem // k))
    assign = _balance_sizes(xy, assign, k, lo, hi, ids[rem_pos])

    outer = {str(ids[p]): -1 for p in vault_pos}
    for p, pos in enumerate(rem_pos):
        outer[str(ids[pos])] = int(assign[p])

    # Inner: the frozen calibrate.py derivation, emitted explicitly.
    vset = set(vault)
    inner = {}
    for kk in sorted(set(int(a) for a in assign)):
        remaining = sorted(s for s, f in outer.items()
                           if f != kk and s not in vset)
        r = np.random.default_rng(seed + INNER_SITE_SALT * (kk + 1))
        order = r.permutation(len(remaining))
        inner[str(kk)] = {remaining[ix]: int(pos % n_inner)
                          for pos, ix in enumerate(order)}

    return {"vault_sites": vault,
            "outer_fold_of_site": {s: int(f)
                                   for s, f in sorted(outer.items())},
            "inner_fold_of_site": inner}


# ── Phase 1 ─────────────────────────────────────────────────────────────────

def build_site_folds(aqs_parquet=None, pa_parquet=None, seed=config2.SEED,
                     n_outer=None, path=None):
    """Phase-1 folds2.json dict: site-level assignments, no frame required.

    calibrate.py consumes the result via raw json (load_fold_sites) before
    any frame exists, so this must be a reproducible pure function of the
    committed site index and the seed -- Phase 2 recomputes it and refuses
    to proceed on any drift. Keys: n_rows=0, content_hash="", seed,
    vault_sites, outer_fold_of_site, inner_fold_of_site. When `path` is
    given the dict is also saved there (atomic).
    """
    site_df = load_aqs_site_index(aqs_parquet)
    _say(f"AQS site index: {len(site_df)} sites")
    try:
        pa_df = load_pa_sensor_index(pa_parquet)
        _say(f"PA sensor index: {len(pa_df)} sensors (fold-assigned at "
             "Phase 2 from the frame's unit list)")
    except Exception as exc:                      # inventory only, never fatal
        _say(f"NOTE: PA sensor index unavailable ({type(exc).__name__}: "
             f"{exc}) -- Phase-1 outputs do not depend on it")

    sf = site_assignments(site_df, seed=seed, n_outer=n_outer)
    folds = {"n_rows": 0, "content_hash": "", "seed": int(seed)}
    folds.update(sf)

    sizes = {}
    for s, f in sf["outer_fold_of_site"].items():
        if f >= 0:
            sizes[f] = sizes.get(f, 0) + 1
    _say(f"vault: {len(sf['vault_sites'])} sites {sf['vault_sites']}")
    _say(f"outer fold sizes: {dict(sorted(sizes.items()))}")
    if path:
        save_folds(folds, path)
        _say(f"wrote Phase-1 folds -> {path}")
    return folds


# ── Phase 2 helpers ─────────────────────────────────────────────────────────

def vault_row_mask(frame, folds):
    """Boolean per-row mask of everything the vault airlock forbids.

    True for rows of vault AQS sites AND all rows in the vault period
    (date >= VAULT_DATE_START), any unit. These rows carry -1 in every fold
    array (never scored), but -1 alone means always-TRAIN under
    folds_from_assign -- so every training-row selection must ALSO drop this
    mask. frame2's pool builders assert the same invariant independently.
    """
    vault = {str(s) for s in (folds or {}).get("vault_sites", [])}
    uid = frame["unit_id"].astype(str)
    site = uid.str.replace("^aqs_", "", regex=True)
    is_vault_unit = uid.str.startswith("aqs_") & site.isin(sorted(vault))
    dts = (pd.to_datetime(frame["date"]).dt.normalize()
           .astype("datetime64[ns]"))
    return (is_vault_unit.to_numpy()
            | (dts >= pd.Timestamp(VAULT_DATE_START)).to_numpy())


def _deal_pos_mod(ids, seed, n_folds):
    """{id: fold} via the calibrate.py rng family.

    Sorted unique ids, default_rng(seed).permutation over positions, fold =
    position % n_folds. This is the exact construction of
    calibrate.inner_fold_of_site's fallback; the PA dealing reuses the
    family (with INNER_PA_SALT added to the seed) so both unit types share
    one audited mechanism.
    """
    ids = sorted(str(i) for i in ids)
    if not ids:
        return {}
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    return {ids[ix]: int(pos % n_folds) for pos, ix in enumerate(order)}


def _deal_array_split(ids, seed, n_folds):
    """{id: fold} via the v1 make_loso_folds mechanics.

    Shuffle the SORTED unique-id array with default_rng(seed), then
    np.array_split into min(n_folds, n_ids) contiguous chunks (v1
    validation.make_loso_folds, generalized from sensor_id to unit_id).
    """
    arr = np.array(sorted(str(i) for i in ids), dtype=object)
    if len(arr) == 0:
        return {}
    rng = np.random.default_rng(seed)
    rng.shuffle(arr)
    n_folds = int(min(n_folds, len(arr)))
    out = {}
    for f, part in enumerate(np.array_split(arr, n_folds)):
        for u in part:
            out[str(u)] = int(f)
    return out


def _norm_site_level(sf):
    """Site-level keys in a json-roundtrip-stable comparable form."""
    return {
        "vault_sites": sorted(str(s) for s in sf.get("vault_sites", [])),
        "outer_fold_of_site": {str(s): int(v) for s, v in
                               sf.get("outer_fold_of_site", {}).items()},
        "inner_fold_of_site": {str(k): {str(s): int(j) for s, j in m.items()}
                               for k, m in
                               sf.get("inner_fold_of_site", {}).items()},
    }


# ── Phase 2 ─────────────────────────────────────────────────────────────────

def build_folds(frame, seed=config2.SEED, site_folds=None, path=None,
                n_loso=None):
    """Phase 2: fill every row-level key of the INTERFACES folds2.json schema.

    Verification first: unless `site_folds` is supplied directly (tests,
    callers already holding Phase 1), the Phase-1 json at `path` (default
    config2.artifact("folds2.json")) is loaded and the site-level
    assignments are RECOMPUTED from the current data and asserted equal --
    calibrate's nested pa_cal_f{k}_{j} columns were fit against the Phase-1
    map, so any drift (data or seed changed underneath) is a hard error,
    never a silent re-deal.

    Row-level keys (all positional over frame rows, -1 = never-scored):
      outer_fold          fold of the row's AQS site; -1 for PA rows, vault
                          rows and vault-period rows (see vault_row_mask).
      inner_fold[str(k)]  0..n_inner-1; AQS sites from the frozen Phase-1
                          map (per-context), PA sensors dealt ONCE with the
                          same rng family (SEED + INNER_PA_SALT,
                          k-independent so pooled sel/conf roles stay
                          unit-consistent across contexts);
                          -1 for fold-k outer rows and vault material.
      inner_role[str(k)]  0 = selection (inner < n_inner/2), 1 =
                          confirmation, 2 = excluded (inner == -1).
      loso_fold[str(k)]   unit-grouped folds over outer-k TRAINING units
                          (all PA sensors + non-vault non-fold-k AQS sites);
                          -1 for fold-k test rows and vault material.
      spatial_block_fold  N_SPATIAL_BLOCKS-way seeded k-means over ALL
                          units' median coordinates (vault units included:
                          the array is total; consumers mask the vault via
                          vault_row_mask -- documented).
      temporal_is_test    date >= config2.TEMPORAL_CUTOFF (the 7-day lag
                          embargo is enforced feature-side by frame2's
                          query-date shifting, not encoded here).
      conformal_unit      1 for the conformal calibration units: a disjoint
                          seeded CONFORMAL_PA_FRAC sample of PA sensors plus
                          the inner fold-(n_inner-1) AQS sites of the lowest
                          outer fold (the confirmation-side fold DESIGN S2
                          reserves from admission evaluation); vault rows 0.
    """
    n_loso = config2.LOSO_N_FOLDS if n_loso is None else int(n_loso)
    for c in ("unit_id", "unit_type", "date", "lat", "lon", "y"):
        if c not in frame.columns:
            raise ValueError(f"build_folds: frame lacks required column {c!r}")

    if site_folds is None:
        p = path or config2.artifact("folds2.json")
        if os.path.exists(p):
            loaded = load_folds_raw(p)
            if int(loaded.get("seed", -1)) != int(seed):
                raise RuntimeError(
                    f"folds2.json at {p} was built with seed "
                    f"{loaded.get('seed')} but build_folds got seed {seed} "
                    "-- refusing to mix fold systems")
            n_outer = 1 + max(int(v) for v in
                              loaded["outer_fold_of_site"].values())
            recomputed = build_site_folds(seed=seed, n_outer=n_outer)
            if _norm_site_level(loaded) != _norm_site_level(recomputed):
                raise RuntimeError(
                    "Phase-1 site-level assignments do not reproduce from "
                    "the current data -- the AQS site index or seed changed "
                    "since folds2.json was built. calibrate's nested "
                    "columns are aligned to the OLD map; rebuild the chain "
                    "from folds2 onward (FORCE=1) instead of proceeding.")
            site_folds = recomputed
        else:
            _say(f"NOTE: no Phase-1 json at {p} -- computing site-level "
                 "assignments in-process (calibrate must consume the SAME "
                 "map; run `python folds2.py` to persist it)")
            site_folds = build_site_folds(seed=seed)

    vault = {str(s) for s in site_folds["vault_sites"]}
    omap = {str(s): int(f)
            for s, f in site_folds["outer_fold_of_site"].items()}
    imap = {str(k): {str(s): int(j) for s, j in m.items()}
            for k, m in site_folds["inner_fold_of_site"].items()}
    inner_js = [j for m in imap.values() for j in m.values()]
    n_inner = max(int(config2.INNER_N_FOLDS),
                  (max(inner_js) + 1) if inner_js else 0)
    half = n_inner // 2

    n = len(frame)
    uid = frame["unit_id"].astype(str).to_numpy()
    utype = frame["unit_type"].astype(str).to_numpy()
    dts = (pd.to_datetime(frame["date"]).dt.normalize()
           .astype("datetime64[ns]").to_numpy())

    is_aqs = utype == "aqs"
    is_pa = utype == "pa"
    if not bool(np.all(is_aqs | is_pa)):
        bad = sorted(set(utype) - {"aqs", "pa"})
        raise ValueError(f"build_folds: unknown unit_type values {bad}")
    bad_ids = [u for u, t in zip(uid, utype)
               if (t == "aqs" and not u.startswith("aqs_"))
               or (t == "pa" and not u.startswith("pa_"))]
    if bad_ids:
        raise ValueError("build_folds: unit_id/unit_type prefix mismatch, "
                         f"e.g. {bad_ids[:5]}")

    site_row = np.array([u[4:] if u.startswith("aqs_") else ""
                         for u in uid], dtype=object)
    unknown = sorted(set(site_row[is_aqs]) - set(omap))
    if unknown:
        raise ValueError(
            "build_folds: frame contains AQS sites absent from the fold "
            f"system ({unknown[:8]}...) -- the site index changed since "
            "Phase 1; rebuild folds2.json")

    vault_unit_row = is_aqs & np.isin(site_row, sorted(vault))
    vault_period = dts >= np.datetime64(VAULT_DATE_START)
    vmask = vault_unit_row | vault_period

    uid_s = pd.Series(uid)

    # outer_fold ------------------------------------------------------------
    outer = np.full(n, -1, dtype=np.int64)
    if is_aqs.any():
        outer[is_aqs] = np.array([omap[s] for s in site_row[is_aqs]],
                                 dtype=np.int64)
    outer[vmask] = -1
    outer_ids = sorted({int(v) for v in omap.values() if int(v) >= 0})
    if not outer_ids:
        raise ValueError("build_folds: no outer folds in the site map")

    units = sorted(set(uid))
    pa_units = sorted(set(uid[is_pa]))
    aqs_units = sorted(set(uid[is_aqs]))
    vault_units = {"aqs_" + s for s in vault}

    inner_fold, inner_role, loso_fold = {}, {}, {}
    for k in outer_ids:
        sk = str(k)
        if sk not in imap:
            raise KeyError(f"build_folds: inner_fold_of_site lacks outer "
                           f"fold {sk}")
        smap = imap[sk]
        # PA dealing is deliberately k-INDEPENDENT (no INNER_SITE_SALT*(k+1)
        # term): a PA sensor keeps one inner fold — hence one sel/conf role —
        # across every outer context. The gates stage pools roles across
        # contexts and must demote any unit whose role flips (sel in one
        # context, conf in another would break the paired-bootstrap cluster
        # disjointness); per-context dealing would flip ~94% of PA units and
        # leave admission with no clusters. AQS sites stay per-context via
        # the frozen calibrate.py map.
        pa_map = _deal_pos_mod(pa_units, seed + INNER_PA_SALT, n_inner)
        unit_inner = {}
        for u in aqs_units:
            unit_inner[u] = int(smap.get(u[4:], -1))
        unit_inner.update(pa_map)
        ik = uid_s.map(unit_inner).fillna(-1).to_numpy(dtype=np.int64,
                                                       copy=True)
        ik[outer == k] = -1            # belt-and-braces: fold-k never inner
        ik[vmask] = -1
        inner_fold[sk] = ik

        rk = np.full(n, 2, dtype=np.int64)
        rk[(ik >= 0) & (ik < half)] = 0
        rk[ik >= half] = 1
        inner_role[sk] = rk

        fold_k_units = {"aqs_" + s for s, f in omap.items() if f == k}
        train_units = [u for u in units
                       if u not in vault_units and u not in fold_k_units]
        lmap = _deal_array_split(train_units, seed + LOSO_SALT * (k + 1),
                                 n_loso)
        lk = uid_s.map(lmap).fillna(-1).to_numpy(dtype=np.int64, copy=True)
        lk[vmask] = -1
        loso_fold[sk] = lk

    # spatial_block_fold ----------------------------------------------------
    coords = (frame.assign(_uid=uid)
              .groupby("_uid", sort=True)[["lat", "lon"]].median())
    coords = coords.loc[units]
    xy = _equirect_xy(coords["lat"].to_numpy(), coords["lon"].to_numpy())
    sb_assign = _kmeans(xy, min(N_SPATIAL_BLOCKS, len(units)),
                        seed + SPATIAL_KMEANS_SALT)
    sbmap = {u: int(a) for u, a in zip(units, sb_assign)}
    sb = uid_s.map(sbmap).to_numpy(dtype=np.int64)

    # temporal_is_test ------------------------------------------------------
    tt = (dts >= np.datetime64(str(config2.TEMPORAL_CUTOFF))).astype(np.int64)

    # conformal_unit --------------------------------------------------------
    rng_c = np.random.default_rng(seed + CONFORMAL_SALT)
    n_pick = int(round(float(config2.CONFORMAL_PA_FRAC) * len(pa_units)))
    order = rng_c.permutation(len(pa_units))
    conf_units = {pa_units[ix] for ix in order[:n_pick]}
    k0 = str(outer_ids[0])
    conf_units |= {"aqs_" + s for s, j in imap[k0].items()
                   if int(j) == n_inner - 1}
    cu = np.asarray(uid_s.isin(sorted(conf_units)).to_numpy(),
                    dtype=np.int64).copy()
    cu[vmask] = 0

    folds = {
        "n_rows": int(n),
        "content_hash": content_hash(frame),
        "seed": int(seed),
        "vault_sites": sorted(vault),
        "outer_fold_of_site": {s: int(f) for s, f in sorted(omap.items())},
        "inner_fold_of_site": {k: dict(sorted(m.items()))
                               for k, m in sorted(imap.items())},
        "outer_fold": outer.tolist(),
        "inner_fold": {k: v.tolist() for k, v in inner_fold.items()},
        "inner_role": {k: v.tolist() for k, v in inner_role.items()},
        "loso_fold": {k: v.tolist() for k, v in loso_fold.items()},
        "spatial_block_fold": sb.tolist(),
        "temporal_is_test": tt.tolist(),
        "conformal_unit": cu.tolist(),
    }
    _say(f"Phase 2: {n} rows, {len(units)} units "
         f"({len(aqs_units)} aqs / {len(pa_units)} pa), "
         f"outer folds {outer_ids}, vault rows {int(vmask.sum())}, "
         f"conformal units {len(conf_units)}")
    return folds


# ── Persistence ─────────────────────────────────────────────────────────────

def save_folds(folds, path):
    """Atomic json write (tmp + os.replace, v1 checkpoint discipline)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(folds, fh, separators=(",", ":"))
        fh.write("\n")
    os.replace(tmp, path)


def load_folds_raw(path):
    """Phase-1 consumption: raw json, NO content-hash verification.

    Only for consumers that run before any frame exists (calibrate uses its
    own equivalent, load_fold_sites). Everything downstream of a frame MUST
    go through load_folds.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"folds2.json not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_folds(path, frame):
    """Load folds2.json and verify it belongs to THIS frame.

    Raises ValueError when the stored content hash (or row count) does not
    match the frame -- including the Phase-1 case (content_hash == ""),
    which means Phase 2 has not run yet. v1's n_rows-only guard is the
    audited hazard this replaces: a re-sorted frame of identical length
    silently mis-aligned every fold and override array.
    """
    folds = load_folds_raw(path)
    if int(folds.get("n_rows", -1)) != len(frame):
        raise ValueError(
            f"folds2.json n_rows={folds.get('n_rows')} != frame rows "
            f"{len(frame)} -- fold system does not belong to this frame")
    h = content_hash(frame)
    stored = folds.get("content_hash", "")
    if stored != h:
        raise ValueError(
            "folds2.json content_hash mismatch -- the frame changed (or "
            "only Phase 1 has run) since folds were built; rebuild Phase 2 "
            f"(stored {stored[:12]!r}..., frame {h[:12]!r}...)")
    return folds


# ── CLI (Phase 1) ───────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="AQNet v2 unified fold system -- Phase 1 (site-level)")
    ap.add_argument("--quick", action="store_true",
                    help="2 outer folds (pipeline --quick contract); vault "
                         "and buffer are never shrunk")
    ap.add_argument("--aqs-parquet", default=None)
    ap.add_argument("--pa-parquet", default=None)
    ap.add_argument("--seed", type=int, default=config2.SEED)
    args = ap.parse_args(argv)

    dest = config2.artifact("folds2.json")
    if os.path.exists(dest) and os.environ.get("FORCE") != "1":
        print(f"[aqnet2] folds2: {dest} exists (FORCE=1 to rebuild) -- skip")
        return 0

    print("[aqnet2] ── stage: folds2 ──")
    build_site_folds(args.aqs_parquet, args.pa_parquet, seed=args.seed,
                     n_outer=2 if args.quick else None, path=dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
