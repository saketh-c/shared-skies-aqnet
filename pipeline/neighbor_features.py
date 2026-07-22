"""Same-day MULTI-RADIUS neighbor PM2.5 features — single source of truth.

This is the ONE implementation of the dominant model feature group
(nbr_pm25_25/50/100km + counts + nbr_std_50km) shared by:

  1. pipeline/03_train_enhanced.py  load_data()  — pool = the full dataset
     (every sensor), to build the features the DEPLOYED model trains on.
  2. pipeline/03_train_enhanced.py  loso_cv()    — pool = every sensor EXCEPT
     the held-out one, so each Leave-One-Site-Out fold recomputes its training
     rows' neighbor features WITHOUT the held-out sensor. That removes the
     same-day target leakage where a training sensor near the held-out site
     carried a neighbor mean that had averaged in the held-out site's own
     same-day PM2.5 — which previously inflated the reported LOSO-CV R².

It is also mirrored at inference by backend/purpleair.compute_neighbor_features
(tract centroids as query points, live PurpleAir sensors as the pool).

`pipeline/test_loso_neighbors.py` proves that, with pool == query, this function
reproduces the original inline computation exactly, and that the leave-one-out
fold path equals a brute-force recompute on the reduced sensor set.

Assumption: at most one reading per (sensor_id, date) — true for the daily-mean
training data. Self-exclusion is therefore by sensor_id (a query row never counts
its own sensor as a neighbor), which generalizes cleanly to the LOSO pool where
the query rows are a subset of the pool.
"""
import numpy as np
import pandas as pd

EARTH_R_KM = 6371.0
RADII_KM = (25.0, 50.0, 100.0)


def compute_neighbor_features_df(query_df, pool_df, target_col="pm25",
                                 radii=RADII_KM):
    """Compute same-day multi-radius neighbor PM2.5 features for ``query_df``
    rows, using ``pool_df`` as the reference set of sensor readings.

    Both frames must have columns: latitude, longitude, date, sensor_id, and
    ``target_col``. For each query row: the mean + count of ``target_col`` from
    OTHER sensors (different sensor_id) within 25 / 50 / 100 km on the SAME date,
    plus the population std at the 50 km anchor. A single BallTree per date is
    queried once at the 100 km max radius (with distances), then masked per
    radius — exact and one query per point.

    Zero-neighbor fallback (mirrors training): the 50 km anchor falls back to the
    same-day LEAVE-ONE-OUT mean over the pool (sum minus the query row's own
    reading, divided by count-1), then to the pool grand mean; the 25 km / 100 km
    radii fall back to the (filled) 50 km value. Counts and std are left at 0 for
    zero-neighbor rows.

    Returns a dict {column_name: np.ndarray} aligned to ``query_df`` row order:
    nbr_pm25_25km, nbr_count_25km, nbr_pm25_50km, nbr_count_50km, nbr_std_50km,
    nbr_pm25_100km, nbr_count_100km.
    """
    from sklearn.neighbors import BallTree

    radii = list(radii)
    max_rad_rad = max(radii) / EARTH_R_KM

    q = query_df.reset_index(drop=True)
    p = pool_df.reset_index(drop=True)
    nq = len(q)

    nbr_mean = {r: np.full(nq, np.nan) for r in radii}
    nbr_cnt = {r: np.zeros(nq, dtype=np.int32) for r in radii}
    nbr_std50 = np.zeros(nq, dtype=np.float64)

    q_coords = np.radians(q[["latitude", "longitude"]].values.astype(np.float64))
    p_coords = np.radians(p[["latitude", "longitude"]].values.astype(np.float64))
    q_sid = q["sensor_id"].astype(str).values
    p_sid = p["sensor_id"].astype(str).values
    p_pm = p[target_col].values.astype(np.float64)

    # Positional indices into `p` grouped by date.
    pool_idx_by_date = p.groupby("date").indices

    for date_val, q_pos in q.groupby("date").indices.items():
        p_pos = pool_idx_by_date.get(date_val)
        if p_pos is None or len(p_pos) == 0:
            continue
        tree = BallTree(p_coords[p_pos], metric="haversine")
        ind, dist = tree.query_radius(q_coords[q_pos], r=max_rad_rad,
                                      return_distance=True)
        for ii in range(len(q_pos)):
            qrow = q_pos[ii]
            local = ind[ii]
            if len(local) == 0:
                continue
            gpos = p_pos[local]                 # global positions in `p`
            d_km = dist[ii] * EARTH_R_KM
            keep = p_sid[gpos] != q_sid[qrow]   # drop the query row's own sensor
            gpos = gpos[keep]
            d_km = d_km[keep]
            if len(gpos) == 0:
                continue
            vals_all = p_pm[gpos]
            for r in radii:
                m = d_km <= r
                if not m.any():
                    continue
                vr = vals_all[m]
                nbr_mean[r][qrow] = vr.mean()
                nbr_cnt[r][qrow] = len(vr)
                if r == 50.0 and len(vr) > 1:
                    nbr_std50[qrow] = vr.std()

    # ── Zero-neighbor fallbacks (mirror pipeline/03 load_data) ──
    q_pm = q[target_col].values.astype(np.float64)
    pool_day = pd.DataFrame({"date": p["date"].values, "_pm": p_pm})
    day_sum = pool_day.groupby("date")["_pm"].sum()
    day_cnt = pool_day.groupby("date")["_pm"].count()
    qd = pd.Series(q["date"].values)
    q_day_sum = qd.map(day_sum).to_numpy(dtype=np.float64)
    q_day_cnt = qd.map(day_cnt).to_numpy(dtype=np.float64)
    # Same-day leave-one-out mean over the pool (query rows are in the pool, so
    # subtract their own reading). Denominator clipped at 1 to match training.
    loo_day_mean = (q_day_sum - q_pm) / np.clip(q_day_cnt - 1.0, 1.0, None)
    pool_grand_mean = float(np.nanmean(p_pm)) if len(p_pm) else 0.0

    m50 = pd.Series(nbr_mean[50.0]).fillna(pd.Series(loo_day_mean)).fillna(pool_grand_mean).to_numpy()
    m25 = pd.Series(nbr_mean[25.0]).fillna(pd.Series(m50)).to_numpy()
    m100 = pd.Series(nbr_mean[100.0]).fillna(pd.Series(m50)).to_numpy()

    return {
        "nbr_pm25_25km": m25,
        "nbr_count_25km": nbr_cnt[25.0],
        "nbr_pm25_50km": m50,
        "nbr_count_50km": nbr_cnt[50.0],
        "nbr_std_50km": nbr_std50,
        "nbr_pm25_100km": m100,
        "nbr_count_100km": nbr_cnt[100.0],
    }
