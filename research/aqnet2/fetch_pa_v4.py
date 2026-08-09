"""AQNet v4 PurpleAir archival fetcher (pull once, keep forever).

Tiers (built from ~/scratch/aqnet/pa_selection.parquet + the AQS site
index at run time):
  A: the 5 longest-lived sensors within 10 km of each FRM site,
     6-hour averages (average=360) — the calibration backbone with
     local-day alignment and completeness QC.
  B: every other alive outdoor sensor, daily averages (average=1440).
Fields (both tiers): pm2.5_cf_1_a, pm2.5_cf_1_b, pm2.5_atm_a,
pm2.5_atm_b, humidity, temperature.

Storage: DATA_DIR/pa_v4/<tier>/<sensor_index>.parquet, one file per
sensor covering its whole in-window life, written atomically. A manifest
(pa_v4_manifest.json) records completion per sensor: finished sensors are
never re-fetched. Point usage is read from every API response and
projected; the run aborts if the projection exceeds POINT_BUDGET.

Run:  python fetch_pa_v4.py [--tier A|B|all] [--shard i/n]
Key:  ~/.purpleair_key (never committed).
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config2

KEY_PATH = os.path.expanduser("~/.purpleair_key")
SEL_PATH = os.path.expanduser("~/scratch/aqnet/pa_selection.parquet")
OUT_DIR = os.path.join(config2.DATA_DIR, "pa_v4")
MANIFEST = os.path.join(OUT_DIR, "pa_v4_manifest_%s.json")
FIELDS = ("pm2.5_cf_1_a,pm2.5_cf_1_b,pm2.5_atm_a,pm2.5_atm_b,"
          "humidity,temperature")
W0 = pd.Timestamp("2021-01-01")
W1 = pd.Timestamp("2026-08-08")
POINT_BUDGET = 55_000_000
PAIR_KM = 10.0
PER_SITE = 5
CALL_SLEEP = 1.1          # stay far under rate limits
MAX_WINDOW_DAYS = {360: 90, 1440: 365}   # rows/call stays modest


def _say(msg):
    print("[pa_v4] %s" % msg, flush=True)


def api(path, params, key):
    q = "&".join("%s=%s" % kv for kv in params.items())
    req = urllib.request.Request(
        "https://api.purpleair.com/v1/%s?%s" % (path, q),
        headers={"X-API-Key": key})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            if e.code >= 500:
                time.sleep(10 * (attempt + 1))
                continue
            if e.code == 402:
                raise SystemExit(
                    "[pa_v4] API KEY OUT OF POINTS (402): halting so no "
                    "sensor gets falsely marked failed. Add points, then "
                    "resume; archive and manifests stay valid.")
            if e.code in (401, 403):
                raise SystemExit(
                    "[pa_v4] AUTH FAILURE (%d): key missing/invalid; "
                    "halting." % e.code)
            if 400 <= e.code < 500:
                return None      # sensor gone/private: caller records + skips
            raise
    raise RuntimeError("rate-limited after retries: %s" % path)


def load_manifest(shard_tag):
    path = MANIFEST % shard_tag
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"done": {}, "points_used_est": 0, "_path": path}


def save_manifest(m):
    path = m.get("_path") or (MANIFEST % "x")
    payload = {k: v for k, v in m.items() if k != "_path"}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)
    m["_path"] = path


def build_tiers():
    sel = pd.read_parquet(SEL_PATH)
    aq = pd.read_parquet(config2.canonical_aqs_path())
    sites = aq.groupby("site_id").agg(lat=("lat", "median"),
                                      lon=("lon", "median")).reset_index()
    plat = np.radians(sel.latitude.to_numpy(dtype=float))
    plon = np.radians(sel.longitude.to_numpy(dtype=float))
    R = 6371.0088
    tier_a = set()
    for _, s in sites.iterrows():
        dl = plat - np.radians(s.lat)
        dn = plon - np.radians(s.lon)
        h = (np.sin(dl / 2) ** 2
             + np.cos(np.radians(s.lat)) * np.cos(plat)
             * np.sin(dn / 2) ** 2)
        km = 2 * R * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
        near = sel[km <= PAIR_KM]
        if len(near):
            tier_a.update(near.sort_values("days", ascending=False)
                          .head(PER_SITE).sensor_index.tolist())
    a = sel[sel.sensor_index.isin(tier_a)].copy()
    b = sel[~sel.sensor_index.isin(tier_a)].copy()
    _say("tier A: %d sensors (6-hour); tier B: %d sensors (daily)"
         % (len(a), len(b)))
    return {"A": (a, 360), "B": (b, 1440)}


def fetch_sensor(row, avg, key, m):
    si = int(row.sensor_index)
    dest = os.path.join(OUT_DIR, "A" if avg == 360 else "B",
                        "%d.parquet" % si)
    if os.path.exists(dest) or str(si) in m["done"]:
        return 0    # the archive file itself is the completion record
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    t0 = max(pd.to_datetime(row.date_created, unit="s"), W0)
    t1 = min(pd.to_datetime(row.last_seen, unit="s"), W1)
    if t1 <= t0:
        m["done"][str(si)] = 0
        return 0
    chunks = []
    step = pd.Timedelta(days=MAX_WINDOW_DAYS[avg])
    cur = t0
    rows = 0
    while cur < t1:
        end = min(cur + step, t1)
        d = api("sensors/%d/history" % si,
                {"start_timestamp": int(cur.timestamp()),
                 "end_timestamp": int(end.timestamp()),
                 "average": avg, "fields": FIELDS}, key)
        if d is None:
            _say("sensor %d: 4xx (gone/private) -- skipped" % si)
            m["done"][str(si)] = -1
            return 0
        data = d.get("data", [])
        if data:
            df = pd.DataFrame(data, columns=d["fields"])
            chunks.append(df)
            rows += len(df)
        cur = end
        time.sleep(CALL_SLEEP)
    out = (pd.concat(chunks) if chunks
           else pd.DataFrame(columns=FIELDS.split(",")))
    out["sensor_index"] = si
    tmp = dest + ".tmp"
    out.to_parquet(tmp, index=False)
    os.replace(tmp, dest)   # zero-row file still marks completion on disk
    m["done"][str(si)] = rows
    m["points_used_est"] += rows
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="all", choices=["A", "B", "all"])
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args(argv)
    key = open(KEY_PATH).read().strip()
    i, n = (int(x) for x in args.shard.split("/"))
    tiers = build_tiers()
    m = load_manifest(args.shard.replace("/", "of"))
    todo = []
    for t, (df, avg) in tiers.items():
        if args.tier not in ("all", t):
            continue
        for _, row in df.iterrows():
            if int(row.sensor_index) % n == i:
                todo.append((row, avg))
    _say("shard %s: %d sensors to fetch (%d already done)"
         % (args.shard, sum(1 for r, _ in todo
                            if str(int(r.sensor_index)) not in m["done"]),
            len(m["done"])))
    done_ct = 0
    for row, avg in todo:
        rows = fetch_sensor(row, avg, key, m)
        done_ct += 1
        if done_ct % 20 == 0:
            save_manifest(m)
            _say("progress: %d/%d this shard, est points used %s"
                 % (done_ct, len(todo),
                    format(m["points_used_est"], ",")))
            if m["points_used_est"] > POINT_BUDGET:
                save_manifest(m)
                raise SystemExit("[pa_v4] POINT BUDGET EXCEEDED: halting "
                                 "(archive + manifest intact)")
    save_manifest(m)
    _say("shard complete; est points used %s"
         % format(m["points_used_est"], ","))
    return 0


if __name__ == "__main__":
    sys.exit(main())
