"""T3 — masked-pretrained multimodal field residual + INR point decoder.

DESIGN.md §8. Residual target r2 = y − F2_oof, where F2 is the composed
incumbent below T3 (T1 skeleton OOF plus the T2 graph residual when its OOF
artifact exists). Two stages:

  fieldpre   masked multimodal autoencoder over the v2 gridded stack — the
             representation learner. 60% random 16x16-patch masking plus
             whole-channel-group drops (p=0.15), reconstruction loss ONLY on
             finite pixels of masked regions (per-channel finite-mask planes
             define validity; no fill value ever enters the loss), the raw-PA
             observation raster INSIDE the masking objective, and a
             discrepancy head predicting (obs − geoscf_pm25) at masked
             station pixels so the representation encodes CTM error structure
             instead of collapsing into a CTM autoencoder.
  fieldres   INR point decoder fine-tuned per (outer k, inner j) fold on the
             FROZEN pretrained encoder: query (lat, lon, t) -> multi-scale
             bilinear samples of 3 encoder depths at the exact point + HR
             statics at the exact point + band-limited Fourier features of
             the sub-cell offset -> MLP (3x256) -> (r2_hat, log sigma^2).
             Supervision at TRUE station coordinates — no rint() snapping —
             so co-located urban sensors get distinct supervision (closes
             v1's rint defect, DESIGN §0.5). OOF at 100% of stack-covered
             rows via the nested fold ensemble.

WHY this shape (v1 evidence):

  * v1's FusionUNet earned composite weight ~0 and its per-group "flags"
    channels were dead inputs (audit §2): softmax-over-sources fusion plus
    mean-filled planes gave the U-Net no way to know WHICH pixel of a group
    was real. Here every real channel carries its own 0/1 finite-mask plane
    ("chanmask" group, v1 flags retired), masking/fill is explicit at the
    input, and the loss reads only pre-fill-finite pixels.
  * v1 snapped supervision to grid pixels with rint() (dataset.py:281),
    merging co-located sensors and blurring sub-cell gradients. The INR head
    samples encoder features bilinearly at the exact fractional pixel and
    adds Fourier features of the sub-cell offset, band-limited to 5–200 km
    wavelengths as a hard cap against between-sensor hallucination.
  * No pa_cal raster and no T2 surfaces enter the shared pretrain stack:
    both are FRM-informed, so including them would leak fold information
    through pretraining (fold honesty, DESIGN §8). Pretraining consumes raw
    PA observations only (correction="raw").

Fill policy (documented per BUILD_NOTES): norm stats are computed over
FINITE pixels pre-fill (v1 models_deep._compute_norm_stats_prefill), then
NaN -> 0 in NORMALIZED space is permitted for INPUT tensors ONLY, because
the per-channel mask planes carry validity — the loss never reads filled
pixels (the reconstruction mask is chanmask AND masked-region, both
computed pre-fill).

Budget sizing (documented per task): full stack ~1947 days on the 0.1-deg
grid (~113x139). Pretrain: crops 96x96, batch 16 -> ~120 steps/epoch; 120
epochs ~ 14.5k UNet(base 48) steps with AMP — a few GPU-hours on
RTX6000/L40S, well inside the 8 h budget. Fine-tunes: the encoder is FROZEN
(pretraining used no FRM-derived labels, so one shared encoder is
fold-honest; the fold purity of supervision lives entirely in the head), so
per-day encoder features are computed ONCE and the 20 = 5x4 per-(k, j) head
fits are minutes each — the whole fieldres stage sits far under 6 h.

Checkpoints follow the v2 contract (BUILD_NOTES #7): atomic tmp+os.replace,
keys {model, optimizer, scheduler, scaler, rng_state, epoch, cfg, fold_id},
saved every epoch AND every 30 min (first save immediately after init, well
inside the 1-h protected window); --resume autodetects last.pt.

CLI:
    python field_res.py fieldpre  [--quick] [--resume] [--variant temporal]
    python field_res.py fieldres  [--quick] [--resume] [--variant temporal]
    python field_res.py predict   [--quick] [--variant temporal]

Sentinels: fieldpre -> fieldpre_state[ _temporal].json,
fieldres -> oof_tier3[ _temporal].npz. FORCE=1 re-runs.
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd

import config2

# ── Constants ───────────────────────────────────────────────────────────────

SLV_CHANNELS = ["merra2_t2m", "merra2_rh2m", "merra2_u10", "merra2_v10"]
STATIC_HR_CHANNELS = ["st_elev", "st_road_km_5km", "st_nei_pm25_20km",
                      "st_pop_density"]
OBS_CHANNELS = ["obs_pa_raw", "obs_count"]
HMS_CHANNELS = ["hms_smoke"]

MASK_RATIO = 0.60          # fraction of 16x16 patches hidden per crop
PATCH = 16                 # patch side (= UNet 2^DEPTH, so masks tile cleanly)
GROUP_DROP_P = 0.15        # whole-channel-group drop probability per batch
LAMBDA_DISC = 0.10         # disc head weight: raw-scale huber(delta 15) runs
                           # ~10x the normalized recon huber(delta 1) — this
                           # keeps the two terms comparable in magnitude
RECON_HUBER_DELTA = 1.0    # loss in NORMALIZED units
DISC_HUBER_DELTA = 15.0    # raw ug/m3 — v1 models_deep.HUBER_DELTA

BASE_WIDTH = 48
CROP_SIZE = 96
PRETRAIN_EPOCHS, PRETRAIN_BATCH = 120, 16
FINETUNE_EPOCHS, FINETUNE_BATCH = 40, 4096
QUICK_EPOCHS = 2
LR_PRETRAIN, LR_FINETUNE = 1e-3, 1e-3
WEIGHT_DECAY = 1e-3
CKPT_EVERY_SEC = 1800      # 30-min wall-clock checkpoint (contract #7)
LOG_SIGMA2_CLAMP = 6.0

# Fourier features of the sub-cell offset: 4 wavelengths per axis x (sin,
# cos) = 8 pairs = 16 features. Log-spaced 5..200 km; nothing shorter than
# 5 km can be expressed, the hard cap against between-sensor hallucination.
FOURIER_WAVELENGTHS_KM = [5.0, 17.1, 58.5, 200.0]
KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON_EQ = 111.320

VAULT_DATE_START = getattr(config2, "VAULT_DATE_START", "2026-01-01")


def _say(msg):
    print(f"[aqnet2] {msg}", flush=True)


def _banner(name):
    _say(f"── stage: {name} " + "─" * max(0, 58 - len(name)))


# ── v1 module bootstrap (lazy — pipeline2 owns the canonical one) ───────────

def _v1_modules():
    """Import the reused v1 modules via a lazy sys.path bootstrap.

    research/aqnet (grids, models_deep, config, corrections) and
    research/deeplearning (dataset, models, train) are stable committed
    assets — audit contract #8. Import order matters: the DL dir first so
    v1 grids' own bootstrap finds `dataset` under the same name.
    """
    for p in (config2.DL_DIR, config2.V1_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    import dataset as dl_dataset
    import grids as v1_grids
    import models_deep as md
    return {"dl_dataset": dl_dataset, "grids": v1_grids, "md": md}


def _require_torch():
    """torch + v1 deep modules, or a clear degradation error (v1 style)."""
    v1 = _v1_modules()
    torch, dl_models, dl_train = v1["md"]._require_torch()
    return torch, dl_models, dl_train, v1


# ── External paths ──────────────────────────────────────────────────────────

def _external_paths():
    """external_paths.json from the data stage, with committed fallbacks."""
    ext = {}
    p = config2.artifact("external_paths.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as fh:
            ext = {k: v for k, v in json.load(fh).items() if v}
    ext.setdefault("hms_grid",
                   os.path.join(config2.PIPELINE_DIR, "hms_grid.parquet"))
    ext.setdefault("statics",
                   os.path.join(config2.PIPELINE_DIR,
                                "static_covariates.parquet"))
    geoscf_default = os.path.join(config2.V1_DIR, "data",
                                  "geoscf_pm25_20210101_20260501.parquet")
    if "geoscf" not in ext and os.path.exists(geoscf_default):
        ext["geoscf"] = geoscf_default
    return ext


# ── Stack v2 builder ────────────────────────────────────────────────────────

def _grid_hms_raster(path, dates, lat_axis, lon_axis, grid_deg):
    """hms_grid.parquet (0.1-deg polygon raster) -> (D, 1, H, W) planes.

    The parquet is already a raster, so cells are scattered by EXACT index —
    nearest-cell gridding would smear smoke across the whole state on days
    with few positive cells. Semantics mirror frame2.hms_join: inside the
    raster's date coverage a missing cell means NO smoke polygon (0.0);
    days outside coverage stay honestly NaN (data absence is not 'no
    smoke').
    """
    D, H, W = len(dates), len(lat_axis), len(lon_axis)
    arr = np.full((D, 1, H, W), np.nan, dtype=np.float32)
    if not path or not os.path.exists(path):
        _say(f"stack2: hms_grid parquet missing, NaN planes: {path}")
        return arr
    df = pd.read_parquet(path)
    if "cell_lat" in df.columns:
        df = df.rename(columns={"cell_lat": "lat", "cell_lon": "lon"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if "hms_smoke" not in df.columns:
        _say("stack2: hms_grid parquet lacks hms_smoke column, NaN planes")
        return arr
    dmin, dmax = df["date"].min(), df["date"].max()
    # datetime64 vs Timestamp hash mismatch — normalize keys (audit §1).
    by_date = {d: sub for d, sub in df.groupby("date")}
    n_cov = 0
    for i, d in enumerate(dates):
        day = pd.Timestamp(d)
        if day < dmin or day > dmax:
            continue
        arr[i, 0] = 0.0
        n_cov += 1
        sub = by_date.get(day)
        if sub is None:
            continue
        rows = np.rint((sub["lat"].to_numpy(np.float64) - lat_axis[0])
                       / grid_deg).astype(np.int64)
        cols = np.rint((sub["lon"].to_numpy(np.float64) - lon_axis[0])
                       / grid_deg).astype(np.int64)
        vals = sub["hms_smoke"].to_numpy(np.float64)
        ok = ((rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
              & np.isfinite(vals))
        arr[i, 0, rows[ok], cols[ok]] = vals[ok].astype(np.float32)
    _say(f"stack2: hms_grid rasterized {n_cov}/{D} days in coverage")
    return arr


def _obs_raster(data):
    """Raw-PA observation raster: per-day per-cell mean + obs-count channel.

    Built from the stack's own sparse obs records (raw ATM values,
    correction='raw'; in-Texas supervision sensors). Mean is NaN where a
    cell-day has no sensor; count is 0 there — an honest measured zero, not
    a fill (the count channel is exactly the coverage-density signal the
    dispatch pattern uses downstream).
    """
    D = len(data["dates"])
    H, W = len(data["lat"]), len(data["lon"])
    obs = data["obs"]
    sums = np.zeros((D, H, W), dtype=np.float64)
    cnt = np.zeros((D, H, W), dtype=np.float64)
    np.add.at(sums, (obs["day"], obs["row"], obs["col"]),
              obs["pm25"].astype(np.float64))
    np.add.at(cnt, (obs["day"], obs["row"], obs["col"]), 1.0)
    mean = np.where(cnt > 0, sums / np.maximum(cnt, 1.0), np.nan)
    out = np.empty((D, 2, H, W), dtype=np.float32)
    out[:, 0] = mean.astype(np.float32)
    out[:, 1] = cnt.astype(np.float32)
    return out


def _statics_planes(path, dates, grid_pts, shape, dl_dataset):
    """HR-static covariate planes (D, 4, H, W), tiled over days.

    static_covariates.parquet (0.01-deg table from fetchers2.statics) is
    gridded nearest-point onto the 0.1-deg stack grid. Column names are
    accepted with or without the st_ prefix. A 'year' column (the NEI
    year key) selects, per stack day, the latest static year <= that day's
    year (frame2.statics_join rule); otherwise one plane is tiled. Missing
    parquet/columns -> NaN planes with a printed notice.
    """
    D = len(dates)
    C = len(STATIC_HR_CHANNELS)
    arr = np.full((D, C, shape[0], shape[1]), np.nan, dtype=np.float32)
    if not path or not os.path.exists(path):
        _say(f"stack2: static_covariates.parquet missing, NaN planes: {path}")
        return arr
    st = pd.read_parquet(path)
    if "latitude" in st.columns and "lat" not in st.columns:
        st = st.rename(columns={"latitude": "lat", "longitude": "lon"})
    colmap = {}
    for want in STATIC_HR_CHANNELS:
        bare = want[3:]  # strip "st_"
        if want in st.columns:
            colmap[want] = want
        elif bare in st.columns:
            colmap[want] = bare
    missing = [c for c in STATIC_HR_CHANNELS if c not in colmap]
    if missing:
        _say(f"stack2: statics columns absent, NaN planes: {missing}")
    if not colmap:
        return arr

    def _plane_set(sub):
        planes = np.full((C, shape[0], shape[1]), np.nan, dtype=np.float32)
        p_lat = sub["lat"].to_numpy(np.float64)
        p_lon = sub["lon"].to_numpy(np.float64)
        for j, want in enumerate(STATIC_HR_CHANNELS):
            src = colmap.get(want)
            if src is None:
                continue
            planes[j] = dl_dataset._nearest(
                p_lat, p_lon, sub[src].to_numpy(np.float64), grid_pts, shape)
        return planes

    if "year" in st.columns:
        years = np.sort(st["year"].unique())
        qy = pd.DatetimeIndex(dates).year.to_numpy()
        pick = np.searchsorted(years, qy, side="right") - 1
        pick = years[np.clip(pick, 0, len(years) - 1)]
        for yr in np.unique(pick):
            planes = _plane_set(st[st["year"] == yr].reset_index(drop=True))
            arr[pick == yr] = planes[None]
        _say(f"stack2: statics gridded per year-key {list(years)}")
    else:
        arr[:] = _plane_set(st)[None]
        _say("stack2: statics gridded (single epoch, tiled)")
    return arr


def _append_chanmask(data):
    """Replace v1's per-GROUP flags with per-CHANNEL finite-mask planes.

    One 0/1 channel per real channel, order-aligned with the concatenated
    real channel order (dict insertion order IS channel order — audit §1).
    Computed after all groups are built and BEFORE any fill, so the masks
    record true per-pixel validity; mask channels are never NaN. v1's
    day-level per-group flags were measured dead inputs (audit §2) because
    they could not say WHICH pixel of a plane was real.
    """
    if "chanmask" in data["groups"]:
        raise AssertionError("chanmask already present — double append")
    names = [n for n in data["groups"] if n != "chanmask"]
    D = len(data["dates"])
    H, W = len(data["lat"]), len(data["lon"])
    n_real = sum(len(data["channels"][n]) for n in names)
    mask = np.empty((D, n_real, H, W), dtype=np.float32)
    mnames = []
    j = 0
    for name in names:
        g = data["groups"][name]
        for c, ch in enumerate(data["channels"][name]):
            mask[:, j] = np.isfinite(g[:, c]).astype(np.float32)
            mnames.append(f"mask_{ch}")
            j += 1
    data["groups"]["chanmask"] = mask
    data["channels"]["chanmask"] = mnames
    _say(f"stack2: chanmask appended ({n_real} per-channel finite planes; "
         f"v1 per-group flags retired)")


def build_stack2(start, end, grid_deg, external_paths):
    """Stack v2: v1 extended stack + slv/hms_grid/obs/statics_hr/chanmask.

    Calls v1 grids.build_extended_stack with correction='raw' (raw PA obs —
    no FRM-derived calibration may reach the shared pretrain stack), then:

      drop  'flags'      (v1 per-group day flags -> per-channel chanmask)
      drop  'smoke'      (v1 by-sensor nearest-smear smoke raster)
      add   'slv'        MERRA-2 SLV met (t2m/rh2m/u10/v10) nearest-cell
                         from the combined merra2 parquet — winds feed the
                         advection structure ERA5 could not provide (no CDS
                         creds on Phoenix; SLV substitution is final,
                         BUILD_NOTES #2)
      add   'hms_grid'   0.1-deg HMS polygon raster (exact-index scatter,
                         0 = no polygon inside coverage, NaN outside)
      add   'obs'        raw-PA observation raster: per-day cell mean +
                         obs-count channel (INSIDE the masking objective)
      add   'statics_hr' st_elev / st_road_km_5km / st_nei_pm25_20km /
                         st_pop_density planes, tiled
      add   'chanmask'   one 0/1 finite plane per real channel (LAST)

    No pa_cal raster and no T2 surfaces (fold honesty, DESIGN §8). The
    returned dict keeps the dataset.build_dataset schema so
    dataset.save_cache/load_cache work unchanged.
    """
    v1 = _v1_modules()
    grids, dl_dataset = v1["grids"], v1["dl_dataset"]
    ext = dict(external_paths or {})

    data = grids.build_extended_stack(
        start=start, end=end, grid_deg=grid_deg,
        geoscf_parquet=ext.get("geoscf"),
        merra2_parquet=ext.get("merra2"),
        correction="raw")

    # Retire v1 groups replaced by v2 ones (order of remaining groups is
    # preserved; new groups append after them).
    for gone in ("flags", "smoke"):
        if gone in data["groups"]:
            del data["groups"][gone]
            del data["channels"][gone]
    _say("stack2: dropped v1 groups ['flags', 'smoke']")

    grid_pts = dl_dataset._grid_points(data["lat"], data["lon"])
    shape = (len(data["lat"]), len(data["lon"]))

    merra2_path = ext.get("merra2")
    if merra2_path and os.path.exists(str(merra2_path)):
        data["groups"]["slv"] = grids._grid_daily_group(
            str(merra2_path), SLV_CHANNELS, data["dates"], grid_pts, shape,
            "slv")
    else:
        _say(f"stack2: merra2 parquet missing, slv NaN planes: {merra2_path}")
        data["groups"]["slv"] = np.full(
            (len(data["dates"]), len(SLV_CHANNELS), shape[0], shape[1]),
            np.nan, dtype=np.float32)
    data["channels"]["slv"] = list(SLV_CHANNELS)

    data["groups"]["hms_grid"] = _grid_hms_raster(
        ext.get("hms_grid"), data["dates"], data["lat"], data["lon"],
        float(data["grid_deg"]))
    data["channels"]["hms_grid"] = list(HMS_CHANNELS)

    data["groups"]["obs"] = _obs_raster(data)
    data["channels"]["obs"] = list(OBS_CHANNELS)

    data["groups"]["statics_hr"] = _statics_planes(
        ext.get("statics"), data["dates"], grid_pts, shape, dl_dataset)
    data["channels"]["statics_hr"] = list(STATIC_HR_CHANNELS)

    _append_chanmask(data)
    return data


def _guard_device(dev, quick):
    """Full-mode CPU fallback is a hard error (AQNET2_ALLOW_CPU=1 to
    override) -- the Phoenix cu13x incident trained silently on CPU."""
    dtype = getattr(dev, "type", str(dev))
    if (not quick and dtype != "cuda"
            and os.environ.get("AQNET2_ALLOW_CPU") != "1"):
        raise SystemExit(
            f"[aqnet2] field_res: resolved device is {dtype!r} in FULL mode "
            "-- a full training run must not silently fall back to CPU "
            "(cu126 wheels work on Phoenix). Set AQNET2_ALLOW_CPU=1 to "
            "override deliberately.")
    print(f"[aqnet2] field_res: device {dtype}", flush=True)
    return dev


def _stack_window(quick):
    start, end = config2.DATE_START, config2.DATE_END
    if quick:
        # Fixed pre-vault summer window (matches frame2/calibrate/priors
        # --quick). The former trailing-92-days window sat entirely inside
        # the vault period, leaving a quick fieldres with zero trainable
        # rows (review finding).
        start, end = "2024-07-01", "2024-09-30"
    return start, end


def stack2_cache_path(quick, grid_deg, start, end):
    """Cache filename embeds window + grid + correction (v1 precedent: a
    raw stack must never be reused for another config — audit §4)."""
    tag = "quick" if quick else "full"
    s = pd.Timestamp(start).strftime("%Y%m%d")
    e = pd.Timestamp(end).strftime("%Y%m%d")
    return os.path.join(config2.CACHE_DIR,
                        f"stack2_{tag}_{grid_deg:g}deg_raw_{s}_{e}.npz")


def load_or_build_stack2(quick=False, grid_deg=None, cache_path=None):
    """Cached stack v2 (dataset.save_cache schema, atomic write)."""
    v1 = _v1_modules()
    dl_dataset = v1["dl_dataset"]
    grid_deg = float(grid_deg if grid_deg is not None else config2.GRID_DEG)
    start, end = _stack_window(quick)
    # NOTE: FORCE=1 (the stage sentinel override) deliberately does NOT
    # invalidate this cache — validity is keyed by the filename config; a
    # forced fieldres re-run must not trigger an hours-long stack rebuild.
    # Delete the npz to rebuild.
    path = cache_path or stack2_cache_path(quick, grid_deg, start, end)
    if os.path.exists(path):
        _say(f"stack2: loading cache {path}")
        data = dl_dataset.load_cache(path)
        if "chanmask" not in data["groups"]:
            raise AssertionError(f"cache {path} lacks the chanmask group — "
                                 f"not a stack v2 cache; delete and rebuild")
        return data
    _say(f"stack2: building ({start}..{end}, {grid_deg:g} deg)")
    data = build_stack2(start, end, grid_deg, _external_paths())
    tmp = path + ".tmp.npz"
    dl_dataset.save_cache(data, tmp)
    os.replace(tmp, path)
    _say(f"stack2: cached -> {path}")
    return data


# ── Channel layout / normalization ──────────────────────────────────────────

def _channel_layout(data):
    """[(group, channel_name), ...] over real groups, dict order (=chanmask
    order). The layout is persisted in checkpoints and asserted on reload."""
    return [(g, ch) for g in data["groups"] if g != "chanmask"
            for ch in data["channels"][g]]


def _normalize_stack(data, md, dl_dataset, stats=None):
    """Prefill norm stats -> mean-fill -> standardize, on a PRIVATE copy.

    Stats over finite pixels only (v1 prefill fix — post-fill stats shrink
    std toward zero on sparse channels); chanmask gets identity stats so the
    0/1 validity signal arrives unscaled. Filling with the per-channel mean
    then standardizing lands every filled pixel at exactly 0 in normalized
    space — permitted for INPUT tensors only, because chanmask carries
    validity and the loss reads only pre-fill-finite pixels.
    Returns (groups_copy, stats).
    """
    groups = {n: a.copy() for n, a in data["groups"].items()}
    if stats is None:
        stats = md._compute_norm_stats_prefill(groups)
        stats["chanmask"] = {
            "mean": [0.0] * groups["chanmask"].shape[1],
            "std": [1.0] * groups["chanmask"].shape[1]}
    dl_dataset.fill_missing(
        groups, fill_values={n: stats[n]["mean"] for n in groups})
    dl_dataset.apply_norm_stats(groups, stats)
    return groups, stats


# ── Models (torch-lazy factories — module imports cleanly without torch) ────

def _build_field_net(torch, dl_models, in_ch, n_real, base_width):
    """Encoder-decoder over v1 UNet blocks with the softplus head replaced.

    The v1 UNet blocks (DoubleConv/UpBlock, GroupNorm, replicate padding to
    /16) are reused as-is per the audit contract; the forward is re-wired
    through the same submodules so the 3 encoder depths (e1, e2, e3) are
    exposed for the INR decoder, and the softplus regression head is
    replaced by linear 1x1 heads — reconstruction targets are SIGNED
    normalized values and the discrepancy (obs − CTM) is signed too, so the
    non-negativity clamp of the stock head would be a wiring bug here.
    """
    nn = torch.nn
    F = torch.nn.functional

    class FieldNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.unet = dl_models.UNet(in_channels=in_ch,
                                       base_width=base_width,
                                       out_channels=1)  # stock head unused
            self.recon_head = nn.Conv2d(base_width, n_real, kernel_size=1)
            self.disc_head = nn.Conv2d(base_width, 1, kernel_size=1)

        def encode(self, x):
            h, w = x.shape[-2:]
            mult = 2 ** self.unet.DEPTH
            pad_h = (mult - h % mult) % mult
            pad_w = (mult - w % mult) % mult
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
            u = self.unet
            e1 = u.enc1(x)
            e2 = u.enc2(u.pool(e1))
            e3 = u.enc3(u.pool(e2))
            e4 = u.enc4(u.pool(e3))
            z = u.bottleneck(u.pool(e4))
            d4 = u.up4(z, e4)
            d3 = u.up3(d4, e3)
            d2 = u.up2(d3, e2)
            d1 = u.up1(d2, e1)
            return d1, (e1, e2, e3), (h, w)

        def forward(self, x):
            d1, feats, (h, w) = self.encode(x)
            recon = self.recon_head(d1)[..., :h, :w]
            disc = self.disc_head(d1)[..., :h, :w]
            return recon, disc, feats

    return FieldNet()


def _build_inr_head(torch, in_dim, hidden=256):
    """INR point decoder: MLP (3x256) -> (r2_hat, log sigma^2)."""
    nn = torch.nn

    class INRHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.SiLU(inplace=True),
                nn.Linear(hidden, hidden), nn.SiLU(inplace=True),
                nn.Linear(hidden, hidden), nn.SiLU(inplace=True),
                nn.Linear(hidden, 2))

        def forward(self, x):
            out = self.mlp(x)
            mu = out[:, 0]
            log_s2 = torch.clamp(out[:, 1], -LOG_SIGMA2_CLAMP,
                                 LOG_SIGMA2_CLAMP)
            return mu, log_s2

    return INRHead()


# ── Checkpoint contract (BUILD_NOTES #7) ────────────────────────────────────

def _rng_capture(torch, np_rng):
    state = {"torch": torch.get_rng_state(),
             "numpy": np_rng.bit_generator.state if np_rng is not None
             else None}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _rng_restore(torch, np_rng, state):
    if state is None:
        return
    try:
        torch.set_rng_state(state["torch"].cpu()
                            if hasattr(state["torch"], "cpu")
                            else state["torch"])
        if np_rng is not None and state.get("numpy") is not None:
            np_rng.bit_generator.state = state["numpy"]
        if torch.cuda.is_available() and state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as e:  # a mismatched device layout must not kill resume
        _say(f"WARNING: RNG state not fully restored ({e}); continuing")


def _save_ckpt(torch, path, model, optimizer, scheduler, scaler, epoch, cfg,
               fold_id, np_rng=None):
    """Atomic checkpoint with the frozen v2 key set. v1 train_fusion_unet
    lacked optimizer/scheduler/RNG state and atomic writes (audit §2) —
    this is the superset, not a copy."""
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng_state": _rng_capture(torch, np_rng),
        "epoch": int(epoch),
        "cfg": cfg,
        "fold_id": fold_id,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _load_ckpt(torch, path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=False)


def _variant_tag(variant):
    return "temporal" if str(variant or "") == "temporal" else "primary"


def _pretrain_paths(variant):
    tag = _variant_tag(variant)
    last = config2.artifact(f"fieldpre_{tag}_last.pt", sub="field")
    best = config2.artifact(f"fieldpre_{tag}_best.pt", sub="field")
    suffix = "_temporal" if tag == "temporal" else ""
    state = config2.artifact(f"fieldpre_state{suffix}.json")
    return last, best, state


def _finetune_path(variant, k, j):
    tag = _variant_tag(variant)
    return config2.artifact(f"fieldres_{tag}_f{k}_{j}.pt", sub="field")


def _oof_path(variant):
    suffix = "_temporal" if _variant_tag(variant) == "temporal" else ""
    return config2.artifact(f"oof_tier3{suffix}.npz")


# ── Stage: fieldpre ─────────────────────────────────────────────────────────

def pretrain(cfg):
    """Masked multimodal autoencoder pretraining. Returns the ckpt path.

    Objective per crop (96x96, one item per day via the reused v1
    _CropDayDataset — its supervision mask carries the DISC-target validity,
    so crops are drawn from days with at least one station pixel and biased
    toward stations, which feeds the discrepancy head):

      hide  60% of 16x16 patches (fresh Bernoulli patch grid per sample)
            + each channel group dropped whole with p=0.15 per batch
      input concat(real_channels * keep, chanmask * keep) — hidden or
            dropped pixels are zeroed AND their mask planes zeroed, so the
            encoder always knows data-vs-void (v1 zeroed values but left
            flags up — the model could not tell dropout from presence)
      loss  masked huber (normalized units) on pixels that are BOTH
            pre-fill-finite AND hidden/dropped — filled pixels are
            structurally outside the loss — plus LAMBDA_DISC * masked huber
            (raw ug/m3) of the disc head against obs − geoscf_pm25 at
            station pixels whose obs channel is hidden.

    --variant temporal restricts pretraining days to < TEMPORAL_CUTOFF —
    the sole basis for temporal-holdout claims (DESIGN §8).
    """
    torch, dl_models, dl_train, v1 = _require_torch()
    md, dl_dataset = v1["md"], v1["dl_dataset"]
    quick = bool(cfg.get("quick"))
    variant = _variant_tag(cfg.get("variant"))
    seed = int(cfg.get("seed", config2.SEED))
    epochs = int(cfg.get("epochs") or (QUICK_EPOCHS if quick
                                       else PRETRAIN_EPOCHS))
    batch = int(cfg.get("batch_size") or PRETRAIN_BATCH)
    crop = int(cfg.get("crop_size") or CROP_SIZE)
    base_width = int(cfg.get("base_width") or BASE_WIDTH)
    lr = float(cfg.get("lr") or LR_PRETRAIN)

    data = load_or_build_stack2(quick=quick, grid_deg=cfg.get("grid_deg"),
                                cache_path=cfg.get("stack_path"))

    # Temporal purity: slice days BEFORE any statistics so the temporal
    # variant's norm stats cannot see post-cutoff data either.
    if variant == "temporal":
        keep = data["dates"] < np.datetime64(pd.Timestamp(
            config2.TEMPORAL_CUTOFF))
        if not keep.any():
            raise AssertionError("temporal variant selected but no stack "
                                 "days precede TEMPORAL_CUTOFF")
        sel = np.flatnonzero(keep)
        remap = {int(d): i for i, d in enumerate(sel)}
        okeep = np.isin(data["obs"]["day"], sel)
        data = {
            "groups": {n: a[sel] for n, a in data["groups"].items()},
            "channels": data["channels"],
            "lat": data["lat"], "lon": data["lon"],
            "dates": data["dates"][sel],
            "obs": {k2: (np.array([remap[int(d)] for d in
                                   data["obs"]["day"][okeep]], dtype=np.int64)
                         if k2 == "day" else data["obs"][k2][okeep])
                    for k2 in data["obs"]},
            "grid_deg": data["grid_deg"],
        }
        _say(f"fieldpre: temporal variant — {len(sel)} days "
             f"< {config2.TEMPORAL_CUTOFF}")

    layout = _channel_layout(data)
    n_real = len(layout)
    real_groups = [g for g in data["groups"] if g != "chanmask"]
    group_slices = {}
    pos = 0
    for g in real_groups:
        c = len(data["channels"][g])
        group_slices[g] = (pos, pos + c)
        pos += c
    if pos != n_real:
        raise AssertionError("channel layout / group slice mismatch")
    obs_lo, _obs_hi = group_slices["obs"]  # obs_pa_raw is channel obs_lo

    # Discrepancy target on the RAW scale, computed pre-normalization.
    obs_raw = data["groups"]["obs"][:, 0].astype(np.float32)
    have_ctm = "ctm" in data["groups"]
    if have_ctm:
        disc = obs_raw - data["groups"]["ctm"][:, 0]
    else:
        _say("fieldpre: WARNING — no ctm group (geoscf parquet absent); "
             "discrepancy head disabled, pure MAE objective")
        disc = np.full_like(obs_raw, np.nan)
    disc_mask = np.isfinite(disc).astype(np.float32)
    disc_y = np.where(np.isfinite(disc), disc, 0.0).astype(np.float32)
    if not have_ctm:
        # _CropDayDataset needs >=1 supervised pixel per day; fall back to
        # station-pixel masks so crop placement still works.
        disc_mask = (data["groups"]["obs"][:, 1] > 0).astype(np.float32)

    groups, stats = _normalize_stack(data, md, dl_dataset)

    ds = md._CropDayDataset(groups, disc_y, disc_mask, crop_size=crop)
    if len(ds) == 0:
        raise AssertionError("no stack day has a station pixel — cannot "
                             "place crops")
    device = _guard_device(md._resolve_device(torch, cfg.get("device", "auto")), bool(cfg.get("quick")))
    use_amp = device.type == "cuda"
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=True,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=device.type == "cuda")

    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    model = _build_field_net(torch, dl_models, in_ch=2 * n_real,
                             n_real=n_real, base_width=base_width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=WEIGHT_DECAY)
    scheduler, warmup = md._make_scheduler(torch, optimizer, epochs)
    scaler = md._make_grad_scaler(torch) if use_amp else None

    ckpt_cfg = {
        "stage": "fieldpre", "variant": variant, "quick": quick,
        "seed": seed, "epochs": epochs, "batch_size": batch,
        "crop_size": crop, "base_width": base_width, "lr": lr,
        "in_ch": 2 * n_real, "n_real": n_real,
        "channel_layout": [list(t) for t in layout],
        "group_slices": {g: list(v) for g, v in group_slices.items()},
        "norm_stats": stats,
        "grid_deg": float(data["grid_deg"]),
        "lat0": float(data["lat"][0]), "lon0": float(data["lon"][0]),
        "H": len(data["lat"]), "W": len(data["lon"]),
        "mask_ratio": MASK_RATIO, "patch": PATCH,
        "group_drop_p": GROUP_DROP_P, "lambda_disc": LAMBDA_DISC,
        "have_ctm": bool(have_ctm),
    }

    last_path, best_path, state_path = _pretrain_paths(variant)
    start_epoch, best_loss = 0, float("inf")
    if cfg.get("resume") and os.path.exists(last_path):
        ck = _load_ckpt(torch, last_path, device)
        model.load_state_dict(ck["model"])
        if ck.get("optimizer"):
            optimizer.load_state_dict(ck["optimizer"])
        if ck.get("scheduler"):
            scheduler.load_state_dict(ck["scheduler"])
        if scaler is not None and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        _rng_restore(torch, np_rng, ck.get("rng_state"))
        start_epoch = int(ck["epoch"]) + 1
        _say(f"fieldpre: resumed {last_path} at epoch {start_epoch}")

    # First checkpoint immediately — inside the 1-h protected window even
    # if the first epoch is slow (contract #7).
    _save_ckpt(torch, last_path, model, optimizer, scheduler, scaler,
               start_epoch - 1, ckpt_cfg, None, np_rng)
    last_save = time.time()

    n_hide = None
    history = []
    for ep in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        tot_r = tot_d = n_b = 0.0
        for g, y, m in loader:
            x_real = torch.cat([g[gr] for gr in real_groups], dim=1)
            cm = g["chanmask"]
            x_real = x_real.to(device, non_blocking=True)
            cm = cm.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            B, _, h, w = x_real.shape

            # 60% random patch masking, fresh per sample.
            ph, pw = h // PATCH, w // PATCH
            if n_hide is None:
                n_hide = f"{ph * pw} patches/crop"
            hide = (torch.rand(B, 1, ph, pw, device=device)
                    < MASK_RATIO).float()
            hide = hide.repeat_interleave(PATCH, dim=2)
            hide = hide.repeat_interleave(PATCH, dim=3)[..., :h, :w]

            # Whole-channel-group drops, per batch (v1 modality-dropout
            # idiom, numpy rng).
            dch = torch.zeros(1, n_real, 1, 1, device=device)
            obs_dropped = 0.0
            for gr in real_groups:
                if np_rng.random() < GROUP_DROP_P:
                    lo, hi2 = group_slices[gr]
                    dch[0, lo:hi2] = 1.0
                    if gr == "obs":
                        obs_dropped = 1.0

            keep = (1.0 - hide) * (1.0 - dch)
            x_in = torch.cat([x_real * keep, cm * keep], dim=1)
            # Loss mask: pre-fill-finite AND (patch-hidden OR group-dropped).
            lm = cm * torch.clamp(hide + dch, max=1.0)
            obs_hidden = torch.clamp(hide + obs_dropped, max=1.0)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.autocast(device_type="cuda"):
                    recon, dpred, _ = model(x_in)
                    loss_r = md.masked_huber(recon, x_real, lm,
                                             delta=RECON_HUBER_DELTA)
                    if have_ctm:
                        dm = m.unsqueeze(1) * obs_hidden
                        loss_d = md.masked_huber(dpred, y.unsqueeze(1), dm,
                                                 delta=DISC_HUBER_DELTA)
                    else:
                        loss_d = torch.zeros((), device=device)
                    loss = loss_r + LAMBDA_DISC * loss_d
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                recon, dpred, _ = model(x_in)
                loss_r = md.masked_huber(recon, x_real, lm,
                                         delta=RECON_HUBER_DELTA)
                if have_ctm:
                    dm = m.unsqueeze(1) * obs_hidden
                    loss_d = md.masked_huber(dpred, y.unsqueeze(1), dm,
                                             delta=DISC_HUBER_DELTA)
                else:
                    loss_d = torch.zeros((), device=device)
                loss = loss_r + LAMBDA_DISC * loss_d
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            tot_r += float(loss_r.item())
            tot_d += float(loss_d.item())
            n_b += 1
            if time.time() - last_save > CKPT_EVERY_SEC:
                # Mid-epoch wall-clock save; resume replays this epoch.
                _save_ckpt(torch, last_path, model, optimizer, scheduler,
                           scaler, ep - 1, ckpt_cfg, None, np_rng)
                last_save = time.time()

        scheduler.step()
        mean_r = tot_r / max(n_b, 1)
        mean_d = tot_d / max(n_b, 1)
        ep_loss = mean_r + LAMBDA_DISC * mean_d
        history.append({"epoch": ep, "recon": mean_r, "disc": mean_d,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "sec": round(time.time() - t0, 1)})
        _say(f"fieldpre epoch {ep + 1}/{epochs} recon={mean_r:.4f} "
             f"disc={mean_d:.3f} ({time.time() - t0:.0f}s)")
        _save_ckpt(torch, last_path, model, optimizer, scheduler, scaler,
                   ep, ckpt_cfg, None, np_rng)
        last_save = time.time()
        if ep_loss < best_loss:
            best_loss = ep_loss
            _save_ckpt(torch, best_path, model, optimizer, scheduler, scaler,
                       ep, ckpt_cfg, None, np_rng)

    state = {"variant": variant, "quick": quick, "epochs_run": epochs,
             "n_days": len(data["dates"]), "n_real_channels": n_real,
             "base_width": base_width, "masking": n_hide,
             "have_ctm": bool(have_ctm), "best_loss": best_loss,
             "ckpt_last": last_path, "ckpt_best": best_path,
             "history": history[-20:]}
    tmp = state_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, state_path)
    _say(f"fieldpre: state -> {state_path}")
    return best_path if os.path.exists(best_path) else last_path


# ── Shared fine-tune / predict context ──────────────────────────────────────

def _load_frame_folds():
    frame_path = config2.artifact("frame_truth.parquet")
    folds_path = config2.artifact("folds2.json")
    if not os.path.exists(frame_path):
        raise SystemExit("[aqnet2] frame_truth.parquet missing — run the "
                         "features stage first")
    if not os.path.exists(folds_path):
        raise SystemExit("[aqnet2] folds2.json missing — run the features "
                         "stage first")
    frame = pd.read_parquet(frame_path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    with open(folds_path, "r", encoding="utf-8") as fh:
        folds = json.load(fh)
    return frame, folds


def _load_f2(frame, folds):
    """F2 incumbent OOF below T3: T1 oof + gated T2 residual when available.

    The gates stage runs AFTER fieldres in the DAG, so gates.json normally
    does not exist yet. Fallback ladder (recorded in the meta artifact):
      1. gates.json has a "tier2" entry  -> compose.apply_gates (exact)
      2. oof_tier2.npz exists            -> provisional alpha=1 add on
                                            avail rows (residual is NaN off
                                            avail by contract, never filled)
      3. neither                          -> T1 alone (documented fallback)
    """
    n = len(frame)
    t1_path = config2.artifact("oof_tier1.npz")
    if not os.path.exists(t1_path):
        raise SystemExit("[aqnet2] oof_tier1.npz missing — run the skeleton "
                         "stage first")
    with np.load(t1_path, allow_pickle=True) as z:
        t1 = np.asarray(z["oof"], dtype=np.float64)
    if len(t1) != n:
        raise AssertionError(f"oof_tier1 length {len(t1)} != frame rows {n}")

    t2_path = config2.artifact("oof_tier2.npz")
    if not os.path.exists(t2_path):
        _say("fieldres: oof_tier2.npz absent — F2 = T1 alone (fallback)")
        return t1, "t1_only"
    with np.load(t2_path) as z:
        oof_r = np.asarray(z["oof_r"], dtype=np.float64)
        avail = np.asarray(z["avail"]).astype(bool)
        pattern = np.asarray(z["pattern_id"])
    if len(oof_r) != n:
        raise AssertionError(f"oof_tier2 length {len(oof_r)} != frame rows {n}")
    if avail.any() and not np.isfinite(oof_r[avail]).all():
        raise AssertionError("oof_tier2: non-finite residual on avail rows")
    if (~avail).any() and not np.isnan(oof_r[~avail]).all():
        raise AssertionError("oof_tier2: residual not NaN on unavail rows — "
                             "a fill is riding in the residual channel")

    gates_path = config2.artifact("gates.json")
    if os.path.exists(gates_path):
        try:
            import compose
            gates = compose.load_gates(gates_path)
            if "tier2" in gates:
                stratum = _coverage_bins(frame)
                f2 = compose.apply_gates(t1, oof_r, avail, pattern, stratum,
                                         gates["tier2"])
                _say("fieldres: F2 = T1 + gated T2 (gates.json)")
                return f2, "t1_plus_gated_t2"
        except Exception as e:
            _say(f"fieldres: gates.json unusable ({e}); provisional add")
    f2 = t1.copy()
    add = avail & np.isfinite(oof_r) & np.isfinite(t1)
    f2[add] = t1[add] + oof_r[add]
    _say(f"fieldres: F2 = T1 + provisional alpha=1 T2 on "
         f"{int(add.sum()):,}/{n:,} rows (gates not fit yet — documented)")
    return f2, "t1_plus_provisional_t2"


def _coverage_bins(frame):
    """Coverage-density bin per row from nbr_pacal_count_50km:
    0 (no station in 50 km), 1 (1-3), 2 (>=4). Contract #4 binning."""
    col = "nbr_pacal_count_50km"
    if col not in frame.columns:
        _say(f"WARNING: {col} missing from frame — pattern_id all 0 "
             f"(off-support)")
        return np.zeros(len(frame), dtype=np.int8)
    cnt = pd.to_numeric(frame[col], errors="coerce").to_numpy(np.float64)
    cnt = np.where(np.isfinite(cnt), cnt, 0.0)
    out = np.zeros(len(frame), dtype=np.int8)
    out[(cnt >= 1) & (cnt < 4)] = 1
    out[cnt >= 4] = 2
    return out


def _vault_mask(frame, folds):
    """Vault airlock rows: vault units OR vault-period dates. These rows are
    never trained on and never receive a T3 residual (avail=0)."""
    import frame2
    vault_units = frame2._as_unit_set(folds.get("vault_sites", []))
    uid = frame["unit_id"].astype(str)
    # copy: pandas-3 CoW can return read-only arrays from to_numpy()
    m = np.array(uid.isin(vault_units).to_numpy(), copy=True)
    m |= (frame["date"] >= pd.Timestamp(VAULT_DATE_START)).to_numpy()
    return m


def _fourier_features(lats, rf, cf, grid_deg):
    """Band-limited Fourier features of the sub-cell offset (8 sin/cos
    pairs: 4 wavelengths x 2 axes). Offsets are km from the containing cell
    center — the INR's only sub-cell position signal, so its spatial
    expressiveness is hard-capped at 5 km wavelength."""
    dy = (rf - np.rint(rf)) * grid_deg * KM_PER_DEG_LAT
    dx = ((cf - np.rint(cf)) * grid_deg * KM_PER_DEG_LON_EQ
          * np.cos(np.radians(lats)))
    feats = np.empty((len(rf), 4 * len(FOURIER_WAVELENGTHS_KM)),
                     dtype=np.float32)
    for i, lam in enumerate(FOURIER_WAVELENGTHS_KM):
        w = 2.0 * np.pi / lam
        feats[:, 4 * i + 0] = np.sin(w * dx)
        feats[:, 4 * i + 1] = np.cos(w * dx)
        feats[:, 4 * i + 2] = np.sin(w * dy)
        feats[:, 4 * i + 3] = np.cos(w * dy)
    return feats


def _sample_row_features(frame, pre_ckpt, cfg):
    """Per-row INR inputs from the FROZEN pretrained encoder.

    One full-grid encoder forward per stack day (shared by all 20 fold
    heads — the encoder is frozen, see module docstring), then bilinear
    sampling of the 3 encoder depths at each row's exact fractional pixel
    (NO rint snapping), plus HR statics + statics validity sampled at the
    same exact point, plus the sub-cell Fourier features.

    Returns (X float32 [n, dim], ok bool [n]) — ok=False rows (date outside
    the stack, point outside the grid) get zero rows that are never used
    (avail=0 downstream).
    """
    torch, dl_models, _dl_train, v1 = _require_torch()
    md, dl_dataset = v1["md"], v1["dl_dataset"]
    ck_cfg = pre_ckpt["cfg"]

    data = load_or_build_stack2(quick=bool(cfg.get("quick")),
                                grid_deg=cfg.get("grid_deg"),
                                cache_path=cfg.get("stack_path"))
    layout = [list(t) for t in _channel_layout(data)]
    if layout != [list(t) for t in ck_cfg["channel_layout"]]:
        raise AssertionError("stack channel layout differs from the "
                             "pretrain checkpoint — rebuild the stack or "
                             "re-pretrain (never sample across layouts)")
    groups, _ = _normalize_stack(data, md, dl_dataset,
                                 stats=ck_cfg["norm_stats"])

    real_groups = [g for g in data["groups"] if g != "chanmask"]
    st_lo, st_hi = ck_cfg["group_slices"]["statics_hr"]
    base_width = int(ck_cfg["base_width"])
    n_real = int(ck_cfg["n_real"])
    feat_dim = 7 * base_width + 2 * (st_hi - st_lo) \
        + 4 * len(FOURIER_WAVELENGTHS_KM)

    device = _guard_device(md._resolve_device(torch, cfg.get("device", "auto")), bool(cfg.get("quick")))
    model = _build_field_net(torch, dl_models, in_ch=2 * n_real,
                             n_real=n_real, base_width=base_width).to(device)
    model.load_state_dict(pre_ckpt["model"])
    model.eval()

    lat0, lon0 = float(data["lat"][0]), float(data["lon"][0])
    g = float(data["grid_deg"])
    H, W = len(data["lat"]), len(data["lon"])
    lats = frame["lat"].to_numpy(np.float64)
    lons = frame["lon"].to_numpy(np.float64)
    rf = (lats - lat0) / g
    cf = (lons - lon0) / g
    day_of_row = pd.DatetimeIndex(data["dates"]).get_indexer(
        pd.DatetimeIndex(frame["date"]))
    inb = ((rf >= -0.5) & (rf <= H - 0.5) & (cf >= -0.5) & (cf <= W - 0.5)
           & (day_of_row >= 0))

    n = len(frame)
    X = np.zeros((n, feat_dim), dtype=np.float32)
    fourier = _fourier_features(lats, rf, cf, g)
    mult = 16
    Hp, Wp = H + (mult - H % mult) % mult, W + (mult - W % mult) % mult

    used_days = np.unique(day_of_row[inb])
    _say(f"fieldres: encoding {len(used_days)} stack days for "
         f"{int(inb.sum()):,}/{n:,} in-coverage rows")
    t0 = time.time()
    with torch.no_grad():
        for di, d in enumerate(used_days):
            x_real = torch.from_numpy(np.ascontiguousarray(
                np.concatenate([groups[gr][d] for gr in real_groups],
                               axis=0)))[None]
            cm = torch.from_numpy(np.ascontiguousarray(
                groups["chanmask"][d]))[None]
            x_in = torch.cat([x_real, cm], dim=1).to(device)
            _d1, (e1, e2, e3), _hw = model.encode(x_in)

            rows = np.flatnonzero(inb & (day_of_row == d))
            yn = torch.tensor((2.0 * (rf[rows] + 0.5) / Hp) - 1.0,
                              dtype=torch.float32, device=device)
            xn = torch.tensor((2.0 * (cf[rows] + 0.5) / Wp) - 1.0,
                              dtype=torch.float32, device=device)
            grid_pad = torch.stack([xn, yn], dim=-1)[None, :, None, :]
            samples = []
            for feat in (e1, e2, e3):
                s = torch.nn.functional.grid_sample(
                    feat.float(), grid_pad, mode="bilinear",
                    padding_mode="border", align_corners=False)
                samples.append(s[0, :, :, 0].T)  # (n_rows, C)
            # Statics + their validity, sampled at the exact point from the
            # UNPADDED input tensor (frame2's 0.01-deg join is the tabular
            # path; the field path stays self-contained on the stack).
            yn2 = torch.tensor((2.0 * (rf[rows] + 0.5) / H) - 1.0,
                               dtype=torch.float32, device=device)
            xn2 = torch.tensor((2.0 * (cf[rows] + 0.5) / W) - 1.0,
                               dtype=torch.float32, device=device)
            grid_raw = torch.stack([xn2, yn2], dim=-1)[None, :, None, :]
            st_vals = torch.nn.functional.grid_sample(
                x_real[:, st_lo:st_hi].to(device), grid_raw, mode="bilinear",
                padding_mode="border", align_corners=False)[0, :, :, 0].T
            st_mask = torch.nn.functional.grid_sample(
                cm[:, st_lo:st_hi].to(device), grid_raw, mode="bilinear",
                padding_mode="border", align_corners=False)[0, :, :, 0].T
            samples.extend([st_vals, st_mask])
            feats = torch.cat(samples, dim=1).cpu().numpy()
            X[rows, :feats.shape[1]] = feats
            if (di + 1) % 200 == 0:
                _say(f"  encoded {di + 1}/{len(used_days)} days "
                     f"({time.time() - t0:.0f}s)")
    X[:, feat_dim - fourier.shape[1]:] = fourier
    X[~inb] = 0.0
    return X, inb


def _build_shared(cfg, frame=None, folds=None):
    """Everything the 20 fine-tunes and predict_oof share: frame, folds,
    per-row encoder features, F2 incumbent and the r2 target. frame/folds
    may be passed pre-loaded (predict_oof's public signature); defaults are
    the canonical artifacts."""
    torch, _m, _t, _v1 = _require_torch()
    if frame is None or folds is None:
        frame_d, folds_d = _load_frame_folds()
        frame = frame_d if frame is None else frame
        folds = folds_d if folds is None else folds
    frame = frame.reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    variant = _variant_tag(cfg.get("variant"))
    last_path, best_path, _state = _pretrain_paths(variant)
    pre_path = cfg.get("pretrain_ckpt") or (
        best_path if os.path.exists(best_path) else last_path)
    if not os.path.exists(pre_path):
        raise SystemExit(f"[aqnet2] pretrain checkpoint missing ({pre_path})"
                         f" — run fieldpre first")
    pre_ckpt = _load_ckpt(torch, pre_path)
    X, ok = _sample_row_features(frame, pre_ckpt, cfg)
    f2, f2_source = _load_f2(frame, folds)
    y = frame["y"].to_numpy(np.float64)
    r2 = y - f2
    w = frame["w"].to_numpy(np.float64)
    vault = _vault_mask(frame, folds)
    outer = np.asarray(folds["outer_fold"], dtype=int)
    if len(outer) != len(frame):
        raise AssertionError("folds2 outer_fold length != frame rows")
    inner = {int(k): np.asarray(v, dtype=int)
             for k, v in folds["inner_fold"].items()}
    return {"frame": frame, "folds": folds, "X": X, "ok": ok, "r2": r2,
            "w": w, "vault": vault, "outer": outer, "inner": inner,
            "f2": f2, "f2_source": f2_source, "pre_ckpt": pre_ckpt,
            "pre_path": pre_path, "variant": variant}


# ── Stage: fieldres (per-fold INR fine-tunes) ───────────────────────────────

def finetune(cfg, fold, shared=None):
    """Fine-tune the INR head for (outer k, inner j). Returns the ckpt path.

    Training rows: stack-covered, finite r2/weight, outer_fold != k, and
    inner_fold[k] != j — so every row of inner fold j (and every row of
    outer fold k) is out-of-sample for this head. Vault rows are excluded
    and asserted out. The temporal variant additionally trains only on
    pre-cutoff rows. Loss: precision-weighted Gaussian NLL on (r2_hat,
    log sigma^2) — residuals are SIGNED; nothing here routes through
    models_tabular (its predictions clip at 0, contract #8).
    """
    torch, _dl_models, _dl_train, v1 = _require_torch()
    md = v1["md"]
    k, j = int(fold[0]), int(fold[1])
    if shared is None:
        shared = _build_shared(cfg)
    quick = bool(cfg.get("quick"))
    variant = shared["variant"]
    seed = int(cfg.get("seed", config2.SEED))
    epochs = int(cfg.get("ft_epochs") or (QUICK_EPOCHS if quick
                                          else FINETUNE_EPOCHS))
    batch = int(cfg.get("ft_batch") or FINETUNE_BATCH)
    lr = float(cfg.get("ft_lr") or LR_FINETUNE)

    out_path = _finetune_path(variant, k, j)
    if (os.path.exists(out_path) and not cfg.get("refit")
            and os.environ.get("FORCE") != "1"):
        _say(f"fieldres f{k}_{j}: {os.path.basename(out_path)} exists -- skip")
        return out_path

    inner_k = shared["inner"].get(k)
    if inner_k is None:
        raise AssertionError(f"folds2 inner_fold has no entry for outer {k}")
    frame = shared["frame"]
    tr = (shared["ok"] & np.isfinite(shared["r2"]) & np.isfinite(shared["w"])
          & (shared["w"] > 0) & ~shared["vault"]
          & (shared["outer"] != k) & (inner_k != j) & (inner_k >= 0))
    if variant == "temporal":
        tr &= (frame["date"] < pd.Timestamp(config2.TEMPORAL_CUTOFF)
               ).to_numpy()
    n_tr = int(tr.sum())
    if n_tr < 100:
        raise AssertionError(f"fieldres f{k}_{j}: only {n_tr} training rows")
    # Vault airlock re-assert (belt and braces — never a warning).
    import frame2
    vault_units = frame2._as_unit_set(shared["folds"].get("vault_sites", []))
    breach = set(frame.loc[tr, "unit_id"].astype(str)) & vault_units
    if breach:
        raise AssertionError(f"vault airlock breach in fieldres training "
                             f"rows: {sorted(breach)[:5]}")

    device = _guard_device(md._resolve_device(torch, cfg.get("device", "auto")), bool(cfg.get("quick")))
    torch.manual_seed(seed + 1000 * k + j)
    in_dim = shared["X"].shape[1]
    head = _build_inr_head(torch, in_dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr,
                                  weight_decay=WEIGHT_DECAY)
    scheduler, _warm = md._make_scheduler(torch, optimizer, epochs)

    Xt = torch.from_numpy(shared["X"][tr])
    rt = torch.from_numpy(shared["r2"][tr].astype(np.float32))
    wt = torch.from_numpy(shared["w"][tr].astype(np.float32))
    idx_all = np.sort(np.flatnonzero(tr))  # sorted ids, dtype-stable shuffle
    order = np.arange(len(idx_all))
    rng = np.random.default_rng(seed + 1000 * k + j)

    ckpt_cfg = {"stage": "fieldres", "variant": variant, "outer_k": k,
                "inner_j": j, "epochs": epochs, "batch": batch, "lr": lr,
                "in_dim": in_dim, "n_train_rows": n_tr,
                "pretrain_ckpt": shared["pre_path"],
                "f2_source": shared["f2_source"], "seed": seed}
    # In-progress checkpoints go to a .part path: out_path itself exists
    # ONLY when the head is fully trained, so the exists-means-complete skip
    # above stays truthful across embers preemptions (review finding: a
    # half-trained head saved AT out_path would be mistaken for done).
    part_path = out_path + ".part"
    last_save = time.time()
    for ep in range(epochs):
        head.train()
        rng.shuffle(order)
        tot, nb = 0.0, 0
        for lo in range(0, len(order), batch):
            sel = order[lo:lo + batch]
            xb = Xt[sel].to(device)
            rb = rt[sel].to(device)
            wb = wt[sel].to(device)
            optimizer.zero_grad(set_to_none=True)
            mu, log_s2 = head(xb)
            nll = 0.5 * (log_s2 + (rb - mu) ** 2 / torch.exp(log_s2))
            loss = (wb * nll).sum() / wb.sum().clamp_min(1e-9)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
            tot += float(loss.item())
            nb += 1
            if time.time() - last_save > CKPT_EVERY_SEC:
                _save_ckpt(torch, part_path, head, optimizer, scheduler,
                           None, ep - 1, ckpt_cfg, [k, j])
                last_save = time.time()
        scheduler.step()
        if ep == 0 or ep == epochs - 1:
            _say(f"fieldres f{k}_{j} epoch {ep + 1}/{epochs} "
                 f"wNLL={tot / max(nb, 1):.4f} (n={n_tr:,})")
        _save_ckpt(torch, part_path, head, optimizer, scheduler, None, ep,
                   ckpt_cfg, [k, j])
        last_save = time.time()
    os.replace(part_path, out_path)
    return out_path


# ── OOF prediction ──────────────────────────────────────────────────────────

def predict_oof(frame, folds, ckpts, cfg=None, shared=None):
    """Assemble oof_tier3.npz {oof_r, sigma, avail, pattern_id}.

    Every row's prediction comes only from heads that never saw it:
      * rows in outer fold k0: mean of the 4 inner heads of system k0 (all
        trained with fold k0 excluded) — the fold-assignment epistemic
        ensemble; sigma^2 = mean per-head sigma^2 + spread of the means.
      * always-train rows (outer == -1, i.e. PA rows): for each outer
        system k, the head M_{k, inner_fold[k][row]} held this row out —
        mean/spread over those honest OOF predictions.

    avail=0 (and oof_r=NaN — contract #5's hard 100-km alpha=0 zone) where:
    the stack has no date/pixel for the row, nbr_pacal_avail_100km == 0,
    the row is vault (unit or period), or no honest head covers it.
    """
    cfg = cfg or {}
    torch, _m, _t, _v1 = _require_torch()
    if shared is None:
        shared = _build_shared(cfg, frame=frame, folds=folds)
    frame = shared["frame"]
    folds = shared["folds"]
    n = len(frame)
    variant = shared["variant"]
    device = "cpu"

    fold_ckpts = {}
    for key, path in (ckpts or {}).items():
        if key == "pretrain":
            continue
        kk, jj = (int(key[0]), int(key[1])) if isinstance(key, tuple) \
            else (int(str(key).split(",")[0]), int(str(key).split(",")[1]))
        fold_ckpts[(kk, jj)] = path
    if not fold_ckpts:
        raise SystemExit("[aqnet2] predict_oof: no fine-tune checkpoints "
                         "given — run fieldres first")

    in_dim = shared["X"].shape[1]
    Xt = torch.from_numpy(shared["X"])
    heads = {}
    for (kk, jj), path in sorted(fold_ckpts.items()):
        ck = _load_ckpt(torch, path, device)
        head = _build_inr_head(torch, in_dim)
        head.load_state_dict(ck["model"])
        head.eval()
        heads[(kk, jj)] = head

    # One forward per head over all rows (cheap MLP), then per-row honest
    # selection.
    mus, sigs = {}, {}
    with torch.no_grad():
        for key, head in heads.items():
            out_mu = np.empty(n, dtype=np.float64)
            out_s = np.empty(n, dtype=np.float64)
            for lo in range(0, n, 65536):
                mu, log_s2 = head(Xt[lo:lo + 65536])
                out_mu[lo:lo + 65536] = mu.numpy()
                out_s[lo:lo + 65536] = np.exp(0.5 * log_s2.numpy())
            mus[key], sigs[key] = out_mu, out_s

    ks = sorted({k for k, _j in heads})
    outer, inner = shared["outer"], shared["inner"]
    oof_r = np.full(n, np.nan)
    sigma = np.full(n, np.nan)
    got = np.zeros(n, dtype=bool)

    def _combine(rows, keys_of_row):
        for i in rows:
            keys = keys_of_row(i)
            if not keys:
                continue
            m = np.array([mus[key][i] for key in keys])
            s = np.array([sigs[key][i] for key in keys])
            oof_r[i] = m.mean()
            sigma[i] = float(np.sqrt(np.mean(s ** 2) + np.var(m)))
            got[i] = True

    for k0 in ks:
        keys_k0 = [key for key in heads if key[0] == k0]
        rows = np.flatnonzero(outer == k0)
        _combine(rows, lambda i, kk=tuple(keys_k0): list(kk))
    rows_free = np.flatnonzero(outer < 0)
    _combine(rows_free, lambda i: [
        (k, int(inner[k][i])) for k in ks
        if k in inner and inner[k][i] >= 0 and (k, int(inner[k][i])) in heads])

    avail = shared["ok"] & got & np.isfinite(oof_r) & ~shared["vault"]
    col = "nbr_pacal_avail_100km"
    if col in frame.columns:
        a100 = pd.to_numeric(frame[col], errors="coerce").to_numpy(np.float64)
        avail &= np.isfinite(a100) & (a100 > 0)
    else:
        _say(f"WARNING: {col} missing — 100-km hard-zero zone cannot be "
             f"applied from the frame; marking ALL rows unavailable is the "
             f"only safe default")
        avail &= False
    oof_r[~avail] = np.nan
    sigma[~avail] = np.nan
    if avail.any() and not np.isfinite(oof_r[avail]).all():
        raise AssertionError("non-finite oof_r on avail rows post-mask")

    pattern_id = _coverage_bins(frame)
    out_path = _oof_path(variant)
    tmp = out_path + ".tmp.npz"
    np.savez_compressed(tmp,
                        oof_r=oof_r.astype(np.float64),
                        sigma=sigma.astype(np.float64),
                        avail=avail.astype(np.uint8),
                        pattern_id=pattern_id.astype(np.int8))
    os.replace(tmp, out_path)
    meta = {"variant": variant, "n_rows": int(n),
            "n_avail": int(avail.sum()),
            "avail_frac": float(avail.mean()),
            "f2_source": shared["f2_source"],
            "pretrain_ckpt": shared["pre_path"],
            "heads": sorted(f"f{k}_{j}" for k, j in heads),
            "pattern_counts": {str(b): int((pattern_id == b).sum())
                               for b in (0, 1, 2)}}
    meta_path = config2.artifact(
        f"oof_tier3{'_temporal' if variant == 'temporal' else ''}_meta.json")
    tmp = meta_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    os.replace(tmp, meta_path)
    _say(f"fieldres: oof -> {out_path} ({int(avail.sum()):,}/{n:,} avail)")
    return {"oof_r": oof_r, "sigma": sigma, "avail": avail,
            "pattern_id": pattern_id, "path": out_path}


# ── Stage runners (sentinel + FORCE, v2 CLI idiom) ──────────────────────────

def run_fieldpre(cfg):
    _banner("fieldpre")
    _last, _best, state_path = _pretrain_paths(cfg.get("variant"))
    if os.path.exists(state_path) and os.environ.get("FORCE") != "1":
        _say(f"{state_path} exists (FORCE=1 to re-run) -- skip")
        return 0
    t0 = time.time()
    pretrain(cfg)
    _say(f"── stage fieldpre done in {time.time() - t0:.1f}s")
    return 0


def run_fieldres(cfg):
    _banner("fieldres")
    out_path = _oof_path(cfg.get("variant"))
    if os.path.exists(out_path) and os.environ.get("FORCE") != "1":
        _say(f"{out_path} exists (FORCE=1 to re-run) -- skip")
        return 0
    t0 = time.time()
    shared = _build_shared(cfg)
    quick = bool(cfg.get("quick"))
    n_outer = 2 if quick else int(config2.OUTER_N_FOLDS)
    n_inner = int(config2.INNER_N_FOLDS)
    ckpts = {"pretrain": shared["pre_path"]}
    for k in range(n_outer):
        for j in range(n_inner):
            ckpts[(k, j)] = finetune(cfg, (k, j), shared=shared)
    predict_oof(None, None, ckpts, cfg=cfg, shared=shared)
    _say(f"── stage fieldres done in {time.time() - t0:.1f}s")
    return 0


def run_predict(cfg):
    _banner("fieldres-predict")
    variant = _variant_tag(cfg.get("variant"))
    ckpts = {}
    n_outer = 2 if cfg.get("quick") else int(config2.OUTER_N_FOLDS)
    for k in range(n_outer):
        for j in range(int(config2.INNER_N_FOLDS)):
            p = _finetune_path(variant, k, j)
            if os.path.exists(p):
                ckpts[(k, j)] = p
    if not ckpts:
        raise SystemExit("[aqnet2] no fieldres checkpoints found — run "
                         "fieldres first")
    predict_oof(None, None, ckpts, cfg=cfg)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="AQNet v2 T3 masked field pretrain + INR residual "
                    "decoder")
    ap.add_argument("stage", choices=["fieldpre", "fieldres", "predict"])
    ap.add_argument("--quick", action="store_true",
                    help="3-month window, 2 outer folds, 2 epochs")
    ap.add_argument("--resume", action="store_true",
                    help="resume pretraining from last.pt")
    ap.add_argument("--variant", default=None, choices=["temporal"],
                    help="temporally-pure (< cutoff) variant")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--base-width", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--grid-deg", type=float, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--stack", default=None,
                    help="explicit stack v2 npz cache path")
    ap.add_argument("--pretrain-ckpt", default=None,
                    help="explicit pretrain checkpoint (fieldres/predict)")
    args = ap.parse_args(argv)

    cfg = {"quick": args.quick, "resume": args.resume,
           "variant": args.variant, "epochs": args.epochs,
           "batch_size": args.batch_size, "base_width": args.base_width,
           "lr": args.lr, "grid_deg": args.grid_deg, "device": args.device,
           "stack_path": args.stack, "pretrain_ckpt": args.pretrain_ckpt,
           "seed": config2.SEED}
    if args.stage == "fieldpre":
        return run_fieldpre(cfg)
    if args.stage == "fieldres":
        return run_fieldres(cfg)
    return run_predict(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
