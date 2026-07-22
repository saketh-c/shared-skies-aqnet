"""Train FusionUNet on the gridded Texas stack with sparse sensor supervision.

Loss is masked MSE evaluated only at grid pixels that contain PurpleAir
sensors. Validation uses a GROUPED SITE HOLDOUT: entire sensor pixels — all
of their days — are held out together, mirroring the leave-one-sensor-out
ethos of the production ensemble. Random day splits would leak spatial
information (a site's readings on other days are highly autocorrelated), so
they are deliberately not offered.

Each epoch reports R2 / RMSE / MAE on the held-out sensor pixels and keeps
the best checkpoint by validation RMSE. Runs on CPU or GPU (Colab-friendly:
pass --device cuda or leave the default auto-detect).

Run:
    python research/deeplearning/train.py --epochs 40 --lr 1e-3 \
        --cache research/deeplearning/cache/texas_grid.npz
"""
import os
import time
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from dataset import (build_dataset, load_cache, save_cache, fill_missing,
                     compute_norm_stats, apply_norm_stats)
from models import FusionUNet


# ── Data preparation ────────────────────────────────────────────────────────

def load_or_build(args):
    """Load the .npz cache if present, otherwise build (and cache) it."""
    if args.cache and os.path.exists(args.cache):
        print(f"Loading cache {args.cache}")
        return load_cache(args.cache)
    data = build_dataset(start=args.start, end=args.end, grid_deg=args.grid_deg)
    if args.cache:
        save_cache(data, args.cache)
        print(f"Cached dataset to {args.cache}")
    return data


def split_sites(obs, n_cols, holdout_frac, seed):
    """Grouped site holdout: partition unique sensor PIXELS, not days.

    A site is a grid pixel containing at least one sensor; all sensors that
    share a pixel are held out together so no site straddles the split.
    Returns (train_keep, val_keep) boolean masks over the observation records
    and the sorted list of held-out site ids (row * n_cols + col).
    """
    site_ids = obs["row"] * n_cols + obs["col"]
    uniq = np.unique(site_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(holdout_frac * len(uniq))))
    val_sites = uniq[:n_val]
    val_keep = np.isin(site_ids, val_sites)
    return ~val_keep, val_keep, sorted(int(s) for s in val_sites)


def rasterize_targets(obs, keep, n_days, height, width):
    """Average kept observations into per-day target grids and masks.

    Multiple sensors in the same pixel on the same day are averaged.
    Returns (y, mask), both float32 (n_days, height, width); mask is 1 where
    a target exists, 0 elsewhere (y is 0 where mask is 0).
    """
    y_sum = np.zeros((n_days, height, width), dtype=np.float64)
    count = np.zeros((n_days, height, width), dtype=np.float64)
    d = obs["day"][keep]
    r = obs["row"][keep]
    c = obs["col"][keep]
    v = obs["pm25"][keep].astype(np.float64)
    np.add.at(y_sum, (d, r, c), v)
    np.add.at(count, (d, r, c), 1.0)
    has = count > 0
    y = np.zeros_like(y_sum, dtype=np.float32)
    y[has] = (y_sum[has] / count[has]).astype(np.float32)
    return y, has.astype(np.float32)


class DayDataset(Dataset):
    """One item per day: {group: (C, H, W)} inputs, target grid, mask grid."""

    def __init__(self, groups, y, mask):
        self.groups = groups
        self.y = y
        self.mask = mask

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        g = {name: torch.from_numpy(arr[i]) for name, arr in self.groups.items()}
        return g, torch.from_numpy(self.y[i]), torch.from_numpy(self.mask[i])


# ── Loss & metrics ──────────────────────────────────────────────────────────

def masked_mse(pred, y, mask):
    """MSE over masked pixels only; safe when a batch has an empty mask."""
    num = ((pred - y) ** 2 * mask).sum()
    den = mask.sum().clamp_min(1.0)
    return num / den


def regression_metrics(pred, target):
    """R2 / RMSE / MAE for flat arrays of held-out predictions."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    err = pred - target
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"r2": r2, "rmse": rmse, "mae": mae, "n": int(len(target))}


# ── Train / eval loops ──────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device):
    """One training pass; returns mean masked-MSE loss over batches."""
    model.train()
    total, n_batches = 0.0, 0
    for g, y, mask in loader:
        g = {k: v.to(device) for k, v in g.items()}
        y = y.to(device)
        mask = mask.to(device)
        pred, _ = model(g)
        loss = masked_mse(pred.squeeze(1), y, mask)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float(loss.item())
        n_batches += 1
    return total / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Collect predictions at held-out sensor pixels; return metrics dict."""
    model.eval()
    preds, targets = [], []
    for g, y, mask in loader:
        g = {k: v.to(device) for k, v in g.items()}
        pred = model(g)[0].squeeze(1).cpu().numpy()
        m = mask.numpy() > 0
        if m.any():
            preds.append(pred[m])
            targets.append(y.numpy()[m])
    if not preds:
        return None
    return regression_metrics(np.concatenate(preds), np.concatenate(targets))


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Train FusionUNet on sparse Texas PM2.5.")
    ap.add_argument("--cache", default=os.path.join(here, "cache", "texas_grid.npz"),
                    help=".npz dataset cache (built automatically if missing)")
    ap.add_argument("--checkpoint-dir", default=os.path.join(here, "checkpoints"))
    ap.add_argument("--grid-deg", type=float, default=0.1, help="grid resolution (degrees)")
    ap.add_argument("--start", default=None, help="first training date (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="last training date (YYYY-MM-DD)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--holdout-frac", type=float, default=0.2,
                    help="fraction of sensor SITES held out for validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--embed-dim", type=int, default=32, help="fusion embedding width")
    ap.add_argument("--base-width", type=int, default=32, help="UNet base channel width")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # auto: CUDA, else Apple MPS, else CPU (MPS measured ~3x CPU here).
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available()
              else ("mps" if torch.backends.mps.is_available() else "cpu")))
    print(f"Device: {device}")

    # ── Data ──
    data = load_or_build(args)
    groups = data["groups"]
    obs = data["obs"]
    n_days = len(data["dates"])
    height, width = len(data["lat"]), len(data["lon"])

    fill_values = fill_missing(groups)
    norm_stats = compute_norm_stats(groups)
    apply_norm_stats(groups, norm_stats)

    train_keep, val_keep, val_sites = split_sites(
        obs, width, args.holdout_frac, args.seed)
    n_sites = len(np.unique(obs["row"] * width + obs["col"]))
    print(f"Sites: {n_sites} total, {len(val_sites)} held out "
          f"({int(train_keep.sum()):,} train / {int(val_keep.sum()):,} val readings)")

    y_tr, m_tr = rasterize_targets(obs, train_keep, n_days, height, width)
    y_va, m_va = rasterize_targets(obs, val_keep, n_days, height, width)

    train_loader = DataLoader(
        DayDataset(groups, y_tr, m_tr), batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(
        DayDataset(groups, y_va, m_va), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers)

    # ── Model ──
    group_channels = {name: len(chs) for name, chs in data["channels"].items()}
    model = FusionUNet(group_channels, embed_dim=args.embed_dim,
                       base_width=args.base_width).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FusionUNet: {n_params:,} parameters, groups {group_channels}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── Loop ──
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_rmse = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        dt = time.time() - t0

        ckpt = {
            "model_state": model.state_dict(),
            "group_channels": group_channels,
            "channels": data["channels"],
            "embed_dim": args.embed_dim,
            "base_width": args.base_width,
            "grid_deg": data["grid_deg"],
            "lat": data["lat"],
            "lon": data["lon"],
            "norm_stats": norm_stats,
            "fill_values": fill_values,
            "val_sites": val_sites,
            "args": vars(args),
            "epoch": epoch,
            "val_metrics": metrics,
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, "fusion_unet_last.pt"))

        if metrics is None:
            print(f"epoch {epoch:3d}  loss {train_loss:.4f}  (no val obs)  {dt:.1f}s")
            continue
        marker = ""
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "fusion_unet_best.pt"))
            marker = "  *best"
        print(f"epoch {epoch:3d}  loss {train_loss:.4f}  "
              f"val R2 {metrics['r2']:.4f}  RMSE {metrics['rmse']:.3f}  "
              f"MAE {metrics['mae']:.3f}  ({metrics['n']:,} px)  {dt:.1f}s{marker}")

    print(f"\nDone. Best val RMSE {best_rmse:.3f}. "
          f"Checkpoints in {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
