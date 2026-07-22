"""Export a predicted PM2.5 surface from a trained FusionUNet checkpoint.

Loads a checkpoint written by train.py, rebuilds the gridded input stack for
the requested date range with the SAME grid, fill values, and normalization
statistics used at training time (all stored in the checkpoint, so there is
no train/serve skew), runs inference, and writes:

  <out>.npz    pm25 (days x H x W, ug/m3), lat, lon, dates
               (+ per-source attention maps with --attention)
  <out>.json   manifest: grid definition, date range, channel groups, provenance

Run:
    python research/deeplearning/export_surface.py \
        --checkpoint research/deeplearning/checkpoints/fusion_unet_best.pt \
        --start 2025-06-01 --end 2025-06-30 \
        --out research/deeplearning/surfaces/pm25_june2025
"""
import os
import json
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from dataset import build_dataset, load_cache, fill_missing, apply_norm_stats
from models import FusionUNet


# ── Input assembly ──────────────────────────────────────────────────────────

def load_inputs(args, ckpt):
    """Gridded inputs for the requested dates, on the checkpoint's grid.

    Uses the .npz cache (sliced to the date range) when provided, otherwise
    rebuilds from the pipeline files. Verifies the grid matches the one the
    model was trained on.
    """
    if args.cache and os.path.exists(args.cache):
        print(f"Loading cache {args.cache}")
        data = load_cache(args.cache)
        keep = np.ones(len(data["dates"]), dtype=bool)
        if args.start:
            keep &= data["dates"] >= np.datetime64(pd.Timestamp(args.start))
        if args.end:
            keep &= data["dates"] <= np.datetime64(pd.Timestamp(args.end))
        idx = np.where(keep)[0]
        data["groups"] = {name: arr[idx] for name, arr in data["groups"].items()}
        data["dates"] = data["dates"][idx]
    else:
        data = build_dataset(start=args.start, end=args.end,
                             grid_deg=ckpt["grid_deg"])

    if len(data["dates"]) == 0:
        raise SystemExit("No days in the requested date range.")
    if (len(data["lat"]) != len(ckpt["lat"])
            or len(data["lon"]) != len(ckpt["lon"])
            or abs(data["grid_deg"] - ckpt["grid_deg"]) > 1e-9):
        raise SystemExit(
            f"Grid mismatch: inputs are {len(data['lat'])}x{len(data['lon'])} at "
            f"{data['grid_deg']} deg but the checkpoint was trained on "
            f"{len(ckpt['lat'])}x{len(ckpt['lon'])} at {ckpt['grid_deg']} deg.")

    # Training-time fills and normalization, straight from the checkpoint.
    fill_missing(data["groups"], fill_values=ckpt["fill_values"])
    apply_norm_stats(data["groups"], ckpt["norm_stats"])
    return data


# ── Inference ───────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, groups, device, batch_size=16, want_attention=False):
    """Run the model over all days; returns (pm25, attention or None)."""
    n_days = next(iter(groups.values())).shape[0]
    surfaces, attentions = [], []
    for s in range(0, n_days, batch_size):
        batch = {name: torch.from_numpy(arr[s:s + batch_size]).to(device)
                 for name, arr in groups.items()}
        pred, attn = model(batch)
        surfaces.append(pred.squeeze(1).cpu().numpy())
        if want_attention:
            attentions.append(attn.cpu().numpy())
    pm25 = np.concatenate(surfaces, axis=0).astype(np.float32)
    attention = (np.concatenate(attentions, axis=0).astype(np.float32)
                 if want_attention else None)
    return pm25, attention


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Export FusionUNet PM2.5 surfaces.")
    ap.add_argument("--checkpoint", required=True, help="path to a train.py checkpoint")
    ap.add_argument("--start", default=None, help="first date (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="last date (YYYY-MM-DD)")
    ap.add_argument("--out", default=os.path.join(here, "surfaces", "pm25_surface"),
                    help="output basename (writes <out>.npz and <out>.json)")
    ap.add_argument("--cache", default=None,
                    help="optional dataset .npz cache to slice instead of rebuilding")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--attention", action="store_true",
                    help="also store per-source attention maps in the .npz")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    args = ap.parse_args()

    # auto: CUDA, else Apple MPS, else CPU (matches train.py's resolution).
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available()
              else ("mps" if torch.backends.mps.is_available() else "cpu")))

    # Checkpoint holds numpy arrays and plain dicts alongside the weights.
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = FusionUNet(ckpt["group_channels"], embed_dim=ckpt["embed_dim"],
                       base_width=ckpt["base_width"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded {args.checkpoint} (epoch {ckpt.get('epoch')}) on {device}")

    data = load_inputs(args, ckpt)
    dates = data["dates"]
    print(f"Predicting {len(dates)} days "
          f"({pd.Timestamp(dates[0]).date()} .. {pd.Timestamp(dates[-1]).date()})")
    pm25, attention = predict(model, data["groups"], device,
                              batch_size=args.batch_size,
                              want_attention=args.attention)

    # ── Write .npz + manifest ──
    base = args.out[:-4] if args.out.endswith(".npz") else args.out
    os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)
    date_strs = np.array([str(pd.Timestamp(d).date()) for d in dates])
    payload = {"pm25": pm25, "lat": data["lat"], "lon": data["lon"], "dates": date_strs}
    if attention is not None:
        payload["attention"] = attention
        payload["attention_sources"] = np.array(list(ckpt["group_channels"]))
    np.savez_compressed(base + ".npz", **payload)

    manifest = {
        "product": "FusionUNet PM2.5 surface (Shared Skies deep-learning research track)",
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "date_start": str(pd.Timestamp(dates[0]).date()),
        "date_end": str(pd.Timestamp(dates[-1]).date()),
        "n_days": int(len(dates)),
        "grid_deg": float(data["grid_deg"]),
        "lat_min": float(data["lat"][0]),
        "lat_max": float(data["lat"][-1]),
        "lon_min": float(data["lon"][0]),
        "lon_max": float(data["lon"][-1]),
        "shape": [int(s) for s in pm25.shape],
        "orientation": "pm25[d, i, j] is the value at lat[i], lon[j]; "
                       "lat ascending south to north, lon ascending west to east",
        "units": "ug/m3",
        "channel_groups": ckpt["channels"],
        "attention_included": bool(attention is not None),
        "note": "Research-track output; the production live map serves the "
                "tree ensemble, not these surfaces.",
    }
    with open(base + ".json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {base}.npz ({pm25.shape[0]} days x "
          f"{pm25.shape[1]}x{pm25.shape[2]} grid) and {base}.json")


if __name__ == "__main__":
    main()
