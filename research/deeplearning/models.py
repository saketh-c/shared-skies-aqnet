"""Model architectures for the deep-learning PM2.5 research track.

Three building blocks:

  UNet                    encoder-decoder with skip connections (depth 4),
                          GroupNorm (stable at the small batch sizes daily
                          grids allow), bilinear upsampling, and a softplus
                          output head so predicted PM2.5 is non-negative.
  SpatialAttentionFusion  fuses the source groups (aerosol, smoke,
                          meteorology, static, temporal) with per-pixel
                          per-source softmax attention plus squeeze-excite
                          channel attention. The attention maps are returned
                          from forward() for interpretability — they show
                          which source the model trusts at each pixel.
  FusionUNet              SpatialAttentionFusion -> UNet -> continuous
                          PM2.5 surface.

Inputs are dicts of tensors keyed by group name, each (B, C_g, H, W) on the
same lat/lon grid. Arbitrary H/W are handled by internal padding to a
multiple of 16 (2^depth), cropped back before returning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Shared layers ───────────────────────────────────────────────────────────

def _group_norm(channels):
    """GroupNorm with the largest group count <= 8 that divides `channels`."""
    groups = 8
    while groups > 1 and channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class DoubleConv(nn.Module):
    """Two 3x3 convolutions, each followed by GroupNorm and SiLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            _group_norm(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            _group_norm(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Bilinear upsample -> 1x1 channel reduction -> concat skip -> DoubleConv."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.reduce(self.up(x))
        return self.conv(torch.cat([skip, x], dim=1))


# ── UNet ────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """Standard U-Net, depth 4, for dense regression on gridded inputs.

    Parameters
    ----------
    in_channels : int
        Number of input channels (for FusionUNet this is the fusion embed dim).
    base_width : int
        Channel width of the first encoder stage; doubles at each depth.
    out_channels : int
        Number of output channels (1 for a PM2.5 surface).

    forward(x) with x (B, in_channels, H, W) returns (B, out_channels, H, W),
    non-negative via softplus.
    """

    DEPTH = 4

    def __init__(self, in_channels, base_width=32, out_channels=1):
        super().__init__()
        b = base_width
        self.enc1 = DoubleConv(in_channels, b)
        self.enc2 = DoubleConv(b, 2 * b)
        self.enc3 = DoubleConv(2 * b, 4 * b)
        self.enc4 = DoubleConv(4 * b, 8 * b)
        self.bottleneck = DoubleConv(8 * b, 16 * b)
        self.pool = nn.MaxPool2d(2)
        self.up4 = UpBlock(16 * b, 8 * b, 8 * b)
        self.up3 = UpBlock(8 * b, 4 * b, 4 * b)
        self.up2 = UpBlock(4 * b, 2 * b, 2 * b)
        self.up1 = UpBlock(2 * b, b, b)
        self.head = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x):
        h, w = x.shape[-2:]
        mult = 2 ** self.DEPTH
        pad_h = (mult - h % mult) % mult
        pad_w = (mult - w % mult) % mult
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        z = self.bottleneck(self.pool(e4))

        d4 = self.up4(z, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        out = F.softplus(self.head(d1))
        return out[..., :h, :w]


# ── SpatialAttentionFusion ──────────────────────────────────────────────────

class SpatialAttentionFusion(nn.Module):
    """Fuse several source groups of channels with per-pixel attention.

    Each source group (e.g. aerosol, smoke, meteorology, static, temporal)
    passes through its own small conv encoder into a shared embedding space.
    A per-source score head produces a per-pixel logit; a softmax ACROSS
    SOURCES turns those into attention weights, and the fused embedding is
    the attention-weighted sum of source embeddings. A squeeze-excite block
    then re-weights the fused embedding channels globally.

    Parameters
    ----------
    group_channels : dict[str, int]
        Ordered mapping of source-group name -> number of input channels.
    embed_dim : int
        Width of the shared embedding space.
    se_reduction : int
        Reduction ratio of the squeeze-excite bottleneck.

    forward(groups) with groups a dict of (B, C_g, H, W) tensors returns
    (fused, attention) where fused is (B, embed_dim, H, W) and attention is
    (B, n_sources, H, W) with weights summing to 1 across sources.
    """

    def __init__(self, group_channels, embed_dim=32, se_reduction=4):
        super().__init__()
        self.group_names = list(group_channels)
        self.embed_dim = embed_dim
        self.encoders = nn.ModuleDict()
        self.score_heads = nn.ModuleDict()
        for name, in_ch in group_channels.items():
            self.encoders[name] = nn.Sequential(
                nn.Conv2d(in_ch, embed_dim, kernel_size=3, padding=1),
                _group_norm(embed_dim),
                nn.SiLU(inplace=True),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                _group_norm(embed_dim),
                nn.SiLU(inplace=True),
            )
            self.score_heads[name] = nn.Conv2d(embed_dim, 1, kernel_size=3, padding=1)
        hidden = max(embed_dim // se_reduction, 4)
        self.se = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, groups):
        embeddings, scores = [], []
        for name in self.group_names:
            e = self.encoders[name](groups[name])
            embeddings.append(e)
            scores.append(self.score_heads[name](e))

        attention = torch.softmax(torch.cat(scores, dim=1), dim=1)  # (B, S, H, W)
        stacked = torch.stack(embeddings, dim=1)                    # (B, S, E, H, W)
        fused = (attention.unsqueeze(2) * stacked).sum(dim=1)       # (B, E, H, W)

        # Squeeze-excite channel attention on the fused embedding.
        gate = self.se(fused.mean(dim=(2, 3)))                      # (B, E)
        fused = fused * gate[:, :, None, None]
        return fused, attention


# ── FusionUNet ──────────────────────────────────────────────────────────────

class FusionUNet(nn.Module):
    """SpatialAttentionFusion -> UNet -> continuous PM2.5 surface.

    Parameters
    ----------
    group_channels : dict[str, int]
        Ordered mapping of source-group name -> number of input channels.
    embed_dim : int
        Fusion embedding width (also the UNet input width).
    base_width : int
        UNet base channel width.

    forward(groups) returns (surface, attention): surface (B, 1, H, W)
    non-negative PM2.5, attention (B, n_sources, H, W) per-pixel source
    weights for interpretability.
    """

    def __init__(self, group_channels, embed_dim=32, base_width=32):
        super().__init__()
        self.fusion = SpatialAttentionFusion(group_channels, embed_dim=embed_dim)
        self.unet = UNet(in_channels=embed_dim, base_width=base_width)

    def forward(self, groups):
        fused, attention = self.fusion(groups)
        surface = self.unet(fused)
        return surface, attention
