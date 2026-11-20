"""Height-folded residual UNet used by every baseline."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from borrowedsight.grid import GRID_SIZE, linear_index, voxel_centers_torch, voxel_index_torch
from borrowedsight.labels import N_CLASS

GX, GY, GZ = GRID_SIZE


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.conv2(F.gelu(self.conv1(x))))


class CoarseSDPA(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("SDPA channels must be divisible by heads")
        self.heads = heads
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        qkv = self.qkv(x).reshape(batch, 3, self.heads, channels // self.heads, height * width)
        query = qkv[:, 0].permute(0, 1, 3, 2)
        key = qkv[:, 1].permute(0, 1, 3, 2)
        value = qkv[:, 2].permute(0, 1, 3, 2)
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.permute(0, 1, 3, 2).reshape(batch, channels, height, width)
        return x + self.proj(attended)


def _match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


class SenderEncoder(nn.Module):
    def __init__(self, feat_dim: int = 16):
        super().__init__()
        self.feat_dim = int(feat_dim)
        folded = self.feat_dim * GZ
        self.in_proj = nn.Conv3d(4, self.feat_dim, 1)
        self.stem = nn.Conv2d(folded, 64, 3, padding=1)
        self.enc1 = ResBlock(64)
        self.down1 = nn.Conv2d(64, 96, 3, stride=2, padding=1)
        self.enc2 = ResBlock(96)
        self.down2 = nn.Conv2d(96, 128, 3, stride=2, padding=1)
        self.bottleneck = ResBlock(128)
        self.attn = CoarseSDPA(128, heads=4)
        self.up2 = nn.ConvTranspose2d(128, 96, 2, stride=2)
        self.dec2 = ResBlock(96)
        self.up1 = nn.ConvTranspose2d(96, 64, 2, stride=2)
        self.dec1 = ResBlock(64)
        self.out_feat = nn.Conv2d(64, folded, 1)
        self.out_sem = nn.Conv2d(64, N_CLASS * GZ, 1)

    def forward(self, points: torch.Tensor, mask: torch.Tensor) -> dict:
        pooled = self._pool(points, mask)
        hidden = self.in_proj(pooled)
        batch = hidden.shape[0]
        folded = hidden.reshape(batch, self.feat_dim * GZ, GX, GY)
        skip1 = self.enc1(F.gelu(self.stem(folded)))
        skip2 = self.enc2(self.down1(skip1))
        coarse = self.attn(self.bottleneck(self.down2(skip2)))
        up2 = self.dec2(_match(self.up2(coarse), skip2) + skip2)
        up1 = self.dec1(_match(self.up1(up2), skip1) + skip1)
        feat = self.out_feat(up1).reshape(batch, self.feat_dim, GZ, GX, GY)
        sem = self.out_sem(up1).reshape(batch, N_CLASS, GZ, GX, GY)
        # Public layout is [B, C, X, Y, Z], the same order as voxel_index.
        feat = feat.permute(0, 1, 3, 4, 2).contiguous()
        sem = sem.permute(0, 1, 3, 4, 2).contiguous()
        return {"feat": feat, "sem": sem}

    def _pool(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch = points.shape[0]
        pooled = points.new_zeros(batch, 4, GZ, GX, GY)
        for b in range(batch):
            pts = points[b][mask[b]]
            if pts.numel() == 0:
                continue
            idx, valid = voxel_index_torch(pts)
            if int(valid.sum().item()) == 0:
                continue
            idx = idx[valid]
            kept = pts[valid].to(dtype=torch.float32)
            centers = voxel_centers_torch(idx).to(dtype=torch.float32)
            feat = torch.cat(
                (kept - centers, torch.ones(kept.shape[0], 1, device=kept.device, dtype=kept.dtype)),
                dim=1,
            )
            lin = linear_index(idx)
            acc = kept.new_zeros(GX * GY * GZ, 4)
            acc.index_add_(0, lin, feat)
            count = kept.new_zeros(GX * GY * GZ)
            count.index_add_(0, lin, torch.ones(kept.shape[0], device=kept.device, dtype=kept.dtype))
            mean = acc / count.clamp(min=1).unsqueeze(1)
            pooled[b] = mean.view(GX, GY, GZ, 4).permute(3, 2, 0, 1).contiguous()
        return pooled


class ReceiverDecoder(nn.Module):
    """Same UNet shape for every baseline. `extra` is visibility, carving, age, and motion."""

    def __init__(self, feat_dim: int = 16, extra_channels: int = 7):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.extra_channels = int(extra_channels)
        self.stem = nn.Conv2d(self.feat_dim * GZ + self.extra_channels, 64, 3, padding=1)
        self.enc1 = ResBlock(64)
        self.down1 = nn.Conv2d(64, 96, 3, stride=2, padding=1)
        self.enc2 = ResBlock(96)
        self.down2 = nn.Conv2d(96, 128, 3, stride=2, padding=1)
        self.bottleneck = ResBlock(128)
        self.attn = CoarseSDPA(128, heads=4)
        self.up2 = nn.ConvTranspose2d(128, 96, 2, stride=2)
        self.dec2 = ResBlock(96)
        self.up1 = nn.ConvTranspose2d(96, 64, 2, stride=2)
        self.dec1 = ResBlock(64)
        self.sem_head = nn.Conv2d(64, N_CLASS * GZ, 1)
        self.tri_head = nn.Conv2d(64, 3 * GZ, 1)

    def forward(self, feat: torch.Tensor, extra: torch.Tensor) -> dict:
        batch = feat.shape[0]
        flat = feat.permute(0, 1, 4, 2, 3).reshape(batch, self.feat_dim * GZ, GX, GY)
        if extra.ndim == 4:
            extra_map = extra
        elif extra.ndim == 3:
            extra_map = extra.unsqueeze(0)
        else:
            extra_map = extra.reshape(batch, self.extra_channels, 1, 1).expand(batch, self.extra_channels, GX, GY)
        hidden = torch.cat((flat, extra_map), dim=1)
        skip1 = self.enc1(F.gelu(self.stem(hidden)))
        skip2 = self.enc2(self.down1(skip1))
        coarse = self.attn(self.bottleneck(self.down2(skip2)))
        up2 = self.dec2(_match(self.up2(coarse), skip2) + skip2)
        up1 = self.dec1(_match(self.up1(up2), skip1) + skip1)
        sem = self.sem_head(up1).reshape(batch, N_CLASS, GZ, GX, GY).permute(0, 1, 3, 4, 2)
        tri = self.tri_head(up1).reshape(batch, 3, GZ, GX, GY).permute(0, 1, 3, 4, 2)
        return {"sem": sem.contiguous(), "tri": tri.contiguous()}
