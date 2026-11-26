"""Height-folded residual UNet used by every baseline."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from borrowedsight.grid import GRID_SIZE, linear_index, voxel_centers_torch, voxel_index_torch
from borrowedsight.labels import N_CLASS

GX, GY, GZ = GRID_SIZE


class ResBlock(nn.Module):
    def __init__(self, channels: int, use_norm: bool = False):
        super().__init__()
        self.use_norm = bool(use_norm)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        if self.use_norm:
            self.norm1 = nn.GroupNorm(8, channels)
            self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_norm:
            h = F.gelu(self.norm1(self.conv1(x)))
            h = self.norm2(self.conv2(h))
            return F.gelu(x + h)
        return F.gelu(x + self.conv2(F.gelu(self.conv1(x))))


def _build_2d_sincos_pe(channels: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    quarter = channels // 4
    omega = torch.exp(torch.arange(quarter, device=device, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0, device=device)) / max(1, quarter)))
    gy = torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32)
    gx = torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(gy, gx, indexing="ij")
    out_y = yy.reshape(-1, 1) * omega.reshape(1, -1)
    out_x = xx.reshape(-1, 1) * omega.reshape(1, -1)
    pe = torch.cat((torch.sin(out_y), torch.cos(out_y), torch.sin(out_x), torch.cos(out_x)), dim=-1)
    return pe.transpose(0, 1).reshape(1, channels, height, width).to(dtype=dtype)


class CoarseSDPA(nn.Module):
    def __init__(self, channels: int, heads: int = 4, use_pe: bool = False):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("SDPA channels must be divisible by heads")
        self.heads = heads
        self.use_pe = bool(use_pe)
        if self.use_pe:
            self.norm = nn.GroupNorm(8, channels)
            self._pe_cache: dict[tuple[str, int, int, torch.dtype], torch.Tensor] = {}
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        inp = x
        if self.use_pe:
            inp = self.norm(x)
            key_pe = (str(x.device), height, width, inp.dtype)
            pe = self._pe_cache.get(key_pe)
            if pe is None:
                pe = _build_2d_sincos_pe(channels, height, width, x.device, inp.dtype)
                self._pe_cache[key_pe] = pe
            inp = inp + pe
        qkv = self.qkv(inp).reshape(batch, 3, self.heads, channels // self.heads, height * width)
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
    def __init__(self, feat_dim: int = 16, use_norm_pe: bool = False, use_max_pool: bool = False):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.use_norm_pe = bool(use_norm_pe)
        self.use_max_pool = bool(use_max_pool)
        self.pool_channels = 7 if self.use_max_pool else 4
        folded = self.feat_dim * GZ
        self.in_proj = nn.Conv3d(self.pool_channels, self.feat_dim, 1)
        self.stem = nn.Conv2d(folded, 64, 3, padding=1)
        self.enc1 = ResBlock(64, use_norm=self.use_norm_pe)
        self.down1 = nn.Conv2d(64, 96, 3, stride=2, padding=1)
        self.enc2 = ResBlock(96, use_norm=self.use_norm_pe)
        self.down2 = nn.Conv2d(96, 128, 3, stride=2, padding=1)
        self.bottleneck = ResBlock(128, use_norm=self.use_norm_pe)
        self.attn = CoarseSDPA(128, heads=4, use_pe=self.use_norm_pe)
        self.up2 = nn.ConvTranspose2d(128, 96, 2, stride=2)
        self.dec2 = ResBlock(96, use_norm=self.use_norm_pe)
        self.up1 = nn.ConvTranspose2d(96, 64, 2, stride=2)
        self.dec1 = ResBlock(64, use_norm=self.use_norm_pe)
        self.out_feat = nn.Conv2d(64, folded, 1)
        self.out_sem = nn.Conv2d(64, N_CLASS * GZ, 1)

    def forward(self, points: torch.Tensor, mask: torch.Tensor) -> dict:
        pooled = self._pool(points, mask)
        return self.forward_from_pooled(pooled)

    def forward_from_pooled(self, pooled: torch.Tensor) -> dict:
        """Forward directly from pre-pooled voxel tensor [B, C_pool, GZ, GX, GY] (enables exact D4 BEV aug)."""
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
        pooled = points.new_zeros(batch, self.pool_channels, GZ, GX, GY)
        n_vox = GX * GY * GZ
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
            offsets = kept - centers
            lin = linear_index(idx)
            count = kept.new_zeros(n_vox)
            count.index_add_(0, lin, torch.ones(kept.shape[0], device=kept.device, dtype=kept.dtype))
            if not self.use_max_pool:
                feat = torch.cat(
                    (offsets, torch.ones(kept.shape[0], 1, device=kept.device, dtype=kept.dtype)),
                    dim=1,
                )
                acc = kept.new_zeros(n_vox, 4)
                acc.index_add_(0, lin, feat)
                mean = acc / count.clamp(min=1).unsqueeze(1)
                pooled[b] = mean.view(GX, GY, GZ, 4).permute(3, 2, 0, 1).contiguous()
            else:
                acc = kept.new_zeros(n_vox, 3)
                acc.index_add_(0, lin, offsets)
                mean_off = acc / count.clamp(min=1).unsqueeze(1)
                # D4-equivariant peak descriptors: max(|dx|), max(|dy|), max(dz)
                abs_xy_z = torch.stack((offsets[:, 0].abs(), offsets[:, 1].abs(), offsets[:, 2]), dim=1)
                mx = kept.new_full((n_vox, 3), -1.0e4)
                mx.scatter_reduce_(0, lin.unsqueeze(1).expand(-1, 3), abs_xy_z, reduce="amax", include_self=True)
                mx = torch.where((count > 0).unsqueeze(1), mx, torch.zeros_like(mx))
                log_den = torch.log1p(count).unsqueeze(1) / 4.0
                vox_feat = torch.cat((mean_off, mx, log_den), dim=1)
                pooled[b] = vox_feat.view(GX, GY, GZ, 7).permute(3, 2, 0, 1).contiguous()
        return pooled


class ReceiverDecoder(nn.Module):
    """Same UNet shape for every baseline. `extra` is visibility, carving, age, motion, and optional z_max.

    When `decouple_env=True` (Task-Decoupled Visibility Fusion, TDVF), the shared semantic UNet stem
    receives strictly token-derived features + carving/age/motion (preventing geometric shortcut
    overfitting on small scene counts), while the coarse Zstd visibility envelope (`free, occ, unk, z_max`)
    feeds a lightweight late-fusion refinement block exclusively for the tri-state visibility head.
    """

    def __init__(
        self,
        feat_dim: int = 16,
        extra_channels: int = 7,
        use_norm_pe: bool = False,
        decouple_env: bool = False,
        use_sem_film: bool = False,
        use_code_fill: bool = False,
        use_sarc: bool = False,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.extra_channels = int(extra_channels)
        self.use_norm_pe = bool(use_norm_pe)
        self.decouple_env = bool(decouple_env)
        self.use_sem_film = bool(use_sem_film)
        self.use_code_fill = bool(use_code_fill)
        self.use_sarc = bool(use_sarc)
        stem_extra = 4 if self.decouple_env else self.extra_channels
        if self.use_code_fill:
            # CodeFilling-style codebook-guided sparse feature completion prior to UNet decoding
            self.code_fill = nn.Sequential(
                nn.Conv2d(self.feat_dim * GZ + 1, 64, 3, padding=1),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv2d(64, 64, 3, padding=2, dilation=2),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv2d(64, self.feat_dim * GZ, 1),
            )
        self.stem = nn.Conv2d(self.feat_dim * GZ + stem_extra, 64, 3, padding=1)
        self.enc1 = ResBlock(64, use_norm=self.use_norm_pe)
        self.down1 = nn.Conv2d(64, 96, 3, stride=2, padding=1)
        self.enc2 = ResBlock(96, use_norm=self.use_norm_pe)
        self.down2 = nn.Conv2d(96, 128, 3, stride=2, padding=1)
        self.bottleneck = ResBlock(128, use_norm=self.use_norm_pe)
        self.attn = CoarseSDPA(128, heads=4, use_pe=self.use_norm_pe)
        self.up2 = nn.ConvTranspose2d(128, 96, 2, stride=2)
        self.dec2 = ResBlock(96, use_norm=self.use_norm_pe)
        self.up1 = nn.ConvTranspose2d(96, 64, 2, stride=2)
        self.dec1 = ResBlock(64, use_norm=self.use_norm_pe)
        if self.use_sem_film and self.decouple_env:
            # Zero-initialized FiLM gate injecting 25x25 z_max height prior into sem_head without early shortcutting
            self.sem_film = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.GroupNorm(8, 32),
                nn.GELU(),
                nn.Conv2d(32, 128, 1),
            )
            nn.init.zeros_(self.sem_film[-1].weight)
            nn.init.zeros_(self.sem_film[-1].bias)
        self.sem_head = nn.Conv2d(64, N_CLASS * GZ, 1)
        if self.use_sarc:
            # Semantic Anchor Residual Connection (SARC): 13-scalar bounded logit skip from B4's
            # explicit 4-bit wire class ID (0 extra wire bytes). Initialized to 4*sigmoid(0)=2.0.
            self.sarc_gate = nn.Parameter(torch.zeros(N_CLASS))
        if self.decouple_env:
            env_ch = 4 if self.extra_channels >= 8 else 3
            self.vis_refine = nn.Sequential(
                nn.Conv2d(64 + env_ch, 64, 3, padding=1),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.GroupNorm(8, 64),
                nn.GELU(),
            )
        self.tri_head = nn.Conv2d(64, 3 * GZ, 1)

    def forward(self, feat: torch.Tensor, extra: torch.Tensor, cls_prior: torch.Tensor | None = None) -> dict:
        batch = feat.shape[0]
        flat = feat.permute(0, 1, 4, 2, 3).reshape(batch, self.feat_dim * GZ, GX, GY)
        if self.use_code_fill:
            occ_mask = (flat.abs().sum(dim=1, keepdim=True) > 1e-6).to(flat.dtype)
            filled = self.code_fill(torch.cat((flat, occ_mask), dim=1))
            flat = torch.where(occ_mask > 0.5, flat, filled)
        if extra.ndim == 4:
            extra_map = extra
        elif extra.ndim == 3:
            extra_map = extra.unsqueeze(0)
        else:
            extra_map = extra.reshape(batch, self.extra_channels, 1, 1).expand(batch, self.extra_channels, GX, GY)
        if self.decouple_env:
            # Semantic UNet stem ingests strictly carving + age + ego motion (no envelope shortcut).
            stem_in = torch.cat((flat, extra_map[:, 3:7]), dim=1)
        else:
            stem_in = torch.cat((flat, extra_map), dim=1)
        skip1 = self.enc1(F.gelu(self.stem(stem_in)))
        skip2 = self.enc2(self.down1(skip1))
        coarse = self.attn(self.bottleneck(self.down2(skip2)))
        up2 = self.dec2(_match(self.up2(coarse), skip2) + skip2)
        up1 = self.dec1(_match(self.up1(up2), skip1) + skip1)
        if self.use_sem_film and self.decouple_env and self.extra_channels >= 8:
            gb = self.sem_film(extra_map[:, 7:8])
            gamma, beta = gb.chunk(2, dim=1)
            sem_in = up1 * (1.0 + 0.5 * torch.tanh(gamma)) + 0.5 * beta
        else:
            sem_in = up1
        sem = self.sem_head(sem_in).reshape(batch, N_CLASS, GZ, GX, GY).permute(0, 1, 3, 4, 2)
        if self.use_sarc and cls_prior is not None:
            if self.training:
                keep = (torch.rand((batch, 1, GX, GY, GZ), device=cls_prior.device) > 0.25).to(cls_prior.dtype)
                prior_use = cls_prior * keep
            else:
                prior_use = cls_prior
            gate = (4.0 * torch.sigmoid(self.sarc_gate)).to(sem.dtype).view(1, N_CLASS, 1, 1, 1)
            sem = sem + gate * prior_use.to(sem.dtype)
        if self.decouple_env:
            env_maps = (
                torch.cat((extra_map[:, 0:3], extra_map[:, 7:8]), dim=1)
                if self.extra_channels >= 8
                else extra_map[:, 0:3]
            )
            tri_feat = up1 + self.vis_refine(torch.cat((up1, env_maps), dim=1))
        else:
            tri_feat = up1
        tri = self.tri_head(tri_feat).reshape(batch, 3, GZ, GX, GY).permute(0, 1, 3, 4, 2)
        return {"sem": sem.contiguous(), "tri": tri.contiguous()}
