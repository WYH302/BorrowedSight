"""Pure-PyTorch occupancy losses (Lovasz-Softmax + asymmetric UFE safety penalty) and D4 BEV augmentation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from borrowedsight.labels import IGNORE_INDEX


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp(min=1.0)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_masked(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Multi-class Lovasz-Softmax loss (Berman et al., CVPR 2018) over present observed voxels."""
    n_class = int(logits.shape[1])
    probs = F.softmax(logits.float(), dim=1)
    # Flatten spatial dims [B, C, X, Y, Z] -> [N, C]
    probs_flat = probs.permute(0, 2, 3, 4, 1).reshape(-1, n_class)
    tgt_flat = targets.reshape(-1)
    valid = tgt_flat != int(ignore_index)
    if not bool(valid.any().item()):
        return logits.new_zeros((), dtype=torch.float32)
    probs_v = probs_flat[valid]
    tgt_v = tgt_flat[valid]
    losses = []
    for c in range(n_class):
        fg = (tgt_v == c).float()
        if float(fg.sum().item()) == 0.0:
            continue
        errors = (fg - probs_v[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    if not losses:
        return logits.new_zeros((), dtype=torch.float32)
    return torch.stack(losses).mean()


def asymmetric_ufe_loss(tri_logits: torch.Tensor, tri_targets: torch.Tensor) -> torch.Tensor:
    """Directly penalize False-Free predictions (class 1) on ground-truth Unknown (class 0) voxels."""
    tri_prob = F.softmax(tri_logits.float(), dim=1)
    unk = tri_targets == 0
    if not bool(unk.any().item()):
        return tri_logits.new_zeros((), dtype=torch.float32)
    p_free_on_unk = tri_prob[:, 1][unk]
    return (-torch.log((1.0 - p_free_on_unk).clamp(min=1.0e-5))).mean()


def bev_d4_augment(
    feat: torch.Tensor,
    extra: torch.Tensor,
    sem_t: torch.Tensor,
    tri_t: torch.Tensor,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact 8-way D4 symmetry (flips + 90-deg rotations) on ego BEV (X, Y) plane."""
    # feat: [B, C, X, Y, Z], extra: [B, E, X, Y], sem_t/tri_t: [B, X, Y, Z]
    flip_x = bool(rng.integers(0, 2))
    flip_y = bool(rng.integers(0, 2))
    rot_k = int(rng.integers(0, 4))
    if flip_x:
        feat = torch.flip(feat, dims=(2,))
        extra = torch.flip(extra, dims=(2,))
        sem_t = torch.flip(sem_t, dims=(1,))
        tri_t = torch.flip(tri_t, dims=(1,))
    if flip_y:
        feat = torch.flip(feat, dims=(3,))
        extra = torch.flip(extra, dims=(3,))
        sem_t = torch.flip(sem_t, dims=(2,))
        tri_t = torch.flip(tri_t, dims=(2,))
    if rot_k:
        feat = torch.rot90(feat, k=rot_k, dims=(2, 3))
        extra = torch.rot90(extra, k=rot_k, dims=(2, 3))
        sem_t = torch.rot90(sem_t, k=rot_k, dims=(1, 2))
        tri_t = torch.rot90(tri_t, k=rot_k, dims=(1, 2))
    return feat.contiguous(), extra.contiguous(), sem_t.contiguous(), tri_t.contiguous()
