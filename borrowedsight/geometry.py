"""Per-voxel geometric statistics used by the quantizer input."""

from __future__ import annotations

import torch

from borrowedsight.grid import GRID_SIZE, INTERVALS, linear_index, voxel_index_torch


def geo_stats(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return coords [M,3], stats [M,6], and the linear index [M] for occupied voxels.

    The six numbers are log-count, mean z, z range, linearity, planarity, scattering.
    """
    device = points.device
    empty_c = torch.zeros((0, 3), dtype=torch.long, device=device)
    empty_g = torch.zeros((0, 6), dtype=torch.float32, device=device)
    empty_l = torch.zeros((0,), dtype=torch.long, device=device)
    if points.numel() == 0:
        return empty_c, empty_g, empty_l
    idx, valid = voxel_index_torch(points[:, :3])
    if not bool(valid.any()):
        return empty_c, empty_g, empty_l
    pts = points[valid, :3].to(torch.float32)
    idx = idx[valid]
    lin = linear_index(idx)
    uniq, inverse = torch.unique(lin, return_inverse=True)
    m = int(uniq.shape[0])
    ones = torch.ones(pts.shape[0], dtype=torch.float32, device=device)
    count = torch.zeros(m, dtype=torch.float32, device=device).index_add(0, inverse, ones)
    sum_xyz = torch.zeros(m, 3, dtype=torch.float32, device=device).index_add(0, inverse, pts)
    mean = sum_xyz / count.clamp(min=1).unsqueeze(1)
    delta = pts - mean[inverse]
    outer = torch.stack(
        (
            delta[:, 0] * delta[:, 0],
            delta[:, 1] * delta[:, 1],
            delta[:, 2] * delta[:, 2],
            delta[:, 0] * delta[:, 1],
            delta[:, 0] * delta[:, 2],
            delta[:, 1] * delta[:, 2],
        ),
        dim=1,
    )
    cov_flat = torch.zeros(m, 6, dtype=torch.float32, device=device).index_add(0, inverse, outer)
    cov_flat = cov_flat / count.clamp(min=1).unsqueeze(1)
    cov = pts.new_zeros(m, 3, 3)
    cov[:, 0, 0] = cov_flat[:, 0]
    cov[:, 1, 1] = cov_flat[:, 1]
    cov[:, 2, 2] = cov_flat[:, 2]
    cov[:, 0, 1] = cov_flat[:, 3]
    cov[:, 1, 0] = cov_flat[:, 3]
    cov[:, 0, 2] = cov_flat[:, 4]
    cov[:, 2, 0] = cov_flat[:, 4]
    cov[:, 1, 2] = cov_flat[:, 5]
    cov[:, 2, 1] = cov_flat[:, 5]
    evals = torch.linalg.eigvalsh(cov).clamp(min=0)
    e3, e2, e1 = evals[:, 0], evals[:, 1], evals[:, 2]
    denom = e1.clamp(min=1.0e-8)
    linearity = (e1 - e2) / denom
    planarity = (e2 - e3) / denom
    scattering = e3 / denom
    few = count < 3
    linearity = torch.where(few, torch.zeros_like(linearity), linearity)
    planarity = torch.where(few, torch.zeros_like(planarity), planarity)
    scattering = torch.where(few, torch.zeros_like(scattering), scattering)
    z_mean = (mean[:, 2] - float(INTERVALS[2] * 0 + (-2.5))) / 3.4
    z_lo = torch.full((m,), 1.0e9, device=device)
    z_hi = torch.full((m,), -1.0e9, device=device)
    z_lo.scatter_reduce_(0, inverse, pts[:, 2], reduce="amin", include_self=True)
    z_hi.scatter_reduce_(0, inverse, pts[:, 2], reduce="amax", include_self=True)
    z_range = (z_hi - z_lo) / float(INTERVALS[2])
    stats = torch.stack(
        (
            torch.log1p(count) / torch.log1p(torch.tensor(64.0, device=device)),
            z_mean.clamp(-1, 2),
            z_range.clamp(0, 8),
            linearity.clamp(0, 1),
            planarity.clamp(0, 1),
            scattering.clamp(0, 1),
        ),
        dim=1,
    )
    gx, gy, gz = GRID_SIZE
    coords = torch.stack((uniq // (gy * gz), (uniq // gz) % gy, uniq % gz), dim=1)
    return coords.to(torch.long), stats.to(torch.float32), uniq.to(torch.long)
