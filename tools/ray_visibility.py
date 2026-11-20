"""GPU tri-state visibility from sender lidar rays.

0 Unknown, 1 Observed Free, 2 Occupied.
Samples lie on the segment from the sensor origin to each return, at a step
finer than one CoHFF voxel, so an axis-aligned ray marks every voxel it
crosses. The endpoint is Occupied. Samples only mark a voxel the segment
actually enters. A missed corner stays Unknown, which is the safe error.
"""

from __future__ import annotations

import math

import torch

from borrowedsight.grid import GRID_SIZE, INTERVALS, MAX_BOUND, MIN_BOUND, voxel_index_torch

UNKNOWN = 0
FREE = 1
OCCUPIED = 2
_CELLS = GRID_SIZE[0] * GRID_SIZE[1] * GRID_SIZE[2]
_STEP_CACHE: dict = {}
_BOUND_CACHE: dict = {}


def _as_origin(origin: torch.Tensor, n: int) -> torch.Tensor:
    if origin.ndim == 1:
        return origin.reshape(1, 3).expand(n, 3)
    if origin.shape[0] != n:
        raise ValueError(f"origin batch {origin.shape[0]} does not match {n} rays")
    return origin


def ray_visibility(endpoints: torch.Tensor, origin: torch.Tensor) -> torch.Tensor:
    """Return a uint8 [100,100,8] tri-state grid for one sweep."""
    device = endpoints.device
    grid = torch.zeros(_CELLS, dtype=torch.int16, device=device)
    if endpoints.numel() == 0:
        return grid.view(GRID_SIZE).to(torch.uint8)
    ends = endpoints.reshape(-1, 3).to(dtype=torch.float32)
    if origin.ndim == 1:
        origins = origin.to(device=device, dtype=torch.float32).reshape(1, 3).expand(ends.shape[0], 3)
    else:
        origins = origin.to(device=device, dtype=torch.float32)
    vec = ends - origins
    # 96 samples. An 18 m frame is stepped at about 0.19 m, under one voxel.
    t = _steps(device)
    pos = origins[:, None, :] + vec[:, None, :] * t[None, :, None]
    free_lin = _fast_linear(pos.reshape(-1, 3))
    free_lin = free_lin[free_lin >= 0]
    if free_lin.numel() > 0:
        grid.scatter_reduce_(0, free_lin, torch.ones(free_lin.shape[0], dtype=torch.int16, device=device), reduce="amax", include_self=True)
    occ_lin = _fast_linear(ends)
    occ_lin = occ_lin[occ_lin >= 0]
    if occ_lin.numel() > 0:
        grid.scatter_reduce_(0, occ_lin, torch.full((occ_lin.shape[0],), 2, dtype=torch.int16, device=device), reduce="amax", include_self=True)
    return grid.view(GRID_SIZE).to(torch.uint8)


def union_tristate(grids: list[torch.Tensor]) -> torch.Tensor:
    if not grids:
        return torch.zeros(GRID_SIZE, dtype=torch.uint8)
    acc = grids[0].to(torch.int16)
    for other in grids[1:]:
        acc = torch.maximum(acc, other.to(torch.int16))
    return acc.to(torch.uint8)


def _steps(device: torch.device) -> torch.Tensor:
    key = str(device)
    cached = _STEP_CACHE.get(key)
    if cached is None:
        cached = torch.linspace(0.0, 1.0, 64, device=device, dtype=torch.float32)[:-1]
        _STEP_CACHE[key] = cached
    return cached


def _bounds(device: torch.device):
    key = str(device)
    cached = _BOUND_CACHE.get(key)
    if cached is None:
        cached = (
            torch.as_tensor(MIN_BOUND, dtype=torch.float32, device=device),
            torch.as_tensor(MAX_BOUND, dtype=torch.float32, device=device),
            torch.as_tensor(INTERVALS, dtype=torch.float32, device=device),
            torch.tensor(GRID_SIZE, dtype=torch.float32, device=device),
        )
        _BOUND_CACHE[key] = cached
    return cached


def _fast_linear(points: torch.Tensor) -> torch.Tensor:
    """Float32 CoHFF index. Invalid points are -1."""
    min_b, max_b, intervals, grid = _bounds(points.device)
    pts = points.to(dtype=torch.float32)
    idx = torch.floor((pts - min_b) / intervals)
    valid = (pts > min_b).all(-1)
    valid = valid & (pts < max_b).all(-1)
    valid = valid & (idx >= 0).all(-1)
    valid = valid & (idx < grid).all(-1)
    gx, gy, gz = GRID_SIZE
    lin = idx[..., 0].to(torch.long) * (gy * gz) + idx[..., 1].to(torch.long) * gz + idx[..., 2].to(torch.long)
    return torch.where(valid, lin, torch.full_like(lin, -1))


def _flat(idx: torch.Tensor) -> torch.Tensor:
    _gx, gy, gz = GRID_SIZE
    return idx[:, 0] * (gy * gz) + idx[:, 1] * gz + idx[:, 2]


def np_min_interval() -> float:
    return float(min(INTERVALS.tolist()))


def main() -> None:
    import time

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elev = torch.linspace(-0.25, 0.1, 48, device=device)
    az = torch.linspace(0, 2 * math.pi, 360, device=device)
    ee, aa = torch.meshgrid(elev, az, indexing="ij")
    dirs = torch.stack((torch.cos(ee) * torch.cos(aa), torch.cos(ee) * torch.sin(aa), torch.sin(ee)), dim=-1)
    ends = dirs.reshape(-1, 3) * 18.0
    origin = torch.zeros(3, device=device)
    for _ in range(5):
        ray_visibility(ends, origin)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    repeats = 10
    for _ in range(repeats):
        ray_visibility(ends, origin)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000.0 / repeats
    print(f"device={device} rays={ends.shape[0]} ms={dt_ms:.3f}")


if __name__ == "__main__":
    main()
