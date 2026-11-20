"""Ray tri-state: axis-aligned coverage and the 3090 timing budget."""

import math

import torch

from borrowedsight.grid import voxel_index
from tools.ray_visibility import ray_visibility


def test_axis_aligned_ray_marks_free_then_occupied_then_unknown():
    origin = torch.zeros(3)
    end = torch.tensor([[10.0, 0.0, 0.0]])
    tri = ray_visibility(end, origin)
    hit, hit_ok = voxel_index(end.numpy())
    start, start_ok = voxel_index(torch.zeros(1, 3).numpy())
    assert bool(hit_ok[0]) and bool(start_ok[0])
    x0, y0, z0 = (int(start[0, 0]), int(start[0, 1]), int(start[0, 2]))
    x1 = int(hit[0, 0])
    assert x1 > x0
    for x in range(x0, x1):
        assert int(tri[x, y0, z0]) == 1, (x, int(tri[x, y0, z0]))
    assert int(tri[x1, y0, z0]) == 2
    assert int(tri[min(x1 + 3, 99), y0, z0]) == 0


def test_one_synthetic_frame_is_under_two_milliseconds_on_cuda():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    elev = torch.linspace(-0.25, 0.1, 48, device=device)
    azim = torch.linspace(0.0, 2.0 * math.pi, 360, device=device)
    ee, aa = torch.meshgrid(elev, azim, indexing="ij")
    dirs = torch.stack((torch.cos(ee) * torch.cos(aa), torch.cos(ee) * torch.sin(aa), torch.sin(ee)), -1)
    ends = dirs.reshape(-1, 3) * 18.0
    origin = torch.zeros(3, device=device)
    for _ in range(3):
        ray_visibility(ends, origin)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        ray_visibility(ends, origin)
    stop.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(stop) / 10.0
    assert ms < 2.0, ms
