"""Token selection and ego-frame scatter. No ego sensor data enters here."""

from __future__ import annotations

import numpy as np
import torch

from borrowedsight.codec import decode_tokens, demorton3, encode_tokens
from borrowedsight.grid import GRID_SIZE, linear_index, transform_points_torch, voxel_centers_torch, voxel_index_torch
from borrowedsight.labels import N_CLASS, ROAD, THINGS, VERTICAL
from borrowedsight.packer import pack_bytes

GX, GY, GZ = GRID_SIZE

# 0=other_non_road, 1=thing, 2=vertical, 3=road
_QUOTA_LUT_CPU = torch.zeros(N_CLASS, dtype=torch.int8)
for _c in THINGS:
    _QUOTA_LUT_CPU[_c] = 1
for _c in VERTICAL:
    _QUOTA_LUT_CPU[_c] = 2
_QUOTA_LUT_CPU[ROAD] = 3
_QUOTA_LUT_CACHE: dict[str, torch.Tensor] = {}


def _quota_lut(device: torch.device) -> torch.Tensor:
    key = str(device)
    cached = _QUOTA_LUT_CACHE.get(key)
    if cached is None:
        cached = _QUOTA_LUT_CPU.to(device)
        _QUOTA_LUT_CACHE[key] = cached
    return cached


def quota_indices(
    scores: torch.Tensor,
    classes: torch.Tensor,
    k: int,
    road_cap: float = 0.45,
    thing_floor: float = 0.15,
    vertical_floor: float = 0.15,
) -> torch.Tensor:
    """Hard thing / vertical floors and a road cap (zero-sync GPU vectorized).

    Returns sender-voxel rows satisfying:
    1) Up to floor(k * thing_floor) highest-scoring Things tokens;
    2) Up to floor(k * vertical_floor) highest-scoring Vertical tokens;
    3) Remaining budget filled by highest-scoring non-Road tokens;
    4) Any remaining slots filled by Road tokens capped at road / (road + other) <= road_cap.
    """
    count = int(scores.shape[0])
    if count == 0 or k <= 0:
        return torch.zeros(0, dtype=torch.long, device=scores.device)
    k_int = int(k)
    order = torch.argsort(scores, descending=True)
    cls_sorted = classes[order].clamp(0, N_CLASS - 1)
    g = _quota_lut(scores.device)[cls_sorted]

    t_lim = int(k_int * thing_floor)
    v_lim = int(k_int * vertical_floor)

    is_t = g == 1
    keep_t = is_t & (torch.cumsum(is_t, dim=0) <= t_lim) if t_lim > 0 else torch.zeros_like(is_t)
    is_v = g == 2
    keep_v = is_v & (torch.cumsum(is_v, dim=0) <= v_lim) if v_lim > 0 else torch.zeros_like(is_v)

    floor_keep = keep_t | keep_v
    rem_nr = (g != 3) & (~floor_keep)
    nr_slots = (k_int - floor_keep.sum()).clamp(min=0)
    extra_nr = rem_nr & (torch.cumsum(rem_nr, dim=0) <= nr_slots)

    nr_keep = floor_keep | extra_nr
    n_other = nr_keep.sum()
    slots = (k_int - n_other).clamp(min=0)
    cap = float(road_cap)
    cap_ratio = cap / max(1e-6, 1.0 - cap)
    max_road = torch.floor(n_other.to(torch.float64) * cap_ratio).to(torch.long)
    road_limit = torch.minimum(slots, max_road)

    is_r = g == 3
    road_keep = is_r & (torch.cumsum(is_r, dim=0) <= road_limit)
    keep_sorted = nr_keep | road_keep
    return order[keep_sorted]


def topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    if scores.numel() == 0 or k <= 0:
        return torch.zeros(0, dtype=torch.long, device=scores.device)
    n_take = min(int(k), int(scores.shape[0]))
    return torch.topk(scores, n_take).indices


def scatter_values(feat: torch.Tensor, weight: torch.Tensor, idx: torch.Tensor, values: torch.Tensor, conf: torch.Tensor) -> None:
    """Add token features into feat [C,X,Y,Z] and weight [X,Y,Z]."""
    if idx.numel() == 0:
        return
    lin = linear_index(idx)
    weighted = values * conf.unsqueeze(1)
    acc = feat.new_zeros(feat.shape[0], GX * GY * GZ)
    acc.index_add_(1, lin, weighted.transpose(0, 1))
    wacc = feat.new_zeros(GX * GY * GZ)
    wacc.index_add_(0, lin, conf)
    feat.add_(acc.view(feat.shape[0], GX, GY, GZ))
    weight.add_(wacc.view(GX, GY, GZ))


def coords_to_ego(coords: torch.Tensor, sender_pose, ego_pose):
    centers = voxel_centers_torch(coords)
    return transform_points_torch(centers, sender_pose, ego_pose)


def encode_packet(
    method: str,
    sender_id: int,
    receiver_id: int,
    timestamp_s: float,
    pose,
    speed: float,
    yaw_rate: float,
    coords: np.ndarray,
    payload: dict,
    envelope: np.ndarray | None,
    stage: int = 1,
    pad_slots: int | None = None,
    tail_bytes: int = 0,
    include_envelope: bool | None = None,
) -> bytes:
    tokens = encode_tokens(method, coords, payload, stage=stage)
    order = np.argsort(decode_sort_key(tokens), kind="mergesort") if tokens.shape[0] else np.zeros(0, dtype=np.int64)
    tokens = tokens[order]
    return pack_bytes(
        method,
        sender_id,
        receiver_id,
        timestamp_s,
        pose,
        speed,
        yaw_rate,
        tokens,
        envelope=envelope,
        stage=stage,
        pad_slots=pad_slots,
        tail_bytes=tail_bytes,
        include_envelope=include_envelope,
    )


def decode_sort_key(tokens: np.ndarray) -> np.ndarray:
    from borrowedsight.codec import _read_morton

    if tokens.shape[0] == 0:
        return np.zeros(0, dtype=np.uint32)
    return _read_morton(tokens)


def packet_coords(tokens_morton: np.ndarray) -> np.ndarray:
    if tokens_morton.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    return demorton3(tokens_morton)
