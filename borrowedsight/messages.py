"""Token selection and ego-frame scatter. No ego sensor data enters here."""

from __future__ import annotations

import numpy as np
import torch

from borrowedsight.codec import decode_tokens, demorton3, encode_tokens
from borrowedsight.grid import GRID_SIZE, linear_index, transform_points_torch, voxel_centers_torch, voxel_index_torch
from borrowedsight.labels import ROAD, THINGS, VERTICAL
from borrowedsight.packer import pack_bytes

GX, GY, GZ = GRID_SIZE


def quota_indices(
    scores: torch.Tensor,
    classes: torch.Tensor,
    k: int,
    road_cap: float = 0.45,
    thing_floor: float = 0.15,
    vertical_floor: float = 0.15,
) -> torch.Tensor:
    """Hard thing / vertical floors and a road cap. Returns sender-voxel rows."""
    count = int(scores.shape[0])
    if count == 0 or k <= 0:
        return torch.zeros(0, dtype=torch.long, device=scores.device)
    things = torch.isin(classes, torch.as_tensor(THINGS, device=classes.device))
    vertical = torch.isin(classes, torch.as_tensor(VERTICAL, device=classes.device))
    road = classes == ROAD
    keep = torch.zeros(count, dtype=torch.bool, device=scores.device)

    def take(mask: torch.Tensor, limit: int) -> None:
        if limit <= 0:
            return
        free = torch.logical_and(mask, torch.logical_not(keep))
        idx = torch.nonzero(free, as_tuple=False).flatten()
        if idx.numel() == 0:
            return
        n_take = min(int(limit), int(idx.numel()))
        chosen = idx[torch.topk(scores[idx], n_take).indices]
        keep[chosen] = True

    take(things, int(k * thing_floor))
    take(vertical, int(k * vertical_floor))
    non_road = torch.logical_not(road)
    take(non_road, k - int(keep.sum().item()))
    slots = k - int(keep.sum().item())
    take(road, slots)
    chosen = torch.nonzero(keep, as_tuple=False).flatten()
    if chosen.numel() > 0:
        road_sel = road[chosen]
        n_other = int(torch.logical_not(road_sel).sum().item())
        # road / (road + other) <= cap  =>  road <= other * cap / (1 - cap)
        cap = float(road_cap)
        max_road = int(n_other * cap / max(1e-6, 1.0 - cap))
        road_idx = chosen[road_sel]
        if int(road_idx.numel()) > max_road:
            if max_road <= 0:
                keep[road_idx] = False
            else:
                best = road_idx[torch.topk(scores[road_idx], max_road).indices]
                keep[road_idx] = False
                keep[best] = True
    return torch.nonzero(keep, as_tuple=False).flatten()


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
