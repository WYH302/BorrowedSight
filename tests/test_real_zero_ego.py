"""Strict 6-item Zero-Ego leakage audit directly on REAL OPV2V + Semantic-OPV2V data."""
from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from borrowedsight.data import receiver_inputs
from borrowedsight.model import BorrowedSight
from borrowedsight.opv2v_data import RealOPV2VDataset, SEM_VOX_ROOT
from borrowedsight.packer import TARGET_BYTES, packet_layout
from tools.extract_cache_real import encode_one_sender


def _real_sample():
    if not (SEM_VOX_ROOT / "validate").is_dir() or not list((SEM_VOX_ROOT / "validate").glob("*.npz")):
        pytest.skip("Real Semantic-OPV2V validate voxels not on disk")
    ds = RealOPV2VDataset("validate", stride=10)
    return ds.__getitem__(0, device=torch.device("cpu"), load_points=True)


def _model_and_blobs(sample: dict):
    torch.manual_seed(0)
    model = BorrowedSight("B4", feat_dim=16, codebook_size=32)
    model.eval()
    budget = packet_layout("B4", TARGET_BYTES, stage=1, include_envelope=True)
    blobs = []
    with torch.no_grad():
        for slot in range(len(sample["sender_ids"])):
            row = encode_one_sender(
                model,
                sample["points"][slot],
                sender_id=int(sample["sender_ids"][slot]),
                pose=sample["sender_pose"][slot],
                timestamp=sample["timestamp"],
                speed=float(sample["sender_speeds"][slot]),
                device=torch.device("cpu"),
            )
            row_t = dict(row)
            for k in ("coords", "feats", "geo", "pred_class"):
                row_t[k] = torch.from_numpy(row[k]).to(torch.long if k in ("coords", "pred_class") else torch.float32)
            sent = model.transmit(row_t, sample["ego_pose"], int(sample["receiver_id"]), budget["n_tokens"], budget=budget)
            blobs.append(sent["blob"])
    return model, blobs


def test_real_1_ego_lidar_replaced_with_noise_output_identical():
    sample = _real_sample()
    model, blobs = _model_and_blobs(sample)
    rng = np.random.default_rng(42)
    ego_real_like = rng.normal(size=(5000, 3)).astype(np.float32)
    ego_noise = rng.uniform(-50, 50, size=(8000, 3)).astype(np.float32)
    with torch.no_grad():
        out_a = model(*receiver_inputs(sample, ego_lidar=ego_real_like, messages=[blobs]))
        out_b = model(*receiver_inputs(sample, ego_lidar=ego_noise, messages=[blobs]))
    assert torch.equal(out_a["sem"], out_b["sem"])
    assert torch.equal(out_a["tri"], out_b["tri"])


def test_real_2_no_ego_camera_images_required():
    sample = _real_sample()
    for banned_key in ("camera", "img", "image", "ego_lidar", "ego_points"):
        assert all(banned_key not in k.lower() for k in sample.keys())
    model, blobs = _model_and_blobs(sample)
    with torch.no_grad():
        out = model(*receiver_inputs(sample, messages=[blobs]))
    assert out["sem"].shape == (1, 13, 100, 100, 8)
    assert out["tri"].shape == (1, 3, 100, 100, 8)


def test_real_3_forward_signature_has_no_ego_sensor_args():
    names = list(inspect.signature(BorrowedSight.forward).parameters)
    assert names == ["self", "ego_pose", "timestamp", "motion", "messages"]


def test_real_4_receiver_id_never_in_real_sender_ids():
    sample = _real_sample()
    recv = int(sample["receiver_id"])
    senders = [int(s) for s in sample["sender_ids"]]
    assert len(senders) >= 1
    assert recv not in senders
    assert len(set(senders)) == len(senders)


def test_real_5_all_senders_off_degrades_to_pure_unknown():
    sample = _real_sample()
    model, _blobs = _model_and_blobs(sample)
    with torch.no_grad():
        out = model(*receiver_inputs(sample, messages=[[]]))
    assert torch.all(out["tri"].argmax(dim=1) == 0)


def test_real_6_neighbor_points_exclude_ego_sensor_file():
    sample = _real_sample()
    # Number of point clouds loaded equals strictly len(sender_ids), never len(sender_ids) + 1 (EGO)
    assert len(sample["points"]) == len(sample["sender_ids"])
    assert int(sample["receiver_id"]) not in set(int(s) for s in sample["sender_ids"])
