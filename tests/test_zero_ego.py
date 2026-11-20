"""Six Zero-Ego leakage checks from the project constitution."""

import inspect

import numpy as np
import torch

from borrowedsight.codec import encode_tokens
from borrowedsight.data import RECEIVER_ID, neighbor_points_in_ego, receiver_inputs, simulate_frame
from borrowedsight.model import BorrowedSight
from borrowedsight.packer import pack_bytes


def _blob() -> bytes:
    coords = np.array([[40, 50, 2]], dtype=np.int64)
    payload = {
        "cls": np.array([8], dtype=np.uint8),
        "code": np.array([1], dtype=np.uint8),
        "conf": np.array([200], dtype=np.uint8),
        "evidence": np.array([2], dtype=np.uint8),
    }
    tokens = encode_tokens("B4", coords, payload, stage=1)
    return pack_bytes(
        "B4", 1, 0, 0.0, np.zeros(6, dtype=np.float32), 1.0, 0.0, tokens,
        envelope=np.zeros((20, 20), dtype=np.uint8),
    )


def _model(method: str) -> BorrowedSight:
    torch.manual_seed(0)
    model = BorrowedSight(method, feat_dim=16, codebook_size=32)
    model.eval()
    return model


def test_1_ego_lidar_noise_does_not_change_the_output():
    sample = simulate_frame(0, 1, n_beams=2, n_azimuth=4)
    blob = _blob()
    model = _model("B4")
    real = np.random.default_rng(0).normal(size=(12, 3)).astype(np.float32)
    noise = np.random.default_rng(1).normal(size=(40, 3)).astype(np.float32)
    out_real = model(*receiver_inputs(sample, ego_lidar=real, messages=[[blob]]))
    out_noise = model(*receiver_inputs(sample, ego_lidar=noise, messages=[[blob]]))
    assert torch.equal(out_real["sem"], out_noise["sem"])
    assert torch.equal(out_real["tri"], out_noise["tri"])


def test_2_no_ego_image_directory_is_required():
    sample = simulate_frame(0, 2, n_beams=2, n_azimuth=4)
    assert "img" not in sample and "image" not in sample and "camera" not in sample
    model = _model("B0")
    out = model(*receiver_inputs(sample, messages=[[]]))
    assert out["tri"].shape[-3:] == (100, 100, 8)


def test_3_forward_parameters_exclude_ego_sensor_features():
    names = list(inspect.signature(BorrowedSight.forward).parameters)
    assert names == ["self", "ego_pose", "timestamp", "motion", "messages"]
    banned = ("lidar", "image", "camera", "radar", "sensor", "label", "points")
    for name in names:
        lowered = name.lower()
        assert all(word not in lowered for word in banned)


def test_4_receiver_is_not_a_sender():
    sample = simulate_frame(1, 4, n_beams=2, n_azimuth=4)
    assert int(sample["receiver_id"]) == RECEIVER_ID
    assert RECEIVER_ID not in set(int(v) for v in sample["sender_ids"])
    assert len(set(int(v) for v in sample["sender_ids"])) == len(sample["sender_ids"])


def test_5_no_messages_outputs_unknown_not_a_full_map():
    model = _model("B4")
    pose = torch.zeros(1, 6)
    stamp = torch.zeros(1)
    motion = torch.zeros(1, 2)
    out = model(pose, stamp, motion, [[]])
    assert torch.all(out["tri"].argmax(dim=1) == 0)
    b0 = _model("B0")
    out_b0 = b0(pose, stamp, motion, [[_blob()]])
    assert torch.all(out_b0["tri"].argmax(dim=1) == 0)


def test_6_visualized_input_points_are_only_neighbors():
    sample = simulate_frame(0, 5, n_beams=4, n_azimuth=8)
    ego_only = np.array([[1.25, -2.5, 0.25], [40.0, 40.0, 40.0]], dtype=np.float64)
    seen = neighbor_points_in_ego(sample, ego_lidar=ego_only)
    seen_again = neighbor_points_in_ego(sample, ego_lidar=np.zeros((3, 3)))
    assert np.allclose(seen, seen_again)
    if seen.size:
        match = np.all(np.isclose(seen[:, None, :], ego_only[None, :, :], atol=1e-6), axis=-1)
        assert not np.any(match)
