"""Coordinate convention copied from CoHFF / OpenCOOD."""

import ast
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from borrowedsight.grid import (
    GRID_SIZE,
    INTERVALS,
    MAX_BOUND,
    MIN_BOUND,
    transform_points,
    voxel_index,
    x_to_world,
)
from borrowedsight.labels import RAW_TO_MAPPED, load_cohff_name_table, official_label_yaml
from borrowedsight.metrics import IGNORE_INDEX, semantic_target
from borrowedsight.paths import COHFF_ROOT
from borrowedsight.weather import load_weather


def test_ego_origin_is_zero():
    pose = np.array([12.0, -4.0, 1.6, 0.0, 25.0, 1.0])
    out = transform_points(np.array([[0.0, 0.0, 0.0]]), pose, pose)
    assert np.allclose(out, 0.0, atol=1e-8)


def test_same_static_point_from_two_vehicles_matches():
    world = np.array([[5.0, 2.0, 0.4]])
    identity = np.zeros(6)
    ego = np.array([0.0, 0.0, 1.6, 0.0, 0.0, 0.0])
    other = np.array([4.0, -2.0, 1.6, 0.0, 15.0, 0.0])
    in_other = transform_points(world, identity, other)
    back = transform_points(in_other, other, ego)
    direct = transform_points(world, identity, ego)
    assert np.allclose(back, direct, atol=1e-6)


def test_yaw_sign_matches_cohff():
    # Yaw +90 degrees takes vehicle +x to world +y.
    source = np.array([0.0, 0.0, 0.0, 0.0, 90.0, 0.0])
    identity = np.zeros(6)
    out = transform_points(np.array([[1.0, 0.0, 0.0]]), source, identity)
    assert np.allclose(out[0], [0.0, 1.0, 0.0], atol=1e-6)


def test_meters_are_not_mixed_with_centimeters_or_a_second_grid():
    interval = (20.0 - (-20.0)) / (100 - 1)
    expected = math.floor((10.0 - (-20.0)) / interval)
    assert expected == 74
    idx, valid = voxel_index(np.array([[10.0, 0.0, 0.0]]))
    assert bool(valid[0])
    assert int(idx[0, 0]) == expected
    # 10 cm would land in a different cell if meters and centimeters were mixed.
    idx_cm, valid_cm = voxel_index(np.array([[0.10, 0.0, 0.0]]))
    assert bool(valid_cm[0])
    assert int(idx_cm[0, 0]) != expected
    assert GRID_SIZE == (100, 100, 8)
    assert np.allclose(MAX_BOUND, [20.0, 20.0, 0.9])
    assert np.allclose(MIN_BOUND, [-20.0, -20.0, -2.5])
    assert np.allclose(INTERVALS[0], interval)


def test_rotation_order_is_roll_yaw_pitch_degrees():
    # A pure yaw of 90 degrees is not a pure roll. Degrees, not radians:
    # 90 radians would be a different matrix.
    yaw90 = x_to_world([0, 0, 0, 0, 90, 0])
    roll90 = x_to_world([0, 0, 0, 90, 0, 0])
    yaw90_rad_as_deg = x_to_world([0, 0, 0, 0, math.degrees(90.0), 0])
    assert not np.allclose(yaw90, roll90, atol=1e-6)
    assert np.allclose(yaw90[:3, :3] @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-6)
    # Feeding radians as if they were degrees does not match the degree matrix.
    as_if_radians = x_to_world([0, 0, 0, 0, 90.0 * math.pi / 180.0 * 180.0 / math.pi, 0])
    assert np.allclose(as_if_radians, yaw90, atol=1e-6)
    assert not np.allclose(yaw90_rad_as_deg, x_to_world([0, 0, 0, 0, 1.0, 0]), atol=1e-5)


def test_frame_timestamp_matches_the_stored_clock():
    from borrowedsight.data import simulate_frame

    frame = simulate_frame(0, 6, n_beams=2, n_azimuth=4)
    assert frame["timestamp"] == pytest.approx(0.6)
    assert frame["frame"] == 6


def test_unlabeled_zero_is_not_observed_free():
    sem = np.zeros((100, 100, 8), dtype=np.uint8)
    unknown = np.zeros_like(sem)
    target = semantic_target(sem, unknown)
    assert np.all(target == IGNORE_INDEX)
    occupied = unknown.copy()
    occupied[10, 10, 2] = 2
    occupied_target = semantic_target(sem, occupied)
    assert int(occupied_target[10, 10, 2]) == IGNORE_INDEX
    free = unknown.copy()
    free[10, 10, 2] = 1
    free_target = semantic_target(sem, free)
    assert int(free_target[10, 10, 2]) == 0


def test_grid_and_names_match_the_cohff_checkout():
    task = COHFF_ROOT / "cohff_opv2v" / "config" / "train_config" / "task_config.yml"
    if not task.exists():
        pytest.skip("refs/CoHFF is not present")
    with task.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    assert cfg["vox_range"] == [20, 20, 0.9, -20, -20, -2.5]
    assert cfg["h"] == 100 and cfg["w"] == 100 and cfg["z"] == 8
    official = load_cohff_name_table(official_label_yaml())
    vendored = load_cohff_name_table()
    assert official == vendored
    assert official[0] == "Unlabeled"
    assert official[5] == "Road"
    assert official[8] == "Vehicles"
    mapping_path = COHFF_ROOT / "cohff_opv2v" / "config" / "label_config" / "opv2v_label_mapping.py"
    module = ast.parse(mapping_path.read_text(encoding="utf-8"))
    found = None
    for node in module.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "opv2v_label_mapping_dict" for t in node.targets):
            found = ast.literal_eval(node.value)
    assert found == RAW_TO_MAPPED


def test_lisa_weather_parameters_are_disjoint_from_training():
    cfg = load_weather()
    assert "alpha_per_m" not in cfg["train"]
    assert cfg["test_fog"]["alpha_per_m"] != cfg["train"]["drop_prob"]
    assert cfg["test_rain"]["rainfall_mm_h"] > 0


def test_envelope_warps_from_sender_to_ego_frame():
    import torch
    from borrowedsight.model import _paint_envelope

    ego_pose = np.array([0.0, 0.0, 1.6, 0.0, 0.0, 0.0], dtype=np.float32)
    # Sender is at (x=6m, y=0m) rotated +90 deg yaw (its +x points along ego +y).
    sender_pose = np.array([6.0, 0.0, 1.6, 0.0, 90.0, 0.0], dtype=np.float32)
    # Mark coarse cell (cx=15, cy=10) => sender x in [+10..+12m], y in [0..+2m] as FREE (bit 0).
    # In EGO frame, sender point (11, 1, 0) transforms to (6 - 1, 0 + 11, 0) = (5m, 11m, 0).
    env = np.full((20, 20), 4, dtype=np.uint8)  # Unknown everywhere else
    env[15, 10] = 1  # Observed Free only
    extra = torch.zeros(7, 100, 100, dtype=torch.float32)
    extra[2] = 1.0
    free_ego = _paint_envelope(extra, env, sender_pose=sender_pose, ego_pose=ego_pose)

    idx_hit, valid_hit = voxel_index(np.array([[5.0, 11.0, 0.0]]))
    assert bool(valid_hit[0])
    ix, iy = int(idx_hit[0, 0]), int(idx_hit[0, 1])
    assert float(free_ego[ix, iy]) == 1.0
    assert float(extra[0, ix, iy]) == 1.0
    assert float(extra[2, ix, iy]) == 0.0

    # The UN-transformed sender-local cell (x=11m, y=1m) must NOT be falsely marked free in EGO!
    idx_wrong, _ = voxel_index(np.array([[11.0, 1.0, 0.0]]))
    wx, wy = int(idx_wrong[0, 0]), int(idx_wrong[0, 1])
    assert float(free_ego[wx, wy]) == 0.0
    assert float(extra[2, wx, wy]) == 1.0

    # Also verify 25x25 (1.6 m) envelope warps identically from sender to EGO!
    # Cell (cx=19, cy=13) in 25x25 covers sender x in [10.4..12.0m], y in [0.8..2.4m]; center=(11.2, 1.6)
    # In EGO frame (sender at (6, 0), +90 deg yaw): (6 - 1.6, 0 + 11.2) = (4.4m, 11.2m).
    env25 = np.full((25, 25), 4, dtype=np.uint8)
    env25[19, 13] = 1
    extra25 = torch.zeros(7, 100, 100, dtype=torch.float32)
    extra25[2] = 1.0
    free25 = _paint_envelope(extra25, env25, sender_pose=sender_pose, ego_pose=ego_pose)
    idx25, valid25 = voxel_index(np.array([[4.4, 11.2, 0.0]]))
    assert bool(valid25[0])
    ix25, iy25 = int(idx25[0, 0]), int(idx25[0, 1])
    assert float(free25[ix25, iy25]) == 1.0
    assert float(extra25[0, ix25, iy25]) == 1.0
    assert float(extra25[2, ix25, iy25]) == 0.0


def test_sender_origin_is_sensor_not_grid_corner():
    import torch
    from borrowedsight.model import BorrowedSight

    model = BorrowedSight("B4", feat_dim=16, codebook_size=32)
    ego_pose = np.array([10.0, 5.0, 1.6, 0.0, 0.0, 0.0], dtype=np.float32)
    sender_pose = np.array([16.0, 2.0, 1.6, 0.0, 45.0, 0.0], dtype=np.float32)
    cached = {
        "sender_id": 1,
        "pose": sender_pose,
        "timestamp": 0.5,
        "speed": 2.0,
        "yaw_rate": 0.0,
        "coords": torch.tensor([[50, 50, 3]], dtype=torch.long),
        "feats": torch.randn(1, 16),
        "geo": torch.randn(1, 6),
        "pred_class": torch.tensor([8], dtype=torch.long),
        "envelope": np.zeros((20, 20), dtype=np.uint8),
    }
    sent = model.transmit(cached, ego_pose, receiver_id=0, n_tokens=16)
    placed_tx = model.placed_from_transmit(sent, ego_pose, ego_time=0.5)
    placed_rx = model._placed_from_blob(sent["blob"], torch.as_tensor(ego_pose), ego_time=0.5)
    expected = np.array([6.0, -3.0, 0.0], dtype=np.float32)
    assert np.allclose(placed_tx["origin_ego"].detach().cpu().numpy(), expected, atol=1e-4)
    assert np.allclose(placed_rx["origin_ego"].detach().cpu().numpy(), expected, atol=1e-4)
