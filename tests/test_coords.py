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
