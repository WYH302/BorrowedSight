"""Real pack_bytes / unpack_bytes round trip and matched byte budgets."""

import numpy as np

from borrowedsight.codec import demorton3, encode_tokens
from borrowedsight.packer import (
    HEADER_SIZE,
    TARGET_BYTES,
    budget_for,
    drop_block,
    morton3,
    pack_bytes,
    transmit,
    unpack_bytes,
)


def _packet(method: str, n_tokens: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    spec = budget_for(method)
    width = spec["token_bytes"]
    tokens = rng.integers(0, 255, size=(n_tokens, width), dtype=np.uint8)
    envelope = None
    if method == "B4":
        envelope = rng.integers(0, 255, size=(20, 20), dtype=np.uint8)
    pose = np.array([1.0, -2.0, 1.6, 0.0, 15.0, 0.0], dtype=np.float32)
    blob = pack_bytes(
        method,
        sender_id=3,
        receiver_id=0,
        timestamp_s=1.25,
        pose=pose,
        speed=2.5,
        yaw_rate=-0.1,
        tokens=tokens,
        envelope=envelope,
        stage=1,
    )
    return blob, tokens, envelope, pose


def test_round_trip_for_every_communicating_method():
    for method in ("B1", "B2", "B3", "B4"):
        blob, tokens, envelope, pose = _packet(method, 40, seed=method.__len__())
        view = unpack_bytes(blob)
        assert view["method"] == method
        assert view["sender_id"] == 3
        assert view["receiver_id"] == 0
        assert abs(view["timestamp_s"] - 1.25) < 1e-6
        assert np.allclose(view["pose"], pose, atol=1e-6)
        assert np.array_equal(view["tokens"], tokens)
        if method == "B4":
            assert view["envelope"].shape == (400,)
            assert np.array_equal(view["envelope"].reshape(20, 20), envelope)
        else:
            assert view["envelope"].size == 0


def test_dropping_one_block_does_not_shift_the_others():
    blob, tokens, _envelope, _pose = _packet("B4", 64, seed=7)
    view = unpack_bytes(blob)
    assert view["n_blocks"] == 4
    damaged = drop_block(blob, 1)
    kept = unpack_bytes(damaged)
    expected = np.concatenate([tokens[:16], tokens[32:]], axis=0)
    assert np.array_equal(kept["tokens"], expected)
    assert kept["n_blocks"] == 3
    # The surviving blocks are still the original bytes, not a shifted parse.
    assert np.array_equal(kept["tokens"][:16], tokens[:16])
    assert np.array_equal(kept["tokens"][16:], tokens[32:])


def test_budgets_match_within_two_percent():
    sizes = {method: budget_for(method)["nbytes"] for method in ("B1", "B2", "B3", "B4")}
    assert budget_for("B0")["nbytes"] == 0
    for method, size in sizes.items():
        assert abs(size - TARGET_BYTES) / TARGET_BYTES < 0.02, (method, size)
    values = list(sizes.values())
    for left in values:
        for right in values:
            assert abs(left - right) / TARGET_BYTES < 0.02
    empty = pack_bytes("B3", 1, 0, 0.0, np.zeros(6), 0.0, 0.0, np.zeros((0, 8), np.uint8))
    assert len(empty) == HEADER_SIZE
    assert transmit(empty, "B0") == b""
    assert transmit(empty, "B3") == empty


def test_morton_round_trip_and_token_codec():
    coords = np.array([[0, 0, 0], [74, 49, 3], [99, 10, 7], [12, 80, 1]], dtype=np.int64)
    code = morton3(coords[:, 0], coords[:, 1], coords[:, 2])
    assert np.array_equal(demorton3(code), coords)
    payload = {
        "cls": np.array([8, 5, 1, 11], dtype=np.uint8),
        "code": np.array([3, 9, 0, 31], dtype=np.uint8),
        "conf": np.array([10, 20, 30, 40], dtype=np.uint8),
        "evidence": np.array([2, 2, 2, 2], dtype=np.uint8),
    }
    raw = encode_tokens("B4", coords, payload, stage=1)
    blob = pack_bytes("B4", 2, 0, 0.2, np.zeros(6), 1.0, 0.0, raw, envelope=np.zeros((20, 20), np.uint8))
    back = unpack_bytes(blob)
    assert np.array_equal(back["tokens"], raw)
