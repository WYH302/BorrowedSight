"""Real pack_bytes / unpack_bytes round trip and matched byte budgets."""

import numpy as np

from borrowedsight.codec import decode_tokens, demorton3, encode_tokens
from borrowedsight.packer import (
    BLOCK_TOKENS,
    FAIR_BUDGET_BYTES,
    HEADER_SIZE,
    PAD_BYTE,
    TARGET_BYTES,
    budget_for,
    drop_block,
    morton3,
    pack_bytes,
    packet_layout,
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


def _short_payload(method: str, n: int, rng) -> dict:
    if method == "B1":
        return {"feat": rng.normal(size=(n, 16)).astype(np.float16)}
    if method == "B2":
        return {"feat": rng.normal(size=(n, 8)).astype(np.float16)}
    if method == "B3":
        return {
            "code": rng.integers(0, 32, size=n, dtype=np.uint8),
            "conf": rng.integers(0, 255, size=n, dtype=np.uint8),
        }
    return {
        "cls": rng.integers(1, 13, size=n, dtype=np.uint8),
        "code": rng.integers(0, 32, size=n, dtype=np.uint8),
        "conf": rng.integers(1, 255, size=n, dtype=np.uint8),
        "evidence": np.full(n, 2, dtype=np.uint8),
    }


def test_short_sender_pads_to_one_budget_and_does_not_shift():
    """Fewer cells than the budget still round-trips, and B1–B4 share one length."""
    target = FAIR_BUDGET_BYTES
    n_cells = 100
    rng = np.random.default_rng(11)
    lengths = {}
    pad_morton = int.from_bytes(bytes([PAD_BYTE]) * 4, "little")
    for method in ("B1", "B2", "B3", "B4"):
        layout = packet_layout(method, target)
        assert n_cells < layout["n_tokens"]
        assert layout["tail"] < layout["block_size"]
        coords = np.stack(
            (
                np.arange(n_cells) % 100,
                (np.arange(n_cells) * 3) % 90,
                np.arange(n_cells) % 8,
            ),
            axis=1,
        ).astype(np.int64)
        raw = encode_tokens(method, coords, _short_payload(method, n_cells, rng), stage=1)
        envelope = np.arange(400, dtype=np.uint8).reshape(20, 20) if method == "B4" else None
        pose = np.array([1.0, -2.0, 1.6, 0.0, 15.0, 0.0], dtype=np.float32)
        blob = pack_bytes(
            method,
            sender_id=3,
            receiver_id=0,
            timestamp_s=1.25,
            pose=pose,
            speed=2.5,
            yaw_rate=-0.1,
            tokens=raw,
            envelope=envelope,
            stage=1,
            pad_slots=layout["n_tokens"],
            tail_bytes=layout["tail"],
        )
        assert len(blob) == target
        view = unpack_bytes(blob)
        assert np.array_equal(view["tokens"], raw)
        assert view["pad_tokens"] == layout["n_tokens"] - n_cells
        assert view["pad_bytes"] == bytes([PAD_BYTE]) * layout["tail"]
        decoded = decode_tokens(method, view["tokens"], stage=1)
        assert decoded["morton"].shape[0] == n_cells
        assert pad_morton not in {int(value) for value in decoded["morton"]}
        if method == "B4":
            assert np.array_equal(view["envelope"].reshape(20, 20), envelope)
            assert np.array_equal(decoded["evidence"], np.full(n_cells, 2, dtype=np.uint8))
        width = layout["token_bytes"]
        block_bytes = 8 + BLOCK_TOKENS * width
        env = 512 if method == "B4" else 0
        last = layout["n_blocks"] - 1
        start = HEADER_SIZE + env + last * block_bytes
        assert int.from_bytes(blob[start + 4 : start + 6], "little") == 0
        assert blob[start + 8 : start + block_bytes] == bytes([PAD_BYTE]) * (BLOCK_TOKENS * width)
        damaged = drop_block(blob, 1)
        kept = unpack_bytes(damaged)
        expect = np.concatenate([raw[:BLOCK_TOKENS], raw[2 * BLOCK_TOKENS :]], axis=0)
        assert np.array_equal(kept["tokens"], expect)
        assert kept["pad_bytes"] == view["pad_bytes"]
        assert np.array_equal(unpack_bytes(drop_block(blob, last))["tokens"], raw)
        lengths[method] = len(blob)
    values = list(lengths.values())
    mean = float(np.mean(values))
    assert max(abs(left - right) for left in values for right in values) / mean < 0.02
    layout = packet_layout("B3", target)
    empty = np.zeros((0, layout["token_bytes"]), dtype=np.uint8)
    blob = pack_bytes(
        "B3",
        1,
        0,
        0.0,
        np.zeros(6),
        0.0,
        0.0,
        empty,
        pad_slots=layout["n_tokens"],
        tail_bytes=layout["tail"],
    )
    assert len(blob) == target
    view = unpack_bytes(blob)
    assert view["tokens"].shape == (0, layout["token_bytes"])
    assert view["pad_tokens"] == layout["n_tokens"]
    assert decode_tokens("B3", view["tokens"])["code"].shape[0] == 0


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
