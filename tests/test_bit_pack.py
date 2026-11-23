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


def test_vq_gradients_and_stage2_round_trip():
    import torch
    from borrowedsight.model import BorrowedSight

    torch.manual_seed(0)
    ego_pose = np.zeros(6, dtype=np.float32)
    for stage in (1, 2):
        model = BorrowedSight("B4", feat_dim=16, codebook_size=32, default_stage=stage)
        model.train()
        cached = {
            "sender_id": 1,
            "pose": ego_pose,
            "timestamp": 0.1,
            "speed": 1.0,
            "yaw_rate": 0.0,
            "coords": torch.tensor([[50, 50, 3], [52, 51, 2], [48, 49, 4]], dtype=torch.long),
            "feats": torch.randn(3, 16),
            "geo": torch.randn(3, 6),
            "pred_class": torch.tensor([8, 5, 1], dtype=torch.long),
            "envelope": np.ones((20, 20), dtype=np.uint8),
        }
        budget = packet_layout("B4", FAIR_BUDGET_BYTES, stage=stage, include_envelope=True)
        sent = model.transmit(cached, ego_pose, receiver_id=0, n_tokens=budget["n_tokens"], stage=stage, budget=budget)
        assert len(sent["blob"]) == FAIR_BUDGET_BYTES
        loss = sent["recon"].pow(2).mean() + sent["vq_loss"]
        loss.backward()
        fuse_grad = sum(p.grad.abs().sum().item() for p in model.fuse.parameters() if p.grad is not None)
        assert fuse_grad > 0.0, f"stage={stage} starved self.fuse of gradients"

        model.eval()
        with torch.no_grad():
            sent_eval = model.transmit(cached, ego_pose, receiver_id=0, n_tokens=budget["n_tokens"], stage=stage, budget=budget)
            placed_rx = model._placed_from_blob(sent_eval["blob"], torch.as_tensor(ego_pose), ego_time=0.1)
            # Token order on wire is Morton-sorted; compare sorted sets.
            tx_vals = sent_eval["recon"].detach().cpu().numpy()
            rx_vals = placed_rx["values"].detach().cpu().numpy()
            assert rx_vals.shape == tx_vals.shape
            for row in tx_vals:
                dists = np.linalg.norm(rx_vals - row[None, :], axis=1)
                assert float(dists.min()) < 1e-4, f"stage={stage} wire lookup mismatch: {dists.min()}"


def test_ablation_switches_match_exact_byte_budget():
    import torch
    from borrowedsight.model import BorrowedSight

    ego_pose = np.zeros(6, dtype=np.float32)
    cached = {
        "sender_id": 2,
        "pose": ego_pose,
        "timestamp": 0.2,
        "speed": 2.0,
        "yaw_rate": 0.0,
        "coords": torch.stack([torch.arange(40) + 20, torch.arange(40) + 20, torch.arange(40) % 8], dim=1).long(),
        "feats": torch.randn(40, 16),
        "geo": torch.randn(40, 6),
        "pred_class": torch.tensor([(i % 12) + 1 for i in range(40)], dtype=torch.long),
        "envelope": np.ones((20, 20), dtype=np.uint8),
    }
    variants = [
        ("full", dict(use_envelope=True, use_grouped_books=True, use_quota=True, default_stage=1)),
        ("no_envelope", dict(use_envelope=False, use_grouped_books=True, use_quota=True, default_stage=1)),
        ("shared_book", dict(use_envelope=True, use_grouped_books=False, use_quota=True, default_stage=1)),
        ("no_quota", dict(use_envelope=True, use_grouped_books=True, use_quota=False, default_stage=1)),
        ("stage2", dict(use_envelope=True, use_grouped_books=True, use_quota=True, default_stage=2)),
    ]
    for target in (FAIR_BUDGET_BYTES, TARGET_BYTES):
        for name, kw in variants:
            m = BorrowedSight("B4", feat_dim=16, codebook_size=32, **kw)
            m.eval()
            budget = packet_layout("B4", target, stage=kw["default_stage"], include_envelope=kw["use_envelope"])
            with torch.no_grad():
                sent = m.transmit(cached, ego_pose, receiver_id=0, n_tokens=budget["n_tokens"], budget=budget)
                assert len(sent["blob"]) == target, (name, target, len(sent["blob"]))
                out = m(torch.as_tensor(ego_pose).unsqueeze(0), torch.tensor([0.2]), torch.zeros(1, 2), [[sent["blob"]]])
                assert out["sem"].shape == (1, 13, 100, 100, 8)
                assert out["tri"].shape == (1, 3, 100, 100, 8)


def test_envelope_25x25_zstd_fits_exact_512b_slot_and_roundtrips():
    import torch
    from borrowedsight.model import BorrowedSight
    from borrowedsight.packer import coarse_from_tristate

    rng = np.random.default_rng(123)
    tri = np.zeros((100, 100, 8), dtype=np.uint8)
    tri[15:85, 20:80, 1:5] = 1
    tri[30:36, 40:44, 2:5] = 2
    tri[60:64, 55:60, 1:4] = 2
    env25 = coarse_from_tristate(tri, cells=25)
    assert env25.shape == (25, 25)
    # Verify vectorized coarse_from_tristate matches elementwise reference
    env20 = coarse_from_tristate(tri, cells=20)
    assert env20.shape == (20, 20)
    assert int(env25[8, 10] & 2) == 2  # occupied at x=32..35, y=40..43

    ego_pose = np.zeros(6, dtype=np.float32)
    cached = {
        "sender_id": 2,
        "pose": ego_pose,
        "timestamp": 0.2,
        "speed": 2.0,
        "yaw_rate": 0.0,
        "coords": torch.stack([torch.arange(32) + 20, torch.arange(32) + 20, torch.arange(32) % 8], dim=1).long(),
        "feats": torch.randn(32, 16),
        "geo": torch.randn(32, 6),
        "pred_class": torch.tensor([(i % 12) + 1 for i in range(32)], dtype=torch.long),
        "envelope": env20,
        "envelope25": env25,
    }
    m = BorrowedSight("B4", feat_dim=16, codebook_size=32, envelope_cells=25).eval()
    budget = packet_layout("B4", TARGET_BYTES, stage=1, include_envelope=True)
    with torch.no_grad():
        sent = m.transmit(cached, ego_pose, receiver_id=0, n_tokens=budget["n_tokens"], budget=budget)
        assert len(sent["blob"]) == TARGET_BYTES
        view = unpack_bytes(sent["blob"])
        assert view["envelope"].shape == (625,)
        assert np.array_equal(view["envelope"].reshape(25, 25), env25)
        out = m(torch.as_tensor(ego_pose).unsqueeze(0), torch.tensor([0.2]), torch.zeros(1, 2), [[sent["blob"]]])
        assert out["sem"].shape == (1, 13, 100, 100, 8)
        assert out["tri"].shape == (1, 3, 100, 100, 8)


def test_vectorized_quota_and_eval_vq_match_reference():
    import torch
    from borrowedsight.labels import ROAD, THINGS, VERTICAL
    from borrowedsight.messages import quota_indices
    from borrowedsight.model import BorrowedSight

    torch.manual_seed(42)
    n = 1200
    scores = torch.randn(n)
    classes = torch.randint(1, 13, (n,))
    # Force heavy road dominance (70% road)
    classes[:840] = ROAD
    chosen = quota_indices(scores, classes, k=400, road_cap=0.45, thing_floor=0.15, vertical_floor=0.15)
    assert chosen.numel() <= 400
    sel_cls = classes[chosen]
    road_frac = float((sel_cls == ROAD).float().mean().item())
    assert road_frac <= 0.45 + 1e-6, f"road_frac={road_frac} exceeded cap"
    n_things = int(torch.isin(sel_cls, torch.as_tensor(THINGS)).sum().item())
    n_vert = int(torch.isin(sel_cls, torch.as_tensor(VERTICAL)).sum().item())
    avail_things = int(torch.isin(classes, torch.as_tensor(THINGS)).sum().item())
    avail_vert = int(torch.isin(classes, torch.as_tensor(VERTICAL)).sum().item())
    assert n_things >= min(int(400 * 0.15), avail_things)
    assert n_vert >= min(int(400 * 0.15), avail_vert)

    # Check eval fast-path quantization matches training ResidualSimVQ path exactly
    m = BorrowedSight("B4", feat_dim=16, codebook_size=32)
    fused = torch.randn(64, 16)
    cls_sub = torch.randint(1, 13, (64,))
    for st in (1, 2):
        m.train()
        recon_ref, codes_ref, _ = m._quantize_grouped(fused, cls_sub, st)
        m.eval()
        with torch.no_grad():
            recon_fast, codes_fast, _ = m._quantize_grouped(fused, cls_sub, st)
            lookup_fast = m._lookup_grouped(codes_fast, cls_sub, st)
        assert torch.equal(codes_ref, codes_fast), f"stage {st} codes mismatch"
        assert torch.allclose(recon_ref.detach(), recon_fast, atol=1e-5)
        assert torch.allclose(recon_fast, lookup_fast, atol=1e-5)
