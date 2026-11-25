"""Morton micro-block packer.

Sixteen tokens share a fixed-size block. Each token carries an absolute voxel
Morton code, so deleting one block does not retarget the others. B4 adds a
fixed 512-byte coarse visibility slot, about half a kilobyte.

A short sender still emits the shared budget. Unused token slots and the
tail after the last block are explicit padding. Unpack returns them as
padding, not as anchors.
"""

from __future__ import annotations

import struct

import numpy as np
import zstandard

MAGIC = b"BS04"
HEADER_SIZE = 64
BLOCK_TOKENS = 16
BLOCK_HDR = 8
ENVELOPE_SLOT = 512
TARGET_BYTES = 32768
# Phase-4 serialized length. Block sizes differ, so each method's whole
# blocks land under this and a short tail makes the packets equal.
FAIR_BUDGET_BYTES = 16384
PAD_BYTE = 0xA5
COARSE = 20

TOKEN_BYTES = {"B1": 36, "B2": 20, "B3": 8, "B4": 8}
COMPACT_TOKEN_BYTES = {"B1": 36, "B2": 20, "B3": 4, "B4": 5}
METHOD_ID = {"B0": 0, "B1": 1, "B2": 2, "B3": 3, "B4": 4}
ID_METHOD = {value: key for key, value in METHOD_ID.items()}
FLAG_ENVELOPE = 1
FLAG_COMPACT = 2

_LE = chr(60)
_HEADER = struct.Struct(_LE + "4sBBHHIHBBBB6fffH10s")
_BLOCK = struct.Struct(_LE + "IH2s")
_U16B = struct.Struct(_LE + "HB")
_U16 = struct.Struct(_LE + "H")
_U32 = struct.Struct(_LE + "I")
_COMPRESSOR = zstandard.ZstdCompressor(level=3)
_COMPRESSOR_HIGH = zstandard.ZstdCompressor(level=9)
_DECOMPRESSOR = zstandard.ZstdDecompressor()

_MORTON_X = np.zeros(128, dtype=np.uint32)
_MORTON_Y = np.zeros(128, dtype=np.uint32)
_MORTON_Z = np.zeros(8, dtype=np.uint32)
for _i in range(128):
    _mx = 0
    for _bit in range(7):
        if (_i >> _bit) & 1:
            _mx |= 1 << (3 * _bit)
    _MORTON_X[_i] = _mx
    _MORTON_Y[_i] = _mx << 1
for _i in range(8):
    _mz = 0
    for _bit in range(3):
        if (_i >> _bit) & 1:
            _mz |= 1 << (3 * _bit + 2)
    _MORTON_Z[_i] = _mz


def _low(value: int, bits: int) -> int:
    return int(value) % (2 ** int(bits))


def morton3(x, y, z) -> np.ndarray:
    """Interleave 7-bit x, 7-bit y and 3-bit z."""
    xx = np.asarray(x, dtype=np.int64) & 0x7F
    yy = np.asarray(y, dtype=np.int64) & 0x7F
    zz = np.asarray(z, dtype=np.int64) & 0x07
    return (_MORTON_X[xx] | _MORTON_Y[yy] | _MORTON_Z[zz]).astype(np.uint32, copy=False)


def token_width(method: str, stage: int = 1, compact: bool = False) -> int:
    if method == "B0":
        return 8
    if compact:
        base = int(COMPACT_TOKEN_BYTES[method])
        # Compact B4 already packs both stage-1 (6b) and stage-2 (6b) inside 5 bytes (40 bits).
        if method == "B3" and int(stage) > 1:
            return base + 1
        return base
    extra = 1 if int(stage) > 1 else 0
    return int(TOKEN_BYTES[method]) + extra


def block_size(token_bytes: int) -> int:
    return BLOCK_HDR + BLOCK_TOKENS * int(token_bytes)


def _has_envelope(method: str, include_envelope: bool | None = None) -> bool:
    if include_envelope is None:
        return method == "B4"
    return bool(include_envelope)


def nbytes_for_count(
    method: str,
    n_tokens: int,
    stage: int = 1,
    include_envelope: bool | None = None,
    compact: bool = False,
) -> int:
    """Serialized size of a packet that carries exactly this many tokens."""
    if method == "B0" or int(n_tokens) <= 0:
        return 0
    width = token_width(method, stage, compact=compact)
    n_blocks = (int(n_tokens) + BLOCK_TOKENS - 1) // BLOCK_TOKENS
    env = ENVELOPE_SLOT if _has_envelope(method, include_envelope) else 0
    return HEADER_SIZE + env + n_blocks * block_size(width)


def budget_for(
    method: str,
    target: int = TARGET_BYTES,
    stage: int = 1,
    include_envelope: bool | None = None,
    compact: bool = False,
) -> dict:
    """Largest whole-block packing that does not exceed target."""
    if method == "B0":
        return {"method": method, "n_tokens": 0, "n_blocks": 0, "nbytes": 0, "token_bytes": 0, "compact": bool(compact)}
    width = token_width(method, stage, compact=compact)
    env = ENVELOPE_SLOT if _has_envelope(method, include_envelope) else 0
    overhead = HEADER_SIZE + env
    bsize = block_size(width)
    n_blocks = (int(target) - overhead) // bsize
    nbytes = overhead + n_blocks * bsize
    return {
        "method": method,
        "n_tokens": int(n_blocks * BLOCK_TOKENS),
        "n_blocks": int(n_blocks),
        "nbytes": int(nbytes),
        "token_bytes": int(width),
        "block_size": int(bsize),
        "envelope": int(env),
        "compact": bool(compact),
    }


def packet_layout(
    method: str,
    target: int = FAIR_BUDGET_BYTES,
    stage: int = 1,
    include_envelope: bool | None = None,
    compact: bool = False,
) -> dict:
    """Token count from the byte budget and this method's own token width.

    ``serialized`` is ``target`` for B1–B4. ``tail`` is padding after the last
    micro-block and is shorter than one block, so dropping a block cannot
    promote that tail into a token.
    """
    spec = dict(budget_for(method, target=int(target), stage=stage, include_envelope=include_envelope, compact=compact))
    spec["target"] = int(target)
    if method == "B0":
        spec["tail"] = 0
        spec["serialized"] = 0
        return spec
    if int(spec["n_blocks"]) <= 0:
        raise RuntimeError(f"{method} does not fit one block in {int(target)} bytes")
    tail = int(target) - int(spec["nbytes"])
    if tail < 0 or tail >= int(spec["block_size"]):
        raise RuntimeError(
            f"{method} padding tail {tail} is not shorter than one {spec['block_size']}-byte block"
        )
    spec["tail"] = int(tail)
    spec["serialized"] = int(target)
    return spec


def _tail_field(tail_bytes: int) -> bytes:
    if int(tail_bytes) < 0 or int(tail_bytes) > 0xFFFF:
        raise ValueError("tail padding does not fit in the header")
    return struct.pack("<H8s", int(tail_bytes), b"\x00" * 8)


def coarse_from_tristate(tristate: np.ndarray, cells: int = COARSE) -> np.ndarray:
    """Pack a 100x100x8 tri-state grid into a coarse flag image (vectorized)."""
    tri = np.asarray(tristate)
    if tri.shape != (100, 100, 8):
        raise ValueError(f"tristate grid must be 100x100x8, got {tri.shape}")
    cells_i = int(cells)
    if cells_i <= 0 or 100 % cells_i != 0:
        raise ValueError(f"cells must evenly divide 100, got {cells}")
    scale = 100 // cells_i
    blocks = tri.reshape(cells_i, scale, cells_i, scale, 8)
    has_free = np.any(blocks == 1, axis=(1, 3, 4))
    occ_z = np.any(blocks == 2, axis=(1, 3))
    has_occ = np.any(occ_z, axis=2)
    has_unk = np.any(blocks == 0, axis=(1, 3, 4))
    z_idx = np.arange(8, dtype=np.uint8)[None, None, :]
    z_max = np.max(np.where(occ_z, z_idx, np.uint8(0)), axis=2)
    out = (
        has_free.astype(np.uint8)
        + has_occ.astype(np.uint8) * np.uint8(2)
        + has_unk.astype(np.uint8) * np.uint8(4)
        + np.where(has_occ, z_max, np.uint8(0)) * np.uint8(8)
    )
    return out


def envelope50_bitplanes(tristate: np.ndarray) -> np.ndarray:
    """Pack a 100x100x8 tri-state grid into 3 bit-planes at 50x50 (0.8m) resolution (939 bytes)."""
    tri = np.asarray(tristate)
    if tri.shape != (100, 100, 8):
        raise ValueError(f"tristate grid must be 100x100x8, got {tri.shape}")
    blocks = tri.reshape(50, 2, 50, 2, 8)
    has_free = np.any(blocks == 1, axis=(1, 3, 4)).reshape(-1)
    has_occ = np.any(blocks == 2, axis=(1, 3, 4)).reshape(-1)
    has_unk = np.any(blocks == 0, axis=(1, 3, 4)).reshape(-1)
    return np.concatenate((np.packbits(has_free), np.packbits(has_occ), np.packbits(has_unk))).astype(np.uint8)


def pack_zmax25_nibbles(env25: np.ndarray) -> np.ndarray:
    """Pack 25x25 max-height bin (3-bit values from coarse_from_tristate) into 313 nibble bytes."""
    zm = ((np.asarray(env25, dtype=np.uint8).reshape(-1) >> np.uint8(3)) & np.uint8(0x07))
    zm_pad = np.pad(zm, (0, 1))
    return ((zm_pad[0::2] & np.uint8(0x0F)) | ((zm_pad[1::2] & np.uint8(0x0F)) << np.uint8(4))).astype(np.uint8)


def unpack_zmax25_nibbles(nibbles: np.ndarray) -> np.ndarray:
    arr = np.asarray(nibbles, dtype=np.uint8).reshape(-1)
    out = np.empty(arr.size * 2, dtype=np.uint8)
    out[0::2] = arr & np.uint8(0x0F)
    out[1::2] = (arr >> np.uint8(4)) & np.uint8(0x0F)
    return out[:625].reshape(25, 25)


def pack_hybrid_env50(env50_bp: np.ndarray, env25: np.ndarray | None = None) -> np.ndarray:
    """Concatenate 50x50 bit-planes (939 B) with optional 25x25 z_max nibbles (313 B -> 1252 B) and 25x25 fallback."""
    bp = np.ascontiguousarray(env50_bp, dtype=np.uint8).reshape(-1)
    if bp.size != 939:
        raise ValueError(f"env50_bp must be 939 bytes, got {bp.size}")
    if env25 is None:
        return bp
    zm = pack_zmax25_nibbles(env25)
    e25 = np.ascontiguousarray(env25, dtype=np.uint8).reshape(-1)
    # 939 + 313 + 625 = 1877 B sentinel triggers adaptive framing inside _frame_envelope (never sent uncompressed).
    return np.concatenate((bp, zm, e25)).astype(np.uint8)


def _frame_envelope(raw: np.ndarray | None) -> bytes:
    slot = bytearray(ENVELOPE_SLOT)
    if raw is None:
        return bytes(slot)
    payload = np.ascontiguousarray(raw, dtype=np.uint8).reshape(-1)
    max_body = ENVELOPE_SLOT - 3
    # Adaptive multi-resolution envelope (1877 B = 939 B bp50 + 313 B zmax25 + 625 B env25 fallback):
    # 1) Try 50x50 bit-planes + 25x25 z_max (1252 B raw);
    # 2) Fallback to 50x50 bit-planes (939 B raw);
    # 3) Fallback to 25x25 flags+z_max (625 B raw, always <= 250 B zstd).
    if payload.size == 1877:
        cand_hybrid = payload[:1252].tobytes()
        for comp in (_COMPRESSOR, _COMPRESSOR_HIGH):
            c = comp.compress(cand_hybrid)
            if 0 < len(c) <= max_body:
                slot[0:3] = _U16B.pack(len(c), 0)
                slot[3 : 3 + len(c)] = c
                return bytes(slot)
        cand_bp50 = payload[:939].tobytes()
        for comp in (_COMPRESSOR, _COMPRESSOR_HIGH):
            c = comp.compress(cand_bp50)
            if 0 < len(c) <= max_body:
                slot[0:3] = _U16B.pack(len(c), 0)
                slot[3 : 3 + len(c)] = c
                return bytes(slot)
        payload = payload[1252:]
    raw_bytes = payload.tobytes()
    compressed = _COMPRESSOR.compress(raw_bytes)
    if len(compressed) > max_body:
        compressed = _COMPRESSOR_HIGH.compress(raw_bytes)
    use_zstd = len(compressed) > 0 and len(compressed) < len(raw_bytes)
    use_zstd = use_zstd and len(compressed) <= max_body
    if use_zstd:
        body = compressed
        codec = 0
    elif len(raw_bytes) <= max_body:
        body = raw_bytes
        codec = 1
    else:
        raise ValueError(
            f"envelope ({len(raw_bytes)} B raw, {len(compressed)} B zstd) does not fit in the 512-byte slot"
        )
    slot[0:3] = _U16B.pack(len(body), codec)
    slot[3 : 3 + len(body)] = body
    return bytes(slot)


def _read_envelope(slot: bytes) -> np.ndarray:
    length, codec = _U16B.unpack_from(slot, 0)
    if length == 0:
        return np.zeros((0,), dtype=np.uint8)
    body = bytes(slot[3 : 3 + length])
    raw = _DECOMPRESSOR.decompress(body) if codec == 0 else body
    return np.frombuffer(raw, dtype=np.uint8).copy()


def pack_bytes(
    method: str,
    sender_id: int,
    receiver_id: int,
    timestamp_s: float,
    pose,
    speed: float,
    yaw_rate: float,
    tokens: np.ndarray,
    envelope: np.ndarray | None = None,
    stage: int = 1,
    pad_slots: int | None = None,
    tail_bytes: int = 0,
    include_envelope: bool | None = None,
    compact: bool = False,
) -> bytes:
    if method not in METHOD_ID:
        raise KeyError(method)
    pose = np.asarray(pose, dtype=np.float32).reshape(6)
    width = token_width(method, stage, compact=compact)
    if method == "B0":
        tokens = np.zeros((0, width), dtype=np.uint8)
        pad_slots = None
        tail_bytes = 0
    tokens = np.ascontiguousarray(tokens, dtype=np.uint8)
    bad_width = tokens.shape[0] > 0 and int(tokens.shape[1]) != width
    if tokens.ndim != 2 or bad_width:
        raise ValueError(f"{method} expected tokens of width {width}, got {tokens.shape}")
    count = int(tokens.shape[0])
    tail_bytes = int(tail_bytes)
    if tail_bytes < 0 or tail_bytes >= block_size(width):
        raise ValueError("padding tail must be shorter than one micro-block")
    explicit_pad = pad_slots is not None
    if explicit_pad:
        pad_slots = int(pad_slots)
        if pad_slots < count:
            raise ValueError(f"pad_slots {pad_slots} is smaller than {count} real tokens")
        if pad_slots % BLOCK_TOKENS != 0:
            raise ValueError("pad_slots must cover whole micro-blocks")
        n_blocks = pad_slots // BLOCK_TOKENS
    else:
        n_blocks = int(np.ceil(count / BLOCK_TOKENS)) if count else 0
    padded = np.zeros((n_blocks * BLOCK_TOKENS, width), dtype=np.uint8)
    if count:
        padded[:count] = tokens
    use_compact = bool(compact and method in {"B3", "B4"})
    has_env = _has_envelope(method, include_envelope)
    flags = (FLAG_ENVELOPE if has_env else 0) | (FLAG_COMPACT if use_compact else 0)
    env_slot = ENVELOPE_SLOT if has_env else 0
    header = _HEADER.pack(
        MAGIC,
        1,
        METHOD_ID[method],
        _low(sender_id, 16),
        _low(receiver_id, 16),
        _low(round(float(timestamp_s) * 1000.0), 32),
        _low(n_blocks, 16),
        _low(width, 8),
        _low(stage, 8),
        flags,
        0,
        float(pose[0]),
        float(pose[1]),
        float(pose[2]),
        float(pose[3]),
        float(pose[4]),
        float(pose[5]),
        float(speed),
        float(yaw_rate),
        env_slot,
        _tail_field(tail_bytes),
    )
    if len(header) != HEADER_SIZE:
        raise RuntimeError(f"header is {len(header)} bytes")
    parts = [header]
    if env_slot:
        parts.append(_frame_envelope(envelope))
    if n_blocks > 0:
        bsize = block_size(width)
        padded_3d = padded.reshape(n_blocks, BLOCK_TOKENS, width)
        block_ids = np.arange(n_blocks, dtype=np.int64)
        n_valid = np.clip(count - block_ids * BLOCK_TOKENS, 0, BLOCK_TOKENS)
        if explicit_pad:
            n_pad = BLOCK_TOKENS - n_valid
            pad_mask = np.arange(BLOCK_TOKENS, dtype=np.int64)[None, :] >= n_valid[:, None]
            if np.any(pad_mask):
                padded_3d[pad_mask] = np.uint8(PAD_BYTE)
        else:
            n_pad = np.zeros(n_blocks, dtype=np.int64)
        blocks = np.zeros((n_blocks, bsize), dtype=np.uint8)
        if width >= 4:
            has_first = n_valid > 0
            if np.any(has_first):
                blocks[has_first, 0:4] = padded_3d[has_first, 0, 0:4]
        blocks[:, 4:6] = n_valid.astype(np.dtype("<u2")).view(np.uint8).reshape(n_blocks, 2)
        if explicit_pad:
            has_pad = n_pad > 0
            blocks[has_pad, 6] = n_pad[has_pad].astype(np.uint8)
            blocks[has_pad, 7] = np.uint8(PAD_BYTE)
        blocks[:, BLOCK_HDR:] = padded_3d.reshape(n_blocks, BLOCK_TOKENS * width)
        parts.append(blocks.tobytes())
    if tail_bytes:
        parts.append(bytes([PAD_BYTE]) * tail_bytes)
    return b"".join(parts)


def unpack_bytes(blob: bytes) -> dict:
    if len(blob) < HEADER_SIZE:
        raise ValueError("packet shorter than the header")
    fields = _HEADER.unpack_from(blob, 0)
    if fields[0] != MAGIC:
        raise ValueError("bad magic")
    method = ID_METHOD[int(fields[2])]
    n_blocks = int(fields[6])
    width = int(fields[7])
    env_slot = int(fields[19])
    tail_len = int.from_bytes(bytes(fields[20])[:2], "little")
    cursor = HEADER_SIZE
    envelope = np.zeros((0,), dtype=np.uint8)
    if env_slot:
        if len(blob) < cursor + env_slot:
            raise ValueError("truncated envelope")
        envelope = _read_envelope(blob[cursor : cursor + env_slot])
        cursor += env_slot
    if tail_len > len(blob) - cursor:
        raise ValueError("padding tail is longer than the packet")
    block_end = len(blob) - tail_len
    bsize = block_size(width)
    available = (block_end - cursor) // bsize if bsize else 0
    n_parse = min(n_blocks, int(available))
    pad_tokens = 0
    if n_parse > 0:
        view_2d = np.frombuffer(blob, dtype=np.uint8, count=n_parse * bsize, offset=cursor).reshape(n_parse, bsize)
        n_valids = np.ascontiguousarray(view_2d[:, 4:6]).view(np.dtype("<u2")).reshape(n_parse).astype(np.int64)
        n_valids = np.clip(n_valids, 0, BLOCK_TOKENS)
        pad_counts = np.where(view_2d[:, 7] == PAD_BYTE, view_2d[:, 6].astype(np.int64), 0)
        pad_counts = np.clip(pad_counts, 0, BLOCK_TOKENS - n_valids)
        pad_tokens = int(pad_counts.sum())
        payload_3d = view_2d[:, BLOCK_HDR:].reshape(n_parse, BLOCK_TOKENS, width)
        if np.all(n_valids == BLOCK_TOKENS):
            tokens = payload_3d.reshape(n_parse * BLOCK_TOKENS, width).copy()
        else:
            slot_mask = np.arange(BLOCK_TOKENS, dtype=np.int64)[None, :] < n_valids[:, None]
            tokens = payload_3d[slot_mask].copy()
            if tokens.size == 0:
                tokens = np.zeros((0, width), dtype=np.uint8)
    else:
        tokens = np.zeros((0, width), dtype=np.uint8)
    return {
        "method": method,
        "sender_id": int(fields[3]),
        "receiver_id": int(fields[4]),
        "timestamp_s": int(fields[5]) / 1000.0,
        "pose": np.asarray(fields[11:17], dtype=np.float32),
        "speed": float(fields[17]),
        "yaw_rate": float(fields[18]),
        "stage": int(fields[8]),
        "flags": int(fields[9]),
        "compact": bool(int(fields[9]) & FLAG_COMPACT),
        "n_blocks_header": n_blocks,
        "n_blocks": n_parse,
        "tokens": tokens,
        "envelope": envelope,
        "pad_tokens": int(pad_tokens),
        "pad_bytes": bytes(blob[block_end:]),
        "nbytes": len(blob),
    }


def drop_block(blob: bytes, block_index: int) -> bytes:
    """Remove one whole block. Later blocks keep their own boundaries."""
    view = unpack_bytes(blob)
    width = int(view["tokens"].shape[1]) if view["tokens"].size else int(blob[16])
    env_slot = ENVELOPE_SLOT if int(view["flags"]) % 2 == 1 else 0
    bsize = block_size(width)
    start = HEADER_SIZE + env_slot + int(block_index) * bsize
    end = start + bsize
    too_early = start < HEADER_SIZE + env_slot
    too_late = end > len(blob)
    if too_early or too_late:
        raise IndexError(f"block {block_index} is outside the packet")
    return blob[:start] + blob[end:]


def transmit(blob: bytes, method: str) -> bytes:
    if method == "B0":
        return b""
    return blob
