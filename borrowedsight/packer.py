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
METHOD_ID = {"B0": 0, "B1": 1, "B2": 2, "B3": 3, "B4": 4}
ID_METHOD = {value: key for key, value in METHOD_ID.items()}

_LE = chr(60)
_HEADER = struct.Struct(_LE + "4sBBHHIHBBBB6fffH10s")
_BLOCK = struct.Struct(_LE + "IH2s")
_U16B = struct.Struct(_LE + "HB")
_U16 = struct.Struct(_LE + "H")
_U32 = struct.Struct(_LE + "I")
_COMPRESSOR = zstandard.ZstdCompressor(level=3)
_DECOMPRESSOR = zstandard.ZstdDecompressor()


def _low(value: int, bits: int) -> int:
    return int(value) % (2 ** int(bits))


def morton3(x, y, z) -> np.ndarray:
    """Interleave 7-bit x, 7-bit y and 3-bit z."""
    xx = np.asarray(x, dtype=np.uint64) % np.uint64(128)
    yy = np.asarray(y, dtype=np.uint64) % np.uint64(128)
    zz = np.asarray(z, dtype=np.uint64) % np.uint64(8)
    code = np.zeros(np.broadcast(xx, yy, zz).shape, dtype=np.uint64)
    for bit in range(7):
        code += ((xx // np.uint64(2 ** bit)) % np.uint64(2)) * np.uint64(2 ** (3 * bit))
        code += ((yy // np.uint64(2 ** bit)) % np.uint64(2)) * np.uint64(2 ** (3 * bit + 1))
    for bit in range(3):
        code += ((zz // np.uint64(2 ** bit)) % np.uint64(2)) * np.uint64(2 ** (3 * bit + 2))
    return code.astype(np.uint32)


def token_width(method: str, stage: int = 1) -> int:
    if method == "B0":
        return 8
    extra = 1 if int(stage) > 1 else 0
    return int(TOKEN_BYTES[method]) + extra


def block_size(token_bytes: int) -> int:
    return BLOCK_HDR + BLOCK_TOKENS * int(token_bytes)


def nbytes_for_count(method: str, n_tokens: int, stage: int = 1) -> int:
    """Serialized size of a packet that carries exactly this many tokens."""
    if method == "B0" or int(n_tokens) <= 0:
        return 0
    width = token_width(method, stage)
    n_blocks = (int(n_tokens) + BLOCK_TOKENS - 1) // BLOCK_TOKENS
    env = ENVELOPE_SLOT if method == "B4" else 0
    return HEADER_SIZE + env + n_blocks * block_size(width)


def budget_for(method: str, target: int = TARGET_BYTES, stage: int = 1) -> dict:
    """Largest whole-block packing that does not exceed target."""
    if method == "B0":
        return {"method": method, "n_tokens": 0, "n_blocks": 0, "nbytes": 0, "token_bytes": 0}
    width = token_width(method, stage)
    env = ENVELOPE_SLOT if method == "B4" else 0
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
    }


def packet_layout(method: str, target: int = FAIR_BUDGET_BYTES, stage: int = 1) -> dict:
    """Token count from the byte budget and this method's own token width.

    ``serialized`` is ``target`` for B1–B4. ``tail`` is padding after the last
    micro-block and is shorter than one block, so dropping a block cannot
    promote that tail into a token.
    """
    spec = dict(budget_for(method, target=int(target), stage=stage))
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
    """Pack a 100x100x8 tri-state grid into a coarse flag image."""
    tri = np.asarray(tristate)
    if tri.shape != (100, 100, 8):
        raise ValueError(f"tristate grid must be 100x100x8, got {tri.shape}")
    scale = 100 // int(cells)
    out = np.zeros((cells, cells), dtype=np.uint8)
    for ix in range(cells):
        for iy in range(cells):
            x0 = ix * scale
            y0 = iy * scale
            block = tri[x0 : x0 + scale, y0 : y0 + scale]
            flags = 0
            if np.any(block == 1):
                flags += 1
            if np.any(block == 2):
                flags += 2
            if np.any(block == 0):
                flags += 4
            occupied_z = np.any(block == 2, axis=(0, 1))
            if np.any(occupied_z):
                z_max = int(np.max(np.nonzero(occupied_z)[0]))
                flags += (z_max % 8) * 8
            out[ix, iy] = np.uint8(flags)
    return out


def _frame_envelope(raw: np.ndarray | None) -> bytes:
    slot = bytearray(ENVELOPE_SLOT)
    if raw is None:
        return bytes(slot)
    payload = np.ascontiguousarray(raw, dtype=np.uint8).reshape(-1)
    if int(payload.size) > ENVELOPE_SLOT - 3:
        raise ValueError("envelope does not fit in the 512-byte slot")
    compressed = _COMPRESSOR.compress(payload.tobytes())
    use_zstd = len(compressed) > 0 and len(compressed) < int(payload.size)
    use_zstd = use_zstd and len(compressed) <= ENVELOPE_SLOT - 3
    if use_zstd:
        body = compressed
        codec = 0
    else:
        body = payload.tobytes()
        codec = 1
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
) -> bytes:
    if method not in METHOD_ID:
        raise KeyError(method)
    pose = np.asarray(pose, dtype=np.float32).reshape(6)
    width = token_width(method, stage)
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
    flags = 1 if method == "B4" else 0
    env_slot = ENVELOPE_SLOT if method == "B4" else 0
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
    for block_i in range(n_blocks):
        chunk = padded[block_i * BLOCK_TOKENS : (block_i + 1) * BLOCK_TOKENS]
        n_valid = int(min(BLOCK_TOKENS, max(0, count - block_i * BLOCK_TOKENS)))
        n_pad = BLOCK_TOKENS - n_valid if explicit_pad else 0
        if n_pad:
            chunk[n_valid:] = np.uint8(PAD_BYTE)
        if width >= 4 and n_valid > 0:
            first = int(_U32.unpack(bytes(chunk[0, :4]))[0])
        else:
            first = 0
        reserved = bytes((n_pad, PAD_BYTE)) if n_pad else b"\x00\x00"
        parts.append(_BLOCK.pack(_low(first, 32), n_valid, reserved))
        parts.append(chunk.tobytes())
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
    chunks = []
    pad_tokens = 0
    for _ in range(n_parse):
        n_valid = int(_U16.unpack_from(blob, cursor + 4)[0])
        n_valid = min(max(n_valid, 0), BLOCK_TOKENS)
        pad_count = int(blob[cursor + 6])
        pad_mark = int(blob[cursor + 7])
        if pad_mark != PAD_BYTE:
            pad_count = 0
        pad_count = min(max(pad_count, 0), BLOCK_TOKENS - n_valid)
        raw = np.frombuffer(
            blob, dtype=np.uint8, count=BLOCK_TOKENS * width, offset=cursor + BLOCK_HDR
        ).copy()
        if n_valid:
            chunks.append(raw.reshape(BLOCK_TOKENS, width)[:n_valid])
        pad_tokens += pad_count
        cursor += bsize
    if chunks:
        tokens = np.concatenate(chunks, axis=0)
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
