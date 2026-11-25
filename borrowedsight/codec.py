"""Token payloads for B1-B4. The packer itself does not interpret these bytes."""

from __future__ import annotations

import numpy as np

from borrowedsight.packer import TOKEN_BYTES, morton3, token_width


def demorton3(code: np.ndarray) -> np.ndarray:
    """Inverse of packer.morton3 for 7-bit x, 7-bit y, 3-bit z (bit-shift vectorized)."""
    packed = np.asarray(code, dtype=np.uint32)
    x = np.zeros(packed.shape, dtype=np.uint32)
    y = np.zeros(packed.shape, dtype=np.uint32)
    z = np.zeros(packed.shape, dtype=np.uint32)
    for bit in range(7):
        x |= ((packed >> np.uint32(3 * bit)) & np.uint32(1)) << np.uint32(bit)
        y |= ((packed >> np.uint32(3 * bit + 1)) & np.uint32(1)) << np.uint32(bit)
    for bit in range(3):
        z |= ((packed >> np.uint32(3 * bit + 2)) & np.uint32(1)) << np.uint32(bit)
    return np.stack((x, y, z), axis=-1).astype(np.int64)


def morton_from_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int64)
    if coords.size == 0:
        return np.zeros((0,), dtype=np.uint32)
    return morton3(coords[:, 0], coords[:, 1], coords[:, 2])


def _write_morton(buf: np.ndarray, morton: np.ndarray) -> None:
    packed = np.ascontiguousarray(morton, dtype=np.dtype("<u4"))
    buf[:, 0:4] = packed.view(np.uint8).reshape(-1, 4)


def pack_coord17(coords: np.ndarray) -> np.ndarray:
    """Lossless 17-bit coordinate packing for 100x100x8: x(7b) | y(7b)<<7 | z(3b)<<14."""
    c = np.asarray(coords, dtype=np.uint64).reshape(-1, 3)
    x = c[:, 0] & np.uint64(0x7F)
    y = c[:, 1] & np.uint64(0x7F)
    z = c[:, 2] & np.uint64(0x07)
    return x | (y << np.uint64(7)) | (z << np.uint64(14))


def unpack_coord17(packed: np.ndarray) -> np.ndarray:
    p = np.asarray(packed, dtype=np.uint64).reshape(-1)
    x = p & np.uint64(0x7F)
    y = (p >> np.uint64(7)) & np.uint64(0x7F)
    z = (p >> np.uint64(14)) & np.uint64(0x07)
    return np.stack((x, y, z), axis=-1).astype(np.int64)


def _read_morton(buf: np.ndarray, compact: bool = False) -> np.ndarray:
    if buf.size == 0:
        return np.zeros((0,), dtype=np.uint32)
    raw = np.ascontiguousarray(buf[:, 0:4]).view(np.dtype("<u4")).reshape(-1)
    if compact:
        coords = unpack_coord17(raw & np.uint32(0x1FFFF))
        return morton_from_coords(coords)
    return raw


def encode_tokens(method: str, coords, payload: dict, stage: int = 1, compact: bool = False) -> np.ndarray:
    width = token_width(method, stage, compact=compact)
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    n = int(coords.shape[0])
    buf = np.zeros((n, width), dtype=np.uint8)
    if n == 0:
        return buf
    if compact and method == "B3":
        c1 = np.asarray(payload["code"], dtype=np.uint64).reshape(n) & np.uint64(0x3F)
        cf = np.asarray(payload["conf"], dtype=np.uint64).reshape(n) & np.uint64(0xFF)
        c17 = pack_coord17(coords)
        if stage > 1:
            c2 = np.asarray(payload["code2"], dtype=np.uint64).reshape(n) & np.uint64(0x3F)
            packed = c17 | (c1 << np.uint64(17)) | (c2 << np.uint64(23)) | (cf << np.uint64(29))
            buf[:] = np.ascontiguousarray(packed, dtype=np.dtype("<u8")).view(np.uint8).reshape(n, 8)[:, :5]
        else:
            packed = (c17 | (c1 << np.uint64(17)) | (cf << np.uint64(23))).astype(np.dtype("<u4"))
            buf[:] = np.ascontiguousarray(packed).view(np.uint8).reshape(n, 4)
        return buf
    if compact and method == "B4":
        c17 = pack_coord17(coords)
        cl = np.asarray(payload["cls"], dtype=np.uint64).reshape(n) & np.uint64(0x0F)
        c1 = np.asarray(payload["code"], dtype=np.uint64).reshape(n) & np.uint64(0x3F)
        if stage > 1 and "code2" in payload and payload["code2"] is not None:
            c2 = np.asarray(payload["code2"], dtype=np.uint64).reshape(n) & np.uint64(0x3F)
        else:
            c2 = np.zeros(n, dtype=np.uint64)
        cf_u8 = np.asarray(payload["conf"], dtype=np.float32).reshape(n)
        cf7 = np.clip(np.rint(cf_u8 * (127.0 / 255.0)), 0, 127).astype(np.uint64) & np.uint64(0x7F)
        packed = c17 | (cl << np.uint64(17)) | (c1 << np.uint64(21)) | (c2 << np.uint64(27)) | (cf7 << np.uint64(33))
        buf[:] = np.ascontiguousarray(packed, dtype=np.dtype("<u8")).view(np.uint8).reshape(n, 8)[:, :5]
        return buf
    morton = morton_from_coords(coords)
    _write_morton(buf, morton)
    if method == "B1":
        feat = np.ascontiguousarray(payload["feat"], dtype=np.dtype("<f2")).reshape(n, 16)
        buf[:, 4:36] = feat.view(np.uint8).reshape(n, 32)
    elif method == "B2":
        feat = np.ascontiguousarray(payload["feat"], dtype=np.dtype("<f2")).reshape(n, 8)
        buf[:, 4:20] = feat.view(np.uint8).reshape(n, 16)
    elif method == "B3":
        buf[:, 4] = np.asarray(payload["code"], dtype=np.uint8).reshape(n)
        buf[:, 5] = np.asarray(payload["conf"], dtype=np.uint8).reshape(n)
        if stage > 1:
            buf[:, 8] = np.asarray(payload["code2"], dtype=np.uint8).reshape(n)
    elif method == "B4":
        buf[:, 4] = np.asarray(payload["cls"], dtype=np.uint8).reshape(n)
        buf[:, 5] = np.asarray(payload["code"], dtype=np.uint8).reshape(n)
        buf[:, 6] = np.asarray(payload["conf"], dtype=np.uint8).reshape(n)
        buf[:, 7] = np.asarray(payload["evidence"], dtype=np.uint8).reshape(n)
        if stage > 1:
            buf[:, 8] = np.asarray(payload["code2"], dtype=np.uint8).reshape(n)
    else:
        raise KeyError(method)
    if buf.shape[1] != width:
        raise RuntimeError(f"encoded width {buf.shape[1]} != {width}")
    if (not compact) and method in TOKEN_BYTES and stage == 1 and buf.shape[1] != TOKEN_BYTES[method]:
        raise RuntimeError("stage-1 width drifted from the budget table")
    return buf


def decode_tokens(method: str, tokens: np.ndarray, stage: int = 1, compact: bool = False) -> dict:
    use_compact = bool(compact and method in {"B3", "B4"})
    tokens = np.asarray(tokens, dtype=np.uint8)
    width = token_width(method, stage, compact=use_compact)
    if tokens.size == 0:
        tokens = np.zeros((0, width), dtype=np.uint8)
    if tokens.shape[1] != width:
        raise ValueError(f"{method} token width {tokens.shape[1]} != {width}")
    n = int(tokens.shape[0])
    morton = _read_morton(tokens, compact=use_compact)
    out = {"morton": morton, "coords_morton": morton}
    if n == 0:
        out["feat"] = np.zeros((0, 16 if method == "B1" else 8), dtype=np.float16)
        out["code"] = np.zeros((0,), dtype=np.uint8)
        out["cls"] = np.zeros((0,), dtype=np.uint8)
        out["conf"] = np.zeros((0,), dtype=np.uint8)
        out["evidence"] = np.zeros((0,), dtype=np.uint8)
        if stage > 1:
            out["code2"] = np.zeros((0,), dtype=np.uint8)
        return out
    if use_compact and method == "B3":
        if stage > 1:
            pad8 = np.zeros((n, 8), dtype=np.uint8)
            pad8[:, :5] = tokens
            val = pad8.view(np.dtype("<u8")).reshape(n)
            out["code"] = ((val >> np.uint64(17)) & np.uint64(0x3F)).astype(np.uint8)
            out["code2"] = ((val >> np.uint64(23)) & np.uint64(0x3F)).astype(np.uint8)
            out["conf"] = ((val >> np.uint64(29)) & np.uint64(0xFF)).astype(np.uint8)
        else:
            val = np.ascontiguousarray(tokens).view(np.dtype("<u4")).reshape(n)
            out["code"] = ((val >> np.uint32(17)) & np.uint32(0x3F)).astype(np.uint8)
            out["conf"] = ((val >> np.uint32(23)) & np.uint32(0xFF)).astype(np.uint8)
        return out
    if compact and method == "B4":
        pad8 = np.zeros((n, 8), dtype=np.uint8)
        pad8[:, :5] = tokens
        val = pad8.view(np.dtype("<u8")).reshape(n)
        out["cls"] = ((val >> np.uint64(17)) & np.uint64(0x0F)).astype(np.uint8)
        out["code"] = ((val >> np.uint64(21)) & np.uint64(0x3F)).astype(np.uint8)
        if stage > 1:
            out["code2"] = ((val >> np.uint64(27)) & np.uint64(0x3F)).astype(np.uint8)
        cf7 = ((val >> np.uint64(33)) & np.uint64(0x7F)).astype(np.float32)
        out["conf"] = np.clip(np.rint(cf7 * (255.0 / 127.0)), 0, 255).astype(np.uint8)
        out["evidence"] = np.full(n, 2, dtype=np.uint8)
        return out
    if method == "B1":
        raw = np.ascontiguousarray(tokens[:, 4:36])
        out["feat"] = raw.view(np.dtype("<f2")).reshape(n, 16).copy()
    elif method == "B2":
        raw = np.ascontiguousarray(tokens[:, 4:20])
        out["feat"] = raw.view(np.dtype("<f2")).reshape(n, 8).copy()
    elif method == "B3":
        out["code"] = tokens[:, 4].copy()
        out["conf"] = tokens[:, 5].copy()
        if stage > 1:
            out["code2"] = tokens[:, 8].copy()
    elif method == "B4":
        out["cls"] = tokens[:, 4].copy()
        out["code"] = tokens[:, 5].copy()
        out["conf"] = tokens[:, 6].copy()
        out["evidence"] = tokens[:, 7].copy()
        if stage > 1:
            out["code2"] = tokens[:, 8].copy()
    else:
        raise KeyError(method)
    return out
