"""Token payloads for B1-B4. The packer itself does not interpret these bytes."""

from __future__ import annotations

import numpy as np

from borrowedsight.packer import TOKEN_BYTES, morton3, token_width


def demorton3(code: np.ndarray) -> np.ndarray:
    """Inverse of packer.morton3 for 7-bit x, 7-bit y, 3-bit z."""
    packed = np.asarray(code, dtype=np.uint64)
    x = np.zeros(packed.shape, dtype=np.uint64)
    y = np.zeros(packed.shape, dtype=np.uint64)
    z = np.zeros(packed.shape, dtype=np.uint64)
    for bit in range(7):
        x = x + ((packed // np.uint64(2 ** (3 * bit))) % np.uint64(2)) * np.uint64(2 ** bit)
        y = y + ((packed // np.uint64(2 ** (3 * bit + 1))) % np.uint64(2)) * np.uint64(2 ** bit)
    for bit in range(3):
        z = z + ((packed // np.uint64(2 ** (3 * bit + 2))) % np.uint64(2)) * np.uint64(2 ** bit)
    return np.stack((x, y, z), axis=-1).astype(np.int64)


def morton_from_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int64)
    if coords.size == 0:
        return np.zeros((0,), dtype=np.uint32)
    return morton3(coords[:, 0], coords[:, 1], coords[:, 2])


def _write_morton(buf: np.ndarray, morton: np.ndarray) -> None:
    packed = np.ascontiguousarray(morton, dtype=np.dtype("<u4"))
    buf[:, 0:4] = packed.view(np.uint8).reshape(-1, 4)


def _read_morton(buf: np.ndarray) -> np.ndarray:
    if buf.size == 0:
        return np.zeros((0,), dtype=np.uint32)
    return np.ascontiguousarray(buf[:, 0:4]).view(np.dtype("<u4")).reshape(-1)


def encode_tokens(method: str, coords, payload: dict, stage: int = 1) -> np.ndarray:
    width = token_width(method, stage)
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    n = int(coords.shape[0])
    buf = np.zeros((n, width), dtype=np.uint8)
    if n == 0:
        return buf
    _write_morton(buf, morton_from_coords(coords))
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
    if method in TOKEN_BYTES and stage == 1 and buf.shape[1] != TOKEN_BYTES[method]:
        raise RuntimeError("stage-1 width drifted from the budget table")
    return buf


def decode_tokens(method: str, tokens: np.ndarray, stage: int = 1) -> dict:
    tokens = np.asarray(tokens, dtype=np.uint8)
    width = token_width(method, stage)
    if tokens.size == 0:
        tokens = np.zeros((0, width), dtype=np.uint8)
    if tokens.shape[1] != width:
        raise ValueError(f"{method} token width {tokens.shape[1]} != {width}")
    n = int(tokens.shape[0])
    out = {"morton": _read_morton(tokens), "coords_morton": _read_morton(tokens)}
    if n == 0:
        out["feat"] = np.zeros((0, 16 if method == "B1" else 8), dtype=np.float16)
        out["code"] = np.zeros((0,), dtype=np.uint8)
        out["cls"] = np.zeros((0,), dtype=np.uint8)
        out["conf"] = np.zeros((0,), dtype=np.uint8)
        out["evidence"] = np.zeros((0,), dtype=np.uint8)
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
