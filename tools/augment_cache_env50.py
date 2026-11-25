"""Fast additive helper: compute sender-ray 50x50 3-bit bit-plane envelopes (939 B)
and append `envelope50` into existing sharded sparse caches without touching any other field.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from borrowedsight.opv2v_data import RealOPV2VDataset
from borrowedsight.packer import envelope50_bitplanes
from borrowedsight.paths import DATA_ROOT
from borrowedsight.weather import lisa_fog, lisa_rain, load_weather
from tools.ray_visibility import ray_visibility

CACHE_ROOT = DATA_ROOT / "cache"


def augment_split(split: str, weather_mode: str, cache_dir: Path, device: torch.device, seed: int) -> None:
    shards = sorted(cache_dir.glob("shard_*.npz"))
    if not shards:
        print(f"SKIP {cache_dir}: no shards found", flush=True)
        return
    # Check if already present in all shards
    if all("envelope50" in np.load(sp, allow_pickle=False).files for sp in shards):
        print(f"OK {weather_mode}/{split}: envelope50 already present in all {len(shards)} shards", flush=True)
        return

    weather_cfg = load_weather()
    rng = np.random.default_rng(seed)
    ds = RealOPV2VDataset(split, stride=1)
    t0 = time.perf_counter()

    # Build lookup by (scene_name, frame) -> list of envelope50 per sender_id
    all_env50: list[np.ndarray] = []
    for idx in range(len(ds)):
        sample = ds.__getitem__(idx, device=device, load_points=True)
        for slot in range(len(sample["sender_ids"])):
            pts = sample["points"][slot]
            if weather_mode == "fog" and len(pts) > 0:
                pts = lisa_fog(pts, rng, weather_cfg).astype(np.float32)
            elif weather_mode == "rain" and len(pts) > 0:
                pts, _ = lisa_rain(pts, rng, weather_cfg)
                pts = pts.astype(np.float32)
            if len(pts) == 0:
                tri_np = np.zeros((100, 100, 8), dtype=np.uint8)
            else:
                pts_t = torch.as_tensor(np.asarray(pts, dtype=np.float32), device=device)
                tri_np = ray_visibility(pts_t, torch.zeros(3, device=device)).detach().cpu().numpy()
            all_env50.append(envelope50_bitplanes(tri_np))

    cursor = 0
    for sp in shards:
        with np.load(sp, allow_pickle=False) as blob:
            data = {k: blob[k] for k in blob.files}
        n_s = int(len(data["sender_id"]))
        chunk = np.stack(all_env50[cursor : cursor + n_s], axis=0).astype(np.uint8) if n_s else np.zeros((0, 939), np.uint8)
        cursor += n_s
        data["envelope50"] = chunk
        tmp = sp.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **data)
        tmp.replace(sp)

    if cursor != len(all_env50):
        raise RuntimeError(f"sender count mismatch in {weather_mode}/{split}: {cursor} != {len(all_env50)}")
    print(f"DONE envelope50 [{weather_mode}/{split}]: {cursor} senders across {len(shards)} shards in {time.perf_counter()-t0:.1f}s", flush=True)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for split in ("train", "validate", "test"):
        augment_split(split, "clear", CACHE_ROOT / "opv2v_clear" / split, device, seed=0)
    for mode, sd in (("fog", 101), ("rain", 202)):
        for split in ("validate", "test"):
            augment_split(split, mode, CACHE_ROOT / f"opv2v_lisa_{mode}" / split, device, seed=sd)


if __name__ == "__main__":
    main()
