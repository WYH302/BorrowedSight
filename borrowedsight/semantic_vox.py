"""Fast exact CoHFF Semantic-OPV2V PCD parser and 100x100x8 voxelizer.

Matches refs/CoHFF/cohff_opv2v/opv2v_dataset/opv2v_dataset.py:
- CoHFF grid: [-20, 20] x [-20, 20] x [-2.5, 0.9], 100x100x8
- Majority vote per voxel over raw CARLA semantic tags, then RAW_TO_MAPPED (0..12)
- Supports reading from disk paths OR directly from raw bytes inside a ZipFile
- Uses ProcessPoolExecutor for multi-core ASCII PCD parsing & voxelization
"""
from __future__ import annotations

import concurrent.futures
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import yaml

from borrowedsight.grid import GRID_SIZE, transform_points, voxel_index
from borrowedsight.labels import RAW_TO_MAPPED

_SEM_STEM_RE = re.compile(r"(?:filtered_)?(\d+)_semantic\.pcd$")


class _OPV2VYamlLoader(yaml.SafeLoader):
    """Safe YAML loader supporting OPV2V pickled numpy scalar tags without arbitrary code exec."""


def _construct_tuple(loader: yaml.Loader, node: yaml.Node):
    return tuple(loader.construct_sequence(node))


def _construct_np_dtype(loader: yaml.Loader, node: yaml.Node):
    if isinstance(node, yaml.MappingNode):
        mapping = loader.construct_mapping(node, deep=True)
        args = mapping.get("args") or ["f8"]
        return np.dtype(args[0])
    seq = loader.construct_sequence(node, deep=True)
    return np.dtype(seq[0]) if seq else np.dtype("f8")


def _construct_np_scalar(loader: yaml.Loader, node: yaml.Node):
    seq = loader.construct_sequence(node, deep=True)
    if len(seq) >= 2 and isinstance(seq[1], (bytes, bytearray)):
        dt = seq[0] if isinstance(seq[0], np.dtype) else np.dtype("f8")
        return float(np.frombuffer(seq[1], dtype=dt, count=1)[0])
    return 0.0


_OPV2VYamlLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_tuple)
_OPV2VYamlLoader.add_constructor("tag:yaml.org,2002:python/object/apply:numpy.dtype", _construct_np_dtype)
_OPV2VYamlLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar", _construct_np_scalar
)
_OPV2VYamlLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:numpy._core.multiarray.scalar", _construct_np_scalar
)


def load_opv2v_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=_OPV2VYamlLoader)


def _extract_semantic_stem(filename: str) -> str | None:
    m = _SEM_STEM_RE.match(filename)
    return m.group(1) if m else None

NX, NY, NZ = GRID_SIZE
_RAW_LUT = np.zeros(256, dtype=np.uint8)
for _k, _v in RAW_TO_MAPPED.items():
    _RAW_LUT[int(_k)] = int(_v)


def parse_semantic_pcd_bytes(raw_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Parse ASCII or binary Semantic-OPV2V _semantic.pcd bytes -> (xyz [N,3] float32, raw_cls [N] uint8)."""
    header_end = raw_bytes.find(b"DATA ascii")
    if header_end >= 0:
        nl = raw_bytes.find(b"\n", header_end)
        body = raw_bytes[nl + 1 :] if nl >= 0 else b""
        if not body.strip():
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.uint8)
        arr = np.fromstring(body.decode("ascii", errors="ignore"), sep=" ", dtype=np.float32)
        n = arr.size // 4
        if n == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.uint8)
        arr = arr[: n * 4].reshape(n, 4)
        xyz = np.ascontiguousarray(arr[:, :3], dtype=np.float32)
        raw_cls = (arr[:, 3].astype(np.int32) & 0xFF).astype(np.uint8)
        return xyz, raw_cls

    bio = io.BytesIO(raw_bytes)
    header = []
    while True:
        line = bio.readline()
        if not line:
            break
        s = line.decode("ascii", errors="ignore").strip()
        header.append(s)
        if s.startswith("DATA"):
            break
    fields, sizes, types, counts = [], [], [], []
    width, npoints = 0, 0
    for s in header:
        if s.startswith("FIELDS"):
            fields = s.split()[1:]
        elif s.startswith("SIZE"):
            sizes = [int(v) for v in s.split()[1:]]
        elif s.startswith("TYPE"):
            types = s.split()[1:]
        elif s.startswith("COUNT"):
            counts = [int(v) for v in s.split()[1:]]
        elif s.startswith("WIDTH"):
            width = int(s.split()[1])
        elif s.startswith("POINTS"):
            npoints = int(s.split()[1])
    if not counts:
        counts = [1] * len(fields)
    dtype_map = {
        "F": {4: "<f4", 8: "<f8"},
        "U": {1: "u1", 2: "<u2", 4: "<u4"},
        "I": {1: "i1", 2: "<i2", 4: "<i4"},
    }
    dt = []
    for name, sz, tp, ct in zip(fields, sizes, types, counts):
        base = dtype_map.get(tp, {}).get(sz, "<f4")
        dt.append((name, base) if ct == 1 else (name, base, (ct,)))
    rec = np.frombuffer(bio.read(), dtype=np.dtype(dt), count=npoints or width)
    if rec.size == 0 or "x" not in rec.dtype.names:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.uint8)
    xyz = np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float32)
    if "rgb" in rec.dtype.names:
        rgb_col = rec["rgb"]
        rgb_i = rgb_col.view(np.uint32) if rgb_col.dtype.kind == "f" else rgb_col.astype(np.uint32)
        raw_cls = (rgb_i & 0xFF).astype(np.uint8)
    else:
        raw_cls = np.zeros((len(xyz),), dtype=np.uint8)
    return xyz, raw_cls


def parse_raw_pcd_xyz(path: Path) -> np.ndarray:
    """Fast binary/ASCII reader for raw OPV2V .pcd -> xyz [N,3] float32."""
    raw_bytes = path.read_bytes()
    bio = io.BytesIO(raw_bytes)
    header = []
    data_type = "binary"
    while True:
        line = bio.readline()
        if not line:
            break
        s = line.decode("ascii", errors="ignore").strip()
        header.append(s)
        if s.startswith("DATA"):
            data_type = s.split()[-1].lower()
            break
    fields, sizes, types, counts = [], [], [], []
    width, npoints = 0, 0
    for s in header:
        if s.startswith("FIELDS"):
            fields = s.split()[1:]
        elif s.startswith("SIZE"):
            sizes = [int(v) for v in s.split()[1:]]
        elif s.startswith("TYPE"):
            types = s.split()[1:]
        elif s.startswith("COUNT"):
            counts = [int(v) for v in s.split()[1:]]
        elif s.startswith("WIDTH"):
            width = int(s.split()[1])
        elif s.startswith("POINTS"):
            npoints = int(s.split()[1])
    if not counts:
        counts = [1] * len(fields)
    if data_type == "ascii":
        body = bio.read().decode("ascii", errors="ignore")
        ncols = sum(counts)
        arr = np.fromstring(body, sep=" ", dtype=np.float32)
        n = arr.size // max(1, ncols)
        if n == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return np.ascontiguousarray(arr[: n * ncols].reshape(n, ncols)[:, :3], dtype=np.float32)
    dtype_map = {
        "F": {4: "<f4", 8: "<f8"},
        "U": {1: "u1", 2: "<u2", 4: "<u4"},
        "I": {1: "i1", 2: "<i2", 4: "<i4"},
    }
    dt = []
    for name, sz, tp, ct in zip(fields, sizes, types, counts):
        base = dtype_map.get(tp, {}).get(sz, "<f4")
        dt.append((name, base) if ct == 1 else (name, base, (ct,)))
    rec = np.frombuffer(bio.read(), dtype=np.dtype(dt), count=npoints or width)
    if rec.size == 0 or "x" not in rec.dtype.names:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float32)


def voxelize_majority(xyz: np.ndarray, raw_cls: np.ndarray) -> np.ndarray:
    """Exact CoHFF majority vote per voxel on 100x100x8, mapped through RAW_TO_MAPPED (0..12)."""
    grid = np.zeros(GRID_SIZE, dtype=np.uint8)
    if xyz.size == 0:
        return grid
    idx, valid = voxel_index(xyz)
    if not np.any(valid):
        return grid
    idx = idx[valid]
    cls = raw_cls[valid].astype(np.int64)
    lin = idx[:, 0] * (NY * NZ) + idx[:, 1] * NZ + idx[:, 2]
    keys = lin * 256 + cls
    uniq_keys, counts = np.unique(keys, return_counts=True)
    vox = uniq_keys // 256
    ucls = (uniq_keys % 256).astype(np.uint8)
    order = np.lexsort((ucls, -counts, vox))
    vox_s = vox[order]
    cls_s = ucls[order]
    first = np.empty(len(vox_s), dtype=bool)
    first[0] = True
    first[1:] = vox_s[1:] != vox_s[:-1]
    flat = grid.reshape(-1)
    flat[vox_s[first]] = _RAW_LUT[cls_s[first]]
    return grid


def sorted_cav_ids(raw_scene_dir: Path) -> list[str]:
    cavs = sorted(
        [d.name for d in raw_scene_dir.iterdir() if d.is_dir() and d.name.lstrip("-").isdigit()],
        key=lambda x: int(x),
    )
    if cavs and int(cavs[0]) < 0:
        neg = [c for c in cavs if int(c) < 0]
        pos = [c for c in cavs if int(c) >= 0]
        cavs = pos + neg
    return cavs


def _process_one_frame_task(args: tuple) -> tuple:
    """Top-level picklable worker for parallel multi-core frame voxelization."""
    fi, stem, raw_scene_str, cav_ids, zip_path_str, zip_members, sem_dir_str = args
    raw_scene_dir = Path(raw_scene_str)
    n_k = len(cav_ids)
    poses_f = np.zeros((n_k, 6), dtype=np.float32)
    speeds_f = np.zeros((n_k,), dtype=np.float32)
    sender_sem_f = np.zeros((n_k, NX, NY, NZ), dtype=np.uint8)

    metas = []
    for ki, cav in enumerate(cav_ids):
        meta = load_opv2v_yaml(raw_scene_dir / cav / f"{stem}.yaml")
        metas.append(meta)
        poses_f[ki] = np.asarray(meta["lidar_pose"], dtype=np.float32)
        speeds_f[ki] = float(meta.get("ego_speed", 0.0))

    ego_pose = poses_f[0]
    ego_xy = ego_pose[:2]
    seen_v = set()
    for meta in metas:
        for vid, v in (meta.get("vehicles") or {}).items():
            if vid in seen_v or not isinstance(v, dict):
                continue
            loc = v.get("location")
            if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                if abs(float(loc[0]) - float(ego_xy[0])) <= 20.0 and abs(float(loc[1]) - float(ego_xy[1])) <= 20.0:
                    seen_v.add(vid)
    n_veh = len(seen_v)

    ego_xyz_parts = []
    ego_cls_parts = []
    zf = zipfile.ZipFile(zip_path_str, "r") if zip_path_str else None
    try:
        for ki, cav in enumerate(cav_ids):
            if zf is not None:
                raw_b = zf.read(zip_members[ki])
            else:
                p_plain = Path(sem_dir_str) / cav / f"{stem}_semantic.pcd"
                p_filt = Path(sem_dir_str) / cav / f"filtered_{stem}_semantic.pcd"
                raw_b = (p_plain if p_plain.exists() else p_filt).read_bytes()
            xyz, raw_cls = parse_semantic_pcd_bytes(raw_b)
            sender_sem_f[ki] = voxelize_majority(xyz, raw_cls)
            if xyz.size > 0:
                if ki == 0:
                    ego_xyz_parts.append(xyz.astype(np.float64))
                else:
                    ego_xyz_parts.append(transform_points(xyz, poses_f[ki], ego_pose))
                ego_cls_parts.append(raw_cls)
    finally:
        if zf is not None:
            zf.close()

    if ego_xyz_parts:
        collab_f = voxelize_majority(np.concatenate(ego_xyz_parts, axis=0), np.concatenate(ego_cls_parts, axis=0))
    else:
        collab_f = np.zeros(GRID_SIZE, dtype=np.uint8)
    return fi, poses_f, speeds_f, n_veh, sender_sem_f, collab_f


def build_scene_voxels(
    raw_scene_dir: Path,
    out_npz: Path,
    semantic_scene_dir: Path | None = None,
    semantic_zip_path: Path | None = None,
    workers: int = 6,
) -> dict:
    """Build compact exact CoHFF semantic voxel archive for one OPV2V scene (multi-core)."""
    cav_ids = sorted_cav_ids(raw_scene_dir)
    if len(cav_ids) < 2:
        raise RuntimeError(f"Scene {raw_scene_dir.name} has < 2 CAVs: {cav_ids}")

    zip_index: dict[tuple[str, str], str] = {}
    if semantic_zip_path is not None:
        with zipfile.ZipFile(semantic_zip_path, "r") as zf:
            for name in zf.namelist():
                if not name.endswith("_semantic.pcd"):
                    continue
                parts = name.replace("\\", "/").split("/")
                if len(parts) < 2:
                    continue
                stem = _extract_semantic_stem(parts[-1])
                if stem is None:
                    continue
                key = (parts[-2], stem)
                # Prefer non-filtered if both somehow exist, otherwise store matched entry
                if key not in zip_index or not parts[-1].startswith("filtered_"):
                    zip_index[key] = name

    stem_sets = []
    for cav in cav_ids:
        cdir = raw_scene_dir / cav
        raw_stems = {
            p.stem
            for p in cdir.glob("*.yaml")
            if p.stem.isdigit() and (cdir / f"{p.stem}.pcd").exists()
        }
        if semantic_zip_path is not None:
            sem_stems = {stem for (c, stem) in zip_index if c == cav}
        elif semantic_scene_dir is not None and (semantic_scene_dir / cav).is_dir():
            sem_stems = {
                s
                for p in (semantic_scene_dir / cav).glob("*_semantic.pcd")
                if (s := _extract_semantic_stem(p.name)) is not None
            }
        else:
            sem_stems = set()
        stem_sets.append(raw_stems & sem_stems)

    stems = sorted(set.intersection(*stem_sets)) if stem_sets else []
    if not stems:
        raise RuntimeError(f"No matched raw+semantic frames for {raw_scene_dir.name}")

    n_f = len(stems)
    n_k = len(cav_ids)
    poses = np.zeros((n_f, n_k, 6), dtype=np.float32)
    speeds = np.zeros((n_f, n_k), dtype=np.float32)
    sender_sem = np.zeros((n_f, n_k, NX, NY, NZ), dtype=np.uint8)
    collab_sem = np.zeros((n_f, NX, NY, NZ), dtype=np.uint8)
    n_vehicles = np.zeros((n_f,), dtype=np.int16)

    tasks = []
    for fi, stem in enumerate(stems):
        zmem = [zip_index[(cav, stem)] for cav in cav_ids] if semantic_zip_path is not None else []
        tasks.append(
            (
                fi,
                stem,
                str(raw_scene_dir),
                cav_ids,
                str(semantic_zip_path) if semantic_zip_path is not None else "",
                zmem,
                str(semantic_scene_dir) if semantic_scene_dir is not None else "",
            )
        )

    if workers > 1 and n_f >= 8:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            for fi, p_f, sp_f, nv_f, ss_f, cs_f in ex.map(_process_one_frame_task, tasks, chunksize=2):
                poses[fi] = p_f
                speeds[fi] = sp_f
                n_vehicles[fi] = nv_f
                sender_sem[fi] = ss_f
                collab_sem[fi] = cs_f
    else:
        for t in tasks:
            fi, p_f, sp_f, nv_f, ss_f, cs_f = _process_one_frame_task(t)
            poses[fi] = p_f
            speeds[fi] = sp_f
            n_vehicles[fi] = nv_f
            sender_sem[fi] = ss_f
            collab_sem[fi] = cs_f

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_npz.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        scene=np.array(raw_scene_dir.name),
        cav_ids=np.array(cav_ids),
        stems=np.array(stems),
        poses=poses,
        speeds=speeds,
        n_vehicles=n_vehicles,
        sender_sem=sender_sem,
        collab_sem=collab_sem,
    )
    if out_npz.exists():
        out_npz.unlink()
    tmp.rename(out_npz)
    return {
        "scene": raw_scene_dir.name,
        "cavs": cav_ids,
        "frames": n_f,
        "npz_bytes": int(out_npz.stat().st_size),
        "classes_present": [int(c) for c in np.unique(collab_sem) if c > 0],
    }
