"""Phase 5 on Mini-200: test-time stress, then seeds 1 and 2.

Budgets, sender counts, weather, pose, latency, and packet loss reuse the
seed-0 B3 and B4 weights. Plots and LaTeX are written by reading the CSV
back, so a number in those files was measured in this process.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from borrowedsight.data import Mini200
from borrowedsight.metrics import finalize_counts, new_counts, update_counts
from borrowedsight.model import BorrowedSight
from borrowedsight.packer import drop_block, packet_layout, unpack_bytes
from borrowedsight.paths import CACHE_CLEAR, CHECKPOINTS, CONFIGS, OUTPUTS
from borrowedsight.registry import log_run
from borrowedsight.weather import lisa_fog, lisa_rain
from tools.extract_sparse_cache import encode_sender
from tools.train_receivers import load_cache, move_sender, preload, train_one

OUT = OUTPUTS / "phase5"
MEASUREMENTS = OUT / "measurements.csv"
COLUMNS = [
    "exp_id",
    "model_name",
    "seed",
    "family",
    "level",
    "bytes_per_frame",
    "bytes_min",
    "bytes_max",
    "n_packets",
    "mIoU",
    "vehicle_IoU",
    "road_IoU",
    "unknown_F1",
    "unobserved_as_free_err",
    "status",
]
BANDWIDTHS = (2048, 4096, 8192, 12288, 16384)
POSES = ((0.0, 0.0), (0.5, 2.0), (1.0, 5.0))
DELAYS = (0, 1, 2, 5)
DROP_RATES = (0.0, 0.1, 0.25, 0.5)
WEATHERS = ("clear", "fog", "rain")
ABLATION_SPECS = (
    ("B4_full", dict(use_envelope=True, use_grouped_books=True, use_quota=True, default_stage=1)),
    ("no_envelope", dict(use_envelope=False, use_grouped_books=True, use_quota=True, default_stage=1)),
    ("shared_codebook", dict(use_envelope=True, use_grouped_books=False, use_quota=True, default_stage=1)),
    ("no_road_quota", dict(use_envelope=True, use_grouped_books=True, use_quota=False, default_stage=1)),
    ("second_code_level", dict(use_envelope=True, use_grouped_books=True, use_quota=True, default_stage=2)),
)


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def load_receiver(method: str, device, seed: int, feat_dim: int, codebook_size: int) -> BorrowedSight:
    path = CHECKPOINTS / (f"{method}_mini.pt" if seed == 0 else f"{method}_mini_seed{seed}.pt")
    model = BorrowedSight(method, feat_dim=feat_dim, codebook_size=codebook_size).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def cache_row(raw: dict) -> dict:
    return {
        "sender_id": int(raw["sender_id"]),
        "pose": np.asarray(raw["pose"], dtype=np.float32),
        "timestamp": float(raw["timestamp"]),
        "speed": float(raw["speed"]),
        "yaw_rate": float(raw["yaw_rate"]),
        "coords": torch.from_numpy(np.asarray(raw["coords"], dtype=np.int64)),
        "feats": torch.from_numpy(np.asarray(raw["feats"], dtype=np.float32)),
        "geo": torch.from_numpy(np.asarray(raw["geo"], dtype=np.float32)),
        "pred_class": torch.from_numpy(np.asarray(raw["pred_class"], dtype=np.int64)),
        "envelope": np.asarray(raw["envelope"], dtype=np.uint8),
    }


def weather_points(points, mode: str, rng: np.random.Generator) -> np.ndarray:
    pts = np.asarray(points)
    if mode == "fog":
        return lisa_fog(pts, rng)
    if mode == "rain":
        degraded, _flags = lisa_rain(pts, rng)
        return degraded
    return pts


def build_weather_cache(model, samples, mode: str, device, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    built = {}
    for key, sample in samples.items():
        rows = []
        for slot, points in enumerate(sample["points"]):
            raw = encode_sender(model, sample, slot, weather_points(points, mode, rng), device)
            rows.append(cache_row(raw))
        built[key] = rows
    return built


def encoders_match(left: BorrowedSight, right: BorrowedSight) -> bool:
    one = left.encoder.state_dict()
    two = right.encoder.state_dict()
    if one.keys() != two.keys():
        return False
    return all(torch.equal(one[key], two[key]) for key in one)


def _senders_for(cache, scene: int, frame: int, delay: int, sender_k):
    source = cache.get((scene, frame - int(delay)), [])
    rows = sorted(source, key=lambda row: int(row["sender_id"]))
    if sender_k is not None:
        rows = rows[: int(sender_k)]
    return rows


def _pose_copy(row, rng, xy_std: float, yaw_std: float):
    copied = dict(row)
    if xy_std == 0.0 and yaw_std == 0.0:
        return copied
    pose = np.array(row["pose"], dtype=np.float32, copy=True)
    pose[0] = pose[0] + rng.normal(0.0, xy_std)
    pose[1] = pose[1] + rng.normal(0.0, xy_std)
    pose[4] = pose[4] + rng.normal(0.0, yaw_std)
    copied["pose"] = pose
    return copied


def _drop_blocks(blob: bytes, drop_p: float, rng) -> bytes:
    if not blob or drop_p <= 0.0:
        return blob
    n_blocks = int(unpack_bytes(blob)["n_blocks"])
    for index in range(n_blocks - 1, -1, -1):
        if rng.random() < drop_p:
            blob = drop_block(blob, index)
    return blob


def measure(model, samples, cache, budget, rng, sender_k=None, xy_std=0.0, yaw_std=0.0, delay=0, drop_p=0.0):
    model.eval()
    acc = new_counts()
    sizes = []
    n_tokens = int(budget["n_tokens"])
    with torch.no_grad():
        for (scene, frame), sample in samples.items():
            pose = torch.as_tensor(sample["ego_pose"], dtype=torch.float32, device=next(model.parameters()).device)
            pose = pose.reshape(1, 6)
            stamp = torch.tensor([float(sample["timestamp"])], dtype=torch.float32, device=pose.device)
            motion = torch.as_tensor(sample["motion"], dtype=torch.float32, device=pose.device).reshape(1, 2)
            blobs = []
            for row in _senders_for(cache, scene, frame, delay, sender_k):
                noisy = _pose_copy(row, rng, xy_std, yaw_std)
                sent = model.transmit(
                    move_sender(noisy, pose.device),
                    pose[0].detach().cpu().numpy(),
                    int(sample["receiver_id"]),
                    n_tokens,
                    budget=budget,
                )
                blob = _drop_blocks(sent["blob"], drop_p, rng)
                if blob:
                    blobs.append(blob)
                    sizes.append(len(blob))
            pred = model(pose, stamp, motion, [blobs])
            sem = pred["sem"].argmax(dim=1)[0].detach().cpu().numpy()
            tri = pred["tri"].argmax(dim=1)[0].detach().cpu().numpy()
            update_counts(acc, sem, tri, sample["gt_semantic"], sample["gt_tristate"])
    metrics = finalize_counts(acc)
    metrics["bytes_per_frame"] = float(np.mean(sizes)) if sizes else 0.0
    metrics["bytes_min"] = float(np.min(sizes)) if sizes else 0.0
    metrics["bytes_max"] = float(np.max(sizes)) if sizes else 0.0
    metrics["n_packets"] = int(len(sizes))
    return metrics


def add_row(rows, method, seed, family, level, metrics):
    rows.append(
        {
            "exp_id": f"{method}_s{seed}_{family}_{level}",
            "model_name": method,
            "seed": int(seed),
            "family": family,
            "level": str(level),
            "bytes_per_frame": metrics.get("bytes_per_frame"),
            "bytes_min": metrics.get("bytes_min"),
            "bytes_max": metrics.get("bytes_max"),
            "n_packets": metrics.get("n_packets"),
            "mIoU": metrics.get("mIoU"),
            "vehicle_IoU": metrics.get("vehicle_IoU"),
            "road_IoU": metrics.get("road_IoU"),
            "unknown_F1": metrics.get("unknown_F1"),
            "unobserved_as_free_err": metrics.get("unobserved_as_free_err"),
            "status": "measured",
        }
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row.get(key)) for key in COLUMNS})


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float_rows(rows, family: str):
    grouped = {"B3": [], "B4": []}
    for row in rows:
        if row["status"] != "measured" or row["family"] != family:
            continue
        if row["model_name"] not in grouped or row["mIoU"] == "":
            continue
        grouped[row["model_name"]].append(row)
    return grouped


def _plot_family(rows, family: str, xlabel: str, x_of, path: Path, ticks=None) -> None:
    grouped = _float_rows(rows, family)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), constrained_layout=True)
    for method, color in (("B3", "#4C566A"), ("B4", "#1F4E79")):
        points = [row for row in grouped[method] if row["mIoU"] != "" and row["vehicle_IoU"] != ""]
        points = sorted(points, key=lambda row: x_of(row))
        xs = [x_of(row) for row in points]
        axes[0].plot(xs, [float(row["mIoU"]) for row in points], marker="o", color=color, label=method)
        axes[1].plot(xs, [float(row["vehicle_IoU"]) for row in points], marker="o", color=color, label=method)
    axes[0].set_ylabel("mIoU")
    axes[1].set_ylabel("vehicle IoU")
    for axis in axes:
        axis.set_xlabel(xlabel)
        axis.legend(frameon=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if ticks:
            axis.set_xticks([item[0] for item in ticks])
            axis.set_xticklabels([item[1] for item in ticks])
    fig.suptitle(family)
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def write_plots(rows) -> None:
    _plot_family(
        rows,
        "bandwidth",
        "bytes per packet",
        lambda row: float(row["bytes_per_frame"]),
        OUT / "bandwidth.png",
    )
    _plot_family(
        rows,
        "senders",
        "senders kept",
        lambda row: float(row["level"]),
        OUT / "senders.png",
    )
    _plot_family(
        rows,
        "weather",
        "weather",
        lambda row: {"clear": 0, "fog": 1, "rain": 2}[row["level"]],
        OUT / "weather.png",
        ticks=((0, "clear"), (1, "fog"), (2, "rain")),
    )
    _plot_family(
        rows,
        "pose",
        "xy noise std (m)",
        lambda row: float(row["level"].split("_")[0]),
        OUT / "pose.png",
    )
    _plot_family(
        rows,
        "latency",
        "delay (frames)",
        lambda row: float(row["level"]),
        OUT / "latency.png",
    )
    _plot_family(
        rows,
        "packet_loss",
        "block drop probability",
        lambda row: float(row["level"]),
        OUT / "packet_loss.png",
    )
    _plot_family(
        rows,
        "seed",
        "seed",
        lambda row: float(row["level"]),
        OUT / "seeds.png",
    )


def _tex_cell(value: str) -> str:
    return "---" if value == "" else value


def write_tex(rows) -> None:
    lines = [
        "% Generated from outputs/phase5/measurements.csv.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llrrrrr}",
        "\\hline",
        "method & level & bytes & mIoU & vehicle & UFE \\\\",
        "\\hline",
    ]
    for row in rows:
        if row["status"] != "measured":
            continue
        lines.append(
            f"{row['model_name']} & {row['family']}={row['level']} & {_tex_cell(row['bytes_per_frame'])} & "
            f"{_tex_cell(row['mIoU'])} & {_tex_cell(row['vehicle_IoU'])} & {_tex_cell(row['unobserved_as_free_err'])} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    (OUT / "stress_tables.tex").write_text("\n".join(lines), encoding="utf-8")


def run_ablations(train_samples, val_samples, cache, cfg, device) -> list[dict]:
    rows = []
    for name, mkw in ABLATION_SPECS:
        print(f"training ablation {name}", flush=True)
        record = train_one("B4", train_samples, val_samples, cache, cfg, device, model_kwargs=mkw, tag=f"B4_{name}")
        m = record["metrics"] or {}
        row = {
            "name": name,
            "status": "measured",
            "bytes_per_frame": _fmt(m.get("bytes_per_frame")),
            "mIoU": _fmt(m.get("mIoU")),
            "vehicle_IoU": _fmt(m.get("vehicle_IoU")),
            "road_IoU": _fmt(m.get("road_IoU")),
            "unknown_F1": _fmt(m.get("unknown_F1")),
            "unobserved_as_free_err": _fmt(m.get("unobserved_as_free_err")),
            "codebook_usage": _fmt(m.get("codebook_usage")),
            "road_token_fraction": _fmt(m.get("road_token_fraction")),
        }
        rows.append(row)
        log_run(
            exp_id=f"B4_ablation_{name}_s{cfg['seed']}",
            model_name=f"B4_{name}",
            seed=int(cfg["seed"]),
            bytes_per_frame=m.get("bytes_per_frame"),
            metrics=m,
            status=f"ablation:{name}",
        )
    return rows


def write_ablations(ab_rows: list[dict]) -> None:
    path = OUT / "ablations.csv"
    cols = [
        "name", "status", "bytes_per_frame", "mIoU", "vehicle_IoU",
        "road_IoU", "unknown_F1", "unobserved_as_free_err", "codebook_usage", "road_token_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        writer.writerows(ab_rows)


def write_summary(rows, ab_rows: list[dict] | None = None) -> None:
    lines = [
        "# Phase 5",
        "",
        "Mini-200 validation (fixed EGO-warped envelope, sensor-origin ray carving, intact VQ gradients).",
        "Numbers below are copied from `measurements.csv` and `ablations.csv` by this script.",
        "",
    ]
    if ab_rows:
        lines.extend([
            "## Core Ablations (Matched 16,384 Serialized Bytes)",
            "",
            "| variant | bytes | mIoU | vehicle IoU | road IoU | unknown F1 | UFE |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for r in ab_rows:
            lines.append(
                f"| {r['name']} | {r['bytes_per_frame']} | {r['mIoU']} | {r['vehicle_IoU']} | "
                f"{r['road_IoU']} | {r['unknown_F1']} | {r['unobserved_as_free_err']} |"
            )
        lines.append("")
    lines.append("| family | method | level | bytes | mIoU | vehicle IoU | UFE |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        if row["status"] != "measured":
            continue
        lines.append(
            f"| {row['family']} | {row['model_name']} | {row['level']} | {row['bytes_per_frame']} | "
            f"{row['mIoU']} | {row['vehicle_IoU']} | {row['unobserved_as_free_err']} |"
        )
    lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def log_rows(rows) -> None:
    for row in rows:
        if row["status"] != "measured":
            continue
        metrics = {
            "mIoU": float(row["mIoU"]) if row["mIoU"] != "" else None,
            "vehicle_IoU": float(row["vehicle_IoU"]) if row["vehicle_IoU"] != "" else None,
            "road_IoU": float(row["road_IoU"]) if row["road_IoU"] != "" else None,
            "unknown_F1": float(row["unknown_F1"]) if row["unknown_F1"] != "" else None,
            "unobserved_as_free_err": float(row["unobserved_as_free_err"]) if row["unobserved_as_free_err"] != "" else None,
        }
        log_run(
            exp_id=row["exp_id"],
            model_name=row["model_name"],
            seed=int(row["seed"]),
            bytes_per_frame=float(row["bytes_per_frame"]) if row["bytes_per_frame"] != "" else None,
            metrics=metrics,
            status=f"{row['family']}:{row['level']}",
        )


def stress(models, samples, clear_cache, weather_caches, device) -> list[dict]:
    rows = []
    for method, model in models.items():
        for target in BANDWIDTHS:
            budget = packet_layout(method, int(target))
            metrics = measure(model, samples, clear_cache, budget, np.random.default_rng(0))
            add_row(rows, method, 0, "bandwidth", target, metrics)
            print(f"{method} bandwidth {target} bytes {metrics['bytes_per_frame']:.4g} mIoU {metrics['mIoU']:.4g}", flush=True)
        full = packet_layout(method, 16384)
        for count in (1, 2, 3):
            metrics = measure(
                model, samples, clear_cache, full, np.random.default_rng(0), sender_k=count
            )
            add_row(rows, method, 0, "senders", count, metrics)
            print(f"{method} senders {count} mIoU {metrics['mIoU']:.4g}", flush=True)
        for mode in WEATHERS:
            metrics = measure(model, samples, weather_caches[mode], full, np.random.default_rng(0))
            add_row(rows, method, 0, "weather", mode, metrics)
            print(f"{method} weather {mode} mIoU {metrics['mIoU']:.4g}", flush=True)
        for xy_std, yaw_std in POSES:
            level = f"{xy_std}_{yaw_std}"
            metrics = measure(
                model,
                samples,
                clear_cache,
                full,
                np.random.default_rng(1),
                xy_std=xy_std,
                yaw_std=yaw_std,
            )
            add_row(rows, method, 0, "pose", level, metrics)
            print(f"{method} pose {level} mIoU {metrics['mIoU']:.4g}", flush=True)
        for delay in DELAYS:
            metrics = measure(model, samples, clear_cache, full, np.random.default_rng(0), delay=delay)
            add_row(rows, method, 0, "latency", delay, metrics)
            print(f"{method} latency {delay} mIoU {metrics['mIoU']:.4g}", flush=True)
        for drop_p in DROP_RATES:
            metrics = measure(
                model,
                samples,
                clear_cache,
                full,
                np.random.default_rng(2),
                drop_p=drop_p,
            )
            add_row(rows, method, 0, "packet_loss", drop_p, metrics)
            print(f"{method} loss {drop_p} bytes {metrics['bytes_per_frame']:.4g} mIoU {metrics['mIoU']:.4g}", flush=True)
    return rows


def extra_seeds(train_samples, val_samples, cache, cfg, device) -> list[dict]:
    rows = []
    for seed in (1, 2):
        cfg_seed = dict(cfg)
        cfg_seed["seed"] = seed
        for method in ("B3", "B4"):
            print(f"training {method} seed {seed}", flush=True)
            record = train_one(method, train_samples, val_samples, cache, cfg_seed, device)
            metrics = record["metrics"] or {}
            metrics = dict(metrics)
            metrics["n_packets"] = ""
            add_row(rows, method, seed, "seed", seed, metrics)
    return rows


def main() -> None:
    with (CONFIGS / "train.yaml").open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    with (CONFIGS / "mini200.yaml").open("r", encoding="utf-8") as handle:
        data_cfg = yaml.safe_load(handle)
    device = torch.device("cuda")
    clear_cache = load_cache(CACHE_CLEAR)
    train_samples = preload(Mini200(data_cfg["root"], "train", int(data_cfg["val_every"])))
    val_samples = preload(Mini200(data_cfg["root"], "val", int(data_cfg["val_every"])))
    feat_dim = int(cfg["feat_dim"])
    codebook_size = int(cfg["codebook_size"])
    models = {
        method: load_receiver(method, device, 0, feat_dim, codebook_size) for method in ("B3", "B4")
    }
    if encoders_match(models["B3"], models["B4"]):
        source = models["B4"]
        weather_caches = {
            mode: build_weather_cache(source, val_samples, mode, device, seed=10 + index)
            for index, mode in enumerate(WEATHERS)
        }
    else:
        weather_caches = {}
        for method, model in models.items():
            weather_caches[method] = {
                mode: build_weather_cache(model, val_samples, mode, device, seed=10 + index)
                for index, mode in enumerate(WEATHERS)
            }
    if "clear" in weather_caches:
        caches_for = {method: weather_caches for method in models}
    else:
        caches_for = weather_caches
    rows = []
    for method, model in models.items():
        print(f"stress {method}", flush=True)
        rows.extend(stress({method: model}, val_samples, clear_cache, caches_for[method], device))
    # Seed 0 at the training budget is already the bandwidth 16384 row. Record it again
    # under family=seed so the seed table has the same measured operating point.
    for row in list(rows):
        if row["family"] == "bandwidth" and row["level"] == "16384":
            copied = dict(row)
            copied["family"] = "seed"
            copied["level"] = "0"
            copied["exp_id"] = f"{row['model_name']}_s0_seed_0"
            rows.append(copied)
    rows.extend(extra_seeds(train_samples, val_samples, clear_cache, cfg, device))
    ab_rows = run_ablations(train_samples, val_samples, clear_cache, cfg, device)
    write_csv(MEASUREMENTS, rows)
    write_ablations(ab_rows)
    stored = read_csv(MEASUREMENTS)
    write_plots(stored)
    write_tex(stored)
    write_summary(stored, ab_rows)
    log_rows(stored)
    print(f"wrote {MEASUREMENTS}", flush=True)


if __name__ == "__main__":
    main()
