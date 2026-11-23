"""Train B0-B4 at one real byte budget and apply the Gate 3 stop rule."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from borrowedsight.data import Mini200, loss_targets
from borrowedsight.labels import GROUND, IGNORE_INDEX, THINGS, VERTICAL
from borrowedsight.metrics import finalize_counts, new_counts, update_counts
from borrowedsight.model import BorrowedSight, score_bce
from borrowedsight.packer import FAIR_BUDGET_BYTES, packet_layout
from borrowedsight.paths import CACHE_CLEAR, CHECKPOINTS, CONFIGS, OUTPUTS
from borrowedsight.registry import log_run


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cache(root: Path) -> dict:
    frames = {}
    for path in sorted(root.glob("shard_*.npz")):
        blob = np.load(path)
        for i in range(len(blob["frame"])):
            start, end = int(blob["offsets"][i]), int(blob["offsets"][i + 1])
            key = (int(blob["scene"][i]), int(blob["frame"][i]))
            frames.setdefault(key, []).append(
                {
                    "sender_id": int(blob["sender_id"][i]),
                    "pose": blob["pose"][i].astype(np.float32),
                    "timestamp": float(blob["timestamp"][i]),
                    "speed": float(blob["speed"][i]),
                    "yaw_rate": float(blob["yaw_rate"][i]),
                    "coords": torch.from_numpy(blob["coords"][start:end].astype(np.int64)),
                    "feats": torch.from_numpy(blob["feats"][start:end].astype(np.float32)),
                    "geo": torch.from_numpy(blob["geo"][start:end].astype(np.float32)),
                    "pred_class": torch.from_numpy(blob["pred_class"][start:end].astype(np.int64)),
                    "envelope": blob["envelope"][i].astype(np.uint8),
                }
            )
    return frames


def move_sender(row, device):
    out = dict(row)
    for key in ("coords", "feats", "geo", "pred_class"):
        out[key] = row[key].to(device)
    return out


def preload(dataset: Mini200) -> dict:
    items = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        items[(int(sample["scene"]), int(sample["frame"]))] = sample
    return items


def step_loss(model, sample, senders, n_tokens, device, budget):
    pose = torch.as_tensor(sample["ego_pose"], dtype=torch.float32, device=device).reshape(1, 6)
    stamp = torch.tensor([float(sample["timestamp"])], dtype=torch.float32, device=device)
    motion = torch.as_tensor(sample["motion"], dtype=torch.float32, device=device).reshape(1, 2)
    sent = []
    for row in senders:
        sent.append(
            model.transmit(
                move_sender(row, device),
                pose[0].detach().cpu().numpy(),
                int(sample["receiver_id"]),
                n_tokens,
                budget=budget,
                serialize=False,
            )
        )
    blobs = [item["blob"] for item in sent if item["blob"]]
    if model.method == "B0" or not blobs:
        pred = model(pose, stamp, motion, [blobs])
        vq = pose.new_zeros(())
        score = pose.new_zeros(())
    else:
        placed = [[model.placed_from_transmit(item, pose[0], float(sample["timestamp"])) for item in sent if item["blob"]]]
        pred = model.render(placed, pose, stamp, motion)
        vq_terms = [item["vq_loss"] for item in sent if torch.is_tensor(item["vq_loss"])]
        vq = torch.stack(vq_terms).mean() if vq_terms else pose.new_zeros(())
        score_terms = []
        for item in sent:
            if item["score_logit"].numel() == 0:
                continue
            score_terms.append(score_bce(item["score_logit"], item["all_classes"]))
        score = torch.stack(score_terms).mean() if score_terms else pose.new_zeros(())
    target = loss_targets(sample)
    sem = target["semantic"].to(device).unsqueeze(0)
    tri = target["tristate"].to(device).unsqueeze(0)
    loss_sem = F.cross_entropy(pred["sem"], sem, ignore_index=IGNORE_INDEX)
    loss_tri = F.cross_entropy(pred["tri"], tri)
    loss = loss_sem + 0.5 * loss_tri + vq + 0.05 * score
    return loss, sent, pred


def evaluate(model, samples, cache, n_tokens, device, budget):
    model.eval()
    acc = new_counts()
    sizes = []
    codes = []
    classes = []
    with torch.no_grad():
        for key, sample in samples.items():
            senders = cache.get(key, [])
            pose = torch.as_tensor(sample["ego_pose"], dtype=torch.float32, device=device).reshape(1, 6)
            stamp = torch.tensor([float(sample["timestamp"])], dtype=torch.float32, device=device)
            motion = torch.as_tensor(sample["motion"], dtype=torch.float32, device=device).reshape(1, 2)
            blobs = []
            for row in senders:
                sent = model.transmit(
                    move_sender(row, device),
                    pose[0].detach().cpu().numpy(),
                    int(sample["receiver_id"]),
                    n_tokens,
                    budget=budget,
                )
                if sent["blob"]:
                    blobs.append(sent["blob"])
                    sizes.append(len(sent["blob"]))
                if sent["code"] is not None and sent["classes"].numel():
                    codes.append(sent["code"].detach().cpu().numpy())
                    classes.append(sent["classes"].detach().cpu().numpy())
            pred = model(pose, stamp, motion, [blobs])
            sem = pred["sem"].argmax(dim=1)[0].detach().cpu().numpy()
            tri = pred["tri"].argmax(dim=1)[0].detach().cpu().numpy()
            update_counts(acc, sem, tri, sample["gt_semantic"], sample["gt_tristate"])
    metrics = finalize_counts(acc)
    metrics["bytes_per_frame"] = float(np.mean(sizes)) if sizes else 0.0
    metrics["bytes_min"] = float(np.min(sizes)) if sizes else 0.0
    metrics["bytes_max"] = float(np.max(sizes)) if sizes else 0.0
    metrics["codebook_usage"] = _usage(
        model.method, codes, classes, model.codebook_size, grouped=model.use_grouped_books
    )
    metrics["road_token_fraction"] = _road_fraction(classes)
    return metrics


def _usage(method, codes, classes, codebook_size: int, grouped: bool = True):
    if method not in {"B3", "B4"} or not codes:
        return None
    code = np.concatenate(codes)
    cls = np.concatenate(classes)
    if method == "B4" and grouped:
        groups = [THINGS, GROUND, VERTICAL]
        denom = float(codebook_size)
    else:
        groups = [None]
        denom = float(codebook_size * (3 if (method == "B4" and not grouped) else 1))
    used = []
    for group in groups:
        mask = np.ones(len(code), dtype=bool) if group is None else np.isin(cls, group)
        if not np.any(mask):
            continue
        used.append(len(np.unique(code[mask])) / denom)
    if not used:
        return None
    return float(np.mean(used))


def _road_fraction(classes) -> float | None:
    if not classes:
        return None
    cls = np.concatenate(classes)
    if len(cls) == 0:
        return None
    return float(np.mean(cls == 5))


def train_one(method, train_samples, val_samples, cache, cfg, device, model_kwargs=None, tag=None):
    set_seed(int(cfg["seed"]))
    mkw = dict(model_kwargs or {})
    model = BorrowedSight(method, feat_dim=int(cfg["feat_dim"]), codebook_size=int(cfg["codebook_size"]), **mkw).to(device)
    ckpt = torch.load(CHECKPOINTS / "sender_mini.pt", map_location=device, weights_only=False)
    model.encoder.load_state_dict(ckpt["encoder"])
    for param in model.encoder.parameters():
        param.requires_grad = False
    params = [param for param in model.parameters() if param.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=1e-4)
    budget = packet_layout(
        method,
        int(cfg.get("fair_bytes", FAIR_BUDGET_BYTES)),
        stage=model.default_stage,
        include_envelope=model.use_envelope,
    )
    n_tokens = int(budget["n_tokens"])
    label = tag or method
    print(
        f"{label} token budget {n_tokens} serialized {budget['serialized']} tail {budget['tail']}",
        flush=True,
    )
    keys = list(train_samples)
    rng = np.random.default_rng(int(cfg["seed"]))
    losses = []
    status = "running"
    started = time.perf_counter()
    limit = float(cfg["max_minutes_receiver"]) * 60.0
    vram_cap = float(cfg["vram_abort_gb"]) * (1024 ** 3)
    torch.cuda.reset_peak_memory_stats()
    step = 0
    if method != "B0":
        model.train()
        while True:
            if time.perf_counter() - started > limit:
                status = "time_cap"
                break
            key = keys[int(rng.integers(0, len(keys)))]
            opt.zero_grad(set_to_none=True)
            loss, _sent, _pred = step_loss(
                model, train_samples[key], cache.get(key, []), n_tokens, device, budget
            )
            if not torch.isfinite(loss):
                status = "nan"
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            step += 1
            peak = torch.cuda.max_memory_allocated()
            if peak >= vram_cap:
                status = "vram_abort"
                break
            if step >= int(cfg["min_steps"]) and len(losses) >= 30:
                base = float(np.mean(losses[:10]))
                recent = float(np.mean(losses[-10:]))
                if base > 0 and recent < float(cfg["loss_drop_ratio"]) * base:
                    status = "loss_dropped"
                    break
            if step % 10 == 0:
                print(f"{label} step {step} loss {losses[-1]:.4f}", flush=True)
    else:
        status = "no_communication"
    metrics = None
    if status != "nan":
        metrics = evaluate(model, val_samples, cache, n_tokens, device, budget)
    record = {
        "method": label,
        "status": status,
        "steps": step,
        "n_tokens": n_tokens,
        "serialized_budget": int(budget["serialized"]),
        "losses": losses[-20:],
        "metrics": metrics,
    }
    payload = {"model": model.state_dict(), "method": method, "model_kwargs": mkw, "status": status, "seed": int(cfg["seed"])}
    torch.save(payload, CHECKPOINTS / f"{label}_mini_seed{int(cfg['seed'])}.pt")
    if int(cfg["seed"]) == 0:
        torch.save(payload, CHECKPOINTS / f"{label}_mini.pt")
    return record


def gate(records: dict) -> dict:
    def metric(method, key):
        item = records[method]["metrics"] or {}
        return item.get(key)

    sizes = [metric(method, "bytes_per_frame") for method in ("B1", "B2", "B3", "B4")]
    sizes = [value for value in sizes if value is not None]
    byte_error = None
    if sizes:
        mean_size = float(np.mean(sizes))
        byte_error = None if mean_size <= 0 else max(abs(a - b) for a in sizes for b in sizes) / mean_size
    b4_iou, b3_iou = metric("B4", "mIoU"), metric("B3", "mIoU")
    b4_veh, b3_veh = metric("B4", "vehicle_IoU"), metric("B3", "vehicle_IoU")
    b4_ufe, b3_ufe = metric("B4", "unobserved_as_free_err"), metric("B3", "unobserved_as_free_err")
    usage = metric("B4", "codebook_usage")
    road = metric("B4", "road_token_fraction")
    reasons = []
    if records["B4"]["status"] == "nan" or records["B3"]["status"] == "nan":
        reasons.append("loss is NaN")
    if usage is None or usage < 0.40:
        reasons.append(f"codebook usage {usage} is below 0.40")
    if road is None or road > 0.60:
        reasons.append(f"road token fraction {road} is above 0.60")
    if byte_error is None or byte_error > 0.02:
        reasons.append(f"byte error {byte_error} is above 0.02")
    win = False
    if not reasons and None not in (b4_iou, b3_iou):
        if b4_iou >= b3_iou + 0.015:
            win = True
        elif None not in (b4_veh, b3_veh, b4_ufe, b3_ufe) and b4_veh >= b3_veh + 0.05 and b4_ufe <= 0.7 * b3_ufe:
            win = True
        elif abs(b4_iou - b3_iou) <= 0.005 and metric("B4", "bytes_per_frame") <= 0.7 * metric("B3", "bytes_per_frame"):
            win = True
        else:
            reasons.append("B4 did not clear a Gate 3 win against B3")
    return {
        "pass": bool(win and not reasons),
        "reasons": reasons,
        "byte_error": byte_error,
        "b4_mIoU": b4_iou,
        "b3_mIoU": b3_iou,
        "b4_vehicle_IoU": b4_veh,
        "b3_vehicle_IoU": b3_veh,
        "b4_ufe": b4_ufe,
        "b3_ufe": b3_ufe,
        "codebook_usage": usage,
        "road_token_fraction": road,
        "decision_rule": "mIoU fraction +0.015, or vehicle IoU +0.05 and UFE <= 70% of B3, or within 0.005 mIoU at <= 70% of the bytes",
    }


def write_summary(records, decision, path: Path) -> None:
    lines = ["# Gate 3", ""]
    lines.append(f"Pass: {decision['pass']}")
    lines.append("")
    lines.append("Decision rule used on the measured fractions: " + decision["decision_rule"])
    lines.append("")
    lines.append("| method | status | bytes | mIoU | vehicle IoU | road IoU | unknown F1 | UFE |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    spans = []
    for method, record in records.items():
        metrics = record["metrics"] or {}

        def cell(key, metrics=metrics):
            value = metrics.get(key)
            return "" if value is None else f"{value:.6g}"

        lines.append(
            f"| {method} | {record['status']} | {cell('bytes_per_frame')} | {cell('mIoU')} | "
            f"{cell('vehicle_IoU')} | {cell('road_IoU')} | {cell('unknown_F1')} | {cell('unobserved_as_free_err')} |"
        )
        if method != "B0" and metrics.get("bytes_min") is not None:
            spans.append(f"{method} packet bytes min {cell('bytes_min')} max {cell('bytes_max')}.")
    lines.append("")
    lines.extend(spans)
    if spans:
        lines.append("")
    lines.append(
        "Byte error, max pairwise gap divided by the mean of the measured B1-B4 packets: "
        f"{decision['byte_error']}"
    )
    lines.append(f"Shared serialized budget: {decision.get('serialized_budget')}")
    lines.append("Token slots a short sender cannot fill are padding. They are not anchors.")
    lines.append(f"B4 codebook usage: {decision['codebook_usage']}")
    lines.append(f"B4 road token fraction: {decision['road_token_fraction']}")
    lines.append("")
    if decision["reasons"]:
        lines.append("Stop reasons:")
        for reason in decision["reasons"]:
            lines.append(f"- {reason}")
        lines.append("")
        lines.append("No further architecture change is applied.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    with (CONFIGS / "train.yaml").open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    with (CONFIGS / "mini200.yaml").open("r", encoding="utf-8") as handle:
        data_cfg = yaml.safe_load(handle)
    device = torch.device("cuda")
    cache = load_cache(CACHE_CLEAR)
    train_samples = preload(Mini200(data_cfg["root"], "train", int(data_cfg["val_every"])))
    val_samples = preload(Mini200(data_cfg["root"], "val", int(data_cfg["val_every"])))
    records = {}
    for method in ("B0", "B1", "B2", "B3", "B4"):
        print(f"training {method}", flush=True)
        record = train_one(method, train_samples, val_samples, cache, cfg, device)
        records[method] = record
        metrics = record["metrics"] or {}
        log_run(
            exp_id=f"{method}_mini_seed{cfg['seed']}",
            model_name=method,
            seed=int(cfg["seed"]),
            bytes_per_frame=metrics.get("bytes_per_frame"),
            metrics=metrics,
            status=record["status"],
        )
        if record["status"] in {"nan", "vram_abort"}:
            break
    needed = ("B0", "B1", "B2", "B3", "B4")
    if all(method in records and records[method]["metrics"] is not None for method in needed):
        decision = gate(records)
        decision["serialized_budget"] = records["B4"].get("serialized_budget")
    else:
        decision = {
            "pass": False,
            "reasons": ["training stopped before every baseline had a measured eval"],
            "byte_error": None,
            "b4_mIoU": None,
            "b3_mIoU": None,
            "b4_vehicle_IoU": None,
            "b3_vehicle_IoU": None,
            "b4_ufe": None,
            "b3_ufe": None,
            "codebook_usage": None,
            "road_token_fraction": None,
            "decision_rule": "incomplete",
            "serialized_budget": None,
        }
    out = OUTPUTS / "phase4_gate3" / "summary.md"
    write_summary(records, decision, out)
    (out.with_suffix(".json")).write_text(json.dumps({"records": records, "decision": decision}, default=str, indent=2), encoding="utf-8")
    print(json.dumps(decision, default=str), flush=True)
    if not decision["pass"]:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
