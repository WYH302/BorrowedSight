"""Quick 2-minute verification on Mini-200 of our 3 major model.py bug fixes
(EGO-warped envelope, (0,0,0) sensor origin ray carving, rotation-trick VQ gradient flow
& stage-2 code2) plus all 4 core ablation switches.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from borrowedsight.data import Mini200, loss_targets
from borrowedsight.labels import IGNORE_INDEX
from borrowedsight.metrics import finalize_counts, new_counts, update_counts
from borrowedsight.model import BorrowedSight, score_bce
from borrowedsight.packer import FAIR_BUDGET_BYTES, packet_layout
from borrowedsight.paths import CACHE_CLEAR, CHECKPOINTS, CONFIGS, OUTPUTS
from tools.train_receivers import load_cache, move_sender, preload, set_seed

OUT = OUTPUTS / "phase4_gate3_fixed_check"


def run_one(tag: str, method: str, mkw: dict, seed: int, train_samples: dict, val_samples: dict, cache: dict, cfg: dict, device: torch.device, steps: int = 260) -> dict:
    set_seed(seed)
    model = BorrowedSight(method, feat_dim=int(cfg["feat_dim"]), codebook_size=int(cfg["codebook_size"]), **mkw).to(device)
    ckpt = torch.load(CHECKPOINTS / "sender_mini.pt", map_location=device, weights_only=False)
    model.encoder.load_state_dict(ckpt["encoder"])
    for p in model.encoder.parameters():
        p.requires_grad = False
    budget = packet_layout(method, FAIR_BUDGET_BYTES, stage=model.default_stage, include_envelope=model.use_envelope)
    n_tok = int(budget["n_tokens"])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=1e-4)
    keys = list(train_samples)
    rng = np.random.default_rng(seed)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model.train()
    for step in range(steps):
        key = keys[int(rng.integers(0, len(keys)))]
        sample = train_samples[key]
        senders = cache.get(key, [])
        pose = torch.as_tensor(sample["ego_pose"], dtype=torch.float32, device=device).reshape(1, 6)
        stamp = torch.tensor([float(sample["timestamp"])], dtype=torch.float32, device=device)
        motion = torch.as_tensor(sample["motion"], dtype=torch.float32, device=device).reshape(1, 2)
        opt.zero_grad(set_to_none=True)
        sent = [model.transmit(move_sender(r, device), sample["ego_pose"], int(sample["receiver_id"]), n_tok, budget=budget, serialize=False) for r in senders]
        non_empty = [s for s in sent if s["blob"]]
        placed = [[model.placed_from_transmit(s, pose[0], float(sample["timestamp"])) for s in non_empty]]
        pred = model.render(placed, pose, stamp, motion)
        vq_terms = [s["vq_loss"] for s in sent if torch.is_tensor(s["vq_loss"])]
        vq = torch.stack(vq_terms).mean() if vq_terms else pose.new_zeros(())
        sc_terms = [score_bce(s["score_logit"], s["all_classes"]) for s in sent if s["score_logit"].numel() > 0]
        sc = torch.stack(sc_terms).mean() if sc_terms else pose.new_zeros(())
        tgt = loss_targets(sample)
        loss = F.cross_entropy(pred["sem"], tgt["semantic"].to(device).unsqueeze(0), ignore_index=IGNORE_INDEX) + 0.5 * F.cross_entropy(pred["tri"], tgt["tristate"].to(device).unsqueeze(0)) + vq + 0.05 * sc
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    model.eval()
    acc = new_counts()
    sizes = []
    with torch.no_grad():
        for key, sample in val_samples.items():
            pose = torch.as_tensor(sample["ego_pose"], dtype=torch.float32, device=device).reshape(1, 6)
            stamp = torch.tensor([float(sample["timestamp"])], dtype=torch.float32, device=device)
            motion = torch.as_tensor(sample["motion"], dtype=torch.float32, device=device).reshape(1, 2)
            blobs = []
            for r in cache.get(key, []):
                s = model.transmit(move_sender(r, device), sample["ego_pose"], int(sample["receiver_id"]), n_tok, budget=budget, serialize=True)
                if s["blob"]:
                    blobs.append(s["blob"])
                    sizes.append(len(s["blob"]))
            pred = model(pose, stamp, motion, [blobs])
            sem = pred["sem"].argmax(dim=1)[0].detach().cpu().numpy()
            tri = pred["tri"].argmax(dim=1)[0].detach().cpu().numpy()
            tri_p = F.softmax(pred["tri"].float(), dim=1)[0].detach().cpu().numpy()
            update_counts(acc, sem, tri, sample["gt_semantic"], sample["gt_tristate"], tri_probs=tri_p)
    m = finalize_counts(acc)
    m["bytes"] = float(np.mean(sizes)) if sizes else 0.0
    m["peak_vram_mb"] = round(peak_mb, 1)
    m["sec"] = round(time.perf_counter() - t0, 1)
    print(
        f"{tag:18s} bytes={m['bytes']:.0f} mIoU={m['mIoU']:.4f} veh={m['vehicle_IoU']:.4f} "
        f"road={m['road_IoU']:.4f} unkF1={m['unknown_F1']:.4f} UFE={m['unobserved_as_free_err']:.4f} "
        f"peak={peak_mb:.0f}MB ({m['sec']}s)",
        flush=True,
    )
    return m


def main() -> None:
    with (CONFIGS / "train.yaml").open("r", encoding="utf-8") as h:
        cfg = yaml.safe_load(h)
    with (CONFIGS / "mini200.yaml").open("r", encoding="utf-8") as h:
        dcfg = yaml.safe_load(h)
    device = torch.device("cuda")
    cache = load_cache(CACHE_CLEAR)
    train_s = preload(Mini200(dcfg["root"], "train", int(dcfg["val_every"])))
    val_s = preload(Mini200(dcfg["root"], "val", int(dcfg["val_every"])))

    specs = [
        ("B3", "B3", {}),
        ("B4_full", "B4", dict(use_envelope=True, use_grouped_books=True, use_quota=True, default_stage=1)),
        ("B4_no_envelope", "B4", dict(use_envelope=False, use_grouped_books=True, use_quota=True, default_stage=1)),
        ("B4_shared_book", "B4", dict(use_envelope=True, use_grouped_books=False, use_quota=True, default_stage=1)),
        ("B4_no_quota", "B4", dict(use_envelope=True, use_grouped_books=True, use_quota=False, default_stage=1)),
        ("B4_stage2", "B4", dict(use_envelope=True, use_grouped_books=True, use_quota=True, default_stage=2)),
    ]
    res = {}
    for tag, method, mkw in specs:
        res[tag] = run_one(tag, method, mkw, 0, train_s, val_s, cache, cfg, device)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
