"""B0-B4 share this module. Only the packet contents change."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualSimVQ

from borrowedsight.labels import GROUND, N_CLASS, ROAD, THINGS, VERTICAL
from borrowedsight.grid import voxel_index_torch
from borrowedsight.messages import coords_to_ego, encode_packet, quota_indices, scatter_values, topk_indices
from borrowedsight.nets import GX, GY, GZ, ReceiverDecoder, SenderEncoder
from borrowedsight.packer import unpack_bytes

GROUP_OF = torch.full((N_CLASS,), -1, dtype=torch.long)
for _cls in THINGS:
    GROUP_OF[_cls] = 0
for _cls in GROUND:
    GROUP_OF[_cls] = 1
for _cls in VERTICAL:
    GROUP_OF[_cls] = 2


def _make_vq(feat_dim: int, codebook_size: int) -> ResidualSimVQ:
    return ResidualSimVQ(
        dim=feat_dim,
        num_quantizers=2,
        codebook_size=codebook_size,
        rotation_trick=True,
        quantize_dropout=True,
        quantize_dropout_cutoff_index=0,
    )


def fp16_ste(values: torch.Tensor) -> torch.Tensor:
    rounded = values.detach().to(torch.float16).to(values.dtype)
    return values + (rounded - values).detach()


def unknown_output(batch: int, device: torch.device) -> dict:
    sem = torch.zeros(batch, N_CLASS, GX, GY, GZ, device=device)
    tri = torch.zeros(batch, 3, GX, GY, GZ, device=device)
    tri[:, 0] = 8.0
    return {"sem": sem, "tri": tri}


class BorrowedSight(nn.Module):
    def __init__(self, method: str = "B4", feat_dim: int = 16, codebook_size: int = 32):
        super().__init__()
        if method not in {"B0", "B1", "B2", "B3", "B4"}:
            raise KeyError(method)
        self.method = method
        self.feat_dim = int(feat_dim)
        self.codebook_size = int(codebook_size)
        self.encoder = SenderEncoder(feat_dim)
        self.decoder = ReceiverDecoder(feat_dim)
        self.fuse = nn.Sequential(
            nn.Linear(feat_dim + 6, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )
        self.score = nn.Linear(feat_dim, 1)
        if method == "B2":
            self.proj = nn.Linear(feat_dim, 8)
            self.lift = nn.Linear(8, feat_dim)
        if method == "B3":
            self.quant = _make_vq(feat_dim, codebook_size)
        if method == "B4":
            self.books = nn.ModuleList([_make_vq(feat_dim, codebook_size) for _ in range(3)])
        self.register_buffer("group_table", GROUP_OF.clone(), persistent=False)

    def forward(self, ego_pose, timestamp, motion, messages):
        pose = ego_pose if ego_pose.ndim == 2 else ego_pose.reshape(1, 6)
        stamp = timestamp.reshape(-1)
        move = motion.reshape(-1, 2)
        batch = int(pose.shape[0])
        packets = _normalize_messages(messages, batch)
        if self.method == "B0":
            return unknown_output(batch, pose.device)
        placed = []
        for b in range(batch):
            item = []
            for blob in packets[b]:
                if not blob:
                    continue
                item.append(self._placed_from_blob(blob, pose[b], float(stamp[b].detach().cpu())))
            placed.append(item)
        if all(len(item) == 0 for item in placed):
            return unknown_output(batch, pose.device)
        return self.render(placed, pose, stamp, move)

    def render(self, placed, pose, stamp, move) -> dict:
        batch = int(pose.shape[0])
        device = pose.device
        feat = pose.new_zeros(batch, self.feat_dim, GX, GY, GZ)
        weight = pose.new_zeros(batch, GX, GY, GZ)
        extra = pose.new_zeros(batch, 7, GX, GY)
        extra[:, 2] = 1.0
        extra[:, 5] = move[:, 0] / 20.0
        extra[:, 6] = move[:, 1] / 20.0
        nonempty = torch.zeros(batch, dtype=torch.bool, device=device)
        for b, items in enumerate(placed):
            if not items:
                continue
            nonempty[b] = True
            for item in items:
                self._paint(feat[b], weight[b], extra[b], item)
        feat = feat / weight.unsqueeze(1).clamp(min=1.0e-6)
        pred = self.decoder(feat, extra)
        return _force_unknown_where(pred, torch.logical_not(nonempty))

    def transmit(self, cached: dict, ego_pose, receiver_id: int, n_tokens: int, stage: int = 1) -> dict:
        """Build one real packet from cached sender features. `recon` matches the bytes."""
        fused = self.fuse(torch.cat((cached["feats"], cached["geo"]), dim=-1))
        logits = self.score(fused).squeeze(-1)
        classes = cached["pred_class"].long()
        usable = classes > 0
        logits = torch.where(usable, logits, torch.full_like(logits, -1.0e4))
        if self.method == "B1":
            salience = torch.where(usable, cached["geo"][:, 0].detach(), torch.full_like(logits, -1.0e4))
            chosen = topk_indices(salience, n_tokens)
        elif self.method == "B4":
            chosen = quota_indices(logits.detach(), classes, n_tokens)
        else:
            chosen = topk_indices(logits.detach(), n_tokens)
        if chosen.numel() > 0:
            chosen = chosen[classes[chosen] > 0]
        fused_k = fused[chosen]
        class_k = classes[chosen]
        coords = cached["coords"][chosen]
        recon, wire, code, vq_loss = self._represent(fused_k, class_k, stage)
        conf = torch.sigmoid(logits[chosen].detach()).clamp(0, 1)
        payload = _numpy_payload(self.method, wire.detach(), code, class_k.detach(), conf, stage)
        envelope = cached.get("envelope") if self.method == "B4" else None
        blob = b""
        if self.method != "B0":
            blob = encode_packet(
                self.method,
                int(cached["sender_id"]),
                int(receiver_id),
                float(cached["timestamp"]),
                cached["pose"],
                float(cached["speed"]),
                float(cached["yaw_rate"]),
                coords.detach().cpu().numpy(),
                payload,
                None if envelope is None else np.asarray(envelope, dtype=np.uint8),
                stage=stage,
            )
        return {
            "blob": blob,
            "recon": recon,
            "coords": coords,
            "conf": conf,
            "classes": class_k,
            "code": code,
            "vq_loss": vq_loss,
            "score_logit": logits,
            "all_classes": classes,
            "pose": cached["pose"],
            "envelope": envelope,
            "timestamp": float(cached["timestamp"]),
        }

    def placed_from_transmit(self, sent: dict, ego_pose, ego_time: float) -> dict:
        coords = sent["coords"]
        device = ego_pose.device if torch.is_tensor(ego_pose) else sent["recon"].device
        origin = coords_to_ego(torch.zeros(1, 3, device=device, dtype=torch.long), sent["pose"], ego_pose)[0]
        values = sent["recon"]
        conf = sent["conf"]
        if coords.numel() == 0:
            ego_idx = torch.zeros(0, 3, dtype=torch.long, device=device)
        else:
            centers = coords_to_ego(coords, sent["pose"], ego_pose)
            ego_idx, valid = voxel_index_torch(centers)
            ego_idx = ego_idx[valid]
            values = values[valid]
            conf = conf[valid]
        age = abs(float(sent.get("timestamp", ego_time)) - float(ego_time))
        return {
            "values": values,
            "coords_ego": ego_idx,
            "conf": conf,
            "envelope": sent.get("envelope"),
            "origin_ego": origin,
            "age": age,
        }

    def _represent(self, fused: torch.Tensor, classes: torch.Tensor, stage: int):
        empty = fused.reshape(0, self.feat_dim)
        if fused.numel() == 0:
            return empty, empty, None, fused.new_zeros(())
        if self.method == "B1":
            wire = fp16_ste(fused)
            return wire, wire, None, fused.new_zeros(())
        if self.method == "B2":
            thin = fp16_ste(self.proj(fused))
            return self.lift(thin), thin, None, fused.new_zeros(())
        if self.method == "B3":
            recon, code, loss = _quantize_book(self.quant, fused, stage)
            return recon, recon, code, loss
        if self.method == "B4":
            recon, code, loss = self._quantize_grouped(fused, classes, stage)
            return recon, recon, code, loss
        return fused, fused, None, fused.new_zeros(())

    def _quantize_grouped(self, fused: torch.Tensor, classes: torch.Tensor, stage: int):
        recon = fused.new_zeros(fused.shape[0], self.feat_dim)
        code = torch.full((fused.shape[0],), 255, dtype=torch.long, device=fused.device)
        loss = fused.new_zeros(())
        groups = self.group_table[classes.clamp(0, N_CLASS - 1)]
        used = 0
        for group_id, book in enumerate(self.books):
            mask = groups == group_id
            if int(mask.sum().item()) == 0:
                continue
            part, part_code, part_loss = _quantize_book(book, fused[mask], stage)
            recon[mask] = part
            code[mask] = part_code
            loss = loss + part_loss
            used += 1
        if used:
            loss = loss / used
        return recon, code, loss

    def _placed_from_blob(self, blob: bytes, ego_pose, ego_time: float) -> dict:
        view = unpack_bytes(blob)
        from borrowedsight.codec import decode_tokens

        tokens = decode_tokens(view["method"], view["tokens"], stage=int(view["stage"]))
        morton = tokens["morton"]
        if morton.size == 0:
            coords = torch.zeros(0, 3, dtype=torch.long, device=ego_pose.device)
            values = ego_pose.new_zeros(0, self.feat_dim)
            conf = ego_pose.new_zeros(0)
        else:
            coords_np = __import__("borrowedsight.codec", fromlist=["demorton3"]).demorton3(morton)
            coords_s = torch.as_tensor(coords_np, dtype=torch.long, device=ego_pose.device)
            centers = coords_to_ego(coords_s, view["pose"], ego_pose.detach().cpu().numpy())
            idx, valid = voxel_index_torch(centers)
            values, conf = self._tensor_from_tokens(tokens, int(view["stage"]))
            coords = idx[valid]
            values = values[valid]
            conf = conf[valid]
        origin = coords_to_ego(
            torch.zeros(1, 3, dtype=torch.long, device=ego_pose.device),
            view["pose"],
            ego_pose.detach().cpu().numpy(),
        )[0]
        return {
            "values": values,
            "coords_ego": coords,
            "conf": conf,
            "envelope": view["envelope"] if view["flags"] % 2 == 1 else None,
            "origin_ego": origin,
            "age": abs(float(view["timestamp_s"]) - float(ego_time)),
        }

    def _tensor_from_tokens(self, tokens: dict, stage: int):
        device = self.group_table.device
        if self.method in {"B1", "B2"}:
            raw = torch.as_tensor(np.asarray(tokens["feat"], dtype=np.float32), device=device)
            if self.method == "B2":
                raw = self.lift(raw)
            return raw, torch.ones(raw.shape[0], device=device)
        code = torch.as_tensor(np.asarray(tokens["code"], dtype=np.int64), device=device)
        if self.method == "B4":
            classes = torch.as_tensor(np.asarray(tokens["cls"], dtype=np.int64), device=device)
            values = self._lookup_grouped(code, classes, stage)
            conf = torch.as_tensor(np.asarray(tokens["conf"], dtype=np.float32), device=device) / 255.0
            return values, conf
        values = _lookup_book(self.quant, code, stage)
        conf_np = tokens.get("conf")
        if conf_np is None:
            conf = torch.ones(code.shape[0], device=device)
        else:
            conf = torch.as_tensor(np.asarray(conf_np, dtype=np.float32), device=device) / 255.0
        return values, conf

    def _lookup_grouped(self, code, classes, stage: int) -> torch.Tensor:
        out = code.new_zeros(code.shape[0], self.feat_dim, dtype=torch.float32)
        groups = self.group_table[classes.clamp(0, N_CLASS - 1)]
        for group_id, book in enumerate(self.books):
            mask = groups == group_id
            if int(mask.sum().item()) == 0:
                continue
            out[mask] = _lookup_book(book, code[mask], stage)
        return out

    def _paint(self, feat, weight, extra, item) -> None:
        envelope = item.get("envelope")
        if envelope is not None and np.asarray(envelope).size == 400:
            _paint_envelope(extra, np.asarray(envelope, dtype=np.uint8))
        if item["values"] is not None and item["coords_ego"].numel() > 0:
            scatter_values(feat, weight, item["coords_ego"], item["values"], item["conf"].clamp(min=1.0e-3))
            self._carve(extra, item)
        age_value = min(float(item.get("age", 0.0)), 5.0) / 5.0
        extra[4] = torch.maximum(extra[4], extra.new_full((), age_value))

    def _carve(self, extra, item) -> None:
        from tools.ray_visibility import ray_visibility

        centers = voxel_centers_from_idx(item["coords_ego"])
        carved = ray_visibility(centers.detach(), item["origin_ego"].detach())
        free = (carved == 1).any(dim=-1).to(dtype=extra.dtype)
        if item.get("envelope") is not None and np.asarray(item["envelope"]).size == 400:
            free = free * extra[0]
        extra[3] = torch.maximum(extra[3], free)


def voxel_centers_from_idx(idx: torch.Tensor) -> torch.Tensor:
    from borrowedsight.grid import voxel_centers_torch

    if idx.numel() == 0:
        return idx.new_zeros(0, 3, dtype=torch.float32)
    return voxel_centers_torch(idx).to(dtype=torch.float32)


def _quantize_book(book: ResidualSimVQ, fused: torch.Tensor, stage: int):
    _rotated, indices, loss = book(fused.float()[None])
    stage_idx = indices[:, :, : int(stage)]
    recon = book.get_output_from_indices(stage_idx)[0]
    code = indices[0, :, 0].to(torch.long)
    return recon.to(fused.dtype), code, loss.mean()


def _lookup_book(book: ResidualSimVQ, code: torch.Tensor, stage: int) -> torch.Tensor:
    idx = code.view(1, -1, 1).to(torch.long)
    return book.get_output_from_indices(idx)[0]


def _paint_envelope(extra: torch.Tensor, flags: np.ndarray) -> None:
    image = flags.reshape(20, 20)
    free = torch.as_tensor(image % 2 == 1, device=extra.device, dtype=extra.dtype)
    occ = torch.as_tensor((image // 2) % 2 == 1, device=extra.device, dtype=extra.dtype)
    unk = torch.as_tensor((image // 4) % 2 == 1, device=extra.device, dtype=extra.dtype)
    free = free.repeat_interleave(5, 0).repeat_interleave(5, 1)
    occ = occ.repeat_interleave(5, 0).repeat_interleave(5, 1)
    unk = unk.repeat_interleave(5, 0).repeat_interleave(5, 1)
    extra[0] = torch.maximum(extra[0], free)
    extra[1] = torch.maximum(extra[1], occ)
    evidence = extra[0] + extra[1]
    extra[2] = torch.where(evidence > 0, torch.minimum(extra[2], unk), extra[2])


def _numpy_payload(method, recon, code, classes, conf, stage: int) -> dict:
    conf_u8 = np.clip(np.rint(conf.detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)
    if method == "B1":
        feat = recon.detach().to(torch.float16).cpu().numpy()
        return {"feat": feat, "conf": conf_u8}
    if method == "B2":
        # The packet stores the 8-d vector, which is `proj` before `lift`.
        # Caller passes that 8-d tensor via recon only for B1. B2 payload is filled
        # by transmit() below if this helper receives the thin vector.
        feat = recon.detach().to(torch.float16).cpu().numpy()
        return {"feat": feat, "conf": conf_u8}
    code_u8 = code.detach().cpu().numpy().astype(np.uint8)
    if method == "B3":
        return {"code": code_u8, "conf": conf_u8}
    return {
        "cls": classes.detach().cpu().numpy().astype(np.uint8),
        "code": code_u8,
        "conf": conf_u8,
        "evidence": np.full(code_u8.shape, 2, dtype=np.uint8),
    }


def _force_unknown_where(pred: dict, empty: torch.Tensor) -> dict:
    if not bool(empty.any().item()):
        return pred
    unknown = unknown_output(pred["sem"].shape[0], pred["sem"].device)
    mask = empty.view(-1, 1, 1, 1, 1)
    return {
        "sem": torch.where(mask, unknown["sem"], pred["sem"]),
        "tri": torch.where(mask, unknown["tri"], pred["tri"]),
    }


def _normalize_messages(messages, batch: int):
    if not messages:
        return [[] for _ in range(batch)]
    if isinstance(messages[0], (bytes, bytearray)):
        if batch != 1:
            raise ValueError("a flat packet list is only valid for one receiver")
        return [list(messages)]
    if len(messages) != batch:
        raise ValueError("message lists must match the batch")
    return [list(item) for item in messages]


def score_bce(logits: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
    target = torch.zeros_like(logits)
    things = torch.isin(classes, torch.as_tensor(THINGS, device=classes.device))
    vertical = torch.isin(classes, torch.as_tensor(VERTICAL, device=classes.device))
    target = torch.where(torch.logical_or(things, vertical), torch.ones_like(target), target)
    road = classes == ROAD
    target = torch.where(road, torch.zeros_like(target), target)
    return F.binary_cross_entropy_with_logits(logits, target)
