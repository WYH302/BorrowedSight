"""B0-B4 share this module. Only the packet contents change."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualSimVQ

from borrowedsight.grid import (
    INTERVALS,
    MIN_BOUND,
    transform_points_torch,
    voxel_centers_torch,
    voxel_index_torch,
    x1_to_x2,
)
from borrowedsight.labels import GROUND, N_CLASS, ROAD, THINGS, VERTICAL
from borrowedsight.messages import coords_to_ego, encode_packet, quota_indices, scatter_values, topk_indices
from borrowedsight.nets import GX, GY, GZ, ReceiverDecoder, SenderEncoder
from borrowedsight.packer import pack_hybrid_env50, unpack_bytes, unpack_zmax25_nibbles

GROUP_OF = torch.full((N_CLASS,), -1, dtype=torch.long)
for _cls in THINGS:
    GROUP_OF[_cls] = 0
for _cls in GROUND:
    GROUP_OF[_cls] = 1
for _cls in VERTICAL:
    GROUP_OF[_cls] = 2

_EGO_BEV_XY_CACHE: dict[str, torch.Tensor] = {}
_ENV_GRID_CACHE: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
_SCORE_TARGET_CPU = torch.zeros(N_CLASS, dtype=torch.float32)
for _cls in THINGS:
    _SCORE_TARGET_CPU[_cls] = 1.0
for _cls in VERTICAL:
    _SCORE_TARGET_CPU[_cls] = 1.0
_SCORE_TARGET_CPU[ROAD] = 0.0
_SCORE_TARGET_CACHE: dict[str, torch.Tensor] = {}

VALID_ENVELOPE_SIZES = {400: 20, 625: 25, 939: 50, 1252: 50, 1877: 50}


def _ego_bev_xy(device: torch.device) -> torch.Tensor:
    key = str(device)
    cached = _EGO_BEV_XY_CACHE.get(key)
    if cached is None:
        ix = torch.arange(GX, dtype=torch.float32, device=device)
        iy = torch.arange(GY, dtype=torch.float32, device=device)
        gx, gy = torch.meshgrid(ix, iy, indexing="ij")
        xs = float(MIN_BOUND[0]) + (gx + 0.5) * float(INTERVALS[0])
        ys = float(MIN_BOUND[1]) + (gy + 0.5) * float(INTERVALS[1])
        zs = torch.zeros_like(xs)
        cached = torch.stack((xs, ys, zs), dim=-1).reshape(-1, 3)
        _EGO_BEV_XY_CACHE[key] = cached
    return cached


def _env_grid_params(device: torch.device, cells: int) -> tuple[torch.Tensor, torch.Tensor]:
    key = (str(device), int(cells))
    cached = _ENV_GRID_CACHE.get(key)
    if cached is None:
        scale = float(GX // int(cells))
        min_xy = torch.as_tensor(MIN_BOUND[:2], dtype=torch.float32, device=device)
        inv_cell = torch.as_tensor(1.0 / (INTERVALS[:2] * scale), dtype=torch.float32, device=device)
        cached = (min_xy, inv_cell)
        _ENV_GRID_CACHE[key] = cached
    return cached


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
    def __init__(
        self,
        method: str = "B4",
        feat_dim: int = 16,
        codebook_size: int = 32,
        use_envelope: bool | None = None,
        use_grouped_books: bool = True,
        use_quota: bool = True,
        default_stage: int = 1,
        envelope_cells: int = 20,
        use_cls_embed: bool = True,
        compact_wire: bool = False,
        true_residual: bool = False,
        use_zmax_env: bool = False,
        use_norm_pe: bool = False,
        use_time_decay: bool = False,
        road_floor: float = 0.0,
        decouple_env: bool = False,
        sender_norm_pe: bool = False,
        sender_max_pool: bool = False,
        use_sem_film: bool = False,
        use_code_fill: bool = False,
        fusion_mode: str = "mean",
        softmax_temp: float = 4.0,
        use_w2c_gate: bool = False,
        use_sarc: bool = False,
    ):
        super().__init__()
        if method not in {"B0", "B1", "B2", "B3", "B4"}:
            raise KeyError(method)
        if int(envelope_cells) not in {20, 25, 50}:
            raise ValueError(f"envelope_cells must be 20, 25, or 50, got {envelope_cells}")
        self.method = method
        self.feat_dim = int(feat_dim)
        self.codebook_size = int(codebook_size)
        self.use_envelope = (method == "B4") if use_envelope is None else bool(use_envelope)
        self.use_grouped_books = bool(use_grouped_books) if method == "B4" else False
        self.use_quota = bool(use_quota) if method == "B4" else False
        self.default_stage = int(default_stage)
        self.envelope_cells = int(envelope_cells)
        self.use_cls_embed = bool(use_cls_embed) if method == "B4" else False
        self.compact_wire = bool(compact_wire)
        self.true_residual = bool(true_residual) if method == "B4" else False
        self.use_zmax_env = bool(use_zmax_env)
        self.use_norm_pe = bool(use_norm_pe)
        self.use_time_decay = bool(use_time_decay)
        self.road_floor = float(road_floor)
        self.decouple_env = bool(decouple_env)
        self.sender_norm_pe = bool(sender_norm_pe)
        self.sender_max_pool = bool(sender_max_pool)
        self.use_sem_film = bool(use_sem_film)
        self.use_code_fill = bool(use_code_fill)
        self.fusion_mode = str(fusion_mode)
        self.softmax_temp = float(softmax_temp)
        self.use_w2c_gate = bool(use_w2c_gate)
        self.use_sarc = bool(use_sarc) if method == "B4" else False
        self.extra_channels = 8 if self.use_zmax_env else 7
        self.encoder = SenderEncoder(
            feat_dim,
            use_norm_pe=self.sender_norm_pe,
            use_max_pool=self.sender_max_pool,
        )
        self.decoder = ReceiverDecoder(
            feat_dim,
            extra_channels=self.extra_channels,
            use_norm_pe=self.use_norm_pe,
            decouple_env=self.decouple_env,
            use_sem_film=self.use_sem_film,
            use_code_fill=self.use_code_fill,
            use_sarc=self.use_sarc,
        )
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
            if self.use_cls_embed:
                self.cls_embed = nn.Embedding(N_CLASS, feat_dim)
            if self.use_grouped_books:
                self.books = nn.ModuleList([_make_vq(feat_dim, codebook_size) for _ in range(3)])
            else:
                # Matched total prototype capacity (3 * codebook_size) in a single shared book.
                self.quant = _make_vq(feat_dim, codebook_size * 3)
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

    def build_inputs(self, placed, pose, stamp, move) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = int(pose.shape[0])
        device = pose.device
        feat = pose.new_zeros(batch, self.feat_dim, GX, GY, GZ)
        weight = pose.new_zeros(batch, GX, GY, GZ)
        extra = pose.new_zeros(batch, self.extra_channels, GX, GY)
        extra[:, 2] = 1.0
        extra[:, 5] = (move[:, 0] / 20.0).view(batch, 1, 1)
        extra[:, 6] = (move[:, 1] / 20.0).view(batch, 1, 1)
        nonempty = torch.zeros(batch, dtype=torch.bool, device=device)
        for b, items in enumerate(placed):
            for item in items:
                if not _item_has_evidence(item):
                    continue
                nonempty[b] = True
                self._paint(feat[b], weight[b], extra[b], item)
        feat = feat / weight.unsqueeze(1).clamp(min=1.0e-6)
        return feat, extra, nonempty

    def build_cls_prior(self, placed, pose) -> torch.Tensor | None:
        """Build [B, N_CLASS, GX, GY, GZ] normalized wire semantic class prior for SARC (B4 only)."""
        if not self.use_sarc:
            return None
        from borrowedsight.grid import linear_index

        batch = int(pose.shape[0])
        n_vox = GX * GY * GZ
        prior = pose.new_zeros(batch, N_CLASS, GX, GY, GZ, dtype=torch.float32)
        for b, items in enumerate(placed):
            vote = pose.new_zeros(N_CLASS, n_vox, dtype=torch.float32)
            fresh = pose.new_zeros(n_vox, dtype=torch.float32)
            has_any = False
            for item in items:
                cls = item.get("cls")
                idx = item.get("coords_ego")
                if cls is None or idx is None or idx.numel() == 0:
                    continue
                has_any = True
                age = float(item.get("age", 0.0))
                age_decay = float(np.exp(-age / 0.5)) if (self.use_time_decay and age > 0.0) else 1.0
                w = item["conf"].to(torch.float32).clamp(min=1.0e-3) * age_decay
                lin = linear_index(idx)
                oh = F.one_hot(cls.to(torch.long).clamp(0, N_CLASS - 1), num_classes=N_CLASS).to(torch.float32)
                vote.index_add_(1, lin, (oh * w.unsqueeze(1)).transpose(0, 1))
                fresh.scatter_reduce_(0, lin, torch.full_like(w, age_decay), reduce="amax", include_self=True)
            if has_any:
                vsum = vote.sum(dim=0, keepdim=True)
                normed = torch.where(vsum > 0, (vote / vsum.clamp(min=1.0e-6)) * fresh.unsqueeze(0), torch.zeros_like(vote))
                prior[b] = normed.view(N_CLASS, GX, GY, GZ)
        return prior

    def render(self, placed, pose, stamp, move) -> dict:
        feat, extra, nonempty = self.build_inputs(placed, pose, stamp, move)
        cls_prior = self.build_cls_prior(placed, pose)
        pred = self.decoder(feat, extra, cls_prior=cls_prior)
        return _force_unknown_where(pred, torch.logical_not(nonempty))

    def transmit(
        self,
        cached: dict,
        ego_pose,
        receiver_id: int,
        n_tokens: int,
        stage: int | None = None,
        budget=None,
        serialize: bool = True,
    ) -> dict:
        """Build one real packet from cached sender features. `recon` matches the bytes."""
        stage_n = self.default_stage if stage is None else int(stage)
        fused = self.fuse(torch.cat((cached["feats"], cached["geo"]), dim=-1))
        logits = self.score(fused).squeeze(-1)
        classes = cached["pred_class"].long()
        usable = classes > 0
        logits = torch.where(usable, logits, torch.full_like(logits, -1.0e4))
        pad_slots = None
        tail_bytes = 0
        if budget is not None and self.method != "B0":
            pad_slots = int(budget["n_tokens"])
            tail_bytes = int(budget["tail"])
            n_tokens = pad_slots
        if self.method == "B1":
            salience = torch.where(usable, cached["geo"][:, 0].detach(), torch.full_like(logits, -1.0e4))
            chosen = topk_indices(salience, n_tokens)
        elif self.method == "B4" and self.use_quota:
            chosen = quota_indices(logits.detach(), classes, n_tokens, road_floor=self.road_floor)
        else:
            chosen = topk_indices(logits.detach(), n_tokens)
        if chosen.numel() > 0:
            chosen = chosen[classes[chosen] > 0]
        fused_k = fused[chosen]
        class_k = classes[chosen]
        coords = cached["coords"][chosen]
        gate_k = torch.sigmoid(logits[chosen]) if chosen.numel() else fused.new_zeros(0)
        recon, wire, codes, vq_loss = self._represent(fused_k, class_k, stage_n, gate=gate_k)
        code_l0 = None if codes is None else codes[:, 0]
        if self.method == "B2" and self.use_w2c_gate:
            conf = recon.detach().norm(dim=-1).clamp(min=1.0e-3) if chosen.numel() else fused.new_zeros(0)
        else:
            conf = gate_k.detach().clamp(0, 1) if chosen.numel() else fused.new_zeros(0)
        if self.use_envelope:
            if self.envelope_cells == 50 and cached.get("envelope50") is not None:
                envelope = pack_hybrid_env50(cached["envelope50"], cached.get("envelope25"))
            elif self.envelope_cells >= 25 and cached.get("envelope25") is not None:
                envelope = cached["envelope25"]
            else:
                envelope = cached.get("envelope")
        else:
            envelope = None
        blob = b""
        if serialize and self.method != "B0" and (wire.numel() > 0 or pad_slots):
            if wire.numel() > 0:
                payload = _numpy_payload(self.method, wire.detach(), codes, class_k.detach(), conf, stage_n)
                coords_np = coords.detach().cpu().numpy()
            else:
                payload = {}
                coords_np = np.zeros((0, 3), dtype=np.int64)
            blob = encode_packet(
                self.method,
                int(cached["sender_id"]),
                int(receiver_id),
                float(cached["timestamp"]),
                cached["pose"],
                float(cached["speed"]),
                float(cached["yaw_rate"]),
                coords_np,
                payload,
                None if envelope is None else np.asarray(envelope, dtype=np.uint8),
                stage=stage_n,
                pad_slots=pad_slots,
                tail_bytes=tail_bytes,
                include_envelope=self.use_envelope,
                compact=self.compact_wire,
            )
        elif not serialize and self.method != "B0" and (wire.numel() > 0 or pad_slots):
            blob = b"\x01"
        return {
            "blob": blob,
            "recon": recon,
            "coords": coords,
            "conf": conf,
            "classes": class_k,
            "code": code_l0,
            "codes": codes,
            "vq_loss": vq_loss,
            "score_logit": logits,
            "all_classes": classes,
            "pose": cached["pose"],
            "envelope": envelope,
            "timestamp": float(cached["timestamp"]),
            "stage": stage_n,
        }

    def placed_from_transmit(self, sent: dict, ego_pose, ego_time: float) -> dict:
        coords = sent["coords"]
        device = ego_pose.device if torch.is_tensor(ego_pose) else sent["recon"].device
        ego_np = ego_pose.detach().cpu().numpy() if torch.is_tensor(ego_pose) else np.asarray(ego_pose, dtype=np.float32)
        sender_np = np.asarray(sent["pose"], dtype=np.float32)
        origin = transform_points_torch(
            torch.zeros(1, 3, device=device, dtype=torch.float64), sender_np, ego_np
        )[0].to(torch.float32)
        values = sent["recon"]
        conf = sent["conf"]
        cls_t = sent.get("classes") if self.method == "B4" else None
        if coords.numel() == 0:
            ego_idx = torch.zeros(0, 3, dtype=torch.long, device=device)
            if cls_t is not None:
                cls_t = torch.zeros(0, dtype=torch.long, device=device)
        else:
            centers = coords_to_ego(coords, sender_np, ego_np)
            ego_idx, valid = voxel_index_torch(centers)
            ego_idx = ego_idx[valid]
            values = values[valid]
            conf = conf[valid]
            if cls_t is not None:
                cls_t = cls_t[valid]
        age = abs(float(sent.get("timestamp", ego_time)) - float(ego_time))
        return {
            "values": values,
            "coords_ego": ego_idx,
            "conf": conf,
            "cls": cls_t,
            "envelope": sent.get("envelope"),
            "sender_pose": sender_np,
            "ego_pose": ego_np,
            "origin_ego": origin,
            "age": age,
        }

    def _represent(self, fused: torch.Tensor, classes: torch.Tensor, stage: int, gate: torch.Tensor | None = None):
        if fused.numel() == 0:
            empty = fused.reshape(0, self.feat_dim)
            return empty, empty, None, fused.new_zeros(())
        if self.method == "B1":
            wire = fp16_ste(fused)
            return wire, wire, None, fused.new_zeros(())
        if self.method == "B2":
            projected = self.proj(fused)
            if self.use_w2c_gate and gate is not None:
                projected = projected * gate.unsqueeze(-1).to(projected.dtype)
            thin = fp16_ste(projected)
            return self.lift(thin), thin, None, fused.new_zeros(())
        if self.method == "B3":
            recon, codes, loss = _quantize_book(self.quant, fused, stage)
            return recon, recon, codes, loss
        if self.method == "B4":
            if self.use_cls_embed and self.true_residual:
                cls_proto = self.cls_embed(classes.clamp(0, N_CLASS - 1)).to(fused.dtype)
                res_in = fused - cls_proto
                if self.use_grouped_books:
                    recon_res, codes, loss = self._quantize_grouped(res_in, classes, stage)
                else:
                    recon_res, codes, loss = _quantize_book(self.quant, res_in, stage)
                recon = cls_proto + recon_res
                return recon, recon, codes, loss
            if self.use_grouped_books:
                recon, codes, loss = self._quantize_grouped(fused, classes, stage)
            else:
                recon, codes, loss = _quantize_book(self.quant, fused, stage)
            # Factorized Category-Geometry reconstruction: add the semantic class embedding
            # corresponding to the 1-byte `cls` field already transmitted in every B4 wire token.
            if self.use_cls_embed:
                recon = recon + self.cls_embed(classes.clamp(0, N_CLASS - 1)).to(recon.dtype)
            return recon, recon, codes, loss
        return fused, fused, None, fused.new_zeros(())

    def _quantize_grouped(self, fused: torch.Tensor, classes: torch.Tensor, stage: int):
        stage_n = max(1, min(int(stage), 2))
        groups = self.group_table[classes.clamp(0, N_CLASS - 1)]
        n = int(fused.shape[0])
        n_grp = len(self.books)
        valid = groups >= 0
        valid_f = valid.unsqueeze(-1).to(torch.float32)
        safe_g = groups.clamp(0, n_grp - 1)
        residual = fused.float()
        recon_f = torch.zeros_like(residual)
        row_idx = torch.arange(n, device=fused.device)
        all_codes = []

        if not self.training and not torch.is_grad_enabled():
            for q_idx in range(stage_n):
                cb_all = torch.stack([book.layers[q_idx].codebook for book in self.books], dim=0)  # [G, K, D]
                cb_n2 = (cb_all * cb_all).sum(dim=-1)  # [G, K]
                dots = (residual @ cb_all.reshape(-1, residual.shape[-1]).T).view(n, n_grp, -1)
                idx = (cb_n2[safe_g] - 2.0 * dots[row_idx, safe_g]).argmin(dim=-1)
                q = cb_all[safe_g, idx] * valid_f
                residual = residual - q
                recon_f = recon_f + q
                all_codes.append(torch.where(valid, idx, torch.full_like(idx, 255)))
            return recon_f.to(fused.dtype), torch.stack(all_codes, dim=-1), fused.new_zeros(())

        # Zero-sync vectorized grouped ResidualSimVQ training (100% bit-exact parity with per-group loop,
        # eliminating GPU->CPU synchronization stalls from boolean mask indexing and .any().item()).
        from vector_quantize_pytorch.sim_vq import rotate_to

        all_losses = []
        cw = float(self.books[0].layers[0].commitment_weight)
        icw = float(self.books[0].layers[0].input_to_quantize_commit_loss_weight)
        valid_1d = valid.to(torch.float32)
        for q_idx in range(stage_n):
            cb_all = torch.stack([book.layers[q_idx].codebook for book in self.books], dim=0)  # [G, K, D]
            with torch.no_grad():
                cb_n2 = (cb_all * cb_all).sum(dim=-1)  # [G, K]
                dots = (residual @ cb_all.reshape(-1, residual.shape[-1]).T).view(n, n_grp, -1)
                idx = (cb_n2[safe_g] - 2.0 * dots[row_idx, safe_g]).argmin(dim=-1)
            q_raw = cb_all[safe_g, idx]
            tok_err = (
                (residual.detach() - q_raw).pow(2).mean(dim=-1)
                + (residual - q_raw.detach()).pow(2).mean(dim=-1) * icw
            ) * valid_1d
            g_sum = residual.new_zeros(n_grp).scatter_add_(0, safe_g, tok_err)
            g_cnt = residual.new_zeros(n_grp).scatter_add_(0, safe_g, valid_1d)
            g_has = (g_cnt > 0).to(torch.float32)
            q_loss = ((g_sum / g_cnt.clamp(min=1.0)) * g_has).sum() / g_has.sum().clamp(min=1.0) * cw
            q_rot = rotate_to(residual.unsqueeze(0), q_raw.unsqueeze(0)).squeeze(0) * valid_f
            residual = residual - q_rot.detach()
            recon_f = recon_f + q_rot
            all_codes.append(torch.where(valid, idx, torch.full_like(idx, 255)))
            all_losses.append(q_loss)
        return recon_f.to(fused.dtype), torch.stack(all_codes, dim=-1), torch.stack(all_losses).mean()

    def _placed_from_blob(
        self,
        blob: bytes,
        ego_pose,
        ego_time: float,
        prev_packets_veh_w: list[tuple[np.ndarray, float]] | None = None,
    ) -> dict:
        view = unpack_bytes(blob)
        from borrowedsight.codec import decode_tokens, demorton3

        stage = int(view["stage"])
        compact = bool(view.get("compact", False))
        tokens = decode_tokens(view["method"], view["tokens"], stage=stage, compact=compact)
        morton = tokens["morton"]
        ego_np = ego_pose.detach().cpu().numpy() if torch.is_tensor(ego_pose) else np.asarray(ego_pose, dtype=np.float32)
        sender_np = np.asarray(view["pose"], dtype=np.float32)
        age = abs(float(view["timestamp_s"]) - float(ego_time))
        if morton.size == 0:
            coords = torch.zeros(0, 3, dtype=torch.long, device=ego_pose.device)
            values = ego_pose.new_zeros(0, self.feat_dim)
            conf = ego_pose.new_zeros(0)
            cls_t = torch.zeros(0, dtype=torch.long, device=ego_pose.device) if view["method"] == "B4" else None
        else:
            coords_np = demorton3(morton)
            if (
                prev_packets_veh_w
                and age > 1.0e-3
                and view["method"] == "B4"
                and "cls" in tokens
            ):
                from borrowedsight.kinematics import warp_tokens_ss_kfw

                centers_np, _ = warp_tokens_ss_kfw(
                    coords_np,
                    tokens["cls"],
                    sender_np,
                    ego_np,
                    dt_delay=age,
                    prev_packets_veh_w=prev_packets_veh_w,
                )
                centers = torch.as_tensor(centers_np, dtype=torch.float32, device=ego_pose.device)
            else:
                coords_s = torch.as_tensor(coords_np, dtype=torch.long, device=ego_pose.device)
                centers = coords_to_ego(coords_s, sender_np, ego_np)
            idx, valid = voxel_index_torch(centers)
            values, conf = self._tensor_from_tokens(tokens, stage)
            coords = idx[valid]
            values = values[valid]
            conf = conf[valid]
            cls_t = (
                torch.as_tensor(np.asarray(tokens["cls"], dtype=np.int64), device=ego_pose.device)[valid]
                if (view["method"] == "B4" and "cls" in tokens)
                else None
            )
        origin = transform_points_torch(
            torch.zeros(1, 3, device=ego_pose.device, dtype=torch.float64), sender_np, ego_np
        )[0].to(torch.float32)
        return {
            "values": values,
            "coords_ego": coords,
            "conf": conf,
            "cls": cls_t,
            "envelope": view["envelope"] if view["flags"] % 2 == 1 else None,
            "sender_pose": sender_np,
            "ego_pose": ego_np,
            "origin_ego": origin,
            "age": age,
        }

    def _tensor_from_tokens(self, tokens: dict, stage: int):
        device = self.group_table.device
        if self.method == "B2":
            raw = torch.as_tensor(np.asarray(tokens["feat"], dtype=np.float32), device=device)
            raw = self.lift(raw)
            if self.use_w2c_gate:
                conf = raw.detach().norm(dim=-1).clamp(min=1.0e-3)
                return raw, conf
            return raw, torch.ones(raw.shape[0], device=device)
        if self.method == "B1":
            raw = torch.as_tensor(np.asarray(tokens["feat"], dtype=np.float32), device=device)
            return raw, torch.ones(raw.shape[0], device=device)
        code0 = torch.as_tensor(np.asarray(tokens["code"], dtype=np.int64), device=device)
        if int(stage) > 1 and "code2" in tokens:
            code1 = torch.as_tensor(np.asarray(tokens["code2"], dtype=np.int64), device=device)
            codes = torch.stack((code0, code1), dim=-1)
        else:
            codes = code0.unsqueeze(-1)
        if self.method == "B4":
            classes = torch.as_tensor(np.asarray(tokens["cls"], dtype=np.int64), device=device)
            if self.use_grouped_books:
                values = self._lookup_grouped(codes, classes, stage)
            else:
                values = _lookup_book(self.quant, codes, stage)
            if self.use_cls_embed:
                values = values + self.cls_embed(classes.clamp(0, N_CLASS - 1)).to(values.dtype)
            conf = torch.as_tensor(np.asarray(tokens["conf"], dtype=np.float32), device=device) / 255.0
            return values, conf
        values = _lookup_book(self.quant, codes, stage)
        conf_np = tokens.get("conf")
        if conf_np is None:
            conf = torch.ones(code0.shape[0], device=device)
        else:
            conf = torch.as_tensor(np.asarray(conf_np, dtype=np.float32), device=device) / 255.0
        return values, conf

    def _lookup_grouped(self, codes, classes, stage: int) -> torch.Tensor:
        stage_n = max(1, min(int(stage), 2))
        groups = self.group_table[classes.clamp(0, N_CLASS - 1)]
        valid = groups >= 0
        safe_g = groups.clamp(0, len(self.books) - 1)
        idx = codes.unsqueeze(-1) if codes.ndim == 1 else codes
        cb0 = torch.stack([book.layers[0].codebook for book in self.books], dim=0)
        out = cb0[safe_g, idx[:, 0].to(torch.long)]
        if stage_n > 1:
            cb1 = torch.stack([book.layers[1].codebook for book in self.books], dim=0)
            out = out + cb1[safe_g, idx[:, 1].to(torch.long)]
        return (out * valid.unsqueeze(-1).to(out.dtype)).to(torch.float32)

    def _paint(self, feat, weight, extra, item) -> None:
        envelope = item.get("envelope")
        env_free_ego = None
        if envelope is not None and int(np.asarray(envelope).size) in VALID_ENVELOPE_SIZES:
            env_free_ego = _paint_envelope(
                extra,
                np.asarray(envelope, dtype=np.uint8),
                item.get("sender_pose"),
                item.get("ego_pose"),
            )
        age = float(item.get("age", 0.0))
        if item["values"] is not None and item["coords_ego"].numel() > 0:
            w = item["conf"].clamp(min=1.0e-3)
            if self.use_time_decay and age > 0.0:
                w = w * float(np.exp(-age / 0.5))
            if self.fusion_mode == "softmax":
                # Temperature-scaled exponential competition across overlapping senders at the same voxel:
                # sharply favors fresher, higher-confidence observations under pose/latency perturbations.
                w = torch.exp((w * float(self.softmax_temp)).clamp(max=12.0))
            scatter_values(feat, weight, item["coords_ego"], item["values"], w)
            self._carve(extra, item, env_free_ego=env_free_ego)
        age_value = min(age, 1.0) if self.use_time_decay else (min(age, 5.0) / 5.0)
        extra[4] = torch.maximum(extra[4], extra.new_full((), age_value))

    def _carve(self, extra, item, env_free_ego: torch.Tensor | None = None) -> None:
        from tools.ray_visibility import ray_visibility

        centers = voxel_centers_from_idx(item["coords_ego"])
        carved = ray_visibility(centers.detach(), item["origin_ego"].detach())
        free = (carved == 1).any(dim=-1).to(dtype=extra.dtype)
        if env_free_ego is not None:
            free = free * env_free_ego
        extra[3] = torch.maximum(extra[3], free)


def _item_has_evidence(item) -> bool:
    """Padding-only packets are not anchors and not free-space evidence."""
    coords = item.get("coords_ego")
    if torch.is_tensor(coords) and coords.numel() > 0:
        return True
    envelope = item.get("envelope")
    if envelope is None:
        return False
    return int(np.asarray(envelope).size) in VALID_ENVELOPE_SIZES


def voxel_centers_from_idx(idx: torch.Tensor) -> torch.Tensor:
    if idx.numel() == 0:
        return idx.new_zeros(0, 3, dtype=torch.float32)
    return voxel_centers_torch(idx).to(dtype=torch.float32)


def _quantize_book(book: ResidualSimVQ, fused: torch.Tensor, stage: int):
    """Run exact ResidualSimVQ layers 0..stage-1, keeping rotation-trick gradients."""
    stage_n = max(1, min(int(stage), len(book.layers)))
    if not book.training and not torch.is_grad_enabled():
        residual = fused.float()
        quantized_out = torch.zeros_like(residual)
        all_indices = []
        for q_idx in range(stage_n):
            cb = book.layers[q_idx].codebook
            idx = torch.cdist(residual, cb).argmin(dim=-1)
            q = cb[idx]
            residual = residual - q
            quantized_out = quantized_out + q
            all_indices.append(idx)
        return quantized_out.to(fused.dtype), torch.stack(all_indices, dim=-1), fused.new_zeros(())
    x = fused.float()[None]
    residual = x
    quantized_out = x.new_zeros(x.shape)
    all_indices = []
    all_losses = []
    for q_idx in range(stage_n):
        q, embed_idx, q_loss = book.layers[q_idx](residual)
        residual = residual - q.detach()
        quantized_out = quantized_out + q
        all_indices.append(embed_idx[0].to(torch.long))
        all_losses.append(q_loss)
    codes = torch.stack(all_indices, dim=-1)
    loss = torch.stack(all_losses).mean()
    return quantized_out[0].to(fused.dtype), codes, loss


def _lookup_book(book: ResidualSimVQ, codes: torch.Tensor, stage: int) -> torch.Tensor:
    stage_n = max(1, min(int(stage), len(book.layers)))
    idx = codes.unsqueeze(-1) if codes.ndim == 1 else codes
    out = book.layers[0].codebook[idx[:, 0].to(torch.long)]
    for q_idx in range(1, stage_n):
        out = out + book.layers[q_idx].codebook[idx[:, q_idx].to(torch.long)]
    return out.to(torch.float32)


def _unpack_env50_bp(raw939: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw939, dtype=np.uint8).reshape(-1)
    f = np.unpackbits(arr[0:313])[:2500].reshape(50, 50)
    o = np.unpackbits(arr[313:626])[:2500].reshape(50, 50)
    u = np.unpackbits(arr[626:939])[:2500].reshape(50, 50)
    return (f + o * 2 + u * 4).astype(np.uint8)


def _paint_envelope(
    extra: torch.Tensor,
    flags: np.ndarray,
    sender_pose=None,
    ego_pose=None,
) -> torch.Tensor:
    """Warp sender-frame 20x20, 25x25, or 50x50 (+ optional z_max) visibility envelope into EGO 100x100 grid."""
    flat = np.asarray(flags, dtype=np.uint8).reshape(-1)
    size = int(flat.size)
    if size not in VALID_ENVELOPE_SIZES:
        raise ValueError(f"unsupported envelope size {size}; expected one of {sorted(VALID_ENVELOPE_SIZES)}")
    zmax_grid_np = None
    zmax_cells = 0
    if size in {939, 1252, 1877}:
        cells = 50
        img_np = _unpack_env50_bp(flat[:939])
        if size in {1252, 1877}:
            zmax_grid_np = unpack_zmax25_nibbles(flat[939:1252])
            zmax_cells = 25
    else:
        cells = VALID_ENVELOPE_SIZES[size]
        img_np = flat.reshape(cells, cells)
        zmax_grid_np = ((img_np >> np.uint8(3)) & np.uint8(0x07)).astype(np.uint8)
        zmax_cells = cells

    scale = GX // cells
    image = torch.as_tensor(img_np, device=extra.device, dtype=torch.int64)
    want_zmax = int(extra.shape[0]) >= 8 and zmax_grid_np is not None
    zmax_ego = None
    if sender_pose is None or ego_pose is None:
        free = ((image & 1) != 0).to(dtype=extra.dtype).repeat_interleave(scale, 0).repeat_interleave(scale, 1)
        occ = ((image & 2) != 0).to(dtype=extra.dtype).repeat_interleave(scale, 0).repeat_interleave(scale, 1)
        unk = ((image & 4) != 0).to(dtype=extra.dtype).repeat_interleave(scale, 0).repeat_interleave(scale, 1)
        if want_zmax:
            zm_scale = GX // zmax_cells
            zm_t = torch.as_tensor(zmax_grid_np, device=extra.device, dtype=extra.dtype) / 7.0
            zmax_ego = zm_t.repeat_interleave(zm_scale, 0).repeat_interleave(zm_scale, 1)
    else:
        # Backward warp: transform EGO BEV cell centers into the sender's local frame.
        ego_pts = _ego_bev_xy(extra.device)
        mat = torch.as_tensor(x1_to_x2(ego_pose, sender_pose), dtype=torch.float32, device=extra.device)
        pts_s = ego_pts @ mat[:3, :3].T + mat[:3, 3]
        min_xy, inv_cell = _env_grid_params(extra.device, cells)
        cxy = torch.floor((pts_s[:, :2] - min_xy) * inv_cell).to(torch.long)
        valid = (cxy[:, 0] >= 0) & (cxy[:, 0] < cells) & (cxy[:, 1] >= 0) & (cxy[:, 1] < cells)
        cx = cxy[:, 0].clamp(0, cells - 1)
        cy = cxy[:, 1].clamp(0, cells - 1)
        sampled = image[cx, cy]
        free = (((sampled & 1) != 0) & valid).view(GX, GY).to(dtype=extra.dtype)
        occ = (((sampled & 2) != 0) & valid).view(GX, GY).to(dtype=extra.dtype)
        unk = ((((sampled & 4) != 0) & valid) | (~valid)).view(GX, GY).to(dtype=extra.dtype)
        if want_zmax:
            zm_t = torch.as_tensor(zmax_grid_np, device=extra.device, dtype=extra.dtype) / 7.0
            if zmax_cells == cells:
                zmax_ego = (zm_t[cx, cy] * valid.to(dtype=extra.dtype)).view(GX, GY)
            else:
                min_z, inv_z = _env_grid_params(extra.device, zmax_cells)
                zxy = torch.floor((pts_s[:, :2] - min_z) * inv_z).to(torch.long)
                zvalid = (zxy[:, 0] >= 0) & (zxy[:, 0] < zmax_cells) & (zxy[:, 1] >= 0) & (zxy[:, 1] < zmax_cells)
                zx = zxy[:, 0].clamp(0, zmax_cells - 1)
                zy = zxy[:, 1].clamp(0, zmax_cells - 1)
                zmax_ego = (zm_t[zx, zy] * zvalid.to(dtype=extra.dtype)).view(GX, GY)
    extra[0] = torch.maximum(extra[0], free)
    extra[1] = torch.maximum(extra[1], occ)
    evidence = free + occ
    extra[2] = torch.where(evidence > 0, torch.minimum(extra[2], unk), extra[2])
    if zmax_ego is not None:
        extra[7] = torch.maximum(extra[7], zmax_ego)
    return free


def _numpy_payload(method, recon, codes, classes, conf, stage: int) -> dict:
    conf_u8 = np.clip(np.rint(conf.detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)
    if method == "B1":
        feat = recon.detach().to(torch.float16).cpu().numpy()
        return {"feat": feat, "conf": conf_u8}
    if method == "B2":
        feat = recon.detach().to(torch.float16).cpu().numpy()
        return {"feat": feat, "conf": conf_u8}
    codes_np = codes.detach().cpu().numpy()
    if codes_np.ndim == 1:
        code0 = codes_np.astype(np.uint8)
        code1 = None
    else:
        code0 = codes_np[:, 0].astype(np.uint8)
        code1 = codes_np[:, 1].astype(np.uint8) if codes_np.shape[1] > 1 else None
    if method == "B3":
        out = {"code": code0, "conf": conf_u8}
        if int(stage) > 1 and code1 is not None:
            out["code2"] = code1
        return out
    out = {
        "cls": classes.detach().cpu().numpy().astype(np.uint8),
        "code": code0,
        "conf": conf_u8,
        "evidence": np.full(code0.shape, 2, dtype=np.uint8),
    }
    if int(stage) > 1 and code1 is not None:
        out["code2"] = code1
    return out


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
    key = str(classes.device)
    lut = _SCORE_TARGET_CACHE.get(key)
    if lut is None:
        lut = _SCORE_TARGET_CPU.to(classes.device)
        _SCORE_TARGET_CACHE[key] = lut
    target = lut[classes.clamp(0, N_CLASS - 1)].to(dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, target)
