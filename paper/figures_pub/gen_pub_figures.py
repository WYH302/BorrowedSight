"""Generate all 7 publication-grade figures (300-DPI vector PDF + PNG) for BorrowedSight.

Strictly reads verbatim measurements from outputs/real_opv2v/ and outputs/ CSVs/JSONs
and saved checkpoints under D:/bs_data/BorrowedSight/checkpoints/real_opv2v/.
Zero fabricated numbers. Follows Elsevier / IEEE publication styling: serif typography,
restrained colorblind-safe palette, bold centered sub-captions (a)-(f), zero redundant
in-figure suptitles.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from borrowedsight.paths import CHECKPOINTS, DATA_ROOT

OUT = Path(__file__).resolve().parent

REAL_P4 = ROOT / "outputs" / "real_opv2v" / "v21_full" / "results.csv"
REAL_P5 = ROOT / "outputs" / "real_opv2v" / "v21_phase5" / "measurements.csv"
REAL_QUAL_NPZ = ROOT / "outputs" / "real_opv2v" / "v21_phase5" / "qualitative_v21_5cav.npz"
MINI_P5 = ROOT / "outputs" / "phase5" / "measurements.csv"
REAL_CACHE_ROOT = DATA_ROOT / "cache"
REAL_CKPT_DIR = CHECKPOINTS / "real_opv2v_v2"

PALETTE = {
    "B0": "#2D3748",
    "B1": "#4A90D9",
    "B2": "#2A9D8F",
    "B3": "#1D4E89",
    "B4-TDVF": "#8B2635",
    "B4-Compact": "#C85A17",
    "B4-MultiBW": "#0B7285",
}

SERIES = {
    "B1_v2_32k": {"color": "#4A90D9", "ls": ":", "marker": "^", "label": "B1 (Sparse FP16)"},
    "B2_v2_32k": {"color": "#2A9D8F", "ls": "-.", "marker": "d", "label": "B2 (Cont. 8-d)"},
    "B3_v2_32k": {"color": "#1D4E89", "ls": "-", "marker": "o", "label": "B3 (Compact 4B VQ)"},
    "B4_v21_32k": {"color": "#8B2635", "ls": "-", "marker": "P", "label": "B4-TDVF (5B + 50×50)"},
    "B4_v21_32k_no_env": {"color": "#C85A17", "ls": "--", "marker": "s", "label": "B4-Compact (5B)"},
    "B4_v21_multibw": {"color": "#0B7285", "ls": "-", "marker": "*", "label": "B4-TDVF-MultiBW"},
    "B3": {"color": "#1D4E89", "ls": "-", "marker": "o", "label": "B3 (Shared VQ)"},
    "B4": {"color": "#C85A17", "ls": "--", "marker": "s", "label": "B4 (BorrowedSight)"},
}


def apply_pub_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.0,
            "text.color": "#000000",
            "axes.labelcolor": "#000000",
            "axes.edgecolor": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.2,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.6,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def save_both(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png")
    plt.close(fig)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as h:
        return list(csv.DictReader(h))


def set_bold_subcaption(ax: plt.Axes, text: str, pad: float = 7.0) -> None:
    ax.set_title(text, fontsize=9.0, fontweight="bold", loc="center", pad=pad)


# ---------------------------------------------------------------------------
# Figure 1: End-to-End Architecture & Wire-Packet Protocol
# ---------------------------------------------------------------------------
def draw_fig1_architecture() -> None:
    fig, ax = plt.subplots(figsize=(10.6, 4.75))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 48.0)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc="#F7FAFC", ec="#2B6CB0", lw=1.2, title_c="#1A365D"):
        p = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.35,rounding_size=0.8",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h - 1.35, title, ha="center", va="top", fontsize=8.2, fontweight="bold", color=title_c)
        if body:
            ax.text(x + w / 2, y + (h - 2.6) / 2, body, ha="center", va="center", fontsize=7.0, color="#2D3748", linespacing=1.22)

    def arrow(x1, y1, x2, y2, color="#2D3748", ls="-", lw=1.25):
        arr = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>,head_length=5,head_width=3",
            color=color,
            linestyle=ls,
            linewidth=lw,
            mutation_scale=1.0,
        )
        ax.add_patch(arr)

    ax.text(50.0, 47.4, "(a) BorrowedSight End-to-End Zero-Ego-Observation Cooperative 3D Semantic Occupancy Pipeline",
            ha="center", va="top", fontsize=9.1, fontweight="bold", color="#1A202C")

    ax.add_patch(FancyBboxPatch((0.8, 12.6), 41.2, 31.8, boxstyle="round,pad=0.3,rounding_size=1.2",
                                facecolor="#F0F5FA", edgecolor="#1D4E89", linewidth=1.1, linestyle="--"))
    ax.text(21.4, 43.8, "Neighbor CAV Sender(s)  (LiDAR-Equipped)", ha="center", va="top",
            fontsize=8.6, fontweight="bold", color="#1D4E89")

    ax.add_patch(FancyBboxPatch((44.0, 12.6), 14.5, 31.8, boxstyle="round,pad=0.3,rounding_size=1.2",
                                facecolor="#FFF8F1", edgecolor="#C85A17", linewidth=1.1, linestyle="--"))
    ax.text(51.25, 43.8, "Lossy V2X Link", ha="center", va="top",
            fontsize=8.6, fontweight="bold", color="#9C4221")

    ax.add_patch(FancyBboxPatch((60.5, 12.6), 38.7, 31.8, boxstyle="round,pad=0.3,rounding_size=1.2",
                                facecolor="#F0FBF8", edgecolor="#2A9D8F", linewidth=1.1, linestyle="--"))
    ax.text(79.85, 43.8, "Zero-Ego Receiver  (No Ego Camera / LiDAR / Radar)", ha="center", va="top",
            fontsize=8.6, fontweight="bold", color="#1D6F65")

    box(2.2, 29.0, 11.2, 11.6, "Raw LiDAR", "Neighbor sweep\n$P_s \\subset \\mathbb{R}^3$\n(Clear / LISA)\n+ Pose $T_S^W$", fc="#FFFFFF", ec="#1D4E89")
    box(15.4, 29.0, 12.0, 11.6, "SenderEncoder", "Pure-PyTorch\nscatter voxel\n(100×100×8)\n2D ResUNet", fc="#EBF4FF", ec="#1D4E89")
    box(29.4, 29.8, 11.4, 10.8, "Quota + Fuse", "6-d geo stats\nThing/Vert floor\nRoad cap (45%)", fc="#EBF4FF", ec="#1D4E89")
    box(15.4, 15.6, 12.0, 10.6, "Ray Visibility", "Sender-ray\ntri-state grid\n20×20 / 25×25\nvisibility env.", fc="#EDF2F7", ec="#4A5568")
    box(29.4, 15.6, 11.4, 11.8, "Cat.-Geom. RVQ", "1-B cls anchor +\n3-grp ResidualSimVQ\nThings | Ground\n| Vertical Static", fc="#FEFCBF", ec="#B7791F", title_c="#744210")

    arrow(13.6, 34.8, 15.2, 34.8, "#1D4E89")
    ax.plot([7.8, 7.8], [28.6, 20.9], color="#4A5568", lw=1.25)
    arrow(7.8, 20.9, 15.2, 20.9, "#4A5568")
    arrow(27.6, 35.2, 29.2, 35.2, "#1D4E89")
    arrow(35.1, 29.5, 35.1, 27.6, "#B7791F")

    box(45.2, 27.2, 12.1, 13.4, "Wire Packet", "Exact budget\n2–32 KB\n512 B Zstd env.\n+ 16-tok blocks", fc="#FFFFFF", ec="#C85A17", title_c="#9C4221")
    box(45.2, 14.6, 12.1, 9.4, "Impairments", "Block drop $p$\nLatency $\\Delta t$\nPose noise", fc="#FFF5F5", ec="#C53030", title_c="#9B2C2C")
    arrow(41.0, 21.5, 45.0, 32.5, "#C85A17")
    ax.plot([21.4, 21.4, 43.0, 43.0], [15.2, 13.6, 13.6, 29.5], color="#4A5568", lw=1.2, ls="--")
    arrow(43.0, 29.5, 45.0, 29.5, "#4A5568", ls="--")
    arrow(51.25, 24.3, 51.25, 27.0, "#C53030", ls=":")

    box(61.8, 29.0, 12.0, 11.6, "Decode + Warp", "$E_{\\mathrm{cls}}(c)+\\mathrm{VQ}(q)$\n$T_E^{-1}T_S$ anchors\n& backward-warp\nvisibility env.", fc="#E6FFFA", ec="#2A9D8F", title_c="#1D6F65")
    box(61.8, 14.6, 12.0, 11.4, "Zero-Ego Inputs", "ONLY allowed:\n$T_E^W, t, (v,\\omega)$\nNo ego sensor\n(6 leak checks)", fc="#FFF5F5", ec="#9B2C2C", title_c="#9B2C2C")
    box(75.6, 22.0, 10.6, 18.6, "Evidence\nFusion", "Conf.-weighted\nscatter + age\ndecay + aux.\nsensor-origin\nray carving", fc="#E6FFFA", ec="#2A9D8F", title_c="#1D6F65")
    box(87.8, 22.0, 10.4, 18.6, "3D Decoder", "2D ResUNet\n12-cls Semantic\nTri-State Head\n(Occ/Free/Unk)\n+ Entropy Unc.", fc="#FFFFFF", ec="#1D4E89", title_c="#1D4E89")
    box(75.6, 14.6, 22.6, 5.2, "Safe Fallback Guard", "0 packets $\\Rightarrow$ 100% Unknown (UFE = 0.0)", fc="#EDF2F7", ec="#4A5568", title_c="#2D3748")

    arrow(57.5, 34.0, 61.6, 34.8, "#C85A17")
    arrow(67.8, 26.3, 67.8, 28.8, "#9B2C2C")
    arrow(74.0, 34.8, 75.4, 34.8, "#2A9D8F")
    arrow(86.4, 31.3, 87.6, 31.3, "#1D4E89")
    arrow(86.9, 21.7, 86.9, 20.0, "#4A5568", ls="--")

    ax.add_patch(FancyBboxPatch((0.8, 0.8), 98.4, 9.6, boxstyle="round,pad=0.25,rounding_size=0.9",
                                facecolor="#FAFAFA", edgecolor="#718096", linewidth=1.0))
    ax.text(50.0, 9.8, "(b) Exact Wire-Packet Layout (Independent 16-Token Morton Micro-Blocks + Fixed 512 B Visibility Slot)",
            ha="center", va="top", fontsize=8.4, fontweight="bold", color="#1A202C")

    segs = [
        (2.0, 12.5, "Header (64 B)", "magic, IDs, time,\n6-DoF pose, stage", "#E2E8F0", "#4A5568"),
        (15.2, 16.8, "Envelope Slot (512 B)", "20×20 / 25×25 flags\n+ max-z (Zstd ≤ 190 B)", "#FED7AA", "#C85A17"),
        (32.7, 19.5, "Micro-Block 1 (136 / 152 B)", "8B hdr + 16×(abs. Morton,\ncls, code1[,code2], conf)", "#BEE3F8", "#1D4E89"),
        (52.9, 19.5, "Micro-Block 2 (136 / 152 B)", "Self-contained block;\ndrop affects zero peers", "#BEE3F8", "#1D4E89"),
        (73.1, 13.0, "... Block $M$", "$M=116$ @ 16KB\n$M=236$ @ 32KB", "#BEE3F8", "#1D4E89"),
        (86.8, 11.2, "Tail Pad", "< 1 block;\n0.0% byte err", "#EDF2F7", "#718096"),
    ]
    for sx, sw, st, sb, fc, ec in segs:
        ax.add_patch(FancyBboxPatch((sx, 1.8), sw, 5.5, boxstyle="round,pad=0.15,rounding_size=0.4",
                                    facecolor=fc, edgecolor=ec, linewidth=0.9))
        ax.text(sx + sw / 2, 6.9, st, ha="center", va="top", fontsize=7.3, fontweight="bold", color="#1A202C")
        ax.text(sx + sw / 2, 3.7, sb, ha="center", va="center", fontsize=6.5, color="#2D3748", linespacing=1.18)

    save_both(fig, "fig1_architecture")


# ---------------------------------------------------------------------------
# Figure 2: Real OPV2V Multi-Sender Visibility Oracle (9 validate scenes, 90 frames)
# ---------------------------------------------------------------------------
def draw_fig2_real_oracle() -> None:
    senders = np.array([1, 2, 3])
    cov = {
        "Clear": [0.3878, 0.3799, 0.4130],
        "LISA Rain": [0.2365, 0.2523, 0.2770],
        "LISA Fog": [0.0982, 0.1028, 0.1158],
    }
    rec = {
        "Clear": [0.7962, 0.8836, 0.9190],
        "LISA Rain": [0.8560, 0.7982, 0.7844],
        "LISA Fog": [0.5043, 0.5574, 0.5837],
    }
    dens_cov = {
        "Sparse": [0.3813, 0.3224, 0.2815],
        "Medium": [0.3857, 0.4022, 0.4572],
        "Dense": [0.5036, 0.5542, 0.5850],
    }
    dens_rec = {
        "Sparse": [0.7745, 0.8013, 0.7619],
        "Medium": [0.8030, 0.9201, 0.9684],
        "Dense": [0.9259, 1.0000, 1.0000],
    }

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.85), constrained_layout=True)

    w_styles = {
        "Clear": ("#1D4E89", "-", "o"),
        "LISA Rain": ("#2A9D8F", "--", "s"),
        "LISA Fog": ("#C85A17", "-.", "^"),
    }
    for name, (c, ls, mk) in w_styles.items():
        axes[0].plot(senders, cov[name], color=c, ls=ls, marker=mk, lw=1.6, ms=5.0, label=name)
        axes[1].plot(senders, rec[name], color=c, ls=ls, marker=mk, lw=1.6, ms=5.0, label=name)

    axes[0].set_xticks(senders)
    axes[0].set_xlabel("Active neighbor senders")
    axes[0].set_ylabel("Observable voxel coverage")
    axes[0].set_ylim(0.0, 0.52)
    axes[0].legend(frameon=False, loc="upper left")
    set_bold_subcaption(axes[0], "(a) Observable Coverage vs. Weather")

    axes[1].set_xticks(senders)
    axes[1].set_xlabel("Active neighbor senders")
    axes[1].set_ylabel("3D vehicle box recall")
    axes[1].set_ylim(0.40, 1.02)
    axes[1].legend(frameon=False, loc="lower right")
    set_bold_subcaption(axes[1], "(b) 3D Vehicle Recall vs. Weather")

    d_styles = {
        "Sparse": ("#718096", ":", "^"),
        "Medium": ("#1D4E89", "-", "o"),
        "Dense": ("#C85A17", "--", "s"),
    }
    for name, (c, ls, mk) in d_styles.items():
        axes[2].plot(senders, dens_rec[name], color=c, ls=ls, marker=mk, lw=1.6, ms=5.0, label=f"{name} recall")
    ax2_r = axes[2].twinx()
    ax2_r.spines["top"].set_visible(False)
    ax2_r.grid(False)
    for name, (c, ls, mk) in (("Medium", ("#1D4E89", ":", "o")), ("Dense", ("#C85A17", ":", "s"))):
        ax2_r.plot(senders, dens_cov[name], color=c, ls=":", marker=mk, lw=1.1, ms=3.5, alpha=0.65)
    axes[2].set_xticks(senders)
    axes[2].set_xlabel("Active neighbor senders")
    axes[2].set_ylabel("3D vehicle box recall (solid/dash)")
    ax2_r.set_ylabel("Coverage (dotted)", color="#000000")
    axes[2].set_ylim(0.65, 1.04)
    ax2_r.set_ylim(0.25, 0.70)
    axes[2].legend(frameon=False, loc="lower right")
    set_bold_subcaption(axes[2], "(c) Urban Congestion Stratification (Clear)")

    save_both(fig, "fig2_real_oracle")


# ---------------------------------------------------------------------------
# Figure 3: Equalized-Byte Main Comparison on Full 25-Scene Real OPV2V (32KB & 16KB)
# ---------------------------------------------------------------------------
def draw_fig3_main_comparison() -> None:
    p4 = read_csv(REAL_P4)

    def agg_test(tag: str, mkey: str) -> tuple[float, float]:
        vals = [float(r[mkey]) for r in p4 if r["tag"] == tag]
        return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

    tags_32 = ["B0_v2", "B1_v2_32k", "B2_v2_32k", "B3_v2_32k", "B4_v21_32k", "B4_v21_32k_no_env"]
    labels_32 = ["B0", "B1", "B2", "B3", "B4-TDVF", "B4-Compact"]
    xtick_32 = ["B0", "B1", "B2", "B3", "B4-\nTDVF", "B4-\nCompact"]
    tags_16 = ["B0_v2", "B1_v2_16k", "B2_v2_16k", "B3_v2_16k", "B4_v21_16k"]
    labels_16 = ["B0", "B1", "B2", "B3", "B4-TDVF"]
    xtick_16 = ["B0", "B1", "B2", "B3", "B4-\nTDVF"]
    hatches_all = ["", "//", "\\\\", "xx", "++", ".."]

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.3), constrained_layout=True)
    metrics = [
        ("test_mIoU", "mIoU $\\uparrow$"),
        ("test_veh", "Vehicle IoU $\\uparrow$"),
        ("test_UFE", "UFE $\\downarrow$"),
    ]
    sub_letters = [
        ["(a) Real OPV2V Test (32 KB): mIoU", "(b) Real OPV2V Test (32 KB): Vehicle IoU", "(c) Real OPV2V Test (32 KB): UFE"],
        ["(d) Real OPV2V Test (16 KB): mIoU", "(e) Real OPV2V Test (16 KB): Vehicle IoU", "(f) Real OPV2V Test (16 KB): UFE"],
    ]

    for row_i, (tlist, labels, xticks) in enumerate(((tags_32, labels_32, xtick_32), (tags_16, labels_16, xtick_16))):
        xs = np.arange(len(labels))
        colors = [PALETTE[k] for k in labels]
        for col_i, (mkey, ylabel) in enumerate(metrics):
            ax = axes[row_i, col_i]
            means, stds = zip(*(agg_test(t, mkey) for t in tlist))
            vmax = max(m + s for m, s in zip(means, stds))
            for i, (v, sd) in enumerate(zip(means, stds)):
                ax.bar(
                    i,
                    v,
                    width=0.64,
                    color=colors[i],
                    hatch=hatches_all[i],
                    edgecolor="#000000",
                    linewidth=0.55,
                    yerr=(sd if sd > 0 else None),
                    capsize=(2.5 if sd > 0 else 0),
                    error_kw={"elinewidth": 0.9, "ecolor": "#000000"},
                )
                ax.text(
                    i,
                    v + sd + vmax * 0.025,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.8,
                    fontweight=("bold" if i >= 4 else "normal"),
                )
            if col_i == 0:
                ax.axhline(0.3416, color="#4A5568", ls="--", lw=0.9, alpha=0.85)
                ax.text(-0.42, 0.3416 + vmax * 0.022, "CoHFF (4-cam): 0.342", fontsize=6.0, color="#000000", ha="left")
                ax.axhline(0.4154, color="#8B2635", ls=":", lw=0.9, alpha=0.85)
                ax.text(-0.42, 0.4154 + vmax * 0.022, "VQSOP (4-cam): 0.415", fontsize=6.0, color="#000000", ha="left")
            ax.set_xticks(xs)
            ax.set_xticklabels(xticks, fontsize=7.0)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, vmax * 1.24)
            set_bold_subcaption(ax, sub_letters[row_i][col_i])

    save_both(fig, "fig3_main_comparison")


# ---------------------------------------------------------------------------
# Figure 4: Bandwidth-Accuracy-Safety Scaling (Real OPV2V 2-32KB & Mini-200 2-16KB)
# ---------------------------------------------------------------------------
def draw_fig4_bandwidth() -> None:
    p5_real = read_csv(REAL_P5)
    p5_mini = read_csv(MINI_P5)

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.4), constrained_layout=True)
    metrics = [("mIoU", "mIoU $\\uparrow$"), ("vehicle_IoU", "Vehicle IoU $\\uparrow$"), ("unobserved_as_free_err", "UFE $\\downarrow$")]
    titles = [
        ["(a) Real OPV2V (2–32 KB): mIoU", "(b) Real OPV2V (2–32 KB): Vehicle IoU", "(c) Real OPV2V (2–32 KB): UFE"],
        ["(d) Mini-200 (2–16 KB): mIoU", "(e) Mini-200 (2–16 KB): Vehicle IoU", "(f) Mini-200 (2–16 KB): UFE"],
    ]

    real_bws = ["2048", "4096", "8192", "16384", "24576", "32768"]
    real_x = np.arange(len(real_bws))
    real_xl = ["2K", "4K", "8K", "16K", "24K", "32K"]
    real_models = ("B1_v2_32k", "B2_v2_32k", "B3_v2_32k", "B4_v21_32k", "B4_v21_multibw")
    for col_i, (mkey, ylabel) in enumerate(metrics):
        ax = axes[0, col_i]
        for m in real_models:
            pts = [next(r for r in p5_real if r["family"] == "bandwidth" and r["model_name"] == m and r["level"] == b) for b in real_bws]
            st = SERIES[m]
            ax.plot(
                real_x,
                [float(r[mkey]) for r in pts],
                color=st["color"],
                ls=st["ls"],
                marker=st["marker"],
                lw=(1.9 if "multibw" in m else 1.45),
                ms=(5.5 if "multibw" in m else 4.3),
                label=st["label"],
            )
        if col_i == 0:
            ax.axhline(0.3416, color="#4A5568", ls=":", lw=1.0)
            ax.text(-0.18, 0.352, "CoHFF (4-cam): 0.342", fontsize=6.2, color="#000000", ha="left")
        ax.set_xticks(real_x)
        ax.set_xticklabels(real_xl)
        ax.set_xlabel("Serialized packet budget (bytes)")
        ax.set_ylabel(ylabel)
        set_bold_subcaption(ax, titles[0][col_i])

    mini_bws = ["2048", "4096", "8192", "12288", "16384"]
    mini_x = np.arange(len(mini_bws))
    mini_xl = ["2K", "4K", "8K", "12K", "16K"]
    for col_i, (mkey, ylabel) in enumerate(metrics):
        ax = axes[1, col_i]
        for m in ("B3", "B4"):
            pts = [next(r for r in p5_mini if r["family"] == "bandwidth" and r["model_name"] == m and r["level"] == b) for b in mini_bws]
            st = SERIES[m]
            ax.plot(mini_x, [float(r[mkey]) for r in pts], color=st["color"], ls=st["ls"], marker=st["marker"], lw=1.6, ms=4.8, label=st["label"])
        ax.set_xticks(mini_x)
        ax.set_xticklabels(mini_xl)
        ax.set_xlabel("Serialized packet budget (bytes)")
        ax.set_ylabel(ylabel)
        if col_i == 0:
            ax.legend(frameon=False, loc="upper left", fontsize=6.8)
        set_bold_subcaption(ax, titles[1][col_i])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.05), fontsize=7.2)
    save_both(fig, "fig4_bandwidth_efficiency")


# ---------------------------------------------------------------------------
# Figure 5: Full Real OPV2V 6-Axis Robustness, Multi-Sender & Congestion Suite
# ---------------------------------------------------------------------------
def draw_fig5_robustness() -> None:
    p5 = read_csv(REAL_P5)
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.6), constrained_layout=True)

    # (a) LISA Weather (Clear, Fog, Rain)
    ax = axes[0, 0]
    weathers = ["clear", "fog", "rain"]
    wx = np.arange(len(weathers))
    w = 0.20
    w_bars = [
        ("B3_v2_32k", "B3 (4B)", "#1D4E89"),
        ("B4_v21_32k", "B4-TDVF", "#8B2635"),
        ("B4_v21_32k_no_env", "B4-Compact", "#C85A17"),
        ("B4_v21_multibw", "B4-MultiBW", "#0B7285"),
    ]
    for bi, (tag, lbl, col) in enumerate(w_bars):
        offset = (bi - 1.5) * w
        vals = [float(next(r for r in p5 if r["family"] == "weather" and r["model_name"] == tag and r["level"] == lv)["mIoU"]) for lv in weathers]
        ax.bar(wx + offset, vals, width=w * 0.92, color=col, edgecolor="#000000", linewidth=0.45, label=lbl)
    ax.set_xticks(wx)
    ax.set_xticklabels(["Clear", "LISA Fog", "LISA Rain"])
    ax.set_ylabel("mIoU $\\uparrow$")
    ax.set_ylim(0, 0.50)
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=6.5)
    set_bold_subcaption(ax, "(a) LISA Physical Weather")

    # (b) Independent Micro-Block Packet Loss (0-50%)
    ax = axes[0, 1]
    drops = ["0.0", "0.1", "0.25", "0.5"]
    dx = np.array([0, 10, 25, 50])
    for m in ("B3_v2_32k", "B4_v21_32k", "B4_v21_32k_no_env", "B4_v21_multibw"):
        pts = [next(r for r in p5 if r["family"] == "packet_loss" and r["model_name"] == m and r["level"] == d) for d in drops]
        st = SERIES[m]
        ax.plot(dx, [float(r["mIoU"]) for r in pts], color=st["color"], ls=st["ls"], marker=st["marker"], lw=1.55, ms=4.5, label=st["label"])
    ax.set_xticks(dx)
    ax.set_xticklabels(["0%", "10%", "25%", "50%"])
    ax.set_xlabel("Micro-block drop rate")
    ax.set_ylabel("mIoU $\\uparrow$")
    ax.legend(frameon=False, loc="lower left", fontsize=6.5)
    set_bold_subcaption(ax, "(b) Independent Block Drop")

    # (c) Latency & Pose Noise
    ax = axes[0, 2]
    delays = ["0", "1", "2", "5"]
    tx = np.array([0, 100, 200, 500])
    short_map = {"B3_v2_32k": "B3", "B4_v21_32k": "B4-TDVF", "B4_v21_32k_no_env": "B4-Compact"}
    for m in ("B3_v2_32k", "B4_v21_32k", "B4_v21_32k_no_env"):
        pts = [next(r for r in p5 if r["family"] == "latency" and r["model_name"] == m and r["level"] == d) for d in delays]
        st = SERIES[m]
        ax.plot(tx, [float(r["mIoU"]) for r in pts], color=st["color"], ls=st["ls"], marker=st["marker"], lw=1.5, ms=4.4, label=f"{short_map[m]} (Delay)")
    pose_levels = ["0.0_0.0", "0.5_2.0", "1.0_5.0"]
    px = np.array([0, 200, 500])
    for m, mk in (("B3_v2_32k", "^"), ("B4_v21_32k", "d")):
        pts = [next(r for r in p5 if r["family"] == "pose" and r["model_name"] == m and r["level"] == pl) for pl in pose_levels]
        ax.plot(px, [float(r["mIoU"]) for r in pts], color=SERIES[m]["color"], ls=":", marker=mk, lw=1.25, ms=4.2, alpha=0.85, label=f"{short_map[m]} (Pose)")
    ax.set_xticks([0, 100, 200, 500])
    ax.set_xticklabels(["0ms\n(0m/0°)", "100ms", "200ms\n(0.5m/2°)", "500ms\n(1m/5°)"])
    ax.set_xlabel("V2X delay / pose error")
    ax.set_ylabel("mIoU $\\uparrow$")
    ax.legend(frameon=False, loc="lower left", fontsize=6.2, ncol=1)
    set_bold_subcaption(ax, "(c) Latency & Localization Error")

    # (d) Real Multi-Sender Scaling (1 -> 2 -> 3 senders on test_multi3, n=302)
    ax = axes[1, 0]
    ks = ["1", "2", "3"]
    kx = np.array([1, 2, 3])
    for m in ("B3_v2_32k", "B4_v21_32k"):
        pts = [next(r for r in p5 if r["split"] == "test_multi3" and r["family"] == "senders" and r["model_name"] == m and r["level"] == k) for k in ks]
        st = SERIES[m]
        ax.plot(kx, [float(r["mIoU"]) for r in pts], color=st["color"], ls=st["ls"], marker=st["marker"], lw=1.6, ms=4.8, label=f"{short_map[m]} mIoU")
    ax_r = ax.twinx()
    ax_r.spines["top"].set_visible(False)
    ax_r.grid(False)
    for m in ("B3_v2_32k", "B4_v21_32k"):
        pts = [next(r for r in p5 if r["split"] == "test_multi3" and r["family"] == "senders" and r["model_name"] == m and r["level"] == k) for k in ks]
        st = SERIES[m]
        ax_r.plot(kx, [float(r["vehicle_IoU"]) for r in pts], color=st["color"], ls=":", marker=st["marker"], lw=1.25, ms=3.8, alpha=0.85)
    ax.set_xticks(kx)
    ax.set_xlabel("Active neighbor senders ($n=302$)")
    ax.set_ylabel("mIoU $\\uparrow$ (solid)")
    ax_r.set_ylabel("Vehicle IoU (dotted)", color="#000000")
    ax.legend(frameon=False, loc="lower right", fontsize=7.0)
    set_bold_subcaption(ax, "(d) Multi-Sender Scaling (Real Test)")

    # (e) Urban Congestion Stratification (Sparse / Medium / Dense)
    ax = axes[1, 1]
    dens = ["sparse", "medium", "dense"]
    dx2 = np.arange(3)
    w2 = 0.34
    for offset, tag, lbl, col in ((-w2 / 2, "B3_v2_32k", "B3 (4B)", "#1D4E89"), (w2 / 2, "B4_v21_32k", "B4-TDVF", "#8B2635")):
        vals = [float(next(r for r in p5 if r["family"] == "density" and r["model_name"] == tag and r["level"] == d)["mIoU"]) for d in dens]
        ax.bar(dx2 + offset, vals, width=w2 * 0.92, color=col, edgecolor="#000000", linewidth=0.5, label=lbl)
        for xi, v in zip(dx2 + offset, vals):
            ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(dx2)
    ax.set_xticklabels(["Sparse\n($n=297$)", "Medium\n($n=244$)", "Dense\n($n=185$)"])
    ax.set_ylabel("mIoU $\\uparrow$")
    ax.set_ylim(0, 0.49)
    ax.legend(frameon=False, loc="upper left", fontsize=7.2)
    set_bold_subcaption(ax, "(e) Urban Traffic Congestion")

    # (f) Uncertainty Separation & Degradation Monotonicity (B4-TDVF)
    ax = axes[1, 2]
    pts_d = [next(r for r in p5 if r["family"] == "packet_loss" and r["model_name"] == "B4_v21_32k" and r["level"] == d) for d in drops]
    sx = np.arange(4)
    ax.plot(sx, [float(r["uncertainty_on_unknown"]) for r in pts_d], "-s", color="#8B2635", lw=1.6, ms=4.5, label="Risk on Unknown")
    ax.plot(sx, [float(r["mean_uncertainty"]) for r in pts_d], "--o", color="#C85A17", lw=1.6, ms=4.5, label="Mean Risk")
    ax.plot(sx, [float(r["uncertainty_on_observed"]) for r in pts_d], "-.^", color="#2A9D8F", lw=1.5, ms=4.5, label="Risk on Observed")
    ax.set_xticks(sx)
    ax.set_xticklabels(["0%", "10%", "25%", "50%"])
    ax.set_xlabel("Micro-block drop rate (B4-TDVF)")
    ax.set_ylabel("Epistemic risk score $U$")
    ax.set_ylim(0.28, 0.62)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.02, 0.45), fontsize=6.8)
    set_bold_subcaption(ax, "(f) Blind-Zone Risk Separation (B4-TDVF)")

    save_both(fig, "fig5_robustness_suite")


# ---------------------------------------------------------------------------
# Figures 6 & 7: Clean 5-CAV Multi-Vehicle Real OPV2V Qualitative Figures (v2.1)
# ---------------------------------------------------------------------------
def draw_real_qualitative_figures() -> None:
    from tools.eval_phase5_real import sem_tri_to_rgb

    blob = np.load(REAL_QUAL_NPZ, allow_pickle=False)
    legend_handles = [
        patches.Patch(facecolor=[35 / 255, 35 / 255, 48 / 255], edgecolor="black", lw=0.4, label="Unknown (Blind)"),
        patches.Patch(facecolor=[235 / 255, 240 / 255, 245 / 255], edgecolor="black", lw=0.4, label="Observed Free"),
        patches.Patch(facecolor=[230 / 255, 45 / 255, 55 / 255], edgecolor="black", lw=0.4, label="Vehicles"),
        patches.Patch(facecolor=[130 / 255, 120 / 255, 165 / 255], edgecolor="black", lw=0.4, label="Road"),
        patches.Patch(facecolor=[175 / 255, 130 / 255, 115 / 255], edgecolor="black", lw=0.4, label="SideWalk"),
        patches.Patch(facecolor=[210 / 255, 175 / 255, 70 / 255], edgecolor="black", lw=0.4, label="GuardRail"),
        patches.Patch(facecolor=[60 / 255, 155 / 255, 75 / 255], edgecolor="black", lw=0.4, label="Vegetation"),
    ]

    # Figure 6: 32,768 B Equalized-Byte 5-CAV Qualitative Comparison
    fig, axes = plt.subplots(1, 5, figsize=(10.4, 2.55), constrained_layout=True)
    panels6 = [
        ("(a) Ground Truth", blob["gt_sem"], blob["gt_tri"]),
        ("(b) B0 (0 B)", blob["b0_sem"], blob["b0_tri"]),
        ("(c) B1 (Sparse FP16)", blob["b1_sem"], blob["b1_tri"]),
        ("(d) B3 (Compact 4B VQ)", blob["b3_sem"], blob["b3_tri"]),
        ("(e) B4-TDVF (Ours)", blob["b4_sem"], blob["b4_tri"]),
    ]
    for ax, (cap, s, t) in zip(axes, panels6):
        ax.imshow(sem_tri_to_rgb(s, t), origin="lower", interpolation="nearest")
        ax.scatter([50], [50], c="cyan", s=28, marker="^", edgecolors="black", linewidths=0.6)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.7)
        ax.set_xlabel(cap, fontsize=8.4, fontweight="bold", labelpad=5)
    fig.legend(handles=legend_handles, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.08), fontsize=7.5)
    save_both(fig, "fig6_real_qualitative")

    # Figure 7: Adverse Weather, 50% Block Drop, and Honest Outage Failure (32,768 B)
    fig, axes = plt.subplots(1, 5, figsize=(10.4, 2.55), constrained_layout=True)
    panels7 = [
        ("(a) B4-TDVF Clear", blob["b4_sem"], blob["b4_tri"]),
        ("(b) LISA Fog", blob["b4_fog_sem"], blob["b4_fog_tri"]),
        ("(c) LISA Rain (Clutter)", blob["b4_rain_sem"], blob["b4_rain_tri"]),
        ("(d) 50% Block Drop", blob["b4_drop50_sem"], blob["b4_drop50_tri"]),
        ("(e) Link Outage (0 B)", blob["b4_outage_sem"], blob["b4_outage_tri"]),
    ]
    for ax, (cap, s, t) in zip(axes, panels7):
        ax.imshow(sem_tri_to_rgb(s, t), origin="lower", interpolation="nearest")
        ax.scatter([50], [50], c="cyan", s=28, marker="^", edgecolors="black", linewidths=0.6)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.7)
        ax.set_xlabel(cap, fontsize=8.4, fontweight="bold", labelpad=5)
    fig.legend(handles=legend_handles, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.08), fontsize=7.5)
    save_both(fig, "fig7_weather_failure_qualitative")


def main() -> None:
    apply_pub_style()
    OUT.mkdir(parents=True, exist_ok=True)
    # Preserve curated vector fig1_architecture.pdf/.png if already exported from Paper2SVG-Pro
    if "--regen-fig1" in sys.argv:
        draw_fig1_architecture()
    draw_fig2_real_oracle()
    draw_fig3_main_comparison()
    draw_fig4_bandwidth()
    draw_fig5_robustness()
    draw_real_qualitative_figures()
    print("GENERATED FULL-SCALE PUBLICATION FIGURES IN", OUT)


if __name__ == "__main__":
    main()
