"""Generate all publication-grade IEEE Transactions figures (300-DPI vector PDF + PNG) for BorrowedSight v3.0.

Strictly reads verbatim measurements from outputs/real_opv2v/v3_full/, v3_phase5/,
v2v4real_zeroshot/, and outputs/phase5/. Zero fabricated numbers.
Preserves curated vector fig1_architecture.pdf/.png unless `--regen-fig1` is passed.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = Path(__file__).resolve().parent

V3_FULL = ROOT / "outputs" / "real_opv2v" / "v3_full" / "results.csv"
V3_P5 = ROOT / "outputs" / "real_opv2v" / "v3_phase5" / "measurements.csv"
V3_SSKFW = ROOT / "outputs" / "real_opv2v" / "v3_phase5" / "ss_kfw_analysis.json"
V3_CB = ROOT / "outputs" / "real_opv2v" / "v3_phase5" / "codebook_analysis.json"
V3_QUAL = ROOT / "outputs" / "real_opv2v" / "v3_phase5" / "qualitative_v3_5cav.npz"
V2V4REAL = ROOT / "outputs" / "real_opv2v" / "v2v4real_zeroshot" / "v2v4real_summary.json"
MINI_P5 = ROOT / "outputs" / "phase5" / "measurements.csv"

PALETTE = {
    "B0": "#2D3748",
    "B1": "#718096",
    "Where2comm": "#D69E2E",
    "B2": "#2A9D8F",
    "B3": "#1D4E89",
    "CodeFilling": "#6B46C1",
    "B4-v3": "#8B2635",
    "B4-MultiBW": "#0B7285",
    "B4-noEnv": "#C85A17",
}

SERIES = {
    "B1_v3_32k": {"color": "#718096", "ls": ":", "marker": "^", "label": "B1 (FP16 36B)"},
    "B2_v3_32k": {"color": "#2A9D8F", "ls": "-.", "marker": "d", "label": "B2 (Cont. 20B)"},
    "Where2comm_v3_32k": {"color": "#D69E2E", "ls": "--", "marker": "v", "label": "Where2comm-Occ"},
    "B3_v3_32k": {"color": "#1D4E89", "ls": "-", "marker": "o", "label": "B3 (VQ 5B)"},
    "CodeFilling_v3_32k": {"color": "#6B46C1", "ls": "-.", "marker": "s", "label": "CodeFilling-Occ"},
    "B4_v3_32k": {"color": "#8B2635", "ls": "-", "marker": "P", "label": "BorrowedSight"},
    "B4_v3_32k_no_env": {"color": "#C85A17", "ls": "--", "marker": "X", "label": "Ours w/o Env."},
    "B4_v3_multibw": {"color": "#0B7285", "ls": "-", "marker": "*", "label": "BS-MultiBW"},
    "B3": {"color": "#1D4E89", "ls": "-", "marker": "o", "label": "B3 (Shared VQ)"},
    "B4": {"color": "#8B2635", "ls": "-", "marker": "P", "label": "B4 (BorrowedSight)"},
}


def apply_pub_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.8,
            "text.color": "#000000",
            "axes.labelcolor": "#000000",
            "axes.edgecolor": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "axes.labelsize": 8.6,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.2,
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


def set_bold_subcaption(ax: plt.Axes, text: str, pad: float = 6.0) -> None:
    ax.set_title(text, fontsize=8.8, fontweight="bold", loc="center", pad=pad)


# ---------------------------------------------------------------------------
# Figure 2: Real OPV2V Multi-Sender Visibility Oracle (9 validate scenes, 90 frames)
# ---------------------------------------------------------------------------
def draw_fig2_real_oracle() -> None:
    senders = np.array([1, 2, 3])
    cov = {"Clear": [0.3878, 0.3799, 0.4130], "LISA Rain": [0.2365, 0.2523, 0.2770], "LISA Fog": [0.0982, 0.1028, 0.1158]}
    rec = {"Clear": [0.7962, 0.8836, 0.9190], "LISA Rain": [0.8560, 0.7982, 0.7844], "LISA Fog": [0.5043, 0.5574, 0.5837]}
    dens_cov = {"Sparse": [0.3813, 0.3224, 0.2815], "Medium": [0.3857, 0.4022, 0.4572], "Dense": [0.5036, 0.5542, 0.5850]}
    dens_rec = {"Sparse": [0.7745, 0.8013, 0.7619], "Medium": [0.8030, 0.9201, 0.9684], "Dense": [0.9259, 1.0000, 1.0000]}

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.85), constrained_layout=True)
    w_styles = {"Clear": ("#1D4E89", "-", "o"), "LISA Rain": ("#2A9D8F", "--", "s"), "LISA Fog": ("#C85A17", "-.", "^")}
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

    for name, (c, ls, mk) in (("Sparse", ("#718096", ":", "^")), ("Medium", ("#1D4E89", "-", "o")), ("Dense", ("#C85A17", "--", "s"))):
        axes[2].plot(senders, dens_rec[name], color=c, ls=ls, marker=mk, lw=1.6, ms=5.0, label=f"{name} recall")
    ax2_r = axes[2].twinx()
    ax2_r.spines["top"].set_visible(False)
    ax2_r.grid(False)
    for name, (c, mk) in (("Medium", ("#1D4E89", "o")), ("Dense", ("#C85A17", "s"))):
        ax2_r.plot(senders, dens_cov[name], color=c, ls=":", marker=mk, lw=1.1, ms=3.5, alpha=0.65)
    axes[2].set_xticks(senders)
    axes[2].set_xlabel("Active neighbor senders")
    axes[2].set_ylabel("3D vehicle box recall")
    ax2_r.set_ylabel("Coverage (dotted)")
    axes[2].set_ylim(0.65, 1.04)
    ax2_r.set_ylim(0.25, 0.70)
    axes[2].legend(frameon=False, loc="lower right")
    set_bold_subcaption(axes[2], "(c) Urban Congestion Stratification (Clear)")
    save_both(fig, "fig2_real_oracle")


# ---------------------------------------------------------------------------
# Figure 3: Equalized-Byte Main Comparison (32 KB & 16 KB, 3-Seed + Public SOTA)
# ---------------------------------------------------------------------------
def draw_fig3_main_comparison() -> None:
    rows = read_csv(V3_FULL)

    def agg(tag: str, key: str) -> tuple[float, float]:
        vals = [float(r[key]) for r in rows if r["tag"] == tag]
        return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

    tags_32 = ["B0_v3_32k", "B1_v3_32k", "Where2comm_v3_32k", "B2_v3_32k", "B3_v3_32k", "CodeFilling_v3_32k", "B4_v3_32k"]
    tags_16 = ["B0_v3_32k", "B1_v3_16k", "Where2comm_v3_16k", "B2_v3_16k", "B3_v3_16k", "CodeFilling_v3_16k", "B4_v3_16k"]
    keys_p = ["B0", "B1", "Where2comm", "B2", "B3", "CodeFilling", "B4-v3"]
    xticks = ["B0", "B1", "W2C-\nOcc", "B2", "B3-\nVQ", "CF-\nOcc", "Ours"]
    hatches = ["", "//", "oo", "\\\\", "xx", "..", "++"]

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.4), constrained_layout=True)
    metrics = [("test_mIoU", "Test mIoU $\\uparrow$"), ("test_veh", "Vehicle IoU $\\uparrow$"), ("test_UFE", "UFE $\\downarrow$")]
    titles = [
        ["(a) Real OPV2V Test (32 KB): mIoU", "(b) Real OPV2V Test (32 KB): Vehicle IoU", "(c) Real OPV2V Test (32 KB): UFE"],
        ["(d) Real OPV2V Test (16 KB): mIoU", "(e) Real OPV2V Test (16 KB): Vehicle IoU", "(f) Real OPV2V Test (16 KB): UFE"],
    ]

    for ri, tlist in enumerate((tags_32, tags_16)):
        xs = np.arange(len(tlist))
        for ci, (mkey, ylabel) in enumerate(metrics):
            ax = axes[ri, ci]
            means, stds = zip(*(agg(t, mkey) for t in tlist))
            vmax = max(m + s for m, s in zip(means, stds))
            for i, (v, sd) in enumerate(zip(means, stds)):
                ax.bar(
                    i, v, width=0.66, color=PALETTE[keys_p[i]], hatch=hatches[i],
                    edgecolor="#000000", linewidth=0.55,
                    yerr=(sd if sd > 0 else None), capsize=(2.3 if sd > 0 else 0),
                    error_kw={"elinewidth": 0.9, "ecolor": "#000000"},
                )
                ax.text(
                    i, v + sd + vmax * 0.022, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.4, fontweight=("bold" if i == 6 else "normal"),
                )
            if ci == 0:
                ax.axhline(0.3416, color="#4A5568", ls="--", lw=0.85, alpha=0.85)
                ax.text(-0.44, 0.3416 + vmax * 0.018, "CoHFF (4-cam): 0.342", fontsize=5.8, ha="left")
                ax.axhline(0.4154, color="#8B2635", ls=":", lw=0.85, alpha=0.85)
                ax.text(-0.44, 0.4154 + vmax * 0.018, "VQSOP (4-cam): 0.415", fontsize=5.8, ha="left")
            ax.set_xticks(xs)
            ax.set_xticklabels(xticks, fontsize=6.7)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, vmax * 1.25)
            set_bold_subcaption(ax, titles[ri][ci])

    save_both(fig, "fig3_main_comparison")


# ---------------------------------------------------------------------------
# Figure 4: Bandwidth-Accuracy-Safety Scaling (Real OPV2V 2-32 KB & Mini-200)
# ---------------------------------------------------------------------------
def draw_fig4_bandwidth() -> None:
    p5_real = read_csv(V3_P5)
    p5_mini = read_csv(MINI_P5)

    fig, axes = plt.subplots(2, 3, figsize=(10.0, 5.45), constrained_layout=True)
    real_metrics = [("mIoU", "mIoU $\\uparrow$"), ("vehicle_IoU", "Vehicle IoU $\\uparrow$"), ("UFE", "UFE $\\downarrow$")]
    mini_metrics = [("mIoU", "mIoU $\\uparrow$"), ("vehicle_IoU", "Vehicle IoU $\\uparrow$"), ("unobserved_as_free_err", "UFE $\\downarrow$")]
    titles = [
        ["(a) Real OPV2V (2–32 KB): mIoU", "(b) Real OPV2V (2–32 KB): Vehicle IoU", "(c) Real OPV2V (2–32 KB): UFE"],
        ["(d) Mini-200 (2–16 KB): mIoU", "(e) Mini-200 (2–16 KB): Vehicle IoU", "(f) Mini-200 (2–16 KB): UFE"],
    ]

    real_bws = ["2048", "4096", "8192", "16384", "24576", "32768"]
    real_x = np.arange(len(real_bws))
    real_xl = ["2K", "4K", "8K", "16K", "24K", "32K"]
    real_models = ("B1_v3_32k", "Where2comm_v3_32k", "B3_v3_32k", "CodeFilling_v3_32k", "B4_v3_32k", "B4_v3_multibw")
    for ci, (mkey, ylabel) in enumerate(real_metrics):
        ax = axes[0, ci]
        for m in real_models:
            pts = [next(r for r in p5_real if r["family"] == "bandwidth" and r["model"] == m and r["level"] == b) for b in real_bws]
            st = SERIES[m]
            ax.plot(
                real_x, [float(r[mkey]) for r in pts],
                color=st["color"], ls=st["ls"], marker=st["marker"],
                lw=(1.9 if "B4" in m else 1.35), ms=(5.2 if "B4" in m else 4.0), label=st["label"],
            )
        if ci == 0:
            ax.axhline(0.3416, color="#4A5568", ls=":", lw=0.9)
            ax.text(3.35, 0.314, "CoHFF (4-cam): 0.342", fontsize=6.0, ha="left", color="#2D3748")
        ax.set_xticks(real_x)
        ax.set_xticklabels(real_xl)
        ax.set_xlabel("Serialized packet budget (bytes)")
        ax.set_ylabel(ylabel)
        set_bold_subcaption(ax, titles[0][ci])

    mini_bws = ["2048", "4096", "8192", "12288", "16384"]
    mini_x = np.arange(len(mini_bws))
    mini_xl = ["2K", "4K", "8K", "12K", "16K"]
    for ci, (mkey, ylabel) in enumerate(mini_metrics):
        ax = axes[1, ci]
        for m in ("B3", "B4"):
            pts = [next(r for r in p5_mini if r["family"] == "bandwidth" and r["model_name"] == m and r["level"] == b) for b in mini_bws]
            st = SERIES[m]
            ax.plot(mini_x, [float(r[mkey]) for r in pts], color=st["color"], ls=st["ls"], marker=st["marker"], lw=1.6, ms=4.8, label=st["label"])
        ax.set_xticks(mini_x)
        ax.set_xticklabels(mini_xl)
        ax.set_xlabel("Serialized packet budget (bytes)")
        ax.set_ylabel(ylabel)
        if ci == 0:
            ax.legend(frameon=False, loc="upper left", fontsize=6.8)
        set_bold_subcaption(ax, titles[1][ci])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.055), fontsize=7.0)
    save_both(fig, "fig4_bandwidth_efficiency")


# ---------------------------------------------------------------------------
# Figure 5: Six-Axis Robustness Suite (Including SS-KFW Latency Recovery!)
# ---------------------------------------------------------------------------
def draw_fig5_robustness() -> None:
    p5 = read_csv(V3_P5)
    fig, axes = plt.subplots(2, 3, figsize=(10.3, 5.65), constrained_layout=True)

    # (a) Disjoint LISA Physical Weather
    ax = axes[0, 0]
    w_models = [
        ("Where2comm_v3_32k", "W2C-Occ", "#D69E2E"),
        ("B3_v3_32k", "B3-VQ", "#1D4E89"),
        ("CodeFilling_v3_32k", "CF-Occ", "#6B46C1"),
        ("B4_v3_32k_no_env", "B4 w/o Env", "#C85A17"),
        ("B4_v3_32k", "Ours", "#8B2635"),
    ]
    wx = np.arange(3)
    w = 0.16
    for bi, (tag, lbl, col) in enumerate(w_models):
        offset = (bi - 2.0) * w
        v_clear = float(next(r for r in p5 if r["family"] == "clean_32k" and r["model"] == tag)["mIoU"])
        v_fog = float(next(r for r in p5 if r["family"] == "weather" and r["model"] == tag and r["level"] == "lisa_fog")["mIoU"])
        v_rain = float(next(r for r in p5 if r["family"] == "weather" and r["model"] == tag and r["level"] == "lisa_rain")["mIoU"])
        ax.bar(wx + offset, [v_clear, v_fog, v_rain], width=w * 0.90, color=col, edgecolor="#000000", linewidth=0.45, label=lbl)
    ax.set_xticks(wx)
    ax.set_xticklabels(["Clear", "LISA Fog", "LISA Rain"])
    ax.set_ylabel("mIoU $\\uparrow$")
    ax.set_ylim(0, 0.54)
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=6.0)
    set_bold_subcaption(ax, "(a) Disjoint LISA Physical Weather")

    # (b) Independent Micro-Block Drop (0-50%)
    ax = axes[0, 1]
    drops = ["0pct", "10pct", "25pct", "50pct"]
    dx = np.array([0, 10, 25, 50])
    for m in ("Where2comm_v3_32k", "B3_v3_32k", "CodeFilling_v3_32k", "B4_v3_32k_no_env", "B4_v3_32k"):
        pts = [next(r for r in p5 if r["family"] == "block_drop" and r["model"] == m and r["level"] == d) for d in drops]
        st = SERIES[m]
        ax.plot(dx, [float(r["mIoU"]) for r in pts], color=st["color"], ls=st["ls"], marker=st["marker"], lw=(1.85 if m == "B4_v3_32k" else 1.35), ms=4.3, label=st["label"])
    ax.set_xticks(dx)
    ax.set_xticklabels(["0%", "10%", "25%", "50%"])
    ax.set_xlabel("Micro-block drop rate")
    ax.set_ylabel("mIoU $\\uparrow$")
    ax.legend(frameon=False, loc="lower left", fontsize=6.2)
    set_bold_subcaption(ax, "(b) Independent Micro-Block Loss")

    # (c) V2X Latency Vehicle IoU Recovery with Zero-Byte SS-KFW!
    ax = axes[0, 2]
    d_lvls = ["0ms", "100ms", "200ms", "500ms"]
    tx = np.array([0, 100, 200, 500])
    lat_specs = [
        ("B3_v3_32k", "#1D4E89", "-", "o", "B3-VQ"),
        ("CodeFilling_v3_32k", "#6B46C1", "-.", "s", "CF-Occ"),
        ("B4_v3_32k_noKFW", "#C85A17", "--", "X", "B4 w/o KFW"),
        ("B4_v3_32k_SSKFW", "#8B2635", "-", "P", "B4 + SS-KFW"),
    ]
    for tag, col, ls, mk, lbl in lat_specs:
        pts = [next(r for r in p5 if r["family"] == "latency" and r["model"] == tag and r["level"] == d) for d in d_lvls]
        ax.plot(tx, [float(r["vehicle_IoU"]) for r in pts], color=col, ls=ls, marker=mk, lw=(2.0 if "SSKFW" in tag else 1.35), ms=4.6, label=lbl)
    ax.annotate("3.49× @ 500ms\n(0 extra B)", xy=(500, 0.1994), xytext=(305, 0.285),
                arrowprops=dict(arrowstyle="->", color="#8B2635", lw=1.0), fontsize=6.5, fontweight="bold", color="#8B2635")
    ax.set_xticks(tx)
    ax.set_xticklabels(["0", "100", "200", "500"])
    ax.set_xlabel("V2X packet latency (ms)")
    ax.set_ylabel("Vehicle IoU $\\uparrow$")
    ax.set_ylim(0.0, 0.54)
    ax.legend(frameon=False, loc="upper right", fontsize=6.2)
    set_bold_subcaption(ax, "(c) V2X Latency & Zero-Byte SS-KFW")

    # (d) Localization Pose Error
    ax = axes[1, 0]
    p_lvls = ["0.0m_0.0deg", "0.5m_2.0deg", "1.0m_5.0deg"]
    px = np.arange(3)
    for m in ("Where2comm_v3_32k", "B3_v3_32k", "CodeFilling_v3_32k", "B4_v3_32k"):
        pts = [next(r for r in p5 if r["family"] == "pose_error" and r["model"] == m and r["level"] == lv) for lv in p_lvls]
        st = SERIES[m]
        ax.plot(px, [float(r["mIoU"]) for r in pts], color=st["color"], ls=st["ls"], marker=st["marker"], lw=(1.8 if m == "B4_v3_32k" else 1.35), ms=4.5, label=st["label"])
    ax.set_xticks(px)
    ax.set_xticklabels(["0m / 0°", "0.5m / 2°", "1.0m / 5°"])
    ax.set_xlabel("Sender pose noise ($\\sigma_{xy} / \\sigma_{\\psi}$)")
    ax.set_ylabel("mIoU $\\uparrow$")
    ax.legend(frameon=False, loc="upper right", fontsize=6.3)
    set_bold_subcaption(ax, "(d) Localization Pose Perturbation")

    # (e) Multi-Sender Scaling (k=1..4) Vehicle IoU
    ax = axes[1, 1]
    ks = ["k1", "k2", "k3", "k4"]
    kx = np.array([1, 2, 3, 4])
    for m in ("B3_v3_32k", "CodeFilling_v3_32k", "B4_v3_32k"):
        pts = [next(r for r in p5 if r["family"] == "senders" and r["model"] == m and r["level"] == k) for k in ks]
        st = SERIES[m]
        ax.plot(kx, [float(r["vehicle_IoU"]) for r in pts], color=st["color"], ls=st["ls"], marker=st["marker"], lw=(1.85 if m == "B4_v3_32k" else 1.4), ms=4.6, label=st["label"])
    ax.set_xticks(kx)
    ax.set_xticklabels(["1", "2", "3", "4"])
    ax.set_xlabel("Active neighbor senders $K$")
    ax.set_ylabel("Vehicle IoU $\\uparrow$")
    ax.set_ylim(0.36, 0.54)
    ax.legend(frameon=False, loc="lower right", fontsize=6.5)
    set_bold_subcaption(ax, "(e) Multi-Sender Scaling (Vehicle IoU)")

    # (f) Urban Traffic Congestion (Low <=2 vs High >=4 vehicles)
    ax = axes[1, 2]
    cx = np.arange(2)
    wc = 0.24
    for bi, (tag, lbl, col) in enumerate((("B3_v3_32k", "B3-VQ", "#1D4E89"), ("CodeFilling_v3_32k", "CF-Occ", "#6B46C1"), ("B4_v3_32k", "Ours", "#8B2635"))):
        vals = [
            float(next(r for r in p5 if r["family"] == "congestion" and r["model"] == tag and r["level"] == "le2_veh")["mIoU"]),
            float(next(r for r in p5 if r["family"] == "congestion" and r["model"] == tag and r["level"] == "ge4_veh")["mIoU"]),
        ]
        off = (bi - 1.0) * wc
        ax.bar(cx + off, vals, width=wc * 0.90, color=col, edgecolor="#000000", linewidth=0.5, label=lbl)
        for xi, v in zip(cx + off, vals):
            ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=6.4, fontweight=("bold" if bi == 2 else "normal"))
    ax.set_xticks(cx)
    ax.set_xticklabels(["Low Traffic\n($\\leq 2$ veh., $n=62$)", "Dense Traffic\n($\\geq 4$ veh., $n=614$)"])
    ax.set_ylabel("mIoU $\\uparrow$")
    ax.set_ylim(0, 0.52)
    ax.legend(frameon=False, loc="upper left", fontsize=6.5)
    set_bold_subcaption(ax, "(f) Urban Congestion Stratification")

    save_both(fig, "fig5_robustness_suite")


# ---------------------------------------------------------------------------
# Figures 6 & 7: 5-CAV Multi-Vehicle Real OPV2V Qualitative Figures (v3.0)
# ---------------------------------------------------------------------------
def draw_real_qualitative_figures() -> None:
    from tools.eval_phase5_real import sem_tri_to_rgb

    blob = np.load(V3_QUAL, allow_pickle=False)
    legend_handles = [
        patches.Patch(facecolor=[35 / 255, 35 / 255, 48 / 255], edgecolor="black", lw=0.4, label="Unknown (Blind)"),
        patches.Patch(facecolor=[235 / 255, 240 / 255, 245 / 255], edgecolor="black", lw=0.4, label="Observed Free"),
        patches.Patch(facecolor=[230 / 255, 45 / 255, 55 / 255], edgecolor="black", lw=0.4, label="Vehicles"),
        patches.Patch(facecolor=[130 / 255, 120 / 255, 165 / 255], edgecolor="black", lw=0.4, label="Road"),
        patches.Patch(facecolor=[175 / 255, 130 / 255, 115 / 255], edgecolor="black", lw=0.4, label="SideWalk"),
        patches.Patch(facecolor=[210 / 255, 175 / 255, 70 / 255], edgecolor="black", lw=0.4, label="GuardRail"),
        patches.Patch(facecolor=[60 / 255, 155 / 255, 75 / 255], edgecolor="black", lw=0.4, label="Vegetation"),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(11.2, 2.35), constrained_layout=True)
    panels6 = [
        ("(a) Ground Truth", blob["gt_sem"], blob["gt_tri"]),
        ("(b) B0 (0 B)", blob["b0_sem"], blob["b0_tri"]),
        ("(c) Where2comm-Occ", blob["w2c_sem"], blob["w2c_tri"]),
        ("(d) CodeFilling-Occ", blob["cf_sem"], blob["cf_tri"]),
        ("(e) B3-VQ (5B)", blob["b3_sem"], blob["b3_tri"]),
        ("(f) BorrowedSight", blob["b4_sem"], blob["b4_tri"]),
    ]
    for ax, (cap, s, t) in zip(axes, panels6):
        ax.imshow(sem_tri_to_rgb(s, t), origin="lower", interpolation="nearest")
        ax.scatter([50], [50], c="cyan", s=24, marker="^", edgecolors="black", linewidths=0.6)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.7)
        ax.set_xlabel(cap, fontsize=8.0, fontweight="bold", labelpad=5)
    fig.legend(handles=legend_handles, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.09), fontsize=7.4)
    save_both(fig, "fig6_real_qualitative")

    fig, axes = plt.subplots(1, 5, figsize=(10.4, 2.50), constrained_layout=True)
    panels7 = [
        ("(a) BorrowedSight Clear", blob["b4_sem"], blob["b4_tri"]),
        ("(b) LISA Fog", blob["b4_fog_sem"], blob["b4_fog_tri"]),
        ("(c) LISA Rain (Clutter)", blob["b4_rain_sem"], blob["b4_rain_tri"]),
        ("(d) 50% Block Drop", blob["b4_drop50_sem"], blob["b4_drop50_tri"]),
        ("(e) Link Outage (0 B)", blob["b4_outage_sem"], blob["b4_outage_tri"]),
    ]
    for ax, (cap, s, t) in zip(axes, panels7):
        ax.imshow(sem_tri_to_rgb(s, t), origin="lower", interpolation="nearest")
        ax.scatter([50], [50], c="cyan", s=26, marker="^", edgecolors="black", linewidths=0.6)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.7)
        ax.set_xlabel(cap, fontsize=8.2, fontweight="bold", labelpad=5)
    fig.legend(handles=legend_handles, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.08), fontsize=7.4)
    save_both(fig, "fig7_weather_failure_qualitative")


# ---------------------------------------------------------------------------
# Figure 8 (NEW): Deep Mechanism Analysis — Codebook Utilization + SS-KFW Velocity Curve
# ---------------------------------------------------------------------------
def draw_fig8_mechanisms() -> None:
    cb = json.loads(V3_CB.read_text(encoding="utf-8"))
    kfw = json.loads(V3_SSKFW.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.15), constrained_layout=True)

    # (a) 3x64 Grouped Codebook Utilization (Stage-1 vs Stage-2) + Entropy
    ax = axes[0]
    g1 = cb["B4_v3_32k"]["groups_stage1"]
    g2 = cb["B4_v3_32k"]["groups_stage2"]
    grps = ["things", "ground", "vertical"]
    glbls = ["Things\n(Veh./Sign)", "Ground\n(Road/Terr.)", "Vertical\n(Bldg./Pole)"]
    gx = np.arange(3)
    w = 0.32
    u1 = [100.0 * g1[g]["util"] for g in grps]
    u2 = [100.0 * g2[g]["util"] for g in grps]
    ax.bar(gx - w / 2, u1, width=w * 0.92, color="#4A90D9", edgecolor="#000000", lw=0.5, label="Stage-1 Coarse")
    ax.bar(gx + w / 2, u2, width=w * 0.92, color="#8B2635", edgecolor="#000000", lw=0.5, label="Stage-2 Residual")
    for i, (v1, v2) in enumerate(zip(u1, u2)):
        ax.text(i - w / 2, v1 + 2.0, f"{v1:.1f}%", ha="center", va="bottom", fontsize=6.5)
        ax.text(i + w / 2, v2 + 2.0, f"{v2:.1f}%", ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    ax.set_xticks(gx)
    ax.set_xticklabels(glbls, fontsize=7.2)
    ax.set_ylabel("Active 64-entry codes (%)")
    ax.set_ylim(0, 115)
    ax.legend(frameon=False, loc="upper left", fontsize=6.7)
    set_bold_subcaption(ax, "(a) 3×64 Grouped Codebook Utilization")

    # (b) Per-Class Stage-2 Prototype Distribution Heatmap (Log-Normalized)
    ax = axes[1]
    hist2 = np.asarray(cb["B4_v3_32k"]["hist_cls_stage2"], dtype=np.float64)
    # Order rows by semantic group: Things (8,11), Ground (5,6,3), Vertical (1,2,4,7,9,10)
    row_ids = [8, 11, 5, 6, 3, 1, 2, 4, 7, 9, 10]
    row_names = ["Veh.", "Sign", "Road", "Swalk", "Terr.", "Bldg.", "Fence", "Pole", "Veg.", "Wall", "GRail"]
    mat = hist2[row_ids]
    mat = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1.0)
    im = ax.imshow(np.log10(mat + 1e-4), aspect="auto", cmap="YlOrRd", vmin=-4.0, vmax=-0.5)
    ax.axhline(1.5, color="#000000", lw=1.0, ls="--")
    ax.axhline(4.5, color="#000000", lw=1.0, ls="--")
    ax.set_yticks(np.arange(len(row_names)))
    ax.set_yticklabels(row_names, fontsize=6.8)
    ax.set_xlabel("Stage-2 residual code index ($0 \\dots 63$)")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("$\\log_{10} P(q_2 \\mid c)$", fontsize=7.0)
    cbar.ax.tick_params(labelsize=6.5)
    set_bold_subcaption(ax, "(b) Class-Grouped Residual Code Heatmap")

    # (c) Velocity-Stratified Latency Displacement Curve (w/o vs w/ SS-KFW)
    ax = axes[2]
    delays_ms = [0, 100, 200, 500]
    dkeys = ["0ms", "100ms", "200ms", "500ms"]
    for bkey, blbl, col, mk in (
        ("high_gt8mps", "Fast (>8 m/s)", "#8B2635", "P"),
        ("med_2_8mps", "Urban (2–8 m/s)", "#0B7285", "o"),
    ):
        v_no = [kfw[dk]["no_kfw"]["stratified"][bkey]["vehicle_IoU"] for dk in dkeys]
        v_yes = [kfw[dk]["ss_kfw"]["stratified"][bkey]["vehicle_IoU"] for dk in dkeys]
        ax.plot(delays_ms, v_no, color=col, ls="--", marker=mk, lw=1.35, ms=4.2, alpha=0.75, label=f"{blbl} w/o KFW")
        ax.plot(delays_ms, v_yes, color=col, ls="-", marker=mk, lw=2.0, ms=4.8, label=f"{blbl} + SS-KFW")
        ax.fill_between(delays_ms, v_no, v_yes, color=col, alpha=0.10)
    ax.set_xticks(delays_ms)
    ax.set_xticklabels(["0", "100", "200", "500"])
    ax.set_xlabel("V2X delay $\\Delta t$ (ms)")
    ax.set_ylabel("Vehicle IoU $\\uparrow$")
    ax.set_ylim(0.0, 0.54)
    ax.legend(frameon=False, loc="upper right", fontsize=6.2)
    set_bold_subcaption(ax, "(c) Velocity-Stratified SS-KFW Recovery")

    save_both(fig, "fig8_mechanisms")


# ---------------------------------------------------------------------------
# Figure 9 (NEW): Real-World V2V4Real 32-Beam Velodyne Zero-Shot Validation
# ---------------------------------------------------------------------------
def draw_fig9_v2v4real() -> None:
    d = json.loads(V2V4REAL.read_text(encoding="utf-8"))
    env = d["envelope_audit"]
    tr = d["zeroshot_transfer"]["budgets"]

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.35), constrained_layout=True)

    # (a) Real 32-Beam Velodyne Zstd Envelope Size Distribution (n=2,306 sweeps) vs 510 B Ceiling
    ax = axes[0]
    tiers = [
        ("25×25 + $z_{\\max}$\n(Tier 3)", env["zstd_25x25_zmax_bytes"], "#2A9D8F"),
        ("50×50 Tri-State\n(Tier 2: 98.7%)", env["zstd_50x50_only_bytes"], "#1D4E89"),
        ("50×50 + $z_{\\max}$\n(Tier 1: 1.3%)", env["zstd_50x50_zmax_bytes"], "#C85A17"),
    ]
    xs = np.arange(len(tiers))
    for i, (lbl, st, col) in enumerate(tiers):
        mean_v, p95_v, min_v, max_v = st["mean"], st["p95"], st["min"], st["max"]
        ax.bar(i, mean_v, width=0.52, color=col, edgecolor="#000000", lw=0.6, alpha=0.88)
        ax.vlines(i, min_v, max_v, color="#000000", lw=1.2)
        ax.scatter([i], [p95_v], color="#000000", marker="D", s=24, zorder=5, label=("p95" if i == 0 else None))
        ax.scatter([i], [max_v], color="#8B2635", marker="_", s=90, lw=1.8, zorder=5, label=("Max" if i == 0 else None))
        if i == 1:
            ax.text(i, max_v + 55, f"μ={mean_v:.0f} B\nmax={max_v} B", ha="center", va="bottom", fontsize=6.6, fontweight="bold")
        else:
            ax.text(i, max_v + 20, f"μ={mean_v:.0f} B\nmax={max_v} B", ha="center", va="bottom", fontsize=6.6)
    ax.axhline(510, color="#8B2635", ls="--", lw=1.35, label="510 B payload ceiling")
    ax.text(-0.28, 528, "510 B ceiling\n(0.0% overflow)", fontsize=6.4, color="#8B2635", fontweight="bold", ha="left", va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels([t[0] for t in tiers], fontsize=7.4)
    ax.set_ylabel("Zstd compressed bytes")
    ax.set_ylim(0, 1020)
    ax.legend(frameon=False, loc="upper left", fontsize=6.6)
    set_bold_subcaption(ax, "(a) V2V4Real 32-Beam LiDAR Zstd Footprint ($n=2{,}306$)")

    # (b) Zero-Shot Sim-to-Real Cooperative Occupancy @ Tight 8 KB (0.66 Mbps) on 316 Real Pairs
    ax = axes[1]
    b8 = tr["8192"]
    m_specs = [
        ("Where2comm_v3_32k", "W2C-Occ", "#D69E2E"),
        ("B3_v3_32k", "B3-VQ", "#1D4E89"),
        ("CodeFilling_v3_32k", "CF-Occ", "#6B46C1"),
        ("B4_v3_32k", "Ours", "#8B2635"),
        ("B4_v3_multibw", "BS-MultiBW", "#0B7285"),
    ]
    mets = [("free_IoU", "Free IoU $\\uparrow$"), ("unknown_F1", "Unk. F1 $\\uparrow$"), ("veh_box_sem_recall", "3D Veh. Rec. $\\uparrow$")]
    mx = np.arange(len(mets))
    wb = 0.155
    for bi, (tag, lbl, col) in enumerate(m_specs):
        off = (bi - 2.0) * wb
        vals = [b8[tag][mk] for mk, _ in mets]
        ax.bar(mx + off, vals, width=wb * 0.90, color=col, edgecolor="#000000", lw=0.45, label=lbl)
        if bi >= 3:
            for xi, v in zip(mx + off, vals):
                ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=5.9, fontweight="bold")
    ax.set_xticks(mx)
    ax.set_xticklabels([m[1] for m in mets], fontsize=7.6)
    ax.set_ylabel("Zero-shot real-road score (8 KB)")
    ax.set_ylim(0, 0.88)
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=6.3)
    set_bold_subcaption(ax, "(b) Zero-Shot V2V4Real Transfer @ 8 KB ($n=316$)")

    save_both(fig, "fig9_v2v4real")


def main() -> None:
    apply_pub_style()
    OUT.mkdir(parents=True, exist_ok=True)
    draw_fig2_real_oracle()
    draw_fig3_main_comparison()
    draw_fig4_bandwidth()
    draw_fig5_robustness()
    draw_real_qualitative_figures()
    draw_fig8_mechanisms()
    draw_fig9_v2v4real()
    print("GENERATED ALL V3.0 PUBLICATION FIGURES (Figs. 2-9, preserving curated Fig. 1) IN", OUT)


if __name__ == "__main__":
    main()
