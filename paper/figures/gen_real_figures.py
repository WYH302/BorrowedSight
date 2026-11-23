"""Generate publication-grade figures (300 DPI PDF/PNG) and booktabs LaTeX tables
verbatim from the measured Real OPV2V + Semantic-OPV2V CSVs.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures_real"
P4_CSV = ROOT / "outputs" / "real_opv2v" / "phase4_gate3" / "results.csv"
P5_CSV = ROOT / "outputs" / "real_opv2v" / "phase5" / "measurements.csv"
COMPUTE_JSON = ROOT / "outputs" / "real_opv2v" / "phase5" / "compute_benchmark.json"

SERIES = {
    "B1": {"color": "#56B4E9", "linestyle": ":", "marker": "^", "label": "B1 (Sparse FP16)"},
    "B2": {"color": "#009E73", "linestyle": "-.", "marker": "d", "label": "B2 (Cont. 8-d)"},
    "B3": {"color": "#0072B2", "linestyle": "-", "marker": "o", "label": "B3 (Shared VQ)"},
    "B4": {"color": "#D55E00", "linestyle": "--", "marker": "s", "label": "B4 (BorrowedSight)"},
    "B4_no_envelope": {"color": "#CC79A7", "linestyle": ":", "marker": "v", "label": "B4 w/o Env"},
}
BAR_STYLE = {
    "B0": {"color": "#333333", "hatch": ""},
    "B1": {"color": "#56B4E9", "hatch": "//"},
    "B2": {"color": "#009E73", "hatch": "\\\\"},
    "B3": {"color": "#0072B2", "hatch": "xx"},
    "B4": {"color": "#D55E00", "hatch": ".."},
}


def apply_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.size": 9.5,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as h:
        return list(csv.DictReader(h))


def fmt(val: str | float | None, digits: int = 4) -> str:
    if val is None or val == "":
        return "--"
    return f"{float(val):.{digits}f}"


def tex_esc(s: str) -> str:
    return s.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def write_table(path: Path, caption: str, label: str, headers: list[str], body: list[list[str]]) -> None:
    align = "l" + "r" * (len(headers) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for r in body:
        lines.append(" & ".join(r) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def save_fig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png")
    plt.close(fig)


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    p4 = read_csv(P4_CSV)
    p5 = read_csv(P5_CSV)

    # 1. Main Gate 3 table (32KB & 16KB on held-out TEST)
    test_p4 = [r for r in p4 if r["split"] == "test" and r["seed"] == "0"]
    main_tags = ["B0", "B1", "B2", "B3", "B4"]
    main_rows = [next(r for r in test_p4 if r["tag"] == t) for t in main_tags if any(r["tag"] == t for r in test_p4)]
    write_table(
        OUT / "tab_real_main_32k.tex",
        "Real OPV2V + Semantic-OPV2V held-out test comparison at matched serialized budget (32,768 bytes/packet, 2.62 Mbps at 10 Hz, seed 0).",
        "tab:real_main_32k",
        ["Method", "Bytes", "mIoU", "Veh IoU", "Road IoU", "Unk F1", "UFE $\\downarrow$", "Unc."],
        [
            [
                tex_esc(r["tag"]),
                fmt(r["bytes_per_frame"], 0),
                fmt(r["mIoU"]),
                fmt(r["vehicle_IoU"]),
                fmt(r["road_IoU"]),
                fmt(r["unknown_F1"]),
                fmt(r["unobserved_as_free_err"]),
                fmt(r["mean_uncertainty"]),
            ]
            for r in main_rows
        ],
    )

    # Gate 3 bar chart
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6), constrained_layout=True)
    xs = np.arange(len(main_rows))
    for ax, (col, ylabel) in zip(
        axes,
        [("mIoU", "mIoU $\\uparrow$"), ("vehicle_IoU", "Vehicle IoU $\\uparrow$"), ("unobserved_as_free_err", "UFE $\\downarrow$")],
    ):
        for i, r in enumerate(main_rows):
            st = BAR_STYLE[r["tag"]]
            ax.bar(i, float(r[col]), width=0.68, color=st["color"], hatch=st["hatch"], edgecolor="black", linewidth=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([r["tag"] for r in main_rows])
        ax.set_ylabel(ylabel)
    save_fig(fig, "fig_real_gate3_32k")

    # 2. Tight-budget 16KB table (if present)
    tags_16k = ["B1_16k", "B2_16k", "B3_16k", "B4_16k"]
    rows_16k = [next(r for r in test_p4 if r["tag"] == t) for t in tags_16k if any(r["tag"] == t for r in test_p4)]
    if rows_16k:
        write_table(
            OUT / "tab_real_main_16k.tex",
            "Real OPV2V held-out test comparison at tight serialized budget (16,384 bytes/packet, 1.31 Mbps at 10 Hz, seed 0).",
            "tab:real_main_16k",
            ["Method", "Bytes", "mIoU", "Veh IoU", "Road IoU", "Unk F1", "UFE $\\downarrow$"],
            [
                [
                    tex_esc(r["tag"]),
                    fmt(r["bytes_per_frame"], 0),
                    fmt(r["mIoU"]),
                    fmt(r["vehicle_IoU"]),
                    fmt(r["road_IoU"]),
                    fmt(r["unknown_F1"]),
                    fmt(r["unobserved_as_free_err"]),
                ]
                for r in rows_16k
            ],
        )

    # 3. Core 4-Ablation table (32KB & 16KB on held-out TEST)
    ab_tags = ["B4", "B4_no_envelope", "B4_shared_book", "B4_no_quota", "B4_stage2"]
    ab_rows = [next(r for r in test_p4 if r["tag"] == t) for t in ab_tags if any(r["tag"] == t for r in test_p4)]
    ab_tags16 = ["B4_16k", "B4_16k_no_envelope", "B4_16k_shared_book", "B4_16k_no_quota", "B4_16k_stage2"]
    ab_rows16 = [next(r for r in test_p4 if r["tag"] == t) for t in ab_tags16 if any(r["tag"] == t for r in test_p4)]
    write_table(
        OUT / "tab_real_ablations.tex",
        "Real OPV2V core ablations at exact matched serialized bytes (32 KB and 16 KB, held-out test, seed 0).",
        "tab:real_ablations",
        ["Variant", "Bytes", "mIoU", "Veh IoU", "Road IoU", "Unk F1", "UFE $\\downarrow$"],
        [
            [
                tex_esc(r["tag"]),
                fmt(r["bytes_per_frame"], 0),
                fmt(r["mIoU"]),
                fmt(r["vehicle_IoU"]),
                fmt(r["road_IoU"]),
                fmt(r["unknown_F1"]),
                fmt(r["unobserved_as_free_err"]),
            ]
            for r in (ab_rows + ab_rows16)
        ],
    )

    # 4. Multi-seed stability table (seeds 0, 1, 2 on held-out TEST)
    seed_tags = ["B3", "B4", "B4_no_envelope", "B4_shared_book"]
    seed_body = []
    for t in seed_tags:
        sub = [r for r in p4 if r["split"] == "test" and r["tag"] == t and r["seed"] in {"0", "1", "2"}]
        if not sub:
            continue
        mi = [float(r["mIoU"]) for r in sub]
        ve = [float(r["vehicle_IoU"]) for r in sub]
        uf = [float(r["unobserved_as_free_err"]) for r in sub]
        seed_body.append(
            [
                tex_esc(t),
                str(len(sub)),
                f"{np.mean(mi):.4f} $\\pm$ {np.std(mi):.4f}",
                f"{np.mean(ve):.4f} $\\pm$ {np.std(ve):.4f}",
                f"{np.mean(uf):.4f} $\\pm$ {np.std(uf):.4f}",
            ]
        )
    write_table(
        OUT / "tab_real_seeds.tex",
        "Multi-seed stability (seeds 0, 1, 2) on real OPV2V held-out test at 32,768 bytes/packet.",
        "tab:real_seeds",
        ["Variant", "Seeds", "mIoU (mean $\\pm$ std)", "Veh IoU (mean $\\pm$ std)", "UFE $\\downarrow$ (mean $\\pm$ std)"],
        seed_body,
    )

    # 5. Stress curves from Phase 5 measurements.csv
    def plot_p5_curve(family: str, methods: list[str], xlabel: str, stem: str, categorical: bool = False):
        fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.7), constrained_layout=True)
        for ax, (col, ylabel) in zip(
            axes,
            [("mIoU", "mIoU $\\uparrow$"), ("vehicle_IoU", "Vehicle IoU $\\uparrow$"), ("unobserved_as_free_err", "UFE $\\downarrow$")],
        ):
            for m in methods:
                pts = [r for r in p5 if r["family"] == family and r["model_name"] == m and r[col] != ""]
                if not pts:
                    continue
                st = SERIES[m]
                xs = list(range(len(pts))) if (categorical or family == "bandwidth") else [float(r["level"]) for r in pts]
                ax.plot(
                    xs,
                    [float(r[col]) for r in pts],
                    label=st["label"],
                    color=st["color"],
                    linestyle=st["linestyle"],
                    marker=st["marker"],
                    markersize=4.5,
                    linewidth=1.5,
                )
                ax.set_xticks(xs)
                labels = [f"{int(r['level'])//1024}KB" for r in pts] if family == "bandwidth" else [r["level"] for r in pts]
                ax.set_xticklabels(labels, rotation=0)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.13))
        save_fig(fig, stem)

    plot_p5_curve("bandwidth", ["B1", "B2", "B3", "B4"], "Packet budget (bytes)", "fig_real_bandwidth")
    plot_p5_curve("weather", ["B1", "B2", "B3", "B4", "B4_no_envelope"], "LISA Weather", "fig_real_weather", categorical=True)
    plot_p5_curve("senders", ["B3", "B4"], "Number of Senders", "fig_real_senders")
    plot_p5_curve("density", ["B3", "B4"], "Traffic Density", "fig_real_density", categorical=True)
    plot_p5_curve("pose", ["B3", "B4"], "Pose Noise (xy_m / yaw_deg)", "fig_real_pose", categorical=True)
    plot_p5_curve("latency", ["B3", "B4"], "Latency (frames @ 10Hz)", "fig_real_latency")
    plot_p5_curve("packet_loss", ["B3", "B4"], "Micro-block Drop Rate", "fig_real_packet_loss")

    # 6. Uncertainty & Unknown monotonicity figure (B4 across Weather, Senders, Drop Rate)
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.6), constrained_layout=True)
    for ax, (fam, xl, cat) in zip(
        axes,
        [("weather", "LISA Weather", True), ("senders", "Senders", False), ("packet_loss", "Block Drop Rate", False)],
    ):
        pts = [r for r in p5 if r["family"] == fam and r["model_name"] == "B4" and r["mean_uncertainty"] != ""]
        if not pts:
            continue
        xs = list(range(len(pts))) if cat else [float(r["level"]) for r in pts]
        ax.plot(xs, [float(r["mean_uncertainty"]) for r in pts], "-s", color="#D55E00", label="Mean Uncertainty")
        ax.plot(xs, [float(r["pred_unknown_frac"]) for r in pts], "--o", color="#0072B2", label="Pred Unknown Frac")
        ax.set_xticks(xs)
        ax.set_xticklabels([r["level"] for r in pts])
        ax.set_xlabel(xl)
        ax.set_ylabel("Fraction / Risk")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.13))
    save_fig(fig, "fig_real_uncertainty_monotonicity")

    # 7. Compute benchmark LaTeX table (B6)
    if COMPUTE_JSON.exists():
        cb = json.loads(COMPUTE_JSON.read_text(encoding="utf-8"))
        c_rows = []
        for m in ("B3", "B4"):
            for k, d in sorted(cb.get(m, {}).items()):
                c_rows.append(
                    [
                        m,
                        k,
                        f"{d['tx_pack_ms']:.2f}",
                        f"{d['rx_decode_ms']:.2f}",
                        f"{d['total_ms']:.2f}",
                        f"{d['peak_vram_mb']:.1f}",
                        f"{d['total_frame_bytes']:,}",
                    ]
                )
        write_table(
            OUT / "tab_real_compute.tex",
            "Runtime, peak VRAM, and total wire bytes on RTX 3090 vs sender count (B6).",
            "tab:real_compute",
            ["Method", "Senders", "TX+Pack (ms)", "RX+Dec (ms)", "Total (ms)", "Peak VRAM (MB)", "Total Bytes"],
            c_rows,
        )
    print(f"WROTE publication figures and LaTeX tables to {OUT}", flush=True)


if __name__ == "__main__":
    main()
