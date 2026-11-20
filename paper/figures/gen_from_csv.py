"""Copy paper tables and plots from the measurement files.

Reads experiments/registry.csv, outputs/phase5/measurements.csv, and
outputs/phase5/ablations.csv. Does not train, and does not format metric
cells into new numeric strings.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REGISTRY = ROOT / "experiments" / "registry.csv"
MEASUREMENTS = ROOT / "outputs" / "phase5" / "measurements.csv"
ABLATIONS = ROOT / "outputs" / "phase5" / "ablations.csv"

GATE_METHODS = ("B0", "B1", "B2", "B3", "B4")
GATE_STATUS = {"no_communication", "loss_dropped"}
STRESS_FIELDS = (
    "model_name",
    "level",
    "bytes_per_frame",
    "bytes_min",
    "bytes_max",
    "mIoU",
    "vehicle_IoU",
    "road_IoU",
    "unknown_F1",
    "unobserved_as_free_err",
)
GATE_FIELDS = (
    "model_name",
    "bytes_per_frame",
    "mbps_10hz",
    "mIoU",
    "vehicle_IoU",
    "road_IoU",
    "unknown_F1",
    "unobserved_as_free_err",
    "status",
)
FAMILIES = (
    ("bandwidth", "byte budget", "fig_bandwidth", "tab_bandwidth"),
    ("senders", "sender count", "fig_senders", "tab_senders"),
    ("weather", "weather", "fig_weather", "tab_weather"),
    ("pose", "pose level", "fig_pose", "tab_pose"),
    ("latency", "latency level", "fig_latency", "tab_latency"),
    ("packet_loss", "packet-loss level", "fig_packet_loss", "tab_packet_loss"),
    ("seed", "seed", "fig_seeds", "tab_seeds"),
)
CURVE_METRICS = (
    ("mIoU", "mIoU"),
    ("vehicle_IoU", "vehicle IoU"),
    ("unobserved_as_free_err", "unobserved-as-free error"),
)
BAR_METRICS = CURVE_METRICS
SERIES = {
    "B3": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
    "B4": {"color": "#D55E00", "linestyle": "--", "marker": "s"},
}
BAR_STYLE = {
    "B0": {"color": "#000000", "hatch": ""},
    "B1": {"color": "#56B4E9", "hatch": "//"},
    "B2": {"color": "#009E73", "hatch": "\\\\"},
    "B3": {"color": "#0072B2", "hatch": "xx"},
    "B4": {"color": "#D55E00", "hatch": ".."},
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def tex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def cell(value: str) -> str:
    if value == "":
        return "--"
    return tex_escape(value)


def assert_in_source(values: list[str], source: str, label: str) -> None:
    missing = [value for value in values if value != "" and value not in source]
    if missing:
        raise SystemExit(f"{label} has cells that are not in the source file: {missing[:8]}")


def latest_gate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    chosen: dict[str, dict[str, str]] = {}
    for row in rows:
        if row["model_name"] in GATE_METHODS and row["status"] in GATE_STATUS:
            chosen[row["model_name"]] = row
    missing = [name for name in GATE_METHODS if name not in chosen]
    if missing:
        raise SystemExit(f"registry is missing a gate row for {missing}")
    return [chosen[name] for name in GATE_METHODS]


def write_tabular(
    path: Path,
    comment: str,
    caption: str,
    label: str,
    headers: list[str],
    body: list[list[str]],
) -> None:
    align = "l" * len(headers)
    lines = [
        comment,
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in body:
        lines.append(" & ".join(cell(value) for value in row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def apply_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.size": 10,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png")
    plt.close(fig)


def plot_gate(rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45), constrained_layout=True)
    names = [row["model_name"] for row in rows]
    positions = list(range(len(names)))
    for axis, (key, ylabel) in zip(axes, BAR_METRICS):
        for position, row in zip(positions, rows):
            style = BAR_STYLE[row["model_name"]]
            axis.bar(
                position,
                float(row[key]),
                width=0.72,
                color=style["color"],
                hatch=style["hatch"],
                edgecolor="black",
                linewidth=0.4,
            )
        axis.set_xticks(positions)
        axis.set_xticklabels(names)
        axis.set_ylabel(ylabel)
    save(fig, "fig_gate3")


def plot_family(rows: list[dict[str, str]], family: str, xlabel: str, stem: str) -> None:
    grouped = {
        method: [row for row in rows if row["family"] == family and row["model_name"] == method]
        for method in ("B3", "B4")
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55), constrained_layout=True)
    for axis, (key, ylabel) in zip(axes, CURVE_METRICS):
        for method in ("B3", "B4"):
            points = [row for row in grouped[method] if row[key] != ""]
            if not points:
                continue
            style = SERIES[method]
            xs = [float(row["level"]) if family not in {"weather", "pose"} else None for row in points]
            if family in {"weather", "pose"}:
                xs = list(range(len(points)))
            axis.plot(
                xs,
                [float(row[key]) for row in points],
                label=method,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=4.5,
                linewidth=1.4,
            )
            axis.set_xticks(xs)
            axis.set_xticklabels([row["level"] for row in points], rotation=30 if family == "bandwidth" else 0)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.14))
    save(fig, stem)


def write_ablation_note(rows: list[dict[str, str]]) -> None:
    lines = [
        "% Ablations were not run. This is not a result table.",
        "% Reasons are copied from outputs/phase5/ablations.csv.",
        "\\paragraph{Ablations.}",
        "These ablations were not run. No metric cells are filled.",
        "\\begin{itemize}",
    ]
    for row in rows:
        lines.append(
            f"\\item \\texttt{{{tex_escape(row['name'])}}}: {tex_escape(row['status'])}. {tex_escape(row['reason'])}"
        )
    lines.extend(["\\end{itemize}", ""])
    (OUT / "ablations_not_run.tex").write_text("\n".join(lines), encoding="utf-8")


def write_includes() -> None:
    lines = [
        "% Include from the paper/ directory. Tables already contain table environments.",
        "% Requires booktabs. Synthetic Mini-200 only; not OPV2V acceptance.",
        "",
        "\\input{figures/tab_gate3.tex}",
        "",
        "\\begin{figure}[t]",
        "\\centering",
        "\\includegraphics[width=0.95\\textwidth]{figures/fig_gate3.pdf}",
        "\\caption{Synthetic Mini-200 gate comparison copied from the last matching rows of \\texttt{experiments/registry.csv}. Bar heights are those cells. This is not acceptance on real OPV2V.}",
        "\\label{fig:gate3}",
        "\\end{figure}",
        "",
    ]
    for family, _xlabel, stem, table in FAMILIES:
        lines.extend(
            [
                f"\\input{{figures/{table}.tex}}",
                "",
                "\\begin{figure}[t]",
                "\\centering",
                f"\\includegraphics[width=0.95\\textwidth]{{figures/{stem}.pdf}}",
                f"\\caption{{Synthetic Mini-200 \\texttt{{{tex_escape(family)}}} rows copied from \\texttt{{outputs/phase5/measurements.csv}}. Tick labels are the CSV level strings.}}",
                f"\\label{{fig:{family}}}",
                "\\end{figure}",
                "",
            ]
        )
    lines.extend(
        [
            "\\input{figures/ablations_not_run.tex}",
            "",
        ]
    )
    (OUT / "latex_includes.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    apply_style()
    registry_text = REGISTRY.read_text(encoding="utf-8")
    measurement_text = MEASUREMENTS.read_text(encoding="utf-8")
    ablation_text = ABLATIONS.read_text(encoding="utf-8")
    registry = read_rows(REGISTRY)
    measurements = read_rows(MEASUREMENTS)
    ablations = read_rows(ABLATIONS)

    gate = latest_gate_rows(registry)
    gate_cells = [row[field] for row in gate for field in GATE_FIELDS]
    assert_in_source(gate_cells, registry_text, "gate table")
    commits = sorted({row["git_commit"] for row in gate})
    write_tabular(
        OUT / "tab_gate3.tex",
        "% Cells copied from experiments/registry.csv. "
        "Selection: last no_communication or loss_dropped row for B0, B1, B2, B3, and B4. "
        f"git_commit values in those rows: {', '.join(commits)}.",
        "Synthetic Mini-200 gate rows copied from the registry. "
        "Each cell is the last \\texttt{no\\_communication} or \\texttt{loss\\_dropped} row of that method. "
        "This is not acceptance on real OPV2V.",
        "tab:gate3",
        [
            "method",
            "bytes",
            "mbps at 10 Hz",
            "mIoU",
            "vehicle IoU",
            "road IoU",
            "unknown F1",
            "UFE",
            "status",
        ],
        [[row[field] for field in GATE_FIELDS] for row in gate],
    )
    plot_gate(gate)

    measured = [row for row in measurements if row["status"] == "measured"]
    for family, xlabel, stem, table in FAMILIES:
        family_rows = [row for row in measured if row["family"] == family]
        if not family_rows:
            raise SystemExit(f"no measured rows for {family}")
        cells = [row[field] for row in family_rows for field in STRESS_FIELDS]
        assert_in_source(cells, measurement_text, table)
        write_tabular(
            OUT / f"{table}.tex",
            f"% Cells copied from outputs/phase5/measurements.csv, family={family}.",
            f"Synthetic Mini-200 \\texttt{{{tex_escape(family)}}} rows copied from "
            "\\texttt{outputs/phase5/measurements.csv}. Empty CSV cells are left as --.",
            f"tab:{family}",
            [
                "method",
                "level",
                "bytes",
                "bytes min",
                "bytes max",
                "mIoU",
                "vehicle IoU",
                "road IoU",
                "unknown F1",
                "UFE",
            ],
            [[row[field] for field in STRESS_FIELDS] for row in family_rows],
        )
        plot_family(measured, family, xlabel, stem)

    if any(row["status"] != "not_run" for row in ablations):
        raise SystemExit("ablations.csv contains a status other than not_run; refusing to invent a result table")
    for row in ablations:
        for field in ("name", "status", "reason"):
            if row[field] not in ablation_text:
                raise SystemExit(f"ablation text missing from source: {row['name']} {field}")
    write_ablation_note(ablations)
    write_includes()
    print(f"gate commits: {commits}")
    print(f"gate rows: {[row['exp_id'] for row in gate]}")
    print(f"measured rows: {len(measured)}")
    print(f"ablations: {[row['name'] + ':' + row['status'] for row in ablations]}")


if __name__ == "__main__":
    main()
