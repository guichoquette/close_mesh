"""
Figure 3 of the abstract: volume of the closed meshes along each dissection.

(a) Small multiples, one per specimen: volume against epoch (main sequence,
"b" epochs left out), with the closing uncertainty (+/- 3.4 %, the largest
difference between the planar caps and a flat fan on the same loops over the
130 epochs) as a grey band around each epoch. Red is reserved for the changes
where the volume increases.
(b) How much the closing method changes the volume: each epoch closed
otherwise (flat fan on the same loops, MeshFix, fold variants A and B),
against our closing. Reads ``diagnostics/closing_uncertainty.csv``
(``closing_uncertainty.py``), ``diagnostics/meshfix_volumes.csv``
(``compare_meshfix.py``) and ``brain_data/verification/options/comparison.csv``
(``experiments/compare_options.py``).

Usage
-----
python abstract/scripts/fig2_volume_vs_epoch.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "abstract" / "figures" / "fig2_volume_vs_epoch.png"
BAND = 3.4  # % : largest planar cap vs flat fan difference (README, 130 epochs)

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BASE, BAND_FILL = "#8f8e87", "#cfccc3"
INCREASE = "#d0312d"      # reserved for volume increases
METHODS = ["#2a78d6", "#1f9e89", "#7b4fc9", "#c48a00"]  # one per closing method, panel (b)


def read_column(path: Path, column: str, sign: float = 1.0) -> np.ndarray:
    """Values of a column of a CSV file, skipping empty cells."""
    if not path.exists():
        return np.array([])
    with open(path, newline="", encoding="utf-8") as f:
        return np.array([sign * float(r[column]) for r in csv.DictReader(f) if r[column]])


def read_volumes() -> dict[str, np.ndarray]:
    """Epochs and volumes (cm3) of the main sequence of each specimen."""
    by_spc: dict[str, list[tuple[int, float]]] = {}
    with open(ROOT / "brain_data" / "output" / "results.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["volume"] and not r["epoch"].endswith("b"):
                by_spc.setdefault(r["specimen"], []).append((int(r["epoch"]), float(r["volume"]) * 1e6))
    return {spc: np.array(sorted(pts)).T for spc, pts in sorted(by_spc.items())}


def specimen_panel(ax: plt.Axes, spc: str, e: np.ndarray, v: np.ndarray) -> tuple[int, int]:
    """Volume of one specimen; return (steps, increases)."""
    ax.fill_between(e, v * (1 - BAND / 100), v * (1 + BAND / 100), color=BAND_FILL, lw=0, zorder=1)
    ax.plot(e, v, color=BASE, lw=1.0, marker="o", ms=2.0, mew=0, zorder=2)
    increases = 0
    for i in range(len(v) - 1):
        d = (v[i + 1] - v[i]) / v[i] * 100
        if d > 0:
            increases += 1
            ax.plot(e[i:i + 2], v[i:i + 2], color=INCREASE, lw=1.8, solid_capstyle="round", zorder=3)
            ax.plot(e[i + 1], v[i + 1], "o", color=INCREASE, ms=2.8, mec="white", mew=0.5, zorder=4)
            if d >= 5:
                ax.annotate(f"+{d:.0f}%", (e[i + 1], v[i + 1]), xytext=(0, 4), textcoords="offset points",
                            fontsize=5.5, color=INK, ha="center", va="bottom", zorder=5)
    ax.text(0.97, 0.95, spc, transform=ax.transAxes, fontsize=6.5, color=INK, ha="right", va="top")
    ax.grid(axis="y", color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    return len(v) - 1, increases


def closing_methods(bx: plt.Axes) -> None:
    """Panel (b): volume difference of each other closing against ours, one row per method."""
    diag = ROOT / "abstract" / "diagnostics"
    options = ROOT / "brain_data" / "verification" / "options" / "comparison.csv"
    rows = [("flat fan on the same openings", read_column(diag / "closing_uncertainty.csv", "wedge_%", -1)),
            ("MeshFix (independent tool)", read_column(diag / "meshfix_volumes.csv", "difference_%")),
            ("fold variant A (always planar)", read_column(options, "option_a_volume_change_%")),
            ("fold variant B (planar if ≤5%)", read_column(options, "option_b_volume_change_%"))]
    rng = np.random.default_rng(2)
    bx.axvspan(-BAND, BAND, color=BAND_FILL, alpha=0.35, lw=0, zorder=0)
    bx.axvline(0, color=INK2, lw=0.6, zorder=1)
    for k, ((_, d), color) in enumerate(zip(rows, METHODS)):
        # Box (quartiles, whiskers at 1.5 IQR) under the points, which show every epoch
        bx.boxplot(d, positions=[-k], orientation="horizontal", widths=0.55, showfliers=False, zorder=2,
                   boxprops={"color": color, "lw": 0.8}, whiskerprops={"color": color, "lw": 0.8},
                   capprops={"color": color, "lw": 0.8}, medianprops={"color": color, "lw": 1.4})
        y = -k + rng.uniform(-0.18, 0.18, len(d))
        bx.scatter(d, y, s=8, color=color, alpha=0.75, edgecolor="white", lw=0.3, zorder=3)
    bx.set_yticks(-np.arange(len(rows)), [f"{name}, n={len(d)}" for name, d in rows], fontsize=6.5)
    for label, color in zip(bx.get_yticklabels(), METHODS):
        label.set_color(color)
    bx.tick_params(axis="y", length=0)
    bx.set_xlim(-4.5, 12.5)
    bx.set_ylim(-len(rows) + 0.5, 0.5)
    bx.text(BAND, 0.52, f"±{BAND}%", ha="right", va="bottom", fontsize=6, color=INK2)
    bx.set_xlabel("Volume difference from our closing, per epoch (%)")
    bx.spines[["top", "right", "left"]].set_visible(False)


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.linewidth": 0.6})
    volumes = read_volumes()
    fig = plt.figure(figsize=(7.0, 5.8))
    grid = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.12, left=0.08, right=0.985,
                            top=0.935, bottom=0.40)
    steps = increases = 0
    axes = [fig.add_subplot(grid[i // 4, i % 4]) for i in range(12)]
    for ax in axes[1:]:
        ax.sharex(axes[0])
        ax.sharey(axes[0])
    for k, (ax, (spc, (e, v))) in enumerate(zip(axes, volumes.items())):
        s, n = specimen_panel(ax, spc, e, v)
        steps += s
        increases += n
        ax.set_xticks([1, 5, 10, 14])
        ax.set_ylim(140, 720)
        ax.tick_params(labelsize=6)
        if k % 4:
            ax.tick_params(labelleft=False)
        if k < 8:
            ax.tick_params(labelbottom=False)
    axes[4].set_ylabel("Closed-mesh volume (cm³)")
    fig.text(0.53, 0.355, "Dissection epoch", ha="center", fontsize=6.5, color=INK)
    fig.text(0.01, 0.985, "a", fontsize=8, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.985, f"Volume along each dissection: red = increase ({increases} of {steps} "
             f"epoch-to-epoch changes); "
             f"grey band: closing uncertainty (±{BAND}%)", fontsize=7, color=INK, va="top")

    bx = fig.add_gridspec(1, 1, left=0.33, right=0.985, top=0.255, bottom=0.07).subplots()
    closing_methods(bx)
    fig.text(0.01, 0.305, "b", fontsize=8, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.305, "How much does the closing method change the volume? "
             "(one point per epoch; box: quartiles)", fontsize=7, color=INK, va="top")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, facecolor="white")
    print(OUT, f"steps={steps} increases={increases}")


if __name__ == "__main__":
    main()
