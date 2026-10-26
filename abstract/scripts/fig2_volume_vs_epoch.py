"""
Figure 2 of the abstract: volume of the closed meshes along each dissection.

(a) Volume against epoch for the 12 specimens (main sequence, "b" epochs left
out); the steps where the volume increases are drawn in orange.
(b) Relative volume change of every step, against the uncertainty band of the
closing (+/- 3.4 %, the largest difference between the planar caps and a flat
fan on the same loops over the 130 epochs).

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
BASE, ACCENT, BAND_FILL = "#a3a29b", "#eb6834", "#cde2fb"
# Horizontal offset (points) and alignment of the labels of the largest increases
LABEL_OFFSETS = {"spc-17": (-4, "right")}


def main() -> None:
    by_spc: dict[str, list[tuple[int, float]]] = {}
    with open(ROOT / "brain_data" / "output" / "results.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["volume"] and not r["epoch"].endswith("b"):
                by_spc.setdefault(r["specimen"], []).append((int(r["epoch"]), float(r["volume"]) * 1e6))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.linewidth": 0.6})
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.4, 3.9), gridspec_kw={"height_ratios": [1.6, 1]})

    changes, inc = [], []
    for spc, pts in sorted(by_spc.items()):
        pts.sort()
        e, v = np.array(pts).T
        ax.plot(e, v, color=BASE, lw=1.0, marker="o", ms=2.2, mew=0, zorder=2)
        for i in range(len(v) - 1):
            d = (v[i + 1] - v[i]) / v[i] * 100
            changes.append(d)
            if d > 0:
                inc.append(d)
                ax.plot(e[i:i + 2], v[i:i + 2], color=ACCENT, lw=1.8, solid_capstyle="round", zorder=3)
                ax.plot(e[i + 1], v[i + 1], "o", color=ACCENT, ms=3.2, mec="white", mew=0.6, zorder=4)
                if d > 10:
                    dx, ha = LABEL_OFFSETS.get(spc, (4, "left"))
                    ax.annotate(f"{spc} +{d:.0f}%", (e[i + 1], v[i + 1]), xytext=(dx, 4),
                                textcoords="offset points", fontsize=5.5, color=INK2, ha=ha)
    ax.set_xlabel("Dissection epoch")
    ax.set_ylabel("Closed-mesh volume (cm³)")
    ax.set_xticks(range(1, 15))
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.plot([], [], color=BASE, lw=1.0, marker="o", ms=2.2, label="12 specimens")
    ax.plot([], [], color=ACCENT, lw=1.8, label=f"volume increase ({len(inc)} of {len(changes)} steps)")
    ax.legend(frameon=False, fontsize=6, loc="upper right", labelcolor=INK)
    ax.set_title("a", loc="left", fontweight="bold", fontsize=8, color=INK)

    c = np.array(changes)
    rng = np.random.default_rng(1)
    y = rng.uniform(-0.35, 0.35, len(c))
    bx.axvspan(-BAND, BAND, color=BAND_FILL, lw=0, zorder=0)
    bx.plot([0, 0], [-0.6, 0.5], color=INK2, lw=0.6, zorder=1)
    bx.scatter(c[c <= 0], y[c <= 0], s=9, color=BASE, edgecolor="white", lw=0.4, zorder=2)
    bx.scatter(c[c > 0], y[c > 0], s=12, color=ACCENT, edgecolor="white", lw=0.4, zorder=3)
    within = int((np.abs(c) < BAND).sum())
    bx.text(0, 0.6, f"closing uncertainty ±{BAND}%\n{within} of {len(c)} steps inside",
            ha="center", va="bottom", fontsize=5.8, color=INK2, zorder=4)
    bx.set_ylim(-0.6, 1.25)
    bx.set_yticks([])
    bx.set_xlabel("Volume change between consecutive epochs (%)")
    bx.spines[["top", "right", "left"]].set_visible(False)
    bx.set_title("b", loc="left", fontweight="bold", fontsize=8, color=INK)

    fig.tight_layout(h_pad=1.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, facecolor="white")
    print(OUT, f"steps={len(c)} increases={len(inc)} inside band={within}")


if __name__ == "__main__":
    main()
