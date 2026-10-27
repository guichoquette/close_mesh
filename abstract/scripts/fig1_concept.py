"""
Figure 1 of the abstract: why a watertight volume can still be wrong.

Schematic cross-section of a dissected hemisphere, open on the unphotographed
face (bottom). (a) The cap follows the edge of the scanned surface. (b) If this
edge lies 1 mm further out at the next epoch, the slab between the two caps is
added to the volume (2 to 4 % per mm). (c) A three-vertex fold of the boundary
makes its projection cross itself, and the planar cap is replaced by a warped
one. (d) An inverted piece of debris subtracts its volume. Not to scale.

Usage
-----
python abstract/scripts/fig1_concept.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "abstract" / "figures" / "fig1_concept.png"

INK, INK2 = "#0b0b0b", "#52514e"
TISSUE, SURFACE = "#dcdad4", "#6b6a64"
CAP, SLAB, INVERTED = "#9b1b1f", "#eb6834", "#2f6db5"

A, B = 1.0, 0.62      # half-axes of the hemisphere section
CUT = -0.18           # height of the scanned edge (unphotographed face below)
SHIFT = 0.12          # outward shift of the edge in (b), exaggerated
LEADER = {"arrowstyle": "-", "color": INK2, "lw": 0.5, "shrinkB": 2}


def scanned_arc(cut: float) -> np.ndarray:
    """Points of the scanned surface: the ellipse above the height ``cut``, right to left."""
    t0 = np.arcsin(cut / B)
    t = np.linspace(t0, np.pi - t0, 200)
    return np.c_[A * np.cos(t), B * np.sin(t)]


def section(ax: plt.Axes, cut: float = CUT, tissue: bool = True) -> np.ndarray:
    """Draw the closed section (tissue and scanned surface) and return the arc."""
    arc = scanned_arc(cut)
    if tissue:
        ax.add_patch(Polygon(arc, closed=True, fc=TISSUE, ec="none", zorder=1))
    ax.plot(*arc.T, color=SURFACE, lw=1.6, solid_capstyle="round", zorder=3)
    return arc


def cap(ax: plt.Axes, arc: np.ndarray) -> None:
    """Draw the planar cap joining the two ends of the scanned edge."""
    ax.plot(*arc[[0, -1]].T, color=CAP, lw=2.2, solid_capstyle="round", zorder=4)


def label(ax: plt.Axes, letter: str, text: str) -> None:
    """Panel letter and title above the drawing."""
    ax.text(-1.1, 0.95, letter, fontsize=8, fontweight="bold", color=INK, va="top")
    ax.text(-0.95, 0.95, text, fontsize=7, color=INK, va="top")


def note(ax: plt.Axes, text: str) -> None:
    """Explanation under the drawing."""
    ax.text(-1.1, -0.42, text, fontsize=6.2, color=INK2, ha="left", va="top", linespacing=1.35)


def panel_a(ax: plt.Axes) -> None:
    arc = section(ax)
    cap(ax, arc)
    ax.text(0, 0.2, "closed volume", fontsize=6.2, color=INK2, ha="center")
    ax.text(0, CUT - 0.05, "cap", fontsize=6.2, color=CAP, ha="center", va="top")
    ax.annotate("edge of the\nscanned surface", arc[0], xytext=(1.25, 0.05), fontsize=6.2,
                color=INK2, ha="left", va="center", arrowprops=LEADER)
    ax.annotate("unphotographed\nface (open)", (0.7, CUT - 0.02), xytext=(1.25, -0.3),
                fontsize=6.2, color=INK2, ha="left", va="center", arrowprops=LEADER)
    note(ax, "The mesh is open where it was not photographed;\n"
             "the cap closes it at the edge of the scanned surface.")
    label(ax, "a", "The cap follows the scanned edge")


def panel_b(ax: plt.Axes) -> None:
    old = scanned_arc(CUT)
    new = section(ax, CUT - SHIFT)
    low = new[:, 1] <= CUT
    slab = np.r_[new[low & (new[:, 0] > 0)], new[low & (new[:, 0] < 0)]]
    ax.add_patch(Polygon(slab, closed=True, fc=SLAB, ec="none", zorder=2))
    ax.plot(*old[[0, -1]].T, color=CAP, lw=1.0, ls=(0, (2, 1.5)), zorder=4)
    cap(ax, new)
    ax.annotate("previous cap", (0.85, CUT), xytext=(1.25, 0.05), fontsize=6.2, color=CAP,
                ha="left", va="center", arrowprops={**LEADER, "color": CAP})
    ax.annotate("added slab\n(edge 1 mm out)", (0.6, CUT - SHIFT / 2),
                xytext=(1.25, -0.3), fontsize=6.2, color=INK2, ha="left", va="center",
                arrowprops=LEADER)
    note(ax, "At the next epoch the edge lies 1 mm further out:\n"
             "volume +2 to 4 %, as much as one epoch of dissection.")
    label(ax, "b", "The scanned edge moves")


def panel_c(ax: plt.Axes) -> None:
    arc = section(ax, tissue=False)
    x = np.linspace(arc[0, 0], arc[-1, 0], 200)
    u = (x - x[0]) / (x[-1] - x[0])
    y = CUT + 0.10 * np.sin(3 * np.pi * u) * np.sin(np.pi * u)
    ax.add_patch(Polygon(np.r_[arc, np.c_[x, y][::-1]], closed=True, fc=TISSUE, ec="none",
                         zorder=1))
    ax.plot(*arc[[0, -1]].T, color=CAP, lw=0.8, ls=(0, (2, 1.5)), zorder=3)
    ax.plot(x, y, color=CAP, lw=2.2, solid_capstyle="round", zorder=4)
    # Zoom on the boundary: three consecutive vertices whose projection crosses itself
    zx, zy, r = 1.62, 0.25, 0.34
    ax.add_patch(Circle((zx, zy), r, fc="white", ec=INK2, lw=0.6, zorder=5))
    fold = np.array([[-0.8, 0.0], [0.25, 0.0], [-0.1, 0.4], [-0.1, -0.4], [0.8, -0.05]])
    ax.plot(*(fold * r + [zx, zy]).T, color=SURFACE, lw=1.0, marker="o", ms=2.4, zorder=6)
    ax.plot(zx - 0.1 * r, zy, marker="x", color=CAP, ms=4, mew=1.0, zorder=7)
    ax.annotate("", arc[0], (zx - r * 0.7, zy - r * 0.7),
                arrowprops={**LEADER, "shrinkA": 0})
    ax.text(zx, zy - r - 0.05, "3-vertex fold", fontsize=6.2, color=INK2, ha="center", va="top")
    note(ax, "Projected on the cap plane, the fold crosses itself:\n"
             "a warped cap replaces the planar one (dashed).")
    label(ax, "c", "A tiny fold warps the cap")


def panel_d(ax: plt.Axes) -> None:
    arc = section(ax)
    cap(ax, arc)
    cx, cy, r = 0.1, 0.18, 0.14
    ax.add_patch(Circle((cx, cy), r, fc="white", ec=INVERTED, lw=1.4, zorder=4))
    for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        d = np.array([np.cos(a), np.sin(a)])
        ax.add_patch(FancyArrowPatch((cx, cy) + d * (r + 0.1), (cx, cy) + d * (r + 0.01),
                                     arrowstyle="-|>", mutation_scale=8, color=INVERTED,
                                     lw=0.9, zorder=5))
    ax.annotate("debris with\nnormals inward", (cx + r + 0.11, cy), xytext=(1.25, 0.15),
                fontsize=6.2, color=INVERTED, ha="left", va="center",
                arrowprops={**LEADER, "color": INVERTED})
    note(ax, "Debris wound inward is invisible from outside,\n"
             "but its volume is subtracted instead of added.")
    label(ax, "d", "Inverted debris")


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.5))
    for ax, draw in zip(axes.flat, (panel_a, panel_b, panel_c, panel_d)):
        draw(ax)
        ax.set_xlim(-1.15, 2.15)
        ax.set_ylim(-0.75, 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.text(0.995, 0.01, "Not to scale", fontsize=5.5, color=INK2, ha="right", va="bottom")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02, wspace=0.04, hspace=0.08)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
