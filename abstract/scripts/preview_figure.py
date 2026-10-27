"""
Preview figure of the abstract (ECHO "Preview Figure", not reviewed).

"Closed, but not trustworthy": left, a closed epoch with its caps in red
(panel c of fig1_pipeline.png); right, two consecutive epochs overlaid, the
later one lying outside the earlier one (panel a of fig5_scanned_edge.png).
Run render_figures.py first.

Usage
-----
python abstract/scripts/preview_figure.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG = Path(__file__).resolve().parents[2] / "abstract" / "figures"
OUT = FIG / "preview.png"
INK, INK2, CAP, ACCENT = "#0b0b0b", "#52514e", "#9b1b1f", "#eb6834"


def crop(path: Path, x0: float, x1: float, top: float = 0.08) -> np.ndarray:
    """Part of a figure between the fractions x0 and x1 of its width, white margins removed."""
    img = plt.imread(path)[..., :3]
    h, w = img.shape[:2]
    img = img[int(top * h):, int(x0 * w):int(x1 * w)]
    ys, xs = np.where((img < 0.97).any(axis=2))
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    closed = crop(FIG / "fig1_pipeline.png", 0.66, 1.0, top=0.12)
    overlay = crop(FIG / "fig5_scanned_edge.png", 0.0, 0.43, top=0.16)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.0, 4.0))  # 3:2 required by ECHO
    for a, img in ((ax, closed), (bx, overlay)):
        a.imshow(img)
        a.set_axis_off()
    fig.text(0.26, 0.76, "Closed: 130/130 watertight", fontsize=11.5, color=CAP,
             fontweight="bold", ha="center")
    fig.text(0.74, 0.76, "…yet +15 % after dissection", fontsize=11.5, color=ACCENT,
             fontweight="bold", ha="center")
    # No caption or small text: ISMRM preview figure rule
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.0, wspace=0.06)
    fig.savefig(OUT, dpi=300, facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
