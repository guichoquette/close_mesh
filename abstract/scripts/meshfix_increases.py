"""Compare the volume increases found with our closing and with MeshFix.

Reads ``diagnostics/meshfix_volumes.csv`` (``compare_meshfix.py``) and prints
the difference between the two closings (mean and largest absolute value),
then the steps between consecutive epochs ("b" epochs left out) where the
volume increases with each closing. The steps are written to
``diagnostics/meshfix_increases.csv``.

Usage: python abstract/scripts/meshfix_increases.py
"""

import csv
from pathlib import Path

import numpy as np

DIAG = Path(__file__).resolve().parents[1] / "diagnostics"


def main():
    rows = list(csv.DictReader((DIAG / "meshfix_volumes.csv").open()))
    diff = np.array([float(r["difference_%"]) for r in rows])
    print(f"epochs={len(rows)}  watertight={sum(r['watertight'] == 'True' for r in rows)}  "
          f"mean |diff|={np.abs(diff).mean():.2f}%  max |diff|={np.abs(diff).max():.2f}%  "
          f"mean diff={diff.mean():+.2f}%")

    by_spc = {}
    for r in rows:
        if not r["epoch"].endswith("b"):
            by_spc.setdefault(r["specimen"], []).append(
                (int(r["epoch"]), float(r["volume_ours_cm3"]), float(r["volume_meshfix_cm3"])))
    steps = []
    for spc, pts in sorted(by_spc.items()):
        pts.sort()
        for (e0, o0, m0), (e1, o1, m1) in zip(pts, pts[1:]):
            steps.append({"specimen": spc, "epoch_from": f"{e0:02d}", "epoch_to": f"{e1:02d}",
                          "change_ours_%": f"{(o1 - o0) / o0 * 100:.2f}",
                          "change_meshfix_%": f"{(m1 - m0) / m0 * 100:.2f}"})
    with (DIAG / "meshfix_increases.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, list(steps[0]))
        writer.writeheader()
        writer.writerows(steps)

    ours = {(s["specimen"], s["epoch_from"]) for s in steps if float(s["change_ours_%"]) > 0}
    mesh = {(s["specimen"], s["epoch_from"]) for s in steps if float(s["change_meshfix_%"]) > 0}
    print(f"steps={len(steps)}  increases: ours={len(ours)}  MeshFix={len(mesh)}  both={len(ours & mesh)}")
    for s in steps:
        key = (s["specimen"], s["epoch_from"])
        if key in ours | mesh:
            tag = "both" if key in ours & mesh else ("ours only" if key in ours else "MeshFix only")
            print(f"  {s['specimen']} {s['epoch_from']}->{s['epoch_to']}  ours {float(s['change_ours_%']):+6.1f}%  "
                  f"MeshFix {float(s['change_meshfix_%']):+6.1f}%  {tag}")


if __name__ == "__main__":
    main()
