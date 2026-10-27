"""
Closing uncertainty of every epoch: volume added by the caps against a flat fan.

For each closed epoch of ``brain_data/output/results.csv``, the wedge of volume
between its caps and a flat fan on the same loops (``cap_wedges`` of
``diagnose_volume_increases.py``) is written, in % of the closed volume, to
``abstract/diagnostics/closing_uncertainty.csv``. Panel (c) of Figure 3 reads it.

Usage
-----
python abstract/scripts/closing_uncertainty.py [--workers 4]
"""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyvista as pv

from diagnose_volume_increases import OUT, cap_wedges, read_results

FIELDS = ["specimen", "epoch", "volume_cm3", "cap_area_cm2", "wedge_cm3", "wedge_%"]


def measure(row: dict) -> dict:
    inp, closed = pv.read(row["input"]), pv.read(row["output"])
    area, wedge = cap_wedges(closed, np.asarray(inp.points))
    volume = float(row["volume"])
    return {"specimen": row["specimen"], "epoch": row["epoch"], "volume_cm3": f"{volume * 1e6:.2f}",
            "cap_area_cm2": f"{area * 1e4:.2f}", "wedge_cm3": f"{wedge * 1e6:.3f}",
            "wedge_%": f"{wedge / volume * 100:.3f}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    rows = [r for epochs in read_results().values() for r in epochs.values() if r["volume"]]
    with ProcessPoolExecutor(args.workers) as pool:
        out = list(pool.map(measure, rows))
    out.sort(key=lambda r: (r["specimen"], r["epoch"]))
    with open(OUT / "closing_uncertainty.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out)
    w = np.abs([float(r["wedge_%"]) for r in out])
    print(f"{len(out)} epochs: mean |wedge| {w.mean():.2f} %, max {w.max():.2f} %")


if __name__ == "__main__":
    main()
