"""Close every BraDiPho epoch with MeshFix, as an independent baseline.

MeshFix (Attene 2010, through PyMeshFix) repairs the raw input mesh and fills
its holes; it keeps only the largest component. Its volumes are written to
``abstract/diagnostics/meshfix_volumes.csv`` next to the volumes of our
pipeline (``brain_data/output/results.csv``), so that the volume increases
between consecutive epochs can be checked with both closing methods.

The CSV is written after every epoch, and epochs already in it are skipped,
so an interrupted run can simply be started again.

Usage: python abstract/scripts/compare_meshfix.py [--workers 4]
"""

import argparse
import csv
import glob
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OURS = ROOT / "brain_data" / "output" / "results.csv"
OUT = ROOT / "abstract" / "diagnostics" / "meshfix_volumes.csv"
FIELDS = ["specimen", "epoch", "volume_ours_cm3", "volume_meshfix_cm3",
          "difference_%", "watertight", "seconds"]


def close_with_meshfix(path):
    """Return (volume in cm3, watertight, seconds) of an input mesh closed by MeshFix."""
    import pymeshfix
    import pyvista as pv

    start = time.time()
    mesh = pv.read(path).triangulate().clean()
    fix = pymeshfix.MeshFix(mesh)
    fix.repair()
    closed = fix.mesh
    watertight = closed.n_open_edges == 0 and closed.is_manifold
    return abs(closed.volume) * 1e6, watertight, time.time() - start


def input_path(specimen, epoch):
    """Return the BraDiPho input mesh of an epoch."""
    pattern = (f"brain_data/{specimen}_hem-*_epo-all_dissection/bradipho_data/"
               f"{specimen}/dissection/{specimen}_hem-*_epo-{epoch}_dissection.ply")
    return glob.glob(str(ROOT / pattern))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    ours = {(r["specimen"], r["epoch"]): float(r["volume"]) * 1e6
            for r in csv.DictReader(OURS.open()) if r["volume"]}
    done = set()
    if OUT.exists():
        done = {(r["specimen"], r["epoch"]) for r in csv.DictReader(OUT.open())}
    else:
        with OUT.open("w", newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()

    todo = [key for key in sorted(ours) if key not in done]
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(close_with_meshfix, input_path(*key)): key for key in todo}
        for job in as_completed(jobs):
            specimen, epoch = jobs[job]
            volume, watertight, seconds = job.result()
            mine = ours[(specimen, epoch)]
            row = {"specimen": specimen, "epoch": epoch,
                   "volume_ours_cm3": f"{mine:.2f}",
                   "volume_meshfix_cm3": f"{volume:.2f}",
                   "difference_%": f"{(volume - mine) / mine * 100:.2f}",
                   "watertight": watertight, "seconds": f"{seconds:.0f}"}
            with OUT.open("a", newline="") as f:
                csv.DictWriter(f, FIELDS).writerow(row)
            print(specimen, epoch, row["difference_%"], "%", flush=True)


if __name__ == "__main__":
    main()
