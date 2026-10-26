#!/usr/bin/env python3
"""
Compare options A and B (see `close_planar_hole_option_a.py` and `close_planar_hole_option_b.py`).

Each epoch is closed with both options. For each one, the closed mesh is
exported with its caps in red (for a visual check in CloudCompare) to
``<out>/option_a/`` and ``<out>/option_b/``, and a table is printed and saved
to ``<out>/comparison.csv``: watertightness, volume against the current
script, number of curved caps, and inverted cap faces (faces whose normal
points into the closed mesh).

Usage
-----
    python experiments/compare_options.py [--out brain_data/verification/options] [EPOCH ...]

Without epochs, the epochs that still had a defect at the last visual check are used.

Author: Guillaume Choquette, Université de Sherbrooke (SCIL), 2025.
"""

import argparse
import csv
import importlib.util
import io
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
OPTIONS = {
    "current": ROOT / "close_opening_scripts" / "close_planar_hole.py",
    "option_a": ROOT / "experiments" / "close_planar_hole_option_a.py",
    "option_b": ROOT / "experiments" / "close_planar_hole_option_b.py",
}
DEFAULT_EPOCHS = [
    "spc-10_hem-L_epo-07", "spc-12_hem-R_epo-08b", "spc-12_hem-R_epo-10", "spc-15_hem-L_epo-06",
    "spc-15_hem-L_epo-08", "spc-07_hem-L_epo-02", "spc-07_hem-L_epo-03", "spc-07_hem-L_epo-06",
    "spc-07_hem-L_epo-07", "spc-07_hem-L_epo-07b", "spc-07_hem-L_epo-09", "spc-07_hem-L_epo-10b",
]


def load_module(name: str):
    """Import one version of the closing script."""
    spec = importlib.util.spec_from_file_location(name, OPTIONS[name])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inverted_cap_faces(mesh: pv.PolyData, cap_vertex: np.ndarray) -> tuple[int, float]:
    """
    Cap faces whose normal points into the closed mesh.

    A point slightly in front of the face (along its normal) inside the mesh,
    and one slightly behind it outside, means the face is wound inward.

    Returns
    -------
    count : int
        Number of inverted cap faces.
    area : float
        Their area, in mm² (mesh in metres).
    """
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    tri = mesh.points[faces[cap_vertex[faces].all(axis=1)]].astype(float)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(normal, axis=1) / 2
    keep = area > 1e-14
    tri, normal, area = tri[keep], normal[keep], area[keep]
    unit = normal / np.linalg.norm(normal, axis=1, keepdims=True)
    centre = tri.mean(axis=1)
    probes = pv.PolyData(np.vstack([centre + 2e-5 * unit, centre - 2e-5 * unit]))
    inside = np.asarray(probes.select_interior_points(mesh, check_surface=False).point_data["selected_points"])
    inside = inside.astype(bool)
    inverted = inside[:len(centre)] & ~inside[len(centre):]
    return int(inverted.sum()), float(area[inverted].sum() * 1e6)


def close_epoch(job: tuple[str, str, str]) -> dict:
    """Close one epoch with one option, export it with red caps and measure it."""
    option, path, out_dir = job
    module = load_module(option)
    args = argparse.Namespace(input=path, output="", capsOnly=False, debug=None, debug_loop="",
                              seedStepFactor=4.0, seedDecimator=4.0, meshFilter=1e-5)
    log = io.StringIO()
    with redirect_stdout(log):
        closed = module.create_caps(args).clean(point_merging=False, polys_to_lines=False)
    name = Path(path).stem.replace("_dissection", "")
    row = {"epoch": name, "option": option, "curved_caps": log.getvalue().count("closed with a curved cap")}
    edges = closed.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False,
                                         non_manifold_edges=True).n_cells
    row["watertight"] = edges == 0
    row["volume"] = closed.volume
    if option == "current":
        return row

    raw = pv.read(path).triangulate().clean()
    cap_vertex = cKDTree(np.asarray(raw.points, float)).query(np.asarray(closed.points, float))[0] > 1e-12
    row["inverted_faces"], row["inverted_mm2"] = inverted_cap_faces(closed, cap_vertex)
    colors = np.tile(np.array([200, 200, 200, 255], dtype=np.uint8), (closed.n_points, 1))
    colors[cap_vertex] = [220, 30, 30, 255]
    target = Path(out_dir) / option
    target.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=np.asarray(closed.points), faces=closed.faces.reshape(-1, 4)[:, 1:],
                    vertex_colors=colors, process=False).export(target / f"{name}_closed_caps_red.ply")
    row["guard"] = "; ".join(line for line in log.getvalue().splitlines() if "OPTION B" in line or "would add" in line)
    return row


def main() -> None:
    """Run both options on the epochs, export them and print the comparison."""
    parser = argparse.ArgumentParser(description="Compare options A and B on problem epochs.")
    parser.add_argument("epochs", nargs="*", default=DEFAULT_EPOCHS, help="Epochs, e.g. spc-10_hem-L_epo-07.")
    parser.add_argument("--out", default=str(ROOT / "brain_data" / "verification" / "options"), help="Output folder.")
    parser.add_argument("--workers", type=int, default=4, help="Closings run in parallel (default: 4).")
    args = parser.parse_args()

    paths = {}
    for epoch in args.epochs:
        found = sorted(ROOT.glob(f"brain_data/*_epo-all_dissection/bradipho_data/*/dissection/{epoch}_dissection.ply"))
        if not found:
            raise SystemExit(f"Epoch not found: {epoch}")
        paths[epoch] = str(found[0])
    jobs = [(option, path, args.out) for path in paths.values() for option in OPTIONS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(close_epoch, jobs))

    by_key = {(r["epoch"], r["option"]): r for r in rows}
    table = []
    for epoch in args.epochs:
        current = by_key[(epoch, "current")]
        line = {"epoch": epoch, "current_curved_caps": current["curved_caps"]}
        for option in ("option_a", "option_b"):
            r = by_key[(epoch, option)]
            line[f"{option}_watertight"] = r["watertight"]
            line[f"{option}_volume_change_%"] = round((r["volume"] - current["volume"]) / current["volume"] * 100, 2)
            line[f"{option}_curved_caps"] = r["curved_caps"]
            line[f"{option}_inverted_faces"] = r["inverted_faces"]
            line[f"{option}_inverted_mm2"] = round(r["inverted_mm2"], 2)
        line["option_b_guard"] = by_key[(epoch, "option_b")]["guard"]
        table.append(line)
        summary = [f"{epoch:22s}"]
        for option, label in (("option_a", "A"), ("option_b", "B")):
            summary.append(f"{label}: {line[f'{option}_volume_change_%']:+6.2f} % vol, "
                           f"{line[f'{option}_curved_caps']} curved, {line[f'{option}_inverted_faces']:3d} inverted "
                           f"({line[f'{option}_inverted_mm2']:.2f} mm2)")
        summary.append(f"current: {current['curved_caps']} curved")
        print(" | ".join(summary), flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"Exports in {out / 'option_a'} and {out / 'option_b'}; table in {out / 'comparison.csv'}")


if __name__ == "__main__":
    main()
