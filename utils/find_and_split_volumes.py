#!/usr/bin/env python3
"""
Count the disconnected parts of a mesh and export each one to its own file.

A debugging tool: it showed that some BraDiPho dissection meshes contain
several disconnected shells (floating islands). It can also isolate the
individual closed shells of a processed mesh.

Each part is written as ``mesh_body_XXX.ply`` in the output folder, and its
number of points, number of cells and volume are printed.

Usage
-----
    python utils/find_and_split_volumes.py -i INPUT.ply -o OUTPUT_DIR

Author: Guillaume Choquette, Université de Sherbrooke (SCIL), 2025.
"""

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv


def split_volumes(mesh: pv.PolyData, output_dir: str | Path) -> int:
    """
    Export each connected part of a mesh to ``output_dir/mesh_body_XXX.ply``.

    Parameters
    ----------
    mesh : pv.PolyData
        Mesh to split.
    output_dir : str or Path
        Output folder, created if needed.

    Returns
    -------
    n_parts : int
        Number of connected parts.
    """
    n_regions = len(np.unique(mesh.connectivity()["RegionId"]))
    print("Number of disconnected parts:", n_regions)

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, body in enumerate(mesh.split_bodies()):
        # split_bodies returns unstructured grids: back to a surface, whose volume is defined
        surface = body.extract_surface(algorithm=None)
        print(f"Body {i}: n_points={surface.n_points}, n_cells={surface.n_cells}, volume={surface.volume}")
        out_path = output_dir / f"mesh_body_{i:03d}.ply"
        surface.save(out_path)
        print("Written:", out_path)
    return n_regions


def main() -> None:
    """Split the mesh given on the command line into its connected parts."""
    parser = argparse.ArgumentParser(description="Count and export the disconnected parts of a mesh.")
    parser.add_argument("-i", "--input", required=True, help="Input mesh (PLY).")
    parser.add_argument("-o", "--output", required=True, help="Output folder.")
    args = parser.parse_args()
    split_volumes(pv.read(args.input), args.output)


if __name__ == "__main__":
    main()
