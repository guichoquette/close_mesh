#!/usr/bin/env python3
"""
Compute the signed volume of a closed triangle mesh.

For each triangle (v0, v1, v2), the divergence theorem gives the contribution
v0 · (v1 × v2) / 6 to the volume enclosed by the surface. The sign tells the
orientation: positive when the normals point outward, negative when they
point inward; the absolute value is the enclosed volume. The result is only
meaningful for a watertight, consistently oriented surface.

Usage
-----
    python utils/compute_volume.py [-i brain_data/closed_solid/output.ply]

Author: Guillaume Choquette, Université de Sherbrooke (SCIL), 2025.
"""

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv


def compute_signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """
    Signed volume of a closed triangle mesh.

    Parameters
    ----------
    vertices : np.ndarray, shape (N, 3)
        Vertex positions.
    faces : np.ndarray, shape (M, 3)
        Vertex indices of each triangle.

    Returns
    -------
    volume : float
        Signed volume: positive for outward normals, negative for inward normals.
    """
    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(faces)]
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def compute_signed_volume_ply(path: str | Path) -> float:
    """
    Load a mesh file (triangulated if needed) and compute its signed volume.

    Parameters
    ----------
    path : str or Path
        Mesh file (PLY, VTK, OBJ, STL, ...).

    Returns
    -------
    volume : float
        Signed volume of the mesh.
    """
    mesh = pv.read(path).triangulate()
    return compute_signed_volume(mesh.points, mesh.faces.reshape(-1, 4)[:, 1:])


def main() -> None:
    """Print the signed and absolute volume of the mesh given on the command line."""
    parser = argparse.ArgumentParser(description="Compute the signed volume of a closed triangle mesh.")
    parser.add_argument("-i", "--input", default="./brain_data/closed_solid/output.ply",
                        help="Closed mesh (default: ./brain_data/closed_solid/output.ply).")
    args = parser.parse_args()

    volume = compute_signed_volume_ply(args.input)
    print("Signed volume:", volume)
    print("Absolute volume:", abs(volume))


if __name__ == "__main__":
    main()
