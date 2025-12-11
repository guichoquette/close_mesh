"""
compute_volume.py
-----------------
Utility functions to compute the **signed volume** of a closed triangular mesh.

This module implements a classical divergence-theorem–based approach:
for each triangular face (v0, v1, v2), the quantity
    v0 · (v1 × v2)
contributes to the oriented volume. Summing all contributions and dividing
by 6 yields the total signed volume of the mesh.

Signed vs absolute volume
-------------------------
- The **sign** indicates mesh orientation:
    * positive  → outward-facing normals
    * negative  → inward-facing normals
- The **absolute** value corresponds to the geometric enclosed volume.

Functions
---------
compute_signed_volume(vertices, faces)
    Computes signed volume from numpy arrays of vertices and triangular faces.

compute_signed_volume_ply(path)
    Loads a PLY file using PyVista, triangulates it if needed, extracts raw
    vertices and faces, and returns the signed volume.

Example
-------
Running this file as a script prints the signed and absolute volume of:
    ./brain_data/closed_solid/output.ply

Notes
-----
- The input mesh *must* represent a watertight, consistently oriented surface
  for the signed volume to be meaningful.
- PyVista face arrays are stored as: [3, i, j, k, 3, i, j, k, ...].
  This module reformats them into an (M × 3) integer array.

Author: Guillaume Choquette (Stage SCIL 2025)
"""

import numpy as np
import pyvista as pv

def compute_signed_volume(vertices, faces):
    """
    Compute the **signed volume** of a closed triangular mesh using a
    divergence-theorem formulation.

    Parameters
    ----------
    vertices : (N, 3) ndarray of float
        Array of 3D vertex positions.

    faces : (M, 3) ndarray of int
        Array of triangular faces, each entry being indices (i, j, k)
        into the vertex array.

    Returns
    -------
    float
        Signed volume of the mesh. Orientation determines the sign:
        positive = outward normals, negative = inward normals.
    """

    V = 0.0

    # Iterate over all triangles
    for f in faces:
        i, j, k = f
        v0 = vertices[i]
        v1 = vertices[j]
        v2 = vertices[k]

        # Contribution of the triangle to the signed volume:
        V += np.dot(v0, np.cross(v1, v2))   # v0 · (v1 × v2)

    return V / 6.0

def compute_signed_volume_ply(path):
    """
    Load a surface mesh from a PLY file and compute its signed volume.

    The mesh is triangulated if necessary before volume evaluation.

    Parameters
    ----------
    path : str or Path
        Path to the PLY file.

    Returns
    -------
    float
        Signed volume of the loaded mesh.
    """

    mesh = pv.read(path)

    # Ensure the mesh is fully triangulated.
    mesh = mesh.triangulate()

    # Extract vertex coordinates
    vertices = mesh.points

    # Extract triangular faces from PyVista's compact format:
    # PyVista stores faces as: [3, i, j, k, 3, i, j, k, ...]
    raw = mesh.faces.reshape(-1, 4)
    faces = raw[:, 1:4]

    return compute_signed_volume(vertices, faces)

# -------------------------------------------------------------------------
# Script execution (example)
# -------------------------------------------------------------------------
vol = compute_signed_volume_ply("./brain_data/closed_solid/output.ply")
print("Signed volume:", vol)
print("Absolute volume:", abs(vol))
