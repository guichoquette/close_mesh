#!/usr/bin/env python3
"""
Transfer vertex colors from a source mesh to a target mesh without colors.

Used to copy the colors (e.g. sulcal/gyral texture) of an original brain mesh
to the mesh closed by `close_planar_hole.py`. Each target vertex takes the
color of the nearest source vertex (KD-tree); a texture on the source mesh is
first baked into vertex colors. Cap vertices thus take the color of the
nearest point of the original surface.

Usage
-----
    python utils/copy_colors.py [-s SOURCE] [-t TARGET] [-o OUTPUT]

Author: Guillaume Choquette, Université de Sherbrooke (SCIL), 2025.
"""

import argparse

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def source_vertex_colors(source: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """
    Vertex positions and RGBA colors of the source mesh.

    Parameters
    ----------
    source : trimesh.Trimesh
        Mesh loaded with ``process=False`` (to keep its visual information).

    Returns
    -------
    points : np.ndarray, shape (N, 3)
        Vertex positions.
    colors : np.ndarray, shape (N, 4)
        RGBA color of each vertex; a texture is baked into vertex colors.
    """
    if isinstance(source.visual, trimesh.visual.texture.TextureVisuals):
        source = source.copy()
        source.visual = source.visual.to_color()
    return source.vertices, source.visual.vertex_colors


def transfer_colors(source_points: np.ndarray, source_colors: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """
    Color of the nearest source vertex for each target vertex.

    Parameters
    ----------
    source_points : np.ndarray, shape (N, 3)
        Source vertex positions.
    source_colors : np.ndarray, shape (N, 4)
        Source RGBA colors.
    target_points : np.ndarray, shape (M, 3)
        Target vertex positions.

    Returns
    -------
    colors : np.ndarray, shape (M, 4)
        RGB of the nearest source vertex, fully opaque.
    """
    _, nearest = cKDTree(source_points).query(target_points, k=1)
    rgb = source_colors[nearest, :3].astype(np.uint8)
    return np.hstack([rgb, np.full((len(rgb), 1), 255, dtype=np.uint8)])


def main() -> None:
    """Copy the colors of the source mesh to the target mesh and export the result."""
    parser = argparse.ArgumentParser(description="Transfer the vertex colors of a mesh to another mesh.")
    parser.add_argument("-s", "--source", default="./brain_data/sub-02_epo-01_ref-01_mesh.ply",
                        help="Mesh with colors or texture.")
    parser.add_argument("-t", "--target", default="./brain_data/closed_solid/output.ply",
                        help="Mesh without colors.")
    parser.add_argument("-o", "--output", default="./brain_data/closed_solid/output_scalar_fields.ply",
                        help="Output file.")
    args = parser.parse_args()

    source_points, source_colors = source_vertex_colors(trimesh.load(args.source, process=False))
    target = trimesh.load(args.target, process=False)
    target.visual.vertex_colors = transfer_colors(source_points, source_colors, target.vertices)
    target.export(args.output)
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
