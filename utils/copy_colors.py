"""
copy_colors.py
==============

Transfer vertex colors from a *source* mesh to a *target* mesh that does not
contain any color information. The mapping is done using a nearest-neighbor
search in 3D space (KD-Tree).

This script is used notably to transfer sulcal/gyral color information from an
original brain mesh to a closed/processed mesh.

Workflow
--------
1. Parse command-line arguments (source mesh, target mesh, output path)
2. Load the source mesh (trimesh), preserving visual information
3. Ensure the source has per-vertex colors:
    - If using textures → convert (bake) them to vertex colors
4. Load the target mesh (without color)
5. Build a KD-Tree on the source vertex positions
6. For each target vertex, find the nearest source vertex
7. Copy its RGBA color to the target
8. Export the colorized mesh

Author
------
Guillaume Choquette
Université de Sherbrooke — SCIL Lab
"""

import trimesh
import trimesh.visual

import argparse
import numpy as np
from scipy.spatial import cKDTree
import trimesh


# =============================================================
# Build the parser
# -------------------------------------------------------------
p = argparse.ArgumentParser(description="Transfer the color from the input file to the outputfile.")
p.add_argument("-s", "--source", help="Input mesh.", default="./brain_data/sub-02_epo-01_ref-01_mesh.ply")
p.add_argument("-t", "--target", help="Input mesh without colors.", default="./brain_data/closed_solid/output.ply")
p.add_argument("-o", "--output", help="Output file.", default="./brain_data/closed_solid/output_scalar_fields.ply")
args = p.parse_args()


# IMPORTANT : mets process=False pour ne pas perdre les infos visuelles
src = trimesh.load(args.source, process=False)

# Si on a une texture (material), trimesh utilise TextureVisuals
if isinstance(src.visual, trimesh.visual.texture.TextureVisuals):
    # "Bake" la texture en couleurs par sommet
    src_colored = src.copy()
    src_colored.visual = src.visual.to_color()  # convertit en VertexColorVisuals
else:
    src_colored = src  # il y a déjà des vertex colors

src_pts = src_colored.vertices
src_colors = src_colored.visual.vertex_colors  # (N, 4) en RGBA

# Mesh de sortie (celui que tu produis avec ton code de fermeture)
dst = trimesh.load(args.target, process=False)
dst_pts = dst.vertices

# KD-tree sur le mesh source
tree = cKDTree(src_pts)
dist, idx = tree.query(dst_pts, k=1)

# On récupère les couleurs des points les plus proches
rgb = src_colors[idx, :3]
alpha = 255 * np.ones((rgb.shape[0], 1), dtype=np.uint8)
dst_colors = np.hstack([rgb.astype(np.uint8), alpha])

dst.visual.vertex_colors = dst_colors

dst.export(args.output)
print(f"Output writen to : {args.output}")
