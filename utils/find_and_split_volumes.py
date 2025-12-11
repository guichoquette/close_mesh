"""
find_and_split_volumes.py
=========================

Analyze a mesh to:
  1) Count how many disconnected volumes (connected components) it contains.
  2) Optionally export each connected volume to a separate PLY file.

I personaly used this script as a debugging tool to verify and visualise
what was the problem with my brain meshes of epocs with disections (in input).
This is how I discovered that some of them contained multiple disconnected shells.

Note:
This could be useful, in the future for detecting stray components or to isolate
individual closed shells in a brain surface after processing.

Workflow
--------
1. Parse command-line arguments (input mesh, output directory).
2. Load the input mesh with PyVista.
3. Run a connectivity analysis to obtain a RegionId for each component.
4. Count and print the number of disjoint regions.
5. Use `split_bodies()` to separate each connected volume as its own mesh.
6. For each body:
    - Print basic information (points, cells, volume).
    - Save it as `mesh_body_XXX.ply` into the output directory.

Author
------
Guillaume Choquette
Université de Sherbrooke — SCIL Lab
"""

import argparse
import pyvista as pv
import numpy as np
from pathlib import Path
import pyvista as pv

# ======================================================================
# Argument parsing
# ======================================================================
p = argparse.ArgumentParser(description="Compute the number of disjoint volumes.")
p.add_argument("-i", "--input", required=True, help="Input mesh (PLY).")
p.add_argument("-o", "--output", required=True, help="Output directory.")
args = p.parse_args()

# Read the input mesh (typically a surface or a closed volume).
mesh = pv.read(args.input)

# 1) Connectivity analysis: each region receives a RegionId
connected = mesh.connectivity()  # vtkConnectivityFilter derrière

# List of distinct RegionIds present in the mesh
region_ids = np.unique(connected["RegionId"])
n_regions = len(region_ids)

print("Nombre de volumes disjoints :", n_regions)

# 2) Split the mesh into separate bodies (one mesh per connected volume)
bodies = mesh.split_bodies()
print("Nombre de volumes :", len(bodies))

# 3) Export each body as an individual PLY file
for i, body in enumerate(bodies):
    print(f"Body {i}: n_points={body.n_points}, n_cells={body.n_cells}, volume={body.volume}")

    # Construct output path: mesh_body_000.ply, mesh_body_001.ply, etc.
    out_path = Path(args.output).resolve() / f"mesh_body_{i:03d}.ply"
    print("Écrit :", out_path)

    # Make sure the output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the mesh using meshio backend (supports PLY, etc.)
    pv.save_meshio(out_path, body)

