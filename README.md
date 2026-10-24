# 🧠 Brain Surface Hole Closing – Internship Project

Automatically close an open mesh with a plane and headband. Also does a color transfer for brain meshes.

This project provides a full workflow to:

1. **Automatically detect and close the openings of a 3D surface mesh**, with a planar cap when the opening allows it and a curved cap otherwise,
2. **Repair the defects of scanned meshes** that prevent a clean closing (duplicated triangles, flaps, non-manifold edges, inconsistent orientation),
3. **Copy the vertex colors** (e.g., a sulcal/gyral texture) from the original open mesh to the closed mesh.
4. **Calculate the volume of closed meshes**, after making sure the new mesh is watertight. The script prints it; `utils/compute_volume.py` computes it for any closed mesh.

---

## 📁 Project Structure

```text
close_mesh/
│
├── brain_data/
│   ├── sub-02_epo-01_ref-01_mesh.ply        # Original mesh with openings
│   ├── sub-02_epo-01_ref-01_mesh.jpg        # Original color map
│   ├── closed_solid/
│   │   ├── output.ply                       # Closed mesh produced by script
│   │   └── output_colored.ply               # Mesh with copied scalars
│
├── close_opening_scripts/
│   └── close_planar_hole.py                 # Main hole-closing algorithm
│
├── tests/
│   └── test_close_planar_hole.py            # Unit tests of the closing pipeline
│
├── utils/
│   ├── compute_volume.py                    # Script to compute the volume of a closed mesh
│   ├── copy_colors.py                       # Scalar-field transfer tool
│   └── find_and_split_volumes.py            # Debug tool to inspect individual disconnected surfaces
│
├── batch_close_all.py                       # Closes every epoch of every specimen + results.csv
├── batch-extraction.sh                      # Older per-epoch script for spc-02, kept for debugging
├── requirements.txt
├── ruff.toml                                # Code style rules (ruff check .)
└── README.md
```

---

# 🧠 Brain Data for the Project

You can download the meshes used during development from the official BraDiPho dataset:

🔗 **BraDiPho – Brain Digital Fossils Archive**
https://bradipho.eu/6-download.html

All specimens and epochs are publicly available and easy to download. Only the `.ply` files are needed (the `.bin` files are CloudCompare projects).

The folder `brain_data` also contains the **first epoch** of a specimen along with its **texture file**.
These two files are confirmed to work well with the full pipeline and are ideal for a **quick test run** of the scripts.

Downloaded specimens (`brain_data/*_epo-all_dissection/`) are ignored by git.

# 📦 Installation

Create a virtual environment:

python3 -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Main libraries:

- pyvista / vtk
- numpy
- scipy (sparse solver, Qhull Delaunay, connected components)
- shapely (polygon validity and inward offset)
- triangle (constrained Delaunay triangulation of the planar cap)
- trimesh
- matplotlib

---

# 🏃 Quick Test Run

This part explains how to run the full hole-closing and scalar-transfer pipeline starting from a clean environment.

```bash
# Create environment
python3.12 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Close holes
python close_opening_scripts/close_planar_hole.py \
    -i ../brain_data/sub-02_epo-01_ref-01_mesh.ply \
    -o ../brain_data/closed_solid/output.ply \
    --meshFilter 1e-5 \
    --debug ../brain_data/closed_solid/

# Transfer scalar fields
python utils/copy_colors.py
```

You can now view the closed mesh in Cloudcompare or MeshLab. You can find the closed mesh in brain_data/closed_solid. The interesting files for this quick test run are: `output.ply` and `output_scalar_fields.ply`.

If you don't have Cloudcompare or MeshLab, I made a quick viewer for you. All you need to do is type in the terminal:

```bash
./view_ply.sh
```

NOTE: You will be able to see the closed mesh followed by the closed mesh with his scalars fields once you closed the first opened window.

# 🔧 Cap Closing Algorithm (`close_planar_hole.py`)

This script **automatically closes every open boundary loop** of a triangulated mesh and checks that the result is watertight.

### How it works

**1. Load and clean the mesh**

- Read and triangulate the mesh, merge duplicated points.
- Optionally collapse edges shorter than `--meshFilter` (recommended: `1e-5` for BraDiPho meshes).

**2. Repair the input mesh**

Scanned dissection meshes contain defects that make the boundary ambiguous. They are repaired before any cap is built:

- `remove_duplicate_faces` — removes triangles present twice (often two copies wound in opposite directions, a zero-thickness flap).
- `remove_small_flaps` — removes small patches (≤ 20 faces) that are only glued on the surface by a non-manifold edge.
- `split_non_manifold_edges` — where two sheets of surface meet along an edge shared by more than two faces, the faces are paired by angle around the edge and the vertices are duplicated so that each sheet becomes manifold. No geometry is moved or removed. Vertices on a boundary are not split, so that two openings touching at a vertex keep two separate caps.
- `make_orientation_consistent` — propagates a consistent winding across manifold edges only (VTK's own propagation also crosses non-manifold edges and can flip whole regions), keeping the majority winding of each patch.
- `orient_outward` — flips the whole mesh if its signed volume is negative.

**3. Extract the boundary loops**

`extract_boundary_loops` walks the boundary edge by edge. Where two openings share a vertex, the loop is cut into simple loops; a boundary chain left open by a non-manifold edge is closed between its two ends.

**4. Build one cap per loop**

- The loop is ordered **against the mesh winding** (`_orient_loop_against_mesh`), so that the cap is oriented consistently with the mesh by construction.
- A plane is fitted to the loop by **SVD**. If the loop projects onto it as a **simple polygon**, a **planar cap** is built:
  1. The plane is moved to the highest point of the loop, **on the outer side of the mesh** (given by the normals of the faces along the boundary). The sign of the SVD normal is arbitrary: without this, the plane could land inside the brain and the cap would cut through the mesh.
  2. The loop is projected onto the plane and a regular grid of seed points is kept inside an inward offset of the polygon.
  3. The polygon is triangulated with **Triangle** (constrained Delaunay): every loop edge is kept, triangles outside the polygon are removed, and no point is added on the loop.
  4. A **side band** connects the original 3D loop to its projection.
- Otherwise (the opening is too warped and its projection crosses itself), a **curved cap** is built (`make_curved_cap`) and a `WARNING` is printed:
  1. The loop is mapped onto a unit disk by arc length, seeds are placed inside the polygon, and the disk is triangulated (Qhull).
  2. Loop vertices keep their exact position; interior vertices are placed by **harmonic interpolation** (Laplace equation), which gives a smooth membrane spanning the loop.

**5. Assemble**

- Every cap starts with its loop vertices, in loop order. Caps are attached to the mesh **by vertex index** (`_attach_caps`), never by merging coincident points, which would join again the vertices duplicated in step 2 (a loop may even go through two coincident copies of a vertex).

**6. Final repair passes**

On the capped mesh, up to three passes fix what step 2 had to leave:

- `split_non_manifold_edges` again: the caps are attached, so the vertices that were on a boundary can now be split safely.
- `drop_extra_faces_on_non_manifold_edges`: some non-manifold edges cannot be split (the faces around them reconnect around both vertices: a small fold or pocket of the scan). Only the two faces that best continue the surface are kept.
- The tiny holes this opens are capped (`_make_repair_cap`): the regular cap, or a fan around the hole centroid (`make_fan_cap`) when the regular cap would link two loop vertices already linked in the mesh.

**7. Check**

- Normals are computed without letting VTK reorder faces.
- The result is **watertight** when it has no boundary edge and no non-manifold edge; only then is the volume computed and printed.

### Run

python close_opening_scripts/close_planar_hole.py \
 -i brain_data/sub-02_epo-01_ref-01_mesh.ply \
 -o brain_data/closed_solid/output.ply \
 --meshFilter 1e-5

The script prints `watertight=True  volume=...` or `watertight=False  volume=N/A`.

---

# 🗂️ Batch Processing of All Specimens (`batch_close_all.py`)

Closes every epoch of every specimen downloaded in `brain_data/`, and summarises the results in a CSV file.

### Input

The specimen folders as unzipped from BraDiPho:

```text
brain_data/<specimen>_hem-<L|R>_epo-all_dissection/bradipho_data/<specimen>/dissection/*_dissection.ply
```

### Run

```bash
python batch_close_all.py
```

Options:

- `--data` — folder containing the specimen folders (default: `brain_data`)
- `--out` — output folder (default: `brain_data/output`)
- `--workers` — number of epochs processed in parallel (default: 4; each worker needs about 2 GB of RAM)
- `--meshFilter` — short edge collapse tolerance (default: `1e-5`)
- `--overwrite` — process again the epochs already done

### Output

```text
brain_data/output/
├── results.csv                              # One line per epoch
├── spc-01/
│   ├── spc-01_hem-R_epo-01_closed.ply       # Closed mesh of each epoch
│   └── ...
└── ...
```

`results.csv` columns: `specimen`, `hemisphere`, `epoch`, `watertight`, `volume` (empty when not watertight), `curved_caps` (number of openings closed with a curved cap), `seconds`, `input`, `output`, `error`.

Each epoch goes through exactly the same code as the command line script. The CSV is written after every epoch, and epochs already in it are skipped, so an interrupted run can simply be started again. All 130 epochs take about 30 to 45 minutes with 4 workers. `brain_data/output/` is ignored by git.

`batch-extraction.sh` is the older per-epoch script for spc-02, kept for debugging. Its input paths (`brain_data/spc-02/dissection/`) follow an older folder layout and must be adapted before use.

---

# 📊 Results on the BraDiPho dissection meshes

All 130 epochs of the 12 dissected specimens available on BraDiPho were closed with `--meshFilter 1e-5` (results of 2026-10-24).

An epoch is counted as **watertight** when the script reports `watertight=True` and therefore gives a volume (no boundary edge and no non-manifold edge). The volume difference is measured against a reference where every hole is closed by a flat fan to its loop centroid.

| Specimen | Epochs | Watertight (volume given) | Epochs with a curved cap | Mean volume difference | Max volume difference |
|---|---|---|---|---|---|
| spc-01 | 11 | 11 | 1 | 1.6 % | 3.4 % |
| spc-02 | 10 | 10 | 0 | 0.9 % | 3.2 % |
| spc-04 | 10 | 10 | 3 | 0.1 % | 0.2 % |
| spc-06 | 10 | 10 | 3 | 0.2 % | 1.2 % |
| spc-07 | 12 | 12 | 8 | 0.2 % | 0.6 % |
| spc-10 | 12 | 12 | 4 | 0.3 % | 0.6 % |
| spc-12 | 12 | 12 | 7 | 0.3 % | 0.7 % |
| spc-15 | 14 | 14 | 4 | 0.6 % | 2.2 % |
| spc-16 | 12 | 12 | 2 | 0.1 % | 0.3 % |
| spc-17 | 7 | 7 | 3 | 0.2 % | 0.3 % |
| spc-18 | 8 | 8 | 0 | 0.1 % | 0.4 % |
| spc-19 | 12 | 12 | 3 | 0.1 % | 0.3 % |
| **Total** | **130** | **130** | **38** | **0.4 %** | **3.4 %** |

- Every epoch is watertight and gets a volume. One epoch (spc-17 epo-05) keeps 5 edges with an inconsistent orientation, which may slightly affect its volume.
- "Epochs with a curved cap" counts the epochs where at least one opening, often a tiny one, had no valid projection plane.
- The planar cap, placed on the outer side of the opening, adds a small wedge of volume: this is why the largest differences are positive.

---

# 🔍 Visual Check of the BraDiPho Results

Watertight does not mean correct: every closed epoch (`brain_data/output/`) was also inspected visually, in CloudCompare (results of 2026-10-24).

### Observations

| Specimen | Epochs | Epochs with a visible defect after closing | Floating blob far from the brain |
|---|---|---|---|
| spc-01 | 11 | — | — |
| spc-02 | 10 | — | — |
| spc-04 | 10 | 01, 09 | — |
| spc-06 | 10 | 07, 09b | — |
| spc-07 | 12 | 02, 03, 06, 07, 07b, 09, 09b, 10b | 09b |
| spc-10 | 12 | 01, 07, 10 | 07b |
| spc-12 | 12 | 06, 08b, 09b, 10, 10b | — |
| spc-15 | 14 | 01, 02, 06, 08, 11 | 02 |
| spc-16 | 12 | — | — |
| spc-17 | 7 | 02, 04, 08 | — |
| spc-18 | 8 | — | — |
| spc-19 | 12 | 05 | — |

All the epochs of spc-01, spc-02, spc-16 and spc-18 are closed without any visible problem. In some cases, removing the floating blob gives an epoch without any problem.

The defects of spc-06 epo-07, spc-15 epo-11 and spc-17 epo-04 are tiny spots displayed in blue by CloudCompare on the planar cap: the normal is inverted there.

### The "b" epochs

The meaning of the "b" epochs (e.g. `epo-07` and `epo-07b`) is not documented in the downloaded data. Visually, an epoch and its "b" variant are very similar, with small variations: they could be two versions of the same mesh. In some cases one could replace the other:

| Specimen | Candidate replacement |
|---|---|
| spc-10 | epo-07b could replace epo-07, and epo-10b could replace epo-10 |
| spc-12 | epo-09 could replace epo-09b |

Until their meaning is known, the "b" epochs should not be mixed with the main sequence when comparing volumes across epochs (some of them break the decrease of volume expected along a dissection).

### Causes found

| Cause | Epochs concerned | How it was found |
|---|---|---|
| **Curved cap on a large opening** | 25 of the 29 epochs with a defect: spc-04 epo-01, 09; spc-06 epo-09b; spc-07 epo-02, 03, 06, 07, 07b, 09, 10b; spc-10 epo-01, 07, 10; spc-12 epo-06, 08b, 09b, 10, 10b; spc-15 epo-01, 02, 06, 08; spc-17 epo-02, 08; spc-19 epo-05 | 22 of them have a curved cap of 700 to 6100 vertices, and spc-12 epo-10b one of 111 vertices; spc-15 epo-02 (47) and spc-17 epo-08 (4) only have small ones. Conversely, every epoch with a curved cap of more than 100 vertices has a defect, while the curved caps of the epochs without defect are tiny (4 to 6 vertices). |
| **Floating blob** (isolated piece of scan far from the brain) | Seen: spc-07 epo-09b (55 mm), spc-10 epo-07b (42 mm), spc-15 epo-02 (36 mm). Also found by the analysis: spc-10 epo-06 (27 mm), epo-10 (66 mm), spc-07 epo-10b (23 mm) | Connected parts of each closed mesh and their distance to the main part. Explains spc-07 epo-09b, which has no curved cap. |
| **Small open fragment closed inward** (blue spot) | spc-06 epo-07, spc-15 epo-11, part of spc-17 epo-04 | A small isolated piece of scan (57 to 130 faces) lying at the cut, with a hole. It is closed like the rest, but only the mesh as a whole is oriented outward, so this tiny closed piece can stay inward. |
| **Tissue flap at the cut, under the planar cap** (blue spot) | spc-17 epo-04 (the remaining spot, 50 faces) | A small hole of the main surface, 0.1 mm under the planar cap: the two surfaces nearly overlap and the back of one of them is seen. |
| **Inverted debris** | Up to 40 closed fragments per epoch (spc-15, spc-17) | Many small closed pieces of the scans are oriented inward. They are not visible from outside, but their volume is subtracted instead of added (about 0.04 % of the total). |

**Why so many curved caps:** a curved cap is used when the opening, projected on its plane, crosses itself. On most large openings, the crossing is a **single micro-fold of three consecutive vertices** (two segments one vertex apart), with no measurable area: one tiny defect of the boundary is enough to replace the whole planar cut by a curved membrane, which follows the warped boundary and looks wrong. Only a few openings are really folded over tens or hundreds of vertices (spc-07 epo-02, spc-17 epo-02).

### How they could be closed

1. **Untangle the small folds of the projected loop** (prototype tested). When the projected loop crosses itself over at most 20 vertices, these vertices are aligned on the chord joining both sides of the fold, in the plane, and the planar cap is kept; the curved cap is used only for real folds. On the 25 epochs above, the prototype keeps all of them watertight and, of the 23 epochs with a curved cap of more than 100 vertices, **removes every such cap from 11** (spc-04 epo-01, 09; spc-06 epo-09b; spc-07 epo-09; spc-10 epo-01, 10; spc-12 epo-06, 09b; spc-15 epo-01, 08; spc-19 epo-05). The volume changes by −0.1 % to +2.5 % (the planar cap adds a small wedge, as everywhere else). The 12 others keep one or two large curved caps where the fold is larger.
2. **Remove the floating blobs**: keep only the parts of the mesh close to the brain (e.g. within 10 mm of the largest part), before capping. Their volume is negligible, but they are part of the closed mesh.
3. **For the real folds** (spc-07, spc-12, spc-15, spc-17): a better curved cap (e.g. minimal surface or fairing instead of harmonic interpolation), or splitting the opening into parts that can each be projected.
4. **Remove the small open fragments** (prototype tested): before capping, drop the isolated parts that are open and have at most 1000 faces (scan debris). On spc-06 epo-07 and spc-15 epo-11, the blue spots disappear (no inverted face left in the cap plane), the epochs stay watertight and the volume changes by less than 0.05 %.
5. **Orient each closed part outward** (prototype tested): at the end, flip every connected part whose signed volume is clearly negative (not only the mesh as a whole). It fixes the inverted debris, and is safe for flat parts (whose volume is about zero, they are left as they are).
6. **For the tissue flaps under the cap** (spc-17 epo-04): detect the surface of the scan lying within a fraction of a millimetre of the planar cap, inside the opening, and remove it before capping.
7. **For the "b" epochs**: find their meaning in the BraDiPho documentation, or choose one version per epoch as suggested above.

---

# 🧰 Utilities (`utils/`)

### Scalar transfer (`copy_colors.py`)

Transfers the **vertex colors** of the open mesh (a texture is first baked into vertex colors) to the closed mesh, using **KD-Tree nearest-neighbor mapping**: cap vertices take the color of the nearest point of the original surface.

```bash
python utils/copy_colors.py [-s SOURCE] [-t TARGET] [-o OUTPUT]
```

Defaults: `brain_data/sub-02_epo-01_ref-01_mesh.ply` → `brain_data/closed_solid/output.ply` → `brain_data/closed_solid/output_scalar_fields.ply`.

### Volume (`compute_volume.py`)

Prints the signed and absolute volume of a closed mesh (divergence theorem, in double precision).

```bash
python utils/compute_volume.py [-i brain_data/closed_solid/output.ply]
```

### Disconnected parts (`find_and_split_volumes.py`)

Counts the disconnected parts of a mesh and exports each one as `mesh_body_XXX.ply`, with its number of points, cells and volume.

```bash
python utils/find_and_split_volumes.py -i INPUT.ply -o OUTPUT_DIR
```

---

# ▶️ Full Workflow

### 1. Close the holes

python close_opening_scripts/close_planar_hole.py \
 -i brain_data/sub-02_epo-01_ref-01_mesh.ply \
 -o brain_data/closed_solid/output.ply \
 --meshFilter 1e-5 \
 --debug brain_data/debug/

### 2. Transfer scalar fields

python utils/copy_colors.py

### 3. Visualize in CloudCompare, MeshLab, or PyVista

---

# 🧪 Debug Outputs

When using `--debug`, the script exports, for each loop (`loop1/`, `loop2/`, ...):

- the boundary loop
- the projected loop and the inward-offset polygon
- grid seeds and filtered seeds
- the planar triangulation, or the curved cap (`debug_curved_cap.ply`)

and, for the whole mesh, the filtered mesh (`--meshFilter`) and the remaining open edges.

These outputs help inspect and validate each internal step.

---

# 📘 Examples

### Example 1 — Close holes with filtering and debug output

python close_opening_scripts/close_planar_hole.py \
 -i brain_data/sub-02_epo-01_ref-01_mesh.ply \
 -o brain_data/closed_solid/output.ply \
 --meshFilter 1e-5 \
 --debug brain_data/debug/

### Example 2 — Generate caps only

python close_opening_scripts/close_planar_hole.py \
 -i brain_data/sub-02_epo-01_ref-01_mesh.ply \
 -o brain_data/closed_solid/caps_only.ply \
 --capsOnly

### Example 3 — Transfer curvature/thickness fields

python utils/copy_colors.py \
 -s brain_data/sub-02_epo-01_ref-01_mesh.ply \
 -t brain_data/closed_solid/output.ply \
 -o brain_data/closed_solid/output_scalar_fields.ply

---

# ⚠️ Limitations / Known Issues

### 🔹 1. Warped openings get a curved cap

The planar cap needs the opening to project onto its best-fit plane as a simple (non self-intersecting) polygon. When it does not, the opening is closed with a curved (harmonic) cap instead and a `WARNING` is printed. Such a cap is not flat, which slightly changes how the closed volume is defined at that opening.

The planar cap is placed on the outer side of the opening, at its highest point; it therefore adds a small wedge of volume (typically under 1 %, up to ~3 % on very warped openings).

### 🔹 2. Input defects are repaired locally

Small flaps and the extra faces of non-manifold edges that cannot be split are removed, and the holes this opens are capped. These are a few triangles per epoch, but the output is not strictly the input surface plus caps.

### 🔹 3. Duplicated vertices in the output

Non-manifold edges are repaired by duplicating vertices at the same position. The output must not be cleaned with point merging (e.g. `clean()` with default options), or the repaired sheets would be joined again.

### 🔹 4. Sensitive to boundary noise

Meshes with extremely short or irregular edges should be pre-filtered using `--meshFilter`.

### 🔹 5. Processing time

The repair passes are written in Python: a BraDiPho epoch takes from a few seconds to about 6 minutes (median under a minute).

### 🔹 6. Scalar transfer uses nearest-neighbor

This may smooth or blur sharp scalar variations.
Future improvement: weighted or barycentric interpolation.

### 🔹 7. Large meshes increase debug output size

The triangulation and debug exports scale with mesh size.

---

# 🚀 Future Work / Enhancements

### 🔧 Algorithmic Improvements

- **Visual validation of the curved caps**
  The results above are checked numerically; the curved caps (38 epochs) should also be inspected visually.

- **Faster repair passes**
  Vectorize the orientation propagation and the non-manifold repairs, which dominate the processing time.

- **Adaptive seeding strategies**
  Increase sampling density near the boundary, reduce it inside.

- **Weighted scalar transfer**
  Replace nearest-neighbor with inverse-distance or barycentric interpolation.

### 🎨 Mesh Quality Enhancements

- **Post-cap smoothing** (Taubin, Laplacian, or HC smoothing)
  Improves blending between cap and original surface.

- **Side-band optimization**
  Reduce long triangles by adding local remeshing.

---

# 🛠️ Development

### Tests

```bash
python -m unittest tests.test_close_planar_hole -v
```

The tests cover the side band winding, the input repairs, boundary loop extraction (pinched boundaries), loop orientation, planar, curved and fan caps for every combination of mesh winding and loop order, the side of the cap plane, and regressions found on BraDiPho meshes.

### Code style

The code follows PEP 8 and documents every module and function with a [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) docstring (PEP 257). The rules are set in `ruff.toml`:

```bash
pip install ruff
ruff check .
```

Command line options keep their original names (`--meshFilter`, `--capsOnly`, `--seedStepFactor`, `--seedDecimator`) for compatibility.

### Layout of `close_planar_hole.py`

| Section | Content |
|---|---|
| Debug exports | `_save`, `_save_lines`, `_save_points` |
| Mesh helpers | edge incidence, manifold patches, signed area and volume; `_get_bool_attr_or_method` (not used at the moment) |
| Input repair | `collapse_short_edges`, `remove_duplicate_faces`, `remove_small_flaps`, `split_non_manifold_edges`, `drop_extra_faces_on_non_manifold_edges`, `make_orientation_consistent`, `orient_outward`, `repair_input_mesh` |
| Boundary loops | `extract_boundary_loops`, loop orientation, outward direction |
| Planar cap | plane fit and placement, seeding, `triangulate_planar_cap`, `create_side_band`, `make_cap`; `keep_seeds_far_from_loop` (not used at the moment) |
| Curved and fan caps | `make_curved_cap`, `make_fan_cap`, repair caps |
| Hole-closing pipeline | `_attach_caps`, final repair passes, `create_caps`, `close_mesh_holes` |
| Command line | `build_argparser`, `main` |

Functions marked *not used at the moment* are kept for possible later use; their docstring explains why they are not called (see the `Notes` section).

---

# 👤 Author

Developed by **Guillaume Choquette**
Université de Sherbrooke — **SCIL** 2025 Internship Project
