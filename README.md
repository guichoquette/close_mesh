# 🧠 Brain Surface Hole Closing – Internship Project

Automaticaly close an open mesh with a plane and headband. Also does a scalar-field transfer for brain meshes.

This project provides a full workflow to:

1. **Automatically detect and close planar (or nearly planar) openings in a 3D mesh**,
2. **Generate triangulated caps** using projection, seeding, and constrained Delaunay triangulation,
3. **Copy all per-vertex scalar data** (e.g., thickness, curvature, sulcal depth) from the original open mesh to the closed mesh.
4. **Calculate the volume of closed meshes**, it make sure the new mesh is watertight. You can compute the volume with compute_volume.py or the pyvista option

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
├── utils/
│   ├── compute_volume.py                    # Script to compute the volume of a closed mesh
│   ├── copy_color.py                        # Scalar-field transfer tool
│   └── find_and_split_volume.py             # Debug tool to inspect individual disconnected surfaces
│
├── requirements.txt
└── README.md
```

---

# 🧠 Brain Data for the Project

You can download the meshes used during development from the official BraDiPho dataset:

🔗 **BraDiPho – Brain Digital Fossils Archive**
https://bradipho.eu/6-download.html

All specimens and epochs are publicly available and easy to download.

The folder `brain_data` also contains the **first epoch** of a specimen along with its **texture file**.
These two files are confirmed to work well with the full pipeline and are ideal for a **quick test run** of the scripts.

# 📦 Installation

Create a virtual environment:

python3 -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Main libraries:

- pyvista
- numpy
- scipy
- shapely
- trimesh
- matplotlib
- vtk

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

This script **automatically closes open boundary loops** in a triangulated mesh.

### How it works

1. Detect boundary loops
2. Fit a plane via **SVD** on loop vertices
3. Project loop points onto the plane
4. Generate a 2D sampling grid
5. Filter seeds inside polygon offsets
6. Perform constrained **Delaunay triangulation**
7. Keep only triangles inside the loop
8. Generate a side band
9. Merge caps into the mesh
10. Recompute normals

### Run

python close_opening_scripts/close_planar_hole.py \
 -i brain_data/sub-02_epo-01_ref-01_mesh.ply \
 -o brain_data/closed_solid/output.ply

---

# 🎨 Scalar Transfer Tool (`copy_color.py`)

Transfers **all per-vertex scalar arrays** from the open mesh to the newly closed mesh using **KD-Tree nearest-neighbor mapping**.

### Run

python utils/copy_color.py

Output:

brain_data/closed_solid/output_scalar_fields.ply

---

# ▶️ Full Workflow

### 1. Close the holes

python close_opening_scripts/close_planar_hole.py \
 -i brain_data/sub-02_epo-01_ref-01_mesh.ply \
 -o brain_data/closed_solid/output.ply \
 --debug brain_data/debug/

### 2. Transfer scalar fields

python utils/copy_color.py

### 3. Visualize in CloudCompare, MeshLab, or PyVista

---

# 🧪 Debug Outputs

When using `--debug`, the script exports:

- boundary loops
- projected loops
- grid seeds
- filtered seeds
- triangulation
- side band
- remaining open edges

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

python utils/copy_color.py \
 -s brain_data/sub-02_epo-01_ref-01_mesh.ply \
 -t brain_data/closed_solid/output.ply \
 -o brain_data/closed_solid/output_scalar_fields.ply

---

# ⚠️ Limitations / Known Issues

### 🔹 1. Smaller opening meshes increase debug output size

My algorithm works really well with larger holes. But when it finds smaller one it has trouble closing them.

### 🔹 2. Issue with the BraDiPho Meshes with epochs disection

As soon as disections comme in consideration, the meshes found on the BraDiPho website have structural holes, floting islands & the meshes are fractured in smallers surfaces. It is impacting my code making the reconstruction invalid for the moment.

### 🔹 3. Works best for planar or near-planar holes

Highly curved openings may distort projection or triangulation.

### 🔹 4. Sensitive to boundary noise

Meshes with extremely short or irregular edges should be pre-filtered using `--meshFilter`.

### 🔹 5. Concave openings may cause projection overlap

The algorithm assumes a single, well-behaved projected polygon.

### 🔹 6. Scalar transfer uses nearest-neighbor

This may smooth or blur sharp scalar variations.
Future improvement: weighted or barycentric interpolation.

### 🔹 7. Large meshes increase debug output size

The triangulation and debug exports scale with mesh size.

---

# 🚀 Future Work / Enhancements

### 🔧 Algorithmic Improvements

- **Adaptive seeding strategies**
  Increase sampling density near the boundary, reduce it inside. (reduce the over estimation of the volume)

- **Weighted scalar transfer**
  Replace nearest-neighbor with inverse-distance or barycentric interpolation.

- **Small hole handlers**
  I still need to find a way to adapt the script to handle diferently smaller holes.

### 🎨 Mesh Quality Enhancements

- **Pre-closing mesh clean up**
  To make the reconstruction valid on epochs with disection, we need a way to fuse the seperated surfaces of the mesh in a clean way. being able to get rid of the structural holes & floating island would also be a must.

- **Post-cap smoothing** (Taubin, Laplacian, or HC smoothing)
  Improves blending between cap and original surface.

- **Side-band optimization**
  Reduce long triangles by adding local remeshing.

---

# 👤 Author

Developed by **Guillaume Choquette**
Université de Sherbrooke — **SCIL** 2025 Internship Project
