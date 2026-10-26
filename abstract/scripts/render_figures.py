"""
Figures 1 and 3 of the abstract: off-screen PyVista renders of closed epochs.

Figure 1: an epoch that closes well (spc-01 epo-01): the textured photogrammetric
mesh, its opening, and the closed mesh with the caps in red.
Figure 3: what does not work: a folded opening closed by a curved cap
(spc-07 epo-02), small holes and tissue under the planar cap (spc-15 epo-08),
and a volume increase between two epochs (spc-17 epo-07 -> 08): the surface of
the later epoch lying outside the earlier one, in orange.

Cap faces are the faces of the closed mesh using a vertex that is not in the
input mesh. Back faces are drawn in blue, as CloudCompare shows inverted normals.

Usage
-----
python abstract/scripts/render_figures.py [fig1] [fig3]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "brain_data"
FIG = ROOT / "abstract" / "figures"

GRAY, RED, BLUE, ORANGE = "#d9d6cf", "#c0262d", "#2a78d6", "#eb6834"
INK = "#0b0b0b"


def paths(spc: str, epoch: str) -> tuple[Path, Path]:
    """Input and closed mesh of an epoch."""
    inp = next(DATA.glob(f"{spc}_hem-*_epo-all_dissection/bradipho_data/{spc}/dissection/"
                         f"{spc}_hem-*_epo-{epoch}_dissection.ply"))
    closed = next((DATA / "output" / spc).glob(f"{spc}_hem-*_epo-{epoch}_closed.ply"))
    return inp, closed


def cap_mask(closed: pv.PolyData, inp: pv.PolyData) -> np.ndarray:
    """Faces of the closed mesh that belong to a cap."""
    d, _ = cKDTree(np.asarray(inp.points)).query(np.asarray(closed.points), k=1)
    faces = closed.faces.reshape(-1, 4)[:, 1:]
    return (d > 1e-9)[faces].any(axis=1)


def largest_opening(inp: pv.PolyData) -> tuple[np.ndarray, np.ndarray]:
    """Centroid and outward normal of the largest opening of an input mesh."""
    edges = inp.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                      manifold_edges=False, non_manifold_edges=False).connectivity()
    best = None
    for rid in np.unique(edges["RegionId"]):
        pts = np.asarray(edges.extract_cells(np.where(edges.cell_data["RegionId"] == rid)[0]).points)
        if best is None or len(pts) > len(best):
            best = pts
    c = best.mean(axis=0)
    n = np.linalg.svd(best - c, full_matrices=False)[2][2]
    if n @ (c - np.asarray(inp.points).mean(axis=0)) < 0:
        n = -n
    return c, n


def plotter() -> pv.Plotter:
    p = pv.Plotter(off_screen=True, window_size=(1000, 1000))
    p.set_background("white")
    p.enable_anti_aliasing("ssaa")
    return p


def look(p: pv.Plotter, mesh: pv.PolyData, direction: np.ndarray, up=(0, 0, 1), zoom=1.0) -> None:
    """Camera looking at the mesh centre from ``direction``."""
    c = np.array(mesh.center)
    r = mesh.length
    p.camera_position = [tuple(c + direction / np.linalg.norm(direction) * r * 1.6), tuple(c), up]
    p.camera.zoom(zoom)


def add_with_backfaces(p: pv.Plotter, mesh: pv.PolyData, color: str, back_color: str = BLUE,
                       **kw) -> None:
    """Add a mesh whose back faces are drawn in ``back_color`` (blue: inverted normals)."""
    actor = p.add_mesh(mesh, color=color, smooth_shading=False, **kw)
    back = pv.Property(color=back_color)
    actor.backface_prop = back


def split_caps(closed: pv.PolyData, inp: pv.PolyData) -> tuple[pv.PolyData, pv.PolyData]:
    mask = cap_mask(closed, inp)
    return closed.extract_cells(np.where(~mask)[0]).extract_surface(), \
        closed.extract_cells(np.where(mask)[0]).extract_surface()


def shot(p: pv.Plotter) -> np.ndarray:
    img = p.screenshot(return_img=True)
    p.close()
    # Crop the white margin
    nz = np.where((img < 250).any(axis=2))
    if len(nz[0]):
        y0, y1, x0, x1 = nz[0].min(), nz[0].max(), nz[1].min(), nz[1].max()
        pad = 15
        img = img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
    return img


def panel(images: list[np.ndarray], titles: list[str], out: Path, width_in: float = 7.0) -> None:
    fig, axes = plt.subplots(1, len(images), figsize=(width_in, width_in / len(images) * 1.05))
    for ax, img, t in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(t, fontsize=8, color=INK, loc="left")
    fig.tight_layout(w_pad=0.5)
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(out)


def fig1() -> None:
    inp_path, closed_path = paths("spc-01", "01")
    inp, closed = pv.read(inp_path), pv.read(closed_path)
    c, n = largest_opening(inp)
    side = np.cross(n, [0, 0, 1])
    side = side if np.linalg.norm(side) > 1e-6 else np.array([1.0, 0, 0])
    view_open = n + 0.35 * side

    tex_path = inp_path.with_suffix(".jpg")
    imgs = []
    # (a) textured photogrammetric mesh, lateral view (opposite to the cut)
    p = plotter()
    if tex_path.exists() and "TCoords" in inp.point_data:
        inp.active_texture_coordinates = inp.point_data["TCoords"]
        p.add_mesh(inp, texture=pv.read_texture(str(tex_path)))
    else:
        p.add_mesh(inp, color=GRAY)
    look(p, inp, -n + 0.35 * side)
    imgs.append(shot(p))
    # (b) the opening: input mesh seen from the cut, boundary in red; the inside of
    # the open shell (its back faces) is seen through the opening, in dark gray
    p = plotter()
    add_with_backfaces(p, inp, GRAY, back_color="#77746d")
    edges = inp.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                      manifold_edges=False, non_manifold_edges=False)
    p.add_mesh(edges, color=RED, line_width=3)
    look(p, inp, view_open)
    imgs.append(shot(p))
    # (c) closed, watertight mesh, caps in red
    surf, caps = split_caps(closed, inp)
    p = plotter()
    add_with_backfaces(p, surf, GRAY)
    add_with_backfaces(p, caps, RED)
    look(p, closed, view_open)
    imgs.append(shot(p))
    panel(imgs, ["a  Photogrammetric mesh", "b  Open boundary (cut)", "c  Closed, watertight"],
          FIG / "fig1_pipeline.png")


def fig3() -> None:
    imgs, titles = [], []
    for spc, epoch, title in [("spc-07", "02", "a  Folded opening: curved cap"),
                              ("spc-15", "08", "b  Holes under the planar cap")]:
        inp_path, closed_path = paths(spc, epoch)
        inp, closed = pv.read(inp_path), pv.read(closed_path)
        c, n = largest_opening(inp)
        side = np.cross(n, [0, 0, 1])
        surf, caps = split_caps(closed, inp)
        p = plotter()
        add_with_backfaces(p, surf, GRAY)
        add_with_backfaces(p, caps, RED)
        look(p, closed, n + 0.5 * side)
        imgs.append(shot(p))
        titles.append(title)

    # (c) volume increase spc-17 epo-07 -> 08: surface of 08 outside 07
    _, a_path = paths("spc-17", "07")
    _, b_path = paths("spc-17", "08")
    a, b = pv.read(a_path), pv.read(b_path)
    b = b.compute_implicit_distance(a)
    outside = np.asarray(b["implicit_distance"]) > 1e-3
    faces = b.faces.reshape(-1, 4)[:, 1:]
    fo = outside[faces].all(axis=1)
    inp_b, _ = paths("spc-17", "08")
    c, n = largest_opening(pv.read(inp_b))
    side = np.cross(n, [0, 0, 1])
    p = plotter()
    p.add_mesh(a, color="#8a877f", opacity=0.25)
    p.add_mesh(b.extract_cells(np.where(~fo)[0]), color=GRAY)
    p.add_mesh(b.extract_cells(np.where(fo)[0]), color=ORANGE)
    look(p, b, n + 0.5 * side)
    imgs.append(shot(p))
    titles.append("c  Epoch 8 outside epoch 7 (+15 %)")
    panel(imgs, titles, FIG / "fig3_failures.png")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["fig1", "fig3"]
    FIG.mkdir(parents=True, exist_ok=True)
    if "fig1" in todo:
        fig1()
    if "fig3" in todo:
        fig3()
