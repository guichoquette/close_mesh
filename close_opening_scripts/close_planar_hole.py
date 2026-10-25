#!/usr/bin/env python3
"""
Close the openings of a triangulated surface mesh and check that it is watertight.

Developed for the dissected brain meshes of the BraDiPho dataset, whose
openings are mostly near-planar cuts, but usable on any surface mesh.

Pipeline (see `create_caps`)
----------------------------
1. Load the mesh, optionally collapse very short edges (``--meshFilter``).
2. Repair the input: remove duplicated triangles, small flaps and small open
   fragments, split the sheets meeting on non-manifold edges, make the
   winding consistent and orient the mesh outward.
3. Extract the boundary loops.
4. Close each loop with a cap:

   - a **planar cap** when the loop projects on its best-fit (SVD) plane as a
     simple polygon, after straightening its small folds: the plane is placed
     at the top of the loop on the outer side of the mesh, the polygon is
     triangulated with seed points (constrained Delaunay, Triangle) and
     joined to the 3D loop by a side band;
   - a **curved (harmonic) cap** otherwise.

5. Attach the caps to the mesh by vertex index.
6. Run final repair passes on the capped mesh (remaining non-manifold edges,
   small holes they leave).
7. Remove the fragments floating inside the brain or far from it, and orient
   every remaining part outward.
8. Report whether the result is watertight and, if so, its volume.

Usage
-----
    python close_planar_hole.py -i input.ply -o output.ply --meshFilter 1e-5 [--debug DIR]

Relative paths are resolved from this script's folder.

Dependencies: numpy, scipy, pyvista (VTK), shapely, triangle, trimesh, matplotlib.

Author: Guillaume Choquette, Université de Sherbrooke (SCIL), 2025.
"""

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyvista as pv
import triangle
import trimesh
from matplotlib.path import Path as MplPath
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
from scipy.spatial import Delaunay, cKDTree
from shapely.geometry import LineString, Polygon
from shapely.strtree import STRtree

# Maximum size (in faces) of a patch removed as a flap by `remove_small_flaps`
MAX_FLAP_FACES = 20
# Distance under which a loop vertex and its projection on the cap plane are merged
ON_PLANE_TOLERANCE = 1e-6
# Number of final repair passes run on the capped mesh
FINAL_REPAIR_PASSES = 3
# Largest fold of the projected loop (in vertices) straightened to keep a planar cap
MAX_FOLD_VERTICES = 20
# Largest isolated open part (in faces) removed as scan debris before capping
MAX_OPEN_FRAGMENT_FACES = 1000
# Parts farther than this fraction of the brain size (bounding box diagonal) are removed
FAR_PART_DISTANCE = 0.05
# A closed part is oriented by the sign of its volume only if |V| > FLAT_PART_RATIO * area^1.5
# (a sphere has V = 0.094 area^1.5; a flat, zero-thickness part has V ~ 0 and no inside)
FLAT_PART_RATIO = 0.01


# ============================================================================
# Debug exports
# ============================================================================

def _debug_path(args: argparse.Namespace, file_name: str) -> Path:
    """
    Path of a debug file, in the folder of the loop being processed.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments; ``args.debug`` is the debug folder and
        ``args.debug_loop`` the sub-folder of the current loop.
    file_name : str
        Name of the debug file (e.g. ``"debug_delaunay.ply"``).

    Returns
    -------
    path : Path
        Full path; its parent folder is created if needed.
    """
    path = Path(args.debug).resolve() / args.debug_loop / file_name
    path.parent.mkdir(exist_ok=True)
    return path


def _save_lines(points: np.ndarray, file_name: str, args: argparse.Namespace) -> None:
    """
    Export a closed polyline as a thin red tube, viewable in CloudCompare.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
        Polyline vertices, in order.
    file_name : str
        Name of the PLY file written in the debug folder.
    args : argparse.Namespace
        CLI arguments (debug folder).
    """
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    polyline = pv.PolyData(pts)
    polyline.lines = np.hstack(([n + 1], np.arange(n), 0)).astype(np.int64)  # closed: [n+1, 0, ..., n-1, 0]

    tube = polyline.tube(radius=0.0001, n_sides=16).triangulate()
    if tube.n_points == 0:
        return
    mesh = trimesh.Trimesh(vertices=tube.points, faces=tube.faces.reshape(-1, 4)[:, 1:], process=False)
    mesh.visual.vertex_colors = np.tile(np.array([255, 0, 0, 255], dtype=np.uint8), (tube.n_points, 1))
    mesh.export(_debug_path(args, file_name))


def _save_points(points: pv.PolyData, file_name: str, args: argparse.Namespace,
                 sphere_radius: float = 0.0001) -> None:
    """
    Export a point cloud as small white spheres, viewable in CloudCompare.

    Parameters
    ----------
    points : pv.PolyData
        Points to export (only their positions are used).
    file_name : str
        Name of the PLY file written in the debug folder.
    args : argparse.Namespace
        CLI arguments (debug folder).
    sphere_radius : float, optional
        Radius of each sphere. Default is 0.0001.
    """
    sphere = pv.Sphere(radius=sphere_radius, phi_resolution=16, theta_resolution=16)
    glyphs = points.glyph(scale=False, orient=False, geom=sphere).triangulate()
    if glyphs.n_points == 0:
        return
    mesh = trimesh.Trimesh(vertices=glyphs.points, faces=glyphs.faces.reshape(-1, 4)[:, 1:], process=False)
    mesh.visual.vertex_colors = np.full((glyphs.n_points, 4), 255, dtype=np.uint8)
    mesh.export(_debug_path(args, file_name))


def _save(poly_data: pv.PolyData, file_name: str, args: argparse.Namespace) -> None:
    """
    Export a mesh in the debug folder, in the format given by its extension.

    Parameters
    ----------
    poly_data : pv.PolyData
        Mesh to export.
    file_name : str
        File name. ``.ply``, ``.vtp``, ``.vtk``, ``.obj`` and ``.stl`` are
        written as meshes, ``.csv`` as point coordinates only; any other
        extension is replaced by ``.ply``.
    args : argparse.Namespace
        CLI arguments (debug folder).
    """
    path = _debug_path(args, file_name)
    ext = path.suffix.lower()
    if ext in {".ply", ".vtp", ".vtk", ".obj", ".stl"}:
        poly_data.save(path)
    elif ext == ".csv":
        xyz = poly_data.points if poly_data.n_points > 0 else np.empty((0, 3))
        np.savetxt(path, xyz, delimiter=",", header="x,y,z", comments="")
    else:
        poly_data.save(path.with_suffix(".ply"))


# ============================================================================
# Mesh helpers
# ============================================================================

def _faces_to_polydata(points: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    """
    Build a triangle mesh from points and (M, 3) face indices.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
        Vertex positions.
    faces : np.ndarray, shape (M, 3)
        Vertex indices of each triangle.

    Returns
    -------
    mesh : pv.PolyData
        Triangle mesh.
    """
    faces_flat = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(points, faces_flat)


def _edge_incidence(faces: np.ndarray, n_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Group the edges of all faces by unordered vertex pair.

    Parameters
    ----------
    faces : np.ndarray, shape (M, 3)
        Triangle vertex indices.
    n_points : int
        Number of mesh points (used to build the edge keys).

    Returns
    -------
    key : np.ndarray, shape (3 * M,)
        Unordered edge key ``min * n_points + max`` of every face edge; the
        edge k of face f is at row ``3 * f + k``.
    order : np.ndarray, shape (3 * M,)
        Row indices sorting `key`.
    start : np.ndarray
        Position in `order` of the first row of each distinct edge.
    counts : np.ndarray
        Number of faces sharing each distinct edge.
    """
    a = faces.ravel()
    b = np.roll(faces, -1, axis=1).ravel()
    key = np.minimum(a, b).astype(np.int64) * n_points + np.maximum(a, b)
    order = np.argsort(key, kind="stable")
    _, start, counts = np.unique(key[order], return_index=True, return_counts=True)
    return key, order, start, counts


def _manifold_patches(faces: np.ndarray, n_points: int) -> np.ndarray:
    """
    Label the patches of faces connected through manifold edges.

    Parameters
    ----------
    faces : np.ndarray, shape (M, 3)
        Triangle vertex indices.
    n_points : int
        Number of mesh points.

    Returns
    -------
    patch : np.ndarray, shape (M,)
        Patch id of every face; two faces are in the same patch when they
        are connected through edges shared by exactly two faces.
    """
    _, order, start, counts = _edge_incidence(faces, n_points)
    first, second = order[start[counts == 2]] // 3, order[start[counts == 2] + 1] // 3
    graph = coo_matrix((np.ones(len(first)), (first, second)), shape=(len(faces), len(faces)))
    return connected_components(graph, directed=False)[1]


def _remove_degenerate_faces(mesh: pv.PolyData) -> pv.PolyData:
    """
    Drop the triangles that reference the same vertex twice.

    They appear when a loop vertex lies on the cap plane and is merged with
    its projection; they would otherwise add an edge traversed both ways.

    Parameters
    ----------
    mesh : pv.PolyData
        Triangle mesh.

    Returns
    -------
    mesh : pv.PolyData
        Same mesh without degenerate triangles.
    """
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    return _faces_to_polydata(mesh.points, faces[keep])


def _signed_area_uv(uv: np.ndarray) -> np.ndarray:
    """
    Signed area of one or several 2D polygons.

    Parameters
    ----------
    uv : np.ndarray, shape (..., N, 2)
        Polygon vertices, in order.

    Returns
    -------
    area : np.ndarray, shape (...)
        Signed area, positive for counter-clockwise polygons.
    """
    x, y = uv[..., 0], uv[..., 1]
    return 0.5 * np.sum(x * np.roll(y, -1, axis=-1) - np.roll(x, -1, axis=-1) * y, axis=-1)


def _get_bool_attr_or_method(obj: object, name: str, default: bool = True) -> bool:
    """
    Read a boolean attribute of an object, calling it if it is a method.

    Older PyVista versions exposed some mesh checks (e.g. ``is_manifold``) as
    methods and newer ones as properties; this reads both.

    Parameters
    ----------
    obj : object
        Object to query.
    name : str
        Name of the attribute or method.
    default : bool, optional
        Value returned if the attribute does not exist. Default is True.

    Returns
    -------
    value : bool
        The attribute (or the result of calling it without arguments; if the
        call raises a TypeError, the attribute itself), cast to bool.

    Notes
    -----
    Not used at the moment: `close_mesh_holes` reads ``is_manifold`` and
    ``is_all_triangles`` directly, as properties of the PyVista version in
    use. Kept for compatibility with older PyVista versions.
    """
    value = getattr(obj, name, default)
    if callable(value):
        try:
            value = value()
        except TypeError:
            pass
    return bool(value)


def _connected_parts(mesh: pv.PolyData) -> np.ndarray:
    """
    Label the connected parts of a mesh (faces sharing a vertex).

    Parameters
    ----------
    mesh : pv.PolyData
        Triangle mesh.

    Returns
    -------
    labels : np.ndarray, shape (M,)
        Part id of every face, in the order of ``mesh.faces``.
    """
    return np.asarray(mesh.connectivity(extraction_mode="all").cell_data["RegionId"])


def _signed_volume(mesh: pv.PolyData) -> float:
    """
    Signed volume of a triangle mesh (divergence theorem).

    Parameters
    ----------
    mesh : pv.PolyData
        Triangle mesh.

    Returns
    -------
    volume : float
        Positive when the normals of a closed mesh point outward.
    """
    tri = mesh.points[mesh.faces.reshape(-1, 4)[:, 1:]].astype(np.float64)
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


# ============================================================================
# Input repair
# ============================================================================

def collapse_short_edges(mesh: pv.PolyData, tol: float = 1e-5) -> pv.PolyData:
    """
    Collapse the edges shorter than `tol` and remove the degenerate triangles.

    The two vertices of each short edge are merged at their midpoint
    (union-find), the triangles are remapped and those with a repeated
    vertex are removed.

    Parameters
    ----------
    mesh : pv.PolyData
        Mesh to clean (triangulated if needed).
    tol : float, optional
        Maximum length of a collapsed edge. Default is 1e-5.

    Returns
    -------
    mesh : pv.PolyData
        Cleaned triangle mesh.
    """
    mesh = mesh.triangulate().copy(deep=True)
    pts = mesh.points.copy()
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    n_pts = pts.shape[0]

    # Union-find structure remapping vertices after the collapses
    parent = np.arange(n_pts, dtype=np.int64)

    def find(x: int) -> int:
        """Representative of `x`, with path compression."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> int:
        """Merge the sets of `a` and `b`; the smallest index represents the set."""
        ra, rb = find(a), find(b)
        if ra == rb:
            return ra
        if ra < rb:
            parent[rb] = ra
            return ra
        parent[ra] = rb
        return rb

    # 1) Edges shorter than the tolerance
    short_edges = set()
    for i, j, k in faces:
        for a, b in ((i, j), (j, k), (k, i)):
            if a != b and np.linalg.norm(pts[a] - pts[b]) < tol:
                short_edges.add(tuple(sorted((a, b))))
    if not short_edges:
        return mesh

    # 2) Merge their vertices at the midpoint
    for i, j in short_edges:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        midpoint = 0.5 * (pts[ri] + pts[rj])
        pts[union(ri, rj)] = midpoint

    # 3) Remap the faces and drop those with a repeated vertex
    for idx in range(n_pts):
        parent[idx] = find(idx)
    new_faces = parent[faces]
    valid = ((new_faces[:, 0] != new_faces[:, 1]) & (new_faces[:, 1] != new_faces[:, 2])
             & (new_faces[:, 0] != new_faces[:, 2]))
    new_faces = new_faces[valid]
    if new_faces.shape[0] == 0:
        return pv.PolyData(pts)  # everything collapsed
    return _faces_to_polydata(pts, new_faces)


def remove_duplicate_faces(mesh: pv.PolyData) -> pv.PolyData:
    """
    Remove every triangle that appears more than once, in any winding.

    Scanned meshes sometimes contain two identical triangles wound in
    opposite directions: a zero-thickness flap. Each of its edges is shared
    by three or more faces, and the two copies contradict each other's
    orientation. Removing all copies leaves the surrounding surface intact.

    Parameters
    ----------
    mesh : pv.PolyData
        Triangle mesh.

    Returns
    -------
    mesh : pv.PolyData
        Mesh without duplicated triangles.
    """
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    _, inverse, counts = np.unique(np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True)
    unique_faces = counts[inverse.ravel()] == 1
    if unique_faces.all():
        return mesh
    return _faces_to_polydata(mesh.points, faces[unique_faces]).clean()


def remove_small_flaps(mesh: pv.PolyData, max_faces: int = MAX_FLAP_FACES) -> pv.PolyData:
    """
    Remove the small patches attached to the rest of the mesh only by non-manifold edges.

    Scanned meshes contain small flaps (a few triangles) glued on the surface
    along an edge shared by three or more faces. A patch is the set of faces
    reachable through manifold edges; patches of at most `max_faces` faces
    that touch a non-manifold edge are removed. The surface they were glued
    on stays intact; at worst a small hole appears, which is capped later.

    Parameters
    ----------
    mesh : pv.PolyData
        Triangle mesh.
    max_faces : int, optional
        Largest patch considered as a flap. Default is `MAX_FLAP_FACES`.

    Returns
    -------
    mesh : pv.PolyData
        Mesh without small flaps.
    """
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    _, order, start, counts = _edge_incidence(faces, mesh.n_points)
    non_manifold = counts > 2
    if not non_manifold.any():
        return mesh
    nm_faces = np.concatenate([order[s:s + c] // 3
                               for s, c in zip(start[non_manifold], counts[non_manifold], strict=True)])

    patch = _manifold_patches(faces, mesh.n_points)
    patch_size = np.bincount(patch)
    flap_patches = np.unique(patch[nm_faces][patch_size[patch[nm_faces]] <= max_faces])
    if len(flap_patches) == 0:
        return mesh
    return _faces_to_polydata(mesh.points, faces[~np.isin(patch, flap_patches)])


def _fan_around_edge(faces: np.ndarray, points: np.ndarray, edge_faces: np.ndarray,
                     a: int, b: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Sort the faces incident to edge (a, b) by angle around it.

    Parameters
    ----------
    faces : np.ndarray, shape (M, 3)
        Triangle vertex indices of the mesh.
    points : np.ndarray, shape (N, 3)
        Vertex positions of the mesh.
    edge_faces : np.ndarray
        Indices of the faces sharing the edge.
    a, b : int
        Vertices of the edge.

    Returns
    -------
    fan : np.ndarray
        Face indices sorted by angle around the edge.
    angles : np.ndarray
        Their angle around the edge, in radians.
    forward : dict
        For each face, whether it traverses the edge as a -> b.
    """
    axis = points[b] - points[a]
    axis = axis / (np.linalg.norm(axis) + 1e-30)
    angles = []
    ref = ref2 = None
    for f in edge_faces:
        apex = points[next(v for v in faces[f] if v not in (a, b))] - points[a]
        radial = apex - (apex @ axis) * axis
        if ref is None:
            ref = radial / (np.linalg.norm(radial) + 1e-30)
            ref2 = np.cross(axis, ref)
        angles.append(np.arctan2(radial @ ref2, radial @ ref))
    order = np.argsort(angles)
    fan = np.asarray(edge_faces)[order]
    forward = {int(f): faces[f, (list(faces[f]).index(a) + 1) % 3] == b for f in fan}
    return fan, np.asarray(angles)[order], forward


def split_non_manifold_edges(mesh: pv.PolyData) -> pv.PolyData:
    """
    Separate the sheets of surface that meet along non-manifold edges.

    Around an edge shared by more than two faces, the faces are sorted by
    angle and paired with an angular neighbour (so that sheets do not cross),
    preferring pairs that traverse the edge in opposite directions (a
    consistently oriented sheet). Each pair then forms its own manifold
    sheet: every vertex of these edges is duplicated, so that each group of
    faces connected around it (through manifold edges or paired faces) gets
    its own copy.

    Vertices on a boundary are not split: two openings touching at such a
    vertex would be joined into a single figure-of-eight loop, closed by one
    cap instead of two. An edge with both ends on a boundary therefore stays
    non-manifold; it is split again once the caps are attached.

    No geometry is removed or moved: duplicated vertices share the same
    position, so coincident points must not be merged afterwards (e.g. by
    ``clean()``), or the sheets would be joined again.

    Parameters
    ----------
    mesh : pv.PolyData
        Triangle mesh.

    Returns
    -------
    mesh : pv.PolyData
        Mesh whose non-manifold edges are split.
    """
    faces = mesh.faces.reshape(-1, 4)[:, 1:].copy()
    points = np.asarray(mesh.points)
    n_points = mesh.n_points
    key, order, start, counts = _edge_incidence(faces, n_points)
    non_manifold = np.where(counts > 2)[0]
    if len(non_manifold) == 0:
        return mesh

    # Pairs of faces allowed to stay connected through each non-manifold edge
    paired = []
    for e in non_manifold:
        rows = order[start[e]:start[e] + counts[e]]
        a, b = divmod(int(key[rows[0]]), n_points)
        fan, _, forward = _fan_around_edge(faces, points, rows // 3, a, b)
        # Two ways to pair angular neighbours: keep the one with most consistently oriented pairs
        best = None
        for shift in (0, 1):
            ring = np.roll(fan, -shift)
            pairs = [(int(ring[i]), int(ring[i + 1])) for i in range(0, len(ring) - 1, 2)]
            score = sum(forward[f1] != forward[f2] for f1, f2 in pairs)
            if best is None or score > best[0]:
                best = (score, pairs)
        paired += [(f1, f2, a, b) for f1, f2 in best[1]]

    # Vertices to split: those of non-manifold edges, except boundary vertices
    nm_vertices = np.unique(np.concatenate(divmod(key[order[start[non_manifold]]], n_points)))
    boundary_vertices = np.unique(np.concatenate(divmod(key[order[start[counts == 1]]], n_points)))
    nm_vertices = np.setdiff1d(nm_vertices, boundary_vertices)
    touches_nm = np.isin(faces, nm_vertices).any(axis=1)

    # Face adjacency through manifold edges, plus the chosen pairs on non-manifold edges
    two = counts == 2
    first, second = order[start[two]] // 3, order[start[two] + 1] // 3
    shared = key[order[start[two]]]
    adjacency = defaultdict(list)
    for f1, f2, k in zip(first.tolist(), second.tolist(), shared.tolist(), strict=True):
        if touches_nm[f1] or touches_nm[f2]:
            edge = divmod(k, n_points)
            adjacency[f1].append((f2, edge))
            adjacency[f2].append((f1, edge))
    for f1, f2, a, b in paired:
        adjacency[f1].append((f2, (a, b)))
        adjacency[f2].append((f1, (a, b)))

    # One copy of each vertex per connected fan of faces around it
    vertex_faces = defaultdict(list)
    for f in np.where(touches_nm)[0]:
        for v in faces[f]:
            if v in nm_vertices:
                vertex_faces[int(v)].append(int(f))

    new_points = [points]
    next_index = n_points
    for v, fan_faces in vertex_faces.items():
        fan_set = set(fan_faces)
        root_of = {}
        for seed in fan_faces:
            if seed in root_of:
                continue
            root_of[seed] = seed
            stack = [seed]
            while stack:
                f = stack.pop()
                for g, edge in adjacency[f]:
                    if v in edge and g in fan_set and g not in root_of:
                        root_of[g] = seed
                        stack.append(g)
        roots = list(dict.fromkeys(root_of.values()))
        for root in roots[1:]:
            for f, r in root_of.items():
                if r == root:
                    faces[f][faces[f] == v] = next_index
            new_points.append(points[v][None, :])
            next_index += 1

    return _faces_to_polydata(np.vstack(new_points), faces)


def drop_extra_faces_on_non_manifold_edges(mesh: pv.PolyData) -> pv.PolyData:
    """
    On each non-manifold edge, keep only the two faces that best continue the surface.

    Some non-manifold edges cannot be split: the faces around them reconnect
    around both end vertices (a small fold or pocket of the scan). On such an
    edge, the kept pair is the one that traverses the edge in opposite
    directions (consistent orientation) and is the closest to flat; the other
    faces are removed. The small holes this opens are capped afterwards.

    Parameters
    ----------
    mesh : pv.PolyData
        Triangle mesh.

    Returns
    -------
    mesh : pv.PolyData
        Mesh without the extra faces.
    """
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    points = np.asarray(mesh.points)
    key, order, start, counts = _edge_incidence(faces, mesh.n_points)
    non_manifold = np.where(counts > 2)[0]
    if len(non_manifold) == 0:
        return mesh

    drop = set()
    for e in non_manifold:
        rows = order[start[e]:start[e] + counts[e]]
        a, b = divmod(int(key[rows[0]]), mesh.n_points)
        fan, angles, forward = _fan_around_edge(faces, points, rows // 3, a, b)
        best = None
        for i in range(len(fan)):
            for j in range(i + 1, len(fan)):
                opening = abs(angles[i] - angles[j])
                opening = min(opening, 2 * np.pi - opening)  # pi for a flat sheet
                score = (forward[int(fan[i])] != forward[int(fan[j])], opening)
                if best is None or score > best[0]:
                    best = (score, {int(fan[i]), int(fan[j])})
        drop |= {int(f) for f in fan} - best[1]

    keep = np.ones(len(faces), dtype=bool)
    keep[list(drop)] = False
    return _faces_to_polydata(mesh.points, faces[keep])


def remove_small_open_fragments(mesh: pv.PolyData, max_faces: int = MAX_OPEN_FRAGMENT_FACES) -> pv.PolyData:
    """
    Remove the small isolated parts that are open: pieces of scan debris.

    Such a fragment, often lying at the cut, would otherwise be closed as a
    tiny flat part with its own cap, which may end up wound inward and is
    seen as an inverted spot on the cap. Closed fragments are kept here (see
    `remove_inner_and_far_parts`), and the largest part is always kept.

    Parameters
    ----------
    mesh : pv.PolyData
        Triangle mesh.
    max_faces : int, optional
        Largest fragment removed. Default is `MAX_OPEN_FRAGMENT_FACES`.

    Returns
    -------
    mesh : pv.PolyData
        Mesh without small open fragments.
    """
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    labels = _connected_parts(mesh)
    size = np.bincount(labels)
    _, order, start, counts = _edge_incidence(faces, mesh.n_points)
    is_open = np.zeros(len(size), dtype=bool)
    is_open[labels[order[start[counts == 1]] // 3]] = True
    drop = is_open & (size <= max_faces)
    drop[size.argmax()] = False
    if not drop.any():
        return mesh
    return _faces_to_polydata(mesh.points, faces[~drop[labels]])


def make_orientation_consistent(mesh: pv.PolyData) -> pv.PolyData:
    """
    Wind neighbouring triangles consistently, without crossing non-manifold edges.

    The orientation is propagated face to face across manifold edges only
    (edges shared by exactly two faces). Within each patch reached this way,
    the winding held by the majority of its faces is kept, so a mostly
    correct surface is only changed where it was wrong.

    VTK's ``consistent_normals`` also walks through non-manifold edges, which
    can flip whole regions of a scanned mesh; this is why it is not used.

    Parameters
    ----------
    mesh : pv.PolyData
        Triangle mesh.

    Returns
    -------
    mesh : pv.PolyData
        Mesh with a consistent winding within each manifold patch.
    """
    faces = mesh.faces.reshape(-1, 4)[:, 1:].copy()
    n_faces = faces.shape[0]

    # Directed edges (a -> b) of every face, keyed by their unordered pair
    a = faces.ravel()
    b = np.roll(faces, -1, axis=1).ravel()
    face_of_edge = np.repeat(np.arange(n_faces), 3)
    forward = a < b
    key = np.minimum(a, b).astype(np.int64) * mesh.n_points + np.maximum(a, b)

    # Manifold edges: exactly two incident faces. The same direction on both means they disagree.
    order = np.argsort(key, kind="stable")
    _, start, counts = np.unique(key[order], return_index=True, return_counts=True)
    first = order[start[counts == 2]]
    second = order[start[counts == 2] + 1]
    disagree = forward[first] == forward[second]

    neighbours = defaultdict(list)
    for i, j, d in zip(face_of_edge[first].tolist(), face_of_edge[second].tolist(), disagree.tolist(),
                        strict=True):
        neighbours[i].append((j, d))
        neighbours[j].append((i, d))

    # Walk each patch, recording which faces must flip, then keep its majority winding
    flip = np.zeros(n_faces, dtype=bool)
    visited = np.zeros(n_faces, dtype=bool)
    for seed in range(n_faces):
        if visited[seed]:
            continue
        visited[seed] = True
        patch = [seed]
        stack = [seed]
        while stack:
            current = stack.pop()
            for other, d in neighbours[current]:
                if not visited[other]:
                    visited[other] = True
                    flip[other] = flip[current] ^ d
                    patch.append(other)
                    stack.append(other)
        if 2 * flip[patch].sum() > len(patch):
            flip[patch] = ~flip[patch]

    if not flip.any():
        return mesh
    faces[flip] = faces[flip][:, ::-1]
    return _faces_to_polydata(mesh.points, faces)


def orient_outward(mesh: pv.PolyData) -> pv.PolyData:
    """
    Flip the whole mesh if its signed volume is negative (normals pointing inward).

    Parameters
    ----------
    mesh : pv.PolyData
        Consistently oriented triangle mesh.

    Returns
    -------
    mesh : pv.PolyData
        Same mesh, wound so that its normals point outward.
    """
    return mesh.flip_faces() if _signed_volume(mesh) < 0 else mesh


def repair_input_mesh(mesh: pv.PolyData) -> pv.PolyData:
    """
    Repair a mesh so that each boundary edge has a well-defined winding the caps can follow.

    Removes duplicated triangles, small flaps and small open fragments,
    separates the sheets meeting on non-manifold edges, propagates a
    consistent winding and orients the mesh outward.

    Parameters
    ----------
    mesh : pv.PolyData
        Triangle mesh.

    Returns
    -------
    mesh : pv.PolyData
        Repaired mesh, oriented outward.
    """
    mesh = remove_duplicate_faces(mesh)
    mesh = remove_small_flaps(mesh)
    mesh = remove_small_open_fragments(mesh)
    mesh = split_non_manifold_edges(mesh)
    return orient_outward(make_orientation_consistent(mesh))


# ============================================================================
# Boundary loops
# ============================================================================

def extract_boundary_loops(mesh: pv.PolyData) -> list[np.ndarray]:
    """
    Extract the boundary loops (openings) of a surface mesh.

    Boundary edges (used by a single triangle) are chained into loops by
    walking the boundary edge by edge. A vertex shared by two openings
    (pinched boundary) is visited twice: the sub-cycle closed at that point
    is cut out as its own loop, so every loop is simple.

    Non-manifold edges can cut a boundary into open chains (odd-degree end
    vertices): walks start from those ends first. The chains are then joined
    end to end through the non-manifold edges linking their ends, which
    rebuilds the opening they belong to; a chain that cannot be joined is
    closed between its ends. (Closing every chain on itself could add an
    edge of several centimetres across the opening.)

    Parameters
    ----------
    mesh : pv.PolyData
        Triangle mesh.

    Returns
    -------
    loops : list of np.ndarray
        Vertex indices into `mesh.points` of each loop, ordered along the
        boundary (in an arbitrary direction).
    """
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    edge_count = Counter(tuple(sorted((int(f[k]), int(f[(k + 1) % 3])))) for f in faces for k in range(3))
    neighbours = defaultdict(list)
    nm_neighbours = defaultdict(set)
    for (a, b), count in edge_count.items():
        if count == 1:
            neighbours[a].append(b)
            neighbours[b].append(a)
        elif count > 2:
            nm_neighbours[a].add(b)
            nm_neighbours[b].add(a)

    used = set()
    loops = []
    open_chains = []
    starts = sorted(neighbours, key=lambda v: len(neighbours[v]) % 2 == 0)  # open chain ends first
    for start in starts:
        for first in neighbours[start]:
            if tuple(sorted((start, first))) in used:
                continue
            path = [start]
            position = {start: 0}
            cur, nxt = start, first
            while True:
                used.add(tuple(sorted((cur, nxt))))
                if nxt in position:
                    # Closed a cycle: emit it and resume from the vertex where it started
                    cut = position[nxt]
                    loops.append(np.array(path[cut:], dtype=int))
                    for v in path[cut + 1:]:
                        del position[v]
                    path = path[:cut + 1]
                    if nxt == start and len(path) == 1:
                        break
                else:
                    position[nxt] = len(path)
                    path.append(nxt)
                cur = nxt
                candidates = [w for w in neighbours[cur] if tuple(sorted((cur, w))) not in used]
                if not candidates:
                    if len(path) > 1:
                        open_chains.append(path)  # open chain, joined to the others below
                    break
                nxt = candidates[0]
    return loops + _join_open_chains(open_chains, nm_neighbours)


def _join_open_chains(chains: list[list[int]], nm_neighbours: dict[int, set[int]]) -> list[np.ndarray]:
    """
    Join open boundary chains end to end through the non-manifold edges linking their ends.

    Parameters
    ----------
    chains : list of list of int
        Open chains of boundary vertices, in order.
    nm_neighbours : dict
        For each vertex, the vertices it shares a non-manifold edge with.

    Returns
    -------
    loops : list of np.ndarray
        Closed loops (more than two vertices). A loop is closed by a
        non-manifold edge when one links its ends, otherwise directly
        between its ends.
    """
    pool = [list(chain) for chain in chains]
    loops = []
    while pool:
        loop = pool.pop(0)
        while loop[-1] not in nm_neighbours.get(loop[0], ()) or len(loop) < 3:
            end = loop[-1]
            linked = nm_neighbours.get(end, set()) | {end}
            for k, chain in enumerate(pool):
                if chain[0] in linked:
                    loop += chain[1:] if chain[0] == end else chain
                elif chain[-1] in linked:
                    loop += chain[-2::-1] if chain[-1] == end else chain[::-1]
                else:
                    continue
                del pool[k]
                break
            else:
                break  # no chain to join: closed between its ends
        if loop[-1] == loop[0]:
            loop = loop[:-1]
        if len(loop) > 2:
            loops.append(np.array(loop, dtype=int))
    return loops


def _loop_edges(loop_indices: np.ndarray) -> set[tuple[int, int]]:
    """
    Unordered edges of a closed loop.

    Parameters
    ----------
    loop_indices : np.ndarray
        Loop vertex indices, in order.

    Returns
    -------
    edges : set of tuple of int
        Edges ``(min, max)`` between consecutive vertices, closing edge included.
    """
    loop = np.asarray(loop_indices, dtype=int)
    return {tuple(sorted((loop[i], loop[(i + 1) % len(loop)]))) for i in range(len(loop))}


def _orient_loop_against_mesh(mesh: pv.PolyData, loop_indices: np.ndarray) -> np.ndarray:
    """
    Order a boundary loop so that it runs opposite to the mesh winding.

    Each boundary edge belongs to a single triangle, which traverses it in a
    given direction; a cap glued on that edge must traverse it the other way
    for the closed surface to be consistently oriented. The order returned by
    `extract_boundary_loops` is arbitrary, so it is fixed here once, and
    every cap is then built following this order. Where the input mesh has
    local defects, a few edges may disagree: the majority is followed.

    Parameters
    ----------
    mesh : pv.PolyData
        Triangle mesh containing the opening.
    loop_indices : np.ndarray
        Loop vertex indices into `mesh.points`, in order.

    Returns
    -------
    loop_indices : np.ndarray
        Same loop, reversed if needed so that each edge (L[i], L[i+1]) is
        traversed by the mesh as L[i+1] -> L[i].

    Raises
    ------
    ValueError
        If no loop edge is found in the mesh, or if as many edges run each way.
    """
    loop = np.asarray(loop_indices, dtype=int)
    loop_edges = _loop_edges(loop)

    # Loop edges, as traversed by the mesh triangles
    mesh_directed = set()
    for face in mesh.faces.reshape(-1, 4)[:, 1:]:
        for k in range(3):
            a, b = int(face[k]), int(face[(k + 1) % 3])
            if tuple(sorted((a, b))) in loop_edges:
                mesh_directed.add((a, b))

    same = opposite = 0
    for i in range(len(loop)):
        a, b = int(loop[i]), int(loop[(i + 1) % len(loop)])
        if (a, b) in mesh_directed and (b, a) not in mesh_directed:
            same += 1
        elif (b, a) in mesh_directed and (a, b) not in mesh_directed:
            opposite += 1

    if same == 0 and opposite == 0:
        raise ValueError("Cannot orient loop: no boundary edge found in the mesh faces.")
    if same == opposite:
        raise ValueError(f"Cannot orient loop: mesh winding is ambiguous along the boundary "
                         f"({same} edges each way).")
    return loop[::-1].copy() if same > opposite else loop


def _outward_direction(mesh: pv.PolyData, loop_indices: np.ndarray) -> np.ndarray:
    """
    Mean outward normal of the mesh faces touching a loop.

    Parameters
    ----------
    mesh : pv.PolyData
        Triangle mesh, oriented outward (see `orient_outward`).
    loop_indices : np.ndarray
        Loop vertex indices into `mesh.points`.

    Returns
    -------
    direction : np.ndarray, shape (3,)
        Unnormalised vector pointing to the outer side of the opening.
    """
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    tri = np.asarray(mesh.points, dtype=float)[faces[np.isin(faces, loop_indices).any(axis=1)]]
    return np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).sum(axis=0)


def perimeter_mean_length(loop_points: np.ndarray) -> float:
    """
    Mean edge length of a closed loop.

    Parameters
    ----------
    loop_points : np.ndarray, shape (N, 3)
        Loop vertices, in order.

    Returns
    -------
    mean_length : float
        Mean length of the loop edges, closing edge included (0 if N < 2).
    """
    n = len(loop_points)
    if n < 2:
        return 0.0
    segments = loop_points[(np.arange(n) + 1) % n] - loop_points
    return float(np.linalg.norm(segments, axis=1).mean())


# ============================================================================
# Planar cap
# ============================================================================

def fit_plane_svd(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Best-fit plane of a set of points (SVD).

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
        Points to fit.

    Returns
    -------
    center : np.ndarray, shape (3,)
        Mean of the points, on the plane.
    u, v : np.ndarray, shape (3,)
        Orthonormal in-plane basis.
    n : np.ndarray, shape (3,)
        Unit normal of the plane (its sign is arbitrary).
    """
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    return center, vh[0, :], vh[1, :], vh[2, :]


def _crossing_segments(uv: np.ndarray) -> list[tuple[int, int]]:
    """
    Pairs of non-adjacent edges of a closed 2D polyline that intersect.

    Parameters
    ----------
    uv : np.ndarray, shape (N, 2)
        Polyline vertices; edge i joins vertex i to vertex i + 1 (closing edge included).

    Returns
    -------
    pairs : list of tuple of int
        Indices (i, j), i < j, of the crossing edges.
    """
    n = len(uv)
    segments = [LineString([uv[i], uv[(i + 1) % n]]) for i in range(n)]
    tree = STRtree(segments)
    pairs = []
    for i, segment in enumerate(segments):
        for j in tree.query(segment):
            j = int(j)
            if j > i + 1 and not (i == 0 and j == n - 1) and segment.intersects(segments[j]):
                pairs.append((i, j))
    return pairs


def untangle_projected_loop(uv: np.ndarray, max_fold: int = MAX_FOLD_VERTICES,
                            max_rounds: int = 5) -> np.ndarray | None:
    """
    Straighten the small folds of a projected loop so that it becomes a simple polygon.

    On most large BraDiPho openings, the projected loop crosses itself only at
    a micro-fold of a few vertices. The vertices between two crossing edges
    (along the shorter side, at most `max_fold` vertices) are moved onto the
    chord joining both sides of the fold, in the plane. The side band still
    joins each 3D vertex to its (moved) projection, so the cap stays closed.

    Parameters
    ----------
    uv : np.ndarray, shape (N, 2)
        Projected loop in the (u, v) frame of the cap plane.
    max_fold : int, optional
        Largest fold straightened. Default is `MAX_FOLD_VERTICES`.
    max_rounds : int, optional
        Number of straightening rounds before giving up. Default is 5.

    Returns
    -------
    uv : np.ndarray, shape (N, 2) or None
        The loop as a simple polygon, or None if a fold is too large.
    """
    uv = np.array(uv, dtype=float)
    n = len(uv)
    for _ in range(max_rounds):
        if Polygon(uv).is_valid:
            return uv
        pairs = _crossing_segments(uv)
        if not pairs:
            return None
        for i, j in pairs:
            if j - i <= max_fold:
                first, last = i, j + 1          # edge i starts at vertex i, edge j ends at vertex j + 1
            elif n - (j - i) <= max_fold:
                first, last = j, i + 1 + n      # the shorter side wraps around the start of the loop
            else:
                return None
            moved = np.arange(first + 1, last) % n
            weight = (np.arange(1, last - first) / (last - first))[:, None]
            uv[moved] = (1 - weight) * uv[first % n] + weight * uv[last % n]
    return uv if Polygon(uv).is_valid else None


def _find_projection_plane(loop_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Return the SVD best-fit plane if the loop projects on it as a simple polygon.

    Small folds of the projected loop are accepted: they are straightened by
    `untangle_projected_loop`. A really warped opening projects to a
    self-intersecting polygon, for which no valid planar cap exists; it gets
    a curved cap instead. (Tilted planes were tried once, but on large
    openings the cap then lies far from the loop and the side band adds or
    removes a large wedge of volume.)

    Parameters
    ----------
    loop_points : np.ndarray, shape (N, 3)
        Loop vertices, in order.

    Returns
    -------
    plane : tuple of np.ndarray or None
        ``(center, u, v, normal)`` of the SVD plane, or None if the projected
        loop is not a simple polygon, even after straightening its small folds.
    """
    center, u, v, n = fit_plane_svd(loop_points)
    rel = loop_points - center
    uv = np.column_stack([rel @ u, rel @ v])
    polygon = Polygon(uv)
    if (polygon.is_valid and polygon.area > 0) or untangle_projected_loop(uv) is not None:
        return center, u, v, n
    return None


def place_plane_on_loop_top(center: np.ndarray, normal: np.ndarray, loop_points: np.ndarray,
                            outward_hint: np.ndarray | None = None) -> np.ndarray:
    """
    Slide a plane along its normal so that it touches the top of the opening.

    Parameters
    ----------
    center : np.ndarray, shape (3,)
        Point on the plane.
    normal : np.ndarray, shape (3,)
        Plane normal.
    loop_points : np.ndarray, shape (N, 3)
        Loop vertices.
    outward_hint : np.ndarray, shape (3,), optional
        Direction of the outer side of the mesh. The normal is flipped to
        point that way first, so that "top" means the outer side; without it,
        the side depends on the arbitrary sign of the normal.

    Returns
    -------
    center : np.ndarray, shape (3,)
        Point of the moved plane, at the loop vertex furthest along the normal.
    """
    n = normal / (np.linalg.norm(normal) + 1e-12)
    if outward_hint is not None and np.dot(n, outward_hint) < 0:
        n = -n
    top = float(np.max(loop_points @ n))
    return center + (top - float(np.dot(center, n))) * n


def project_points_on_plane(points: np.ndarray, center: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    Orthogonal projection of points on a plane.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
        Points to project.
    center : np.ndarray, shape (3,)
        Point on the plane.
    normal : np.ndarray, shape (3,)
        Plane normal.

    Returns
    -------
    projected : np.ndarray, shape (N, 3)
        Projected points.
    """
    n = normal / (np.linalg.norm(normal) + 1e-12)
    distance = (points - center) @ n
    return points - np.outer(distance, n)


def offset_polygon_inner(poly_uv: np.ndarray, delta: float) -> np.ndarray:
    """
    Inward offset of a 2D polygon (negative Shapely buffer).

    Parameters
    ----------
    poly_uv : np.ndarray, shape (N, 2)
        Polygon vertices, in order (implicitly closed).
    delta : float
        Offset distance (positive, inward).

    Returns
    -------
    inner : np.ndarray, shape (M, 2)
        Exterior ring of the offset polygon (largest part if it splits), or
        an empty (0, 2) array if the polygon is completely eroded.
    """
    if poly_uv.shape[0] < 3:
        return np.zeros((0, 2), dtype=float)
    polygon = Polygon(poly_uv)
    if not polygon.is_valid:
        polygon = polygon.buffer(0.0)  # simple repair
        if polygon.is_empty:
            return np.zeros((0, 2), dtype=float)

    inner = polygon.buffer(-float(delta))
    if inner.is_empty:
        return np.zeros((0, 2), dtype=float)
    if inner.geom_type == "MultiPolygon":
        inner = max(inner.geoms, key=lambda g: g.area)

    coords = np.asarray(inner.exterior.coords)
    if coords.shape[0] >= 2 and np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]  # drop the duplicated closing point
    return coords


def seed_grid_on_plane(u_hat: np.ndarray, v_hat: np.ndarray, plane_point: np.ndarray,
                       projected_loop: np.ndarray, mean_seg: float,
                       args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Regular grid of candidate seed points covering the projected loop.

    Parameters
    ----------
    u_hat, v_hat : np.ndarray, shape (3,)
        Orthonormal basis of the cap plane.
    plane_point : np.ndarray, shape (3,)
        Origin of the (u, v) frame, on the plane.
    projected_loop : np.ndarray, shape (N, 3)
        Loop vertices projected on the plane.
    mean_seg : float
        Mean loop edge length.
    args : argparse.Namespace
        CLI arguments; the grid spacing is ``mean_seg * args.seedStepFactor``.

    Returns
    -------
    poly_uv : np.ndarray, shape (N, 2)
        Loop vertices in (u, v).
    uu, vv : np.ndarray
        Grid coordinates (``np.meshgrid`` output).
    grid_points : np.ndarray, shape (M, 3)
        3D positions of the grid points, on the plane.
    """
    rel = projected_loop - plane_point
    u_loop, v_loop = rel @ u_hat, rel @ v_hat

    delta = float(mean_seg) * args.seedStepFactor
    u_vals = np.arange(u_loop.min(), u_loop.max() + 0.5 * delta, delta)
    v_vals = np.arange(v_loop.min(), v_loop.max() + 0.5 * delta, delta)
    uu, vv = np.meshgrid(u_vals, v_vals, indexing="xy")

    grid_points = (plane_point + uu[..., None] * u_hat + vv[..., None] * v_hat).reshape(-1, 3)
    return np.column_stack([u_loop, v_loop]), uu, vv, grid_points


def keep_seeds_inside_loop(poly_uv: np.ndarray, grid_uv: np.ndarray, grid_points: np.ndarray,
                           delta: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep the seed points inside the loop polygon shrunk by `delta`.

    Parameters
    ----------
    poly_uv : np.ndarray, shape (N, 2)
        Loop polygon in (u, v).
    grid_uv : np.ndarray, shape (M, 2)
        Candidate points in (u, v).
    grid_points : np.ndarray, shape (M, k)
        Candidate points as returned (same order as `grid_uv`).
    delta : float
        Minimal distance kept between a seed and the loop.

    Returns
    -------
    seeds : np.ndarray, shape (K, k)
        Points of `grid_points` inside the shrunk polygon.
    inner_poly_uv : np.ndarray, shape (P, 2)
        Shrunk polygon.
    """
    inner_poly_uv = offset_polygon_inner(poly_uv, delta)
    # closed=True would make matplotlib ignore the last vertex; contains_points closes the path anyway
    inside = MplPath(inner_poly_uv).contains_points(grid_uv)
    return grid_points[inside], inner_poly_uv


def keep_seeds_far_from_loop(poly_uv: np.ndarray, seeds: np.ndarray, seeds_uv: np.ndarray, mean_seg: float,
                             args: argparse.Namespace) -> np.ndarray:
    """
    Keep the seed points whose distance to the loop edges is at least ``mean_seg * args.seedDecimator``.

    For each seed, the distance to every loop segment is computed in the
    (u, v) frame and the minimum is kept.

    Parameters
    ----------
    poly_uv : np.ndarray, shape (N, 2)
        Loop polygon in (u, v).
    seeds : np.ndarray, shape (M, 3)
        Candidate seeds (3D), already inside the polygon.
    seeds_uv : np.ndarray, shape (M, 2)
        Same seeds in (u, v).
    mean_seg : float
        Mean loop edge length.
    args : argparse.Namespace
        CLI arguments; the minimal distance is ``mean_seg * args.seedDecimator``.

    Returns
    -------
    seeds : np.ndarray, shape (K, 3)
        Seeds far enough from the loop.

    Notes
    -----
    Not used at the moment: `keep_seeds_inside_loop` keeps the seeds inside
    an inward offset of the polygon, which gives the same margin from the
    loop. Formerly named ``keepSeededPointsFarFromTheEdges``.

    Warning: the projection parameter is computed as ``t = -dot / |V|^2``
    while the projection of P on segment [A, A + V] is ``t = dot / |V|^2``
    (with ``dot = (P - A) . V``). With the minus sign, `t` is mostly clamped
    to 0 and the distance measured is often the distance to the segment
    start rather than to the segment. Check this before using the function.
    """
    # Loop segments [a, a + v] in (u, v)
    a = poly_uv
    v = np.roll(poly_uv, -1, axis=0) - a
    seg_len2 = np.sum(v ** 2, axis=1)

    # Distance of every seed to every segment: (M, 1, 2) against (1, N, 2)
    p = seeds_uv[:, None, :]
    dot = np.sum((p - a[None, :, :]) * v[None, :, :], axis=2)
    t = np.clip(-dot / (seg_len2[None, :] + 1e-12), 0.0, 1.0)  # see the warning above about the sign
    projection = a[None, :, :] + t[..., None] * v[None, :, :]
    min_dist = np.sqrt(np.min(np.sum((p - projection) ** 2, axis=2), axis=1))

    return seeds[min_dist >= mean_seg * args.seedDecimator]


def triangulate_planar_cap(projected_loop: np.ndarray, all_points: np.ndarray, plane_point: np.ndarray,
                           u_hat: np.ndarray, v_hat: np.ndarray, args: argparse.Namespace) -> pv.PolyData:
    """
    Constrained Delaunay triangulation of the loop polygon and its seeds.

    The points are expressed in the (u, v) frame of the cap plane and
    triangulated with Triangle (J. R. Shewchuk), with the loop edges as
    constraints: every loop edge is kept, the triangles outside the polygon
    are removed (``p``), and no point is added on the loop (``Y``), so that
    the loop vertices keep their one-to-one match with the side band.

    VTK's Delaunay2D was used before; it triangulates in the global XY
    plane, drops points closer than its tolerance and can lose long
    constraint edges, each of which left holes in the cap.

    Parameters
    ----------
    projected_loop : np.ndarray, shape (N, 3)
        Loop vertices projected on the cap plane, in loop order.
    all_points : np.ndarray, shape (M, 3)
        Points to triangulate: the projected loop first, then the seeds.
    plane_point : np.ndarray, shape (3,)
        Origin of the (u, v) frame.
    u_hat, v_hat : np.ndarray, shape (3,)
        Orthonormal basis of the cap plane.
    args : argparse.Namespace
        CLI arguments (debug export).

    Returns
    -------
    cap : pv.PolyData
        Triangulated polygon, on the points of `all_points`.

    Raises
    ------
    ValueError
        If the triangulation had to add points.
    """
    n_loop = projected_loop.shape[0]
    rel = all_points - plane_point
    pts_uv = np.column_stack([rel @ u_hat, rel @ v_hat])
    segments = np.column_stack([np.arange(n_loop), (np.arange(n_loop) + 1) % n_loop])

    result = triangle.triangulate({"vertices": pts_uv, "segments": segments}, "pYQ")  # Q: quiet
    if len(result["vertices"]) != len(all_points):
        raise ValueError("Triangulation added points to the cap; cannot map it back to 3D.")

    cap = _faces_to_polydata(np.asarray(all_points, dtype=float), result["triangles"])
    if args.debug:
        _save(cap, "debug_delaunay.ply", args)
    return cap


def _orient_cap_faces_like_loop(cap: pv.PolyData, plane_point: np.ndarray, u_hat: np.ndarray,
                                v_hat: np.ndarray, poly_uv: np.ndarray) -> pv.PolyData:
    """
    Wind every planar cap triangle in the same direction as the loop polygon.

    Each triangle is checked individually: its signed area in the (u, v)
    frame must have the same sign as the loop polygon.

    Parameters
    ----------
    cap : pv.PolyData
        Planar triangulated cap.
    plane_point : np.ndarray, shape (3,)
        Origin of the (u, v) frame.
    u_hat, v_hat : np.ndarray, shape (3,)
        Orthonormal basis of the cap plane.
    poly_uv : np.ndarray, shape (N, 2)
        Loop vertices in (u, v), in loop order.

    Returns
    -------
    cap : pv.PolyData
        Cap whose boundary edges follow the loop order.
    """
    loop_sign = np.sign(_signed_area_uv(poly_uv))
    faces = cap.faces.reshape(-1, 4)[:, 1:].copy()
    rel = cap.points[faces] - plane_point
    tri_uv = np.stack([rel @ u_hat, rel @ v_hat], axis=-1)
    reversed_tri = np.sign(_signed_area_uv(tri_uv)) == -loop_sign
    faces[reversed_tri] = faces[reversed_tri][:, ::-1]
    return _faces_to_polydata(cap.points, faces)


def create_side_band(loop_points: np.ndarray, projected_loop: np.ndarray) -> pv.PolyData:
    """
    Band of triangles joining the 3D loop to its projection on the cap plane.

    Each quad between loop edge (i, j) and its projection (i+N, j+N) is split
    into the triangles [i, j, i+N] and [i+N, j, j+N], wound the same way.

    Parameters
    ----------
    loop_points : np.ndarray, shape (N, 3)
        Loop vertices, in loop order.
    projected_loop : np.ndarray, shape (N, 3)
        Their projection on the cap plane.

    Returns
    -------
    side_band : pv.PolyData
        Band on the points [loop, projected loop]. It traverses the 3D loop
        as i -> j and the projected loop as j+N -> i+N, so the planar cap
        must follow the loop order to be consistent with it.
    """
    n = loop_points.shape[0]
    i = np.arange(n)
    j = (i + 1) % n
    faces = np.vstack([np.column_stack([i, j, i + n]), np.column_stack([i + n, j, j + n])])
    faces = faces.reshape(2, n, 3).transpose(1, 0, 2).reshape(-1, 3)  # interleave the two triangles of each quad
    return _faces_to_polydata(np.vstack([loop_points, projected_loop]), faces)


def make_cap(mesh: pv.PolyData, loop_indices: np.ndarray, args: argparse.Namespace) -> pv.PolyData:
    """
    Build the cap closing one opening (planar, or curved if needed).

    1. Order the loop against the mesh winding.
    2. Fit its SVD plane; if the loop does not project on it as a simple
       polygon, return a curved cap (`make_curved_cap`) instead.
    3. Move the plane to the top of the loop, on the outer side of the mesh.
    4. Project the loop on the plane (straightening its small folds), seed a
       regular grid inside it.
    5. Triangulate the polygon and its seeds, wind the triangles like the loop.
    6. Join the 3D loop to the projected loop with a side band.

    Parameters
    ----------
    mesh : pv.PolyData
        Mesh containing the opening, oriented outward.
    loop_indices : np.ndarray
        Loop vertex indices into `mesh.points`.
    args : argparse.Namespace
        CLI arguments (seeding, debug exports).

    Returns
    -------
    cap : pv.PolyData
        Cap wound opposite to the mesh along the loop. Its first points are
        the loop vertices, in the order of `_orient_loop_against_mesh`, with
        their exact coordinates (relied upon by `_attach_caps`).
    """
    # 1) Loop ordered against the mesh winding (points kept in the mesh precision for the plane fit)
    loop_indices = _orient_loop_against_mesh(mesh, loop_indices)
    loop_points = mesh.points[loop_indices]
    mean_seg = perimeter_mean_length(loop_points)

    # 2) Plane on which the loop projects as a simple polygon
    plane = _find_projection_plane(loop_points)
    if plane is None:
        print(f"WARNING: opening of {len(loop_indices)} vertices has no valid projection plane, "
              f"closed with a curved cap instead of a planar one.")
        return make_curved_cap(mesh, loop_indices, args)
    plane_point, u_hat, v_hat, plane_normal = plane

    # 3) Plane at the top of the opening, on its outer side. The sign of the SVD normal is arbitrary:
    #    without the hint, the plane could land inside the mesh and the cap would cut through it.
    plane_point = place_plane_on_loop_top(plane_point, plane_normal, loop_points=loop_points,
                                          outward_hint=_outward_direction(mesh, loop_indices))

    # 4) Projected loop, small folds straightened, and seeds inside an inward offset of the polygon
    projected_loop = project_points_on_plane(loop_points, plane_point, plane_normal)
    rel = projected_loop - plane_point
    loop_uv = np.column_stack([rel @ u_hat, rel @ v_hat])
    if not Polygon(loop_uv).is_valid:
        loop_uv = untangle_projected_loop(loop_uv)
        projected_loop = plane_point + loop_uv[:, 0][:, None] * u_hat + loop_uv[:, 1][:, None] * v_hat
    poly_uv, uu, vv, grid_points = seed_grid_on_plane(u_hat, v_hat, plane_point, projected_loop, mean_seg, args)
    if args.debug:
        _save_points(pv.PolyData(grid_points), "debug_grid_pts.ply", args)
        _save_lines(projected_loop, "debug_projectedLoop.ply", args)
    seeds, inner_poly_uv = keep_seeds_inside_loop(poly_uv, np.column_stack([uu.ravel(), vv.ravel()]),
                                                  grid_points, mean_seg * args.seedDecimator)
    if args.debug:
        inner_poly = plane_point + inner_poly_uv[:, 0][..., None] * u_hat + inner_poly_uv[:, 1][..., None] * v_hat
        _save_lines(inner_poly, "debug_inner_poly.ply", args)
        _save_points(pv.PolyData(seeds), "debug_grid_pts_inside.ply", args)

    # 5) Constrained triangulation, wound like the loop
    planar = triangulate_planar_cap(projected_loop, np.vstack([projected_loop, seeds]),
                                    plane_point, u_hat, v_hat, args)
    planar = _orient_cap_faces_like_loop(planar, plane_point, u_hat, v_hat, poly_uv)

    # 6) Side band, assembled with the planar part by index on the points [3D loop, projected loop, seeds].
    #    A projected point is merged only with its own loop point, when they coincide.
    loop_points = np.asarray(loop_points, dtype=float)
    n_loop = len(loop_points)
    side_faces = create_side_band(loop_points, projected_loop).faces.reshape(-1, 4)[:, 1:]
    points = np.vstack([loop_points, np.asarray(planar.points, dtype=float)])
    faces = np.vstack([side_faces, planar.faces.reshape(-1, 4)[:, 1:] + n_loop])
    on_plane = np.where(np.linalg.norm(loop_points - projected_loop, axis=1) < ON_PLANE_TOLERANCE)[0]
    remap = np.arange(len(points))
    remap[on_plane + n_loop] = on_plane
    return _remove_degenerate_faces(_faces_to_polydata(points, remap[faces]))


# ============================================================================
# Curved and fan caps
# ============================================================================

def make_curved_cap(mesh: pv.PolyData, loop_indices: np.ndarray, args: argparse.Namespace) -> pv.PolyData:
    """
    Close an opening with a curved (harmonic) cap, when it has no valid projection plane.

    1. The loop is mapped onto the unit circle by arc length: a circle is
       always a simple polygon, so the 2D triangulation is always valid.
    2. Seeds are placed on a regular grid inside the loop polygon on the
       disk, away from its border, and the disk is triangulated. The disk is
       convex, so a plain Delaunay triangulation (Qhull) covers it exactly;
       VTK's Delaunay2D is not used, as it produces overlapping triangles
       when many points lie on the same circle.
    3. Loop vertices keep their exact position; interior vertices are placed
       by solving a Laplace equation (each vertex at the average of its
       neighbours), which gives a smooth membrane spanning the loop.

    Parameters
    ----------
    mesh : pv.PolyData
        Mesh containing the opening.
    loop_indices : np.ndarray
        Loop vertex indices into `mesh.points`.
    args : argparse.Namespace
        CLI arguments (seeding, debug export).

    Returns
    -------
    cap : pv.PolyData
        Cap wound opposite to the mesh along the loop, starting with the loop
        vertices in the order of `_orient_loop_against_mesh`.

    Raises
    ------
    ValueError
        If the disk triangulation skipped loop vertices.
    """
    loop_indices = _orient_loop_against_mesh(mesh, loop_indices)
    loop_points = np.asarray(mesh.points[loop_indices], dtype=float)
    n_loop = len(loop_indices)

    # 1) Loop on the unit circle, by arc length
    seg = np.linalg.norm(np.roll(loop_points, -1, axis=0) - loop_points, axis=1)
    angles = 2 * np.pi * np.concatenate([[0.0], np.cumsum(seg)[:-1]]) / seg.sum()
    disk_loop = np.column_stack([np.cos(angles), np.sin(angles)])

    # 2) Seeds spaced like the loop. The polygon is inscribed in the circle: a long loop segment gives
    #    a chord well inside it, so seeds are tested against the polygon itself, not the circle.
    mean_seg = 2 * np.pi / n_loop
    step = mean_seg * args.seedStepFactor
    ticks = np.arange(-1.0, 1.0 + step, step)
    uu, vv = np.meshgrid(ticks, ticks)
    grid = np.column_stack([uu.ravel(), vv.ravel()])
    seeds, _ = keep_seeds_inside_loop(disk_loop, grid, grid, mean_seg * args.seedDecimator)

    disk_pts = np.vstack([disk_loop, seeds])
    disk = _faces_to_polydata(np.column_stack([disk_pts, np.zeros(len(disk_pts))]), Delaunay(disk_pts).simplices)
    disk = _orient_cap_faces_like_loop(disk, np.zeros(3), np.array([1.0, 0.0, 0.0]),
                                       np.array([0.0, 1.0, 0.0]), disk_loop)
    faces = disk.faces.reshape(-1, 4)[:, 1:]

    used = np.zeros(len(disk_pts), dtype=bool)
    used[faces.ravel()] = True
    if not used[:n_loop].all():
        raise ValueError("Disk triangulation skipped loop vertices; cannot build the curved cap.")
    # Drop the seeds left out of the triangulation (they would make the system singular)
    faces = (np.cumsum(used) - 1)[faces]
    disk_pts = disk_pts[used]

    # 3) Harmonic interior positions with the loop fixed: L_ii x_i - sum_j x_j = 0
    n_pts = len(disk_pts)
    i_idx = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
    j_idx = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
    adjacency = coo_matrix((np.ones(len(i_idx)), (i_idx, j_idx)), shape=(n_pts, n_pts)).tocsr()
    adjacency.data[:] = 1.0  # an edge shared by two triangles counts once
    laplacian = diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency

    points_3d = np.zeros((n_pts, 3))
    points_3d[:n_loop] = loop_points
    if n_pts > n_loop:
        inner = slice(n_loop, n_pts)
        rhs = -laplacian[inner, :n_loop] @ loop_points
        solver = splu(laplacian[inner, inner].tocsc())
        points_3d[inner] = np.column_stack([solver.solve(rhs[:, k]) for k in range(3)])

    cap = _faces_to_polydata(points_3d, faces)
    if args.debug:
        _save(cap, "debug_curved_cap.ply", args)
    return cap


def make_fan_cap(mesh: pv.PolyData, loop_indices: np.ndarray) -> pv.PolyData:
    """
    Close a small opening with a fan of triangles around a new vertex at its centroid.

    Used for the tiny holes opened by the final repair passes: unlike the
    other caps, it never adds an edge between two loop vertices, which could
    recreate the non-manifold edge the pass has just removed.

    Parameters
    ----------
    mesh : pv.PolyData
        Mesh containing the opening.
    loop_indices : np.ndarray
        Loop vertex indices into `mesh.points`.

    Returns
    -------
    cap : pv.PolyData
        Cap wound opposite to the mesh along the loop, starting with the loop
        vertices in the order of `_orient_loop_against_mesh`.
    """
    loop_indices = _orient_loop_against_mesh(mesh, loop_indices)
    loop_points = np.asarray(mesh.points[loop_indices], dtype=float)
    n_loop = len(loop_indices)
    faces = np.column_stack([np.arange(n_loop), (np.arange(n_loop) + 1) % n_loop, np.full(n_loop, n_loop)])
    return _faces_to_polydata(np.vstack([loop_points, loop_points.mean(axis=0)]), faces)


def _make_repair_cap(mesh: pv.PolyData, loop_indices: np.ndarray, args: argparse.Namespace) -> pv.PolyData:
    """
    Cap for a hole opened by the final repair passes.

    The regular cap is used, unless it fails or links two loop vertices that
    are already linked by a mesh edge (it would recreate the non-manifold
    edge just removed); a fan cap is used instead.

    Parameters
    ----------
    mesh : pv.PolyData
        Mesh containing the hole.
    loop_indices : np.ndarray
        Loop vertex indices into `mesh.points`.
    args : argparse.Namespace
        CLI arguments.

    Returns
    -------
    cap : pv.PolyData
        Regular or fan cap.
    """
    try:
        cap = make_cap(mesh, loop_indices, args)
    except ValueError:
        return make_fan_cap(mesh, loop_indices)

    # Caps start with the loop vertices, in oriented loop order (see `_attach_caps`)
    loop = _orient_loop_against_mesh(mesh, np.asarray(loop_indices, dtype=int))
    position = {int(v): i for i, v in enumerate(loop)}
    cap_to_mesh = [int(loop[j]) if j < len(loop) else None for j in range(cap.n_points)]
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    near = faces[np.isin(faces, loop).any(axis=1)]
    mesh_edges = {tuple(sorted((int(f[k]), int(f[(k + 1) % 3])))) for f in near for k in range(3)}
    for f in cap.faces.reshape(-1, 4)[:, 1:]:
        for k in range(3):
            a, b = cap_to_mesh[f[k]], cap_to_mesh[f[(k + 1) % 3]]
            if (a is not None and b is not None and (min(a, b), max(a, b)) in mesh_edges
                    and (position[a] - position[b]) % len(loop) not in (1, len(loop) - 1)):
                return make_fan_cap(mesh, loop_indices)
    return cap


# ============================================================================
# Fragments of the closed mesh
# ============================================================================

def remove_inner_and_far_parts(mesh: pv.PolyData, far_distance: float = FAR_PART_DISTANCE) -> pv.PolyData:
    """
    Remove the parts floating inside the brain and the blobs far from it.

    The scans contain many small closed fragments. Those inside the brain
    (the largest part) are pieces of scan debris: depending on their
    orientation, their volume was subtracted from or added twice to the
    total. Those far from the brain (blobs) are not part of it either. The
    small fragments close to the outer surface are kept.

    Parameters
    ----------
    mesh : pv.PolyData
        Closed triangle mesh (each part closed).
    far_distance : float, optional
        Parts whose closest vertex is farther from the brain than this
        fraction of its bounding box diagonal are removed. Default is
        `FAR_PART_DISTANCE`.

    Returns
    -------
    mesh : pv.PolyData
        Mesh without inner and far parts.
    """
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    labels = _connected_parts(mesh)
    n_parts = labels.max() + 1
    if n_parts == 1:
        return mesh
    brain = np.bincount(labels).argmax()
    brain_mesh = _faces_to_polydata(mesh.points, faces[labels == brain]).clean(point_merging=False)
    brain_points = np.asarray(brain_mesh.points)
    diagonal = np.linalg.norm(brain_points.max(axis=0) - brain_points.min(axis=0))
    tree = cKDTree(brain_points)

    # A few vertices of each other part: distance to the brain, and inside test (majority)
    drop = np.zeros(n_parts, dtype=bool)
    samples, owner = [], []
    for part in range(n_parts):
        if part == brain:
            continue
        vertices = np.unique(faces[labels == part])
        if tree.query(mesh.points[vertices])[0].min() > far_distance * diagonal:
            drop[part] = True
            continue
        picked = vertices[np.linspace(0, len(vertices) - 1, min(5, len(vertices))).astype(int)]
        samples.append(mesh.points[picked])
        owner += [part] * len(picked)
    if samples:
        inside = pv.PolyData(np.vstack(samples)).select_interior_points(brain_mesh, check_surface=False)
        is_inside = np.asarray(inside.point_data["selected_points"]).astype(bool)
        owner = np.asarray(owner)
        for part in np.unique(owner):
            drop[part] = is_inside[owner == part].mean() > 0.5

    if not drop.any():
        return mesh
    return _faces_to_polydata(mesh.points, faces[~drop[labels]])


def orient_parts_outward(mesh: pv.PolyData) -> pv.PolyData:
    """
    Orient every closed part of the mesh outward, not only the mesh as a whole.

    Many small closed fragments of the scans are wound inward: they are seen
    as inverted spots and their volume is subtracted. Each part whose signed
    volume is clearly negative is flipped; flat parts (zero thickness, no
    inside) are left as they are.

    Parameters
    ----------
    mesh : pv.PolyData
        Closed triangle mesh.

    Returns
    -------
    mesh : pv.PolyData
        Mesh whose parts all have a positive volume.
    """
    faces = mesh.faces.reshape(-1, 4)[:, 1:].copy()
    labels = _connected_parts(mesh)
    tri = mesh.points[faces].astype(np.float64)
    volume = np.bincount(labels, weights=np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0)
    area = np.bincount(labels, weights=np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                                                      axis=1) / 2.0)
    inverted = (volume < 0) & (np.abs(volume) > FLAT_PART_RATIO * area ** 1.5)
    flip = inverted[labels]
    if not flip.any():
        return mesh
    faces[flip] = faces[flip][:, ::-1]
    return _faces_to_polydata(mesh.points, faces)


def clean_closed_parts(mesh: pv.PolyData) -> pv.PolyData:
    """
    Remove the inner and far parts, then orient the remaining parts outward.

    Parameters
    ----------
    mesh : pv.PolyData
        Closed triangle mesh.

    Returns
    -------
    mesh : pv.PolyData
        Cleaned mesh.
    """
    return orient_parts_outward(remove_inner_and_far_parts(mesh))


# ============================================================================
# Hole-closing pipeline
# ============================================================================

def _attach_caps(mesh: pv.PolyData, loops: list[np.ndarray], caps: list[pv.PolyData]) -> pv.PolyData:
    """
    Add caps to a mesh, gluing each one on the vertices of its own loop.

    Every cap starts with its loop vertices, in the order given by
    `_orient_loop_against_mesh`, with their exact coordinates; they are
    replaced by the mesh indices by position. This also works when a loop
    goes through two coincident copies of a vertex (duplicated by
    `split_non_manifold_edges`), which merging coincident points would join.
    A cap that does not start with its loop is matched by coordinates,
    against that loop only.

    Parameters
    ----------
    mesh : pv.PolyData
        Mesh with openings.
    loops : list of np.ndarray
        Loop vertex indices into `mesh.points`, one per cap.
    caps : list of pv.PolyData
        Cap built for each loop.

    Returns
    -------
    mesh : pv.PolyData
        Mesh and caps in a single surface, without degenerate triangles.
    """
    points = [np.asarray(mesh.points, dtype=float)]
    faces = [mesh.faces.reshape(-1, 4)[:, 1:]]
    n_total = mesh.n_points
    for loop_indices, cap in zip(loops, caps, strict=True):
        loop = _orient_loop_against_mesh(mesh, np.asarray(loop_indices, dtype=int))
        n_loop = len(loop)
        loop_points = np.asarray(mesh.points[loop], dtype=float)
        cap_points = np.asarray(cap.points, dtype=float)
        index = np.empty(len(cap_points), dtype=np.int64)
        if len(cap_points) >= n_loop and np.array_equal(cap_points[:n_loop], loop_points):
            index[:n_loop] = loop
            rest = np.arange(n_loop, len(cap_points))
            index[rest] = n_total + np.arange(len(rest))
            new = cap_points[rest]
        else:
            lookup = {tuple(p): int(i) for p, i in zip(loop_points, loop, strict=True)}
            new = []
            for j, p in enumerate(cap_points):
                i = lookup.get(tuple(p))
                if i is None:
                    index[j] = n_total + len(new)
                    new.append(p)
                else:
                    index[j] = i
            new = np.array(new).reshape(-1, 3)
        points.append(new)
        n_total += len(new)
        faces.append(index[cap.faces.reshape(-1, 4)[:, 1:]])
    return _remove_degenerate_faces(_faces_to_polydata(np.vstack(points), np.vstack(faces)))


def _final_repair_passes(mesh: pv.PolyData, args: argparse.Namespace) -> pv.PolyData:
    """
    Repair what the input repair had to leave, once the caps are attached.

    - The non-manifold edges left on purpose (both ends on a boundary) are
      split: their vertices are no longer on a boundary.
    - On the non-manifold edges that still cannot be split, only the best
      pair of faces is kept (`drop_extra_faces_on_non_manifold_edges`).
    - The small holes this opens, and those left by open boundary chains,
      are capped (`_make_repair_cap`).

    Parameters
    ----------
    mesh : pv.PolyData
        Capped mesh.
    args : argparse.Namespace
        CLI arguments.

    Returns
    -------
    mesh : pv.PolyData
        Repaired mesh, oriented outward.
    """
    mesh = orient_outward(make_orientation_consistent(split_non_manifold_edges(mesh)))
    for _ in range(FINAL_REPAIR_PASSES):
        if np.all(_edge_incidence(mesh.faces.reshape(-1, 4)[:, 1:], mesh.n_points)[3] == 2):
            break  # every edge shared by exactly two faces: closed and manifold
        mesh = orient_outward(make_orientation_consistent(drop_extra_faces_on_non_manifold_edges(mesh)))
        loops = [loop for loop in extract_boundary_loops(mesh) if loop.shape[0] > 2]
        if not loops:
            break
        caps = [_make_repair_cap(mesh, loop, args) for loop in loops]
        mesh = orient_outward(make_orientation_consistent(_attach_caps(mesh, loops, caps)))
    return mesh


def create_caps(args: argparse.Namespace) -> pv.PolyData:
    """
    Close all the openings of the input mesh.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments (see `build_argparser`), plus ``debug_loop`` (sub-folder
        of the debug exports, set while processing each loop).

    Returns
    -------
    mesh : pv.PolyData
        The closed mesh with normals, or, with ``--capsOnly``, the cap of the
        last opening (empty if the mesh has no opening).
    """
    # 1) Load, optionally collapse very short edges (noisy boundaries)
    base = pv.read(str(Path(args.input).resolve())).triangulate().clean()
    if args.meshFilter:
        base = collapse_short_edges(base, tol=args.meshFilter)
        if args.debug:
            _save(base, "debug_meshFiltered.ply", args)

    # 2) Repair the input mesh
    base = repair_input_mesh(base)

    # 3) Openings
    loops = extract_boundary_loops(base)
    if not loops:
        if args.capsOnly:
            return pv.PolyData()
        return clean_closed_parts(orient_outward(base)).compute_normals(consistent_normals=False, inplace=False)

    # 4) One cap per opening
    caps, cap_loops = [], []
    for number, loop in enumerate((loop for loop in loops if loop.shape[0] > 2), start=1):
        args.debug_loop = f"loop{number}"
        if args.debug:
            _save_lines(base.points[loop], "debug_opening_loop.ply", args)
        caps.append(make_cap(base, loop, args))
        cap_loops.append(loop)
    args.debug_loop = ""
    if args.capsOnly:
        return caps[-1]

    # 5) Attach the caps, run the final repair passes, remove the inner and far parts and orient each part
    #    outward. Caps are wound against the mesh by construction, so VTK must not reorder faces when
    #    computing normals (it would cross non-manifold edges).
    out = clean_closed_parts(_final_repair_passes(_attach_caps(base, cap_loops, caps), args))
    return out.compute_normals(consistent_normals=False, inplace=False)


def close_mesh_holes(args: argparse.Namespace) -> tuple[bool, float | None]:
    """
    Close the openings of the input mesh, save the result and check it.

    With ``--capsOnly``, only the cap is saved, as ``<output>_caps.<ext>``,
    and watertightness is not evaluated. Otherwise, the closed mesh is saved
    to ``args.output``; it is watertight when it has no boundary edge and no
    non-manifold edge, and its volume is then computed.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments (see `build_argparser`).

    Returns
    -------
    watertight : bool
        Whether the closed mesh is watertight (False with ``--capsOnly``).
    volume : float or None
        Volume of the closed mesh if watertight, None otherwise.
    """
    output = Path(args.output).resolve()
    output.parent.mkdir(exist_ok=True)

    if args.capsOnly:
        create_caps(args).save(output.with_stem(output.stem + "_caps"))
        return False, None

    # Clean without merging coincident points: split non-manifold edges must stay apart
    out = create_caps(args).clean(point_merging=False, polys_to_lines=False)

    open_edges = out.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                           manifold_edges=False, non_manifold_edges=False)
    if open_edges.n_points and args.debug:
        _save_points(pv.PolyData(open_edges), "debug_remaining_edges.ply", args)

    # is_manifold: no boundary and no non-manifold edge
    watertight = bool(out.is_manifold and out.is_all_triangles and open_edges.n_points == 0)
    volume = out.volume if watertight else None
    out.save(str(output))
    return watertight, volume


# ============================================================================
# Command line
# ============================================================================

def build_argparser() -> argparse.ArgumentParser:
    """
    Build the command line parser.

    Returns
    -------
    parser : argparse.ArgumentParser
        Parser of the script options.
    """
    parser = argparse.ArgumentParser(
        description="Close the openings of a surface mesh and check that it is watertight.")
    parser.add_argument("-i", "--input", required=True, help="Input mesh (PLY/VTK/OBJ/STL, etc.).")
    parser.add_argument("-o", "--output", required=True, help="Output file (closed mesh, or cap with --capsOnly).")
    parser.add_argument("--capsOnly", action="store_true",
                        help="Save only the cap of the last opening instead of the closed mesh.")
    parser.add_argument("--debug", default=None, help="Folder where debug PLY/CSV files are exported.")
    parser.add_argument("--seedStepFactor", type=float, default=4.0,
                        help="Seed spacing, as a multiple of the mean loop edge length (default: 4.0).")
    parser.add_argument("--seedDecimator", type=float, default=4.0,
                        help="Minimal distance between seeds and the loop, as a multiple of the mean loop "
                             "edge length (default: 4.0).")
    parser.add_argument("--meshFilter", type=float,
                        help="Collapse edges shorter than this length before capping (e.g. 1e-5).")
    return parser


def main() -> None:
    """Parse the command line, close the mesh and print ``watertight=... volume=...``."""
    args = build_argparser().parse_args()
    args.debug_loop = ""
    watertight, volume = close_mesh_holes(args)
    if args.capsOnly:
        print("Cap surface saved (capsOnly=True).")
    else:
        print(f"watertight={watertight}  volume={volume if volume is not None else 'N/A'}")


if __name__ == "__main__":
    # Relative paths are resolved from this script's folder
    os.chdir(Path(__file__).resolve().parent)
    main()
