#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
close_planar_hole.py
--------------------
Fill planar openings on a surface mesh by building a fan-like cap
and optionally seeding points on a regular grid in the cap plane.

Pipeline
-------------------
1. Read a triangulated surface mesh (e.g., an open brain surface).
2. Detect boundary loops (open edges) on the mesh.
3. For each boundary loop:
   - Fit an average plane using SVD on the loop vertices.
   - Slide the plane along its normal so that it coincides with the
     "top" of the opening (max along the normal).
   - Project the loop onto that plane.
   - Build a regular 2D grid (u, v) covering the projected loop.
   - Keep only grid points:
       * inside the loop polygon (supports concave loops),
       * and far enough from the border (to avoid noisy skinny triangles).
   - Run constrained 2D Delaunay triangulation inside the polygon.
   - Remove triangles whose barycenter lies outside the projected polygon.
   - Build a side band that connects the 3D loop to the projected loop.
   - Merge cap + side band into a watertight cap.
4. Merge all caps back into the base mesh (unless --capsOnly is used).
5. Optionally:
   - Collapse very short edges before capping (--meshFilter).
   - Export debug meshes (loops, grid, caps, etc.) in the debug directory.

Dependencies: numpy, pyvista (VTK), trimesh, shapely, matplotlib
"""

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyvista as pv
import triangle
import trimesh
import trimesh.visual
from matplotlib.path import Path as mplPath
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
from scipy.spatial import Delaunay
from shapely.geometry import Polygon

def _get_bool_attr_or_method(obj, name, default=True):
    """
    Safely query a boolean attribute or method on an object.

    If the requested attribute is callable, this function will attempt to call
    it with no arguments. Any TypeError is ignored and the raw attribute is
    used instead. The result is finally cast to bool.

    Parameters
    ----------
    obj : object
        Object on which we query the attribute or method.
    name : str
        Attribute or method name to look up.
    default : bool, optional
        Fallback value if the attribute does not exist. Default is True.

    Returns
    -------
    bool
        Boolean result of the attribute or method, or `default` if missing.
    """
    val = getattr(obj, name, default)
    if callable(val):
        try:
            val = val()
        except TypeError:
            pass
    return bool(val)


def _save_lines(points, fileName, args):
    """
    Save a closed polyline as a colored mesh (tube) for visualization/debugging.

    The function:
    - Builds a closed polyline from the given 3D points.
    - Converts it to a tubular mesh (small radius).
    - Exports it as a PLY mesh that can be opened in CloudCompare.

    Parameters
    ----------
    points : array-like, shape (N, 3)
        Points defining the polyline in 3D.
    fileName : str
        Filename (e.g. "debug_loop.ply") appended to args.debug.
    args : argparse.Namespace
        Parsed CLI arguments (needs args.debug to know where to save).
    """

    # lines : tableau (N, 3)
    pts = np.asarray(points, dtype=float)

    # 1) Create a closed polyline from the points
    poly = pv.PolyData(pts)
    n = pts.shape[0]

    # Closed polyline: [n+1, 0, 1, 2, ..., n-1, 0]
    lines = np.hstack(([n + 1], np.arange(n), 0)).astype(np.int64)
    poly.lines = lines

    # 2) Build a tube around this polyline for better visualization
    radius = 0.0001   # visual thickness of the line
    tube = poly.tube(radius=radius, n_sides=16)
    tube = tube.triangulate()

    # 3) Convert PyVista mesh -> trimesh
    verts = tube.points  # (Nv, 3)
    faces_vtk = tube.faces.reshape(-1, 4)[:, 1:]  # VTK format -> triangles (Nf, 3)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces_vtk, process=False)

    # 4) Assign a red color per vertex (RGBA)
    n_verts = verts.shape[0]
    # the file is not created if the number of vertices is 0
    if n_verts != 0:
        rgba = np.zeros((n_verts, 4), dtype=np.uint8)
        rgba[:, 0] = 255      # R
        rgba[:, 1] = 0        # G
        rgba[:, 2] = 0        # B
        rgba[:, 3] = 255      # A (opaque)

        mesh.visual.vertex_colors = rgba

        # 5) Export PLY (readable by CloudCompare)
        out_path = Path(args.debug).resolve() / args.debug_loop / fileName
        out_path.parent.mkdir(exist_ok=True)
        mesh.export(out_path)


def collapse_short_edges(mesh: pv.PolyData, tol: float = 1e-5) -> pv.PolyData:
    """
    Collapse (merge) edges shorter than `tol` and remove degenerate triangles.

    This is a simple topology-cleaning step that:
    - Detects edges whose length is below `tol`,
    - Collapses them by merging their endpoints into a single vertex at
      the midpoint,
    - Remaps triangles to the new vertex indices,
    - Removes degenerate triangles with repeated vertex indices.

    Parameters
    ----------
    mesh : pv.PolyData
        Triangulated mesh to be cleaned.
    tol : float, optional
        Maximum edge length to collapse. Default is 1e-5.

    Returns
    -------
    new_mesh : pv.PolyData
        Cleaned triangulated mesh.
    """
    mesh = mesh.triangulate().copy(deep=True)

    pts = mesh.points.copy()                           # (N_pts, 3)
    faces = mesh.faces.reshape(-1, 4)[:, 1:]           # (N_tri, 3)

    n_pts = pts.shape[0]

    # Disjoint-set / union-find structure to remap vertices after collapses.
    parent = np.arange(n_pts, dtype=np.int64)

    def find(x):
        """Find representative of x with path-compression."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        """
        Merge the sets of a and b.

        Chooses the smallest index as representative to keep indices compact.
        """
        ra, rb = find(a), find(b)
        if ra == rb:
            return ra
        # smallest index as representative to keep indices compact.
        if ra < rb:
            parent[rb] = ra
            return ra
        else:
            parent[ra] = rb
            return rb

    # 1) Detect edges shorter than the tolerance
    bad_edges = set()
    for tri in faces:
        i, j, k = tri
        for (a, b) in [(i, j), (j, k), (k, i)]:
            if a == b:
                continue
            pa = pts[a]
            pb = pts[b]
            if np.linalg.norm(pa - pb) < tol:
                # store edge as unordered pair
                e = tuple(sorted((a, b)))
                bad_edges.add(e)

    if not bad_edges:
        # No opening / Nothing to do
        return mesh

    # 2) Collapse small edges by merging vertices
    for (i, j) in bad_edges:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        # New vertex is midpoint of the two
        p_new = 0.5 * (pts[ri] + pts[rj])
        r = union(ri, rj)
        pts[r] = p_new

    # 3) Remap faces through the parent array
    for idx in range(n_pts):
        parent[idx] = find(idx)
    new_faces = parent[faces]

    # 4) Remove degenerate triangles with repeated indices
    mask_valid = np.logical_and.reduce([
        new_faces[:, 0] != new_faces[:, 1],
        new_faces[:, 1] != new_faces[:, 2],
        new_faces[:, 0] != new_faces[:, 2],
    ])
    new_faces = new_faces[mask_valid]

    if new_faces.shape[0] == 0:
        # Everything collapsed (extreme case)
        return pv.PolyData(pts)

    # 5) Rebuild faces in VTK format: [3, i, j, k, 3, ...]
    n_tri = new_faces.shape[0]
    faces_flat = np.hstack(
        [np.full((n_tri, 1), 3, dtype=np.int64), new_faces]
    ).ravel()

    new_mesh = pv.PolyData(pts, faces_flat)

    return new_mesh


def _faces_to_polydata(points: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    """Build a triangle PolyData from (N, 3) face indices."""
    faces_flat = np.hstack(
        [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]
    ).ravel()
    return pv.PolyData(points, faces_flat)


def remove_duplicate_faces(mesh: pv.PolyData) -> pv.PolyData:
    """
    Remove every triangle that appears more than once (in any winding).

    Scanned meshes sometimes contain pairs of identical triangles wound in
    opposite directions: a zero-thickness flap. Each of its edges is shared
    by three or more faces (non-manifold) and the two copies contradict each
    other's orientation. Removing all copies leaves the surrounding surface
    intact.

    Parameters
    ----------
    mesh : pv.PolyData
        Triangulated mesh.

    Returns
    -------
    mesh : pv.PolyData
        Mesh without duplicated triangles.
    """
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    _, inverse, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    unique_faces = counts[inverse.ravel()] == 1
    if unique_faces.all():
        return mesh
    return _faces_to_polydata(mesh.points, faces[unique_faces]).clean()


def _edge_incidence(faces: np.ndarray, n_points: int):
    """
    Group the face edges by unordered vertex pair.

    Returns
    -------
    key : np.ndarray, shape (3 * M,)
        Unordered edge key of every face edge (face f, edge k at row 3 * f + k).
    order : np.ndarray
        Row indices sorting `key`.
    start, counts : np.ndarray
        Start (in `order`) and number of rows of each distinct edge.
    """
    a = faces.ravel()
    b = np.roll(faces, -1, axis=1).ravel()
    key = np.minimum(a, b).astype(np.int64) * n_points + np.maximum(a, b)
    order = np.argsort(key, kind="stable")
    _, start, counts = np.unique(key[order], return_index=True, return_counts=True)
    return key, order, start, counts


def _manifold_patches(faces: np.ndarray, n_points: int) -> np.ndarray:
    """Patch id of every face, faces being connected through edges shared by exactly two faces."""
    _, order, start, counts = _edge_incidence(faces, n_points)
    first, second = order[start[counts == 2]] // 3, order[start[counts == 2] + 1] // 3
    graph = coo_matrix((np.ones(len(first)), (first, second)), shape=(len(faces), len(faces)))
    return connected_components(graph, directed=False)[1]


def remove_small_flaps(mesh: pv.PolyData, max_faces: int = 20) -> pv.PolyData:
    """
    Remove small patches only attached to the rest of the mesh by non-manifold edges.

    Scanned meshes contain small flaps (a few triangles) glued on the surface
    along an edge shared by three or more faces. A patch is the set of faces
    reachable through manifold edges; patches of at most `max_faces` faces
    that touch a non-manifold edge are removed. The surface they were glued on
    is left intact; at worst a small hole appears, which is capped later.

    Parameters
    ----------
    mesh : pv.PolyData
        Triangulated mesh.
    max_faces : int, optional
        Largest patch size considered a flap. Default is 20.

    Returns
    -------
    mesh : pv.PolyData
        Mesh without small flaps.
    """
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    _, order, start, counts = _edge_incidence(faces, mesh.n_points)
    nm = counts > 2
    if not nm.any():
        return mesh
    nm_faces = np.concatenate([order[s:s + c] // 3 for s, c in zip(start[nm], counts[nm])])

    patch = _manifold_patches(faces, mesh.n_points)
    patch_size = np.bincount(patch)
    flap_patches = np.unique(patch[nm_faces][patch_size[patch[nm_faces]] <= max_faces])
    if len(flap_patches) == 0:
        return mesh
    keep = ~np.isin(patch, flap_patches)
    return _faces_to_polydata(mesh.points, faces[keep])


def split_non_manifold_edges(mesh: pv.PolyData) -> pv.PolyData:
    """
    Separate the sheets of surface that meet along non-manifold edges.

    Around an edge shared by more than two faces, the faces are sorted by
    angle around the edge and paired with an angular neighbour (so that sheets
    do not cross), preferring pairs that traverse the edge in opposite
    directions (a consistently oriented sheet). Each pair then forms its own
    manifold sheet: every vertex of these edges is duplicated so that each
    group of faces connected around it (through manifold edges or paired
    faces) gets its own copy. Vertices on a boundary are kept shared, so an
    edge with both ends on a boundary stays non-manifold.

    No geometry is removed or moved: duplicated vertices share the same
    position. Coincident points must therefore not be merged afterwards
    (e.g. by `clean()`), or the sheets would be joined again.

    Parameters
    ----------
    mesh : pv.PolyData
        Triangulated mesh.

    Returns
    -------
    mesh : pv.PolyData
        Mesh whose non-manifold edges are split.
    """
    faces = mesh.faces.reshape(-1, 4)[:, 1:].copy()
    points = np.asarray(mesh.points)
    n_points = mesh.n_points
    key, order, start, counts = _edge_incidence(faces, n_points)
    nm = np.where(counts > 2)[0]
    if len(nm) == 0:
        return mesh

    # Pairs of faces allowed to stay connected through each non-manifold edge
    paired = []
    for e in nm:
        rows = order[start[e]:start[e] + counts[e]]
        fs = rows // 3
        a, b = divmod(int(key[rows[0]]), n_points)
        axis = points[b] - points[a]
        axis = axis / (np.linalg.norm(axis) + 1e-30)
        angles = []
        ref = ref2 = None
        for f in fs:
            apex = points[next(v for v in faces[f] if v != a and v != b)] - points[a]
            radial = apex - (apex @ axis) * axis
            if ref is None:
                ref = radial / (np.linalg.norm(radial) + 1e-30)
                ref2 = np.cross(axis, ref)
            angles.append(np.arctan2(radial @ ref2, radial @ ref))
        fan = fs[np.argsort(angles)]
        forward = {int(f): faces[f, (list(faces[f]).index(a) + 1) % 3] == b for f in fan}

        # Two ways to pair angular neighbours; keep the one with most consistently oriented pairs
        best = None
        for shift in (0, 1):
            ring = np.roll(fan, -shift)
            pairs = [(int(ring[i]), int(ring[i + 1])) for i in range(0, len(ring) - 1, 2)]
            score = sum(forward[f1] != forward[f2] for f1, f2 in pairs)
            if best is None or score > best[0]:
                best = (score, pairs)
        paired += [(f1, f2, a, b) for f1, f2 in best[1]]

    # Face adjacency through edges: manifold edges, plus the chosen pairs on non-manifold edges
    two = counts == 2
    first, second = order[start[two]] // 3, order[start[two] + 1] // 3
    shared = key[order[start[two]]]
    nm_vertices = np.unique(np.concatenate(divmod(key[order[start[nm]]], n_points)))
    # Boundary vertices are not split: two openings touching at such a vertex would be
    # joined into a single figure-of-eight loop and closed by one cap instead of two.
    # Splitting the other end is enough for the copies of the edge to be distinct.
    boundary_vertices = np.unique(np.concatenate(divmod(key[order[start[counts == 1]]], n_points)))
    nm_vertices = np.setdiff1d(nm_vertices, boundary_vertices)
    touches_nm = np.isin(faces, nm_vertices).any(axis=1)
    adjacency = defaultdict(list)
    for f1, f2, k in zip(first.tolist(), second.tolist(), shared.tolist()):
        if touches_nm[f1] or touches_nm[f2]:
            edge = divmod(k, n_points)
            adjacency[f1].append((f2, edge))
            adjacency[f2].append((f1, edge))
    for f1, f2, a, b in paired:
        adjacency[f1].append((f2, (a, b)))
        adjacency[f2].append((f1, (a, b)))

    # Split each vertex of a non-manifold edge into one copy per connected fan of faces around it
    vertex_faces = defaultdict(list)
    for f in np.where(touches_nm)[0]:
        for v in faces[f]:
            if v in nm_vertices:
                vertex_faces[int(v)].append(int(f))

    new_points = [points]
    next_index = n_points
    for v, fs in vertex_faces.items():
        fs_set = set(fs)
        root_of = {}
        for seed in fs:
            if seed in root_of:
                continue
            root_of[seed] = seed
            stack = [seed]
            while stack:
                f = stack.pop()
                for g, edge in adjacency[f]:
                    if v in edge and g in fs_set and g not in root_of:
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


def make_orientation_consistent(mesh: pv.PolyData) -> pv.PolyData:
    """
    Make neighbouring triangles wound consistently, without crossing non-manifold edges.

    Orientation is propagated face to face across manifold edges only (edges
    shared by exactly two faces). Within each patch reached this way, the
    winding held by the majority of its faces is kept, so a mostly correct
    surface is only touched where it was wrong.

    VTK's `consistent_normals` also walks through non-manifold edges, which
    can flip whole regions of a scanned mesh; this is why it is not used here.

    Parameters
    ----------
    mesh : pv.PolyData
        Triangulated mesh.

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

    # Manifold edges: exactly two incident faces
    order = np.argsort(key, kind="stable")
    _, start, counts = np.unique(key[order], return_index=True, return_counts=True)
    first = order[start[counts == 2]]
    second = order[start[counts == 2] + 1]
    f1, f2 = face_of_edge[first], face_of_edge[second]
    # Same direction on the shared edge means the two faces disagree
    disagree = forward[first] == forward[second]

    neighbours = defaultdict(list)
    for i, j, d in zip(f1.tolist(), f2.tolist(), disagree.tolist()):
        neighbours[i].append((j, d))
        neighbours[j].append((i, d))

    # Breadth-first walk of each patch, recording which faces must flip
    flip = np.zeros(n_faces, dtype=bool)
    visited = np.zeros(n_faces, dtype=bool)
    for seed in range(n_faces):
        if visited[seed]:
            continue
        visited[seed] = True
        patch = [seed]
        queue = [seed]
        while queue:
            current = queue.pop()
            for other, d in neighbours[current]:
                if not visited[other]:
                    visited[other] = True
                    flip[other] = flip[current] ^ d
                    patch.append(other)
                    queue.append(other)
        # Keep the majority winding of the patch
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
        Consistently oriented, closed triangulated mesh.

    Returns
    -------
    mesh : pv.PolyData
        Same mesh, wound so that normals point outward.
    """
    tri = mesh.points[mesh.faces.reshape(-1, 4)[:, 1:]].astype(np.float64)
    signed_volume = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0
    return mesh.flip_faces() if signed_volume < 0 else mesh


def offset_polygon_inner(poly_uv: np.ndarray, delta: float) -> np.ndarray:
    """
    Compute an inward-offset polygon at distance `delta` from the original polygon.

    The offset is computed in 2D (u, v) using Shapely's negative buffer.

    Parameters
    ----------
    poly_uv : np.ndarray, shape (N, 2)
        Vertices of the input polygon in 2D (u, v), ordered along the boundary.
        The polygon is implicitly closed (last vertex connected to the first).
    delta : float
        Positive inward offset distance. The offset polygon corresponds to
        `poly.buffer(-delta)`.

    Returns
    -------
    inner_poly : np.ndarray, shape (M, 2)
        Vertices of the inward-offset polygon. If the erosion completely
        removes the polygon, returns an empty array with shape (0, 2).
    """
    if poly_uv.shape[0] < 3:
        # Not enough points to define a polygon
        return np.zeros((0, 2), dtype=float)

    # Create Shapely polygon
    poly = Polygon(poly_uv)

    if not poly.is_valid:
        # Attempt simple repair by zero-distance buffer
        poly = poly.buffer(0.0)
        if poly.is_empty:
            return np.zeros((0, 2), dtype=float)

    # Inward offset: negative buffer distance
    inner = poly.buffer(-float(delta))

    if inner.is_empty:
        # Polygon has been completely eroded
        return np.zeros((0, 2), dtype=float)

    # In case of multiple disjoint components, keep the largest one
    if inner.geom_type == "MultiPolygon":
        # Keep the largest connected component by area
        biggest = max(inner.geoms, key=lambda g: g.area)
        inner = biggest

    # Extract exterior ring of the offset polygon
    exterior = inner.exterior
    coords = np.asarray(exterior.coords)  # shape (M+1, 2), last point = first

    # Drop duplicated closing point if present
    if coords.shape[0] >= 2 and np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]

    return coords

# --------------------------- Basic geometry helpers ---------------------------

def fit_plane_svd(points: np.ndarray):
    """
    Fit a best-fit plane using SVD.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
        3D points used to estimate the plane.

    Returns
    -------
    center : np.ndarray, shape (3,)
        Mean of the points (point on the plane).
    u : np.ndarray, shape (3,)
        First in-plane basis vector.
    v : np.ndarray, shape (3,)
        Second in-plane basis vector.
    n : np.ndarray, shape (3,)
        Normal vector of the plane.
    """
    c = points.mean(axis=0)
    P = points - c

    # SVD: right-singular vectors in vh
    _, _, vh = np.linalg.svd(P, full_matrices=False)
    n = vh[2, :]

    # In-plane orthonormal basis
    u = vh[0, :]
    v = vh[1, :]
    return c, u, v, n

def project_points_on_plane(points: np.ndarray,
                            center: np.ndarray,
                            normal: np.ndarray) -> np.ndarray:
    """
    Orthogonally project a set of 3D points onto a plane.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
        3D points to project.
    center : np.ndarray, shape (3,)
        Point on the plane (typically the barycenter).
    normal : np.ndarray, shape (3,)
        Normal vector of the plane.

    Returns
    -------
    projected : np.ndarray, shape (N, 3)
        Projected points lying on the plane.
    """
    n = normal / (np.linalg.norm(normal) + 1e-12)
    vec = points - center           # vector center→point
    dist = vec @ n                  # signed distances to plane (N,)
    projected = points - np.outer(dist, n) # equivalent to: projected = points - dist[:, None] * n
    return projected


def extract_boundary_loops(mesh: pv.PolyData):
    """
    Extract boundary loops (open edges) from a surface mesh.

    Boundary edges (used by a single triangle) are read from the mesh faces,
    then chained into ordered loops by following adjacency. Each loop is
    returned as a list of point indices in `mesh.points`.

    Parameters
    ----------
    mesh : pv.PolyData
        Input mesh.

    Returns
    -------
    loops : list of np.ndarray
        List of loops, each loop being an array of vertex indices into
        `mesh.points`, ordered along the boundary.
    """
    faces = mesh.faces.reshape(-1, 4)[:, 1:]

    # Boundary edges: unordered pairs used by exactly one triangle
    edge_count = Counter(
        tuple(sorted((int(f[k]), int(f[(k + 1) % 3])))) for f in faces for k in range(3)
    )
    nbrs = defaultdict(list)
    for (a, b), count in edge_count.items():
        if count == 1:
            nbrs[a].append(b)
            nbrs[b].append(a)

    # Walk the boundary edge by edge. A vertex shared by two openings (pinched
    # boundary) is visited twice: the sub-cycle closed at that point is cut out
    # as its own loop, so every loop is simple.
    # Non-manifold edges can leave a boundary as an open chain (odd-degree end
    # vertices). Walks start from those ends first, and a chain that cannot be
    # closed by real boundary edges is closed between its two ends.
    used = set()
    loops = []
    starts = sorted(nbrs, key=lambda v: len(nbrs[v]) % 2 == 0)
    for start in starts:
        for first in nbrs[start]:
            if tuple(sorted((start, first))) in used:
                continue
            path = [start]
            position = {start: 0}
            cur, nxt = start, first
            while True:
                used.add(tuple(sorted((cur, nxt))))
                if nxt in position:
                    # Closed a cycle: emit it and resume from where it started
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
                candidates = [w for w in nbrs[cur] if tuple(sorted((cur, w))) not in used]
                if not candidates:
                    # Open chain: close it between its ends
                    if len(path) > 2:
                        loops.append(np.array(path, dtype=int))
                    break
                nxt = candidates[0]

    return loops


# --------------------------- Perimeter measurements & helpers ---------------------------

def perimeter_mean_length(loop_pts: np.ndarray) -> float:
    """
    Compute the mean segment length of a closed loop.

    Parameters
    ----------
    loop_pts : np.ndarray, shape (N, 3)
        Loop vertices ordered along the boundary.

    Returns
    -------
    mean_length : float
        Mean length of the loop edges (including closing edge).
    """
    n = len(loop_pts)
    if n < 2:
        return 0.0
    segs = loop_pts[(np.arange(n) + 1) % n] - loop_pts
    lengths = np.linalg.norm(segs, axis=1)
    return float(lengths.mean())


def place_plane_on_loop_top(c: np.ndarray,
                            n: np.ndarray,
                            loopOpeningPoints: np.ndarray,
                            outward_hint: np.ndarray | None = None) -> np.ndarray:
    """
    Slide a plane so that it coincides with the "top" of the opening.

    Given a plane (c, n) and the 3D points of an opening loop, this function:
    - Optionally orients n using an outward_hint vector.
    - Finds the maximum projection of loop points onto the plane normal.
    - Translates the plane along n so that it touches the top of the loop.

    Parameters
    ----------
    c : np.ndarray, shape (3,)
        Initial point on the plane.
    n : np.ndarray, shape (3,)
        Initial plane normal.
    loopOpeningPoints : np.ndarray, shape (N, 3)
        Points forming the boundary loop of the opening.
    outward_hint : np.ndarray of shape (3,), optional
        Vector used to orient the plane normal consistently with
        the outside direction (if provided).

    Returns
    -------
    c_shift : np.ndarray, shape (3,)
        New plane center, shifted along n.
    """
    nn = n / (np.linalg.norm(n) + 1e-12)

    # Orient n towards "outside" if a hint is given
    if outward_hint is not None and np.dot(nn, outward_hint) < 0:
        nn = -nn

    d_target = float(np.max(loopOpeningPoints @ nn))     # top of the loop along n̂
    d_cur    = float(np.dot(c, nn))                      # current plane position
    c_shift  = c + (d_target - d_cur) * nn              # translated center on this face

    return c_shift


def _loop_edges(loop_indices: np.ndarray) -> set[tuple[int,int]]:
    """
    Helper to build a set of unoriented edges from a loop of indices.

    Parameters
    ----------
    loop_indices : np.ndarray
        Loop vertex indices.

    Returns
    -------
    edges : set of tuple[int, int]
        Unoriented edges (min(i, j), max(i, j)).
    """
    L = np.asarray(loop_indices, dtype=int)
    return set(tuple(sorted((L[i], L[(i+1) % len(L)]))) for i in range(len(L)))


def _orient_loop_against_mesh(mesh: pv.PolyData, loop_indices: np.ndarray) -> np.ndarray:
    """
    Order a boundary loop so that it runs opposite to the mesh winding.

    Each boundary edge belongs to exactly one mesh triangle, which traverses
    it in a given direction. A cap glued on that edge must traverse it the
    other way for the closed surface to be consistently oriented. The order
    returned by `extract_boundary_loops` is arbitrary, so it is fixed here
    once, and every cap part is then built following this order.

    Parameters
    ----------
    mesh : pv.PolyData
        Triangulated mesh containing the opening.
    loop_indices : np.ndarray
        Ordered loop vertex indices into `mesh.points`.

    Returns
    -------
    loop_indices : np.ndarray
        Same loop, reversed if needed so that each edge (L[i], L[i+1])
        is traversed by the mesh as L[i+1] -> L[i].
    """
    L = np.asarray(loop_indices, dtype=int)
    loop_edges = _loop_edges(L)

    # Directed boundary edges as traversed by the mesh triangles
    mesh_directed = set()
    for face in mesh.faces.reshape(-1, 4)[:, 1:]:
        for k in range(3):
            a, b = int(face[k]), int(face[(k + 1) % 3])
            if tuple(sorted((a, b))) in loop_edges:
                mesh_directed.add((a, b))

    same = opposite = 0
    for i in range(len(L)):
        a, b = int(L[i]), int(L[(i + 1) % len(L)])
        if (a, b) in mesh_directed and (b, a) not in mesh_directed:
            same += 1
        elif (b, a) in mesh_directed and (a, b) not in mesh_directed:
            opposite += 1

    if same == 0 and opposite == 0:
        raise ValueError("Cannot orient loop: no boundary edge found in the mesh faces.")
    # A few edges may disagree where the input mesh has local defects: follow the majority
    if same == opposite:
        raise ValueError(
            f"Cannot orient loop: mesh winding is ambiguous along the boundary "
            f"({same} edges each way)."
        )

    return L[::-1].copy() if same > opposite else L


def _orient_cap_faces_like_loop(cap: pv.PolyData, planePoint, u_hat, v_hat, poly_uv) -> pv.PolyData:
    """
    Wind every planar cap triangle in the same direction as the loop polygon.

    `delaunay_2d` gives no guarantee on the winding of its output relative to
    the loop order, so each triangle is checked individually: its signed area
    in the (u, v) frame must have the same sign as the loop polygon.

    Parameters
    ----------
    cap : pv.PolyData
        Planar triangulated cap.
    planePoint : np.ndarray, shape (3,)
        Reference point on the cap plane.
    u_hat, v_hat : np.ndarray, shape (3,)
        Orthogonal basis for the cap plane.
    poly_uv : np.ndarray, shape (N, 2)
        Loop vertices in (u, v), in loop order.

    Returns
    -------
    cap : pv.PolyData
        Cap whose boundary edges follow the loop order.
    """
    loop_sign = np.sign(_signed_area_uv(poly_uv))

    faces = cap.faces.reshape(-1, 4)[:, 1:].copy()
    rel = cap.points[faces] - planePoint
    tri_uv = np.stack([rel @ u_hat, rel @ v_hat], axis=-1)
    reversed_tri = np.sign(_signed_area_uv(tri_uv)) == -loop_sign
    faces[reversed_tri] = faces[reversed_tri][:, ::-1]

    faces_flat = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(cap.points, faces_flat)


def _remove_degenerate_faces(poly: pv.PolyData) -> pv.PolyData:
    """
    Drop triangles that reference the same vertex twice.

    They appear when a loop vertex already lies on the cap plane and gets
    merged with its projection; they would otherwise add a spurious edge
    traversed in both directions.
    """
    faces = poly.faces.reshape(-1, 4)[:, 1:]
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    faces_flat = np.hstack([np.full((int(keep.sum()), 1), 3, dtype=np.int64), faces[keep]]).ravel()
    return pv.PolyData(poly.points, faces_flat)

# --------------------------- Cap construction ---------------------------

def  seedPointsOnThePlane(u_hat, v_hat, planePoint, projectedLoop, mean_seg, args: argparse.Namespace):
    """
    Generate a regular 2D grid of sample points in the cap plane.

    The grid is aligned with the (u_hat, v_hat) basis and covers the bounding
    rectangle of the projected loop.

    Parameters
    ----------
    u_hat : np.ndarray, shape (3,)
        First unit basis vector of the cap plane.
    v_hat : np.ndarray, shape (3,)
        Second unit basis vector of the cap plane.
    planePoint : np.ndarray, shape (3,)
        Reference point on the plane.
    projectedLoop : np.ndarray, shape (N, 3)
        Loop vertices projected onto the cap plane.
    mean_seg : float
        Mean loop segment length (used to define the grid spacing).
    args : argparse.Namespace
        CLI arguments containing `seedStepFactor`.

    Returns
    -------
    poly_uv : np.ndarray, shape (N, 2)
        Loop vertices in (u, v) coordinates.
    UU, VV : np.ndarray
        2D arrays of u- and v-coordinates of the grid.
    grid_points : np.ndarray, shape (M, 3)
        3D positions of grid points in the cap plane.
    """
    # 2D coordinates of the perimeter in the (u, v) basis
    relativeProjectedLoopPoints = projectedLoop - planePoint
    uLoopCoords = relativeProjectedLoopPoints @ u_hat
    vLoopCoords = relativeProjectedLoopPoints @ v_hat

    # Bounding rectangle for the perimeter in (u, v)
    uLoopCoordMins, uLoopCoordMaxima = uLoopCoords.min(), uLoopCoords.max()
    vLoopCoordMins, vLoopCoordMaxima = vLoopCoords.min(), vLoopCoords.max()

    # Regular grid in the (u, v) plane
    delta = float(mean_seg) * args.seedStepFactor
    u_vals = np.arange(uLoopCoordMins, uLoopCoordMaxima + 0.5 * delta, delta)
    v_vals = np.arange(vLoopCoordMins, vLoopCoordMaxima + 0.5 * delta, delta)

    UU, VV = np.meshgrid(u_vals, v_vals, indexing="xy")

    # Convert grid to 3D positions in the cap plane)
    grid_points = planePoint + UU[..., None] * u_hat + VV[..., None] * v_hat
    grid_points = grid_points.reshape(-1, 3)

    poly_uv = np.column_stack([uLoopCoords, vLoopCoords])

    return poly_uv, UU, VV, grid_points


def keepSeededPointsInTheLoop(poly_uv, grid_uv, grid_points, delta):
    """
    Keep only seeded grid points that are inside an inward-offset version
    of the loop polygon.

    Parameters
    ----------
    poly_uv : np.ndarray, shape (N, 2)
        Loop polygon vertices in (u, v).
    grid_uv : np.ndarray, shape (M, 2)
        Grid sample coordinates in (u, v).
    grid_points : np.ndarray, shape (M, 3)
        Corresponding 3D positions of grid points.
    delta : float
        Inward offset distance used to shrink the polygon
        before testing the inclusion.

    Returns
    -------
    grid_pts_inside : np.ndarray, shape (K, 3)
        Grid points inside the shrunken polygon.
    inner_poly_uv : np.ndarray, shape (P, 2)
        Inward-offset polygon vertices.
    """
    inner_poly_uv = offset_polygon_inner(poly_uv, delta)
    # closed=True would make matplotlib ignore the last vertex; contains_points closes the path anyway
    poly_path = mplPath(inner_poly_uv)
    inside_mask = poly_path.contains_points(grid_uv)
    grid_pts_inside = grid_points[inside_mask]

    return grid_pts_inside, inner_poly_uv


def keepSeededPointsFarFromTheEdges(poly_uv, grid_pts_inside, grid_uv_inside, mean_seg, args: argparse.Namespace):
    """
    Filter out grid points that are too close to the loop edges.

    For each candidate grid point in 2D (u, v), we compute the minimal
    distance to the loop segments and keep only points whose distance is
    >= mean_seg * seedDecimator.

    Parameters
    ----------
    poly_uv : np.ndarray, shape (N, 2)
        Loop polygon vertices in (u, v).
    grid_pts_inside : np.ndarray, shape (M, 3)
        Candidate 3D grid points already inside the polygon.
    grid_uv_inside : np.ndarray, shape (M, 2)
        Corresponding (u, v) coordinates of grid_pts_inside.
    mean_seg : float
        Mean segment length along the loop.
    args : argparse.Namespace
        CLI arguments containing `seedDecimator`.

    Returns
    -------
    grid_pts_filtered : np.ndarray, shape (K, 3)
        Grid points that are sufficiently far from the loop boundary.
    """
    # Build perimeter segments in 2D
    A = poly_uv
    B = np.roll(poly_uv, -1, axis=0)
    V = B - A
    seg_len2 = np.sum(V**2, axis=1)

    # Evaluate point–segment distances for ALL grid points
    P = grid_uv_inside[:, None, :]      # (M,1,2)
    A_exp = A[None, :, :]               # (1,N,2)
    V_exp = V[None, :, :]               # (1,N,2)
    PA = P - A_exp
    dot = np.sum(PA * V_exp, axis=2)
    seg_len2_exp = seg_len2[None, :]

    # Projection factor t for projection onto segment line
    eps = 1e-12
    t = -dot / (seg_len2_exp + eps)
    t_clamped = np.clip(t, 0.0, 1.0)

    # Compute projection points and distances
    proj = A_exp + t_clamped[..., None] * V_exp
    diff_seg = P - proj
    dist2_seg = np.sum(diff_seg**2, axis=2)
    min_dist = np.sqrt(np.min(dist2_seg, axis=1))

    # Keep points that are far enough from the border
    mask_far = min_dist >= (mean_seg * args.seedDecimator)
    grid_pts_filtered = grid_pts_inside[mask_far]

    return grid_pts_filtered


def runDelaunay2dOnConcavePolygon(projectedLoop, all_pts1, planePoint, u_hat, v_hat, args: argparse.Namespace):
    """
    Run 2D constrained Delaunay triangulation inside a (possibly concave) polygon.

    The points are expressed in the (u, v) frame of the cap plane and
    triangulated with Triangle (J. R. Shewchuk), using the loop edges as
    constraints. Triangle recovers every constraint edge, removes the
    triangles outside the polygon itself, and is forbidden to insert points on
    the loop (-Y) so that loop vertices keep their one-to-one match with the
    side band.

    VTK's Delaunay2D was used before; it triangulates in the global XY plane,
    drops points closer than its tolerance, and can lose long constraint edges,
    each of which left holes in the cap.

    Parameters
    ----------
    projectedLoop : np.ndarray, shape (N, 3)
        Projected loop vertices in 3D (lying in the cap plane), in loop order.
    all_pts1 : np.ndarray, shape (M, 3)
        All points to consider for triangulation (loop first, then interior seeds).
    planePoint : np.ndarray, shape (3,)
        Reference point on the cap plane.
    u_hat, v_hat : np.ndarray, shape (3,)
        Orthogonal basis for the cap plane.
    args : argparse.Namespace
        CLI arguments, used for debug exports.

    Returns
    -------
    cap : pv.PolyData
        Triangulated cap filling the polygon, with the points of `all_pts1`.
    """
    n_loop = projectedLoop.shape[0]
    rel = all_pts1 - planePoint
    pts_uv = np.column_stack([rel @ u_hat, rel @ v_hat])
    segments = np.column_stack([np.arange(n_loop), (np.arange(n_loop) + 1) % n_loop])

    # p: constrained by the segments, triangles outside them removed; Y: no point added on them; Q: quiet
    result = triangle.triangulate({"vertices": pts_uv, "segments": segments}, "pYQ")
    if len(result["vertices"]) != len(all_pts1):
        raise ValueError("Triangulation added points to the cap; cannot map it back to 3D.")

    cap = _faces_to_polydata(np.asarray(all_pts1, dtype=float), result["triangles"])
    if args.debug:
        _save(cap, "debug_delaunay.ply", args)

    return cap


def _signed_area_uv(uv: np.ndarray) -> np.ndarray:
    """Signed area of polygon(s) given as (..., N, 2) arrays (positive = counter-clockwise)."""
    x, y = uv[..., 0], uv[..., 1]
    return 0.5 * np.sum(x * np.roll(y, -1, axis=-1) - np.roll(x, -1, axis=-1) * y, axis=-1)


def createSideBand(loopOpeningPoints, projectedLoop):
    """
    Create a side band of triangles between the 3D loop and its projection.

    For each edge (i, j) of the original 3D loop, this function creates two
    triangles connecting it with the corresponding edge in the projected loop:
        [i, j, i+N] and [i+N, j, j+N]

    Parameters
    ----------
    loopOpeningPoints : np.ndarray, shape (N, 3)
        3D loop vertices on the original mesh.
    projectedLoop : np.ndarray, shape (N, 3)
        Corresponding loop vertices projected onto the cap plane.

    Returns
    -------
    sideBand : pv.PolyData
        Triangulated side band mesh. It traverses the 3D loop as i -> j and
        the projected loop as j+N -> i+N, so the planar cap must follow the
        loop order to be consistent with it.
    """
    # Prepare full point list (first the 3D loop, then projection)
    N = loopOpeningPoints.shape[0]
    all_pts = np.vstack([loopOpeningPoints, projectedLoop])

    # Build triangles connecting each loop edge to its projected edge
    faces = []
    for i in range(N):
        j = (i + 1) % N
        a, b = i, j
        ap, bp = i + N, j + N
        # Triangle filling the quad: (a → b → a')
        faces.append([3, a, b, ap])
        # Second triangle: (a' → b → b'), same winding as the first one
        faces.append([3, ap, b, bp])

    # Convert face list into PyVista format and create mesh
    faces = np.array(faces, dtype=np.int64).reshape(-1)
    sideBand = pv.PolyData(all_pts, faces)

    return sideBand


def _find_projection_plane(loopOpeningPoints: np.ndarray):
    """
    Return the SVD best-fit plane if the loop projects on it to a simple polygon.

    A warped opening can project to a self-intersecting polygon, for which no
    valid planar cap exists. Tilted planes were tried once, but on large
    openings the cap then lies far from the loop and the side band adds or
    removes a large wedge of volume; such openings get a curved cap instead.

    Parameters
    ----------
    loopOpeningPoints : np.ndarray, shape (N, 3)
        Loop vertices, in loop order.

    Returns
    -------
    plane : tuple (center, u_hat, v_hat, normal) or None
        Unit in-plane basis and normal of the SVD plane, or None if the
        projected loop is not a simple polygon.
    """
    center, u, v, n = fit_plane_svd(loopOpeningPoints)
    rel = loopOpeningPoints - center
    polygon = Polygon(np.column_stack([rel @ u, rel @ v]))
    if polygon.is_valid and polygon.area > 0:
        return center, u, v, n
    return None


def _outward_direction(mesh: pv.PolyData, loop_indices) -> np.ndarray:
    """
    Mean outward normal of the mesh faces touching the loop.

    The mesh is expected to be oriented outward (see `orient_outward`); the
    returned vector then points to the outer side of the opening.
    """
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    tri = np.asarray(mesh.points, dtype=float)[faces[np.isin(faces, loop_indices).any(axis=1)]]
    return np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).sum(axis=0)


def make_curved_cap(mesh: pv.PolyData, loop_indices, args: argparse.Namespace):
    """
    Close an opening with a curved (harmonic) cap, for loops with no valid projection plane.

    1) The loop is mapped onto the unit circle, by arc length. A circle is
       always a simple polygon, so the 2D triangulation is always valid.
    2) Seeds are placed on a regular grid inside the disk, away from its
       border, and the disk is triangulated. The disk is convex, so a plain
       Delaunay triangulation covers it exactly; no boundary constraint is needed.
       Qhull (scipy) is used rather than VTK, whose Delaunay2D produces
       overlapping triangles when many points lie on the same circle.
    3) Loop vertices keep their exact 3D positions; interior vertices are
       placed by solving a Laplace equation (each vertex at the average of
       its neighbours), which gives a smooth membrane spanning the loop.

    Parameters
    ----------
    mesh : pv.PolyData
        Base mesh containing the opening.
    loop_indices : np.ndarray
        Indices of the boundary loop vertices in `mesh.points`.
    args : argparse.Namespace
        CLI arguments controlling seeding.

    Returns
    -------
    cap : pv.PolyData
        Curved cap, wound opposite to the mesh along the shared boundary.
    """
    loop_indices = _orient_loop_against_mesh(mesh, loop_indices)
    loopOpeningPoints = np.asarray(mesh.points[loop_indices], dtype=float)
    n_loop = len(loop_indices)

    # 1) Loop on the unit circle, by arc length
    seg = np.linalg.norm(np.roll(loopOpeningPoints, -1, axis=0) - loopOpeningPoints, axis=1)
    angles = 2 * np.pi * np.concatenate([[0.0], np.cumsum(seg)[:-1]]) / seg.sum()
    disk_loop = np.column_stack([np.cos(angles), np.sin(angles)])

    # 2) Seeds inside the loop polygon on the disk, spaced like the loop (mean segment length on the circle).
    #    The polygon is inscribed in the circle: a long loop segment gives a chord well inside it, so
    #    seeds are tested against the polygon itself, not the circle, or they would end up outside.
    mean_seg = 2 * np.pi / n_loop
    step = mean_seg * args.seedStepFactor
    margin = mean_seg * args.seedDecimator
    ticks = np.arange(-1.0, 1.0 + step, step)
    UU, VV = np.meshgrid(ticks, ticks)
    grid = np.column_stack([UU.ravel(), VV.ravel()])
    seeds, _ = keepSeededPointsInTheLoop(disk_loop, grid, grid, margin)

    disk_pts = np.vstack([disk_loop, seeds])
    disk = _faces_to_polydata(
        np.column_stack([disk_pts, np.zeros(len(disk_pts))]), Delaunay(disk_pts).simplices
    )
    disk = _orient_cap_faces_like_loop(
        disk, np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), disk_loop
    )
    faces = disk.faces.reshape(-1, 4)[:, 1:]

    used = np.zeros(len(disk_pts), dtype=bool)
    used[faces.ravel()] = True
    if not used[:n_loop].all():
        raise ValueError("Disk triangulation skipped loop vertices; cannot build the curved cap.")
    # Drop seeds left out of the triangulation (they would make the system singular)
    new_index = np.cumsum(used) - 1
    faces = new_index[faces]
    disk_pts = disk_pts[used]

    # 3) Harmonic interior positions with the loop fixed: L_ii x_i - sum_j x_j = 0
    n_pts = len(disk_pts)
    i_idx = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
    j_idx = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
    adjacency = coo_matrix((np.ones(len(i_idx)), (i_idx, j_idx)), shape=(n_pts, n_pts)).tocsr()
    adjacency.data[:] = 1.0  # duplicated edges count once
    laplacian = diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency

    points_3d = np.zeros((n_pts, 3))
    points_3d[:n_loop] = loopOpeningPoints
    if n_pts > n_loop:
        inner = slice(n_loop, n_pts)
        rhs = -laplacian[inner, :n_loop] @ loopOpeningPoints
        solver = splu(laplacian[inner, inner].tocsc())
        points_3d[inner] = np.column_stack([solver.solve(rhs[:, k]) for k in range(3)])

    cap = _faces_to_polydata(points_3d, faces)
    if args.debug:
        _save(cap, "debug_curved_cap.ply", args)
    return cap


def make_cap(mesh: pv.PolyData, loop_indices, args: argparse.Namespace):
    """
    Generate a triangulated cap that closes a single opening on a 3D mesh.

    Detailed steps
    --------------
    1) Order the loop against the mesh winding and extract its vertices.
    2) Estimate a best-fit plane via SVD; if the loop does not project to a
       simple polygon on it, fall back to a curved cap (`make_curved_cap`).
    3) Translate this plane so that it touches the top of the opening, on
       the outer side of the mesh.
    4) Project the loop vertices onto the plane.
    5) Generate a regular 2D grid of candidate seed points.
    6) Keep only seeds:
         - inside a shrunken version of the loop polygon,
         - sufficiently far from the border.
    7) Run constrained 2D Delaunay triangulation on all points
       (loop + seeds), restricted to the inside of the loop polygon.
    8) Wind the triangles in the loop order.
    9) Build a side band connecting the original 3D loop to its projection.
    10) Merge inner cap and side band into a single watertight surface.

    Parameters
    ----------
    mesh : pv.PolyData
        Base mesh containing the opening, oriented outward.
    loop_indices : np.ndarray
        Indices of the boundary loop vertices in `mesh.points`.
    args : argparse.Namespace
        CLI arguments controlling seeding and debug outputs.

    Returns
    -------
    cap_merged : pv.PolyData
        Final cap mesh, wound opposite to the mesh along the shared boundary.
    """

    # 1) Order the loop against the mesh winding, then extract its points and mean segment length
    loop_indices = _orient_loop_against_mesh(mesh, loop_indices)
    loopOpeningPoints = mesh.points[loop_indices]
    mean_seg = perimeter_mean_length(loopOpeningPoints)


    # 2) Estimate a plane on which the loop projects to a simple polygon: c = center, (u, v) = in-plane basis, n = normal
    plane = _find_projection_plane(loopOpeningPoints)
    if plane is None:
        # No planar cap is possible without overlaps: fall back to a curved cap
        print(f"WARNING: opening of {len(loop_indices)} vertices has no valid projection plane, "
              f"closed with a curved cap instead of a planar one.")
        return make_curved_cap(mesh, loop_indices, args)
    planePoint, u_hat, v_hat, planeNormal = plane


    # 3) Slide the plane so that it aligns with the "top" of the opening, on its outer side.
    #    The sign of the SVD normal is arbitrary: without the hint, the plane could land on the
    #    inner side and the cap would cut through the mesh, removing a slice of volume.
    planePoint = place_plane_on_loop_top(planePoint, planeNormal, loopOpeningPoints=loopOpeningPoints,
                                         outward_hint=_outward_direction(mesh, loop_indices))

    # 4) Project loop onto this plane
    projectedLoop = project_points_on_plane(loopOpeningPoints, planePoint, planeNormal)

    # 5) Seed points on the plane and build a grid
    poly_uv, UU, VV, grid_points = seedPointsOnThePlane(u_hat, v_hat, planePoint, projectedLoop, mean_seg, args)
    grid_uv = np.column_stack([UU.ravel(), VV.ravel()])
    if args.debug:
        _save_points(pv.PolyData(grid_points), "debug_grid_pts.ply", args)
        _save_lines(projectedLoop, "debug_projectedLoop.ply", args)


    # 6) Keep seeds inside an inward-offset version of the polygon
    grid_pts_inside, inner_poly_uv = keepSeededPointsInTheLoop(poly_uv, grid_uv, grid_points, mean_seg * args.seedDecimator)
    if args.debug:
        inner_poly = planePoint + inner_poly_uv[:, 0][..., None] * u_hat + inner_poly_uv[:, 1][..., None] * v_hat
        _save_lines(inner_poly, "debug_inner_poly.ply", args)
        _save_points(pv.PolyData(grid_pts_inside), "debug_grid_pts_inside.ply", args)

    # 7) Assemble points: projected loop + interior filtered points
    all_pts = np.vstack([projectedLoop, grid_pts_inside])

    # 8) Constrained Delaunay in 2D
    cap = runDelaunay2dOnConcavePolygon(projectedLoop, all_pts, planePoint, u_hat, v_hat, args)

    # 9) Wind the triangles in the loop order
    cap = _orient_cap_faces_like_loop(cap, planePoint, u_hat, v_hat, poly_uv)

    # 10) Build side band between 3D loop and its projection
    sideBand = createSideBand(loopOpeningPoints, projectedLoop)

    # 11) Merge cap interior and side band
    # The side band comes first so that merged points keep the exact mesh loop coordinates
    cap_merged = sideBand.merge(cap, merge_points=True, tolerance=1e-6)
    return _remove_degenerate_faces(cap_merged)


# --------------------------- IO helpers ---------------------------

def _save_points(points_pd: pv.PolyData, fileName, args: argparse.Namespace, sphere_radius: float = 0.0001):
    """
    Save a point cloud as a small-sphere mesh for visualization/debugging.

    Each point is replaced by a small triangulated sphere (glyph). The output
    is exported through trimesh, allowing colored meshes.

    Parameters
    ----------
    points_pd : pv.PolyData
        Input points (only `.points` are used).
    fileName : str
        Filename (e.g. "debug_loop.ply") appended to args.debug.
    args : argparse.Namespace
        Parsed CLI arguments (needs args.debug to know where to save).
    sphere_radius : float, optional
        Radius of the glyph spheres. Default is 0.0001.
    """
    sphere = pv.Sphere(
        radius=sphere_radius,
        phi_resolution=16,
        theta_resolution=16,
    )
    # Generate one sphere per point (glyphs)
    glyphs = points_pd.glyph(scale=False, orient=False, geom=sphere)

    glyphs = glyphs.triangulate()
    verts = glyphs.points  # (Nv, 3)
    faces_arr = glyphs.faces.reshape(-1, 4)   # [3, i0, i1, i2] par face
    faces = faces_arr[:, 1:]                  # (Nf, 3)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # White color per vertex (RGBA)
    n_verts = verts.shape[0]
    if n_verts != 0:
        rgba = np.full((n_verts, 4), [255, 255, 255, 255], dtype=np.uint8)
        mesh.visual.vertex_colors = rgba

            # 5) Export PLY (readable by CloudCompare)
        out_path = Path(args.debug).resolve() / args.debug_loop / fileName
        out_path.parent.mkdir(exist_ok=True)
        mesh.export(out_path)


def _save(poly_data: pv.PolyData, fileName, args: argparse.Namespace):
    """
    Save a PolyData object to a file, inferring format from the extension.

    Supported formats: .ply, .vtp, .vtk, .obj, .stl, .csv

    For .csv, only the point coordinates are saved.

    Parameters
    ----------
    poly_data : pv.PolyData
        Mesh to save.
    fileName : str
        Filename (e.g. "debug_loop.ply") appended to args.debug.
    args : argparse.Namespace
        Parsed CLI arguments (needs args.debug to know where to save).
    """
    path = Path(args.debug).resolve() / args.debug_loop / fileName
    path.parent.mkdir(exist_ok=True)
    ext = path.suffix.lower()
    if ext in {".ply", ".vtp", ".vtk", ".obj", ".stl"}:
        poly_data.save(path)
    elif ext == ".csv":
        xyz = poly_data.points if poly_data.n_points > 0 else np.empty((0,3))
        header = "x,y,z"
        np.savetxt(path, xyz, delimiter=",", header=header, comments="")
    else:
        # Default: save as PLY
        poly_data.save(path.with_suffix(".ply"))


# --------------------------- Hole-closing pipeline ---------------------------

def _attach_caps(mesh: pv.PolyData, loops, caps) -> pv.PolyData:
    """
    Add caps to the mesh, gluing each one on the vertices of its own loop.

    A cap's loop vertices carry the exact coordinates of the mesh loop
    vertices; they are matched against that loop only, by position, and
    replaced by the mesh indices. Merging all coincident points instead would
    also join the vertices duplicated by `split_non_manifold_edges`.

    Parameters
    ----------
    mesh : pv.PolyData
        Base mesh.
    loops : list of np.ndarray
        Loop vertex indices into `mesh.points`, one per cap.
    caps : list of pv.PolyData
        Caps built for these loops.

    Returns
    -------
    out : pv.PolyData
        Mesh and caps in a single surface.
    """
    points = [np.asarray(mesh.points, dtype=float)]
    faces = [mesh.faces.reshape(-1, 4)[:, 1:]]
    n_total = mesh.n_points
    for loop, cap in zip(loops, caps):
        loop = np.asarray(loop, dtype=int)
        lookup = {tuple(p): int(i) for p, i in zip(np.asarray(mesh.points[loop], dtype=float), loop)}
        cap_points = np.asarray(cap.points, dtype=float)
        index = np.empty(len(cap_points), dtype=np.int64)
        new = []
        for j, p in enumerate(cap_points):
            i = lookup.get(tuple(p))
            if i is None:
                index[j] = n_total + len(new)
                new.append(p)
            else:
                index[j] = i
        if new:
            points.append(np.array(new))
        n_total += len(new)
        faces.append(index[cap.faces.reshape(-1, 4)[:, 1:]])
    return _faces_to_polydata(np.vstack(points), np.vstack(faces))


def create_caps(args):
    """
    Build caps for all boundary openings of an input mesh.

    Steps
    -----
    1) Load and clean the input mesh.
    2) Optionally collapse very short edges (meshFilter).
    3) Remove duplicated triangles and make the winding consistent.
    4) Extract all boundary loops.
    5) For each loop, create a cap.
    6) Either:
       - Return only the union of caps (--capsOnly), or
       - Merge all caps into the original mesh (closed surface).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    out : pv.PolyData
        Either:
        - The union of caps in capsOnly mode, or
        - The closed mesh with caps merged, oriented outward and normals recomputed.
    """
    # 1) Load the mesh and ensure a clean, triangulated representation
    input_path = Path(args.input).resolve()
    base = pv.read(str(input_path)).triangulate().clean()

    # 2) Optionally collapse small edges to improve loop detection and robustness of the cap creation. Useful when noisy boundaries
    if args.meshFilter:
        base = collapse_short_edges(base, tol=args.meshFilter)
        if args.debug:
            _save(base, "debug_meshFiltered.ply", args)

    # 3) Repair the input so that each boundary edge has a well-defined winding the caps can follow:
    #    remove duplicated triangles and small flaps, separate sheets meeting on non-manifold edges,
    #    then propagate a consistent winding.
    base = remove_duplicate_faces(base)
    base = remove_small_flaps(base)
    base = split_non_manifold_edges(base)
    base = orient_outward(make_orientation_consistent(base))

    # 4) Detect all open boundary loops. Each loop corresponds to a hole that must be closed with a cap.
    loops = extract_boundary_loops(base)

    if not loops:
        if args.capsOnly:
            # Return an empty mesh since there are no caps to extract.
            out_cap = pv.PolyData()
            return out_cap
        else:
            # No openings → orient outward and return.
            return orient_outward(base).compute_normals(consistent_normals=False, inplace=False)

    # 5) Build a cap for each boundary loop. Optionally export debug
    caps = []
    cap_loops = []
    i = 1
    for loop in loops:
        if loop.shape[0] > 2:
            args.debug_loop = f"loop{i}"
            if args.debug:
                _save_lines(base.points[loop], "debug_opening_loop.ply", args)

            # Construct the cap for this loop and store it.
            cap = make_cap(base, loop, args)
            caps.append(cap)
            cap_loops.append(loop)
            i += 1
    args.debug_loop = ""
    # In capsOnly mode, return the last cap (for Debugging / and backward compatibility).
    if args.capsOnly:
        return cap

    # 6) merge all generated caps into the base mesh to form a fully closed watertight surface, then recompute normals.
    # Caps are wound against the mesh by construction, so VTK must not reorder faces: it would
    # propagate across non-manifold edges and flip whole regions. Only the global orientation is fixed.
    out = _attach_caps(base, cap_loops, caps)
    out = orient_outward(out).compute_normals(consistent_normals=False, inplace=False)

    return out


def close_mesh_holes(args):
    """
    High-level function: close (planar-ish) openings of a mesh and save results.

    Behavior
    --------
    - If --capsOnly is used:
        * Only the cap mesh is generated and saved as `<output>_caps.*`.
        * Returns (False, None) because watertightness is not evaluated.
    - Otherwise:
        * Caps are merged into the input mesh.
        * Remaining open edges are exported for debugging, if any.
        * Watertightness is evaluated (manifold, triangles only, no open edges).
        * Volume is computed only if the resulting mesh is watertight.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    watertight : bool
        True if the output mesh is watertight, False otherwise.
    vol : float or None
        Mesh volume if watertight, otherwise None.
    """
    # Case 1: Caps-only workflow
    Path(args.output).resolve().parent.mkdir(exist_ok=True)

    if args.capsOnly:
        out = create_caps(args)
        # Save result as <output>_caps.ext (matching input extension)
        filePath = Path(args.output).resolve()
        out.save(filePath.with_stem(filePath.stem + "_caps"))
        return False, None

    # Case 2: Full mesh closing workflow
    out = create_caps(args)
    # Clean without merging coincident points: split non-manifold edges must stay apart
    out = out.clean(point_merging=False, polys_to_lines=False)

    # Check watertightness (compat with various PyVista versions)
    edges = out.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                      manifold_edges=False, non_manifold_edges=False)
    n_open = edges.n_points
    if n_open and args.debug:
        _save_points(pv.PolyData(edges), "debug_remaining_edges.ply", args)

    # Evaluate watertightness based on: manifoldness, triangle-only faces and zero boundary edges.
    is_manifold = _get_bool_attr_or_method(out, "is_manifold", True)
    is_tris     = _get_bool_attr_or_method(out, "is_all_triangles", True)
    watertight  = is_manifold and is_tris and (n_open == 0)

    # Compute volume only for a fully watertight mesh
    vol = out.volume if watertight else None

    # Save final output mesh (capped or partially capped)
    out.save(str(Path(args.output).resolve()))

    return watertight, vol


# --------------------------- CLI ---------------------------

def build_argparser():
    """
    Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser for this script.
    """
    p = argparse.ArgumentParser(description="Close near-planar openings on a surface mesh with a fan-like cap and optional radial seeding.")
    p.add_argument("-i", "--input", required=True, help="Input mesh (PLY/VTK/OBJ/STL, etc.).")
    p.add_argument("-o", "--output", required=True, help="Output file (closed mesh or cap-only if --capsOnly).")
    p.add_argument("--capsOnly", action="store_true",
                   help="Save only the generated cap surface instead of the merged mesh.")
    p.add_argument("--debug", default=None,
                   help="Directory path where debug PLY/CSV files will be exported.")
    p.add_argument("--seedStepFactor", type=float, default=4.0,
                   help="Multiplicative factor applied to the mean perimeter edge length "
                        "to define the spacing between seed points (default: 4.0).")
    p.add_argument("--seedDecimator", type=float, default=4.0,
                   help="Multiplicative factor applied to the mean perimeter edge length "
                        "to define the minimal distance from the border kept for seeds "
                        "(default: 4.0).")
    p.add_argument("--meshFilter", type=float,
                   help="Collapse edges shorter than this tolerance before capping "
                        "(e.g., 1e-5).")
    return p


def main():
    """
    Parse CLI arguments, close mesh holes and print a short summary.

    In normal mode, prints:
        watertight=<True/False>  volume=<value or N/A>

    In capsOnly mode, prints a confirmation message that only the cap was saved.
    """
    args = build_argparser().parse_args()
    args.debug_loop = ""
    ok, vol = close_mesh_holes(args)

    if not args.capsOnly:
        print(f"watertight={ok}  volume={vol if vol is not None else 'N/A'}")
    else:
        print("Cap surface saved (capsOnly=True).")


if __name__ == "__main__":
    # Change current working directory to the script's directory
    script_dir = Path(__file__).resolve()
    os.chdir(script_dir.parent)

    main()
