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
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh
import trimesh.visual
from matplotlib.path import Path as mplPath
from scipy.spatial import cKDTree
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

    This bloc uses PyVista's `extract_feature_edges` to obtain boundary edges, then
    reconstructs ordered loops by following adjacency. Each loop is returned as
    a list of point indices in the original mesh.

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
    edges = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False
    )
    if edges.n_points == 0:
        return []

    # Build 1D connectivity from segments in edges.lines
    # recover segments 2-verts with edges.lines: (n, 3) -> [2, i, j]
    lines = edges.lines.reshape(-1, 3)[:, 1:3]

    # Map: point -> neighbour
    nbrs = defaultdict(list)
    for a, b in lines:
        nbrs[a].append(b)
        nbrs[b].append(a)

    visited = set()
    loops = []

    for start in range(edges.n_points):
        if start in visited:
            continue
        if len(nbrs[start]) == 0:
            visited.add(start)
            continue

        # Walk around the boundary to build an ordered loop
        loop = [start]
        visited.add(start)
        cur = start
        prev = None

        # Progress as long as it didn't loop around or blocked
        while True:
            nxt_candidates = [x for x in nbrs[cur] if x != prev]
            if not nxt_candidates:
                break
            nxt = nxt_candidates[0]
            if nxt == start:
                # Returned to starting point => closed loop
                break
            if nxt in visited and nxt != start:
                # Shouldn't happen for a proper boundary; break to avoid infinite loops
                break
            loop.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt

        # Convert edge vertex indices back to original mesh vertices.
        # edges.points is a reordered subset; we need to return it
        # Correspondance to mesh.points => use a KDTree for projetection.
        # PyVista generaly conserve the topologie: edges.point_data["vtkOriginalPointIds"]
        ids = None
        # If available, use "vtkOriginalPointIds"; otherwise, fall back to KDTree.
        if "vtkOriginalPointIds" in edges.point_data:
            ids = edges.point_data["vtkOriginalPointIds"][loop]
        else:
            # Fallback (slower): position base research
            tree = cKDTree(mesh.points)
            _, ids = tree.query(edges.points[loop])

        loops.append(ids)

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
    poly_path = mplPath(inner_poly_uv, closed=True)
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


def runDelaunay2dOnConcavePolygon(projectedLoop, all_pts1, cloud, args: argparse.Namespace):
    """
    Run 2D constrained Delaunay triangulation inside a (possibly concave) polygon.

    Parameters
    ----------
    projectedLoop : np.ndarray, shape (N, 3)
        Projected loop vertices in 3D (lying in the cap plane).
    all_pts1 : np.ndarray, shape (M, 3)
        All points to consider for triangulation (loop + interior seeds).
    cloud : pv.PolyData
        Point cloud carrying `all_pts1`.
    args : argparse.Namespace
        CLI arguments, used for debug exports.

    Returns
    -------
    cap : pv.PolyData
        Triangulated cap filling the polygon.
    """
    # Build a closed polyline (edge source) for constrained triangulation
    N_loop = projectedLoop.shape[0]
    lines = np.empty((1, N_loop + 2), dtype=np.int64)
    lines[0, 0] = N_loop + 1
    lines[0, 1:-1] = np.arange(N_loop)
    lines[0, -1] = 0
    edge_poly = pv.PolyData(all_pts1, lines=lines)

    # Constrained 2D Delaunay triangulation
    cap = cloud.delaunay_2d(edge_source=edge_poly)
    cap = cap.triangulate()
    if args.debug:
        _save(cap, "debug_delaunay.ply", args)

    return cap


def deleteFacesOutsidePolygon(cap, planePoint, u_hat, v_hat, poly_uv, args: argparse.Namespace):
    """
    Remove triangles whose barycenter lies outside the original polygon.

    Parameters
    ----------
    cap : pv.PolyData
        Triangulated cap (in 3D).
    planePoint : np.ndarray, shape (3,)
        Reference point on the cap plane.
    u_hat, v_hat : np.ndarray, shape (3,)
        Orthogonal basis for the cap plane.
    poly_uv : np.ndarray, shape (N, 2)
        Polygon vertices in (u, v) of the original (non-offset) loop.
    args : argparse.Namespace
        CLI arguments (used for debug export).

    Returns
    -------
    cap_final : pv.PolyData
        Cap with triangles outside the polygon removed.
    """
    # Extract triangle connectivity and compute triangle centroids
    faces = cap.faces.reshape(-1, 4)[:, 1:]
    pts_cap = cap.points
    tri_pts = pts_cap[faces]
    tri_centers = tri_pts.mean(axis=1)

    # Project each triangle center onto the UV frame of the cap
    rel_c = tri_centers - planePoint
    tri_u = rel_c @ u_hat
    tri_v = rel_c @ v_hat
    tri_uv = np.column_stack([tri_u, tri_v])

    # Determine which triangles lie inside the polygon in 2D
    poly_path = mplPath(poly_uv, closed=True)
    inside_tri = poly_path.contains_points(tri_uv)
    faces_inside = faces[inside_tri]

    # Build a new PolyData containing only the inside faces
    if faces_inside.shape[0] == 0:
        cap_final = cap
    else:
        n_tri = faces_inside.shape[0]
        faces_flat = np.hstack(
            [np.full((n_tri, 1), 3, dtype=np.int64), faces_inside]
        ).ravel()
        cap_final = pv.PolyData(pts_cap, faces_flat)

    # Optional debug export of the filtered cap
    if args.debug:
        _save(cap_final, "debug_delaunay_concave.ply", args)

    return cap_final


def createSideBand(loopOpeningPoints, projectedLoop):
    """
    Create a side band of triangles between the 3D loop and its projection.

    For each edge (i, j) of the original 3D loop, this function creates two
    triangles connecting it with the corresponding edge in the projected loop:
        [i, j, i+N] and [i+N, j+N, j]

    Parameters
    ----------
    loopOpeningPoints : np.ndarray, shape (N, 3)
        3D loop vertices on the original mesh.
    projectedLoop : np.ndarray, shape (N, 3)
        Corresponding loop vertices projected onto the cap plane.

    Returns
    -------
    sideBand : pv.PolyData
        Triangulated side band mesh.
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
        # Second triangle: (a' → b' → b)
        faces.append([3, ap, bp, b])

    # Convert face list into PyVista format and create mesh
    faces = np.array(faces, dtype=np.int64).reshape(-1)
    sideBand = pv.PolyData(all_pts, faces)

    return sideBand


def make_cap(mesh: pv.PolyData, loop_indices, args: argparse.Namespace):
    """
    Generate a triangulated cap that closes a single opening on a 3D mesh.

    Detailed steps
    --------------
    1) Extract the boundary loop vertices.
    2) Estimate a best-fit plane via SVD.
    3) Translate this plane so that it touches the top of the opening.
    4) Project the loop vertices onto the plane.
    5) Generate a regular 2D grid of candidate seed points.
    6) Keep only seeds:
         - inside a shrunken version of the loop polygon,
         - sufficiently far from the border.
    7) Run constrained 2D Delaunay triangulation on all points
       (loop + seeds).
    8) Remove triangles whose barycenter is outside the loop polygon.
    9) Build a side band connecting the original 3D loop to its projection.
    10) Merge inner cap and side band into a single watertight surface.

    Parameters
    ----------
    mesh : pv.PolyData
        Base mesh containing the opening.
    loop_indices : np.ndarray
        Indices of the boundary loop vertices in `mesh.points`.
    args : argparse.Namespace
        CLI arguments controlling seeding and debug outputs.

    Returns
    -------
    cap_merged : pv.PolyData
        Final cap mesh with faces flipped (to match orientation).
    """

    # 1) Extract boundary loop points and mean segment length
    loopOpeningPoints = mesh.points[loop_indices]
    mean_seg = perimeter_mean_length(loopOpeningPoints)


    # 2) Estimate plane using SVD: c = center, (u, v) = in-plane basis, n = normal
    planePoint, u, v, planeNormal = fit_plane_svd(loopOpeningPoints)
    u_hat = u / (np.linalg.norm(u) + 1e-12)
    v_hat = v / (np.linalg.norm(v) + 1e-12)


    # 3) Slide the plane so that it aligns with the "top" of the opening
    planePoint = place_plane_on_loop_top(planePoint, planeNormal, loopOpeningPoints=loopOpeningPoints)

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
    cloud = pv.PolyData(all_pts)

    # 8) Constrained Delaunay in 2D
    cap = runDelaunay2dOnConcavePolygon(projectedLoop, all_pts, cloud, args)

    # 9) Remove triangles lying outside the polygon
    cap = deleteFacesOutsidePolygon(cap, planePoint, u_hat, v_hat, poly_uv, args)

    # 10) Build side band between 3D loop and its projection
    sideBand = createSideBand(loopOpeningPoints, projectedLoop)

    # 11) Merge cap interior and side band
    cap_merged = cap.merge(sideBand, merge_points=True, tolerance=1e-6)
    cap_merged = cap_merged

    # Flip faces for consistent orientation
    return cap_merged.flip_faces()


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

def create_caps(args):
    """
    Build caps for all boundary openings of an input mesh.

    Steps
    -----
    1) Load and clean the input mesh.
    2) Optionally collapse very short edges (meshFilter).
    3) Extract all boundary loops.
    4) For each loop, create a cap.
    5) Either:
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
        - The closed mesh with caps merged and normals recomputed.
    """
    # 1) Load the mesh and ensure a clean, triangulated representation
    input_path = Path(args.input).resolve()
    base = pv.read(str(input_path)).triangulate().clean()

    # 2) Optionally collapse small edges to improve loop detection and robustness of the cap creation. Useful when noisy boundaries
    if args.meshFilter:
        base = collapse_short_edges(base, tol=args.meshFilter)
        if args.debug:
            _save(base, "debug_meshFiltered.ply", args)

    # 3) Detect all open boundary loops. Each loop corresponds to a hole that must be closed with a cap.
    loops = extract_boundary_loops(base)

    if not loops:
        if args.capsOnly:
            # Return an empty mesh since there are no caps to extract.
            out_cap = pv.PolyData()
            return out_cap
        else:
            # No openings → ensure normals are consistent and return.
            return base.copy().compute_normals(auto_orient_normals=True, inplace=False)

    # 4) Build a cap for each boundary loop. Optionally export debug
    caps = []
    i = 1
    for loop in loops:
        if loop.shape[0] > 2:
            args.debug_loop = f"loop{i}"
            if args.debug:
                _save_lines(base.points[loop], "debug_opening_loop.ply", args)

            # Construct the cap for this loop and store it.
            cap = make_cap(base, loop, args)
            caps.append(cap)
            i += 1
    args.debug_loop = ""
    # In capsOnly mode, return the last cap (for Debugging / and backward compatibility).
    if args.capsOnly:
        return cap

    # 5) merge all generated caps into the base mesh to form a fully closed watertight surface, then recompute normals.
    out = base.copy()
    for cap in caps:
        out = out.merge(cap, merge_points=True)
    out = out.compute_normals(auto_orient_normals=True, inplace=False)

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
    out = out.clean(polys_to_lines=False)   # Clean mesh to ensure proper connectivity,

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
