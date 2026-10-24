"""
Unit tests of the hole-closing pipeline (`close_opening_scripts/close_planar_hole.py`).

Each test builds a small synthetic mesh (sphere, open hemisphere, tetrahedra)
reproducing a situation met on the BraDiPho meshes, and checks the property
the pipeline relies on: consistent orientation, no open edge, no
non-manifold edge, cap on the outer side of the opening, etc.

Run from the repository root:
    python -m unittest tests.test_close_planar_hole -v
"""

import argparse
import unittest
from collections import Counter

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from close_opening_scripts.close_planar_hole import (
    _find_projection_plane,
    _orient_loop_against_mesh,
    create_side_band,
    drop_extra_faces_on_non_manifold_edges,
    extract_boundary_loops,
    make_cap,
    make_curved_cap,
    make_fan_cap,
    make_orientation_consistent,
    orient_outward,
    remove_duplicate_faces,
    remove_small_flaps,
    split_non_manifold_edges,
    triangulate_planar_cap,
)

CAP_ARGS = argparse.Namespace(debug=None, debug_loop="", seedStepFactor=4.0, seedDecimator=4.0)


# ============================================================================
# Helpers
# ============================================================================

def mesh_from_faces(points: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    """Triangle mesh from points and (M, 3) face indices."""
    return pv.PolyData(points, np.hstack([np.full((len(faces), 1), 3), faces]).ravel())


def faces_of(mesh: pv.PolyData) -> np.ndarray:
    """(M, 3) face indices of a triangle mesh."""
    return mesh.faces.reshape(-1, 4)[:, 1:]


def edge_counts(mesh: pv.PolyData) -> Counter:
    """Number of faces sharing each unordered edge."""
    return Counter(tuple(sorted((int(f[k]), int(f[(k + 1) % 3])))) for f in faces_of(mesh) for k in range(3))


def directed_edge_conflicts(mesh: pv.PolyData) -> int:
    """Number of directed edges traversed twice in the same direction (orientation conflicts)."""
    counts = Counter((int(f[k]), int(f[(k + 1) % 3])) for f in faces_of(mesh) for k in range(3))
    return sum(1 for n in counts.values() if n > 1)


def open_edges(mesh: pv.PolyData) -> int:
    """Number of boundary edges."""
    return mesh.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                      manifold_edges=False, non_manifold_edges=False).n_cells


def signed_volume(mesh: pv.PolyData) -> float:
    """Signed volume (positive when the normals point outward)."""
    tri = mesh.points[faces_of(mesh)].astype(float)
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def sphere() -> pv.PolyData:
    """Closed triangulated sphere, oriented outward."""
    return pv.Sphere(theta_resolution=24, phi_resolution=24).triangulate().clean()


def open_hemisphere() -> pv.PolyData:
    """Sphere cut by a slightly tilted plane: one near-planar opening."""
    return pv.Sphere(radius=1.0, theta_resolution=48, phi_resolution=48).clip(
        normal=(0.1, 0.0, 1.0), invert=True).triangulate().clean()


def sphere_with_fin() -> tuple[pv.PolyData, np.ndarray]:
    """Sphere with a one-triangle fin glued on an edge (shared by three faces), and the sphere faces."""
    base = sphere()
    faces = faces_of(base)
    a, b = (int(v) for v in faces[0][:2])
    points = np.vstack([base.points, [[0.0, 0.0, 2.0]]])
    return mesh_from_faces(points, np.vstack([faces, [[b, a, len(points) - 1]]])), faces


def outward_tetrahedron(points: np.ndarray, ids: tuple[int, int, int, int]) -> np.ndarray:
    """Faces of the tetrahedron `ids`, wound outward."""
    faces = []
    centre = points[list(ids)].mean(axis=0)
    for k in range(4):
        f = [ids[j] for j in range(4) if j != k]
        n = np.cross(points[f[1]] - points[f[0]], points[f[2]] - points[f[0]])
        faces.append(f if n @ (points[f[0]] - centre) > 0 else f[::-1])
    return np.array(faces)


def two_tetrahedra_sharing_an_edge() -> pv.PolyData:
    """Two closed tetrahedra glued along edge 0-1 only: that edge is shared by four faces."""
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0.2], [0.5, 0.3, 1],
                       [0.5, -1, 0.2], [0.5, -0.3, -1]], dtype=float)
    faces = np.vstack([outward_tetrahedron(points, (0, 1, 2, 3)), outward_tetrahedron(points, (0, 1, 4, 5))])
    return mesh_from_faces(points, faces)


# ============================================================================
# Input repair
# ============================================================================

class InputRepairTests(unittest.TestCase):
    """Repairs applied to the input mesh before capping."""

    def setUp(self):
        self.sphere = sphere()
        self.faces = faces_of(self.sphere)

    def test_removes_opposite_wound_duplicate_flap(self):
        """Both copies of a triangle duplicated with the opposite winding are removed."""
        mesh = mesh_from_faces(self.sphere.points, np.vstack([self.faces, self.faces[0][::-1]]))
        repaired = remove_duplicate_faces(mesh)

        self.assertEqual(repaired.n_cells, self.sphere.n_cells - 1)
        self.assertEqual(directed_edge_conflicts(repaired), 0)

    def test_removes_small_flap_glued_on_an_edge(self):
        """A one-triangle fin glued on an edge is removed, the sphere is untouched."""
        mesh, faces = sphere_with_fin()
        np.testing.assert_array_equal(faces_of(remove_small_flaps(mesh)), faces)

    def test_restores_consistent_winding_with_majority_orientation(self):
        """A few flipped triangles are flipped back, keeping the majority winding."""
        faces = self.faces.copy()
        faces[:10] = faces[:10, ::-1]
        repaired = make_orientation_consistent(mesh_from_faces(self.sphere.points, faces))

        self.assertEqual(directed_edge_conflicts(repaired), 0)
        self.assertAlmostEqual(signed_volume(repaired), signed_volume(self.sphere))

    def test_does_not_propagate_through_non_manifold_edges(self):
        """The winding of the sphere is not changed by a fin glued on one of its edges."""
        mesh, faces = sphere_with_fin()
        np.testing.assert_array_equal(faces_of(make_orientation_consistent(mesh))[:-1], faces)

    def test_orient_outward_fixes_inward_mesh(self):
        """A mesh with inward normals is flipped."""
        inward = self.sphere.flip_faces()
        self.assertLess(signed_volume(inward), 0)
        self.assertGreater(signed_volume(orient_outward(inward)), 0)


class NonManifoldEdgeTests(unittest.TestCase):
    """Repair of the edges shared by more than two faces."""

    def test_separates_two_closed_sheets_sharing_an_edge(self):
        """Splitting gives two manifold tetrahedra, with the same volume."""
        mesh = two_tetrahedra_sharing_an_edge()
        split = split_non_manifold_edges(mesh)

        self.assertEqual(set(edge_counts(split).values()), {2})
        self.assertEqual(split.n_points, mesh.n_points + 2)
        self.assertEqual(directed_edge_conflicts(split), 0)
        self.assertAlmostEqual(signed_volume(split), signed_volume(mesh))

    def test_keeps_one_consistent_pair_on_an_edge_that_cannot_be_split(self):
        """Only two consistently wound faces are kept on the edge."""
        mesh = two_tetrahedra_sharing_an_edge()
        kept = drop_extra_faces_on_non_manifold_edges(mesh)

        self.assertEqual(kept.n_cells, mesh.n_cells - 2)
        self.assertEqual(edge_counts(kept)[(0, 1)], 2)
        self.assertEqual(directed_edge_conflicts(kept), 0)


# ============================================================================
# Boundary loops
# ============================================================================

class BoundaryLoopTests(unittest.TestCase):
    """Extraction and orientation of the boundary loops."""

    def test_splits_pinched_boundary_into_simple_loops(self):
        """Two openings touching at one vertex (bow-tie) give two loops."""
        points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [-1, 0, 0], [-1, -1, 0]], dtype=float)
        mesh = pv.PolyData(points, np.array([3, 0, 1, 2, 3, 0, 3, 4]))
        loops = extract_boundary_loops(mesh)

        self.assertEqual(sorted(sorted(loop.tolist()) for loop in loops), [[0, 1, 2], [0, 3, 4]])

    def test_every_boundary_edge_belongs_to_exactly_one_loop(self):
        """On a sphere with many holes, the loops cover each boundary edge once."""
        base = sphere()
        faces = faces_of(base)
        mesh = mesh_from_faces(base.points, faces[np.arange(len(faces)) % 7 != 0])
        boundary = {e for e, n in edge_counts(mesh).items() if n == 1}
        loop_edges = [tuple(sorted((int(loop[i]), int(loop[(i + 1) % len(loop)]))))
                      for loop in extract_boundary_loops(mesh) for i in range(len(loop))]

        self.assertEqual(len(loop_edges), len(set(loop_edges)))
        self.assertEqual(set(loop_edges), boundary)

    def test_reverses_loop_that_follows_mesh_winding(self):
        """A loop running like the mesh triangle is reversed."""
        mesh = pv.PolyData(np.eye(3), np.array([3, 0, 1, 2]))
        np.testing.assert_array_equal(_orient_loop_against_mesh(mesh, np.array([0, 1, 2])), [2, 1, 0])

    def test_keeps_loop_that_opposes_mesh_winding(self):
        """A loop already running against the mesh triangle is kept."""
        mesh = pv.PolyData(np.eye(3), np.array([3, 0, 1, 2]))
        np.testing.assert_array_equal(_orient_loop_against_mesh(mesh, np.array([2, 1, 0])), [2, 1, 0])

    def test_rejects_ambiguous_mesh_winding(self):
        """Two triangles of opposite windings give a loop with as many edges each way."""
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
        mesh = pv.PolyData(points, np.array([3, 0, 1, 2, 3, 1, 2, 3]))
        with self.assertRaises(ValueError):
            _orient_loop_against_mesh(mesh, np.array([0, 1, 3, 2]))


# ============================================================================
# Caps
# ============================================================================

class SideBandTests(unittest.TestCase):
    """Band joining the loop to its projection."""

    def test_side_band_is_consistently_wound(self):
        """The two triangles of every quad are wound the same way."""
        t = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        loop = np.c_[np.cos(t), np.sin(t), 0.1 * np.sin(3 * t)]
        projected = loop.copy()
        projected[:, 2] = 0.2
        self.assertEqual(directed_edge_conflicts(create_side_band(loop, projected)), 0)


class PlanarTriangulationTests(unittest.TestCase):
    """Constrained triangulation of the planar cap."""

    def test_keeps_long_constraint_edge_near_a_zigzag(self):
        """Regression: VTK lost the long edge 3 -> 4, which passes close to vertex 2."""
        loop_uv = np.array([[0, 0], [1, 0], [1.2, 0.05], [1.0, -0.01], [3, 0.4], [3, 2], [0, 2]], dtype=float)
        points = np.c_[loop_uv, np.zeros(len(loop_uv))]
        cap = triangulate_planar_cap(points, points, np.zeros(3), np.array([1.0, 0, 0]),
                                     np.array([0, 1.0, 0]), argparse.Namespace(debug=None))
        n = len(loop_uv)
        boundary = {e for e, c in edge_counts(cap).items() if c == 1}
        self.assertEqual(boundary, {tuple(sorted((i, (i + 1) % n))) for i in range(n)})


class ProjectionPlaneTests(unittest.TestCase):
    """Choice between a planar and a curved cap."""

    def test_flat_loop_uses_best_fit_plane(self):
        """A flat circle is capped on its own plane."""
        t = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        _, _, _, normal = _find_projection_plane(np.c_[np.cos(t), np.sin(t), np.zeros_like(t)])
        self.assertAlmostEqual(abs(normal[2]), 1.0)

    def test_knotted_loop_has_no_valid_plane(self):
        """A trefoil knot crosses itself in every planar projection."""
        t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        loop = np.c_[np.sin(t) + 2 * np.sin(2 * t), np.cos(t) - 2 * np.cos(2 * t), -np.sin(3 * t)]
        self.assertIsNone(_find_projection_plane(loop))


class PlaneSideTests(unittest.TestCase):
    """The planar cap lies on the outer side of the opening, whatever the sign of the SVD normal."""

    def test_cap_plane_is_outside_the_mesh(self):
        """On a dome with a wavy opening, every cap point is above the loop (outer side)."""
        ball = pv.Sphere(radius=1.0, theta_resolution=60, phi_resolution=60).triangulate().clean()
        centres = ball.cell_centers().points
        keep = centres[:, 2] < 0.5 + 0.15 * np.sin(3 * np.arctan2(centres[:, 1], centres[:, 0]))
        dome = ball.extract_cells(np.where(keep)[0]).extract_surface(algorithm=None).triangulate().clean()
        for upside_down in (False, True):
            with self.subTest(upside_down=upside_down):
                mesh = dome.rotate_x(180) if upside_down else dome
                outward = np.array([0.0, 0.0, -1.0 if upside_down else 1.0])
                (loop,) = extract_boundary_loops(mesh)
                cap = make_cap(mesh, loop, CAP_ARGS)

                loop_height = (mesh.points[loop] @ outward).max()
                is_loop = cKDTree(mesh.points[loop]).query(cap.points)[0] < 1e-9
                self.assertGreaterEqual((cap.points[~is_loop] @ outward).min(), loop_height - 1e-6)


class PlanarCapTests(unittest.TestCase):
    """A cap closes the opening without orientation conflict, whatever the mesh winding and loop order."""

    def build_cap(self, mesh: pv.PolyData, loop: np.ndarray) -> pv.PolyData:
        """Cap under test."""
        return make_cap(mesh, loop, CAP_ARGS)

    def test_all_winding_and_loop_order_combinations(self):
        """The open hemisphere is closed for the four combinations of winding and loop order."""
        for flip_mesh in (False, True):
            for reverse_loop in (False, True):
                with self.subTest(flip_mesh=flip_mesh, reverse_loop=reverse_loop):
                    mesh = open_hemisphere()
                    if flip_mesh:
                        mesh = mesh.flip_faces()
                    (loop,) = extract_boundary_loops(mesh)
                    if reverse_loop:
                        loop = loop[::-1]
                    cap = self.build_cap(mesh, loop)
                    self.assertEqual(directed_edge_conflicts(cap), 0)

                    closed = mesh.merge(cap, merge_points=True).clean(polys_to_lines=False)
                    self.assertEqual(open_edges(closed), 0)
                    self.assertEqual(directed_edge_conflicts(closed), 0)


class CurvedCapTests(PlanarCapTests):
    """Same guarantees for the curved (harmonic) cap."""

    def build_cap(self, mesh: pv.PolyData, loop: np.ndarray) -> pv.PolyData:
        """Cap under test."""
        return make_curved_cap(mesh, loop, CAP_ARGS)


class FanCapTests(PlanarCapTests):
    """Same guarantees for the fan cap of the final repair passes."""

    def build_cap(self, mesh: pv.PolyData, loop: np.ndarray) -> pv.PolyData:
        """Cap under test."""
        return make_fan_cap(mesh, loop)


if __name__ == "__main__":
    unittest.main()
