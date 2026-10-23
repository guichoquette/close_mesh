import argparse
import unittest
from collections import Counter

import numpy as np
import pyvista as pv

from close_opening_scripts.close_planar_hole import (
    _orient_loop_against_mesh,
    createSideBand,
    deleteFacesOutsidePolygon,
    extract_boundary_loops,
    make_cap,
    make_orientation_consistent,
    orient_outward,
    remove_duplicate_faces,
)


def directed_edge_conflicts(poly: pv.PolyData) -> int:
    """Count directed edges traversed twice in the same direction (orientation conflicts)."""
    faces = poly.faces.reshape(-1, 4)[:, 1:]
    counts = Counter(
        (int(f[k]), int(f[(k + 1) % 3])) for f in faces for k in range(3)
    )
    return sum(1 for n in counts.values() if n > 1)


def open_edges(poly: pv.PolyData) -> int:
    return poly.extract_feature_edges(
        boundary_edges=True, feature_edges=False,
        manifold_edges=False, non_manifold_edges=False,
    ).n_cells


def open_hemisphere() -> pv.PolyData:
    """Sphere cut by a slightly tilted plane: one near-planar opening."""
    sphere = pv.Sphere(radius=1.0, theta_resolution=48, phi_resolution=48)
    return sphere.clip(normal=(0.1, 0.0, 1.0), invert=True).triangulate().clean()


class SideBandTests(unittest.TestCase):
    def test_side_band_is_consistently_wound(self):
        t = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        loop = np.c_[np.cos(t), np.sin(t), 0.1 * np.sin(3 * t)]
        projected = loop.copy()
        projected[:, 2] = 0.2

        self.assertEqual(directed_edge_conflicts(createSideBand(loop, projected)), 0)


def signed_volume(poly: pv.PolyData) -> float:
    tri = poly.points[poly.faces.reshape(-1, 4)[:, 1:]].astype(float)
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def with_faces(poly: pv.PolyData, faces: np.ndarray) -> pv.PolyData:
    return pv.PolyData(poly.points, np.hstack([np.full((len(faces), 1), 3), faces]).ravel())


class InputRepairTests(unittest.TestCase):
    def setUp(self):
        self.sphere = pv.Sphere(theta_resolution=24, phi_resolution=24).triangulate().clean()
        self.faces = self.sphere.faces.reshape(-1, 4)[:, 1:]

    def test_removes_opposite_wound_duplicate_flap(self):
        flap = self.faces[0][::-1]
        mesh = with_faces(self.sphere, np.vstack([self.faces, flap]))
        repaired = remove_duplicate_faces(mesh)

        self.assertEqual(repaired.n_cells, self.sphere.n_cells - 1)
        self.assertEqual(directed_edge_conflicts(repaired), 0)

    def test_restores_consistent_winding_with_majority_orientation(self):
        faces = self.faces.copy()
        faces[:10] = faces[:10, ::-1]
        repaired = make_orientation_consistent(with_faces(self.sphere, faces))

        self.assertEqual(directed_edge_conflicts(repaired), 0)
        # The majority (unflipped) winding is kept
        self.assertAlmostEqual(signed_volume(repaired), signed_volume(self.sphere))

    def test_does_not_propagate_through_non_manifold_edges(self):
        # A fin glued on a sphere edge: the edge becomes shared by three faces
        a, b = (int(v) for v in self.faces[0][:2])
        points = np.vstack([self.sphere.points, [[0.0, 0.0, 2.0]]])
        fin = np.array([[b, a, len(points) - 1]])
        mesh = pv.PolyData(points, np.hstack([np.full((len(self.faces) + 1, 1), 3),
                                              np.vstack([self.faces, fin])]).ravel())
        repaired = make_orientation_consistent(mesh)

        np.testing.assert_array_equal(repaired.faces.reshape(-1, 4)[:-1, 1:], self.faces)

    def test_orient_outward_fixes_inward_mesh(self):
        inward = self.sphere.flip_faces()
        self.assertLess(signed_volume(inward), 0)
        self.assertGreater(signed_volume(orient_outward(inward)), 0)


class PolygonInsideTests(unittest.TestCase):
    def test_last_loop_vertex_is_part_of_the_polygon(self):
        # Regression: matplotlib Path(closed=True) ignores the last vertex
        poly_uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        cap = pv.PolyData(np.c_[poly_uv, np.zeros(4)], np.array([3, 0, 1, 2, 3, 0, 2, 3]))
        kept = deleteFacesOutsidePolygon(cap, np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                                         poly_uv, argparse.Namespace(debug=None))
        self.assertEqual(kept.n_cells, 2)


class BoundaryLoopTests(unittest.TestCase):
    def test_splits_pinched_boundary_into_simple_loops(self):
        # Two triangles touching at vertex 0 only (bow-tie): two openings sharing a vertex
        points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [-1, 0, 0], [-1, -1, 0]], dtype=float)
        mesh = pv.PolyData(points, np.array([3, 0, 1, 2, 3, 0, 3, 4]))
        loops = extract_boundary_loops(mesh)

        self.assertEqual(sorted(sorted(loop.tolist()) for loop in loops), [[0, 1, 2], [0, 3, 4]])

    def test_every_boundary_edge_belongs_to_exactly_one_loop(self):
        sphere = pv.Sphere(theta_resolution=24, phi_resolution=24).triangulate().clean()
        faces = sphere.faces.reshape(-1, 4)[:, 1:]
        mesh = with_faces(sphere, faces[np.arange(len(faces)) % 7 != 0])
        boundary = {
            tuple(sorted(e)) for e, n in Counter(
                tuple(sorted((int(f[k]), int(f[(k + 1) % 3])))) for f in mesh.faces.reshape(-1, 4)[:, 1:] for k in range(3)
            ).items() if n == 1
        }
        loop_edges = [
            tuple(sorted((int(loop[i]), int(loop[(i + 1) % len(loop)]))))
            for loop in extract_boundary_loops(mesh) for i in range(len(loop))
        ]
        self.assertEqual(len(loop_edges), len(set(loop_edges)))
        self.assertEqual(set(loop_edges), boundary)


class LoopOrientationTests(unittest.TestCase):
    def setUp(self):
        # Single triangle: its boundary is the loop 0 -> 1 -> 2
        self.mesh = pv.PolyData(np.eye(3), np.array([3, 0, 1, 2]))

    def test_reverses_loop_that_follows_mesh_winding(self):
        np.testing.assert_array_equal(
            _orient_loop_against_mesh(self.mesh, np.array([0, 1, 2])), [2, 1, 0]
        )

    def test_keeps_loop_that_opposes_mesh_winding(self):
        np.testing.assert_array_equal(
            _orient_loop_against_mesh(self.mesh, np.array([2, 1, 0])), [2, 1, 0]
        )

    def test_rejects_inconsistent_mesh_winding(self):
        # Two triangles sharing edge 1-2 with opposite windings
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
        mesh = pv.PolyData(points, np.array([3, 0, 1, 2, 3, 1, 2, 3]))
        with self.assertRaises(ValueError):
            _orient_loop_against_mesh(mesh, np.array([0, 1, 3, 2]))


class MakeCapTests(unittest.TestCase):
    """The cap must close the opening without orientation conflicts, whatever
    the mesh winding and the order in which the boundary loop was extracted."""

    args = argparse.Namespace(debug=None, debug_loop="", seedStepFactor=4.0, seedDecimator=4.0)

    def assert_closes_consistently(self, mesh, loop):
        cap = make_cap(mesh, loop, self.args)
        self.assertEqual(directed_edge_conflicts(cap), 0)

        closed = mesh.merge(cap, merge_points=True).clean(polys_to_lines=False)
        self.assertEqual(open_edges(closed), 0)
        self.assertEqual(directed_edge_conflicts(closed), 0)

    def test_all_winding_and_loop_order_combinations(self):
        for flip_mesh in (False, True):
            for reverse_loop in (False, True):
                with self.subTest(flip_mesh=flip_mesh, reverse_loop=reverse_loop):
                    mesh = open_hemisphere()
                    if flip_mesh:
                        mesh = mesh.flip_faces()
                    (loop,) = extract_boundary_loops(mesh)
                    if reverse_loop:
                        loop = loop[::-1]
                    self.assert_closes_consistently(mesh, loop)


if __name__ == "__main__":
    unittest.main()
