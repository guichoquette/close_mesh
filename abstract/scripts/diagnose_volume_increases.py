"""
Diagnose the volume increases between consecutive epochs of BraDiPho dissections.

A dissection only removes tissue, so the volume of a closed epoch should never
be larger than the volume of the epoch before. For each pair of consecutive
epochs (the "b" epochs are left out), this script measures:

- the shape of each closed mesh (extent along its principal axes, area),
- the openings of each input mesh (number, total area, position of the largest
  one along the long axis of the brain),
- the wedge of volume added by the caps of each closed mesh, against a flat fan
  on the same loops,
- the volume of the later epoch lying outside the earlier one, and of the
  earlier one lying outside the later one (voxels, without any registration),
  and where the excess lies (near the caps or elsewhere),
- a scale factor between the two epochs, by a trimmed similarity ICP.

Usage
-----
python abstract/scripts/diagnose_volume_increases.py [--grid 1.5] [--pairs all|increases]
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "brain_data"
OUT = ROOT / "abstract" / "diagnostics"

# Pairs with a normal decrease, used as controls
CONTROLS = [("spc-01", "01", "02"), ("spc-01", "07", "08"), ("spc-06", "07", "08"),
            ("spc-16", "07", "08"), ("spc-18", "03", "04"), ("spc-19", "09", "10"),
            ("spc-17", "04", "05")]


def read_results() -> dict[str, dict[str, dict]]:
    """Read results.csv as {specimen: {epoch: row}}."""
    by_spc: dict[str, dict[str, dict]] = {}
    with open(DATA / "output" / "results.csv", newline="") as f:
        for row in csv.DictReader(f):
            by_spc.setdefault(row["specimen"], {})[row["epoch"]] = row
    return by_spc


def consecutive_pairs(by_spc: dict) -> list[tuple[str, str, str, float]]:
    """Consecutive epochs of the main sequence, with their relative volume change in %."""
    pairs = []
    for spc, epochs in sorted(by_spc.items()):
        main = sorted(e for e in epochs if not e.endswith("b") and epochs[e]["volume"])
        for e0, e1 in zip(main, main[1:]):
            v0, v1 = float(epochs[e0]["volume"]), float(epochs[e1]["volume"])
            pairs.append((spc, e0, e1, (v1 - v0) / v0 * 100))
    return pairs


def faces_of(mesh: pv.PolyData) -> np.ndarray:
    """Triangle faces as an (n, 3) array."""
    return mesh.faces.reshape(-1, 4)[:, 1:]


def principal_extents(points: np.ndarray) -> np.ndarray:
    """Extent (max - min) of the points along their three principal axes, longest first."""
    centred = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centred[:: max(1, len(centred) // 200000)], full_matrices=False)
    proj = centred @ vt.T
    return proj.max(axis=0) - proj.min(axis=0)


def boundary_loops(mesh: pv.PolyData) -> list[dict]:
    """Openings of an input mesh: number of vertices, area of the projected loop, centroid."""
    edges = mesh.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                       manifold_edges=False, non_manifold_edges=False)
    if edges.n_points == 0:
        return []
    loops = []
    conn = edges.connectivity()
    for rid in np.unique(conn["RegionId"]):
        part = conn.extract_cells(np.where(conn.cell_data["RegionId"] == rid)[0])
        pts = np.asarray(part.points)
        if len(pts) < 3:
            continue
        # Order the loop by walking its edges
        lines = part.cells.reshape(-1, 3)[:, 1:] if part.n_cells else np.empty((0, 2), int)
        area = 0.0
        if len(lines):
            centre = pts.mean(axis=0)
            _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
            uv = (pts - centre) @ vt[:2].T
            # Shoelace on each edge, oriented by the walk: |sum| of signed areas of the
            # triangles (centre, a, b) taken with a consistent orientation of the chain
            adj: dict[int, list[int]] = {}
            for a, b in lines:
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)
            start = lines[0, 0]
            order, prev, cur = [start], -1, start
            for _ in range(len(lines)):
                nxt = [n for n in adj[cur] if n != prev]
                if not nxt or nxt[0] == start:
                    break
                prev, cur = cur, nxt[0]
                order.append(cur)
            p = uv[order]
            area = 0.5 * abs(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))
        loops.append({"n": len(pts), "area": area, "centroid": pts.mean(axis=0)})
    return loops


def cap_wedges(closed: pv.PolyData, original_points: np.ndarray) -> tuple[float, float]:
    """
    Area of the caps and wedge of volume they add against a flat fan on their loops.

    Cap vertices are the vertices of the closed mesh that are not vertices of the
    input mesh. Cap faces are the faces using at least one of them. Each connected
    group of cap faces is closed by a fan from its boundary to the centroid of the
    boundary, oriented against the cap; the signed volume of this small closed
    surface is the wedge (positive when the cap lies outside the fan).
    """
    tree = cKDTree(original_points)
    d, _ = tree.query(np.asarray(closed.points), k=1)
    new = d > 1e-9
    f = faces_of(closed)
    cap = f[new[f].any(axis=1)]
    if len(cap) == 0:
        return 0.0, 0.0
    pts = np.asarray(closed.points, dtype=float)
    a, b, c = pts[cap[:, 0]], pts[cap[:, 1]], pts[cap[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    vol_cap = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6

    # Boundary edges of the cap faces (directed as they appear in the cap faces)
    directed = np.concatenate([cap[:, [0, 1]], cap[:, [1, 2]], cap[:, [2, 0]]])
    key = np.sort(directed, axis=1)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    border = directed[counts[inv.ravel()] == 1]
    if len(border) == 0:
        return area, 0.0
    # Group border edges into loops by connectivity of their vertices
    verts, local = np.unique(border, return_inverse=True)
    local = local.reshape(-1, 2)
    parent = np.arange(len(verts))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for u, v in local:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    roots = np.array([find(i) for i in range(len(verts))])
    edge_root = roots[local[:, 0]]
    vol_fan = 0.0
    for r in np.unique(edge_root):
        e = border[edge_root == r]
        centre = pts[np.unique(e)].mean(axis=0)
        # Fan triangle (b, a, centre) closes the cap edge (a, b)
        p, q = pts[e[:, 1]], pts[e[:, 0]]
        vol_fan += np.einsum("ij,ij->i", p, np.cross(q, np.broadcast_to(centre, p.shape))).sum() / 6
    return area, vol_cap + vol_fan


def voxel_inside(mesh: pv.PolyData, grid_pts: np.ndarray) -> np.ndarray:
    """Boolean mask of the grid points inside a closed mesh."""
    sel = pv.PolyData(grid_pts).select_enclosed_points(mesh, check_surface=False)
    return np.asarray(sel["SelectedPoints"]).astype(bool)


def similarity_icp(src: np.ndarray, dst: np.ndarray, keep: float = 0.6, iters: int = 40):
    """
    Trimmed similarity ICP of src onto dst (Umeyama at each step).

    Only the fraction ``keep`` of the closest pairs is used at each step, so that
    the tissue removed between the two epochs does not pull the fit.

    Returns
    -------
    scale, rotation angle (deg), translation norm (m), median residual (m) of kept pairs
    """
    tree = cKDTree(dst)
    s_tot, r_tot, t_tot = 1.0, np.eye(3), np.zeros(3)
    cur = src.copy()
    for _ in range(iters):
        d, idx = tree.query(cur, k=1)
        thr = np.quantile(d, keep)
        m = d <= thr
        x, y = cur[m], dst[idx[m]]
        mx, my = x.mean(0), y.mean(0)
        xc, yc = x - mx, y - my
        cov = yc.T @ xc / len(x)
        u, sv, vt = np.linalg.svd(cov)
        sign = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
        r = u @ sign @ vt
        s = np.trace(np.diag(sv) @ sign) / (xc ** 2).sum(1).mean()
        t = my - s * r @ mx
        cur = s * cur @ r.T + t
        s_tot, r_tot, t_tot = s * s_tot, r @ r_tot, s * r @ t_tot + t
    d, _ = tree.query(cur, k=1)
    angle = np.degrees(np.arccos(np.clip((np.trace(r_tot) - 1) / 2, -1, 1)))
    return s_tot, angle, np.linalg.norm(t_tot), float(np.median(d[d <= np.quantile(d, keep)]))


def load_epoch(spc: str, epoch: str, results: dict) -> dict:
    """Load the input and closed meshes of an epoch and measure what does not depend on the pair."""
    row = results[spc][epoch]
    inp = pv.read(row["input"])
    closed = pv.read(row["output"])
    loops = boundary_loops(inp)
    loops.sort(key=lambda L: -L["area"])
    cap_area, wedge = cap_wedges(closed, np.asarray(inp.points))
    return {"input": inp, "closed": closed, "volume": float(row["volume"]),
            "extents": principal_extents(np.asarray(closed.points)),
            "area": closed.area, "loops": loops, "cap_area": cap_area, "wedge": wedge,
            "curved_caps": int(row["curved_caps"])}


def diagnose_pair(spc, e0, e1, change, results, grid_mm, cache) -> dict:
    """Measure one pair of consecutive epochs."""
    t0 = time.time()
    for e in (e0, e1):
        if (spc, e) not in cache:
            cache[(spc, e)] = load_epoch(spc, e, results)
    A, B = cache[(spc, e0)], cache[(spc, e1)]

    # Voxels on a common grid, no registration: the epochs are used as given
    lo = np.minimum(A["closed"].bounds[::2], B["closed"].bounds[::2]) - 1e-3
    hi = np.maximum(A["closed"].bounds[1::2], B["closed"].bounds[1::2]) + 1e-3
    h = grid_mm * 1e-3
    axes = [np.arange(lo[i], hi[i], h) for i in range(3)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    inA, inB = voxel_inside(A["closed"], g), voxel_inside(B["closed"], g)
    vox = h ** 3
    b_out = inB & ~inA
    a_out = inA & ~inB

    # Where is the tissue of B outside A: near the caps of A or B (cut / coverage) or elsewhere?
    near_cap_frac = np.nan
    if b_out.sum():
        cap_pts = []
        for E in (A, B):
            d, _ = cKDTree(np.asarray(E["input"].points)).query(np.asarray(E["closed"].points), k=1)
            cap_pts.append(np.asarray(E["closed"].points)[d > 1e-9])
        cap_pts = np.concatenate(cap_pts)
        if len(cap_pts):
            dc, _ = cKDTree(cap_pts).query(g[b_out], k=1)
            near_cap_frac = float((dc < 5e-3).mean())

    # Mean thickness of B outside A over the surface of B: a uniform shell suggests scale
    shell_mm = b_out.sum() * vox / B["area"] * 1e3

    # Scale between epochs: trimmed similarity ICP of B surface onto A surface
    rng = np.random.default_rng(0)
    pa = np.asarray(A["input"].points)
    pb = np.asarray(B["input"].points)
    pa = pa[rng.choice(len(pa), min(len(pa), 60000), replace=False)]
    pb = pb[rng.choice(len(pb), min(len(pb), 20000), replace=False)]
    d0, _ = cKDTree(pa).query(pb, k=1)
    scale, angle, trans, resid = similarity_icp(pb, pa)

    la, lb = A["loops"], B["loops"]
    # Position of the largest opening along the long axis of A
    centred = np.asarray(A["closed"].points) - np.asarray(A["closed"].points).mean(0)
    _, _, vt = np.linalg.svd(centred[::50], full_matrices=False)
    axis = vt[0]
    origin = np.asarray(A["closed"].points).mean(0)
    pos = lambda L: float((L[0]["centroid"] - origin) @ axis * 1e3) if L else np.nan  # noqa: E731

    return {
        "specimen": spc, "epoch_from": e0, "epoch_to": e1,
        "volume_from_cm3": A["volume"] * 1e6, "volume_to_cm3": B["volume"] * 1e6,
        "change_%": change,
        "extent_from_mm": "/".join(f"{x * 1e3:.1f}" for x in A["extents"]),
        "extent_to_mm": "/".join(f"{x * 1e3:.1f}" for x in B["extents"]),
        "area_from_cm2": A["area"] * 1e4, "area_to_cm2": B["area"] * 1e4,
        "openings_from": len(la), "openings_to": len(lb),
        "opening_area_from_cm2": sum(L["area"] for L in la) * 1e4,
        "opening_area_to_cm2": sum(L["area"] for L in lb) * 1e4,
        "largest_opening_from_cm2": la[0]["area"] * 1e4 if la else 0,
        "largest_opening_to_cm2": lb[0]["area"] * 1e4 if lb else 0,
        "largest_opening_pos_from_mm": pos(la), "largest_opening_pos_to_mm": pos(lb),
        "cap_area_from_cm2": A["cap_area"] * 1e4, "cap_area_to_cm2": B["cap_area"] * 1e4,
        "wedge_from_%": A["wedge"] / A["volume"] * 100, "wedge_to_%": B["wedge"] / B["volume"] * 100,
        "curved_caps_from": A["curved_caps"], "curved_caps_to": B["curved_caps"],
        "to_outside_from_cm3": b_out.sum() * vox * 1e6,
        "from_outside_to_cm3": a_out.sum() * vox * 1e6,
        "to_outside_near_caps_frac": near_cap_frac,
        "to_outside_shell_mm": shell_mm,
        "median_dist_to_from_mm": float(np.median(d0) * 1e3),
        "icp_scale": scale, "icp_angle_deg": angle, "icp_translation_mm": trans * 1e3,
        "icp_residual_mm": resid * 1e3,
        "seconds": round(time.time() - t0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--grid", type=float, default=1.5, help="voxel size in mm")
    parser.add_argument("--pairs", choices=["all", "increases"], default="all")
    parser.add_argument("--only", nargs="*", help="restrict to specimens")
    args = parser.parse_args()

    results = read_results()
    pairs = consecutive_pairs(results)
    todo = [(s, a, b, c, "increase") for s, a, b, c in pairs if c > 0]
    if args.pairs == "all":
        ctrl = {(s, a, b) for s, a, b in CONTROLS}
        todo += [(s, a, b, c, "control") for s, a, b, c in pairs if (s, a, b) in ctrl]
    if args.only:
        todo = [t for t in todo if t[0] in args.only]

    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / "volume_increases.csv"
    rows, cache = [], {}
    for spc, e0, e1, change, kind in sorted(todo):
        row = {"kind": kind, **diagnose_pair(spc, e0, e1, change, results, args.grid, cache)}
        rows.append(row)
        print(f"{spc} {e0}->{e1} {kind} {change:+.1f}%  B\\A={row['to_outside_from_cm3']:.1f} cm3 "
              f"near caps={row['to_outside_near_caps_frac']:.2f}  scale={row['icp_scale']:.4f} "
              f"wedge={row['wedge_from_%']:.2f}/{row['wedge_to_%']:.2f}%  ({row['seconds']} s)", flush=True)
        # Keep the cache small: only the epochs of the current specimen
        cache = {k: v for k, v in cache.items() if k[0] == spc}
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
