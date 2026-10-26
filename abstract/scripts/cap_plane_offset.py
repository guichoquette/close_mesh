"""
Offset of the main cap between two consecutive epochs, measured on the closed meshes.

The main cap of a closed epoch is the largest connected group of cap faces
(faces using a vertex that is not in the input mesh): it closes the unscanned
cut face of the hemisphere. For a pair of epochs, this script reports the angle
between the two main caps, the distance between them along the normal of the
earlier one, and the slab of volume this offset represents (offset x cap area),
to compare with the volume change of the pair.

Usage
-----
python abstract/scripts/cap_plane_offset.py spc-17 07 08 [spc ep0 ep1 ...]
"""

import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_figures import cap_mask, paths  # noqa: E402


def main_cap(spc: str, epoch: str) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Main cap of a closed epoch: area-weighted centre, outward unit normal, area,
    and the vertices of its loop (cap vertices that belong to the input mesh,
    i.e. where the scanned surface ends).
    """
    inp_path, closed_path = paths(spc, epoch)
    inp, closed = pv.read(inp_path), pv.read(closed_path)
    caps = closed.extract_cells(np.where(cap_mask(closed, inp))[0]).extract_surface()
    caps = caps.connectivity("largest").extract_surface()
    caps = caps.compute_cell_sizes(length=False, volume=False)
    areas = np.asarray(caps["Area"])
    centres = np.asarray(caps.cell_centers().points)
    c = (centres * areas[:, None]).sum(0) / areas.sum()
    n = np.linalg.svd(centres - c, full_matrices=False)[2][2]
    if n @ (c - np.asarray(closed.points).mean(axis=0)) < 0:
        n = -n
    d, _ = cKDTree(np.asarray(inp.points)).query(np.asarray(caps.points), k=1)
    return c, n, float(areas.sum()), np.asarray(caps.points)[d <= 1e-9]


def main(args: list[str]) -> None:
    print("pair            angle_deg  cap_offset_mm  loop_offset_mm  slab_cm3  cap_area_cm2")
    for i in range(0, len(args), 3):
        spc, e0, e1 = args[i:i + 3]
        ca, na, aa, la = main_cap(spc, e0)
        cb, nb, _, lb = main_cap(spc, e1)
        # Median height of the loop vertices (end of the scanned surface) along na
        loop_off = np.median(lb @ na) - np.median(la @ na)
        angle = np.degrees(np.arccos(np.clip(na @ nb, -1, 1)))
        off = (cb - ca) @ na
        print(f"{spc} {e0}->{e1}   {angle:8.1f}  {off * 1e3:12.2f}  {loop_off * 1e3:14.2f}  "
              f"{off * aa * 1e6:8.1f}  {aa * 1e4:11.1f}",
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
