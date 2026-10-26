"""
Add the offset of the main cap and the most likely cause to volume_increases.csv.

Reads the output of diagnose_volume_increases.py (volume_increases.csv) and of
cap_plane_offset.py (cap_plane_offset.txt), both in abstract/diagnostics/.

Rules (thresholds chosen from the control pairs, whose cap offset stays
within -1.5 to +1.9 mm and whose volume of the later epoch outside the earlier
one is 4 to 20 cm3):

- "cut face shifted": the main cap and its loop (where the scanned surface
  ends) moved outward by >= 2 mm, and >= 70 % of the later epoch lying outside
  the earlier one is within 5 mm of a cap;
- "closing of the earlier epoch": the earlier epoch has >= 4 curved caps
  (folded openings, visual defects) and the later one at most 1;
- "within uncertainty": increase below the closing uncertainty (3.4 %);
- otherwise "unexplained".

Usage
-----
python abstract/scripts/classify_increases.py
"""

import csv
from pathlib import Path

DIAG = Path(__file__).resolve().parents[2] / "abstract" / "diagnostics"
BAND = 3.4


def main() -> None:
    offsets = {}
    for line in (DIAG / "cap_plane_offset.txt").read_text().splitlines()[1:]:
        f = line.split()
        spc, (e0, e1) = f[0], f[1].split("->")
        offsets[(spc, e0, e1)] = {"cap_angle_deg": f[2], "cap_offset_mm": f[3],
                                  "loop_offset_mm": f[4], "cap_slab_cm3": f[5]}
    with open(DIAG / "volume_increases.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r.update(offsets.get((r["specimen"], r["epoch_from"], r["epoch_to"]), {}))
        if r["kind"] == "control":
            r["cause"] = "control"
            continue
        change = float(r["change_%"])
        off = float(r.get("cap_offset_mm", "nan"))
        near = float(r["to_outside_near_caps_frac"])
        curved_from, curved_to = int(r["curved_caps_from"]), int(r["curved_caps_to"])
        if off >= 2 and near >= 0.7:
            r["cause"] = "cut face shifted"
        elif curved_from >= 4 and curved_to <= 1:
            r["cause"] = "closing of the earlier epoch (folded openings)"
        elif change < BAND:
            r["cause"] = "within uncertainty"
        else:
            r["cause"] = "unexplained"
    fields = list(rows[0]) + [k for k in ("cap_angle_deg", "cap_offset_mm", "loop_offset_mm",
                                          "cap_slab_cm3", "cause") if k not in rows[0]]
    with open(DIAG / "volume_increases.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['specimen']} {r['epoch_from']}->{r['epoch_to']} {float(r['change_%']):+6.1f}%  "
              f"cap {r.get('cap_offset_mm', '?'):>6} mm  {r['cause']}")


if __name__ == "__main__":
    main()
