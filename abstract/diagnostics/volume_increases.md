# Diagnosis of volume increases between epochs (2026-10-26)

A dissection only removes tissue: the volume of a closed epoch should never exceed that of the previous epoch. Of the 104 steps between consecutive epochs ("b" epochs excluded), **15 show an increase** (+0.3 to +15.3 %). Each increase was compared with 7 "normal" pairs used as controls.

Scripts: `abstract/scripts/diagnose_volume_increases.py`, `cap_plane_offset.py`, `classify_increases.py`. Numbers: `volume_increases.csv`, `cap_plane_offset.txt`.

## Ruled out

| Hypothesis | Measurement | Result |
|---|---|---|
| Different units / frame | bounds, ICP | Everything is in metres. The epochs of a specimen are already in the same frame: the median distance from one epoch to the previous one is 0.4 to 0.7 mm, and the ICP translation is 0.07 to 0.36 mm. |
| Scale | trimmed similarity ICP (60 % closest pairs), extent along the long axis | Factor of 0.992 to 1.008 for the increases, versus 0.998 to 1.007 for the controls: indistinguishable from noise. The long axis stays stable within ±0.5 mm in most cases. Scale can explain at most ~±2 % of volume, and nothing distinguishes it from the controls. |
| Volume wedge added by the caps | cap compared with a fan on the same loops | 0.04 to 2.0 % per epoch, as in the controls. Cannot explain +13 to +15 %. |

## Causes found

| Cause | Pairs | Numbers |
|---|---|---|
| **Shifted cut face**: the main cap and its loop (where the scanned surface stops) are further outward in the next epoch | 9 / 15: spc-07 09→10 (+13.4 %), spc-12 02→03, spc-15 10→11 (+5.8 %), spc-15 13→14, spc-17 02→03 (+6.8 %), spc-17 07→08 (+15.3 %), spc-18 02→03, spc-19 08→09, spc-19 11→12 (+12.9 %) | Cap shifted outward by +2.2 to +5.1 mm; the loop moves by the same amount (difference < 0.6 mm between the two), with nearly parallel planes (< 1.5°). Corresponding slab: 19 to 57 cm³, the same order as the increases (e.g. spc-17 07→08: +34 cm³ of volume for a 40 cm³ slab). 73 to 93 % of the next epoch's volume lying outside the previous one is within 5 mm of a cap (controls: 8 to 25 %, except spc-01 01→02 at 79 % and spc-18 03→04 at 69 %). Hemisphere thickness (3rd principal axis) increases by 3 to 6 mm for the largest increases (spc-07 09→10, spc-17 02→03 and 07→08, spc-18 02→03, spc-19 11→12); for the others, it varies from −0.2 to +2.7 mm. |
| **Closing of the previous epoch** (folded openings, curved caps) | 2 / 15: spc-07 03→05 (+3.2 %), spc-15 08→09 (+5.2 %) | The previous epoch has 9 and 4 curved caps and 54 and 41 openings, versus 1 and 0 curved caps and 5 and 1 openings for the next epoch. spc-15 epo-08 is the "holes under the cap" epoch (fig. 3b). The main cap does not move (+1.3 and 0.0 mm). |
| **Within uncertainty** (< 3.4 %) | 4 / 15: spc-02 07→08, spc-10 05→06, spc-12 05→06, spc-16 06→07 (+0.3 to +1.9 %) | Cap shifted by 0.5 to 1.2 mm, which is within the range of the controls. |

## Conclusion

- **The large increases come from the mesh, not from the closing.** The scanned surface does not stop at the same place from one epoch to the next near the non-photographed face (medial/cut face). The cap faithfully follows this edge, and any reasonable closing (planar, fan, mean plane) would give the same offset, since it is the loop itself that moves.
- **Useful order of magnitude for the abstract**: in the controls, the cap already moves by −2.2 to +1.9 mm from one epoch to the next. With a cut face of about 70 to 120 cm², 1 mm is worth 7 to 12 cm³, i.e. 2 to 4 % of the volume. **The uncertainty on the cut-face position (~±2 mm, i.e. ~±5 to 8 %) exceeds that of the closing (≤ 3.4 %)**, and it is of the same order as the volume removed in a typical step (median: 8.7 %).
- Two increases are partly due to our imperfect closings (folded openings of spc-07 and spc-15 epo-08).

## Caveats (not verified)

- I do not know *why* the scanned edge moves: tissue sagging on its support, different manual cropping of the meshes (support, table), different photographic coverage, or registration of the epochs to each other by BraDiPho. This would require the BraDiPho documentation or the original photos.
- I have not verified that no tissue is removed from the medial side during the dissection. If it were, part of the *negative* cap shifts would be real.
- Similarity ICP is disturbed by the tissue removed between epochs. The scale factor is only reliable to within ~±0.5 %.
- Voxels are 1.5 mm: the volume "outside the other epoch" contains about 4 to 10 cm³ of discretization noise (control values).
- The classification thresholds (2 mm, 70 %) were chosen from only 7 controls.
