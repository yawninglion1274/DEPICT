# Supplementary Figure 8

Generate all eight panels from the local full-volume inputs and the shared
Figure3 240-angle lipid measurements. Run from the `DEPICT` root:

```sh
python Figures/Supplementary_Figure_8/rebuild.py
```

Use `--panels d h` to select panel letters. Each panel exports only
`panel_x/panel_x_with_text.svg`, including editable text, titles, axis labels,
ticks and legends. Image data are embedded. No additional figure format or
text-free version is produced. The quantitative panels compare 240 with 24 angles.

| Panel | Content |
| --- | --- |
| a | HEK293T, 24 angles |
| b | HeLa, 24 angles |
| c | COS-7, 24 angles |
| d | ID lipid isotropy: HEK293T / HeLa / COS-7 |
| e | U2OS, 24 angles |
| f | Interphase T cell, 24 angles |
| g | Mitotic T cell, 24 angles |
| h | OOD lipid isotropy: U2OS / interphase T / mitotic T |

`inputs/volumes/*.tif` contains the complete 24-angle 3D TIFFs, including all
256 Z planes. `inputs/lipid_isotropy` is a relative symlink to the single shared
dataset at `../shared_inputs/lipid_isotropy/`; Figure3 points to the same files.
It contains the complete six 240-angle TIFFs and
`reference_droplets_240.csv`, measured from the same volumes as Figure3.
The 240-angle analysis uses the 145 valid DEPICT coordinates in that shared
reference. `inputs/droplet_coordinates_24.csv` contains only the 121 final valid
24-angle coordinates, with the same lipid IDs and positions as the corresponding
shared table entries.

`config.json` records all relative paths, transforms and analysis settings;
`inputs/panels.csv` supplies the panel map, with all paths relative to this
figure directory.

The ratio is `FWHM_z / ((FWHM_x + FWHM_y) / 2)`. Coordinates are zero-based in
source `z[64:-64]` after a 90-degree XY rotation, with no XY crop. The
FWHM sampler is included without detection or segmentation code. The 24-angle
coordinates are a valid subset of the fixed 240-angle coordinates. Each group
uses its own final analysis set. All 240-angle profiles and 24-angle ID
profiles require contrast >= 0.01 and FWHM <= 28 pixels. The 24-angle OOD
profiles disable the minimum contrast and maximum width filters. Every supplied
analysis coordinate must yield finite X/Y/Z widths and a finite ratio;
rebuilding fails if a retained point is invalid.

All groups are measured from their complete TIFFs at the supplied valid
coordinates. The 240-angle ratios are checked against the shared Figure3
measurement table.

| Condition | Final 240-angle coordinates | Final 24-angle coordinates |
| --- | ---: | ---: |
| HEK293T | 54 | 54 |
| HeLa | 24 | 24 |
| COS-7 | 6 | 6 |
| U2OS | 2 | 2 |
| Interphase T | 28 | 28 |
| Mitotic T | 31 | 7 |
| Total | 145 | 121 |

Each quantitative panel saves only final valid measurements, plot values,
and sample sizes beside its SVG. The seven 24-angle mitotic T-cell lipid IDs
are 13, 15, 21, 26, 27, 32 and 51. The box plots retain 1.5-IQR whiskers,
hidden outliers and the ratio=1 reference line.

Image panels use center crops, configured slices and offsets, and an RI display
range (1.33-1.39). HEK293T uses the 2024-12-19 acquisition in native orientation
at the crop center. Display transforms do not change the quantitative
coordinate system.

Shared code and dependencies are in `../../figure_reproduction/`.
