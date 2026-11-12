# Figure3

Generate the eight panels from complete local 3D TIFFs and lipid
coordinates. Run from the `DEPICT` root:

```sh
python Figures/Figure3/rebuild.py
```

Use `--panels a b` or `--panels d h` to rebuild selected panels. A full run also
writes `Figure3_reproduced.pdf`.

| Panel | Content | Outputs in `panel_x/` |
| --- | --- | --- |
| a | HEK293T, 240 angles | RGB TIFF, annotated SVG, numerical section NPZ |
| b | HeLa, 240 angles | RGB TIFF, annotated SVG, numerical section NPZ |
| c | COS-7, 240 angles | RGB TIFF, annotated SVG, numerical section NPZ |
| d | ID isotropy: HEK293T / HeLa / COS-7 | Box-plot PDF, annotated SVG, FWHM and summary CSVs |
| e | U2OS, 240 angles | RGB TIFF, annotated SVG, numerical section NPZ |
| f | Interphase T cell, 240 angles | RGB TIFF, annotated SVG, numerical section NPZ |
| g | Mitotic T cell, 240 angles | RGB TIFF, annotated SVG, numerical section NPZ |
| h | OOD isotropy: U2OS / interphase T / mitotic T | Box-plot PDF, annotated SVG, FWHM and summary CSVs |

The 12 Rytov/TV-BPM volumes belong in `inputs/volumes/`; the six DEPICT
volumes are accessed through `inputs/lipid_isotropy`, a relative link to
`../shared_inputs/lipid_isotropy/`. Supplementary Figure 8 uses the same six
volumes and 240-angle coordinate/measurement tables. `inputs/volumes.csv`
indexes the inputs; `config.json` supplies paths, panel identities, slice
settings and ROIs. Restore the separate volume dataset before rebuilding.

Image panels use center crops, configured orientations and display offsets,
diagonal comparisons and 50 x 50 pixel zoom regions. The RGB TIFFs are
3804 x 1779 pixels at 300 dpi; standalone annotated panels are 8.5 cm wide. Native data are RI
contrast relative to 1.33, displayed over 0 to 0.06, corresponding to absolute
RI 1.33 to 1.39. Voxels are 0.09 um isotropic. The HEK293T image uses the
2024-12-19 acquisition. Display crops and transforms are separate from the
3D quantitative coordinate system.

Lipid analysis starts with supplied coordinates, without detection or
segmentation. Profiles use linear interpolation at 0.25 pixel spacing,
half-width 16 pixels, Gaussian sigma 1 profile smoothing, an edge baseline
from 20% of the samples and a peak search within 4 pixels. Valid profiles
require contrast >= 0.01 and FWHM <= 28 pixels. The ratio is
`FWHM_z / ((FWHM_x + FWHM_y) / 2)`. Coordinates refer to source `z[64:-64]`
after a 90-degree XY rotation, without a lateral crop.

The quantitative analysis uses 145 DEPICT-valid lipid coordinates from
`inputs/lipid_isotropy/reference_droplets_240.csv`. All three methods are
sampled at those positions and apply their own FWHM validity checks.

| Condition | Common coordinates / valid DEPICT | Valid TV-BPM | Valid Rytov |
| --- | ---: | ---: | ---: |
| HEK293T | 54 | 54 | 19 |
| HeLa | 24 | 24 | 24 |
| COS-7 | 6 | 6 | 6 |
| U2OS | 2 | 2 | 2 |
| Interphase T | 28 | 17 | 15 |
| Mitotic T | 31 | 31 | 31 |
| Total | 145 | 134 | 97 |

To regenerate the shared measurement tables from the complete volumes:

```sh
python Figures/shared_inputs/lipid_isotropy/rebuild_reference.py
```

`scripts/sections.py` implements cropping, orientation and slice extraction;
`scripts/render_panels_with_zoom.py` renders the image panels and zoom regions.
`scripts/analyze_isotropy.py` calls the shared FWHM sampler in
`../../figure_reproduction/`. `scripts/plot_panels.py` adds editable
labels, plots and the assembled overview. Dependencies are in `requirements.txt`.
