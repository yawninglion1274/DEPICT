# Shared Figure3 / Supplementary Figure 8 lipid inputs

Both figures' `inputs/lipid_isotropy` links resolve to this directory. Keep it
alongside the figure folders so their relative input paths resolve.

All measurements use the complete TIFFs used for Figure3's image panels.
To regenerate the measurement tables, run from the `DEPICT` root:

```sh
python Figures/shared_inputs/lipid_isotropy/rebuild_reference.py
```

`analysis_config.json` maps the full three-method volumes and FWHM settings.
The six DEPICT TIFFs are in `volumes/`; the Rytov and TV-BPM TIFFs are shared
with Figure3's `inputs/volumes/`. The script samples the supplied coordinates,
applies the configured profile quality filters and rebuilds all shared tables.
No lipid detection or segmentation code is needed or provided in this package.

| File | Meaning |
| --- | --- |
| `droplet_coordinates_240.csv` | The 148 fixed input coordinates for rebuilding the shared tables. |
| `fwhm_measurements_240.csv` | The 444 method measurements, including profile validity and reasons. |
| `reference_droplets_240.csv` | The 145 DEPICT-valid common coordinates, with three-method widths, QC and ratios. |
| `reference_ratios_240.csv` | All 376 valid method rows, computed from those widths. |

The valid counts are DEPICT 145, TV-BPM 134 and Rytov 97. Both figures use
the same 28 interphase T-cell coordinates. Supplementary Figure 8's 24-angle
analysis uses its own valid subset in `inputs/droplet_coordinates_24.csv`.

Coordinates are zero-based Z/Y/X in source `z[64:-64]` after
`np.rot90(..., axes=(1,2))`, with no lateral crop, at 0.09 um per voxel.
The ratio is `FWHM_z / ((FWHM_x + FWHM_y) / 2)`.

Figure3 and Supplementary Figure 8 remeasure their quantitative panels from
the complete TIFFs and verify their 240-angle results against the same shared
measurement table.
