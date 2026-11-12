# Supplementary Figure 6

Reproduce all six panels from complete local 3D TIFF volumes:

```sh
python -B Figures/Supplementary_Figure_6/rebuild.py
```

The command works from any working directory when the script path is absolute.
Use `--panels a c` to select panel letters. Every panel has exactly one figure
export, `panel_x/panel_x_with_text.svg`, with editable SVG text, labels, ticks,
legends and image annotations. Image pixels are embedded in the SVG; no external
image links are required. No TIFF, PNG, PDF or text-free figure is exported.

| Panel | Content | Numerical input |
| --- | --- | --- |
| a | HEK293T, DEPICT (BPM) | `inputs/volumes/293t_bpm.tif` |
| b | HEK293T, DEPICT (Rytov) | `inputs/volumes/293t_rytov.tif` |
| c | HEK293T, three matched line profiles | The same a/b volumes and `inputs/line_coordinates.csv` |
| d | HeLa, DEPICT (BPM) | `../shared_inputs/lipid_isotropy/volumes/hela_240.tif` |
| e | HeLa, DEPICT (Rytov) | `inputs/volumes/hela_rytov.tif` |
| f | HeLa, three matched line profiles | The same d/e volumes and `inputs/line_coordinates.csv` |

`config.json` records all relative input paths, slice indices and display
transforms. `inputs/panels.csv` maps the panel letters to their script and output;
all paths in that table are relative to this figure directory. The HeLa BPM volume is
also used for Supplementary Figure 8 and is stored only once.

Volumes store RI contrast. Images use contrast limits 0-0.06, displayed as
absolute RI 1.33-1.39. They retain the original center crop to 256 x 450 x 450,
slice indices and orientation. Colored sampling strokes use the same endpoints
for both methods. Each line is 4 um long with 179 bilinear samples; no smoothing,
normalization or clipping is applied. Profiles are recomputed from the TIFFs and
saved beside c/f as `line_profile_samples.csv`. The supplied reference sample
CSV is only a numerical validation target and is not read by the plotting code.

The internal plane keys `xy/xz/yz` retain the original renderer's naming.
Following the reference PDF, their displayed labels are `xy/yz/xz`; the rotated
XY display exchanges the names of the two axial views.

Shared processing and rendering code is in `../../figure_reproduction/`.
Dependencies are listed in its `requirements.txt`. Restore the volume
inputs described in [Data](../../docs/data.md) before rebuilding.
