# Shared Postprocessing

This package provides Kalman smoothing, 3D optical flow, RI masks, strain
analysis and rendering for Figure 5, Supplementary Figure 11 and Extended
Data Figure 3. Figure entry points supply a JSON configuration. Native input
recordings and optional read-only caches are configured separately; see
[Data](../docs/data.md) for installation on another machine.

## Modules

| Module | Responsibility |
| --- | --- |
| `kalman_filter.py` | Per-pixel temporal Kalman forward filter and backward RTS; complete native XY planes and complete temporal context. |
| `optical_flow.py`, `_tvl1.py` | Coarse-to-fine full-volume dense 3D TV-L1; ordered `(dz,dy,dx)` in native voxels/frame. |
| `masks.py` | Fixed RI-based tissue support shared by image rendering and strain analysis. |
| `visualize.py`, `_streamlines.py` | RI/speed TIFFs and streamline geometry. |
| `areal_strain_rate.py` | Given masks and displacement fields, fit unweighted affine OLS and save numerical NPZ/CSV/JSON. No plotting. |
| `flow_statistics.py` | Calibrated section-speed signals, spatial median/quartiles, per-time histograms and regional onset criteria. |
| `kymograph.py` | Fixed native XY sampling of cached Kalman volumes. |
| `plotting.py` | Annotated SVG rate, regional speed statistics/distributions and kymograph panels from config and existing caches. |
| `volume_io.py` | Native input manifests, volume/section coordinates, rotations, crops and provenance. |
| `pipeline.py` | Config loading, stage orchestration, cache checking and output placement. |

Streamline geometry supports XY and rectangular XZ sections with the same
display-grid scale in both directions. Rendering uses white strokes, black
outlines and filled triangular heads.

## Shared Processing Contract

1. Kalman RTS uses `map_smooth=3`, `qr_scale=2`. Every pixel is filtered with
   the entire supplied time series, even if only a shorter event window is
   stored or analyzed. Raw RI is preserved as float32 without clipping,
   normalization, notch filtering or quantization.
2. Optical flow clips a separate solver input to RI `[0,0.06]` and divides
   by 0.06. The solver uses attachment 15, tightness 0.3, 5 warps, 10
   iterations, tolerance 1e-4 and no median prefilter. It processes complete
   native ZYX volumes; a section is extracted only after solving 3D flow.
3. Mask generation uses the temporal mean of Kalman RI, Gaussian sigma 1
   native pixel, strict `RI > threshold`, the largest connected component
   and binary hole filling. The default XY threshold is 0.007. Figure 5 XZ
   sections retain 0.006 through the same function. Optional disjoint ROIs
   define multiple masks using identical logic. Mask generation never uses
   motion, speed or onset. No additional display-envelope closing is applied.
4. Full 3D speed is `norm(dz,dy,dx)*voxel_um*fps`. Streamlines show the actual
   in-plane projected vector: `(dy,dx)` for XY and `(dz,dx)` for XZ. Clockwise
   rotation transforms vector signs as well as scalar-image coordinates.
5. Streamline visibility uses only projected speed >=0.25 um/s, path length
   >=26 reference display-grid units and boundary containment. The grid has
   431 columns and proportionate XZ rows. Seed density stays 1.6; there is
   no component-area, curvature, extent, turn-angle or explicit head-spacing
   cutoff. Line strokes and complete heads must fit inside the tissue/speed
   domain, including internal low-speed gaps. A roughly two-grid-unit
   boundary allowance is applied. No vector smoothing is introduced.
6. Areal strain rate fits all supplied mask pixels using
   `[u_column,u_row] = b + B*[column,row]`, centered spatial coordinates and
   float64 unweighted OLS. Rate uses positive divergence, `+fps*trace(B)`.
   Outputs include coefficients, gradients, residual
   RMSE and native midpoint times. No temporal smoothing or plotting is
   applied; display-specific smoothing can be a later plotting decision.

Calibration is isotropic `voxel_um`, with positive `fps`. Existing datasets
use 32.5 volumes/s and 0.0900811958 um/voxel, except single-angle panel a,
which retains its native 0.09 um/voxel calibration.

## Config And Inputs

All paths in a config resolve relative to that config. Input CSV paths
resolve relative to the CSV; supplied figure CSVs record absolute native
volume paths plus file size/mtime to detect changed inputs. A series must
contain at least two consecutive native frames in increasing order.

`source` accepts one of:

- `{"manifest": "inputs/cell.csv"}`: CSV fields `frame_idx,path`, optionally
  `bytes,mtime_ns`. Each file is one ZYX TIFF or NPY volume.
- `{"paths": ["v0.tif", "v1.tif"], "frame_indices": [40,41]}`.
- `{"path": "timeseries.npy", "first_frame": 40}`: one TZYX NPY/TIFF array.
  NPY is preferable for memory mapping; compressed TIFF stacks may require
  decompression into memory.

Example using a native time-series array:

```json
{
  "schema_version": 1,
  "settings": {
    "fps": 32.5,
    "voxel_um": 0.0900811958,
    "visualize": {"speed_range": [0.0, 6.5]}
  },
  "datasets": [{
    "id": "cell_a",
    "panel": "a",
    "source": {"path": "recording.npy", "first_frame": 40},
    "ri_frame": 49,
    "flow_frames": [61, 68, 82],
    "analysis_frame_range": [40, 143],
    "time_origin": 40,
    "sections": [{"name": "xy", "plane": "xy", "index": 115}]
  }]
}
```

`ri_frame` and `flow_frames` are native indices, with each flow frame naming
the **left** frame of its adjacent pair. Input storage and time origins are
separate. Flow time is `(native_left_frame-time_origin+0.5)/fps`.
`analysis_frame_range` gives inclusive first/last **volume** indices: `[40,143]`
means 103 adjacent pairs, 40->41 through 142->143. Omitting it analyzes all
input pairs. `flow_frames` order is preserved in exported image rows.

Each section supports:

- `plane`: `xy` or `xz`; `index`: native z or displayed XZ y.
- `rotation_cw`: 0 or 90. At 90, XY is rotated clockwise and displayed XZ
  cuts native YZ at x=`index` with reversed Y and transformed flow vectors.
- `crop`: optional `[y_start,y_stop,x_start,x_stop]` in the displayed section,
  with exclusive stops. Mask generation/OLS precede the display crop.
- `mask.threshold`, `mask.sigma`, or `mask.regions` for necessary section
  settings. A region is `{"name":"upper","roi":[25,275,25,475]}`.
- `mask.path`: an explicit boolean NPY/NPZ mask or region-mask stack instead
  of generation; NPZ key defaults to `masks`, and `mask.region_names` labels
  the regions. Supplied masks use full displayed-section coordinates.
- `display_mask`: optional independent rendering-mask settings. Without it,
  rendering uses the union of the analysis masks. `{"regions": null}` uses
  the complete section's RI tissue support while preserving region masks for
  strain fitting. Section mask threshold/sigma are inherited; an explicit
  display setting overrides them. The same mask-generation function is used.
  The full display mask is saved as `*_display_mask.npz`; `mask` in
  `*_section_data.npz` records the cropped support used for speed and arrows.

Common defaults live in the library; configs override only needed settings.
Figure5 and Supplementary Figure 11 retain speed limits 0-6.5 um/s;
Extended Data Figure 3 retains 0-3.6 um/s. All share RI 0-0.06 and the same
colormaps, arrow dimensions and visibility rules. Set `settings.visualize.
speed_range` identically in configs when a common numerical color scale is
needed for a comparison. Per-image saturation fractions are recorded.

## Run

From the `DEPICT` root after installing `requirements-figures.txt`:

```sh
python -B Figures/Figure5/rebuild.py
python -B Figures/Extended_Data_Figure_3/rebuild.py
python -B Figures/Supplementary_Figure_11/rebuild.py
```

Every entry delegates to the same `pipeline.main`. For the generic package
entry point, run from the parent directory of `DEPICT`:

```sh
cd ..
python -B -m DEPICT.postprocessing.pipeline --config DEPICT/Figures/Figure5/config.json
```

Options are `--panels a b`, `--stage all|kalman|masks|flow|render|rates`,
`--workers N`, `--device auto|cpu|cuda:0`, and `--no-external-cache`.
Default `all` runs image reconstruction and numerical XY strain analysis.
`render` generates masks and clean images without rate fitting. `rates`
generates/loads masks and saves strain results without image exports.
Set a dataset's `strain_rate` to false to disable it in `all`.
After a display-mask-only change, `render` preserves numerical strain caches
and refreshes the completed `all_manifest.json` without discarding its
analysis-pair flow provenance.

## Annotated SVG Panels

Image reconstruction and numerical strain fitting remain separate from plot
export. Figure5 and Supplementary Figure 11 have a `plot_remaining.py` entry
point which reads the same adjacent `config.json` and calls `plotting.main`:

```sh
python -B Figures/Figure5/plot_remaining.py
python -B Figures/Supplementary_Figure_11/plot_remaining.py
```

The `plots` list selects `areal_strain_rate`, `speed_statistics` or `kymograph`
for each remaining panel. `source_panel` links to the existing a/b dataset;
the dataset settings and source signatures must match its completed
`all_manifest.json`. Marked frame pairs must match the image selections.
`--panels c d` limits the plot export. `plot_style` declares Arial 6 pt and
explicit local font files configured in `plot_style.font_files`.

`speed_regions` uses named analysis masks and draws each region through the
same `draw_speed` function in aligned rows with common time/speed axes. Its
`regions` list supplies each mask name, label and color; `axes_boxes_inches`
positions the rows. `time_tick_step_s` controls time ticks. By default the
window covers the actual analyzed pair range and the common speed limit is
chosen from all regions' upper quartiles. Region identity is part of each
statistics cache key. A config may set `source_config` to an existing figure
config, then choose `output_directory` and `filename` for the combined SVG,
without copying its volume paths, section indices or analysis settings.

`speed_distribution` draws regional time/speed histograms from an explicit
cached calibrated `(T,Y,X)` speed signal. `speed_cache` supplies `values`
(NPY), `alignment` (NPZ with `time_s` and `masks`) and `metadata` (onset
analysis JSON with calibration, threshold, quiet window and existing
smoothing settings). These paths resolve relative to the plot config.
The loader verifies exact mask equality and native pair-midpoint times
against the source image pipeline. `speed_bins_um_s` supplies the common
range and width; bins must include every in-mask pixel. Each frame's
percentages use all pixels in that region and sum to 100% over the complete
speed range. Strict instantaneous threshold fractions are counted from
samples independently of the bin edges, with no sustained-onset condition.
`time_window_ms` and `speed_window_um_s` only crop the display. Numerical
NPZ outputs preserve the complete time and speed ranges. Both rows share
one color normalization and colorbar; `percentage_gamma` defaults to 1
(linear), and 0.5 selects square-root color mapping with ticks still in
actual percentages. No new flow or filtering is performed.

An explicit `threshold_um_s` overrides the cached onset threshold and is
recorded as a fixed plot-config threshold. Optional `regional_onset`
settings define `minimum_percentage`, `consecutive_frames`, `after_ms`
(strictly after this reference) and `reference_region`. Regional onset is
the first native sample with sufficient above-threshold mask coverage
throughout the required run; the same individual pixels need not remain
above speed threshold. The plot marks each onset, repeats the reference
region's time on the other row and shows the timing difference. These
measurements and observed percentages are saved alongside the histograms.

`heatmap_interpolation` defaults to `cells`, which keeps rectangular vector
histogram cells. `nearest` and `bilinear` render embedded heatmap images
while keeping text, axes, colorbar and reference lines as vectors. Bilinear
display interpolates percentage values before color mapping, using regular
time and speed bins. It never changes saved numerical distributions, native
onset sample positions or the acquisition interval. In this mode an SVG
can look smooth without claiming finer temporal measurements.

`speed_cache_parameters` prepares a local speed signal from flow fields
already registered in the source image manifest. It specifies
`smoothing_sigma_t_y_x` and `quiet_window_s`; Gaussian padding is nearest
with truncate=4. Full native section vectors provide all three speed
components. Values, alignment/masks and recipe metadata are saved to the
configured `speed_cache` paths, with implementation/input identities for
reuse. A matching cache does not reopen original flow files.

`areal_strain_rate_regions` overlays named regions using `load_rate`, which
reads the existing local numerical rate NPZ and checks its masks/calibration
against the image pipeline. It reuses cached fits, optionally smooths a
display copy once and marks the configured speed-image pair midpoints.
Region colors, axes layout and panel-letter placement are config settings.

Plots with `shared_time_axis=true` inherit `plot_time_axis` from the figure
config before rendering. That shared object supplies `time_unit`,
`zero_time_s`, `time_window_ms` and `xticks_ms`. Regional rate curves and
speed-frame reference lines apply the same native-time conversion as the
heatmap; numerical cache times are preserved. Rate plot arrays additionally
save the displayed time vector, unit and zero reference.

For these two plot kinds, `cache_only=true` checks dataset agreement with
the completed local manifest and validates local caches without reopening
original raw/filtered volumes. Their `plot_manifest.json` sources identify
the local masks and numerical arrays actually read. Preparing a missing or
outdated speed signal still needs the registered original flow fields.

- Rate plots read the local `xy_strain_rate/areal_strain_rate.npz`. They do
  not refit OLS or change its sign. `display_sigma_frames=1` applies the
  reference figure's Gaussian smoothing once to a display copy, with nearest
  padding and truncate 4. The numerical strain cache is unchanged.
- Speed curves use all three cached displacement components at the configured
  section, and the same saved RI mask as a/b. Each native pair supplies one
  spatial median and p25-p75 band. The numerical statistics are cached as
  `speed_statistics.npz/csv/json` with source, mask and calibration identities.
  No optical flow is recomputed and no smoothing is applied.
- Kymographs resample the existing Kalman volume cache at the explicit
  `sampling_cache` coordinates, using bilinear interpolation and averaging
  across the saved transverse samples. Coordinates remain in native XY,
  including when the image panel is rotated. An RI baseline of 1.33 is added
  only for absolute RI display; saved numerical contrast is unchanged.

The only new figure exports are `panel_*/panel_*_with_text.svg`. Labels stay
editable SVG text; curves, axes, markers and bands remain vectors. Only the
kymograph and optional interpolated heatmap images are embedded as rasters.
The SVGs retain rectangular axes,
native midpoint labels, interval brackets and configured event shading.
No phase-name text or direction arrows are drawn. Existing clean TIFFs and
numerical strain caches are preserved. Each plot has a provenance manifest.

## Caches And Outputs

The supplied configs explicitly identify existing **Kalman-only full-volume**
caches and dense 3D flow caches, with their recipes. Their files are opened
read-only. Every requested XY z plane is independently reprocessed from the
complete raw input context and checked bit for bit against stored Kalman
frames before reuse. These checks are cached using source/cache identities
and filter settings. Full-context XY masks remain available when the stored
3D volume contains only an event window. Each used dense field's native
shape/dtype, source identity and declared recipe are checked/recorded;
finite section values are checked when rendering or fitting. Reuse does not
pretend to recompute an external flow solver or verify every undisplayed voxel.

Remove a dataset's `cache` block, or use `--no-external-cache`, to process raw
inputs entirely through this package. Only required frames are stored, but
Kalman always uses all input context. Kalman resumes at completed z planes;
flow resumes at completed native pairs. Local cache keys include parameters,
input identities and implementation identities. Temporal-mean sections are cached with input,
filter and section-coordinate identities, avoiding repeated strided reads of
the full volume when changing displayed frames. Large raw recordings need
substantial storage/memory; `--workers` controls simultaneous Kalman planes.
The default worker count is one. Dense flow uses CUDA when available and
otherwise supports CPU.

Each figure writes `panel_a/` and `panel_b/`:

- Per-section RI and selected speed **clean TIFFs only**, uint8 RGB, lossless
  deflate, 300 dpi, 1000 pixels wide. There are no labels, scale bars, axes,
  colorbars or sampling/section-marker overlays.
- `panel_a_xy_clean.tif` and optional XZ row, plus `panel_a_clean.tif` combining
  all section rows; corresponding files are supplied for b.
- `*_masks.npz`, `*_section_data.npz`, numerical `*_strain_rate/` outputs and
  the stage manifest with inputs, parameters, geometry statistics and code
  identities. Original float32 RI, flow and speed values remain numerical
  arrays; display normalization never changes those values.

Supplementary Figure 11's a/b image rows show reference and no-drug data.
The reference uses a 104-frame Kalman context and an 80-frame analysis window.
Extended Data Figure 3's first displayed speed pair is 23->24.

## Direct Numerical API And Validation

```python
from DEPICT.postprocessing.kalman_filter import kalman_rts_smooth
from DEPICT.postprocessing.optical_flow import dense_flow
from DEPICT.postprocessing.masks import generate_masks
from DEPICT.postprocessing.areal_strain_rate import analyze_and_save

# stack is (T,Y,X); volumes are ZYX; flow sections are (3,Y,X).
filtered_plane = kalman_rts_smooth(stack)
field = dense_flow(reference_volume, moving_volume, device="cuda:0")
region_masks, names = generate_masks(mean_ri_section)
analyze_and_save(flow_sections, region_masks, native_left_frames=left_frames,
                 output="rate_results", fps=32.5, region_names=names)
```

Run the scientific and raw-input integration checks from the parent of
`DEPICT`:

```sh
python -B -m unittest discover -s DEPICT/postprocessing/tests -v
```

Checks cover the established Kalman implementation, skimage TV-L1 agreement
and translation axes, affine-field strain sign, native midpoint times,
RI masks/holes/ROIs, clockwise scalar/vector transforms, streamline containment,
minimum-length rejection, rectangular streamlines and a
complete raw TZYX -> Kalman -> dense flow -> masks -> clean TIFF/strain run
with external caches disabled.

From the parent of `DEPICT`, check figure exports and write a local
`validation_report.json`:

```sh
python -B -m DEPICT.postprocessing.validate --config DEPICT/Figures/Figure5/config.json
```

This verifies input/cache identities, RI/flow/speed arrays, mask construction,
independently sampled streamline/head containment, affine fits using an
independent covariance solution and clean RGB TIFF/300 dpi exports.
