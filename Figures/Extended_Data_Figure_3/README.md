# Extended Data Figure 3

Reproduce RI images, speed fields, regional speed distributions and areal
strain rates from native volumetric time series. Input recordings are listed
in `inputs/*.csv`; `config.json` specifies frames, calibration, masks and
rendering settings. Restore or configure the inputs described in
[Data](../../docs/data.md) first.

Run from the `DEPICT` root after installing `requirements-figures.txt`:

```bash
python -B Figures/Extended_Data_Figure_3/rebuild.py
python -B Figures/Extended_Data_Figure_3/plot_remaining.py
```

`rebuild.py` calls the shared postprocessing pipeline and writes RI/speed RGB
TIFFs, masks, numerical section/strain arrays and CSV tables. Use
`--panels b --stage render` to render panel b from matching caches.
`plot_remaining.py --panels c` selects one annotated SVG panel.

| Panel | Content | Output |
| --- | --- | --- |
| a | RI and flow sections | RGB TIFFs, masks and numerical arrays |
| b | RI and flow sections with a connected display mask | RGB TIFFs, masks and strain-rate arrays |
| c | Upper/lower regional speed distributions and onset times | `panel_c/panel_c_with_text.svg` |
| d | Upper/lower areal strain rates | `panel_d/panel_d_with_text.svg` |

## Masks And Speed Signals

Panel b uses a complete connected RI tissue mask for its speed background
and streamlines. Separate upper/lower ROI masks define the quantitative
analysis regions. `panel_b/xy_display_mask.npz` stores the full display mask;
`panel_b/xy_masks.npz` stores the analysis masks. The section-data `mask` is
the cropped display support.

Panels c/d use the same upper/lower analysis masks, with 70372/57837 pixels.
Panel c reads `panel_b/cache/speed/xy/speed_um_s.npy` with adjacent alignment
and signal metadata. The signal uses full 3D speed magnitudes at native z=148
for 45 flow pairs, 58->59 through 102->103. Gaussian smoothing uses sigma
`(time,y,x)=(0.8,6,6)`, nearest padding and truncate 4, applied once.

A matching local cache permits `cache_only=true` rendering without reopening
raw volumes or external flow files. Input settings, masks, calibration and
native times must match the cache recipe. Rebuilding an absent or stale speed
cache requires the corresponding flow fields.

## Regional Onset And Strain

Regional onset is the first post-zero native sample for which at least 5% of
region pixels have speed strictly above 0.50 um/s for two consecutive samples.
The time origin is 1.800 s. Upper/lower onsets are 30.769231/92.307692 ms, with
19.759279%/6.663554% coverage at onset. The upper region precedes the lower by
61.538462 ms. Speed and coverage thresholds are separate configuration values.

Panel c displays -150 to 400 ms relative to the time origin; absolute axis
labels span 1.7-2.2 s. Its heatmap is embedded at 300 dpi, while text, axes,
colorbar, onset markers and brackets remain editable SVG elements.

Panel d reads `panel_b/xy_strain_rate/areal_strain_rate.npz`. Areal strain
rate is positive divergence, `+fps*trace(B)`, from affine OLS fits. Display
smoothing uses sigma=1 frame, nearest padding and truncate 4. The plot spans
1.492308-2.846154 s, with speed-frame markers at 1.892308, 1.984615 and
2.292308 s. Curves and text are vector elements.

See the [postprocessing guide](../../postprocessing/README.md) for the full
configuration schema, processing stages and numerical validation commands.
