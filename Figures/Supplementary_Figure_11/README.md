# Supplementary Figure 11

`rebuild.py` only selects `config.json` and calls `DEPICT.postprocessing.pipeline`.
Native input paths and frame identities are recorded in `inputs/*.csv`.
All processing and mask/arrow/render implementations live in the shared library.

Run from the `DEPICT` root after installing `requirements-figures.txt`:

```sh
python -B Figures/Supplementary_Figure_11/rebuild.py
```

The default `all` stage writes only clean RGB TIFF image exports, plus masks,
numerical section/strain arrays, CSV tables and provenance JSON. No annotated
figures or strain-rate plots are generated. See `../../postprocessing/README.md`
for the config schema, cache validation, processing settings and stage options.

Output panels a/b are the reference/no-drug image rows.
Reference filtering uses all 104 raw frames
with the same Kalman-only cache as Figure5, while analysis uses frames 0-79.

Rebuild the remaining annotated SVG speed panels:

```sh
python -B Figures/Supplementary_Figure_11/plot_remaining.py
```

`config.json` supplies `plots` and `plot_style` to the shared plotting library;
the entry point contains no analysis or drawing implementation.

- `panel_c/panel_c_with_text.svg`: reference, Treated with isoprenaline.
- `panel_d/panel_d_with_text.svg`: no-drug, Untreated.

The SVGs follow the reference PDF's teal/orange median curves, spatial p25-p75
bands, rectangular axes, Arial 6 pt labels, common time and speed limits, and
native image-pair midpoint markers. The selected pairs remain 8->9, 12->13 for
c, and 117->118, 124->125 for d. No smoothing is applied.

Statistics use the shared pipeline's full 3D displacement caches and
the same fixed RI masks as a/b. Reference data use Kalman temporal smoothing
before flow estimation.
Each curve has 79 actual pair samples. Numerical p25/median/p75 values are
cached beside the SVG with input, mask and calibration identities. The only
image exports are annotated SVGs; existing clean TIFFs and strain caches are
preserved. `--panels c` selects a single plot.
