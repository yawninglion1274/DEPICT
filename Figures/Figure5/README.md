# Figure 5

`rebuild.py` only selects `config.json` and calls `DEPICT.postprocessing.pipeline`.
Native input paths and frame identities are recorded in `inputs/*.csv`.
All processing and mask/arrow/render implementations live in the shared library.

Run from the `DEPICT` root after installing `requirements-figures.txt`:

```sh
python -B Figures/Figure5/rebuild.py
```

The default `all` stage writes only clean RGB TIFF image exports, plus masks,
numerical section/strain arrays, CSV tables and provenance JSON. No annotated
figures or strain-rate plots are generated. See `../../postprocessing/README.md`
for the config schema, cache validation, processing settings and stage options.

Rebuild the remaining annotated SVG panels:

```sh
python -B Figures/Figure5/plot_remaining.py
```

The `plots` and `plot_style` entries in `config.json` supply parameters to the
shared `DEPICT.postprocessing.plotting` implementation. The entry point has no
plotting or analysis logic. `--panels c d` exports only the requested panels.

- `panel_c/panel_c_with_text.svg`: panel a's cached areal strain rate.
- `panel_d/panel_d_with_text.svg`: panel b's cached areal strain rate.
- `panel_e/panel_e_with_text.svg`: two fixed-coordinate Kalman RI kymographs.

Presentation uses Arial 6 pt, rectangular axes, native image-pair midpoint
markers, event shading,
interval brackets, and e1/e2 with a common absolute RI colorbar (1.33-1.43).
No phase-name text or direction arrows are drawn. Rate sign is positive
divergence, `+fps*trace(B)`. Local cached numerical rates are read unchanged;
one Gaussian sigma=1 frame pass is applied to the display curve. Event
intervals use minima within the configured native-frame windows. Panel d's
two cyan spans use the configured interval bounds.

Kymograph coordinates come from the reference NPZ files and are resampled from
the same cached Kalman volumes used by a/b, with no display rotation of native
sampling coordinates. Sampled numerical arrays and plot provenance are saved
beside the SVG. The only image exports from this entry point are annotated
SVGs with live text. Existing a/b TIFFs and strain caches are preserved.
