# Data And Models

The GitHub repository contains source code, configuration, documentation and
small CSV tables. Observation arrays, model checkpoints, reconstructed volumes
and image exports are distributed separately. A public Hugging Face dataset
or model repository has not yet been assigned.

## Reconstruction Inputs

Restore the observation collection into `complex_field/`, preserving this
layout for each configured acquisition:

```text
complex_field/<acquisition>/
  sp_amp/<data_name>.npy
  sp_phs/<data_name>.npy
  bg_amp/<data_name>.npy
  bg_phs/<data_name>.npy
  k_scan/<data_name>.npy
```

The four amplitude/phase arrays contain float32 data with shape
`(observations, height, width)`. `k_scan` has shape `(observations, 2)`.

| Samples | Observations | Height x width |
| --- | ---: | --- |
| HEK293T, HeLa, COS7, U2OS | 240 each | 1200 x 1200 |
| Interphase and mitotic T cells | 1910 each | 532 x 532 |

These are preprocessed wavefront measurements. Camera holograms and
reconstructed refractive-index volumes are different data products.
Sample paths and selection rules are in [Reconstruction](reconstruction.md).

## Checkpoints

The model YAMLs expect:

```text
models/
  3_cells_iteration_230000ema.pt
  iteration_200000ema.pt
```

Checkpoints serialize Diffusers `UNet2DModel` objects. Use the PyTorch and
Diffusers versions in `requirements.txt` and load checkpoints from a trusted
source.

## Figure Inputs

| Component | Required data |
| --- | --- |
| Figure 3 | `Figures/Figure3/inputs/volumes/` and shared lipid inputs |
| Supplementary Figure 6 | `Figures/Supplementary_Figure_6/inputs/volumes/` and shared HeLa volume |
| Supplementary Figure 8 | `Figures/Supplementary_Figure_8/inputs/volumes/` and shared lipid inputs |
| Figure 4 and Supplementary Figures 9-10 | Localization CSVs in `inputs/angle*/`; optional precomputed `analysis/` tables |
| Figure 5, Supplementary Figure 11, Extended Data Figure 3 | Native RI time series listed in each figure's `inputs/*.csv` |

`Figures/shared_inputs/lipid_isotropy/` contains the common 240-angle volumes
and coordinate/measurement tables. Keep the relative `inputs/lipid_isotropy`
links in Figure 3 and Supplementary Figure 8 pointing to this shared directory.

The native time-series CSVs for Figure 5, Supplementary Figure 11 and Extended
Data Figure 3 currently refer to local recordings outside this repository.
Before using these figures on another machine, set each CSV's `path` field to
your copy of the corresponding frame. Relative paths resolve from the CSV's
directory. Preserve `frame_idx`; update or omit optional `bytes` and `mtime_ns`
fields after copying inputs. The supplied configs also reference external
Kalman and flow caches. Remove their `cache` entries or run with
`--no-external-cache` to compute from the native inputs.

Annotated panels use Arial fonts. Install Arial locally and set the figure's
`plot_style.font_files` to your font files, or select an installed family with
`plot_style.font_family`. Font binaries are excluded from Git. The trajectory
plots expect a local font at
`Figures/Figure4_Supplementary_Figure_9_10/inputs/fonts/Arial.TTF`.

## Download From Hugging Face

After the dataset and model repositories are published, set their identifiers
and run from the `DEPICT` root:

```bash
export HF_DATASET_REPO="your-account/your-dataset"
export HF_MODEL_REPO="your-account/your-models"
python -m pip install -U huggingface_hub
```

```python
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["HF_DATASET_REPO"],
    repo_type="dataset",
    local_dir=".",
)
snapshot_download(
    repo_id=os.environ["HF_MODEL_REPO"],
    repo_type="model",
    local_dir="models",
    allow_patterns=["*.pt"],
)
```

The dataset should preserve repository-relative paths for `complex_field/`
and `Figures/`. Keep generated outputs under `results/` and logs under `log/`;
neither directory is part of the source release.
