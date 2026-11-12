# DEPICT

DEPICT reconstructs three-dimensional refractive-index volumes from optical
wavefront measurements using a diffusion prior and a beam propagation forward
model. The repository also includes figure reproduction, lipid-droplet motion
analysis, and volumetric postprocessing with Kalman smoothing and 3D optical
flow.

## Installation

Use Python 3.9. Run the commands below from the `DEPICT` directory. Keep that
directory name when cloning: figure entry points import the `DEPICT` package
from its parent directory.

```bash
conda create -n depict python=3.9
conda activate depict
python -m pip install -r requirements.txt
```

Reconstruction requires a CUDA-capable GPU. The launcher can validate input
paths and array headers without a GPU, and its preprocessing check runs on
the CPU. Install volume-figure and postprocessing dependencies when needed:

```bash
python -m pip install -r requirements-figures.txt
```

Figure 4 and Supplementary Figures 9-10 use a separate Python 3.9 environment:

```bash
conda create -n depict-tracking python=3.9
conda activate depict-tracking
python -m pip install -r Figures/Figure4_Supplementary_Figure_9_10/requirements.txt
```

LapTrack requires `scikit-image==0.19.3` in this environment; volumetric
postprocessing uses `scikit-image==0.24.0`. Keep these environments separate.

Component-specific dependency files are also available in
`figure_reproduction/`, `postprocessing/` and the relevant figure directories.

## Data And Models

Observation arrays, checkpoints and volume/image assets are separate from the
source repository. Place observations in `complex_field/`, model weights in
`models/`, and figure inputs under `Figures/`, preserving their configured
relative paths. See [Data and models](docs/data.md) for layouts, array shapes
and input requirements. Public download repositories have not yet been
assigned.

## Reconstruction

Preview the six-sample, two-angle-count batch and check inputs:

```bash
conda activate depict
DRY_RUN=1 bash scripts/solve_adam_reddiff_tri_resample_angle_group_grid_search.sh
```

Reconstruct one sample with 24 angles on GPU 0:

```bash
SAMPLES=293t ANGLE_GROUPS=24 GPUS=0 bash scripts/solve_adam_reddiff_tri_resample_angle_group_grid_search.sh
```

Select samples, GPUs and angle counts through environment variables. Results
are written under `results/` and logs under `log/` by default. See
[Reconstruction](docs/reconstruction.md) for sample configurations, observation
selection, optimization settings and Slurm usage.

## Figure Reproduction

Run figure entry points from the `DEPICT` root after installing the figure
dependencies and restoring their inputs:

```bash
python Figures/Figure3/rebuild.py
python Figures/Supplementary_Figure_8/rebuild.py --panels d h
```

For trajectory analysis, activate `depict-tracking` and run
`python Figures/Figure4_Supplementary_Figure_9_10/rebuild.py`.

| Component | Documentation |
| --- | --- |
| Figure 3: cell volumes and lipid isotropy | [Figure 3](Figures/Figure3/README.md) |
| Figure 4 d-f and Supplementary Figures 9-10: lipid trajectories | [Trajectory analysis](Figures/Figure4_Supplementary_Figure_9_10/README.md) |
| Supplementary Figure 6: forward-model comparisons | [Supplementary Figure 6](Figures/Supplementary_Figure_6/README.md) |
| Supplementary Figure 8: sparse-angle comparisons | [Supplementary Figure 8](Figures/Supplementary_Figure_8/README.md) |
| Figure 5: volumetric motion and strain | [Figure 5](Figures/Figure5/README.md) |
| Supplementary Figure 11: motion statistics | [Supplementary Figure 11](Figures/Supplementary_Figure_11/README.md) |
| Extended Data Figure 3: regional dynamics | [Extended Data Figure 3](Figures/Extended_Data_Figure_3/README.md) |

The [postprocessing guide](postprocessing/README.md) describes the numerical
methods, configuration schema, cache handling and validation commands. Figure
5, Supplementary Figure 11 and Extended Data Figure 3 require native time
series outside this repository; configure their input CSVs before running on a
new machine.

## Repository Layout

```text
configs/                 Optical, model, data and reconstruction settings
src/                     Reconstruction operators and optimization routines
utils/                   Data loading and image utilities
scripts/                 Reconstruction batch launcher
figure_reproduction/     Shared volume rendering and FWHM analysis
postprocessing/          Kalman, optical flow, masks and strain analysis
Figures/                 Figure entry points, configurations and small tables
docs/                    Reconstruction, data and publication guides
```

`manifest/`, `complex_field/`, `results/` and `log/` are local directories
excluded from GitHub. Model weights, binary figure assets and generated caches
are excluded as well. See [Publishing](docs/publishing.md) for the code/data
split and release commands.

## Contributing And Support

Use GitHub issues to report problems. Include the command, Python environment,
relevant configuration and error output. Changes to numerical methods should
describe their expected effect and include an appropriate validation case.

## License

A project license has not yet been selected.
