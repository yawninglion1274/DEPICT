# DEPICT

This repository implements three reconstruction methods for optical diffraction
tomography: **Rytov**, **TV-BPM**, and **DEPICT**. DEPICT combines a diffusion
prior with a beam propagation forward model to reconstruct three-dimensional
refractive-index distributions from optical wavefront measurements.

Six example datasets and a pretrained model are provided on
[Zenodo](https://doi.org/10.5281/zenodo.24165471) for reproducing the
reconstructions and getting started with the code.

## 1. Installation

Use Python 3.9 and a CUDA-capable NVIDIA GPU. Run the following commands from
the repository root:

```bash
conda create -n depict python=3.9
conda activate depict
python -m pip install -r requirements.txt
```

## 2. Example Data And Model

Create the local directories for the model weights, observations, and outputs.
These contents are not included in the GitHub repository:

```bash
mkdir -p models complex_field results
```

Download the example observations and pretrained model from
[Zenodo (DOI: 10.5281/zenodo.24165471)](https://doi.org/10.5281/zenodo.24165471).
The observations are preprocessed sample/background amplitudes and phases,
with illumination coordinates.

| Archive | Contents | Location in this repository |
| --- | --- | --- |
| `DEPICT_models.tar` | Pretrained diffusion model | `models/` |
| `293t_commercial.tar` | HEK293T observations | `complex_field/293t_commercial/` |
| `hela_commercial.tar` | HeLa observations | `complex_field/hela_commercial/` |
| `cos7_commercial.tar` | COS7 observations | `complex_field/cos7_commercial/` |
| `u2os_commercial.tar` | U2OS observations | `complex_field/u2os_commercial/` |
| `T_cell.tar` | Interphase and mitotic T-cell observations | `complex_field/T_cell/` |

Extract `DEPICT_models.tar` and place `3_cells_iteration_230000ema.pt` directly
in `models/`, keeping its filename unchanged.
Extract the observation archives and place the sample folders in
`complex_field/`, preserving their directory names and filenames.
Before running the scripts, the directory layout should be:

```text
models/
  3_cells_iteration_230000ema.pt
complex_field/
  293t_commercial/241219/
  hela_commercial/hela_1/
  cos7_commercial/cos7_1111/
  u2os_commercial/u2os_1109/
  T_cell/20260611_0p5_full1910/
results/
```

`results/` is initially empty; the scripts create output subdirectories as
needed. The model path must match `model_dir` in
`configs/model_config/3_cells_230000ema.yaml`.

Each acquisition directory contains `sp_amp/`, `sp_phs/`, `bg_amp/`, `bg_phs/`
and `k_scan/`, with files named `<data_name>.npy`. The amplitude and phase
arrays have shape `(observations, height, width)`; `k_scan` has shape
`(observations, 2)`.

| Sample ID | Cell sample | Data configuration in `configs/isorecon_data_config/` |
| --- | --- | --- |
| `293t` | HEK293T | `old_293t_170932.yaml` |
| `hela` | HeLa | `hela-21-17-21.yaml` |
| `cos7` | COS7 | `cos7_223610.yaml` |
| `u2os` | U2OS | `u2os_test.yaml` |
| `tcell` | Interphase T cell (`sam11_1910`) | `20260611_T_0.5_11.yaml` |
| `tcell_mitosis` | Mitotic T cell (`sam2_1910`) | `20260611_T_0.5_2.yaml` |

The four commercial acquisitions contain 240 observations each; the T-cell
acquisitions contain 1910 each. The default reconstructions use all 240
commercial observations and the first 240 observations of each T-cell sample.
The supplied configurations select the spatial crops and optical parameters.

## 3. Reconstruct The Six Samples

All commands below run from the repository root with the `depict` environment
active. First validate the input paths, array shapes and crop bounds without
starting a reconstruction:

```bash
DRY_RUN=1 bash scripts/run_bpm.sh
DRY_RUN=1 bash scripts/run_depict.sh
```

Both checks should report `Dry run complete`. Then run the launchers below;
`GPUS=0` selects the first GPU. Each launcher computes its initialization
directly from the observations.

### Rytov And TV-BPM

Run the Rytov initialization followed by TV-regularized BPM optimization:

```bash
GPUS=0 bash scripts/run_bpm.sh
```

This saves both the Rytov volume and the final TV-BPM volume for every sample.
The TV-BPM settings are in
`configs/reconstruction_config/beam_propagation/TV.yaml`: 200 iterations,
step size `0.0005`, and TV weight `5e-6`.

### DEPICT

Run the diffusion-prior reconstruction:

```bash
GPUS=0 bash scripts/run_depict.sh
```

All six samples use the pretrained model configured in
`configs/model_config/3_cells_230000ema.yaml`. The launcher selects the optical
and optimization settings for each sample.

### Select Samples And GPUs

Both launchers accept `SAMPLES` and `GPUS` as space-separated lists. For
example, reconstruct only HEK293T with all three methods:

```bash
SAMPLES=293t GPUS=0 bash scripts/run_bpm.sh
SAMPLES=293t GPUS=0 bash scripts/run_depict.sh
```

To distribute all six samples across several GPUs:

```bash
GPUS="0 1 2" bash scripts/run_bpm.sh
GPUS="0 1 2" bash scripts/run_depict.sh
```

Each launcher runs at most one job per listed GPU, waits for all jobs to
finish, and reports any failures. Set `LOG_DIR` to choose a log directory.

DEPICT also supports 24-observation reconstructions:

```bash
ANGLE_GROUPS=24 GPUS=0 bash scripts/run_depict.sh
```

## Outputs

Results are saved under `results/`, following each data configuration's
`expriment_name`, `save_name`, and `data_name`. The volumes store refractive-index
contrast, `delta_n = n - n_medium`, in `(z, x, y)` order.
The supplied example configurations reconstruct volumes of shape
`(256, 500, 500)`.

| Method | Output relative to `results/<expriment_name>/<save_name>/` |
| --- | --- |
| Rytov | `<data_name>/rytov.tif` |
| TV-BPM | `bpm_regu/<data_name>.tif` |
| DEPICT | `<data_name>/adam_REDdiff_.../<model>/<settings>/<settings>/iso.tif` |

TV-BPM checkpoints (`bpm_regu_step*.tif`) and DEPICT checkpoints (`iso_*.tif`)
are saved in their parameter directories. Each solver prints its output paths;
the launchers write per-sample logs under `log/`.

## Use Your Own Measurements

Follow the example array layout, add a data YAML under
`configs/isorecon_data_config/`, and specify the matching optical parameters
under `configs/physics_config/`. Paths in YAML files are relative to the
repository root. For a direct TV-BPM run:

```bash
python experiment_solve_bpm_clean_style.py \
    --data_config configs/isorecon_data_config/old_293t_170932.yaml \
    --physics_config configs/physics_config/commercial_ODT.yaml \
    --reconstruction_config configs/reconstruction_config/beam_propagation/TV.yaml
```

Use `python adam_REDDiff_resample_BPM_resample_triplane_angle_group.py --help`
for the DEPICT solver's arguments. The batch launcher's dry run prints the
complete command for each example sample.
