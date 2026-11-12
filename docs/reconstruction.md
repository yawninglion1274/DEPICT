# Reconstruction

Run commands from the `DEPICT` root. The batch launcher reconstructs six cell
samples at 240 and 24 illumination angles using a diffusion prior and a beam
propagation forward model.

## Inputs

Place observation arrays under `complex_field/` and model checkpoints under
`models/`, as described in [Data](data.md). Paths in the reconstruction YAMLs
are relative to the repository root.

| Sample ID | Data YAML in `configs/isorecon_data_config/` | Input directory under `complex_field/` |
| --- | --- | --- |
| `293t` | `old_293t_170932.yaml` | `293t_commercial/241219` |
| `hela` | `hela-21-17-21.yaml` | `hela_commercial/hela_1` |
| `cos7` | `cos7_223610.yaml` | `cos7_commercial/cos7_1111` |
| `u2os` | `u2os_test.yaml` | `u2os_commercial/u2os_1109` |
| `tcell` | `20260611_T_0.5_11.yaml` | `T_cell/20260611_0p5_full1910`, sample `sam11_1910` |
| `tcell_mitosis` | `20260611_T_0.5_2.yaml` (240 angles), `20260611_T_0.5_2_time.yaml` (24 angles) | `T_cell/20260611_0p5_full1910`, sample `sam2_1910` |

Each sample has five arrays, named `<data_name>.npy`, in the `sp_amp`, `sp_phs`,
`bg_amp`, `bg_phs` and `k_scan` subdirectories. Amplitude and phase arrays have
shape `(observations, height, width)`; illumination coordinates have shape
`(observations, 2)`.

Commercial samples use `commercial_ODT.yaml`, `raw_size: 2400` and
`parent_size: 1200`, with 0.09 um reconstruction voxels. T cells use `lzr.yaml`,
`raw_size: 1024` and `parent_size: 532`. The COS7 crop shift is `[239, -117]`.
Both T-cell samples use `shift_idx: [-15, 0]`. All samples use `ROI: null`,
so the prior operates on the full reconstruction volume.

## Sample Settings

Model `3` selects `configs/model_config/3_cells_230000ema.yaml`; model `4`
selects `configs/model_config/4_cells_200000ema.yaml`. All jobs use Rytov
initialization from the selected observations.

| Sample | Angles | Selection / `frame_idx` | Model | `lambda` | Step size | `ave_start` |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 293t | 240 | all / 0 | 3 | 0.03 | 0.0018 | 75 |
| 293t | 24 | uniform / 0 | 3 | 0.02 | 0.002 | 100 |
| hela | 240 | all / 0 | 4 | 0.03 | 0.0018 | 75 |
| hela | 24 | uniform / 0 | 3 | 0.03 | 0.002 | 100 |
| cos7 | 240 | all / 0 | 4 | 0.03 | 0.002 | 75 |
| cos7 | 24 | uniform / 0 | 3 | 0.03 | 0.002 | 100 |
| u2os | 240 | all / 0 | 4 | 0.03 | 0.002 | 75 |
| u2os | 24 | uniform / 5 | 3 | 0.03 | 0.002 | 100 |
| tcell | 240 | contiguous / 0 | 3 | 0.015 | 0.0025 | 100 |
| tcell | 24 | contiguous / 40 | 3 | 0.03 | 0.002 | 100 |
| tcell_mitosis | 240 | contiguous / 0 | 3 | 0.015 | 0.0025 | 100 |
| tcell_mitosis | 24 | contiguous / 40 | 3 | 0.03 | 0.002 | 100 |

Uniform selection (`sparse_view`) uses indices `0,10,...,230`, except for U2OS,
which uses `5,15,...,235`. Contiguous selection (`fast_acq`) uses
`[frame_idx * angles : (frame_idx + 1) * angles]`: `[0:240]` for the default
240-angle T-cell jobs and `[960:984]` for the 24-angle jobs. The actual input
length determines valid selections; `lzr.yaml`'s `angle_num` does not truncate
the 1910-observation inputs.

All jobs use square-root coefficients, `resample_times=3`, `resample_end=0`,
`data_weight=1` and a diffusion schedule from `start_time=999` to zero.
`resample_start=100` for 240-angle T-cell jobs and 75 otherwise.
Prior/BPM batch sizes are 15/3 for commercial 240-angle jobs, 80/6 for T-cell
240-angle jobs and 22/6 for 24-angle jobs.

## Run

Preview all 12 jobs and validate NumPy headers, crop bounds and model paths:

```bash
DRY_RUN=1 bash scripts/solve_adam_reddiff_tri_resample_angle_group_grid_search.sh
```

Check preprocessing and Rytov initialization for one sample on the CPU:

```bash
SAMPLES=293t ANGLE_GROUPS=24 PRE_RECON_CHECK=1 bash scripts/solve_adam_reddiff_tri_resample_angle_group_grid_search.sh
```

Run one reconstruction on GPU 0:

```bash
SAMPLES=293t ANGLE_GROUPS=24 GPUS=0 bash scripts/solve_adam_reddiff_tri_resample_angle_group_grid_search.sh
```

Run all samples at both angle counts on four GPUs:

```bash
GPUS="0 1 2 3" bash scripts/solve_adam_reddiff_tri_resample_angle_group_grid_search.sh
```

The launcher's default GPU list is `4 5 6 7`. It runs at most one job per GPU
and returns a nonzero exit status if any job fails. Normal reconstruction
requires CUDA. Logs are written under `log/six_cells_<timestamp>_<pid>/`.
Outputs are written under the YAML's `save_path` (`results/` by default), with
the angle selection and optimization settings in the directory name.
`iso.tif` is the final reconstruction; `iso_4.tif` and `iso_9.tif` are
intermediate outputs.

## Launcher Options

| Variable | Purpose |
| --- | --- |
| `SAMPLES` | Space-separated sample IDs |
| `ANGLE_GROUPS` | `240`, `24`, or both |
| `GPUS` | Space-separated visible GPU indices |
| `START_TIME` | First diffusion timestep, default 999 |
| `PRIOR_BATCH_SIZE` | Override the prior batch size |
| `T240_FRAME_IDX`, `T24_FRAME_IDX` | T-cell contiguous selection offsets |
| `LOG_DIR` | Output log directory |
| `PYTHON` | Python interpreter |
| `CONDA_SH`, `CONDA_ENV` | Optional Conda activation script and environment |

For Slurm, adapt the launcher's resource directives to your cluster, create
`log/`, and submit from the repository root:

```bash
mkdir -p log
sbatch scripts/solve_adam_reddiff_tri_resample_angle_group_grid_search.sh
```
