#!/bin/bash
#SBATCH -o log/depict_six_cells_%j.out
#SBATCH --partition=GPUA40
#SBATCH --job-name=depict_six_cells
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --mem=120G

set -euo pipefail

# Slurm runs a spooled copy; submit from the DEPICT root in that case.
solver="adam_REDDiff_resample_BPM_resample_triplane_angle_group.py"
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "${project_root}/${solver}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    project_root="${SLURM_SUBMIT_DIR}"
fi
cd -- "${project_root}"

if [[ -n "${CONDA_SH:-}" ]]; then
    source "${CONDA_SH}"
fi
if [[ -n "${CONDA_ENV:-}" ]]; then
    conda activate "${CONDA_ENV}"
fi
python_bin="${PYTHON:-python}"

is_true() {
    case "$1" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

dry_run="${DRY_RUN:-0}"
pre_recon_check="${PRE_RECON_CHECK:-0}"
start_time="${START_TIME:-999}"
read -r -a samples <<< "${SAMPLES:-293t hela cos7 u2os tcell_mitosis tcell}"
read -r -a angle_groups <<< "${ANGLE_GROUPS:-240 24}"
read -r -a gpus <<< "${GPUS:-4 5 6 7}"
if (( ${#samples[@]} == 0 || ${#angle_groups[@]} == 0 || ${#gpus[@]} == 0 )); then
    echo 'SAMPLES, ANGLE_GROUPS and GPUS must not be empty.' >&2
    exit 1
fi
for gpu in "${gpus[@]}"; do
    if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU index: ${gpu}" >&2
        exit 1
    fi
done

declare -A data_configs=(
    [293t]='configs/isorecon_data_config/old_293t_170932.yaml'
    [hela]='configs/isorecon_data_config/hela-21-17-21.yaml'
    [cos7]='configs/isorecon_data_config/cos7_223610.yaml'
    [u2os]='configs/isorecon_data_config/u2os_test.yaml'
    [tcell]='configs/isorecon_data_config/20260611_T_0.5_11.yaml'
    [tcell_mitosis]='configs/isorecon_data_config/20260611_T_0.5_2_time.yaml'
)

# Parameter definitions and sample settings are in ../docs/reconstruction.md.
# Commercial 24-angle jobs take every tenth observation (U2OS offset 5, others 0).
resolve_job() {
    local sample=$1 angles=$2
    data_config="${data_configs[$sample]}"
    physics_config='configs/physics_config/commercial_ODT.yaml'
    model_config='configs/model_config/3_cells_230000ema.yaml'
    reconstruction_config='configs/reconstruction_config/DiffPIR_FISTA/adam_RED.yaml'
    lambda_scale='0.03'
    step_size='0.002'
    angle_group_mode='sparse_view'
    frame_idx=0
    prior_batch_size=22
    bpm_batch_size=6
    resample_start=75

    if [[ "$sample" == tcell || "$sample" == tcell_mitosis ]]; then
        physics_config='configs/physics_config/lzr.yaml'
        angle_group_mode='fast_acq'
        if (( angles == 240 )); then
            frame_idx="${T240_FRAME_IDX:-0}"
            step_size='0.0025'
            lambda_scale='0.015'
            resample_start=100
            prior_batch_size=80
            if [[ "$sample" == tcell_mitosis ]]; then
                data_config='configs/isorecon_data_config/20260611_T_0.5_2.yaml'
            fi
        else
            frame_idx="${T24_FRAME_IDX:-40}"
        fi
    elif (( angles == 240 )); then
        reconstruction_config='configs/reconstruction_config/DiffPIR_FISTA/adam_RED_240.yaml'
        prior_batch_size=15
        bpm_batch_size=3
        if [[ "$sample" != 293t ]]; then
            model_config='configs/model_config/4_cells_200000ema.yaml'
        fi
        if [[ "$sample" == 293t || "$sample" == hela ]]; then
            step_size='0.0018'
        fi
    elif [[ "$sample" == 293t ]]; then
        lambda_scale='0.02'
    elif [[ "$sample" == u2os ]]; then
        frame_idx=5
    fi
    prior_batch_size="${PRIOR_BATCH_SIZE:-$prior_batch_size}"
}

# Inspect headers only: all inputs and crop bounds must be valid before any job.
validate_inputs() {
    "$python_bin" - "$data_config" "$physics_config" "$model_config" "$reconstruction_config" "$start_time" <<'PY'
from pathlib import Path
import sys
import numpy as np
import yaml

configs = [yaml.safe_load(Path(name).read_text()) for name in sys.argv[1:5]]
data, physics, model, reconstruction = configs
if not Path(model['model_dir']).is_file():
    raise FileNotFoundError(model['model_dir'])
if not 0 <= int(sys.argv[5]) < model['num_inference']:
    raise ValueError('START_TIME must be within the model diffusion schedule')
arrays = {
    kind: np.load(Path(data['data_path']) / kind / (data['data_name'] + '.npy'), mmap_mode='r')
    for kind in ['sp_amp', 'sp_phs', 'bg_amp', 'bg_phs', 'k_scan']
}
shape = arrays['sp_amp'].shape
if len(shape) != 3 or any(arrays[kind].shape != shape for kind in ['sp_phs', 'bg_amp', 'bg_phs']):
    raise ValueError(f'Inconsistent amplitude/phase shapes: {data["data_name"]}')
if arrays['k_scan'].ndim != 2 or arrays['k_scan'].shape[0] != shape[0] or arrays['k_scan'].shape[1] < 2:
    raise ValueError('k_scan must have shape (observations, >=2)')
center = data['parent_size'] // 2
if center == data['spec_size'][0]:
    if list(shape[1:]) != data['spec_size']:
        raise ValueError('Uncropped field shape differs from spec_size')
else:
    for axis, size in enumerate(data['spec_size']):
        begin = center - size // 2 + data['shift_idx'][axis]
        end = center + size // 2 + data['shift_idx'][axis]
        if not 0 <= begin < end <= shape[axis + 1] or end - begin != size:
            raise ValueError(f'Crop exceeds field: axis={axis}, [{begin}:{end}], shape={shape}')
if data['domain_size'][1:] != data['spec_size'] or data['raw_size'] <= 0:
    raise ValueError('Invalid domain_size/spec_size/raw_size')
print(shape[0])
PY
}

declare -a job_samples=() job_angles=()
for sample in "${samples[@]}"; do
    if [[ -z "${data_configs[$sample]:-}" ]]; then
        echo "Unknown sample: ${sample}" >&2
        exit 1
    fi
done
# Queue jobs by angle count before assigning them to GPU slots.
for angles in "${angle_groups[@]}"; do
    if [[ "$angles" != 240 && "$angles" != 24 ]]; then
        echo "Unsupported angle count: ${angles}; expected 240 or 24." >&2
        exit 1
    fi
    for sample in "${samples[@]}"; do
        resolve_job "$sample" "$angles"
        observations="$(validate_inputs)"
        if [[ ! "$frame_idx" =~ ^[0-9]+$ ]] || (( angles > observations || frame_idx >= observations / angles )); then
            echo "Invalid selection: ${sample}, angles=${angles}, frame_idx=${frame_idx}, observations=${observations}" >&2
            exit 1
        fi
        if [[ "$angle_group_mode" == sparse_view ]] && (( observations != 240 )); then
            echo "sparse_view requires 240 observations: ${sample} has ${observations}" >&2
            exit 1
        fi
        job_samples+=("$sample")
        job_angles+=("$angles")
    done
done

run_id="$(date +%Y%m%d_%H%M%S)_$$"
log_dir="${LOG_DIR:-log/six_cells_${run_id}}"
if ! is_true "$dry_run"; then
    mkdir -p "$log_dir"
fi
printf 'Validated %s jobs. GPUs: %s. DRY_RUN=%s PRE_RECON_CHECK=%s START_TIME=%s\n' \
    "${#job_samples[@]}" "${gpus[*]}" "$dry_run" "$pre_recon_check" "$start_time"

declare -a gpu_pids=() gpu_jobs=()
failures=0
wait_for_gpu() {
    local slot=$1
    if [[ -n "${gpu_pids[$slot]:-}" ]]; then
        if wait "${gpu_pids[$slot]}"; then
            echo "Completed ${gpu_jobs[$slot]} on GPU ${gpus[$slot]}"
        else
            echo "Failed ${gpu_jobs[$slot]} on GPU ${gpus[$slot]}; see ${log_dir}" >&2
            failures=$((failures + 1))
        fi
        gpu_pids[$slot]=''
    fi
}

for index in "${!job_samples[@]}"; do
    sample="${job_samples[$index]}"
    angles="${job_angles[$index]}"
    slot=$((index % ${#gpus[@]}))
    gpu="${gpus[$slot]}"
    wait_for_gpu "$slot"
    resolve_job "$sample" "$angles"
    cuda_devices="$gpu"
    extra_args=(--require_cuda)
    if is_true "$pre_recon_check"; then
        cuda_devices=''
        extra_args=(--stop_before_reconstruction)
    fi
    cmd=(
        "$python_bin" "$solver"
        --data_config "$data_config"
        --physics_config "$physics_config"
        --model_config "$model_config"
        --reconstruction_config "$reconstruction_config"
        --lambda_scale "$lambda_scale" --step_size "$step_size"
        --resample_times 3 --resample_start "$resample_start" --start_time "$start_time"
        --angle_group "$angles" --angle_group_mode "$angle_group_mode" --frame_idx "$frame_idx"
        --prior_batch_size "$prior_batch_size" --bpm_batch_size "$bpm_batch_size"
        "${extra_args[@]}"
    )
    printf 'Job %s/%s: %s angles=%s mode=%s frame_idx=%s GPU=%s lambda=%s step=%s resample_start=%s prior_batch=%s\n' \
        "$((index + 1))" "${#job_samples[@]}" "$sample" "$angles" "$angle_group_mode" "$frame_idx" "$gpu" "$lambda_scale" "$step_size" "$resample_start" "$prior_batch_size"
    if is_true "$dry_run"; then
        printf '[DRY_RUN] CUDA_VISIBLE_DEVICES=%q' "$cuda_devices"
        printf ' %q' "${cmd[@]}"
        printf '\n'
    else
        (
            export CUDA_VISIBLE_DEVICES="$cuda_devices"
            printf 'Command:'
            printf ' %q' "${cmd[@]}"
            printf '\n'
            exec "${cmd[@]}"
        ) > "${log_dir}/${sample}_${angles}.log" 2>&1 &
        gpu_pids[$slot]=$!
        gpu_jobs[$slot]="${sample}_${angles}"
    fi
done
for slot in "${!gpus[@]}"; do
    wait_for_gpu "$slot"
done
if (( failures > 0 )); then
    echo "${failures} reconstruction job(s) failed. Logs: ${log_dir}" >&2
    exit 1
fi
if is_true "$dry_run"; then
    echo "Dry run complete: ${#job_samples[@]} jobs."
else
    echo "All ${#job_samples[@]} jobs completed. Logs: ${log_dir}"
fi
