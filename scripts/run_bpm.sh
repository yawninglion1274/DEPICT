#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$project_root"

python_bin="${PYTHON:-python}"
solver='experiment_solve_bpm_clean_style.py'
config='configs/reconstruction_config/beam_propagation/TV.yaml'
read -r -a samples <<< "${SAMPLES:-293t hela cos7 u2os tcell tcell_mitosis}"
read -r -a gpus <<< "${GPUS:-0}"
if (( ${#samples[@]} == 0 || ${#gpus[@]} == 0 )); then
    echo 'SAMPLES and GPUS must not be empty.' >&2
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
    [tcell_mitosis]='configs/isorecon_data_config/20260611_T_0.5_2.yaml'
)

# Validate all selected inputs before starting a reconstruction.
for sample in "${samples[@]}"; do
    if [[ -z "${data_configs[$sample]:-}" ]]; then
        echo "Unknown sample: ${sample}" >&2
        exit 1
    fi
    physics='commercial_ODT'
    if [[ "$sample" == tcell || "$sample" == tcell_mitosis ]]; then
        physics='dual_dmd'
    fi
    "$python_bin" "$solver" --data_config "${data_configs[$sample]}" \
        --physics_config "configs/physics_config/${physics}.yaml" \
        --reconstruction_config "$config" --dry-run
done
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo 'Dry run complete; no reconstructions started.'
    exit 0
fi

log_dir="${LOG_DIR:-log/bpm_$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p -- "$log_dir"
declare -a gpu_pids=() gpu_jobs=()
failures=0
wait_for_gpu() {
    local slot=$1
    if [[ -n "${gpu_pids[$slot]:-}" ]]; then
        if wait "${gpu_pids[$slot]}"; then
            echo "Completed ${gpu_jobs[$slot]} on GPU ${gpus[$slot]}"
        else
            echo "Failed ${gpu_jobs[$slot]}; see ${log_dir}" >&2
            failures=$((failures + 1))
        fi
        gpu_pids[$slot]=''
    fi
}

for index in "${!samples[@]}"; do
    sample="${samples[$index]}"
    slot=$((index % ${#gpus[@]}))
    wait_for_gpu "$slot"
    physics='commercial_ODT'
    if [[ "$sample" == tcell || "$sample" == tcell_mitosis ]]; then
        physics='dual_dmd'
    fi
    (
        export CUDA_VISIBLE_DEVICES="${gpus[$slot]}"
        "$python_bin" -c 'import torch; assert torch.cuda.is_available(), "A CUDA GPU is required"'
        exec "$python_bin" -u "$solver" --data_config "${data_configs[$sample]}" \
            --physics_config "configs/physics_config/${physics}.yaml" \
            --reconstruction_config "$config"
    ) > "$log_dir/$sample.log" 2>&1 &
    gpu_pids[$slot]=$!
    gpu_jobs[$slot]="$sample"
    echo "Started ${sample} on GPU ${gpus[$slot]}"
done
for slot in "${!gpus[@]}"; do
    wait_for_gpu "$slot"
done
if (( failures > 0 )); then
    echo "${failures} reconstruction job(s) failed. Logs: ${log_dir}" >&2
    exit 1
fi
echo "All ${#samples[@]} jobs completed. Logs: ${log_dir}"
