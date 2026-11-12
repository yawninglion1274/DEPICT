#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_root"

python_bin="${PYTHON:-/data3/rsc/miniconda3/envs/diffusion/bin/python}"
gpu="${GPU:-5}"
log_file="${LOG_FILE:-log/tcell_240_prior20_$(date +%Y%m%d_%H%M%S)_$$.log}"
if [[ "${1:-}" == --worker ]]; then
    log_file="$2"
fi
mkdir -p "$(dirname -- "$log_file")"

if [[ "${1:-}" != --worker ]]; then
    nohup setsid bash "${BASH_SOURCE[0]}" --worker "$log_file" \
        > "$log_file" 2>&1 < /dev/null &
    task_pid=$!
    printf '%s\n' "$task_pid" > "${log_file}.pid"
    printf '后台任务已提交，PID: %s\n' "$task_pid"
    printf '日志: %s\n' "$log_file"
    printf '查看进度: tail -f %q\n' "$log_file"
    exit 0
fi

cmd=(
    "$python_bin" adam_REDDiff_resample_BPM_resample_triplane_angle_group.py
    --data_config configs/isorecon_data_config/20260611_T_0.5_11.yaml
    --physics_config configs/physics_config/lzr.yaml
    --model_config configs/model_config/3_cells_230000ema.yaml
    --reconstruction_config configs/reconstruction_config/DiffPIR_FISTA/adam_RED.yaml
    --lambda_scale 0.015
    --step_size 0.0025
    --resample_times 3
    --resample_start 100
    --start_time 999
    --angle_group 240
    --angle_group_mode fast_acq
    --frame_idx 0
    --prior_batch_size 20
    --bpm_batch_size 6
    --require_cuda
)

{
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "$gpu"
    printf 'Log: %s\n' "$log_file"
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
}

set +e
CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "${cmd[@]}"
status=$?
set -e

if [[ "$status" -eq 0 ]]; then
    printf 'tcell_240 completed successfully.\n'
else
    printf 'tcell_240 failed with exit code %s. See %s\n' "$status" "$log_file" >&2
fi
exit "$status"
