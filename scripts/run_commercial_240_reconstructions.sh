#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_root"

export SAMPLES='293t hela cos7 u2os'
export ANGLE_GROUPS=240
export LOG_DIR="${LOG_DIR:-log/commercial_240_$(date +%Y%m%d_%H%M%S)_$$}"

exec bash scripts/solve_adam_reddiff_tri_resample_angle_group_grid_search.sh
