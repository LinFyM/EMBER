#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG="$ROOT/configs/gate_zero_target_support_audit.toml"
GATE_ZERO="$ROOT/configs/gate_zero_oracle_pilot.toml"
PHASE0="$ROOT/configs/phase0.toml"
COMPETENCE="$ROOT/configs/gate_zero_source_competence.toml"
PYTHON=${EMBER_PYTHON:-"$ROOT/.venv/bin/python"}

gpus=""
fit_root=""
freeze_dir=""
output_dir=""
latest_link=""
reuse_freeze=false
dry_run=false
sampler=""

die() { printf 'error: %s\n' "$*" >&2; exit 2; }
stop_sampler() {
  if [[ -n "$sampler" ]]; then
    kill "$sampler" 2>/dev/null || true
    wait "$sampler" 2>/dev/null || true
    sampler=""
  fi
}
handle_signal() {
  local rc=$1
  stop_sampler
  exit "$rc"
}
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

while (($#)); do
  case "$1" in
    --gpus=*) gpus=${1#*=} ;;
    --fit-root=*) fit_root=${1#*=} ;;
    --screening-freeze-dir=*) freeze_dir=${1#*=} ;;
    --output-dir=*) output_dir=${1#*=} ;;
    --latest-link=*) latest_link=${1#*=} ;;
    --reuse-screening-freeze) reuse_freeze=true ;;
    --dry-run) dry_run=true ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

[[ "$gpus" =~ ^[0-7](,[0-7]){0,3}$ ]] || \
  die "--gpus must contain one, two, or four indices"
IFS=',' read -r -a gpu_indices <<< "$gpus"
gpu_count=${#gpu_indices[@]}
[[ "$gpu_count" = 1 || "$gpu_count" = 2 || "$gpu_count" = 4 ]] || \
  die "--gpus count must be one, two, or four"
declare -A seen_gpus=()
for gpu in "${gpu_indices[@]}"; do
  [[ -z "${seen_gpus[$gpu]:-}" ]] || die "--gpus contains duplicate index $gpu"
  seen_gpus[$gpu]=1
done
[[ "$fit_root" = /* ]] || die "--fit-root must be absolute"
[[ "$freeze_dir" = /* ]] || die "--screening-freeze-dir must be absolute"
[[ "$output_dir" = /* ]] || die "--output-dir must be absolute"
if [[ -z "$latest_link" ]]; then
  latest_link="$(dirname "$output_dir")/latest"
fi
[[ "$latest_link" = /* ]] || die "--latest-link must be absolute"

if [[ -f "$ROOT/.env.local" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env.local"
  set +a
fi
: "${EMBER_OUTPUT_ROOT:?set EMBER_OUTPUT_ROOT}"
: "${HF_HOME:?set HF_HOME}"
: "${LIBERO_CONFIG_PATH:?set LIBERO_CONFIG_PATH}"
[[ -x "$PYTHON" ]] || die "locked Python is missing"

grant="$freeze_dir/screening_grant.json"
source_checkpoint=$(
  "$PYTHON" - "$CONFIG" "$EMBER_OUTPUT_ROOT" <<'PY'
import pathlib
import sys
import tomllib

spec = tomllib.load(open(sys.argv[1], "rb"))
root = pathlib.Path(sys.argv[2]) / spec["authority"]["source_base_output_relative_path"]
print(root / "checkpoints" / f'{spec["authority"]["source_base_checkpoint_step"]:06d}')
PY
)

freeze_command=(
  "$PYTHON" -m ember.gate_zero_support.screen
  --config "$CONFIG"
  --gate-zero-contract "$GATE_ZERO"
  --phase0-contract "$PHASE0"
  --source-competence-contract "$COMPETENCE"
  --fit-root "$fit_root"
  --grant-path "$grant"
)
screen_command=(
  "$PYTHON" -m torch.distributed.run
  --standalone
  "--nproc-per-node=$gpu_count"
  -m ember.gate_zero_support.screen_runtime
  --config "$CONFIG"
  --gate-zero-contract "$GATE_ZERO"
  --phase0-contract "$PHASE0"
  --source-competence-contract "$COMPETENCE"
  --source-base-checkpoint "$source_checkpoint"
  --screening-grant "$grant"
  --fit-root "$fit_root"
  --output-dir "$output_dir"
  --latest-link "$latest_link"
  --physical-gpus "$gpus"
)

if $dry_run; then
  printf 'PYTHONPATH=%q ' "$ROOT/src"
  printf '%q ' "${freeze_command[@]}"
  printf '\n'
  printf 'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=%q MUJOCO_GL=egl ' "$gpus"
  printf 'TRACKIO_DIR=%q HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=%q ' \
    "$EMBER_OUTPUT_ROOT/trackio" "$ROOT/src"
  printf '%q ' "${screen_command[@]}"
  printf '\n'
  exit 0
fi

mapfile -t fit_names < <(
  "$PYTHON" - "$CONFIG" <<'PY'
import sys
import tomllib

spec = tomllib.load(open(sys.argv[1], "rb"))
for variant in spec["variants"]:
    for task_id in spec["task_ids"]:
        print(f"{variant}_task{task_id}")
PY
)
for name in "${fit_names[@]}"; do
  [[ -d "$fit_root/$name/selected" ]] || die "selected fit output is missing: $name"
done
if $reuse_freeze; then
  [[ -f "$grant" ]] || die "reused screening grant is missing"
else
  [[ ! -e "$freeze_dir" ]] || die "refusing to overwrite screening-freeze directory"
fi
[[ ! -e "$output_dir" ]] || die "refusing to reuse screening output directory"

for gpu in "${gpu_indices[@]}"; do
  active_compute=$(
    nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null |
      sed '/^[[:space:]]*$/d'
  )
  [[ -z "$active_compute" ]] || \
    die "GPU $gpu has active compute PID(s): ${active_compute//$'\n'/,}"
  memory_used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "$memory_used" =~ ^[0-9]+$ ]] || die "cannot parse GPU $gpu memory usage"
  ((memory_used < 1000)) || die "GPU $gpu already uses ${memory_used} MiB"
done

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=20260718
export OMP_NUM_THREADS=2
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if ! $reuse_freeze; then
  "${freeze_command[@]}"
fi

mkdir -p "$output_dir"
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
telemetry="$output_dir/gpu_telemetry_${run_stamp}.csv"
nvidia-smi -i "$gpus" \
  --query-gpu=timestamp,index,uuid,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw \
  --format=csv -lms 500 > "$telemetry" &
sampler=$!

export CUDA_VISIBLE_DEVICES="$gpus"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TRACKIO_DIR="$EMBER_OUTPUT_ROOT/trackio"
export TRACKIO_STORAGE_MODE=sqlite

set +e
/usr/bin/time -v "${screen_command[@]}"
main_rc=$?
set -e
stop_sampler
trap - INT TERM
if ((main_rc == 0)); then
  (
    cd "$output_dir"
    sha256sum "$(basename "$telemetry")" >> checksums.sha256
  )
fi
exit "$main_rc"
