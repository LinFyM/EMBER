#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG="$ROOT/configs/gate_zero_oracle_execution.toml"
GATE_ZERO="$ROOT/configs/gate_zero_oracle_pilot.toml"
PHASE0="$ROOT/configs/phase0.toml"
COMPETENCE="$ROOT/configs/gate_zero_source_competence.toml"
PYTHON=${EMBER_PYTHON:-"$ROOT/.venv/bin/python"}

variant=""
task_id=""
gpu=""
output_dir=""
latest_link=""
resume=false
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
    --variant=*) variant=${1#*=} ;;
    --task-id=*) task_id=${1#*=} ;;
    --gpu=*) gpu=${1#*=} ;;
    --output-dir=*) output_dir=${1#*=} ;;
    --latest-link=*) latest_link=${1#*=} ;;
    --resume) resume=true ;;
    --dry-run) dry_run=true ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

[[ "$variant" = "lora" || "$variant" = "partial_upper_bound" ]] || \
  die "--variant must be lora or partial_upper_bound"
[[ "$task_id" = "3" || "$task_id" = "4" ]] || die "--task-id must be 3 or 4"
[[ "$gpu" =~ ^[0-7]$ ]] || die "--gpu must be one physical GPU index"
[[ "$output_dir" = /* ]] || die "--output-dir must be absolute"
if [[ -z "$latest_link" ]]; then
  latest_link="$(dirname "$output_dir")/latest_${variant}_task${task_id}"
fi
[[ "$latest_link" = /* ]] || die "--latest-link must be absolute"

if [[ -f "$ROOT/.env.local" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env.local"
  set +a
fi
: "${EMBER_ASSET_ROOT:?set EMBER_ASSET_ROOT}"
: "${EMBER_DATA_ROOT:?set EMBER_DATA_ROOT}"
: "${EMBER_OUTPUT_ROOT:?set EMBER_OUTPUT_ROOT}"
: "${HF_HOME:?set HF_HOME}"
: "${LIBERO_CONFIG_PATH:?set LIBERO_CONFIG_PATH}"
[[ -x "$PYTHON" ]] || die "locked Python is missing"

mapfile -t paths < <(
  "$PYTHON" - "$CONFIG" "$GATE_ZERO" "$EMBER_OUTPUT_ROOT" "$EMBER_DATA_ROOT" <<'PY'
import pathlib
import sys
import tomllib

execution = tomllib.load(open(sys.argv[1], "rb"))
gate_zero = tomllib.load(open(sys.argv[2], "rb"))
output_root = pathlib.Path(sys.argv[3])
data_root = pathlib.Path(sys.argv[4])
print(output_root / gate_zero["authority"]["canonical_manifest_relative_path"])
print(data_root / gate_zero["authority"]["dataset_relative_path"])
print(output_root / execution["authority"]["source_competence_result_relative_path"])
source_base = output_root / execution["authority"]["source_base_output_relative_path"]
print(source_base / "checkpoints" / f'{execution["authority"]["source_base_checkpoint_step"]:06d}')
PY
)
manifest=${paths[0]}
dataset_root=${paths[1]}
competence_result=${paths[2]}
source_checkpoint=${paths[3]}

command=(
  "$PYTHON" -m ember.gate_zero_oracle_fit
  --config "$CONFIG"
  --gate-zero-contract "$GATE_ZERO"
  --phase0-contract "$PHASE0"
  --source-competence-contract "$COMPETENCE"
  --source-competence-result "$competence_result"
  --manifest "$manifest"
  --dataset-root "$dataset_root"
  --source-base-checkpoint "$source_checkpoint"
  --output-dir "$output_dir"
  --latest-link "$latest_link"
  --variant "$variant"
  --task-id "$task_id"
  --physical-gpu "$gpu"
)
$resume && command+=(--resume)

if $dry_run; then
  printf 'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=%q MUJOCO_GL=egl ' "$gpu"
  printf 'TRACKIO_DIR=%q HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=%q ' \
    "$EMBER_OUTPUT_ROOT/trackio" "$ROOT/src"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

active_compute=$(
  nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null |
    sed '/^[[:space:]]*$/d'
)
[[ -z "$active_compute" ]] || die "GPU $gpu has active compute PID(s): ${active_compute//$'\n'/,}"
memory_used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
[[ "$memory_used" =~ ^[0-9]+$ ]] || die "cannot parse GPU $gpu memory usage"
((memory_used < 1000)) || die "GPU $gpu already uses ${memory_used} MiB"

if $resume; then
  [[ -d "$output_dir" ]] || die "resume output directory is missing"
else
  [[ ! -e "$output_dir" ]] || die "refusing to reuse output directory"
  mkdir -p "$output_dir"
fi

run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
telemetry="$output_dir/gpu_telemetry_${run_stamp}.csv"
nvidia-smi -i "$gpu" \
  --query-gpu=timestamp,index,uuid,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw \
  --format=csv -lms 500 > "$telemetry" &
sampler=$!

export CUDA_VISIBLE_DEVICES="$gpu"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=20260718
export OMP_NUM_THREADS=2
export TRACKIO_DIR="$EMBER_OUTPUT_ROOT/trackio"
export TRACKIO_STORAGE_MODE=sqlite
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

set +e
/usr/bin/time -v "${command[@]}"
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
