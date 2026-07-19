#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${EMBER_PYTHON:-"$ROOT/.venv/bin/python"}
CONFIG="$ROOT/configs/writer_cold_start.toml"
train_config=
mode=train
output_dir=
stop_after_step=
resume_checkpoint=
writer_checkpoint=
resume=false
dry_run=false
sampler=

die() { printf 'error: %s\n' "$*" >&2; exit 2; }
cleanup() {
  if [[ -n "$sampler" ]]; then
    kill "$sampler" 2>/dev/null || true
    wait "$sampler" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

while (($#)); do
  case "$1" in
    --mode=*) mode=${1#*=} ;;
    --config=*) train_config=${1#*=} ;;
    --output-dir=*) output_dir=${1#*=} ;;
    --stop-after-step=*) stop_after_step=${1#*=} ;;
    --resume-checkpoint=*) resume_checkpoint=${1#*=} ;;
    --writer-checkpoint=*) writer_checkpoint=${1#*=} ;;
    --resume) resume=true ;;
    --dry-run) dry_run=true ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

[[ "$mode" = train || "$mode" = smoke || "$mode" = validate ]] ||
  die "--mode must be train, smoke, or validate"
[[ "$output_dir" = /* ]] || die "--output-dir must be absolute"
if [[ -z "$train_config" ]]; then
  if [[ "$mode" = validate ]]; then
    train_config="$ROOT/configs/writer_cold_start_validation.toml"
  else
    train_config="$CONFIG"
  fi
fi
[[ "$train_config" = /* && -f "$train_config" ]] ||
  die "--config must name an existing absolute file"
if [[ -f "$ROOT/.env.local" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env.local"
  set +a
fi
: "${EMBER_OUTPUT_ROOT:?set EMBER_OUTPUT_ROOT}"
: "${EMBER_DATA_ROOT:?set EMBER_DATA_ROOT}"
: "${HF_HOME:?set HF_HOME}"
: "${LIBERO_CONFIG_PATH:?set LIBERO_CONFIG_PATH}"

if [[ "$mode" = validate ]]; then
  [[ "$writer_checkpoint" = /* ]] || die "--writer-checkpoint must be absolute for validation"
  command=(
    "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8
    -m ember.writer.validation
    --config "$train_config"
    --output-root "$EMBER_OUTPUT_ROOT"
    --data-root "$EMBER_DATA_ROOT"
    --output-dir "$output_dir"
    --writer-checkpoint "$writer_checkpoint"
  )
  $resume && command+=(--resume)
else
  command=(
    "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8
    -m ember.writer.train
    --config "$train_config"
    --output-root "$EMBER_OUTPUT_ROOT"
    --data-root "$EMBER_DATA_ROOT"
    --output-dir "$output_dir"
    --mode "$mode"
  )
  [[ -z "$stop_after_step" ]] || command+=(--stop-after-step "$stop_after_step")
  [[ -z "$resume_checkpoint" ]] || command+=(--resume-checkpoint "$resume_checkpoint")
fi

if $dry_run; then
  printf 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=%q ' "$ROOT/src"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

for gpu in {0..7}; do
  active=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
  [[ -z "$active" ]] || die "GPU $gpu has active compute PID(s): ${active//$'\n'/,}"
done
[[ ! -e "$output_dir" || -n "$resume_checkpoint" || "$resume" = true ]] ||
  die "refusing to reuse output directory"
mkdir -p "$output_dir"

telemetry="$output_dir/gpu_telemetry_$(date -u +%Y%m%dT%H%M%SZ).csv"
nvidia-smi --query-gpu=timestamp,index,uuid,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw \
  --format=csv -lms 500 > "$telemetry" &
sampler=$!

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=20260719
export OMP_NUM_THREADS=2
export TRACKIO_DIR="$EMBER_OUTPUT_ROOT/trackio"
export TRACKIO_STORAGE_MODE=sqlite
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

/usr/bin/time -v "${command[@]}"
