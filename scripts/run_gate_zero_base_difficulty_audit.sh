#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG="$ROOT/configs/gate_zero_base_difficulty_audit.toml"
EVIDENCE="$ROOT/configs/gate_zero_evidence_repair.toml"
SPLIT="$ROOT/configs/libero90_split_reseal.json"
PYTHON=${EMBER_PYTHON:-"$ROOT/.venv/bin/python"}

gpus=""
output_dir=""
latest_link=""
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
    --output-dir=*) output_dir=${1#*=} ;;
    --latest-link=*) latest_link=${1#*=} ;;
    --dry-run) dry_run=true ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

[[ "$gpus" =~ ^[0-7](,[0-7]){0,3}$ ]] || die "--gpus must contain one, two, or four GPU indices"
IFS=',' read -r -a gpu_indices <<< "$gpus"
gpu_count=${#gpu_indices[@]}
[[ "$gpu_count" = 1 || "$gpu_count" = 2 || "$gpu_count" = 4 ]] || die "GPU count must be 1, 2, or 4"
declare -A seen=()
for gpu in "${gpu_indices[@]}"; do
  [[ -z "${seen[$gpu]:-}" ]] || die "duplicate GPU $gpu"
  seen[$gpu]=1
done
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

readarray -t authority_paths < <(
  "$PYTHON" - "$CONFIG" <<'PY'
import sys, tomllib
authority = tomllib.load(open(sys.argv[1], "rb"))["authority"]
print(authority["source_base_checkpoint_relative_path"])
print(authority["source_competence_result_relative_path"])
PY
)
checkpoint="$EMBER_OUTPUT_ROOT/${authority_paths[0]}"
competence_result="$EMBER_OUTPUT_ROOT/${authority_paths[1]}"

command=(
  "$PYTHON" -m torch.distributed.run
  --standalone
  "--nproc-per-node=$gpu_count"
  -m ember.gate_zero_support.base_difficulty_audit
  --config "$CONFIG"
  --evidence-contract "$EVIDENCE"
  --split-reseal "$SPLIT"
  --repo-root "$ROOT"
  --output-root "$EMBER_OUTPUT_ROOT"
  --checkpoint "$checkpoint"
  --competence-result "$competence_result"
  --output-dir "$output_dir"
  --latest-link "$latest_link"
  --physical-gpus "$gpus"
)

if $dry_run; then
  printf 'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=%q MUJOCO_GL=egl ' "$gpus"
  printf 'TRACKIO_DIR=%q HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=%q ' "$EMBER_OUTPUT_ROOT/trackio" "$ROOT/src"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

for gpu in "${gpu_indices[@]}"; do
  active_compute=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')
  [[ -z "$active_compute" ]] || die "GPU $gpu has active compute PID(s): ${active_compute//$'\n'/,}"
  memory_used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "$memory_used" =~ ^[0-9]+$ ]] || die "cannot parse GPU $gpu memory"
  ((memory_used < 1000)) || die "GPU $gpu already uses ${memory_used} MiB"
done

[[ ! -e "$output_dir" ]] || die "refusing to reuse output directory"
mkdir -p "$output_dir"
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
telemetry="$output_dir/gpu_telemetry_${run_stamp}.csv"
nvidia-smi -i "$gpus" --query-gpu=timestamp,index,uuid,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw --format=csv -lms 500 > "$telemetry" &
sampler=$!

export CUDA_VISIBLE_DEVICES="$gpus"
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
