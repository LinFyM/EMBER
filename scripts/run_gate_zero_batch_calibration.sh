#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG="$ROOT/configs/gate_zero_oracle_pilot.toml"
PHASE0="$ROOT/configs/phase0.toml"
PYTHON="$ROOT/.venv/bin/python"

gpu=""
output_dir=""
latest_link=""
dry_run=false

die() { printf 'error: %s\n' "$*" >&2; exit 2; }

while (($#)); do
  case "$1" in
    --gpu=*) gpu=${1#*=} ;;
    --output-dir=*) output_dir=${1#*=} ;;
    --latest-link=*) latest_link=${1#*=} ;;
    --dry-run) dry_run=true ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

[[ "$gpu" =~ ^[0-7]$ ]] || die "--gpu must be one physical GPU index"
[[ "$output_dir" = /* ]] || die "--output-dir must be absolute"
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

readarray -t paths < <("$PYTHON" - "$CONFIG" <<'PY'
import sys, tomllib
spec=tomllib.load(open(sys.argv[1], "rb"))
print(spec["authority"]["model_revision"])
print(spec["authority"]["canonical_manifest_relative_path"])
print(spec["authority"]["source_normalization_relative_path"])
print(spec["authority"]["dataset_relative_path"])
PY
)
base_path="$EMBER_ASSET_ROOT/models/smolvla_base/${paths[0]}"
vlm_revision=$("$PYTHON" - "$PHASE0" <<'PY'
import sys, tomllib
print(tomllib.load(open(sys.argv[1], "rb"))["models"]["smolvlm_constructor_dependency"]["revision"])
PY
)
vlm_path="$EMBER_ASSET_ROOT/models/SmolVLM2-500M-Video-Instruct/$vlm_revision"
manifest="$EMBER_OUTPUT_ROOT/${paths[1]}"
normalization="$EMBER_OUTPUT_ROOT/${paths[2]}"
dataset_root="$EMBER_DATA_ROOT/${paths[3]}"

command=(
  "$PYTHON" -m ember.gate_zero_batch_calibration
  --config "$CONFIG"
  --phase0-contract "$PHASE0"
  --manifest "$manifest"
  --normalization "$normalization"
  --dataset-root "$dataset_root"
  --base-path "$base_path"
  --vlm-path "$vlm_path"
  --output-dir "$output_dir"
  --latest-link "$latest_link"
)

if $dry_run; then
  printf 'CUDA_VISIBLE_DEVICES=%q TRACKIO_DIR=%q ' "$gpu" "$EMBER_OUTPUT_ROOT/trackio"
  printf 'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=%q ' "$ROOT/src"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

[[ ! -e "$output_dir/calibration_result.json" ]] || die "completed output already exists"
mkdir -p "$output_dir"
telemetry="$output_dir/gpu_telemetry.csv"
nvidia-smi -i "$gpu" --query-gpu=timestamp,index,uuid,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw --format=csv -lms 500 > "$telemetry" &
sampler=$!

export CUDA_VISIBLE_DEVICES="$gpu"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=20260718
export OMP_NUM_THREADS=4
export TRACKIO_DIR="$EMBER_OUTPUT_ROOT/trackio"
export TRACKIO_STORAGE_MODE=sqlite
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

set +e
/usr/bin/time -v "${command[@]}"
main_rc=$?
set -e
kill "$sampler" 2>/dev/null || true
wait "$sampler" 2>/dev/null || true
exit "$main_rc"
