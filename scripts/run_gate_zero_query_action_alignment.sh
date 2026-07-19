#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG="$ROOT/configs/gate_zero_query_action_alignment_audit.toml"
GATE_ZERO="$ROOT/configs/gate_zero_oracle_pilot.toml"
PHASE0="$ROOT/configs/phase0.toml"
COMPETENCE="$ROOT/configs/gate_zero_source_competence.toml"
LORA_FIT="$ROOT/configs/gate_zero_mature_lora_lr_recovery.toml"
ACTION_FIT="$ROOT/configs/gate_zero_mature_action_expert_lr_recovery.toml"
CAPACITY="$ROOT/configs/gate_zero_action_expert_capacity_closed_loop.toml"
PYTHON=${EMBER_PYTHON:-"$ROOT/.venv/bin/python"}
gpus=""; output_dir=""; latest_link=""; dry_run=false; sampler=""

die() { printf 'error: %s\n' "$*" >&2; exit 2; }
stop_sampler() {
  if [[ -n "$sampler" ]]; then
    kill "$sampler" 2>/dev/null || true
    wait "$sampler" 2>/dev/null || true
    sampler=""
  fi
}
trap 'stop_sampler; exit 130' INT
trap 'stop_sampler; exit 143' TERM

while (($#)); do
  case "$1" in
    --config=*) CONFIG=${1#*=} ;;
    --gpus=*) gpus=${1#*=} ;;
    --output-dir=*) output_dir=${1#*=} ;;
    --latest-link=*) latest_link=${1#*=} ;;
    --dry-run) dry_run=true ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

[[ "$CONFIG" = /* && -f "$CONFIG" ]] || die "--config must be an existing absolute path"
[[ "$gpus" =~ ^[0-7],[0-7]$ ]] || die "--gpus must contain exactly two indices"
IFS=',' read -r -a gpu_indices <<< "$gpus"
[[ "${gpu_indices[0]}" != "${gpu_indices[1]}" ]] || die "--gpus contains a duplicate"
[[ "$output_dir" = /* ]] || die "--output-dir must be absolute"
[[ -n "$latest_link" ]] || latest_link="$(dirname "$output_dir")/latest"
[[ "$latest_link" = /* ]] || die "--latest-link must be absolute"

if [[ -f "$ROOT/.env.local" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env.local"
  set +a
fi
: "${EMBER_DATA_ROOT:?set EMBER_DATA_ROOT}"
: "${EMBER_OUTPUT_ROOT:?set EMBER_OUTPUT_ROOT}"
: "${HF_HOME:?set HF_HOME}"
: "${LIBERO_CONFIG_PATH:?set LIBERO_CONFIG_PATH}"
[[ -x "$PYTHON" ]] || die "locked Python is missing"

mapfile -t paths < <(
  "$PYTHON" - "$CONFIG" "$GATE_ZERO" "$EMBER_OUTPUT_ROOT" "$EMBER_DATA_ROOT" <<'PY'
import pathlib, sys, tomllib
spec = tomllib.load(open(sys.argv[1], "rb")); gate = tomllib.load(open(sys.argv[2], "rb"))
out = pathlib.Path(sys.argv[3]); data = pathlib.Path(sys.argv[4])
base = out / spec["authority"]["source_base_output_relative_path"]
print(out / gate["authority"]["canonical_manifest_relative_path"])
print(data / gate["authority"]["dataset_relative_path"])
print(base / "checkpoints" / f'{spec["authority"]["source_base_checkpoint_step"]:06d}')
print(out / spec["authority"]["lora_fit_root_relative_path"])
print(out / spec["authority"]["action_expert_fit_root_relative_path"])
print(out / spec["authority"]["capacity_result_relative_path"])
print(spec["resources"]["minimum_free_memory_mib"])
PY
)
manifest=${paths[0]}; dataset_root=${paths[1]}; source_checkpoint=${paths[2]}
lora_fit_root=${paths[3]}; action_fit_root=${paths[4]}; capacity_result=${paths[5]}
minimum_free_memory_mib=${paths[6]}
command=(
  "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=2
  -m ember.gate_zero_query_action_alignment
  --config "$CONFIG" --gate-zero-contract "$GATE_ZERO" --phase0-contract "$PHASE0"
  --source-competence-contract "$COMPETENCE" --lora-fit-contract "$LORA_FIT"
  --action-expert-fit-contract "$ACTION_FIT" --capacity-contract "$CAPACITY"
  --capacity-result "$capacity_result" --manifest "$manifest" --dataset-root "$dataset_root"
  --source-base-checkpoint "$source_checkpoint" --lora-fit-root "$lora_fit_root"
  --action-expert-fit-root "$action_fit_root" --output-dir "$output_dir"
  --latest-link "$latest_link" --physical-gpus "$gpus"
)

if $dry_run; then
  printf 'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=%q MUJOCO_GL=egl ' "$gpus"
  printf 'TRACKIO_DIR=%q HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=%q ' \
    "$EMBER_OUTPUT_ROOT/trackio" "$ROOT/src"
  printf '%q ' "${command[@]}"; printf '\n'; exit 0
fi
for path in "$manifest" "$dataset_root" "$source_checkpoint" "$lora_fit_root" \
  "$action_fit_root" "$capacity_result"; do
  [[ -e "$path" ]] || die "required authority is missing: $path"
done
for gpu in "${gpu_indices[@]}"; do
  active=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
  [[ -z "$active" ]] || die "GPU $gpu has active compute PID(s): ${active//$'\n'/,}"
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  total=$(nvidia-smi -i "$gpu" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "$used" =~ ^[0-9]+$ && "$total" =~ ^[0-9]+$ ]] || die "cannot parse GPU $gpu memory"
  ((used < 1000 && total - used >= minimum_free_memory_mib)) || die "GPU $gpu is unavailable"
done
[[ ! -e "$output_dir" ]] || die "refusing to reuse audit output"
mkdir -p "$output_dir"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
telemetry="$output_dir/gpu_telemetry_${stamp}.csv"
nvidia-smi -i "$gpus" --query-gpu=timestamp,index,uuid,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw --format=csv -lms 500 > "$telemetry" &
sampler=$!

export CUDA_VISIBLE_DEVICES="$gpus" CUDA_DEVICE_ORDER=PCI_BUS_ID MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=20260719 OMP_NUM_THREADS=2
export TRACKIO_DIR="$EMBER_OUTPUT_ROOT/trackio" TRACKIO_STORAGE_MODE=sqlite
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
set +e; /usr/bin/time -v "${command[@]}"; main_rc=$?; set -e
stop_sampler; trap - INT TERM
if ((main_rc == 0)); then
  (cd "$output_dir"; sha256sum "$(basename "$telemetry")" >> checksums.sha256; sha256sum -c checksums.sha256)
fi
exit "$main_rc"
