#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG="$ROOT/configs/gate_zero_task_local_rl_critic_warmup.toml"
GATE_ZERO="$ROOT/configs/gate_zero_oracle_pilot.toml"
PHASE0="$ROOT/configs/phase0.toml"
FIT="$ROOT/configs/gate_zero_mature_lora_lr_recovery.toml"
HEADROOM="$ROOT/configs/gate_zero_mature_lora_headroom_screen.toml"
DIAGNOSTIC="$ROOT/configs/gate_zero_mature_lora_candidate_step_diagnostic.toml"
PYTHON=${EMBER_PYTHON:-"$ROOT/.venv/bin/python"}

gpus=""
output_dir=""
latest_link=""
stop_after_episodes=""
training_seed=""
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
    --config=*) CONFIG=${1#*=} ;;
    --gpus=*) gpus=${1#*=} ;;
    --output-dir=*) output_dir=${1#*=} ;;
    --latest-link=*) latest_link=${1#*=} ;;
    --stop-after-episodes=*) stop_after_episodes=${1#*=} ;;
    --training-seed=*) training_seed=${1#*=} ;;
    --resume) resume=true ;;
    --dry-run) dry_run=true ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

[[ "$CONFIG" = /* ]] || die "--config must be absolute"
[[ -f "$CONFIG" ]] || die "--config does not exist"
[[ "$gpus" =~ ^[0-7],[0-7],[0-7],[0-7]$ ]] || die "--gpus must contain exactly four indices"
IFS=',' read -r -a gpu_indices <<< "$gpus"
[[ "$(printf '%s\n' "${gpu_indices[@]}" | sort -u | wc -l)" -eq 4 ]] ||
  die "--gpus contains a duplicate"
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

spec = tomllib.load(open(sys.argv[1], "rb"))
gate_zero = tomllib.load(open(sys.argv[2], "rb"))
output_root = pathlib.Path(sys.argv[3])
data_root = pathlib.Path(sys.argv[4])
runtime = spec.get("runtime")
base = spec
if runtime is not None:
    root = pathlib.Path(sys.argv[1]).parents[1]
    base = tomllib.load(open(root / runtime["base_contract_relative_path"], "rb"))
authority = base["authority"]
source_base = output_root / authority["source_base_output_relative_path"]
print(output_root / gate_zero["authority"]["canonical_manifest_relative_path"])
print(data_root / gate_zero["authority"]["dataset_relative_path"])
print(source_base / "checkpoints" / f'{authority["source_base_checkpoint_step"]:06d}')
for key in ("fit_root", "headroom_result", "candidate_diagnostic_result", "previous_awr_result", "previous_signed_result", "previous_temporal_result"):
    print(output_root / authority[f"{key}_relative_path"])
print(output_root / authority["previous_critic_result_relative_path"] if "previous_critic_result_relative_path" in authority else "-")
print(output_root / authority["support_replay_result_relative_path"] if "support_replay_result_relative_path" in authority else "-")
print(output_root / authority["horizon_credit_result_relative_path"] if "horizon_credit_result_relative_path" in authority else "-")
print(authority.get("horizon_credit_result_sha256", "-"))
print(output_root / authority["horizon_support_replay_result_relative_path"] if "horizon_support_replay_result_relative_path" in authority else "-")
print(authority.get("horizon_support_replay_result_sha256", "-"))
print(base["resources"]["minimum_free_memory_mib"])
nodes = runtime["interaction_episode_nodes"] if runtime is not None else base["training_interaction"]["interaction_episode_nodes"]
print(",".join(str(value) for value in nodes))
print(runtime["active_training_seed"] if runtime is not None else "-")
PY
)
manifest=${paths[0]}
dataset_root=${paths[1]}
source_checkpoint=${paths[2]}
fit_root=${paths[3]}
headroom_result=${paths[4]}
diagnostic_result=${paths[5]}
previous_awr_result=${paths[6]}
previous_signed_result=${paths[7]}
previous_temporal_result=${paths[8]}
previous_critic_result=${paths[9]}
support_replay_result=${paths[10]}
horizon_credit_result=${paths[11]}
horizon_credit_result_sha256=${paths[12]}
horizon_support_replay_result=${paths[13]}
horizon_support_replay_result_sha256=${paths[14]}
minimum_free_memory_mib=${paths[15]}
interaction_episode_nodes=${paths[16]}
default_training_seed=${paths[17]}
[[ -n "$training_seed" ]] || training_seed=$default_training_seed
first_interaction_node=${interaction_episode_nodes%%,*}
[[ ",$interaction_episode_nodes," == *",$stop_after_episodes,"* ]] ||
  die "--stop-after-episodes is outside the selected contract: $interaction_episode_nodes"
if [[ "$stop_after_episodes" == "$first_interaction_node" ]]; then
  $resume && die "the first stage must start fresh"
else
  $resume || die "later stages must exact-resume the prior output"
fi

command=(
  "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=4
  -m ember.gate_zero_task_local_rl
  --config "$CONFIG"
  --gate-zero-contract "$GATE_ZERO"
  --phase0-contract "$PHASE0"
  --fit-contract "$FIT"
  --headroom-contract "$HEADROOM"
  --diagnostic-contract "$DIAGNOSTIC"
  --manifest "$manifest"
  --dataset-root "$dataset_root"
  --source-base-checkpoint "$source_checkpoint"
  --fit-root "$fit_root"
  --headroom-result "$headroom_result"
  --diagnostic-result "$diagnostic_result"
  --previous-awr-result "$previous_awr_result"
  --previous-signed-result "$previous_signed_result"
  --previous-temporal-result "$previous_temporal_result"
  --output-dir "$output_dir"
  --latest-link "$latest_link"
  --physical-gpus "$gpus"
  --stop-after-episodes "$stop_after_episodes"
)
[[ "$training_seed" == "-" ]] || command+=(--training-seed "$training_seed")
if [[ "$previous_critic_result" != "-" ]]; then
  command+=(--previous-critic-result "$previous_critic_result")
fi
if [[ "$support_replay_result" != "-" ]]; then
  command+=(--support-replay-result "$support_replay_result")
fi
$resume && command+=(--resume)

if $dry_run; then
  printf 'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=%q MUJOCO_GL=egl ' "$gpus"
  printf 'TRACKIO_DIR=%q HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=%q ' \
    "$EMBER_OUTPUT_ROOT/trackio" "$ROOT/src"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

for path in "$manifest" "$dataset_root" "$source_checkpoint" "$fit_root" \
  "$headroom_result" "$diagnostic_result" "$previous_awr_result" \
  "$previous_signed_result" "$previous_temporal_result"; do
  [[ -e "$path" ]] || die "required authority is missing: $path"
done
for path in "$previous_critic_result" "$support_replay_result"; do
  [[ "$path" == "-" || -e "$path" ]] || die "required recovery authority is missing: $path"
done
for path in "$horizon_credit_result" "$horizon_support_replay_result"; do
  [[ "$path" == "-" || -e "$path" ]] || die "required coverage authority is missing: $path"
done
if [[ "$horizon_credit_result" != "-" ]]; then
  [[ "$(sha256sum "$horizon_credit_result" | cut -d' ' -f1)" == "$horizon_credit_result_sha256" ]] ||
    die "horizon-credit terminal result hash changed"
  [[ "$(sha256sum "$horizon_support_replay_result" | cut -d' ' -f1)" == "$horizon_support_replay_result_sha256" ]] ||
    die "horizon support-replay result hash changed"
fi
for gpu in "${gpu_indices[@]}"; do
  active_compute=$(
    nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null |
      sed '/^[[:space:]]*$/d'
  )
  [[ -z "$active_compute" ]] || die "GPU $gpu has active compute PID(s): ${active_compute//$'\n'/,}"
  memory_used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  memory_total=$(nvidia-smi -i "$gpu" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "$memory_used" =~ ^[0-9]+$ && "$memory_total" =~ ^[0-9]+$ ]] ||
    die "cannot parse GPU $gpu memory"
  ((memory_used < 1000)) || die "GPU $gpu already uses ${memory_used} MiB"
  ((memory_total - memory_used >= minimum_free_memory_mib)) || die "GPU $gpu lacks OOM headroom"
done
if $resume; then
  [[ -d "$output_dir" ]] || die "resume output directory does not exist"
  [[ ! -e "$output_dir/task_local_rl_recovery_result.json" ]] ||
    die "refusing to resume a terminal RL recovery"
else
  [[ ! -e "$output_dir" ]] || die "refusing to reuse RL recovery output directory"
  mkdir -p "$output_dir"
fi

run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
telemetry="$output_dir/gpu_telemetry_ep${stop_after_episodes}_${run_stamp}.csv"
nvidia-smi -i "$gpus" \
  --query-gpu=timestamp,index,uuid,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw \
  --format=csv -lms 500 > "$telemetry" &
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
(
  cd "$output_dir"
  sha256sum "$(basename "$telemetry")" > "telemetry_ep${stop_after_episodes}.sha256"
  if ((main_rc == 0)) && [[ -f checksums.sha256 ]]; then
    cat "telemetry_ep${stop_after_episodes}.sha256" >> checksums.sha256
    sha256sum -c checksums.sha256
  fi
)
exit "$main_rc"
