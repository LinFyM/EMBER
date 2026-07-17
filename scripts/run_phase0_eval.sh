#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONTRACT="$ROOT/configs/phase0.toml"
PYTHON="$ROOT/.venv/bin/python"
LEROBOT_EVAL="$ROOT/.venv/bin/lerobot-eval"

gpu=""
task_suite="libero_spatial"
task_ids="[0]"
episodes=1
batch_size=1
async_envs=false
seed=1000
output_dir=""
dry_run=false

usage() {
  cat <<'EOF'
Usage: scripts/run_phase0_eval.sh OPTIONS

Required:
  --gpu=INDEX              One physical GPU index for this process (0-7).
  --output-dir=ABS_PATH    Fresh or incomplete external run directory.

Optional:
  --task-suite=NAME        LIBERO suite (default: libero_spatial).
  --task-ids=JSON_LIST     Task IDs within the suite (default: [0]).
  --episodes=N             Episodes per task (default: 1).
  --batch-size=N           Vector environments per task (default: 1).
  --async-envs=BOOL        Run vector environments in subprocesses (default: false).
  --seed=N                 First rollout seed (default: 1000).
  --dry-run                Print the offline command without touching assets/GPU.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

while (($#)); do
  case "$1" in
    --gpu=*) gpu=${1#*=} ;;
    --task-suite=*) task_suite=${1#*=} ;;
    --task-ids=*) task_ids=${1#*=} ;;
    --episodes=*) episodes=${1#*=} ;;
    --batch-size=*) batch_size=${1#*=} ;;
    --async-envs=*) async_envs=${1#*=} ;;
    --seed=*) seed=${1#*=} ;;
    --output-dir=*) output_dir=${1#*=} ;;
    --dry-run) dry_run=true ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

[[ "$gpu" =~ ^[0-7]$ ]] || die "--gpu must be one physical GPU index from 0 through 7"
[[ "$task_suite" =~ ^[a-z0-9_]+$ ]] || die "invalid --task-suite"
[[ "$task_ids" =~ ^\[[0-9]+(,[0-9]+)*\]$ ]] || die "--task-ids must be a compact JSON integer list"
is_positive_integer "$episodes" || die "--episodes must be a positive integer"
is_positive_integer "$batch_size" || die "--batch-size must be a positive integer"
[[ "$async_envs" = true || "$async_envs" = false ]] || die "--async-envs must be true or false"
[[ "$seed" =~ ^[0-9]+$ ]] || die "--seed must be a non-negative integer"
((episodes % batch_size == 0)) || die "--episodes must be divisible by --batch-size; LeRobot discards excess rollouts"
[[ "$output_dir" = /* ]] || die "--output-dir must be an absolute path"

if [[ -f "$ROOT/.env.local" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env.local"
  set +a
fi

: "${EMBER_ASSET_ROOT:?set EMBER_ASSET_ROOT or provide it in .env.local}"
: "${EMBER_DATA_ROOT:?set EMBER_DATA_ROOT or provide it in .env.local}"
: "${HF_HOME:?set HF_HOME or provide it in .env.local}"
: "${LIBERO_CONFIG_PATH:?set LIBERO_CONFIG_PATH or provide it in .env.local}"
[[ -x "$PYTHON" ]] || die "locked environment is missing; run scripts/bootstrap_env.sh"

smoke_revision=$(
  "$PYTHON" -c \
    'import sys,tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["models"]["smolvla_libero_smoke"]["revision"])' \
    "$CONTRACT"
)
policy_path="$EMBER_ASSET_ROOT/runtime/smolvla_libero/$smoke_revision"

command=(
  "$LEROBOT_EVAL"
  "--policy.path=$policy_path"
  "--policy.device=cuda"
  "--policy.empty_cameras=1"
  "--env.type=libero"
  "--env.task=$task_suite"
  "--env.task_ids=$task_ids"
  "--env.max_parallel_tasks=1"
  '--env.camera_name_mapping={"agentview_image":"camera1","robot0_eye_in_hand_image":"camera2"}'
  "--eval.batch_size=$batch_size"
  "--eval.n_episodes=$episodes"
  "--eval.use_async_envs=$async_envs"
  "--eval.recording=false"
  "--seed=$seed"
  "--job_name=phase0_smolvla_$task_suite"
  "--output_dir=$output_dir"
)

if $dry_run; then
  printf 'CUDA_VISIBLE_DEVICES=%q MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ' "$gpu"
  printf 'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 '
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

[[ ! -e "$output_dir/eval_info.json" ]] || die "refusing to overwrite a completed evaluation: $output_dir"
mkdir -p "$output_dir"

export CUDA_VISIBLE_DEVICES="$gpu"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m ember.phase0_runtime \
  --contract "$CONTRACT" \
  --asset-root "$EMBER_ASSET_ROOT" \
  --data-root "$EMBER_DATA_ROOT" \
  --libero-config-root "$LIBERO_CONFIG_PATH"

/usr/bin/time -v "${command[@]}"

PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m ember.eval_artifacts \
  --run-dir "$output_dir" \
  --latest-link "$(dirname "$output_dir")/latest"
