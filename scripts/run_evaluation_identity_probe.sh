#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONTRACT="$ROOT/configs/phase0.toml"
CONFIG="$ROOT/configs/gate_minus1_identity.toml"
PYTHON="$ROOT/.venv/bin/python"

gpu=""
output_dir=""
dry_run=false
mechanics_only=false

usage() {
  cat <<'EOF'
Usage: scripts/run_evaluation_identity_probe.sh OPTIONS

Required:
  --gpu=INDEX              One physical GPU index for the policy layer (0-7).
  --output-dir=ABS_PATH    Fresh external probe directory.

Optional:
  --config=ABS_PATH        Predeclared probe TOML.
  --mechanics-only         Stop after reset/fixed-action identity checks.
  --dry-run                Print the offline command without touching assets/GPU.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --gpu=*) gpu=${1#*=} ;;
    --output-dir=*) output_dir=${1#*=} ;;
    --config=*) CONFIG=${1#*=} ;;
    --mechanics-only) mechanics_only=true ;;
    --dry-run) dry_run=true ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

[[ "$gpu" =~ ^[0-7]$ ]] || die "--gpu must be one physical GPU index from 0 through 7"
[[ "$output_dir" = /* ]] || die "--output-dir must be an absolute path"
[[ "$CONFIG" = /* ]] || die "--config must be an absolute path"
[[ -f "$CONFIG" ]] || die "probe config does not exist: $CONFIG"

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
  "$PYTHON"
  -m ember.evaluation_identity
  "--config=$CONFIG"
  "--policy-path=$policy_path"
  "--output-dir=$output_dir"
  "--physical-gpu=$gpu"
)

if $mechanics_only; then
  command+=(--mechanics-only)
fi

if $dry_run; then
  printf 'CUDA_VISIBLE_DEVICES=%q MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ' "$gpu"
  printf 'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 '
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

[[ ! -e "$output_dir/probe_result.json" ]] || die "refusing to overwrite a completed probe: $output_dir"
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

PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" /usr/bin/time -v "${command[@]}"
