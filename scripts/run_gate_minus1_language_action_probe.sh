#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ACTION_CONFIG="$ROOT/configs/gate_minus1_language_action_probe.toml"
PILOT_CONFIG="$ROOT/configs/gate_minus1_specification_pilot.toml"
CONTRACT="$ROOT/configs/phase0.toml"
PYTHON="$ROOT/.venv/bin/python"

gpu=""
output_dir=""
latest_link=""
prior_result=""
dry_run=false

usage() {
  cat <<'EOF'
Usage: scripts/run_gate_minus1_language_action_probe.sh OPTIONS

Required:
  --gpu=INDEX              One physical GPU index (0-7).
  --output-dir=ABS_PATH    Fresh external evidence directory.

Optional:
  --prior-result=ABS_PATH  Frozen overlap pilot result (default: canonical recovery2).
  --latest-link=ABS_PATH   Atomic report link (default: "language_action_latest").
  --dry-run                Print the offline command without touching GPU state.
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
    --prior-result=*) prior_result=${1#*=} ;;
    --latest-link=*) latest_link=${1#*=} ;;
    --dry-run) dry_run=true ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

[[ "$gpu" =~ ^[0-7]$ ]] || die "--gpu must be one physical GPU index from 0 through 7"
[[ "$output_dir" = /* ]] || die "--output-dir must be an absolute path"
if [[ -z "$latest_link" ]]; then
  latest_link="$(dirname "$output_dir")/language_action_latest"
fi
[[ "$latest_link" = /* ]] || die "--latest-link must be an absolute path"

if [[ -f "$ROOT/.env.local" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env.local"
  set +a
fi

: "${EMBER_ASSET_ROOT:?set EMBER_ASSET_ROOT or provide it in .env.local}"
: "${EMBER_DATA_ROOT:?set EMBER_DATA_ROOT or provide it in .env.local}"
: "${EMBER_OUTPUT_ROOT:?set EMBER_OUTPUT_ROOT or provide it in .env.local}"
: "${HF_HOME:?set HF_HOME or provide it in .env.local}"
: "${LIBERO_CONFIG_PATH:?set LIBERO_CONFIG_PATH or provide it in .env.local}"
[[ -x "$PYTHON" ]] || die "locked environment is missing; run scripts/bootstrap_env.sh"
if [[ -z "$prior_result" ]]; then
  prior_result="$EMBER_OUTPUT_ROOT/gate_minus1/specification/pilot_recovery2_20260717T180100Z/probe_result.json"
fi
[[ "$prior_result" = /* && -f "$prior_result" ]] || die "prior pilot result is missing"
[[ ! -e "$output_dir" ]] || die "refusing to reuse output directory: $output_dir"

smoke_revision=$(
  "$PYTHON" -c \
    'import sys,tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["models"]["smolvla_libero_smoke"]["revision"])' \
    "$CONTRACT"
)
policy_path="$EMBER_ASSET_ROOT/runtime/smolvla_libero/$smoke_revision"

command=(
  "$PYTHON" -m ember.language_action_probe
  --action-config "$ACTION_CONFIG"
  --pilot-config "$PILOT_CONFIG"
  --prior-result "$prior_result"
  --contract "$CONTRACT"
  --policy-path "$policy_path"
  --output-dir "$output_dir"
  --latest-link "$latest_link"
  --physical-gpu "$gpu"
)

if $dry_run; then
  printf 'CUDA_VISIBLE_DEVICES=%q MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ' "$gpu"
  printf 'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 '
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

active_compute=$(
  nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null |
    sed '/^[[:space:]]*$/d'
)
[[ -z "$active_compute" ]] || die "GPU $gpu has active compute PID(s): ${active_compute//$'\n'/,}"
memory_used=$(
  nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits |
    tr -d '[:space:]'
)
[[ "$memory_used" =~ ^[0-9]+$ ]] || die "cannot parse GPU $gpu memory usage"
((memory_used < 1000)) || die "GPU $gpu already uses ${memory_used} MiB"

export CUDA_VISIBLE_DEVICES="$gpu"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" -m ember.phase0_runtime \
  --contract "$CONTRACT" \
  --asset-root "$EMBER_ASSET_ROOT" \
  --data-root "$EMBER_DATA_ROOT" \
  --libero-config-root "$LIBERO_CONFIG_PATH"

exec /usr/bin/time -v "${command[@]}"
