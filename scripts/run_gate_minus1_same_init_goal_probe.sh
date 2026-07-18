#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG="$ROOT/configs/gate_minus1_same_init_goal_probe.toml"
CONTRACT="$ROOT/configs/phase0.toml"
SEAL="$ROOT/configs/libero90_split_reseal.json"
PYTHON="$ROOT/.venv/bin/python"

output_dir=""
latest_link=""
manifest=""
dry_run=false

usage() {
  cat <<'EOF'
Usage: scripts/run_gate_minus1_same_init_goal_probe.sh OPTIONS

Required:
  --output-dir=ABS_PATH    Fresh external evidence directory.

Optional:
  --manifest=ABS_PATH      Resealed canonical manifest (default: current latest).
  --latest-link=ABS_PATH   Atomic report link (default: sibling "latest").
  --dry-run                Print the offline CPU-only command.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --output-dir=*) output_dir=${1#*=} ;;
    --manifest=*) manifest=${1#*=} ;;
    --latest-link=*) latest_link=${1#*=} ;;
    --dry-run) dry_run=true ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

[[ "$output_dir" = /* ]] || die "--output-dir must be an absolute path"
if [[ -z "$latest_link" ]]; then
  latest_link="$(dirname "$output_dir")/latest"
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
: "${LIBERO_CONFIG_PATH:?set LIBERO_CONFIG_PATH or provide it in .env.local}"
[[ -x "$PYTHON" ]] || die "locked environment is missing; run scripts/bootstrap_env.sh"

if [[ -z "$manifest" ]]; then
  manifest="$EMBER_OUTPUT_ROOT/phase0/libero90_manifest/latest/manifest.json"
fi
[[ "$manifest" = /* ]] || die "--manifest must be an absolute path"
[[ -f "$manifest" ]] || die "canonical manifest is missing: $manifest"
[[ -f "$CONFIG" && -f "$CONTRACT" && -f "$SEAL" ]] || die "checked-in probe authority is incomplete"

revision=$(
  "$PYTHON" -c \
    'import sys,tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["datasets"]["libero_90"]["revision"])' \
    "$CONTRACT"
)
dataset_root="$EMBER_DATA_ROOT/LIBERO-datasets/$revision/libero_90"
[[ -d "$dataset_root" ]] || die "canonical dataset directory is missing: $dataset_root"
[[ -f "$LIBERO_CONFIG_PATH/config.yaml" ]] || die "LIBERO runtime config is missing"
[[ ! -e "$output_dir" ]] || die "refusing to reuse output directory: $output_dir"

command=(
  "$PYTHON" -m ember.counterfactual_goal_probe
  --config "$CONFIG"
  --contract "$CONTRACT"
  --seal "$SEAL"
  --manifest "$manifest"
  --dataset-root "$dataset_root"
  --libero-config-root "$LIBERO_CONFIG_PATH"
  --output-dir "$output_dir"
  --latest-link "$latest_link"
)

if $dry_run; then
  printf 'CUDA_VISIBLE_DEVICES=%q HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ' ""
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

manifest_dir=$(dirname "$(readlink -f "$manifest")")
[[ -f "$manifest_dir/checksums.sha256" ]] || die "manifest checksum file is missing"
(cd "$manifest_dir" && sha256sum -c --quiet checksums.sha256)

export CUDA_VISIBLE_DEVICES=""
unset MUJOCO_GL PYOPENGL_PLATFORM
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" -m ember.phase0_runtime \
  --contract "$CONTRACT" \
  --asset-root "$EMBER_ASSET_ROOT" \
  --data-root "$EMBER_DATA_ROOT" \
  --libero-config-root "$LIBERO_CONFIG_PATH"

exec /usr/bin/time -v "${command[@]}"
