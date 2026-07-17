#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONTRACT="$ROOT/configs/phase0.toml"
PYTHON="$ROOT/.venv/bin/python"

output_dir=""
latest_link=""
workers=4
dry_run=false

usage() {
  cat <<'EOF'
Usage: scripts/build_libero_manifest.sh --output-dir=ABS_PATH [OPTIONS]

Build the canonical LIBERO-90 task/data/normalization manifest without a GPU.

Required:
  --output-dir=ABS_PATH    Fresh external artifact directory.

Optional:
  --latest-link=ABS_PATH   Symlink to update (default: sibling "latest").
  --workers=N              Concurrent file auditors, 1-8 (default: 4).
  --dry-run                Print resolved inputs without auditing files.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --output-dir=*) output_dir=${1#*=} ;;
    --latest-link=*) latest_link=${1#*=} ;;
    --workers=*) workers=${1#*=} ;;
    --dry-run) dry_run=true ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

[[ "$output_dir" = /* ]] || die "--output-dir must be an absolute path"
[[ "$workers" =~ ^[1-8]$ ]] || die "--workers must be an integer from 1 through 8"
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

: "${EMBER_DATA_ROOT:?set EMBER_DATA_ROOT or provide it in .env.local}"
: "${LIBERO_CONFIG_PATH:?set LIBERO_CONFIG_PATH or provide it in .env.local}"
[[ -x "$PYTHON" ]] || die "locked environment is missing; run scripts/bootstrap_env.sh"

revision=$(
  "$PYTHON" -c \
    'import sys,tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["datasets"]["libero_90"]["revision"])' \
    "$CONTRACT"
)
dataset_root="$EMBER_DATA_ROOT/LIBERO-datasets/$revision/libero_90"
hub_tree="$EMBER_DATA_ROOT/LIBERO-datasets/$revision/.cache/huggingface/trees/$revision.json"

[[ -d "$dataset_root" ]] || die "canonical dataset directory is missing: $dataset_root"
[[ -f "$hub_tree" ]] || die "pinned Hub tree metadata is missing: $hub_tree"
[[ -f "$LIBERO_CONFIG_PATH/config.yaml" ]] || die "LIBERO runtime config is missing"
[[ ! -e "$output_dir" ]] || die "refusing to overwrite output directory: $output_dir"

command=(
  "$PYTHON" -m ember.libero_manifest
  --workspace "$ROOT"
  --contract "$CONTRACT"
  --dataset-root "$dataset_root"
  --hub-tree "$hub_tree"
  --libero-config-root "$LIBERO_CONFIG_PATH"
  --output-dir "$output_dir"
  --latest-link "$latest_link"
  --workers "$workers"
)

if $dry_run; then
  printf 'PYTHONPATH=%q ' "$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec /usr/bin/time -v "${command[@]}"
