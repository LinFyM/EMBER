#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG="$ROOT/configs/gate_minus1_video_information_probe.toml"
PAIR_CONFIG="$ROOT/configs/gate_minus1_same_init_goal_probe.toml"
CONTRACT="$ROOT/configs/phase0.toml"
SEAL="$ROOT/configs/libero90_split_reseal.json"
PYTHON="$ROOT/.venv/bin/python"

gpu=""
output_dir=""
latest_link=""
manifest=""
dry_run=false

usage() {
  cat <<'EOF'
Usage: scripts/run_gate_minus1_video_information_probe.sh OPTIONS

Required:
  --gpu=INDEX              One free physical GPU index (0-7).
  --output-dir=ABS_PATH    Fresh external evidence directory.

Optional:
  --manifest=ABS_PATH      Resealed canonical manifest (default: current latest).
  --latest-link=ABS_PATH   Atomic report link (default: video_information_latest).
  --dry-run                Print the fully offline command without allocating a GPU.
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
    --manifest=*) manifest=${1#*=} ;;
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
  latest_link="$(dirname "$output_dir")/video_information_latest"
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
[[ -f "$CONFIG" && -f "$PAIR_CONFIG" && -f "$CONTRACT" && -f "$SEAL" ]] ||
  die "checked-in video probe authority is incomplete"

if [[ -z "$manifest" ]]; then
  manifest="$EMBER_OUTPUT_ROOT/phase0/libero90_manifest/latest/manifest.json"
fi
[[ "$manifest" = /* && -f "$manifest" ]] || die "canonical manifest is missing: $manifest"
[[ ! -e "$output_dir" ]] || die "refusing to reuse output directory: $output_dir"

readarray -t authorities < <(
  "$PYTHON" -c '
import sys,tomllib
c=tomllib.load(open(sys.argv[1], "rb"))
print(c["datasets"]["libero_90"]["revision"])
print(c["datasets"]["libero_90"]["subdir"])
m=c["models"]["smolvlm_constructor_dependency"]
print(m["repo_id"].split("/")[-1])
print(m["revision"])
' "$CONTRACT"
)
dataset_root="$EMBER_DATA_ROOT/LIBERO-datasets/${authorities[0]}/${authorities[1]}"
model_path="$EMBER_ASSET_ROOT/models/${authorities[2]}/${authorities[3]}"
[[ -d "$dataset_root" && -f "$model_path/model.safetensors" ]] ||
  die "pinned dataset or model snapshot is incomplete"

command=(
  "$PYTHON" -m ember.video_information_probe
  --config "$CONFIG"
  --source-pair-config "$PAIR_CONFIG"
  --contract "$CONTRACT"
  --seal "$SEAL"
  --manifest "$manifest"
  --dataset-root "$dataset_root"
  --model-path "$model_path"
  --output-dir "$output_dir"
  --latest-link "$latest_link"
  --physical-gpu "$gpu"
)

if $dry_run; then
  printf 'CUDA_VISIBLE_DEVICES=%q HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ' "$gpu"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

manifest_dir=$(dirname "$(readlink -f "$manifest")")
[[ -f "$manifest_dir/checksums.sha256" ]] || die "manifest checksum file is missing"
(cd "$manifest_dir" && sha256sum -c --quiet checksums.sha256)

active_compute=$(
  nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null |
    sed '/^[[:space:]]*$/d'
)
[[ -z "$active_compute" ]] || die "GPU $gpu has active compute PID(s): ${active_compute//$'\n'/,}"
memory_used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
[[ "$memory_used" =~ ^[0-9]+$ ]] || die "cannot parse GPU $gpu memory usage"
((memory_used < 1000)) || die "GPU $gpu already uses ${memory_used} MiB"

personal_root=$(dirname "$EMBER_OUTPUT_ROOT")
personal_bytes=$(du -sb "$personal_root" | awk '{print $1}')
expected_bytes=214748365
personal_cap=$((500 * 1024 * 1024 * 1024))
((personal_bytes + expected_bytes <= personal_cap)) || die "projected output exceeds the 500 GiB personal cap"
available_bytes=$(df --output=avail -B1 "$personal_root" | tail -n 1 | tr -d '[:space:]')
((available_bytes > expected_bytes)) || die "insufficient filesystem space for the probe"

export CUDA_VISIBLE_DEVICES="$gpu"
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
