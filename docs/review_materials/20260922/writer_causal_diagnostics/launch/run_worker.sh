#!/usr/bin/env bash
# One bounded GPU phase.  The controller supplies an absolute deadline and owns sequencing.
set -uo pipefail

STUDY="$1"
NAME="$2"
PHYSICAL_GPU="$3"
DEADLINE_EPOCH="$4"
shift 4

RUNTIME=/data1/user/ymdai/projects/EMBER/.codex/tmp/writer-causal-runtime-6924f5b8
PYTHON=/data1/user/ymdai/projects/EMBER/.venv/bin/python
EXPECTED=6924f5b84ef770299982c4d3b5118dc5dd1be474
EXIT_FILE="$STUDY/launch/${NAME}.exit"
STDOUT_FILE="$STUDY/launch/${NAME}.stdout.log"
STDERR_FILE="$STUDY/launch/${NAME}.stderr.log"

mkdir -p "$STUDY/launch"
observed=$(git -C "$RUNTIME" rev-parse HEAD 2>/dev/null || true)
if [[ "$observed" != "$EXPECTED" ]] || [[ -n "$(git -C "$RUNTIME" status --porcelain 2>/dev/null)" ]]; then
  printf '%s\n' 97 >"$EXIT_FILE"
  exit 97
fi

remaining=$((DEADLINE_EPOCH - $(date +%s)))
if (( remaining <= 0 )); then
  printf '%s\n' 124 >"$EXIT_FILE"
  exit 124
fi

export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export PYTHONPATH="$RUNTIME/src"
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

timeout --signal=TERM --kill-after=60 "${remaining}s" \
  "$PYTHON" "$RUNTIME/scripts/run_writer_causal_diagnostics.py" --output "$STUDY" "$@" \
  >"$STDOUT_FILE" 2>"$STDERR_FILE"
code=$?
printf '%s\n' "$code" >"$EXIT_FILE"
exit "$code"
