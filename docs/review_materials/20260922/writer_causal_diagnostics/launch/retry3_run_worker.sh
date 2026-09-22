#!/usr/bin/env bash
set -uo pipefail

STUDY="$1"; NAME="$2"; PHYSICAL_GPU="$3"; DEADLINE_EPOCH="$4"; shift 4
RUNTIME=/data1/user/ymdai/projects/EMBER/.codex/tmp/writer-causal-runtime-75b1ead6
PYTHON=/data1/user/ymdai/projects/EMBER/.venv/bin/python
EXPECTED=75b1ead6ce76c0584a351538110da8b8a73d725f
EXIT_FILE="$STUDY/launch/retry3_${NAME}.exit"
STDOUT_FILE="$STUDY/launch/retry3_${NAME}.stdout.log"
STDERR_FILE="$STUDY/launch/retry3_${NAME}.stderr.log"
mkdir -p "$STUDY/launch"
observed=$(git -C "$RUNTIME" rev-parse HEAD 2>/dev/null || true)
if [[ "$observed" != "$EXPECTED" ]] || [[ -n "$(git -C "$RUNTIME" status --porcelain 2>/dev/null)" ]]; then printf '%s\n' 97 >"$EXIT_FILE"; exit 97; fi
remaining=$((DEADLINE_EPOCH - $(date +%s)))
if (( remaining <= 0 )); then printf '%s\n' 124 >"$EXIT_FILE"; exit 124; fi
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" PYTHONPATH="$RUNTIME/src" TOKENIZERS_PARALLELISM=false NCCL_P2P_DISABLE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
timeout --signal=TERM --kill-after=60 "${remaining}s" "$PYTHON" "$RUNTIME/scripts/run_writer_causal_diagnostics.py" --output "$STUDY" "$@" >"$STDOUT_FILE" 2>"$STDERR_FILE"
code=$?; printf '%s\n' "$code" >"$EXIT_FILE"; exit "$code"
