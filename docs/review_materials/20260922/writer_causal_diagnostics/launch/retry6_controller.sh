#!/usr/bin/env bash
set -uo pipefail

STUDY=/data0/user/ymdai/ember_runs/writer_causal_diagnostics_20260922
WORKER="$STUDY/launch/retry6_run_worker.sh"
RUNTIME=/data1/user/ymdai/projects/EMBER/.codex/tmp/writer-causal-runtime-107806f5
PYTHON=/data1/user/ymdai/projects/EMBER/.venv/bin/python
PRIOR_VALID_CONTROLLER_ELAPSED=3873
REMAINING_WALLCLOCK=$((7200 - PRIOR_VALID_CONTROLLER_ELAPSED))
START_EPOCH=$(date +%s); DEADLINE_EPOCH=$((START_EPOCH + REMAINING_WALLCLOCK))
SIGNAL=ember-writer-causal-diagnostics-20260922-retry6-complete
printf '{"started_epoch":%s,"deadline_epoch":%s,"remaining_workload_wallclock_seconds":%s,"original_workload_wallclock_seconds":7200,"prior_valid_controller_elapsed_seconds":3873,"reused_complete_d1_parts":true,"reused_complete_d3_parts":true,"purpose":"repair_D2_group_reporting_field_only","runtime_commit":"107806f5"}\n' "$START_EPOCH" "$DEADLINE_EPOCH" "$REMAINING_WALLCLOCK" >"$STUDY/launch/retry6_controller_started.json"
PIDS=()
finish() { code="$1"; printf '%s\n' "$code" >"$STUDY/launch/retry6_controller.exit"; tmux wait-for -S "$SIGNAL" || true; exit "$code"; }
start() { host="$1"; gpu="$2"; name="$3"; shift 3; if (( $(date +%s) >= DEADLINE_EPOCH )); then return 124; fi; ssh -o BatchMode=yes "$host" "$WORKER" "$STUDY" "$name" "$gpu" "$DEADLINE_EPOCH" "$@" & PIDS+=("$!"); }
wait_group() { code=0; for pid in "${PIDS[@]}"; do wait "$pid" || code=1; done; PIDS=(); return "$code"; }
run_finalize() { PYTHONPATH="$RUNTIME/src" "$PYTHON" "$RUNTIME/scripts/run_writer_causal_diagnostics.py" --output "$STUDY" finalize --d3-status complete >"$STUDY/launch/retry6_finalize.stdout.log" 2>"$STUDY/launch/retry6_finalize.stderr.log"; code=$?; printf '%s\n' "$code" >"$STUDY/launch/retry6_finalize.exit"; return "$code"; }

for branch in J Q M; do test -f "$STUDY/microtrain/$branch/completion.json" || finish 98; done
start gpu01 0 d2_C600 d2 --asset C600 --device cuda:0 || finish 124
start gpu02 1 d2_O1200 d2 --asset O1200 --device cuda:0 || finish 124
wait_group || finish 1
run_finalize || finish 1
finish 0
