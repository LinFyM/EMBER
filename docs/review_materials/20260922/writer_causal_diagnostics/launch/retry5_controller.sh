#!/usr/bin/env bash
set -uo pipefail

STUDY=/data0/user/ymdai/ember_runs/writer_causal_diagnostics_20260922
WORKER="$STUDY/launch/retry5_run_worker.sh"
RUNTIME=/data1/user/ymdai/projects/EMBER/.codex/tmp/writer-causal-runtime-f9fa880b
PYTHON=/data1/user/ymdai/projects/EMBER/.venv/bin/python
PRIOR_VALID_CONTROLLER_ELAPSED=931
REMAINING_WALLCLOCK=$((7200 - PRIOR_VALID_CONTROLLER_ELAPSED))
START_EPOCH=$(date +%s); DEADLINE_EPOCH=$((START_EPOCH + REMAINING_WALLCLOCK)); D3_MIN_REMAINING=3900
SIGNAL=ember-writer-causal-diagnostics-20260922-retry5-complete
printf '{"started_epoch":%s,"deadline_epoch":%s,"remaining_workload_wallclock_seconds":%s,"original_workload_wallclock_seconds":7200,"prior_valid_controller_elapsed_seconds":931,"d3_min_remaining_seconds":3900,"reused_complete_d1_parts":true,"supersedes_implementation_commit":"75b1ead6","runtime_commit":"f9fa880b"}\n' "$START_EPOCH" "$DEADLINE_EPOCH" "$REMAINING_WALLCLOCK" >"$STUDY/launch/retry5_controller_started.json"
PIDS=()
finish() { code="$1"; printf '%s\n' "$code" >"$STUDY/launch/retry5_controller.exit"; tmux wait-for -S "$SIGNAL" || true; exit "$code"; }
start() { host="$1"; gpu="$2"; name="$3"; shift 3; if (( $(date +%s) >= DEADLINE_EPOCH )); then return 124; fi; ssh -o BatchMode=yes "$host" "$WORKER" "$STUDY" "$name" "$gpu" "$DEADLINE_EPOCH" "$@" & PIDS+=("$!"); }
wait_group() { code=0; for pid in "${PIDS[@]}"; do wait "$pid" || code=1; done; PIDS=(); return "$code"; }
run_finalize() { status="$1"; PYTHONPATH="$RUNTIME/src" "$PYTHON" "$RUNTIME/scripts/run_writer_causal_diagnostics.py" --output "$STUDY" finalize --d3-status "$status" >"$STUDY/launch/retry5_finalize.stdout.log" 2>"$STUDY/launch/retry5_finalize.stderr.log"; code=$?; printf '%s\n' "$code" >"$STUDY/launch/retry5_finalize.exit"; return "$code"; }

start gpu01 0 d2_C600 d2 --asset C600 --device cuda:0 || finish 124
start gpu02 1 d2_O1200 d2 --asset O1200 --device cuda:0 || finish 124
wait_group || finish 1

if (( $(date +%s) + D3_MIN_REMAINING <= DEADLINE_EPOCH )); then
  start gpu01 0 d3_J d3 --branch J --device cuda:0 --physical-gpu-id 0 || finish 124
  start gpu02 1 d3_Q d3 --branch Q --device cuda:0 --physical-gpu-id 1 || finish 124
  start gpu02 2 d3_M d3 --branch M --device cuda:0 --physical-gpu-id 2 || finish 124
  wait_group || finish 1
  run_finalize complete || finish 1
else
  printf '{"status":"not_started_insufficient_preregistered_time","checked_epoch":%s,"deadline_epoch":%s,"required_remaining_seconds":3900}\n' "$(date +%s)" "$DEADLINE_EPOCH" >"$STUDY/launch/retry5_d3_not_started.json"
  run_finalize not_started_insufficient_preregistered_time || finish 1
fi
finish 0
