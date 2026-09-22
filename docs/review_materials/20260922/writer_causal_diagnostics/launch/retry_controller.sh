#!/usr/bin/env bash
# Fixed retry controller.  Only the verified asset-root repair differs from the retained failed launch.
set -uo pipefail

STUDY=/data0/user/ymdai/ember_runs/writer_causal_diagnostics_20260922
WORKER="$STUDY/launch/retry_run_worker.sh"
RUNTIME=/data1/user/ymdai/projects/EMBER/.codex/tmp/writer-causal-runtime-7fe99d30
PYTHON=/data1/user/ymdai/projects/EMBER/.venv/bin/python
START_EPOCH=$(date +%s)
DEADLINE_EPOCH=$((START_EPOCH + 7200))
D3_MIN_REMAINING=3900
SIGNAL=ember-writer-causal-diagnostics-20260922-retry-complete

printf '{"started_epoch":%s,"deadline_epoch":%s,"wallclock_limit_seconds":7200,"d3_min_remaining_seconds":3900}\n' \
  "$START_EPOCH" "$DEADLINE_EPOCH" >"$STUDY/launch/retry_controller_started.json"

PIDS=()

finish() {
  code="$1"
  printf '%s\n' "$code" >"$STUDY/launch/retry_controller.exit"
  tmux wait-for -S "$SIGNAL" || true
  exit "$code"
}

start() {
  host="$1"
  gpu="$2"
  name="$3"
  shift 3
  if (( $(date +%s) >= DEADLINE_EPOCH )); then
    return 124
  fi
  ssh -o BatchMode=yes "$host" "$WORKER" "$STUDY" "$name" "$gpu" "$DEADLINE_EPOCH" "$@" &
  PIDS+=("$!")
}

wait_group() {
  code=0
  for pid in "${PIDS[@]}"; do
    wait "$pid" || code=1
  done
  PIDS=()
  return "$code"
}

run_one() {
  start "$@" || finish 124
  wait_group || finish 1
}

run_finalize() {
  status="$1"
  PYTHONPATH="$RUNTIME/src" "$PYTHON" "$RUNTIME/scripts/run_writer_causal_diagnostics.py" \
    --output "$STUDY" finalize --d3-status "$status" \
    >"$STUDY/launch/retry_finalize.stdout.log" 2>"$STUDY/launch/retry_finalize.stderr.log"
  code=$?
  printf '%s\n' "$code" >"$STUDY/launch/retry_finalize.exit"
  return "$code"
}

run_one gpu01 0 d1_path_C600 d1-path --asset C600 --device cuda:0
run_one gpu01 0 d1_path_O1200 d1-path --asset O1200 --device cuda:0
run_one gpu01 0 d1_path_C1200 d1-path --asset C1200 --device cuda:0
run_one gpu01 0 d1_baseline_M300 d1-baseline --asset M300 --device cuda:0
run_one gpu01 0 d1_baseline_Source d1-baseline --asset Source --device cuda:0

run_one gpu01 0 d1_rollout_C600_CC d1-rollout --asset C600 --arm CC --device cuda:0 --physical-gpu-id 0
run_one gpu01 0 d1_rollout_C600_CW d1-rollout --asset C600 --arm CW --device cuda:0 --physical-gpu-id 0
run_one gpu01 0 d1_rollout_C600_WC d1-rollout --asset C600 --arm WC --device cuda:0 --physical-gpu-id 0
run_one gpu01 0 d1_rollout_C600_WW d1-rollout --asset C600 --arm WW --device cuda:0 --physical-gpu-id 0
run_one gpu01 0 d1_rollout_C600_CO d1-rollout --asset C600 --arm CO --device cuda:0 --physical-gpu-id 0
run_one gpu01 0 d1_rollout_O1200_CC d1-rollout --asset O1200 --arm CC --device cuda:0 --physical-gpu-id 0
run_one gpu01 0 d1_rollout_O1200_WW d1-rollout --asset O1200 --arm WW --device cuda:0 --physical-gpu-id 0

run_one gpu01 0 d2_C600 d2 --asset C600 --device cuda:0
run_one gpu01 0 d2_O1200 d2 --asset O1200 --device cuda:0

if (( $(date +%s) + D3_MIN_REMAINING <= DEADLINE_EPOCH )); then
  run_one gpu01 0 d3_J d3 --branch J --device cuda:0 --physical-gpu-id 0
  run_one gpu01 0 d3_Q d3 --branch Q --device cuda:0 --physical-gpu-id 0
  run_one gpu01 0 d3_M d3 --branch M --device cuda:0 --physical-gpu-id 0
  run_finalize complete || finish 1
else
  printf '{"status":"not_started_insufficient_preregistered_time","checked_epoch":%s,"deadline_epoch":%s,"required_remaining_seconds":3900}\n' \
    "$(date +%s)" "$DEADLINE_EPOCH" >"$STUDY/launch/retry_d3_not_started.json"
  run_finalize not_started_insufficient_preregistered_time || finish 1
fi
finish 0
