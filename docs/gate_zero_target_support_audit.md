# Gate 0 target-support recovery contract

This is the single bounded recovery authorized after the frozen two-task Gate 0
report failed. It is predeclared before any target-support fit outcome. It does
not lower the original Gate thresholds, access held data, authorize Writer by
itself, or turn the pilot target set into the final Writer contract.

The executable authority is
`configs/gate_zero_target_support_audit.toml`. It binds the failed locked report,
its immutable selection grant, the source-base checkpoint, the source-only
manifest, and all prior contracts by SHA256.

## Rank-8 support candidates

| ID | Exact scope | Targets | Trainable LoRA parameters |
| --- | --- | ---: | ---: |
| `last_two_qv_r8` | q/v in action-expert layers 14--15 | 4 | 40,320 |
| `all_expert_qv_r8` | q/v in all 16 action-expert layers | 32 | 322,560 |
| `official_default_r8` | all expert q/v plus state/action/time projections | 37 | 371,328 |

Every candidate uses rank 8, alpha 8, dropout 0, exact-zero physical LoRA
initialization, the same source support/query rows, sampler, effective batch 64,
and common random numbers. The one acquisition repair is AdamW learning rate
`1e-4` instead of `3e-4`, with query candidates at steps
`0,25,50,100,150,250,500,750`. This directly tests the observed early
overfit/drift failure without adding an optimizer search.

## Selection, confirmation, and stop rule

1. Each task/support pair selects its minimum fixed-query flow MSE candidate
   under the unchanged `0.02` drift cap. Locked report demos are inaccessible.
2. Query screening ranks supports by positive-task count, median query loss
   reduction, then fewer parameters. All rank-8 candidates receive matched
   source closed-loop screening on the untouched init states 24--31 so the
   smallest support satisfying the unchanged Gate criteria can win.
3. The winner is frozen before confirmation. Confirmation uses untouched init
   states 32--39 and only then reopens demos 46--49 as the locked reporting
   surface; no selection may change afterward.
4. Only if no rank-8 support passes the declared query plus screening contract,
   one rank-16 version of the best declared support is allowed. It uses fresh
   screening/confirmation init states and no other rank, layer, or optimizer
   search. If that also fails, preserve the failure packet and return to the
   owner decision point rather than weakening the claim.

The six primary fits run through the existing recoverable oracle fitter and
Trackio project `EMBER_gate0`; there is no second trainer. At most four
independent jobs run concurrently. Selected states, compact metrics, decisions,
one bounded video gallery, telemetry, and checksums are retained; completed
recovery state and superseded regenerable media are removed after validation.

