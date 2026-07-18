# Gate 0 SmolVLA Useful-Update Pilot Contract

This document explains the checked-in machine-readable contract in
`configs/gate_zero_oracle_pilot.toml`. The TOML is authoritative. This pilot was
frozen before any LIBERO-90 source policy training or policy outcome was read.

## Scientific scope

The pilot asks whether a rank-8 task-local physical update exists at four
predeclared SmolVLA action-expert matrices for resealed source tasks 3 and 4.
It is a source-only mechanism test. A positive result permits Gate 0 expansion;
it cannot pass Gate -1, pass Gate 1, or authorize Writer training by itself.

The shared source base uses all 60 resealed source tasks and only demonstrations
8--27. Vision and the VLM are frozen; the action expert and state projection are
fit with the already sealed all-source normalization. Each task-local oracle is
then initialized independently from the identical frozen source base and may
optimize only demonstrations 28--39.

## Access ledger

| Surface | Episodes | Allowed use before report freeze |
| --- | --- | --- |
| Writer specification | 0--7 | not used by Gate 0 optimization |
| Shared source-base fit | 8--27 | source-only shared fitting and normalization |
| Oracle support | 28--39 | task-local adapter optimization |
| Functional query | 40--45 | fixed-noise checkpoint selection only |
| Locked source report | 46--49 | inaccessible until selected adapter hashes are frozen |

The earlier Gate -1 video probe read RGB, and only RGB, from demonstrations
40--47. Therefore 46--47 are action/reward/policy-outcome locked rather than
fully untouched; 48--49 remain fully pristine. No validation or held numeric
field is legal in this pilot.

HDF5 demonstration initial states do not equal the official LIBERO pruned-init
rows. Offline action-loss evidence therefore uses the declared HDF5 episodes,
while closed-loop evidence independently uses official init-state indices
16--23. It never treats a demonstration number as a simulator init-state ID.

## Primary update and metrics

The primary adapter is rank 8, alpha 8, dropout zero, with a functional-zero
orthogonal initialization at the q/v projections of action-expert layers 14 and
15. Those are the last action self-attention and VLM-to-expert cross-attention
blocks. The four weight shapes imply exactly 40,320 trainable adapter scalars.

Checkpoint selection minimizes task-wise fixed-noise flow MSE on episodes
40--45 among candidates that satisfy the predeclared fixed-noise Gaussian mean-
action drift proxy. SmolVLA has no exact tractable policy likelihood; the proxy
is labeled as such. Final evidence uses episodes 46--49 plus matched official
rollouts for frozen base, own adapter, and swapped adapter.

The unchanged thresholds are median success gain of 15 percentage points,
median locked action-loss reduction of 20%, positive gain on at least 70% of
tasks, and median drift proxy no greater than 0.02. With two pilot tasks, the
positive-task rule requires both tasks to improve.

## Predeclared recovery and resources

Source-base failure is diagnosed before adapter failure. Its only predeclared
recovery is continuing the identical 10,000-step recipe to 20,000 steps once.
If the primary adapter is mechanically correct but representationally
insufficient, at most one source/query-selected scientific recovery may either
add `action_out_proj` at rank 8 or raise the same four matrices to rank 16.
Thresholds, splits, held access, and the shared-frozen contract cannot change.

One GPU is used for smoke and batch calibration; two GPUs may run the two task
oracles in parallel only after the measured launch contract. Technical batch
selection chooses the fastest microbatch that retains at least 10 GiB free on
an A100, while gradient accumulation preserves effective batch 64. Reusable raw
HDF5 streaming avoids a duplicate converted video dataset. Canonical reports
include a bounded local gallery; regenerable duplicate media and rotating
recovery checkpoints are cleaned only after hashes and retained evidence pass.
