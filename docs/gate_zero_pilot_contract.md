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

The primary adapter is rank 8, alpha 8, dropout zero, with nonzero seeded `A`
and zero `B` at the q/v projections of action-expert layers 14 and 15. The
physical update and fixed-batch loss must both be exactly unchanged at step
zero. Those are the last action self-attention and VLM-to-expert cross-attention
blocks. The four weight shapes imply exactly 40,320 trainable adapter scalars.

The original checked-in value requested PEFT's `orthogonal` initializer while
describing it as functional-zero. A pre-training mechanics probe showed that
PEFT 0.19.1 instead produced nonzero physical deltas and changed one fixed
support-batch loss by `8.08e-5`. Before any multi-step base/oracle training,
query/report access, or rollout result, the contract was narrowly amended to
PEFT's default nonzero-`A`/zero-`B` no-op initialization. The observed loss
direction and magnitude played no role in this repair; the prior run and its
provenance remain retained.

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
oracles in parallel only after the measured launch contract. Calibration loads
the model once and streams the all-60-source base-fit surface with four
persistent workers. Before every microbatch candidate it restores one identical
trainable-state snapshot, resets the same global RNG, and constructs a new
empty AdamW optimizer. An absolute optimizer-step/effective-batch-slot sampler
keeps the 64 examples and fixed flow noise/time identical across accumulation
partitions. The retained per-step row-key digests must match before selection.
Each candidate gets one warmup and two measured optimizer steps; timing includes
data loading, while loss values and all policy outcomes are forbidden from the
result. Selection chooses the fastest candidate that retains at least 10 GiB
free and stops larger candidates after the first OOM.

The first resource run measured useful throughput and memory, but its candidates
successively reused updated model/optimizer state and changed draws with the
accumulation partition. Its artifact remains immutable diagnostic provenance;
its former microbatch-64 selection is explicitly unauthorized. One predeclared
matched recovery is required before a batch authority is frozen. Before the
formal 10,000-step fit, a separate source-only mechanics probe must compare
uninterrupted step 2 against step 1 checkpoint plus resume to step 2. Model,
optimizer, scheduler, RNG, and next sampler batch must match exactly.
Checkpoints use the pinned LeRobot safetensors state format inside an atomic
directory rename, hash every retained file, keep only the latest two
recoverable checkpoints, and retain step 10,000 as a candidate pending source-
competence evaluation rather than calling it a successful source base.

Reusable raw HDF5 streaming avoids a duplicate converted video dataset.
Canonical reports include a bounded local gallery; regenerable duplicate media
and rotating recovery checkpoints are cleaned only after hashes and retained
evidence pass.
