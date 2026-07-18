# EMBER Active Execution Brief

## 1. Authority and objective

This file is the active research contract. `docs/expert_plan.md` remains an
unchanged historical expert record; its eight-GPU assumption and its proposed
canonical-bank/geometry route are not active authority. The owner reviewed the
original design conversation on 2026-07-18 and superseded every current Goal or
document clause that made a canonical representation Gate a Writer prerequisite.
The legacy `task_geometry` field in the already hash-bound `configs/phase0.toml`
is preserved as non-executable provenance so completed evidence is not
retroactively rewritten; it has no scheduling or implementation authority.

The current EMBER objective is to test this causal chain within one robot
embodiment and simulator dynamics family:

```text
language + action-hidden robot video
    -> direct Writer
    -> complete task-specific LoRA initialization
    -> immediate zero-interaction functional utility
    -> ordinary matched-budget task-local LoRA RL in the same LoRA space
    -> source-only reward/meta objective improves future Writer initializations
    -> shared-frozen held-task evaluation
```

The static Writer is a **task-conditioned hypernetwork and complete-LoRA
initializer**. It is not a universal meta-optimizer and does not predict an
update direction or constrain the later ordinary RL search.

## 2. Exact parameter and adaptation contract

The common structural search space is fixed before outcomes:

- an enumerated set of target matrices;
- LoRA rank and therefore parameter count;
- initialization, scaling, precision, and application semantics; and
- the optimizer and interaction budget used by ordinary task-local RL.

LoRA is the only adaptation mechanism in the current EMBER project. Do not add
bottleneck adapters, IA3, prefix tuning, a shared base adapter/shared LoRA, or
another parallel parameter-efficient state. Although LoRA is low-rank relative
to full fine-tuning, jointly trainable A/B factors are not a fixed shared linear
span of the removed bank kind.

For each task, the Writer emits all LoRA factors required by that contract. The
generated LoRA is applied functionally to the frozen shared base and must improve
independent task behavior before target-task interaction. Ordinary task-local RL
then updates those same LoRA parameters in place from the Writer initialization.
The Writer emits no second object that constrains RL.

The removed assistant/expert route had a different purpose: a canonical bank
supplied a shared basis/span, task-conditioned soft geometry scaled or
preconditioned bank coordinates, and residual escape allowed optimization to
leave that span. Together they imposed a second, narrower task-conditioned
search space or preconditioner on local RL. Canonical banks, shared task-update
subspaces, predicted bases, masks, metrics, radii, learning rates, soft geometry,
and residual escape are all outside the current EMBER project and long-term
Goal. Do not implement, benchmark, schedule, or reserve code paths for them.

A future independent program could reconsider shared structure only after a new
bottleneck and matched evidence justify it; no bank is presumed. Updating shared
base weights or a shared LoRA during source outer RL would likewise be a
separate matched ablation, not the default and not required for completion.

## 3. Model and benchmark path

### Development path

- Policy: immutable `lerobot/smolvla_base` revision.
- Simulator: pinned LIBERO with audited LIBERO-90 task map, language, BDDL/init
  authority, cameras, controller, demonstrations, and source-only normalization.
- Gate 0 pilot target: the predeclared last-two action-expert q/v matrices and
  rank-8 LoRA contract. This is a low-cost useful-update probe, not the final
  Writer support.
- Compute ceiling: at most four A100 80GB GPUs across all EMBER work.

SmolVLA plus LIBERO is the formal mechanism-development surface, not a disposable
toy. OpenVLA-OFT starts only after task information, useful-update existence,
direct Writer zero-interaction utility, ordinary local RL, and the frozen-held
contract survive at lower cost.

### Scale confirmation

The initial OpenVLA-OFT candidate targets are the two 4096-to-4096 residual
linear layers in the L1 action head. Before launch, re-pin the active revision,
recompute the four-GPU recipe, and run a measured memory/throughput pilot. The
confirmation may not change the scientific contract merely to fit the larger
model.

## 4. Information-flow experiment

The Writer input and online policy prompt are separate causal paths:

1. **Parameter-compilation setting:** the Writer sees language/video and the
   online policy receives a fixed neutral prompt.
2. **Practical instruction setting:** the Writer and online policy both receive
   the task instruction, with capacity- and inference-matched direct
   conditioning controls.

Required specification arms include language-only, video-only,
language-plus-video, wrong same-scene video, shuffled/reversed video,
first/last/scene-only video, task-ID, average adapter, nearest/retrieved adapter,
direct conditioning, standard task-specific LoRA, and a capacity-matched
HyPoGen/DISC-style direct parameter generator. The first claim uses successful
same-embodiment robot video with actions hidden from the Writer; it does not
claim human-video, cross-embodiment, real-robot, or cheaper-data transfer.

## 5. Training contract

### 5.1 Gate 0 independent task-local oracle

Fit each source-task oracle independently in the exact target-layer/rank
contract. Select on an independent source query surface and report a locked
closed-loop source surface. A positive oracle proves that useful updates exist
and supplies an upper bound and strong baseline; it does not prove that
language/video can acquire the update, and its parameters are not the sole
teacher that Writer must imitate.

Gate 0 includes a bounded LoRA capacity audit: compare zero/base with the
independently trained task-local LoRA under the matched contract. An
action-expert partial/full update or full fine-tuning may be recorded only as a
non-matched capability upper bound. If only such an upper bound succeeds, the
target-layer/rank LoRA contract is too narrow and enters recovery before Writer;
do not attribute that failure to Writer acquisition.

The current last-two q/v rank-8 contract remains the frozen Gate 0 pilot only.
Before any Writer target contract is sealed, run one predeclared,
source/validation-only, held-zero-access support audit. Its minimum candidate
set is: (a) the current last-two q/v pilot, (b) q/v in every action-expert
layer, and (c) support close to [SmolVLA v0.6.0's default PEFT target
set](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/policies/smolvla/modeling_smolvla.py#L2816-L2833):
all action-expert q/v projections plus `state_proj`, `action_in_proj`,
`action_out_proj`, `action_time_mlp_in`, and `action_time_mlp_out`. Rank may be
adjusted only within this bounded audit. Choose the smallest support that gives
robust closed-loop oracle utility, then permanently seal exact target names,
rank, alpha, dropout, and trainable parameter count. OpenVLA and
[OpenVLA-OFT](https://github.com/moojink/openvla-oft/blob/main/vla-scripts/finetune.py)
using rank 32 and `all-linear` are broad-support maturity references, not
instructions to copy their contract mechanically.

Every downstream Writer, zero-init ordinary-LoRA RL, average/retrieval,
language-only direct generator, and matched control must use that identical
sealed support and parameter budget. If emitting a broad adapter is difficult,
the bounded design options are structured layer/module embeddings, shared
layer-aware decoders, chunked or per-module generation, memory/Perceiver
queries, and type-specific heads. [SHINE](https://arxiv.org/abs/2602.06358)
and [Doc-to-LoRA](https://arxiv.org/abs/2602.15902) are architecture references
for scalable structured LoRA generation; their task domains or narrow target
choices are not EMBER target templates. None of these options may create a
canonical bank, shared update span, geometry, mask, radius, or later-RL search
constraint.

### 5.2 Direct Writer acquisition

Split each source meta-episode into a Writer-visible specification and an
independent executable query. Apply the generated LoRA functionally to the
frozen base and differentiate query action, flow-matching, behavioral, or return
loss through the adapter into the Writer. This independent functional loss is
primary.

Raw LoRA-factor MSE is prohibited as the primary objective because factor gauge
and parameter non-identifiability can decouple it from behavior. Oracle physical
delta or update imitation is allowed only as a predeclared auxiliary. Writer
selection requires zero-interaction functional utility plus matched modality,
negative, retrieval, average, direct-conditioning, standard-LoRA, and
capacity-matched direct-generator comparisons.

### 5.3 Ordinary task-local LoRA RL

Run three core causal arms: A) zero-LoRA initialization plus ordinary RL; B)
cold-start Writer LoRA initialization plus identical RL; and C)
reward-outer-trained Writer LoRA initialization plus identical RL. Update the
same task-local LoRA under identical target layers/rank/count, RL algorithm,
hyperparameters, seeds, reward, interaction budget, and environment budget;
also retain average, retrieval, language-only direct-generator, and other
declared baselines. Report J0, the full success/return curve, AUC,
time-to-threshold, J_K, J_K-J0, drift, uncertainty, memory, and wall time. A
claim about improved learning dynamics rather than a better initial policy also
requires a matched-initial-performance or equivalent control. There is no
Writer-predicted RL constraint object.

### 5.4 Source-only reward or delayed outer learning

During the default Writer reward/meta-RL stage:

- the shared base policy remains frozen;
- inner source adaptation updates task-local LoRA;
- the source-only outer objective updates Writer parameters so future
  zero-step LoRA initializations improve; and
- gradients may pass through the inner adaptation or use one predeclared stable
  estimator.

No extra shared base adapter or shared policy parameter is trained in this
mainline. The model-side learning required by the contract is the task-local
LoRA update itself.

### 5.5 Held-task evaluation

Before reading held outcomes, permanently freeze the base checkpoint, Writer,
encoders, target/rank contract, optimizer, budgets, thresholds, and all shared
state. At held evaluation the Writer may produce the zero-step task-local LoRA
from legal language/action-hidden video; only that predeclared task-local LoRA
may subsequently adapt from held reward. Held actions, labels, proprioceptive
trajectories, terminals, filenames, task IDs, and hidden normalization remain
inaccessible.

## 6. Staged authorization

1. **Phase 0:** reproduce the official path and freeze revisions, data/task
   manifests, normalization authority, evaluator identity, GPU/storage envelope,
   and reviewable artifacts.
2. **Gate -1:** validate task/specification information with same-scene,
   same-init where legal, language, video-content, and temporal controls.
3. **Gate 0:** establish an independent useful task-local LoRA oracle and its
   locked closed-loop gain.
4. **Direct Writer:** acquire complete task-specific LoRA with independent-query
   functional supervision and prove immediate utility.
5. **Ordinary local RL:** run the matched task-local LoRA comparison from Writer
   and standard initializations.
6. **Source reward/meta learning:** improve Writer initializations while base
   remains frozen and task-local LoRA is the only model-side adaptive state.
7. **Frozen held evaluation:** execute the sealed comparison once shared choices
   are frozen.
8. **OpenVLA-OFT confirmation:** scale only a surviving mechanism.

There is no mandatory Gate 1 between Gate 0 and Writer, and no positive
bank/geometry result is a completion criterion.

These stages form one long-term Goal. Environment/code completion, exact
resume, throughput selection, Gate -1, Gate 0, a single training run, or merely
authorizing Writer cannot complete it. Positive completion requires the frozen
held Writer and strong-baseline result, matched A/B/C RL evidence, cold-start
versus source-reward-outer-trained Writer evidence, causal language/video
controls with predeclared seeds/confidence intervals and reproducible reruns,
and OpenVLA-OFT scale confirmation.

## 7. Four-GPU efficiency and artifact rules

- Never allocate more than four A100s across concurrent EMBER jobs.
- Smoke on one GPU, then use at most one necessary four-GPU short
  throughput/stability window after the current clean world-size-4 recovery.
  Do not rerun a full 1/2/4 scaling curve solely to polish systems evidence;
  when one DDP job scales poorly, occupy useful devices with independent
  arm/task/seed jobs instead.
- After correctness, tune batch, accumulation, feature caching, simulator/task
  parallelism, and I/O so an allocated A100 normally retains about 10GB average
  headroom; do not allocate dummy tensors merely to fill memory.
- Preserve global effective batch, sample/flow authority, optimizer steps, and
  schedules when comparing topologies.
- Every multi-GPU run records devices, process topology, expected/observed
  memory, wall time, stop condition, resume, and cleanup.
- Maintain Trackio and bounded local HTML/video galleries for real-time or later
  inspection. Retain canonical evidence and `latest`; remove only validated,
  regenerable, unpinned bulk artifacts with a recorded cleanup.
- Exact-resume, bitwise/RNG digests, telemetry checksums, and log-byte identity
  are infrastructure diagnostics, not research Gates. Operational sufficiency
  means a loadable checkpoint with correct model, optimizer, scheduler, global
  step/data cursor and a short non-crashing resume whose loss or functional
  behavior is within a predeclared tolerance. For non-scientific anomalies use
  one reproduction, one narrow repair, and one verification, then proceed
  unless recoverability, sampled data, closed-loop success, a Gate decision,
  matched fairness, or held isolation can change.

## 8. Gate recovery without scientific drift

A failure packet must identify the exact metric, confidence interval, mechanics,
data/access authority, action/rollout diagnostics, likely layer, excluded
causes, ranked bounded remedies, and cheapest discriminating follow-up. Allowed
repairs include implementation/data fixes, one recorded target-layer/rank
expansion, better direct-LoRA parameterization or temporal representation, a
stable optimizer/RL estimator, and a smaller faithful model.

Do not use held labels or normalization, update shared state on held tasks,
change the split after held outcomes, lower thresholds to manufacture a pass,
delete strong baselines, hide extra budgets, or attribute direct-conditioning
gain to generated parameters. A negative conclusion follows only after
mechanical issues and reasonable bounded remedies are exhausted.

## 9. Immediate work package

The clean world-size-4 resume result is the final bounded resume verification.
Run at most one necessary short four-GPU throughput/stability window, then run
the already frozen source competence surface. Complete the remaining Gate -1
causal evidence and Gate 0 independent task-local oracle plus LoRA capacity
audit without reading validation/held numeric outcomes. If Gate 0 is positive,
implement only the direct full-LoRA Writer path above; do not create a
canonical-representation or geometry branch.
Continuously update `task_plan.md`, `findings.md`, and `progress.md` and commit
each reproducible milestone without weights, datasets, private host details, or
large outputs in public Git.
