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

Owner update, 2026-07-19: Gate -1 is sealed as passed with residuals and Gate 0
is passed for rapid mechanism development, while retaining its limited
task3/task4, n=32-per-arm coverage. The former strict per-task CI veto and fixed
n=32 sufficiency rule are historical contracts, not current Writer blockers.
Use enough independent samples for the observed variance and expected effect,
judge multi-category validation evidence as a whole, and report adverse results
unchanged. The active next stage is the direct Writer cold start specified in
`configs/writer_cold_start.toml`.

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
independent task behavior before target-task interaction. After supervised cold
start, an independent Writer-only RL stage freezes the base, uses each generated
LoRA only as the functional policy output, and updates Writer parameters from
rollout reward; it does not optimize generated LoRA in place. Ordinary task-local
RL is a separate experiment: it freezes Writer and base, then updates those same
LoRA parameters in place from the Writer initialization.
Writer emits no second object that constrains RL.

Gate 0 is not restricted to supervised fine-tuning. If supervised LoRA improves
independent query loss but lacks stable closed-loop utility, a bounded
source-only recovery may compare frozen base, supervised LoRA, matched zero-init
LoRA plus ordinary task-local RL, and supervised-init LoRA plus the identical RL.
Only task-local LoRA updates in this Gate: no Writer or shared state participates.

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
- Compute ceiling: effective 2026-07-19, at most eight A100 80GB GPUs across all
  EMBER work. Use all live-free devices for useful data/task/rollout parallelism
  and tune batch for roughly 10GB average headroom per A100; never manufacture
  utilization with dummy memory or interfere with unrelated jobs.

SmolVLA plus LIBERO is the formal mechanism-development surface, not a disposable
toy. OpenVLA-OFT starts only after task information, useful-update existence,
direct Writer zero-interaction utility, ordinary local RL, and the frozen-held
contract survive at lower cost.

### Scale confirmation

The initial OpenVLA-OFT candidate targets are the two 4096-to-4096 residual
linear layers in the L1 action head. Before launch, re-pin the active revision,
recompute the topology within the active eight-GPU ceiling, and run a measured
memory/throughput pilot. The confirmation may not change the scientific
contract merely to fit the larger model.

## 4. Information-flow experiment

The Writer input and online policy prompt are separate causal paths:

1. **Parameter-compilation setting:** the Writer sees language/video and the
   online policy receives a fixed neutral prompt.
2. **Practical instruction setting:** the Writer and online policy both receive
   the task instruction, with capacity- and inference-matched direct
   conditioning controls.

The current rapid-development arm is language-plus-video; the owner removed
standalone language-only and video-only Writer arms. Compare it first with the
frozen base and matched action-supervised task-local LoRA, and defer the broader
negative/retrieval/direct-generator matrix until the mechanism survives. The
first claim uses successful same-embodiment robot video with actions hidden
from the Writer; it does not claim human-video, cross-embodiment, real-robot,
or cheaper-data transfer.

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

The Gate-0 evidence contract distinguishes smoke, development, and
confirmation. The existing eight-rollout task-3/task-4 surfaces are development
smokes only; a single episode is 12.5 percentage points and one or two paired
wins cannot support a positive or negative mechanism claim. Candidate evidence
requires at least 32 paired rollouts per task/arm over multiple policy RNG
seeds, retained per-episode rows, paired bootstrap and exact intervals, and at
least two (preferably three) independent training seeds. Evaluation RNG does
not replace independent training. A separate confirmation uses source tasks or
physical init states disjoint from development, binds their identities/hashes
before LoRA outcomes, and preferably covers two to four distinct primitives
with result-blind base competence and headroom.

RL training and the primary Gate evaluation both execute horizon 16. Horizon 50
is reported separately as canonical/deployment robustness. Binary simulator
success stays primary; legal source-only grasp, correct object/region,
drawer-close, completion progress, and time-to-success are diagnostics. Report
drift/KL from the frozen base and from each arm's own RL initialization, and do
not count replay epochs over related transitions as new independent samples.

The executed temporal-credit path is named `custom chunk-level flow-loss PPO
pilot`: eight flow-sample losses are averaged before one transition ratio. The
historical `flow_sample_group_size` field never controlled runtime ratio
granularity and is forbidden in the active contract. This pilot is not a full
FPO++ reproduction. Before an ordinary task-local RL negative claim, use the
bounded faithful FPO++ core: group-size-one/per-flow-sample ratios, the
MSE-preserving modified Huber loss, old-loss and log-ratio clamps, PPO trust
region, and matched horizon 16. This imports the scientific core, not the
paper-scale budget.

The last-two q/v rank-8 pilot and the subsequent default-like rank-16 run use 12
support demonstrations, 750 optimizer steps, and a custom early-candidate
schedule. They diagnose only that acquisition recipe. Regardless of outcome,
they cannot alone establish a final LoRA, Gate 0, Writer, or EMBER negative, and
the old rank-16 stop clause is superseded wherever a known-positive mature LoRA
competence control is still missing. Support/query separation is an independence
principle, not a fixed small-data or small-step budget.

Before sealing Writer targets, run one bounded mature-recipe positive control.
The executable predeclaration uses 40 source support demonstrations (episode
roles `writer_spec`, `source_base_fit`, and `oracle_support`), six independent
source query demonstrations, and fresh source closed-loop init states 40--47.
Actions from source support may supervise this Gate 0 oracle but remain hidden
from Writer inputs. Validation and held numeric access remain zero. The primary
LoRA uses the 37 [SmolVLA v0.6.0 default-like PEFT targets](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/policies/smolvla/modeling_smolvla.py#L2816-L2833),
rank 32, alpha 16, dropout 0, Gaussian exact-physical-zero initialization,
20k steps, effective batch 64, SmolVLA AdamW plus warmup/cosine scheduling,
compatible 90--100% random-resized crops, and fixed-final-step selection.
The 20k value is a maximum mature budget, not one blind run: execute the same
trajectory through exact-resume candidate boundaries 1k, 2k, 5k, 10k, and 20k.
Apply the pre-outcome source-query continuation rules after each segment, keep
the first segment below about 30 minutes, and do not consume the final fresh
closed-loop surface until the final candidate is frozen.

This provenance is deliberately qualified. SmolVLA's roughly 50-episode,
batch-64, 20k-step successful recipe trains its action expert/projections and
is not validated LoRA evidence; LeRobot's PEFT default targets/rank are an API
anchor only. [OpenVLA](https://github.com/openvla/openvla/blob/main/vla-scripts/finetune.py)
and [OpenVLA-OFT](https://github.com/moojink/openvla-oft/blob/main/vla-scripts/finetune.py)
provide empirical rank-32, broad-support, long-training and augmentation anchors
on another architecture, so only compatible principles transfer. A mechanically
valid primary failure permits at most one predeclared all-action-expert-linear
rank-32 compatibility recovery with the same recipe and unchanged thresholds.
No unbounded literature sweep, layer/rank grid, split change, or held-driven
choice is permitted.

Once closed-loop utility is demonstrated, permanently seal that empirically
successful support/rank/scale rather than shrinking it merely to simplify
Writer generation. If neither bounded LoRA positive control succeeds, preserve
the failure packet and escalate the LoRA-capacity decision before attributing
failure to Writer acquisition.

Every downstream Writer, zero-init ordinary-LoRA RL, and matched control must
use that identical sealed support and parameter budget. If emitting a broad
adapter is difficult,
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
delta or update imitation is allowed only as a predeclared auxiliary. Rapid
Writer development requires zero-interaction functional utility relative to the
frozen base; matched direct task-local LoRA is the action-supervised upper
bound. Freeze the complete strong-baseline matrix only for final experiments.

### 5.3 Ordinary task-local LoRA RL

Run three core causal arms: A) zero-LoRA initialization plus ordinary RL; B)
cold-start Writer LoRA initialization plus identical RL; and C)
reward-outer-trained Writer LoRA initialization plus identical RL. Update the
same task-local LoRA under identical target layers/rank/count, RL algorithm,
hyperparameters, seeds, reward, interaction budget, and environment budget;
also retain the final declared baselines. Report J0, the full success/return curve, AUC,
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

Gate -1 is now resolved as **passed with residuals** under explicit owner
authority. The immutable action-hidden-video packet remains 19/24 (0.7917) for
ordered and wrong-video accuracy and 15/24 paired, below the original 0.80
content threshold with recorded drop-last sensitivity. This is sufficient for
the present staged program without rewriting the old threshold or result; no
additional compute is authorized merely to reach 0.80, and Gate -1 no longer
blocks Writer once Gate 0 evidence is sufficient.

These stages form one long-term Goal. Environment/code completion, exact
resume, throughput selection, Gate -1, Gate 0, a single training run, or merely
authorizing Writer cannot complete it. Positive completion requires the frozen
held Writer and strong-baseline result, matched A/B/C RL evidence, cold-start
versus source-reward-outer-trained Writer evidence, causal language/video
controls with predeclared seeds/confidence intervals and reproducible reruns,
and OpenVLA-OFT scale confirmation.

## 7. Eight-GPU efficiency and artifact rules

- Never allocate more than eight A100s across concurrent EMBER jobs. This
  supersedes the former four-GPU project ceiling without reviving historical
  eight-GPU experiment commands.
- Smoke on one GPU, then use the minimum faithful topology per job. When an
  independent task, arm, evaluation shard, or training seed is ready and GPUs
  are live-free, schedule it concurrently through the same canonical entrypoint
  rather than leaving devices idle.
  Do not rerun a full 1/2/4 scaling curve solely to polish systems evidence;
  when one DDP job scales poorly, occupy useful devices with independent
  arm/task/seed jobs instead.
- After correctness, tune batch, accumulation, feature caching, simulator/task
  parallelism, and I/O so an allocated A100 normally retains about 10GB average
  headroom; do not allocate dummy tensors merely to fill memory.
- Preserve global effective batch, sample/flow authority, optimizer steps, and
  schedules when comparing topologies.
- Resource parallelism changes scheduling only. It never changes model/LoRA
  capacity, batch, legal data, training steps/interactions, evaluation sample,
  or statistical units. Train a required model once; do not repeat it merely to
  occupy devices.
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

Gate -1 is closed as passed with residuals; do not reopen it to polish the
0.7917 result. Gate 0 is passed for mechanism development from the immutable
task-3/task-4 SFT-LoRA improvements, with the near-similar-task and n=32/arm
coverage limitation retained. A multi-category supplement is optional only if
frozen checkpoints can be reused without retraining or material preparation;
it does not block Writer.

The active work is direct Writer cold start on all 60 source tasks, followed by
five-category validation of frozen base, Writer LoRA, and the same-space direct
task-local LoRA upper bound. Use the current eight-GPU ceiling for useful work,
keep one-to-two-hour feedback segments and exact resume, publish raw stage
performance only after the full declared denominator, and retain a compact
video gallery. Then proceed in order to Writer-only RL, matched task-local LoRA
RL, source-only outer learning, and shared-frozen held evaluation. Do not build
recovery families or a bank/geometry branch. Update the durable state after
each real milestone without weights, datasets, private host details, or large
outputs in public Git.
