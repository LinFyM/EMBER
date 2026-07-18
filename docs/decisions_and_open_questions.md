# Decisions, Open Questions, and Go/No-Go Gates

## Authority correction

On 2026-07-18 the owner confirmed that EMBER's original core was direct
language/action-hidden-video to task-specific LoRA generation. The mandatory
canonical-bank/geometry path in earlier plans was an assistant/expert addition
and is superseded. It is outside the current project and long-term Goal.

That removed mechanism used a shared bank as a basis/span, task-conditioned
soft geometry as a coordinate preconditioner, and residual escape to leave the
span. Current EMBER has no Writer-predicted bank, basis, mask, metric, radius,
learning rate, preconditioner, or other second object controlling RL. Do not
implement, benchmark, schedule, or reserve code paths for this route.

## Settled working decisions

1. **Operational claim.** The Writer is an amortized task-conditioned
   parameter-update generator/hypernetwork, not a universal meta-optimizer.
2. **Standalone project.** EMBER uses lessons recorded from MemLLM but no code,
   Wiki/QA mechanism, or architecture dependency.
3. **Current input.** Writer arms are language-only, action-hidden robot
   video-only, and language-plus-video within one embodiment and simulator
   dynamics family.
4. **Immediate utility.** A generated LoRA must improve behavior before held
   interaction to support the bootstrapping claim.
5. **Executable source supervision.** Source actions, demonstrations, teacher
   policies, and rewards may supervise the objective while remaining hidden
   from Writer input.
6. **Only one LoRA search space.** Target matrices, rank/scaling, and parameter
   count are predeclared. Writer emits all task-specific factors in this full
   allowed space; ordinary local RL updates the same factors in place.
7. **Functional primary loss.** Independent source-query action,
   flow-matching, behavioral, or return loss differentiates through functional
   adapter application. Factor MSE is never primary; oracle-update imitation is
   optional auxiliary only when frozen in advance.
8. **Frozen shared base.** The base stays frozen during direct Writer cold-start
   and default source reward/meta-RL. Inner learning updates task-local LoRA;
   source outer learning updates Writer parameters.
9. **Held freeze.** Base, Writer, encoders, target/rank, normalization,
   optimizer, thresholds, and all shared state freeze before held outcomes.
   Only the declared task-local LoRA may adapt from held reward.
10. **Strong baselines.** Capacity-matched DISC/HyPoGen-style generation,
    direct conditioning, standard task-specific LoRA, average/retrieval, and
    modality/negative controls are mandatory.
11. **Benchmark path.** SmolVLA plus LIBERO is the formal development surface;
    OpenVLA-OFT is confirmation only after the mechanism survives.
12. **Resource ceiling.** At most four A100 80GB GPUs. Topology is selected by
    measured useful throughput under invariant scientific budgets.
13. **Failure handling.** A failed gate triggers mechanics/data/access
    verification and bounded source/validation recovery, not silent contract
    weakening or immediate project abandonment.

## Current complete design

The active staged design contains exactly:

1. benchmark/specification information validity;
2. independent useful task-local LoRA oracles;
3. direct full-LoRA Writer acquisition with immediate utility;
4. matched-budget ordinary task-local LoRA RL;
5. source-only reward/delayed outer learning of better Writer initializations;
6. shared-frozen held evaluation with complete controls; and
7. conditional OpenVLA-OFT confirmation.

Canonical representation, shared update structure, predicted geometry,
residual escape, and shared-base outer training are not missing stages. A future
shared-base/shared-LoRA source outer experiment would require its own matched
ablation and justification and is not needed for completion.

## Resolved P0 contracts

### Scientific abstraction and distribution

The first claim concerns held-out compositions of known task-relevant roles in
one robot embodiment and simulator family. It does not claim new embodiments,
human video, real robots, or arbitrary task primitives. The permanently resealed
60/15/15 split defines source, validation, and reporting-only held tasks.

### Benchmark and base policy

Use pinned SmolVLA/LIBERO revisions and the canonical LIBERO-90 data/task/BDDL/
init-state/controller/normalization manifests. Start adapters at the declared
action-expert targets. One recorded target/rank expansion is a Gate-recovery
option only if useful-update evidence shows the first contract is insufficient.

### Writer meta-episode

- Writer support: legal language and/or action-hidden video.
- Source-only hidden executable surfaces: disjoint support/query/locked report
  demonstrations and later source reward rollouts.
- Base: frozen.
- Writer output: all LoRA factors for the declared matrices/rank.
- Primary objective: independent query functional loss.
- Selection: task-level uncertainty and zero-interaction functional utility.

### Task-local RL

Use the simplest stable ordinary LoRA optimizer compatible with SmolVLA and the
declared interaction budget. The optimizer must be identical across Writer and
baseline initializations. Sparse outcome reward is reported separately from any
shaping. No shared parameter updates occur inside a task.

### Source reward/meta objective

Adapt each source task's local LoRA, evaluate fresh query return, then update
Writer parameters through a predeclared differentiable path or stable estimator.
The shared base and any shared adapter remain frozen in the default mainline.

## P1 choices to freeze before Writer outcomes

### Writer architecture

- language/video encoder revisions and whether they are frozen;
- temporal representation and cache authority;
- missing-modality representation;
- tensor-generation parameterization and functional adapter API;
- capacity matching for the strongest direct generator; and
- source task/meta-batch sampling and optimizer schedule.

Architecture may directly generate flattened factors or use an internal neural
decoder, but its externally visible output must be the complete task-specific
LoRA. An internal implementation is not allowed to become a shared RL search
constraint.

### Baselines and controls

At minimum:

- frozen base/no adaptation;
- direct language/video conditioning;
- capacity-matched language-only DISC/HyPoGen-style parameter generation;
- standard task-specific LoRA trained with matched source supervision;
- ordinary task-local LoRA RL from standard initialization;
- average and nearest/retrieved source adapters;
- random/task-ID/scene-only adapters;
- language-only, video-only, combined, wrong same-scene video,
  shuffled/reversed video, first/last/static controls; and
- Writer initialization followed by the same ordinary local RL.

### Metrics

- zero-interaction action loss, success, and return;
- matched-minus-control specificity;
- success/return-versus-interaction AUC;
- steps/episodes to threshold and fixed-budget final performance;
- task-level confidence intervals and seed variance;
- policy/action drift and unrelated-task harm;
- gradient/estimator diagnostics where relevant; and
- GPU-hours, wall time, simulator throughput, peak memory, storage, and
  amortized versus deployment cost.

### Resource and artifact envelope

Every launch records exact GPUs, process topology, scientific budgets,
expected/observed memory, wall time, stop condition, resume, and cleanup. Use
feature/cache reuse when scientifically identical. Maintain Trackio and compact
local galleries; retain canonical evidence and prune only verified-regenerable,
unpinned bulk artifacts with a recorded audit.

## Explicitly out of current scope

- canonical operator/update banks;
- shared task-update subspaces or fixed/global policy subspaces;
- Writer-predicted bases, masks, gates, metrics, radii, learning rates, or
  preconditioners for local RL;
- residual escape from such a span;
- shared base/shared LoRA source outer training as a default stage;
- arbitrary third-person human tutorial video or cross-embodiment transfer;
- real-robot online RL and continual shared updates during deployment; and
- speculative code paths for a future unified multi-task/delayed-feedback
  program.

## Go/no-go gates

Gate outcomes are evidence and diagnosis points. Follow `AGENTS.md` recovery
rules before narrowing a claim.

### Gate -1: information and benchmark validity

Language/video controls must demonstrate task-relevant information beyond
scene/task-ID shortcuts on a mechanically valid, sealed benchmark surface.

### Gate 0: useful-update existence

Independent task-local LoRA must improve query and locked closed-loop source
behavior at the intended targets/rank without unacceptable drift. Failure first
tests mechanics, base competence, optimization, and one bounded target/rank
recovery.

### Gate A: direct Writer utility

The generated full LoRA must beat base, average, retrieval, random/task-ID,
direct-conditioning, standard-LoRA, and capacity-matched direct-generator
controls before target interaction. Failure localizes Writer input,
representation, supervision, or acquisition; it does not resurrect a bank.

### Gate B: ordinary local-RL value

Writer initialization must improve AUC, stability, or time-to-threshold under
the same ordinary LoRA optimizer and interaction budget without concealing a
material final-performance penalty.

### Gate C: source reward/meta learning

Source-only outer reward must improve future Writer zero-step initialization or
matched adaptation curves while the shared base remains frozen.

### Gate D: frozen meta-generalization

The selected direct Writer/local-RL method must hold on the predeclared held
split with every shared module frozen and only task-local LoRA adapting.

### Gate E: multimodal necessity

Action-hidden video must add value beyond language/task-ID and pass matched
wrong/content/temporal controls. Failure narrows the modality claim; it cannot
be disguised as a parameter-generation gain from direct conditioning.
