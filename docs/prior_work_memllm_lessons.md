# Prior Work: Transferable Lessons from MemLLM

## Document role and boundary

EMBER is a standalone project and does not depend on MemLLM code, benchmarks,
or architecture. MemLLM is recorded here because it is the preceding research
program in which the Writer idea was repeatedly implemented, falsified, and
refined. Its negative results contain practical lessons for EMBER.

The remote state summarized here was verified at
[`LinFyM/Episodic-Memory-Chatbot@8afb451`](https://github.com/LinFyM/Episodic-Memory-Chatbot/tree/8afb451bf7a9d150ed3654f7ca394fe15c485dfa)
on 2026-07-17. This document is a compact interpretation for EMBER, not a
replacement for that repository's evidence ledger.

## 1. What MemLLM was trying to demonstrate

The durable MemLLM question was not a particular Wiki QA score. Its target
contract was:

```text
raw information
-> signals available from the information itself
-> a fast learned parameter write without deployment-time backward
-> one fixed-capacity mergeable short-term parameter state
-> later source-hidden functional utility
-> outcome-based attribution of useful directions
-> selective consolidation and exact merge into the backbone
```

The deployed Writer could see raw input and natural forward signals, but not a
future question, answer, mapping, retrieval result, alias, or correctness label.
Training could use source-hidden supervision and reward to learn the writing
rule. This distinction between **allowed Writer input** and **allowed outer
training signal** directly motivates EMBER.

The current repository implements substantial infrastructure rather than only
a concept: standard mergeable Fast LoRA state, raw-input signal extraction,
source-local teacher and PPO paths, shared-state writes, matched/control gates,
delayed WriteJournal experiments, exact merge, and frozen evaluation tooling.
However, the full scientific program has not passed. The main unresolved
bottleneck is predicting a content-specific useful parameter action from raw
input and accumulating such actions safely.

## 2. High-information failures

### 2.1 Generic improvement is not content acquisition

An earlier coordinate-memory Writer produced nearly the same QA improvement
from Related and QA-blind Unrelated Wiki. The final states had cosine
`0.999399`. The apparent gain was therefore a generic QA adapter, not evidence
that the Writer had written the source content.

**Transfer to EMBER:** compare a matched task specification with shuffled,
source-disjoint, and task-ID controls. A policy that improves from any tutorial
video may only have received a generic action prior.

### 2.2 Representability is not amortized acquisition

Target-aware update oracles repeatedly showed that useful directions existed in
the declared low-rank coordinate. Some decoder and replay probes could reproduce
those target actions exactly. Yet raw-input Writers remained nearly orthogonal
to the useful target direction and failed held functional gates.

**Transfer to EMBER:** distinguish at least:

1. whether a useful adapter exists;
2. whether the selected LoRA/basis representation can express it;
3. whether language/video can predict it;
4. whether the predicted update remains useful on held-out tasks;
5. whether RL can refine and accumulate it.

An oracle pass at step 1 or exact parameter replay at step 2 does not validate
the Writer.

### 2.3 Parameter targets can be badly conditioned

Raw coefficient objectives were affected by heavy-tailed magnitudes,
factorization gauge, and weak alignment with actual physical/functional update
directions. Loss reduction was often dominated by magnitude while downstream
utility stayed flat.

**Transfer to EMBER:** do not make teacher-LoRA MSE the primary objective.
Prefer action/query behavior, return, normalized physical update direction, and
post-adaptation utility. Treat parameter imitation as auxiliary evidence.

### 2.4 Update diversity is not task specificity

Some learned updates were content-distinct in parameter cosine but provided no
matched-over-control benefit. Healthy gradients, low loss, nonzero norms, and
different updates were all insufficient.

**Transfer to EMBER:** success must be behavioral. Report matched gain,
matched-minus-control specificity, preservation/control harm, and held-task
return, not only adapter cosine or norm.

### 2.5 Attribution is not successful application

Delayed-credit experiments could identify a relevant historical write above
chance: journal specificity and top-1 attribution passed. The learned shared
transition still produced common-mode control harm and failed final utility.

**Transfer to EMBER:** showing that reward gradients reach the right Writer
output does not prove that the proposed optimizer update is safe, sufficiently
expressive, or useful. Credit assignment and policy improvement require
separate gates.

### 2.6 A learned shared coordinate may collapse task distinctions

Projection into a shared low-rank basis sometimes increased similarity between
matched and control updates, saturated a common norm cap, or preserved only a
small fraction of native proposal energy. A numerically convenient basis was
not necessarily a utility-preserving basis.

**Transfer to EMBER:** compare direct task-specific LoRA, a fixed global basis,
Writer-gated shared directions, and an oracle basis. Measure behavior after
projection and allow a bounded residual escape path.

### 2.7 Common-mode harm can be larger than matched gain

Several candidates produced a small positive matched mean while an unrelated
control gained as much or more, or while control harm dominated. Increasing the
update magnitude after seeing this result would amplify the wrong direction.

**Transfer to EMBER:** pair immediate utility with a behavioral trust region and
matched/control evaluation. Do not rescue weak direction learning by increasing
LoRA scale, epochs, or RL budget after reading the held result.

### 2.8 Mechanical correctness is not scientific success

MemLLM repeatedly passed CUDA execution, exact replay, cap safety, mergeability,
artifact completeness, and deterministic reconstruction while failing the
scientific utility gate.

**Transfer to EMBER:** maintain separate mechanical, representational,
optimization, and behavioral gates. A successful training run is not a
successful method.

## 3. Process lessons worth retaining

### Freeze the deployment contract first

Specify exactly what the Writer sees, which signals are training-only, which
state is task-local, what is frozen at evaluation, and when source information
disappears. Architecture should follow this contract rather than quietly
changing the task to make optimization easier.

### Use staged, predecessor-gated experiments

A useful order is:

```text
useful-update oracle
-> representation/replay gate
-> Writer acquisition gate
-> held immediate-utility gate
-> task-local adaptation gate
-> shared or long-horizon accumulation gate
-> full evaluation
```

Do not build downstream machinery before its predecessor passes. This saved
MemLLM from spending the full data and Test budget on mechanisms that already
failed source-local acquisition.

### Predeclare held surfaces and stop rules

MemLLM froze Train/Eval/Test authority, read some held surfaces only once, and
prohibited seed/epoch/scale rescues after a binding result. This made negative
results interpretable and prevented adaptive overfitting.

EMBER should similarly predeclare task splits, seeds, interaction budgets,
thresholds, and the one allowed response to each failure class.

### Separate dense bootstrap signals from the final environment contract

Teacher actions, NLL, or shaped loss may be legitimate source-task cold-start
signals without becoming required deployment inputs. The final mechanism must
still qualify under the feedback actually available at deployment.

For EMBER, action-labeled robot trajectories may bootstrap the Writer, while
held deployment still provides only language/video followed by environment
interaction.

### Keep feedback out of the instantaneous Writer input unless explicitly modeled

Reward can train the Writer through the outer loop without being an input to a
static Writer. If the Writer must react to a particular task's failures, that is
a separate feedback-aware design with rollout history as a declared input.

### Treat missing feedback as unknown

An input with no observed reward is not automatically useless or negative.
Fabricating no-write labels from missing outcomes biases the Writer toward the
observed benchmark.

### Separate amortized training cost from deployment cost

Writer pretraining, teacher generation, and outer-loop RL may be expensive.
Report them separately from one-specification Writer latency, task-local RL
steps, and time to first useful policy.

### Compare deployment contracts fairly

Main baselines should share the backbone, data authority, task split,
interaction budget, trainable parameter count where possible, and evaluation
protocol. A method requiring different supervision should be labeled as an
oracle or related-work comparison rather than forced into the main table.

### Batch size and GPU count are execution choices

Data-parallel world size should not silently change adapter rank, number of
memory directions, or the scientific meaning of an episode. Resource scaling
and optimization changes must be explicit.

## 4. What EMBER should not inherit automatically

- Wiki/QA as the benchmark or task format;
- one shared fixed-capacity memory for a 200k stream;
- selective long-term consolidation and exact backbone merge;
- the historical CDB, B0, P/Q/S, rank-64, or layer choices;
- backward-free behavior for the later task-local RL phase;
- one specific teacher, PPO formulation, critic, or journal representation;
- assumptions that language-model factual memory and continuous robot control
  have the same optimal adapter geometry.

EMBER inherits the **research question and experimental discipline**, not the
old implementation.

## 5. Concrete implications for the EMBER plan

The remote expert should ensure the roadmap includes:

1. a task-specific useful-update oracle before training the Writer;
2. a representation gate showing the chosen LoRA target can express that
   oracle behavior;
3. a matched/shuffled/task-ID control gate for Writer acquisition;
4. behavioral rather than raw parameter supervision as the primary objective;
5. a separate test of immediate Writer gain and adaptation-geometry value;
6. predeclared held tasks and no post-result hyperparameter rescue;
7. a clear failure taxonomy: information insufficiency, representation,
   acquisition, optimization, credit assignment, or adaptation;
8. explicit separation of one-time meta-training compute and per-task
   deployment interaction.

These lessons should reduce repeated failure, not constrain the expert to the
MemLLM architecture.
