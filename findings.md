# EMBER Durable Findings

## Research framing

- The broad information-to-parameter idea is meaningful only over a structured
  task family with executable source-task bridge supervision. It is not a
  universal optimizer claim.
- A static task-specification-to-adapter Writer is best described as an
  amortized task-conditioned parameter-update generator or hypernetwork.
- The candidate novelty lies in the complete controlled combination, not in
  language-conditioned weights, action-free video, LoRA, subspaces, or RL alone.

## Adopted scientific lessons

- Separate useful-update existence, representation, and amortized Writer
  acquisition into different gates.
- Functional behavior and closed-loop return take priority over raw parameter or
  LoRA-factor distance.
- Update diversity does not prove task specificity; average, retrieval,
  wrong-spec, scene-only, and shuffled-video controls are mandatory.
- Immediate gain, control harm, adaptation speed, final performance, and
  one-time meta-training cost must be reported separately.

## Adopted execution decisions

- Active compute cap: four A100 80GB GPUs, normally one or two for pilots.
- SmolVLA plus LIBERO is the primary development surface. OpenVLA-OFT is scale
  confirmation after lower-cost gates survive.
- Neutral-prompt parameter compilation is a co-primary mechanism test.
- A language-only HyPoGen/DISC-style parameter generator is a required strong
  baseline.
- The geometry receives a real training signal through a differentiable
  low-dimensional source support/query loop before reward-based refinement.
- The default adaptation representation is a canonical center and soft geometry
  with residual escape, not an inescapable hard subspace.

## Verified design risks

- A successful same-embodiment action-hidden robot video proves an
  information/supervision conversion mechanism, not lower data-collection cost
  or human-to-robot transfer.
- LIBERO task scenes, layouts, language templates, filenames, episode length,
  and normalization statistics can leak task identity.
- The proposed LIBERO-90 task split remains a hypothesis until a task-factor and
  initialization audit is generated from pinned files.
- A geometry emitted by the Writer is meaningless unless its training objective
  and matched unit/global-metric comparisons are explicit.
- Joint Writer/base optimization creates a moving parameter coordinate system;
  shared base adaptation is optional and comes only after a frozen-base result.

## Unknowns requiring evidence

- Whether language leaves measurable incremental information for video on the
  selected task subset.
- Whether the intended action-policy matrices admit useful, safe local updates.
- Whether task oracles share a compact canonical functional representation.
- Whether a predicted geometry transfers from offline support/query learning to
  sparse-reward local adaptation.
- Actual simulator throughput, GPU memory, storage footprint, and research
  iteration cost under the four-GPU ceiling.
