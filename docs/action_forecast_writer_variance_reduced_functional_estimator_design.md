# Variance-Reduced Functional Estimator

## Status

2026-08-03 post-seal candidate. This changes the stochastic estimator used by
AS-Writer training, not the Writer architecture or its information wall.
Longest-105 B20 and fresh/exact-resume vertical paths pass at `50662a8`; the BCI
six-rank logical-B20/policy-microbatch2 path is now sealed from clean pushed
commit `391f183`. No formal 0→200 training or closed-loop evaluation has been
run.

## Question

Semantic Factor Basis demonstrates learned task-conditioned routing and a
causal Core/Program-to-effective-LoRA/action path, yet its late training
gradients remain nearly orthogonal and its task-mean direction is only about
1.29% of per-video sample energy. The matched gradient audit attributes large
centered energy to flow time/noise and query-by-flow interaction. This candidate
tests whether reducible Monte Carlo variance, rather than missing task routing,
is a primary cause of checkpoint task rotation.

## Fixed contract

- Keep the exact Semantic Factor Basis Writer and 11,159,296 parameters.
- Keep one teacher video, one complete rank-16 LoRA, B20 independent same-task
  cross-episode action queries, all 24 tasks exactly once, task-equal raw mean,
  one clip/AdamW/scheduler update, and the sealed optimizer/LR schedule.
- Keep the source policy, splits, normalization, video stride, public 38-target
  LoRA topology, functional loss, and all information-wall restrictions.
- Do not add gates, scales, auxiliary losses, task weights, outcome labels,
  checkpoint fusion, or multi-video averaging.

## Estimator

Within each even B20 task batch:

1. Draw one randomized Latin stratum from the exact PI05 Beta(1.5, 1) flow-time
   distribution for each query, then randomly permute strata across queries.
2. Draw ten standard-normal flow-noise tensors, append their negatives, then
   randomly permute all twenty tensors across queries.
3. Key both random streams by the existing immutable task/query identity and
   restore global CPU/CUDA RNG state after the functional forward.

Every individual query retains the original time and Gaussian-noise marginal.
The task-batch mean remains an unbiased estimator of the existing objective;
only within-batch dependence changes. Antithetic cancellation is exact for the
noise mean, while nonlinear policy responses need not cancel.

## Why this is the minimum change

The current evidence says that task routing exists and video-dependent content
reaches policy action. Adding more routing capacity would not test the earliest
remaining failure interface. Increasing B would raise memory and query cost.
Latin stratification and antithetic pairs use the already approved B20 forward
and directly target two measured noise sources without changing the desired
objective or architecture.

## Risks and falsification

- Antithetic samples can be less effective if nonlinear flow gradients are not
  locally monotone; randomized assignment preserves marginal validity but does
  not guarantee lower variance.
- Time plus noise reduction is one deliberate estimator bundle, so a positive
  result does not separately attribute gains to either component.
- Exact zero noise mean must not be confused with a zero-variance gradient.

The hypothesis is weakened if matched diagnostics do not materially increase
task-mean/sample energy, successive same-task gradient cosine, or cross-task
coherence. If those improve but paired closed-loop performance and breadth do
not, the next root cause is objective-to-closed-loop manifold mismatch rather
than estimator variance. If neither improves, retain the SFB routing evidence
but reject this estimator and revisit the complete training/functional target.

## Execution gate

Run only the shortest real vertical path: targeted CPU tests, longest-105-frame
B20 fresh 0→1 and exact resume 1→3 on six BCI ranks, then fresh 0→200 with
checkpoints every 25. Evaluate paired correct400 at 50/100/150/200. Continue to
400 only if absolute performance, breadth, right-edge trend, or internal
gradient stabilization supplies positive evidence.

## Live vertical-path evidence (2026-08-03)

The retained implementation commit is `50662a8`. The one-line runtime mode
routing omission found by the first real launch was fixed with a focused
regression; 4 relevant tests pass. A fresh longest-105-frame run then completed
three full24 macros on four ranks at B20 in `60.83s`, with peak CUDA reserved
memory `83,508,592,640` bytes, zero clipping/non-finite values, all five main
blocks reachable after the identity lifecycle, and no validation/test action
reads. Root:

```text
/data/ymdai/outputs/ember/pi05_as_writer_semfactor_vr_postseal_long105_profile_r4_b20_seed172_50662a8_20260803
```

The formal seed separately completed fresh `0→1` then exact resume `1→3`, with
continuous macro/checkpoint/cursor state and contract
`5111fa16b2b1db875ae80d79113516bfb8c853f76508b16979f4c7f9de558921`:

```text
/data/ymdai/outputs/ember/pi05_as_writer_semfactor_vr_postseal_formalseed_resume_r4_b20_seed7_50662a8_20260803
```

Against the ordinary SFB formal-seed vertical path with identical task/video
assignments, mean raw-full24 gradient-energy retention changed
`.11346→.13255`; same-task successive CountSketch cosine changed
`.26439→.29206` (factor-only `.27354→.29758`). This is small, directionally
positive mechanism evidence, not proof of improved drift or closed-loop
performance. The next valid experiment is fresh `0→200` followed by paired
correct400 at `50/100/150/200`.

## BCI 46GB adaptation evidence (2026-08-03)

The BCI port keeps each task's logical B20 estimator intact. One generated LoRA
is reused while the frozen-policy functional forward is sliced into ten B2
microbatches. Each slice regenerates and selects its exact portion of the keyed
full-B20 Latin-time and antithetic-noise draws; losses and LoRA-leaf gradients
are sample-weighted back to the same task mean. Gradient accumulation is FP32
for BF16/FP16 leaves and casts once at the bridge boundary. Six ranks own four
tasks each, so the global update remains the exact equal-weight full24 raw mean
with one clip, AdamW, and scheduler update.

The provisional engineering root
`runs/acceptance/ember_bci_vr_effective_b20_micro2_r6_profile_20260803T1600/train`
completed fresh `0→1` and exact resume `1→3`. All three macros covered 24 tasks,
480 logical queries and 240 physical forwards. Step times were
`33.973/31.686/31.240s`; peak CUDA allocated/reserved was
`34,970,270,208/47,108,325,376` bytes. Main-path gradients were finite/nonzero
from macro2 and no validation/test actions were read. Because the source was
dirty, the result is only a sizing/mechanics pass; the identical vertical path
must be replayed from a clean pushed commit before formal launch.

That replay passed from clean pushed `391f183` at
`runs/acceptance/ember_bci_vr_effective_b20_micro2_r6_profile_391f183_20260803T0735Z/train`.
Fresh `0→1` and exact resume `1→3` retained contract
`31ea4bc9a65ca0805ea3a49e7a33d07b40fe84e680df26607512d9e248455de0`;
step times were `33.514/32.050/31.326s`, peak allocated/reserved was
`34,970,270,720/47,108,325,376` bytes, and step3 sealed 1,440 queries/72 videos
with all five main blocks reachable after the identity lifecycle and zero
validation/test action reads. One first resume attempt stalled before its
invocation record; a same-six-rank object-collective probe and the unchanged
resume command both subsequently passed. This remains an unattributed transient
runtime observation, not a verified code defect; retained live monitoring is
required for formal startup and any later resume.
