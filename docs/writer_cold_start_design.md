# Writer cold-start development contract

This page freezes the first direct-Writer experiment before implementation or
outcomes. The executable authority is `configs/writer_cold_start.toml`.

- Input: task language plus three temporally ordered third-person frames from
  each action-hidden specification episode. Frozen SmolVLM features summarize
  language and video; no action, proprioception, reward, terminal, task-ID, or
  filename is a Writer feature.
- Output: every A/B factor in the fixed 37-target, rank-32, alpha-16,
  dropout-0 LoRA (1,485,312 task-local parameters). A layer/module/rank-aware
  decoder uses shared width-typed heads. It emits no bank, basis, geometry,
  mask, metric, radius, learning rate, or later-RL constraint.
- Loss: apply the generated LoRA functionally to the frozen source-base policy
  and backpropagate independent query flow-matching loss into Writer only.
  Raw factor MSE is not a training objective.
- Data: all 60 sealed source tasks; episodes 0--7 provide action-hidden video,
  episodes 8--39 provide functional training queries, and episodes 40--45 are
  reserved for source-side selection diagnostics. Validation uses sealed task
  IDs 11, 21, 51, 70, and 86, spanning five task categories. Test/held remains
  unopened.
- Freeze/update: cold start freezes base and feature encoder and updates Writer;
  Writer-only RL later keeps the same freeze; task-local RL freezes base/Writer
  and updates only the emitted LoRA; source outer learning updates Writer; held
  evaluation freezes every shared object.
- First segment: eight-rank DDP, per-rank batch 256/global batch 2,048, up to
  1,000 steps with atomic
  exact-resume points every 250 steps. Mechanical smoke checks shape, finite
  values, gradient flow, freeze identity, data isolation, and resume only.
- Stage validation: 64 independent rollouts per task/arm across eight policy RNG
  seeds, unified horizon/evaluator/precision, comparing frozen base, Writer, and
  the exact-capacity direct task-local LoRA upper bound. Raw stage results are
  retained even when unfavorable.
