# Novelty and Baseline Landscape

## EMBER 的问题

直接 action-SFT 很强，但依赖稀缺、embodiment-specific 的机器人 action trajectories。EMBER 研究的是 action-hidden teaching video 是否能先把策略初始化到一个更有用的位置；reward practice 是可选的第二阶段，而非方法定义。

当前最清楚的证据链是：

1. generic π0.5 在 held task 不看视频；
2. EMBER 看相同 held task 的一条 action-hidden teacher video，生成完整 LoRA；
3. ViVLA-style 方法看同一条视频并直接 condition action policy；
4. 若做 reward adaptation，identity-init 与 Writer-init 使用相同 interactions。

## ViVLA 是最直接的 matched baseline

ViVLA 在 source 上学习如何用 expert video condition 当前 robot policy，在 held LIBERO task 上只看视频、不读取 held actions，也不做 target action-SFT。这与 EMBER 的信息条件可以直接匹配；是否每个 policy step 重看视频不是不公平，因为那是方法差异。

公平比较固定：相同 π0.5 base、24/32 source tasks、one-video sampling、held task IDs、current observation、target action wall 和 rollout evaluator。ViVLA 输出在线 conditioned actions；EMBER 一次把视频编译为可复用 LoRA。报告 success、video preprocessing time、policy latency、memory，并在需要时给相同 reward budget。

## Direct LoRA oracle

target-action-supervised LoRA 是“教练拉着手”的 privileged upper bound。它回答相同 LoRA 空间是否能学会任务，但不是同信息墙 baseline。论文应明确展示性能差距，而不是要求 EMBER 击败 action oracle。

## 其他 baseline

- generic/frozen π0.5：没有 target video 的下界，也是当前 feasibility test。
- `Source-SFT π0.5`：在最终 32 source tasks 上按 AS-Writer 相同 optimizer-step budget 做 action-SFT，test 不看 held video；它控制 source-side training，并检验 EMBER 额外读取 held video 的价值。
- language-only parameter generator：检验视频是否提供语言之外的信息；最终可选成熟 HyPoGen/DISC-style 方法，不需要重复多个近同构 arm。
- retrieval/average source LoRA：检验 Writer 是否只是 nearest-task selection。
- matched ordinary LoRA RL：检验 video initialization 是否提高 reward efficiency。

R+X、SeeTraceAct、RAD、RoboCasa 等工作继续作为场景与相关工作参考，但当前 benchmark 决策已经收敛到与 ViVLA 同口径的 LIBERO-40，不在本轮扩散到 sim-to-real 或真实机器人。

## 当前不允许的结论

- generic π0.5 base 的 zero-shot 成败不是 EMBER 的最终成败；它只决定后续是否需要 source-base calibration。
- 旧 SmolVLA 70/10/10 结果不能当新协议结果。
- direct LoRA 强不能证明视频无用，因为其使用了额外 target action labels。
- 不把 task-local RL 写成 EMBER 的必要组成，也不声称 outer learning 已验证。
