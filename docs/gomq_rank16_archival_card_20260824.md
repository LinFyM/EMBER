# GOMQ cycle2 有效 rank-16 Phase 0 复现卡

## 目的

本轮只回答一个问题：历史 GOMQ cycle2 的 `151/400` 是否能够在不重新训练 Writer、
不重新读取教学视频、也不保留伪 rank-32 表示的前提下，作为真实的完整 rank-16
single-checkpoint strict-400 基线复现。

它是 ECP 最终执行序列的 Phase 0 档案基线，不是新 ECP Program、`q_pi`、`q_V` 或
Writer 训练，也不决定后续架构是否成立。

## 冻结输入

- 历史结果根：
  `runs/outputs/pi05_v6_lpcp_cfmg_gomq_cycle2_k4_correct400_noreplacement_seed7_trainr6_evalr6_8553b61_gpu01p012457_b16_retry1_20260817`
- checkpoint：该结果已经使用的 GOMQ cycle2 Writer checkpoint；本轮不重新运行 Writer。
- episode-LoRA：上述结果根中已经封存的 400 个逐状态缓存。
- policy/source、validation8、每任务 states `0..49`、environment/policy RNG、K=4 correct
  video ordinal 与原 strict-400 完全相同。
- 规范化 authority：
  `docs/evidence/gomq_20260823/gomq_cycle2_rank16_archival_projection.json`
- 输出 LoRA authority：`configs/pi05_lora_v1.json` 的完整 38-target rank-16 合同。

不做新训练、不挑 episode、不换 checkpoint、不融合结果，也不运行 causal controls。
shuffled/reversed 等视频时序资格测试留给最终 ECP checkpoint；本轮不能产生新的视频因果结论。

## 唯一变换

历史每个 target 的公开状态为：

```text
A32 = [A0; A0]
B32 = [B0, deltaB]
```

本轮逐 target 生成：

```text
A16 = A0
B16 = B0 + deltaB
```

要求所有 38 个 target 的两个 `A0` 半块逐元素相等；任一 target 不满足就停止，不能近似
拟合或另找低秩分解。`B` 在原生存储 dtype 中相加：Action in/out 保持 F32，其余保持
BF16。正常 BF16 舍入被接受，不额外扩 dtype 追求逐元素一致。

转换后的每个 episode 仍只对应原来的一个 language、K=4 action-hidden videos 和一个
完整 38-target LoRA；不会把多个 LoRA 平均，也不会在 rollout 中重新读取视频。

## 正式评测与裁决

只运行一次 fresh strict paired-400 correct arm，并与历史 151-row success set 按
`suite/task/state` 配对。报告：

- overall、per-suite、per-task；
- breadth；
- retained/gained/lost、churn 与 success-set overlap；
- 400 行完整覆盖和 single-checkpoint 身份。

裁决预先固定为：

1. 新结果 `>=145/400`：把该结果登记为 ECP Phase 0 的绝对 rank-16 GOMQ 基线；
2. 新结果 `<145/400`：只承认代数上的有效 rank-16 结构，历史 `151/400` 保留为机制与
   历史证据，不再作为后续 ECP 的绝对闭环基线；
3. 无论得分如何，都不根据本轮结果调 rank、scale、seed、dtype 或 checkpoint 后重跑。

Phase 0 结束后，下一项是 process minimal-pair Gate，而不是继续迭代 GOMQ。
