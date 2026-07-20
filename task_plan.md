# EMBER Task Plan

最后更新：2026-07-20。状态词只描述当前新 70/10/10 主线；旧 60/15/15 实验不计为这些阶段完成。

## 完成定义

长期工作只有在以下链条全部完成后才可标记 Goal complete：

1. specification-only 同分布 70/10/10 split 与新 manifest 永久封存；
2. 70×50 source embodiment base 可复现训练并具备基本 source competence；
3. Writer cold start 在多个不同类别 validation tasks 上明显超过 frozen base；
4. Writer-only RL 保存独立产物并报告相对 cold-start 的 validation 原始结果；
5. Writer-init task-local LoRA RL 与 matched zero-init ordinary LoRA RL 在相同 interaction/update 预算下完成；
6. source-only outer learning 保存独立产物并报告 outer 前后 zero-step 与 adaptation curve；
7. 方案冻结后，所有方法从规定初态、全部 70×50 数据重训，完成完整 validation/test、seeds、强 baselines 和必要消融；
8. shared-frozen held/test 合同成立；
9. surviving mechanism 在 OpenVLA-OFT 上完成规模确认。

代码完成、环境完成、一个 checkpoint、Gate -1、Gate 0、一次 source 结果或“可以进入 Writer”都不能完成长期 Goal。

## Phase A：永久封存 70/10/10

状态：下一步，未开始。

动作：

- [ ] 从 pinned LIBERO-90 task language/scene 重建 90-row role-aware factor table。
- [ ] 明确 verb、moved object、target receptacle、target relation、order/composition 和 specification difficulty。
- [ ] 未知/歧义 instruction fail closed。
- [ ] 在只读 specification surface 上确定 deterministic algorithm、seed 和 objective。
- [ ] 搜索或安排 70 train / 10 validation / 10 test。
- [ ] 验证 task IDs 互斥且覆盖 0..89。
- [ ] 验证每个 val/test task-relevant role 在 train 有多个近邻。
- [ ] 验证三组 scene/category/difficulty 大致同分布，val/test 各含多个类别。
- [ ] 在任何新协议 policy outcome 前封存 factor rules、IDs、algorithm/seed、manifest 和 SHA256。
- [ ] 重新审计 HDF5、BDDL、init states、controller、camera、50 success demos/task。
- [ ] normalization 只使用新 train tasks；validation/test numeric access 为零。

停止条件：存在多个实质改变论文解释、无法用 specification-only objective 排序的 split 时才询问 owner。不得用旧 policy performance 选 split。

## Phase B：Source embodiment base

状态：等待 Phase A。

合同：

- 起点：`lerobot/smolvla_base` 固定 revision。
- 数据：全部 70×50 teacher episodes。
- 训练对象：官方成熟 SmolVLA action expert + 必要 projections；一个共享多任务 base。
- 混合抽样：70-task deterministic no-replacement cycles。
- 资源：8 GPU DDP，一卡一 rank；每卡约 10GB headroom。
- 初始 wall-clock：约 30 分钟。
- checkpoint：估算总 steps 后在约 1/3、2/3、3/3，且每个边界完整覆盖 70 tasks。
- 选择：只用 train/source development，不看 validation/test。

动作：

- [ ] 核验官方 recipe、trainable names、normalization 和 exact loss。
- [ ] 写唯一 canonical config/runner，不恢复旧 Gate0 runner。
- [ ] 1–5 分钟吞吐/显存 smoke，只读 mechanics。
- [ ] 固定 global batch 与联动学习率。
- [ ] 运行约 30 分钟 exact-resume trajectory。
- [ ] 第一个完整阶段结果前确认 3500 条 episode 均贡献训练信号，并记录完成的 corpus epochs/consumed chunks。
- [ ] 标准 LIBERO h50 闭环测 train/source development。
- [ ] 冻结 source base、完整训练状态、hash 和 compute ledger。

如果 30 分钟后 competence 明显仍在上升，可从同一 checkpoint 续一个有理由的短段；不得从头重跑。

## Phase C：基础参考

状态：等待 source base。

- [ ] Frozen source base：10 validation tasks，标准 init states，h50。
- [ ] Frozen source base：10 test tasks，作为提前固定的 reference；结果不得用于模型/方法决策。
- [ ] Direct LoRA oracle：每个 validation/test task 用全部 50 teacher episodes、统一 per-task steps 和完整 LoRA contract。
- [ ] 报告 direct LoRA 是 target-action-supervised oracle/reference，不是信息匹配 baseline。
- [ ] 若最终正式阶段重训 source base/direct LoRA，统一重测，不复用开发 reference 冒充正式结果。

每个任务先用 50 个标准 init states；若独立 policy RNG 方差需要，再扩为 100 episodes/task。

## Phase D：Writer cold start

状态：架构内核保留，训练合同需适配新 split。

固定结构：

- language + 任意数量/长度 action-hidden videos；
- 冻结 VLM features 可缓存；
- trainable temporal/episode/set attention、fusion、task memory、layer-aware LoRA decoder；
- 输出完整 37-target rank-32/alpha-16/dropout-0 LoRA。

动作：

- [ ] 将 `writer/model.py`、`temporal.py`、`data.py` 接入新 source base 与新 manifest。
- [ ] 构建每 task 全部 50 条完整视频的 frozen feature cache，保留 episode boundaries。
- [ ] 70-task mixed/no-replacement sampler。
- [ ] 70-task cycle 之上再保证每 task 50 episodes 的最小覆盖；记录 consumed episode/chunk identity。
- [ ] functional action/flow loss；action 不进入 Writer。
- [ ] 只做 shape/OOM/NaN/gradient/freeze/resume/leakage smoke。
- [ ] 8 GPU 真实训练，初始总预算 ≤约 90 分钟。
- [ ] checkpoints 位于估算总 steps 的 1/3、2/3、3/3，每个边界覆盖 70 tasks。
- [ ] 在 10 个跨类别 validation tasks 比 frozen base、Writer、direct LoRA oracle。
- [ ] 在按类别预先固定的少量 train tasks 比 frozen base、Writer、direct LoRA，定位 source acquisition 与 validation generalization。
- [ ] 保存原始 per-task successes、rollouts、seeds、steps、数据、GPU 和 wall-clock。

判断：跨多个类别明显超过 frozen base即可进入下一阶段；低于 direct LoRA 是优化信号，不是一票否决。效果差时先看实现、数据规模和 source base，再从同一 checkpoint 加训；数据充分仍失败才改 architecture/loss。

## Phase E：Writer-only RL

状态：等待 cold start。

- [ ] 只用 70 train/source tasks。
- [ ] base 冻结；生成 LoRA 不原位更新；reward 只更新 Writer。
- [ ] 70-task balanced mixed rollout/update cycles。
- [ ] 总 wall-clock 目标 ≤约 90 分钟。
- [ ] checkpoints 在 1/3、2/3、3/3 full-task cycles。
- [ ] 每个候选在 validation 做 frozen-Writer zero-interaction h50 评估。
- [ ] 报告 cold-start 与 Writer-only-RL 的原始数字。
- [ ] 复用同一 frozen source diagnostic tasks 报告 base/Writer/direct 原始数字，不新增训练。
- [ ] 在 validation 选择一个最佳 Writer initialization 进入首轮 task-local RL。

## Phase F：Task-local LoRA RL

状态：等待 validation 选出的 Writer。

Arm A：

- source base；
- zero/identity LoRA；
- ordinary task-local LoRA RL。

Arm B：

- source base；
- best Writer-generated LoRA；
- 完全相同 ordinary task-local LoRA RL。

动作：

- [ ] 在 validation 选择成熟 RL algorithm、K interactions、U updates、optimizer 和 selection rule。
- [ ] 初始 profile 约 10–15 分钟/task/arm 等价预算，8 卡并行。
- [ ] 若不足，在 test 前统一提升到约 20 分钟/task；不得按 task outcome 临时改预算。
- [ ] checkpoints K/3、2K/3、K。
- [ ] 每 task 可按预算内 adaptation reward 选择 checkpoint。
- [ ] 用 fresh rollout 评估所选 checkpoint。
- [ ] 报告 J0、curve、AUC、time-to-threshold、JK、JK−J0、interactions、updates、wall-clock。

所有 target tasks 的总 wall-clock 是预算对象，不允许每 task 各跑 90 分钟。

## Phase G：Source-only outer learning

状态：等待 task-local RL。

- [ ] inner/source RL 更新 task-local LoRA。
- [ ] outer source reward/meta objective 更新 Writer。
- [ ] base 和额外 shared state 冻结。
- [ ] validation 比较 outer 前后 zero-step utility 和 matched adaptation curve。
- [ ] 保存独立 stage artifact 与 raw results。
- [ ] 首轮总 wall-clock 目标约 90 分钟，后续只在 validation 证据支持时 exact-resume。

不把 Writer-only RL 与 outer learning合并成一个阶段。

## Phase H：快速 baseline 与机制判断

快速核心：

- [ ] frozen base；
- [ ] EMBER zero-interaction；
- [ ] direct LoRA SFT oracle；
- [ ] zero-LoRA ordinary task-local RL；
- [ ] best Writer-init task-local RL。

当前不做：

- standalone Language-only Writer；
- standalone Video-only Writer；
- full/action-expert task-local RL upper bound；
- 完整论文 baseline 矩阵；
- OpenVLA-OFT。

## Phase I：最终正式实验

进入条件：完整 SmolVLA EMBER 在多类别 validation 上有明显 base 增益，并且 RL/outer 阶段已有可解释结果。

- [ ] 冻结 split、architecture、LoRA、loss、data、steps、RL、evaluator 和 baselines。
- [ ] 所有方法从规定初态使用全部 70×50 数据重训。
- [ ] EMBER 重走 cold start、Writer-only RL、task-local RL、outer learning。
- [ ] 多个训练 seeds。
- [ ] 完整 validation 选择。
- [ ] 完整 test/held 一次性报告。
- [ ] HyPoGen/DISC-style、ViVLA/DAML-style、direct conditioned、retrieval/average 和必要 reward-adaptation baselines。
- [ ] 统一 task/init/RNG/h50/precision/budget。
- [ ] 报告 data、steps、interactions、GPU-hours、wall-clock 和原始 rows。
- [ ] OpenVLA-OFT scale confirmation。

## 每次运行前的最小检查

- [ ] Git clean、正确 main revision。
- [ ] `nvidia-smi` 实时确认设备和所有者。
- [ ] 不超过 8 张 EMBER A100。
- [ ] 个人数据占用预计峰值低于 500GB。
- [ ] exact command/config/revisions/output root 唯一。
- [ ] 不覆盖已有 artifact。
- [ ] split/surface/visible fields 正确。
- [ ] checkpoint/resume stop condition 清楚。
- [ ] 视频/gallery 保留和清理策略清楚。

## 每次 meaningful 结果后

- [ ] 更新 `findings.md`：事实、指标、局限、解释边界。
- [ ] 更新 `progress.md`：命令入口、产物、当前 Git、下一动作。
- [ ] 更新本计划状态，不复制长日志。
- [ ] targeted verification。
- [ ] commit + push。
