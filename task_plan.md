# EMBER Task Plan

最后更新：2026-07-20。状态词只描述当前新 70/10/10 主线；旧 60/15/15 实验不计为这些阶段完成。

本文后续的 frozen source embodiment base 统一且只指：通用预训练 `lerobot/smolvla_base` → 在 70 个 train tasks、每任务全部 50 条成功 teacher episodes 上联合训练 → 得到一个共享、多任务、语言条件的 source embodiment base → 训练完成后冻结。EMBER、target-action-supervised direct LoRA oracle 和 ordinary task-local LoRA RL 都以它为共同起点；它只能按 train/source evidence 选择。

## 完成定义

长期工作只有在以下链条全部完成后才可标记 Goal complete：

1. specification-only 同分布 70/10/10 split 与新 manifest 永久封存；
2. 70×50 shared source embodiment base 可复现训练、按 train/source evidence 选择并冻结；
3. Writer cold start 在多个不同类别 validation tasks 上明显超过 frozen source embodiment base；
4. Writer-only RL 保存独立产物并报告相对 cold-start 的 validation 原始结果；
5. Writer-init task-local LoRA RL 与 matched zero-init ordinary LoRA RL 在相同 interaction/update 预算下完成；
6. 全部方法、checkpoint、预算、selection rule 和 baseline 冻结后，补齐必要独立 seeds、尚缺 matched baselines，并统一完成 reporting-only test；
7. shared-frozen held/test 合同成立。

完全符合最终合同的开发 trajectory 可直接作为正式 seed 1；只重训受 split、source base、architecture、LoRA、loss、data、optimizer、RL 或 evaluator 变化影响的 arms。source-only reward/meta outer learning 只可在 Phase F 完成后作为可选增强，不是完成条件；当前 Goal 不包含 OpenVLA-OFT 或 source direct-LoRA localization。

代码完成、环境完成、一个 checkpoint、Gate -1、Gate 0、一次 source 结果或“可以进入 Writer”都不能完成长期 Goal。

## Phase A：永久封存 70/10/10

状态：已完成；canonical seal 为 `configs/libero90_70_10_10/`。

动作：

- [x] 从 pinned LIBERO-90 task language/scene 重建 90-row role-aware factor table。
- [x] 明确 verb、moved object、target receptacle、target relation、order/composition 和 specification difficulty。
- [x] 未知/歧义 instruction fail closed。
- [x] 在只读 specification surface 上确定 deterministic algorithm、seed 和 objective。
- [x] 搜索并冻结 70 train / 10 validation / 10 test。
- [x] 验证 task IDs 互斥且覆盖 0..89。
- [x] 验证每个 val/test task-relevant exact role atom 在 train 至少有 2 个支持，并封存同 family top-neighbor audit。
- [x] 验证 val/test 各有 10 个不同 scene、相同 scene/category/difficulty 配额。
- [x] 在任何新协议 policy outcome 前封存 factor rules、IDs、algorithm/seed、manifest 和 SHA256。
- [x] 重新审计 HDF5、BDDL、init states、controller、camera、50 success demos/task。
- [x] normalization 只使用新 train tasks；validation/test numeric access 为零。

停止条件：存在多个实质改变论文解释、无法用 specification-only objective 排序的 split 时才询问 owner。不得用旧 policy performance 选 split。

## Phase B：Source embodiment base

状态：进行中；Phase A 已满足。

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
- [ ] 冻结后在 10 个 validation tasks 建立 frozen source embodiment base reference。
- [ ] 每个 validation task 用自己的全部 50 条 teacher action episodes 训练 matched direct LoRA oracle，并明确标注 action-supervised reference。
- [ ] Phase F 解封前不运行任何 test policy evaluation、不训练 test direct LoRA，也不读取 test actions/reward/success。

如果 30 分钟后 competence 明显仍在上升，可从同一 checkpoint 续一个有理由的短段；不得从头重跑。

## Phase C：Writer cold start

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
- [ ] 在 10 个跨类别 validation tasks 比 frozen source embodiment base、Writer、direct LoRA oracle。
- [ ] 保存原始 per-task successes、rollouts、seeds、steps、数据、GPU 和 wall-clock。

判断：跨多个类别明显超过 frozen source embodiment base 即可证明核心机制产生功能价值并进入下一阶段；低于 direct LoRA 是优化信号，不是一票否决。效果差时先看实现、数据规模和 source base，再从同一 checkpoint 加训；数据充分仍失败才改 architecture/loss。

## Phase D：Writer-only RL

状态：等待 cold start。

- [ ] 只用 70 train/source tasks。
- [ ] base 冻结；生成 LoRA 不原位更新；reward 只更新 Writer。
- [ ] 70-task balanced mixed rollout/update cycles。
- [ ] 总 wall-clock 目标 ≤约 90 分钟。
- [ ] checkpoints 在 1/3、2/3、3/3 full-task cycles。
- [ ] 每个候选在 validation 做 frozen-Writer zero-interaction h50 评估。
- [ ] 报告 cold-start 与 Writer-only-RL 的原始数字。
- [ ] 在 validation 选择一个最佳 Writer initialization 进入首轮 task-local RL。

## Phase E：Task-local LoRA RL

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

## Phase F：合同冻结与统一 reporting-only test

进入条件：Writer cold start 已在多个类别 validation tasks 上明显优于 frozen source embodiment base，Writer-only RL 和 matched task-local RL 已完成 validation 选择。

- [ ] 冻结 split、source base、Writer/architecture、LoRA、loss、data、optimizer、steps、RL algorithm、interaction/update budget、checkpoint selection rule、evaluator 和 baseline 集合。
- [ ] 保留从一开始就使用完整 70×50 数据且完全符合最终合同的开发 trajectory 作为正式 seed 1。
- [ ] 只重训因合同变化失去可比性的 arms，并补齐必要独立 training seeds、尚缺 matched baselines 和 validation rows。
- [ ] 冻结并完成 frozen source embodiment base、最佳 EMBER zero-interaction、zero/identity-init ordinary LoRA RL、best Writer-init matched LoRA RL，以及必要同信息墙强 baseline。
- [ ] 最后统一解封 test，运行上述方法和 target-action-supervised direct LoRA oracle；test direct LoRA 每 task 可用全部 50 条 teacher action episodes，但不进入信息匹配主结论。
- [ ] test task-local RL 只按 validation 已冻结的 reward budget/selection rule 适应，最终性能使用 fresh rollouts；test 结果不反向改方法。
- [ ] 统一 task/init/RNG/h50/precision/budget，报告 data、steps、interactions、GPU-hours、wall-clock 和原始 rows。

当前不做 standalone Language-only Writer、standalone Video-only Writer、source direct-LoRA localization、full/action-expert task-local RL upper bound 或 OpenVLA-OFT。

## Phase F 之后的可选增强：Source-only reward/meta outer learning

状态：只在 Phase F 完成后按价值决定，不阻塞 Phase F 或 Goal complete。

- [ ] 若执行，inner/source RL 只更新 task-local LoRA。
- [ ] outer source reward/meta objective 只更新 Writer；shared source embodiment base 始终冻结。
- [ ] 只用 source 训练、validation 选择，不影响已经冻结并报告的核心 Phase F 结果。
- [ ] 有收益时作为后续额外 arm；未实现或负结果不否定核心 EMBER。若未来要给该 arm 增加 test，必须另行冻结并声明为 Phase F 之后的扩展。

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
