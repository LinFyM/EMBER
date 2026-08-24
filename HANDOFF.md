# EMBER temporary handoff

此文件只服务本次跨session交接。新session完整消费后删除并提交，不把它保留为长期状态文档。

## 1. 交接结论

- canonical branch为`main`；科学起点`7ab5a04`，瘦身基础`6fdaeb8`，最终文档提交见Git HEAD；
- 没有新的ECP训练或GPU job，没有联系或回复外部专家；
- 专家最新回复已完整处理，active design是**ECP Native-Factor Compiler**；
- 原回复共1416行，没有原样复制进仓库以避免再次制造重复长文档；所有可执行架构、shape、数据、loss、Gate、耗时、final controls和
  停止条件已无损结构化进`docs/event_conditioned_policy_compiler_design.md`，长期结论进入`findings.md`与`research_history.md`；
- 专家审查的`7ab5a04`早于`6fdaeb8`瘦身，但期间没有新增科学结果；当前代码也复核了专家指出的native hook缺口。

## 2. 路线变化，必须先读

旧交接中的`q_pi -> shared realizer`已经被专家明确否决，不能恢复：

```text
exact language + K ordered action-hidden videos
  -> Pass A owner-specific Video Program
  -> Pass B Program-conditioned pooling of target-native inputs/outputs
  -> mobile rank4 residual + frozen rank12 carrier
  -> one complete 38-target rank16 LoRA
```

privileged experts/effects只作nonparametric set-valued functional critic，不输出Program、不进入deployment。唯一Program schema为
`P_lang[38,128]`、`P_scene[38,128]`、`P_process[8,38,128]`、`rho[8]`、`tau[8,2]`、`sigma[8,38,128]`。

## 3. 新session的唯一当前任务

不要训练fresh Stage 0、shared compiler、joint Writer或outer credit。先完成G1 native-factor task-local capacity oracle：

1. 实现38-target真实linear input/output hooks；
2. 按frame chunk构造absolute/adjacent/init/goal output banks；
3. 实现Program-conditioned signed rank4 pooling、target scales与small-core SVD；
4. 在fold0 held5只优化task-local rank/event/pooling/scale free code；
5. 完成carrier/direct/mobile/native四arm strict250。

通过门：relative recovery至少0.70（当前参考约90/250）、breadth5/5、Goal/Long均非零、4/5 tasks高于carrier、carrier保留至少33/43，
且只有一套完整rank16 adapter。一次完整周期预计约1个工作日。失败只允许一次read-only span分析和一次实现修正，再失败即停止该
架构，不把失败推给数据量。

G1通过后的G2--G5与final顺序、训练数据、冻结关系、门槛和耗时全部以`task_plan.md`和active design为准。

## 4. 当前代码与资产边界

保留：source/corpus/SFT、LoRA、task experts、Stage 0 v3、transition/event modules、policy effects、functional loss、reward occupancy、
dynamic evaluator、successful members、carrier/mobile-rank4 evidence。

尚无：native target bank hooks、online accumulator、signed compiler、free-code optimizer和G1 evaluation wiring。必须在一个canonical
implementation surface内新增；不得恢复旧Writer、ECP v24、MDCO/PECS、fixed/two-sided realizer、人工process或GOMQ路径。

## 5. owner稳定要求

- 不制作人工dataset/task/controller trajectory，只用现成授权LIBERO；
- 效率优先但不丢专家关键意见；先判断最早接口，不盲目版本迭代或小超参扫；
- 复用现有长训练资产，smoke只做最小必要验证，不做checksum、防御性矩阵扫描或冗余测试提交；
- 单job一个节点最多6张A40；EMBER通常总量不超过6，大量空闲才最多8；launch前live检查gpu01/gpu02；
- 可安全共驻低显存/低util卡，不kill/reset他人进程；gpu01物理0若仍prohibited则不用；
- 关键Gate汇报即可；遇到困难先回看active design和专家裁决；
- canonical是`main`，仅在隔离/并发必要时开`codex/*` worktree，及时合并、推送、清理；
- 活动树只保留一个实现面，代码/配置/文档/runs同步清理；
- 未经owner当次明确允许，不得向外部专家发送任何内容；若需要，只给owner精简可复制prompt；
- 只有owner明确要求时才使用goal机制；routine实现决定自主解决，不反问owner；
- owner问具体问题时直接回答，不擅自扩成新方案或审批请求。

## 6. 唯一保留的后期政策差异

Action Meta当前默认关闭；base Writer有明确闭环增量后必须做一次matched attempt。专家要求“明确净收益且无breadth/retention
损害”才启用，owner此前要求“只要无负面效果就启用”。这不阻塞G1--G3；执行到该门时按owner最新指示，不要提前询问。

## 7. 新session启动清单

1. 按`AGENTS.md`完整读mandatory docs和本文件；
2. 核对当前HEAD、origin/main、worktree、ignored formal assets与GPU live state；
3. 删除本文件并提交；
4. 从G1实现与最小smoke开始；
5. formal launch前遵守clean pushed commit、frozen worktree、storage quota与GPU preflight合同。
