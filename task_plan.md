# EMBER task plan

更新时间：2026-08-24。

## 当前目标

把仓库整理成可由新session直接接手的唯一活动面；等待owner已发送的专家回复，保存其原意并据此冻结自然LIBERO-only的ECP
架构与推进合同。回复到达前不启动新架构、不训练GPU、不自行联系专家。

## 当前完成条件

- [x] 从`main`清除退役Writer、ECP Stage 1 v1--v24后继、MDCO/PECS、人工process和失败realizer执行路径；
- [x] evaluator收回到source SFT与task-expert静态adapter基础，等待新ECP Writer接入；
- [x] 删除过时配置、脚本、重复测试、41份旧Markdown和87份分散证据JSON；
- [x] 删除可重建的人工process datasets和对应运行产物，保留原始LIBERO与唯一formal资产；
- [x] 将owner要求、ECP概念、当前架构、历史结论和交接状态改写为少量canonical文档；
- [x] 运行完整CPU测试、引用审计、文件/行数审计和Git diff审查；
- [x] 提交并推送`main`，确认只有一个worktree且无task-owned branch；
- [ ] 收到专家回复后，保存裁决、更新设计与下面的执行阶段，再把工作交给新session。

## 专家回复处理

收到回复后的当前session只做四件事：

1. 保存专家原始回复或不失真的结构化裁决，不擅自扩写；
2. 对照`docs/event_conditioned_policy_compiler_design.md`标出确认、修改、否决与仍开放部分；
3. 把最终架构、数据、训练阶段、实验、通过条件和停止条件写成唯一执行合同；
4. 更新`progress.md`与`HANDOFF.md`、提交并推送，然后停止，让owner开启新session。

若专家回复仍存在真正改变架构的歧义，记录疑问并给owner一段精简可复制prompt；未经明确允许不直接发送。

## 回复后预期推进阶段

以下是待专家确认的骨架，不是当前launch授权。

### A. 自然LIBERO资产与映射审计

数据：train24、过滤后的non-held LIBERO-90、已有task experts、successful trajectories与action-hidden videos。

工作：明确独立任务mapping、重复语义、policy lineages、可用occupancy和video覆盖；不生成新任务或人工数据。

通过：无validation/test梯度、无重复任务泄漏、无task ID route，资产足够支持下一门。

### B. Program与shared realizer Gate

模型：专家确认的Program schema、privileged `q_pi`和一套跨任务共享的Program-to-LoRA realizer。

工作：使用授权meta tasks分阶段训练并冻结坐标；用train24 held5/多fold做唯一LoRA闭环评测。

通过：显著高于source/shared baseline、接近successful-member breadth、Goal/Long非零、相邻checkpoint稳定。失败则定位Program、
posterior或realizer，不训练`q_V`。

### C. Video posterior Gate

模型：frame observer、event binding、Dynamic-K aggregator和`q_V`；Program坐标与realizer先冻结。

工作：从exact language+action-hidden video预测与`q_pi`同构的Program，video/action query跨episode。

通过：full video闭环显著胜language/no-video、scene/first+final和wrong；same-task其它视频高retention；Goal/Long有贡献。

### D. 全Writer联合训练

冻结PI0.5 backbone，解冻专家允许的全部Writer参数，以小学习率联合优化并保留阶段checkpoint。Action Meta-LoRA必须完成一次
matched attempt；无负面且有净收益则启用并冻结。

### E. Structured outer credit与validation8

只有Stage C/D已有视频闭环增量，才用train/meta simulator reward训练共享outer objective。随后从fresh使用全部授权开发数据训练，
做single-checkpoint strict paired400、per-task/per-suite、breadth、retention/churn和相邻稳定性。

### F. 最终controls与Test

最终checkpoint冻结后才评测shuffled/reversed时序特异性；Test8只在方法选择完全结束后使用。

## 路线边界

- 不恢复人工process dataset/controller acquisition；
- 不把GOMQ当ECP Phase 0或要求重跑所有历史架构；
- 不恢复PECS、v24、fixed-A、rank12+rank4或raw-factor短solver的小变体；
- 不用loss、reconstruction或LoRA geometry代替closed-loop；
- 不在回复前自行确定realizer、posterior或训练顺序中仍开放的关键选择。
