# EMBER Task Plan

## Goal

全面整理EMBER仓库并系统重建截至当前的项目认知，使代码、脚本、配置、文档与实验设计证据形成清晰、唯一、
可追溯的权威结构；在此基础上从宏观目标和具体历史结果逐步推导后续架构方向。整理阶段不直接实现新架构或启动
GPU实验。

## Done when

- active tree只有当前仍有owner的代码、脚本、配置和测试；退役路径由Git/formal artifacts保存。
- `AGENTS.md`只包含稳定项目总览和原则。
- owner目标与讨论共识、概念定义、历史架构账本和跨实验结论各有唯一canonical文档。
- 不存在互相矛盾的“active/next/resume”文档，也不存在持久handoff。
- 全部历史架构的假设、结果、实际否决边界和检索入口可从统一ledger找到。
- 明确临时文件被清理，formal runs、checkpoints、data、evidence和所有权不清内容完整保留。
- import/config/link扫描、compile、全量CPU测试和Git diff检查通过。
- 形成由证据约束的设计推理顺序，但不抢先宣布未验证的新架构。

## Constraints

- 不使用subagents。
- 不启动GPU训练或评测。
- 不删除formal evidence、唯一checkpoint、dataset或所有权不清内容。
- 不为清理添加兼容fallback、防御性hash或新框架。
- 不把memory token、rank或具体decoder写成goal；它们只是候选方法。
- 历史文档和artifact中的旧“下一步”不得恢复执行。

## Work plan

- [x] 完整阅读authority与mandatory reading，确认Git、进程、quota和当前权限。
- [x] 盘点tracked/untracked文件、引用关系、worktree、tmux、runtime资产和大目录。
- [x] 移除103个历史worktrees与死tmux sessions；确认五个scratch branches已patch-equivalent后删除，保留正式资产。
- [x] 删除退役Writer/v6-prior/reward-gate运行路径及其专属configs/tests。
- [x] 收缩task-expert和evaluator运行面，封闭terminal GOMQ训练入口。
- [x] 通过第一轮相关CPU测试（119 passed）。
- [x] 将`AGENTS.md`收缩为稳定总览；恢复`task_plan/findings/progress`三文件工作状态结构。
- [x] 重写README、owner requirements、concept、research history和findings。
- [x] 建立“重新认知与设计推理”文档，区分已解决接口、未解决接口和待检验问题。
- [x] 删除已合并进ledger的重复per-design docs、过期迁移文档和失效引用。
- [x] 清理明确临时cache、bytecode、pytest cache、egg-info和`.codex/tmp`。
- [x] 完成引用、config、import、compile、targeted/full CPU和Git最终验证。
- [x] 更新`progress.md`与最终仓库地图，形成clean commit并按仓库Git合同交付。

## Decision after cleanup

整理完成后先回答以下问题，再决定是否建立active design：

1. EMBER的输入信息中，哪些是language query、video dynamic Value和有向过程证据？
2. 历史上哪几个接口已经可靠接通，哪一个是最早仍未解决的接口？
3. same-task跨video共同表示与cross-task可分表示是否同时存在于现有artifact？
4. 失败主要发生在Program形成、native LoRA compiler、reward direction还是shared retention？
5. 下一实验怎样只改变一个主要因果变量，并以最小机制证据快速否决？
6. 何时必须做strict paired400、相邻checkpoint和六臂controls？

在这些问题没有从整理后的证据中逐步闭合前，不把MCPS、memory token、V6 tail或任何其它候选提升为active design。
