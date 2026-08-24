# EMBER temporary handoff

本文件只是本次跨session的临时入口；没有任何长期信息只存在于这里。新session完成下面清单后删除并提交。

## 1. 持久authority

- 稳定目标、数据/GPU/Git/沟通约束：`docs/current_owner_requirements.md`与`AGENTS.md`；
- 专家2026-08-24回复原文：`docs/expert_review_20260824_native_factor.md`；
- 当前架构、shape、rank证据分支、全部Gate与停止条件：`docs/event_conditioned_policy_compiler_design.md`；
- 当前执行计划：`task_plan.md`；跨轮结论：`findings.md`；即时状态：`progress.md`；历史证据：`docs/research_history.md`。

若本文件与上述持久文件不同，以owner最新表达和持久authority为准；不得从临时handoff恢复旧路线。

## 2. 当前交接点

- canonical为clean pushed `main`，最终提交以Git HEAD/origin为准；
- 本session没有启动GPU训练或实现新Writer，也没有联系外部专家；
- active路线为ECP Native-Factor Compiler；neural `q_pi`、fixed-effect/fit-span realizer和人工process路线已关闭；
- Action Meta只在base Writer已有增量后做matched control，只有明确净收益且无breadth/retention损害才加入；
- rank12 carrier + mobile rank4是首版有证据配置，不是永久硬约束；rank重开条件见active design。

## 3. 新session动作

1. 按`AGENTS.md`完整读取mandatory docs、专家原文和本文件；
2. 核对HEAD、origin/main、worktree、formal assets、storage与GPU live state；
3. 删除本文件并提交；
4. 从`task_plan.md`的G1 native-factor task-local capacity oracle开始，不先训练fresh Program/compiler/joint Writer；
5. formal launch前遵守clean pushed commit与frozen worktree合同。
