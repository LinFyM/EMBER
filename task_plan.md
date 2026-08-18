# EMBER Task Plan

状态：2026-08-18 **completed，远程仓库独立复核快照已整理、验证并同步**。

## Goal

把EMBER当前代码、配置、项目认知、历史证据和最新实验裁决收敛成一个远程可见、内部一致、可由外部专家独立
复核的仓库状态。整理不提出或暗示下一套架构/训练方案，不恢复旧方法执行路径，也不启动训练或评测。

## Done when

- 稳定目标、科学合同、当前架构事实、历史结果和即时状态各自只有一个明确authority；
- 远程可见文档足以解释当前架构流水线、strict结果、直接配对差异、已排除解释和仍未区分的问题；
- 所有带方向性的“下一步”“首选候选”从当前authority中移除，历史语句不会被误认成执行授权；
- tracked代码只保留canonical入口和仍有调用者的实现，formal配置和历史证据保持不变；
- 全量CPU测试、文档引用、Git状态和remote同步均通过复核。

## Boundaries

- 不写下一步科学方案，避免给外部专家预设答案；
- 不直接回退V6/LPCP/GOMQ，不以旧Writer、旧compiler或旧LoRA作为当前执行基础；
- 当前被测方法始终是EMBER-LMMPC Core-Addressed Reader及其完整rank16 LoRA输出；
- 不删除formal run、checkpoint、raw rows、dataset、唯一资产或所有权不清文件；
- 不使用subagents，不启动GPU工作，不新增持久handoff；
- Git历史保存退役实现，active tree不保留无调用者的实验路径。

## Completed work

- [x] 完整核对owner requirements、AGENTS、三份状态文档、concept、research history和封存LMMPC设计；
- [x] 审计tracked源代码、脚本、配置、文档、测试、入口、ignored artifacts和detached worktrees；
- [x] 将`architecture_reasoning.md`改为不含方案推荐的独立复核证据综述；
- [x] 将LMMPC设计文档从过期active authority收敛为当前被测架构的封存事实；
- [x] 补充当前123与LPCP143、GOMQ151的严格配对差异，区分absolute缺口与checkpoint漂移；
- [x] 从当前authority删除on-policy replay等未经实施的下一步候选；
- [x] 删除无调用者的旧slot-set、reward实验模块和旧reader类，保留Git provenance；
- [x] 清理可恢复的clean detached worktrees和临时cache；
- [x] 完成commit与remote push；本地全量CPU验证、引用检查和diff审计均已通过。

## Current decision

当前没有active design、active experiment或预选successor。最新被测架构正式non-pass，完整结果和未决归因见
`docs/architecture_reasoning.md`；历史演进见`docs/research_history.md`。外部专家应在这些事实基础上独立判断，
仓库不提供推荐的下一步实现。
