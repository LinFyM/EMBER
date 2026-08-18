# EMBER Progress

更新时间：2026-08-18。本文只记录当前可执行状态；稳定目标见`docs/current_owner_requirements.md`，持久认知见
`findings.md`，完整历史见`docs/research_history.md`。

## Current authority and scope

- 当前远程仓库独立复核整理已完成；本次没有使用subagents或启动GPU工作；
- canonical workspace为`/data1/user/ymdai/projects/EMBER`，主写分支为`codex/bci-continuation`；
- 当前没有active scientific design、训练、评测、resume或successor；
- 当前最新被测架构是EMBER-LMMPC Core-Addressed Reader，设计已封存且formal non-pass；
- owner明确要求后续只能基于当前架构推进，不得直接恢复V6/LPCP/GOMQ、旧carrier、旧compiler或双Writer路径；
- 本次远程整理不写入任何下一步科学方案，外部专家应独立判断未决归因。
- 固定提交`947c0e3`的外部独立复核已经收到并登记；其建议仍是advisory evidence，不自动成为active design或launch
  authority；
- 仓库侧已静态确认当前formal路径把fresh `patch_grounding`、逐帧`language_projection`和
  `interaction_projection`的输出再次detach，现有“全动态路径梯度”测试没有覆盖这些模块；尚未进行干预实验，
  因而不能把它直接升级成123低上限或checkpoint漂移的唯一根因；
- owner最新明确：后续canonical Writer移除Text Meta-LoRA；历史formal config的rank4事实保留，Action Meta-LoRA
  不随该决定自动删除。
- 已建立外部复核逐项审计计划：先做claim ledger、raw evidence重建和current checkpoint零训练诊断，再按门控分别
  裁决Text Meta-LoRA、fresh前端detach、occupancy、decoder和shared-gradient hypotheses；当前尚未启动这些步骤。

## Sealed scientific state

- 同一fresh world6/topology run的macro25/50/75/100 K4 strict paired400为
  `123 -> 84 -> 89 -> 87`，breadth为`8 -> 5 -> 6 -> 4`；
- 400个固定rows只有49行四点始终成功、150行曾成功。macro25到50丢失52行，到macro100仅恢复15行；
- 固定K4+B20 loss为`.112124 -> .099353 -> .098427 -> .101337`，loss改善没有产生held共同积累；
- 当前best123相对同schedule LPCP143为`100 retained / 23 gained / 43 lost`，相对GOMQ151为`100/23/51`；
- 当前相对151的28分缺口主要来自Long`-23`和Object`-12`，并由Spatial`+3`、Goal`+4`部分抵消；
- Core-addressed reader相对matched旧reader使macro25从104提高到123，是保留的正机制；
- 当前recipe同时存在最佳点absolute不足和相邻checkpoint漂移；没有完成视频六臂因果资格。

## Remote-visible review map

- 第一性原理与稳定目标：`docs/concept.md`、`docs/current_owner_requirements.md`；
- 当前架构的完整封存事实：`docs/layer_matched_memory_program_compiler_design.md`；
- 不含推荐方案的当前证据综述：`docs/architecture_reasoning.md`；
- 外部复核原意、实验建议和仓库侧核验：`docs/external_review_20260818.md`；
- 全历史架构与实验ledger：`docs/research_history.md`；
- 跨实验耐久结论：`findings.md`；
- canonical代码：`src/ember/writer/`；
- 正式冻结config：`configs/pi05_writer_layer_matched_memory_program_compiler_v5.json`。

大型`runs/`、checkpoint、raw rows和rollout media由`.gitignore`排除，外部reviewer无法从远程读取。影响当前判断的
aggregate、paired transition、per-task/per-suite和stage数据均已重述到上述tracked文档。formal config顶层
`active_formal_ready`是启动时冻结字段，不代表当前仍有active run。

## Local-only canonical evidence

- train：`runs/outputs/pi05_lmmpc_v5_formal_fresh_r6_b20_aecbce5_gpu01p124567_20260818`；
- four-checkpoint trajectory：`runs/analysis/lmmpc_four_checkpoint_strict_trajectory_20260818.json`；
- Program x FactorHeads cross-decode：
  `runs/analysis/lmmpc_program_factorheads_cross_decode_macro25_50_75_100_20260818.json`；
- combined diagnosis：`runs/analysis/lmmpc_macro25_50_75_100_drift_diagnosis_20260818.json`。

这些unique formal artifacts保留，不因仓库整理删除；远程文档不把本地路径假装成外部可访问链接。

## Repository cleanup status

- tracked代码审计已确认一个canonical Writer launcher；
- 无调用者的旧slot-set、旧reward实验模块和旧reader类已从active tree删除，Git历史保留；
- 过期active design语义和未经实施的on-policy候选已从当前authority移除；
- 18个clean detached worktrees已移除，只保留canonical workspace；临时Codex、pytest和Python cache已清理；
- 当前测试集`287 passed`；canonical Writer编译/导入、20份JSON config、9个本地Markdown引用、Git diff和
  authority禁用候选检查均已通过；
- review snapshot已经commit并同步到`origin/codex/bci-continuation`。

当前不允许从本文推导自动launch；仓库不登记preferred successor。
