# EMBER

EMBER研究如何把目标task的exact language和一条或多条action-hidden教学视频，在rollout前一次性编译成冻结PI0.5 Action
Expert的一套完整LoRA，使policy零交互完成任务。

当前方法为ECP Native-Factor Compiler。活动树只保留source/task-expert基础设施、严格评测、ECP Stage 0以及最终Writer会复用的
数据/functional组件。历史Writer、neural q_pi、ECP Stage 1 v1--v24、MDCO/PECS、人工process任务和fixed-effect/fit-span
realizer均由Git保存，不再作为可执行fallback。

## 当前状态

- source、固定24/8/8 split、task experts、rank16 LoRA与评测合同已建立；
- task-local rank16 oracle为250/400，证明Action Expert LoRA有闭环容量；
- ECP Stage 0 native v3可复用，但其q/v owners还不是真实target input/output captures；
- 专家已冻结owner-specific Program与native-factor signed compiler；rank12+rank4是首版有证据配置，保留capacity证据触发的重开分支；
- 旧privileged q_pi/realizer和人工process路线已正式关闭；
- 当前唯一下一步是fold0 held5 native-factor task-local free-code capacity oracle。

本session只完成交接，不启动训练；新session先按`task_plan.md`读通目标、历史、专家原文、代码/数据/脚本与formal assets，再从
capacity oracle开始。临时交接状态见`HANDOFF.md`。

## 阅读顺序

1. `AGENTS.md`：仓库总合同；
2. `docs/current_owner_requirements.md`：owner稳定目标与约束；
3. `task_plan.md`、`findings.md`、`progress.md`：当前计划、结论和进度；
4. `docs/concept.md`：科学问题与ECP假设；
5. `docs/expert_review_20260824_native_factor.md`：2026-08-24专家回复原文；
6. `docs/event_conditioned_policy_compiler_design.md`：基于原文和owner裁决的当前唯一架构、阶段Gate与停止条件；
7. `docs/research_history.md`：影响当前决策的历史证据。

## 目录

```text
configs/                 固定split、source、LoRA、task-expert和Stage 0合同
src/ember/ecp/           ECP Stage 0候选表示
src/ember/expert_manifold/ task expert训练与静态评测
src/ember/pi05_eval/     动态队列、恢复、聚合和评测合同
src/ember/reward/        训练期privileged rollout/occupancy工具
src/ember/source_sft/    source SFT训练、checkpoint与validation
src/ember/writer/        跨路线复用的数据、functional与Meta-LoRA工具
scripts/                 canonical训练、封存与评测入口
tests/                   当前活动面的定向测试
```

`data/`、`models/`、`runs/`和`.venv/`是ignored本地资产，不提交远端。现成LIBERO数据、tokenizer、唯一formal checkpoints和结果
应复用；人工process数据已删除。

## 本地验证

仓库使用Python 3.12和本地`.venv`：

```bash
PYTHONPATH=src .venv/bin/python -m compileall -q src scripts tests
PYTHONPATH=src .venv/bin/python -m pytest -q
```

主要入口：

```text
scripts/train_source_sft.py
scripts/train_task_experts.py
scripts/train_ecp_stage0.py
scripts/train_ecp_stage0_action_meta.py
scripts/evaluate_source_sft_validation_loss.py
scripts/evaluate_ecp_stage0.py
scripts/evaluate_pi05.py
```

正式GPU运行前必须读取项目合同，检查两个GPU节点与对应storage quota，并从clean pushed commit的frozen worktree启动。
