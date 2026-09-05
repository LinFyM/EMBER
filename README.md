# EMBER

EMBER研究如何把目标task的exact language和一条或多条action-hidden教学视频，在rollout前一次性编译成冻结PI0.5 Action
Expert的一套完整LoRA，使policy零交互完成任务。

最新已裁决方法为Unified Policy-Native Factor Writer。它保留ECP已通过的PI0.5原生policy-response、ordered video
dynamics、真实native X/Y、signed pooling、rank4与唯一LoRA物化；主图只有一种可复制factor block，每层以同一query并行cross-attend
exact language、image patches、完整policy response和side-native bank；四个source各自softmax后直接相加，再沿teacher time与
rank/side交互并直接pool raw X/Y。历史Writer、neural q_pi、旧Policy-Response relation/gain链、ECP Stage 1 v1--v24、
MDCO/PECS、人工process任务及已裁决G3/PNBTT实现不作为active fallback。该v4候选的held5相邻点没有形成稳定增量，现已sealed；
当前没有active goal、active design、训练或评测任务，仓库处于交接状态。

## 当前状态

- source、固定24/8/8 split、task experts、rank16 LoRA与评测合同已建立；
- task-local rank16 oracle为250/400，证明Action Expert LoRA有闭环容量；
- G1真实38-target native banks与task-local signed pooling Gate已经通过；G2 boundary-anchored Natural Program Gate已经通过；
- P0/P1证明current-bank operator与exact signed replay具有强task-local容量；随后旧G3的
  `summary -> family-scalar gate -> shared event-additive anchor`在充分校准后仍不能同时保留correct并压低wrong，已经正式停止；
- PNBTT完成single/family chart、两次spectrum、full-rank16与gate-aligned E1后稳定`non_pass`；它能压低wrong但不能同时恢复
  task1/93 correct/held，现只作历史证据；
- 2026-09-02完整历史复核后，owner确认Policy-Response Event-to-Factor Writer路线：冻结PI0.5逐帧捕获
  layer x horizon x probe response，沿视频时间形成ordered events，再由events在当前视频真实X/Y bank中直接执行signed selection；
- 训练只使用正确视频的cross-episode functional；所有负controls在selected checkpoint冻结后评测；
- 旧privileged q_pi/realizer和人工process路线已正式关闭；
- 唯一正式性能目标线是validation8 strict paired correct严格`>145/400`，并同时要求稳定性、breadth、四suite非零、
  Goal/Long贡献、same-task鲁棒性和视频因果controls；
- Final保留从已验证组件初始化和整套Writer完全随机初始化后端到端fresh联合训练两类候选；G1--G3不是Final强制课程。

Unified common-base v3的matched task1/task93 25/50控制通过，但73-task shared的m100/m200 held5只有`35/31`，低于carrier43并随
训练退化。职责替换显示learned evidence可跨task给出正增量，失败集中在重复factor blocks；信息流又定位到exact language在与256 patch、
400 response共享softmax时多数层仅约2.2%质量。v4只让同一套policy-attention权重分别读取language、patch、response，各自
softmax后与side-native read相加，不增加参数、stage、gate或手工校正。其73-task m25/m50 held5为`45/40`，carrier为43；两点
breadth均`3/5`且Goal/Long为0，m25的小幅净增没有被m50保持。因此v4短资格non-pass，不直接续训或进入mixed-K/Final。

## 阅读顺序

1. `AGENTS.md`：仓库总合同；
2. `docs/current_owner_requirements.md`：owner稳定目标与约束；
3. `task_plan.md`、`findings.md`、`progress.md`：当前计划、结论和进度；
4. `docs/concept.md`：科学问题与ECP假设；
5. `docs/expert_review_20260824_native_factor.md`：2026-08-24专家回复原文；
6. `docs/expert_review_20260826_bank_conditioned_native_factor.md`：2026-08-26第二次专家回复原文；
7. `docs/expert_review_20260828_g3_functional_sketch.md`：2026-08-28第三次专家回复原文；
8. `docs/expert_review_20260829_joint_program_primal.md`：2026-08-29第四次专家回复原文；
9. `docs/expert_review_20260830_program_bank_interaction.md`：2026-08-30第五次专家回复原文；
10. `docs/expert_review_20260831_event_conditioned_bank_set_relative_interaction.md`：2026-08-31第六次专家回复原文；
11. `docs/expert_review_20260901_program_through_bank_bottleneck.md`：2026-09-01第七次专家回复原文；
12. `docs/expert_review_20260902_global_route_reassessment.md`：2026-09-02全局路线复核原文；
13. `docs/expert_review_20260902_full_history_policy_native_meta_writer.md`：完整EMBER历史复核原文；
14. `docs/expert_review_20260902_policy_response_event_to_factor_writer_clarification.md`：最新架构澄清原文；
15. `docs/unified_policy_native_factor_writer_design.md`：最新已裁决并sealed的design；
16. `docs/axial_policy_response_native_factor_writer_design.md`：已裁决的直接前身；
17. `docs/policy_response_event_to_factor_writer_design.md`：已裁决的早期直接前身；
18. `docs/program_conditioned_native_bank_tangent_transport_design.md`：已裁决PNBTT历史合同；
19. `docs/research_history.md`：影响当前决策的历史证据。

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
tests/                   canonical与历史可复用运行面的定向测试
```

`data/`、`models/`、`runs/`和`.venv/`是ignored本地资产，不提交远端。现成LIBERO数据、tokenizer、唯一formal checkpoints和结果
应复用；人工process路线和约11.6GB可重建主要产物已删除，recovery Gate A残留作为历史formal evidence保留。

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
scripts/train_ecp_policy_response_writer.py
scripts/evaluate_source_sft_validation_loss.py
scripts/evaluate_ecp_stage0.py
scripts/evaluate_pi05.py
```

正式GPU运行前必须由新owner/goal登记active design，读取项目合同，检查两个GPU节点与对应storage quota，并从clean pushed commit的
frozen worktree启动。跨session接手时先读根目录临时`HANDOFF.md`，消费后删除；长期事实仍以正式账本为准。
