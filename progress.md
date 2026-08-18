# EMBER Progress

更新时间：2026-08-18。本文只记录当前可执行状态；目标计划见`task_plan.md`，跨实验认知见`findings.md`，历史精确
结果见`docs/research_history.md`。

## Current authority and scope

- 本轮有限goal已完成，当前停止；暂不使用subagents。
- canonical workspace为`/data1/user/ymdai/projects/EMBER`，主写分支为`codex/bci-continuation`。
- 当前没有active design、训练、评测或successor。`docs/layer_matched_memory_program_compiler_design.md`已封存为
  EMBER-LMMPC Core-Addressed Reader的终局设计与证据，不得把其中历史“下一步”恢复为authority。
- 下一session必须依据owner最新要求重新建立goal；本轮仅登记一个未经实施的单变量候选，不授权自动launch。

## Sealed scientific state

- 同一world6/topology formal run已exact-resume到macro100，macro25/50/75/100四个K4 strict paired400为
  `123→84→89→87`，breadth为`8→5→6→4`。best仍是macro25，未达到145资格门，未做六臂controls。
- 400个固定rows中49个始终成功、150个曾成功。macro25→50丢失的52行只有22行在macro75或100任一点恢复，
  macro100只恢复15行；macro25→50新增13行到macro100只保留6行。后期小幅回升是循环task换手，不是共同积累。
- 固定K4+B20 train24 loss为`.112124→.099353→.098427→.101337`。25→50 offline support显著改善时held
  strict净丢39；75→100固定support也开始忘记，证明functional surrogate和held occupancy长期错位。
- 相邻compiled Program relative-L2为`.770/.730/.710`。FactorHeads主导25→50 norm扩张，但后两段heads-only和
  Program-only BA变化相当；FactorHeads是错误credit的放大器和载体，不是唯一根因，Program也持续漂移。
- 终局最早失效接口是
  `static offline functional query occupancy -> shared support-preserving credit -> held closed-loop occupancy`。
  Core-Addressed Reader、ordered Core/Procedure、layer/rank memory、Dynamic-K、bounded K-set/M2P与native rank16
  输出是保留机制；本负结果不授权回头增加Procedure negative、扩大memory或扫rank/scale/LR/seed。

## Canonical evidence

- formal train：
  `runs/outputs/pi05_lmmpc_v5_formal_fresh_r6_b20_aecbce5_gpu01p124567_20260818`；
- macro75 strict：
  `runs/outputs/pi05_lmmpc_core_addressed_macro0075_k4_correct400_noreplacement_seed7_trainr6_evalr3_f42edfc_gpu02p237_20260818`；
- macro100 strict：
  `runs/outputs/pi05_lmmpc_core_addressed_macro0100_k4_correct400_noreplacement_seed7_trainr6_evalr5_f42edfc_gpu01p12456_20260818`；
- four-checkpoint strict trajectory：
  `runs/analysis/lmmpc_four_checkpoint_strict_trajectory_20260818.json`；
- Program×FactorHeads cross-decode：
  `runs/analysis/lmmpc_program_factorheads_cross_decode_macro25_50_75_100_20260818.json`；
- B20 panels：`runs/analysis/lmmpc_b20_task_gradient_macro25_to50_20260818.json`、
  `runs/analysis/lmmpc_b20_task_gradient_macro50_to75_20260818.json`、
  `runs/analysis/lmmpc_b20_task_gradient_macro75_to100_20260818.json`、
  `runs/analysis/lmmpc_b20_task_gradient_macro100_20260818.json`；
- combined diagnosis：
  `runs/analysis/lmmpc_macro25_50_75_100_drift_diagnosis_20260818.json`。

## Next-session boundary

唯一优先候选是保持Reader、Program、compiler、FactorHeads和rank16全部不变，只把functional query distribution从
静态cross-episode demo-state B20替换为train24 on-policy state replay，并由冻结task-local experts提供action targets。
它测试的是occupancy-matched credit能否让同一checkpoint同时保留breadth和相邻strict success set；部署仍为
exact language + action-hidden videos一次生成LoRA，不携带expert，也不做生成后task-local RL。

该候选不是active authority。新session若选择实施，应先精确定义state collection、expert labeling、task等权和fresh
边界，再做单变量机制/profile/formal实验；若train on-policy breadth与相邻strict400仍不共同改善，责任才后移到
Program/decoder本身。本轮到此停止，不续macro、不启动GPU、不创建持久handoff。
