# EMBER cross-session handoff

本文件只提供接手入口与现场快照；长期科学事实已经写入`task_plan.md`、`findings.md`、`progress.md`和
`docs/research_history.md`。新owner完整消费后应删除本文件。

## 现场

- canonical workspace：`/data1/user/ymdai/projects/EMBER`；接手时先确认`HEAD == origin/main == remote main`。
- 本次收口所基于的最后科学提交为`24819ed5daacdeb03c1b71b7baef00b9314de4b8`；交接文档与清理另由其后的main提交记录。
- 当前没有active goal、active design、active配置、训练、评测、GPU任务或EMBER进程。
- 清理完成后只保留canonical worktree，`.codex/tmp`为空。
- 本地与远端只保留`main`分支。旧G3独占实现`2295f481dcd284e4bae92afeaf2cf5c4b2d3e5c2`由归档标签
  `archive/g3-vector-interaction`保留，原`codex/g3-vector-interaction`分支已删除；`findings.md`与
  `docs/research_history.md`中的旧分支名称是历史定位，不代表active路线。

## 科学停止点

最新已裁决设计是sealed的Unified Policy-Native Factor Writer v4：

- shared m25/m50内部true-held functional均有微弱正信号；
- held5 strict paired correct分别为`45/250`与`40/250`，stable carrier为`43/250`；
- 逐task Long/Goal/Object/Spatial0/Spatial9分别为`0/0/4/37/4`与`0/0/3/34/3`；
- breadth均`3/5`，Goal/Long均0；m25到m50为`38 retained / 2 gained / 7 lost`。

因此v4短资格non-pass：m25的净`+2`没有相邻稳定性或困难suite贡献，不允许直接续训、mixed-K、fully-random Final、validation8或
negative controls。输出并未坍缩、全部参数组确实移动，所以这是科学non-pass而非工程故障。

仍有效的正证据包括G1真实native X/Y、signed pooling与rank4 task-local容量，G2完整PI0.5 response中的ordered video dynamics，
task-local Writer容量，以及v4将language/patch/response分源读取后改善task grounding的内部信号。负结果只淘汰实际测试的shared
参数化，不否定整个ECP或EMBER。

## Formal evidence roots

- shared训练：
  `runs/outputs/pi05_ecp_policy_response_writer_source_separated_73task_k1_component_s50_f02f9148_gpu01p036_sharedmmap_20260905/`
- m25物化：
  `runs/outputs/pi05_ecp_policy_response_writer_source_separated_m25_held5_correct_k1_materialized_f02f9148_gpu01p3_20260905/`
- m25 strict250：
  `runs/outputs/pi05_ecp_policy_response_writer_source_separated_m25_held5_correct_k1_strict250_f02f9148_gpu01p036_20260905/`
- m50物化：
  `runs/outputs/pi05_ecp_policy_response_writer_source_separated_m50_held5_correct_k1_materialized_f02f9148_gpu02p2_20260905/`
- m50 strict250：
  `runs/outputs/pi05_ecp_policy_response_writer_source_separated_m50_held5_correct_k1_strict250_f02f9148_gpu02p236_20260905/`

以上roots的completion、raw rows、run contract和launch bundle完整；不得作为普通缓存删除。更早证据从
`docs/research_history.md`选择性定位，不要重新遍历全部退役代码。

## 接手边界

1. 严格按`AGENTS.md`的task-scoped reading读取相关authority到EOF；当前没有active design，不能自动恢复旧文档中的“next”。
2. 新工作必须由owner重新授权、建立goal并在`progress.md`登记active design后开始。
3. full 50-step horizon是硬边界，绝不恢复coarse、horizon mean或等价抹平。
4. 架构必须优雅、职责清楚并可通过复制少数标准attention/MLP层扩展；不得用gate、summary、whitening、transport、手工增益或
   calibration等数学补丁链挽救non-pass。
5. correct相对wrong/乱序的优势应自然产生；negative controls不进入训练loss，selected checkpoint冻结后才评测。
6. 最终唯一目标仍是validation8 strict paired correct稳定`>145/400`，并满足breadth、四suite、Goal/Long、same-task及视频因果合同。
7. 真正launch前同时live检查gpu01/gpu02和对应quota；至多使用单节点6张真正提高吞吐的A40，可安全共驻但不得干扰他人。

## 清理说明

tracked历史代码、配置、专家原文与formal artifacts被有意保留用于审计和复用；它们不是active fallback。此次删除了`.codex/tmp`、
pytest/Python cache、已完成detached worktree，以及唯一一份61,030,186,361-byte的退役G3 frozen-condition cache；该cache的重建命令与
provenance仍在2026-08-31的launcher/analysis中。`runs/logs`和formal/analysis roots作为唯一科研证据保留，避免在没有继任设计的情况下
破坏可复核历史。
