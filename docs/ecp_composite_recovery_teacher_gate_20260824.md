# ECP composite-context recovery teacher Gate

状态：2026-08-24，owner已授权执行。本文把
`docs/ecp_recovery_teacher_expert_ruling_20260824.md`落实为唯一formal训练与闭环裁决合同。

## 1. 科学问题

检验A3真实成功轨迹中的第二阶段动作，能否把原primitive expert校准成在“另一个goal已完成”状态下仍可靠的recovery
expert。该实验只建立训练期privileged composite teacher，不是EMBER部署Writer，也不重开当前已关闭的shared realizer。

## 2. 两个固定训练臂

两个臂都冻结source PI0.5，只训练一套38-target rank16 LoRA，不共享参数：

- `yellow_after_red`：从task68 step1000 primitive LoRA初始化；phase language为
  `put the yellow and white mug on the right plate`；
- `red_after_yellow`：从task65 step1000 primitive LoRA初始化；phase language为
  `put the red mug on the left plate`。

每步batch固定为16，其中8条从A3成功episode的first-event completion之后到episode end等概率抽取，8条从对应50条
primitive成功episodes等概率抽取。每个domain独立确定性shuffle；两臂均训练1000 steps，AdamW、BF16、peak LR
`1e-5`、warmup 25、cosine decay到`1e-7`，只保留step1000 checkpoint。不同方向不单独调参或选step。

action query从当前segment内最多读取50步；不足50步时padding并以`action_is_pad`从flow loss中排除，绝不跨first-event
boundary或episode end。只使用真实执行且最终成功的动作，不读取A3失败states和已取消distillation的`2773/3998`
weak-teacher labels。
两类HDF5的原始RGB分别为256与128；mixed batch在collate前分别调用PI0.5原生`resize_with_pad_torch`到model224，使结果与
模型原本逐样本预处理一致，不让任一domain决定另一domain的图像尺度。

配置：

- `configs/pi05_ecp_composite_recovery_experts_v1/yellow_after_red.json`
- `configs/pi05_ecp_composite_recovery_experts_v1/red_after_yellow.json`

## 3. Formal privileged controller

每个方向的第一event固定使用原primitive step1000 expert；event完成即丢弃旧action chunk，重新观察并切换到对应recovery
expert完成第二event。统一使用现有separate-plates family language、initial states、temporal wrapper、RNG与公开video信息墙。
ledger必须额外记录每次replan使用的是primitive还是direction-specific recovery checkpoint，route mismatch为硬失败。
`scripts/adjudicate_ecp_recovery_teacher_gate.py`从privileged ledgers核对该route、两向state/noise pairing、公开video信息墙、
first-predicate post-completion drop与全部固定阈值；发生drop后又恢复的episode不计strict success。裁决同时按variant报告
first/second event完成与first-event drop，并与A3同100行authority报告retained/gained/lost；后者只作定位，不改变Gate阈值。

## 4. Gate A与停止规则

唯一100-row panel固定为每方向50个相同state IDs：

- red-first至少`20/50`；
- yellow-first至少`20/50`；
- total至少`50/100`；
- wrong-first invalid、phase/expert route mismatch、state/noise pair mismatch和public-video privileged leakage全部为0；
- 每个success保持第一predicate并最终完成conjunction。

不按state、结果或checkpoint筛选。通过后立即运行原Gate B。若失败，则关闭task65/68 primitive composition、composite SFT、
phase-expert distillation和second-phase recovery SFT；不延长、不扫step/LR/rank/seed、不换task66/67重复，转向必须先独立
通过同一Gate A的planner、human/teleoperation、MPC或task-local simulator RL composite controller。
