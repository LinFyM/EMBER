# Writer 因果路径与辅助梯度诊断（2026-09-22）

## 状态与目的

本设计响应 Owner 转交的专家后续意见，状态为**已登记、待实现核验与资源 preflight**。它是冻结资产上的有界
分析实验，不是新的 Writer 候选训练，也不改变已完成的跨 episode 辅助配对结论。

目标只有两项：

1. 在相同目标语言下，量化 Core 与 Procedure 分别从正确、同任务其它或同 suite donor 视频取得信息时，对生成
   LoRA、动作探针和小训练面板闭环的影响；
2. 在当前 `cross_episode` 辅助合同下，区分主 FM 梯度、端点前缀辅助梯度、AdamW 动量/weight-decay 与总步幅的
   短窗作用。

它不声称定位历史性能下降的唯一根因，不选择正式 checkpoint，不产生 Test/FT/RL/外部比较，也不据结果自动启动
纯主 FM、改 LR、换 seed、改 head 或延长训练。

## 冻结资产、数据边界与预算

| 代号 | 冻结资产 | 本设计中的用途 |
| --- | --- | --- |
| O1200 | `video_teaching_20260919/.../macro_00001200` | D1 CC/WW、D2 梯度/虚拟更新 |
| C600 | `coverage_retraining_cross_episode_aux_20260922/.../macro_00000600` | D1 全五臂、D2、D3 父状态 |
| C1200 | 同一 fresh cross-episode run 的 `macro_00001200` | D1 全五臂，仅冻结路径比较 |
| M300 / Source | 已冻结 MT-BC300 / Source-71 | D1 正确输入动作探针参照，仅 32 条各一组 |

所有动作 query、视频和闭环任务均来自当前 36 个训练侧允许集合；不会读取 validation/test action 或 reward。D1/D2/D3
分别使用同一 current cross-episode event generator 的原始注册事件；raw video cache 只能缓存 RGB 与 frame index，不能
缓存参数相关的 Meta activation。

总上限在第一次 GPU launch 前固定为：最多 120 分钟 GPU 墙钟、D1 544 条函数 probe、D2 72 个唯一 task condition 与
16 个虚拟候选、D3 最多 54 个真实更新、最多 160 条训练侧闭环。D3 只在 D1/D2 全部通过实现/完整性检查、且保守估算能在
上限内完成其全部三个分支和终点测量时启动；不因任一部分分数选择性启动或取消某一分支。

## D1：Core / Procedure 路径消融

固定八个当前训练任务：Spatial `5,7`，Object `12,14`，Goal `20,25`，Long `34,37`；donor 配对固定为
`5↔7`、`12↔14`、`20↔25`、`34↔37`。每个目标 task 使用两组已登记条件：

| 组 | teacher / 独立动作 query / same-task-other | rollout init state |
| --- | --- | --- |
| A | `46 / 47 / 48` | `0` |
| B | `48 / 49 / 46` | `1` |

每组各取 query episode 的 25% 和 75% 两个位置，固定噪声由 manifest 的条件键确定。因此每 model/arm 是
`8 × 2 × 2 = 32` 条 probe；不是事后选样本。

所有 arm 始终传入**目标 task 的 exact language**。视频在其各自目标语言下先独立编码，再只在明确接口处替换：Core 与
`valid_core` 一起旅行，Procedure 与 positions/valid-frame mask 一起旅行。绝不截断、平均 Procedure，不跨 model 混
hidden，也不替换语言。

| arm | Core 来源 | Procedure 来源 |
| --- | --- | --- |
| CC | 目标正确 teacher | 同一正确 teacher |
| CW | 目标正确 teacher | paired donor 的同编号视频 |
| WC | paired donor 的同编号视频 | 目标正确 teacher |
| WW | paired donor 的同编号视频 | 同一 donor 视频 |
| CO | 目标正确 teacher | 目标 task 的 same-task-other 视频 |

O1200、C600、C1200 完成五臂（480 条）；M300 和 Source 各完成正确输入 32 条（64 条）。C600 五臂各做
`8 × state(0,1)=16` 条训练侧闭环，O1200 仅 CC/WW（32 条），共 112 条。预登记仅保存 `global5/state0` 和
`global37/state1` 的紧凑轨迹；其余仍保留 official 行、success、步骤、stage predicate、最终状态及 actions/states
的轻量索引。

每个 probe 写入随机-flow 全50/前5 FM MSE、`tau=1`前5 MSE、相同 noise 的10-flow前5 action MSE、相对同 asset
CC 的 action delta，以及 E（frame evidence）、H（horizon）、Core、Procedure、compiler fused slots、LoRA A/B 的相对
变化。效应表只机械报告 `CC−CW`、`WC−WW`、`CC−CW−WC+WW`，闭环 success 与负 action error 分开。

## D2：梯度、AdamW 和虚拟更新

对 O1200 与 C600，使用当前 cross-episode 原始 events `604..612`。其中 B4 是原始 step604 的四 task 平均，B36 是
九个四 task batch 的完整平均；同一36个 task condition 的逐 draw 主/辅助梯度只计算一次再按这两个窗口聚合。`g_Q`
为原生随机-flow完整50-horizon主项，`g_A`为固定`tau=1`、前5、且已经乘入`1/3`的辅助项；二者都穿过完整 Writer
（包括三组 Meta）。

参数按互斥的七组记录：Text Meta、VL Meta、Action Meta、剩余 semantic projections/Core、Procedure、Compiler、
FactorHeads。记录每组 norm、Q/A dot/cos、B4/B36方向差与九个四-task batch的Gram。参数命名、optimizer parameter
identity 和 Adam state 必须一一对应；无任何参数重复计数。

每个 asset×窗口从同一完整 AdamW state 构造四个候选：

| 候选 | 操作 |
| --- | --- |
| J | `AdamW(g_Q+g_A)`，含现有 global clip 和 weight decay |
| Q | `AdamW(g_Q)`，其它完全相同 |
| M | 先执行本状态的 J 以得到其 Adam state 和 displacement `u_J`，再把 parameter displacement 写为 `alpha*u_J`，`alpha=||u_Q||/||u_J||` |
| Z | 对每个参数提供 zero current gradient 的 AdamW step；不是 `grad=None`，因此保留动量和 weight decay 的真实影响 |

每个候选立即从父状态恢复后进行独立32条固定 probe，最多512条。候选不保存为正式模型，测量后恢复父 state。
每个 asset 使用其 checkpoint 中恢复的单一 AdamW parameter-group LR；D2 比较同一父状态内的 J/Q/M/Z，不能把
O1200 与 C600 的绝对 displacement 当作 LR 已匹配的跨资产比较。

## D3：18步三臂微训练

仅在 D1/D2 的 launch gate 通过后，从完整 C600 checkpoint 独立复制三条状态，使用同一固定 LR
`0.0002992056748283996`（C600 后 step601 的实际 LR），原始 events `601..618`，各18次实际更新：

- J：主项 + cross-episode 辅助；
- Q：仅原始21条主 query；
- M：每一步都从 **M 当前 parameters 与 Adam state** 分别提出 Q/J，再用该步的 `alpha*u_J` 更新 parameters，同时保留
  J 所产生的 Adam state；不借用 Q 分支 state。

每个新 step 都重新视频编码、重新生成 LoRA/cotangent；不会复用更新前 Meta activation。只在终点运行各 branch 的
CC/WW 32条 probe 与 CC16条闭环，并与 D1 C600 初始 CC/WW 配对；WW不做rollout，也不能作为视频特异性证据。禁止中间
probe、curve 或基于中间分数的续训。

## 交付与验收

### 实现所有权与生命周期

`model.py`仍是唯一 Core/Procedure→compiler→FactorHeads 解码器；本轮只向它暴露同一条内部 bundle 接口。
`stability_diagnostics.py`仍拥有 checkpoint/AdamW 恢复与真实 functional credit，`stability_rollouts.py`仍拥有官方
`rollout_shard` 小面板。本设计新增的 `causal_diagnostics.py` 只编排这三个既有能力的冻结测量，
`run_writer_causal_diagnostics.py` 只是单 phase runner，不分配 GPU、不启动 worker、不充当第二套 controller。
它们不接入普通训练、materialization 或正式 Validation 入口；若 Owner 不采纳任何由本轮提出的后继假设，在本轮结果
封存并可由 Git/正式原件复算后，将在下一次相关 cleanup 中退役该专用 runner/module，而不是成为平行 Writer 路径。

study 根拟为 `/data0/user/ymdai/ember_runs/writer_causal_diagnostics_20260922`。必须保留：

`path_probe_rows.csv`、`path_rollout_rows.csv`、`path_effects.csv`、`gradient_rows.csv`、`virtual_update_rows.csv`、
`microtrain_steps.csv`、`microtrain_probe_rows.csv`、`microtrain_rollout_rows.csv`、`trajectory_manifest.csv`、`costs.json` 和
`completion.json`，以及注册资产/manifest、launch contract 和所需轻量 trajectory artifact。

运行前的 required tests：CC重建与普通 forward 等价；Core/Procedure和mask/position成组耦合；主+辅助梯度等于联合反传；
J/Q/Z/M与独立 PyTorch AdamW参照一致；optimizer parameter identity无重复；参数改变后不能复用旧 Meta activation。正式
launch前还要检查 clean pushed detached commit、两节点GPU、两块独立quota、资产存在和预估峰值。

结果解释只按预登记证据路由：Q相对J/M的差异可支持检验辅助目标整体或步幅的后继假设，不能分解`tau=1`、前5与`1/3`
的各自作用；任何正/负结果都不自动改变正式训练路线。
