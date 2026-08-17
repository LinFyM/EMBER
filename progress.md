# EMBER Progress

更新时间：2026-08-18。本文只记录当前可执行状态；目标计划见`task_plan.md`，跨实验认知见`findings.md`，历史精确
结果见`docs/research_history.md`。

## Current state

- 持续研究goal为active；当前不使用subagents。
- canonical workspace为`/data1/user/ymdai/projects/EMBER`，主写分支为`codex/bci-continuation`。
- 当前active design为`docs/layer_matched_memory_program_compiler_design.md`。
- LMMPC-v2已经完成fresh macro25/50、两次strict paired400和逐stage终局分析；旧checkpoint不得resume。
- 当前主工作树从clean pushed `df3ae632fad644e7018887f5f0cea3dcd2ad0389`原位实现LMMPC-v3，尚未形成clean
  commit或启动v3 GPU运行。
- v3唯一科学变量是把已经定位会覆盖Core-fused Program的unbounded axial M2P改为逐cell identity-anchored
  bounded residual commitment；四流、V6 Core/Procedure、layer/rank memory、K-set、rank16 native A/B和训练合同不变。
- fresh v3同步移除冻结、B=0、从不更新的VL Meta-LoRA及其hook；这是行为等价工程清理，不算第二个科学变量。

## LMMPC-v2 terminal evidence

同一fresh world5 run完成macro25和exact-resume macro50：

| checkpoint | correct | breadth | per-task | per-suite |
| --- | ---: | ---: | --- | --- |
| macro25 | `71/400` | 6 | `2/0/31/2/0/34/1/1` | `2/33/34/2` |
| macro50 | `73/400` | 6 | `1/0/35/13/5/15/4/0` | `1/48/20/4` |

- 25→50=`42 retained / 31 gained / 29 lost`、churn60、net`+2`、Jaccard`.411765`；Object净增15而Goal
  净丢14，属于task换手而非shared accumulation。
- macro50相对LPCP143=`61 retained / 12 gained / 82 lost`、churn94、net`-70`；相对v1 macro50 101为
  `49/24/52`、churn76、net`-28`。
- macro41--50 loss约`.112`平台；25个额外macro没有形成共同closed-loop上升，因此v2不续macro75、不补六臂。
- macro50 formal strict root：
  `runs/outputs/pi05_lmmpc_v2_macro0050_k4_correct400_noreplacement_seed7_trainr5_evalr6_df3ae63_gpu01p01456_20260818`。
- 初次六卡evaluation因另一用户随后在物理2启动高显存任务而单worker OOM；失败证据保留，未触碰外部进程。正式结果
  来自重新live选取物理`0/1/4/5/6`的完整五卡panel，400 rows和60/60 jobs均完成。

## Earliest failure localization

v2确实解决了v1的Procedure-reader endpoint bypass：完整Procedure阶段进入每个固定layer/rank memory address，顺序信号
在reader和Core fusion后仍然material。新的最早断点是后置M2P覆盖已经有用的Program：

| stage | between-task cosine | correct/reverse relative-L2 | same-task four-K4 cosine |
| --- | ---: | ---: | ---: |
| Core-fused grid | `.338088` | `.257271` | `.992207` |
| M2P block0 | `.4943` | `.1502` | — |
| M2P block1 | `.6560` | `.0938` | — |
| final effective BA | — | `.086224` | `.999775` |

- 第一层相对anchor改写`4.50034x`，第二层再相对输入改写`1.75311x`；两层共同把task分离和order差异压平。
- output RMSNorm单独几乎不改变task/order指标，因而不是最早断点。
- Procedure本身correct/reverse relative-L2=`.820147`，H_set=`.398023`；因此不能把v2低分重新归因于carrier没有
  读取视频或完整stage reader未接通。
- v2 macro50 hidden的只读bounded counterfactual：逐cell residual初始`.25x`/上限`.5x`时，between-task分别
  `.360797/.405605`、order分别`.247873/.230772`、same-task K4分别`.992778/.993867`。这证明bounded设计直接
  针对已定位接口，但不替代fresh训练和closed-loop结果。

精确analysis artifacts：

- `runs/analysis/lmmpc_v2_macro50_strict_paired_comparison_df3ae63_20260818.json`
- `runs/analysis/lmmpc_v2_macro50_validation8_stage_localization_df3ae63_20260818.json`
- `runs/analysis/lmmpc_v2_macro50_m2p_block_localization_df3ae63_20260818.json`
- `runs/analysis/lmmpc_v2_macro50_m2p_signal_localization_df3ae63_20260818.json`
- `runs/analysis/lmmpc_v2_macro50_m2p_bounded_counterfactual_df3ae63_20260818.json`

## Active LMMPC-v3 decision

v3保持统一主链：

```text
exact language + dynamic-K ordered action-hidden videos
  -> V6 task-grounded Semantic Core
  -> Action-query Visual-Transition Causal Procedure P[1:T]
  -> 16 one-way layer/rank memory states M[t,l,r]
  -> every fixed (l,r) reads all Procedure stages
  -> address-preserving K-set consensus
  -> dynamic Core fusion gives anchor Y[group,rank]
  -> same two axial blocks propose Z[group,rank]
  -> cellwise bounded commitment Y + gate * limited(Z-Y)
  -> jointly trained native rank16 A/B
  -> one complete 38-target LoRA
```

commitment合同：

```text
delta = Z - Y
limited = delta * min(1, RMS(Y) / max(RMS(delta), 1e-6))
gate = 0.5 * sigmoid(g)       # fresh g=0, initial gate=.25
committed = Y + gate * limited
```

每个`group × rank`地址上的M2P correction始终不超过anchor RMS的`.5x`；axial blocks仍能跨层/跨rank协调，但不能
再覆盖已有动态Program。rank保持16；不加入`H_correct-H_reverse`、matching loss、reverse训练臂、reward/RL、expert
bank或其它新监督。

当前实现状态：

- canonical runtime、config/checkpoint/eval schemas已升级为fresh-incompatible v3；v2 config原位退役。
- bounded commitment、可学习gate和零输入identity已实现。
- fresh LMMPC不再实例化VL Meta-LoRA，legacy V6显式传rank4的路径仍保留。
- 定向CPU=`76 passed`、全量CPU=`285 passed`、architecture guard无hard violation；当前diff-check通过。
- v3 config仍为`unsealed_pending_live_profile`，不得在完成clean真实机制/吞吐门前启动formal训练。

## Immediate next work

1. 完成progress和窄合同收尾，重跑CPU/compile/diff/architecture检查，提交并push v3 mechanism candidate。
2. 从clean pushed detached commit同时live检查两节点和quota，按实际可用A40运行真实K4机制、两macro吞吐和371-frame门。
3. 验证VL Meta-LoRA参数/hook为0、source zero-grad、constant identity、K置换、repeated-Procedure响应、完整
   correct/reverse/shuffle重前向、M2P correction bound/gate/blocks与八family梯度、native BA/action及无OOM/nonfinite。
4. 用clean evidence封存v3 profile；重新profile K4 Writer generation batch，随后从fresh train24到macro25。
5. 执行strict paired400和逐task/suite/stage/retention/churn分析；若链路和closed-loop仍在共同上升，继续macro50及相邻
   checkpoint，不在性能峰值前过早终止。首次约145且retention合理时立即补六臂和same-task不同视频鲁棒性。

## Fixed scientific baselines

- v6-fast五臂：`143/135/125/128/129`；
- LPCP K4 correct：`143/400`、breadth7；
- SFMC单点：`144/400`但lost15/churn31且无六臂；
- GOMQ：`151→135→131`，证明learned memory有真实增益但旧rank32 direct-B/shared reward不稳定；
- LMMPC-v1：`81→101`，fresh memory-grid Writer可学习，但旧reader被endpoint旁路；
- LMMPC-v2：`71→73`，完整stage reader接通，但unbounded M2P覆盖Core-fused Program。

## Storage and runtime note

历史整理时`/data1`个人用量约542.2 GiB / 1 TiB，该值会漂移。任何formal launch前重新在`strg01`检查独立quota、
测实际用量并估计峰值；每次GPU launch前同时live检查gpu01/gpu02，按实时余量使用单节点至多6张真正提高吞吐的
A40。低util、少量显存占用的设备在峰值余量足够时可与ycliu共驻。
