# EMBER Progress

更新时间：2026-08-18。本文只记录当前可执行状态；目标计划见`task_plan.md`，跨实验认知见`findings.md`，历史精确
结果见`docs/research_history.md`。

## Current state

- 持续研究goal为active；当前不使用subagents。
- canonical workspace为`/data1/user/ymdai/projects/EMBER`，主写分支为`codex/bci-continuation`。
- 当前active design为`docs/layer_matched_memory_program_compiler_design.md`。
- LMMPC-v2已经完成fresh macro25/50、两次strict paired400和逐stage终局分析；旧checkpoint不得resume。
- LMMPC-v3 canonical实现与formal recipe由clean pushed `af76558075315b6ea954e60feff44dfaac0637e3`封存。
  fresh world3/B20 formal macro1--25已经完整exit0并保留checkpoint25；同一run正按锁定的gpu02物理`1/3/7`
  exact-resume到macro50，未改变架构、优化器、sampler或recipe。
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
- clean world3两macro=`58.55/55.32s`，functional=`.156120→.153991`，每轮K1--K4各6 tasks；peak
  allocated/reserved=`35,437,871,616 / 35,720,790,016` bytes，无OOM/nonfinite/forbidden read。
- 371-frame完整五任务序列自然完成，peak allocated/reserved=`41,851,758,080 / 42,393,927,680` bytes。
- macro2真实K4中，raw axial proposal相对anchor达`32.23x`，bounded commitment却只改写`.250003` relative-L2；
  320个live cells最大correction/anchor RMS=`.250395`，低于结构上限`.5`。gate=`.249998`且gate/blocks均有梯度。
- repeated-last使parameter memory relative-L2=`.999716`、BA=`.493903`；reverse/shuffle完整重前向后的compiled
  relative-L2=`1.06281/.36850`，BA=`.39081/.13498`。这些只证明链路读取顺序，不是correct优于controls的成功证据。
- constant/template和K置换max-abs均为0；八factor family和reader路径全有梯度；source policy非零gradient tensor为0；
  checkpoint中VL Meta-LoRA参数为0，deployment recompile逐元素一致。
- 新K4 validation8×4 fixed-panel generation profile由同一clean v3 runtime完成；batch8/16/32均稳定，LoRA/s分别
  `.21314/.21489/.21627`，batch32 peak reserved=`20,231,225,344` bytes且余量`27,468,496,896` bytes，零
  OOM/nonfinite/禁读，因此部署继续选batch32。旧v2 generation profile不再作为v3 authority。
- fresh macro1--25每轮24 tasks、K1--K4各6，五个loss窗口为
  `.152438→.137125→.129451→.122818→.119420`；checkpoint、optimizer/scheduler和三rank RNG完整，零
  OOM/nonfinite/禁读。曲线尚未见峰，故在macro25 strict并行期间exact-resume到macro50。
- macro2 validation8×4分层基线已完成：Core-fused→compiled的between-task cosine `.37559→.35412`，
  correct/reverse relative-L2 `.87045→.91225`，表明fresh bounded M2P没有复现v2的初始过度平滑；这仍不是
  closed-loop成绩，必须与macro25/50同口径stage和strict结果比较。

## Immediate next work

1. 验证并提交/push新的v3 K4 generation profile authority，从clean detached commit准备macro25 formal评测。
2. macro25执行strict paired400、validation8逐stage和逐task/suite/retention/churn分析；同时保持同一run exact-resume
   到macro50，不在仍下降的训练曲线上过早终止。
3. macro50重复同口径strict/stage并分析25→50共同积累或换手。首次约145且retention合理时立即补六臂和same-task
   不同视频鲁棒性；否则依据最早失效接口只做LMMPC主链内的局部单变量改进。

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
