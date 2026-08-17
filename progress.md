# EMBER Progress

更新时间：2026-08-18。本文只记录当前可执行状态；目标计划见`task_plan.md`，跨实验认知见`findings.md`，历史精确
结果见`docs/research_history.md`。

## Current state

- 持续研究goal为active；当前不使用subagents。
- canonical workspace为`/data1/user/ymdai/projects/EMBER`，主写分支为`codex/bci-continuation`。
- 当前active design为`docs/layer_matched_memory_program_compiler_design.md`。
- LMMPC-v1/v2/v3均已完成fresh macro25/50、strict paired400和逐stage终局分析；旧checkpoint不得resume。
- v3由clean pushed `af76558075315b6ea954e60feff44dfaac0637e3`训练，同一world3/B20 run从macro25 exact-resume
  到macro50；strict由`102`降到`60`，不是可继续训练挽救的上升曲线。
- 当前active successor是fresh-incompatible LMMPC-v4。唯一科学变量是把v3的unbounded nonlinear K-set correction
  改为per-video mean-anchored逐cell bounded commitment；四流、V6 Core/Procedure、layer/rank memory、Core fusion、
  v3 bounded M2P、native rank16 A/B和B20 functional合同不变。

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

## LMMPC-v3 terminal evidence and v4 decision

v3保持统一主链并解决了v2的M2P overwrite，但正式结果终局为：

| checkpoint | correct | breadth | per-task | per-suite |
| --- | ---: | ---: | --- | --- |
| macro25 | `102/400` | 5 | `2/0/47/8/0/37/8/0` | `2/55/37/8` |
| macro50 | `60/400` | 6 | `2/0/24/2/1/26/5/0` | `2/26/27/5` |

- 25→50=`46 retained / 14 gained / 56 lost`、churn70、net`-42`、Jaccard`.396552`；Object/Goal/Long均下降。
- macro50相对LPCP143=`46/14/97`、churn111、net`-83`。v3不续macro75或六臂。
- M2P实际commitment相对Core-fused anchor仅`.24979/.24953`；它没有复现v2的compiler覆盖。
- 新的最早断点是K-set nonlinear correction：raw set output相对per-video mean在macro25/50改写`10.188x/5.831x`，
  把between-task cosine由`.6543/.7672`推高到`.9025/.9218`，同时把same-task condition cosine由`.9954/.9967`
  降到`.9649/.9804`。Core fusion随后恢复部分task分离，所以它不是更早断点。

v4保持的统一主链：

```text
exact language + dynamic-K ordered action-hidden videos
  -> V6 task-grounded Semantic Core
  -> Action-query Visual-Transition Causal Procedure P[1:T]
  -> 16 one-way layer/rank memory states M[t,l,r]
  -> every fixed (l,r) reads all Procedure stages
  -> per-video mean-anchored bounded K-set consensus
  -> dynamic Core fusion gives anchor Y[group,rank]
  -> same two axial blocks propose Z[group,rank]
  -> cellwise bounded commitment Y + gate * limited(Z-Y)
  -> jointly trained native rank16 A/B
  -> one complete 38-target LoRA
```

K-set和M2P都使用同形commitment合同：

```text
delta = Z - Y
limited = delta * min(1, RMS(Y) / max(RMS(delta), 1e-6))
gate = 0.5 * sigmoid(g)       # fresh g=0, initial gate=.25
committed = Y + gate * limited
```

K-set中的anchor是同地址per-video mean，M2P中的anchor是Core-fused grid；两个learned branch都能保留有用修正，
但每个固定地址上的correction始终不超过各自anchor RMS的`.5x`。rank保持16；不加入`H_correct-H_reverse`、
matching loss、reverse训练臂、reward/RL、expert bank或其它新监督。

当前实现状态：

- canonical runtime、config/checkpoint/eval schemas已升级为fresh-incompatible v4；v3 config原位退役。
- K1继续exact identity；K>1的raw DeepSets proposal由fresh gate初始`.25`、结构上限`.5`绑定到per-video mean。
- clean `8c40a56`的world3两macro为`58.13/54.83s`，functional=`.156120→.153994`，K1--K4各6 tasks；peak
  allocated/reserved=`35,550,641,664 / 37,971,034,112` bytes，零OOM/nonfinite/禁读。
- 真实K4中raw K-set proposal仍改写mean `6.403x`，实际commitment仅`.250010x`；最大逐cell correction/anchor
  RMS=`.250531<.5`，gate和全部set branches均有gradient。M2P也保持`.250005x`与全路径梯度。
- repeated-last使parameter memory relative-L2=`.999913`、BA=`.933892`；reverse/shuffle完整重前向后的BA差异为
  `.385271/.084918`。constant/template、K置换、deployment recompile均为0；source policy零梯度、VL Meta-LoRA缺席。
- validation8×4分层门中，per-video mean→K-set的between-task cosine仅`.40147→.40572`、within-task
  `.99100→.97952`，没有复现v3的`.65/.77→.90+`覆盖；order relative-L2为`.49115→.51687`，compiled/BA仍有
  `.49489/.23376`响应。这只证明最早结构门关闭，不是closed-loop成绩。
- 371-frame完整五任务序列完成，peak allocated/reserved=`41,987,913,216 / 44,920,995,840` bytes。正式训练
  profile authority已经封存。
- 同一clean runtime的validation8×4固定panel完成K4 deployment generation profile：batch `8/16/32`分别为
  `.212889/.214594/.216135 LoRA/s`，均覆盖最长226帧且稳定，selected batch=`32`；peak reserved仅
  `20,231,225,344` bytes，零OOM/nonfinite/禁读，Writer modules在handoff前释放。

## Immediate next work

1. fresh train24已从clean formal-seal commit启动；训练到macro25后执行strict paired400、validation8逐stage和
   逐task/suite/retention/churn分析；只有仍存在
   genuine shared上升证据时才按同一run exact-resume到macro50。
2. 首次约145且retention合理时立即补六臂和same-task不同视频鲁棒性；否则依据最早失效接口只做LMMPC主链内的
   局部单变量改进。

## Fixed scientific baselines

- v6-fast五臂：`143/135/125/128/129`；
- LPCP K4 correct：`143/400`、breadth7；
- SFMC单点：`144/400`但lost15/churn31且无六臂；
- GOMQ：`151→135→131`，证明learned memory有真实增益但旧rank32 direct-B/shared reward不稳定；
- LMMPC-v1：`81→101`，fresh memory-grid Writer可学习，但旧reader被endpoint旁路；
- LMMPC-v2：`71→73`，完整stage reader接通，但unbounded M2P覆盖Core-fused Program；
- LMMPC-v3：`102→60`，bounded M2P关闭该覆盖，但unbounded K-set更早破坏per-video task structure。

## Storage and runtime note

历史整理时`/data1`个人用量约542.2 GiB / 1 TiB，该值会漂移。任何formal launch前重新在`strg01`检查独立quota、
测实际用量并估计峰值；每次GPU launch前同时live检查gpu01/gpu02，按实时余量使用单节点至多6张真正提高吞吐的
A40。低util、少量显存占用的设备在峰值余量足够时可与ycliu共驻。
