# EMBER Progress

更新时间：2026-08-17。本文只记录当前可执行状态；目标计划见`task_plan.md`，跨实验认知见`findings.md`，历史精确
结果见`docs/research_history.md`。

## Current state

- 持续研究goal为active；当前不使用subagents。
- canonical workspace为`/data1/user/ymdai/projects/EMBER`，主写分支为`codex/bci-continuation`。
- 本轮开始时HEAD与origin均为`5b3ad390e440708934103beead743a378f43217e`；当前主工作树已实现并验证
  fresh-incompatible LMMPC-v2，尚未commit/push，也没有EMBER GPU进程。
- 当前active design为`docs/layer_matched_memory_program_compiler_design.md`。
- LMMPC-v1 macro25/50及其逐接口分析已经终局；旧checkpoint不得resume到75/100。

## LMMPC-v1 terminal result

- macro25 K4 strict=`81/400`、breadth5、per-task=`2/0/32/3/0/39/5/0`。
- macro50 K4 strict=`101/400`、breadth5、per-task=`3/1/48/0/3/46/0/0`。
- 25→50=`68 retained / 33 gained / 13 lost`、churn46、net`+20`；能力仍明显换手。
- macro50相对LPCP143=`83 retained / 18 gained / 60 lost / 239 both-fail`。
- functional/matching从macro25的`.115512/.021139`降到macro50的`.105596/.001378`，但loss下降没有转化为
  Procedure task separation或稳定closed-loop支持。

逐stage cross-task cosine：

```text
                         macro25    macro50
Semantic Core             .92185     .84859
Causal Procedure          .78993     .95031
natural per-video memory  .71089     .73133
Core-fused grid           .49402     .44842
compiled grid             .68476     .75632
final effective BA        .74886     .64307
```

same-task four-K4 final BA一致性很高（约`.974--.998`），所以K聚合并非最早断点；高一致性只是稳定了当前task mean。
BA norm从macro25的`38.39`增至macro50的`61.56`，仍低于LPCP的`108.70`，但写出并非近identity。

## Earliest failure localization

当前`Procedure -> layer/rank memory reader`存在结构旁路：

1. v1对每个时间步产生Procedure query，却只返回`attended[:, -1]`，所以仅`P_last`直接查询memory；
2. reader再独立加入`R_memory[last]-R_memory[first]`，该endpoint完全绕过Procedure；
3. macro25/50中attention/output norm仅`.02753/.02461`，endpoint/attention为`41.04x/45.73x`；
4. 把每个video整条Procedure替换为重复`P_last`，direct H逐元素不变；
5. macro50换成另一task完整Procedure时，direct H平均只改`.125%`，最终BA只改`6.23%`、cosine`.99769`；
6. reverse endpoint逐元素严格为负，解释了旧`correct-reverse`通道近乎完美的反号响应；
7. Procedure cross-task cosine训练到macro50反而升至`.95031`，matching接近0只学会了generic forward sign。

因此不能把81→101归因于逐渐学会高层Procedure；增益主要来自Core、endpoint和compiler。继续旧训练可能继续改善
某些任务，但不针对EMBER所需的有向阶段知识，故不启动macro75/100。

## Active LMMPC-v2 decision

v2保留统一主链：

```text
exact language + dynamic-K ordered action-hidden videos
  -> V6 task-grounded Semantic Core
  -> Action-query Visual-Transition Causal Procedure P[1:T]
  -> 16 one-way layer/rank memory states M[t,l,r]
  -> each fixed (l,r) address reads all Procedure stages
     keys = P[1:T], values = centered dynamic M[1:T,l,r]
  -> address-preserving K-set consensus
  -> dynamic Core fusion
  -> the same 20x16 grid enters axial M2P
  -> jointly trained native rank16 A/B
  -> one complete 38-target LoRA
```

唯一主要修正是Procedure路径合同：删除独立endpoint、内部`0.5*(correct-reverse)`和language/order matching heads。
训练与部署只运行一次正确正序的dense functional B20图；reverse/shuffle只在评测中重排raw frames并完整重前向。
这三个删除属于同一个已证实shortcut，而非另换架构路线。Core、Action/Procedure、memory token、K-set、M2P、factor
heads、rank16和optimizer recipe均保持。

当前主工作树已完成：

- stage-addressed centered-memory reader；若Procedure阶段全部重复，centered Value经均匀attention相消；
- training/deployment统一为一次正序图，删除matching参数与三臂重复计算；
- config/checkpoint/eval/launch family升级到fresh-incompatible v2，formal状态回到pending profile；
- 全量CPU=`284 passed`，compile、diff-check和architecture guard无hard violation；
- world5 worktree profile两macro均严格K1--K4各6 tasks，macro=`39.02/36.05s`，functional=
  `.156120→.153996`，peak allocated/reserved=`31.80/34.20GB`，无OOM/nonfinite；
- 真实task38/K4/323-frame机制探针中，重复`P_last`后的per-video memory仅为正常norm的`.01390`，relative-L2=
  `.99998`；zero Procedure得到相同消失结果，证明完整stage轴是必要路径；
- reverse相对correct的parameter-memory relative-L2=`.87522`且negative-relative-L2=`1.55334`，不再是硬反号；
  shuffle在最终BA上仍只改`.08864`，这是训练前待学习性质，不算视频因果门通过；
- constant memory/template严格为0，K置换LoRA max-abs严格为0，8个factor family及reader query/key/address/value均有
  非零梯度，source policy非零gradient tensor数为0；
- 真实调度最大task38/K4/371-frame及随后四任务完整结束，peak allocated/reserved=
  `41,987,227,136 / 44,912,607,232` bytes，无OOM/nonfinite。

上述GPU证据来自当前worktree，只用于在提交前否决结构/显存错误；formal profile仍须从clean pushed detached commit
重跑并写回seal evidence，尚未启动formal训练。

## Immediate next work

1. 审查task-scoped diff并提交/push当前pending-profile实现；
2. 从该clean commit重跑world5两macro与371-frame条件，写回formal seal evidence，再次commit/push；
3. 从最终clean pushed detached commit fresh train24到macro25/50有信息量节点；
4. strict paired400报告逐task/suite、breadth、retained/gained/lost/churn，并沿Core→Procedure→H→K-set→M2P→BA→
   action定位下一个接口；不在性能峰值和趋势未观察清楚时过早终止。

## Fixed scientific baselines

- v6-fast五臂：`143/135/125/128/129`；
- LPCP K4 correct：`143/400`、breadth7；
- SFMC单点：`144/400`但lost15/churn31且无六臂；
- GOMQ：`151→135→131`，证明learned memory有真实增益但旧rank32 direct-B/shared reward不稳定；
- LMMPC-v1：`81→101`，证明fresh memory-grid Writer可学习，但当前Procedure reader被endpoint旁路。

## Storage and runtime note

历史整理时`/data1`个人用量约542.2 GiB / 1 TiB，该值会漂移。任何formal launch前重新在`strg01`检查独立quota、
测实际用量并估计峰值；每次GPU launch前同时live检查gpu01/gpu02，按实时余量使用单节点至多6张真正提高吞吐的
A40。低util、少量显存占用的设备在峰值余量足够时可与ycliu共驻。
