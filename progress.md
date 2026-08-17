# EMBER Progress

更新时间：2026-08-17。本文只记录当前可执行状态；目标计划见`task_plan.md`，跨实验认知见`findings.md`，历史精确
结果见`docs/research_history.md`。

## Current state

- 当前goal为active：推进LMMPC从设计、实现、充分fresh训练和strict评测到局部迭代与终局裁决。
- canonical workspace为`/data1/user/ymdai/projects/EMBER`，主写分支为`codex/bci-continuation`。
- 当前主分支与origin均包含`dd81b94`；失败formal训练冻结于clean pushed `de0b298`，micro5 profile来自`dd81b94`。
- 当前active design为`docs/layer_matched_memory_program_compiler_design.md`，已升级为implementation authority。
- 当前phase为fresh formal macro0→25；clean `ca40d88`已在gpu02物理`1/2/3/4/7`以world5启动，失败run无状态复用。
- 当前协作边界：不使用subagents。

## Active architecture decision

当前统一主链为：

```text
exact language + ordered action-hidden videos
  -> V6 task-grounded Semantic Core
  -> Action-query Visual-Transition Causal Procedure
  -> Procedure reads per-frame layer/rank one-way memory
  -> frame-to-video reduction preserving layer/rank
  -> permutation-invariant K-set consensus preserving layer/rank
  -> dynamic-Procedure-gated Core fusion
  -> the same 20x16 memory grid enters group/rank axial M2P
  -> jointly trained native rank16 A/B
  -> one complete 38-target LoRA
```

本轮讨论纠正了旧提案中的两个接口：

1. memory states已经携带layer/rank地址，不再创建独立320 routing slots；`20×16=320`只描述最终memory-grid cells；
2. Action state不与memory任意合成Query。Action沿V6已验证路径解释visual transition形成Procedure，Procedure再读取
   memory时间序列，把高层过程落到参数坐标。

Core保留V6的强对象/关系/目标语义，但只能由非零动态memory Program打开；language/static Core不能独立写LoRA。
正式训练必须fresh，旧V6/LPCP只允许短暂机制bring-up。

## Implementation and acceptance evidence

- terminal reward/GOMQ executable runtime已经退役；历史设计、结果与Git证据保留。当前只有一个canonical LMMPC
  Writer入口、config、checkpoint和eval schema。
- LMMPC封存时全量CPU=`282 passed`；补回已有generation profile模块的canonical evaluator入口后全量CPU=`283 passed`，
  `compileall`和`git diff --check`通过；architecture guard为`REVIEW`、无hard blocker，
  active-source净减少约4468行。
- 真实worktree profile首先在functional microbatch8复现空A40峰值OOM；这是运行时显存切片问题，未发生optimizer
  update，不构成架构负结果。
- microbatch4完成一个full24 macro，`36.69s`、rank0 peak allocated=`33.31GB`；随后按既定“matching标量约为
  functional 10%”一次校准到weight=`0.04`，并将microbatch提高到6。
- 最终acceptance profile用gpu01物理`0/1/2/4/5/6`、world6完成2 macros：`31.68/34.62s`，peak
  allocated=`40.04GB`，每轮K1/K2/K3/K4严格各6 tasks；functional均值`.15609→.15395`，Program matching
  `.36196→.30070`，gradient norm`.16124→.16898`，无OOM/nonfinite。
- macro1到2的checkpoint delta覆盖semantic encoder/Core、visual transition、Procedure、layer/rank memory reader、
  K-set、Core/M2P compiler及factor heads；template保持冻结，排除了“只有匹配头更新”的接线假象。
- clean `bd2ee35`机制探针随后发现variable tail microbatch把正常BF16 batch-shape差异放大为伪Program：constant输入的
  raw layer-memory max差为`4.0`，最终effective-BA L2为`.5097`，K置换LoRA max差为`6.37e-4`。最早断点明确在
  native frame encoder的尾batch shape，不在Core/Procedure语义或M2P拓扑。
- 唯一局部修正是用会被切掉的zero rows把尾microbatch补到固定32；有效frames仍各forward一次。用同一macro2
  checkpoint复测最长K4后，constant从raw memory到effective BA全链精确为0，K置换76 tensors最大差精确为0，
  同时correct effective-BA L2=`.56955`、correct/reverse memory relative-L2=`1.99998`，说明只移除了伪方向。
- 修正后的clean pushed `4b6316a`已重新fresh完成world6两macro：`32.97/29.83s`，peak allocated=`40.52GB`，
  functional`.15609→.15395`、Program matching`.36194→.30113`，无OOM/nonfinite。该fresh macro2的最长K4机制
  复测仍为constant全链0、K置换0、correct BA L2=`.56423`、correct/reverse=`1.99998`，formal recipe因此封存。
- clean `de0b298` world5 formal随后完成macro1--16，functional `.15609→.11888`、Program matching
  `.36194→.17621`，说明尚无训练峰值或科学non-pass；macro17时四rank进入flat-gradient all-reduce而rank2未进入。
  精确重放确认rank2首个task38/K4/359帧在microbatch6下单卡OOM，仅差约96 MiB，故NCCL timeout是下游表象。
- 唯一recipe修正是functional microbatch `6→5`：B20仍恰为4次policy forward，且不改Writer、数据、K、loss或任何
  视频帧。它已在同一物理A40完整通过原故障rank2五任务序列；进一步扫描100 macros后又通过真实schedule最大
  task38/K4/371帧，峰值reserved=`45,283,803,136` bytes。formal当时退回pending，等待clean full24 profile。
- clean pushed `dd81b94`随后在gpu02物理`1/2/3/4/7`完成world5两macro profile：`39.22/36.22s`，functional
  `.15612→.15399`、Program matching `.36194→.30113`，K1--K4各6 tasks，无OOM/nonfinite，micro5正式recipe已
  重新sealed。部署生成还移除了matching-only shuffle分支，测试证明primary 76 tensors逐元素不变。

## Immediate next work

1. 完成`pi05_lmmpc_formal_fresh_micro5_r5_b20_ca40d88_gpu02p12347_20260817`到macro25并核验完整world/topology checkpoint；
2. 用独立空闲A40完成K4 generation profile，封存真实batch后启动首个strict paired400与完整逐接口分析；
3. 若macro25仍在共同上升，exact-resume到macro50及相邻checkpoint，不在未见峰值时过早判死；
4. 只在结果定位出明确断点后于LMMPC主链内做局部迭代。

## Training and decision boundary

- fresh train24真实覆盖K1--K4、task等权、video/action同task跨episode；首轮使用dense functional B20和轻量
  language/directed-Program matching，不混入reward。
- 首个strict节点为macro25或等价完整task exposure，但若stage与closed-loop仍共同上升，继续训练到相邻有意义节点，
  不因尚未观察到峰值而终止。
- 首次约145且retention合理立即补六臂并继续相邻checkpoint；稳定约145且视频因果合格也可构成有价值结果。
- 每轮报告per-task/per-suite/breadth/retained/gained/lost/churn/Jaccard，并定位最早失效接口。
- 局部迭代保持LMMPC主链，不做rank/scale/seed小扫、不挑checkpoint、不融合多LoRA。

## Fixed scientific baselines

- v6-fast五臂历史最好：`143/135/125/128/129`；
- LPCP K4 correct：`143/400`、breadth7；
- SFMC单点：`144/400`但lost15/churn31且无六臂；
- GOMQ：`151→135→131`，证明learned memory有真实增量但当前rank32 direct-B/shared reward不稳定。

## Storage and runtime note

整理时观测`/data1`个人用量约542.2 GiB / 1 TiB，但该值会漂移。任何formal training前必须重新在`strg01`检查
独立quota、测实际用量并估计checkpoint/log峰值；GPU launch前同时live检查gpu01/gpu02，按实时余量使用单节点至多
6张真正提高吞吐的A40，可在低util且显存余量足够时与ycliu共驻。
