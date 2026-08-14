# EMBER Execution Brief

更新时间：2026-08-14。本文只定义当前实验与持续迭代的执行语义；实时进度见`active_session_handoff.md`，稳定
owner原则见`current_owner_requirements.md`，历史负结果见`research_history.md`。

## 1. Latest completed experiment and active successor

V6 Semantic-Core Common-Value Set Bridge已完成macro25 K4 strict paired correct400：`133/400`、breadth6，
per-task=`2/3/48/31/0/35/14/0`、per-suite=`5/79/35/14`、top3=`85.71%`。相对Semantic-Core135严格
配对=`118/15/17` retained/gained/lost、net`-2`；相对Shared-Core139=`119/14/20`、net`-6`；Long suite
相对135净丢7，未形成共同积累。400 LoRAs、72 jobs、400 rows、18 workers全部exit0；按`<140`且breadth`<7`
双门终局，不resume、不补controls、不扫参。

机制假设本身成功接通但没有转成有效policy方向：raw Common-Value把Core correction从centered路径的
`1.8275e-5`打开到`.065856`，current→zero effective-BA从`.001763`打开到`.053648`，task-mean同样
`.053633`；attention entropy/log4仍为`.999885`。所以失败不再是Value相消或compiler衰减，而是offline B20
functional credit把强common-mean修正对到了held on-policy无效且相互换手的方向。补充train-seen output-zero
反事实为trained/zero=`63/59`、paired net`+4`，说明task-local on-policy credit不是完全没有，但未外推到held。

owner已授权继续。active successor为V6 Shared-Core Ordered-Procedure Common-Value：恢复matched139的冻结
shared-Core边界，只把原Procedure-Set的Value由centered residual改成raw common ordered Procedure；trainable
Value因此必须来自有向video过程，不能只由静态语言产生。完整authority见
`action_forecast_writer_v6_shared_core_ordered_procedure_common_value_design.md`。

## 2. Active single changed variable and training semantics

- v6的language-conditioned evidence、Semantic Core、有向Procedure、native compiler remainder、rank16 topology
  和factor heads全部加载macro400并冻结；
- 当前唯一变量相对matched Shared-Core139，是Procedure-Set Value由weighted centered residual改为weighted raw
  ordered Procedure；不改位置、query/key、参数预算、memory、rank、negative、expert、reward或LoRA mapper；
- 24 train tasks构成一个完整macro，task内B20 mean后24-task等权；
- 每macro K1/K2/K3/K4各6，各task每四个macro覆盖全部K；
- K条video同task、action-hidden、互不重复且与action episodes错开，每条video保留stride-5完整序列；
- source policy与v6底座trainable参数为0；K1严格保留、K2--K4提供Procedure Common-Value functional gradient；
- profile只裁决真实wall/显存/batch，训练loss不选择checkpoint。

## 3. Closed-loop adjudication

Common-Value strict133/breadth6已触发终止门。没有macro50 resume，也没有K scaling或correct/same/wrong/
shuffled/reversed/no-video补测；这些controls只在absolute先过门后才有科学价值。下一轮必须先讨论并重新写单变量
authority，不能从本checkpoint续训或用scale/LR/K/seed救援。

报告aggregate、8项per-task、4 suite totals、breadth、retained/gained/lost、top-task concentration和K1→K4
success-set变化。不能用K1/K4 union、LoRA norm或functional loss冒充同一condition的能力。

## 4. Continuous adjudication loop

每轮strict结果完成后，按以下顺序分析：

1. absolute、per-task/per-suite、breadth；
2. 相对最接近方法和历史强基线的retained/gained/lost与能力集中；
3. 若有相邻checkpoint，分析persistent/gained/lost与union gap；
4. 沿`input evidence -> Core/Procedure -> set/compiler -> effective BA -> fixed action -> rollout`定位最早失效接口；
5. 分离科学non-pass与明确工程合同违约；
6. 只对最早接口提出一个主要因果变量，写可证伪authority后实现；
7. 做最小必要CPU/机制验证和吞吐profile，尽快回到真实paired400。

不得用loss、cosine、rank或漂亮五臂margin代替absolute，也不得通过rank/scale/seed/dtype/temperature小扫救一个
失败checkpoint。owner的局部建议不能导致整套已认可方案无证据重写。

## 5. GPU and efficiency

- launch前同时live检查gpu01/gpu02；单节点至多6张，有多少真正合适就用多少；
- 允许在显存峰值余量充足、低util且不明显干扰他人的卡上共驻；不抢占、kill、reset或dummy占卡；
- evaluator当前保守共驻门为util≤10%、已用≤8GiB且剩余≥32GiB；任一越界即拒绝；
- 多卡训练设置`NCCL_P2P_DISABLE=1`、GPU-local NUMA和deferred NCCL；
- fresh可用world1--6；exact resume锁原world topology；
- evaluator用动态cost queue和persistent model/env，不静态拆task；
- 以真实samples/s、LoRA/s、最长视频稳定性和显存峰值选择batch；
- 接受正常BF16/TF32低位差异，不重复forward、固定batch1、扩dtype、逐tensor scan或增加hash。

## 6. Storage, Git and artifacts

- 大run前查询`strg01`上的独立user quota，测canonical root并估计checkpoint/cache/temp峰值；
- formal训练与评测来自clean pushed commit的detached worktree；
- 新run使用fresh output root，不覆盖或部分复用中止/不兼容root；
- formal保留run contract、metrics、macro checkpoints、completion、400 raw rows、aggregate和decision analysis；
- profile/smoke roots只作机制/吞吐证据，不冒充formal；
- meaningful结果更新current state、execution brief、current design、task plan、findings和research history；
- 不把历史命令重新复制进多个文档，精确命令以run contract/invocations为准。

## 7. Collaboration boundary

owner授权在核心目标、信息墙与效率原则内持续自主迭代。当前暂不使用subagents。只有真正需要改变核心目标、
扩大权限、处理破坏性操作或遇到无法从本地证据解决的关键歧义时才向owner停下来请求决定。
