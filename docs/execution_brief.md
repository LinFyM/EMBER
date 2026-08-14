# EMBER Execution Brief

更新时间：2026-08-15。本文只定义当前实验与持续迭代的执行语义；实时进度见`active_session_handoff.md`，稳定
owner原则见`current_owner_requirements.md`，历史负结果见`research_history.md`。

## 1. Latest completed experiment and next decision boundary

最新完成实验是**V6-LPCP Cross-Video Causal Success Distillation**（CV-CSD）。full24 cycle1与K4 strict paired
correct400均来自clean frozen `c1d8952`：closed-loop=`134/400`、breadth7、per-task=
`1/2/47/32/0/36/15/1`、per-suite=`3/79/36/16`、top3=`115/134=.85821`。

相对LPCP143严格=`122 retained / 12 gained / 21 lost / 245 both-fail`、churn33、net`-9`、
Jaccard`.78710`，四个suite全部下降；相对AS139严格=`121/13/18/248`、churn31、net`-5`；相对PCSD135
严格=`115/19/20/246`、churn39、net`-1`。count-only相对v6-fast143/old134/compiler138/online128=
`-9/0/-4/+6`。五项cycle1门中只有breadth通过，因此CV-CSD终局，不做cycle2、controls或小参数扫。

训练合同本身完整且高效：24 tasks/48 paired states/96 rollouts仍产生与PCSD完全相同的33/34两臂成功、5/4
单臂成功和9个active tasks；36个credit conditions的LoRA/query gradients全部finite/nonzero。cycle wall=
`863.432s`，仅为PCSD的`1.0307x`；3 ranks各8 tasks/3 active tasks，记录负载max/min=`1.0828x`。

决定性non-pass不是134这个单点，而是部署方向分析。全400 CV-CSD相对LPCP的effective-BA relative-L2
mean/median=`.00068370/.00067774`，gained/lost均约`.000679`。FP64 first4中，同task四个K4 correct conditions
的CV-CSD增量pairwise cosine平均=`.000205`、task-mean/sample energy=`.250155`；相对PCSD也为
`-.001908/.248578`。即使同一真实成功trajectory在四个正确视频条件下各自完整反传，shared query-gradient mean
仍经video-specific Jacobian变成近正交局部BA修正。

最早失效接口因此是**Program/evidence到policy topology的query-only commitment**，不是LPCP没有读视频、reward
没有内容、LoRA坍缩或多卡负载错误。当前没有active successor；下一authority要在保留V6/LPCP强absolute carrier、
rank16完整LoRA和cross-video成功信用的前提下，只改变layer/rank/target-aligned commitment接口。memory token是有
证据触发的候选机制，但不是项目目标，也不能只是替换已通过的carrier。

训练root=
`runs/outputs/pi05_v6_lpcp_cross_video_causal_success_distillation_formal_cycle0to1_r3_k4_views4_nmc4_b8_c1d8952_gpu01_20260815`；
strict与全部paired/BA/FP64分析root=
`runs/outputs/pi05_v6_lpcp_cross_video_causal_success_distillation_cycle1_k4_correct400_noreplacement_seed7_trainr3_evalr3_c1d8952_gpu01_20260815`。

## 2. Completed CV-CSD variable and exact conclusion

- 部署继续是exact language + dynamic K ordered action-hidden videos一次生成一套38-target rank16 LoRA；
- anchor K4只产生AS139/LPCP paired唯一成功trajectory；同一trajectory在4个disjoint same-task correct K4
  conditions下分别通过完整Writer→LoRA→policy CFM；
- 四view只在65,536参数`query_delta.weight`梯度处等权汇合，不平均video、Program或LoRA；
- 只训练query commitment，LPCP carrier、AS139 tail、source policy、optimizer与rollout数不变；
- 结果只否定“query-only map + four-view exact selected-success mean”，不否定few-shot、memory、reward或生成LoRA。

## 3. Closed-loop adjudication

Ordered-Procedure AS139、raw reward138、ADSP138、V6-LPCP、PCSD与CV-CSD均已终局且不得resume或小扫。CV-CSD
训练root、strict root、逐episode transitions、全400 BA与FP64跨video evidence均已封存。`>150`仍是更高性能追求；约145
也可成为有效结果，但必须由相邻single checkpoints低churn、same-task-other鲁棒和correct相对wrong/shuffled/
reversed/no-video的明确优势共同认证。单点145或151都不算完成。

报告aggregate、8项per-task、4 suite totals、breadth、retained/gained/lost、top-task concentration和K1→K4
success-set变化。不能用K1/K4 union、LoRA norm或functional loss冒充同一condition的能力。

CV-CSD因cycle1未过门而没有相邻checkpoint与controls；这不是缺失分析，而是预注册停止规则。下一架构仍须先以
cycle1 absolute/breadth/retention筛选；过门后必须评相邻checkpoint，再做same/wrong/shuffled/reversed/no-video。
单点145或151都不能跳过稳定性与视频因果资格。

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
