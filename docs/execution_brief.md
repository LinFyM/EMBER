# EMBER Execution Brief

更新时间：2026-08-15。本文只定义当前实验与持续迭代的执行语义；实时进度见`active_session_handoff.md`，稳定
owner原则见`current_owner_requirements.md`，历史负结果见`research_history.md`。

## 1. Latest completed experiment and next decision boundary

V6-LPCP PCSD cycle1已完成full24 paired reward training与K4 strict paired correct400。closed-loop=
`135/400`、breadth6、per-task=`0/4/48/32/0/35/15/1`、per-suite=`4/80/35/16`、top3=
`115/135=.85185`。相对直接底座LPCP143严格=`121 retained / 14 gained / 22 lost / 243 both-fail`、
churn36、net`-8`、Jaccard`.77070`；相对AS139严格=`115/20/24/241`、churn44、net`-4`。count-only
相对v6-fast143/old134/compiler138/online128=`-8/+1/-3/+7`。correct、breadth、support retention与净增四个
预注册门全部失败，PCSD终局；不做cycle2、controls或参数小扫。

失败不是reward没有信息，也不是LoRA/action链路断开。train24的48 paired states产生9条discordant成功轨迹，
candidate/reference gains=`5/4`；positive CFM给出非零gradient、query-delta更新与fixed-action response RMS
`.00215--.00407`。但全400 PCSD相对LPCP的effective-BA relative-L2 mean/median只有
`.0006834/.0006767`，比LPCP相对AS139的`.002653`再小约3.9倍；gained/lost改写mean=
`.0006873/.0006830`，幅度不能选择方向。

更早且可解释的断点来自FP64 first4：同一validation task的四个不同K4 correct video sets所产生的PCSD增量，
pairwise cosine跨8 tasks平均`-.00187`，task-mean/sample energy ratio=`.24860`，几乎正好是四个正交修正
平均后的`1/4`。因此最早失效接口是**稀疏paired reward trajectories经一个shared query commitment后仍成为
video-set-specific局部方向，未被合并成跨video、跨task可保留的高层程序**。下一设计只改变reward credit如何
跨same-task video sets形成共同方向；不重做已通过的LPCP carrier，也不预设literal memory或rank变化为答案。

当前active design是CV-CSD：每个active task仍只用anchor K4产生paired唯一成功trajectory，但让同一trajectory
分别监督四个互不重叠的same-task correct K4 conditions。每个view独立完成Writer→完整LoRA→selected-success
functional CFM，复用相同replay/time/noise，只在共享`query_delta.weight`梯度处等权汇合。deployment、rank16、
AS139 tail、optimizer、rollout数量和信息墙全部不变；不平均输入或LoRA，不混入memory、negative或rank变量。
精确合同见`docs/action_forecast_writer_v6_lpcp_cross_video_causal_success_distillation_design.md`。

canonical实现已通过task4 live smoke：4 rollouts不变，4个K4 credit views共16 unique demos，四view LoRA/query
credit与最终BA/action response均非零；cycle=`145.526s`、peak reserved=`40.752GB`，无OOM/nonfinite/禁读。
formal config已seal；下一动作是clean pushed commit上的fresh full24 cycle1，不resume PCSD或smoke。

## 2. Completed PCSD variable and training semantics

- 冻结完整LPCP carrier、AS139 Semantic Core/Procedure/K-set/fusion/compiler与38-target rank16 FactorHeads；
- 同一K4 context只编码一次，reference把`query_delta`精确置零，candidate使用当前LPCP query；
- train24每task两个同初态/同policy RNG arms，只对唯一成功arm的executed trajectory做positive CFM，ties为零；
- 只训练65,536参数`query_delta.weight`，source policy、reader/controller与全部LoRA tail参数trainable为0；
- Writer仍由exact language和action-hidden videos一次生成完整LoRA，rollout期间不反复观看视频。

## 3. Closed-loop adjudication

Ordered-Procedure AS139、raw reward138、ADSP138、V6-LPCP AS与PCSD均已终局且不得resume或小扫。PCSD训练root、
strict root、逐episode transitions、全400 BA与FP64跨video evidence均已封存。`>150`仍是更高性能追求；约145
也可成为有效结果，但必须由相邻single checkpoints低churn、same-task-other鲁棒和correct相对wrong/shuffled/
reversed/no-video的明确优势共同认证。单点145或151都不算完成。

报告aggregate、8项per-task、4 suite totals、breadth、retained/gained/lost、top-task concentration和K1→K4
success-set变化。不能用K1/K4 union、LoRA norm或functional loss冒充同一condition的能力。

CV-CSD cycle1只有在correct`>=144`、breadth`>=7`、相对LPCP lost`<=10`、gained>lost且至少3 suites不降时才
进入cycle2。稳定资格要求两个相邻single checkpoints均`>=144`、均值`>=145`、breadth均`>=7`、相邻churn
`<=20`、Jaccard`>=.85`；随后才做same/wrong/shuffled/reversed/no-video，要求same/correct至少`.9`且correct
对每个negative/no-video至少高8。单点145或151都不能跳过这些门。

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
