# EMBER Execution Brief

更新时间：2026-08-15。本文只定义当前实验与持续迭代的执行语义；实时进度见`active_session_handoff.md`，稳定
owner原则见`current_owner_requirements.md`，历史负结果见`research_history.md`。

## 1. Latest completed experiment and next decision boundary

最新完成实验是**V6-LPCP Gradient-Open Semantic Commitment**。同一clean `eb543d3` cycle1 checkpoint的
K4 strict paired correct400=`141/400`、breadth7、per-task=`1/3/48/29/0/36/23/1`、per-suite=
`4/77/36/24`。相对LPCP143严格=`128 retained / 13 gained / 15 lost / 244 both-fail`、churn28、net`-2`、
Jaccard`.82051`；suite net=`-1/-6/-2/+7`。相对SFMC144=`124/17/20`、churn37。预注册的absolute、lost、
net和suite四项门失败，因此本架构终局，不续cycle2或六臂controls。

稳定FP64证明gradient-open解决了真实但非充分的接口：相对LPCP all400 effective-BA relative-L2 mean=
`9.6632e-6`，约为SFMC的`33.3x`，q/v/action非零样本=`400/399/368`；但gained/lost幅度仍不可分，first4
同task四个disjoint correct K4增量cosine=`.0001442`、mean/sample energy=`.250124`。所以最早失败接口已从
SFMC的sub-ULP writeout后移到**shared semantic address/cross-video success credit经video-conditioned
Jacobian仍写成近正交policy directions**。下一变量必须直接形成跨video可复现的causal task Program，不能继续
放大anchor、增加cycle或扫LR/rank/scale。

当前active successor是**V6-LPCP Causal Coefficient Transport**，authority=
`docs/action_forecast_writer_v6_lpcp_causal_coefficient_transport_design.md`。唯一主变量是把K-set后的256维
video `factor_memory`从hidden Value direction改为每个320 policy/rank slot的两个causal coefficients；同task
exact language经冻结V6 W1/GELU定义共享的family-specific output axes。它保持step0 exact LPCP、四view
selected-success和完整rank16 compiler，不加入coherence loss、memory token、rank变化或第二数据源。当前尚未
实现或启动GPU；先过跨video span、顺序、q/v/action response与吞吐机制门，再决定是否fresh full24。

直接前序**V6-LPCP Semantic Factor-Memory Commitment**（SFMC）的full24 cycle1与K4 strict paired
correct400均来自clean frozen `8994180`。closed-loop=`144/400`、breadth7、per-task=
`1/3/47/36/0/38/18/1`、per-suite=`4/83/38/19`、top3=`121/144=.84028`。

相对LPCP143严格=`128 retained / 16 gained / 15 lost / 241 both-fail`、churn31、net`+1`、Jaccard
`.805031`、McNemar `p=1`；suite net=`-1/0/0/+2`。相对AS139/PCSD135/CV-CSD134分别为
`121/23/18`、`119/25/16`、`124/20/10`，count-only相对v6-fast143/old134/compiler138/online128=
`+1/+10/+6/+16`。它恢复了CV-CSD丢失的absolute，但没有减少能力换手；预注册门只有LPCP lost≤10失败，
因此不续cycle2或六臂controls。

训练工程合同完整：24 tasks/48 paired states/96 rollouts、34/34两臂成功、8个active tasks、32个credit
conditions与128个unique videos；8/8 family maps均更新。cycle wall=`920.555s`=`1.0662x` CV-CSD，三rank
任务数=`8/9/7`、记录时长max/min=`1.0653x`，没有rank分配或吞吐问题。semantic query/basis-key delta仅约
`1.7e-9`，zero-init staging在cycle1主要只打开family maps。

稳定FP64差分把失效接口进一步推进：相对LPCP的effective-BA relative-L2 mean/median=
`2.899e-7/1.066e-9`，255/400样本有任何非零变化；其中q为249、v为16、action仅1。first4修正
pairwise cosine=`-8.10e-6`、mean/sample energy=`.249995`，没有形成跨video共同方向；SFMC相对CV-CSD的
`.000675/.000669`和`.000205/.250154`则几乎复现CV-CSD→LPCP距离，说明candidate在部署上基本回到LPCP。
最早失败接口是**continuous factor-memory residual经冻结W2写成native public LoRA时被压到稀疏q-family ULP
crossing**，且learned semantic router尚未形成，而不是memory未算、reward无梯度、GPU负载或LoRA链路未接通。

随后检验的successor是**V6-LPCP Gradient-Open Semantic Commitment**，authority=
`docs/action_forecast_writer_v6_lpcp_gradient_open_semantic_commitment_design.md`。它不续SFMC cycle1，只从sealed
LPCP macro25 fresh初始化，在同一factor commitment中加入冻结V6-W1 policy-aligned anchors，使step0仍严格
identity，同时让family delta maps和semantic query在第一次selected-success update中都有梯度。carrier、K4
four-view credit、rank16、optimizer与信息墙不变。clean pushed `5b14c89` task4真实smoke中8/8 maps更新，
semantic query delta=`1.1979e-4`（SFMC为`1.7564e-9`），q/v/action effective-BA response=
`6.6169e-7/9.1517e-7/4.8908e-8`，总BA=`6.9391e-7`（SFMC的`19.7x`），fixed-action=`.0027033`；
cycle=`132.458s`=`.9501x` SFMC。该机制/效率证据只授权full24，并不提供性能结论。
SFMC因cycle1未过稳定门而没有启动same/wrong/shuffled/reversed/no-video，因此不能宣称视频鲁棒性或特异性。

fresh full24 cycle1已由clean detached `eb543d3`在gpu01 world5完整exit0：24 tasks/48 pairs/96 rollouts，
candidate/reference=`33/31`、gains=`6/4`、10 active tasks覆盖四suite、40 credit views/160 unique videos。
semantic query delta=`6.9499e-5`，为SFMC约3.96万倍；5/5 probes的q/v、3/5的action native BA非零，说明
router与v-family写出在full24仍打开，但action写出尚不均匀。cycle=`581.924s`；rank任务数虽为`3/5/2/5/9`，
recorded wall只有`462.083--560.082s`、max/min=`1.2121x`，动态队列按成本而非数量平衡。world5相对SFMC
world3 wall=`.6321x`，约95%理想扩展效率；完整checkpoint/completion已保留。训练outcome跨world不作严格架构
比较。其strict终局结果与接口定位见本节开头；当前没有可resume的旧checkpoint，CCT必须fresh。

训练root=
`runs/outputs/pi05_v6_lpcp_semantic_factor_memory_commitment_formal_cycle0to1_r3_k4_views4_nmc4_b8_8994180_gpu01_20260815`；
strict与终局分析root=
`runs/outputs/pi05_v6_lpcp_semantic_factor_memory_commitment_cycle1_k4_correct400_noreplacement_seed7_trainr3_evalr3_8994180_gpu01_20260815`。

Gradient-Open训练root=
`runs/outputs/pi05_v6_lpcp_gradient_open_semantic_commitment_formal_cycle0to1_r5_k4_views4_nmc4_b8_eb543d3_gpu01_20260815`；
strict与终局分析root=
`runs/outputs/pi05_v6_lpcp_gradient_open_semantic_commitment_cycle1_k4_correct400_noreplacement_seed7_trainr5_evalr5_eb543d3_gpu01_20260815`。

## 2. Completed CV-CSD variable and exact conclusion

- 部署继续是exact language + dynamic K ordered action-hidden videos一次生成一套38-target rank16 LoRA；
- anchor K4只产生AS139/LPCP paired唯一成功trajectory；同一trajectory在4个disjoint same-task correct K4
  conditions下分别通过完整Writer→LoRA→policy CFM；
- 四view只在65,536参数`query_delta.weight`梯度处等权汇合，不平均video、Program或LoRA；
- 只训练query commitment，LPCP carrier、AS139 tail、source policy、optimizer与rollout数不变；
- 结果只否定“query-only map + four-view exact selected-success mean”，不否定few-shot、memory、reward或生成LoRA。

## 3. Closed-loop adjudication

Ordered-Procedure AS139、raw reward138、ADSP138、V6-LPCP、PCSD、CV-CSD、SFMC与Gradient-Open均已终局且
不得resume或小扫。
各自训练root、strict root、逐episode transitions和必要BA/跨video evidence均已封存。`>150`仍是更高性能追求；约145
也可成为有效结果，但必须由相邻single checkpoints低churn、same-task-other鲁棒和correct相对wrong/shuffled/
reversed/no-video的明确优势共同认证。单点145或151都不算完成。

报告aggregate、8项per-task、4 suite totals、breadth、retained/gained/lost、top-task concentration和K1→K4
success-set变化。不能用K1/K4 union、LoRA norm或functional loss冒充同一condition的能力。

SFMC因cycle1 retention未过门而没有相邻checkpoint与controls；这不是缺失分析，而是预注册停止规则。下一架构仍须先以
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
