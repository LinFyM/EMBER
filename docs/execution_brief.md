# EMBER Execution Brief

更新时间：2026-08-15。本文只定义当前实验与持续迭代的执行语义；实时进度见`active_session_handoff.md`，稳定
owner原则见`current_owner_requirements.md`，历史负结果见`research_history.md`。

## 1. Latest completed experiment and next decision boundary

最新完成实验是**V6-LPCP Native Probe-Value Commitment**（NPVC），authority=
`docs/action_forecast_writer_v6_lpcp_native_probe_value_commitment_design.md`。clean pushed/frozen `606ac76`从sealed
LPCP fresh完成full24 cycle1：24 tasks/48 paired states/96 rollouts，candidate/reference=`33/32`、gains=`5/4`，
9 active tasks覆盖四suite，36 credit conditions/144 unique videos，cycle=`584.053s`；完整world5 checkpoint、
completion与0禁读/OOM/nonfinite均保留。

同一cycle1 checkpoint的K4 strict paired correct400=`136/400`、breadth6、per-task=
`1/2/48/33/0/34/18/0`、per-suite=`3/81/34/18`。相对LPCP143严格=`120 retained / 16 gained / 23 lost /
241 both-fail`、churn39、net`-7`、Jaccard`.754717`；相对GOSC141/SFMC144/CCT142净为`-5/-8/-6`，count-only
相对v6-fast143/old134/compiler138/online128=`-7/+2/-2/+8`。correct≥140、breadth≥7、lost≤15三门失败，按
authority终局：不resume cycle2，不补same/wrong/shuffled/reversed/no-video，不做scale/rank/axis/LR/seed小扫。

本轮真正推进了因果定位。稳定FP64 all400显示NPVC相对LPCP effective-BA relative-L2 mean/median=
`.0004683/.0003708`、absolute L2 mean=`.05234`，q/v/action均为原生尺度非零。post-train validation8每task四个
不同correct K4的pure-NPVC cosine/energy平均=`.40870/.54227`、7/8 tasks过原几何门；natural→reversed的
probe/BA relative-L2=`1.84084/1.60518`。因此CCT的held compiler消失已被关闭，video evidence、顺序与
policy-effective写出均工作。

但写入内容与reward不对齐：相对LPCP的retained/gained/lost/retained-failure BA relative-L2 mean=
`.000320/.000412/.000436/.000549`，lost略大于gained，持续失败样本最大。full24后train task4的four-view
cosine/energy又从preformal`.5929/.6792`降到`.0569/.2951`。所以最早失败接口是**selected-success credit如何
选择native Value中改善held on-policy occupancy的组件和符号，以及异质task方向如何在一个full24 Writer中
共存**，而不是carrier、LoRA幅度、量化或单纯跨视频coherence。

下一design authority尚未建立。它必须只针对reward-useful direction/content selection与多task共存，不能只
加scale、capacity、coherence或support-only guard。memory token、rank8、dynamic K与其它可扩展hypernetwork
形式仍开放；只有当它们提供layer-aligned、可被reward选择且可跨task共存的Value方向时才构成因果干预。
稳定约145资格高于单点分数：至少需要相邻single checkpoints低churn/high-overlap、same-task-other鲁棒，并在
同一final checkpoint上证明correct相对wrong/shuffled/reversed/no-video的明确paired优势。

NPVC训练root=
`runs/outputs/pi05_v6_lpcp_native_probe_value_commitment_formal_cycle0to1_r5_k4_views4_nmc4_b8_606ac76_gpu01_20260815`；
strict与终局分析root=
`runs/outputs/pi05_v6_lpcp_native_probe_value_commitment_cycle1_k4_correct400_noreplacement_seed7_trainr5_evalr5_606ac76_gpu01_20260815`。

## 2. Completed CV-CSD variable and exact conclusion

- 部署继续是exact language + dynamic K ordered action-hidden videos一次生成一套38-target rank16 LoRA；
- anchor K4只产生AS139/LPCP paired唯一成功trajectory；同一trajectory在4个disjoint same-task correct K4
  conditions下分别通过完整Writer→LoRA→policy CFM；
- 四view只在65,536参数`query_delta.weight`梯度处等权汇合，不平均video、Program或LoRA；
- 只训练query commitment，LPCP carrier、AS139 tail、source policy、optimizer与rollout数不变；
- 结果只否定“query-only map + four-view exact selected-success mean”，不否定few-shot、memory、reward或生成LoRA。

## 3. Closed-loop adjudication

Ordered-Procedure AS139、raw reward138、ADSP138、V6-LPCP、PCSD、CV-CSD、SFMC、Gradient-Open与CCT均已终局且
不得resume或小扫。
各自训练root、strict root、逐episode transitions和必要BA/跨video evidence均已封存。`>150`仍是更高性能追求；约145
也可成为有效结果，但必须由相邻single checkpoints低churn、same-task-other鲁棒和correct相对wrong/shuffled/
reversed/no-video的明确优势共同认证。单点145或151都不算完成。

报告aggregate、8项per-task、4 suite totals、breadth、retained/gained/lost、top-task concentration和K1→K4
success-set变化。不能用K1/K4 union、LoRA norm或functional loss冒充同一condition的能力。

SFMC与CCT均因cycle1预注册门失败而没有相邻checkpoint与controls；这不是缺失分析，而是预注册停止规则。下一架构仍须先以
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
