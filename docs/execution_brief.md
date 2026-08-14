# EMBER Execution Brief

更新时间：2026-08-14。本文只定义当前实验与持续迭代的执行语义；实时进度见`active_session_handoff.md`，稳定
owner原则见`current_owner_requirements.md`，历史负结果见`research_history.md`。

## 1. Latest completed experiment and active method

V6 Semantic-Core Set Bridge已完成macro25 K4 strict paired correct400：`135/400`、breadth7，per-task=
`1/2/46/30/0/35/20/1`、per-suite=`3/76/35/21`。相对matched K4 Shared-Core139为
`120 retained / 15 gained / 19 lost`、net`-4`、churn34、suite net=`-2/-4/-1/+3`；Long2从0到1但Goal3仍0。
相对old134净`+1`、v6-fast143净`-8`。400 LoRAs、72 jobs、18 workers均完整exit0；按`<140`门终局non-pass。

关键接口分析否定了“只要把trainable set前移到Semantic Core就能学习共有内容”：trained output归零只改变
`.001763` effective BA，task-mean`.001472`；原始Core correction相对完整Core仅`1.8275e-5`，K4 attention
entropy/log4=`.999885`。native compiler实际把微小Core差放大为BA churn；最早失效发生在set内部，因为当前Value=
`sum alpha(C_k-mean C)`，attention近均匀时按构造相消。无参数shared Core union仍贡献K1→K4约`.039675` BA变化。

当前active方法是V6 Semantic-Core Common-Value Set Bridge：位置、参数量、底座、rank16、B20、动态K与后端均
不变，只把Value改为`sum alpha C_k`，让跨video共有Semantic Core本身可训练；K1显式旁路保持任意参数下native
v6恒等。canonical实现已原位切换fresh schema；正式环境full CPU=`374 passed`。gpu01 world6 full24 B20
profile已完成：macro1/2=`25.930/22.530s`、K各6、最长323帧无截断、reserved`40.758GB`、0 OOM/nonfinite；
gradient norm=`.002698/.002795`，较centered路径约`.00000325`打开约三阶，macro1→2 q/k均非零更新。formal
config已seal，当前立即fresh macro0→25。完整authority见
`action_forecast_writer_v6_semantic_core_common_value_set_bridge_design.md`。

## 2. Single changed variable and training semantics

- v6的language-conditioned evidence、Semantic Core、有向Procedure、native compiler remainder、rank16 topology
  和factor heads全部加载macro400并冻结；
- 当前唯一变量是Semantic-Core set的Value由centered residual改为raw common Core；不改位置、参数预算、
  memory、rank、negative、expert、reward或LoRA mapper；
- 24 train tasks构成一个完整macro，task内B20 mean后24-task等权；
- 每macro K1/K2/K3/K4各6，各task每四个macro覆盖全部K；
- K条video同task、action-hidden、互不重复且与action episodes错开，每条video保留stride-5完整序列；
- source policy与v6底座trainable参数为0；K1严格保留、K2--K4提供Common-Value set functional gradient；
- profile只裁决真实wall/显存/batch，训练loss不选择checkpoint。

## 3. Closed-loop adjudication

Common-Value的K1/step0/set/order/gradient门和full24 profile已通过；下一步fresh macro0→25并做K4 strict paired400。
K4若低于140或breadth低于7即终止；140..150只有相对matched139净增、至少3 suites不下降并解锁Goal3/Long2才
resume；超过150后补K1--K4 scaling及correct/same/wrong/shuffled/reversed/no-video controls。

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
