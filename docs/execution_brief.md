# EMBER Execution Brief

更新时间：2026-08-14。本文只定义当前实验与持续迭代的执行语义；实时进度见`active_session_handoff.md`，稳定
owner原则见`current_owner_requirements.md`，历史负结果见`research_history.md`。

## 1. Latest completed experiment and next decision boundary

V6 Actual-Delta Success-Support Projection已完成fresh full24与K4 strict paired correct400：`138/400`、breadth7、
per-task=`3/2/45/30/0/36/21/1`、per-suite=`5/75/36/22`。相对同schedule AS139严格配对=
`116 retained / 22 gained / 23 lost`、churn45；相对raw reward138=`117/21/21`、churn42。按absolute`<144`、
lost`>10`且gained不超过lost三项门终局，不运行cycle2、不补controls、不调约束/scale/LR/rank。

这不是projection没运行：clean `ad2e1be` formal的22条support constraints中raw违反6条，投影激活6项并得到0
final violation；preference descent与delta energy分别保留`.963787/.980958`，BA/action响应非零，wall=
`1033.501s`、0 forbidden read/OOM/nonfinite。strict使用5卡15 persistent workers完成400 rows，wall=
`1306.681s`、exit0。

projection把raw的Long1从15恢复到21，却由Spatial/Object净`-2/-4`支付；AS→ADSP effective-BA relative-L2=
`.002976`，比AS→raw的`.003323`更小，但held churn反而从37升到45。最早失效接口因此不再是“raw shared update
没有任何support约束”，而是**train24成功prefix的一阶局部support不能代表held闭环support，且更近的LoRA几何
不能保证能力共存**。active successor已经冻结为V6 Layerwise Action-Probe Conditioned
Procedure Reader。它不原样恢复低分memory架构，而是冻结AS139完整强路径，在同一次真实context forward旁读
18层native Action-probe states，以shared rank-query和video内causal delta形成zero-init Procedure-query
conditioner；rank16、factor heads、B20与dynamic-K recipe均不变。literal memory token只在native probe carrier
被机制证据否决后，作为同一下游接口的下一单变量。

canonical实现与fresh schema/config现已完成，CPU机制测试和全量CPU suite=`402 passed`。实现没有增加backbone forward；
18层tap在现有joint forward中旁读，step0退化为AS139，首步只训练zero-init query projection、第二步梯度进入
probe reader与causal controller，base/K-set始终冻结。吞吐审计否决了“288个slot各跑一条重型时序网络”，当前
每video只跑一次共享causal controller，再按有向context汇聚各layer/rank delta。clean pushed `ffa06d4`的
gpu02 world3 live profile已完成：macro wall=`66.134/61.544s`，K1--K4各6、最长323帧完整、peak reserved=
`41.385GB`、0 OOM/nonfinite；真实79帧载体smoke的joint forward=`4`等于native预期，倒序使query-delta/
Program relative-L2=`2.0572/.40414`，常量视频query-delta max-abs=`3.38e-8`。这些只seal机制与吞吐，尚无
新closed-loop成绩。clean detached `515f91e` world6 fresh macro0->25已经完整exit0：25/25 metrics、完整
checkpoint，macro mean=`26.462s`，loss `.10115173->.09563028`，K各6、最长359帧完整、0 OOM/nonfinite。
K4 generation profile的B8/B16/B32=`.221225/.221402/.221500 LoRA/s`，三档稳定，按最高吞吐锁B32。下一动作
是该single macro25 checkpoint的K4 strict400；内部参数展开与吞吐仍不能冒充性能。

## 2. Completed changed variable and training semantics

- v6 evidence、shared Core、有向Procedure、Common-Value operator、native compiler与rank16全部沿用macro25；
- 只训练Procedure q/k/output；source policy与v6底座trainable参数为0；
- 每个train task由K4 videos生成一套LoRA，在四个random resets闭环执行；video仍action-hidden；
- reward阶段source/teacher/validation/test action reads为0，只保留当前policy真实executed prefixes；
- LOO binary advantage、episode/task等权、Nmc4 executed-prefix CFM仍形成同一个raw shared AdamW candidate；
- 每个至少一条成功rollout的train task额外形成一个task-equal success-support tangent；只有最终actual parameter
  delta被投影到全部support half-spaces，video/representation/compiler/optimizer proposal均不变；
- dynamic work queue只改变physical owner，task/video/env/policy/flow seeds不含rank；
- fresh constrained cycle通过机制门后已立即完成strict paired400，reward objective和train80未选择checkpoint。

## 3. Closed-loop adjudication

Ordered-Procedure AS139、raw reward138与ADSP138均已终局且不得resume。ADSP已触发`<144`、相对139 lost>10、
gained不超过lost三项终局门；不得用cycle2或小扫救回。新design authority为
`action_forecast_writer_v6_layerwise_probe_conditioned_procedure_design.md`；先实现one-forward tap、zero-AS139
identity与full24 profile，macro25立即做strict400。首次>=144才讨论续到50，最终成功仍要求strict>150与健康
video controls。

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
