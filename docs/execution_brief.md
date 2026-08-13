# EMBER Execution Brief

更新时间：2026-08-13。本文只定义当前实验与持续迭代的执行语义；实时进度见`active_session_handoff.md`，稳定
owner原则见`current_owner_requirements.md`，历史负结果见`research_history.md`。

## 1. Latest completed closed-loop experiment and active method

当前active方法是Dynamic-K Task-Grounded Visual-Value Rank-8 Writer。它从Direct-Family-B保持：

- exact language + dynamic K1--K4 action-hidden same-task ordered videos；
- 每帧真实π0.5 joint image/language/Action-probe context中的8个memory tokens；
- per-video signed adjacent transition、terminal goal residual与Query-only semantic address；
- causal temporal、permutation-invariant set、20x8 policy-group/rank M2P；
- fixed template A、完整38-target rank-8 LoRA、train24 full-task B20 functional recipe。

唯一主要变量是同一次joint backbone forward中的task-conditioned raw patch Value：它形成有向visual transition
和terminal goal，并以semantic address + layer/rank route读入现有Program。没有额外backbone forward、prediction/
negative loss、expert、reward或language-only Value。完整公式与门见
`action_forecast_writer_dynamic_k_task_grounded_visual_value_design.md`。

formal从clean `caa2e30`在gpu01物理`4,5,6`、world3 fresh完成macro0→100；macro50/100 K1 strict400=
`88/86`、breadth=`5/6`。两点严格配对churn50，说明aggregate近似稳定但能力仍换手。该早期曲线不是终局门，
同一root正锁定world3 topology exact-resume到150；完整曲线继续150/200。精确host、devices、paths与artifact只取
`active_session_handoff.md`。

## 2. Why this experiment is valid

Direct-Family-B的K1/K4=`102/98`且same-task BA方差约降`6.3x`，说明dynamic K/set能过滤nuisance，但被稳定的
per-video task mean本身不够policy-effective。历史v5.2/v6又证明task-token查询raw patch Value与visual transition
是未被否定的强机制。因此本轮只检验：恢复task-grounded视觉过程Value，能否让现有Dynamic-K memory结构写出
更有用且可共存的LoRA。

这轮没有改变video数量、memory token数、rank、set、M2P、mapper、loss、数据或optimizer。若完整曲线失败，只
淘汰当前“无VL Meta、同forward task query、raw D/G reader、B20 functional recipe”的组合，不否定Dynamic-K、
few-shot、memory tokens、所有视觉目标或LoRA输出。

## 3. Training semantics

- 24 train tasks构成一个完整macro；先task内B20 mean，再24-task等权；
- 每macro恰好K1/K2/K3/K4各6 tasks，各task随macro轮换cardinality；
- 每条video内部保序，K条video先独立产生Program，再沿K轴set attention与symmetric reduction；
- K>1时以weight`.05`让sealed singleton Program逼近stop-gradient full-set Program；K1值为0；
- video与action queries同task但跨episode，不允许逐帧动作复制；
- frozen source policy trainable参数为0；validation/test action/reward不产生梯度；
- precision BF16/TF32，AdamW peak LR`3e-4`、warmup17、cosine400、clip1.0；
- checkpoint每25个完整macro写出，含Writer、optimizer、scheduler、sampler/RNG和world topology；
- 当前formal已完整到macro150并锁world3 exact-resume 150→200；后续resume继续锁同一topology。

训练曲线只检查进程健康、finite、K平衡和明显异常，不能选checkpoint或证明方法。

## 4. Current macro50/100 K1 strict400 adjudication

正式arm为：

- role=`validation`，8 tasks x 50 states=`400` rows；
- correct-video，without-replacement seed7；
- one generated LoRA per rollout condition；
- Writer generation batch=`8`，来自sealed throughput profile；
- official π0.5/LIBERO preprocessing、dynamic rollout queue和persistent workers；
- 同一macro50 single checkpoint，不挑task、不融合checkpoint；generation B8。

比较：

- strict paired：semantic101、Dynamic-K100、old134、compiler138、online128；
- aggregate/per-task：v6-fast143；其teacher schedule不同，不伪写成episode-level strict pairing。

报告aggregate、8项per-task、4 suite totals、breadth、retained/gained/lost、churn、top-task concentration；必要时再
看task-mean effective BA、action-target energy和Program->mapper几何解释结果。

结果为`88/400`、breadth5、per-task=`4/0/34/2/0/41/7/0`、per-suite=`4/36/41/7`，top3占
`82/88=93.18%`。相对Direct-Family-B 102为`74 retained/14 gained/28 lost`、churn42、net`-14`；相对
old134/compiler138/online128分别净`-46/-50/-40`。相对v6-fast143逐task差为
`+4/-3/-12/-35/0/+5/-13/-1`。

exact effective-BA诊断没有发现视频噪声或identity坍缩：task/video SNR `16.34→19.05`，task-mean offdiag
cosine`.749→.707`，norm均值`136.64→126.47`；但新旧BA平均cosine仅`.831`、relative-L2`.584`。functional
loss轨迹与Direct-B几乎相同，却让held rollout更差。因此目前最早断点是raw visual D/G经当前functional credit
学到的LoRA方向缺少held on-policy usefulness，而非“没有读取视觉”、set不稳或LoRA过小。

macro100同一panel=`86/400`、breadth6、per-task=`1/3/34/0/0/35/12/1`。macro50→100严格同episode为
`62 retained/24 gained/26 lost`、churn50，union=`112`而single best只有88。BA平均cosine`.80856`、
relative-L2`.69630`、norm ratio`1.17376`；action norm ratio`.98674`但方向cosine`.73896`。因此不是整体scale
变化，而是task-specific方向继续重写；Object1在总数34不变时仍有7 gain/7 lost尤其说明aggregate掩盖漂移。

历史v6-fast早期也弱于后期，design已预注册macro50/100不作终局门。macro150完整checkpoint已封存，当前
world3 exact-resume 150→200且macro150 strict400并行运行；只有完整曲线best≥125、
breadth≥6且macro200未相对best崩落>15，才resume 200→400。

额外fixed-A诊断表明：old134有效BA的逐样本最优rank8保留`.999999`能量，当前随机固定A只可达`.01950`；
train24 experts对应`.99809/.18450`。train24最优共享A虽在experts保留`.94063`，到old134 held只剩`.06811`。
这只预注册“若完整曲线non-pass，则单独让A随task/video生成”的候选；不提前改变当前训练，也不把offline几何当性能。

## 5. K4 result and next design boundary

同一checkpoint的nested K4 strict=`98/400`，相对K1为`80 retained/18 gained/22 lost`、net`-4`，没有解锁
新task。K4把same-task effective-BA相对方差从`.021674`降到`.003438`，而task mean K1→K4 cosine仍
`.99604`。所以动态K/set的功能是明确的nuisance reduction，但当前被聚合的per-video mean本身不是足够高层、
policy-effective的任务知识。

下一fresh architecture只改变set之前的evidence接口：复用同一次joint backbone已计算的task-conditioned视觉
hidden，让真实image content成为bias-free Value，构成per-video Semantic Core及按实际输入顺序重算的visual
transition；现有Action-memory作为layer/rank对齐carrier、动态K/set、M2P、direct mapper、rank8 LoRA和B20
full24 recipe保持。它直接继承v5.2/v6的强机制，但不恢复旧多前端、VL Meta-LoRA或rank16 compiler。

active design为`action_forecast_writer_dynamic_k_task_grounded_visual_value_design.md`。实现只允许在同一次joint
forward中压缩task/patch hidden，并用raw visual D/G作为memory-cell Value；不加prediction loss、negative loss、
额外forward或并行Writer。canonical实现为`9d43e82`，`690dea5`只做数学等价吞吐优化；完整CPU=`378 passed`。
同gpu02物理0--3、world4、B20 matched profile为Direct-Family-B `46.2242s`、successor `49.0775s`，比例
`1.061727x`，通过`1.15x`门；峰值allocated/reserved=`39.303/45.561GB`。正式fresh 0→50已从clean pushed
`caa2e30`在gpu01物理`4,5,6`以world3启动，首个macro健康；完成后做K1 strict correct400，再按design继续
100/150/200。macro25 checkpoint上的K1部署定标为B8/B16/B32 `.984266/.976097/.971736 LoRA/s`，全部稳定并
锁B8。formal已完整到macro150，macro100 strict400已结束、macro150 strict400运行中，同拓扑150→200
exact-resume继续运行。精确活动root
与状态只取`active_session_handoff.md`；profile和内部几何不冒充性能结果。

每轮strict结果完成后，按以下顺序分析：

1. absolute、per-task/per-suite、breadth；
2. 相对最接近方法和历史强基线的retained/gained/lost与能力集中；
3. 若有相邻checkpoint，分析persistent/gained/lost与union gap；
4. 沿`input evidence -> per-video Program -> set -> M2P -> mapper -> effective BA -> fixed action -> rollout`定位
   最早失效接口；
5. 分离科学non-pass与明确工程合同违约；
6. 只对最早接口提出一个主要因果变量，写可证伪authority后实现；
7. 做最小必要CPU/机制验证和吞吐profile，尽快回到真实paired400。

不得用loss、cosine、rank或漂亮五臂margin代替absolute，也不得通过rank/scale/seed/dtype/temperature小扫救一个
失败checkpoint。owner的局部建议不能导致整套已认可方案无证据重写。

## 6. GPU and efficiency

- launch前同时live检查gpu01/gpu02；单节点至多6张，有多少真正合适就用多少；
- 允许在显存峰值余量充足、低util且不明显干扰他人的卡上共驻；不抢占、kill、reset或dummy占卡；
- evaluator当前保守共驻门为util≤10%、已用≤8GiB且剩余≥32GiB；任一越界即拒绝；
- 多卡训练设置`NCCL_P2P_DISABLE=1`、GPU-local NUMA和deferred NCCL；
- fresh可用world1--6；exact resume锁原world topology；
- evaluator用动态cost queue和persistent model/env，不静态拆task；
- 以真实samples/s、LoRA/s、最长视频稳定性和显存峰值选择batch；
- 接受正常BF16/TF32低位差异，不重复forward、固定batch1、扩dtype、逐tensor scan或增加hash。

## 7. Storage, Git and artifacts

- 大run前查询`strg01`上的独立user quota，测canonical root并估计checkpoint/cache/temp峰值；
- formal训练与评测来自clean pushed commit的detached worktree；
- 新run使用fresh output root，不覆盖或部分复用中止/不兼容root；
- formal保留run contract、metrics、macro checkpoints、completion、400 raw rows、aggregate和decision analysis；
- profile/smoke roots只作机制/吞吐证据，不冒充formal；
- meaningful结果更新current state、execution brief、current design、task plan、findings和research history；
- 不把历史命令重新复制进多个文档，精确命令以run contract/invocations为准。

## 8. Collaboration boundary

owner授权在核心目标、信息墙与效率原则内持续自主迭代。当前暂不使用subagents。只有真正需要改变核心目标、
扩大权限、处理破坏性操作或遇到无法从本地证据解决的关键歧义时才向owner停下来请求决定。
