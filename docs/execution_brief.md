# EMBER Execution Brief

更新时间：2026-08-13。本文只定义当前实验与持续迭代的执行语义；实时进度见`active_session_handoff.md`，稳定
owner原则见`current_owner_requirements.md`，历史负结果见`research_history.md`。

## 1. Latest completed closed-loop experiment and active successor

最新完成方法是Dynamic-K Semantic-Address Direct-Family-B Rank-8 Writer。它保持：

- exact language + dynamic K1--K4 action-hidden same-task ordered videos；
- 每帧真实π0.5 joint image/language/Action-probe context中的8个memory tokens；
- per-video signed adjacent transition、terminal goal residual与Query-only semantic address；
- causal temporal、permutation-invariant set、20x8 policy-group/rank M2P；
- fixed template A、完整38-target rank-8 LoRA、train24 full-task B20 functional recipe。

唯一主要变量是删除旧mapper的四个family hidden/GELU与inactive dynamic-A heads，让shared `256->1024`
projector后由四个bias-free zero-init linears直接生成q/v/action-in/action-out B。完整公式与门见
`action_forecast_writer_dynamic_k_semantic_address_direct_family_b_design.md`。

formal从clean `c5353f3` fresh macro0完整训练至50。第一次world6 run在owner停止时停于macro16且没有
checkpoint，不resume；world5 retry完整封存macro25/50。同一macro50 checkpoint的K1/K4 strict400分别为
`102/98`、breadth均5，终局non-pass，不resume到100。精确host、devices、paths与artifact只取
`active_session_handoff.md`。

## 2. Why this experiment is valid

该方法不是根据最终LoRA猜测瓶颈。上一代semantic-address macro50=`101/400`后，validation8 x 4 ordinals x five-arm
逐接口probe显示：M2P/final/shared projector保留较多task/order结构；旧family hidden/GELU是第一个明显增加
common direction、压小order contrast的接口。因此本轮只检验：去掉这个没有独立依据的nonlinear bottleneck，
能否让已有Program写成更policy-effective、任务可分的LoRA。

这轮没有改变video数量、memory、rank、set、M2P、loss、数据或optimizer，因而macro50与semantic101的差异可以
归因于mapper变量。若失败，只淘汰这个direct readout假设，不否定Dynamic-K、memory tokens、few-shot、rank8或
视频学习整体。

## 3. Training semantics

- 24 train tasks构成一个完整macro；先task内B20 mean，再24-task等权；
- 每macro恰好K1/K2/K3/K4各6 tasks，各task随macro轮换cardinality；
- 每条video内部保序，K条video先独立产生Program，再沿K轴set attention与symmetric reduction；
- K>1时以weight`.05`让sealed singleton Program逼近stop-gradient full-set Program；K1值为0；
- video与action queries同task但跨episode，不允许逐帧动作复制；
- frozen source policy trainable参数为0；validation/test action/reward不产生梯度；
- precision BF16/TF32，AdamW peak LR`3e-4`、warmup17、cosine400、clip1.0；
- checkpoint只在完整macro25/50写出，含Writer、optimizer、scheduler、sampler/RNG和world topology；
- 当前fresh world5；若获准50->100，必须同一world5 exact-resume。

训练曲线只检查进程健康、finite、K平衡和明显异常，不能选checkpoint或证明方法。

## 4. K1 strict400 adjudication

正式arm为：

- role=`validation`，8 tasks x 50 states=`400` rows；
- correct-video，without-replacement seed7；
- one generated LoRA per rollout condition；
- Writer generation batch=`8`，来自sealed throughput profile；
- official π0.5/LIBERO preprocessing、dynamic rollout queue和persistent workers；
- 同一macro50 single checkpoint，不挑task、不融合checkpoint。

比较：

- strict paired：semantic101、Dynamic-K100、old134、compiler138、online128；
- aggregate/per-task：v6-fast143；其teacher schedule不同，不伪写成episode-level strict pairing。

报告aggregate、8项per-task、4 suite totals、breadth、retained/gained/lost、churn、top-task concentration；必要时再
看task-mean effective BA、action-target energy和Program->mapper几何解释结果。

结果为`102/400`、breadth5、per-task=`0/1/40/11/0/43/7/0`、per-suite=`1/51/43/7`，top3占
`94/102`。相对semantic101虽净`+1`，但有`20 gained/19 lost`；相对old134为`22 gained/54 lost`。task-mean
effective-BA cosine仅从`.77947`降至`.74895`，没有带来绝对或breadth增益。按`<120`或breadth<6门终止，
不resume、不做小扫、不补K1完整controls。

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
100/150/200。精确活动root与运行状态只取`active_session_handoff.md`；profile不冒充性能结果。

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
