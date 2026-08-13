# EMBER Execution Brief

更新时间：2026-08-14。本文只定义当前实验与持续迭代的执行语义；实时进度见`active_session_handoff.md`，稳定
owner原则见`current_owner_requirements.md`，历史负结果见`research_history.md`。

## 1. Latest completed experiment and active method

最新完成方法是Dynamic-K Task-Grounded Visual-Value Rank-8 Writer。其K1 strict曲线为
`88/86/86/96`，macro200 breadth6、per-task=`1/0/37/2/0/42/13/1`、top3=`92/96`。150→200为
`71 retained/25 gained/15 lost`，仍churn40；相对old134=`68/28/66`。它未达到best125门，终局non-pass，
不resume、不补五臂controls。

当前active方法是Dynamic-K Task-Grounded Full-Factor Rank-8 Writer。它完整保留Visual-Value的：

- exact language + dynamic K1--K4 action-hidden same-task ordered videos；
- 每帧真实π0.5 joint image/language/Action-probe context中的8个memory tokens；
- per-video signed adjacent transition、terminal goal residual与Query-only semantic address；
- causal temporal、permutation-invariant set、20x8 policy-group/rank M2P；
- 完整38-target rank-8 LoRA、train24 full-task B20 functional recipe。

唯一新增机制是同一projected Program上的四个bias-free direct family-A residual readouts。A/B heads全部
zero-init，step0仍`A=A_template,B=0`；第一步functional credit只打开B，之后B非零才让A获得真实policy gradient。
没有额外backbone forward、prediction/negative loss、expert、reward、rank变化或language-only Value。完整公式与
门见`action_forecast_writer_dynamic_k_task_grounded_full_factor_design.md`。

Full-Factor当前CPU与机制验证完成：全量`383 passed`；live world4 full24 B20 profile=`47.4409s/macro`、peak
reserved=`45.563GB`、K1--K4各6、无OOM/nonfinite，已seal formal。没有第二套实现，尚无closed-loop结果。精确
host、devices、paths与artifact只取`active_session_handoff.md`。

## 2. Why this experiment is valid

Visual-Value已证明joint multimodal evidence会materially改变task/video-specific BA，但fixed random A只开放
`.01950`的old134有效右子空间；same-task跨video的理想A行空间又保持`.9997255`。历史v6-fast完整动态A/B达到143。
因此本轮只检验：让现有Program条件化生成完整A/B，能否把已经读到的视觉过程写入policy-effective task subspace。

这轮不改变video数量、memory token数、rank、visual D/G、temporal、set、M2P、loss、数据或optimizer。若失败，
只淘汰当前Program+direct full-factor+B20组合，不否定dynamic K、few-shot、memory tokens或所有LoRA输出。

## 3. Training semantics

- 24 train tasks构成一个完整macro；先task内B20 mean，再24-task等权；
- 每macro恰好K1/K2/K3/K4各6 tasks，各task随macro轮换cardinality；
- 每条video内部保序，K条video先独立产生Program，再沿K轴set attention与symmetric reduction；
- K>1时以weight`.05`让sealed singleton Program逼近stop-gradient full-set Program；K1值为0；
- video与action queries同task但跨episode，不允许逐帧动作复制；
- frozen source policy trainable参数为0；validation/test action/reward不产生梯度；
- precision BF16/TF32，AdamW peak LR`3e-4`、warmup17、cosine400、clip1.0；
- checkpoint每25个完整macro写出，含Writer、optimizer、scheduler、sampler/RNG和world topology；
- fresh训练只在live profile seal后启动；formal起始段为macro0→100。

训练曲线只检查进程健康、finite、K平衡和明显异常，不能选checkpoint或证明方法。

## 4. Closed-loop adjudication

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

macro150仍为`86/400`、breadth6、per-task=`1/0/36/1/1/40/7/0`。100→150为`62/24/24` retained/gained/lost、
churn48、net0；50/100/150 union125而single-best88、intersection53。BA cosine/relative-L2=`.878/.528`，
action=`.859/.621`，说明相同aggregate下仍在实质重写方向和episode能力。

macro200最终为`96/400`、breadth6、per-task=`1/0/37/2/0/42/13/1`，top3=`92/96`。150→200严格为
`71/25/15` retained/gained/lost、churn40；相对old134=`68/28/66`，相对compiler138=`73/23/65`。完整曲线
`88/86/86/96`未过125门，Visual-Value终局non-pass。

额外fixed-A诊断表明：old134有效BA的逐样本最优rank8保留`.999999`能量，当前随机固定A只可达`.01950`；
train24 experts对应`.99809/.18450`。train24最优共享A虽在experts保留`.94063`，到old134 held只剩`.06811`。
这只支持单独让A随task/video生成的当前候选，不把offline几何当性能。
同一old134上按task做3-video-fit/1-video-held-out时，rank8 A行空间保留`.9997255`，8个task均高于`.99916`：
所需A应是从完整language+video Program得到、跨same-task视频稳定的task program，不是静态全局basis，也不是任意
per-video correction。现有shared/singleton Program consistency会直接约束候选A，无需新增LoRA consistency objective。

## 5. K4 result and active design boundary

同一checkpoint的nested K4 strict=`98/400`，相对K1为`80 retained/18 gained/22 lost`、net`-4`，没有解锁
新task。K4把same-task effective-BA相对方差从`.021674`降到`.003438`，而task mean K1→K4 cosine仍
`.99604`。所以动态K/set的功能是明确的nuisance reduction，但当前被聚合的per-video mean本身不是足够高层、
policy-effective的任务知识。

active design为`action_forecast_writer_dynamic_k_task_grounded_full_factor_design.md`。唯一变化是同一projected
Program增加四个zero-init direct family-A residual readouts，和原有四个B readouts共同生成完整LoRA。实现已原位
替换canonical mapper，CPU=`383 passed`且live profile通过；当前启动fresh formal。若best不能达到125，不做mapper/
rank/scale小修，转向v6-fast性能骨架的受控桥接。

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
