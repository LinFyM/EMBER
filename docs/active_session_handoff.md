# EMBER Active Session Handoff

## 1. Current truth

这是当前跨session科研入口，覆盖所有历史文档里的旧“当前”“下一步”和阶段性暂停。

- Goal仍未完成：同一shared method、同一single checkpoint的strict paired correct必须严格超过
  `150/400`并尽可能继续提高，同时要求真实视频时序因果性、same-task鲁棒、breadth、低checkpoint
  漂移和可重复累积。
- Owner已授权持续自主推进，只有实质阻塞才沟通，并已允许subagent承担独立、只读或隔离写入的
  加速工作；主进程仍负责统一科研判断和最终验证。
- canonical仓库是`/data1/user/ymdai/projects/EMBER`，主写分支是`codex/bci-continuation`。
  retained formal GPU工作必须来自clean pushed commit的frozen worktree。
- 当前没有EMBER GPU进程。任何新GPU工作前必须实时比较`gpu01/gpu02`，只用空闲A40、合计最多6张，
  不干扰他人；多卡固定`NCCL_P2P_DISABLE=1`、NUMA physical/local rank映射和deferred-NCCL。
- 历史最好single checkpoint仍是v6-fast macro400的`143/400`。v6-prior已完成formal 0→50；同一
  schedule macro0/10/25/50 strict correct400=`134/127/105/123`，correct80=`26/26/24/27`。小panel在
  macro50看似上升而full400仍下降，进一步证明不能用screen替代正式裁决。
- 四点严格逐row配对分析：0→10 gained/lost=`19/26`；0→25=`19/48`、McNemar
  `p=.000522`；0→50=`20/31`。四点success union=`172`、intersection=`77`、逐task envelope=`147`，
  但union/envelope均不能作checkpoint融合。macro0仍是当前schedule单点winner。
- 内部根因不是“expert loss没动”：generated norm约`140.97→107.00`，cosine仅
  `.02194→.02630`，expert loss下降的约`94.2%`来自log-norm径向项；绝对expert投影系数
  `a=<G,E>/||E||²`反而`.736→.662`且23/24 tasks下降。held state0 macro50相对macro0的norm ratio/
  cosine/radial coefficient/orthogonal residual/base/delta/base均值=`.7180/.9755/.7007/.1551/.3373`。
  因而当前训练主要缩小已有v6 LoRA，而不是补足有用expert方向。
- 当前代码已完成对`0894856`后batch1/逐元素复现策略的撤回。A40已证明batch8只产生普通BF16
  batch-shape roundoff（max`.001953125`、mean约`4.70e-5`，direct-repeat为零）；owner明确要求不为这种
  微差牺牲吞吐。新路径在同一个固定request/总帧panel上从稳定且有显存余量的候选中选择实测LoRAs/s
  最高的batch，使用原生BF16/F32
  LoRA cache、零重复Writer forward、更少host sync和2-worker action prefetch。clean pushed
  `ded0c80`的A40 fixed-panel profile选择batch8；随后8-task纵向smoke完整通过并由retained artifacts
  组装evaluation seal。gradient/resume结构化artifact verifier与只读checkpoint comparator也已完成
  CPU验证。clean frozen`a17805c`随后两次启动六卡macro49 gradient profile：默认allocator和唯一一次
  `expandable_segments:True`重试都在第一个PI05 policy functional B20的Gemma MLP前向发生容量OOM；后者把
  reserved-unallocated从约`1.29GiB`降到约`157MiB`仍无法分配`606MiB`，因此碎片不是主因。两个root均只有
  run contract/invocation、没有gradient/completion，不能seal或resume，也没有产生方法性能结论。
- 当前修复保持logical B20、每task mean、train24×20=`480/480` unique queries和objective分布不变；
  完整有序logical-B20 panel keyed的physical slicing使用FP32 leaf-gradient加权累积，seed为轻量固定64-bit
  整数mix，不调用SHA/MD5。clean frozen `eddba96`的B16+4在六rank第一条functional attention一致OOM：
  allocated=`42.49GiB`、reserved-unallocated=`1.25GiB`、free=`235.31MiB`，尚需`254MiB`。所以当前只把
  physical microbatch改成balanced B10+10；这仍是A40容量实现变量，不是减小scientific batch。
  policy activation checkpointing目前不启用，因为
  OOM在frozen PI05 policy而现有checkpoint flag只覆盖Writer，启用policy重算会是更侵入且可能更慢的变量。
- clean frozen `9c814ff`的balanced B10+10已完整通过macro49：wall=`21.095s`、input wait=`.076s`
  （`.36%`）、peak allocated/reserved=`40.332/43.859GiB`、0 OOM/nonfinite；完整assembler通过。唯一权重为
  expert=`.008355172068998324`、ranking=`.28570466890490887`，两者在compiler各等于positive梯度的`.25`，
  在factor仅`.05254/.03993`。config已原样写入gradient evidence。
- strict后继`5fbcb27`已在`gpu02:0--5`完成fresh0→1+same-root exact-resume1→3和独立contiguous0→3；
  两root合同相同、各3 metrics、macro1/3 checkpoints和completion，0 OOM/nonfinite/clip。contiguous/
  resumed总step wall=`61.368/64.450s`，峰值allocated/reserved=`43.266/47.119GB`，steady-state input wait
  约`.0006s`。所有cursor/RNG/scheduler/AMP/frozen tensors精确相等，trainable Writer与Adam逐tensor科学
  门通过。原比较器误把近零Adam moments套入Writer aggregate relative门，已只修离线state-specific
  tolerance而未改训练或重跑GPU；retained artifacts重新assemble通过，profile/formal现已正式sealed。

第33节whole-LoRA direction/norm与第34节Expert-Component Projection（ECP）均已由formal
closed-loop证据退役。ECP保留原v6架构和上游冻结边界，只把auxiliary换为
`a=<G,E>/(||E||²+epsilon)→1`与bounded negative ranking；因此这是对expert component假设的
干净单变量检验，不是新video encoder或LoRA topology实验。

clean pushed/frozen`450e688`的formal root=
`runs/outputs/pi05_v6_ecp_formal_r6_lb20_mb10_450e688_20260809`，fresh0→10后又按预注册门
exact-resume10→25；25 metrics、macro10/25 checkpoints、optimizer/scheduler/sampler/六rank RNG与
completion完整，0 OOM/nonfinite/clip。macro1→10的`a_correct=.736184→.828442`、expert
component=`3.06189→3.44394`、generated norm=`140.973→151.343`，24/24 tasks的`a`向1移动且
component上升。macro10 strict correct=`133/400`、breadth6，对同schedule macro0=`134`的精确
gained/lost=`22/23`、net=`-1`。这证明ECP修复了部分旧objective的径向伤害，但没有建立
held共同改善。

10→25的内部机制继续按目标运作：`a_correct=.828442→.884127`、component=
`3.44394→3.67225`、generated norm=`151.343→159.817`；23/24 tasks的`a`向1移动，24/24
component和norm上升。但expert-orthogonal norm约`151.303→159.774`，增量`8.471`远大于
component增量`.228`。macro25 strict correct反而降到`120/400`、breadth6、per-task=
`0/1/43/27/0/33/15/1`；相对macro0的严格配对gained/lost=`13/27`、net=`-14`、
McNemar `p=.038477`，suite net=`-4/-12/-2/+4`。macro10→25也是`18/31`、net=`-13`，
四个suite全部净下降。

裁决：ECP不续50/100、不扫权重、不为loser补六臂。直接增大expert component权重已被
证据禁止。第35节已完成与历史anchor/tangent/distillation去重，并在同一canonical vertical path
原位实现Condition-Local Dynamic Expert Tangent Tube：historical v6对correct和当前negative的同一
language/video/order输出分别作局部baseline，只惩罚student增量的expert-orthogonal分量。新v3 config、
training-only decoder anchor、trainable-only resume/deployment load、双臂metrics及独立评测family已通过
exact-D/gauge/gradient oracle、全仓`276 passed`、compileall与diff-check；clean push/frozen前禁止GPU。
当前没有EMBER GPU进程。

## 2. EMBER problem and information wall

EMBER不是video imitation replay。它要求：

```text
exact task language + one action-hidden teacher video
                    │
                    ▼
               shared Writer
                    │  one pre-rollout forward
                    ▼
       one complete 38-target rank-16 LoRA
                    │
                    ▼
       frozen π0.5 source policy + live observation/state
                    │
                    ▼
             closed-loop task execution
```

Writer只能从语言与视频抽取任务的高层语义、对象关系、阶段和动作顺序；teacher action、proprio、reward、
terminal、task ID、filename、object pose和hidden normalization均在墙外。视频是唯一dynamic value；语言
可以定位“要解决什么”，不能单独形成LoRA旁路。video和functional action query必须错开episode，避免把
教学轨迹的低层运动与监督动作机械对齐。最终要求同一生成LoRA泛化到该任务不同初始化，而不是复现视频
轨迹。

one-shot是当前目标合同。few-shot的合理作用是从多个同任务视频中提取共同程序并消除单视频偶然细节，
但历史K4已证明“看多个视频”本身不解决condition-to-policy credit、共享梯度抵消或正确时序辨识。
因此只有当当前one-shot的同任务跨video方差被closed-loop证据定位为最早瓶颈后，才恢复固定K聚合；不能
通过平均多个无效LoRA或expert route伪造提升。

## 3. Current architecture

部署图恢复历史v6-fast完整video-to-LoRA生成器：

1. frozen π0.5 multimodal hidden对exact language和raw video形成task-grounded per-frame evidence；
2. Semantic Core聚合跨帧稳定语义；visual transition显式计算相邻变化；Causal Procedure保留动作阶段
   和顺序；
3. 320-slot compiler将Core/Procedure写入public LoRA topology；
4. 8个factor heads直接生成38个policy targets的完整rank-16 A/B；
5. LoRA只生成一次，Writer释放，原source policy原位加载该LoRA做闭环rollout。

初始化是历史v6-fast macro400 checkpoint：

`runs/outputs/pi05_as_writer_v6_decay400_taskcomplete_dev_r4_b20_seed7_s2400_4efa737_20260729/checkpoints/step_00000400`

其600个Writer tensors只作load-only初始化。encoder/Core/transition/Procedure共483 tensors、
`7,060,992` parameters冻结；仅训练compiler+factor heads的41 tensors、`3,714,304` parameters，使用全新
optimizer/scheduler/sampler/RNG。这样第一次干预只针对历史证据定位的Procedure→effective LoRA写出
接口，同时保护143起点已经具有的video representation；这不是永久宣称上游最优。

每个train task的统一step2000 expert `E_t`只作监督；当前第35节目标为：

- correct臂接受真实logical B20 action functional loss；physical slicing只允许改变执行显存，不改变20条
  query与task mean；
- 用全部38 targets的gauge-invariant effective`BA`内积计算原ECP系数
  `a_t^x=<G_t^x,E_t>/(||E_t||²+epsilon)`；correct仍只要求`a_t^correct→1`，不把global norm或整套LoRA
  拉向expert；
- runtime在historical macro400同步完成、任何resume load之前，冻结复制恰好compiler+factor heads作为
  macro0 decoder。对每个correct/negative condition复用同一份Core/Procedure memories，得到
  `G0_t^x`；令`Delta=G-G0`，用exact nonzero `D=||E||²`只惩罚
  `||Delta||²-<Delta,E>²/D`。两臂取算术均值，因此不会因增加negative anchor机械翻倍；
- reversed、shuffled和cross-suite wrong只进入bounded
  `a_t^correct-a_t^negative` ranking，达到margin后停止推动；negative tube同时阻止共享参数更新连带破坏
  其非expert方向来伪造margin；
- same-task不同视频都是共同positive，不互相排斥。

dynamic anchor不进optimizer/checkpoint/deployment；exact resume从immutable warm-start重建它，只恢复
student的41个trainable tensors。部署仍是单一600-tensor student，不读expert bank或phase cache，不选、
混合或近邻复制train expert。

## 4. What task experts solve and do not solve

正式expert root：

`runs/outputs/pi05_task_expert_bank_formal_step1000_r6_81101fe_20260807`

24/24 train tasks均统一续到step2000，共120 checkpoints。统一step250/500/1000/1500/2000的
development-train direct-expert closed-loop为`432/557/624/638/658` of 1200；step2000有23/24 tasks非零、
task9仍为0，因此统一选择aggregate最强的step2000但不把它写成完美oracle，也不按task混点。

Experts解决：

- 在同一frozen source policy和public rank-16 topology上，给出“这个train task的什么参数更新确实能
  闭环工作”的policy-effective target；
- 提供正常task-local SFT LoRA的能量、rank坐标、跨target分配和有效方向参考；
- 避免meta-Writer只被高方差functional query推动、完全不知道有效参数流形在哪里。

Experts不解决：

- held task如何从video生成可迁移LoRA；
- 同一task的多个video之间应关注哪些共同程序；同task expert target对所有video恒定；
- correct、shuffled和reversed为什么不同，或时间顺序是否被真正理解；
- train24 expert是否可直接充当validation expert字典。soft/hard部署的`15/80`和`3/80`已明确否定这一点。

因此expert reconstruction loss和健康几何只能作训练辅助；真正裁决仍是paired closed-loop五臂/六臂。

## 5. Cumulative experiment lineage

下表是设计决策的连续证据链，不是候选排行榜。精确roots和完整分析仍保留在对应design、`findings.md`、
`progress.md`、Git和formal artifacts中。

| 方法/干预 | 最强strict证据 | 实际证明 | 失败或未证明 | 当前保留结论 |
| --- | ---: | --- | --- | --- |
| frozen source base | `48/400` | generic-source policy无目标适配也有非零能力 | 不读video，未检验video adaptation | 所有Writer共享的frozen起点 |
| mixed-task Source-SFT rank128 | `109/400` | direct target action可形成共享适配 | privileged且仍低于目标 | 参数预算/闭环参照，不是同信息墙baseline |
| v5.2 old recipe | `132/138/74/82/83` | 当前最强correct-vs-negative视频特异性 | absolute未过150 | 动态写出与顺序margin可实现，不能遗忘 |
| v5.2 task-complete | `120/109/107/111/124` | recipe会改变video传递 | absolute和margin均退化 | task-complete并非普遍改进 |
| v6 old recipe | `121/122/111/84/47` | 强时序差异可传到闭环 | absolute低、task旋转 | old recipe增强动态也增强不稳定 |
| v6-fast task-complete | `143/135/125/128/129` | 历史最佳eligible raw single-checkpoint absolute；Procedure差异可传到LoRA/action | 原objective续到450/500/550/600降为`131/130/132/126`；冻结上游迁移性仍是假设 | 当前representation prior与macro0基线，不续旧objective |
| CV-ADR RAW / GROUP4 | best`117` / `110` | 更大coherent更新不等于更好闭环 | 曲线漂移；video梯度主效应约`.1%` | query/flow variation主导；仅flow MC noise可约且不是主因，credit仍错位 |
| Target-Bound | best`120` | remove-A/D和memory reversal 8/8达门，动态路径工作 | correct漂移、共享factor共存失败 | 不再把首因归为video完全未使用 |
| Semantic Factor-Basis | best`127`，union`193` | common accumulation一度改善 | envelope gap66、严重换手 | shared credit仍未稳定积累 |
| variance-reduced estimator | best`126` | 精确Beta/antithetic MC只小幅改善gradient consistency | held loss改善但closed-loop退化 | flow Monte Carlo方差不是主因 |
| Semantic Direction Store | best`129` | 独立store改善早期acquisition | 同分checkpoint breadth不同；Program→factor压缩 | parameter coexistence是局部因素，不是根因 |
| Policy-Target-Owned Factor | best`99` | 解除38-target共享显著改善跨层异质性 | action效果和性能仍差 | 健康跨target几何不是充分条件 |
| Policy-Lane | best`70` | 形成约10条有效output lanes和SFT量级专门化 | video BA能量约`.02%` | 容量/形态健康不能替代动态credit |
| Policy-Wide Atom Dictionary | best`80` | 64 atoms广泛使用 | mixing/effective LoRA近rank1 | 不用增atom/rank/正交loss救活 |
| Factorized Condition-Kernel | best`49` | kernel full-rank、stable且跨video差异大 | LoRA约比direct SFT小200×，identity-like | 低增益decoder曾是明确瓶颈，但非唯一根因 |
| Few-Shot Invariant-Program K4 | best`108` | K4置换、same/LOO/wrong/order路径都工作 | full24 gradient retention约`.043` | few-shot可去偶然性，不能自动解决共享credit |
| K4 Policy-Layer Trace | best`99` | all-layer trace带来correct>wrong | reversal仍高、逐频单位化把低能量DCT高频放大约`140×` | 被放大的高频不能替代有物理意义的时序程序 |
| Energy-Preserving Trace | best`85` | 修复原始频率能量比例 | correct/wrong从`99/57`收缩到`85/80`；effective groups`13.97→10.63` | 能量保真本身不等于语义保真 |
| Evidence-Factorized Trace | best`84` | correct>wrong且trace→BA→action闭合 | shared Reader retention约`.05` | 参数隔离值得检验，但不是直接答案 |
| Sparse Semantic-Expert | best`78` | expert-local retention提高 | language route固定owners，wrong/order更成功 | language-only ownership不够；video须参与credit |
| Grounded-Video Expert | best`88` | video route、Reader、BA、action和rank均material | correct无margin、task轮换 | video sensitivity与parameter isolation仍不充分 |
| K4 Phase-Aligned v6 | best`108`; reversed`121` | video未被忽略 | 近rank1、高能量、程序retention约`.04` | phase alignment不能独立解决语义/credit |
| AS125 + semantic progress RL | `97→104→102` | failure轨迹可获得非零semantic credit | breadth下降、继续训练换手 | reward信号存在，但共享更新不稳定 |
| Program-Credit RL | `106` | lockstep CRN和program gradient可达 | task cotangent几乎正交却被压成common update | shared condition map会吞掉task-specific credit |
| SFT-Anchored Tangent-Basis | `143→142` | 在强warm-start上小幅reward更新可运行 | gained/lost`20/21`，无净提升 | 不能把warm-start保持分数冒充生成器改进 |
| task experts step2000 | train`658/1200` | aggregate最高且`23/24` tasks非零的privileged train target | task9仍为0、存在state turnover；不证明video或held泛化 | 保留为监督流形，不作部署字典 |
| addressless Expert-Manifold | `48/400` | raw-expert reconstruction能训练出norm约`4.55` | 与source exact同分、paired`5/5`；topology identity在decoder后坍缩，nearest expert cosine约`.008` | 无显式topology address的decoder已证伪 |
| topology-address binding | `75/400` | 静态chunk/rank坐标可乘性调制video dynamic value并进入闭环 | 输出仍高度task-common，held绝对性能低 | 地址辨识修复有效；不能单独调address解决迁移 |
| Causal Barycentric | `63/400` | temporal coefficients和raw-factor组合可运行 | `k≠j` cross terms使raw A/B组合不保持effective update；未单独隔离held support | policy-effective compiler必须先于组合几何 |
| policy-effective soft / hard bank | `15/80` / `3/80` | hard compiler近精确复现所选expert | 当前causal reader + 24个step2000 experts的soft/hard held support均失败 | 关闭当前24-expert online部署字典，不外推所有未来流形方法 |
| v6-prior whole-LoRA objective | `134→127→105→123` | 冻结上游、只训写出端可高吞吐稳定运行；晚段可部分回升/breadth7 | 整体方向+norm吸引主要径向收缩，macro0仍最佳，绝对expert投影下降 | 退役该objective，不外推v6表示无效 |
| v6 Expert-Component Projection | `134→133→120` | `a_correct`与component按构造上升，修复旧径向收缩 | 正交漂移继续增大，macro25 paired net`-14`、`p=.038477` | 退役；不续、不扫权重 |
| current Condition-Local Tangent Tube | CPU oracle/合同`276 passed`；无GPU成绩 | same-input dynamic baseline可在effective space精确隔离expert平行/正交增量 | 尚未证明tube能保住143起点或改善held闭环 | 当前唯一活动候选；先profile再短训strict裁决 |

任何需要精确数字的决策必须回到对应design/artifact，而不是从本表反推未列指标。

## 6. Stable cross-experiment cognition

1. **视频被使用不等于视频被正确使用。** 多条路线都证明wrong/shuffle/reverse能显著改变hidden、BA和
   action，但correct仍可能更差。下一分析必须问改变是否沿policy-effective方向，而非只看差异大小。
2. **LoRA健康度是约束，不是目标。** 低能量、过度rank1和高列相似度曾解释部分失败；但形成SFT量级
   能量、多个lanes、跨target异质或正确expert cosine也未自动提高closed loop。不得单指标优化。
3. **functional surrogate长期错位。** held functional loss下降、gradient更稳定或full-rank kernel均曾与
   closed-loop退化共存；关键checkpoint必须及时rollout，不能用loss挑点。
4. **task drift不是一种表象对应一个单因。** query/flow variation（其中只有flow MC noise可约）、full24正交抵消、shared parameter
   coexistence、Program→factor压缩和condition-map common update都被逐步检验；其中每个只解释局部。
5. **正常时序必须有因果意义。** shuffled/reversed真正破坏frame展示顺序；模型不能依靠原时间戳恢复。
   correct必须同时接近有效policy update并超过negative，不能仅把negative推向坏LoRA。
6. **架构与recipe耦合。** v5.2/v6交叉结果证明不能按某一architecture aggregate整体判死，也不能直接
   恢复old recipe；需要对比最早传递接口和任务换手。
7. **当前假设是局部且可证伪的。** v6 checkpoint已达143且上游仍保留material Procedure response；
   冻结部分是否足以迁移仍是假设。目前只修compiler/factor如何把它写到expert定义的policy-effective流形。若目标/梯度成立而closed-loop仍无共同提高，才重新打开
   Procedure或更早表示，不因一次低分立即换整条路线。

## 7. Current engineering state

当前实现与单卡live seal已完成：

- evaluator取消historical smoke中的8次冗余direct Writer forward和`1e-5`逐tensor门；batch默认8并要求
  profile至少实测`8/16/32`。三者处理同一32-request longest-first panel和同一总帧数，只改变实际
  forward分批，最终从稳定且有显存余量的候选中取LoRAs/s最高值；
- 76-tensor LoRA保持template原生dtype：72个BF16、4个F32，单entry tensor bytes从强制FP32的
  `5,148,672`降到`2,641,920`；batch GPU→CPU staging只同步一次；
- functional objective仍是logical B20；live A40已依次证明single physical B20和B16+4不能容纳，当前使用
  balanced physical B10+10，通过完整有序logical panel identity重放同一20个独立Beta/Gaussian draws，
  并用76个FP32 leaf-gradient buffers按`10/20`和`10/20`加权累积；
- PI05 formal functional路径不再调用只供日志使用的`.cpu().numpy().tolist()`/`.item()`；loss-only实现与
  原forward的loss及LoRA leaf gradients由固定noise/time测试验证一致，通用details接口保持不变；
- correct effective alignment只计算一次，task metrics和gradient norms合并成少量host transfer；
- action DataLoader默认2 workers、spawn、persistent workers和prefetch2；确定性sampler的serial、
  prefetched和prefix+resume rows已精确一致；
- Writer offsets、frame ordinal/order、language span/condition ownership的重复CUDA→host门已合并或
  hoist到CPU，vectorized token packing不再逐row产生动态CUDA selection；必要的D2H handoff和宏步wall
  synchronization保留；
- 新`profile-writer-generation`在同一loaded source policy/Writer上做真实video→LoRA→native D2H sweep，
  记录fixed-panel actual forward batches、repeat wall、longest video、peak allocated/reserved和headroom；
  launcher及独立单卡worker均在模型load前live检查空闲NVIDIA A40，worker还核对clean pushed checkout；
  普通evaluator在spawn前拒绝忙卡、非A40或合计超过6张卡，历史8卡root也不能绕过该启动门；
- evaluation seal只能由profile root与vertical smoke root的retained artifacts组装，校验三候选完整request/
  sampled-frame panel严格相同、warmup/repeats、最长视频、selected throughput、native cache、release/reuse
  和单次成功launcher；
- config状态图已修为`blocked → evaluation sealed/gradient ready → gradient+aux sealed/profile ready →
  profile sealed/formal ready`，不再自我依赖；task-expert authority使用窄loader，不会因已退役topological
  Writer/meta字段变化阻断当前训练监督；
- clean frozen `ded0c80`在live空闲`gpu02:0`完成32-request、1093 sampled-frame fixed panel。
  batch8/16/32吞吐分别为`.911427/.905107/.906432 LoRA/s`，三者均稳定且峰值reserved约
  `12.82--12.85GB`；batch8按封存规则实测最快。大batch没有吞吐收益，剩余显存本身不是选慢配置的理由；
- 同提交fresh vertical root完成8 videos→8 LoRAs/cache→8 rollouts，单次attempt、`0` retry/failure/
  OOM/nonfinite/forbidden reads。Writer生成`10.597s`，peak allocated/reserved=
  `11,651,564,544/12,811,501,568` bytes；release后source policy原位复用且未reload。总wall=
  `325.540s`、rollout window=`196.816s`，进程结束后GPU回到0MiB；
- artifact assembler已从两个单卡retained roots重建evaluation evidence；gradient assembler又从macro49
  retained root重建权重和完整provenance；
  gradient assembler现会精确重建macro49的24-task teacher-demo/counterfactual schedule、480 unique
  queries、canonical config、clean pushed Git、frozen target manifest/HDF5 frame metadata与六卡拓扑；
  resume assembler会比较fresh/resume/
  contiguous的contract、cursor、6-rank RNG、600 Writer tensors、41 trainable tensors、Adam moments、
  scheduler/AMP和scientific tolerance，并要求gradient→profile的strict Git ancestry；
- frozen`a17805c`在当时live空闲`gpu01:0,1,2,4,5,7`的3+3 NUMA拓扑完成了两次有效工程诊断。默认allocator
  OOM时PyTorch allocated=`42.29GiB`、reserved-unallocated=`1.29GiB`、free=`395.31MiB`；唯一allocator
  retry为allocated=`43.43GiB`、reserved-unallocated约`157MiB`、free=`389.31MiB`，仍请求`606MiB`失败。
  这关闭“只调allocator即可保留physical B20”，但不关闭logical B20、当前Writer或任何科研假设。两次
  launcher退出后所选六卡均释放，未触碰当时由他人占用的GPU3/6。
- clean frozen`eddba96`在新一次live preflight后复用当时仍为空闲的同一3+3拓扑。首个非持久SSH后台
  launcher只写contract/invocation便exit0，没有start/gradient/completion，作为无效进程托管证据保留；
  改用tmux的fresh retry完整进入start，六rank均在第一条functional eager-attention申请`254MiB`时OOM，
  allocated=`42.49GiB`、reserved-unallocated=`1.25GiB`、free=`235.31MiB`。因此B16不存在可比较的吞吐点，
  当前canonical config已转为B10+10；上述root都不能seal、resume或选择auxiliary weight。
- clean frozen`9c814ff`随后用同一拓扑完成B10+10。24-task/480-query/105-frame panel、Git/config/
  HDF5、NUMA/NCCL、default allocator和single invocation全部经assembler闭合；input wait仅`.36%`，
  不再测试workers4。macro0 generated/expert effective norm mean=`140.52/4.182`、cosine=`.02196`，而
  reversed/shuffled margin仅`.000832/.000634`；这是当前要由受控expert/ranking更新纠正并由closed-loop
  证伪的核心机制矛盾，不是新增性能成绩。
- clean frozen`5fbcb27`的正式retry1比较root位于
  `runs/outputs/pi05_v6_prior_profile_resume_r6_lb20_mb10_5fbcb27_retry1_20260809`和
  `runs/outputs/pi05_v6_prior_profile_contiguous_r6_lb20_mb10_5fbcb27_retry1_20260809`。macro3 Writer
  maxabs/relative-L2=`4.6033e-5/1.06393e-5`，只占两步update L2的`1.023%`；Adam maxabs=`2.6865e-6`，
  `.007719` relative值来自近零moment分母。离线v2门改为Writer global relative L2`≤.002`，Adam每个
  moment的symmetric norm ratio`≥.99`且cosine`≥.999`；raw maxabs/relative-L2只诊断，并保留逐tensor
  `2e-4/2e-3`、全部语义exact和frozen exact门。config现为profile/formal
  `sealed_from_live_a40_resume_profile_evidence`；该seal只证明工程连续性，不是性能成绩。
- v2 retained assembler、config load和状态机均重新通过；聚焦checkpoint/contract为`11 passed`，加载
  `.env.local`后的全仓CPU回归为`247 passed`。未为本次seal启动任何额外GPU工作。

被撤回的失败root仍保留科学诊断：

`runs/outputs/pi05_v6_prior_warmstart_reproduction_smoke_validation8_correct_gpu02g0_30b2ccf_20260809`

其中`diagnostics/batch_equivalence.json`只证明BF16 batch-shape差异，不是performance evidence。

## 8. Key uncertainties and structural risks

- 冻结上游可能保护143，也可能冻结了task-complete造成的弱video margin；当前实验必须先判断写出修正
  能否提高absolute，再决定是否开放Procedure。
- 一个task恒定expert target可能诱导Writer忽略same-task视频细节；ranking提供顺序/错误约束，但仍可能
  被language+粗task identity满足。五臂/六臂和same-task跨video是唯一裁决。
- exact effective-BA expert监督来自train24，可能把held输出拉向训练任务流形而减少组合泛化；soft/hard
  bank负结果使这一风险很真实。auxiliary必须被positive functional gradient限制，不能主导。
- compiler与factor heads只有约3.7M可训练参数；若gradient可达但不同task更新继续正交抵消，单纯延长训练
  会重现checkpoint换手。必须看per-task acquisition/forgetting和block gradient retention。
- 训练negative schedule只采一种counterfactual/visit；margin改善可能来自识别低层反常而非理解任务
  程序。需要跨video、cross-suite wrong和真实closed-loop同时验证。
- 吞吐优化改变BF16低位和cache dtype，但不改变计算图/信息墙；profile必须验证无串样、finite、显存、
  release/reuse和端到端rollout，不再要求无意义的逐元素同一。

## 9. Evidence order and next action

证据关系固定为：

1. full-bank geometry和expert closed-loop已经完成，证明step2000是统一且policy-effective的train目标；
   它们不能选择held Writer，也不能证明视频因果性。
2. 统一续训到2000已完成且只改善train expert target，不再是当前决策分支。
3. 单卡吞吐/显存profile和vertical smoke已通过，只证明高效端到端实现成立，不证明当前方法有效。
4. 六卡gradient profile只选择一次`lambda_projection/lambda_rank`并验证resume；它不证明方法有效。
5. formal关键checkpoints必须及时跑paired correct400并和同schedule macro0、历史143以及最邻近旧架构
   逐task比较。
6. single winner首次超过历史143即跑完整correct/same/wrong/shuffled/reversed/no-video；若之后不同
   checkpoint严格超过150，再对实际goal winner重跑六臂。未过门则按最早失败接口做单变量修正并继续循环。

当前具体下一步：将已通过第35节CPU dense/low-rank/gauge/gradient oracle、same-memory anchor、
checkpoint/deployment ownership和全仓合同门的唯一canonical v3 config与实现clean commit/push，再从
该commit创建frozen worktree。随后重新live比较`gpu01/gpu02`和quota，只用最多6张空闲A40，先做一次train24
gradient profile以及fresh0→1、same-root exact-resume1→3、independent contiguous0→3；不为BF16低位
一致降低B10+10、六卡并行或吞吐。profile checkpoint永久不进formal。

profile seal后formal仍从historical v6 macro400 fresh训练，先到macro10并立即跑paired correct400；
correct80只能从这400 rows派生。historical immutable macro0=`134`由generic historical-baseline analyzer
按各自native family验证并逐row核对共同state/RNG/language/video后比较，绝不重标family或混入同family
checkpoint curve。macro10 `≤129`且广泛净损失即停；`130--134`仅在tube、breadth`≥6`、churn`≤35`和
右端斜率共同成立时到25；macro25需`≥135`且至少3 tasks/2 suites净正增。首次single checkpoint
`≥144`即跑六臂，若不同goal winner首次`≥151`再跑一次。不能用profile loss、expert norm、correct80或
历史143替代full400裁决。

对应CPU-only入口已原位加入canonical evaluator：
`scripts/evaluate_pi05.py historical-baseline-transition --legacy-root ... --current-root ... --output ...`。
它不放宽四点checkpoint-curve；legacy只读immutable `results.json`，candidate从raw root重新aggregate，
两个native family分别验证后才比较。实现保留legacy+ECP历史入口并新增tangent v3 candidate family；
checkpoint curve仍严格single-family。

## 10. Canonical assets

- source policy：由当前config/CLI显式传入的frozen generic-source step1000 asset；不是
  `pi05_libero`，也不支持source-SFT exact resume。
- task experts：上述formal root的统一step2000 checkpoints。
- historical Writer prior：上述v6-fast macro400 checkpoint。
- current config：`configs/pi05_v6_condition_local_tangent_tube_writer_v3.json`。
- training entry：`scripts/train_v6_prior_writer.py`。
- evaluation entry：`scripts/evaluate_pi05.py`。
- target split：`configs/libero_24_8_8_v1/`。
- current source policy、tokenizer、data、video和simulation asset的BCI绝对路径均由CLI或`.env.local`
  提供；历史A100路径只作provenance，不原位改写。

旧formal outputs、diagnostics和设计文档保留。旧活动worktree、临时cache或重复本地branch只有在确认无进程、
无未合并唯一改动且远端/Git/artifact已保存证据后才清理。
