# Video Functional Writer：任务语义轴与执行条件化功能信用

## 1. 当前授权、目标与理论边界

Owner于2026-09-11接受本轮综合历史形成的候选，授权高效自主推进设计、实现、学习及证据驱动修正。
本设计由progress登记为唯一active design；旧Horizon/C/无变化参照、95-task均不恢复。
当前goal是有益的视频特异性：正确教学内容及内部顺序在**一次生成的完整LoRA**中产生可重复闭环增量，
在同task不同视频、执行初态、相邻checkpoint上保持，并迁移到固定validation8。
暂不以145/400、SFT或source总分作为额外硬门槛；绝对分数始终报告，错误条件退化、内部差异和辅助头成绩不能替代收益。

原专家推导及最终修正原样保存于[初案](review_materials/video_specificity_20260911/expert_joint_functional_proposal.md)、
[机制修订](review_materials/video_specificity_20260911/expert_functional_mechanism_revision.md)、
[最终信用分配](review_materials/video_specificity_20260911/expert_final_functional_credit.md)。以下为实际执行选择，专家意见不是额外权限来源。

历史约束：v5.2两种同图配方同时改变更新数、LR时标和任务混合，不能单因归因；v6/GOMQ有更高单项结果。
完整H、C消费、无变化参照均未保证有益过程；J2已经共同训练Program及映射，仍不代表共享获取成立。
本候选新增的是**执行query直接查询同一视频表示的真实动作信用**，不是“首次联合训练”。
其理由是降低有限优化时必须同时协调的映射数量；不声称精确VJP丢失信息，也不声称监督排除了共同策略解。

## 2. 输入与唯一部署数据流

沿用generic pi05建立的冻结source、source-only normalization、固定train24/val8/test8及合法原生prefix。
首轮K1、stride5保留末帧、teacher agentview；部署policy仍双相机官方处理。无teacher actions/state/terminal/reward/taskID输入。
训练teacher0–15，动作queries16–41，动作诊断42–45，held正确视频46–49；同task跨episode。
每update四suite各一task、每task一条K1×64queries，task权重1/4，总256queries；暂不增加meta tasks或RL。

```text
exact language + RGB video
 → frozen native prefix：contextual task tokens及真实patch Z
 → Action Expert + reading Meta：每帧完整R[T,50,1024]
 → 逐task-token视觉grounding z[T,L,256]
 → 相邻两帧的task-token query读取两端完整H → e[T,L,256]
 → 两轮：逐帧语言轴交互 + 每token过去单向时序交互
 → E[T,L,256]
 → 两层target/rank Compiler → 完整native D → 唯一38-target rank16 A/B
 → 冻结policy依据自身图像、state、noise与flow time生成动作
```

H首次压缩由前后两帧z共同提出query，一次读取拼接的100个H内容；H位置与前/后端身份只进入K路由。
首帧使用self-pair，无虚构帧；H不是已经采样出的未来轨迹，不预设h差对应物理时间差。
grounding使用实际contextual task token查询非task的有效图像patch；不把pad、state或文件身份混入视频内容。
e由当前语义、前后语义与H读取的融合构成，所有时序处理保留L轴；L是准确语言token位置，不是人工物体标签。
轴向block采用标准attention/MLP，默认width256、heads8、两轮；时序past+self，位置采用原frame index/5的QK RoPE。
删除旧四组H对应、逐H GRU、长程回写及nochange通路；保留完整H原则，不沿用其旧具体实现。

Compiler读取flatten(T,L)内容，target+rank身份query，无额外静态语言query仿射。
按每条video的token数给-log(T×L)先验，videos独立编码后集合读取，输出不平均多套LoRA。
输出复用完整native D：A=A0+D_A GELU(U_A c)，B=D_B GELU(U_B c)，不加carrier/span或第二adapter。
部署仅保留生成的完整A/B；辅助头、E与teacher video不参与后续闭环。

## 3. 辅助动作读出与真实FM

执行端query取**未挂载生成LoRA的冻结source**在真实FM输入下、action_out_proj前的完整[batch,50,1024]激活。
使用同一执行观测、action chunk、noise、flow time得到source速度u0与学生速度S；noise/time沿用现有keyed逻辑batch合同。
q0及u0无梯度；动作与query仅到训练读出/执行policy，绝不成为E或Compiler输入。

辅助头width256、8heads：r=Attention(project(LN(q0)),E,E)，
T=u0+W_o[GELU(U_q LN(q0)) ⊙ U_r r]。W_o无bias、零初始化，其余读取/投影正常非零初始化。
允许完整语义及过程Value，**不强制中心化、不扣除学生残差、不惩罚错误视频**。
常量视频读出仍可形成共同动作修正；这是存在的竞争解，不能把通路限制当视频资格。
所有损失按当前官方实现的有效7动作维度和完整50-H求均值（数据沿用末动作补齐），source的32维pad不贡献loss。
目标为原生FM y=noise-actions，辅助头承担完整source残差而非已识别的“动态专用误差”。

## 4. 分组梯度与初始化

参数严格分组：phi=视频encoder+Action Meta；eta=辅助头；psi=Compiler+完整因子decoder。

| 参数 | 真实LoRA FM L_C | 辅助真实FM L_R | 蒸馏L_D=mean(S-stopgrad(T))² |
| --- | ---: | ---: | ---: |
| phi | 1 | mu=1 | 0 |
| eta | 0 | mu=1 | 0 |
| psi | 1-rho | 0 | rho |
| source | 0 | 0 | 0 |

首版辅助监督从step1开始，rho在前32次更新为0，随后到step100线性增加至0.25并保持。
这是固定工程选择，不根据wrong/shuffle结果调整或触发蒸馏；不声称辅助教师已获因果资格。
纯FM参照不构造辅助头，mu=rho=0，phi/psi均接完整L_C。辅助模块在writer/Meta初始化后构造，避免扰动匹配初始值。
LoRA保持A0/B0合法identity，D零初始化；不要求第1步encoder已有非零LoRA梯度。辅助输出W_o=0也只先训练出口。

沿用分块VJP：同一参数版本算E、完整A/B与policy cotangent；L_C与L_D分别保留。
phi仅接J(G,E)^T v_C加L_R；psi接J(G,psi)^T[(1-rho)v_C+rho v_D]。
不能混合cotangent后一路传回encoder；不对整段Compiler使用no_grad来代替停止psi梯度。
各分支完成后才统一optimizer.step；只缓存冻结prefix，Meta响应/E不跨update复用。
LR=3e-5，AdamW betas(.9,.95)、eps1e-8、wd1e-4，8updates warmup后保持，clip1；正常BF16/FP32更新。
样本/条件权重、各loss单独报告；梯度路由代理的标量和不能作为性能。

## 5. 首轮对照、预算与裁决

先实现一个canonical model、一个trainer、一个evaluator。首轮主方案与同新表示纯FM参照fresh训练，
配套无序视觉参照用于验收过程增量；不默认同时开启四条新训练。
无序参照保持相同参数及完整帧信息：每帧self-pair、无端点时间路由、无时序position、全帧集合attention，
Compiler时间路由也关闭，严格帧置换不变；不把冻结重复首帧当公平训练baseline。
是否补旧表示+新信用只在主方案/纯FM出现需区分的结果时决定，旧off数字只作历史参照。

先做真实小型profile定物理batch与每段约1h对应更新量，首段至少两个50/100倍数相邻节点。
最长训练视频实测蒸馏阶段每64queries约20s；现有3卡首段登记50/100更新（约1h），在正式学习前固定。训练初期loss不能替代闭环。
开发train24 states32–35、teacher pool46–49为固定机制96；correct/other由统一video schedule登记，有限池复用单独标记。
开发输入干预只对训练tasks、无梯度，用于区分内容/静态/过程；旧已用过的shuffle不是sealed最终证据。
validation沿用官方strict400全50 teacher无放回与state-video映射，开发seed20260911。correct/other在有信息量节点评测，不等145门槛。
辅助读出仅作为单独标记的诊断控制器，其闭环绝不计入EMBER部署分数。

### 首段后的有界曝光登记（2026-09-11，续训前）

首段主方案/纯FM训练correct26→35与30→41，仍在获取；validation主方案69→72、纯FM69→56。
以主方案为匹配无序参照的对象，不宣称辅助优势已被证明。主方案从完整100 checkpoint按原config、原world3及
GPU/NUMA拓扑exact-resume到200/300；frame_set分两段，fresh检查50/100，再原拓扑exact-resume检查200/300，两者最终均76,800queries。
只改变有序/无序表示，保持主方案辅助与蒸馏、采样、LR和所有信息墙；不追加pureFM续训，不登记300以后的更新。
200/300沿用同train96、validation strict400 correct/other及固定映射；CLI checkpoint节点与
`runs/analysis/video_functional_20260911/bounded_300_registration.json`记录本段覆盖原首段节点列表。
原固定留出动作诊断仅0/100；不据后续训练FM宣称留出动作平台，后续以真实训练/validation闭环和相邻证据裁决。
本窗口确保不把弱训无序参照或两个早期节点直接当资格/平台；下述完成口径不变。

### 阶段goal完成口径（登记后不按新分数移动）

1. 选定单checkpoint的correct与same-task-other，均在validation400相对匹配训练无序视觉参照有正收益；
   correct-primary按task为cluster的paired bootstrap95%下界>0，other方向一致，至少两个suite有净收益；
   同时报告source和所有任务，不以reference退化/弱训制造差额。若reference尚在明显学习，先匹配其充分曝光。
2. 相邻single checkpoints的correct动态增量同向，不依赖单个峰值；报告per-task/suite、breadth、retained/gained/lost、churn/Jaccard。
   阶段允许未达到长期breadth或145门槛，但不能只有一个task贡献全部收益；不以checkpoint union/fusion取得资格。
3. 方法及选点冻结后，按事先登记、未用于开发选点的state-video重新配对映射做correct/other/cross-suite-wrong/shuffled/reversed配对确认。
   sealed最终映射固定seed20260912，仍使用同一合法50视频池，不声称获得新视频。内容与顺序破坏均须损失一部分正确视频收益；不要求每个task、每个wrong都为零。
   主要correct-wrong与correct-shuffled差额按task-cluster bootstrap95%下界>0，reverse作为方向特异性支持单独报告。
   时间重参数化仅在保持stride5和真实帧合同允许范围内预注册，不用新增干预反复选模型。
4. train任务上的机制获取仅为里程碑，不能替代validation迁移；验证和Test均无梯度，Test继续封存。

所有paired bootstrap固定task-cluster重采样20000次、seed20260911、双侧percentile95%区间；每次保留整task配对rows。

若辅助头有收益而LoRA没有，聚焦当前编译/优化接口，不能直接宣布rank容量不足；若两者没有收益，
区分表示、读取函数类与优化，不能直接宣布输入无信息。若纯FM已足够，优先减少无收益辅助复杂度。
若收益曾形成后丢失，研究保持；不恢复旧C/95任务或无依据的rank/seed/LR/dtype小扫。

## 6. 工程所有权与保留范围

新video.py拥有encoder+Compiler，function_reader.py拥有辅助头，function_credit.py拥有原生FM功能读出及cotangent；
supervised.py编排同condition重放，runtime.py统一构建，training.py统一调度/checkpoint。
保留native.py、native_factor.py、data/schedule与官方evaluator；退役horizon/relation/semantic/frame_evidence及仅测旧图的测试。
保留现有脚本文件名train_horizon_writer.py/materialize_horizon_writer.py作唯一入口，更新schema拒绝旧checkpoint，
旧科学模型只由原detached commit运行。没有平行fallback；pureFM和frame_set是同一模型合同的必要对照配置。
新代码与测试在独立codex工作树完成，合main并push；formal使用clean pushed detached冻结工作树。
精确GPU、quota、命令与峰值在launch记录中，不把动态资源状态放AGENTS。

## 7. 有界冻结表示读出拟合诊断（2026-09-12，额外拟合前登记）

主方案200→300两组validation视频均74→32，50/100/200匹配frame_set未获得可信有序增益。
已有训练同批与train24留出动作诊断均显示辅助读出只小幅改善source，且明显落后LoRA学生。
因此当前不成立“好教师已经学会、只有LoRA编译失败”的前提。先用一个低成本干预检验现有读出是否还能学习，
不追加canonical Writer更新，不用该诊断选择部署checkpoint或宣称视频因果资格。

- 诊断锚点固定为main200（退化前节点，非qualified/selected checkpoint）。冻结source、Meta、视频encoder、
  Compiler、完整LoRA decoder；缓存原生source执行query、source速度、FM target及视频E，只让现有ExecutionVideoReader学习。
  这不是Writer分阶段训练课程；输出仅为不可部署的独立读出probe，不回写原checkpoint。
- 仅train24。每task固定一条训练池video（global task modulo 16）、64条动作queries（episodes16–41，query seed=20260912+task）；
  同task跨episode、stride5、exact language和source-only normalization不变。固定缓存共1,536条独立query条件，后续轮次是复用，不能称新增数据。
- 留出沿用原train24动作诊断：episodes42–45、每task32queries、seed20260908+task，teacher46–49按task modulo4。
  这些768条queries从不产生梯度。validation8/Test均不访问；这不是held benchmark动作学习。
- 从main200读出权重出发，fresh AdamW与原LR3e-5/betas/wd/clip、8updates warmup。每epoch每task恰好一次，
  各suite内按seed20260912+epoch打乱，四suite各一task组成一次等权更新；固定64epochs=384updates，无LR/seed/架构扫描。
  这是一项“额外读出拟合”干预，包含fresh optimizer与固定缓存复用，不声称隔离了某一个优化器超参数。
- 只报告epoch0/4/16/64的support/held逐task FM、冻结source/原LoRA学生FM与最终probe状态。所有报告节点保留，不按结果延长或选点。
  同时作无梯度的cross-task E替换（固定选不同suite的donor）：这是表示接口依赖诊断，不是raw-video wrong/shuffle控制，
  不替代视频内容或顺序因果证据。正确条件改善才有意义，donor退化本身不是目标。
- 判读：若固定表示下读出能改善留出动作，说明该接口存在可学习修正，值得进一步查原共同训练中的获取和传递；
  仅support改善说明固定样本拟合不足以泛化；两者都弱仍不能区分函数类与优化，也不能断言视频无信息。
  无论结果如何，FM不能完成当前closed-loop goal，probe不会直接进入部署。
- 先做4个训练tasks、各8条support/held queries、1epoch的smoke，只验真实native/cache/梯度机制。
  正式诊断1GPU，CPU缓存不落盘，预计持久增量小于1GiB（含约217MiB隔离checkout），运行占卡计入Owner全局上限。

`ember.writer.reader_diagnostic`只拥有此诊断编排，复用现有runtime、数据、视频与FM函数；不增加第二Writer或评测器。
它是有明确问题的临时研究入口，由当前任务维护，诊断结论被后续方法吸收或该问题关闭后退役，证据由Git及probe artifacts保留。

## 8. 保留辅助真实FM、关闭蒸馏的有界比较（2026-09-12，学习前登记）

第7节诊断结束：读出held FM .149662→.140251，23/24task改善，但24/24仍落后原学生（均值.110025）。
已有同批三项MSE表明main/frame_set后段蒸馏与真实FM在**预测空间**的梯度cos≈−.15；不据此推断参数/Adam
冲突或断言蒸馏损害迁移。弱教师也可能产生正则化收益，且旧pureFM100验证56低于main72；旧两臂同时改变了
辅助表示监督与蒸馏，无法区分两者。此处只检验一个明确因素，保留已经对齐的表示与完整LoRA架构。

**单一变化：rho恒为0；mu仍为1。** phi接受完整L_C+L_R，eta接受L_R，psi仅接受L_C。
辅助头职责收窄为视频表示的直接真实动作信用，不向Compiler/nativeD提供教师目标。数学上L_D在rho0时
只可作为诊断值，不产生蒸馏cotangent。复用现有支持rho0的canonical trainer，无新参数、模块或并行fallback。

- 从step0 fresh Writer/Meta/reader及合法identity LoRA开始，fresh optimizer/scheduler/sampler/RNG。
  不继承main200或额外拟合reader；原训练seed7、LR、任务/视频/query采样、source、rank16、宽度、K1与全部信息墙不变。
- 固定训练上限200updates=800 K1条件=51,200queries，保留100/200完整checkpoints，不追加300。
  两节点均做原train96与validation strict400的correct/other，原seed/state-video映射、配对推理及bootstrap不变。
  留出动作诊断0/100/200，均无梯度；它只能定位，不能选择checkpoint。
- 当前canonical配置`configs/pi05_video_functional.json`登记本配方；旧rho.25配置由原run_contract及冻结commit保存。
  现有frame_set配置同步同配方供未来匹配资格参照，但**本段不自动启动第二条训练**。
- 先与同曝光main100/200比较以辨别去蒸馏的实际收益/代价，pureFM100作辅助表示信用的有限匹配参照。
  必须报告全部task/suite、source、breadth、R/G/L/churn/Jaccard与相邻证据；不因teacher FM或输出方向选模型。
- 本段不能独立声称过程资格：若correct与other在两个节点均相对原main没有正向总分变化，且main中已有的
  局部收益/保持亦未改善，则不为本配方追加匹配frame_set或延长训练；记录所检验因素的负结果。
  若出现值得验证的保持/迁移改善，再登记同rho0、匹配曝光的fresh frame_set资格参照，沿第5节原有
  correct-primary CI与相邻/同视频要求裁决；原rho.25无序分数不能冒充这一匹配参照。
- 不开展reader额外更新、LR/seed/容量扫描、RL、95-task或sealed最终controls；Test继续封存。
  资源与正式命令另写launch contract，仍要求clean pushed detached代码与Owner全局用卡上限。

此比较可能只改善共享动作修正，也可能失去原蒸馏的正则化收益；只有之后的匹配行为证据才能判断有益过程是否形成。
第7节临时诊断入口随问题关闭退役，原状态与证据留在Git和artifacts，不成为新的课程或部署路径。
