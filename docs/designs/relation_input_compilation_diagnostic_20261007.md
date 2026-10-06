# 同一 LoRA 编译链中的预测关系／真实关系匹配检验

2026-10-07。Owner持续自主授权；本合同由progress登记后为唯一active分析批次。
§370关闭的共同冷启动G/F组合保持关闭。本批是训练任务上的有限机制学习，既不是恢复该候选，也不是新的部署方法。

## 1. 要改变的完整方法判断

已有G450的教学手运动与手物关系不可靠，F450虽然读取真实关系，seen仍仅66/144，低于G84和T109。
但F采用自身source H0/v0上的动作残差，G采用完整38处LoRA；两者还具有不同的当前状态输入。
因此66对84不能隔离“准确教学关系能否帮助这套LoRA编译与控制”。仅改Phi、加大动态幅度或移除蒸馏均没有完整依据。

本批只问：在同一G450公共策略、native读取、生成器起点、动作监督和有限学习预算下，
将教学关系由预测字段替换为原训练视频的真实字段，是否使实际编译出的单LoRA获得更好的闭环能力？
这直接影响是否值得继续研究物理信息获取：若准确关系仍不能带来有覆盖的控制收益，不能把更好感知当下一完整修正；
若有明显收益，只建立该编译图在训练任务上的信息利用正例，后续仍必须解释如何从RGB取得并迁移它，不能自动进入fresh大训练。

最近完整证据一起约束解释：ControlCalibrated提高teacher动作读出却将同龄137/91降到122/77；
JointActionEffect中普通FM的父13→24/32是真实正例，增加位移信用J22没有额外收益；
同父C450支持12→71、180步后为156对154，得27失25，不能由更多映射直接推定修复。
后者新增任务仅3–4次曝光，也没有否定充分曝光的95-task fresh。完整历史见findings§143、§271、§296、§370–371。

## 2. 唯一干预及输入图

两臂都从原G450完整权重出发，source/prefix、公共A0/B0及Phi固定；只学习原Omega、RelationalRead和38个ConditionalTarget。
原F完全不加载、不参与动作、目标或loss。没有新头、新decoder、尺度变化、额外物理辅助、公共FM或RL。

```text
P：同一训练RGB/L → 固定G450 native H/X → 固定Phi预测物理字段
Q：同一训练RGB/L → 固定G450 native H/X ＋ 对应视频的既存真实物理字段
                                     ↓
              原Omega → 原RelationalRead(H,w) → 原38处ConditionalTarget
                                     ↓
                 一套(A0+S,B0+M) LoRA → 原source自身RGB/state闭环
```

干预点是Omega输入的整组物理字段，不是F输出、LoRA平均、对象目标表或人工阶段。
P只能使用预测p/R/semantic/presence/joint_type/joint_q/hand_q；Q使用同一字段和单位的GT。
Q只在**原36个训练任务**编译时读取GT teacher，闭环不读取GT自身物理状态；自身policy观测仍是官方RGB/8维state。
GT来自既存label cache，不使用teacher action、reward、terminal、goal/OOI、task embedding或实例名作为模型条件。
真实实体数及存在位属于Q的特权输入，必须明确标注；不能把Q称为合法视频Writer、EMBER成绩或部署上界。
这些字段不含接触、抓取点或所有可控状态，“真实关系”只指已登记字段的恢复值，不叫完美操作知识。

GT按原registry存储顺序放入1hand＋32匿名槽，padding presence为0；不把slot索引或registry身份嵌入网络。
原Omega/Read对匿名实体联合置换不变，故无须使用Phi的Hungarian assignment对齐GT，也不将matching变成条件。
语义仍是原seed20261006冻结词向量投影，字段尺度与原实现不变。

**共同处理缺失末帧。** 原teacher末帧没有完整物理标签；两臂都在native/Phi/Omega之前去掉原采样序列最后一帧，
其余按原stride5和真实frame indices保序。不得只在Q补0、复制前一物理状态、插值、保留一个突变的invalid节点，或偷偷恢复新标签。
只对本批定义共同的教学前缀，不改原数据/视频/旧结果。记录原/使用frame indices和长度；须确认所用GT帧确有已登记字段。
若还有内部缺帧、schema或同步问题，按实际边界回报，不静默多删帧、删task或造值。
因此旧G45084不能充当本批P0；学习前两臂都实际编译/评测，独立显示截帧与输入替换后的起点。

## 3. 为什么这是有界学习检验

令q为上述物理字段，`c,d=Read(H,Omega(q))`，每层按原算子生成S、M；最终执行修正是
`[(B0+M)(A0+S)-B0 A0] h_own`。准确q可能改变相对运动、关系选择和写入方向，
但只有这些变化经S/M响应自身hidden、改变实际动作并提高闭环，才支持继续研究物理信息获取。
动态幅度、可分辨性、非零梯度或更低FM均不能单独建立这项支持。

公共beta和Phi固定，使两臂的视觉特征、基础执行参数及可学习参数集合相同；
FM从实际LoRA回传到Compiler、Read、Omega，不经过被固定的Phi/native参数。
这不遵循新的分阶段训练课程，不为后继fresh方法提供已训练初始化；它是一项限定接口的因果学习干预。
两臂都沿用G450的下游权重，Q存在由预测分布转到GT分布的适应问题；保留学习前后结果而不只看冻结替换。
180步仍是有限窗口，阴性降低“在现有链上只改关系获取即可修复”的优先级，不证明所有GT控制、所有LoRA或重新联合训练不可能。

## 4. 数据、学习与恢复

唯一父点：`relation_grounded_writer_20261006/train/attempts/fresh/checkpoints/macro_00000450`，训练源码85614d9c。
只读复用原`frozen_PEFTfix`的实际Phi/Omega/Read、原ConditionalTarget、source1000、processor、normalization和36-task allowlist。
全部teacher GT直接复用原root的`labels/`；本批不建立query GT、恢复新物理state、增task/episode或扩充数据池。

两臂使用原`continuation2340_spec.json`事件的绝对451…630，共180更新；每步原4tasks×28跨episode queries，
每task20教学条件/560queries，合计720条件/20,160queries/臂。teacher/action query仍不同episode。
相同事件、z/tau/y及完整50×7主FM均值，四task各1/4；完整LoRA均rank128/alpha128、38-target。
不因物理卡数、帧长、两臂并发或失败恢复改变逻辑权重、有效batch和查询流。

只用真实主FM。两臂加载同一父G optimizer/scheduler及绝对时钟450，然后固定公共beta/Phi，
这些参数没有梯度、decay或状态更新；Omega/Read/Compiler保留父moment，按原绝对LR继续451…630。
原AdamW betas(.9,.95)、eps1e-8、wd1e-4、clip1与正常BF16/TF32不变，不重新warmup、不做LR/seed/precision扫描。
原optimizer可以保留冻结参数的既存state，但实际更新集合必须明确核验，不能错按过滤后的参数序号映射moments。
这是改变loss/可学习集合/输入的匹配干预，**不称原G/F训练的exact-resume**。

u0引用原完整父状态并记录两臂输入合同；每臂u90/u180保存完整模型、optimizer/scheduler、sampler/cursor、rank RNG、topology/schema。
物理world1…6按实际吞吐决定，必要时在完整checkpoint边界迁移，记录父world4和新拓扑/RNG来源；不能直接绕过旧校验。
冻结模块应复用同一真实native读取，不建立全teacher H/X的磁盘缓存，也不复制source、labels或dataset。

## 5. 固定读出与裁决范围

四个固定面板为P0、Q0、P180、Q180；每个使用原36-task seen144（states32–35、seed20260928及原state-video映射）。
全部576条新行严格配对task/state/scene/env RNG/policy RNG/teacher ordinal；不拿原完整视频G450代替P0。
这是有限训练诊断面板，每task四条视频，明确沿原映射复用；不是全50无放回paired400，也不选checkpoint。
source、官方render256/model224、双相机rotate、10 flow steps、前5动作replan、settling/horizon/success合同不变。
P/Q的LoRA都在rollout前一次生成，Writer和GT读取在闭环开始前退出，不使用F/live GT/优化/重复教学读取。

主比较Q180对P180，同时报告P180对P0、Q180对Q0、Q0对P0及全部per-task/suite、target24/support12、
breadth、retained/gained/lost、churn及成功集合重合。旧G84、T109、MT93只作完整视频/不同学习合同背景，不冒称单变量配对。
不取四臂最大值，不因u90的训练loss/中途分数改终点；u90只供完整恢复，不增加其闭环或held/Test。

- 若Q学习后有跨task的明显净收益，并改善原target24能力缺口，支持准确教学关系在这套LoRA链上的有限可利用性；
  必须交代效应大小、保持损失、任务集中程度，以及是否足以解释G的原缺口。它不证明RGB可获取、跨task迁移或仅几何字段的单独因果作用。
- 若P/Q类似改善，主要证据是共同FM学习有效，不把收益归给GT。
- 若Q只降低FM、只在个别task换得少量成功，或完整能力仍弱，不以此继续感知头修补、尺度/aux权重扫描或延长该学习窗口。
- 若P/Q都弱或Q更弱，明确保留GT分布适应和有限学习边界；当前物理编译假说继续降权，不再沿此接口追加局部探针保护它。

利用已有真实训练日志与上述物化记录核干预：各臂实际字段来源/帧、可学习集合、主FM到Omega/Read/Compiler的梯度、
c/d和S/M的按条件摘要；不额外开固定query、逐层VJP或参数几何扫描。指标只说明干预是否实际生效。
全部保留实际连续动作/提案与原生谓词；每臂每task init32 full双RGB，其余compact。行为分析须保留不利例与完整任务终点，
不能把接触/移动当抓取或用人工阶段分数代替成功。原scene/teacher/RNG与旧分析器复用。

## 6. 资源、实现和整批交付

唯一root `/data1/user/ymdai/ember_runs/relation_input_compilation_20261007`，所有新增产物只在data1。
预计3–5小时；硬8 wall-hours、12完整GPU-hours、40GiB新增峰，包括工程/失败/profile/物化/评测/冻结/临时。
依据原450共同G/F训练9.12GPUh、本批两臂各180且冻结native/Phi、不再算F，以及原1088读出成本；
只是预算估计，实际profile和现场准入决定物理布局。达到6wallh或9GPUh时若剩余无法在硬限完成，报告具体范围，不减面板或隐含续预算。
checkpoint约5GiB、576 LoRA约12GiB，RGB/raw/临时与工程余量纳入40GiB；不得缓存数百条全层H/X而超额。

实验session承接后独占tracked/Git与实现，main停止并发写入。先独立strg01 data1 quota/个人用量/共享容量及双节点GPU准入；
遵AGENTS总8/空闲≤10时6、单节点6，按实际吞吐并行两臂学习、已就绪物化与评测，不由一臂world收紧整批卡数。
初始P0/Q0和学习可独立调度，最终固定点就绪即读出；无按进度轮询、无逐面板自通知。
有明显显存余量时实测更大microbatch/frame chunk/worker，最多两次首合法宏步的丢弃profile并恢复全部学习/RNG状态；
记录真实吞吐、峰值与停止放大的依据，不能直接沿用保守默认。

用独占codex worktree实现、消费者检查、main集成/push、clean detached新冻结；旧冻结不得热改。
复用原真实模块和现有native/数据/FM/ECP/官方评测器，只增加本批必要输入选择、冻结学习和训练任务准入编排；
不恢复原F长训路径、不另造评测器、不在大run.py堆新科学逻辑。GT入口必须有固定训练allowlist，不能进入held或一般部署。
工程实际检查聚焦GT/pred分支信息墙、共同截帧、完整A/B消费者、梯度与冻结集合、配对及恢复；不加hash/全树完整性/低位一致检查。
专用入口与临时注册在整批结束后退役，保留原件、失败、完整checkpoint、Git和frozen；不留平行fallback。

工程违约可按原范围修复并新push/freeze，科学/预算边界回main。完整报告须区分原训练代码、新学习代码和读取代码。
只在整批完成或真实边界时一次可靠回报main并交回窗口；main收到后直接消费原件、裁决完整方法与接续。
本批没有自动fresh/更多任务/续训/held400/controls/Test资格；Owner持续自主目标仍是可部署EMBER稳定大幅超过强MT。
