# EMBER task plan

更新时间：2026-08-25。

## 当前目标

owner已正式许可推进ECP Native-Factor Compiler。Phase G1 task-local free-code capacity oracle已经通过four-arm strict250 Gate；
当前进入Phase G2 Natural Program。G1只证明真实native banks与signed pooling存在强rank4 residual，不证明deployment Writer或
shared Program-to-attention映射成立；单独完成任一阶段仍不代表整体项目goal完成。

## 当前G1里程碑

- [x] 从`main@13ca366`建立隔离的`codex/ecp-native-factor-g1`实现面；
- [x] 接通真实38-target native X/Y hooks、四类output banks与跨chunk/video边界状态；
- [x] 接通task-local free-code signed pooling、rank4 outer products、small-core SVD和唯一rank12+4 rank16 materialization；
- [x] 接通四类G1 loss、optimizer、checkpoint、静态task-LoRA evaluator和four-arm Gate report；
- [x] 用最小真实CUDA forward/gradient/materialization smoke证明全部free variables有有限非零梯度、Action Meta实际未加载、
  source/Stage 0无trainable parameter且checkpoint为38-target/76-tensor rank16；
- [x] 完成首轮代码/测试/diff审查，集成到clean pushed`main@9a6f434`并从detached worktree执行formal；
- [x] 完成首轮5-task optimization、唯一rank16 strict250与four-arm Gate：`88/250`、breadth3/5、Goal/Long 0，结论non-pass；
- [x] 完成read-only output span与paired response projection，定位scalar q-output pooling的列空间上限，并以`109/250`、
  Goal/Long仍为0验证被排除方向具有闭环后果；
- [x] 实现真实q八头独立signed measure修正，并通过CPU合同及task93真实forward/gradient/materialization smoke；
- [x] 完成q-head修正diff审查、全量CPU回归、main集成与detached formal worktree；
- [x] fresh重跑q-head 5-task formal optimization、strict250与同一G1 Gate：`84/250`、breadth3/5、Goal/Long 0，仍non-pass；
- [x] 用真实bank稳定子空间投影和paired strict250把最早接口定位到free-logit可达优化：`94/250`、breadth5/5、Goal/Long非零，
  但retention`22/43`；
- [x] 实现reference-projected positive/negative simplex初始化与frozen native chunk cache，并通过task93真实一步gradient/materialization
  profile和140项CPU回归；
- [x] 保留optimizer前step0，完成latest-only五task formal与strict250：`100/250`、breadth4/5、Goal3/Long0、retention`22/43`，
  Gate non-pass；
- [x] 以paired fixed50证据定位set-valued选择接口，并实现每task在carrier/latest/independent/earliest中选最强verified member；
- [x] 通过141项CPU回归、task90 zero-residual initialization-only与task94 independent真实gradient/materialization smoke；
- [x] 完成set-valued clean formal与strict250：`111/250`、recovery`1.015`、retention`34/43`，但breadth4/5、Long0、仅3/5高于carrier；
- [x] 由task94 minimum direction cosine `0.978/0.883`定位FP32 inverse-scatter数值失真，并用真实FP64 smoke恢复到两侧
  `>=0.99999988`；
- [x] 完成FP64 signed-solve diff审查、全量回归、main集成与detached formal worktree；
- [x] fresh生成五task step0并完成single-checkpoint strict250：`116/250`、recovery`1.090`、retention`35/43`，但breadth4/5、
  Long0且仅3/5高于carrier，Gate仍non-pass；
- [x] 用paired response只把task94的action-in target恢复为known-success independent mobile，Long从`0/50`变为`1/50`；完整
  counterfactual为`118/250`、breadth5/5、4/5高于carrier、retention`35/43`，定位whole-vector action-in
  `span(column_space(W),bias)`上限具有独立闭环后果；该privileged替换不是G1 candidate；
- [x] 实现action-in按其native input width形成32个真实32D Y blocks的独立signed measures，并通过142项CPU合同检查；
- [x] 完成task94真实forward/gradient/materialization smoke：32个output blocks均stable rank32、两侧minimum cosine
  `>=0.99999988`、全部26,208,000个output logits有非零有限梯度、Action Meta 0、唯一rank16，峰值约29.77GB；
- [x] 完成全量diff审查、142项CPU回归、clean pushed `main@31f0053`集成与detached formal；
- [x] fresh生成五task step0并完成同一strict250 Gate：`114/250`、逐task`35/31/45/2/1`、relative recovery`1.060`、
  breadth5/5、Goal2/Long1、4/5 task高于carrier、retention`35/43`，G1正式通过；54/54 shards、250/250 rows、18/18 workers正常，
  Action Meta关闭且输出为single complete rank16。

## Phase R：全仓库理解与资产映射（已完成）

这是开始实现前的必需orientation，不是让owner重新解释项目。全程先只读，不启动GPU或修改科学状态。

1. 按`AGENTS.md`顺序完整阅读owner requirements、plan/findings/progress、concept、research history、专家原文和active design；
2. 用自己的话写清EMBER的输入、输出、zero-interaction目标、信息墙、成功标准与ECP Native-Factor因果链；
3. 对照历史分数解释哪些结论已成立、哪些接口失败、GOMQ/PECS/v24/人工process为何不属于当前主线；
4. 阅读并映射`configs/`、`src/ember/ecp/`、`source_sft/`、`expert_manifold/`、`writer/`、`reward/`、`pi05_eval/`、`scripts/`与
   `tests/`的职责、入口和当前保留/缺失能力；
5. 只读盘点ignored `data/`、`models/`、`runs/`、formal checkpoints/raw rows、task expert bank、Stage 0、carrier/mobile evidence、
   fold manifests与evaluator，不复制或删除资产；
6. 核对Git HEAD/origin、worktree/branch、当前无运行任务假设，以及GPU/storage约束的live检查方式；
7. 向owner给出一份简洁的“我对EMBER和当前仓库的理解”：目标、架构数据流、已有证据、可复用资产、实现缺口、后续阶段和G1
   首步。若发现文档、原文和代码冲突，先列出证据并修正文档，不凭摘要猜测。

Phase R已完成，`HANDOFF.md`已消费并删除；owner随后已明确许可进入G1。后续不重做orientation或退役路线考古，整体推进
不设阶段工期和修正次数上限，但每次scientific修正必须有机制证据。

## Phase G1：Native-factor capacity oracle

### 目的

不训练共享video-to-LoRA映射，只回答：自然视频在PI0.5各LoRA目标产生的native input/output vectors，能否通过小型task-local
selection形成强闭环mobile rank4 residual。

### 输入与冻结

- fold0 held5 natural teacher videos；
- frozen current Stage 0 v3；
- known successful task experts/mobile-rank4 projections；
- frozen carrier43、source与原始PI0.5；
- 不读取validation/test，不制作新数据。

### 实现工作

1. 为18层q/v与action-in/out增加真实linear input/output hooks，probe轴保留；
2. 输入候选使用`n_A=(video,frame,probe,horizon)`的真实`X`，输出候选使用额外带`type in {abs,adj,init,goal}`的`n_B`与真实
   `Y^type`；不得把`X`复制到无意义的type轴；
3. 构造absolute/adjacent/init/goal output banks，按frame chunk在线读取与累计；online softmax除running maximum、normalizer和weighted
   sum外，还须按video保持首帧、末帧及跨chunk previous activation，并与non-chunked reference数值等价；
4. 实现two-branch signed pooling、per-target scales、rank4 outer products与small-core SVD canonicalization；G1允许直接优化task-local
   selection logits/weights，不要求共享Program-query到candidate-key映射；
   首轮证据已证明对q的整条2048维value强制共享一个scalar measure会把所有输出限制在base-weight的1024维列空间；当前修正保持
   原candidate index和真实Y值不变，只按PI0.5原生八个query heads分别归一化signed measure并拼接，v/action-in/action-out不变；
5. 实现每task free-code optimizer，只优化4 rank queries、event weights、输入/输出pooling weights或logits和scales；`K>1`时固定
   `beta_k=1/K`并做video内assignment归一化，`K=1`为identity，不学习video reliability；
6. 明确走纯Native Stage 0 observer加载路径，在run contract与最小真实forward中核对实际module/trainable parameter，证明Action Meta
   未被旧loader装载；
7. 接通global-member effect、sensitivity-normalized update、independent functional与carrier-preservation loss；
8. 生成唯一rank12+rank4 complete adapter，接入现有strict evaluator。

先做最小真实forward/gradient smoke，再进行5-task optimization与strict250；不增加通用框架、checksum或与Gate无关的测试。
G1通过只证明native banks加signed pooling形式存在强rank4 residual，不证明deployment Writer或共享Program-to-attention映射成立。

### 通过门

同一strict250比较carrier、direct latest、known mobile projection和native-factor free-code，必须同时满足：

- relative oracle recovery至少0.70，按43/110参考约`>=90/250`；
- breadth 5/5，Goal与Long均非零，至少4/5 tasks高于carrier；
- carrier successes保留至少33/43；
- single rank16 adapter、strict pairing、无second adapter。

失败先做read-only span/response分析，再按最早失效机制修正hook、bank、pooling或优化并复评。不设修正次数上限；每次修正必须
有新证据和明确假设，不能变成slot/width/seed小扫。只有充分尝试后持续证明native basis不可达，才停止Native-Factor。

## G1通过后的固定序列

### G2 Natural Program

meta56+target-fit19，K均匀采样1/2/4；训练owner-specific language/scene、ordered events与K aggregation。meta-held15+target-held5检查
same-task separation、probe stability、event non-collapse、full相对endpoints的held loss增量，以及每video event alignment、variance、
uncertainty、`K=1` identity与video集合置换不变性。修正应限于证据定位到的native capture、event grounding或owner-specific
language/scene，不设次数上限。

当前实现节点：

- [x] 固定Program schema、owner-specific language/scene readers、两条antithetic probes、ordered event decoder和monotonic canonical alignment；
- [x] meta56/held15与target-fit19/held5角色、K=1/2/4 task-equal schedule、跨episode video/action监督和`beta_k=1/K`；
- [x] 95-task BDDL progress/rising、真实simulator contact及terminal contact mask的CPU label authority；
- [x] action/progress/predicate/contact/scene、same-task event、probe、speed/crop、contrast、occupancy/tau/uncertainty losses；
- [x] 每条video完全独立native encoding；真实K4检查把集合置换误差从`0.132`降到`2.38e-7`，K1保持exact identity；
- [x] 真实K4 forward/backward使84/84 trainable parameter tensors进入optimizer state，Action Meta module/parameter为0，峰值约18.85GB；
- [x] formal前复核已消除rank-local顺序导致的辅助loss不等权：每个task都计算一次speed/crop robustness，contrast对每task使用固定数量、
  两种fit role各半且与rank/world-size无关的language negatives；action与全部动态标签共享唯一action-episode query index；
- [x] clean pushed `main@141a110`的macro10 formal与held20 Gate完成；除full-vs-endpoints仅改善`0.0226%`外其它资格项全部通过，
  因而未进入G3；read-only消融把最早接口定位为training decoder的静态旁路，而非native动态捕获。
- [x] 移除`P_lang/P_scene`到`P_process` fusion及时序heads的直接加性旁路，保留独立scene head；clean pushed
  `main@30b98ef`的fresh macro10仍non-pass，full相对endpoints为`-0.0570%`，one-event `0.30`、probe margin `0.65`。
- [x] read-only temporal与event-grounding诊断定位到G2训练侵蚀已有Stage 0 v3结构：初始event/owner relative RMS
  `0.06069/0.36992`降为raw encoder `0.02601/0.22824`；这不是query-time weighting或静态旁路残留。
- [x] 冻结Stage 0 v3 observer，只训练新的Program readers/fusion/alignment/diagnostic heads；task92真实K4 smoke确认46/46个新增
  parameter tensors进入optimizer state、39个encoder tensors保持逐tensor不变，run contract确认native observer/source policy
  trainable均为0、observer处于eval且Action Meta为0。
- [x] 从clean pushed `main@db84a50` fresh训练并exact-resume到macro20；同一held20 Gate中same-task、K1/K4、active-event范围继续
  通过，但macro10/macro20的full相对endpoints分别仅`+0.0051%/-0.0207%`，macro20 probe margin为`0/40`，因此仍未进入G3。
- [x] macro20无梯度层级诊断确认frozen Stage 0 raw event/owner结构保持`0.06252/0.36771`，full/endpoints差异也真实存在；但共享
  `Linear(128,1)` owner score对固定38-owner轴严格置换不变，owner entropy为`0.99898`，action prediction temporal std仅
  `0.00173`而target为`0.32725`。继续训练没有修复最早readout接口。
- [x] 只把training-only temporal owner score替换为38个固定语义owner各自的shared-across-task linear query；queries从旧共享
  Linear完全相同的向量初始化以保持其余head的RNG序列，不修改Stage 0、
  Program schema、probe处理、scene head、数据、loss、seed/LR/slot/width或Gate。真实K4 profile确认owner-query gradient norm
  `0.01827`、一步后query rows已分化、46个Program tensors trainable、observer/source/Action Meta trainable均为0，peak约10.02GB。
- [x] 集成并推送owner-structured readout，从clean detached commit fresh训练并exact-resume到macro20；full相对endpoints为
  `+0.0158%/-0.0340%`、probe均为`0/40`，query分化从`1.58%`增至`2.94%`却未改变shared解，故该scalar selection不是充分修正。
- [x] 无梯度Stage0-transfer反事实确认raw process+既有action head把absolute action loss降至`0.20767`，但full增量仍仅`0.2467%`；
  最早接口不是简单旧head丢失，而是absolute MSE的trajectory-mean解未约束query-time residual。
- [x] 保留absolute action/progress并新增等权query-centered temporal residual MSE；真实K4 profile确认两个新loss有限、owner-query
  gradient非零、frozen observer 39 tensors不变且Action Meta为0。
- [x] temporal-residual objective由clean pushed `main@68f8705`完成并从fresh训练到macro10；held20的same-task、K1/K4、event范围
  继续通过，但full相对endpoints只改善`0.0381%`、probe margin为`0/40`，故仍未进入G3。
- [x] 冻结该轮Program做readout/label/optimization可证伪诊断：full-owner temporal readout相对endpoints可产生`15.17%`改善，证明
  动态信号可读；tied与independent query初始化曲线近乎相同，cross-episode监督可识别；而旧macro10实际只有10次Adam更新，
  frozen readout在10/60步几乎不动、200/500步才明显下降。最早接口因此是optimizer cadence，不是新的Program架构缺口。
- [x] 保持模型、数据、loss、K和Gate不变，实现每macro 10个role-balanced optimizer steps：常规2 target+2 meta，旋转尾部1+1；
  scheduler与resume按真实step计数。单卡与world4真实profile均完成finite forward/backward/materialization/checkpoint，world4实际聚合
  2+2任务、46/46参数进入Adam、Action Meta/source/observer trainable均为0。
- [x] 从clean detached `main@49e7769` fresh运行macro10/100 optimizer steps并复评同一held20 Gate：full相对endpoints改善
  `0.3080%`、probe `13/40`，其余资格项通过；相对旧10-update结果动态增量约`8.1x`且17/20 held task方向改善，但仍明确non-pass。
- [x] 冻结该checkpoint，用12个role/K平衡fit task做gradient diagnostic且held gradient为0：Program full/endpoints差异真实存在，
  temporal梯度没有被强方向性抵消，但在Program process/temporal decoder上分别比non-temporal小约`10x/21x`，prediction temporal std
  仍比target小约`93x/203x`。最早接口是近常数readout造成的temporal gradient starvation。
- [x] 按同一commit、world4 topology与run目录exact-resume到macro20/200 updates；held full增量升至`8.6878%`、probe升至
  `36/40`，fit prediction temporal std增长约`9x/30x`，验证readout学习时标；但所有K>1条件坍为one-event，Gate仍non-pass。
- [x] 用K分解与fit-only no-gradient alignment反事实定位根因：K1保留平均`6.42` events，K2/K4 local presence未坍缩但DP将约
  `6/8` path mass集中到单一canonical slot；boundary-only锚点恢复K>1为3 events且不损失视频增量。
- [x] 只把K>1 monotonic DP改为首尾canonical边界锚定、保留中间stay/skip与既有content/time score；真实K4 profile
  已完成4-video/102-frame forward、backward与optimizer step，active events为2、one-event为0，Action Meta module/parameter均为0，
  source policy与native observer均冻结；全量合同测试`155 passed`。
- [x] 从clean pushed `main@c1493a1`的detached frozen worktree fresh训练并exact-resume到macro20/200 updates；held20 Gate全部通过：
  full相对endpoints改善`22.2047%`，median active events为4、one-event为0，probe为`38/40`，same-task、K1 identity和K4 permutation
  均为1.0，tau violation仅`0.00357`。冻结`macro_00000020` Program进入G3。

G2只有一个canonical入口`scripts/train_ecp_natural_program.py`。模块ownership固定为：`natural_program.py`拥有部署Program schema与
Pass-A图，`natural_program_data.py`拥有fold/schedule及跨episode packing，`natural_program_labels.py`只拥有training-only派生标签，
`natural_program_objective.py`拥有机制loss，`natural_program_authority.py`拥有run provenance与信息墙inventory，
`natural_program_training.py`拥有macro/checkpoint编排，`natural_program_train_step.py`拥有唯一role-balanced optimizer update，
`natural_program_gate.py`拥有无梯度held20 Gate。Stage0 encoder、通用
checkpoint和既有video/action stores只复用，不复制。G3复用并冻结通过Gate的Program schema/model；G2 trainer、label sealer和Gate在
formal结论固化后仅作为可复现实证runner保留，不成为平行Writer或deployment fallback。

### G3 Frozen-Program shared compiler

冻结Program，用自然videos与95-task/118-member evidence训练共享Program-query到native-candidate-key的content attention、signed pooling、
scales和bounded K correction，禁止task/frame查表。依据G2证据可从均匀跨video权重初始化有界learned `beta_k`或其他bounded correction，
并防止单条video覆盖其余videos。held5要求full
`>=60/250`、breadth`>=4/5`、retention`>=33/43`、Goal/Long至少一项非零、相对language和first+final各`+5`、same-task
retention`>=80%`。可以按mapping/compiler/critic证据修正，不设结构版本上限，但无机制差异的小变体不算推进。

- [x] 以通过G2 Gate的`c1493a1/macro_00000020`为唯一frozen Program authority，先复用G1真实native X/Y capture、四类output banks、
  action-in native blocks、small-core SVD和rank12+4 materialization；实现共享content-derived query-key signed attention、target scales与
  由uniform初始化的bounded K correction，不保留task/frame free-logit路径。
- [x] 完成上述共享Pass B、95-task/118-member与set-valued critic的CPU实现合同：input/output candidate索引分离、跨chunk视频边界、
  per-video单位measure、K=1 identity、K=4置换不变和无task/frame free logits均已有定向回归；真实GPU机制smoke仍属于下一项。
- [x] 完成最小真实forward/gradient/materialization、真实cross-episode action-flow、same-task consistency、长K4显存与信息墙检查；
  三步profile均exit0，Action Meta 0，source/Program trainable 0，唯一rank16被policy实际消费。
- [x] 接通冻结checkpoint的held5 `correct_full/first_final/same_task_other`静态rank16物化、fit75 learned language-only control和
  paired five-arm Gate report；language control只用frozen `P_lang`映射fit-task verified rank4 effect，不读取held video/action/reward。
- [x] 从clean pushed detached `5140362`完成macro5/95 updates formal、三条video bank、learned language control与五臂paired
  strict250 Gate：carrier/language/full/endpoints/same-task分别为`43/42/35/40/44`，full breadth`3/5`、carrier retention
  `28/43`、Goal/Long均0、相对language/endpoints为`-7/-5`；只有same-task retention `33/35=94.3%`及全部adapter authority检查通过，
  所以macro5明确non-pass。
- [ ] 不改代码、配置、seed、LR、loss或数据，按原formal schedule把同一架构fresh运行到预注册macro10，检验macro5是否只是
  warmup后有效更新不足；macro10若没有显著改善shared selection方向与closed-loop Gate，不自动续到macro20/40，先修正
  signed-pooling置信度/初始化或shared supervision这一最早接口再fresh复评。gpu01在exact-resume launch前失联且run contract锁定
  visible devices/NUMA/affinity，因此不放宽resume合同，改在gpu02保持world2做唯一fresh macro10候选。

### G4 Joint Writer

解冻全部Writer、冻结backbone/carrier/experts；默认直接联合优化已通过组件Gate的完整Writer，并优先保留支撑闭环因果问题的最小loss
集合。只有真实机制证据表明需要时才使用有退出条件的functional warmup；随后可在student visited states上用verified short
continuation筛选有效expert members。至少两个train24 folds要求recovery`>=0.40`、breadth5/5、Goal/Long均非零、carrier
retention`>=75%`、same-task retention`>=85%`和相邻checkpoint稳定。

### G5 Structured outer credit（条件阶段）

只有G4已证明full高于carrier/language/endpoints且breadth成立才进行。只更新event/Program/rank attention/scales；最终outer节点相对G4
净增至少10且breadth、Goal/Long、same-task retention不下降。无提升时按credit机制证据分析、修正或更换estimator，不设次数上限，
但不做无信息超参扫。

### Final fresh与正式评测

使用全部71 meta+train24，role仍50/50；fresh初始化已验证组件后直接做完整Writer joint training，G1--G3的冻结分段只作架构因果
验证，不机械重演为最终训练流程。若联合训练证据要求短暂分段预热，必须登记最早失效接口与退出条件；loss默认从最小充分集合开始，
仅在closed-loop或明确机制证据支持时增加。只有base joint通过时才启用已验证的outer。
validation8只比较三个预注册相邻checkpoints；资格arm先跑correct/same-task/language/first+final。冻结selected后补完整controls，最后才跑
shuffled/reversed；方法完全冻结后只打开Test8一次。

`current_owner_requirements`中的32-task fresh refit与上述71 meta+train24 development recipe的精确顺序待Final前owner裁决；
该冲突不阻塞G1--G5，当前不为任一解释启动数据合并或训练。

## 路线边界

- 不恢复人工process dataset/controller acquisition；
- 不训练neural `q_pi`，不恢复fixed-effect/two-sided realizer；
- 不把GOMQ、PECS、v24或历史solver当ECP前置；
- 不在G1前训练fresh Stage 0、shared compiler、joint Writer或outer credit；
- 不用loss、geometry或checkpoint union替代single-checkpoint closed loop；
- shuffled/reversed只在最终selected checkpoint选定并冻结后测试时序特异性，不进入训练、loss、checkpoint选择、
  G1--G5 Gate或架构修正依据；
- Action Meta只在base Writer有闭环增量后做matched controls，只有明确净收益且无breadth/retention损害才启用；
- rank12 carrier + mobile rank4是G1的首版可证伪配置，不是永久硬约束；只有active design登记的rank-ceiling证据链成立才重开分配；
- 不人为限制各阶段时间、修正次数、结构版本或训练轮数；遇到scientific non-pass先按Gate定位接口，有新证据就修正，无新机制的
  slot/width/rank/seed版本链不算有效尝试；
- 优先复用、并行和提高吞吐，进展顺利时力争数天内完成整体架构实现并推进关键Gate。
