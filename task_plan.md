# EMBER task plan

更新时间：2026-08-28。

## 当前目标

owner已正式许可推进ECP Native-Factor Compiler。Phase G1 task-local free-code capacity oracle和Phase G2 Natural Program已经
分别通过对应Gate；当前推进Phase G3 low-dimensional bank-adaptive shared compiler。G1只证明真实native banks与signed pooling存在强rank4
residual，G2只证明Natural Program保留视频动态；二者都不证明deployment Writer的shared Program-to-attention映射成立，单独完成
任一阶段仍不代表整体项目goal完成。

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

冻结Program，先用无训练低维bank-adaptive functional sketch证明当前native/key functional image能以数量级更低成本保留；通过后再用
自然videos与95-task/118-member evidence训练共享full-Program+current-bank-summary到native-candidate content attention、signed pooling、
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
- [x] 从同一clean detached `5140362`按原合同fresh完成macro10/190 updates及五臂strict250：carrier/language/full/endpoints/
  same-task=`43/42/38/39/40`，full breadth`3/5`、retention`32/43`、Goal/Long均0、相对language/endpoints为`-4/-1`，仅
  same-task retention `84.2%`通过；因此“只是macro5 warmup后更新不足”被证伪，没有续到macro20或做超参小扫。
- [x] 用fit-only、held-gradient 0的K1 native-span诊断定位最早接口：6 tasks/9 members的native update cosine median `0.7029`、
  named/global functional retention median `0.7855/0.7981`且action benefit `9/9`为正，说明rank4压缩与真实native bank有足够功能容量；
  v1的全局clip由scale path主导，shared query/key更新不足且没有直接native-feasible selection supervision。
- [x] 从clean pushed detached `93dffc7`在gpu02 p4/p5/p6封存formal40 schedule实际覆盖的50个fit tasks、451个K1
  task-video条件、68个covered-task verified members及662个teacher states；只保存pre-scale A/B directions、scales与provenance，
  root为828MiB，held/Action Meta/deployment reads均0，三个worker与aggregate均exit0。
- [x] 在保持frozen G2 Program、sampler/K/LR/rank/bounded beta不变的首版修正中：K1以detached set-valued responsibilities监督
  shared query-key产生的input/output subspace、paired update direction和独立small-core spectrum；K2/K4 teacher tensor reads为0；
  selection与scale/video各自clip且scale/video不反传shared context。真实三步K1/K2/K4 profile已验证teacher lookup、gradient wall、
  same-task、bounded beta、长K4与唯一rank16，峰值29.32GB，Action Meta/source/Program trainable均0。
- [x] 从clean pushed detached `2a7f760` fresh运行G3 v2到macro5/95 updates并完成同一五臂strict250：carrier/language/full/
  endpoints/same-task=`43/42/41/38/37`，full breadth`3/5`、carrier retention`33/43`、Goal/Long均0、相对language/endpoints
  `-1/+3`、same-task retention`73.2%`，故Gate明确non-pass；所有bank、单checkpoint、唯一rank16、配对及Action Meta 0检查通过，
  shuffled/reversed未使用。
- [x] 对同一真实fit K1 bank做固定条件、梯度分解和teacher-only反事实：macro5相对step0未改善最终paired update，spectrum反而更差；
  旧functional对selection的梯度范数约为teacher的`67x`，teacher spectrum与旧scale梯度cosine为`-0.9897`。因此最早失效接口是
  同一步内相互冲突的shared selection/scale credit，而非Program、native bank、rank、K聚合或Action Meta；不能沿v2混合loss直接续训。
- [x] 在同一真实target20 bank上完成free-query与解析dual对照：普通梯度500步仍仅`0.1624` update cosine，FP64 weighted
  inverse-covariance dual在冻结G2 measure下达到`0.9975`；缩到最大logit`0.1`后用现有antithetic softmax仍为`0.9975`。
  因此最早接口是高条件数dual的shared acquisition，不是banks、signed pooling或chunked实现。
- [x] 从clean pushed detached `e7d86b0`完成fit-only、按task leave-out的四family target-native dual-basis解析oracle：50 tasks、
  98个确定性K1 conditions的full-dual reference task-mean update cosine median/p10/min为
  `0.996949/0.995468/0.993884`；但最大128维LOTO basis只有`0.288444/0.249615`，`0/50` tasks达到`0.95`，q/v/action-out
  family median分别为`0.000490/-0.000586/0.146885`，故Gate明确non-pass。bank-conditioned functional least-squares虽证明旧欧氏
  投影不正确，128维仍不足，扩展raw/effect basis又需要约384--512维；因此淘汰compact fixed dual/effect basis，不扩到38 targets，
  不恢复fixed realizer。
- [x] 对owner-native direct-key候选完成same-task三video的raw-query、event-query、event-anchor、nonlinear content-key和直接score
  acquisition反事实。teacher update与G2 Program跨video稳定，但minimum-norm dual随bank covariance旋转；2000步已能明显拟合train
  score，held q/v仍弱或负相关，完整held update约为`-0.001/-0.003`，action-out也仅`0.114`。因此不把逐video analytic
  dual/score、raw native key或event query当作已确定的canonical G3修正，也不通过续训或width/LR/seed扫掩盖结构问题。
- [x] 全新专家已基于远程`main@ed2883b`及其完整可达历史复核结构分叉；1538行原文逐字保存为
  `docs/expert_review_20260826_bank_conditioned_native_factor.md`。裁决保留G1/G2、真实banks、exact signed pooling、rank4、small-core
  SVD、carrier12与唯一rank16；G3改为B0累计current-bank statistics/native anchors、regularized solve、B1重放exact pooling的
  bank-conditioned compiler，并在同一实现中保留一次`global_statistics_off`决定性消融。
- [x] owner明确Final保留整套Writer完全随机初始化、直接端到端fresh联合训练的正式可选项；G1--G3只作因果验证，不构成Final
  强制课程。该裁决覆盖专家“默认从通过组件初始化”的偏好，但不改写专家原文。
- [x] F0：clean pushed detached `19b5b3f`完成真实K1/K4 bank-conditioned forward/gradient/materialization。Program/source冻结、
  Action Meta 0、K4 teacher reads 0、76 tensors/38 targets及policy实际消费唯一rank16全部通过；同一cached native bank的chunk4与
  one-chunk最终更新cosine最低`0.99999976`、相对误差最高`0.00066443`，K4置换误差`1.43e-6`且权重严格均匀。raw rank-slot
  最大差`0.00311`只记录为small-core SVD gauge诊断，不再错误充当最终LoRA等价Gate。
- [x] F1：以50 tasks/98 conditions的既有authority完成operator capacity；free/analytic native anchors隔离shared mapping，要求q/v/
  action-in/out各family materialized与streaming replay median update cosine`>=0.995`、minimum`>=0.99`，chunk/full等价。若显式
  covariance实现不能恢复而materialized FP64 reference能恢复，才根据operator证据切换matrix-free block-CG/Lanczos。clean
  detached `435cb4a`的四family task-mean median为`0.999871/0.999824/0.999960/0.999884`，minimum为
  `0.999757/0.999544/0.999951/0.999743`，chunk/full row minimum`0.99999988`；Action Meta 0、held reads 0，Gate pass。
- [x] F2：从clean pushed detached `2199a76` fresh完成`global_statistics_off/C=I`到macro5/25 updates，并由六个独立worker完整评估
  50 K1-covered tasks/451 task-video（329 fit、40 held-video、82 task-holdout）。fit/held-video/task-holdout task recovery median为
  `.022243/.022858/.018919`；held-video action-in/action-out/q/v median为`.039958/.022185/.004722/.023158`，三个primary checks
  全部失败。Action Meta、held gradient及shuffled/reversed use均为0。由于fit本身已失败，结论是candidate-local first-moment
  compatibility没有形成，不继续F2或做LR/seed/width小扫；后续actual-operator证据指定v4后已删除off执行面，formal evidence保留。
- [ ] F3：开启bank-conditioned solve训练shared mapping；held-video recovery median`>=0.75`、p10`>=0.50`、train/held ratio
  （实际口径为held-video task median / fit task median）`>=0.8`且相邻checkpoint稳定。不得恢复逐video dual/score监督、task/video
  lookup、FactorHead或fixed realizer。clean detached `c1e26ce`已从fresh训练至macro5并按原world6 topology exact-resume至
  macro10；同一451-condition held median/p10由`.048433/.037740`升至`.089704/.072144`，held/fit由`1.019827`变为`.997650`，
  泛化和相邻task delta稳定但绝对Gate仍明确失败。macro10 held family median action-in/action-out/q/v为
  `.125947/.177230/.013288/.052761`。同checkpoint factor/gradient分解已定位到update-only双线性credit造成两侧subspace acquisition
  starvation：q/v的一侧span ceiling与key gradient显著弱于action families。首个单变量修正保留完整update选出的一个global member，
  对其input/output gauge-invariant subspace与paired update固定等权；不改变rank、width、group gain、LR、seed、data或Gate。六卡真实
  5-macro qualification三项loss均连续下降且Action Meta 0，但clean detached `84903aa` fresh macro5/macro10的451-condition
  held median/p10最终只有`.073029/.057174`，held/fit `.998320`；macro10四family仅
  `.098990/.146806/.008482/.040693`，所以equal-subspace credit作为充分修正已被formal证伪。跨family及两个独立fit condition的
  固定target gradient分解进一步显示近正交目标和最高约`20.5x` sensitivity imbalance；family/fixed-owner修正已由clean detached
  `c3fc8e3` formal到macro10，但fit/held/task-holdout仅`.074715/.074620/.081644`、held p10 `.058381`，仍明确non-pass。
  fixed-key/raw/FiLM tangent和direct-native F0后续复核又证明required directions处于极弱奇异尾，且direct-native solve代数上退化为
  raw-query transfer，因此没有启动无信息的`4117117` formal。same-task三video bank-global oracle显示共同feature code的transductive
  q/v/action-out约`.90--.93`，但两video minimum-norm inductive近零，最早接口为task-stable code识别。当前单一修正使用冻结G2
  exact-language-only `P_lang`生成same-task稳定anchor query，动态Program只控制event/frame measure，并对每video、每event candidate
  features做detached symmetric inverse-square-root；仍保留真实X/Y、native solve、两softmax之差与唯一rank16。clean detached
  `20acc33`的F0通过，fresh macro10完整451-condition fit/held/task-holdout提升到`.141080/.142120/.145828`，40/40 held tasks从
  macro5改善且held/fit为`1.00737`，证明稳定anchor修复了迁移但仍未过`.75/.50`。六task单任务20-step probes的另一fit/held均跟随
  train，但q仅`.0197--.0277`；task93 q的18个target在shared input/output query head上合成梯度只有norm和的`.272/.268`，而
  candidate trunk已有fixed-owner FiLM。当前单一修正因此给query trunks增加zero-init bounded fixed-owner input及fixed-owner/
  output-group FiLM；不重训G2，不改rank、bank、loss、data或Gate。实现通过184项CPU回归和architecture hard checks。
- [x] fixed-owner/group query FiLM formal F0：`7e232b0`首次运行在GPU计算前发现内部`_apply`覆盖PyTorch模块生命周期方法；
  `d64f7ad`以唯一rename修复并新增`.to(device)`回归。clean pushed detached `d64f7ad`随后通过真实K1/K4
  forward/gradient/chunk/materialization全部资格项，新input/output owner-query gradients为`.015828/.000958`，Action Meta 0，
  K4置换误差`1.91e-6`，chunk有效更新cosine最低`.99999826`且相对误差最高`.001863`，唯一38-target rank16被policy消费。
- [x] clean pushed detached `3e4e9a0`的fixed-owner/group query FiLM已从fresh运行macro5并按相同world6 topology exact-resume到
  macro10，451-condition fit/held/task-holdout为`.162011/.163128/.164562`、held p10 `.133783`；40/40 held tasks相对stable
  anchor改善，但增量主要来自action-in，`.75/.50` primary与相邻Gate仍non-pass，不续macro20。
- [x] 在保留F1 operator、G2 Program、rank、bank、data与Gate的前提下完成candidate-key/compatibility image解析容量裁决。浅层
  target0/1在`1e-6`谱尾可达约`.994/.997`，但layer9 target18/19仅`.5186/.5583`、layer17 target34/35仅
  `.6537/.6079`，相同direct-native reference仍约`.995--.997`；单target的多种pair-credit对照也只到约`.06--.21`。因此首因是
  线性dot-product score image对深层input失容，而不是再调loss、owner projection、LR或训练步数。
- [x] 用一个canonical family-shared additive joint compatibility替换“只有线性点积”的限制，同时保留点积残差及既有Program、
  candidate encoder、B0/solve/B1、rank、data、loss和Gate。先完成定向CPU合同、真实K1/K4 F0、显存/吞吐profile及Action Meta 0证明；
  通过后从clean pushed detached commit fresh运行相同451-condition F3，不把内部容量或loss下降冒充mapping Gate。首轮
  `a2a56a7` F0只在chunk有效更新上失败；固定bank因果对照排除单纯scale并证明antithetic signed rows加`.03`非零joint residual
  可恢复合同。clean pushed detached `e784eb9`完整F0的chunk cosine/relative error为`.9999965/.002641`，K4、全部joint梯度、
  Action Meta 0和唯一rank16均通过；world6真实一步为`89.83s`、单卡峰值约25.60GB，现从fresh进入相同F3。
- [x] clean pushed detached `55710bb`的joint compatibility已fresh完成macro5/macro10及完整451-condition F3；macro10
  fit/held/p10/task-holdout为`.126205/.128720/.103610/.129465`，Gate non-pass且低于`3e4e9a0`。四task path ablation证明
  dot-only更新几乎不变、joint-only recovery近零；wrong Program替换证明当前query近task-agnostic。task85跨三video final-factor
  对照又证明action-in可达约`.996`，但浅层q/v只有`.42/.49`、layer9 held约`.24/.22`、layer17 q held`.13`，raw-native
  projection也不充分。当前停止该checkpoint与macro20，不启动新formal；下一实现前先用fit-only最小正对照同时证明full Program
  task content、实质nonlinear interaction及condition-stable深层q/v factor acquisition，不能继续点积残差小修或超参扫。
- [x] 完成40-task fit-only稳定functional target审计：只用mapping-fit videos形成task/member rank4 consensus update，在预注册held
  video上的overall recovery median/p10/min为`.945820/.935622/.908717`；member set上界几乎相同，证明旧`.13`不是teacher
  inconsistency。固定target native input谱又显示family-shared top128对q/v仅约`.797`，target-owned top128在抽查深层targets为
  `.959--.976`，给出ownership修正的直接机制依据。
- [x] 用一个canonical replacement接通full G2 `rank_event` query、38-target native basis后接family trunk、无residual bypass的
  normalized bilinear compatibility，以及fit-video consensus paired-update-only mapping credit；原逐video input/output subspace只作
  诊断，mapping held video严格排除在consensus之外。真实K1/K4 smoke通过chunk/置换/唯一rank16/Action Meta 0及全部主路径梯度；
  world6真实一步profile覆盖6 tasks/12 K1 conditions，耗时`91.44s`、peak allocated约`25.65GB`，所有Q/K/owner/gain probes
  finite/nonzero。首次profile精确暴露旧P_lang-only稳定神经支路64个无梯度参数，已删除该冗余旁路并以确定性P_lang+owner/rank
  稳定视图修复，未放宽跨卡梯度完整性检查。
- [x] 上述唯一实现面已集成至clean pushed `main@3062de8`，并从detached frozen worktree fresh完成macro5/25 steps及完整
  451-condition F3。fit/held-video/p10/task-holdout task recovery为`.084298/.082754/.072027/.093856`、held/fit
  `.981684`；held q/v/action-in/action-out为`.020707/.065711/.084290/.171636`。primary Gate明确non-pass，不续macro10。
- [x] 完成non-pass后的结构根因审计。fit-only universal rank4在held-video/task-holdout上已达`.825054/.835443`，证明当前
  residual label含有很大的task-independent common correction；`carrier12 + universal4`重新压回rank12的update cosine为
  `.998741`，但把common项直接从task residual相减后，task85 q/v在真实bank中的解析input可达性只剩约`.828/.765`，所以不能把
  代数重心化冒充native-feasible新teacher。四task因果干预又显示wrong Program后的最终update仍为q/v/action-in/action-out
  `.973/.981/.992/.948`，而wrong bank已降为`.863/.834/.9999/.569`；Program路径虽连通，却没有成为task selection owner。
  task-local两fit-video正对照中，current keys对held q/v input subspace只到`.188/.177`，新target-native pointwise projection也仅
  `.171/.130`，相同原teacher的direct native reference约`.997`。fit-only backward约`99.88%`原始gradient energy落在candidate
  encoders/trunks。该证据链同时定位到错误carrier/residual分解与pointwise functional canonicalizer/Program acquisition，不是
  train/held泛化、operator、chunk、Action Meta、欠训或普通超参问题。
- [x] 在继续改结构前复核F1与canonical B1的数值口径，发现F1解析上限使用FP64，而runtime继承source初始化的TF32；既有F0只比较
  两条相同TF32路径，未能发现共同偏差。真实深层q、v与action-out的受监督native-anchor panel中，IEEE FP32相对FP64的最大
  update-cosine绝对误差仅`7.4e-5`，TF32误差median约`.52--.68`；held learned-anchor recovery分别由TF32
  `.256/.178/.262`恢复为IEEE `.705/.798/.673`。旧`3062de8` checkpoint改用IEEE/FP64只读重放仍约`.08165`，所以不得
  post-hoc冒充成功，必须从fresh检验正确梯度。
- [x] 将唯一canonical compiler的native dual score/reduction固定为IEEE FP32并保持该process setting穿过backward；不改Program、
  bank、query-key公式、loss、rank、data、optimizer或Gate。clean pushed `main@78b7e58`已通过`186`项CPU回归；4卡真实fresh一步
  profile覆盖固定6 tasks/12 K1 conditions，全部主路径gradient finite/nonzero、Action Meta 0，耗时`123.62s`、峰值约
  `25.65GB`。
- [x] clean pushed detached `78b7e58`的真实F0通过新增IEEE数值资格：TF32实际关闭，chunk4/one-chunk最低update cosine
  `.99999955`、最大相对误差`.000945`，K4置换误差`1.43e-6`，全部关键gradient非零，Action Meta 0，唯一38-target
  rank16被policy实际消费。
- [x] clean pushed detached `78b7e58`已从fresh完成IEEE F3 macro5与完整451-condition primary。fit/held/p10/task-holdout为
  `.086508/.083131/.072629/.096191`，held/fit `.960958`；q/v/action-in/action-out held median为
  `.021698/.065269/.085933/.173804`。除held/fit外Gate均失败，证明TF32是必要修正但不是`.08`量级shared mapping的主因。
- [x] 沿实际post-`Wk` bank把旧Euclidean query坐标与真实`C_r C_0^+ H` functional image逐层对照。深层q/v与两个action
  family的task-local functional-polar见证约为`.996/.999/1.000/.998`；跨rank共享polar使v/action-out降至`.915/.831`，raw
  non-whitened chart使q降至`.911`。因此最早接口是Program query在错误metric中被单位化，而不是bank/key/rank/G2/optimizer失容。
- [x] 对唯一v4 functional-polar实现完成真实K1分段profile和有边界的执行优化：`da3fd3e`复用单次frozen X/Y capture、按shape合批
  functional polar、以IEEE FP32累计/求解并用thin-QR small SVD；全仓`189 passed`。condition由`82.114s`降至`58.332s`，但25-step
  macro5在六卡上的理想训练下限仍约`49min`且未含451评测，故吞吐资格non-pass；未运行K4 F0、formal F3、训练或评测。
- [x] 全新专家已锁定远程`main@9b52e59`及其可达历史，完整审计G3 formal/diagnostic evidence与full-polar profile；1033行原文逐字
  保存为`docs/expert_review_20260828_g3_functional_sketch.md`。裁决full functional-polar只作fit-only teacher/reference，当前唯一
  deployment候选改为low-dimensional bank-adaptive sketch与轻量shared student；不再发射full v4 F0/F3。
- [x] S1：接通F1 condition authority、sealed fixed nested projection、current-bank native/key cross-image、`r_s={16,32,64}`共享前缀、
  `C_rQ` full-native free-query与exact signed replay。clean detached `27bde62`的task93/q20 formal早停反例中，同条件F1 positive为
  `.99556--.99791`，rank64仅`.15669--.15744`，chunk最低`.9999769`；因此含row minimum`.95`的容量Gate确定non-pass，不再运行其余
  96 conditions，也不把native-Q sketch训练成shared student。该早停不估计全panel分布。
- [ ] S2：按专家的S1失败分支实现不经过`Q_g q_tilde`native瓶颈的pure low-dimensional set-summary student。固定6 meta+6 target，
  其中每role各1个true task-holdout；其余tasks两条fit video、一条zero-gradient video-holdout。scale、G2/source/carrier冻结；先在
  同一candidate-logit/exact-X/Y执行面跑task-local free-query正控，正控通过后才训练shared full-Program+bank-summary query。shared
  结果前用universal negative、free-query positive和`78b7e58`失败checkpoint一次校准并sealed absolute/causal Gate，禁止按结果移动。
  首先只跑task93/q20机制witness：同一task-local code共同拟合两条video，第三条严格zero-gradient；fit median`>=.90`、held及input/output
  pushforward各`>=.80`、held/fit`>=.8`才进入12-task正控。首版使用measure-normalized mean/variance DeepSets summary和共享bounded
  candidate scalar energy，冻结现有candidate encoder；失败只淘汰这一明确函数类。
  - [x] 接通共享frozen native-bank runtime、separable scalar-energy、exact signed pooling、固定final-step runner及三视频真实gradient smoke；
    31项定向合同通过，Action Meta和全部旧authority实际冻结。
  - [x] clean detached `4d84dee`完成首轮1000-step witness：fit/held仅`.328/.175`。nested free-logit oracle在global/eventwise均约
    `1.000`，而固定或fresh 128D score basis即使移除summary映射并加入强factor credit仍不超过约`.36`；首轮runtime进一步确认所谓
    existing candidate encoder实际是未加载checkpoint的fresh seeded projection，因此该结果只淘汰这一错误authority组合。
  - [ ] 单一修正为显式加载并冻结`78b7e58/macro5` fit-trained candidate encoder/trunk/metadata/key projection；保持summary、score、loss、
    videos、1000 steps与Gate不变。真实smoke证明authority与Action Meta 0后，从clean detached commit运行v2 witness；若仍不通过，再依据
    fit score决定解冻/替换candidate encoder，不提前改summary或进入12-task/shared训练。
- [ ] S3：只有S2的absolute、video/task泛化和Program×bank因果同时通过，才从clean pushed detached commit恢复完整451-condition F3。
  保留held median`>=.75`、p10`>=.50`、held/fit`>=.8`和相邻checkpoint稳定作为absolute必要条件；另须通过sealed
  leave-one-task-out universal-centered、wrong Program、wrong bank、crossed interaction、q/v与own-vs-wrong teacher Gate。
- [ ] 条件式做fit-only decomposition-feasibility oracle：仅当functional-polar mapping已显著取得selection而残余证据仍指向common
  correction/carrier时，只用授权fit tasks形成shared correction并候选重拟合carrier12；随后针对
  **新carrier**从完整expert update重新计算每task residual，再投影回每条真实native bank。必须同时证明carrier压缩/retention、
  四family native direct/free-code可达性、跨video consensus与唯一rank16；若不成立，不保留该carrier，不能直接复用代数差分factor。
- [ ] 条件式只有S3已取得task-specific selection而残余证据仍指向decomposition时，才重开carrier/common correction；不得绕回逐video
  dual、task/frame lookup、高维factor head，也不得用universal shortcut、内部loss或普通超参扫通过Gate。candidate basis、Program和
  scale必须保持明确parameter ownership；rank spectrum/scale只在selection取得后用隔离credit处理。
- [ ] F4：恢复全部75 fit tasks的scale/functional/flow/preservation职责；mapping loss保护selection，scale/video独立更新；teacher
  paired update不退化。只有mapping已学会而低置信随机residual仍破坏carrier时，才加入deployment-visible confidence退回机制。
- [ ] F5：按K1到K2再到K4恢复多视频职责，K2/K4 teacher reads保持0；验证K1 identity、集合置换不变、bounded beta和same-task
  mapping retention`>=80%`。
- [ ] F6：冻结单一checkpoint执行held5 carrier/language/full/first+final/same-task五臂strict250；沿用现有G3 Gate。formal runtime
  使用固定3+3 role-balanced全局task group并按实时1--6张有效GPU弹性分片；world size不得改变task权重、optimizer cadence或Gate
  口径，exact-resume锁定原topology。

### G4 Joint Writer

解冻全部Writer、冻结backbone/carrier/experts；默认直接联合优化已通过组件Gate的完整Writer，并优先保留支撑闭环因果问题的最小loss
集合。G3的native-feasible LoRA teacher是组件验证工具，不是G4/Final必须存在的数据依赖；正式训练不得预设有目标LoRA，可直接以
授权fit/meta teacher actions、functional与on-policy闭环信号监督完整Writer。只有真实机制证据表明需要时才使用有退出条件的functional
warmup；随后可在student visited states上用verified short
continuation筛选有效expert members。至少两个train24 folds要求recovery`>=0.40`、breadth5/5、Goal/Long均非零、carrier
retention`>=75%`、same-task retention`>=85%`和相邻checkpoint稳定。

### G5 Structured outer credit（条件阶段）

只有G4已证明full高于carrier/language/endpoints且breadth成立才进行。只更新event/Program/rank attention/scales；最终outer节点相对G4
净增至少10且breadth、Goal/Long、same-task retention不下降。无提升时按credit机制证据分析、修正或更换estimator，不设次数上限，
但不做无信息超参扫。

### Final fresh与正式评测

使用全部71 meta+train24，role仍50/50；Final保留两种fresh初始化候选：（a）从已验证Program/compiler初始化后完整Writer joint
training；（b）整套Writer完全随机初始化后从头直接端到端joint training，让梯度下降自行形成内部功能分化。两者均用fresh
optimizer/scheduler和同一信息墙、数据及closed-loop合同。G1--G3的冻结分段只作架构因果验证，不机械重演为最终训练流程；若联合
训练证据要求短暂分段预热，必须登记最早失效接口与退出条件。loss默认从最小充分集合开始，仅在closed-loop或明确机制证据支持时
增加；不预设Final有目标LoRA。只有base joint通过时才启用已验证的outer。
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
