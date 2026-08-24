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

### G3 Frozen-Program shared compiler

冻结Program，用自然videos与95-task/118-member evidence训练共享Program-query到native-candidate-key的content attention、signed pooling、
scales和bounded K correction，禁止task/frame查表。依据G2证据可从均匀跨video权重初始化有界learned `beta_k`或其他bounded correction，
并防止单条video覆盖其余videos。held5要求full
`>=60/250`、breadth`>=4/5`、retention`>=33/43`、Goal/Long至少一项非零、相对language和first+final各`+5`、same-task
retention`>=80%`。可以按mapping/compiler/critic证据修正，不设结构版本上限，但无机制差异的小变体不算推进。

### G4 Joint Writer

解冻全部Writer、冻结backbone/carrier/experts；先functional warmup，再用student visited states上的verified short continuation筛选有效
expert members。至少两个train24 folds要求recovery`>=0.40`、breadth5/5、Goal/Long均非零、carrier retention`>=75%`、same-task
retention`>=85%`和相邻checkpoint稳定。

### G5 Structured outer credit（条件阶段）

只有G4已证明full高于carrier/language/endpoints且breadth成立才进行。只更新event/Program/rank attention/scales；最终outer节点相对G4
净增至少10且breadth、Goal/Long、same-task retention不下降。无提升时按credit机制证据分析、修正或更换estimator，不设次数上限，
但不做无信息超参扫。

### Final fresh与正式评测

使用全部71 meta+train24，role仍50/50；fresh carrier -> Program -> frozen-Program compiler -> joint Writer -> 仅在通过时outer。
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
