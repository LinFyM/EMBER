# EMBER task plan

更新时间：2026-08-24。

## 当前目标

完成专家回复固化与跨session交接。当前session不启动GPU、不实现新架构；文档提交并推送后停止。下一session先完整读通EMBER的
目标、历史证据、专家原文、代码/数据/脚本资产与执行合同，形成准确的仓库理解；随后从ECP Native-Factor Compiler的task-local
capacity oracle开始，不能在理解仓库前盲目实现，也不能跳到fresh Program、shared compiler或joint Writer。

## 当前session完成条件

- [x] 清除退役Writer、ECP v1--v24后继、MDCO/PECS、人工process与失败realizer执行路径；
- [x] 删除过时配置、脚本、测试、重复文档/证据与约11.6GB人工资产；
- [x] 专家回复完整阅读并对照当前保留实现；
- [x] 专家1416行原文逐字保存，active design与原文明确分层；
- [x] 固化专家确认的Program schema、native-factor compiler、rank、训练阶段、Gate、final controls与停止条件；
- [x] 明确删除神经`q_pi`与fixed effect-code realizer，保留effect evidence为training critic；
- [x] 运行文档一致性/Git审查，提交并推送`main`；
- [ ] owner开启新session后，新session消费并删除`HANDOFF.md`。

## 下一session Phase R：全仓库理解与资产映射

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

Phase R完成后删除`HANDOFF.md`并提交；随后直接进入G1，不把orientation变成长时间文档考古。整体推进不设阶段工期和修正次数
上限，但应在保质前提下尽可能快，顺利时力争数天内完成整体架构实现并推进关键Gate。

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
2. 构造absolute/adjacent/init/goal output banks，按frame chunk在线读取与累计；
3. 实现Program-conditioned two-branch signed pooling、per-target scales、rank4 outer products与small-core SVD canonicalization；
4. 实现每task free-code optimizer，只优化4 rank queries、event/pooling weights和scales；
5. 接通global-member effect、sensitivity-normalized update、independent functional与carrier-preservation loss；
6. 生成唯一rank12+rank4 complete adapter，接入现有strict evaluator。

先做最小真实forward/gradient smoke，再进行5-task optimization与strict250；不增加通用框架、checksum或与Gate无关的测试。

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
same-task separation、probe stability、event non-collapse、full相对endpoints的held loss增量与K合同。修正应限于证据定位到的native
capture、event grounding或owner-specific language/scene，不设次数上限。

### G3 Frozen-Program shared compiler

冻结Program，用自然videos与95-task/118-member evidence训练rank queries、signed pooling、scales和bounded K correction。held5要求full
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

## 路线边界

- 不恢复人工process dataset/controller acquisition；
- 不训练neural `q_pi`，不恢复fixed-effect/two-sided realizer；
- 不把GOMQ、PECS、v24或历史solver当ECP前置；
- 不在G1前训练fresh Stage 0、shared compiler、joint Writer或outer credit；
- 不用loss、geometry或checkpoint union替代single-checkpoint closed loop；
- Action Meta只在base Writer有闭环增量后做matched controls，只有明确净收益且无breadth/retention损害才启用；
- rank12 carrier + mobile rank4是G1的首版可证伪配置，不是永久硬约束；只有active design登记的rank-ceiling证据链成立才重开分配；
- 不人为限制各阶段时间、修正次数、结构版本或训练轮数；遇到scientific non-pass先按Gate定位接口，有新证据就修正，无新机制的
  slot/width/rank/seed版本链不算有效尝试；
- 优先复用、并行和提高吞吐，进展顺利时力争数天内完成整体架构实现并推进关键Gate。
