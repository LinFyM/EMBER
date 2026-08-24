# EMBER task plan

更新时间：2026-08-24。

## 当前目标

完成专家回复固化与跨session交接。当前session不启动GPU、不实现新架构；文档提交并推送后停止。下一session从唯一活动设计
ECP Native-Factor Compiler开始，首先完成task-local capacity oracle，不能先训练fresh Program、shared compiler或joint Writer。

## 当前session完成条件

- [x] 清除退役Writer、ECP v1--v24后继、MDCO/PECS、人工process与失败realizer执行路径；
- [x] 删除过时配置、脚本、测试、重复文档/证据与约11.6GB人工资产；
- [x] 专家回复完整阅读并对照当前保留实现；
- [x] 专家1416行原文逐字保存，active design与原文明确分层；
- [x] 固化专家确认的Program schema、native-factor compiler、rank、训练阶段、Gate、final controls与停止条件；
- [x] 明确删除神经`q_pi`与fixed effect-code realizer，保留effect evidence为training critic；
- [x] 运行文档一致性/Git审查，提交并推送`main`；
- [ ] owner开启新session后，新session消费并删除`HANDOFF.md`。

## 下一session当前唯一执行阶段：G1 capacity oracle

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

预计一个完整周期约1个工作日。失败只允许一次read-only span分析和一次hook/pooling实现修正；再失败即停止Native-Factor，不训练
后续模块，也不把失败归因于数据量。

## G1通过后的固定序列

### G2 Natural Program（1--2个工作日）

meta56+target-fit19，K均匀采样1/2/4；训练owner-specific language/scene、ordered events与K aggregation。meta-held15+target-held5检查
same-task separation、probe stability、event non-collapse、full相对endpoints的held loss增量与K合同。只允许一次限域结构修正。

### G3 Frozen-Program shared compiler（2--3个工作日）

冻结Program，用自然videos与95-task/118-member evidence训练rank queries、signed pooling、scales和bounded K correction。held5要求full
`>=60/250`、breadth`>=4/5`、retention`>=33/43`、Goal/Long至少一项非零、相对language和first+final各`+5`、same-task
retention`>=80%`。最多一个完整结构版本。

### G4 Joint Writer（2--4个工作日/轮，最多两轮）

解冻全部Writer、冻结backbone/carrier/experts；先functional warmup，再用student visited states上的verified short continuation筛选有效
expert members。至少两个train24 folds要求recovery`>=0.40`、breadth5/5、Goal/Long均非零、carrier retention`>=75%`、same-task
retention`>=85%`和相邻checkpoint稳定。

### G5 Structured outer credit（条件阶段，2--3个工作日）

只有G4已证明full高于carrier/language/endpoints且breadth成立才进行。只更新event/Program/rank attention/scales；最终outer节点相对G4
净增至少10且breadth、Goal/Long、same-task retention不下降，否则停止estimator。

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
- Action Meta只在base Writer有闭环增量后做一次matched attempt，只有明确净收益且无breadth/retention损害才启用；
- rank12 carrier + mobile rank4是G1的首版可证伪配置，不是永久硬约束；只有active design登记的rank-ceiling证据链成立才重开分配；
- 遇到科学non-pass先按Gate判定最早失效接口，不进入slot/width/rank/seed版本链。
