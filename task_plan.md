# EMBER Task Plan

状态：2026-08-18 **in progress，外部复核逐项审计goal已启动**。

## Goal

全面审计`docs/external_review_20260818.md`中的每条重要意见：把代码事实、历史实验事实、因果推断、反证、建议干预
和数值门槛逐项拆开；先复核零训练成本证据，再用最少的当前架构单变量实验裁决主要根因。最终形成一张每条主张为
`confirmed / refuted / underdetermined`的claim ledger，而不是机械执行专家给出的全部建议。

## Done when

- 专家A--G各部分的关键主张均有代码/本地artifact证据、反证和当前裁决；
- current formal provenance、paired rows、McNemar、breadth@1/@5/@10、task/suite集中度全部从本地原始证据复核；
- intended trainable modules的实际gradient表覆盖fresh和macro25，而不再只依赖`requires_grad`或参数数量；
- current macro25的视频因果面板明确correct、same-task-other、wrong、shuffle、reverse与no-video关系；
- Text Meta-LoRA影响与fresh前端detach影响在实验上可区分，所有新canonical训练均无Text/VL Meta-LoRA；
- absolute support形成与checkpoint retention分别得到最小干预裁决；
- occupancy、FactorHead reachability/co-drift和shared-gradient conflict只在前置证据支持时进入下一阶段；
- 每个实验报告strict paired400、per-task/suite、breadth、retained/gained/lost和churn，并更新持久文档。
- 专家报告中的每项建议最终都有明确回应：无条件诊断实际完成；专家原文中的条件分支在前提成立时执行，前提不成立
  时以证据化`not applicable`裁决，不得静默跳过。

## Fixed boundaries

- 只基于当前EMBER-LMMPC Core-Addressed Reader主链，不恢复V6/LPCP/GOMQ、旧Writer carrier、旧compiler或旧LoRA残差；
- 后续canonical Writer不使用Text Meta-LoRA或VL Meta-LoRA；exact language仍通过冻结原生text/VLM表示进入Program；
- Action Meta-LoRA保持独立变量，本计划前段不改；rank16、memory、Reader、K-set和bounded M2P同样先保持；
- 不把专家建议的`+15`、`90%`等数值直接升级成owner硬合同；它们作为预注册参考，与EMBER既有资格标准并列报告；
- 不用loss、gradient、norm、cosine或oracle选择最终方法；formal选择仍认single-checkpoint strict paired400；
- 一次只裁决一个主要变量，不用rank/LR/scale/seed/dtype sweep，不添加防御性hash或重复forward；
- 不使用subagents；GPU launch仍遵守单节点最多6张、全局EMBER最多8张和live availability合同；
- 本goal授权在上述边界内完成全部审计和有前提的实验；每个formal launch仍须先写清单变量、资源和裁决合同。

## Phase 0 — Claim ledger与静态代码核验

- [x] 把专家报告逐项编号为code fact、experiment fact、hypothesis、recommendation或threshold；
- [x] 对detach、Text Meta-LoRA、Action-50-token mean、Reader时间中心化、多套address identity、M2P RMSNorm、
  action-in/out派生、FactorHead子空间/B-first冷启动、信息墙和optimizer aggregation逐项建立代码证据；
- [x] 对每条代码事实补当前测试覆盖、未覆盖模块和是否需要新增稳定regression test；
- [x] 核对旧V6与当前路径的精确差异，但只作反事实provenance，不恢复执行。

**Gate 0：** 每项必须区分“代码必然如此”与“因此导致性能下降”的额外推断；后者没有干预不得标记confirmed。

## Phase 1 — 本地formal evidence重建

- [x] 从local raw rows重建macro25/50/75/100、LPCP143、GOMQ151的统一paired矩阵；
- [x] 复算McNemar、retained/gained/lost、Jaccard、breadth@1/@5/@10、top-3 concentration及suite minimum；
- [x] 核对每个formal run的commit/config/checkpoint manifest与review snapshot是否一致；
- [x] 按task、suite/horizon、initialization和teacher video IDs保留逐行lost/gained/retained证据；
- [x] 把可安全提交的聚合/脱敏row evidence补到remote，避免结论只依赖本地叙述。

**Gate 1：** 若原始rows与tracked汇总不一致，先修正事实authority；一致后才讨论根因。

## Phase 2 — 零训练成本或低成本机制审计

### 2A. 全模块gradient audit

- [x] 在fresh initialization和macro25各运行一个canonical task step；
- [x] 记录`patch_grounding`、`interaction_projection`、`language_projection`、Text/Action Meta-LoRA、Core、Procedure、
  memory tokens、Reader、K-set、M2P和八个FactorHeads的grad `None/nonzero/finite`；
- [x] 同时确认source policy nonzero gradient tensors为0；
- [ ] 把intended-path梯度变成稳定测试，避免以后再用部分模块代表全路径。

### 2B. Current macro25视频因果面板

- [ ] 在同一400 rows、state/env/policy RNG、K和video ordinal上评测correct、same-task-other、cross-suite-wrong、
  shuffled、shuffled-keep-first、reversed与no-video；
- [ ] 报告paired gains/losses、McNemar、per-suite方向、same-task retention和任务集中度；
- [ ] 将专家建议门槛与项目原有因果资格标准分开呈现，不因单一p值宣称理解高层过程。

**Gate 2：** gradient audit裁决当前mechanism合同；视频面板裁决当前123究竟更像correct-process依赖还是
nonconstant-video carrier/static shortcut。两项均不修改checkpoint。

## Phase 3 — 分离Text Meta-LoRA与输出detach

为了遵守“以后不再使用Text Meta-LoRA”且避免两变量混淆，采用三点链，其中sealed A已存在、只需两个fresh新run：

| arm | Text Meta-LoRA | fresh前端输出detach | 角色 |
| --- | --- | --- | --- |
| A | rank4 | 保留 | 已有sealed baseline `123→84→89→87` |
| B | 移除 | 保留 | bounded attribution baseline，只测移除Text；不作为最终方法 |
| C | 移除 | 移除 | 当前架构的canonical corrected front end |

- [ ] B只修改Text Meta-LoRA；冻结原生text representation和Writer-local projection保持；
- [ ] C相对B只删除`frame_evidence/grounded_evidence/interactions`的输出detach，保留frozen hidden detach；
- [ ] B/C均从fresh训练，锁定同一data、K schedule、rank16、optimizer和formal评测；
- [ ] B至少到macro25并做strict400；C到macro25，若有absolute或视频因果正证据则继续macro50判断retention；
- [ ] A→B只解释Text Meta-LoRA效应，B→C只解释front-end credit效应，A→C只报告combined canonical变化。

**Gate 3：** 若B→C paired net变化小于5且breadth、controls、stability均无改善，detach不是主要性能根因；若absolute
明显改善但25→50仍崩落，问题正式后移到occupancy/retention；若C仍低且视频controls弱，先审查process identifiability，
不直接扩大decoder。

## Phase 4 — Occupancy mismatch裁决

固定状态occupancy审计无论前面结果如何都完成；只有corrected C仍出现“offline改善、strict support丢失”时才执行
occupancy-matched训练干预：

- [ ] 对相邻checkpoint lost/gained/retained rows保存rollout state和首次行为分歧时间；
- [ ] 构造两checkpoint访问状态的固定union，在同一states上用冻结train24 expert/teacher reference比较action/flow error；
- [ ] 区分visitation shift、同一状态policy direction恶化、loss-success错位和failure tail；
- [ ] 只有固定状态证据支持occupancy mismatch后，才预冻结一个occupancy-matched training panel作单变量替换；
- [ ] 任何occupancy surrogate最终仍以相邻strict paired400、breadth和retention裁决。

## Phase 5 — Decoder reachability与co-drift

FactorHead freeze、reachability和endpoint/family审计均需完成，优先使用corrected C checkpoint；结果是否支持decoder
根因只决定是否继续修改decoder：

- [ ] FactorHead-freeze diagnostic：从一个有support的checkpoint冻结八个heads续训，测旧success retention；
- [ ] train24 reachability oracle：固定heads、自由优化`20x16x256` Program逼近policy-effective experts，评测投影后
  train-task closed loop；
- [ ] 分family报告q/v/action-in/action-out reachability，特别分离人为派生endpoint rows；
- [ ] 若oracle保留expert success≥90%，停止扩大head/rank，责任回到video-to-Program或credit；
- [ ] 若明确不可达，再设计一个decoder单变量，而不是同时动rank、width和坐标。

## Phase 6 — Shared-gradient conflict（最后条件分支）

在前端、occupancy和decoder审计后执行一个matched shared-gradient comparison，以直接回应专家最后一项建议：

- [ ] 固定per-task gradient、AdamW、LR、tasks和data，只替换一个预注册的conflict-safe aggregation rule；
- [ ] macro25相当且25→50 lost显著减少、breadth稳定，才支持shared-gradient conflict；
- [ ] 若仍复现漂移，撤回“optimizer/mean是主因”，回到objective occupancy与parameterization交互。

## Reporting and stop rules

- 每个phase结束更新`findings.md`、`progress.md`和外部review claim ledger；只有formal结果进入`research_history.md`；
- 机制smoke只回答图是否接通，不能选模型；
- 明确否定的分支立即停止，不做小扫；好结果继续到相邻checkpoint，达到约145时补完整稳定性与视频因果资格；
- 若C及有证据支持的Phase 4/5局部修复仍多次无法改善absolute和retention，则完成本审计goal，明确当前架构剩余的
  最早未解接口，不自行大幅改成另一套架构。
- 最终新增一份面向外部专家的报告，逐条引用其A--G意见、给出实施/不适用状态、原始证据、结果、我们同意或修正的
  结论以及仍需专家判断的问题；与给owner的总结、claim ledger和全部可提交evidence一起推送远程。
