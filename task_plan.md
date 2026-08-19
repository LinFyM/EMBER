# EMBER Task Plan

状态：2026-08-19 **complete；专家A--G/F0--F5已逐项裁决，报告与远程e证据已封存**。

## Goal

全面审计`docs/external_review_20260818.md`中的每条重要意见：把代码事实、历史实验事实、因果推断、反证、建议干预
和数值门槛逐项拆开；先复核零训练成本证据，再用最少的当前架构单变量实验裁决主要根因。最终形成一张每条主张为
`confirmed / refuted / underdetermined`的claim ledger，并让专家报告的每一项建议都有实际结果或证据化裁决。

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
- [x] 把intended-path梯度变成稳定测试，避免以后再用部分模块代表全路径。

### 2B. Current macro25视频因果面板

- [x] 在同一400 rows、state/env/policy RNG、K和video ordinal上评测correct、same-task-other、cross-suite-wrong、
  shuffled、shuffled-keep-first、reversed与no-video；
- [x] 报告paired gains/losses、McNemar、per-suite方向、same-task retention和任务集中度；
- [x] 将专家建议门槛与项目原有因果资格标准分开呈现，不因单一p值宣称理解高层过程。

**Gate 2：** gradient audit裁决当前mechanism合同；视频面板裁决当前123究竟更像correct-process依赖还是
nonconstant-video carrier/static shortcut。两项均不修改checkpoint。

## Phase 3 — 分离Text Meta-LoRA与输出detach

为了遵守“以后不再使用Text Meta-LoRA”且避免两变量混淆，采用三点链，其中sealed A已存在、只需两个fresh新run：

| arm | Text Meta-LoRA | fresh前端输出detach | 角色 |
| --- | --- | --- | --- |
| A | rank4 | 保留 | 已有sealed baseline `123→84→89→87` |
| B | 移除 | 保留 | bounded attribution baseline，只测移除Text；不作为最终方法 |
| C | 移除 | 移除 | 当前架构的canonical corrected front end |

- [x] B只修改Text Meta-LoRA；冻结原生text representation和Writer-local projection保持；
- [x] C相对B只删除`frame_evidence/grounded_evidence/interactions`的输出detach，保留frozen hidden detach；
- [x] B/C均从fresh训练，锁定同一data、K schedule、rank16、optimizer和formal评测；
- [x] B到macro25并做strict400；C到macro50并完成macro25/50 strict400稳定性判断；
- [x] A→B只解释Text Meta-LoRA效应，B→C只解释front-end credit效应，A→C只报告combined canonical变化；
- [x] B/C macro25完整视频controls与same-task稳定性面板；两者7面板均400 rows严格配对，
  mismatch均0。

**Gate 3：** 若B→C paired net变化小于5且breadth、controls、stability均无改善，detach不是主要性能根因；若absolute
明显改善但25→50仍崩落，问题正式后移到occupancy/retention；若C仍低且视频controls弱，先审查process identifiability，
不直接扩大decoder。

## Phase 4 — Occupancy mismatch裁决

固定状态occupancy审计无论前面结果如何都完成；只有corrected C仍出现“offline改善、strict support丢失”时才执行
occupancy-matched训练干预：

- [x] 对相邻checkpoint lost/gained/retained rows保存rollout state和首次行为分歧时间；
- [x] 构造两checkpoint访问状态的固定union并完成两checkpoint action比较；validation expert不存在且held teacher action
  禁止读取，正确性reference明确裁决为`underdetermined-after-audit`；
- [x] 区分visitation shift、同一状态policy direction恶化、loss-success错位和failure tail的可判与不可判部分；
- [x] fixed-state方向不支持简单occupancy claim，因此occupancy-matched训练干预裁决为`not-applicable`；
- [x] occupancy surrogate未用于选择方法。

## Phase 5 — Decoder reachability与co-drift

FactorHead freeze、reachability和endpoint/family审计均需完成，优先使用corrected C checkpoint；结果是否支持decoder
根因只决定是否继续修改decoder：

- [x] FactorHead-freeze diagnostic：从一个有support的checkpoint冻结八个heads续训，测旧success retention；
- [x] train24 reachability oracle：固定heads、自由优化`20x16x256` Program逼近policy-effective experts，评测投影后
  train-task closed loop；
- [x] 分family报告q/v/action-in/action-out reachability，特别分离人为派生endpoint rows；
- [x] oracle为659/1200、direct为658/1200，明确通过90%门并停止扩大head/rank；
- [x] 不启动decoder扩容；fixed-head reachability并非当前瓶颈。

## Phase 6 — Shared-gradient conflict（最后条件分支）

在前端、occupancy和decoder审计后执行一个matched shared-gradient comparison，以直接回应专家最后一项建议：

- [x] 固定per-task gradient、AdamW、LR、tasks和data，只替换预注册`deterministic_pcgrad_v1`；
- [x] macro25相当但25→50 lost减少未达显著且breadth仍6→4，故不支持shared-gradient conflict为主要根因；
- [x] PCGrad降低churn同时显著抑制gained并降低absolute，撤回“arithmetic mean是主因”；Adam moment独立效应因两臂
  均使用AdamW而保留为`underdetermined-after-audit`。

## Reporting and stop rules

- 每个phase结束更新`findings.md`、`progress.md`和外部review claim ledger；只有formal结果进入`research_history.md`；
- 机制smoke只回答图是否接通，不能选模型；
- 明确否定的分支立即停止，不做小扫；好结果继续到相邻checkpoint，达到约145时补完整稳定性与视频因果资格；
- 若C及有证据支持的Phase 4/5局部修复仍多次无法改善absolute和retention，则完成本审计goal，明确当前架构剩余的
  最早未解接口，不自行大幅改成另一套架构。
- 最终新增一份面向外部专家的报告，逐条引用其A--G意见、给出实施/不适用状态、原始证据、结果、我们同意或修正的
  结论以及仍需专家判断的问题；与给owner的总结、claim ledger和全部可提交evidence一起推送远程。

## Final outcome

- 没有新arm达到约145或稳定方法资格，本goal以negative/mixed scientific result完成；
- front-end detach是真实credit缺陷并改善视频方向资格，但不是absolute/stability首因；
- 简单occupancy divergence未获支持，FactorHead co-drift是放大器而非reachability瓶颈，PCGrad改变换手但未解决共同积累；
- `docs/external_review_claim_ledger_20260818.md`的113个编号条目均已有实施结果、反驳、
  `not-applicable`或`underdetermined-after-audit`边界，无queued项；
- 本goal封存后不登记active successor，等待owner与外部专家的下一轮裁决。
