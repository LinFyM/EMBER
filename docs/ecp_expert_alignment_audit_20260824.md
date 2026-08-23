# EMBER-ECP 全过程专家对齐审计

状态：2026-08-24 **专家复核已完成**。第1--8节保留提交复核时的原始自查，便于追踪哪些判断由仓库先提出；
第9节记录专家对`main@6a97185126ab640c3f9a6a719084dc0268ddd8e9`的最终修正。后续执行以第9节、
`docs/event_conditioned_policy_compiler_design.md`和`task_plan.md`为准，第7节旧候选顺序不再授权执行。

## 1. 审计结论先行

当前仓库保留了大量真实、可复用的ECP证据，但此前的阶段命名过强：

1. Stage 0 v3是一个非退化的native observer candidate；它还没有证明完整process semantics。
2. 当前48-state、three-member occupancy effect bank完成的是successful-policy evidence与state coverage前置资产，不等于
   专家定义的distributional privileged teacher `q_pi(P)`已经建立。
3. 当前retained Stage 1代码直接以privileged owner/flow/action effects驱动fixed solver，并不读取结构化Program。它是一项
   合法且有价值的direct-effect realization子门，但不是完整的
   `policy evidence -> q_pi(P) [Program posterior] -> policy-effect distribution -> fixed solver -> LoRA`链路。
4. 历史v1--v24确实实现过visible Program、privileged teacher和learned A/B compiler的多个版本，但teacher最终仍是
   deterministic/mean式坐标，compiler长期共同旋转；该家族的负结果不能替代对专家最终要求的distributional `q_pi(P)`与
   Program-to-effect接口的检验。
5. `q_V(P|L,V)`、完整Program-conditioned effect compiler、普通Writer联合训练以及最终outer credit均未完成。
6. process-identifying data是最终方法资格的重要缺口，但当前证据不足以把它升级为唯一根因或立即取代缺失的`q_pi(P)`主线。

所以当前最准确的状态不是“Stage 1A已通过、Stage 1B失败并阻止Stage 1C”，而是：

> successful-policy/occupancy evidence前置资产已完成；direct-effect realization已做出多项有效正负裁决；专家最终架构所需的
> distributional `q_pi(P)`与Program-to-effect桥仍未实现，因此完整ECP Stage 1尚未被执行完，也尚未被证伪。

## 2. 专家原始方案与修正案的共同核心

### 2.1 部署合同

部署输入始终只有exact language与K条same-task、action-hidden、内部有序的视频。Writer在rollout前运行一次，最终只生成一套
38-target rank16 LoRA；source PI0.5、VLM与Action Expert保持冻结，闭环中不再观看teacher video。

### 2.2 Program合同

专家要求的共享中间对象不是global code或LoRA rank grid，而是保留policy对应关系的结构化Program：

```text
P = {
  P_lang[layer/family],
  P_scene[layer/family],
  P_process[event, layer/family],
  rho[event],
  sigma[event, layer/family]
}
```

当前38个owner可作为18层q、18层v、action-in、action-out的具体索引。event容量可固定为8，active event数、视频段落到slot的
对应及event duration由内容学习；rank仍只属于最终LoRA参数化，不属于Program语义。

### 2.3 两个同构posterior

原始方案要求：

```text
q_pi(P | multiple successful policies, occupancies, policy responses)
q_V (P | exact language, action-hidden videos)
```

两者必须输出同一Program schema。`q_pi`应保留mixture/particles或mean+structured covariance，而不是把多个successful
members压成一个唯一Program。video可识别的task/process信息与rollout-only recovery信息必须显式区分；后者不能成为
deployment输入或要求`q_V`逐项预测的target。

### 2.4 policy-native编译合同

专家先否定了在少量task mappings上直接学习大规模Program-to-A/B hyperdecoder，随后将推荐路径收敛为：

```text
Program
  -> event/layer/family policy-effect distribution
  -> fixed differentiable solver or small fixed target-local preconditioner
  -> one complete rank16 LoRA
```

同一个compiler/realizer必须处理full Program与`P_process=0`的prior反事实；strong carrier可以保留，但最终仍只部署一套LoRA，
并且必须报告video/process correction创造和丢失的closed-loop successes。

### 2.5 阶段顺序

专家原始方案的顺序是：observer与Program calibration；privileged `q_pi(P)`和compiler realizability；冻结compiler后训练
`q_V(P|L,V)`；full video已产生闭环增量后才加入structured outer credit。最后一次修正又把Stage 1拆成successful-policy
equivalence、occupancy-complete privileged realization、后续shared compiler/video inference三道门。

这里存在一个需要专家重新定稿的口径：修正案允许Stage 1B直接以exact privileged policy effects检验fixed solver，而完整方法
仍要求Program预测policy-effect distribution。仓库此前把这个lower-level oracle直接等同于完整Stage 1A/1B，造成了阶段误报。

## 3. 实际推进全过程

### 3.1 Stage 0 v1--v3与Action Meta

| 工作 | 结果 | 能证明什么 | 不能证明什么 |
| --- | --- | --- | --- |
| native v1 | presence近全局、task表示坍缩，Gate fail | 首版全局duration/mean-action有捷径 | 不否定native Action Expert observer |
| native v2 | active events坍缩为1，Gate fail | posterior需要在segmentation前获得时序grounding | 不否定event slots |
| native v3 | 48/48 nearest margins为正，held5 10/10，active events约6.48 | 跨episode、跨task非退化的有序event表示可学 | 尚未证明same-endpoint process、完整probe invariance或closed-loop价值 |
| Action Meta | matched panel变化约`1e-4`，无稳定正负作用 | 共享Meta路径可运行且未观察到负面 | 不足以默认加载；应继续只是后续matched control |

当前Stage 0代码输出`process/presence/uncertainty`，并保留language summary与scene transition；但language/scene尚未在retained
compiler中成为最终layer/family Program，因此只能称candidate observer。

### 3.2 Learned ECP Stage 1 v1--v24

这一阶段曾真实实现visible Program、privileged teacher、single-LoRA compiler、multi-state support、action/response loss与多种
closed-loop credit。完整逐版本表在`docs/ecp_stage1_iteration_retrospective_20260823.md`，关键边界是：

- v1是唯一早期held5完整闭环，`27/250`，低于stable shared `43/250`；
- v2--v24连续23版没有新的held5 rows，主要以LoRA geometry与308-panel open-loop support决定去留；
- v13的support barrier通过8项门中的7项，但没有用bounded held5 rollout校准proxy；
- v19才首次固定compiler更新Program，v20才固定Program训练compiler，v21/v22才做free-Program fixed-compiler reachability；
- v23才恢复同一prior/full direct surface，v24才实现38个target-local heads与continuous static/process fusion；
- v24仍只有own retrieval `1/24`，full/prior held response均差于shared。

这组工作对address/value旁路、moving gauge、shared preservation、target-local response与closed-loop credit给出了大量有效负边界，
但它没有实现最终要求的distributional `q_pi(P)`：多个members最终仍被压成确定性teacher坐标，且video-observable与policy-only
recovery没有形成可供`q_V`对齐的显式posterior分解。

### 3.3 MDCO

MDCO把mapping扩到71个source-seen non-held tasks加fit19，共90 tasks，并完成540 task-equal visits与一次structured
calibration。held5 strict250只有`20`，低于source `21`与shared `43`，Goal/Long为0。它证明更多现有mappings与当前
`q_pi/compiler`目标不足以修复该learned compiler family，也证明当时open-loop proxy与closed-loop符号只对齐`2/5`。

它不能证明distributional `q_pi(P)`或所有Program-to-effect compiler无效；71项都已被source训练见过，也不是干净的
source-unseen adaptation mappings。

### 3.4 PECS

PECS删除learned Program mapping，直接用selected teacher contexts上的exact policy effects和fixed solver生成LoRA。local与完整
10-step denoising trajectory分别得到`58/250`与`59/250`，显著高于MDCO20和shared43，但Goal/Long仍为0。

它证明policy-native exact-effect constraints与固定求解链能创造真实闭环增量，也证明仅在稀疏teacher contexts把effect target
做得更完整仍不足以定义跨初始化successful-policy basin。它是direct-effect oracle，不是video-to-Program或完整ECP实现。

### 3.5 GOMQ因果补审

GOMQ是ECP之前的退役架构，不是ECP Stage 1。冻结cycle2的validation8 correct/same/wrong/shuffled/reversed为
`151/139/131/127/115`；correct相对wrong、shuffled、reversed有真实配对优势，说明action-hidden视频内容与顺序确实能生成
闭环有用更新。但same-task retention只有81.46%，相邻checkpoint为`151 -> 135 -> 131`，breadth为6/8且两项为0。

它应作为“strong carrier + video-conditioned small correction”与视频因果的历史锚点，而不是ECP成果、可恢复checkpoint或最终方法。

### 3.6 专家修正后的occupancy/effect realization工作

| 工作 | formal结果 | 正确解释 |
| --- | ---: | --- |
| 独立step2000 experts | held5合计`113/250`，5/5 tasks非零 | 独立successful members与Goal/Long成功occupancy已补齐 |
| 48-state effect banks | 每task initial8/successful24/candidate8/recovery8，三member particles | successful-policy evidence与闭环state coverage前置资产完成；不是`q_pi(P)` |
| fixed-A learned solver | `78/250`，breadth3/5，Goal/Long0 | 当前effect objective + fixed-A + 12-step solver有增量但未过realization门 |
| fixed-A analytic projections | latest/independent/earliest=`49/41/35` | stable fixed-A row space具有行为binding |
| mobile-rank4 analytic projections | `110/120/76`，全部5/5 | `carrier12 + mobile residual4`具备direct级absolute容量；Long exact-row有policy churn |
| mobile-rank4 raw-factor solver | `49/250` | zero-start raw-factor dynamics没有进入known-success effect basin |
| matrix-free effective-update profile | 负方向导数存在，但5个固定尺度均不下降，0 accepted steps | 只关闭当前sketch normalization/backtracking operator；未运行held5 |

这些结果全部可复用，但它们的输入是privileged effect bank。当前`src/ember/ecp/stage1_oracle.py`直接加载
`effect_bank_manifest`；`Stage1EffectBank`保存prefix/noise/category/stage/progress和source/carrier/member的owner/flow/action，
retained source中没有输出同构Program posterior的`q_pi`，也没有Program-to-effect模块。

## 4. 对照矩阵

| 专家要求 | 当前落实程度 | 证据与问题 |
| --- | --- | --- |
| exact language + action-hidden ordered videos，single LoRA | 忠实 | 信息墙、评测配对和single-rank16合同保持 |
| native observer读取真实prefix与Action Expert时间结构 | 已实现candidate | v3非退化；process/probe语义仍未完证 |
| language/scene/process/event/layer/family Program | 部分 | process/presence/uncertainty有实现；完整同构Program没有贯穿retained Stage 1 |
| distributional `q_pi(P)` | 未实现最终版 | 历史teacher为确定性mean式；当前effect bank不是Program posterior |
| successful/learner/recovery与multiple members | evidence资产完成 | 48-state、three-particle banks可作为`q_pi`训练输入；recovery的deployment边界已记录 |
| Program到policy-effect distribution | 未实现 | 当前solver绕过Program直接读取effect bank |
| fixed policy-native realization | 多个子门已实现并裁决 | PECS、fixed-A、mobile-rank4与effective-update operator结果均有效但未得到强pass |
| held leave-task-out完整privileged compiler gate | 未按最终链完成 | direct-effect held oracle完成；冻结`q_pi(P)->effect->solver`没有完成 |
| `q_V(P|L,V)`同构对齐 | 未开始 | 没有部署视频posterior checkpoint |
| full video相对language/static/endpoints增量 | 最终ECP未评 | GOMQ有历史正证据，不可代替新架构资格 |
| 除backbone外Writer联合训练 | 未开始 | 不是永久禁止，但依赖前序坐标成立 |
| structured outer credit | 历史局部实现；最终链未做 | v11--v19/MDCO证明梯度可达，不等于最终Program posterior上的outer实验 |
| process-identifying source-unseen tasks | 未建立 | feasibility已做；它是最终claim/data缺口，当前不是已证实唯一根因 |

## 5. 已确认的路线偏移与需要保留的成果

### 5.1 路线偏移

1. 把effect/evidence bank完成写成Stage 1A整体通过，遗漏了distributional `q_pi(P)`。
2. 把direct privileged effect solver直接称为完整Stage 1B，没有明确它绕过Program。
3. 由direct-effect operator non-pass直接阻止全部Program/shared inference，结论范围过宽。
4. 将process-identifying data从最终资格缺口提前成唯一下一主线，掩盖了尚未实现的Program桥。
5. v2--v24过度依赖未校准的geometry/open-loop proxy，fixed compiler和早期closed-loop顺序落实太晚。
6. GOMQ补审有价值，但它与ECP Stage 1无直接实现依赖；此前把它混入ECP叙事会造成进度误判。

### 5.2 应保留的成果

- Stage 0 v3 candidate observer、event assignment与全部训练/evaluation data path；
- Action Meta checkpoint与matched neutral control，不默认启用；
- stable source/shared carrier、task experts、independent members、successful occupancies和48-state banks；
- native owner/flow/action capture、official denoising、LoRA VJP、single-rank16 materialization与paired evaluator；
- fixed-A binding和mobile-rank4 capacity的closed-loop边界；
- v13 barrier、v19--v24互补因果锁、MDCO proxy校准和PECS exact-effect正增量；
- GOMQ 151的完整因果画像，作为最终性能/因果对照而非ECP组件。

## 6. 仍不明确、请专家重点裁决的问题

1. 最后修正案中Stage 1A/1B的精确定义是什么：direct effect bank/solver是否只是独立lower-level realizability prerequisite，
   还是Stage 1B正式输入就应当是`q_pi(P)`产生的Program/effect posterior？
2. `q_pi(P)`应如何获得可识别且固定的Program坐标？它与Program-to-effect模块应联合学习后冻结，还是需要先用额外functional
   anchor单独校准？哪些量允许来自rollout-only recovery，哪些只能进入uncertainty/prior？
3. expert所说的fixed solver或small target-local preconditioner应如何建立与冻结？现有mobile-rank4 analytic capacity很强，
   但当前三种数值operator均未过门；这些结果要求换realization objective、换operator，还是先等待Program-conditioned effects？
4. held5上的下一次完整gate应冻结哪些模块、输入什么、允许哪些训练数据？成功标准应继续相对shared43与multi-member union，
   还是需要新的Program-level可观察性门？
5. process-identifying source-unseen meta tasks应在`q_pi/compiler`之前建立，还是只在`q_V`与最终process claim前加入？
6. Stage 0的`P_lang/P_scene`是否必须重构为显式38-owner Program后才能继续，还是当前summary可由shared projection补齐？
7. Action Meta应继续只作后续matched control，还是你认为其无负面即可默认冻结启用？
8. 在新路线中，哪些历史模块可以复用，哪些必须fresh重建，才能避免重新引入deterministic mean teacher和moving decoder？

## 7. 仓库侧候选推进顺序（等待专家修正，不是active授权）

1. 冻结当前代码与结果，不再创建solver或ECP版本；先由专家重新定稿Stage边界、输入输出与最早closed-loop gate。
2. 把现有independent members、occupancy与policy effects作为evidence输入，fresh实现同构、distributional `q_pi(P)`；明确
   video-observable与recovery-only posterior分量，fit/meta有梯度，held5无task-local code或自由参数。
3. 实现唯一的`Program -> event/layer/family policy-effect distribution`桥。先在privileged链中固定observer，避免同时移动
   video representation、Program坐标和realizer。
4. 复用已验证的mobile-rank4 capacity、native VJP和paired evaluator；根据专家裁决选择固定solver或很小共享preconditioner，
   直接运行冻结`q_pi(P) -> policy-effect distribution -> solver -> LoRA`的held5 closed-loop oracle，不再用LoRA cosine筛掉首轮结果。
5. 只有完整privileged链明显高于shared、覆盖Goal/Long并保留multi-member support，才训练同构`q_V(P|L,V)`；每视频保序、
   跨视频只在Program集合层聚合，Dynamic-K真实训练。
6. 在`q_V`阶段建立learned language-only、scene/endpoints、static repeated与same-task-other对照；shuffled/reversed只用于最终冻结
   checkpoint的严格配对时序特异性测试，不进入训练或选模。
7. process-identifying source-unseen tasks按专家指定时机加入；不得用更多同task episodes或source-seen tasks冒充独立mappings。
8. 通过video增量门后，保持realizer固定，先做普通Writer参数联合训练，再按需要加入event/layer/family结构化outer credit；最后
   用全部train授权数据fresh训练并在validation8作single-checkpoint开发，Test8留到方法冻结。

## 8. 证据索引

- 全历史：`docs/research_history.md`
- v1--v24、MDCO、PECS与GOMQ复盘：`docs/ecp_stage1_iteration_retrospective_20260823.md`
- 当前设计文档：`docs/event_conditioned_policy_compiler_design.md`
- occupancy oracle：`docs/evidence/ecp_20260823/ecp_occupancy_complete_oracle_gate_20260823.json`
- fixed-A capacity：`docs/evidence/ecp_20260823/ecp_fixed_a_capacity_gate_20260823.json`
- mobile-rank4 capacity：`docs/evidence/ecp_20260823/ecp_mobile_rank4_residual_capacity_gate_20260823.json`
- mobile-rank4 solver：`docs/evidence/ecp_20260823/ecp_mobile_rank4_solver_gate_20260823.json`
- effective-update profile：`docs/evidence/ecp_20260823/ecp_effective_update_profile_gate_20260823.json`
- PECS complete trajectory：`docs/evidence/ecp_20260823/pecs_complete_trajectory_held5_gate_20260823.json`
- MDCO：`docs/evidence/ecp_20260823/stage1_mapping_diverse_compiler_oracle_tv540_gate.json`
- GOMQ causal controls：`docs/evidence/gomq_20260823/gomq_cycle2_causal_adjudication.json`

## 9. 专家最终复核裁决

### 9.1 最核心的未解问题

专家确认ECP尚未被完整实现，也没有被整体证伪。比“缺`q_pi(P)`”更深的核心缺口是
**deployment-time occupancy completion**：Writer在rollout前只能看language与action-hidden videos，不能读取未来initial、
candidate或recovery occupancy，却必须生成一套在这些未见状态上稳定工作的静态LoRA。task-local occupancy solver可作强oracle，
但不能成为deployment compiler；teacher-frame PECS可部署兼容但state coverage不足；历史amortized compiler可部署兼容却欠识别。
后继必须正面建立`video-visible Program -> global policy adaptation`这座桥。

### 9.2 最终阶段分类

| 阶段 | 输入与输出 | 当前状态 |
| --- | --- | --- |
| Stage 0-V | language/videos -> `P_lang/P_scene/P_process/rho/sigma` | 部分实现；v3只过非退化门 |
| Stage 1A-E | successful policies/occupancies/responses -> evidence particles | fold0前置资产部分完成 |
| Stage 1A-P | visible anchors + verified policy evidence -> distributional `q_pi(P_visible,Z_robust)` | 未实现 |
| Stage 1B-R0 | exact privileged effects -> LoRA lower-bound diagnostic | 多个子门完成，未强通过 |
| Stage 1B-C | Program-only、deployment-compatible fixed compiler/realizer | 未实现 |
| Stage 1B-O | `q_pi(P)` -> fixed realizer -> held LoRA | 未执行 |
| Stage 2 | language/videos ->同构`q_V(P|L,V)` | 未实现 |
| Stage 3 | 普通Writer联合训练 | 未开始 |
| Stage 4 | structured rollout outer credit | 最终链未执行 |

direct-effect realization是完整ECP的必要下游条件与lower-bound diagnostic，不是普通消融；但它读取future occupancy，因此不能
冒充Stage 1B-C。当前48-state资产改称**four-category structured occupancy panel**，不再称occupancy-complete。其antithetic
probe在保存前被平均、off-policy member response未验证为恢复target、stage-wise soft-min可拼出不存在的混合policy，均须在
下一次teacher/effect设计中修正。

### 9.3 冻结的架构原则

1. Program schema先固定为`P_lang[38,128]`、`P_scene[38,128]`、
   `P_process[8,38,128]`、`rho[8]`及结构化uncertainty；owner-specific language/scene reads必须重建。
2. 先固定Program语义、effective-update/effect coordinate和deployment-compatible realizer，再训练`q_pi`；`q_pi`通过并冻结后
   才训练`q_V`。不得先联合学习任意latent，再让decoder共同旋转寻找语义。
3. 默认输出拓扑为`rank12 stable carrier + mobile rank4 residual`。解析投影`110/120/76`已强支持其容量；rank只作数值
   参数化，不承载skill/event语义。
4. 首选固定canonical residual coordinate加小型target-local amortized realizer；realizer inference只能读取Program，不读取
   future occupancy。task-local occupancy solver只保留为上限诊断。
5. `q_pi`必须保留member/probe/uncertainty distribution；global member identity贯穿trajectory。只有经验证的skill composition
   才允许event级member切换；member-state response须带on-policy、continuation/progress与recovery-validity信息。
6. `P_visible`必须与`q_V`同构；rollout-only recovery information只进入`Z_robust`、共享robustness prior或realizer训练，不能要求
   deployment posterior逐项复制。
7. Action Meta默认关闭，只作matched control；只有明确改善probe stability、process-held或最终closed-loop才启用并冻结。

### 9.4 最终执行顺序

0. 将冻结GOMQ cycle2有效更新确定性canonicalize为真实rank16，完成一次strict400 archival baseline；不恢复训练。
1. 在任何新`q_pi/q_V`训练前完成一个process-identifying最小pair feasibility；完整family-disjoint process-meta suite在最终
   Stage 0、`q_pi`和`q_V`共同训练前建立。
2. 与process数据准备并行，使用known-success mobile projections校准effect objective、member validity与固定canonical
   rank4 residual coordinate；训练并冻结小型deployment-compatible target-local realizer。
3. fresh重建完整owner-specific Stage 0 Program，保留probe particles而非提前antithetic averaging，并以process-meta
   识别约束替代pure relative-time schedule捷径。
4. 在fit/meta与process tasks补足独立successful lineages，训练distributional Stage 1A-P `q_pi`；held folds不拟合free code。
5. 冻结Program schema、`q_pi`、carrier、coordinate、realizer与checkpoint规则，运行多fold Stage 1B-O privileged full-chain
   closed-loop gate。
6. 只有privileged full chain显著超过carrier、覆盖Goal/Long并恢复best-member gap，才训练同构deployment `q_V`。
7. `q_V`已有full-video闭环增量后，先在fixed realizer上做普通Writer联合训练，再按结构化Program/mobile coefficients加入outer
   credit；每个主要节点直接closed-loop，不再由open-loop proxy自动派生版本。
8. 用全部授权train/meta fresh训练，validation8作single-checkpoint paired400；方法与controls冻结后才打开Test8。

Phase 0--2是当前立即执行面。其余阶段的详细输入、冻结项、Gate与停止条件已经同步进active design与`task_plan.md`。
