# EMBER Progress

更新时间：2026-08-23。稳定目标见`docs/current_owner_requirements.md`，active计划见`task_plan.md`，历史见
`docs/research_history.md`。

## Current authority

- active design：`docs/event_conditioned_policy_compiler_design.md`；
- active falsification card：`docs/ecp_occupancy_complete_oracle_card_20260823.md`；
- active goal：完整实现并验证EMBER-ECP；goal仍在进行中；
- canonical workspace：本仓库`main`；当前没有task-owned worktree或active GPU job。

## Current scientific state

- ECP核心假设未被证伪；native Stage 0 v3只作为candidate observer。它通过non-degeneracy与task separation，但未完全通过
  process semantics和probe invariance。现有数据只支持scene/goal/order claim。
- 历史Action Meta matched结果中性，现保留为control而非默认authority。
- learned `q_pi -> Program -> A/B hyperdecoder`家族已经关闭。v24、MDCO以及width/rank/head/fusion/LR/seed后继均不再恢复。
- PECS在同一held5 fixed250上由stable carrier43提高到58/59，证明policy-effect realization有真实增量；但breadth仅3/5、
  Goal/Long为0，且只在教学视频稀疏support frames上拟合，没有覆盖candidate/recovery occupancy，所以不能代替新的强oracle。
- GOMQ历史151只作为“强carrier + 小有效更新可保留support”的结构证据，不恢复其Writer或checkpoint作为答案。

## Verified reusable assets

- source policy、normalization、tokenizer与fixed evaluation assets保持canonical，不重训；
- fit19 stable shared carrier：
  `runs/outputs/pi05_train24_stable_shared_prior_formal_r6_v48_e948fca_gpu02p123467_20260821/shared_prior.safetensors`，
  held5 fixed250为43；
- fold0 held5 source/direct-earliest/direct-latest为`21/74/108`；
- 24-task task-expert bank与earliest/latest成功occupancy完整保留；
- Stage 1 authority含95 tasks、118 successful members与`[118,8,32]` phase response；
- PECS local/trajectory adapters及paired rows完整保留，可作为candidate occupancy生成policy；
- validation8 sealed task-local rank16 oracle为250/400，只作ceiling evidence。

## Corrections now applied

- 当前不再把更多task数量等同于更多可识别信息：71个LIBERO-90 tasks全部被source见过，现成source-unseen mapping只有
  target train24；现有BDDL没有证明same-endpoint/different-required-process pair。
- held5复用仅为与43/58/59/74/108做一次单变量机制比较，不再用于调shared模型。若oracle通过，Stage 1C前必须轮换fold。
- earliest/latest来自同一优化lineage，不能冒充多个独立successful policies；Stage 1A将为五个held tasks各补一个不同seed、
  固定step2000的独立task expert，并捕获其strict-success occupancy。
- 下一次LoRA实验不是video Writer：它直接在PI0.5官方双相机rollout observations上查询source/carrier/多个expert particles，
  覆盖initial、successful、prior-candidate与recovery states。
- realization从stable carrier出发，固定rank16 A、只求Delta-B，使zero correction精确返回carrier、effective update严格相加且
  不产生A/B交叉项。
- 第一次完整oracle直接closed loop，不再由geometry预筛；重大失败后暂停，不自动创建下一版本。

## Immediate execution order

1. 完成Stage 1A独立expert配置与official occupancy/effect bank实现；
2. 从clean pushed commit并行训练五个独立expert并捕获成功occupancy；
3. 实现fixed-A particle-equivalence solver并做一个真实数值/资源profile；
4. 固定合同后并行求解held5五套oracle LoRA；
5. 直接运行原fixed250并按falsification card裁决；
6. 暂停复盘。只有通过才轮换fold并进入shared video-to-effect学习。

## Current blockers and risks

- 新独立expert必须在Goal/Long上实际产生strict success；若没有，Stage 1A不完整，不能用checkpoint数量冒充policy diversity；
- recovery occupancy是rollout-only privileged information，后续不得泄漏进deployment Program；
- 现有数据不足以最终检验general process understanding。即使本次realization oracle通过，process-identifying meta data仍是
  Stage 1C方法资格的前置条件；
- fixed-A可能限制task-specific row space，但这是本卡明确、可证伪的首轮carrier-preserving参数化，不通过时不能靠小扫掩盖。
