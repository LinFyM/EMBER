# EMBER Task Plan

状态：2026-08-23 **occupancy-complete privileged realization oracle已判为non-pass；现只执行预注册的fixed-A闭环容量诊断，未启动后继Writer。**

active design：`docs/event_conditioned_policy_compiler_design.md`

已完成falsification card：`docs/ecp_occupancy_complete_oracle_card_20260823.md`

当前diagnostic card：`docs/ecp_fixed_a_capacity_card_20260823.md`

历史：`docs/research_history.md`与`docs/ecp_stage1_iteration_retrospective_20260823.md`

## Goal

让shared Writer仅根据exact task language与K条action-hidden、内部有序的正确教学视频，在rollout前一次生成唯一一套
完整38-target rank16 LoRA，使冻结source PI0.5在未见初始化上获得强、稳定、广泛且具有视频时序特异性的
zero-interaction闭环能力。正式目标仍为strict paired correct严格`>145/400`，同时满足高breadth、低churn、跨video
鲁棒、Goal/Long贡献和最终correct相对wrong/shuffled/reversed/no-video优势。

## 当前科学锁

旧learned `q_pi -> Program -> A/B hyperdecoder`家族已经关闭；不恢复v24、MDCO或小结构/超参变体。PECS只证明稀疏
video-frame exact effects能从shared43提高到58/59，但没有覆盖policy真实执行的initial/successful/candidate/recovery
occupancy，不能作为ECP最终反证。

本轮已经回答：完整闭环状态支持上的多成功策略分布可把stable carrier从`43/250`提高到`78/250`，但只覆盖3/5 tasks，
Goal/Long仍为0，oracle-normalized recovery只有`.304`且仅3/5 tasks为正，故未通过完整门。当前只分离fixed-A容量与
effect objective/calibration两个竞争解释：解析投影三个既有成功members后直接做paired closed loop；不训练video predictor、
shared compiler、joint Writer或outer credit，也不扫solver、rank或插值系数。

## Fixed boundaries

- deployment输入只有exact language与K条same-task action-hidden ordered videos；Writer只运行一次；
- source PI0.5、PaliGemma/VLM与native Action Expert冻结；
- native v3是candidate observer；历史Action Meta matched中性，只作control，不默认加载；
- shuffled/reversed不进入训练、loss或checkpoint选择，只在最终候选checkpoint冻结后评测；
- 每个condition只输出一套complete rank16 LoRA，不平均video LoRA、不部署第二adapter、不用task-ID/expert route；
- validation/test action或reward不训练共享模型；fold0 held5只作当前同面板privileged机制比较；
- 当前LIBERO只支持scene/goal/order claim；general process claim等待process-identifying source-unseen数据；
- formal train/eval来自clean pushed commit的detached frozen worktree；不重复已有source、stable carrier或兼容轨迹资产。

## Phase A — 阶段重构与合同

- [x] 冻结learned compiler家族并保留Git/formal历史；
- [x] 将Stage 0 v3降级为candidate，Action Meta降级为control；
- [x] 将Stage 1拆为equivalence identification、privileged realization、shared video inference三个门；
- [x] 将compiler合同改为`Program -> policy-effect distribution -> fixed solver -> LoRA`；
- [x] 将claim收窄到现有数据实际能识别的scene/goal/order；
- [x] 预注册首个occupancy-complete oracle的数据、参数化和直接闭环门。

## Phase B — Stage 1A policy-equivalence bank

- [x] 为fold0五项各训练一个不同seed的独立rank16 task expert；只读train actions，固定step2000，不按task选step；
- [x] 对新member跑fixed50 closed loop，五项均捕获一条strict-success完整occupancy；
- [x] 复用现有earliest/latest成功occupancy与stable carrier，不重复大规模资产；
- [x] 收集每task四类等权state bank：8 initial、24 successful、8 prior-candidate、8 recovery；
- [x] 实现官方双相机policy prefix、fixed antithetic noise及source/carrier/三个member的owner/flow/action particle缓存；
- [x] 报告member independence、success union、state/stage覆盖、disagreement和video-observable/recovery信息分界。

**Gate 1A：** 每task至少两个独立optimization lineages有strict success；48个anchor四类齐全；particle轴未被均值压平；
Goal/Long也有成功member与完整occupancy。**已通过：** 新独立members fixed250为`113`，逐task
`26/32/37/13/5`，5/5成功occupancy及五个48-state effect banks均完整。

## Phase C — Stage 1B fixed-A privileged realization

- [x] 从verified stable carrier出发，固定全部`A_c`，只优化`Delta B`，确保zero correction精确返回carrier且无A/B交叉项；
- [x] 实现stage-consistent particle soft-min、carrier no-worse barrier、source/shared preservation、trust与category balance；
- [x] 在一个fit task做真实数值/吞吐profile，只将microbatch定为4，未用held结果调科学量纲；
- [x] 从clean pushed commit并行求解held5五套task-local rank16 oracle LoRA；
- [x] 第一次完整设计后直接跑原fixed250 strict closed loop，不用geometry预筛；
- [x] 按absolute、per-task oracle-normalized recovery、carrier retention、member success union、breadth、Goal/Long裁决；
- [x] 重大失败后暂停复盘，未自动建立下一版本。

**Gate 1B：** final12至少74/250、相对carrier净增至少20、5/5非零、4/5严格胜carrier、Goal/Long各非零、carrier
retention至少33/43、overall oracle-normalized recovery至少0.35。**Realization non-pass：** final为`78/250`，absolute、净增
`+35`及carrier retention `35/43`通过；breadth `3/5`、严格胜carrier `2/5`、Goal/Long `0/0`、recovery
`35/115=.304`且仅3/5为正，因此整体失败。正式证据：
`docs/evidence/ecp_20260823/ecp_occupancy_complete_oracle_gate_20260823.json`。

## Phase C2 — Fixed-A capacity separation

- [x] 重新对齐专家最终复核与owner合同，确认fixed-A是候选carrier-preserving参数化而非不可更改的ECP核心；
- [x] 复用三个successful members完成零训练行空间审计：correction energy coverage为`83.3%--96.7%`，但expert absolute
  update coverage只有`41.5%--62.7%`，且Long高于Goal，内部几何不能直接解释共同的0分；
- [x] 预注册解析最优fixed-A投影、三个固定member arms、paired750与事前裁决边界；
- [ ] 从clean pushed frozen authority解析生成15套single-LoRA projected member adapters；
- [ ] 分别跑latest/independent/earliest原fixed250 matched panels并报告retained/gained/lost；
- [ ] 裁决fixed-A是capacity-supported、capacity-binding或mixed，再决定唯一下一科学问题。

本诊断没有optimizer、checkpoint/member选择、interpolation或video输入，不构成Stage 1C授权。card：
`docs/ecp_fixed_a_capacity_card_20260823.md`。

## Phase D — 继续被Realization non-pass阻止

- [ ] 轮换train24 fold并建立新的shared-model selection面；
- [ ] 审计或构建source-unseen adaptation meta tasks；若要process claim，必须包含真正process-identifying mappings；
- [ ] 学习`ECP Program -> effect distribution`，保持privileged recovery information不进入deployment输入；
- [ ] 固定realization operator后训练Dynamic-K video inference；
- [ ] 分阶段通过后再做除backbone外Writer普通参数联合训练；
- [ ] 之后才允许structured train/meta outer credit；
- [ ] 全train授权数据fresh训练，validation8 single-checkpoint development，最终Test8；
- [ ] final paired400补相邻稳定、same-task-other、wrong/shuffled/reversed/no-video与Dynamic-K资格。

## Done when

- 最终shared Writer满足输入墙与single-LoRA部署合同并达到正式性能/因果/稳定性资格；或
- 完成专家定义的强oracle（包括process-identifying source-unseen data）后仍触发广义ECP停止条件，并以closed-loop证据定位
  最早根本失败接口；
- 验证后的代码、配置、文档和remote-safe evidence及时合并`main`并推送，task-owned worktree/branch/temp产物清理完毕。
