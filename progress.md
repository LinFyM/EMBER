# EMBER Progress

更新时间：2026-08-23。稳定目标见`docs/current_owner_requirements.md`，active计划见`task_plan.md`，历史见
`docs/research_history.md`。

## Current authority

- active design：`docs/event_conditioned_policy_compiler_design.md`；
- completed falsification card：`docs/ecp_occupancy_complete_oracle_card_20260823.md`；
- formal adjudication：`docs/evidence/ecp_20260823/ecp_occupancy_complete_oracle_gate_20260823.json`；
- active goal：完整实现并验证EMBER-ECP；goal仍在进行中；
- canonical workspace：本仓库`main`；本卡GPU工作已经结束，当前处于科学暂停而不是运行或训练状态。

## Current scientific state

- ECP核心假设未被证伪；native Stage 0 v3只作为candidate observer。它通过non-degeneracy与task separation，但未完全通过
  process semantics和probe invariance。现有数据只支持scene/goal/order claim。
- 历史Action Meta matched结果中性，现保留为control而非默认authority。
- learned `q_pi -> Program -> A/B hyperdecoder`家族已经关闭。v24、MDCO以及width/rank/head/fusion/LR/seed后继均不再恢复。
- occupancy-complete oracle已经补齐PECS缺少的真实initial/successful/candidate/recovery occupancy以及独立成功策略。它从
  stable carrier `43/250`提高到`78/250`，说明realization仍有真实增量；但breadth只有3/5，Goal/Long仍为0，完整门失败。
- Stage 1A teacher bank通过：五个seed37独立members fixed250为`113`，逐task为`26/32/37/13/5`；Goal/Long均有strict
  success，五个48-state、三particle effect banks完整。当前失败不能再归因于没有独立teacher或没有闭环occupancy。
- Stage 1B为Realization non-pass：final逐task`36/12/30/0/0`，只在2/5 tasks严格胜carrier；carrier retention为
  `35/43`，但oracle-normalized recovery为`35/115=.304`且仅3/5 tasks为正。
- GOMQ历史151只作为“强carrier + 小有效更新可保留support”的结构证据，不恢复其Writer或checkpoint作为答案。

## Verified reusable assets

- source policy、normalization、tokenizer与fixed evaluation assets保持canonical，不重训；
- fit19 stable shared carrier：
  `runs/outputs/pi05_train24_stable_shared_prior_formal_r6_v48_e948fca_gpu02p123467_20260821/shared_prior.safetensors`，
  held5 fixed250为43；
- fold0 held5 source/direct-earliest/direct-latest为`21/74/108`；
- 新独立members为`113/250`；earliest/latest/independent逐row success union为`146/250`，逐task
  `38/40/41/16/11`；
- 24-task task-expert bank、三套member成功occupancy、五个48-state effect banks及五套final LoRA完整保留；
- Stage 1 authority含95 tasks、118 successful members与`[118,8,32]` phase response；
- PECS local/trajectory adapters及paired rows完整保留，可作为candidate occupancy生成policy；
- validation8 sealed task-local rank16 oracle为250/400，只作ceiling evidence。

## Corrections now applied

- 当前不再把更多task数量等同于更多可识别信息：71个LIBERO-90 tasks全部被source见过，现成source-unseen mapping只有
  target train24；现有BDDL没有证明same-endpoint/different-required-process pair。
- held5只用于与`43/58/59/74/108`的预注册机制比较，没有用于训练shared模型或选择solver量纲。
- earliest/latest不再被冒充为独立lineages；五个不同seed、固定step2000的独立task experts已经补齐并全部产生strict success。
- oracle直接在PI0.5官方双相机rollout observations上查询source/carrier/三个expert particles，已经覆盖initial、successful、
  prior-candidate与recovery states。
- realization从stable carrier出发，固定rank16 A、只求Delta-B，使zero correction精确返回carrier、effective update严格相加且
  不产生A/B交叉项。
- 第一次完整oracle已经直接closed loop，没有geometry预筛；重大失败后现已暂停，没有创建下一版本。

## Completed execution

1. 五个独立experts固定step2000完成，fixed250及成功occupancy完成；
2. 五个48-state effect banks完成，每项保留initial8、successful24、candidate8、recovery8及三member轴；
3. fit ordinal71 profile完成，只把实现microbatch固定为4；12-step objective ratio为`.59779`，峰值18.94 GB；
4. 五个held solvers从clean pushed `c2aaac1`完成，objective ratio为
   `.5040/.5667/.5278/.6055/.4373`，均无trust penalty；
5. original fixed250 strict paired panel完成，250行相对source/carrier/direct/independent均无episode、seed、language或
   policy-noise common-prefix mismatch；
6. Gate 1B判为Realization non-pass并暂停，未补step10/11、未扫solver、未训练video predictor。

## Current implementation milestone

- 已接通真实PI0.5 official prefix cache与10-step denoising路径：在同一policy observation/noise上与official action输出的
  max-abs差约`0.00668`、RMS约`0.00208`，属于允许的BF16/batch reduction差异；
- native owner为`[batch,38,4,128]`，同时保留`[batch,10,50,32]` flow与`[batch,10,50,7]` integrated action；
- fixed-A路径只为38个target建立`Delta B`叶子；真实PI0.5 smoke中38/38梯度均finite且非零，峰值allocated约18.72GB；
- 已实现48-state effect bank、stage-consistent particle soft-min、carrier barrier、preservation/trust及统一12-step solver；
- profile没有借held5做量纲选择：另用ordinal71/global2独立member、四类occupancy和同构solver冻结实现合同；
- projection helper现可直接解析并行solver的per-task子目录，不再需要临时symlink surface。

## Current pause and unresolved interface

- 最早失效接口已从teacher/state coverage收窄到realization：当前owner/flow/action effect distance加fixed-A Delta-B求解器，
  即使inner objective在Goal/Long明显下降，也没有进入它们的closed-loop success basins；
- fixed-A capacity与effect objective/calibration仍是两个竞争解释。本卡只关闭二者当前组合，不把局部失败外推成整个ECP反证；
- recovery occupancy是rollout-only privileged information，任何后续deployment Program仍不得读取；
- 现有数据仍不足以最终检验general process understanding；process-identifying source-unseen meta data仍是未来方法资格前置；
- 按预注册合同，当前等待owner/专家层面的架构复盘；没有获得新科学裁决前不启动successor、小扫或Stage 1C。
