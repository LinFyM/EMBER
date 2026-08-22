# ECP Stage 1 Mapping-Diverse Compiler Oracle（MDCO）

状态：2026-08-23 **唯一Stage 1 falsification contract；derived authority与校准图已就绪，formal 540尚未启动**。
它不是`v25`，也不是一套新的EMBER架构；它只回答v1--v24复盘后仍未回答的一个问题。

## 1. 唯一问题

> 在保持EMBER-ECP的frozen observer、event × layer × target-family Program、privileged `q_pi`、layer/family-local
> direct-absolute single-surface compiler和single rank16 LoRA合同不变时，把shared Stage 1看到的独立
> task-to-successful-policy mappings从19扩大到90，能否让冻结`q_pi + compiler`在完全留出的target held5上直接生成
> policy-effective LoRA？

这轮唯一主要科学变量是**独立policy mapping diversity**。同task更多视频、更多状态和两条trajectory只增加occupancy覆盖，
不计作更多mapping；不同时改变rank、width、probe、Program轴、fusion、loss family、source或部署接口。

若这轮仍出现“held生成LoRA不超过shared、Goal/Long为0、direct support大面积丢失”的同类closed-loop签名，则结论不是
“还需要v25/v26继续调结构”，而是：**更多现有LIBERO task mappings本身不足以识别当前Program/compiler接口**。

## 2. 数据与信息墙

shared-gradient fit集合固定为90个task mappings：

- 71个`configs/libero90_nonheld_meta_v1/protocol.json`已审计、与target40零重合的LIBERO-90任务；
- train24 fold0的19个fit tasks；
- 两组按task等权，多个members或trajectories不增加该task权重；
- asset key使用`source90:<task_id>`与`target40:<global_task_id>`命名空间；数值task ID只负责sampler/asset ownership，永不进入
  `q_pi`、Program或compiler。

held集合仍是train24 fold0的global task `[0, 9, 18, 25, 36]`：

- held successful adapters、actions、states、responses和reward只供冻结`q_pi + compiler`前向及无梯度诊断；
- 不拟合held code，不更新shared参数，不按held结果选择video/member；
- validation8与Test8 actions/reward读取和梯度均为0；
- 71个source-seen meta tasks只能训练shared mapping，不能替代held5的source-unseen结论。

现有资产审计见`docs/evidence/ecp_20260823/stage1_mapping_diversity_asset_audit.json`。不重新训练71-task experts，也不重跑
3550-row全量评测。每个nonheld task至多复跑两个已有结果中已经成功的fixed rows，优先一条source→direct gained与一条
retained-success，用于完整successful occupancy；最多142条rollout。这里的两条trajectory仍只算一个task mapping。

## 3. 固定的方法面

以下内容在本轮不可作为调参变量：

1. source PI0.5、normalization、38-target rank16 LoRA合同全部冻结；
2. Stage 0 native v3 + Action Meta v3 observer authority永久冻结；
3. Program仍为language/scene/process与event × owner/layer × target-family结构，event存在和uncertainty保留；
4. `q_pi`从visible event anchors与privileged successful-policy evidence输出Program分布，不读取task-ID route，不把raw A/B或
   单条response当作确定性`P*`；
5. compiler保留v24已经落实的target-local、layer-resolved、direct-absolute、prior/full同一surface；无template bypass、
   shared carrier、第二adapter或checkpoint fusion；
6. `q_pi`与compiler只在90个fit mappings上按专家Stage 1合同共同识别；到预注册节点后一起冻结，held5前向期间任何参数都不动；
7. objective继续使用cross-episode successful action、multi-state owner/flow response、member uncertainty、source/shared
   baseline-relative preservation、prior/full反事实，以及90-task等权的fit simulator success/progress；closed-loop credit沿
   event/owner/family结构分配，不恢复global 16D单方向估计；raw/effective LoRA cosine只作诊断；
8. shuffled/reversed不参与本轮训练、checkpoint选择或Stage 1 oracle。

允许的工程变化只有把现有train24-only Stage 1 data/support owner一般化为上述namespaced多域authority，并去掉写死的24-task、
47-member与`range(24)`假设。必须替换canonical owner，不能新增一套平行Stage 1 trainer。

## 4. 首个信息节点

正式训练从clean pushed commit和detached frozen worktree启动。先完成90个fit tasks各6 visits、总计540次dense
policy-evidence visits；随后在同一冻结数据合同上完成一次90-task等权structured success/progress calibration，每task的reward
贡献相等。world-size按live GPU选择。该节点对应v24的“每task 6 visits”信息量并补齐专家Stage 1明确要求的fit closed-loop
目标，不用loss、raw geometry或support proxy提前挑checkpoint。

该首节点只物化一个single checkpoint：

- held5每task由冻结`q_pi`读取预登记K=2视频和全部授权privileged evidence，输出一个posterior；
- compiler取posterior mean，一次生成唯一一套完整rank16 LoRA；
- 同一held task不生成多套member LoRA、不选择LoRA、不平均LoRA；
- 直接运行既有fixed 50 states/task、seed与policy-noise的strict paired250。

同时计算full/prior的open-loop functional support与LoRA geometry，但它们不再拦截closed loop。本次paired250用于校准support
proxy：若support相对shared的per-task符号与closed-loop增量在少于4/5 tasks一致，后续不得再用该proxy替代早期闭环裁决。

## 5. 唯一通过门

primary reference为既有`direct_latest=108/250`，因为它是同一预登记member family中更强的task-local successful-policy面；
`direct_earliest=74/250`只报告等价类鲁棒性。source与stable shared分别为`21/250`和`43/250`。

MDCO通过必须同时满足：

1. candidate相对source严格配对显著改善，5/5 tasks不低于source且至少3/5严格提高；
2. 保留至少75%的direct-latest成功，即至少`81/108` rows；
3. 保留至少60%的direct-latest source-failure gains，即至少`58/96` rows；
4. 保留至少75%的stable-shared成功，即至少`33/43` rows，并且candidate总分严格高于43；
5. breadth@1为5/5，Goal与Long都至少产生一个成功，而不是只要求二者不同时为0；
6. 增量不能只来自held task0；报告per-task、retained/gained/lost、churn及与earliest/latest member的成功集重合。

raw A/B reconstruction、own-direct cosine、retrieval、norm ratio、training loss和support aggregate均不能单独promote或否决。

## 6. 一次修正上限与停止条件

首节点后只有一种科学性追加：在**不改任何结构、数据、loss、LR、rank、width、seed或video选择**的前提下exact-resume到
每task 12 visits（总计1080），而且仅当540节点已经同时满足：

- candidate严格高于shared；
- breadth至少4/5且Goal、Long已有非零；
- direct success retention至少65%、direct gain retention至少50%，即只差强门不超过10个百分点。

不满足这些near-pass条件就立即停止当前compiler family。1080节点仍未过完整门也立即停止；不再运行第三节点，不扫超参，
不以另一个open-loop proxy继续延长。发现信息墙、task权重、checkpoint载入或paired-row等工程合同违反时，该run判无效并只允许
修复真实违反后fresh重跑；这不算科学修正，也不能把有效负结果重新解释成bug。

## 7. 裁决后的唯一分支

- **通过：** 冻结`q_pi + compiler`，再轮换一个预登记train24 19/5 fold验证不是fold0偶然；复现后才进入Dynamic-K `q_V`。
- **失败：** 明确淘汰“仅靠增加现有71个source-seen meta mappings即可挽救当前ECP Stage 1 compiler”这一假设。保留Stage 0、
  local oracle和数据资产，但停止同类compiler版本增长；重新回到successful-policy等价类/Program可观察性或替代路线，而不是
  继续局部架构修补。

这份合同把第一次昂贵裁决重新放回source-unseen closed loop，并使一次实验同时回答“19 mappings是否是主要瓶颈”以及
“open-loop support proxy是否有资格继续筛选checkpoint”。
