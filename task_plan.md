# EMBER Task Plan

状态：2026-08-22 **active — EMBER-ECP architecture implementation**。

唯一active design：`docs/event_conditioned_policy_compiler_design.md`。

旧16维functional-adaptation、phase decoder、shared12/task4 residual、single-direction outer credit和最初送审的
dual-time transport方案均已封存，不再启动formal训练。它们的实现与checkpoint只作为可复用资产、历史反事实和evaluator
基础。历史事实见`docs/research_history.md`，当前资产与执行状态见`progress.md`。

## Goal

全面实现并验证 **EMBER-ECP（Event-Conditioned Policy Compiler）**：让一个shared Writer仅根据exact task language与K条
action-hidden、内部有序的正确教学视频，在rollout前一次生成唯一一套完整38-target rank16 LoRA，使冻结source PI0.5在
未见初始化上获得强、稳定、广泛且具有视频时序特异性的zero-interaction闭环能力。

完整落实专家最终意见与owner边界，复用既有昂贵资产，按因果接口逐阶段实现和裁决。正式目标为strict paired correct
严格`>145/400`，并且高breadth、低churn、same-task跨视频鲁棒、Goal/Long贡献及最终correct相对wrong/shuffled/
reversed/no-video优势必须同时成立。若经过合理结构、数据、联合训练与outer-credit修正仍存在根本障碍，则以完整
证据明确失败接口并转向最有依据的替代路线。

## Done when

- EMBER-ECP成为仓库唯一canonical Writer训练与部署运行面，旧活动路径由Git、sealed configs与formal artifacts复现；
- native observer、Action Meta-LoRA独立对照、privileged policy teacher、target-family compiler、Dynamic-K video encoder、
  Writer联合训练和structured outer credit均已实现并得到相应阶段裁决；
- compiler在video inference进入前通过task-level leave-out closed-loop oracle gate，而非只通过latent/BA/flow loss；
- final Writer只读language与action-hidden videos，输出一套完整LoRA，不读取task ID、teacher action/state/reward或第二expert；
- final training使用全部授权train tasks，validation8仅以部署输入做single-checkpoint development evaluation，Test8在方法冻结后
  才使用；
- final checkpoint报告paired400、per-task/suite、breadth、retained/gained/lost、churn、相邻稳定、same-task-other、
  Dynamic-K以及最终无梯度wrong/shuffled/reversed/no-video controls；
- 方法通过并达到预期结果，或合理范围内的结构、数据、joint training与outer credit均完成后，以证据证明最早根本失败接口；
- 验证后的代码、配置、文档和remote-safe evidence均及时合并`main`、推送远程，并清理task-owned worktree/branch/temp产物。

## Fixed boundaries

- exact language与K条same-task、action-hidden、内部有序正确视频是唯一deployment输入；Writer只在rollout前运行一次；
- source PI0.5、PaliGemma/VLM与native Action Expert冻结；不训练Text/VL Meta-LoRA；
- Action Meta-LoRA必须独立尝试。先建立native observer，再单独校准shared Action Meta-LoRA；没有matched负面效果就采用并
  永久冻结，不能与compiler或video inference共同旋转；
- shuffled/reversed不进入训练、loss、阶段选择或checkpoint选择，只在最终候选checkpoint冻结后作无梯度时序特异性评测；
- Dynamic-K若声明支持，训练真实覆盖每个cardinality；每条video独立保序，跨video只在event-aligned Program层聚合；
- 一个condition只生成一套完整rank16 LoRA；不平均video LoRA、不部署第二adapter、不用expert/task-ID route或checkpoint
  fusion；
- validation/test action或reward不训练共享模型。已完成的validation8 task-local oracle只作ceiling evidence；
- held5是train24内部privileged/compiler leave-task-out诊断，不替代validation8。最终架构冻结后使用全部train24重新训练，
  validation8只以language+action-hidden video评测；
- formal train/eval来自clean pushed commit的detached frozen worktree；大训练前live检查GPU和独立storage quota；
- 不重复训练canonical source、expert bank、successful trajectories或其它合同兼容的大资产。

## Evidence ladder

1. **Contract gate**：信息墙、tensor owner、single-LoRA、task roles、pairing和冻结backbone正确。
2. **Observer gate**：正确视频的event/owner结构跨episode、速度与替代probe稳定；native与Action Meta-LoRA得到独立裁决。
3. **Compiler oracle gate**：冻结`q_pi + compiler`在train24 leave-task-out tasks上显著保留direct successful-policy support。
4. **Video inference gate**：冻结compiler后，action-hidden full video相对language+scene/endpoints创造新的闭环success，且跨video/
   Dynamic-K稳定。
5. **Joint Writer gate**：除backbone/privileged teacher/已冻结observer calibration外的普通Writer参数联合训练，相对frozen-
   compiler checkpoint提高或至少保持absolute、breadth、retention和video necessity。
6. **Outer-credit gate**：structured train/meta reward相对matched joint Writer提高held/validation闭环，而非只降低train loss。
7. **Method qualification**：single checkpoint在paired400、all suites、Long、breadth、相邻稳定、same-video及最终时序controls
   上共同成立。

## Phase 0 — Active design与实现合同（complete）

- [x] 专家最终复核确认：Action horizon是joint future coordinate，不是第二因果时间轴；
- [x] 核心机制改为monotone video-event segmentation + Event-Conditioned Horizon Binding；
- [x] Program改为`P_lang / P_scene / P_process[event,target-family] + rho + sigma`；
- [x] deterministic `P*`改为privileged `q_pi(P)`与deployment `q_V(P|L,V)`的分布式共同识别；
- [x] owner确认Action Meta-LoRA必须独立尝试、无负面效果即采用；
- [x] owner确认shuffled/reversed只作最终checkpoint时序特异性评测，不进入训练；
- [x] held5与validation8角色已澄清；
- [x] 增加Stage 3最终普通Writer参数联合训练；
- [x] 完成active design、owner requirements、progress与模块ownership同步；
- [x] 完成结构门，冻结Stage 0/1最小实现范围与首个checkpoint schema。

**Gate 0：** 文档、tensor contract、数据角色与代码owner一致；旧16维运行面不能被新入口隐式复活。

## Phase 1 — Native observer与Action Meta-LoRA对照

- [x] 实现每帧以exact language、真实image prefix、fixed Gaussian suffix `[50,32]`、flow time `u=1`运行native graph；
- [x] 捕获18层Action input/residual，立即投影为38-owner × 50-horizon compact lattice；
- [x] 实现task-grounded visual transition、局部event candidates、双向event/lattice co-attention；
- [x] 实现固定容量8、动态presence的learned ordered semi-Markov segmenter；
- [x] 在真实冻结PI0.5上完成单帧forward、梯度边界、显存与吞吐smoke；
- [x] 接通71个non-held与fit19的task-equal correct-video训练数据、跨episode action grounding、保序速度视图、
  动态cost-balanced多卡macro与exact-resume checkpoint；
- [x] 实现固定train24 observer面板：fit19/held5各自报告跨episode、2x保序速度、antithetic probe与跨task margin；
- [x] 使用correct videos、cross-episode teacher actions和不改event order的速度扰动完成首个native observer macro10；
- [x] 固定面板裁决首版native observer为Gate 1科学不通过：task/event表示近全局坍缩，未进入compiler；
- [x] 实现独立shared Action Meta-LoRA训练、checkpoint与matched panel运行面；正式科学arm等待修正后的native authority；
- [x] 以固定速度归一occupancy presence和逐帧posterior action reconstruction修正最早失效接口，并通过真实profile；
- [x] 从fresh clean-pushed authority训练v2 native macro10并重跑同一固定面板；固定presence消除了全局分母捷径，但
  posterior在macro6前仍坍缩为单event，Gate 1继续失败；
- [x] 以8 tasks×2 views的真实cross-episode action targets确认有序8-bin oracle相对最优常数预测降低80.57% MSE，排除
  “teacher action本身没有phase信号”；
- [x] fresh实现v3 pre-segmentation frame-action grounding：在event pooling前直接约束frame-bound owner evidence，并与
  event reconstruction共享action decoder；建立frame phase grounding前关闭premature consistency/uncertainty/sparsity/
  entropy项；
- [x] 在真实83/42帧pair上profile v3的显存、梯度、frame/event action loss与事件数；
- [x] 从clean pushed authority训练v3 macro10并运行同一固定observer panel；48/48 nearest cross-task margins为正，native
  首次非退化并授权进入Action Meta matched arm，但antithetic probe稳定性仍待裁决；
- [x] 固定native baseline后单独校准shared Action Meta-LoRA；matched panel无可复现负面效果、也无显著收益，按owner规则采用；
- [x] 永久冻结`native v3 macro10 + Action Meta v3 macro10`为observer authority并保存remote-safe裁决证据。

**Gate 1：** task/event结构必须跨episode和probe稳定；Action Meta-LoRA不能引入task-ID/noise shortcut或性能退化。

## Phase 2 — Privileged policy teacher与compiler

- [x] 实现visible-event-anchored `q_pi(P)`，输入多个successful members、完整successful occupancy、policy response、
  reliability与uncertainty；首个warm-start明确排除未经验证且历史闭环反向的learner/recovery states；
- [x] 实现38 target-owner、rank-query、family-specific A/B heads和强layer-local bias的compiler；
- [x] 保留完整rank16 prior/full LoRA，不恢复shared12/task4硬拆或full A/B相加；
- [x] 接通train24固定19/5 fold的task-equal world-size6训练合同，held5不拟合code、不更新teacher/compiler；真实普通/
  functional单卡profile均已通过；
- [ ] 从clean pushed frozen authority正式训练fit19并保存相邻task-visit checkpoints；228节点已完成，570 exact-resume进行中；
- [ ] held5 oracle Program生成一套LoRA并直接closed-loop，报告source/direct/generated三联面板；228节点已完成且未过门，等待
  首个functional 570节点复验；
- [ ] 轮换固定fold，确认不是单一held5偶然结果。

**Gate 2：** 默认要求generated显著高于source，direct success retention `>=75%`，direct gain retention `>=60%`，增量跨tasks，
Goal/Long不能系统为零。失败则停在Program/compiler，不训练`q_V`。

## Phase 3 — Dynamic-K video-to-Program

- [ ] 冻结source、observer authority、`q_pi`和compiler；
- [ ] 只用correct action-hidden videos训练`q_V`，真实覆盖K1--K4并用同task跨episode action query；
- [ ] 每video先独立形成完整Program，再做event alignment、mean/variance/presence/bounded residual聚合；
- [ ] 用Program分布对齐、compiler后policy response、cross-video/K consistency和task-equal functional panels训练；
- [ ] fixed checkpoint后运行language-only、language+scene、first+final与same-task-other信息路径baseline；shuffled/reversed
  仍保留到最终候选checkpoint；
- [ ] 多fold通过后冻结架构/loss，使用全部train24与授权meta data fresh训练final development checkpoint；
- [ ] validation8只用language+action-hidden videos做正式评测。

**Gate 3：** full video相对learned language+scene/endpoints必须创造新success；same-task-other retention至少90%；Goal/Long有
真实贡献，收益不能只集中Object。

## Phase 4 — Writer联合训练

- [ ] 从Stage 2已通过checkpoint初始化；
- [ ] 冻结PaliGemma/native Action Expert/source policy、train-only `q_pi`及已裁决observer calibration；
- [ ] 解冻全部普通deployment Writer参数，包括frontend projections、transition、binding、segmenter、K aggregator、`q_V`、
  compiler与LoRA heads；
- [ ] compiler用慢但非零学习率，`q_pi`分布与Stage 1 multi-state policy support作为坐标/行为锚；
- [ ] 用correct-video cross-episode action与task-equal functional evidence联合收敛；
- [ ] 与Stage 2 frozen-compiler checkpoint严格配对，不因“全部解冻”自动promote。

**Gate 4：** joint checkpoint必须提高或至少保持absolute、breadth、retention、same-video和full-video necessity；出现moving-
coordinate退化则保留Stage 2并定位最早接口。

## Phase 5 — Structured outer credit

- [ ] 仅在Gate 4通过后，在授权train/meta simulator按event/layer/family/phase分配closed-loop credit；
- [ ] source、backbone、observer calibration与validation/Test边界保持冻结；
- [ ] matched joint-only/outer arms快速进入held/validation closed-loop；
- [ ] 不复用旧single-direction 16D estimator，不以train reward或critic loss选模型。

**Gate 5：** outer必须带来paired absolute、breadth或retention净改善且不破坏video necessity；否则关闭该实现。

## Phase 6 — Formal qualification或根本失败裁决

- [ ] single-checkpoint paired400、per-suite/task、breadth、Long、top-task concentration；
- [ ] 相邻checkpoints、retained/gained/lost、churn、Jaccard；
- [ ] same-task-other与Dynamic-K nested sets；
- [ ] 最终checkpoint冻结后一次性运行wrong/shuffled/reversed/no-video/static/endpoints controls，不产生梯度或选模；
- [ ] 多fold复现；方法冻结后进入Test8；
- [ ] 若合理结构/data/joint/outer修正后仍触发active design停止条件，转入证据最强的替代路线。

## Current next actions

1. 完成从228到570 visits的exact-resume，使Stage 1首次纳入multi-state functional policy-response监督；
2. 物化570 single-LoRA并复用同一held5 fixed250 source/direct/shared-prior rows，按`.75/.60` support-retention门裁决；
3. 若570出现广泛闭环/support改善，继续预注册1140节点并检查相邻稳定性；若仍无机制性移动，定位`q_pi`、compiler geometry或
   functional-support接口中最早失败处，不用无关超参小扫拖延；
4. 每个节点及时更新remote-safe进度、清理task-owned临时脚本/日志，并推送`main`；
5. 只有Gate 2通过并经预注册fold复现后才进入Dynamic-K `q_V`，不触碰validation8/Test8。
