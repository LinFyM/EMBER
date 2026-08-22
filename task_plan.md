# EMBER Task Plan

状态：2026-08-23 **active goal — v24 closed; MDCO occupancy captured; combined authority construction in progress**。

长期research design authority：`docs/event_conditioned_policy_compiler_design.md`。当前唯一Stage 1执行合同是MDCO，不建立
v25；v24后的复盘与后继授权条件见`docs/ecp_stage1_iteration_retrospective_20260823.md`。

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
- [x] 从clean pushed frozen authority完成fit19的228/570/1140相邻task-visit checkpoints及两次exact-resume；
- [x] held5 oracle Program在三个节点各生成一套LoRA并完成source/shared/direct/generated fixed250严格配对闭环；首版
  23→27→27，Goal/Long持续为0，Gate 2失败；
- [x] 定位首版最早接口：`q_pi`全局`.1` residual幅度不足，compiler把仍有差异的Program压成跨task cosine`.9968`的近共享LoRA；
- [x] 用content-gated `q_pi`与prior/full absolute compiler替换首版唯一运行面，加入gauge-canonical coordinate warm-start，
  删除首版active config/schema/evaluator路径并通过聚焦CPU合同；
- [x] 完成v2真实K2 profile与fresh 228-visits训练；预注册几何门仍失败，未运行held closed loop；
- [x] 切断Program/compiler中的address-to-output捷径并从第一个update启用successful-policy functional response，完成真实
  K2+functional profile；
- [x] 从fresh clean pushed authority完成v3的228-visits geometry gate；candidate已非坍缩但own-direct只有`.0128`、
  `2/24`检索正确，故未运行held5；
- [x] 保持同一Program/q_pi/compiler结构，实现228-visits coordinate bootstrap并用真实K2单卡profile确认零functional cache与
  正确objective phase；
- [x] 从fresh clean pushed authority完成v4 228-visits与24-task geometry gate；监督坐标loss下降且跨task输出已分离，但
  own-direct仅`.0184`、自身检索`1/24`、norm ratio`.0913`，故未运行held5；
- [x] active v5保持零内容不能写LoRA的结构不变量，以target/rank query乘性调制cross-attended Program content，并完成
  真实K2单卡coordinate profile；
- [x] 从fresh clean pushed v5 authority完成同一228-visits coordinate gate；own-direct虽由`.0184`提高到`.0821`，但仍低于
  nearest-other `.1077`，自身检索仅`3/24`、norm ratio `.0864`，未运行held5；
- [x] 保持v5 Program/compiler content-address合同，重建专家要求的policy-support teacher：successful occupancy、经过
  success/一致性加权的learner occupancy、source/shared support、多成员集合一致性与fit-task reward/progress共同进入Stage 1；
- [x] 完成v6 retained implementation：full-layer response bank、五通道q_pi support attention、successful/learner交替functional
  loss与source/shared局部support；删除v5 active config和successful-only panel owner，聚焦CPU合同通过；
- [x] 从clean pushed authority并行生成一次可复用的train24 policy-support bank，并完成successful/learner双分支真实
  capture/profile；
- [x] 从fresh policy-support authority完成228-visits短程信息量节点与24-task物化；输出幅度恢复但task方向门失败，按门不运行
  held5、不延长同曲线；
- [x] 对冻结single checkpoint运行完整successful/learner panel support audit：24/24任务优于source但24/24差于stable shared，
  证明full absolute surface丢失已有support；
- [x] 用differentiable low-rank union把stable shared effective update与generated residual合并并重新压回一套rank16 LoRA，
  并通过successful/learner两条真实BF16训练路径profile；
- [x] 完成v7 fresh 228-visits、24-task物化与冻结全bank audit；相对shared从v6的`1.400/1.277`改善到
  `1.024/1.100`，但只在`9/19、1/5` tasks胜出，support门仍失败；
- [x] 保持v7全部结构与数据，fresh训练取消direct BA/canonical参数坐标梯度的v8 functional-equivalence union，并复跑同一
  冻结support门；fit/held均在0个task胜过shared，且发现228节点访问数为每task `5--18`而非task-equal 12；
- [x] 修正formal schedule，使每个可裁决prefix按6个visit rounds成块、兼顾task balance与cost-balanced rank分配；fresh复验
  的19个fit task均恰好12 visits，但fit/held仅`2/19、2/5` tasks胜过shared，v8最终关闭；
- [x] 用bounded、exact-prior的policy-functional rank selector替换唯一active compiler中的Frobenius top-SVD；删除旧v8
  active config/union调用并通过40项聚焦合同；
- [x] 完成selector successful/learner真实双visit profile；两步均finite、约2.3/2.1秒，selector从exact-zero安全打开；
- [x] 从clean pushed authority按balanced schedule fresh训练228 visits并物化24 tasks；norm受控但cross-task cosine`.99779`、
  own retrieval `1/24`，几何门失败；
- [x] 对冻结single checkpoint复跑308-panel support gate；fit aggregate略优于shared，但fit/held breadth仅`10/19、2/5`且held
  aggregate为`1.00692x`，v9按门关闭；
- [x] 保持bounded exact-prior rank selector与全部训练变量，只把compiler改为language/scene-conditioned、process-value-only
  reader；删除v9 active路径并通过41项聚焦合同；
- [x] 完成v10真实successful/learner双visit profile；两步梯度finite、约2.3/2.1秒，selector从exact shared安全打开；
- [x] 从clean pushed v10 authority完成fresh balanced 228 visits和24-task geometry；task-equal与数值合同通过，但own retrieval仍为`1/24`；
- [x] 完成同一308-panel frozen support audit；fit改善到`.96892x` shared和`12/19` breadth，但held仍为
  `1.00285x`和`2/5`，v10按门关闭；
- [x] 实现并裁决 **Stage 1 OCPB v11/v12（Outcome-Calibrated Program--Policy Binding）**：从v10冻结checkpoint初始化，
  以task-equal fit19 success/BDDL progress分别校准`event x owner` privileged Program binding与`layer x family`
  compiler binding；每个macro仍同时使用完整multi-state functional/support锚，不读取held/validation/Test reward；
- [x] 完成OCPB v11 retained实现、两坐标真实profile、formal macro1、train24物化与同一308-panel冻结审计；macro1使
  fit/held candidate-to-shared从v10的`.96892/1.00285`小幅改善到`.96786/1.00171`，但breadth仍为`12/19、2/5`，
  因而不进入`q_V`，只按已登记的交替坐标继续macro2；
- [x] 完成v11 macro2、物化与冻结审计，并定位shared-rank credit的16倍尺度违反；该checkpoint只作工程诊断，不作
  compiler-binding科学反证，旧macro3/4取消；
- [x] 从v11 macro1完整恢复model/optimizer/RNG/topology，以相同paired seeds运行唯一v12 corrected compiler-binding，
  随后立即物化与复跑同一geometry/308-panel门；
- [x] 对v10、v11 macro1与v12 corrected相邻single checkpoints物化全部train24并复跑同一308-panel frozen support；v12
  geometry与support早门失败，按逐级合同没有启动held5 oracle；
- [x] 实现OCPB v13的baseline-relative functional support barrier：只惩罚candidate相对source/shared已有expert-response
  baseline的退化，不再把所有偏离source/shared的响应都当作损失；替换v12 active config/schema且保留唯一运行面；
- [x] 保持Program、compiler、rank、数据、paired seeds与outcome coordinate不变，从同一v11 macro1 authority完成一次matched
  corrected compiler-binding profile/formal复验，并立即物化与复跑同一geometry/308-panel门；v13首次同时达到fit/held
  support breadth门，但task-relative geometry不变且held aggregate仍略差于shared，故关闭同曲线；
- [x] 实现OCPB v14 owner-resolved policy-response distillation：v2 support panel在同状态保存source与successful members的
  `2 × 38 × 4 × 128`冻结response，candidate同一次forward可微捕获owner response并直接监督family-specific factor heads；
  v13 barrier、最终flow、Program、rank与outcome coordinate不变，raw A/B仍为零梯度诊断；
- [x] 构建一次v2 frozen support bank，完成真实单task profile并确认owner loss、可微compiler路径、显存与信息墙；
- [x] 从v13冻结checkpoint运行一个task-equal v14 outcome macro，立即物化并复跑同一geometry/308-panel门；一更新
  checkpoint未改善geometry或support，不能进入held5或`q_V`；
- [x] 将v14 outcome-only入口收敛回唯一Stage 1 trainer，实现v15 task-balanced owner-response bootstrap；删除两套退役
  active configs、四个outcome orchestration模块与旧入口，保留一个canonical训练/物化运行面；
- [x] 将专家要求的multi-state owner-resolved policy-response distillation从昂贵rollout macro中拆成task-balanced优化阶段：
  从v13 model weights与fresh optimizer完成114-visits/19-updates信息节点，保持v13 barrier与全部Program/compiler结构；
- [x] 114节点先物化geometry；task-relative方向出现实质移动后exact-resume到228 visits并完成冻结support audit。最终
  pair cosine降至`.98727`，但own retrieval仍`1/24`，fit/held support退到`.97253/1.02171x`，因此联合门失败且不运行
  outcome-calibration、held5或`q_V`；
- [x] 实现 **OCPB v16 owner-local activation-effect distillation**：在真实successful/learner occupancy的同一冻结source target
  input上，直接匹配gauge-invariant的`B(Ax_ref)`局部功能增量，保留完整owner/horizon结构；不再把可被所有上游targets共同改变的下游hidden
  当作“对应owner”目标，不读取raw A/B；
- [x] 从v13 authority完成同一114/228 task-balanced裁决：target-local cosine retrieval由v13 `1/24`提高到`11/24`，但
  candidate correction仍比expert correction显著更task-common，308-panel fit/held相对shared退到`1.08581/1.08815x`与
  `2/19、0/5` breadth；因此v16关闭，不接outcome、held5或`q_V`；
- [x] 实现 **OCPB v17 action-grounded composed-policy recovery**：复用v16已形成的task-discrimination model weights并创建fresh
  optimizer，对successful或verified-success跨episode action panel直接计算冻结PI0.5的exact flow-matching loss及LoRA leaf gradient，
  再反传到`q_pi`/compiler；failed learner action不作oracle，v16 local effect与v13 barrier只作结构/support锚；
- [x] 用真实successful-panel单task profile确认exact action loss、LoRA-leaf到FactorHead梯度、显存和吞吐；
- [x] 运行114-visits bounded task-balanced节点并完成24-task geometry与同一308-panel frozen support audit；输出虽明显分散，
  own-policy方向和shared support均失败，因此不续到228、不进入held5；
- [x] 实现 **OCPB v18 action-guided structured outcome binding**：以每task exact action leaf gradient定义38-owner/family局部
  proposal direction，再由fit19 paired simulator success/progress给这些方向分配task-equal closed-loop credit；训练只保留一个
  `train_ecp_stage1.py`入口，已删除v17 visit-step和旧selector/program offset运行面；
- [x] 完成单task paired rollout profile并确认38/38 owner proposal、严格paired simulator、非零reward advantage、
  LoRA-leaf到FactorHead反传、显存和吞吐全部接通；首个`.01x` profile还发现closed-loop leaf被旧outcome尺度任意压低，
  formal前改用已由antithetic estimator完整归一化的`1.0x`自然梯度尺度；matched profile已确认outcome leaf按预期放大100倍，
  FactorHead/total gradient保持finite且显存不变；
- [x] 完成4个task-equal outcome macros及macro2/macro4两次24-task geometry与308-panel audit；credit在每轮覆盖`8--10/19`
  tasks，但own retrieval始终`1/24`，macro4 fit/held relative-shared退到`1.14317/1.13244x`、breadth仍`2/19、0/5`，
  因此v18关闭，不进入held5或`q_V`；
- [x] 实现 **OCPB v19 fixed-compiler structured Program binding**：回到support最强的v13完整坐标，永久冻结compiler，
  exact action gradient只在compiler可达的event × layer-group × target-family Program切空间定义proposal；每次paired panel只裁决
  一个预注册family block，reward只更新privileged Program inference，不再让scalar outcome旋转FactorHeads；active tree已替换为
  v19单路径并通过47项聚焦CPU合同；
- [x] 完成v19单task reachability profile：`.05` q-family Program扰动产生`.110166` compiled relative delta，paired credit
  非零，`policy_teacher`梯度`.560583`且compiler/visible Program梯度严格为零；
- [x] 从clean pushed frozen authority完成fit19 macro2、24-task geometry与同一308-panel support audit；q/v credit均真实存在，
  但v19 single checkpoint的absolute geometry/support门失败，且仍处于历史v13的坍缩区间，因此不exact-resume到macro4，
  v19关闭；二者materialization videos不同，不把微小差值解释为matched训练变化；
- [x] 实现 **OCPB v20 Program-Locked Compiler Identification（PLCI）**：从v13 weights/fresh optimizer固定visible Program与
  privileged `q_pi`坐标，只训练compiler；用successful cross-episode exact action、owner-local/multi-state response和v13
  barrier做task-balanced dense识别，避免Program/decoder共同旋转；旧v19 config、perturbation和outcome runtime已删除，
  active tree仅保留一个Stage 1 trainer，55项Stage 0/1、reward与expert-manifold聚焦CPU合同通过；
- [x] 完成v20真实单task profile：action/FactorHead/compiler梯度finite非零，`q_pi`/visible梯度精确为0，
  峰值显存约16.39GB；
- [x] 运行114 visits/19 updates并完成matched 24-task geometry与308-panel support；candidate更分散但own retrieval仍
  `1/24`，fit/held support退到shared的`1.00874/1.02932x`，v20关闭；
- [x] 实现 **OCPB v21 Fixed-Compiler Free-Program Reachability（FPR）**：固定v20 compiler，仅为fit19优化task-local
  privileged free Programs，以同一exact action、multi-state response和support区分compiler image不可达与shared `q_pi`
  inference失败；free Programs永不进入deployment或held route；旧v20 config已删除，56项聚焦CPU合同通过；
- [x] 完成v21真实单task profile：free-Program梯度`.052106`且一步process相对修正`.092908`，compiler/`q_pi`/visible
  梯度严格为0，update `1.26s`、峰值约16.40GB；
- [x] 从clean pushed frozen authority完成v21 228 visits/38 updates、24-task materialization与308-panel support audit；free
  Programs已形成明显task-specific corrections，但compiled LoRA仍own retrieval `1/24`，fit/held support为shared的
  `1.02708/1.02932x`、breadth`7/19、2/5`，因此bounded rank-mode retraction image判失败并关闭；
- [x] 实现 **OCPB v22 Direct-Absolute Compiler Free-Program Reachability（DA-FPR）**：复用v6 macro228已经训练完成的
  family-specific direct A/B absolute compiler与其`q_pi`，保持v21 task-local free-Program oracle、fit19数据、task-equal
  schedule和geometry/support门不变；checkpoint 153/153 keys strict-load、55项聚焦合同及真实单卡profile已通过，不重复训练
  一套长compiler；
- [x] 从clean pushed frozen authority完成v22 228 visits/38 updates、24-task geometry与308-panel support gate；direct surface
  将candidate pair cosine降到`.70280`并把own retrieval提高到`10/24`，但fit/held相对shared为`1.45056/1.27628x`且
  breadth`0/19、0/5`，故v22关闭；
- [x] 实现 **OCPB v23 Single-Surface Absolute Compiler（SSAC）**：删除prior exact-template hard bypass，让prior/full都经同一
  direct family A/B heads；冻结visible Program与`q_pi`，只训练compiler并同时锚定prior shared support和full own-policy功能；
- [x] 完成v23真实单task双路径profile：两条功能路径、action leaf与compiler/FactorHead梯度均finite非零，冻结模块梯度为0；
  根据首轮真实梯度量级一次性校准prior权重后，update约1.43秒、峰值约22.80GB；
- [x] 从clean pushed frozen authority完成v23 fit19 task-balanced 114-visit节点及matched full/prior geometry/support；两条surface
  均未保留stable shared support，v23关闭且不进入held5 oracle；
- [x] 实现 **OCPB v24 Layer-Resolved Single-Surface Compiler（LR-SSC）**：static/process content独立local read并在head前连续
  非线性融合；用38个target-local A/B heads保留layer correspondence；从v23迁移shared权重，并以fit19 prior hidden对stable
  shared做minimum-change head calibration，不建立template bypass、task table或第二adapter；
- [x] 完成v24真实单task双路径profile：prior/shared起点、process fusion、Action LoRA leaf、target heads和完整compiler梯度均有效，
  冻结模块梯度为0，单步约1.44秒、峰值约22.92GB；
- [x] 从clean pushed authority完成v24 114-visit节点、materialization及一次执行同时得到full/prior的308-panel audit；
  candidate pair cosine`.97092`、own retrieval`1/24`，full/prior support两臂均失败，故不进入held5；
- [x] 完成Stage 1 v1--v24系统复盘，区分约13组主要因果问题与11个窄变体/修复节点，并冻结“不得直接建立v25”的纠偏规则；
- [ ] Stage 1联合geometry/support门通过后轮换固定fold，确认不是单一held5偶然结果。

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

1. v24已经完成并关闭：formal、materialization、dual 308-panel audit与remote-safe evidence全部落盘；不续训、不扫
   LR/rank/seed/dtype/width/fusion，不运行held5或`q_V`；
2. Stage 1 v1--v24复盘已经完成。复盘确认主要问题是geometry/open-loop support门长期
   未用held closed-loop校准、fixed compiler与专家direct layer/family-local surface落实过晚、19个独立task mappings不足以及
   版本号增长快于独立信息增长；
3. [x] 完成non-held meta资产审计：71个step1000 expert checkpoints与3550条successful demos均可复用；与fit19合成90个
   task-equal mappings，held5继续留出；不重训experts、不把trajectory计为mapping；
4. [x] 登记唯一 **MDCO** falsification contract：主要变量仅为mapping diversity；固定ECP结构与信息墙；90 tasks各6次dense
   visits并完成一次task-equal structured fit success/progress calibration后，直接运行held5 strict paired250；只允许一次
   预注册near-pass exact-resume，失败即关闭当前compiler family；
5. raw/effective LoRA own cosine与retrieval降为定位指标，不再作为成功策略必要门；新的open-loop support proxy在连续主导
   决策前必须用一次train-authorized held5 paired closed-loop校准；
6. [x] 已按唯一授权只一般化canonical Stage 1 data/support owner；不增加平行trainer或新架构版本；
7. [x] canonical Stage 1代码已一般化为namespaced 95-task/118-member authority、90-task动态world-size等权schedule、每member
   多trajectory聚合、fit90 `q_pi + compiler`联合训练与held5单checkpoint物化；71-task fixed occupancy selection已登记，v24专用
   support-audit运行面已退役；
8. [x] 从clean pushed detached worktree完成142条fixed non-held occupancy复现；141条保持成功，task32/state22一条未复现
   成功并从successful-policy evidence中剔除，不整批重跑。71个mapping仍全部覆盖，其中70个各2条成功trajectory、1个1条；
9. [ ] 用已保留的141条成功trajectory构建combined 90-fit/5-held action-response authority与95-task support bank；
10. [ ] 在正式训练前实现并验证540节点后的唯一一次90-task等权structured success/progress calibration，不得以配置声明代替
   真实simulator credit；
11. [ ] 从clean pushed detached worktree运行MDCO首节点和held5 strict paired250。只有完整门通过才轮换fold并进入`q_V`；失败按
    合同停止；
12. 完成本轮合同/evidence集成、推送`main`；只保留canonical formal产物，不删除或重复生成昂贵资产。
