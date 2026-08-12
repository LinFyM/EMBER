# EMBER Repository Instructions

## Authority and current truth

本文件、`docs/active_session_handoff.md`和`docs/execution_brief.md`共同定义当前authority。历史设计、旧日志、
formal artifact与Git快照中的“当前/下一步/active”都只表示当时状态，不能自行恢复执行。

2026-08-12当前真相：

- 长期目标未完成：同一shared method、同一single checkpoint的strict paired correct必须严格超过
  `150/400`，并继续提高absolute、task breadth、稳定积累和teacher-video时序因果性。
- 历史最好single checkpoint仍是v6-fast macro400：
  `correct/same/wrong/shuffled/reversed=143/135/125/128/129`。
- 最新uniform pivot-rank14路线已经终局non-pass并退役：online=`128/400`；old-cache compiler-only=
  `138/400`、breadth7，但相对old134 retained/gained/lost=`119/19/15`，违反lost`<=10`。
- compression和online regeneration均造成独立能力换手；Gate C、cycle1、controls和新训练未授权。
- 该结果不等于视频、Reward、continuous tangent、task experts或所有rank-reserved topology整体无效。
- PICK的raw-frame门通过；world6 discarded full48 profile只因regularized Gram condition=`483.61515>200`
  non-pass，其余14项机制、动作与吞吐门通过，未获formal训练资格。
- PICK-GC已完成formal fresh`0→10`和strict paired400：`138/400`、breadth6，相对immutable macro0
  retained/gained/lost=`118/20/16`、churn36，未过`correct>=144`与`lost<=8`门。其resume、controls和sweep
  全部关闭；只淘汰PICK-GC+blind offline source-action credit组合。
- OSG-PC已完成唯一一次world6 discarded profile并因工程hard gate退役：rank5在`profile_max_seconds`等待
  600s后NCCL watchdog，至少一个rank未到达；从run-contract发布到timeout已`969.9709s`，是matched
  `507.3054s`的至少`1.912x>1.25x`。无mechanism report/checkpoint，formal、deployment与评测均未授权；
  只淘汰当前full-replay per-success VJP执行图，不否定所有success constraint。
- SKNC已完成formal fresh`0→5`和strict paired400并正式non-pass：`137/400`、breadth7、per-task=
  `1/3/45/32/0/37/18/1`；相对immutable old134 retained/gained/lost=`121/16/13`、churn29。Program closure、
  projected rank/energy和outcome-only信息墙均健康，但未过`correct>=140`、`lost<=8`及单task增益集中门；
  `5→10` resume、controls和参数sweep全部关闭。它只淘汰PICK-GC+first-all-success-key nullspace+blind B20。
- SRTP已在profile工程门终局non-pass并退役：`d172add`首个world3 macro因decoder graph跨K4保留而三rank在mixed
  reward CFM处OOM；`e31e2fd`释放原graph、只在Nmc4后重解compiler的唯一同合同reprofile仍三rank OOM，申请
  `254/484/484 MiB`时分别只余`19/16.31/417.06 MiB`。两次均无mechanism report/checkpoint，deployment/formal
  未授权。第二次把最早工程失效接口定位到完整logical B<=16 landmark policy-gradient本体，而非decoder graph；
  不得再降batch、改dtype、开allocator选项或补第三次修复。该结果淘汰当前SRTP执行合同，不否定constant-memory
  landmarks、shared reward half-space或所有on-policy credit。
- PCUG已在唯一discarded live macro的工程门终局non-pass：clean pushed`238cab4`、gpu02物理3/4/5/6 world4，
  从run-contract发布到owner按不可逆wall失败停止的下界为`809.72185s`，是scaled matched SKNC baseline的
  `2.25568x>1.5x`。rank3/物理6先等待，物理3--5持续100%，run尚未完成Phase A full24 gather，也未开始paired
  probe；无OOM、nonfinite、mechanism report或checkpoint。formal/deployment关闭，不重跑同配置。该结果只淘汰
  当前static world4 PCUG execution contract，不检验actual candidate paired-guard的科学假设。
- Work-Queue PCUG已完成clean world3完整机制profile并退役：Phase A=`44.74125s`、total=`558.05862s / 1.16596x`
  matched SKNC，48 exact pairs有7 discordance、3 gains/4 losses、3 harmful tasks跨2 suites；correct guard closure、
  rank33和energy`.76492`健康。但blind negative ratio从`.03991`被correct-only final projection放大到`.50179`，
  wrong/shuffled/reversed均`0/8`达门。19项checks只有`negative_null`失败，deployment/formal关闭，不重跑或sweep。
- Negative-Preserving Candidate Guard已完成formal fresh`0→5`和macro5 strict paired400并退役：`135/400`、
  breadth5、相对old134 retained/gained/lost=`117/18/17`。五宏negative/correct closure、rank和energy健康，
  但point key到held same-task videos仍有平均`.40954`正交残差；不resume、不补controls、不做point/threshold/
  seed sweep。
- Cross-Video Equivariant Candidate Guard是当前唯一active successor authority：primary one-shot schedule、
  ordered negative、work queue、paired candidate、NPCG affine correction和native rank16 compiler全部保留；
  每task每macro只增加一条训练期action-hidden ordered companion，以`E=phi(companion)-phi(primary)`约束每个
  shared Program write满足`E D=0`。部署仍严格一条video、一套LoRA；尚未实现或运行。

canonical仓库是`/data1/user/ymdai/projects/EMBER`，唯一主写分支是`codex/bci-continuation`。正式训练或
评测必须来自该分支clean pushed commit的detached frozen worktree。

## Mandatory reading

修改代码、配置、数据、split、模型或实验状态，或启动任何GPU工作前，主进程必须完整读到EOF：

1. `README.md`
2. `docs/active_session_handoff.md`
3. `docs/execution_brief.md`
4. `docs/action_forecast_writer_cross_video_equivariant_candidate_guard_design.md`
5. `docs/action_forecast_writer_negative_preserving_candidate_guard_design.md`
6. `docs/action_forecast_writer_work_queue_candidate_guard_design.md`
7. `docs/action_forecast_writer_paired_candidate_update_guard_design.md`
8. `docs/action_forecast_writer_shared_reward_tangent_projection_design.md`
9. `docs/action_forecast_writer_success_key_nullspace_consolidation_design.md`
10. `docs/action_forecast_writer_on_policy_success_guarded_program_credit_design.md`
11. `docs/action_forecast_writer_policy_innovation_goal_causal_key_design.md`
12. `docs/research_history.md`
13. `task_plan.md`
14. `findings.md`
15. `docs/concept.md`

涉及历史架构细节时先查`docs/research_history.md`；只有确需精确旧公式/命令时再从Git commit`3a6f801`
读取对应旧design、旧`findings.md`或旧`progress.md`。涉及迁移/路径恢复时再读
`docs/a100_to_bci_migration_handoff.md`。不要把读完数万行历史追加日志当成推进前置流程。

## Objective and scientific decision rule

EMBER从generic`lerobot/pi05_base`出发，在与目标40去重的LIBERO-90 source tasks上训练并冻结共享
π0.5-LIBERO source policy；随后研究能否把目标task的语言和action-hidden教学视频一次性编译为完整LoRA，
使frozen policy在未见初始化上闭环完成任务。

最终checkpoint同时要求：

- strict paired correct严格`>150/400`且继续提高；
- 多task breadth、相邻checkpoint低换手、可重复共同积累；
- correct实质优于wrong、shuffled、reversed和no-video；
- same-task-other video鲁棒；
- 视频语义/时序经representation、compiler、effective LoRA传到policy action；
- LoRA能量、秩和跨target结构足以驱动policy，但几何健康度本身不是目标。

每次结果必须与最接近历史架构、143起点、逐task成功集合、gained/lost和视频controls比较。先定位最早失效
接口，只改变一个有因果指向的结构或objective变量。负结果只淘汰实际检验的假设；没有证据不得180度转向，
也不得因某结构试过一次就遗忘其中仍成立的子机制。

训练loss、held functional loss、reconstruction、smoke、LoRA norm/rank/cosine、hidden差异、small panel、
checkpoint union或多checkpoint融合都不能选择或宣告方法。最终只认真实、严格配对的closed-loop
single-checkpoint结果。

## Data and split

- benchmark是LIBERO Spatial/Object/Goal/Long共40 tasks。
- development split封存在`configs/libero_24_8_8_v1/`：每suite 6 train / 2 validation / 2 test，总计24/8/8；
  不得按outcome改变task IDs。
- LIBERO-90 specification-only audit排除19个与目标40 exact semantic/composition重合的tasks，保留71；
  source base使用每task全部50条成功teacher episodes。
- 不得使用读过目标40 actions的`pi05_libero`。
- normalization只从过滤后的source actions/states计算并冻结；validation/test不得重算。
- validation选择方法后才合并32 source / 8 test并从规定初态重训。

## Information wall and deployment

canonical benchmark目前保持one-shot：

- Writer输入只能是exact task language加恰好一条action-hidden teacher video。
- video是唯一dynamic value；language可作query/context/address，但不能单独生成LoRA或形成bypass。
- Writer不得读取teacher action、proprio、reward、terminal、task ID、filename、object pose、hidden
  normalization或其它元数据。
- 每episode一条video生成一套完整38-target public rank-16 LoRA；不做video/LoRA/checkpoint平均或融合，
  不挑最好video。
- frame stride固定5；frozen source policy无trainable parameters；no-video/step0必须functional identity。
- task experts和历史feature cache只能作train24监督或机制分析，不能成为held部署输入、nearest-expert route或
  第二套LoRA。

few-shot是允许研究的未来变量，但不能悄悄改变one-shot基线：必须另写authority，固定`k`或定义可变数量的
集合聚合，保持action-hidden、禁止video挑选，并与相同计算/评测口径的one-shot对照。K4历史已经证明
few-shot可减少部分偶然性，但不自动解决共享credit、正确时序或task drift。

## Training contract for any successor

- development只用24 train tasks产生梯度；validation/test actions或reward不得进入训练。
- video与action query同task但跨episode独立采样，不能用原始逐帧对应制造低层捷径。
- task aggregation必须等权或由新authority明确定义；不能由易task、长度或GPU分配隐式加权。
- frozen source policy、normalization、split、public LoRA topology和official policy interface默认固定。
- checkpoint必须含model、optimizer/stateful estimator、sampler/cursor、每rank RNG、world-size/topology和schema；
  exact-resume不得伪装fresh或改变拓扑。
- 一个短机制/profile门后尽快训练到足以判断趋势，并及时做paired400；不让弱surrogate拖延真实性能测试。
- 只在absolute、breadth、趋势和内部传递共同支持时续下一段；不得因训练“还能跑”自动续训。

## Evaluation

- official preprocessing保持render256/model224、两相机180° rotate、state/action 7维、10 flow steps、执行前5
  actions后replan、dummy settling10、成功即终止、suite horizons 220/280/300/520。
- zero-interaction每rollout从正确task的50条teacher videos无放回取一条；不挑video。
- correct/same-task-other/cross-suite-wrong/shuffled/reversed/no-video严格配对task/state、env/policy RNG、
  video ordinal和处理；shuffled/reversed对真实输入frames重排后完整forward。
- evaluator用cost-balanced dynamic queue、long-first和persistent model/env；不静态task/GPU分配，不dummy占卡。
- 报告aggregate、per-task、per-suite、breadth、retained/gained/lost、churn、union/intersection仅作诊断，以及
  representation→compiler→effective BA→fixed-action传递。
- 80-row screen只可作工程诊断，不能选择checkpoint；正式比较以400 paired rows为准。

## Throughput and numerical policy

科学精度指信息墙、paired identity、真实closed-loop与状态完整，不指底层逐元素复现。

- 原生BF16/TF32、batch shape、kernel和reduction order的正常低位差异可接受。
- 不因`.001953125`级roundoff固定batch1、重复single forward、扩dtype、关闭高效kernel或逐tensor扫描。
- profile从合理大batch开始，扩大到吞吐平台或显存风险；选择samples/s最高且能稳定覆盖最长video的候选，
  尽可能利用显存。
- 不新增SHA-256、MD5或大量内容hash/防御性校验；已有manifest hash只作历史provenance，不在热路径重算。
- 门禁shape、finite、信息墙、明显串样、OOM、错误asset、pairing和resume语义。

## GPU and host

- 每次GPU launch前必须同时live检查`gpu01`与`gpu02`，区分空闲、忙碌和故障卡。
- 选择一个节点，使用至多6张健康、低利用率、显存余量足够且能提高有效吞吐的A40；非零显存或低利用率
  进程不自动排除，但不得抢占或明显干扰他人。不等待凑满6卡、不dummy occupancy、不跨节点拼碎片。
- 不reset、kill、pause、抢占或干扰他人进程。设备边界、ownership和telemetry都是live state。
- 独立evaluator无NCCL；多卡训练固定`NCCL_P2P_DISABLE=1`，遵守NUMA physical/local rank映射和deferred-NCCL。
- exact-resume锁定原world size；fresh world size按live可用卡与task-complete合同在`1--6`内决定，不能硬编码6。

## Storage and artifacts

- 大资产留在`/data0/user/ymdai`或`/data1/user/ymdai`，优先复用canonical roots、symlink和manifest。
- 大copy/cache/training前查询`strg01`上对应filesystem的独立user quota，测当前使用并估计峰值；`df -h`不是
  quota检查。不得把`/data0`与`/data1`预算相加。
- 正式result保留run contract、checkpoint manifest、metrics、raw rows、aggregate、completion和必要analysis；
  screen/profile/smoke不得冒充formal。
- 正式datasets、checkpoints、400-row roots和唯一机制证据默认保留；只删除生命周期明确的失败/临时/重复输出。
- 临时文件放`.codex/tmp/`并在任务结束时删除；不把废弃内容移入in-tree archive。

## Engineering, Git and cleanup

- 一个canonical active implementation。替换行为时旧实现由Git、frozen config、formal artifacts和
  `docs/research_history.md`保存，不保留可执行平行版本或兼容stub。
- 先检查现有owner/caller/CLI/config再新增模块、fallback或入口；不要为猜测的未来需求抽象。
- 主工作树保持clean、diff聚焦；正式run使用frozen worktree；多个write-capable agent不得写同一worktree。
- subagent可用于边界清晰的独立审计/实现/验证；主进程负责统一科研判断、集成和最终验证。
- meaningful状态只更新当前authority文档，不再向数十个历史design和逐日ledger重复追加。
- 不提交dataset、cache、checkpoint、大binary、secret或host-private配置。
- 删除前核验target、进程、dirty state和唯一provenance；已合并clean worktree/branch与明确临时文件可清理，
  ambiguous或未合并唯一工作必须保留并报告。

## Current retained assets

- source policy：`runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000`。
- task experts：`runs/outputs/pi05_task_expert_bank_formal_step1000_r6_81101fe_20260807`内统一step2000。
- historical best Writer：v6-fast macro400，精确root见`docs/active_session_handoff.md`。
- latest rank14 decision root：
  `runs/outputs/pi05_v6_qv_rank_reserved_compiler_only_old134_to_rank14_correct400_20260811`。
- PICK/PICK-GC train-only design-selection cache：
  `runs/outputs/pi05_expert_manifold_feature_cache_train24x50_r6_222d3ac_20260808`；它不是新方法成绩。
- PICK-GC formal训练与strict400：
  `runs/outputs/pi05_pick_gc_goal_causal_formal_fresh0to10_r4_b20_c2e1ff8_20260811`和
  `runs/outputs/pi05_pick_gc_goal_causal_correct400_noreplacement_seed7_macro0010_retry1_398425e_20260811`。
- 最新retired config：`configs/pi05_v6_on_policy_success_guarded_program_credit_v1.json`；状态为
  `profile_result_sealed_nonpass`，不能profile、formal或resume。
- 最新retired config：`configs/pi05_v6_success_key_nullspace_consolidation_v1.json`；状态为
  `formal_result_sealed`。formal训练root与macro5 strict400 root分别为
  `runs/outputs/pi05_sknc_success_key_nullspace_formal_fresh0to5_r3_b20_e3863cb_20260812`和
  `runs/outputs/pi05_sknc_success_key_nullspace_correct400_noreplacement_seed7_macro0005_e3863cb_20260812`；
  decision evidence、paired transition、raw rows和checkpoint均保留，但不能resume或补controls。
- 最新retired design authority：`docs/action_forecast_writer_shared_reward_tangent_projection_design.md`；config
  `configs/pi05_v6_shared_reward_tangent_projection_v1.json`状态为`profile_result_sealed_nonpass`。`d172add`与
  `e31e2fd`两个OOM roots/logs/failure artifacts保留，均无checkpoint/成绩；deployment、formal和同配置重跑关闭。
- 最新retired authority：`docs/action_forecast_writer_paired_candidate_update_guard_design.md`；canonical config
  `configs/pi05_v6_paired_candidate_update_guard_v1.json`状态为`profile_result_sealed_nonpass`。world4 failure root、
  run contract、launch contract、log和exit保留，无paired evidence、checkpoint或成绩；同配置重跑、deployment和
  formal关闭。
- split：`configs/libero_24_8_8_v1/`；数据、tokenizer、simulation assets由CLI/`.env.local`提供。

旧方法的可执行入口被清理不代表实验记录丢失。需要恢复某个已证明有效的子机制时，先从
`docs/research_history.md`确定证据边界，再从对应Git commit选择性移植，而不是恢复整条退役路线。
