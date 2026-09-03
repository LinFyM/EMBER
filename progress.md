# EMBER progress

更新时间：2026-09-03。

## 当前快照

- canonical集成目标为clean pushed `main`；当前最新clean pushed代码tip为
  `66826f86495f31e77a3b866b526e8d8fcf421de5`。最新专家补充意见已逐字归档；归档没有修改科学代码或实验配置。
- owner于2026-09-02完成最后审查，正式确认Policy-Response Event-to-Factor Writer并要求立即推进。系统goal已重新建立并保持
  active，不设置token或阶段工期预算。
- 当前唯一active design为`docs/policy_response_event_to_factor_writer_design.md`。它保留PI0.5
  layer x horizon x probe原生响应、G2 ordered events、当前视频真实X/Y、G1 signed pooling、rank4和唯一rank16物化；主要learned
  模块收敛为可复制的Video Process Encoder与Native Factor Composer。
- 最后审查已把四项修正写入合同：causal process auxiliary严格prefix-only；预测target冻结；task1/task93 Composer正控先确认
  G1 bank容量；full/coarse对照只裁决前端表示。
- PNBTT E1及其single/family chart、两次spectrum、full-rank16和gate-aligned necessity均已完成并稳定`non_pass`。PNBTT、
  EBSRI、Program-through-bank和旧summary/gate/anchor均不是active fallback，历史证据与formal artifacts继续保留。
- 已从clean pushed `main@194b91b2ae34efcb042a6c838973ba5d57ceda55`建立唯一
  `codex/policy-response-writer`实现worktree。Frozen Capture、repeatable Frame/Event blocks、Current-Video Native Factor Composer、
  task-local functional runtime、formal checkpoint/resume与唯一rank16 materialization已经接通；旧PNBTT/J2 runtime只复用数据、
  functional与checkpoint基础设施，不是active fallback。
- 新Composer沿用G1已证明的非对称启动：native A方向在初始化存在，`tanh(scale_head)`令有效B与mobile update严格为零；真实功能梯度
  第一步只打开scale，第二步再进入Frame/Event/Composer，避免随机新Writer在训练前破坏carrier。相同projected native keys在context
  read与signed pooling间复用，task-local阶段缓存冻结Process输出，不改变任何科学变量。
- 相关定向测试为`23 passed`；完整`tests/ecp`排除既有失败后为`141 passed`，唯一未通过项
  `test_shared_compiler_mapping.py::test_mapping_credit_is_set_valued_family_balanced_and_scale_stopped`在未修改的canonical main上同样失败，
  是退役SharedCompiler测试与当前`TangentTransportResult`字段不一致的既有问题，不由本实现引入。
- task1真实smoke保存在disposable
  `.codex/tmp/policy_response_smoke_zero_scale_task1_20260902/`：51帧、19 layer x 50 horizon x 2 probe与38-target X/Y一次捕获，
  初始A非零/B严格零；打开scale后functional梯度到达Frame `.002874`、Event `.003822`、Composer `.237721`，prefix-only frozen-target
  process梯度到达Frame/Event/predictor，76 tensors、rank12+4到唯一rank16全部通过，峰值allocated约`30.59GB`。
- task1/task93两步task-local profile只作工程证据。step1均只有scale-head梯度，step2 input/output branches与task query均有非零梯度；
  task93三个correct Panel-B视频两步后均自发略优于carrier。去重后task93 step耗时由`34.80/34.15s`降至`28.47/26.24s`，
  allocated峰值由`39.63GB`降至`32.89GB`、reserved由`47.30GB`降至`40.22GB`。该实现随后已提交、合并并推送，并从clean
  detached authority并行运行task1/task93的110步formal正控及step70/110只读Panel-B。
- 上述两条正式正控现已完整结束。task1 step70/110的Panel-B fit recovery为`.260876/.276421`、held-video为
  `.207341/.244598`；task93 fit为`.337207/.346604`、held-video为`.300885/.280724`。两task、两checkpoint的全部fit/held视频均
  优于carrier，确认新Composer不是零容量接口；task93 held后段轻微回落，不把单点峰值当结论。
- shared K1运行面已在同一Writer与唯一CLI内接通：每步固定3 meta + 3 target、task/role等权；每task固定一个CPU evidence-cache owner；
  functional使用正确视频跨episode Panel-A exact LoRA-leaf VJP，process使用strict prefix-only frozen target，preservation只作同一正样本
  单侧hinge；task-held、same-task held与Panel-B均零梯度。单卡两步profile和双卡单步profile均通过，后者覆盖一个rank有task、一个rank
  显式零梯度仍完成deferred NCCL/all-reduce/gather。结构审查已从单个851行shared driver拆为orchestration、optimizer、只读评估与
  authority四个owner，当前无hard architecture violation。
- 专家规定的process Huber loss与无量纲权重`1.0`已在最终代码上完成额外真实单卡一步profile：task1冻结process normalizer为
  `.0558371`，首步normalized process loss为`1.17047`；Frame/Event/Predictor/Composer梯度范数分别为
  `1.23987/.991191/2.50437/3.34321`，峰值allocated/reserved仍为`26.64/35.52GB`。该profile只验证最终目标图与资源，不作方法选择。
- shared实现及最终Huber目标图已经完成回归并提交为`0c5c7e99`；launch合同随clean pushed
  `main@1290673a0b51158c0a4f1fc02ff0f32a729996e0`冻结。12-task K1 component-init full/coarse两条matched formal及四个held5
  correct-only strict250均已完整结束、零错误。full step70/110的Panel-B gradient fit/held benefit为
  `.000542/.000199`与`.001032/.000586`，true-task-held两点均为负；对应closed-loop为`33/31`。coarse为
  `.000684/.000596`与`.000828/.000715`，true-task-held仍均为负；closed-loop为`43/41`。四个checkpoint的Goal/Long均为0。
  coarse逐task按Long/Goal/Object/Spatial0/Spatial9为`0/0/2/40/1`与`0/0/4/34/3`，相对carrier分别保留`37/43`与
  `35/43`；full只有`29/43`与`25/43`。因此full没有证明复杂19层response前端的增量，coarse更稳定但仍只达到carrier，四臂均
  为科学non-pass。
- 等待训练期间已在唯一Writer CLI内接通`materialize` phase：冻结shared checkpoint后，用held5每task固定correct demo5各调用Writer
  一次，物化五套独立且完整的38-target rank16 adapters，并交给既有`evaluate_pi05.py` static task-LoRA运行面做correct-only
  strict250。配置固定为`configs/pi05_ecp_policy_response_writer_held5_eval_v1.json`；deployment runtime不实例化functional action/state
  dataset或processor，不读取held action/reward，不生成wrong、shuffle、reverse、no-video或language-only条件，也不复制Evaluator。
  clean pushed `main@e7631247`上的单卡真实runtime smoke确认authority IDs `71/76/81/86/91`精确对应global
  `0/9/18/25/36`，language tokens同集合、Panel数为0、query dataset/processor均为`None`；冻结资源峰前常驻约
  `9.38GB allocated / 19.23GB reserved`。该路径随后已物化四个checkpoint并完成上述1000条正式rollout；没有为失败checkpoint运行
  wrong、shuffle、reverse、no-video或language-only controls。
- 首个有依据修正不改模型、functional/process目标、rank或scale：选择较简单的coarse与K1，把梯度映射从5 meta + 5 target扩大为
  55个经审计meta与18个target-fit，task2/74仍作true-task-held，另外15个meta held和5个target held不产生梯度。每task只缓存两条
  outcome-independent fit正确视频，最后一条正确视频作held；6卡时按role与cache owner平衡，绝大多数update同时使用6个真实任务。
  配置为`configs/pi05_ecp_policy_response_writer_scale_v1.json`，10 warmup + 1200 effective updates，保存macro610/1210；macro610
  一出现即可在训练继续时并行做held5 correct250，不新增人为内部门。单卡两步真实profile已解析两套completed Panel roots中的75个
  task authority，完成真实capture、functional反传、process反传和唯一rank16物化；step2 Frame/Event/Predictor/Composer梯度均非零，
  峰值allocated/reserved为`27.55/38.17GB`。profile当时为资源上界读取4 fit + 1 held视频；正式首轮已缩为2 fit，profile不作方法选择。
- 73-task scale component-init formal launch contract：scientific implementation为`1a11115bd719aa00ec4a80ac61ee09a200944443`，
  formal从包含本条合同的最新clean pushed detached `main`运行。唯一科学变量是gradient mapping数量；固定coarse、K1、component-init、
  同一Writer/model/rank/seed/LR、correct-only cross-episode functional、positive prefix-only process与preservation。55 meta与18 target按role
  各占每update一半，task在role内等权；task2/74及所有其它held任务零梯度。10 warmup + 1200 effective updates，保存macro610/1210；
  macro610出现后训练继续，另行物化并做held5 correct-only strict250。00:05第一次尝试在gpu02物理4/5/6装载时，另一用户在物理5的
  新任务从`1.48GB`增长到`24.4GB`并开始计算；EMBER在生成run contract、output root、checkpoint或任何科学step前主动停止，未触碰
  对方进程，也没有产生可恢复或可解释的科学状态。修正后输出root固定为
  `runs/outputs/pi05_ecp_policy_response_writer_scale_73task_k1_component_coarse_s1210_1a11115b_gpu01p2456_20260903/`且launch前不存在。
  2026-09-03 00:23 CST同时live检查两节点：gpu01物理2/4/5/6分别为`15/15/98/15 MiB`、util全`0%`且无compute process；gpu02
  物理5已有`26520MiB/74%`动态任务，其余可共驻卡不比gpu01四张空卡更合适，故不跨节点拼卡。四卡真实75-task schedule owner各持
  `18/19/19/19` tasks、采样frame cost为`1818/1850/1815/1807`；只依据静态task/cache owner自动选择两role固定phase后，完整1210步
  每rank最多2个task，`905/1210`步四卡都有真实task，其余305步为三卡各2个task，meta task恰为66次、target为201或202次，不读取
  outcome且不改变科学权重。gpu01 available host memory为`247500165120 bytes`；按75-task真实video长度共5285个采样帧、已测
  `22164019 bytes/frame`估计训练加Panel-B冻结cache峰值`109.1GiB`，四个model process仍有充分余量。`/data1` quota blocks为
  `773575340/1073741824`、limit `1084227584 KiB`；旧shared run约`82MB`，本run保守新增小于`2GB`且不复制dataset/model。
  固定`NCCL_P2P_DISABLE=1`、world-size4与每rank GPU-local NUMA绑定；exact resume只允许同commit、同物理卡、world size、config、输入与
  output root。exact process command为：
  `cd /data1/user/ymdai/projects/EMBER-worktrees/policy-response-writer-scale-formal-1a11115b && NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=2,4,5,6 PYTHONPATH=src /data1/user/ymdai/projects/EMBER/.venv/bin/torchrun --standalone --nproc_per_node=4 scripts/train_ecp_policy_response_writer.py --config configs/pi05_ecp_policy_response_writer_scale_v1.json --asset-root /data1/user/ymdai/projects/EMBER --data-root /data1/user/ymdai/projects/EMBER/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_ecp_policy_response_writer_scale_73task_k1_component_coarse_s1210_1a11115b_gpu01p2456_20260903 --phase shared --representation coarse --initialization component --mode formal`。
- 上述scale formal已于2026-09-03 00:27 CST从clean pushed detached `df7a7f5a`启动，tmux session为
  `ember_prw_scale73_coarse`。run contract精确锁定同一commit、world-size4、物理2/4/5/6、73个gradient tasks、K1及positive-only
  信息墙；四rank冻结cache完成后首个optimizer step在00:34结束。step10完成时最近5步平均`28.92s`，Frame/Event/Predictor/Composer
  梯度均finite nonzero，最大CUDA reserved为`40.883GiB`，gpu01 available host memory仍为`122612895744 bytes`。当前训练继续；
  macro610 ETA只作运行调度估计，不作科学证据。已完成旧12-task formal的clean detached worktree已删除，commit与全部formal outputs保留。
- scale macro610 checkpoint已完整保存并在训练继续时完成held5 correct-only物化与strict250。物化root为
  `runs/outputs/pi05_ecp_policy_response_writer_scale73_coarse_m610_held5_correct_k1_materialized_df7a7f5a_gpu02p4_20260903/`；
  五个held task各只调用Writer一次，held action/reward/state、validation/test、wrong、shuffle和reverse读取均为0，输出均为唯一完整
  38-target rank16。旧evaluator把合法Writer checkpoint硬编码为`{70,110}`；`6ddceff5`将其收敛为正数macro且目录名必须精确匹配，
  相关测试`5 passed`并对真实macro610 bank完成reinspection。该修复不改变model、adapter或评测科学合同。
- macro610 strict250 root为
  `runs/outputs/pi05_ecp_policy_response_writer_scale73_coarse_m610_held5_correct_k1_strict250_6ddceff5_gpu02p46_r2_20260903/`；
  250个严格配对状态、20个shards和四worker均完整，return code全0。结果为`26/250`，逐task Long/Goal/Object/Spatial0/Spatial9为
  `0/0/1/25/0`，breadth `2/5`。相对carrier43保留`22`、获得`4`、丢失`21`，paired exact p约`.00091`；相对coarse
  macro70同样为`22/4/21`，因此是显著净退化。materialized mobile4整体函数范数已从旧macro70/110约为carrier12的
  `.49--.66`倍增至macro610的`1.81--2.15`倍；38 targets普遍放大而非单target爆炸。训练functional benefit仍持续上升，故当前最早
  警报是train functional proxy与held closed-loop脱节并伴随shared residual外推过强，而不是Writer没动或评测故障。
- 随后的48-state固定observation/noise只读功能诊断直接重算carrier、G1与macro610的PI0.5 owner/flow/action response，raw evidence在
  `runs/analysis/pi05_ecp_policy_response_writer_scale73_m610_g1_effect_alignment_5df9406_gpu02p46_20260903/`。五task
  successful-member effect loss均值为carrier/G1/Writer=`.914596/.238841/1.023186`，Writer只在`1/5` task略优于carrier、平均反而
  恶化`.108590`。四个G1非零mobile task上，Writer相对G1的member-scale-whitened功能方向cosine仅
  `.05044--.30470`、中位`.14753`，功能响应norm却只有G1的`.4563--.7444`。因此参数残差虽过大，真正policy effect并非“正确方向
  过强”，而是主要落在低效或错误功能方向；事后缩放只会趋近carrier，不能补出G1效应。该诊断零梯度、使用封存G1 privileged
  effect reference，只作non-pass定位，不参与checkpoint选择。
- 对照专家§7.5与§9.4继续审计后确认，首版只实现逐rank bounded gain，却遗漏完整per-target effective-update RMS cap。macro610
  held5有`94/190`个task-target完整mobile `B@A`超过fit19 task-equal全局`s_ref`、最大`2.2433 x`；fit-only shared template为
  `0/38`，正式G1 held5只有`5/190`个轻微超过。固定压回`1 x s_ref`的post-hoc strict250已经完整结束：`33/250`、breadth
  `1/5`，Long/Goal/Object/Spatial0/Spatial9=`0/0/0/33/0`。相对原macro610 retained/gained/lost=`20/13/6`、paired exact
  `p=.16707`；它局部恢复Spatial0净8条，却丢掉Object唯一成功，证明scale boundary有保护作用但不能补出跨suite正确方向。root为
  `runs/analysis/pi05_ecp_policy_response_writer_scale73_coarse_m610_sref_effective_cap_b8ad986_gpu02p46_20260903/`，manifest明确登记
  `training_gradient_use=false`与`checkpoint_selection_use=false`，四worker、250 rows及return code均完整。
- 原训练前804步的global clip触发率为`.8781`；scale-head/其余方向norm中位为`2.5992/.5839`，同一norm分组后方向侧预计只有
  `.0386`的step触发、有效方向倍率中位恢复`2.6533 x`。唯一实现worktree现为
  `codex/policy-response-writer-scale-boundary`：Composer按完整rank4 Gram矩阵计算RMS并在不物化dense `B@A`时统一缩放B，shared
  optimizer把`scale_head`与其余Writer参数各自按原norm `1.0`裁剪；active scale config已显式记录两项。focused tests为
  `9 passed`，包含dense等价、cap内外、零初始化finite gradient、完整Composer integration和独立裁剪预算。task1/task93两步真实
  shared profile均已`exit 0`完成：step耗时分别为`15.73/13.81s`与`30.14/25.96s`，Frame/Event/Process/Composer梯度均finite且
  nonzero；峰值allocated/reserved分别为`23.47/34.81GB`与`33.30/41.56GB`。source policy、native observer与task-local参数均
  trainable=`0`，wrong/held/Panel-B backward及shuffle/reverse reads均为`0`，输出仍为唯一完整rank16；profile只验证真实图、边界与资源，
  两步内部functional数值不作科学选择。gpu02物理4/6随后释放，总EMBER物理卡回到旧训练使用的4张。
- 该单点不运行controls、不进入mixed-K、fully-random或validation。旧73-task训练继续到预注册macro1210，以相邻checkpoint判断其
  实际未限幅/global-clip parameterization是过渡还是稳定退化；它不再承担停止整个函数类的资格。修正版验证后及时merge/push，
  并从clean detached authority fresh运行同一73-task/coarse/K1/component-init matched实验。等待期间只删除了一个无
  invocation、无shard且已被完整retry1取代的80KiB旧full-macro110 prepare root；formal evidence未删除。
- fresh启动前对照专家动态必要性合同发现，首版Event未使用真实`frame_positions`，却把可学习slot
  position直接混入value，并用slot-specific logits选择relation value。完全重复的静态8帧因此仍产生
  event/frame innovation RMS `.19244/.13996`，scale打开后4个构造target全部打满`.20` cap，是可复现的
  架构合同违反。当前唯一`codex/policy-response-writer-dynamic-necessity` worktree正把position限于emission/
  transition/QK路由、使relation value在slot间共享并做frame-common中心化聚合。CPU结构检查已降至
  `7.23e-8/6.17e-8`，合成mobile RMS最大`4.50e-5`；focused tests当前`10 passed`。task1/task93两步真实shared
  profile均已`exit 0`：步耗时分别为`17.42/18.18s`与`40.28/30.95s`，Frame/Event/Process/Composer梯度均
  finite nonzero，峰值allocated/reserved分别为`23.47/34.81GB`与`33.30/41.56GB`。source policy、native observer与
  task-local参数全部冻结，wrong/held/Panel-B backward与shuffle/reverse reads均为0，输出仍是唯一完整rank16。
  旧macro1210因不含该修正，只作旧parameterization相邻证据，不能替代fresh corrected formal。
- Policy-Response Writer shared matched formal launch contract：scientific implementation为`0c5c7e99`，formal从包含本条合同的
  最新clean pushed detached `main`运行；两臂共用唯一配置`configs/pi05_ecp_policy_response_writer_v1.json`、固定source、Stage0、
  carrier12、s_ref、J2 Panel A/B、mapping split与数据`data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a`。gradient tasks固定为
  meta `1/8/9/32/52`及target `72/73/75/93/94`，task2/74只作true task-held零梯度评估；每step固定3 meta + 3 target，K1、
  两条fit正确视频交替训练，第三条same-task视频只评估。两臂均为component-init、single-process/single-A40、10 warmup + 100
  effective updates、正确视频cross-episode Panel-A 16 rows、Huber positive process权重`1.0`、单侧preservation `.05`，保存
  step70/110并在12 tasks三条视频上各做16次零梯度Panel-B。两臂唯一差异是Process读取full 19-layer x 50-horizon x 2-probe
  response或coarse final-layer horizon mean；Composer与完整dynamic X/Y bank、参数量、seed、schedule和world size完全相同。
  full固定gpu01:2，输出`runs/outputs/pi05_ecp_policy_response_writer_shared_12task_k1_component_full_s110_0c5c7e99_gpu01p2_20260902/`；
  coarse固定gpu02:5，输出`runs/outputs/pi05_ecp_policy_response_writer_shared_12task_k1_component_coarse_s110_0c5c7e99_gpu02p5_20260902/`。
  合同记录时live状态分别为`15MiB/0%`与`159MiB/0%`，后者只有gqma `148MiB`低占用进程；launch前必须再同时检查两节点，若
  状态漂移则不盲目占用。`/data1` quota blocks为`773293620/1073741824`、limit `1084227584 KiB`，两个root均不存在；参考正式run
  每条约`45MB`，即使保守估计小于`1GB`也远低于余量，冻结evidence cache仅驻CPU内存。exact resume只允许同commit、同node/GPU、
  config、输入及world size；无效root不覆盖。两条exact commands为：
  `cd /data1/user/ymdai/projects/EMBER-worktrees/policy-response-writer-formal-0c5c7e99 && NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src /data1/user/ymdai/projects/EMBER/.venv/bin/python scripts/train_ecp_policy_response_writer.py --config configs/pi05_ecp_policy_response_writer_v1.json --asset-root /data1/user/ymdai/projects/EMBER --data-root /data1/user/ymdai/projects/EMBER/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_ecp_policy_response_writer_shared_12task_k1_component_full_s110_0c5c7e99_gpu01p2_20260902 --phase shared --representation full --initialization component --mode formal`；
  `cd /data1/user/ymdai/projects/EMBER-worktrees/policy-response-writer-formal-0c5c7e99 && NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=5 PYTHONPATH=src /data1/user/ymdai/projects/EMBER/.venv/bin/python scripts/train_ecp_policy_response_writer.py --config configs/pi05_ecp_policy_response_writer_v1.json --asset-root /data1/user/ymdai/projects/EMBER --data-root /data1/user/ymdai/projects/EMBER/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_ecp_policy_response_writer_shared_12task_k1_component_coarse_s110_0c5c7e99_gpu02p5_20260902 --phase shared --representation coarse --initialization component --mode formal`。
- Policy-Response Writer task-local formal launch contract：scientific implementation为clean pushed `66df1974`，formal从包含本条合同的
  最新clean pushed detached `main`运行；唯一配置`configs/pi05_ecp_policy_response_writer_v1.json`，复用其中固定source checkpoint、
  Stage0/native observer、carrier12、s_ref、J2 Panel A/B、mapping fit/held split与数据
  `data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a`。task1使用fit videos 5/6、held 24，task93使用2/3、held 46；
  每task独立single-process A40、10 warmup+100 effective updates、Panel-A correct-only 16 rows/step、checkpoints 70/110，每checkpoint
  对三条correct视频各做16次只读Panel-B。task1命令固定`CUDA_VISIBLE_DEVICES=5`在gpu02，输出
  `runs/outputs/pi05_ecp_policy_response_writer_tasklocal_task1_full_s110_66df1974_gpu02p5_20260902/`；task93固定
  `CUDA_VISIBLE_DEVICES=2`在gpu01，输出
  `runs/outputs/pi05_ecp_policy_response_writer_tasklocal_task93_full_s110_66df1974_gpu01p2_20260902/`；两者均设置
  `NCCL_P2P_DISABLE=1 PYTHONPATH=src`并使用canonical `.venv/bin/python scripts/train_ecp_policy_response_writer.py --phase task-local
  --representation full --mode formal`及相同asset/data roots。launch前live状态为gpu01:2 `15MiB/0%`，gpu02:5 `159MiB/0%`且只有
  gqma `148MiB`低占用进程；不会触碰他人进程。`/data1` user blocks为`772868852/1084227584 KiB`，两个run含四枚checkpoint
  保守峰值小于`2GB`，3.04/5.67GB frozen evidence与小型Process cache仅驻内存；两个目标root均不存在。裁决比较step70/110
  fit/held correct functional recovery与既有同task free-primal正控，不使用wrong、held或Panel-B梯度。只允许同commit、同节点/物理卡、
  config、输入与single-process拓扑exact resume；无效或superseded root不覆盖，另名保留。
  两条exact process commands分别为：
  `cd /data1/user/ymdai/projects/EMBER-worktrees/policy-response-writer-formal-66df1974 && NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=5 PYTHONPATH=src /data1/user/ymdai/projects/EMBER/.venv/bin/python scripts/train_ecp_policy_response_writer.py --config configs/pi05_ecp_policy_response_writer_v1.json --asset-root /data1/user/ymdai/projects/EMBER --data-root /data1/user/ymdai/projects/EMBER/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_ecp_policy_response_writer_tasklocal_task1_full_s110_66df1974_gpu02p5_20260902 --phase task-local --task 1 --representation full --mode formal`；
  `cd /data1/user/ymdai/projects/EMBER-worktrees/policy-response-writer-formal-66df1974 && NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src /data1/user/ymdai/projects/EMBER/.venv/bin/python scripts/train_ecp_policy_response_writer.py --config configs/pi05_ecp_policy_response_writer_v1.json --asset-root /data1/user/ymdai/projects/EMBER --data-root /data1/user/ymdai/projects/EMBER/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_ecp_policy_response_writer_tasklocal_task93_full_s110_66df1974_gpu01p2_20260902 --phase task-local --task 93 --representation full --mode formal`。

## 最新科学结论

### 仍成立的正证据

- frozen source validation8为`48/400`，validation8 task-local rank16 oracle为`250/400`。
- held5 source/carrier/independent successful members为`21/43/113`；mobile-rank4解析容量覆盖held5五个task。
- G1 action-in native-block free-code strict250为`114/250`，breadth5/5、Goal2、Long1，正式通过。
- G2 boundary-anchored Natural Program的held full相对endpoints改善`22.2047%`，probe`38/40`、median active events`4`，
  same-task/K1/K4均通过。
- P0/P1、R5等正控证明真实native bank、current-bank operator和task-local功能方向具有容量；它们不证明shared mapping。

### 已裁决G3/PNBTT最早接口

- Program-through-bank topology-matched free-summary S0双task正式通过：task1 correct/held约`.974--.989`、wrong约`-.565`；
  task93 correct/held约`.917--.947`、wrong约`-.342--.394`。
- fresh real Program-through-bank S1正式non-pass：task1 correct fit0/fit1/held为`.826825/.855228/.797545`，task93为
  `.776511/.792673/.719798`；wrong、margin、all-pairs、信息墙、Action Meta 0和唯一rank16均通过。按预注册条件没有运行shared S2。
- §7.1 bank-conditioned-primal恢复correct，但wrong specificity不足：原双tasktask1 wrong为`.428/.477`，task93为`.627/.654`。
- calibrated Q_free把wrong从`.815/.832`降到`.526/.534`，同时把correct降到`.808/.826/.795`，确认capacity--specificity权衡。
- base-LR A_free虽然233个anchors全部更新，但RMS仅`.0094`、约为candidate的`3.7%`，因此只淘汰under-travel版本。
- 最终calibrated A_free把free-anchor RMS提高到`.17664`，已与candidate anchor`.188--.192`同量级。task93 correct
  fit0/fit1/held为`.853296/.858892/.818467`，wrong为`.611592/.668511`；all-pairs通过，wrong和margin正式non-pass。
- 同checkpoint精确F=0后correct升至`.879708/.883433/.849663`、wrong升至`.750229/.756445`。F确实更强抑制wrong，
  但也伤害correct；candidate delta的correct/wrong cosine约`.718--.772`，占主导的free delta约`.993--.995`。
- 最早缺口因此是高相似summary经family-scalar gate调制共享event-additive anchor时只能近同向移动correct/wrong，无法把bank内容差异
  放大为所需功能分离。停止边界只覆盖这一具体parameterization。
- PNBTT的single-key chart、首次tangent spectrum、family-key chart、v2 spectrum和唯一full-rank16 oracle已按专家§5.10顺序
  全部执行。full-rank16 macro70上task1 correct fit0/fit1/held为`.953328/.933839/.941449`、wrong为
  `.648060/.719726`；task93 correct为`.557237/.561168/.411465`、wrong为`-.001312/-.007719`。macro110为
  task1 `.960297/.941644/.948351` vs `.634156/.711548`，task93 `.586174/.595686/.449605` vs
  `-.006466/-.021862`。两checkpoint均为`non_pass`，任务依赖反转稳定。
- 正证据是task16 transport在task1可恢复高correct，在task93可产生强specificity；gate-aligned rank4也能同时把两个wrong arm压到很低。
  负证据是这些性质不能在两个任务上同时恢复E1要求的absolute correct/held；full-rank16没有相对rank4呈现一致、广泛、明显更优，
  gate-aligned spectrum也没有扩容触发。因此不做中间rank或LR/seed/width/scale/chart小扫。
- 该已裁决路线的最早缺口是PNBTT E1 free-query real-bank transport函数类的absolute correct capacity。它不裁决冻结的Natural
  Program、G2、native X/Y、signed pooling、rank4或整个ECP。

完整历史及每个旧架构的结果在`docs/research_history.md`；长期跨轮结论在`findings.md`；全部专家原文均位于`docs/`。

## 已裁决的PNBTT实现

- PNBTT保留G2 Natural Program、真实38-target X/Y及四类output bank、frame quadrature、exact signed replay、small-core
  canonicalization与首版carrier12+residual4。
- Program只产生低维query；当前bank的真实candidate产生key并继续作为唯一native value。B0只做可微key-space whitening，B1在同一bank
  上执行一次联合measure的antithetic signed transport；没有base primal、bounded correction、family scalar gate或free anchor。
- 首个E1 single-key-chart与family-key v2均稳定non-pass。v2 spectrum相对首版的q correct-preserve-wrong中位仅从input
  `.555`到`.566`，四类output约为`.174/.235/.220/.224`；v input从`.463`到`.476`，abs改善到`.643`，但adj/init/goal反而从
  `.808/.727/.734`降到`.769/.685/.693`。尾端10%谱能量仍近零；family chart主要把action-out adj/goal operator cosine从
  `.839/.748`降到`.712/.627`，与formal wrong改善一致，却未补足correct容量。因此不增加`m`或继续改chart。
- 唯一full-rank16 oracle只比较rank分配端点：保持相同family-key PNBTT、free query、数据、loss、Gate和110步cadence，将
  `carrier12+task4`改为`carrier0+task16`；最终仍是单一38-target rank16，不形成rank28或第二adapter。task16冻结幅度先验由与`s_ref`
  一致的fit19、非held task-local rank16 Action Experts做exact small-core singular component RMS后task-equal median得到；不读取
  validation/test，也没有task/video lookup。该oracle已稳定non-pass，未触发rank重开条件。
- 最后裁决config为`configs/pi05_ecp_pnbtt_e1_gate_aligned_necessity_v1.json`；它从family-key v2机械派生，唯一科学改动是将
  `normalized_necessity_margin` 从`.10`对齐到formal Gate `.50`。该轮已稳定`non_pass`，所以没有重算E2所需G2 Program tensors。

## 最新formal evidence

- PNBTT E1 free-query transport：
  `runs/outputs/pi05_ecp_pnbtt_e1_free_query_s110_2664e0d_gpu01p12_20260902/`；110步、macro70/110 Panel-B与
  `evaluations/qualification.json`均完成，最终为相邻一致`non_pass`。
- PNBTT E1 tangent spectrum：
  `runs/analysis/pi05_ecp_pnbtt_e1_tangent_spectrum_m128_step110_8306a4c_gpu01p12_20260902/`；task1/93共380个
  target-side spectra、16个Panel-A visits、三条gradient arms，`completion.json`完整，耗时`376.97s`。
- PNBTT family-key E1：
  `runs/outputs/pi05_ecp_pnbtt_e1_family_key_s110_02633a39_gpu01p12_20260902/`；训练authority固定为clean detached
  `75db5f84`，gpu01物理1/2双rank；110步、macro70/110五臂各16次Panel-B、两个checkpoint与
  `evaluations/qualification.json`完整，最终为相邻一致`non_pass`。
- PNBTT family-key tangent spectrum：
  `runs/analysis/pi05_ecp_pnbtt_e1_family_key_tangent_spectrum_m128_step110_75db5f84_gpu01p12_20260902/`；同一v2 macro110、
  task1/93各16个Panel-A visits、共380个target-side spectra，held/Panel-B/validation/test均未使用，`completion.json`完整，
  耗时`381.48s`。
- PNBTT full-rank16 oracle：
  `runs/outputs/pi05_ecp_pnbtt_e1_fullrank16_oracle_s110_57969a68_gpu01p12_20260902/`；训练authority为clean detached
  `1897b8dceecf93d1b3063b6f42a78f286cb699b2`，110步、macro70/110 checkpoints、两次五臂各16次Panel-B、raw metrics、
  run contracts、completion与`evaluations/qualification.json`均完整；总体与逐task均相邻一致`non_pass`。
- PNBTT gate-aligned necessity E1：
  `runs/outputs/pi05_ecp_pnbtt_e1_gate_aligned_necessity_s110_e65c6388_gpu01p12_20260902/`；训练authority为clean detached
  `2050de9e7583955fa0c62eaeb375eb5b3847500a`，110步、macro70/110 checkpoints、两次五臂各16次Panel-B、raw metrics、
  run contracts、completion、训练/评测logs与`evaluations/qualification.json`均完整；总体与逐task均相邻一致`non_pass`。
- PNBTT gate-aligned tangent spectrum：
  `runs/analysis/pi05_ecp_pnbtt_e1_gate_aligned_tangent_spectrum_m128_step110_2050de9e_gpu01p12_20260902/`；同一step110、task1/93
  各16个Panel-A visits、共380个target-side spectra，held/Panel-B/validation/test均未使用，`completion.json`、`result.json`、shards、
  run contract与analysis log完整，耗时`382.57s`。

- Program-through-bank S0：
  `runs/outputs/pi05_ecp_program_through_bank_bottleneck_s0_gate_s110_b11dc3e_gpu01p23_20260901/`
- Program-through-bank S1：
  `runs/outputs/pi05_ecp_program_through_bank_bottleneck_s1_gate_s110_9047230_gpu01p23_20260901/`
- bank-conditioned-primal双task：
  `runs/outputs/pi05_ecp_bank_conditioned_primal_gate_s110_eb9f295_gpu01p12_20260901/`
- calibrated Q_free：
  `runs/outputs/pi05_ecp_bank_conditioned_primal_qfree_calibrated_task93_s110_fdc669f_gpu01p0_20260901/`
- base-LR A_free：
  `runs/outputs/pi05_ecp_bank_conditioned_primal_afree_task93_s110_b0d81bb_gpu01p0_20260901/`
- calibrated A_free：
  `runs/outputs/pi05_ecp_bank_conditioned_primal_afree_calibrated_task93_s110_e02f4ca_gpu02p4_20260901/`
- A_free逐层与F=0因果审计：
  `runs/analysis/pi05_ecp_bank_conditioned_primal_afree_causal_audit_144d59b_gpu02p46_20260901/`

以上formal evidence、唯一checkpoints、raw rows、aggregate、run contracts与completion均保留；没有因交接清理删除。

## 仓库与workspace整理

- 交接前58个累积worktree已清理；首个E1、family-key E1、三次spectrum、full-rank16及gate-aligned formal结束后对应detached evidence worktree均已删除。
  PNBTT首轮结论完整快进至远程`main`后，已合并的`codex/pnbtt`实现worktree与本地/远程分支均已清理。目标函数错配审计后，已从最新clean pushed
  `main@9afca0bb`建立的`codex/pnbtt-gate-aligned-necessity`也已合并并清理；最后formal固定在detached `2050de9e`完成后，该worktree同样删除。
  训练、评测与分析日志均已移入各自formal root，`.codex/tmp`已为空；当前只剩canonical worktree且没有local `codex/*` branch。
- full-rank16 formal启动后已删除被提交记录取代的两步disposable profile及两个非运行worktree的Python/pytest cache。为保持一个
  canonical Writer运行面，删除4个只暴露已退役EBSRI/J3/routing-control路线的旧runner：`train_ecp_bank_set_shared.py`、
  `train_ecp_bank_set_tasklocal.py`、`evaluate_ecp_bank_set_shared.py`、`evaluate_ecp_joint_program_primal.py`；历史modules、configs、tests和
  formal artifacts仍保留审计，PNBTT的已结束训练与checkpoint评测均由唯一`train_ecp_joint_program_primal.py`运行面执行。
- 删除8个local `codex/*` branch：已合并分支由`main`保存；两个未合并EBSRI S2草案因S1预注册non-pass而失去执行资格；历史
  `g3-vector-interaction@2295f48`仍由`origin/codex/g3-vector-interaction`保存。
- 已删除完整并入`main`的远程`codex/g3-bank-set-relative-interaction`与`codex/g3-v4-evaluator-authority`；未合并的
  `origin/codex/g3-vector-interaction@2295f48`明确保留。
- 两个旧dirty worktree分别是已被clean S0/S1链和后续G3历史取代的实现草案；确认无运行进程、无formal authority引用后随worktree清理，
  未提交内容不可恢复。
- `.codex/tmp`中约`5.1GB`旧smoke/profile/script/cross-language临时cache已删除；其中影响决策的结论均已进入`findings.md`或
  `docs/research_history.md`。后续profile只作可删除工程证据，不与formal roots混存。
- 未删除或移动dataset、models、formal runs、checkpoints、raw rows、aggregate、source policy、task experts、condition caches或
  ownership不清资产。
- tracked旧科学代码、测试和历史configs在新实现复用/退休审计前保留；`main`是canonical source，旧结果或仍存在的config不得自行
  恢复为路线。

## 已裁决PNBTT执行细节

- 第八次专家原文、代码/config/authority冲突和formal evidence已完成逐项复核；未发现推翻主路线判断的结果错误。
- PNBTT canonical compiler已接通：Natural Program只供query，real bank同时供key/value；包含可微batched key whitening、joint-K等video
  质量、exact chunked antithetic signed replay、四类output scope、38-target rank4 materialization及唯一carrier12+residual4 rank16。
- E1 task-local free-query训练与Panel-B evaluator已接通；policy、Program、carrier/scale、native values与Action Meta均冻结，correct fit0/fit1和
  wrong fit0产生梯度，held/wrong fit1/Panel-B为零梯度。task8/94只提供unrelated Panel-A states，preservation用同一keyed flow
  time/noise比较generated与carrier真实action velocity；wrong-video仍有单侧carrier上界。run contract从真实policy/Program模块审计
  Action Meta，而非声明式写零。
- E0 synthetic hard checks通过：zero native value给出zero residual；candidate/video排列与chunked replay误差仅为FP32低位；K2 video质量各
  `.5`；bank swap改变方向；forward/gradient finite；真实policy消费唯一38-target rank16。
- 首个真实双卡profile因一次性保留38-target covariance/Cholesky autograd图在A40约44GiB OOM；按target即时链式回传后不改变梯度
  （synthetic leaf-gradient最大误差`0`）。接入真实`D_policy`后的最新两步profile在task1/task93 microbatch 8/4下稳定完成，分别为
  `25.000/24.665s`；rank0/1峰值allocated为`39.773/36.154GB`、reserved为`46.376/44.109GB`。step1 free-query梯度非零且shared key按
  非对称LoRA零初始化预期为0；step2 shared-key梯度为`.293542`，task1/93 paired policy distance为`.003844/.002297`，correct/wrong已分离。
- 上述profile只验证工程图与吞吐，不参与E1科学Gate。E1 macro70/110均完成五臂各16次Panel-B；两枚checkpoint的task gate均为
  `non_pass`，总体与逐task结论一致。step110相对step70的correct/held改善仅`.013--.037`；near-bound最大值从未超过`.022005`，
  因此首个E1失败不是softmax饱和、训练过短或Natural Program；后续已按专家指定完成`T=Cov(v,k)`功能梯度投影谱。
- `T=Cov(v,k)`诊断已自然完成：380个谱均来自train-side Panel-A，operator列数固定1024；除结构性零bank外，99%谱能量rank远低于
  1024且末端10%能量通常不超过`1e-6`量级，因此不增加`m`。q/v的功能梯度保留与correct/wrong operator重合暴露的是chart
  表达问题。family-shared nonlinear trunk + target-specific rank16 low-rank projection已经接入；`m=128`、rank4、query、loss、
  数据与Gate未改，35项PNBTT/shared-compiler/joint-primal focused tests通过。
- v2 implementation `02633a3964ecfd9d40f9827ba98456c87c07552b`已在clean pushed main完成双A40两步真实profile。step2
  family-key aggregate gradient为`.155687`，task1/93 free-query gradient为`13.945/9.212`，correct/wrong已分离；单步
  `25.266s`，两rank峰值allocated为`39.789/37.260GB`、reserved为`46.272/44.082GB`，无OOM或non-finite。
- `0f052cccc9ddb96fbcaaa2a036fdc61ee190d945`在不改变当前K1 E1的前提下补齐E2前置硬合同：每条视频在每个有效
  event/scope先归一为等质量再按`1/K`混合，并缓存授权内容排序键以稳定集合归约；K2每半event mass精确为`.5`，相同Program
  context下的native内容换序测试通过。`a2c3fe9e`同时把canonical runner默认配置从退役J3收敛到当时PNBTT v2；两提交均已
  fast-forward并推送至`main`，family-key E1在运行期间仍固定在其祖先`75db5f84`。
- fresh E1 formal launch：从`02633a39`之后只增加本记录的clean pushed detached `75db5f84`运行；配置为
  `configs/pi05_ecp_pnbtt_e1_family_key_v2.json`，task1/93双rank DDP、110 optimizer steps、macro70/110 checkpoints，数据、
  Panel-A/B、loss与Gate完全复用首个E1。使用gpu01物理1/2，launch瞬间两卡均空闲，固定`NCCL_P2P_DISABLE=1`和NUMA0；输出
  `runs/outputs/pi05_ecp_pnbtt_e1_family_key_s110_02633a39_gpu01p12_20260902/`在launch时为fresh空目录。`/data1`当前user用量
  `772469868/1073741824 KiB`，参考上一E1的`257MB`，本轮含两个checkpoint峰值估计小于`1GB`。只允许同commit、同world-size2、
  同config exact resume；不覆盖无效root。科学裁决仍只认macro70/110五臂各16次Panel-B及相邻一致E1 Gate。
- family-key E1已经自然完成。macro70 task1 correct fit0/fit1/held为`.598648/.599961/.581859`、wrong为
  `.028320/.041884`；task93 correct为`.693744/.706930/.650097`、wrong为`.036270/.224452`。macro110 task1 correct为
  `.616630/.620958/.601512`、wrong为`.027332/.051458`；task93 correct为`.707775/.725727/.655429`、wrong为
  `.047247/.223365`。wrong、all-pairs与near-bound均通过，task1 margin也通过；两task correct/held和task93 margin稳定不足，
  70到110的correct/held改善只有`.0053--.0210`。因此family-key提高了specificity但没有恢复absolute capacity，不追加训练且不进入E2。
- v2 tangent spectrum也已自然完成：仍为380个train-side Panel-A spectra、每task 16 visits，耗时`381.48s`。相对首版，q/v input的
  correct-preserve-wrong中位只小幅变化为`.566/.476`；q四类output为`.174/.235/.220/.224`，v为
  `.643/.769/.685/.693`，没有形成correct容量所需的新可达方向。action-out adj/goal correct--wrong operator cosine降至
  `.712/.627`，解释了wrong specificity改善；但q/v input仍约`.958`，abs仍约`.927/.963`。全部非结构性operator的尾端谱能量仍远低于
  width上限，因此停止增加`m`或继续改key chart。该诊断本身不证明rank4 ceiling；只准入专家限定的一次同构full-rank16 oracle。
- full-rank16 oracle实现`57969a6895adfe2e336e5d83a30d1a80c12d47d2`保持一个参数化运行面：PNBTT residual rank由配置取4或16，
  rank4仍走原12+4拼接，唯一oracle直接物化task16；overcomplete action-out canonicalization以small-core SVD后零填充保持合法rank16
  shape。16项native/PNBTT与22项shared-compiler/functional focused tests通过。gpu01物理1/2两步真实profile自然完成，step1/2为
  `29.469/28.910s`；step2 task1/93 free-query梯度`9.883/11.488`、shared-key梯度`.205305`，全部finite。两rank峰值
  allocated为`39.841/38.584GB`、reserved为`45.722/44.080GB`，没有OOM；相对rank4约`25.3s`只增加约17%步时。
- full-rank16 formal已从clean detached `1897b8dceecf93d1b3063b6f42a78f286cb699b2`自然完成，root为
  `runs/outputs/pi05_ecp_pnbtt_e1_fullrank16_oracle_s110_57969a68_gpu01p12_20260902/`。配置固定
  `configs/pi05_ecp_pnbtt_e1_fullrank16_oracle_v1.json`，task1/93双rank、110步、macro70/110、五臂各16次Panel-B；除rank分配和对应
  fit19冻结task16 scale prior外，E1数据、三项loss、LR、seed与Gate均不变。两枚checkpoint、Writer、optimizer/trainer state、raw rows、
  run contracts、completion、evaluation logs和qualification均完整，所有launcher exit为0。
- formal launch preflight已同时检查两节点：gpu01物理1/2均仅`15MiB`、util `0%`，物理3/4也空闲；gpu02物理5空闲、4/6可共驻，
  0--3与7为他人高负载任务。训练选择gpu01物理1/2与NUMA0，因为两task一rank一卡已是有效拓扑且复用该节点23GB condition cache；
  不跨节点拼卡、不干扰他人。`/data1` user blocks为`772567180/1073741824 KiB`，参考上一formal仅`94684 KiB`且本轮两个更大
  Writer checkpoints仍预估小于`1GB`；目标root确认不存在。固定`NCCL_P2P_DISABLE=1`、world-size2、相同commit exact resume，
  macro70出现后可在仍空闲的物理3/4并行Panel-B以隐藏评测时间。
- full-rank16 macro70 task1 correct fit0/fit1/held为`.953328/.933839/.941449`、wrong为`.648060/.719726`；task93
  correct为`.557237/.561168/.411465`、wrong为`-.001312/-.007719`。macro110 task1 correct为
  `.960297/.941644/.948351`、wrong为`.634156/.711548`；task93 correct为`.586174/.595686/.449605`、wrong为
  `-.006466/-.021862`。两枚task gate均为`non_pass`，overall/per-task conclusion consistent与training complete均为true。该结果在
  task1通过correct/held却失败wrong/margin，在task93则通过wrong/margin却失败correct/held；两task的all-pairs和near-bound都通过。
  因此它是稳定科学non-pass，不是训练未完成、饱和、OOM或评测错误。
- `c992b3f0d1fc5954f55ad939368881aa7a78a52e`已删除430行仅绑定退役primal/gate/anchor拓扑的stale tests，保留active cache、
  set不变性、信息墙和member-effect合同；25项focused tests通过。该清理提交已fast-forward至`main`，不改变当时冻结的
  detached scientific authority，该formal运行现已完成。
- `50f876cb0e5e2e3623a4b77e768d67658960fccc`修正detached formal评测把会正常前进的`origin/main` tip误当训练身份的问题；
  现在仍锁定实际commit、clean/detached拓扑与全部科学合同，只允许包含该commit的authority tip前进。26项focused tests通过。
- 重新核对专家§6/7.2与三个formal metrics后，确认旧`.10` necessity hinge在formal `.50` margin失败时已关闭，先前的route/authority blocker裁决因此撤销。
  唯一`.50` config已由`e65c63888033639c58d29f285aed6cd8331c07e8`提交并推送。gpu01物理1/2双rank两步真实profile自然完成：step1/2
  `active_necessity_fraction`均为`1.0`，step2 task1/93 free-query梯度为`15.319/9.216`、shared-key梯度为`.171069`，correct/wrong已分离；
  单步`26.063/25.363s`，两rank峰值allocated约`39.79/37.26GB`、reserved约`46.27/44.08GB`，无OOM或non-finite。该profile只证明
  gate-aligned hinge真实接通，不作科学Gate证据。
- gate-aligned E1 formal launch contract：从包含`e65c6388`且clean pushed的detached `main`运行
  `scripts/train_ecp_joint_program_primal.py`，配置固定`configs/pi05_ecp_pnbtt_e1_gate_aligned_necessity_v1.json`，source checkpoint、
  tokenizer、dataset、G2/native authority与两个frozen cache均复用family-key E1；task1/93双rank DDP、110 optimizer steps、macro70/110，
  唯一科学变量是necessity margin `.10 -> .50`。环境固定`CUDA_VISIBLE_DEVICES=1,2`、`NCCL_P2P_DISABLE=1`、NUMA0与canonical venv；
  输出`runs/outputs/pi05_ecp_pnbtt_e1_gate_aligned_necessity_s110_e65c6388_gpu01p12_20260902/`在preflight时不存在。live检查时gpu01物理1--4
  均空闲，训练用1/2、保留3/4给checkpoint Panel-B并行；gpu02物理5空闲，4/6可安全共驻，其余为他人高负载任务。`/data1` user blocks为
  `772766460/1073741824 KiB`，参考同构rank4 formal峰值估计小于`1GB`。只允许同commit、world-size2、config和输入的exact resume；无效root
  不覆盖而另名保留。裁决只认macro70/110五臂各16次Panel-B与相邻一致E1 Gate。
- gate-aligned E1已从clean detached `2050de9e7583955fa0c62eaeb375eb5b3847500a`自然完成。110条raw training rows、macro70/110
  两枚checkpoint、五臂各16次Panel-B、contracts、completion与qualification完整。macro70 task1 correct fit0/fit1/held为
  `.585596/.592489/.541733`、wrong为`-.176695/-.153551`；task93 correct为`.707213/.715694/.676823`、wrong为
  `-.055836/.018941`。macro110 task1 correct为`.607645/.609189/.561628`、wrong为`-.171164/-.149315`；task93 correct为
  `.710657/.721565/.686395`、wrong为`-.086657/.006107`。两checkpoint每个task都通过wrong、margin、all-pairs与near-bound，
  只失败correct/held；总体与逐task均相邻一致`non_pass`。70到110的correct/held增益仅约`.003--.022`，不追加训练。
- `.50` necessity objective已真实行使：`active_necessity_fraction`在step1--10为`.95`、11--70为`.3083`、71--110为`.05`；
  末步task1/93 free-query梯度为`.1701/.1801`、shared-key梯度为`.04281`，preservation平均激活率`.9909`。因此结果不是旧`.10`
  hinge错配、梯度断开、anchor未移动、OOM或训练未完成。
- gate-aligned step110 train-only tangent spectrum随后在同一clean authority与gpu01物理1/2完成，380个target-side spectra、每task
  16个Panel-A visits、correct fit0/fit1与wrong fit0三条gradient arms均完整，held/Panel-B/validation/test读取为0。最大末端10%谱能量
  `1.3664e-5`，与旧v2的`1.3675e-5`等价；q/v input correct-preserve-wrong中位为`.5584/.4806`、operator cosine为
  `.9580/.9577`，action-out adj/goal operator cosine为`.7039/.6365`，均未形成新的correct可达方向。故不触发增大`m`、继续key chart
  或重开rank分配。
- 当时专家规定E1通过后才进入E2；次选B要求E1通过而真实frozen Program E2失败；whole-Writer joint也要求上游A路线Gate成立。
  这些条件均未发生，所以该PNBTT authority在当时没有后续active route。停止只覆盖已实际检验的PNBTT E1 transport函数类，不外推为
  Natural Program、G2、native X/Y、signed pooling、rank4、ECP或zero-interaction根本失败。
- 对专家原文§5.10、E1--E4及次选B做了第二次逐条件路线审计，并逐层复核当前query/key、Cholesky whitening、联合measure、antithetic
  real-value replay、固定type normalization、rank4物化与三项loss。没有发现工程合同偏离，也没有尚未执行且满足触发条件的专家分支：
  `m`未截谱、family chart已执行、full-rank16未明显优于rank4，E2/B/joint仍分别被E1前置条件阻断。
- workspace cleanup发现canonical joint runner仍默认指向已裁决的`.10` family-key config。该陈旧默认已删除，`--config`现在必须显式提供，
  从而保留sealed configs与复现实验能力但不会误把旧路线当作active；没有修改任何scientific config、模型或formal artifact。PNBTT与joint
  定向测试`30 passed`，CLI help及missing-config fail-fast均通过。
- `HANDOFF.md`已消费并删除；长期信息全部由authority、已裁决PNBTT design、本文件与Git保存。
