# EMBER Progress

更新时间：2026-08-22。本文只记录当前可执行状态；稳定目标见`docs/current_owner_requirements.md`，耐久结论见
`findings.md`，完整历史见`docs/research_history.md`。

## Current authority and executable state

- 专家最终复核与owner逐项讨论已经完成，新架构正式命名为 **EMBER-ECP（Event-Conditioned Policy Compiler）**；唯一
  active design为`docs/event_conditioned_policy_compiler_design.md`。核心结构是ordered video-event segmentation、
  Event-Conditioned Horizon Binding、target-family-aware Program、分布式privileged `q_pi(P)`、Dynamic-K `q_V(P|L,V)`与
  完整rank16 compiler。Action horizon保留为50个joint future positions，不再作为第二因果时间轴；旧hard dual-time
  transport与deterministic `P*`均已纠正。
- owner已建立EMBER-ECP正式goal并授权继续推进。Action Meta-LoRA改为Stage 0必做的独立共享校准对照：先建立native
  observer，若matched evidence无负面效果则采用并永久冻结。shuffled/reversed不进入训练、loss或checkpoint选择，只在最终
  候选checkpoint冻结后评测时序特异性。held5只作train24内部privileged/compiler leave-task-out机制门；最终shared components
  使用全部授权train data训练，validation8只以language+action-hidden videos作development evaluation。
- 训练顺序固定为：native observer与event binding → Action Meta-LoRA独立裁决 → privileged `q_pi + compiler` oracle gate →
  frozen-compiler Dynamic-K `q_V` → 除backbone/privileged teacher/已冻结observer calibration外的普通Writer参数联合训练 →
  通过视频必要性门后structured outer credit。Phase 0和Phase 1已完成；`native v3 macro10 + Action Meta v3 macro10`已经
  永久冻结。Phase 2首版已从clean pushed `6d71cb8`在detached frozen worktree完成全部1,140 visits及228/570/1140三个
  预注册节点；Gate 2明确失败并停止，未启动`q_V`。v2 absolute与v3 content/address separation也已各完成228-visits几何裁决；
  v4 coordinate bootstrap与v5 query-content bootstrap都已完成228-visits正式裁决；两者均未过预注册几何门。v6 policy-support、
  v7 prior union与v8 functional-only union也已完成同一短节点和冻结全bank审计。v8严格task-equal复验仍失败并已关闭；
  Frobenius top-SVD已经由bounded、exact-prior的policy-functional rank selector在唯一active路径中替换；当前工作是完成
  successful/learner真实双visit profile，并在通过后从clean pushed authority fresh训练balanced 228-visits。Program、`q_pi`、
  support bank、schedule和functional objective保持不变，不延长v8，也不围绕query、rank或loss权重做局部微调。
- Stage 1首个realizability warm-start的数据与坐标合同已冻结：47个train24成功策略成员提供完整rank16 direct adapter、
  8-phase successful-occupancy Action Expert response、reliability与member disagreement；23个task有2个独立成员、task39有1个。
  首版只读取已验证成功occupancy，不把此前闭环反向的learner-state residual重新引入。compiler以完整stable shared-prior
  rank16 LoRA为template，但全部16 ranks均可写；它用38 owner × 16 rank query、family-specific A/B heads和固定
  family/layer/owner locality bias一次生成全部76 tensors/1,287,168 values，不恢复旧16维decoder、shared12/task4硬拆或
  full-factor相加。
- Stage 1 canonical运行面已经接通为`visible Program projector → train-only q_pi → shared target-family compiler`。每条
  action-hidden video先经永久冻结的native+Action-Meta observer独立保序编码，K个video只在Program层做置换不变的
  presence-weighted mean/variance聚合；q_pi correction只能写入visible event presence已经激活的位置，privileged evidence
  中没有task ID。fit19的task-equal schedule为19×60=1,140 visits、world-size6的190次更新；held5保持零shared gradient。
  训练目标同时使用gauge-invariant exact-BA、member/consensus、stable-prior counterfactual、multi-phase完整PI0.5 functional
  response和locality，不把内部loss当作Gate 2闭环结论。
- 两个真实A40单卡profile均已通过。K2视频共45个stride-5 frames时，完整observer+q_pi+compiler forward/backward为
  2.24秒、峰值10,127,157,760 bytes；加入成功occupancy functional response后为2.15秒、峰值16,353,665,536 bytes，
  functional loss `.40906`且梯度finite。首次profile暴露observer未处于BF16 autocast的dtype接口错误，按已验证Stage 0
  调用边界修正后两臂均成功；三个task-owned临时profile目录已删除。228-visits正式段完成38 updates，checkpoint、optimizer、
  scheduler与6份rank RNG均完整，最大峰值10,411,287,040 bytes；该段按合同尚未启用functional response。
- 228-visits checkpoint已为全部train24任务物化一套完整rank16/76-tensor LoRA；held5仍是零shared gradient，物化输入为固定
  K2 correct action-hidden videos（没有最终LoRA平均）。同一held5 fixed250上，source/shared-prior/direct-earliest/
  direct-latest/generated为`21/43/74/108/23`。generated只在Spatial task0得到`23/50`，其余4 tasks均为0；source→generated
  为15 retained、8 gained、6 lost、净`+2`、McNemar `p=.79053`，而shared-prior→generated净`-20`。相对两套direct member的
  success retention为`13/74=.17568`与`12/108=.11111`，gain retention为`2/62=.03226`与`3/96=.03125`；Goal/Long均为0，
  因此该节点明确未过Gate 2。它只淘汰“纯几何warm-start在228 visits已经足够”的假设；当时按预注册schedule继续到首次包含
  functional policy-response监督的570-visits节点，不据此启动`q_V`或推翻ECP。
- 570节点完成后generated从23升到`27/250`，但仍低于shared prior 43，Goal/Long为0；相对source净`+6`但不显著，direct
  earliest/latest success retention只有`.16216/.12963`。继续到1140的理由只剩判断同一机制是否仍在移动，而不是把内部loss
  当成有效性。1140最终仍为`27/250`，逐task为Spatial0 `24`、Spatial9 `1`、Object8 `2`、Goal5 `0`、Long6 `0`；570→1140
  恰为`17 retained / 10 gained / 10 lost`，绝对零增长且Jaccard`.45946`。相对source为`12 retained / 15 gained / 9 lost`、
  净`+6`、McNemar `p=.30746`；相对shared prior净`-16`。direct earliest/latest success retention为`.16216/.13889`，gain
  retention为`.06452/.07292`，Gate 2所有核心机制门继续失败。
- 最早接口诊断已完成而且不是held特有过拟合：1140 materialization的fit19/held5 member exact-BA loss为`.92146/.89844`。
  held生成update相对selected direct的平均norm ratio/cosine虽从228的`.2518/.1117`、570的`.3200/.2721`继续到
  `.3694/.3290`，闭环却完全停滞。更决定性的证据是24个生成LoRA跨task effective cosine均值`.996807`，direct step2000
  只有`.131914`；mean own-direct cosine`.25076`反而低于nearest-other`.36600`，只有`1/24`自身检索正确。visible anchor与
  `q_pi` teacher process跨task cosine均约`.946`，q_pi correction虽把direct geometry correlation从`.4271`提高到`.4993`，
  但全局`residual_scale`仍停在`.1006`；compiler又把差异压到`.996807`。所以失败同时经过“privileged correction幅度过小”
  与“stable-prior A/B template residual输出近全局取消”两个连续接口，不能靠相同训练延长解决。
- 唯一Stage 1后继已写回active design：`q_pi`改为event/owner/content-dependent evidence gate；compiler的prior-only与full
  Program由同一网络分别输出完整**绝对**rank16 LoRA，full不再对stable-prior A/B各加residual；gauge-canonical factors只作
  坐标warm-start，仍以exact-BA、multi-state functional support、train reward/progress和held closed loop裁决。昂贵闭环前先
  要求compiled LoRA摆脱跨task坍缩与nearest-other错配；随后补source/shared support及授权non-held meta-task diversity。
  remote-safe证据为`docs/evidence/ecp_20260822/stage1_privileged_compiler_fold0_gate2.json`。
- 上述后继已在唯一Stage 1代码路径完成fresh-incompatible v2替换：47个successful member LoRA和stable prior先经compact-SVD
  进入统一rank16 gauge；`q_pi`删除全局scalar，输出`member×event×owner` sigmoid evidence gate且仍乘visible presence；同一
  compiler对zero-process精确返回prior完整LoRA，对full process直接输出absolute A/B，不再叠加template factors。训练新增
  member/consensus canonical-factor warm-start，同时保留exact-BA与functional目标；materialization一次forward train24时直接
  计算candidate/direct跨taskBA cosine、own-direct retrieval和norm ratio，未过门不启动fixed250。聚焦CPU合同共7项通过，
  覆盖BA保持的确定gauge、prior/full绝对面、q_pi信息墙、完整rank16输出及新projection evaluator。
- v2真实K2+functional单卡profile已在gpu01 physical1通过且未触碰prohibited physical0：一次更新2.17秒、峰值
  16,358,487,552 bytes，q_pi gate覆盖`.2539--.7773`而非全局scalar，prior-only exact-BA loss为0。初版统一head scale使
  initial effective loss达到`19.5`；直接统计47个gauge-canonical successful members后改为family-specific A/B scale，并以
  其隐含expert factor RMS的`.4x`初始化，initial member/consensus effective loss降到`1.1172`、canonical factor loss降到
  `1.2098`、裁剪前梯度从`3345.1`降到`61.0`。连续6个不同K2/functional visits均finite、每步约2.0--2.3秒；该profile只验证
  数值起点与运行稳定性，不作为科学曲线或checkpoint选择。
- v2从clean pushed `7ca808d`在gpu01 physical `1,2,3,4,5,7`完成fresh 228 visits/38 updates，exit0且六卡峰值约
  10.25 GB；但该段继承`functional_start_task_visits=228`，所以没有一次successful-policy functional update。24-task单次
  materialization的candidate跨task effective cosine仍为`.994192`，own-direct `.183969`低于nearest-other `.282906`，
  own retrieval仅`2/24`，effective norm ratio`.099771`；全部未过预注册门，故held rollout为0、同曲线停止。
- 回看专家Stage 1原文后，v2还存在一个此前未隔离的实现偏差：numeric owner/rank/event地址既进入compiler value，又以
  `hidden + query`直达factor heads；q_pi factor tokens也注入owner/rank常量。这允许地址在不保留task Program内容时写出
  近共享LoRA，而专家要求target query**读取**event/layer/family Program。当前唯一active修正因此是content/address分离：
  地址只影响keys、queries和locality，values与factor hidden必须来自Program；visible Program和q_pi evidence values不再重复
  注入地址常量；successful-policy functional从第一个update启用，BA/canonical降为低权重warm-start。v2证据见
  `docs/evidence/ecp_20260822/stage1_absolute_compiler_fold0_geometry.json`。
- v3聚焦CPU合同8项通过，其中新增反事实明确证明Program content全零时，numeric address/query不能独立写出full LoRA。真实
  K2+functional单卡profile在gpu01 physical1完成：functional从第一个update实际参与，loss `1.00456`；member/consensus
  exact-BA `1.1875`、canonical约`1.21995`，低权重组合total `1.84336`，裁剪前梯度`42.09`，2.20秒、峰值
  16,348,378,624 bytes，全部finite且prior-only loss为0。该节点授权fresh短段与几何门，不授权held rollout。
- v3从clean pushed `cba8caf`在gpu01 physical `1,2,3,4,5,7`完成fresh 228 visits/38 updates，exit0、checkpoint完整，
  prohibited physical0未使用。24-task物化显示candidate跨task cosine `.939205`、norm ratio `.487844`，相对v2的
  `.994192/.099771`证明content/address separation真正恢复了task-dependent输出；但own-direct/nearest-other只有
  `.012822/.026238`，own retrieval `2/24`，几何门失败，held5 rows为0。
- v3最早失败接口不是task差异仍被地址淹没，而是joint objective的到达顺序。训练前5到后5 updates的functional response
  `.995844→.871159`，member exact-BA却`1.14167→1.37474`、canonical factor `1.24101→1.56286`；物化fit19 member loss
  `1.48396`，比stable prior `1.10362`更差，held5也没有特殊断裂。当时后继v4因此保留全部结构，只让前228 visits使用
  exact-BA/canonical coordinate bootstrap且不缓存functional panels；几何过门后才exact-resume并补functional、source/shared
  support与fit reward/progress。证据：`docs/evidence/ecp_20260822/stage1_policy_functional_compiler_fold0_geometry.json`。
- v4当时替换了唯一active config/schema/evaluator路径；v3未保留兼容分支。真实K2单卡coordinate profile在gpu01 physical1通过：
  `objective_phase=coordinate_bootstrap`、functional loss与loaded panels均为0，member/consensus exact-BA为`1.1875`、canonical
  约`1.21995`、total `2.98603`、裁剪前梯度`140.48`，1.99秒、峰值10,125,596,160 bytes；临时profile目录已删除。
  clean pushed `fc0b84e`随后在gpu01 physical `1,2,3,4,5,7`完成fresh 228 visits/38 updates；前5到后5 updates的member
  exact-BA由`1.05391`降到`1.00938`、canonical由`1.14880`降到`1.08904`，但24-task materialization的own/nearest-other仅
  `.01840/.02945`、自身检索`1/24`、norm ratio`.09134`，故几何门失败且held5 rows为0。candidate跨task cosine`.85891`
  说明不是简单全局输出；gauge-invariant participation rank只有`1.0733`、top1 energy`.9664`，比direct的`1.2616/.9127`
  更集中。证据为`docs/evidence/ecp_20260822/stage1_coordinate_bootstrap_fold0_geometry.json`。
- active v5不恢复v2的`hidden + query`或address value旁路；target/rank query只以`1+tanh(Wq)`乘性调制已经cross-attend得到的
  Program content，因此Program全零仍严格不能写full LoRA，但不同target/rank可形成不同content-conditioned factors。
  active tree只保留v5 config/schema/evaluator。24项聚焦CPU合同通过，包括零内容反事实与modulation梯度；真实K2单卡
  coordinate profile也在gpu01 physical1通过，functional panels/updates均为0，member/consensus exact-BA为
  `1.17188/1.16406`、canonical约`1.2160`、total `2.95281`、裁剪前梯度`115.25`，2.02秒、峰值
  10,125,858,304 bytes，Prohibited physical0未用。临时profile目录已删除。
- v5已从clean pushed `ae15e47`在gpu01 physical `1,2,3,4,5,7`完成fresh 228 visits/38 updates；全段为coordinate
  bootstrap、functional panels/updates均为0，exit0且checkpoint完整。member exact-BA从前5个update均值`1.05208`降到
  后5个`1.00924`，canonical factor由`1.14952`降到`1.13216`。24-task物化中candidate pair cosine为`.87652`，own-direct
  从v4的`.01840`回升到`.08214`、自身检索从`1/24`升到`3/24`，证明乘性query-content路径确实接通；但own仍低于
  nearest-other `.10771`、norm ratio降到`.08643`，所有预注册门仍失败。fit19/held5 member loss分别`.99507/.99141`，
  held5 closed-loop rows为0，`q_V`未启动。正式证据为
  `docs/evidence/ecp_20260822/stage1_query_content_bootstrap_fold0_geometry.json`。
- v5把compiler的最后一个局部读出疑点缩小后，最早未覆盖接口已回到专家原始Stage 1合同：当前`q_pi`只读取successful
  member factors与压缩的successful-occupancy response；训练没有learner-policy occupancy、source/shared support或task-equal
  success/progress，relative exact-BA又允许接近零update取得约1.0 loss。下一fresh-incompatible Stage 1因此保持v5 compiler，
  将既有successful panels、30条projected learner trajectories及source/shared在相同状态面的response组成policy-support
  teacher；失败learner states按member agreement/outcome降权，并在warm-start后接fit simulator reward/progress。rank仍不获得
  技能语义，Program继续只有event/layer/family结构。
- active v6 policy-support代码面已经替换v5单路径：删除v5 active config与successful-only `stage1_panels.py`，新增一次性
  full-layer response bank builder、运行时support loader和一个v6 config/schema。每个occupancy state在paired RNG下捕获19层×
  50 Action tokens，经永久冻结owner projector和DCT4形成五类successful/learner/source/shared response channels；q_pi的
  channel/basis地址只进入attention key，Value来自response content。训练从首update交替successful/learner panels，candidate
  一次forward同时承担multi-member response、source/shared local support和低权重BA/canonical/locality；failed learner按
  member agreement与`.25` outcome base连续降权。旧v5 checkpoint与证据仍由Git/formal artifacts保存，不能resume到v6。
  35项聚焦CPU合同通过；实际30条learner asset loader确认19 fit tasks、30 trajectories、12 successes、held use为0。
- clean pushed `85477ea`已在gpu01 physical `1--6`并行构建78 MiB固定support bank：24 tasks、188 successful panels、
  120 learner panels，五通道RMS均非零；Prohibited physical0未使用。随后physical1双visit profile分别走通successful与learner，
  每步约`2.23/2.04s`、峰值16,401,996,288 bytes、梯度finite，临时profile已在证据固化后删除。
- 同commit的fresh formal完成228 visits/38 updates，115/113条successful/learner records，耗时115.68秒、六卡峰值
  16,626,005,504 bytes。moving-panel functional response从前5步`.64456`降到末5步`.50289`，candidate/direct norm ratio从
  v5 `.08643`恢复到`.64465`，说明policy-support确实写出了material update；但member exact-BA由`1.15677`恶化到`1.90182`。
  24-task物化own/nearest-other cosine为`.01618/.02816`、自身检索`2/24`，虽candidate pair cosine`.85242`且effective rank
  `1.3871`，仍未对准本任务完整policy方向。预注册几何门失败，held5 rows为0、`q_V`未启动，同曲线不续训。
- 当前最早判别是冻结checkpoint的**完整panel functional support**，不能只用moving training loss或direct参数cosine替代：下一步
  遍历successful/learner bank并分fit19/held5与source/shared报告。若冻结support成立，接入专家已要求的fit-task task-equal
  success/progress；若不成立，回到policy-support teacher最早接口。证据为
  `docs/evidence/ecp_20260822/stage1_policy_support_fold0_tv228_geometry.json`。
- clean pushed `a4928ce`已在gpu01 physical `1--6`完成冻结全bank task-equal audit。candidate在fit19的268 panels上相对
  source为`.80282x`且`19/19` tasks更好，held5的40 panels为`.90167x`且`5/5`更好，确认multi-policy support不是纯训练日志
  假象；但相对stable shared分别为`1.39966x/1.27745x`，`0/19`与`0/5` tasks胜出，预注册八条件只通过四个source条件。
  因此不加入reward、不评held闭环，最早断点明确为独立absolute full surface在写task LoRA时丢掉shared policy support。
- active v7只改变这个major variable：同一Program/q_pi/support与rank16目标不动，compiler heads改为generated residual；
  每target用thin-QR与`32x32` core SVD把`shared rank16 + residual rank16`的effective-update union重压为一套rank16 LoRA。
  这样保留shared起点、避免旧raw A/B相加交叉项，也不恢复第二adapter或固定`12+4`分槽；先复跑同一短门，过冻结support门后
  才接fit-task task-equal success/progress。
- v7 retained implementation已从clean pushed `6987933`完成真实BF16双visit profile。首次GPU profile在任何optimizer update前
  暴露autocast把FP32 QR core乘法转回BF16、CUDA batched SVD不支持的工程错误；修复把完整小矩阵分解显式留在FP32，CUDA
  autocast反向与22项聚焦合同均通过，失败临时目录已删除。修复后的successful/learner路径各完成一次update，耗时
  `2.696/2.245s`、峰值`16,457,865,216` bytes，梯度finite；初始candidate对stable-shared functional response loss为
  `.01638/.00284`，对source为`.75365/.66979`，符合prior-preserving起点。该profile只授权fresh 228-visits与同一冻结
  full-bank support gate，不构成科学promote或held闭环证据。
- v7正式228 visits/38 updates及24-task物化已完成，clean pushed `55b9065`随后并行遍历全部308个冻结support panels。
  prior union把v6的fit/held candidate-to-shared ratio从`1.39966/1.27745`修复到`1.02429/1.09995`，胜过shared的task
  breadth从`0/19、0/5`提高到`9/19、1/5`；相对source为`.58751/.77638`且24/24 tasks更好。但shared aggregate与两个
  breadth门仍失败，因此held closed-loop、reward和`q_V`均未启动，同曲线不续训。
- v7目标分解发现新的最早实现偏差：四项direct BA/canonical“低权重坐标锚”在前5步贡献`1.06492/1.65750`总loss，末5步
  仍贡献`.45761/.86428`，实际一直占多数；shared-support同时由`.03149`恶化到`.12364`。这与专家明确的“Stage 1目标不是
  raw A/B重建”冲突。下一fresh v8保持Program、q_pi、prior union、数据、seed和短节点不变，只把四项参数坐标loss从梯度中
  移除并继续作为诊断；同一冻结support门通过后才允许加入task-equal simulator success/progress。证据为
  `docs/evidence/ecp_20260822/stage1_prior_union_fold0_tv228_support.json`。
- v8从clean pushed `ae4805e`完成228 visits/38 updates，clean pushed `1659bb6`完成24-task物化与308-panel冻结审计。参数坐标
  梯度确为0，但fit/held candidate-to-shared反而为`1.15980/1.14903`，胜过shared的task breadth均为0；candidate/direct norm
  ratio膨胀到`6.54391`，candidate pair cosine升到`.97804`，own-direct仅`.01411`。因此该checkpoint不进入held闭环、reward或
  `q_V`。随后对formal metrics反查发现，完整456-visits schedule虽最终每task 24次，但全局cost-sort后随机group使228决策前缀
  每task只有`5--18`次访问，只有3/19 tasks恰为应有的12次。这违反短节点声明的task-equal裁决合同；当前只授权修正为每6轮
  task visit一个平衡block并fresh复验v8一次，仍失败才替换SVD union。证据为
  `docs/evidence/ecp_20260822/stage1_functional_union_fold0_tv228_support.json`。
- schedule修复从clean pushed `0b63da1`完成fresh 228-visits balanced v8：19个fit task均恰好12 visits，successful/learner各
  114 records，排除了短节点采样不平衡。训练并未改善，前5到末5 updates的total/functional/shared-support从
  `.40702/.29527/.02232`恶化到`.72780/.55854/.21882`；24-task物化的candidate/direct norm ratio继续升到`8.75029`，
  candidate pair cosine升到`.99433`，own-direct仅`.01106`且自身检索`1/24`。
- clean pushed `c1f485d`随后完成同一308-panel冻结审计。fit19 candidate/source/shared为`.47735/.70633/.40514`，相对
  source/shared为`.67581/1.17823`且只在`18/19、2/19` tasks胜出；held5为`.69756/.88454/.62434`，ratio
  `.78862/1.11729`且只在`4/5、2/5` tasks胜出。88/308 panels胜过shared，但四个shared aggregate/breadth条件均失败。
  v8因此最终关闭，不进入held闭环、reward或`q_V`；下一fresh-incompatible实现只替换rank16选择机制。
- v9已在唯一Stage 1代码路径完成fresh-incompatible替换。每个target的raw replacement A/B先在FP32形成独立正交方向，按该
  target stable-shared canonical factors的跨rank RMS能量定标；从Program-addressed hidden读取的零初始化per-rank selector
  angle只在对应shared/replacement rank-one mode间作有界retraction。selector为零时完整76 tensors逐值返回shared；raw head
  幅度放大7倍不改变输出；全部16 ranks均可写且不再运行Frobenius SVD。训练metrics和materialization新增唯一相关诊断
  `rank_replacement_fraction`。旧v8 active config与union调用已删除，40项聚焦CPU合同通过，包括exact prior、幅度不变性、
  selector先获梯度且打开后factor heads获梯度、零Program不能写LoRA、完整rank16输出及新projection authority。
- v9真实successful/learner双visit profile已在gpu01 physical1通过，prohibited physical0未使用。两次update耗时
  `2.3076/2.0783s`，峰值`16,438,259,200` bytes，裁剪前梯度`0.6435/1.4051`且finite；首步
  `rank_replacement_fraction=0`精确处于shared，第二步为`3.9241e-6`，证明selector已打开且没有幅度跳变。successful/learner
  functional response分别`.40044/.25523`，第二步shared-support仅`.00121`。该profile只授权fresh balanced 228-visits，
  不构成support、几何或held闭环pass。
- v9从clean pushed `dc5dff6`在gpu01 physical `1,2,3,4,5,6`完成fresh balanced 228 visits/38 updates；19个fit task
  均恰好12 visits，successful/learner各114 records，六卡峰值`16,657,711,616` bytes，最大裁剪前梯度`14.5268`且finite。
  前5到末5 updates的replacement fraction从`4.85e-5`升到`.08093`，shared-support仅从`.00222`升到`.02773`，明显低于
  v8末段`.21882`；但functional response从`.28632`恶化到`.40139`，不能由训练loss promote。
- 24-task K2物化已完成。bounded selector把candidate/direct norm ratio从v8的`8.75029`压到`1.94717`，但candidate跨task
  cosine反而为`.99779`，own-direct `.04015`仍低于nearest-other `.06247`，自身检索`1/24`；mean replacement fraction
  `.08783`且task范围仅`.08031--.09164`。几何门失败，held闭环仍为0。
- clean pushed `bb3bc59`完成308-panel冻结support audit。v9使fit candidate/source/shared为`.39853/.70633/.40514`，
  candidate相对source/shared为`.56423/.98369`，19/19优于source且10/19优于shared；held为`.62866/.88454/.62434`，
  ratio `.71072/1.00692`，5/5优于source但仅2/5优于shared。132/308 panels优于shared，297/308优于source；fit aggregate
  首次略胜shared，但fit/held breadth和held aggregate三个shared门仍失败。candidate-minus-shared correction跨task cosine仍
  `.97482`、相对shared norm仅`.24768`，证明它主要学到一个安全的近全局修正。v9最终关闭，不续训、不进入held闭环/reward/`q_V`。
  当前唯一后继保留bounded exact-prior selector，但让replacement值只来自present process tokens，language/scene只条件化query，
  去掉full/prior开关可凭静态token写出近全局修正的路径。
- Stage 0首个retained source里程碑已实现为唯一`ember.ecp`包：从canonical 38-target LoRA合同直接建立owner顺序，捕获
  native Action Expert全部18层输入与残差并立即投影为`[38,50,128]` lattice；task-grounded四类局部transition candidates
  与全部50 horizon双向绑定后，由固定容量8、动态presence的有序分段器形成`[8,38,128]` process与uncertainty。3项聚焦
  CPU测试通过，覆盖真实target顺序、冻结source/Writer梯度边界、末端horizon敏感性、T=1/变长mask和monotone slot
  posterior。随后在`gpu02:0`以真实source PI0.5、exact language和task0 action-hidden真视频帧完成同一native forward的
  forward/backward：patch=`[1,256,128]`、owner lattice=`[1,38,50,128]`、process=`[1,8,38,128]`，计算1.0203秒、峰值
  9,441,286,144 bytes，source梯度tensor为0、Writer 34个梯度tensor全部finite。该smoke证明真实底座接口与梯度墙接通，
  不代表Stage 0 event学习或性能已经通过。
- Stage 0A correct-only训练运行面已经接通：71个audited non-held tasks与train24 fold0 fit19组成90个collision-free
  authorities；每个正式macro对90 tasks各访问一次，6卡时每rank 15 tasks并按真实视频长度动态平衡；每次采样两条正确
  action-hidden视频、两条互斥action episodes和`1x/2x`保序速度视图。训练只更新native observer、transition/binding/
  segmenter与training-only action grounding head，source PI0.5保持零梯度；checkpoint保存Writer、optimizer/scheduler、
  rank RNG、world topology与macro cursor，不做hash校验。一次真实双视频单任务profile使用83/42帧，完整forward/backward
  为6.886秒、峰值11,524,790,784 bytes；据此formal frame microbatch从8提高到16以利用显存余量。该profile为吞吐证据，
  不是observer科学结果。
- Stage 0A retained source按当前真实职责拆分为`stage0`模型、`stage0_data`数据/调度、`stage0_objective`科学目标、
  `stage0_train_step`分布式task-equal macro、`stage0_training`运行编排和共享`ecp/checkpoint`续训合同；入口只有
  `scripts/train_ecp_stage0.py`。这些模块共同属于唯一`ember.ecp`运行面，不保留旧16维或LMMPC active fallback。
- Stage 0A macro1--10正式启动合同已冻结：实现authority为`f1eaee8`，从包含本记录的clean pushed `main`建立detached
  frozen worktree；在`gpu01`仅使用physical GPU `1,2,3,4,5,7`做world-size6 DDP，明确排除Prohibited GPU0并设置
  `NCCL_P2P_DISABLE=1`、GPU-local NUMA和deferred NCCL。规模为10 macros × 90 task-equal visits，每visit两条correct
  action-hidden视频与两条cross-episode action demonstrations；输入复用source step1000、71+19 audited HDF5和OpenPI
  tokenizer，输出固定为`runs/outputs/pi05_ecp_stage0_native_fold0_m10_r6_gpu01p123457_20260821/`。首段预计只新增一个
  约14 MB macro10 checkpoint及小量JSON/log，完整60-macro上限低于80 MB；启动前`/data1` user quota使用约624 GiB、
  limit约1,034 GiB，共享filesystem尚余84 TiB。只以correct-only训练loss、跨episode/速度/替代probe固定面板裁决observer；
  shuffled/reversed不进入训练或选点。仅允许相同world-size6与同一run contract从macro10 exact resume，不覆盖已有输出。
- Stage 0A正式段已从detached clean pushed `9b69e92`启动；生成的run contract确认world-size6、每rank 15 tasks、
  `CUDA_VISIBLE_DEVICES=1,2,3,4,5,7`、两组GPU-local NUMA affinity、frame microbatch16与1,133,487个trainable Writer参数，
  六份source policy均为冻结权重。启动后每张目标卡驻留约18.8 GiB，Prohibited GPU0保持14 MiB且无compute process。
- 上述首次启动在macro0完成90-task backward、尚未optimizer step或写入metrics/checkpoint时触发non-finite gradient并退出，
  因而是工程失败而非科学结果。最小CPU复现确认根因是event slot经验方差恰为0时对`sqrt(0)`反传，后续uncertainty平方形成
  `0 * inf = NaN`；固定`1e-4` variance floor后同一zero-variance复现从3648个NaN gradient变为全部finite，11项聚焦回归
  全部通过。失败输出只有run contract/invocation/log，登记后清理；正式段从包含该窄修复的下一clean pushed commit按原合同fresh
  重启，不复用失败状态。
- 固定observer panel运行面覆盖全部train24的fit19与held5、每task两个预注册demo pairs，
  对每个pair比较canonical correct video、同视频保序2x速度、same-task other video和同视频antithetic Gaussian probe；同时报告
  event/summary cosine、presence差异及same-task相对mean/nearest cross-task margin。该面板不读取action/reward，也不构造
  shuffled、reversed或wrong-video条件。首版macro10已完成48-row固定面板并产生下述Gate 1裁决。
- Stage 0A首版正式段在修复zero-variance工程问题后，从clean pushed `f6389af`以同一world-size6合同fresh完成10 macros、
  900个task-equal visits；全部loss、gradient与checkpoint finite，训练段417.55秒、单卡峰值11,664,018,432 bytes。
  total/action-alignment由`.587576/.144328`降到`.447487/.065853`，但cross-task contrast由`1.721745`恶化到
  `1.759694`，posterior entropy由`1.464783`坍缩到`.241540`。macro10 mean presence为`.348687`，90-task标准差仅
  `.001770`、范围`.345384--.355119`，source71与fit19均值又几乎相同（`.348722/.348560`），说明动态slot数量没有随
  task/video形成，而是共享了一个全局presence尺度。
- 同一固定macro10的observer panel进一步确认科学不通过：same-task other-video summary cosine虽为`.999985`，但mean/
  nearest cross-task cosine也高达`.996493/.999125`；fit19的nearest margin仅`.000927`，held5仅`.000604`。antithetic
  probe summary/event cosine仍为`.998766/.998409`。因此首版只证明图和监督可优化，没有识别task-conditioned event
  Program；不续训、不扫LR/rank/seed，也不把坍缩observer交给compiler或Action Meta正式arm。remote-safe裁决见
  `docs/evidence/ecp_20260822/stage0_native_macro10_gate1.json`。
- 最早失效接口的一组耦合修正已实现为canonical Stage 0 v2：删除所有task共享的learned `minimum_duration`分母，presence
  固定使用`1-exp(-(occupancy/valid_frames)/.08)`；event action head不再回归frame targets的event均值，而是经每帧soft
  posterior重构逐帧cross-episode action target。它不规定视频段落对应哪个slot，不加入shuffled/reversed训练，也不改变
  38-owner、50-horizon或动态E目标。v1 config已由v2单路径替换，旧formal合同只通过Git authority复现。
- Stage 0 v2真实83/42-frame单任务profile已通过：action alignment `.252319`、mean active events `3.5`、presence sum
  `4.43023`、gradient norm `.954190`，完整forward/backward 4.26秒、峰值11,497,050,624 bytes，checkpoint finite。38项
  聚焦测试通过，并新增两个直接回归：相同比例occupancy在1x/2x帧数下presence完全相同；把动作不同的两帧坍缩到一个
  event即使event均值预测正确仍产生`.25`逐帧误差。该profile只解除运行门，下一步仍须fresh macro10与固定panel裁决。
- Stage 0 v2实现authority为clean pushed `7bcda8f`。首段formal合同固定为：从包含本记录的clean pushed `main`创建
  detached frozen worktree，在gpu01仅使用physical `1,2,3,4,5,7`做world-size6，继续明确排除Prohibited physical0；
  复用同一source step1000、71+19 audited tasks、tokenizer、task-equal schedule与correct-only信息墙，fresh训练10 macros、
  900 visits，不读取旧v1 checkpoint。输出唯一固定为
  `runs/outputs/pi05_ecp_stage0_native_v2_fold0_m10_r6_gpu01p123457_20260822/`，预计新增低于100 MB；macro10立即运行同一
  correct/speed/same-task-other/antithetic固定panel。只允许同commit、world-size6、v2 run schema exact-resume；任何
  shuffled/reversed/wrong条件仍不进入训练、选点或本阶段panel。
- Stage 0 v2 formal已从clean pushed `395912a`完成10 macros、900 task-equal visits；训练段253.23秒、单卡峰值
  11,652,725,248 bytes，10行metrics、gradient与macro10 checkpoint全部finite。可是active events从macro1的`6.85`快速
  变为`5.26→3.02→1.39→1.03`，macro6--10严格为`1.0`；action reconstruction只由`.312241`降到`.251337`，cross-task
  contrast反而由`1.721222`升到`1.733799`。因此固定presence尺度消除了v1的可学习分母，却没有阻止event posterior本身
  先坍缩。
- 修复optional Action Meta路径解析后的clean pushed `3b6df9a`完成同一48-row panel。correct、2x speed、same-task other和
  antithetic四种条件在fit19与held5上全部只有1个active event；same-task-other summary cosine`.999981`，mean/nearest
  cross-task cosine仍为`.992826/.998369`，nearest margin仅`.001611`。v2 Gate 1继续失败，未启动Action Meta、compiler或
  held closed loop；panel的首次CLI失败发生在任何forward前、没有输出，窄修复后fresh重跑，不构成科学arm。
- 一个不训练模型的action-target可识别性诊断在8个固定tasks×2 views上比较最优常数event与有序8-bin均值：MSE为
  `.178693`对`.034727`，后者降低`80.57%`；而实际macro10 action loss`.251337`甚至没有达到常数上界。故数据中确有很强
  有序phase信号，最早失败是teacher action只在event pooling之后才产生梯度：随机frame evidence与event head尚未学会phase，
  sparsity/entropy/uncertainty/一致性已经先把posterior压成一个event。下一版本按专家原始Stage 0意见，在segmentation之前
  增加training-only frame action grounding并与event action head共享decoder；直接grounding建立前暂不施加collapse-promoting
  regularizers。remote-safe证据见`docs/evidence/ecp_20260822/stage0_native_v2_macro10_gate1.json`。
- Stage 0 v3已经替换唯一active config与checkpoint schema，不保留v2并行运行面。binding后的每帧`[4,38,128]`evidence先按
  candidate confidence聚合成`[38,128]`，与event process共用同一owner pooling和action decoder，分别产生直接frame action
  grounding与posterior event reconstruction；same-task consistency、uncertainty、presence consistency/sparsity和posterior
  entropy在direct grounding阶段权重为0，cross-task contrast保留为anti-collapse项。source、PaliGemma与Action Expert继续冻结，
  deployment没有新增action输入。聚焦9 tests通过，architecture guard无hard violation。
- live GPU预检后在gpu02 physical1完成真实83/42帧单卡profile；gpu01 prohibited physical0未使用。一个完整macro耗时4.43秒，
  峰值11,506,725,376 bytes，frame/event action loss为`.253187/.252328`，grad norm`1.706204`且finite，初始active events`3.5`、
  presence sum`4.430230`。相对v2同shape profile没有显存或吞吐回退，已具备fresh formal macro10的机械条件；profile临时产物
  不作为科学证据并在提交前删除。
- Stage 0 v3首段formal合同固定为：从包含本记录的clean pushed `main`创建detached frozen worktree，在gpu01只使用physical
  `1,2,3,4,5,7`做world-size6，明确排除Prohibited physical0；fresh训练10 macros/900 task-equal visits，不读取v1/v2 observer
  checkpoint，复用source step1000、71+fit19 audited tasks、tokenizer、correct-only cross-episode pair与v3静态objective。输出唯一为
  `runs/outputs/pi05_ecp_stage0_native_v3_fold0_m10_r6_gpu01p123457_20260822/`，预计新增低于100 MB；macro10立即运行同一48-row
  correct/speed/same-task-other/antithetic panel。launch前两节点live预检确认所选6张A40均0% util、14 MiB占用；`/data1` user
  usage为654,627,428 KiB、quota 1,073,741,824 KiB、limit 1,084,227,584 KiB，共享filesystem仍有84 TiB，空间充分。只允许
  同commit、world-size6、v3 run schema exact-resume；shuffled/reversed/wrong仍不进入训练、选点或本阶段panel。
- Stage 0 v3 native formal已从clean pushed `2d19ea8`完成10 macros/900 visits，训练段252.66秒、峰值11,664,016,896 bytes，
  metrics、gradient与macro10 checkpoint全部finite。frame/event action loss由`.312545/.312241`降到`.243966/.246427`；active
  events不再坍缩，10-macro轨迹为`6.85,6.75,5.93,6.19,6.06,5.91,6.21,6.67,7.01,6.97`；cross-task contrast由
  `1.721222`持续改善到`1.376669`。这解除v2最早的pre-segmentation identifiability失败，但不单凭训练loss裁决observer。
- 同一checkpoint的固定48-row panel在clean authority上完整结束：correct平均`6.48`个active events且跨rows覆盖4--8，
  same-task-other summary/event cosine为`.999601/.999270`，mean/nearest cross-task cosine降到`.909019/.980528`，对应margin
  `.090582/.019073`；48/48 rows、held5的10/10 rows nearest margin均为正。速度视图summary/event cosine为
  `.999975/.999871`。因此v3 native首次形成跨episode且泛化到held5的task-specific event geometry，授权进入独立Action
  Meta-LoRA matched arm。它还不是最终observer authority：antithetic summary/event cosine为`.978224/.976424`，只有16/48
  rows的antithetic same-task cosine高于canonical nearest-cross cosine，probe稳定性仍是Stage 0B的主要裁决项。native结果见
  `docs/evidence/ecp_20260822/stage0_native_v3_macro10_gate1.json`；compiler、`q_pi`和closed loop仍未启动。
- v3 Action Meta-LoRA在冻结native macro10上完成真实83/42帧单卡profile：只有626,688个shared rank4 adapter values可训练，
  source与native observer其余参数无gradient；frame/event loss`.200828/.206732`、adapter grad norm`.000751`均finite，13.56秒，
  峰值15,083,217,920 bytes，active events`6.5`。profile checkpoint已删除，只解除v3 objective下的机械/显存门。
- Action Meta v3 formal合同固定为：从包含本记录的clean pushed `main`创建detached frozen worktree，冻结native
  `runs/outputs/pi05_ecp_stage0_native_v3_fold0_m10_r6_gpu01p123457_20260822/checkpoints/macro_00000010`，在gpu01 physical
  `1,2,3,4,5,7`做world-size6，继续排除Prohibited physical0；只训练同一shared rank4 Action Meta-LoRA，保持90 tasks、pair
  schedule、v3 objective、correct-only信息墙和10 macros/900 visits完全matched。输出唯一为
  `runs/outputs/pi05_ecp_stage0_action_meta_v3_fold0_m10_r6_gpu01p123457_20260822/`，预计低于100 MB；macro10后用native
  checkpoint加该Meta checkpoint运行完全相同48-row panel。launch前`/data1` usage 654,639,124 KiB、quota 1,073,741,824 KiB、
  shared free 84 TiB；只允许同commit/world-size6 exact-resume，不与compiler共同训练。
- Action Meta v3已从clean pushed `a42601a`完成10 macros/900 visits，耗时825.15秒、峰值15,432,632,320 bytes，所有adapter
  gradients与macro10 checkpoint finite；active events在全部macros保持`6.83--6.91`。matched 48-row panel保留native的48/48
  positive nearest margins、held5 10/10和antithetic-closer 16/48；nearest margin由`.01907267`变为`.01908323`，mean margin由
  `.09058216`变为`.09061162`，antithetic summary cosine由`.97822402`变为`.97822047`。逐row变化均在约`1e-4`以内，panel
  耗时188.85秒、峰值与native完全相同。因此它是无可复现负面效果、也无可测显著收益的中性校准；按owner既定规则采用并
  永久冻结`native macro10 + Action Meta macro10`为唯一Stage 0 observer authority。它只在Writer观察教学视频时安装，最终
  rollout LoRA仍只有compiler生成的一套，不部署第二adapter。Gate 1完成，证据见
  `docs/evidence/ecp_20260822/stage0_action_meta_v3_gate1.json`。
- 独立Action Meta-LoRA运行面已经实现：复用唯一`MetaLoRAStack` owner，对18层Action Expert的q/k/v/o投影加shared rank4、
  只在observer calibration时安装，checkpoint与panel均独立于native；source、native observer及其post-capture参数冻结，部署
  不携带第二adapter。单卡首个profile暴露全层adapter反传的activation OOM，已改为复用PI0.5原生per-layer activation
  checkpoint并让adapter hooks覆盖recompute/backward；这只是运行面修正。由于native v1已坍缩，Action Meta正式科学训练
  仍明确withhold，待修正native通过同一固定panel后再作matched对照。修正后单卡真实83/42-frame profile已成功：只训练
  626,688个rank4 adapter参数，macro耗时13.50秒、峰值15,009,683,456 bytes、gradient norm `.00011523`且checkpoint
  finite；因此Action Meta运行门已解除，临时profile产物在记录本结果后删除，不当作科学证据保留。
- 外部专家A--G/F0--F5逐项复核goal已完成；113个编号claim均已实施、反驳或以有证据的
  `not-applicable` / `underdetermined-after-audit`收口，没有queued项。
- `docs/functional_adaptation_successor_design.md`继续描述已经封存的16维代码与实验authority；
  `docs/policy_native_dual_time_program_compiler_review_20260821.md`保留最初送审方案及专家纠正的provenance。二者均不是
  implementation authority，当前16维process warm-start、旧single-direction outer和任何held复验都不再启动。
- canonical workspace与集成目标为`/data1/user/ymdai/projects/EMBER`的`main`；开发只在从最新`main`创建的短期
  `codex/<topic>` worktree隔离，验证后的独立里程碑立即合并、推送并清理，不长期积压巨型分支。
- 来自clean pushed `7b6d768`的train24 fold0 fixed functional decoder formal评测已完成。修正投影wiring后的F4
  free-Program 1200行也已完成：projected=`307/1200`、direct=`658/1200`，253 retained、54 gained、405 lost、
  Jaccard `.35534`，只保留direct的`46.66%`，明确未过90%门。旧`659/1200`来自没有实际安装投影LoRA的错误评测，
  已撤销并重导出remote-safe证据。
- non-held meta expert bank已经按固定uniform-step合同完成71/71 tasks、每task 1000 steps；canonical输出仍为
  `runs/outputs/pi05_nonheld_meta_expert_bank_step1000_r6_650d922_gpu01p012457_20260819/`，后续direct、decoder与projection均
  复用这套昂贵产物，没有重训。相同固定rows上的direct experts为meta-train `2519/2800`、meta-validation `684/750`；
  相对source分别净增`+247`与`+38`，证明当前pool存在可裁决的policy-effective增量，而不是由高source identity自动过门。
- 同一15-task meta-validation面板的frozen source为`646/750`、direct为`684/750`、fixed-decoder projected为
  `659/750`。source→projected为612 retained、47 gained、34 lost、净`+13`、churn81，exact McNemar `p=.18208`；
  direct→projected保留621/684=`90.79%`的direct successes。15/15 tasks均有breadth@10，task73从source `4/50`
  提升到projected `15/50`，但增量不显著、6 tasks正/6负，且只复现direct gain rows的54.67%，不能宣称decoder方法通过。
- 其余56个meta-train tasks的source/direct/projected分别为`2272/2800`、`2519/2800`、`2451/2800`。
  source→projected为2157 retained、294 gained、115 lost、净`+179`、churn409，projected保留direct successes的
  92.62%；34 tasks正、11负、11持平。Study净`+99`、pick-place净`+102`，说明改善不是只靠高base aggregate；结合完整
  71-task source `2918/3550`，当前不触发无差别source重训。source覆盖摘要见
  `docs/evidence/functional_adaptation_20260819/nonheld_meta_source_coverage_71.json`。
- default fold0 functional decoder formal从fit loss `1.040443`降至`.481957`、held loss从`1.035567`降至`.830093`，
  14/15 held tasks改善；冻结后导出71套完整LoRA并完成上述严格paired closed loop。当时Gate 2据此只作
  `qualified_pass_to_writer_inference`并启动macro10；后续坐标审计已撤销其Writer泛化authority，因为train/held codes
  分布不一致。该结果只继续支持decoder range值得重验，不能升级为fixed-coordinate科学结论。完整原裁决见
  `docs/evidence/functional_adaptation_20260819/nonheld_meta_fixed_decoder_fold0_20260820.json`。
- successor Writer formal已经从clean pushed detached `main@a2999d5`完成56-task、6-GPU、macro1--10，耗时
  780.43秒。total objective由`2.082406→1.853265`，combined/language/video分别由
  `.995820/.994485/.984880→.838643/.861964/.974670`，10行metrics与gradient均finite；峰值32.60 GB。
  Text/VL Meta-LoRA均为0，fixed decoder与VLM保持冻结，唯一macro10 checkpoint含Writer、trainer与world-size6状态。
- 同一macro10 checkpoint已在一张空闲A40完成真实video→完整LoRA的B8/B16/B32 profile；三者均稳定、零OOM，吞吐分别
  `.898928/.897234/.896828 LoRA/s`，B8按预注册规则胜出，最长48 sampled frames，最小显存余量35.50 GB。active config
  已封存该证据并只允许profile支持的batch；该昂贵checkpoint与cache继续保留作反事实，不重复训练。
- macro10的15-task×10-state matched screen为correct/language-only/video-only/first+final/reversed/shuffled/static分别
  `131/130/134/130/134/133/132`。correct相对source为123 retained、8 gained、8 lost、净0；相对video-only与reversed均
  只有1 gained、4 lost。所有arms breadth@1均15，但correct没有full-video、order或dynamic Value优势，Gate 3明确失败。
- 最早失效接口不是简单“macro10还没训够”：56个训练code的coordinate std约1、平均task norm`5.589`，15个held codes
  从零自由拟合后std仅`.089`、平均norm`.505`，最近train code平均距离`4.523`。Writer训练与leave-out评测不在同一个
  code分布。effective-BA诊断进一步显示video-only的task-mean只有1/15最近邻到正确projected task，却距15-task共享均值
  仅`.000998` relative error；其134分主要是shared carrier/base competence，不是video-conditioned task inference。
- 统一fingerprint formal已复用现有71-task experts完成：feature width `25600`，train-only 32维PCA解释方差`.887516`；
  train/held coordinate std为`1.0000/.7248`、平均norm为`5.5701/4.1436`。held只做同一固定变换且不产生梯度，旧版
  `.505` near-zero held code问题已解决。这套昂贵fingerprint artifact继续复用，不重复收集。
- 固定上述codes的flow-only Decoder将fit/held flow loss由`1.040443→.445721`和`1.035567→.664218`，但同一750 rows
  closed loop仅`644/750`，低于source `646`、direct `684`和旧projected `659`。相对source净`-2`，相对direct净`-40`
  且`p=3.67e-5`；task73仍为`4/50`。因此没有启动56-task复评或Writer。
- flow-only生成effective `BA`相对direct的relative-L2 `2.8576`、cosine `.0254`、norm ratio `2.7004`，说明有限flow
  queries允许巨大近正交off-manifold解；该objective已关闭。
- shared-zero carrier formal已完成`640/750`，相对source净`-6`；shared-zero→task fingerprint只有30 gained、26 lost、
  净`+4`、`p=.68888`。上一轮`644`几乎由共享输出解释，task-specific code无可靠增量；该carrier只作诊断，不作为fallback。
- 固定8-probe effective Decoder从clean pushed `c3e5bc1`完成1120 steps，7分46秒、峰值18.92 GB；完整effective `BA`
  的train/held relative-L2为`1.1387/1.1292`、cosine仅`.0642/.0449`，按固定方向过拟合关闭。
- exact低秩Gram Decoder从clean pushed `423a9b2`完成1120 steps，324秒、峰值18.92 GB；train/held BA geometry改善到
  relative-L2 `.8423/.9591`、cosine`.5365/.3032`，但同一held750 rows只有`638`，低于source646、direct684、旧
  projected659、flow-only644与shared-zero640。相对source净`-8`、相对direct净`-46`且`p=1.96e-5`；held loss在
  step280--1120约`.926→.921`平台化，不续训。
- 无训练仿射full-BA诊断的train/held relative-L2为`.5244/.9797`、cosine`.8439/.3648`，设计condition number`1.009`；
  排除“只需更小线性decoder或canonical factor”的窄解释。当前最早失效接口上移到单expert功能标签与source/meta任务角色。
- validation8 sealed task-local oracle合同已完成：八套独立rank16 LoRA统一训练到预注册step2000，只在step1000
  exact-resume，不更新共享Writer/decoder、不选checkpoint、不读取Test。clean detached `5fd224a`上的strict400为
  `250/400`，既有frozen source为`48/400`；严格配对43 retained、207 gained、5 lost、145 retained failures，净`+202`、
  churn212，McNemar exact `p≈1.06e-54`。八项全为正净增量，breadth@1/@5/@10均8，四suite分别为Spatial `73`、
  Object `78`、Goal `58`、Long `41`，预注册强门明确返回`advance_to_successful_on_policy_manifold_panel`。remote-safe
  400-row与stage证据见`docs/evidence/functional_adaptation_20260819/validation8_task_local_oracle_step2000_20260821.json`。
- oracle stage trace显示Long的剩余问题主要是多阶段完成而非完全无primitive：Long task1中cream-cheese ever/final为
  `31/27`、butter为`13/13`、full peak与最终成功均`12/50`，四行在完成第一对象后又丢失；Long task2的stove-on为
  `50/50`、moka-on-stove与最终成功均`29/50`。Goal task3的BDDL只有最终谓词，不能观测开drawer中间阶段。该trace是
  无序final-goal代理，不是完整有序procedure标签。
- source/meta角色分离的19/5 formal诊断已经完成。相对旧重叠任务面的无训练仿射held cosine `.3648`，role-disjoint
  flow/action单标签分别为`.4310/.4341`，说明改变数据角色后几何有方向性改善；但任务集合也改变，不能把差值当成纯因果
  归因。两种16维train-only-whitened code的held std均约`.78`，不再出现near-zero held坐标。
- 同一轨迹的成功阈值多checkpoint等价集只把flow/action aggregate held cosine提高到`.4355/.4394`，证明“多checkpoint”
  本身没有解决标签欠识别，也不能冒充独立成功策略分布。关键分层是：有checkpoint达到`25/50`的held tasks 0/9/18上，
  action-response达到cosine `.5942`、relative-L2 `.8394`；没有成功checkpoint的tasks 25/36只有`.2071/1.0766`。
  这把下一裁决集中到task-local ceiling与成功occupancy，而不是继续换Decoder objective。
- denoised action-response已按共享显式noise、official 10-step integration与完整`50×7`action chunk完成formal收集；相同
  anchors的flow为`50×32`。action在有成功expert的held子集优于flow的`.5753/.8712`，但全5-task aggregate仍未过
  预注册几何screen；两者都只是无闭环、未物化adapter的定位证据。remote-safe摘要见
  `docs/evidence/functional_adaptation_20260819/role_disjoint_manifold_20260821.json`。
- 精确policy-JVP机制smoke已在一项成功train expert上接通：完整`50×32`action-sequence JVP finite，峰值`17.44 GiB`，
  expert-source JVP差RMS `.028757`、cosine `.999663`。它只证明专家建议的Jacobian标签可实际计算；在validation8
  ceiling裁决前不扩为全量fingerprint或Decoder objective。证据见
  `docs/evidence/functional_adaptation_20260819/policy_jvp_feasibility_20260821.json`。
- formal-validation-only BDDL stage capture已接通：同一rollout内记录goal predicates的初值与change points、ever/final及
  peak count；真实LIBERO wrapper与两谓词任务解析通过，聚焦测试3项通过。它使用privileged simulator state但不读
  teacher action/reward、不产生梯度或改变success，且只作为无序最终合取的阶段代理。step2000 strict400将同时收集，
  不额外重跑rollout。证据见
  `docs/evidence/functional_adaptation_20260819/stage_predicate_capture_smoke_20260821.json`。
- successful/on-policy panel已从clean detached `febdff0`完成：四个non-held tasks的8/8预注册direct rows全部成功，
  没有替换；task23两条为208/233步、task26为178/105、task80为107/120、task86为135/139。八条trajectory sidecars共
  约372 MiB，动作、RNG和BDDL stage同轮保存，未读取held或重训expert。
- clean detached `1e45c66`在四张A40上对每轨迹8个progress strata重新配对source/expert：denoised action只有task23/86
  通过full+early same-task mutual-cosine-nearest门，即`2/4`；task26的gained trajectory最近是task86，task80的retained
  trajectory最近也为task86。exact JVP只有task80通过，即`1/4`，按预注册规则不能覆盖action失败。全部early states和
  事实上全部64个selected states都尚未完成BDDL goal conjunction，因此不是final-predicate捷径造成的假失败。
  该结果淘汰直接concatenation，不否定phase-aligned action family。证据见
  `docs/evidence/functional_adaptation_20260819/successful_onpolicy_response_panel_20260821.json`。
- 回查专家原始意见后，当前最早接口进一步具体化为`successful adapter equivalence + occupancy/phase alignment`：下一面
  复用target train24现有step250--2000 checkpoints与formal rows，每task取最早/最晚成功checkpoint各自一条最短成功轨迹；
  23 tasks可形成K2，唯一只有step1000一次成功的task形成K1。fit19学习单调phase alignment与固定坐标，held5只做固定
  变换；JVP不再是primary label，aligned representation过门前不重建decoder。
- 该面已在clean detached `545b43c/7258487`完成：四个checkpoint capture共47/47条预注册成功轨迹，无替换、无expert
  重训；完整每-replan `50x7` action delta经fit19-only、task/member/state等权PCA/whitening后，32维解释方差`.923430`。
  held5的等时间与功能弧长表示均为`5/5`同task mutual-nearest，功能弧长在`4/5`任务提高same-task cosine，按预注册门
  `advance_to_phase_aligned_fixed_decoder`。fit19的mutual-nearest从等时间`15/18`变为弧长`14/18`，因此当前结论是组合
  标签已具备leave-task-out可识别性，不是弧长在所有面板单调占优。证据见
  `docs/evidence/functional_adaptation_20260819/train24_successful_equivalence_phase_20260821.json`。
- fresh Decoder合同已在任何优化前写入`configs/pi05_train24_phase_aligned_decoder_v1.json`：task consensus PCA16只拟合
  fit19，多个成功成员各自的8个真实phase states轮换提供完整flow监督；held5 earliest/latest member code均零步优化、分别
  物化完整LoRA。训练固定5-rank、950 task visits/190 optimizer updates；最终functional门只作安全诊断，是否进入新Writer
  由预注册held5两套strict250闭环联合门决定。新入口`train_phase_aligned_functional_decoder.py`是本轮唯一active Decoder
  candidate；旧trainer只维持sealed历史复现，若本轮闭环通过则退役旧入口，若失败则删除新candidate并保留失败证据。
  实现所有权按单一流水线拆分：`phase_code_building`只构造冻结坐标，`phase_decoder_codes`只校验code authority，
  `phase_decoder_panels`只绑定成功轨迹监督，`phase_decoder_training`只负责分布式优化/精确恢复，
  `phase_decoder_projection`只物化两套评测bank；两个`scripts/`文件均为薄入口，不构成平行算法实现。
- clean pushed `73c2a32`上的fresh Decoder以5-rank完成950 task visits/190 updates；fit/held identity-relative flow loss为
  `.323930/.616152`，earliest/latest held family为`.636755/.595550`，先通过内部安全门。随后同一held5固定250 rows上，
  source/direct-earliest/projected-earliest/direct-latest/projected-latest为`21/74/44/108/44`。两套投影相对source均净
  `+23`且5项不退化、3项严格提高，earliest/latest成功集Jaccard `.466667`；但direct success retention只有
  `20/74=.27027`与`28/108=.25926`，direct gain retention只有`.17742/.21875`，显著低于预注册`.75/.60`。
  Gate 2因此返回`do_not_promote_decoder_to_video_writer`。最早失效接口不是phase code完全无效，而是只在successful
  expert occupancy做flow distillation无法守住decoded policy自身闭环occupancy上的support。下一步复用全部昂贵产物，
  只增加fit19 projected-policy state aggregation；若仍失败，再落实已被证据触发的shared prior + task residual单LoRA。
  remote-safe证据见`docs/evidence/functional_adaptation_20260819/train24_phase_decoder_held5_20260821.json`。
- learner-occupancy聚合已在任何新优化前冻结为
  `configs/pi05_train24_phase_decoder_onpolicy_state_aggregation_v1.json`：只用旧Decoder的fit19 consensus投影从30个既定
  successful初态采集30条轨迹，并把重复初态分别绑定到37个earliest/latest privileged expert成员；每成员按projected
  action-chunk功能弧长取8个真实learner states。warm-start旧Decoder、fresh optimizer，以successful/learner panels严格
  1:1配对，6-rank完成912 task visits/152 updates。held5、phase code、expert bank与闭环门均不改；这一轮失败后不续做
  state-bank小扫，直接转入已触发的shared prior + task residual独立架构裁决。
- clean pushed `966353e`上的learner-state聚合训练已完成：30条fit19 projected trajectories、37个member targets，
  learner-state identity-relative flow loss从`.629034`降到`.155116`，held mean从旧`.616152`降到`.560983`。同一held5
  fixed250上earliest/latest为`54/47`，相对旧投影分别22 gained/12 lost、18 gained/15 lost，即净`+10/+3`；相对source
  净`+33/+26`。但direct success retention仅`23/74=.31081`与`30/108=.27778`，direct gain retention`.19355/.26042`；
  latest只有1/5 task严格正增量，earliest/latest成功集Jaccard`.40278<.44`。Gate 2返回`do_not_promote_decoder`。
  remote-safe证据见
  `docs/evidence/functional_adaptation_20260819/train24_phase_decoder_state_aggregation_held5_20260821.json`。该结果只关闭
  当前一次staged learner-state实现，不关闭occupancy、stage behavior或outer reward；下一步不重跑本训练，直接实施
  shared prior + task residual且保留shared-only matched baseline。
- 回查专家挑战十二的原始公式后，当前参数化已在任何新优化前进一步冻结：不把两个full-rank A/B states直接相加，避免
  `BA`交叉项；rank16精确分成shared rank12与task residual rank4。stage1用固定zero code与fit19
  successful/learner 1:1 panels学习task-independent prior，stage2冻结prior并用zero-code-centered phase-code residual只写
  后4 ranks；最终按rank维组成一套complete LoRA，`D(0)`逐tensor等于shared-only。两阶段各6-rank、912 visits，复用
  已有state bank且不重采；正式闭环同时比较shared-only、earliest composite与latest composite。
- clean pushed `e948fca`上的两阶段formal与三臂held5 fixed250已完成。shared stage的fit/held mean为
  `.575078/.680319`，residual stage降到`.403687/.659049`，两者内部functional门均通过；闭环却为source21、shared43、
  earliest37、latest33。source→shared是17 retained、26 gained、4 lost、净`+22`；shared→earliest为29/8/14、净`-6`，
  shared→latest为29/4/14、净`-10`。composite direct success retention仅`.22973/.15741`，direct gain retention仅
  `.09677/.07292`。earliest/latest Jaccard`.62791`说明两者可稳定地共享同一窄support，但不能替代task增量。M方向因此
  mixed：shared底座获支持，当前12+4 functional residual淘汰；不续训、不扫rank/LR/seed。remote-safe证据见
  `docs/evidence/functional_adaptation_20260819/train24_shared_prior_residual_held5_20260821.json`。
- 回查专家原文后，下一主线不是丢掉shared结构，也不是把43分当作视频能力，而是实施挑战十四/方向E尚未覆盖的
  train/meta closed-loop outer objective。先复用现有`ember.reward` rollout/seed/occupancy和functional warm-start资产，
  冻结shared prior与held5边界，只让授权train/meta reward为task-conditioned inference提供外层credit；matched
  shared-only与functional-only仍保留。held validation/test reward不产生梯度，post-generation task-local RL仍单列。
- 挑战十四/方向E的首个实现已在clean pushed `506daa2`完成并由`2f9fdb0` frozen evaluator裁决。两轮functional
  warm-start后的macro2在held5 fixed250为`41/250`、breadth `3/5`，其成功集是shared-only `43/250`的严格子集；world6
  exact-resume后一次outer macro在fit19以success、efficiency与BDDL progress得到10/19 nonzero-advantage tasks，但macro3
  反而为`39/250`。macro2→3是37 retained/2 gained/4 lost，shared→macro3是39/0/4，没有任何shared support之外的新成功，
  Goal与Long仍为0。因此按Gate 4停止macro4，不扫rank/LR/epsilon/seed；只关闭当前单方向antithetic finite-difference实现，
  不关闭outer credit一般。证据见
  `docs/evidence/functional_adaptation_20260819/train24_functional_outer_credit_held5_20260821.json`。
- macro2 checkpoint的matched held5 fixed250面板已全部完成：correct/language-only/video-only/first+final/
  same-task-other=`41/39/40/39/40`，breadth均`3/5`，Goal与Long全0。language→correct为35 retained/6 gained/4 lost、
  `p=.75391`；first+final→correct为36/5/3、`p=.72656`；correct→same为36/4/5，correct-success retention仅
  `.87805`。因此完整视频没有显著语言增量或端点之外的过程价值，最早失效接口在outer之前。
- 回查专家原文及run contract确认没有遗漏56-task meta process预训练：历史Writer的212 tensors/8,121,416 values已迁移，
  旧decoder未迁移；仅三个`32→16` code最终层因shape变化重建。当前新heads只经两轮LR `2e-5` correct-only训练，
  control/action权重为0。下一节点复用已有checkpoint、fixed decoder与训练owner，把跨episode action-phase alignment、
  reversed/shuffled/first+final/endpoints-middle-shuffled和K1--4直接施加到新16维heads；先复验matched panel，过门后才启动
  结构不同的outer estimator，失败则继续按专家stop gate核对meta-task/multisplit并准备替代路线。
- process-supervised运行面已实现且没有新增平行Writer：现有functional-code task-loss owner现在可按权重选择action/control，
  outer warm-start复用该owner；旧correct-only config保持action读取为0，新config固定decoder并声明四类真实帧control、K1--4、
  fit19跨episode action-phase与12-macro warm-start checkpoints。gpu02/p7单任务K1 reversed真实smoke在3.22秒训练段得到
  total loss `2.25301`、action alignment `.22331`、control update `.002316`、finite grad norm `8.39418`，峰值
  `32,486,524,416` bytes；只证明图接通，不作性能选择。
- owner要求当前实现收口后进入中期讨论，因此formal process warm-start及matched held5复验尚未启动。当前goal继续active，
  下一授权动作由讨论结果决定；不会把smoke写成专家方向已通过。
- `main`上的已封存Writer仍是Core-Addressed Reader主架构：Dynamic-K、rank16、38 targets、Action Meta-LoRA、
  layer/rank memory、Reader、K-set、bounded M2P和FactorHeads；原生language保留，Text/VL Meta-LoRA已从
  canonical config/code contract移除。该实现只作为sealed baseline和可复用组件来源，不再作为后继增量路线。
- 不直接返回V6/LPCP/GOMQ，也不恢复旧Expert-Manifold为held dictionary；历史实现只提供paired反事实、functional
  probe、checkpoint/evaluation等可审计复用候选。

## Latest owner decisions for successor planning

- 允许train24 privileged experts训练共享functional decoder；也允许使用LIBERO-90中经审计、排除固定validation/test
  tasks及其重复项的non-held任务，必须保留显式allowlist与provenance；
- 允许learned language-only adapter作为baseline，用于裁决video条件增量；
- 允许在授权train/meta tasks上用simulator reward训练共享Writer/functional code inference。该outer RL仍以held
  zero-interaction LoRA为部署对象，不等于生成LoRA后的task-local RL；
- 允许冻结模型、无梯度、无checkpoint选择的sealed held action/reward诊断；Test默认留到最终方法冻结后；
- 合理的新架构均可考虑，包括rollout前合并为唯一完整LoRA的shared prior/base adapter + video-conditioned residual；
  不允许部署第二adapter、expert route、task-ID字典或checkpoint融合；
- 主写与集成目标改为`main`；需要隔离时从最新`main`创建`codex/<topic>`分支/worktree，验证后及时合并并推送。

## Sealed functional-adaptation predecessor snapshot

以下顺序与实现描述的是EMBER-ECP之前已经完成并封存的functional-adaptation路线，不再授权继续执行：

1. 审计现有expert manifold、Writer、reward/evaluation与LIBERO数据owner，建立non-held meta allowlist、task-level folds、
   process controls和source/task-expert ceiling协议；
2. 用policy-functional response而非raw A/B几何学习compact code与固定complete-LoRA decoder，并以leave-task-out
   closed loop作为进入门；
3. 固定decoder后学习language prior + action-hidden video process posterior，保留完整Action probe与有向阶段结构；
4. functional warm-start后在train/meta simulator接入closed-loop outer credit；
5. 用strict paired400、相邻checkpoint、same-task不同视频、Long、breadth和多split复现选择或停止方法。

专家方向A--N和五个替代研究问题都已进入ledger。runtime video policy、task-local RL、richer sensing以及
video-to-reward/skill/plan不是被丢弃，而是在核心single-LoRA路线触发预注册stop gate后按证据启动；train/meta action
alignment、mergeable base+residual与sealed diagnostics已经获准进入对应phase。

已完成的Phase 0实现：

- `configs/libero90_nonheld_meta_v1/protocol.json`显式保留71个去重non-held tasks、排除19个target-overlap tasks，并建立
  5个不读取结果的task-level folds；默认56 meta-train / 15 meta-validation，冻结后轮换复现；
- `ember.functional_adaptation.contract`加载allowlist/folds并验证source manifest与语义overlap audit一致；
- strict video conditions已增加first-only、final-only、first+final、endpoints-fixed-middle-shuffled与monotone-sparse，
  真实frames经选择/重排后重新完整forward；
- 当时的新模块owner与旧`expert_manifold`/Writer/evaluator复用、退役边界已写入sealed predecessor design；旧bank route不恢复。
- `FunctionalCodebook`与`FunctionalAdapterDecoder`已经建立32维whitened task code到全部38-target/76-tensor LoRA的
  单一生成面；decoder以functional identity初始化，Action in/out保持独立，不import旧V6 bank route；
- policy-functional probe会捕获完整`[batch, 50, 32]` Action Expert flow response，并以expert相对identity的响应能量
  归一化监督，避免source policy的大幅公共响应淹没task adapter信号；首轮相关20项CPU测试通过。
- non-held meta expert合同已固定71 tasks中的56 meta-train / 15 meta-validation-oracle，并复用唯一task-expert训练owner；
  fixed decoder也已能从该bank按角色拟合/冻结和导出32维code，不建立task-ID deployment route。
- 后继Writer运行面已实现为`language prior z_L + ordered-video posterior delta(L,V) -> frozen decoder -> one complete LoRA`：
  每条视频独立保序编码initial/goal/event/transition，跨K只聚合完整video program；保留50个Action probe并加入仅训练期
  meta-action phase alignment，同时提供真正不读language/action probe的video-only baseline。模块按decoder、inference、
  schedule/step/checkpoint和privileged-action owner拆分；旧LMMPC继续只作为sealed历史基线，不形成并行active fallback。
- successor已经接入现有唯一PI0.5 evaluator、episode LoRA cache、persistent rollout worker和online generation profiler；支持
  fixed 56-task meta-train / 15-task architecture-validation角色及correct、same-task-other、wrong、language-only、video-only与
  真实帧时序controls。learned language-only部署分支不打开视频，video-only分支不读取language或Action probes；训练期
  teacher-action alignment改为同task但确定性不同episode，并按归一化过程相位配对，避免逐帧动作复制。
- process panel新增两个诚实条件：把真实首帧重复到原长度并保留source-time positions的`static_first_repeated`，以及读取
  同一episode同一时刻`eye_in_hand_rgb`的`eye_in_hand_view`。HDF5只含双路RGB且没有depth/segmentation；robot/object mask
  不会通过teacher state重渲染伪造，可信RGB-only flow仍登记为未解决数据/表示缺口。
- 当前代码里程碑的67项定向evaluator/cache/runtime测试、honest baseline分支smoke和统一cache dispatch smoke均通过；结构门
  无hard violation。该证据只说明运行面接通，不是Writer性能或fixed-decoder gate通过。
- 原两个decoder profile入口已收敛为唯一`train_functional_adapter_decoder.py`：直接优化完整PI0.5 flow response，保存
  task-equal phase cursor、system/held codes、optimizer与Python/NumPy/Torch RNG，可从阶段checkpoint精确续训；56/15 formal
  schedule及下游formal-authority门已经冻结。它只补齐正式训练责任，不声称non-held decoder结果已经通过。

当前train24非正式机制profile（不是模型选择或closed-loop证据）：

- 结果无关fold0以19 tasks拟合decoder、5 tasks冻结decoder后只拟合新code；五折将轮换，19/5不是永久丢弃任务；
- gauge-invariant `BA·probe`预热在380/250步把fit mean从`1.000`降到`0.447`、held code mean降到`0.805`，但其
  PI0.5完整flow初始loss仍约`0.999/1.008`，证明effective-update几何不能替代policy-functional目标；
- 完整50-token flow短profile仅给每个fit task 2步、held task 5步，独立demo40--49评测从`0.999→0.833`和
  `1.008→0.933`；18/19 fit与4/5 held优于identity，仍各有一个退化task，因此只支持“链路有可学习信号”，尚不通过
  fixed-decoder realizability gate；
- A40单卡峰值18.81 GB，38+25个实际更新约22秒，主要固定成本是policy加载与成对probe缓存。下一节点应扩大独立panel和
  task-equal更新次数，而不是扫rank、scale、seed或dtype。

当前train24 fold0 formal closed-loop结果：

- fixed decoder单checkpoint为`388/1200`，direct task experts为`658/1200`；严格配对是332 retained、56 gained、
  326 lost，Jaccard `.46499`；
- 19个decoder-fit tasks为`326/950`，对应direct `550/950`；5个decoder-held tasks为`62/250`，对应direct
  `108/250`。fit与held都只保留约六成expert aggregate，不是只在held code拟合处失效；
- 因此train24版明确不通过functional realizability gate，内部flow loss下降不能替代该结论。下一步不是扫小超参，
  而是按已冻结合同训练56/15 non-held meta expert family，再重新拟合和裁决固定decoder。

## Final external-review result

| arm | macro25 | macro50 | 25→50 retained/gained/lost | breadth@1 |
| --- | ---: | ---: | ---: | ---: |
| A Text+detach | 123 | 84 | 71 / 13 / 52 | 8→5 |
| B noText+detach | 104 | — | — | 6 |
| C noText+credit | 110 | 101 | 77 / 24 / 33 | 6→4 |
| F5 C+PCGrad | 107 | 96 | 82 / 14 / 25 | 6→4 |
| F3 A+frozen heads | 123 | 117 | 90 / 27 / 33 | 8→6 |

完整macro25视频面板（correct / same / wrong / shuffle / keep-first / reverse / no-video）：

- A：`123 / 125 / 81 / 122 / 131 / 90 / 48`；
- B：`104 / 101 / 65 / 83 / 90 / 96 / 47`；
- C：`110 / 111 / 54 / 91 / 93 / 69 / 47`；
- F5：`107 / 111 / 51 / 92 / 105 / 53 / 47`。

三个no-Text arm均显著优于no-video和wrong，说明Writer确实使用视频，不是language-only。C是唯一在aggregate上
同时显著优于wrong/shuffle/keep-first/reverse/no-video的arm，但收益高度集中Object、Long reverse反向，且
same-task correct-success retention只有87.27%。因此视频因果资格得到部分改善，方法未达absolute、
稳定、same-video robustness和跨suite高层Program的联合目标。

## Root-cause adjudication

1. **Fresh front-end detach是真实工程缺陷。** A/B在macro1/25的`patch_grounding`/
   `interaction_projection`均无gradient；C修macro1首次有credit。修复将correct-reverse margin从8提到41，
   但correct只104→110且继续漂移，所以它是视频方向资格的一个前端因素，不是absolute/stability首因。
2. **Text Meta-LoRA提供真实但混合的support。** 移除它使correct掉19，同时shuffle/keep-first各掉39/41、
   reverse反而升6；这不是纯language shortcut，也不是科学上干净的正机制。owner的no-Text边界继续有效。
3. **简单self-occupancy divergence未获支持。** lost rows没有出现预期的macro50-self-occupancy disagreement增大；
   validation expert不存在且held teacher action受信息墙禁止，动作正确性只能记为审计后不可判。
4. **FactorHead co-drift和reachability都是实证问题。** freeze使84升到117但仍丢33；修正wiring后的fixed-head
   free-Program仅307/1200，对照direct experts 658/1200，253 retained / 54 gained / 405 lost，未过90%门。旧659
   是未安装投影LoRA的无效结果。后继fixed functional decoder正面检验稳定功能坐标；若nonheld held-task gate仍失败，
   必须考虑架构性扩大或重参数化，不能以小扫掩盖。
5. **Cross-task conflict会改变换手，standard PCGrad不是解法。** 它将lost 33→25、churn 57→39，但gained
   24→14且有显著抑制，score更低、breadth仍收缩，并把keep-first margin压到2。Adam moment独立效应仍不可由本arm裁决。

当前最早未解接口被收窄为：固定输出坐标能否覆盖policy-effective directions、四条信息流能否为未见task预测这些
directions，以及shared objective/更新能否在同一checkpoint保留它们。本轮没有性能pass；当前登记的后继架构用
functional fingerprints + fixed decoder把前两项拆成独立gate。

## Remote-visible review map

- 本轮policy-native双时间架构候选与专家问题：
  `docs/policy_native_dual_time_program_compiler_review_20260821.md`；
- 原专家报告：`docs/external_review_20260818.md`；
- 113项claim ledger：`docs/external_review_claim_ledger_20260818.md`；
- 本轮面向专家的结果报告：`docs/external_review_followup_20260819.md`；
- 给新session复制的独立复核prompt：`docs/external_review_followup_prompt_20260819.md`；
- 证据索引与全部remote-safe JSON：`docs/evidence/external_review_20260818/README.md`；
- 持久结论与历史：`findings.md`、`docs/research_history.md`。

## Verification and cleanup

- 最近完整回归仍为`293 passed`；fixed-decoder正式训练入口另有8项聚焦测试通过；
- B/C/F5各7个视频面板均为400 rows，pairing mismatch全为0；全部tracked/forced evidence JSON可解析；
- 本轮只运行必要的聚焦回归：autocast-safe confidence objective、successor authority配置与detached frozen authority
  各1项通过；Writer profile成功后
  未重复大规模训练。已完成的direct/projected evaluator均退出且无遗留`ymdai` GPU进程；正式证据、唯一expert bank、
  decoder/projection与成功profile保留，失败profile临时目录已删除。
- projected formal实际使用commit `247e6a8`的SQLite `DELETE` journal；继承run contract中的旧`WAL`描述只是标签滞后，
  不改变rows、pairing或adapter。active evaluation config已更正为`sqlite_delete_full_sync_atomic_claim`，证据中显式记录
  该provenance。
- validation8 strict400、8-row occupancy capture及四task action/JVP分析的worker均已exit0，GPU显存已释放；对应三个
  detached formal worktree均在证据落盘后删除，当前`git worktree list`只保留canonical `main`。372 MiB成功trajectory是
  唯一phase follow-up输入而保留，不重复rollout；临时selection与旧冻结worktree未残留。
