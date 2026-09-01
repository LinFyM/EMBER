# EMBER owner requirements

本文只记录长期有效的目标、边界与协作要求。动态进度见`progress.md`，执行计划见`task_plan.md`，历史事实见
`docs/research_history.md`。

## 1. 最终目标

EMBER必须从generic `lerobot/pi05_base`建立的冻结PI0.5-LIBERO source policy出发，只接收：

- 目标task的exact language；
- 一条或多条同task、action-hidden、内部有序的正确教学视频。

Writer在rollout前运行一次，直接生成一套覆盖Action Expert全部38个目标层的完整task-conditioned LoRA。该次调用内部允许对同一组
授权视频及其冻结policy native activations做固定、只读的多阶段流式读取与重放；这不等于rollout期间重复调用Writer，也不构成
task-local适配。冻结policy加载这
一套LoRA后，应从未见初始化闭环完成相同或相近场景中的任务。部署时不得再次观看视频，也不得进行环境交互、task-local
优化或第二阶段适配。

语言负责说明目标与关注对象，视频必须提供不可被语言或静态端点取代的动态证据。模型应理解“在什么条件下、按什么过程
完成任务”，而不是记住task ID、文件名、场景模板或少量训练任务特征。

## 2. 不可改变的部署合同

- 输入：exact language + `K`条action-hidden ordered videos。
- 输出：唯一一套完整rank16 LoRA。首版canonical采用有解析容量证据的frozen rank12 carrier + native-factor mobile rank4 residual，
  但这不是不可改变的架构公理，也不代表专家证明了12+4全局最优。若native bank可表达、rank4 free-code已经收敛、剩余误差由
  rank ceiling造成，且同构full-rank16 oracle显著通过，则按证据重新分配task/carrier rank；不能因历史惯性或便利随意改变。
- source PI0.5完全冻结；默认只修改Action Expert，不让Writer改变Gemma权重。
- 每条视频独立保序编码，跨视频只做置换不变聚合；不得平均frames、raw features或最终LoRA。
- 每个condition只生成一套LoRA；不得挑video、融合checkpoint、部署第二adapter或并行expert。
- Writer只在rollout前运行一次；一次调用内部可有固定的read-only native-bank统计与重放子阶段；zero-interaction分数不混入
  生成后的task-local RL。
- deployment Writer不得读取teacher action、state/proprio、reward、terminal、task ID、filename、pose或policy outcome。

训练期可在授权的non-held tasks上使用actions、privileged task experts、simulator reward和occupancy学习共享机制，但这些信息
不得成为deployment输入、held dictionary或task-ID route。
G3的native-feasible LoRA teacher只用于验证shared compiler接口；G4/Final训练合同不得预设每个任务存在目标LoRA。正式联合训练可直接
使用授权fit/meta tasks的teacher actions、functional或on-policy闭环信号，具体最小监督集合由机制与closed-loop证据决定。
G1--G3的冻结/分段只为逐接口验证，不是Final的强制训练课程。Final正式候选既可从已通过Gate的Program/compiler参数初始化，也必须
保留整套Writer完全随机初始化、从头直接端到端联合优化的fresh选项；两者使用fresh optimizer/scheduler和同一数据、评测与信息墙，
最终由稳定closed-loop表现决定。若随机初始化能够通过整体梯度下降形成内部功能分化，就不应人为重演G1--G3的分段训练。

## 3. 数据边界

- 固定benchmark为LIBERO Spatial/Object/Goal/Long 40 tasks。
- 固定development split为`configs/libero_24_8_8_v1/`的24 train / 8 validation / 8 test。
- source corpus为LIBERO-90排除与目标40重合的19项后剩余71 tasks，每task 50条成功episode。
- 后继meta-training可以使用train24和经精确语义审计、排除validation/test及重复项的其它现成LIBERO tasks。
- 当前不制作人工任务、人工process数据集、人工controller trajectory或额外仿真场景；推进必须直指ECP核心。
- 不得使用读过目标40 actions的`pi05_libero`。
- video和action query同task但跨episode采样，阻断逐帧轨迹复制。
- validation/test actions或reward不得产生梯度；Test默认留到最终方法冻结。

held5只是train24内部的leave-task-out机制门，用于在不消费validation设计信号的前提下快速检查共享映射。正式开发选择必须回到
固定validation8；方法确定后才按合同从fresh使用32 source tasks训练并评测test8。

## 4. 当前方法方向

经过2026-09-02全局专家复核与owner正式裁决，当前唯一active方向是ECP Native-Factor Compiler中的
**Program-Conditioned Native-Bank Tangent Transport（PNBTT）**。完整方法合同见
`docs/program_conditioned_native_bank_tangent_transport_design.md`；专家原文见
`docs/expert_review_20260902_global_route_reassessment.md`。它不是GOMQ、PECS、历史v24或旧G3的改名。其核心是：

1. 保留冻结PI0.5原生language、patch和Action Expert内部时序结构形成的owner-specific、ordered Natural Program；
2. Program只能产生低维query；当前真实X/Y candidate产生低维key，最终value始终是原始真实native candidate；
3. B0只流式建立当前bank的key-space mean/covariance/whitening，B1在同一bank上执行唯一antithetic exact signed selection；
   不再产生base primal，也不再使用`base + bounded correction`、family scalar gate、shared/free native anchor或summary token；
4. 每条视频独立保序编码，跨视频以固定等质量`1/K`形成一个联合candidate measure；不得学习video reliability、平均raw
   features或先生成多套LoRA再平均；
5. 首版继续把唯一native-bank rank4 residual与frozen rank12 carrier严格拼成一套38-target rank16；rank分配只由同构容量证据重开；
6. privileged policy/effect evidence只作set-valued functional critic，不产生神经`q_pi`或部署latent；
7. E1 free-query transport与E2真实frozen Natural Program资格通过后，立即进入shared/whole-Writer训练；Final必须matched比较
   component-init与同拓扑fully-random fresh候选。

当前旧G3的`summary -> family-scalar gate -> shared event-additive anchor`函数类正式退役；旧sealed configs和代码只作历史复现与
选择性kernel复用，不构成active fallback。若E1通过而E2在排除bank chart/优化问题后系统性失败，按专家条件转入直接由language与
ordered native-bank tokens产生signed measure的B路线；只有matched whole-Writer两臂都不能产生稳定闭环增量，才讨论根本停止。

唯一Program schema为`P_lang[38,128]`、`P_scene[38,128]`、`P_process[8,38,128]`、`rho[8]`、`tau[8,2]`和
`sigma[8,38,128]`。最大`E=8`固定，slot激活数量与视频段落分配动态学习；跨视频只在保序event alignment后聚合。

首版不启用Action Meta-LoRA。只有base Writer已有明确闭环增量后才做matched controls；Stage 0和compiler冻结，只有出现明确
净收益且不损害breadth/retention才加入并永久冻结，否则保持关闭。

shuffled/reversed不进入训练、loss、checkpoint选择、G1--G5 Gate或架构修正依据。它们只在最终selected
checkpoint已选定并冻结后作为严格配对的时序特异性测试；正确视频应稳定优于打乱与倒序输入。full video
还必须优于language/no-video、scene/first+final和wrong-video controls。

## 5. 成功标准

- 唯一正式性能目标线是validation8 strict paired correct严格`>145/400`。
- 该分数必须由相邻single checkpoints、低churn、高breadth、四个suite均非零、Goal/Long真实贡献、same-task
  不同视频鲁棒性和视频因果controls共同证明，不能用偶然峰值通过。
- full video必须有必要条件增量，并在多数任务上形成收益；same-task其它视频应保持高retention。
- shuffled/reversed最终表现应揭示真实时序特异性，而不是仅让内部latent距离变大。
- closed-loop absolute表现优先；loss、reconstruction、LoRA norm/cosine、hidden margin和surrogate只用于定位。

一次实验必须报告per-task、per-suite、breadth、retained/gained/lost、churn和相邻checkpoint success-set重合。明确坏结果不靠
小幅seed/LR/rank/scale或超长续训挽救；负结果只淘汰实际检验的组合，不外推为整个EMBER目标失败。

## 6. 推进方式

- 先理解因果链和最早失效接口，再实现；不得用连续版本号替代思考。
- G2、G3、G4及后续阶段出现显著non-pass时，先冻结该轮结果与controls，区分工程合同错误和真实科学失败，再用可证伪的
  read-only消融、decodability、gradient或closed-loop probe定位最早失效接口。只有新的机制证据支持时才修改对应接口；不得把
  盲目迭代架构、微调超参或内部loss下降包装成根因分析。
- 每个阶段都要说明：输入数据、训练模块、冻结模块、输出、验证问题、通过条件和失败后的分支。
- 优先做能改变路线判断的实验；不钻无关紧要的shape、低位浮点误差、防御性代码或冗余测试。
- 不新增MD5/SHA-256 sidecar和大规模逐tensor校验；只保留信息墙、shape、finite、OOM、pairing、asset、checkpoint与resume所需
  的直接检查。
- 复用已经训练出的可用资产，避免重复长训练；profile和smoke只做最小必要验证。
- 不人为给各阶段规定工期、修正次数、版本数量或总轮数。Gate用于判断证据和下一接口，不是日历或尝试次数上限；有新机制证据
  支持时可以继续修正，不能因为预设次数耗尽而停止，也不能用无新信息的seed/LR/width小扫冒充修正。
- 在证据质量不下降的前提下尽可能快地推进，积极复用资产、并行独立工作并提高代码和GPU吞吐；进展顺利时应力争数天内完成
  整体架构实现并推进关键Gate，不能借“分阶段”把工作人为拉长。
- 一旦canonical代码通过最小真实forward/gradient/materialization smoke并具备有效科学裁决条件，应立即启动有信息量的实验；文档整理、
  通用重构、非必要合同、清理和补充分析不得阻塞科学结果，能在训练或评测等待期间并行完成的工作应移到等待期间。
- 自行提出的throughput阈值只用于发现执行结构是否明显失衡，不是科学authority；若阈值与真实工作量不匹配，应直接修订或删除，不能让
  不合理的自设Gate阻塞实验。反之，少量更新却需要几十分钟或数小时的明显失衡仍必须先优化，不能要求owner接受原始吞吐。
- subagent只在存在可独立、并行且能显著缩短关键路径的实现、审计或评测工作时使用；不为形式并行，也不让多代理协调反而拖慢主结果。
- 遇到困难先回看专家原始意见与修正，检查执行是否偏移，再决定是否实验或咨询。
- 专家意见是设计约束与启发：不能为了速度随意丢弃，也不能不经理解机械照搬。
- 只有性能显著跃升、路线存在实质歧义或需要新增权限时，在关键节点暂停和owner讨论；不频繁汇报。

## 7. GPU、仓库与文档

- 每次GPU launch前同时live检查gpu01/gpu02；单个job只用一个节点，最多6张真正提高吞吐的A40。
- gpu01和gpu02都属于可用计算池；不存在按节点名或逻辑index永久禁用的设备。任何临时prohibited状态只能按当时明确的UUID/serial和
  owner指令继承，并在每次launch前用live身份、进程、显存与utilization重新裁决。节点暂时离线或重启不代表长期禁用。
- 正式训练实现不得把world size固定为2；在保持全局task group、role权重、optimizer cadence和科学口径不变的前提下，按launch时
  实际可用卡数在1--6张之间弹性分片。exact-resume仍锁定该run启动时的world topology。
- 吞吐优化同时约束卡数与每卡有效利用率：即使只用单卡，也应按真实LoRA/s、step wall time、计算段SM/UTL、memory UTL与显存峰值
  调整microbatch、frame chunk、任务分片和数据供给。不能用空tensor、dummy进程或单纯占满显存冒充利用率；若SM已持续饱和，未占满
  48GB本身不构成低效。反之也不得自设`35GiB`或其它固定显存上限：最长真实样本、allocator波动和共驻进程仍有安全余量且不OOM时，
  可以使用更高显存；最终选择以真实吞吐、持续利用率和稳定余量为准。
- 实际墙钟成本必须与训练/评测规模相称。formal launch前要用真实condition/step profile外推完整训练和Gate评测；若一个只有少量更新的
  资格实验仍需几十分钟或数小时，且瓶颈来自每condition重复的大算子，就应先判定吞吐资格non-pass并修正执行结构，不能靠堆更多GPU、
  缩减必要评测或要求owner接受原始吞吐来掩盖。
- EMBER并发总量通常不超过6张；只有大量空闲时最多8张。可与低显存、低util进程安全共驻，但不得抢占、kill或reset。
- 调度应优先使用满足峰值余量的真正空闲卡；只有空闲卡不合适或并行布局确有收益时才与他人低显存、低util进程共驻，不能在有等价空闲卡
  时无故挤到他人设备。允许共驻不等于降低单卡利用率要求，也不允许干扰对方任务。
- gpu01历史上曾标记prohibited的设备只能按当时UUID/serial身份继承，不能把任何节点重启后的逻辑index 0机械等同于旧设备。当前没有
  按逻辑index永久禁止的GPU记录；身份、枚举映射、健康、进程、显存与utilization每次launch都必须live确认。
- 正式训练遵守storage quota、clean pushed commit和frozen worktree合同；探索实验不做冗余流程。
- canonical集成目标是`main`。只有需要隔离或并发写入时创建`codex/*`分支和worktree，验证后尽快合并、推送并清理。
- 不在活动树保留退役实现、平行fallback、过时配置、重复文档或临时结果；历史由Git、formal artifacts和一份精简历史记录保存。
- 代码、文档、branch、worktree和运行产物应在每个阶段及时整理，不等到几十版后集中失控。

## 8. 沟通与交接

- 未经owner当次明确允许，绝不能直接向外部专家发送消息；只能提供可复制prompt给owner。
- 给专家的prompt只补充他未知的新事实、结果与问题，不重复整段既有对话，也不人为限制专家的核心判断。
- 只有owner明确要求时才创建或设置goal；不得因任务复杂、跨session或自主推进而自行调用goal机制。
- owner询问具体问题时先直接回答该问题，不擅自扩成新方案、审批请求或外部沟通。
- owner主要语音输入；明显同音词或断句错误要按EMBER上下文理解。
- `HANDOFF.md`只能是消费后删除的临时索引，不得成为任何长期要求、架构决定、科学结论或执行计划的唯一载体。稳定要求进入
  本文件，架构进入active design，跨轮结论进入`findings.md`/`research_history.md`，计划与即时状态进入`task_plan.md`/`progress.md`。
- 跨session前，上述持久文件必须完整；新session不应要求owner重新解释项目、专家讨论或GPU约束。
