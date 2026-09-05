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
- Action Expert的50个relative horizon positions必须完整保留到task/relation-conditioned learned read；不得用final-layer
  horizon mean、coarse response或任何等价的无条件平滑替代主路径。既有coarse代码、checkpoint与结果只作历史审计，不得再用于
  active训练、模型选择、初始化或部署。
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

## 4. 方法方向与架构边界

主方向是充分利用冻结PI0.5 Action Expert的原生动作时序知识形成视频过程理解，并与整套LoRA生成共同学习。
Gemma提供逐帧图文语义；其特征不被假定跨帧恒定，Action Expert逐帧响应也不自动构成视频理解或正确动作真值。
过程状态与整策略queries交互是owner认可的候选方向，具体token、attention布局、factor readout和rank12+4分配由证据裁决。
当前执行阶段与active design只在`task_plan.md`/`progress.md`登记；专家原文见
`docs/expert_review_20260905_full_history_joint_process_policy_writer.md`。历史v5.2/v6、G1/G2与后续结果均须保留其正证据和适用边界，
不得把局部non-pass外推为整条路线无效，也不得把保留模块职责宣称为已经保留行为。继任路线须遵守：

1. 冻结PI0.5内部Action Expert信息是视频理解的核心动态证据来源；teacher-frame time、50-step action horizon、flow time、layer depth
   与probe必须作为不同轴处理。full 50-step horizon是唯一获准表示，不得恢复coarse、horizon mean或等价抹平；
2. G1已证明真实native X/Y、signed pooling和rank4 task-local容量，G2已证明ordered PI0.5 response包含视频动态；这些是应吸收的正证据，
   但不是强迫后继复制v4具体token、head或rank分配的架构公理；
3. language与静态context不能独立写出有效mobile residual，video dynamic evidence必须是必要Value路径；
4. correct相对wrong、shuffle和reverse的优势应从正确视频监督自发产生；negative controls不进入训练loss或架构修正；
5. learned主干必须由少数职责清楚、可复制扩展的标准attention/MLP类模块组成。扩大模型应主要复制同构层或统一改变width/heads，
   不得恢复Natural Program到summary、covariance、whitening、transport、anchor、family scalar gate或其它连续专用数学链；
6. Final候选仍须考虑同拓扑fully-random fresh端到端训练，并由closed-loop而非内部loss选择。

owner于2026-09-04及2026-09-05进一步明确：只要有可复核证据并经过深入分析，可以实质重构Writer；但结构自由度不能再次演化为
连续数学补丁，整体结构必须优雅并能通过复制同一种层扩展。
若同一接口反复non-pass，应替换其责任模块，而不是继续在前后叠加summary、solve、recenter、whitening、transport、gate或等价专用
变换。后继主干应保持少数职责清楚的learned模块并优先采用同构attention/MLP block；手工运算只保留信息墙、
轴/mask、必要的数值归一化、完整候选归约和唯一LoRA物化等明确科学边界。

G2已经通过的Natural Program仍是ordered event、初始化与机制证据，但其固定
`P_lang/P_scene/P_process/rho/tau/sigma` tuple不再是deployment下游的唯一硬瓶颈。PNBTT、旧
`summary -> family-scalar gate -> shared event-additive anchor`、EBSRI与其它已裁决G3实现只作历史复现和kernel复用，不构成
active fallback。

首轮Action Meta-LoRA关闭，优先解决如何读取Action Expert响应。observer侧共享适配是有限可选项：若在可学习的同一Writer/读出下
出现可定位的教学输入域不足，可按matched证据审视，不把“必须先有base Writer闭环收益”当成不可讨论的科学公理。
启用时须明确observer坐标与执行source/raw native bank的关系、训练/冻结阶段及cache有效性；不得默认它能解决跨视角或跨具身泛化。
执行端若使用Meta，必须计入唯一完整LoRA的rank预算；不得另挂第二adapter。

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
- 性能低于或接近baseline时，不得随意找一个可疑现象命名为根因并立即修补。必须认真审视整体图、历史等价尝试与竞争解释；必要时
  自主开展更多分析实验。只有直接工程合同证据或能区分竞争解释的实际干预支持时，才在相应范围使用“根因”结论。
- 每个主要修改须说明最近旧尝试、旧证据实际排除什么、本次改变的因果变量、预期分支与行为代价。attention、rank、cosine、梯度或
  loss变化只作定位；不能为解释得更清楚而默认接受更弱闭环。负结果不自动授权无证据小修，也不要求每轮回到owner。
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
- 在既有科学与资源合同内，实验设计、分析、实施、相关修复、吞吐优化和证据支持的模块重构由接管者自主连续推进，无需逐项询问。
  只有改变目标或信息墙、引入新数据或资源权限，以及证据无法裁决且显著改变投入方向的路线歧义，才带具体结果与推荐选择回到owner。
  性能跃升继续完成必要相邻验证并报告，不因单点好坏中断授权流程。外部联系仍按第8节取得当次明确授权。

## 7. GPU、仓库与文档

- 每次GPU launch前同时live检查gpu01/gpu02；单个job只用一个节点，最多6张真正提高吞吐的A40。
- gpu01和gpu02都属于可用计算池；不存在按节点名或逻辑index永久禁用的设备。任何临时prohibited状态只能按当时明确的UUID/serial和
  owner指令继承，并在每次launch前用live身份、进程、显存与utilization重新裁决。节点暂时离线或重启不代表长期禁用。
- 正式训练实现不得把world size固定为2；在保持全局task group、role权重、optimizer cadence和科学口径不变的前提下，按launch时
  实际可用卡数在1--6张之间弹性分片。exact-resume仍锁定该run启动时的world topology。
- `meta`/`target`是否同时参与、两者采样比例以及每个optimizer step包含多少task均由具体实验的数据与采样设计决定；owner没有规定
  固定`3+3`、固定6-task batch或必须同时包含两类。执行优化必须接受任意已配置task group，只改变其设备放置和流水，不得反向把
  当前资格实验的采样选择固化为长期科学合同。
- 训练、functional forward、推理和closed-loop评测的吞吐优化都同时约束卡数与每卡有效利用率：即使只用单卡，也应按真实LoRA/s、
  samples/s、step wall time、计算段SM/UTL、memory UTL与显存峰值调整microbatch、frame chunk、任务分片和数据供给。不能用空tensor、
  dummy进程或单纯占满显存冒充利用率；若SM已持续饱和，未占满
  48GB本身不构成低效。反之也不得自设`35GiB`或其它固定显存上限：最长真实样本、allocator波动和共驻进程仍有安全余量且不OOM时，
  可以使用更高显存；最终选择以真实吞吐、持续利用率和稳定余量为准。
- 实际墙钟成本必须与训练/评测规模相称。formal launch前要用真实condition/step profile外推完整训练和Gate评测；若一个只有少量更新的
  资格实验仍需几十分钟或数小时，且瓶颈来自每condition重复的大算子，就应先判定吞吐资格non-pass并修正执行结构，不能靠堆更多GPU、
  缩减必要评测或要求owner接受原始吞吐来掩盖。
- 在新架构还没有用真实closed-loop证据显示超过已知carrier/直接可比基线之前，不得默认启动约10小时的大规模训练。
  应先在不改变核心函数类、必要时序信息和闭环口径的前提下，缩小任务数据和更新数做最短有信息量资格实验；只有架构有效后才恢复扩展成本。
- EMBER并发总量不超过6张物理卡；只要总量未达上限、增加设备确实提高吞吐且实时余量安全，就应使用合适的空闲卡。可与低显存、
  低util进程安全共驻，但不得抢占、kill或reset。
- 调度应优先使用满足峰值余量的真正空闲卡；只有空闲卡不合适或并行布局确有收益时才与他人低显存、低util进程共驻，不能在有等价空闲卡
  时无故挤到他人设备。允许共驻不等于降低单卡利用率要求，也不允许干扰对方任务。
- gpu01历史上曾标记prohibited的设备只能按当时UUID/serial身份继承，不能把任何节点重启后的逻辑index 0机械等同于旧设备。当前没有
  按逻辑index永久禁止的GPU记录；身份、枚举映射、健康、进程、显存与utilization每次launch都必须live确认。
- 正式训练遵守storage quota、clean pushed commit和frozen worktree合同；探索实验不做冗余流程。
- canonical集成目标是`main`。只有需要隔离或并发写入时创建`codex/*`分支和worktree，验证后尽快合并、推送并清理。
- 不在活动树保留退役实现、平行fallback、过时配置、重复文档或临时结果；历史由Git、formal artifacts和一份精简历史记录保存。
- 代码、文档、branch、worktree和运行产物应在每个阶段及时整理，不等到几十版后集中失控。
- 训练、评测或其它长GPU阶段等待期间，只有没有推进相关的实质性实现、分析或下一节点准备可做时，才增量清理已确认的退役代码、
  陈旧文档、temporary artifacts和workspace；训练或评测结果一到立即停止清理并继续科学推进。清理不得占用关键GPU资源、干扰
  运行进程或反过来延迟下一科学节点。

## 8. 沟通与交接

- 未经owner当次明确允许，绝不能直接向外部专家发送消息；只能提供可复制prompt给owner。
- 给专家的prompt只补充他未知的新事实、结果与问题，不重复整段既有对话，也不人为限制专家的核心判断。
- 只有owner明确要求时才创建或设置goal；不得因任务复杂、跨session或自主推进而自行调用goal机制。
- owner询问具体问题时先直接回答该问题，不擅自扩成新方案、审批请求或外部沟通。
- owner主要语音输入；明显同音词或断句错误要按EMBER上下文理解。
- `HANDOFF.md`只能是消费后删除的临时索引，不得成为任何长期要求、架构决定、科学结论或执行计划的唯一载体。稳定要求进入
  本文件，架构进入active design，跨轮结论进入`findings.md`/`research_history.md`，计划与即时状态进入`task_plan.md`/`progress.md`。
- 跨session前，上述持久文件必须完整；新session不应要求owner重新解释项目、专家讨论或GPU约束。
