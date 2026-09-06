# EMBER owner requirements

本文保存owner的长期目标、研究原则与协作要求。科学、数据、评测、资源和Git的具体合同以 [AGENTS.md](../AGENTS.md) 为准；
当前授权只看 [progress.md](../progress.md)，执行计划只看 [task_plan.md](../task_plan.md)。已经结束的讨论与旧实验不构成重新启动授权。

## 1. 科学精神与目标

人可以从他人或不同身体的教学视频中理解目标、条件和操作过程，再迁移到自己的身体及当前场景。EMBER探索把这种能力落实为
视频到策略参数的编译：从generic `lerobot/pi05_base`建立的冻结source policy出发，输入exact task language和一条或多条
同task、action-hidden、内部有序的正确视频，一次生成完整task-conditioned LoRA，随后从未见初始化闭环完成任务。
跨具身视频是科学动机；目前LIBERO实验本身不证明已经获得跨人类、机器人身体或视角的泛化。

- Writer初次生成的LoRA应立即有效；rollout期间不重复看teacher video，不做task-local优化、环境试错或第二阶段适配。
- 语言说明目标和关注对象；正确视频的动态过程必须带来相对language/static prior的必要条件增量。
- 输出是一套覆盖Action Expert全部38个目标的完整LoRA，联合生成A/B，无独立carrier、任务字典或第二套执行adapter。
  rank16是已对齐候选的首选容量，不把rank、memory tokens、FactorHeads或某种decoder当成研究目标。
- 部署输入不得包含teacher actions、state/proprio、reward、terminal、task ID、filename、pose、hidden normalization或policy outcome。
  执行policy读取自己的当前观测和state；不能把执行输入与teacher-video信息墙混淆。
- 一次Writer调用内部允许固定、只读、多阶段读取与重放同一组授权视频或native activations；这不是task-local训练。
- 冻结source无可训练参数；共享observer适配只改变读取侧。若将共享prior用于执行，必须与条件残差合并为唯一完整LoRA，计入总rank预算。

## 2. 架构与推导原则

- Action Expert的原生动作生成知识应参与视频理解。逐帧Gemma图文语义、Action Expert响应、跨帧过程理解和参数生成各自承担明确职责。
  捕获full horizon、存在梯度、attention或模块名称，都不能单独证明这一科学机制已经兑现。
- teacher-video time、relative action horizon、flow time、layer depth分别处理；horizon不是事件标签，计算深度不是任务阶段。
  frame stride固定5，完整50-horizon在有实际任务条件与跨帧消费的learned read之前保留，不能恢复coarse或horizon mean。
- 教学过程按时间向前理解：后帧读取前帧，早期表示不依赖未来图像或由全视频长度产生的位置编码。最终参数生成可以读取整段已编码视频。
- 每条视频独立保序编码；只在集合阶段置换不变地合并证据。不得平均frames、raw features或最终LoRAs，不挑最好video。
  声称dynamic K就必须真实训练对应cardinalities，不能重复同一条视频凑K。one-shot/few-shot设定由真实能力决定，不故意削弱强方案。
- 观察侧Meta-LoRA应有明确输入域与学习职责，必须保留其真实梯度以及cache有效性。已对齐设计采用Action Expert共享Meta适配，
  vision/Gemma保持冻结；其具体rank、投影集合、probe和readout以登记设计为准，不把历史默认当作永久规定。
- 显式读X/Y与把因子限制在X/Y的span是两个独立选择。G1证明过局部native-factor容量，不强迫后继复刻signed pooling；
  原生状态或压缩的过程表示也不自动等同于原始算子X/Y。观察侧与执行侧激活坐标必须区分。
- 保持少数职责清楚、可重复扩展的标准attention/MLP模块。不要沿用无用途的双probe、重复读取或旁路，也不要连续叠加summary、
  covariance、whitening、transport、anchor、gate或校准链。保留与删除都需要说明当前用途和行为代价。
- owner只评论局部时，保留已对齐且未被否定的部分；不把局部疑问当作推翻整图的指令。先说明完整数据流水线，再讨论局部模块。
- 数学推导从需求、少量符号和直观例子逐步展开；区分推导结论、归纳偏置、实现默认和待检验假设。结构合理不等于性能得到保证。

## 3. 证据与推进判断

- 唯一正式目标是validation8 strict single-checkpoint paired correct严格 >145/400，同时满足相邻稳定、低churn、高breadth、
  四suite非零、Goal/Long贡献、same-task不同视频鲁棒性及最终视频因果controls。正式选择不使用80-row screen、checkpoint union或融合。
- 闭环绝对性能优先。functional loss、reconstruction、norm/rank/cosine、内部margin和surrogate仅用于定位；不能用漂亮数值接受明显更差行为。
- 历史v5.2/v6的强闭环能力、G1容量、G2动态、后续局部正结果及失败边界都要保留。不同checkpoint/配方的优点不能拼成一个不存在的强结果。
- 使用正确视频与同task不同episode的action queries训练。额外non-held meta tasks须审计固定validation/test及重复specification排除，
  保留allowlist/provenance；更多同task视频不等于更多独立meta-task映射。不得制造人工process数据或新仿真任务来绕开当前问题。
- validation/test不得产生梯度。shuffled/reversed仅在selected checkpoint选定并冻结后测试，不进入训练、loss、Gate、checkpoint选择或架构修改。
  no-video/language、static端点、wrong-video等资格或诊断使用时须事先明确用途，不能悄悄把最终controls变成架构搜索信号。
- 主训练必须保留同拓扑fully-random fresh端到端候选；G1--G3的阶段冻结只服务机制验证，不构成Final强制课程。
- 先用有信息量的短学习与闭环证据判断投入。未证明基础行为前不默认启动约10小时长训练；接近强基线或目标后及时做strict400，
  好趋势继续训练到足以判断相邻稳定，明确坏结果不靠无限续训或无依据的seed/LR/rank/scale/width小扫挽救。
- 每轮记录per-task、per-suite、breadth、retained/gained/lost、churn、相邻success-set重合和实际样本曝光；训练步数本身不足以比较配方。
- 负结果只淘汰真正测试的组合。先区分工程合同错误、有效科学non-pass与证据不足；不要把可疑现象或一次梯度cosine称作根因。
- 诊断应能区分竞争解释并定位最早失效接口。先查历史同类尝试、原始评审及后续修正，再做最小有信息量的干预；
  明确输入变化、旧证据排除什么、新证据如何改变判断。新证据支持模块职责替换时可以实质重构，避免围绕同一接口原地打补丁。
- 不人为规定总工期、修正次数、版本数或总轮数。停止无信息重复，同时允许有新机制证据的合理深入。

## 4. 授权与自主协作

- 事前讨论自主接管不等于立即启动。owner仍在对齐、暂停或安排交接时，按明确授权做相应工作；新session须先理解当前状态，
  得到owner明确同意后才能正式推进。旧goal、旧active design或未勾选清单不能覆盖当前暂停。
- 获准接管后，在既有目标、信息墙与资源合同内，实验设计、实现、分析、相关修复、吞吐优化和证据支持的模块重构由接管者连续完成，
  无需逐项询问。不因一个侧面问题、单点好坏或常规技术检查停止已授权流程。
- 改变科学目标或信息墙、引入未授权数据/资源、无法裁决且显著改变投入方向的路线歧义、删除所有权不明或唯一资产时，
  带具体事实和推荐选择回到owner。不得创造额外审批流程。
- 主agent主动识别真正能缩短总耗时的独立工作，直接派发必要subagents；例如设计记录、源码审计、存储审计可并行。
  明确职责、停止条件和写入范围，独立worktree隔离写入，主agent负责集成与比例验证；不用形式化并行、重复方案或常规审查浪费token。
- 不把再次完整专家审查作为每轮开工前置。已有明确可检验设计时先获得具体实现与证据；需要专家时给有针对性的问题和新事实。
  未经owner当次明确授权，不向外部专家发送消息，只提供可复制prompt。
- 只有owner明确要求时创建或设置goal，不因任务复杂或跨session自行启用goal机制。

## 5. GPU效率、工程与资产生命周期

- 从算法设计阶段就考虑GPU：批量张量、高效attention、明确布局，减少逐项Python循环、CPU/GPU往返和重复大算子。
  同时审视训练、functional forward、物化与闭环评测；按真实LoRA/s、samples/s、step墙钟、SM/util与显存峰值衡量。
- 使用1--6张当下适合的同节点GPU，保持全局task group、role权重和optimizer cadence；不固定2卡、3+3角色或6-task batch。
  exact-resume仍锁原world topology。两节点live检查、NUMA、deferred NCCL和NCCL_P2P_DISABLE=1按AGENTS执行。
- 不以最低显存为目标，不人为设置35GiB等统一上限，也不以占满显存冒充效率。优先空闲设备；必要共驻须有真实吞吐收益、足够峰值余量且不干扰他人。
  节点/index不永久代表某块好坏GPU；每次按UUID/serial和现场证据判断。EMBER同时占用总量不超过6张物理卡。
- 真实长视频profile外推完整训练与评测成本；少量更新因重复大算子消耗几十分钟或数小时的明显失衡应先修正。
  自设吞吐阈值不合适时可以修订，不可让它取代科学判断或靠堆卡掩盖算法问题。
- 保持完整视频/horizon、信息墙、梯度语义、任务权重、checkpoint与resume；接受正常BF16/TF32和高效kernel的低位差异。
  不新增防御性hash sidecars、逐tensor一致性扫描或无意义测试。只做与实际声明相称的验证。
- canonical资产复用，不复制数据、模型、环境与大缓存。大增长前检查strg01上的独立user quota，不能只看共享df空间。
- 退役代码、脚本、配置和设计通过Git与有索引的正式证据保留，退出活动树。可重建临时缓存、重复物化结果在验证生命周期后删除；
  唯一checkpoint、原始数据、正式raw rows/metrics/manifest及所有权不清内容保留。
- main是集成目标；完成验证后及时集成、推送并清理已合并task worktrees。新架构只有一套canonical实现，不保留平行fallback。
- 有效实验已经可运行时，非必要重构、文档或清理不阻塞科学节点；独立工作利用等待期完成。收到新实验结果及时回到科学推进。

## 6. 沟通和交接

- 默认实用中文，先直接回答具体问题，再给证据和边界。owner主要语音输入，应主动修正明显同音词、断句和术语识别错误。
- 讨论像共同推导：不把回答写成教科书岔路，不反复使用“不是……而是……”式对立话术，不把未接受的建议说成owner要求。
- 持久文档职责固定：concept讲科学精神，设计文档讲推导与方法，findings讲跨轮结论，research_history讲分层历史与证据，
  task_plan讲下一阶段，progress讲授权与现场；AGENTS只写稳定合同。
- HANDOFF.md只是消费后删除的临时入口，不能独占长期要求、架构决定、历史结论或执行计划。跨session前正式文档必须完整，
  新session应能自主恢复理解，不要求owner再次解释整段历史。
