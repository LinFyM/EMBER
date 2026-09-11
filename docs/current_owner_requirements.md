# EMBER owner requirements

本文保存owner的长期目标、研究原则与协作要求。科学、数据、评测、资源和Git的具体合同以 [AGENTS.md](../AGENTS.md) 为准；
当前授权只看 [progress.md](../progress.md)，执行计划只看 [task_plan.md](../task_plan.md)。已经结束的讨论与旧实验不构成重新启动授权。

## 当前阶段的优先级：视频特异性与综合理论重构（Owner 2026-09-11）

Owner最新选择先专注恢复有益的视频特异性：正确教学视频中的任务内容与内部顺序应使生成的LoRA具有真实执行价值，
错任务或破坏顺序的输入不应得到同样的教学收益。暂不强制绝对性能；下文>145/400、SFT与source能力要求保留为项目长期目标
或参照，不作为本次理论咨询的额外完成门槛。仍需区分正确条件获得收益与仅让错误条件退化；具体机制验收口径尚待推导、登记。

下一步先深入挖掘全部既有架构、训练和干预证据，提出同时解释正例与负例的机制理论，再从原始输入、表示、LoRA参数作用一路
推导到训练与验证。不能把更多实验当作开始理论工作的前置，也不能先试出结果再补无法被否定的解释。

明确不直接回到v5.2，也不预设以v5.2为底座继续改进。v5.2的绝对性能与稳定性不达标，但其普通监督下的视频依赖正例是必须解释
的重要证据；后继尚无同时覆盖能力、稳定性与视频特异性的整体胜出方案。v6/GOMQ等更高单项成绩和其它局部正证据同样保留，
不把“未整体超越”写成“所有后继每项指标都更差”。应结合全部历史重新设计，也不预设保留当前Horizon的全部组件。

Owner随后明确授权“开始推进”，给予足够自由度并要求高效率。本轮已选定
[Video Functional Writer](video_functional_writer_design.md)：时间×任务token有序表示＋执行query条件化功能读出，
真实LoRA FM与辅助真实FM共同训练视频表示，只有Compiler/native D接受归一化蒸馏。
下文纯FM课程与Horizon具体结构属于此前选定方法，本轮由新active design替换；输入信息墙、完整H、冻结source、
唯一完整LoRA与部署零交互等科学边界保持。允许实现、profile、formal学习、必要对照及有依据的修正，
不自动恢复95-task或此前未完成实验。

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
- 教学视频在rollout前完整可用，必须保留视频内部顺序和时间方向；具体可见范围由Owner选择和登记设计定义。
  当前选定方法使用过去四帧局部读取及过去单向长程，所有组和回写遵守视频前缀依赖；完整H内部仍可双向交互。
  不因完整视频可用或专家原文推荐而恢复双向长程。计算上的前缀因果性与视频对行为的必要性不同，后者由冻结后的controls裁决。
- 每条视频独立保序编码；只在集合阶段置换不变地合并证据。不得平均frames、raw features或最终LoRAs，不挑最好video。
  声称dynamic K就必须真实训练对应cardinalities，不能重复同一条视频凑K。one-shot/few-shot设定由真实能力决定，不故意削弱强方案。
- 观察侧Meta-LoRA应有明确输入域与学习职责，必须保留其真实梯度以及cache有效性。已对齐设计采用Action Expert共享Meta适配，
  vision/Gemma保持冻结；其具体rank、投影集合、probe和readout以登记设计为准，不把历史默认当作永久规定。
- 显式读X/Y与把因子限制在X/Y的span是两个独立选择。G1证明过局部native-factor容量，不强迫后继复刻signed pooling；
  原生状态或压缩的过程表示也不自动等同于原始算子X/Y。观察侧与执行侧激活坐标必须区分。
- 保持少数职责清楚、可重复扩展的标准attention/MLP及短序列递推模块；有序GRU不是跨rollout记忆。
  不把保留18层响应当作必要原则，当前选定接口直接读取最终动作投影前的完整H。不要沿用无用途的双probe、重复读取或旁路，也不要连续叠加summary、
  covariance、whitening、transport、anchor、gate或校准链。保留与删除都需要说明当前用途和行为代价。
- 为解决实际能力与学习缺口，允许提出并在隔离探索中验证实质性的创新架构；不将分析限定为现有模块的小修或消融。
  创新候选须说明针对的机制、相对历史尝试的实质差异和可辨别的行为预测；历史优势不自动构成逐一重训的理由。
  新候选先按核心假设、信息路径、参数生成和学习机制核对近等价历史；不得重做已失败方案。若没有新证据或实质机制差异
  能改变原失败判断，不因改名、重新组合或代码已准备而启动；历史否决仍只覆盖实际检验的条件，不扩大为否定全部相关方法。
- owner只评论局部时，保留已对齐且未被否定的部分；不把局部疑问当作推翻整图的指令。先说明完整数据流水线，再讨论局部模块。
- 数学推导从需求、少量符号和直观例子逐步展开；区分推导结论、归纳偏置、实现默认和待检验假设。结构合理不等于性能得到保证。

- Meta属于Writer内部读取模块；所有应训练的内部模块共同更新，不能因称谓统一切断梯度。
- 当前先集中K=1：训练、训练侧诊断、闭环及后续共享Writer RL均使用单视频。通过绝对性能、相邻稳定、
  同task换视频鲁棒性和最终视频因果验证后再开展K>1；保留集合架构，暂不采样或评测K2/4。
- 本轮K1主线明确fresh：旧混合K仅历史探索证据，不继承其权重、优化器、scheduler、sampler或RNG。
  复用已验证架构/source/资产，Writer全部可训练参数从合法identity fresh初始化，fresh学习状态，从step0正式启动。

## 3. 证据与推进判断

- 唯一正式目标是validation8 strict single-checkpoint paired correct严格 >145/400，同时满足相邻稳定、低churn、高breadth、
  四suite非零、Goal/Long贡献、same-task不同视频鲁棒性及最终视频因果controls。正式选择不使用80-row screen、checkpoint union或融合。
- 闭环绝对性能优先。functional loss、reconstruction、norm/rank/cosine、内部margin和surrogate仅用于定位；不能用漂亮数值接受明显更差行为。
- 有信息量且口径可比的正式学习后，若仍不能超过source或仅略超source、长期低于或仅略超SFT，应视为严重能力缺口，
  不能以“较source有提升”“loss在降”把它降格成小调参问题。source现存参照47/400（另一历史面板48/400），
  train24 rank128 SFT相邻109/107；来源与比较边界见research_history。低分不是某个唯一根因的证明，也不能因初始identity/smoke低分直接推翻全图。
- 历史v5.2/v6的强闭环能力、G1容量、G2动态、后续局部正结果及失败边界都要保留。不同checkpoint/配方的优点不能拼成一个不存在的强结果。
- 使用正确视频与同task不同episode的action queries训练。额外non-held meta tasks须审计固定validation/test及重复specification排除，
  保留allowlist/provenance；更多同task视频不等于更多独立meta-task映射。不得制造人工process数据或新仿真任务来绕开当前问题。
- validation/test不得产生梯度。shuffled/reversed仅在selected checkpoint选定并冻结后测试，不进入训练、loss、Gate、checkpoint选择或架构修改。
  no-video/language、static端点、wrong-video等资格或诊断使用时须事先明确用途，不能悄悄把最终controls变成架构搜索信号。
- 当前主线为Writer（含内部读取模块Meta）从头初始化，以fresh optimizer/scheduler直接端到端联合训练；source基础权重始终冻结。
  G1--G3的阶段冻结属于历史机制验证，不实施为当前课程，也不为旧措辞额外建立阶段初始化与随机初始化两套候选。
  LoRA采用合法identity初始化；从头初始化不要求每个张量都随机非零。短学习、扩大覆盖与闭环是实验节点，不是冻结阶段。
- 当前训练顺序为先纯监督FM、后独立共享Writer RL。监督阶段Writer fresh端到端共同学习，source冻结；
  同task跨episode动作监督，不计算RL loss、不采集用于RL更新的rollout、不做RL KL候选接受或整步回滚。
  保留已验证的执行一致性修复；旧联合profile不算正式监督结果，RL未决问题不阻塞监督启动。
- 监督平台要结合真实曝光、训练侧独立动作验证、训练task闭环及预登记validation8相邻checkpoint，不能只看loss。
  有实质改善就继续，连续有信息量节点不改善再判断；充分监督仍弱须先定位并允许实质改进，不以饱和为由交给RL救场。
- 独立RL从选定并保留的单个监督checkpoint初始化Writer，fresh RL optimizer/scheduler和stage记录，默认仅RL目标。
  探索、信用与更新约束根据监督后行为重新审视，不机械复用停滞设置；报告相对监督起点的收益、遗忘、breadth和稳定性。
  这是跨任务共享Writer训练，不能混同部署时task-local LoRA优化；监督checkpoint保留为可回退基线。
- 先用有信息量的短学习与闭环证据判断投入。未证明基础行为前不默认启动约10小时长训练；接近强基线或目标后及时做strict400，
  好趋势继续训练到足以判断相邻稳定，明确坏结果不靠无限续训或无依据的seed/LR/rank/scale/width小扫挽救。
- 每轮记录per-task、per-suite、breadth、retained/gained/lost、churn、相邻success-set重合和实际样本曝光；训练步数本身不足以比较配方。
- 负结果只淘汰真正测试的组合。先区分工程合同错误、有效科学non-pass与证据不足；不要把可疑现象或一次梯度cosine称作根因。
- 诊断应能区分竞争解释并定位最早失效接口。先查历史同类尝试、原始评审及后续修正，再做最小有信息量的干预；
  明确输入变化、旧证据排除什么、新证据如何改变判断。新证据支持模块职责替换时可以实质重构，避免围绕同一接口原地打补丁。
- 反复出现的架构/性能问题必须先查最近等价旧尝试及其结果，明确本次新增的机制、信息或监督；不换名字重做已失败的同一组合。
  重点防止把完整输入/非零梯度当理解，把几何/稳定参数当行为，用新视频或未见task解释训练熟悉视频也弱的结果，
  以及靠堆summary/gate/校准、扩大少数同task样本或无限续训掩盖共享能力不足。
- 不人为规定总工期、修正次数、版本数或总轮数。停止无信息重复，同时允许有新机制证据的合理深入。

- 每段连续训练约一小时；按K1优化后的实测速率，在看到分数前登记中间和末尾两个等间隔附近的checkpoint。
  保存点用50或100的倍数，不机械沿用24/64/128/192。当前主要跑K1 correct strict400，train96按获取/泛化诊断需要安排（held视频46–49在states32–35各一次，另建同口径source比较）。
  早期绝对性能低且仍获取能力时延后other；接近或超过目标、有相邻稳定候选时补资格，冻结选点后再做最终controls。
- 记录累计optimizer updates、FM queries、每task条件曝光和墙钟。历史v5.2为75600 queries、v6-fast为192000、
  SFT参照为230400，仅作曝光尺度参考；64或192步不能自动证明充分训练或平台。

## 4. 授权与自主协作

- 以owner最新授权为准，跨session先理解当前状态并读取progress中的持续授权。已有明确科研执行授权时，理解与进度说明后
  立即推进，不重复请求实施计划批准。只有owner明确暂停或撤回时才停止相应工作；旧交接限制不覆盖新授权。
- 获准接管后，在既有目标、信息墙与资源合同内，实验设计、实现、分析、相关修复、吞吐优化和证据支持的模块重构由接管者连续完成，
  无需逐项询问。不因一个侧面问题、单点好坏或常规技术检查停止已授权流程。
- Owner给予新session充分的证据驱动自主权：核心思想保持冻结动作知识参与有序视频理解、以真实视觉核实过程、一次编译完整策略参数并由闭环裁决；
  具体读取、关系/时序模块、回写、读出、监督/RL与优化实现可以据充分证据修改或重构，不限于小补丁。
  当前设计是须完整理解和实施的首版起点，不是永久不可修订的图；充分已有证据或实际问题证明需要修订时，先明确理由并同步正式合同再完整实现，
  不强行跑已知错误的配置。禁止静默缩水或只因实现方便偏离设计；合同内改动无需再次批准，科学目标/信息墙等真实边界保持。
- 改变科学目标或信息墙、引入未授权数据/资源、无法裁决且显著改变投入方向的路线歧义、删除所有权不明或唯一资产时，
  带具体事实和推荐选择回到owner。不得创造额外审批流程。
- 不把再次完整专家审查作为每轮开工前置。已有明确可检验设计时先获得具体实现与证据；需要专家时给有针对性的问题和新事实。
  未经owner当次明确授权，不向外部专家发送消息，只提供可复制prompt。
- 只有owner明确要求时创建或设置goal，不因任务复杂或跨session自行启用goal机制。

## 5. GPU效率、工程与资产生命周期

- 从算法设计阶段就考虑GPU：批量张量、高效attention、明确布局，减少逐项Python循环、CPU/GPU往返和重复大算子。
  同时审视训练、functional forward、物化与闭环评测；按真实LoRA/s、samples/s、step墙钟、SM/util与显存峰值衡量。
- 每个optimizer update固定四suite各一个task、每task64FM queries，共256queries，task权重1/4。
  教学条件分配由active design显式登记；允许每task两个独立K1条件各32queries、每条件权重1/8，不混同K2或LoRA平均。
  GPU1--6、分工、microbatch和累积次数只决定执行；全局batch完成后clip、optimizer.step、scheduler.step各一次。
  使用真正提高吞吐的同节点GPU，不能扩大逻辑batch或dummy占卡。拓扑变化须有受控迁移合同，保留学习状态和逻辑cursor。
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

- 正常训练、轮询与等待保持静默，不每隔几分钟播报步数、句柄存活或“继续等待”，也不在自动继续任务时反复发送同类结束语。仅在完整结果、实质结论变化、需要Owner处理的问题，或Owner主动询问时汇报；后台观察继续，不把静默误解成停止工作。

- 分析排查须遵守Owner明确的计算预算与训练次数上限；完整训练、重训练不能仅以“探索”名义绕过。达到上限后使用现有证据和轻量诊断，保留未识别边界；当前具体上限见progress与task_plan。

- 默认实用中文，先直接回答具体问题，再给证据和边界。owner主要语音输入，应主动修正明显同音词、断句和术语识别错误。
- 已对齐的部署adapter/信息墙边界不在每次解释中反复强调；涉及变更、违规或用户疑问时再明确说明。
- 讨论像共同推导：不把回答写成教科书岔路，不反复使用“不是……而是……”式对立话术，不把未接受的建议说成owner要求。
- 持久文档职责固定：concept讲科学精神，设计文档讲推导与方法，findings讲跨轮结论，research_history讲分层历史与证据，
  task_plan讲下一阶段，progress讲授权与现场；AGENTS只写稳定合同。
- HANDOFF.md只是消费后删除的临时入口，不能独占长期要求、架构决定、历史结论或执行计划。跨session前正式文档必须完整，
  新session应能自主恢复理解，不要求owner再次解释整段历史。

正式K1评测必须同task、同臂、同轮50个init覆盖50条teacher视频各一次；跨checkpoint和paired controls复用固定canonical state-video映射。此范围不得缩小为单次K集合内不重复。
