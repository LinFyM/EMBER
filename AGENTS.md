# EMBER Repository Instructions

## 1. Authority

当前信息按以下顺序解释：

1. owner最新明确表达；
2. `docs/current_owner_requirements.md`：稳定目标、原则、方法边界与协作要求；
3. 本文件：科研、数据、评测、GPU、存储、Git和工程合同；
4. `task_plan.md`、`findings.md`、`progress.md`：目标计划、持久发现和当前进度；
5. 当前active design；只有`progress.md`明确登记后才存在；
6. `docs/research_history.md`、Git与formal artifacts：历史事实。

旧design、日志、checkpoint和Git快照中的“当前”“下一步”“active”只表示当时时点，不能恢复执行或覆盖上级
authority。owner主要使用语音输入；明显同音词、术语识别和断句错误应结合EMBER上下文主动纠正。
后续所有科研判断与自主推进还须遵循本文件§12的长期原则；它们不能被某一版design或阶段性结果绕过。

## 2. Task-scoped reading

先理解当前目标、授权和状态，再按任务读取相关材料；不把全部历史全文重读作为每次工作的前置流程。

- 新任务先读`docs/current_owner_requirements.md`，以及`task_plan.md`和`progress.md`的当前目标、快照与授权状态；
  区分当前记录与文件内的历史执行段落。没有active design时，不从旧设计或未完成清单自行恢复实验。
- 修改科研代码、数据、模型、实验配置或开展设计判断时，读取`docs/concept.md`、相关`findings.md`结论，以及
  `progress.md`明确登记的active design中涉及的接口、数据、训练和裁决合同。必要时扩展到完整设计，不能断章取义。
- 涉及架构取舍、负结果解释或历史路线时，先查`docs/research_history.md`，沿其索引读取相关专家评审、修正意见、
  Git快照与formal evidence；读完整的相关论证和适用条件，保留已确认的边界，不必重读无关评审。
- 窄范围修复只补读相关实现、调用方与验证合同。纯文档、技能或工具配置维护读取当前目标和相关规则即可；
  不因此检查GPU、重跑实验或加载全部科研历史。
- 本任务已经读过且未变化的材料直接复用。跨上下文窗口恢复时，先核对最新owner消息与当前状态，再按缺失信息检索；
  只有依赖实时GPU、进程、存储或Git状态的操作才刷新对应证据。

GPU launch与formal train/eval仍须满足下文完整科学、资源、checkpoint、Git和评测合同。按需阅读不降低这些执行要求。

## 3. Repository role

`AGENTS.md`只记录稳定的项目总览、科学合同和执行原则。它不记录active run、最新checkpoint、实验分数、动态下一步
或临时协作状态。仓库级工作状态只使用`task_plan.md`、`findings.md`和`progress.md`；历史结果只进入
`research_history.md`。临时handoff只在真实跨session交接时创建，消费后删除。

## 4. Scientific objective

EMBER研究能否从generic `lerobot/pi05_base`建立的冻结π0.5-LIBERO source policy出发，把目标task的exact language
和一条或多条action-hidden正确教学视频，在最终评估前编译为一套完整task-conditioned LoRA，使policy从
未见初始化闭环完成任务。

编译期允许有预算的真实实践、任务内学习与阶段性LoRA修订：可使用自身观测、实际执行动作、环境状态、接触及奖励等反馈，
最终固定一套完整LoRA，在未参与该condition适应的新初态评估。试做预算、重置与最终评测须预先分开登记；
有交互适应不得记作zero-interaction。离线teacher episode的动作/state等配套标签仍不可读；
自身交互所得reward/success/terminal可以用于适应。观看、实践和修订可多轮进行，具体学习组织由active design登记。

性能接受标准以`docs/current_owner_requirements.md`中的owner最新取舍为准；不得把历史`>145/400`合同
自动恢复为新的硬门槛。第一阶段以完整固定LoRA明显超过强MT为先，不把胜过同预算充分语言适应或补齐全部视频controls作为开工前提。
正式结论仍须用strict single-checkpoint paired400、相邻能力保持、任务/suite覆盖、
same-task换视频和视频因果证据说明结果，区分正常小幅交换与大范围能力丢失，不以偶然峰值代表稳定能力。

closed-loop absolute性能首先选择方法。LoRA norm/rank/cosine、reconstruction、functional loss、内部时序margin、
hidden差异和surrogate只作定位证据，不能为了数值漂亮接受明显更差的闭环性能。

## 5. Input, output and information wall

- 输入必须包含exact task language和一条或多条同task、action-hidden、内部有序teacher videos。
- language说明关注什么和目标是什么；方法须解释video操作内容怎样参与理解、实践和最终能力，视频增量主张须有实际证据。
- 教学条件不得包含teacher episode的action、proprio/state、reward、terminal、task ID、filename、object pose、
  hidden normalization或特权policy outcome，也不得额外调取held task的离线训练资料。
  编译期自身真实交互的RGB/proprio/动作、环境状态、接触、reward/success/terminal等可用于学习，须与teacher标签区分。
  授权的non-held meta tasks可在训练时使用action、privileged expert或on-policy reward学习共享Writer/functional decoder，
  这些监督标签不得成为deployment输入或task-ID route。
- validation/test离线示范标签及最终评测反馈不得产生梯度；获准的validation任务内适应可用自身交互学习，
  须明确任务内参数副本与共享训练状态，不把held适应经验混成未登记的共享训练。
  允许模型冻结、无checkpoint选择、预注册的一次性sealed post-hoc held诊断；Test默认保留到最终方法冻结后，提前使用必须明确登记且不得反哺设计。
- 每个condition最终只部署一套完整38-target task LoRA；编译期可修订中间LoRA，但不生成多套video LoRA后平均，
  不挑video、不融合checkpoint、不部署第二套expert adapter。
- Writer/Compiler只在最终评估前的适应期运行，可包含前向生成或任务内优化；可重复读取合法教学及自身实践，并让当前LoRA参与后续理解。
  最终同一教学condition的一套LoRA冻结后，Writer、搜索和外部阶段选择等适应辅助退出；最终闭环不看teacher video或继续适应，旧零交互实验按原合同解释。
- frame stride固定为5；frozen source policy无trainable parameters。允许learned language-only诊断baseline，以及
  rollout前合并为一套LoRA的principled shared prior/base adapter + video-conditioned residual；canonical仍必须证明
  video相对language/static prior有必要条件增量，且不得部署并行carrier、expert或第二adapter。
- task experts可作为train24及经审计的non-held LIBERO-90 meta tasks的privileged teacher或几何诊断；不得成为held
  dictionary、task-ID route或第二套LoRA。

Dynamic-K若被方法声称支持，训练必须真实覆盖各cardinality。每条video先独立保序编码，videos只在集合阶段做
置换不变聚合；不得平均frames、raw features或最终LoRAs。one-shot、few-shot或动态K最终采用哪种论文设定，只由
真实性能决定，不为形式公平故意削弱更强方案。

memory token、LoRA rank、FactorHeads、layer correspondence和具体decoder都是候选方法，不是目标或硬约束。
不得为了“用上backbone”构造无意义的zero-image、fake action query或缺失原生prefix的forward。

## 6. Fixed data and benchmark contract

- LIBERO Spatial/Object/Goal/Long共40 tasks；
- development使用active design登记的唯一显式24 train / 8 validation / 8 test协议；历史协议保留，不覆盖原结果。
  Owner明确授权的覆盖重划须在fresh训练前冻结并审计完整任务等价泄漏，不得在运行中按结果改ID；
- source corpus由LIBERO-90 specification audit排除与目标40重合的19 tasks后保留71 tasks，每task 50条成功episode；
- successor Writer/meta-training可使用train24，以及LIBERO-90中经过精确语义/specification审计、明确排除固定
  validation/test tasks及其重复项的其它任务；必须保存显式allowlist与provenance，不得以更多同task episodes冒充
  更多独立meta-task mappings；
- 不得使用读过目标40 actions的`pi05_libero`；
- normalization只从过滤后的source actions/states计算并冻结；validation/test不得重算；
- 合并Validation重训只在Owner明确授权的合同中允许；不得从历史final-stage配置自行恢复。

## 7. Training and decision contract

- 共享训练的target监督只来自24 train tasks；active design可登记额外、经审计的non-held LIBERO-90 meta-task
  gradients。所有授权meta tasks按预注册口径等权或显式分层；held离线标签不产生梯度，自身交互任务内学习按§5单独登记。
- 主video-conditioned LoRA的功能监督使用同task但跨episode的video/action query，阻断逐帧轨迹复制。
  经owner授权并在active design登记，训练期可用授权episode自身的RGB转移与动作作辅助监督，包括经同一套
  生成LoRA回传的功能误差；须显式登记标签来源、梯度消费者和与跨episode主项的权重。
  标签不得进入视频表示／Compiler的条件输入，teacher视频仍action-hidden，部署不得做逐轨迹适配。
- 多卡可按K、帧数和历史cost平衡负载，但不得改变task权重。
- formal checkpoint保存Writer、optimizer、scheduler/scaler、sampler/cursor、rank RNG、world topology和schema。
- incompatible架构必须fresh。同拓扑exact-resume保留原world size/topology；Writer和MT-BC均可在完整checkpoint边界显式迁移物理卡数、设备与分片，保持逻辑查询流、任务/loss权重、有效batch、optimizer/scheduler及选择合同，记录新旧拓扑与RNG恢复/新增来源，不称bitwise exact。
- 物理rank数是吞吐安排，不是科学超参数或永久上限。不得因一次smoke或旧恢复实现锁死卡数；迁移须由实际实现正确承接，不能直接绕过校验。已近完成的训练不为形式上的扩卡而中断、丢弃未保存更新或重复计算。
- source基础权重始终冻结。共享Writer复现实验遵循其登记的fresh初始化、真实FM共同训练与独立共享RL合同，
  不将历史配方套作所有后继方法的固定课程。后继可从强MT控制起点做任务内学习；初始化、共享/局部参数、
  FM/价值学习/策略优化的组织与消费者须在完整设计中明确，teacher辅助动作query仍不得进入教学条件。
  G1--G3分段冻结属于历史机制验证，不自动构成新课程。合法identity不要求每个张量随机非零，学习与闭环节点由真实证据裁决。
- 机制smoke只证明图接通。到有信息量的预注册节点后及时做strict paired400；loss不能代替闭环。
- 好结果应训练到足以判断相邻稳定性；明确坏结果不得靠无限续训或rank/scale/seed/LR/dtype小扫挽救。
- 每轮必须报告per-task、per-suite、breadth、retained/gained/lost、churn及相邻success-set重合，并定位最早失效接口。
- 一次尽量只改变一个主要因果变量。负结果只淘汰实际检验的假设，不能因局部失败整套180度转向。
- 正式checkpoint选择只使用active design预注册的qualification arms与相邻稳定性；selected checkpoint选定并
  冻结后再补视频因果controls。shuffled/reversed最后测试，只确认时序特异性；它们不进入训练、loss、
  checkpoint选择、G1--G5 Gate或架构修正依据。

## 8. Evaluation contract

- official preprocessing：render256/model224、双相机180度rotate、8维state（position3 + axis-angle3 + gripper qpos2）/7维action、10 flow steps、执行前5 actions后
  replan、dummy settling10、成功即终止、suite horizon 220/280/300/520。
- 正式K=1的一轮50个init states中，同task、同一评测臂必须使用全部50条合法teacher videos各一次；
  validation8应为400个不同task-video条件。不重复按task与video共同判断，不能缩小为单次K集合内部。
  跨checkpoint必须复用同一固定state-video映射；correct/other各自整轮无重复且逐行视频不同，controls按配对合同复用映射。
  调度复用`src/ember/expert_manifold/video_schedule.py`，只由登记seed/task/init state决定，与GPU、worker顺序、分片及恢复无关。
  有限池训练诊断必须登记面板与复用范围；视频不足不得宣称整轮无放回或静默重复。
- correct/same-task-other/cross-suite-wrong/shuffled/reversed/no-video严格配对task、state、env/policy RNG和video
  ordinal；shuffle/reverse必须重排真实frames后重新完整forward。
- evaluator采用cost-balanced dynamic queue、long-first和persistent workers，不做静态task/GPU分配或dummy占卡。
- 正式选择只认single-checkpoint 400 paired rows；80-row screen、checkpoint union、融合和内部surrogate不能选模型。

## 9. GPU, throughput and numerical policy

- 每次launch前同时live检查gpu01与gpu02，按节点、GPU index、显存、utilization和process判断空闲、可共驻、忙或故障。
- 两节点合计同时最多使用8张物理GPU；两节点空闲卡总数不超过10张时，合计上限降为6张。训练、物化、
  评测与共驻占用统一计数，每次launch或resume前核验当前总量及启动后总量。单节点仍至多6张；
  只使用真正提高吞吐的A40，不等待凑卡、不跨节点拼训练碎片、不dummy占卡。
- Owner允许短时分析例外：现有实验已占满上述卡数上限，而已授权的小分析确需GPU时，可以临时突破卡数上限，
  分析完成立即释放额外占用。记录用途、预计时长、实际开始/退出和成本；仍须现场准入、不干扰他人任务并计入对应预算。
  此例外不扩大实验科学范围，不把额外卡留给常规长训/整轮评测，也不通过连续短任务规避常规上限。
- 少量显存占用或低util进程不自动排除GPU；只要峰值余量足够且不会显著干扰即可共驻，但不得kill、pause、
  reset或抢占他人任务。
- 多卡训练固定`NCCL_P2P_DISABLE=1`、GPU-local NUMA映射和deferred NCCL；独立evaluator不用NCCL。
- 接受硬件、设备分配、物理batch/microbatch、正常BF16/TF32、kernel和reduction order带来的微小数值差异。
  不影响有效梯度、更新语义、数值稳定性和实际训练/推理行为的差异无需追查或消除，不把逐元素或逐bit一致作为验收目标。
  仅在有证据表明差异破坏上述要求时开展针对性诊断和修复；不为低位一致固定batch1、重复forward、扩dtype、
  关闭高效kernel或逐tensor扫描。
- 不新增SHA-256、MD5或大量防御性校验。只保留信息墙、shape、finite、OOM、asset、pairing、checkpoint和resume
  正确性所需检查。
- profile以真实LoRA/s、samples/s、最长视频稳定性、GPU利用率和显存峰值选择batch，不以最低显存为目标。
- 必须主动利用可用显存提高实际吞吐。实测仍有明显显存余量时，在当批授权范围和预算内验证更大的物理microbatch、
  frame chunk或合适的并行worker；按真实吞吐选择配置，不得直接沿用保守默认值、只报告低显存占用后忽略余量。
  配置选择须留下吞吐、显存峰值和未继续放大的具体依据；有余量却不利用的决定须有实测无收益或明确资源限制支持，
  不能等Owner再次提醒。只改变物理执行安排，保持逻辑有效batch、任务/loss权重、optimizer更新、数据流和科学合同；
  不为填满显存占卡、扩大科学矩阵或做参数小扫。
- 效率优先：在上述资源上限和实际预算内按吞吐选择训练rank数，及时并行独立训练、物化与评测，不让已就绪工作等待无关阶段。额外收紧整批并发上限须有具体资源/吞吐依据，不能由单个训练world size或渐进实验要求直接推出。
- 正式长任务正常运行期间，等待进程退出事件或完成标记；不要按固定间隔反复读取训练进度、日志、checkpoint或缓存。
  使用一次持续的退出事件等待；不要把等待拆成每隔几十秒的空查询，也不要在没有变化时反复发送“仍在等待”的状态消息。
  进程结束后再统一核验退出码、完成记录和产物。仅在出现故障迹象、需要即时资源调度或Owner明确要求状态时，
  做有针对性的轻量检查，避免重复读取共享缓存。
- 同一任务已经直接等待子进程退出时，不要让该进程再向本任务发送`codex queue`自通知；等待返回后直接处理退出回执。
  确需跨回合接续且没有持续等待者时，只为整批完成或异常安排一条可存活的唤醒消息，不按训练、物化和各面板分别预排自通知。
  完成回报前核对已有自通知回执；迟到消息按完成记录判重，不重新执行，也不逐条发送无意义的状态回复。

## 10. Storage, artifacts, Git and documentation

- EMBER后续新增代码、数据、模型、checkpoint、缓存、实验输出及导出一律放`/data1/user/ymdai`，适用于所有后继任务和session，
  不是单批例外或“优先”建议。历史`/data0`资产可按原路径只读复用；不得据旧合同在data0新增产物，也不因本规则迁移或删除历史原件。
  大copy/cache/training前在`strg01`检查data1的独立user quota、测实际用量并合计代码/缓存/输出峰值；`df -h`不是quota检查。
  data1预计不足时，先在既有授权内处理已核实可清理内容或回报具体缺口，不自行切回data0。
- 复用canonical policy、dataset、tokenizer、assets和manifest，不复制大资产。
- formal结果保留run contract、checkpoint manifest、metrics、raw rows、aggregate、completion和必要analysis；
  profile/smoke不得冒充formal。
- active tree只保留一个canonical Writer运行面。退役实现由Git、sealed configs、formal artifacts和
  `research_history.md`保存，不保留平行fallback。
- canonical workspace是本仓库，主写与集成目标为`main`。需要隔离、并发写入或独立实现时，从最新`main`创建
  `codex/<topic>`分支与独立worktree；验证后及时合并回`main`并推送远端，确认集成完整后清理task-owned worktree。
  formal train/eval必须来自clean pushed commit的detached frozen worktree。
- 不提交dataset、cache、checkpoint、大binary、secret或host-private配置。
- 稳定目标写入`current_owner_requirements`和`concept`；当前goal与计划写入`task_plan.md`，即时进度写入
  `progress.md`；历史结果写入`research_history.md`，跨轮结论写入`findings.md`。不得向`AGENTS.md`追加动态
  实验年表。
- 只删除生命周期已核实结束的内容；数据集、原始formal结果、关键模型和当前依赖保留。历史checkpoint不因唯一或formal标签永久保留：
  按Owner最新清理授权择点保存关键权重，退休无恢复用途的训练状态及冗余中间点，记录删除清单和实际可用范围。
  权重存档与完整训练checkpoint须区分，不把已删除optimizer的存档继续称为exact-resume资产；所有权不清的内容不动。

## 11. Collaboration

owner授权在上述边界内自主推进。具体协作模式以owner最新要求和`progress.md`为准。反馈是设计约束或启发，不是要求
机械照搬；不得因owner提出一个局部问题就推翻所有已对齐内容。需要解释架构时，先给出完整数据流水线和每个模块
的因果作用，区分目标、原则、诊断与具体方法。

执行者可以自行修复已定位的接口、记录或调度违约，并在原授权科学范围与资源预算内继续执行；冻结代码不得原地修改，
不等于每项工程修复都须停下来重新派发。修复须在独占分支完成、通过实际消费者的针对性检查、推送并使用新的clean detached
冻结版本，保留原代码/有效产物/失败记录及完整计费，区分训练来源与后续读取/评测代码身份。
代码实现、测试、运行排障与Git集成由实验session闭环；主讨论负责科学分析、设计与取舍，不重复工程审查或测试。
科研记录写入与代码集成串行协调，避免同一worktree并发写入；不以主讨论工程验收作为执行停点。
若修复改变模型计算、数据或标签、损失/更新语义、checkpoint选择、评测口径或信息墙，原因尚不确定，或将超出授权预算，
则报告具体边界由主讨论裁决；不得借接口修复进行科学改动，也不得将有效负结果当工程故障修补。

## 12. Persistent scientific principles

以下为owner明确要求长期执行的科研原则，适用于所有后续方案、实验和session。
它们约束实际决策，不是新增审批、逐模块闯关或每轮补齐所有对照的清单；详细解释与历史依据见
`docs/current_owner_requirements.md`和`findings.md`中的长期推进错误回顾。

1. **围绕最终目标选择工作。** 每项分析或干预须说明它解释哪项实际不足、会改变什么完整方法判断。
   不能沿局部问题不断追查其下游，最后让通用现象、内部指标或工程形式替代EMBER目标。
2. **实质进步包含机制理解。** 应逐步更清楚为什么能学会、为什么未学会、怎样才能学成；
   架构版本、工程完成、实验/文献/报告数量和关闭了多少分支，都不能替代这种认识的推进。
3. **数学解释落实到实际特征与算子。** 说明输入及隐藏特征的来源、计算与依赖，信息如何被编码、选择、
   汇聚、归一化或调制，怎样形成各层LoRA，怎样作用于自身执行hidden和动作，以及真实梯度如何教会并保持这条联系。
   模块名称、一般链式求导、复杂公式或可达性证明不单独构成机制解释；可表示、可学到、可迁移分别论证。
   将特征语义假设与实证分开，不能由注意力、可解码性或表示差异直接宣称理解了操作知识。
4. **完整分析结果，理论与证据相互约束。** 延续owner认可的旧主讨论后期分析风格，将有决策价值的新结果放回
   EMBER完整问题：交代原机制假设、竞争解释与可失败预测，说明实际干预改变了什么、原件支持什么。
   对照最近似历史方案的正反证据、学习阶段与实际差异，按本节第3项把数学推导落实到视频／语言、原生特征、
   读写算子、LoRA及自身执行hidden／动作的完整链路，说明可行或受限的条件；检查局部机制效应能否解释总体性能，
   不能由微小内部改善直接归因主要闭环收益。保留不利个例、效应大小、任务分布和不确定性，明确哪些解释被加强、
   削弱或仍未区分，形成自己对训练、架构或数据取舍的判断。向owner连贯说明这条推理过程，不能只交分数、
   边界声明和下一项实验。分析深度不以篇幅、公式数量或固定清单衡量，也不要求每轮额外开探针或补齐全部对照。
   方法投入须受上述机制与预测约束，结果必须更新支持度和取舍；不在分数出来后补不可区分的故事，也不要求
   先证明一定成功或确诊唯一根因才允许有界检验。理论在推进中建立，不把无限纸面推导变成新的停滞方式。
5. **尊重原件和未知。** 核对相关源码、数据及实际执行，不能只转述执行者总结；区分事实、推导、假设和因果结论。
   工程接通不等于有效，涨分不等于机制证实，科学阴性不自动变成bug或普遍不可能。
   没考虑充分、没比较、没验证的地方直接承认；不得把候选、局部通过或未实施后段写成已证实结果。
6. **比较必须改变判断且公平。** 事前明确主要干预、对照、结果分支和适用范围；检查原有数据/方法是否已提供
   所需功能，新构造相对它究竟增加了什么。隔离配对效应不自动回答构造数据相对原始数据的价值。
   不削弱强参照、隐藏高点、只胜上一版失败方法或用代理指标替代闭环；保留能力得失、覆盖和不确定性。
7. **继承完整历史，避免近似重犯。** 核对最近似完整方案及其修订，比较实际特征、算子、标签、梯度消费者、
   学习条件和部署作用。改名、换维度或重新组合不能清零失败与成本；未试过本身不是投入理由。
   负证据须降低相关假设的支持度和优先级，正证据及原有适用边界也不得被抹去；不为证明区别重跑旧实验。
8. **同时防止补丁循环和换架构循环。** 停止线须约束继承的主假设，不能只关闭一个版本名称。
   分阶段用于控制成本和获得判断，不是局部过关后的自动晋级；不同条件下的局部正例不能拼成端到端资格。
   不以无限续训、rank/LR/seed小扫或不断追加局部探针保护同一弱假设。
9. **架构基础可以撤换。** 不因继承关系或沉没投入保护当前模块，也不因单次低分整体转向。
   根据结构约束、累积证据和完整替代原理判断；重构须说明为何改变旧失败预测，保持特征—算子—梯度—部署自洽。
10. **主讨论独立判断并承担责任。** 结合证据形成自己的理论与取舍，不等待owner逐步指定研究方向。
    owner的疑问和例子是约束与启发，不能机械逐条开实验或因一句反馈推翻全部已对齐判断；遗漏和错误如实承担。
11. **投入渐进且有时间预期。** 每批给出wall-clock预计及依据、资源上限和停止条件；先取得能改变判断的最小证据，
    再按实际缺口扩大。检查时间不是制造忙碌或铺大矩阵的理由；显著超期、超预算或方向改变须明确说明。
12. **自主推进、沟通和接续必须真实。** 仍有可做工作且没有实际独立执行或必要阻塞时，主讨论不得结束后等owner催促，
    也不得主动结束再给自己Queue。执行者长任务期间不陪跑空等或反复查状态；主讨论推进独立有价值工作。
    有实际独立任务和可靠整批回报时可结束回合，回报后须主动验收、裁决和接续；计划文字和口头承诺不算后台工作。
    不为持续运行制造无判断价值的任务。直接向owner解释认识、证据、未知和下一决策，同时将关键约束、历史错误与
    判断保存在现有记录，避免换session遗忘；不只交长报告，不用心跳、重复回执或自通知冒充进展。
