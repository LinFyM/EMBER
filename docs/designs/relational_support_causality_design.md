# 任务关系支持的学习干预

2026-09-24，主讨论登记的下一批有界训练；是否active以`progress.md`为准。
机器合同：`configs/relational_support_causality_v1/experiment_spec.json`；完整任务审计、protocol、manifest及evaluation在同目录。

**评测规模修订（2026-09-24 14:38 UTC，任何本批正式闭环结果产生前）：** 六臂训练与1260主因果问题保持，
仅执行1050/1260两个固定评测节点，1260为事前指定的唯一报告模型，1050只检验相邻保持；取消最佳点选择。
闭环从原上限21868收缩为9932条。原六节点计划由Git保留，不再授权执行210/420/630/840评测或selected额外controls。
冻结训练配置中的六个`evidence.evaluation_updates`与spec的对应optimization字段保留为原训练元数据；
实际评测节点的唯一authority是`evaluation.executed_updates`，不能据旧训练元数据启动已取消的面板。
本修订另允许机械同步评测注册、调度、固定1260记录和结果验收/汇总；这些改动与被动采集修复一起集成到唯一E。
不能借规模修订改变policy forward、环境/动作/成功规则或原预定义1260科学对比。
主讨论`01a0cd94-65da-7b22-8ca9-7ba35f454632`负责设计和解释；现有Sol
`01a0cd90-ebb7-77a1-a20b-a858825d2f66`负责实现、执行和原件回报，使用Queue接收后续任务。

## 1. 从哪些证据出发

五批诊断已经缩小了问题，但尚未找到统一根因。原四臂的语言B selected124/400、视频C110/400；
C有Object14的46/50新能力，也有Goal21从Source40/50到25/50的保持损害。
冻结前段和动作通道干预说明Goal早期水平命令有因果作用；Object方向不同，不能整体换成B或回退Source。
同query交叉50条正确视频，C/B早期函数差异主要为正确池共有分量；这既可能是合理任务抽象，也可能是错误迁移，
不能把小方差判成视频无用。最终action_out的冻结干预没有兑现Goal功能方向预测，且损失Object能力，已结束删层/通道扫描。
完整证据见findings§135–139及各formal root，尤其§139的独立原始行、实际10-flow和逐state复算。

现在干预**适配如何学出来**。任务关系支持是竞争解释之一，不是上述推理干预已经证明的原因。
固定跨episode风险允许同task视频不变的条件解；学会单个操作与把操作知识迁移到新组合也不同。
Source71已经见过本批全部新增任务，变化只发生在Writer的共享meta监督关系中。

历史上扩大meta任务已经失败过，不能声称从未试过或只试过更多同task episode：

- findings§68和research_history“meta73、target18与fully-random”：73/18任务对照同时改变单target权重、更新组成和总queries，
  screen80没有明确优势，训练侧也不充分。它否定“更多映射自然有效”，没有隔离固定预算下具体关系支持的效应。
- 旧G2、局部动作读取、Core/Procedure交换、末P修复、任务隔离及private/freeAB已存在正负证据；
  本批不增加读取器、输出rank、辅助loss、更新缩放或公共底座课程来混合解释。
- Object14在fit28缺少ketchup动作目标时仍得到46/50，是“缺某类训练对象就必然失败”的直接反例。
  本批保留它及全部held8，不为修Goal牺牲已获能力。

## 2. 数学模型、可检验预测与边界

用一个**待检验的低阶模型**表述可组合控制知识，而非声称现有网络已实现它：

\[
c(o,d)=u_o+v_d,\qquad c\in\mathbb R^m.
\]

这里`o`是被操作对象类型，`d`是完整目标类型（谓词On/In、装置/对象类型、region）；`c`是假设可迁移的任务控制表示。
每条监督关系对应设计矩阵一行`e_o+e_d`。在不连通的对象—目标二部图中，某分量上的
`u_o→u_o+a, v_d→v_d−a`不改变该分量的所有已见关系，却改变跨分量的新关系。
若两个节点在同一连通分量，目标行属于已见行空间，可由路径上的交替和表示。

本批路径为：黑碗—盘子—白碗—双层架顶面—平底锅—炉面。假设满足时有

\[
c(黑碗,炉面)=c(黑碗,盘子)-c(白碗,盘子)+c(白碗,架顶面)
             -c(平底锅,架顶面)+c(平底锅,炉面).
\]

这不是对实际LoRA、动作或视频做加减，不是部署算法，也不是现有函数类的不可学习定理。
实际生成和闭环都保持原模型；矩阵计算只产生干预预测。语言语义、Source先验、场景、初始关系与反向操作可以突破此低阶模型，
例如原Spatial7已经教黑碗从炉上移到盘子。场景依赖项并未由上述代数证明抵消。
本批不是通过BDDL覆盖图证明网络内存在某种表示。

完整metadata审计在`partition_audit.json`：按真实类型和region构图，保留所有fit任务的On/In目标。
S00/S01/S10的Goal21目标行不在行空间，S11在；相应投影残差约.601/.527/.584/3.6e−14。
读过实际完整BDDL，两组替换对都只有`:language`与`:goal`不同，regions/fixtures/objects/init一致。
`wooden_cabinet`与`wooden_two_layer_shelf`不能因为自然语言都叫cabinet便合并。

**事前预测**：如果两处关系缺口是当前C迁移困难的有效来源，补齐两处应产生超过两个单独替换的Goal21收益，
即同一训练节点的`I_C=p11−p10−p01+p00>0`，同时C11相对C00有绝对改善；另一条正确视频也应兑现，而非只改变wrong。
这将测试数据关系怎样影响学习，不仅比较哪组分数最高。

竞争解释与裁决：

| 解释 | 对照预期与解释限制 |
| --- | --- |
| 一般放高/On相对In或某条任务易学 | j替换在两个i水平都改善，主要是j主效应，不能声称缺两条组合联系 |
| 白碗放盘操作或任务自身的普遍收益 | 主要是i主效应，同样不足以支持两处连接机制 |
| 一般数据关系改善，不涉及视频独有价值 | B11−B00与C11−C00相似；保留这一有效结果，但不能称视频修复 |
| 关系已足够，现有学习/编译不能兑现 | 支持任务操作已掌握，C交互/绝对收益仍不成立，削弱本窗口中的支持修复预测 |
| 新支持任务本身没有学好 | 看每臂四个支持任务的真实闭环和Source保持；不能拿未习得的关系否定所有可组合学习 |
| 非线性的任务难度、场景或优化交互 | 即使I_C为正仍可能存在；单seed两处任务替换不能唯一证明内部加性表示 |

主对比是1260时Goal21 correct的交互及配对区间；其余节点完整报告，不能事后挑交互最大的点。
强支持需要交互与C11绝对改善同时成立，other方向一致，且晚期相邻节点不是一闪而过。
若区间跨零，只能说证据不足；若只见主效应，就修订解释，不把任何涨分都归于图连通。
若关键支持任务近乎不成功或显著损害Source能力，明确保留“支持操作尚未习得”；支持任务高分本身也不证明可组合。

视频收益要求正确和另一正确视频的绝对能力、相同数据的fresh语言B、同模型wrong对照一起解释。
差中差`(C11−C00)−(B11−B00)`受语言B已有较高分和成功率上界影响，不能单独证明视频更会组合。
C11即使高于B11，也还涉及不同活动模块；正确内容相对wrong的益处必须来自correct/other改善，不能靠wrong退步。
本批不声明动态时序必要性，亦不把单seed诊断held8当新方法的官方泛化资格。

## 3. 精确数据干预

官方coverage24/8/8完全保持。诊断held8仍为`[0,1,14,15,20,21,36,38]`，均属官方Train，
但本系列不进入optimizer。固定seen_target16为`[2,4,5,7,12,13,17,19,22,25,28,29,32,34,35,37]`。

原fit28在所有新臂共同去掉`[43,56,95,96]`，保留24；共同增加58和83，再各加入i、j对应的一条，共28。
这些共同背景变化不能与旧C420比较后冒称单条关系效应；本批主比较全部来自新训练。

| 构成 | global ID（LIBERO90 local=global−40） | 完整目标 |
| --- | --- | --- |
| 共同 | 58（18） | 平底锅On炉面 |
| 共同 | 83（43） | 白碗On双层架顶面 |
| i=0 | 77（37） | 白碗On桌面上盘子右侧region |
| i=1 | 76（36） | 白碗On盘子 |
| j=0 | 80（40） | 平底锅In双层架上层内部region |
| j=1 | 81（41） | 平底锅On双层架顶面 |

四池共同26任务：`[2,4,5,7,12,13,17,19,22,25,28,29,32,34,35,37,42,51,55,58,62,64,73,83,97,101]`。
四池完整列表、类型/初始/目标、矩阵边与来源见机器合同和partition_audit，禁止按新结果换任务。
六个新增候选都在经审计Source71，完整任务不等价于目标40；明确不能把“元任务非官方held”简化成只比语言。
新protocol的辅助元数据是原12加新增6共18，manifest覆盖40+18；每个训练臂只允许其fit28，
42个metadata Train不等于42个梯度任务。旧protocol/manifest/config和全部formal原件不修改。

六臂：`C_S00/C_S01/C_S10/C_S11`及`B_S00/B_S11`。
四个C分别只变两项训练任务；两B分别匹配端点S00/S11的数据，都是fresh真正text-only训练。
B无图像/VL Meta/Action Meta/Procedure，不是把视频C推理输入置零。

允许fit28 demo0..45产生监督；teacher同池，主21条不同episode且排除teacher，额外7条同task非teacher episode。
demo46..49只作无梯度评测；不新增held expert action/state读取、统计拟合或梯度。
部署仍exact language＋action-hidden agentview视频一次生成完整38-target rank16 LoRA；source和normalization冻结。
任务ID/BDDL仅调度与审计，不进入Writer输入。无task-local优化、RL、第二adapter或现成共同底座初始化。

## 4. 保持的学习合同

逐项复用原B/C配置的Source、模型、observer、identity初始化、活动参数、优化器及21+7随机tau/full-H50 FM，
详见原四臂设计§3–5和本批spec。新增study身份/显式任务池/metadata authority，不冒用旧experiment。
额外组不是tau1辅助；所有C共享同一架构和fresh seed7，B共享模块按原构建顺序fresh seed7。

- 每宏步四task，各`mean21 + (1/3)mean7`，四task取平均；112queries总尺度4/3，每条1/84。
- 28task每7宏步等权一次；1260宏步=180轮=5040条件=141120query/臂。不得按GPU再除权重。
- 原sampler20260721、teacher20260722、teaching20260919、model/optimization7；offset1、stride5真实末帧、K1。
- 四池sorted slots一致，共同26task的teacher/query episode/frame/flow RNG及macro位置逐项相同；同pool的B/C全部流相同。
  被替换的两task保留canonical实际task-ID派生随机流，真实轨迹、长度与噪声不同；不能冒称异task逐action配对。
  不为制造一致添加随机种子别名或重写采样算法。
- AdamW lr3e−4、betas(.9,.95)、eps1e−8、wd1e−4、clip1；warmup150、decay18000、floor1e−5，
  tail1350..2250/ratio.1在本窗口不启动。每宏步一次更新；不继承旧optimizer/checkpoint。
- 完整checkpoint每105宏步，共12/臂；实际评测只执行1050/1260。保存完整恢复状态及applied LR，不删除原checkpoint。
  窗口不足以声称收敛/容量上限，完成后不因结果差自行续训或扫seed/LR/rank。

## 5. 闭环、controls与学习作用检查

原官方执行合同保持：render256/model224、双相机rotate180、8维state/7维action、10-flow、执行前5再replan、
settling10、成功即停，目标四suite horizon220/280/300/520；新增LIBERO90 support horizon400。
动态队列、long-first和persistent workers复用canonical evaluator，不另写近似环境。

1. 六臂1050/1260两个固定节点各held400＋seen64，共5568条。held task各50states/50正确teacher无放回，seed20260911，跨臂/节点固定映射。
   seen16×4，state0..3对应teacher46..49。B同task可复用一套LoRA，标注实际teacher values read=0。
2. **六臂均在任何结果产生前固定1260为唯一报告模型**，1050只作相邻保持参照，不按结果换点。
   不搜索最佳checkpoint；没有六节点曲线或早期峰值的结论。机制主比较始终共用1260，原来主要假设和检验不变。
3. 六臂全部训练、两个correct节点完成并核对固定1260记录后，四C的1260各跑held400 other（偏移17），
   C00/C11的1260各跑held400 wrong（原跨suite固定donor），共2400 control rows；没有selected额外面板。
   不跑shuffle/reverse、不以wrong约束训练或按control重选模型。
4. 每臂1260评其四个新增支持任务，各50states/50teacher无放回，共1200。
   evaluator用nonheld_meta加**显式global/local正确转换的subset**，不是运行全71，也不是旧15架构validation。
   这是学过操作的获取/保持检查，不是held泛化分数，不影响选点。
5. 新运行同合同Source：held400、seen64、六个新增support各50，共764；新批主配对不拿旧Source行替代。

共9932闭环（5568 correct＋2400 controls＋1200 support＋764 Source）。完整保存全部逐state success、实际动作、stage predicates、连续对象/EEF/夹爪和条件引用。
1260的六correct、四other、两wrong和Source在task14/21、state0/25固定双相机full；
support每臂四task的state0及Source六task state0也full，共82cases，事前固定，不按成败挑例。

所有主要对比见spec：i/j各条件效应、交互、C端点、B端点、同pool C−B及差中差；
逐task/suite/held400、seen64与1050→1260相邻节点报告Source R/G/L、churn、breadth、成功集合Jaccard。
controls同时报告correct/other的绝对数、配对保持交换和wrong，不只给差值。
bootstrap20000/seed20260924：总体以task为cluster，各task以state为cluster；跨臂/视频/节点共同抽样。
区间只反映这些初始化/任务的不确定性，不覆盖单training seed的训练方差，不能把多对比择优当事前显著结果。

核对事前固定1260记录且correct面板完成后，用交叉视频study已封存300个实际query（14/21各50state、原C轨迹0/10/20时点）做3300次真实10-flow：
四C的1260 correct/other、两B1260及Source共11条件。复用query、stateless noise及条件映射；不读取新expert标签，不梯度、不重新选点。
保存full50×7 normalized/environment动作和前5 scaled OSC xyz，复算四C动作函数交互及相对B/Source差异。
朝封存旧B630−C420方向的投影是连接前段因果结果的辅助读数，不是expert目标，也不是修复的必要充分条件。
若成功改善而早期函数不变，须修订“早期接近解释了本轮收益”；若函数变了而闭环未改善，不能称问题已解决。

## 6. 实现、资源与验收

Sol在独立开发worktree从最新main集成；主讨论拥有design/spec/metadata及主线状态，Sol拥有训练配置、实现、测试、run产物。
复用`writer/conditional_contract.py`、`learning_data.py`、`supervised.py`、训练/物化/恢复和canonical evaluator。
把本研究的显式pool和schema注册到共享路径；不能删旧assert、伪造旧study、复制完整trainer或修改旧封存spec。
源检查仍要求原B/C拓扑与目标一致；source checkpoint和normalization内容不变，仅新data/eval authority允许已审计辅助任务。
支持任务评测需使用完整38-target银行和正确global/local身份，不绕过meta subset的配对检查。
保留结构性改动应用code-architecture-gate，合理复用/抽取，不以新研究为由再造整套数据/评测框架。

CPU验证应直接覆盖：四池/信息墙、共同26和B/C共享事件、权重与每7步等权、预算/身份/梯度模式、
checkpoint恢复、50视频映射、support subset/400步horizon、固定cases和统计对比。复用现有测试，不按数量造测试。
实际GPU smoke每臂连续4宏步与2→4完整恢复；验证source冻结、后续活动路径非零梯度、形状finite、B对teacher编号不变、
真实LoRA→物化→GPU evaluator接口。最长fit视频仍global29/demo0原347frames，stride5含末帧71帧；验证fullH50峰值。
至少各一条新white-bowl/pan数据走真实训练与一次support评测smoke，避免只测旧task掩盖metadata/资产身份错误。
smoke独立且不可作formal初始化/成绩；正常低位差异接受，不扩大dtype或逐bit追查。

资源合同用formal-training-launch技能。原登记要求正式六臂及新闭环/函数预测来自同一实现commit；
2026-09-24正式闭环尚未启动时，发现该提交的普通capture未实现登记的逐控制步对象/EEF/夹爪，
且阶段谓词仅对full cases启用。按下述显式工程修订实行**分阶段唯一clean pushed detached commit**：

- 六臂训练、完整恢复、Writer物化及3300函数预测仍固定
  `7dc95edbba00cf61439700d77fb321eb8df95c07`，原树
  `/data1/user/ymdai/projects/EMBER-relational-support-formal`保持只读；未启动臂也不切训练实现，不重训。
- 全部新正式闭环，包括六臂两个固定节点、controls、support和Source，统一使用一个新增的评测冻结提交E，
  树`/data1/user/ymdai/projects/EMBER-relational-support-evaluation-formal`。
  E从最新main集成，精确commit在首个正式评测前写入研究根的capture修订launch合同；不能按臂或节点混用评测实现。
- E只允许修改被动轨迹采集、对应prepare/resume/row验收与必要测试/文档；训练配置、Source、Writer、
  sampler/优化器/损失、LoRA物化与应用、policy forward、动作预处理/后处理/执行、RNG、环境reset/settling、
  success/horizon/排队规则均保持。若实际修复需要越过该边界，暂停受影响评测并向主讨论报告。
- `training_commit`、bank物化来源和`evaluation_commit`分别如实记入run/完成provenance；
  不追改原训练合同，不把两个阶段冒称同一实现提交。旧五批冻结树/原件不动。

采集修订不接受原件缺项，也不重跑正式闭环补采集。其原21868预算已由本设计开头的独立成本修订收缩为9932，
82个full cases、信息墙、主要对比、bootstrap和资源上限保持；不把规模修订混称为被动采集代码修复。
采集语义为settling结束时t=0与每个实际`env.step(action)`后各一次：T条7维实际动作对应T+1条
物体body位置、EEF位置/姿态、夹爪及BDDL谓词；不称MuJoCo内部每个积分步采样。
对象身份从各task实际环境/BDDL注册取得，不复用旧两任务的硬编码roles，也不以8D state推算对象轨迹。
所有compact/full行均须有这些原件、终止前缀一致和路径索引；full仍仅登记82案例。环境privileged读数仅写结果，
不得进入policy/Writer条件、梯度或checkpoint选择。注册selection、prepare、resume与逐行验收须显式拒绝缺项，不能静默降级。

实施前后用一个最小、真实的重复动作流证明被动读取不增加`env.step`、不消耗policy RNG、不改变动作和终止语义；
覆盖Source无adapter、B、C以及compact/full和nonheld support的接口。新GPU工程smoke最多12条episode，
使用已有smoke模型/Source、已授权训练任务和固定state；独立标明工程用途，不用于科学分数、选点或正式初始化。
Sol须事先登记smoke病例及观测目标，复用现有smoke资产，不为低位数值一致反复重跑。
实际记录采集开销/每行大小并重估峰值；无合格GPU余量时先做CPU和环境检查，不打断正式训练腾卡。
全部针对性检查通过、E集成push并冻结、原件修订合同齐全后，按既有授权自主启动全部新正式评测，无需再次审批。

研究根`/data0/user/ymdai/ember_runs/relational_support_causality_20260924`；先在strg01检查data0/data1独立quota、
共享容量和实际个人用量，再建大输出。初估banks80＋完整checkpoint10＋闭环25＋临时5≈120GiB，硬上限128GiB；
新增data1开发/冻结代码≤1GiB。已有一份C held400 bank约2GiB、完整C/B checkpoint约142/95MiB，是初估依据，启动前需实测细化。
不复制source/dataset/tokenizer/assets，不以共享df替代quota，不删旧原件腾空间。

本批同时最多6物理GPU，所有训练、物化、评测统一计数；每launch/resume两节点live identity/owner准入，
仍服从项目更严限制。C在2/3/4卡中按实际吞吐profile选**四臂同一world size**，B在1/2卡中选两臂同一world size；
正式前登记每臂拓扑，resume锁定，不跨节点DDP。不等凑卡、不dummy占卡；物理batch可按峰值优化但不改逻辑流/权重。
保留NCCL_P2P_DISABLE=1、GPU-local NUMA与deferred NCCL。精确命令/env/资源/ETA/峰值预算写一份launch_contract。
全部检查通过、main集成push并冻结后发简短ready即可启动，**无需再等Owner或主讨论作例行批准**。

遇到quota/GPU不可核验、OOM/nonfinite、无有效梯度、恢复失败、身份/信息墙/数据等价问题，停止受影响launch，保留原件并回报确切原因；
普通工程修复自行完成，不热改formal运行树。科学参数要变化则先回主讨论，不自行加臂/改ID/扩窗口。
长任务一次持续等待退出/完成事件，不固定轮询日志、checkpoint或共享cache；故障/调度时才针对性检查。

## 7. 完成信号和主讨论裁决

输出唯一completion、完整launch/worker退出回执、每臂实际5040events/141120queries、12完整checkpoint、全部面板逐行原件、
1050/1260配对结果、事前固定1260记录、controls、支持任务成绩、配对/相邻RGL、预定bootstrap、3300预测及82case索引、实际存储。
统计代码放run的launch/analysis owner，主讨论能从原行重算；不要只给排名或面向Owner长报告。
派发/接收凭据登记实际双方UUID，Sol完成后主动Queue回主讨论，停止新增实验，等待下一份具体任务。

主讨论持续负责核验与解释：有交互不等于统一根因，有数据收益不等于视频收益，单点收益不等于能力保持。
信息量边界：核心Goal21交互仍只有50个配对初始化、一个训练seed；反复评早期节点不会增加独立训练重复。
本批可区分具体关系支持预测、一般数据效应和支持操作尚未习得，不能一次裁决所有架构/训练/数据根因。
所有9932条是当前事前确定的有界证据集，不因中途分数好坏再改节点、任务、视频controls或补回旧曲线。
9932限制本批自动执行，不是永久禁止追加评测。主讨论分析后若发现会影响下一步判断的具体证据缺口，
可按已有授权登记有界补测，说明竞争解释、所需新增信息、追加面板/预算与停止条件，再派发Sol。
保留当前原件、关键checkpoint和原主比较；补测明确标为后续验证，不自动补完整矩阵或为择优追逐分数。
若本机制非通过，记录失败的具体预测与操作习得情况，按证据决定功能信用或架构机制干预；不自动换成更多任务/更大rank/更久训练。
批次结束是执行者的边界，不是主讨论停止分析、等待Owner催促的理由。
