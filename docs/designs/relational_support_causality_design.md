# 任务关系支持的学习干预

2026-09-24，主讨论登记的下一批有界训练；是否active以`progress.md`为准。
机器合同：`configs/relational_support_causality_v1/experiment_spec.json`；完整任务审计、protocol、manifest及evaluation在同目录。

**当前执行修订（2026-09-24 16:39 UTC／北京时间9月25日00:39，仍无本批正式闭环结果）：**
六臂训练与固定1260主问题保持，先完成§5的1500条核心诊断，然后向主讨论回报并停止新增评测。
原21868全矩阵及14:38 UTC的9932方案均由Git/原launch记录保留，不能自动执行；是否追加由实际证据缺口决定。
1260仍是事前固定模型，没有最佳点选择；1050相邻点及全面性能验证暂缓，不能由本阶段声称已通过。
冻结训练配置中的六个`evidence.evaluation_updates`与spec的对应optimization字段保留为原训练元数据；
实际评测节点的唯一authority是`evaluation.executed_updates`，不能据旧训练元数据启动已取消的面板。
本修订允许机械同步评测注册、task/state子集调度、固定1260记录和阶段验收/汇总；这些改动与被动采集修复一起集成到唯一E。
随后核实7dc物化入口不接受本阶段子集，§6已授权把选择/校验修正一并纳入E；训练和Writer生成计算保持原语义。
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
  第一阶段保留它作为能力保持参照；其它held任务若未补评则不声称完整能力保持。

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

主对比是1260时Goal21 correct的交互及配对区间；不搜索其它节点或事后挑交互最大的点。
第一阶段看交互与C11绝对改善是否同时出现、other是否同向。相邻保持暂未检验，阳性只能作为需要进一步验证的局部证据。
若区间跨零，只能说证据不足；若只见主效应，就修订解释，不把任何涨分都归于图连通。
若关键支持任务近乎不成功或显著损害Source能力，明确保留“支持操作尚未习得”；支持任务高分本身也不证明可组合。

最终视频收益需要正确和另一正确视频的绝对能力、相同数据的fresh语言B、同模型wrong对照一起解释。
第一阶段没有wrong，只辨别局部关系支持与条件收益，不能宣称视频因果修复已经验证。
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
- 完整checkpoint每105宏步，共12/臂；当前实际评测只执行1260。保存完整恢复状态及applied LR，不删除原checkpoint。
  窗口不足以声称收敛/容量上限，完成后不因结果差自行续训或扫seed/LR/rank。

## 5. 闭环、controls与学习作用检查

原官方执行合同保持：render256/model224、双相机rotate180、8维state/7维action、10-flow、执行前5再replan、
settling10、成功即停，目标四suite horizon220/280/300/520；新增LIBERO90 support horizon400。
动态队列、long-first和persistent workers复用canonical evaluator，不另写近似环境。

当前只授权以下**第一阶段机制诊断**，不是正式held400性能资格。数据和执行preprocessing不变；
所有模型事前固定1260，不选checkpoint。主问题Goal21保持原50个初态和50条teacher无放回映射，未因缩规模减少主要比较的样本量。

| 面板 | 模型/条件 | task与state | 行数 |
| --- | --- | --- | --- |
| 目标与保持参照correct | 六臂 | Object14、Goal21，各state0..49 | 600 |
| 主任务另一正确视频 | 四C | Goal21，state0..49，原offset17映射 | 200 |
| 新增支持操作 | 六臂 | 每臂原四个support任务，各state0..19 | 480 |
| 新Source目标参照 | Source | Object14、Goal21，各state0..49 | 100 |
| 新Source支持参照 | Source | 原六个新增support任务，各state0..19 | 120 |
| 合计 | 18个面板 | 所有面板固定后才开始读取新结果 | 1500 |

支持20行必须截取原50-state/50-video映射的state0..19，不因缩小面板重新排列teacher；
只是合法50视频池中无重复的20条，不能称使用了全部50条。追加时可补state20..49并复用原20行，无需重跑。
不同task的支持面板不冒称动作/环境逐行配对；同task的Source、B/C和共同支持任务按原state/RNG配对。

保留所有行实际动作、stage predicates、连续对象/EEF/夹爪trace和条件引用。
原固定full病例取当前面板的交集：六correct加Source的14/21×state0/25为28例，四other的Goal21×state0/25为8例，
六臂support各四task state0及Source六task state0为30例，共66例。不为补回其它full病例增加rollout。
不读取新held expert action/state，不读官方Validation/Test，不由局部成绩改变当前训练或第一阶段面板。

**当前不启动：** 1050相邻点、其余held6、seen64、Object14 other、全部wrong、3300个封存query功能预测。
旧六节点和9932自动入口不能绕过当前stage范围启动这些工作，也不预物化暂缓面板的LoRA。
原件、训练checkpoint、映射和原预算记录保留；暂缓项目不是永久禁止，也不是第一阶段完成时必须自动补齐的清单。

**第一阶段分析与验收：**

- 维持Goal21 correct的原1260主交互`C11-C10-C01+C00`、C11-C00、两个因子的条件效应；other报告相同四C对比。
  两任务correct报告B11-B00、同池C-B、差中差及Source R/G/L；other报告同模型correct-other和相应Source配对。
- support逐task报告绝对成功数、Source参照及同task R/G/L；20状态只能初步区分明显未习得和较好表现，
  中间结果或宽区间不能宣布“全部支持充分掌握”。这些检查也不证明所有旧共同任务或全部关系路径已掌握。
- bootstrap20000/seed20260924保持；每个任务按state联合重采样，保持四C/B/Source/video条件配对。
  两任务分别报告，不计算或外推八任务总体、四suite覆盖、相邻稳定性或正式held400分数。
- 输出1500唯一原始行、18个面板回执、固定1260身份、per-task及配对对比、66cases和trace索引；
  第一阶段完成信号为`analysis/stage1_completion.json`，明确deferred工作和未验证主张，不能写整套9932已完成。

**阶段间裁决：** Sol不按中途分数自行扩展，完成1500后主动Queue主讨论并停止新增评测；主讨论核对后决定下一项。
若出现局部绝对收益和预期交互，再针对视频因果、其它任务保持或1050相邻稳定性中最关键的缺口追加；
若支持任务表现差，先定位操作未习得；若支持已有较好表现而主任务没有收益，削弱该具体支持修复预测，
但不直接定罪架构，也不自动补全矩阵。区间宽到影响决策时，明确需要新的初态、独立训练重复还是特定对照，
不得把同一状态/模型反复评测当作独立信息。追加前写明问题、面板/样本、预算与停止条件，保留本阶段主比较。

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

- 六臂训练、完整恢复保持固定；暂缓的函数预测如后续派发也沿用
  `7dc95edbba00cf61439700d77fb321eb8df95c07`，原树
  `/data1/user/ymdai/projects/EMBER-relational-support-formal`保持只读；未启动臂也不切训练实现，不重训。
- 本阶段全部新bank物化和正式闭环，包括六臂1260子集、controls、support和Source闭环，统一使用一个新增冻结提交E，
  树`/data1/user/ymdai/projects/EMBER-relational-support-evaluation-formal`。
  E从最新main集成，精确commit在首个正式bank/评测前写入研究根的修订launch合同；不能按臂或节点混用物化/评测实现。
- E只允许修改被动轨迹采集、当前阶段的物化选择/校验和子集注册/调度/分析、对应prepare/resume/row验收与必要测试/文档；训练配置、Source、Writer、
  sampler/优化器/损失、实际单condition LoRA生成与应用、policy forward、动作预处理/后处理/执行、RNG、环境reset/settling、
  success/horizon/排队规则均保持。若实际修复需要越过该边界，暂停受影响评测并向主讨论报告。
- `training_commit`、bank物化来源和`evaluation_commit`分别如实记入run/完成provenance；
  不追改原训练合同，不把两个阶段冒称同一实现提交。旧五批冻结树/原件不动。

**第一阶段子集物化裁决（2026-09-25，北京时间）：** 冻结7dc的`_validate_conditional_selection`要求完整held8或四support，
并匹配原50个states；`request_init_state_ids`的count-only仅接受10/50。主讨论从冻结树实际复现两类拒绝，
另确认Goal21-only other的声明及paired-correct校验仍依赖旧完整held400。这是阶段入口合同不相容，不是科学阴性。
明确撤销“本阶段实际bank必须从7dc运行”的限制，改为**materialization_commit = evaluation_commit = E**；
训练提交、权重、真实生成计算与原逐条件输入身份不变，不通过预生成暂缓bank来绕过边界。

- 只准许本批schema及`evaluation.stage1.panels`中该arm固定1260的正式模型面板；Source无需Writer bank。
  task/state/condition、role、seed、K1、50视频池、完整checkpoint与信息墙均须显式校验；旧研究及官方Val/Test的准入不放宽。
  复用现有materialization/controls/evaluation owner，按实际请求路径修正解析、选择、封存、加载与prepare/resume；
  不用monkeypatch、伪造完整manifest或新建平行编译器绕开校验，非法面板须在启动GPU前拒绝。
- 令`L(t,s)=G_phi(language_t,V[pi_t(s)])`。缩减只取登记的`(t,s)`子集，不改`G_phi`、`pi_t`或其输入；
  support保持原50排列的前20个state。逐行对照旧完整面板的condition ID、teacher编号、配对other与video ordinal，
  检验这一限制操作不重新抽样，也不改变帧选择、public probe或单condition生成语义。
- Goal21 other仍绑定本臂同一1260的correct bank，允许从Object14/Goal21 correct100中投影出Goal21的50行作为配对参照；
  其checkpoint/source/生成合同、逐state正确视频与other offset17均须校验，不能只把400数量断言删掉。
  correct的Goal21已包含全部50teacher，other按对应condition复用现有LoRA及真实来源，仅重排episode映射，不重复Writer forward。
- 针对性CPU验收覆盖实际JSON请求→selection→bank/诊断声明→evaluator prepare/resume链，
  以及被暂缓held6/support state20..49、wrong、错误arm/checkpoint/映射和官方Val/Test越界的拒绝。
  从冻结7dc生成的原50-state排列表做逐行映射参照，不能只用新版函数自证。生成核心的task-scoped Git diff须确认未改。
  GPU接口复用已登记Source/B/C/support工程smoke及现成smoke资产，把新路径纳入原≤12条episode预算；
  不额外训练、不运行完整旧bank作对照、不为逐bit一致重复生成或重跑闭环。
- 在`launch/stage1_subset_materialization_amendment.json`记录E、训练7dc、映射验收、bank范围/复用、命令和原预算内smoke结果。
  若实现需要改变单condition生成计算或映射，停止受影响bank并回报；窄修正验收并冻结E后自主继续1500阶段。
  其它科学对比、18面板/1500行/66cases/0函数预测、数据墙、GPU/存储上限及阶段结束信号均保持。

采集修订不接受原件缺项，也不重跑正式闭环补采集。其原21868预算随后经9932修订，当前只授权1500条第一阶段；
66个当前full cases、信息墙、Goal21主要对比、bootstrap和资源上限按本版§5，不把规模修订混称为被动采集代码修复。
采集语义为settling结束时t=0与每个实际`env.step(action)`后各一次：T条7维实际动作对应T+1条
物体body位置、EEF位置/姿态、夹爪及BDDL谓词；不称MuJoCo内部每个积分步采样。
对象身份从各task实际环境/BDDL注册取得，不复用旧两任务的硬编码roles，也不以8D state推算对象轨迹。
所有compact/full行均须有这些原件、终止前缀一致和路径索引；full仅登记本阶段66案例。环境privileged读数仅写结果，
不得进入policy/Writer条件、梯度或checkpoint选择。注册selection、prepare、resume与逐行验收须显式拒绝缺项，不能静默降级。

实施前后用一个最小、真实的重复动作流证明被动读取不增加`env.step`、不消耗policy RNG、不改变动作和终止语义；
覆盖Source无adapter、B、C以及compact/full和nonheld support的接口。新GPU工程smoke最多12条episode，
使用已有smoke模型/Source、已授权训练任务和固定state；独立标明工程用途，不用于科学分数、选点或正式初始化。
Sol须事先登记smoke病例及观测目标，复用现有smoke资产，不为低位数值一致反复重跑。
实际记录采集开销/每行大小并重估峰值；无合格GPU余量时先做CPU和环境检查，不打断正式训练腾卡。
全部针对性检查通过、E集成push并冻结、阶段launch合同齐全后，按既有授权自主启动第一阶段，无需再次审批。
Source与已完成训练臂可在剩余臂继续训练时物化/评测：只核验对应臂1260完整checkpoint和launcher exit，
不得为启动一个已就绪面板等待六臂全部训练结束；仍须每launch双节点live准入及合计≤6物理GPU。
利用已结束lane释放的设备，不中断训练、抢占他人任务或改变训练拓扑；保持退出事件接续，不新增进度轮询。

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

输出`analysis/stage1_completion.json`、18面板完整launch/worker退出回执、1500唯一原始行、固定1260记录、
新增支持任务成绩、预定义配对对比/区间、66case/trace索引及实际存储。训练仍按原合同核验六臂5040events/141120queries、
12完整checkpoint；先完成的训练与第一阶段面板可以先准备，但科学阶段裁决等待所有登记面板齐全。
统计代码复用run的launch/analysis owner，主讨论能从原行重算；不为改面板另造并行evaluator。
实际双方UUID、Queue/Steer回执和新阶段合同保存于coordination及launch；Sol完成1500后主动Queue主讨论，停止新增评测。

当前只支持固定两任务和新增支持操作的局部机制判断；视频因果修复、完整held400能力、suite覆盖和相邻稳定性均待相应补证。
主讨论分析后按已有授权决定必要补测，不把批次结束当作停止科学分析的理由，也不自动恢复旧9932/21868矩阵。
