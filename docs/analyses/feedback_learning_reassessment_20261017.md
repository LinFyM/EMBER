# 以真实动作后果学习连接教学和完整参数修订

2026-10-17，主讨论科学议案。

状态：回应Owner“继续想清楚完整方案、暂不派实验”的科学议案。不是active design、launch合同、已实现方法或性能结论。此前whole_method_reassessment_20261017.txt的撤回状态保持；本稿不能恢复该稿的执行资格。native首写有限批已经完成；本稿不产生新执行授权。

## 选择
从冻结raw source和无有效改动的完整38-target/rank8任务LoRA开始，以真实实践校准可学习的动作—物理后果模型，再由教学条件的学习准则通过该模型和当前policy原生导数修订整套LoRA。共享训练同时教会局部修订和修订后完整策略的跨初态能力。最终只保留一套完整LoRA，辅助模型退出。
这不是首次提出H、学习能量、native J或完整回报信用；实际新学习条件是：动作余切必须通过有独立事实监督、且在新condition真实更新的动作后果模型形成。不能以重新组合或未试过证明其优先级。

## 历史约束
1. 此前几个月无实践直接生成的性能路线已被Owner放弃，不恢复为本方案起点。近期G/D的真实反馈方法仍保留175/168对MT153的正例及相邻交换、任务集中限制；旧MT起点的正例不能当作source起点成绩。
2. source/r8参考在训练域选点读出502/576、source218/576；它支持控制容量存在，不证明held迁移，不隐藏选点面板偏差。MT340/576是同36任务历史强参照。
3. N16实践66→77/72/80而独立读出61/71/62对MT67；近期改变参数的105条件实践156→298，query51→53，完整held117→119仍低MT153。不能用局部效果接近视频或实践择优替代新初态能力。
4. FR已做E(c,a)动作余切→native J→LoRA+跨episode FM，结果143，实际参数变化极弱；Experience已有H条件FM与真实回报；source64已有九次整路径PG。不得宣称补上这些原则即是新解法。
5. LocalActionGrounded、LocalField、joint action-effect、物理J probe及Phi/DTW各自有明确负证据。物理J probe的效果预测相对零变化误差下降70.26%，但残差比例.29736未过.25。它支持局部后果可建模的有限依据，不是完整方法资格。在本次核对的这些近邻中，没有已实现本稿的真实转移拟合、condition内校准、教学学习准则与完整LoRA共享学习全部相连的方案；这只是核对范围内的区别。

## 完整回报后的新增约束

本稿形成期间，唯一执行者交付native180完整批并归还canonical写窗口。主讨论读取报告、原始144×8行、现行算子合同，并从每行success独立重建成功集合：native64/source58/reference127/Reader89/first73/full79/random86/functional93；对source R53/G11/L5，对Reader R60/G4/L29，对functional93为R61/G3/L32。target19/96、support45/48，Object0；task0净增2、35净增3等正例保留，Reader task12/14各4→0、20/22各3→0等失去不能藏在总数中。breadth20与source同、低Reader29；参考新增70只保10。

真实FM/完整算子导数/QR-alpha和非identity改动均已经存在；不能再把效果不足简单归于未接梯度、固定A首步死区或参数从未改变。当前证据没有区分压力学习不足、有限source支撑的函数可达性与有限学习量，不支持指定唯一根因，也不自动恢复训练量/尺度小扫。原件：[完整交付](/data1/user/ymdai/ember_runs/source_native_operator_compiler_20261017/first_write_seed7/analysis/engineering_completion.md)。无新GPU/model/env forward。

这降低了把原生写入视作已获控制能力底座的支持。本稿保留J仅作为当前可微policy的参数更新工具，不能将此组件阴性反说成后续成功基础；只在原p头前加f、保持同一静态source-H首写，并没有回应完整缺口。完整提案要求真实模型校准、当前LoRA与新访态上的更新、共享跨初态功能/真实回报学习共同存在。循环与完整回报本身已有旧阴性，不作为新发明。f改变学习依据，不扩大给定J的线性像空间；迭代重算当前J与访态有机会改变可达范围，但没有由这批证据保证。若f局部可逆，分解甚至可以表达旧energy同类方向，其理由是事实监督和真实任务内校准，不是输出维度或数学表达集天然优越。

本稿没有宣称旧失败由动力学误差导致。新增的可失败假设是：有事实约束、能被实践校准的动作后果关系，可以比直接从V/H猜余切更可靠地承接教学条件的修订目标；最终仍须落到完整固定LoRA的广泛控制。native180维持停止，当前没有新active design、训练/评测或自动后继。

## 具体模块和条件内过程
A. 视频记忆M：stride5双相机真实RGB经冻结source视觉特征，再保序编码。保留patch和时序内容，不把整个视频压成固定Phi坐标/DTW距离。exact language用于对象/目标查询，policy仍读exact language；没有独立task-ID或语言到参数的索引分支。视频侧不读teacher动作、state、成功或对象pose。当前讨论K1，不据此宣称dynamic-K已实现。
B. 真实H：当前完整LoRA实际执行产生自身RGB、proprio、执行动作及mask、reward/terminal；允许新增自身物理观测x，包括手端/夹爪、对象/关节状态和接触等。当前原生H缓存没有完备物理标签；不能把policy-code零向量当物理state。既有PhysicalEntities只提供部分位置/朝向/关节结构，速度/接触/逐动作物理后果需要实际采集，不能从teacher标签补。
C. 物理后果模型f_psi(x,a_1:5)：预测实际将执行动作所造成的逐步物理变化及接触分布。包括机器人运动，不能只预测物体位移而令接触前信用全部为零。预测值不是观测事实。f不输入teacher视频/任务ID，也不生成正确动作。
   新condition复制共享f，更新带state-action交互的局部适应块psi_c；损失为真实执行后的物理预测误差。不是仅改与动作无关的偏置。旧行为的动作和后果始终成对；当前LoRA在旧真实观测上的新动作只能用f作反事实预测，不能冒充已经发生。
D. 学习准则E_theta：以自身当前视觉/物理状态、实际变化、模型残差和H为query，读取有序M；对f预测的物理后果输出标量代价。可微候选动作只经f进入E，E没有另一条直接候选动作→余切通路。H中的已执行动作是固定事实条件。视频的贡献是学习哪些后果与操作内容有用；f负责哪些动作可能造成这些后果。注意力位置或物理预测准确不单独构成理解证据。
E. 参数写入：a_i=F_Lambda(o_i,L,epsilon_i)必须用真实RGB/proprio、完整原生prefix及十步flow，取前五动作；J_i是当前完整38处附加权重算子的导数。连续部分p_i=-d_a E_theta(M,x_i,f_psi_c(x_i,a),H)|a_i。于是G=sum_i J_i^T p_i/n，Lambda_new=R8(Lambda+eta G)。R8表示每层完整算子重新表示为rank8，两侧空间可变；可沿现有范围迭代接口构造，不声称是最优SVD。尺度按真实动作响应处理，继承native首写的功能尺度理由，不称新原理。
   该式定义递推方法，不把现有仅source首写的代码宣称为已经完成递推。所有38处共同作用于自身执行hidden；不限定旧参考bank或固定A子空间。rank8、有限支撑点和跨状态Jacobian仍约束可达修改，完整38不等于任意函数修改。
   夹爪部署取sign，不能用普通导数穿过sign。用f分别预测真实开/关模式后果，按raw输出诱导的Bernoulli概率计算期望E并形成模式信用；这是可训练替代目标，最终硬符号控制仍须由真实闭环回报约束。连续接触的后果导数也来自学得模型，不称真实模拟器精确梯度。
F. 执行Lambda_new，获取新H，校准f并重新读V、形成后继完整Lambda。当前参数沿这条真实学习链更新，不以二值成功必须严格增加作为每次更新的接受条件，避免全失败时所有中间修改都被退回；不再训练一个独立候选排序器。轮数由实际预算、信息和表现决定；实践覆盖不同起始布局，所有检查计入适应预算，最终评测场景隔离。最终冻结本链最后实际执行过的完整Lambda；预算尾尚未实际执行的新提案不能当最终产物。共享学习和独立新初态回报必须训练同一出口，不能训练末端却部署oracle最佳点。最终仅source+一套Lambda运行，f、E、H、特权state和修订器全部退出。不在最终执行时看video或继续适应。

## 共享训练的实际标签和梯度消费者
1. f的训练数据是真实(x,a,x')，短真实动作分支约束邻近动作的不同后果。只拟合同一已见动作的值不能识别斜率。共享初始化采用过去转移更新psi_c、另一批后续真实转移评价更新后的预测损失，使模型实际学会校准，而非全局训练后假定随意微调有效。成功和失败转移都有标签。物理模型只接受事实预测信用，不由FM把它变成无物理含义的余切decoder。
2. Reader/E的密集目标：在合法non-held任务上，真实当前Lambda/H经过上面完整编译得到Lambda_new；同task、不同teacher episode的原始动作query只在外层给出原生FM损失。梯度FM→完整Lambda_new→rank8算子/尺度/J→p→E/Reader。动作query、参考答案不进入Compiler条件。当前事件固定incoming/J，对theta可求真实混合导数，不需对source权重求Hessian；此前修订/环境分布不伪称自动被这项事件梯度覆盖。
3. 完整回报目标：在non-held任务上实际运行整条编译过程，然后用未参加该过程适应的新初态评估最终固定LoRA。共享E产生有已知密度的余切提案q_theta(p|M,H,Lambda)，native写入本身给定p后确定；真实回报通过(R-b)sum_j grad_theta log q_theta(p_j|M,H_j,Lambda_j)训练整条提案路径，包括最终退出部署的中间尝试，不能只给最后一次修订信用。H、incoming及样本p按score-function的条件取值处理，不能误将重参数化样本代入logprob消掉均值梯度。f有独立事实学习路径；本条只训练共享准则/读取器。
4. FM提供密集操控学习，完整回报约束真实访问分布、信息获取和跨初态效果；可在同一共享训练中交错，不把旧固定两阶段或固定轮数当硬原则。训练必须真实消费当前模型产生的新参数/H，包括失败与成功、早期与后期；不能只在同一source-H缓存训练后宣称已学多轮反馈。
5. 共享梯度仅来自train24及明确审计授权的non-held任务；新validation/test教学不读离线标签。新condition的自身动作只作为f的输入及真实经历，不成为最终LoRA的动作拟合目标。此方法不准备动作轨迹后BC，不从最终评测反馈产生梯度。

## 为何值得考虑及什么仍是科学未知
任务内失败也提供大量动作后果标签；共享训练负责把教学内容和这些可校准后果连接到完整策略能力，避免任务内全失败时只能等待首次终局成功才有学习信号。它不能保证新task目标或未访问阶段自动学会：E的迁移、f的局部反事实质量、J与rank8的跨状态作用均须真实结果支持。
超过MT的机会来自教学决定操作偏好、实践补上当前动作后果证据、最终参数形成可随自身状态变化的task专化控制；不是因为MT没有闭环RGB反馈，也不假设MT必可由本rank8参数族复现。已有reference容量和G/D增益支持存在余量，近期实践/query分离则要求完整共享学习真正针对泛化。
本稿不把动力学不准定为旧失败唯一根因。若已有f在当前状态本来准确，校准应接近不动；不能为制造H作用而人为要求改模型。若只改善f/实践而完整LoRA不改善，则这条分解未解决EMBER，而不能改口以局部效果为目标。现有native首写批至多回答写入/共享FM的有限问题，不能承担本稿完整方法证明。

## 依据入口
docs/current_owner_requirements.md 顶部source/r8与无实践路线取舍。
docs/analyses/whole_method_reassessment_20261015.md §§2–7：历史正负、Q斜率/全失败反例、跨状态限制。
docs/analyses/source_control_space_learning_20261015.md §§7、21及实践/held完整读出：容量、既有H功能学习、实践/query分离。
docs/designs/functional_revision_compiler_20261010.md §§3–4：旧E(c,a)、J、FM和实际回报合同。
docs/analyses/video_effect_learning_chain_20261015.md：旧Phi/DTW及真实动作分支写入链。
docs/analyses/operation_semantics_feasibility.md：真实gripper sign与物理J原件。
docs/designs/source_native_operator_compiler_20261017.md §§3–5：当前源首写、完整算子rank8范围和功能尺度。
GrBAL https://arxiv.org/abs/1803.11347 ：过去数据适应、未来预测监督的训练组织；其执行MPC不证明EMBER固定LoRA方案。
