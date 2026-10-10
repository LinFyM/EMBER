# EMBER：教学条件参数生成与实践决策的专家审阅稿

日期：2026-10-10。状态：候选算法审阅，尚未实现，不是active design或实验启动合同。
论证基础为32a8b5cd中的讨论稿§9–10；本次补充弱起点、原样父策略试做、预算截断、经历来源与历史问题覆盖。
本目录是本次审阅入口；此前20261008_method_rethink目录及讨论稿§3–9保留各时点的历史，不作为并行候选。

Owner本次问：不从强起点是否也可以训练；新设计是否已没有逻辑或推导缺项，且已尽可能解决允许交互后遇到的问题。
主讨论的回答不是两个无条件的“是”：强T/MT起点不是数学必要条件；几项明确缺项已修正，但有效提案、视频指导、
能力传递等核心命题尚未解决。请独立审阅这项判断，主动寻找反例，不把本文当成已获认证的设计。
可直接使用[专家提示词](EXPERT_PROMPT.md)；专家由Owner自行联系，本次没有向外发送消息。

## 1. 阅读顺序与材料边界

1. 本文：完整当前候选、数学条件、弱起点解释、历史问题覆盖和审阅问题。
2. [长期要求](../../current_owner_requirements.md)、[科学概念](../../concept.md)、[AGENTS](../../../AGENTS.md)：
   以Owner最新信息来源和方法边界为准；旧固定课程、固定次数和历史门槛不自动恢复。
3. [连续推导](../../analyses/architecture_rethink_discussion_20261010.md)：§9.3–9.5说明监督与参数生成，
   §10为当前完整组织，§11为本次澄清；§3–8是曾提出后被修正的论证。
4. 第7节列出的交互后原件；必要时沿[研究历史](../../research_history.md)核对最近似的更早完整机制。

当前源码的experience_compiler仍是上一轮功能导数修订方法。它用于核对历史事实和可复用接口，
**没有实现本稿的新G或类别决策pi**。不能运行旧CLI后宣称验证了本方案。
远程包含设计、源码、分析和部分小型证据；不包含全部模型、数据、视频、raw rows和大张量。
未取得的原件或未运行的检查请明确，不以报告转述冒充重新计算。

## 2. 研究对象与不可改变的边界

目标是把exact language L、action-hidden正确教学V和有预算的自身实践，编译为一套完整固定LoRA，
使冻结source在未参与该condition适应的新初态闭环完成任务。当前首先追求明显超过强MT；
视频提供实际操作知识仍是研究核心，不能用额外任务内学习的收益代替视频贡献。

- source来自generic pi05_base及经过重合任务排除的LIBERO-90 Source-71；不用读过目标40动作的pi05_libero。
  source基础权重和原生图文prefix冻结，动作/state normalization固定。
- 目标40任务使用固定24/8/8划分；本候选共享训练仍限24 train＋12个已审计support，共36任务。
  [协议](../../../configs/libero_24_8_8_coverage_v1/protocol.json)和
  [既有36任务定义](../../../configs/libero_24_8_8_coverage_v1/writer_task_diversity.json)提供数据身份，
  其中旧训练超参数、选点和课程不继承为新合同。更多episode不等于更多独立meta tasks。
- 教学只输入RGB和exact language；teacher动作、state/proprio、物体真值、reward、terminal、task ID、
  文件名和特权policy outcome均不输入。合法non-held标签可以监督共享模型，不成为新condition输入。
- 自身真实交互的RGB/proprio、实际动作、环境状态、物体位姿、接触、reward/success/terminal可以用于适应。
  预测动作、视觉推断和真实环境事实必须区分。最终评测反馈不参与适应、选择或共享训练。
- 明确排除新condition的“恢复/合成/搜索/采集动作轨迹，再用这些轨迹微调最终LoRA”。
  共享训练期以合法动作/功能监督训练参数生成器或训练教师允许；本候选新condition不运行该教师或动作拟合。
- 最终只有一套完整38-target LoRA。当前候选rank128，所有A/B均可输出；不平均候选、不融合checkpoint、不部署第二expert。
  rank128、强起点、按episode修订和冻结G的课程是本候选选择，不是永久Owner约束。
- 当前只讨论K1，双相机不等于K2。stride5保序；不平均raw frames/features或最终LoRA，不挑视频。
- 原生执行为render256/model224、双相机180度rotate、state8/action7、完整50×32 latent与10步flow，
  每次执行前5动作后重规划，dummy settling10，成功终止，suite horizon220/280/300/520。
  正式结论仍须single-checkpoint strict paired400，50条teacher/task各一次及固定配对；不以候选union选模型。
- 历史Test暴露按原协议保留，本次未新使用Test。shuffled/reversed只作最终冻结后的时序检查，不进入训练或架构修正。

## 3. 当前完整Writer

记Lambda为38处完整(A_l,B_l)，source层实际作用为W_l h_l+B_l A_l h_l；既有缩放并入其冻结执行约定。
T为已有合法视频Writer，MT为同source的强共同LoRA。G_psi为新参数提案生成器，pi_theta为新编译决策策略。
H仅记录该condition已有真实经历及其参数身份。共享训练和新condition的可训练状态不得混淆。

### 3.1 教学表示与自身事实

真实双相机冻结patch特征Z及T的原生特征保序进入Reader，得到教学memory m_t。
合法训练状态可监督相对位置变化、夹爪变化、物体共动等可测事实的分布；只有能取得的接触才标注接触。
事实预测保留不确定性，原图特征继续存在，不设置必须精确恢复全部3D或操作程序的硬瓶颈。

自身事件e_i含前后RGB、proprio、实际动作/mask、状态/接触、结果及产生它的完整参数版本。
自身查询读取整段教学memory，不按归一化时间强对齐；重试、失败与不同episode保留顺序。
事实辅助只监督“发生了什么”的部分，不能独自认证知道怎样修LoRA。

### 3.2 G：完整因子的联合条件生成

G读取(V,L,H,Lambda_parent,xi)，直接生成全部76个因子的修改。
因子按target、A/B、rank和数值坐标分块，保留当前值和噪声变量；块间交换信息，并读取教学/实践memory。
输出不经固定低维码或单位置动作Jacobian展开；各块也不能独立采样后便称为联合控制器。
这里描述了计算职责与数据依赖，具体分块、耦合层、宽度及吞吐尚未冻结，应接受专家质疑。

候选监督是实际教师编辑delta_Lambda。令S为只从合法训练数据确定的固定正因子坐标尺度，d_star=S^{-1}delta_Lambda：

    xi ~ N(0,I), t ~ Uniform(0,1)
    x_t = (1-t)xi + t d_star
    L_G = E ||v_psi(x_t,t,V,L,H,Lambda_parent) - (d_star-xi)||^2
    Lambda_candidate = Lambda_parent + S * IntegratedFlow_psi(xi | V,L,H,Lambda_parent)

这是参数空间的条件Flow Matching，不是生成机器人动作轨迹后拟合LoRA。
每个样本是一整套协调因子；多样本只形成候选，不求LoRA均值。
参数loss小、形式全支撑或输出全维均不保证有用控制；A/B重参数化、目标多模态及有限教师样本仍可能使学习困难。

### 3.3 pi：依据教学、经历和候选实际作用作决策

对预算允许的真实自身观测集合，每项候选运行完整原生10-flow：

    a_ik = F(Lambda_k, L, actual_RGB_i, actual_proprio_i, saved_noise_i)
    d_ik = a_ik - F(Lambda_parent, same_actual_inputs_i)

候选比较使用同一noise；这些是动作预测，不是实际转移。H为空时该集合为空，不补造图像或state。
决策器同时读取完整因子的带地址块表示，不假定有限观测上的响应已是策略充分统计。
将e_i、d_ik及候选块形成q_ik，再查询有序教学：

    alpha_ikt = softmax_t((Wq q_ik)^T(Wk m_t))
    c_ik = sum_t alpha_ikt Wv m_t
    logits = Head_theta(ordered_history({q_ik,c_ik}), candidate_blocks, remaining_budget)

候选按集合处理；编号不携带task或教师身份。注意力差异不等于理解，必须由实际后果学习检验。
G的全部条件映射归psi；pi有独立可训练读取/融合/概率头，只在冻结边界复用特征。
不能更新theta时同时改变G的条件编码，却仍声称G固定。

### 3.4 新condition编译伪代码

    freeze source, T, MT, G_psi, pi_theta
    H = empty
    P = {T(V,L), MT}                  # 当前候选的强起点选择；弱起点见第5节
    while registered resources allow another practice:
        choice ~ pi_theta(STOP or parent in P | V,L,H,P,budget)
        log the actual decision probability
        if choice is STOP: break
        parent = choice
        proposals = finite samples from frozen G_psi(V,L,H,parent)
        charge all attempts; record invalid samples without silent resampling
        C = {unchanged parent} union valid proposals
        selected ~ pi_theta(C | V,L,H,candidate_parameters_and_predictions,budget)
        log the actual decision probability
        practice selected with one fixed full LoRA
        append actual observations/actions/outcomes and parameter identity to H
        retain selected in P
    final ~ pi_theta(P | V,L,H,P,budget)
    log final-selection probability; freeze final
    exit all adaptation helpers; evaluate final on new initial states

原样父策略在实践集合中是显式选项，不依赖连续G恰好生成零编辑。
这项边界在本次整理补充：否则“能最终选择MT”并不等于“能先直接试做MT”。
新提案全无效也不能无限重采；父策略仍可选择，预算不足则按固定规则结束。
success、环境horizon和预算截断分开记录。成功只是证据，不自动认证泛化或强制停止；
较差中间点可保留作父策略，不设置逐步必须涨分的门。
资源规则使过程有限，不预设观看/实践轮数。当前目标先在硬预算内最大化成功，尚未证明最佳早停或最少读取。
训练与编译采用同一随机决策规则；换成argmax是另一个执行规则，需要单独承担效果判断。

## 4. 共享训练与完整信用

### 4.1 训练教师与G监督

只在授权non-held任务，以实际选择的Lambda_in父策略和真实历史H形成训练事件。
H每段标注真正产生它的参数，可以来自不同历史候选；选择较早父策略不意味着其产生了全部H。
训练教师在独立参数副本利用合法跨episode动作/功能监督产生Lambda_out，并通过独立初态的真实执行核对收益和损失。
若查询自身访态上的专家，必须验证专家在这些状态的适用性；不能从原初态成功率推出普遍恢复能力。
旧base专家可以在经核对时提供训练用功能，但不能直接将其adapter挂到当前source上。

    D_edit = {(V,L,H,Lambda_in,delta_Lambda,paired_outcomes,provenance)}

搜索/拟合数据与效果确认数据分开；失败、全部成本、任务覆盖和选择偏差保留。
没有取得有效编辑的任务不造正标签，不将编辑更多的task隐式赋更大权重；既有36任务等权须真实承接。
已知四个对齐专家只有局部正例，不等于已经有覆盖充分的D_edit。
学生实际参数/访态应进入后续教师数据；条件监督可以明确复用旧事件，但不能将旧事件称为新版本on-policy梯度。

这一教师的具体优化配方尚未冻结，可取得的有效编辑率尚未测得；这是实质设计/证据缺口，不是已存在的oracle。
监督G加事实辅助形成初始提案能力；密集参数梯度不由二值回报或旧P J^T路径提供。

### 4.2 候选后果监督

在同一合法X=(V,L,H,Lambda_parent)下，由固定G形成有限集合C；各候选在同组、未进入H的新初态实际执行。
Rhat_k只进入目标，不进入本次选择器输入。包括父策略和失败项，不只保留有利比较。

    J_local(theta;X,C) = sum_k pi_theta(k|X,C) Rhat_k
    two candidates: dJ_local/dlogit = p(1-p)(Rhat_1-Rhat_2)

真实能力差为教学/自身/候选的联合读取提供用途标签；相同能力的风格不需强行区分，全失败不制造正确方向。
有限query仍有噪声；这只训练一次最终选择，不等于已经学会有探索价值的实践。

### 4.3 完整适应回报

随后固定G及全部条件编码、T、MT、source和资源规则，只训练pi。
把全部可学习编译决策记为a_j、决策前合法原始信息记为I_j，其余实际生成/执行机制为K：

    p_theta(h) = p(initial) product_j pi_theta(a_j|I_j) K(next_j|I_j,a_j)
    J(theta) = E_h R_query(Lambda_final(h))
    grad_theta J = E[sum_j (R_query-b_j(I_j)) grad_theta log pi_theta(a_j|I_j)]

query在最终LoRA固定后使用独立non-held初态，执行与正式部署一致的官方ODE；不为动作log-prob另改机器人控制核。
当给定历史后的K不含theta、所有可学习选择均计入、支持及求导条件成立时，完整历史分布的信用已包含在score中。
策略内部对历史的编码正常求导；原始环境事实停止梯度。

必要条件包括：同版本完整采集与更新；抽样值不作错误的路径梯度；父策略、实践、停止、最终选择全部记分；
baseline在当前决策前确定，不由该次选择或未来结果构造，且不以共享actor参数额外引入未声明目标；
旧路径未经校正不能作新theta的on-policy数据；改变G后须取得匹配新版本的学习数据。
本文没有推导一个同时无偏更新psi与theta的联合外层梯度，不能把分阶段组织冒称为该结论。

类别logit的score平方范数至多2，只避免了直接给全部因子加独立探索噪声的显式维数问题；
不约束神经参数Jacobian、长过程方差、稀疏成功或有限任务泛化。冻结G也可能引出决策策略变化后的候选覆盖缺口。
这几项是实质风险，不因概率恒等式正确而消失。

## 5. 第一项问题：没有强起点是否可训练

可以定义并训练，但没有实证保证。把P的初始集合改为恒等/弱完整LoRA，公式中的初始分布随之改变；
只要该初始规则不含当前theta，完整决策score仍成立。G与pi的共享参数可以fresh，合法参数监督可在回报全失败时训练G。
实际条件是D_edit和后果训练覆盖这些弱父策略及其H；不能沿用只在强起点附近学到的G便宣称支持弱起点。

必须分别说明三类先验：执行LoRA起点、共享训练教师/标签、预训练source/特征。
本结论不意味着从完全随机VLA、没有合法监督或没有有效候选的纯稀疏RL也可在当前预算内学成。
当前Reader还复用T特征；去掉初始T候选不等于全系统无T，完全去T需重新建立并训练合法source特征上的Reader。
若声称不借助强执行起点，不应仍把强T/MT保留为可选最终答案。

恒等LoRA满足BA=0。通过原生功能损失训练教师时，A=B=0会令两个因子的直接一阶梯度均为零；
非零A、B=0可保持相同恒等函数而避开这个死点。G的因子监督不具有完全相同的梯度限制。
本次仅澄清必要条件，没有自动把强起点候选改成弱起点实验。

## 6. 第二项问题：哪些已修正，哪些仍未解决

不能认证“逻辑上绝无问题、没有未具体化之处、所有交互后困难均已解决”。
更窄而可辩护的判断是：目标、信息来源和完整信用可在声明条件下对应；若干已知缺项已在算法定义中修正。
下面的修正均尚未实现，也不是历史结果的唯一根因归因。

| 交互后已知问题 | 新设计实际改变 | 当前能作的判断 |
| --- | --- | --- |
| 旧experience省略实践分布导数；functional只穿过当前编辑 | 对完整父策略/实践/停止/最终选择记score，G全部映射固定 | 条件式数学修正；不保证低方差 |
| SDE或动作噪声PG与正式原生ODE不同 | 随机性放在编译决策，query与最终控制使用同一ODE | 移除这项执行目标差异；不证明新RL已有效 |
| 专家FM更好但完整修订丢成功 | 参数拟合只学提案，真实候选后果及完整回报教用途 | 效用目标对齐；有限样本、优化和泛化仍未解决 |
| 功能写入极小、上游梯度弱 | 密集完整编辑监督生成有限修改，回报改变选用概率 | 撤换旧衰减路径；G仍可能无效或近identity |
| 单位置7维拉回不能协调某些跨状态改动 | 完整A/B联合生成，无固定该局部像空间 | 去掉特定硬限制；不证明实际控制族覆盖 |
| 真实参数入口与非零梯度仍没带来净修订 | 明确实际教师编辑、候选可用能力和最终选择损失 | 能力传递仍未解决，不能靠全维输出宣称解决 |
| 编辑/递推有效曝光不足 | 教师数据覆盖学生实际Lambda/H，保持真实任务权重 | 新覆盖尚不存在；旧批任务等权已成立，不能冒称首次修复 |
| 训练教师在失败访态不普遍有效 | 要求当前source兼容和真实独立执行核对 | 合格教师与足够有益编辑仍未建立 |
| 视频/经历改变行为但有得有失 | 视频与候选作用共同读取，由真实用途训练 | 明确学习压力；视频指导与迁移仍未解决 |
| 起点漂移、最终丢已有成功 | 冻结并保留T/MT，可原样实践及最终选择 | 起点参数漂移移除；误选与能力丢失仍可能发生 |
| success先停缩窄修订经验 | success进入H，继续/停止成为完整决策 | 去掉该固定门；最优探索/早停/读取效率未解决 |

当前首要未知是有益教师编辑能否形成、G能否保留其控制、pi能否从视频和经历学到可迁移用途。
特权监督可合法使用，不等于所需修改必然可由V/L/H辨认；生成分布可以表达歧义，不能创造缺失信息。
无视频强解存在本身不是理论错误；但若最终只学会语言记忆或总选MT，不能据优化正确宣称EMBER成功。

能力分解只用于定位，不能用于最终评测选策略：对保留集合C的真实新初态成功率p_k，

    p_selected-p_MT = (p_T-p_MT) + (max_C p_k-p_T) - (max_C p_k-p_selected)

其中退回已有MT/T与G新增能力必须分开；max_C是不可用于部署的分析量，不是candidate union正式成绩。

## 7. 交互开放后的核心原件

| 原件 | 必须保留的事实与边界 |
| --- | --- |
| [experience设计](../../designs/experience_conditioned_compiler_20261009.md) §4.2；[结果](../../analyses/experience_conditioned_compiler_20261009.md) | 旧法已有全部A/B、真实实践、FM与终局PG；formal135/400对MT153，训练也无净修订；明确采用实践分布semi-gradient |
| [parameter结果](../../analyses/parameter_conditioned_compiler_20261009.md) | 36任务等权、实际梯度已成立；非MT81/1440仅覆盖16任务，不能称充分递推覆盖；180为127/400 |
| [parameter完整后续](../../analyses/parameter_compiler_fullreadout_20261009.md) | 360为130/400，Long1有正例但总体仍弱；不抹去局部增益或原停止 |
| [固定incoming诊断](../../analyses/parameter_edit_credit_diagnostic_20261009.md) | I/E+/E0均13/24；E有5增5失，存在FM更好却丢成功，不支持统一删除E或只停成功后编辑就能解决 |
| [functional设计](../../designs/functional_revision_compiler_20261010.md) §4；[实测](../../analyses/functional_revision_compiler_20261010.md) | 真实VJP/JVP与FM2/PG1已接通；PG用动作扰动且未作PG后能力读出，不能把诊断当学习收益 |
| [functional正式结果](../../analyses/functional_revision_learning_20261010.md) | 143/400对MT153，R127/G16/L26；train29→25对MT27；实际修订极小、非MT覆盖16/18任务，未证明视频/经验因果 |
| [对齐教师](../../analyses/aligned_teacher_recovery_20261008.json) | 四任务只有有限初态/接手正例，不能当全面恢复oracle |

更早的[功能监督复核](../../analyses/flow_supervision_history_20260926.md)、
[Native Correction](../../designs/native_correction_writer_design.md)及[完整历史索引](../../research_history.md)
约束decoder能力传递、关系监督、纠正教师和共享RL的预期。请比较真实算子、标签、梯度消费者和学习条件，不能凭改名清零。
工程层面的统计/缓存/恢复修复已有独立记录；它们不是科学低分的通用解释，也不是新架构已经解决的问题。

## 8. 请专家独立裁决

1. 给出明确等级：算法自洽但待验证、仍有关键设计空缺、或存在逻辑/数学错误；不要默认主讨论正确。
2. 核对完整计算图和信用：theta是否仍通过未记分的通道改变G/K，停止、最终选择、历史编码、baseline、mask和版本条件是否足够。
3. 核对G：完整因子条件Flow Matching是否适合这个监督对象；分块耦合、A/B非唯一性、控制传递、教师标签和可观察条件是否隐藏难题。
   不要只用通用表达能力定理或增加候选数代替解释。
4. 回答弱起点问题，分开“公式可训练”“有足够监督/探索”“有限预算可学成”；区分控制起点、教师和特征先验。
5. 围绕视频指导逐步说明实际特征如何影响参数/决策、什么标签训练该联系、在何种情形会退化成语言或场景记忆。
   不把注意力、可解码事实、wrong变差或参数依赖当作视频增益证明，也不要求排除有用的共同控制解。
6. 核对第6节是否遗漏交互后的主要问题、把未解问题包装成风险已降，或近似重犯旧decoder/纠正/RL路线。
7. 若有硬缺口，给具体反例及最小必要修改；若需要撤换核心，给完整替代原理，而非模块菜单或rank/LR/seed小扫。
   有效修改须继续满足信息墙、共享训练/新condition区别及最终唯一LoRA。
8. 只提出最小而完整的后继验证：教师监督形成、生成学习、后果决策、实际适应、新初态能力与适量视频证据。
   区分候选没有增益与有好候选却选错，包含强MT/T、任务覆盖、得失和成本。不要把全部消融设为开工前置门槛。

请将已成立的推导、需要条件的命题、未定义的设计、经验假说和已失败证据分开。
具体网络/教师配方/预算未冻结是审阅范围，不能全部下放给实现者；无法从现有材料判断时请精确指出缺少什么。
共享教师构建、G学习、候选生成/原生预测、实践、query和正式400都计成本；无实测依据时不要捏造精确ETA。
本次仅审阅，不要求访问私有运行资产、启动训练或联系他人。
