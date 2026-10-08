# 讨论衔接：视频到LoRA编译的重新设计

截至2026-10-09。本文综合已执行证据、七轮专家建议及Owner最新排除条件，供新专家独立设计完整方法。
当前没有active design、新实现或实验；本次任务是架构、数学推导与详细推进规划，唯一当前委托见[提示词](EXPERT_PROMPT_FRESH.md)。
历史建议、主讨论判断与已执行结果须分别解释，不从旧“当前／接受／下一步”恢复方法或运行。

## 1. 最新方法边界与本次目标

Owner在第七轮回复后明确：**任何先准备动作轨迹数据，再通过微调得到最终LoRA的路线，都不接受。**
轨迹来自视频恢复、模型合成、规划／搜索还是自身真实交互，数据准备器是否经过共享学习，以及采集／微调是否交错，均不豁免。
先编译初始LoRA再用动作数据做末端微调也在排除范围。性能更好、视频确有贡献或最终只留一套LoRA，都不能改变这一判定。
原话见[Owner记录第6节](OWNER_FOLLOWUP_REQUESTS.md)，稳定要求见[最新方法边界](../../current_owner_requirements.md)。

EMBER要研究的是把exact language与action-hidden教学视频的操作知识编译为完整task-conditioned LoRA，
让冻结source在未参与适应的新初态执行。共享Writer可以使用合法动作与功能监督训练，这与面对新condition准备轨迹再微调策略不同。
自身真实实践、反馈、多轮观看和LoRA修订仍可用于编译；不重新规定单网络、单次前向或一律禁止梯度。
具体方法须给出视频／经验到参数生成或修订的实际计算，不能以可学习数据生成器、蒸馏或经验吸收的名称恢复被排除的机制。

第五至第七轮的“教学引导行为发现→有效动作经验→直接训练最终LoRA”主线已被否决，完整组合没有实施。
这是一项方法边界裁决，不是新实验阴性。主讨论此前认可其可实施性，并在首次质疑后仍为可学习数据准备保留例外，现已撤回；
责任和适用范围见[findings](../../../findings.md)§406。新专家无需维护或逐项修补上述方案，也不因该否决自动恢复旧T/C。

当前第一阶段仍以完整固定LoRA明显超过强MT为主要性能目标，同时逐步建立稳定性、覆盖和有益视频增量。
无需前置全部controls或先胜同预算充分语言适应；视频确实提供操作知识仍是方法核心，超过静态MT本身不足以证明。
本次要求先有完整架构及落实到实际特征、算子、标签和消费者的数学推导，再有渐进详细规划及关键困难处理原则。

## 2. 当前信息边界：合法证据与被排除的参数形成方式分开

| 阶段／来源 | 可用信息与用途 | 边界 |
| --- | --- | --- |
| 共享训练 | 合法训练任务的动作、状态、关系、奖励等监督，训练共享Writer及必要模型 | 主功能监督守同task跨episode；不混入held离线标签；新增meta tasks须审计与allowlist |
| validation离线教学 | 指定action-hidden视频及exact language | 不读取示范episode的动作、proprio/state、奖励等配套标签，不另调该task离线训练资料 |
| validation自身交互 | 实际动作、RGB/proprio、环境状态、位姿、接触、reward/success/terminal | 可以帮助理解教学、评价和编译修订；不能准备成动作训练集后微调最终LoRA |
| 模型内部计算 | 既有模型特征、完整action horizon、hidden、候选与预测 | 区分推断和真实后果，不虚构teacher proprio或原生prefix，不把预测当teacher标签 |
| 最终评测 | 冻结source＋同一套完整38-target LoRA，原政策合法的自身观测和语言 | 使用未参与该condition适应的新初态；不再更新、读取教学、运行搜索／外部阶段选择或读取特权状态；Test默认封存 |

Source基础权重及原生图文prefix、source normalization保持冻结。强MT是有价值的起点，其合法作用由新方法决定。
每个condition最终仅一套完整LoRA，不挑视频、不平均最终LoRA、不部署第二expert adapter。
教学stride5、单视频保序、多视频集合聚合合同保持；不因旧方法rank或memory token选型限制新结构。
观看、实践和修订可以多轮；交互预算、reset、停止与最终评测分开登记。有交互不得称zero-interaction。
合法信息可用不意味着每个新任务都要重训一套感知、动力学和评价器；必要性与重复成本由完整机制解释。

## 3. 已有系统与基准，避免重新发明已经做过的工作

Source由generic `lerobot/pi05_base`在经过specification审计的LIBERO-90中71 tasks×50成功episodes建立，随后冻结；
不是读过目标40动作的`pi05_libero`。目标40含Spatial/Object/Goal/Long四suite，当前显式24 train/8 validation/8 test。
T/C主要在train24＋12个审计过的support tasks学习，共36个视频→适配映射；source见过71个任务不等于Writer学过71个条件映射。
后续曾有support71支线，实际曝光与完成边界见原件，不把“更多同task视频”等同更多独立规则。

教学保留视频内部顺序，stride5及末帧。最终策略双相机render256/model224、双图180度rotate，8维state与7维action，
10步flow生成50-action horizon，每次执行前5步再观测，settling10，suite horizon为220/280/300/520。
完整38-target LoRA覆盖18层Q/V及action_in/out；当前T/C rank128。共享Writer总参数、输出rank、共同子空间是不同概念，
owner没有要求当前阶段匹配MT参数量，也没有要求rank扫描。

原T的实际学习链并非“视频表示没有收到动作信用”：

1. 双RGB和exact language走真实冻结图文prefix，teacher prompt省略State段；公共LoRA参与native读取。
   固定50×32 Gaussian probe、flow time=1得到每帧完整H与各层实际projection输入X。
2. 各层 `K=normalize(A X)`，Value由K、H及相邻H变化构成；顺序delta写入从0累积M。
   最终为完整 `(A, B0+M)`，实际执行时M作用于机器人自己的、随状态与flow变化的hidden。
3. 同task、不同episode的真实RGB/state/action query通过同一生成LoRA做完整FM；同一参数版本重放编译与native信用。
   常用每macro4tasks×28queries=112，真实梯度到达公共参数与视频读写模块。

C进一步用4层因果解释器处理完整时序/horizon，视频修订A得到A0+S，再用最终A编译B0+M；真实FM共同训练。
这些方法已经包含完整A/B、时序、真实自身状态调用和功能信用。新的方法不能只靠重新命名这些性质证明进步。
同时，H是模型计算响应，固定probe下的H差分不是teacher动作或已经理解的接触效果。
详细算子与实际代码身份见[MECHANISM](../20261007_research_reassessment/MECHANISM.md)。

## 4. 最能约束新判断的正反证据

以下只在各自明确的配对协议内比较；旧协议的143、151、174等不能直接与当前153排名。

| 已执行工作 | 实际结果 | 支持什么、没有证明什么 |
| --- | --- | --- |
| 当前强T与MT | T2340 161/400，MT300 153；T相对MT保留117、新增44、丢失36。T较晚相邻点147/154/160/154/159 | 有真实新能力，但得失大、净增小，未形成明显且广泛的优势 |
| T换视频与删分支 | correct/同task另一视频/public-only/wrong=161/150/103/65；主要增量集中task3/11，task23/39都0 | 教学条件确有作用；public-only不是单独训练的强语言baseline，wrong变差也不等于正确视频带来可迁移知识 |
| 条件读写C | held450/630/900=137/154/140；seen91→99→115/144 | 已见映射学习改善，没有自然转为held控制；不能把局部可学性拼成泛化资格 |
| 第三轮建议的Within/Product | 同T起点、同边际/query/噪声/曝光，各216updates、24,192queries；correct/other为147/148与152/151，原T161/150 | 配对改变有有限相对差额，没有总体修复；降低这个具体假说优先级，不宣布所有跨情境学习不可能 |
| 当前source的四个独立任务专家 | 初态12/16；从T执行50步后接手9/16，对T救6、丢3，task29接手0/4 | 确有可学控制增量，尚非可靠、覆盖充分的纠正教师；后续视频学生蒸馏没有实施 |
| 共享Writer RL | 72updates、1,152个train on-policy episodes；官方63/72为154/158，T起点161；确有非零return信用 | 不是“从未获得奖励梯度”；该窗口未突破，不否定所有RL或新任务内适应 |
| 状态回报与完整G信用分析 | 完整G的State/Context二阶量比.882919，视频组.947462；9/24与11/24条件反而增大；全局clip会改变幅度解释 | 有有限信用变化，但0策略更新、0新闭环，不能登记为actor-critic训练失败或控制改善 |
| 旧LoRA重读self-read | 两次共享native/Writer与完整FM，fresh450为116/400，同龄Context126、T122 | 重读本身不保证进步；旧中间LoRA没有进入真实环境，不能说已经否定“实践后再理解” |
| 强MT上的在线读视频 | 同MT继续训练的V88/M93（/144）；特权动作1NN接手也有救11、丢29 | 尚未获得可无损蒸馏的强视频控制教师；不能先假定只剩编译瓶颈 |

原件入口：[T/C/RL/在线控制完整证据](../20261007_research_reassessment/EVIDENCE.md)、
[跨情境配对](../../analyses/cross_context_pairing_20261008.json)、
[独立教师](../../analyses/aligned_teacher_recovery_20261008.json)、
[完整G信用](../../analyses/state_baseline_full_writer_credit_20261008.json)、
[self-read](../../analyses/operator_self_read_evidence_20261001.json)。
历史完整路线见[HISTORY_MAP](../20261007_research_reassessment/HISTORY_MAP.md)与[research_history](../../research_history.md)。

更早的自由query、独立clone、完整A/B、特权纠正、局部动作/效果监督、关系与阶段模型各有正反证据。
它们共同要求区分**能表示、局部学到、共享获取、跨初态调用、完整任务成功**，但没有证明所有方法不可能。
原生多horizon、LoRA梯度非零、局部教师较好或attention有差异，都不能单独解释完整收益。

## 5. 七轮专家意见及其实际实施边界

1. 第一轮重建历史，指出已有条件控制未转成新任务上可靠的自身调用；提出更自由参数输出与自身状态监督的候选。
2. 第二轮强调自由A/B与更近标签不足以解释共享教学迁移，转向跨情境的功能学习关系。
3. 第三轮收窄为同T的Within/Product比较。该比较已执行且未总体修复；新encoder＋自由A/B＋MT起点的完整组合没有实施。
   随后的独立教师与状态信用分析是主讨论自行选择，不能合并成专家全部方案失败。
4. 第四轮在当时无奖励试做边界下提出效果元编译器，以共享更新器依据教学、当前参数与自身后果生成修订。
   它未实施；一轮试做、固定MT、仅前向更新是当时方案选择，不能按最新许可恢复成硬约束或自动选回该候选。
5. 第五轮改为教学引导行为发现，再直接训练最终LoRA；Owner当时基本接受，要求完整架构。
6. 第六轮具体化操作段、状态校准、双Q、候选编辑、完整FM／局部energy score与裸actor。主讨论当时认可，仍指出监督与实际控制的缺口。
7. 第七轮改为带不确定性的物理轨迹读出、五步动力学预测、真实实践与经验认证，局部新噪声前缀回归和完整成功FM直接优化LoRA。
   其完整结构与成本更具体，但仍属于轨迹数据准备后微调。Owner最终否决这一整类形式，主讨论撤回实施建议。

[第六轮原文](EXPERT_RESPONSE_ROUND6.md)、[第七轮完整原文](EXPERT_RESPONSE_ROUND7.md)按历史归档；
[第六轮审阅](ARCHITECTURE_REVIEW.md)的认可与待补问题也只指当时时点，不能成为新专家必须继承的模块清单。
第五至第七轮没有新的共享训练、任务内适应或正式评估结果；其预算、首批矩阵和停止条款都不是现行实验合同。

另有综述启发见[findings](../../../findings.md)§402，包含HOST/XSkill、Doc-to-LoRA、WAM-TTT、ICRT等的实际适用条件。
它们只是思想来源，不能把在线教学、可训练基座或额外训练数据的收益直接搬进固定LoRA合同；依赖外部主张时须核对一手论文。

## 6. 本次委托与后续接续

Owner最新请求见[第7节](OWNER_FOLLOWUP_REQUESTS.md)，完整问题由[唯一当前提示词](EXPERT_PROMPT_FRESH.md)提出。
专家应独立选择一套完整主设计，具体定义实际特征、算子、各层LoRA形成与自身hidden调用、共享训练信用、
实践反馈的编译作用及能力保持；区分可表示、可学到和可迁移，不以一般链式法则或模块名称代替机制。
用一个现有任务贯穿数据与参数流，给出训练／编译伪代码，并与最近似历史方案的正反证据比较。

推进从小而完整的学习与闭环循环开始，按真实依赖给出产物、判据、成本、扩大及停止条件；不将局部模块通过拼成方法资格。
区分共享训练成本、每教学condition编译／实践成本与最终执行成本。正式paired400对应400个task–video条件，
不能按8套LoRA估算全部交互适应，也不能以小面板重复初态冒充正式资格；不要求首批立即承担全部矩阵。
相邻能力、任务覆盖与公平视频证据按阶段建立，shuffled／reversed只在选点后用于时序特异性，不影响训练或选点。

主讨论负责独立科学分析、设计取舍与接续；实验session负责实现、测试、运行、资源与Git交付。
首批合同仍待专家回复后审阅与登记，不由提示词自动启动；本次只整理推送材料，由Owner自行联系新专家。
