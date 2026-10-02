# EMBER task plan

## 科学目标与当前批次

从冻结π0.5 source出发，利用exact language及action-hidden教学视频生成初次即有效的一套完整task LoRA，
由严格配对闭环证明绝对能力、相邻保持和有益教学增量。稳定边界见
[Owner要求](docs/current_owner_requirements.md)、[AGENTS](AGENTS.md)与[concept](docs/concept.md)。

Owner于2026-10-02在交接与现状讨论后恢复自主推进，不对分析设置时间限制；此前上午暂停已解除。
唯一active设计为[原生prefix变化Value](docs/designs/native_prefix_change_value_design.md)，科学登记完成，授权唯一实验session按正式派发闭环实施。
保留T的实际读写坐标及原动态Value，用同一原生动作query读取相邻真实prefix的响应差补充Value内容；
经新增E及原O/递推直接形成单套LoRA，fresh/full-only共同学习。数学、信用、竞争解释与历史边界见设计及findings§275。
执行唯一fresh450：原36task/50视频池及50,400query；270/450各correct400，450另seen144，共944行/52 full。
硬限14GPUh/48GiB，预计9–11GPUh、含工程3–5小时；到齐停止、不自动900/controls/Test/RL或小扫。

最近完成条件A冻结重表达：Original13/32→Reexpressed14/32、R13/G1/L0，0.281819GPUh，原件齐备。
主讨论机制§97/findings§274保留其有限表达证据和主要未修复失败，不部署解析C、不自动去S训练或追加投影探针。
Owner已固定现有数据规模，禁止扩数据；后继方法与学习机制须在这一约束内研究，详见Owner要求§4。
主目标是在相对稳定的情况下大幅提高绝对能力、超过强MT；T的有效证据是参照，不是不可改变的架构或保持率门槛。
上一批已检验教学输入上的条件读取重表达能否到达自身执行消费者：900父点、原四train任务各两teacher，
完整38处原LoRA对解析重表达，固定A28功能读取及64条有限配对闭环，硬限3GPUh/12GiB、预计含工程1.5–2.5小时。
没有训练、扩数据或held/controls读取；全部结果已到齐并停止，不能把局部保留自动转换成新架构资格。
原通用图文候选未实施；新的具体原生读取图仍不自动改变调用核，也不据新增输入宣称已修复迁移，完整判断见新设计。
此前完整训练合同为[条件读写§15](docs/designs/conditional_read_write_architecture.md#15-固定目标曝光的辅助任务分布短窗检验2026-10-02)：

1. 已完成§13 fresh450及§14固定图900。validation137→140/400、seen91→115/144；
   训练侧确有对象选择和多阶段获取，held主要损失仍在3/31，完整优势及相邻资格未形成。原同池续训停止。
   主讨论完整分析见机制§95/findings§270及900证据JSON；两批成本31.737894GPUh，旧原件保持。
2. 已完成唯一D71短窗：从450恢复共同学习状态，到630新增180更新，仅将支持分布12→审计71；
   target24原事件、13,440query及2/3名义权重保持，6,720个支持query按71任务等权校正，不改模型/监督/绝对LR。
   原已存C12_630为同父同龄对照，不重复其训练，不从历史点选峰值。
3. 两个固定630的400＋144、24份固定A28及16条被动native均完成，共1,088环境原行。
   Held为D71 156/C12 154，R129/G27/L25、净+2、churn52，八task簇区间[-9,15]；breadth均6/8。
   Seen为98/99，target24为53/55、原support12为45/44；完整来源、goal/continuous及不利原行见readback。
   两个630仍为有界诊断，未selected；新540仅供恢复，没有新增controls/Test/RL或自动后续训练。
4. 预算12GPUh/64GiB内完成：实际8.477763GPUh，观察53.338696GiB、保守60GiB；全部进程exit0，新增计算已停止。
   唯一root`/data1/user/ymdai/ember_runs/conditional_support_diversity_pilot_20261002`的completion/readback验收complete。
   训练/读取796d7a9e、C12旧训练85919994及450父身份分别保留；首GPU启动至整批退出约1小时33分，不含启动前工程。
   主讨论完整分析已收束为机制§96/findings§271及支持分布证据JSON：广泛迁移预测没有通过，未定位唯一神经主因。
   该批已完成汇报；不由总分或局部拟合自动开fresh，扩数据已被Owner禁止。

工程/运行/Git由实验session闭环；主讨论负责科学判断。整批科学结果或确需裁决的实质边界按Owner要求§6回报；
其它工程阶段事实保存在已有记录。实际代码、冻结、运行与窗口状态只看[progress](progress.md)。

## 仓库维护

本轮独立整理的实际改动、保留依赖、536项CPU检查、合并后38项消费者检查、已安装skill维护与回滚位置见
[研究历史](docs/research_history.md#2026-10-02整仓整理与历史状态入口)。整理承接最新实验修复，交付身份见progress与Git。
没有改动当前实验模型、数据/标签、FM、信息墙、选择合同或预算；科学接续按上方Owner最新授权与固定数据约束处理。

## 历史

多轮旧执行计划的完整快照为`git show 797ae01f:task_plan.md`；设计、实际结果、负证据和失败原因沿
[research_history](docs/research_history.md)追溯。历史未完成清单不是现行待办；不会靠整理覆盖其当时事实。
