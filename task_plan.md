# EMBER task plan

## 科学目标与当前批次

从冻结π0.5 source出发，利用exact language及action-hidden教学视频生成初次即有效的一套完整task LoRA，
由严格配对闭环证明绝对能力、相邻保持和有益教学增量。稳定边界见
[Owner要求](docs/current_owner_requirements.md)、[AGENTS](AGENTS.md)与[concept](docs/concept.md)。

Owner于2026-10-02在交接与现状讨论后恢复自主推进，不对分析设置时间限制；此前上午暂停已解除。
当前没有active计算设计。[原生prefix变化Value](docs/designs/native_prefix_change_value_design.md)在派发后由主讨论撤回：
Owner指出“具体task失败→有证据的方法缺陷→干预改变失败预测”链条不足、block17仅启发；
主讨论承担方法选择错误。允许有界检验不能跳过机制辨识，也不能让正式训练替方法选择寻找理由。
本批因方法选择依据不足撤回，非工程失败或性能阴性；设计与findings§275只保留当时判断，不恢复执行授权。
原fresh450＋944行/52 full合同未完成：实际完成两步profile及已登记71次正式更新、284条件/7,952query，
SIGTERM收束，未到90，完整ECP为0，已登记1–71更新未保存；全部后继训练/bank/评测已取消。
累计1.452021841GPUh、root观察约0.69GiB，本批GPU/进程已退出；source/spec/冻结代码、日志和退出费用保留。
收束原件在`/data1/user/ymdai/ember_runs/native_prefix_change_value_20261002/`，实际身份及读取边界见progress。
实验session完成记录/Git交付后释放写入窗口并停止本批；主讨论负责科学纠正及后继判断，无新计算待办。

Owner最新GPU要求已加强为AGENTS§9长期规则：显存有明显余量必须主动验证更大物理批量/帧分块或合适并行的
真实吞吐，不沿用保守默认值后只报低显存；配置及未放大依据须有记录，逻辑batch/任务权重/更新与科学合同保持。
本批沿用frame8却没有据22.805GiB峰值验证更大分块，作为未落实要求的执行事实保留，不追加已撤回计算。

最近完成条件A冻结重表达：Original13/32→Reexpressed14/32、R13/G1/L0，0.281819GPUh，原件齐备。
主讨论机制§97/findings§274保留其有限表达证据和主要未修复失败，不部署解析C、不自动去S训练或追加投影探针。
Owner已固定现有数据规模，禁止扩数据；后继方法与学习机制须在这一约束内研究，详见Owner要求§4。
主目标是在相对稳定的情况下大幅提高绝对能力、超过强MT；T的有效证据是参照，不是不可改变的架构或保持率门槛。
上一批已检验教学输入上的条件读取重表达能否到达自身执行消费者：900父点、原四train任务各两teacher，
完整38处原LoRA对解析重表达，固定A28功能读取及64条有限配对闭环，硬限3GPUh/12GiB、预计含工程1.5–2.5小时。
没有训练、扩数据或held/controls读取；全部结果已到齐并停止，不能把局部保留自动转换成新架构资格。
原通用图文候选未实施，原生读取图批次已撤回；不据新增输入或工程接通宣称已修复迁移。
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
