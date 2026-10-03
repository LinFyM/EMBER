# EMBER task plan

2026-10-03 实验session01a0fabb-f7a0-7100-93d8-6a0f66055553已实际接手 `self_image_attention_transfer_20261003` 的canonical tracked/Git独占窗口；main只读科学分析。
完整合同、机制§108–109/findings§287–288及Owner资源/信息墙要求已读；独占工程树为`/data1/user/ymdai/projects/EMBER-self-image-attention-transfer`，基线d15cba34。
当前在实现同输入、同prefix的18层/50slot/10flow图像内部条件分布传递；原四臂48行/12 full/36 compact及teacher/scene/RNG固定。尚未启动模型或环境。
strg01 data1个人用量1167868424KiB、quota2147483648KiB、共享余89.023TB；本批预计新增峰6.5GiB<8GiB，硬限3完整GPUh。正式消费者将由clean pushed detached来源运行；完整交付后退役本次入口/hooks并交回窗口。

## 科学目标与当前批次

机制§109/findings§288登记唯一[自身图像读取分布交叉诊断](docs/designs/self_image_attention_transfer_diagnostic.md)。
同一自身输入下只交换C450/N450的图像内部attention分布，保留recipient的图像质量及其它计算权重；用真实闭环区分读取与后续控制。
固定task4/13/56×init32–35×四臂48行，12 full；不训练、不扩数据、无官方held/Test，不把混合臂当部署候选。
预计含工程3–5小时、3完整GPUh/8GiB；当前已登记、尚未接手/运行，main将交给唯一实验session，接手以progress为准。
下方无active段落保留先前裁决时点，不覆盖本登记；没有自动grounding loss、蒸馏或formal fresh。

main已补充机制§108/findings§287：复用旧对应和已有P/F预测，17训练task出现获取—迁移反转。
同一选帧权重下P较F改善teacher风险，却在全部17task提高跨episode风险；特权状态匹配自身仍有正收益。
该CPU读回0GPU、不扩数据，原件已保存；继续推导可重用任务关系怎样形成自身控制，不以更准局部指令或查表替代。
当前无active实验，main独占记录/Git；下方§107状态保留其完整科学裁决，不恢复已关闭配方。

main科学消费已完成（机制§107/findings§286）：控制内容调制读写在固定450的correct400=122、seen144=77，未支持完整方法预测，本配方关闭。
下一判断回到教学状态/作用与自身调用之间的可学习联系；不以再提高q回归、调制小扫或延长该450替代它，不扩数据。
当前没有active实验；main独占记录/Git，继续自主推导，保留本批原件及完整450状态等待后继明确接口与生命周期。

### 已完成批次的登记与执行事实

唯一fresh450与全部读回已完成：correct400=122/400，Seen144=77/144（train24=36/96、support12=41/48）；严格544行/44 full/500 compact、原A28的12 FM/8被动条件、六完整ECP与50400主query/55858aux区间均已实际读回。实际train/consumer2c630fb3、11.471727418GPUh、新增实占51.444GiB/保守峰72.06GiB，均在32/80硬限内，GPU与16个consumer PID全部释放。未胜强MT153或成熟T161；全参照与正反原件保留，无自动selected/续900/扫描/controls/Test/RL。唯一new canonical实现及450完整恢复保留待main裁决；整批Git/push及一次回报后交回canonical窗口，实验session停止本项。当前没有active计算，下方运行段落保留其历史时点。


当时登记的合同为[控制内容调制读写](docs/designs/control_calibrated_read_write_design.md)，机制§106/findings§285说明事前取舍。
fresh Gamma的完整控制估计直接调制同一LoRA A/B Value，真实跨episode FM学习自身调用；不以获取通过代替完整能力。
单条fresh450、原36task/50teacher，固定correct400/seen144/原A28；预计6–10小时、32完整GPUh/80GiB，实际接手/运行看progress。
无自动900、扫描、Reader/蒸馏、Test/RL/controls；旧§105未登记后继状态由本段更新。

2026-10-03main完成原生转移校准科学消费（机制§105/findings§284），直接读实际源码及176份预测。
P在四内部task相对F均改善，但task32全四视频仍差于裸mu；主要取得五步区间平均控制，非精细时序或完整反馈规律。
继续推导状态、实际变化与控制内容如何经同一LoRA和跨episode真实FM形成自身作用；旧辅助头、LocalField及Pullback负证据保持。
当前main持canonical记录/Git窗口，无active计算或已登记后继；不能由获取通过自动启动fresh/蒸馏，也不能以尚未选定消费者结束自主推进。

从冻结π0.5 source出发，利用exact language及action-hidden教学视频生成初次即有效的一套完整task LoRA，
由严格配对闭环证明绝对能力、相邻保持和有益教学增量。稳定边界见
[Owner要求](docs/current_owner_requirements.md)、[AGENTS](AGENTS.md)与[concept](docs/concept.md)。

Owner于2026-10-02在交接与现状讨论后恢复自主推进，不对分析设置时间限制；此前上午暂停已解除。
2026-10-03继续推导已形成机制§104/findings§283，登记唯一新
[原生可见转移动作校准辨识](docs/designs/native_transition_action_calibration_diagnostic.md)。
检验实际到达画面能否在固定原生动作估计之上提供跨任务可学的已发生控制信息：相同P/F读出仅替换第二memory。
固定现有train24，fit20×demo16–19、内部留task[0,12,20,32]无读出梯度，统一demo42–45读回；176条已有视频。
两臂fresh500、相同80,000区间曝光；无LoRA/Writer训练、官方Val/Test、环境、扩数据或自动后继。
预计含工程70–140分钟、3完整GPUh/8GiB；主讨论登记完成，实际接手/launch状态以progress为准。
这项只检验获取假设；完整编译还须保留状态—变化—控制的联合内容，不能把预测残差或局部动作拟合当作任务知识。
固定500及176视频/5,882合法五步区间的完整读回现已完成，P/F各80,000区间曝光，完整0/250/500恢复点和全部raw保存。
内部留task等权mu/F/P all7=.132368525/.142491026/.114196534、4/4相对F改善，获取判据通过；task32四视频仍劣于mu、rotation等不利项保留。
本批费用.134531062/3GPUh、阶段实占2.051GiB/保守峰2.3/8GiB；实际消费者a4da650a，GPU与CPUexit0且GPU已释放，三个专用源码已退役。
整批Git交付push与一次来源明确的回报后实验session停止并交回canonical窗口；主讨论消费原件并接续理论判断，无任何自动后继计算。
以下此前“无active计算/下一判断”属于§103收束时点，不覆盖新合同。
最近自身功能信用两臂已完成科学消费，机制§103/findings§282：live/stop学生20/24，Reader19/23；
live对stop R18/G2/L6，没有新增query信用的净控制收益，辅助读出亦未建立强教师。
保留stop24正例及历史比较的native重构边界；关闭该构造续训、扩头/换层/调权、蒸馏与正式fresh的当前理由。
本项结束、无active计算，main持canonical记录/Git窗口。下一方法判断须解释教学可观察内容与可迁移动作修正的联系，
不以新增梯度边、局部高点或关闭分支替代完整目标；没有自动后继实验。
以下保留最近登记及执行事实，以本段和progress顶部的最新消费状态为准。
主讨论已完成最近视频来源批的原件消费与机制§101/findings§280：不支持把早/后更新直接划为开炉/搬壶技能。
早段增量的后段地址与完整父LoRA依赖已明确；关闭由这一分段推出门控或继续局部扫描的路线。
主讨论随后完成机制§102/findings§281，登记唯一
[自身执行特征功能信用辨识](docs/designs/state_coupled_functional_credit_diagnostic.md)。
旧T2340四train任务/八teacher/原A28，两臂64共享更新；辅助头读同Z与当前执行hidden，只比较其query梯度live/stop。
固定终点各student/Reader读原32行，总128新增/24 full；主要判断合法单LoRA学生的实际控制，Reader单独收益不能替代。
预计75–135分钟、3完整GPUh/8GiB，无新数据、held/Test、蒸馏、正式fresh、自动续训或超参数扫描；状态见progress。
这是一项学习联系的辨识，未确认新架构或完整修复；历史Reader负例、原S/P/D全部正反与正常数值边界保持。
两臂完整64更新/恢复点及功能预测、128/128环境行与24真实full已完成，实际训练795e86a0、读取1449c908。
live/stop学生20/24、Reader19/23；live学生对stop R18/G2/L6/churn8，没有本有限面板总成功改善，完整正反原件待主讨论裁决。
全部费用1.377360/3GPUh，阶段观察4.451GiB、保守峰6/8GiB；进程/GPU已释放，入口/hooks已退役。
Git交付与一次整批回报后canonical窗口交回main，实验session停止本项；不自动选部署Reader或续训/fresh/追加扫描。
以下保留此前完成的冻结分析及接续依据，当前唯一合同与状态见上方及progress：
[task32已学修正的执行投影分组](docs/designs/task32_learned_operator_groups_diagnostic.md)。
只比较旧D17/S43各自的Q/非Q学习增量在原四初态的作用；16新增行、4 full/12 compact与完整行为读回已交付，0.078771GPUh。
17 Q/R成功2/4、2/4但集合不同，43 Q/R成功3/4、2/4；本有限面板的父成功行全部保留、完整学习的部分新增收益未保持。
专用入口/hooks已退役、原件与clean pushed detached22faa196保留；main消费完整结果后解释，实验session停止，不自动追加训练或探针。
main现已完成机制§100/findings§279：全部16份continuous与四份RGB直接读回，正反交互均保留。
同一学习更新的足够作用随初态改变，Q/R不能充当两个技能的天然分区；关闭按该分组选择全局修复或继续细分投影。
后继关注视频特征对应的有用/有害学习作用如何被自身状态调用，尚未选择新架构或正式训练。
当时登记的唯一后继为[已学修正的视频来源诊断](docs/designs/task32_learned_video_segment_diagnostic.md)：
同S64的两teacher、实际炉面变红边界80/95、全部38处共同分解E/L学习增量，共16新行/4 full、复用16参照。
若后段来源保留放置同时损害开炉，支持该有限调用解释；若早段/联合或分布混合，按合同修订并关闭相应阶段局部化理由。
无新学习/标签/held，16/16行与完整读回已交付：17 E/L成功3/4、2/4，43为3/4、1/4；实际0.082022GPUh，4 full/12 compact及H/全38 K保留。
专用入口/hook退役，本批完成回报后实验session停止并释放窗口；主讨论消费正反与数值差后裁决，不自动扫切点、重训或选择部署混合bank。
最近[task32状态/冻结LoRA交叉续行诊断](docs/designs/task32_state_policy_crossover_diagnostic.md)已完成，
主讨论直接消费原件并完成机制§98/findings§277，把具体行为、实际native/S/M读写、自身调用和真实FM信用联系起来。
同query两teacher风险分解表明离线共同误差占主导，四格则显示局部后段控制可用与到达/及时纠正不足可以同时存在。
不从其中一项直接选择新Value、去S或一致性训练；数据规模保持固定，没有新梯度、Val/Test/controls或选点。
既存强MT/旧T的同输入CPU审计已完成，0GPUh；当前前5FM不劣于MT、十步motion6更低而gripper更高，
并非统一动作欠拟合，也没有证明该通道误差导致当前自身轨迹失败；具体正反证据见机制§98.6。
Owner随后纠正主讨论用MT成败筛选EMBER失败案例的错误：MT只是参照，目标是EMBER大幅超过MT。
为这一错误筛选标准提出的两条MT补测尚未派发即撤销，0新增GPU/环境；已有参照与理论分析保留。
下一判断回到EMBER自身缺口与已知可学改进，不要求MT先成功。16行批次已完成并交回canonical写入/Git窗口，实际状态见progress。
旧T父/S/P/D的task32完整32条CPU读回已完成，0GPUh；main直接消费原件并完成机制§99/findings§278。
得到的是实际控制获取与交换：D17补齐搬壶，D43未同样学成，S/P有其独立收益及丢失。
同task两D使用相同功能目标，差别来自条件初始函数及其学习；私有参数修正不是唯一视频语义标签。
结合已完成PZ及跨条件外推，继续区分可表达的控制、共享获取与从视频推断新条件所需作用；
不重复旧投影/参数空间扩大，也不由旧T局部成功直接选新架构或正式训练。
同B20的学习修正分解进一步显示task32共有修正占主导；已完成的Q/R干预揭示状态相关的互补与损害，
不把组名当感知/运动语义，不据任何单一分支自动重训或追加层位扫描。实际派发/运行看progress。
Owner要求的决策错误已在394b5f79写入稳定要求及历史。

四行与完整读回已完成：success矩阵按前缀17/43、LoRA17/43为[[真,真],[假,假]]，两条对角重现且anchor配对误差0。
抬壶与放置分开：s43/W17在509步才抬高3cm，520仍未放置；不能把该失败概括为完全没有抬壶。
全部4 full、连续/goal/动作和首次F2×2有限差齐备；累计0.029409131GPUh含首加载失败，阶段观察1.489/4GiB。
GPU已释放；有效冻结读取e84712d8，任务专用active入口/hooks已退役，原件和失败保留。
执行者整批交付后停止本项并释放canonical/Git窗口；主讨论已核对原件，后继为上述完整机制分析。

最近[原生prefix变化Value](docs/designs/native_prefix_change_value_design.md)在派发后由主讨论撤回：
Owner指出“具体task失败→有证据的方法缺陷→干预改变失败预测”链条不足、block17仅启发；
主讨论承担方法选择错误。允许有界检验不能跳过机制辨识，也不能让正式训练替方法选择寻找理由。
本批因方法选择依据不足撤回，非工程失败或性能阴性；设计与findings§275只保留当时判断，不恢复执行授权。
原fresh450＋944行/52 full合同未完成：实际完成两步profile及已登记71次正式更新、284条件/7,952query，
SIGTERM收束，未到90，完整ECP为0，已登记1–71更新未保存；全部后继训练/bank/评测已取消。
累计1.452021841GPUh、root观察约0.69GiB，本批GPU/进程已退出；source/spec/冻结代码、日志和退出费用保留。
收束原件在`/data1/user/ymdai/ember_runs/native_prefix_change_value_20261002/`，实际身份及读取边界见progress。
实验session已完成记录/Git交付并停止该批；后继四行冻结分析也已完成，不恢复被撤回的训练。

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
