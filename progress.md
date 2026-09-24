# EMBER progress

## 当前状态：保留六臂训练，先执行1500条核心诊断再裁决补测（2026-09-25）

Owner最新指令是由本任务接管主讨论与科学决策，以历史证据、竞争机制和可反驳干预推进；具体实验交由现有Sol执行，
每批完成主动回报，待主讨论分析后再派发下一步。这取代旧全局暂停；不恢复旧deadline或GPU特例。
Owner随后明确给予充足时间和持续优化的广泛分析/实验授权：主讨论按证据自主推进后继机制验证和改进，
不能停留在现象、随意试改法，必须逐轮保存假设、预测、原始证据与经验。授权覆盖四臂之后的研究，
当前已启动批次仍按冻结合同执行；科学信息墙、资源限制和结果可追溯要求继续适用。
Owner又明确本次授权**不向Sol转发**，让其专注具体实验；主讨论自行维护长期判断和记录，收到本批结果后再给出具体下一步。
主讨论：`01a0cd94-65da-7b22-8ca9-7ba35f454632`（接管 EMBER 科学决策与实验）。
实际执行者：`01a0cd90-ebb7-77a1-a20b-a858825d2f66`（当前标题“接管 EMBER 实验”，Owner指定Sol）；已核对同仓库/主机，五批已完成。
双方保持现有模型配置；旧Luna和旧主讨论仅作历史provenance，不再作为收件人。
**当前active design为[任务关系支持的学习干预](docs/designs/relational_support_causality_design.md)**，
机器合同`configs/relational_support_causality_v1/experiment_spec.json`。主讨论已完成metadata/完整BDDL/可辨识关系审计，
冻结四个fit28池、四C与两B、共同1260节点的因果对比及资源/停止合同；原六节点及9932方案已被下述1500条阶段合同替代。
本批C_S01 correct100已完成；其余面板待E3被动区域修正后继续，尚未进行完整矩阵的科学裁决。
设计已在`6972486e`集成推送。2026-09-24 09:19 UTC以Queue派发，message
`01a0d2b6-24e5-7ae0-a1a9-30077bd8cc4e`；09:19:59 UTC从现有app-server核对**完整正文**进入Sol的
inProgress turn `01a0d2b6-24ee-7401-bdb0-7ea22629e35b`，当前cwd及实际标题亦已核对。
正文/入队回执/逐字投递核验保存在前批readout study的
`coordination/relational_support_dispatch_20260924.{txt,json}`与`relational_support_delivery_20260924.json`。
上述Queue记录证明当时已接收任务；随后实现、正式启动及工程补验记录见下，尚无本批科学结果。
Owner明确纠正“下一合同尚未定稿便结束主讨论推进”的错误。执行者完成一批等待派发不约束主讨论继续分析，
主讨论已把下一项推进到可执行合同；已有授权内不重复索要许可，也不为让实验不停而仓促试改法。

新study根`/data0/user/ymdai/ember_runs/relational_support_causality_20260924`，正式树
`/data1/user/ymdai/projects/EMBER-relational-support-formal`。六臂各1260更新/141120query保持，当前阶段仅1500闭环、
66固定full cases，不启动3300函数预测；data0峰值≤128GiB/data1新增代码≤1.25GiB（保留E2并另冻E3），同时最多6物理GPU。
实现已集成推送`7dc95edbba00cf61439700d77fb321eb8df95c07`，正式树为该clean detached commit。
Sol完成CPU 91+9项、六臂4更新/2→4恢复、最长full-H50及新support物化→评测smoke，查询strg01独立quota与共享容量；
预计data0峰值新增120GiB低于128GiB，data1开发/正式树合计约503MiB。精确原合同为`launch/formal_launch_contract.json`。
主讨论已调用canonical task authority验证新manifest58任务/Train42/Val8/Test8；每个optimizer的白名单仍只有登记fit28。
Source71及官方24/8/8未改；无新增held expert或官方Val/Test读取。完整机制与竞争解释见task_plan第七阶段和findings§140。

### 当前第一阶段合同（2026-09-24 16:39 UTC／北京时间9月25日00:39）

Owner先要求只回答剩余时间和价值，随后明确要求按“核心诊断先做、分析后再补”指导Sol。
主讨论核对本批evaluation的run_contract/results/launcher_completion仍均为0，先登记18面板1500行，不读取成绩后筛面板。
全部模型仍固定1260：六臂14/21各50state为600，四C的Goal21 other各50为200，六臂新增四support各20为480，
Source两目标100加六support各20为120。50-state主问题与原teacher映射完整保留，support只截原映射state0..19。
66个full cases是原病例在当前面板中的交集；全行被动采集合同继续。1050、其它held6、seen64、Object other、wrong和3300预测暂缓。

六臂训练和12个完整checkpoint保留；00:26按Owner状态请求核对前三C已1260/exit0，C11和两B已按原lane启动。
第一阶段允许已完成臂/Source在剩余训练进行时先评，按对应臂checkpoint/exit与E/采集验证准入，仍合计≤6物理GPU。
执行者输出`analysis/stage1_completion.json`后主动Queue主讨论并停止新增评测；主讨论核对原件后自主裁决必要补测。
当前只有局部机制判断，不能宣称held400性能、完整保持、相邻稳定性或有益视频因果修复。新合同详见design§5/spec stage1。
合同在`057c57e2`集成推送。16:49 UTC向当时idle的Sol以Queue派发，回执
`01a0d452-8a88-7702-b8e4-de5c4ec1518b`，新turn为`01a0d452-8a9e-7a90-ba43-46992e91beb3`。
已核对该turn为inProgress；当次app-server full items仍为空，当时未逐字核对正文或读到Sol回应，未重发。
随后主讨论收到Sol明确接续回报：仅执行18面板/1500行/66 full cases，原7dc训练继续，E完成后按阶段白名单启动；
1500行完成即主动Queue原件与缺项并停止新增评测，正常运行不例行轮询。由此确认执行者已接收并理解阶段范围。
完整派发正文、接受回执和当时查询证据在`coordination/stage1_core_{dispatch.txt,dispatch_receipt.json,delivery.json}`，
本次回报见`coordination/stage1_core_acknowledgement.json`。该回报不代表E已冻结、执行器修订已验收或正式评测已经完成。

### 第一阶段物化接口冲突与裁决（2026-09-25）

Sol回报7dc完整held/support选择限制与1500阶段不相容；主讨论从冻结树CPU复现held两任务及support20拒绝，
并确认Goal21-only other还受旧完整held400诊断声明/配对参照限制。原50-video排列表取state子集的映射本身不变。
按设计§6/spec的新subset物化修订，允许将选择/校验及其prepare/resume适配纳入尚未冻结的唯一E，
本阶段`materialization_commit = evaluation_commit = E`，训练及单condition生成语义仍固定7dc；不预物化暂缓面板。
Goal21 other绑定同1260 correct100的Goal21投影，按原offset17复用已生成的50个teacher条件，不增加Writer forward。
目标、采样、科学对比、18面板1500行66cases及资源预算未改；实现和实际GPU接口仍由Sol完成验收，当前不称修复已完成。
裁决在`32d5a7b1`集成推送；17:04 UTC以Steer送入Sol活跃turn
`01a0d452-8a9e-7a90-ba43-46992e91beb3`，接口返回同一turnId。当次full查询未显示完整新正文或后续回复，
因此只记Steer已接受，不称执行者已处理；未重发。派发、回执、查询与主讨论CPU复现证据位于
`coordination/stage1_subset_materialization_{dispatch.txt,dispatch_receipt.json,delivery.json,coordinator_evidence.json}`。

### E1评测准备失败，执行者继续窄修正（2026-09-25）

Sol已实现并推送E1=`bda68cd86ca86d17802cda8b5a6fdbe0b9ba6b86`，登记6条工程smoke通过；
主讨论核对`C_S01_1260_core_correct`已封存100个condition，bank记录训练7dc、物化E1与原50-video映射。
该面板在prepare退出1，尚未进入start：失败日志指向`registered_passive_capture.prepare_selection`的旧
`launch/selectors`约束，而真实参数为`launch/stage1/selectors`；对应evaluation目录尚未生成，无闭环行。
原件见`launch/eval_C_S01_1260_core_correct{.log,_command.json,_exit.json}`及该E1 bank manifest。
主讨论检查隔离开发diff，仅修正selector目录约束/路径解析及相应测试；Sol回报相关31项通过，正完成剩余回归与E2集成。
按既有授权继续：保留E1失败证据，以新clean pushed detached E2统一有效bank与评测；具体E2及真实prepare成功尚待验收。
训练、数据、18面板1500行66cases和停止合同未改，无新的科学结果；不新增指令或打断执行者当前修复。

### E2完成100行后遇arena区域采集故障，保留原行并准许E3窄修（2026-09-25）

E2=`e9e518cd623b58ad6a23ebec376383c565da7dda`的C_S01 correct100已完整；主讨论独立验收唯一task/state集合、
4full、100条连续trace及4个worker exit0，所有goal region均走不受修正影响的对象/fixture分支。
support80在global77 t0遇arena workspace不在普通fixture字典的采集错误；真实site/parent及八任务BDDL审计见
`smoke/stage1_region_registry_probe.json`、`launch/stage1_region_bddl_audit.json`。主讨论审阅窄修diff确认只扩展该被动登记分支。
科学裁决保留全部E2 correct100及两个合法E2 bank，剩余17面板1400行使用E3；C_S01 other继续复用E2 correct条件。
不按成绩筛行、不为统一commit重评；按设计§6/spec `passive_arena_region_exception`逐panel保存真实版本和生成来源。
E2原runtime/selector/原件保持，E3另冻新树；新增代码上限仅为此从1GiB调到1.25GiB，复制前仍核验quota，GPU/data0不变。
主讨论核验launcher摘要时见到总成功计数，尚未分析逐task效果；本裁决依据修正范围与原件完整性，不宣称全程盲态。

失败原件`evaluation/C_S01_1260_support_core/failures/launcher_1790274533615647026.json`显示0完整行，
两个global77 job报错、两个global58 job被终止；只读queue核对四job合计20个登记state上界。其它worker是否已执行控制步
不能从零完整行推断；中止尝试据实单列。本次明确授权修复后重启该未完成support80一次，不增加100条重评或自动无限重试。
E3真实无动作接口、回归和最终集成仍待Sol完成，不称故障已修复或本批已完成；阶段科学对比和结束信号保持。

### 前一次成本修订（2026-09-24 14:38 UTC；已被第一阶段合同取代）

Owner询问本批成本与信息量是否匹配。主讨论核对发现16704/21868条用于六节点完整曲线，
而核心1260交互仍只有一个训练seed、Goal21的50配对状态；增加早期节点不会补足独立重复。
此时evaluation下run_contract/results/launcher_completion均为0，没有读取本批闭环分数后选择性删减。

保留六臂训练与全部12个checkpoint、1260主比较、支持操作获取、Source保持与视频对照；
只执行1050/1260的六臂held400+seen64（5568），固定1260报告、不再择优，controls2400、support1200、Source764，合计9932。
3300功能预测与82固定full cases保持。取消210/420/630/840评测和selected额外other，不取消任何已产生证据。
代价是没有早期学习曲线与最佳点结论；本批定位具体关系支持假说，不能以大矩阵数量保证统一根因。
新authority为spec的`evaluation.executed_updates`；旧optimization/evidence节点表只保留原冻结训练身份。
实际调度/分析须由Sol接续该修订并验收，旧自动评测脚本不可直接运行；原7dc训练不热改、不重训。
合同修订已在`a473a6b6`推送。14:43 UTC以Queue送达当时idle的Sol任务，回执
`01a0d3df-3b39-7a63-b970-7e7a5713b30e`；14:44 UTC核对完整正文进入新turn
`01a0d3df-3b3d-77d2-85a4-b0a9e933c360`，Sol明确接续新节点/固定点/机械分析，同时保持训练等待退出事件。
这是已接收并开始处理，尚不等于执行器已修订验收。正文、回执和完整正文核验见本study
`coordination/evaluation_scope_reduction_{dispatch.txt,dispatch_receipt.json,delivery.json}`。

### 正式评测前的采集缺口与裁决（2026-09-24）

Sol报告普通canonical trajectory capture不含逐控制步对象/EEF/夹爪，正式闭环尚未启动。
主讨论核对`pi05_evaluation.py`、`episode.py`、`trajectory_capture.py`及`readout_trace.py`确认：
旧连续trace只受readout intervention门控，且其roles仅支持旧两任务；普通8D replan state不能重建对象轨迹。
同一检查还确认`preparation._registered_trajectory_capture`仅为full cases启用stage predicates，
也不满足本批全部逐state保存谓词的要求。此前主讨论ready检查未覆盖这两项原件字段，不能据ready称其已验收。

主讨论裁决并修订active design§6/spec：保留全部六臂训练及其`7dc95edb`冻结树，
训练/恢复/物化/3300函数预测仍从该提交执行；所有尚未启动的新正式闭环统一固定新的评测实现提交E。
E仅修复被动采集和prepare/resume/逐行验收，须验证不改变控制、RNG、LoRA/policy或成功/时限规则。
不接受缺项，不重训，不用额外正式闭环补trace；该采集裁决当时保持21868/82case，后续独立成本修订将条数降为9932。
精确E与修订launch/provenance由Sol实现、验证、集成后登记，原训练合同不追改；该裁决登记时E尚未产生，后续状态见上方E1/E2记录。
具体验收、工程smoke和停止边界见设计§6。正在运行的训练持续等待退出事件；只暂缓受影响的正式评测。
修订已在`c3f4a768`推送；10:48 UTC以Steer送入Sol当前turn `01a0d2b6-24ee-7401-bdb0-7ea22629e35b`。
10:49 UTC通过该turn的full items核对完整正文，并读取Sol明确回应：保留7dc训练/物化，建立独立评测E并先验证采集。
这证明接续修复已被处理，尚不证明修复完成。正文/接受回执/逐字投递及回应在本study的
`coordination/passive_capture_fix_dispatch.txt`、`passive_capture_fix_dispatch_receipt.json`和`passive_capture_fix_delivery.json`。

2026-09-24 11:21 UTC，Sol的隔离实现为`0da61891`，尚未冻结为正式评测E。主讨论审阅完整source diff，
确认只涉及采集/prepare/resume/row验收六个文件；AST复核动作规划、rollout、初态和逐行验收除明确采集调用外保持。
主讨论独立运行采集/runtime两文件22项通过；另发现旧固定case测试的两个调用漏传新增`repo_root`，已复现失败。
11:20 UTC以Steer送达，11:21 UTC核对完整正文及Sol明确接续修正；不据此声称该回归已复验通过。
工程smoke预登记6条（上限12），任务均属共同26；真实环境/模型检查尚待合格GPU余量。
旧C/B工程checkpoint实际来自`6972486e`开发状态，须保留真实来源；不能冒充7dc正式bank或声称已通过正式prepare。
Sol已确认该边界，正式bank的prepare/resume仍按严格来源验收。审阅记录为
`coordination/main_passive_capture_code_recheck.json`，正文/回执/回应为`passive_capture_review_{steer.txt,receipt.json,delivery.json}`。
本次未读取正式训练进度、metrics或checkpoint；无新的科学结果。

### 本批启动核对与已落实的执行修正（2026-09-24 10:31 UTC）

主讨论核对源码和六份真实`training_events.json`：每臂5040条件、每task 180次，共同26个任务的完整事件与宏步位置相同，
同池B/C的完整事件流相同；跨episode/21+7权重、Source冻结、活动梯度、最长347原始帧→71采样帧及新support接口通过。
独立核对脚本和结果为本study的`coordination/main_ready_recheck.{py,json}`；工程smoke不代表模型效果。

启动记录显示原方案六臂在同一对卡串行、训练约25小时。一次双节点实时快照确认gpu02有充足共驻余量；
同时发现B两臂完整恢复仅用world1，正式world2此前只有一步profile。主讨论于10:19 UTC以**Steer**纠正当前执行轮，
要求保留已启动C_S00、补齐B的正式拓扑恢复并并行尚未启动的臂。正文/接受回执为
`coordination/ready_execution_steer{.txt,_receipt.json}`；Sol的处理回报为`launch/schedule_revision_message.txt`，
其Queue回执`01a0d2f6-1af0-7150-9d61-08ec0c591a18`发回本主讨论。摘要API未显示后追加Steer正文，
不据此声称逐字可见；接受回执、执行者明确回应和完成产物共同验证已处理，见`ready_execution_steer_delivery.json`。

- B_S00/B_S11在同一冻结实现、world2各完成4更新/448真实query和2→4恢复，四段exit0。
  主讨论直接核对完整checkpoint中的optimizer步数、scheduler、sampler cursor、两rank状态/RNG及后续text-meta梯度，
  Source仍冻结；补验权重不用于formal初始化。原world1证据保留，缺项已关闭。
- 三个C的正式启动合同已核对：C_S00在gpu01/1,4，C_S01在gpu02/0,1，C_S10在gpu02/2,3，共6物理卡、每臂world2。
  后续依次复用为C_S11、B_S00、B_S11；每次launch仍由Sol重新做双节点live准入。C_S00未中断。
- 修订合同`launch/formal_launch_contract_amendment.json`和`formal_schedule_handoff.json`保留原记录；
  原串行控制器只负责C_S00，后五臂无启动地交接，新调度使用唯一launch记录避免重复。训练估计约11小时，后续评测ETA未实测。

独立补验与三臂启动核对见`coordination/main_ready_amendment_recheck.{py,json}`；模型、训练数据、科学参数和冻结实现均未变。
这次只核对已完成smoke和正式启动合同，没有读取正式训练的进度、metrics或checkpoint。
随后按退出/完成或异常事件接续；等待整批原件后作科学裁决，不以工程通过宣布根因或修复。

最近完成的[动作读出与内部适配因果分解](docs/designs/readout_realization_causality_design.md)，
机器规格`configs/readout_realization_causality_v1/experiment_spec.json`，设计提交`4e1f5cfc`，正式实现`ed2df051`。
2026-09-24 07:05 UTC曾核对Queue正文进入Sol进行中turn：Queue `01a0d23b-bf4c-75d1-bdb5-5f434a1b3540`，
turn `01a0d23b-bf53-7841-8e28-6018b004591c`。正文/回执及投递核验在前批交叉视频study的
`coordination/readout_dispatch_20260924.{txt,json}`和`readout_delivery_20260924.json`；不据入队声称正式运行已启动。
完成根`/data0/user/ymdai/ember_runs/readout_realization_causality_20260924`，formal树
`/data1/user/ymdai/projects/EMBER-readout-realization-formal`保持clean detached ed2df051。
固定C420的最终action_out与其余37个LoRA四格开/关，300既有query共1200预测、两任务50初态共400配对闭环已完成。
主讨论独立复算全部预测与闭环主对比，核对配对/trace及16固定双相机case，见findings§139与本study的coordination。
只读出真实flow计算和冻结参数干预，无训练、新expert标签或新Writer运行；不把本批当部署成绩。

前批[交叉视频函数诊断](docs/designs/crossed_video_action_field_design.md)已完成关闭，正式实现`2cfd1da9`；
设计提交`d19fb26a`，2026-09-24 05:42 UTC曾核对Queue正文进入Sol的turn：
Queue `01a0d1ee-e815-7da3-b863-52e83708b5c5`，turn `01a0d1ee-e818-78a3-88ba-f091050d29db`。
正文、入队及投递核验在前批动作通道study的`coordination/crossed_video_dispatch_20260924.{txt,json}`
和`crossed_video_delivery_20260924.json`。完成根`/data0/user/ymdai/ember_runs/crossed_video_action_field_20260924`；
300queries、15900实际10-flow预测与四worker exit0，主讨论已独立核对并重算全部主分量及区间，见findings§138。
前批[动作通道干预](docs/designs/approach_channel_causality_design.md)已完成关闭；原Queue回执`01a0d183-f68c-7bc3-b566-b8343b712cf8`，
turn `01a0d183-f68f-7932-aea8-daa787fd268a`；正文与回执仍在前段study的`coordination/approach_dispatch_20260924.{txt,json}`。
前一批[冻结前段交换](docs/designs/frozen_prefix_causality_design.md)已完成并关闭；原入队回执
`01a0d0e9-1dee-79b3-abfd-2c59a20e20ab`及正文保留在四臂study的`coordination/prefix_dispatch_20260924.{txt,json}`。
原[四臂诊断](docs/designs/conditional_compilation_diagnostics_design.md)已完成：54面板13200行、四臂全部1260更新、
selected controls及480条登记动作probe齐全；Sol主动完成回报后已停止新增实验，registration为`registered_batch_complete`。
主讨论独立原始行复核和裁决见findings§135–139。前段交换研究根为`/data0/user/ymdai/ember_runs/frozen_prefix_causality_20260924`，
动作通道根为`/data0/user/ymdai/ember_runs/approach_channel_causality_20260924`，均有`analysis/completion.json`。
五个已完成study、冻结运行树和原件只读保留；协调复核记录单独放study的coordination目录。
Sol在独立worktree拥有代码、训练配置、测试和run产物；主讨论拥有科学解释及主线状态文档，集成前协调避免覆盖。
短期步骤、结果解释分支和派发要求见[task_plan](task_plan.md)。
接手后在既有授权内完成实现/核验/有界诊断；旧的组会deadline、无上限GPU及其它历史运行许可不恢复。
长任务正常期间等待完成或异常事件，不固定间隔读取训练进度、日志、checkpoint或共享缓存；
9月23日长期授权本身未转发Sol；现在只派发经过分析后形成的具体实验。

### Owner方法边界补充与并行研究

Owner明确允许参考元学习、VLA并实质调整架构/训练，同时要求保留两个月形成的自身特色。
已将该要求写入稳定要求；主讨论完成一轮[外部机制与历史证据对照](docs/analyses/external_mechanisms_and_ember_identity_20260924.md)。
该分析保留一次教学编译、原生动作知识、零交互跨初态执行和有益视频增量，开放具体实现与训练组织。
该文献分析未选择新训练或修改当时1000分支；后继冻结函数诊断依据本批原始证据单独登记。

### 动作读出干预完成与裁决

正式1200预测、400闭环、16固定full cases齐全；五个预测worker正常完成、八个evaluator worker exit0。
主讨论重新读取300份预测NPZ、400原始行及对象/接触trace，复算12个成功对比和36个函数对比的联合bootstrap。
初态/controller/噪声/条件配对通过；四个预定case条件的八份派生LoRA逐A/B检查正确；四张16-case拼图已查看。
复算Euler/分解最大余项分别6.68e−7/7.87e−7，读出关闭时实际delta为0。
原件/独立复核在本study的`coordination/main_recheck.{py,json}`、`mask_fixed_cases_recheck.json`和`fixed_cases_*`。

00/01/10/11（无适配/仅读出/仅内部37/完整C）成功数：Object0/0/41/46，Goal35/33/30/28，各50。
去掉读出使Goal净+2，描述性95%区间[−4,+8]例、6得4失；Object净−5，区间[−10,0]例、1得6失。
Goal初态10−11朝B−11的投影均值−.00248且区间为负，未兑现事前的正向功能预测。
早期xy读出关闭的RMS命令变化仅约.057–.087mm，而内部适配相对Source约4.85–14.22mm；这是作用位置，不是训练根因。
不采纳冻结action_out，不将Goal小幅回升称修复；结束局部模块开关链。下一步须区分任务关系支持与功能学习信用等来源，
完整目标仍包含正确/另一正确视频的绝对收益及能力保持，不能只针对Goal回退或制造wrong下降。

### 15900交叉视频预测完成与裁决

唯一实现`2cfd1da9`，formal树`/data1/user/ymdai/projects/EMBER-crossed-video-field-formal`。
300query×53条件的全部原预测、原100条C轨迹/bank/噪声/OSC/两因素分量及bootstrap已由主讨论独立复核。
代码与复算在本study的`coordination/completed_recheck.{py,json}`；source/readout矩阵的补充事后描述在`output_operator_geometry.json`。

Object的V/T为.113/.183/.388%，Goal为3.532/1.419/3.735%；C/B共有分量为主，限幅不是主要解释。
wrong仍改变动作，不能称视频关闭；小幅条件变化也不能直接换算为闭环影响。Source原Goal40→C25、Object0→C46，保持与获得须一起解释。
关闭本批，下一项在action_out这一个明确计算边界分解直接读出和flow反馈，并以四格配对闭环核验；不直接采纳删层或训练修改。
首次gpu02的0/1/2/3准入被拒时零预测启动，随后0/1/2完成；保留拒绝原件，无科学缺项。

### 1000动作通道干预完成与裁决

正式实现`6a9ea40c`；14面板、48 worker exit0、1000唯一分支、500同prefix双接手历史、40固定full cases完整。
主讨论从原results、动作/NPZ/contact独立复核并重算全部32个配对对比及bootstrap，与Sol机械汇总一致；
原件与复算在study的`coordination/completed_recheck.{py,json}`、`geometry_recomputed.jsonl`及四张固定case拼图。

Goal主C接手：纯C/换xy/换z/换xyz/纯B为27/40/19/40/40。xy改变实际接近并改善目标/干扰物接触与位移，
z确实抬高末端却未改善；交互区间跨0。Object为46/40/38/37/39，存在正确目标已移动后的失败，不能统一修复。
当前只确认局部动作作用，未证明Writer学习根因或视频净增量。新旧纯前段成功集合变化单列，主比较限本批内部。
据此结束命令层扫描，下一批固定相同机器人query交叉C的视频条件，追查共有映射和视频变化各自的函数作用。

### 1600前段干预完成与裁决

Sol主动回报完成并停止新增实验。唯一正式实现`1f583d9e`；64固定双相机full cases、400旧k0引用与40登记配对比较齐全。
主讨论从20份原始results独立复核1600唯一分支、原bank/teacher/RNG、同锚历史，并复算配对R/G/L及bootstrap，均与汇总一致。
见本study的`coordination/completed_recheck.{py,json}`、`approach_geometry.{py,json}`和四张固定case拼图。

Goal同一接手者在B前段后点估计均更高；Object无B前段普遍优势。第25步物体body位置相同，但C的末端较低，
水平接近在Goal较远、Object较近。这支持早期机器人接近状态的局部因果作用，未确定动作通道或Writer学习根因。
同历史correct/wrong差异未得到统一方向的证据，也不等于视频无作用。新旧纯策略续接的差异继续保留。
完整判断见findings§136；下一批用x/y与z来源2×2干预、完整B参照及新合同内对照，区分水平偏置与下降/对齐协调。
单纯换动作不是部署修复；有行为因果证据后还须追查生成LoRA的相应计算/学习原因。

### 冻结干预执行与资源中断恢复历史（2026-09-24 02:16 UTC）

正式实现`1f583d9e31e55c798ee98833f48ddf740d05247d`已集成推送；运行树为
`/data1/user/ymdai/projects/EMBER-frozen-prefix-formal`。Sol报告66项针对性测试、固定case smoke和pilot16完成。
本批目标仍是1600分支，未改模型、条件、训练或科学参数。

- 首次资源阻塞发生时已完成pilot16与后续四面板396，共412个唯一分支；第五面板
  `B_k50_to_B_remaining`只完成prepare。gpu01 GPU查询多次超时，SSH正常；原控制器exit1和失败日志保留。
  这是资源准入中断，不是模型或科学假设失败。
- 主讨论在2026-09-24 02:16 UTC按节点本机运行原identity-aware helper，两节点GPU和进程归属均返回，
  未知节点/进程检查错误均为空。该快照只能证明查询当时恢复；超时的驱动/硬件根因未定。
  原件在本study的`coordination/resource_recheck_20260924T021615Z_gpu{01,02}.json`。
- 随后的针对性核对确认Sol已自行通过新的双节点准入，启动原`run_remaining_continuation.sh`，
  第五面板使用gpu02的0/1/2/3四张卡；登记状态为`remaining_continuation_starting`，尚无整批完成信号。
  没有重复派发或另起worker；后续等待Sol完成/异常事件，不周期读取运行进度。
- 主讨论只读取阻塞前已完成的八份面板，独立核对412唯一分支、worker exit0、原bank/teacher/RNG及重放误差；
  已登记的28个full cases对应元数据齐全。B/k25的两任务四followers共400分支，前段动作、对象轨迹、
  初态/截断完整sim state和controller状态配对一致；保存state的最大重放误差为0。
  复算代码及原件索引在`coordination/partial_recheck_412.{py,json}`。
- B/k25同策略接手相对旧原行：Object14为45→45（保留43、得2、失2），Goal21为42→44（保留38、得6、失4）。
  这说明新旧执行存在成功集合变化，不能把旧自然rollout与新分支的所有差异算作因果收益。
  不追逐逐bit复现；按原设计，以新执行合同内的配对分支为主要比较，并等待C前段及k50完整结果后统一裁决。

### 四臂结果与下一步理由

Source held58/400、seen9/64；A/B/C/D selected为420/630/420/1050，held108/124/110/123，
对应Source R/G/L为49/59/9、48/76/10、34/76/24、40/83/18。B后四节点高于A，
不支持生成链普遍不会学习；C有能力获得，但保留损失更大。C controls110/98/126、D123/110/80，
D相对wrong有优势，却尚未证明优于语言B的正确视频增量；D末段held丢60得28、seen仍改善。

实际flow probe与闭环不一致，固定画面出现错误对象和同任务反例；历史已有异质失败及空间监督的局限。
据此下一批冻结B630/C420，以两任务、50初态、两种保存动作前段、25/50步截断和四个接手条件组成1600个分支，
直接区分状态历史与同状态的策略/视频条件作用。无训练、无新增held expert actions、无官方Validation/Test或checkpoint选择。
这仍是有反驳条件的定位干预；待原件回报后由主讨论继续分析，不把分支成功率当新部署方法成绩。

### 四臂工程核验及启动历史快照（2026-09-23）

Study根：`/data0/user/ymdai/ember_runs/conditional_compilation_diagnostics_20260923`。
正式运行树：`/data1/user/ymdai/projects/EMBER-conditional-formal`，主讨论实查为clean、detached的`43d801b1`。
`launch/formal_launch_contract.json`登记精确命令、四臂共同commit、资源、预计增长和恢复合同；
`registration.json`已由Sol更新为`formal_training_CD_running`。

- Sol报告CPU针对性检查304+17+5通过。主讨论已核对源码中`SupervisedEngine.validate`归属修复、
  四臂smoke的1..4宏步记录、各自2→4恢复日志及第4步checkpoint manifest；每臂累计448条动作查询。
- `launch/profile_longest_C.json`和`profile_longest_D.json`记录最长fit视频347原始帧→71采样帧、full H50、
  21+7查询及非零活动梯度，峰值reserved约22.69GiB。`materialization_eval_interface_smoke.json`
  记录一次Writer编译、真实adapter加载及1×50×7动作输出，明确`formal_score=false`；这些只证明工程接通。
- D于10:32:12 UTC在gpu01的0/4卡启动，C于10:32:35 UTC在gpu02的0/1/2/3卡启动，合计6张物理卡。
  主讨论18:38北京时间只读核对formal run contract：同一commit、dirty为空、world size分别2/4；
  当时D已18步、C已27步。A/B尚未启动，按登记顺序随后运行；不以此瞬时步数解释科学效果。
- 已核对launch前两节点GPU记录及strg01两文件系统独立quota原件。合同预计data0新增峰值约76.5GiB，低于96GiB预算；
  每次后续launch/resume仍由执行者刷新资源状态。训练约7–8小时是基于profile的估计，评测ETA待首个完整面板实测。

本次只更新运行状态，不改变科学参数、节点或选点规则，不触碰冻结运行树；无需重复启动批准。
继续按六节点完整held400/seen64、Source相对保持/获得/丢失及冻结选点后的C/D视频controls裁决。
执行者完成本批主动回报并停止追加实验，主讨论核对原件后才派发下一项针对性干预。

### 本次接管的证据核验与判断

已按README读取当前合同、concept、findings§127–133，沿历史审计回看7–9月路线、9月11日完整专家论证、
9月19日教学提案及最终修订、近期输出空间/因果/配对报告和实际Core/Procedure/FactorHeads/采样/功能信用代码。
本轮只做历史只读复算，没有新增模型forward、梯度、rollout或Test读取。

- 原始行重算旧v5.2/v6的四个匹配访问面板，确认132/51与95/111的配方交互，仍保留Adam次数/LR/查询数混杂。
- 原始行重算C与夜间fresh各六correct节点及各自other/wrong：16个完整400面板，唯一行、50视频及worker退出通过。
  两模型selected600之间1200行逐臂核对task/state/语言、env/policy噪声共同前缀、真实teacher帧元数据、Source与normalization一致。
  correct154→142是保留94/新增48/丢失60；other160→132是88/44/72；wrong153→105是81/24/72。
  因而correct−wrong从1增至37伴随两种正确输入绝对下降；这是跨运行描述，不是隔离了单一训练机制的因果估计。
- C600局部接口五臂的160条原始行重新计数为9/16/11/11/14（每臂32），没有自由A/B显著占优的依据。
  该检查有task-local、有限预算和步幅校准限制，不能反证所有共享生成的优化问题。

小型复算证据在`coordination/scientific_recheck_20260923.json`；跨轮记录见findings§134。
当前竞争解释、可反驳预测及四臂不能回答的部分写入task_plan；未登记任何第五臂或自动后继训练。

此前Owner认为仅清理缓存不充分，明确要求进一步裁剪历史checkpoint。追加清理按关键权重/历史参照/当前依赖/恢复用途择点，
额外回收360.759 GiB；连同首轮53.007 GiB，两轮累计413.766 GiB。data1用户配额实测从846.3降至485.6/1024 GiB，
data0仍为122.1/1024 GiB。原始评测、配置、关键模型及当前诊断依赖继续保留。

两轮清理按当时暂停研究的要求执行；Luna由Owner停止，**没有创建Sol任务**。
长期理解、文档/代码/缓存与worktree整理、追加资产退休已完成并在`b711c2af`集成推送。

## 追加checkpoint裁剪

- 9月8–14日33条结束路线：回收222.737 GiB，保留70个关键权重路径；8月3–7日12条结束/作废路线：
  回收115.220 GiB，保留24个代表权重及64份完整评测原件。合计174个历史checkpoint：94个`weights_only`、80个`metadata_only`。
- Source1000预先固定使用raw policy；退休无使用用途的optimizer及未选EMA副本，回收22.802 GiB。
  raw policy的canonical inspector在删除前后返回相同结果，正式推理仍可用；Source完整训练恢复和EMA不再可用。
- 合计删除256个大载荷路径、255个独立inode：trainer约287.989 GiB、Source optimizer 14.090 GiB、
  非关键Writer权重49.968 GiB、未用EMA 8.712 GiB。保留原合同/manifest、日志/指标、原始结果和小型rank RNG记录。
- 删除trainer的历史Writer不再exact resume；其原冻结CLI会因完整文件校验而拒绝加载，今后重放需要明确支持
  weights-only的读取路径，不能伪造trainer或重写历史manifest。已导出9月模型的标量/结构元数据；8月模型所需元数据在原合同/consumed中。
  9个受影响的物化缓存退役记录已同步上游可用性。当前checkpoint retirement记录优先于旧日志的“完整保留/可直接重建”描述。
- 94个保留权重的safetensors头均可读；O1200/C600/C1200/MT-BC300的全部manifest文件和大小核对通过，
  库存无计划外消失文件。数据、teacher/native observer/stable carrier/fit19依赖、旧V5.2参照和近期完整训练轨迹未动。
  本轮没有运行模型、GPU实验或重算科研结果。data1已盘点checkpoint载荷从592.366降至231.607 GiB；
  剩余小型历史节点不一概宣称必不可少，但本轮未对缺少逐项退休依据的资产继续扩大删除。
- 精确清单及验证：`runs/analysis/workspace_cleanup_20260923/checkpoint_retirement/closeout.json`；
  每个受影响checkpoint原目录有`checkpoint_retirement.json`，历史manifest保持原内容。

## 接手先读

1. [Owner要求](docs/current_owner_requirements.md)：固定LoRA可达下界并非公共底座课程；有益视频利用、理论解释深度与协作边界。
2. [Concept](docs/concept.md)：完整pipeline，以及生成Jacobian、端点/前缀监督、任务关系覆盖三类待检验解释。
3. [Findings](findings.md)§127–133与[研究历史](docs/research_history.md)：近期原件、正负证据和不能外推的结论。
4. [接续的条件编译设计](docs/designs/conditional_compilation_diagnostics_design.md)：四臂用途、混杂与未覆盖的数据原因。

## 当前实证快照

| 方法／固定checkpoint | coverage Validation correct/other/wrong | coverage Test |
| --- | --- | --- |
| Source1000 | 51／不适用／不适用 | 75 |
| MT-BC300，rank128 | 155／不适用／不适用 | 121 |
| 原跨episode辅助Writer C600 | 154/160/153 | 未跑 |
| 夜间12任务条件fresh，selected600 | 142/132/105 | 未跑 |

所有表中分数均为每臂400条；不同task集合不能跨Val/Test逐行配对。baseline Test是Owner授权的历史暴露任务补测，
不能描述为新盲测，不反哺选点/划分/方法。Source Test更高、MT-BC更低不证明划分无效，也不支持“Test统一更难”。
夜间1200步fresh完整曲线103/117/142/141/138/107；训练侧小试收益没有转为超过155的正式泛化结果。
已完成correct/other/wrong；shuffle/reverse在完成前被停止，没有完整分数；当前EMBER Test、FT、RL均未启动。
目前没有验证出同时解决绝对性能与有益视频特异性的方案，亦没有证据证明架构理论上已到上限。

原件根：
- `/data0/user/ymdai/ember_runs/overnight_root_cause_20260922`：近期有界诊断和小试。
- `/data0/user/ymdai/ember_runs/coverage_task_mixing_20260923`：夜间fresh、完整checkpoint、配对原件与selection。
- `/data0/user/ymdai/ember_runs/coverage_baseline_test_20260923`：Source/MT-BC Test原件，交付commit`7b18030c`。
- [夜间小型结果包](docs/review_materials/20260923/overnight_results/README.md)：当时PPT使用的指标与图，不包含后续baseline Test。

## 原WIP交接与已完成集成

原分支 **`codex/conditional-compilation-diagnostics@73267f53`** 保存Luna未完成实现，仅作历史交接来源。
Sol从最新main隔离开发并逐项修复、验证后，以`43d801b1`集成推送；没有整体恢复旧文档或已退役实现。
当前设计、`experiment_spec.json`和`partition_audit.json`的科学合同保持；运行状态以本文件顶部和study登记为准。

候选fit28＝官方Train中的seen target16＋已审计aux12；diagnostic-held8是其余官方Train任务，官方24/8/8未改。
A直接共享rank16、B语言Writer、C完整视频Writer、D相同视频结构但第二监督组改tau1/前5步；均计划fresh。
这只能定位整体环节，不能唯一分离参数化/活动容量，也没有数据构成干预。旧MT-BC看过这组held8，不能充当A的未见任务结果。
完整分组、步数、节点与每个比较的边界以冻结spec和设计为准；工程验证和C/D启动不构成新的性能结论。

原WIP的`validate`曾错误缩进在`configured_endpoint`的return之后；现已恢复为`SupervisedEngine`方法，
并补齐实际forward/backward、保存恢复及物化/评测接口验证。原WIP分支不作为formal运行来源。
新主讨论依据既定四臂合同推进并独立判断结果；执行任务不能把此交接扩大为任意新实验。

## 首轮收尾记录（追加checkpoint裁剪前）

- 长期要求和concept已整合；docs按designs/analyses/review_materials分工，历史材料有明确状态。
- task_plan和progress只保留当前状态，重复历史移至既有研究历史/发现入口；原详细过程可从Git恢复。
- 已结束的stability/output-space/causal专用诊断和low-LR执行面退役；历史实现以`7b18030c`及run记录的commit恢复。
- 代码retirement净删5101行、删除15个专用文件；保留常规resume、topology migration和1500→2100 continuation。
- 108项受影响CPU检查：107项首次通过；1项引用旧文档路径的断言修正后单独通过。三个canonical CLI的`--help`均exit0。
  40份配置JSON解析通过，Markdown本地链接无失效，源码/脚本/测试无对退役诊断模块的残留引用；未运行GPU或新科研验证。
- 三个临时worktree已移除，仅剩main；已合并的cleanup-code分支删除，未合并WIP的本地/远程分支保留且远端核对为`73267f53`。
- 大资产已完成回收：52个可再生LoRA bank共9739个路径（9339个唯一inode），44.924 GiB；
  明确一次性的profile载荷0.276 GiB；worktree副本0.731 GiB、已消费handoff/下载缓存/派生预览0.246 GiB。
  追加核实574个旧物化manifest、1717个payload，回收6.794 GiB。连同生成cache和本次临时文件，总计回收53.007 GiB：
  data0为38.682 GiB、data1为14.326 GiB。Hardlink按实际inode计数；原manifest和全部重建依赖保留，载荷原目录有退役标记。
  strg01收尾配额实测data0 122.1/1024 GiB、data1 846.3/1024 GiB；这是实时用户用量，不是共享磁盘空闲量。
- 清理账目：`runs/analysis/workspace_cleanup_20260923/storage_cleanup.json`；Source/teacher/正式checkpoint、raw rows与
  不可确认可再生的authority/task-local/轨迹证据保留。data1的大头是约530 GiB formal checkpoint载荷，另有约59 GiB探索checkpoint，
  不能根据负结果或目录名当作缓存删除。277组authority/task-local/特殊转换或provenance不全的产物有逐项保留原因，
  不凭“cache”命名推断可删。旧scratch中的唯一诊断记录和可用uv安装器也保留。
- 最终main包含文档/代码清理与验证修正，按仓库交付规则推送远端；仅保留canonical worktree。
  收尾没有启动新GPU实验、创建Sol或给任何任务派发后继实验。

上述首轮收尾段是历史事实；当前接续授权、实际收件人及四臂启动状态以本文件顶部为准。
