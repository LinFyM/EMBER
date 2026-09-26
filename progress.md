# EMBER progress

## 当前状态：科学主讨论与新执行者均已登记；S0已验收，Reader未实施、未向新执行者派发（2026-09-26）

Owner要求主讨论不因执行者仍在实验或等待派发就停止。当前已完成S0整批原件核验、特征机制审查及后继裁决；
长期目标仍为一次生成完整LoRA的绝对能力超过强MT-BC、有益视频贡献和能力保持，未发现统一根因或验证修复。

**唯一已登记后继设计（未实施）**：[native_conditional_reader_design](docs/designs/native_conditional_reader_design.md) **仅§4工程准备**；
机器合同`configs/native_conditional_reader_v1/engineering_spec.json`，study `native_conditional_reader_engineering_20260926`。
**当前科学主讨论：`01a0dd74-4c71-7c82-8629-8333ef74dfdd`**，按Owner本轮明确指令接管理论分析、实验设计、
原始证据核验与科学裁决；实际身份由本session的`CODEX_THREAD_ID`/`CODEX_SESSION_ID`核实。
**当前实验执行者：`01a0dd6c-f2e5-7971-821a-56766e1c0f22`**，Owner本轮明确指定。
通过只读thread查询核实其标题为“接管 EMBER 实验执行会话”、cwd为本仓；该会话也已按Owner通知记录当前主讨论UUID。
本轮Owner先要求了解主讨论的项目理解与整体规划，故只登记身份、开展科学讨论，没有发送消息或派发工程合同。
后续具体实验由该新执行者接收有界合同；不再联系旧主讨论或旧Sol。
旧主讨论 `01a0cd94-65da-7b22-8ca9-7ba35f454632`仅为交接与历史证据来源，不是当前协调或回报地址。
旧Sol `01a0cd90-ebb7-77a1-a20b-a858825d2f66`已按Owner迁移安排停止接单，不再向其派发。
范围：实现joint Reader R_V/R_L共享接口，各4宏步及2→4 world2恢复、一次最长视频condition反传/profile、
6条train-only canonical接口episode；上限0.75完整GPUh/2GiB新data0/768MiB代码，本批最多6物理卡。
新正式630训练、held、第二臂完整训练、蒸馏、400、S0修复或续训均未激活。工程完成主动Queue，主讨论验收后裁决。
11:11:11 UTC曾将合同2cd9a9bf以Queue送达旧Sol，回执01a0dd69-74ca-7500-ac9d-aa33641b94ed；
11:11:46核对正文进入turn01a0dd69-74cb-77d1-b890-e442c1f85050。进一步核对时才发现迁移记录，
11:12:58确认旧Sol明确拒绝在旧会话启动实现/GPU，turn已interrupted、会话idle；这不是已执行的工程批次。
Owner随后在旧主讨论明确：“新session我还没创建，所以你等一下。”当时因而等待准确的新会话身份，没有自动创建、重发或改投。
正文、接受和未启动回复在`.codex/tmp/native_conditional_reader_engineering_20260926/`。
新执行者现已登记，但尚未派发；Reader仍只保留同一有界工程范围，不重复S0，不把旧Queue回执当成新执行记录。

### 本次接管恢复与原件核验

旧主讨论交接前已消费Owner转入的Sol S0完成信号，并核对
`analysis/video_mapping_audit.json`的448行、固定池映射和空mismatches，与此前独立验收一致。
该延迟回报不触发重跑、重新派发或开启下一阶段。工程study根尚未创建，旧Sol已明确未启动实现/GPU。

新主讨论已从README按交接顺序恢复Owner要求、当前状态、concept、findings§156–159、特征机制分析、
Reader设计/spec与相关历史论证；接管时登记了自己的实际UUID，执行者当时为空，随后由Owner指定为上述新会话。临时交接页
`.codex/plans/ember_scientific_handoff_20260926/state.md`已消费并按其生命周期删除，持久结论仍以跟踪文档为准。

本session直接读取S0的8个原始`evaluation/*/*/results.json`，对照448条汇总行，核对语言/seed共同前缀、
C0/S0条件一致，重算三组主要R/G/L及各20000次task-bootstrap，均与§159一致；24full数量只复核metadata。
从已退出的两臂`training/*/metrics.jsonl`核实630更新/70560查询，以及Action Meta活动区间C0为3..630、S0为450..630。
另沿fixed400与E/H批次的`source_results`逐行回到原results，复算1600行的57/121/120/132及320行的20/24/20/27。
本次没有重载轨迹/权重、重做整批训练/trace验收或运行任何模型；先前完整验收原件保持原状。
旧固定native reader原件确认其末点24/24任务的留出FM弱于学生，这是一项离线拟合反例，不能冒称Reader闭环比较。

接管后的科学分析见[机制分析§12](docs/analyses/feature_to_operator_mechanism_20260926.md#12-接管后的科学判断可迁移知识与实际执行作用)。
Reader仍是联合参数化的有界候选；新推导区分其状态调制与状态选择教学内容两项作用，单次涨分不能唯一归因后者。
不改变已有特征/监督/预算与投资线，不开启新探针、蒸馏或实验。工程study根核查仍不存在。

### S0完成与科学判断

研究根`/data0/user/ymdai/ember_runs/learned_initial_content_causality_20260926`；完成标记之后才读取整批分数。
主讨论核定448唯一行、224新S0银行/224原C0引用、24full、全部T+1动作/谓词/初态和配对seed；
8阶段worker正常退出，14有向比较和20000次task-bootstrap独立复算一致。
`coordination/main_recheck.{py,json}`、`main_training_recheck.{py,json}`保留检查与原始来源。
完整事件plan、2520次exposure、70560查询和LR匹配；正式GPU4f6c75e4，两个held-other仅CPU校验窄修b9e15410。
完整计量4.669167 GPUh≤9，study约4.48GiB≤10。原bank/有效行未重跑，无登记科学原件缺项。

C0/S0 held correct27/22、other27/17、seen26/22；主correct差+6.25pp，描述区间[-3.75,17.5]pp。
正确面板R/G/L=18/9/4、other=14/13/3、seen=13/13/9。完整内容优势不均匀，correct主要来自Object14，
Goal21反为-1。历史C0三面板24/30/27到27/27/26的变化另列，不混入主比较；详见findings§159。

读分前的训练审查发现S0 Action Meta只在450..630共181步非零，C0从3开始共628步；
105..420四个S0 checkpoint的过程读出/调制与Action Meta B仍为0，525/630已非零。
[机制分析§11](docs/analyses/feature_to_operator_mechanism_20260926.md#11-首帧学习结果与可证明的初始化交互)
证明首帧常量P、中心化和零调制在精确算术下构成双零不动子空间；实际Compiler合成CPU核对通过。
这使S0不仅改变信息，还改变有效学习路径。它不是实施错误，不解释完整C0与语言近似持平，不触发精度/修首帧实验。
S0有限对照至此结束；支持后续内容在现预算的有限净收益，尚未隔离操作关系、顺序或信息必要性。

### Reader投入的理由与尚未验证的部分

机制分析§10已经纠正：当前LoRA的BAh本来就有query级信用；线性K/V读出与其前向及梯度可以完全等价。
新候选实际改变的是按执行h重新选择完整C/P并进行非线性调制，作用于原生index9的真实q/v投影。
它是执行中消费预计算教学memory的诊断教师，不能冒充最终EMBER单LoRA部署；强教师及其可编译性均未建立。
旧固定native reader、Semantic-Path、统一Writer与输出头反例继续约束这项有界投入，不用“原生/完整”保证成功。
教学侧保持原C0 agentview/256patch，执行/full采集双相机；不增加视角共变。

只先核定真实接口与成本。后续R_V固定630学习及224面板仍须主讨论在工程结果后正式登记；
若登记，沿此前读S0分数前冻结的投资规则：correct至少35/80（max(C0=27,S0=22)+8），
other至少27/80，correct相对C0至少两个suite净增，才默认值得投入R_L对照。
这只是投资线，不是模型资格/显著性。无实质能力便下调该路线，不自动换层、增rank、KD或去噪探针。

### 已完成S0的原登记与派发记录

历史design：[learned_initial_content_causality](docs/designs/learned_initial_content_causality_design.md)，
机器规格`configs/learned_initial_content_causality_v1/experiment_spec.json`，此批现在不再执行。
只新增S0 fresh630，同模型/fit28/70560queries/world2/LR，广播每视频E0/H0/I0并沿时间求和反传；保留原时钟。
原C0@630及F银行只读。两模型各held correct80/other80/seen64，共448新闭环、24full/全T+1。
合同e9e3cd30及补充d9a01caf于06:25:54 UTC Queue给上述Sol，回执01a0dc64-4912-7f50-94c6-2931c611d716；
06:26:20核对正文进入turn01a0dc64-4915-7873-b02a-962c74083c00并获明确接手。
完成Queue回执01a0dd5b-5f04-7773-a8d4-68b79b074345已核对；本轮通过完成标记主动消费原件，不等待重复消息。
正文/派发回执在本仓`.codex/tmp/learned_initial_content_causality_20260926/`，主讨论复核脚本也保存在study coordination。

### 已完成native-feature批次的原登记与派发记录（2026-09-26）

当时设计：[native_feature_change_causality](docs/designs/native_feature_change_causality_design.md)，
规格`configs/native_feature_change_causality_v1/experiment_spec.json`。以下保留当时登记；本批现已完成，不再启动。
实际执行者仍为Sol `01a0cd90-ebb7-77a1-a20b-a858825d2f66`，主讨论 `01a0cd94-65da-7b22-8ca9-7ba35f454632`。

fixed400新1280条及原320条已从原results/合同独立复核；全部新trace、视频50覆盖、配对种子前缀与全部统计通过。
原件及主讨论检查：`/data0/user/ymdai/ember_runs/language_content_path_fixed400_20260926/coordination/main_recheck.{py,json}`。
Source/B/C0 correct/other完整400为57/121/120/132，新增320为41/100/96/102。
相对语言的两个task-cluster区间均包含0，Goal21旧大幅缺口未复现；视频有益增量仍未识别，findings§154。
新GPU统一d18a8a0e，原行/训练dca1b550，B银行43d801b1；五个GPU进程完整计时5.904167小时，低于8，科学矩阵无缺项。
只读CPU字段名修正均保留，不改变任何模型或原行。Cplus学习分支继续关闭。

接续固定C0@630，80原正确条件只编码一次；分别保留E/H逐帧变化或广播各自首帧，H初始化投影同步重算。
四格320完整LoRA/320闭环，pilot8计入总数，全T+1/16full；保存原生输入至Compiler的特征与正式首轮动作。
不新增训练、独立function query、teacher action/state/reward、节点、wrong/shuffle/reverse或官方Validation/Test。
预算2.5完整GPU-hours/6GiB新data0/768MiB代码，本批最多6卡；00不是learned static baseline，混合格分布偏移边界写入合同。
本批用于确定逐帧特征是否有净执行价值，不把特征差异或对照退化当作根因、方法资格或视频必要性证明。
工程/资源准入通过后按同一合同接续，完成主动Queue主讨论、停止新增实验。
合同/裁决在1edffbf3推送；2026-09-26 04:11:45 UTC（北京时间12:11）以Queue派给上述实际Sol，
回执`01a0dbe9-7661-7b92-99e6-b2cc23344882`。04:12:29核对完整正文进入新turn
`01a0dbe9-7662-77f0-95b1-8a6e7a3e1b32`，Sol明确接续同名批次、核对合同并实施有界改动；没有重发。
正文/入队/逐字送达和接手回执位于本仓`.codex/tmp/native_feature_change_causality_20260926/`。

### 已完成fixed400的原派发记录

当时设计：[language_content_path_fixed400](docs/designs/language_content_path_fixed400_design.md)，
机器规格`configs/language_content_path_fixed400_v1/experiment_spec.json`；它只授权1280条有限补评，不授权新训练。
执行者仍为Owner指定Sol `01a0cd90-ebb7-77a1-a20b-a858825d2f66`，主讨论身份保持。

上一批F=dca1b550两臂630宏步/70560查询、448新bank、10面板736条/28full/全T+1已完成。
主讨论直接核验原results/run contracts/trace/训练事件与查询，复算全部配对和bootstrap，
证据在`/data0/user/ymdai/ember_runs/language_content_path_causality_20260926/coordination/main_recheck.{py,json}`。
C0/Cplus正确24/16、另一正确30/19（各80），seen均27/64；保持预测未兑现，findings§153登记关闭此参数化。
原C0与B均丢Source成功5条，原相对缺口未在本screen复现；C0对B的小优势区间仍宽，不能直接称视频修复。
原批已计时15.35 GPU-hours，工程端到端时长未知；不能把16.35当实际总上界。科学矩阵无缺项，计量缺口保留。

接续只固定C0@630/B630/Source，held8×states10..49，C0 correct/other+B+Source共1280新增，
640新bank、320既有B引用、16新full、全行T+1，引用原四格320行组成各400。
预算独立8 GPU-hours/6GiB新增data0/768MiB代码；CPU范围/来源/调度改动可用新E，科学GPU计算不变，
事前允许原行F/新行E并逐行登记，不热改F或为单提交形式重跑有效行。完成主动Queue并停止新增计算。
设计/spec及裁决已在ab2dd647推送。2026-09-26 01:28:40 UTC（北京时间09:28）以Queue派给上述实际Sol，
回执`01a0db54-26dd-7203-a18e-a1ebf9716939`。01:29:51核对完整正文进入新turn
`01a0db54-26df-71f3-ad41-4b4544a7f43e`，Sol明确回复按fixed400、隔离实现、只补states10..49和退出事件接续。
正文/接受/送达保存在本仓`.codex/tmp/language_content_path_fixed400_20260926/`。
原生即时状态查询未返回，复用已核实app-server作一次有界只读核验后通过官方CLI入队；未重发或创建任务。
以上是当时接手记录；本批现已完成且独立验收，不从该历史记录继续补评。

### 已完成两臂学习的ready与派发记录

当时设计：[language_content_path_causality](docs/designs/language_content_path_causality_design.md)，
机器规格`configs/language_content_path_causality_v1/experiment_spec.json`；该矩阵现已完成，不重复执行。
Sol隔离CPU提交41a0f4f4已ff集成main：2项构造与10项旧回归通过，主讨论审阅非零head完整输出和9行源码、
独立复跑2项通过；a0对C及构造对B maxabs0、视频扰动 .44040、a梯度/内积 .07022614。
CPU回执在`/data1/user/ymdai/projects/EMBER-language-content-path-cpu-dev/.codex/tmp/language_content_path_cpu_result.json`。
这些仅验证合成构造，不是原生读取、学习收益或根因证明；该CPU阶段GPU实际启动为零。

正式实现F=`dca1b5500ac0f912d56cc1c76382004457e807a4`已推送，冻结树
`/data1/user/ymdai/projects/EMBER-language-content-path-formal`为clean detached。
Sol报告CPU32项、两臂world2各1..4宏步及2→4恢复、Cplus最长视频反传、两臂真实LoRA→10flow接口通过。
主讨论已审阅F相对学习派发提交的改动、两臂配置、训练/物化/评测范围约束和被动采集接口；
核对上述冻结树身份、已结束smoke的completion/checkpoint manifest、两臂50×7动作接口原件及启动脚本。
源码中唯一学习机制改动仍为FrameRead后零初始化共享标量`a*q`；附加读数只汇总已有forward，
标量梯度在完整跨rank归并后、clip前记录。两臂均从fresh初始化训练630更新，不从smoke恢复。
启动合同与精确命令在study根`launch/launch_contract.json`、`formal_C0.sh`、`formal_Cplus.sh`。
合同预算包含工程和正式计算；Sol估计并行训练2.1–3小时，物化/评测另计且总额仍限18 GPU-hours。
本次ready是工程准入完成信号，不是训练完成或科学结果；实际启动/退出以各臂回执为准。
Sol已获同一合同下实时双节点准入后直接启动、按退出事件接续的授权，不再追加审批或派发重复任务。
主讨论未轮询正在执行的训练日志、checkpoint或缓存；完成信号到达后独立核验原件并裁决。

新范围为C0/Cplus各fresh630，仅一个内容路径开关不同；只评固定末点630的736条，448新条件bank，
144个B630旧bank引用，28full，全行T+1；Source/B均重新闭环。预算18 GPU-hours/12GiB data0/768MiB代码。
重点检验额外能力损失能否减少且原C新增能否保住，正确及另一正确视频是否超过语言基准。
新合同在1bd4e2a7推送；2026-09-25 18:32:14 UTC（北京时间9月26日02:32）以Queue派给实际Sol，
回执01a0d9d6-e3b0-7c82-af73-2d7f5862a122。18:33:36核对完整正文进入新turn
01a0d9d6-e3b2-7922-91bb-10f21d29d2fc及Sol明确接手实现/验收后冻结启动。
正文/回执/送达保存在本仓`.codex/tmp/language_content_path_causality_20260926/`；
已授权其技术/资源验收通过后直接按同一合同接续，不等待Owner醒来重复确认。
CPU完成回报已消费，后续收到同一Queue信号不重复派发；新学习完成后仍由主讨论独立复核原件再裁决。
033c19db只澄清日志时点：a在更新前后读值，完整归并后clip前读梯度；forward只汇总已有q/A读数。
18:35:13 UTC以Steer送入同一新turn，18:36:19核对逐字送达与Sol明确确认接入；
证据同目录`scalar_logging_{steer.txt,receipt.json,delivery.json}`，未改变科学变量、优化或总规模。

### 已完成CPU批次的派发与授权记录

Owner再次强调以EMBER最终目标为中心、避免局部问题递归，以及架构/训练的数学合理性与历史解释；
已纳入稳定Owner要求。主讨论完成[语言内容路径审计](docs/analyses/language_content_path_audit_20260926.md)，
结论与边界见findings§152。候选函数包含仅为可达性证明，不是已验证的原因或收益。
接续范围仅该文§7 `language_content_path_cpu_20260926`：实际Sol隔离实现与CPU验证，无GPU、真实输入或闭环，
当时不集成main、不改变canonical默认行为，完成主动回报后停止；该CPU阶段现已完成，由主讨论验收后集成。
合同在248c566d推送；2026-09-25 18:10:43 UTC（北京时间9月26日02:10）以Queue送给上述实际Sol，
回执01a0d9c3-3457-73f1-8f00-9381c414499b。主讨论核对完整正文进入新turn
01a0d9c3-345a-7d32-9b99-95cc35a4ed79及Sol明确回复只做CPU、隔离分支、不集成/不启GPU。
正文/接受/送达与接手回执暂存本仓`.codex/tmp/language_content_path_cpu_20260926/`；本段保留持久接续身份。
Owner随后要求按上述原则持续推进，休息后希望看到能改变判断的进展；该CPU批次本身未授权GPU，
后续GPU范围由上面的独立学习合同登记并派发。

### 持续授权与已撤回flow分支

Owner最新指令是由本任务接管主讨论与科学决策，以历史证据、竞争机制和可反驳干预推进；具体实验交由现有Sol执行，
每批完成主动回报，待主讨论分析后再派发下一步。这取代旧全局暂停；不恢复旧deadline或GPU特例。
Owner随后明确给予充足时间和持续优化的广泛分析/实验授权：主讨论按证据自主推进后继机制验证和改进，
不能停留在现象、随意试改法，必须逐轮保存假设、预测、原始证据与经验。授权覆盖四臂之后的研究，
当前已启动批次仍按冻结合同执行；科学信息墙、资源限制和结果可追溯要求继续适用。
Owner又明确本次授权**不向Sol转发**，让其专注具体实验；主讨论自行维护长期判断和记录，收到本批结果后再给出具体下一步。
主讨论：`01a0cd94-65da-7b22-8ca9-7ba35f454632`（接管 EMBER 科学决策与实验）。
实际执行者：`01a0cd90-ebb7-77a1-a20b-a858825d2f66`（当前标题“接管 EMBER 实验”，Owner指定Sol）；已核对同仓库/主机，最近128条探索目标交叉已完成。
双方保持现有模型配置；旧Luna和旧主讨论仅作历史provenance，不再作为收件人。
**下列为已撤回flow分支的历史停点。** d7f500f0的flow_path_intervention已由主讨论撤回实施；
Owner指出其没有解释通用VLA训练/采样性质为何是EMBER特有问题，复核与裁决见findings§151。
撤回原32query/64闭环的新增实现及GPU计算，不把未测假设记成科学阴性；保留已有工作与原件。
主讨论继续理论与历史证据综合，不从任何旧active恢复实验，不仓促派发替代矩阵。全项目长期研究授权仍有效。

2026-09-25 17:18:33 UTC（北京时间9月26日01:18），主讨论以Steer修正原Sol当前turn
01a0d97c-56c2-7291-ab9c-e5494c7b1141，接口已接受。要求停止新增GPU进程/query/episode，
若已有工作则在最小原子query/episode结束后停止接续，保存私有工作树和原件，不影响其它任务。
17:20:59 UTC核对完整正文已进入该turn及Sol明确停止回报，turn已结束；随后读取新study的
`coordination/stop_point.json`：本批GPU进程/时长、正式query、闭环、full case均为0，无活动控制器。
正文/接受/逐字送达与回应在objective-alignment study `coordination/flow_path_withdrawal_{dispatch.txt,receipt.json,delivery.json}`。
已有CPU实现保存在`/data1/user/ymdai/projects/EMBER-flow-path-intervention-dev`，本地分支
`codex/flow-path-intervention-20260926`，提交610c136de634ae8f7463cd8e1248a9a275324c66；未推送、未冻结、未集成。
Sol记录30项针对性CPU测试通过，不构成GPU或科学验证；本批原件根
`/data0/user/ymdai/ember_runs/flow_path_intervention_causality_20260926`只含CPU检查及少量配置。

### 已撤回flow诊断的原派发记录

原design/spec在d7f500f0ed69bfb7f88a0e27ef2abeb2e7c235be推送，拟定32保存query/64闭环、最多2 GPU-hours。
2026-09-25 16:53:19 UTC以Queue派给实际Sol，回执01a0d97c-56a3-7c22-a450-3eadbcc4ec7d；
16:54:09核对完整正文进入上述turn及其明确接手实现。原消息已由17:18:33的Steer覆盖，不再作为启动许可。
原正文/接受/送达与回应在同一旧study `coordination/flow_path_{dispatch.txt,dispatch_receipt.json,delivery.json}`。

### 最近完成：探索目标对齐及主讨论裁决（2026-09-26）

Study `/data0/user/ymdai/ember_runs/return_objective_alignment_causality_20260925`。
E=29634cc9f75143cdd6d70863e6d24ef1a8d3770d保留16bank/4pilot；按59e7c1ad例外，
E2=61974dee6047ecf86a2155ab63219367f7c1f51d完成余124及分析。只有CPU验收/精确来源绑定修正，GPU科学计算未改。
128唯一行、16full、全T+1 trace/谓词、全部worker exit0；主讨论独立审阅diff并逐行重算五对比、联合bootstrap、
6382次注入/完整seed/实际动作、首轮函数和资源，`coordination/main_recheck.{py,json}`保存原件。
P_J0/P_JS/RB_J0/RB_JS为18/19/15/14，任务breadth8/8/6/6；RB−P为−3/−5，交互−2（均/32）。
探索目标收益和原条件迁移预测没有兑现，区间宽，不是总体无效的证明；完整裁决findings§150。
§147–150有限回报分支关闭，不补seed/scale或长RL；统一根因、可靠修复、有益视频必要增量仍未验证。
完整GPU回执.856028747小时、data0约1.2GiB、两代码树约504MiB，缺项0。

### 已完成探索目标交叉的协作与工程记录

原设计/spec在273e23c1推送；Queue回执01a0d926-13ce-7252-a9e1-1605a10e52d4，
15:19:57 UTC核对完整正文进入turn01a0d926-13d1-76a3-8027-5257bc9422b6及Sol明确接手。
正文/回执/送达在score-update study `coordination/objective_alignment_*`。
E的pilot误要求成功即停的完整seed列表等长；主讨论核对每条完整公式和共同前缀，未按成绩放行。
59e7c1ad阶段例外于15:50:30 UTC以Steer送入同turn，15:51:08核对逐字送达和Sol明确确认，
证据在本study `coordination/pilot_seed_exception_*`及`main_pilot_pairing_recheck.json`。
E原16bank/4pilot未重做，错误回执保留；本批现已由E2完成，不恢复历史阻塞或自动追加。

### 最近完成：score-update stage1及主讨论裁决（2026-09-25）

统一正式实现/候选/物化/评测34ea27bdd5fa5064e16f821edf5e1ebc0968253c，根
`/data0/user/ymdai/ember_runs/return_score_update_causality_20260925`。
48bank/96唯一行/12full及全T+1 trace齐全，全部worker exit0，失败正式episode0。
主讨论核对每行completion/轨迹/trace/候选权重，独立复算三对比、bootstrap和首轮函数，
`coordination/main_recheck.{py,json}`保存证据；完整判断findings§149。
P/RAW/RB为15/13/15；RB−RAW为13/2/0，新增都在teacher46，且不是恢复RAW丢掉的父成功。
RB−P为12/3/3，RAW−P为12/1/3；相同的三条父成功仍丢失，任务breadth6/4/5。
强能力保持预测未兑现，不续训或补states34/35；下一项只检验旧批事前保留的探索目标差异。
同alpha更新步长RAW/RB约.170765/.180095，固定77buffer未变；首态夹爪sign三对比均未改变。
旧P同subset18→本批15单列，不混入主对照或据正常差异重跑。
实测GPU计时合计.7020605942小时，data0 1221.21MiB/代码504.16MiB在限内。

### 已完成score-update的派发记录

设计/spec及§148裁决在`325a20b0`推送；Queue回执`01a0d8e1-6298-7b61-9302-4269e510c783`。
14:05:02 UTC核对完整正文进入turn `01a0d8e1-629b-7b71-b998-77b7e9716803`及Sol明确回应。
正文/送达/回应在score-conditioning study的`coordination/score_update_{dispatch.txt,delivery.json}`；没有重复派发。
本批现已完成，不从旧active记录恢复其它state。

### 最近完成：冻结信用分解与主讨论裁决（2026-09-25）

实现d6348660399d0a24841af30ed9a98ff64b5078e3，根`/data0/user/ymdai/ember_runs/return_score_conditioning_20260925`。
512decision/96非零输入、192真实十步flow、五活动task×parity/11零组、16梯度文件齐全，无参数更新/新环境步。
主讨论审阅条件score、实际bank与完整Writer反传，从所有原latent及向量独立复算身份/score/分组和方向；
`coordination/main_recheck.{py,json}`保存证据，findings§148为完整裁决。
RAW→RB相对改变量.587716、cos .837657（33.106度），RAW对旧E2重算差.001609；这不是无关紧要的数值变化。
偶/奇cos −.026864→−.024499、task37份额.68625→.71362，未验证信用一致性改善，更无闭环收益。
两者范数286.5847/302.2425：后继固定同一SGD系数，RB步长约高5.46%，不声称等函数/参数步幅。
两卡260.816秒=.144898 GPU-hours，CPU阶段未计时；data0 740MiB、代码约505MiB，在限内。

### 前批冻结分解的启动记录（已完成，不再恢复）

机器合同`configs/return_score_conditioning_v1/experiment_spec.json`。复用前批512保存decision/128条采集与父C_S00@1155，
仅对96个非零信用decision计算RAW与按实际夹爪sign条件化的RB两种完整梯度，最多192次真实十步forward。
不更新参数、不建候选或bank、不读新标签、不新增环境初始化/闭环；完成后主动Queue，停止新增实验。
新study `/data0/user/ymdai/ember_runs/return_score_conditioning_20260925`，data0≤2GiB、data1代码≤768MiB，
全计算含初始化/工程/失败≤1 GPU-hour，最多同节点两卡、项目仍≤6卡；由Sol负责实际资源准入与统一冻结实现。
科学合同及§147裁决在`91d63a18`集成推送。使用Queue派给上述原Sol任务，回执
`01a0d8ae-cc20-73e1-9a39-42dfd0182333`；13:09:29 UTC核对完整正文进入inProgress turn
`01a0d8ae-cc23-7563-b17a-c304e58c3516`，并读到Sol明确接手本批、核对合同/只读来源及隔离实现的回应。
上述是当时接手记录；本批现已完成，裁决见上。正文及逐字送达/回应在旧return-credit study的
`coordination/score_conditioning_{dispatch.txt,delivery.json}`；未重复发送。

### 最近完成：成功信用方向及主讨论裁决（2026-09-25）

原采集dd2e00bc、完整梯度/候选/新bank/评测4e3ade36；128采集、512保存decision、1536重放、256闭环、384trace/16full齐全。
主讨论从原始面板/轨迹/trace和权重独立复核全部六个对比及bootstrap、参数步和函数分解，见findings§147。
P/R/NEG/FM为35/26/26/29，R/NEG总分相同但各有6条独有成功；R未兑现收益预测，不启动长程RL或尺度扫描。
仅6/32组非零信用，偶/奇初态梯度cosine−.02686；task17的R/NEG均0/8，首态已有较大的相反平移变化。
仍未分开信用方差、状态异质性、有限步非线性、J_Sigma/J0及video/state迁移；没有统一根因/修复。
独立CPU检查还发现96个非零decision的夹爪均值均距开合边界>9.5sigma；其随机幅度贡献3.7131%的动作端score平方和。
该比例不代表Writer方向占比；下一项只检验它经完整Jacobian的影响，不直接接受架构/训练修改。
原件根`/data0/user/ymdai/ember_runs/return_credit_direction_causality_20260925`，主讨论核验为
`coordination/main_recheck.{py,json}`、`main_gripper_score_{audit.json,rows.jsonl}`、`resource_recheck.json`。
有回执GPU时长3.339812h，另.05h为未精确计时诊断的预留而非验证上界；data0 5601MiB/代码504MiB在限内。

### 前批成功信用方向的启动记录（已完成，不再恢复）

机器合同`configs/return_credit_direction_v1/experiment_spec.json`。固定C_S00@1155、八个合法fit任务，
一次128条探索采集产生共享回报方向；比较其正/反方向、同teacher原FM方向及未更新父。
候选各自只做一次fresh SGD、固定参数步长，不继承父Adam、不做连续训练或部署适应。
随后独立states32..35与teacher46/47最多256闭环、16full；总384正式episode，工程smoke最多4。
没有held输入/梯度、官方Val/Test、strict400资格或新视频必要性声明。全组无reward变化则报告未识别并停止候选评测。
新data0≤8GiB/data1代码≤768MiB，正式≤4 GPU-hours、项目≤6物理卡。先验收现有evaluator/真实10-flow VJP，
由Sol负责隔离实现、准入、执行和退出后主动Queue。合同及§146裁决已在`99a71271`集成推送。
09:26:28 UTC（北京时间17:26）以Queue派给现有Sol，回执`01a0d7e3-3cae-7380-b13d-21d204fc5da1`；
09:27:45核对完整正文进入inProgress turn `01a0d7e3-3cb1-7503-a115-5a14f19c10b5`，并读到Sol明确接手
合同核对、隔离实现、工程验收及冻结执行的回应。当时为已接手实现；本批已完成，最终核验及裁决见上。
正文/回执/逐字送达与回应在metatask study的`coordination/return_credit_{dispatch.txt,dispatch_receipt.json,delivery.json}`。

### 历史工程节点：梯度重放超限及一次阶段例外（已由E2完成）

正式GPU实现dd2e00bc，128采集/32组/512保存decision已齐；主讨论独立核对组身份、八个bank引用、
所有128条T+1与谓词。6组LOO非零，分布于task12/17/22/37；不根据这些结果改变任务或权重。
world2梯度在task22全task replay RMS=.022431>.01退出1，无候选、完整梯度或新评测；部分梯度封存。
原件根`/data0/user/ymdai/ember_runs/return_credit_direction_causality_20260925`，
`launch/gradient.exit.json`及`audit/gradient_{flow_parity_one_decision,writer_condition_one_teacher}.json`。
单点实际bank的canonical/可微flow一致、对旧均值RMS=.001674；重编译LoRA相对L2约0.28%，尚未证明全task误差已解决。
另发现采集显式TF32、梯度及诊断脚本未对齐该设置；需一次直接验证，不视为已验证唯一原因。

按design§6/spec登记：保留全部dd2采集及bank；验证后由唯一新冻结E2完成完整梯度/候选/replay/评测，
允许一次明确阶段来源例外，旧树不热改、不重采集、失败部分不混入新方向。
策略VJP固定实际采集bank，Writer重算完整Jacobian并记录差异；原score的mu_old、.01阈值、步长及全部科学参数保持。
至多64个原task22 decision用于修复验证，优先并入E2正式检查；原4 GPU-hours/8GiB/768MiB总预算包含失败及修复成本。
裁决在`5b138fcf`集成推送。11:07:57 UTC以Steer送入Sol当时活跃turn
`01a0d83d-e3d0-7af1-9641-7976a86a3c8f`；11:08:36逐字核对完整正文进入同一轮，
当次尚未读到后续明确回应。只记录已送达，不称实现已修复；实际E2/真实验收由执行者接续登记。
正文/接受/送达及主讨论核验记录在本study的`coordination/bank_replay_exception_*`，未重复发送。

### 最近完成：lookahead七组及主讨论裁决（2026-09-25）

唯一实现`99c6491a27f68d124bb4a13d5e23ed1c2632665b`，根
`/data0/user/ymdai/ember_runs/metatask_lookahead_credit_20260925`；七case、224bank、3584FM、448实际flow，全部exit0。
主讨论审阅真实权重/梯度路径并从所有rank rows/flow NPZ独立复算六个bootstrap和配对、有限分解、实际步幅，
原件`analysis/`，核验`coordination/main_recheck.{py,json}`。完整解释见findings§146。
TASK−BASE独立FM及前缀MSE均无收益依据；TASK−MIX前缀差约94%集中在单个夹爪query，不能当普遍修复。
1.224%最大分组梯度余项限制细小修正归因，不据此追加精度试验或认定工程错误。
本批关闭，不延长lookahead、不把单步代理当闭环。case核心计算计时1.0195 GPU-hours（不含初始化），data0新增3.294GiB。
原派发为Queue message `01a0d77d-2983-7970-b915-a2b5c93210bc`，turn `01a0d77d-2988-7ef1-8a81-e13ad54bccca`；
当时完整正文与Sol明确回应已核对，记录在前批support-slot study的`coordination/metatask_lookahead_*`。

### 最近完成：单项监督的有限学习与裁决（2026-09-25）

正式唯一实现/训练/物化/预测/闭环`e58e8169ed16b0a3aa65601d2a508184e447c126`，
根`/data0/user/ymdai/ember_runs/support_slot_credit_causality_20260925`；三臂84宏步、400闭环、16full、400连续trace，22份退出回执均0。
主讨论审阅真实fork/当前信用恢复实现，独立复算所有原始行、配对/谓词/动作、12对比bootstrap、首slot30预测及128供体FM配对，
原件`analysis/`、核验`coordination/main_recheck.{py,json}`。P/KEEP/SWAP/DROP：Object46/41/43/39，Goal40/42/45/48。
SWAP首步五个固定Goal初态均远离plate，反驳预定的局部有害新增方向；KEEP对DROP的Goal净−6例，未兑现保护预测。
DROP虽保住KEEP全部Goal成功并新增6例，但Object丢3得1；没有联合修复或新视频增量，完整边界见findings§145。
实测data0新增4.423GiB、保守正式1.954 GPU-hours。本批关闭，不自动补评或延长28步。

前批[单个任务当前信用的有限学习干预](docs/designs/support_slot_credit_causality_design.md)的启动记录：
机器合同`configs/support_slot_credit_causality_v1/experiment_spec.json`。同一C_S00@1155完整起点，
KEEP77/SWAP76/DROP77各28个更新（1156..1183），固定另27任务；加未更新父参照，两任务共400闭环。
首次被干预更新1160后30次真实10-flow、合法76/77共128条无梯度FM读出，16full及全400条连续trace。
唯一问题是区分76监督的有害新增、77监督的保护丢失和共同学习/历史路径；不冻结模块、扩训或自动补旧矩阵。
父完整checkpoint已核对，原两training_events的108个共同事件完全相同、4个替换slot与有效query预算已CPU审计。
新根`/data0/user/ymdai/ember_runs/support_slot_credit_causality_20260925`；data0≤8GiB、新代码≤768MiB，
每训练臂world2、项目≤6物理GPU、正式合计≤8 GPU-hours。实际资源准入、实现验证、完整恢复与执行由Sol负责。
设计/spec与§144裁决已在`22c762b90606e7e0047492c78f1ddf37893ddfbc`推送。
04:55:07 UTC（北京时间12:55）以Queue派发给现有Sol，回执`01a0d6ea-ce93-7fc3-b4a8-450f346c70dc`；
04:55:38核对完整正文进入inProgress turn `01a0d6ea-ce98-7402-84f9-85d05376a818`，
并读到Sol明确接手fork/物化/passive接口审阅、验证后执行84宏步与登记冻结评测的回应。
正文、入队回执、逐字投递及回应保存在native-reader study的
`coordination/support_slot_{dispatch.txt,dispatch_receipt.json,delivery.json}`。
上述是当时接手记录；本批现已完成，科学裁决见上文和findings§145。

### 最近完成：原生读取与其余Writer交叉及科学裁决（2026-09-25）

正式实现/新评测`edca1a3548af538d5724431ef53a7293653eb42b`，clean pushed detached树
`/data1/user/ymdai/projects/EMBER-native-reader-transfer-formal`；两个self bank来自原E3，两个混合bank各100次新完整生成。
研究根`/data0/user/ymdai/ember_runs/native_reader_transfer_causality_20260925`，8面板400有效行、16full、400条T+1 trace，全部worker exit0。
主讨论检查实际N/W载入、self恢复，独立复算所有原始行、12组R/G/L、16对比及bootstrap、100首轮分解和连续几何，均一致；
原件`analysis/`，复核`coordination/main_recheck.{py,json}`，后验plate方向`coordination/goal_bowl_plate_projection.json`。
四格N0W0/N1W0/N0W1/N1W1：Object45/42/36/37，Goal39/38/19/28；N只含三Meta、W为其它全部可学习映射。
Goal换W两背景的描述性区间均在零以下，N1在W1下净恢复9条；不能据此冻结N、归罪单一head或把混合当修复，完整裁决见findings§144。
Goal先动非目标物主要是plate；W首轮沿黑碗→plate的命令偏移50/50同向。该后验方向将接受下一批事前检验。
新self相对旧stage1有正常成功集合变化，主要对比只用新四格。实际保守1.091 GPU-hours、data0新增2.18GiB。
本批关闭，Sol已停止新增实验；主讨论继续设计并派发，不将执行者的停止点当成主讨论停工理由。
原派发使用Queue，回执01a0d69b-dec0-7ed1-8dcb-9dc99e21f7a1及正文接收记录，
保留在关系支持study的`coordination/native_reader_transfer_{dispatch.txt,dispatch_receipt.json,delivery.json}`。

### 最近完成的关系支持批次及历史启动记录

前批[任务关系支持](docs/designs/relational_support_causality_design.md)及其机器合同保留；主讨论完成metadata/完整BDDL/可辨识关系审计，
固定四个fit28池、四C与两B、1260因果对比；其1500第一阶段已关闭，不因旧六节点或9932记录恢复自动执行。
设计已在`6972486e`集成推送。2026-09-24 09:19 UTC以Queue派发，message
`01a0d2b6-24e5-7ae0-a1a9-30077bd8cc4e`；09:19:59 UTC从现有app-server核对**完整正文**进入Sol的
inProgress turn `01a0d2b6-24ee-7401-bdb0-7ea22629e35b`，当前cwd及实际标题亦已核对。
正文/入队回执/逐字投递核验保存在前批readout study的
`coordination/relational_support_dispatch_20260924.{txt,json}`与`relational_support_delivery_20260924.json`。
上述Queue记录证明当时已接收任务；随后实现、正式启动及工程补验记录见下，现已完成的科学裁决见findings§143。
Owner明确纠正“下一合同尚未定稿便结束主讨论推进”的错误。执行者完成一批等待派发不约束主讨论继续分析，
主讨论已把下一项推进到可执行合同；已有授权内不重复索要许可，也不为让实验不停而仓促试改法。

前批study根`/data0/user/ymdai/ember_runs/relational_support_causality_20260924`，正式树
`/data1/user/ymdai/projects/EMBER-relational-support-formal`。六臂各1260更新/141120query保持，当前阶段仅1500闭环、
66固定full cases，不启动3300函数预测；data0峰值≤128GiB/data1新增代码≤1.25GiB（保留E2并另冻E3），同时最多6物理GPU。
实现已集成推送`7dc95edbba00cf61439700d77fb321eb8df95c07`，正式树为该clean detached commit。
Sol完成CPU 91+9项、六臂4更新/2→4恢复、最长full-H50及新support物化→评测smoke，查询strg01独立quota与共享容量；
预计data0峰值新增120GiB低于128GiB，data1开发/正式树合计约503MiB。精确原合同为`launch/formal_launch_contract.json`。
主讨论已调用canonical task authority验证新manifest58任务/Train42/Val8/Test8；每个optimizer的白名单仍只有登记fit28。
Source71及官方24/8/8未改；无新增held expert或官方Val/Test读取。完整机制与竞争解释见task_plan第七阶段和findings§140。

### 关系支持stage1完成及主讨论核验（2026-09-25）

完成信号`analysis/stage1_completion.json`，18/18、1500/1500、66/66和所有worker/launcher exit0；
没有额外函数预测或自动补评。C_S11三个剩余面板已正常完成，此前接续遗漏已经结束，不再等待其训练退出。
主讨论独立读取所有results/连续NPZ、版本/噪声/teacher及原worker回执，重新计算全部29对比/44配对与bootstrap；
原件和独立脚本/数值在`coordination/main_stage1_recheck.{py,json}`，连续几何行在`main_stage1_geometry_rows.jsonl`。
已查看固定Goal state0/25的Source/B00/C00/C10/C11双相机图；这十例均成功，不把它们当失败分布的代表。
新增支持操作多数已较好，但C11不优于C00或同池语言B；同一数据规模下C00→C10的Goal损失与更偏的早期接近相伴。
本阶段关闭，不自动补其它held/seen/wrong/1050或3300预测；新冻结交叉只检验这项数据敏感性的学习传递路径。

### C_S11训练退出后的接续遗漏与恢复（2026-09-25，北京时间）

Owner询问为何停下后，主讨论做一次针对性核查：六臂训练均正常退出，C_S11最后于05:27退出0，1260完整checkpoint
及manifest所列文件齐全。15个已验收面板共1270行、56full，worker退出均为0；剩余仅C_S11 correct100、Goal other50、
support80，整批`analysis/stage1_completion.json`尚未产生。E3被动采集修复已经完成，不再是当前阻塞。

Sol上一轮的训练退出等待工具实际以`KeyboardInterrupt`/exit130结束，随后该任务于04:13结束，未接续05:27的训练完成事件；
中断由谁或什么触发尚未核实。旧lane退出Queue helper只处理此前一个lane，不能视为仍在等待C_S11的通知器。
截至本次10:50核查，Sol空闲，约五个半小时未接续；这是执行与主讨论的衔接遗漏，不是实验科学失败。

10:53以Queue恢复既定最后230条，回执`01a0d67b-545c-7982-815e-78810c54f30c`，活跃turn
`01a0d67b-545e-7d51-b02d-4f137f0a383b`；已核对完整正文与Sol明确回应。正文、接受与送达记录为
`coordination/stage1_final230_resume_{dispatch.txt,dispatch_receipt.json,delivery.json}`。
Sol已核对唯一启动记录、checkpoint、双节点GPU与quota后启动三面板；主讨论核对三份实际启动回执与bank正常退出，
correct用gpu01/1,4、other用gpu02/2,3、support用gpu02/0,1，共6物理卡。other从correct复用50条件，无新增Writer forward。
精简核验见`coordination/stage1_final230_launch_confirmation.json`。不重评已有1270行、不改变E2/E3例外或科学合同。

执行者接续现有进程到退出、完成1500行机械分析后主动Queue主讨论，停止额外评测；主讨论收到原件后作科学裁决。
退出等待不能以中断后的任务结束代替完成交接；确需结束任务时，须保证已有退出通知能继续存活并送达。
正常运行仍不轮询日志、分数或共享缓存。本次只恢复登记工作，不据工程完成宣称关系支持假说成立或根因已修复。

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
区域修复的真实无动作接口已通过：主讨论核对`smoke/stage1_region_registry_probe_fix.json`，其开发提交`96b11092`
把global77目标记为arena region，site3及父body1/table，普通对象/fixture registry保持；0 env.step、0 policy forward。
Sol回报相关104项CPU回归通过、工程初始化累计8/12。版本例外的来源准入/阶段入口及最终E3冻结仍待完成，不能称正式续跑已验收。
裁决在`07feec2d`集成推送（同时已含Sol的arena窄修`eca62e49`，完整例外准入尚待接续）。18:46 UTC核对Sol上一轮已完成，
未发送Steer，改用Queue接续既有任务，回执`01a0d4be-0b65-75e0-8680-b76d3d5e0da0`；18:47 UTC从新活跃turn
`01a0d4be-0b69-73d3-a1c7-8f91886c43b3`核对完整正文，尚未见后续回复，不称实现完成。
正文、回执、送达与独立验收记录见`coordination/arena_region_capture_exception_{dispatch.txt,dispatch_receipt.json,delivery.json,coordinator_evidence.json}`。
随后从同一新turn读取Sol明确回应：保留E2 correct100和两个bank、不重跑100行、其余17面板用E3，正在接续来源校验与阶段入口；
E2树/结果/合同只读。这已确认裁决被接收并处理，回报中的“仍待裁决”不是当前状态；未重复派发。
原文见`coordination/arena_region_capture_exception_acknowledgement.json`；尚无E3新闭环完成信号。

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
