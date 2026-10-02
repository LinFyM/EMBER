# EMBER progress

## 当前授权与最近完成批次

实验session已于2026-10-02T00:37:10.793592+00:00实际承接§15；独占dev与canonical集成窗口。新root已判重，正在接通固定D71事件分叉和C12/D71的630实际消费者；尚未启动新训练。data1现场quota 996.3G/2T、个人实占1069732286464B、共享82T余量，64GiB新增峰值可准入。全部新增data1，旧资产只读；工程阶段不通知主讨论。

Owner原有自主推进授权持续。主讨论已完成900原件、行为、native实际动作读出及数据/历史分析，
登记唯一active [条件读写设计§15](docs/designs/conditional_read_write_architecture.md#15-固定目标曝光的辅助任务分布短窗检验2026-10-02)。
当前问题是已见能力获取未转成跨任务调用与保持；数据支持不足仍为待检验解释，不是已经定位的唯一根因。
完整解释见机制§95、findings§270和`docs/analyses/conditional_read_write900_evidence_20261002.json`。

§15从真实450父点训练唯一D71分支180更新到630，保持target24的原事件/次数/权重，仅将support12分布替换为审计source71。
复用既存C12_630为同父同龄对照，不重训control；两个固定630各读correct400、原seen144、A28及8条被动native。
新增20,160个完整query中13,440个target query保持、6,720个support query改变分布；每source task只有3/4个teacher条件。
这是有界学习诊断，不是95task fresh或selected资格，不恢复原同池路线选点。
唯一新root为`/data1/user/ymdai/ember_runs/conditional_support_diversity_pilot_20261002`；新增硬限12GPUh/64GiB，预计2–4小时。
科学合同已登记，具体派发后由原实验session实施、冻结和运行；本次文档提交不声称GPU已经启动。
540只供新分支恢复，630读出完成即停；没有自动D71→900、完整fresh、其它checkpoint/controls/Test/RL或小扫。

已完成§14为900 validation140/400、seen115/144（target70/96、support45/48），held相对450仅+3，task31丢失16条旧成功。
810分支未触发；全部544环境原行及A28/native验收，新增12.464601GPUh。观察存储35.598759GiB、保守60/64GiB。
已完成§13为fresh450、validation137/400、seen91/144，19.273293GPUh；两批合计31.737894GPUh，全部计算已结束。
两个原合同、完成记录及训练/读取身份均保持；§14同池继续学习的科学预测未通过，不因§15读取C12_630而反选旧峰值。

唯一实验执行者`01a0f018-69af-7b00-b614-7e117540051b`负责代码、测试、排障、资源调度、Git及冻结运行。
主讨论`01a0ed66-cda4-7a23-90f0-e0d3a06a1d36`负责科学合同、原件解释、机制及后继/预算裁决。
既有自主授权持续生效，不恢复旧暂停段落，不等待主讨论工程复审。
沟通边界按[Owner要求§6](docs/current_owner_requirements.md#6-沟通与交接)：整批科学结果或确需裁决的实质边界只回报一次；
工程阶段、可自行修复的故障和普通调度记入已有记录。写入/Git窗口与实际冲突方直接串行协调。

## §15科学登记与交付范围（2026-10-02）

同父450、模型/full-only/Adam/绝对LR保持；target24沿451…630全部480条件，support240个slot改为71task。
固定permutation seed `[20260928,15,cycle]`，task次数3/4分别以80/71、60/71校正，使target/support总权重仍480/240。
其余59source任务对当前Writer是新映射，对source policy不是；完整19个target40等价排除项不恢复。
source71不存在任何basket或In-microwave目标，不能把扩分布写成直接补齐held16/39。
metadata预算表在900分析JSON：teacher总帧24,188对原23,636，最大同95帧，每步4个不同task；无新模型执行。

C12_630引用§14 root的`conditional_read_write/train/attempts/continuation/checkpoints/macro_00000630`，
旧实际训练为85919994；D71从§13的完整450（a0末段）分叉。新读取/训练身份由执行者分别登记，不改旧冻结树。
新分支540/630完整ECP；两个固定630共1,088环境行、24份A28及16条被动native，不能按部分结果取消对照或补读其它点。
原36-task seen面板不扩分母，保留所有原能力和反例。正收益只提高当前数据解释支持；阴性不自动延期或开完整fresh。
预计8–10GPUh有用计算、硬限12；额外产物估计约53GiB、峰值限64。实际GPU和strg01 quota准入由原执行者现场核验。
canonical科研文档在本次登记后随具体派发一并交给执行者的tracked/Git窗口；不得与main重叠写入。

## §14整批交付（2026-10-02）

唯一root为`/data1/user/ymdai/ember_runs/conditional_read_write_continuation900_20261002`。`completion.json`及`analysis/readback.json`验收complete，
整批readout owner exit0，全部有效训练/bank/evaluator/A28 exit0，原工程失败及计费保留。
仅从实际450完整ECP恢复451..900；新增450更新、1,800条件、50,400 full query，累计100,800，public训练FM为0。
36task每条原50teacher完成第二轮，累计100访次/50不同teacher；不是50条新视频。
Adam/scheduler/scaler/sampler/原五rank RNG和450 metrics前缀恢复，绝对LR不重启；540/630/720/810/900完整ECP齐备。
训练/读取实际身份均`85919994aef11c17b49b7d0e70a2c110158bff61`，来自clean pushed detached
`/data1/user/ymdai/projects/EMBER-conditional-read-write-continuation900-formal`；父450的a0及更早797/0e2/ebbb来源不改。

900完整validation为140/400，breadth7/8；task3/6/11/16/23/26/31/39依次32/7/45/1/1/41/13/0，
Spatial39、Object46、Goal42、Long13。对自身450137为R107/G33/L30、净+3、churn63、Jaccard0.629412，
整task簇95%差额[-30,36]；对T900148为R114/G26/L34、净−8；对Context900151为R117/G23/L34、净−11；
对MT153为R102/G38/L51、净−13、churn89、Jaccard0.534031、簇区间[-50,25]。
Task11净+11、26净+5，但3净−5、31净−11（原24只留8，得5失16）；16/23各仅1、39仍0，全部反例保留。
140≤153，按预登记不生成或读取810，不倒选其它checkpoint；没有相邻资格或selected声明。

900完整seen为115/144，breadth35/36；train24为70/96、support12为45/48。
对45091为R87/G28/L4、净+24、簇区间[12,37]；train净+25、support净−1。
对MT93为R87/G28/L6、净+22、簇区间[9,36]；train净+24、support净−2。
对Context900103为R95/G20/L8、净+12、簇区间[2,23]。四初态/task、已训练50池不能冒称held-video或正式400资格。
全部12份A28仍为固定query/noise/tau的FM速度预测；八条H/c/d、Q8/V8/action_out被动字段不增加forward或标签。
原行入口为`conditional_read_write/evaluation/900/correct400/results.json`、
`conditional_read_write_seen/evaluation/900/correct144/results.json`；A28/native为`analysis/A28/conditional_read_write/`。

训练GPU02:0/1/2/3/7、world5、µ28/frame32是现场选择，执行面仍支持1–6实际rank；
新增450步mean16.7869s/median15.7107s，训练含恢复/保存10.586330523091GPUh。
400 bank五卡与seen bank GPU01:2并行；400/144均五卡×每卡3 persistent replicas/dynamic long-first queue；
A28在GPU01:2与400独立并行。无dummy/hold/profile、无历史重评、无新增科学面板。
一次备用卡调度漏排除已派发但context尚未显存可见的卡，被launch准入拒绝；原bank无产物即退出−15，
0.007435357122GPUh全计费。已在独立读出owner窄修排除running receipts，通过原现场CPU复现，
未改冻结科学代码或重训，真实后段全exit0。细节保留`launch/scheduling_repair.json`和原/新owner记录。
全部真实新增12.464600929159463GPUh，旧19.273293210686425单列，两批31.737894139846GPUh。
阶段du高水位35.598759GiB、保守上界60/64GiB，未称连续精确峰值；全部新增data1，data0只读。
新增计算已停止，最终现场两节点无项目GPU context；完整科学交付只发主讨论一次，普通过程无广播。

## §13整批交付（2026-10-02）

唯一运行根为`/data1/user/ymdai/ember_runs/conditional_read_write_fresh_20261001`。
`completion.json`及`analysis/readback.json`已验收为complete，missing/invalid均为空；整批owner exit0。
450宏步、1,800条件、50,400完整FM query、36task各50不同teacher，以及90/180/270/360/450完整ECP全部核实。
公共训练FM为0，profile权重未进入正式训练。4×28有效query、事件流、绝对LR、Adam/scheduler及恢复边界保持。

完整validation400为137/400，breadth5/8；Spatial43、Object34、Goal36、Long24（各100行）。
对Context450126为R91/G46/L35、净+11；对T450122为R97/G40/L25、净+15；
对self_read450116为R87/G50/L29、净+21；对强MT153为R110/G27/L43、净−16。
对MT的whole-task-cluster95%差额为[-37,-1]，对Context为[-28,52]；不是seed不确定性。
Task31为24/50，但task16与23均0/50，保留全部有利及不利原行，不据单点或内部量宣布机制成立。

完整seen144为91/144，breadth29/36；train24为45/96，support12为46/48。
对同场景MT93为R78/G13/L15、净−2、churn28、Jaccard0.73585，whole-task-cluster95%差额[-13,9]。
Seen各task只有四个初态、来自已训练teacher池，不能冒称held-video或正式400资格。
全部544条环境原行、goal、连续trace、实际动作和RNG，以及固定A28的8full+4public速度预测、8条被动机制记录齐备。
原行入口为`conditional_read_write/evaluation/correct400/results.json`、
`conditional_read_write_seen/evaluation/correct144/results.json`；A28与机制入口为`analysis/A28/conditional_read_write/`。
分布、success sets、配对、簇区间和全部原件引用统一在`analysis/readback.json`，未重评参照或增开其它面板。

首段训练来源`797ae01f`（1..90），后续`0e2d6c79`（91..180）、`ebbb5e78`（181..270）、
`a0e0248d`（271..450及全部读取）；后续工程交付已承接整理后的`16241717`，旧冻结树与失败原件保持不变。
最终冻结树为`/data1/user/ymdai/projects/EMBER-conditional-read-write-mlp-formal`；
完整450在`conditional_read_write/train/attempts/resume360_native_packing/checkpoints/macro_00000450`。
真实来源、spec、topology/RNG与恢复链在`launch/code_identity.json`，不把今天文档提交改写成训练来源。

已交付按实际卡数自动target分片和完整cotangent返传，以及teacher作用域SDPA/冻结MLP checkpoint优化；
91..450由GPU02五卡真实训练，第五rank承担target与信用计算。最后段按实测峰值自动选择frame上限32、policy microbatch28。
实际361耗时12.93秒、当步最大帧批28，任务rank峰值约30–35GiB；这些是该事件的实测值，非最长视频或全程吞吐测量。
400 bank五卡耗时307.97秒，A28在GPU01独立并行，正式评测使用动态persistent queue。
已知NCCL、native OOM及profile replay工程失败均已独立修复并计费；失败attempt的181..185未保存更新从完整180重算。
针对实际源模块的8项attention及12项MLP CPU检查、实际通信/恢复/完整训练与读取消费者均有原件；未重复最长GPU profile。

整批成本19.273293GPUh，其中有效训练段10.876456GPUh、显式占卡5.085877GPUh，全部失败/加载/后段成本均包括。
最早登记占用到completion约4小时29分钟；不是不含工程的训练时长。
阶段边界观察新增最高37.058697GiB、保守峰值界64GiB，分别低于40GPUh/80GiB；不声称连续精确峰值。
strg01 data1独立quota2T现场准入及双节点launch检查在已有合同内，所有新增产物均data1，data0只读复用。
§13计算已停止，原合同不自动续900或其它checkpoint。主讨论消费完整原件后的独立后继已登记为上方§14；
原批完整科学分析、数据和成本保留，不因新合同覆盖其当时停止边界。

## 整仓整理交付

整理会话`01a0f7fa-63b3-7a42-a196-4c0fd145b10c`在独占`codex/ember-cleanup-20261001`完成源代码、
测试、脚本、配置、入口、文档及已核实临时内容整理；从`927b1498`建立，承接到`0e2d6c79`。
Owner纠正后的无token budget goal用于这项独立任务；没有新增session、子代理或GPU实验。
实验dev/frozen树未被修改，科学合同及必要原件保留。

已在本树退役旧Writer训练/生成与已关闭的专用诊断执行面，保留当前共享组件、sealed配置、实际bank/原行读取及科学原件；
统一README、稳定规则、当前状态和历史索引，修正Source配置导航变化导致的误拒绝。
源码提交`540fb773`；536项保留CPU测试及十个实际CLI help通过，承接96b07612后的实际消费者38项通过；
配置、import与文档引用已检查。57个退役文件及具体保留理由见research_history。
已实际修改个人workspace-cleanup skill及其inventory helper，symlink/边界/CLI验证通过；回滚在
`/data1/user/ymdai/skill-maintenance/ember-cleanup-20261001/workspace-cleanup/`。
已删除已消费的旧交接文件、canonical废弃pytest夹具及整理自产的大型临时检查内容。
交付以main包含本整理提交、与origin一致为准；task-owned临时树在交付后清除。
详细实改、验证、skill回滚及生命周期未明的保留范围见[research_history](docs/research_history.md#2026-10-02整仓整理与历史状态入口)。
实验后继窗口仍与实际冲突方串行协调；不将整理的工程检查转给主讨论。

## 已完成研究与原件可用性

最近的self_read450为116/400，Context450126、T450122、强MT153；完整正负证据、task/suite与机制边界见
[研究历史](docs/research_history.md#2026-10-01共享参数两次native的fresh450完整阴性与分析后讨论)、
[机制分析](docs/analyses/feature_to_operator_mechanism_20260926.md)§87及findings§261。
原400/144 scenes、MT/T/Context/self_read原始结果和固定A28输入
`operator_chain_diagnosis_20260929/functional_credit_transport/group0`均为当前依赖，继续保留。
已完成§40及更早路线不因历史“下一步”自动恢复；科学阴性、失败原因和有效正例完整留证。

2026-09-23历史存储裁剪账目为`runs/analysis/workspace_cleanup_20260923/storage_cleanup.json`及
`checkpoint_retirement/closeout.json`；当时checkpoint累计回收413.766GiB，94个weights_only、80个metadata_only。
这些是当时的裁剪事实，不是当前存储实测。Source1000 frozen policy可读取，原训练optimizer/EMA载荷已退休。
每项资产的当前可用性由其`checkpoint_retirement.json`、`payload_retirement.json`与实际文件共同说明；
weights_only不含完整训练恢复状态，metadata_only不含权重。不得仅按原manifest宣称可重放或exact resume。
本次源代码整理不删除source/dataset、formal原始rows/metrics、关键模型、运行根或冻结树。

## 历史状态的恢复入口

旧progress与task_plan累积了多轮互相覆盖的“当前/暂停/尚未实施”。完整当时过程保存在Git：
`git show 797ae01f:progress.md`、`git show 797ae01f:task_plan.md`。
按日期/方案追溯的入口为[research_history](docs/research_history.md)，跨轮结论在[findings](findings.md)。
这些历史快照记录当时授权、成本、失败与交接，不作为今天的执行authority；不新增平行状态或in-tree archive。
