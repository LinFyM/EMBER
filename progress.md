# EMBER progress

## 当前授权与最近完成批次

Owner于2026-10-02在完成交接与现状讨论后明确恢复自主推进，不对分析设置时间限制；本项覆盖当日上午分析后暂停的要求。
主讨论当前负责在既有数据内统合机制与历史证据、选择可失败的有界干预，再由唯一实验session完成工程与运行。
目标是在相对稳定的情况下大幅超过强MT，不把严格保持T的架构、稳定性或逐例成功作为新门槛。
当前active设计为[条件A函数重表达诊断](docs/designs/conditional_A_reexpression_diagnostic.md)；
数学推导/CPU原件回算已完成；实验session已接收工程与运行授权，独占分支实现及37项CPU验证已完成，正在集成并冻结，尚未启动GPU。§13–§15已结束，其续训或数据扩展不自动恢复。

同时维持Owner已明确的数据约束：现有数据规模固定，后继不增加训练任务、示范数量或引入额外数据来源扩量。
稳定约束已登记于`docs/current_owner_requirements.md`§4；不以增加任务、示范或额外数据源解决当前性能瓶颈。
分析不限人为时长不等于计算预算无限；每批仍须登记主要干预、完整参照、预计耗时、资源预算与停止线。

最新判断见函数重表达设计、findings§272与两份20261002专家审计/条件A证据JSON。
固定数据审计未发现匹配当前source/标签/训练量的全36专家上界；不能以“同数据专家已经都学成”为前提恢复蒸馏路线。
现有900的四train×两teacher原件中，S在Q8/V8/out的作用大多能用A0响应重表达，但只证实教学native分布上的局部事实。
本批以完整38处重表达检验自身query/十步动作/64有限闭环，硬限3GPUh/12GiB，预计含工程1.5–2.5小时。
无优化器、Val/Test/controls或选点；结果到齐即停止。新图文Value与fixed-A训练均未选定，不随诊断自动启动。

最近训练结果完整判断见机制§96、findings§271与`docs/analyses/conditional_support_diversity_evidence_20261002.json`：
扩支持只带来净+2、得27失25，未兑现广泛迁移/保持预测；原C12 137→154→140与seen91→99→115揭示获取和held保持分离。
native真实动作读回有小幅改善，仍未形成广泛收益；没有把对象/阶段行为定位冒称唯一神经根因，未选择新的架构或辅助目标。
主讨论未增加模型/环境前向或GPU分析，工程验证由实验session闭环。三批总40.215657GPUh，2,176条新增闭环原行。

§15整批已完成并停止新增计算：同父同龄固定630的D71为156/400、C12为154/400；配对R129/G27/L25、净+2、churn52、Jaccard0.712707，八task簇95%差额[-9,15]。原36task seen为D71 98/144、C12 99/144；target24为53/55，原support12为45/44。1,088环境原行、24份A28及16条被动native全部验收，missing/invalid为空，整批owner与全部消费者exit0。新增8.477763GPUh，阶段观察53.338696GiB、保守60/64GiB，旧31.737894GPUh单列。completion/readback与完整训练/读取身份见下方交付段；两个630仍为有界诊断，没有selected或后继训练资格声明。主讨论据整批原件作科学判断。

metadata/protocol先于新label读取封存；实际事件确认480个target条件/13,440query及flow完全保持，71项支持240条件按80/71或60/71直接进入原四条件损失。核心11项CPU、消费者37项CPU及集成后19项针对检查通过。唯一data/credit/trainer/scope与官方队列复用，新增support_diversity只拥有固定事件/权重；源码训练/读取796d7a9e来自clean pushed detached冻结，旧原件保持。新root/data1启动现场quota 996.3G/2T、个人实占1069732286464B、共享82T余量，全部新增data1，data0只读。普通工程事实只入launch记录，整批科学结果一次交付。

主讨论已完成900原件、行为、native实际动作读出及数据/历史分析；最近完成合同为
[条件读写设计§15](docs/designs/conditional_read_write_architecture.md#15-固定目标曝光的辅助任务分布短窗检验2026-10-02)；新批范围只按本文件顶部及新诊断设计。
当前问题是已见能力获取未转成足够的跨任务绝对能力与保持；数据支持不足未被证实为唯一根因，扩数据不再是可选后继。
完整解释见机制§95、findings§270和`docs/analyses/conditional_read_write900_evidence_20261002.json`。

§15从真实450父点训练唯一D71分支180更新到630，保持target24的原事件/次数/权重，仅将support12分布替换为审计source71。
复用既存C12_630为同父同龄对照，不重训control；两个固定630各读correct400、原seen144、A28及8条被动native。
新增20,160个完整query中13,440个target query保持、6,720个support query改变分布；每source task只有3/4个teacher条件。
这是有界学习诊断，不是95task fresh或selected资格，不恢复原同池路线选点。
唯一新root为`/data1/user/ymdai/ember_runs/conditional_support_diversity_pilot_20261002`；新增硬限12GPUh/64GiB，预计2–4小时。
科学合同及实际执行均已完成；新540/630完整ECP保留，540仅供恢复，两个630读出齐备后已停止。
没有自动D71→900、完整fresh、其它checkpoint/controls/Test/RL或小扫；后继由主讨论消费原件后裁决。

已完成§14为900 validation140/400、seen115/144（target70/96、support45/48），held相对450仅+3，task31丢失16条旧成功。
810分支未触发；全部544环境原行及A28/native验收，新增12.464601GPUh。观察存储35.598759GiB、保守60/64GiB。
已完成§13为fresh450、validation137/400、seen91/144，19.273293GPUh；两批合计31.737894GPUh，全部计算已结束。
两个原合同、完成记录及训练/读取身份均保持；§14同池继续学习的科学预测未通过，不因§15读取C12_630而反选旧峰值。

Owner于2026-10-02指定新的唯一实验执行者`01a0fabb-f7a0-7100-93d8-6a0f66055553`
（hostId：`remote-ssh-discovered:BCI-GPU02`），接替`01a0f018-69af-7b00-b614-7e117540051b`，负责代码、测试、排障、资源调度、Git及冻结运行。
继任主讨论`01a0faba-4bb9-7ca1-b3b4-6d05bec44e33`接替`01a0ed66-cda4-7a23-90f0-e0d3a06a1d36`，
负责科学合同、原件解释、机制及后继/预算裁决。继任后Owner已恢复自主推进；只读专家审计已完成，当前阶段如本文件顶部。
不从历史暂停或设计中的“下一步”推断现行状态；工程检查仍由实验session闭环。
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

## §15整批交付（2026-10-02）

唯一root为`/data1/user/ymdai/ember_runs/conditional_support_diversity_pilot_20261002`。
`completion.json`与`analysis/readback.json`均为complete，全部44项读取/配对/来源/费用检查validated，missing/invalid为空；
`launch/batch_owner_exit.json`确认exit0。两个固定630各400＋144、8full＋4public A28及8条被动native，
共1,088条环境原行、24份速度预测、16条合法teacher机制字段，所有goal/continuous/action/RNG原件齐备。
每个400为8full/392compact、每个144为36full/108compact；参照无新GPU消费者。

Held task3/6/11/16/23/26/31/39的C12→D71为30→28、7→5、44→44、2→4、0→0、43→48、28→27、0→0。
对应R/G/L为24/4/6、4/1/3、39/5/5、1/3/1、0/0/0、41/7/2、20/7/8、0/0/0。
两臂breadth均6/8；Spatial37→33、Object46→48、Goal43→48、Long28→27。
主比较D71−C12为净+2，R129/G27/L25、churn52、Jaccard0.712707，whole-task簇95%差额[-9,15]，不是训练seed不确定性。
D71对共同450137为R110/G46/L27、净+19；对既有900140为R114/G42/L26、净+16；
对MT153为R109/G47/L44、净+3，对T900148为R113/G43/L35、净+8，对Context900151为R115/G41/L36、净+5。
C12对共同450为R110/G44/L27、净+17，对MT为R112/G42/L41、净+1；全部成功集合和其它配对保留readback。
630事先固定，两个诊断点均未selected；没有根据这次分数恢复§14选点或读810，成熟T2340仍是不同年龄背景。

原seen144为C12 99、D71 98，R86/G12/L13、净−1、churn25、Jaccard0.774775，36task簇95%差额[-10,8]。
Target24为55→53（R42/G11/L13），原support12为44→45（R44/G1/L0）；breadth34→31/36。
D71对45091为R80/G18/L11、净+7，对900115为R91/G7/L24、净−17；
对MT93为R84/G14/L9、净+5，对Context900103为R88/G10/L15、净−5。
四初态/task、原50teacher池及有限面板边界保持，seen分母没有扩为95task。

A28仍为原query/noise/tau的FM速度预测，无额外十步采样。两臂8full/4public及全部逐query、first5/full50、motion6/gripper记录已验收；
八full的D71−C12平均risk差first5 −0.00184794、full50 +0.00032395，四public为−0.00037572/+0.00029511。
这些是固定训练面板的功能读回，不构成闭环或视频因果结论。各臂8份真实H/c/d与Q8/V8/action_out的X/A0/S/B0/M保留，未增加native forward。
原件入口为`C12|D71/conditional_read_write/evaluation/630/correct400/results.json`、
`C12|D71/conditional_read_write_seen/evaluation/630/correct144/results.json`和`C12|D71/analysis/A28/conditional_read_write/`。
完整逐task/suite、原行得失、goal阶段和连续动作引用在`analysis/readback.json`，全部有利和不利样本保持。

仅新训D71的451..630共180更新、720条件、20,160完整query，full-only，public训练FM为0。
Actual target条件/visit/teacher/query episode/frame/flow与C12原事件完全相同；支持240slot按固定71task permutation和3/4次权重执行。
窗口target/support权重480/240，每source累计240/71，实际损失为sum(weight×mean28fullFM)/4，不按宏步权重和或物理rank归一化。
新增59项是Writer的新映射；每source只读3/4不同teacher，不声明数据普遍可行或不可能。
完整450的Writer/Adam/scheduler/scaler与五rank RNG恢复，绝对LR保持；sampler为登记科学分叉，不称原轨迹exact resume。
首451恢复证据在`launch/actual_resume451.json`，完整新540/630和实际事件审计在readback.training.D71。
旧父450为a0e0248d，C12_630真实训练85919994、新读取796d7a9e；D71训练/读取均796d7a9e。
冻结树`/data1/user/ymdai/projects/EMBER-support-diversity-formal`及两个arm的spec身份独立登记，旧冻结树/原件未改。

D71训练实际GPU02:7/1/2/3/0、world5、µ28/frame32，执行面支持1–6rank；180步mean17.4291s、median16.9755s。
C12 bank/A28/seen在GPU01:2与D71训练并行，D71 bank400及两个400/末点seen144在GPU02五卡动态队列运行，
官方评测每卡3 persistent replicas；D71 bank144/A28在GPU01:2与C12 400并行。全部有效进程exit0。
训练加载/保存4.621526GPUh，C12 bank400/144 .372669/.132272、A28 .025758、seen144 .307342、400 1.016590；
D71 bank400/144 .344510/.115134、A28 .020905、400 1.085736、seen144 .435324；总8.477763278407GPUh。
启动到整批owner退出约1小时33分；此数不包含启动前工程实施时间。原两批31.737894GPUh单列，无预算结转或扩大。
无GPU失败/hold/profile；一次首451 CPU包装读错键为record，已按真实row字段修正，0GPUh，原失败记录保留。
阶段du高水位53.338696GiB、保守60/64GiB，非连续精确峰值；所有新增data1，原data0只读。
新增计算已停止；整批科学交付一次发送主讨论，常规阶段无跨session广播。

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
