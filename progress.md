# EMBER progress

## 当前授权与活动批次

Owner于2026-10-02在完成交接与现状讨论后明确恢复自主推进，不对分析设置时间限制；本项覆盖当日上午分析后暂停的要求。
主讨论当前负责在既有数据内统合机制与历史证据、选择可失败的有界干预，再由唯一实验session完成工程与运行。
目标是在相对稳定的情况下大幅超过强MT，不把严格保持T的架构、稳定性或逐例成功作为新门槛。
main现已直接消费下述视频来源批的16份continuous、四份RGB及真实构造代码，完成机制§101/findings§280。
后段单独造成“得到放壶、失去开炉”的预测未获支持；早段增量也能补出完整操作，但仍依赖完整父LoRA与后段地址。
关闭依此事件分区选择阶段门控、部署E或继续扫切点/层位/scale；历史完整S与新增量约.82%–1.23%重构差限制精确归因。
本批及其原件消费已完成，canonical tracked/Git窗口回main；当前无active训练、GPU分析或排队后继。
主讨论继续统合共享视频到完整控制函数的学习条件与最近似完整历史；尚未选定新修正，不恢复旧训练或以局部高分自动推进。

主讨论完成上一批原件与机制§100后，登记唯一后继冻结分析：
[task32已学修正的视频来源](docs/designs/task32_learned_video_segment_diagnostic.md)。
同一共享S64在teacher17/43上，仅按真实教学炉面首次变红的固定采样帧80/95，分解早/后转移的完整38处学习增量。
两teacher×E/L×原四init=16新行，父/S原16参照复用；四full/十二compact，只有两条旧父native重读，无学习/新标签/held。
它辨识S43放壶收益与开炉损害的实际视频来源；若不能按此事件定位，不继续切点/层位扫描或自动开训练。
预计35–75分钟含工程，硬限1完整GPUh/4GiB，预计新增峰值3GiB；唯一root为
`/data1/user/ymdai/ember_runs/task32_learned_video_segment_20261002/`。
本批已完成16/16新增行、4 full/12 compact与全行continuous/goal/actions；两条public native重读及四bank/H/全38 K齐备。
实验session 01a0fabb-f7a0-7100-93d8-6a0f66055553从clean pushed a2c39b5a接手窗口、独占分支实施；四项CPU检查通过。
构造/实际消费者clean pushed detached 8b90f31cd3af60045b14ea73e6320529edd73d5b，原父训练e2afbfd7、学习092a0ae8、旧读取5313257c分别保留。
2026-10-02T19:47:09.740844+08:00–2026-10-02T19:52:05.019625+08:00在gpu02:7运行PID3861322，消费者/CPU读回均exit0；GPU进程与占用已释放，无GPU失败。
一次source加载、native全部51/49帧分别18.920/20.463 frame/s、闭环最大batch16；1296个完整50×7生成chunk、forward9.190 chunk/s。
全部授权帧/case已打包，未为填显存增计算；未测试第二卡，不宣称其速度劣势。峰allocated13.983/reserved15.486GiB。
含加载/退出累计0.082021883726GPUh；root阶段观察1.280GiB、保守新增峰3/4GiB。现场strg01 data1个人实占约1.134TB、quota2TiB、共享89.099TB；本用户0→1卡，上限6，148MiB/util0共驻未动他人。
17 E/L成功[0,1,2]/[0,1]（3/4、2/4），43 E/L为[0,1,2]/[1]（3/4、1/4）。对父R/G/L依次1/2/0、1/1/0、2/1/0、1/0/1；对完整S为2/1/0、1/1/1、2/1/1、1/0/2。
全部16行最终炉目标真，7个不利行均为placement未满足；保留17 E3、L2/3、43 E3、L0/2/3。
原S43/init2仅放置无开炉，新E在133/296开炉/放置并成功，新L145开炉却未放置；43/init3完整S成功，E/L均未保留。
四份实际双RGB读回；E43/init3在474才描述性>3cm、最大4.628cm，520仍未放置；3cm/中心/闭合命令不证明抓持，旧S43无RGB仍缺。
封存初态body/EEF/夹爪/谓词误差全部0、scene/root7与绝对policy噪声时钟全行匹配；official256→224、十步、前5、成功停通过。
同一重读特征上的E+L closure relativeL2为0.000224/0.000203；对旧parent/S整组重构为0.78–1.07%，与旧S-parent增量合计差1.229%/0.824%。
正常BF16/TF32/batch/reduction差保留，不称逐bit原bank复现，也不重跑完整S或作dtype/batch小扫。
任务专用入口/测试与canonical episode-context hook已退役，原运行树/Git/原件/0GPUh CPU fixture失败保留；current runtime拒绝已归档本批合同。
此Git交付完成后仅一次整批回报主讨论并交回canonical tracked/Git窗口；实验session停止本项，无自动后继、训练、FM、held/controls/Test/RL。
原件索引：上述root的completion.json、construction_readback.json、analysis/readback.json、analysis/rows.jsonl、analysis/pairing.json、launch/gpu_ledger.json。


当前处于完整学习机制的统合分析，没有active训练；以下为更早已完成的冻结分析：
[task32已学修正的执行投影分组](docs/designs/task32_learned_operator_groups_diagnostic.md)。
仅旧D17/S43的Q/非Q学习增量、两teacher×两臂×四init共16新行，复用原parent/完整学习行。
预算1完整GPUh/4GiB；16/16新增行、4 full/12 compact及全行continuous/goal/actions已齐，GPU消费者与CPU读回均exit0。
实验session（01a0fabb-f7a0-7100-93d8-6a0f66055553）当时从clean pushed c17d4c80接手并创建独占工程worktree；
整批已在8a6f8d6b交付并释放canonical tracked/Git窗口，现由main维护科研记录。
9项针对性CPU检查通过；实现/构造/读取clean pushed detached 22faa1966f43b14555088af61dc445f67f95ca5b。
四份完整38-target银行已构造；2026-10-02T18:55:21.952181+08:00实际在gpu02:7启动16合法case最大packing、一次source加载，
消费者PID3624517，含加载/退出预算1GPUh；四full/十二compact与全行continuous/goal/actions由同一canonical consumer捕获。
现场本用户既有GPU为0，新增1，总卡数准入上限6；gpu7低占用148MiB/util0可共驻，未改变他人进程。
data1现场XFS个人实占1.0T、quota2T/limit2.0T，共享82T；新增峰值预计2.5GiB、硬限4GiB。
整批实际完成：旧T2340、teacher17/D64的Q/R各2/4，成功init为[1,2]/[0,1]；teacher43/S64的Q/R为3/4、2/4，成功init为[0,1,2]/[0,1]。
对父分别R/G/L=1/1/0、1/1/0、2/1/0、2/0/0；对完整学习臂为2/0/2、2/0/2、2/1/1、2/0/1。
两组均未保留D17/init3及S43/init3的新增完整成功；D17/init0只R保留、init2只Q保留，不能净分选全局赢家。
S43/init2原完整S“放置但没开炉”，新Q开炉141/放置281并成功，新R开炉121但没有放置；原件正反均保留。
全部16条最终开炉为真，7条完整失败均未放置；Q17/init3、R17/init2/3、R43/init3有描述性>3cm抬高却未放置。
四条真实双RGB已读回，3cm/中心/闭合命令仍不证明抓持或接触；不据组名宣布感知/运动根因。
封存scene/body/EEF/夹爪/谓词初始误差均0、root7逻辑噪声全行匹配；实际消费者50×7/十步/前5/成功停合同通过。
GPU消费者18:55:21–19:00:05(gpu02:7) exit0，含加载与退出累计0.078771065871GPUh，无GPU加载/运行失败。
一次source加载、最大batch16、1,312个完整50×7生成chunk；实际forward约9.2093 chunk/s，rollout249.751s；
峰allocated11.4464755/reserved12.7695313GiB，一次NVML实际util100%。全部授权case已打包，无额外case可放大；未测试第二卡，不作其速度优劣声明。
root阶段观察约1.193GiB，保守新增峰值2.5/4GiB；CPU fixture/准入/读取脚本的失败均0GPUh并留回执。
双节点退出快照本用户GPU进程0，消费者PID3624517已消失，未修改他人进程；没有新训练、Writer/native编译、离线FM或held/controls读取。
9项实现CPU检查和实际16行验证完成；专用入口/测试及重复init hook已从active tree退役，canonical rollout接口恢复，原件与22faa196 frozen代码保留。
退役后旧owner两项检查及active运行拒绝已归档新合同的检查通过；只增加退役guard，不建立长期并行consumer。
completion/readback/全16行/四full/RGL/费用/释放/失败与实际冻结身份统一在
`/data1/user/ymdai/ember_runs/task32_learned_operator_groups_20261002/`。
本批工程、运行与读回结束；一次整批消息发给主讨论后释放canonical tracked/Git窗口并停止，不自动追加任何计算。
main已直接读取全部16份continuous及四份双RGB，完整解释见机制§100/findings§279。
D17/init0只R保留新增成功、init2只Q保留；D17及S43的init3均只有完整更新成功，S43/init2却只Q成功。
有限成功交互既有+1也有−1，不能把它等同神经Hessian或能力比例；关闭按Q/R分组选修复和继续拆投影的路线。
下一判断仍须把实际视频特征的学习改变与自身状态调用联系起来，不能从本次互补直接选择新架构或正式训练。
Owner最新明确：MT只是参照，EMBER应大幅超过MT；不能用MT同样失败降低EMBER自身失败案例的研究优先级。
主讨论此前据错误筛选标准提出的两条MT补测，在派发前撤销，0新增GPU/环境、无新run root或实现。
保留已有MT比较作为能力证据，继续从EMBER自身的绝对不足及已知可学改进解释教学—算子—自身控制，不要求参照先成功。
最近[task32自身状态与冻结LoRA交叉续行](docs/designs/task32_state_policy_crossover_diagnostic.md)已完成并由主讨论直接消费原件，
完整科学解释见机制§98/findings§277：两套策略都有从s17完成后段的控制，却均未从s43在原时限内完成；
不能把具体失败简化成缺少第二阶段Value，也不能从一个选例推出普遍状态/接触根因。
main另对已有四train任务、同query两teacher预测作CPU风险分解：十步前5的teacher差异项只占总风险.0961%–.7586%，
大部是平均预测残余；与具体闭环中的前缀敏感共同解释，不能归罪公共B0或重开一致性loss。
实际native→S/M→自身调用→真实FM信用、任务内获取/跨任务迁移/保持已合入同一推导，未选择新的架构或正式重训。

同query强MT/旧T的CPU审计已完成并停止，0GPUh、约9.6分钟含交付、1.75MiB，无模型/环境/新预测或tracked/Git写入。
四task A28两消费者完整配对，Current/T/MT前5FM=.128471/.128661/.131079；十步all7=.124717/.124228/.119610，
当前motion6更低、gripper更高；任务/逐query不利项及masked两种权重见机制§98.6与findings§277，不把离线通道差直接当闭环根因。
审计原件已归档至四格root `analysis/conditional900_reference_audit_20261002/`，原tmp路径仅保留别名，单一实体。
上述CPU参照审计结束时，canonical科研记录/Git窗口由main独占，实验session没有新GPU任务；
随后新16行合同另行移交，当前状态见本节顶部。
**原四行及CPU参照审计均已结束；两条MT补测未派发，不恢复任何历史训练。**

旧T2340父/S共享/P任务私有/D条件私有的task32全部32条既存轨迹CPU读回已完成并停止，0GPUh、约1.11MiB。
main已直接读16条关键continuous、双RGB及实际学习源码，完整解释见机制§99/findings§278。
D17从1/4到4/4，新增三行保留开炉并完成搬壶；D43仍2/4，S/P在其init3反而成功。
S17两行学会抬壶却未放置，S43/init2得到放置而丢失开炉；3cm仅为描述阈值，父有低于该阈值的完整成功。
两组D面对同一query/flow/功能目标，差别由视频生成的初始B及其后学习轨迹造成；不将D修正当唯一视频知识标签。
继承已完成PZ投影及其跨条件外推失败，不重复投影/共享解除/输出扩张，也不将旧T和当前900混成同一模型。
原件在旧run `analysis/task32_absolute_learning_readback_20261002/`；无新模型/环境/动作标签/held读取。
main另读已有S/P/D同B20预测作学习修正分解：task32修正差异能量仅.346%/.479%/.789%，两D修正余弦.9844，
几乎全部平均改善来自两teacher共有部分；task20与task12的不利/高差异项保留，不能外推为所有任务一致。
结果及执行Q/R的精确含义见机制§99.5。后继冻结分析不由共有分量直接宣称公共B0不足，也不重开一致性训练。
该合同已完成并由main消费，canonical窗口已经交回；没有新的架构/fresh训练资格。

以下保留最近四格批次的完整执行事实：
按(i,j)=(17,17)/(17,43)/(43,17)/(43,43)，success为真/真/假/假，结束步278/275/520/520；
首次抬壶3cm为210/210/509/无，放置谓词为278/275/无/无，最大抬高14.771/15.362/7.299/0.668cm。
两条对角重现原一成一败；原17为277步、本次278步，不追微小数值一致。同i两行anchor状态与新双RGB误差均0，
相对原continuous/compact的EEF、夹爪、物体、谓词、raw state8误差均0；原state2没有RGB，未编造历史RGB。
完整4 full/continuous/goal/physical action齐备：144个重放prefix chunk明确无新policy forward，
175个合法续行50×7 chunk，first consumer为batch4、index36、共同seed、官方十步；有限差D=D_W+D_s误差0。
W17在s43到509才抬壶且未放置的不利例保留；最终成功随前缀的事实不能直接宣称具体几何根因或新架构资格。

沿用canonical rollout、原FrozenOperatorAdapter和初始scene/full捕获；31项针对CPU及10项封存来源/数值/case检查通过。
实际有效读取为clean pushed detached `e84712d8b6f3641968d820a45500670a58196c66`，原900训练85919994与原读取923ff89b单列。
16:52首次加载因LIBERO新root缺配置退出1、0环境行，费用0.004473948GPUh；封存spec搬迁的CPU来源错误也保留。
复用原runtime配置owner修正后16:56:08–16:57:38在gpu02:7执行四行batch4、一次source加载，
有效consumer/CPU readback均exit0，0.024935183GPUh；累计**0.029409131GPUh**含失败/加载/退出，硬限1GPUh。
峰allocated9.424913/reserved9.968750GiB；四行已经是本科学范围最大packing，没有额外工作可加入。
data1 quota现场2T/limit2.0T、报告使用1.0T、个人实占1131008344064B、共享82T；
root阶段观察约1.489GiB、保守峰值2.5/4GiB。双节点核实本批owned GPU为0、两个consumer PID已消失，未动他人进程。
原source/spec/frozen与失败保留；任务专用入口/新case hooks在交付后从active tree退役，Git及实际冻结源码保留，
封存spec身份的通用读取修正与回归检查保留；退役后35项针对检查通过，canonical rollout/capture恢复为原单一路径。
completion/readback/report、原F2×2和分解、费用/退出/释放回执均在
`/data1/user/ymdai/ember_runs/task32_state_policy_crossover_20261002/`。整批报告一次交给主讨论并释放canonical写入/Git窗口。
Owner要求的决策错误已写入current_owner_requirements§3、findings§276和research_history，394b5f79已推送。

## 最近撤回批次及保留事实

[原生prefix变化Value的共同学习](docs/designs/native_prefix_change_value_design.md)曾获派发，
但Owner指出没有建立“具体task失败→有证据的方法缺陷→干预改变失败预测”的决定性链条，block17选择也仅有启发；
主讨论承担方法选择错误，立即撤回该批剩余训练、物化与评测。允许有界检验不等于可以跳过机制辨识、
让正式训练替方法选择寻找理由。**本批因方法选择依据不足撤回，非工程失败或性能阴性。**
Owner恢复自主推进的总体授权仍在；本实验session只完成撤回收束，不自行恢复本批或其它历史计算。

实际工程在独占worktree完成原生attention字段、零E、full-only合同及原消费者接入，29项CPU检查通过；
clean pushed detached训练/读取身份为`ab8d2c7ee22447da133cde59b975133b5988520f`，source aligned1000保持冻结。
最长视频517原帧/105采样帧两步profile完成，micro28/frame8、38.93/33.34秒、峰reserved22.805GiB；
导入失败exit1与有效profile exit0合计0.044224GPUh，profile权重只作一次性工程产物，未进入正式初始化。
正式fresh在gpu02的0/1/2/3运行到已登记update71（284条件、7,952个query），主讨论于2026-10-02 15:46:56
向已核实的本批PGID2709066发SIGTERM；实际torchrun exit1、四rank已退出。未到首个90 ECP，完整checkpoint为0，
无本批可恢复checkpoint；已登记1–71更新均未保存，第72步可能在途，无额外完整更新的证据。
270→450、全部bank/400/seen及后继未启动且已取消，三个task-owned启动器已加撤回退出保护，冻结源码未改。
新闭环0/944、full0/52；没有本批held native读取/物化/评测，也没有读取Val/Test动作或产生held梯度。
曾只读核对既存Val400结果元数据/成功原行及seen参照，不将其冒称完全没有接触held结果。
累计1.452021841GPUh（含加载/profile/启动失败/训练退出），root阶段观察峰值约0.69GiB；
双节点现场核实本批GPU占用为0，所有本批producer/rank PID消失；没有终止他人进程。
原source/spec/frozen、profile、日志、失败及停止回执均保留；completion/readback和精炼closure见
`/data1/user/ymdai/ember_runs/native_prefix_change_value_20261002/`，原停止回执见本树
`.codex/tmp/native_prefix_change_main_stop.json`。原944行合同未完成，不能将收束完成写成科学实验完成。

Owner同时纠正GPU利用：实测22.805GiB仍有余量，实验session沿用frame8却未验证更大物理分块，
未落实已有吞吐要求。Owner要求已加强为AGENTS§9长期规则：必须主动验证显存余量的吞吐用途，
以实测选择物理配置并记录未放大的依据，保持逻辑batch/权重/更新及科学范围；本批撤回后没有追加profile。
实验session的运行收束已在863eef2b推送并释放canonical窗口；主讨论随后在394b5f79完成稳定要求的科学纠正。
当前写入分工见顶部，已结束的CPU临时目录权限不延续为并发canonical写入权。

最近完成[条件A函数重表达诊断](docs/designs/conditional_A_reexpression_diagnostic.md)：Original13/32、Reexpressed14/32，
R13/G1/L0；448 FM＋448十步query、64配对闭环和16 full齐备，费用0.281819GPUh、阶段观察3.113960GiB。
主讨论直接消费后完成机制§97/findings§274：不部署解析C、不自动去S训练或追加投影探针，主要能力缺口仍在。
原生Value批次已撤回，不由该冻结保留结果自动推出后继；条件读写§13–§15仍不恢复。

同时维持Owner已明确的数据约束：现有数据规模固定，后继不增加训练任务、示范数量或引入额外数据来源扩量。
稳定约束已登记于`docs/current_owner_requirements.md`§4；不以增加任务、示范或额外数据源解决当前性能瓶颈。
分析不限人为时长不等于计算预算无限；每批仍须登记主要干预、完整参照、预计耗时、资源预算与停止线。

条件A诊断的交付事实见findings§273；前置判断见函数重表达设计、findings§272与两份20261002专家审计/条件A证据JSON。
固定数据审计未发现匹配当前source/标签/训练量的全36专家上界；不能以“同数据专家已经都学成”为前提恢复蒸馏路线。
现有900的四train×两teacher原件中，S在Q8/V8/out的作用大多能用A0响应重表达，但只证实教学native分布上的局部事实。
该诊断已完成完整38处重表达、自身query/十步动作/64有限闭环，原硬限3GPUh/12GiB、预计含工程1.5–2.5小时。
无优化器、Val/Test/controls或选点；结果已到齐并停止。通用图文支路/条件图去S仍未选择；没有自动恢复的Value图。

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

## 条件A重表达诊断：整批交付（2026-10-02）

唯一root为`/data1/user/ymdai/ember_runs/conditional_A_reexpression_diagnostic_20261002`，
`completion.json`／`readback.json` complete，精炼结果在`report.md`与`delivery_summary.json`，实际原件读取在`verification.json`。
8条件×38处默认float64 gelsd，原S/M保留，16份完整rank128两格LoRA；448 FM与448官方十步query记录，
独立task-query只有112。64闭环为四train×两teacher×states0–3×两格，各格32、16不同物理初态；
全部continuous／goal／实际action／compact齐备，state0全16行full双相机，未作新训练、held/controls或选点。

Original13、Reexpressed14，R13/G1/L0、churn1、Jaccard0.928571；唯一得例12/14/0。
task0/12/20/32为8→8、2→3、2→2、1→1；每teacher成功集合及RGL完整在readback／report。
固定task内四state成组bootstrap保留两teacher重复结构，净差95%区间[0,3]，不是大样本泛化区间。
FM全50/前5 MSE .110917/.128471→.111018/.128774；十步全50/前5 .141694/.124717→.141308/.123937。
有效future、有效前5、motion6/gripper分别保留；两格前5直接输出差/原输出范数等条件均值FM0.899%、十步2.249%。

教学E/完整LoRA中位数Q8/V8/out为0.0364%/0.0682%/0.1734%；Original十步自身hidden为5.547%/9.870%/7.415%。
全38处最不利条件task0/teacher40/V7为FM69.032%、十步73.381%，不把焦点层位的小值概括成全部层近等价。
194/224个条件query至少一项所列风险恶化（包含微小数值变化），Original19／新格18共37失败行均保留。
教学拟合不能直接搬到自身hidden；本有限面板控制大体保留不证明S无用、fixed-A可重新学成或held能力修复。

实现／实际读取`923ff89b60f4f8a269e5352d8d82189d72a9cd0a`已push main，消费者来自root下clean detached `frozen`。
原900训练`85919994aef11c17b49b7d0e70a2c110158bff61`、aligned source1000和冻结normalization保持，原件只读。
37项针对CPU验证通过；完成后读回全64小continuous原件的实际action/EEF/物体/夹爪/goal形状及finite，
单条full的双256相机、normalized完整50×7与physical前5核实；policy仍官方256→224/10step/前5/settling10/成功终止。

现场strg01 data1 quota2T/limit2.0T、个人实占1,127,743,614,976B，共享82T；全部新增data1。
单producer后两卡×3persistent workers，总费用0.281818972GPUh，含加载／CPU解析期间占用／故障／恢复；
root阶段观察峰值3.113960GiB，保守估计7GiB，低于3GPUh/12GiB。不声称连续精确峰值。
首GPU14:02:43、末GPU约14:12:37，全部GPU释放，只保留原gqma低占用上下文，未中断其它用户。
首临时等待包装器误用系统Python而exit1；其producer因重父化OS退出码不可观测、明确null。
producer completion与全部原件已验证，未重启其有效工作；四闭环、CPU readback和恢复owner均exit0。
完整退出／费用／释放回执见root/launch；不伪报全部exit0。工程／计算已闭环，整批一次回报后释放canonical窗口并停止本项。

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
