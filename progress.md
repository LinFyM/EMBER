# EMBER progress

## 当前快照（2026-09-11，自主执行已恢复）

Owner最新明确授权Codex在其休息期间持续自主推进方法修正、实验验证、分析和再次修正，并要求创建goal；active goal已建立。此前咨询暂停与完整训练限制已被覆盖，后续不重复请求逐项批准。实现与profile通过，R/C首轮正式训练完成；R/C首轮validation与train96均完整完成：R37→83、C43→50（各400），train96分别46/49；S59→45（各400），train96=52；首轮三臂所有训练与九个闭环面板均完整结束。目标未达标。

唯一active design：[普通FM下语义条件化过程消费](docs/video_consumption_writer_design.md)。首轮R/C/S比较保留普通FM、完整H和独立D，分别检验条件分配、消费接口及主动训练无序帧集合参照；R/C保留过去定向，S逐帧独立。具体随机性、曝光、节点和裁决见设计。C/S不追加，R虽增长但低于旧同预算off200；本轮两项改动均未带来未见任务净收益。off7探索性续训及评测全部结束：400/500/600 correct126/73/54，train400→600为56→64/96，停止该未改配方的续训。R原200完整恢复后的300 strict400已完成：83→63，保留43/新增20/丢失40，四suite0/24/34/5、breadth4；400训练继续，按§8完成有限获取节点及train96/独立动作验证，不自动追加后续训练。

[专家第二轮原文](docs/review_materials/20260911/expert_review_round2.md)已归档。独立判断：末端改动依赖P4已有可用信息；新增语义S可能独立贡献；无序多帧仍有状态变化信息。C/S必须共享语义、融合和D，不能把不同模型分数差唯一归因于时序。目标仍未达标；Test封存。

当前执行：R/C/S架构与双K1条件训练已合入main（513d1ec4），共享functional完整64随机批后切32的语义检查通过。整合回归377通过，唯一失败为测试fixture未解析worktree数据symlink；修正fixture后相关49项通过，全部378项已覆盖通过。三份正式配置通过合同解析。物化metadata已准确区分S的无序帧集合（7fedbe85），相关60项通过；本轮首个正式闭环结果见下文。

本轮只增加semantic与frame_evidence两个凝聚模块，复用现有训练/物化/eval入口。架构检查无hard violation；既有长函数和测试文件的局部增长已审查，未复制runner。实验模式由同一入口显式限制，候选选择后退役非选中模式。

R/C/S真实GPU完整梯度与最长457帧/93输入帧检查通过；C完整checkpoint2→4 exact resume通过，source无trainable或gradient。双卡每更新R约29–34秒、C34–39秒、S27–31秒；最长C约14秒。R的共驻micro6约36–47秒，reserved峰值34.7GiB；空闲卡micro8全段最高41.8GiB。仅说明工程可运行，不是科研达标。

正式runtime为clean pushed detached `7fedbe85`（`.codex/worktrees/video-consumption`）。C训练使用gpu01 1/4、micro8/8；R使用gpu02 4/6、micro6/6，均已exit0。两组各fresh200/51,200queries/1,600K1 conditions，训练加独立动作验证约115分钟。完整命令、GPU UUID、profile、storage与后续结果在`runs/analysis/video_consumption_20260911/`，输出为`runs/outputs/video_consumption_{r,c,s}_seed7_20260911/`。

R/C第100、200次正式checkpoint均已保存并通过现有formal inspector；记录见`checkpoint100_inspection.json`与`rc200_learning_summary.json`。两组200均覆盖378/384个训练task/video组合；相同25,600queries时新方案覆盖330个，旧单条件259个，具体见曝光summary。独立动作FM：R0.151459→0.110515，C0.151461→0.109569；两者接近，不据此选模型或宣称闭环进步。

第100次节点没有适合并发评测的设备，原strict400保留排队，未缩成80条screen。训练完成后已刷新两节点：S在释放的gpu01 1/4、micro8/8上按同合同fresh训练；R/C各由gpu02 6/4单卡批量物化checkpoint100 validation400、checkpoint200 validation400及train96（每模型896个独立完整LoRA）。没有混合checkpoint或平均LoRA。全部六个bank已sealed，物化均exit0，R/C各耗时约22/23分钟，见rc_materialization_summary.json。六组validation400和三组train96命令均已准备。

C100/C200两组strict400均完整exit0，每组6个persistent worker、约29.5/29.6分钟，source/config/RNG和实际teacher配对由canonical comparer通过。结果43→50/400，Spatial/Object/Goal/Long从0/37/6/0到3/29/16/2，breadth4→6；相邻保留27/新增23/丢失16、churn39/400、Jaccard0.4091。global tasks1/3/11/13/23/26/31/32从0/0/34/3/2/4/0/0到0/3/29/0/1/15/1/1。相对source47，100/200分别R/G/L=9/34/38和18/32/29。C100主要获得Object并丢失source Goal，200只部分恢复；两点尚远低于目标，依登记规则不追加C训练，不归因整类有序过程失败。C train96为49/96，四suite16/12/15/6、breadth18；相对旧off20040，R/G/L34/15/6。C有训练任务获取，未见任务迁移仍弱。

R100/R200 strict400均完整exit0，37→83，四suite1/26/10/0→0/41/37/5、breadth5→5；相邻保留23/新增60/丢失14、churn74、J=.23711。global tasks1/3/11/13/23/26/31/32从0/1/24/2/1/9/0/0到0/0/24/17/1/36/0/5。R200相对C200多33次成功；相对同51,200queries旧off200108却少25，R/G/L63/20/45。R train96=46，四suite12/11/15/8、breadth18；旧off20040→R46保留33/新增13/丢失7。当前证据显示两项改动均提高训练任务分数，却未带来未见任务净收益；不能单独归因为没有学习或全部有序过程失效。

S100 strict400=59，四suite2/37/16/4、breadth7；global tasks1/3/11/13/23/26/31/32为0/2/34/3/1/15/2/2。相对匹配C10043保留36/新增23/丢失7，churn30、J=.54545。无序多帧参照早期更强，但不唯一归因时序。S200训练完整exit0，200checkpoint经formal inspector通过、source trainable=0；总6048秒、平均更新28秒，独立动作FM0.151461→0.109258。三臂各51200queries/1600conditions/800taskoccurrences/378unique task-video一致，见rcs_training_summary.json。

S200 strict400=45，四suite0/21/24/0、breadth3；100→200保留32/新增13/丢失27、churn40、J=.44444，早期优势未保持。S200 train96=52，四suite22/10/14/6、breadth18；比C49多3但validation低5，比R46多6但validation低38。三臂最终validation/train96为R83/46、C50/49、S45/52，同预算旧off200108/40。它们均有训练任务获取，但新条件分配和消费接口没有得到迁移净收益；不能按S100单点宣称无序模型解决问题或再加时间模块即可修复。

全部六组validation400和三组train96共2688条闭环完整exit0；S200 validation用15worker/5GPU耗时859秒，train96用6worker/2GPU耗时508秒。三臂共153600监督queries，checkpoint、896×3唯一完整LoRA及配对原件保留。统一结果入口runs/analysis/video_consumption_20260911/first_round_summary.json。

首轮后首先登记的学习见active design§7：对保留最强off126追加500/600监督节点，每点correct400，600 train96，两个独立动作验证。旧400仍增长、未证明平台；C/S不追加。原gpu01 5/6他人满负载，采用有lineage的三rank物理迁移，完整权重/optimizer/scheduler/sampler/rank RNG保留，原冻结45e16633代码不改，继续探索性mode/stage。新root horizon_off7_continuation_20260911，不改原run、不冒充formal fresh或原拓扑exact-resume。已在gpu02 0/1/2、micro8/8/8启动；原生合同及完整400恢复通过，实际401的四task、occurrence、video、query seed、frames逐项匹配恢复sampler的预期，每task64queries/权重1/4。401约27秒、峰值reserved41.7GiB。401–500平均25.34秒，500独立动作FM0.104097（400为0.104946），仅小幅下降；500完整checkpoint已通过原生inspector，source trainable=0。完整迁移/launch证据在video_consumption_20260911/continuation；新增峰值16GiB、strg01 used782678700KiB/soft1073741824KiB。

初轮总峰值预算96GiB；formal launch前strg01/data1 used764989296KiB/soft1073741824KiB（含约21GiB可删除profile checkpoint）。profile检查点完成消费后删除，仅保留合同、metrics、梯度/最长视频/恢复报告；正式checkpoint与数据不删除。大资产全部复用，source不复制。条件训练独立worktree已完成集成并移除。

off7续训500 correct400已完整exit0，9worker/3GPU耗时1218.65秒，42shards/400rows。结果73/400，S/O/G/L=1/51/12/9，breadth6；global1/3/11/13/23/26/31/32=0/1/31/20/0/12/6/3。与父400的126实际输入/执行配对通过：保留60/新增13/丢失66，churn79/400、J=.43165；Goal36→12、Long19→9。独立动作FM只微降，但未见任务行为明显下降，不支持把继续监督曝光视为已获得进步。600已完成，完整后续裁决见下段，未登记600之后续训。原件continuation/step500/completed_summary.json、old400_vs_500.json。

off600完整correct400=54，四suite1/24/22/7、breadth6；global1/3/11/13/23/26/31/32=0/1/22/2/0/22/6/1。500→600保留31/新增23/丢失42，churn65/J=.32292；400→600为43/11/83，churn94/J=.31387。train96=64，四suite20/16/16/12、breadth22，相对400的56为46/18/10；独立动作FM600=.104786。两组400与一组96全部exit0、实际配对通过，600 val1205.90秒/train822.95秒；训练200新增更新5216.95秒。继续监督获得训练任务能力，却丢失未见任务能力，停止这条off7未改配方续训，保留126 checkpoint。统一原件continuation/continuation_summary.json。当前仅R的300/400登记段在运行。

后续launch存储复核used752234200KiB/soft1073741824KiB，另计尚在保存的两份200 checkpoint约8.4GiB、R/C物化约8.8GiB及S约13GiB，仍低于原总峰值预算和独立quota。共享文件系统另有84TiB；额度证据见`storage_after_rc200.txt`。

S100后存储复核：strg01/data1 used773630184KiB/soft1073741824KiB，余下S200 checkpoint、S物化及小型评测预计新增不超过16GiB，仍在原96GiB峰值与独立quota内；见storage_s100.txt。

## 方法身份与学习结果

下表为本轮新实验前的历史参照：main canonical曾启用Compiler额外语言query；126候选使用独立冻结commit45e16633的`backend_conditioning=local_h_read`关闭该route，保留local/H-read/exact language与独立D。两者不能混同。main已实现同步双视角输入，实际有分数的checkpoint全部为agentview单视角；双视角只做过输入/梯度smoke。

| 固定实验 | validation200→400 /400 | train200→400 /96 | validation自身保留/新增/丢失 |
| --- | --- | --- | --- |
| all init7 | 103→90 | 41→49 | 56/34/47 |
| Compiler-off init7 | 108→126 | 40→56 | 82/44/26 |
| all init11 | 100→108 | 41→56 | 62/46/38 |
| Compiler-off init11 | 104→119 | 44→61 | 67/52/37 |
| 同target跨rank D绑定 init7 | 115→82 | 32→60 | 59/23/56 |

Compiler删除的验证净收益经两个初始化与init7另一正确视频关联复核，但稳定保持没有修复；两个候选仍增长，不能宣布平台。D绑定只淘汰实际检验的组合。原始矩阵、逐task/suite和曝光见最新专家包；历史强模型、source47及SFT109/107的适用边界见research_history和9月7日证据包。

## 最新视频检查与机制边界

最强126候选固定macro400、train24、states32–35、teacher46/other47、单视角九臂864条，18个worker全部complete/exit0，实际输入/执行RNG配对通过。正确/另一正确/同suite错/跨suite错/乱序/倒序/首帧/中帧/末帧为**53/60/59/59/55/52/58/53/53**（各96）；对应all400为51/53/48/46/47/50/52/51/48。当前最强候选在该固定面板亦未建立稳定动态增量。输入改变具体成功集合，不能据此证明完全忽略视频或纯task记忆；见findings§49。

此前原因深化共2701条配对闭环：1728 all视频控制、640双层Q中介、256训练任务P/C/D端点、36有限更新、32 sealed BBQ端点、9橙汁回放。四次其它task真实更新使橙汁转选BBQ，零当前梯度Adam对照仍成功；P变化可独立复现目标转移。BBQ四例中只换D400使3例转向绿色瓶，只换P400为2例，只换C400为0例，另有模块交互。该现象涉及共享条件映射，不能只归Compiler，亦不证明冻结D是修复。

查询差异压缩及其向读取的传递已测到，但扩大双层差异未带来一致即时闭环收益；fresh删除收益的唯一学习中介未识别。先前真实视频/reader信用/执行诊断还降低了强梯度抵消和直接删policy-Q即可修复的解释。完整证据等级见findings§39–49；不要将几何、非零梯度或训练收益当作时序理解/迁移成立。

所有诊断无Test或held梯度。临时真实更新只有4个gradient macro/1024跨episode queries，另4次零当前梯度Adam对照；没有新增完整训练。Owner曾明确允许wrong/shuffle/reverse用于原因诊断，这些已看过结果不能冒充未触碰的最终因果资格。

## 证据与仓库整理

- 最新本地原件：`runs/analysis/horizon_relation_writer_20260908/causal_learning_20260909/`。最新远程副本及源路径映射见专家包`index.json`；大checkpoint、数据和视频张量保留本地。
- canonical源码、数据/训练/eval入口经只读审计，未发现生命周期明确的新退役簇；不根据目录名或静态无引用推测删除。未采纳候选分支和用户dirty worktree保留。
- 删除137个可再生缓存/重复自检文件，共2,684,218 bytes；profile/smoke、唯一checkpoint及历史证据保留。
- README与计划/进度入口已更新；旧逐次运行状态从当前文件中收敛，历史仍由Git、research_history、findings和原件保存。
- 导出保真、119面板/13,901条统计、13组关键配对、全部新材料引用与隐私字段检查通过。没有新增训练/评测或无关测试；main交付以本文件所属提交为准。

整理前完整进度：[043b58ca的progress](https://github.com/LinFyM/EMBER/blob/043b58ca3f1f3e7ed699b876f4964be96654c7e5/progress.md)。后续待办与权限只看当前task_plan，不恢复历史命令。

R追加获取节点已登记于active design§8：仅R从原200完整exact-resume至400，300/400 correct400，400 train96及独立动作验证；总动作曝光达到旧off400相同102400queries。依据是R37→83仍有获取，C/S分别43→50、59→45不追加。已在原gpu02 4/6、micro6/6通过原生合同及完整200恢复；201实际8条件/4task/256queries、每条件1/8，后续约32秒/update、reserved35GiB。off600物化/闭环继续独立完成，总用卡≤6。新增R峰值16GiB与off600余量3GiB已计入strg01 used794434480KiB/soft1073741824KiB；共享83TiB。没有新增架构、保持loss、VL Meta或D变体。

R300全部400rows/42shards/9worker exit0，耗时1206.51秒；300物化400条件615.33秒，source trainable=0、formal inspector通过。200→300 churn60/J=.41748，Long总数5→5但原5全丢并新增5；Object41→24、Goal37→34、Spatial仍0。相对旧off400126为保留51/新增12/丢失75、churn87/J=.36957；比较均通过canonical实际teacher、source/normalizer及执行RNG检查。新增100更新平均32.28秒；R300累计76800queries/2400条件。原件r/step300/completed_summary.json、r200_vs_300.json、old400_vs_r300.json。
