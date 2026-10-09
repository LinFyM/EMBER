# χ360完整补充读出：局部晚期学习没有抵消整体能力损失

2026-10-09；执行/科学消费 session `01a11a0a-fd39-74b1-83ba-001ede5330df`，主讨论
`01a11b05-3460-7eb0-aca8-177d6d86ef48`。合同为[冻结χ360完整补充](../designs/parameter_compiler_fullreadout_20261009.md)。
本批固定原`macro_00000360`，只新增368条件、只读复用原32，完整correct strict paired400为 **130/400**。
对强MT153保持104、获得26、丢失49；对χ180净增3，但相邻获得30、丢失27。Long1晚期改善确实存在，
其它任务退步抵消大部分收益，完整方法仍未超过强MT，也未建立训练域净修订。

这是看到原Long1局部结果后登记的开发补充；原180科学停止、失败和费用保留。没有新训练、刷新、其它checkpoint、
held视频controls或Test，没有重新选点；不作为原预注册通过、独立确认、视频/经验因果或正式相邻稳定资格。
本报告及机器可读成功集合直接来自原始完整结果、配对参照、实际经历、源码及退出/费用。

## 1. 完整能力与相邻得失

| task（suite内ID） | MT | T2340 | 旧135 | χ180 | χ360 | MT→360 R/G/L | 180→360 R/G/L |
|---|---:|---:|---:|---:|---:|---|---|
| Spatial3 | 41 | 45 | 38 | 39 | 36 | 33/3/8 | 34/2/5 |
| Spatial6 | 7 | 6 | 2 | 0 | 1 | 1/0/6 | 0/1/0 |
| Object1 | 36 | 45 | 29 | 29 | 26 | 22/4/14 | 23/3/6 |
| Object6 | 10 | 5 | 14 | 10 | 7 | 1/6/9 | 4/3/6 |
| Goal3 | 0 | 0 | 1 | 0 | 0 | 0/0/0 | 0/0/0 |
| Goal6 | 35 | 36 | 32 | 34 | 33 | 31/2/4 | 29/4/5 |
| Long1 | 24 | 24 | 19 | 15 | 27 | 16/11/8 | 10/17/5 |
| Long9 | 0 | 0 | 0 | 0 | 0 | 0/0/0 | 0/0/0 |
| 合计 | 153 | 161 | 135 | 127 | 130 | 104/26/49 | 100/30/27 |

四suite按Spatial/Object/Goal/Long为 **37/33/33/27**，MT为48/46/35/24，χ180为39/39/34/15。
Long1对MT净增3，却伴随其它suite净丢26；Object是最大损失来源（净丢13），Spatial净丢11，Goal净丢2。
χ180→360的Long1净增12，另七task合计净丢9。breadth从χ180的5增到6，只恢复Spatial6的一行；
仍等于MT/T的6，低于旧135的7。Goal3/Long9仍为0，不能由Long1提高推断广泛新任务能力。

| 参照→χ360 | R/G/L | churn/400 | Jaccard | breadth参照→360 |
|---|---|---:|---:|---:|
| MT | 104/26/49 | 75/400=18.75% | 0.5810 | 6→6 |
| T2340 | 101/29/60 | 89/400=22.25% | 0.5316 | 6→6 |
| experience135 | 96/34/39 | 73/400=18.25% | 0.5680 | 7→6 |
| 180 | 100/30/27 | 57/400=14.25% | 0.6369 | 5→6 |

R/G/L按同一task-state的成功集合定义；churn=G+L，Jaccard=R/(R+G+L)。全部逐task/suite值和成功/损失state集合
保存于`ROOT/scientific_analysis.json`，展开的四参照逐task表在`ROOT/analysis/per_task_comparisons.tsv`。
MT原153成功中丢49（32.0%）；Object6虽有6个新成功，却只保持MT原10中的1个，不能把总分7解释为小幅稳定变化。
χ180原127成功中有27在360丢失，Long1相邻Jaccard仅0.3125；完整相邻对照不支持稳定保持的强主张。

## 2. 原32、新368及Long1剩余18

| 范围 | 行数 | MT/T/旧135/180 | χ360 | 对MT R/G/L | 对180 R/G/L |
|---|---:|---|---:|---|---|
| all400 | 400 | 153/161/135/127 | 130 | 104/26/49 | 100/30/27 |
| original32 | 32 | 16/17/14/6 | 20 | 13/7/3 | 5/15/1 |
| new368 | 368 | 137/144/121/121 | 110 | 91/19/46 | 95/15/26 |
| Long1_remaining18 | 18 | 8/7/5/9 | 7 | 3/4/5 | 5/2/4 |

原32全部是Long1 state0–31，因原long-first启动/停止形成，原始行逐行只读核对相等。
剩余18为state32–49：χ360为7，对MT8/T7/旧5/χ1809；对χ180R5/G2/L4，Jaccard0.4545。
完整Long1为27/50，对χ180R10/G17/L5，对MTR16/G11/L8。
因此原32的晚期增益没有在剩余18重复为净收益，完整task对MT的优势由原32净增4与剩余18净丢1组成。
这是条件范围的实测异质性；两段既不独立随机确认，也没有固定E的纯checkpoint干预。

新增368合计110，对χ180同子集121净丢11；不能把原32的20/32率外推到其余任务。
原32/新368的scene、state-video排程、condition seed、环境/政策RNG、实践预算和最终官方合同一致；
物理worker顺序不选择视频。完整聚合每task50个state和50条合法teacher各一次，适应中的缓存重读另行计费。
原32读取代码为`db53422e3a2e56048b606dc62f5558045de71221`，新368为`f0c49811cda3fcc1a55190d7b46164daef101935`；
本次只修物理H持有/保存并增加真实save事件，不改模型、动作、标签或数值合同，不能借此将两段差异归于新架构。

## 3. 实践成功、重读与最终迁移

| task | own success/50 | 实践steps | 失败steps | 全片神经读取 | final/50 |
|---|---:|---:|---:|---:|---:|
| Spatial3 | 49 | 8,382 | 2,174 | 59 | 36 |
| Spatial6 | 12 | 41,904 | 40,062 | 207 | 1 |
| Object1 | 48 | 15,234 | 6,398 | 71 | 26 |
| Object6 | 27 | 35,006 | 29,352 | 139 | 7 |
| Goal3 | 1 | 50,724 | 50,486 | 198 | 0 |
| Goal6 | 49 | 11,714 | 5,984 | 69 | 33 |
| Long1 | 32 | 35,232 | 23,202 | 77 | 27 |
| Long9 | 0 | 51,200 | 51,200 | 100 | 0 |

完整400合计 **249,396实践steps、920重置**，自身成功218、预算停止182，无state复用。
其中失败episode **208,858steps（83.75%）**；末次失败/截断尾部46,198steps，尾部已包含在失败steps，不能重复相加。
最终新初态另 **117,844steps**，完整读出实践+final共367,240，含原32历史35,641。
实际新增为227,034实践+104,565final=**331,599环境steps**，无新failed/partial开销，低于495,872登记上界。

400次初读发生在首次实际实践后的编辑，520次重读，合计 **920全片神经读取、33,606读取frames**。
原32贡献49读取；本批新增368贡献368初读+503重读=**871全片、31,003frames**，只这部分计新增费用。
本批新增native teacher编码13,360frames；完整含原32为15,061。缓存命中仍要通过神经Reader，照计重读；
这里不将native编码帧数等同神经读取，也不将内部activation重算另算成教学观看。
所有条件固定MT起步，每episode真实成功/1024含settling预算后编辑，成功经验也先进入G再冻结；没有预设J菜单。

| 事后实际J | conditions | own success | final success | 同子集MT | MT→360 R/G/L |
|---|---:|---:|---:|---:|---|
| 1 | 163 | 163 | 99 | 103 | 83/16/20 |
| 2 | 107 | 39 | 23 | 29 | 16/7/13 |
| 3 | 16 | 16 | 5 | 5 | 3/2/2 |
| 4 | 75 | 0 | 3 | 9 | 2/1/7 |
| 5 | 39 | 0 | 0 | 7 | 0/0/7 |

J由实际成功/预算产生，与task难度、episode horizon和已发生轨迹共同变化；该表只作事后描述，不能推出少/多读的因果效应。
218个实践成功条件的最终成功为118，另182个预算停止条件最终成功12。
实践成功和最终成功对应不同初态、参数版本，不能把100个差额直接称为最后编辑造成的遗忘；
它说明成功经验进入编辑本身没有保证新初态能力。Spatial3/Object1/Goal6实践分别49/48/49成功，最终仅36/26/33。
Spatial6/Goal3/Long9共143,828实践steps、505全片读取，最终150行仅1成功，失败和尾部成本不能省去。

一个直接核对的J1不利条件是Object1、teacher33、final state2（`task_11_demos_33`）。
实际MT在实践state33用147含settling steps成功，28次replan，行为版本只有MT；消费成功E后做一次G编辑，
最终state2的MT配对参照成功而end失败。原件路径及episode/endpoint见`ROOT/analysis/representative_own_success_edit_loss.json`。
完整J1子集同样MT103→end99，R83/G16/L20；全部J1的incoming均是MT。
因此“只增加非MT递推曝光就会解释全部损失”受到反证；仍不能据一条episode指定抓取/放置等微观唯一原因。

## 4. 从实际问题连续解释特征、算子、信用和执行

**需要解释的是有用编辑与能力保持没有一起建立。** 原共享训练180/360的固定train48都是MT/end/null=27/25/25；
end相邻R23/G2/L2。360 held完整130并未改变这个训练域事实。因而缺口不能只归为held迁移，
也不能由同分说真实E完全无作用。null只按真实链相同次数/停止点从MT递推，未输入真实中间Λ，属于有限诊断。

**实际输入怎样参与参数。** ParameterEncoder读取当前完整A行/B列，按固定MT scalar RMS归一化，
分别线性投影并拼成每target/rank的256维q，加target/rank地址后LayerNorm。
自身E保留双时点图像Phi、flow时间1和0.1的完整H、proprio、真正执行的前5动作及mask、反馈和episode/step位置；
原生事实停止环境梯度，学习的事实注意力/时序编码形成16slots。
教学Reader用exact language、这些E slots和实际参数q改变8个空间查询，再读取真实Phi/H，保留frame位置/端点后作4层时序编码。
两个axial editor再从教学/事实memory更新q，输出`δq=sigmoid(gate(q))*out(q)`。
这给出了E和incoming影响教学重读及最终参数的实际路径；未给这些token预赋“理解了操作必要性”的语义。

对当前因子的一个64维块x，实际差分decoder为
`b=Wx(x/s)+We e+b0`，`x'=x+s*Wo[SiLU(b+Wq δq)-SiLU(b)]`。
地址e是每target/role/rank/block的可训练坐标，s为固定MT RMS；δq投影和输出投影无破坏零点的独立bias。
初始化末输出投影为零，所以δq=0保留incoming；Wq/Wo非零使δq通常有功能导数。
已训练后不再自动保住x，更不自动保住执行功能。单层LoRA变化为
`(B+δB)(A+δA)-BA=BδA+δB A+δBδA`，作用于当时原生hidden，再经过policy和环境改变下一个观测。
完整接口与首步identity已证明路径可执行；本批J1得失和总体负分证明这不足以保证学成。

**实际信用教会什么。** 每真实事件抽取自己的E endpoint和实际incoming/行为版本，从同task另7个非teacher专家episode
各四时间区间取28个query。监督不进入教学输入，query选择也不依据E具体失败/成功的决策位置。
在同native prefix、noise/time的配对FM下，实际目标为
`l_out+0.2*ReLU(l_out-stop(l_in))`（真实四condition等权及28query均值）。
对outgoing的梯度是`[1+0.2*1(l_out>l_in)]*grad(l_out)`，再通过完整A/B余切、decoder及共享编码/读取完整replay回传。
这只是同一个专家FM方向在回归query上加权20%，没有新增独立匹配incoming动作或已成功行为的方向。
保留项名称不能替代其实际导数，专家query平均更准也不能直接代替自身控制分布更好。

**信用如何到闭环，中间尚未成立的联系。** FM监督50位置、前7 action维的单time速度；部署真实10步积分得到动作，
只执行前5步，再从自己产生的新观测重规划。共享训练改善某些专家query，首先须转成自身到达观测上的有用前5动作，
这些动作还须持续达到任务目标并迁移到未参与实践的新初态，才能形成EMBER收益。
现有后45update平均配对FM改善约0.77%、train无净收益、held丢49个MT成功，说明这些联系没有被结果保证。
本批未加行为定位probe，原件不足以指定上述链路的哪个微观步骤唯一失效；不得由FM改善跳到能力结论，
也不得由闭环阴性跳到“模型忽略视频”“梯度不通”或“FM一律无效”。

**竞争解释受到哪些约束。** 同task不同E的query目标分布相同，理想的共同task参数可同时降低这些query，
不必然要求按自身经历区分编辑；有限G的条件Jacobian仍可依E变化，所以这只是经验针对性学习不足的结构假说。
非MT incoming训练全程仅81/1440事件、未屏蔽74、覆盖16映射，递推分布曝光不足仍可能加重多次编辑；J1负例说明它不是全部原因。
共享条件干扰、教学表征/容量、专家分布与自身状态的信用关系及跨初态保持均未被单独隔离。
原T161有限视频FM正例、旧shared-SDE与experience135完整阴性一起说明：既不能宣布教学/FM不可行，也不能把加RL自动视为答案。
本次Long1局部提升增加了“现有算子/监督能改变部分task能力”的证据，却不证明视频或E各自带来收益，也不覆盖其它task的损失。

**对投入的实际更新。** 完整360否定的是“本次后半刷新+相同FM学习已经带来广泛且净正的修订”这一具体预期；
对MT仍净丢23、train仍不优于MT/null，降低原样续训或只加递推曝光作为近期充分修复的优先级。
不否定实际参数条件或整个视频编译方法族，也不抹去Long1正例。后继须分别推导为什么新的信用/读取/参数约束
能够针对实际缺口改变执行，并预言会恢复哪些原能力、获得哪些新能力；由main结合完整历史裁决有界验证。
本批没有自动开启新方法、RL、controls或rank/LR/seed小扫。

## 5. 物理修复、成本和证据入口

新ROOT为 `/data1/user/ymdai/ember_runs/parameter_compiler_fullreadout_20261009`。
父ROOT为 `/data1/user/ymdai/ember_runs/parameter_conditioned_compiler_20261009`，原队列/early_stop/scientific_stopped只读，未重启。

- `evaluation/formal360/results.json`、contract/400 primary shards/condition记录及完整参数/experience；原32用引用和symlink，载荷没有复制。
- `scientific_analysis.json`与`analysis/consume_fullreadout.py`：完整及分列成功集合、所有R/G/L、churn/Jaccard、读取/实践/费用。
- `reused_conditions.json`：原32身份及父路径；`launch_contract.json`：exact命令/env/Git/设备/quota/矩阵。
- `batch_exit_code.json`与`batch_execution.json`（唯一fullreadout_001）、`costs.jsonl`、六worker回执：exit0/complete，所有新增条件真实执行。
- `implementation/actual_H_storage_check.json`与`startup_health.json`：实际序列化消费者和六worker实际实践/编辑/保存证明。
- `wakeup_receipt.json`：真实batch退出后约7.61秒观察到本消息对应处理，route=start、无设置覆盖，无Queue或第二等待者。

Runner原单slot索引H仍持有全native batch底层CPU storage；本次在持有边界clone紧凑own H，
用真实save_condition/torch.load的B2消费检查及现有执行/存储/配对消费者 **18项通过**。
新实际condition的63条H逻辑与不同底层storage均12,902,400bytes，shape[2,50,1024]/BF16保留。
旧原件不重写；修复减少保存放大，不改变逻辑数值，不能据此解释原科学阴性或宣称受控整批提速。

新增费用 **4.034421GPUh**，12费用窗口均闭合，以node/physicalGPU区间并集去除Supervisor/worker重叠，不双计。
新增batch墙钟 **45.66分钟**（含CPU聚合），科学GPU跨度45.26分钟；新331,599steps对应约121.03steps/s（完整batch墙钟）。
这包括模型加载、实践、重读/decoder、最终环境、保存/队列与退出，是实际六卡吞吐；没有同条件受控两卡/六卡比较，
不能称六卡最优或外推整批1.9倍。费用略高于估计2–4GPUh上端，低于8GPUh/4h观察线；无新增训练/失败成本。
原32和此前训练/失败费用不重复计。父批回执修正13.053185GPUh与本新增相加为约17.087606GPUh；
原保守账本13.480595GPUh仍不改。两批合计903,959实际环境steps，教学神经passes4099+871=4970，原32不重复加入。

## 6. 吞吐、存储、资源退出与交回

| worker | 新conditions | 实际steps | worker wall分钟 | steps/s | native平均batch | allocation GPUh |
|---|---:|---:|---:|---:|---:|---:|
| gpu01:1 | 75 | 61,833 | 37.95 | 27.16 | 13.11 | 0.658351 |
| gpu01:2 | 18 | 22,354 | 40.22 | 9.26 | 9.23 | 0.689692 |
| gpu01:6 | 83 | 65,217 | 37.48 | 29.00 | 13.19 | 0.650588 |
| gpu02:0 | 68 | 67,062 | 37.82 | 29.55 | 13.17 | 0.635079 |
| gpu02:2 | 80 | 68,604 | 38.10 | 30.01 | 13.40 | 0.647704 |
| gpu02:7 | 44 | 46,529 | 44.92 | 17.26 | 13.59 | 0.753007 |

六worker全部exit0，实践和final不同LoRA真实native batch峰均16，平均batch依worker为9.23–13.59；
其余worker速率为17–30steps/s，gpu01:2为9.26，六卡存在真实异质性。任务/轨迹、尾部与共驻负载不同，
没有控制比较足以指定慢卡唯一原因；不能把18个条件的低速当作新方法科学阴性或声称所有卡持续满UTL。
已记录native_prefix/flow、edit、env_wait及各环境operation计时；环境operation是并行环境耗时之和，
与native/edit墙钟有交叠，不能累加成整链占比。完整运行没有连续CUDA显存/UTL峰，保留为未知；启动显存快照不是运行峰。
成本区间显示六卡共同运行约38.07分钟，最后按实际完成退出，没有dummy留卡；最大6卡、每节点3卡。

实际保存368事件的logical payload **45,047,918,330bytes**，allocated **45,052,538,880bytes（41.958GiB）**。
执行器`artifact_growth.observed_payload_bytes`历史字段实际累计allocated_bytes，审计明确纠正名称含义，不改原回执。
保存事件只涵盖condition文件，不含host cache、队列、log、分析和代码；不能把41.958GiB写成完整ROOT占用。
当前ROOT不follow原32 symlink的allocated **105.972GiB**，logical105.914GiB，新增实际增长105.972GiB；
其中evaluation约41.967GiB、host派生cache约64.000GiB。当前低于220GiB计划，连续历史峰未测，至少不低于当前快照；
末次样本推导109.958GiB是payload+64cache+4余量的预测，不是已观测全程峰。

strg01独立data1 user quota当前 **1199.405/2048GiB**（hard2058GiB），软限余量 **848.595GiB**，
共享data1另有约80.053TiB；两者分别核验，不合并预算。user-wide quota比launch增加106.146GiB，不能全部归本ROOT。
本次保留原始经验、完整condition参数、父180/360恢复点及所有失败；无空间不足，没有为掩盖费用/结果删科学原件或改data0写入。
本批代码工作树在Git合并/推送后退役，准确释放量保存在`ROOT/analysis/owned_worktree_retirement.json`；
冻结commit保留，旧路径不再作为可直接执行命令，必要时按该commit显式重建data1 detached tree。

双节点现场检查六个已知worker PID均不存在，无本ROOT剩余CPU工作任务、无ymdai compute GPU进程；
其他用户任务没有被kill/pause/reset。原件为`ROOT/analysis/resource_exit_audit_20261009/{resource_exit_audit.json,raw_snapshots.json,report.md}`，
12窗口及六回执闭合、368 save事件与worker logs匹配，退出快照不被写成未来永久资源保证。

本次报告/状态/history/findings与canonical窄修复由clean pushed main交付；
`ROOT/scientific_completion.json`记录Git、读取身份、原件、实际成本/退出与判重。
只向当前main一次带来源的科学报告，active Steer或idle/notLoaded保留设置resume/start，
以`ROOT/main_delivery_receipt.json`的真实处理为tracked/Git交回证据，发送后实验session不再写tracked。
main须消费完整130及其局部正例、能力损失和未知，连续推导有依据的改进并自主接续；没有另起科学运行或等Owner例行审批。
