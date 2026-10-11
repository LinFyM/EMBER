# 发送端归一化与query16有限批：传播量级受控，首学生未超过MT

冻结fa03656f的新有限批已实际退出0，完成fresh G216、全36首学生及MT/G/E同面板query16，
随后完成原task13/17的初始模型与G216输运对照。原始行独立复算为G339/576、MT340、E455；
MT→G为R304/G35/L36、churn71、Jaccard0.810667。当前没有明显控制增益。
这是授权train24+non-held90 support12上的首学生机制读出，不是完整Actor共同训练或held资格结果。

ROOT：`/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/normalized_flow_query16`。
整批原件为`ROOT/coordinator/normalized_G216_students_query16_02/exit.json`；初次导入失败保留在01及G attempt001，
02按相同冻结代码从0 fresh完成，没有重复已保存训练更新或更改科学合同。
唯一active design仍为[原设计§11](../designs/video_guided_proposal_writer_20261010.md#11-当前有界批gfield发送端归一化与query162026-10-12)，
本报告只消费已执行结果，不登记下一批合同。旧v5 RL9已由主讨论收口，当前结果不恢复旧训练。

## 原件、计算与有效范围

原始全臂行是`ROOT/analysis/first_student_query16_rows.jsonl`（1,728逻辑行、576个配对条件），
对应每task的`student_data/G216_normalized_first_students/task_XXXX/{path.json,parent.json,query,readonly_E}`。
既有summary为`ROOT/analysis/first_student_query16_summary.json`。
独立CPU消费脚本/统计为`ROOT/analysis/whole_completion_20261012/student_readout/{consume_originals.py,statistics.json}`，
其中保留576行配对证据、完整per-task/suite、288条引用来源；anomalies为空。
训练/CP/费用/已保存输运消费为同目录`training_transport_readout.py/json`，anomalies亦为空。
本次消费没有新增model forward、GPU、episode、梯度、工程测试或probe。

唯一生成架构干预是六层G.field的跨层发送量归一化，沿原block→rank→target→rank→block算子限制广播量级。
static、Actor、ContextReader、双meta、CFM目标、完整38-target/rank8坐标和16步Euler保持原职责；
共享G/meta/Actor与optimizer均fresh，旧G参数没有迁移到新算子。
输入仍为真实exact L、action-hidden保序V、MT参数及指定自身MT bootstrap H，E只在固定S_G坐标中的速度标签`(E−MT)/S_G−xi`侧。
教学/native读取与自身执行不混用，query不进入G的条件或后续H；E不成为候选、信用rewards或H producer。
首有效生成参数固定后，在相同初态/scene/env与policy RNG上读取三臂，执行阶段不读取教学。

14/16/16/4四池按预注册init_order完整互斥覆盖50；全部36使用首个有效G_000，一次尝试、一次真实G实践，
没有按成绩补采或挑教学。实际G实践成功18/36。G216本身不消费query16，不能把更细的读出称为Actor信用已学好；
本批没有warm/RL/held训练或评测。
每task两个query初态曾参与E selection，总72行：MT43/G50/E55；另外504行MT297/G289/E400。
G的净变化在两组分别+7/−8，但这不是注册的新qualification arm，也不能据此改变面板或作泛化/因果解释。
该面板独立于本condition自身实践，仍不称独立teacher泛化。

## 配对控制传递、保持与任务分布

| suite | 行数 | MT | G | E | MT→G retained/gained/lost | G净变 |
|---|---:|---:|---:|---:|---|---:|
| Spatial | 96 | 53 | 47 | 79 | 42/5/11 | −6 |
| Object | 96 | 34 | 33 | 68 | 27/6/7 | −1 |
| Goal | 96 | 49 | 46 | 64 | 41/5/8 | −3 |
| Long | 96 | 33 | 37 | 61 | 25/12/8 | +4 |
| non-held90 | 192 | 171 | 176 | 183 | 169/7/2 | +5 |

G任务净优/平/劣为9/14/13；20任务有新增成功、18任务有丢失，有成功任务覆盖MT33/36、G34/36、E36/36。
总体近MT掩盖了71行交换，不能只报告−1或把小幅净差称统计等价。
E相对MT为R321/G134/L19、净+115，22任务净优、2任务净劣，说明当前目标在同一面板确有显著操作能力供给，
不因此认为教师已完善或该优势已由视频编译迁移。
G相对E为R316/G23/L139，净−116；task17 MT/G均0而E12，task29 MT/G均0而E3，不忽略这些明确传递失败。
task32的MT2→G5、task43的MT7→G11等局部正例保留，不能用它们替代总体判断。

| task | suite | MT/16 | G/16 | E/16 | MT→G retained/gained/lost |
|---|---|---:|---:|---:|---|
| 0 | Spatial | 14 | 14 | 15 | 14/0/0 |
| 1 | Spatial | 4 | 3 | 15 | 2/1/2 |
| 2 | Spatial | 14 | 13 | 16 | 13/0/1 |
| 4 | Spatial | 8 | 6 | 7 | 4/2/4 |
| 5 | Spatial | 6 | 7 | 14 | 6/1/0 |
| 7 | Spatial | 7 | 4 | 12 | 3/1/4 |
| 12 | Object | 7 | 7 | 11 | 7/0/0 |
| 13 | Object | 7 | 8 | 7 | 5/3/2 |
| 14 | Object | 11 | 10 | 16 | 9/1/2 |
| 15 | Object | 3 | 3 | 9 | 2/1/1 |
| 17 | Object | 0 | 0 | 12 | 0/0/0 |
| 19 | Object | 6 | 5 | 13 | 4/1/2 |
| 20 | Goal | 8 | 6 | 8 | 4/2/4 |
| 21 | Goal | 14 | 13 | 16 | 13/0/1 |
| 22 | Goal | 10 | 10 | 15 | 8/2/2 |
| 25 | Goal | 2 | 1 | 6 | 1/0/1 |
| 28 | Goal | 15 | 16 | 16 | 15/1/0 |
| 29 | Goal | 0 | 0 | 3 | 0/0/0 |
| 32 | Long | 2 | 5 | 13 | 2/3/0 |
| 34 | Long | 11 | 10 | 10 | 8/2/3 |
| 35 | Long | 12 | 12 | 12 | 11/1/1 |
| 36 | Long | 1 | 3 | 12 | 0/3/1 |
| 37 | Long | 7 | 6 | 12 | 4/2/3 |
| 38 | Long | 0 | 1 | 2 | 0/1/0 |
| 42 | non-held90 | 16 | 16 | 16 | 16/0/0 |
| 43 | non-held90 | 7 | 11 | 15 | 7/4/0 |
| 51 | non-held90 | 14 | 14 | 16 | 14/0/0 |
| 55 | non-held90 | 16 | 16 | 16 | 16/0/0 |
| 56 | non-held90 | 13 | 15 | 13 | 13/2/0 |
| 62 | non-held90 | 14 | 15 | 16 | 14/1/0 |
| 64 | non-held90 | 16 | 16 | 16 | 16/0/0 |
| 73 | non-held90 | 14 | 13 | 14 | 13/0/1 |
| 95 | non-held90 | 13 | 13 | 13 | 13/0/0 |
| 96 | non-held90 | 16 | 16 | 16 | 16/0/0 |
| 97 | non-held90 | 16 | 16 | 16 | 16/0/0 |
| 101 | non-held90 | 16 | 15 | 16 | 15/0/1 |

| E/MT成功类别 | 行数 | G成功 | G成功率 |
|---|---:|---:|---:|
| E成功、MT失败 | 134 | 22 | 16.42% |
| MT成功、E失败 | 19 | 10 | 52.63% |
| E与MT均成功 | 321 | 294 | 91.59% |
| E与MT均失败 | 102 | 13 | 12.75% |

只有22/134个E独有成功由G实现；35个G新增成功中其余13个来自E/MT均失败条件。
G同时丢失36个MT成功，包括27个E/MT均成功条件。局部控制扰动可以产生交换，
本分层没有证明G已学会E能力排序、视频操作知识或H纠错。
同场景首个native前5动作的归一化RMSE为MT→G0.006278、MT→E0.088126；
当前G实际功能扰动比E小，这只定位传递缺口，不以动作距离或内部参数距离替代闭环能力。
逐condition固定选MT或G的事后经验上限为356/576，允许只读E参照则为460；
两者都不是可部署选择策略，也不是逐state拼接参数或新资格成绩。

## 训练与输运：恢复可用量级尚未形成所需方向

实际216次optimizer更新、864个CFM事件，每task24次；每个9-update块完整覆盖36任务一次。
所有事件P=MT，真实H非空、单producer、标签存在、有效逻辑B4/任务权重不变；两rank实际chunk16，source无梯度。
末三个全36块均值1.970163/1.961623/1.975886，对照原固定尺度下task等权零场期望约2，
显示小幅局部学习，不能据loss降到2以下称已建立控制传递。全程平均2.001694包含初期，不能只据全程均值否定后段变化。

既有native诊断原件为`ROOT/analysis/G216_transport/{diagnose.py,statistics.json,exit.json}`，
运行日志和launch收据保留在`ROOT/jobs/G216_transport_task13_17/attempt_001`。
真实V/L/MT-H/P及保存xi保持，两task生成复算与保存候选余弦为1、MSE约1.5e−7，接受正常数值差异。
没有zero-image/fake action query、held数据、新episode或optimizer更新。

六层广播RMS约0.57–0.61，末block RMS约3.7、rank约9.9、target约6.5；
末LayerNorm后的局部block差异RMS约0.48–0.50，对照旧v5诊断的约0.000264。
这支持干预确实限制了原raw-state循环放大并保留局部信息的量级；不再把同一种巨量广播压制原样作为当前唯一解释。
但归一化保障信息可用量级不等于损失会教出正确输运或任务映射。

| 固定context | G216 CFM t=0 MSE | t=.5 MSE | 配对零场MSE | 16步生成与xi余弦 | 生成目标MSE |
|---|---:|---:|---:|---:|---:|
| task13，E=P=MT | 1.031512 | 1.014589 | 0.999460 | 0.986178 | 1.049855 |
| task17，E≠MT | 2.721093 | 2.622730 | 2.703774 | 0.986190 | 2.737675 |

task13的解析标签是−xi：两点预测都弱于零场，field沿−xi的投影系数为−0.00457/−0.00242，所需系数为1；
16步累计修正反而沿xi有+0.01073投影，尚未学到该context的基本反向噪声输运。
task17在t=.5比零场改善约3.00%，label cosine0.19870，而t=0仍略差；这个局部正例保留，
不能将两个task所有预测都称为完全零场。
两task累计field修正RMS约0.170，生成编辑RMS约1.025，最终目标误差仍分别高于单纯xi的零场基线。

新initial模型的field RMS约0.578、生成目标MSE1.33220/3.04122；G216降低这些无关输出，
生成与xi余弦从约0.866增至0.986。这显示真实学习在抑制初始错误输出，但保留了大部分起始噪声，
不把参数更新或预测敏感性当作有效参数编译。
合法噪声翻转应引起沿2xi、RMS约2的速度改变，实际变化RMS约0.267（t=0）/0.135–0.140（t=.5），
方向cosine仍约−0.035至−0.037；反应较旧场大，但方向不符合解析要求。
固定x只改t的变化RMS约0.012，可能离开典型训练密度，仍只作有限定位，不判定唯一时间根因。

这批区分了“巨量消息压制已经缓解”和“所需输运/目标控制尚未学成”。
结合E在同面板的大幅优势、G只传递少量E独有成功及大量保持交换，不能把当前失败仅归因最终Actor选错、
旧query4太粗或教师完全无优势。本批Actor尚未更新，query16也未进入G训练。
剩余竞争解释包括局部坐标/时间的实际利用、条件到具体编辑的共享映射与当前学习组织；
当前原件未唯一分解它们，不扩展探针或提前指定新架构。后继科学取舍由主讨论消费这些事实后接续。
新旧闭环面板从query4变query16，不能直接比较raw分数并把差异归因于发送端归一化。

## 真实成本、退出与恢复能力

38个实际job identity（37个整批窗口任务加1个输运诊断）的latest均退出0；39个job尝试与3个物理profile尝试保存，
总42个实际尝试，2个失败：初次formal缺configs导入、首profile文件名遮蔽stdlib。
失败费用0.019431 GPUh已含，原失败没有CFM有效更新。
最终ROOT launch-inclusive allocation为6.379302 GPUh：G含失败2.992589、首学生3.273161、profile0.102843、输运0.010708。
summary的6.368593是诊断前快照；诊断内段0.006286已包含在launch费用内，不再相加。
原billing.jobs40是receipt项数而非38个独立job，包含profile汇总及G两次尝试；统计范围明确，不改原账目。
共驻时不能将新旧ROOT的分配GPUh相加称去重物理GPU小时；旧教师/MT复用成本不重计。

02整批wall9,482.674秒=2.634h，G成功尝试5,367.450秒=1.491h，符合启动时2.1–5.5h预测。
实际最长rank步平均24.585秒、范围12.305–44.444，峰25.208GiB allocated/26.609GiB reserved；
真实两rank逻辑B4/chunk16消费与吞吐、最长视频profile保留，不按保守默认或空等world4执行。

1,728个逻辑query行中1,440个新增episode：MT432/G576/E432，323,933新增环境步。
另外36次真实G实践9,099步，合计1,476新增episode/333,032步。
144条历史MT query引用及144条E=MT引用构成288条逻辑复用；后者来自本次同一MT行，9任务没有新增E episode。
36条旧MT bootstrap及历史8,131步不重计，其他旧教师/独立report不重计。

训练与全部学生读取来自clean pushed detached fa03656fd61d26f69393fa2940cdea77a0ae441f；冻结树保持clean。
唯一CP216 schema `ember_full36_sender_normalized_flow_writer_v6`，保存psi/theta、完整AdamW、sampler216、
rank RNG×2、world2/logical4及版本216；CPU mmap核saved metadata及485个实际optimizer状态的step均216。
没有执行恢复forward或逐tensor一致性检查。完整CP、fresh initial模型、候选/H/原行及当前引用源继续保留。
新ROOT当前约1.9GiB（du不追踪复用symlink），远低于登记16GiB新增空间预估；没有复制大资产或删除恢复状态。
本批实际止于此，没有新warm/RL/held、Teacher扩展或额外probe；整批结果一次交主讨论科学消费。
此前主讨论旧v5的五份科学原件已按1e96deb7保存于旧ROOT/analysis/whole_completion_20261012/main_scientific_readout，
该路径及记录commit随本次真实整批交付合并提供。
