# EMBER task plan

## 当前goal：有益视频特异性先行，再提升绝对性能

Owner授权依据综合正负证据自主高效推进理论、设计与实验，取得正确视频在唯一完整LoRA中的可重复闭环增量，
并验证跨同task视频、初始化、相邻checkpoint和固定validation迁移。暂不要求145/400；当前goal未完成。

Owner已明确要求重新设置自主推进goal并已激活。依证据自主承担理论分析、方法修正与验证；
当前时间对齐队列与冻结正例复核已完整结束，转入基于全部证据的机制判断。整体目标未达时不标记完成。

**当前active design为[Native Dual-View Writer](docs/native_dual_video_writer_design.md)。** 时间对齐、冻结正例复核及
[冻结局部动作生成诊断](docs/frozen_local_action_decode_audit.md)均已关闭；整体goal保持，当前只执行新登记native双相机候选。

### Native双相机修正：编译完成，配对闭环已启动

新证据是相同输入位置下native双相机动作读出稳定改善；本项只将腕部加入native前缀，
V-JEPA继续读取同一agentview四帧窗口。信息墙、完整H、主FM、fresh共享学习及唯一38-target LoRA保持。
新dual ordered／frame_set各fresh200，固定100/200及8个train96／validation400 correct面板；
复用旧agentview两臂的同节点原件作输入范围参照。相邻有序资格未通过即关闭，不追加局部参数扫描。

已实现显式native／prior相机绑定及新运行身份，旧单相机配置与checkpoint只在冻结历史树中复现；
相机路由／cache／配置／物化／完整模型126项检查通过。最长93帧双相机两次完整更新已exit0，
热条件26.48秒，allocated39.17GiB／reserved42.17GiB；采用frame_chunk4／microbatch8／prior batch4。
profile无正式checkpoint，不改变100/200学习预算或提供初始化。
两臂已从defcf734 clean pushed detached树启动，world3各自fresh0→200；GPU01 ordered=0/5/6、frame_set=1/3/4。
现场966.9GiB快照后，正式launch复核data1为967.1/1024GiB，保守预计999.1GiB；两节点共使用6卡。
两份runtime contract均确认native dual／prior agentview、source与prior trainable参数为0。
两臂均已正常exit0，完成200步／800条件／51,200个动作查询；100与200四份完整检查点已保存并通过物化前检查。
0／100／200各24-task的登记动作诊断完整保留，不用于替代闭环裁决。四模型800条件的18个登记字段完全配对。
四个编译作业均exit0，8个bank均sealed：各checkpoint的train96／validation400完整，共1,984个task-video条件。
step100 train96两臂已完整exit0：ordered36／frame_set32；S/O/G/L为9/11/12/4对5/11/13/3，breadth15/14。
96行task/state、视频与环境／policy RNG配对通过；有序保留29／新增7／丢失3，churn10，J=.74359。
当前+4仅为train单节点结果，不证明validation迁移或相邻保持。controller964812已启动ordered100 validation400，
GPU01=0/1/3/4/5、每卡2 workers；已核实launcher2012389及10个workers实际存活，启动记录为8task／400states。
每面板独立保留资源准入、run contract、raw rows与completion；原始训练配对为同analysis根的`training_pairing.json`。
句柄、精确命令与资源在`runs/analysis/native_dual_video_20260913/launch_contract.json`；当前尚无完整validation结果。
正式两臂及初始8面板估计约26GiB，32GiB阶段峰值预算保持。
条件性other／强静态／最终controls在触发后另核quota与增长，未预先启动。

### 原生端点读出：已完成，保留视觉范围限制

[端点诊断](docs/source_endpoint_readout_audit.md)新增三单元各384位置完整exit0，复用既有dual/full10。
agentview full10/t1均值/t1固定probe为.36548/.34789/.34732，均差于task mean .25060；
dual对应.13673/.13217/.13177，均胜过task mean。三个同读出相机差额均24/24task为正、区间严格为正。
触发视觉范围分支；同视角增加flow步数没有改善，停止以denoise深度为当前首要修复依据。

双相机的单步H有原生可读动作价值，不能转写成当前agentview或学习后Meta／E已充分；
也未区分腕部信息与source输入分布匹配，不能由动作预测直接声称视频过程／LoRA收益。
已核对双视角实现与V-JEPA单视角合同；1,669份范围内库存中38份可判定observer均为agentview。
上述证据已用于登记native双相机与原单视角视频先验的最小输入修正及可失败预测；
不将双视频K结果混为双相机实验，不恢复已关闭checkpoint训练。原始endpoint_*证据完整保留；代码e1a06b46。

### 当前执行计划

新增[冻结source状态输入诊断](docs/source_state_input_audit.md)已完整完成：MSE free/mean/true=.13673/.14173/.12469，
任务动作均值.25060。free−true区间跨零，状态补全前提未通过；free本身相对动作均值的正事实保留。
后续端点×视角诊断也已关闭；新native双相机修正已登记，上节为当前实施计划，不扫描flow阶段。

后续方法选择尚需明确：独立训练的全帧frame_set优势检验是否属于持续硬门槛，还是本轮特定研究问题；
冻结模型的内容／顺序因果要求、跨视频／初始化／相邻稳定及固定validation迁移均保持。当前不据结果自行改变口径，
也不重开已关闭实验。具体区别见[可识别性分析§2](docs/video_information_identifiability.md#2-三种不同的比较对象)。

1. **时间对齐已完成：**fresh200两臂及8面板／1,984rows。train100/200为44/51对36/53，
   validation为61/72对65/81；早期有序优势未保持，validation无资格，不续训或扫描。
2. **冻结正例复核已完成：**固定旧200双模型，24task×4video×8state加source，共1,728rows；
   有序359/768、静态361/768、source32/192。三种有序−静态95%CI均跨零，四video净额0/+4/+2/−8。
   新条件下有序对source能力仍在，原大幅有序优势未复现；按预注册上限关闭，不增加训练或面板。
3. **理论边界复核已形成：**[视频信息与可识别性](docs/video_information_identifiability.md)核对了全帧frame_set、
   train24当前状态目标和原受益9task的新条件净额−1/288；区分独立训练臂比较与固定模型因果干预。
   跨episode FM允许同task视频无关最优解，但不证明有限共享模型不可能从视频学习迁移知识。
   **新增诊断已完成：**原train24留出片段的固定10步/8noise生成MSE为.31625/.31677，均差于task mean .25060；
   有序小幅优势保留，但现成动作中间量前提未通过，关闭且不追加noise／步数／训练。
   **当前理论任务：**明确现有系统缺少何种可迁移操作知识及其合法学习路径；不能把局部FM差额、
   表示可解码或任务内拟合视为已经建立可用于参数编译的过程教师。
   [运动对应审查](docs/video_information_identifiability.md#7-显式运动对应能补什么以及为什么尚不足以启动新writer)
   未形成直接替换／追加dense-flow编码器的充分投入依据。下一重点是目标物体变化如何转为跨初态操作关系，
   以及能分开获取与参数传递不足的合法证据；不以warp质量通过默认启动新Writer。
4. 只有新增机制证据与相对近邻历史的实质区别成立，才登记下一项最小、合法判别实验的预算与停止分支；
   不因CI跨零、局部正数、总分高于source或代码准备好而自动启动完整Writer、参数扫描或最终controls。
5. 整体目标仍是可重复有益视频增量及跨视频／初始化／相邻和固定validation迁移，在合法冻结后确认最终controls。
   当前未选checkpoint，不使用Test、held梯度或RL；局部诊断不构成方法选择或goal完成。

用卡遵循Owner上限：两节点合计最多8张，空闲卡总数不超过10张时最多6张；训练与全部评测共享额度。
formal来自clean pushed frozen commit，后续新增长重新检查独立quota与两节点实时资源。

## 已完成历史与最终口径


1. **已完成：具体设计与实现。** 保留T×L到长程过程表示，明确完整H首次读取；实现辅助真实FM和分组cotangent。
   复用官方source/data/完整LoRA出口/evaluator，退役旧活动Writer路径；旧结果留在Git与formal artifacts。
2. **已完成：**验证梯度分配、identity启动、teacher墙、query/noise配对和checkpoint身份；真实长视频profile决定物理batch与段长。
3. **已完成：**主方案、纯FM首段与匹配frame_set共40面板、9,920闭环rows；所有节点尚未获得可信有序增益。
4. **已完成有界200/300窗口；目标未达。** main与frame_set均76,800queries，后段两者均validation退化，原训练已结束。
   **已完成：**固定表示读出64epochs诊断；留出FM改善但所有train24任务仍落后原LoRA学生，不形成部署资格。
   **已全部完成：**active design第8节去蒸馏fresh200比较；100训练42/38、validation57/61；200训练55/55、validation72/67。
   训练获取提高，四项validation均低于原main，相邻及换视频保持未获可信改善；不延长或自动补rho0无序训练。
   **已完成：**第9节原生中层读出32epochs；held .13979对旧头 .14028，差额CI跨0，仍24/24落后LoRA学生 .11000。
   **已完成：**第10节teacher侧VL Meta fresh200及8面板；validation100=61/61、200=62/64，未形成可信增益与保持，不延长或自动补frame_set。
   **已完成：**第11节单独移除辅助表示FM；fresh200、800条件/51,200queries及8面板匹配。validation100=60/55、200=64/54，未形成可信迁移收益。
   当前候选关闭，转入过程获取的综合机制判断；不由本项单个涨跌自动触发下一轮模块修改。
   不延长读出probe或扫描层位；所有旧Writer/probe冻结，新干预从fresh开始且不继承probe。
5. 达到设计登记的正确视频收益、换视频/相邻保持及迁移后，冻结方法和选点，完成视频内容/顺序因果确认，才完成当前goal。
6. 下一阶段另以保持视频收益并提升绝对性能为goal，恢复长期145/400及完整稳定/breadth资格。

每轮只根据真正检验的因素修正；不以辅助loss、wrong退化或单点峰值完成goal。不启动95-task、不恢复旧候选；
新实验live资源与精确命令写run contract，阶段状态写progress，跨轮结论写findings，历史结果写research_history。
