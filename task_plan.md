# EMBER task plan

## 当前目标：自主推进可重复的有益时序视频编译（2026-09-14）

整体goal保持active且未完成，Owner自主授权持续。目标是正确action-hidden视频经一次生成的唯一完整LoRA获得
可重复闭环增量；有序强于独立fresh全帧frame_set，并跨视频、初始化、相邻checkpoint及固定validation保持。
允许完整视频双向理解，暂不额外强制145/400；信息墙、K1、数据／资源／Git合同保持。

1. **已完成：完整G诊断。** 480episodes按注册关闭，source17/96，四teacher21/25/20/24；
   净+5.73pp、CI[+1.04,+12.5]pp。只支持privileged纠正的有限跨视频／初态控制作用。
2. **已完成关闭：语义状态路径Writer。** 全图实现、最长105帧profile、两臂fresh100、四checkpoint和八配对面板完成。
   train96有序／无序16/22→29/26，validation77/75→48/44；三个资格均失败。完整1,984rows审计通过，39个最终worker exit0。
   保留训练获取和Object增量，停止原组合续训／扫描；不选50峰值，不触发未获资格的other或最终controls。
3. **已完成：冻结实际行为定位。** [工作理论§8](docs/temporal_control_compilation_theory.md#8-语义路径比较后的理论修正2026-09-14)
   降低“重排表示并给普通FM即可得到过程”的支持程度；此前same-object／different-destination现象只来自分数，
   已登记[冻结行为回放](docs/semantic_path_behavior_replay.md)：原四checkpoint、全8tasks、init0／12／25／37，
   共128次已完成，复用原LoRA、视频与RNG配对通过，24个worker exit0；32组固定图像齐全。
   全部32组人工观察已完成，失败同时涉及错物／实例、具体获取／运输和组合目标保持；8/128次回放结果变化完整保留。
   按异质失败分支关闭，不从中选择唯一失败模块，不扩回放／controls或自动加目的地解析、局部动作头或保持损失。
4. **算子核验已通过，进行合法共同学习实现。** [新联合机制](docs/local_correction_field_design.md)已选定：
   同位置过程解释得到的纠正场，既接受真实纠正监督，又直接与裸source状态配对形成LoRA；不保留独立辅助动作出口。
   先按§5固定96条件／1,536 query组合及原source／eta／RNG，检验局部rank16场收缩的t1与full10实际作用。
   4f55968d完成96条件与两读出，三个worker exit0；t1／full10相对source改善区间均严格正、四suite均正。
   按正分支关闭诊断并退役入口；§8已冻结合法Writer及同一纠正场的采样、单位与共同loss；主模型11项机制检查通过，
   在线标签与运行集成已完成，c4c07bbb最长105帧两次共同更新与完整推理exit0；
   全153项检查通过。§9已按实测固定50／100、两臂各400条件／25,600 queries，接下来启动正式相邻比较。
   不扩大oracle或重选其它机制，不把原G分数或新局部误差当目标完成。
5. 仅在完整目标证据成立时完成goal。局部正例、surrogate、单峰、代码或理论文档完成均不算科学目标达成。

当前[局部纠正场§5](docs/local_correction_field_design.md#5-当前激活的固定算子核验)已完整关闭，§2–3机制进入实现细化；
无selected checkpoint、Writer训练或在途GPU运行；回放及行为解释全部关闭，整体goal和自主授权继续。
原件及全量表在[runs/analysis/semantic_path_writer_20260914/READOUT.md](runs/analysis/semantic_path_writer_20260914/READOUT.md)。
下方暂停和未完成旧清单均是历史，不恢复执行。

## 暂停时点：讨论架构与训练的整体原理（2026-09-14）

完整目标仍为正确action-hidden教学视频经唯一完整LoRA产生可重复有益闭环增量，并保持跨同task视频、
初始化、相邻checkpoint和固定validation迁移；暂不强制145/400。**整体goal未完成；
[固定A与幅度／方向诊断](docs/native_correction_acquisition_audit.md)均已完成关闭；当前唯一active design为
[既有真实纠正的闭环前提](docs/native_corrective_closed_loop_audit.md)，已按Owner 2026-09-14最新要求暂停。
source启动尝试在准备阶段被评测范围合同拒绝，零episode；无在途运行或selected checkpoint。**
当前仅进行方法讨论与要求整理，不继续修复、重试或执行下列计划，等待Owner恢复推进。
下一方法的优先设计任务是形成架构与训练的整体工作原理：解释输入理解、可执行的高层操作知识、
唯一LoRA生成、未见任务迁移及能力保持，并明确历史依据、待检验假设和可失败预测。
局部动作／纠正监督尚非已选方案；既有oracle只能检验纠正目标的闭环价值，不能替当前架构的可学习性与迁移背书。
Owner已澄清后继设计边界：允许完整视频双向理解，不再强制过去依赖；保留真实顺序与方向，
有序模型强于匹配的独立全帧无序模型仍是必要方法证据。新设计须解释并预登记如何检验这一优势，
不得据此放宽信息墙、削弱无序参照或重判旧实验；本次讨论不恢复执行。

[Native Correction Writer](docs/native_correction_writer_design.md#10-完整有界结果与关闭裁决)的两臂fresh200、四checkpoint、
8个bank与8个配对面板／1,984rows已完整结束，原始证据审计及全部worker退出通过。
100/200的train有序／无序为15/15、19/20；validation为50/51、50/49。两有序增量区间均未严格正，
增量未相邻同向，按原资格关闭该组合，不续训或扫参，也不补未获资格的controls、Test或RL。

1. 保留真实纠正oracle的跨episode功能正事实，以及本轮有限训练收益、source名义+3、Goal保持与较低churn；
   同时保留validation breadth3、Spatial零、有益有序资格缺失和独立动作／参数拟合有限的事实。完整结论见findings§91。
2. 不把共同标签统计／参数空间存在性当成有限RGB获取证明，也不把弱闭环唯一归到Meta、Compiler、R/A或B。
   新旧教学池及训练目标不同，不以本轮50与旧空间监督71的差额冒充单变量归因。
3. 已核对旧fixed-A／G1／G3 fit-span近邻，8c714e91完成四checkpoint×train24×demo16–19共384次合法生成与原分解。
   实际E约.90–.95、固定A下界F约.27–.31、空间内D约.63–.64，四个F−D区间均严格负；按第二分支关闭。
   66.9%–70.6%误差在现有A空间内，停止以扩展A空间作为必要首项修正；不能由乐观下界宣称有限B或RGB充分。见findings§92。
4. §7只读384条件进一步完成：最优privileged整体倍率只消除实际误差3.87%–6.32%，四个H−J区间均严格负；
   每模型24/24task均值方向缺口更大，按登记关闭。当前A的最小B系数代价较大，限定乐观下界，但不是容量或根因证明。见findings§93。
   下一步先区分可用方向、有限共享纠正、表示／信息与学习信用；已见条件仍有大差额，不能只归新视频。
   不从这些参数定位直接启动新Writer、续训、辅助动作头或参数扫描；先形成可失败且会改变投入判断的最小机制合同。
5. 整体goal继续；已完成的Execution-Aligned、旧冻结52/96正例复核及其它关闭诊断不从旧计划措辞恢复。

暂停中的既有实施项：现有state-free的96套真实纠正标签各运行四个init，共384oracle rows；新source96只运行一次、按task/init配对复用。
全部五面板480episodes固定，检验目标G本身的闭环作用是否跨teacher／init保持；先补既有动作预测正证据到闭环的缺口。
实现只补现有static adapter的真实来源检查，结束后退役；不伪造Writer／G1身份、不复制执行器或LoRA，不以此结果当合法视频生成成绩。

以下为已关闭阶段的计划与结果记录，不由其“当前／下一步”措辞恢复执行。精确运行与授权只看progress顶部。

## 当前物理效果诊断已关闭：部分可预测性成立，替代度量未获资格（2026-09-13）

[操作语义§6–7](docs/operation_semantics_feasibility.md)完成固定train24×demo16–19×3位置的288条件，
4,896条短段动作干预／24,480高层步，CPU、无模型／LoRA／梯度。fcd9a613冻结434.79秒exit0、raw重算通过。
零变化／局部预测MSE=.00089956/.00026749，差额CI[.00048754,.00076067]、23/24task及四suite正；
但预测误差比例.29736不满足预登记<.25（RMS=.54531），因此关闭该J加权FM提案，不实施条件性§6.5。
Object比例.01887，Goal／Long为.41550/.51493；保留部分物理可预测性的正事实，不挑suite或扫描扰动挽救。
夹爪离散分支存在真实效果差，但幅度与连续扰动不同，不能推断夹爪是主因或默认重加权。
原件留在runs/analysis/operation_semantics_20260913/effect_replay，CLI退役。无active design／运行／selected checkpoint，整体goal未完成。

## 当前补充实施已完成：操作语义标签与设计修正（2026-09-13）

Owner最新要求继续仔细推导、实施并按结果调整。[操作语义可行性](docs/operation_semantics_feasibility.md)已完成关闭：
train24×8episode、6,668个CPU状态位置，零梯度／GPU／环境步；源7bd8a6f5、84.34秒exit0，原始数组重算通过。
保留对象／部件及左右指接触，186/192episode有双指接触转换；抽屉8条中6条在采样位置无双指同时接触，
region所属body也不等于区域谓词。因此撤去“通用接触阶段→语义rank”的直接依据，不以数据覆盖默认启动Writer。
新增实际A监督的零B与共享A反例：准确接触响应仍可没有行为／视频贡献，须先明确可执行效果传递，不能重做无关辅助头。
一次性入口已退役，约1.36MiB原件留在runs/analysis/operation_semantics_20260913；无active design、无在途运行。
整体goal未完成；下一项需收敛真实RGB操作证据与唯一LoRA实际效果之间的可失败合同，保持旧失败边界及现行资格。

## 当前goal：有益视频特异性先行，再提升绝对性能

Owner授权依据综合正负证据自主高效推进理论、设计与实验，取得正确视频在唯一完整LoRA中的可重复闭环增量，
并验证跨同task视频、初始化、相邻checkpoint和固定validation迁移。暂不要求145/400；当前goal未完成。

Owner已明确要求重新设置自主推进goal并已激活。依证据自主承担理论分析、方法修正与验证；
当前时间对齐队列与冻结正例复核已完整结束，转入基于全部证据的机制判断。整体目标未达时不标记完成。

**当前无active design；[跨初态相对几何检索](docs/cross_init_relation_retrieval_audit.md)已完成关闭，
[Native Dual-View Writer](docs/native_dual_video_writer_design.md)已关闭。** 时间对齐、冻结正例复核及
[冻结局部动作生成诊断](docs/frozen_local_action_decode_audit.md)均已关闭；整体goal保持，当前进行综合机制分析。

### Native双相机修正：8面板完成，按预登记资格关闭

两臂各fresh200，100/200四checkpoint、8个sealed bank及1,984条闭环记录完整；运行代码defcf734。
新旧四模型800条件的18个采样字段匹配，8面板及相机参照的task/state/language、teacher真实帧索引、
环境／policy RNG配对核验通过。validation各task50条teacher整轮各一次；train96为46–49/states32–35有限池。
全部worker正常退出，累计评测墙钟4417.77秒，controller正常exit0，无在途训练／评测。

| 节点 | train ordered / frame_set | validation ordered / frame_set | validation差额95%CI |
| --- | --- | --- | --- |
| 100 | 36/96 / 32/96 | 64/400 / 56/400 | [0,+4.5]pp |
| 200 | 54/96 / 50/96 | 53/400 / 39/400 | [−.5,+9.25]pp |

100节点S/O/G/L为1/52/8/3对0/48/6/2、breadth6/4；200为0/52/0/1对1/37/0/1、breadth3/4。
200仅Object净正，有序53次成功中52次来自Object；相邻Spatial／Goal归零。
有序相对无序100保留51／新增13／丢失5、churn18、J=.73913；200为35／18／4、churn22、J=.61404。
相邻有序64→53，保留41／新增12／丢失23、churn35、J=.53947；无序56→39为32／7／24、churn31、J=.50794。
两个节点的有序净额均正，但CI下界均未严格>0，200也未达至少两个suite净正；不满足登记资格和稳定保持。

预登记相机参照中，ordered validation单→双为61→64、72→53；frame_set为65→56、81→39。
200同模式差额CI分别[−9.5,−.75]pp、[−20.25,−2]pp；相机×顺序交互虽为正，伴随明显绝对性能损失。
不能把相对无序少退化当作足够的方法收益。冻结source仍为47/400；有序总数高于source但配对task区间跨零。

**当前无active design，无selected checkpoint。** 关闭本有界组合，不追加训练、节点、seed、LR、rank、flow或Meta冻结扫描；
不触发same-task-other／强静态／最终controls、Test或RL。原生双相机动作读出的正证据仍成立，
它未转化为本共享Meta／表示／Compiler的稳定闭环收益，不能唯一归责某个接口或否定全部视频方法。
下一步先综合局部功能诊断、既有共享／蒸馏负结果及当前训练—迁移分离，明确能区分获取与传递不足的预测；
在实质机制与合法判别证据成立前不启动新Writer。完整边界见findings§79。

原件：`runs/analysis/native_dual_video_20260913/`的`paired_summary.json`、`camera_comparison.json`、
`bounded_200_decision.json`、`evidence_audit.json`、`training_pairing.json`和`OVERNIGHT_READOUT.md`。
全部训练／物化／评测contract、四完整checkpoint、8个bank、raw rows和completion保留；整体goal未完成。

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
后续端点×视角诊断与native双相机修正均已关闭；上节为完整裁决，不扫描flow阶段或恢复该候选。

[Meta校准提案复核](docs/video_information_identifiability.md#8-为什么当前不单独追加meta动作校准诊断)已否决
仅由原生动作头MSE决定冻结／替换Meta的下一诊断：它不能区分所需竞争解释，旧校准／读出正数也未修复闭环。
下一步先定义跨初态操作关系和可失败的行为预测，并与旧功能教师、native-span的输入／监督／传递合同逐项区别。
若仍只是局部读出更准、原共享参数映射不变且没有可区分预测，不启动新Writer。

[跨初态关系复核§9](docs/video_information_identifiability.md#9-跨初态操作关系新增监督必须区别于旧状态条件化与任务记忆)
已完成近等价历史检查和四train任务12个存储状态的CPU恢复：状态地址与回顾动作反演已属旧机制；
物体／部件监督有数据来源，但同task的oracle关系图拟合仍可能只是任务索引。下一判别必须排除这一混淆，
不能把可恢复位姿当作视频理解或LoRA传递正证据。无新标签库或模型学习。

据此完成一个无学习映射的直接参照：全部train24的action16–19演示与诊断42–45交叉，固定相对／绝对几何1-NN，
384个episode对、13,064位置完整。绝对／相对／均值MSE=.12443553/.12634790/.25459689，
绝对−相对CI[−.00421347,+.00027549]跨零，四suite均无相对改善；按登记关闭，不扫检索变体。
两种状态匹配都有相对均值的部分动作价值，仍依赖privileged几何／动作；不得直接称为合法过程教师或启动完整Writer。
此结果停止把坐标中心化视为当前主要修复的依据；后续机制须区分真实动作Value、对象／部件获取与参数传递，
不能由此回到已失败的可训练局部头或task图拟合。完整原件及分项限制见findings§81。

当时提出的后续方法选择疑问已由Owner 2026-09-14澄清：有序强于匹配且独立训练的全帧frame_set是持续要求。
冻结模型的内容／顺序因果要求、跨视频／初始化／相邻稳定及固定validation迁移均保持；
不重判或重开已关闭实验。比较对象的区别仍见[可识别性分析§2](docs/video_information_identifiability.md#2-三种不同的比较对象)。

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
