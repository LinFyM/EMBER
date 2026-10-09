# 自身功能修订 Compiler：完整消费者与有界实测

2026-10-10。按[唯一合同](../designs/functional_revision_compiler_20261010.md)完成实现、真实原生导数、FM2/PG1及资源退出。
这是实施/profile交付；没有正式共享长训、新适应、validation/Test、400读出或视频因果结论。
ROOT：`/data1/user/ymdai/ember_runs/functional_revision_compiler_20261010`。
完整分项事实见`analysis/consumer_facts.json`、`profile_envelope.json`、`formal_cost_estimates.json`；原始行/费用保留。

## 1. 实际问题与已经验证的计算联系

χ360完整130对MT153，旧定向I/E+/E0各13/24且存在相反得失，不能由专家FM改善或一次自身成功推出有益修订。
本次依合同撤换raw A/B编码、rank-token编辑及chunk decoder，把当前控制响应与完整参数的真实导数作为写入接口。
原PPW/local field、ADSP/SKNC阴性及T/Long1正例不清零；换成梯度表达不构成新能力证据。

`model.py`保留逐decision事实：真实pre/post Φ、当时两flow时刻H、proprio、实际动作/mask/反馈及episode/time，
先读取全部有序事实，再以其16全局slots和language改变教学空间query，之后才压缩每帧8slots并做4层有序读取。
选定decision token读取教学及全部经验形成256维c；c不读取可微动作。
当前incoming在自身实际RGB/proprio及原noise上走完整50×32、十步原生flow，得到前5×7动作a。
仅小MLP接收可微a；能量末层无bias、零初始化，`q=∂aC`由外层学习，不准备正确动作轨迹。

`execution.py`通过显式76因子输入和每次重算内的绑定产生`Jᵀq`；当前参数、过去经验/编辑及source/prefix停止梯度。
`Runtime.edit_many`为每事件保留自己的M均值，输出完整`Λin−S²ΣJᵀq/M`，没有第二adapter、截断或残差拼接。
外层完整LoRA信用v通过原生forward AD得到`qbar=−J(S²v)/M`，再经小MLP混合导数、读取器反传到共享φ。
不计算source Hessian，不用FD、历史H或单time velocity替代原生十步。
MT固定RMS min/median/max=.002690/.009324/.097685；P确为固定坐标单位，不是gauge不变natural gradient。

原生oracle使用Spatial0/demo29一个原实际decision/noise，所有76因子：

| 检查 | 原件数值 |
|---|---:|
| BF16 VJP/JVP伴随相对误差 | .00226360 |
| checkpoint对不重算方向信用误差 | 0 |
| BF16 ε=.01方向有限差分相对RMS | 2.36259 |
| 同权重/观测/noise的FP32方向有限差分相对RMS | .00322789 |

FP32只用于解释已出现的数值核验异常，之后恢复原dtype；生产仍是正常BF16真实AD。
这支持小扰动BF16差分受舍入影响的解释，没有把BF16逐元素复现当目标，也没有将FD放入梯度消费者。
原件`native_derivatives_0791d711.json`；早期未含FP32核验的5cafacab记录另保留。

## 2. 数据、输出和真实学习消费者

固定task0/demo29、13/demo4、20/demo15、32/demo44，实际incoming依次MT、MT、原incoming_001、原incoming_001。
读取原χ360事件的真实完整E及行为版本，当前F/J重算；没有把旧E称为新方法自己产生的适应。
M分别15/16/16/16，共63。实际successful keep点为15/16/16/0；Long原事件没有成功，不伪造保持目标。
`event_stream.json`保存每事件7非teacher episodes×4区间的28 FM queries；固定四事件各权重.25。
这是四任务的有界实现验证，不能称为36任务等权正式训练或其覆盖证明。

真实FM使用现有offset1/native50×32/前7维及padding；incoming只配对报告，不恢复旧1.2 FM回归项。
keep在原成功自身状态/同noise上比较完整十步前5×7函数，平均点和35坐标；没有把原执行动作改作拟合标签。
FM后fresh RL optimizer，PG无FM，σ=.1 Gaussian原始y/mask、真实success与独立policy/exploration RNG baseline产生score。
每query均匀reservoir16，按N/min(16,N)补偿，对两query及四逻辑事件平均；没对固定16、GPU数或有效动作数再归一化。

实际仅FM1、FM2、PG1三个可丢弃更新。首步只打开零末层的混合导数是预期行为；第二步到Reader的梯度真实但很小：

| 更新 | combined梯度norm | Reader | Experience encoder | Energy |
|---|---:|---:|---:|---:|
| FM1 | 1.8987e−7 | 0 | 0 | 1.8987e−7 |
| FM2 | 1.9002e−7 | 4.8075e−11 | 8.1395e−11 | 1.9002e−7 |
| PG1 | .00416883 | 1.5019e−6 | 2.3280e−6 | .00416883 |

两个FM更新的四事件outgoing FM和incoming相同，成功点keep均0。FM2能量末层weight RMS=3.1353e−5。
因此真实接通不等于形成实用编辑，不能凭非零梯度保证足够长训会解决；FM路径的幅度/投影及优化条件需要main一并考虑。
PG给出的实际信用明显强于这里的FM，而且确实回到视频/经验读取；这削弱“整条共享路径必然断图或只有极弱信用”的解释，
没有证明学到有益pressure、视频操作理解、软保持有效或新初态迁移。两步FM也不够裁决整个假说。

PG前同φ2的四事件q RMS约4.35–4.61e−6；真实qbar RMS依次1.189e−4/2.352e−4/1.316e−3/0。
`qbar=−J_E(Pv)/M`说明外层当前效果只能沿自身支撑Jacobian可达方向教会能量；
新初态改善还需`Jnew P JEᵀq`有合适方向和幅度。现测qbar不保证该跨状态关系，Long无回报差时确实没有PG方向。
旧PPW/local阴性仍限制“局部导数形式天然足够”的支持；当前证据只提高实际完整接口与真实信用可执行性的支持。

## 3. 实际query、负例和停止范围

新初态由原全部实践IDs及32–34排除后取最小两项，四条件均0/1；immutable contract记录实际IDs和独立RNG域。
16条新episode恰好完成，4,456steps含160settling，863个实际replans；9失败耗3,550steps，完整费用计入。
outgoing八条reservoir共128个replans，保存自身双RGB/proprio、50×7均值/commands、Gaussian5×7、noise及真实执行prefix/mask。
对全部16份检查了状态/根seed、noise、mask、形状和finite，逐episode抽一个真实y核对σ=.1噪声消费；没有新环境重采。

| 条件 | outgoing两个success | 独立incoming baseline | advantages |
|---|---|---|---|
| Spatial0/demo29 | 1,1 | 1,0 | 0,+1 |
| Object3/demo4 | 0,0 | 1,1 | −1,−1 |
| Goal0/demo15 | 1,1 | 0,0 | +1,+1 |
| Long2/demo44 | 0,0 | 0,0 | 0,0 |

Object负例和Long零信用保留；没有重采凑非零标签。4/8对3/8来自独立噪声的学习query，不能归因编辑能力提升。
没有PG后能力读出，没有无扰动新初态资格，也没有视频因果/正式相邻稳定资格。
φ2查询行为权重保留`query_behavior_FM2.pt`，明确是weights archive；PG1完整可丢弃实现状态在`implementation_state.pt`。

## 4. 真实吞吐、显存与剩余风险

均为不同原condition/真实query，不是重复tensor的GPU-only伪加速。原件`components.jsonl`和`physical_selection.json`。

| 完整消费者 | physical batch | wall秒 | allocated峰GiB |
|---|---|---|---|
| 四事件F/VJP编辑（63点） | 4/8/16/32/64 | 28.70/16.19/11.42/10.50/10.09 | 10.71/11.93/14.14/18.40/26.84 |
| 真实FM信用后的JVP/共享反传 | 4/8/16/32/64 | 33.18/19.41/12.35/9.86/9.42 | 10.45/10.87/12.02/14.32/18.78 |
| 配对FM112条 | 28/56/112 | 8.48/8.40/8.82 | 17.67/25.43/40.56 |
| PG+keep+共享反传128条 | 4/8/16/32/64 | 105.77/64.00/46.27/44.14/42.18 | 19.03/19.03/19.03/19.03/27.56 |

选择support64、adjoint64、FM56、PG64；112 FM更慢，不能以占满显存择点。
四事件编辑由micro4到64是2.84倍、共享伴随3.52倍、完整PG消费者2.51倍；仅是这些真实组件的短测比值，不是整批方法加速。
编辑context前向约.42秒、原生F/VJP约9.57秒；训练瓶颈首先是真实原生导数/功能消费者，不能由低UTL自动指定GPU算子或只调Reader。
最终PG64：PG完整十步余切20.16秒、keep11.93、支撑JVP8.57、共享context/混合反传1.43。

actual query使用16 bounded persistent slots，真实flow batch为1–13；不padding或dummy。16episodes含重置/采集/保存166.34秒，26.79step/s。
native prefix+flow计109.12秒，主进程env wait28.58秒；余约28.64秒含启动、调度、输入/保存等未进一步分离。
436.64秒env operation是各并发actor求和，不能与wall直接相加或声称CPU占比超过100%。
最终query/PG恢复attempt的设备1秒采样mean UTL67.3%、median89%、max100%、zero9.87%，mem used峰36,405MiB；包含共驻进程，不作本进程独占归因。

PG96/128未测；64→128按现有原生内存曲线推算allocated约45GiB，接近/超过现场可用44.06GiB，非真实OOM结论。
96仍可能可用且有收益，留作正式首步前的物理profile候选；本次唯一PG step已经发生，不在此后重放旧版本扩大矩阵。
四事件64已覆盖全部63支撑，FM112已覆盖全部112真实queries，不能继续靠增加逻辑事件“利用余量”。
source/physical topology扩展未实测，不能由这次单卡锁死world2，也不能预先称world3/4最优。

Runner已把同CPU cohort结束的真实条件接到`edit_many`；保持每slot自己的E、budget/scene/RNG及成功先编辑。
该调度与两个success同时冻结的CPU消费者已验证；本批无新适应，完整新方法practice链/H/Phi/序列化端到端吞吐未实测，必须保留此边界。

![实际消费者吞吐和显存](/data1/user/ymdai/ember_runs/functional_revision_compiler_20261010/analysis/throughput.png)

## 5. 正式投入的可执行成本单位

`formal_cost_estimates.json`是事实推导的规划单位，不是新科学批或选点合同。
单卡逻辑batch4的FM：compile+完整FM/keep/伴随+optimizer平均39.91秒，100共享更新约1.11GPUh，另加冷特征和未单列I/O。
单卡PG：compile+本次16 query+一次选定credit/keep/伴随+optimizer约219.42秒，100更新约6.09GPUh，另加新模型自身适应/刷新。
正式φ版本必须在采集及信用期间一致、RL fresh optimizer，不能拿本次重复profile开销当100次on-policy学习，也不能复用跨版本score作第二轮学习。

刷新/400必须按实际环境steps、实际编辑批次及视频/经验编码、传输/保存另计：
`T≈steps/(观测吞吐)+Σ实际ready编辑batch成本+native features+I/O`，停止仍依实际success/1024预算，不设J菜单。
400条件实践最多409,600steps；完整paired400 final最多136,000steps含settling。
若用当前无H/Phi query速度，分别约4.25/1.41GPUh，只是环境项的乐观规划量；实践另有H/Phi/参数保存和所有F/VJP编辑，不是完整正式ETA。
有用多worker可以降低wall，GPUh/吞吐/内存受cohort、实际scene及共驻影响，须在正式合同内测并记录。
最长视频、205decision E、fresh新模型链、world2/3/4和PG96均未知，不能把4个旧事件profile代替完整成本保证。

现有credit/learning支持完整事件及显式全局权重、梯度归并，Runner复用persistent/原生批量；
正式等权采样、MT/非MT独立层及刷新覆盖、共享更新节点与400/相邻选择还须main登记后绑定，旧180/360课程入口已退役。
main据实际信号和成本形成下一合同；本交付不会凭smoke自动开始未登记长训。

## 6. 工程失败、计费、生命周期与交接

6个GPU费用窗全部闭合；计时日志合计.270509GPUh，科学elapsed78.19分钟。
前两次旧check只从进程内CUDA初始化后起记，按launch-script mtime补计28.17秒的保守启动区间（推导，非直接测量）；
保守记账.278335GPUh，所有失败均保留，CPU实施/消费不冒称GPU计算。第一两次的native特征时间未知，不凭空填零。
已记录神经全片读取92/3,243frames，均计cached重读与profile/replay；另有冻结native编码记录115teacher frames/318自身观测。
旧check的额外native编码缺直接计数，依warm cache/源事件可推26teacher frames/16自身观测，时间未知；不称92为新方法部署J。

失败1为source_record字符串未转Path，按实际caller修复；失败2为LIBERO路径缺配置，CPU真实构造器复现stdin EOF，ROOT配置后构造/退出通过。
8个失败slot均无raw/decision/episode；保留partial和完整计费，从有效FM2接续首次query，没有重采。
失败3为稀疏冻结遗漏版本化configs，actual authority loader在policy/env加载前失败；补足固定src/tests/configs并在新clean冻结承接。
pipe断连也保留pending operation/exit信息并可关闭，未知step不默认为已证明0。
针对实际Reader/原生执行、PEFT/完整信用、数据墙/真实query、存储/IPC及成功先编辑的55项CPU检查通过。
科学计算来源0791d711（AD/FM/profile）与6e4aac1d（实际query/PG），原χ360行为/incoming与本次代码身份分别保存。

结构自审：13个canonical模块、2,710行，核心model298、execution165、credit260、learning109、runtime263、profile389。
Runner460行继续凝聚slot/IPC生命周期；没有第二Writer/Simulator/训练栈或长期兼容分支。
旧decoder、诊断/collection/evaluation/180360 launcher及失效重复测试已退役，实际PEFT/信息墙检查并入现有owner。
原件、必要权重、恢复状态及费用保留；task-owned已合并工作树在Git集成后退役，清单/释放量见ROOT analysis。
退出码0，两节点本项目GPU进程均退出；末次ROOT allocated约2.944GiB，未连续测磁盘峰，不能将末值写成历史峰。
strg01 data1用量1,262,825,392KiB/2,147,483,648KiB，data0预算未合并、不写新产物。
完成记录/Git/资源与一次当前main真实处理回执后串行交回tracked窗口；EMBER后续科学判断由main接续。

## 7. main判断：信用接通以后，首先检验真实学习，而非扩大梯度数字

main直接读取实际model/runtime/native导数与credit消费者、三个更新原件、16条query事实和成本分项，
并只用CPU比较本批FM2行为权重与PG1后权重中能量末层及四个相关投影；没有新GPU、环境或重复工程测试。
这里要解释的实际不足仍是：上一完整编译器会改变目标选择、放置和后段转移，却未把经验作用变成广泛净收益。
本批把写入接口换成真实控制导数，已经解决“如何把功能压力写回完整参数”的可执行性；尚未回答应该写什么、能否学到和能否迁移。

### 7.1 为什么不能把PG梯度大等同于修订大

FM2的末层weight RMS为3.135254e−5；PG1后为5.213884e−5，实际这一步weight差的RMS为2.999623e−5。
Reader图像投影、经验图像投影和动作投影的实际更新RMS分别约4.57e−7、2.79e−6和1.37e−5。
因此PG确实更新了读取和能量通路，但combined梯度约两万倍于FM，并未产生两万倍的末层参数更新。
原始梯度经Adam的一/二阶矩归一化；per-coordinate梯度低于eps时仍受eps限制，不能从一个总norm推算整网学习速度。
本次没有重算PG1后的q、输出LoRA或能力，以上权重变化不充当这些未测结果。

令能量末层为w、小MLP末层之前的特征为h(c,a)，则`q=(∂h/∂a)ᵀw`。
在w=0时q=0，`∂q/∂w`可以非零，而Reader/经验模块收到的混合信用含w，首步为零、初期弱符合这项identity初始化。
这不是永久断图；同样也没有证明若干百步以后上游必然学会。此前经验编译的初始化论证已提醒，越过零点本身不能解释最终能力。
真实写入是`δΛ=−P J_Eᵀq/M`；在新状态的一阶作用为`δF_new≈−J_new P J_Eᵀq/M`。
P的典型因子单位平方约8.7e−5，仅表明坐标缩放；Jacobian包含大量实际hidden与参数贡献，不能只由这个标量断言编辑必然极小。
但它确实没有把`J_E P J_Eᵀ/M`变成单位响应，q同时承担合适方向、幅度及跨状态可用性，当前设计没有免除这些学习负担。

FM使用跨episode专家功能误差，当前强incoming可能已较接近这些查询上的目标，返回能量的余切可能小；PG把实际成功差乘完整episode的score，
可以产生更大信用，但也含长时序噪声，非零或幅度大不说明知道哪次目标获取/放置应该怎样修改。
Object的两个−1及Long的零信用必须保留。incoming/outgoing采用独立采样，4/8对3/8尤其不能解释为几乎identity的编辑已经净增1次成功。
D_keep在这两次FM中为0，与几乎未改变的函数一致；它还没有经历实际有用修订与保持的冲突，不能据零loss称保持问题已解决。

所以本次加强的是“真实自身响应、当前LoRA与共享读取器之间存在可执行信用联系”。
它没有加强到“已经学会操作知识/有益修订”，也没有支持把FM或整条方法判为不可学。
PPW/local field的功能回写阴性、ADSP/SKNC的有限保持阴性、旧shared-SDE父161→158/首批经验135仍然约束预期；
T的161/150/public103及上轮Goal/Long局部有益经验也保持，不能因这次更自然的接口便清零任何一侧。

### 7.2 数据覆盖改变了怎样组织学习

主讨论的只读分工核对两旧池manifest及全部27个含非MT条件的记录/来源：216个condition、362个endpoint，
其中314个MT、48个非MT，后者只在16/36任务存在。每task虽都有6条教学，却并非都经历了真实递推参数。
全局MT/非MT各采一半再在各自支持任务间均匀，会让这16任务过重；旧池/刷新池各半也不等于非MT曝光一半。
因此先保证task等权，再在该task的真实可用层内分配。缺层意味着没有这种经验，不能用复制/换incoming补齐。
新模型冻结180后采72个实际条件，后半只用这些真实事件，给当前分布每task20次明确曝光；缺少递推层、近identity及两条新教学的局限照实保留。
这种组织修正了已知覆盖计量与当前行为来源的问题，不被当作净收益的充分原因，也不能让重复事件伪装成更多独立映射。

### 7.3 有界投入与能改变判断的结果

当前最小有价值的下一步是学习轨迹加完整能力读出，而非再用一个方向导数或梯度放大试验替代学习。
main登记[同一设计§6](../designs/functional_revision_compiler_20261010.md#6-首批正式学习fresh-fm360真实分布刷新与一个固定400节点)：
全模型fresh FM360、180时72真实条件刷新、原固定train48在180/360各读一次、唯一预定360完成correct paired400。
零头/P/M/优化器/精度均不改，不从本批三个可丢弃更新接训。360不是收敛声明，也不是用train小面板选出的点。

若q/参数逐渐变化但仍没有净控制收益，需降低当前功能压力学习的支持，辨认尺度、方向和目标尚未区分的部分；
若自身成功点保持但新状态丢失，说明局部保持不足以覆盖实际闭环分布；若train净正而held不正，迁移解释才获得更多支持。
若完整能力改善，则保留收益及任务得失，之后仍需相邻held与公平视频因果证据。单点过MT不能结束EMBER问题。
实际共享RL的消费者已经实现，但本批先建立有能力读出的监督起点；接RL不是自动晋级，也不要求FM先胜MT这个新门槛。
FM负结果只约束本阶段实际检验的假说，不能偷换为已检验完整FM+RL方法。

现有成本单位支持监督360约3.99GPUh；刷新/400中的真实H/Phi、长E和I/O仍需完整链实测。
本批预计10–16GPUh、科学3–6小时，观察线20GPUh/8小时、峰224GiB；工程接续预计1–3小时，资源须另行live准入。
这是一轮能产生学习和闭环判断的有限投入。无论结果正负，main收到完整原件后自主接续，不让Owner再为同一推进请求许可。
