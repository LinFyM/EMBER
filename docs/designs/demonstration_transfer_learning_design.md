# 教学参考与跨初态功能查询：对应学习候选

2026-09-27。这是尚未验证的完整学习假设。§5/7/8数据构造及配对修正均已完成并独立核验；
§10/11/13完整学习与官方比较工程已验收；§15正式运行面转换已审阅并集成，允许指定执行者按登记精确commit和现场preflight
开展首个P/I288完整批次。实际承接/启动以progress及launch原件为准，没有576、MT长训练、controls或数据扩建授权。
实际执行状态只看[progress](../../progress.md)。科学依据见[机制分析§36–37](../analyses/feature_to_operator_mechanism_20260926.md)。
Reader、条件速度270/450及其它已关闭运行不恢复；本设计不宣称已找到它们的统一根因。

## 1. 改变的学习关系与完整竞争解释

现有主FM用同task、独立episode的教学V和查询(x,a)。它正确阻断逐帧复制，却不保证教学中的具体操作参考
与查询动作来自同一反馈规则。给V再加更精确的动作/关系标签，也不自动改变另一独立query给予最终LoRA的信用。

本候选先在训练侧固定一条完整示范作为参考c，在不同真实初态下变换物体相对末端目标并实际执行，得到查询轨迹Q_c。
部署输入仍只有原示范的合法RGB与exact language；特权pose/action/分段仅构造训练标签，不进入Writer。
拟检验的是`p(V_c,Q_c|task)`的功能对应，相对于保持相同视频和查询边际的`p(V_c|task)p(Q_c'|task)`。
它没有增加deployment任务信息，而是改变什么真实功能误差能给教学特征信用。

解释例子见机制§37.3：当动作来自`mu_c(x)=k(c-x)`，同一参考的跨初态FM直接监督c对应的条件场；
独立参考监督的是关于c'的混合场。完整FM可以表达多峰，不以“必然平均出坏动作”为立项理由。
潜在收益来自合法教学选择可迁移、跨决策一致的操作参考，减少有限模型需要拟合的条件歧义。
是否真的提高成功而非只降低MSE，须由完整LoRA验证。

竞争解释至少包括：

- 对应关系让RGB中的对象角色、接近方向、接触前后变化获得可迁移功能信用；同数据的对应臂应优于独立配对臂。
- 只是新查询质量或覆盖更好；使用相同新查询的MT-BC和独立臂也改善，对应本身没有额外价值。
- 参考只规定无益风格或时钟，或合法RGB不能恢复训练侧使用的参考；训练拟合改善而未见任务闭环不增，甚至保持更差。
- 共享生成器/执行特征未学到可靠规则；改变配对仍未改善完整结果。不能因此自动添加语义头、保持正则或另一架构。

前两者也可能同时成立；任何中间拟合或轨迹生成成功都不等于EMBER通过。

## 2. 合法特征、唯一LoRA与真实梯度

完整候选的数据流为：

```text
exact L + 原教学agentview RGB（stride5及真实末帧）
  → 原生Text/VL/Action读取，fresh rank4 Meta；真实prefix、完整50-horizon H
  → 现有Core与Procedure：语言所指视觉Value、相邻画面变化、有序原生动作响应
  → 现有完整rank16 FactorHeads生成全部38目标的A_v(C),B_v(C)
  → 与fresh、共同学习的完整rank128 β在算子层相加并合并
  → 唯一rank144/alpha144 LoRA；自身双相机/state上的真实FM与闭环执行
```

此处选择固定函数类供对应/独立两臂比较，不把扩大条件出口或公共路径作为新的修复主张。
β和视频分支均fresh、同一主FM联训，不加载MT-BC或任何旧Writer权重，不冻结公共分支、不建立阶段课程。
完整公共参数用于学习可复用的自身状态反馈，视频分支可以改变Q/V的选择与Value及action-in/out；
常量函数可达只排除一个表示阻断，不保证优化、视频增益或保持。β不能被当作已证实的能力保险。
该组合本身也不是投入理由；唯一主要因果假设是上述新功能对应，两臂的架构、初始化、预算均相同。

对每个目标m，令`W_m(C)=W0_m+Bβ_m Aβ_m+Bv_m(C) Av_m(C)`，实际部署因子为
`A_m=[Aβ_m;Av_m(C)]`、`B_m=[Bβ_m,Bv_m(C)]`；不得同rank分别相加造成交叉项。
所有38目标均如此合并，只有76个canonical因子，无第二adapter、外部controller或在线视频读取。
β的A为合法非零identity模板、B零；视频分支沿现有identity模板和零末层，不改rank/scale寻找结果。

教学读取只装现有读取Meta，不新增β参数回读这一变量。Core/Procedure和完整生成器沿共享owner，
不移植条件速度R/U或撤回的Reader。完整H和相邻E已经存在，不声称过去没有这些信息。
视频中具体的相对控制参考是待学语义，不把native响应、注意力或可解码性直接叫作已经理解。

查询自己的`x=(RGB,state)`与真实50步动作a产生`z_t=(1-t)a+t e`，标签`e-a`。
唯一loss是实际最终LoRA上的普通全horizon、前7真实动作维FM；无同视频辅助、语义头、蒸馏或RL。
其梯度经`d loss/d W_m`到公共β、完整A_v/B_v和条件编码器；Q更新实际改变自己的注意力选择，
V更新实际改变聚合Value，action-in/out分别改变去噪输入和速度读出。source基础权重可微参与计算但0参数可训练。
对象pose、BDDL身份、阶段、c索引和生成是否成功不作为条件feature；c索引仅用于训练配对/provenance。

固定LoRA如何承担反馈：在理想特征`[x,1]`下`[-k,kc]`即可实现`k(c-x)`，无需执行时回读示范。
真实原生hidden是否提供可学坐标、时钟轨迹能否蒸馏成自身状态反馈，以及共享更新是否保持作用，均是待检验的假设。
本批§5不实现或运行上述模型；未来若改变函数类，必须另记改变及对应对照，不能静默称为本方法。

## 3. 与最近完整历史的实质区别

| 已有工作 | 已实现且必须保留的事实 | 本候选真正改变之处与未解决之处 |
| --- | --- | --- |
| 旧同视频辅助 | 同一完整LoRA得到自身示范动作信用；1500有165/147、2100为158/159，36任务另一合同反向 | 查询实际来自不同物理初态，主FM本身对应同一参考；不重新加权旧辅助。可迁移与保持仍未知 |
| LocalActionGrounded/NativeCorrection/LocalField | 前后RGB—动作、真实cotangent、同参数消费者及跨episode FM均有实际先例 | 新增的是教学与新初态执行的联合数据；不把它们说成没有动作关系或真实梯度 |
| SEOD/GOMQ及phase | 真实十步端点、学生状态expert信用，有局部闭环增益及未保持 | 旧teacher供同task不同视频共用，未以每条教学的操作参考生成独立初态查询；新controller也远非任意状态强expert |
| R2及功能code | 完整教师成员选择、R10真实功能跃升及held不足 | 不在参数/函数集合中取min，不输入特权后验code；监督仍由唯一合法生成LoRA消费 |
| 相对几何1-NN | 换相对距离后取另一示范的原动作，未优于绝对距离 | 本批变换控制目标并执行新轨迹；旧阴性保持，不恢复检索、坐标/尺度扫描 |
| Unified/DJNFR/条件速度 | 公共部分并未带来可靠保持，条件速度V151→102/L147→112 | 本候选不把公共+完整残差本身当修复；由新数据的对应/独立配对直接检验区别，旧阴性继续约束 |

MimicGen仅为训练侧轨迹变换的实现依据，固定源码及边界见机制§37.4。
不引用外部成功率担保LIBERO可用，不安装/运行外部项目或扩大文献线。

## 4. 完整学习的主要比较与裁决原则

未来正式比较必须同时冻结：同一生成查询数据、同一原教学多重集合、任务/源参考权重、查询次数、noise/time、
初始化和更新时标。对应P臂输入该查询的源参考V_c；独立I臂在同task内按预登记平衡随机调度输入V_c'。
不能仅固定交换两条视频：确定性互换仍能传递完整参考身份，不是独立。
须允许配到原参考，并在完整训练窗口报告源参考×条件参考计数；不得借此重排视频内部帧。
同一query在两臂的自身图像/state/action和FM随机量完全相同；改变条件会改变完整联合风险，不能只称改变一个协方差项。

若成功筛选后不同参考覆盖不同初态，须保留失败账，并在有共同覆盖的初态上做实际交叉对应；
不得让某条视频总对应更容易的初态。完整样本量、任务支持和采样规则须在后继训练前冻结。
工程两个task的成功率不允许扩展成36任务通用teacher。无法在明确成本内形成足够任务覆盖则关闭本具体构造。

具体地，源参考c与初态u在尝试前独立，成功指标为`S(c,u)`。只保留成功后，实际联合分布正比于
`p(c)p(u)S(c,u)`，通常已不独立；不能因尝试是全交叉就说最终训练初态无选择偏差。
对预先固定的一组参考使用共同集合`U*=intersection_c {u:S(c,u)=1}`，在该集合上等权交叉可消除
这一组内初态与参考的成功选择关联，但仍只证明U*上的支持，不代表所有初态或整库可迁移。
逐状态当前occupancy可因参考而不同，这是本干预的真实后果；P/I须使用完全相同的query记录保持它不变。
不能靠重加权凭空补出S=0处的成功数据，也不按结果挑某两个容易参考代替原固定组。

另一个独立限制是教师的内部路点时钟k。数据动作实际为`a=K(x,c,k)`；单次生成的部署policy只能用
自己的x与已写入参数的c。如果访问支持上不存在由`(x,c)`确定正确操作阶段的反馈函数，
真实FM也不能把额外时钟信息凭空交给部署policy。对象是否已到目标、手与对象的当前关系等是可能的阶段证据，
不是已经被原生hidden学会的事实。不同初态的新轨迹提供了变化的自身状态样本，却不保证偏离路点后的纠正覆盖。
因此不增加部署时钟、teacher回放、phase标签输入或外部controller绕过；生成成功仅解除数据来源问题，
最终自身状态闭环和能力保持仍是裁决。若只有示范式拟合，这个完整方法失败，不自动续接旧phase/DAgger课程。

强MT-BC需使用相同新查询数据与忠实训练预算，区分数据收益和视频对应收益；已有155/400强参照仍保留。
不因新MT分数低而降低最终参照。最终还需真实learned language参照/换正确视频及冻结后的controls说明视频作用；
这些并非本批一次铺开的多臂矩阵，先据工程profile另冻最小正式比较的节点与预算。
正式闭环仍为single-checkpoint strict paired400，50个不同合法教学/任务、固定state-video调度和自己的policy RNG。
相邻能力保持、per-task/suite、breadth、R/G/L、churn与success-set重合均保留，Test和最终时序controls当前不动。

P仅降低loss、仅训练task成功、或只胜弱化参照，都不支持最终方法；P≈I而新MT也提高，更支持数据解释。
P有真实held正确/换正确视频增益且超过强MT、保持相邻能力，才支持所提完整学习联系。
P有增益也不能唯一归因“模式混合是旧根因”。完整有信息量窗口失败时关闭组合，不以换rank、aux、seed、冻结β或无限续训挽救。
正式学习的GPU-hour须由本模型的真实profile计算，不能沿用Reader或条件速度的数字。§10另行限定工程更新，
不授权正式学习、held评测或将工程权重续成候选模型。

主讨论随后补足的投入边界见机制§37.8：额外功能信用必须超出查询自身状态/带噪动作/语言，
不能由源身份、轨迹差异或连接段拟合认定；静态参考和视频动态的增量也须区分，不自动追加局部探针。
36个已授权训练task的BDDL中30项剩余目标只含On/In，但其中global25是推盘，不能假定全部适用抬升分段；
另6项含开关等不同操作。这里只做元数据可行性判断，没有授权读取更多源标签或扩建。
后继若扩大，需一项事前固定、跨任务共用的分段/迁移规则及完整失败账；不为每task修补controller，也不削减最终评测。

## 5. 首批已执行：16次以内训练侧迁移工程

任务名`demonstration_transfer_engineering_20260927`。唯一执行者`01a0dd6c-f2e5-7971-821a-56766e1c0f22`。
目的为实现并核对一种完整训练关系所需的真实查询来源；不是语义/接触/生成器等模块必须逐个过关的课程。
本批结束仅回报一次完整结果或实质阻塞，执行者不自动扩建数据、训练模型或开展其它实验。

### 5.1 固定输入和物理控制语义

只用coverage_v1的train global34、38，原示范各demo0、demo1（共4条，取自已核8条RGB内，不扩样找模式）。
canonical原件为`data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a/libero_10/`中的
`LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo.hdf5`及
`KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5`。复用原件，不复制HDF5。
本批明确允许读取这4条**train**的动作、sim state、模型XML与BDDL，仅作离线标签和数据构造。
不得读取held action/state标签，不加载source、Writer或其它神经模型。

物体/目标使用官方BDDL的精确实例和谓词，固定次序如下；CPU恢复若发现此角色与源示范不符则报告，不换示范：

| task | 第一物体与放置目标 | 第二物体与放置目标 |
| --- | --- | --- |
| 34 | porcelain_mug_1 → plate_1 | white_yellow_mug_1 → plate_2 |
| 38 | moka_pot_2 → flat_stove_1_cook_region | moka_pot_1 → flat_stove_1_cook_region |

训练侧使用四段：第一物体接近/抬起、第一物体运送/释放、第二物体接近/抬起、第二物体运送/释放。
这只是固定路点分段，**不是可靠grasp/contact语义标签或自适应阶段策略**。
源边界固定为：各物体body原点相对本源初始z首次上升至少0.03m，分别记l1/l2；
第一物体的官方On谓词为真且源gripper命令<0的第一个l1后时点记r1，边界必须`0<l1<=r1<l2<N`。
用源控制步区间`[0,l1), [l1,r1+1), [r1+1,l2), [l2,N)`，均非空；最后一段终点为源控制序列末。
未满足边界、初始目标已完成或源在段序上不符的参考记不可用，不调阈值、不换边界算法、不补demo。
0.03m是本构造的固定分段约定，不把抬高解释成抓稳，也不拿它评判任务成功。

接近段的参考frame为该物体root body，运送段为目的plate或stove root body。
region与其所属body不可混同：stove body仅用于刚体坐标变换，成功仍读官方cook_region谓词。
对每段，读取其源起点的参考`T_O`和各控制步实际OSC目标`T_E`；在新轨迹段开始读`T'_O`一次，
设目标`T'_E(k)=T'_O T_O^(-1) T_E(k)`。整个episode始终固定同一源，不逐段换源、按新初态挑源或失败回退。

必须沿实际OSC_POSE、20Hz、delta/fixed接口还原目标和逆变换动作，使用已有scale/rotation定义。
installed robosuite的旋转为左乘`R_delta R_eef`；全零旋转命令会保留上一goal_ori，
不能每个源时刻清空controller后假设目标等于当前姿态。源读取须保留该最小控制历史。
实际`set_goal`/`scale_action`与末端坐标所属body/site由已有owner核定，不用body质心冒充EEF。
新query在每步按自己的当前末端计算delta动作，正常限幅，原源gripper命令随路点播放。
每段前固定10步pose插值连接（平移线性/旋转SLERP），夹爪沿上一段末命令，首段沿源首命令；
连接步也是实际执行数据，不能只存变换后理想目标当动作。action noise固定0，不扫插值长度或控制增益。

### 5.2 精确尝试范围和输出

每task的官方init state40–43，各与两条固定源交叉，**最多16次**；次序按task、state、demo升序。
需确认query初始物理场景区别于对应源（报告目标物体位姿差，不能只凭索引/文件名）；重复初态不替换。
复用canonical资产与环境初始化，dummy settling10、horizon520、官方成功即停；不做原示范完整重放作为额外episode。
段路点耗尽仍未成功则当失败结束，不额外hold/重抓/重启；全部尝试包括失败都记录。
CPU源状态恢复只做kinematics/谓词/目标提取，不推进源环境，不生成新数据池；只有上述16次query允许env.step。

新query存T个实际环境动作、T+1 proprio/EEF/body/BDDL谓词及控制目标/段索引；自己的双相机RGB在step0/5/10…及最后保存，
render256、canonical 180度方向，保存原始图像方向说明以免再次rotate。所有失败同样保留这些证据。
本query图像在动作执行**之前**保存，动作chunk起点offset0；原HDF5 obs[i]与actions[i+1…]的历史offset1不能直接套用。
不在本批生成FM缓存，保存动作与时间索引供后续原始核验。合法教学仍复用原视频，不把新query当成teacher重新循环生成。
manifest写出源demo、源边界、对象/目标reference、query init、真实控制步数、每段开始的变换、成功/失败原因和实际成本。
source索引及privileged字段必须与未来合法Writer输入明确隔离；训练数据接口只可在loss側读取query动作/state。

### 5.3 投入上限和停止线

实现及CPU核对预计45–75分钟，运行/完整核验预计10–25分钟，总预期60–100分钟、硬复核上限120分钟。
运行估计以旧24,480物理步约435秒的CPU短段重放为粗基线：16×520仅8,320控制步；新渲染、加载、源恢复会增加成本，
所以它只是计划区间，不称实测吞吐，也不据此估算后继神经训练。时间上限到达保存现有结果、停止新增尝试并回报。
最多1张渲染GPU，完整占用硬限0.5 GPU-hour（含加载/失败/等待）；若实际纯CPU则如实计0，不能把EGL占卡叫CPU。
先做CPU source提取/小型算子核对，再取得渲染设备；不空占卡等待编码。GPU启动前live检查两节点及全项目总卡数。

新root仅为`/data1/user/ymdai/ember_runs/demonstration_transfer_engineering_20260927`，原件总峰值2GiB，
开发+冻结代码768MiB。16条最长query的stride5双RGB未压缩约0.62GiB，额外动作/state/目标及小型日志远小于剩余预算；
仍须由执行者按实际schema在strg01核data1独立quota/共享容量和新增峰值后创建，复用canonical环境/模型资产，不安装MimicGen。
只保留一个有界入口和必要原件，不另建通用轨迹平台或新评测器；按架构技能明确owner和退役触发。

先完成最小控制语义/SE(3)的CPU一致性核对、提交并推送，环境运行从clean detached commit执行。
语义接口/有限性/时间索引等工程错误可在本批范围修正并留失败账；实际抓取、释放或任务失败不是修改分段/控制器的许可证。
不会为了获得16个成功而替换源/初态或追加重试。无新checksum、逐bit要求或无关完整测试。

本具体构造进入下一设计判断至少需要：四个参考都能迁移到两个以上不同初态，且每task至少两个固定init同时被两个源成功覆盖。
这是可做交叉对应比较的最低覆盖要求，不是泛化/方法资格；必须同时列完整16行、每参考成功/失败及时间成本。
未达则关闭这个固定构造，不追逐lift阈值、边界、源筛选、轨迹长度或插值变体。
达成后仍不自动扩建数据：主讨论用实测成本与源失败情况裁决能否支持足够的non-held任务/参考覆盖，
直接进入完整方法比较的准备或关闭，不拆成无尽“先把每个模块做强”的课程。

## 6. 交付、所有权和后继

执行者复用其可用隔离开发树，从派发所列main开始，分支`codex/demonstration-transfer`；不修改main文档或旧冻结树。
主讨论拥有本设计/当前状态与科学裁决；执行者拥有本批数据构造入口、资源账与原件完整性。
保留实际launch命令/环境、冻结commit、固定源与init合同、逐次结果、失败日志、CPU核对及完整资源计费，结束清理自身进程。
不要把source/demo时点恢复写成旧训练/评测重跑，也不要把新轨迹成功称为LoRA收益。
完成后只有一条Queue给主讨论，不能逐实现/源读取/渲染/分析分别通知，不安排本任务的自通知。

本批不实现β/Writer、加载模型、计算梯度、run bank、对held评测、扩充教师或运行RL。
后继正式数据及模型实验必须据实测结果另冻预算、采样、学习节点与单checkpoint比较；本页未填的运行参数不构成授权。
科学关闭时入口从活动树退役，Git与原件保留；若进入完整方法，则集成唯一数据owner，不并存旧/新fallback。

## 7. 原件验收：14条真实成功保留，共同物理初态尚未通过

执行commit `29c35f2619f2734fb4ec1f3ce57f55067a53868f`，四源边界分别为34/d0=(59,116,194)、
34/d1=(64,129,217)、38/d0=(130,184,369)、38/d1=(134,191,363)。16条全部执行，14成功、两条38/d1路点耗尽。
主讨论独立读取全部16份NPZ：T/T+1、有限动作/几何、双RGB时点、成功首次终止与最终官方谓词一致；
总6,079控制步、实际渲染147秒/0.040833GPUh，原件约156MiB。未训练或运行神经模型。

**共同初态覆盖结论不予验收**：task34四对的全部初始body/EEF/gripper/双RGB一致；
task38两示范在同state40/41/42/43的stove root初位分别相差7.7099/18.6644/10.7054/21.4734毫米，
双RGB初帧平均绝对差3.5361/4.9269/3.7856/6.1390像素。机器人和壶初位一致不足以代表完整场景相同。
两条失败的pot2 On从未满足是真实行程结果，但不能用它们归因参考差异，或宣称两个参考具有共同物理初态覆盖。
主讨论复核原件为`.codex/tmp/demonstration_transfer_main_review_20260927/raw_review.json`。

代码层已确认：LIBERO `set_init_state`只恢复sim state；fixture位置存于`sim.model.body_pos/body_quat`，
不在该state向量中。此次每臂虽执行相同seed/reset，实际fixture仍不同；随机序列差异的更深来源尚未定位。
这是初始化配对合同的工程缺口，不是负结果后的控制器修补。保留全部旧原件、14/16逐条事实和旧执行资源账，
不修改completion原文以掩盖缺口，不将此结果标成Writer或配对机制成功。

### 7.1 唯一纠正范围

仍只由指定执行者负责。先以CPU核对当前入口的初始化恢复，明确完整物理场景与controller/gripper的起点，
在数据构造owner中以固定scene快照或等效机制保证同(task,state)双参考共用同一初态。
不能仅再次调用seed后假定通过；实际fixture/body、EEF、gripper及官方predicate须从保存值核对。
允许最多4次无渲染初始化核对，每次仅既定10步dummy（合计<=40步），不运行控制轨迹或模型。
不修改第三方安装、通用evaluator、旧正式数据或旧冻结树；若修正需跨出数据构造owner则先回报具体原因。
所有源分段、目标变换、连接10步、gripper、限幅、horizon与成功判定保持§5原定值。

CPU复现和修正后，冻结新clean pushed commit，仅重新执行受配对缺口影响的task38全部8行：
state40–43×demo0/1。**不是只重试两个失败**；原task34八行原件直接复用，不重新运行，原四源提取结果只读复用。
新的8行都保留，不按结果选择旧/新某臂。禁止扩大初态/示范、调整controller/分段或失败补跑。
重建起点必须在参考c选择之前冻结；两臂的完整初始scene证据要落盘，不能只存相同state编号或8维proprio。
本批8行可用性由初始化配对与完整执行共同决定，行为失败不改工程接口或重复执行。

本纠正预期CPU定位/修正/核对15–25分钟、运行/原件核验3–8分钟，总20–35分钟、上限45分钟。
最多1渲染GPU，完整硬限0.15GPUh；按刚完成的8条38约79秒循环/整批147秒加载计费作规划，留出加载和验证余量。
新根为旧study下`paired_initialization_repair/`，新增原件峰值256MiB，study总仍<=2GiB；
新增冻结代码<=256MiB、开发和全部相关冻结树合计<=1GiB。创建前刷新data1独立quota/共享容量并复用资产，launch前live检查双节点。
旧16行+新8行是24次实际尝试，分两批如实计数，不把新8行藏进原16次上限内；累计GPU账单单列。

### 7.2 纠正后的裁决

原task34四个共同成功初态继续保留；task38只用修正后的8行重算四参考/共同初态覆盖门槛。
如果配对有效而覆盖未达，按原原则关闭固定构造；如果配对仍无法在本预算内落实，报告工程未完成，不无限追加修复。
即使通过也只支持数据构造，不能自动扩建数据、实施Writer、跑held或RL。
一条完整完成/实质阻塞Queue给主讨论；主讨论负责源码集成、数据裁决与下一方法判断。

### 7.3 独立验收与源码集成（已完成）

修正commit81dc1e45冻结后，8条新增轨迹7成功、3,829行为步，state40/41/42两参考均成功；
43/demo1路点耗尽失败保留。主讨论逐一读取8份NPZ及4份scene快照，保存的全部任务body/EEF/夹爪/谓词初值和
双RGB首帧成对一致，快照与轨迹起点一致；T/T+1、有限限幅、stride5+末帧和首次成功终止均通过。
完整model body pose、sim state、controller goal的行为前一致性另由已审代码中的强制检查与exit0支持，
没有把仅有任务body的轨迹数组冒称逐臂保存了全部模型属性。控制/分段相对29c35f26没有改变。
原task34有4个共同成功init，新task38有3个，最小覆盖通过；旧14/16及新7/8独立记录，实际24次/9,908步。
新增87秒/.0241667GPUh，两批累计.065GPUh；study247,910,543字节、相关代码775,867,049字节，均在上限内。
主讨论原件`.codex/tmp/demonstration_transfer_main_review_20260927/pairing_repair_acceptance.json`。
源4条未重提取、旧16条未覆盖，主讨论未运行环境/模型；只跑两个纯CPU算子/恢复检查，不重复来源恢复测试。

源码以merge集成到main，保留原两次冻结commit和结果。一个数据构造owner复用canonical资产、谓词、官方OSC及存档；
516行源+85行测试是本次实际增长，不把其称为通用教师平台。结构扫描提示_extract_one的78行/complexity31和
_run_one的113行/complexity25；前者临时保留为同一源的连续控制历史恢复与边界核对，避免在本次验收中改变已运行的物理语义。
这是agent-owned的有界内聚例外，不是要求Owner批准。后继统一构造时须拆出纯边界判定、替换旧固定采集入口，不能继续增加特例。
测试移至现有tests/engineering，避免向已拥挤的tests顶层继续加文件；不改变运行源码或旧冻结树。
移动后扫描仅余_extract_one的上述hard信号，原始工具状态仍为block；本次采用已说明的内聚例外，不能报告扫描全绿。
原source/episodes/paired CLI只服务已结束的固定批，后继实现应在同owner替换，不能作为可再次启动的备用实验。

## 8. 后继唯一有界批：完整对应学习所需的训练任务支持

任务名`demonstration_transfer_training_support_20260927`，仅由原指定执行者实施。
本次把已通过的移动物相对目标构造用于一组事前固定的训练任务，目的为一次决定它是否能支撑完整P/I学习比较。
不是再在34/38上训练小模型，也不要求各个机械动作模块逐关通过。没有模型、Writer、LoRA、held评测或RL授权。
机器规格为`configs/demonstration_transfer_v1/training_support_spec.json`；派发状态以progress为准。

### 8.1 固定支持和数据边界

coverage_v1原36训练任务中，预先选定剩余目标为刚性移动物On/In的27项：
`[0,1,2,4,5,7,12,13,14,15,17,19,21,22,28,29,34,35,36,37,38,55,56,95,96,97,101]`。
组成是Spatial6、Object6、Goal4、Long5、额外non-held LIBERO-90六项。
排除依据来自任务规格而非采集结果：20/32/43/51/62/73有不同的开关操作，25为推盘，42/64为抽屉内放置，
后两项当前虽已打开抽屉，其关节目标frame也超出本次固定刚性root参考约定；不追加相关控制分支。
这些排除只约束**新增对应数据**，不更改原36任务训练许可、目标40任务划分或最终评测覆盖。

每task固定原教学demo0–3，查询使用官方init44–47，整组4×4交叉，最多432次新行为。
统一采用未采过的44–47，避免把只保存任务body/EEF而无完整sim快照的旧34查询混入四参考共同scene声明。
原34/38的16条有效工程查询和8条旧未配对查询都保留，但不混入本轮学习数据；不重跑旧case。
原四条源提取数组/metadata只读复用，其余104条训练源允许读取action、sim state、XML、BDDL，**源env.step=0**。
原示范仍只提供action-hidden RGB和exact L给未来Writer；所有几何/动作/阶段/成功/源索引留在数据构造及loss侧。
没有新示范筛选：某固定源不可用就记录，不换demo4或其它init补齐。

### 8.2 一种统一分段与同一实际控制算子

从官方goal中取尚未在init中成立的On/In移动物，要求每物体恰有一个目标；其它goal只能是init已成立的守护条件。
目标为对象时用其root；为region时用官方region的target root，包含实际arena body。使用官方对象/fixture registry，
或精确的arena body名；禁止从文件名猜角色、逐task补alias、任意换nearest site或重新拟合控制frame。
查不到精确frame或存在本合同不支持的移动关节目标就登记源不适用，不添加特例。

各移动物仍以body原点相对源初位首次抬高0.03m的l_i分段；按l_i升序确定**该条示范自己的顺序**，不预设goal列表顺序。
对非最后物体，r_i仍为l_i之后目标官方谓词为真且源gripper<0的首时点；要求
`0 < l_1 <= r_1 < l_2 <= r_2 < ... < l_m < N`，不得并列或交叉，初始移动目标不能已成立。
从`b_1=0`、`b_i=r_(i-1)+1`形成接近段`[b_i,l_i)`和运送段；非末段为`[l_i,r_i+1)`，末段为`[l_m,N)`。
每段必须非空；单物体即两段，多物体仍只重复同一对操作。与旧四源一致时直接沿用其已核数组/边界，不重提取。
源缺lift、释放边界不成立、操作交织或goal不在本范围，均为该统一构造的不可用事实，不调阈值/边界追结果。

段起点一次相对frame变换、20Hz真实OSC目标/左乘旋转、10步平移/SLERP连接、gripper历史、action noise0与限幅沿§5。
不同源不得逐段拼接；每次查询只固定一条完整原教学。成功即停，路点耗尽即失败，不hold、回抓、重启或行为重试。
query cap为Spatial220/Object280/Goal300/Long520；额外train LIBERO-90本构造cap520，不改任何held评测合同。
实际动作仍逐步由自己的EEF反馈计算并保存，禁止直接保存理想变换目标冒充执行标签。

### 8.3 场景配对、尝试账和输出

CPU先核所有固定源，仅恢复保存状态、不推进源环境。一个task只要有一条固定源结构不适用，该task不进入采集，
四源情况和全部16个未执行case原因都保留；不能挑其中两条容易源另开小组。
若CPU源阶段已使本批支持task数量或分组下界不可能满足，直接关闭构造，不再占GPU采集其它case。
对其余task，先于任一参考的行为执行冻结四个init的完整model body pose、post-dummy sim state和controller起点。
每个case按§7修正过的同一路径恢复、刷新观测，行为前核对scene、自己的状态/谓词和双RGB。
每case仍有固定dummy10，不改变scene后追加settling或根据source重新放置对象。配对工程错误停止该批并回报，不静默换编号。
所有预定case按task/state/demo固定顺序尝试，含失败均存T动作/T+1状态/末端/任务body/目标谓词、目标与段索引、step0/5/末帧双RGB。
RGB仍为256/canonical180，query offset0；不生成FM缓存或复制原教学HDF5，不改变原source归一化。

逐task计算固定四源的共同成功init集合U*。只有四源结构有效且`|U*|>=2`的task进入未来可用支持表，
其训练查询仅为四源×U*的完整等权交叉；其它成功和失败仍保留原件，不偷换成额外训练集。
这只消除了该固定组内的成功选择关联，不能代表所有初态、参考、任意状态反馈或视频收益。
本次投入条件为至少20个支持task，且上述五组各至少2项；这是避免再次以两task局部拟合代替完整学习的资源裁决条件，
不是有限学习充分性定理。未达则关闭这个固定构造，不追加demo/init、缩小四源组、任务专用controller或下一轮支持扫描。
即使通过，也只交付有完整尝试账的训练数据支持；主讨论另冻完整P/I/相同数据MT及profile预算，执行者不自动开模型。

### 8.4 时间、GPU和存储上限

108条源中104条新提取，共16,220个保存状态；数字来自已封存manifest长度，不是新的数据扫描。
全部432case在已知长度+每段10步连接且按suite截断下，控制步上界80,360；RGB最多16,664组双帧，
未压缩uint8上界6.103GiB。首批含加载6,079步/147秒外推约32.4分钟，只作采集量级依据，新task资产加载/重置会增加时间。
预计实现及源提取45–75分钟，采集35–60分钟，原件核验10–20分钟，总90–155分钟；**硬复核上限180分钟**。
最多1张渲染GPU，完整硬限1.5GPUh，含加载、失败和等待；源提取/实现不占GPU，不凑卡或开第二卡。
CPU阶段完毕后再取得渲染卡；GPU前live检查双节点及项目总量，达到任一硬限停止新增尝试并完整回报。

新root为`/data1/user/ymdai/ember_runs/demonstration_transfer_training_support_20260927`，新原件峰值8GiB，
包含全部source、scene、成功/失败、临时压缩、日志和metadata；不得依赖平均压缩率越过预算。
新冻结代码<=256MiB，开发+三次相关冻结树合计<=1.25GiB；原两批study只读保留，source采用只读链接/manifest而非复制。
执行者创建前刷新strg01 data1独立quota、当前目录用量和共享容量，估计其自己产生的临时文件后执行；不改用data0。
预算依据见`.codex/tmp/demonstration_transfer_main_review_20260927/training_support_budget_basis.json`。

### 8.5 一次实施、交付及退役边界

复用原隔离开发树和分支，从本次派发main集成；不得修改main或已冻结树。保留单一数据构造owner，
用配置表达task/source/init与统一角色，而非29/27个特例、平行编号模块或新的通用仿真平台。
替换已结束source/episodes/paired固定CLI，旧行为由原29c35f26/81dc1e45和immutable artifacts保留；
拆出纯边界判定以收敛原复杂提取函数，复用既有scene/SE(3)/OSC/捕获，保留少量真正验证这些接口的测试。
不重复运行原四源恢复测试、旧16/8行为或旧学习；若有新的工程接口错误可在本合同内修正并记录，物理失败不能调控制法。
CPU完成后clean pushed detached冻结再采集；完整交付只发一次完成/实质阻塞Queue，不逐task、阶段或定时回报。
报告源码diff/结构、108源逐条情况、27task完整支持/未执行原因、所有432计划case及实际尝试数、共同覆盖、成本和资源退出。
本批没有神经训练或正式性能分数。构造关闭则必要入口由Git保留并退役；构造支持完整方法时仍保留一个canonical数据owner。

### 8.6 完成后独立验收（2026-09-27）

108固定源中100新源兼容、4旧源只读复用；101的四条封存XML与官方BDDL对象注册不同，未改资产，16查询均未执行。
416次实际行为共71,520步，367成功/48路点耗尽/1到horizon；不是正式policy分数。主讨论重新读取全部416份NPZ与104份scene，
核定T/T+1、有限限幅动作、RGB时点和首次成功终止；104组保存body位姿/EEF/夹爪/谓词/双RGB初态严格一致。
完整model/sim/controller每臂一致由已审实际行为前assert及运行退出支持，未声称逐臂另存了完整sim快照。

共同四源至少2个成功init的任务恰20个：Spatial5/Object4/Goal3/Long4/meta90四项，共296条完整等权交叉。
固定投入条件通过；5/12/14/22/36/56共同支持不足，101源结构不适用，不补采、不替换参考，不缩减原36任务学习。
首冻结1988c7eb执行144条后，在15/44/demo0行为前触发旧的“被搬物初位必须不同”断言；f8dd3770删除这个§8未规定的门槛，
保留全body差值，再完成272条。失败前0动作，全部物理失败未重试，分段/目标变换/OSC/连接/horizon未改。
独立复算该case源/query保存body最大差17.5669mm，所有416行该差值至少4.8903mm；
但有16个case-物体项的被搬物距离<=0.1mm。因此不把“新场景”夸大为每个目标物体都换位置或已学会相对控制。

完整计费690+1375秒=.573611GPUh，最大1卡；观察窗口约07:25–08:36 UTC，低于预计90–155分钟。
数据2,470,234,402B<8GiB，开发+四冻树1,292,068,469B<1.25GiB（十进制GB不可混当GiB）。
主讨论3项纯CPU测试通过，结构guard REVIEW无hard；+523/-277、三个文件，单一CLI与纯source分段owner，旧固定入口退役。
source提取和连续OSC执行保留为各自完整职责，非学习代码；本构造关闭时退役，历史冻结树/原件保留。
源码f8dd3770由main合并4aeaf5b5；独立复核为`.codex/tmp/demonstration_transfer_main_review_20260927/training_support_acceptance.json`。
原件根`/data1/user/ymdai/ember_runs/demonstration_transfer_training_support_20260927`。

## 9. 完整学习取舍（正式学习仍须另冻）

新增对应数据仅覆盖一种操作构造，不能因此把36任务缩成采集成功子集，也不能让失去源支持的任务从保持要求中消失。
后继完整目标选择为：原36任务等权；有新支持的task由原始跨episode查询和新对应查询各占一半，
没有新支持的task全部保留原始跨episode查询。两类都使用同一完整LoRA、同样普通全horizon FM，
不新增端点/前5步辅助、分阶段冻结、标签头或task-local优化。
二分权重是本次完整方法的固定投入选择，旨在保留原任务/状态支持同时给予新对应足够信用，**不是保持保证或新扫描轴**。
保留已有原始数据不证明优化不会遗忘；相邻闭环能力仍须实际判断。

具体地，令S为本次构造最终合格task集合，`L_old,t`为既有独立跨episode FM，`L_new,t^P/I`为新查询上的两种对应，
则拟议完整风险为

```text
L_P/I = (1/36) sum_t [
    t in S : (1/2) L_old,t + (1/2) L_new,t^P/I ;
    t not in S : L_old,t
].
```

这是数据联合关系的干预，架构/β/初始化/训练时标在P/I保持一致。
旧查询事件沿原36任务学习合同使用demo0–45的46条，且action episode与该教学不同；新查询事件的教学条件只取固定demo0–3。
先前本段写“原50条”不准确：原配置的46–49用于训练侧诊断；在任何模型运行前更正，不扩大旧查询动作池。
这不改变正式held评测的每task50条无放回视频要求，环境init44–47也不是teacher demo44–47。
新query自己的RGB/state/action、源参考/共同init采样、padding、FM noise/time在P/I逐事件一致，仅改变教学视频与新query的对应。
不把源数组、索引、段/时钟、生成成败或数据类型送入Writer；旧事件和新事件也不使用不同部署头。
教学和query图像分别按已有预处理消费，自己的state仍按冻结source统计进执行prompt；不重算归一化。
新query RGB在动作前，offset0，末尾无后续动作的terminal RGB只作证据，不作训练query；
旧HDF5仍offset1。动作不足50步沿既有末动作重复规则，不趁换数据改变FM horizon/padding目标。

### 9.1 有限训练里怎样做到真实的独立参照

只把两个视频固定互换是可逆编码，已被§4排除。四参考时同样不能固定一个permutation当独立臂。
对每task，每16次新查询事件使用全部16个`(query_source c, condition_reference d)`格各一次，以独立登记seed打乱事件顺序。
P输入V_c，I输入V_d；因此每个完整块两臂都恰好使用每视频4次，query源也各4次，I的源×条件表为精确均匀乘积。
每格的28-query块（若沿现有28约定）仅依赖task/c和独立query RNG，从共同init/可用RGB时点采样；
FM随机量也由独立固定事件流产生，不能根据d选择初态、帧、noise或time。P/I复用这些实际查询和随机量。
这建立源身份的精确平衡及随机query层的独立构造；**不声称每一条独特query都枚举了四个视频**，
也不为形式一致让两臂重复同一noise/query四次。有限样本的不确定性仍保留。
旧事件的条件与查询完全共用，不参与新对应的置换。视频内部顺序从不因P/I改变，shuffled/reversed仍封存到最终controls。

若沿用36task、每宏步4task、每条件28query，并在每task两次访问中各安排一次旧/新事件，
一个新对应平衡块需32次task访问，即288宏更新/32,256查询；这是从完整对照周期推得的数，
不是恢复旧270/450时点或当前训练授权。task轮次应交错旧/新事件以免全批周期性切换目标，权重和边际保持。
正式批量、评测节点和总预算仍须由真实完整模型profile与数据实际支持另冻，不能直接用条件速度/Reader吞吐担保。

### 9.2 比较结果究竟淘汰什么

§37.8的纯新数据风险恒等式不能直接套到上述混合最优风险：旧/新视频池不同，事件类型未作为模型输入，
一个共享Writer还要同时拟合两类分布。P/I的因果解释来自逐事件相同的query、完整块相同的视频边际及同一学习程序，
不把一个条件期望恒等式冒称实际网络的增益数值或全局收敛保证。
同数据MT使用相同任务/查询分布和忠实强训练方式，不能为了对齐Writer而削弱其已有效的训练程序；
它与P/I未必逐优化步相同，故单个分差不定位某模块。已有强MT155仍为参照，不能由新MT变弱降低接受标准。

P相对I的完整闭环优势才能支持“新对应关系帮助合法条件编译”，还必须考察未见task、另一正确视频和相邻保持；
仅比I好但仍弱于强MT、仅降低训练FM或只学会新查询轨迹，都不能接受为EMBER方法。
若P/I/同数据MT共同改善而P无额外收益，更支持数据质量/覆盖解释；若两Writer共同弱或继续大量丢能力，
该完整混合方法关闭，不自动扫描二分权重、扩到更多参考/初态、冻结β或增加辅助头。
这段明确完整方法决策，不将数据构造通过当作模型收益。当前有界工程权限另见§10；没有正式训练授权。

### 9.3 真实支持内容怎样限定解释

主讨论在工程派发后仅从既存source分段和296查询的`step_kind/rgb_steps`核定监督内容；没有新数据/模型/环境。
20支持task的80固定示范中，只有34/37/38涉及两个尚需搬运的物体，另外17task为一个；
**每task四源的物体搬运次序均相同**。这不说明视频动作完全一致或所有历史没有顺序分歧，但本批不提供这种分歧的正例。
因此P胜过I不能被归因为修复“示范之间顺序切换”；候选依据是具体几何/接近/运送参考与新初态功能查询的对应。
分段只是特权构造元数据，不能由它断言这些参考已在合法RGB特征中可识别。

按任务/源/共同init等权、每轨迹动作前RGB时点均匀的预定口径，新query当前处于连接段的比例为14.261%，
恰在共同初态k=0的比例为3.342%。50-horizon标签时间格中，8.580%为真实连接动作，75.081%为变换后路点的真实执行，
16.338%为末动作重复。这里只计算loss中目标时间格的采样权重，**不是loss值、梯度大小或行为贡献**；
不能据此断言连接信用可忽略，亦不能把大部分目标说成仅由人工连接构成。
同一init交叉也不等于后续query状态仍相同：参考改变occupancy，而自己的状态/带噪动作可能已经包含参考信息。
P/I共用实际query才能保留这一事实；新对应未必带来额外可用条件信息，更未保证其能迁移到未见task。

这收紧后继解释，不改变已派§10、删连接步、扫描padding/权重或另开局部探针。
正式闭环若P/I/同数据MT都提高而P没有额外收益，保留数据解释；若仅P训练拟合好而held/保持没有改善，不能接受为EMBER方法。
原件`.codex/tmp/demonstration_transfer_main_review_20260927/learning_support_semantics.json`。

## 10. 完整LoRA对应学习的有界工程与真实profile

任务`demonstration_transfer_learning_engineering_20260927`，唯一执行者仍为原实验session。
本批一次接通§2与§9的完整方法，并测明真实成本；不再追加局部表示/标签探针，不扩大数据，不选模型。
机器规格`configs/demonstration_transfer_v1/learning_engineering_spec.json`；只有progress登记和真实派发后执行。

### 10.1 固定函数类、信息墙与训练算子

复用`CompleteLoRAWriter`、原生读取/Core/Procedure/FactorHeads、真实FM和ECP公共owner，不恢复Reader/条件速度或旧训练CLI。
source仍为`pi05_source_aligned_seed7_1k_20260915/step_00001000`，归一化/分词器/图像处理按既有source authority。
physical policy仅注入完整rank144/alpha144的38-target接口，所有source/physical参数冻结；教学读取时它保持identity，
不将公共β或本次输出装回教学读取。公共rank128参数与完整生成rank16输出在因子维拼接，不能同rank相加。
可由同一deterministic identity144模板分出公共128和条件16；Writer要求的rank16模板保留，完整FM只接144合同。
初始所有B为零；公共与条件参数均fresh、同一AdamW联合更新。全局/Writer seed7，identity模板沿LoRA合同seed20260721；
lr3e-4、betas(.9,.95)、eps1e-8、wd1e-4、clip1，
前150次线性warmup；本批只到4，不据此预定正式长程scheduler。工程权重不得成为正式训练初值。

每次宏更新4个不同task×28个真实query，任务预权重1/4；多卡SUM已加权梯度，不再除world size。
28个query全部普通FM、50完整horizon、前7真实action维；不拆旧21+7辅助，不启用endpoint/前5步或额外loss。
允许复用“生成→完整LoRA真实cotangent→同次参数重放VJP”的现有精确一阶链式求导以节省显存；
cotangent必须来自本次最终144输出且回传公共与条件两边，不用teacher更新标签、跨更新缓存或替代梯度。
两臂各参数组记录有限梯度/实际更新；初始零B造成首步上游梯度为零属预期，第2步起应有实际条件链信用。
不以梯度非零、loss变化或工程episode成功宣称学习收益。

### 10.2 同一查询分布及可核查的P/I事件

原36任务等权，固定支持集合为§8.6的20项；其旧/新事件交替各半，其余16项全旧。
事件按task轮次交错，使完整两轮每task旧/新数准确，不按耗时/轨迹数重加权。
新查询只读296交叉：先等权源c及共同init，再均匀选该轨迹合法已保存的动作前RGB时点k<T；
取自身EEF位置、`quat2axisangle`和夹爪qpos构成8维state，动作从k开始取50步，不足沿末动作重复。
NPZ双RGB已经canonical180，不能再次翻转；原HDF5仍由既有owner按offset1处理。禁止把geometry/阶段/目标/成功/source编号
送入Writer或policy；读取器只向Writer提供原教学RGB/L，向policy提供自身RGB/state及loss动作。

每支持task的16次新事件枚举四源×四条件全部格，种子乱序；P用V_c、I用V_d。
query/noise/time随机流不依赖d，P/I逐事件完全共用；新源、视频边际在完整块一致，源×条件为乘积表。
CPU仅用manifest/封存查询检查完整288更新事件表：36task各32次，20支持task各16新/16旧；
总32,256query，其中新8,960、旧23,296。原始和新查询事件分别保留时间偏移，不混用视频与环境init编号。
不为了精确平衡强制一个独特连续query四次重复，不把有限表平衡当作所有x上的统计独立或收益保证。
随机种子及物理分片与事件游标完整保存；CPU事件表不要求运行288模型更新。

### 10.3 仅允许的模型计算与实际接口case

1. P/I各fresh4；每臂保存2和4的完整ECP，再从各自2恢复到4，独立输出root。
   合计12个实际宏更新、1,344训练query。核前两行历史及3/4事件、LR、optimizer/sampler/rank RNG恢复；不追逐逐bit。
2. P fresh4后一次最长合法训练视频profile：manifest固定global38/demo36，517原帧、stride5加末帧105帧；
   28条旧跨episode真实FM查询、一次完整VJP，无optimizer更新，不新增数据或挑更容易视频。
   测读取/生成/FM/重放和总耗时、峰值allocated/reserved、稳态query/s，不能只报最短视频吞吐。
3. P fresh4生成两套分别对应两个独立case的完整LoRA：train global2/init0/demo46和global38/init0/demo46。
   每case rollout前一次生成、真实canonical双相机/8state/10flow/前5action/dummy10/官方horizon、成功即停。
   复用现有official episode及被动捕获owner；两条均full双RGB、T动作/T+1几何/谓词与真实初态留存。
   这两条只核最终38-target接口，不组成收益面板、不配对打分或据结果改方法；不跑Source/MT/额外对照episode。

两臂在同节点world2任务分片，最多2张物理卡，物理卡选择按live资源；不跨节点拼训练或等待凑卡。
若当前只能合理使用1张，工程可在world1运行全部fresh/resume并据实计成本；本批同一臂恢复不换拓扑。
起始query microbatch28、frame chunk8；仅实际OOM时可降物理microbatch14/7或frame chunk4，事件/目标/噪声不变并记录失败。
BF16/TF32与既有activation checkpoint可用；不扫描dtype、rank、LR/seed、损失权重或更新次数追效果。
shape/finite/信息墙/合法视频/实际source冻结/恢复错误须停止并修正真正工程违约；物理case失败不构成bug。

### 10.4 时间、资源、存储和正式投入边界

预计实现/CPU75–120分钟，两卡模型计算15–30分钟wall，收尾10–20，总100–170分钟；
若采用world1，模型预计25–50分钟、总110–190分钟，统一210分钟复核上限。
完整GPU硬限1.0GPUh，含加载/失败/等待，峰值2卡；达到时间或资源限不加量，报告完成和缺项。
预算依据仅为先前Reader12更新/恢复/profile/6case完整.5126GPUh的量级，以及本批两臂均完整视频、
公共144执行与105帧最长输入的额外不确定性；不把旧15.53秒/宏步当本方法实测或正式训练预算。
原36任务数据、296查询和所有源资产只读复用。新study
`/data1/user/ymdai/ember_runs/demonstration_transfer_learning_engineering_20260927`峰值4GiB；
新冻树每份<=256MiB，必要工程修正累计新增代码<=512MiB，开发+现存相关冻结总计<=1.75GiB。
执行者创建前核strg01 data1独立quota及共享空间；不复制模型/HDF5、不建整库cache、所有新增data1。

通过后只给出实测成本、完整恢复/接口证据及P/I学习和同数据MT的后继估计；不自动开288/576、bank400、held/Test、controls或RL。
科学裁决依然要求完整closed-loop和相邻保持；工程非零梯度不能预定结果。无法在固定函数类/合法数据/合理预算内运行，
回报具体限制，不私自替换成小模型、低rank或缓存改造后冒称本候选。

### 10.5 一个活动学习owner与交付

复用原独占开发树，从派发main合入后实施；不写main，不改已冻结树或sealed原件。
现有`writer/training.py`已有813行和历史合同分支，本候选不继续往其中塞experiment/fallback。
新增候选学习代码放一个内聚子目录：数据/事件、公共+条件完整模型、训练/有界工程三个职责；CLI仅薄转发。
原生编码/FactorHeads、source/LoRA/FM、ECP、replay和official rollout复用当前owner，不复制旧训练框架或评测平台。
预期净增约900–1500行；超过1000时交付明确实际所有权和退役关系，不以机械拆文件解决guard。
不建立平行P/I模型，二者仅条件对应调度不同；同数据MT后继沿强训练配方，不在本批另造第三训练框架。
本方法关闭时退役私有模型/事件/运行入口；必要旧结果读取与原件保留。collector不再获采集权限。
先少量真正验证因子拼接/数据时点/对应事件/恢复的CPU检查；clean pushed detached后执行模型，保留完整预算和失败账。
一次完成/实质阻塞Queue交付精确commit/diff、执行项、训练与恢复原件、profile/case、成本/资源退出与缺项；不逐阶段自通知。

### 10.6 主讨论独立验收：部分工程成立，两项接口尚未完成

执行commit44cf63fa（基于0f38d17c，六文件+1000）保留在执行分支/原冻结树；尚未合入main。
主讨论读完全部新增实现，三项CPU测试通过，结构guard REVIEW、无hard。数据/组合/运行/接口为四个明确owner，
复用原生编码、FM、ECP与official rollout；没有复制旧813行trainer。源码集成待下节修正一并完成。
已核288事件全表及P/I fresh/resume记录、六个ECP的optimizer/scheduler/sampler/rank RNG游标与两条NPZ/PT。
新8,960/旧23,296计划query、12实际宏步/1,344query、恢复前缀及source_trainable=0成立。
517→105帧的28-query完整旧数据FM/VJP为18.594秒、峰值allocated20.746/reserved21.143GiB；
五段完整计费.206436GPUh、均exit0，原件约1.403GiB。两条行为的220/520动作、T/T+1和双RGB成立，均失败只作接口事实。
独立记录为`.codex/tmp/demonstration_transfer_main_review_20260927/learning_engineering_acceptance.json`。

**未通过完整工程准入，理由为实际合同缺口，而非科学负结果：**

1. `PairedEvents.event`使所有支持task在偶数visit全旧、奇数visit全新，连续9宏步的新事件数为0/20交替，
   没有落实§9.1避免整轮同步切换的交错安排。首4步与最长profile全部旧查询，故新增NPZ虽通过CPU时点核对，
   尚未在实际模型中消费。主讨论的原机器规格只写“交替”，未将task相位写清，应补成确定规则；不能由CPU表冒称新路径已运行。
2. `episode._run_cases`在第一条装入生成LoRA后，第二次compile前没有恢复物理source的identity。
   原生Action读取直接消费该expert投影，Meta hook不取消已安装LoRA。因此global38的教学读取不满足§10.1，
   撤回其独立冻结source生成的资格；不声称已量出影响大小或解释520步失败。训练FM的functional_call恢复原参数，
   不把这项顺序case问题泛化为训练污染；第一条global2、最长profile和恢复证据继续有效。

另作规格解释纠正：全部组从第3步活动，不是所有组第2步都应非零。FactorHead末层和Compiler Procedure调制都零初始化：
第1步先动head，第2步Core/调制得到信用，第3步Procedure/Action路径才打开。这由实际串联导数和原始梯度共同支持，
原§10.1“第2步起”过宽；不改初始化、不为提前非零重跑旧实验，也不由非零梯度宣称能力已获得。
这些裁决不改变§9完整科学方法；288/576、held、MT、controls和RL仍未获本批启动许可。

## 11. 唯一后继有界工程：混合查询实际学习及生成源隔离

任务`demonstration_transfer_learning_closure_20260927`，只派原指定执行者。沿用同一机器规格路径并升v2；
旧v1、44cf63fa冻结树和所有原件只读保留，不留两个可重新启动的活动工程CLI。
本批只补足§10.6的具体缺口，模型/目标/数据池/二分权重/优化超参全部不变；无新采集/正式训练/MT/held/RL。

### 11.1 固定修正及允许的模型工作

- 支持task按已冻结升序集合的零基ordinal定义`phase=ordinal%2`；当`(visit+phase)%2==1`取新事件。
  每9宏步恰10新/26旧，每task两visit仍一旧一新；16次新事件继续取全部4×4格，288总query及P/I配对保持。
  相位只用于训练调度，不进入任何模型输入；sampler schema保存相位并拒绝旧v1恢复。
  种子/任务顺序不因smoke结果改动。固定前缀的新事件为第1步task29、第2步35/4、第6步97；
  第3/4/5步没有新事件，因此只跑旧4步会错过上游全开后的新查询信用。
- **P/I各fresh6，另仅P从自己的新ECP2恢复到6**。保存各fresh2/6及P恢复6；合计16实际宏步/1,792query。
  不加载旧P/I工程权重、不另做I重复恢复。CPU全288事件核对相位、边际、合法时点；实际第6步新query必须经过
  自身RGB/state预处理→完整144普通FM→cotangent→同参数VJP→optimizer。记录实际事件/有限FM与梯度，不新增机制probe。
  核真实source_trainable=0、恢复前2行及3–6事件/LR/游标；低位差异接受，不扫描逐tensor一致。
- 每次教学compile前由同一运行owner保证物理source为完整identity；部署输出不得遗留到下一教学读取。
  用小型CPU回归覆盖“装入前一输出后再compile”的顺序，不改共享native算子或创建第二policy副本。
  GPU只用新P6顺序生成global2/demo46并安装（**不运行其环境**），随后按修正的正常入口从identity source生成
  global38/demo46，执行唯一global38/init0 train-only full双RGB canonical episode，horizon520。
  这是受影响接口的补全，不为成功重试，不重复第一条有效global2或最长profile；结果好坏不改方法。

### 11.2 预算、交付及主讨论并行工作

按首批12更新/恢复/长profile/两case总.206436GPUh，追加16更新/一次case预计.20–.30完整GPUh，
硬限**.40GPUh**、最多2卡同节点；GPU时含加载、失败、等待，原批成本另计。CPU/实现20–40分钟、模型10–20、
收尾10–15，总40–75分钟、90分钟复核上限。新数据FM代价仍有不确定性，硬限不因此放宽。
新root为原study下`closure`，新增峰值2GiB、全study峰值4GiB；只增一份<=256MiB冻结代码，
当前相关代码约1520MiB，预计峰值1776MiB<1.75GiB。执行者创建前核data1独立quota、容量及双节点live GPU。
若必要额外冻结会越界，先报告而非覆盖旧树；源资产/旧工程/原采集数据不复制、不删除。
分支复用codex/demonstration-transfer，先合入指定新main文档/规格，保留44cf历史；同一实现替换当前有界运行面。
CPU回归/结构检查后clean pushed detached运行，一次完整完成/实质阻塞Queue回主讨论，注明新增与累计成本和缺项。

主讨论同时准备完整P/I与忠实同数据MT的比较、预算和真实初态配对方案。MT的36task×16query宏步与Writer的4×28不同，
首批用28-query FM推得的“.85GPUh/288”只是局部计算下界，不能当MT的完整成本或公平训练预算。
本批不顺带运行MT/profile、400 bank/held或扩建评测平台；这些需要形成完整合同和成本依据后再投入。

### 11.3 补全已验收（2026-09-27）

399e2610相对44cf63fa四文件+194/-137，当前完整候选六文件净+1057；主讨论读完差异、四项CPU测试通过，
架构guard REVIEW无hard。新增职责仍为数据/唯一组合/训练/接口，Runtime统一在compile前恢复identity，旧profile-cases入口已移除。
主讨论从全部288事件重算每轮10新、36task各32visit和全部4×4格；从真实metrics重算16宏步/1,792query，
其中新252、旧1540。第6步P/I同task97/source0/query/flow seed，教学分别0/3，真实FM cotangent均约.00977；
实际代码将其送回同一完整144的公共与条件参数并更新。不以相近loss或非零梯度裁决对应收益。
P fresh2/6、I fresh2/6及P resume6五套ECP的optimizer/scheduler/sampler/rank RNG游标成立，前两日志行原样保留。
顺序接口实际先装global2输出，物理source差范数1.234047；第二次普通compile后为0，两次identity恢复成立。
唯一global38的520动作/521状态、104次双RGB重规划和T/T+1已独立核，失败仅为接口事实。未重复原global2环境或最长profile。
四段exit0，新增完整.236187GPUh，累计.442623；closure约1.141GiB/全study2.544GiB，相关代码1774MiB，均在限额内。
实际派发到完成约23分钟，低于40–75分钟预期；不把节省时间转成额外未登记模型工作。
独立原件`.codex/tmp/demonstration_transfer_main_review_20260927/learning_closure_acceptance.json`。
源码以merge合入main，冻结树/旧原件不变。§11工程缺口关闭，**未产生正式closed-loop或有益视频证据**；工程权重不作正式初值。

## 12. 完整比较的已确定口径与尚缺准入（主讨论准备，不增加§11执行范围）

### 12.1 同数据MT必须保持强训练程序

同数据是同任务、原动作池demo0–45、新296查询、source统计及目标分布，不是相同宏步的查询数。
Writer288为32,256query、每task896；忠实MT仍每步36task×16=576query，288步165,888query、300步172,800query。
拟定MT每支持task每步8旧/8新、其它task16旧，即每步旧416/新160；每task等权，20支持task中两类各半，
与§9的期望风险一致，但梯度估计的batch相关性及有限样本不同。新源/共同init/保存时点各自等权，不能把296轨迹的
所有帧平铺均匀而悄悄按轨迹长度加权；原episode/chunk抽样继续沿强MT的层次无放回原则。
MT自己的RGB/state和动作只走执行loss，exact language为唯一任务条件；无teacher视频、源参考ID/几何/阶段或Writer输入。
shared rank128、fresh identity/AdamW、150 warmup及1200绝对步衰减至1e-5的强配方保留，不为对齐4×28削弱任务覆盖。
P/I正式长期scheduler沿同一已核clamped函数；它不使两种宏步的曝光或优化轨迹相同。

已有旧MT用50原episode/选中300得到155，仍为强参照；新匹配池46是本候选已冻结数据边界，不能把新MT较低当作接受线下调。
新MT按新数据和真实闭环判断有效训练范围，不能把旧300直接称为新数据的最优点，也不能只以查询数相同强制提前停止。
最终需同时对照已有155和如实训练的新MT；P/I差异只识别这项训练对应干预，不单独证明部署视频必要或某模块已学好。
后者仍需有益绝对能力、另一正确视频、learned language/static参照、相邻保持，最终时间controls不反哺设计。

首批“.85GPUh/288”按112query宏步推得，不能用于576query MT。由同一2.653秒/28query线性折算为**4.366GPUh/288**；
这仍只是该FM调用的计算量参考，既不是新MT实测，也不是严格性能下界。旧真实单卡36×16 profile稳态148.668秒/步，
折算288为11.893GPUh，同样不担保新loader/packing/直接反传耗时。正式预算须测新数据上的完整576-query更新、保存与加载，
不能用更多卡隐藏GPU-hour，亦不先承诺一口气跑三条长训练。这里只读旧三行profile，没有运行或恢复旧MT。

### 12.2 唯一LoRA的物化与严格配对

38-target FP32完整144因子每condition为46,338,048B，重复400份仅张量就18,535,219,200B。
选择公共128仅存一次（41,189,376B）加每condition完整16因子（5,148,672B），400条件为2,100,658,176B；
evaluator加载时沿已测因子拼接生成唯一144，仍是`BβAβ+BvAv`，不是平均/选择LoRA或部署第二adapter。
bank manifest须绑定单checkpoint、精确video映射、source与完整rank/target合同；编译policy保持identity，执行policy只装完整结果。
接入复用现有official队列、BatchedLoRAInference和捕获owner，不复制退休velocity训练平台；该旧bank读接口保持只读。
上述只是精确tensor payload，实际预算另外计ECP、临时物化、轨迹、scene和代码，不能用它冒充全study峰值。

新P/I/MT及相邻节点必须共享在任何policy行为前冻结的canonical起点：官方init、seed7、dummy10，
保存全model body pose、post-dummy sim state、controller起点及首帧双RGB，并在各臂第一动作前恢复/核对。
已有collector的恢复语义可复用到一个有明确调用方的初始化owner，不让official evaluator长期依赖退休采集入口。
不从held示范的action/state生成scene，不按policy结果选择起点；raw t0字段和源scene引用保留。
仅同state ID/seed不足以取代这些配对证据；旧MT/Source没有同类scene原件，只作原有历史参照，不称与新臂完整物理配对。

### 12.3 下一完整投入的边界

§11通过后一次冻结完整学习/物化/strict400，而不是再拆表示、LoRA范数或局部loss关卡。
正式模型fresh，先按完整P/I交叉周期规划首节点288；这个数是有限对照覆盖，不是已证明充分学习的上限。
后继完整块、同数据MT节点、first400后的继续/停止线及GPU硬预算须在正式学习前一起明确；不能结果后任意挑时点。
实际还缺新数据MT完整更新成本，以及新bank/共享scene在官方调用链的有限工程证据；不把这些缺项写成已完成或据此启动GPU。
最小后继工程只应验证该完整比较需要的实际接口与成本，并接入同一活动实现，不增加科学probe、数据支持或模型变体。
算术原件：`.codex/tmp/demonstration_transfer_main_review_20260927/learning_comparison_budget_basis.json`。

## 13. 唯一后继：完整比较的成本与official接口准入

任务`demonstration_transfer_comparison_admission_20260927`，仍只派原指定执行者。
本批一次补齐同数据MT实际成本、紧凑bank和共享canonical scene的官方调用链；不再分成局部科学Gate。
同一机器规格`configs/demonstration_transfer_v1/learning_engineering_spec.json`升v3，取代已结束的v2活动入口。
§2/9的完整模型、数据、目标保持。没有正式P/I训练、held/Test、controls、RL、新采集或旧实验重跑授权。

### 13.1 忠实同数据MT的有限真实更新

MT为同一Source1000上的一套fresh自由rank128/alpha128、38-target A/B；无Writer/Meta/视频读取，source始终冻结。
每宏步36task×16query：支持20task各8旧/8新，其它16task各16旧，总576=416旧+160新。
原动作池demo0–45，offset1；新296交叉、offset0、自己的双RGB/8state、50horizon末动作重复，复用当前数据owner。
原查询沿强MT层次无放回：task内部old查询游标推进，按46 episode permutation取demo、再按各episode frame permutation取chunk；
支持task游标每步增8，其它增16。新查询每task每步四源各2条，源内共同init无放回循环，再在该轨迹合法RGB时点无放回循环。
各层permutation使用独立登记seed，不能平铺轨迹帧、按长度/成功率重加权或借query/source编号向policy输任务码。
原层次抽样seed20260723；新init/frame层次seed2026092704并分离层次tag；flow seed2026092705，模型/优化器seed7。
配置应保存这两类逻辑游标；query顺序/物理分片不改变每task16及两类权重。无需把MT query与Writer逐行强行相同。

普通真实FM仅50×7；AdamW seed7、lr3e-4、betas(.9,.95)、eps1e-8、wd1e-4、clip1、150 warmup与既有clamped scheduler不变。
允许复用已核`paired_functional_credit`到自由A/B的精确cotangent/VJP，其最后映射只是直接参数，等价于该真实FM的直接梯度；
不得用回归更新标签、伪梯度、低rank或缩小task batch代替强MT。公共参数属于唯一生成/执行LoRA，物理source不另留可训练副本。
复用当前学习loop、ECP、source/processor、FM及source_sft/control和层次抽样规则；可为实际第二消费者做内聚重构，
不复制Source-SFT或新建第三套训练框架。无MT视频/native假读取；P/I的函数和数值学习程序保持。

仅MT fresh3，存1/3 ECP，再从自己的1恢复至3：**5实际宏步/2,880query，新800/旧2,080**。
同节点world2优先，world1允许；物理query microbatch16，逻辑576不变，恢复锁原拓扑。每rank按task平衡分片，不绑定固定task卡。
测真实全宏步、数据/FM/反传/optimizer、checkpoint和完整占卡成本、进程峰值；完整source/LoRA trainable集合、finite、恢复游标须成立。
工程M3不可成为正式初值，不以其loss/后续接口成功选方法。
同时CPU将P/I事件流完整定义到两个288块：第一块与已核v2一致，第二块各task重排全部16源×条件格；
新query/flow仍与独立d无关，P/I完全共用。核576总64,512query（新17,920/旧46,592），但**不运行P/I任何新更新**。
不把扩展CPU事件表当作已授权576；正式节点/继续停止线由主讨论随后冻结。

### 13.2 同一bank与official调用链

只读新P6/I6工程ECP（399e2610）作bank接口权重；须显式核该兼容来源与同一模型/算子，不从它们续训。
固定train global2/38、init0–3、K1 correct；按既有video_schedule、seed7、50视频池、per-init ordinal映射。
P和I各8个合法视频条件、8个episode索引；M3仅一套shared128、同样8个episode索引，无视频值读取。
P/I每bank公共128存一次+每条件完整16，载入拼成唯一144；M为唯一shared128，不输入task ID或按task选权重。
复用同一新bank owner实现检查/物化/episode证据与BatchedLoRAInference，接入现有`scripts/evaluate_pi05.py`；
不复制队列、rollout、资源/恢复平台或退休velocity实现，不用手写`rollout_shard`调用冒充official入口接入。
bank须绑定精确checkpoint/source/scope/video映射、完整shape/finite及调用次数；只验证必需身份，不新增hash或逐tensor一致扫描。
报告实际bank总时长/LoRA每秒、加载开销与峰值，由此估计400物化；不从8个工程条件推出held结果。

### 13.3 真实配对起点与固定24条接口

先于任一policy行为冻结8个train canonical scene：官方init0–3、seed7、dummy10，无模型选点。
复用已核全model body pose、post-dummy sim state和controller/观测再生成语义，存自己的起点/双RGB。
每臂每条仍作官方初始化/dummy10，再恢复同一冻结起点；第一真实动作前核完整登记scene、EEF/gripper/谓词和双RGB。
不从teacher action/state生成起点，不额外settling，不使用held初始化/示范标签；配对失败作为工程错误停止，不能只靠state ID宣称配对。
初始化公共逻辑若移入`pi05_eval`，由collector和official两处实际消费，删除原重复实现；不得让正式evaluator反向依赖候选采集CLI。
原已冻结collector/数据不改、不重跑；本批新scene只服务这次训练侧接口及未来同实现的正式配对。

**P6/I6/M3各一次8行，共24条**。完整official预处理/10 flow/前5action/成功停/220或520horizon，最多8,880真实policy控制步。
每臂两条full（global2/38各state0），其余6条compact；全部保存T动作/T+1 body/EEF/gripper/BDDL和scene引用。
cost-balanced long-first dynamic queue、persistent evaluator；优先2卡、每卡2 replicas、每replica1 env，显存不足可用1 replica或1卡，
不为凑卡等待，不改变episode/video/RNG。只跑这24条，不加Source、旧MT、其它case或失败重试。
成功数仅保留原行，不作为方法收益；同时测完整worker加载/退出成本，为正式400给实测依据与长尾余量。

### 13.4 资源、所有权和一次交付

预计CPU/实现60–100分钟，GPU全阶段15–30分钟，收尾15–20，总90–150分钟，180分钟复核。
MT旧实测每576约149秒，5宏步约.207GPUh；既存接口每740控制步约60秒，8,880上界约.20GPUh；
另计新bank、scene、worker加载/恢复/IO与失败，预计完整.45–.75GPUh，**硬限1.0GPUh、峰值2卡**。
这些是投入估计，不冒称新MT或official队列实测；达到任一硬限停止新增执行，保留真实缺项。
新root `/data1/user/ymdai/ember_runs/demonstration_transfer_comparison_admission_20260927`，峰值4GiB；
原工程study2.544GiB只读。新冻树每份<=256MiB、最多两份，共新增代码<=512MiB；相关代码从1774MiB增至最多2286MiB，限2.25GiB。
启动前核data1独立quota/共享容量和双节点live GPU，完整计费含加载/失败/等待；不复制source/HDF5/296查询、不建整库cache。

原独占分支从指定main承接；当前私有学习owner扩展到确有第二用途的直接MT，bank为一个独立owner，scene为共同初始化owner。
可把run中的共同macro/ECP循环提取为内聚学习执行职责，使条件准备与执行循环分离；不是机械拆大文件或保留两个trainer。
替换手写engineering episode入口，由Git保留已完成case；旧v1/v2活动CLI及无消费者的临时实现退役。
预期净增约700–1200行，超出须说明实际职责/替换/退役关系；结构guard的REVIEW由执行者和主讨论处理，不机械请求Owner确认。
少量针对性的CPU检查后从clean pushed detached执行。完整源码/原件/成本/缺项以一次Queue交付，不逐阶段自通知。
本批只测完整比较的可执行性与成本；正式长训练、400 held与后继方法裁决仍由主讨论另冻，不能顺带自动启动。

### 13.5 主讨论独立验收及证据勘误

38303429已由481beb22合入main；M实际学习源485856c0、P6/I6源399e2610分别保留，不能把最后评测commit称作全部权重来源。
主讨论读完相关diff，冻结树八项针对性CPU测试通过，diff check通过；独立核576事件、五次M更新/三套ECP元数据、
24份NPZ/PT与8个canonical scene。M为2,880真实query（旧2,080/新800），混合task的两个8-query段各占该task一半，
36task等权的真实FM未被缩小；source可训练0、恢复前缀与逻辑游标成立。P/I首288事件与已核v2一致，均无新增更新。
八组三臂保存body/EEF/gripper/predicate初态一致，六份full首帧双RGB与两份canonical场景逐值相同；
完整model/sim/controller和其它scene双RGB的证据来自已审行为前assert，不冒称所有完整model快照逐臂另存。
24行共8,665实际控制步，T动作/T+1状态均成立；P/I各1/8、M0/8只是工程结果。

两项报告口径须更正：M日志约.0152–.0162是整个shared A/B参数组的范数，不是A、B各自范数。
结构guard实际有两个hard信号：eval_adapters.validate_episode_adapter_fields复杂度27、registered_passive_capture.validate_contract为28。
本次新增的是各自集中式协议分派的窄分支；保留一个校验owner比为阈值拆分更清楚，主讨论登记内聚例外，
不据此要求Owner额外审批。候选关闭时退役该分派与私有bank运行面，公共evaluator保留；其它REVIEW亦非无风险声明。
源文件净增709行、总diff净增711；新增bank/scene职责清楚，旧手写episode入口已删除，无平行训练平台。

实际official合同envs_per_replica=8，八个队列shard却均只有一个state，故有效每replica并发1，不能由配置推断batch8吞吐。
PT在重规划时存完整前5动作计划；两条提前成功的末段分别多3/2个未执行指令。截到真实T后与NPZ实际env.step动作相等，
不把这些计划称为实际执行，也不为此改公共capture格式。官方policy输出不另硬截断到[-1,1]，与采集控制器限幅合同区分。

完整12次GPU尝试含失败/加载/等待合计1,498.50GPU秒=.41625GPUh，峰值2卡，study约1,058MiB。
启动窗口11:08:50–11:46:58 UTC约38分钟；从10:34派发算约73分钟，低于90–150分钟估计，不把38分钟写成含实现总耗时。
原件`/data1/user/ymdai/ember_runs/demonstration_transfer_comparison_admission_20260927`；主讨论只读核验
`.codex/tmp/demonstration_transfer_main_review_20260927/comparison_admission_acceptance.json`。
所有旧冻结树/原件保持；工程准入完成，尚无正式能力、视频收益或保持结论，不再追加相同smoke/profile。

## 14. 正式比较的事前裁决框架（在§13期间冻结，首批预算见§15）

本节在§13工程期间、任何正式学习/held结果出现前固定科学判断口径；当时只授权§13。
本节本身不构成派发；完整首批预算和恢复时点随后由§15补齐，实际启动仍以progress为准。

### 14.1 两个完整P/I周期，先判断整套方法

首节点为fresh P/I各288，各32,256query；每个支持task正好完成一个4×4格、原/新各16次。
预留唯一后继为同臂续至576，各64,512query；第二个完整格重新排列，不改变模型、loss、数据或LR时钟。
288已经过150 warmup，但只有896 query/task（其中支持task新448）；它是完整有限对照，不是已证明收敛。
因此不在288单凭loss或接近的P/I分数宣称视频无用；也不借“尚未收敛”把576之后变成无限续训。
每个正式节点只做correct paired400，不插144/432小面板、不挑未登记checkpoint，不提前运行时间controls。

可在288提前关闭本候选的明确负证据为：P/I两臂均未超过历史强参照155，且P相对I的任务簇描述bootstrap
95%区间上端小于0。八个task的配对成功差等权，固定10,000次bootstrap、seed2026092710；用于本轮投入决定，
不是8任务足以证明总体显著性，也不度量重训seed不确定性。没有满足此条件不意味着通过，只容许一次已预留完整后继。
数值/数据/信息墙合同错误另按工程错误处理；不得拿无效实验作科学否决或反过来用“工程可能有问题”保护有效阴性。

576是本组合的有限学习窗口上限。不能自动转864、调rank/seed/权重、冻结公共β或补一个aux；
若未达到实用能力，本窗口只关闭这一完整组合的继续投入，不证明所有对应学习都不可能。
若I更强，仍保留I作为经验候选，不能为了原假设选较弱P；这时不声称对应干预有效。

### 14.2 绝对能力、保持与强MT是不同判据

因果比较固定为P288−I288、P576−I576，不能用P的最佳时点减I的另一时点。
本轮候选选择只考虑最新576的P/I，取合格臂中correct400更高者，平分取P；288保留为真实相邻参照，不摘出峰值代替保持。
把“实质丢失”事前定义为相邻丢失超过20条成功行（整个400面板的5%），或总成功净降超过8条（2个百分点）；
触及任一条件，该臂可有绝对性能进展，但不进入“能力已保持”的候选资格。阈值是本轮明确的研究取舍，不由数学定理推出，
也不是修改历史结果或要求强MT必须满足同一资格。每task/suite、breadth、R/G/L、churn及Jaccard全部报告；
总数通过仍不能掩盖整类操作的消失。这里的20是丢失数上限，不能用新增成功抵消后只看净差。

如两臂最新点都未超过已有155或都不满足保持范围，停止本组合的长训练/因果controls投入，完整保留局部正例和失败边界。
若有臂满足，才投入新的同数据强MT正式比较，避免一开始同时铺开三条长训练与全部controls。
MT计划fresh并保留150/300/450三个correct400节点，选其中最高、平分取较早；保持36×16、rank128及自己的完整查询流。
450为259,200 query（每task7,200），不能按Writer的每步112折算或把相同宏步称为相同曝光。
旧MT选中300=155仍报告；新MT若较低，不降低方法目标，新MT若更高则以更高的真实参照裁决。新MT的峰值不因相邻退化而删除。
三节点不是新数据最优点保证；若450仍明确上升，不能据此宣布已经压过充分训练的MT，须另评有界后继的必要性及成本。
没有本次同scene的MT原件前，不报告与旧155之间的strict物理配对R/G/L；旧历史总分与新的配对比较分开。

### 14.3 结果究竟能支持什么

P胜I首先支持这项有限训练对应改善了所学完整映射；I仍读取合法视频，故它不是learned language-only参照。
P/I对M还改变条件参数化、rank与每步task/query布局，不能由该差值唯一归因视频动态。
新M与旧155还涉及新query、46/50原episode池和物理scene边界，不把差值直接叫作新数据的纯因果效应。
若候选胜强MT且相邻保持，才在单checkpoint冻结后补足另一正确视频及learned language/static比较，
最后进行独立时序controls；这些后继仍须单独完整预算，不能由本框架自动启动。

成本按三部分核算：`train = 宏步数 × 实测每步完整GPU秒 / 3600 + 载入/保存`，
另加各单checkpoint完整400条件物化与official400的实测外推、长视频/Long长尾和失败余量。
第一P/I节点、第二节点及强MT分别有独立硬限；总计划须可承受才启动首段，不能先启动再用沉没成本索要无限续训。
本节不填未经§13测得的MT/bank/eval GPU数字；工程24条的成功或失败不会用于改动上述模型、节点或资格口径。

## 15. 首个正式节点：P/I各288与共同物理场景的correct400

本节冻结下一完整科学批次，§14的后继/选点/停止线保持。任务分为必要的CPU运行面转换与随后正式执行，
不是再设一轮GPU工程准入：当前先派`demonstration_transfer_formal_transition_20260927`，只改代码/CPU检查。
正式evaluator要求冻结commit属于origin/main；故CPU实现先交主讨论集成推送，再以精确commit登记并派发
`demonstration_transfer_learning_stage1_20260927`。不得修改Git authority绕过这一已有合同，不需Owner重复许可。

### 15.1 唯一学习干预、真实单位及事前解释

P/I均从相同fresh初始化学习公共128+条件16的一套完整144、38-target LoRA，冻结Source1000；
原生读取/Meta/Core/Procedure/FactorHeads、源/数据/优化器及§9/11相位与§13两块事件流保持。
P只在新query时用其来源教学c，I用预定4×4格的独立d；旧query/教学完全相同，两臂所有query/noise/time一致。
实际干预占每臂8,960/32,256=5/18的query，另外23,296条原始跨episodequery保留，不能说全部数据都接受新对应监督。
每臂288×4task×28=32,256query，36task各32次访问；支持20task新/旧各16，其它task32次旧。
学习唯一loss仍为自身双RGB/8state上50×7的普通真实FM，通过同一最终LoRA回传；不添辅助或动作后验码。
优化器AdamW、lr3e-4、betas(.9,.95)、eps1e-8、wd1e-4、clip1、150 warmup/绝对1200衰减至1e-5沿用。

完整比较回答对应信用有没有改善有限学习，不把P>I直接叫作动态视频必要。
P/I均强但相近，只支持该完整方法的经验潜力，不能说新对应已起决定作用；需后继M区分数据与条件参数化等作用。
两臂都弱，则这项完整模型/目标/支持在本窗口未达目标，不因中间FM下降重新补模块。
P弱于I且满足§14.1明确早停条件，停止本组合；未满足只允许主讨论考虑一次预留576，不自动续。
288即使高于155也不称保持、最终优于充分训练的新MT或视频因果成立。
工程实现记录和数学作用位置见机制§37.9；不按工程1/8结果改模型或节点。

### 15.2 正式数据流、checkpoint及只读held物化

唯一活动机器规格改为`configs/demonstration_transfer_v1/learning_spec.json`，替换旧engineering_spec及工程硬编码运行范围。
已验收38303429规格中的source/model/operator/data/optimization数值是基准；其中“仅工程”状态、执行范围与旧checkpoint引用按本节替换。
保留一个run/bank/scene owner及同一FM/ECP循环，M数值能力可复用但本批M训练/物化/评测入口拒绝执行。
原P6/I6/M3、3/6步smoke、8条件bank、小面板活动入口退出；历史在原commit与冻结树保留，不加v3 fallback。

P/I各fresh288，存72/144/216/288完整ECP；这些中间点仅恢复用途，不做bank或环境评测，不用于选点。
不额外重复恢复试验。若真实基础设施故障，允许从同臂最新完整登记ECP恢复至288，原始失败日志/未完成输出保留，
原metrics前缀复制完整，optimizer/scheduler/sampler/rank RNG与源/模型/数据/逻辑流和原world topology严格核对。
同一节点继续使用精确冻结commit；origin/main之后增加文档不改变该冻结身份，不把动态远端tip作为数值合同。
在恢复段登记实际重复计算/丢失步和完整成本，不能将重算步混作新增科学查询。不得从工程权重、其它臂或旧run恢复。
单臂逻辑112、同节点world2；microbatch28、framechunk8，OOM时仅按原合同28→14→7和chunk8→4降低物理计算量。
world size/逻辑事件/optimizer时钟不能静默变化；不为了逐bit一致固定低效batch或新增dtype探针。

只有两臂各自macro288进入bank。每臂公共128只存一次、400个合法条件各存完整16，装入时拼接唯一144；
不是400份重复公共权重，不平均/挑教学，不部署第二adapter。每次compile都恢复同一source identity。
validation任务global IDs `[3,6,11,16,23,26,31,39]`，沿coverage_v1唯一8task协议；state0–49、seed7、K1 correct，
每task50条合法教学各用一次，两臂及将来相邻点严格复用同一调度。训练仍只36个登记train/meta task。
物化使用既有`load_learning_tasks(...role=validation)`只读authority元数据与`RawTeacherVideoStore`的RGB，
不能为了复用TransferData而构造held FunctionalQueryDataset或读取held teacher action/state/reward。task ID只作调度，不能入模型。
部署前编译固定条件，官方rollout只消费生成LoRA及自己的观测；不在rollout继续读取教学。

### 15.3 一次冻结400个场景与800条官方执行

正式root `/data1/user/ymdai/ember_runs/demonstration_transfer_learning_20260927/stage1`；P/I训练根各为`P/train`与`I/train`，
bank各为`P/banks/288`/`I/banks/288`，评测各为`P/evaluation/correct400`/`I/evaluation/correct400`。
共同场景根在study级`/data1/user/ymdai/ember_runs/demonstration_transfer_learning_20260927/scenes`，未来节点只读复用，不再重新抽样。
在任何本study policy行为前，仅以官方init/seed/dummy10冻结8×50个model body/post-dummy sim/controller场景；
保存自身EEF/gripper/谓词与首帧双RGB，不读held教师标签、不按outcome选择。仍在本批完整资源账内。
每条首动作前恢复同一scene、再生obs并核完整scene与双RGB；不用环境编号一致代替物理配对。

只用`scripts/evaluate_pi05.py`的coverage_v1 `validation/formal/state-count50`、persistent/long-first动态队列，
render256/model224、双相机180度、state8/action7、10flow/前5动作重规划、dummy10、成功停与220/280/300/520 horizon不变。
每臂400行，8条full（每task state0）+392 compact；每条真实T动作/T+1 body/EEF/gripper/谓词与condition/scene/RNG齐全。
全批最多264,000 policy控制步，不能增加诊断/Source/旧MT/第二正确视频或失败行为重试。
`envs_per_replica=8`沿既有官方配置，默认2 replicas/GPU；这是物理执行选择，预算不假定未测过的batch8加速。
OOM/现场余量仅允许既有1 replica回退，或在同一评测协议下用更少GPU继续未完成队列；正常微小浮点差异接受。
不得静默切换scene、video/state映射或把成功行重跑挑选；工程失败的未完成行按既有原件/队列恢复合同处理。
若确需改变其它物理配置，先报告实际问题与已用预算，不自行扩模型/面板。

收尾报告完整success sets、每task/suite/breadth、P−I的R/G/L/churn/Jaccard及§14.1描述bootstrap；
历史MT155/Source51作原分数参照，不假称与本次新scene已经物理严格配对。相邻保持待后继节点，当前不虚构。
bootstrap仅用这800条已产生结果，不再采样环境；10000次、task cluster、seed2026092710，给出8个task差额原值。

### 15.4 有实测依据的渐进资源预算

P/I closure6步宏时均值18.115/15.148秒，world2外推288各臂合计5.322GPUh；1.35长尾因子为7.185，含载入/保存预留8。
前缀负载不足覆盖全计划的长视频尾部，不能称新288实测。首个完整P/I训练是本批科学样本，不再开一次额外profile。
两bank用包含加载的45.26/38.19秒按8→400线性保守外推1.159GPUh，预算1.5；
两official用完整125.40/128.78秒×2GPU按8→400外推7.061GPUh，预算8；共享场景28.58秒外推.397，预算.5。
这些外推约13.939GPUh，其中重复加载外推较保守，未测长视频/批量scene/队列尾部仍有不确定性。
**首批完整GPU预期13–16、硬限18GPUh；阶段分配训练8、scene .5、bank1.5、official8，总计18。**
加载、等待、失败、重算、物化、渲染及评测均计入。任一阶段预计超出分配先停止新增相关执行并报告，不以剩余阶段未用预算掩盖异常。

项目本批峰值4张物理卡：两臂各world2可并行，每臂保持同节点；只有2–3张适用卡时顺序，不等凑卡、不跨节点拼训练。
官方可用最多4卡的真实persistent工作，阶段释放设备，不dummy占卡。每次launch现场核两节点及全项目占用，服从全局总量上限。
预计GPU阶段4卡4–6小时、2卡7–9小时；这是按上述保守吞吐与阶段并行安排的范围，不承诺固定完成时间。
CPU转换预期45–75分钟、90分钟检查实际剩余工作，不因墙钟到点自停或制造额外probe；其后不需要另加GPU smoke。

首批输出峰值12GiB（含study共享scene）：8套恢复ECP约1.58GiB，两bank张量约3.91GiB，16条full按最大horizon约1.55GiB，
compact/原始轨迹/scene/日志/临时checkpoint另留余量。相关旧study只读；新冻树每份<=256MiB、最多两份、新增代码512MiB，
相关代码总限3GiB。创建root/冻结前由执行者查strg01 data1独立quota、目录用量和共享容量，不将home/data0当新输出。

预留后继不是当前授权：第二P/I576节点预计同量级、上限18；新M450训练按34.18秒×2×450外推8.545GPUh，
加三个official400约10.58及载入/余量，暂预留22。全部通过才可能合计58GPUh、输出30GiB（尚不含后续controls），
每次须按前一批新实测和科学资格另冻，不以沉没成本要求继续，不提前启动MT或储存其正式权重。
成本算术原件`.codex/tmp/demonstration_transfer_main_review_20260927/formal_stage1_budget_basis.json`。

### 15.5 当前CPU交付范围与正式启动条件

唯一指定执行者在原独占dev/分支从本次main承接；源码修改限run/data/bank、共用scene及必要官方注册和对应配置/CPU测试。
沿既有owner将固定工程常量改为本节正式范围，预计净增不超过250行；若实际职责需要更多，在交付说明，不机械切文件规避guard。
model.py及共享native/FM/LoRA算子数值保持；不得新建trainer、评测框架、第二scene恢复或新的运行平台。
CPU验证真实事件前缀、登记游标恢复/拒绝工程来源、held视频读取不构造动作query、正式范围与capture/adapter准入；
只做这些变更需要的检查，不运行模型forward/backward、环境、GPU、held像素扫描或新数据构造，不创建正式run root。
给出唯一可执行命令、环境、最终源码commit与差异；不要填假设备，由正式启动时的live preflight补设备与资源字段。

原engineering_spec退出，新的learning_spec与代码一同推送隔离分支，以一次完成/实质阻塞Queue交主讨论。
主讨论检查并合入main后再登记准确formal commit和启动消息；当前CPU任务不自行开始288、bank400或scene/eval。
正式批次随后亦仅一次整批完成/实质阻塞Queue，不逐训练/物化/评测自通知，不陪跑轮询。
这两个交接由已有main authority合同所需；不是再次向Owner请示，也不是增加科学局部关卡。

### 15.6 正式运行面独立验收与唯一集成修正

3bee0a1b由9b480d2d合入main，7文件+584/-355、活动源码净增125，没有新source模块。
主讨论逐项核对source/model/operator/data/optimization/mt六个字典与6b6260e2完全相同；
仅活动范围、held只读bank、官方capture、ECP游标和冻结authority发生转换。机器status随后改为已登记可launch，
runtime补齐已有3GiB代码上限，optimization.after_warmup仅删除已失效的“CPU definition only/no long training”文字；
所有模型/采样/优化数值不变，当前授权由本节/progress确定。
八项定向CPU测试在交付树独立运行通过（17.70秒），guard REVIEW无新hard；新增中央路由3行的preparation为800行，
保留一个官方capture分派owner，旧两处复杂度的内聚例外不因此消失。没有模型/环境/GPU验证或held像素读取。
旧工程活动spec已删除/改名，不把已完成P6/I6/M3兼容分支保留为fallback。

主讨论集成时发现并修正一个确定的完成边界：从登记macro288 ECP恢复时没有剩余更新，原代码会将final pointer写向
新attempt中不存在的checkpoint。现复用同一个ECP保存owner，在新attempt重存已恢复的完整288状态，再发布completion/pointer，
实际新增更新/query均0，旧attempt/日志/ECP不覆盖。这只处理“已保存最终ECP但完成记录尚未发布”的中断，
不执行新的模型前后传，不允许再训练或选择不同权重。新增一个纯CPU控制流fixture验证实际指向存在、零更新与零query，12.41秒通过；
它模拟既有ECP保存入口，不冒称实际world2终点恢复已重跑。普通更新保存也复用该owner，数值操作未变。

正式恢复的现存边界明确保留：尚未产生首个72 ECP时发生故障，现入口会保留失败原件并停止，不能自动抹掉fresh目录再来一遍；
此时由主讨论根据真实失败和剩余预算决定处理。已登记的物理OOM缩小用于有合法ECP的恢复，不能把入口限制称为已验证任何故障可自动恢复。
这不是要求增加一次smoke或新恢复平台；正式执行仍遵循18GPUh/800行/单一source与上述信息墙。
