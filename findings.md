# EMBER findings

本文件保留跨轮结论及其适用边界；历史段落的“当前／下一步／active”只表示当时时点。
当前Owner要求见[稳定要求](docs/current_owner_requirements.md)，当前授权、设计和执行状态只看[progress](progress.md)。
完整历史索引、旧设计及原始证据入口见[research_history](docs/research_history.md)。

§94–100记录固定纠正闭环、语义路径比较、冻结行为回放、局部场、G P rank前提及Process Pullback纯FM窗口的完整结果。
这些结果不能混成一套已经取得有益视频特异性的模型；旧普通FM及其它合法视频Writer的能力正例也继续保留。

## 1. 先分清三个问题

- Task-local LoRA能否完成任务：已有强正证据，validation8 privileged rank16专家250/400。
- 视频到LoRA能否产生真实能力：早期v5.2/v6已有正证据，v6曾143/400。
- 共享Writer能否从正确视频稳定迁移、持续积累并满足因果性：仍未解决，不能由前两项推出。

[基线与早期证据](docs/research_history.md#baseline)。各种clone、oracle、free code只说明其实际接口和预算的能力；不能部署成task字典。

## 2. 原生信息要有实际消费者

完整50-horizon、层轴、native prefix及有效梯度不自动构成过程理解。H必须在有意义的读取前保持完整，
但“使用了原生信息”不能代替动作先验被实际消费的证据，也不要求永久保留某个层轴、DP/event或关系模块。
对齐后内容差为零不等于过程未推进；对应位置也可能含信息。固定probe形成的共同结构或斜带不能证明物理动作对应。
离线双向读取符合rollout前一次编译，有限上下文的效果与视频因果必要性仍须分别验证。
[原生容量与动态](docs/research_history.md#native-capacity)。

## 3. X/Y、输出span与真实功能是三个层次

G1证明某些native X/Y signed pooling具有局部容量，也通过投影干预证明过窄Y span会丢失Goal/Long必要方向。
显式读取X/Y与强制因子在其span中，是两项独立选择。真实policy反传的gxᵀ本身提供原生参数坐标，Y=WX+b不等于应该施加的修正。
压缩E不是X/Y无损副本，观察Meta侧激活也不是执行场景激活。

普通family head的固定末投影确实限制生成方向，但早期FactorHeads也有强行为；不能把这个几何事实直接称作新近低分根因。
坐标条件MLP是明确的候选解除方式，没有通用性能保证。

旧native坐标初始化及读出常量方向的诊断，只支持相应学习条件检查，未唯一定位共享学习缺口。
具体对照与数值见[学习历史](docs/research_history.md#recent-learning)；几何改善不能替代行为。

## 4. 参数稳定性不能代替成功集合稳定性

极高BA cosine、较低same-task参数方差、更高名义rank与实际success churn并不等价。正式结论同时看绝对成功、R/G/L、breadth、
suite贡献与相邻/跨视频重合。稳定的低分、一次高分或不同checkpoint优点的并集都不足以通过。
GOMQ的151未保持，原rank32写法有效rank本来≤16；重物化136不能被简化为有效rank容量损失。
[早期稳定化边界](docs/research_history.md#early-writers)。

## 5. 更多视频、更多任务和更多训练量要分别比较

K是一次condition真正使用的视频数；视频池大小是跨训练可见的不同演示数；meta-task数是独立映射数；每task query/updates是监督预算。
旧两视频池→四视频池仍然K1，固定预算使单视频曝光减半，它的负结果没有检验K1→K4。

多K可以减少独立干扰、增加关键证据覆盖，但相关偏差留下误差floor，多个正确策略也可能冲突；实际Writer不保证K增大单调更强。
必须独立保序编码、集合共同读取、真实训练cardinality、无放回不同视频，并保持task权重。
正式K1的无重复还要求同task同臂跨50个init各用50条视频一次；不能以单次K集合不重复替代整轮覆盖。旧Horizon64的99/95不满足该合同，只作历史探索。

## 6. 共同学习获取与未见任务迁移须分别建立

旧完整输出、target18／meta73和clone／shared对照说明，部分组合连训练任务的能力获取也不足，
不能把全部负结果归为未见task迁移或先学会再遗忘。包含多个变化的收益不拆成单因果解释。
容量、条件表示、优化和任务支持仍是竞争解释；gradient cosine或更低loss不能单独裁决。
[学习对照及适用范围](docs/research_history.md#recent-learning)。未测闭环的候选不补写好坏结论。

## 7. 短面板必须有代表性，局部监督不等价闭环

受监督Object仍弱，失败涉及接触、抓取、放置等不同接口。Long93的专家本身弱，不能用它概括Long；短面板需有实质容量的Long参照。
从teacher状态接续仍可能失败，occupancy不是统一解释。functional改善但闭环弱，应定位真实失败阶段和最早可证伪接口，避免泛化命名。

小面板服务投入判断，不选择最终checkpoint；最终恢复完整train24及固定validation8/400合同，不能长期停留在18任务和两视频支持。

## 8. 保留可验证的机制，停止专用补丁链

G3/primal/PNBTT等存在真实operator/task-local正证据；shared或correct/wrong冲突的non-pass只淘汰其实际函数类。
PNBTT停在free-query E1，真实Program E2未运行，不能借此否定G2。

多次在一个失效接口前后加summary、whitening、transport、anchor、gain或gate，没有形成稳定完整Writer。新一轮改动应写清
最近等价旧尝试、它排除了什么、本次主要变量及预期分支；真正证据支持职责替换时可以重构，不能用连续版本号代替判断。
[共享编译器历史](docs/research_history.md#program-compilers)。

## 9. 速度首先是算法布局问题，不能改变科学语义

已核验旧full Writer同4卡从34.394→4.054秒/step，8.48倍，来自exact批量/SDPA/融合/placement与有效microbatch。
共享mmap改善负载，重复物化复用resident policy减少加载；端到端收益应与单算子收益分开报告。
[准确吞吐范围](docs/research_history.md#throughput)。

Meta更新后不得复用旧版本激活；冻结prefix、同step临时激活、policy VJP及分块重放可按真实依赖复用。
成本由当前模型、合法cardinality、最长真实视频和实际queries测量，不能继承旧设计的加速倍数或profile结论。

## 10. 旧分层图形成时需要回答的科学问题（历史）

1. 单probe+Action Meta能否为新图产生可学的教学响应，且真实视频变化进入过程Value？
2. 分层局部对应/过程和集合编译能否让同task的新视频保留功能，而不依赖静态目标识别？
3. 坐标MLP能否在共享训练中得到有用的完整A/B；改进来自何处，是否付出旧能力损失？
4. 真正K1/2/4训练与更广视频支持是否带来性能/coverage增益，而非仅降低参数方差？
5. 在完整目标支持和合理训练量下，能否达到single-checkpoint >145/400，并通过稳定性、困难suite和最终因果controls？

这些问题不授权任意扫描。按 [task_plan.md](task_plan.md) 和已登记的持续执行授权推进，先用有信息量的实际证据区分分支。

## 11. 新图已观察到的输出学习条件

原std0.02 native坐标短学习没有广泛稳定增益；匹配std1坐标对照改善了functional学习，96步K1 correct/other从4/4到6/6，
K4从6到8（各40）。收益主要在Long，所有节点Object/Goal仍为0；不能把局部收益称为共享或泛化问题已解决。
原生通道对比恢复后，生成B的rank槽仍高度相似，真实功能梯度却需要不同的槽更新；全局共享末读出也耦合不同target的梯度尺度。
这些是继续检验参数共享方式的证据，不是唯一根因或性能承诺。下一离散对照只解除末读出的target/rank共享，保留其它图和已观察到的正证据，
由相同曝光和真实闭环决定投入。详见[历史§15](docs/research_history.md#15-native坐标初始化对照的局部收益与边界2026-09-07)。

## 12. 末读出共享约束可以影响真实行为，但几何不是裁决器（2026-09-07）

在坐标std1和完整图保持不变的前提下，末读出从全局[p]改为[target,rank,p]、增加77,696参数；384条件/6144queries及所有采样/RNG均匹配。
short4 96步K1从6/6到11/11，两正确视频success集合完全相同，并首次覆盖Object（2/10）；Long6、Spatial3、Goal0。K4为10/40。
代表target的rank趋同解除，但native常量能量反而上升；真实闭环变好不能归约为几何指标变漂亮。相邻48→96仍有7/40churn。
这支持保留当前参数共享方式扩大到fresh train24，不能证明视频因果、未见task迁移或最终稳定性；不追加没有行为证据的decoder小修。
完整受控证据与适用边界见research_history§16、target_rank_readout_control/registration.json、panel_summary.json和readout_output_contrast.json。

## 13. 首个train24节点保留Object增益，尚未形成广泛稳定迁移（2026-09-07）

当前完整图fresh训练192步、每task512queries后，strict400 correct/other为69/72，对冻结source47；
两组分别保留34/35、新增35/37、丢失13/12，增益主要是未见Object1的5→29/31。
Spatial3/4、Goal37/37、Long0/0，breadth5/8；两视频success Jaccard60/81=.740741，不满足登记的.80。
train24 held-video paired120为22，对source16，RGL11/11/5，Long仍0。该面板不能单独区分训练拟合与新视频泛化缺口。
这既不是整条图失败，也不足以宣称广泛迁移或视频必要性。按原节点续训至384并获得相邻strict400，
检验是否出现更广行为与稳定趋势；保持科学non-pass的边界，不转入输出几何小扫。证据见research_history§17及train24_shared/。

## 14. 增加到每task1024queries未带来广泛稳定扩展（2026-09-07）

同run exact-resume至384后，strict400 correct/other为67/64（192为69/72）；Object33/31、Long4/3，
但Goal降至27/26，Spatial3/4，breadth5/6。两组都保留Object局部增益并出现稀疏Long成功，总分未提升。
相邻correct RGL46/21/23、churn44/400、J46/90；other47/17/25、churn42/400、J47/89。
同点跨视频52/12/15、churn27/400、J52/79=.658228，仍非稳定且广泛的迁移。未发现工程合同违反，不把non-pass解释为代码故障。
已暂停按schedule继续训练，先做冻结384、相同train120初始化的已训练视频与held视频诊断，区分更早的行为接口缺口。
该诊断不选checkpoint、不产生梯度，不以此声称视频必要性；不能仅用已有held-video训练面板断定唯一泛化根因。
完整配对、成本和下一步依据见research_history§18与train24_shared/decision_after384.json。

冻结384熟悉/held视频诊断最终为21/18（各120），Long均0，source16；两组RGL14/4/7，held还从192的22降至18。
当前缺口已在训练任务与熟悉视频出现，新视频代价不是充分解释，编码器/解码器/共同优化仍需区分。
本run已止于384。不能再用source47的局部增益或较好输出几何开脱相对SFT109/107的性能失败；
下一离散输出参数化对照保留上游完整图与target/rank独立性，代价与效果由实测报告，未预认定解码器是唯一根因。

## 15. 完整H查询、有序窗口与单向长程的最终选择（2026-09-08）

Owner与专家多轮讨论后收口：直接读取最终post-norm、action_out_proj前50个hidden；无18层保留或多层融合。
当前帧只读过去4个实际采样位置，帧对软对应后沿完整H联合形成50个视觉query，分别核实两端Z，再按历史u短GRU更新。
新H-query增加的是不同新匹配行之间到视觉query的直接依赖，不能由“原生AE已有H交互”自动替代，也不能据此预告控制能力。
重叠的[u,t]证据不是相接动作段；GRU给显式有序条件化，不保证去重或贝叶斯更新。

四组局部—长程交替保留完整U，临时H-read后全局时间信息按当前h状态非线性回写，前三组回写、第四组直接送compiler。
Owner最终选择过去单向long，覆盖专家原文的双向；每组t表示仅依赖视频前缀，完整H内部仍可双向。
单Key attention的softmax恒1，不能冒充逐H内容条件回写；当前选择的MLP融合U_h和P_t具有相应交互通路。

首版FM辅助共享Writer RL使用真实action-Gaussian探索和同版本单次联合更新。监督/强化同一步是训练选择，
不由Writer/Meta端到端自动推出；LOO全同结果组没有RL信用，J_Sigma改善也不能替代正式J0。
新图尚未实现、没有新训练或闭环证据；下一步是实现与有效实验，不继续悬置已定计算含义。
原文、Owner覆盖与源码缺口见[讨论索引](docs/review_materials/20260908/README.md)和正式设计。

## 16. 基线差距不能降格为小调参问题，具体方法可据证据实质调整（2026-09-08）

本次复算已存导出400行面板：source47/400，rank128 train24 SFT step400/425为109/107；没有新环境评测。
另历史source48是另一面板。原件及比较边界见research_history基线；SFT不是同参数量对照，但其行为能力不能被忽略。
旧图67/64虽超source47却低于SFT109/107，不能以source局部增益、几何更漂亮或loss更低继续解释为基本成功。

Owner要求接班者在有信息量的可比正式节点，若不及或仅略超这些基线，按严重能力缺口重新定位实质机制；
不能把它当作只需小修小补，也不能反复把足够学习的判断推迟。低分本身不唯一指认某个decoder、梯度冲突或数据根因。
改动前检查最近等价旧实验，明确本次新增了什么信息、监督或算子，以及怎样区分竞争解释，避免重复专用补丁链和无依据扫参。

核心思想保持，但具体读取/关系/时序/回写/读出/训练与工程机制可以在硬合同内自主修订或重构，并记录证据和更新正式设计。
完整忠实实施禁止静默缩水，不意味着机械坚持已知错误配置；新session获充分证据驱动自由度，不限于表面补丁或逐项请求批准。

## 17. Gaussian trust不能把活动batch数值变化当作策略更新（2026-09-08）

完整新图首profile的同参数KL在提前成功后batch缩小的tasks非零，task21超过既定trust阈值，
而始终batch4的五个task均为0。真实固定参数、观测和噪声重放证明原batch尺寸自比较为0，统一batch4产生伪KL；
改变同尺寸内行排列仍为0。证据见[history §22](docs/research_history.md#22-2026-09-08完整horizon首版与真实联合profile)。
因此RL VJP和trust保留采集batch形状，填充仅为执行且不贡献梯度或统计权重；不改collected m_old/Sigma/threshold。
这是已定位的执行数值接口修复，不能推出接受后的行为一定改善，也不支持以扣底噪、扩大dtype或阈值替代实际候选验证。

## 18. 参数相邻值敏感性与同参数重放误差是两种证据（2026-09-08）

batch形状修复后同版本KL均0，但八档回溯仍1/4接受，新增小尺度没有解除停滞。
固定已生成A/B、真实观测、flow noise和原batch后，完全同B重放KL0；将680448个非零B元素各朝+infinity
移动一个BF16相邻值（聚合relative L2 .57214%）则task KL=.034327、最大动作差=.0512085。
这证明执行端单独足以产生该量级的参数扰动响应，不能把它称作随机重放底噪，也不能仅凭此证明dtype缺陷。
此控制是特定整体参数方向，不能冒充所有候选更新或全模型最小扰动。下一精度因果对照需要单独证据；
不据此替换训练旧均值、放宽KL、跳过实际联合更新验证或宣称闭环性能。原件见research_history §22及joint_profile/adjacent_b/。

## 19. 等价验证必须沿真实调用上下文，非零adapter须覆盖物化路径（2026-09-08）

将参考policy也包入训练用autocast，会让两条受同样修改的路径看似一致，却未检验真实evaluator。
本次实际evaluator无外层autocast，保持FP32 action/time heads；旧flow外层BF16把这些输出也降成BF16。
非零LoRA原生reference对旧flow有明确变化，因此v4恢复原生执行边界，同时保留Writer/Meta/FM的正常BF16训练。
批量LoRA还应保留PEFT的累加顺序，不能先舍入delta再相加。该接口已用CPU舍入边界复现；不把一般kernel低位差异
升格为逐元素一致要求，不用额外dtype或更小回溯序列替代真正合同修复。相关原件见research_history §22。

修正后真实native flow在固定非零LoRA的64个decision与物理PEFT输出一致；实际物理36个expert targets A/B是BF16，
两个action head targets是FP32。因此先前把全部state.float的探针仍未匹配物理合同，按owner参数dtype与连续布局适配才是正确路径。
实际批量执行仍KL .016349，不将bmm与linear的数值差异当成新的逐元素一致门槛。完整v4 profile已2/2接受，
第二步Meta梯度非零；这些证据允许开始正式共享学习，但仍不能证明闭环改善。原件joint_profile/native_fixed/与native_e1ea3596/。

## 20. 监督获取与共享RL分阶段裁决（2026-09-08 Owner选择）

Writer与读取Meta共同训练，不要求FM与RL在同一步混合。最新要求先fresh纯监督FM，通过实际曝光、独立held动作验证、
train闭环和相邻validation判断平台，再从单个保留监督checkpoint开始独立共享RL。弱监督平台须先定位，不能默认让RL救场。
此前joint profile只证明已检验的机制/执行/成本，既不证明监督饱和，也不阻塞当前纯监督训练；不得用其checkpoint充当正式监督起点。

监督曝光不能用step数直接与历史SFT比较：SFT rank128 step400/425实际已有230400/244800动作queries，
每step全部24task各24条、每task50动作episodes；当前Writer每step四task各64条，动作池16–41共26episodes，
64/128/192节点分别16384/32768/49152 queries。192是证据节点，不是充分学习或总预算的自动证明；
是否续训仍按held-action与闭环改善预登记，不靠凑齐SFT相同query数宣称公平或有效。数据支持、模型、rank与训练设置不同，
这些计数只能解释曝光差异，不能单独因果归因。原始run_contract/metrics和汇总见
`runs/analysis/horizon_relation_writer_20260908/supervised/supervision_exposure_context.json`。

Fresh K1采用合规固定视频映射后的55→110→86表明，早期总分获取并非单调保持：200→300主要丢失集中在一个Object task，
同时Goal仍有净获取。后续须结合相邻task成功集合与训练侧held诊断区分能力获取、遗忘和泛化，不能以总FM下降替代，
也不把单次回落自动判为工程bug或整套架构失败。具体原件与适用范围见[macro300历史记录](docs/research_history.md#2026-09-08-fresh-k1-macro300及200400续训完成)。

比较监督量时，除action queries，还应同时记录teacher条件数、每update覆盖task数和每task视频池。当前K1 step400=102400queries/1600条件/378种task-video；历史v6-fast step200=96000queries/4800条件/1200种task-video。
相近query量并非相同的Writer条件学习量；采样、架构、LR等共同变化时不能单因果归因。训练task同分布held FM下降也不能回答训练task闭环是否学会，更不能保证未见task恢复；该中间证据随后由冻结200/400 held-video train96补齐，见下节。


## 21. 训练任务换视频可用，不等于跨task迁移或全部任务能力已解决（2026-09-08）

Fresh K1同一held-video train96从200的52到400的59（source15），breadth20→21；与此同时validation correct110→87。
这使全局训练能力未获取或全局换视频崩溃不足以单独解释validation缺口，支持优先审视跨task迁移；不能从不同任务集合、4/50状态数的原始成功率差直接估计因果泛化差距。
Long训练侧9→7、无新增而丢2，400三个零task都在Long。因此局部训练缺口仍可与跨task问题并存；全量熟悉视频不是自动下一步，针对局部疑点时才追加必要的预先固定配对诊断。

历史meta73→target18虽保持target条件/query曝光，却改变target权重、每update组成和总queries，且旧loss有逐task normalizer。
它说明该旧组合移除meta后的训练收益，没有单独裁决独立映射数量。旧clone强于共享也没有分离容量、优化和条件表示，不能搬作当前唯一根因。
原件与三项有界历史比较见`runs/analysis/horizon_relation_writer_20260908/k1_fresh/train96_diagnostic/`；完整500/600节点仍须独立裁决后期获取与保持。


## 22. 已有证据审计：不能把当前缺口归为四任务训练本身（2026-09-09）

Owner暂停所有实验并要求深入分析。原K1六点correct55/110/86/87/70/82；600的81/82成功来自source原全部成功所在三个cream-cheese相关task，其它task从200的28降至1。训练held-video200/400/600为52/59/67，600有12个task在固定小面板4/4、四个task0/4；训练获取与局部缺口、未见task迁移/保持问题并存，不能把train/validation不同面板成功率直接相减估计因果泛化损失。

旧v5.2old/v6old每步同样四任务、单视频、多action queries、纯FM，已有132/121；旧task-complete最好点分别120/143，效果方向相反。旧轮遍/视频池/queries/LR与架构都不同，不能只拿v6-fast143优先定责当前4task；也不能因24个mapping就宣布不够。v6old900/TC150实际3600条teacher按task/visit配对一致，旧分析笼统“teacher realization不同”需缩窄，optimizer聚合/LR/flow RNG仍不同。

当前比旧强v6多了裸语言compiler残差，改为单主language code、统一P4及按target/rank独立native D；旧有contextual task-token/Core-Procedure及共享family heads。这些是可解释任务特化/迁移弱的实质归纳偏置变化，不是已定位的语言捷径或D容量根因。旧强模型也single probe/positiveFM/learned256span，因此这些共同因素不能单独判死刑；G2直接监督和接口不同，不证明当前四组已学动态。

CPU重算当前完整sampler/action traces未发现恢复/配对错位，600步无global clipping，Meta及过程/编译模块确有参数更新；未发现缓存旧R、重复LoRA或当前schedule的确定性故障。已有native parity只覆盖旧joint macro4任务面板，不冒充当前全量验证。既有600 LoRA在同task内较接近、task间差异更大；有效BA几何不能证明视频必要性，也不能当坏结果原因。

当前最有支持的工作解释是条件表示/参数共享与正样本FM耦合，继续适配训练任务时未形成或保留广泛迁移；训练组织/池/优化差异只是可能放大因素。具体首次失效模块、视频因果贡献和动作失败阶段仍不可由现有证据唯一识别。完整证据、反证、覆盖范围及原因排序见[审计报告](docs/horizon_k1_evidence_review_20260909.md)。24-task草稿与所有实验保持暂停。

## 23. 头部共享和路由限制有近等价历史，先分析实际功能再改模（2026-09-09）

当前D约占Writer89.38%，不据此认定过拟合。旧Target-Owned已经使用同target内跨rank共享native读出，20,594,688末投影参数，正式99/76/86/68；当前把329,515,008独立D收为同样大小的rank共享D并非未试过的新机制。旧前端与配方不同，不能单独否定当前受控对照，但它应降低无新机制证据就重训的优先级。

旧v5.2/v6已有纯language只Q、video content作Value/残差的真实实现；Dynamic-K Semantic-Address、DirectFamilyB、task-grounded D/G也表明这种限制并不充分。删除当前裸语言残差只能排除该条可达路径，不能宣称视频动态必要性或闭环根因已证实，更不能扩大为删除全部视频条件共同语义Value。

Owner已授权直接做冻结分析实验与架构内部拆解，先分析后决定正式修改/训练。应同时补实际行为阶段与正常输入下的功能敏感性；分量大小、局部梯度、目标谓词或小面板分别保留适用边界。完整历史与来源见research_history §2补充，当前固定诊断口径见active design §8.2.4。


## 24. 冻结接口依赖与实际失败阶段已分开，仍不能唯一识别泛化根因（2026-09-09）

原200/600在相同train24×teacher46×16queries上，P4/cross实际非零且单cross lesion会改变真实FM预测。首层language残差移除FM增加.026782/.028808，23/24与24/24任务变差；只移除首层language检索增加.000188/减少.000107、方向混合。当前训练侧功能对首层直接language内容依赖较强，但后续路由/跨层补偿没有被固定；不是全局路由无用、语言捷径或fresh删除有效的证明。

预登记validation8×四state×两checkpoint回放完成，9/32→7/32；实际配对通过，但与各自历史7/32成功集合不完全相同，因此图像只解释本次轨迹，原正式分数不改。所见Spatial错误对象/实例、BBQ200正确目标到600干扰物、Long独立子目标与混合抓取/干扰，以及实际成功双目标组合，均不能统一为motor失效或单个过程层故障。

新证据提高了对compiler直接条件内容路径做单变量fresh检验的信息价值；该对照保留首次language检索、原前端/decoder/4×64，不同时改D共享或task组织。它是检验归纳偏置的学习实验，成功/失败必须由strict400与训练侧行为裁决，不将冻结activation lesion的OOD损失变化当作预先答案。完整原件、范围和边界见[冻结诊断报告](docs/horizon_k1_frozen_diagnostics_20260909.md)。

## 25. 首层直接语言内容移除尚未成为整体修复（2026-09-09）

完整单变量fresh首段100/200 correct75/110对原55/110；新200和原200总分/breadth相同，保留86、新增24、丢24。新100→200仍净增35、四suite净增，但breadth6/8不变、global1/23持续0，churn75/J=.4231。新200 train96=46、breadth18，低于原52/breadth20；训练侧S/O/G/L为12/15/16/9→11/12/14/9，三个suite下降、Long持平。同3072query held FM均值.111184/.111353接近并未对应训练行为等价。

因此不能把原裸language残差认定为唯一泛化根因，也不能称删除后修复。新模型仍有真实学习获取，首段未覆盖原300/400由110回落86/87的区间；同变量固定续至300/400是保持检验，不是新的架构配方或按loss无界续训。低分只约束实测阶段和接口，后续必须用完整保持区间与训练侧证据裁决。完整表、配对与范围见[首段报告](docs/horizon_k1_first_query_only_20260909.md)。

## 26. 语言内容移除的局部收益不能替代任务保持与迁移（2026-09-09）

新100/200/300/400 correct75/110/106/103，对原55/110/86/87。后两节点有20/16个净成功优势，但自身200后不增长；BBQ28→24→10的回落仍在，cream cheese等增益掩盖部分损失。300→400 R/G/L77/26/29、churn55、J=.5833；400breadth6、Long9，两个task全程0。Spatial任务3在200/400均3/50却成功集合无交集，Long双物9→8只保留1，说明近似总分不等于稳定能力。

新400 held-video train96=59、breadth22，对新20046/breadth18为35/24/11；与原40059/breadth21为46/13/13。新400 S/O/G/L12/21/19/7，对原17/18/17/7，Spatial少5由Object/Goal增益抵消；训练获取真实，不能解释为全局未学会。配对3072query held FM新400.105738、原.106272，只有11/24任务新更低，内部均值接近不代表行为等价。

本轮否定该处内容移除足以修复当前缺口，保留局部贡献与其它接口的不确定性。不自动沿用已重现的“训练能力继续提高、目标能力不扩展”趋势续500/600，也不把它误写成训练完全饱和。下一步须有新的可区分机制和近等价历史边界；§23关于Target-Owned rank共享已有失败的事实仍有效。完整报告[首层内容对照至400步](docs/horizon_k1_first_query_only_20260909.md)，本段原件`k1_first_query_only/segment200_400/round_evidence.json`。

## 27. 训练内功能对应的增强并未带来验证迁移（2026-09-09）

按active design§8.2.7，新200/400、teacher46/47的既有train24 LoRA在固定32query上交叉执行，含source共74,496query预测，全部无梯度且实际time/noise配对。400的24task在两teacher、两个16-query半面板均比其余23task adapter均值低FM；跨task margin由.009396增至.015491，自身FM由.111621降至.106557。与train96 46→59、validation110→103并列，进一步降低“训练内条件功能普遍未形成”的优先级。

同suite margin从.003599到.005578，但Object及部分Long细分差值小，400仅18/24task满足两teacher、两半面板均优于同suite均值。policy本身读正确language，合理共享通用技能无需每task独有LoRA；本矩阵既不单独定位错误路由，也不证明FM足以支持闭环。数值缓存核对约.9%–1.1%逐点loss相对差，不把细小排名当精确机制证据。

当前global语言读出有位置而非无序mean，但其静态token到单query attention尚无预训练上下文task-token语义；现有Z视觉重读已经能读取这些上下文tokens，不能说语义缺失。若后继检验更直接的上下文过程条件，只能作为同源信息访问的受控变量，保留四组动作关系/视觉核实主图；不能据结构名字宣称修复。完整报告[冻结功能对应](docs/horizon_k1_functional_assignment_20260909.md)，原件`k1_first_query_only/functional_assignment/`。

## 28. 上下文过程条件首段未显示整体优势，早期恢复仍需与保持分开（2026-09-09）

只替换四组过程条件为同次冻结Gemma的逐帧exact task-token读取，保留reader参数、first-query-only、完整过程图/native D及全部实际采样。fresh100/200 correct52/103，前轮75/110；200 train96为41，前轮46。全部完整checkpoint、896个新bank条件、曝光和三面板配对通过，均exit0；没有发现工程失败足以解释结果。

200 S/O/G/L1/54/37/11、breadth7，但Spatial1与Goal23仅各1/50；100/200的Goal23成功state不同，Long4→11也只保留1次。前轮200→本轮200总R/G/L76/27/34、Long2/9/8，不把Long总数或breadth当稳定能力。train96为35/6/11，breadth18保持，局部双moka新增2被book丢3等抵消。held3072query均值.111031/.111184接近仍不能推出行为等价。

相对早期缺口23→7、自身52→103支持继续区分较慢获取与后续保持，不能说明当前改动优于前轮。是否值得投入一个固定300/400区间与是否已经有收益是不同判断；后续若无实质获取/保持优势，不按loss无限延长。负结果只约束本次表示来源及共享reader组合，不独自否定上下文语义、整个过程图或纯FM。完整报告[上下文条件首段](docs/horizon_k1_frame_contextual_20260909.md)。

## 29. 上下文条件完整保持段未修复获取/保持，FM改善不能替代行为（2026-09-09）

本轮100/200/300/400 correct52/103/79/90，前轮75/110/106/103；400 train96为49，前轮59。相邻300→400 R/G/L63/27/16，200→40056/34/47；BBQ25→3→1且200原25成功到400全部丢失。400 global1/3各1、23/32为0，四suite非零不等于稳定广度。完整source/视频/state/RNG与checkpoint配对通过，没有工程失败证据解释整体负结果。

400 held FM .105074533低于本轮200全部24task，且均值低于前轮400 .105737594；训练面板41→49仍有获取，不能将问题全称为训练任务普遍退化。平均拟合、更丰富上下文访问与实际可保持闭环能力之间仍有缺口；本次干预无整体收益，结束原样续训。该结论只限制实际检验的条件来源/共享reader组合，不单独否定完整H/过程图/原生语言，也不授权新正式方法。

完整[100–400报告](docs/horizon_k1_frame_contextual_20260909.md)与`k1_frame_contextual/segment200_400/round_evidence.json`保存原件。Owner目前仅授权原因诊断，实际状态以progress为准。


## 30. 完整因果诊断：功能拟合、接口获取和后续执行必须分开（2026-09-09）

B2完整native FM和10-step动作MSE仍无法排序闭环；context400比first-query400前5动作MSE在19/24 task更低，train/validation却49/90对59/103。padding拟合不能解释大部分改善，夹爪事件覆盖存在，小面板事件退化未稳健复现。

B3全24task×四state×五臂480行完整：normal51，H-read/Compiler/visual零54/52/54，分别R/G/L47/7/4、40/12/11、45/9/6；全后端语言零40、33/7/18。当前单支路贡献混合，联合语言依赖存在；不是fresh删除或视频必要性证明。

八task局部final64的C闭环normal/P4/C/AB/expert18/17/20/15/16（各32）；task7 normal/P4均1，C/AB均4，说明当前冻结D有局部有效策略可达，但不能区分P4信息与Compiler/优化。C净增伴随Goal/Long损失。直接A/B fit .095526→.039659，独立 .101330→.116502；换time/noise后原fit收益只剩6.8%。固定随机点局部求解不是容量上界，也不证明正式训练复用了噪声（1600condition各自seed不同）。

Long36/38固定state32回放均已转向第二对象后未完成操作；旧专家同状态也失败。不能把这些例子统称未选择下一目标或Writer独有问题，也不能把专家658/1200当每个任务的能力保证。后续优先验证有效功能修正和实际到达/恢复状态上的行为，再分别检验语言简化与P4→C获取；不按D参数占比、局部loss或任务数量猜测直接大改。

完整[诊断报告](docs/horizon_k1_causal_diagnostics_20260909.md)与`causal_diagnostics_20260909/summary.json`保留全部原件；新诊断闭环676行完整，正式架构/训练未改、未启动正式训练，无held梯度或最终视频controls。


## 31. 因果解释纠正与探索授权澄清（2026-09-09）

Owner指出并确认：冻结置零不能替代删路径后fresh学习，局部decoder可达不能证明整个架构易于学出正确表示；旧强v5.2/v6的同类FM能力要求实际解释架构/配方差距。§30及诊断报告中的语言“非统一解释”不能作为排除主要原因的结论；现有实测数值保留，尚未完成根因与修正验证。

Owner允许为探索修改架构和训练方式并进行实验性训练，限制在正式采纳候选并启动下一轮正式训练之前汇报。本次重新设立独立原因分析goal，具体计划见`docs/horizon_causal_learning_plan_20260909.md`，当前状态以progress顶部为准。

## 32. 验证下降集中于反复丢失BBQ，实例呈目标选择变化（2026-09-09）

原始/first-query/contextual三版200→400的train96分别52→59、46→59、41→49；validation110→87、110→103、103→90。BBQ分别24→3、28→10、25→1，扣除它后其它七task总成功86→84、82→93、78→89。因此不能将总分下降概括为所有任务同步遗忘；持续弱的Spatial/Goal23/Long32获取问题与BBQ保持问题需要分别解释。全部原contract、teacher/state/RNG实际配对复算保留于`causal_learning_20260909/existing_learning_and_retention.json`。

当前contextual200/400在固定states0/12/25/37的8条正常correct回放全部完成，200为3/4、400为0/4，8条均复现历史成功/失败，终态BDDL谓词一致。双相机轨迹显示200四例均操作正确BBQ（state0过晚运输而超时）；400四例均转向绿色干扰瓶，其中state25/37抓起并运到篮子区域，正确BBQ留在桌面。这些实例支持目标选择变化，反对把它们一概解释为抓取/运输能力消失；无法单独确定语言、过程表示或生成LoRA哪层导致该变化。

task因重复下降事后选定、states沿用旧诊断固定集合，属于描述性实例，不能作全局率、checkpoint选择或根因识别。原件、逐例解释及限制在`causal_learning_20260909/retention_replay/completed_summary.json`，完整索引见[当前因果计划](docs/horizon_causal_learning_plan_20260909.md)。

## 33. fresh联合路径删除有局部迁移收益，仍未解决主要能力与保持缺口（2026-09-09）

同一实际训练曝光下，保留local上下文、关闭H-read/Compiler额外条件的local_only，相对contextual在200/400 validation均提高；全部删除none未消除训练改善与验证下降。冻结后的依赖损失确实不能预言fresh删除后的学习效果。但联合删除还不能分辨两检索入口及其交互，不能把本结果统称为语言有害。

两条简化臂到400的train96均超过contextual，仍未超过上一版first-query-only；local_only验证总分较好却有Spatial归零、BBQ丢失和较大相邻churn。由此分开裁决已取得的总分收益、持续弱任务的获取和原成功集合的保持，不将局部修复当作v5.2/v6总体分差的解释。首轮全部1984条新闭环及实际配对见既有实验记录（docs/research_history.md及原始分析产物）与其总索引；下一实验及授权只按progress，不从本结论恢复执行。

## 34. 分离学习定位Compiler额外条件的贡献，H-read联合删除并非稳定最优（2026-09-09）

固定local开启的2×2 fresh学习全部完成。关闭Compiler在H-read开/关背景的validation效应，200为+5/+16，400为+36/+20；关闭H-read则为−5/+6和0/−16。方向一致的验证效应来自Compiler入口，不能把两入口都笼统判为有害，也不能把200联合删除较好外推到400。训练侧效应不同，局部拟合收益不等同迁移。

只关闭Compiler的local_h_read为validation108→126、train40→56，BBQ29→32（R/G/L23/9/6）；没有改FM、source、输入、共享或其它图就避免这次BBQ后期崩落。它是当前协议下具有实测收益的具体修正，尚不是全部缺口的解释：400 Spatial1/100，相邻验证仍26丢失；对all400虽净+36，也有52新增/16丢失。这里只支持该入口的学习效应，未证明唯一内部优化机制或全部v5.2/v6历史分差。完整因果表、配对原件和下一步授权见既有实验记录（docs/research_history.md及原始分析产物）及progress；复核结果不得从本段推定。


## 35. Compiler候选换视频复核保留部分保持收益，整体稳定性仍不足（2026-09-10）

固定seed7两臂两节点的另一套无放回正确视频关联已完整1600行。all200→400为99→82，local_h_read为101→126；候选相对基线200仅+2、400+44，方向同原关联，但不能把早期和末期效果混为一谈。候选BBQ26→31保留17/新增14/丢9，对照23→1仅保留1；说明避免这项后期崩落的效应不限于原teacher/state关联。

候选相邻总计71保留/55新增/30丢失，churn85/J=.4551，breadth6→4；Spatial task3的5个成功和moka任务的4个成功全部丢失。因此126及部分BBQ保持不是整体稳定或广度修复，独立初始化效应仍须其自身完整闭环确认。新关联复用原50个官方state及原视频资产，不是独立state池。全部strict配对、逐task/suite和worker证据见既有实验记录（docs/research_history.md及原始分析产物）与`compiler_confirmation/reassignment_validation_step{200,400}_analysis/`。


## 36. Compiler删除的验证净增跨初始化保留，但稳定保持修复未获支持（2026-09-10）

init11两臂固定200/400全部validation400完成：all100→108，local_h_read104→119；候选净增从早期+4到末期+11，400同初始化R/G/L86/33/22。结合seed7原关联+5/+36与新关联+2/+44，删除该额外Compiler仿射查询支路的验证收益方向得到复核；其幅度依赖初始化，不能把+36当固定效应。干预包含query_language的bias，仍不能将净效应单独归给文本内容。

保持结论必须收窄：init11基线BBQ27→5保留4/丢23，候选35→19保留15/丢20；候选自身总计67保留/52新增/37丢失、churn89/J=.4295、breadth7→6。基线大量BBQ丢失复现，候选在seed7的多数旧成功保持未复现；候选400相对基线多14个BBQ成功，不等于自身没有遗忘。候选Spatial仍1/100、Goal23为零，不能将这项局部因果贡献当成完整稳定方案或全部旧新能力差距的根因。

init11基线总分上升而BBQ崩落，其余七task73→103，再次要求区分任务获取与局部保持，不能用总分或训练FM定义统一退化。完整原始配对与两臂相邻证据见`compiler_confirmation/init11_validation_step400_analysis/`和既有实验记录（docs/research_history.md及原始分析产物）；后续原因工作与正式采纳授权仍以progress为准。


Compiler固定确认全部完成后，init11候选train96为44→61、相邻R/G/L36/25/8、breadth17→22；400相对all56为48/13/8。训练任务与验证总分都有改善，但未见任务保持不足的判断不变。完整12面板3584行与固定回放索引`compiler_confirmation/completed_confirmation_matrix.json`。输出rank独立性、读取侧可学习性和旧新配方不是由本次支路删除直接识别的因素；不能把本次正效应扩展为其它结构或训练因素已排除。

## 37. Compiler检索差异被共用位移压缩，净学习与保持须分开（2026-09-10）

已有all/local_h_read × init7/init11 × 200/400八个checkpoint的CPU选参诊断定位到第一Compiler的“语言位移相加→LayerNorm→Q投影”：all的608个槽检索差异幅度仅为同权重、位移置零代数反事实的3.5%–4.6%，同task query cosine约.9985–.9992；候选实际约.044–.055。语言位移以共同方向为主，显式affine bias本身不足以解释该程度；all在200→400的Q及共用静态语言读出漂移也明显更大。它支持具体查询组织机制，不证明真实attention饱和、选错帧、最终代码坍缩或该机制独自导致全部行为分差；残差、后续Compiler与独立D仍可能补偿，静态/上下文reader梯度也尚未独立分开。

Compiler两个候选200→400均有真实净学习：init7验证108→126、train40→56；init11验证104→119、train44→61。相邻验证仍分别丢26/37，保持未修复；两个节点不足以证明平台、已穷尽或再训无效。四条完整Compiler日志均未触发clip1，故裁剪不是该四臂差异的实测原因。停止新增完整训练来自Owner授权上限，并非科学平台结论。删除额外query仍是本轮最有实测支持的局部修正，尚未取得完整科学资格。

## 38. 当前D绑定改善后期训练获取，却未修复未见任务保持（2026-09-10）

同target跨rank、A/B两侧D绑定的完整400探索及四面板992行已结束。validation115→82，400低于all90；自身R/G/L59/23/56、breadth7→5，BBQ26→3仅保留3，Long早期13个成功全丢。train96却32→60，400高于all49；自身29/31/3、保留90.6%、breadth16→20。早期train代价反转、validation优势消失，不能把任一节点或训练任务的保持外推为稳定泛化。训练侧churn34主要来自31新增，不能当遗忘数量；验证侧较all更小的churn79对81也不改变净能力与breadth下降。

固定32条件×38target的200/400几何确认共享B/BA能量近单方向，而D_B自身首方向仅约39%–74%，到400还更分散；native输出32/256的目标也如此。因此形式rank16未被硬降、字典不只可表达rank1，集中发生于代码经U_B/GELU与字典的实际使用。它与train60并存，不能直接称有害缺陷或据此加rank损失；也不表示不同task输出相同LoRA。尚未分开C、U_B、D_B使用及Compiler检索之间的共同适应。

实际参数绑定同时改变方向字典、同target/side跨rank梯度求和与AdamW状态；不是纯减参或纯正则化实验。共享完整400日志同样未触发clip1，因此实际裁剪不是该效应的原因。当前结果不支持把这项绑定采纳为迁移/保持修复；否决仅适用于已测绑定、all背景、seed7和预算，不外推全部共享或未测Compiler×共享组合。历史Target-Owned负例保留，亦不能由本轮重解释全部旧新架构分差。

Owner要求直接在对话中说明原因、机制与建议，独立报告已删除；本文件保留跨轮结论。原件总索引`causal_learning_20260909/sharing/completed_matrix.json`，机制原件`causal_learning_20260909/mechanism/`。本轮分析交付后已停在正式修改/采纳及下一轮训练前；未识别的读取侧可学习性、meta-task数量、闭环状态覆盖和旧新协议效应不自动触发新实验。


## 39. 真实视频检索呈槽间广播，但任务条件仍存在（2026-09-10）

后续640次完整真实视频forward已完成：all/off两初始化及D绑定init7，各200/400、train24+validation8、每task固定两条既有teacher条件；无训练、held梯度或Test。真实P4和V有时间差异，例如all7_400 validation的P4时间scatter/RMS=.296、V=.285/.297，故不能把弱分工归为输入完全常量。all第一Compiler attention归一化熵=.922–.963，第二=.994–.997，属于宽分布，否定“softmax饱和到少数帧”的先前猜测。

Compiler-off在400将第一层真实Value读取的槽间变化/RMS由init7 .001917升至.051669、init11 .002258升至.045938；与原query压缩诊断和fresh闭环净增共同支持第一查询组织存在有害因素。第二层仍仅约.0031，最终同target各rank代码cosine仍>.9992。残差及self-attention/FFN持续叠加共用内容，第一层删除没有建立完整的输出槽分工；这一观察不证明分工不足是所有行为差距的唯一中介。

必须区分槽间相似与任务无关：all验证task间C变化占总RMS由init7 .408升至.535、init11 .454升至.592；每task两视频C相对差约7%–10%。Compiler-off的init11 C相邻漂移.355与all .351相当，不能再把静态Q漂移改善推广为整个视频代码稳定。冻结置零/仅bias/身份范数配平仍未消除最终rank相似，只测即时函数作用，不代表fresh修复。原件`mechanism/actual_video/{summary,decoder_kernel_summary}.json`。

## 40. D绑定把相似代码转为跨rank学习耦合，不等于减参自动正则化（2026-09-10）

当前实际B为独立映射b_r=D_r h_r，绑定后B=D H^T。固定H、只看D的局部普通梯度下降贡献，绑定的delta B=-eta G_B(H H^T)，独立则delta b_r=-eta ||h_r||² g_r。前者会混合rank梯度，后者保留rank独立方向；A侧有同类残差关系，但完整A仍包括A0。这不是对整个AdamW训练的等式，也不把局部贡献当最终行为。

全部640条件的小CPU复算显示，共享臂验证H_B Gram第一特征方向占99.951%/99.953%，与统一rank方向对齐>.99998；零和rank子空间的平均响应/共同方向约3.37e-5/3.17e-5。相同H下独立映射的对角核各rank强度最小/最大约.976/.980。因此高相似代码与D绑定结合会强烈偏向共同更新，补全了先前“D本身多方向、实际B/BA近单方向”的学习机制解释。该代码下的条件数现象可解释绑定代价，不唯一证明它导致val115→82；共享train32→60及更好保持仍是反证，不能据此追求秩数值或强加正交loss。CPU使用已保存真实C及小投影权重，FP32复算与native统计近似一致，无模型更新。原件`actual_video/decoder_kernel_{registration,rows,summary}.json`。

## 41. 共享reader的两支信用强抵消未获直接梯度支持（2026-09-10）

all两初始化200/400、8个预登记train task、每条件两组独立32query，共32条件/64组完整FM信用分解；保留正常前向和LoRA cotangent，分静态/上下文reader反传，未更新任何参数。全reader两支梯度cos范围−.01256至+.00443，近正交。400每suite一task追加已有Adam二阶矩固定度量，8条件/16组；信用cos范围−.01005至+.00391，预条件更新向量cos−.01714至+.00465，主要参数组也未显示强稳定抵消。

因此降低“同一reader两支信用相互破坏是主要根因”以及立即拆reader/PCGrad的优先级。它不排除历史其它task间干扰，也不证明所有时点无局部负向更新。两组查询的总梯度方向常不一致，样本与FM time/noise均在变化，不能将此直接当bug、平台或held遗忘原因。源码和实际记录另确认source冻结、Meta第二步后有真实梯度、无陈旧R缓存，五个400学习臂均未触发clip1。原件`mechanism/reader_credit/{summary,optimizer_metric/summary}.json`。

## 42. BBQ动作分叉主要经Q投影传递，删除Q不能恢复正确策略（2026-09-10）

已有all200/400真实轨迹的BBQ四state，在固定replan0/1/2/4/8/16观测和原噪声上重放all/off init7两节点、normal/Q零/V零/action-in-out零，共768个10-flow预测；无新环境交互、无梯度或训练。正常前5动作重建RMS=.00226–.00486，完整动作重建均在预登记.03内。

all200→400前5平移变化RMS=.15858；两节点均删除Q后降至.04430（.279倍），在完全相同初始观测上为.200倍；删V或IO没有同样效果。off平移变化也主要依赖Q。这支持变化先于环境轨迹分叉、并主要经Q相关功能交互传递；不能进一步宣称已经看见错误patch或把因果归给Q张量自身漂移。直接在400删Q相对正常200参考的动作距离却变为1.81倍、初始1.96倍，说明同时移除了有用适配，不能作为修复；参考动作也不是所有帧的专家真值。四state是沿用的描述面板，不外推全任务率。原件`mechanism/execution_projection/summary.json`。

## 43. 当前可行改进按证据分级，未识别因素不自动触发训练（2026-09-10）

有学习/闭环支持的局部候选仍是只去Compiler额外仿射query，保留exact language、local/H-read与视频Value、独立D；两初始化400净增36/11，另一正确视频关联净增44，仍未通过稳定/breadth/Spatial与>145资格。不能称完整解决，也不能称候选已平台。

更深入但尚未学习验证的候选是分离持久target/rank检索身份与已读取视频内容：当前第二层使用已吸收近共用Value的内容作为下一次Q，即使首层去额外语言也重新趋同。可在现有Compiler内让两层检索身份保持独立规范化/独立查询流，视频内容走独立更新流后解码；不先叠加模块、加rank loss或绑定D。它改变的是两层地址和内容的耦合，不能仅凭“Q-only”名称宣称创新或保证收益；旧Semantic-Address、Target-Owned及当前first-query-only的具体失败仍须保留。现有冻结配平只支持即时几何，未检验该完整候选的可学习性与闭环，未实现或正式采纳。

监督目标的可辨识性缺口仍存在：固定训练task的语言和正确视频都预测同一task动作，平均跨episode FM不强制学习视频动态增量；旧v5.2/v6同类FM较强，故不能推出必须换RL。若局部查询修正后的能力仍不足，值得单独裁决的是现成、审计排除held的non-held meta-task映射扩展，或train-only实际到达/失败邻域的可信动作监督；前者针对可迁移映射，后者针对执行状态覆盖，不能一次都加、不能制作人工任务绕过限制。冻结Gemma prefix是否限制细粒度目标/关系提取、这些数据因素和旧配方的贡献，现有非匹配历史与checkpoint不能唯一分开。监督信号须先可执行且有可靠行为参照，不复用已失败的固定噪声局部fit作为修复证明。

本次补充分析在完整训练上限内结束，结论直接在对话中交付；没有新独立报告、正式方法修改或新训练。明确保留未识别的全部历史分差及未验证的完整解决方案，不将诊断数量作为科学成功标准。

## 44. 四次真实更新已复现跨任务损害，历史动量不能单独解释该实例（2026-09-10）

从all7完整400恢复临时模型、AdamW、scheduler、sampler及rank0 RNG，仅做预注册401–404四macro；原四task各0.25权重、每task64跨episode FM queries保持，16条件共1024queries，观察的task0/19/22/34均未参加这16条件。单GPU串行是诊断拓扑，不是world4 formal exact-resume。完整274.20秒、峰值26.97GiB、exit0；总梯度norm .115–.174，未触发clip1，无source或held梯度。

四个预选原成功条件全部新重放成功。原400与同初态下四次零当前梯度AdamW（M，保留m/v更新、decay与scheduler）均4/4；四次真实更新FULL404为3/4，task19“把橙汁放进篮子”state32/demo48丢失。其余三例仍成功，因此这是实际跨task损害实例，不是整体退化估计；M不丢该成功，不能把这次损失全部归给历史动量。

使用M+(FULL−M)的P/C/D全八组合，橙汁在M、C、D成功，在P、PC、PD、CD、PCD失败。P含Meta、共享语言/视频reader及过程编码；C为Compiler及查询/时间参数；D为完整native decoder。该固定实例有两条足以造成失效的功能变化组合：P变化，或C与D共同变化。它反对把跨任务损害只归到一个查询入口；当前梯度经共享函数与模块交互改变另一条件的有效LoRA。FULL−M是四步真实梯度序列及其Adam状态变化的有限效应，不是固定度量的一阶梯度近似；混合状态未共同训练，不能把组合效应当成独立训练各模块的效果。

其它任务的混合结果也有补偿：D-only使双杯例失败，C+D恢复；P+C使酒瓶例失败，加入D恢复，而完整真实更新保留两例成功。不能单按混合失败宣布整个P/C/D有害。全部36行配对、源参数冻结、无受害task曝光与三worker exit0已核对；原件`mechanism/deep_causal_20260910/bounded_updates/{manifest,factorial_analysis}.json`和`closedloop/summary.json`。橙汁九臂带轨迹回放已全部完成、配对且逐臂重现原结果。normal/M/C/D都抓起橙汁送入篮子；P/PC/PD/PCD转而操作右侧BBQ瓶、橙汁留下；CD则仍围绕橙汁盒调整却未完成运输。故该实际完整更新的目标选择转移已由P变化的冻结替换复现，C+D还有不同的操作失效路径，不能把全部失败都叫视觉误选。照片和原始轨迹见`orange_replay/{replay_verification,visual_review}.json`与`contact/`；九臂第一步实际native图像/语言输入完全相同，首个前5步动作已不同，因此变化先于环境轨迹分叉；动作差异范数本身不决定成败。这里证明执行目标变化，不等于测得内部错误patch，也不解释全部held BBQ历史损失。

## 45. 200/400模块端点分解显示获取与丢失分散在模块及其交互中（2026-09-10）

八个固定train task、原teacher ordinal0、states32–35的P/C/D全八组合256行已完整，所有worker exit0、实际配对通过。每个P重新完整读取真实单视角视频及Meta，不拼接缓存R；P含静态/上下文reader，C含Compiler额外查询及其它Compiler参数，D含完整native decoder。全部是冻结函数替换，无模型更新。全200/全400为16→18/32，R/G/L10/8/6、churn14，不能由净增2掩盖六次丢失。

按P/C/D依次取200或400，000/001/010/011/100/101/110/111成功数为16/22/19/19/17/19/13/18。新D在旧P/C上可达22，而新P/C配旧D为13，配新D为18，支持decoder变化中有真实有用的功能调整，部分收益与读取/编译函数的后续变化和共同适应有关；不能将400性能概括为decoder普遍没有学会。与此同时000→001也丢5、新增11；task7从4/4变0/4，故该混合并非保持修复，也不是待选择的新模型。

六个全端点丢失条件对回退P/C/D的反应不同，其中task7/state32只有全200组合成功。与§44的有限真实更新互补，这定位到分布在读取、编译、解码及交互中的功能保持问题，不支持仅按某个模块漂移范数或Q相似度统一追责。混合状态没有共同训练，不能把冻结替换等价为分模块训练课程。全200新native结果与Q中介的缓存复算normal在32个配对状态上成功集合完全一致，降低该面板缓存精度重建改变基线行为的疑虑。原件`mechanism/deep_causal_20260910/endpoint_swaps/{manifest,factorial_analysis}.json`及`closedloop/summary.json`；已知held BBQ仍由独立预注册四实例面板直接检验，不从train8外推。

## 46. BBQ失效的端点因果贡献来自decoder、读取端及交互，不能只归Compiler（2026-09-10）

预注册sealed冻结validation task13的原四例state0/12/25/37、原teacher42/21/5/43，P/C/D全八组合32行及完整轨迹均已完成，全部worker exit0、配对通过。原全200/全400再现3/4、0/4及四例“正确BBQ→绿色干扰瓶”的原行为；全部case的初始实际native输入跨八臂完全相同，动作变化在环境轨迹分叉前已经出现。无held梯度、optimizer、checkpoint选择或新增state/teacher搜索；四例是沿用的描述面板，不是全局率。生成16.92秒、峰值10.45GiB，原件`deep_causal_20260910/bbq_endpoint/`。

固定P200/C200，仅把D换成400，states0/12/37都转向绿色瓶，只有25仍选正确BBQ并成功（1/4）。固定C200/D200，只换P400，states0/37转向绿色瓶，12/25仍正确完成（2/4）。固定P200/D200，只换C400，四例仍操作正确BBQ，成功总数仍3/4，但成功state0替代37。故这些实例中“读取正确视频/Compiler已变坏”不是decoder致错的必要前提：保持旧读取和旧Compiler，decoder变化已足以产生三个目标转移。也不能将读取端贡献忽略。

state25呈明确模块交互：P、C或D单独换400都仍选正确BBQ；D400与P400或C400组合后才转向绿色瓶。全八组合按000/001/010/011/100/101/110/111成功为3/1/3/0/2/0/1/0。全400只回退D可恢复state25，另外state12虽改回操作BBQ仍执行失败；因此目标选择修正与完整成功恢复必须分开。视频显示多种D400混合能抓起绿色瓶并运入篮子，说明这些失败不能统称抓取运输能力丢失。

这比“policy Q路径传递动作差异”进一步定位了生成函数：D是跨任务共享的代码到38-target LoRA映射，旧P/C保持也不保证该映射更新后仍生成选择正确对象的策略。§45中D400改善train8若干任务、此处却损害held BBQ，与§44中真实其它任务梯度损害橙汁一起，支持共享条件映射的获取与保持发生有害耦合。它不等于全部参数共享有害，不证明单独冻结D就是修复，也不从两个端点追责某次历史更新；当前Compiler额外入口的fresh删除收益仍成立，但“查询压缩独自解释BBQ后期转向”的强解释不成立。完整逐例目标判断与初始函数证据见`{visual_review,initial_function_comparison,replay_verification,factorial_analysis}.json`，公开结论不把视觉回放当内部错误patch测量。

## 47. 原all7模型的正确动态教学尚未形成稳定整体增益，视频作用局部且伴随成功集合置换（2026-09-10）

Owner本轮明确授权的单视角真实RGB视频对照已全部完成：all7固定200/400 × train24 × 9条件 × states32–35，共1728配对rollouts。正确demo46、另一正确demo47；同suite和跨suite错误视频按固定循环映射、保留目标exact language；真实frames乱序/倒序后重新完整native prefix+Meta+Writer；首/中/末帧分别重复到原长度，保留呈现时间位置，三种静态都报告。全部生成/worker exit0、跨臂与跨checkpoint实际teacher/task/state/env/policy RNG配对通过，无训练、held梯度或Test，不能充当正式qualification/checkpoint选择。

| 教学条件 | 200 / 96 | 400 / 96 |
| --- | ---: | ---: |
| 正确完整视频 | 41 | 51 |
| 同task另一正确视频 | 43 | 53 |
| 同suite错误task视频 | 43 | 48 |
| 跨suite错误task视频 | 46 | 46 |
| 真实帧乱序 | 42 | 47 |
| 真实帧倒序 | 42 | 50 |
| 重复首帧 | 45 | 52 |
| 重复中帧 | 43 | 51 |
| 重复末帧 | 43 | 48 |

400完整视频对错误视频有局部净增3/5、对乱序净增4，不能说视频完全无作用；但首帧/中帧/倒序均接近或超过完整视频，缺少正确动态过程的稳定整体优势。400同task另一视频为R/G/L47/6/4；同suite错视频39/9/12、跨suite错视频39/7/12、乱序40/7/11、倒序41/9/10、首帧39/13/12、中帧42/9/9、末帧37/11/14。静态首帧52不表示每个完整视频成功都可替代，它同时丢12、增13。正确200→40041→51为33/18/8，静态三臂也增长7–8；不能把正常训练收益全部归成动态学习。

Suite层面也有反向效果：400 Long完整/倒序/首帧=8/8/11（各24），Goal完整13而倒序/中帧/末帧均18；Object完整15比这些控制的11–12高。两条正确视频均成功的47个state中，只有1个在三种静态下都失败；该state（Spatial task4/state33）在全部七个错误/破坏条件均失败，说明存在局部视频证据增量，不能全盘否定。上述联合条件只是描述性逐state对照，不是部署union、新资格线或普遍时序理解证明。

架构与目标解释应据此收窄：当前R/P4虽沿真实视频链生成，却已经包含exact language的原生图文条件，图上“必须经过video Value”不保证有效LoRA必须依赖视觉动态。跨episode FM约束正确动作拟合，固定task的语言/静态场景也可提供可拟合线索。结果支持被测all7模型没有把动态教学变成足够稳定的必要增量，该面板未覆盖Compiler-off候选，补测见§49；仍不能唯一断言纯task记忆、完全语言独立解或冻结prefix无能力，也不能由此宣称必须换RL。未来真实学习比较需区分language/static prior与视频条件增量，而不是只让P4更复杂或attention差异更大。原件`deep_causal_20260910/video_controls/{dependence_summary.json,step200/closedloop/summary.json,step400/closedloop/summary.json}`保存全部逐task/suite、状态得失与阶段谓词。

## 48. 双层查询差异干预有局部功能作用，强制分工未成为即时修复（2026-09-10）

all7/off7 × 200/400、八train任务及原视频、states32–35的双层Q中介共640行完整，所有worker exit0与实际配对通过。仅对cross.query投影结果的槽间中心化偏差改RMS，保留当层共用分量和差异方向；L/H由既有train统计预定，不看结果选尺度。normal重建原C相对差约.0002；真实读取差异按预期改变，例如all400双层从约.0019/.0014增为.054/.0384，off400第一层压低后约.0013，说明操作确实触及目标机制。

| 模型与节点 | native normal | LL | HL | LH | HH |
| --- | ---: | ---: | ---: | ---: | ---: |
| all200 | 16 | 16 | 18 | 15 | 16 |
| all400 | 18 | 14 | 18 | 14 | 17 |
| off200 | 16 | 15 | 17 | 14 | 15 |
| off400 | 20 | 18 | 18 | 19 | 18 |

每格32个配对episodes，第一字母为第一层，第二字母为第二层。不能把native normal等同LL或HL：固定第二层为L/H时，第一层L→H在all200增加2/1、all400增加4/3、off200增加2/1，而off400为0/−1。故首层分工有局部即时功能贡献，但不是跨模型/节点统一增益。双层同时H在四面板均未超过native normal；off400即使压低第一层仍保留多数成功，查询差异的几何幅度不能当作能力刻度。

本结果降低“通过幅度恢复两层rank/target检索分工即可直接修复”的优先级，不能据此默认采用rank正交loss、更大Q尺度或持久地址流。持久地址/内容分离仍是未经学习验证的不同候选，并非被本冻结缩放完整否定；已有Compiler额外仿射入口删除的fresh验证净收益亦保留，但“共用位移压缩→该学习收益/BBQ保持改善”的唯一中介链没有成立。§44–47的真实更新、P/D端点和视频控制给出更直接的功能保持及动态增量问题。原件`query_mediator/{manipulation_summary,behavior_mediation_summary}.json`及四面板`closedloop/summary.json`；无新模型更新、held梯度或checkpoint选择。

## 49. 最强Compiler-off候选补测：正确视频的动态增量仍未建立（2026-09-11）

Owner指出此前all九臂没有覆盖当前最强模型后，明确要求先补完最强候选检查。本次固定`horizon_causal_local_h_read_seed7_20260909` macro400，即已有validation126/400的候选；不是all，也没有改为其它seed或选择新的checkpoint。完整复用§47的24训练tasks、states32–35、teacher46/other47、九种实际RGB输入变换与执行条件，共216套完整LoRA、864条closed-loop rows。此处每臂96是训练任务冻结诊断，不能与validation126/400或原先每state使用不同teacher的train96直接比较。

| 视频条件 | all400 /96 | 最强候选off400 /96 | 候选相对自身完整视频 保留/新增/丢失 |
| --- | ---: | ---: | --- |
| 正确完整视频 | 51 | 53 | 53/0/0 |
| 同task另一正确视频 | 53 | 60 | 47/13/6 |
| 同suite错误task视频 | 48 | 59 | 48/11/5 |
| 跨suite错误task视频 | 46 | 59 | 49/10/4 |
| 真实帧乱序 | 47 | 55 | 45/10/8 |
| 真实帧倒序 | 50 | 52 | 44/8/9 |
| 重复首帧 | 52 | 58 | 47/11/6 |
| 重复中帧 | 51 | 53 | 45/8/8 |
| 重复末帧 | 48 | 53 | 45/8/8 |

在本固定面板上，完整53、另一正确60，没有相对错误视频59/59、首帧58及其它静态53/53形成稳定的正确动态证据优势。删除Compiler额外支路的既有validation收益保留，但不能把收益解释成已建立必要视频动态。all→off的完整正确仅51→53（保留43、新增10、丢失8、churn18）；其它正确+7、错误同suite+11/跨suite+13、乱序+8、倒序+2、三静态+6/+2/+5，说明改善并不专属于正确过程，不能从该面板定责全部收益来源。

候选自身四suite按Spatial/Object/Goal/Long依次：完整16/17/14/6，另一正确16/20/17/7，首帧15/19/15/9，中帧15/17/14/7，末帧13/19/13/8；完整breadth20/24，另一正确22/24。Spatial完整比三个静态高1/1/3，但没有跨suite稳定优势。输入仍有实际作用：首帧条件丢失完整视频的6个成功、另增11；跨suite错误丢4增10；倒序丢9增8。不能据总数相近说逐state等效、完全不看视频、纯语言解或纯task记忆已被证明。

两条正确视频均成功的47个state中，没有一个同时在三种静态条件全部失败，也没有一个在两种错误视频下均失败；有5个在乱序和倒序下均失败。它们只是固定逐state描述，不能在部署中挑静态条件组成union，也不构成新资格门槛或普遍时序理解证明。单视角、每task两条正确视频及四个初态、冻结替换的分布变化和未训练language/static baseline均限制结论范围；init11及其它checkpoint未在本次补测。

生成使用候选原clean pushed detached45e16633、原生`backend_conditioning=local_h_read`，无事后hook替代；完整216条件332.95秒，peak10.508GiB。执行沿用§47的9abc9b95运行面；18个动态queue/persistent workers全部complete/exit0，最长1041.89秒。实际核对两模型216条件的language、donor/demo、帧数量、呈现及来源索引、像素置换、相机，以及864 rows的task/state/env/policy-noise配对，全部通过；216 jobs全部complete，无失败重试。只读模型，无梯度、Test、正式选点或方法修改；最终双节点无本任务tmux。

原件`runs/analysis/horizon_relation_writer_20260908/causal_learning_20260909/mechanism/deep_causal_20260910/best_model_video_controls/`保留结果前registration、GPU/launch记录、候选checkpoint引用、生成manifest、全部raw rows和`step400/closedloop/summary.json`；`comparison_with_all400.json`保存输入与执行配对、逐臂/逐suite模型差异和描述性逐state集合。新增实际791MiB，低于登记3GiB峰值预算。此前§47只解释all的边界已修正；本节才是对当前最强候选的视频检查结论。

## 50. 普通监督已有视频依赖正证据，捷径存在不足以解释当前失败（2026-09-11）

Owner在讨论[专家首轮回复](docs/review_materials/20260911/expert_review_round1.md)后强调：v5.2纯正常训练已产生明显正确视频依赖，不能把当前问题直接推成需要辅助loss；架构及学习路径必须进入核心解释。核对原commit `529da6bbe290f7393422937aa7cc278cee732107`的v5.2设计、配置、as_step与functional入口，训练为normal-order positive-only动作监督，无contrast/order辅助目标。step900五臂汇总为132/138/74/82/83（各400），原配置登记75,600动作queries、3,600单视频条件曝光。具体远程引用见[合并追问](docs/review_materials/20260911/FOLLOWUP_PROMPT.md)。

这支持普通监督能够学出正确视频依赖；它不证明旧模型已满足全部科学资格，不将历史400与当前train24诊断96混作配对差值，也不唯一归因到某个架构模块。旧v5.2 task-complete的120/109/107/111/124仍约束架构与recipe交互解释。“同task独立采样允许不依赖V的最优解”是监督识别边界，不能单独解释为何不同参数化/优化学到不同依赖，也不能据此判定纯FM做不到。后续讨论应说明真实视频过程如何被参数生成和执行消费；功能保持、任务共现或P4动作读出改善各自指标均不自动证明动态增量。未采纳新方案，科学执行继续暂停。

## 51. 双视频条件与新消费接口增加训练任务获取，未见迁移收益（2026-09-11）

R/C/S按同一普通FM、双独立K1条件、完整H与独立D，各fresh200更新/51,200queries/1,600条件。共有D初始化与读取Meta随机流一致，新增模块没有改变共有D的种子；函数类及新增模块的学习影响仍不能单独分离。

| 候选 | validation100→200 /400 | 200 Spatial/Object/Goal/Long | breadth100→200 | train96 |
| --- | --- | --- | --- | --- |
| R，原off消费+双K1条件 | 37→83 | 0/41/37/5 | 5→5 | 46 |
| C，语义后过程消费 | 43→50 | 3/29/16/2 | 4→6 | 49 |
| S，匹配无序帧集合 | 59→45 | 0/21/24/0 | 7→3 | 52 |

同51,200queries旧off200为108/400、40/96。R200比旧off少25，R/G/L63/20/45；train96旧→R为33/13/7、旧→C34/15/6、旧→S33/19/7。三臂增加训练任务分数，却没有得到未见任务净收益；不能把低validation概括为整图不工作。训练任务诊断采用独立teacher46–49及states32–35，无held梯度。

相邻100→200的R/G/L分别为R23/60/14（churn74、J=.23711）、C27/23/16（39、.40909）、S32/13/27（40、.44444）。C相对source47的100/200分别R/G/L9/34/38与18/32/29；早期Object获取伴随Goal丢失，200部分恢复Goal但Object下降。S100强于C/R的优势没有保持，不能据单点选择无序模型或再加时间模块就认为能够修复。

C/S不追加；负结果只淘汰实际消费/条件组合在该预算下的收益，不否定普通FM、全部有序过程或视频条件。R增长46但低于旧参照，尚不能判平台；后续有限R300/400节点按active design§8另行登记。全部六组400和三组96共2688rows完整exit0、canonical实际输入与执行配对通过。统一原件runs/analysis/video_consumption_20260911/first_round_summary.json，各arm/step*/的completed_summary、相邻/旧off比较和train96_summary保留细节；checkpoint、sealed banks、raw rows完整保留。没有Test、wrong/shuffle/reverse或held梯度。

## 52. 强旧候选继续监督仍增加训练能力，却失去未见任务能力（2026-09-11）

off7在原macro400=126/400、train56/96仍增长时曾因阶段预算停止，不能据此宣布平台。本轮按登记合同完整保留学习状态，物理三rank受控迁移后以原冻结45e16633继续200更新/51,200queries；保留探索性标记和lineage，不把它改称formal fresh。

400/500/600 correct为126/73/54。500/600四suite分别1/51/12/9与1/24/22/7，breadth均6；400→500保留60/新增13/丢失66，500→600为31/23/42，400→600为43/11/83（churn94、J=.31387）。未见任务能力继续下降，相邻稳定性未建立。

与此同时，train96从56提高到64（四suite20/16/16/12、breadth22），保留46/新增18/丢失10；独立动作FM400/500/600为0.104946/0.104097/0.104786。该实际配方仍学会训练任务，但训练侧拟合和获取没有转成稳定的未见task能力。它不证明全部监督已到平台、唯一根因是D/读取Meta，或加入保持约束/RL必然有效。

因此停止这条未改配方的off7续训，不登记700/800；原126和所有学习状态保留。两个400及一个96共896闭环rows完整exit0、source/normalizer/实际teacher与执行配对通过。原件runs/analysis/video_consumption_20260911/continuation/continuation_summary.json、step500/old400_vs_500.json、step600/{old400_vs_600,adjacent500_600,old400_vs_600_train96}.json。R的后续获取是另一个已登记对照，不由旧off的下降预先裁决。

## 53. R增加教学条件的后续获取未在300持续（2026-09-11）

按预登记§8从原200完整exact-resume，R300累计76,800queries/2,400条件，correct strict400=63，四suite0/24/34/5、breadth4。R100/200/300为37/83/63；200→300保留43/新增20/丢失40，churn60/J=.41748。Object41→24、Goal37→34、Spatial仍0；Long5→5但没有任何原成功保留，5个新增抵消5个丢失，总数相同不代表行为稳定。

相对旧off400126，R300保留51/新增12/丢失75、churn87/J=.36957；后者动作曝光不同，只作能力参照。该节点不支持更多教学条件持续改善迁移；最终等102400queries及train96比较须等登记的400结果，不提前宣称终点或因一次下降追加新loss/架构。400rows/42shards与9worker全部exit0，1206.51秒；原件runs/analysis/video_consumption_20260911/r/step300/completed_summary.json及两份canonical配对比较。

R400已完整结束，将§53更新为有限续训终点结论：correct85/400、四suite1/45/32/7、breadth6；300→400保留50/新增35/丢失13（churn48/J=.51020），200→400为53/32/30（62/.46087）。旧off400在相同102400 target queries下为126；旧→R85保留69/新增16/丢失57（73/.48592）。train96=56，四suite17/17/14/8、breadth20；R20046→56为36/20/10，而旧off40056→R56为43/13/13。故更频繁条件曝光没有提高终点训练能力，也没有改善未见任务迁移；停止R，不因300→400局部回升再延长。

独立动作FM400=.105399，24×128queries无梯度、289.10秒；新增训练200updates/51200queries完整6422.31秒，累计3200条件/383种task-video仍只有24个独立任务映射。400 val15worker/60shards798.74秒、train3worker/36shards888.14秒，496条件物化747.14秒；新增896闭环rows全部exit0，实际配对通过。完整原件runs/analysis/video_consumption_20260911/r/continuation/continuation_summary.json及step400/；该结论不证明全部普通FM或所有更大meta-task集合无效。


## 54. C学习到source以上训练能力，但正确过程无净收益（2026-09-11）

冻结C100/200六臂train24×states32–35、teacher46/other47全部配对：correct32→42、other31→48、同suite错27→46、跨suite错33→46、乱序29→45、首帧29→49；S30→51，source15。C200对source多27且task bootstrap区间为正，但correct对首帧少7，区间[-.125,-.031]；C100→200正确增加10、static增加20。correct四suite8/8/11/5→14/11/12/5、breadth15→17，相邻R/G/L20/22/12、churn34/J=.370。它仍在获取能力，两个早期节点不构成收敛证据。

固定八task的语义S，correct/错误过程/静态过程/zero过程/错误语义为15/18/15/14/21，各32；正确动态未产生稳定净收益。配对动作14336次预测中，C correct.112128、S.111931、source.152494；换static过程仅+.000044。静态首帧仍产生正常视频0.692–0.935倍中心化P4，证明过程Value包含时间/窗口响应，不能自动解读为视觉动态。此表示性质是具体修正对象，尚非全部失败的唯一原因。

采用局部无变化参照作为下一受控候选，保留C其余结构和普通FM；须经真实profile和fresh后续节点才能判断。完整论证、有限面板和路径替换边界见[机制复核§9–10](docs/video_mechanism_reassessment.md)。1472条新闭环及逐行配对、逐task/suite、success-set和task bootstrap在runs/analysis/video_mechanism_20260911/{behavior_summary,functional_comparison}.json及paths/behavior_summary.json。现有validation C43/50对source47未建立可信稳定source收益，EMBER目标与本次goal均未达成。


## 55. 无变化参照首段出现小幅训练视频分化，尚无迁移收益（2026-09-11）

只替换C局部关系更新的无变化参照，保持普通正样本FM、train24、2K1条件/task与32queries/条件，fresh训练200更新。固定train-only留出动作128queries/task配对通过，24/24任务FM下降，平均.151451→.109323；该动作证据不能证明有益视频因果依赖。

同一固定teacher46、states32–35的train100/200 correct=36/45（source15、旧C100/200=32/42）。100→200保留22/新增23/丢失14，churn37/J=.37288，不能把+9总分视为成功集合稳定。200六臂正确/换同task视频/同suite错/跨suite错/乱序/静态为45/42/41/37/41/40，各96；correct对两错/乱序/静态的task-bootstrap95%差额区间全部含0。由原C200的42/46/46/45/49转为小幅正差额是初步方向信号，正确自身仅+3，不能声称机制已修复。

200 validation strict400=34，source47、旧C20050；S/O/G/L=4/24/5/1，breadth5。相对source保留6/新增28/丢失41，churn69/J=.08；相对旧C保留18/新增16/丢失32，churn48/J=.27273。global task26 source41→2、task11 source5→24，再次存在任务间得失抵消。该节点没有source以上的未见任务能力；已登记300/400尚未完成，不能从首段宣称收敛或否定整个候选。

原件`runs/analysis/video_change_reference_20260911/step200/paired_summary.json`、`train_mechanism/step100`与`step200`、`first_segment_action_diagnostic.json`及对应formal输出。当前机制goal是可信稳定的correct优于错误视频且高于source；项目最终>145/400资格另行保留，不加为本次goal完成门槛。


## 56. 视频功能信用首段：辅助教师尚弱，训练获取不等于迁移（2026-09-11）

新表示的主方案与纯FM都完成fresh100/25,600queries，实际任务、video、query及RNG曝光匹配。主方案留出动作FM从.154849降至.116660，辅助reader仅降至.153163；纯FM学生从.154865降至.114877。当前辅助头未成为强功能教师，不能凭存在直接信用就断言表示已经学到有益过程，也不能把学生表现不理想归到编译容量。

首个主方案50 validation为69/400，配对source47；Object5→48同时Goal41→16，task-cluster增益95%CI[-.1700,.2975]，尚非可信广泛迁移。主方案100 train35/96高于source15，证明当前训练任务行为获取；两个配方的同曝光闭环、相邻及换视频尚未收齐，暂不决定辅助信用去留。无序视觉参照和最终sealed controls尚未执行，当前goal未完成。完整指标及success-set见`runs/analysis/video_functional_20260911/paired_summary.json`；动态后续仅看progress。

主方案correct validation50→100为69→72，纯FM为69→56；100主方案比纯FM多16，但task-cluster95%CI[-.0075,.0925]含0。训练correct则主方案26→35、纯FM30→41。主方案50换视频71与correct69相近；当前支持保留主方案作匹配frame_set比较对象，尚不支持辅助有效、视频必要性或稳定迁移资格。纯FM训练获取更强却未保持validation，再次要求将获取与迁移分开解释。

首段3968条闭环最终收齐：主方案validation correct/other在50步69/71、100步72/71，pureFM对应69/64、56/60。主方案100对pureFM的correct/other净增16/11，其task-cluster95%CI均含0（[-.0075,.0925]/[-.0225,.0775]）。主方案换视频较稳健（100 correct/other J=.7654），相邻correct/other J=.5495/.4947仍显示更替；首段不能作为视频特异性资格。采用主方案的匹配frame_set比较，bounded300提高曝光，不把早期共同策略或弱训参照当有益动态证明。


匹配frame_set首个50节点（2026-09-12）：有序main69/400、无序66/400，差额95%task-clusterCI[-.015,.0275]；训练为26/96、29/96，区间同样含0。100步两者固定留出动作FM .116660/.116912，均24/24任务改善且实际曝光匹配。现有正样本功能训练下，两类表示已获得近似的早期动作拟合与验证行为，不能把source以上总分自动归于有序过程；也不能用这个早期节点宣称充分训练后必然无效。main train100→200为35→45但Goal15→10，继续区分总获取、suite迁移和视频增量。后续只按既有bounded300与相邻配对合同裁决。


main200 validation74与100的72近似，却只保留36条成功、新增38/丢36，J=.3273；Goal/Long改善伴随Object下降、Spatial仍0。训练correct/other45/47的高重合（J=.8776）不能弥补validation相邻不稳。frame_set100与main100训练均35/96且31条成功相同，进一步限制了从早期总分归因有序过程的解释；继续等待预注册的充分曝光匹配。


100步匹配验证main72、frame_set73，差额95%CI[-.0225,.015]；连同50步69/66，有序差额+3/-1未保持方向。两个训练面板100步也均为correct35/other38。这些已完成比较尚不支持有益有序机制，不能由source以上的共同能力或训练换视频稳定代替；充分曝光200/300尚待完成。

### 2026-09-12：功能信用方案的后段迁移保持失败

主方案200→300训练correct45→43/96，validation却74→32/400，paired task-cluster95%差额CI[-.18,-.0325]；相邻保留21/新增11/丢53，J=.24706。训练获取近似保持不能证明未见task迁移保持；当前不能把新增执行query功能信用或新有序表示视为已解决此问题。同曝光frame_set及换视频结果仍待收齐，尚不据此指定表示、编译或优化单一根因。原件见`runs/analysis/video_functional_20260911/paired_summary.json`。

主方案后段退化亦在独立同task换视频映射复现：validation other200→300同为74→32/400，95%CI[-.1925,-.0225]；300 correct/other32/32，成功重合23。它不只发生在某一组教学视频条件上。训练other47→46/96，进一步说明训练获取保持不能代替迁移保持；未因此确认某个模块为根因。匹配frame_set200 correct69对main74，差额95%CI[-.025,.05]，仍未得到可信有序增益。

### 2026-09-12：辅助读出尚未形成更好的同批FM教师

已有日志的固定50-update窗口中，主方案151–200步source/reader/student FM为.153444/.149475/.106168，251–300为.157011/.148191/.102797；reader相对同批source有改善，但明显落后LoRA学生。frame_set151–200为.153444/.149457/.106165，与有序方案接近。因此当前没有同批动作拟合证据支持辅助读出已经是更好的蒸馏教师；这不是held动作或闭环指标，不能据此选点、证明无视频信息，或单因归因迁移退化。完整数值在`runs/analysis/video_functional_20260911/training_functional_fit_windows.json`，后续定位应保持辅助获取、编译传递与保持问题的区分。

现有首段预注册训练任务留出动作诊断（train24、每task32queries、无梯度）亦显示：100步main source/reader/student=.154849/.153163/.116660，frame_set=.154849/.153170/.116912；pureFM学生=.114877，其辅助关闭，日志中的reader/source零占位不代表实际误差。该已有证据支持辅助动作修正获取很弱，尚不足以将失败描述为“好教师已学会、只有LoRA编译失败”。汇总见`runs/analysis/video_functional_20260911/held_functional_fit_summary.json`。

### 2026-09-12：完整300步匹配仍未获得可信有序增益

40面板/9,920rows收齐后，main相对frame_set四节点correct增量+3/-1/+5/+5、other+4/-3/+13/+1，所有task-cluster95%CI均跨0。
frame_set300两臂也降至27/31，各自相对200的CI均严格负；后段退化并非仅有序表示或某组教学视频才发生。
当前检验没有解决有益过程及迁移保持，亦不能证明视频信息不存在；不将单项FM、差额方向或共性退化当作单一根因。
后续先检验固定表示下现有辅助读出还能学到什么，保持获取、编译传递与跨task保持三个问题的区分。

## 58. 固定表示读出可学，但弱教师与学生目标的张力已实际出现（2026-09-12）

在main200表示/source/Compiler全部冻结下，仅对辅助reader做固定1536支持queries、64epochs/384updates额外拟合。
support .150983→.136589（24/24任务改善），同train24留出768queries .149662→.140251（23/24改善），
留出平均降幅.009411、task-cluster95%CI[.006285,.012963]。原LoRA学生留出.110025、每task均更好，
最终reader超额.030226、CI[.023314,.038454]。因此当前接口确有可学习修正；不能断言视频无信息，亦无依据把
失败归为已学会强教师后仅编译失败。固定缓存额外拟合不是canonical阶段课程，不回写Writer或作为选点。

最终held跨task E替换误差.148876，原条件优势.008625、CI[.005880,.011906]（23/24task）；E同时包含语言、
静态与过程，替换整个E只能证明该接口依赖，不能识别raw-video内容或时序收益。精确报告在
`runs/analysis/video_functional_20260911/{reader_fit_diagnostic/results,reader_fit_analysis}.json`。

同一训练batch已记录的三个MSE给出代数恒等式`<S-y,S-T>=(L_C+L_D-L_R)/2`。main及frame_set的151–200、
251–300窗口各50/50更新为负，pooled输出梯度cos≈−.15，混合方向在真实FM输出梯度上的投影约为纯FM的.73。
这里只是预测空间的同权配对，不证明经native Jacobian及Adam后仍反向，不证明蒸馏无正则化收益，也不单因解释
validation退化。已有pureFM同时删除辅助表示信用和蒸馏，不能分开两种作用。下一项受控假设仅移除蒸馏，
保留辅助真实FM；不按此负内积推定改后一定更好。原件`distillation_output_geometry.json`，无新增GPU计算。


## 59. 去蒸馏改善训练获取，未解决迁移与保持（2026-09-12）

保持辅助真实FM、只令rho=0的fresh200比较全部完成。与原main在task/video/query/RNG/权重上逐条件匹配：
100训练correct/other42/38对35/38，200为55/55对45/47；200 correct配对净增10的task-cluster95%CI
[.020833,.1875]。这项训练收益不能被validation non-pass抹去，也不能据此归因于有益动态。

100 validation57/61对原72/71，200为72/67对原74/74。两个节点、两臂均没有正总分收益；200差额CI
[-.0875,.06]/[-.0925,.0425]。200 Spatial/Object/Goal/Long为3/47/5/17、2/47/6/12，存在局部
Spatial/Long增加，但Goal较main少15/12，source原41成功的Goal task6只剩5/5。去蒸馏尚未修复迁移。

自身validation100→200 correct的R/G/L32/40/25、J=.32990，other33/34/28、J=.34737；原main
对应J=.32727/.39423。200同task两组视频J从原main .66292变为.52747（48共同成功、24新增/19丢失）。
现有证据没有可信相邻或换视频保持改善，不以较低起点后的净增长制造修复结论。

本项不追加300或自动补rho0 frame_set。尚未做该配方的匹配无序对照，故不能断言它没有任何过程收益；
更不能从弱教师FM或预测空间负内积推出蒸馏是原失败的唯一原因。保留辅助FM的100结果又仅比pureFM多1/1
个validation成功，尚无额外辅助迁移信用的行为证据。后续机制判断必须同时解释训练获取、未见task损失、
历史普通FM视频依赖正例；不把该non-pass当工程bug或继续增加辅助头训练的理由。

本项8面板1,984rows，完整study48面板11,904rows；原件`runs/analysis/video_functional_20260911/paired_summary.json`，
限定裁决`auxiliary_fm/bounded_200_decision.json`，留出动作轨迹`auxiliary_fm/held_action_trajectory.json`。
没有selected checkpoint、Test或最终sealed controls；当前goal未完成。


## 60. 原生中层读取可学习，但未形成更强功能教师（2026-09-12）

固定main200表示与source，fresh同类hidden-residual reader在第10层输入归一化后读E，经后续冻结Action Expert
输出动作；32epochs/192updates，原1536支持queries复用、768留出无梯度。实际48条件及采样与旧动作头诊断匹配。
新头held .154897→.139792，source同缓存路径 .154875；平均改善.015083、task-cluster95%CI[.010621,.020128]，
22/24任务。16→32仍改善.004677、CI[.002959,.006544]，说明存在继续学习，不构成收敛证明。

但旧64epochs动作头同缓存重算 .140281，新头仅改善.000488、CI[-.003509,.003975]，15/24更好；原LoRA学生
.110005仍24/24更好，新头超额.029787、CI[.021072,.039954]。因此该具体中层消费函数没有形成更强教师证据，
不据此接蒸馏、追加训练或扫描层位。它不否定全部原生条件策略，也不能独立区分表示与函数类容量/优化。

最终held原E相对跨suite E优势.001718、CI[.001015,.002516]，19/24任务；该表示含语言/静态因素，不能推出
正确视频时序收益。两类辅助函数均有有限获取且明显弱于完整LoRA学生，继续把整体问题只归为弱教师或编译容量均缺乏依据。
原件`runs/analysis/video_functional_20260911/native_reader_diagnostic/`，配对统计`native_reader_analysis.json`；
只使用train24，没有deployment checkpoint、Test或最终sealed controls。临时入口由Git b1d3fa25保留后退役。


## 61. 单独开放teacher图文融合未改善有益迁移（2026-09-12）

在第59节mu1/rho0配方上，仅新增teacher PaliGemma 18层q/k/v/o rank4 VL Meta；共同模块初始化、
800条件/51,200queries与task/video/action/query/RNG/权重匹配，fresh200。Z直接路径和R经KV路径均真实反传，
完整重放与直接梯度一致，source trainable=0；本项没有观测到应按工程故障解释的合同违例。

训练100 correct/other34/37对参照42/38，200为56/56对55/55；200差额均+1且task-cluster95%CI跨0。
同checkpoint换视频200重合46、J=.69697，参照50、J=.83333。留出student FM200=.107352440，参照.107338454，
差额很小且CI跨0；reader=.149220013的微小改善不能代替行为增益。

validation100=61/61对57/61，差额CI[-.02,.0425]/[-.0325,.025]；200=62/64对72/67，
差额CI[-.0575,.0025]/[-.0375,.03]。100 correct Long+7未在other复现（−3）；200 S/O/G/L为
2/42/6/12、2/45/4/13，breadth5/6，source47只保留8/6。正确视频的局部收益不具跨视频、相邻一致性。

100→200 validation correct R/G/L32/30/29、churn59/J=.35165，other35/29/26、churn55/J=.38889；
相对参照J=.32990/.34737仅有小幅数值增加，correct保留数同为32、获取减少且breadth8→5，
不足以认定保持改善。200两视频重合44、J=.53659，差额CI跨0。

本项不延长或扫描VL rank/层位/LR，也不据100单臂+4自动补无序训练。只限定本干预、图和曝光，
不证明视频无信息、所有VL学习无用或优化已收敛；未训练匹配VL frame_set，不能宣称时间增量已被排除。
此结果与两个弱读出诊断共同说明：增加一条上游可学习路径并未兑现更强可迁移教学；继续只沿弱教师/单个冻结边界
解释整体失败缺少支持。下一个变化须有独立机制理由，不以参数微小差别或负结果本身触发数值调查。

新增8面板/1,984rows，study共56面板/13,888rows；原件`runs/analysis/video_functional_20260911/teacher_vl/`，
限定裁决`bounded_200_decision.json`及统一`paired_summary.json`保留逐task/suite、source、相邻、换视频与CI。
没有selected checkpoint、Test或最终sealed controls，当前goal未完成。


## 62. 从局部消融转向主假设裁决（Owner纠正，2026-09-12）

Owner指出连续数小时没有根本进展。问题不只是分数没涨：此前每轮都限定负结果的外推边界，却没有同样明确地
下调主假设的支持程度和追加投入的优先级；局部干预逐一结束，整体解释仍几乎原样保留。
科学上避免过度否定是必要的，但“尚未普遍证伪”不能成为研究上继续投入的充分理由。

当前核心预测是：保留任务token与完整H的有序表示，再给予执行query条件化直接功能信用，能学出比静态/无序
信息更有益、可编译且可迁移的教学关系。已有证据应合起来裁决这个预测：

- 原main与匹配frame_set在50/100/200/300均未获得可信correct有序增量，且200→300都显著退化。
  因而“增加这种有序表达与信用即可兑现过程收益”的预测没有得到支持；图接通、输入完整不能抵消行为负证据。
- 两种额外拟合的执行读出都可学习，却仍24/24任务落后完整LoRA学生。当前不应再把“已学出强功能教师，
  只差编译”列作首要解释，也不继续围绕同一弱读出追加拟合、层位或辅助loss。
- 去蒸馏确有训练获取收益，但未改善validation与保持；teacher VL适配亦未修复。
  因而“蒸馏或单个冻结边界是主要瓶颈，解除后即可迁移”的简单解释失去支持。

以上是对本候选及已检验补救方式的研究决策，不是对所有视频表示、功能监督或VL学习的不可能性证明。
历史v5.2等普通监督视频依赖正例及v6等局部能力正例仍须被解释；不能为了维护当前图而忽略它们，亦不自动恢复旧底座。

剩余关键解释至少包括：当前学习主要获得语言/静态条件下的共享动作修正，过程内容没有形成额外执行价值；
或者部分教学相关修正已形成，但其条件化参数作用不能在未见任务与相邻学习中保持。现有跨task E替换包含语言/静态，
无法区分这两种解释；训练FM、总分或参数差异也不能代替区分。当前尚无证据把其中一个命名为唯一根因。

下一步先复核现有证据分别支持和反对什么，再提出能够改变选择的最小诊断或实质修正。每项候选必须说明
竞争解释的不同预测、相对最近等价历史的实质差异、各结果对应的保留/修改/停止决策；若结果无论怎样都只会
导向“换下一模块”，先不运行。不默认以再跑一轮完整Writer作为诊断，也不为填表新增仪表或防御性检查。

第11节当前100检查点已生成，200及原配对证据仍在途；其结果尚未裁决，不用本节预先判失败。
完成这组已登记比较后，先进行上述综合裁决，不自动启动新配方或frame_set。即使本项涨分，也须说明改变了
哪条机制判断，不能仅借较弱参照宣布恢复。目标、信息墙、K1、sealed controls及既定资格门槛保持。

本次按该方式复核历史后，进一步排除三种看似尚未尝试的默认修复：

- **同预算增加K1条件已试过。** R在同4tasks/256queries/400updates、原图/LR/任务流下，把每task
  1×64改为2个无放回K1×32；条件1600→3200，unique task-video378→383，训练56→56/96，
  validation126→85/400。原件`runs/analysis/video_consumption_20260911/r/continuation/continuation_summary.json`，
  `7fedbe85:docs/video_consumption_writer_design.md`§3及`src/ember/writer/learning_data.py`。
  因此不能再把条件调用次数与独立任务映射混同，或仅因当前表示不同就重做条件数小扫。
- **早期recipe的正负交互不是单纯更多视频。** 按每task150次visit比较，v5.2 old900/TC150均3600条件，
  queries75600/72000、correct132/51；v6均3600条件/72000queries，correct95/111。旧v6按(task,visit)
  的3600条teacher已有实际一致审计；变化在4→24任务聚合、900→150更新、LR和flow RNG时间线。
  原件`runs/outputs/pi05_as_writer_v52_v6_recipe_matched_exposure_seed7_20260801/analysis.json`及
  `docs/horizon_k1_evidence_review_20260909.md`§4.1。支持架构与学习组织交互，不支持任一未识别配方因素自动成为修复。
- **输出更独立、更视频敏感也有负例。** `34be4a0`的Target-Owned Factor实际解除跨层硬共享，
  50/100/150/200仅99/76/86/68；扩大参数条件差异没有兑现行为能力。完整设计及裁决在
  `3a6f801d:docs/action_forecast_writer_target_owned_factor_design.md`§8。不再用参数几何改善替代实际执行价值。

早期50-episode视频与动作sampler没有显式同episode排除，旧日志也缺逐action-demo trace；因此旧正结果继续保留，
但不能把其训练合同说成与当前16/26分池逐条匹配，也不猜测碰撞率以解释成绩。未识别变量应标为未识别，
不能顺势变成下一轮默认实验。以上复核仅用已有源码、合同和分析，不新增GPU干预或使用旧sealed controls选架构。


## 63. 外部权重生成正例的适用范围（2026-09-12）

补充检索发现[WIZARD，arXiv:2606.07217v1](https://arxiv.org/html/2606.07217v1)直接研究语言/视频到冻结π0.5 LoRA。
其meta训练以专家adapter权重为监督，生成范围覆盖视觉、语言和动作模块，采用三suite训练、第四suite留出的划分；
因此不能把它的分数与EMBER固定24/8/8及38-target source policy作配对比较，也不能仅凭标题恢复旧expert-bank路线。
论文给出的模态消融支持视觉条件贡献，但未提供本项目所需的匹配有序/无序学习、相邻保持和最终顺序因果资格。

本项改变的是相关工作与证据覆盖：通用“语言+视频一次生成LoRA”已有直接外部先例；EMBER仍须兑现并证明
有益过程、完整唯一adapter及迁移保持，不能只以接口形式作为贡献。它也提示功能学习与权重监督须按真实合同比较，
不是发现一篇正结果就改loss。当前[公开仓库](https://github.com/Fascetta/WIZARD)列出网页与README，
尚未据公开训练实现复核其完整运行细节；这里保留论文级证据，不宣称已经复现，不启动新训练。

对外部正例补做最邻近历史核对后，**专家原始因子MSE＋方向／尺度监督并不是EMBER尚未执行的变量**：
`925e7b1:src/ember/expert_manifold/model.py`的真实目标为masked raw-factor MSE加cosine和log-scale项，
train24同初始化step2000 experts提供38-target完整rank16目标。原图macro50为48/400，topology address绑定后
fresh macro50为75/400（matched gained31/lost4、breadth4/8）；修复有真实收益，后者仍向跨task公共方向收缩。
完整执行与裁决在`3a6f801:docs/action_forecast_writer_video_expert_manifold_design.md`§24–29；两次只完成
预注册50节点，不能扩大为所有专家权重监督已被否定，也不能说这类loss完全没试过。

GOMQ则是成功专家occupancy上的函数蒸馏，具有强carrier、K4、fixed-A/B-only残差等不同合同，不能归为直接
专家权重回归的正例。其151→135→131及既有视频依赖证据均保留，不能只挑151或只写失败。另一路固定functional
code decoder后训练视频Writer，macro10的correct131/150与language-only130/150近似；后继仅decoder的实验
没有继续训练视频Writer，不能将它们重复计为端到端失败。对应证据索引见research_history§2，以及
`ac233fa0:docs/evidence/functional_adaptation_20260819/writer_macro10_inference_gate_20260820.json`。

旧直接重建用冻结16-phase视频特征、仅Action Expert38-target生成；WIZARD所述全视觉/语言/动作生成及suite
留出合同有实质区别，但这些区别没有配对隔离，不能据此认定补全模块或照搬loss会修复当前问题。
这一审计取消的是“尚未试过权重监督，所以默认改loss”的理由，不启动新路线。专家target有效、decoder可实现、
video能推断有益更新、该更新闭环有效且稳定，是需要分别有证据的接口，不能由前两项替后两项背书。

## 64. 视频预测预训练是不同知识来源，仍非已定位修复（2026-09-12）

当前及已审计后继主要将图文表征、Action Expert的原生响应与本项目内学习的时序模块结合；
保留完整H不等于这些表示已从观察视频学过物理变化。这个事实提出一种不同于增加头容量或调整辅助loss的候选解释：
用于理解教学过程的先验可能不足。但v5.2的正例仍限制该解释，不能将视频预训练宣称为理论必要条件。

[V-JEPA 2](https://arxiv.org/abs/2506.09985)通过无动作标签视频预测学习表征；其action-conditioned机器人规划版
另有DROID交互数据及在线规划合同，不能搬来替代EMBER一次LoRA部署。
[JEPA-VLA](https://arxiv.org/html/2602.11832v1)把冻结V-JEPA 2的最近两帧特征并入在线VLA，报告相对其控制器基线的收益。
这项证据覆盖执行观测表征与在线策略学习，不覆盖action-hidden教学视频到参数的编译，也不是固定24/8/8的未见task比较。
因此它支持区分“预训练知识来源”与“新增时序算子”，不支持直接预测EMBER换编码器就会成功。

本项只有文献与合同核对，未下载模型、构建新cache或启动实验，也未登记为active方法。
若继续考虑，必须明确额外冻结先验与fresh Writer边界、teacher侧与执行侧用途、前缀依赖，以及保留全部静态帧信息的
匹配参照，并在实验前写清竞争预测及停止条件。方向是否成立仍由正确视频闭环增量及保持裁决，
不以状态预测probe较好、模型更大或文献总分高代替；也不能因第11节负面而自动接上这个候选。

## 65. 纯FM/VL比较收尾：关闭当前主假设的局部修补投入（2026-09-12）

第11节只移除辅助表示FM，fresh200/800条件/51,200queries与teacher_vl逐字段匹配。训练100两臂37/40，
200为55/59；主要参照34/37、56/56，200差额CI[-.072917,.052083]/[-.03125,.09375]。
相对原蒸馏main的200为+10/+12，CI下界均正；这支持保留去蒸馏后的训练获取，不能把它单因归给最后一次去辅助。
200训练换视频重合52、J=.83871；这项正证据也不能代替过程与未见task收益。

validation100=60/55、200=64/54。主要参照61/61、62/64；100差额−1/−6的CI[-.04,.035]/[-.0475,.02]，
200差额+2/−10的CI[-.04,.0425]/[-.0575,.005]。200的S/O/G/L=2/46/6/10、1/41/6/6，breadth均7；
相对auxiliary_fm仍−8/−13，相对原main为−10/−20。100的Object增加伴随Long−8/−5；200correct的微增没有跨视频复现。

自身100→200 validation correct R/G/L38/26/22、churn48/J=.44186；other35/19/20、churn39/J=.47297。
相对teacher_vl的J=.35165/.38889，correct保留数32→38、other35→35，存在局部重合改善；但不能抹去迁移总分和
第二组视频收益缺失。200两正确视频重合45、churn28/J=.61644，correct−other=10的CI[.0025,.05]说明合法视频与
初始化配对仍影响行为，不能把它冒充正确对错误视频或顺序的因果增量。source47只保留7/8，广泛保留也未建立。

**综合后的决定比“本轮不通过”更明确。** 首版main与匹配frame_set在50/100/200/300的train correct为
26/35/45/43对29/35/46/48；other为29/38/47/46对24/38/47/44。各节点差额CI均未严格高于零，
validation亦未获得可靠有序增量。因此当前主候选连训练侧可重复过程获取也没有建立，不能默认它已经形成，
然后把下一步继续放在保持、编译几何或上游一个冻结边界上。

两个弱读出诊断、去蒸馏、VL适配和最终去辅助比较合起来，没有兑现“补直接执行功能信用即可得到可迁移教学”的预测。
本设计及其局部修补投入到此关闭：不续训、不扫辅助权重/层位/LR/rank/seed、不自动补纯FM frame_set。
这不是所有纯FM或视频表示的否证；第11节没有自己匹配的frame_set，不能把首版时序结果外推为该配方的直接时序否证。
旧v5.2和GOMQ的正例继续是对任何下一理论的约束，而非恢复旧实现的依据。

下一项可执行设计首先必须解释如何获得有益过程，给出与已试方式的实质差异及竞争预测；不能只承诺保留一个尚未证明
存在的过程信号。额外视频预训练目前只是§64的候选知识来源，未被采纳；不以新文献或新增模块替代这次理论责任。

本项8面板/1,984rows，study累计64面板/15,872rows；全部banks/checkpoints/raw rows/aggregate/completion保留。
原件`runs/analysis/video_functional_20260911/direct_fm_vl/bounded_200_decision.json`及统一`paired_summary.json`。
全部进程已退出，无selected checkpoint、Test、held梯度或最终sealed controls；目标未完成，当前无active design。

## 66. 局部变化的动作标签与跨episode功能标签是不同监督（2026-09-12）

下一项机制分析见[过程获取提案](docs/video_process_acquisition_analysis.md)。源码复核纠正一种容易混同的历史：
Action-Forecast v4预测未来动作；v5/v6使用固定probe响应；Stage0/G2的真实动作grounding仍来自另一episode，
按归一化进度取未来动作并组成phase标签。`c1493a1:src/ember/privileged_actions.py:61–110`直接拒绝
video/action demos交集，Stage0与G2各自sampler也明确分开。这些并不是对观察到的局部变化作实际动作反演。
所查历史范围内没有找到后者的完整执行链，不能扩大为全历史从未尝试。

由此推荐优先推导局部观察—动作配对对共享视频表示的学习作用，暂不同时引入额外视频骨干。
新增信息是标签与实际观察转移的对应关系；原辅助FM只是改变跨episode任务监督的读取路径。
该区别不证明反演一定可学、局部动作一定足以表示目标过程，或编译后一定迁移；v5.2普通FM正例仍反对
把局部反演称作视频依赖的必要条件。提案分别规定局部获取不成立、可解码但LoRA无收益、有稳定闭环收益时的不同决策。

提案保持teacher视频动作隐藏与主LoRA跨episode，但辅助分支拟对action池中的同episode RGB/动作学习，
需要Owner明确现有跨episode合同的例外范围。没有静默改sampler、训练目标或部署输入；当前无active design。
额外冻结视频模型也并非即插即用：普通整视频双向编码后的下游causal mask无法恢复前缀因果性，
编码后的无序token集合仍可能包含上游顺序；参照必须从实际输入与编码路径定义。来源与细节见提案§6。

## 67. 当前LIBERO观察为post-action，局部监督须按实际时间对齐（2026-09-12）

固定官方生产代码先env.step再保存RGB／proprio，而states与actions仍使用相同原索引。
本地train0/12/20/34各demo0的前11行obs关节均与下一行states相符，四个完整时间轴连续约20Hz且无next_obs；
检查未读取action数值、held数据或运行环境。具体生产来源、数据revision及数值见[提案§9](docs/video_process_acquisition_analysis.md)。

因此obs[p]→obs[q]的真实区间动作应为actions[p+1:q+1]。stride5末尾追加帧可能不足五步，不能用padding补成真实转移。
该结论直接约束拟议局部反演标签；现有主FM实际采用obs[i]/actions[i:]同索引，尚未量化其时间差的行为影响，
不能将它定为此前科学失败的唯一原因，也没有在这次只读核对中变更原训练合同。

## 68. 扩展任务映射已有直接历史，不能作为负结果后的默认修复（2026-09-12）

局部动作窗口收尾期间，重新核对“只有24个任务，所以未获得可迁移过程”这一竞争解释。95是曾登记的
71个审计LIBERO-90任务加train24总池，不等于每条历史路线实际对95个任务产生梯度；以下数字为不同task
spec的数量，也不等于已经证明相互独立、能区分操作顺序的教学映射。

- 冻结G2行为读出`5781694`实际训练56 meta＋19 target共75任务，另20任务零梯度；四类reader各1000步。
  held full/language梯度因子几何恢复约.2695/.2687，behavior-span oracle约.7160（`factor_cosine`，不是安装LoRA后的FM收益；见§87）。G2-B `5cbe76e`随后在相同75任务上
  共同训练表征，60macros/420更新/2280条件，held recovery最高约.294、终点.283，未获行为充分性。
  前者是冻结读出，后者保留跨episode动作/进度监督；都没有分别训练的ordered/frame_set Writer。
- 自然组合Writer `5534cb14`实际55 meta＋18 target共73任务，400步/4800条件/76,800 functional rows，
  每task65–67次；扩大了不同任务映射，不能说只增加同task视频。held5为30→32/250，rank-balance后继为45→35，
  carrier43。该图有G2组件初始化、冻结observer和rank12 carrier＋生成rank4，不能作当前fresh完整输出的单变量参照。
- 更接近当前输出合同的`041aff55`已经生成全部38-target rank16 A/B。其73-task组与`351feb48`仅18-target组
  都训练128更新、每target1024 action queries；后者的实际configuration audit确认只移除55个meta目标，
  保持target身份、采样、noise及optimizer；两组各自task等权，单个target权重实际从1/73变为1/18，
  每update组成及总queries也改变，不能据此单独归因映射数量或梯度冲突。四点validation screen80为15/19/19/19对17/17/20/16；
  终点18训练task诊断为42/180对55/180、breadth9对13。这支持特定共享训练组合的能力取舍，
  不支持更多任务自然改善迁移，也不能把screen80外推为strict400；两组均未训练匹配无序参照。

以上共同排除“扩展任务尚未试过”及“历史扩展只是更多同task样本”。它们未排除当前图的覆盖效应，
但也没有把覆盖不足识别为唯一或首要根因。旧覆盖审计确有同语言跨场景及gradient-to-held bridge，
仍存在具体过程缺口；统计task总数或重新训练小reader不能回答扩大覆盖是否增加有益的有序闭环差额。
仅凭本轮负结果、当前图不同或任务数较少，不恢复95-task，不把新的匹配训练作为开始理论判断的前置。

原件与完整适用范围：

- `runs/analysis/pi05_ecp_g2_behavior_sufficiency_probe75_20_5781694_gpu01p6_20260829/report.json`；
  `runs/outputs/pi05_ecp_natural_program_g2_behavior_fold0_m10_5cbe76e_gpu01p012345_r6_20260829/run_contract.json`。
- `runs/outputs/pi05_ecp_policy_response_writer_factorial_73task_k1_component_s400_5534cb14_gpu01p0156_cache8g_20260903/run_contract.json`；
  `runs/analysis/pi05_ecp_policy_response_writer_factorial_coverage_v1_20260903/report.json`。
- `runs/analysis/pi05_ecp_prw_complete_meta73_20260906/decision.json`；
  `runs/analysis/pi05_ecp_prw_complete_target18_20260906/decision.json`与`launch/configuration_audit.json`；
  更广行为诊断见[research_history§6](docs/research_history.md#recent-learning)。

## 69. 真实局部动作配对出现小幅预测优势，仍未形成有益任务过程（2026-09-12）

Local Action Grounded的ordered/frame_set两臂fresh200有界比较及全部16面板完成，登记闭环3,968rows。
每臂800条件、51,200主FM queries、800局部片段/6,400noise draws，实际task/teacher/query/局部片段流匹配；
这里800是条件曝光次数，不是800个独立任务映射。

| 节点 | train96 ordered correct/other | train96 frame_set | validation400 ordered | validation400 frame_set |
| --- | --- | --- | --- | --- |
| 100 | 40/35 | 39/38 | 45/48 | 58/53 |
| 200 | 46/49 | 42/45 | 30/27 | 30/26 |

validation100两视频有序差额−13/−5，task-cluster95%CI[−.06,−.01]/[−.025,−.0025]，没有suite净收益。
200差额0/+1，CI[−.015,.0125]/[−.005,.01]；四项train匹配差额区间也均跨零。
ordered validation100→200 correct R/G/L=21/9/24、J=.38889，other18/9/30、J=.31579；Long两视频均8→0。
frame_set相邻correct R/G/L=22/8/36、J=.33333，other21/5/32、J=.36207，Long14/9→0/0。
两臂共同退化，不能用更弱参照或某个suite的零星净增制造有益过程结论。

局部动作留出确有有序优势：100改善.00006528、CI[.00000521,.00012871]；200改善.00117193、
CI[.00090867,.00144457]且24/24任务正向。主FM相应差额−.00018689/+.00003742，区间均跨零。
因此本轮确实学到小幅局部有序预测优势，但未兑现“由此获得足够的任务过程并编译为有益LoRA”的预测。
局部FM优势属于共享读取器与专用局部头整体；后者拥有独立时间Key投影，不更新Compiler的时间投影。
不能据此认定共享E已经充分、Compiler是唯一失败点；局部机械运动、目标物/接触语义、长程组合及编译消费仍未分开。

与§62/65的旧功能读出、去蒸馏、VL适配证据合并后，应停止依靠更短信用路径或局部可解码性背书有益过程的投入。
当前局部监督配方关闭，不延长、不扫头容量/系数/LR/rank/seed；没有qualified checkpoint，不启动pureFM第三臂或最终controls。
该归因臂未运行，所以不能单因断言局部监督导致与旧配方的差异；v5.2等普通FM正例仍反对“FM或视频编译普遍不可能”。
本轮也不自动恢复95-task（§68）或采纳外部视频骨干（§64）：下一候选须提供实质新增的过程知识或学习机制及可辨别预测。

原件：`runs/analysis/local_action_grounded_20260912/bounded_200_decision.json`、`paired_summary.json`、
`step100/learning_comparison.json`、`step200/learning_comparison.json`；完整任务/suite及source成功集合均保留。
这是有效科学non-pass；当前goal未完成，未使用Test、held梯度或最终sealed视频controls。


## 70. 冻结视频先验得到单节点训练收益，未形成稳定迁移（2026-09-12）

Pretrained Video Grounded两臂fresh200及全部8个correct面板完成，共1,984闭环rows。
每臂800条件／51,200主FM queries，18个采样字段及world4曝光实际匹配；0/100/200各24-task留出诊断完整。
使用同一冻结V-JEPA2.1与当前完整native H／LoRA编译路径，主比较为过去四帧与逐帧四张重复图静态参照。

| 节点 | train96 ordered / frame_set | validation400 ordered / frame_set |
| --- | --- | --- |
| 100 | 41 / 41 | 53 / 48 |
| 200 | 52 / 40 | 33 / 48 |

训练200差额+12，task-cluster95%CI[.05208,.20833]，Spatial/Object/Goal/Long净额+4/+5/0/+3；
100差额0、CI[−.0625,.0625]。该单节点行为正证据不同于§65/69的辅助可解码性，不能用“从未学到任何有益过程”概括本轮。
它仍不证明共享E已经充分、只剩Compiler失败，也没有证明跨视频和相邻稳定的过程获取。

validation100差额+5、CI[0,.025]，下界未严格大于零；200差额−15、CI[−.0825,−.005]。
200有序相对静态保留24／新增9／丢失24，churn33，J=.42105，suite净额+1/−4/−11/−1。
有序100→200保留17／新增16／丢失36，churn52，J=.24638；Object33→17、Long5→0、breadth7→5。
静态100→200虽总分同为48，却只保留22／新增26／丢失26，churn52，J=.29730；Object32→21、Goal12→26、Long4→1。
静态200相对source47仅+1，CI[−.1175,.0925]；有序200比source少14，CI[−.1925,.0675]。
这些宽区间不证明唯一总体原因，但实际正确条件退化及资格失败已足以停止本候选追加投入。

主FM留出100有序改善.0002066、CI跨零；200差额约−.000000342、CI亦跨零。
主FM几乎无差额与train200行为正差额并存，再次限制用平均离线损失代替闭环的解释。
本轮停止的是冻结四帧先验与当前消费／编译／fresh200配方的组合，不是所有视频预训练或普通FM。
未做匹配无先验臂，不能把与历史局部辅助配方的差异单因归给encoder；原生image强静态参照与other未触发，
没有selected checkpoint或最终内容／顺序controls，不能补写未发生的因果确认。

关闭本候选，不续训、不扫层位／窗口／rank／LR／seed或辅助loss，不启动条件性other／image。
下一理论须同时解释本轮单节点训练正例、相邻与迁移负例，以及旧v5.2等正证据；不恢复已经否决的局部补丁链。
现有target/rank查询残差与共享D已有表示跨条件公共LoRA的容量；这不证明学出了好基底，也不自动给显式prior＋residual提供修复依据。
原件：`runs/analysis/pretrained_video_grounded_20260912/bounded_200_decision.json`、`paired_summary.json`、
`step100/learning_comparison.json`与`step200/learning_comparison.json`。冻结实现`611770d1`，八面板全部完整、worker exit0。
目标仍未完成；未使用Test、held梯度或最终sealed controls。


## 71. 先校正执行监督的时间对应，不能把旧蒸馏重新命名为新机制（2026-09-12）

本轮回到旧v5.2/v6源码与近邻干预：当前已有VL＋Action Meta，旧shared family heads与新独立D均有能力和迁移边界；
旧配方的架构×更新组织反向效果不支持默认恢复全任务batch、改D共享或增加同task视频。
`3a6f801d`的model/temporal/video_program及Target-Owned完整设计，与§62/70的实际闭环事实共同约束该判断。

“把监督移到真正的10步执行动作”也不是尚未试过的原则。`8553b613`的SEOD设计§4–6明确使用成功expert occupancy、
相同观测／noise的expert与student实际执行动作、unit-residual目标；GOMQ只打开memory-query梯度。
SEOD129→135→143→136、GOMQ151→135→131均未稳定。旧expert在离线B20上的flow audit仅2/24任务优于baseline，
因此不能把任何expert的任意离线预测直接当更可信标签。强carrier、K4、fixed-A/B residual及成功状态选择限制外推，
但足以取消“改成endpoint／专家目标就新增了科学信息”的默认理由；本次未启动新的专家或endpoint蒸馏。

一个更具体的监督语义问题已在§67确认：obs[i]是执行actions[i]后的观测，主FM却从actions[i]开始。
本轮修改前源码仍如此。此次只读train24的action16–41共624episode／107,825行，按task／episode等权，
相邻原始7维控制MSE=.0094648085，前6维=.0016271944，夹爪切换率=.0141226233。
差异主要集中在夹爪切换；这些原始控制量不能与归一化FM直接比较，也不证明它解释了全部旧失败。
原件`runs/analysis/pretrained_video_grounded_20260912/posthoc_execution_alignment_audit.json`，没有新forward或闭环。

[Execution-Aligned设计](docs/execution_aligned_writer_design.md)据实际post-action时序，将唯一Writer的主FM与动作留出
改为obs[i]/actions[i+1:]，最后无未来标签的观测不参与query采样。完整teacher末帧仍保留；source及normalization冻结。
这首先是监督时间一致性修正，是否使已有训练有序收益迁移仍待两臂fresh100/200有界比较。
新旧合法query支持不同，旧分数不冒充匹配训练反事实；不由旧强模型的存在否定正确时间对应，也不由修正正确性宣称目标完成。


## 72. 正确时间对应仍未获得相邻有序增量，转入固定正例复核（2026-09-13）

Execution-Aligned在冻结2ecf1770运行面完成两臂fresh200、全部8个correct面板／1,984rows；
每臂800条件／51,200主FM queries，配对采样、offset1、finite、完整checkpoint与全部worker exit0均通过。

| 节点 | train96 ordered / frame_set | validation400 ordered / frame_set |
| --- | --- | --- |
| 100 | 44 / 36 | 61 / 65 |
| 200 | 51 / 53 | 72 / 81 |

train100差额+8、task-cluster95%CI[.03125,.13542]；200差额−2、CI[−.08333,.04167]。
有序相邻R/G/L=36/15/8、J=.61017，静态28/25/8、J=.45902。两者训练能力提高，早期有序优势未保持。
validation100差额−4、CI[−.0325,.01]；200差额−9、CI[−.0425,0]。
200有序相对静态R/G/L=62/10/19、churn29、J=.68132，breadth4对6；S/O/G/L为0/62/8/2对1/67/12/1。
CI上界等于0不是严格负区间，也不能解释为已证明零效应；未获正向资格是明确的决策事实。

有序validation61→72：R/G/L=50/22/11、churn33、J=.60241，breadth5→4、Goal13→8；
静态65→81：55/26/10、churn36、J=.60440，breadth5→6。总分提高与视频增量及覆盖保持再次分离。
两臂200均超过source47，但有序对sourceR/G/L=13/59/34，task-cluster差额CI[−.1975,.3275]；
静态15/66/32、CI[−.1675,.36]。任务间效果高度异质，总体提高不能替代有益过程和稳定迁移。
主FM留出100/200的静态减有序差额仅.0000044525/.0000065973，task区间均跨零，不提供独立的有序读出优势。

正确obs[i]→actions[i+1:]监督合同保留。与旧611770d1相比的绝对分数改善只作历史描述：
合法query支持和训练执行不同，不是逐query匹配的offset反事实，不能单因宣称时间错位解释了旧失败。
本轮否定的是“该修正与现有配方在登记200步窗口足以建立可重复有序增量”的预测，不是所有FM或视频编译。
已有局部正例仍限制普遍否定，但当前证据也不足以认定表示已充分、只剩Compiler。
因此停止连续完整Writer训练和局部参数／模块补丁；不因总分上涨追加300或恢复旧课程。

条件性[冻结正例审计§6](docs/frozen_positive_replication_audit.md#6-条件性正确收益复核合同新outcome产生前登记)
在本轮新outcome前登记，现由完整non-pass触发。固定旧200有序52/96对静态40/96，
用train24×teacher46–49×states0–7的交叉面板检验该能力差额是否依赖原先四个对角video/state格点。
该实验不新增训练或编译，用1,728实际rollout区分稳定较大优势、跨视频异质性和未能复现／不确定性；
不直接定位表示／编译根因，不证明输入顺序因果或未见任务迁移，阳性不自动授权加模块，阴性不扩大为普遍不可能。
原始收益、所有分组与三种预定交叉统计完整保留；未通过不自动扩大预算。

本轮原件：`runs/analysis/execution_aligned_video_20260912/paired_summary.json`、`bounded_200_decision.json`、
`training_step200_pairing.json`、`diagnostic_step200_comparison.json`及8项completion。
全部checkpoint／LoRA／raw rows保留。没有selected checkpoint，未触发other／强image参照、最终controls、Test或RL。
整体goal未完成；后续活跃状态以progress为准。


## 73. 冻结200正例的跨条件复核未通过；任务适应能力与有序增量须分开（2026-09-13）

按新outcome前登记的[复核合同§6](docs/frozen_positive_replication_audit.md#6-条件性正确收益复核合同新outcome产生前登记)，
固定旧611770d1 ordered200与frame_set200，对train24×teacher46–49×states0–7完整交叉评测；
两模型各768，source单独192，共1,728次实际rollout。零新训练／Writer调用，复用192个原LoRA，
source广播只用于构造等权差额，未增加样本数。初始化对固定模型是新的，教师已开发，不是新task或新训练seed。

| video | ordered /192 | frame_set /192 | 有序净增 |
| --- | --- | --- | --- |
| 46 | 90 | 90 | 0 |
| 47 | 93 | 89 | +4 |
| 48 | 89 | 87 | +2 |
| 49 | 87 | 95 | −8 |
| 合计 | 359/768 | 361/768 | −2 |

整体有序−静态为−.26042pp。预注册seed20260912、20,000次配对bootstrap的区间为：

| 口径 | 有序−静态95%CI（pp） | 有序−source95%CI（pp） |
| --- | --- | --- |
| 固定24task，task内video/state分别抽样后交叉 | [−4.03646,+3.51563] | [+23.82813,+36.32813] |
| 再按task重采样，内部仍为交叉结构 | [−4.68750,+4.29688] | [+18.88021,+41.66667] |
| 只重采样task均值 | [−2.73438,+2.21354] | [+20.96354,+39.71354] |

三种有序−静态区间均跨零；总体点估计非正、四video并非均正，预注册的跨条件优势判据未通过。
在登记的交换性与重采样假设下，新面板不支持旧12.5pp量级的平均增益，仍未排除小幅正负效果。
旧52/96对40/96仍是其原四个video/state配对格点的有效事实；新旧state不同，旧面板不交叉且阳性受多轮开发后关注影响，
不能唯一归因初始化、video或训练随机性，也不能把CI跨零写成零效应或普遍不可能性证明。

source32/192=16.67%；有序46.74%、静态47.01%。有序−source+30.07813pp，三种CI下界均严格正，
说明已训练的条件化系统能在新初始化上产生有用LoRA；没有证明这种能力必须使用视频动态，也不证明Compiler已充分。
这是正证据与负证据的分界：整体任务适应能力保留，而额外有序收益未在新的条件面板建立。

有序相对静态R/G/L=316/43/45，churn88、J=.78218；24task净额8正／6零／10负。
S/O/G/L有序114/84/104/57、静态118/87/102/54，净额−4/−3/+2/+3；source各suite实际14/0/16/2，共32/192。
两模型breadth为22/24、21/24。192个task/state中，四条video均成功74/69，至少一条成功107/109；
正确video两两成功集合J分别.782–.853、.750–.822。完整分布保留，不能以单项all-video成功数提升替代增量判据。

### 对下一步机制判断的约束

- 现有证据支持“学到了可跨初始化使用的任务适应能力，但有序组织没有可信附加收益”这一现象描述。
  它未隔离语言与静态视觉贡献，不能直接定性为某一种输入捷径。
- 时间对齐两节点的有序优势不保持与本次固定正例不复现，共同降低继续同族完整Writer训练的投入价值。
  不默认通过增加任务、rank、训练步数、读出头或共享／独立D再试一轮；也不能由此否定早期v5.2/GOMQ在其条件下的正事实。
- 表示是否充分、Compiler怎样消费与跨task更新竞争仍未唯一定位。原生H、辅助可解码性或有用LoRA的存在，
  都不等于特定接口已经解决；下一项需要可使竞争解释产生不同观察的预测。
- 等待期只读核对了教师prefix：当前native.py使用真实RGB与`Task: ...; Action:`，不含teacher proprio/state；
  execution processor则使用真实机器人state。这是信息墙下已规定的输入差异，历史3a6f801d的feature_cache／live_adapter／v6 runtime
  采用同一Pi05TeacherPrefixTokenizer；不是本轮新出现的工程缺陷，也不足以自动启动state估计模块或宣称唯一根因。

按原预算关闭本复核，不追加面板、完整Writer训练或参数扫描，不运行最终controls、Test或RL；无selected checkpoint。
当前回到综合理论与可识别性判断，先给出新增机制证据、近邻历史区别与可失败预测，才登记下一项合法最小实验。
整体goal未完成，实验完成本身不能完成目标。

全部1,728行及固定model/video/task/state/env-policy seed与共有noise序列核验通过，9面板全部worker exit0，
累计评测墙钟4089.738秒，controller已退出。source首个prepare在GPU模型／rollout前退出，
既有registered-subset入口增补screen8后116项相关测试通过；source用clean pushed c5a28747冻结运行面，
其评测路径与611770d1仅差该准入行，两个Writer臂仍用原611770d1。原失败与正常完整证据均保留，无重复rollout。

原件：`runs/analysis/frozen_positive_replication_20260913/REPLICATION_READOUT.md`、`paired_replication_summary.json`、
`replication_decision.json`、`study_contract.json`、`mapping_provenance.json`及同名outputs根的9项raw rows／completion。

## 74. 多状态视频、显式顺序与共享学习的可识别性边界（2026-09-13）

完整推导、竞争预测及停止分支见[视频信息与可识别性](docs/video_information_identifiability.md)。
本次只读源码、train24 specification和既有outcomes，未调用新模型或使用最终controls。

- `frame_set`逐个读取整条视频的全部采样画面，取消局部跨帧证据与显式时间路由，保留全部T×L内容。
  它不是单张静态图，也不等于语言参照；有序／frame_set比较是两种独立学习系统的总效果，不能直接等同固定模型的顺序干预。
- 审计24个训练task：19个单原子、5个多原子合取，On17/In10/Open1/Turnon1/Close1。实际成功检查以当前
  接触、位置、区域与关节条件为依据，不记录目标完成的历史顺序。物理可行性、遮挡和前置条件仍可让过程知识有用；
  不从终态目标推出视频时序无用，也不修改benchmark制造顺序奖励。
- 原9个净正task贡献全部+12/96，同task在新交叉面板合计−1/288，4正／1零／4负；其余15task净额−1/480。
  这是事后描述，反对“原受益群保持、仅被别的task抵消”的事实描述，不能唯一归因训练seed或选择噪声。
- 当前独立video/query采样下，给定已知任务T，期望FM为E_V[R_T(W(L_T,V))]≥inf_w R_T(w)。
  若函数类可实现逐task最优LoRA，同task恒定输出允许达到下界；采样本身不强制个体视频差异。
  实际函数类、优化与未见task知识均不由此解决，不能把任务ID可由语言识别误写成视频不可能帮助学习或迁移。

本次明确比较语义与学习压力，未唯一定位剩余瓶颈。历史v5.2固定模型的视频依赖与本轮独立训练臂打平
是不同测量，可以同时成立；不抹去前者，也不恢复旧架构。现有读出与跨条件复核不足以支持“已有强过程知识、
只差Compiler／保持”的单因解释。下一机制须说明它增加了何种影响闭环的知识及怎样被合法证伪，
不因本节分析而自动启动完整Writer、追加probe或新controls；整体goal保持未完成。

原件：`runs/analysis/frozen_positive_replication_20260913/conditional_information_audit.py`及同名JSON。

## 75. 局部有序优势延续到生成误差，但现成动作估计仍不如任务均值（2026-09-13）

按新增outcome前登记的[冻结生成诊断](docs/frozen_local_action_decode_audit.md)，固定原local两臂step200，
复用train24×16个diagnostic42–45片段，从纯Gaussian噪声经10步Euler生成15×7动作，每片段8次。
真实动作不进入生成过程，只在输出后评分；全部768片段完整、两臂exit0，无梯度／新LoRA／rollout／validation或test动作。

| 8次生成均值的动作MSE | ordered | frame_set | task mean |
| --- | ---: | ---: | ---: |
| train24等权 | .31625053 | .31676502 | .25059744 |

task mean−ordered差额−.06565309、task-cluster95%CI[−.08381484,−.05098097]，24/24task均不如均值。
frame_set−ordered差额+.00051449、CI[+.00032845,+.00070563]，21task正／3负、四suite净额均正。
所以小幅有序优势不只出现在带噪标签的FM中，亦出现在固定生成协议的误差上；但未满足两个参照同时通过的前提。
独立训练两臂仍不能唯一识别固定模型的顺序因果作用，不能仅因相对无序为正而称已有强动作教师。

另以训练action16–41的98,465个合法15步窗口计算episode等权Gaussian矩，解析预测带噪动作FM；
24task留出平均.26389586，而旧两臂.81011446/.81128639，24/24task均落后这一无视频去噪参照。
解析参照使用训练task身份与noisy actions，是离线诊断，不是合法部署字典、语言模型或无视频闭环policy。
384个step0零输出loss重建核对通过；不把这类MSE或内部loss替代最终行为。

停止把本现成局部读出直接接入参数生成的提案依据，不增加noise／Euler步数／训练／头容量。
8样本均值误差仍含采样误差，不能外推所有逆动力学失败或表示毫无知识；也没有唯一归责E或Compiler。
解析参照使用全部授权训练池，曝光不与原800个局部片段匹配，不作为同预算算法比较或唯一根因证据。
下一机制仍需解释如何获得并利用可迁移的操作知识，不能从“可解码”直接跳到“只差编译”。整体goal未完成。

代码冻结017430b3，模型原5f4f440c；每臂诊断循环约98.3秒、allocated峰值10.155GiB，
有限入口已从active scripts退役。完整原件：`runs/analysis/local_action_grounded_20260912/inverse_action_READOUT.md`、
`inverse_action_summary.json`、`inverse_action_marginal.json`、逐臂raw/samples及`inverse_action_launch_contract.json`。

## 76. 点对应提供运动归纳偏置，尚未补上跨初态到参数行为的联系（2026-09-13）

源码与原始文献审查见[可识别性分析§7](docs/video_information_identifiability.md#7-显式运动对应能补什么以及为什么尚不足以启动新writer)。
当前任务token视觉读取／native H端点读取没有显式同物理点约束；不因此否认其隐式运动知识。
跟踪可见表面与生成具体7维机器人动作是不同问题，旧局部动作头失败不直接否决前者。

Im2Flow2Act等正证据同时依赖目标物体绑定、执行状态对应或动作重定向，不能只抽取“flow有效”来支持新Writer。
固定运动先验仍是RGB的函数；点对应能改善可用表示，不自动解决跨演示初态的操作关系及唯一LoRA的行为传递。
仅评估光度warp或展示轨迹图无法区分获取不足与编译不足，也可能只反映背景／机器人／相机运动。

当前不采用直接换成／追加dense-flow编码器、其余跨episode FM与Compiler不变的提案；不启动只验证跟踪后
必然进入Writer训练的probe链。目标物体运动仍保留为候选，重提时须给出可改变决策的跨初态功能判别，
并区分感知失败、任务相关性不足与编译传递失败。尚无新实验design、权重下载、forward或闭环结果；goal未完成。

## 77. 冻结source在state-free输入下已有动作生成能力，补state的预登记前提未通过（2026-09-13）

[预注册输入诊断](docs/source_state_input_audit.md)固定train24×16位置，逐臂8个配对噪声、10步原生采样；
三臂共1,152位置完整exit0，无Writer、Meta、LoRA、梯度或环境。仅以obs[p]预测actions[p+1:p+16]，
相机、标签和noise保持；均值state来自train16–41、真实state只是train-side执行查询的离线oracle，不是部署输入。

生成均值MSE为free .13672735、mean-state .14173378、true-state .12469236，任务动作均值 .25059744。
free−true差额+.01203498、task-cluster95%CI[−.00247592,+.02751481]，16task正／8负；
mean-state−true差额+.01704142、CI[+.00965492,+.02384471]，21正／3负；action-mean−true的CI也严格为正。
第一项未通过，按原合同停止状态补全提案依据。不能把状态相对均值确有作用抹去，也不能把缺state称为已定位主因。

描述性追加计算保留另一项正证据：action-mean−free为+.11387009、区间[+.08481614,+.13969462]，22正／2负。
合法state-free RGB＋语言输入足以让冻结source完整采样优于该任务均值；这个新事实不依赖训练新读出头。
但它没有分离语言／视觉贡献，没有验证顺序必要性，也不能推出time1单步H、经过学习的Meta/E或Compiler已保留这些知识。
此前局部逆动作头看四帧、本项source看首帧并预测未来，不能把两个MSE差额直接当作同输入模块替换效果。

下一项先检查原生完整采样与单步响应的具体关系及旧forecast失败假设；不因本正数自动扫描flow阶段、层位或重训Writer。
每臂约311.18秒、allocated峰值9.88GiB。code6fa6177f、原始rows/samples、summary、READOUT和launch contract
在`runs/analysis/source_state_input_20260913/`完整保存，临时入口退役。整体goal仍未完成。

## 78. 原生动作读出的主要差异随相机范围改变，增加flow深度没有补偿（2026-09-13）

[预登记端点×视角诊断](docs/source_endpoint_readout_audit.md)先确认当前execution-aligned observer为agentview，
而§77 source输入实验为dual；两者不能作单一采样阶段比较。固定同384个train24位置、exact language、
state-free输入、原八噪声与15×7标签，新增agentview/full10、agentview/t1、dual/t1，完整复用dual/full10。
两个t1均另报告原observer seed1729唯一公共probe。全部新增1,152位置exit0，无Writer、Meta、LoRA、梯度或rollout。

MSE：agentview full10/mean8/public=.36548145/.34788516/.34731886；
dual full10/mean8/public=.13672735/.13217241/.13176813；task mean=.25059744。
agentview三项相对任务均值改善的95%CI均严格负、四suite均负；dual三项均严格正、四suite均正、22/24task为正。
在相同读出下，dual相对agentview的改善分别+.22875410/+.21571275/+.21555073，
全部24/24task为正，95%CI分别[+.20100961,+.25719621]、[+.18717154,+.24534044]、[+.18742479,+.24470501]。

agentview从t1增加到full10没有改善，反而MSE增加约.01760/.01816、区间均正；dual的full10也未优于端点，
mean8差额区间跨零而public差额略正，二者全部保留，不选择新的probe或一步rollout采样器。
按合同触发视觉范围分支，停止把增加denoise深度作为当前首要修复投入依据。

在双相机条件下，t1完整H经现有原生头已有动作预测价值；不能把这一正数外推到当前单相机或适配后的Meta／E。
结果尚未分开腕部内容、source训练分布匹配及其交互，也没有测视频动态、跨初态参数编译或闭环因果收益。
这是具体输入效应，不是现有所有下游失败的唯一根因。下一机制仍须证明这项知识能进入唯一LoRA并带来保持及迁移。

原生双视角支持见99ee2d03与Horizon设计§3.1，不能把接口实现当作已完成性能实验；旧双视频K比较不是双相机比较。
当前V-JEPA合同限制agentview，新增输入须明示与该先验的关系，不静默更改已有run配置或冒充exact-resume。
范围内1,669份run contracts中38份具有内嵌observer，均显式／缺省agentview；其余合同不能据此判定，
不把该有限库存审计当作全部历史未做双相机实验的证明。逐份路径和状态保存在endpoint_camera_contract_audit.json。

代码e1a06b46，loader6fa6177f；三臂约302.89/248.95/249.84秒，峰值9.88/10.04/10.04GiB。
raw/samples、完整逐task／suite、所有配对差额、summary代码、READOUT与launch均在
`runs/analysis/source_state_input_20260913/endpoint_*`。临时active入口退役，原始冻结代码保留；整体goal未完成。

## 79. 原生双相机读出改善未转成当前Writer的稳定迁移（2026-09-13）

[Native双相机设计](docs/native_dual_video_writer_design.md)的两臂fresh200及8面板全部完成，1,984行，
source／V-JEPA冻结，唯一变化是native增加同步wrist，prior仍agentview。四模型800条件的18个采样字段匹配；
全量task/state/language、环境与policy RNG、真实teacher帧索引及跨相机映射核对通过。validation各task50视频无放回，
train96为登记有限池；全部worker exit0，冻结代码defcf734，累计评测墙钟4417.77秒。

| 节点 | train ordered / frame_set | validation ordered / frame_set | validation有序差额95%CI |
| --- | --- | --- | --- |
| 100 | 36/96 / 32/96 | 64/400 / 56/400 | [0,+4.5]pp |
| 200 | 54/96 / 50/96 | 53/400 / 39/400 | [−.5,+9.25]pp |

两节点有序点差额+8/+14全部保留，不将区间未通过误写为有序无效。100有四suite净正；
200 S/O/G/L有序0/52/0/1、静态1/37/0/1，仅Object净正。breadth100为6/4，200为3/4。
100有序相对静态保留51／新增13／丢失5、churn18、J=.73913；200为35／18／4、churn22、J=.61404。
两节点CI下界均未严格>0，200也未达至少两个suite净正。按预登记关闭，无条件性other／image资格或模型选择。

有序相邻64→53，保留41／新增12／丢失23、churn35、J=.53947、差额CI[−7,+.5]pp；
静态56→39为32／7／24、churn31、J=.50794、CI[−9.75,0]pp。train两臂却都净增18。
这支持本组合存在训练任务适应与held迁移分离；仅凭两节点不能证明唯一的过拟合或优化机制，不能默认早停解决。
相对source47/400，有序100/200总数+17/+6，但task-cluster区间分别[−20.5,+27.75]/[−27.5,+26.75]pp；
200仅保留source的5次成功、新增48、丢失42，churn90，不是原能力普遍保持的证据。

预登记同模式相机比较：ordered validation单→双为61→64、72→53；frame_set为65→56、81→39。
100差额CI分别[−2.25,+3.5]pp、[−4.5,−.25]pp；200为[−9.5,−.75]pp、[−20.25,−2]pp。
相机×顺序交互为+3/+5.75pp，CI[+.75,+6]/[+.25,+13]pp均正，是必须保留的正事实；
200同时表现为两臂绝对能力下降且静态损失更大，不能由正交互推出更好的有序policy或成功转移局部动作知识。
这是固定seed训练系统比较，非固定checkpoint输入干预，也未证明跨训练seed稳定。

综合§72–78后的解释边界与下一判别条件：

- 冻结source的双相机端点读出确有局部动作价值；“补足native视野即可由现有共享链兑现稳定收益”的本次预测未通过。
  这淘汰当前组合的充分性，不推翻局部读出事实，也不能改写为腕部视频普遍有害。
- 共享Meta可能改变原生可读知识，表示读取也可能丢失或未利用它；当前闭环结果不能分开二者。
  下一项获取／保留诊断必须在同一合法train位置、相同原生读出合同下产生可比较功能证据，不能仅看hidden距离或FM。
- 即使某接口保持可读信息，也不保证跨episode、跨初态的参数行为传递。旧局部动作教师、功能共享／蒸馏及forecast
  的失败约束继续有效；不能从这次负结果跳到“只差Compiler”或再接一次已失败的功能中间量。
  传递假设必须提供能改变决策的跨初态功能预测，并明确上游失败时停止，不把probe通过自动转成新Writer训练。
- 相邻train增长而validation下降也兼容任务共性、样本曝光与条件映射问题；固定24-task两节点尚不能唯一归因。
  已审计的跨episode FM允许视频无关解，但不是视频学习不可能定理；不以更多同task视频、seed或超参扫描代替机制证据。

本候选关闭，不追加checkpoint、相机融合、Meta冻结、flow、rank、LR或训练seed；不运行最终controls、Test或RL。
完整原件在`runs/analysis/native_dual_video_20260913/`的`paired_summary.json`、`camera_comparison.json`、
`bounded_200_initial_decision.json`、`bounded_200_decision.json`、`evidence_audit.json`、`training_pairing.json`及
`OVERNIGHT_READOUT.md`；四完整checkpoint、8个bank、raw rows与全部completion保留。整体goal未完成。

## 80. 跨初态关系监督有数据来源，但同task关系到LoRA仍可退化为任务记忆（2026-09-13）

[可识别性分析§9](docs/video_information_identifiability.md#9-跨初态操作关系新增监督必须区别于旧状态条件化与任务记忆)
完整记录近邻历史、数据检查与判别限制。现有LoRA的`B(Ah)`已随执行状态变化；旧native-factor已用X/Y生成因子，
旧Local Action Grounded已反演真实帧之间的动作。仅改称状态地址或回顾反演没有新增机制。

只读检查train任务0/12/20/34的demo16，每个episode首／中／末三个存储状态用既有MuJoCo在CPU恢复，
共12个forward完成、状态宽度与nq/nv匹配、位置有限。物体位姿可从现存states/XML/assets恢复，但没有现成关系标签。
目标实例、其它移动物体与抽屉子部件需要区别；柜体根位置不变不代表抽屉操作不存在。没有推进仿真、模型forward或梯度。
这是标签来源的可行性，不是动作、接触、视频获取或闭环能力证据；没有生成标签库。

若每task对应固定关系r(t)，同task跨episode／初始化验证不能排除关系到LoRA的任务索引记忆。
因此oracle关系图训练内成功不能独自定位视频读取失败；纯终态标签也不能证明动态视频增量。
下一候选须同时说明合法RGB获取何种方向性实例／部件关系，以及怎样验证其经唯一LoRA跨初态和任务传递。
不能由位姿可恢复默认训练完整Writer，也不把这些限制扩大为否定所有关系监督。当前无新active design。

## 81. Privileged状态匹配能传递部分动作价值，相对几何替换未获支持（2026-09-13）

[登记诊断](docs/cross_init_relation_retrieval_audit.md)已完成：train24、action16–19单演示与诊断42–45交叉，
固定stride5、post-action后的完整5×7动作，384个episode对、13,064位置；无训练、source模型或LoRA。
对象集合来自官方obj_of_interest，含body与site；两种检索只改变平移坐标，动作Value及其它距离项相同。
没有可训练task→参数映射，但privileged对象对应与真实teacher actions仍不构成合法deployment输入。

绝对／相对／video_mean归一化MSE=.12443553/.12634790/.25459689；
绝对−相对差额−.00191237，task-cluster95%CI[−.00421347,+.00027549]，仅8/24task为正、四suite点差额均负。
均值−相对+.12824898，CI[+.11063319,+.14508720]、24/24task为正；绝对亦在24/24task胜过均值。
登记两项条件未同时满足，关闭相对中心化的本具体替换依据，不追加坐标、尺度、特征或邻居数扫描。

四个teacher中，相对只有demo19略优于绝对；两种检索相对均值的收益在四teacher均保持。
总体收益来自平移与夹爪：绝对／相对／均值的旋转分项为.09144200/.09160858/.08597376，匹配未优于均值。
因此保留部分状态条件化动作传递能力，不能将它称为完整动作教师或由此替换原Writer。
此检验没有给出“只差世界坐标对齐”的证据，也没有评测RGB关系获取、顺序因果或参数编译；旧功能教师负结果仍有效。

执行冻结9f90a14d、99.37秒、exit0；192条episode的6,513个几何状态与全部raw rows重算通过。
初次54be9ca3在评分前发现2ms末次积分缓存差；按已安装step/sensor语义回退一个子步恢复几何后吻合，
动作offset1与生产Writer不变。修正前的失败、登记、脚本Git及全部原始证据保留，非一次科学non-pass重试。
原件位于`runs/analysis/cross_init_relation_retrieval_20260913/completed/`，完整索引见诊断§7；约1.3MiB。
一次性入口退役，无active design、selected checkpoint或新训练；整体goal未完成。


## 82. 物理标签有动态覆盖，不足以定义通用语义rank（2026-09-13）

[操作语义可行性](docs/operation_semantics_feasibility.md)完成train24固定192episode、6,668状态位置的CPU恢复，
源7bd8a6f5、84.34秒exit0。两指同对象接触共2,878对象帧、493次转换，186/192episode存在转换。
这证明所选物理事实不是task常数；不证明RGB可读、动作适用性、稳定抓持或视频必要性。
抽屉task20有6/8episode在stride5采样位置没有双指同时接触；region所属body还可能覆盖整柜、整桌或整架。
因此不把双指proxy及region-body接触直接升格为通用阶段／支撑／前置条件，也不扫阈值或挑task修补。

直接监督生成A的实际响应虽然区别于无关辅助头，仍有明确反例：B=0或rank更新抵消时语义可准确而行为无贡献；
source h已有language时共享A也可能不需要视频。标签覆盖不能解除这两项传递与特异性问题。
撤去“通用接触阶段→语义rank”的直接实施依据；保留分部件变化与左右指物理事实，完整Writer尚无合格启动合同。
新原件约1.36MiB、raw重算及时间对应通过，CLI已退役；本项零梯度、零GPU、零环境步，无held/Test或动作数值读取。


## 83. 真实短段动作效果部分可预测，局部J替代未达登记精度（2026-09-13）

[操作语义§6–7](docs/operation_semantics_feasibility.md)将旧activation/action effect与真实对象／部件变化分开：
旧OCPB/MDCO已有success/progress的物理outcome信用，本项新增的是固定状态下明确动作干预造成的dense位姿变化。
fcd9a613冻结完成288条件×17分支、24,480高层动作步，434.79秒exit0，无模型、LoRA或梯度。
12个轴向干预的中心差分预测两个未拟合组合干预；零变化／局部MSE=.00089956/.00026749，
差额CI[.00048754,.00076067]、23/24task和四suite正。但误差比例.29736未达<.25，按登记停止J加权FM提案。
Object比例.01887、Spatial.23625、Goal.41550、Long.51493，不能只引用70.26%总体降低称为通用效果教师或挑suite挽救。
该J只覆盖5动作共同运动偏移，不覆盖时变30维误差与离散夹爪。普通sign梯度几乎处处为零；
open/close真实效果不同，但其扰动幅度和连续ε不同，不能据效果量大小宣布夹爪主导失败或直接增权。
原始数组重算通过；基准对旧存储终态position RMS平均.345mm/P95=1.935mm，不能称原采集轨迹精确复现。
原件约.79MiB保留于runs/analysis/operation_semantics_20260913/effect_replay；CLI退役，无新Writer资格。

## 84. 实际视觉注意力的物体／运动监督有区别且数据合同通过，学习收益尚未检验（2026-09-13）

v4错误目标绑定回放、Horizon错误对象／实例及共享更新改变目标选择，支持检验绑定信用不足；不能据此解释全部失败。
限定近邻审计未找到v4/v5/Horizon用真实可见mask或部件运动直接监督视觉cross-attention Q/K；旧task/causal/padding mask、
H×H对应和冻结回放均不属于该监督。新[设计](docs/visible_object_grounded_writer_design.md)保持完整生成链，
标签只训练实际native/prior空间读取分布，主FM仍跨episode；没有新可训练模块或部署标签输入。

train24 teacher0–15的384条CPU构建完成，05fe7ebe冻结257.81秒exit0；13,626帧中13,242可恢复且可见OOI，
10,819帧有可见运动质量。末帧没有对应存储state时仅跳过监督，原始末帧仍入模型；关节body独立保留。
静态target对可见OOI等权，motion按真实body表面位移上界及其可见mask构成；region owner只表示相关实体表面。
原始数组、时间与两种patch布局重算通过，固定四suite双相机／prior crop叠图通过。原件在
runs/analysis/visible_object_grounding_20260913，构建入口退役，源保留Git。

128项原有合同检查及真实标签＋合成小特征的联合梯度smoke通过，只证明工程图成立。
定位准确仍可能被Value／下游忽略；原生模型学习、配对闭环和跨task迁移尚未产生，不能把数据通过当作机制通过。
当前登记两臂各fresh200与100/200的8面板，沿原严格资格／停止分支。已关闭的J、语义rank等候选不由本数据恢复。

## 85. 空间信用有后段绝对收益，但未形成相邻有益有序增量（2026-09-13）

§84的学习检验已完成：b304cde6冻结两臂各200updates、800条件、51,200queries，四完整checkpoint、
八sealed bank和1,984闭环rows，全部worker／阶段exit0。新旧四模型18个曝光字段、0/100/200独立诊断配对、
raw/aggregate、teacher真实stride5帧与末帧、RNG、single Writer invocation和完整LoRA身份审计通过。
validation每task50teacher各一次，train96明确复用46–49/states32–35。原件为
runs/analysis/visible_object_grounding_20260913，最终bounded_200_decision与完整OVERNIGHT_READOUT均保留。

train有序／无序100为32/33、200为50/50；validation为20/24、71/70，净率95%CI[−3,+1]pp与[−2,+2.25]pp。
validation100 S/O/G/L=0/16/3/1对1/19/4/0、breadth5/4；200=0/59/9/3对0/61/7/2、breadth6/5。
有序对无序100 R/G/L=12/8/12、churn20、J=.375；200=62/9/8、churn17、J=.78481。
相邻有序20→71为19/52/1、churn53、J=.26389；95%旧成功保留，低J由新增主导，不能写成晚期崩溃。
正确条件后段改善通过这一项，但CI下界不严格正、增量−4→+1不相邻同向，100也只有一个suite净正，故基础资格不通过。

预先固定的同模式监督参照显示：validation100纯FM→空间监督为有序64→20、无序56→24；200为53→71、39→70，
后两者CI[+1.5,+8.5]pp、[+1,+16.75]pp。空间监督有真实后段绝对收益，并改变了学习轨迹，不能称为完全无效。
但监督×顺序交互200为−3.25pp、CI[−11,+2]pp；上述绝对改善没有证明有序结构更好使用监督或视频必要Value。
实际注意力拟合不等于对象语义角色充分，收益可能包含一般读取正则／共享参数改善；当前证据不能唯一定位Meta或Compiler。
frame_set包含全部真实frames，独立训练比较也不同于冻结模型顺序干预；没有运行后者，不推断canonical顺序不变。

source47/400的S/O/G/L为0/5/41/1；有序200=0/59/9/3，R/G/L=12/59/35、churn94、J=.11321，
task-cluster净率区间[−20.25,+31.75]pp。Goal41→9仅保留7、新增2、丢失34，Object5→59保留5、新增54。
比source更高总数与源能力保持不足同时成立；source区间只限定解释，不追加原合同外硬门槛。

按原设计§5关闭当前空间监督组合，不续训／扫参／改标签定义，不补other、frame_set_image、跨初始化资格、
最终内容／shuffled/reversed controls、Test或RL；无selected checkpoint，整体goal未完成。
后续必须同时解释局部原生动作可读、同模式后段监督收益、跨任务行为偏移与稳定有序收益缺失；
停止把“再给一个辅助信用”默认当成完整机制修复，也不能从本次non-pass推成所有空间监督或视频方法无效。

## 86. 空间边际、动作纠正与可迁移策略作用的三个未等价接口（2026-09-13）

[可识别性分析§10–12](docs/video_information_identifiability.md#10-空间边际监督学到了什么尚不能据它推断什么)
结合§85完整结果核对了实际loss：KL监督head/query平均后的patch概率，任意置换query的物体分配不改变该loss。
两query分别读A/B或B/A都可对(.5,.5)目标取得KL=0，实际Value却分别为(+.8,−.8)与(−.8,+.8)。
此CPU代数反例保存在当前analysis的attention_objective_identifiability.json；不是真实模型坍缩测量。
运动质量也是无符号位移上界；它不直接标注方向，但不能据此声称整条标签或模型在倒序下不变。
因此边际定位拟合不足以证明对象角色／操作获取已解决，不能用它把余下失败唯一归到Compiler。

另一个直接反例：若仅蒸馏source在完全相同输入／state／noise合同下的自预测，identity LoRA已使函数MSE为0。
教师状态支持本身没有给该目标添加正确偏离source的方向；跨输入域或真实转移纠正是不同目标，需要明确登记。

近邻源码审计确认，v4原生forecast及latest−earlier预测差实际进入LoRA，但原生prior受Meta影响且无真实转移纠正监督；
Local Action Grounded已有回顾实际动作标签，却是独立fresh FM头，其输出不进入Compiler。
两者与“固定原生prior＋真实转移纠正＋纠正直接作Value”不同；此限定新意不能被扩大为全历史从未尝试或有效性证明。

动作残差本身也不是完整迁移规则：source可在教学轨迹上预测正确而在新初始化失败，故δa=0不能自动解释为视频无用；
非零教师动作残差同样不能直接照搬到新的机器人状态。先明确状态／部件条件与效果的联合关系怎样约束执行策略，
再判断合法RGB可否获取它；当前不训练新的残差动作头、不恢复裸forecast编译、不构建新cache。
本项是理论与历史机制裁决，无新模型／环境／最终controls，整体goal未完成。

## 87. 真实动作纠正到原生参数作用：梯度几何、训练信用与前向构造须分开（2026-09-13）

[原生纠正传递诊断](docs/native_corrective_transfer_audit.md)把条件与纠正的联合关系落实到
`g_l=Σ c_li x_liᵀ`：c是同次forward的输出cotangent，x是该层真实输入；它区别于native Y或forecast差。
其跨episode作用仍需实际测量，教师位置下降不保证另一初态下降，state-free与true-state的Jacobian也不自动匹配。

近邻审计修正一个证据口径：旧95-task behavior authority的.716/.801与.2695来自
`fcdb6e43:src/ember/ecp/behavior/gate.py::_exact_rows`的`factor_cosine`，不是实际跨episodeFM／闭环。
`5cbe76e0:scripts/seal_ecp_g2_behavior_codes.py`只加载外部rank4因子，`source_policy_loaded:false`；
本次限定检索未找到原始生成器，不能推断它们的source/carrier运行点、state、noise或flow time。
P1的mapping容量、J2实际100步FM正控，以及EBSRI/PNBTT的生成器VJP是不同实验；它们的正负边界分别保留。

因此没有把梯度一词当作新意，也没有由几何正数假定当前传递通过。新有界诊断将给定真实动作、固定一次38目标rank16构造，
在独立query中同时检验t1与full10真实输出；不先训练RGB残差头。该项是privileged训练侧oracle，登记时尚未产生结果，
不能包装成action-hidden部署或放开禁止task-local优化的边界。完整采样、配对、停止与资源合同在上述登记文档。

## 88. 同帧条件与真纠正的一次原生参数构造具有跨episode功能前提（2026-09-13）

[原生纠正传递§7](docs/native_corrective_transfer_audit.md#7-完整结果与关闭裁决)完成f39d594f冻结的192套完整rank16 LoRA、
3,072个condition-query组合。固定train24/demo16–19构造，42–45的384个真实query仅预测后评分；source始终冻结，
无optimizer、query梯度或环境步。三个worker均exit0、墙钟110.63／113.15／113.28秒、峰值11.014GiB。

state-free t1 MSE .11977705→.11351420，改善.00626285、95%CI[.00251749,.01101049]、17/24tasks正；
full10 .16493043→.15574849，改善.00918194、CI[.00322478,.01688248]、21/24tasks正。
true-state t1/full10为.11022396/.15252233，改善CI[.00485088,.01539874]/[.00534316,.02148158]，21/22tasks正。
四单元均四suite净正、四teacher序号汇总正，两oracle均按原共同资格通过；不是在t1/full10间选优。
state-free full10 S/O/G/L改善.00332542/.00471454/.02799325/.00069453，83/96条件正；task2/34/37非正，保留负条件。

完整raw MSE、方向幅度、配对与76张量shape通过；原HDF的768个teacher／384个query动作标签、冻结quantile及采样位置补核通过。
原件位于runs/analysis/native_corrective_transfer_20260913，TRANSFER_READOUT.md包含全task／suite／teacher与证据限制。
当前具体结论是：给定真实纠正，state-free的原生条件×cotangent联系可在所测分布改变另一episode的原生输出。
这比因子cosine或局部辅助头更直接，但仍是privileged oracle；没有证明合法视频获取、共同偏置之外的视频作用、闭环或held迁移。

按正分支关闭诊断、退役一次性入口；下一项可继续推导共享前向生成的获取与传递合同。
不能用运行时loss/VJP/任务更新规避信息墙，也不能仅给普通自由B head改名、拼接历史正数就假定Writer已获支持。
完整有益视频特异性goal仍未完成，没有恢复旧Writer、扩展扫描、Test或最终controls。

## 89. 从oracle到合法前向生成的具体区别与存在性边界（2026-09-13）

[Native Correction Writer](docs/native_correction_writer_design.md)用`ΔW=B(RX)`保留同位置输入与预测纠正的乘积，
避免把部署梯度更新包装为Writer，也避免先独立pool两侧后引入跨位置项。B自由而A由实际裸source X构成，
旧nativeD的自由B及EBSRI/PNBTT的训练VJP本身都不是这项新意；两者的负例和能力限制继续保留。

oracle更新右空间属于其输入X的行空间，完整视频X包含oracle四采样位置，因此任意R的rank16构造有表示该更新的空间。
这个存在性不证明有限attention网络能预测R，也不证明RGB足够；监督只约束预测纠正对X的参数作用，
不声称逐位置cotangent可唯一恢复。目标直接使用已具实际跨episode功能证据的source更新，按完整ΔW误差训练，
避免SVD因子符号／旋转和native Y span的限制；部署生成不用SVD。

保留原有读取及实际空间信用；使用授权action池16–41自己的RGB/纠正label，主FM必须逐条件排除其教学episode。
这是新episode合同，不伪装成与旧0–15教学逐行匹配；目标标签不进入部署。拟合、identity与参数空间都不等于行为，
最终仍由两个节点的完整配对闭环、跨视频／初始化保持与冻结后的controls裁决。两臂fresh200和8个闭环面板已完成，按原资格关闭，见§91。

初始最长profile的图与梯度成立，但第一步参数误差1→2910.36。固定Q/A的解析首步2906.34表明输出坐标尺度足以解释主要放大，
单独单位A仍162.19。按[设计§9](docs/native_correction_writer_design.md#9-正式学习前的因子单位修正)在正式学习前修正：
单位行A与训练标签RMS确定的共享B单位共同定义参数坐标，原物理ΔW loss不变；38个常数无task／held条件统计。
这改变有限共享模型的参数化与优化轨迹，不能说成已证明的最优范数、原实现bug或实际视频收益；需fresh profile及原行为裁决。

## 90. 原生纠正标签具有共同成分，但这不是合法获取或行为收益证明（2026-09-13）

对固定train24/demo16–41的624套已封存state-free更新做只读统计，逐task计算26×26完整ΔW Frobenius Gram；
没有模型forward、优化器、额外动作读取或held数据。每task的均值更新能量／平均更新能量之比，task等权均值为.468739。
按原相对Frobenius度量求无秩约束task常量的样本内最小误差，task等权均值.516543，S/O/G/L为
.550768/.557875/.274493/.683035；逐task范围.142856–.817885。完整更新能量的87.415%位于action_out_proj。

这保留了标签中存在共同结构的事实，不能把较弱参数拟合仅归为目标没有共同成分；但这个参照使用完整训练标签与task分组，
没有rank16／当前生成器约束，既不是可部署task字典，也不是RGB可获取性、不可约误差或与训练曝光匹配的模型比较。
它不能区分有限参数化、优化、读取与视频信息的各自限制，更不能替代当前8个配对闭环面板的裁决。

原件在runs/analysis/native_correction_writer_20260913的label_structure_registration、label_structure.py/json/log；
CPU 6.73秒完成，保留全部Gram、逐demo能量／误差和逐layer统计，因子Gram与实际dense更新内积的单项复核通过。
该只读分析不产生初始化、adapter、梯度或checkpoint选择，不改变当前active design与停止条件。

## 91. 原生纠正Writer保留source能力，但未获得有益有序增量（2026-09-13）

[Native Correction Writer§10](docs/native_correction_writer_design.md#10-完整有界结果与关闭裁决)完成两臂各fresh200、
四完整checkpoint及8个面板／1,984条配对记录。100/200的train有序／无序为15/15、19/20；
validation为50/51、50/49。有序−frame_set的task-cluster95%净率区间分别[−.75,0]pp、[−.75,+1.5]pp，
净正suite数0/1、增量−1→+1；没有通过严格正下界、至少两个suite净正与相邻同向的原资格。

有序两个validation节点S/O/G/L均为0/6/41/3，frame_set为0/6/41/4及0/6/42/1，breadth均3/8。
source47为0/5/41/1；有序各相对source保留45／新增5／丢失2、churn7、J=.86538，净率区间[0,+1.75]pp。
名义+3、Goal总数保持与低churn是有限正事实，不证明统计明确的source收益；Spatial仍零成功，共五个validation task零成功。
有序对frame_set两节点的R/G/L为45/5/6、46/4/3，churn11/7、J=.80357/.86792；
有序相邻50→50为46/4/4、churn8、J=.85185，无序51→49为45/4/6、churn10、J=.81818。
成功总数相同不等于相邻成功集合相同，较低churn也不等于视频必要性。

训练侧frame_set15→20保留15／新增5／丢失0、净率区间[+1.042,+9.375]pp，相对source15也有正区间；
有序15→19的相邻区间跨零，四模型breadth均7/24。保留该有限训练收益，不能写成完全没有学习。
固定独立动作FM初始.15328534，100有序／无序.15307564/.15304530，200为.15281900/.15280163；
末25条件L_update约.90642/.90657，空间信用有拟合。窗口teacher draws不同，不将其与§90的样本内常量参照
冒充匹配曝光比较，也不以参数误差代替闭环裁决。

本轮有序200的Goal41优于上一空间监督组合9，Object6低于59、总数50低于71。
教学池16–41与旧0–15不同，出口及纠正监督共同改变，不能将差额唯一归责原生因子出口。
§88的真实纠正oracle跨episode功能正事实仍成立；任意R的空间存在性、共同标签结构与本有限共享模型获取是不同问题。
本轮失败不唯一识别Meta、Compiler、有限R/A空间、B纠正预测／学习信用或RGB信息限制。

09c1a0d6 clean pushed frozen运行，训练18个曝光字段、0/100/200独立诊断10个字段配对通过；
全部真实帧／末帧、state-video无放回与有限池、环境／policy RNG、76因子和single-checkpoint身份、raw及aggregate通过。
全部阶段和96个评测worker均exit0，评测launcher累计4,051.92秒，controller正常退出、无在途本研究Python进程。
原件在runs/analysis/native_correction_writer_20260913的完整readout、paired_summary、最终decision、audit及training原始目录。

按预登记停止该共享获取／原生因子组合，不追加训练或rank／λ／seed／LR／scale扫描，不补未获资格的
other／强静态／跨初始化、最终controls、Test或RL；没有selected checkpoint，完整goal未完成。
下一项先区分有限原生输入空间与纠正获取的竞争解释，核对近等价历史并登记有停止条件的最小冻结诊断；
本结果不自动授权恢复本组合或连续新增完整Writer。

## 92. 当前A空间内的纠正获取误差占主导，不能据此宣称RGB充分（2026-09-13）

[固定A误差分解](docs/native_correction_acquisition_audit.md#6-完整结果与关闭裁决)完成四checkpoint×train24/demo16–19，
384次冻结合法Writer生成，使用同视频既有state-free纠正标签。原生完整X的空间存在性不等于当前16行A；
本次精确分解`E=||BA−G||²=F+ D`，F为`||G(I−P_A)||²`，D为A空间内尚未拟合的部分，保持原共同参数度量。

100有序／无序E/F/D为.951379/.313762/.637617、.950511/.314918/.635593；
200为.900546/.270922/.629623、.900279/.265043/.635235。
四个task-cluster95% F−D区间分别[−.371229,−.277388]、[−.366038,−.275823]、
[−.404789,−.314671]、[−.411855,−.330517]，均严格负；每个模型24/24task均值及全部suite／teacher序号汇总D>F。
实际误差中66.9%–70.6%在当前A空间内，触发原第二分支，停止把扩展A空间当下一修正的必要先决条件。

实际见过的教学条件100为49/96、200为67/96，两臂相同；200已见的D仍为有序.628765／无序.636050。
不能把这个拟合缺口只归为未见video，但seen/unseen不是随机因果比较。
相邻有序E下降.05083中F下降.04284、D下降.00799；无序E下降.05023中F下降.04987、D下降.00036。
保留A覆盖变化与空间内获取有限的事实，不从参数曲线推断已找到闭环根因、平台或更多训练必然有效。

本下界允许任意条件专属B及任意幅度，且真实A存在很弱的奇异方向；低F不证明有限共享B易学或稳定可达。
F仍约.27–.31，A也非完全充分；本96标签action-out占能量87.6726%，总参数尺度更不是行为价值权重。
当前不能唯一分离有限B、读取表示、RGB信息、可用坐标或优化信用，也不因此恢复旧dual／fixed-A／分段冻结方案。
旧投影丢失闭环行为、G1局部容量与G3共享获取负例的限制保留；该结果不是视频必要性或新的闭环正例。

8c714e91 clean pushed detached，四worker均exit0、501–505秒；CPU26.286秒exit0。
384个条件、同视频label、76因子／finite／真实帧／单次Writer和checkpoint身份、四bank与E=F+D全通过；
满秩／零A／rank1和实际action-out dense/Gram核验通过。无新动作读取、模型更新、环境步、held数据或checkpoint选择。
原件在runs/analysis/native_correction_writer_20260913/acquisition_audit的完整readout、summary、decision、
四decomposition／bank及launch/log/exit。按登记关闭，无active design，完整goal未完成；下一项先推导纠正获取的可失败解释。

## 93. 整体幅度只能消除少量现存误差，可表达方向仍有获取与坐标代价（2026-09-13）

[获取诊断§7–8](docs/native_correction_acquisition_audit.md#8-幅度方向分析完整结果与关闭裁决)在3c3c4897登记后，
CPU只读既有四模型384条件完成全局ray分解。每个条件对完整38-target只取一个privileged最优非负倍率，
不构造或部署新adapter。保持原video／task等权参数度量，`E=F+J+H`，J为最优幅度后仍缺方向，H为整体幅度可消除项。

100有序／无序J/H=.599946/.037671、.598835/.036758；200为.582501/.047122、.578330/.056906。
四个task-cluster95% H−J区间依次为[−.599817,−.523732]、[−.599820,−.523410]、
[−.575439,−.495857]、[−.568200,−.473465]，全部严格负；每模型24/24task均值及所有suite／teacher汇总J>H。
整体倍率只消除当前实际误差的3.87%–6.32%，D的91.04%–94.22%仍需改变方向，按登记停止仅整体gain的主要修复依据。
不能从较小预测范数推成只差放大；完整向量中的层间相对幅度属于方向，本项没有排除所有层间机制或证明闭环影响。

200已见67条件／24task，组内task等权J/H为有序.585914/.047277、无序.585044/.058159；
未见29条件／18task为.568859/.049578、.561166/.060472。曝光分组不是随机因果比较，条件均值亦另存以对应§92口径。
在当前单位行A坐标下，最小B范数对正交行坐标的逐条件比率中位数，100为259.23／255.96，200为370.56／386.61；
能量汇总比率294.19／293.43、895.20／723.05。完整38层及384条件原始量保留，没有按层重新选择gain或奇异阈值。
高比率显示乐观固定A投影可依赖昂贵弱方向；范数具有坐标依赖，不能将它当有限网络容量上限或唯一学习根因。

14.183秒exit0；六组固定小矩阵的ray／最小B公式、真实投影重算、三项恒等式及完整task等权汇总通过。
原件为`runs/analysis/native_correction_writer_20260913/acquisition_audit/ray_*`，含全量readout／原始统计／分组／decision及运行源码日志。
本项关闭，无新模型forward、动作读取、梯度、query、环境步、held读取或checkpoint选择；整体goal未完成。
后续须把方向获取与坐标代价分别解释，保留原生oracle功能、标签共同结构及旧校准／rank均衡的局部正负边界；
不能由这两个定位数值直接恢复dual、白化、calibration或另一个完整Writer。

## 94. 固定真实纠正在独立初始化上取得有限但可重复的闭环净收益（2026-09-14）

原注册train24×demo16–19×init32–35共384oracle episodes及一次source96全部结束。
source17/96，四teacher分别21/25/20/24；按四teacher复用source参照为68/384，对oracle90/384。
24task等权净率+5.72917pp，20,000次task-cluster bootstrap、seed20260926，95%CI[+1.04167,+12.5]pp。
四teacher净增+4/+8/+3/+7，四init净增+1/+5/+8/+8，S/O/G/L净增+5/0/+5/+12；原四项前提全部通过。
聚合R/G/L=60/30/8、churn38、J=.612245；source breadth7，oracle跨teacher并集breadth10。

本结果将§88–93中“G只被验证具有独立动作预测价值”的边界推进到该有限面板的实际闭环作用。
它仍不证明合法RGB共享获取、时序必要性或held泛化，不与不同teacher池的旧Writer直接作因果比较。
只有7task净正、1task净负，task35贡献11/22净增，Object完全未获取；不能写成广泛任务能力已解决。
子teacher／init区间不都严格正，注册要求的是主区间严格正和所有子组净额同向；不升级其统计结论。

据此优先研究合法条件控制作用的获取，同时不将旧G特定参数坐标当唯一中间目标。
语义状态路径候选以完整native R/Z、显式有向状态交互及自由完整A/B承担这条共同FM学习链；
它须重新证明获取、ordered−frame_set和相邻／跨视频／held保持，不能继承oracle的分数或资格。
不继续旧oracle、扩面板、调整幅度或续训已关闭Writer。

运行代码clean pushed 2c0a8dde、恢复登记c8181ac8／5f1c25e8，45worker和controller全部exit0，
累计launcher1943.48秒；真实480rows、执行合同和policy RNG公共前缀配对重算通过，运行进程已退出。
原件在`runs/analysis/native_correction_writer_20260913/oracle_rollout/`；完整结果见
[闭环诊断§7](docs/native_corrective_closed_loop_audit.md#7-完整结果与关闭裁决2026-09-14)，临时来源准入按关闭合同退役。


## 95. 显式二阶状态路径有训练获取，未获得重复过程增量或未见任务保持（2026-09-14）

Semantic Path两臂fresh100、四checkpoint和全部八面板完成，共1,984闭环rows；每臂400条件／25,600queries，
曝光18字段、独立诊断12字段匹配。完整R/Z、双向理解、自由完整A/B与唯一跨episodeFM的组合由独立fresh全帧集合直接比较。

| 节点 | train96有序／无序 | validation400有序／无序 |
| --- | --- | --- |
| 50 | 16 / 22 | 77 / 75 |
| 100 | 29 / 26 | 48 / 44 |

新source400为47，训练source17/96。validation有序−无序95%区间[−1.25,+2.50]／[−1.75,+5.00]pp，
净正suite2／1；有序−source区间[−2.25,+23.50]／[−19.25,+17.25]pp。原两个节点及相邻资格全部失败。
有序S/O/G/L为0/38/36/3→1/32/14/1，无序0/35/38/2→1/34/7/2；breadth分别5→6、3→6。
有序相邻R/G/L30/18/47、churn65、J=.31579，无序23/21/52、churn73、J=.23958。
两个模型都获得部分Object而失去Goal；不能用无序退化或早期77的单峰宣布时序恢复。

有序train16→29、breadth8→12，100相对source净+12、CI[+2.08,+22.92]pp；相邻净率CI[+5.21,+21.88]pp。
但有序−无序−6／+3，CI[−11.46,−1.04]／[−6.25,+13.54]pp；没有可重复训练有序优势。
两组固定动作FM为.153285→.130330→.121139及.153285→.130311→.121899，50→100各24/24task改善。
这反对“没有任何学习”的说法，也明确拟合／部分训练获取没有形成held保持；不把100称为充分训练或平台。

原始语言显示，held的cream cheese→basket从source5改善到有序35／27、无序35／28；
cream cheese→bowl则从source41变为有序36／13、无序38／6。只有任务条件与成功计数，尚无目的地混淆或操作阶段根因证明。
未来若核对这些行为，须固定全任务／checkpoint／等距init及原state-video映射，不按结果挑实例，也不能先改模型再补故事。

原四checkpoint、八bank、真实帧／末帧、38-target／76-factor、单次Writer、source与RNG配对均通过。
39个最终worker exit0，八面板有效累计10,202.87秒；有序50的一次EGL初始失败按原contract恢复完整400，原件保留、根因未定。
运行代码5116deb0 clean pushed frozen。原件位于`runs/analysis/semantic_path_writer_20260914/`的READOUT、paired_readout、
bounded100_decision、各training／materialized／evaluation及launch/profile/recovery。

按注册关闭本主假设组合，不续训或扫路径阶数／宽度、head、rank、LR、seed、scale、辅助loss。
没有selected checkpoint、other、最终内容／时序controls、Test、held梯度或RL。整体goal未完成，自主授权持续。
下一解释须同时保留§94的privileged控制正事实、这里的一般训练获取与稳定／过程负证据，不默认另一个完整Writer或保持补丁。


## 96. 固定四模型回放显示条件绑定与操作／组合保持的异质缺口（2026-09-14）

语义路径原四模型、全validation8、各init0/12/25/37，共128次正常correct回放及32组双相机人工观察完整。
原400子集→回放成功O50 7→5、F50 5→9、O100 3→5、F100 1→1；120/128结果相同。
模型、完整LoRA、teacher/task/state与环境／policy RNG公共前缀配对通过；8条结果变化原样保留，原400与Writer关闭裁决不改。
24个worker exit0，累计launcher812.30秒，轨迹11.74GiB，双节点无本研究在途进程。

奶油奶酪Goal6的100节点六条失败中，目标仍留桌面而手已转向碗附近；50的另一失败已把目标运到碗上方，
但目标谓词始终未成立。Object1有目标附近反复获取、运输迟滞，也保留有序100比50更快成功的初态反例。
这组没有清楚的错误目的地证据，却不能外推为语义绑定已正确：Spatial可见错实例碗，Object3可见错瓶，
Goal3停留抽屉开合，Long存在错包装盒／平底锅及组合目标未保持。不能统一归因抓取或occupancy。

完整谓词还修正了稀疏图像可能形成的过度解释：Long1/state0/F50的cream cheese223步入篮、361步失去；
Long2/state12/F50开炉70步成立、120步失去。它们不能写成从未完成对应子目标。
稀疏图像、遮挡与最后replan并不唯一说明接触、释放或落点；某些相近棕色瓶身份明确保留不确定性。

按登记的异质失败分支关闭，不扩病例、模型、checkpoint或controls，也不由该回放自动启动解析、局部动作头、
保持蒸馏或状态扩池。下一方法仍须联合解释实体／条件绑定、具体控制和训练保持；没有唯一失效模块或新的正向方法被证明。
旧teacher-state接续、expert occupancy蒸馏和相邻更新干预已显示相关混合缺口，本轮新增的是这些四个模型的实际行为。

合同见[冻结行为回放§5](docs/semantic_path_behavior_replay.md#5-完整结果与关闭)，代码5116deb0、登记3bd9db19。
原件`runs/analysis/semantic_path_writer_20260914/behavior_replay/`保留READOUT、replay_readout、decision、
两份覆盖全部32组的人工观察、固定图像／元数据、128条逐replan轨迹及完整launch／completion。
整体goal未完成，自主授权持续，无active design或在途运行；不从历史关闭段恢复训练。

## 97. 在局部纠正场上限制rank16保留了实际跨episode作用（2026-09-14）

[局部纠正场§5–7](docs/local_correction_field_design.md#7-固定算子完整结果与下一阶段)固定原train24/demo16–19、
四位置、source、eta及query42–45，先对真实输出cotangent场C作rank16，再与同位置X收缩。
4f55968d完成96条件／1,536 query组合与两读出，三个worker均exit0，原身份／truth／noise配对及finite通过。

t1 source .11977705→.11351439，净改善CI[.00253443,.01103297]，17/24task正；
full10 .16493043→.15574597，CI[.00326525,.01693516]，21/24task正；两读出均四suite净正，原共同资格通过。
与旧G均差约−1.84e−7／+2.52e−6、区间均跨零，不能声称等价、严格非劣或更优。
逐条件局部C能量保留率范围.998731–.999738，总参数相对原G的Frobenius差范围.007706–.027575；
这类几何只作描述，不替代上述实际函数结果。三worker68.620／70.805／69.517秒，峰值11.014GiB。

原始场、完整LoRA、预测、逐task／suite／teacher及裁决在runs/analysis/local_correction_field_20260914。
按正分支关闭诊断、退役入口；只支持该局部场收缩的privileged函数前提，未检验合法获取、全视频各位置、闭环或held。
下一阶段完成已选同位置过程→纠正場→唯一LoRA的共同学习机制，不重开独立动作头或由这个正数宣称goal完成。

## 98. 同位置纠正监督及直接参数消费仍未形成有益共享获取（2026-09-14）

[局部纠正场§8–10](docs/local_correction_field_design.md#10-完整50100结果与关闭裁决2026-09-14)让同一预测场接受
真实局部cotangent监督，并与同位置裸X直接收缩为完整LoRA。两臂独立fresh100、各400条件／25,600queries，
四checkpoint及全部八面板／1,984rows完成；源模型、实际视频／初态／RNG、完整目标及采样曝光审计通过。
24个最终worker和所有launcher均exit0，无失败重试。这是有效科学non-pass，未发现运行合同错误。

| 节点 | train96有序／无序 | validation400有序／无序 | validation有序−无序95%CI |
| --- | --- | --- | --- |
| 50 | 18 / 16 | 48 / 47 | [0,+.75]pp |
| 100 | 17 / 17 | 47 / 50 | [−1.75,0]pp |

source为train17/96、validation47/400。两有序相对source的CI为[0,+.75]／[−.75,+.75]pp，
有序−无序净正suite1／0；50／100两个节点资格及相邻保持资格均失败，没有选择任何checkpoint。
四组validation的S/O/G/L分别为0/5/42/1、0/4/42/1、0/5/40/2、0/7/40/3（O50/F50/O100/F100）；
breadth均3、Spatial均零，train breadth均7，与source仍在相同的有限任务上成功。
有序相邻R/G/L=41/6/7、churn13、J=.75926；无序41/9/6、churn15、J=.73214。
这些较小churn主要围绕弱source附近的行为，不能转译为新能力获取后的保持改善。

固定训练侧动作FM有序0／50／100为.153285339／.153149835／.152403202，无序为
.153285339／.153149911／.152400151；末节点约0.58%改善，有序23/24、无序24/24task方向改善。
保留这些有限学习事实，但训练闭环仍无可信source以上增量，不能把主要缺口仅归于未见任务、换视频或晚期遗忘。
100updates是预登记有界投入，并非普遍收敛或信息不可能性证明；微小梯度、非零参数及闭环相近也不能唯一认定优化或数值根因。

本轮降级“把局部监督接到实际参数出口就足以建立共享可迁移纠正知识”的具体联合解释。
它没有否定§94／97的privileged功能前提，也没有否定所有视频信息、固定LoRA控制或早期普通FM的合法能力。
本次同时改变局部预测、输出参数化与辅助监督，不能用它和Semantic Path旧分数的差额冒充单变量因果效应。

按登记关闭当前组合，停止续训及rank／scale／seed／LR／字段loss等小扫；不扩oracle或补未获资格的controls。
完整原件在`runs/analysis/local_correction_field_writer_20260914/`：READOUT、paired_readout、bounded100_decision、
全部训练／物化／评测合同、checkpoint、bank、raw rows及completion。训练代码50dafb05，推理／评测46aaf22d。
整体goal未完成；接续先综合已有学习正负证据，不从这个non-pass自动启动新的局部头、监督或优化链。

## 99. 全视频native X的PCA16出口保留部分真实作用，但不是共享获取证据（2026-09-14）

Process Pullback Writer在正式学习前，按已登记96条件复用原teacher16–19、四支持的真实纠正q／eta及query42–45。
本次唯一主要变量是完整stride5视频的native X所确定的PCA16投影G P；q用T/4还原原四支持平均，余下50×7位置为零。
de7237a8完成两个worker全部exit0、1,536条件query及t1／full10两读出；query用自身执行state，全部finite。

t1 source .11977705→G P .11523266，改善95%CI [.00142637,.00853139]，16/24task与四suite净正；
full10 .16493043→.15792518，CI [.00191831,.01375751]，20/24task与三suite净正。原注册功能前提通过。
相对原G平均改善保留72.56%／76.29%，原G−G P的MSE差CI均严格负，投影有实际功能损失，不能宣称无损或非劣。

该前提只支持继续检验此固定出口的合法学习，不证明无动作视频能预测有效q、不证明闭环或held转移，也不抵销§98的获取缺口。
原件、逐task／suite和task-cluster bootstrap在`runs/analysis/process_pullback_writer_20260914/functional/summary.json`。

## 100. 固定参数编译后的纯FM有局部获取，但未建立广泛迁移保持（2026-09-14）

Process Pullback Writer按登记完成fresh900更新、3,600条件／230,400跨episode queries，实际覆盖623/624个教学条件，
独立meta tasks仍为24。同步双路视频、过程Value、7维q、裸source固定导数及全视频PCA16出口共同接受唯一主FM；
本轮没有q辅助、frame_set训练臂、RL、Test或部署优化。训练代码d6defa69，物化／评测f5d9db78，六面板／1,488rows完整有效。

300／600／900的train96为24／22／26，source17；validation400为64／72／64，source47。
900的train source差额CI[+2.08,+17.71]pp、breadth9/24，固定独立动作FM19/24task改善且均值下降约3.08%。
这些是真实局部可学性；不能因最终未获资格改写成完全没有获取，也不能把更多同task条件说成更多独立任务。

Validation相邻R/G/L56/16/8、52/12/20，churn24／32、Jaccard .7000／.6190；train相邻则17/5/7、21/5/1。
训练任务后段保持改善没有同步转化为未见task保持。Validation Object task1为5→26→29→18，Long为1→2→3→0；
900新增Spatial task1与Goal task3各2/50，三个节点均未形成四suite同时非零能力。900 source差额CI[0,+10.75]pp。
固定q到LoRA关系因此没有充分保证这套共享读取／学习组合的广泛能力保持；本轮不把>145作为额外硬门槛。

未获能力／相邻前置资格，没有selected checkpoint、same-task-other或最终wrong／no-video／shuffled／reversed。
正确条件涨分与零变化结构均不能证明视频必要性或顺序特异性。§99的privileged功能前提继续保留，不能替代本次合法闭环。
本轮同时改变过程读取、参数出口并扩大了学习窗口，不能把相对旧局部场或语义路径的分差归因单一模块或取消辅助监督。

实际首步后的899次Writer／Action Meta／VL Meta梯度记录均finite非零、source冻结，未见所检查运行合同错误。
这不识别唯一优化根因；q、投影、读取器、任务支持及学习条件仍是竞争解释。train96用未参与梯度的teacher46–49，
不能单凭它区分训练池拟合与同task换视频／初始化泛化，900也不证明普遍收敛或不可学习性。
按本轮窗口停止原样续训和小扫，保留方法与完整证据；后续q辅助或其它实质修订、是否回到v5.2由owner决定。
完整原件在`runs/analysis/process_pullback_writer_20260914/`的READOUT、paired_readout、bounded900_decision及全部checkpoint／rows／合同。

## 101. 有界闭环对照定位固定出口缺口，方向特异性仍未成立（2026-09-15）

按原设计诊断注册固定v1终点900，完整train24同teacher16、共同A/B起点、相同支持动作和LBFGS预算，
source与Writer冻结，仅分别优化完整q或rank16 A/B。独立states32–35的source／原输出／free_q／free_AB为
17／20／18／46（均/96）；free_AB breadth19，S/O/G/L11/17/13/5，相对原输出R/G/L16/30/4，
相对free_q12/34/6、差额task-bootstrap95%CI[+13.54,+44.79]pp。三臂全部24任务有效，无选中间点或held梯度。

独立动作full10 MSE原输出／free_q／free_AB为.171922／.150169／.140512，实际前5步为
.144629／.129600／.118305。free_q功能误差改善未增加闭环，完整A/B则有跨suite实际能力；
因此固定编译出口的参数化／优化约束值得优先修正。相同优化器预算不等于相同优化难度，结果不证明数学容量上界、
PCA单一根因、可泛化共享L/R足够或合法RGB获取已解决。诊断拟合只作定位，永不作为后继初始化。

训练池teacher16–19的300/600/900为18/20/22，对独立teacher46–49的24/22/26，没有一致训练池优势。
换视频过拟合不能单独解释整体缺口。32组三节点共96条correct回放均已实际查看，95/96复现原成败；
唯一差异为Long task2/state0/600原成功、本次失败，不改原400。已确认错对象／实例、获取、放置及组合只做一项等
不同失败阶段，不能统一写成后半段遗忘，也不能由稀疏画面指认具体模型模块。模糊瓶标签保留不确定。

固定900的correct／other／wrong／shuffled／reversed／no-video为64／65／48／35／67／47（均/400）。
相对correct的other／wrong／shuffled／reversed差额CI分别[-1.75,+2.50]／[-9.50,-.25]／[-18.50,+1.75]／[-1,+2.50]pp。
内容匹配有支持，correct未优于倒序，乱序差额也有跨task不确定性；不能宣称有益时间方向或完整视频必要性已建立。
这组controls只描述旧模型，未用于后继出口选择。它不改写原窗口的未获资格结论。

所有数据原件在`runs/analysis/process_pullback_writer_20260914/causal_diagnostics/`：diagnostic_metrics、
reachability_functional_metrics、replay_pairing_metrics、各任务拟合、raw rows／completion、注册和来源。
临时拟合及static bank准入由已推送`adc31a15`与冻结运行树保存，完成使命后从canonical main退役。

## 102. 共享L/R出口扩大了局部验证收益，尚未建立广泛能力（2026-09-15）

依§101的train-only有界q／完整A/B差距，唯一主要修订是在原G P两侧加入共享、identity起步的乘法变换L/R。
保留双路RGB过程读取、完整50-horizon、7维q、冻结裸source导数、38-target唯一LoRA及同task跨episode纯FM。
两Meta与Writer fresh共同训练，未复用特权拟合初始化、未引入q辅助或RL；旧900视频controls未参与修订。
实现`e1ca29b9`、正式训练和全部新评测来自clean pushed detached `85ecfd18`。

Fresh900实际完成3,600条件／230,400queries、623/624合法teacher条件、九份完整checkpoint。
三段训练合计15,663.382秒，最大reserved43.809GiB；899次identity后更新的Writer及两Meta信用均finite非零，source冻结。
独立训练动作FM从.153279524降至.140811848，22/24task改善、均差task-bootstrap95%CI[.007163,.019066]。
训练闭环没有对应的普遍提升；loss下降和有效信用不能代替能力证据。

| 节点 | Train /96 | Train breadth /24 | Validation /400 | Validation breadth /8 | Validation S/O/G/L |
| --- | ---: | ---: | ---: | ---: | --- |
| Source | 17 | 7 | 47 | 3 | 0/5/41/1 |
| 300 | 21 | 9 | 50 | 4 | 0/16/32/2 |
| 600 | 20 | 8 | 75 | 3 | 0/38/34/3 |
| 900 | 18 | 11 | 79 | 3 | 0/40/39/0 |

Validation300→600与600→900的R/G/L为39/36/11、63/16/12，churn47／28、Jaccard .4535／.6923。
Train相邻为13/7/8、12/6/8，churn15／14、Jaccard .4643／.4615。
900对source的validation R/G/L37/42/10、CI[-2.5,+26.25]pp；train9/9/8、CI[-7.29,+9.38]pp。
900的77/79次成功集中于奶油奶酪入篮40及入碗37，余下2次为Goal3；Spatial与Long均零。
这说明局部验证收益在后两节点有所保持，但广泛获取、组合能力和迁移保持仍未成立。

固定旧900→新900的validation为64→79、R/G/L48/31/16、CI[-4,+16]pp；train26→18、13/5/13、CI[-19.79,+2.08]pp。
验证净增15来自Object1 +22、Goal6 −5和Spatial1 −2；保留实际局部改进，不能描述为全面修复或完全无效。
有限预算完整A/B拟合的正例只证明一处出口参数化／优化约束，不能推出共享L/R足够，或把剩余缺口唯一归于读取器、PCA或优化器。
900窗口也不是所有训练条件下的不可学习性证明。

能力non-pass和terminal900在所有新controls／Test之前登记并冻结，没有选择合格checkpoint。
最终correct／other／wrong／shuffled／reversed为79／81／53／70／78（均/400），四个新增面板的48个workers全exit0。
相对correct的四个差额CI为[-1.75,+2.75]／[-22,+2.75]／[-9.5,+2.25]／[-2.25,+1.5]pp，均包含零。
Other R/G/L67/14/12、churn26、Jaccard .7204；Wrong42/11/37、churn48；Shuffled57/13/22、churn35；Reversed67/11/12、churn23。
Correct相对wrong的局部优势主要来自Object1的40对11；倒序整体78接近correct79，有益时间方向未成立。
Same-task换视频的整体分数接近，不等于所有初态或未成功任务均鲁棒；这些controls仅作冻结后的描述，不反馈方法。
No-video生成8套完整零LoRA条件、覆盖400配对初态；最终身份与执行合同核验通过，复用source47，不冒充新增400条rollouts。

固定Test只在方法冻结及视频controls全部完成后准入，source86／correct49（均/400），breadth5→6，
S/O/G/L从19/0/45/22变为5/6/27/11。R/G/L34/15/52、churn67、Jaccard .3366；
差额−9.25pp，task-bootstrap95%CI[-20.25,+.75]pp。区间包含零，但该固定面板不能作为泛化改善证据。

| Test task（每task50条） | Source | 900 | Retained / Gained / Lost |
| --- | ---: | ---: | --- |
| Spatial6 | 2 | 0 | 0/0/2 |
| Spatial8 | 17 | 5 | 3/2/14 |
| Object0 | 0 | 6 | 0/6/0 |
| Object7 | 0 | 0 | 0/0/0 |
| Goal4 | 45 | 26 | 25/1/20 |
| Goal7 | 0 | 1 | 0/1/0 |
| Long0 | 4 | 2 | 1/1/3 |
| Long3 | 18 | 9 | 5/4/13 |

Object0和Goal7的新增能力保留为局部正例；其它三个suite的已有能力损失更大，不能由breadth增加宣称能力保持改善。
Test不反哺方法、训练或选点，不改变此前冻结的能力non-pass。新方法12个面板共3,888条rows、126个workers均正常结束，
各面板墙钟累计10,522.596秒（含跨节点并行）。完整配对、freeze、400-row视频臂全50视频无放回、checkpoint／method及no-video审计通过。
关闭本次修订窗口，不追加训练、小扫或新的设计；该non-pass只限制本轮检验的共享出口／过程读取／纯FM组合。
原件根`runs/analysis/process_pullback_learned_outlet_20260915/`保留paired_readout、paired_endpoint_revision_comparison、
final_paired_readout、bounded900_decision、method_freeze、全部checkpoint／bank／raw rows／completion及训练、物化和最终读出launch contracts。

## 103. 固定v5.2历史参考保留了能力量级，现行基线迁移须独立计量（2026-09-15）

原529da6b的step900在当前A40环境、原400个state–video与RNG映射下为125/400，历史132/400；
S/O/G/L从19/63/39/11变为15/58/41/11，breadth均6。逐行R/G/L111/14/21、churn35、Jaccard .7603；
任务配对差额95%CI[-4.25,+.75]pp。完整原件见runs/analysis/v52_return_20260915/reference_readout.json。
复核没有参数更新、换点、controls或Test；结果保留旧能力参照，但不是>145的正式资格，也不唯一分开硬件与闭环敏感性。

外层exit1来自旧汇总器把GPU3硬编码NUMA0；实际该卡在NUMA1，12个workers均exit0且400行完整。
基于PCI/NUMA/affinity原件仅修正汇总判断后完成汇总，原exit1与修正脚本保留，未重跑挑结果。

新fresh基线保留Core/Procedure/完整A/B和三组Meta，仅模型接口迁移双相机与完整50-horizon learned read；
另外明确登记offset0→1、严格跨episode及46/4训练／独立视频分池，不能把新旧差值单归某个模型模块。
旧v5.2实际RoPE读取raw frame indices，历史文字的sampled ordinal并未被执行；当前保留真实旧图。
当前依赖scheduler会把decay自动缩到run budget，入口明确保留原100 warmup／12000 decay时钟。
这些是新基线的可复核实施边界，不是能力改进结论；任务共现仍待在同事件／同更新对照中检验。

## 104. 正确时间对齐的裸source未证实稳健净改善，Writer效应须另行计量（2026-09-16）

从同revision generic π0.5，以审计后71tasks、全参数、1000更新／global256及原optimizer、LR、normalization重训，
监督从`obs[i]→actions[i]`改为真实未来的`actions[i+1:i+51]`，固定raw1000而不选EMA或中间点。
实际256000queries，全部1000行计数及loss／gradient／LR正常；四A40 micro8／accum8维持原logical8tasks×32采样。
完整模型／optimizer状态已落盘后的最终保存超时仅影响发布；恢复原调度和元数据未改写参数、EMA或rank RNG。
恢复事实和原始失败记录均保留，不把launcher exit1写成正常训练退出。

| 面板 | 旧source | 对齐source | Breadth旧→新 | 新S/O/G/L | Retained/Gained/Lost | Churn | Jaccard | task-bootstrap差值95%CI |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | --- |
| Validation | 47/400 | 50/400 | 3→3 | 0/1/42/7 | 36/14/11 | 25 | .5902 | [-2.75,+4.75]pp |
| Train | 17/96 | 13/96 | 7→8 | 2/1/6/4 | 8/5/9 | 14 | .3636 | [-13.5417,+5.2083]pp |

Validation按global1/3/11/13/23/26/31/32逐task为0/0/1/0/0/42/7/0，仍主要依赖Goal6。
Long由1→7，Object由5→1，Goal41→42，Spatial仍0；train的Long1→4、Object0→1，同时Spatial7→2、Goal9→6。
这些是固定面板上的局部获取与损失，不能抹掉其一，也不能把两个跨零区间当作等效证明。

两组496条闭环、15个workers正常完成，task/state/env/policy RNG及normalization配对检查通过；
bootstrap按task聚类、20000次、seed20260915。Validation外层Bash因运行中修改launcher末尾文本而exit2，
实际Python聚合、12个workers与400条原件全部成功；保留异常说明，没有重跑或筛选结果。
训练96首次准备缺少显式面板而被拒，补用旧source的完整train24固定32–35面板后正常完成，不更改任务或初态。

此时只能说裸policy尚无稳健整体收益。Source同时供执行与教学视频的原生观察计算，不能据此认定Writer侧影响有限，
也不能认定时间错位解释了全部历史负结果。新source上的fresh A/B及旧B同节点比较仍是独立、已登记的必要证据；
不据本结果扩source训练、挑checkpoint、恢复旧C或重启其它历史架构。
原件见`runs/analysis/source_alignment_20260915/source/{validation,train}`与两份`*_comparison.json`；
原训练／恢复及评测记录分别在`source_launch_contract.json`、`source_checkpoint_recovery.json`和`source_evaluation_launch_contract.json`。

## 105. 对齐source上的A出现获取，保持与source下游效应仍待比较（2026-09-16）

单agentview／完整50-horizon mean的A从`575c189a`fresh联合训练300次，保留v5.2共有初始化流、
baseline分组、每更新4task×21queries及严格跨episode／46+4分池。实际1200条件／25200queries，
3422.504秒；100／200／300完整checkpoint、原拓扑RNG和采样游标均已核验。
身份初始化打开后的298次更新均有Writer和Text／VL／Action Meta有效finite信用，source始终冻结。

| A300面板 | 新source | A | Breadth source→A | A S/O/G/L | Retained/Gained/Lost | Churn | Jaccard | task-bootstrap差值95%CI |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | --- |
| Validation | 50/400 | 99/400 | 3→5 | 10/38/31/20 | 30/69/20 | 89 | .2521 | [-3,+31.25]pp |
| Train | 13/96 | 36/96 | 8→18 | 11/11/10/4 | 9/27/4 | 31 | .2250 | [+12.5,+35.4167]pp |

Validation按global1/3/11/13/23/26/31/32为0/10/35/3/0/31/20/0；train按固定24task顺序为
2/4/1/1/1/2/1/3/1/2/3/1/0/4/3/0/3/0/2/1/1/0/0/0。
训练任务的独立视频闭环获取与覆盖增加；验证四suite均非零，但净收益区间仍跨零、原有成功丢失20条，
单节点尚不能判断保持。该99不满足>145，也不证明视频必要性或source相对旧底座的因果增益。
旧A900的125还涉及不同训练合同和曝光，不能直接与本300节点归因比较；后续节点及是否补新B服从owner于9月16日收敛后的范围。

496条件均一次完整编译，validation全50视频各一次，所有source／normalization／checkpoint／task／state／RNG配对通过；
18个最终workers均exit0。Validation首次在准备后被GPU实时准入拒绝，没有worker或rollout，
双节点复查后从原prepared queue启动并完整成功，原exit1未抹除；未改准入逻辑、重选视频或重跑结果。
两个面板墙钟1009.883／560.788秒，原件及逐任务配对为`runs/analysis/source_alignment_20260915/A/paired_readout.json`，
训练与评测launch contracts保留全部命令、资源和失败记录。Controls与Test均未开放。

## 106. A600继续增加训练获取，验证能力仍明显交换（2026-09-16）

对齐source上的A在相同四rank拓扑从300精确恢复至600；累计2400条件／50400queries，
六份完整checkpoint，step3起598次四组有效finite梯度保留，source冻结。第二段3392.728秒；
独立train held动作FM .112965→.105642，闭环仍是能力判断依据。

| A600面板 | 新source | A | Breadth | A S/O/G/L | Source配对R/G/L | Churn | Jaccard | 净差95%CI |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |
| Validation | 50/400 | 88/400 | 6 | 13/35/36/4 | 32/56/18 | 74 | .3019 | [-2.75,+23.75]pp |
| Train | 13/96 | 47/96 | 17 | 11/19/13/4 | 12/35/1 | 36 | .2500 | [+21.875,+50]pp |

300→600的validation R/G/L为57/31/42，churn73、Jaccard .4385，净差95%CI[-13,+5.75]pp；
train为29/18/7，churn25、Jaccard .5370，净差95%CI[+1.0417,+21.875]pp。
验证逐task（global1/3/11/13/23/26/31/32）为0/13/26/9/1/35/4/0；固定train24顺序为
3/3/1/1/3/0/4/4/1/3/4/3/2/4/3/0/4/0/2/2/0/0/0/0。

训练面板新增能力多于丢失，但验证99→88、Long20→4，源于不同条件能力的交换；不能由FM下降或breadth5→6
宣称迁移保持已修复。两个节点均无>145资格。旧B600为105，新A600为88，但底座及读取模式不同，
此处只保留曲线交叉事实；不能提前把B<A当成所有节点、所有底座上的定论。
按最新owner范围完成A的900／1200及共享SFT后，再判断是否需新B；不因此启动新方法或扩大预算。

两个600面板均完成完整严格配对，全部18个workers与两个launcher exit0；墙钟1120.709／667.014秒。
视频无放回、同checkpoint、source／normalization、state与RNG审计通过，原件仍为study的`A/paired_readout.json`和launch contracts。

## 107. 对齐A900达到140，获取回升而相邻稳定尚未建立（2026-09-16）

A从600在原四rank拓扑精确恢复至900；累计3600条件／75600queries、九份完整checkpoint。
本段3285.437秒、累计10100.668秒；step3起898次更新的Writer及三Meta梯度均finite非零，source冻结。
独立train动作FM .105642→.103778，能力结论取自完整correct400／train96。

| A900面板 | Source | A | Breadth | A S/O/G/L | Source配对R/G/L | Churn | Jaccard | 任务bootstrap净差95%CI |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |
| Validation | 50/400 | 140/400 | 6 | 17/60/30/33 | 36/104/14 | 118 | .2338 | [+1.5,+46.2563]pp |
| Train | 13/96 | 54/96 | 19 | 16/18/13/7 | 13/41/0 | 41 | .2407 | [+30.2083,+55.2083]pp |

600→900 validation R/G/L为68/72/20，churn92、Jaccard .425、净差95%CI[0,+28.75]pp；
train为39/15/8，churn23、Jaccard .6290、净差95%CI[-1.0417,+15.625]pp。
验证增加主要来自Long4→33和Object35→60，Spatial13→17，但Goal36→30；breadth仍6，不能只凭净增52称作保持修复。
Validation按global1/3/11/13/23/26/31/32各50条为1/16/44/16/0/30/33/0；
train按global0/2/4/5/7/9/12/14/15/16/18/19/20/21/22/25/28/29/34/35/36/37/38/39各4条为
4/3/3/2/2/2/3/4/2/2/4/3/2/4/3/0/4/0/2/4/0/0/1/0。

A的训练获取36→47→54、验证99→88→140，说明这个结构在正确时间对齐和严格跨episode合同下仍能产生较强能力。
但140没有超过145，相邻保持及两项零成功task仍未解决；该correct节点读出时没有selected checkpoint或视频controls，后续单独授权的封闭诊断见§109。
旧A900本机复核125→新140，名义+15、state/RNG配对R/G/L86/54/39、churn93、CI[-6.5,+16.75]pp；
旧原132→新140为90/50/42、churn92、CI[-7.5,+15.25]pp。旧A与本轮video seed分别7／20260911，
两者每task均全50视频各一次，但只有8/400行state–video完全相同；这些差额包含视频分配、source、标签及训练采样变化，
不能唯一归因时间修正，也不把宽区间解释成等效。
旧B900的77→新A140保持相同state–video映射，R/G/L57/83/20、churn103、Jaccard .35625、CI[+4.25,+28]pp；
仍同时改变source与读取模式，不能单独归因相机或H-read。A600为88而旧B600为105的曲线交叉事实仍保留。

两bank完整496条件sealed；validation9和train3个workers、两个launcher全部exit0，source／normalization／checkpoint／
state／env与policy RNG审计通过。Validation每task全50视频各一次，train沿用登记的4条独立教学视频；两个面板墙钟1147.038／868.728秒。
原件为`runs/analysis/source_alignment_20260915/A/{paired_readout.json,training_audit_900.json,node900_reference_comparisons.json}`及launch contracts。
Owner明确要求到此暂停讨论，不启动1200或B；已运行的SFT仅后台继续到400自动停止，后续评测和425／450续训均暂停。

## 108. 旧B的H-read未发生强权重集中，但差距仍不能唯一归因（2026-09-16）

为准备owner要求的B/A讨论，仅CPU读取旧B300／600／900／1200保存的1024维query与50维bias，
并核对原`7f9f11a3:src/ember/writer/video_program.py`。没有新增模型forward、native激活读取、训练、rollout或视频controls。
原公式用无affine RMSNorm的H产生logit，因此对任意H都有`||RMSNorm(H)|| <= sqrt(1024)`；
所有horizon logits的跨度至多`D = 2||q|| + (max(b)-min(b))`，单个softmax权重不超过`1/(1+49 exp(-D))`。
四点q范数为.11328／.16078／.21298／.26040，对应单位置权重上界2.533%／2.797%／3.106%／3.417%，均匀值为2%。
这排除当前B在这些点把完整50-H压为少数位置的强权重集中解释；不是实际注意力分布的测量，也不能证明小读出变化
对后续表示、信用分配或闭环无影响，更不能据此把差距全部归于双视角。

旧B自身validation97→105→77→85，而train35→49→50→56。600→900的验证损失包含source已成功条件保留39→20，
也包含相对source新增成功66→57；1200部分恢复source成功，却减少另一部分新增能力，不能概括成单一source遗忘。
旧A900本机125→旧B900的77还混有训练合同和392/400行teacher配对变化，两个架构角不能唯一分开双视角与H-read。
这些边界及精确参数上界分别保存在study的`archival_A_B_comparison_boundary.json`和`archival_B_horizon_weight_bound.json`；
不据此自动补2×2、改架构或恢复训练，后续由owner讨论决定。

## 109. 新A900有有限内容特异性，旧强视频／时序特异性未复现（2026-09-16）

Owner单独指定已完成的A900做视频特异性诊断；固定`macro_00000900`，复用correct400，补五个完整400面板。
全部初态、env／policy RNG、video ordinal及seed20260911严格配对。每个有视频臂逐task全50条视频各一次；
other逐行换同task视频，wrong保留目标language并使用固定跨suite donor，shuffled／reversed重排真实RGB后完整forward。
这不是合格checkpoint选择；不反馈训练或架构，Test关闭，A1200及其它暂停阶段不启动。

下表R/G/L均从correct140转向相应对照，区间为8task、20000次bootstrap（seed20260915）的对照减correct差值。

| 条件 | 成功/400 | S/O/G/L | Breadth/8 | Correct配对R/G/L | Churn | Jaccard | 净差95%CI |
| --- | ---: | --- | ---: | --- | ---: | ---: | --- |
| Correct | 140 | 17/60/30/33 | 6 | — | — | — | — |
| Same-task-other | 136 | 17/59/25/35 | 5 | 117/19/23 | 42 | .7358 | [-4,+1.5]pp |
| Cross-suite-wrong | 116 | 19/45/24/28 | 5 | 92/24/48 | 72 | .5610 | [-12,-.5]pp |
| No-video／零LoRA | 48 | 0/1/42/5 | 4 | 31/17/109 | 126 | .1975 | [-47.5,-1.25]pp |
| Shuffled | 128 | 13/46/35/34 | 5 | 100/28/40 | 68 | .5952 | [-8.25,+2.25]pp |
| Reversed | 139 | 10/59/36/34 | 6 | 104/35/36 | 71 | .5943 | [-5.25,+4.5]pp |

按global1/3/11/13/23/26/31/32，每task50条：correct为1/16/44/16/0/30/33/0，
other为0/17/45/14/0/25/35/0，wrong为0/19/40/5/0/24/28/0，no-video为0/0/1/0/0/42/4/1，
shuffled为0/13/37/9/0/35/34/0，reversed为1/9/42/17/0/36/34/0。

Same-task换视频只净少4条，支持本面板上的总体鲁棒性，但仍有42个初态成败交换，不能宣称逐初态不变或统计等效。
Wrong相对correct净少24条／6pp，损失主要在Object（60→45），Goal30→24、Long33→28，Spatial17→19；
相对other净少20条，R/G/L87/29/49、churn78，区间[-10.5,-.25]pp。两种正确视频映射下方向一致，
支持有限的视频内容特异性；wrong仍达到116/400，不足以证明正确教学内容是所有已获能力的必要条件。

Shuffled相对correct净少12条／3pp，reversed仅净少1条／.25pp，两区间均跨零；
相对other的差值区间也均跨零（shuffled[-8.25,+5.25]pp、reversed[-6,+8.25]pp）。
倒序的Goal30→36、Long33→34抵消了Spatial17→10及Object60→59；总分接近伴随71个初态交换，
所以不能说顺序完全不影响输出，但没有建立正确时间方向或有序动态的稳定性能优势。

No-video按既有合同使用完整零A／B LoRA、零Writer调用和零teacher RGB读取，等价于移除生成的适配器。
新48与先前裸source50接近：source→no-video为45/3/5、churn8、Jaccard .8491、区间[-2.25,+.75]pp。
这不要求跨物理分片的逐bit或逐episode一致，也不能把整套适配器带来的增益全部解释成视频条件增量；
它不是language-only Writer，更不是静态图像对照。本次没有新增这两种基线或进一步训练。

五个新面板共2000条，48个最终workers与五个最终launcher均成功；checkpoint／source／normalization、
400条件映射、信息墙、真实RGB变换和逐行manifest证据均审计通过。Other／wrong／no-video／shuffled／reversed
墙钟为1174.783／1555.150／1694.250／920.068／954.094秒。Other和no-video各有一次启动前GPU准入拒绝，
均没有rollout；原失败日志保留，后续成功结果独立核验。Owner要求只在完成后看结果，后半程采用后台顺序执行与完成事件等待。
原件为`runs/analysis/source_alignment_20260915/A/video_specificity_step900/{readout.json,readout.md,launch_contract.json}`及各臂rows／completion。
诊断已结束；140仍未过>145及相邻稳定资格，不选模型、不反馈训练或架构，继续保持owner暂停边界。

## 110. 对齐source的共享SFT三点85／89／86，有限净获取未补足广度（2026-09-17）

Fresh共享rank128 LoRA从对齐source raw1000出发，固定`16c81e29`运行树，450更新／259200 queries，
每task10800queries。保持历史global576、原2400步LR时间轴及原生LoRA dtype；监督使用offset1，
24个train tasks每次更新各24queries，全部50 episodes/task覆盖，validation/test actions读取为零。
400→425→450均在原gpu02两rank完整恢复，训练与checkpoint／optimizer／scheduler／sampler／RNG／cursor审计通过。

三个正式面板均为同一组validation8×50初态及env／policy RNG，每个suite分母100；逐task按global1/3/11/13/23/26/31/32。

| 节点 | 成功/400 | S/O/G/L | Breadth/8 | 逐task成功数 | 相对source50的R/G/L | 净差95%CI |
| --- | ---: | --- | ---: | --- | --- | --- |
| 400 | 85 | 7/43/25/10 | 6 | 1/6/39/4/0/25/10/0 | 25/60/25 | [-9,+31]pp |
| 425 | 89 | 9/48/23/9 | 5 | 0/9/45/3/0/23/9/0 | 25/64/25 | [-10.25,+35.25]pp |
| 450 | 86 | 10/36/27/13 | 5 | 0/10/33/3/0/27/13/0 | 28/58/22 | [-6.75,+27.5]pp |

区间为8task cluster bootstrap、20000次、seed20260915。相对source的churn为85／89／80，Jaccard .2273／.2193／.2593。
相邻400→425的R/G/L为66/23/19、churn42、Jaccard .6111、净差区间[-2,+4.75]pp；
425→450为64/22/25、churn47、Jaccard .5766、区间[-8,+4.25]pp。
总分接近不代表成功集合不变；global23与32三个点均零，Object主要依赖奶酪任务。四suite均非零且名义净获取保留，
但未过>145，也没有充分任务广度；宽区间不能解释成与source等效。

同节点旧SFT109／107／74→新85／89／86，R/G/L分别49/36/60、49/40/58、34/52/40，
churn96／98／92，净差区间[-19.5,+4]／[-17.75,+7]／[-6.5,+12.75]pp。
source与训练动作offset共同变化，且旧A100／runtime面板没有重跑，不能把差额唯一归于时间修正或某个硬件变量。
也不能由此减轻新基线本身的能力缺口，或把它当Writer视频增量的证明。

450步连续finite，训练墙钟32994.300秒（9.165h），三次评测共4980.081秒（1.383h）；
首启至末评测结束38528.929秒（10.702h）。三个面板各400行、各6 workers，所有最终launcher／workers均exit0，
policy／环境／RNG／normalization配对审计通过。没有工程异常、checkpoint选择或Test，全部SFT作业已停止，不续训。
原件为`runs/analysis/source_alignment_20260915/SFT/{paired_readout.json,training_audit.json,completion.json,training_launch_contract.json}`。

## 111. 对齐A至1800仍有能力交换，视频特异性没有随续训单调改善（2026-09-17）

Owner授权从A900继续观察性能和视频特异性，并在1800有反弹时继续，直到能判断明显过拟合趋势。
以下只覆盖已经完整结束的1200／1500／1800；没有选择checkpoint、开启Test或预定本轮B。
架构、source raw1000、四rank、global84、46/4分池、跨episode、offset1及原optimizer／LR均保持。
1200来自原冻结575c189a，1500／1800来自仅扩预算的6393cbe1；原事件前缀及完整恢复合同审计通过。
1800累计7200个条件／151200queries，每task300个条件，1104个不同teacher条件各暴露6–7次；
source冻结，identity后Writer与三Meta梯度finite非零，累计训练19901.951秒（5.528h）。

每个节点均为同一固定teacher–初态映射的correct400与train96；S/O/G/L分母分别100／100／100／100和24／24／24／24。
表中R/G/L从前一个300更新节点出发，区间为任务cluster bootstrap、20000次、seed20260915。

| 节点 | Validation/400 | S/O/G/L | Breadth/8 | 相邻R/G/L | Churn | Jaccard | 净差95%CI | Train/96 | Train S/O/G/L | Train breadth/24 |
| --- | ---: | --- | ---: | --- | ---: | ---: | --- | ---: | --- | ---: |
| 1200 | 135 | 14/55/42/24 | 6 | 98/37/42 | 79 | .5537 | [-8.5,+6.5]pp | 62 | 20/19/13/10 | 20 |
| 1500 | 112 | 4/47/40/21 | 6 | 82/30/53 | 83 | .4970 | [-15,+2.5]pp | 56 | 19/17/13/7 | 19 |
| 1800 | 122 | 13/46/35/28 | 5 | 79/43/33 | 76 | .5097 | [-3.75,+9]pp | 63 | 19/21/12/11 | 20 |

Validation按global1/3/11/13/23/26/31/32逐task分别为0/14/37/18/2/40/24/0、
0/4/44/3/1/39/21/0、0/13/39/7/0/35/28/0。Spatial1与Long2三个点都零，1800又失去Goal3的少量成功。
Train相邻R/G/L为48/14/6、50/6/12、50/13/6，churn20／18／19，Jaccard .7059／.7353／.7246；
净差区间[0,+16.6667]／[-15.625,+2.0833]／[0,+15.625]pp。完整train24逐task数据保留在原始JSON。
全部correct轨迹为99→88→140→135→112→122，train为36→47→54→62→56→63。
1500两侧都回落、1800两侧都反弹，不能单独把1500命名为明确过拟合，也不能把1800净增10命名为保持修复。
训练获取长程扩大而未见任务未超过900，但还要按owner要求继续观察多个完整节点；未达>145或相邻稳定资格。

每个有视频臂每task全50条teacher各一次；other换同task视频，wrong保持目标language，时间臂重排真实RGB后完整forward。
No-video始终复用同source零LoRA的48/400，不能解释成language-only Writer或静态视频baseline。

| 节点 | Correct | Other | Wrong | No-video | Shuffled | Reversed | Correct−wrong | Correct−shuffled | Correct−reversed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 900 | 140 | 136 | 116 | 48 | 128 | 139 | 24 | 12 | 1 |
| 1200 | 135 | 128 | 106 | 48 | 107 | 121 | 29 | 28 | 14 |
| 1500 | 112 | 120 | 104 | 48 | 108 | 89 | 8 | 4 | 23 |
| 1800 | 122 | 127 | 98 | 48 | 112 | 113 | 24 | 10 | 9 |

1200 correct相对shuffled／reversed的优势区间为[+1,+13.25]／[+.25,+8]pp，确有该节点上的有益顺序证据。
1500相对shuffled为[-1,+3]pp，reverse为[+1,+13.25]pp；1800两者又为[-4.5,+8.5]／[-3.5,+9]pp。
900→1500的reverse差额1→23，来自correct少28、reverse少50，其中差额增幅22中的21来自Goal6；
这主要是倒序对照退化，不能叫正确视频能力提高。1800 correct回升10，但reverse回升24，差额随即23→9。
900→1800 wrong差额同为24，是correct与wrong各少18；shuffle差额12→10、reverse1→9。
这些相对900的差额变化区间均跨零，没有持续增强的视频特异性证据，旧v5.2强时序特异性仍未复现。
1800 correct和other相对wrong仍分别净多24和29，优势区间[+.5,+12]／[+2,+13]pp，保留有限内容特异性。
Other三个新节点相对correct为−7／+8／+5，但仍有67／38／51个初态成败交换；总分接近不代表逐初态稳定或统计等效。

三个新节点各2096条、72个最终workers，共6288条及216个最终workers均成功；source／checkpoint／normalization、
teacher／初态／RNG及逐行manifest审计通过，时间臂验证真实RGB重排。各节点评测墙钟5117.693／5104.992／5109.434秒。
1200 train和1800 wrong曾在worker前GPU准入被拒，均无当次rollout，从同prepared队列恢复；原失败记录保留。
视频controls只用于本次授权的演化与问题分析，不用于续训预算、checkpoint选择或训练loss；1800续训判断仅来自主面板反弹。
原件为`runs/analysis/source_alignment_20260915/A/continuation_readout.json`、各`video_specificity_step*/`、
`training_completion_*.json`、原与扩展launch contracts、完整checkpoints及两份`continuation_failure_*.json`。

## 112. A2100总分持平仍在交换任务能力，继续观察过拟合趋势（2026-09-17）

2100从1800完整恢复，使用clean pushed detached329987c0；只扩训练预算，架构、配方、四rank／GPU UUID和原LR不变。
累计8400条件／176400queries，每task350条件，1104个teacher条件各7–8次；完整checkpoint和原事件前缀审计通过。
source冻结，Writer与三Meta的identity后梯度finite非零，本段3224.685秒，累计训练23126.636秒（6.424h）。

| 面板 | 成功 | S/O/G/L | Breadth | 相对source的R/G/L | 1800→2100 R/G/L | 相邻churn | 相邻Jaccard | 相邻净差95%CI |
| --- | ---: | --- | ---: | --- | --- | ---: | ---: | --- |
| Validation | 122/400 | 1/53/42/26 | 5/8 | 43/79/7 | 92/30/30 | 60 | .6053 | [-7.75,+6.5]pp |
| Train | 60/96 | 20/20/12/8 | 19/24 | 11/49/2 | 52/8/11 | 19 | .7324 | [-12.5,+6.25]pp |

Validation逐task按global1/3/11/13/23/26/31/32为0/1/42/11/0/42/26/0。
相对1800，Spatial13→1、Long28→26，被Object46→53与Goal35→42抵消；仍有30次获得和30次丢失。
对source原成功的保留34→43，但新增88→79，两者也相互抵消；总分持平不意味着能力组合稳定。
两个各42次成功的task占84/122（68.9%），三个task仍零，任务广度没有扩大。
Train63→60，源于获得8、丢失11；这段没有训练能力增长与验证持续下降的共同证据，尚不足以判明显过拟合。
依据已完成主面板登记2400继续观察，保存2200／2300／2400，累计目标9600条件／201600queries；
决定与原件保存在`A/node2100_primary_readout.json`及`A/trend_continuation_launch_contract.json`，controls没有参与该判断。

同task换视频、跨suite错视频及真实RGB重排均沿原配对合同。下表R/G/L从correct122转向对照，
区间按correct减对照给出（任务cluster bootstrap、20000次、seed20260915）。

| 条件 | 成功/400 | Correct→control R/G/L | Churn | Jaccard | Correct−control 95%CI |
| --- | ---: | --- | ---: | ---: | --- |
| Other | 121 | 99/22/23 | 45 | .6875 | [-2.25,+3]pp |
| Wrong | 110 | 92/18/30 | 48 | .6571 | [-1.75,+8.25]pp |
| No-video | 48 | 41/7/81 | 88 | .3178 | [+2.25,+39.75]pp |
| Shuffled | 106 | 89/17/33 | 50 | .6403 | [-1.25,+10.75]pp |
| Reversed | 111 | 86/25/36 | 61 | .5850 | [-2.25,+8.75]pp |

Correct相对wrong／shuffled／reversed净多12／16／11，三个区间均跨零；other相对shuffled净多15，
区间[+.5,+8.75]pp，保留这项局部顺序证据，但other相对wrong／reversed的区间跨零。
1800→2100的wrong差距24→12来自wrong提高12而correct不变；shuffle／reverse差距10→16与9→11
来自两个对照分别下降6和2，也不是正确视频能力提高。相对900的三项差距变化区间均跨零，没有持续增强的特异性。
No-video仍复用同source零LoRA48，不是learned language-only或静态视频对照；总适配收益不能全部归给有序动态。

2100全部2096条新评测与72个最终workers均成功，评测墙钟5015.263秒；source／checkpoint／normalization、
固定初态／RNG／teacher、每task全50视频无放回、逐行manifest及真实RGB重排全部审计通过。本节点没有工程异常。
完整原件为`runs/analysis/source_alignment_20260915/A/continuation_readout.json`、`video_specificity_step2100/`、
`training_completion_2100.json`及`trend_continuation_completion_2100.json`；后续状态归progress，不将本节点当作A或B全目标完成。

## 113. A2400两侧总成功回落，训练覆盖扩展，时序优势仍未恢复（2026-09-17）

2400沿原四rank从2100完整恢复，冻结329987c0、原模型／配方／LR不变；累计9600条件／201600queries，
每task400条件，1104个teacher各8–9次。完整恢复点、原事件前缀与source冻结审计通过，identity后的四组梯度finite非零。
本段3150.601秒，累计训练26277.237秒（7.299h）。

| 面板 | 成功 | S/O/G/L | Breadth | 相对source的R/G/L | 2100→2400 R/G/L | 相邻churn | 相邻Jaccard | 相邻净差95%CI |
| --- | ---: | --- | ---: | --- | --- | ---: | ---: | --- |
| Validation | 106/400 | 5/40/40/21 | 5/8 | 37/69/13 | 87/19/35 | 54 | .6170 | [-9.5,+1]pp |
| Train | 58/96 | 17/16/15/10 | 22/24 | 12/46/1 | 47/11/13 | 24 | .6620 | [-14.5833,+9.375]pp |

Validation逐task按global1/3/11/13/23/26/31/32为0/5/38/2/0/40/21/0。
相对2100，Spatial1→5，但Object53→40、Goal42→40、Long26→21；总成功净少16且旧成功丢失35。
Train总成功少2而任务覆盖19→22；Object20→16、Spatial20→17，Goal12→15和Long8→10。
覆盖扩展伴随原有成功条件丢失，不能仅用总分下降说没有学习，也不能把两侧回落直接叫典型过拟合。
按owner要求追加2700（2500／2600／2700保存、累计10800条件／226800queries），判断仅用主面板，
已写`A/node2400_primary_readout.json`及扩展launch contract，视频controls未参与是否延长。

完整correct／other／wrong／no-video／shuffled／reversed为106／103／91／48／107／95。
从correct到各对照的R/G/L分别为87/16/19、70/21/36、36/12/70、81/26/25、76/19/30，
churn35／57／82／51／49，Jaccard .7131／.5512／.3051／.6136／.6080。
Correct减wrong／shuffled／reversed的优势区间为[-2,+10.75]／[-3.5,+2.25]／[-.75,+6.5]pp，均跨零；
other相对这三个对照的区间也均跨零。Correct与other本身差3、区间[-.75,+2.25]pp，并非统计等效证明。
相对2100，wrong差距12→15由correct少16、wrong少19形成；shuffle差距16→−1因correct少16而shuffle多1，
reverse差距保持11但两侧都少16。没有正确视频能力或时序利用随训练增强的证据，不把对照退化计为改进。
No-video仍是同source零LoRA48，非learned language-only或静态视频baseline；信息与因果解释边界不变。

2096条新评测及72个最终workers全部成功，评测墙钟5137.134秒；source／checkpoint／normalization、
初态／RNG／teacher、全50视频无放回、逐行manifest与真实RGB重排审计通过，本节点没有工程异常。
完整原件为study的`A/continuation_readout.json`、`video_specificity_step2400/`、`training_completion_2400.json`和
`trend_continuation_completion_2400.json`。该节点未达性能或稳定资格，后续状态归progress，不提前裁决本轮B。

## 114. A2700训练成功回升，验证仍在低位，correct与倒序的差额转负（2026-09-17）

2700从2400完整恢复，329987c0、原四rank／GPU UUID、模型／配方／LR不变；累计10800条件／226800queries，
每task450条件，1104个teacher条件各9–10次。完整checkpoint、原事件前缀、source冻结与四组有效梯度审计通过。
本段3069.152秒，累计训练29346.389秒（8.152h），reserved峰值24.305GiB。

| 面板 | 成功 | S/O/G/L | Breadth | 相对source R/G/L | 2400→2700 R/G/L | 相邻churn | 相邻Jaccard | 相邻净差95%CI |
| --- | ---: | --- | ---: | --- | --- | ---: | ---: | --- |
| Validation | 108/400 | 6/50/37/15 | 5/8 | 35/73/15 | 81/27/25 | 52 | .6090 | [-5,+7]pp |
| Train | 64/96 | 20/20/16/8 | 20/24 | 11/53/2 | 48/16/10 | 26 | .6486 | [-2.0833,+15.625]pp |

Validation按global1/3/11/13/23/26/31/32为0/6/38/12/0/37/15/0；Object13从2→12，
抵消Goal26从40→37、Long31从21→15的退化及其它交换，名义净回升2不等于广泛恢复。
Train按固定24task顺序为3/4/3/3/4/3/4/4/4/2/3/3/2/4/4/2/4/0/2/3/3/0/0/0；
成功增加6但覆盖22→20，Long37／38重新归零，Long总成功10→8。获取与保持仍需分开判断。
相对1200，validation135→108、R/G/L82/26/53，净差95%CI[-11.75,-2]pp；train62→64、51/13/11，CI[-7.2917,+12.5]pp。
相对900，validation140→108、83/25/57，CI[-18,+1.25]pp；train54→64、47/17/7，CI[+1.0417,+20.8333]pp。
较长窗口已有训练能力保持／提高而验证回落的分离，不能将最近+2视作已恢复900／1200能力；
同时遵循owner“反弹继续”的要求，再登记3000确认后续趋势。该决定只用完整主面板，保存于`A/node2700_primary_readout.json`。

| 条件 | 成功/400 | Correct→control R/G/L | Churn | Jaccard | Correct−control 95%CI |
| --- | ---: | --- | ---: | ---: | --- |
| Other | 111 | 90/21/18 | 39 | .6977 | [-3.5,+1]pp |
| Wrong | 93 | 70/23/38 | 61 | .5344 | [-2,+10.25]pp |
| No-video | 48 | 33/15/75 | 90 | .2683 | [+.5,+34]pp |
| Shuffled | 106 | 74/32/34 | 66 | .5286 | [-2,+4.25]pp |
| Reversed | 115 | 79/36/29 | 65 | .5486 | [-6,+2.5]pp |

Correct相对wrong／shuffled／reversed净差15／2／−7，三个区间均跨零；other对这三臂的区间也跨零。
2400→2700的wrong差距仍为15，两臂都增加2；shuffle差距−1→2，由correct多2、shuffle少1形成。
Reverse从95→115，而correct仅106→108，使correct−reverse从11→−7，差额变化CI[-8.5,-.75]pp。
这是该区间相对倒序优势的下降；当前correct与reverse本身差值的区间仍跨零，不能称倒序普遍更好。
从900到2700的wrong／shuffle／reverse差额变化−9／−10／−8，三个区间仍跨零。
旧v5.2的强时序特异性未复现，1200的局部时序正证据没有随继续训练保持。No-video仍复用同source零LoRA48，
不是learned language-only或静态视频baseline，不能将全部适配收益归给有序动态。

完整2096条新评测、72个最终workers通过，评测墙钟5096.471秒；source／checkpoint／normalization、
固定teacher／初态／RNG、全50视频无放回、逐行manifest与真实RGB重排全部审计通过。
Validation首次五卡准备后在worker前被GPU准入拒绝，保留原未启动队列，在可用四卡上按相同科学条件重新准备；
其后temporal物化前的一次quota SSH连接关闭经复查恢复，未重训或重跑已完成面板。两次失败和完整完成证据均保留。
原件为study的`A/continuation_readout.json`、`video_specificity_step2700/`、`training_completion_2700.json`、
`trend_continuation_completion_2700.json`与`trend_continuation_failure_2700_01/02.*`；尚未达到性能或相邻稳定资格。

## 115. 全面证据审读限定v5.2解释，并形成原生中层跨帧设计（2026-09-17）

Owner要求先全面整理正负证据与验证强度，再从实际task及理论数学交付完整架构；本次仅分析与文档，无新实验。
首轮原件为[证据审计](docs/v52_evidence_audit_20260917.md)和Git `426dc5be:docs/v52_evidence_based_writer_design.md`；
[架构推导](docs/v52_evidence_based_writer_design.md)为同用途持续更新的canonical文档，后续统一设计见§116。首轮审计
覆盖46个编号证据组及12组bank中间路线、预算/曝光、代码、比较混杂、充分/不足范围和专家论证修正。

旧v5.2普通FM的真实能力及视频依赖不能抹除；但同主图的task-complete和当前A说明其性质不由架构名称自动保证。
同曝光配方交互、具体固定出口损伤、P/Q完整出口局部增益均成立；完整P/Q、Semantic-Path和旧source dual/H50 B的负例，
又阻止把“真实视频、完整H、共享块、fullAB、纯FM”共同框架当作v5.2的独有成功原因。
较可信的是内容保留、条件化且归一化的共享输出坐标、共同读取适配与优化过程的组合解释，未识别唯一原因。
没有找到v5.2同前端/训练下删P或AdaLN、两层P改summary的matched fresh行为消融；保留尾端不等于证明各部件必需。

审计纠正：PNBTT只跑task-local free-query E1，Natural Program E2未跑；functional-polar只有profile；
P/Q width256有训练/功能但无闭环；LocalField100仅400总条件、单task8–25次；G2-B macro60实际420次Adam。
Layered192/384的correct400只有255个不同task-video条件，不满足现行无放回合同；
但旧v5.2原始132已按每task全50视频各一次执行，不能按日期将所有早期结果统一降级。
更完整的预算与边界见证据文档；短窗口non-pass不是整个函数类的充分否定，长步数局部solver也不是合法共享学习。

对当前真实源码可作两个确定推导：旧v5.2逐帧native之后的Core是帧集合算子，固定参数下对帧置换不变；
原生prefix不读Action suffix，所以H-only中层跨帧注入有∂Core/∂H=0。所有旧显式顺序影响只能经晚期P调制进入输出。
因此本次设计选择在原生第9层后，取真实task-span Z与全部H50，经同构帧内混合/双向时间attention/FFN块，
同时残差写回Z/H，再由第10–18层消费；保留v5.2内容读取与完整输出端。首个实例N4、d256，解析总规模约16M。
时间位置只用于时间Q/K，帧内块不读frame位置，故重复静态视频不会单靠位置产生变化；同时保留非零静态Core能力。

真实train0/20/37/39说明应处理初始关系选物、接触与状态转换、物理前置和目标合取：
微波炉39初始已开，双物入篮37未规定物体先后，不能虚构总序。固定LoRA仍根据机器人自身观测执行，不能复制teacher时钟。
跨episode FM允许已知task的静态解；新设计提供过程进入主生成路径的接口，不能强制优化学会有用动态。
同样，native后半段能否消费新上下文、是否保持旧能力、是否改善合理条件变化下的特异性与保持，都尚未实测。
文档已明确fresh全Writer/三Meta联合纯FM、建议曝光、matched dual/H50基线、成本和可证伪预测；不注册为可执行active run。

## 116. 统一结构继承处理原则，不把缺少消融误作保留旧模块的理由（2026-09-17）

Owner要求再次设goal，检查更协调的统一架构及原多通路形成链，提前回答可预见问题并形成明确设计立场。
本轮复用完整历史证据审计，只新增文档/数学/接口分析，没有新模型forward、训练、评测或held数据使用。

最初新方案的多通路是：丰富内容可直达compiler；跨帧关系触发真实帧重读并进入AE内部；原始H及真实prefix保留。
shared compiler当时尚未规定Core/P/AdaLN。后改成单次forward中层桥，再为控制改变量接回旧尾端；
曾把信息访问重叠误当功能可替代，进而建议删P，这一理由不成立。反过来，没有matched删除消融也不禁止整体重构。
本轮明确继承因果职责而非模块清单，不再混用实验归因与最终架构统一两种选择标准。

选择[统一设计](docs/v52_evidence_based_writer_design.md)：native1–9→联合Z/H block×2→双残差写回→native10–18及final norms→
同构block×2→唯一M。320个slot的S0=0，首个同构decoder从M语义位置初始化内容，第二个由S读取全M；
保留归一化、八组256→216→native完整A/B heads与三个rank4 Meta。取消独立Core/P、P中心化及专门AdaLN。
M语义位置已经联合H与时间，不是静态Core；原生完整patch/H保留到对应读取，末端没有先做H均值。
此选择不是旧图函数类包含定理，旧AdaLN可能有用；整体fresh匹配行为负责，而非宣称模块已被证明冗余。

严格限定旧H-only结论：它不能通过native mask改变旧Core；新末端联合块已可让H影响语义M与输出。
新设计双写回的理由是让原生后半VL与Action均消费上下文，不能继续误用旧Core零导数声称新图H-only完全无输出路径。
若过程知识被广播为所有帧相同b(V)，P中心化会从Value删掉b；统一M可保留其直接Value作用。
该代数例子说明去掉一个具体限制，不证明旧模型实际因此失败或新模型必然学到b。

在逐token norm、同probe、时间仅进逐role Q/K、末端无frame地址、正确padding等条件下，整个新Writer对完全重复静态帧的次数T不变。
不同真实帧的顺序敏感只是函数类能力；若frame连同原时间标签一起置换，仍是同一个带位置集合。
现有frame_control将内容置换与natural positions分开，继续满足真正顺序干预的语义。
零B/U只带来有限的信用开启过程，不形成随段数相乘的门；非零路径也不保证梯度足够。source冻结不允许no_grad切断中间输入信用。
S0=0与地址仅Q/K只保证M=0时不产生内容；语言/静态仍可通过真实native Value和寻址被学习，不能据此宣称视频必用。

按明确规格独立复核参数13,451,008；T105/m25的新增attention逻辑pairs每head为11,922,800（不含原生网络），
不是实测吞吐/峰值，也不是线性T复杂度。尚未profile，当前不为文档工作检查GPU或quota。
新图仍以合法同task跨episode纯FM共同学习，source、split、数据墙和正式paired400资格不变；所有旧运行继续暂停。
内容保留、统一表示、共享坐标与identity起点不构成闭环保持保证。判断因新证据或真实合同冲突而修订，不因提出一个已登记限制就反转。

## 117. 统一Writer正式窗口：早期获取后在1200出现明显验证回落（2026-09-18）

按Owner授权实现两处统一Z/H处理、一次原生双写回及连续参数读出，三Meta与Writer fresh共同纯FM。
六卡两组三卡分别执行真实native帧与query分片，每次仍为4task／84queries，一次梯度汇总和optimizer更新；
原生不等长及空分片、完整FM汇总梯度与恢复验证通过，source始终冻结。
同事件六卡热身8.61秒对四卡12.52秒为1.45倍吞吐；最长105帧完整更新选择20帧chunk，
实际allocated峰值34.04GiB、15.18秒。这些是执行证据，不是科学能力收益。

首段600更新＝2400条件／50400queries，含启动及训练任务诊断共6100.5秒；每100完整恢复状态保留。
首个正式correct400与固定train96面板全部完成，实际teacher／state／RNG、source、normalization、
official处理及所有workers退出均通过原有比较器检查；validation每task全部50条teacher各一次。
历史A使用agentview／H均值，新方法为双RGB／fullH；本轮是整体方法比较，不能把差值单因归给统一结构。

| Step | 新模型validation / train | 既有A validation / train | 新模型breadth | Validation S/O/G/L | Train S/O/G/L |
| ---: | --- | --- | --- | --- | --- |
| 600 | 90/400；38/96 | 88/400；47/96 | 4/8；17/24 | 0/37/32/21 | 9/11/12/6 |
| 900 | 109/400；54/96 | 140/400；54/96 | 6/8；18/24 | 9/43/31/26 | 15/20/12/7 |
| 1200 | 67/400；54/96 | 135/400；62/96 | 5/8；20/24 | 1/17/36/13 | 16/17/12/9 |

600相对A的validation R/G/L为47/43/41、churn84、Jaccard .3588；train为28/10/19、churn29、Jaccard .4912。
任务cluster bootstrap新−A差值95%CI为[-9.75,+12.25]pp与[-20.83,+2.08]pp，首点没有整体提升证据。
新模型validation Long21对A4，但Spatial0对A13；train Object11对A19。差距同时涉及早期训练获取与未见任务覆盖，
还不能只归因为迁移，或指认某个内部模块失败。这里两个不同方法的R/G/L不能直接解释为同一模型的时间遗忘。

相对裸source50/400、13/96，validation R/G/L31/59/19、churn78，train9/29/4、churn33。
差值95%CI分别[-3.5,+27]pp与[+13.54,+39.58]pp；训练任务有获取，validation的跨任务不确定性仍大。
没有learned language/static对照或最终controls，不能把新增适配归给动态视频。
600点按预注册1200前不以一个低分点否定结构，保持全部训练条件与优化时钟续到900；
没有进行rank／scale／seed／LR小扫、Test读取或用最终controls返工；v5.2比较只复用已有正式结果，
此前误启动后中止的重复run不进入科学比较。

900点validation相对A少31个成功，差值95%CI[-13.5,-1.75]pp；train总分54追平A54，
训练Object由11→20、Spatial由9→15，而未见Object43仍低于A60、Spatial9低于A17、Long26低于A33。
目前差距更集中于未见任务表现；这不识别某个模块的因果失败，也不证明后续训练一定能追上。
原v5.2固定900原132、现环境复核125均高于新109，但source、输入与video映射不同，只有8/400行teacher配对一致；
这部分只作描述性整体参照，未重训旧方法来补齐因果比较。

600→900相邻validation R/G/L65/44/25、churn69、Jaccard .4851；train32/22/6、churn28、Jaccard .5333。
新−旧差值95%CI分别[+1,+8.75]pp与[+4.17,+29.17]pp，两组主面板仍有获取。
历史A同段validation为68/72/20、churn92；新模型较低churn来自新增更少（44对72），丢失反而更多（25对20），
保留率65/90也低于68/88，故不能据此宣称保持改善。Goal3、Long2两个验证task在两点均为零。
600→900训练段3160.0秒，累计训练程序时间9260.6秒；900点496条件、496条闭环和30个workers均通过。
基于持续获取而非FM下降，保持原配方、完整状态及六卡拓扑续到1200；尚无资格点，也不启动最终controls返工。

1200的validation相对900净降42，R/G/L52/15/57、churn72、Jaccard .4194，差值95%CI[-22.5,-.25]pp；
Object43→17、Long26→13、Spatial9→1，Goal31→36。相对A1200少68，差值95%CI[-29,-6.5]pp。
Train仍54/96，breadth18→20，相邻R/G/L43/11/11、churn22、Jaccard .6615；A同期升到62。
训练最近100更新FM均值.10595→.10417，训练任务保留样本的FM诊断.10351→.10197，
均未指向优化数值崩溃，却没有阻止未见任务闭环大幅回落。不能用loss继续下降给方法辩护，也不能直接称软件bug。
当前更明确的是跨任务迁移与成功保持失败；这些主面板仍没有识别唯一内部失败模块。
900→1200训练段2975.2秒，累计程序训练12235.8秒，4800条件/100800queries；所有checkpoint、496条件及496条评测通过。
按已登记节点再取1500确认回落是否持续，保持训练配置与全部恢复状态；这是一次相邻确认，不以FM下降无限延长窗口。
修正是否开展仍需具体失败证据、主要改变变量、预测与停止标准，最终controls不作为返工依据。

原件：`runs/analysis/unified_writer_20260917/unified/paired_readout.json`、`analysis/step{600,900,1200}_*`、
`training/{checkpoints,materialized,evaluation}`与`step600_execution.json`。600评测用五卡×三workers；
避开p2当时升高的其他任务负载，不改变训练的六卡拓扑或配对条件。一次物化设备参数格式错误在CLI解析时退出，
改为`cuda:N`后496条件fresh物化成功，原失败及完成记录均保留。
