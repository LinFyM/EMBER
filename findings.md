# 当前实施边界（2026-09-11）

Owner已授权Video Functional Writer设计、实现和实验。专家最终方案的机制理由是直接执行query功能信用；
这不是首次联合训练，也没有数学上排除共同动作捷径。J2真实联合训练已核验，辅助头失败不证明无信息，
辅助头强/LoRA弱不证明rank容量不足。具体设计见docs/video_functional_writer_design.md；首段实测见§56，尚无视频特异性资格。

# EMBER findings

Owner最新阶段为有益视频特异性优先，已授权新候选自主实施；旧路线保持暂停。
最新已完成证据见§51–56，远程副本与跨历史阅读入口见[全新专家材料](docs/review_materials/video_specificity_20260911/README.md)。
以下各节保留当时结论与边界，其旧“当前/下一步/目标”不覆盖最新Owner要求；本次不预设恢复v5.2或保留Horizon全图。

当前方法见[正式设计](docs/video_functional_writer_design.md)，当前执行计划与授权见[progress](progress.md)。
最新学习与原因分析集中在§39–49；[2026-09-11专家材料](docs/review_materials/20260911/README.md)提供远程可读原配置、逐条结果、机制记录及当前看法。文内本地runs路径通过该材料的index映射到已提交副本。
§50记录Owner对首轮专家意见的修正重点与历史正证据，不代表新实验或正式方法采纳。
以下§1–14记录此前各轮的持久发现，其中“新图/当前/下一轮”按当时路线解释，不恢复旧18层图或旧run；§15–16记录方法收口与接续裁决要求，§17记录实际新图的数值重放发现。

这里只保留会改变下一轮决策的结论与开放问题，不再复制逐轮实验年表。证据、数值和旧原文入口集中在
[research_history.md](docs/research_history.md)；已对齐候选的完整推导在
[horizon_relation_video_writer_design.md](docs/horizon_relation_video_writer_design.md)。当前授权和现场只看 [progress.md](progress.md)。

## 1. 先分清三个问题

- Task-local LoRA能否完成任务：已有强正证据，validation8 privileged rank16专家250/400。
- 视频到LoRA能否产生真实能力：早期v5.2/v6已有正证据，v6曾143/400。
- 共享Writer能否从正确视频稳定迁移、持续积累并满足因果性：仍未解决，不能由前两项推出。

[基线与早期证据](docs/research_history.md#baseline)。各种clone、oracle、free code只说明其实际接口和预算的能力；不能部署成task字典。

## 2. 原生信息要有实际消费者

当时分层图保留完整50-horizon、各层状态、native prefix和有效梯度；这些工程条件不自动构成视频过程理解，也不证明后继必须保留层轴。
早期把原生响应压缩后做泛化时序attention，或晚期让参数queries读取全部证据，都不足以证明动作序列先验已被有效利用。
新候选在H压缩前进行同层、双向局部帧对的跨horizon处理，并区分T/H/J；其效果仍需检验。

9月7日推导明确：对齐后内容差为零，不代表过程未推进；对应位置本身也可提供证据。相对位移分布rho只是A行的索引重排，
同一关系MLP应在逐帧聚合前消费内容与对应模式。固定probe造成的共同结构或漂亮斜带，都不足以证明物理动作对应。
这是一项建模依据；首轮短学习16步的训练侧闭环尚未见广泛改善，不能将其当作方法通过。离线双向读取不违反rollout前一次编译；有限上下文与视频因果必要性分别验证。

G2证明有序response包含功能动态；固定DP/event schema不是后继必须保留的形式。
[原生容量与动态](docs/research_history.md#native-capacity)。

## 3. X/Y、输出span与真实功能是三个层次

G1证明某些native X/Y signed pooling具有局部容量，也通过投影干预证明过窄Y span会丢失Goal/Long必要方向。
显式读取X/Y与强制因子在其span中，是两项独立选择。真实policy反传的gxᵀ本身提供原生参数坐标，Y=WX+b不等于应该施加的修正。
压缩E不是X/Y无损副本，观察Meta侧激活也不是执行场景激活。首版不加完整X/Y bank，保留对功能缺口有针对性的后续审视。

普通family head的固定末投影确实限制生成方向，但早期FactorHeads也有强行为；不能把这个几何事实直接称作新近低分根因。
坐标条件MLP是明确的候选解除方式，没有通用性能保证。

新图首轮std0.02 native坐标初始化下，16/48步训练侧闭环仅4/6与6/5（各40，source4），未形成广泛能力。
96步两个训练任务的三个代表target中B近乎native-channel常量，但真实policy梯度绝大部分不在该方向；
code RMS约1.1、坐标0.02。这支持检查坐标初始化的学习条件，不证明共同学习缺口已被唯一归因。
当前仅以标准正态native坐标、其它科学变量不变的fresh短对照检验；几何改善不能替代行为。

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

## 6. 最新共同学习缺口是事实，唯一根因仍未知

完整输出四任务短学习主要改善Goal，但Spatial有损失；这是移除carrier/解除span/完整rank/head变化的耦合收益，不能拆成单因果解释。
同预算target18相对mixed meta73恢复部分目标行为；两种初始化仍未建立广泛稳定能力。同两弱训练任务，clone14/20，对shared3或4/20，
而shared历程没有先达到强能力再遗忘。这使“只有未见task迁移困难”不足以解释现象。

容量、条件表示、优化和任务支持仍是竞争解释；gradient cosine或更低loss不能单独裁决。
[近期学习对照](docs/research_history.md#recent-learning)。width256仅确认训练结束，无新闭环分数，不为其补写好坏结论。

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

新Meta-on图不能复用跨step frozen R cache。可保留的是冻结prefix、同step临时R、policy VJP、Writer replay与observer chunk replay的
链式法则。清理后已有query-microbatch VJP和通用replay基础；新图R-leaf与Meta重放已真实验证，最长K4已profile；跨condition batch尚未实现。
只按实际最长K1/K4和真实queries测成本，不宣称设计图已经具有历史倍数加速。

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
