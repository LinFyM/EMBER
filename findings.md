# EMBER findings

当前方法见[正式设计](docs/horizon_relation_video_writer_design.md)，当前执行计划与授权见[progress](progress.md)。
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
