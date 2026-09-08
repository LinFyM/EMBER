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
